"""Enumerating a job board through its public API.

Path B's entry point. One adapter per board kind, all reduced to the same
return shape so the discover task never learns which board it is talking to.

**Why APIs rather than scraping**, restating spec section 9 item 3 because it
is the decision most likely to be quietly reversed later: an API gives stable
ids -- which become `external_id` and therefore `canonical_key` -- structured
fields, and no breakage when a designer ships a redesign. HTML parsing is the
fallback for `careers_page` sources, not the default.

**The three boards do not agree on anything**, and absorbing that is the whole
job of this module. Verified against live responses:

    board       change signal        inline content
    ---------   ------------------   -------------------------
    Greenhouse  `updated_at` (ISO)   `content`, HTML-escaped
    Lever       none                 `descriptionPlain`
    Ashby       none                 `descriptionPlain`

That table decides how much work a re-crawl costs, and since all three now
carry content inline, **a crawl of any board here is one HTTP request,
total.** There is never a per-posting fetch on the happy path, which makes
every board faster and politer than anything we could manage by being clever
about which pages to request.

Greenhouse is the one that had to be argued into that column -- it needs
`?content=true`, which this module used to refuse. See `enumerate_greenhouse`
for the measurement that changed the answer, and for the escaping trap that
comes with it.

Greenhouse also uniquely publishes a change signal, which is still worth
having: `updated_at` lets the ingest side skip an unchanged posting *before*
hashing it, where the content hash can only skip after.

Neither shortcut costs closure detection, because closures are derived from
what is *absent from the enumeration*, not from what was fetched.
"""

import html
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

import httpx

from app.config import settings
from app.workers.fetch import (
    PermanentFetchError,
    TransientFetchError,
    _classify_status,
    html_to_text,
)

logger = logging.getLogger(__name__)

# A board listing is one request returning every posting, so it is far larger
# than any single page and the per-page cap does not apply. Greenhouse with
# descriptions is ~5.8 MB; without them a 550-posting board is a few hundred
# KB. This bounds the pathological case without rejecting the normal one.
MAX_LISTING_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class DiscoveredPosting:
    """One posting as the board describes it, before we fetch anything."""

    external_id: str
    url: str
    title: str | None = None

    # The board's own statement of when this last changed, when it offers
    # one. This is what lets a re-crawl skip a posting *without fetching it*
    # -- the content hash cannot do that, because computing it requires
    # already having the content.
    updated_at: datetime | None = None

    # Full text, when the listing includes it. Present for Lever and Ashby,
    # absent for Greenhouse. When set, the ingest step needs no HTTP request
    # at all and goes straight to hashing.
    content: str | None = None


def _get_json(url: str) -> object:
    """Fetch and parse a board listing.

    Deliberately not `fetch_posting_text`: that path is for arbitrary
    user-supplied URLs and carries robots checks, the per-domain token
    bucket, and a 5 MB cap sized for one HTML page. A board's own documented
    JSON API is a different thing -- we are the intended consumer, it is one
    request rather than hundreds, and the response is legitimately larger
    than any page.

    Failures are classified into the same transient/permanent split the rest
    of the system speaks, so the discover task can decide between backoff and
    tripping the circuit breaker without knowing what httpx is.
    """
    timeout = httpx.Timeout(
        connect=settings.fetch_connect_timeout,
        read=settings.fetch_read_timeout * 2,  # a full board is a big response
        write=settings.fetch_read_timeout,
        pool=settings.fetch_connect_timeout,
    )
    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": settings.fetch_user_agent, "Accept": "application/json"},
        ) as client:
            with client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise _classify_status(response.status_code, url)

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > MAX_LISTING_BYTES:
                        raise PermanentFetchError(
                            f"board listing exceeded {MAX_LISTING_BYTES} bytes at {url}"
                        )
                    chunks.append(chunk)
                body = b"".join(chunks)
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        raise TransientFetchError(f"{type(exc).__name__} enumerating {url}: {exc}") from exc

    import json

    try:
        return json.loads(body)
    except ValueError as exc:
        # Permanent: a board answering 200 with non-JSON has changed its API
        # or is serving an error page, and neither improves on retry.
        raise PermanentFetchError(f"board listing at {url} was not JSON: {exc}") from exc


def _iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Normalized to UTC so comparisons against a timestamptz column are not
    # quietly comparing an aware value to a naive one.
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def enumerate_greenhouse(board_token: str) -> list[DiscoveredPosting]:
    """Greenhouse: ids, URLs, titles, `updated_at`, and inline descriptions.

    `boards-api.greenhouse.io` rather than `boards.greenhouse.io/embed/...`,
    and that is a compliance decision rather than a preference: the embed
    path is `Disallow`ed in Greenhouse's robots.txt, verified against the
    live file. The API host is not.

    **`?content=true`, which this used to refuse.** The original reasoning was
    that `updated_at` already narrows a re-crawl to what changed, so paying
    ~5.8 MB daily to learn that nothing did was the worse trade. That
    optimizes the steady state and ignores the cold one -- and the cold state
    is exactly what adding a board means. Measured against the live API:

        anthropic   394 jobs   5.7 MB   0.3s
        stripe      546 jobs   4.1 MB   0.2s

    Against the alternative, which is one fetch per posting through a 1 rps
    per-domain bucket that every Greenhouse board shares: Stripe's first
    crawl is nine minutes of rate-limited fetching, or one request taking two
    tenths of a second. `updated_at` is still parsed and still useful -- the
    ingest side uses it to skip unchanged postings before hashing -- but it is
    no longer the only thing standing between a new board and a backlog.

    Note the double-decode. Greenhouse returns the description **HTML-escaped**,
    so `content` is a string of `&lt;div&gt;...`. Running the HTML stripper on
    that once yields the markup as literal text -- tags and all -- and every
    downstream keyword, skill, and hash then matches against angle brackets
    instead of prose. It has to be unescaped first, then stripped.
    """
    payload = _get_json(
        f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true"
    )
    if not isinstance(payload, dict) or "jobs" not in payload:
        raise PermanentFetchError(f"unexpected Greenhouse payload for {board_token!r}")

    postings = []
    for job in payload["jobs"]:
        if not job.get("id") or not job.get("absolute_url"):
            continue
        raw = job.get("content")
        # unescape before html_to_text -- see the docstring. Guarded rather
        # than assumed: a posting with an empty body leaves `content` None and
        # falls back to the per-posting fetch, which is the old behaviour and
        # still correct for that one row.
        content = html_to_text(html.unescape(raw)) if raw else None
        postings.append(
            DiscoveredPosting(
                external_id=str(job["id"]),
                url=job["absolute_url"],
                title=job.get("title"),
                updated_at=_iso(job.get("updated_at")),
                content=content or None,
            )
        )
    return postings


def enumerate_lever(board_token: str) -> list[DiscoveredPosting]:
    """Lever: a bare JSON array, with the full description inline.

    Note the response is a list, not an object -- unlike both other boards.
    Getting this wrong reads as "the board returned nothing" rather than as a
    parse error, which is the kind of bug that looks like an empty company.

    `descriptionPlain` means a Lever crawl never fetches a posting page. The
    listing is the crawl.
    """
    payload = _get_json(f"https://api.lever.co/v0/postings/{board_token}?mode=json")
    if not isinstance(payload, list):
        raise PermanentFetchError(f"unexpected Lever payload for {board_token!r}")

    postings = []
    for job in payload:
        if not job.get("id") or not job.get("hostedUrl"):
            continue
        # Lever assembles a posting from several plain-text parts; joining
        # them is what reconstructs the document a human reads. Using
        # `descriptionPlain` alone drops the requirements lists, which is
        # exactly the half the scorer needs.
        body = "\n\n".join(
            part
            for part in (
                job.get("text"),
                job.get("descriptionPlain"),
                job.get("additionalPlain"),
            )
            if part
        )
        postings.append(
            DiscoveredPosting(
                external_id=str(job["id"]),
                url=job["hostedUrl"],
                title=job.get("text"),
                # Lever exposes createdAt (epoch ms) and no updatedAt, so
                # there is no change signal to use. The content hash is what
                # catches edits here instead -- which works precisely because
                # the content arrives free.
                updated_at=None,
                content=body or None,
            )
        )
    return postings


def enumerate_ashby(board_token: str) -> list[DiscoveredPosting]:
    """Ashby: an object with `jobs`, descriptions inline, no change signal.

    Only listed postings are returned to the crawler. `isListed` false means
    the company has taken it off their public board, so treating it as
    present would keep a withdrawn role in the feed.
    """
    payload = _get_json(
        f"https://api.ashbyhq.com/posting-api/job-board/{board_token}"
    )
    if not isinstance(payload, dict) or "jobs" not in payload:
        raise PermanentFetchError(f"unexpected Ashby payload for {board_token!r}")

    postings = []
    for job in payload["jobs"]:
        if not job.get("id") or not job.get("jobUrl") or job.get("isListed") is False:
            continue
        content = job.get("descriptionPlain")
        if not content and job.get("descriptionHtml"):
            content = html_to_text(job["descriptionHtml"])
        postings.append(
            DiscoveredPosting(
                external_id=str(job["id"]),
                url=job["jobUrl"],
                title=job.get("title"),
                updated_at=None,
                content=content or None,
            )
        )
    return postings


# Dispatch table rather than a chain of ifs, so adding a board is one entry
# and one function. `careers_page` is deliberately absent: HTML scraping of
# arbitrary company sites is the fallback spec section 9 item 3 warns against
# defaulting to, and a source of that kind will raise below rather than
# silently enumerate nothing.
ADAPTERS: dict[str, Callable[[str], list[DiscoveredPosting]]] = {
    "greenhouse": enumerate_greenhouse,
    "lever": enumerate_lever,
    "ashby": enumerate_ashby,
}


def enumerate_source(kind: str, board_token: str) -> list[DiscoveredPosting]:
    """Enumerate one board. Raises for a kind with no adapter."""
    adapter = ADAPTERS.get(kind)
    if adapter is None:
        raise PermanentFetchError(f"no adapter for source kind {kind!r}")
    return adapter(board_token)
