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
    ---------   ------------------   --------------
    Greenhouse  `updated_at` (ISO)   no
    Lever       none                 `descriptionPlain`
    Ashby       none                 `descriptionPlain`

That table decides how much work a re-crawl costs, and it decides it
differently per board:

- **Lever and Ashby hand back the full description in the listing**, so a
  crawl of either is *one HTTP request, total*. There is never a per-posting
  fetch, which makes them both faster and politer than anything we could do
  by being clever.
- **Greenhouse does not**, but it does say when each posting last changed. So
  a re-crawl fetches only postings that are new or whose `updated_at` moved
  -- which on a stable board is a handful rather than four hundred.

Neither shortcut costs closure detection, because closures are derived from
what is *absent from the enumeration*, not from what was fetched.
"""

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
    """Greenhouse: ids, URLs, titles, and `updated_at`. No descriptions.

    `boards-api.greenhouse.io` rather than `boards.greenhouse.io/embed/...`,
    and that is a compliance decision rather than a preference: the embed
    path is `Disallow`ed in Greenhouse's robots.txt, verified against the
    live file. The API host is not.

    `?content=true` would return descriptions inline and make this one
    request like the others -- measured at 5.8 MB for 399 postings. It is not
    used because `updated_at` already reduces a re-crawl to only what
    changed, which is cheaper than moving 5.8 MB every day to discover that
    nothing did.
    """
    payload = _get_json(
        f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
    )
    if not isinstance(payload, dict) or "jobs" not in payload:
        raise PermanentFetchError(f"unexpected Greenhouse payload for {board_token!r}")

    return [
        DiscoveredPosting(
            external_id=str(job["id"]),
            url=job["absolute_url"],
            title=job.get("title"),
            updated_at=_iso(job.get("updated_at")),
        )
        for job in payload["jobs"]
        if job.get("id") and job.get("absolute_url")
    ]


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
