"""Enumerating a job board through its public API.

Path B's entry point. One adapter per board kind, all reduced to the same
return shape so the discover task never learns which board it is talking to.

**Why APIs rather than scraping**, restating spec section 9 item 3 because it
is the decision most likely to be quietly reversed later: an API gives stable
ids -- which become `external_id` and therefore `canonical_key` -- structured
fields, and no breakage when a designer ships a redesign. HTML parsing is the
fallback for `careers_page` sources, not the default.

**No two boards agree on anything**, and absorbing that is the whole job of
this module. Verified against live responses:

    board            change signal       inline content
    --------------   -----------------   --------------------------
    Greenhouse       `updated_at` (ISO)  `content`, HTML-escaped
    Lever            none                `descriptionPlain`
    Ashby            none                `descriptionPlain`
    Workable         none                `description`, HTML
    SmartRecruiters  `releasedDate`      none -- per-posting fetch
    BreezyHR         `published_date`    none -- per-posting fetch
    Rippling         none                none -- per-posting fetch
    USAJOBS          `PublicationStart`  `JobSummary` + duties

That table decides how much a re-crawl costs, and the second column is the
one to read first, because `_posting_needs_fetch` treats a missing change
signal as "must look". A board with neither column filled is re-fetched in
full on every crawl.

**Two classes of board, and the difference is an order of magnitude.** The
five with inline content are one HTTP request per crawl, total -- the listing
*is* the crawl. The three without are one request plus one per posting,
through the 1 rps per-domain bucket, and the arithmetic is worth stating
before adding another of them:

    Rippling `rippling`   738 postings, no change signal
                          -> ~12 minutes of fetching, every crawl

Rippling is the expensive case on both counts: its listing publishes no date
at all, so nothing can be skipped before fetching. SmartRecruiters and Breezy
at least date their postings, which lets a re-crawl skip the unchanged ones --
see `enumerate_smartrecruiters` for why that signal is weaker than
Greenhouse's and what it costs when it is wrong.

Two mitigations already exist and neither is in this module: `sources.
crawl_interval_seconds` is per-source, so a fetch-heavy board can be crawled
weekly rather than daily; and postings from one provider share a domain, so
the bucket serialises them regardless of how many boards of that kind are
enabled. Adding twenty SmartRecruiters companies does not cost twenty times
as much wall-clock -- it costs one queue, twenty times as long.

Greenhouse is the one that had to be argued into the inline column -- it needs
`?content=true`, which this module used to refuse. See `enumerate_greenhouse`
for the measurement that changed the answer, and for the escaping trap that
comes with it.

None of this costs closure detection, because closures are derived from what
is *absent from the enumeration*, not from what was fetched.
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


def _get_json(url: str, headers: dict[str, str] | None = None) -> object:
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
    # Caller headers merge over the defaults rather than replacing them, so an
    # authenticated board (USAJOBS) can add its key and override the User-Agent
    # its terms require without losing the Accept negotiation.
    request_headers = {
        "User-Agent": settings.fetch_user_agent,
        "Accept": "application/json",
    }
    request_headers.update(headers or {})

    try:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers=request_headers,
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


def enumerate_workable(board_token: str) -> list[DiscoveredPosting]:
    """Workable: the widget endpoint, with descriptions inline.

    `?details=true` is doing the same job `?content=true` does for Greenhouse,
    and without it this would be a per-posting-fetch board. Measured against
    `blueground`: 28 postings, descriptions averaging ~4.5 KB of HTML, one
    request. Worth checking that parameter survives any future edit here --
    dropping it does not break anything visibly, it just quietly turns a
    one-request crawl into a 28-request one.

    **The v1 widget path, not v3.** `apply.workable.com/api/v3/accounts/{token}
    /jobs` answers 404; the widget route is the one that serves an unauthenticated
    public board. v3 is the authenticated partner API and needs a key per account,
    which is the wrong shape for a catalog of boards we do not own.

    **A wrong token answers 200 with zero jobs.** Verified: `hotjar`, `typeform`
    and `oneflow` all resolve and return `jobs: []` because they have nothing
    open, while a genuinely bad token 404s -- but a *retired* account behaves
    like the empty one. So an empty list here is indistinguishable from a board
    that has simply stopped hiring, and it must not be treated as an error:
    `consecutive_failures` would never trip on it, and closure detection will
    correctly tombstone the postings that stopped appearing.

    `shortcode` is the stable id. `published_on` is a publication date rather
    than a modification one, so it is not used as a change signal -- the content
    hash catches edits, which is affordable precisely because the content is
    free.
    """
    payload = _get_json(
        f"https://apply.workable.com/api/v1/widget/accounts/{board_token}?details=true"
    )
    if not isinstance(payload, dict) or "jobs" not in payload:
        raise PermanentFetchError(f"unexpected Workable payload for {board_token!r}")

    postings = []
    for job in payload["jobs"]:
        if not job.get("shortcode") or not job.get("url"):
            continue
        raw = job.get("description")
        # HTML, not escaped HTML -- so a single strip, unlike Greenhouse. The
        # requirements and benefits blocks are separate fields and are appended,
        # for the same reason Lever's parts are joined: the requirements are the
        # half the scorer actually needs.
        parts = [
            part
            for part in (
                job.get("description"),
                job.get("requirements"),
                job.get("benefits"),
            )
            if part
        ]
        content = html_to_text("\n".join(parts)) if parts else None
        postings.append(
            DiscoveredPosting(
                external_id=str(job["shortcode"]),
                url=job["url"],
                title=job.get("title"),
                updated_at=None,
                content=content or None,
            )
        )
    return postings


# How many postings SmartRecruiters returns per page, and how many pages we
# will walk. 100 is their documented maximum; the page cap bounds a runaway
# without truncating any real board -- 50 pages is 5,000 postings, and the
# largest board in the catalog is an order of magnitude below that.
_SMARTRECRUITERS_PAGE = 100
_SMARTRECRUITERS_MAX_PAGES = 50


def enumerate_smartrecruiters(board_token: str) -> list[DiscoveredPosting]:
    """SmartRecruiters: paginated, dated, and with no description in the listing.

    The first board here that is genuinely paged. `totalFound` is authoritative
    and `offset`/`limit` walk it, so this loops rather than trusting one
    response -- the default page is 10, which would silently cap every board at
    ten postings and look like a catalog of very small companies.

    **`releasedDate` as the change signal, with a caveat worth recording.** It
    is when the posting was released, not when it was last edited, so a posting
    revised in place without being re-released keeps its old date and this will
    skip it. That is a weaker guarantee than Greenhouse's `updated_at`, and the
    alternative is worse: with no signal at all, every posting on every
    SmartRecruiters board is re-fetched on every crawl, and they all share the
    `jobs.smartrecruiters.com` bucket. Taking the imperfect date trades a rare
    stale posting for an hours-long daily crawl, and `crawl_interval_seconds`
    is the lever if that trade turns out wrong for a given board.

    **No content inline.** The listing carries `defaultJobAd` as a *boolean*,
    not an ad; the text lives at `/postings/{id}` under `jobAd.sections`. Left
    to the per-posting fetch rather than pulled in here, because doing N detail
    calls inside the adapter would hold the discovery worker for the whole
    board -- which is exactly the head-of-line problem the separate `discovery`
    queue exists to prevent.
    """
    postings: list[DiscoveredPosting] = []
    offset = 0

    for _ in range(_SMARTRECRUITERS_MAX_PAGES):
        payload = _get_json(
            f"https://api.smartrecruiters.com/v1/companies/{board_token}/postings"
            f"?limit={_SMARTRECRUITERS_PAGE}&offset={offset}"
        )
        if not isinstance(payload, dict) or "content" not in payload:
            raise PermanentFetchError(
                f"unexpected SmartRecruiters payload for {board_token!r}"
            )

        page = payload["content"]
        for job in page:
            if not job.get("id"):
                continue
            # `postingUrl` is absent from the list response, so the public URL
            # is composed. Documented and stable, and the id in it is the same
            # one `external_id` uses, which keeps `canonical_key` aligned with
            # what a per-posting fetch will resolve.
            url = job.get("postingUrl") or (
                f"https://jobs.smartrecruiters.com/{board_token}/{job['id']}"
            )
            postings.append(
                DiscoveredPosting(
                    external_id=str(job["id"]),
                    url=url,
                    title=job.get("name"),
                    updated_at=_iso(job.get("releasedDate")),
                    content=None,
                )
            )

        offset += len(page)
        # Both conditions matter. An empty page ends a board whose `totalFound`
        # over-reports (their count includes postings the public API will not
        # return), and the offset check ends the normal case without spending a
        # request to discover the list has run out.
        if not page or offset >= int(payload.get("totalFound") or 0):
            break

    return postings


def enumerate_rippling(board_token: str) -> list[DiscoveredPosting]:
    """Rippling: the whole board in one response, with nothing to skip on.

    The most expensive board kind here, and the reason the module docstring
    names it. Measured against `rippling` itself: 738 postings, no description
    and **no date of any kind** in the listing -- so `_posting_needs_fetch`
    falls through to "must look" for every one of them, every crawl, and they
    all share the `ats.rippling.com` bucket. That is ~12 minutes of fetching
    per crawl and it does not improve with familiarity.

    Worth pairing with a longer `crawl_interval_seconds` than the 86400 default
    when seeding one of these. The mitigation is a seeding decision rather than
    a code one, so it is recorded here rather than enforced: a board whose
    postings change weekly does not need looking at daily.

    The date lives on the detail endpoint (`createdOn`), which is no help --
    reaching it costs the request the signal exists to avoid.
    """
    payload = _get_json(
        f"https://api.rippling.com/platform/api/ats/v1/board/{board_token}/jobs"
    )
    # Rippling has answered with both shapes across versions, so both are
    # accepted rather than guessed at -- and unlike Lever, a bare list here is
    # the *unexpected* one, so it is handled rather than rejected.
    if isinstance(payload, dict):
        jobs = payload.get("items") or payload.get("jobs") or []
    elif isinstance(payload, list):
        jobs = payload
    else:
        raise PermanentFetchError(f"unexpected Rippling payload for {board_token!r}")

    postings = []
    for job in jobs:
        if not job.get("uuid") or not job.get("url"):
            continue
        postings.append(
            DiscoveredPosting(
                external_id=str(job["uuid"]),
                url=job["url"],
                title=job.get("name"),
                updated_at=None,
                content=None,
            )
        )
    return postings


def enumerate_breezy(board_token: str) -> list[DiscoveredPosting]:
    """BreezyHR: a bare list on a per-company subdomain, dated, no content.

    A list rather than an object, like Lever -- and the same trap applies, since
    reading it as a dict yields nothing rather than raising.

    **`/json/{id}` is not a detail endpoint.** It answers 200 with HTML, so
    there is no structured way to reach the description and the per-posting
    fetch goes at the public page like any other. Checked rather than assumed,
    because the listing path being `/json` makes a matching detail route look
    obvious.

    One genuine advantage over the other fetch-heavy boards: postings live on
    `{token}.breezy.hr`, a subdomain per company, so two Breezy boards do not
    contend for the same token bucket the way two SmartRecruiters boards do.
    """
    payload = _get_json(f"https://{board_token}.breezy.hr/json")
    if not isinstance(payload, list):
        raise PermanentFetchError(f"unexpected BreezyHR payload for {board_token!r}")

    postings = []
    for job in payload:
        if not job.get("id") or not job.get("url"):
            continue
        postings.append(
            DiscoveredPosting(
                external_id=str(job["id"]),
                url=job["url"],
                title=job.get("name"),
                updated_at=_iso(job.get("published_date")),
                content=None,
            )
        )
    return postings


def _flatten(value: object) -> str:
    """Render a USAJOBS detail field, which may be a string or a list of them.

    `MajorDuties` is an array and `JobSummary` is a string, in the same object.
    Interpolating the array with `str()` would put a Python list repr --
    brackets, quotes, commas -- into the text every skill and keyword is then
    matched against, which is the same class of corruption as Greenhouse's
    double-escaped HTML and just as invisible downstream.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_flatten(item) for item in value if item)
    return str(value)


# USAJOBS caps a page at 500 and paginates from 1, not 0. The page cap bounds
# the largest agency: Veterans Affairs runs several thousand openings, and 40
# pages is 20,000.
_USAJOBS_PAGE = 500
_USAJOBS_MAX_PAGES = 40


def enumerate_usajobs(board_token: str) -> list[DiscoveredPosting]:
    """USAJOBS: one federal agency per source, descriptions inline.

    **Modelled per-agency, and that is the load-bearing decision.** USAJOBS is
    a search API over the whole federal corpus, not a board -- the obvious way
    to add it is one source issuing keyword queries, and that breaks two things
    this system relies on. There would be no `board_token` for a source row to
    hold, and `sources.display_name` would read "USAJOBS" rather than an
    employer, so `_company_for_source` would stamp that on every posting and
    undo the company backfill for all of them.

    Scoping each source to one agency via `Organization` keeps the existing
    shape exactly: `board_token` is the agency code, `display_name` is the
    agency name, and the employer on every posting is correct for the same
    reason it is correct for Greenhouse. A per-agency crawl cadence comes free
    with it.

    **`Authorization-Key` is required and there is no anonymous mode** -- the
    endpoint answers 401 without it. The key is free and instant from
    developer.usajobs.gov, and their terms ask that the User-Agent be the email
    the key was registered to, which is why that is a separate setting rather
    than reusing `fetch_user_agent`.

    Raising rather than returning empty when unconfigured, because an
    unconfigured source that enumerates nothing looks exactly like an agency
    with no openings -- and closure detection would then tombstone every
    posting the agency has, on a crawl that "succeeded".

    **Not verified against a live response.** Every other adapter in this module
    was written against a recorded payload; this one was written against the
    documented shape, because the key is per-developer and none was available.
    The field access is defensive throughout for that reason, and the first live
    crawl is the real test.
    """
    if not settings.usajobs_api_key or not settings.usajobs_email:
        raise PermanentFetchError(
            "usajobs source requires USAJOBS_API_KEY and USAJOBS_EMAIL "
            "(free key from developer.usajobs.gov)"
        )

    headers = {
        "Host": "data.usajobs.gov",
        "User-Agent": settings.usajobs_email,
        "Authorization-Key": settings.usajobs_api_key,
    }

    postings: list[DiscoveredPosting] = []

    for page in range(1, _USAJOBS_MAX_PAGES + 1):
        payload = _get_json(
            "https://data.usajobs.gov/api/search"
            f"?Organization={board_token}&ResultsPerPage={_USAJOBS_PAGE}&Page={page}",
            headers=headers,
        )
        if not isinstance(payload, dict) or "SearchResult" not in payload:
            raise PermanentFetchError(
                f"unexpected USAJOBS payload for {board_token!r}"
            )

        result = payload["SearchResult"] or {}
        items = result.get("SearchResultItems") or []
        for item in items:
            job = (item or {}).get("MatchedObjectDescriptor") or {}
            external_id = (item or {}).get("MatchedObjectId") or job.get("PositionID")
            if not external_id or not job.get("PositionURI"):
                continue

            # The description is spread across several fields under UserArea,
            # and joining them is what reconstructs the posting -- the same
            # shape as Lever's parts. JobSummary alone is a paragraph of
            # preamble with none of the requirements the scorer needs.
            details = (job.get("UserArea") or {}).get("Details") or {}
            parts = [
                text
                for text in (
                    _flatten(details.get("JobSummary")),
                    _flatten(details.get("MajorDuties")),
                    _flatten(details.get("Requirements")),
                    _flatten(details.get("Qualifications")),
                    _flatten(details.get("Evaluations")),
                )
                if text
            ]
            content = html_to_text("\n\n".join(parts)) if parts else None

            postings.append(
                DiscoveredPosting(
                    external_id=str(external_id),
                    url=job["PositionURI"],
                    title=job.get("PositionTitle"),
                    updated_at=_iso(job.get("PublicationStartDate")),
                    content=content or None,
                )
            )

        # `SearchResultCountAll` is the corpus-wide total for the query and
        # `SearchResultCount` is this page, so the page being short is the end
        # rather than the count matching anything.
        if len(items) < _USAJOBS_PAGE:
            break

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
    "workable": enumerate_workable,
    "smartrecruiters": enumerate_smartrecruiters,
    "rippling": enumerate_rippling,
    "breezy": enumerate_breezy,
    "usajobs": enumerate_usajobs,
}


def enumerate_source(kind: str, board_token: str) -> list[DiscoveredPosting]:
    """Enumerate one board. Raises for a kind with no adapter."""
    adapter = ADAPTERS.get(kind)
    if adapter is None:
        raise PermanentFetchError(f"no adapter for source kind {kind!r}")
    return adapter(board_token)
