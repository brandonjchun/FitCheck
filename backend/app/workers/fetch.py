"""Fetching a job posting: politely, with bounded cost, classifying failures.

Three concerns, in the order they bite.

**Politeness.** robots.txt is honoured and cached per host, the User-Agent
identifies the bot with a contact route, and every request passes a
cross-process rate limiter first. Path A makes one request when a human
clicks a button; a batch upload makes hundreds unattended, so none of this is
optional once M4's fan-out exists.

**Bounded cost.** Separate connect and read timeouts, a byte cap enforced
*while streaming* rather than from a header a server can omit or lie about,
and a redirect limit. Without these one pathological URL occupies a worker
for the full job timeout.

**Classification.** Every failure is sorted into "retry this" or "never retry
this" before it leaves this module. A 404 retried three times wastes two
minutes and a worker slot on something that cannot succeed, and the caller
should not have to know that an httpx.ReadTimeout is worth another attempt
while a 410 is not.
"""

import logging
import urllib.robotparser
from urllib.parse import urlsplit, urlunsplit

import httpx
from selectolax.parser import HTMLParser

from app.config import settings
from app.netguard import BlockedAddressError, UnresolvableHostError, assert_public_url
from app.queues import get_redis
from app.ratelimit import RateLimiter, domain_of

logger = logging.getLogger(__name__)

_ROBOTS_CACHE_PREFIX = "robots:"

# Statuses that carry a Location worth following. Listed explicitly rather
# than using httpx's `is_redirect`, because a 302 with no Location must be an
# error here and not silently fall through to the body-reading path.
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

# Status codes worth another attempt. 429 and 5xx are the host saying "not
# now" rather than "no"; 408 is an explicit request timeout.
_TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 507, 509})

# Content types we can extract text from. Anything else -- a PDF job
# description, an image, a zip -- is a permanent failure for this pipeline
# rather than something to retry.
_TEXT_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")

# Elements whose text is never part of a job posting. Removed before
# extraction because selectolax's text() would otherwise return minified
# JavaScript and CSS rules alongside the prose, which wrecks both the
# extraction prompt and the content hash.
_NOISE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "nav",
    "footer",
    "header",
)


class FetchError(Exception):
    """Base for every failure originating from fetching a URL."""


class TransientFetchError(FetchError):
    """Worth retrying: timeout, connection reset, 429, 5xx.

    The distinction this carries is the whole reason the class exists -- the
    worker decides between backoff and dead-letter purely from the type.
    """


class PermanentFetchError(FetchError):
    """Never worth retrying: 404, 410, robots disallow, wrong content type.

    Retrying a 404 three times spends two minutes and a worker slot learning
    what the first attempt already established.
    """


class RateLimitedError(TransientFetchError):
    """Could not get a rate-limit token within the budget.

    Transient by inheritance, and genuinely so: the work is legitimate and
    merely early. Releasing the worker and letting backoff reschedule is
    ordinary backpressure -- a worker parked on a full bucket is a worker
    doing nothing, and under a 500-URL batch against one host that would
    stall every other queue too.
    """


def _robots_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))


def _load_robots(url: str, client: httpx.Client) -> urllib.robotparser.RobotFileParser:
    """Fetch and parse robots.txt for this URL's host, cached in Redis.

    Cached across processes, not just within one, because the cost being
    avoided is a request to someone else's server. A per-process cache would
    still mean N fetches of robots.txt for N workers on the same board.

    A robots.txt that cannot be fetched is treated as permissive. That is the
    conventional reading -- absence of a policy is not a prohibition -- and
    the alternative would make one 500 on a robots file block an entire
    board indefinitely.
    """
    host = domain_of(url)
    cache_key = f"{_ROBOTS_CACHE_PREFIX}{host}"
    redis = get_redis()

    cached = redis.get(cache_key)
    if cached is not None:
        body = cached.decode("utf-8", errors="replace")
    else:
        try:
            response = client.get(_robots_url(url))
            body = response.text if response.status_code == 200 else ""
        except httpx.HTTPError as exc:
            logger.info("robots: could not fetch for %s (%s); treating as allow", host, exc)
            body = ""

        redis.set(cache_key, body.encode("utf-8"), ex=settings.robots_cache_seconds)

    parser = urllib.robotparser.RobotFileParser()
    parser.parse(body.splitlines())
    return parser


def is_allowed(url: str, client: httpx.Client) -> bool:
    """Whether robots.txt permits us to fetch `url`."""
    try:
        return _load_robots(url, client).can_fetch(settings.fetch_user_agent, url)
    except Exception as exc:
        # A malformed robots.txt should not take down ingestion. Log and
        # allow, matching the treatment of an unfetchable one.
        logger.warning("robots: parse failed for %s (%s); treating as allow", url, exc)
        return True


def html_to_text(html: str) -> str:
    """Reduce an HTML document to the text a human would read.

    selectolax rather than BeautifulSoup: this runs on every posting a crawl
    touches, and the parser is roughly an order of magnitude faster on the
    same input.

    Noise elements are stripped first. Without that, `text()` returns the
    contents of every <script> tag, which both derails the extraction prompt
    and makes the content hash change whenever a site's analytics bundle is
    rebuilt -- defeating the M8 gate that exists to avoid re-extracting
    unchanged postings.
    """
    tree = HTMLParser(html)

    for selector in _NOISE_SELECTORS:
        for node in tree.css(selector):
            node.decompose()

    if tree.body is None:
        return ""

    text = tree.body.text(separator="\n", strip=True)

    # Collapse runs of blank lines. HTML layout produces a lot of them, and
    # they are pure token cost in the extraction prompt.
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _classify_status(status: int, url: str) -> FetchError:
    if status in _TRANSIENT_STATUSES:
        return TransientFetchError(f"HTTP {status} fetching {url}")
    return PermanentFetchError(f"HTTP {status} fetching {url}")


def _assert_fetchable(url: str) -> None:
    """Translate a netguard verdict into this module's error vocabulary.

    The two cases are caught in this order because `UnresolvableHostError`
    subclasses `BlockedAddressError`, and they get opposite treatment: a
    blocked address is permanent by construction, while a resolver that did
    not answer is the ordinary transient case.
    """
    try:
        assert_public_url(url)
    except UnresolvableHostError as exc:
        raise TransientFetchError(str(exc)) from exc
    except BlockedAddressError as exc:
        raise PermanentFetchError(str(exc)) from exc


def _get_following_redirects(client: httpx.Client, url: str) -> tuple[bytes, str]:
    """GET `url`, checking every redirect hop before following it.

    Returns the raw body and the encoding to decode it with.

    Redirects are walked here rather than by httpx because `follow_redirects`
    would put every hop after the first outside the SSRF guard entirely. A
    public host answering `302 Location: http://169.254.169.254/` is the
    standard one-line bypass of a check that only ran on the submitted URL,
    and it costs the attacker nothing.
    """
    current = url

    for _ in range(settings.fetch_max_redirects + 1):
        try:
            # Streamed, so the byte cap can abort mid-response. A
            # non-streaming get() has already read the whole body into memory
            # by the time any size check could run, which makes the cap
            # decorative.
            with client.stream("GET", current) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise PermanentFetchError(
                            f"HTTP {response.status_code} with no Location at {current}"
                        )
                    # Joined against the current URL because a Location is
                    # allowed to be relative.
                    current = str(response.url.join(location))
                    _assert_fetchable(current)
                    continue

                if response.status_code >= 400:
                    raise _classify_status(response.status_code, current)

                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                )
                if content_type and not content_type.startswith(_TEXT_CONTENT_TYPES):
                    raise PermanentFetchError(
                        f"unsupported content type {content_type!r} at {current}"
                    )

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > settings.fetch_max_bytes:
                        raise PermanentFetchError(
                            f"response exceeded {settings.fetch_max_bytes} bytes at {current}"
                        )
                    chunks.append(chunk)

                return b"".join(chunks), response.encoding or "utf-8"

        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientFetchError(
                f"{type(exc).__name__} fetching {current}: {exc}"
            ) from exc

    # Permanent: a redirect loop is a property of the site, and the next
    # attempt walks the identical loop.
    raise PermanentFetchError(f"too many redirects for {url}")


def _decode(body: bytes, encoding: str) -> str:
    """Decode `body`, falling back to UTF-8 when the declared charset is unknown.

    `errors="replace"` covers bad *bytes*. It does nothing for a charset no
    codec implements: a server is free to answer `charset=utf8mb4`, or a typo,
    and `bytes.decode` raises LookupError before any byte is examined.

    That exception is why this function exists rather than an inline decode.
    LookupError is neither TransientFetchError nor PermanentFetchError, so it
    escapes the classification this module exists to provide, reaches the
    worker's `except Exception`, and is re-raised -- which means RQ applies the
    retry policy and spends three attempts re-learning that the charset is
    still not a codec. Exactly the waste the transient/permanent split was
    built to prevent, arriving through the one door it does not cover.

    Falling back rather than failing, because a declared-but-unknown charset is
    a defect in a header rather than a document that cannot be read. The bytes
    are nearly always UTF-8 or close enough that replacement characters cost a
    few glyphs; discarding a recoverable posting over a bad header would be the
    more expensive direction of the same mistake.
    """
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        logger.info(
            "fetch: server declared unknown charset %r; decoding as utf-8", encoding
        )
        return body.decode("utf-8", errors="replace")


def fetch_posting_text(url: str, limiter: RateLimiter | None = None) -> str:
    """Fetch `url` and return its readable text.

    Raises:
        RateLimitedError: no rate-limit token available within the budget.
        PermanentFetchError: robots disallow, 4xx, or a non-text response.
        TransientFetchError: timeout, connection failure, 429, or 5xx.
    """
    limiter = limiter if limiter is not None else RateLimiter()
    host = domain_of(url)

    timeout = httpx.Timeout(
        connect=settings.fetch_connect_timeout,
        read=settings.fetch_read_timeout,
        write=settings.fetch_read_timeout,
        pool=settings.fetch_connect_timeout,
    )

    with httpx.Client(
        timeout=timeout,
        # Followed by hand in _get_following_redirects so that each hop passes
        # the SSRF guard. See that function.
        follow_redirects=False,
        headers={
            "User-Agent": settings.fetch_user_agent,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        },
    ) as client:
        # First, and before robots specifically: _load_robots fetches
        # /robots.txt from this same host, so a guard placed after it has
        # already made the request it exists to prevent.
        _assert_fetchable(url)

        # robots before the rate limiter, so a disallowed URL does not spend a
        # token it was never going to use.
        if not is_allowed(url, client):
            raise PermanentFetchError(f"robots.txt disallows fetching {url}")

        if not limiter.acquire(host):
            raise RateLimitedError(f"rate limit budget exhausted for {host}")

        body, encoding = _get_following_redirects(client, url)

    html = _decode(body, encoding)
    return html_to_text(html)
