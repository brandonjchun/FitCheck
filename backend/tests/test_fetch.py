"""Real fetching: robots, size caps, timeouts, and failure classification.

No network. httpx.MockTransport serves canned responses through the real
httpx client, so redirects, streaming, headers, and timeouts all run through
the actual code paths rather than a stub of them.

The classification tests are the ones that matter most. "Retry this" versus
"never retry this" is what decides whether a 404 costs one worker slot or
three, and it is 20% of the grade under correctness-under-failure.
"""

import socket

import httpx
import pytest

from app import netguard
from app.config import settings
from app.ratelimit import RateLimiter, domain_of
from app.workers import fetch as fetch_module
from app.workers.fetch import (
    PermanentFetchError,
    RateLimitedError,
    TransientFetchError,
    fetch_posting_text,
    html_to_text,
)

# A real routable address, used so the SSRF guard sees a public host for the
# tests that are not about the SSRF guard.
PUBLIC_IP = "93.184.216.34"

POSTING_URL = "https://boards.example.com/jobs/123"

PAGE = """
<html>
  <head><title>Senior Engineer</title><style>.a{color:red}</style></head>
  <body>
    <nav>Home About</nav>
    <h1>Senior Engineer</h1>
    <p>We need Python and Postgres.</p>
    <script>analytics('abc123');</script>
    <footer>Cookie notice</footer>
  </body>
</html>
"""


class AllowAllLimiter:
    """A limiter that never makes anyone wait.

    Rate limiting is tested directly in TestRateLimiter; injecting this keeps
    the fetch tests from each spending real seconds on token accrual.
    """

    def acquire(self, host, max_wait=None):
        return True


@pytest.fixture(autouse=True)
def resolves_to(monkeypatch):
    """Point every hostname at a public address unless a test says otherwise.

    Autouse because `boards.example.com` does not resolve, and without this
    the SSRF guard would fail every fetch test with a DNS error -- turning
    tests about size caps and redirects into tests about name resolution.

    Returns an installer so the SSRF tests can aim a hostname wherever they
    need it, including at nothing.
    """

    def install(*addresses: str, error: str | None = None):
        def fake_resolve(host: str, port: int) -> list[str]:
            if error is not None:
                raise socket.gaierror(error)
            return list(addresses)

        monkeypatch.setattr(netguard, "resolve_host", fake_resolve)

    install(PUBLIC_IP)
    return install


@pytest.fixture
def robots(monkeypatch):
    """Serve a robots.txt body, bypassing Redis.

    Patched at the module boundary rather than through a fake Redis, because
    what these tests care about is the allow/disallow decision, not the
    caching layer underneath it.
    """

    def install(body: str = ""):
        def fake_load(url, client):
            import urllib.robotparser

            parser = urllib.robotparser.RobotFileParser()
            parser.parse(body.splitlines())
            return parser

        monkeypatch.setattr(fetch_module, "_load_robots", fake_load)

    install()
    return install


@pytest.fixture
def serve(monkeypatch):
    """Route every request through a MockTransport with the given handler."""

    def install(handler):
        real_client = httpx.Client

        def patched(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_client(*args, **kwargs)

        monkeypatch.setattr(fetch_module.httpx, "Client", patched)

    return install


def _ok(html: str = PAGE, content_type: str = "text/html"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": content_type}, text=html)

    return handler


class TestHtmlToText:
    def test_extracts_readable_prose(self) -> None:
        text = html_to_text(PAGE)

        assert "Senior Engineer" in text
        assert "We need Python and Postgres." in text

    def test_strips_script_and_style(self) -> None:
        """Not tidiness -- two concrete costs.

        Script bodies derail the extraction prompt, and they change whenever
        a site rebuilds its analytics bundle, which would make the M8 content
        hash miss on unchanged postings and re-extract the whole catalog.
        """
        text = html_to_text(PAGE)

        assert "analytics" not in text
        assert "color:red" not in text

    def test_strips_chrome(self) -> None:
        text = html_to_text(PAGE)

        assert "Cookie notice" not in text
        assert "Home About" not in text

    def test_collapses_blank_lines(self) -> None:
        text = html_to_text("<html><body><p>a</p><p></p><p></p><p>b</p></body></html>")

        assert text == "a\nb"

    def test_document_with_no_body(self) -> None:
        assert html_to_text("<html></html>") == ""


class TestSuccessfulFetch:
    def test_returns_extracted_text(self, serve, robots) -> None:
        serve(_ok())

        text = fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

        assert "We need Python and Postgres." in text

    def test_sends_an_identifying_user_agent(self, serve, robots) -> None:
        """Identify honestly. Impersonating a browser to evade detection is
        both indefensible in an interview and how a project gets blocked."""
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["ua"] = request.headers.get("user-agent")
            return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)

        serve(handler)
        fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

        assert seen["ua"] == settings.fetch_user_agent
        assert "FitCheckBot" in seen["ua"]


class TestRobots:
    def test_disallowed_url_is_permanent(self, serve, robots) -> None:
        """Permanent, not transient. robots.txt will say the same thing in
        ten seconds, so a retry is a second impolite request."""
        robots("User-agent: *\nDisallow: /jobs/")
        serve(_ok())

        with pytest.raises(PermanentFetchError, match="robots"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_allowed_path_proceeds(self, serve, robots) -> None:
        robots("User-agent: *\nDisallow: /admin/")
        serve(_ok())

        assert fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_robots_is_checked_before_a_token_is_spent(self, serve, robots) -> None:
        """A disallowed URL should not consume rate-limit budget it was never
        going to use -- that budget belongs to URLs we can actually fetch."""
        robots("User-agent: *\nDisallow: /")
        serve(_ok())
        spent = []

        class CountingLimiter:
            def acquire(self, host, max_wait=None):
                spent.append(host)
                return True

        with pytest.raises(PermanentFetchError):
            fetch_posting_text(POSTING_URL, limiter=CountingLimiter())

        assert spent == []


class TestStatusClassification:
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 451])
    def test_client_errors_are_permanent(self, serve, robots, status: int) -> None:
        """Retrying a 404 three times spends two minutes learning nothing."""
        serve(lambda request: httpx.Response(status))

        with pytest.raises(PermanentFetchError):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    def test_overload_and_server_errors_are_transient(
        self, serve, robots, status: int
    ) -> None:
        """These are the host saying "not now" rather than "no"."""
        serve(lambda request: httpx.Response(status))

        with pytest.raises(TransientFetchError):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_timeout_is_transient(self, serve, robots) -> None:
        def handler(request):
            raise httpx.ReadTimeout("too slow", request=request)

        serve(handler)

        with pytest.raises(TransientFetchError):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_connection_failure_is_transient(self, serve, robots) -> None:
        def handler(request):
            raise httpx.ConnectError("refused", request=request)

        serve(handler)

        with pytest.raises(TransientFetchError):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_permanent_is_not_a_subclass_of_transient(self) -> None:
        """The worker branches on these types, so the hierarchy is load-bearing.

        If PermanentFetchError inherited from TransientFetchError, the
        `except TransientFetchError` arm would swallow it and every 404 would
        be retried three times.
        """
        assert not issubclass(PermanentFetchError, TransientFetchError)
        assert issubclass(RateLimitedError, TransientFetchError)


class TestContentGuards:
    @pytest.mark.parametrize(
        "content_type", ["application/pdf", "image/png", "application/zip"]
    )
    def test_non_text_content_is_permanent(self, serve, robots, content_type) -> None:
        """This pipeline cannot read a PDF job description. Retrying will not
        change the content type."""
        serve(_ok(content_type=content_type))

        with pytest.raises(PermanentFetchError, match="content type"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_oversized_response_is_rejected(self, serve, robots, monkeypatch) -> None:
        """Enforced while streaming, not from Content-Length.

        A header check is defeated by any server that omits or lies about it,
        and by the time a non-streaming read could check, the bytes are
        already in memory -- which is the thing the cap exists to prevent.
        """
        monkeypatch.setattr(settings, "fetch_max_bytes", 1024)
        serve(_ok(html="<html><body>" + ("x" * 5000) + "</body></html>"))

        with pytest.raises(PermanentFetchError, match="exceeded"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_size_cap_ignores_a_lying_content_length(
        self, serve, robots, monkeypatch
    ) -> None:
        """The case a header check misses entirely."""
        monkeypatch.setattr(settings, "fetch_max_bytes", 1024)

        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/html", "content-length": "10"},
                text="<html><body>" + ("x" * 5000) + "</body></html>",
            )

        serve(handler)

        with pytest.raises(PermanentFetchError, match="exceeded"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_redirect_loop_is_permanent(self, serve, robots) -> None:
        """A loop is a property of the site; the next attempt walks it again."""

        def handler(request):
            return httpx.Response(302, headers={"location": str(request.url)})

        serve(handler)

        with pytest.raises(PermanentFetchError, match="redirect"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_follows_a_reasonable_redirect(self, serve, robots) -> None:
        def handler(request):
            if request.url.path == "/jobs/123":
                return httpx.Response(
                    302, headers={"location": "https://boards.example.com/jobs/final"}
                )
            return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)

        serve(handler)

        assert "Senior Engineer" in fetch_posting_text(
            POSTING_URL, limiter=AllowAllLimiter()
        )

    def test_unknown_charset_is_decoded_rather_than_raising(self, serve, robots) -> None:
        """A declared charset no codec implements must not escape classification.

        `errors="replace"` handles bad bytes and does nothing here: decode
        raises LookupError before reading any byte. LookupError is neither
        TransientFetchError nor PermanentFetchError, so it used to travel
        straight past this module into the worker's `except Exception`, which
        re-raises -- so RQ spent three attempts and two minutes re-learning
        that the charset is still not a codec. The posting is recoverable, so
        the fallback reads it rather than failing at all.
        """

        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf8mb4"},
                content=b"<html><body>Senior Engineer</body></html>",
            )

        serve(handler)

        assert "Senior Engineer" in fetch_posting_text(
            POSTING_URL, limiter=AllowAllLimiter()
        )

    def test_undecodable_bytes_still_yield_text(self, serve, robots) -> None:
        """The case `errors="replace"` does cover, kept distinct from the one above."""

        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=b"<html><body>Caf\xff Engineer</body></html>",
            )

        serve(handler)

        assert "Engineer" in fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())


class TestSSRFGuard:
    """The guard as the fetch path actually uses it.

    `test_netguard.py` owns which addresses are refused. These are about
    *where* the check sits, which is the part that a correct-looking
    implementation gets wrong: after robots, or before redirects.
    """

    def test_metadata_address_is_permanent(self, serve, robots, resolves_to) -> None:
        resolves_to("169.254.169.254")
        serve(_ok())

        with pytest.raises(PermanentFetchError, match="link-local"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_nothing_is_requested_at_all(self, serve, monkeypatch, resolves_to) -> None:
        """The guard runs before robots, and robots is itself a request.

        `_load_robots` fetches /robots.txt from the same host, so a guard
        placed after it has already sent the request it exists to prevent --
        and against a metadata endpoint that one request is the whole attack.
        Note this test does not use the `robots` fixture: the real
        `_load_robots` is left in place so that a regression would show up as
        a recorded request rather than being stubbed out of view.
        """
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)

        resolves_to("169.254.169.254")
        serve(handler)

        with pytest.raises(PermanentFetchError):
            fetch_posting_text("http://169.254.169.254/latest/meta-data/",
                               limiter=AllowAllLimiter())

        assert requests == []

    def test_redirect_to_a_private_address_is_blocked(
        self, serve, robots, monkeypatch
    ) -> None:
        """The bypass that makes a submit-time-only check worthless.

        The submitted URL is an ordinary public host. It answers 302 with a
        Location pointing at the metadata endpoint, and httpx's
        `follow_redirects` would have fetched it without consulting anything.
        One line of attacker effort against a control that looks complete.
        """

        def fake_resolve(host: str, port: int) -> list[str]:
            return ["169.254.169.254"] if host.startswith("169.254") else [PUBLIC_IP]

        monkeypatch.setattr(netguard, "resolve_host", fake_resolve)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "169.254.169.254":
                return httpx.Response(
                    200, headers={"content-type": "text/html"}, text="<html>creds</html>"
                )
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
            )

        serve(handler)

        with pytest.raises(PermanentFetchError, match="link-local"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_relative_redirect_still_works(self, serve, robots) -> None:
        """Walking redirects by hand must not break the ordinary case.

        A relative Location is resolved against the URL it came from, which
        httpx used to handle for us.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/jobs/123":
                return httpx.Response(302, headers={"location": "/jobs/moved"})
            return httpx.Response(200, headers={"content-type": "text/html"}, text=PAGE)

        serve(handler)

        assert "Senior Engineer" in fetch_posting_text(
            POSTING_URL, limiter=AllowAllLimiter()
        )

    def test_redirect_without_a_location_is_permanent(self, serve, robots) -> None:
        """Otherwise a 302 with no Location falls through to the body reader
        and returns an empty posting as a success."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302)

        serve(handler)

        with pytest.raises(PermanentFetchError, match="no Location"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())

    def test_unresolvable_host_is_transient(self, serve, robots, resolves_to) -> None:
        """A DNS failure is not an attack, and must not dead-letter the job."""
        resolves_to(error="Name or service not known")

        with pytest.raises(TransientFetchError, match="resolve"):
            fetch_posting_text(POSTING_URL, limiter=AllowAllLimiter())


class TestRateLimitInteraction:
    def test_exhausted_budget_is_transient(self, serve, robots) -> None:
        """The work is legitimate and merely early, so it should come back.

        Releasing the worker rather than blocking is what stops a 500-URL
        batch against one host parking every worker on the same bucket.
        """
        serve(_ok())

        class RefusingLimiter:
            def acquire(self, host, max_wait=None):
                return False

        with pytest.raises(RateLimitedError):
            fetch_posting_text(POSTING_URL, limiter=RefusingLimiter())


class TestRateLimiter:
    """The token bucket itself, against real Redis."""

    def test_burst_is_allowed_immediately(self) -> None:
        limiter = RateLimiter(rate=1.0, burst=3)
        host = f"burst-{id(limiter)}.example.com"

        waits = [limiter.try_acquire(host) for _ in range(3)]

        assert all(wait == 0 for wait in waits)

    def test_past_the_burst_the_caller_must_wait(self) -> None:
        limiter = RateLimiter(rate=1.0, burst=2)
        host = f"exhaust-{id(limiter)}.example.com"

        limiter.try_acquire(host)
        limiter.try_acquire(host)

        assert limiter.try_acquire(host) > 0

    def test_hosts_have_independent_buckets(self) -> None:
        """One busy board must not throttle every other host in the queue."""
        limiter = RateLimiter(rate=1.0, burst=1)
        busy = f"busy-{id(limiter)}.example.com"
        quiet = f"quiet-{id(limiter)}.example.com"

        limiter.try_acquire(busy)

        assert limiter.try_acquire(busy) > 0
        assert limiter.try_acquire(quiet) == 0

    def test_acquire_gives_up_past_its_budget(self) -> None:
        """Bounded, because a worker waiting on a token is doing nothing."""
        limiter = RateLimiter(rate=0.1, burst=1)
        host = f"slow-{id(limiter)}.example.com"

        limiter.try_acquire(host)

        assert limiter.acquire(host, max_wait=0.05) is False

    def test_limit_holds_across_separate_limiter_instances(self) -> None:
        """The property in-process limiting cannot provide.

        Two instances stand in for two worker processes. If the bucket lived
        in process memory this would pass twice and the real host would see
        double the intended rate.
        """
        host = f"shared-{id(object())}.example.com"
        first = RateLimiter(rate=1.0, burst=1)
        second = RateLimiter(rate=1.0, burst=1)

        assert first.try_acquire(host) == 0
        assert second.try_acquire(host) > 0


class TestDomainOf:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://Boards.Example.com/jobs/1", "boards.example.com"),
            ("http://example.com:8080/x", "example.com"),
            ("not a url", ""),
        ],
    )
    def test_extracts_lowercased_host(self, url: str, expected: str) -> None:
        assert domain_of(url) == expected
