"""URL normalization and list parsing.

Pure functions -- no database, no network. This is the messiest input the
system takes: a hand-collected list pasted from address bars, newsletters,
and shared links.
"""

import pytest

from app.urls import TRACKING_PARAMS, normalize_url, parse_url_list


class TestNormalizeUrl:
    def test_already_canonical_is_unchanged(self) -> None:
        url = "https://boards.greenhouse.io/acme/jobs/123"

        assert normalize_url(url) == url

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("HTTPS://Example.COM/Jobs/1", "https://example.com/Jobs/1"),
            ("https://EXAMPLE.com/x", "https://example.com/x"),
        ],
    )
    def test_scheme_and_host_lowercased_but_not_path(self, raw, expected) -> None:
        """Scheme and host are case-insensitive per RFC 3986. Paths are not.

        Lowercasing the path would be a correctness bug, not a tidy-up --
        some boards serve case-sensitive slugs, so /Jobs/1 and /jobs/1 can be
        different documents.
        """
        assert normalize_url(raw) == expected

    def test_fragment_dropped(self) -> None:
        """A fragment never reaches the server, so it cannot select a document."""
        assert (
            normalize_url("https://example.com/jobs/1#requirements")
            == "https://example.com/jobs/1"
        )

    def test_tracking_params_stripped(self) -> None:
        assert (
            normalize_url("https://example.com/j/1?utm_source=twitter&utm_medium=social")
            == "https://example.com/j/1"
        )

    def test_meaningful_params_survive(self) -> None:
        """The reason tracking params are a denylist rather than an allowlist.

        ?gh_jid= selects which posting you get. An allowlist would have to
        enumerate every meaningful parameter on every job board, and dropping
        one it had not heard of would silently fetch the wrong page --
        keeping an unknown parameter only costs a duplicate row.
        """
        assert (
            normalize_url("https://example.com/embed?gh_jid=4001&utm_source=x")
            == "https://example.com/embed?gh_jid=4001"
        )

    def test_query_order_does_not_matter(self) -> None:
        assert normalize_url("https://e.com/j?b=2&a=1") == normalize_url(
            "https://e.com/j?a=1&b=2"
        )

    def test_blank_valued_param_survives(self) -> None:
        """?debug distinguishes a page even with no value; dropping it merges two."""
        assert normalize_url("https://e.com/j?debug") == "https://e.com/j?debug="

    def test_trailing_slash_removed(self) -> None:
        assert normalize_url("https://e.com/jobs/1/") == "https://e.com/jobs/1"

    def test_root_path_collapses_with_the_bare_host(self) -> None:
        """"e.com" and "e.com/" name one resource and must produce one key."""
        assert normalize_url("https://e.com/") == normalize_url("https://e.com")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://e.com:443/j", "https://e.com/j"),
            ("http://e.com:80/j", "http://e.com/j"),
        ],
    )
    def test_default_port_dropped(self, raw, expected) -> None:
        assert normalize_url(raw) == expected

    def test_non_default_port_kept(self) -> None:
        assert normalize_url("https://e.com:8443/j") == "https://e.com:8443/j"

    def test_bare_host_gets_https(self) -> None:
        """"example.com/jobs/1" is a common paste and is not worth discarding."""
        assert normalize_url("example.com/jobs/1") == "https://example.com/jobs/1"

    def test_surrounding_whitespace_ignored(self) -> None:
        assert normalize_url("  https://e.com/j  ") == "https://e.com/j"

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "not a url at all", "Here are the jobs I found:", "just-words"],
    )
    def test_unusable_lines_return_none(self, raw: str) -> None:
        assert normalize_url(raw) is None

    @pytest.mark.parametrize(
        "raw",
        [
            "file:///etc/passwd",
            "ftp://example.com/x",
            "mailto:jobs@example.com",
            "javascript:alert(1)",
            "data:text/html,<script>",
        ],
    )
    def test_non_http_schemes_rejected(self, raw: str) -> None:
        """Not tidiness -- these strings become fetch targets.

        A worker handed file:///etc/passwd is an SSRF primitive, so rejecting
        the scheme here is a security boundary rather than input hygiene.
        """
        assert normalize_url(raw) is None

    @pytest.mark.parametrize(
        "raw",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://127.0.0.1:6379/",
            "http://localhost:5432/",
            "http://10.0.0.5/admin",
            "http://192.168.1.1/",
            "http://172.16.0.1/",
            "http://[::1]/",
            "http://0.0.0.0/",
        ],
    )
    def test_literal_internal_addresses_rejected(self, raw: str) -> None:
        """Rejecting non-http schemes does nothing about these.

        Every one is a well-formed http URL with a real host, so it passes
        every other rule in normalize_url. The first is the AWS/GCP/Azure
        metadata endpoint, which serves instance credentials to anything on
        the local link, and the second is our own session store.
        """
        assert normalize_url(raw) is None

    def test_hostnames_are_not_rejected_here(self) -> None:
        """The literal check does no DNS on purpose -- a 500-URL batch would
        otherwise cost 500 resolutions inside a request handler. Hostnames
        pointing at private addresses are caught at fetch time instead."""
        assert normalize_url("https://boards.example.com/jobs/1") is not None

    def test_tracking_param_matching_is_case_insensitive(self) -> None:
        assert normalize_url("https://e.com/j?UTM_Source=x") == "https://e.com/j"

    def test_every_denylist_entry_is_lowercase(self) -> None:
        """Matching lowercases the key, so a capitalised entry is unreachable."""
        assert [param for param in TRACKING_PARAMS if param != param.lower()] == []


class TestParseUrlList:
    def test_parses_one_url_per_line(self) -> None:
        urls, rejected, duplicates = parse_url_list(
            "https://e.com/1\nhttps://e.com/2\n", cap=10
        )

        assert urls == ["https://e.com/1", "https://e.com/2"]
        assert (rejected, duplicates) == (0, 0)

    def test_preserves_order(self) -> None:
        """The order someone wrote their list in is meaningful to them."""
        urls, _, _ = parse_url_list("https://e.com/b\nhttps://e.com/a", cap=10)

        assert urls == ["https://e.com/b", "https://e.com/a"]

    def test_blank_lines_are_not_rejections(self) -> None:
        """A file with paragraph spacing is formatted, not half-broken."""
        urls, rejected, _ = parse_url_list("https://e.com/1\n\n\n  \nhttps://e.com/2", cap=10)

        assert len(urls) == 2
        assert rejected == 0

    def test_duplicates_counted_after_normalization(self) -> None:
        """The case a pasted list actually hits.

        The same posting arrives twice under different tracking URLs, which
        is invisible until both are normalized.
        """
        text = "https://e.com/j/1?utm_source=a\nhttps://e.com/j/1?utm_source=b\n"

        urls, _, duplicates = parse_url_list(text, cap=10)

        assert urls == ["https://e.com/j/1"]
        assert duplicates == 1

    def test_unusable_lines_counted_as_rejected(self) -> None:
        urls, rejected, _ = parse_url_list("https://e.com/1\nnot a url\n", cap=10)

        assert len(urls) == 1
        assert rejected == 1

    def test_cap_applied_after_dedupe(self) -> None:
        """A list padded with one posting under many tracking URLs is not
        penalised for it -- the cap counts distinct postings."""
        text = "\n".join(f"https://e.com/j/1?utm_source={i}" for i in range(20))

        urls, rejected, duplicates = parse_url_list(text, cap=2)

        assert urls == ["https://e.com/j/1"]
        assert duplicates == 19
        assert rejected == 0

    def test_over_cap_urls_are_reported_not_dropped(self) -> None:
        """The count has to come back, or the user cannot tell what is missing.

        Silently ingesting 2 of 5 is worse than refusing, because nothing
        tells them which 3 were left out.
        """
        text = "\n".join(f"https://e.com/j/{i}" for i in range(5))

        urls, rejected, _ = parse_url_list(text, cap=2)

        assert len(urls) == 2
        assert rejected == 3

    def test_every_non_blank_line_is_accounted_for(self) -> None:
        """accepted + rejected + duplicates == non-blank lines. The invariant
        that makes the response to a batch upload trustworthy."""
        text = (
            "https://e.com/1\n"
            "https://e.com/2\n"
            "https://e.com/1\n"       # duplicate
            "garbage\n"                # rejected
            "\n"                       # blank, not counted
            "https://e.com/3\n"
            "https://e.com/4\n"        # over cap of 3
        )

        urls, rejected, duplicates = parse_url_list(text, cap=3)

        assert len(urls) + rejected + duplicates == 6

    def test_empty_input(self) -> None:
        assert parse_url_list("", cap=10) == ([], 0, 0)

    def test_handles_crlf_line_endings(self) -> None:
        """A .txt written on Windows, which is the likely case here."""
        urls, _, _ = parse_url_list("https://e.com/1\r\nhttps://e.com/2\r\n", cap=10)

        assert urls == ["https://e.com/1", "https://e.com/2"]
