"""Board adapters: three APIs that agree on nothing.

Fixtures are trimmed copies of real responses captured on 2026-08-03, not
invented shapes. That matters because the bugs these guard against are all
"the board does not look like the other boards":

    board       envelope        id       url            title   change signal   content
    ---------   -------------   ------   ------------   -----   -------------   -----------
    Greenhouse  {"jobs": [..]}  id       absolute_url   title   updated_at      none
    Lever       [..] bare list  id       hostedUrl      text    none            inline
    Ashby       {"jobs": [..]}  id       jobUrl         title   none            inline

Every column differs somewhere, and the network is stubbed at `_get_json` so
these run offline and deterministically -- the live checks belong in the
crawl, not in a suite that has to pass on a plane.
"""

import pytest

from app import boards
from app.boards import (
    DiscoveredPosting,
    enumerate_ashby,
    enumerate_greenhouse,
    enumerate_lever,
    enumerate_source,
)
from app.workers.fetch import PermanentFetchError

GREENHOUSE = {
    "jobs": [
        {
            "id": 5101378008,
            "absolute_url": "https://job-boards.greenhouse.io/anthropic/jobs/5101378008",
            "title": "Account Executive, Public Sector",
            "updated_at": "2026-07-14T18:35:00-04:00",
        },
        {
            "id": 4444,
            "absolute_url": "https://job-boards.greenhouse.io/anthropic/jobs/4444",
            "title": "Research Engineer",
            "updated_at": "2026-07-20T09:00:00-04:00",
        },
    ]
}

LEVER = [
    {
        "id": "890b2c0f-f46f-4a4b-bb73-3a6af6e0edd5",
        "hostedUrl": "https://jobs.lever.co/spotify/890b2c0f",
        "text": "Advertiser Solutions Vendor Lead",
        "descriptionPlain": "Sell what you love. " * 20,
        "additionalPlain": "We are an equal opportunity employer.",
        "createdAt": 1784569799619,
    }
]

ASHBY = {
    "jobs": [
        {
            "id": "7458d4e9-da2e-47bd-98cb-adfda43d42b2",
            "title": "Engineering Manager - EU",
            "jobUrl": "https://jobs.ashbyhq.com/ashby/7458d4e9",
            "descriptionPlain": "Lead a team building developer tools. " * 20,
            "publishedAt": "2024-03-04T14:29:08.532+00:00",
            "isListed": True,
        },
        {
            "id": "withdrawn-1",
            "title": "Unlisted Role",
            "jobUrl": "https://jobs.ashbyhq.com/ashby/withdrawn-1",
            "descriptionPlain": "Should never be crawled.",
            "isListed": False,
        },
    ]
}


@pytest.fixture
def payload(monkeypatch):
    """Serve a canned board response in place of the network."""

    def install(value):
        monkeypatch.setattr(boards, "_get_json", lambda url: value)

    return install


class TestGreenhouse:
    def test_maps_the_fields(self, payload) -> None:
        payload(GREENHOUSE)

        postings = enumerate_greenhouse("anthropic")

        assert len(postings) == 2
        first = postings[0]
        assert first.external_id == "5101378008"
        assert first.url.endswith("/jobs/5101378008")
        assert first.title == "Account Executive, Public Sector"

    def test_external_id_is_a_string(self, payload) -> None:
        """Greenhouse sends an integer; Lever and Ashby send UUIDs. The
        canonical key is built from this, so a type that varies by board
        would make one board's keys unequal to their own stored form."""
        payload(GREENHOUSE)

        assert isinstance(enumerate_greenhouse("anthropic")[0].external_id, str)

    def test_updated_at_is_parsed_and_utc(self, payload) -> None:
        """The whole reason a re-crawl is cheap. A naive datetime here would
        raise on comparison against the timestamptz column, and a wrong
        offset would make every posting look freshly changed."""
        payload(GREENHOUSE)

        updated = enumerate_greenhouse("anthropic")[0].updated_at

        assert updated is not None
        assert updated.tzinfo is not None
        assert updated.hour == 22  # 18:35-04:00 is 22:35Z

    def test_carries_no_inline_content(self, payload) -> None:
        """Greenhouse omits descriptions unless asked, so these postings must
        be fetched individually -- which is what `updated_at` then gates."""
        payload(GREENHOUSE)

        assert all(p.content is None for p in enumerate_greenhouse("anthropic"))

    def test_rows_missing_an_id_or_url_are_dropped(self, payload) -> None:
        payload({"jobs": [{"title": "Broken"}, *GREENHOUSE["jobs"]]})

        assert len(enumerate_greenhouse("anthropic")) == 2

    def test_a_list_payload_is_permanent(self, payload) -> None:
        """Greenhouse wraps in an object. Receiving Lever's shape means the
        API changed or a proxy is answering, and neither improves on retry."""
        payload([])

        with pytest.raises(PermanentFetchError):
            enumerate_greenhouse("anthropic")


class TestLever:
    def test_reads_a_bare_list(self, payload) -> None:
        """Lever returns a top-level array where the others return an object.

        Getting this wrong does not raise -- `{}.get("jobs")` on a list
        raises, but a defensive `or []` would read as "the board has no
        postings", so a live company silently contributes nothing.
        """
        payload(LEVER)

        postings = enumerate_lever("spotify")

        assert len(postings) == 1
        assert postings[0].external_id == "890b2c0f-f46f-4a4b-bb73-3a6af6e0edd5"
        assert postings[0].url == "https://jobs.lever.co/spotify/890b2c0f"

    def test_title_comes_from_text_not_title(self, payload) -> None:
        payload(LEVER)

        assert enumerate_lever("spotify")[0].title == "Advertiser Solutions Vendor Lead"

    def test_joins_the_description_parts(self, payload) -> None:
        """Lever splits a posting across several plain-text fields, and the
        requirements usually live outside `descriptionPlain`. Taking that
        field alone drops exactly the half the scorer needs."""
        payload(LEVER)

        content = enumerate_lever("spotify")[0].content

        assert content is not None
        assert "Sell what you love." in content
        assert "equal opportunity employer" in content

    def test_has_no_change_signal(self, payload) -> None:
        """Lever publishes createdAt and no updatedAt, so there is nothing to
        compare. The content hash covers edits instead -- which is affordable
        precisely because the content arrives free."""
        payload(LEVER)

        assert enumerate_lever("spotify")[0].updated_at is None

    def test_an_object_payload_is_permanent(self, payload) -> None:
        payload({"jobs": []})

        with pytest.raises(PermanentFetchError):
            enumerate_lever("spotify")


class TestAshby:
    def test_maps_the_fields(self, payload) -> None:
        payload(ASHBY)

        postings = enumerate_ashby("ashby")

        assert postings[0].url == "https://jobs.ashbyhq.com/ashby/7458d4e9"
        assert postings[0].title == "Engineering Manager - EU"
        assert postings[0].content is not None

    def test_unlisted_postings_are_excluded(self, payload) -> None:
        """`isListed: false` means the company pulled it from their public
        board. Crawling it anyway keeps a withdrawn role in the feed, and
        closure detection would never remove it because the board still
        returns it."""
        payload(ASHBY)

        ids = [p.external_id for p in enumerate_ashby("ashby")]

        assert "withdrawn-1" not in ids
        assert len(ids) == 1

    def test_falls_back_to_html_when_plain_text_is_absent(self, payload) -> None:
        payload(
            {
                "jobs": [
                    {
                        "id": "html-only",
                        "title": "Backend Engineer",
                        "jobUrl": "https://jobs.ashbyhq.com/x/html-only",
                        "descriptionHtml": "<div><p>We need Python.</p><script>x()</script></div>",
                    }
                ]
            }
        )

        content = enumerate_ashby("x")[0].content

        assert "We need Python." in content
        assert "x()" not in content  # script stripped by html_to_text


class TestDispatch:
    @pytest.mark.parametrize(
        "kind, fixture", [("greenhouse", GREENHOUSE), ("lever", LEVER), ("ashby", ASHBY)]
    )
    def test_every_declared_kind_has_an_adapter(self, payload, kind, fixture) -> None:
        payload(fixture)

        assert enumerate_source(kind, "token")

    def test_an_unknown_kind_is_permanent(self) -> None:
        """`careers_page` is a declared source kind with no adapter, because
        HTML scraping of arbitrary company sites is the fallback spec section
        9 item 3 warns against defaulting to. Raising is better than
        enumerating nothing, which would look like an empty board and trip
        closure detection across it."""
        with pytest.raises(PermanentFetchError, match="no adapter"):
            enumerate_source("careers_page", "acme")

    def test_returns_discovered_postings(self, payload) -> None:
        """One shape out, whatever went in -- that is the seam's whole job."""
        payload(GREENHOUSE)

        assert all(
            isinstance(p, DiscoveredPosting) for p in enumerate_source("greenhouse", "x")
        )
