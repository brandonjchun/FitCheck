"""Board adapters: eight APIs that agree on nothing.

Fixtures are trimmed copies of real responses -- the first three captured
2026-08-03, the next four 2026-08-06 -- not invented shapes. That matters
because the bugs these guard against are all "the board does not look like the
other boards":

    board            envelope           id          url            title   change signal    content
    --------------   ----------------   ---------   ------------   -----   --------------   -------
    Greenhouse       {"jobs": [..]}     id          absolute_url   title   updated_at       inline
    Lever            [..] bare list     id          hostedUrl      text    none             inline
    Ashby            {"jobs": [..]}     id          jobUrl         title   none             inline
    Workable         {"jobs": [..]}     shortcode   url            title   none             inline
    SmartRecruiters  {"content": [..]}  id          composed       name    releasedDate     none
    BreezyHR         [..] bare list     id          url            name    published_date   none
    Rippling         {"items": [..]}    uuid        url            name    none             none
    USAJOBS          nested 3 deep      MatchedId   PositionURI    Position PublicationStart inline

Every column differs somewhere. Two are worth calling out because they are not
merely different but actively misleading: SmartRecruiters carries
`defaultJobAd` as a **boolean**, so an adapter that reads it as the ad gets
`True` where prose should be; and USAJOBS mixes a string field with an array
field inside one object, so interpolating both the same way puts a Python list
repr into the text.

The network is stubbed at `_get_json`, so these run offline and
deterministically -- the live checks belong in the crawl, not in a suite that
has to pass on a plane. `enumerate_usajobs` is the one adapter never verified
against a live response, because its key is per-developer; these fixtures are
its only evidence and are built from the documented shape.
"""

import pytest

from app import boards
from app.boards import (
    DiscoveredPosting,
    enumerate_ashby,
    enumerate_breezy,
    enumerate_greenhouse,
    enumerate_lever,
    enumerate_rippling,
    enumerate_smartrecruiters,
    enumerate_source,
    enumerate_usajobs,
    enumerate_workable,
)
from app.workers.fetch import PermanentFetchError

# `content` is HTML-escaped exactly as the live API returns it -- a string
# whose *characters* are `&lt;p&gt;`, not `<p>`. Writing the fixture in the
# convenient form instead would make the decode test vacuous, since a stripper
# run on already-unescaped markup passes either way.
GREENHOUSE = {
    "jobs": [
        {
            "id": 5101378008,
            "absolute_url": "https://job-boards.greenhouse.io/anthropic/jobs/5101378008",
            "title": "Account Executive, Public Sector",
            "updated_at": "2026-07-14T18:35:00-04:00",
            "content": "&lt;div&gt;&lt;p&gt;Sell to the public sector.&lt;/p&gt;&lt;/div&gt;",
        },
        {
            "id": 4444,
            "absolute_url": "https://job-boards.greenhouse.io/anthropic/jobs/4444",
            "title": "Research Engineer",
            "updated_at": "2026-07-20T09:00:00-04:00",
            "content": "&lt;p&gt;Train &amp;amp; evaluate models.&lt;/p&gt;",
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


# Trimmed from `apply.workable.com/api/v1/widget/accounts/blueground?details=true`.
# `description` is real HTML, not escaped HTML -- the opposite of Greenhouse, so
# a single strip is correct here and a double one would eat the prose.
WORKABLE = {
    "name": "Blueground",
    "jobs": [
        {
            "shortcode": "186545F8C1",
            "title": "Client Experience Coordinator",
            "url": "https://apply.workable.com/j/186545F8C1",
            "description": "<h3>Redefining how people live.</h3><p>Join us.</p>",
            "requirements": "<ul><li>3 years of Greek</li></ul>",
            "benefits": "<p>Flexible hours</p>",
            "published_on": "2026-02-12",
            "created_at": "2025-09-30",
            "experience": "Associate",
            "employment_type": "Full-time",
            "telecommuting": False,
        },
        # No shortcode -- unaddressable, so it cannot become a canonical_key.
        {"title": "Ghost", "url": "https://apply.workable.com/j/nope"},
    ],
}

# Trimmed from `api.smartrecruiters.com/v1/companies/Visa/postings`. Note
# `defaultJobAd: true` -- a boolean, and the trap this fixture exists to hold.
SMARTRECRUITERS = {
    "offset": 0,
    "limit": 100,
    "totalFound": 2,
    "content": [
        {
            "id": "744000133907678",
            "uuid": "e0a1f0c2-0000-4000-8000-000000000001",
            "name": "Sr. Manager",
            "company": {"identifier": "Visa", "name": "Visa"},
            "releasedDate": "2026-07-30T10:15:00.000Z",
            "experienceLevel": {"id": "director_and_above"},
            "location": {"city": "Austin", "country": "us"},
            "defaultJobAd": True,
        },
        {
            "id": "744000133907679",
            "name": "Staff Software Engineer",
            "releasedDate": "2026-08-01T09:00:00.000Z",
            "postingUrl": "https://jobs.smartrecruiters.com/Visa/744000133907679",
            "defaultJobAd": True,
        },
    ],
}

# Trimmed from `api.rippling.com/platform/api/ats/v1/board/rippling/jobs`.
# No date field of any kind, which is the property the adapter documents.
RIPPLING = {
    "items": [
        {
            "uuid": "2f0674e6-f01f-4ecd-b459-e947241c211f",
            "name": "Account Executive - Accountants Channel",
            "department": {"id": "Sales", "label": "Sales"},
            "url": "https://ats.rippling.com/rippling/jobs/2f0674e6-f01f-4ecd-b459-e947241c211f",
            "workLocation": {"label": "New York"},
        },
        {"name": "No uuid", "url": "https://ats.rippling.com/rippling/jobs/x"},
    ]
}

# Trimmed from `breezy.breezy.hr/json` -- a bare list, like Lever.
BREEZY = [
    {
        "id": "98323abf2296",
        "friendly_id": "98323abf2296-employee-12",
        "name": "Employee #12",
        "url": "https://breezy.breezy.hr/p/98323abf2296-employee-12",
        "published_date": "2024-02-15T14:37:22.684Z",
        "type": {"id": "full-time", "name": "Full-Time"},
    },
]

# Built from the documented USAJOBS shape, not a captured response -- see the
# module docstring. `MajorDuties` is an array while `JobSummary` is a string,
# in the same object, which is the corruption this fixture guards against.
USAJOBS = {
    "SearchResult": {
        "SearchResultCount": 1,
        "SearchResultCountAll": 1,
        "SearchResultItems": [
            {
                "MatchedObjectId": "830216900",
                "MatchedObjectDescriptor": {
                    "PositionID": "IRS-26-0001",
                    "PositionTitle": "Data Scientist",
                    "PositionURI": "https://www.usajobs.gov/job/830216900",
                    "PublicationStartDate": "2026-07-02",
                    "UserArea": {
                        "Details": {
                            "JobSummary": "Analyse compliance data.",
                            "MajorDuties": [
                                "Build models.",
                                "Brief leadership.",
                            ],
                            "Qualifications": "Five years of Python.",
                        }
                    },
                },
            },
            # No PositionURI -- nothing to fetch, so it cannot be stored.
            {"MatchedObjectId": "1", "MatchedObjectDescriptor": {"PositionTitle": "X"}},
        ],
    }
}


@pytest.fixture
def payload(monkeypatch):
    """Serve a canned board response in place of the network.

    The stub takes `headers` because USAJOBS passes them and every other
    adapter does not. A one-argument stub here would raise TypeError from
    inside the adapter under test, which reads as a bug in the adapter rather
    than in the fixture.
    """

    def install(value):
        monkeypatch.setattr(boards, "_get_json", lambda url, headers=None: value)

    return install


@pytest.fixture
def pages(monkeypatch):
    """Serve a different response per call, for the paginated boards.

    Records the URLs it was asked for, because with pagination the *requests*
    are half the behaviour under test -- an adapter that returns the right rows
    while walking the offset wrongly still stops early on a real board.
    """

    def install(values):
        seen: list[str] = []
        queue = list(values)

        def fake(url, headers=None):
            seen.append(url)
            return queue.pop(0) if queue else values[-1]

        monkeypatch.setattr(boards, "_get_json", fake)
        return seen

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

    def test_carries_inline_content(self, payload) -> None:
        """`?content=true` is what keeps a new board off the per-posting
        fetch path, so every row must arrive with its description."""
        payload(GREENHOUSE)

        assert all(p.content for p in enumerate_greenhouse("anthropic"))

    def test_content_is_unescaped_before_stripping(self, payload) -> None:
        """The trap this board carries alone. Greenhouse escapes its HTML, so
        the stripper has to unescape first -- run once on the raw string it
        yields the markup as literal text, and every skill, keyword, and
        content hash downstream then matches on angle brackets rather than
        prose. Nothing else fails; the text is simply wrong forever.
        """
        payload(GREENHOUSE)

        text = enumerate_greenhouse("anthropic")[0].content

        assert text is not None
        assert "Sell to the public sector." in text
        # Neither the escaped form nor the decoded-but-unstripped form.
        for leak in ("&lt;", "&gt;", "&quot;", "<div>", "<p>"):
            assert leak not in text

    def test_a_posting_with_no_body_falls_back_to_fetching(self, payload) -> None:
        """An empty `content` must leave `content` None rather than "", or the
        ingest side reads a blank description as a real one and hashes it --
        skipping the per-posting fetch that would have found the actual text.
        """
        payload({"jobs": [{**GREENHOUSE["jobs"][0], "content": ""}]})

        assert enumerate_greenhouse("anthropic")[0].content is None

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


class TestWorkable:
    def test_maps_the_fields(self, payload) -> None:
        payload(WORKABLE)

        rows = enumerate_workable("blueground")

        assert len(rows) == 1
        assert rows[0].external_id == "186545F8C1"
        assert rows[0].url == "https://apply.workable.com/j/186545F8C1"
        assert rows[0].title == "Client Experience Coordinator"

    def test_strips_the_html_once_not_twice(self, payload) -> None:
        """Workable returns real HTML, unlike Greenhouse's escaped form. Running
        the unescape step on it first would be harmless here but the reverse --
        skipping the strip -- leaves tags in the text every skill is matched
        against."""
        payload(WORKABLE)

        content = enumerate_workable("blueground")[0].content

        assert "<h3>" not in content
        assert "Redefining how people live." in content

    def test_appends_requirements_and_benefits(self, payload) -> None:
        """The requirements block is a separate field, and it is the half the
        scorer needs -- `description` alone is the pitch."""
        payload(WORKABLE)

        content = enumerate_workable("blueground")[0].content

        assert "3 years of Greek" in content
        assert "Flexible hours" in content

    def test_publishes_no_change_signal(self, payload) -> None:
        """`published_on` is a publication date, not a modification one. Using
        it would skip an edited posting forever; the content hash catches edits
        instead, which is affordable because the content arrives free."""
        payload(WORKABLE)

        assert enumerate_workable("blueground")[0].updated_at is None

    def test_an_empty_board_is_not_an_error(self, payload) -> None:
        """Verified live: `hotjar` and `typeform` both resolve and return zero
        jobs because they have nothing open. Raising would trip
        `consecutive_failures` on a healthy board that is simply not hiring."""
        payload({"name": "Hotjar", "jobs": []})

        assert enumerate_workable("hotjar") == []

    def test_a_payload_without_jobs_is_permanent(self, payload) -> None:
        payload({"name": "Hotjar"})

        with pytest.raises(PermanentFetchError, match="Workable"):
            enumerate_workable("hotjar")


class TestSmartRecruiters:
    def test_maps_the_fields(self, payload) -> None:
        payload(SMARTRECRUITERS)

        rows = enumerate_smartrecruiters("Visa")

        assert [r.external_id for r in rows] == ["744000133907678", "744000133907679"]
        assert rows[0].title == "Sr. Manager"

    def test_never_reads_defaultJobAd_as_content(self, payload) -> None:
        """It is a boolean. An adapter that treated it as the ad would store
        `True` as a posting body, hash it, and score against it -- and the row
        would look populated the whole way down."""
        payload(SMARTRECRUITERS)

        assert all(r.content is None for r in enumerate_smartrecruiters("Visa"))

    def test_composes_a_url_when_the_listing_omits_one(self, payload) -> None:
        """`postingUrl` is absent from the list response for some postings, and
        the composed form has to use the same id `external_id` does or the
        canonical_key will not match what the fetch resolves."""
        payload(SMARTRECRUITERS)

        rows = enumerate_smartrecruiters("Visa")

        assert rows[0].url == "https://jobs.smartrecruiters.com/Visa/744000133907678"
        assert rows[1].url == "https://jobs.smartrecruiters.com/Visa/744000133907679"

    def test_uses_releasedDate_as_the_change_signal(self, payload) -> None:
        payload(SMARTRECRUITERS)

        assert enumerate_smartrecruiters("Visa")[0].updated_at is not None

    def test_walks_every_page(self, pages) -> None:
        """The default page size is 10, so an adapter that trusts one response
        caps every board at ten postings and looks like a catalog of very small
        companies. This is the loop's only test -- no live board large enough to
        page was reachable, since SmartRecruiters ids are not brand slugs."""
        first = {
            "offset": 0,
            "limit": 100,
            "totalFound": 150,
            "content": [
                {"id": str(i), "name": f"Role {i}", "releasedDate": "2026-08-01"}
                for i in range(100)
            ],
        }
        second = {
            "offset": 100,
            "limit": 100,
            "totalFound": 150,
            "content": [
                {"id": str(i), "name": f"Role {i}", "releasedDate": "2026-08-01"}
                for i in range(100, 150)
            ],
        }
        seen = pages([first, second])

        rows = enumerate_smartrecruiters("Big")

        assert len(rows) == 150
        assert "offset=0" in seen[0]
        assert "offset=100" in seen[1]
        # Stopped rather than asking for a third page it did not need.
        assert len(seen) == 2

    def test_an_empty_page_ends_the_walk(self, pages) -> None:
        """`totalFound` over-reports on real boards -- it counts postings the
        public API will not return -- so an empty page has to end the loop or it
        spins to the page cap."""
        seen = pages(
            [
                {"totalFound": 999, "content": [{"id": "1", "name": "One"}]},
                {"totalFound": 999, "content": []},
            ]
        )

        assert len(enumerate_smartrecruiters("Weird")) == 1
        assert len(seen) == 2


class TestRippling:
    def test_maps_the_fields(self, payload) -> None:
        payload(RIPPLING)

        rows = enumerate_rippling("rippling")

        assert len(rows) == 1
        assert rows[0].external_id == "2f0674e6-f01f-4ecd-b459-e947241c211f"
        assert rows[0].title == "Account Executive - Accountants Channel"

    def test_has_neither_content_nor_a_change_signal(self, payload) -> None:
        """Both absent is what makes this the expensive board kind: every
        posting is re-fetched on every crawl, 738 of them on the live board."""
        payload(RIPPLING)

        row = enumerate_rippling("rippling")[0]
        assert row.content is None
        assert row.updated_at is None

    def test_accepts_a_bare_list_too(self, payload) -> None:
        """Rippling has answered with both shapes across versions. Unlike
        Lever, the bare list is the unexpected one here -- so it is handled
        rather than rejected, because the alternative reads as an empty board
        and trips closure detection across it."""
        payload(RIPPLING["items"])

        assert len(enumerate_rippling("rippling")) == 1


class TestBreezy:
    def test_maps_the_fields(self, payload) -> None:
        payload(BREEZY)

        rows = enumerate_breezy("breezy")

        assert len(rows) == 1
        assert rows[0].external_id == "98323abf2296"
        assert rows[0].url == "https://breezy.breezy.hr/p/98323abf2296-employee-12"
        assert rows[0].title == "Employee #12"

    def test_a_bare_list_is_the_expected_shape(self, payload) -> None:
        """Reading it as a dict yields nothing rather than raising -- the same
        trap Lever's docstring names."""
        payload({"jobs": BREEZY})

        with pytest.raises(PermanentFetchError, match="BreezyHR"):
            enumerate_breezy("breezy")

    def test_dates_its_postings(self, payload) -> None:
        payload(BREEZY)

        assert enumerate_breezy("breezy")[0].updated_at is not None


class TestUsajobs:
    def test_requires_a_key_and_says_so(self, monkeypatch) -> None:
        """Raising rather than returning empty. An unconfigured source that
        enumerates nothing is indistinguishable from an agency with no
        openings, and closure detection would tombstone the agency's whole
        catalog on a crawl that reported success."""
        monkeypatch.setattr(boards.settings, "usajobs_api_key", "")
        monkeypatch.setattr(boards.settings, "usajobs_email", "")

        with pytest.raises(PermanentFetchError, match="USAJOBS_API_KEY"):
            enumerate_usajobs("TR")

    @pytest.fixture
    def configured(self, monkeypatch):
        monkeypatch.setattr(boards.settings, "usajobs_api_key", "test-key")
        monkeypatch.setattr(boards.settings, "usajobs_email", "dev@example.com")

    def test_maps_the_nested_fields(self, payload, configured) -> None:
        payload(USAJOBS)

        rows = enumerate_usajobs("IRS")

        assert len(rows) == 1
        assert rows[0].external_id == "830216900"
        assert rows[0].url == "https://www.usajobs.gov/job/830216900"
        assert rows[0].title == "Data Scientist"
        assert rows[0].updated_at is not None

    def test_joins_an_array_field_without_leaking_a_list_repr(
        self, payload, configured
    ) -> None:
        """`MajorDuties` is an array and `JobSummary` is a string, in the same
        object. Interpolating both with `str()` puts brackets, quotes and commas
        into the text every skill and keyword is matched against -- the same
        class of silent corruption as double-escaped HTML."""
        payload(USAJOBS)

        content = enumerate_usajobs("IRS")[0].content

        assert "Build models." in content
        assert "Brief leadership." in content
        assert "[" not in content
        assert "'," not in content

    def test_carries_the_qualifications(self, payload, configured) -> None:
        """JobSummary alone is preamble. The qualifications are the half the
        scorer needs, so dropping them would leave a posting that reads fine and
        scores against nothing."""
        payload(USAJOBS)

        assert "Five years of Python." in enumerate_usajobs("IRS")[0].content

    def test_sends_the_key_and_the_registered_email(self, monkeypatch, configured) -> None:
        """Their terms ask that the User-Agent be the email the key was
        registered to, which is why it is a separate setting from
        `fetch_user_agent` rather than a reuse of it."""
        captured: dict = {}

        def fake(url, headers=None):
            captured["headers"] = headers or {}
            return USAJOBS

        monkeypatch.setattr(boards, "_get_json", fake)
        enumerate_usajobs("IRS")

        assert captured["headers"]["Authorization-Key"] == "test-key"
        assert captured["headers"]["User-Agent"] == "dev@example.com"

    def test_scopes_the_query_to_one_agency(self, monkeypatch, configured) -> None:
        """The whole reason this is per-agency: it keeps `board_token` meaningful
        and `sources.display_name` an employer, so `_company_for_source` stamps
        the agency rather than "USAJOBS" on every posting."""
        seen: list[str] = []

        def fake(url, headers=None):
            seen.append(url)
            return USAJOBS

        monkeypatch.setattr(boards, "_get_json", fake)
        enumerate_usajobs("IRS")

        assert "Organization=IRS" in seen[0]


class TestDispatch:
    @pytest.mark.parametrize(
        "kind, fixture",
        [
            ("greenhouse", GREENHOUSE),
            ("lever", LEVER),
            ("ashby", ASHBY),
            ("workable", WORKABLE),
            ("smartrecruiters", SMARTRECRUITERS),
            ("rippling", RIPPLING),
            ("breezy", BREEZY),
        ],
    )
    def test_every_declared_kind_has_an_adapter(self, payload, kind, fixture) -> None:
        payload(fixture)

        assert enumerate_source(kind, "token")

    def test_usajobs_dispatches_too(self, payload, monkeypatch) -> None:
        """Separate because it is the one kind that needs configuration, and a
        parametrized case would fail for the wrong reason."""
        monkeypatch.setattr(boards.settings, "usajobs_api_key", "k")
        monkeypatch.setattr(boards.settings, "usajobs_email", "e@example.com")
        payload(USAJOBS)

        assert enumerate_source("usajobs", "IRS")

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
