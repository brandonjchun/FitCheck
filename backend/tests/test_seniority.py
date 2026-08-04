"""Reading a level off a job title.

Every title in this file is a real one from the live catalog, which is the only
reason the false-positive cases below are here at all -- "Software Engineer,
International" and "Global Associate Director, Experiential & Content
Production" are not hypotheticals someone invented to be clever, they are rows
a substring match gets wrong.
"""

import pytest

from app.seniority import seniority_from_title


class TestStaff:
    """The bug that started this: 10 staff rows in a catalog of 1,587."""

    @pytest.mark.parametrize(
        "title",
        [
            "Staff Software Engineer, iOS",
            "Staff Engineer",
            "Staff Data Engineer — Subscriptions User Understanding",
            "Staff Platform Engineer - EU",
            "Staff Product Manager, Data Platform",
            "Staff Product Designer, Design Systems",
        ],
    )
    def test_a_staff_title_is_staff(self, title: str) -> None:
        assert seniority_from_title(title) == "staff"

    def test_senior_staff_is_staff_not_senior(self) -> None:
        """Titles stack, so the order the patterns run in is load-bearing.

        Checking "senior" first would demote every one of these, which is the
        more insidious version of the bug: a plausible answer, one rung down.
        """
        assert (
            seniority_from_title("Senior Staff Machine Learning Engineer, Content Platform")
            == "staff"
        )

    def test_a_plus_in_the_title_changes_nothing(self) -> None:
        """"Staff+ Engineer" is a staff role.

        Worth stating outright because the natural suspicion when titles go
        missing is that punctuation was rejected somewhere. Nothing in this
        path filters characters, and nothing should start: the live catalog
        holds "Solutions Engineer, Commercial+ - APJ", "Senior Software
        Engineer - TV Playback (C++)", and "Culture Marketing Lead - MENAT+",
        all stored and rendered intact.
        """
        assert seniority_from_title("Staff+ Engineer") == "staff"
        assert seniority_from_title("Staff+ Software Engineer (C++)") == "staff"

    @pytest.mark.parametrize(
        "title",
        [
            "Principal Product Manager, Engagement Journeys",
            "Distinguished Engineer",
        ],
    )
    def test_principal_and_distinguished_collapse_into_staff(self, title: str) -> None:
        """At or above staff on every ladder that uses them.

        They collapse rather than getting their own bucket because the
        vocabulary is fixed by `Seniority` and a promoted column with a
        frontend filter attached -- widening it is a schema decision, not
        something to smuggle in with a bug fix.
        """
        assert seniority_from_title(title) == "staff"

    def test_chief_of_staff_is_not_a_staff_engineer(self) -> None:
        """An executive-adjacent business role that happens to share a word."""
        assert seniority_from_title("Chief of Staff to the CTO") is None


class TestSenior:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Senior Engineer", "senior"),
            ("Senior Fullstack Engineer, Emerging Products", "senior"),
            ("Backend Senior Software Engineer, Identity", "senior"),
            ("Senior II Fullstack Software Engineer, GRC", "senior"),
            ("Sr. Backend Engineer", "senior"),
            ("Snr Data Engineer", "senior"),
            # "Associate" is not a level marker here -- see TestNoLevelStated.
            ("Senior Associate, Strategic Finance", "senior"),
        ],
    )
    def test_senior_titles(self, title: str, expected: str) -> None:
        assert seniority_from_title(title) == expected


class TestJunior:
    @pytest.mark.parametrize(
        "title",
        [
            "Junior Software Engineer",
            "Jr. Analyst",
            "Software Engineer Internship, Android",
            "Software Engineering Intern",
            "New Grad Software Engineer",
            "New Graduate Engineer, Platform",
            "Entry-Level Support Specialist",
            "Apprentice Technician",
        ],
    )
    def test_junior_titles(self, title: str) -> None:
        assert seniority_from_title(title) == "junior"

    def test_international_is_not_an_internship(self) -> None:
        """The word-boundary case, and a real row rather than a contrivance.

        "International" contains "intern". A substring match files three live
        postings as internships, including a Director role -- which is the kind
        of wrong that never looks wrong in a filter, only in the result count.
        """
        assert seniority_from_title("Software Engineer, International") is None
        assert seniority_from_title("Director, International Operations") is None
        assert seniority_from_title("Financial Partnerships Manager, International") is None


class TestNoLevelStated:
    """None, not "unknown" -- the distinction is the whole interface.

    "unknown" is an answer: the extraction read the posting and could not tell.
    None means this function was not asked something it can answer, so the
    caller defers to the extraction instead of overwriting it with something
    worse.
    """

    @pytest.mark.parametrize(
        "title",
        [
            "Software Engineer",
            "Music Editor, Indonesia",
            "Account Executive, SMB | Canada",
            # "Associate" reads as junior on a sales ladder and as senior on
            # several others. Guessing trades one wrong bucket for another.
            "Associate Director, Portfolio & Monetization Architecture",
            "Global Associate Director, Experiential & Content Production",
            "Associate Counsel, Commercial",
            # "Lead" is a scope word across whole non-engineering ladders.
            "Lead Counsel, UK & Europe",
            "Growth Lead",
            "Detection & Response, Lead",
        ],
    )
    def test_no_level_means_none(self, title: str) -> None:
        assert seniority_from_title(title) is None

    @pytest.mark.parametrize("title", [None, "", "   "])
    def test_a_missing_title_is_not_an_error(self, title: str | None) -> None:
        """Path A reaches here with nothing: a bare user-submitted URL has no
        board behind it to state a title."""
        assert seniority_from_title(title) is None

    def test_there_is_no_rule_producing_mid(self) -> None:
        """No title says "mid-level" in practice, and an unqualified "Software
        Engineer" is the closest thing -- reading a level into the *absence* of
        a word is exactly the inference this module exists to avoid."""
        assert seniority_from_title("Software Engineer") is None
        assert seniority_from_title("Mid-Level Software Engineer") is None


class TestVocabulary:
    def test_every_answer_is_a_value_the_column_can_hold(self) -> None:
        """The return type has to stay inside `Seniority`, because this value
        goes into a promoted column that `WHERE seniority = ANY(...)` reads and
        that the feed offers as a fixed dropdown. A sixth value would be
        silently unfilterable."""
        from typing import get_args

        from app.extraction import Seniority

        allowed = set(get_args(Seniority))
        titles = [
            "Staff Engineer",
            "Principal Engineer",
            "Senior Engineer",
            "Junior Engineer",
            "Software Engineer",
        ]
        for title in titles:
            answer = seniority_from_title(title)
            assert answer is None or answer in allowed
