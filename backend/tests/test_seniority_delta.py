"""Comparing a candidate's level against a role's.

The other half of `app/seniority.py`. `test_seniority.py` covers reading a level
off a title; this covers what happens once both sides have one.

Split into its own file because the two halves fail differently. Title parsing
is a regex problem and its tests are a corpus of real titles. This is a
semantics problem, and almost every test below is about refusing to answer:
"unknown" must not become a rung, a missing level must not become a match, and
a years gap must not invent a direction. The bug this guards against is not a
wrong number, it is a confident number derived from an absence.
"""

from decimal import Decimal

import pytest

from app.seniority import rank_of, seniority_delta


class TestLadder:
    """`rank_of` decides whether a delta is computable at all, so what it
    refuses to rank matters as much as what it ranks."""

    @pytest.mark.parametrize(
        ("level", "expected"),
        [("junior", 0), ("mid", 1), ("senior", 2), ("staff", 3)],
    )
    def test_each_rung_has_its_place(self, level: str, expected: int) -> None:
        assert rank_of(level) == expected

    def test_the_ladder_is_ordered_weakest_first(self) -> None:
        """The sign of `steps` means nothing if the rungs are not monotone."""
        ranks = [rank_of(level) for level in ("junior", "mid", "senior", "staff")]
        assert ranks == sorted(ranks)

    @pytest.mark.parametrize("level", ["unknown", None])
    def test_an_absent_level_is_not_a_rung(self, level: str | None) -> None:
        """`unknown` is a real value in both columns and it means what a NULL
        means here. Ranking it -- anywhere, including above staff -- would let
        an absence of evidence produce a confident delta."""
        assert rank_of(level) is None

    @pytest.mark.parametrize("level", ["principal", "lead", "Senior", "SENIOR", ""])
    def test_anything_outside_the_vocabulary_does_not_rank(self, level: str) -> None:
        """Exact match against the `Seniority` literal, deliberately. Both
        columns are written from that vocabulary, so a value outside it is
        corrupt data -- and answering None makes the delta read `unknown`
        rather than quietly placing the row on a rung it never claimed.

        Note `principal` in particular: `seniority_from_title` collapses it to
        `staff` on write, so it should never reach a column. If it does, that
        is a bug to surface, not to paper over with a lenient lookup.
        """
        assert rank_of(level) is None


class TestDirection:
    """The field a UI branches on."""

    def test_the_same_rung_is_a_match(self) -> None:
        assert seniority_delta("senior", "senior").direction == "match"

    @pytest.mark.parametrize(
        ("profile", "posting"),
        [("junior", "mid"), ("mid", "staff"), ("senior", "staff"), ("junior", "staff")],
    )
    def test_a_role_above_the_candidate_is_under(
        self, profile: str, posting: str
    ) -> None:
        assert seniority_delta(profile, posting).direction == "under"

    @pytest.mark.parametrize(
        ("profile", "posting"),
        [("mid", "junior"), ("staff", "mid"), ("staff", "junior"), ("senior", "mid")],
    )
    def test_a_role_below_the_candidate_is_over(
        self, profile: str, posting: str
    ) -> None:
        assert seniority_delta(profile, posting).direction == "over"

    @pytest.mark.parametrize(
        ("profile", "posting"),
        [
            ("senior", None),
            (None, "senior"),
            (None, None),
            ("senior", "unknown"),
            ("unknown", "senior"),
            ("unknown", "unknown"),
        ],
    )
    def test_either_side_missing_makes_it_unknown(
        self, profile: str | None, posting: str | None
    ) -> None:
        assert seniority_delta(profile, posting).direction == "unknown"

    def test_direction_is_never_none(self) -> None:
        """It is the one field a client can always branch on, which is why it
        is carried rather than derived from the sign of `steps`: 0 and None
        both fail a `steps < 0` test, for opposite reasons."""
        assert seniority_delta(None, None).direction == "unknown"


class TestSteps:
    def test_steps_are_signed_from_the_candidates_side(self) -> None:
        """Negative means the role sits above them, matching `years_gap`, which
        is also negative when they fall short. Two gap figures disagreeing on
        which direction was bad would be read wrong by somebody eventually."""
        assert seniority_delta("mid", "staff").steps == -2
        assert seniority_delta("staff", "mid").steps == 2

    def test_a_match_is_zero_not_none(self) -> None:
        assert seniority_delta("senior", "senior").steps == 0

    def test_steps_span_the_whole_ladder(self) -> None:
        assert seniority_delta("staff", "junior").steps == 3
        assert seniority_delta("junior", "staff").steps == -3

    @pytest.mark.parametrize("posting", [None, "unknown"])
    def test_steps_is_none_when_the_delta_is_unknown(self, posting: str | None) -> None:
        """None rather than 0. Zero would be indistinguishable from a genuine
        match, which is the reading that turns "we could not tell" into "this
        role is pitched exactly at you"."""
        assert seniority_delta("senior", posting).steps is None


class TestYearsGap:
    def test_the_gap_is_candidate_minus_required(self) -> None:
        delta = seniority_delta("senior", "senior", candidate_years=6, required_years=4)
        assert delta.years_gap == 2.0

    def test_falling_short_is_negative(self) -> None:
        delta = seniority_delta("mid", "senior", candidate_years=2, required_years=5)
        assert delta.years_gap == -3.0

    def test_decimal_years_do_not_raise(self) -> None:
        """Both columns are Numeric(4,1), so SQLAlchemy hands back Decimal --
        and `Decimal("5.0") - 3.0` raises TypeError. This is the case that
        reaches the coercion: a posting *and* a profile that both state
        years."""
        delta = seniority_delta(
            "senior",
            "senior",
            candidate_years=Decimal("5.5"),
            required_years=Decimal("3.0"),
        )

        assert delta.years_gap == pytest.approx(2.5)
        assert isinstance(delta.years_gap, float)

    def test_years_are_floats_not_decimals(self) -> None:
        """They go straight into JSONB, and `json.dumps` refuses a Decimal.
        Coercing at the boundary beats discovering it at the write."""
        delta = seniority_delta(
            "senior",
            "senior",
            candidate_years=Decimal("5.5"),
            required_years=Decimal("3.0"),
        )

        assert isinstance(delta.candidate_years, float)
        assert isinstance(delta.required_years, float)

    @pytest.mark.parametrize(
        ("candidate", "required"),
        [(None, 5), (5, None), (None, None)],
    )
    def test_one_side_missing_leaves_no_gap(self, candidate, required) -> None:
        """A threshold with nobody measured against it, or a candidate with no
        threshold to meet, is not a gap of zero."""
        delta = seniority_delta(
            "senior", "senior", candidate_years=candidate, required_years=required
        )
        assert delta.years_gap is None

    def test_zero_years_required_is_not_the_same_as_unstated(self) -> None:
        """The distinction `retrieval._where_clauses` already draws for
        `min_years`: an unstated requirement is no evidence of a barrier,
        whereas a stated zero is a real threshold that everybody clears."""
        stated = seniority_delta("mid", "mid", candidate_years=3, required_years=0)
        unstated = seniority_delta("mid", "mid", candidate_years=3)

        assert stated.years_gap == 3.0
        assert unstated.years_gap is None


class TestTheTwoHalvesAreIndependent:
    """A posting can state a level, a years threshold, both, or neither.

    Nothing here infers one from the other. That inference is what
    `seniority_from_title` exists to avoid doing from prose, and doing it here
    would reintroduce it one layer down.
    """

    def test_years_are_reported_without_any_levels(self) -> None:
        delta = seniority_delta(None, None, candidate_years=8, required_years=3)

        assert delta.direction == "unknown"
        assert delta.steps is None
        assert delta.years_gap == 5.0

    def test_levels_are_reported_without_any_years(self) -> None:
        delta = seniority_delta("staff", "senior")

        assert delta.direction == "over"
        assert delta.steps == 1
        assert delta.years_gap is None

    def test_a_years_gap_does_not_invent_a_direction(self) -> None:
        """Eight years against a three-year threshold does not make somebody
        `over` a role whose level nobody stated."""
        delta = seniority_delta(None, None, candidate_years=8, required_years=3)
        assert delta.direction == "unknown"

    def test_a_level_gap_does_not_invent_years(self) -> None:
        assert seniority_delta("junior", "staff").candidate_years is None


class TestBothLevelsAreAlwaysCarried:
    """"The posting never stated a level" and "we could not read yours" call
    for different copy, and only these fields distinguish them."""

    def test_the_levels_survive_an_unknown_direction(self) -> None:
        delta = seniority_delta("senior", None)

        assert delta.direction == "unknown"
        assert delta.profile_level == "senior"
        assert delta.posting_level is None

    def test_unknown_is_carried_verbatim_rather_than_nulled(self) -> None:
        """It is what the extraction actually returned, and flattening it to
        None would lose the difference between a posting that was read and
        could not be placed and one that was never extracted at all."""
        delta = seniority_delta("unknown", "staff")

        assert delta.profile_level == "unknown"
        assert delta.posting_level == "staff"
