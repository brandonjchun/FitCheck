"""The scorer: what the number means, and what it refuses to claim.

Pure functions over already-extracted data, so these tests need no database,
no network, and no model. That is the same property M9's two-stage retrieval
depends on -- if scoring were expensive, reranking 200 candidates per profile
would not be viable and the whole design would have to change.

Written as claims about ranking rather than as assertions on constants. A
test that pins `score == 0.6` breaks the moment anyone tunes a weight, which
trains people to update the number without reading why it moved. A test that
says "a candidate meeting every requirement outranks one missing a mandatory
skill" survives tuning and fails on the thing that would actually be wrong.
"""

import pytest

from app.scoring import (
    PREFERRED_WEIGHT,
    REQUIRED_WEIGHT,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
    blend,
    score_skills,
)


def need(name, necessity="required", min_years=None, evidence="quoted"):
    return {
        "name": name,
        "necessity": necessity,
        "min_years": min_years,
        "evidence": evidence,
    }


def has(name, years=None):
    return {"name": name, "years": years, "source": "experience"}


class TestBuckets:
    def test_present_skill_matches(self) -> None:
        result = score_skills([need("Python")], [has("Python")])

        assert [v.bucket for v in result.verdicts] == ["matched"]

    def test_absent_skill_is_missing(self) -> None:
        result = score_skills([need("Kubernetes")], [has("Python")])

        assert [v.bucket for v in result.verdicts] == ["missing"]

    def test_short_on_years_is_partial(self) -> None:
        """The bucket that makes the output feel like judgement rather than a
        set difference. Four years against a five-year ask is a real
        candidate, and calling it 'missing' would say they have never touched
        the skill."""
        result = score_skills([need("Go", min_years=5)], [has("Go", years=4)])

        assert result.verdicts[0].bucket == "partial"

    def test_meeting_the_threshold_matches(self) -> None:
        result = score_skills([need("Go", min_years=3)], [has("Go", years=3)])

        assert result.verdicts[0].bucket == "matched"

    def test_unstated_years_against_a_threshold_is_partial(self) -> None:
        """The posting made years a criterion and the resume does not say.

        Calling this `matched` would assert something the resume never
        claimed; calling it `missing` would ignore a skill they demonstrably
        have. Partial is the only honest answer, and it is the common case --
        most resumes date roles, not skills.
        """
        result = score_skills([need("Rust", min_years=2)], [has("Rust", years=None)])

        assert result.verdicts[0].bucket == "partial"

    def test_no_threshold_means_years_are_irrelevant(self) -> None:
        result = score_skills([need("Rust")], [has("Rust", years=None)])

        assert result.verdicts[0].bucket == "matched"


class TestRankingProperties:
    """Stated as orderings, so tuning a weight cannot silently invert them."""

    def test_full_beats_partial_beats_missing(self) -> None:
        requirement = [need("Python", min_years=3)]

        full = score_skills(requirement, [has("Python", 5)]).score
        near = score_skills(requirement, [has("Python", 1)]).score
        none = score_skills(requirement, [has("Go", 9)]).score

        assert full > near > none

    def test_a_missing_required_costs_more_than_a_missing_preferred(self) -> None:
        """If this inverts, `necessity` has stopped meaning anything and the
        feed will rank roles the candidate cannot get above ones they can."""
        posting = [need("Python"), need("Go", necessity="preferred")]

        lost_required = score_skills(posting, [has("Go")]).score
        lost_preferred = score_skills(posting, [has("Python")]).score

        assert lost_preferred > lost_required

    def test_preferred_skills_still_count(self) -> None:
        """Not zero-weighted. Two candidates differing only in nice-to-haves
        should not score identically -- the posting listed them for a
        reason."""
        posting = [need("Python"), need("Go", necessity="preferred")]

        with_bonus = score_skills(posting, [has("Python"), has("Go")]).score
        without = score_skills(posting, [has("Python")]).score

        assert with_bonus > without

    def test_unknown_necessity_is_weighted_as_preferred(self) -> None:
        """Deliberately the lenient direction. Treating an ambiguous mention
        as required invents a gate the posting never stated, which pushes a
        candidate below roles they could actually get -- and they see only an
        absence, with nothing to inspect."""
        assert (
            score_skills([need("X", necessity="unknown")], []).score
            == score_skills([need("X", necessity="preferred")], []).score
        )

    def test_required_outweighs_preferred_in_the_constants(self) -> None:
        assert REQUIRED_WEIGHT > PREFERRED_WEIGHT > 0


class TestScoreRange:
    def test_meeting_everything_is_one(self) -> None:
        posting = [need("Python"), need("Go", necessity="preferred")]

        assert score_skills(posting, [has("Python"), has("Go")]).score == 1.0

    def test_meeting_nothing_is_zero(self) -> None:
        assert score_skills([need("Python")], [has("Rust")]).score == 0.0

    def test_a_posting_with_no_requirements_scores_one(self) -> None:
        """There is nothing to fail to meet. Zero would bury every
        vaguely-worded posting beneath every specific one for reasons
        unrelated to fit."""
        assert score_skills([], [has("Python")]).score == 1.0

    def test_a_candidate_with_no_skills_still_scores(self) -> None:
        """Zero, not a crash. An un-extracted profile is a real state."""
        assert score_skills([need("Python")], []).score == 0.0

    @pytest.mark.parametrize("years", [0, 1, 100])
    def test_score_stays_in_range(self, years: int) -> None:
        result = score_skills(
            [need("A", min_years=5), need("B", necessity="preferred")],
            [has("A", years), has("B")],
        )

        assert 0.0 <= result.score <= 1.0


class TestBreakdown:
    def test_separates_missing_required_from_missing_preferred(self) -> None:
        """The only count that decides anything. Four missing nice-to-haves
        is a fine applicant; one missing must-have usually is not."""
        posting = [need("Python"), need("Go", necessity="preferred")]

        result = score_skills(posting, [])

        assert len(result.missing) == 2
        assert [v.name for v in result.missing_required] == ["Python"]

    def test_carries_the_posting_evidence_through(self) -> None:
        """The quote is what makes a verdict auditable -- a user can see the
        phrase the score reacted to instead of being handed a number."""
        posting = [need("Python", evidence="5+ years of Python required")]

        result = score_skills(posting, [])

        assert result.verdicts[0].evidence == "5+ years of Python required"

    def test_records_both_sides_of_a_years_gap(self) -> None:
        result = score_skills([need("Go", min_years=5)], [has("Go", 2)])

        assert result.verdicts[0].required_years == 5
        assert result.verdicts[0].candidate_years == 2

    def test_skills_with_no_name_are_dropped(self) -> None:
        """An unnamed requirement cannot be matched, so counting it in the
        denominator would penalise every candidate for a defective
        extraction."""
        assert score_skills([need(""), need("Python")], [has("Python")]).score == 1.0


class TestNormalizationAssumption:
    def test_comparison_is_exact_on_names(self) -> None:
        """Documents the contract rather than the implementation.

        This function does NOT normalize -- callers pass Profile.skills and
        JobPosting.skills, which already did. If someone passes raw
        extraction blobs instead, "JS" and "JavaScript" read as unrelated and
        every such pair is reported as a gap. The test exists so that
        behaviour is a stated assumption rather than a surprise.
        """
        assert score_skills([need("JavaScript")], [has("JS")]).score == 0.0


class TestBlend:
    def test_weights_sum_to_one(self) -> None:
        assert SEMANTIC_WEIGHT + SKILL_WEIGHT == pytest.approx(1.0)

    def test_semantic_is_weighted_above_skill(self) -> None:
        """Inverted at scorer v2. Skill overlap is the sharper signal but the
        more brittle one -- it is only ever as good as the extraction under
        it, and a loosely-worded posting or one missed requirement produces a
        confidently low score for a role that fits. Semantic similarity
        degrades more gently under the same noise, which is what a ranking
        wants. The skill breakdown still explains the placement.
        """
        assert SEMANTIC_WEIGHT > SKILL_WEIGHT

    def test_a_missing_requirement_outweighs_a_close_read(self) -> None:
        """Survives the v2 inversion, and only just -- which is the point.

        An embedding barely moves when one skill is absent, so a resume can
        sit very close to a job it is not qualified for. At 0.4/0.6 this
        ordering held by a wide margin. At 0.6/0.4 it holds by 0.01, and the
        assertion below is doing real work rather than restating arithmetic:
        weighting semantic any higher inverts it, and a thematically-close
        candidate missing every requirement would outrank a qualified one.

        That makes this the test that fails first if the weights drift again,
        which is exactly the tripwire wanted around a hand-set constant.
        """
        thematic_but_unqualified = blend(0.95, 0.0)
        offbeat_but_qualified = blend(0.30, 1.0)

        assert offbeat_but_qualified > thematic_but_unqualified

    @pytest.mark.parametrize(
        "semantic, skill", [(-1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (-0.5, 0.5)]
    )
    def test_result_is_always_a_readable_score(self, semantic, skill) -> None:
        """Cosine is mathematically in [-1, 1], and a negative match score is
        not a thing anyone can read."""
        assert 0.0 <= blend(semantic, skill) <= 1.0

    def test_perfect_on_both_is_one(self) -> None:
        assert blend(1.0, 1.0) == pytest.approx(1.0)
