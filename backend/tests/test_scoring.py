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

import json
from decimal import Decimal

import pytest

from app.scoring import (
    PREFERRED_WEIGHT,
    REQUIRED_WEIGHT,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
    blend,
    build_breakdown,
    score_skills,
)
from app.seniority import seniority_delta


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


class TestSpellingDoesNotDecideAMatch:
    """A posting's capitalisation must not cost a candidate the role.

    This was a live bug. `normalize_skill` returns an unrecognised name
    unchanged, so only the ~30 names in the alias map were ever compared
    reliably; for everything else a resume saying "Redis" against a posting
    asking for "redis" produced two different keys and scored as a *missing
    required skill*. Comparison now runs on `canonical_key`.
    """

    @pytest.mark.parametrize(
        "posting_name, resume_name",
        [
            ("redis", "Redis"),
            ("Redis", "redis"),
            ("GraphQL API", "GraphQL"),
            ("graphql-api", "graphql"),
            ("Distributed systems", "Distributed Systems"),
            ("Machine Learning Frameworks", "Machine Learning"),
        ],
    )
    def test_variants_of_one_skill_count_as_held(self, posting_name, resume_name) -> None:
        breakdown = score_skills([need(posting_name)], [has(resume_name)])

        assert breakdown.score == 1.0
        assert [v.bucket for v in breakdown.verdicts] == ["matched"]

    def test_the_posting_keeps_its_own_wording_in_the_breakdown(self) -> None:
        """Only which names count as equal changed. The explanation still
        quotes the posting, because that is what makes it auditable."""
        breakdown = score_skills([need("GraphQL API")], [has("graphql")])

        assert breakdown.verdicts[0].name == "GraphQL API"

    def test_a_genuinely_absent_skill_is_still_missing(self) -> None:
        """The guard against over-merging: this must not turn into a scorer
        that matches everything."""
        breakdown = score_skills([need("Kubernetes")], [has("Redis")])

        assert breakdown.score == 0.0
        assert [v.bucket for v in breakdown.verdicts] == ["missing"]

    def test_a_sub_technology_does_not_satisfy_its_parent(self) -> None:
        """"React" on a resume must not answer a posting asking for "React
        Native" -- they are different competencies."""
        assert score_skills([need("React Native")], [has("React")]).score == 0.0

    def test_years_still_apply_across_a_variant(self) -> None:
        """The merge must not lose the partial bucket on the way."""
        breakdown = score_skills(
            [need("GraphQL API", min_years=5)], [has("graphql", years=2)]
        )

        assert [v.bucket for v in breakdown.verdicts] == ["partial"]


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


class TestSeniorityInTheBreakdown:
    """The level gap travels in the stored explanation.

    `test_seniority_delta.py` covers what the delta *says*; this covers the
    two promises made by putting it here rather than in the blend.
    """

    def test_the_delta_is_carried_in_full(self) -> None:
        """All seven fields, because the UI branches on `direction` but needs
        the levels to name which side was missing and the years to say by how
        much."""
        payload = build_breakdown(
            0.5,
            score_skills([], []),
            seniority=seniority_delta(
                "mid", "staff", candidate_years=3, required_years=8
            ),
        )

        assert payload["seniority"] == {
            "profile_level": "mid",
            "posting_level": "staff",
            "steps": -2,
            "direction": "under",
            "candidate_years": 3.0,
            "required_years": 8.0,
            "years_gap": -5.0,
        }

    def test_the_score_does_not_move(self) -> None:
        """The claim that justifies not bumping SCORER_VERSION.

        The version tracks whether two stored scores are comparable. Nothing in
        the delta reaches `blend`, so a match scored with it is comparable to
        one scored without -- and this is the test that fails if somebody later
        folds seniority into the blend without bumping the version, which would
        silently make position 3 beat position 4 for reasons the feed cannot
        explain.
        """
        skill = score_skills([need("Python")], [has("Python")])

        without = build_breakdown(0.7, skill)
        with_delta = build_breakdown(
            0.7,
            skill,
            seniority=seniority_delta(
                "junior", "staff", candidate_years=1, required_years=10
            ),
        )

        for key in ("semantic_score", "skill_score", "final_score"):
            assert without[key] == with_delta[key]

    def test_a_decimal_delta_survives_json(self) -> None:
        """The payload is written to JSONB, and `json.dumps` refuses a Decimal.
        Both year columns are Numeric(4,1), so this is the shape that actually
        reaches the write -- not a hypothetical."""
        payload = build_breakdown(
            0.5,
            score_skills([], []),
            seniority=seniority_delta(
                "senior",
                "staff",
                candidate_years=Decimal("4.5"),
                required_years=Decimal("7.0"),
            ),
        )

        assert json.loads(json.dumps(payload))["seniority"]["years_gap"] == -2.5

    def test_omitting_it_stores_null_rather_than_a_fabricated_gap(self) -> None:
        """A caller with nothing to compare gets None. That is distinct from a
        delta whose direction is "unknown": the first says nobody asked, the
        second says we asked and the data could not answer."""
        assert build_breakdown(0.5, score_skills([], []))["seniority"] is None

    def test_an_unknown_delta_is_stored_rather_than_dropped(self) -> None:
        """The opposite case, and the reason the two are not collapsed. The
        comparison ran and failed, which is a thing worth telling the user --
        and the levels explain why it failed."""
        payload = build_breakdown(
            0.5,
            score_skills([], []),
            seniority=seniority_delta("senior", None),
        )

        assert payload["seniority"]["direction"] == "unknown"
        assert payload["seniority"]["profile_level"] == "senior"
        assert payload["seniority"]["posting_level"] is None

    def test_the_key_is_always_present(self) -> None:
        """Present-and-null, not absent. `_to_response` reads it with `.get`
        either way, but a key that appears only sometimes makes every consumer
        downstream guess which generation of row it is holding."""
        assert "seniority" in build_breakdown(0.5, score_skills([], []))
