"""Scoring one profile against one posting.

Pure Python over data that has already been extracted. No network call, no
model inference, no database access -- which is not an accident of the
implementation but the property the whole M9 design rests on: stage one
narrows 10,000 postings to ~200 with an indexed vector search, and stage two
runs *this* over those 200. The asymmetry only works if this half is cheap.

**Two scores, deliberately not one.**

`semantic` is cosine similarity between embeddings. It captures theme -- that
a backend posting and a backend resume are talking about the same kind of
work -- and it is blind to hard requirements. A resume can sit very close to
a job in embedding space while missing the single mandatory skill, because
"missing Kubernetes" moves a 384-dimensional vector almost not at all.

`skill` is explicit set overlap over required skills. It is the half that can
say no.

Blended 0.6/0.4 in favour of semantic similarity. That weighting is a
judgment call and is labelled as one -- there is no labelled data here to
derive it from, and claiming otherwise would be the kind of unearned
precision this project avoids. What makes it defensible is that both halves
and the full breakdown are always surfaced, so the number is inspectable
rather than oracular.

**Three buckets, not two.** matched / missing / partial, where partial means
the candidate has the skill but not the years the posting asks for. Two
buckets would force a 4-years-of-5-required candidate into either "matched",
overstating the fit, or "missing", which reads as though they have never
touched it. The partial bucket is also the one a human finds most useful,
because it is the set of gaps that are arguably negotiable.
"""

from dataclasses import dataclass, field
from typing import Literal

from app.skills import canonical_key

# Which generation of scoring rules produced a stored score.
#
#   1 -- initial: weighted required/preferred overlap, 0.4/0.6 blend, MiniLM
#        embeddings at 384 dimensions.
#   2 -- blend inverted to 0.6/0.4 in favour of semantic. Overlap rules,
#        bucket credits, and the embedding model are unchanged; only the
#        weights moved, which is enough to make a v1 score incomparable.
#
# Bumped whenever the blend weights, the bucket credits, or the embedding
# model change -- anything that makes an old score incomparable to a new one.
# Without it a feed sorted across two generations is silently wrong: position
# 3 beats position 4 because it was scored under different rules, not because
# it fits better, and nothing about the output says so.
#
# Note what does *not* bump it: adding a skill alias. Normalization runs on
# read, so the next score picks it up with no re-run, and the stored ones
# stay comparable because the change is monotone -- the same reasoning that
# keeps alias edits out of the extraction versions.
SCORER_VERSION = 2

# Weights for the blend. Named rather than inlined so the number appearing in
# a stored score can be traced to a decision.
#
# Inverted from the original 0.4/0.6 at v2. The first weighting reasoned that
# hard requirements outrank thematic similarity because a recruiter rejects on
# a missing must-have. That is still true of a *hiring* decision, but this
# blend does not make one -- it ranks a discovery feed, and the failure modes
# there are not symmetric. Skill overlap is only as good as the extraction
# behind it: a posting that lists its requirements loosely, or an extractor
# that misses one, produces a confidently low skill score for a role that fits
# fine. Semantic similarity degrades more gracefully under the same noise.
# Leaning on the sturdier signal for ranking, while the skill breakdown stays
# fully surfaced as the thing that explains *why* a posting placed where it
# did, is the trade this version makes.
#
# Still a judgment call, and still the coefficient a learning-to-rank model
# would fit from labelled `match_feedback` -- which is why that table is
# collected before anything reads it.
SEMANTIC_WEIGHT = 0.6
SKILL_WEIGHT = 0.4

# How much a preferred skill counts against a required one.
#
# Not zero: a posting listing eight preferred skills is telling you something
# real about the role, and ignoring them entirely makes two candidates who
# differ only in the nice-to-haves score identically. Not one either, or
# `necessity` would be decorative. A third means roughly three preferred
# skills trade for one required.
REQUIRED_WEIGHT = 1.0
PREFERRED_WEIGHT = 1.0 / 3.0

# An "unknown" necessity is weighted as preferred rather than required.
#
# The asymmetry is deliberate. Treating an ambiguous mention as required
# invents a gate the posting never stated, and a false *gate* pushes a
# candidate below roles they could actually get -- invisible to them, because
# what they see is an absence. Treating it as preferred at worst slightly
# understates a real requirement, which surfaces as a rank they can inspect.
UNKNOWN_WEIGHT = PREFERRED_WEIGHT

Bucket = Literal["matched", "partial", "missing"]


@dataclass(frozen=True)
class SkillVerdict:
    """One posting requirement, judged against the candidate."""

    name: str
    necessity: str
    bucket: Bucket
    required_years: float | None = None
    candidate_years: float | None = None
    evidence: str | None = None

    @property
    def weight(self) -> float:
        if self.necessity == "required":
            return REQUIRED_WEIGHT
        if self.necessity == "preferred":
            return PREFERRED_WEIGHT
        return UNKNOWN_WEIGHT


@dataclass(frozen=True)
class SkillBreakdown:
    """The full accounting behind a skill score.

    Kept as structure rather than reduced to a number on the way out. The
    number is what ranks; this is what makes the ranking arguable, and spec
    section 8.4 requires it to be surfaced.
    """

    score: float
    verdicts: list[SkillVerdict] = field(default_factory=list)

    def _in(self, bucket: Bucket) -> list[SkillVerdict]:
        return [v for v in self.verdicts if v.bucket == bucket]

    @property
    def matched(self) -> list[SkillVerdict]:
        return self._in("matched")

    @property
    def partial(self) -> list[SkillVerdict]:
        return self._in("partial")

    @property
    def missing(self) -> list[SkillVerdict]:
        return self._in("missing")

    @property
    def missing_required(self) -> list[SkillVerdict]:
        """The gaps that actually disqualify, which is what a reader wants
        first. A missing preferred skill is noise next to these."""
        return [v for v in self.missing if v.necessity == "required"]


def _years_of(item: dict) -> float | None:
    value = item.get("years")
    return float(value) if isinstance(value, (int, float)) else None


def _required_years_of(item: dict) -> float | None:
    value = item.get("min_years")
    return float(value) if isinstance(value, (int, float)) else None


def score_skills(
    posting_skills: list[dict], profile_skills: list[dict]
) -> SkillBreakdown:
    """Judge a posting's requirements against a candidate's skills.

    Both lists are the *normalized* form -- JobPosting.skills and
    Profile.skills, which canonicalize names on read. Passing raw extraction
    blobs here would compare "JS" to "JavaScript" and report a match as a
    gap.

    Returns a breakdown whose `score` is the fraction of required weight the
    candidate covers, in [0, 1]:

        score = sum(weight(s) * credit(s) for s in posting) / sum(weight(s))

    where credit is 1.0 for matched, 0.5 for partial, 0.0 for missing.

    **Half credit for partial is a real choice.** Zero would make "4 of the 5
    years asked for" identical to never having touched the skill, which is
    plainly wrong and would rank a career-changer above a near-miss. Full
    credit would make the years requirement decorative. Half keeps the
    ordering right -- full match > near miss > nothing -- which is all a
    ranking needs from it.

    A posting with no requirements scores 1.0 rather than 0.0. There is
    nothing to fail to meet, and 0.0 would bury every vaguely-worded posting
    beneath every specific one for reasons that have nothing to do with fit.
    """
    # Keyed on the canonical key rather than the display name, which is a bug
    # fix rather than a refinement. `normalize_skill` returns an unrecognised
    # name unchanged, so before this a resume saying "Redis" and a posting
    # saying "redis" produced two different dict keys and scored as a *missing
    # required skill* -- and the same for "GraphQL" against "GraphQL API".
    # Only the ~30 names in the alias map were immune. The candidate was
    # silently penalised for the posting's capitalisation.
    #
    # The stored breakdown still carries the posting's own display name, so
    # nothing about the explanation changes; only which names count as equal.
    candidate_years: dict[str, float | None] = {
        canonical_key(item["name"]): _years_of(item) for item in profile_skills
    }

    verdicts: list[SkillVerdict] = []
    for item in posting_skills:
        name = item.get("name", "")
        if not name:
            continue

        necessity = item.get("necessity", "unknown")
        required = _required_years_of(item)
        key = canonical_key(name)
        held = candidate_years.get(key)

        if key not in candidate_years:
            bucket: Bucket = "missing"
        elif required is not None and (held is None or held < required):
            # Has the skill, short of the threshold -- or has it with no
            # stated duration against a posting that asks for one. The second
            # case is a partial rather than a match because the posting made
            # years a criterion and we cannot show it is met; calling it
            # matched would assert something the resume never said.
            bucket = "partial"
        else:
            bucket = "matched"

        verdicts.append(
            SkillVerdict(
                name=name,
                necessity=necessity,
                bucket=bucket,
                required_years=required,
                candidate_years=held,
                evidence=item.get("evidence"),
            )
        )

    total_weight = sum(v.weight for v in verdicts)
    if total_weight == 0:
        return SkillBreakdown(score=1.0, verdicts=verdicts)

    credit = {"matched": 1.0, "partial": 0.5, "missing": 0.0}
    earned = sum(v.weight * credit[v.bucket] for v in verdicts)

    return SkillBreakdown(score=earned / total_weight, verdicts=verdicts)


def blend(semantic: float, skill: float) -> float:
    """Combine the two sub-scores into the number that ranks.

    Clamped to [0, 1] because cosine similarity is mathematically in [-1, 1]:
    a genuinely opposite pair would otherwise drag the blend negative, and a
    negative match score is not a thing anyone can read. In practice MiniLM
    on English prose almost never returns a negative, so this is a guard
    rather than a routine correction.
    """
    combined = SEMANTIC_WEIGHT * semantic + SKILL_WEIGHT * skill
    return max(0.0, min(1.0, combined))


def build_breakdown(
    semantic: float,
    skill: SkillBreakdown,
    *,
    extraction_failed: bool = False,
) -> dict:
    """Assemble the JSONB explanation stored on a match.

    Extracted here rather than left inline in the scoring task because M9 has
    a second caller: the recommender reranks ~200 candidates per profile and
    has to produce byte-identical payloads to the ones Path A writes. Two
    copies of this dict would drift, and the drift would be invisible --
    both feeds would render, one would quietly be missing a field the UI
    reads with `.get`.

    The rounding is deliberate at six places. These are floats being written
    to JSONB and then compared in tests; full repr precision makes an
    equality assertion depend on the platform's last bit.
    """
    return {
        "semantic_score": round(semantic, 6),
        "skill_score": round(skill.score, 6),
        "final_score": round(blend(semantic, skill.score), 6),
        "skills": [
            {
                "name": v.name,
                "necessity": v.necessity,
                "bucket": v.bucket,
                "required_years": v.required_years,
                "candidate_years": v.candidate_years,
                "evidence": v.evidence,
            }
            for v in skill.verdicts
        ],
        "counts": {
            "matched": len(skill.matched),
            "partial": len(skill.partial),
            "missing": len(skill.missing),
            "missing_required": len(skill.missing_required),
        },
        # Stored in the row rather than read from the constants at display
        # time, so an old match still explains itself under the weights it
        # was actually scored with.
        "weights": {"semantic": SEMANTIC_WEIGHT, "skill": SKILL_WEIGHT},
        # True when the posting could not be extracted, so the skill half is
        # empty and the score is semantic-only. Without this the UI would
        # render a confident 0.4 with no skills listed and no way to tell
        # that from a genuine total mismatch.
        "extraction_failed": extraction_failed,
    }
