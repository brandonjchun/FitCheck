"""The structured shape we ask the LLM to produce from a resume.

These are a third kind of Pydantic model, distinct from the two we already
have:

    models.py      -> SQLAlchemy ORM. How data is stored.
    schemas.py     -> API contract. What clients see.
    extraction.py  -> LLM contract. What the model must return.

Keeping them separate matters for the same reason as before: the extraction
shape is driven by what an LLM can reliably produce, the storage shape by
what Postgres indexes well, and the API shape by what a client needs. They
move independently.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Structured outputs compile this class into a JSON Schema the model is
# constrained to follow, so the response is guaranteed parseable. That
# constraint cuts both ways -- a few JSON Schema features are unsupported:
# no recursive schemas, no numeric bounds (ge/le), no string length limits.
# The Python SDK strips those and validates them client-side instead, so
# they are not errors -- just not enforced by the model.

Seniority = Literal["junior", "mid", "senior", "staff", "unknown"]

# Stamped onto every profile at extraction time. Bump it whenever the prompt
# or the schema below changes in a way that makes an older extraction
# non-comparable to a new one.
#
# It exists because a content hash cannot see a prompt change. The gate in
# spec section 6.7 skips re-extraction when the source text is unchanged, and
# without a version alongside it, improving the prompt would silently leave
# every stored profile on the old behaviour with no way to find them. With
# it, "what needs re-extracting" is a query:
#
#     WHERE extraction_version < CURRENT_EXTRACTION_VERSION
#
# Note what does NOT belong here: skill-alias changes. Normalization is
# applied when reading (see models.Profile.skills), not when writing, so
# growing the alias map changes what scoring sees without re-running anything.
# Only a change to what the *model* is asked to produce bumps this.
#
#   1 -- initial prompt and schema.
#   2 -- instruct per-skill `years` to be inferred from the date range of the
#        role the skill appears in. Version 1 returned null for every skill,
#        which leaves the partial-match bucket (has the skill, insufficient
#        years) with nothing to work from at M7.
#   3 -- add `source` to SkillItem. Every skill carried verbatim evidence, but
#        nothing recorded whether that evidence was shipped work or a bare
#        keyword list -- a difference that is most of what a human reviewer
#        weighs. Added before M7 rather than after on purpose: once scoring
#        exists, a new skill field also invalidates every tuned blend weight
#        and every stored score, so the same change costs a scorer_version
#        bump and a full re-score on top of the re-extraction.
CURRENT_EXTRACTION_VERSION = 3


# Where in the resume a skill was found. Ordered strongest to weakest, which
# is also the precedence the prompt applies when a skill appears in more than
# one place -- and most do, since a keyword section usually repeats what the
# bullets already demonstrated.
#
# The distinction this captures is most of what a human reviewer does: "Go"
# in a technologies list is a claim, "built and shipped a service in Go" is
# evidence. M7 weights them differently; without this field the scorer cannot
# tell them apart.
SkillSource = Literal["experience", "project", "education", "skills_list", "unknown"]


class SkillItem(BaseModel):
    """One skill, with the evidence that justifies claiming it."""

    name: str = Field(description="The skill as written in the resume, e.g. 'JavaScript'.")
    years: float | None = Field(
        default=None,
        description="Years of experience with this skill, if the resume states or implies it.",
    )
    evidence: str | None = Field(
        default=None,
        description=(
            "The phrase from the resume supporting this skill. Quote it verbatim; "
            "do not paraphrase."
        ),
    )
    # Deliberately required, with "unknown" as the escape hatch rather than a
    # default. A Pydantic default would drop the field from the schema's
    # `required` list, letting the model omit it -- which is exactly how
    # version 1 ended up with `years: null` on 29 of 29 skills. Forcing a
    # choice and providing somewhere to put "I can't tell" is the same shape
    # `seniority` already uses.
    source: SkillSource = Field(
        description=(
            "Where in the resume this skill appears: 'experience' for a bullet "
            "under a job, 'project' for a personal or academic project, "
            "'education' for coursework or a degree, 'skills_list' for a bare "
            "skills or technologies section, 'unknown' if it cannot be placed."
        ),
    )


class EducationItem(BaseModel):
    """One degree or credential."""

    institution: str
    degree: str | None = None
    field_of_study: str | None = None
    graduation_year: int | None = None


class ExtractedProfile(BaseModel):
    """Everything we derive from a resume's raw text.

    This is what lands in profiles.extracted (JSONB), with seniority and
    total_years_experience promoted to real columns for filtering -- the
    hybrid from spec section 3.3.
    """

    skills: list[SkillItem]
    total_years_experience: float | None = None
    seniority: Seniority
    education: list[EducationItem]
