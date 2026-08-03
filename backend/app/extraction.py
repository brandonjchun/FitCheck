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
