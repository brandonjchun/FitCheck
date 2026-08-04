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

# Extraction versions, stamped at extraction time and bumped whenever a
# prompt or schema change makes an older extraction non-comparable to a new
# one.
#
# They exist because a content hash cannot see a prompt change. The gate in
# spec section 6.7 skips re-extraction when the source text is unchanged, so
# without a version beside it, improving a prompt would silently leave every
# stored row on the old behaviour with no way to find them. With one, "what
# needs re-extracting" is a query:
#
#     WHERE extraction_version < PROFILE_EXTRACTION_VERSION
#
# **Two counters, because there are two pipelines.** Resumes and job postings
# are extracted by different prompts against different schemas, and they will
# change on different days. One shared constant would mean bumping the resume
# prompt marks every posting in the catalog stale -- at M8 that is a
# re-extraction of the whole catalog at full LLM cost, for a change that had
# nothing to do with postings. The reverse is equally wrong.
#
# Note what does NOT belong in either: skill-alias changes. Normalization is
# applied when reading (see models.Profile.skills), not when writing, so
# growing the alias map changes what scoring sees without re-running anything.
# Only a change to what the *model* is asked to produce bumps a version.
#
# Resume extraction (ExtractedProfile, below):
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
PROFILE_EXTRACTION_VERSION = 3

# Job-posting extraction (ExtractedPosting, below).
#
#   1 -- initial JD prompt and schema, M7.
#   2 -- give the prompt the board's own title, make `title` a required field,
#        and tell it to read the level off the title. Version 1 was shown the
#        posting body and nothing else, so `seniority` was reconstructed from
#        prose that usually never restates the level -- 34 of the catalog's
#        "Staff ..." postings came back "unknown" and 15 came back "junior".
#        `title` was optional-with-a-default, which is the SkillItem.source
#        mistake again: the model was free to omit it, and did.
#
#        Note this bump is what re-extracts the catalog; the deterministic
#        override in app.seniority repairs `seniority` without waiting for it.
#
# The design question recorded here at M5 has been answered: postings get
# their own schema rather than reusing ExtractedProfile. The spec suggests
# sharing one, and that is wrong for a reason that only shows up in the
# scorer. A resume says "4 years of Python"; a job says "Python required,
# 3+ years". Those are not the same field with different values -- one is a
# possession and the other is a threshold, and section 8.3's overlap formula
# reads `required` and `preferred` directly to weight them. Forced through
# one schema, either every posting carries dead resume fields (education,
# years-per-skill as a possession) or the necessity distinction has to be
# recovered by parsing prose at scoring time, which is the regex resume
# parsing that section 8.1 exists to forbid.
POSTING_EXTRACTION_VERSION = 2


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


# --- Job postings -------------------------------------------------------
#
# The posting half of the contract. Everything below describes what a job
# *demands*, where the resume models above describe what a candidate *has*.

# Whether a posting treats a skill as a gate or a bonus.
#
# This is the field the whole scorer turns on. Section 8.3 weights required
# above preferred, and the reason is that they answer different questions: a
# missing required skill is usually disqualifying, a missing preferred one is
# a rounding error. Scoring them alike is what produces a ranked feed full of
# roles the candidate cannot actually get.
#
# "unknown" exists for the genuinely ambiguous case -- plenty of postings
# list technologies in a paragraph with no signal either way -- and follows
# the same reasoning as SkillItem.source: a required field with an explicit
# escape hatch, never an optional one with a default.
Necessity = Literal["required", "preferred", "unknown"]

RemoteType = Literal["remote", "hybrid", "onsite", "unknown"]


class PostingSkill(BaseModel):
    """One skill a posting asks for, and how hard it asks."""

    name: str = Field(
        description="The skill as written in the posting, e.g. 'PostgreSQL'."
    )
    necessity: Necessity = Field(
        description=(
            "'required' if the posting treats this as a must-have (listed under "
            "requirements, or worded as 'must have' / 'X+ years of'); "
            "'preferred' if it is a nice-to-have (worded as 'preferred', "
            "'bonus', 'a plus', 'desirable'); 'unknown' if the posting gives no "
            "signal either way."
        )
    )
    min_years: float | None = Field(
        default=None,
        description=(
            "Minimum years this posting asks for in this specific skill, if it "
            "states one. Null when the posting does not say. Do not infer a "
            "number from the seniority of the role."
        ),
    )
    evidence: str | None = Field(
        default=None,
        description=(
            "The phrase from the posting supporting this requirement. Quote it "
            "verbatim; do not paraphrase."
        ),
    )


class ExtractedPosting(BaseModel):
    """Everything we derive from a job posting's text.

    Lands in job_postings.extracted (JSONB), with title, company, location,
    remote_type, seniority, and min_years promoted to real columns -- the
    same hybrid as profiles, and the columns already exist for it.

    `min_years_experience` is the role-level threshold ("3+ years of backend
    engineering") and is separate from any per-skill `min_years`. A posting
    can state either, both, or neither, and they mean different things: one
    gates the candidate, the other gates a single skill. Collapsing them
    would make "5 years of Kubernetes" read as five years of total career.
    """

    # Required with null as the escape hatch, not optional with a default --
    # the same shape as SkillItem.source and for the same measured reason. A
    # Pydantic default drops the field from the schema's `required` list, and a
    # field a model is allowed to omit is a field it will omit: every posting
    # reached by a bare URL rendered as "Untitled posting" in the feed.
    #
    # Board postings do not depend on this, because the board states the title
    # and `_prepare_posting` prefers it. This is the answer for the postings
    # that have no board behind them.
    title: str | None = Field(
        description=(
            "The job title. When a KNOWN TITLE is supplied with the posting "
            "text, return it verbatim. Otherwise take it from the posting's "
            "own heading. Null only if the text states no title at all."
        )
    )
    company: str | None = None
    location: str | None = None
    remote_type: RemoteType
    seniority: Seniority
    min_years_experience: float | None = None
    skills: list[PostingSkill]
