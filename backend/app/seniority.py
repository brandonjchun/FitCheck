"""Reading a role's level off its title.

A companion to `skills.py`: both take a string the LLM had a hand in and
resolve it to the vocabulary the rest of the system compares against.

**Why this exists rather than trusting the extraction.** `seniority` is a
promoted column, so it is what `WHERE seniority = ANY(...)` reads and what the
M9 feed filter offers as a dropdown. Postings are extracted by a local 8B
model (see `config.llm_provider_posting` for why), and measured against the
live catalog it gets this field wrong in a way no prompt fixes:

    title contains "staff"    unknown 34   junior 15   senior 4   staff 3

Fifteen postings titled "Staff Engineer" were filed as *junior*. The result is
a filter that looks functional and returns almost nothing: 10 staff rows out
of 1,587, against a catalog where "Staff <X> Engineer" is one of the commonest
titles on the board.

The title is not a guess. It arrives from the board's own API as structured
data, and a role whose title says "Staff" is a staff role -- there is nothing
for a model to infer. So the deterministic answer wins where the title states
a level, and the extraction is consulted only for the titles that do not
(which is most non-engineering ladders, and every posting reached by a bare
user-submitted URL).

This is the same call spec section 9 item 3 makes for board metadata generally
-- prefer the API's structured fields over inference -- applied to a field the
LLM had been left to derive from prose.

**Applied on write, not on read.** The opposite of `skills.normalize_skill`,
and for a concrete reason: skills live in JSONB and are only ever displayed,
so canonicalizing them at read time makes an alias fix retroactive for free.
Seniority is filtered on in SQL, so a read-time property would be invisible to
the query that matters. It has to be in the column.
"""

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from app.extraction import Seniority

# Matched with word boundaries throughout, which is the difference between
# working and not. "International" contains "intern", and a substring match
# files every "Software Engineer, International" as an internship -- so
# `\bintern\b` (no boundary before "ational") is doing real work here, not
# decoration.
#
# Ordered strongest-first below, because titles stack: "Senior Staff Machine
# Learning Engineer" is a staff role that happens to contain the word senior,
# and checking senior first would demote it.

# "Distinguished" and "Principal" sit at or above staff on every ladder that
# uses them. They collapse into "staff" rather than getting their own bucket
# because the vocabulary is fixed by `Seniority` and the promoted column, and
# widening it is a schema decision with a frontend filter attached -- not
# something to smuggle in with a bug fix.
#
# Deliberately absent: "Fellow" and "Lead". Both are genuinely ambiguous in
# the live catalog rather than merely rare. "Research Fellow" is often a
# postdoc, and "Lead" is a scope word across whole non-engineering ladders --
# "Lead Counsel", "Growth Lead", "Agency Development Lead", "Detection &
# Response, Lead". Guessing on those trades one wrong bucket for another,
# where falling through to the extraction at least lets the posting's own
# prose decide.
_STAFF = re.compile(r"\b(?:staff|principal|distinguished)\b", re.IGNORECASE)

_SENIOR = re.compile(r"\b(?:senior|snr|sr)\b", re.IGNORECASE)

# No "associate". It reads as junior on a sales ladder and as senior on
# several others -- the catalog holds "Associate Director", "Associate Counsel"
# and "Global Associate Director, Experiential & Content Production", none of
# which are junior roles.
_JUNIOR = re.compile(
    r"\b(?:junior|jr|intern(?:s|ship|ships)?|apprentice|trainee"
    r"|new\s+grad(?:uate)?|entry[\s-]level|co-?op)\b",
    re.IGNORECASE,
)

# "Chief of Staff" is an executive-adjacent business role and has nothing to do
# with a staff engineer. Removed from the string before the level patterns run,
# rather than short-circuiting the whole function, so that a hypothetical
# "Senior Chief of Staff" still resolves on the word that does carry a level.
_NOT_A_LEVEL = re.compile(r"\bchief of staff\b", re.IGNORECASE)


def seniority_from_title(title: str | None) -> Seniority | None:
    """Return the level a title states outright, or None if it states none.

        seniority_from_title("Staff Software Engineer")        -> "staff"
        seniority_from_title("Senior Staff ML Engineer")       -> "staff"
        seniority_from_title("Staff+ Engineer")                -> "staff"
        seniority_from_title("Sr. Backend Engineer")           -> "senior"
        seniority_from_title("Software Engineer Internship")   -> "junior"
        seniority_from_title("Software Engineer")              -> None
        seniority_from_title("Software Engineer, International") -> None

    None rather than "unknown", and the distinction is the whole interface.
    "unknown" is an answer -- it is what the extraction returns when it has
    read the posting and cannot tell. This function returning None means it
    was not asked a question it can answer, so the caller should fall through
    to the extraction rather than overwrite it with a worse answer.

    There is deliberately no rule producing "mid". No title says "mid-level"
    in practice; an unqualified "Software Engineer" is the closest thing, and
    reading a level into the *absence* of a word is exactly the inference this
    module exists to avoid.
    """
    if not title:
        return None

    text = _NOT_A_LEVEL.sub(" ", title)

    if _STAFF.search(text):
        return "staff"
    if _SENIOR.search(text):
        return "senior"
    if _JUNIOR.search(text):
        return "junior"
    return None


# --------------------------------------------------------------------------
# Comparing a candidate's level against a role's
# --------------------------------------------------------------------------

# The ladder, weakest first.
#
# "unknown" is deliberately absent, and that is the same distinction
# `seniority_from_title` draws between None and "unknown" one level up. A rung
# is a claim about where a role sits; "unknown" is a statement that nobody
# could tell. Ranking it -- anywhere, including as a fifth rung above staff --
# would let an absence of evidence produce a confident delta, which is exactly
# the failure this module was written to stop.
_LADDER: tuple[Seniority, ...] = ("junior", "mid", "senior", "staff")

_RANK: dict[str, int] = {level: index for index, level in enumerate(_LADDER)}

# What the comparison concluded, from the *candidate's* point of view.
#
#   match    the candidate sits on the role's rung
#   under    the role is above them -- a stretch
#   over     the role is below them -- likely a step down
#   unknown  one side or both never stated a level
#
# A separate field rather than something the UI derives from the sign of
# `steps`, because "unknown" and "match" both leave `steps` unusable for that:
# one is None and the other is 0, and a client testing `steps > 0` reads them
# identically.
Direction = Literal["match", "under", "over", "unknown"]


@dataclass(frozen=True)
class SeniorityDelta:
    """The gap between a candidate's level and a posting's, as structure.

    Explanation, not score. Nothing here feeds the blend -- see the note in
    `scoring.build_breakdown` for why adding it does not bump SCORER_VERSION.

    Both levels are carried even when the delta is unknown, so the UI can say
    *which* side was missing rather than falling back to silence. "The posting
    never stated a level" and "we could not read yours" call for different
    copy, and only the fields distinguish them.
    """

    profile_level: str | None
    posting_level: str | None
    steps: int | None
    direction: Direction
    candidate_years: float | None
    required_years: float | None
    years_gap: float | None


def rank_of(level: str | None) -> int | None:
    """Where a level sits on the ladder, or None if it names no rung.

        rank_of("junior")   -> 0
        rank_of("staff")    -> 3
        rank_of("unknown")  -> None
        rank_of(None)       -> None

    "unknown" and None collapse to the same answer on purpose. Both columns
    this reads are `Text` holding a `Seniority`, so "unknown" is a value that
    genuinely appears in the database -- and it means the same thing as a NULL
    for every question a caller can ask here.
    """
    if level is None:
        return None
    return _RANK.get(level)


def _as_float(value: Decimal | float | int | None) -> float | None:
    """Coerce a years figure to a plain float.

    Load-bearing, not defensive. `profiles.years_experience` and
    `job_postings.min_years` are both `Numeric(4,1)`, so SQLAlchemy hands back
    `Decimal` -- and `Decimal` mixes with neither of the two things that happen
    to these numbers next. `Decimal("5.0") - 3.0` raises TypeError, and
    `json.dumps(Decimal("5.0"))` raises TypeError too, so an uncoerced value
    would break the subtraction below or the JSONB write in `build_breakdown`
    depending on which side was missing. Both failures are only reachable with
    a posting *and* a profile that state years, which is why coercing at the
    boundary beats hoping the call sites remember.
    """
    if value is None:
        return None
    return float(value)


def seniority_delta(
    profile_level: str | None,
    posting_level: str | None,
    *,
    candidate_years: Decimal | float | int | None = None,
    required_years: Decimal | float | int | None = None,
) -> SeniorityDelta:
    """Compare a candidate's level and years against a posting's.

        seniority_delta("senior", "senior")  -> steps 0,  direction "match"
        seniority_delta("mid", "staff")      -> steps -2, direction "under"
        seniority_delta("staff", "junior")   -> steps 3,  direction "over"
        seniority_delta("senior", "unknown") -> steps None, direction "unknown"

    `steps` is signed from the candidate's side: negative means the role is
    above them. That orientation is chosen so the sign matches `years_gap`,
    which is `candidate - required` and therefore also negative when they fall
    short. Two gap figures that disagreed on which direction was bad would be
    read wrong by somebody eventually.

    **The two halves are independent, and both are reported.** A level is a
    rung the posting named; years is a threshold it set. A posting can state
    one, the other, both, or neither, and each is useful alone -- so this never
    infers a level from years or a threshold from a level. That inference is
    precisely what `seniority_from_title` exists to avoid doing from prose,
    and doing it here would reintroduce it a layer down.
    """
    candidate = _as_float(candidate_years)
    required = _as_float(required_years)

    profile_rank = rank_of(profile_level)
    posting_rank = rank_of(posting_level)

    if profile_rank is None or posting_rank is None:
        steps: int | None = None
        direction: Direction = "unknown"
    else:
        steps = profile_rank - posting_rank
        if steps == 0:
            direction = "match"
        elif steps < 0:
            direction = "under"
        else:
            direction = "over"

    years_gap = None
    if candidate is not None and required is not None:
        years_gap = round(candidate - required, 6)

    return SeniorityDelta(
        profile_level=profile_level,
        posting_level=posting_level,
        steps=steps,
        direction=direction,
        candidate_years=candidate,
        required_years=required,
        years_gap=years_gap,
    )
