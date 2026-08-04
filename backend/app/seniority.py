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
