"""What the scores add up to, across a whole feed rather than one match.

A single match answers "am I a fit for this posting". Fifty matches contain a
different and more useful answer -- *which missing skill costs you the most
postings* -- and nothing in the product surfaced it, because the breakdown
blob was only ever read one row at a time.

**Why this is the right thing to build on top of M9/M10.** The data has been
accumulating since M7 and cost an LLM call per posting to produce. Aggregating
it is nearly free, needs no new extraction, and turns a pile of per-match
explanations into the one thing a job seeker can actually act on: a ranked
list of what to learn next.

**Unnesting runs in Postgres; the final grouping runs in Python.**
`jsonb_array_elements` still does the expensive part next to the data --
exploding every breakdown and counting per name -- so what crosses the wire is
one row per distinct skill *spelling*, not every match's full skills array.

The last step is Python because the key it groups on lives here.
`skills.canonical_key` is what decides that "GraphQL", "GraphQL API", and
"graphql-api" are one requirement, and reimplementing it as a SQL expression
would put the same rule in two places written two ways -- which is how the
report and the scorer end up disagreeing about what counts as the same skill.
Reading the raw counts back and folding them here keeps one definition.

The row count that crosses the wire is bounded by distinct spellings across
one user's matches -- on a real 50-match feed that was 125 rows.
"""

import logging
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Profile, User
from app.schemas import SkillGap, SkillGapReport
from app.security import current_user
from app.skills import canonical_key, normalize_skill

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/insights", tags=["insights"])

# Enough to see the pattern, few enough to act on. A list of forty gaps is a
# list nobody reads, and the tail is dominated by one-off requirements from a
# single posting.
MAX_GAPS = 12


# `necessity = 'required'` is the filter that makes this advice rather than
# trivia: a missing "nice to have" did not cost the candidate the posting, and
# including those would rank politeness words above the things that actually
# disqualify.
#
# `bucket` is counted rather than filtered so partial credit stays visible --
# "you have 2 of the 5 years asked for" is a different problem from "you have
# never touched this", and they call for different responses.
_SKILL_GAP_SQL = """
WITH exploded AS (
    SELECT
        s->>'name'      AS name,
        s->>'bucket'    AS bucket,
        s->>'necessity' AS necessity
      FROM matches m
      JOIN profiles p ON p.id = m.profile_id
      CROSS JOIN LATERAL jsonb_array_elements(
          COALESCE(m.breakdown -> 'skills', '[]'::jsonb)
      ) AS s
     WHERE p.user_id = :user_id
       AND (CAST(:profile_id AS bigint) IS NULL OR m.profile_id = CAST(:profile_id AS bigint))
)
SELECT
    name,
    COUNT(*) FILTER (WHERE bucket = 'missing')                        AS missing,
    COUNT(*) FILTER (WHERE bucket = 'partial')                        AS partial,
    COUNT(*) FILTER (WHERE bucket = 'matched')                        AS matched,
    COUNT(*) FILTER (WHERE bucket = 'missing' AND necessity = 'required')
                                                                      AS blocking
  FROM exploded
 WHERE name IS NOT NULL
 GROUP BY name
"""


# No HAVING and no LIMIT in the SQL above, and both omissions are deliberate.
#
# Filtering to "has a gap" before merging would drop a spelling whose own rows
# are all matched, losing its matched count from the merged total -- the
# denominator the UI renders the bar against. And a LIMIT before merging would
# take the top N *spellings*, which is exactly the bug this whole change is
# about: five GraphQL variants filled five of twelve slots and the real
# top gap never appeared. Both now happen after the merge.


def _merge_variants(rows) -> list[SkillGap]:
    """Fold spellings of one skill together, and pick a name to show.

    Everything that reduces to the same `canonical_key` is one requirement, so
    the counts add. What is left is choosing which of the observed spellings a
    human should see, in this order:

    1. **The alias map, if it knows this skill.** `normalize_skill` is the
       project's existing authority on canonical names, and deferring to it is
       what keeps the report saying "GraphQL" rather than whichever variant
       happened to be most common this week.
    2. **Fewest words.** "Machine Learning" over "Machine Learning
       Frameworks" -- the shorter form is the skill, the longer one is the
       skill plus filler.
    3. **Most frequent**, then **most capitalised**, so "Redis" beats "redis"
       and "CSS" beats "css".
    4. **Alphabetical**, purely so the choice is deterministic. Without a
       total order the displayed name could change between two runs over
       identical data, which looks like the data moved.
    """
    merged: dict[str, dict] = {}

    for row in rows:
        key = canonical_key(row.name)
        if not key:
            continue

        entry = merged.setdefault(
            key,
            {"variants": Counter(), "missing": 0, "partial": 0, "matched": 0, "blocking": 0},
        )
        seen = int(row.missing) + int(row.partial) + int(row.matched)
        entry["variants"][row.name] += seen
        entry["missing"] += int(row.missing)
        entry["partial"] += int(row.partial)
        entry["matched"] += int(row.matched)
        entry["blocking"] += int(row.blocking)

    gaps: list[SkillGap] = []
    for entry in merged.values():
        variants: Counter = entry["variants"]

        def rank(name: str) -> tuple:
            return (
                len(name.split()),
                -variants[name],
                -sum(character.isupper() for character in name),
                name,
            )

        best = min(variants, key=rank)
        # `normalize_skill` returns an unrecognised name unchanged, so this is
        # "use the alias map when it has an opinion" rather than a second
        # normalization step.
        gaps.append(
            SkillGap(
                name=normalize_skill(best),
                missing=entry["missing"],
                partial=entry["partial"],
                matched=entry["matched"],
                blocking=entry["blocking"],
            )
        )

    # Ranked by blocking first -- a requirement the posting called required and
    # the candidate lacks -- so a widely-listed nice-to-have cannot outrank the
    # thing actually disqualifying them.
    gaps.sort(key=lambda g: (-g.blocking, -g.missing, -g.partial, g.name))
    return [gap for gap in gaps if gap.missing or gap.partial][:MAX_GAPS]


@router.get("/skill-gaps", response_model=SkillGapReport)
def skill_gaps(
    response: Response,
    profile_id: int | None = Query(default=None),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> SkillGapReport:
    """The requirements this user fails most often, across their matches.

    Scoped to the caller's own profiles by joining through `profiles` on
    `user_id`, not by filtering afterwards -- the same structural approach the
    match endpoints use, so there is no version of this query that can read
    somebody else's feed.

    `profile_id` is optional. Omitted, it aggregates across every resume the
    user has, which is the more interesting view: a gap that shows up under
    all of them is a gap in the person rather than in one document.
    """
    if profile_id is not None:
        owned = db.scalar(
            select(Profile.id).where(
                Profile.id == profile_id, Profile.user_id == user.id
            )
        )
        if owned is None:
            raise HTTPException(
                status_code=404, detail=f"Profile {profile_id} does not exist"
            )

    rows = db.execute(
        text(_SKILL_GAP_SQL),
        {"user_id": user.id, "profile_id": profile_id},
    ).all()

    gaps = _merge_variants(rows)

    # Denominator for the percentages the UI renders. Without it "missing in
    # 9" is unreadable -- 9 out of 10 and 9 out of 400 are opposite findings.
    scored = db.scalar(
        text(
            """
            SELECT COUNT(*)
              FROM matches m
              JOIN profiles p ON p.id = m.profile_id
             WHERE p.user_id = :user_id
               AND (CAST(:profile_id AS bigint) IS NULL OR m.profile_id = CAST(:profile_id AS bigint))
            """
        ),
        {"user_id": user.id, "profile_id": profile_id},
    )

    response.headers["Cache-Control"] = "no-store"
    return SkillGapReport(
        profile_id=profile_id, matches_analyzed=int(scored or 0), gaps=gaps
    )
