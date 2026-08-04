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

**Aggregation runs in Postgres, not Python.** `jsonb_array_elements` unnests
the stored breakdown so the grouping happens next to the data. The Python
version is easier to read and pulls every match row plus its full skills array
into the app to count strings, which is a lot of network and parsing for an
answer the database can compute in one pass.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Profile, User
from app.schemas import SkillGap, SkillGapReport
from app.security import current_user

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
        m.job_posting_id,
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
HAVING COUNT(*) FILTER (WHERE bucket IN ('missing', 'partial')) > 0
 ORDER BY blocking DESC, missing DESC, partial DESC
 LIMIT :limit
"""


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
        {"user_id": user.id, "profile_id": profile_id, "limit": MAX_GAPS},
    ).all()

    gaps = [
        SkillGap(
            name=row.name,
            missing=int(row.missing),
            partial=int(row.partial),
            matched=int(row.matched),
            blocking=int(row.blocking),
        )
        for row in rows
    ]

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
