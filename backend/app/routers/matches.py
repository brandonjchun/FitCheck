"""Reading scored matches.

The read half of Path A. Submission still goes through `POST /api/jobs` --
there is deliberately no second endpoint that takes a URL, because two
entry points doing the same thing is two places for the dedupe and SSRF
rules to disagree.

Everything here is scoped to the caller's own profiles. A match names a
profile, and a profile belongs to a user, so an unscoped read here would
expose one person's resume analysis to anyone who could guess an integer.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import JobPosting, Match, MatchFeedback, Profile, User
from app.queues import FAILURE_TTL, QUEUE_SCORING, RESULT_TTL, get_queue
from app.schemas import (
    FeedbackCreate,
    FeedbackResponse,
    MatchResponse,
    RecommendationRun,
)
from app.scoring import SCORER_VERSION
from app.security import current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/matches", tags=["matches"])

# Validated in the router rather than by a DB constraint, so a bad value gets
# a 422 that names the alternatives instead of a 500 from an integrity error.
ORIGINS: frozenset[str] = frozenset({"user_submission", "recommendation"})

# Longer than the shared JOB_TIMEOUT, because this job is a different shape
# from the ones that constant was set for. A feed build is one recall plus up
# to 200 reranks, and while each rerank is microseconds, the profile embedding
# it may have to compute first is local model inference. 120s is comfortable
# for a single pair and tight for a batch.
RECOMMEND_TIMEOUT = 600

# The feed is paged, and the page is capped. An unbounded `limit` is a way to
# ask the database for every match a user has ever had in one query, which
# gets slower precisely as the product gets more successful.
MAX_LIMIT = 100


def _to_response(match: Match, posting: JobPosting | None) -> MatchResponse:
    """Flatten a Match plus its posting into the wire shape.

    The stored `breakdown` blob is unpacked here rather than returned raw, so
    its internal shape stays free to change without that being a breaking API
    change -- the same separation between storage and contract that keeps
    models.py and schemas.py apart.

    `.get` with defaults throughout, because a row written by an older
    scorer will not have every key a newer response model names. Returning a
    500 for a two-generation-old match would be a worse answer than an
    incomplete one.
    """
    breakdown = match.breakdown or {}
    counts = breakdown.get("counts") or {}

    return MatchResponse(
        id=match.id,
        profile_id=match.profile_id,
        job_posting_id=match.job_posting_id,
        semantic_score=match.semantic_score,
        skill_score=match.skill_score,
        final_score=match.final_score,
        origin=match.origin,
        scorer_version=match.scorer_version,
        scored_at=match.scored_at,
        counts={
            "matched": counts.get("matched", 0),
            "partial": counts.get("partial", 0),
            "missing": counts.get("missing", 0),
            "missing_required": counts.get("missing_required", 0),
        },
        skills=breakdown.get("skills", []),
        weights=breakdown.get("weights", {}),
        extraction_failed=breakdown.get("extraction_failed", False),
        posting_url=posting.url if posting else None,
        posting_title=posting.title if posting else None,
        posting_company=posting.company if posting else None,
    )


@router.get("", response_model=list[MatchResponse])
def list_matches(
    response: Response,
    profile_id: int,
    limit: int = Query(default=25, ge=1, le=MAX_LIMIT),
    origin: str | None = Query(default=None),
    remote_only: bool = Query(default=False),
    seniority: list[str] | None = Query(default=None),
    max_min_years: float | None = Query(default=None, ge=0),
    include_closed: bool = Query(default=False),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[MatchResponse]:
    """Matches for one of the caller's profiles, best first.

    This is the query `matches_feed_idx` exists for -- `(profile_id,
    final_score DESC)` serves both the filter and the ordering, so the
    planner never sorts.

    Ownership is checked on the *profile* rather than filtered on the
    matches. Filtering alone would return an empty list for someone else's
    profile, which is a subtly worse answer: it says "this profile has no
    matches" where the truth is "this profile is not yours", and a client
    cannot tell an empty feed from a forbidden one.

    **The filters here are not the ones in `retrieval.FeedFilters`, and the
    duplication is deliberate.** Those run at recall time and decide what gets
    *scored*; these run at read time and decide what gets *shown*. Collapsing
    them would mean re-running a 200-candidate rerank every time somebody
    ticked "remote only", which is a scoring job's worth of work on the
    request path -- exactly what section 6.9 puts on a queue instead.

    **Closed postings are hidden by default rather than deleted from the
    feed.** A role filled after it was scored is still a true record of what
    was recommended, and `include_closed` keeps it reachable; showing it
    unasked would be presenting a dead link as a live opportunity.
    """
    profile = db.scalar(
        select(Profile).where(Profile.id == profile_id, Profile.user_id == user.id)
    )
    if profile is None:
        # 404 rather than 403, so the response does not confirm which profile
        # ids exist. Same reasoning as `owned_profile`.
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} does not exist")

    if origin is not None and origin not in ORIGINS:
        raise HTTPException(
            status_code=422,
            detail=f"origin must be one of {', '.join(sorted(ORIGINS))}",
        )

    query = (
        select(Match, JobPosting)
        .join(JobPosting, Match.job_posting_id == JobPosting.id)
        .where(Match.profile_id == profile_id)
    )

    if origin is not None:
        query = query.where(Match.origin == origin)
    if not include_closed:
        query = query.where(JobPosting.closed_at.is_(None))
    if remote_only:
        query = query.where(JobPosting.remote_type == "remote")
    if seniority:
        query = query.where(JobPosting.seniority.in_(seniority))
    if max_min_years is not None:
        # NULL means the posting never stated a requirement, which is not the
        # same as stating zero -- the same reasoning as the recall filter.
        query = query.where(
            (JobPosting.min_years.is_(None)) | (JobPosting.min_years <= max_min_years)
        )

    rows = db.execute(
        query.order_by(Match.final_score.desc()).limit(limit)
    ).all()

    # Scores change as postings are re-fetched and re-scored, so a cached
    # feed is a stale ranking presented as a current one.
    response.headers["Cache-Control"] = "no-store"

    return [_to_response(match, posting) for match, posting in rows]


@router.get("/{match_id}", response_model=MatchResponse)
def get_match(
    match_id: int,
    response: Response,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> MatchResponse:
    """One match with its full skill breakdown.

    Joined through `profiles` rather than checked afterwards, so a match
    belonging to another user is indistinguishable from one that does not
    exist -- the row never leaves the database.
    """
    row = db.execute(
        select(Match, JobPosting)
        .join(Profile, Match.profile_id == Profile.id)
        .join(JobPosting, Match.job_posting_id == JobPosting.id)
        .where(Match.id == match_id, Profile.user_id == user.id)
    ).one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Match {match_id} does not exist")

    response.headers["Cache-Control"] = "no-store"
    match, posting = row
    return _to_response(match, posting)


@router.post("/recommendations", response_model=RecommendationRun, status_code=202)
def build_recommendations(
    profile_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> RecommendationRun:
    """Ask for this profile's feed to be built, if it needs building.

    Section 6.9 option 1 -- lazy, with a warm state. The three properties that
    make this the chosen strategy over a nightly sweep are all visible here:
    the work is queued rather than run on the request path, it happens because
    somebody asked rather than on a schedule, and the cost therefore tracks
    active users rather than registered ones.

    **202 always, never 200 with results.** Building a feed is a recall plus
    200 reranks; doing it inline would make the first feed request the slowest
    request in the application and hold a worker-equivalent of work open on a
    web connection. The client gets an acknowledgement and polls the feed it
    was already polling.

    **`already_current` is the interesting return.** A feed that was scored
    under the current `scorer_version` does not need rebuilding, so a client
    that polls this endpoint cannot stampede the queue -- which matters
    because the natural client implementation calls it whenever the feed looks
    empty, and an empty feed is exactly the state a user stares at.
    """
    profile = db.scalar(
        select(Profile).where(Profile.id == profile_id, Profile.user_id == user.id)
    )
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} does not exist")

    existing = db.scalar(
        select(func.count())
        .select_from(Match)
        .where(
            Match.profile_id == profile_id,
            Match.origin == "recommendation",
            Match.scorer_version == SCORER_VERSION,
        )
    )
    if existing:
        return RecommendationRun(
            profile_id=profile_id, status="already_current", queued=False
        )

    # A profile with no vector cannot be recalled against, and saying so is
    # more use than queueing a job that will return "no_embedding" into a log
    # nobody is reading. The extraction that produces the vector is itself a
    # queued job, so this is a normal transient state right after upload.
    if profile.embedding is None:
        return RecommendationRun(
            profile_id=profile_id, status="profile_not_ready", queued=False
        )

    try:
        get_queue(QUEUE_SCORING).enqueue(
            "app.workers.tasks.score_profile",
            profile_id,
            job_timeout=RECOMMEND_TIMEOUT,
            result_ttl=RESULT_TTL,
            failure_ttl=FAILURE_TTL,
        )
    except Exception as exc:
        # Redis down is not the caller's problem to solve, but it is their
        # problem to know about: unlike an upload, there is no durable row
        # here that a later sweep would pick up, so silently returning
        # "queued" would promise a feed that nothing is building.
        logger.error("could not enqueue score_profile for %s: %s", profile_id, exc)
        raise HTTPException(
            status_code=503, detail="Could not queue the feed build. Try again shortly."
        )

    logger.info("recommendations: queued score_profile for %s", profile_id)
    return RecommendationRun(profile_id=profile_id, status="queued", queued=True)


@router.post(
    "/{match_id}/feedback",
    response_model=FeedbackResponse,
    status_code=201,
)
def submit_feedback(
    match_id: int,
    payload: FeedbackCreate,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """Record what the caller thought of a recommendation.

    Section 8.6's data collection. Nothing reads this yet -- the blend weights
    stay hand-set this semester -- and collecting it anyway is the point: a
    label cannot be gathered retroactively, so the cost of starting late is
    the whole dataset.

    **201 with a new row on every call, including a repeat.** The table is
    append-only, so marking a posting `interested` and later `applied` records
    both, in order. That sequence is the funnel, and it is exactly what a
    later ranking model would learn from; an upsert keyed on the match would
    quietly discard it.

    Ownership joins through `profiles` for the same reason `get_match` does:
    a match belonging to somebody else must be indistinguishable from one that
    does not exist, or this endpoint becomes a way to enumerate match ids.
    """
    match = db.scalar(
        select(Match)
        .join(Profile, Match.profile_id == Profile.id)
        .where(Match.id == match_id, Profile.user_id == user.id)
    )
    if match is None:
        raise HTTPException(status_code=404, detail=f"Match {match_id} does not exist")

    feedback = MatchFeedback(match_id=match_id, verdict=payload.verdict)
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    logger.info("feedback: match=%s verdict=%s", match_id, payload.verdict)
    return FeedbackResponse(
        id=feedback.id,
        match_id=feedback.match_id,
        verdict=feedback.verdict,
        created_at=feedback.created_at,
    )
