"""Routes for submitting and polling job-posting URLs."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Job, Profile, User, hash_url
from app.queues import (
    FAILURE_TTL,
    JOB_TIMEOUT,
    MAX_RETRIES,
    QUEUE_INTERACTIVE,
    RESULT_TTL,
    RETRY_INTERVALS,
    get_queue,
)
from app.schemas import JobResponse, JobSubmitRequest
from app.security import current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=202)
def submit_job(
    payload: JobSubmitRequest,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Job:
    """Queue a job-posting URL for processing.

    Returns **202 Accepted**, not 201. The distinction is the whole point of
    this milestone: 201 would promise the work is done and the result is
    available at the returned location. 202 promises only that the request
    was accepted for processing -- which is exactly the truth, because
    fetching an arbitrary URL takes 200ms to 30s, fails often, and cannot
    happen inside a request handler.

    Submitting the same URL twice for the same profile returns the existing
    job rather than creating a duplicate scrape.
    """
    url = str(payload.url)
    url_digest = hash_url(url)

    # The profile id arrives in the body rather than the path, so
    # `owned_profile` (which reads a path parameter) does not apply -- but the
    # predicate is the same and skipping it would be the same vulnerability.
    # Without `user_id` here, anyone could enqueue outbound fetches against
    # any profile id they cared to guess. 404 rather than 403, so the response
    # does not confirm which ids exist.
    profile = db.scalar(
        select(Profile).where(
            Profile.id == payload.profile_id,
            Profile.user_id == user.id,
        )
    )
    if profile is None:
        raise HTTPException(
            status_code=404, detail=f"Profile {payload.profile_id} does not exist"
        )

    # Fast path for the common case. This check is an optimization, not the
    # guarantee -- two concurrent submissions can both pass it. The unique
    # constraint below is what actually prevents the duplicate.
    existing = db.execute(
        select(Job).where(
            Job.profile_id == payload.profile_id, Job.url_hash == url_digest
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    job = Job(
        profile_id=payload.profile_id,
        url=url,
        url_hash=url_digest,
        status="queued",
    )
    db.add(job)

    try:
        db.commit()
    except IntegrityError:
        # Lost the race. Another request inserted the same (profile_id,
        # url_hash) between our check and our commit. That is the constraint
        # doing its job -- roll back and return whichever row won.
        db.rollback()
        winner = db.execute(
            select(Job).where(
                Job.profile_id == payload.profile_id, Job.url_hash == url_digest
            )
        ).scalar_one()
        logger.info("submit_job: lost insert race, returning job %s", winner.id)
        return winner

    db.refresh(job)

    # Enqueue AFTER the commit. Enqueueing first opens a window where a worker
    # picks up the job and queries for a row that has not been committed yet.
    # This ordering has the opposite failure mode -- a committed row whose
    # enqueue then fails, leaving it stuck in `queued` forever -- which is
    # recoverable by a requeue sweep and is the far better trade.
    from rq import Retry

    try:
        # `interactive`, because a human submitted this one URL and is
        # watching for the result. The same function runs on `ingest` when a
        # batch upload produces it -- identical work, different urgency, and
        # the queue is the only thing that carries that distinction.
        rq_job = get_queue(QUEUE_INTERACTIVE).enqueue(
            "app.workers.tasks.process_job_url",
            job.id,
            retry=Retry(max=MAX_RETRIES, interval=RETRY_INTERVALS),
            job_timeout=JOB_TIMEOUT,
            result_ttl=RESULT_TTL,
            failure_ttl=FAILURE_TTL,
        )
        job.rq_job_id = rq_job.id
        db.commit()
        db.refresh(job)
    except Exception as exc:
        # Redis being down must not lose the submission. The row stays
        # `queued` with no rq_job_id, which is precisely the signature a
        # requeue sweep looks for.
        logger.error("submit_job: enqueue failed for job %s: %s", job.id, exc)

    return job


@router.get("/{job_id}", response_model=JobResponse)
def get_job(
    job_id: int,
    response: Response,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Job:
    """Current state of one job. This is the endpoint the frontend polls.

    Sets `Cache-Control: no-store` because a cached job status is worse than
    useless -- a proxy holding a `queued` response for 30 seconds makes the
    UI look stuck when the work has already finished.

    Ownership is inherited through the profile rather than stored on the job.
    A job belongs to whoever owns the resume it was submitted for, so joining
    keeps one source of truth; a `user_id` column here could disagree with
    `profiles.user_id` and there would be no way to say which was right.
    """
    job = db.scalar(
        select(Job)
        .join(Profile, Job.profile_id == Profile.id)
        .where(Job.id == job_id, Profile.user_id == user.id)
    )
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    response.headers["Cache-Control"] = "no-store"
    return job


@router.get("", response_model=list[JobResponse])
def list_jobs(
    profile_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[Job]:
    """List jobs, newest first. Filterable by profile and status.

    Both filters hit indexed columns. This is the query the M10 ops dashboard
    is built on -- `?status=dead` is the dead-letter list.

    The ownership join is not optional and not a filter the caller supplies.
    `?profile_id=` narrows within what this user owns; it cannot widen beyond
    it, so passing someone else's id returns an empty list rather than their
    jobs. A listing endpoint is the easiest place to leak everything at once,
    because one missing predicate returns every row in the table instead of
    one.
    """
    query = (
        select(Job)
        .join(Profile, Job.profile_id == Profile.id)
        .where(Profile.user_id == user.id)
        .order_by(Job.created_at.desc())
        .limit(min(limit, 200))
    )

    if profile_id is not None:
        query = query.where(Job.profile_id == profile_id)
    if status is not None:
        query = query.where(Job.status == status)

    return list(db.execute(query).scalars().all())
