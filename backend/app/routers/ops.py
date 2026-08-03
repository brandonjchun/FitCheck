"""Operational visibility: queue depth, worker coverage, and the dead letter.

Separate from `/api/jobs` on purpose, and the difference is authorization,
not convenience. `/api/jobs` carries a mandatory ownership join -- it answers
"what have *I* submitted". These endpoints answer "what is the *system*
doing", which is a cross-user question and therefore a different contract
with a different threat model.

Gated on `require_admin`, not merely on being signed in. What is exposed is
counts, queue names, job ids, and error strings -- no resume text and no
personal data -- but it is the whole system rather than the caller's own
rows, and "anyone who can register" is the wrong audience for that. A
`last_error` from a failed fetch can quote a URL somebody else submitted,
which on its own is enough reason not to leave this open.

The flag is granted out of band with SQL; see models.User.is_admin for why
no endpoint grants it.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from rq import Queue, Worker
from rq.registry import (
    DeferredJobRegistry,
    FailedJobRegistry,
    ScheduledJobRegistry,
    StartedJobRegistry,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import IngestJob
from app.queues import (
    FAILURE_TTL,
    JOB_TIMEOUT,
    QUEUE_NAMES,
    RESULT_TTL,
    get_queue,
    get_redis,
)
from app.schemas import (
    DeadLetterItem,
    OpsOverview,
    QueueHealth,
    RequeueResponse,
    StatusCount,
    WorkerInfo,
)
from app.security import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ops", tags=["ops"], dependencies=[Depends(require_admin)])


def _discover_redis_queues() -> set[str]:
    """Every queue RQ currently knows about, declared by us or not.

    Deliberately reads Redis rather than trusting QUEUE_NAMES. A queue that
    exists in Redis but not in our constants is the signature of a rename
    that left work behind -- an API process enqueueing to the old name while
    the workers moved to the new ones. That job is not retrying and not
    failing; nothing is looking at it, and no amount of staring at the code
    reveals it. This is the one view that does.
    """
    try:
        return {q.name for q in Queue.all(connection=get_redis())}
    except Exception as exc:  # pragma: no cover - depends on live Redis
        logger.warning("ops: could not enumerate queues: %s", exc)
        return set()


@router.get("/overview", response_model=OpsOverview)
def overview(db: Session = Depends(get_db)) -> OpsOverview:
    """Everything the dashboard polls, in one round trip.

    One endpoint rather than five because the panels are read together and a
    dashboard that fires five requests every two seconds is its own load
    problem. The cost here is bounded: a handful of Redis reads plus one
    grouped SQL count.
    """
    redis = get_redis()

    # --- Workers, and what each one actually drains --------------------
    workers: list[WorkerInfo] = []
    covered: set[str] = set()
    try:
        for w in Worker.all(connection=redis):
            names = sorted(w.queue_names())
            covered.update(names)
            workers.append(
                WorkerInfo(
                    name=w.name,
                    state=str(w.get_state() or "unknown"),
                    queues=names,
                    current_job_id=w.get_current_job_id(),
                    successful_jobs=w.successful_job_count,
                    failed_jobs=w.failed_job_count,
                )
            )
    except Exception as exc:  # pragma: no cover - depends on live Redis
        logger.warning("ops: could not enumerate workers: %s", exc)

    # --- Queues ---------------------------------------------------------
    # Union of what we declare and what Redis actually holds, so an orphaned
    # queue appears in the list instead of being invisible by omission.
    known = set(QUEUE_NAMES)
    all_names = sorted(known | _discover_redis_queues())

    queues: list[QueueHealth] = []
    for name in all_names:
        q = Queue(name, connection=redis)
        queues.append(
            QueueHealth(
                name=name,
                depth=q.count,
                started=len(StartedJobRegistry(name, connection=redis)),
                failed=len(FailedJobRegistry(name, connection=redis)),
                deferred=len(DeferredJobRegistry(name, connection=redis)),
                scheduled=len(ScheduledJobRegistry(name, connection=redis)),
                declared=name in known,
                worker_count=sum(1 for w in workers if name in w.queues),
            )
        )

    # --- IngestJob rows, counted by status ------------------------------------
    rows = db.execute(select(IngestJob.status, func.count()).group_by(IngestJob.status)).all()
    by_status = [StatusCount(status=str(s), count=int(c)) for s, c in rows]

    return OpsOverview(
        queues=queues,
        workers=workers,
        jobs_by_status=sorted(by_status, key=lambda r: r.status),
        job_timeout_seconds=JOB_TIMEOUT,
        result_ttl_seconds=RESULT_TTL,
        failure_ttl_seconds=FAILURE_TTL,
    )


@router.get("/dead-letter", response_model=list[DeadLetterItem])
def dead_letter(limit: int = 50, db: Session = Depends(get_db)) -> list[DeadLetterItem]:
    """Jobs that exhausted their retries, newest first.

    Sourced from Postgres rather than RQ's FailedJobRegistry because the row
    is the durable record: it survives the registry's TTL and carries the
    attempt count and the truncated error. The registry is the transport's
    view; `ingest_jobs` is ours.
    """
    stmt = (
        select(IngestJob)
        .where(IngestJob.status.in_(("dead", "failed")))
        .order_by(IngestJob.updated_at.desc())
        .limit(min(limit, 200))
    )
    return [
        DeadLetterItem(
            id=j.id,
            url=j.url,
            status=j.status,
            attempts=j.attempts,
            last_error=j.last_error,
            updated_at=j.updated_at,
        )
        for j in db.execute(stmt).scalars().all()
    ]


@router.post("/jobs/{job_id}/requeue", response_model=RequeueResponse, status_code=202)
def requeue(job_id: int, db: Session = Depends(get_db)) -> RequeueResponse:
    """Put a failed job back on its queue.

    Resets `attempts` to zero. A requeue is an operator asserting the cause
    was fixed, so the job deserves a fresh retry budget rather than
    dead-lettering again on its next tick because the counter was already
    exhausted.
    """
    job = db.get(IngestJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status not in ("dead", "failed"):
        # Requeueing a running job would produce a second worker on the same
        # row. The handlers are idempotent so it would converge, but it
        # wastes a slot and muddies the audit trail for no gain.
        raise HTTPException(
            status_code=409,
            detail=f"IngestJob {job_id} is {job.status}; only failed or dead jobs can be requeued",
        )

    job.status = "queued"
    job.attempts = 0
    job.last_error = None
    db.commit()

    # Enqueued after the commit, same ordering as the submit path: a worker
    # must never look up a row whose new state has not been written yet.
    rq_job = get_queue().enqueue(
        "app.workers.tasks.process_job_url",
        job.id,
        job_timeout=JOB_TIMEOUT,
        result_ttl=RESULT_TTL,
        failure_ttl=FAILURE_TTL,
    )
    job.rq_job_id = rq_job.id
    db.commit()

    return RequeueResponse(id=job.id, status=job.status, queue=rq_job.origin)
