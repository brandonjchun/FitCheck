"""Path A-bulk: upload a .txt of job-posting URLs, poll one progress endpoint.

The distinction from Path A is not size, it is latency class. One submitted
URL means a human is watching a spinner; a 500-URL list means they uploaded
it and walked away. Routing the second onto the interactive queue would put
every single-URL submission behind it, which is the head-of-line blocking the
four-queue split exists to prevent -- so everything here lands on `ingest`.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from rq import Retry
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import IngestJob, Profile, UrlBatch, User, hash_url
from app.queues import (
    FAILURE_TTL,
    JOB_TIMEOUT,
    MAX_RETRIES,
    QUEUE_INGEST,
    RESULT_TTL,
    get_queue,
    retry_intervals,
)
from app.schemas import BatchCreateResponse, BatchStatusResponse
from app.security import current_user
from app.urls import parse_url_list

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/batches", tags=["batches"])

# Statuses that mean a job has not finished. A batch is complete when none of
# its jobs are in one of these.
_IN_FLIGHT = ("queued", "running")


@router.post("", response_model=BatchCreateResponse, status_code=202)
def create_batch(
    profile_id: int,
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> BatchCreateResponse:
    """Accept a URL list and fan it out onto the ingest queue.

    202, not 201: nothing has been fetched. The batch row exists and N jobs
    are queued, which is exactly what "accepted for processing" means.

    Parsing happens in the handler rather than in a job. It is line splitting
    and string normalization over an already-uploaded file -- microseconds,
    local CPU, no network -- so pushing it onto a queue would add a round
    trip and a second polling state to save nothing. The expensive part is
    the N fetches, and those are queued.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Ownership before anything else. Without this predicate the endpoint
    # accepts a profile id belonging to someone else and turns one authenticated
    # request into N outbound fetches attributed to their account -- a request
    # amplifier, not merely a read leak. 404 rather than 403 so the response
    # does not confirm which ids exist.
    profile = db.scalar(
        select(Profile).where(Profile.id == profile_id, Profile.user_id == user.id)
    )
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

    # Per-request caps are not caps if a user can open unlimited requests.
    open_batches = db.scalar(
        select(func.count())
        .select_from(UrlBatch)
        .join(IngestJob, IngestJob.batch_id == UrlBatch.id)
        .where(UrlBatch.user_id == user.id, IngestJob.status.in_(_IN_FLIGHT))
    )
    if open_batches and open_batches >= settings.max_open_batches_per_user:
        raise HTTPException(
            status_code=429,
            detail=(
                f"You already have {settings.max_open_batches_per_user} batches in "
                "progress. Wait for one to finish before uploading another."
            ),
        )

    raw = file.file.read()
    try:
        # utf-8-sig, not utf-8: strips a leading byte order mark if one is
        # there and behaves identically if it is not.
        #
        # Notepad and PowerShell's `-Encoding utf8` both prepend EF BB BF to a
        # file they save. Plain utf-8 decodes that faithfully into a U+FEFF
        # character glued to the front of line one, so the first URL no longer
        # starts with "http" and is counted as unreadable -- while looking
        # perfectly correct to anyone who opens the file, because the
        # character is invisible.
        #
        # Worse than losing a row: it corrupts the accounting this endpoint
        # exists to provide. With line one discarded, a later repeat of that
        # same URL becomes its first occurrence and is accepted rather than
        # counted as a duplicate, so `rejected` and `duplicates` are both
        # wrong rather than merely off by one.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # A .txt that is not UTF-8 is usually a mis-picked file rather than an
        # exotic encoding, and guessing at encodings produces URLs with
        # mojibake in them that fail confusingly three steps later.
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 text, one job posting URL per line.",
        ) from None

    urls, rejected, duplicates = parse_url_list(text, settings.max_urls_per_batch)

    if not urls:
        raise HTTPException(
            status_code=422,
            detail=(
                "No usable job posting URLs found. Expected one http(s) URL per "
                f"line; {rejected} line(s) could not be read as URLs."
            ),
        )

    batch = UrlBatch(
        user_id=user.id,
        profile_id=profile.id,
        original_filename=file.filename,
        total_urls=len(urls),
        rejected_urls=rejected,
        duplicate_urls=duplicates,
    )
    db.add(batch)
    db.flush()  # assigns batch.id without ending the transaction

    # One INSERT for N rows rather than N inserts. At 500 URLs the difference
    # is a round trip versus five hundred of them, and the whole point of this
    # endpoint is that it returns quickly.
    #
    # ON CONFLICT DO NOTHING on (profile_id, url_hash): a user may already
    # have submitted some of these URLs individually, and re-fetching a page
    # this profile has already been scored against is wasted work and an
    # impolite second request to someone else's server. The existing job
    # stands; the duplicate simply does not join this batch.
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    rows = [
        {
            "profile_id": profile.id,
            "batch_id": batch.id,
            "url": url,
            "url_hash": hash_url(url),
            "status": "queued",
        }
        for url in urls
    ]
    created = db.execute(
        pg_insert(IngestJob)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["profile_id", "url_hash"])
        .returning(IngestJob.id)
    ).scalars().all()

    # Count only what this upload actually created, so the reported total
    # matches the number of jobs the status endpoint will ever see.
    batch.total_urls = len(created)
    batch.duplicate_urls = duplicates + (len(urls) - len(created))

    db.commit()
    db.refresh(batch)

    # Enqueue after commit, for the same reason as every other producer here:
    # a worker must never look up a row that has not been written yet. The
    # opposite failure -- committed rows whose enqueue failed -- leaves them
    # `queued` with a null rq_job_id, which a requeue sweep can find.
    if created:
        try:
            # enqueue_many pipelines the whole batch into one Redis round trip.
            # Five hundred individual enqueue calls would put the round trips
            # back into the request this endpoint is trying to keep short.
            queue = get_queue(QUEUE_INGEST)
            queue.enqueue_many(
                [
                    queue.prepare_data(
                        "app.workers.tasks.process_job_url",
                        (job_id,),
                        # `timeout`, not `job_timeout`. Queue.enqueue takes
                        # the latter; prepare_data takes the former for the
                        # same concept, and passing the wrong one is a
                        # TypeError at enqueue time rather than a signature
                        # error at import.
                        timeout=JOB_TIMEOUT,
                        # Fresh jitter per job. Every item in this batch is
                        # aimed at a handful of hosts and enqueued in the same
                        # instant, so a shared schedule would have them all
                        # retry together on the first 429.
                        retry=Retry(max=MAX_RETRIES, interval=retry_intervals()),
                        result_ttl=RESULT_TTL,
                        failure_ttl=FAILURE_TTL,
                    )
                    for job_id in created
                ]
            )
        except Exception as exc:
            logger.error("create_batch: enqueue failed for batch %s: %s", batch.id, exc)

    return BatchCreateResponse(
        id=batch.id,
        profile_id=batch.profile_id,
        filename=batch.original_filename,
        accepted=batch.total_urls,
        rejected=batch.rejected_urls,
        duplicates=batch.duplicate_urls,
        created_at=batch.created_at,
    )


@router.get("/{batch_id}", response_model=BatchStatusResponse)
def get_batch(
    batch_id: int,
    response: Response,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> BatchStatusResponse:
    """Aggregate progress for one batch. This is what the client polls.

    One query for the whole batch. The alternative -- the client polling N
    job endpoints -- turns a 500-item progress view into 250 requests per
    second at a 2s interval, which would make the progress display the
    heaviest thing in the system.
    """
    batch = db.scalar(
        select(UrlBatch).where(UrlBatch.id == batch_id, UrlBatch.user_id == user.id)
    )
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    # Derived, never read from a counter column. N workers incrementing a
    # stored count is the `counter = counter + 1` double-count that
    # at-least-once delivery guarantees, and a summary disagreeing with the
    # rows it summarizes is worse than no summary. The index on batch_id is
    # what keeps this cheap enough to poll.
    counts = dict(
        db.execute(
            select(IngestJob.status, func.count())
            .where(IngestJob.batch_id == batch.id)
            .group_by(IngestJob.status)
        ).all()
    )

    response.headers["Cache-Control"] = "no-store"

    return BatchStatusResponse(
        id=batch.id,
        profile_id=batch.profile_id,
        filename=batch.original_filename,
        total=batch.total_urls,
        rejected=batch.rejected_urls,
        duplicates=batch.duplicate_urls,
        created_at=batch.created_at,
        counts=counts,
        is_complete=not any(counts.get(status) for status in _IN_FLIGHT),
    )


@router.get("", response_model=list[BatchStatusResponse])
def list_batches(
    limit: int = 20,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[BatchStatusResponse]:
    """This user's batches, newest first.

    Scoped by ownership before any caller-supplied parameter is applied, so
    there is no argument that widens the result past what this user owns.
    """
    batches = (
        db.execute(
            select(UrlBatch)
            .where(UrlBatch.user_id == user.id)
            .order_by(UrlBatch.created_at.desc())
            .limit(min(limit, 100))
        )
        .scalars()
        .all()
    )
    if not batches:
        return []

    # One grouped query for every batch on the page rather than one per batch.
    # The per-batch version is the N+1 that makes a list endpoint slow in
    # exactly the situation it is meant for -- a user with many batches.
    grouped = db.execute(
        select(IngestJob.batch_id, IngestJob.status, func.count())
        .where(IngestJob.batch_id.in_([batch.id for batch in batches]))
        .group_by(IngestJob.batch_id, IngestJob.status)
    ).all()

    counts_by_batch: dict[int, dict[str, int]] = {}
    for batch_id, status, count in grouped:
        counts_by_batch.setdefault(batch_id, {})[status] = count

    return [
        BatchStatusResponse(
            id=batch.id,
            profile_id=batch.profile_id,
            filename=batch.original_filename,
            total=batch.total_urls,
            rejected=batch.rejected_urls,
            duplicates=batch.duplicate_urls,
            created_at=batch.created_at,
            counts=counts_by_batch.get(batch.id, {}),
            is_complete=not any(
                counts_by_batch.get(batch.id, {}).get(status) for status in _IN_FLIGHT
            ),
        )
        for batch in batches
    ]
