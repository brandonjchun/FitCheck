"""Functions RQ executes in a worker process.

Two rules govern everything in this file.

**A job may run more than once.** A worker can complete its side effects and
die before reporting success; RQ then requeues the job and the work happens
again. Exactly-once delivery is not available -- at-least-once is what these
systems provide -- so every handler here is written so that running it twice
produces the same end state as running it once. In practice that means each
one starts by asking "is this already done?" and returns early if so, and
writes with keyed updates rather than blind inserts.

**These run outside the request scope.** There is no FastAPI dependency
injection here, so each task opens and closes its own database session. The
`_session()` helper below is the worker-side equivalent of `get_db`.
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.extraction import CURRENT_EXTRACTION_VERSION
from app.models import Job, Profile
from app.providers import LLMError, LLMPermanentError
from app.workers.extract import extract_profile

logger = logging.getLogger(__name__)

# Errors are truncated before they reach the database. An unbounded traceback
# in a text column is a slow way to fill a disk, and the first 2000 characters
# carry the exception type and message, which is what a human actually reads.
MAX_ERROR_CHARS = 2000


@contextmanager
def _session() -> Iterator[Session]:
    """Open a database session for the duration of one task.

    Same guarantee as get_db: the close runs in a finally, so a task that
    raises still returns its connection to the pool. A worker leaking
    connections is worse than a request leaking them -- it runs for days.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def extract_profile_task(profile_id: int) -> str:
    """Derive structured data for an already-uploaded resume.

    Idempotent: if the profile already has a current-generation extraction,
    this returns without calling the LLM. That check is what makes a
    duplicate delivery free rather than a second API charge.

    The guard is on `extraction_is_current`, not merely on the blob being
    present. A profile extracted under an older prompt has an extraction and
    still needs a new one, so keying the early return on presence alone would
    make a version bump unactionable -- the sweep would find stale rows and
    the task would decline to do anything about them.

    Raises on transient failure so RQ retries. Returns normally on permanent
    failure -- retrying a bad API key, an unknown model, or a resume with no
    text layer burns worker slots on something that cannot succeed.
    """
    with _session() as db:
        profile = db.get(Profile, profile_id)
        if profile is None:
            # The profile was deleted between enqueue and execution. Nothing
            # to do, and nothing worth failing over.
            logger.warning("extract_profile_task: profile %s is gone", profile_id)
            return "profile_missing"

        if profile.extraction_is_current:
            logger.info("extract_profile_task: profile %s already extracted", profile_id)
            return "already_done"

        raw_text = profile.raw_text

    # The LLM call is deliberately outside the session. It takes 26-57 seconds,
    # and holding a pooled connection open for that long would exhaust a pool
    # of 10 with a handful of concurrent workers.
    try:
        extracted = extract_profile(raw_text)
    except LLMPermanentError as exc:
        logger.error("extract_profile_task: permanent failure for %s: %s", profile_id, exc)
        return "permanent_failure"

    with _session() as db:
        profile = db.get(Profile, profile_id)
        if profile is None:
            return "profile_missing"

        # Re-check under the second session: another worker may have finished
        # this same profile while the LLM call was in flight. Last write wins
        # and both writes are equivalent, so this is belt-and-braces rather
        # than load-bearing -- but it keeps the "already done" path honest.
        if profile.extraction_is_current:
            return "already_done"

        # The blob holds exactly what the model returned, un-canonicalized.
        # Skill names are normalized when read back (Profile.skills).
        profile.extracted = extracted.model_dump(mode="json")
        profile.seniority = extracted.seniority
        profile.years_experience = extracted.total_years_experience
        # Written in the same commit as the blob. Setting it separately would
        # allow a crash between the two to leave an extraction whose
        # generation is unknown, which is worse than either value alone.
        profile.extraction_version = CURRENT_EXTRACTION_VERSION
        db.commit()

    return "extracted"


def process_job_url(job_id: int) -> str:
    """Process one submitted job-posting URL.

    M4 SKELETON. The state machine, the attempt counter, and the error
    classification are real; the fetch is not. M5 replaces the placeholder
    with an actual HTTP request plus robots.txt, per-domain rate limiting,
    size caps, and timeouts.

    That the fetch is still a no-op is what makes the M4 batch upload safe:
    a 500-URL list exercises the queues and the fan-out without generating a
    single outbound request. The token bucket has to land with the real fetch
    in M5, before the same upload becomes 500 live HTTP calls.

    The lifecycle this drives (spec section 6.4):

        queued -> running -> succeeded
                          -> (retry, attempts < max) -> queued
                          -> (attempts exhausted) -> dead
    """
    with _session() as db:
        job = db.get(Job, job_id)
        if job is None:
            logger.warning("process_job_url: job %s is gone", job_id)
            return "job_missing"

        # Idempotency guard. A redelivered job that already succeeded must
        # not re-run: at M5 that would mean a second HTTP request to someone
        # else's server, which is exactly the behaviour robots.txt compliance
        # is supposed to prevent.
        if job.status == "succeeded":
            logger.info("process_job_url: job %s already succeeded", job_id)
            return "already_done"

        job.status = "running"
        job.attempts += 1
        db.commit()

        url = job.url
        attempts = job.attempts

    logger.info("process_job_url: job %s attempt %s -> %s", job_id, attempts, url)

    try:
        # --- M5 replaces this block with a real fetch + parse ---------------
        # Deliberately does nothing rather than pretending to. A fake result
        # row here would make the M5 diff look like a refactor instead of the
        # feature it is, and would make this milestone's tests pass for the
        # wrong reason.
        result = "fetch_not_implemented"
        # --------------------------------------------------------------------
    except Exception as exc:
        _record_failure(job_id, exc)
        # Re-raised so RQ sees the failure and applies the retry policy. The
        # database row and RQ's own registry have to agree about what
        # happened, and RQ only learns from the exception propagating.
        raise

    with _session() as db:
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "succeeded"
            job.last_error = None
            db.commit()

    return result


def _record_failure(job_id: int, exc: Exception) -> None:
    """Write a failed attempt back to the job row.

    Sets `dead` once retries are exhausted so the dead-letter list in the M10
    ops dashboard is a plain query rather than a join against RQ's registries.
    """
    from app.queue import MAX_RETRIES

    with _session() as db:
        job = db.get(Job, job_id)
        if job is None:
            return

        job.last_error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        job.status = "dead" if job.attempts >= MAX_RETRIES else "failed"
        db.commit()


def _is_retryable(exc: Exception) -> bool:
    """Whether a failure is worth another attempt.

    Kept for M5/M6, where fetch errors get classified the same way LLM errors
    already are: a 404 will never succeed and retrying it three times wastes
    two minutes and a worker slot, while a 429 or a timeout usually will.
    """
    return not isinstance(exc, LLMPermanentError) and isinstance(exc, LLMError)
