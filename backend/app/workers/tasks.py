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

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.embeddings import cosine_similarity, embed_text
from app.extraction import POSTING_EXTRACTION_VERSION, PROFILE_EXTRACTION_VERSION
from app.models import IngestJob, JobPosting, Match, Profile, canonical_key_for_url
from app.providers import LLMError, LLMPermanentError
from app.scoring import (
    SCORER_VERSION,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
    blend,
    score_skills,
)
from app.workers.extract import extract_posting, extract_profile
from app.workers.fetch import PermanentFetchError, fetch_posting_text

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
        profile.extraction_version = PROFILE_EXTRACTION_VERSION
        db.commit()

    # Embedded after extraction rather than beside it, and deliberately not
    # inside the same transaction. The two describe the resume for different
    # consumers -- the blob feeds skill overlap, the vector feeds semantic
    # similarity -- and only one of them costs an LLM call. If embedding
    # fails, the extraction that was expensive to obtain is already durable,
    # and `embedding IS NULL AND extracted IS NOT NULL` is a query that finds
    # exactly what needs redoing.
    _embed_profile(profile_id)

    return "extracted"


def _embed_profile(profile_id: int) -> bool:
    """Compute and store a profile's embedding. Returns whether it wrote one.

    Idempotent by value: a profile that already has an embedding is left
    alone. That is keyed on presence rather than on a version because the
    embedding has no version of its own -- changing model bumps
    SCORER_VERSION and requires a deliberate backfill, not an opportunistic
    recompute inside whichever task happened to run next.
    """
    with _session() as db:
        profile = db.get(Profile, profile_id)
        if profile is None:
            return False
        if profile.embedding is not None:
            return False
        raw_text = profile.raw_text

    # Outside the session for the same reason the LLM call is: model load is
    # seconds on a cold worker, and inference is hundreds of milliseconds on
    # a long document. Neither is worth a held connection.
    try:
        vector = embed_text(raw_text)
    except ValueError:
        # No embeddable text. Not an error worth retrying -- the upload path
        # already rejects empty documents, so reaching here means the row
        # predates that check.
        logger.warning("_embed_profile: profile %s has no embeddable text", profile_id)
        return False

    with _session() as db:
        profile = db.get(Profile, profile_id)
        if profile is None or profile.embedding is not None:
            return False
        profile.embedding = vector
        db.commit()

    return True


def process_job_url(job_id: int) -> str:
    """Process one submitted job-posting URL.

    Fetches the page (robots-checked, rate-limited, size-capped, timed out),
    reduces it to text, and upserts a JobPosting keyed on canonical_key.

    The upsert is what makes redelivery free. Two users submitting the same
    posting, or the same job replayed after a worker died mid-flight, converge
    on one row rather than creating duplicates -- and a replay costs one
    UPDATE rather than a second request to someone else's server.

    Failures are classified before they reach RQ: a PermanentFetchError (404,
    robots disallow, wrong content type) is recorded and *not* re-raised, so
    the retry policy never fires on work that cannot succeed. A
    TransientFetchError propagates, because RQ only learns to retry from an
    exception escaping.

    The lifecycle this drives (spec section 6.4):

        queued -> running -> succeeded
                          -> (retry, attempts < max) -> queued
                          -> (attempts exhausted) -> dead
    """
    with _session() as db:
        job = db.get(IngestJob, job_id)
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
        raw_text = fetch_posting_text(url)
    except PermanentFetchError as exc:
        # Recorded but NOT re-raised. Letting this propagate would hand it to
        # RQ's retry policy, which would spend two more attempts and four
        # minutes of worker time re-learning that a 404 is still a 404.
        logger.info("process_job_url: job %s permanently failed: %s", job_id, exc)
        _record_failure(job_id, exc, permanent=True)
        return "permanent_failure"
    except Exception as exc:
        _record_failure(job_id, exc)
        # Re-raised so RQ sees the failure and applies the retry policy. The
        # database row and RQ's own registry have to agree about what
        # happened, and RQ only learns from the exception propagating.
        raise

    if not raw_text.strip():
        # A page that fetched cleanly and yielded nothing is the HTML analogue
        # of a scanned PDF: retrying re-downloads the same empty document.
        # Usually a JavaScript-rendered posting, which this fetcher cannot
        # see and which a headless browser would be needed to read.
        _record_failure(
            job_id,
            PermanentFetchError(f"no readable text at {url}"),
            permanent=True,
        )
        return "no_content"

    posting_id = _upsert_posting(url, raw_text)

    with _session() as db:
        job = db.get(IngestJob, job_id)
        if job is not None:
            job.status = "succeeded"
            job.last_error = None
            job.job_posting_id = posting_id
            db.commit()
            profile_id = job.profile_id
        else:
            profile_id = None

    if profile_id is not None:
        _enqueue_scoring(profile_id, posting_id)

    return "fetched"


def _enqueue_scoring(profile_id: int, posting_id: int) -> None:
    """Hand the fetched posting to the scoring queue.

    A separate queue rather than doing it inline, because the two halves have
    opposite shapes: fetching waits on somebody else's server while holding
    almost no CPU, and scoring is an LLM call plus local model inference.
    Running them in one task means a 500-URL batch cannot start scoring
    anything until it has finished fetching everything, and one slow model
    stalls the fetches.

    Enqueued *after* the commit above, so a worker picking this up
    immediately finds a job row that says `succeeded` and a posting that
    exists. The reverse order has a window where scoring reads a posting the
    fetch has not committed yet -- the same ordering rule the API endpoints
    follow.

    A broker failure here is logged and swallowed. The fetch genuinely
    succeeded and the posting is durable, so failing the job would discard
    real work and, worse, hand it back to the retry policy -- which would
    re-fetch a page we already have. `matches` missing a row is recoverable
    by a sweep; a re-fetch is a request to a third party that cannot be taken
    back.
    """
    from app.queues import (
        FAILURE_TTL,
        JOB_TIMEOUT,
        QUEUE_SCORING,
        RESULT_TTL,
        get_queue,
    )

    try:
        get_queue(QUEUE_SCORING).enqueue(
            "app.workers.tasks.score_posting_for_profile",
            profile_id,
            posting_id,
            job_timeout=JOB_TIMEOUT,
            result_ttl=RESULT_TTL,
            failure_ttl=FAILURE_TTL,
        )
    except Exception as exc:
        logger.error(
            "could not enqueue scoring for profile=%s posting=%s: %s",
            profile_id,
            posting_id,
            exc,
        )


def _upsert_posting(url: str, raw_text: str) -> int:
    """Store the fetched text as a JobPosting, keyed on canonical_key.

    ON CONFLICT DO UPDATE rather than a check-then-insert. The check races --
    two workers fetching the same posting both read "absent" and both insert
    -- and under at-least-once delivery a replay is normal operation rather
    than an edge case, so the database has to be the thing that decides.

    `last_seen_at` is bumped on every pass, including one where the text is
    byte-identical. That heartbeat is what closure detection reads at M8:
    a posting absent from a complete crawl is closed, and "absent" is
    measured by this column not having moved.
    """
    content_hash = hashlib.sha256(
        # Normalized before hashing so that whitespace reflow -- a template
        # change, a different CDN minifier -- does not read as new content and
        # trigger a re-extraction that costs an LLM call for zero information.
        "\n".join(line.strip() for line in raw_text.splitlines() if line.strip()).encode(
            "utf-8"
        )
    ).hexdigest()

    key = canonical_key_for_url(url)

    with _session() as db:
        statement = (
            pg_insert(JobPosting)
            .values(
                canonical_key=key,
                url=url,
                content_hash=content_hash,
                raw_text=raw_text,
            )
            .on_conflict_do_update(
                index_elements=["canonical_key"],
                set_={
                    "url": url,
                    "content_hash": content_hash,
                    "raw_text": raw_text,
                    "last_seen_at": func.now(),
                },
            )
            .returning(JobPosting.id)
        )
        posting_id = db.execute(statement).scalar_one()
        db.commit()

    return posting_id


def _record_failure(job_id: int, exc: Exception, permanent: bool = False) -> None:
    """Write a failed attempt back to the job row.

    Sets `dead` once retries are exhausted so the dead-letter list in the M10
    ops dashboard is a plain query rather than a join against RQ's registries.

    `permanent` goes straight to `dead` regardless of the attempt count. A 404
    on attempt one is as final as a 404 on attempt three, and leaving it
    `failed` would misreport it as something a requeue sweep should pick up.
    """
    from app.queues import MAX_RETRIES

    with _session() as db:
        job = db.get(IngestJob, job_id)
        if job is None:
            return

        job.last_error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
        if permanent or job.attempts >= MAX_RETRIES:
            job.status = "dead"
        else:
            job.status = "failed"
        db.commit()


def _prepare_posting(posting_id: int) -> str | None:
    """Ensure a posting is extracted and embedded. Returns a failure reason.

    Split out of the scoring task because the two halves are prerequisites
    with different costs and different failure modes: extraction is an LLM
    call that can fail permanently, embedding is local and effectively
    cannot. Returning a reason rather than raising lets the caller decide --
    scoring can still produce a semantic-only result when extraction is what
    failed, which is a worse answer than a full one and a far better answer
    than none.

    Both steps are guarded, so a redelivered job pays for neither twice.
    """
    with _session() as db:
        posting = db.get(JobPosting, posting_id)
        if posting is None:
            return "posting_missing"
        needs_extraction = not posting.extraction_is_current
        needs_embedding = posting.embedding is None
        raw_text = posting.raw_text

    if needs_extraction:
        try:
            extracted = extract_posting(raw_text)
        except LLMPermanentError as exc:
            logger.error("_prepare_posting: permanent failure for %s: %s", posting_id, exc)
            return "extraction_permanent_failure"

        with _session() as db:
            posting = db.get(JobPosting, posting_id)
            if posting is None:
                return "posting_missing"
            if not posting.extraction_is_current:
                posting.extracted = extracted.model_dump(mode="json")
                # Promoted columns, written in the same commit as the blob so
                # the two cannot disagree about which extraction produced
                # them. These are what M9's feed filters on.
                posting.title = extracted.title
                posting.company = extracted.company
                posting.location = extracted.location
                posting.remote_type = extracted.remote_type
                posting.seniority = extracted.seniority
                posting.min_years = extracted.min_years_experience
                posting.extraction_version = POSTING_EXTRACTION_VERSION
                db.commit()

    if needs_embedding:
        try:
            vector = embed_text(raw_text)
        except ValueError:
            logger.warning("_prepare_posting: posting %s has no embeddable text", posting_id)
            return "no_embeddable_text"

        with _session() as db:
            posting = db.get(JobPosting, posting_id)
            if posting is None:
                return "posting_missing"
            if posting.embedding is None:
                posting.embedding = vector
                db.commit()

    return None


def score_posting_for_profile(
    profile_id: int, posting_id: int, origin: str = "user_submission"
) -> str:
    """Score one posting against one profile and persist the match.

    The task that completes Path A. Runs on the `scoring` queue rather than
    beside the fetch, because the two have very different shapes: a fetch is
    IO-bound waiting on somebody else's server, while this is an LLM call
    plus local inference. Mixing them means a crawl backlog delays a user's
    score, and a slow model delays the crawl.

    Idempotent, keyed on `scorer_version` rather than on the match row
    existing. A match scored under older weights is a match that needs
    re-scoring, so an existence check would make a version bump unactionable
    -- the same trap `extraction_is_current` avoids on the extraction side.
    """
    with _session() as db:
        existing = (
            db.query(Match)
            .filter_by(profile_id=profile_id, job_posting_id=posting_id)
            .one_or_none()
        )
        if existing is not None and existing.scorer_version == SCORER_VERSION:
            logger.info(
                "score_posting_for_profile: %s/%s already scored", profile_id, posting_id
            )
            return "already_done"

    failure = _prepare_posting(posting_id)
    if failure == "posting_missing":
        return failure

    # The profile may predate M7 and carry no vector. Cheap to check, and
    # scoring without it would quietly contribute a zero semantic score to
    # every match rather than announcing the gap.
    _embed_profile(profile_id)

    with _session() as db:
        profile = db.get(Profile, profile_id)
        posting = db.get(JobPosting, posting_id)
        if profile is None or posting is None:
            return "profile_or_posting_missing"

        # Semantic similarity, or 0.0 when either side has no vector. Zero is
        # the honest floor: it says "no evidence of thematic fit", which is
        # what a missing embedding actually means, and the blend still
        # produces a usable ranking from the skill half alone.
        if profile.embedding is not None and posting.embedding is not None:
            semantic = cosine_similarity(list(profile.embedding), list(posting.embedding))
        else:
            semantic = 0.0
            logger.warning(
                "score_posting_for_profile: %s/%s scored without an embedding",
                profile_id,
                posting_id,
            )

        # Both sides normalized on read, which is what makes the comparison
        # work at all -- a resume saying "JS" and a posting saying
        # "JavaScript" are one requirement only after the alias map.
        breakdown = score_skills(posting.skills, profile.skills)
        final = blend(semantic, breakdown.score)

        payload = {
            "semantic_score": round(semantic, 6),
            "skill_score": round(breakdown.score, 6),
            "final_score": round(final, 6),
            "skills": [
                {
                    "name": v.name,
                    "necessity": v.necessity,
                    "bucket": v.bucket,
                    "required_years": v.required_years,
                    "candidate_years": v.candidate_years,
                    "evidence": v.evidence,
                }
                for v in breakdown.verdicts
            ],
            "counts": {
                "matched": len(breakdown.matched),
                "partial": len(breakdown.partial),
                "missing": len(breakdown.missing),
                "missing_required": len(breakdown.missing_required),
            },
            # Stored in the row rather than read from the constants at display
            # time, so an old match still explains itself under the weights it
            # was actually scored with.
            "weights": {"semantic": SEMANTIC_WEIGHT, "skill": SKILL_WEIGHT},
            # True when the posting could not be extracted, so the skill half
            # is empty and the score is semantic-only. Without this the UI
            # would render a confident 0.4 with no skills listed and no way
            # to tell that from a genuine total mismatch.
            "extraction_failed": failure is not None,
        }

        # Upsert rather than insert. Re-scoring is normal -- a version bump, a
        # re-extraction, a redelivered job -- and appending would put one
        # posting in one feed twice at two different ranks.
        statement = (
            pg_insert(Match)
            .values(
                profile_id=profile_id,
                job_posting_id=posting_id,
                semantic_score=semantic,
                skill_score=breakdown.score,
                final_score=final,
                breakdown=payload,
                origin=origin,
                scorer_version=SCORER_VERSION,
            )
            .on_conflict_do_update(
                index_elements=["profile_id", "job_posting_id"],
                set_={
                    "semantic_score": semantic,
                    "skill_score": breakdown.score,
                    "final_score": final,
                    "breakdown": payload,
                    "scorer_version": SCORER_VERSION,
                    "scored_at": func.now(),
                },
            )
        )
        db.execute(statement)
        db.commit()

    logger.info(
        "scored %s/%s: semantic=%.3f skill=%.3f final=%.3f",
        profile_id,
        posting_id,
        semantic,
        breakdown.score,
        final,
    )
    return "scored"
