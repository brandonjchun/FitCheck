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
from datetime import UTC, datetime

from rq import Retry
from sqlalchemy import func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.boards import DiscoveredPosting, enumerate_source
from app.db import SessionLocal
from app.embeddings import cosine_similarity, embed_text
from app.extraction import POSTING_EXTRACTION_VERSION, PROFILE_EXTRACTION_VERSION
from app.models import (
    IngestJob,
    JobPosting,
    Match,
    Profile,
    Source,
    canonical_key_for_url,
    dedupe_key_for_crawl,
    hash_url,
)
from app.providers import LLMError, LLMPermanentError
from app.scoring import (
    SCORER_VERSION,
    SEMANTIC_WEIGHT,
    SKILL_WEIGHT,
    blend,
    score_skills,
)
from app.workers.extract import extract_posting, extract_profile
from app.queues import (
    FAILURE_TTL,
    JOB_TIMEOUT,
    MAX_RETRIES,
    QUEUE_INGEST,
    RESULT_TTL,
    get_queue,
    get_redis,
    retry_intervals,
)
from app.workers.fetch import FetchError, PermanentFetchError, fetch_posting_text

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


def content_hash_for(raw_text: str) -> str:
    """SHA-256 of a posting's normalized text.

    Whitespace is collapsed before hashing so that a template change or a
    different CDN minifier does not read as new content and trigger a
    re-extraction that costs an LLM call for zero information. That
    normalization is what makes the gate in spec section 6.7 actually fire on
    a real board rather than only on a byte-identical one.
    """
    normalized = "\n".join(
        line.strip() for line in raw_text.splitlines() if line.strip()
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


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


# A posting whose "full text" is this short is a title and nothing else.
# Measured on Lever, where inline descriptions ranged 27 to 3,133 characters
# -- the 27 is a real posting with an empty body. Extracting from it produces
# a confident-looking result built from a job title, which is worse than no
# result because nothing downstream can tell it apart from a real one.
MIN_POSTING_CHARS = 200


def discover_source(source_id: int) -> str:
    """Enumerate one board and fan out the work it produced.

    The Path B tick. Steps follow spec section 6.8, and the ordering of the
    last two is the part that matters:

        1. Skip if disabled or the circuit breaker is open.
        2. Record the attempt (`last_crawled_at`) before doing anything.
        3. Enumerate.
        4. Ingest or enqueue each posting.
        5. **Only on a complete enumeration**, tombstone what is missing.

    Step 5's guard is the whole reason `last_success_at` exists as a separate
    column. "Absent from the board, therefore closed" is only sound if the
    list was complete. If enumeration returned three pages and then timed
    out, the postings on page four are absent from our view and present in
    reality -- and running the closure UPDATE would tombstone most of a board
    out of every user's feed, silently, with nothing logged as an error.
    """
    with _session() as db:
        source = db.get(Source, source_id)
        if source is None:
            logger.warning("discover_source: source %s is gone", source_id)
            return "source_missing"
        if not source.enabled:
            logger.info("discover_source: source %s is disabled", source_id)
            return "disabled"
        if source.circuit_open:
            # Not an error, and deliberately not a retry either. A board that
            # has failed five times running is broken in a way that another
            # request will not fix, and continuing to ask is both useless and
            # impolite. Re-enabling is a human decision.
            logger.warning(
                "discover_source: circuit open for %s (%s consecutive failures)",
                source_id,
                source.consecutive_failures,
            )
            return "circuit_open"

        kind, token = source.kind, source.board_token
        # Stamped before enumerating, so a crawl that dies mid-flight still
        # records that it was attempted. The scheduler reads this to decide
        # what is due; leaving it unset would make a reliably-crashing source
        # look permanently due and get retried in a tight loop.
        source.last_crawled_at = func.now()
        db.commit()

    # The boundary between "we attempted" and "we know what is there".
    crawl_started_at = datetime.now(UTC)

    try:
        discovered = enumerate_source(kind, token)
    except FetchError as exc:
        _record_source_failure(source_id, exc)
        # Not re-raised. The retry policy is the wrong instrument here: a
        # crawl is scheduled work that will come round again on its own
        # interval, and RQ retrying inside the same tick would hammer a board
        # that is already failing. The circuit breaker is the mechanism that
        # handles repetition.
        return "enumeration_failed"

    logger.info("discover_source: %s:%s enumerated %d postings", kind, token, len(discovered))

    seen_keys: list[str] = []
    ingested = skipped = enqueued = 0

    for posting in discovered:
        key = canonical_key_for_url(posting.url)
        seen_keys.append(key)

        if posting.content is not None:
            # The listing already carries the text, so there is nothing to
            # fetch. Lever and Ashby both work this way, which makes a crawl
            # of either exactly one HTTP request.
            if len(posting.content) < MIN_POSTING_CHARS:
                logger.info(
                    "discover_source: skipping %s, only %d chars of content",
                    posting.url,
                    len(posting.content),
                )
                skipped += 1
                continue
            _ingest_inline_posting(source_id, key, posting)
            ingested += 1
            continue

        # No inline content, so this needs a fetch -- but only if it is new
        # or the board says it changed. This is the check that turns a daily
        # re-crawl from 400 requests into a handful.
        if not _posting_needs_fetch(key, posting.updated_at):
            _touch_posting(key)
            skipped += 1
            continue

        if _enqueue_posting_fetch(source_id, key, posting):
            enqueued += 1

    # Step 5, and only now: enumeration completed without raising, so the
    # list we hold is the whole board and absence is meaningful.
    closed = _close_missing_postings(source_id, crawl_started_at)

    with _session() as db:
        source = db.get(Source, source_id)
        if source is not None:
            source.last_success_at = func.now()
            # Reset rather than decrement: this counts *consecutive*
            # failures, so one success clears the history. A board that fails
            # once a week forever is healthy; five in a row is not.
            source.consecutive_failures = 0
            db.commit()

    logger.info(
        "discover_source: %s:%s ingested=%d enqueued=%d skipped=%d closed=%d",
        kind, token, ingested, enqueued, skipped, closed,
    )
    return "discovered"


def _record_source_failure(source_id: int, exc: Exception) -> None:
    """Count a failed crawl toward the circuit breaker."""
    with _session() as db:
        source = db.get(Source, source_id)
        if source is None:
            return
        source.consecutive_failures += 1
        db.commit()
        logger.error(
            "discover_source: %s failed (%d consecutive): %s",
            source_id, source.consecutive_failures, exc,
        )


def _posting_needs_fetch(canonical_key: str, updated_at: datetime | None) -> bool:
    """Whether this posting is new, or the board says it changed.

    Returns True when we have never seen it, when the board publishes no
    change signal (so we cannot know and must look), or when its timestamp
    has moved past what we recorded.
    """
    with _session() as db:
        row = db.execute(
            select(JobPosting.source_updated_at, JobPosting.raw_text)
            .where(JobPosting.canonical_key == canonical_key)
        ).one_or_none()

    if row is None:
        return True
    stored_updated_at, raw_text = row
    if not raw_text:
        # Seen, but we never got usable text out of it. Worth another look.
        return True
    if updated_at is None or stored_updated_at is None:
        # No usable change signal on one side or the other, so fall back to
        # fetching and letting the content hash decide. Erring toward the
        # extra request is the cheap mistake -- the alternative is a posting
        # whose requirements changed being scored against the old ones
        # indefinitely.
        return True
    return updated_at > stored_updated_at


def _touch_posting(canonical_key: str) -> None:
    """Heartbeat a posting we chose not to re-fetch.

    `last_seen_at` is what closure detection compares against, so a posting
    skipped for being unchanged must still be marked as present -- otherwise
    the very optimization that avoids re-fetching it would tombstone it.

    `closed_at` is cleared at the same time: a posting reappearing on a board
    after being marked closed is live again, and leaving the tombstone would
    keep it out of every feed forever.
    """
    with _session() as db:
        db.execute(
            update(JobPosting)
            .where(JobPosting.canonical_key == canonical_key)
            .values(last_seen_at=func.now(), closed_at=None)
        )
        db.commit()


def _ingest_inline_posting(
    source_id: int, canonical_key: str, posting: DiscoveredPosting
) -> None:
    """Store a posting whose text arrived with the listing.

    Runs the same content-hash gate as a fetched one. The gate is about
    skipping *extraction and embedding*, which are the expensive halves, and
    those cost the same whether the text arrived by fetch or by listing.
    """
    content_hash = content_hash_for(posting.content or "")

    with _session() as db:
        existing = db.execute(
            select(JobPosting).where(JobPosting.canonical_key == canonical_key)
        ).scalar_one_or_none()

        if (
            existing is not None
            and existing.content_hash == content_hash
            and existing.extraction_is_current
        ):
            # The gate firing. Heartbeat only -- no LLM call, no embedding.
            existing.last_seen_at = func.now()
            existing.closed_at = None
            existing.source_id = source_id
            db.commit()
            _record_gate_hit(canonical_key, hit=True)
            return

        statement = (
            pg_insert(JobPosting)
            .values(
                canonical_key=canonical_key,
                url=posting.url,
                source_id=source_id,
                content_hash=content_hash,
                raw_text=posting.content,
                title=posting.title,
                source_updated_at=posting.updated_at,
            )
            .on_conflict_do_update(
                index_elements=["canonical_key"],
                set_={
                    "url": posting.url,
                    "source_id": source_id,
                    "content_hash": content_hash,
                    "raw_text": posting.content,
                    "source_updated_at": posting.updated_at,
                    "last_seen_at": func.now(),
                    # Changed content invalidates the old extraction. Clearing
                    # the version is what makes `extraction_is_current` false
                    # and gets it re-extracted, rather than leaving stale
                    # skills attached to new text.
                    "extraction_version": None,
                    "closed_at": None,
                },
            )
            .returning(JobPosting.id)
        )
        posting_id = db.execute(statement).scalar_one()
        db.commit()

    _record_gate_hit(canonical_key, hit=False)
    _enqueue_posting_scoring(posting_id)


def _enqueue_posting_fetch(
    source_id: int, canonical_key: str, posting: DiscoveredPosting
) -> bool:
    """Create an ingest job for a posting that has to be fetched.

    Returns whether a job was created. A duplicate is not an error: the
    partial unique index means an identical job already in flight simply
    wins, which is what makes two overlapping crawl ticks safe.
    """
    with _session() as db:
        statement = (
            pg_insert(IngestJob)
            .values(
                kind="ingest_posting",
                # Global rather than per-profile: a crawled posting enters the
                # shared catalog, so two boards listing one job must collapse
                # to a single fetch.
                dedupe_key=dedupe_key_for_crawl(canonical_key),
                source_id=source_id,
                profile_id=None,
                url=posting.url,
                url_hash=hash_url(posting.url),
                status="queued",
            )
            .on_conflict_do_nothing(
                index_elements=["kind", "dedupe_key"],
                index_where=text("status IN ('queued', 'running')"),
            )
            .returning(IngestJob.id)
        )
        job_id = db.execute(statement).scalar_one_or_none()
        db.commit()

    if job_id is None:
        return False

    try:
        get_queue(QUEUE_INGEST).enqueue(
            "app.workers.tasks.process_job_url",
            job_id,
            retry=Retry(max=MAX_RETRIES, interval=retry_intervals()),
            job_timeout=JOB_TIMEOUT,
            result_ttl=RESULT_TTL,
            failure_ttl=FAILURE_TTL,
        )
    except Exception as exc:
        logger.error("discover_source: could not enqueue fetch for %s: %s", posting.url, exc)
    return True


def _close_missing_postings(source_id: int, crawl_started_at: datetime) -> int:
    """Tombstone postings this source no longer lists.

    Called only after a complete enumeration -- see `discover_source`.

    A tombstone rather than a DELETE, because `matches` rows reference
    postings: deleting would either cascade away a user's history or raise a
    foreign key error in the middle of a crawl. `closed_at` also gives the UI
    an honest state, "this role appears to have been filled", which is more
    useful than a row silently vanishing.
    """
    with _session() as db:
        result = db.execute(
            update(JobPosting)
            .where(
                JobPosting.source_id == source_id,
                JobPosting.closed_at.is_(None),
                # Anything this crawl touched had last_seen_at bumped to now,
                # so comparing against the crawl's start finds exactly what it
                # did not see.
                JobPosting.last_seen_at < crawl_started_at,
            )
            .values(closed_at=func.now())
        )
        db.commit()
        return result.rowcount or 0


def _enqueue_posting_scoring(posting_id: int) -> None:
    """Score a newly-ingested catalog posting against active profiles.

    Deliberately a no-op for now. M9 owns the fan-out from one posting to
    every profile that should see it, and doing it here would mean a crawl
    tick enqueueing (postings x profiles) scoring jobs synchronously -- the
    exact unbounded fan-out section 6.9 says to schedule rather than trigger.
    """
    return


# --- content-hash gate instrumentation ---------------------------------
#
# The spec asks for the gate's hit rate to be *reported*, not assumed, and a
# rate you cannot measure is a claim rather than a result. Counters live in
# Redis so they survive across worker processes and can be read by the ops
# dashboard.

_GATE_HIT_KEY = "gate:hits"
_GATE_MISS_KEY = "gate:misses"


def _record_gate_hit(canonical_key: str, hit: bool) -> None:
    try:
        get_redis().incr(_GATE_HIT_KEY if hit else _GATE_MISS_KEY)
    except Exception:
        # Instrumentation must never break ingestion.
        pass


def gate_stats() -> dict[str, int | float]:
    """Content-hash gate hit rate, for the ops dashboard and the writeup."""
    try:
        redis = get_redis()
        hits = int(redis.get(_GATE_HIT_KEY) or 0)
        misses = int(redis.get(_GATE_MISS_KEY) or 0)
    except Exception:
        hits = misses = 0
    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "total": total,
        "hit_rate": round(hits / total, 4) if total else 0.0,
    }
