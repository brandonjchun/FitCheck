"""Retry, backoff, and dead-lettering, driven through a real RQ worker.

Every other test in this suite calls the task functions directly. That covers
what the *handler* does with a failure and says nothing about what RQ does
with it -- and the retry policy lives in neither place alone. It is a `Retry`
object handed to `enqueue`, interpreted by a worker, parked in a registry, and
eventually abandoned. `test_worker_tasks.py` asserts the policy *reaches*
enqueue; nothing until now asserted that a genuinely failing job comes back.

Two things make this cheap enough to keep in the default suite.

**The worker is `SimpleWorker`.** Production forks a work horse per job, which
`fork()` makes impossible on Windows. Running in-process is what lets
`monkeypatch` reach the task at all -- a forked child would inherit the patch
but a spawned one would not, and neither would let the test count calls. The
retry machinery under test is on `BaseWorker` and is identical either way; the
only thing forking buys is crash isolation, which is not what this file is
about.

**Backoff is asserted, not waited out.** The real schedule is 10s / 60s / 300s,
so sitting through it costs six minutes and buys nothing: the fact under test
is that the worker parks the job at `now + interval`, and that is readable from
the scheduled registry the instant the job fails. `_release_retry` then rewrites
the timestamp into the past and lets RQ's own scheduler move the job back --
the same code path the production `rq worker --with-scheduler` runs, just not
made to wait for a clock.

**The queue name is unique per test.** `docker compose up` leaves four workers
draining `interactive`, `ingest`, `scoring`, and `discovery` against this same
Redis. A test enqueueing onto any of those is racing them for its own job, and
would fail intermittently in a way that looks like a retry bug.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from redis import Redis
from rq import Queue, Retry, SimpleWorker
from rq.job import Job, JobStatus
from rq.registry import FailedJobRegistry, ScheduledJobRegistry
from rq.scheduler import RQScheduler

from app.config import settings
from app.db import SessionLocal
from app.models import IngestJob, Profile, dedupe_key_for_submission, hash_url
from app.queues import (
    FAILURE_TTL,
    JOB_TIMEOUT,
    MAX_RETRIES,
    RESULT_TTL,
    RETRY_BASE_INTERVALS,
    RETRY_JITTER,
    retry_intervals,
)
from app.workers import tasks
from app.workers.fetch import PermanentFetchError, TransientFetchError

# Quiet the worker. A failing job logs its full traceback at ERROR, and this
# file fails jobs deliberately -- four tracebacks per test is noise that hides
# a real one.
WORKER_LOG_LEVEL = "CRITICAL"

# Slack between the scheduled time RQ wrote and the interval it was derived
# from. Covers the job's own execution plus the round trip to Redis, and is
# far below the smallest gap between two intervals (10s -> 60s), so it cannot
# make one interval pass as another.
SCHEDULE_TOLERANCE_SECONDS = 5


@pytest.fixture
def redis():
    connection = Redis.from_url(settings.redis_url)
    yield connection
    connection.close()


@pytest.fixture
def queue(redis):
    """A queue no running worker is listening on.

    Named per test rather than per session: two tests sharing a name would
    also share a scheduled registry, and a job left parked by one would be
    released into the other's burst.
    """
    q = Queue(f"test-retry-{uuid.uuid4().hex[:12]}", connection=redis)
    yield q

    # RQ spreads one queue across several keys and a global set of queue
    # names. Left behind, they show up as phantom queues on the ops dashboard,
    # which reads every name in `rq:queues`.
    for registry in (
        ScheduledJobRegistry(q.name, connection=redis),
        FailedJobRegistry(q.name, connection=redis),
    ):
        redis.delete(registry.key)
    q.empty()
    redis.delete(q.key)
    redis.srem("rq:queues", q.key)


@pytest.fixture
def worker(queue, redis):
    w = SimpleWorker([queue], connection=redis)
    yield w
    w.register_death()


@pytest.fixture
def ingest_job(make_user):
    """A `queued` ingest_jobs row owned by a throwaway user.

    Cleanup rides on the user: deleting them cascades to the profile and
    through it to the job.
    """
    user = make_user()
    url = f"https://example.com/retry/{uuid.uuid4().hex[:8]}"

    db = SessionLocal()
    try:
        profile = Profile(
            user_id=user.id, original_filename="r.pdf", raw_text="Brandon uses Python."
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)

        job = IngestJob(
            profile_id=profile.id,
            url=url,
            url_hash=hash_url(url),
            status="queued",
            dedupe_key=dedupe_key_for_submission(profile.id, hash_url(url)),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id
    finally:
        db.close()


@pytest.fixture
def failing_fetch(monkeypatch):
    """Make every fetch raise, and count the attempts.

    Patched on `app.workers.tasks` because `process_job_url` resolves
    `fetch_posting_text` as a module global at call time -- and because
    `SimpleWorker` performs the job in this process, the patch is live inside
    the worker. Under a forking worker this fixture would be invisible to the
    child, which is the second reason this file does not use one.
    """
    attempts: list[str] = []

    def install(exc_factory=lambda: TransientFetchError("connection reset")):
        def boom(url: str) -> str:
            attempts.append(url)
            raise exc_factory()

        monkeypatch.setattr(tasks, "fetch_posting_text", boom)

    install()
    install.attempts = attempts
    return install


def _enqueue(queue, ingest_job_id: int) -> Job:
    """Enqueue exactly as the submission endpoints do.

    The retry configuration is the thing under test, so it is imported from
    `app.queues` rather than restated here. A test that hard-codes `max=3`
    keeps passing after somebody changes MAX_RETRIES to 5.
    """
    return queue.enqueue(
        "app.workers.tasks.process_job_url",
        ingest_job_id,
        retry=Retry(max=MAX_RETRIES, interval=retry_intervals()),
        job_timeout=JOB_TIMEOUT,
        result_ttl=RESULT_TTL,
        failure_ttl=FAILURE_TTL,
    )


def _scheduled_delay(queue, redis, rq_job: Job) -> float:
    """Seconds from now until RQ intends to run this job again."""
    registry = ScheduledJobRegistry(queue.name, connection=redis)
    scheduled_at = registry.get_scheduled_time(rq_job)
    return (scheduled_at - datetime.now(UTC)).total_seconds()


def _release_retry(queue, redis) -> None:
    """Fast-forward the backoff without waiting it out.

    Rewrites each parked job's timestamp to the past and then runs RQ's own
    `enqueue_scheduled_jobs`, which is what moves a scheduled job back onto
    its queue in production. Re-implementing that move by hand would test the
    test rather than RQ -- the point of going through the scheduler is that a
    change to how RQ releases scheduled work shows up here.
    """
    registry = ScheduledJobRegistry(queue.name, connection=redis)
    parked = registry.get_job_ids()

    for job_id in parked:
        registry.schedule(
            Job.fetch(job_id, connection=redis), datetime.now(UTC) - timedelta(seconds=1)
        )

    scheduler = RQScheduler([queue], connection=redis)
    scheduler.prepare_registries([queue.name])
    scheduler.enqueue_scheduled_jobs()


def _db_job(ingest_job_id: int) -> IngestJob:
    """Read the work record back, detached from any session."""
    db = SessionLocal()
    try:
        job = db.get(IngestJob, ingest_job_id)
        db.refresh(job)
        db.expunge(job)
        return job
    finally:
        db.close()


class TestRetryHappens:
    """The claim `test_worker_tasks` cannot make: the job actually comes back."""

    def test_a_transient_failure_is_rescheduled_not_abandoned(
        self, worker, queue, redis, failing_fetch, ingest_job
    ):
        rq_job = _enqueue(queue, ingest_job)

        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

        assert len(failing_fetch.attempts) == 1
        # Parked, not failed. A job that lands in the failed registry on its
        # first transient error means the Retry object never reached RQ --
        # which is exactly what an `enqueue` call missing `retry=` looks like,
        # and nothing else in the suite would notice.
        assert rq_job.id in ScheduledJobRegistry(queue.name, connection=redis).get_job_ids()
        assert rq_job.id not in FailedJobRegistry(queue.name, connection=redis).get_job_ids()

    def test_the_rescheduled_job_runs_again(
        self, worker, queue, redis, failing_fetch, ingest_job
    ):
        _enqueue(queue, ingest_job)

        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)
        _release_retry(queue, redis)
        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

        assert len(failing_fetch.attempts) == 2

    def test_every_attempt_is_counted_on_the_work_record(
        self, worker, queue, redis, failing_fetch, ingest_job
    ):
        """`attempts` is what the ops dashboard shows and what `_record_failure`
        branches on, so it has to track real executions rather than enqueues."""
        _enqueue(queue, ingest_job)

        for _ in range(3):
            worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)
            _release_retry(queue, redis)

        assert _db_job(ingest_job).attempts == 3


class TestBackoff:
    """Backoff is only real if the worker parks the job that far out."""

    def test_the_delay_matches_the_schedule_the_job_carries(
        self, worker, queue, redis, failing_fetch, ingest_job
    ):
        """Asserted against the job's own jittered schedule, not the base
        constants. Comparing to RETRY_BASE_INTERVALS would need a tolerance
        wide enough to swallow the jitter, which would stop the assertion
        saying anything about which interval was used."""
        rq_job = _enqueue(queue, ingest_job)
        schedule = rq_job.retry_intervals

        for index in range(MAX_RETRIES):
            worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

            delay = _scheduled_delay(queue, redis, rq_job)
            expected = schedule[index]
            assert expected - SCHEDULE_TOLERANCE_SECONDS <= delay <= expected, (
                f"retry {index + 1} parked {delay:.1f}s out, expected ~{expected}s"
            )

            _release_retry(queue, redis)

    def test_the_delay_grows(self, worker, queue, redis, failing_fetch, ingest_job):
        """The whole point of exponential rather than fixed interval: a 503
        from an overloaded host is not resolved by asking again on the same
        cadence."""
        rq_job = _enqueue(queue, ingest_job)
        delays: list[float] = []

        for _ in range(MAX_RETRIES):
            worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)
            delays.append(_scheduled_delay(queue, redis, rq_job))
            _release_retry(queue, redis)

        assert delays == sorted(delays)
        assert delays[0] < delays[-1]

    def test_jitter_is_applied_and_bounded(self):
        """A schedule that never varies is a synchronized retry cohort, and one
        that varies without bound is not a schedule. Both directions matter, so
        both are pinned -- and the varying half is sampled rather than asserted
        once, because a single draw can legitimately land on its base value."""
        schedules = [retry_intervals() for _ in range(50)]

        for schedule in schedules:
            assert len(schedule) == len(RETRY_BASE_INTERVALS)
            for base, jittered in zip(RETRY_BASE_INTERVALS, schedule):
                assert base * (1 - RETRY_JITTER) - 1 <= jittered <= base * (1 + RETRY_JITTER) + 1

        # Two jobs enqueued in the same millisecond must not share a schedule.
        # This is the failure a module-level constant would produce, and it is
        # invisible until a batch of 500 returns in lockstep.
        assert len({tuple(schedule) for schedule in schedules}) > 1


class TestDeadLetter:
    """Where an unfixable job ends up, on both sides of the system."""

    @pytest.fixture
    def exhausted(self, worker, queue, redis, failing_fetch, ingest_job):
        """Drive one job through every attempt RQ will give it."""
        rq_job = _enqueue(queue, ingest_job)

        # One initial run plus MAX_RETRIES retries. The release after the last
        # one is harmless -- there is nothing parked to release.
        for _ in range(MAX_RETRIES + 1):
            worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)
            _release_retry(queue, redis)

        return rq_job

    def test_retries_are_finite(self, exhausted, failing_fetch):
        """`Retry(max=N)` is N retries *after* the first run, so a job that
        cannot succeed costs N+1 executions and then stops. A policy that
        retried forever would look identical for the first few minutes."""
        assert len(failing_fetch.attempts) == MAX_RETRIES + 1

    def test_it_lands_in_the_failed_registry(self, exhausted, queue, redis):
        registry = FailedJobRegistry(queue.name, connection=redis)

        assert exhausted.id in registry.get_job_ids()
        assert exhausted.get_status() == JobStatus.FAILED
        # And is not still parked for a fourth retry.
        assert exhausted.id not in ScheduledJobRegistry(
            queue.name, connection=redis
        ).get_job_ids()

    def test_the_work_record_agrees(self, exhausted, ingest_job):
        """RQ's registry and `ingest_jobs` have to tell the same story.

        The dead-letter list on the ops dashboard is a query against Postgres
        (see routers/ops.py), so a job RQ has given up on that still reads
        `failed` in the database is invisible to the operator who would requeue
        it.
        """
        job = _db_job(ingest_job)

        assert job.status == "dead"
        assert job.is_terminal is True
        assert "connection reset" in job.last_error

    def test_the_queue_is_empty_afterwards(self, exhausted, queue):
        """A job that is neither queued nor scheduled nor running is finished
        with, however unhappily. Anything left here is a leak."""
        assert queue.count == 0


class TestPermanentFailureSkipsTheRetryPolicy:
    """The other half of the classification, proven through RQ rather than
    around it. `test_worker_tasks` shows the task returns normally on a 404;
    this shows what RQ then does with it -- which is nothing, and that is the
    saving."""

    def test_a_404_is_executed_once(
        self, worker, queue, redis, failing_fetch, ingest_job
    ):
        failing_fetch(lambda: PermanentFetchError("HTTP 404 fetching ..."))
        rq_job = _enqueue(queue, ingest_job)

        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)
        _release_retry(queue, redis)
        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

        assert len(failing_fetch.attempts) == 1
        # Finished, not failed: the task caught it and returned a reason
        # string, so RQ has no exception to retry on.
        assert rq_job.get_status() == JobStatus.FINISHED
        assert rq_job.id not in ScheduledJobRegistry(
            queue.name, connection=redis
        ).get_job_ids()

    def test_it_is_dead_on_the_first_attempt(
        self, worker, queue, failing_fetch, ingest_job
    ):
        failing_fetch(lambda: PermanentFetchError("HTTP 404 fetching ..."))
        _enqueue(queue, ingest_job)

        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

        job = _db_job(ingest_job)
        assert job.status == "dead"
        assert job.attempts == 1


class TestStatusReporting:
    """The row has to describe the retry state RQ is actually in.

    Both assertions here failed when this file was written, and neither is
    visible from a direct call to the task -- the fact each one gets wrong is
    about what RQ is *still going to do*, and a unit test has no RQ to be
    relative to. That is the whole argument for driving a real worker.
    """

    def test_a_job_awaiting_retry_is_not_terminal(
        self, worker, queue, redis, failing_fetch, ingest_job
    ):
        """`failed` used to be in TERMINAL_STATUSES.

        `Workspace.tsx`'s JobRow stops polling on `is_terminal`, so that
        froze the row on the label "Failed -- will retry" and never showed the
        retry succeeding seconds later. The label and the polling rule
        disagreed; the label was right.
        """
        _enqueue(queue, ingest_job)

        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

        # RQ is holding it for another go, so by any reading of the word this
        # job is not finished.
        assert ScheduledJobRegistry(queue.name, connection=redis).get_job_ids()

        row = _db_job(ingest_job)
        assert row.status == "failed"
        assert row.is_terminal is False

    def test_a_failed_row_goes_on_to_succeed(
        self, worker, queue, redis, monkeypatch, ingest_job
    ):
        """Which is why `failed` cannot be a state a job never leaves."""
        calls: list[str] = []

        def flaky(url: str) -> str:
            calls.append(url)
            if len(calls) == 1:
                raise TransientFetchError("connection reset")
            return "Senior Engineer\nWe need Python."

        monkeypatch.setattr(tasks, "fetch_posting_text", flaky)
        monkeypatch.setattr(tasks, "_enqueue_scoring", lambda *args, **kwargs: None)

        _enqueue(queue, ingest_job)

        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)
        assert _db_job(ingest_job).status == "failed"

        _release_retry(queue, redis)
        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

        assert _db_job(ingest_job).status == "succeeded"

    def test_the_row_is_not_dead_while_a_retry_is_still_parked(
        self, worker, queue, redis, failing_fetch, ingest_job
    ):
        """`_record_failure` compared `attempts >= MAX_RETRIES`.

        `Retry(max=N)` gives N retries *after* the first run, so the row read
        `dead` for one whole execution while RQ still had a retry scheduled --
        and `ops.py` documents `dead` as "exhausted its retries". An operator
        requeueing at that moment would have put a second worker on a row the
        first was still going to run.
        """
        _enqueue(queue, ingest_job)

        for _ in range(MAX_RETRIES):
            worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

            parked = ScheduledJobRegistry(queue.name, connection=redis).get_job_ids()
            row = _db_job(ingest_job)
            assert not (parked and row.status == "dead"), (
                f"attempt {row.attempts}: row says dead, RQ has a retry scheduled"
            )

            _release_retry(queue, redis)


class TestSuccessAfterRetry:
    """A transient failure that clears is the case the policy exists for.

    Without this, every assertion above is also satisfied by a worker that
    retries and can never succeed.
    """

    def test_a_recovered_fetch_succeeds_and_leaves_no_retry_parked(
        self, worker, queue, redis, monkeypatch, ingest_job
    ):
        calls: list[str] = []

        def flaky(url: str) -> str:
            calls.append(url)
            if len(calls) == 1:
                raise TransientFetchError("connection reset")
            return "Senior Engineer\nWe need Python."

        monkeypatch.setattr(tasks, "fetch_posting_text", flaky)
        # The success path hands off to the scoring queue, which no worker in
        # this test drains. Enqueueing onto a real queue would leave a job for
        # the running compose workers to pick up, so the handoff is stubbed --
        # its routing is covered in test_worker_tasks.TestScoringHandoff.
        monkeypatch.setattr(tasks, "_enqueue_scoring", lambda *args, **kwargs: None)

        rq_job = _enqueue(queue, ingest_job)

        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)
        _release_retry(queue, redis)
        worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)

        assert len(calls) == 2
        assert rq_job.get_status() == JobStatus.FINISHED
        assert rq_job.id not in FailedJobRegistry(
            queue.name, connection=redis
        ).get_job_ids()

        job = _db_job(ingest_job)
        assert job.status == "succeeded"
        assert job.attempts == 2
        assert job.last_error is None
        assert job.job_posting_id is not None
