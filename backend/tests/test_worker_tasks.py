"""Worker task behaviour, especially idempotency.

These call the task functions directly rather than going through RQ. What
matters is that running a task twice produces the same end state as running
it once -- RQ's delivery guarantee is at-least-once, so a redelivered job is
normal operation, not an edge case.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from app.db import SessionLocal
from app.extraction import PROFILE_EXTRACTION_VERSION, ExtractedProfile
from app.models import IngestJob, JobPosting, Profile, hash_url
from app.providers import LLMPermanentError, LLMTransientError
from app.queues import QUEUE_INGEST, QUEUE_INTERACTIVE, QUEUE_SCORING
from app.workers import tasks
from app.workers.fetch import PermanentFetchError, TransientFetchError


@pytest.fixture
def profile_id(make_user):
    """A profile owned by a throwaway user.

    Profiles are NOT NULL on user_id since M3, so there is no such thing as
    an unowned one. Cleanup happens by deleting the user, which cascades to
    the profile and through it to any jobs.
    """
    user = make_user()

    db = SessionLocal()
    try:
        profile = Profile(
            user_id=user.id,
            original_filename="t.pdf",
            raw_text="Brandon uses Python.",
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        pid = profile.id
    finally:
        db.close()

    return pid


@pytest.fixture
def job_id(profile_id):
    db = SessionLocal()
    try:
        job = IngestJob(
            profile_id=profile_id,
            url="https://example.com/posting",
            url_hash=hash_url("https://example.com/posting"),
            status="queued",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        jid = job.id
    finally:
        db.close()
    return jid


def _fake_extraction(**overrides) -> ExtractedProfile:
    """Stands in for a completed extract_profile() call.

    Skill names happen to be canonical here, but nothing requires that any
    more: extract_profile returns the model's raw output and canonicalization
    happens on read, in Profile.skills. Normalization itself is covered in
    test_skills.py.
    """
    base = {
        "skills": [
            {
                "name": "Python",
                "years": 3.0,
                "evidence": "uses Python",
                "source": "experience",
            }
        ],
        "total_years_experience": 3.0,
        "seniority": "mid",
        "education": [],
    }
    base.update(overrides)
    return ExtractedProfile(**base)


class TestExtractProfileTask:
    def test_populates_promoted_columns_and_jsonb(self, monkeypatch, profile_id):
        monkeypatch.setattr(
            tasks, "extract_profile", lambda text: _fake_extraction()
        )

        assert tasks.extract_profile_task(profile_id) == "extracted"

        db = SessionLocal()
        try:
            profile = db.get(Profile, profile_id)
            assert profile.seniority == "mid"
            assert float(profile.years_experience) == 3.0
            assert profile.extracted["skills"][0]["name"] == "Python"
            # Stamped in the same commit as the blob, so the two cannot
            # disagree about which rules produced it.
            assert profile.extraction_version == PROFILE_EXTRACTION_VERSION
            assert profile.extraction_is_current is True
        finally:
            db.close()

    def test_stale_extraction_is_redone(self, monkeypatch, profile_id):
        """A version bump has to actually re-extract, or it is decorative.

        Guarding the early return on `extracted is not None` alone would make
        a prompt improvement unactionable: the sweep finds stale rows, the
        task declines to touch them, and the column becomes a label nobody can
        act on.
        """
        calls = []

        def counting(text):
            calls.append(text)
            return _fake_extraction()

        monkeypatch.setattr(tasks, "extract_profile", counting)

        assert tasks.extract_profile_task(profile_id) == "extracted"

        db = SessionLocal()
        try:
            profile = db.get(Profile, profile_id)
            profile.extraction_version = PROFILE_EXTRACTION_VERSION - 1
            db.commit()
        finally:
            db.close()

        assert tasks.extract_profile_task(profile_id) == "extracted"
        assert len(calls) == 2

        db = SessionLocal()
        try:
            assert db.get(Profile, profile_id).extraction_version == (
                PROFILE_EXTRACTION_VERSION
            )
        finally:
            db.close()

    def test_second_run_is_a_no_op(self, monkeypatch, profile_id):
        """Idempotency: a redelivered job must not pay for a second LLM call."""
        calls = []

        def counting(text):
            calls.append(text)
            return _fake_extraction()

        monkeypatch.setattr(tasks, "extract_profile", counting)

        assert tasks.extract_profile_task(profile_id) == "extracted"
        assert tasks.extract_profile_task(profile_id) == "already_done"
        assert len(calls) == 1

    def test_permanent_failure_does_not_raise(self, monkeypatch, profile_id):
        """A bad key can never succeed on retry, so do not ask RQ to retry it."""

        def boom(text):
            raise LLMPermanentError("bad api key")

        monkeypatch.setattr(tasks, "extract_profile", boom)

        assert tasks.extract_profile_task(profile_id) == "permanent_failure"

        db = SessionLocal()
        try:
            assert db.get(Profile, profile_id).extracted is None
        finally:
            db.close()

    def test_transient_failure_propagates_so_rq_retries(self, monkeypatch, profile_id):
        def boom(text):
            raise LLMTransientError("timeout")

        monkeypatch.setattr(tasks, "extract_profile", boom)

        with pytest.raises(LLMTransientError):
            tasks.extract_profile_task(profile_id)

    def test_missing_profile_is_not_an_error(self):
        assert tasks.extract_profile_task(99_999_999) == "profile_missing"


@pytest.fixture
def fetched(monkeypatch):
    """Stub the network at the task boundary.

    fetch_posting_text has its own tests against httpx.MockTransport; what
    these exercise is the state machine around it. Before M5 this file
    needed no such fixture, because the fetch was a no-op -- these two tests
    were quietly passing by not doing anything.
    """

    def install(text: str = "Senior Engineer\nWe need Python."):
        monkeypatch.setattr(tasks, "fetch_posting_text", lambda url: text)
        return text

    install()
    return install


class TestProcessJobUrl:
    def test_transitions_to_succeeded_and_counts_the_attempt(self, fetched, job_id):
        tasks.process_job_url(job_id)

        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            assert job.status == "succeeded"
            assert job.attempts == 1
            assert job.last_error is None
            assert job.is_terminal is True
            # The result record the work produced.
            assert job.job_posting_id is not None
        finally:
            db.close()

    def test_stores_the_fetched_posting(self, fetched, job_id):
        fetched("Staff Engineer\nRust and Postgres.")
        tasks.process_job_url(job_id)

        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            posting = db.get(JobPosting, job.job_posting_id)
            assert "Rust and Postgres." in posting.raw_text
            assert posting.canonical_key.startswith("url:")
            assert posting.content_hash
        finally:
            db.close()

    def test_second_run_does_not_re_process(self, fetched, job_id):
        """A re-run means a second HTTP request to someone else's server.
        The guard is what makes at-least-once delivery safe."""
        tasks.process_job_url(job_id)
        assert tasks.process_job_url(job_id) == "already_done"

        db = SessionLocal()
        try:
            assert db.get(IngestJob, job_id).attempts == 1  # not incremented twice
        finally:
            db.close()

    def test_permanent_failure_is_dead_without_retrying(
        self, monkeypatch, fetched, job_id
    ):
        """Not re-raised, so RQ's retry policy never fires.

        Three attempts at a 404 spend four minutes of worker time
        re-establishing what the first one already settled. And `dead` on
        attempt one is correct -- a 404 is as final then as on attempt three,
        so leaving it `failed` would misreport it to a requeue sweep.
        """

        def gone(url):
            raise PermanentFetchError("HTTP 404 fetching ...")

        monkeypatch.setattr(tasks, "fetch_posting_text", gone)

        assert tasks.process_job_url(job_id) == "permanent_failure"

        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            assert job.status == "dead"
            assert job.attempts == 1
            assert "404" in job.last_error
        finally:
            db.close()

    def test_transient_failure_propagates_so_rq_retries(
        self, monkeypatch, fetched, job_id
    ):
        def flaky(url):
            raise TransientFetchError("timeout")

        monkeypatch.setattr(tasks, "fetch_posting_text", flaky)

        with pytest.raises(TransientFetchError):
            tasks.process_job_url(job_id)

        db = SessionLocal()
        try:
            assert db.get(IngestJob, job_id).status == "failed"  # not dead, will retry
        finally:
            db.close()

    def test_page_with_no_readable_text_is_dead(self, monkeypatch, fetched, job_id):
        """The HTML analogue of a scanned PDF -- usually a JavaScript-rendered
        posting this fetcher cannot see. Retrying re-downloads the same
        empty document."""
        monkeypatch.setattr(tasks, "fetch_posting_text", lambda url: "   \n  ")

        assert tasks.process_job_url(job_id) == "no_content"

        db = SessionLocal()
        try:
            assert db.get(IngestJob, job_id).status == "dead"
        finally:
            db.close()

    def test_two_jobs_for_one_url_share_a_posting(self, fetched, make_user):
        """Global dedupe on canonical_key: the catalog holds one row.

        Two users submitting the same posting must not create two postings,
        or the crawler will later create a third.
        """
        url = "https://example.com/shared-posting"
        job_ids = []

        for _ in range(2):
            user = make_user()
            db = SessionLocal()
            try:
                profile = Profile(
                    user_id=user.id, original_filename="r.pdf", raw_text="text"
                )
                db.add(profile)
                db.commit()
                db.refresh(profile)
                job = IngestJob(
                    profile_id=profile.id,
                    url=url,
                    url_hash=hash_url(url),
                    status="queued",
                )
                db.add(job)
                db.commit()
                db.refresh(job)
                job_ids.append(job.id)
            finally:
                db.close()

        for job_id in job_ids:
            tasks.process_job_url(job_id)

        db = SessionLocal()
        try:
            postings = {db.get(IngestJob, jid).job_posting_id for jid in job_ids}
            assert len(postings) == 1
        finally:
            db.close()

    def test_tracking_params_collapse_onto_one_posting(self, fetched, make_user):
        """The reason canonical_key normalizes and hash_url does not.

        Two submissions of one posting under different campaign URLs are one
        posting -- otherwise the catalog doubles and so does the extraction
        bill at M7.
        """
        urls = [
            "https://example.com/p/9?utm_source=news",
            "https://example.com/p/9?utm_source=twitter",
        ]
        job_ids = []

        for url in urls:
            user = make_user()
            db = SessionLocal()
            try:
                profile = Profile(
                    user_id=user.id, original_filename="r.pdf", raw_text="text"
                )
                db.add(profile)
                db.commit()
                db.refresh(profile)
                job = IngestJob(
                    profile_id=profile.id,
                    url=url,
                    url_hash=hash_url(url),
                    status="queued",
                )
                db.add(job)
                db.commit()
                db.refresh(job)
                job_ids.append(job.id)
            finally:
                db.close()

        for job_id in job_ids:
            tasks.process_job_url(job_id)

        db = SessionLocal()
        try:
            postings = {db.get(IngestJob, jid).job_posting_id for jid in job_ids}
            assert len(postings) == 1
        finally:
            db.close()

    def test_failure_is_recorded_and_re_raised_for_rq(self, monkeypatch, job_id):
        """The row and RQ's registry must agree; RQ only learns by exception."""
        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            # _record_failure runs *after* process_job_url incremented the
            # counter, so this is the state of a third and final attempt.
            job.attempts = 3  # == MAX_RETRIES
            db.commit()
        finally:
            db.close()

        tasks._record_failure(job_id, RuntimeError("fetch exploded"))

        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            assert job.status == "dead"  # attempts (3) >= MAX_RETRIES
            assert "fetch exploded" in job.last_error
            assert job.is_terminal is True
        finally:
            db.close()

    def test_failure_below_retry_limit_is_not_dead(self, job_id):
        tasks._record_failure(job_id, RuntimeError("transient blip"))

        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            assert job.status == "failed"
            assert job.is_terminal is True
        finally:
            db.close()

    def test_missing_job_is_not_an_error(self):
        assert tasks.process_job_url(99_999_999) == "job_missing"


class TestHashUrl:
    def test_is_stable_and_differs_per_url(self):
        assert hash_url("https://a.com") == hash_url("https://a.com")
        assert hash_url("https://a.com") != hash_url("https://b.com")

    def test_is_not_normalized(self):
        """Documented behaviour: a trailing slash makes a different job.

        Errs toward re-fetching rather than silently returning the wrong
        cached posting, because correct URL normalization is genuinely hard
        (?page=2 matters, ?utm_source=x does not).
        """
        assert hash_url("https://a.com/x") != hash_url("https://a.com/x/")


@dataclass
class RecordingQueue:
    """Records enqueues instead of talking to Redis."""

    name: str
    fail: bool = False
    calls: list = field(default_factory=list)

    def enqueue(self, func, *args, **kwargs):
        if self.fail:
            raise ConnectionError("redis is down")
        self.calls.append((func, args, kwargs))
        return SimpleNamespace(id="rq-fake", origin=self.name)


@pytest.fixture
def scoring_queues(monkeypatch):
    """One recorder per queue name, so routing can be asserted.

    Keyed by name rather than shared: a single recorder would pass whether or
    not the handoff picked the right queue, which is the whole point.
    """
    made: dict[str, RecordingQueue] = {}

    def fake_get_queue(name: str = "interactive") -> RecordingQueue:
        return made.setdefault(name, RecordingQueue(name))

    # Patched on app.queues rather than on tasks, because _enqueue_scoring
    # imports get_queue inside the function body -- the lookup happens at
    # call time, so the module attribute is what it resolves against.
    monkeypatch.setattr("app.queues.get_queue", fake_get_queue)
    return made


class TestScoringHandoff:
    """A fetched posting has to reach the scorer, or Path A stops halfway.

    This is the M6 requeue bug's exact shape -- an enqueue with the wrong
    queue and no test asserting which one. It went unnoticed there because
    everything still ran, just in the wrong lane. Here the failure is worse:
    nothing runs at all and the job still reports `succeeded`.
    """

    def test_a_successful_fetch_enqueues_scoring(self, fetched, scoring_queues, job_id):
        tasks.process_job_url(job_id)

        assert len(scoring_queues[QUEUE_SCORING].calls) == 1

    def test_it_does_not_land_on_the_interactive_queue(
        self, fetched, scoring_queues, job_id
    ):
        """Scoring is an LLM call plus model inference. On `interactive` it
        would sit in front of URL submissions that take milliseconds to
        accept -- the head-of-line problem the four-queue split exists for,
        arriving one lane further down."""
        tasks.process_job_url(job_id)

        assert QUEUE_INTERACTIVE not in scoring_queues
        assert QUEUE_INGEST not in scoring_queues

    def test_it_passes_the_profile_and_the_posting(
        self, fetched, scoring_queues, job_id
    ):
        """Both ids, in that order. Swapped, the scorer would look up a
        profile by a posting id -- which usually finds nothing and returns
        `profile_or_posting_missing`, so the job succeeds and no match is
        ever written."""
        tasks.process_job_url(job_id)

        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            expected = (job.profile_id, job.job_posting_id)
        finally:
            db.close()

        func, args, _ = scoring_queues[QUEUE_SCORING].calls[0]
        assert func == "app.workers.tasks.score_posting_for_profile"
        assert args == expected

    def test_a_permanent_failure_enqueues_nothing(
        self, monkeypatch, scoring_queues, job_id
    ):
        """There is no posting to score. Enqueueing anyway would spend an LLM
        call learning that."""

        def boom(url):
            raise PermanentFetchError("HTTP 404")

        monkeypatch.setattr(tasks, "fetch_posting_text", boom)

        tasks.process_job_url(job_id)

        assert scoring_queues == {}

    def test_an_empty_page_enqueues_nothing(self, monkeypatch, scoring_queues, job_id):
        monkeypatch.setattr(tasks, "fetch_posting_text", lambda url: "   ")

        tasks.process_job_url(job_id)

        assert scoring_queues == {}

    def test_a_broker_outage_does_not_fail_the_fetch(
        self, fetched, monkeypatch, job_id
    ):
        """The fetch genuinely succeeded and the posting is durable.

        Raising here would hand the job back to the retry policy, which would
        re-request a page we already have -- a request to a third party that
        cannot be taken back. A missing `matches` row is recoverable by a
        sweep; an extra outbound fetch is not.
        """
        monkeypatch.setattr(
            "app.queues.get_queue",
            lambda name="interactive": RecordingQueue(name, fail=True),
        )

        assert tasks.process_job_url(job_id) == "fetched"

        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            assert job.status == "succeeded"
            assert job.job_posting_id is not None
        finally:
            db.close()
