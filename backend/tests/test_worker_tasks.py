"""Worker task behaviour, especially idempotency.

These call the task functions directly rather than going through RQ. What
matters is that running a task twice produces the same end state as running
it once -- RQ's delivery guarantee is at-least-once, so a redelivered job is
normal operation, not an edge case.
"""

import pytest

from app.db import SessionLocal
from app.extraction import CURRENT_EXTRACTION_VERSION, ExtractedProfile
from app.models import Job, Profile, hash_url
from app.providers import LLMPermanentError, LLMTransientError
from app.workers import tasks


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
        job = Job(
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
            assert profile.extraction_version == CURRENT_EXTRACTION_VERSION
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
            profile.extraction_version = CURRENT_EXTRACTION_VERSION - 1
            db.commit()
        finally:
            db.close()

        assert tasks.extract_profile_task(profile_id) == "extracted"
        assert len(calls) == 2

        db = SessionLocal()
        try:
            assert db.get(Profile, profile_id).extraction_version == (
                CURRENT_EXTRACTION_VERSION
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


class TestProcessJobUrl:
    def test_transitions_to_succeeded_and_counts_the_attempt(self, job_id):
        tasks.process_job_url(job_id)

        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            assert job.status == "succeeded"
            assert job.attempts == 1
            assert job.last_error is None
            assert job.is_terminal is True
        finally:
            db.close()

    def test_second_run_does_not_re_process(self, job_id):
        """At M4 a re-run means a second HTTP request to someone else's
        server. The guard is what makes at-least-once delivery safe."""
        tasks.process_job_url(job_id)
        assert tasks.process_job_url(job_id) == "already_done"

        db = SessionLocal()
        try:
            assert db.get(Job, job_id).attempts == 1  # not incremented twice
        finally:
            db.close()

    def test_failure_is_recorded_and_re_raised_for_rq(self, monkeypatch, job_id):
        """The row and RQ's registry must agree; RQ only learns by exception."""
        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            # _record_failure runs *after* process_job_url incremented the
            # counter, so this is the state of a third and final attempt.
            job.attempts = 3  # == MAX_RETRIES
            db.commit()
        finally:
            db.close()

        tasks._record_failure(job_id, RuntimeError("fetch exploded"))

        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
            assert job.status == "dead"  # attempts (3) >= MAX_RETRIES
            assert "fetch exploded" in job.last_error
            assert job.is_terminal is True
        finally:
            db.close()

    def test_failure_below_retry_limit_is_not_dead(self, job_id):
        tasks._record_failure(job_id, RuntimeError("transient blip"))

        db = SessionLocal()
        try:
            job = db.get(Job, job_id)
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
