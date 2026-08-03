"""Operational endpoints: overview, dead letter, and requeue.

This module existed nowhere until M6, which mattered more than the usual
coverage argument: `requeue` is the only endpoint in the system that mutates
state on somebody's behalf without an ownership predicate, and it carried two
bugs that a passing test suite said nothing about.

The queue is faked at the boundary so the assertions can be about *which*
queue and *what policy*, which is the whole substance of what was wrong.
"""

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import IngestJob, Profile, UrlBatch, User, hash_url
from app.queues import MAX_RETRIES, QUEUE_INGEST, QUEUE_INTERACTIVE

OPS = "/api/ops"


@dataclass
class FakeRQJob:
    id: str = "rq-fake"
    origin: str = "unknown"


@dataclass
class RecordingQueue:
    name: str
    calls: list = field(default_factory=list)

    def enqueue(self, func, *args, **kwargs):
        self.calls.append((func, args, kwargs))
        return FakeRQJob(origin=self.name)


@pytest.fixture
def queues(monkeypatch) -> dict[str, RecordingQueue]:
    """One recorder per queue name, so tests can assert on routing.

    Keyed by name rather than shared: a single recorder would pass whether or
    not requeue put the job back where it came from, which is exactly the bug
    this file was written for.
    """
    made: dict[str, RecordingQueue] = {}

    def fake_get_queue(name: str = QUEUE_INTERACTIVE) -> RecordingQueue:
        return made.setdefault(name, RecordingQueue(name))

    monkeypatch.setattr("app.routers.ops.get_queue", fake_get_queue)
    return made


@pytest.fixture
def promote():
    """Grant operator access to an existing user.

    Duplicated from test_ops_authorization.py rather than shared, because
    that module owns the question of *who may call these* and this one owns
    what they do once through the door. A fixture in conftest would couple
    the two files' setup to each other for no gain.
    """

    def _promote(user):
        db = SessionLocal()
        try:
            row = db.get(User, user.id)
            row.is_admin = True
            db.commit()
        finally:
            db.close()
        user.is_admin = True
        return user

    return _promote


@pytest.fixture
def user(make_user, as_user, promote):
    """An operator. These endpoints are admin-gated (see require_admin).

    Every test below is about behaviour rather than access, so they all act
    as somebody already allowed in.
    """
    return as_user(promote(make_user()))


@pytest.fixture
def client(user) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def profile_id(user) -> int:
    db = SessionLocal()
    try:
        profile = Profile(
            user_id=user.id, original_filename="r.pdf", raw_text="resume text"
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return profile.id
    finally:
        db.close()


def make_job(profile_id: int, *, status="dead", url=None, batch_id=None) -> int:
    url = url or f"https://example.com/j/{profile_id}-{status}-{batch_id}"
    db = SessionLocal()
    try:
        job = IngestJob(
            profile_id=profile_id,
            batch_id=batch_id,
            url=url,
            url_hash=hash_url(url),
            status=status,
            attempts=3,
            last_error="PermanentFetchError: HTTP 404",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id
    finally:
        db.close()


def make_batch(user_id: int, profile_id: int) -> int:
    db = SessionLocal()
    try:
        batch = UrlBatch(
            user_id=user_id,
            profile_id=profile_id,
            original_filename="urls.txt",
            total_urls=1,
        )
        db.add(batch)
        db.commit()
        db.refresh(batch)
        return batch.id
    finally:
        db.close()


# Who may reach these endpoints is covered in test_ops_authorization.py --
# anonymous, non-admin, and admin, plus the default-restrictive flag. This
# module deliberately does not restate it, because two files asserting the
# same rule means one of them silently stops mattering when the rule moves.


class TestOverview:
    def test_reports_every_declared_queue(self, client) -> None:
        body = client.get(f"{OPS}/overview").json()

        names = {q["name"] for q in body["queues"]}
        assert {"interactive", "ingest", "scoring", "discovery"} <= names

    def test_declared_flag_marks_known_queues(self, client) -> None:
        """A queue in Redis but not in QUEUE_NAMES is work left behind by a
        rename -- not retrying, not failing, and invisible everywhere else."""
        body = client.get(f"{OPS}/overview").json()

        for queue in body["queues"]:
            if queue["name"] in {"interactive", "ingest", "scoring", "discovery"}:
                assert queue["declared"] is True

    def test_counts_jobs_by_status(self, client, profile_id) -> None:
        make_job(profile_id, status="dead")

        body = client.get(f"{OPS}/overview").json()

        counts = {row["status"]: row["count"] for row in body["jobs_by_status"]}
        assert counts.get("dead", 0) >= 1

    def test_exposes_the_queue_policy_constants(self, client) -> None:
        """The dashboard should not hardcode what the backend already knows."""
        body = client.get(f"{OPS}/overview").json()

        assert body["job_timeout_seconds"] == 120
        assert body["result_ttl_seconds"] > 0
        assert body["failure_ttl_seconds"] > body["result_ttl_seconds"]


class TestDeadLetter:
    def test_includes_dead_and_failed(self, client, profile_id) -> None:
        dead = make_job(profile_id, status="dead", url="https://e.com/dead")
        failed = make_job(profile_id, status="failed", url="https://e.com/failed")

        ids = {item["id"] for item in client.get(f"{OPS}/dead-letter").json()}

        assert {dead, failed} <= ids

    def test_excludes_healthy_jobs(self, client, profile_id) -> None:
        queued = make_job(profile_id, status="queued", url="https://e.com/queued")
        ok = make_job(profile_id, status="succeeded", url="https://e.com/ok")

        ids = {item["id"] for item in client.get(f"{OPS}/dead-letter").json()}

        assert queued not in ids
        assert ok not in ids

    def test_carries_the_error_and_attempt_count(self, client, profile_id) -> None:
        """The two things an operator needs before deciding to requeue."""
        job_id = make_job(profile_id, status="dead")

        item = next(
            i for i in client.get(f"{OPS}/dead-letter").json() if i["id"] == job_id
        )

        assert item["attempts"] == 3
        assert "404" in item["last_error"]

    def test_limit_is_capped(self, client) -> None:
        """An unbounded limit is a way to ask the database for everything."""
        assert len(client.get(f"{OPS}/dead-letter?limit=100000").json()) <= 200


class TestRequeue:
    def test_resets_the_row_for_another_attempt(self, client, queues, profile_id):
        """A requeue is an operator asserting the cause was fixed.

        Leaving `attempts` at its exhausted value would dead-letter the job
        again on its first tick, which makes the button look broken.
        """
        job_id = make_job(profile_id, status="dead")

        response = client.post(f"{OPS}/jobs/{job_id}/requeue")

        assert response.status_code == 202
        db = SessionLocal()
        try:
            job = db.get(IngestJob, job_id)
            assert job.status == "queued"
            assert job.attempts == 0
            assert job.last_error is None
            assert job.rq_job_id == "rq-fake"
        finally:
            db.close()

    def test_single_submission_goes_back_to_interactive(
        self, client, queues, profile_id
    ):
        job_id = make_job(profile_id, status="dead", batch_id=None)

        client.post(f"{OPS}/jobs/{job_id}/requeue")

        assert len(queues[QUEUE_INTERACTIVE].calls) == 1
        assert QUEUE_INGEST not in queues

    def test_batch_job_goes_back_to_ingest(self, client, queues, user, profile_id):
        """The bug this file was written for.

        Requeue used to call get_queue() bare, which defaults to
        `interactive`. An operator clearing fifty dead batch items would put
        fifty bulk fetches ahead of every user submission -- the head-of-line
        blocking the four-queue split exists to prevent, arriving through the
        back door.
        """
        batch_id = make_batch(user.id, profile_id)
        job_id = make_job(profile_id, status="dead", batch_id=batch_id)

        client.post(f"{OPS}/jobs/{job_id}/requeue")

        assert len(queues[QUEUE_INGEST].calls) == 1
        assert QUEUE_INTERACTIVE not in queues

    def test_carries_the_retry_policy(self, client, queues, profile_id):
        """The second bug. RQ attaches retry to the job at enqueue time, not
        to the function, so a requeue that omits it produces a job that fails
        permanently on its first transient error -- behaving differently from
        the same job on its original submission.
        """
        job_id = make_job(profile_id, status="dead")

        client.post(f"{OPS}/jobs/{job_id}/requeue")

        _, _, kwargs = queues[QUEUE_INTERACTIVE].calls[0]
        assert kwargs["retry"].max == MAX_RETRIES
        assert len(kwargs["retry"].intervals) == MAX_RETRIES
        assert kwargs["job_timeout"] == 120

    def test_reports_the_queue_it_landed_on(self, client, queues, user, profile_id):
        batch_id = make_batch(user.id, profile_id)
        job_id = make_job(profile_id, status="dead", batch_id=batch_id)

        assert client.post(f"{OPS}/jobs/{job_id}/requeue").json()["queue"] == (
            QUEUE_INGEST
        )

    def test_unknown_job_is_404(self, client, queues):
        assert client.post(f"{OPS}/jobs/99999999/requeue").status_code == 404

    @pytest.mark.parametrize("status", ["queued", "running", "succeeded"])
    def test_only_failed_or_dead_can_be_requeued(
        self, client, queues, profile_id, status
    ):
        """Requeueing a running job puts a second worker on the same row.

        The handlers are idempotent so it converges, but it wastes a slot and
        muddies the audit trail.
        """
        job_id = make_job(profile_id, status=status)

        response = client.post(f"{OPS}/jobs/{job_id}/requeue")

        assert response.status_code == 409
        assert queues == {}

    def test_conflict_message_does_not_leak_the_internal_model_name(
        self, client, queues, profile_id
    ):
        """The table is `ingest_jobs`; the API is /api/jobs.

        A rename of internal vocabulary should not reach a client-facing
        string -- this caught exactly that leak once already.
        """
        job_id = make_job(profile_id, status="running")

        detail = client.post(f"{OPS}/jobs/{job_id}/requeue").json()["detail"]

        assert "IngestJob" not in detail
        assert detail.startswith(f"Job {job_id}")
