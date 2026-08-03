"""Job submission, dedupe, and polling.

The queue is faked at the boundary: `app.routers.jobs.get_queue` is patched
with a recorder, so these tests assert *what would have been enqueued*
without needing Redis running. What actually executes the work is covered in
test_worker_tasks.py, which calls the task functions directly.
"""

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import Job, Profile


@dataclass
class FakeRQJob:
    id: str = "rq-fake-id"


@dataclass
class RecordingQueue:
    """Stands in for rq.Queue, capturing enqueue calls."""

    calls: list[tuple] = field(default_factory=list)
    fail: bool = False

    def enqueue(self, func, *args, **kwargs):
        if self.fail:
            raise ConnectionError("redis is down")
        self.calls.append((func, args, kwargs))
        return FakeRQJob()


@pytest.fixture
def queue(monkeypatch) -> RecordingQueue:
    q = RecordingQueue()
    monkeypatch.setattr("app.routers.jobs.get_queue", lambda: q)
    monkeypatch.setattr("app.routers.profiles.get_queue", lambda: q)
    return q


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def profile_id() -> int:
    """A real profiles row, cleaned up afterwards.

    Jobs carry a FK to profiles with ON DELETE CASCADE, so deleting the
    profile removes any jobs the test created.
    """
    db = SessionLocal()
    try:
        profile = Profile(original_filename="t.pdf", raw_text="resume text")
        db.add(profile)
        db.commit()
        db.refresh(profile)
        pid = profile.id
    finally:
        db.close()

    yield pid

    db = SessionLocal()
    try:
        obj = db.get(Profile, pid)
        if obj is not None:
            db.delete(obj)
            db.commit()
    finally:
        db.close()


class TestSubmitJob:
    def test_returns_202_not_201(self, client, queue, profile_id):
        """202 Accepted, because the work has not happened yet.

        201 would promise a completed resource at the returned location.
        Nothing has been fetched at this point -- only accepted.
        """
        response = client.post(
            "/api/jobs",
            json={"url": "https://example.com/jobs/1", "profile_id": profile_id},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert body["attempts"] == 0
        assert body["is_terminal"] is False

    def test_enqueues_by_dotted_path_with_retry_policy(self, client, queue, profile_id):
        client.post(
            "/api/jobs",
            json={"url": "https://example.com/jobs/2", "profile_id": profile_id},
        )
        assert len(queue.calls) == 1
        func, args, kwargs = queue.calls[0]

        # Enqueued by string, not by reference: the worker imports it by path
        # in a separate process and never shares memory with the API.
        assert func == "app.workers.tasks.process_job_url"
        assert kwargs["job_timeout"] == 120
        assert kwargs["retry"].max == 3
        assert kwargs["retry"].intervals == [10, 60, 300]

    def test_malformed_url_rejected_before_any_row_is_written(
        self, client, queue, profile_id
    ):
        response = client.post(
            "/api/jobs", json={"url": "not-a-url", "profile_id": profile_id}
        )
        assert response.status_code == 422
        assert queue.calls == []

    def test_unknown_profile_returns_404(self, client, queue):
        response = client.post(
            "/api/jobs",
            json={"url": "https://example.com/jobs/3", "profile_id": 99_999_999},
        )
        assert response.status_code == 404
        assert queue.calls == []

    def test_duplicate_url_returns_existing_job_without_re_enqueueing(
        self, client, queue, profile_id
    ):
        """Submitting the same URL twice must not scrape the same page twice."""
        payload = {"url": "https://example.com/jobs/dup", "profile_id": profile_id}

        first = client.post("/api/jobs", json=payload)
        second = client.post("/api/jobs", json=payload)

        assert first.json()["id"] == second.json()["id"]
        assert len(queue.calls) == 1

    def test_same_url_for_different_profile_is_a_separate_job(
        self, client, queue, profile_id
    ):
        """The dedupe key is (profile_id, url_hash), not url alone.

        Two candidates applying to the same posting are two jobs.
        """
        db = SessionLocal()
        try:
            other = Profile(original_filename="o.pdf", raw_text="other")
            db.add(other)
            db.commit()
            db.refresh(other)
            other_id = other.id
        finally:
            db.close()

        url = "https://example.com/jobs/shared"
        a = client.post("/api/jobs", json={"url": url, "profile_id": profile_id})
        b = client.post("/api/jobs", json={"url": url, "profile_id": other_id})

        assert a.json()["id"] != b.json()["id"]
        assert len(queue.calls) == 2

        db = SessionLocal()
        try:
            db.delete(db.get(Profile, other_id))
            db.commit()
        finally:
            db.close()

    def test_redis_down_still_persists_the_job(self, client, queue, profile_id):
        """A broker outage must not lose the submission.

        The row survives in `queued` with no rq_job_id -- exactly the
        signature a requeue sweep looks for.
        """
        queue.fail = True

        response = client.post(
            "/api/jobs",
            json={"url": "https://example.com/jobs/noredis", "profile_id": profile_id},
        )
        assert response.status_code == 202

        db = SessionLocal()
        try:
            job = db.get(Job, response.json()["id"])
            assert job is not None
            assert job.status == "queued"
            assert job.rq_job_id is None
        finally:
            db.close()


class TestPollJob:
    def test_get_job_returns_state_and_forbids_caching(
        self, client, queue, profile_id
    ):
        created = client.post(
            "/api/jobs",
            json={"url": "https://example.com/jobs/poll", "profile_id": profile_id},
        ).json()

        response = client.get(f"/api/jobs/{created['id']}")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["status"] == "queued"

    def test_unknown_job_returns_404(self, client):
        assert client.get("/api/jobs/99999999").status_code == 404

    def test_list_filters_by_profile(self, client, queue, profile_id):
        client.post(
            "/api/jobs",
            json={"url": "https://example.com/jobs/list1", "profile_id": profile_id},
        )
        response = client.get(f"/api/jobs?profile_id={profile_id}")
        assert response.status_code == 200
        assert all(j["profile_id"] == profile_id for j in response.json())
