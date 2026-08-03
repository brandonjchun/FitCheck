"""Batch URL upload: fan-out, caps, and aggregate progress.

The queue is faked at the boundary, so these assert *what would have been
enqueued* without needing Redis. What matters most here is the queue a batch
lands on -- routing it to `interactive` would put every single-URL submission
behind it, which is the failure the four-queue split exists to prevent.
"""

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.db import SessionLocal
from app.main import app
from app.models import IngestJob, Profile
from app.queues import QUEUE_INGEST, QUEUE_INTERACTIVE

BATCH_URL = "/api/batches"


@dataclass
class RecordingQueue:
    """Stands in for rq.Queue, including the enqueue_many path."""

    name: str
    calls: list = field(default_factory=list)
    fail: bool = False

    def prepare_data(
        self,
        func,
        args=None,
        kwargs=None,
        timeout=None,
        result_ttl=None,
        ttl=None,
        failure_ttl=None,
        retry=None,
        **rest,
    ):
        """Mirrors rq.Queue.prepare_data's real parameter names.

        An earlier version took **kwargs and accepted anything, which hid a
        genuine bug: the endpoint was passing `job_timeout=`, which
        Queue.enqueue takes but prepare_data does not. A fake permissive
        enough to accept a call the real object rejects is worse than no
        fake, so the names are spelled out here on purpose.
        """
        return (func, args, {"timeout": timeout, "retry": retry, **rest})

    def enqueue_many(self, datas):
        if self.fail:
            raise ConnectionError("redis is down")
        self.calls.extend(datas)
        return datas

    def enqueue(self, func, *args, **kwargs):
        if self.fail:
            raise ConnectionError("redis is down")
        self.calls.append((func, args, kwargs))
        return None


@pytest.fixture
def queues(monkeypatch) -> dict[str, RecordingQueue]:
    made: dict[str, RecordingQueue] = {}

    def fake_get_queue(name: str = QUEUE_INTERACTIVE) -> RecordingQueue:
        return made.setdefault(name, RecordingQueue(name))

    monkeypatch.setattr("app.routers.batches.get_queue", fake_get_queue)
    monkeypatch.setattr("app.routers.profiles.get_queue", fake_get_queue)
    return made


@pytest.fixture
def user(make_user, as_user):
    return as_user(make_user())


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


def upload(client, profile_id, lines, filename="urls.txt"):
    body = "\n".join(lines).encode("utf-8")
    return client.post(
        f"{BATCH_URL}?profile_id={profile_id}",
        files={"file": (filename, body, "text/plain")},
    )


class TestCreateBatch:
    def test_returns_202_with_per_line_accounting(self, client, queues, profile_id):
        """202, because nothing has been fetched -- only accepted."""
        response = upload(
            client,
            profile_id,
            [
                "https://e.com/j/1",
                "https://e.com/j/2?utm_source=news",   # normalizes to /j/2
                "https://e.com/j/2",                   # duplicate of the above
                "not a url",                           # rejected
                "",                                    # blank, not counted
            ],
        )

        assert response.status_code == 202
        body = response.json()
        assert body["accepted"] == 2
        assert body["duplicates"] == 1
        assert body["rejected"] == 1

    def test_fans_out_to_the_ingest_queue(self, client, queues, profile_id):
        """The whole point. On `interactive` this would starve single URLs."""
        upload(client, profile_id, [f"https://e.com/j/{i}" for i in range(5)])

        assert len(queues[QUEUE_INGEST].calls) == 5
        assert QUEUE_INTERACTIVE not in queues

    def test_creates_one_job_per_url_linked_to_the_batch(
        self, client, queues, profile_id
    ):
        batch_id = upload(
            client, profile_id, ["https://e.com/j/1", "https://e.com/j/2"]
        ).json()["id"]

        db = SessionLocal()
        try:
            jobs = db.query(IngestJob).filter(IngestJob.batch_id == batch_id).all()
            assert len(jobs) == 2
            assert {job.status for job in jobs} == {"queued"}
        finally:
            db.close()

    def test_stores_normalized_urls(self, client, queues, profile_id):
        """Normalization happens before storage, or the dedupe key is wrong."""
        batch_id = upload(
            client, profile_id, ["https://E.com/j/1?utm_source=x#frag"]
        ).json()["id"]

        db = SessionLocal()
        try:
            job = db.query(IngestJob).filter(IngestJob.batch_id == batch_id).one()
            assert job.url == "https://e.com/j/1"
        finally:
            db.close()

    def test_url_count_is_capped(self, client, queues, profile_id, monkeypatch):
        """Bytes are not a bound on work -- 2 MB of text is ~40,000 URLs."""
        monkeypatch.setattr(settings, "max_urls_per_batch", 3)

        body = upload(
            client, profile_id, [f"https://e.com/j/{i}" for i in range(10)]
        ).json()

        assert body["accepted"] == 3
        # Reported, not silently dropped: a batch that quietly ingests 3 of 10
        # is worse than one that refuses, because nothing says which 7 are gone.
        assert body["rejected"] == 7

    def test_a_list_with_no_usable_urls_is_422(self, client, queues, profile_id):
        response = upload(client, profile_id, ["just some notes", "no links here"])

        assert response.status_code == 422
        assert queues == {}

    def test_non_utf8_upload_is_400(self, client, queues, profile_id):
        """A mis-picked file, not an exotic encoding worth guessing at."""
        response = client.post(
            f"{BATCH_URL}?profile_id={profile_id}",
            files={"file": ("urls.txt", b"\xff\xfe\x00bad", "text/plain")},
        )

        assert response.status_code == 400

    def test_a_byte_order_mark_does_not_eat_the_first_url(
        self, client, queues, profile_id
    ):
        """Notepad and PowerShell both write EF BB BF at the head of a file.

        Decoded as plain utf-8 those bytes become a U+FEFF glued to line one,
        so the first URL stops starting with "http" and is counted as
        unreadable -- while looking perfectly correct to anyone who opens the
        file to check, because the character is invisible.

        The duplicate here is the second half of the damage, and the reason
        this is worth a test rather than a shrug. With line one discarded, the
        repeat of that same URL becomes a *first* occurrence and is accepted,
        so `rejected` and `duplicates` are both wrong rather than the count
        merely being short by one. That silently corrupts the per-line
        accounting this endpoint exists to provide.
        """
        # Written as an escape, not as a literal. A test about an invisible
        # character should not contain one.
        body = "\ufeff" + "\n".join(
            [
                "https://e.com/j/1",
                "https://e.com/j/2",
                "https://e.com/j/1",  # repeat of line one
            ]
        )

        response = client.post(
            f"{BATCH_URL}?profile_id={profile_id}",
            files={"file": ("urls.txt", body.encode("utf-8"), "text/plain")},
        )

        assert response.status_code == 202
        payload = response.json()
        assert payload["accepted"] == 2
        assert payload["duplicates"] == 1
        assert payload["rejected"] == 0

        # And the surviving first URL is the real one, not one carrying an
        # invisible character into the dedupe key.
        db = SessionLocal()
        try:
            urls = {
                job.url
                for job in db.query(IngestJob)
                .filter(IngestJob.batch_id == payload["id"])
                .all()
            }
            assert urls == {"https://e.com/j/1", "https://e.com/j/2"}
        finally:
            db.close()

    def test_urls_already_submitted_do_not_duplicate_the_job(
        self, client, queues, profile_id
    ):
        """Re-fetching a page this profile already has is wasted work, and an
        impolite second request to someone else's server."""
        upload(client, profile_id, ["https://e.com/j/1"])
        second = upload(client, profile_id, ["https://e.com/j/1", "https://e.com/j/2"])

        body = second.json()
        assert body["accepted"] == 1
        assert body["duplicates"] == 1

    def test_redis_down_still_persists_the_batch(self, client, queues, profile_id):
        """A broker outage must not lose the upload.

        Rows survive as `queued` with no rq_job_id -- the signature a requeue
        sweep looks for.
        """
        queues.setdefault(QUEUE_INGEST, RecordingQueue(QUEUE_INGEST)).fail = True

        response = upload(client, profile_id, ["https://e.com/j/1"])

        assert response.status_code == 202
        db = SessionLocal()
        try:
            job = db.query(IngestJob).filter(IngestJob.batch_id == response.json()["id"]).one()
            assert job.status == "queued"
            assert job.rq_job_id is None
        finally:
            db.close()


class TestBatchLimits:
    def test_too_many_open_batches_is_429(self, client, queues, profile_id, monkeypatch):
        """A per-request cap is not a cap if requests are unlimited."""
        monkeypatch.setattr(settings, "max_open_batches_per_user", 1)

        upload(client, profile_id, ["https://e.com/j/1"])
        second = upload(client, profile_id, ["https://e.com/j/2"])

        assert second.status_code == 429


class TestBatchStatus:
    def test_counts_are_derived_from_the_jobs(self, client, queues, profile_id):
        batch_id = upload(
            client, profile_id, [f"https://e.com/j/{i}" for i in range(3)]
        ).json()["id"]

        body = client.get(f"{BATCH_URL}/{batch_id}").json()

        assert body["counts"] == {"queued": 3}
        assert body["is_complete"] is False

    def test_reflects_worker_progress(self, client, queues, profile_id):
        """Progress comes from the job rows, so a worker updating one shows up
        here with nothing else being told about it."""
        batch_id = upload(
            client, profile_id, [f"https://e.com/j/{i}" for i in range(3)]
        ).json()["id"]

        db = SessionLocal()
        try:
            jobs = db.query(IngestJob).filter(IngestJob.batch_id == batch_id).all()
            jobs[0].status = "succeeded"
            jobs[1].status = "dead"
            db.commit()
        finally:
            db.close()

        body = client.get(f"{BATCH_URL}/{batch_id}").json()

        assert body["counts"] == {"succeeded": 1, "dead": 1, "queued": 1}
        assert body["is_complete"] is False

    def test_complete_once_nothing_is_in_flight(self, client, queues, profile_id):
        """`dead` counts as finished. A batch where every item failed is done,
        not perpetually in progress."""
        batch_id = upload(
            client, profile_id, ["https://e.com/j/1", "https://e.com/j/2"]
        ).json()["id"]

        db = SessionLocal()
        try:
            for job in db.query(IngestJob).filter(IngestJob.batch_id == batch_id).all():
                job.status = "dead"
            db.commit()
        finally:
            db.close()

        assert client.get(f"{BATCH_URL}/{batch_id}").json()["is_complete"] is True

    def test_forbids_caching(self, client, queues, profile_id):
        batch_id = upload(client, profile_id, ["https://e.com/j/1"]).json()["id"]

        response = client.get(f"{BATCH_URL}/{batch_id}")

        assert response.headers["cache-control"] == "no-store"

    def test_unknown_batch_is_404(self, client):
        assert client.get(f"{BATCH_URL}/99999999").status_code == 404


class TestBatchOwnership:
    def test_cannot_upload_against_someone_elses_profile(
        self, client, queues, make_user
    ):
        """Not merely a read leak -- this turns one request into N outbound
        fetches attributed to another user's account."""
        stranger = make_user()
        db = SessionLocal()
        try:
            victim = Profile(
                user_id=stranger.id, original_filename="v.pdf", raw_text="text"
            )
            db.add(victim)
            db.commit()
            db.refresh(victim)
            victim_id = victim.id
        finally:
            db.close()

        response = upload(client, victim_id, ["https://e.com/j/1"])

        assert response.status_code == 404
        assert queues == {}

    def test_cannot_read_someone_elses_batch(self, client, queues, profile_id, as_user, make_user):
        batch_id = upload(client, profile_id, ["https://e.com/j/1"]).json()["id"]
        as_user(make_user())

        assert client.get(f"{BATCH_URL}/{batch_id}").status_code == 404

    def test_listing_excludes_other_users_batches(
        self, client, queues, profile_id, as_user, make_user
    ):
        upload(client, profile_id, ["https://e.com/j/1"])
        as_user(make_user())

        assert client.get(BATCH_URL).json() == []

    def test_unauthenticated_upload_is_401(self, profile_id):
        # The profile_id fixture authenticates the app, which is exactly what
        # this test needs undone. Dropping the override is what makes the
        # request genuinely anonymous rather than merely using a fresh client.
        app.dependency_overrides.clear()
        anonymous = TestClient(app, raise_server_exceptions=False)

        response = anonymous.post(
            f"{BATCH_URL}?profile_id={profile_id}",
            files={"file": ("urls.txt", b"https://e.com/j/1", "text/plain")},
        )

        assert response.status_code == 401
