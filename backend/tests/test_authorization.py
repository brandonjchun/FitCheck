"""Ownership enforcement: the IDOR tests.

Authentication proves who; authorization proves allowed. Everything here is
about the gap between the two -- a caller with a perfectly valid session
reaching for a row that is not theirs.

`current_user` is overridden so each test can act as a chosen account, but
`owned_profile` and the ownership predicates in the job endpoints are the
real ones. Those are what is under test.

The rule being asserted throughout: **404, not 403.** A 403 confirms the row
exists, which turns any of these endpoints into an oracle for enumerating
other people's ids. Someone who does not own a profile should not be able to
tell it apart from one that was never there.
"""

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import IngestJob, Profile, hash_url


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def intruder(make_user):
    return make_user()


def _make_profile(user_id: int, filename: str = "victim.pdf") -> int:
    db = SessionLocal()
    try:
        profile = Profile(
            user_id=user_id, original_filename=filename, raw_text="resume text"
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return profile.id
    finally:
        db.close()


def _make_job(profile_id: int, url: str = "https://example.com/p") -> int:
    db = SessionLocal()
    try:
        job = IngestJob(
            profile_id=profile_id, url=url, url_hash=hash_url(url), status="queued"
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id
    finally:
        db.close()


class TestProfileOwnership:
    def test_owner_can_read_their_profile(self, client, as_user, owner):
        profile_id = _make_profile(owner.id)
        as_user(owner)

        assert client.get(f"/api/profiles/{profile_id}").status_code == 200

    def test_another_user_gets_404_not_403(self, client, as_user, owner, intruder):
        """The single most common vulnerability in apps shaped like this."""
        profile_id = _make_profile(owner.id)
        as_user(intruder)

        response = client.get(f"/api/profiles/{profile_id}")

        assert response.status_code == 404

    def test_a_foreign_profile_is_indistinguishable_from_a_missing_one(
        self, client, as_user, owner, intruder
    ):
        """Same status *and* same body, or the difference is still an oracle."""
        profile_id = _make_profile(owner.id)
        as_user(intruder)

        foreign = client.get(f"/api/profiles/{profile_id}")
        missing = client.get("/api/profiles/99999999")

        assert foreign.status_code == missing.status_code == 404
        assert foreign.json()["detail"].replace(
            str(profile_id), "X"
        ) == missing.json()["detail"].replace("99999999", "X")

    def test_reextract_is_owner_only(self, client, as_user, owner, intruder):
        """Ownership matters more on this one -- it spends an LLM call.

        Without the predicate, anyone could run up the bill on profile ids
        they guessed.
        """
        profile_id = _make_profile(owner.id)
        as_user(intruder)

        assert client.post(f"/api/profiles/{profile_id}/extract").status_code == 404

    def test_unauthenticated_read_is_401(self, client, owner):
        profile_id = _make_profile(owner.id)

        assert client.get(f"/api/profiles/{profile_id}").status_code == 401


class TestJobOwnership:
    def test_cannot_submit_against_someone_elses_profile(
        self, client, as_user, owner, intruder, monkeypatch
    ):
        """Otherwise anyone can point your workers at URLs of their choosing.

        Not merely a read leak: this enqueues outbound fetches, so it is a
        request amplifier attached to another user's account.
        """
        recorded = []
        monkeypatch.setattr(
            "app.routers.jobs.get_queue",
            lambda name=None: type(
                "Q", (), {"enqueue": lambda self, *a, **k: recorded.append(a)}
            )(),
        )
        profile_id = _make_profile(owner.id)
        as_user(intruder)

        response = client.post(
            "/api/jobs",
            json={"url": "https://example.com/x", "profile_id": profile_id},
        )

        assert response.status_code == 404
        assert recorded == []

    def test_cannot_read_someone_elses_job(self, client, as_user, owner, intruder):
        job_id = _make_job(_make_profile(owner.id))
        as_user(intruder)

        assert client.get(f"/api/jobs/{job_id}").status_code == 404

    def test_owner_can_read_their_job(self, client, as_user, owner):
        job_id = _make_job(_make_profile(owner.id))
        as_user(owner)

        assert client.get(f"/api/jobs/{job_id}").status_code == 200

    def test_listing_excludes_other_users_jobs(
        self, client, as_user, owner, intruder
    ):
        """A listing endpoint is the easiest place to leak everything at once.

        One missing predicate returns every row in the table rather than one.
        """
        _make_job(_make_profile(owner.id), "https://example.com/owner")
        mine = _make_job(_make_profile(intruder.id), "https://example.com/mine")
        as_user(intruder)

        body = client.get("/api/jobs").json()

        assert [job["id"] for job in body] == [mine]

    def test_profile_id_filter_cannot_widen_the_scope(
        self, client, as_user, owner, intruder
    ):
        """?profile_id= narrows within what you own; it must not reach outside.

        The tempting implementation applies the caller's filter and forgets
        that the filter is attacker-controlled.
        """
        victim_profile = _make_profile(owner.id)
        _make_job(victim_profile, "https://example.com/owner")
        as_user(intruder)

        body = client.get(f"/api/jobs?profile_id={victim_profile}").json()

        assert body == []
