"""Listing, activating, and deleting resumes.

These three endpoints are what make an account durable. Before them a profile
was reachable only through an id the client still happened to be holding, so
a page refresh stranded every upload -- the rows were there, nothing could
ask for them.

The invariant under test throughout is `profiles_one_active_per_user`: a
partial unique index permitting exactly one active row per user. It is
asserted against the *database* rather than against a response body in every
case where it matters, because the response is what the handler believes and
the index is what actually happened.
"""

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import IngestJob, Profile


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def make_profile():
    """Insert profile rows directly, bypassing upload.

    Direct inserts rather than POST /api/profiles because upload enqueues an
    extraction job, and these tests are about the lifecycle around a profile
    rather than about producing one. Cleanup rides on the `make_user`
    fixture, which cascades.
    """

    def _make(user, filename="resume.pdf", text="Python and SQL.", active=False, extracted=None):
        db = SessionLocal()
        try:
            profile = Profile(
                user_id=user.id,
                original_filename=filename,
                raw_text=text,
                is_active=active,
                extracted=extracted,
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)
            db.expunge(profile)
            return profile
        finally:
            db.close()

    return _make


def active_ids(user_id: int) -> list[int]:
    """Every active profile id for a user. Length > 1 means the index failed."""
    db = SessionLocal()
    try:
        rows = db.query(Profile.id).filter(Profile.user_id == user_id, Profile.is_active).all()
        return [r[0] for r in rows]
    finally:
        db.close()


EXTRACTED = {
    "skills": [
        {"name": "Python", "years": 3.0, "evidence": "Built the ingest pipeline in Python", "source": "experience"},
        {"name": "SQL", "years": 2.0, "evidence": "Wrote the reporting queries", "source": "experience"},
    ],
    "total_years_experience": 3.0,
    "seniority": "mid",
    "education": [],
}


class TestListProfiles:
    def test_empty_for_a_user_with_no_uploads(self, client, as_user, make_user):
        as_user(make_user())
        response = client.get("/api/profiles")
        assert response.status_code == 200
        assert response.json() == []

    def test_returns_only_your_own(self, client, as_user, make_user, make_profile):
        mine = make_user()
        theirs = make_user()
        make_profile(mine, filename="mine.pdf")
        make_profile(theirs, filename="theirs.pdf")

        as_user(mine)
        names = [p["filename"] for p in client.get("/api/profiles").json()]

        assert names == ["mine.pdf"]

    def test_newest_first(self, client, as_user, make_user, make_profile):
        user = make_user()
        first = make_profile(user, filename="v1.pdf")
        second = make_profile(user, filename="v2.pdf")
        third = make_profile(user, filename="v3.pdf")

        as_user(user)
        ids = [p["id"] for p in client.get("/api/profiles").json()]

        assert ids == [third.id, second.id, first.id]

    def test_omits_raw_text(self, client, as_user, make_user, make_profile):
        """The reason ProfileSummary exists at all.

        Shipping the full text of every resume a user ever uploaded just to
        draw a version picker is fine with three rows and absurd with thirty.
        """
        user = make_user()
        make_profile(user, text="a very long resume " * 500)

        as_user(user)
        payload = client.get("/api/profiles").json()[0]

        assert "raw_text" not in payload
        assert "skills" not in payload
        assert payload["characters"] == len("a very long resume " * 500)

    def test_skill_count_without_the_skills(self, client, as_user, make_user, make_profile):
        user = make_user()
        make_profile(user, extracted=EXTRACTED)

        as_user(user)
        payload = client.get("/api/profiles").json()[0]

        assert payload["skill_count"] == 2
        assert payload["extraction_ok"] is True

    def test_unextracted_profile_reports_zero_not_null(
        self, client, as_user, make_user, make_profile
    ):
        """`extraction_ok` is what distinguishes these, not the count.

        A profile whose extraction never ran and one whose extraction found
        nothing both count zero skills. Only the flag separates them.
        """
        user = make_user()
        make_profile(user, extracted=None)

        as_user(user)
        payload = client.get("/api/profiles").json()[0]

        assert payload["skill_count"] == 0
        assert payload["extraction_ok"] is False

    def test_requires_authentication(self):
        anonymous = TestClient(app, raise_server_exceptions=False)
        assert anonymous.get("/api/profiles").status_code == 401


class TestActivateProfile:
    def test_promotes_and_demotes_in_one_move(
        self, client, as_user, make_user, make_profile
    ):
        user = make_user()
        old = make_profile(user, filename="old.pdf", active=True)
        new = make_profile(user, filename="new.pdf", active=False)

        as_user(user)
        response = client.post(f"/api/profiles/{new.id}/activate")

        assert response.status_code == 200
        assert response.json()["is_active"] is True
        # Asserted against the database, not the response: the index is what
        # actually enforces this and the handler only believes it did.
        assert active_ids(user.id) == [new.id]
        assert old.id not in active_ids(user.id)

    def test_never_leaves_two_active(self, client, as_user, make_user, make_profile):
        user = make_user()
        make_profile(user, filename="a.pdf", active=True)
        b = make_profile(user, filename="b.pdf")
        c = make_profile(user, filename="c.pdf")

        as_user(user)
        client.post(f"/api/profiles/{b.id}/activate")
        client.post(f"/api/profiles/{c.id}/activate")

        assert active_ids(user.id) == [c.id]

    def test_activating_the_active_one_is_a_no_op(
        self, client, as_user, make_user, make_profile
    ):
        """Idempotent by design -- a double-click is not an error."""
        user = make_user()
        current = make_profile(user, active=True)

        as_user(user)
        response = client.post(f"/api/profiles/{current.id}/activate")

        assert response.status_code == 200
        assert active_ids(user.id) == [current.id]

    def test_promotes_when_nothing_was_active(
        self, client, as_user, make_user, make_profile
    ):
        user = make_user()
        orphan = make_profile(user, active=False)

        as_user(user)
        assert client.post(f"/api/profiles/{orphan.id}/activate").status_code == 200
        assert active_ids(user.id) == [orphan.id]

    def test_cannot_activate_someone_elses(
        self, client, as_user, make_user, make_profile
    ):
        """404, not 403 -- see security.owned_profile for why."""
        mine = make_user()
        theirs = make_user()
        target = make_profile(theirs, active=True)

        as_user(mine)
        assert client.post(f"/api/profiles/{target.id}/activate").status_code == 404
        # And it stayed active for its actual owner.
        assert active_ids(theirs.id) == [target.id]

    def test_requires_authentication(self, make_user, make_profile):
        profile = make_profile(make_user())
        anonymous = TestClient(app, raise_server_exceptions=False)
        assert anonymous.post(f"/api/profiles/{profile.id}/activate").status_code == 401


class TestDeleteProfile:
    def test_removes_the_row(self, client, as_user, make_user, make_profile):
        user = make_user()
        profile = make_profile(user)

        as_user(user)
        assert client.delete(f"/api/profiles/{profile.id}").status_code == 204

        db = SessionLocal()
        try:
            assert db.get(Profile, profile.id) is None
        finally:
            db.close()

    def test_deleting_the_active_one_promotes_the_newest_survivor(
        self, client, as_user, make_user, make_profile
    ):
        """The alternative is an account with resumes and no feed.

        Leaving zero active rows is a state with no honest UI, and its only
        exit would be an activate call the client has to know to make.
        """
        user = make_user()
        older = make_profile(user, filename="older.pdf")
        newer = make_profile(user, filename="newer.pdf")
        current = make_profile(user, filename="current.pdf", active=True)

        as_user(user)
        assert client.delete(f"/api/profiles/{current.id}").status_code == 204

        assert active_ids(user.id) == [newer.id]
        assert older.id not in active_ids(user.id)

    def test_deleting_an_inactive_one_leaves_the_active_alone(
        self, client, as_user, make_user, make_profile
    ):
        user = make_user()
        keep = make_profile(user, filename="keep.pdf", active=True)
        drop = make_profile(user, filename="drop.pdf")

        as_user(user)
        assert client.delete(f"/api/profiles/{drop.id}").status_code == 204
        assert active_ids(user.id) == [keep.id]

    def test_deleting_the_last_one_leaves_none_active(
        self, client, as_user, make_user, make_profile
    ):
        """No survivor to promote. Must not raise looking for one."""
        user = make_user()
        only = make_profile(user, active=True)

        as_user(user)
        assert client.delete(f"/api/profiles/{only.id}").status_code == 204
        assert active_ids(user.id) == []

    def test_cascades_to_submitted_jobs(self, client, as_user, make_user, make_profile):
        """ON DELETE CASCADE on ingest_jobs.profile_id, not code in the handler.

        A job scored against a resume that no longer exists can be neither
        re-scored nor explained, and leaving it behind would have the ops
        dashboard counting work whose subject is gone.
        """
        user = make_user()
        profile = make_profile(user, active=True)

        db = SessionLocal()
        try:
            job = IngestJob(
                profile_id=profile.id,
                url="https://example.com/posting",
                url_hash=hashlib.sha256(b"https://example.com/posting").hexdigest(),
                status="queued",
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            job_id = job.id
        finally:
            db.close()

        as_user(user)
        assert client.delete(f"/api/profiles/{profile.id}").status_code == 204

        db = SessionLocal()
        try:
            assert db.get(IngestJob, job_id) is None
        finally:
            db.close()

    def test_cannot_delete_someone_elses(self, client, as_user, make_user, make_profile):
        mine = make_user()
        theirs = make_user()
        target = make_profile(theirs)

        as_user(mine)
        assert client.delete(f"/api/profiles/{target.id}").status_code == 404

        db = SessionLocal()
        try:
            assert db.get(Profile, target.id) is not None
        finally:
            db.close()

    def test_requires_authentication(self, make_user, make_profile):
        profile = make_profile(make_user())
        anonymous = TestClient(app, raise_server_exceptions=False)
        assert anonymous.delete(f"/api/profiles/{profile.id}").status_code == 401

        db = SessionLocal()
        try:
            assert db.get(Profile, profile.id) is not None
        finally:
            db.close()
