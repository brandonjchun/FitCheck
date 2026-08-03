"""Reading matches: ownership, ordering, and the breakdown contract.

The scoring arithmetic is covered in test_scoring.py. This file is about the
HTTP surface, and most of it is about who may see what -- a match names a
profile, a profile names a resume, and a resume is the most personal thing
this system stores.
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import JobPosting, Match, Profile, User
from app.scoring import SCORER_VERSION

MATCHES = "/api/matches"


def make_profile(user_id: int, filename: str = "r.pdf") -> int:
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


def make_posting(slug: str, title: str = "Backend Engineer") -> int:
    """Create a posting with a unique canonical key.

    Randomised for the same reason conftest randomises emails, and it bites
    harder here: `job_postings` is deliberately decoupled from any user
    (spec section 5.6), so deleting the fixture user cascades to profiles and
    jobs but leaves postings behind. A fixed key therefore survives the test
    that made it and collides with the next run's -- which surfaces as a
    UniqueViolation in an unrelated test.
    """
    unique = f"{slug}-{uuid4().hex[:12]}"
    db = SessionLocal()
    try:
        posting = JobPosting(
            canonical_key=f"https://boards.example.com/{unique}",
            url=f"https://boards.example.com/{unique}",
            content_hash=f"hash-{unique}",
            raw_text="posting text",
            title=title,
            company="Example Co",
        )
        db.add(posting)
        db.commit()
        db.refresh(posting)
        return posting.id
    finally:
        db.close()


def make_match(profile_id: int, posting_id: int, final: float = 0.5, **over) -> int:
    breakdown = {
        "semantic_score": over.get("semantic", 0.5),
        "skill_score": over.get("skill", 0.5),
        "final_score": final,
        "skills": over.get(
            "skills",
            [
                {
                    "name": "Python",
                    "necessity": "required",
                    "bucket": "matched",
                    "required_years": 3.0,
                    "candidate_years": 4.0,
                    "evidence": "3+ years of Python",
                },
                {
                    "name": "Kubernetes",
                    "necessity": "required",
                    "bucket": "missing",
                    "required_years": None,
                    "candidate_years": None,
                    "evidence": "experience with Kubernetes",
                },
            ],
        ),
        "counts": over.get(
            "counts",
            {"matched": 1, "partial": 0, "missing": 1, "missing_required": 1},
        ),
        "weights": {"semantic": 0.4, "skill": 0.6},
        "extraction_failed": over.get("extraction_failed", False),
    }
    db = SessionLocal()
    try:
        match = Match(
            profile_id=profile_id,
            job_posting_id=posting_id,
            semantic_score=over.get("semantic", 0.5),
            skill_score=over.get("skill", 0.5),
            final_score=final,
            breakdown=breakdown,
            origin=over.get("origin", "user_submission"),
            scorer_version=over.get("scorer_version", SCORER_VERSION),
        )
        db.add(match)
        db.commit()
        db.refresh(match)
        return match.id
    finally:
        db.close()


@pytest.fixture
def user(make_user, as_user) -> User:
    return as_user(make_user())


@pytest.fixture
def client(user) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def profile_id(user) -> int:
    return make_profile(user.id)


class TestAccess:
    def test_anonymous_is_rejected(self, make_user) -> None:
        """Deliberately does not use the `user` / `client` fixtures.

        `as_user` overrides `current_user` in `app.dependency_overrides`,
        which is process-wide -- so a freshly constructed TestClient is still
        authenticated, and this test passed a 200 as a 401 until it was
        written to avoid pulling that fixture in transitively. Building the
        profile from `make_user` alone leaves no override installed, so the
        real dependency runs.
        """
        profile_id = make_profile(make_user(email="anon-check@fitcheck.dev").id)
        anon = TestClient(app, raise_server_exceptions=False)

        assert anon.get(f"{MATCHES}?profile_id={profile_id}").status_code == 401
        assert anon.get(f"{MATCHES}/1").status_code == 401

    def test_another_users_profile_is_404_not_empty(
        self, client, make_user, profile_id
    ) -> None:
        """404, not an empty list.

        Filtering the matches instead of checking the profile would return
        `[]`, which says "this profile has no matches" where the truth is
        "this profile is not yours" -- and the client cannot tell those
        apart. 404 rather than 403 so the response does not confirm which
        profile ids exist.
        """
        stranger_profile = make_profile(make_user(email="other@fitcheck.dev").id)

        response = client.get(f"{MATCHES}?profile_id={stranger_profile}")

        assert response.status_code == 404

    def test_another_users_match_is_404(self, client, make_user) -> None:
        """The row never leaves the database -- ownership is a join
        predicate, not a check applied after loading it."""
        stranger_profile = make_profile(make_user(email="other2@fitcheck.dev").id)
        posting = make_posting("stranger")
        match_id = make_match(stranger_profile, posting)

        assert client.get(f"{MATCHES}/{match_id}").status_code == 404

    def test_unknown_match_is_404(self, client) -> None:
        assert client.get(f"{MATCHES}/99999999").status_code == 404


class TestListing:
    def test_returns_best_first(self, client, profile_id) -> None:
        """The ordering `matches_feed_idx` exists to serve. A feed that is
        not sorted by score is not a feed."""
        make_match(profile_id, make_posting("low"), final=0.20)
        make_match(profile_id, make_posting("high"), final=0.90)
        make_match(profile_id, make_posting("mid"), final=0.55)

        scores = [m["final_score"] for m in client.get(f"{MATCHES}?profile_id={profile_id}").json()]

        assert scores == sorted(scores, reverse=True)

    def test_excludes_other_profiles(self, client, user, profile_id) -> None:
        """Two resumes belonging to the same person are still separate
        feeds -- a match is scored against one resume and means nothing
        against another."""
        other = make_profile(user.id, "second.pdf")
        make_match(profile_id, make_posting("mine"), final=0.5)
        make_match(other, make_posting("theirs"), final=0.9)

        body = client.get(f"{MATCHES}?profile_id={profile_id}").json()

        assert {m["profile_id"] for m in body} == {profile_id}

    def test_limit_is_capped(self, client, profile_id) -> None:
        """An unbounded limit is a way to ask for every match ever scored."""
        assert client.get(f"{MATCHES}?profile_id={profile_id}&limit=100000").status_code == 422

    def test_empty_feed_is_an_empty_list(self, client, profile_id) -> None:
        assert client.get(f"{MATCHES}?profile_id={profile_id}").json() == []

    def test_forbids_caching(self, client, profile_id) -> None:
        """Scores change as postings are re-fetched and re-scored, so a
        cached feed is a stale ranking presented as a current one."""
        response = client.get(f"{MATCHES}?profile_id={profile_id}")

        assert response.headers["cache-control"] == "no-store"


class TestDetail:
    def test_surfaces_both_sub_scores(self, client, profile_id) -> None:
        """Spec section 8.4: the blend is a judgement call, so the number is
        only defensible if what produced it travels with it."""
        match_id = make_match(profile_id, make_posting("d1"), final=0.62, semantic=0.8, skill=0.5)

        body = client.get(f"{MATCHES}/{match_id}").json()

        assert body["semantic_score"] == pytest.approx(0.8, abs=1e-5)
        assert body["skill_score"] == pytest.approx(0.5, abs=1e-5)
        assert body["final_score"] == pytest.approx(0.62, abs=1e-5)
        assert body["weights"] == {"semantic": 0.4, "skill": 0.6}

    def test_carries_the_skill_breakdown(self, client, profile_id) -> None:
        match_id = make_match(profile_id, make_posting("d2"))

        body = client.get(f"{MATCHES}/{match_id}").json()

        buckets = {s["name"]: s["bucket"] for s in body["skills"]}
        assert buckets == {"Python": "matched", "Kubernetes": "missing"}

    def test_carries_the_posting_evidence(self, client, profile_id) -> None:
        """The quote is what makes a verdict checkable rather than asserted."""
        match_id = make_match(profile_id, make_posting("d3"))

        body = client.get(f"{MATCHES}/{match_id}").json()

        assert body["skills"][0]["evidence"] == "3+ years of Python"

    def test_separates_missing_required(self, client, profile_id) -> None:
        match_id = make_match(profile_id, make_posting("d4"))

        assert client.get(f"{MATCHES}/{match_id}").json()["counts"]["missing_required"] == 1

    def test_denormalizes_the_posting_for_display(self, client, profile_id) -> None:
        """A 50-row feed would otherwise cost 50 extra requests, or a join
        the client has to know to ask for."""
        match_id = make_match(profile_id, make_posting("d5", title="Staff Engineer"))

        body = client.get(f"{MATCHES}/{match_id}").json()

        assert body["posting_title"] == "Staff Engineer"
        assert body["posting_company"] == "Example Co"
        assert "boards.example.com" in body["posting_url"]

    def test_exposes_the_scorer_version(self, client, profile_id) -> None:
        """A client comparing two matches has no other way to know they were
        produced by different rules and are not comparable."""
        match_id = make_match(profile_id, make_posting("d6"))

        assert client.get(f"{MATCHES}/{match_id}").json()["scorer_version"] == SCORER_VERSION

    def test_flags_a_semantic_only_score(self, client, profile_id) -> None:
        """Without this the UI renders a confident number with no skills
        listed, which is indistinguishable from a genuine total mismatch."""
        match_id = make_match(
            profile_id, make_posting("d7"), extraction_failed=True, skills=[],
            counts={"matched": 0, "partial": 0, "missing": 0, "missing_required": 0},
        )

        assert client.get(f"{MATCHES}/{match_id}").json()["extraction_failed"] is True


class TestOlderGenerations:
    def test_a_match_from_an_older_scorer_still_reads(self, client, profile_id) -> None:
        """A row written before a key existed must not 500.

        Returning an error for a two-generation-old match is a worse answer
        than an incomplete one -- the feed would break for everyone who had
        not been re-scored yet.
        """
        posting = make_posting("legacy")
        db = SessionLocal()
        try:
            match = Match(
                profile_id=profile_id,
                job_posting_id=posting,
                semantic_score=0.4,
                skill_score=0.4,
                final_score=0.4,
                breakdown={},  # no counts, no skills, no weights
                origin="user_submission",
                scorer_version=SCORER_VERSION - 1,
            )
            db.add(match)
            db.commit()
            db.refresh(match)
            match_id = match.id
        finally:
            db.close()

        body = client.get(f"{MATCHES}/{match_id}").json()

        assert body["skills"] == []
        assert body["counts"] == {
            "matched": 0, "partial": 0, "missing": 0, "missing_required": 0
        }
