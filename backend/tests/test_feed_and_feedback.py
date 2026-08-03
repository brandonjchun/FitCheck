"""The feed's read-time filters, and the feedback loop behind them.

These filters are not the recall filters. Recall decides what gets *scored*
and runs on a queue; these decide what gets *shown* and run on the request
path. The duplication is deliberate -- see `list_matches` -- and the tests are
separate for the same reason: a bug in one is invisible from the other.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import SessionLocal
from app.extraction import POSTING_EXTRACTION_VERSION
from app.main import app
from app.models import JobPosting, Match, MatchFeedback, Profile
from app.scoring import SCORER_VERSION

client = TestClient(app)


@pytest.fixture
def feed(make_user, as_user):
    """A profile with matches over postings of assorted shapes."""
    user = make_user()
    as_user(user)

    posting_ids: list[int] = []
    match_ids: dict[str, int] = {}

    db = SessionLocal()
    try:
        profile = Profile(
            user_id=user.id, original_filename="t.pdf", raw_text="Python."
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        profile_id = profile.id

        def _posting(key, **kw):
            posting = JobPosting(
                canonical_key=f"feed:{key}:{profile_id}",
                url=f"https://example.com/{key}",
                content_hash=f"h-{key}-{profile_id}",
                raw_text="body",
                extracted={"skills": []},
                extraction_version=POSTING_EXTRACTION_VERSION,
                **kw,
            )
            db.add(posting)
            db.commit()
            db.refresh(posting)
            posting_ids.append(posting.id)
            return posting.id

        def _match(key, posting_id, score, origin):
            match = Match(
                profile_id=profile_id,
                job_posting_id=posting_id,
                semantic_score=score,
                skill_score=score,
                final_score=score,
                breakdown={"counts": {}, "skills": [], "weights": {}},
                origin=origin,
                scorer_version=SCORER_VERSION,
            )
            db.add(match)
            db.commit()
            db.refresh(match)
            match_ids[key] = match.id

        _match("rec", _posting("rec", remote_type="remote", seniority="senior",
                               min_years=2), 0.9, "recommendation")
        _match("sub", _posting("sub", remote_type="onsite", seniority="junior",
                               min_years=8), 0.8, "user_submission")
        _match("unstated", _posting("unstated"), 0.7, "recommendation")

        from sqlalchemy import text as sql_text

        closed_id = _posting("closed", remote_type="remote")
        db.execute(
            sql_text("UPDATE job_postings SET closed_at = now() WHERE id = :i"),
            {"i": closed_id},
        )
        db.commit()
        _match("closed", closed_id, 0.95, "recommendation")
    finally:
        db.close()

    yield {"profile_id": profile_id, "matches": match_ids, "user": user}

    db = SessionLocal()
    try:
        db.query(Match).filter(Match.profile_id == profile_id).delete(
            synchronize_session=False
        )
        db.query(JobPosting).filter(JobPosting.id.in_(posting_ids)).delete(
            synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


def _keys(feed, body) -> set[str]:
    by_id = {v: k for k, v in feed["matches"].items()}
    return {by_id[row["id"]] for row in body if row["id"] in by_id}


def _get(feed, **params) -> list[dict]:
    response = client.get(
        "/api/matches", params={"profile_id": feed["profile_id"], **params}
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestFeedFilters:
    def test_closed_postings_are_hidden_by_default(self, feed):
        """A filled role is a dead link presented as an opportunity.

        Note the closed posting has the *highest* score here, so a regression
        would put it first rather than bury it.
        """
        assert "closed" not in _keys(feed, _get(feed))

    def test_closed_postings_remain_reachable_on_request(self, feed):
        assert "closed" in _keys(feed, _get(feed, include_closed=True))

    def test_origin_filter(self, feed):
        assert _keys(feed, _get(feed, origin="user_submission")) == {"sub"}
        assert _keys(feed, _get(feed, origin="recommendation")) == {"rec", "unstated"}

    def test_unknown_origin_is_rejected_by_name(self, feed):
        response = client.get(
            "/api/matches",
            params={"profile_id": feed["profile_id"], "origin": "nonsense"},
        )
        assert response.status_code == 422
        # The message should name the alternatives rather than just refusing.
        assert "recommendation" in response.text

    def test_remote_only(self, feed):
        assert _keys(feed, _get(feed, remote_only=True)) == {"rec"}

    def test_seniority(self, feed):
        assert _keys(feed, _get(feed, seniority=["senior"])) == {"rec"}

    def test_unstated_years_survives_a_years_filter(self, feed):
        """NULL min_years means "never said", not "demands zero"."""
        assert _keys(feed, _get(feed, max_min_years=3)) == {"rec", "unstated"}

    def test_ordering_is_by_score(self, feed):
        scores = [row["final_score"] for row in _get(feed)]
        assert scores == sorted(scores, reverse=True)

    def test_another_users_profile_is_404_not_empty(self, feed, make_user, as_user):
        """An empty list would say "no matches" where the truth is "not yours"."""
        as_user(make_user())
        response = client.get(
            "/api/matches", params={"profile_id": feed["profile_id"]}
        )
        assert response.status_code == 404


class TestFeedback:
    def test_records_a_verdict(self, feed):
        match_id = feed["matches"]["rec"]
        response = client.post(
            f"/api/matches/{match_id}/feedback", json={"verdict": "interested"}
        )
        assert response.status_code == 201
        body = response.json()
        assert body["match_id"] == match_id
        assert body["verdict"] == "interested"
        assert body["created_at"]

    def test_is_append_only(self, feed):
        """The sequence interested -> applied is the funnel, not a correction."""
        match_id = feed["matches"]["rec"]
        client.post(f"/api/matches/{match_id}/feedback", json={"verdict": "interested"})
        client.post(f"/api/matches/{match_id}/feedback", json={"verdict": "applied"})

        db = SessionLocal()
        try:
            rows = db.execute(
                select(MatchFeedback)
                .where(MatchFeedback.match_id == match_id)
                .order_by(MatchFeedback.created_at)
            ).scalars().all()
            verdicts = [r.verdict for r in rows]
        finally:
            db.close()

        assert verdicts == ["interested", "applied"]

    def test_unknown_verdict_is_rejected(self, feed):
        response = client.post(
            f"/api/matches/{feed['matches']['rec']}/feedback",
            json={"verdict": "maybe_later"},
        )
        assert response.status_code == 422

    def test_another_users_match_is_404(self, feed, make_user, as_user):
        match_id = feed["matches"]["rec"]
        as_user(make_user())
        response = client.post(
            f"/api/matches/{match_id}/feedback", json={"verdict": "interested"}
        )
        assert response.status_code == 404

        db = SessionLocal()
        try:
            leaked = db.execute(
                select(MatchFeedback).where(MatchFeedback.match_id == match_id)
            ).scalars().all()
        finally:
            db.close()
        assert leaked == [], "a rejected request must not write a label"

    def test_deleting_a_match_takes_its_labels(self, feed):
        """An orphaned verdict is not partial data, it is uninterpretable.

        The features a label was a reaction to live on the match, so a label
        that outlives it can never become a training row.
        """
        match_id = feed["matches"]["rec"]
        client.post(f"/api/matches/{match_id}/feedback", json={"verdict": "applied"})

        db = SessionLocal()
        try:
            db.delete(db.get(Match, match_id))
            db.commit()
            remaining = db.execute(
                select(MatchFeedback).where(MatchFeedback.match_id == match_id)
            ).scalars().all()
        finally:
            db.close()

        assert remaining == []
