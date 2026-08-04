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


class RecordingQueue:
    """Captures enqueues instead of reaching Redis."""

    def __init__(self):
        self.calls: list[tuple] = []

    def enqueue(self, func_name, *args, **kwargs):
        self.calls.append((func_name, args))
        return None


@pytest.fixture
def queue(monkeypatch) -> RecordingQueue:
    recorder = RecordingQueue()
    monkeypatch.setattr("app.routers.matches.get_queue", lambda _name=None: recorder)
    return recorder


class TestBuildRecommendations:
    def test_queues_a_build_for_a_profile_with_no_feed(self, feed, queue):
        db = SessionLocal()
        try:
            db.get(Profile, feed["profile_id"]).embedding = [0.0] * 384
            db.commit()
        finally:
            db.close()

        # The fixture's matches are all scorer_version SCORER_VERSION but the
        # recommendation-origin ones exist, so clear them to reach the
        # never-built state this endpoint is for.
        db = SessionLocal()
        try:
            db.query(Match).filter(
                Match.profile_id == feed["profile_id"],
                Match.origin == "recommendation",
            ).delete(synchronize_session=False)
            db.commit()
        finally:
            db.close()

        response = client.post(
            "/api/matches/recommendations", params={"profile_id": feed["profile_id"]}
        )
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert queue.calls and queue.calls[0][0] == "app.workers.tasks.score_profile"

    def test_does_not_rebuild_a_current_feed(self, feed, queue):
        """A client polling an empty-looking feed must not stampede the queue."""
        response = client.post(
            "/api/matches/recommendations", params={"profile_id": feed["profile_id"]}
        )
        assert response.status_code == 202
        assert response.json()["status"] == "already_current"
        assert queue.calls == [], "a current feed must not be rebuilt"

    def test_reports_a_profile_that_cannot_be_recalled_against(self, feed, queue):
        """No embedding means wait for extraction, not for scoring.

        Collapsing this into "nothing happened" would leave the UI telling the
        user to wait for a job that is never going to run.
        """
        db = SessionLocal()
        try:
            db.query(Match).filter(
                Match.profile_id == feed["profile_id"],
                Match.origin == "recommendation",
            ).delete(synchronize_session=False)
            db.get(Profile, feed["profile_id"]).embedding = None
            db.commit()
        finally:
            db.close()

        response = client.post(
            "/api/matches/recommendations", params={"profile_id": feed["profile_id"]}
        )
        assert response.json()["status"] == "profile_not_ready"
        assert queue.calls == []

    def test_another_users_profile_is_404(self, feed, queue, make_user, as_user):
        as_user(make_user())
        response = client.post(
            "/api/matches/recommendations", params={"profile_id": feed["profile_id"]}
        )
        assert response.status_code == 404
        assert queue.calls == []

    def test_a_broker_failure_is_reported_rather_than_promised(
        self, feed, monkeypatch
    ):
        """There is no durable row here for a sweep to pick up later.

        Returning "queued" when the enqueue failed would promise a feed that
        nothing is building.
        """
        db = SessionLocal()
        try:
            db.query(Match).filter(
                Match.profile_id == feed["profile_id"],
                Match.origin == "recommendation",
            ).delete(synchronize_session=False)
            db.get(Profile, feed["profile_id"]).embedding = [0.0] * 384
            db.commit()
        finally:
            db.close()

        def boom(_name=None):
            raise RuntimeError("redis is down")

        monkeypatch.setattr("app.routers.matches.get_queue", boom)

        response = client.post(
            "/api/matches/recommendations", params={"profile_id": feed["profile_id"]}
        )
        assert response.status_code == 503


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


class TestSavedMatches:
    def test_returns_only_matches_the_caller_reacted_to(self, feed):
        client.post(
            f"/api/matches/{feed['matches']['rec']}/feedback",
            json={"verdict": "interested"},
        )

        rows = client.get("/api/matches/saved").json()
        ids = {r["match_id"] for r in rows}

        assert feed["matches"]["rec"] in ids
        assert feed["matches"]["sub"] not in ids, "no verdict means not saved"

    def test_shows_the_latest_verdict_not_every_one(self, feed):
        """A tracker showing the same job twice at two stages is a worse tracker.

        The history stays intact underneath -- see test_is_append_only -- this
        is only about what the current-state view shows.
        """
        match_id = feed["matches"]["rec"]
        client.post(f"/api/matches/{match_id}/feedback", json={"verdict": "interested"})
        client.post(f"/api/matches/{match_id}/feedback", json={"verdict": "applied"})

        rows = [r for r in client.get("/api/matches/saved").json()
                if r["match_id"] == match_id]

        assert len(rows) == 1
        assert rows[0]["verdict"] == "applied"

    def test_filters_by_verdict(self, feed):
        client.post(
            f"/api/matches/{feed['matches']['rec']}/feedback",
            json={"verdict": "applied"},
        )
        client.post(
            f"/api/matches/{feed['matches']['sub']}/feedback",
            json={"verdict": "not_interested"},
        )

        applied = client.get("/api/matches/saved", params={"verdict": "applied"}).json()
        ids = {r["match_id"] for r in applied}

        assert feed["matches"]["rec"] in ids
        assert feed["matches"]["sub"] not in ids

    def test_unknown_verdict_is_rejected(self, feed):
        response = client.get("/api/matches/saved", params={"verdict": "nonsense"})
        assert response.status_code == 422

    def test_a_closed_posting_is_flagged_rather_than_hidden(self, feed):
        """You applied to it; it is still part of your history.

        Hiding it would make an application vanish from the tracker, which is
        worse than showing it marked as filled.
        """
        match_id = feed["matches"]["closed"]
        client.post(f"/api/matches/{match_id}/feedback", json={"verdict": "applied"})

        rows = [r for r in client.get("/api/matches/saved").json()
                if r["match_id"] == match_id]

        assert rows and rows[0]["posting_closed"] is True

    def test_another_users_reactions_are_not_visible(self, feed, make_user, as_user):
        client.post(
            f"/api/matches/{feed['matches']['rec']}/feedback",
            json={"verdict": "interested"},
        )

        as_user(make_user())
        rows = client.get("/api/matches/saved").json()

        assert all(r["match_id"] != feed["matches"]["rec"] for r in rows)


class TestSkillGaps:
    def _with_breakdown(self, feed, key, skills):
        db = SessionLocal()
        try:
            match = db.get(Match, feed["matches"][key])
            match.breakdown = {"skills": skills, "counts": {}, "weights": {}}
            db.commit()
        finally:
            db.close()

    def test_ranks_blocking_gaps_first(self, feed):
        """A missing *required* skill is what actually disqualifies.

        Ranking by raw missing count would let a widely-listed nice-to-have
        outrank the requirement that is costing the candidate the job.
        """
        common_optional = {
            "name": "Kubernetes",
            "bucket": "missing",
            "necessity": "preferred",
        }
        blocker = {"name": "Rust", "bucket": "missing", "necessity": "required"}

        self._with_breakdown(feed, "rec", [common_optional, blocker])
        self._with_breakdown(feed, "sub", [common_optional])
        self._with_breakdown(feed, "unstated", [common_optional])

        report = client.get(
            "/api/insights/skill-gaps",
            params={"profile_id": feed["profile_id"]},
        ).json()

        names = [g["name"] for g in report["gaps"]]
        assert names[0] == "Rust", "the blocking requirement must rank first"
        assert "Kubernetes" in names

    def test_counts_every_bucket(self, feed):
        self._with_breakdown(feed, "rec", [
            {"name": "Go", "bucket": "missing", "necessity": "required"},
        ])
        self._with_breakdown(feed, "sub", [
            {"name": "Go", "bucket": "partial", "necessity": "required"},
        ])
        self._with_breakdown(feed, "unstated", [
            {"name": "Go", "bucket": "matched", "necessity": "required"},
        ])

        report = client.get(
            "/api/insights/skill-gaps",
            params={"profile_id": feed["profile_id"]},
        ).json()
        go = [g for g in report["gaps"] if g["name"] == "Go"][0]

        assert (go["missing"], go["partial"], go["matched"]) == (1, 1, 1)

    def test_a_fully_satisfied_skill_is_not_a_gap(self, feed):
        self._with_breakdown(feed, "rec", [
            {"name": "Python", "bucket": "matched", "necessity": "required"},
        ])
        self._with_breakdown(feed, "sub", [])
        self._with_breakdown(feed, "unstated", [])

        report = client.get(
            "/api/insights/skill-gaps",
            params={"profile_id": feed["profile_id"]},
        ).json()

        assert all(g["name"] != "Python" for g in report["gaps"])

    def test_reports_the_denominator(self, feed):
        """"Missing in 9" is unreadable without knowing 9 out of what."""
        report = client.get(
            "/api/insights/skill-gaps",
            params={"profile_id": feed["profile_id"]},
        ).json()
        assert report["matches_analyzed"] == 4

    def test_another_users_profile_is_404(self, feed, make_user, as_user):
        as_user(make_user())
        response = client.get(
            "/api/insights/skill-gaps", params={"profile_id": feed["profile_id"]}
        )
        assert response.status_code == 404
