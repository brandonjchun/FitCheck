"""Per-source crawl freshness on the ops dashboard.

The distinction under test throughout: *attempted* is not *succeeded*. A
crawler that runs on schedule and fails every time keeps `last_crawled_at`
current while the catalog ages, so a dashboard reading that column alone
reports a healthy board whose data is a week stale. Freshness here is defined
on `last_success_at`, and these tests exist to keep it that way.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import MAX_CONSECUTIVE_FAILURES, JobPosting, Source, User


@pytest.fixture
def promote():
    def _promote(user):
        db = SessionLocal()
        try:
            db.get(User, user.id).is_admin = True
            db.commit()
        finally:
            db.close()
        user.is_admin = True
        return user

    return _promote


@pytest.fixture
def client(make_user, as_user, promote) -> TestClient:
    as_user(promote(make_user()))
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def source():
    """One source, with knobs for the freshness columns."""
    created: list[int] = []
    postings: list[int] = []

    def _make(**overrides):
        db = SessionLocal()
        try:
            row = Source(
                kind="lever",
                board_token=f"tok-{datetime.now(UTC).timestamp()}",
                display_name=overrides.pop("display_name", "Test Board"),
                crawl_interval_seconds=overrides.pop("crawl_interval_seconds", 3600),
                **overrides,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            created.append(row.id)
            return row.id
        finally:
            db.close()

    def _add_posting(source_id: int, *, closed: bool = False):
        db = SessionLocal()
        try:
            row = JobPosting(
                canonical_key=f"fresh:{source_id}:{len(postings)}",
                url="https://example.com/p",
                content_hash=f"h{source_id}-{len(postings)}",
                raw_text="body",
                source_id=source_id,
                closed_at=datetime.now(UTC) if closed else None,
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            postings.append(row.id)
        finally:
            db.close()

    _make.add_posting = _add_posting
    yield _make

    db = SessionLocal()
    try:
        db.query(JobPosting).filter(JobPosting.id.in_(postings)).delete(
            synchronize_session=False
        )
        db.query(Source).filter(Source.id.in_(created)).delete(
            synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


def _row_for(client: TestClient, source_id: int) -> dict:
    response = client.get("/api/ops/overview")
    assert response.status_code == 200, response.text
    body = response.json()
    match = [s for s in body["sources"] if s["id"] == source_id]
    assert match, f"source {source_id} missing from overview"
    return match[0]


class TestSourceFreshness:
    def test_a_source_that_never_succeeded_is_stale(self, client, source):
        """Unknown age must not read as fresh.

        Otherwise the board that has never once worked is the one the
        dashboard is quietest about.
        """
        row = _row_for(client, source())
        assert row["last_success_at"] is None
        assert row["seconds_since_success"] is None
        assert row["is_stale"] is True

    def test_a_recent_success_is_fresh(self, client, source):
        sid = source(last_success_at=datetime.now(UTC))
        row = _row_for(client, sid)
        assert row["is_stale"] is False
        assert row["seconds_since_success"] < 60

    def test_a_success_older_than_the_interval_is_stale(self, client, source):
        sid = source(
            crawl_interval_seconds=3600,
            last_success_at=datetime.now(UTC) - timedelta(hours=3),
        )
        assert _row_for(client, sid)["is_stale"] is True

    def test_attempts_without_success_do_not_count_as_fresh(self, client, source):
        """The failure this whole panel exists to make visible."""
        sid = source(
            crawl_interval_seconds=3600,
            last_crawled_at=datetime.now(UTC),
            last_success_at=datetime.now(UTC) - timedelta(days=7),
        )
        row = _row_for(client, sid)
        assert row["last_crawled_at"] is not None
        assert row["is_stale"] is True, (
            "freshness must be defined on success, not on attempts"
        )

    def test_circuit_state_is_surfaced(self, client, source):
        sid = source(consecutive_failures=MAX_CONSECUTIVE_FAILURES)
        row = _row_for(client, sid)
        assert row["consecutive_failures"] == MAX_CONSECUTIVE_FAILURES
        assert row["circuit_open"] is True

    def test_open_posting_count_excludes_tombstones(self, client, source):
        sid = source()
        source.add_posting(sid)
        source.add_posting(sid)
        source.add_posting(sid, closed=True)

        assert _row_for(client, sid)["open_postings"] == 2


class TestGateStats:
    def test_hit_rate_is_reported(self, client):
        """Closes the M8 gap: the counters existed but nothing read them."""
        body = client.get("/api/ops/overview").json()
        gate = body["gate"]
        assert set(gate) == {"hits", "misses", "total", "hit_rate"}
        assert gate["total"] == gate["hits"] + gate["misses"]
        assert 0.0 <= gate["hit_rate"] <= 1.0
