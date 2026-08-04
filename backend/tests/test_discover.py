"""Crawl tick behaviour, especially what a crawl is allowed to close.

`enumerate_source` is patched throughout rather than hitting a real board.
The adapters' own parsing is a separate concern; what these cover is the
decision logic wrapped around them, and that logic's worst failure is silent.
A crawl that tombstones a board it never finished enumerating removes several
hundred postings from every user's feed and logs it as a success, so most of
this file is about proving closure only happens after a complete enumeration.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.boards import DiscoveredPosting
from app.db import SessionLocal
from app.extraction import POSTING_EXTRACTION_VERSION
from app.models import MAX_CONSECUTIVE_FAILURES, JobPosting, Source
from app.workers import tasks
from app.workers.fetch import TransientFetchError

# Long enough to clear MIN_POSTING_CHARS, so these postings take the inline
# path and never need a fetch queue to exist.
BODY = "We are hiring a backend engineer. " * 10


def _posting(external_id: str, *, body: str = BODY, **overrides) -> DiscoveredPosting:
    defaults = {
        "external_id": external_id,
        "url": f"https://jobs.lever.co/acme/{external_id}",
        "title": f"Role {external_id}",
        "content": body,
    }
    defaults.update(overrides)
    return DiscoveredPosting(**defaults)


@pytest.fixture
def source_id():
    """One enabled Lever source, torn down with everything it created."""
    db = SessionLocal()
    try:
        source = Source(
            kind="lever",
            board_token=f"acme-{datetime.now(UTC).timestamp()}",
            display_name="Acme",
            crawl_interval_seconds=86400,
        )
        db.add(source)
        db.commit()
        db.refresh(source)
        sid = source.id
    finally:
        db.close()

    yield sid

    db = SessionLocal()
    try:
        db.query(JobPosting).filter(JobPosting.source_id == sid).delete()
        db.query(Source).filter(Source.id == sid).delete()
        db.commit()
    finally:
        db.close()


def _enumerate_returns(monkeypatch, postings):
    monkeypatch.setattr(tasks, "enumerate_source", lambda kind, token: list(postings))


def _enumerate_raises(monkeypatch, exc):
    def boom(kind, token):
        raise exc

    monkeypatch.setattr(tasks, "enumerate_source", boom)


def _mark_extracted(source_id: int) -> None:
    """Stand in for a completed extraction on every posting of this source.

    The gate exists to skip extraction and embedding, so it only fires for a
    posting that already has a current one -- see the test below.
    """
    db = SessionLocal()
    try:
        for row in db.execute(
            select(JobPosting).where(JobPosting.source_id == source_id)
        ).scalars():
            row.extracted = {"skills": []}
            row.extraction_version = POSTING_EXTRACTION_VERSION
        db.commit()
    finally:
        db.close()


def _postings_for(source_id: int) -> dict[str, JobPosting]:
    db = SessionLocal()
    try:
        rows = db.execute(
            select(JobPosting).where(JobPosting.source_id == source_id)
        ).scalars().all()
        return {row.canonical_key: row for row in rows}
    finally:
        db.close()


class TestClosureDetection:
    def test_a_posting_the_board_stopped_listing_is_closed(
        self, monkeypatch, source_id
    ):
        _enumerate_returns(monkeypatch, [_posting("a"), _posting("b")])
        assert tasks.discover_source(source_id) == "discovered"
        assert len(_postings_for(source_id)) == 2

        # Second crawl: the board no longer lists "b".
        _enumerate_returns(monkeypatch, [_posting("a")])
        assert tasks.discover_source(source_id) == "discovered"

        rows = _postings_for(source_id)
        # endswith, not `in`. The board token is "acme", so `"/a" in url`
        # matches ".../acme/b" as well -- both rows landed in `still_listed`
        # and the assertion then turned on whichever one Postgres happened to
        # return first. It passed for months and failed the moment unrelated
        # rows changed the row order in the shared dev database.
        still_listed = [r for r in rows.values() if r.url.endswith("/a")]
        dropped = [r for r in rows.values() if r.url.endswith("/b")]

        assert len(still_listed) == 1 and len(dropped) == 1

        assert still_listed[0].closed_at is None
        assert dropped[0].closed_at is not None, "a delisted posting must be tombstoned"

    def test_closure_is_a_tombstone_not_a_delete(self, monkeypatch, source_id):
        """`matches` references postings, so the row has to survive closure."""
        _enumerate_returns(monkeypatch, [_posting("a")])
        tasks.discover_source(source_id)

        _enumerate_returns(monkeypatch, [])
        tasks.discover_source(source_id)

        rows = _postings_for(source_id)
        assert len(rows) == 1, "closure must not remove the row"
        assert next(iter(rows.values())).is_open is False

    def test_failed_enumeration_closes_nothing(self, monkeypatch, source_id):
        """The guard this whole module exists for.

        A board that times out mid-enumeration leaves us holding a partial
        list. Treating absence from a partial list as "closed" would tombstone
        most of the board while reporting nothing wrong.
        """
        _enumerate_returns(monkeypatch, [_posting("a"), _posting("b")])
        tasks.discover_source(source_id)

        _enumerate_raises(monkeypatch, TransientFetchError("board timed out"))
        assert tasks.discover_source(source_id) == "enumeration_failed"

        rows = _postings_for(source_id)
        assert len(rows) == 2
        assert all(r.closed_at is None for r in rows.values()), (
            "a failed crawl must never tombstone anything"
        )

    def test_closure_uses_the_database_clock_not_the_workers(
        self, monkeypatch, source_id
    ):
        """Both sides of the closure comparison must come from one clock.

        Postings are stamped `last_seen_at = func.now()` in Postgres. If the
        crawl boundary came from the worker's own clock instead, any skew
        would make freshly-touched postings look older than the crawl that
        touched them -- and the whole live board gets tombstoned, silently.

        This drives the worker clock an hour into the future, which is what
        skew looks like at its worst. Nothing may close: every posting here
        was seen by this very crawl.
        """

        class FutureDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(tz) + timedelta(hours=1)

        monkeypatch.setattr(tasks, "datetime", FutureDatetime)

        _enumerate_returns(monkeypatch, [_posting("a"), _posting("b")])
        tasks.discover_source(source_id)
        tasks.discover_source(source_id)

        rows = _postings_for(source_id)
        assert len(rows) == 2
        assert all(r.closed_at is None for r in rows.values()), (
            "a worker clock ahead of the database must not close live postings"
        )

    def test_reappearing_posting_is_reopened(self, monkeypatch, source_id):
        """A board that drops a role and re-lists it should not stay closed."""
        _enumerate_returns(monkeypatch, [_posting("a")])
        tasks.discover_source(source_id)

        _enumerate_returns(monkeypatch, [])
        tasks.discover_source(source_id)
        assert all(r.closed_at is not None for r in _postings_for(source_id).values())

        _enumerate_returns(monkeypatch, [_posting("a")])
        tasks.discover_source(source_id)
        assert all(r.closed_at is None for r in _postings_for(source_id).values())


class TestCrawlGuards:
    def test_disabled_source_is_skipped(self, monkeypatch, source_id):
        db = SessionLocal()
        try:
            db.get(Source, source_id).enabled = False
            db.commit()
        finally:
            db.close()

        _enumerate_raises(monkeypatch, AssertionError("must not enumerate"))
        assert tasks.discover_source(source_id) == "disabled"

    def test_open_circuit_stops_the_crawl(self, monkeypatch, source_id):
        db = SessionLocal()
        try:
            db.get(Source, source_id).consecutive_failures = MAX_CONSECUTIVE_FAILURES
            db.commit()
        finally:
            db.close()

        _enumerate_raises(monkeypatch, AssertionError("must not enumerate"))
        assert tasks.discover_source(source_id) == "circuit_open"

    def test_missing_source_is_not_an_error(self, monkeypatch):
        _enumerate_raises(monkeypatch, AssertionError("must not enumerate"))
        assert tasks.discover_source(2**40) == "source_missing"

    def test_failure_increments_the_breaker_and_success_resets_it(
        self, monkeypatch, source_id
    ):
        _enumerate_raises(monkeypatch, TransientFetchError("nope"))
        tasks.discover_source(source_id)
        tasks.discover_source(source_id)

        db = SessionLocal()
        try:
            assert db.get(Source, source_id).consecutive_failures == 2
        finally:
            db.close()

        _enumerate_returns(monkeypatch, [_posting("a")])
        tasks.discover_source(source_id)

        db = SessionLocal()
        try:
            source = db.get(Source, source_id)
            # Reset, not decremented: the column counts *consecutive* failures.
            assert source.consecutive_failures == 0
            assert source.last_success_at is not None
        finally:
            db.close()

    def test_attempt_is_recorded_even_when_enumeration_fails(
        self, monkeypatch, source_id
    ):
        """`last_crawled_at` drives scheduling, so it must survive a crash.

        Left unset, a reliably-failing source looks permanently due and gets
        retried in a tight loop.
        """
        _enumerate_raises(monkeypatch, TransientFetchError("nope"))
        tasks.discover_source(source_id)

        db = SessionLocal()
        try:
            source = db.get(Source, source_id)
            assert source.last_crawled_at is not None
            assert source.last_success_at is None
        finally:
            db.close()

    def test_short_content_is_skipped_rather_than_stored(self, monkeypatch, source_id):
        _enumerate_returns(monkeypatch, [_posting("a", body="Too short.")])
        tasks.discover_source(source_id)
        assert _postings_for(source_id) == {}


class FakeRedis:
    """Just enough of the counter interface for the gate."""

    def __init__(self):
        self.store: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    def get(self, key: str):
        return self.store.get(key)


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(tasks, "get_redis", lambda: redis)
    return redis


class TestContentHashGate:
    def test_unchanged_content_records_a_hit(
        self, monkeypatch, source_id, fake_redis
    ):
        _enumerate_returns(monkeypatch, [_posting("a")])
        tasks.discover_source(source_id)
        assert tasks.gate_stats()["misses"] == 1, "first sight is always a miss"

        _mark_extracted(source_id)

        # Identical content on the next crawl: the gate should fire.
        tasks.discover_source(source_id)
        stats = tasks.gate_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_unextracted_posting_does_not_hit_the_gate(
        self, monkeypatch, source_id, fake_redis
    ):
        """The gate skips extraction, so it cannot fire before one exists.

        Content being unchanged is not sufficient on its own: a posting stored
        but never extracted still owes an LLM call, and treating it as a hit
        would strand it as text with no structured fields, invisible to every
        query that filters on them.
        """
        _enumerate_returns(monkeypatch, [_posting("a")])
        tasks.discover_source(source_id)
        tasks.discover_source(source_id)

        assert tasks.gate_stats()["hits"] == 0
        assert tasks.gate_stats()["misses"] == 2

    def test_changed_content_misses_and_invalidates_extraction(
        self, monkeypatch, source_id, fake_redis
    ):
        _enumerate_returns(monkeypatch, [_posting("a")])
        tasks.discover_source(source_id)

        _mark_extracted(source_id)
        original_hash = next(iter(_postings_for(source_id).values())).content_hash

        _enumerate_returns(monkeypatch, [_posting("a", body=BODY + " Now with Rust.")])
        tasks.discover_source(source_id)

        row = next(iter(_postings_for(source_id).values()))
        assert row.content_hash != original_hash
        assert row.extraction_version is None, (
            "new text must invalidate the old extraction rather than keep stale skills"
        )
        assert tasks.gate_stats()["misses"] == 2

    def test_hit_rate_is_reported(self, fake_redis):
        for _ in range(3):
            tasks._record_gate_hit("k", hit=True)
        tasks._record_gate_hit("k", hit=False)

        stats = tasks.gate_stats()
        assert stats == {"hits": 3, "misses": 1, "total": 4, "hit_rate": 0.75}

    def test_hit_rate_with_no_data_does_not_divide_by_zero(self, fake_redis):
        assert tasks.gate_stats() == {
            "hits": 0,
            "misses": 0,
            "total": 0,
            "hit_rate": 0.0,
        }

    def test_instrumentation_failure_never_breaks_ingestion(
        self, monkeypatch, source_id
    ):
        class BrokenRedis:
            def incr(self, key):
                raise RuntimeError("redis is down")

            def get(self, key):
                raise RuntimeError("redis is down")

        monkeypatch.setattr(tasks, "get_redis", lambda: BrokenRedis())
        _enumerate_returns(monkeypatch, [_posting("a")])

        assert tasks.discover_source(source_id) == "discovered"
        assert len(_postings_for(source_id)) == 1
        assert tasks.gate_stats()["total"] == 0
