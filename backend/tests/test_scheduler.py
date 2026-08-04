"""Which sources a tick considers due.

The scheduler's whole job is this predicate. Getting it wrong is quiet in
both directions: too eager and the crawler hammers a board that asked for
daily, too shy and the catalog silently stops updating while every dashboard
says the source is enabled.
"""

from datetime import UTC, datetime, timedelta

import pytest

from app.db import SessionLocal
from app.models import MAX_CONSECUTIVE_FAILURES, Source
from app.workers import tasks


class RecordingQueue:
    def __init__(self):
        self.calls: list[tuple] = []

    def enqueue(self, func_name, *args, **kwargs):
        self.calls.append((func_name, args))
        return None


@pytest.fixture
def queue(monkeypatch) -> RecordingQueue:
    recorder = RecordingQueue()
    monkeypatch.setattr(tasks, "get_queue", lambda _name=None: recorder)
    return recorder


@pytest.fixture
def source():
    """Sources for one test, removed afterwards.

    The dev database has real sources in it, so every assertion below is
    scoped to ids this fixture created rather than to the queue's total
    length.
    """
    created: list[int] = []

    def _make(**overrides):
        db = SessionLocal()
        try:
            row = Source(
                kind="lever",
                board_token=f"tok-{datetime.now(UTC).timestamp()}-{len(created)}",
                display_name="Test Board",
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

    yield _make

    db = SessionLocal()
    try:
        db.query(Source).filter(Source.id.in_(created)).delete(
            synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


def _queued_ids(queue: RecordingQueue) -> set[int]:
    return {args[0] for _func, args in queue.calls}


class TestDueSelection:
    def test_a_never_crawled_source_is_due(self, queue, source):
        sid = source()
        tasks.tick_sources()
        assert sid in _queued_ids(queue)

    def test_a_source_crawled_within_its_interval_is_not_due(self, queue, source):
        sid = source(
            crawl_interval_seconds=3600,
            last_crawled_at=datetime.now(UTC) - timedelta(minutes=10),
        )
        tasks.tick_sources()
        assert sid not in _queued_ids(queue)

    def test_a_source_past_its_interval_is_due(self, queue, source):
        sid = source(
            crawl_interval_seconds=3600,
            last_crawled_at=datetime.now(UTC) - timedelta(hours=2),
        )
        tasks.tick_sources()
        assert sid in _queued_ids(queue)

    def test_each_source_honours_its_own_interval(self, queue, source):
        """The interval is per source, not a global setting."""
        an_hour_ago = datetime.now(UTC) - timedelta(hours=1, minutes=1)
        frequent = source(crawl_interval_seconds=3600, last_crawled_at=an_hour_ago)
        daily = source(crawl_interval_seconds=86400, last_crawled_at=an_hour_ago)

        tasks.tick_sources()
        queued = _queued_ids(queue)

        assert frequent in queued
        assert daily not in queued

    def test_a_disabled_source_is_never_due(self, queue, source):
        sid = source(enabled=False)
        tasks.tick_sources()
        assert sid not in _queued_ids(queue)

    def test_an_open_circuit_is_not_due(self, queue, source):
        """A board that failed five times running is not fixed by asking again.

        `discover_source` checks this too and is the authority. Checking here
        as well keeps the queue free of jobs whose only outcome is to return
        "circuit_open".
        """
        sid = source(consecutive_failures=MAX_CONSECUTIVE_FAILURES)
        tasks.tick_sources()
        assert sid not in _queued_ids(queue)

    def test_a_source_below_the_threshold_still_crawls(self, queue, source):
        """Failing is not the same as broken."""
        sid = source(consecutive_failures=MAX_CONSECUTIVE_FAILURES - 1)
        tasks.tick_sources()
        assert sid in _queued_ids(queue)


class TestTickBehaviour:
    def test_a_tick_in_progress_does_not_queue_a_second_crawl(self, queue, source):
        """`last_crawled_at` is stamped before enumerating, for this reason.

        Ticking faster than a crawl completes must not stack crawls of the
        same board on somebody else's server.
        """
        sid = source(crawl_interval_seconds=3600)

        tasks.tick_sources()
        assert sid in _queued_ids(queue)

        # Simulate discover_source stamping the attempt as it starts.
        db = SessionLocal()
        try:
            db.get(Source, sid).last_crawled_at = datetime.now(UTC)
            db.commit()
        finally:
            db.close()

        queue.calls.clear()
        tasks.tick_sources()
        assert sid not in _queued_ids(queue)

    def test_one_bad_enqueue_does_not_cost_the_others_their_tick(
        self, monkeypatch, source
    ):
        source()
        source()

        calls: list = []

        class HalfBrokenQueue:
            def enqueue(self, func_name, *args, **kwargs):
                calls.append(args[0])
                if len(calls) == 1:
                    raise RuntimeError("redis hiccup")

        monkeypatch.setattr(tasks, "get_queue", lambda _name=None: HalfBrokenQueue())

        # Must not raise, and must keep going past the failure.
        tasks.tick_sources()
        assert len(calls) >= 2

    def test_reports_when_nothing_is_due(self, queue, source):
        source(last_crawled_at=datetime.now(UTC), crawl_interval_seconds=86400)
        # Other sources in the dev database may legitimately be due, so this
        # asserts on the shape of the answer rather than on "none".
        result = tasks.tick_sources()
        assert result == "none_due" or result.startswith("queued ")
