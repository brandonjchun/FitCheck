"""The fetch path of a crawl: change detection, and fanning out to `ingest`.

`test_discover.py` covers the inline-content path, where Lever and Ashby hand
back the full description and a crawl is one HTTP request. This covers the
other half -- Greenhouse, which publishes `updated_at` but no descriptions,
so each posting has to be fetched and the only question worth asking is
*which ones*.

That question is the difference between a daily crawl of ~400 requests and
one of a handful. The content hash cannot answer it, because computing a hash
requires already having fetched the content; `updated_at` is the only signal
available before spending the request.

**The subtle failure this file exists for** is in
`test_a_skipped_posting_is_still_marked_present`. Skipping a fetch must still
bump `last_seen_at`, because closure detection tombstones anything the crawl
did not see -- so the optimisation that avoids re-fetching an unchanged
posting would, without that heartbeat, close every unchanged posting on the
board. The crawl would report success while quietly emptying the catalog.
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.boards import DiscoveredPosting
from app.db import SessionLocal
from app.models import IngestJob, JobPosting, Source, canonical_key_for_url
from app.workers import tasks

BOARD = "https://job-boards.greenhouse.io/acme/jobs"


def _posting(external_id: str, updated_at: datetime | None = None) -> DiscoveredPosting:
    """A Greenhouse-shaped posting: dated, with no inline content."""
    return DiscoveredPosting(
        external_id=external_id,
        url=f"{BOARD}/{external_id}",
        title=f"Role {external_id}",
        updated_at=updated_at,
        content=None,
    )


@pytest.fixture
def source_id():
    db = SessionLocal()
    try:
        source = Source(
            kind="greenhouse",
            board_token=f"acme-{datetime.now(UTC).timestamp()}",
            display_name="Acme",
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
        db.query(IngestJob).filter_by(source_id=sid).delete()
        db.query(JobPosting).filter_by(source_id=sid).delete()
        db.query(Source).filter_by(id=sid).delete()
        db.commit()
    finally:
        db.close()


@pytest.fixture
def crawl(monkeypatch):
    """Run a crawl returning the given postings, with the queue stubbed."""
    enqueued: list = []

    def fake_queue(name="interactive"):
        class Q:
            def enqueue(self, func, *args, **kwargs):
                enqueued.append((name, func, args))
                return type("J", (), {"id": "rq-fake"})()

        return Q()

    # Patched in both places on purpose. `tasks.py` binds `get_queue` at
    # module import for the fan-out path, while the scoring handoff imports
    # it inside the function -- so the two resolve against different
    # namespaces and patching only one leaves half the enqueues real.
    monkeypatch.setattr(tasks, "get_queue", fake_queue)
    monkeypatch.setattr("app.queues.get_queue", fake_queue)

    def run(postings, source_id):
        monkeypatch.setattr(tasks, "enumerate_source", lambda kind, token: postings)
        return tasks.discover_source(source_id)

    run.enqueued = enqueued
    return run


def _stored(source_id: int) -> dict[str, JobPosting]:
    db = SessionLocal()
    try:
        rows = db.execute(
            select(JobPosting).where(JobPosting.source_id == source_id)
        ).scalars().all()
        return {row.canonical_key: row for row in rows}
    finally:
        db.close()


class TestFanOut:
    def test_a_new_posting_becomes_an_ingest_job(self, crawl, source_id) -> None:
        crawl([_posting("1")], source_id)

        db = SessionLocal()
        try:
            jobs = db.execute(
                select(IngestJob).where(IngestJob.source_id == source_id)
            ).scalars().all()
        finally:
            db.close()

        assert len(jobs) == 1
        assert jobs[0].kind == "ingest_posting"
        assert jobs[0].url == f"{BOARD}/1"

    def test_crawler_jobs_belong_to_no_profile(self, crawl, source_id) -> None:
        """The column M8 forced open. A discovered posting enters the shared
        catalog and is scored against profiles later, by a different job."""
        crawl([_posting("1")], source_id)

        db = SessionLocal()
        try:
            job = db.execute(
                select(IngestJob).where(IngestJob.source_id == source_id)
            ).scalar_one()
            assert job.profile_id is None
        finally:
            db.close()

    def test_work_lands_on_the_ingest_queue(self, crawl, source_id) -> None:
        """Not `interactive`. A crawl tick produces hundreds of fetches, and
        putting them in the small queue is the head-of-line blocking the
        four-queue split exists to prevent -- one crawl would then delay
        every user submission behind ~400 bulk fetches."""
        crawl([_posting("1"), _posting("2")], source_id)

        assert {name for name, _, _ in crawl.enqueued} == {"ingest"}

    def test_the_rq_job_id_is_recorded(self, crawl, source_id) -> None:
        """Correlates the row to the queued job, and every other producer
        here does it.

        Without it the row reads `queued` with a null rq_job_id, which is
        precisely the signature a requeue sweep uses to find work whose
        enqueue failed. A healthy fan-out of 399 jobs would look like 399
        orphans and get re-enqueued wholesale.
        """
        crawl([_posting("1")], source_id)

        db = SessionLocal()
        try:
            job = db.execute(
                select(IngestJob).where(IngestJob.source_id == source_id)
            ).scalar_one()
            assert job.rq_job_id == "rq-fake"
        finally:
            db.close()

    def test_two_overlapping_ticks_enqueue_one_job(self, crawl, source_id) -> None:
        """The partial unique index doing its job. A daily schedule firing
        while yesterday's crawl still drains would otherwise double the
        fan-out at somebody else's server."""
        crawl([_posting("1")], source_id)
        crawl([_posting("1")], source_id)

        db = SessionLocal()
        try:
            jobs = db.execute(
                select(IngestJob).where(IngestJob.source_id == source_id)
            ).scalars().all()
        finally:
            db.close()

        assert len(jobs) == 1


class TestSourceAttribution:
    """A fetched posting has to remember which board it came from.

    Found by running the crawl rather than by reading the code: 399 Greenhouse
    postings fanned out, the workers fetched them, and every resulting row
    landed with `source_id = NULL` -- because `process_job_url` knew the URL
    and not the source.

    The consequence is silent and total. `_close_missing_postings` scans
    `WHERE source_id = :source_id`, so a NULL row can never be tombstoned:
    the board stops listing the job, the crawl completes successfully, and
    the dead posting stays in every feed indefinitely. Closure detection
    appeared to work because the inline-content boards set it correctly.
    """

    def test_a_fetched_posting_keeps_its_source(self, monkeypatch, source_id) -> None:
        monkeypatch.setattr(tasks, "fetch_posting_text", lambda url: "Backend role. " * 30)

        posting_id = tasks._upsert_posting(
            f"{BOARD}/1", "Backend role. " * 30, source_id=source_id
        )

        db = SessionLocal()
        try:
            assert db.get(JobPosting, posting_id).source_id == source_id
        finally:
            db.close()

    def test_a_user_submission_does_not_erase_an_existing_source(
        self, source_id
    ) -> None:
        """Somebody pasting a URL the crawler already owns must not orphan it.

        A user-submitted fetch carries no source, and a plain assignment on
        conflict would write that NULL over the board attribution -- quietly
        removing the posting from closure detection's reach.
        """
        url = f"{BOARD}/shared"
        tasks._upsert_posting(url, "Original text. " * 30, source_id=source_id)

        tasks._upsert_posting(url, "Updated text. " * 30, source_id=None)

        db = SessionLocal()
        try:
            row = db.execute(
                select(JobPosting).where(
                    JobPosting.canonical_key == canonical_key_for_url(url)
                )
            ).scalar_one()
            assert row.source_id == source_id
        finally:
            db.close()


class TestChangeDetection:
    def test_an_unchanged_posting_is_not_refetched(self, crawl, source_id) -> None:
        """The optimisation itself: 400 requests become a handful."""
        stamp = datetime.now(UTC) - timedelta(days=3)
        crawl([_posting("1", stamp)], source_id)

        # Pretend the first crawl's fetch completed and recorded the stamp.
        key = canonical_key_for_url(f"{BOARD}/1")
        db = SessionLocal()
        try:
            db.query(JobPosting).filter_by(canonical_key=key).delete()
            db.add(
                JobPosting(
                    canonical_key=key,
                    url=f"{BOARD}/1",
                    source_id=source_id,
                    content_hash="h",
                    raw_text="the posting text",
                    source_updated_at=stamp,
                )
            )
            db.commit()
        finally:
            db.close()

        before = len(crawl.enqueued)
        crawl([_posting("1", stamp)], source_id)

        assert len(crawl.enqueued) == before

    def test_a_changed_posting_is_refetched(self, crawl, source_id) -> None:
        old = datetime.now(UTC) - timedelta(days=3)
        key = canonical_key_for_url(f"{BOARD}/1")
        db = SessionLocal()
        try:
            db.add(
                JobPosting(
                    canonical_key=key,
                    url=f"{BOARD}/1",
                    source_id=source_id,
                    content_hash="h",
                    raw_text="the posting text",
                    source_updated_at=old,
                )
            )
            db.commit()
        finally:
            db.close()

        before = len(crawl.enqueued)
        crawl([_posting("1", old + timedelta(days=1))], source_id)

        assert len(crawl.enqueued) == before + 1

    def test_a_posting_with_no_change_signal_is_always_refetched(
        self, crawl, source_id
    ) -> None:
        """Erring toward the extra request is the cheap mistake. The other
        direction leaves a posting whose requirements changed being scored
        against the old ones indefinitely, with nothing to reveal it."""
        key = canonical_key_for_url(f"{BOARD}/1")
        db = SessionLocal()
        try:
            db.add(
                JobPosting(
                    canonical_key=key,
                    url=f"{BOARD}/1",
                    source_id=source_id,
                    content_hash="h",
                    raw_text="the posting text",
                    source_updated_at=None,
                )
            )
            db.commit()
        finally:
            db.close()

        before = len(crawl.enqueued)
        crawl([_posting("1", None)], source_id)

        assert len(crawl.enqueued) == before + 1

    def test_a_seen_posting_with_no_text_is_refetched(self, crawl, source_id) -> None:
        """Seen, but the fetch never produced anything usable. Treating that
        as "unchanged" would strand it empty forever."""
        stamp = datetime.now(UTC)
        key = canonical_key_for_url(f"{BOARD}/1")
        db = SessionLocal()
        try:
            db.add(
                JobPosting(
                    canonical_key=key,
                    url=f"{BOARD}/1",
                    source_id=source_id,
                    content_hash="h",
                    raw_text="",
                    source_updated_at=stamp,
                )
            )
            db.commit()
        finally:
            db.close()

        before = len(crawl.enqueued)
        crawl([_posting("1", stamp)], source_id)

        assert len(crawl.enqueued) == before + 1


class TestSkippingDoesNotCloseThings:
    def test_a_skipped_posting_is_still_marked_present(self, crawl, source_id) -> None:
        """The failure this whole file was written for.

        Closure detection tombstones anything whose `last_seen_at` predates
        the crawl. A posting skipped for being unchanged was still *seen* on
        the board, so skipping the fetch must still bump the heartbeat --
        otherwise the optimisation that avoids re-fetching unchanged postings
        closes every unchanged posting, and the crawl reports success while
        emptying the catalog.
        """
        stamp = datetime.now(UTC) - timedelta(days=3)
        key = canonical_key_for_url(f"{BOARD}/1")
        db = SessionLocal()
        try:
            db.add(
                JobPosting(
                    canonical_key=key,
                    url=f"{BOARD}/1",
                    source_id=source_id,
                    content_hash="h",
                    raw_text="the posting text",
                    source_updated_at=stamp,
                    last_seen_at=datetime.now(UTC) - timedelta(days=3),
                )
            )
            db.commit()
        finally:
            db.close()

        crawl([_posting("1", stamp)], source_id)

        row = _stored(source_id)[key]
        assert row.closed_at is None, "an unchanged posting was tombstoned"

    def test_a_skipped_posting_is_reopened_if_it_was_closed(
        self, crawl, source_id
    ) -> None:
        """A posting back on the board is live again, even if we skip
        re-fetching it. Leaving the tombstone keeps it out of every feed."""
        stamp = datetime.now(UTC) - timedelta(days=3)
        key = canonical_key_for_url(f"{BOARD}/1")
        db = SessionLocal()
        try:
            db.add(
                JobPosting(
                    canonical_key=key,
                    url=f"{BOARD}/1",
                    source_id=source_id,
                    content_hash="h",
                    raw_text="the posting text",
                    source_updated_at=stamp,
                    closed_at=datetime.now(UTC) - timedelta(days=1),
                )
            )
            db.commit()
        finally:
            db.close()

        crawl([_posting("1", stamp)], source_id)

        assert _stored(source_id)[key].closed_at is None


class TestPromotedMetadata:
    """The board's structured fields must survive extraction.

    A board API states the title as data; the LLM infers it from prose and
    returns null whenever the description has no header. Assigning the
    extraction's value unconditionally destroys the reliable one -- measured
    in production as 0 of 98 postings keeping a title after their first
    extraction pass.
    """

    def test_a_null_extracted_title_does_not_erase_the_board_title(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        from app.db import SessionLocal
        from app.models import JobPosting
        from app.workers import tasks

        db = SessionLocal()
        try:
            posting = JobPosting(
                canonical_key=f"promote:{datetime.now(UTC).timestamp()}",
                url="https://example.com/p",
                content_hash="promote-hash",
                raw_text="A description with no header line. " * 12,
                title="Staff Backend Engineer",
                company="Acme",
            )
            db.add(posting)
            db.commit()
            db.refresh(posting)
            pid = posting.id
        finally:
            db.close()

        try:
            # An extraction that found no title, which is the common case for
            # a description that is pure prose.
            monkeypatch.setattr(
                tasks,
                "extract_posting",
                lambda _text: SimpleNamespace(
                    title=None,
                    company=None,
                    location=None,
                    remote_type=None,
                    seniority="senior",
                    min_years_experience=None,
                    model_dump=lambda mode=None: {"skills": []},
                ),
            )
            monkeypatch.setattr(tasks, "embed_text", lambda _t: [0.0] * 384)

            tasks._prepare_posting(pid)

            db = SessionLocal()
            try:
                row = db.get(JobPosting, pid)
                assert row.title == "Staff Backend Engineer"
                assert row.company == "Acme"
                # What the extraction *did* find still lands.
                assert row.seniority == "senior"
            finally:
                db.close()
        finally:
            db = SessionLocal()
            try:
                db.query(JobPosting).filter(JobPosting.id == pid).delete()
                db.commit()
            finally:
                db.close()
