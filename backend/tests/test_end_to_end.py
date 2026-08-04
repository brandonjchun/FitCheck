"""Path A, from an empty account to a rendered skill gap.

Every other test in this suite verifies one seam. This one verifies that the
seams line up, by driving the whole of Path A through the real HTTP API and a
real RQ worker:

    register -> log in -> upload a resume -> extract it -> embed it
             -> submit a posting URL -> fetch it -> extract it -> embed it
             -> score the pair -> read the match -> read the skill gaps

Nothing between those steps is called directly. Each stage is reached because
the previous one enqueued it, which is the property a chain of unit tests
cannot establish: `test_worker_tasks` proves `process_job_url` hands off to the
scoring queue, and `test_matches_endpoint` proves the feed reads `matches`, but
neither notices if the two are wired to different queues. That failure mode is
not hypothetical -- M6 shipped a requeue onto the wrong queue and everything
still "worked", just in the wrong lane.

**Three boundaries are stubbed, and only three.** The network, the LLM, and the
embedding model -- the things that cost money, need a GPU, or reach somebody
else's server. Everything else is the real code path: real Postgres, real
Redis, real sessions, real RQ, real SQL. A test that also stubbed the queue or
the database would be asserting on its own mocks.

**Isolation is not optional here.** `docker compose up` leaves four workers
draining `interactive`, `ingest`, `scoring`, and `discovery` against this same
Redis. A test that enqueues onto those names is racing them for its own jobs --
observed directly: a compose worker picked up a job a test had enqueued and
logged `process_job_url: job 11782 is gone` after the test's teardown deleted
the row. So every queue here gets a name nothing else drains, and this test
runs its own worker over them.
"""

import uuid
from hashlib import blake2b

import pytest
from fastapi.testclient import TestClient
from redis import Redis
from rq import Queue, SimpleWorker

from app.config import settings
from app.db import SessionLocal
from app.embeddings import EMBEDDING_DIM
from app.extraction import ExtractedPosting, ExtractedProfile
from app.models import IngestJob, JobPosting, Match, Profile, User
from tests.conftest import make_pdf

WORKER_LOG_LEVEL = "CRITICAL"


# --- the three stubbed boundaries -------------------------------------------


def fake_embedding(text: str) -> list[float]:
    """A deterministic stand-in for MiniLM that still means something.

    Each token is hashed into one of `EMBEDDING_DIM` buckets, so two documents
    sharing vocabulary land near each other and two that share none do not.
    That is enough for cosine similarity to behave directionally, which is all
    this test asks of it -- the real model is exercised in test_embeddings.py.

    Deterministic on purpose: a random vector would make the semantic half of
    every score, and therefore the feed's ordering, differ between runs.
    """
    vector = [0.0] * EMBEDDING_DIM
    for token in text.lower().split():
        digest = blake2b(token.encode(), digest_size=4).digest()
        vector[int.from_bytes(digest, "big") % EMBEDDING_DIM] += 1.0

    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude == 0:
        # pgvector rejects an all-zero vector for cosine distance, and a
        # document with no tokens is not what this test is about.
        vector[0] = 1.0
        return vector
    return [value / magnitude for value in vector]


RESUME_TEXT = (
    "Brandon Carter. Senior Backend Engineer. "
    "Built GraphQL services in Python. Ran Redis caches and Postgres."
)

POSTING_TEXT = (
    "Senior Platform Engineer at Acme. We need graphql api experience, "
    "Python, redis, and Kubernetes. Remote."
)


def fake_profile_extraction(raw_text: str) -> ExtractedProfile:
    """What the LLM would return for RESUME_TEXT.

    Note the spellings: the resume says "GraphQL" and "Redis" while the
    posting below says "graphql api" and "redis". Getting those to count as
    the same requirement is the whole point of `skills.canonical_key`, and
    this is where that is verified against the real scorer rather than in
    isolation.
    """
    return ExtractedProfile(
        skills=[
            {"name": "GraphQL", "years": 4.0, "evidence": "Built GraphQL services", "source": "experience"},
            {"name": "Python", "years": 5.0, "evidence": "in Python", "source": "experience"},
            {"name": "Redis", "years": 3.0, "evidence": "Ran Redis caches", "source": "experience"},
        ],
        total_years_experience=5.0,
        seniority="senior",
        education=[],
    )


def fake_posting_extraction(raw_text: str, title: str | None = None) -> ExtractedPosting:
    return ExtractedPosting(
        title=title or "Senior Platform Engineer",
        company="Acme",
        location="Remote",
        remote_type="remote",
        seniority="senior",
        min_years_experience=5.0,
        skills=[
            {"name": "graphql api", "necessity": "required", "min_years": None, "evidence": "graphql api experience"},
            {"name": "Python", "necessity": "required", "min_years": None, "evidence": "Python"},
            {"name": "redis", "necessity": "preferred", "min_years": None, "evidence": "redis"},
            {"name": "Kubernetes", "necessity": "required", "min_years": None, "evidence": "Kubernetes"},
        ],
    )


# --- isolation ---------------------------------------------------------------


class QueueSet:
    """Isolated queues, plus a worker that drains them to completion.

    `get_queue` is patched in five places rather than one because the routers
    bind it at import (`from app.queues import get_queue`), so patching the
    source module alone leaves their references pointing at the original.
    `app.workers.tasks` binds it both ways -- at module level and again inside
    two functions -- so both are covered.
    """

    def __init__(self, redis: Redis) -> None:
        self.redis = redis
        self.prefix = f"e2e-{uuid.uuid4().hex[:10]}"
        self._queues: dict[str, Queue] = {}

    def get(self, name: str = "interactive") -> Queue:
        if name not in self._queues:
            self._queues[name] = Queue(f"{self.prefix}-{name}", connection=self.redis)
        return self._queues[name]

    def drain(self, max_rounds: int = 8) -> int:
        """Run every queue until nothing new is produced.

        Looped rather than run once, because the pipeline enqueues as it goes:
        a fetch produces a scoring job, and draining `interactive` once would
        leave that scoring job sitting there. The loop is what makes this test
        follow the chain instead of asserting on its first link.
        """
        performed = 0
        for _ in range(max_rounds):
            before = performed
            for queue in list(self._queues.values()):
                worker = SimpleWorker([queue], connection=self.redis)
                worker.work(burst=True, logging_level=WORKER_LOG_LEVEL)
                worker.register_death()
                performed += 1 if queue.count == 0 else 0
            if not any(q.count for q in self._queues.values()) and performed > before:
                break
            if not any(q.count for q in self._queues.values()):
                break
        return performed

    def cleanup(self) -> None:
        from rq.registry import FailedJobRegistry, ScheduledJobRegistry

        for queue in self._queues.values():
            for registry in (
                ScheduledJobRegistry(queue.name, connection=self.redis),
                FailedJobRegistry(queue.name, connection=self.redis),
            ):
                self.redis.delete(registry.key)
            queue.empty()
            self.redis.delete(queue.key)
            self.redis.srem("rq:queues", queue.key)


@pytest.fixture
def pipeline(monkeypatch):
    """The whole rig: isolated queues, stubbed boundaries, a client."""
    from app.main import app
    from app.workers import tasks

    redis = Redis.from_url(settings.redis_url)
    queues = QueueSet(redis)

    for target in (
        "app.queues.get_queue",
        "app.workers.tasks.get_queue",
        "app.routers.jobs.get_queue",
        "app.routers.profiles.get_queue",
        "app.routers.matches.get_queue",
    ):
        monkeypatch.setattr(target, queues.get)

    # The network. fetch_posting_text has its own tests against
    # httpx.MockTransport; what matters here is that the worker reaches it.
    monkeypatch.setattr(tasks, "fetch_posting_text", lambda url: POSTING_TEXT)
    monkeypatch.setattr(tasks, "extract_profile", fake_profile_extraction)
    monkeypatch.setattr(tasks, "extract_posting", fake_posting_extraction)
    monkeypatch.setattr(tasks, "embed_text", fake_embedding)

    with TestClient(app) as client:
        yield client, queues

    queues.cleanup()
    redis.close()


@pytest.fixture
def account(pipeline):
    """A real registered, logged-in account. Cleaned up by cascade."""
    client, _ = pipeline
    email = f"e2e-{uuid.uuid4().hex[:10]}@fitcheck.dev"

    registered = client.post(
        "/api/auth/register", json={"email": email, "password": "e2epassword123"}
    )
    assert registered.status_code in (200, 201), registered.text

    logged_in = client.post(
        "/api/auth/login", json={"email": email, "password": "e2epassword123"}
    )
    assert logged_in.status_code == 200, logged_in.text

    yield email

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).one_or_none()
        if user is not None:
            db.delete(user)
            db.commit()
    finally:
        db.close()


class TestPathAEndToEnd:
    def test_a_resume_and_a_url_become_an_explained_match(self, pipeline, account):
        client, queues = pipeline

        # 1. Upload a resume. The API does local text extraction only and
        #    returns immediately -- the LLM call is queued.
        upload = client.post(
            "/api/profiles",
            files={"file": ("resume.pdf", make_pdf(RESUME_TEXT), "application/pdf")},
        )
        assert upload.status_code == 201, upload.text
        profile_id = upload.json()["id"]
        assert upload.json()["extraction_ok"] is False, (
            "upload must not block on the LLM; that is the whole reason it is queued"
        )

        # 2. Run the queued extraction.
        queues.drain()

        profile = client.get(f"/api/profiles/{profile_id}").json()
        assert profile["extraction_ok"] is True
        assert {s["name"] for s in profile["skills"]} == {"GraphQL", "Python", "Redis"}

        # 3. Submit a posting URL. 202, because nothing is fetched inline.
        submit = client.post(
            "/api/jobs",
            json={"profile_id": profile_id, "url": "https://boards.example.com/e2e/1"},
        )
        assert submit.status_code == 202, submit.text
        job_id = submit.json()["id"]
        assert submit.json()["status"] == "queued"

        # 4. Drain: fetch -> upsert posting -> enqueue scoring -> score.
        queues.drain()

        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "succeeded", job
        assert job["is_terminal"] is True

        # 5. The feed now has the match, with its reasoning attached.
        matches = client.get("/api/matches", params={"profile_id": profile_id}).json()
        assert len(matches) == 1, matches
        match = matches[0]

        assert match["posting_title"] == "Senior Platform Engineer"
        assert match["posting_company"] == "Acme"
        assert 0.0 <= match["final_score"] <= 1.0
        assert match["origin"] == "user_submission"

        by_name = {s["name"]: s for s in match["skills"]}

        # The normalization, proven through the whole stack rather than in a
        # unit test. Two things happened between the LLM and this response.
        #
        # The posting's raw requirements were "graphql api" and "redis"; they
        # reach the client as "GraphQL" and "Redis", because JobPosting.skills
        # canonicalizes on read. And they are *matched* against a resume that
        # said "GraphQL" and "Redis" -- which before `canonical_key` would have
        # been two missing required skills decided by capitalisation.
        assert by_name["GraphQL"]["bucket"] == "matched"
        assert by_name["Redis"]["bucket"] == "matched"
        assert by_name["Python"]["bucket"] == "matched"
        # And a requirement genuinely absent is still missing, so the merge
        # has not turned into a scorer that matches everything.
        assert by_name["Kubernetes"]["bucket"] == "missing"
        assert match["counts"]["missing_required"] == 1

        # 1.0 (GraphQL) + 1.0 (Python) + 1/3 (Redis, preferred) covered, out of
        # those plus 1.0 for the missing Kubernetes. Asserted as an ordering
        # claim rather than a constant: what matters is that a preferred skill
        # counts for less than a required one and that the missing required
        # skill costs real weight.
        assert 0.6 < match["skill_score"] < 0.8

        # 6. Insights aggregates the same breakdown into advice.
        gaps = client.get("/api/insights/skill-gaps").json()
        assert gaps["matches_analyzed"] == 1
        gap_names = [g["name"] for g in gaps["gaps"]]
        assert "Kubernetes" in gap_names
        assert "GraphQL" not in gap_names, "a satisfied requirement is not a gap"

    def test_the_posting_enters_the_shared_catalog(self, pipeline, account):
        """Path A writes to the same `job_postings` the crawler owns.

        If it did not, Path A and Path B scores would be computed over
        different catalogs and would stop being comparable -- the single
        constraint the whole design is built around.
        """
        client, queues = pipeline

        upload = client.post(
            "/api/profiles",
            files={"file": ("resume.pdf", make_pdf(RESUME_TEXT), "application/pdf")},
        )
        profile_id = upload.json()["id"]
        queues.drain()

        client.post(
            "/api/jobs",
            json={"profile_id": profile_id, "url": "https://boards.example.com/e2e/2"},
        )
        queues.drain()

        db = SessionLocal()
        try:
            job = db.query(IngestJob).filter(IngestJob.profile_id == profile_id).one()
            posting = db.get(JobPosting, job.job_posting_id)

            assert posting is not None
            assert posting.canonical_key.startswith("url:")
            assert posting.embedding is not None, "unembedded postings are invisible to the feed"
            assert posting.extraction_is_current
            # No source: a user-submitted URL has no board behind it, and
            # inventing one would let closure detection tombstone it.
            assert posting.source_id is None
        finally:
            db.close()

    def test_resubmitting_the_same_url_does_not_refetch_or_duplicate(
        self, pipeline, account
    ):
        """At-least-once delivery makes this the steady state, not an edge case.

        A second submission must not produce a second posting row, and must not
        cost a second request to somebody else's server.
        """
        client, queues = pipeline
        fetches: list[str] = []

        from app.workers import tasks

        def counting_fetch(url: str) -> str:
            fetches.append(url)
            return POSTING_TEXT

        tasks.fetch_posting_text = counting_fetch

        upload = client.post(
            "/api/profiles",
            files={"file": ("resume.pdf", make_pdf(RESUME_TEXT), "application/pdf")},
        )
        profile_id = upload.json()["id"]
        queues.drain()

        url = "https://boards.example.com/e2e/3"
        first = client.post("/api/jobs", json={"profile_id": profile_id, "url": url})
        queues.drain()
        second = client.post("/api/jobs", json={"profile_id": profile_id, "url": url})
        queues.drain()

        # The dedupe is on the *work*: the same profile submitting the same URL
        # gets its existing job back rather than a new one.
        assert first.json()["id"] == second.json()["id"]
        assert len(fetches) == 1, "the second submission re-fetched a page we already had"

        db = SessionLocal()
        try:
            postings = (
                db.query(JobPosting)
                .filter(JobPosting.url == url)
                .count()
            )
            assert postings == 1

            matches = db.query(Match).filter(Match.profile_id == profile_id).count()
            assert matches == 1
        finally:
            db.close()

    def test_a_dead_fetch_leaves_no_match_and_says_so(self, pipeline, account):
        """The failure path end to end.

        A 404 must reach the user as a dead job rather than as a silent absence
        -- and must not leave a half-built match behind.
        """
        client, queues = pipeline

        from app.workers import tasks
        from app.workers.fetch import PermanentFetchError

        def gone(url: str) -> str:
            raise PermanentFetchError("HTTP 404 fetching ...")

        tasks.fetch_posting_text = gone

        upload = client.post(
            "/api/profiles",
            files={"file": ("resume.pdf", make_pdf(RESUME_TEXT), "application/pdf")},
        )
        profile_id = upload.json()["id"]
        queues.drain()

        submit = client.post(
            "/api/jobs",
            json={"profile_id": profile_id, "url": "https://boards.example.com/e2e/404"},
        )
        queues.drain()

        job = client.get(f"/api/jobs/{submit.json()['id']}").json()
        assert job["status"] == "dead"
        assert job["is_terminal"] is True
        assert "404" in job["last_error"]
        # Dead on the first attempt: a 404 is as final then as on attempt
        # three, so the retry budget is never spent.
        assert job["attempts"] == 1

        assert client.get("/api/matches", params={"profile_id": profile_id}).json() == []

    def test_another_user_cannot_see_the_match(self, pipeline, account):
        """Ownership holds at the end of the pipeline, not just at its start.

        Everything above ran as one account. The feed is per-profile, and the
        predicate that scopes it is the same `owned_profile` dependency the
        upload used -- this checks it did not get lost somewhere in between.
        """
        client, queues = pipeline

        upload = client.post(
            "/api/profiles",
            files={"file": ("resume.pdf", make_pdf(RESUME_TEXT), "application/pdf")},
        )
        profile_id = upload.json()["id"]
        queues.drain()
        client.post(
            "/api/jobs",
            json={"profile_id": profile_id, "url": "https://boards.example.com/e2e/5"},
        )
        queues.drain()

        intruder = f"e2e-x-{uuid.uuid4().hex[:8]}@fitcheck.dev"
        client.post("/api/auth/register", json={"email": intruder, "password": "e2epassword123"})
        client.post("/api/auth/login", json={"email": intruder, "password": "e2epassword123"})

        # 404 rather than 403: a 403 confirms the row exists and turns the
        # endpoint into an id-enumeration oracle.
        assert client.get("/api/matches", params={"profile_id": profile_id}).status_code == 404
        assert client.get(f"/api/profiles/{profile_id}").status_code == 404

        # And their own skill-gap report is empty rather than borrowed.
        assert client.get("/api/insights/skill-gaps").json()["matches_analyzed"] == 0

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == intruder).one_or_none()
            if user is not None:
                db.delete(user)
                db.commit()
        finally:
            db.close()
