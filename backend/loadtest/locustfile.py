"""M11: burst load across all three entry paths, with queue depth sampled underneath.

Run it:

    cd backend
    locust -f loadtest/locustfile.py --host http://localhost:8000

    # or headless, which is what the numbers in the README came from
    locust -f loadtest/locustfile.py --host http://localhost:8000 \
           --headless -u 60 -r 20 -t 3m --csv loadtest/results/burst

**What this measures, and what it deliberately does not.**

It measures the *API tier*: how long the server takes to authenticate, validate,
write a row, enqueue, and return. That is the number the two structural rules in
the README make a claim about -- "the API tier never does slow work" and "the
feed's read path touches no queue and no worker" -- and it is the only half a
load generator can honestly measure from outside.

It does not measure fetch throughput. Path A submits URLs on the reserved
`.invalid` TLD, which never resolves, so the worker fails them fast and no
third party is touched. Pointing a burst generator at real job boards to
measure our own queue would be both rude and a worse experiment: their latency
would dominate ours. Crawl throughput is measured separately and honestly, by
counting postings per hour against a real board.

**Queue depth is the second deliverable and needs no load generator.** Depth is
sampled straight out of Redis on a background greenlet while the burst runs,
because the interesting question is not "how fast does the API accept work" --
it accepts work very fast, that is the whole design -- but "what happens to the
backlog when it does". A p95 of 12 ms on `POST /api/jobs` means nothing on its
own if `ingest` is 40,000 deep behind it.

**Read and write are separate user classes on purpose.** Mixing them into one
task set averages a sub-millisecond indexed SELECT together with an insert plus
an enqueue, and the blended percentile describes no real user. Section 6.1's
whole argument is that these have different latency budgets, so they are
reported apart.
"""

from __future__ import annotations

import base64
import csv
import time
import uuid

import gevent
from locust import HttpUser, between, events, task

from loadtest.config import (
    EMAIL_PREFIX,
    PASSWORD,
    RESULTS,
    RESUME_PDF_B64,
    UNREACHABLE,
)

RESUME_PDF = base64.b64decode(RESUME_PDF_B64)

_samples: list[dict] = []
_sampler: gevent.Greenlet | None = None


def _register_and_login(client) -> str:
    """Give this simulated user its own account.

    One account per user rather than a shared one, because several endpoints
    are per-user by construction: batch caps are counted per user, and a shared
    login would have every simulated user contending on the same three-open-
    batches limit and measuring the 429 path instead of the happy one.
    """
    email = f"{EMAIL_PREFIX}{uuid.uuid4().hex[:12]}@fitcheck.dev"
    client.post(
        "/api/auth/register",
        json={"email": email, "password": PASSWORD},
        name="/api/auth/register",
    )
    client.post(
        "/api/auth/login",
        json={"email": email, "password": PASSWORD},
        name="/api/auth/login",
    )
    return email


def _create_profile(client) -> int | None:
    """Upload a resume and return its id.

    Needed, not decorative: `/api/matches` and `/api/batches` are scoped to a
    profile the caller owns and answer 422 without one. The first version of
    this file skipped it and produced a 56% error rate that looked like a
    server problem and was entirely the harness measuring its own bad fixtures.
    """
    with client.post(
        "/api/profiles",
        files={"file": ("loadtest.pdf", RESUME_PDF, "application/pdf")},
        name="/api/profiles [upload]",
        catch_response=True,
    ) as response:
        if response.status_code == 201:
            response.success()
            return response.json()["id"]
        response.failure(f"upload returned {response.status_code}")
        return None


class FeedReader(HttpUser):
    """Path B's read path -- the one with a human waiting on it.

    The claim under test is that reading a feed is a pure indexed SELECT with
    no queue and no worker in the way, so this is the class whose p95 actually
    matters for the product.
    """

    weight = 6
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        _register_and_login(self.client)
        self.profile_id = _create_profile(self.client)

    @task(5)
    def read_feed(self) -> None:
        if not self.profile_id:
            return
        self.client.get(
            "/api/matches",
            params={"limit": 25, "profile_id": self.profile_id},
            name="/api/matches [feed]",
        )

    @task(2)
    def read_feed_filtered(self) -> None:
        """Filters go through a different plan than the bare feed."""
        if not self.profile_id:
            return
        self.client.get(
            "/api/matches",
            params={
                "limit": 25,
                "profile_id": self.profile_id,
                "remote_only": True,
                "seniority": ["senior"],
            },
            name="/api/matches [filtered]",
        )

    @task(2)
    def read_insights(self) -> None:
        """The heaviest read in the app: unnests every stored breakdown."""
        self.client.get("/api/insights/skill-gaps", name="/api/insights/skill-gaps")

    @task(1)
    def read_saved(self) -> None:
        self.client.get("/api/matches/saved", name="/api/matches/saved")

    @task(1)
    def whoami(self) -> None:
        """A session lookup: one signed-cookie check plus one Redis GET. The
        floor for every other number here."""
        self.client.get("/api/auth/me", name="/api/auth/me")


class UrlSubmitter(HttpUser):
    """Path A -- one URL, enqueued onto `interactive`, 202 immediately."""

    weight = 3
    wait_time = between(0.5, 1.5)

    def on_start(self) -> None:
        _register_and_login(self.client)
        self.job_ids: list[int] = []

    @task(4)
    def submit_url(self) -> None:
        url = f"{UNREACHABLE}/jobs/{uuid.uuid4().hex[:12]}"
        with self.client.post(
            "/api/jobs",
            json={"profile_id": None, "url": url},
            name="/api/jobs [submit]",
            catch_response=True,
        ) as response:
            if response.status_code == 202:
                self.job_ids.append(response.json()["id"])
                response.success()
            elif response.status_code in (400, 422):
                # A submission the API refused on purpose is not a failure of
                # the API. Recording it as one would make the error rate a
                # measure of this file's fixtures rather than of the server.
                response.success()
            else:
                response.failure(f"unexpected {response.status_code}")

    @task(2)
    def poll_job(self) -> None:
        """What the workspace does while a job is in flight."""
        if not self.job_ids:
            return
        job_id = self.job_ids[-1]
        self.client.get(f"/api/jobs/{job_id}", name="/api/jobs/{id} [poll]")

    @task(1)
    def list_jobs(self) -> None:
        self.client.get("/api/jobs", params={"limit": 20}, name="/api/jobs [list]")


class BatchSubmitter(HttpUser):
    """Path A-bulk -- N URLs in one request, fanned out onto `ingest`.

    The class that generates a backlog on demand, which is the whole reason
    the batch endpoint is useful for this milestone: it makes the queue
    separation argument measurable without waiting for a crawl.
    """

    weight = 1
    wait_time = between(2.0, 5.0)

    def on_start(self) -> None:
        _register_and_login(self.client)
        self.profile_id = _create_profile(self.client)

    @task
    def submit_batch(self) -> None:
        if not self.profile_id:
            return
        urls = "\n".join(
            f"{UNREACHABLE}/batch/{uuid.uuid4().hex[:10]}" for _ in range(25)
        )
        with self.client.post(
            "/api/batches",
            data={"urls": urls},
            params={"profile_id": self.profile_id},
            name="/api/batches [create]",
            catch_response=True,
        ) as response:
            if response.status_code in (202, 400, 409, 429):
                # 409/429 is the open-batch cap doing its job under load, which
                # is a pass for the server and would otherwise read as an error
                # rate climbing with concurrency.
                response.success()
            else:
                response.failure(f"unexpected {response.status_code}")


# --- queue depth, sampled underneath the burst -------------------------------


def _sample_queues() -> None:
    """Record depth per queue once a second for the life of the test."""
    from app.queues import queue_depths

    started = time.time()
    while True:
        try:
            depths = queue_depths()
        except Exception:
            # Instrumentation must never take the test down with it.
            depths = {}
        _samples.append({"t": round(time.time() - started, 1), **depths})
        gevent.sleep(1.0)


@events.test_start.add_listener
def _start_sampler(environment, **_kwargs) -> None:
    global _sampler
    _samples.clear()
    _sampler = gevent.spawn(_sample_queues)


@events.test_stop.add_listener
def _stop_sampler(environment, **_kwargs) -> None:
    if _sampler is not None:
        _sampler.kill()

    if not _samples:
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for sample in _samples for key in sample})
    fields.remove("t")
    path = RESULTS / "queue_depth.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", *fields])
        writer.writeheader()
        writer.writerows(_samples)

    peaks = {name: max(s.get(name, 0) for s in _samples) for name in fields}
    finals = {name: _samples[-1].get(name, 0) for name in fields}
    print("\n--- queue depth ---")
    print(f"samples: {len(_samples)} over {_samples[-1]['t']:.0f}s -> {path}")
    for name in fields:
        print(f"  {name:<12} peak {peaks[name]:>6}   at finish {finals[name]:>6}")
