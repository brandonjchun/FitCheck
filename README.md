# FitCheck

Resume/job-description matching with async ingestion, explainable scoring, and a content-based recommendation feed.

> **Status:** In active development against a fixed project specification. Most of this document describes the *target* architecture. See [Roadmap](#roadmap) for what is actually built today, and [Known Limitations](#known-limitations) for the gaps that are by design.

---

## Purpose

FitCheck derives a structured profile from a resume, derives a structured profile from a job posting using the same schema, and scores the pair on two independent signals — semantic similarity and weighted skill overlap — returning an inspectable breakdown rather than a bare percentage.

There are **three entry paths into one scoring engine.** Two are user-triggered, one is scheduled.

| | Path A — On-demand | Path A-bulk — Batch list | Path B — Recommendation feed |
| --- | --- | --- | --- |
| **Trigger** | User pastes one posting URL | User uploads a `.txt` of URLs | Scheduler tick |
| **Fan-out** | 1 job | N jobs, user-bounded | N jobs, board-bounded |
| **Queue** | `interactive` | `ingest` | `ingest` / `scoring` |
| **Latency budget** | Seconds — a human watches a spinner | Minutes — submit and walk away | Minutes–hours for freshness; sub-second to *read* |
| **Auth** | Optional | Required | Required |

The design constraint that holds it together: **all three paths converge on one ingestion pipeline and one scoring function.** Only the trigger, the fan-out, and the ownership differ. Forking the pipeline would produce multiple extraction code paths that silently drift, and Path A scores would stop being comparable to Path B scores.

Path A-bulk requires authentication for a reason Path A does not: an unauthenticated endpoint that accepts a list and fans out to N outbound fetches is an open request amplifier pointed at third-party servers. One URL is a nuisance; 40,000 is someone else's incident.

---

## Why this shape

Nothing here is decorative. Each layer exists because the problem requires it:

| Layer | Why it's unavoidable |
| --- | --- |
| **Async job queue** | Fetching an arbitrary URL takes 200ms–30s and fails often. This cannot happen inside an HTTP request handler. |
| **Queue priority classes** | The crawler generates backlogs in the thousands, and a batch upload lets a *user* generate one on demand. On one shared queue, an interactive submission waits behind all of it. |
| **Admission control on user fan-out** | Path A-bulk turns one HTTP request into N outbound fetches. Unbounded, that is a self-inflicted DoS on someone else's server and an unbounded queue on your own. |
| **Retry / backoff / dead-letter** | Remote pages time out, rate-limit, and return malformed HTML. Failure is the common case, not the edge case. |
| **Relational storage** | A profile has many skills; a match references a profile and a posting; a posting belongs to a source. Join-shaped. |
| **Background workers** | Parsing and embedding are CPU/IO-heavy and must scale independently of the web tier. |
| **Vector index (ANN)** | Path B cannot linearly scan 10k embeddings per user per refresh. |
| **Two-stage retrieval** | Scoring every (profile, posting) pair is O(N·M). Recall-then-rerank is the only tractable shape. |
| **Authentication + authorization** | Path B is inherently per-user. A recommendation feed with no session is not a feature. |
| **Content-hash gating** | A daily crawl re-sees ~99% unchanged postings. Re-extracting them burns LLM budget for zero information gain. |
| **Explainable scoring** | A bare "72% match" is not a product. In a ranked feed the breakdown is the only thing that makes position 7 defensible. |

---

## Stack

| Tier | Technology |
| --- | --- |
| Frontend | React + TypeScript (Vite), TanStack Query |
| API | FastAPI (Python 3.11+), Pydantic |
| Storage | PostgreSQL 16 + pgvector |
| Queue | Redis 7 + RQ |
| Sessions | Redis, opaque session ID in an `HttpOnly` cookie |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` (384-dim), local |
| Extraction | Swappable LLM provider — Gemini or Ollama |

---

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │        Browser (React + TS)         │
                    │  sign-in · upload · submit · feed   │
                    └──────────────┬──────────────────────┘
                                   │ HTTP/JSON + session cookie
                                   ▼
                    ┌─────────────────────────────────────┐
                    │          FastAPI (API tier)         │
                    │  authn/authz · validate · enqueue   │
                    │  read results · serve ranked feed   │
                    └───┬──────────────────────────┬──────┘
  PATH A / A-bulk       │                          │   PATH B: read precomputed feed
  enqueue → 202         │                          │   (pure indexed SELECT, no slow work)
                        ▼                          ▼
              ┌──────────────────┐       ┌────────────────────────┐
              │      Redis       │       │       PostgreSQL       │
              │  queues:         │       │  users · profiles      │
              │   interactive    │◀─────▶│  sources · url_batches │
              │   ingest         │       │  job_postings (catalog)│
              │   scoring        │       │  ingest_jobs (work log)│
              │   discovery      │       │  matches · feedback    │
              │  sessions        │       │  + pgvector (HNSW)     │
              │  domain buckets  │       └───────────▲────────────┘
              └────────┬─────────┘                   │
                       │ BLPOP                       │
    ┌──────────────────┼──────────────────────┐      │
    ▼                  ▼                      ▼      │
┌───────────┐  ┌────────────────┐    ┌─────────────┐ │
│ discovery │  │    ingest      │    │   scoring   │─┘
│  worker   │─▶│    worker      │───▶│   worker    │
│ enumerate │  │ fetch · parse ·│    │ recall(ANN) │
│  boards   │  │ extract · embed│    │  → rerank   │
└─────┬─────┘  └───────┬────────┘    └─────────────┘
      │ scheduled      │ outbound HTTPS
      ▼                ▼
┌──────────────────────────────────────────────────┐
│  Board APIs (Greenhouse · Lever · Ashby)         │
│  company career pages · LLM API · embeddings     │
└──────────────────────────────────────────────────┘
```

Two structural rules govern the whole system:

1. **The API tier never does slow work.** It authenticates, validates, writes a row, pushes a job onto a queue, and returns `202 Accepted`.
2. **The feed's read path touches no queue and no worker.** Reading it is `SELECT ... ORDER BY final_score DESC LIMIT 50` against an indexed table. All expensive work happened earlier, off the request path. A feed endpoint that scores anything at request time is Path A in a trenchcoat.

---

## How It Works

### 1. Resume ingestion

PDF/DOCX upload → text extraction → structured candidate profile (skills with evidence spans, years of experience, seniority, education). Persisted as promoted columns for the fields that get filtered on, plus a JSONB blob holding the full extraction.

Extraction is an LLM call against a constrained JSON schema, validated by Pydantic — not hand-rolled regex parsing, which breaks on the next file. Malformed model output is treated as a *retryable* error.

Skill names are normalized through an alias map, so "JS", "Javascript", and "ECMAScript" collapse to one canonical token. Unglamorous and load-bearing: under Path B a bad alias map doesn't produce one wrong score, it produces a systematically wrong ranking across the whole catalog. Normalization runs on **read**, so the stored blob keeps the model's original spellings and an alias fix applies retroactively without re-running the LLM.

Each skill also records **where it came from** — `experience`, `project`, `education`, or `skills_list`. Listing "Go" in a technologies section is a claim; shipping a service in Go is evidence, and weighting them identically is what resume keyword-stuffing relies on. When a skill appears in several places the strongest wins, so a skill demonstrated in a job bullet never reads as a bare list entry. On a sample resume this separated 20 keyword-list skills from 12 demonstrated ones, and every demonstrated skill also carried an inferred year count.

### 2. Job submission

- **Path A** — a posting URL is validated and enqueued on `interactive`; the API returns `202` with an ingest job id. No request thread waits on the fetch.
- **Path A-bulk** — a `.txt` of URLs is parsed, normalized, and deduplicated; the API writes one `url_batches` row plus N `ingest_jobs` rows and enqueues all N on `ingest`. The client polls **one** aggregate batch endpoint, not N individual ones. Bulk work is routed to a lower-priority queue than interactive submissions — otherwise a 500-URL upload puts every single-URL submission behind it.
- **Path B** — a scheduled `discover` job enumerates a board and fans out to `ingest`.

### 3. Worker processing

The ingest worker fetches the page (respecting `robots.txt`, per-domain rate limits, size caps, and timeouts), extracts a structured job profile using the same schema as the resume, generates an embedding, and upserts on `canonical_key`.

**The content-hash gate is what makes a repeat crawl cheap.** Fetching is one cheap HTTP GET; extraction and embedding are the expensive steps. If the normalized text hashes to what's already stored *and* the extraction version matches, the worker updates `last_seen_at` and returns — skipping the LLM and the embedding entirely. On a stable board that short-circuits on the large majority of postings.

`extraction_version` is part of that condition on purpose: changing the extraction prompt must force re-extraction despite an unchanged hash.

### 4. Scoring

Two independent signals, blended, with both sub-scores and the full breakdown always surfaced:

- **Semantic similarity** — cosine similarity between resume and JD embeddings, stored via pgvector so there's no second datastore to operate and no consistency gap between a posting row and its vector.
- **Structured skill overlap** — weighted intersection of required and preferred skills. Three buckets, not two: **matched**, **missing**, and **partial** (has the skill, insufficient years). The partial bucket is what makes the output feel intelligent.
- **Blend** — `0.6 · semantic + 0.4 · skill`. The weights are a stated judgment call, not a derived constant. Skill overlap is the sharper signal but inherits every extraction error beneath it, so a loosely-worded posting scores confidently low on a role that fits; semantic similarity degrades more gently, and ranking a feed is not the same as making a hiring decision. The breakdown still explains the placement. You'd tune against labeled data if you had any.

Embeddings capture theme but are poor at hard requirements — a resume can score highly against a JD while missing its single mandatory skill. That's why the skill overlap is computed explicitly rather than trusted to the vector.

Scores carry a `scorer_version`. When blend weights or the alias map change, every stored score becomes incomparable to every new one, and a feed sorted across two scoring generations is silently wrong. Versioning makes rescoring a query and lets the feed refuse to mix generations.

### 5. Recommendations — recall, then rank

A 10,000-posting catalog against 1,000 profiles is 10⁷ pairs per refresh. The resolution is the one every search system uses:

- **Stage 1 — recall.** Narrow 10,000 → ~200 with the HNSW vector index plus hard SQL filters. Cheap, indexed, approximate.
- **Stage 2 — rerank.** Run the full skill overlap and blend on those ~200 in Python — pure operations over already-extracted JSONB, microseconds per pair — and persist the top 50 to `matches`.

Two orders of magnitude less work, with recall loss confined to postings that were unlikely to survive reranking anyway.

**The gotcha is post-filtering.** An ANN index finds nearest neighbours *first*, then applies `WHERE`. A restrictive filter can return far fewer rows than the `LIMIT`. The common case is handled with a partial index on `closed_at IS NULL`, so closed rows are never retrieved at all — and verified with `EXPLAIN ANALYZE` rather than reasoned about from first principles.

### 6. Failure handling

Transient failures retry with exponential backoff **plus jitter** — not fixed-interval, because a fan-out creates a failure *cohort* by construction: hundreds of jobs against one host, enqueued in the same second, failing together and retrying together. Without jitter those retries are perfectly synchronized.

Permanent failures are not retried. A `404` will never succeed, and retrying it three times wastes two minutes and a worker slot. On a crawled posting a `404` carries a second meaning — the role was taken down — and is handled as a tombstone rather than a bare dead-letter.

Job handlers are **idempotent**. RQ delivery is at-least-once: a worker can complete its side effects and die before reporting success. Under Path B this stops being an edge case and becomes the steady state, since the crawl deliberately re-runs ingestion on every posting it has ever seen. Handlers use keyed upserts and early-exit on already-done, never blind `INSERT` or `counter = counter + 1`.

### 7. Ops

Queue depth per queue, jobs by status, failure rate, crawl freshness per source, and a dead-letter list with requeue. Cheap to build because `ingest_jobs` *is* the audit log — it exists the instant work is requested and survives every attempt failing.

---

## Data Model

Two tables carry names that are easy to confuse, so the split is deliberate:

- **`ingest_jobs`** — units of async work. A *work record*: exists the moment work is requested, persists even if every attempt fails.
- **`job_postings`** — the things people apply to. A *result record*: exists only on success, and outlives any individual work record since one posting is re-crawled dozens of times.

| Table | Role |
| --- | --- |
| `users` | Identity. Argon2id password hashes. |
| `profiles` | One uploaded resume and everything derived from it. Owned by a user; exactly one `is_active` per user, enforced by a partial unique index. |
| `sources` | Path B crawl targets. Per-source kill switch and circuit-breaker state. |
| `job_postings` | The canonical catalog, decoupled from any user. Unique on `canonical_key`. |
| `ingest_jobs` | Work records across all job kinds. Partial unique index on in-flight work only. |
| `url_batches` | One uploaded URL list. Stores counts, not status. |
| `matches` | One scored (profile, posting) pair. All three paths write here. |
| `match_feedback` | Interested / not-interested / applied. |

**Global dedupe on `canonical_key`.** Two users submitting the same Greenhouse posting must land on one posting row, not two, or the crawler creates a third. The key is `{source_kind}:{board_token}:{external_id}` for crawled postings and `url:{sha256(normalize(url))}` for one-off submissions. URL normalization strips tracking params, lowercases the host, drops the fragment, and removes a trailing slash — without it, `?utm_source=twitter` creates a duplicate posting and a duplicate extraction bill.

**Batch progress is derived, never counted.** `url_batches` deliberately has no `completed_count` column: N concurrent workers doing `count = count + 1` double-count under replay and drift from the rows they summarize. Progress is a `GROUP BY status` over `ingest_jobs` filtered on an indexed `batch_id`.

**Closed postings are tombstoned, never deleted.** `matches` rows reference postings, so deleting one either cascades away a user's history or raises a foreign-key error mid-crawl. `closed_at IS NOT NULL` also gives an honest UI state — *"this role appears to have been filled"* — which beats the row vanishing.

**Closure detection is guarded on full enumeration success.** Absence-implies-closed is only sound if the complete list was actually seen. If enumeration returned a partial page before timing out, mass-closing would silently empty a board out of every user's feed. `last_success_at` is tracked separately from `last_crawled_at` precisely so that distinction is representable.

ORM models and Pydantic schemas stay in separate modules. Conflating them makes a column rename a breaking API change, and leaks `password_hash` the first time an ORM object is returned directly.

---

## Queue Topology

| Queue | Producer | Latency target |
| --- | --- | --- |
| `interactive` | Path A URL submit | Seconds — a human is watching |
| `ingest` | Path A-bulk upload, discovery fan-out | Minutes–hours — the bulk |
| `scoring` | Profile upload, rescore | Minutes — CPU-bound, no external network |
| `discovery` | Scheduler tick | Minutes — cheap, low volume, high fan-out |

A crawl of 40 boards enqueues ~4,000 jobs in seconds. At ~3s each with 4 workers that backlog drains in ~50 minutes — so on a single shared queue, a user pasting one URL at minute 2 waits **48 minutes** for a request that takes 3 seconds to service. That is head-of-line blocking, and no amount of retry logic fixes it. Only queue separation does.

The batch upload makes that argument measurable without waiting for the crawler: a `.txt` of 2,000 URLs produces a reproducible backlog on demand, so the queue design is justified by a number that can be re-run rather than an assertion.

RQ drains queues in the order given on the command line, so interactive workers fall back to bulk work when idle and no capacity is wasted — but a user submission jumps the entire backlog on the next poll:

```powershell
# interactive worker: drains user work first, helps with backlog when idle
rq worker --url redis://localhost:6379 interactive scoring ingest

# bulk worker: never touches interactive work
rq worker --url redis://localhost:6379 ingest
```

---

### Scheduler and discovery

`discover_source` enumerates one board; nothing called it until M8's scheduler landed. A dedicated `scheduler` service asks every few minutes which sources are due and enqueues them onto `discovery`, where `worker-discovery` drains them.

"Due" is computed in SQL against `last_crawled_at`, not in Python. Both sides of that comparison have to come from one clock — the same defect that made closure detection tombstone live boards came from mixing a worker's clock with Postgres's.

The stamp happens *before* enumerating, which is what makes ticking safe: a board mid-crawl is no longer due, so a fast tick cannot stack two crawls of the same board on somebody else's server.

```bash
# one process, deliberately -- see docker-compose.yml
docker compose up -d scheduler worker-discovery
```

## Fetching Responsibly

Path A makes one request when a human clicks a button. Path B makes thousands, unattended, on a schedule. The rules are non-negotiable:

- Respect `robots.txt`, with the parsed result cached per host.
- Build the catalog from **board APIs, not general scraping** — Greenhouse, Lever, Ashby, and Workable expose public JSON endpoints intended for programmatic access. LinkedIn and Indeed prohibit scraping and enforce it. An API also gives stable IDs, which is what `canonical_key` is built from.
- Identify with a real `User-Agent` including contact info. No browser impersonation to evade detection.
- **Rate-limit per domain across all workers** — a Redis token bucket keyed on hostname, consumed atomically. In-process limiting is useless here: four workers each politely limiting themselves to 1 rps produce 4 rps at the target.
- Cap response size and set timeouts, so one pathological URL can't hang a worker for its full job timeout.
- Circuit-break per source after N consecutive failures.
- **Bound what one user can trigger.** A batch upload turns one request into N outbound fetches, and a byte cap is not that bound — a 2 MB `.txt` inside the existing upload limit is roughly 40,000 URLs. Enforce a hard cap on URLs per batch, a cap on concurrent open batches per user, and a per-user daily budget. Report every rejection back with a count rather than truncating silently: a batch that quietly ingests 500 of 4,000 lines is worse than one that refuses, because the user can't tell which 3,500 are missing.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20 LTS *(not needed until the frontend exists — see [Roadmap](#roadmap))*
- Docker Desktop — runs Postgres, Redis, and the workers

### Infrastructure

```powershell
docker compose up -d
docker compose ps
```

This brings up Postgres (the `pgvector/pgvector:pg16` image, so the extension is available without rebuilding the volume later), Redis, and a worker container. Both datastores have healthchecks, so `docker compose ps` reports `healthy` rather than just `running`, and the worker waits on readiness instead of dying on its first query.

The `pgdata` named volume is not optional — without it, every `docker compose down` destroys the database.

### Configuration

There are **two** env files, and they are not interchangeable:

| File | Read by | Purpose |
|---|---|---|
| `.env` (root) | Docker Compose | `POSTGRES_PASSWORD`, `REDIS_PASSWORD` — substituted into `docker-compose.yml` |
| `backend/.env` | the Python process | `DATABASE_URL`, `REDIS_URL`, LLM provider settings |

Copy both examples and fill them in:

```powershell
copy .env.example .env
copy backend\.env.example backend\.env
```

Generate the two infrastructure passwords with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

The passwords in `backend/.env` must match the ones in the root `.env` — the containers and the host-run process reach the same two services. Both files are gitignored; the `.example` files are committed so a fresh clone knows what is required.

Compose only auto-loads a root-level `.env` for substitution, so `docker-compose.yml` names `./backend/.env` explicitly under `env_file:` and then overrides `DATABASE_URL` and `REDIS_URL` — the host-facing values point at `localhost`, which inside the compose network would mean the container itself.

### A note on the exposed ports

Postgres and Redis are published to `127.0.0.1` rather than the unqualified `5432:5432` form, and Redis runs with `--requirepass`. Both matter the moment this runs anywhere with a public IP:

- Docker publishes to **all interfaces** by default, and it does so by writing DNAT rules into the `DOCKER` iptables chain — which is traversed **before** UFW's rules. `ufw deny 6379` does not close a Docker-published port. The firewall reports it blocked while it is open to the internet.
- Redis ships with **no password**, is among the most scanned ports on the internet, and here it holds the session store — read access to it is read access to every logged-in account.

If you deploy, prefer removing the `ports:` blocks entirely. The API and workers reach both services by service name on the Compose network and never need host publishing.

### Before deploying to a cloud VM

**Require IMDSv2.** Every major cloud exposes an instance metadata service at `169.254.169.254` that answers only to requests originating on the instance — and returns the instance's IAM role credentials. Because this app fetches user-submitted URLs, that endpoint is reachable via SSRF unless the fetcher validates its targets (see [Known limitations](#known-limitations)).

IMDSv2 requires a `PUT` with a header to obtain a token before any read, which a forged `GET` cannot perform. It closes the highest-value target with one setting:

```bash
# AWS: enforce on an existing instance
aws ec2 modify-instance-metadata-options \
  --instance-id i-xxxxxxxx \
  --http-tokens required \
  --http-endpoint enabled
```

GCP sets `Metadata-Flavor: Google` as a required header by default. DigitalOcean, Hetzner, and Linode expose metadata at the same address with **no** token requirement — on those, blocking egress to `169.254.0.0/16` from the worker is the equivalent control.

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Interactive API docs at http://localhost:8000/docs. Exercise every endpoint there before writing frontend code.

> Activation is per-shell — re-run `.\.venv\Scripts\Activate.ps1` in every new terminal.

### A development account

Every endpoint except `/health` and the auth routes now requires a session, so a fresh database needs an account before anything is reachable. Either register through `POST /api/auth/register`, or seed the standing dev one:

```powershell
cd backend
python -m scripts.seed_dev_user     # dev@fitcheck.dev / devpassword123
```

This is a script rather than a migration on purpose: migrations run in every environment, and a row with a published password is not something to create in production by accident. It is idempotent.

> The address is `.dev`, not `.local`. `.local`, `.test`, `.localhost`, `.invalid`, and `.example` are IANA special-use domains, and the `email-validator` behind `EmailStr` rejects them — seeding one produces an account that exists in the database and can never log in.

Set `SESSION_SECRET` in production. The default in `config.py` is a development convenience and is published in this repository; rotating it invalidates every outstanding cookie, which is the intended response to a suspected leak. `SESSION_COOKIE_SECURE` must also be turned on anywhere there is TLS — it is off by default because a `Secure` cookie is never sent over plain-http localhost.

### Workers

Workers run in Docker rather than on the Windows host, and the reason is specific: RQ forks a child process per job so a crashed job can't take the worker down with it. `fork()` doesn't exist on Windows, so a host-run `rq worker` needs `SimpleWorker`, which executes jobs in-process and loses that isolation.

There are two worker classes, and the difference between them is the point of running four queues:

| Service | Drains | Role |
| --- | --- | --- |
| `worker-interactive` | `interactive scoring ingest` | Serves user-facing work first; helps clear backlog when idle |
| `worker-bulk` | `ingest` | Batch and crawl work only — can never starve a user request |

RQ checks queues in the order given on the command line every time it looks for work. So the interactive worker pitches in on a backlog when nothing else is waiting, but the instant a user submits a URL, that job is the next one it picks up — ahead of however many thousand batch items are queued behind it.

```powershell
docker compose logs -f worker-interactive    # follow user-facing work
docker compose up -d --scale worker-bulk=4   # drain a batch faster
```

Scaling `worker-bulk` adds throughput without any of those processes being able to take capacity from an interactive submission.

### Tests

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest
```

**The suite provisions its own database and Redis keyspace.** On the first run
it creates `fitcheck_test` beside your development database, migrates it with
`alembic upgrade head`, and uses Redis db `1`. Both are derived from
`DATABASE_URL` and `REDIS_URL`, so a fresh clone needs no extra configuration;
`TEST_DATABASE_URL` and `TEST_REDIS_URL` override them for CI.

This is not tidiness. `docker compose up` leaves four workers and a scheduler
draining `interactive`, `ingest`, `scoring`, and `discovery` against the
development Redis — so when the tests shared it, those containers **consumed
jobs the tests had enqueued**, and any worker could bump the `gate:hits`
counters a test was asserting on. The suite failed a different handful of tests
on every run while each one passed in isolation. With the split it is six-for-six
clean, and you can run it without stopping a crawl.

Because the test database is disposable, the session start truncates it and
flushes the Redis keyspace. Both are guarded: if the test URLs ever resolve to
the development ones, or the Redis db is `0`, the run aborts before touching
anything rather than proceeding.

```powershell
# point somewhere else entirely, e.g. in CI
$env:TEST_DATABASE_URL = "postgresql+psycopg://ci:pw@localhost:5432/ci_db"
$env:TEST_REDIS_URL    = "redis://localhost:6379/9"
```

---

## Roadmap

Milestone numbering follows the project specification. **This differs from the earlier version of the spec** — the old M3 was the queue skeleton and auth did not exist. Work completed under the old numbering carries over; the earlier M3 is now part of M4.

Path A ships first and completely. Path B is built on top of a pipeline that already works.

| Week | Milestone | Definition of done | Status |
| --- | --- | --- | --- |
| 1 | Environment | Docker up with pgvector; `/health` returns 200; Alembic runs | ✅ Done |
| 2 | **M1** Resume ingestion | Upload PDF/DOCX → text extracted → `profiles` row → visible at `/docs` | ✅ Done |
| 3 | **M2** Structured profile | LLM extraction validated by Pydantic; skill aliases normalized; `extraction_version` stored | ✅ Done |
| 4 | **M3** Auth | Register/login/logout; Argon2id; Redis sessions; `owned_profile` dependency; another user's profile returns 404 | ✅ Done |
| 5–6 | **M4** Queues + batch | Four queues declared and separated; submit URL → `202`; `.txt` batch → N `ingest` jobs + one aggregate status endpoint, capped; polling UI | ✅ Done |
| 7 | **M5** Real fetching | Robots, timeouts, size caps, Redis per-domain token bucket, HTML→text, URL normalization → `canonical_key` | ✅ Done |
| 8 | **M6** Failure handling | Error classification, backoff with jitter, dead-letter registry, requeue endpoint | ✅ Done |
| 9 | **M7** Scoring (Path A) | Embeddings in pgvector; skill overlap; blended score persisted to `matches`. **Path A is demoable here** | ✅ Done |
| 10–11 | **M8** Catalog + crawler | `sources` seeded with 5 boards; `discover` enumerates; fan-out; content-hash hit rate measured; guarded closure detection | ✅ Done — gate hit rate **100%** on a steady-state re-crawl |
| 12 | **M9** Recommendations | HNSW index; two-stage recall→rerank; `score_profile` job; feed is a pure indexed SELECT; p95 vs exact scan | ✅ Done — **5.8× at 10k rows, 8.9× at 50k** |
| 13 | **M10** Feed UI + ops | Ranked feed with filters and inline explanation; feedback capture; ops dashboard | ✅ Done |
| 14 | **M11** Load test + document | Locust burst on all paths; p50/p95/p99; queue depth under load | ⏳ Next |

**Ship M1–M7 before touching M8.** A working single-path pipeline that fetches and fails gracefully is a complete, demoable system. A half-built crawler bolted to a queue that can't survive a timeout is not. That ordering was followed.

### What's actually in the repo today

- **M1/M2 complete.** Resume upload, PDF/DOCX text extraction, LLM extraction behind a swappable provider seam (Gemini and Ollama), skill alias normalization, `extraction_version` stamped on every extraction, request size cap enforced in ASGI middleware, and a pytest suite covering documents, extraction, skills, providers, middleware, and endpoints. That suite was 171 tests at M2 and is **661** across the whole backend today.

  Two design points worth knowing. **Skill normalization happens on read, not on write** — `profiles.extracted` holds exactly what the model returned, so adding an alias applies retroactively to every stored profile with no backfill and no second LLM call. And **`extraction_version` is only bumped by prompt or schema changes**, not alias changes, because the latter no longer alter what was extracted.
- **M3 complete.** Register, login, logout, and `/me`; Argon2id password hashing; opaque session ids in `HttpOnly` cookies with the body in Redis; and an `owned_profile` dependency that every profile-scoped endpoint takes so the ownership predicate cannot be left out of one handler. Reaching for another user's profile returns **404, not 403** — a 403 confirms the row exists and turns the endpoint into an id-enumeration oracle.

  Sessions are server-side on purpose: revocation is a `DEL`, so logout takes effect immediately and a replayed cookie fails. A JWT would still verify.

- **M4 backend complete.** Four separated queues with two worker classes, plus the batch `.txt` upload that fans out onto `ingest`. Caps are counts rather than bytes — the 2 MB body limit bounds bytes, and 2 MB of text is roughly 40,000 URLs. Batch progress is derived with a `GROUP BY`, never a stored counter that N workers would double-count.
- **M5 complete.** Real fetching: robots-checked and cached, connect/read timeouts, a size cap enforced *while streaming* rather than from a `Content-Length` a server can omit or lie about, and a Redis token bucket shared across worker processes.

- **M6 complete.** Errors classified transient vs permanent at the provider seam, exponential backoff with jitter, a dead-letter registry, and a requeue endpoint on the ops dashboard.
- **M7 complete.** MiniLM embeddings (384-d) in pgvector for both profiles and postings, weighted skill overlap with partial credit, and a `0.6 semantic / 0.4 skill` blend whose full breakdown is persisted to `matches` and rendered inline. The weights are a stated judgment call, measured against five postings of known correct ordering — see `DECISIONS.md` D-059 and D-066.

  The blend was **inverted from `0.4/0.6` at `SCORER_VERSION` 2**, and the same five postings were recomputed rather than re-argued: the known-correct ordering is unchanged, with the unqualified posting still ranked below every qualified one. What moved is a trade — the qualified/unqualified gap narrows from 0.529 to 0.294, while discrimination *among* qualified roles widens from 0.122 to 0.183. A feed spends nearly all its time on the second question. Scores carry the version because a weight change makes an old score incomparable to a new one.
- **M8 complete.** Five job boards crawled through their public APIs (Greenhouse, Lever, Ashby), a scheduler that decides when each is due, fan-out onto `ingest`, and closure detection guarded on full-enumeration success.

  **The content-hash gate hit 100% on a steady-state re-crawl** — 98 of 98 postings re-enumerated with unchanged content skipped extraction and embedding entirely, turning what would have been 98 LLM calls into zero. Reported live on the ops dashboard rather than asserted here.

  Two failures worth recording because both were silent. Closure detection compared a *worker* timestamp against `last_seen_at` values stamped by *Postgres*, so any clock skew tombstoned the entire live board while logging a successful crawl. And the crawler never triggered extraction, so 549 of 565 postings sat unembedded and invisible to the feed.
- **M9 complete.** An HNSW index partial on open postings, two-stage recall→rerank, a `score_profile` job, and a feed endpoint that is a pure indexed SELECT.

  **Measured against an exact scan** with `scripts/bench_recall.py`, which asserts on the query plan rather than trusting a timing:

  | rows | exact p95 | HNSW p95 | speed-up |
  | --- | --- | --- | --- |
  | 10,000 | 8.08 ms | 1.40 ms | **5.8×** |
  | 50,000 | 19.00 ms | 2.14 ms | **8.9×** |

  The exact scan grows linearly while the index grows logarithmically, which is the entire argument for having one. That harness also caught a regression no test would have: the recall filter used `jsonb_typeof(extracted) = 'object'`, an opaque call the planner cannot estimate, so it silently chose a sequential scan and the index sat unused.
- **M10 complete.** Ranked feed with read-time filters and inline explanations, feedback capture into an append-only `match_feedback` table, and an ops dashboard showing per-queue depth, per-source crawl freshness, and the gate hit rate. Structured logging via `structlog` with request- and job-scoped correlation ids.

  Two tabs unlock on sign-in. **Insights** aggregates every stored breakdown to rank the requirements you fail most often — the one thing 50 per-match explanations can say that one cannot. **Saved** is the application tracker the feedback buttons feed, showing each posting once at its latest stage.

  Verified through the whole stack, not just in tests — API → `ingest` queue → containerised worker → `job_postings`. A four-URL batch drained in 4 seconds: two succeeded, two went straight to `dead` (a 404 and a robots disallow) without consuming a single retry.

- **Frontend complete through M10.** Landing, sign-in, workspace, ranked feed, insights, saved, and the ops dashboard, all wired to session auth. The batch progress view that M4's definition of done asked for is there too — an aggregate poll per batch rather than one per job, with the bar scaled off the batch's own `total` so a missing job row shows as a short bar instead of rescaling to look finished. 127 component and hook tests, `tsc -b` and `oxlint` clean.

  **Nobody has run it and looked at it.** Everything above is verified by typecheck, lint, and tests; layout and spacing are not verified by any of those.

- **Retry behaviour is proven against a real worker.** `tests/test_retry_integration.py` drives an RQ worker through the whole schedule — transient failure, reschedule, backoff, exhaustion, dead-letter — plus the recovery case, without which every other assertion is also satisfied by a worker that retries and can never succeed.

  The backoff is 10s / 60s / 300s, so waiting it out would cost six minutes per run. It doesn't: the fact under test is that the worker parks the job at `now + interval`, which is readable from the scheduled registry the instant it fails, and the test then rewrites that timestamp into the past and lets RQ's own scheduler release it. Sixteen tests in four seconds.

  **It found two defects that no direct call to the task could.** Both were about how the row read *relative to what RQ was still going to do*, which a unit test has no RQ to be relative to. `failed` was in `TERMINAL_STATUSES`, so the workspace stopped polling on the first transient error and froze the row on "Failed — will retry" without ever showing the retry succeed. And `_record_failure` compared `attempts >= MAX_RETRIES` when `Retry(max=N)` allows N retries *after* the first run — so the dead-letter list reported a job as having exhausted its retries for one full execution before it had.

### Known rework, in the order it bites

1. ~~**Ownership columns on a populated table.**~~ Done in M3. `profiles` now carries `user_id` (NOT NULL, indexed, cascade) and `is_active` with the partial unique index. The pre-ownership dev profiles were deleted rather than backfilled, which is the right call for test data and the wrong one for real data — the same migration against production would assign existing rows to a real owner first.
2. ~~**Dedupe scope.**~~ Done in M5. `job_postings.canonical_key` is globally unique, so two users submitting one posting land on one row. `jobs`' per-profile `UNIQUE (profile_id, url_hash)` stays — it dedupes *work*, which is a different question from deduping *postings*.
3. ~~**URL normalization.**~~ Done in M5, and the reasoning is genuinely inverted rather than extended. `hash_url` still does not normalize, which remains right for a per-profile key where re-fetching beats serving the wrong cached posting. `canonical_key_for_url` does normalize, because for a global key un-normalized means one posting stored twice under two tracking URLs and extracted twice at full LLM cost.

None of that is M1–M3 rework.

---

## Known Limitations

Stated rather than hidden. The design-level ones:

- **No collaborative filtering, and there cannot be.** "Users like you liked this posting" needs an interaction matrix across many users and items. On day one there are zero interactions, and at this scale there will never be enough. Path B is purely content-based, deliberately.
- **Career changers rank badly.** An embedding reflects what you *have* done, not what you *want* to do — a backend engineer targeting ML roles matches backend postings. Partial mitigation would be embedding the resume concatenated with a stated target role.
- **Blend weights are hand-set.** `0.6/0.4` is a judgment call, and inverting it at v2 was reasoning rather than fitting. It also amplifies the generic-posting problem directly above: weighting semantic higher is exactly what surfaces a vaguely-worded JD to more people. `match_feedback` is the path out: with a few thousand labels, a learning-to-rank model over the existing features could *derive* the weights instead of asserting them. The data collection is built; the model is not.
- **Generic postings surface for everyone.** A vaguely-worded JD sits close to many resumes in embedding space. Mitigation direction is an IDF-shaped correction penalizing postings with unusually high mean similarity.
- **Cross-source duplicates.** The same role posted to a company's own site *and* its Greenhouse board yields two canonical keys and two rows. True identity resolution needs fuzzy matching on `(company, title, location)` plus an embedding similarity threshold. Out of scope.
- **ANN retrieval is approximate.** It can miss true nearest neighbours. That is an acceptable trade for a recommendation feed and would be unacceptable for, say, financial reconciliation.
