# FitCheck

Resume/JD matching pipeline with async job processing, explainable scoring, and an operations dashboard.

> **Status:** In active development. The sections below describe the target architecture; see [Roadmap](#roadmap) for what is actually built.

---

## Purpose

FitCheck ingests a resume and a job-posting URL, derives structured profiles from both, and scores the match using two independent signals — semantic similarity (embeddings) and structured skill overlap — returning an inspectable breakdown rather than a bare percentage.

The system is architected around one constraint: fetching an arbitrary job-posting URL is unbounded-latency, high-failure work (200ms–30s, frequent timeouts / malformed HTML / rate limits). That constraint forces everything else — an async job queue, retry/backoff with dead-lettering, and workers that scale independently of the API tier.

Nothing here is decorative; each layer exists because the problem requires it:

| Layer | Why it's required |
| --- | --- |
| **Async job queue** | URL fetching can't happen inside an HTTP request handler |
| **Retry / backoff / dead-letter** | Failure is the common case for remote fetches, not the edge case |
| **Relational storage** | Profiles, jobs, and matches are join-shaped |
| **Background workers** | Parsing + embedding is CPU/IO-heavy, scaled separately from the web tier |
| **Explainable scoring** | A bare match percentage isn't a usable product |

---

## Stack

| Tier | Technology |
| --- | --- |
| Frontend | React + TypeScript (Vite), TanStack Query |
| API | FastAPI (Python 3.11+), Pydantic |
| Storage | PostgreSQL + pgvector |
| Queue | Redis + RQ |

---

## How It Works

### 1. Resume ingestion

PDF/DOCX upload → text extraction → structured candidate profile (skills, years of experience, seniority, education), persisted with promoted columns for filterable fields and a JSONB blob for the raw extraction.

### 2. Job submission

A job-posting URL is submitted, validated, and enqueued; the API returns `202 Accepted` with a job ID immediately. No request thread waits on the fetch.

### 3. Worker processing

A worker pulls the job, fetches the page (respecting `robots.txt`, per-domain rate limiting, size caps, and timeouts), extracts a structured job profile via the same schema used for resumes, generates an embedding, and computes:

- **Semantic similarity** — cosine similarity between resume and JD embeddings, stored via pgvector so there's no second datastore to operate.
- **Structured skill overlap** — weighted intersection of required/preferred skills, bucketed into *matched* / *missing* / *partial* (has the skill, insufficient years).
- **Blended score** — `0.4 · semantic + 0.6 · skill`, with both sub-scores and the skill breakdown always surfaced. The weights are a stated judgment call, not a derived constant.

### 4. Failure handling

Transient failures retry with exponential backoff — not fixed-interval, to avoid thundering-herd re-kills of a recovering dependency. Permanent failures (e.g. `404`) are not retried and land in a dead-letter registry for inspection and requeue.

Job handlers are idempotent: replaying any job converges to the same end state, since RQ delivery is at-least-once.

### 5. Ops dashboard

Queue depth, job-state counts, retry counts, and failure rates, backed by structured logs.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20 LTS
- Docker Desktop (runs Postgres and Redis)

### Infrastructure

```powershell
docker compose up -d
docker compose ps
```

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

Interactive API docs are then available at http://localhost:8000/docs.

> Activation is per-shell — re-run `.\.venv\Scripts\Activate.ps1` in every new terminal, including the ones running workers.

### Workers

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
rq worker --url redis://localhost:6379 default
```

---

## Roadmap

| Milestone | Scope | Status |
| --- | --- | --- |
| M1 | Resume ingestion — upload, text extraction, persisted profile | In progress |
| M2 | Structured profile extraction with normalized skill aliases | Not started |
| M3 | Queue skeleton — submit URL, worker transitions, polling UI | Not started |
| M4 | Real fetching — robots, timeouts, size caps, rate limiting | Not started |
| M5 | Failure handling — error classification, backoff, dead-letter | Not started |
| M6 | Scoring — embeddings, skill overlap, blended score | Not started |
| M7 | Match UI — matched/missing/partial groups, sub-scores | Not started |
| M8 | Ops dashboard | Not started |
| M9 | Load test — p50/p95/p99 under burst | Not started |
| M10 | Hardening and documentation | Not started |

