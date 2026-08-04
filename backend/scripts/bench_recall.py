"""Measure what the HNSW index actually bought.

Spec sections 3.5 and 8.5 both say the measurement is the deliverable, not the
index -- "I compared exact scan, HNSW, and HNSW-with-partial-index at 10k rows
and here are the p95 numbers" is the artifact. This produces those numbers.

**Why a synthetic catalog is a prerequisite rather than a shortcut.** The real
catalog is ~1,300 postings. Postgres correctly chooses a sequential scan at
that size, so benchmarking against it compares a sequential scan to a
sequential scan and reports, accurately, no difference. The index only starts
paying above roughly a few thousand rows, so measuring its effect requires
first having enough rows for it to matter. That is a fact about the
measurement, not a way of flattering the result.

**Random vectors, and what that costs.** Embeddings here are random unit
vectors rather than real MiniLM output, because generating 10,000 real ones
takes far longer than the benchmark and the timing does not depend on what the
vectors mean. It does affect one thing worth stating: random high-dimensional
vectors are close to mutually orthogonal, which makes the nearest-neighbour
graph less clustered than a real corpus. HNSW generally does *better* on
clustered data, so treating these numbers as a lower bound on the real
speed-up is the honest reading.

**Three configurations, per section 11.2:**

    1. exact      -- sequential scan, index access disabled
    2. hnsw       -- a full (non-partial) HNSW index
    3. hnsw_part  -- the partial index this project actually ships

The third is the one in production; the middle one exists to show what the
partial predicate is worth on its own, which is otherwise asserted.

Every configuration is verified with EXPLAIN before it is timed. A query that
silently falls back to a sequential scan returns identical rows slightly
slower, and would otherwise be indistinguishable from a fast index.

Usage:
    python scripts/bench_recall.py                 # 10k rows, 200 queries
    python scripts/bench_recall.py --rows 50000
    python scripts/bench_recall.py --cleanup-only  # remove leftovers and exit
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.embeddings import EMBEDDING_DIM  # noqa: E402
from app.logging_setup import configure_logging  # noqa: E402
from app.retrieval import RECALL_LIMIT  # noqa: E402

logger = logging.getLogger("bench")

# Every synthetic row carries this prefix in `canonical_key`, which is what
# makes cleanup exact rather than approximate. Nothing else in the catalog can
# collide with it: real keys are `url:<hash>` or `<kind>:<token>:<id>`.
BENCH_PREFIX = "bench:"

FULL_INDEX = "bench_embedding_hnsw_full"


def _random_unit_vector(rng: random.Random) -> list[float]:
    values = [rng.gauss(0.0, 1.0) for _ in range(EMBEDDING_DIM)]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


def cleanup(db) -> int:
    """Remove synthetic rows and any index this script created.

    Runs at start as well as at the end: a previous run killed partway
    through leaves rows behind, and silently benchmarking against them would
    mean the row count printed in the results is not the row count measured.
    """
    db.execute(text(f"DROP INDEX IF EXISTS {FULL_INDEX}"))
    result = db.execute(
        text("DELETE FROM job_postings WHERE canonical_key LIKE :p"),
        {"p": f"{BENCH_PREFIX}%"},
    )
    db.commit()
    return result.rowcount or 0


def seed(db, rows: int, rng: random.Random) -> None:
    """Insert `rows` synthetic postings.

    Batched inserts rather than one statement per row: at 10,000 rows the
    round-trip cost dominates and seeding becomes slower than the benchmark it
    exists to enable.

    A tenth of them are closed. That ratio is what makes the partial index
    measurable at all -- against an all-open catalog the partial and full
    indexes contain identical rows and the comparison is vacuous. Ten percent
    is a deliberately *conservative* stand-in for a real crawl, where
    tombstones accumulate every day the catalog runs and the gap only widens.
    """
    batch = 500
    inserted = 0

    while inserted < rows:
        chunk = min(batch, rows - inserted)
        values = []
        params: dict = {}

        for i in range(chunk):
            n = inserted + i
            params[f"k{n}"] = f"{BENCH_PREFIX}{n}"
            params[f"u{n}"] = f"https://bench.invalid/{n}"
            params[f"h{n}"] = f"benchhash{n}"
            params[f"e{n}"] = _vector_literal(_random_unit_vector(rng))
            # Every tenth row is closed, so the partial index has something to
            # exclude.
            closed = "now()" if n % 10 == 0 else "NULL"
            values.append(
                f"(:k{n}, :u{n}, :h{n}, 'benchmark posting body',"
                f" CAST(:e{n} AS vector), '{{\"skills\": []}}'::jsonb, 1, {closed})"
            )

        db.execute(
            text(
                "INSERT INTO job_postings "
                "(canonical_key, url, content_hash, raw_text, embedding,"
                " extracted, extraction_version, closed_at) VALUES "
                + ",".join(values)
            ),
            params,
        )
        db.commit()
        inserted += chunk
        logger.info("seeded %d/%d", inserted, rows)


RECALL_SQL = """
SELECT id, 1 - (embedding <=> CAST(:vec AS vector)) AS semantic_score
  FROM job_postings
 WHERE closed_at IS NULL
   AND embedding IS NOT NULL
   AND extraction_version IS NOT NULL
 ORDER BY embedding <=> CAST(:vec AS vector)
 LIMIT :limit
"""


def _configure(db, mode: str) -> None:
    """Put the session into one of the three measured states."""
    if mode == "exact":
        # Index access disabled for this session, which is how you measure the
        # baseline without dropping an index the rest of the system needs.
        db.execute(text("SET enable_indexscan = off"))
        db.execute(text("SET enable_bitmapscan = off"))
    else:
        db.execute(text("SET enable_indexscan = on"))
        db.execute(text("SET enable_bitmapscan = on"))


def _plan(db, vec: str) -> str:
    rows = db.execute(
        text("EXPLAIN " + RECALL_SQL), {"vec": vec, "limit": RECALL_LIMIT}
    ).all()
    return "\n".join(r[0] for r in rows)


def _time_queries(db, vectors: list[str]) -> list[float]:
    timings: list[float] = []
    for vec in vectors:
        start = time.perf_counter()
        db.execute(text(RECALL_SQL), {"vec": vec, "limit": RECALL_LIMIT}).all()
        timings.append((time.perf_counter() - start) * 1000.0)
    return timings


def _percentiles(timings: list[float]) -> dict[str, float]:
    ordered = sorted(timings)
    return {
        "p50": statistics.median(ordered),
        "p95": ordered[int(len(ordered) * 0.95) - 1],
        "p99": ordered[int(len(ordered) * 0.99) - 1],
        "mean": statistics.fmean(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()

    configure_logging()
    rng = random.Random(args.seed)
    db = SessionLocal()

    try:
        removed = cleanup(db)
        if removed:
            logger.info("removed %d leftover rows from a previous run", removed)
        if args.cleanup_only:
            return 0

        logger.info("seeding %d synthetic postings", args.rows)
        seed(db, args.rows, rng)

        # Without fresh statistics the planner is choosing between an index
        # and a scan on row estimates from before these rows existed, so the
        # plan it picks says nothing about the catalog being measured.
        db.execute(text("ANALYZE job_postings"))
        db.commit()

        # The same query vectors for every configuration. Different vectors
        # per configuration would fold vector-to-vector variance into the
        # between-configuration comparison, which is the only comparison this
        # script exists to make.
        vectors = [
            _vector_literal(_random_unit_vector(rng))
            for _ in range(args.queries + args.warmup)
        ]
        warmup, measured = vectors[: args.warmup], vectors[args.warmup :]

        results: dict[str, dict] = {}

        for mode in ("exact", "hnsw", "hnsw_part"):
            if mode == "hnsw":
                logger.info("building full (non-partial) HNSW index")
                db.execute(
                    text(
                        f"CREATE INDEX {FULL_INDEX} ON job_postings "
                        "USING hnsw (embedding vector_cosine_ops)"
                    )
                )
                db.execute(text("ANALYZE job_postings"))
                db.commit()
            if mode == "hnsw_part":
                # Drop the full index so the planner has to use the partial
                # one; with both present it would pick whichever it costed
                # lower and the label on these numbers would be a guess.
                db.execute(text(f"DROP INDEX IF EXISTS {FULL_INDEX}"))
                db.commit()

            _configure(db, mode)

            plan = _plan(db, measured[0])
            uses_index = "Index Scan" in plan
            expected = mode != "exact"
            if uses_index != expected:
                logger.error(
                    "%s: expected index_use=%s but plan says otherwise:\n%s",
                    mode,
                    expected,
                    plan,
                )

            _time_queries(db, warmup)
            timings = _time_queries(db, measured)
            stats = _percentiles(timings)
            results[mode] = {**stats, "uses_index": uses_index, "plan": plan}

            logger.info(
                "%-10s p50=%7.2fms p95=%7.2fms p99=%7.2fms index=%s",
                mode,
                stats["p50"],
                stats["p95"],
                stats["p99"],
                uses_index,
            )

        _report(args, results)
        return 0
    finally:
        try:
            db.execute(text("RESET enable_indexscan"))
            db.execute(text("RESET enable_bitmapscan"))
            cleanup(db)
            logger.info("synthetic rows removed")
        finally:
            db.close()


def _report(args, results: dict[str, dict]) -> None:
    exact = results["exact"]["p95"]
    print()
    print(f"Recall latency at {args.rows:,} postings, {args.queries} queries, "
          f"LIMIT {RECALL_LIMIT}")
    print()
    print(f"{'config':<12}{'p50':>10}{'p95':>10}{'p99':>10}{'vs exact':>12}  plan")
    print("-" * 76)
    for mode in ("exact", "hnsw", "hnsw_part"):
        r = results[mode]
        speedup = f"{exact / r['p95']:.1f}x" if r["p95"] else "-"
        plan = "Index Scan" if r["uses_index"] else "Seq Scan"
        print(
            f"{mode:<12}{r['p50']:>9.2f}m{r['p95']:>9.2f}m{r['p99']:>9.2f}m"
            f"{speedup:>12}  {plan}"
        )
    print()


if __name__ == "__main__":
    raise SystemExit(main())
