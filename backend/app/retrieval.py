"""Stage one of the feed: narrowing the catalog to a rerankable shortlist.

Spec section 8.5. The arithmetic that forces this shape: 10,000 open postings
against 1,000 profiles is 10^7 pairs, and the §8.3 skill overlap is far too
expensive to run on all of them. Recall-then-rerank replaces that with ~200
reranks per profile by letting the vector index do the narrowing.

**The asymmetry is the whole point.** Retrieval is indexed and approximate;
reranking is exact, explainable, and pure Python over JSONB that was already
extracted. Cheap per item, so it is affordable on 200 and unaffordable on
10,000. This module owns only the first half -- it returns ids and semantic
scores and deliberately knows nothing about skills.

**What "approximate" costs here.** HNSW can miss a true nearest neighbour.
For a recommendation feed that is a fine trade, and it is worth being precise
about why: the postings at risk are those ranked near the recall cutoff by
embedding similarity, which are exactly the ones least likely to survive
reranking into the top 50 anyway. A system where the miss mattered -- payment
reconciliation, deduplication -- could not use this and would eat the exact
scan.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

# How many candidates recall hands to the reranker. Section 8.5's number.
#
# The tension: too low and a good posting never gets the chance to be scored
# on skills, too high and the rerank cost creeps back toward the full scan
# this exists to avoid. 200 is roughly 2% of the spec's 10,000-posting target
# and four times the 50 that get persisted, which leaves reranking room to
# reorder substantially rather than merely confirm the embedding's ordering.
RECALL_LIMIT = 200

# How many survive reranking and reach the feed.
FEED_LIMIT = 50


@dataclass(frozen=True)
class Candidate:
    """One posting that survived recall, with its semantic half already known.

    The score is carried out of SQL rather than recomputed in Python from the
    two vectors. Both routes give the same number -- `embed_text` returns unit
    vectors, so cosine distance is exactly `1 - similarity` -- and doing it in
    SQL means the reranker never has to load 200 embeddings into the worker
    just to reproduce a figure Postgres already calculated while sorting.
    """

    posting_id: int
    semantic_score: float


@dataclass(frozen=True)
class FeedFilters:
    """The hard predicates a user can put on their feed.

    Separate from the ranking because these are not preferences to be weighed,
    they are constraints: a candidate who cannot relocate is not served by a
    great on-site match ranked third.
    """

    remote_only: bool = False
    seniority: tuple[str, ...] = ()
    max_min_years: float | None = None

    def is_empty(self) -> bool:
        return not self.remote_only and not self.seniority and self.max_min_years is None


def _vector_literal(embedding: list[float]) -> str:
    """Render a Python list in pgvector's text input format."""
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _where_clauses(filters: FeedFilters) -> tuple[list[str], dict]:
    # `closed_at IS NULL` is first and unconditional, and it is doing two jobs.
    # It is the user-visible rule that a filled role leaves the feed, and it is
    # the predicate the HNSW index is partial on -- so it has to appear
    # verbatim in the query or Postgres cannot match the query to the index.
    #
    # `extracted IS NOT NULL` is not tidiness, it is a correctness guard, and
    # the reasoning is worth keeping because the bug it prevents looks like a
    # feature from the outside. `score_skills` scores an empty requirement
    # list as 1.0 -- correct in isolation, since a posting that asks for
    # nothing is a posting nobody fails to satisfy. But an *unextracted*
    # posting also presents an empty requirement list, via `JobPosting.skills`
    # returning [] when `extracted` is falsy. Those two states are
    # indistinguishable downstream, so an unprepared posting would score a
    # perfect 1.0 on the skill half and blend to at least 0.6 -- ranking it
    # above every posting that was genuinely assessed. A crawl that outruns
    # its extraction backlog would therefore fill the top of every feed with
    # exactly the postings nothing is known about.
    #
    # Filtering here rather than demoting later because a posting with no
    # extraction has no skill evidence at all: there is no honest score to
    # give it, only a rank to withhold until there is.
    #
    # The predicate is `extraction_version IS NOT NULL`, and arriving at that
    # took two corrections worth recording, because both of the more obvious
    # spellings are wrong in ways that do not raise.
    #
    # `extracted IS NOT NULL` is wrong on *correctness*. SQLAlchemy persists a
    # Python `None` into a JSONB column as the JSON value `null`, not as SQL
    # NULL, so a row written with `extracted=None` satisfies `IS NOT NULL`
    # while `JobPosting.skills` still returns [] -- exactly the combination
    # this filter exists to exclude.
    #
    # `jsonb_typeof(extracted) = 'object'` fixes that and is wrong on
    # *performance*, which is worse because it is invisible. Postgres has no
    # statistics for an opaque function call, estimates ~46 matching rows out
    # of 10,000, and on that estimate concludes a sequential scan is cheaper
    # than the HNSW index. The query then returns identical results, slightly
    # slower, with the index sitting there unused -- measured at 10k rows in
    # `scripts/bench_recall.py`, which is the reason that script asserts on
    # the plan rather than trusting a timing.
    #
    # `extraction_version` is a plain integer column, so the planner has real
    # statistics and uses the index. It is also the *honest* spelling of the
    # question: models.py writes the version in the same commit as the blob
    # specifically so the two cannot disagree, which makes "has a version" the
    # authoritative answer to "has an extraction" -- and it is NULL for the
    # JSON-null case too, so the correctness fix survives.
    clauses = [
        "closed_at IS NULL",
        "embedding IS NOT NULL",
        "extraction_version IS NOT NULL",
    ]
    params: dict = {}

    if filters.remote_only:
        clauses.append("remote_type = 'remote'")

    if filters.seniority:
        clauses.append("seniority = ANY(:seniority)")
        params["seniority"] = list(filters.seniority)

    if filters.max_min_years is not None:
        # NULL means the posting never stated a requirement, which is not the
        # same as stating zero. Kept rather than excluded: an unstated
        # requirement is no evidence of a barrier, and dropping those rows
        # would silently hide every posting whose extraction was thin.
        clauses.append("(min_years IS NULL OR min_years <= :max_min_years)")
        params["max_min_years"] = filters.max_min_years

    return clauses, params


def recall_candidates(
    db: Session,
    embedding: list[float],
    *,
    filters: FeedFilters | None = None,
    limit: int = RECALL_LIMIT,
) -> list[Candidate]:
    """Return the `limit` postings nearest this profile, cheapest-first.

    `<=>` is cosine distance, so it ascends where similarity descends, and the
    operator has to be this one: the index is built with `vector_cosine_ops`
    and Postgres will not use it for `<->` or `<#>`. That mismatch does not
    raise -- it falls back to a sequential scan and returns identical rows
    slightly slower, which is precisely the kind of regression that survives
    a test suite. `scripts/bench_recall.py` asserts on the query plan for
    that reason.
    """
    filters = filters or FeedFilters()
    clauses, params = _where_clauses(filters)

    statement = text(
        f"""
        SELECT id, 1 - (embedding <=> CAST(:vec AS vector)) AS semantic_score
          FROM job_postings
         WHERE {' AND '.join(clauses)}
         ORDER BY embedding <=> CAST(:vec AS vector)
         LIMIT :limit
        """
    )

    rows = db.execute(
        statement,
        {**params, "vec": _vector_literal(embedding), "limit": limit},
    ).all()

    return [Candidate(posting_id=row[0], semantic_score=float(row[1])) for row in rows]


def explain_recall(
    db: Session,
    embedding: list[float],
    *,
    filters: FeedFilters | None = None,
    limit: int = RECALL_LIMIT,
) -> str:
    """The query plan for `recall_candidates`, as EXPLAIN ANALYZE text.

    Exists so the benchmark can prove the index is being *used* rather than
    infer it from a timing that could equally reflect a warm cache.
    """
    filters = filters or FeedFilters()
    clauses, params = _where_clauses(filters)

    statement = text(
        f"""
        EXPLAIN ANALYZE
        SELECT id, 1 - (embedding <=> CAST(:vec AS vector)) AS semantic_score
          FROM job_postings
         WHERE {' AND '.join(clauses)}
         ORDER BY embedding <=> CAST(:vec AS vector)
         LIMIT :limit
        """
    )

    rows = db.execute(
        statement,
        {**params, "vec": _vector_literal(embedding), "limit": limit},
    ).all()
    return "\n".join(row[0] for row in rows)
