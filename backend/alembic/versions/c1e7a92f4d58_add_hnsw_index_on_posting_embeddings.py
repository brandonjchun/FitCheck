"""Add the HNSW index over posting embeddings

Revision ID: c1e7a92f4d58
Revises: b8d3f1a5c724
Create Date: 2026-08-03

M9. Recall goes from an exact scan over every open posting to an indexed
nearest-neighbour lookup. Spec section 3.5 chose HNSW over IVFFlat, and
section 8.5 chose the partial predicate; both decisions land here.

**Why the index is partial on `closed_at IS NULL`.** This is the mitigation
section 8.5 ranks first, and it is worth being precise about what it fixes.
An ANN index retrieves neighbours *first* and applies `WHERE` afterwards, so
a query asking for 200 open postings against an index containing closed ones
gets 200 neighbours, then filters, then returns however many survive --
possibly far fewer than 200, with no error and no way to tell from the result
that it happened. Baking the predicate into the index means closed postings
are never candidates in the first place, so the limit means what it says.

It is also the cheaper index: a catalog that has been crawled for a semester
accumulates tombstones indefinitely, and none of them belong in a feed.

**Why `vector_cosine_ops`.** `embeddings.embed_text` returns unit vectors, so
cosine and inner product rank identically -- but the operator class has to
match the operator the query uses (`<=>`), or Postgres silently ignores the
index and falls back to a sequential scan. That failure is invisible in
results and only shows up in `EXPLAIN`, which is why the benchmark script
added alongside this migration asserts on the plan and not just the timing.

**m and ef_construction are left at their defaults** (16 and 64). The spec's
catalog target is a few hundred to a few thousand postings; the defaults are
tuned for far larger, and raising them trades build time and memory for
recall this corpus is too small to need. Recorded as a decision rather than
an oversight -- if recall is measured as insufficient later, `ef_search` is
the runtime knob to reach for first, since it needs no rebuild.

**No index on `profiles.embedding`.** Deliberate. That would be for the
incremental fan-out of section 6.9 option 2 -- scoring one new posting
against every profile -- which this project explicitly does not implement.
Path B here is lazy-per-profile, so profiles are looked up by id and never by
similarity.
"""

from alembic import op

revision = "c1e7a92f4d58"
down_revision = "b8d3f1a5c724"
branch_labels = None
depends_on = None

INDEX_NAME = "job_postings_embedding_hnsw"


def upgrade() -> None:
    # Not CONCURRENTLY: Alembic runs migrations inside a transaction, and
    # CREATE INDEX CONCURRENTLY cannot run in one. On a catalog this size the
    # build is seconds and the write lock is irrelevant; on a production-sized
    # table this would need to be its own out-of-band step.
    op.execute(
        f"""
        CREATE INDEX {INDEX_NAME}
            ON job_postings
         USING hnsw (embedding vector_cosine_ops)
         WHERE closed_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
