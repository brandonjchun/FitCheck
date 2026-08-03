"""Add sources and generalize ingest_jobs for crawler work

Revision ID: b8d3f1a5c724
Revises: a4f8c2e17b90
Create Date: 2026-08-03

M8. `ingest_jobs` was shaped by the assumption that every job came from a
person clicking submit. Path B breaks that: a crawler builds the catalog
ahead of anyone asking for it, so a discovered posting belongs to no
profile and is scored later by a different job.

Three changes, in an order that matters.

**`profile_id` becomes nullable.** Straightforward, and the reason the rest
is not: Postgres treats NULLs as *distinct* in a unique index, so the old
`UNIQUE (profile_id, url_hash)` would have gone on silently permitting
unlimited duplicate crawler rows -- present, named, and constraining
nothing. That failure mode is duplicate outbound fetches against somebody
else's server, with the index still sitting there looking like it worked.

**The constraint becomes partial, on in-flight work only.** A total
constraint was right while jobs came from people and is wrong with a
crawler: a daily re-crawl submits the same URLs every day, so a total
UNIQUE rejects every tick after the first. Restricting to `queued` and
`running` means two overlapping ticks still collapse to one job, while
yesterday's completed row stays as the audit log without blocking today.

**`dedupe_key` is backfilled before the index is built, not after.**
Adding the column with a `""` default and then indexing would put every
existing in-flight row on the same key and fail on the second one. The
backfill runs first so the index is created against data that already
satisfies it.
"""

from alembic import op
import sqlalchemy as sa

revision = "b8d3f1a5c724"
down_revision = "a4f8c2e17b90"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sources",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("board_token", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "crawl_interval_seconds",
            sa.Integer(),
            nullable=False,
            server_default="86400",
        ),
        sa.Column("last_crawled_at", sa.DateTime(timezone=True), nullable=True),
        # Deliberately separate from last_crawled_at. "Attempted but did not
        # finish" has to be representable, or closure detection cannot tell a
        # complete enumeration from a truncated one -- and running the closure
        # UPDATE on a truncated one tombstones an entire board out of every
        # user's feed with no error anywhere.
        sa.Column("last_success_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "consecutive_failures", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("kind", "board_token", name="sources_kind_token_uniq"),
    )

    # The scheduler's query: which sources are due. Partial on `enabled` so a
    # disabled board is never visited by the index at all rather than being
    # retrieved and filtered.
    op.execute(
        "CREATE INDEX ix_sources_due ON sources (last_crawled_at) WHERE enabled"
    )

    # NULL for a user-submitted one-off URL, which belongs to no board.
    op.add_column(
        "job_postings",
        sa.Column(
            "source_id",
            sa.BigInteger(),
            sa.ForeignKey("sources.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    # The board's own statement of when a posting last changed, for boards
    # that offer one. This is what lets a re-crawl skip a posting *without
    # fetching it*: the content hash cannot, because computing it requires
    # already having the content. Greenhouse populates it; Lever and Ashby
    # do not, and hand back the full description instead -- which makes the
    # hash free for them and this column irrelevant.
    op.add_column(
        "job_postings",
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Closure detection scans by source. Without this it is a sequential scan
    # over the whole catalog on every crawl tick.
    op.create_index("ix_job_postings_source_id", "job_postings", ["source_id"])

    op.add_column(
        "ingest_jobs",
        sa.Column(
            "kind", sa.Text(), nullable=False, server_default="ingest_posting"
        ),
    )
    op.add_column(
        "ingest_jobs", sa.Column("dedupe_key", sa.Text(), nullable=False, server_default="")
    )
    op.add_column(
        "ingest_jobs",
        sa.Column(
            "source_id",
            sa.BigInteger(),
            sa.ForeignKey("sources.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_ingest_jobs_source_id", "ingest_jobs", ["source_id"])

    # Backfill BEFORE the unique index exists. Every existing row was a user
    # submission, so its key is the pair the old constraint enforced -- which
    # means in-flight rows that were unique under the old rule stay unique
    # under the new one, and nothing that used to be legal becomes illegal.
    op.execute(
        """
        UPDATE ingest_jobs
           SET dedupe_key = 'profile:' || profile_id || ':' || url_hash
         WHERE dedupe_key = ''
        """
    )

    # The default existed only so the ADD COLUMN could be NOT NULL against
    # existing rows. Dropping it now means a future insert that omits the key
    # fails at the database instead of quietly taking "" and colliding with
    # every other insert that omitted it.
    op.alter_column("ingest_jobs", "dedupe_key", server_default=None)

    op.drop_constraint(
        "ingest_jobs_profile_url_uniq", "ingest_jobs", type_="unique"
    )
    op.alter_column("ingest_jobs", "profile_id", nullable=True)

    op.execute(
        """
        CREATE UNIQUE INDEX ingest_jobs_inflight_uniq
            ON ingest_jobs (kind, dedupe_key)
         WHERE status IN ('queued', 'running')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ingest_jobs_inflight_uniq")

    # Crawler rows have no profile and cannot satisfy a NOT NULL profile_id or
    # the constraint above. Removing them is correct rather than destructive:
    # they are work records for a feature being rolled back, and the postings
    # they produced survive in job_postings, which is the part with value.
    op.execute("DELETE FROM ingest_jobs WHERE profile_id IS NULL")

    op.alter_column("ingest_jobs", "profile_id", nullable=False)
    op.create_unique_constraint(
        "ingest_jobs_profile_url_uniq", "ingest_jobs", ["profile_id", "url_hash"]
    )

    op.drop_index("ix_ingest_jobs_source_id", table_name="ingest_jobs")
    op.drop_column("ingest_jobs", "source_id")
    op.drop_column("ingest_jobs", "dedupe_key")
    op.drop_column("ingest_jobs", "kind")

    op.drop_index("ix_job_postings_source_id", table_name="job_postings")
    op.drop_column("job_postings", "source_updated_at")
    op.drop_column("job_postings", "source_id")

    op.execute("DROP INDEX IF EXISTS ix_sources_due")
    op.drop_table("sources")
