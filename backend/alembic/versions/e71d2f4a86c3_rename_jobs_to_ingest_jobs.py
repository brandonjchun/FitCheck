"""rename jobs to ingest_jobs

Revision ID: e71d2f4a86c3
Revises: d5b17c9e3f04
Create Date: 2026-08-03 05:44:03.612884

Spec section 5.1. With `job_postings` now present, "job" names two different
things -- the unit of async work and the thing a person applies to -- and
every sentence about the system needs a disambiguating clause. Renamed now
rather than after M8, because the crawler and `matches` both add call sites
and the cost of this is proportional to how many exist.

`ALTER TABLE ... RENAME` moves the table and nothing else: indexes, the
unique constraint, the primary key, and the identity sequence all keep their
original names. Left alone they would be a permanent trail of the old name in
`\\d ingest_jobs`, so each is renamed explicitly.

Foreign keys pointing *at* this table (from nothing yet) and *from* it (to
profiles, url_batches, job_postings) follow the rename automatically -- they
reference the table by oid, not by name.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'e71d2f4a86c3'
down_revision: Union[str, Sequence[str], None] = 'd5b17c9e3f04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# (old, new) for everything that carries the table's name.
_INDEXES = [
    ("ix_jobs_batch_id", "ix_ingest_jobs_batch_id"),
    ("ix_jobs_created_at", "ix_ingest_jobs_created_at"),
    ("ix_jobs_job_posting_id", "ix_ingest_jobs_job_posting_id"),
    ("ix_jobs_profile_id", "ix_ingest_jobs_profile_id"),
    ("ix_jobs_status", "ix_ingest_jobs_status"),
]

_CONSTRAINTS = [
    ("jobs_pkey", "ingest_jobs_pkey"),
    ("jobs_profile_url_uniq", "ingest_jobs_profile_url_uniq"),
    ("jobs_profile_id_fkey", "ingest_jobs_profile_id_fkey"),
    ("fk_jobs_batch_id", "fk_ingest_jobs_batch_id"),
    ("fk_jobs_job_posting_id", "fk_ingest_jobs_job_posting_id"),
]


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table("jobs", "ingest_jobs")

    for old, new in _INDEXES:
        op.execute(f"ALTER INDEX {old} RENAME TO {new}")

    for old, new in _CONSTRAINTS:
        op.execute(f"ALTER TABLE ingest_jobs RENAME CONSTRAINT {old} TO {new}")

    # bigserial creates an owned sequence named after the original table. It
    # keeps working under the old name, but leaving it is how a schema ends up
    # with an archaeological layer nobody can explain in week 12.
    op.execute("ALTER SEQUENCE jobs_id_seq RENAME TO ingest_jobs_id_seq")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("ALTER SEQUENCE ingest_jobs_id_seq RENAME TO jobs_id_seq")

    for old, new in _CONSTRAINTS:
        op.execute(f"ALTER TABLE ingest_jobs RENAME CONSTRAINT {new} TO {old}")

    for old, new in _INDEXES:
        op.execute(f"ALTER INDEX {new} RENAME TO {old}")

    op.rename_table("ingest_jobs", "jobs")
