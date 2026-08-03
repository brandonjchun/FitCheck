"""add job_postings

Revision ID: d5b17c9e3f04
Revises: c40a8e17d9b2
Create Date: 2026-08-03 05:02:19.884412

The result record the fetch produces, separate from `jobs` which is the work
record (spec section 5.6). A job exists the instant a URL is submitted and
survives every attempt failing; a posting exists only on success and outlives
any individual attempt -- at M8 one posting is re-crawled dozens of times,
each crawl a new job row against this same posting.

canonical_key is globally unique, not per-profile. Two users submitting the
same Greenhouse posting must land on one row or the crawler will create a
third. This is the constraint that makes the catalog a catalog; the existing
per-profile UNIQUE (profile_id, url_hash) on `jobs` stays where it is, since
it dedupes *work* rather than *postings* and those are different questions.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd5b17c9e3f04'
down_revision: Union[str, Sequence[str], None] = 'c40a8e17d9b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'job_postings',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('canonical_key', sa.Text(), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        # SHA-256 of the normalized text. The M8 gate compares this to decide
        # whether a re-crawl pays for extraction and embedding again, which is
        # what keeps a daily crawl cheap. Recorded from M5 so the gate has
        # history to compare against the first time it runs.
        sa.Column('content_hash', sa.Text(), nullable=False),
        sa.Column('raw_text', sa.Text(), nullable=False),
        # Populated at M7. Present now so the fetch has somewhere to write and
        # the M7 diff is scoring rather than a schema change.
        sa.Column('extracted', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('extraction_version', sa.Integer(), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('company', sa.Text(), nullable=True),
        sa.Column(
            'first_seen_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        # The heartbeat closure detection reads at M8. Updated on every
        # successful re-fetch, including one that skipped extraction.
        sa.Column(
            'last_seen_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('canonical_key', name='job_postings_canonical_uniq'),
    )

    # Set on success, null while queued, and null forever for a job that never
    # succeeded -- the work-record/result-record split expressed as a column.
    op.add_column('jobs', sa.Column('job_posting_id', sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        'fk_jobs_job_posting_id',
        'jobs',
        'job_postings',
        ['job_posting_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.create_index('ix_jobs_job_posting_id', 'jobs', ['job_posting_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_jobs_job_posting_id', table_name='jobs')
    op.drop_constraint('fk_jobs_job_posting_id', 'jobs', type_='foreignkey')
    op.drop_column('jobs', 'job_posting_id')
    op.drop_table('job_postings')
