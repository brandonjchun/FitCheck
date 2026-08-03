"""promote job posting filter columns

Revision ID: f2c9a30be584
Revises: e71d2f4a86c3
Create Date: 2026-08-03 05:58:41.229077

Adds the five promoted columns from spec section 5.2 whose shape is already
settled: location, remote_type, seniority, min_years, closed_at.

All null until M7's extraction fills them. They land now because the two
operations cost very different things:

    ADD COLUMN (nullable)   one catalog write, instant at any table size
    ALTER COLUMN TYPE       rewrites every row, rebuilds every index on it

So the question is never "add now or later" -- both are free -- it is
"guess the shape now and maybe pay to change it, or wait until it is known".
For these five there is nothing to guess, so adding them here means M7 ships
scoring rather than scoring plus a migration.

Two siblings from the same table in the spec are deliberately NOT here:

  source_id  a foreign key to `sources`, which does not exist until M8.
             Postgres refuses a reference to a missing table, so this is not
             a judgment call.
  embedding  needs pgvector enabled and a fixed dimension, and the dimension
             follows an embedding provider that is still undecided. Local
             MiniLM is 384; the hosted alternatives are not. Guessing here
             and choosing otherwise at M7 means ALTER COLUMN TYPE on the
             widest column in the biggest table -- the one case where the
             expensive operation actually hurts.

No indexes yet. An index over a column that is null in every row costs write
overhead and returns nothing, and creating one later is as cheap as this is.
They land with M9, which is the first thing that filters on them.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f2c9a30be584'
down_revision: Union[str, Sequence[str], None] = 'e71d2f4a86c3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('job_postings', sa.Column('location', sa.Text(), nullable=True))

    # Free text rather than a Postgres enum. The vocabulary comes from an LLM,
    # and an enum would mean ALTER TYPE the first time a posting says
    # "flexible" instead of "hybrid" -- which is a migration to gain nothing
    # a CHECK constraint or the application layer could not do more cheaply.
    op.add_column('job_postings', sa.Column('remote_type', sa.Text(), nullable=True))
    op.add_column('job_postings', sa.Column('seniority', sa.Text(), nullable=True))

    # Mirrors profiles.years_experience, so comparing what a posting demands
    # against what a candidate has needs no cast.
    op.add_column(
        'job_postings',
        sa.Column('min_years', sa.Numeric(precision=4, scale=1), nullable=True),
    )

    # Tombstone, never a delete: `matches` will reference postings, so removing
    # a closed one either cascades away a user's history or raises a foreign
    # key error mid-crawl.
    op.add_column(
        'job_postings',
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('job_postings', 'closed_at')
    op.drop_column('job_postings', 'min_years')
    op.drop_column('job_postings', 'seniority')
    op.drop_column('job_postings', 'remote_type')
    op.drop_column('job_postings', 'location')
