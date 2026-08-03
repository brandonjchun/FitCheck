"""add url_batches

Revision ID: c40a8e17d9b2
Revises: b93f612ac05d
Create Date: 2026-08-03 04:31:44.207731

Groups the jobs produced by one uploaded URL list, so a client polls a single
aggregate endpoint rather than N individual job endpoints.

Note the absence of a `completed_count` column. Progress is derived with a
GROUP BY over jobs.batch_id. A stored counter would be incremented by N
concurrent workers -- the `counter = counter + 1` double-count that
at-least-once delivery makes inevitable -- and a summary that disagrees with
the rows it summarizes is worse than no summary at all. The index on
batch_id is what keeps deriving it cheap.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c40a8e17d9b2'
down_revision: Union[str, Sequence[str], None] = 'b93f612ac05d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'url_batches',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('user_id', sa.BigInteger(), nullable=False),
        sa.Column('profile_id', sa.BigInteger(), nullable=False),
        sa.Column('original_filename', sa.Text(), nullable=False),
        # Counts of what happened to the submitted lines. Stored rather than
        # derived because rejected and duplicate lines never became rows --
        # there is nothing to count later, and "you sent 4,000 and we took
        # 500" is precisely what the user needs to be told.
        sa.Column('total_urls', sa.Integer(), nullable=False),
        sa.Column(
            'rejected_urls', sa.Integer(), server_default='0', nullable=False
        ),
        sa.Column(
            'duplicate_urls', sa.Integer(), server_default='0', nullable=False
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['profile_id'], ['profiles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_url_batches_user_id', 'url_batches', ['user_id'])

    # NULL for a single URL submission, which is the common case and why this
    # is nullable rather than a separate table of memberships.
    op.add_column('jobs', sa.Column('batch_id', sa.BigInteger(), nullable=True))
    op.create_foreign_key(
        'fk_jobs_batch_id',
        'jobs',
        'url_batches',
        ['batch_id'],
        ['id'],
        # SET NULL, not CASCADE. A job is a work record with its own history;
        # deleting a batch should orphan the grouping rather than erase the
        # audit trail of what was actually fetched.
        ondelete='SET NULL',
    )
    op.create_index('ix_jobs_batch_id', 'jobs', ['batch_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_jobs_batch_id', table_name='jobs')
    op.drop_constraint('fk_jobs_batch_id', 'jobs', type_='foreignkey')
    op.drop_column('jobs', 'batch_id')
    op.drop_index('ix_url_batches_user_id', table_name='url_batches')
    op.drop_table('url_batches')
