"""carry board title through ingest jobs

Revision ID: a9f4c07b3e18
Revises: d4b2c8e91f37
Create Date: 2026-08-04 15:40:00.000000

One nullable column on `ingest_jobs`, closing the hole the fetch path had in
it: a crawl knows every posting's title, and had nowhere to put it.

`discover_source` fans a board out into one ingest job per posting. Postings
whose text arrives inline go straight to `_ingest_inline_posting`, which stores
the title. Postings that need a fetch go through the job row -- and the row
carried the URL and the source and nothing else, so the title was discarded at
the enqueue boundary and `_upsert_posting` inserted a posting with none. The
column was then left to the LLM, which returns null for most postings, and the
feed rendered "Untitled posting" for 401 rows.

Nullable with no backfill, and both halves are deliberate:

  nullable    a user-submitted URL has no board behind it, so NULL is the
              correct value rather than a missing one.
  no backfill the titles for existing job rows are not recoverable from this
              database -- they only ever existed in the board's API response.
              The next crawl re-enumerates and repairs the postings directly
              via the ON CONFLICT in `_upsert_posting`, which is where the
              user-visible fix actually lands. Backfilling work records that
              nothing reads afterwards would be motion without effect.

No index. Nothing queries by title on this table; it is write-once, read-once,
by primary key, in the worker that owns the row.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a9f4c07b3e18"
down_revision: Union[str, Sequence[str], None] = "d4b2c8e91f37"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("ingest_jobs", sa.Column("title", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("ingest_jobs", "title")
