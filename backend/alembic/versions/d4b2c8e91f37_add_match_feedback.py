"""Add match_feedback

Revision ID: d4b2c8e91f37
Revises: c1e7a92f4d58
Create Date: 2026-08-03

M10. Spec section 8.6's data collection: the labels that would eventually let
`0.4/0.6` be derived rather than asserted.

**Append-only by construction.** No unique constraint on `match_id`, so a
match accumulates a history rather than a current value. A user who marks a
posting interested and later applies has stated two true things in an order
that is itself signal; collapsing them to the latest verdict would discard
the funnel this table exists to observe.

**ON DELETE CASCADE from matches.** A label whose match is gone cannot be
reconstructed into a training row -- the features it was a reaction to
(`semantic_score`, `skill_score`, `scorer_version`) live on the match. An
orphaned verdict is not partial data, it is an uninterpretable string.

**No CHECK on verdict.** Consistent with `status` and `origin` elsewhere in
this schema: the vocabulary is expected to grow, and validation lives in the
Pydantic layer where a rejection can explain itself to a caller. A CHECK
constraint would turn adding `saved` into a migration.
"""

from alembic import op
import sqlalchemy as sa

revision = "d4b2c8e91f37"
down_revision = "c1e7a92f4d58"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "match_feedback",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("match_id", sa.BigInteger(), nullable=False),
        sa.Column("verdict", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["match_id"], ["matches.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_match_feedback_match_id", "match_feedback", ["match_id"])
    op.create_index("ix_match_feedback_created_at", "match_feedback", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_match_feedback_created_at", table_name="match_feedback")
    op.drop_index("ix_match_feedback_match_id", table_name="match_feedback")
    op.drop_table("match_feedback")
