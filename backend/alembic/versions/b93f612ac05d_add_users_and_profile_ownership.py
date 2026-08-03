"""add users and profile ownership

Revision ID: b93f612ac05d
Revises: a17c4e9b2d38
Create Date: 2026-08-03 04:02:17.553918

Introduces accounts and makes every profile belong to one.

**This migration deletes existing profiles.** They predate ownership and have
no owner to assign, and `profiles.user_id` is NOT NULL by design -- an
ownerless profile is invisible to every query in the application, so allowing
the column to be null would preserve rows that nothing can ever read while
forcing every ownership predicate to handle a case that should not exist.
Deletion cascades to `jobs`.

That is a deliberate, destructive choice appropriate to a development
database with test uploads in it. Against real data the same migration would
instead assign existing rows to a real owner before the NOT NULL, and the
delete below would be wrong.

No user is seeded here. Migrations run in every environment, and a row with a
known development password is not something to create in production by
accident -- see scripts/seed_dev_user.py.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b93f612ac05d'
down_revision: Union[str, Sequence[str], None] = 'a17c4e9b2d38'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # citext makes email comparison case-insensitive in the database, so
    # Bob@x.com and bob@x.com cannot become two accounts. Doing it in Python
    # -- lowercasing before each insert and lookup -- holds only until one
    # code path forgets, and the failure is a duplicate account nobody can
    # explain. Shipped with Postgres as a standard extension.
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")

    op.create_table(
        'users',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('email', sa.Text().with_variant(sa.Text(), 'postgresql'), nullable=False),
        sa.Column('password_hash', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    # Applied after create_table because SQLAlchemy has no first-class citext
    # type. The unique index is created afterwards so it is built on the
    # citext column and therefore enforces case-insensitive uniqueness -- the
    # whole point of the extension.
    op.execute("ALTER TABLE users ALTER COLUMN email TYPE citext")
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # Existing profiles predate ownership. See the module docstring.
    op.execute("DELETE FROM profiles")

    op.add_column(
        'profiles',
        sa.Column('user_id', sa.BigInteger(), nullable=False),
    )
    op.add_column(
        'profiles',
        sa.Column(
            'is_active', sa.Boolean(), server_default=sa.text('false'), nullable=False
        ),
    )
    op.create_foreign_key(
        'fk_profiles_user_id',
        'profiles',
        'users',
        ['user_id'],
        ['id'],
        ondelete='CASCADE',
    )
    # Postgres does not index foreign keys automatically, and "this user's
    # profiles" is on every authenticated request path.
    op.create_index('ix_profiles_user_id', 'profiles', ['user_id'])

    # Exactly one active resume per user. Partial, so the constraint binds
    # only on active rows and a user may keep any number of inactive ones.
    # Enforced here rather than in application code because two concurrent
    # "make this active" requests both read "none active" and both write --
    # only the database can reject the second.
    op.create_index(
        'profiles_one_active_per_user',
        'profiles',
        ['user_id'],
        unique=True,
        postgresql_where=sa.text('is_active'),
    )


def downgrade() -> None:
    """Downgrade schema.

    Does not restore the deleted profiles -- nothing recorded them. Dropping
    the extension is also deliberately omitted: other tables may come to use
    citext, and DROP EXTENSION would take them with it.
    """
    op.drop_index('profiles_one_active_per_user', table_name='profiles')
    op.drop_index('ix_profiles_user_id', table_name='profiles')
    op.drop_constraint('fk_profiles_user_id', 'profiles', type_='foreignkey')
    op.drop_column('profiles', 'is_active')
    op.drop_column('profiles', 'user_id')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
