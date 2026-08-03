"""Create the development account.

Deliberately a script rather than a migration. Migrations run everywhere,
including production, and a row with a published password is not something to
create there by accident. This has to be run on purpose.

    cd backend
    .venv\\Scripts\\Activate.ps1
    python -m scripts.seed_dev_user
    python -m scripts.seed_dev_user --admin    # also grants operator access

Idempotent: running it again reports the existing account rather than failing
or creating a second one. `--admin` on an existing account promotes it in
place, so the flag can be added after the fact without deleting anything.

**Why the flag lives here and not in the migration.** `users.is_admin` defaults
to false and nothing grants it, so a fresh database has zero operators and
every `/api/ops/*` route answers 403 -- including for the person who just set
the project up. The fix is not to seed an operator from a migration: migrations
run everywhere, so that would manufacture a privileged account in production
too. This script already has to be run deliberately against a local database,
which makes it the right place. Production promotion stays a SQL `UPDATE`.
"""

import argparse
import sys

from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import User
from app.security import hash_password

# Not .local, and not .test, .localhost, .invalid or .example either. Those
# are IANA special-use domains, and email-validator -- which backs EmailStr --
# rejects them. Seeding an address the login endpoint would refuse produces an
# account that exists in the database and cannot be used, which is a
# confusing thing to debug. .dev is an ordinary gTLD and validates.
DEV_EMAIL = "dev@fitcheck.dev"

# Not a secret, by design -- it is in version control and only ever reaches a
# local database. The 12-character minimum matches what RegisterRequest
# enforces, so the seeded account is one the real registration path would also
# have accepted.
DEV_PASSWORD = "devpassword123"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create (or promote) the local development account."
    )
    parser.add_argument(
        "--admin",
        action="store_true",
        help="grant operator access, required by every /api/ops/* route",
    )
    args = parser.parse_args(argv)

    db = SessionLocal()
    try:
        existing = db.scalar(
            select(User).where(func.lower(User.email) == DEV_EMAIL)
        )
        if existing is not None:
            # Promotion is applied to an account that already exists rather
            # than being skipped along with the creation. Someone who seeded
            # before the flag existed would otherwise have no way to use it
            # short of deleting the row.
            if args.admin and not existing.is_admin:
                existing.is_admin = True
                db.commit()
                print(f"promoted to admin: {DEV_EMAIL} (id={existing.id})")
                return 0

            state = "admin" if existing.is_admin else "not an admin"
            print(f"already exists: {DEV_EMAIL} (id={existing.id}, {state})")
            return 0

        user = User(
            email=DEV_EMAIL,
            password_hash=hash_password(DEV_PASSWORD),
            is_admin=args.admin,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        print(f"created: {DEV_EMAIL} (id={user.id}, admin={user.is_admin})")
        print(f"password: {DEV_PASSWORD}")
        if not args.admin:
            print("note: not an operator -- re-run with --admin to reach /api/ops/*")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
