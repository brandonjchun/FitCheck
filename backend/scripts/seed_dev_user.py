"""Create the development account.

Deliberately a script rather than a migration. Migrations run everywhere,
including production, and a row with a published password is not something to
create there by accident. This has to be run on purpose.

    cd backend
    .venv\\Scripts\\Activate.ps1
    python -m scripts.seed_dev_user

Idempotent: running it again reports the existing account rather than failing
or creating a second one.
"""

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


def main() -> int:
    db = SessionLocal()
    try:
        existing = db.scalar(
            select(User).where(func.lower(User.email) == DEV_EMAIL)
        )
        if existing is not None:
            print(f"already exists: {DEV_EMAIL} (id={existing.id})")
            return 0

        user = User(email=DEV_EMAIL, password_hash=hash_password(DEV_PASSWORD))
        db.add(user)
        db.commit()
        db.refresh(user)

        print(f"created: {DEV_EMAIL} (id={user.id})")
        print(f"password: {DEV_PASSWORD}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
