"""Remove everything a load run created.

    cd backend
    python -m loadtest.cleanup

A burst of 60 users for three minutes registers a few hundred accounts and
enqueues a few thousand jobs. Left behind they are not merely untidy: the ops
dashboard's dead-letter list fills with `loadtest.invalid` rows that mask real
failures, and `users` stops being a number anyone can quote.

Deleting the accounts is enough for almost all of it -- profiles, jobs, and
batches cascade from `users`. `job_postings` does not, because the catalog is
deliberately unowned, so the postings a load run created are removed by
matching the reserved host they were submitted under. Nothing real can share
it: `.invalid` never resolves, so no genuine posting can live there.
"""

from sqlalchemy import text

from app.db import SessionLocal
from loadtest.config import EMAIL_PREFIX, UNREACHABLE


def main() -> None:
    db = SessionLocal()
    try:
        users = db.execute(
            text("SELECT COUNT(*) FROM users WHERE email LIKE :prefix"),
            {"prefix": f"{EMAIL_PREFIX}%"},
        ).scalar()

        # Cascades to profiles -> ingest_jobs -> url_batches -> matches.
        db.execute(
            text("DELETE FROM users WHERE email LIKE :prefix"),
            {"prefix": f"{EMAIL_PREFIX}%"},
        )

        postings = db.execute(
            text("DELETE FROM job_postings WHERE url LIKE :host RETURNING id"),
            {"host": f"{UNREACHABLE}%"},
        ).rowcount

        # Any ingest_jobs that were not owned by a load-test profile -- there
        # should be none, but a stray is cheaper to delete than to explain
        # later on the dead-letter list.
        jobs = db.execute(
            text("DELETE FROM ingest_jobs WHERE url LIKE :host"),
            {"host": f"{UNREACHABLE}%"},
        ).rowcount

        db.commit()
        print(f"removed {users} accounts, {postings} postings, {jobs} stray jobs")
    finally:
        db.close()


if __name__ == "__main__":
    main()
