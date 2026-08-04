"""Re-derive `job_postings.seniority` from titles already in the database.

    python -m scripts.backfill_seniority --dry-run
    python -m scripts.backfill_seniority

Repairs the catalog in place, so the feed's seniority filter is right now
rather than after the next crawl has re-extracted 1,400 postings through a
local LLM at ~11s each.

**Why a script and not a migration.** A migration would have to carry a copy of
the level patterns in SQL, and then there would be two definitions of "what
counts as a staff title" drifting apart -- the one `app.seniority` is tested
against, and a frozen transcription of it in a revision file. This imports the
real function, so there is exactly one.

It is also not a schema change. Nothing about the column moves; the values in
it were wrong because the extraction that wrote them was shown the posting body
and never the title. Re-running the fix over stored titles is a data repair,
and repeatable at that.

**Idempotent, and narrow on purpose.** Only rows whose title states a level are
touched, and only when the derived answer differs from what is stored. A
posting whose title says nothing about level is left exactly as the extraction
left it -- `seniority_from_title` returns None there, and None means defer.
Running this twice changes nothing the second time.
"""

import argparse
from collections import Counter

from sqlalchemy import select

from app.db import SessionLocal
from app.models import JobPosting
from app.seniority import seniority_from_title


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change and write nothing",
    )
    args = parser.parse_args()

    changes: Counter = Counter()
    examples: list[str] = []

    db = SessionLocal()
    try:
        rows = db.execute(
            select(JobPosting).where(JobPosting.title.is_not(None))
        ).scalars().all()

        for row in rows:
            derived = seniority_from_title(row.title)
            if derived is None or derived == row.seniority:
                continue

            changes[f"{row.seniority} -> {derived}"] += 1
            if len(examples) < 15:
                examples.append(f"  {row.seniority!s:>8} -> {derived:<8} {row.title}")
            if not args.dry_run:
                row.seniority = derived

        if not args.dry_run:
            db.commit()
    finally:
        db.close()

    total = sum(changes.values())
    print(f"{len(rows)} titled postings examined, {total} to correct\n")
    for transition, count in changes.most_common():
        print(f"  {count:>5}  {transition}")
    if examples:
        print("\nexamples:")
        print("\n".join(examples))
    print(
        "\ndry run -- nothing written" if args.dry_run else "\nwritten"
    )


if __name__ == "__main__":
    main()
