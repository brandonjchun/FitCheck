"""Set `job_postings.company` from the board each posting came from.

    python -m scripts.backfill_company --dry-run
    python -m scripts.backfill_company

Repairs the catalog in place. Measured before this ran: 19,691 board-sourced
postings held 28 companies between them, and most of those 28 were the *ATS
vendor* rather than the employer -- "Ashby" 22 times, "Greenhouse" once, and one
row that came back as `>{`. The employer name is not reliably in a posting body,
so the extraction was being asked a question the text does not answer, while
`sources.display_name` had the answer already, seeded by hand.

**Why a script and not a migration.** The same reason `backfill_seniority` is a
script: this is a data repair, not a schema change. Nothing about the column
moves. The values in it were wrong because the only writer was an LLM shown the
posting body and never told which board it came from.

**Overwrites rather than filling gaps, and only for board-sourced rows.** The 28
existing values are the ones this exists to correct, so skipping non-null
columns would leave every "Ashby" in place -- the bug would survive its own fix.
The board is the authority for any row with a `source_id`; for a row without one
there is no board to ask, so whatever the extraction produced is the best
answer available and is left alone.

**Idempotent.** Only rows whose stored company differs from their board's
display name are touched, so a second run reports nothing to do.
"""

import argparse
from collections import Counter

from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import JobPosting, Source


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
        # Joined rather than looked up per row. 19,691 postings against 161
        # boards is 19,691 round trips the other way, for a mapping that fits
        # in memory.
        rows = db.execute(
            select(JobPosting, Source.display_name)
            .join(Source, JobPosting.source_id == Source.id)
        ).all()

        for posting, display_name in rows:
            if posting.company == display_name:
                continue

            changes[f"{posting.company!s} -> {display_name}"] += 1
            if len(examples) < 15:
                examples.append(
                    f"  {str(posting.company):>12} -> {display_name:<24} {posting.title}"
                )
            if not args.dry_run:
                posting.company = display_name

        if not args.dry_run:
            db.commit()

        # Reported rather than assumed, because it is the number that says
        # whether "no more nulls" is actually true -- and it can only be answered
        # after the writes above. Split on `source_id`, because the two cases
        # mean different things: a board-sourced row still null after this is a
        # bug in the join, while a sourceless one has no board to inherit from
        # and is the documented limit of what this can reach.
        remaining = dict(
            db.execute(
                select(
                    JobPosting.source_id.is_not(None).label("from_board"),
                    func.count(),
                )
                .where(JobPosting.company.is_(None))
                .group_by(JobPosting.source_id.is_not(None))
            ).all()
        )
    finally:
        db.close()

    total = sum(changes.values())
    print(f"{len(rows)} board-sourced postings examined, {total} to correct\n")
    for transition, count in changes.most_common(20):
        print(f"  {count:>5}  {transition}")
    if examples:
        print("\nexamples:")
        print("\n".join(examples))
    print(
        "\nnull companies remaining (unchanged -- dry run):"
        if args.dry_run
        else "\nnull companies remaining:"
    )
    print(
        f"  {remaining.get(True, 0):>5}  board-sourced"
        f"{'' if args.dry_run else '   (should be 0)'}"
    )
    print(f"  {remaining.get(False, 0):>5}  no source       (no board to ask)")
    print("\ndry run -- nothing written" if args.dry_run else "\nwritten")


if __name__ == "__main__":
    main()
