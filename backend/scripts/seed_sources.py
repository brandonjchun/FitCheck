"""Seed the crawl targets.

    python -m scripts.seed_sources

Idempotent: upserts on (kind, board_token), so re-running adjusts display
names and intervals without duplicating a board or resetting its crawl
history. A script rather than a migration for the same reason the dev user
is -- migrations run in every environment, and which companies to crawl is a
choice, not a schema fact.

**Five boards, per spec section 10.1.** The cut is deliberate: five is enough
to exercise fan-out, dedupe, closure detection, and the content-hash gate,
and small enough to re-crawl in minutes while debugging. Fifty boards teaches
nothing new and costs a week.

**The mix is chosen for cost, not variety.** Lever and Ashby return full
descriptions in the listing, so a crawl of either is exactly one HTTP request
no matter how many postings it holds. Greenhouse does not, so its postings
are fetched individually at one request per second -- which is why only one
Greenhouse board is seeded here. Every Greenhouse board shares the host
`job-boards.greenhouse.io`, so a second one does not crawl in parallel with
the first; it queues behind it.
"""

import logging

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import SessionLocal
from app.models import Source

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("seed_sources")

# Verified reachable and robots-allowed on 2026-08-03, and re-verified as
# *non-empty* -- an earlier list included `lever/plaid`, which answers 200
# with zero postings. That is worth calling out because it fails silently:
# the crawl succeeds, closure detection correctly closes nothing, and the
# board simply never contributes. Only the posting count gives it away.
#
# Sized against spec section 10.1's 300-800 target. `lever/gopuff` was
# rejected for having 807 postings on its own, which would make one company
# the entire catalog.
SOURCES = [
    # Inline content -- one request each, no per-posting fetching.
    ("lever", "spotify", "Spotify", 86400),
    ("ashby", "ashby", "Ashby", 86400),
    ("ashby", "ramp", "Ramp", 86400),
    ("ashby", "vanta", "Vanta", 86400),
    # No inline content, but publishes `updated_at`, so a re-crawl fetches
    # only what changed. The first crawl is the expensive one.
    ("greenhouse", "anthropic", "Anthropic", 86400),
]


def main() -> None:
    db = SessionLocal()
    try:
        for kind, token, name, interval in SOURCES:
            statement = (
                pg_insert(Source)
                .values(
                    kind=kind,
                    board_token=token,
                    display_name=name,
                    crawl_interval_seconds=interval,
                )
                .on_conflict_do_update(
                    index_elements=["kind", "board_token"],
                    # Deliberately does NOT touch `enabled`,
                    # `consecutive_failures`, or the crawl timestamps. Those
                    # are operational state: re-seeding after disabling a
                    # misbehaving board must not silently switch it back on.
                    set_={
                        "display_name": name,
                        "crawl_interval_seconds": interval,
                    },
                )
                .returning(Source.id)
            )
            source_id = db.execute(statement).scalar_one()
            logger.info("  %-11s %-12s -> source %s", kind, token, source_id)
        db.commit()
        total = db.query(Source).count()
        logger.info("%d sources seeded", total)
    finally:
        db.close()


if __name__ == "__main__":
    main()
