"""The process that makes Path B run without a person.

Until this existed, `discover_source` had no caller: the crawler was complete,
tested, and only ever ran when somebody typed a command. That is the gap
D-060 recorded, and this closes it.

**Why a loop rather than rq-scheduler or a self-rescheduling job.**
Three options were considered, and the ranking is worth writing down because
the other two look more sophisticated:

1. *A dedicated loop* -- this file. One process, one timer, and the invariant
   "exactly one scheduler exists" is enforced by there being one container.
   Its state is entirely in the database (`last_crawled_at`), so a restart
   loses nothing and a crash is repaired by starting it again.

2. *A self-rescheduling RQ job* (`enqueue_in` at the end of each tick). Needs
   no extra process, and is genuinely elegant right up until the chain breaks
   or forks. A worker dying between "crawl" and "reschedule" ends the chain
   silently -- nothing is due, nothing errors, crawling just stops. Two
   accidental starts produce two chains that are indistinguishable from one
   running at double rate. Both failures are invisible in exactly the way the
   ops dashboard cannot show.

3. *rq-scheduler / rq's `--with-scheduler`* -- fine, and a dependency plus a
   second scheduling vocabulary to hold in your head for a job that is one
   `while True` away. The cron expression would live somewhere neither the
   database nor this repo's config can see.

The tick interval here is *not* the crawl interval. Sources carry their own
`crawl_interval_seconds`; this only decides how often we ask "is anything
due". Ticking more often than the shortest source interval costs one indexed
query and lets a newly-added board start crawling promptly, which is why the
default is minutes rather than hours.

**Nothing here talks to a job board.** A tick enqueues onto `discovery` and
returns. Politeness -- robots, rate limits, circuit breakers -- lives in the
crawl task, so a scheduler bug can at worst produce extra queue entries, and
never extra requests to somebody else's server.
"""

import logging
import os
import signal
import sys
import time
from types import FrameType

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logging_setup import configure_logging  # noqa: E402
from app.workers.tasks import tick_sources  # noqa: E402

logger = logging.getLogger("scheduler")

# How often to ask whether any source is due. See the module docstring: this
# is not the crawl interval, it is the resolution at which crawl intervals are
# honoured.
TICK_SECONDS = int(os.environ.get("SCHEDULER_TICK_SECONDS", "300"))

_stop = False


def _handle_signal(signum: int, _frame: FrameType | None) -> None:
    """Finish the current tick, then exit.

    A tick is short and idempotent, so being killed mid-tick is survivable --
    but exiting cleanly means the container stops in a second rather than
    after Docker's ten-second SIGKILL timer, which matters when you are
    restarting it repeatedly during development.
    """
    global _stop
    logger.info("scheduler: received signal %s, stopping after this tick", signum)
    _stop = True


def main() -> int:
    configure_logging()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("scheduler: started, ticking every %ss", TICK_SECONDS)

    while not _stop:
        try:
            result = tick_sources()
            logger.info("scheduler: tick -> %s", result)
        except Exception:
            # Never exit on a failed tick. The database being briefly
            # unreachable is a normal event in a compose stack that is being
            # restarted, and a scheduler that dies on it turns a ten-second
            # blip into a crawler that is off until somebody notices.
            logger.exception("scheduler: tick failed, continuing")

        # Slept in one-second slices so a SIGTERM does not wait out the whole
        # interval. A five-minute shutdown looks like a hang.
        for _ in range(TICK_SECONDS):
            if _stop:
                break
            time.sleep(1)

    logger.info("scheduler: stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
