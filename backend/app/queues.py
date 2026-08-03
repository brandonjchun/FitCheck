"""RQ queue construction -- four queues, separated by latency class.

Strip away the framework and a job queue is a list in Redis plus a
convention: enqueue serializes a function reference and its arguments and
RPUSHes them; a worker process does a blocking BLPOP on that list, pops an
item, deserializes it, and calls the function. BLPOP means idle workers burn
no CPU, and Redis's single-threaded command execution guarantees exactly one
worker receives any given item.

Everything RQ adds -- retries, timeouts, result storage, failure registries --
is bookkeeping on top of that primitive.

**Why four queues and not one.** Two producers generate backlogs in the
thousands: the nightly crawler at M8, and -- available now -- a user
uploading a .txt of job URLs. At roughly 3s per fetch with 4 workers, a
2,000-item batch is about 25 minutes of queue. On a single shared list, a
different user pasting one URL waits behind all of it for a request that
takes 3 seconds to serve. That is head-of-line blocking, and no amount of
retry tuning fixes it; only separating the lists does.

The separation is worth nothing without workers that respect it, so see
docker-compose.yml: the interactive worker drains `interactive scoring
ingest` in that order and the bulk workers drain `ingest` alone. RQ takes
queues in command-line order, so an interactive worker finishes whatever it
is holding, then checks `interactive` before it looks at bulk work again. No
capacity is wasted -- an idle interactive worker still helps drain the
backlog -- but a user submission jumps the entire queue on the next poll.
"""

from functools import lru_cache

from redis import Redis
from rq import Queue

from app.config import settings

# Latency classes, named for what produces them.
#
#   interactive  a human is watching a spinner            seconds
#   scoring      CPU-bound, no external network            minutes
#   ingest       the bulk: batch fan-out, crawler fan-out  minutes to hours
#   discovery    scheduled board enumeration               minutes
#
# `discovery` is declared and currently unused. It is here because M8's
# scheduler is the only thing that will ever produce it, and an empty list in
# Redis costs nothing -- whereas renaming queues once workers, compose
# services, and dashboards refer to them is a change across four places.
QUEUE_INTERACTIVE = "interactive"
QUEUE_SCORING = "scoring"
QUEUE_INGEST = "ingest"
QUEUE_DISCOVERY = "discovery"

QUEUE_NAMES: tuple[str, ...] = (
    QUEUE_INTERACTIVE,
    QUEUE_SCORING,
    QUEUE_INGEST,
    QUEUE_DISCOVERY,
)

# Retry schedule for transient failures. Exponential rather than fixed
# interval for two reasons, both real:
#
#   1. A 503 from an overloaded server is not resolved by asking again 100ms
#      later. Transient failures need time to clear.
#   2. Thundering herd. If a dependency drops and 500 jobs fail together,
#      fixed-interval retry means all 500 come back at the same instant the
#      moment it recovers -- knocking it straight back down.
#
# Jitter is still missing and now matters more than it did: a batch upload is
# a failure *cohort* by construction, since a hand-collected URL list is
# usually concentrated on a few job boards. It lands with the real fetch in
# M5, before any of this reaches a third-party server.
RETRY_INTERVALS: list[int] = [10, 60, 300]
MAX_RETRIES: int = 3

# Hard ceiling on one attempt. This is the only thing that stops a pathological
# document or a hung socket from occupying a worker forever -- the failure mode
# the spec calls out as the one people skip and then cannot explain why their
# queue stalled.
JOB_TIMEOUT: int = 120

# Keep finished job results in Redis for a day so the ops dashboard can show
# recent history. Failures are kept ten times longer: a succeeded job is
# uninteresting an hour later, a failed one is exactly what you come looking
# for the next morning.
RESULT_TTL: int = 24 * 60 * 60
FAILURE_TTL: int = 10 * 24 * 60 * 60


@lru_cache(maxsize=1)
def get_redis() -> Redis:
    """One Redis connection pool per process."""
    return Redis.from_url(settings.redis_url)


@lru_cache(maxsize=len(QUEUE_NAMES))
def get_queue(name: str = QUEUE_INTERACTIVE) -> Queue:
    """The named queue, constructed once per process.

    Defaults to `interactive` rather than to a general-purpose queue so that
    a caller who forgets to choose gets the *small* queue. The failure mode
    that way is a bulk job jumping ahead of user work, which is a latency
    annoyance; defaulting to `ingest` would put user-facing work behind a
    crawler backlog, which is the bug this module exists to prevent.
    """
    if name not in QUEUE_NAMES:
        raise ValueError(f"unknown queue {name!r}; expected one of {QUEUE_NAMES}")

    return Queue(name, connection=get_redis())


def queue_depths() -> dict[str, int]:
    """Pending job count per queue.

    The backpressure signal, and the number the M10 ops dashboard is built
    on. Counts waiting jobs only -- not running ones, which have already left
    the list.
    """
    return {name: get_queue(name).count for name in QUEUE_NAMES}
