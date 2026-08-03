"""RQ queue construction.

Strip away the framework and a job queue is a list in Redis plus a
convention: enqueue serializes a function reference and its arguments and
RPUSHes them; a worker process does a blocking BLPOP on that list, pops an
item, deserializes it, and calls the function. BLPOP means idle workers burn
no CPU, and Redis's single-threaded command execution guarantees exactly one
worker receives any given item.

Everything RQ adds -- retries, timeouts, result storage, failure registries --
is bookkeeping on top of that primitive.
"""

from functools import lru_cache

from redis import Redis
from rq import Queue

from app.config import settings

# Retry schedule for transient failures. Exponential rather than fixed
# interval for two reasons, both real:
#
#   1. A 503 from an overloaded server is not resolved by asking again 100ms
#      later. Transient failures need time to clear.
#   2. Thundering herd. If a dependency drops and 500 jobs fail together,
#      fixed-interval retry means all 500 come back at the same instant the
#      moment it recovers -- knocking it straight back down. Spreading the
#      retries prevents a self-inflicted outage.
#
# Production would add jitter (randomize each interval by ~20%) so that
# retries from different jobs do not re-synchronize into the same herd. Not
# implemented here; noted as a stretch goal in the spec.
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


@lru_cache(maxsize=1)
def get_queue() -> Queue:
    """The default work queue.

    A single queue for now. Priority queues (a `high` queue drained before
    `default`) are a stretch goal -- worth it only once there is work that
    genuinely differs in urgency.
    """
    return Queue("default", connection=get_redis())
