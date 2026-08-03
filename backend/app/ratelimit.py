"""Per-domain rate limiting, shared across every worker process.

The reason this lives in Redis rather than in each process: four ingest
workers each politely limiting themselves to 1 request per second produce
4 rps at the target host. In-process limiting does not limit anything that
the host can observe, and the failure mode is an IP ban that looks like a
code bug.

The bucket is a standard token bucket -- tokens accrue at a fixed rate up to
a burst ceiling, and a request costs one token. Burst matters here: a batch
upload of forty URLs from one job board would otherwise serialize into forty
separate one-second waits even when the host is perfectly happy to answer
three at once.
"""

import logging
import time
from urllib.parse import urlsplit

from redis import Redis

from app.config import settings
from app.queues import get_redis

logger = logging.getLogger(__name__)

_KEY_PREFIX = "ratelimit:"

# Token bucket as a Lua script, which Redis runs atomically.
#
# Atomicity is the entire point. The read-modify-write of "how many tokens are
# left, subtract one, store" is three round trips from Python, and two workers
# interleaving inside that window both observe a token and both spend it --
# so the limit silently becomes 2x with two workers and Nx with N. A script
# executes start to finish with nothing else running, which is what makes the
# limit hold no matter how many processes share it.
#
# Returns the seconds to wait: 0 means a token was taken, positive means the
# caller should sleep that long and try again.
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local bucket = redis.call('HMGET', key, 'tokens', 'updated')
local tokens = tonumber(bucket[1])
local updated = tonumber(bucket[2])

if tokens == nil then
  -- First request for this host. Start full, minus the one being taken, so a
  -- single URL against a fresh domain is never made to wait.
  tokens = burst
  updated = now
end

-- Accrue whatever the elapsed time earned, capped at the burst ceiling.
local elapsed = math.max(0, now - updated)
tokens = math.min(burst, tokens + elapsed * rate)

local wait = 0
if tokens >= 1 then
  tokens = tokens - 1
else
  -- Not enough for a whole token: report how long until there is one. The
  -- caller sleeps and retries rather than being refused outright, because
  -- the work is legitimate and merely early.
  wait = (1 - tokens) / rate
end

redis.call('HSET', key, 'tokens', tokens, 'updated', now)
-- Expire idle buckets so one key per host visited does not accumulate
-- forever. Generous enough that an active host's bucket never lapses.
redis.call('EXPIRE', key, 3600)

return tostring(wait)
"""


def domain_of(url: str) -> str:
    """The rate-limit key for a URL: its lowercased host.

    Host rather than registered domain. `boards.greenhouse.io` and
    `example.greenhouse.io` may sit behind the same infrastructure, but
    grouping by registered domain needs a public-suffix list to avoid
    treating everything under `.co.uk` as one host -- a dependency and a
    class of bug that buys little here.
    """
    return (urlsplit(url).hostname or "").lower()


class RateLimiter:
    """Token bucket per host, backed by Redis."""

    def __init__(
        self,
        redis: Redis | None = None,
        rate: float | None = None,
        burst: int | None = None,
    ) -> None:
        self._redis = redis if redis is not None else get_redis()
        self._rate = rate if rate is not None else settings.fetch_rate_per_second
        self._burst = burst if burst is not None else settings.fetch_rate_burst
        # register_script uses EVALSHA with an EVAL fallback, so the body
        # crosses the wire once rather than on every call.
        self._script = self._redis.register_script(_TOKEN_BUCKET_LUA)

    def try_acquire(self, host: str) -> float:
        """Take a token for `host`, or report how long until one exists.

        Returns 0.0 when the caller may proceed immediately.
        """
        if not host:
            # No host means the URL was not fetchable anyway. Returning 0
            # rather than raising keeps this out of the caller's error paths;
            # the fetch itself will reject it.
            return 0.0

        raw = self._script(
            keys=[f"{_KEY_PREFIX}{host}"],
            args=[self._rate, self._burst, time.time()],
        )
        return float(raw)

    def acquire(self, host: str, max_wait: float | None = None) -> bool:
        """Block until a token is available for `host`, up to `max_wait`.

        Returns False if the wait would exceed the budget. That is deliberately
        not an exception: a worker that cannot get a token promptly should
        release its slot and let the retry policy reschedule the job, which is
        ordinary backpressure rather than a failure.

        The wait is bounded because a blocked worker is a worker doing nothing.
        Under a batch of 500 URLs against one host, unbounded waiting would
        park every worker on the same bucket and stall the whole queue --
        including jobs for hosts that are not busy at all.
        """
        budget = (
            max_wait if max_wait is not None else settings.fetch_rate_max_wait_seconds
        )
        deadline = time.monotonic() + budget

        while True:
            wait = self.try_acquire(host)
            if wait <= 0:
                return True

            remaining = deadline - time.monotonic()
            if wait > remaining:
                logger.info(
                    "ratelimit: %s needs %.2fs, only %.2fs of budget left", host, wait, remaining
                )
                return False

            time.sleep(wait)
