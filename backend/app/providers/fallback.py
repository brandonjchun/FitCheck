"""Routing around an exhausted quota, and back again.

Wraps a preferred provider with a standby. While the preferred one is
available every call goes to it; when it reports its quota exhausted, work
moves to the standby and returns there on its own once the quota resets.

**The cooldown is a Redis key with a TTL, not a scheduled check.** That is the
whole design, and it is worth stating why, because "re-check every four
hours" sounds like it wants a cron job:

- **Expiry is the check.** The key vanishes on its own, so the next call after
  it does simply tries the preferred provider again. Nothing polls, nothing
  has to be running, and there is no scheduler to get out of step with
  reality.
- **The check happens exactly when it matters**, which is when there is work
  to do. A four-hourly job that fires at 3am against an empty queue has
  learned something nobody needed.
- **It is shared across processes.** Four workers hitting the same exhausted
  key would otherwise each have to discover the quota independently, spending
  four failed calls per window instead of one.

**The first call to discover the quota still completes.** It is not enough to
route future work away -- the job that found the limit would otherwise fail
and be retried by RQ into a provider that is still exhausted. So a quota
error is caught, the cooldown is recorded, and the same request is
immediately re-issued against the standby.

**The provider's own retry estimate wins where it offers one.** Gemini
answering "retry in 49.7s" against a fixed four-hour cooldown would abandon
the better model for most of a day over what was a per-minute limit.
"""

import logging
import time
from typing import Any

from app.config import settings
from app.providers.base import LLMProvider, LLMQuotaError

logger = logging.getLogger(__name__)

# One key per provider name, so exhausting Gemini does not mark Ollama
# unavailable -- and so the state is legible in redis-cli during a debug.
_COOLDOWN_PREFIX = "llm:cooldown:"


def _redis():
    # Imported lazily: app.queues builds a connection pool at import, and a
    # process that never uses a fallback should not pay for one.
    from app.queues import get_redis

    return get_redis()


def cooldown_remaining(provider_name: str) -> int:
    """Seconds until `provider_name` is worth trying again. 0 means now.

    Redis being unreachable answers 0 -- "try it". Failing open is right
    here: the cost of a wrong "try it" is one failed call that re-arms the
    cooldown, while a wrong "don't" would strand every job on the standby
    provider for as long as Redis stayed down.
    """
    try:
        ttl = _redis().ttl(f"{_COOLDOWN_PREFIX}{provider_name}")
    except Exception:
        return 0
    return ttl if ttl and ttl > 0 else 0


def start_cooldown(provider_name: str, seconds: float) -> None:
    """Mark a provider unavailable for `seconds`."""
    bounded = max(1, min(int(seconds), settings.llm_quota_max_cooldown_seconds))
    try:
        _redis().set(f"{_COOLDOWN_PREFIX}{provider_name}", str(int(time.time())), ex=bounded)
    except Exception as exc:
        # Instrumentation-grade failure. Without the key every call retries
        # the exhausted provider and falls back per call, which is slower and
        # noisier but still correct.
        logger.warning("could not record LLM cooldown: %s", exc)


def clear_cooldown(provider_name: str) -> None:
    try:
        _redis().delete(f"{_COOLDOWN_PREFIX}{provider_name}")
    except Exception:
        pass


class FallbackProvider:
    """A preferred provider with a standby for when its quota runs out."""

    def __init__(self, primary: LLMProvider, standby: LLMProvider) -> None:
        self._primary = primary
        self._standby = standby

    @property
    def name(self) -> str:
        """Reports which provider would serve the *next* call.

        Not a fixed string, because this name reaches error messages and
        logs. "gemini returned output that did not match the schema" when
        Ollama actually produced it would send someone debugging the wrong
        system.
        """
        return (
            self._standby.name
            if cooldown_remaining(self._primary.name)
            else self._primary.name
        )

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> str:
        remaining = cooldown_remaining(self._primary.name)
        if remaining:
            logger.info(
                "%s is in cooldown for %ss; using %s",
                self._primary.name,
                remaining,
                self._standby.name,
            )
            return self._standby.complete_json(prompt, schema)

        try:
            return self._primary.complete_json(prompt, schema)
        except LLMQuotaError as exc:
            # The provider's own estimate if it gave one, otherwise the
            # configured default. Gemini's free tier publishes both a
            # per-minute and a per-day limit and the message distinguishes
            # them, so trusting it is strictly better than guessing.
            cooldown = exc.retry_after_seconds or settings.llm_quota_cooldown_seconds
            start_cooldown(self._primary.name, cooldown)
            logger.warning(
                "%s quota exhausted; falling back to %s for %ss",
                self._primary.name,
                self._standby.name,
                int(cooldown),
            )
            # Re-issued rather than re-raised. Raising would hand this job to
            # RQ's retry policy, which would come back in 10 seconds to a
            # provider that is still exhausted -- burning the retry budget on
            # a wait rather than on the work.
            return self._standby.complete_json(prompt, schema)
