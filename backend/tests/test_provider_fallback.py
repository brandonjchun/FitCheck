"""Routing around an exhausted quota, and coming back.

The behaviour these pin down was written for a measured failure: the Gemini
free tier is 20 requests per day, discovered by exceeding it mid-crawl. The
question is not whether a quota error is handled -- RQ would retry it either
way -- but whether the *retry lands somewhere that can succeed*.

Uses a real Redis, because the cooldown is a Redis key with a TTL and that
TTL is the mechanism rather than an implementation detail. A fake dict would
confirm the code sets a value and prove nothing about expiry, which is the
part that makes the swap-back automatic.
"""

import pytest

from app.providers.base import LLMPermanentError, LLMQuotaError, LLMTransientError
from app.providers.fallback import (
    FallbackProvider,
    clear_cooldown,
    cooldown_remaining,
    start_cooldown,
)

SCHEMA = {"type": "object"}


class FakeProvider:
    """Records calls and raises whatever it was told to."""

    def __init__(self, name: str, raises: Exception | None = None, result: str = "{}"):
        self.name = name
        self._raises = raises
        self._result = result
        self.calls = 0

    def complete_json(self, prompt: str, schema: dict) -> str:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._result


@pytest.fixture(autouse=True)
def clean_cooldowns():
    for name in ("primary", "gemini", "ollama"):
        clear_cooldown(name)
    yield
    for name in ("primary", "gemini", "ollama"):
        clear_cooldown(name)


class TestFailover:
    def test_healthy_primary_is_used(self) -> None:
        primary = FakeProvider("primary", result='{"ok":1}')
        standby = FakeProvider("standby")

        assert FallbackProvider(primary, standby).complete_json("p", SCHEMA) == '{"ok":1}'
        assert standby.calls == 0

    def test_the_call_that_hits_the_quota_still_succeeds(self) -> None:
        """The point of re-issuing rather than re-raising.

        Raising would hand this job to RQ's retry policy, which comes back in
        ten seconds to a provider that is still exhausted -- spending the
        retry budget on waiting instead of on the work. The job that
        discovers the limit is the one most worth completing, because it is
        already at the front of the queue.
        """
        primary = FakeProvider("primary", raises=LLMQuotaError("out of quota"))
        standby = FakeProvider("standby", result='{"from":"standby"}')

        result = FallbackProvider(primary, standby).complete_json("p", SCHEMA)

        assert result == '{"from":"standby"}'
        assert standby.calls == 1

    def test_subsequent_calls_skip_the_exhausted_provider(self) -> None:
        """Otherwise every job pays a failed call to rediscover the quota.

        Across four workers that is four wasted calls per window, against a
        limit of twenty per day.
        """
        primary = FakeProvider("primary", raises=LLMQuotaError("out of quota"))
        standby = FakeProvider("standby")
        provider = FallbackProvider(primary, standby)

        provider.complete_json("first", SCHEMA)
        provider.complete_json("second", SCHEMA)
        provider.complete_json("third", SCHEMA)

        assert primary.calls == 1
        assert standby.calls == 3

    def test_a_non_quota_failure_is_not_routed_around(self) -> None:
        """A malformed response or a network blip says nothing about quota.

        Falling back on those would silently move all work to the weaker
        model on the first transient hiccup and never move it back, because
        no cooldown was ever recorded to expire.
        """
        primary = FakeProvider("primary", raises=LLMTransientError("timeout"))
        standby = FakeProvider("standby")

        with pytest.raises(LLMTransientError):
            FallbackProvider(primary, standby).complete_json("p", SCHEMA)

        assert standby.calls == 0

    def test_a_permanent_failure_is_not_routed_around(self) -> None:
        """A bad API key fails on the standby too, differently, and hides the
        real cause behind a second error."""
        primary = FakeProvider("primary", raises=LLMPermanentError("bad key"))
        standby = FakeProvider("standby")

        with pytest.raises(LLMPermanentError):
            FallbackProvider(primary, standby).complete_json("p", SCHEMA)

        assert standby.calls == 0


class TestCooldownClock:
    def test_the_providers_own_estimate_wins(self) -> None:
        """Gemini says "retry in 49.7s". Applying a four-hour default instead
        would abandon the better model for most of a day over what was a
        per-minute limit."""
        primary = FakeProvider(
            "primary", raises=LLMQuotaError("slow down", retry_after_seconds=45)
        )
        FallbackProvider(primary, FakeProvider("standby")).complete_json("p", SCHEMA)

        remaining = cooldown_remaining("primary")
        assert 0 < remaining <= 45

    def test_the_default_applies_when_no_estimate_is_given(self) -> None:
        from app.config import settings

        primary = FakeProvider("primary", raises=LLMQuotaError("no estimate"))
        FallbackProvider(primary, FakeProvider("standby")).complete_json("p", SCHEMA)

        assert cooldown_remaining("primary") > 60
        assert cooldown_remaining("primary") <= settings.llm_quota_cooldown_seconds

    def test_an_absurd_estimate_is_capped(self) -> None:
        """A provider answering "retry in 30 days" would otherwise become
        permanently disabled by a value nobody chose."""
        from app.config import settings

        start_cooldown("primary", 30 * 24 * 60 * 60)

        assert cooldown_remaining("primary") <= settings.llm_quota_max_cooldown_seconds

    def test_expiry_restores_the_primary(self) -> None:
        """The swap-back, and the reason no scheduler exists.

        The key expires on its own, so the next call after it tries the
        preferred provider again. Nothing polls; expiry *is* the check, and
        it happens exactly when there is work to do rather than at 3am
        against an empty queue.
        """
        primary = FakeProvider("primary", result='{"back":1}')
        standby = FakeProvider("standby")
        provider = FallbackProvider(primary, standby)

        start_cooldown("primary", 1)
        assert provider.complete_json("during", SCHEMA) is not None
        assert standby.calls == 1

        clear_cooldown("primary")  # stands in for the TTL elapsing

        assert provider.complete_json("after", SCHEMA) == '{"back":1}'
        assert primary.calls == 1

    def test_cooldowns_are_per_provider(self) -> None:
        """Exhausting Gemini must not mark Ollama unavailable -- they share
        no quota and one is not evidence about the other."""
        start_cooldown("gemini", 300)

        assert cooldown_remaining("gemini") > 0
        assert cooldown_remaining("ollama") == 0


class TestReportedName:
    def test_names_the_provider_that_will_serve(self) -> None:
        """This string reaches error messages. "gemini returned output that
        did not match the schema" when Ollama produced it sends somebody
        debugging the wrong system."""
        provider = FallbackProvider(FakeProvider("gemini"), FakeProvider("ollama"))
        assert provider.name == "gemini"

        start_cooldown("gemini", 300)

        assert provider.name == "ollama"


class TestFactoryWiring:
    def test_the_profile_path_is_wrapped(self) -> None:
        from app.providers import get_provider

        get_provider.cache_clear()
        try:
            assert isinstance(get_provider("profile"), FallbackProvider)
        finally:
            get_provider.cache_clear()

    def test_no_wrapper_when_the_fallback_is_the_primary(self, monkeypatch) -> None:
        """Postings already run on Ollama, so wrapping them in a fallback to
        Ollama would add a layer that can only route a provider to itself."""
        from app.config import settings
        from app.providers import get_provider
        from app.providers.ollama import OllamaProvider

        monkeypatch.setattr(settings, "llm_provider_posting", "ollama", raising=False)
        monkeypatch.setattr(settings, "llm_fallback_provider", "ollama", raising=False)
        get_provider.cache_clear()
        try:
            assert isinstance(get_provider("posting"), OllamaProvider)
        finally:
            get_provider.cache_clear()
