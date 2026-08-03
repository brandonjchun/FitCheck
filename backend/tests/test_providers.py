"""Tests for provider selection and error classification.

Neither provider is constructed here -- doing so would build a real SDK
client and, for Gemini, require a key. What is tested is the part that
carries the judgment: _classify, which decides whether a failure is worth
retrying. In M5 that decision is the difference between a job that recovers
and one that dead-letters.

Both _classify functions match on message text rather than on SDK exception
types, deliberately, to keep google-genai and ollama out of every module that
handles an extraction failure. Message matching is fragile by nature, which
is precisely why it needs tests.
"""

import pytest

from app.providers import (
    LLMPermanentError,
    LLMProvider,
    LLMTransientError,
    get_provider,
)
from app.providers.gemini import _classify as classify_gemini
from app.providers.ollama import _classify as classify_ollama


class TestGeminiClassification:
    @pytest.mark.parametrize(
        "message",
        [
            "API key not valid. Please pass a valid API key.",
            "401 Unauthorized",
            "PERMISSION_DENIED: caller does not have permission",
            "404 models/gemini-9-ultra is not found for API version v1",
            "Invalid JSON payload received",
        ],
    )
    def test_permanent_failures(self, message: str) -> None:
        assert isinstance(classify_gemini(Exception(message)), LLMPermanentError)

    @pytest.mark.parametrize(
        "message",
        [
            "429 RESOURCE_EXHAUSTED: quota exceeded",
            "503 Service Unavailable",
            "500 Internal error encountered",
            "Connection reset by peer",
            "Deadline exceeded",
        ],
    )
    def test_transient_failures(self, message: str) -> None:
        assert isinstance(classify_gemini(Exception(message)), LLMTransientError)

    def test_matching_is_case_insensitive(self) -> None:
        # SDKs are inconsistent about casing across error paths.
        assert isinstance(classify_gemini(Exception("API KEY INVALID")), LLMPermanentError)

    def test_unrecognised_failure_defaults_to_transient(self) -> None:
        """Defaulting to retry is the safe direction to be wrong in.

        A wrongly-transient error costs a few retries. A wrongly-permanent one
        silently drops recoverable work, which is the failure mode that
        actually loses a user's upload.
        """
        assert isinstance(classify_gemini(Exception("¯\\_(ツ)_/¯")), LLMTransientError)

    def test_original_message_is_preserved(self) -> None:
        # This text ends up in logs and, in M5, in jobs.last_error.
        error = classify_gemini(Exception("429 quota exceeded"))

        assert "429 quota exceeded" in str(error)


class TestOllamaClassification:
    @pytest.mark.parametrize(
        "message",
        [
            'model "llama3.1" not found, try pulling it first',
            "no such model",
        ],
    )
    def test_missing_model_is_permanent(self, message: str) -> None:
        """`ollama pull` is a manual step no amount of waiting performs."""
        assert isinstance(classify_ollama(Exception(message)), LLMPermanentError)

    def test_missing_model_error_names_the_fix(self) -> None:
        error = classify_ollama(Exception("no such model"))

        assert "ollama pull" in str(error)

    @pytest.mark.parametrize(
        "message",
        [
            "connection refused",
            "[Errno 61] Connection refused",
            "read timed out",
            "connection reset by peer",
        ],
    )
    def test_daemon_unavailable_is_transient(self, message: str) -> None:
        """The interesting call, and the opposite of the hosted-API answer.

        Ollama is a local daemon under the operator's control and M5 backs off
        to 10s, 60s, 300s. A daemon that was down when the job was picked up is
        plausibly back five minutes later. Failing permanently would
        dead-letter work a single retry would have completed.
        """
        assert isinstance(classify_ollama(Exception(message)), LLMTransientError)


class TestClassificationBoundary:
    """The two providers disagree about the same string, on purpose."""

    def test_not_found_means_different_things_to_each_provider(self) -> None:
        # Both permanent, but for unrelated reasons: a bad model name to
        # Gemini, an unpulled model to Ollama. Pinned because the shared
        # substring makes it look like duplication worth "unifying" -- and a
        # shared classifier would have to pick one provider's semantics.
        assert isinstance(classify_gemini(Exception("not found")), LLMPermanentError)
        assert isinstance(classify_ollama(Exception("not found")), LLMPermanentError)

    def test_connection_refused_splits_the_two(self) -> None:
        """Same failure, opposite verdicts -- this is the design, not a bug."""
        assert isinstance(
            classify_ollama(Exception("connection refused")), LLMTransientError
        )
        assert isinstance(
            classify_gemini(Exception("connection refused")), LLMTransientError
        )


class TestGetProvider:
    def test_unknown_provider_is_permanent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The unreachable branch, made reachable.

        pydantic-settings rejects anything outside the Literal at startup, so
        this cannot happen today. It exists so that widening the Literal
        without updating the factory fails loudly instead of returning None,
        and that guarantee is worth a test.
        """
        from app.config import settings

        monkeypatch.setattr(
            settings, "llm_provider_profile", "anthropic", raising=False
        )
        get_provider.cache_clear()

        with pytest.raises(LLMPermanentError, match="anthropic"):
            get_provider("profile")

        get_provider.cache_clear()

    def test_provider_is_cached_per_task(self) -> None:
        """One instance per task per process, not one overall.

        Two entries rather than one, because profile and posting are routed
        to different providers -- a single-entry cache would hand whichever
        task ran second the other one's client.
        """
        assert get_provider.cache_info().maxsize == 2

    def test_tasks_route_independently(self, monkeypatch) -> None:
        """The whole point of the split.

        Postings run local in bulk because the Gemini free tier is 20
        requests a day; resumes stay on Gemini because local loses
        years-inference and seniority. If these ever resolve to the same
        provider by accident, one of those two facts is being ignored.
        """
        from app.config import settings

        monkeypatch.setattr(settings, "llm_provider_profile", "gemini", raising=False)
        monkeypatch.setattr(settings, "llm_provider_posting", "ollama", raising=False)
        get_provider.cache_clear()

        assert get_provider("profile").name == "gemini"
        assert get_provider("posting").name == "ollama"

        get_provider.cache_clear()

    def test_an_unnamed_task_gets_the_higher_quality_path(self, monkeypatch) -> None:
        """Defaulting to the profile provider is deliberate.

        A caller that forgets to name its task then fails on quota -- loud --
        rather than quietly extracting at lower quality, which nothing
        downstream can detect.
        """
        from app.config import settings

        monkeypatch.setattr(settings, "llm_provider_profile", "gemini", raising=False)
        monkeypatch.setattr(settings, "llm_provider_posting", "ollama", raising=False)
        get_provider.cache_clear()

        assert get_provider().name == get_provider("profile").name

        get_provider.cache_clear()


class TestProtocolConformance:
    """A structural check that costs nothing and catches a real mistake.

    LLMProvider is a runtime_checkable Protocol, so isinstance verifies the
    method exists. It does not verify signatures -- that is the type checker's
    job -- but it does catch a provider that never implemented the method.
    """

    def test_a_bare_object_with_the_method_satisfies_the_protocol(self) -> None:
        class Minimal:
            name = "minimal"

            def complete_json(self, prompt: str, schema: dict) -> str:
                return "{}"

        assert isinstance(Minimal(), LLMProvider)

    def test_an_object_without_the_method_does_not(self) -> None:
        class NotAProvider:
            name = "nope"

        assert not isinstance(NotAProvider(), LLMProvider)
