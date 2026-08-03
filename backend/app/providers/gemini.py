"""Gemini-backed LLM provider.

WORKED EXAMPLE. Read this, then write providers/ollama.py by analogy.
"""

from typing import Any

from app.config import settings
from app.providers.base import (
    LLMPermanentError,
    LLMQuotaError,
    LLMTransientError,
)


class GeminiProvider:
    """Structured JSON via the Google Gen AI SDK's free tier."""

    name = "gemini"

    def __init__(self) -> None:
        # Imported inside __init__, not at module scope, so that running on
        # the ollama path does not require google-genai to be installed --
        # and vice versa. The factory only ever constructs the one you chose.
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise LLMPermanentError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc

        if not settings.gemini_api_key:
            # A missing key can never succeed on retry, so it is permanent.
            # Failing here, at construction, means the app dies at startup
            # with a clear message rather than on the first upload.
            raise LLMPermanentError(
                "GEMINI_API_KEY is not set. Get a free key at aistudio.google.com "
                "and add it to backend/.env, or set LLM_PROVIDER=ollama."
            )

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> str:
        try:
            interaction = self._client.interactions.create(
                model=self._model,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
            )
        except Exception as exc:
            raise _classify(exc) from exc

        return interaction.output_text


def _classify(exc: Exception) -> LLMTransientError | LLMPermanentError:
    """Map an SDK exception onto our retry/don't-retry split.

    The SDK raises its own exception types, and we deliberately do not import
    them to match on -- that would leak google-genai into every module that
    handles an extraction failure. Matching on the message is cruder but keeps
    the dependency contained to this file.
    """
    text = str(exc).lower()

    # Quota first, and before the permanent markers -- a 429 body mentions
    # "billing details", and "invalid" appearing anywhere in that prose would
    # otherwise classify an exhausted quota as permanent and dead-letter work
    # that resets on its own in under a minute.
    quota_markers = ("quota", "resource_exhausted", "too_many_requests", "rate limit")
    if any(marker in text for marker in quota_markers):
        return LLMQuotaError(
            f"Gemini quota exhausted: {exc}", retry_after_seconds=_retry_after(str(exc))
        )

    permanent_markers = ("api key", "unauthorized", "permission", "not found", "invalid")
    if any(marker in text for marker in permanent_markers):
        return LLMPermanentError(f"Gemini rejected the request permanently: {exc}")

    # Default to transient. Getting this backwards in the safe direction costs
    # a few wasted retries; the other way silently drops recoverable work.
    return LLMTransientError(f"Gemini call failed, retryable: {exc}")


def _retry_after(message: str) -> float | None:
    """Pull Gemini's own "retry in 49.7s" out of the error body.

    Text-matched, like the classification above and for the same reason: the
    structured field lives on an SDK exception type this module declines to
    import. Returns None when absent, and the caller falls back to a
    configured cooldown.
    """
    import re

    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", message, re.IGNORECASE)
    return float(match.group(1)) if match else None
