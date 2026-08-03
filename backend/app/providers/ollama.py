"""Ollama-backed LLM provider.

Runs against a local Ollama daemon. No API key; the equivalent failure modes
are "the daemon isn't running" and "that model isn't pulled".
"""

from typing import Any

from app.config import settings
from app.providers.base import LLMPermanentError, LLMTransientError


class OllamaProvider:
    """Structured JSON via a locally running Ollama daemon."""

    name = "ollama"

    def __init__(self) -> None:
        # Imported here rather than at module scope so the gemini path does
        # not require the ollama package, and vice versa.
        try:
            from ollama import Client
        except ImportError as exc:  # pragma: no cover
            raise LLMPermanentError(
                "ollama is not installed. Run: pip install ollama"
            ) from exc

        self._client = Client(host=settings.ollama_host)
        self._model = settings.ollama_model

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> str:
        try:
            response = self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                format=schema,
            )
        except Exception as exc:
            raise _classify(exc) from exc

        return response.message.content or ""


def _classify(exc: Exception) -> LLMTransientError | LLMPermanentError:
    """Map an Ollama failure onto the retry / don't-retry split.

    The interesting call is "connection refused". It is classified as
    TRANSIENT, which is the opposite of how the same error would be treated
    against a hosted API. The reasoning: Ollama is a local daemon under the
    operator's control, and the retry does not fire immediately -- M5 backs
    off to 10s, 60s, then 300s. A daemon that was down when the job was
    picked up is plausibly back up five minutes later, whether because it
    restarted or because someone launched it. Failing permanently would
    dead-letter work that a single retry would have completed.

    A missing model is the opposite: `ollama pull` is a manual step that no
    amount of waiting performs, so retrying only burns worker slots.
    """
    text = str(exc).lower()

    # Model not pulled -- requires human action, never resolves on its own.
    if "not found" in text or "no such model" in text or "try pulling" in text:
        return LLMPermanentError(
            f"Ollama model {settings.ollama_model!r} is not available. "
            f"Run: ollama pull {settings.ollama_model} ({exc})"
        )

    # Daemon down, timeout, or connection reset -- see docstring.
    return LLMTransientError(f"Ollama call failed, retryable: {exc}")
