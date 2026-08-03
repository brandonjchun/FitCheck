"""Provider selection.

The rest of the application imports get_provider() and never mentions Gemini
or Ollama by name. That is what makes the toggle a config change rather than
a code change -- and what will make deprecating the loser a one-file delete.
"""

from functools import lru_cache

from app.config import settings
from app.providers.base import (
    LLMError,
    LLMPermanentError,
    LLMProvider,
    LLMTransientError,
)

__all__ = [
    "LLMError",
    "LLMPermanentError",
    "LLMProvider",
    "LLMTransientError",
    "get_provider",
]


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    """Return the provider named by settings.llm_provider.

    Cached because construction is not free -- it builds an SDK client and,
    for Gemini, validates that a key exists. One instance per process is
    correct; both underlying clients are safe to share across requests.

    The import happens inside each branch so that choosing one provider never
    requires the other's package to be installed.
    """
    if settings.llm_provider == "gemini":
        from app.providers.gemini import GeminiProvider

        return GeminiProvider()

    if settings.llm_provider == "ollama":
        from app.providers.ollama import OllamaProvider

        return OllamaProvider()

    # Unreachable while llm_provider is a Literal -- pydantic-settings rejects
    # anything else at startup. Kept so that widening the Literal without
    # updating this function fails loudly instead of returning None.
    raise LLMPermanentError(f"Unknown llm_provider: {settings.llm_provider!r}")
