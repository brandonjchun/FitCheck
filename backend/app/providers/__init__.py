"""Provider selection, per task.

The rest of the application imports get_provider(task) and never mentions
Gemini or Ollama by name. That is what makes the routing a config change
rather than a code change -- and what will make deprecating a loser a
one-file delete.

**Why the choice is per task rather than global**, which is the interesting
part and was measured rather than assumed:

    task      volume            local model quality
    -------   ---------------   -------------------------------------------
    posting   hundreds/crawl    matches Gemini -- 5/5 verbatim evidence,
                                identical skill names, and faster (11s v 20s)
    profile   one per signup    materially worse -- 6 skills with inferred
                                years against Gemini's 10, and seniority
                                comes back "unknown" where Gemini says "mid"

The volumes and the quality requirements point in opposite directions, so
one global setting has to be wrong for one of them. Sending the bulk work
local is what makes a 300-800 posting catalog reachable at all: the Gemini
free tier is **20 requests per day**, measured by exceeding it, which would
put a single crawl's extraction two weeks out. Keeping resumes on Gemini
costs roughly one call per signup and protects the input that every
downstream score depends on.

Bigger is not the lever, incidentally. qwen2.5:14b quoted verbatim evidence
for 2 of 16 skills where llama3.1:8b managed 29 of 33 on the same resume --
nearly twice the parameters and dramatically worse at the one instruction
that makes extraction auditable.
"""

from functools import lru_cache
from typing import Literal

from app.config import settings
from app.providers.base import (
    LLMError,
    LLMPermanentError,
    LLMProvider,
    LLMQuotaError,
    LLMTransientError,
)

__all__ = [
    "LLMError",
    "LLMPermanentError",
    "LLMProvider",
    "LLMQuotaError",
    "LLMTransientError",
    "get_provider",
]

# Which extraction is being performed. Not "which model" -- callers name the
# job they are doing and configuration decides what serves it, which is the
# whole point of the seam.
Task = Literal["profile", "posting"]


def _build(name: str) -> LLMProvider:
    """Construct a provider by name.

    The import happens inside each branch so that choosing one provider never
    requires the other's package to be installed.
    """
    if name == "gemini":
        from app.providers.gemini import GeminiProvider

        return GeminiProvider()

    if name == "ollama":
        from app.providers.ollama import OllamaProvider

        return OllamaProvider()

    # Unreachable while the settings fields are Literals -- pydantic-settings
    # rejects anything else at startup. Kept so that widening one without
    # updating this function fails loudly instead of returning None.
    raise LLMPermanentError(f"Unknown provider: {name!r}")


@lru_cache(maxsize=len(("profile", "posting")))
def get_provider(task: Task = "profile") -> LLMProvider:
    """Return the provider configured for `task`.

    Cached per task because construction is not free -- it builds an SDK
    client and, for Gemini, validates that a key exists. One instance per
    task per process is correct; both underlying clients are safe to share.

    Cached also means a settings change needs a restart, which costs ten
    confused minutes the first time you flip the toggle and see no effect.

    Defaults to the profile provider so that a caller which forgets to name
    its task gets the *higher-quality* path rather than the cheaper one. The
    failure mode is then a quota error, which is loud, instead of quietly
    degraded extraction, which is not.
    """
    chosen = (
        settings.llm_provider_posting if task == "posting"
        else settings.llm_provider_profile
    )
    primary = _build(chosen)

    fallback = settings.llm_fallback_provider
    if not fallback or fallback == chosen:
        # Nothing to fall back to, or the fallback *is* the primary -- in
        # which case wrapping would add a layer that can only route a
        # provider to itself.
        return primary

    from app.providers.fallback import FallbackProvider

    return FallbackProvider(primary, _build(fallback))
