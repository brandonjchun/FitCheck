"""Turn a resume's raw text into a validated ExtractedProfile.

Provider-agnostic: the prompt, the validation, and the error classification
live here once, and get_provider() decides which LLM actually runs. Swapping
Gemini for Ollama changes a line in .env and nothing in this file.
"""

import logging

from pydantic import ValidationError

from app.extraction import ExtractedProfile
from app.providers import LLMTransientError, get_provider
from app.skills import normalize_skill

logger = logging.getLogger(__name__)

# Resumes are 1-3 pages; 40k characters is roughly 15 pages and already far
# past anything legitimate. Truncating bounds both cost and context usage,
# and a document this long is malformed rather than thorough.
MAX_RESUME_CHARS = 40_000

SYSTEM_INSTRUCTIONS = """\
You extract structured data from resumes. You are given the raw text of one \
resume and must return JSON matching the provided schema exactly.

Rules:

- Extract only what the resume states or directly implies. Never invent \
skills, employers, dates, or degrees that do not appear in the text.
- For each skill, `evidence` must be a phrase copied verbatim from the \
resume. If you cannot quote a supporting phrase, do not list the skill.
- `years` for a skill is the time the candidate has used it, which is often \
shorter than their total career. Leave it null unless the resume supports a \
number.
- `total_years_experience` is professional working experience. Exclude \
education. Leave it null if the resume gives no way to compute it.
- `seniority` reflects scope and responsibility, not tenure alone. Use \
"unknown" rather than guessing from years alone.

Return only the JSON object.\
"""


def extract_profile(raw_text: str) -> ExtractedProfile:
    """Extract a structured profile from resume text.

    Args:
        raw_text: The document text produced by app.documents.extract_text.

    Returns:
        A validated ExtractedProfile with skill names normalized.

    Raises:
        LLMTransientError: The provider failed in a retryable way, or the
            model returned output that did not satisfy the schema.
        LLMPermanentError: The provider is misconfigured (bad key, missing
            model) in a way retrying cannot fix.
    """
    if not raw_text.strip():
        # An empty document is not an LLM problem -- do not spend a call on it.
        return ExtractedProfile(
            skills=[], total_years_experience=None, seniority="unknown", education=[]
        )

    document = raw_text[:MAX_RESUME_CHARS]
    if len(raw_text) > MAX_RESUME_CHARS:
        logger.warning(
            "resume truncated for extraction: %d chars -> %d",
            len(raw_text),
            MAX_RESUME_CHARS,
        )

    provider = get_provider()
    prompt = f"{SYSTEM_INSTRUCTIONS}\n\n=== RESUME TEXT ===\n{document}"

    raw_json = provider.complete_json(prompt, ExtractedProfile.model_json_schema())

    try:
        profile = ExtractedProfile.model_validate_json(raw_json)
    except ValidationError as exc:
        # Schema-constrained decoding makes this rare but not impossible --
        # models occasionally emit malformed JSON or truncate mid-object.
        # It is TRANSIENT: the same prompt on a fresh call usually succeeds,
        # which is exactly the case retries exist for. The raw output is
        # logged (truncated) because debugging this without it is guesswork.
        logger.warning("extraction failed validation; raw output: %.500s", raw_json)
        raise LLMTransientError(
            f"{provider.name} returned output that did not match the schema: {exc}"
        ) from exc

    return _normalize(profile)


def _normalize(profile: ExtractedProfile) -> ExtractedProfile:
    """Canonicalize skill names, dropping blanks and duplicates.

    Runs after validation rather than inside the schema: normalization is a
    business rule about how we compare skills, not a constraint on what the
    model is allowed to return. Keeping it out of the Pydantic model means
    `extracted` in JSONB preserves what the LLM actually said, while the
    normalized form is what scoring uses.
    """
    seen: set[str] = set()
    normalized = []

    for skill in profile.skills:
        canonical = normalize_skill(skill.name)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(skill.model_copy(update={"name": canonical}))

    return profile.model_copy(update={"skills": normalized})
