"""Turn a resume's raw text into a validated ExtractedProfile.

Provider-agnostic: the prompt, the validation, and the error classification
live here once, and get_provider() decides which LLM actually runs. Swapping
Gemini for Ollama changes a line in .env and nothing in this file.
"""

import logging

from pydantic import ValidationError

from app.extraction import ExtractedProfile
from app.providers import LLMPermanentError, LLMTransientError, get_provider

logger = logging.getLogger(__name__)


class EmptyDocumentError(LLMPermanentError):
    """The document has no extractable text, so there is nothing to send.

    Permanent by inheritance, which is the whole point: no number of retries
    will put a text layer into a scanned PDF. Subclassing rather than raising
    LLMPermanentError directly means callers that already handle the
    permanent case need no change, while logs name the actual cause instead
    of implicating the provider for a failure it had no part in.
    """

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
shorter than their total career. Infer it from the date range of the role or \
project the skill appears in -- a role dated Aug 2022 to Aug 2025 that lists \
Python means roughly 3 years of Python. Sum the ranges when a skill appears \
in several roles. Leave it null only when the skill cannot be attributed to \
any dated role, such as one that appears solely in a skills-section keyword \
list.
- `source` records where the skill appears: "experience" for a bullet under \
a job or role, "project" for a personal or academic project, "education" for \
coursework or a degree, "skills_list" for a bare skills or technologies \
section with no surrounding accomplishment, "unknown" if it cannot be placed. \
A skill usually appears in several of these -- a technologies section \
normally repeats what the bullets already demonstrated -- so when it does, \
pick the strongest, in the order experience > project > education > \
skills_list, and quote the evidence from that same place. Never answer \
"skills_list" for a skill that also appears in a job bullet.
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
        A validated ExtractedProfile holding exactly what the model returned.
        Skill names are NOT canonicalized here -- see skills.normalize_skill_items
        for why that happens on read instead.

    Raises:
        EmptyDocumentError: The document has no extractable text.
        LLMTransientError: The provider failed in a retryable way, or the
            model returned output that did not satisfy the schema.
        LLMPermanentError: The provider is misconfigured (bad key, missing
            model) in a way retrying cannot fix.
    """
    if not raw_text.strip():
        # Still short-circuits before spending an LLM call, but raises rather
        # than returning an empty profile. Returning one meant the caller
        # stored a populated-looking extraction for a document the model never
        # saw, and `extraction_ok` -- whose entire job is to separate "found
        # no skills" from "never ran" -- reported true for the second case.
        raise EmptyDocumentError(
            "document contains no extractable text; a scanned PDF with no text "
            "layer is the usual cause"
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

    # Returned exactly as validated. Normalization deliberately does NOT run
    # here: this value is what gets persisted to profiles.extracted, and the
    # column is worth more holding what the model actually said.
    #
    # The earlier version canonicalized before returning, so the original
    # spellings never reached storage. That made every future alias-map
    # addition a full LLM re-run -- the raw names needed to re-map were gone,
    # and collapsing duplicates had already discarded the losing entry's
    # evidence span. Canonicalizing on read (models.Profile.skills) costs a
    # dict lookup per skill and makes an alias fix retroactive for free.
    return profile
