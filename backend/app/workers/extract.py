"""Turn raw text into validated structured data -- resumes and job postings.

Provider-agnostic: the prompts, the validation, and the error classification
live here once, and get_provider(task) decides which LLM actually runs. Each
extraction names the job it is doing rather than the model it wants, so the
two can be routed to different providers -- which they are, because postings
run local in bulk while resumes stay on Gemini for quality. Changing either
is a line in .env and nothing in this file.

Both extractions share everything except the prompt and the schema, which is
the argument for one module. What they deliberately do *not* share is the
output shape -- see ExtractedPosting for why a resume and a job description
are not the same document with different words in it.
"""

import logging

from pydantic import ValidationError

from app.extraction import ExtractedPosting, ExtractedProfile
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

# Postings get a smaller budget than resumes, which looks backwards until you
# look at what the extra characters are. A fetched posting is a whole web
# page reduced to text: the job description is usually the first 2-4k
# characters, and the rest is the company's boilerplate, benefits, EEO
# statement, and a footer of other openings. Measured across 12 real postings
# the usable range was 1,700-18,000 characters, so 20k keeps every one of
# them intact.
#
# The failure this bounds is worse than cost. A footer listing eight other
# roles is *plausible job description text*, so a model reading past the end
# of the posting will happily attribute those requirements to this one --
# extracting skills for a job that is not the job. Truncation is a blunt
# guard against that, and the noise selectors in fetch.py are the sharp one.
MAX_POSTING_CHARS = 20_000

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

    provider = get_provider("profile")
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


POSTING_INSTRUCTIONS = """\
You extract structured requirements from job postings. You are given the \
text of one job posting and must return JSON matching the provided schema \
exactly.

Rules:

- Extract only what the posting states. Never invent skills, titles, or \
seniority that do not appear in the text.
- The text is a web page reduced to prose, so it may contain navigation, \
benefits, an equal-opportunity statement, or links to *other* openings. \
Extract requirements for the single role this page is about. If a phrase \
belongs to a different job, ignore it.
- `name` must be the short canonical term a resume would use -- \
"TypeScript", "Kubernetes", "Unit testing", "DuckDB". Never put a whole \
requirement sentence in it. "Experience with automated test coverage" is a \
name of "Unit testing"; "comfort with AI-assisted agentic coding tools" is a \
name of "AI coding tools". The full wording belongs in `evidence`, which is \
what it is for.
- Skip requirements that name no skill at all. "Excellent communication", \
"a growth mindset", and "5 years in a fast-paced environment" are real \
things a posting asks for and are not skills; listing them produces \
requirements that no resume can ever satisfy.
- For each skill, `evidence` must be a phrase copied verbatim from the \
posting. If you cannot quote a supporting phrase, do not list the skill.
- `necessity` is the most important field. Answer "required" when the \
posting treats the skill as a must-have -- listed under requirements or \
qualifications, or worded as "must have", "strong background in", or \
"X+ years of". Answer "preferred" when it is explicitly optional -- \
"preferred", "nice to have", "bonus", "a plus", "desirable", "familiarity \
with". Answer "unknown" only when the posting mentions the skill with no \
signal either way, such as in a paragraph describing the team's stack.
- `min_years` on a skill is a threshold the posting asks for in that skill \
specifically, as in "5+ years of Java". Leave it null when the posting does \
not state one. Do not derive it from the seniority of the role.
- `min_years_experience` is the role-level threshold, as in "3+ years of \
professional software engineering". It is separate from any per-skill \
number. Leave it null if the posting does not state one.
- `title` is the role's title. When a KNOWN TITLE block is supplied below, \
that is the board's own structured field -- return it verbatim rather than \
re-deriving it from the page. Without one, take the title from the posting's \
heading, and answer null only if the text states none.
- `seniority` reflects the scope of the role. **Read it off the title \
whenever the title states a level**, because most postings never restate the \
level in the body:

    Staff, Principal, Distinguished    -> "staff"
    Senior, Sr., Snr                   -> "senior"
    Junior, Jr., Intern, New Grad      -> "junior"

  A title carrying a stronger level wins over a weaker one, so "Senior Staff \
Machine Learning Engineer" is "staff", not "senior". When the title states no \
level, use the scope the body describes. Only then does "unknown" apply, and \
it is still the right answer rather than inferring a level from a years \
requirement alone.
- `remote_type` is "remote" only if the role can be performed fully \
remotely. A role requiring any days on site is "hybrid". Use "unknown" when \
the posting does not say -- most do not.

Return only the JSON object.\
"""


def extract_posting(raw_text: str, title: str | None = None) -> ExtractedPosting:
    """Extract structured requirements from a job posting's text.

    Args:
        raw_text: Readable text produced by fetch.html_to_text.
        title: The board's own title for this posting, when there is one.
            Supplied to the prompt rather than only stored beside it, because
            it is the field the level is stated in. A posting body says
            "you will own the reliability of our ingest pipeline" and never
            says "staff"; the title says "Staff Data Engineer". Extracting
            from the body alone left `seniority` to be guessed, and a local 8B
            model guessed badly enough to empty the feed's staff filter -- see
            app.seniority for the measurement.

            Keyword-optional because Path A has no board behind it: a
            user-submitted URL genuinely has no title until this call
            produces one.

    Returns:
        A validated ExtractedPosting holding exactly what the model returned.
        Skill names are NOT canonicalized here, for the same reason resume
        skills are not -- see models.JobPosting.skills.

    Raises:
        EmptyDocumentError: The posting has no extractable text.
        LLMTransientError: The provider failed in a retryable way, or the
            model returned output that did not satisfy the schema.
        LLMPermanentError: The provider is misconfigured in a way retrying
            cannot fix.
    """
    if not raw_text.strip():
        # Same reasoning as the resume path: raising keeps `extraction_ok`
        # able to distinguish "the model found nothing" from "the model never
        # ran". A posting that fetched to an empty body is usually a
        # client-rendered page, which no retry improves.
        raise EmptyDocumentError(
            "posting contains no extractable text; a JavaScript-rendered page "
            "is the usual cause"
        )

    document = raw_text[:MAX_POSTING_CHARS]
    if len(raw_text) > MAX_POSTING_CHARS:
        logger.warning(
            "posting truncated for extraction: %d chars -> %d",
            len(raw_text),
            MAX_POSTING_CHARS,
        )

    provider = get_provider("posting")
    # The title goes *before* the body. A 20k-character page reduced to prose
    # buries a one-line hint appended after it, and the whole point of passing
    # the title is that the model reads it.
    known_title = f"=== KNOWN TITLE ===\n{title.strip()}\n\n" if title and title.strip() else ""
    prompt = (
        f"{POSTING_INSTRUCTIONS}\n\n{known_title}=== JOB POSTING TEXT ===\n{document}"
    )

    raw_json = provider.complete_json(prompt, ExtractedPosting.model_json_schema())

    try:
        posting = ExtractedPosting.model_validate_json(raw_json)
    except ValidationError as exc:
        logger.warning("posting extraction failed validation; raw output: %.500s", raw_json)
        raise LLMTransientError(
            f"{provider.name} returned output that did not match the schema: {exc}"
        ) from exc

    return posting
