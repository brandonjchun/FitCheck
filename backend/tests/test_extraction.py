"""Tests for app.workers.extract -- the LLM extraction pipeline.

No network. LLMProvider is a Protocol, so a plain object with one
complete_json method satisfies it structurally -- which is exactly the
property base.py's docstring claims and this module cashes in.

What is under test is everything provider-independent: the short-circuit on
empty input, truncation, validation of what came back, the retry/don't-retry
classification, and skill normalization. The prompt is checked only where its
content is load-bearing.
"""

import json
from typing import Any

import pytest

from app.extraction import EducationItem, ExtractedProfile, SkillItem
from app.providers import LLMPermanentError, LLMTransientError
from app.workers import extract as extract_module
from app.workers.extract import MAX_RESUME_CHARS, extract_profile

VALID_RESPONSE = json.dumps(
    {
        "skills": [
            {"name": "JS", "years": 4.0, "evidence": "Built a React frontend in JS"},
            {"name": "Rust", "years": None, "evidence": "Hobby projects in Rust"},
        ],
        "total_years_experience": 4.0,
        "seniority": "mid",
        "education": [
            {
                "institution": "UC Berkeley",
                "degree": "BA",
                "field_of_study": "Data Science",
                "graduation_year": 2022,
            }
        ],
    }
)


class FakeProvider:
    """An LLMProvider that returns canned text and records what it was asked.

    Structural typing means this never imports or inherits from anything in
    app.providers -- it just has the right shape.
    """

    name = "fake"

    def __init__(self, response: str = VALID_RESPONSE, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def complete_json(self, prompt: str, schema: dict[str, Any]) -> str:
        self.calls.append((prompt, schema))
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch):
    """Install a FakeProvider and hand it back for inspection.

    Patches the name as imported into app.workers.extract rather than
    app.providers.get_provider, because `from ... import get_provider` binds
    the function into this module's namespace at import time -- patching the
    source would leave the already-bound reference untouched.
    """

    def install(fake: FakeProvider) -> FakeProvider:
        monkeypatch.setattr(extract_module, "get_provider", lambda: fake)
        return fake

    return install


class TestEmptyInput:
    """An empty document is not an LLM problem."""

    @pytest.mark.parametrize("raw_text", ["", "   ", "\n\n\t  \n"])
    def test_empty_text_short_circuits(self, provider, raw_text: str) -> None:
        fake = provider(FakeProvider())

        result = extract_profile(raw_text)

        # The assertion that matters is the empty call list: a scanned PDF
        # produces "" from documents.extract_text, and paying for an LLM call
        # per scanned resume is a real cost at any volume.
        assert fake.calls == []
        assert result == ExtractedProfile(
            skills=[], total_years_experience=None, seniority="unknown", education=[]
        )


class TestPromptConstruction:
    def test_prompt_carries_instructions_and_document(self, provider) -> None:
        fake = provider(FakeProvider())

        extract_profile("Brandon Chun -- Berkeley")

        prompt, _ = fake.calls[0]
        assert "Brandon Chun -- Berkeley" in prompt
        # The anti-hallucination rule is the load-bearing part of the prompt.
        assert "Never invent" in prompt

    def test_schema_sent_is_the_extraction_model(self, provider) -> None:
        fake = provider(FakeProvider())

        extract_profile("some resume text")

        _, schema = fake.calls[0]
        assert schema == ExtractedProfile.model_json_schema()

    @staticmethod
    def _document_section(prompt: str) -> str:
        """Everything after the resume marker.

        Measured this way rather than by counting a character across the whole
        prompt, because the instructions contain their own letters -- an
        earlier version of this test was off by the eight "x"s in the word
        "extract" and its neighbours.
        """
        _, _, document = prompt.partition("=== RESUME TEXT ===\n")
        return document

    def test_oversized_resume_is_truncated(self, provider) -> None:
        """40k characters is ~15 pages -- past this a document is malformed.

        Truncation bounds both cost and context usage. The test pins that the
        cap applies to the document, not to the whole prompt: instructions
        must survive intact or the model loses its rules.
        """
        fake = provider(FakeProvider())

        extract_profile("x" * (MAX_RESUME_CHARS + 5_000))

        prompt, _ = fake.calls[0]
        assert len(self._document_section(prompt)) == MAX_RESUME_CHARS
        assert "Never invent" in prompt

    def test_resume_at_the_cap_is_not_truncated(self, provider) -> None:
        fake = provider(FakeProvider())

        extract_profile("x" * MAX_RESUME_CHARS)

        prompt, _ = fake.calls[0]
        assert len(self._document_section(prompt)) == MAX_RESUME_CHARS


class TestResponseValidation:
    def test_valid_response_is_parsed(self, provider) -> None:
        provider(FakeProvider())

        result = extract_profile("resume text")

        assert result.seniority == "mid"
        assert result.total_years_experience == 4.0
        assert result.education == [
            EducationItem(
                institution="UC Berkeley",
                degree="BA",
                field_of_study="Data Science",
                graduation_year=2022,
            )
        ]

    @pytest.mark.parametrize(
        ("response", "case"),
        [
            ("not json at all", "garbage"),
            ("", "empty string"),
            ('{"skills": [', "truncated mid-object"),
            ('{"skills": []}', "missing required fields"),
            ('{"skills": [], "seniority": "wizard", "education": []}', "bad enum"),
            (
                '{"skills": [{"name": "Go", "years": "four"}],'
                ' "seniority": "mid", "education": []}',
                "wrong scalar type",
            ),
        ],
    )
    def test_unusable_response_is_transient(
        self, provider, response: str, case: str
    ) -> None:
        """Schema-constrained decoding makes this rare, not impossible.

        Transient is the right classification: the same prompt on a fresh call
        usually succeeds, which is what retries are for. Calling it permanent
        would dead-letter a resume over one bad roll.
        """
        provider(FakeProvider(response=response))

        with pytest.raises(LLMTransientError):
            extract_profile("resume text")

    def test_transient_error_names_the_provider(self, provider) -> None:
        # The message lands in logs and, in M5, in jobs.last_error. Which
        # provider produced it is the first thing anyone debugging wants.
        provider(FakeProvider(response="not json"))

        with pytest.raises(LLMTransientError, match="fake"):
            extract_profile("resume text")


class TestProviderErrorsPropagate:
    """Classification made by the provider must survive this layer.

    extract_profile re-raises rather than re-wrapping, and that matters: M5
    decides whether to retry from the exception type. Flattening a permanent
    error into a transient one means retrying a bad API key three times.
    """

    def test_permanent_error_stays_permanent(self, provider) -> None:
        provider(FakeProvider(error=LLMPermanentError("bad api key")))

        with pytest.raises(LLMPermanentError):
            extract_profile("resume text")

    def test_transient_error_stays_transient(self, provider) -> None:
        provider(FakeProvider(error=LLMTransientError("rate limited")))

        with pytest.raises(LLMTransientError):
            extract_profile("resume text")


class TestSkillNormalization:
    def test_skill_names_are_canonicalized(self, provider) -> None:
        provider(FakeProvider())

        result = extract_profile("resume text")

        assert [skill.name for skill in result.skills] == ["JavaScript", "Rust"]

    def test_evidence_and_years_survive_normalization(self, provider) -> None:
        """Only the name changes. Evidence is what makes output auditable.

        model_copy(update=...) rather than constructing a new SkillItem is
        what preserves the rest of the fields; this test is what would catch
        a rewrite that dropped them.
        """
        provider(FakeProvider())

        javascript = extract_profile("resume text").skills[0]

        assert javascript.years == 4.0
        assert javascript.evidence == "Built a React frontend in JS"

    def test_duplicate_skills_collapse_keeping_the_first(self, provider) -> None:
        response = json.dumps(
            {
                "skills": [
                    {"name": "JS", "years": 5.0, "evidence": "five years of JS"},
                    {"name": "JavaScript", "years": 1.0, "evidence": "one year"},
                ],
                "total_years_experience": 5.0,
                "seniority": "senior",
                "education": [],
            }
        )
        provider(FakeProvider(response=response))

        skills = extract_profile("resume text").skills

        assert len(skills) == 1
        assert skills[0].name == "JavaScript"
        assert skills[0].years == 5.0

    def test_blank_skill_names_are_dropped(self, provider) -> None:
        response = json.dumps(
            {
                "skills": [
                    {"name": "   ", "years": None, "evidence": None},
                    {"name": "Go", "years": None, "evidence": "wrote a service in Go"},
                ],
                "total_years_experience": None,
                "seniority": "unknown",
                "education": [],
            }
        )
        provider(FakeProvider(response=response))

        assert [s.name for s in extract_profile("resume text").skills] == ["Go"]

    def test_normalization_is_what_gets_persisted(self, provider) -> None:
        """Documents actual behaviour, which differs from the code comment.

        _normalize's docstring says `extracted` in JSONB "preserves what the
        LLM actually said, while the normalized form is what scoring uses".
        There is only one object: routers/profiles.py dumps the return value
        of extract_profile, so the canonical names are what reach the column
        and the original spelling is not stored anywhere.

        That may well be the right trade -- evidence spans still quote the
        resume verbatim, so the output stays auditable. But the comment
        describes a two-form design that does not exist, and this test pins
        which of the two is real.
        """
        provider(FakeProvider())

        dumped = extract_profile("resume text").model_dump(mode="json")

        assert dumped["skills"][0]["name"] == "JavaScript"
        assert "JS" not in [skill["name"] for skill in dumped["skills"]]


class TestExtractionSchema:
    """The model is the LLM contract; these are the parts it must enforce."""

    def test_optional_skill_fields_default_to_none(self) -> None:
        skill = SkillItem(name="Go")

        assert skill.years is None
        assert skill.evidence is None

    def test_seniority_is_constrained(self) -> None:
        with pytest.raises(ValueError):
            ExtractedProfile(skills=[], seniority="wizard", education=[])

    def test_unknown_is_a_valid_seniority(self) -> None:
        # The prompt instructs the model to prefer "unknown" over guessing, so
        # the schema has to accept it or that instruction is unfollowable.
        profile = ExtractedProfile(skills=[], seniority="unknown", education=[])

        assert profile.seniority == "unknown"

    def test_field_descriptions_reach_the_json_schema(self) -> None:
        """The descriptions are prompt engineering, not documentation.

        They are compiled into the JSON Schema the model is constrained by,
        so stripping them as "just comments" would quietly change extraction
        quality -- particularly the instruction to quote evidence verbatim.
        """
        schema = ExtractedProfile.model_json_schema()
        evidence = schema["$defs"]["SkillItem"]["properties"]["evidence"]

        assert "verbatim" in evidence["description"]
