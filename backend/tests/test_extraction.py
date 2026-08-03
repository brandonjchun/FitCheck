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
from pydantic import ValidationError

from app.extraction import (
    PROFILE_EXTRACTION_VERSION,
    EducationItem,
    ExtractedProfile,
    SkillItem,
)
from app.models import Profile
from app.providers import LLMPermanentError, LLMTransientError
from app.workers import extract as extract_module
from app.workers.extract import EmptyDocumentError, MAX_RESUME_CHARS, extract_profile

VALID_RESPONSE = json.dumps(
    {
        "skills": [
            {
                "name": "JS",
                "years": 4.0,
                "evidence": "Built a React frontend in JS",
                "source": "experience",
            },
            {
                "name": "Rust",
                "years": None,
                "evidence": "Hobby projects in Rust",
                "source": "project",
            },
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
    def test_empty_text_raises_without_calling_the_provider(
        self, provider, raw_text: str
    ) -> None:
        fake = provider(FakeProvider())

        with pytest.raises(EmptyDocumentError):
            extract_profile(raw_text)

        # The empty call list is half the point: a scanned PDF produces "" from
        # documents.extract_text, and paying for an LLM call per scanned resume
        # is a real cost at any volume.
        assert fake.calls == []

    def test_empty_document_error_is_permanent(self) -> None:
        """Retrying will not put a text layer into a scanned PDF.

        Subclassing LLMPermanentError rather than raising it directly means
        existing handlers need no change, while the type still names the real
        cause instead of implicating the provider.
        """
        assert issubclass(EmptyDocumentError, LLMPermanentError)

    def test_raising_is_what_keeps_extraction_ok_meaningful(self, provider) -> None:
        """The reason this raises instead of returning an empty profile.

        Returning ExtractedProfile(skills=[], ...) meant the caller persisted
        a populated-looking blob for a document the model never saw, and
        `extraction_ok` -- whose whole job is separating "found no skills"
        from "never ran" -- reported true for the second case. Raising leaves
        `extracted` null, so the flag stays honest.
        """
        provider(FakeProvider())
        profile = Profile(original_filename="scan.pdf", raw_text="")

        with pytest.raises(EmptyDocumentError):
            profile.extracted = extract_profile(profile.raw_text).model_dump()

        assert profile.extracted is None
        assert profile.extraction_ok is False


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


class TestRawOutputIsPreserved:
    """extract_profile returns what the model said, uncanonicalized.

    Normalization moved to read time (models.Profile.skills). The earlier
    design canonicalized here, so the original spellings never reached
    storage -- which made every future addition to the alias map a full LLM
    re-run, because the raw names it would need to re-map were gone.
    """

    def test_skill_names_are_not_canonicalized(self, provider) -> None:
        provider(FakeProvider())

        result = extract_profile("resume text")

        assert [skill.name for skill in result.skills] == ["JS", "Rust"]

    def test_evidence_and_years_are_untouched(self, provider) -> None:
        """Evidence is what makes the output auditable -- it quotes verbatim."""
        provider(FakeProvider())

        js = extract_profile("resume text").skills[0]

        assert js.years == 4.0
        assert js.evidence == "Built a React frontend in JS"

    def test_duplicates_are_not_collapsed_before_storage(self, provider) -> None:
        """Both spellings survive, so a later alias change can still see them.

        Collapsing here discarded the losing entry's evidence span
        permanently. Collapsing on read keeps both in the column and picks a
        winner per query, which is recoverable.
        """
        response = json.dumps(
            {
                "skills": [
                    {
                        "name": "JS",
                        "years": 5.0,
                        "evidence": "five years of JS",
                        "source": "experience",
                    },
                    {
                        "name": "JavaScript",
                        "years": 1.0,
                        "evidence": "one year",
                        "source": "skills_list",
                    },
                ],
                "total_years_experience": 5.0,
                "seniority": "senior",
                "education": [],
            }
        )
        provider(FakeProvider(response=response))

        skills = extract_profile("resume text").skills

        assert [s.name for s in skills] == ["JS", "JavaScript"]

    def test_raw_names_are_what_gets_persisted(self, provider) -> None:
        """The inverse of the assertion this test used to make.

        routers/profiles.py dumps this value straight into profiles.extracted,
        so what the model actually wrote is what lands in the column.
        """
        provider(FakeProvider())

        dumped = extract_profile("resume text").model_dump(mode="json")

        assert dumped["skills"][0]["name"] == "JS"

    def test_stored_blob_reads_back_canonicalized(self, provider) -> None:
        """The round trip that makes storing raw safe.

        Raw goes into the column; the caller still sees canonical names,
        because Profile.skills normalizes on the way out. This is the test
        that would catch someone reinstating normalization on the write side
        and quietly making alias fixes expensive again.
        """
        provider(FakeProvider())
        profile = Profile(original_filename="r.pdf", raw_text="resume text")

        profile.extracted = extract_profile(profile.raw_text).model_dump(mode="json")

        assert profile.extracted["skills"][0]["name"] == "JS"
        assert [skill["name"] for skill in profile.skills] == ["JavaScript", "Rust"]


class TestExtractionVersioning:
    def test_a_fresh_profile_has_no_version(self) -> None:
        profile = Profile(original_filename="r.pdf", raw_text="text")

        assert profile.extraction_version is None
        assert profile.extraction_is_current is False

    def test_current_version_counts_as_current(self) -> None:
        profile = Profile(original_filename="r.pdf", raw_text="text")
        profile.extracted = {"skills": []}
        profile.extraction_version = PROFILE_EXTRACTION_VERSION

        assert profile.extraction_is_current is True

    def test_older_version_is_stale_despite_having_an_extraction(self) -> None:
        """The case the column exists for.

        A content hash cannot see a prompt change, so this profile looks
        complete by every other measure. Only the version says it was built
        by rules that have since been replaced.
        """
        profile = Profile(original_filename="r.pdf", raw_text="text")
        profile.extracted = {"skills": []}
        profile.extraction_version = PROFILE_EXTRACTION_VERSION - 1

        assert profile.extraction_ok is True
        assert profile.extraction_is_current is False


class TestSkillSource:
    """Separating demonstrated skills from claimed ones.

    "Go" in a technologies list is a claim; shipping a service in Go is
    evidence. Weighting them identically is what a resume keyword-stuffer is
    counting on, and telling them apart is most of what a human reviewer
    does. M7 needs the distinction as data.
    """

    def test_source_survives_extraction(self, provider) -> None:
        provider(FakeProvider())

        skills = extract_profile("resume text").skills

        assert [skill.source for skill in skills] == ["experience", "project"]

    def test_source_survives_the_storage_round_trip(self, provider) -> None:
        """It has to reach the column and come back, or M7 cannot weight it.

        normalize_skill_items rewrites `name` and passes everything else
        through; this is what would catch a rewrite that rebuilt the dict
        from known keys and silently dropped this one.
        """
        provider(FakeProvider())
        profile = Profile(original_filename="r.pdf", raw_text="resume text")

        profile.extracted = extract_profile(profile.raw_text).model_dump(mode="json")

        assert [skill["source"] for skill in profile.skills] == [
            "experience",
            "project",
        ]

    def test_stronger_source_wins_when_a_skill_appears_twice(self, provider) -> None:
        """Dedupe keeps the first occurrence, and the prompt orders by strength.

        A skill listed in both a job bullet and a technologies section must
        not read as `skills_list` -- that would discount the very evidence
        that makes it credible. The prompt asks for the strongest placement;
        this pins that read-time dedupe doesn't then undo it.
        """
        provider(FakeProvider())
        profile = Profile(original_filename="r.pdf", raw_text="resume text")

        # "JS" (experience) and "JavaScript" (skills_list) collapse to one.
        profile.extracted = {
            "skills": [
                {"name": "JS", "years": 5.0, "evidence": "shipped it", "source": "experience"},
                {"name": "JavaScript", "years": None, "evidence": None, "source": "skills_list"},
            ]
        }

        assert [(s["name"], s["source"]) for s in profile.skills] == [
            ("JavaScript", "experience")
        ]

    def test_pre_v3_profiles_read_back_without_a_source(self) -> None:
        """Profiles extracted before version 3 have no `source` key at all.

        The API schema makes the field optional for exactly this reason. A
        required field would turn every older profile into a 500 on read
        rather than a row that simply predates the data.
        """
        profile = Profile(original_filename="r.pdf", raw_text="text")
        profile.extracted = {"skills": [{"name": "js", "years": None, "evidence": None}]}

        skill = profile.skills[0]

        assert skill["name"] == "JavaScript"
        assert skill.get("source") is None


class TestPromptRules:
    def test_source_instruction_gives_a_precedence_order(self) -> None:
        """The rule that matters most, because most skills appear twice.

        A technologies section normally repeats what the bullets already
        demonstrated. Without an explicit precedence the model picks
        arbitrarily, and roughly half the corpus lands in the weakest bucket
        despite having real evidence behind it.
        """
        instructions = extract_module.SYSTEM_INSTRUCTIONS

        assert "experience > project > education > skills_list" in instructions
        assert 'Never answer "skills_list" for a skill that also appears' in instructions

    def test_years_instruction_asks_for_inference_from_dates(self) -> None:
        """Version 1 returned null for `years` on every skill measured.

        The instruction to leave it null unless the resume "supports a number"
        was read as "unless it states one outright". Version 2 asks for the
        inference explicitly, because the partial-match bucket at M7 -- has
        the skill, insufficient years -- is defined entirely by this field and
        collapses to a two-way breakdown without it.
        """
        assert "Infer it from the date range" in extract_module.SYSTEM_INSTRUCTIONS


class TestExtractionSchema:
    """The model is the LLM contract; these are the parts it must enforce."""

    def test_optional_skill_fields_default_to_none(self) -> None:
        skill = SkillItem(name="Go", source="skills_list")

        assert skill.years is None
        assert skill.evidence is None

    def test_source_is_required(self) -> None:
        """No default, on purpose.

        A Pydantic default drops the field from the schema's `required` list,
        and the model then treats it as optional -- which is precisely how
        version 1 produced `years: null` on 29 of 29 skills. "unknown" is the
        escape hatch instead, so the model must still make a call.
        """
        with pytest.raises(ValidationError):
            SkillItem(name="Go")

    def test_source_is_constrained(self) -> None:
        with pytest.raises(ValidationError):
            SkillItem(name="Go", source="linkedin_endorsement")

    def test_source_is_required_in_the_json_schema(self) -> None:
        """What the model is actually constrained by.

        Asserting on the Pydantic class alone would pass even if the field
        never reached the schema sent to the provider, which is the only
        place the constraint has any effect on extraction.
        """
        schema = ExtractedProfile.model_json_schema()

        assert "source" in schema["$defs"]["SkillItem"]["required"]

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
