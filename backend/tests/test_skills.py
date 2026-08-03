"""Tests for app.skills -- canonical skill names.

The spec calls this module "unglamorous and load-bearing", and the load it
bears is M6's scoring. Every one of these tests is really an assertion about
a future match percentage: if "JS" and "JavaScript" stop collapsing to one
token, overlap scoring silently reports a miss on a skill the candidate has.
"""

import pytest

from app.skills import (
    SKILL_ALIASES,
    normalize_skill,
    normalize_skill_items,
    normalize_skills,
)


class TestNormalizeSkill:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("js", "JavaScript"),
            ("JS", "JavaScript"),
            ("Js", "JavaScript"),
            ("  JS  ", "JavaScript"),
            ("ecmascript", "JavaScript"),
            ("PostGres", "PostgreSQL"),
            ("k8s", "Kubernetes"),
            ("amazon web services", "AWS"),
            ("react.js", "React"),
            ("golang", "Go"),
            ("csharp", "C#"),
            (".net", ".NET"),
            ("restful apis", "REST"),
            ("cicd", "CI/CD"),
        ],
    )
    def test_aliases_resolve_to_canonical_form(self, raw: str, expected: str) -> None:
        assert normalize_skill(raw) == expected

    def test_unknown_skill_is_kept_not_dropped(self) -> None:
        """The alias map will never be complete.

        Dropping what it does not recognise would lose real skills -- most of
        a resume's skills are not in a 30-entry map. Passing them through
        means an unrecognised skill still matches an identically-written
        requirement, which is the common case.
        """
        assert normalize_skill("Rust") == "Rust"
        assert normalize_skill("  Elixir  ") == "Elixir"

    def test_unknown_skill_keeps_its_original_casing(self) -> None:
        # Lowercasing everything would make the comparison key "rust" and the
        # display string "rust" -- correct for matching, wrong for the UI.
        assert normalize_skill("GDScript") == "GDScript"

    def test_blank_input_returns_empty_string(self) -> None:
        assert normalize_skill("") == ""
        assert normalize_skill("   ") == ""

    def test_canonical_names_are_stable_under_reapplication(self) -> None:
        """normalize(normalize(x)) == normalize(x) for every alias.

        Idempotence is not decorative here. Skills get normalized on
        extraction and will be normalized again when compared in M6, so a map
        where a canonical value is itself an alias for something else would
        drift on the second pass.
        """
        for alias in SKILL_ALIASES:
            once = normalize_skill(alias)
            assert normalize_skill(once) == once

    def test_parent_skills_are_not_collapsed(self) -> None:
        """"React Native" is not "React", and "AWS Lambda" is not "AWS".

        Merging them would let a candidate match a requirement they do not
        meet -- the exact false positive M7's breakdown exists to expose.
        """
        assert normalize_skill("React Native") == "React Native"
        assert normalize_skill("AWS Lambda") == "AWS Lambda"


class TestNormalizeSkills:
    def test_preserves_order(self) -> None:
        # An LLM lists prominent skills first; that ordering is worth keeping
        # for display even though scoring treats the set as unordered.
        assert normalize_skills(["python", "js", "k8s"]) == [
            "Python",
            "JavaScript",
            "Kubernetes",
        ]

    def test_deduplicates_after_normalization(self) -> None:
        """The whole point: three spellings of one skill are one skill.

        Without this, a resume listing both "JS" and "JavaScript" would count
        that skill twice in a weighted overlap and inflate its own score.
        """
        assert normalize_skills(["JS", "JavaScript", "ecmascript"]) == ["JavaScript"]

    def test_first_occurrence_wins(self) -> None:
        assert normalize_skills(["Rust", "js", "Rust"]) == ["Rust", "JavaScript"]

    def test_drops_blanks(self) -> None:
        assert normalize_skills(["", "   ", "js"]) == ["JavaScript"]

    def test_empty_list(self) -> None:
        assert normalize_skills([]) == []


class TestNormalizeSkillItems:
    """The dict-shaped form, applied when reading the extraction blob back.

    This is where canonicalization happens now. Doing it on read rather than
    before the write is what makes an addition to the alias map retroactive:
    every stored profile picks it up on the next request, with no backfill
    and no second LLM call.
    """

    def test_canonicalizes_names(self) -> None:
        items = [{"name": "JS", "years": 4.0, "evidence": "built a frontend in JS"}]

        assert normalize_skill_items(items)[0]["name"] == "JavaScript"

    def test_years_and_evidence_pass_through_untouched(self) -> None:
        """Only `name` is rewritten.

        Evidence quotes the resume verbatim -- that is the property making
        the extraction auditable, and rewriting it would destroy the ability
        to locate the phrase in the source document.
        """
        items = [{"name": "js", "years": 4.0, "evidence": "Built a React frontend"}]

        assert normalize_skill_items(items)[0] == {
            "name": "JavaScript",
            "years": 4.0,
            "evidence": "Built a React frontend",
        }

    def test_duplicates_collapse_to_the_first_occurrence(self) -> None:
        """Order matters, and the first mention is generally better evidenced.

        An LLM lists prominent skills first, so given "JS" with five years and
        "JavaScript" with one, the five-year entry is the one to keep.
        """
        items = [
            {"name": "JS", "years": 5.0, "evidence": "five years of JS"},
            {"name": "JavaScript", "years": 1.0, "evidence": "one year"},
        ]

        result = normalize_skill_items(items)

        assert len(result) == 1
        assert result[0]["years"] == 5.0
        assert result[0]["evidence"] == "five years of JS"

    def test_drops_blank_names(self) -> None:
        items = [
            {"name": "   ", "years": None, "evidence": None},
            {"name": "Go", "years": None, "evidence": "wrote a service in Go"},
        ]

        assert [item["name"] for item in normalize_skill_items(items)] == ["Go"]

    def test_tolerates_a_missing_name_key(self) -> None:
        """Defensive: the blob is whatever the model produced last time.

        A stored extraction predating a schema change may not have the shape
        the current code expects, and a KeyError on read would take down the
        profile endpoint for a row that is merely old.
        """
        assert normalize_skill_items([{"years": 2.0}]) == []

    def test_extra_keys_survive(self) -> None:
        """Forward compatibility with fields added to SkillItem later.

        `source` is a candidate for the next schema version; a normalizer
        that silently dropped unrecognised keys would erase it on read.
        """
        items = [{"name": "go", "years": None, "evidence": None, "source": "project"}]

        assert normalize_skill_items(items)[0]["source"] == "project"

    def test_does_not_mutate_its_input(self) -> None:
        """The caller's list is the ORM's JSONB value.

        Rewriting names in place would mark the attribute dirty and could
        write canonicalized names back to the column on the next flush --
        reintroducing exactly the lossy write this design removed.
        """
        items = [{"name": "js", "years": None, "evidence": None}]

        normalize_skill_items(items)

        assert items[0]["name"] == "js"

    def test_empty_list(self) -> None:
        assert normalize_skill_items([]) == []


class TestAliasMapIntegrity:
    """Constraints on the map itself, not on the function reading it."""

    def test_every_key_is_lowercase(self) -> None:
        # Lookup lowercases its input, so a capitalised key is unreachable.
        unreachable = [key for key in SKILL_ALIASES if key != key.lower()]
        assert unreachable == []

    def test_every_key_is_stripped(self) -> None:
        # Same reason: lookup strips, so a padded key can never match.
        padded = [key for key in SKILL_ALIASES if key != key.strip()]
        assert padded == []

    def test_every_canonical_name_maps_to_itself(self) -> None:
        """Each canonical value must appear as its own key.

        Otherwise normalizing an already-canonical name is a map miss that
        happens to work only because the fallback returns the input unchanged
        -- and it stops working the moment the casing differs.
        """
        missing = [
            canonical
            for canonical in set(SKILL_ALIASES.values())
            if SKILL_ALIASES.get(canonical.lower()) != canonical
        ]
        assert missing == []
