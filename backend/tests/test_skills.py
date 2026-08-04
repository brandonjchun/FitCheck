"""Tests for app.skills -- canonical skill names.

The spec calls this module "unglamorous and load-bearing", and the load it
bears is M6's scoring. Every one of these tests is really an assertion about
a future match percentage: if "JS" and "JavaScript" stop collapsing to one
token, overlap scoring silently reports a miss on a skill the candidate has.
"""

import pytest

from app.skills import (
    GENERIC_QUALIFIERS,
    SKILL_ALIASES,
    canonical_key,
    normalize_skill,
    normalize_skill_items,
    normalize_skills,
)


class TestCanonicalKey:
    """The key that decides which spellings are the same requirement.

    Written against names taken from the real catalog. Before this existed,
    a 50-match feed produced five separate GraphQL rows -- `graphql`,
    `GraphQL`, `graphql api`, `GraphQL API`, `graphql-api` -- which filled
    five of the twelve slots on the insights page and hid the fact that
    GraphQL was by a distance the most common blocking requirement.
    """

    @pytest.mark.parametrize(
        "variants",
        [
            ["GraphQL", "graphql", "graphql-api", "graphql api", "GraphQL API", "GraphQL APIs"],
            ["Redis", "redis", "REDIS"],
            ["Machine Learning", "machine learning", "Machine Learning Frameworks"],
            ["Distributed Systems", "Distributed systems"],
            ["Cloud Infrastructure", "Cloud infrastructure"],
        ],
    )
    def test_variants_of_one_skill_share_a_key(self, variants: list[str]) -> None:
        keys = {canonical_key(name) for name in variants}
        assert len(keys) == 1, f"expected one key, got {keys}"

    @pytest.mark.parametrize(
        "child, parent",
        [
            ("React Native", "React"),
            ("AWS Lambda", "AWS"),
            ("Google BigQuery", "Google"),
        ],
    )
    def test_a_skill_is_not_collapsed_into_its_parent(self, child: str, parent: str) -> None:
        """The line this module already drew, held.

        Stripping filler words is not the same as merging a sub-technology
        into its parent. "Native" and "Lambda" name technologies; "API" and
        "Frameworks" do not. Collapsing the former would let a candidate match
        a requirement they do not meet -- the false positive the M10 breakdown
        exists to make visible.
        """
        assert canonical_key(child) != canonical_key(parent)

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("C++", "c++"),
            ("C#", "c#"),
            (".NET", ".net"),
            ("Node.js", "node.js"),
            ("CI/CD", "cicd"),
        ],
    )
    def test_punctuation_that_is_part_of_a_name_survives(self, name, expected) -> None:
        """`+`, `#`, and `.` are letters here, not separators. Splitting on
        them would reduce "C++" to "c" and merge it with C."""
        assert canonical_key(name) == expected

    def test_cplusplus_does_not_become_c(self) -> None:
        assert canonical_key("C++") != canonical_key("C")

    def test_a_name_made_only_of_filler_keeps_its_tokens(self) -> None:
        """Otherwise every such name reduces to the empty string and they all
        merge into one meaningless bucket."""
        assert canonical_key("APIs") == "apis"
        assert canonical_key("Frameworks") == "frameworks"
        assert canonical_key("APIs") != canonical_key("Frameworks")

    def test_blank_input_is_empty(self) -> None:
        assert canonical_key("   ") == ""

    def test_is_idempotent(self) -> None:
        """A key fed back in has to survive, or callers cannot key a dict on
        it and then look up by the same function."""
        for name in ["GraphQL API", "Redis", "C++", "Node.js"]:
            assert canonical_key(canonical_key(name)) == canonical_key(name)

    def test_qualifiers_are_lowercase_and_single_words(self) -> None:
        """The set is matched against already-lowercased tokens, so an entry
        with a capital or a space could never fire."""
        for word in GENERIC_QUALIFIERS:
            assert word == word.lower().strip()
            assert " " not in word


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

    @pytest.mark.parametrize(
        "raw",
        ["graphql", "GraphQL", "graphql-api", "graphql api", "GraphQL API", "GraphQL APIs"],
    )
    def test_one_alias_entry_covers_a_whole_family(self, raw: str) -> None:
        """`graphql` is a single row in SKILL_ALIASES and answers for all six.

        This is what keeps the map short. Enumerating every spelling is how it
        becomes the thousand-line unverifiable thing the module docstring
        warns against.
        """
        assert normalize_skill(raw) == "GraphQL"

    def test_an_unmapped_skill_still_keeps_its_own_spelling(self) -> None:
        """The key is for comparison, not display. An unrecognised name is
        returned as written -- a user must never be shown "distributed"."""
        assert normalize_skill("Distributed Systems") == "Distributed Systems"


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
