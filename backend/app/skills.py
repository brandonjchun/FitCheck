"""Canonical skill names.

The spec calls this "unglamorous and load-bearing" and it is right. Every
scoring number in M7 depends on it: if a resume says "JS" and a job posting
says "JavaScript", an un-normalized overlap check scores that as a miss, and
the candidate is silently penalized for a synonym.

This module is pure -- no LLM, no database, no network. It is the one piece
of M2 you can build and test today.
"""

import re

# Maps any alias (lowercased) to its canonical name.
#
# Deliberately small. A hand-maintained map that covers the aliases you
# actually see beats a large speculative one you cannot verify, and every
# entry here is a claim that two strings mean the same thing -- which is
# sometimes wrong (see "Java" vs "JavaScript", below).
SKILL_ALIASES: dict[str, str] = {
    "js": "JavaScript",
    "javascript": "JavaScript",
    "ecmascript": "JavaScript",
    "ts": "TypeScript",
    "typescript": "TypeScript",
    "py": "Python",
    "python": "Python",
    "python3": "Python",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "psql": "PostgreSQL",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud Platform",
    "google cloud platform": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "react": "React",
    "reactjs": "React",
    "react.js": "React",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "golang": "Go",
    "go": "Go",
    "c#": "C#",
    "csharp": "C#",
    "dotnet": ".NET",
    ".net": ".NET",
    "rest": "REST",
    "restful apis": "REST",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "graphql": "GraphQL",
    "redis": "Redis",
    "css": "CSS",
    "html": "HTML",
    "sql": "SQL",
}


# Words that describe a skill rather than name one.
#
# "GraphQL", "GraphQL API", and "graphql-api" are one requirement written
# three ways, and a posting that says "Machine Learning Frameworks" is asking
# for machine learning. Stripping these before comparing turns all of those
# into one key.
#
# **This is not the parent-collapsing that `normalize_skill` refuses to do.**
# The distinction is that every word here is a generic noun -- it narrows
# nothing. "Native" in "React Native" and "Lambda" in "AWS Lambda" name
# different technologies, so they stay, and those skills stay distinct. "API"
# in "GraphQL API" names no technology at all.
#
# Kept short for the same reason the alias map is: each entry is a claim that
# a word carries no meaning in a skill name, and a wrong claim silently merges
# two real requirements. Anything ambiguous is left out -- "engineering",
# "design", and "architecture" are all load-bearing in some skill names
# ("Design Systems" is a real frontend discipline) and are deliberately absent.
GENERIC_QUALIFIERS: frozenset[str] = frozenset(
    {
        "api",
        "apis",
        "framework",
        "frameworks",
        "library",
        "libraries",
        "development",
        "programming",
        "experience",
        "skills",
        "knowledge",
        "proficiency",
        "systems",
        "technologies",
        "tooling",
    }
)


# Split on anything that is not alphanumeric, `+`, `#`, or `.`.
#
# Those three survive because they are part of names rather than punctuation
# between them: "C++", "C#", ".NET", and "Node.js" all lose their identity if
# split on them. Everything else -- spaces, hyphens, slashes, parentheses --
# separates tokens, which is what collapses "graphql-api" and "graphql api".
_TOKENS = re.compile(r"[^a-z0-9+#.]+")


def canonical_key(name: str) -> str:
    """A comparison key that ignores case, punctuation, and filler words.

    The thing two spellings of one skill have in common:

        canonical_key("GraphQL")      -> "graphql"
        canonical_key("graphql-api")  -> "graphql"
        canonical_key("GraphQL API")  -> "graphql"
        canonical_key("Redis")        -> "redis"
        canonical_key("React Native") -> "reactnative"   (still not "react")

    Not a display name -- it is lowercase and stripped of separators, so
    `canonical_key("Distributed Systems")` is `"distributed"`, which nobody
    should ever see. Callers that show a name to a user pick a real spelling;
    this only decides which spellings are the same thing.

    If a name is made *entirely* of qualifiers ("APIs", "Frameworks") the
    tokens are kept rather than reduced to the empty string, which would merge
    every such name into one bucket.
    """
    tokens = [token for token in _TOKENS.split(name.strip().lower()) if token]
    if not tokens:
        return ""

    meaningful = [token for token in tokens if token not in GENERIC_QUALIFIERS]
    return "".join(meaningful or tokens)


# The alias map indexed by canonical key, built once at import.
#
# This is what lets one entry cover a family: "graphql" in SKILL_ALIASES also
# catches "GraphQL API", "graphql-api", and "GraphQL APIs", because all four
# reduce to the same key. Without it every spelling would need its own row and
# the map would grow into the thousand-line unverifiable thing this module's
# docstring warns against.
_ALIASES_BY_KEY: dict[str, str] = {
    canonical_key(alias): canonical for alias, canonical in SKILL_ALIASES.items()
}


# Things a posting asks for that are not skills.
#
# Dropped on read rather than filtered at extraction, for the same reason
# aliases are applied on read: it makes the fix retroactive across every
# posting already stored, with no re-extraction and no LLM cost.
#
# **Why this exists at all.** The posting prompt already says to skip these,
# and Gemini obeys it. The local model that extracts the catalog in bulk
# does not, reliably -- a real Spotify sales posting came back with
# `Integrity` and `Work-life balance` as required skills. Those are the same
# failure as the `"Automated test coverage"` case: no resume can ever match
# them, so they sit in `missing_required` permanently and drag every
# candidate's denominator down. The difference is that one was fixable by
# prompting and this one is not, because the instruction is simply not
# followed often enough.
#
# Deliberately small and hand-maintained, on the same terms as the alias map:
# every entry is a claim that a string is never a skill, and a claim that is
# sometimes wrong is worse than a short list. `Communication` and
# `Leadership` are borderline and stay OUT -- they are real, assessable
# things that a resume can evidence, and dropping them would understate
# genuinely people-heavy roles.
NON_SKILLS: frozenset[str] = frozenset(
    {
        "integrity",
        "work-life balance",
        "work life balance",
        "growth mindset",
        "attention to detail",
        "team player",
        "self-starter",
        "hard working",
        "hard-working",
        "passion",
        "passionate",
        "enthusiasm",
        "curiosity",
        "empathy",
        "positive attitude",
        "fast-paced environment",
        "fast paced environment",
        "willingness to learn",
        "eagerness to learn",
        "adaptability",
        "flexibility",
        "professionalism",
        "reliability",
        "accountability",
        "diversity",
        "inclusion",
        "culture fit",
        "equal opportunity",
    }
)


def is_skill(name: str) -> bool:
    """Whether `name` names something a resume could actually evidence.

    False for the qualities in NON_SKILLS. Comparison is on the stripped,
    lowercased form, so "Integrity " and "INTEGRITY" are both caught.
    """
    return name.strip().lower() not in NON_SKILLS


def normalize_skill(name: str) -> str:
    """Return the canonical name for a skill.

    Lookup is case- and whitespace-insensitive; the returned value keeps the
    map's canonical casing, because this string becomes a comparison key in
    M7 and "javascript" != "JavaScript".

    An unrecognised skill is returned stripped but otherwise unchanged rather
    than dropped -- the alias map will never be complete, and silently
    discarding unknown skills would lose real ones.

        normalize_skill("  JS ")    -> "JavaScript"
        normalize_skill("PostGres") -> "PostgreSQL"
        normalize_skill(" Rust ")   -> "Rust"
        normalize_skill("   ")      -> ""

    Deliberately does NOT collapse a skill into its parent ("React Native"
    stays distinct from "React", "AWS Lambda" from "AWS"). Those are
    different competencies, and merging them would let a candidate match a
    requirement they do not meet -- exactly the false positive the skill
    breakdown in M10 exists to make visible.

    Lookup is tried twice. The exact lowercase form first, so every entry the
    map has always wins and this stays backwards compatible; then the
    canonical key, so one entry covers a family --  "graphql" also answers for
    "GraphQL API", "graphql-api", and "GraphQL APIs".
    """
    cleaned = name.strip()
    if not cleaned:
        return ""

    direct = SKILL_ALIASES.get(cleaned.lower())
    if direct is not None:
        return direct

    return _ALIASES_BY_KEY.get(canonical_key(cleaned), cleaned)


def normalize_skills(names: list[str]) -> list[str]:
    """Normalize a list of skills, dropping blanks and de-duplicating.

    Order is preserved: an LLM tends to list the most prominent skills first,
    and that ordering is worth keeping for display even though scoring treats
    the set as unordered.
    """
    seen: set[str] = set()
    result: list[str] = []

    for name in names:
        canonical = normalize_skill(name)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        result.append(canonical)

    return result


def normalize_skill_items(items: list[dict]) -> list[dict]:
    """Canonicalize stored skill objects, dropping blanks and duplicates.

    The dict-shaped counterpart to normalize_skills, for the extraction blob
    as it comes back out of JSONB. Only `name` is rewritten -- `years` and
    `evidence` are carried through untouched, because evidence quotes the
    resume verbatim and rewriting it would destroy the property that makes
    the extraction auditable.

    Applied on **read** rather than on write. Storing the canonical form and
    discarding the original would mean every future addition to the alias map
    requires re-running the LLM over every stored profile to take effect,
    since the raw names it would need to re-map are gone. Normalizing here
    instead makes an alias fix retroactive for free.

    Duplicates collapse to the first occurrence, which is why order matters:
    an LLM lists the most prominent skills first, so the first mention is
    generally the better-evidenced one. Given "JS" with five years and
    "JavaScript" with one, the five-year entry survives.
    """
    seen: set[str] = set()
    result: list[dict] = []

    for item in items:
        raw = item.get("name", "")
        if not is_skill(raw):
            # A quality rather than a skill. Counting it would put a
            # requirement in the denominator that no candidate can ever
            # satisfy.
            continue
        canonical = normalize_skill(raw)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        result.append({**item, "name": canonical})

    return result
