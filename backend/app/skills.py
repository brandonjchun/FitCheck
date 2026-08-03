"""Canonical skill names.

The spec calls this "unglamorous and load-bearing" and it is right. Every
scoring number in M6 depends on it: if a resume says "JS" and a job posting
says "JavaScript", an un-normalized overlap check scores that as a miss, and
the candidate is silently penalized for a synonym.

This module is pure -- no LLM, no database, no network. It is the one piece
of M2 you can build and test today.
"""

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
}


def normalize_skill(name: str) -> str:
    """Return the canonical name for a skill.

    Lookup is case- and whitespace-insensitive; the returned value keeps the
    map's canonical casing, because this string becomes a comparison key in
    M6 and "javascript" != "JavaScript".

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
    breakdown in M7 exists to make visible.
    """
    cleaned = name.strip()
    return SKILL_ALIASES.get(cleaned.lower(), cleaned)


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
