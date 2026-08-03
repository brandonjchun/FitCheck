import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { errorMessage, matchApi, type Match, type MatchSkill } from "../api/client";
import "./MatchFeed.css";

/* Scoring runs asynchronously after a fetch succeeds, so a match appears some
 * seconds after its job does. There is no `is_complete` to stop on -- a
 * posting can be re-fetched and re-scored at any time -- so this polls at a
 * relaxed interval rather than trying to guess when it is finished. The query
 * is capped at 25 rows and served by `matches_feed_idx`, so it is cheap. */
const MATCH_POLL_MS = 5000;

/** Scores are cosine-blended into [0, 1]. Shown as a percentage. */
function pct(score: number): string {
  return `${Math.round(score * 100)}%`;
}

/**
 * Coarse band for colouring a score.
 *
 * Three bands, not a gradient. A continuous colour ramp implies the number is
 * precise enough to distinguish 61% from 64%, which it is not -- the blend
 * weights are an admitted judgment call. Bands say "strong / worth a look /
 * probably not" and stop there.
 */
function band(score: number): "high" | "mid" | "low" {
  if (score >= 0.7) return "high";
  if (score >= 0.45) return "mid";
  return "low";
}

function SkillRow({ skill }: { skill: MatchSkill }) {
  const years =
    skill.required_years != null
      ? `${skill.candidate_years ?? 0} of ${skill.required_years} yrs`
      : skill.candidate_years != null
        ? `${skill.candidate_years} yrs`
        : null;

  return (
    <li className={`mf-skill bucket-${skill.bucket}`}>
      <span className="mf-skill-dot" aria-hidden="true" />

      <span className="mf-skill-name">
        {skill.name}
        {skill.necessity === "required" && (
          <span className="mf-req" title="The posting lists this as required">
            required
          </span>
        )}
      </span>

      {years && <span className="mf-skill-years">{years}</span>}

      {/* The posting's own words. A score that cannot point at what it reacted
        * to is unfalsifiable, which is the whole argument for storing this. */}
      {skill.evidence && <q className="mf-skill-evidence">{skill.evidence}</q>}
    </li>
  );
}

function MatchCard({ match, rank }: { match: Match; rank: number }) {
  const [open, setOpen] = useState(false);

  const counts = match.counts;
  const title = match.posting_title ?? "Untitled posting";
  const heading = match.posting_company ? `${title} · ${match.posting_company}` : title;

  /* Required gaps first, then partials, then the rest. The ordering is the
   * advice: a missing required skill is the thing that decides, so it should
   * not be buried under eight satisfied nice-to-haves. */
  const ordered = [...match.skills].sort((a, b) => {
    const weight = (s: MatchSkill) =>
      s.bucket === "missing" && s.necessity === "required"
        ? 0
        : s.bucket === "missing"
          ? 1
          : s.bucket === "partial"
            ? 2
            : 3;
    return weight(a) - weight(b);
  });

  return (
    <li className="mf-card">
      <div className="mf-head">
        <div className={`mf-score band-${band(match.final_score)}`}>
          <span className="mf-score-value">{pct(match.final_score)}</span>
          <span className="mf-score-label">fit</span>
        </div>

        <div className="mf-title-block">
          <span className="mf-rank">#{rank}</span>
          {match.posting_url ? (
            <a
              href={match.posting_url}
              target="_blank"
              rel="noreferrer noopener"
              className="mf-title"
            >
              {heading}
            </a>
          ) : (
            <span className="mf-title">{heading}</span>
          )}

          {/* Both halves always visible. A single blended number is
            * unfalsifiable; showing what it was made of is what makes it
            * inspectable. */}
          <p className="mf-subscores">
            semantic <strong>{pct(match.semantic_score)}</strong> · skills{" "}
            <strong>{pct(match.skill_score)}</strong>
            {match.weights?.semantic != null && match.weights?.skill != null && (
              <span className="mf-weights">
                {" "}
                (weighted {match.weights.semantic} / {match.weights.skill})
              </span>
            )}
          </p>
        </div>
      </div>

      <div className="mf-counts">
        <span className="chip chip-matched">{counts.matched} matched</span>
        <span className="chip chip-partial">{counts.partial} partial</span>
        <span className="chip chip-missing">{counts.missing} missing</span>
        {counts.missing_required > 0 && (
          <span className="chip chip-blocker">
            {counts.missing_required} required missing
          </span>
        )}
      </div>

      {/* Surfaced rather than swallowed. Without it a semantic-only score
        * looks like a normal one that happened to find no skills. */}
      {match.extraction_failed && (
        <p className="mf-warn">
          The posting could not be parsed into structured requirements, so this
          is a semantic score only — the skill half found nothing to compare.
        </p>
      )}

      {match.skills.length > 0 && (
        <>
          <button
            className="btn btn-ghost btn-sm mf-toggle"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
          >
            {open ? "Hide breakdown" : `Why this score (${match.skills.length})`}
          </button>

          {open && (
            <ul className="mf-skills">
              {ordered.map((skill, i) => (
                <SkillRow key={`${skill.name}-${i}`} skill={skill} />
              ))}
            </ul>
          )}
        </>
      )}
    </li>
  );
}

export function MatchFeed({ profileId }: { profileId: number }) {
  const matches = useQuery({
    queryKey: ["matches", profileId],
    queryFn: () => matchApi.list(profileId, 25),
    refetchInterval: MATCH_POLL_MS,
  });

  const rows = matches.data ?? [];

  /* Scores from different scorer generations are not comparable, and a feed
   * sorted across two of them is silently wrong -- position 3 beats position 4
   * because it was scored under different rules, not because it fits better.
   * The API exposes the version precisely so a client can say so. */
  const versions = new Set(rows.map((m) => m.scorer_version));

  return (
    <div className="panel card">
      <div className="panel-head">
        <div>
          <h3>Matches</h3>
          <p className="panel-meta">
            Every posting you have submitted, scored against this resume, best
            first.
          </p>
        </div>
      </div>

      {matches.isError && (
        <p className="form-error" role="alert">
          {errorMessage(matches.error, "Could not load matches.")}
        </p>
      )}

      {versions.size > 1 && (
        <p className="mf-warn">
          These matches were produced by {versions.size} different scorer
          versions and are not directly comparable. Re-scoring aligns them.
        </p>
      )}

      {rows.length > 0 ? (
        <ol className="mf-list">
          {rows.map((match, i) => (
            <MatchCard key={match.id} match={match} rank={i + 1} />
          ))}
        </ol>
      ) : (
        <p className="panel-empty">
          {matches.isLoading
            ? "Loading…"
            : "No scores yet. Submit a posting URL above — once it is fetched, it is scored against this resume automatically and appears here."}
        </p>
      )}
    </div>
  );
}
