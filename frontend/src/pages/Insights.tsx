import { Navigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { errorMessage, insightsApi, type SkillGap } from "../api/client";
import { useMe } from "../hooks/useAuth";
import "./Insights.css";

/* Aggregated over stored breakdowns, so it changes only when something is
 * re-scored. Polling it would be asking the database to recount the same
 * JSONB every few seconds to produce the same answer. */
const STALE_MS = 5 * 60_000;

/**
 * How much of a barrier one gap is.
 *
 * Three bands rather than a continuous bar, for the same reason MatchFeed
 * bands its scores: the underlying counts are small integers over a modest
 * sample, and a smooth gradient would imply a precision that 3-of-11 does not
 * have.
 */
function severity(gap: SkillGap, analyzed: number): "high" | "mid" | "low" {
  if (analyzed === 0) return "low";
  const rate = gap.blocking / analyzed;
  if (rate >= 0.4) return "high";
  if (rate >= 0.15) return "mid";
  return "low";
}

function GapRow({ gap, analyzed }: { gap: SkillGap; analyzed: number }) {
  const total = gap.missing + gap.partial + gap.matched;
  const pct = (n: number) => (total ? (n / total) * 100 : 0);

  return (
    <li className={`gap-row sev-${severity(gap, analyzed)}`}>
      <div className="gap-head">
        <span className="gap-name">{gap.name}</span>
        {gap.blocking > 0 && (
          <span className="gap-blocking">
            blocks {gap.blocking} of {analyzed}
          </span>
        )}
      </div>

      {/* One bar, three segments. Showing matched alongside the gaps is what
        * keeps this honest: a skill you already satisfy in half the postings
        * is a different problem from one you satisfy in none, and a
        * missing-only chart cannot tell those apart. */}
      <div
        className="gap-bar"
        role="img"
        aria-label={`${gap.name}: missing in ${gap.missing}, partial in ${gap.partial}, matched in ${gap.matched} postings`}
      >
        <span className="seg seg-missing" style={{ width: `${pct(gap.missing)}%` }} />
        <span className="seg seg-partial" style={{ width: `${pct(gap.partial)}%` }} />
        <span className="seg seg-matched" style={{ width: `${pct(gap.matched)}%` }} />
      </div>

      <div className="gap-legend">
        <span>{gap.missing} missing</span>
        <span>{gap.partial} partial</span>
        <span>{gap.matched} matched</span>
      </div>
    </li>
  );
}

export function Insights() {
  const { data: user, isLoading: loadingUser } = useMe();

  const report = useQuery({
    queryKey: ["skill-gaps"],
    queryFn: () => insightsApi.skillGaps(),
    enabled: Boolean(user),
    staleTime: STALE_MS,
  });

  if (loadingUser) return null;
  if (!user) return <Navigate to="/signin" replace />;

  const gaps = report.data?.gaps ?? [];
  const analyzed = report.data?.matches_analyzed ?? 0;

  return (
    <main id="main" className="page-shell insights">
      <div className="container">
      <header className="page-head">
        <h1>Skill gaps</h1>
        <p className="page-sub">
          Every requirement across your scored postings, ranked by how often it
          is one you do not meet. Aggregated over{" "}
          <strong>{analyzed}</strong> {analyzed === 1 ? "match" : "matches"} from
          all of your resumes.
        </p>
      </header>

      {report.isError && (
        <p className="form-error" role="alert">
          {errorMessage(report.error, "Could not load your skill gaps.")}
        </p>
      )}

      {gaps.length > 0 ? (
        <>
          {/* Stated rather than left to be inferred from the ordering. The
            * ranking is by blocking count, and a reader who assumes it is by
            * raw frequency would draw the wrong conclusion from row two. */}
          <p className="insights-note">
            Ranked by how many postings list the skill as <em>required</em> and
            you do not have it. A missing nice-to-have did not cost you the
            role, so it ranks below one that did.
          </p>

          <ol className="gap-list">
            {gaps.map((gap) => (
              <GapRow key={gap.name} gap={gap} analyzed={analyzed} />
            ))}
          </ol>
        </>
      ) : (
        <p className="page-empty">
          {report.isLoading
            ? "Loading…"
            : analyzed === 0
              ? "Nothing scored yet. Upload a resume and submit a posting, or build a recommendation feed from the workspace — this fills in once there are matches to read."
              : "No gaps found across your matches. Every requirement your postings listed, you meet."}
        </p>
      )}
      </div>
    </main>
  );
}
