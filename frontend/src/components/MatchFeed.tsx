import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  errorMessage,
  matchApi,
  type FeedbackVerdict,
  type FeedFilters,
  type Match,
  type MatchSkill,
} from "../api/client";
import "./MatchFeed.css";

/* Scoring runs asynchronously after a fetch succeeds, so a match appears some
 * seconds after its job does. There is no `is_complete` to stop on -- a
 * posting can be re-fetched and re-scored at any time -- so this polls at a
 * relaxed interval rather than trying to guess when it is finished. The query
 * is capped at 25 rows and served by `matches_feed_idx`, so it is cheap. */
const MATCH_POLL_MS = 5000;

/* Once the feed has rows, it is a precomputed resource: scores change when a
 * posting is re-fetched or the recommender runs, neither of which happens on
 * a five-second cadence. §7.3 option 1, final. */
const FEED_STALE_MS = 5 * 60_000;

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

/**
 * Feedback capture for one match.
 *
 * Records rather than toggles. The table is append-only, so pressing
 * "Interested" and later "Applied" states two true things in order — which is
 * the funnel a ranking model would eventually learn from. Rendering these as
 * mutually-exclusive radio buttons would throw that sequence away in the UI
 * even though the API preserves it.
 *
 * Nothing here re-ranks anything today, and the label says so: promising the
 * user their feedback improves results, when no model reads it this semester,
 * would be a lie the interface tells on the system's behalf.
 */
function FeedbackButtons({ matchId }: { matchId: number }) {
  const [sent, setSent] = useState<FeedbackVerdict | null>(null);
  const [failed, setFailed] = useState(false);

  const record = useMutation({
    mutationFn: (verdict: FeedbackVerdict) => matchApi.feedback(matchId, verdict),
    onMutate: (verdict) => {
      setFailed(false);
      setSent(verdict);
    },
    /* Reverted on failure rather than left showing a confirmation. A label
     * that never reached the server is exactly the data loss this feature
     * exists to prevent, so it must not look like it succeeded. */
    onError: () => {
      setSent(null);
      setFailed(true);
    },
  });

  const options: Array<{ verdict: FeedbackVerdict; label: string }> = [
    { verdict: "interested", label: "Interested" },
    { verdict: "not_interested", label: "Not for me" },
    { verdict: "applied", label: "Applied" },
  ];

  return (
    <div className="mf-feedback">
      <span className="mf-feedback-label">Was this a good match?</span>

      {options.map(({ verdict, label }) => (
        <button
          key={verdict}
          type="button"
          className={`btn btn-ghost btn-sm mf-verdict${sent === verdict ? " is-sent" : ""}`}
          disabled={record.isPending}
          onClick={() => record.mutate(verdict)}
        >
          {label}
        </button>
      ))}

      {sent && (
        <span className="mf-feedback-ack" role="status">
          Recorded — thanks.
        </span>
      )}
      {failed && (
        <span className="form-error" role="alert">
          Could not record that. Try again.
        </span>
      )}
    </div>
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

      <FeedbackButtons matchId={match.id} />
    </li>
  );
}

/** The filter bar. Controlled from the parent so the query key can include it. */
function FilterBar({
  filters,
  onChange,
}: {
  filters: FeedFilters;
  onChange: (next: FeedFilters) => void;
}) {
  const set = (patch: Partial<FeedFilters>) => onChange({ ...filters, ...patch });

  return (
    <div className="mf-filters" role="group" aria-label="Feed filters">
      <label className="mf-filter">
        <span>Source</span>
        <select
          value={filters.origin ?? ""}
          onChange={(e) =>
            set({ origin: (e.target.value || undefined) as FeedFilters["origin"] })
          }
        >
          <option value="">All</option>
          <option value="recommendation">Recommended</option>
          <option value="user_submission">Submitted by me</option>
        </select>
      </label>

      <label className="mf-filter">
        <span>Seniority</span>
        <select
          value={filters.seniority?.[0] ?? ""}
          onChange={(e) =>
            set({ seniority: e.target.value ? [e.target.value] : undefined })
          }
        >
          <option value="">Any</option>
          <option value="junior">Junior</option>
          <option value="mid">Mid</option>
          <option value="senior">Senior</option>
          <option value="staff">Staff</option>
        </select>
      </label>

      <label className="mf-filter mf-filter-check">
        <input
          type="checkbox"
          checked={filters.remote_only ?? false}
          onChange={(e) => set({ remote_only: e.target.checked || undefined })}
        />
        <span>Remote only</span>
      </label>

      {/* Closed roles are hidden by default rather than dropped: a filled
        * posting is still a true record of what was recommended, and the
        * toggle keeps it reachable without presenting a dead link as live. */}
      <label className="mf-filter mf-filter-check">
        <input
          type="checkbox"
          checked={filters.include_closed ?? false}
          onChange={(e) => set({ include_closed: e.target.checked || undefined })}
        />
        <span>Include closed</span>
      </label>
    </div>
  );
}

export function MatchFeed({ profileId }: { profileId: number }) {
  const [filters, setFilters] = useState<FeedFilters>({});

  /* Section 6.9 option 1, client half: a feed is built because somebody asked
   * for it, not on a schedule. The server decides whether there is anything
   * to do -- it answers `already_current` without touching the queue when
   * this profile already has a feed -- so this stays a plain button rather
   * than something that has to reason about freshness itself.
   *
   * Deliberately not automatic on mount. An auto-trigger would fire for every
   * profile a user clicks through, and a feed build is a recall plus 200
   * reranks; making that the cost of *looking* at a profile is how a lazy
   * strategy quietly becomes an eager one. */
  const build = useMutation({
    mutationFn: () => matchApi.recommend(profileId),
  });

  const matches = useQuery({
    /* Filters belong in the key. Without them React Query would serve the
     * previous filter's rows from cache while the new request is in flight,
     * so the list would briefly contradict the controls that produced it. */
    queryKey: ["matches", profileId, filters],
    queryFn: () => matchApi.list(profileId, 25, filters),
    /* Path B's feed is precomputed and does not change second to second, so
     * §7.3 drops the 5s poll here in favour of a stale time plus a refetch
     * when the window regains focus. Polling a precomputed resource every two
     * seconds is pure waste; the case for the old interval was Path A, where
     * a score genuinely appears seconds after a submission. That case is kept
     * alive by `refetchInterval` staying on only while nothing has arrived. */
    staleTime: FEED_STALE_MS,
    refetchOnWindowFocus: true,
    refetchInterval: (query) => {
      const rows = query.state.data ?? [];
      /* Nothing yet: this is the Path A case where a score really does arrive
       * seconds after a submission, so the old interval still earns its keep. */
      if (rows.length === 0) return MATCH_POLL_MS;
      /* A build is running and none of its results have landed. Without this
       * the poll would already be off -- the feed has rows, they are just the
       * wrong ones -- and the recommendations would appear only on the next
       * focus event, which looks like the button did nothing. */
      if (
        build.data?.status === "queued" &&
        !rows.some((m) => m.origin === "recommendation")
      ) {
        return MATCH_POLL_MS;
      }
      return false;
    },
  });

  const rows = matches.data ?? [];

  /* Scores from different scorer generations are not comparable, and a feed
   * sorted across two of them is silently wrong -- position 3 beats position 4
   * because it was scored under different rules, not because it fits better.
   * The API exposes the version precisely so a client can say so. */
  const versions = new Set(rows.map((m) => m.scorer_version));

  /* An empty feed means something different once a filter is on, and saying
   * "submit a posting URL" to somebody who has fifty matches and ticked
   * "remote only" is advice for a problem they do not have. */
  const hasFilters = Object.values(filters).some((v) => v !== undefined);

  return (
    <div className="panel card">
      <div className="panel-head">
        <div>
          <h3>Matches</h3>
          <p className="panel-meta">
            Postings you submitted and postings we found, scored against this
            resume, best first.
          </p>
        </div>

        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={build.isPending}
          onClick={() => build.mutate()}
        >
          {build.isPending ? "Starting…" : "Find matches for me"}
        </button>
      </div>

      {/* Three outcomes, three messages. `already_current` and
        * `profile_not_ready` both mean "no job was queued", and saying so the
        * same way would leave someone waiting on a build that is not coming. */}
      {build.data?.status === "queued" && (
        <p className="mf-note" role="status">
          Building your feed — searching the catalog for postings that fit this
          resume. New matches appear here as they are scored.
        </p>
      )}
      {build.data?.status === "already_current" && (
        <p className="mf-note" role="status">
          Your feed is already up to date with the current scorer.
        </p>
      )}
      {build.data?.status === "profile_not_ready" && (
        <p className="mf-note" role="status">
          This resume is still being processed. Once extraction finishes, we can
          search the catalog against it.
        </p>
      )}
      {build.isError && (
        <p className="form-error" role="alert">
          {errorMessage(build.error, "Could not start the search.")}
        </p>
      )}

      <FilterBar filters={filters} onChange={setFilters} />

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
            : hasFilters
              ? "No matches fit those filters. Widen them to see more."
              : "No scores yet. Submit a posting URL above — once it is fetched, it is scored against this resume automatically and appears here."}
        </p>
      )}
    </div>
  );
}
