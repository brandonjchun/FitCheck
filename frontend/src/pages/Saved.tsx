import { useState } from "react";
import { Navigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { errorMessage, matchApi, type SavedMatch } from "../api/client";
import { useMe } from "../hooks/useAuth";
import "./Saved.css";

const STALE_MS = 60_000;

const VERDICTS = [
  { value: "", label: "Everything" },
  { value: "applied", label: "Applied" },
  { value: "interested", label: "Interested" },
  { value: "not_interested", label: "Not for me" },
] as const;

const VERDICT_LABEL: Record<string, string> = {
  applied: "Applied",
  interested: "Interested",
  not_interested: "Not for me",
};

function pct(score: number): string {
  return `${Math.round(score * 100)}%`;
}

function when(iso: string): string {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 3600) return `${Math.max(1, Math.floor(seconds / 60))}m ago`;
  if (seconds < 172800) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function SavedRow({ row }: { row: SavedMatch }) {
  const title = row.posting_title ?? "Untitled posting";
  const heading = row.posting_company ? `${title} · ${row.posting_company}` : title;

  return (
    <li className={`saved-row verdict-${row.verdict}`}>
      <span className="saved-score">{pct(row.final_score)}</span>

      <div className="saved-main">
        {row.posting_url ? (
          <a
            href={row.posting_url}
            target="_blank"
            rel="noreferrer noopener"
            className="saved-title"
          >
            {heading}
          </a>
        ) : (
          <span className="saved-title">{heading}</span>
        )}

        <span className="saved-meta">
          {VERDICT_LABEL[row.verdict] ?? row.verdict} · {when(row.verdict_at)}
          {/* Flagged, not hidden. You applied to it; a role being filled since
            * is information, and dropping the row would make an application
            * disappear from your own history. */}
          {row.posting_closed && (
            <span className="saved-closed" title="This posting is no longer listed">
              closed
            </span>
          )}
        </span>
      </div>
    </li>
  );
}

export function Saved() {
  const { data: user, isLoading: loadingUser } = useMe();
  const [verdict, setVerdict] = useState<string>("");

  const saved = useQuery({
    queryKey: ["saved", verdict],
    queryFn: () => matchApi.saved(verdict || undefined),
    enabled: Boolean(user),
    staleTime: STALE_MS,
  });

  if (loadingUser) return null;
  if (!user) return <Navigate to="/signin" replace />;

  const rows = saved.data ?? [];
  const applied = rows.filter((r) => r.verdict === "applied").length;

  return (
    <main id="main" className="page-shell saved">
      <div className="container">
      <header className="page-head">
        <h1>Saved</h1>
        <p className="page-sub">
          Everything you reacted to, most recent first. Each posting appears
          once at its latest stage — marking a role applied after marking it
          interested moves it rather than duplicating it.
        </p>
      </header>

      <div className="saved-controls" role="group" aria-label="Filter by verdict">
        {VERDICTS.map((option) => (
          <button
            key={option.value}
            type="button"
            className={`btn btn-ghost btn-sm${verdict === option.value ? " is-active" : ""}`}
            onClick={() => setVerdict(option.value)}
          >
            {option.label}
          </button>
        ))}

        {applied > 0 && !verdict && (
          <span className="saved-count">
            {applied} {applied === 1 ? "application" : "applications"}
          </span>
        )}
      </div>

      {saved.isError && (
        <p className="form-error" role="alert">
          {errorMessage(saved.error, "Could not load your saved postings.")}
        </p>
      )}

      {rows.length > 0 ? (
        <ul className="saved-list">
          {rows.map((row) => (
            <SavedRow key={row.match_id} row={row} />
          ))}
        </ul>
      ) : (
        <p className="page-empty">
          {saved.isLoading
            ? "Loading…"
            : verdict
              ? "Nothing under that filter yet."
              : "Nothing saved yet. Use the buttons under any match in your workspace to mark it interested, applied, or not for you — they show up here."}
        </p>
      )}
      </div>
    </main>
  );
}
