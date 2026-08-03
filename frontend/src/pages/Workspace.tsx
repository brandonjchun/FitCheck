import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { Navigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  errorMessage,
  jobApi,
  profileApi,
  type ExtractedSkill,
  type Job,
  type ProfileSummary,
} from "../api/client";
import { useMe } from "../hooks/useAuth";
import "./Workspace.css";

const POLL_MS = 2000;
/** Extraction measured 26s on Gemini and 57s on a local model, so this is
 *  generous rather than tight. Past it we stop polling and offer a retry
 *  instead of spinning forever against a job that already failed. */
const EXTRACTION_TIMEOUT_MS = 150_000;

/* --- Resume upload ---------------------------------------------------- */

function Dropzone({
  onFile,
  busy,
}: {
  onFile: (file: File) => void;
  busy: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files?.[0];
      if (file) onFile(file);
    },
    [onFile],
  );

  return (
    <div
      className={`dropzone ${dragging ? "is-dragging" : ""} ${busy ? "is-busy" : ""}`}
      onDragOver={(e) => {
        // Without preventDefault the browser navigates to the dropped file,
        // which looks exactly like the app crashing.
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx"
        className="sr-only"
        id="resume-input"
        disabled={busy}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
          // Cleared so re-selecting the same filename still fires change.
          e.target.value = "";
        }}
      />

      <svg viewBox="0 0 24 24" width="34" height="34" fill="none" aria-hidden="true">
        <path
          d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>

      <p className="dropzone-title">
        {busy ? "Uploading…" : "Drop your resume here"}
      </p>
      <p className="dropzone-sub">
        PDF or DOCX, up to 2 MB.{" "}
        <label htmlFor="resume-input" className="dropzone-browse">
          browse instead
        </label>
      </p>
    </div>
  );
}

function SkillPill({ skill }: { skill: ExtractedSkill }) {
  return (
    <li className="skill-pill">
      <span className="skill-name">{skill.name}</span>
      {skill.years != null && <span className="skill-years">{skill.years}y</span>}
      {skill.source && <span className="skill-source">{skill.source}</span>}
      {/* The evidence span is the reason to trust the extraction at all, so
       * it is always rendered rather than hidden behind a hover. */}
      {skill.evidence && <q className="skill-evidence">{skill.evidence}</q>}
    </li>
  );
}

function ProfilePanel({ profileId }: { profileId: number }) {
  const queryClient = useQueryClient();
  const [startedAt, setStartedAt] = useState(() => Date.now());
  const [elapsedMs, setElapsedMs] = useState(0);

  /* Fetched by id rather than handed in as a prop. That is the whole change
   * that makes a refresh survivable: the panel needs nothing but a number,
   * and the number comes from the server's list instead of from whatever the
   * upload mutation happened to leave in memory. */
  const { data: current, isError, error } = useQuery({
    queryKey: ["profile", profileId],
    queryFn: () => profileApi.get(profileId),
    refetchInterval: (query) => {
      if (query.state.data?.extraction_ok) return false;
      if (Date.now() - startedAt > EXTRACTION_TIMEOUT_MS) return false;
      return POLL_MS;
    },
  });

  /* A ticking clock, not a value derived during render.
   *
   * The earlier version computed the timeout as `Date.now() - startedAt >
   * LIMIT` while rendering. That reads correctly and never fires: once the
   * poll interval returns false the query stops refetching, so nothing
   * re-renders this component, so the expression is never evaluated again and
   * the spinner runs forever. A timeout needs something that actually wakes
   * up at the boundary.
   *
   * It doubles as the elapsed counter. A spinner with no number attached is
   * indistinguishable from a hang -- which is exactly the confusion that a
   * job stranded on an orphaned queue produced.
   */
  useEffect(() => {
    if (current?.extraction_ok) return;
    const id = setInterval(() => setElapsedMs(Date.now() - startedAt), 1000);
    return () => clearInterval(id);
  }, [current?.extraction_ok, startedAt]);

  const reextract = useMutation({
    mutationFn: () => profileApi.reextract(profileId),
    onSuccess: () => {
      // Restarting the clock is what resumes polling: the query only
      // re-evaluates refetchInterval after a fetch, and invalidate provides
      // that fetch. Without the reset the fresh attempt would inherit an
      // already-expired deadline and stop again immediately.
      setStartedAt(Date.now());
      setElapsedMs(0);
      queryClient.invalidateQueries({ queryKey: ["profile", profileId] });
    },
  });

  // After every hook, never between them. An early return above any of the
  // above would change the hook count between renders.
  if (!current) {
    return (
      <div className="panel card">
        {isError ? (
          <p className="form-error" role="alert">
            {errorMessage(error, "That resume could not be loaded.")}
          </p>
        ) : (
          <p className="panel-empty">Loading resume…</p>
        )}
      </div>
    );
  }

  const timedOut = !current.extraction_ok && elapsedMs > EXTRACTION_TIMEOUT_MS;
  const elapsedSeconds = Math.floor(elapsedMs / 1000);

  return (
    <div className="panel card">
      <div className="panel-head">
        <div>
          <h3>{current.filename}</h3>
          <p className="panel-meta">
            {current.characters.toLocaleString()} characters extracted
            {current.is_active && <span className="tag tag-ok">active</span>}
          </p>
        </div>
      </div>

      {isError && (
        <p className="form-error" role="alert">
          {errorMessage(error)}
        </p>
      )}

      {!current.extraction_ok && !timedOut && (
        <div className="working" role="status">
          <span className="spinner" aria-hidden="true" />
          <div>
            <p className="working-title">
              Deriving your structured profile
              <span className="elapsed">{elapsedSeconds}s</span>
            </p>
            <p className="working-sub">
              An LLM is reading the text and validating its output against a
              strict schema. This usually takes 20–60 seconds.
              {elapsedSeconds > 75 &&
                " Longer than usual — a local model is slower than a hosted one."}
            </p>
          </div>
        </div>
      )}

      {timedOut && (
        <div className="notice notice-warn">
          <p>
            <strong>
              No result after {Math.floor(EXTRACTION_TIMEOUT_MS / 1000)}s.
            </strong>{" "}
            Extraction is queued but nothing has come back. The usual causes,
            in the order worth checking:
          </p>
          <ul className="notice-list">
            <li>No worker is consuming the queue this job landed on.</li>
            <li>The LLM provider is unreachable or rate-limiting.</li>
            <li>
              The document has no text layer — a scanned PDF is images, not
              characters.
            </li>
          </ul>
          <button
            className="btn btn-ghost"
            onClick={() => reextract.mutate()}
            disabled={reextract.isPending}
          >
            {reextract.isPending ? "Requeueing…" : "Try extraction again"}
          </button>
          {reextract.isError && (
            <p className="form-error" role="alert">
              {errorMessage(reextract.error, "Could not requeue.")}
            </p>
          )}
        </div>
      )}

      {current.extraction_ok && (
        <>
          <div className="profile-facts">
            <div>
              <span className="fact-label">Seniority</span>
              <span className="fact-value">{current.seniority ?? "unknown"}</span>
            </div>
            <div>
              <span className="fact-label">Experience</span>
              <span className="fact-value">
                {current.years_experience != null
                  ? `${current.years_experience} years`
                  : "not stated"}
              </span>
            </div>
            <div>
              <span className="fact-label">Skills found</span>
              <span className="fact-value">{current.skills.length}</span>
            </div>
          </div>

          {current.skills.length > 0 ? (
            <ul className="skill-list">
              {current.skills.map((skill, i) => (
                <SkillPill key={`${skill.name}-${i}`} skill={skill} />
              ))}
            </ul>
          ) : (
            <p className="panel-empty">
              Extraction ran and found no skills it could support with a
              verbatim phrase. That is a real result, not a failure — the
              prompt refuses to list a skill it cannot quote.
            </p>
          )}
        </>
      )}
    </div>
  );
}

/* --- Job submission --------------------------------------------------- */

const STATUS_COPY: Record<string, { label: string; tone: string }> = {
  queued: { label: "Queued", tone: "wait" },
  running: { label: "Running", tone: "wait" },
  succeeded: { label: "Succeeded", tone: "ok" },
  failed: { label: "Failed — will retry", tone: "warn" },
  dead: { label: "Dead-lettered", tone: "bad" },
};

function JobRow({ job }: { job: Job }) {
  const { data } = useQuery({
    queryKey: ["job", job.id],
    queryFn: () => jobApi.get(job.id),
    initialData: job,
    // Stops on its own at a terminal state. `is_terminal` comes from the API
    // precisely so this list does not have to hardcode which statuses those
    // are and drift when one is added.
    refetchInterval: (query) => (query.state.data?.is_terminal ? false : POLL_MS),
  });

  const current = data ?? job;
  const status = STATUS_COPY[current.status] ?? {
    label: current.status,
    tone: "wait",
  };

  return (
    <li className="job-row">
      <div className="job-main">
        <span className={`status-dot status-${status.tone}`} aria-hidden="true" />
        <div className="job-text">
          <a href={current.url} target="_blank" rel="noreferrer noopener" className="job-url">
            {current.url}
          </a>
          <p className="job-meta">
            {status.label}
            {current.attempts > 0 && ` · attempt ${current.attempts}`}
          </p>
          {current.last_error && <p className="job-error">{current.last_error}</p>}
        </div>
      </div>
    </li>
  );
}

function JobPanel({ profileId }: { profileId: number }) {
  const [url, setUrl] = useState("");
  const queryClient = useQueryClient();

  const jobs = useQuery({
    queryKey: ["jobs", profileId],
    queryFn: () => jobApi.list({ profile_id: profileId, limit: 20 }),
  });

  const submit = useMutation({
    mutationFn: () => jobApi.submit(profileId, url),
    onSuccess: () => {
      setUrl("");
      queryClient.invalidateQueries({ queryKey: ["jobs", profileId] });
    },
  });

  return (
    <div className="panel card">
      <div className="panel-head">
        <div>
          <h3>Job postings</h3>
          <p className="panel-meta">
            Submitted URLs are queued, not fetched inline — the API returns 202
            immediately.
          </p>
        </div>
      </div>

      <form
        className="job-form"
        onSubmit={(e) => {
          e.preventDefault();
          if (url.trim()) submit.mutate();
        }}
      >
        <input
          className="input"
          type="url"
          required
          placeholder="https://boards.greenhouse.io/company/jobs/123"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          aria-label="Job posting URL"
        />
        <button className="btn btn-primary" disabled={submit.isPending || !url.trim()}>
          {submit.isPending ? "Queueing…" : "Queue it"}
        </button>
      </form>

      {submit.isError && (
        <p className="form-error" role="alert">
          {errorMessage(submit.error, "Could not queue that URL.")}
        </p>
      )}

      <div className="notice notice-info">
        Queued URLs are really fetched: robots.txt is checked first, requests
        are rate limited per host and size capped, and transient failures retry
        with backoff while a 404 or a robots disallow fails immediately rather
        than burning the retry budget. Two people submitting the same posting
        converge on one stored copy.
      </div>

      {jobs.data && jobs.data.length > 0 && (
        <ul className="job-list">
          {jobs.data.map((job) => (
            <JobRow key={job.id} job={job} />
          ))}
        </ul>
      )}
    </div>
  );
}

/* --- Resume versions -------------------------------------------------- */

function VersionRow({
  version,
  isOpen,
  onOpen,
  onDeleted,
}: {
  version: ProfileSummary;
  isOpen: boolean;
  onOpen: () => void;
  onDeleted: () => void;
}) {
  const queryClient = useQueryClient();
  const [confirming, setConfirming] = useState(false);

  const activate = useMutation({
    mutationFn: () => profileApi.activate(version.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["profiles"] }),
  });

  const remove = useMutation({
    mutationFn: () => profileApi.remove(version.id),
    onSuccess: () => {
      onDeleted();
      queryClient.invalidateQueries({ queryKey: ["profiles"] });
    },
  });

  const busy = activate.isPending || remove.isPending;

  return (
    <li className={`version-row ${isOpen ? "is-open" : ""}`}>
      <button className="version-main" onClick={onOpen} aria-current={isOpen}>
        <span className="version-name">{version.filename}</span>
        <span className="version-meta">
          {new Date(version.created_at).toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
          })}
          {" · "}
          {version.extraction_ok
            ? `${version.skill_count} skill${version.skill_count === 1 ? "" : "s"}`
            : "not extracted"}
        </span>
      </button>

      <div className="version-actions">
        {version.is_active ? (
          <span className="tag tag-ok">active</span>
        ) : (
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => activate.mutate()}
            disabled={busy}
            title="Make this the resume that drives your feed"
          >
            {activate.isPending ? "…" : "Use this"}
          </button>
        )}

        {/* Two-step, because it cascades. Deleting a resume takes every job
          * submitted against it, and there is no undo. */}
        {confirming ? (
          <>
            <button
              className="btn btn-danger btn-sm"
              onClick={() => remove.mutate()}
              disabled={busy}
            >
              {remove.isPending ? "Deleting…" : "Confirm"}
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setConfirming(false)}>
              Cancel
            </button>
          </>
        ) : (
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setConfirming(true)}
            disabled={busy}
          >
            Delete
          </button>
        )}
      </div>

      {confirming && !remove.isPending && (
        <p className="version-warn">
          Deletes this resume and every job submitted against it.
        </p>
      )}

      {(activate.isError || remove.isError) && (
        <p className="form-error" role="alert">
          {errorMessage(activate.error ?? remove.error, "That did not work.")}
        </p>
      )}
    </li>
  );
}

/* --- Page ------------------------------------------------------------- */

export function Workspace() {
  const { data: user, isLoading } = useMe();
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const versions = useQuery({
    queryKey: ["profiles"],
    queryFn: profileApi.list,
    enabled: !!user,
  });

  const upload = useMutation({
    mutationFn: (file: File) => profileApi.upload(file),
    onSuccess: (created) => {
      setSelectedId(created.id);
      queryClient.invalidateQueries({ queryKey: ["profiles"] });
    },
  });

  if (isLoading) return null;
  if (!user) return <Navigate to="/signin" replace />;

  const list = versions.data ?? [];

  /* Which resume is open, in priority order. An explicit click wins, but only
   * while that row still exists -- deleting the open resume must not leave the
   * page pointing at an id the server will 404. Otherwise the active one,
   * since that is what the feed uses; otherwise the newest upload. */
  const selectionIsLive = selectedId != null && list.some((v) => v.id === selectedId);
  const openId =
    (selectionIsLive ? selectedId : null) ??
    list.find((v) => v.is_active)?.id ??
    list[0]?.id ??
    null;

  return (
    <main id="main" className="workspace">
      <div className="container">
        <header className="workspace-head">
          <span className="eyebrow">Workspace</span>
          <h1 className="workspace-title">Your resume, structured.</h1>
          <p className="workspace-sub">Signed in as {user.email}</p>
        </header>

        <div className="workspace-grid">
          <section aria-label="Resume">
            <div className="panel card">
              <Dropzone onFile={(f) => upload.mutate(f)} busy={upload.isPending} />
              {upload.isError && (
                <p className="form-error" role="alert">
                  {errorMessage(upload.error, "That file could not be read.")}
                </p>
              )}

              {list.length > 0 && (
                <>
                  <div className="version-head">
                    <h3>Your resumes</h3>
                    {/* Stated rather than assumed. Upload does not promote, so
                      * without this the second upload looks like it silently
                      * did nothing. */}
                    <p className="panel-meta">
                      A new upload is kept as a version. The active one drives
                      your feed.
                    </p>
                  </div>
                  <ul className="version-list">
                    {list.map((version) => (
                      <VersionRow
                        key={version.id}
                        version={version}
                        isOpen={version.id === openId}
                        onOpen={() => setSelectedId(version.id)}
                        onDeleted={() => setSelectedId(null)}
                      />
                    ))}
                  </ul>
                </>
              )}
            </div>

            {/* Keyed on the id so switching resumes remounts rather than
              * reusing the old one's elapsed clock and timeout deadline. */}
            {openId != null && <ProfilePanel key={openId} profileId={openId} />}
          </section>

          <section aria-label="Job postings">
            {openId != null ? (
              <JobPanel profileId={openId} />
            ) : (
              <div className="panel card panel-locked">
                <h3>Job postings</h3>
                <p className="panel-empty">
                  Upload a resume first. A posting is scored against a profile,
                  so there is nothing to submit one against yet.
                </p>
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
