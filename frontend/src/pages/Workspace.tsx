import { useCallback, useRef, useState, type DragEvent } from "react";
import { Navigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  errorMessage,
  jobApi,
  profileApi,
  type ExtractedSkill,
  type Job,
  type Profile,
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

function ProfilePanel({
  profile,
  onReset,
}: {
  profile: Profile;
  onReset: () => void;
}) {
  const queryClient = useQueryClient();
  const [startedAt] = useState(() => Date.now());

  const { data, isError, error } = useQuery({
    queryKey: ["profile", profile.id],
    queryFn: () => profileApi.get(profile.id),
    initialData: profile,
    refetchInterval: (query) => {
      if (query.state.data?.extraction_ok) return false;
      if (Date.now() - startedAt > EXTRACTION_TIMEOUT_MS) return false;
      return POLL_MS;
    },
  });

  const reextract = useMutation({
    mutationFn: () => profileApi.reextract(profile.id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["profile", profile.id] }),
  });

  const current = data ?? profile;
  const timedOut =
    !current.extraction_ok && Date.now() - startedAt > EXTRACTION_TIMEOUT_MS;

  return (
    <div className="panel card">
      <div className="panel-head">
        <div>
          <h3>{current.filename}</h3>
          <p className="panel-meta">
            {current.characters.toLocaleString()} characters extracted
          </p>
        </div>
        <button className="btn btn-ghost" onClick={onReset}>
          Replace
        </button>
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
            <p className="working-title">Deriving your structured profile</p>
            <p className="working-sub">
              An LLM is reading the text and validating its output against a
              strict schema. This usually takes 20–60 seconds.
            </p>
          </div>
        </div>
      )}

      {timedOut && (
        <div className="notice notice-warn">
          <p>
            Extraction hasn't come back. The provider may be down, or the
            document may have no readable text layer — a scanned PDF, for
            example.
          </p>
          <button
            className="btn btn-ghost"
            onClick={() => reextract.mutate()}
            disabled={reextract.isPending}
          >
            {reextract.isPending ? "Requeueing…" : "Try extraction again"}
          </button>
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
        Fetching lands at M5 — until then a queued job runs the real state
        machine and retry policy against a placeholder that deliberately does
        no network I/O.
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

/* --- Page ------------------------------------------------------------- */

export function Workspace() {
  const { data: user, isLoading } = useMe();
  const [profile, setProfile] = useState<Profile | null>(null);

  const upload = useMutation({
    mutationFn: (file: File) => profileApi.upload(file),
    onSuccess: setProfile,
  });

  if (isLoading) return null;
  if (!user) return <Navigate to="/signin" replace />;

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
            {profile ? (
              <ProfilePanel profile={profile} onReset={() => setProfile(null)} />
            ) : (
              <div className="panel card">
                <Dropzone onFile={(f) => upload.mutate(f)} busy={upload.isPending} />
                {upload.isError && (
                  <p className="form-error" role="alert">
                    {errorMessage(upload.error, "That file could not be read.")}
                  </p>
                )}
              </div>
            )}
          </section>

          <section aria-label="Job postings">
            {profile ? (
              <JobPanel profileId={profile.id} />
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
