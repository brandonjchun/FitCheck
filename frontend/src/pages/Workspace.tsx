import { useCallback, useEffect, useRef, useState, type DragEvent } from "react";
import { Navigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  batchApi,
  errorMessage,
  jobApi,
  profileApi,
  type Batch,
  type ExtractedSkill,
  type Job,
  type ProfileSummary,
} from "../api/client";
import { MatchFeed } from "../components/MatchFeed";
import { useMe } from "../hooks/useAuth";
import "./Workspace.css";

const POLL_MS = 2000;

/**
 * How many polls to give extraction before offering a retry.
 *
 * Counted in **polls, not wall-clock milliseconds**, and the difference is a
 * bug that was observed rather than imagined. The earlier version gave up when
 * `Date.now() - startedAt` passed 150s. Browsers throttle timers in a
 * backgrounded tab while `Date.now()` keeps advancing in real time, so a tab
 * the user alt-tabbed away from could burn the entire budget having asked the
 * server twice — then stop polling for good and render "no result after 150s"
 * over an extraction that had finished. Seen in production data: a resume
 * extracted in 44 seconds, and the retry button was pressed on it 68 minutes
 * later.
 *
 * Counting attempts makes the budget mean what it says: 75 polls is 75 actual
 * questions asked, whether they took two minutes or twenty.
 *
 * 75 × 2s ≈ 150s of foreground polling, which stays generous against the
 * measured extraction times — 26s on Gemini, 57s on a local model.
 */
const MAX_EXTRACTION_POLLS = 75;

/** Only for the elapsed counter the spinner shows. Gates nothing. */
const EXTRACTION_TIMEOUT_MS = MAX_EXTRACTION_POLLS * POLL_MS;

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
  const { data: current, isError, error, dataUpdatedAt, errorUpdatedAt } = useQuery({
    queryKey: ["profile", profileId],
    queryFn: () => profileApi.get(profileId),
    /* On for this query specifically, against the app-wide default.
     *
     * This is the recovery path. Once polling stops there is otherwise no way
     * back to the truth short of a manual reload, so a user who came back to a
     * backgrounded tab saw a stale "no result" over an extraction that had long
     * since finished. A profile is also exactly the kind of resource where a
     * refetch on focus is cheap and the answer may genuinely have changed while
     * the user was away. */
    refetchOnWindowFocus: true,
    refetchInterval: (query) => {
      if (query.state.data?.extraction_ok) return false;
      const asked = query.state.dataUpdateCount + query.state.errorUpdateCount;
      if (asked >= MAX_EXTRACTION_POLLS) return false;
      return POLL_MS;
    },
  });

  /* Attempts, mirrored into state so render can read what refetchInterval
   * decided on. Both timestamps, because a run of failed requests has to count
   * toward the budget as well -- otherwise a backend that is down polls
   * forever. */
  const [polls, setPolls] = useState(0);
  useEffect(() => {
    if (dataUpdatedAt || errorUpdatedAt) setPolls((n) => n + 1);
  }, [dataUpdatedAt, errorUpdatedAt]);

  /* The elapsed counter, and nothing else now.
   *
   * A spinner with no number attached is indistinguishable from a hang, which
   * is the confusion a job stranded on an orphaned queue produced. It used to
   * drive the timeout as well; that job moved to the poll count, because wall
   * clock and work done stop agreeing the moment the tab is backgrounded.
   */
  useEffect(() => {
    if (current?.extraction_ok) return;
    const id = setInterval(() => setElapsedMs(Date.now() - startedAt), 1000);
    return () => clearInterval(id);
  }, [current?.extraction_ok, startedAt]);

  const reextract = useMutation({
    mutationFn: () => profileApi.reextract(profileId),
    onSuccess: () => {
      // Resetting the budget is what resumes polling: the query only
      // re-evaluates refetchInterval after a fetch, and invalidate provides
      // that fetch. Without the reset the fresh attempt would inherit an
      // already-spent budget and stop again immediately.
      setPolls(0);
      setStartedAt(Date.now());
      setElapsedMs(0);
      queryClient.resetQueries({ queryKey: ["profile", profileId] });
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

  // Keyed on questions asked, matching what stopped the polling. Reading it
  // off the clock instead is what let a backgrounded tab claim a timeout it
  // had not actually waited through.
  const timedOut = !current.extraction_ok && polls >= MAX_EXTRACTION_POLLS;
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

/* --- Bulk URL upload -------------------------------------------------- */

/* Slower than the job poll. A batch is the walked-away case by definition --
 * nobody watches 500 fetches tick over -- and one aggregate query every three
 * seconds is the whole reason this endpoint exists instead of polling N jobs. */
const BATCH_POLL_MS = 3000;

/** Mirrors settings.max_urls_per_batch. Copy shows it; the server enforces it. */
const MAX_URLS_PER_BATCH = 500;

function BatchRow({ batch }: { batch: Batch }) {
  const { data } = useQuery({
    queryKey: ["batch", batch.id],
    queryFn: () => batchApi.get(batch.id),
    initialData: batch,
    // `is_complete` comes from the API so this does not hardcode which
    // statuses are terminal and drift when one is added.
    refetchInterval: (query) => (query.state.data?.is_complete ? false : BATCH_POLL_MS),
  });

  const current = data ?? batch;
  const counts = current.counts ?? {};
  const succeeded = counts.succeeded ?? 0;
  const failed = counts.failed ?? 0;
  const dead = counts.dead ?? 0;
  const inFlight = (counts.queued ?? 0) + (counts.running ?? 0);

  /* Widths off the batch's own `total`, not off the sum of the counts. If a
   * job row went missing the bar should come up short and show it, rather
   * than rescaling to look complete. */
  const pct = (n: number) => (current.total > 0 ? (n / current.total) * 100 : 0);

  return (
    <li className="batch-row">
      <div className="batch-top">
        <span className="batch-name">{current.filename}</span>
        <span className={`batch-state ${current.is_complete ? "is-done" : "is-running"}`}>
          {current.is_complete ? "Complete" : `${inFlight} left`}
        </span>
      </div>

      <div
        className="batch-bar"
        role="img"
        aria-label={`${succeeded} of ${current.total} fetched, ${failed} retrying, ${dead} dead`}
      >
        <span className="seg seg-ok" style={{ width: `${pct(succeeded)}%` }} />
        <span className="seg seg-warn" style={{ width: `${pct(failed)}%` }} />
        <span className="seg seg-bad" style={{ width: `${pct(dead)}%` }} />
      </div>

      <p className="batch-meta">
        {succeeded} of {current.total} fetched
        {failed > 0 && ` · ${failed} retrying`}
        {dead > 0 && ` · ${dead} gave up`}
      </p>

      {/* Every line the user sent is accounted for. A batch that quietly
        * ingests 500 of someone's 4,000 lines is worse than one that refuses,
        * because they cannot tell which 3,500 are missing. */}
      {(current.rejected > 0 || current.duplicates > 0) && (
        <p className="batch-note">
          {current.rejected} unreadable · {current.duplicates} duplicate
          {current.duplicates === 1 ? "" : "s"} skipped
        </p>
      )}
    </li>
  );
}

function BatchPanel({ profileId }: { profileId: number }) {
  const queryClient = useQueryClient();
  const [pasted, setPasted] = useState("");

  const batches = useQuery({
    queryKey: ["batches"],
    queryFn: () => batchApi.list(20),
  });

  const upload = useMutation({
    mutationFn: (source: File | string) => batchApi.create(profileId, source),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["batches"] });
      setPasted("");
    },
  });

  /* Counts non-blank lines, which is what the server will parse -- not
   * `split("\n").length`, which reports 1 for an empty box and counts the
   * trailing newline every textarea ends with. The number shown has to be the
   * number acted on, or the cap warning fires a line early. */
  const lineCount = pasted.split("\n").filter((line) => line.trim()).length;
  const overCap = lineCount > MAX_URLS_PER_BATCH;

  return (
    <div className="panel card">
      <div className="panel-head">
        <div>
          <h3>Bulk submit</h3>
          <p className="panel-meta">
            One posting URL per line, up to {MAX_URLS_PER_BATCH}. Paste them or
            upload a .txt.
          </p>
        </div>
      </div>

      {/* Paste is the primary path. Requiring a .txt presumes a list that
        * already exists as a file, which is not how anyone collects postings --
        * the real case is a handful of URLs out of open tabs. */}
      <form
        className="batch-paste"
        onSubmit={(e) => {
          e.preventDefault();
          if (pasted.trim() && !overCap) upload.mutate(pasted);
        }}
      >
        <label htmlFor="batch-paste" className="sr-only">
          Job posting URLs, one per line
        </label>
        <textarea
          id="batch-paste"
          className="batch-textarea"
          rows={5}
          spellCheck={false}
          placeholder={"https://boards.greenhouse.io/…\nhttps://jobs.lever.co/…"}
          value={pasted}
          disabled={upload.isPending}
          onChange={(e) => setPasted(e.target.value)}
        />
        <div className="batch-paste-foot">
          <span className={`batch-hint ${overCap ? "is-over" : ""}`}>
            {lineCount === 0
              ? "One URL per line."
              : `${lineCount} line${lineCount === 1 ? "" : "s"}${
                  overCap ? ` — over the ${MAX_URLS_PER_BATCH} limit` : ""
                }`}
          </span>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={upload.isPending || !pasted.trim() || overCap}
          >
            {upload.isPending ? "Submitting…" : "Submit URLs"}
          </button>
        </div>
      </form>

      <div className="batch-upload">
        <input
          type="file"
          accept=".txt,text/plain"
          id="batch-input"
          className="sr-only"
          disabled={upload.isPending}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) upload.mutate(file);
            // Cleared so re-picking the same filename still fires change.
            e.target.value = "";
          }}
        />
        <label htmlFor="batch-input" className="btn btn-ghost">
          {upload.isPending ? "Uploading…" : "…or upload a .txt"}
        </label>
        <span className="batch-hint">
          Bulk lists land on the <code>ingest</code> queue, so they never delay a
          single submission.
        </span>
      </div>

      {upload.isError && (
        <p className="form-error" role="alert">
          {errorMessage(upload.error, "That list could not be accepted.")}
        </p>
      )}

      {/* The receipt. `accepted + rejected + duplicates` equals the non-blank
        * line count of their file, which is the contract the endpoint makes. */}
      {upload.isSuccess && upload.data && (
        <div className="notice notice-info">
          <strong>{upload.data.accepted} queued.</strong>{" "}
          {upload.data.rejected} line{upload.data.rejected === 1 ? "" : "s"} could
          not be read as a URL, {upload.data.duplicates} already submitted.
        </div>
      )}

      {batches.data && batches.data.length > 0 ? (
        <ul className="batch-list">
          {batches.data.map((batch) => (
            <BatchRow key={batch.id} batch={batch} />
          ))}
        </ul>
      ) : (
        <p className="panel-empty">
          Nothing submitted in bulk yet. Useful when you have a few postings
          open at once rather than one you are looking at right now.
        </p>
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
              <>
                <JobPanel profileId={openId} />
                <BatchPanel profileId={openId} />
              </>
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

        {/* Full width, below the grid. This is the output the rest of the page
          * exists to produce, and the skill breakdown needs horizontal room
          * that a half-width column does not have. */}
        {openId != null && (
          <section aria-label="Matches" className="workspace-matches">
            <MatchFeed profileId={openId} />
          </section>
        )}
      </div>
    </main>
  );
}
