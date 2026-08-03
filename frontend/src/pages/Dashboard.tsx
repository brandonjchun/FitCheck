import { Navigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  errorMessage,
  opsApi,
  type DeadLetterItem,
  type QueueHealth,
  type WorkerInfo,
} from "../api/client";
import { useMe } from "../hooks/useAuth";
import "./Dashboard.css";

/* Fast enough that a queue draining is visible as it happens, slow enough
 * that it is not itself load. The endpoint is a handful of Redis reads plus
 * one grouped count. */
const OPS_POLL_MS = 3000;

/* --- Queue health ----------------------------------------------------- */

type QueueVerdict = {
  tone: "ok" | "warn" | "bad" | "idle";
  label: string;
  detail?: string;
};

/**
 * Turn a queue's numbers into a judgment.
 *
 * A dashboard that only prints counts makes the reader do the diagnosis. The
 * two states worth shouting about are both invisible in a bare depth number:
 * an undeclared queue holding work, and a declared queue with depth but no
 * worker. Both mean jobs that will never run, and neither shows up as an
 * error anywhere -- no retry, no failure, no log line.
 */
function verdictFor(q: QueueHealth): QueueVerdict {
  if (!q.declared && q.depth > 0) {
    return {
      tone: "bad",
      label: "Stranded",
      detail:
        "This queue is not declared in QUEUE_NAMES. Work here is almost always a rename that left a producer behind — nothing will ever consume it.",
    };
  }
  if (q.depth > 0 && q.worker_count === 0) {
    return {
      tone: "bad",
      label: "No consumer",
      detail: "Jobs are waiting and no worker drains this queue.",
    };
  }
  if (q.worker_count === 0) {
    return { tone: "warn", label: "Unwatched", detail: "Empty, but no worker is assigned." };
  }
  if (q.depth > 100) {
    return { tone: "warn", label: "Backlog", detail: "Depth is growing faster than it drains." };
  }
  if (q.depth > 0) return { tone: "ok", label: "Draining" };
  return { tone: "idle", label: "Idle" };
}

function QueueCard({ queue }: { queue: QueueHealth }) {
  const verdict = verdictFor(queue);

  return (
    <li className={`queue-card tone-${verdict.tone}`}>
      <div className="queue-top">
        <h3 className="queue-name">
          {queue.name}
          {!queue.declared && <span className="tag tag-bad">undeclared</span>}
        </h3>
        <span className={`verdict verdict-${verdict.tone}`}>{verdict.label}</span>
      </div>

      <div className="queue-depth">
        <span className="depth-value">{queue.depth}</span>
        <span className="depth-label">waiting</span>
      </div>

      <dl className="queue-registries">
        <div>
          <dt>Running</dt>
          <dd>{queue.started}</dd>
        </div>
        <div>
          <dt>Failed</dt>
          <dd className={queue.failed > 0 ? "is-bad" : ""}>{queue.failed}</dd>
        </div>
        <div>
          <dt>Deferred</dt>
          <dd>{queue.deferred}</dd>
        </div>
        <div>
          <dt>Workers</dt>
          <dd className={queue.worker_count === 0 ? "is-warn" : ""}>
            {queue.worker_count}
          </dd>
        </div>
      </dl>

      {verdict.detail && <p className="queue-detail">{verdict.detail}</p>}
    </li>
  );
}

/* --- Workers ---------------------------------------------------------- */

function WorkerRow({ worker }: { worker: WorkerInfo }) {
  return (
    <li className="worker-row">
      <div className="worker-head">
        <span className={`status-dot status-${worker.state === "busy" ? "wait" : "ok"}`} />
        <code className="worker-name">{worker.name.slice(0, 12)}</code>
        <span className="worker-state">{worker.state}</span>
      </div>
      <div className="worker-queues">
        {/* Order matters and is preserved from the server: RQ checks queues
         * left to right every time it looks for work, so the first name is
         * the one that gets priority. */}
        {worker.queues.map((q, i) => (
          <span key={q} className="tag">
            {i > 0 && <span className="tag-arrow">→</span>}
            {q}
          </span>
        ))}
      </div>
      <p className="worker-stats">
        {worker.successful_jobs} ok · {worker.failed_jobs} failed
      </p>
    </li>
  );
}

/* --- Dead letter ------------------------------------------------------ */

function DeadLetterRow({ item }: { item: DeadLetterItem }) {
  const queryClient = useQueryClient();
  const requeue = useMutation({
    mutationFn: () => opsApi.requeue(item.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["ops", "dead-letter"] });
      queryClient.invalidateQueries({ queryKey: ["ops", "overview"] });
    },
  });

  return (
    <li className="dead-row">
      <div className="dead-main">
        <span className={`tag ${item.status === "dead" ? "tag-bad" : "tag-warn"}`}>
          {item.status}
        </span>
        <a href={item.url} target="_blank" rel="noreferrer noopener" className="dead-url">
          {item.url}
        </a>
        <span className="dead-attempts">
          {item.attempts} attempt{item.attempts === 1 ? "" : "s"}
        </span>
      </div>
      {item.last_error && <p className="dead-error">{item.last_error}</p>}
      <div className="dead-actions">
        <button
          className="btn btn-ghost btn-sm"
          onClick={() => requeue.mutate()}
          disabled={requeue.isPending}
        >
          {requeue.isPending ? "Requeueing…" : "Requeue"}
        </button>
        {requeue.isError && (
          <span className="form-error">{errorMessage(requeue.error)}</span>
        )}
      </div>
    </li>
  );
}

/* --- Page ------------------------------------------------------------- */

export function Dashboard() {
  const { data: user, isLoading: authLoading } = useMe();

  // Gated on the flag so a non-admin who lands here does not sit in a
  // three-second 403 loop against an endpoint that will never answer. The
  // hooks still run unconditionally -- React requires that -- but `enabled`
  // stops the fetch rather than the render.
  const isAdmin = !!user?.is_admin;

  const overview = useQuery({
    queryKey: ["ops", "overview"],
    queryFn: opsApi.overview,
    refetchInterval: OPS_POLL_MS,
    enabled: isAdmin,
  });

  const dead = useQuery({
    queryKey: ["ops", "dead-letter"],
    queryFn: () => opsApi.deadLetter(25),
    refetchInterval: OPS_POLL_MS * 4,
    enabled: isAdmin,
  });

  if (authLoading) return null;
  if (!user) return <Navigate to="/signin" replace />;

  // Rendered rather than redirected. A signed-in user who reached this URL
  // deliberately is better served by being told why they cannot see it than
  // by being bounced somewhere else with no explanation -- and the server has
  // already refused, so nothing is being protected by hiding the page.
  if (!user.is_admin) {
    return (
      <main id="main" className="dashboard">
        <div className="container">
          <div className="alert" role="alert">
            <strong>Operator access required</strong>
            <p>
              This page reports system-wide state — queue depth, worker health,
              and failed jobs across every account — so it is limited to
              operators. Your account is signed in but does not have the flag.
            </p>
            <p>
              It is granted out of band; there is deliberately no endpoint that
              grants it.
            </p>
          </div>
        </div>
      </main>
    );
  }

  const data = overview.data;
  const problems = data?.queues.filter((q) => verdictFor(q).tone === "bad") ?? [];
  const totalDepth = data?.queues.reduce((sum, q) => sum + q.depth, 0) ?? 0;

  return (
    <main id="main" className="dashboard">
      <div className="container">
        <header className="dash-head">
          <div>
            <span className="eyebrow">
              <span className="pulse-dot" aria-hidden="true" />
              Operations
            </span>
            <h1 className="dash-title">System state</h1>
          </div>
          <p className="dash-meta">
            Polling every {OPS_POLL_MS / 1000}s
            {overview.isFetching && <span className="live-tick"> · refreshing</span>}
          </p>
        </header>

        {overview.isError && (
          <p className="form-error" role="alert">
            {errorMessage(overview.error, "Could not read system state.")}
          </p>
        )}

        {/* Surfaced above everything else. The whole point of this page is
          * that a stranded queue is silent — it produces no error, no retry,
          * and no log line, so it has to be the loudest thing on screen. */}
        {problems.length > 0 && (
          <div className="alert" role="alert">
            <strong>
              {problems.length} queue{problems.length === 1 ? "" : "s"} holding work
              nothing will run
            </strong>
            <p>
              {problems.map((q) => q.name).join(", ")} — jobs are waiting with no
              worker consuming them. They are not retrying and not failing.
            </p>
          </div>
        )}

        {data && (
          <>
            <div className="tiles">
              <div className="tile">
                <span className="tile-value">{totalDepth}</span>
                <span className="tile-label">jobs waiting</span>
              </div>
              <div className="tile">
                <span className="tile-value">{data.workers.length}</span>
                <span className="tile-label">workers alive</span>
              </div>
              {data.jobs_by_status.map((s) => (
                <div className="tile" key={s.status}>
                  <span className="tile-value">{s.count}</span>
                  <span className="tile-label">{s.status}</span>
                </div>
              ))}
            </div>

            <section className="dash-section">
              <h2 className="dash-h2">Queues</h2>
              <ul className="queue-grid">
                {data.queues.map((q) => (
                  <QueueCard key={q.name} queue={q} />
                ))}
              </ul>
            </section>

            <section className="dash-section">
              <h2 className="dash-h2">Workers</h2>
              {data.workers.length === 0 ? (
                <p className="dash-empty">
                  No workers registered. Nothing will be processed until one
                  starts — <code>docker compose up -d</code>.
                </p>
              ) : (
                <ul className="worker-list">
                  {data.workers.map((w) => (
                    <WorkerRow key={w.name} worker={w} />
                  ))}
                </ul>
              )}
            </section>
          </>
        )}

        <section className="dash-section">
          <h2 className="dash-h2">Dead letter</h2>
          {dead.data && dead.data.length > 0 ? (
            <ul className="dead-list">
              {dead.data.map((item) => (
                <DeadLetterRow key={item.id} item={item} />
              ))}
            </ul>
          ) : (
            <p className="dash-empty">
              Nothing failed or dead-lettered. Jobs land here after exhausting
              their retry budget.
            </p>
          )}
        </section>

        {data && (
          <p className="dash-footnote">
            Job timeout {data.job_timeout_seconds}s · results kept{" "}
            {Math.round(data.result_ttl_seconds / 3600)}h · failures kept{" "}
            {Math.round(data.failure_ttl_seconds / 86400)}d
          </p>
        )}
      </div>
    </main>
  );
}
