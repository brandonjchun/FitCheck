import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  opsApi: { overview: vi.fn(), deadLetter: vi.fn(), requeue: vi.fn() },
  errorMessage: (_: unknown, fallback = "Something went wrong.") => fallback,
}));

vi.mock("../hooks/useAuth", () => ({ useMe: vi.fn() }));

import { opsApi } from "../api/client";
import { useMe } from "../hooks/useAuth";
import { Dashboard } from "./Dashboard";

/* --- fixtures --------------------------------------------------------- */

const queue = (over: Partial<Record<string, unknown>> = {}) => ({
  name: "interactive",
  depth: 0,
  declared: true,
  started: 0,
  failed: 0,
  deferred: 0,
  worker_count: 1,
  ...over,
});

const overview = (over: Partial<Record<string, unknown>> = {}) => ({
  queues: [queue()],
  workers: [
    {
      name: "worker-abc123456789",
      state: "idle",
      queues: ["interactive", "scoring"],
      successful_jobs: 12,
      failed_jobs: 1,
    },
  ],
  jobs_by_status: [{ status: "succeeded", count: 12 }],
  job_timeout_seconds: 120,
  result_ttl_seconds: 3600,
  failure_ttl_seconds: 604800,
  sources: [],
  gate: { hits: 0, misses: 0, total: 0, hit_rate: 0 },
  ...over,
});

/** One source row, with the freshness columns overridable per test. */
const source = (over: Partial<Record<string, unknown>> = {}) => ({
  id: 1,
  kind: "lever",
  board_token: "acme",
  display_name: "Acme",
  enabled: true,
  crawl_interval_seconds: 3600,
  last_crawled_at: null,
  last_success_at: null,
  consecutive_failures: 0,
  circuit_open: false,
  seconds_since_success: null,
  is_stale: true,
  open_postings: 0,
  ...over,
});

function draw(user: unknown = { id: 1, email: "ops@fitcheck.dev", is_admin: true }) {
  vi.mocked(useMe).mockReturnValue({ data: user, isLoading: false } as never);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
  return { client, ...render(<Dashboard />, { wrapper }) };
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(opsApi.overview).mockResolvedValue(overview() as never);
  vi.mocked(opsApi.deadLetter).mockResolvedValue([] as never);
});

/* --- access ------------------------------------------------------------ */

describe("access", () => {
  it("explains the refusal instead of redirecting a non-admin", async () => {
    /* The server has already refused, so hiding the page protects nothing.
     * A signed-in user who typed this URL deliberately is better served by
     * being told why than by being bounced somewhere with no explanation. */
    draw({ id: 2, email: "user@fitcheck.dev", is_admin: false });

    expect(await screen.findByText(/operator access required/i)).toBeInTheDocument();
  });

  it("does not poll an endpoint that will only ever 403", async () => {
    /* Without the `enabled` gate a non-admin sits in a three-second loop
     * generating 403s forever. */
    draw({ id: 2, email: "user@fitcheck.dev", is_admin: false });

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
    expect(opsApi.overview).not.toHaveBeenCalled();
  });

  it("sends a signed-out visitor to sign in", () => {
    draw(null);

    expect(screen.queryByText(/system state/i)).not.toBeInTheDocument();
  });
});

/* --- the diagnosis ----------------------------------------------------- */

describe("queue verdicts", () => {
  /* The reason this page exists. Both failure states below are *silent*:
   * no retry, no failure, no log line, and a bare depth number looks
   * healthy. If the verdict logic breaks, the dashboard keeps rendering
   * and simply stops telling anyone. */

  it("flags an undeclared queue holding work as stranded", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({ queues: [queue({ name: "defualt", declared: false, depth: 7 })] }) as never,
    );
    draw();

    expect(await screen.findByText(/stranded/i)).toBeInTheDocument();
    expect(screen.getByText(/undeclared/i)).toBeInTheDocument();
  });

  it("flags a declared queue with work and no worker", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({ queues: [queue({ depth: 5, worker_count: 0 })] }) as never,
    );
    draw();

    expect(await screen.findByText(/no consumer/i)).toBeInTheDocument();
  });

  it("raises the alarm above everything else", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({ queues: [queue({ depth: 5, worker_count: 0 })] }) as never,
    );
    draw();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/holding work nothing will run/i);
  });

  it("does not cry wolf over an idle queue with no worker", async () => {
    /* Empty and unwatched is worth noting and is not an outage. Treating it
     * as one trains the reader to ignore the alarm. */
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({ queues: [queue({ depth: 0, worker_count: 0 })] }) as never,
    );
    draw();

    expect(await screen.findByText(/unwatched/i)).toBeInTheDocument();
    expect(screen.queryByText(/holding work nothing will run/i)).not.toBeInTheDocument();
  });

  it("calls a healthy queue draining, not failing", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({ queues: [queue({ depth: 3, worker_count: 2 })] }) as never,
    );
    draw();

    expect(await screen.findByText(/draining/i)).toBeInTheDocument();
  });

  it("warns on a backlog", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({ queues: [queue({ depth: 500, worker_count: 2 })] }) as never,
    );
    draw();

    expect(await screen.findByText(/backlog/i)).toBeInTheDocument();
  });

  it("sums depth across every queue", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({
        queues: [
          queue({ name: "interactive", depth: 2 }),
          queue({ name: "ingest", depth: 40 }),
        ],
      }) as never,
    );
    draw();

    expect(await screen.findByText("42")).toBeInTheDocument();
  });
});

/* --- workers ----------------------------------------------------------- */

describe("workers", () => {
  it("says plainly when nothing will be processed", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(overview({ workers: [] }) as never);
    draw();

    expect(await screen.findByText(/no workers registered/i)).toBeInTheDocument();
  });

  it("preserves queue order, which is the priority order", async () => {
    /* RQ checks a worker's queues left to right every time it looks for
     * work, so the first name is the one that gets priority. Sorting these
     * for tidiness would misreport the scheduling behaviour. */
    draw();

    const row = await screen.findByText(/12 ok/i);
    const tags = row.closest("li")!.querySelectorAll(".worker-queues .tag");

    expect(Array.from(tags, (t) => t.textContent?.replace("→", ""))).toEqual([
      "interactive",
      "scoring",
    ]);
  });
});

/* --- dead letter ------------------------------------------------------- */

describe("dead letter", () => {
  const item = {
    id: 42,
    status: "dead",
    url: "https://boards.example.com/jobs/1",
    attempts: 3,
    last_error: "PermanentFetchError: HTTP 404",
  };

  it("shows the error and attempt count", async () => {
    /* The two things an operator needs before deciding to requeue. */
    vi.mocked(opsApi.deadLetter).mockResolvedValue([item] as never);
    draw();

    expect(await screen.findByText(/3 attempts/i)).toBeInTheDocument();
    expect(screen.getByText(/HTTP 404/)).toBeInTheDocument();
  });

  it("requeues and refreshes both views", async () => {
    /* Requeue changes queue depth as well as the dead list, so leaving the
     * overview stale would show the job in neither place until the next
     * poll. */
    vi.mocked(opsApi.deadLetter).mockResolvedValue([item] as never);
    vi.mocked(opsApi.requeue).mockResolvedValue({ queue: "interactive" } as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /requeue/i }));

    await waitFor(() => expect(opsApi.requeue).toHaveBeenCalledWith(42));
    await waitFor(() => expect(opsApi.overview).toHaveBeenCalledTimes(2));
  });

  it("reports a failed requeue rather than looking successful", async () => {
    vi.mocked(opsApi.deadLetter).mockResolvedValue([item] as never);
    vi.mocked(opsApi.requeue).mockRejectedValue(new Error("409"));
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /requeue/i }));

    expect(await screen.findByText(/something went wrong/i)).toBeInTheDocument();
  });

  it("says nothing is wrong when the list is empty", async () => {
    draw();

    expect(await screen.findByText(/nothing failed or dead-lettered/i)).toBeInTheDocument();
  });
});

/* --- policy ------------------------------------------------------------ */

describe("policy footnote", () => {
  it("reads the constants from the server rather than hardcoding them", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({ job_timeout_seconds: 300, result_ttl_seconds: 7200 }) as never,
    );
    draw();

    expect(await screen.findByText(/job timeout 300s/i)).toBeInTheDocument();
    expect(screen.getByText(/results kept 2h/i)).toBeInTheDocument();
  });
});

/* --- source freshness -------------------------------------------------- */

describe("sources", () => {
  it("reports a recent success as fresh", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({
        sources: [source({ seconds_since_success: 120, is_stale: false })],
      }) as never,
    );
    draw();

    expect(await screen.findByText(/fresh — 2m ago/i)).toBeInTheDocument();
  });

  it("reports a source that has never succeeded rather than calling it fresh", async () => {
    /* Unknown age must not read as healthy — otherwise the board that has
     * never once worked is the one the dashboard is quietest about. */
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({ sources: [source()] }) as never,
    );
    draw();

    expect(await screen.findByText(/never crawled successfully/i)).toBeInTheDocument();
  });

  it("says circuit open rather than stale when a board is broken", async () => {
    /* The circuit explains the staleness. Reporting "stale" alone sends the
     * reader looking for a scheduling problem that is not there. */
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({
        sources: [
          source({
            circuit_open: true,
            consecutive_failures: 5,
            is_stale: true,
            seconds_since_success: 90000,
          }),
        ],
      }) as never,
    );
    draw();

    expect(await screen.findByText(/circuit open/i)).toBeInTheDocument();
    expect(screen.queryByText(/^stale/i)).not.toBeInTheDocument();
  });

  it("marks a stale source without dressing it as a failure", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({
        sources: [source({ is_stale: true, seconds_since_success: 90000 })],
      }) as never,
    );
    draw();

    expect(await screen.findByText(/stale — last success 25h ago/i)).toBeInTheDocument();
  });

  it("reports the gate hit rate when there is data", async () => {
    vi.mocked(opsApi.overview).mockResolvedValue(
      overview({
        gate: { hits: 3, misses: 1, total: 4, hit_rate: 0.75 },
      }) as never,
    );
    draw();

    expect(await screen.findByText(/75% of 4 ingests/i)).toBeInTheDocument();
  });

  it("says nothing about the gate before any ingest has happened", async () => {
    /* A 0% hit rate over zero ingests is not a measurement. */
    draw();

    expect(await screen.findByText(/no sources configured/i)).toBeInTheDocument();
    expect(screen.queryByText(/ingests skipped/i)).not.toBeInTheDocument();
  });
});
