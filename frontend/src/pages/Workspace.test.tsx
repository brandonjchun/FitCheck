import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  profileApi: {
    upload: vi.fn(),
    get: vi.fn(),
    list: vi.fn(),
    activate: vi.fn(),
    remove: vi.fn(),
    reextract: vi.fn(),
  },
  jobApi: { submit: vi.fn(), get: vi.fn(), list: vi.fn() },
  batchApi: { create: vi.fn(), get: vi.fn(), list: vi.fn() },
  errorMessage: (_: unknown, fallback = "Something went wrong.") => fallback,
}));

vi.mock("../hooks/useAuth", () => ({ useMe: vi.fn() }));

/* The feed is the subject of MatchFeed.test.tsx and needs its own API
 * surface. Rendering it here would mean stubbing matchApi for every test in
 * this file to assert nothing about it. */
vi.mock("../components/MatchFeed", () => ({
  MatchFeed: ({ profileId }: { profileId: number }) => (
    <div data-testid="match-feed">feed for {profileId}</div>
  ),
}));

import { batchApi, jobApi, profileApi } from "../api/client";
import { useMe } from "../hooks/useAuth";
import { Workspace } from "./Workspace";

/* --- fixtures --------------------------------------------------------- */

const summary = (over: Partial<Record<string, unknown>> = {}) => ({
  id: 1,
  filename: "resume.pdf",
  characters: 3997,
  created_at: "2026-08-01T10:00:00Z",
  is_active: true,
  extraction_ok: true,
  seniority: "mid",
  years_experience: 4.3,
  skill_count: 30,
  ...over,
});

const profile = (over: Partial<Record<string, unknown>> = {}) => ({
  ...summary(),
  raw_text: "…",
  skills: [],
  ...over,
});

const job = (over: Partial<Record<string, unknown>> = {}) => ({
  id: 100,
  profile_id: 1,
  url: "https://boards.example.com/jobs/1",
  status: "queued",
  attempts: 0,
  last_error: null,
  created_at: "2026-08-01T10:00:00Z",
  updated_at: "2026-08-01T10:00:00Z",
  is_terminal: false,
  ...over,
});

const batch = (over: Partial<Record<string, unknown>> = {}) => ({
  id: 7,
  profile_id: 1,
  filename: "urls.txt",
  total: 10,
  rejected: 0,
  duplicates: 0,
  created_at: "2026-08-01T10:00:00Z",
  counts: { succeeded: 4, failed: 1, dead: 1, queued: 4 },
  is_complete: false,
  ...over,
});

/* Both list rows and the per-row refetch, together.
 *
 * `JobRow` and `BatchRow` seed themselves from the list row via `initialData`
 * and then verify it with a fetch by id. Mocking only the list leaves that
 * fetch resolving `undefined`, which React Query rejects -- so the row would
 * be asserted against data the component was in the middle of discarding. */
function showJobs(...rows: ReturnType<typeof job>[]) {
  vi.mocked(jobApi.list).mockResolvedValue(rows as never);
  vi.mocked(jobApi.get).mockImplementation(
    ((id: number) =>
      Promise.resolve(rows.find((r) => r.id === id) ?? rows[0])) as never,
  );
}

function showBatches(...rows: ReturnType<typeof batch>[]) {
  vi.mocked(batchApi.list).mockResolvedValue(rows as never);
  vi.mocked(batchApi.get).mockImplementation(
    ((id: number) =>
      Promise.resolve(rows.find((r) => r.id === id) ?? rows[0])) as never,
  );
}

function draw(user: unknown = { id: 1, email: "dev@fitcheck.dev" }) {
  vi.mocked(useMe).mockReturnValue({ data: user, isLoading: false } as never);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
  return render(<Workspace />, { wrapper });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(profileApi.list).mockResolvedValue([summary()] as never);
  vi.mocked(profileApi.get).mockResolvedValue(profile() as never);
  vi.mocked(jobApi.list).mockResolvedValue([] as never);
  vi.mocked(batchApi.list).mockResolvedValue([] as never);
});

/* --- which resume is open ---------------------------------------------- */

describe("choosing a resume", () => {
  it("sends a signed-out visitor to sign in", () => {
    draw(null);

    expect(screen.queryByText(/your resume, structured/i)).not.toBeInTheDocument();
  });

  it("opens the active resume, since that is the one the feed uses", async () => {
    vi.mocked(profileApi.list).mockResolvedValue([
      summary({ id: 1, filename: "old.pdf", is_active: false }),
      summary({ id: 2, filename: "current.pdf", is_active: true }),
    ] as never);
    draw();

    await waitFor(() => expect(profileApi.get).toHaveBeenCalledWith(2));
  });

  it("falls back to the newest upload when none is active", async () => {
    vi.mocked(profileApi.list).mockResolvedValue([
      summary({ id: 5, filename: "a.pdf", is_active: false }),
      summary({ id: 6, filename: "b.pdf", is_active: false }),
    ] as never);
    draw();

    await waitFor(() => expect(profileApi.get).toHaveBeenCalledWith(5));
  });

  it("opens the one clicked, over the active one", async () => {
    vi.mocked(profileApi.list).mockResolvedValue([
      summary({ id: 1, filename: "old.pdf", is_active: false }),
      summary({ id: 2, filename: "current.pdf", is_active: true }),
    ] as never);
    draw();
    await waitFor(() => expect(profileApi.get).toHaveBeenCalledWith(2));

    await userEvent.click(await screen.findByRole("button", { name: /old\.pdf/i }));

    await waitFor(() => expect(profileApi.get).toHaveBeenCalledWith(1));
  });

  it("locks the posting panel until there is something to score against", async () => {
    vi.mocked(profileApi.list).mockResolvedValue([] as never);
    draw();

    expect(await screen.findByText(/upload a resume first/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/job posting url/i)).not.toBeInTheDocument();
    expect(screen.queryByTestId("match-feed")).not.toBeInTheDocument();
  });

  it("says a new upload is kept as a version rather than replacing the last", async () => {
    /* Upload does not promote, so without saying so the second upload looks
     * like it silently did nothing. */
    draw();

    expect(await screen.findByText(/a new upload is kept as a version/i)).toBeInTheDocument();
  });
});

/* --- resume versions --------------------------------------------------- */

describe("resume versions", () => {
  it("promotes a version and refreshes the list", async () => {
    vi.mocked(profileApi.list).mockResolvedValue([
      summary({ id: 1, filename: "old.pdf", is_active: false }),
      summary({ id: 2, filename: "current.pdf", is_active: true }),
    ] as never);
    vi.mocked(profileApi.activate).mockResolvedValue(profile({ id: 1 }) as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /use this/i }));

    await waitFor(() => expect(profileApi.activate).toHaveBeenCalledWith(1));
    await waitFor(() => expect(profileApi.list).toHaveBeenCalledTimes(2));
  });

  it("offers no promote control on the resume that is already active", async () => {
    draw();

    await screen.findByText("resume.pdf");
    expect(screen.queryByRole("button", { name: /use this/i })).not.toBeInTheDocument();
  });

  it("does not delete on the first click", async () => {
    /* Deleting a resume cascades to every job submitted against it and there
     * is no undo, so the destructive action is deliberately two steps. */
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /^delete$/i }));

    expect(profileApi.remove).not.toHaveBeenCalled();
    expect(screen.getByText(/deletes this resume and every job/i)).toBeInTheDocument();
  });

  it("deletes on confirmation", async () => {
    vi.mocked(profileApi.remove).mockResolvedValue(undefined as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /^delete$/i }));
    await userEvent.click(screen.getByRole("button", { name: /confirm/i }));

    await waitFor(() => expect(profileApi.remove).toHaveBeenCalledWith(1));
  });

  it("lets the user back out", async () => {
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /^delete$/i }));
    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));

    expect(screen.queryByRole("button", { name: /confirm/i })).not.toBeInTheDocument();
    expect(profileApi.remove).not.toHaveBeenCalled();
  });

  it("does not keep pointing at a resume that was just deleted", async () => {
    /* The selection outlives the row unless it is cleared, and the page would
     * then fetch an id the server 404s. */
    vi.mocked(profileApi.list).mockResolvedValue([
      summary({ id: 1, filename: "one.pdf", is_active: false }),
      summary({ id: 2, filename: "two.pdf", is_active: true }),
    ] as never);
    vi.mocked(profileApi.remove).mockResolvedValue(undefined as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /one\.pdf/i }));
    await waitFor(() => expect(profileApi.get).toHaveBeenCalledWith(1));

    const row = screen.getByRole("button", { name: /one\.pdf/i }).closest("li")!;
    await userEvent.click(within(row).getByRole("button", { name: /^delete$/i }));

    vi.mocked(profileApi.list).mockResolvedValue([
      summary({ id: 2, filename: "two.pdf", is_active: true }),
    ] as never);
    await userEvent.click(within(row).getByRole("button", { name: /confirm/i }));

    await waitFor(() => expect(profileApi.get).toHaveBeenLastCalledWith(2));
  });
});

/* --- extraction state -------------------------------------------------- */

describe("the profile panel", () => {
  it("shows the structured result once extraction lands", async () => {
    vi.mocked(profileApi.get).mockResolvedValue(
      profile({
        skills: [
          { name: "Python", years: 3, source: "experience", evidence: "Built X in Python" },
        ],
      }) as never,
    );
    draw();

    expect(await screen.findByText("Python")).toBeInTheDocument();
    expect(screen.getByText("Built X in Python")).toBeInTheDocument();
    expect(screen.getByText("3y")).toBeInTheDocument();
    // The provenance of a skill is the whole reason the score is defensible.
    expect(screen.getByText("experience")).toBeInTheDocument();
  });

  it("says extraction is running rather than showing an empty profile", async () => {
    vi.mocked(profileApi.get).mockResolvedValue(
      profile({ extraction_ok: false, skills: [] }) as never,
    );
    draw();

    expect(await screen.findByText(/deriving your structured profile/i)).toBeInTheDocument();
  });

  it("distinguishes 'found no skills' from 'has not run yet'", async () => {
    /* An empty skill list after a successful extraction is a real result --
     * the prompt refuses to list a skill it cannot quote -- and rendering it
     * the same as a pending one would report a working system as broken. */
    vi.mocked(profileApi.get).mockResolvedValue(
      profile({ extraction_ok: true, skills: [] }) as never,
    );
    draw();

    expect(await screen.findByText(/refuses to list a skill it cannot quote/i)).toBeInTheDocument();
    expect(screen.queryByText(/deriving your structured profile/i)).not.toBeInTheDocument();
  });

  it("reports a resume it could not load instead of spinning", async () => {
    vi.mocked(profileApi.get).mockRejectedValue(new Error("404"));
    draw();

    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});

/* --- job submission ---------------------------------------------------- */

describe("submitting a posting", () => {
  it("queues a URL and clears the box", async () => {
    vi.mocked(jobApi.submit).mockResolvedValue(job() as never);
    draw();

    const input = await screen.findByLabelText("Job posting URL");
    await userEvent.type(input, "https://boards.example.com/jobs/1");
    await userEvent.click(screen.getByRole("button", { name: /queue it/i }));

    await waitFor(() =>
      expect(jobApi.submit).toHaveBeenCalledWith(1, "https://boards.example.com/jobs/1"),
    );
    await waitFor(() => expect(input).toHaveValue(""));
  });

  it("refuses to submit an empty box", async () => {
    draw();

    await screen.findByLabelText("Job posting URL");
    expect(screen.getByRole("button", { name: /^queue it$/i })).toBeDisabled();
  });

  it("surfaces a rejection rather than looking queued", async () => {
    vi.mocked(jobApi.submit).mockRejectedValue(new Error("422"));
    draw();

    await userEvent.type(
      await screen.findByLabelText("Job posting URL"),
      "https://example.com/x",
    );
    await userEvent.click(screen.getByRole("button", { name: /queue it/i }));

    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});

describe("job status", () => {
  /* The lifecycle spec section 6.4 defines, rendered. `failed` and `dead` are
   * the pair that matter: one is a job that will come back, the other is a job
   * that will not, and showing them the same way loses the distinction the
   * retry policy exists to create. */

  it("says a failed job will retry, because it will", async () => {
    showJobs(job({ status: "failed", attempts: 1, is_terminal: false }));
    draw();

    expect(await screen.findByText(/failed — will retry/i)).toBeInTheDocument();
  });

  it("keeps watching a failed job, since the retry is what the user is waiting for", async () => {
    /* `failed` was in the API's terminal set until an integration test drove a
     * real RQ worker through a transient failure. Polling stopped on the first
     * error, so the row froze on "will retry" and never showed the retry
     * succeeding seconds later -- the label and the polling rule disagreed. */
    vi.mocked(jobApi.list).mockResolvedValue([
      job({ status: "failed", attempts: 1, is_terminal: false }),
    ] as never);
    vi.mocked(jobApi.get).mockResolvedValue(
      job({ status: "succeeded", attempts: 2, is_terminal: true }) as never,
    );
    draw();

    expect(await screen.findByText(/succeeded/i)).toBeInTheDocument();
  });

  it("stops polling a job that has genuinely finished", async () => {
    /* One reconciling fetch is expected -- the list row is initialData and
     * React Query verifies it once. What must not happen is a second one:
     * a dead job never changes again, and a page left open would otherwise
     * ask about it every two seconds forever. */
    showJobs(job({ status: "dead", attempts: 4, is_terminal: true, last_error: "HTTP 404" }));
    draw();

    await screen.findByText(/dead-lettered/i);
    await waitFor(() => expect(jobApi.get).toHaveBeenCalledTimes(1));

    await new Promise((r) => setTimeout(r, 2500)); // comfortably past POLL_MS
    expect(jobApi.get).toHaveBeenCalledTimes(1);
  });

  it("keeps polling a job that has not finished", async () => {
    /* The counterpart, and the reason the test above is not vacuous: without
     * it, a component that never polls at all would pass. */
    showJobs(job({ status: "queued" }));
    draw();

    await waitFor(
      () => expect(vi.mocked(jobApi.get).mock.calls.length).toBeGreaterThan(1),
      { timeout: 4000 },
    );
  });

  it("shows the attempt count and the error", async () => {
    showJobs(
      job({
        status: "dead",
        attempts: 4,
        is_terminal: true,
        last_error: "PermanentFetchError: HTTP 404",
      }),
    );
    draw();

    expect(await screen.findByText(/attempt 4/i)).toBeInTheDocument();
    expect(screen.getByText(/HTTP 404/)).toBeInTheDocument();
  });

  it("renders an unknown status rather than a blank row", async () => {
    /* The status column is plain text in Postgres and nothing at the database
     * level rejects a typo, so the UI must not disappear on one. */
    showJobs(job({ status: "quarantined" as never, is_terminal: true }));
    draw();

    expect(await screen.findByText(/quarantined/i)).toBeInTheDocument();
  });
});

/* --- bulk submit ------------------------------------------------------- */

describe("bulk submit", () => {
  it("counts non-blank lines, not newlines", async () => {
    /* `split("\n").length` reports 1 for an empty box and counts the trailing
     * newline every textarea ends with, so the cap warning would fire a line
     * early and the number shown would not be the number acted on. */
    draw();

    const box = await screen.findByLabelText(/job posting urls/i);
    await userEvent.type(box, "https://a.example/1\n\nhttps://b.example/2\n");

    expect(screen.getByText("2 lines")).toBeInTheDocument();
  });

  it("submits the pasted list", async () => {
    vi.mocked(batchApi.create).mockResolvedValue({
      id: 7,
      accepted: 2,
      rejected: 0,
      duplicates: 0,
    } as never);
    draw();

    await userEvent.type(
      await screen.findByLabelText(/job posting urls/i),
      "https://a.example/1\nhttps://b.example/2",
    );
    await userEvent.click(screen.getByRole("button", { name: /submit urls/i }));

    await waitFor(() =>
      expect(batchApi.create).toHaveBeenCalledWith(
        1,
        "https://a.example/1\nhttps://b.example/2",
      ),
    );
  });

  it("accounts for every line the user sent", async () => {
    /* accepted + rejected + duplicates equals their line count. A batch that
     * quietly ingests 500 of 4,000 lines is worse than one that refuses,
     * because they cannot tell which 3,500 are missing. */
    vi.mocked(batchApi.create).mockResolvedValue({
      id: 7,
      accepted: 1,
      rejected: 1,
      duplicates: 1,
    } as never);
    draw();

    await userEvent.type(await screen.findByLabelText(/job posting urls/i), "https://a.example/1");
    await userEvent.click(screen.getByRole("button", { name: /submit urls/i }));

    expect(await screen.findByText(/1 queued/i)).toBeInTheDocument();
    expect(screen.getByText(/could not be read as a url/i)).toBeInTheDocument();
    expect(screen.getByText(/1 already submitted/i)).toBeInTheDocument();
  });

  it("refuses a list over the cap client-side rather than letting the server reject it", async () => {
    draw();

    const box = await screen.findByLabelText(/job posting urls/i);
    // 501 lines: one past settings.max_urls_per_batch.
    const many = Array.from({ length: 501 }, (_, i) => `https://a.example/${i}`).join("\n");
    await userEvent.click(box);
    await userEvent.paste(many);

    expect(screen.getByText(/over the 500 limit/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /submit urls/i })).toBeDisabled();
  });

  it("reports a refused list", async () => {
    vi.mocked(batchApi.create).mockRejectedValue(new Error("413"));
    draw();

    await userEvent.type(await screen.findByLabelText(/job posting urls/i), "https://a.example/1");
    await userEvent.click(screen.getByRole("button", { name: /submit urls/i }));

    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});

describe("batch progress", () => {
  it("scales the bar off the batch's own total, not the sum of the counts", async () => {
    /* If a job row went missing the bar should come up short and show it.
     * Rescaling to the counts would make an incomplete batch look finished. */
    showBatches(batch());
    draw();

    const bar = await screen.findByRole("img", { name: /4 of 10 fetched/i });
    const [ok, warn, bad] = Array.from(bar.querySelectorAll("span"));

    expect(ok).toHaveStyle({ width: "40%" });
    expect(warn).toHaveStyle({ width: "10%" });
    expect(bad).toHaveStyle({ width: "10%" });
  });

  it("says how many are left while work is in flight", async () => {
    showBatches(batch());
    draw();

    expect(await screen.findByText("4 left")).toBeInTheDocument();
  });

  it("distinguishes retrying from gave up", async () => {
    showBatches(batch());
    draw();

    expect(await screen.findByText(/1 retrying/)).toBeInTheDocument();
    expect(screen.getByText(/1 gave up/)).toBeInTheDocument();
  });

  it("reports lines it could not use", async () => {
    showBatches(batch({ rejected: 3, duplicates: 1 }));
    draw();

    expect(await screen.findByText(/3 unreadable · 1 duplicate skipped/i)).toBeInTheDocument();
  });

  it("reports a finished batch as complete rather than as 0 left", async () => {
    /* The polling rule itself is the same machinery as JobRow's and is
     * exercised there; what is specific here is that `is_complete` comes from
     * the server rather than being inferred from the counts, so a batch whose
     * jobs all went dead still reads as finished. */
    showBatches(batch({ is_complete: true, counts: { dead: 10 } }));
    draw();

    expect(await screen.findByText("Complete")).toBeInTheDocument();
    expect(screen.queryByText(/left$/)).not.toBeInTheDocument();
  });

  it("explains the empty state rather than showing a bare panel", async () => {
    draw();

    expect(await screen.findByText(/nothing submitted in bulk yet/i)).toBeInTheDocument();
  });
});

/* --- the feed ---------------------------------------------------------- */

describe("the match feed", () => {
  it("is given the open resume, not the active one", async () => {
    /* The feed has to describe what the page is showing. Handing it the
     * active id while a different version is open would put a breakdown
     * against one resume under the skills of another. */
    vi.mocked(profileApi.list).mockResolvedValue([
      summary({ id: 1, filename: "old.pdf", is_active: false }),
      summary({ id: 2, filename: "current.pdf", is_active: true }),
    ] as never);
    draw();

    expect(await screen.findByTestId("match-feed")).toHaveTextContent("feed for 2");

    await userEvent.click(screen.getByRole("button", { name: /old\.pdf/i }));

    await waitFor(() =>
      expect(screen.getByTestId("match-feed")).toHaveTextContent("feed for 1"),
    );
  });
});
