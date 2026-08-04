import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  insightsApi: { skillGaps: vi.fn() },
  errorMessage: (_: unknown, fallback = "Something went wrong.") => fallback,
}));

vi.mock("../hooks/useAuth", () => ({ useMe: vi.fn() }));

import { insightsApi } from "../api/client";
import { useMe } from "../hooks/useAuth";
import { Insights } from "./Insights";

const gap = (over: Partial<Record<string, unknown>> = {}) => ({
  name: "Rust",
  missing: 0,
  partial: 0,
  matched: 0,
  blocking: 0,
  ...over,
});

const report = (over: Partial<Record<string, unknown>> = {}) => ({
  profile_id: null,
  matches_analyzed: 10,
  gaps: [],
  ...over,
});

function draw(user: unknown = { id: 1, email: "u@fitcheck.dev", is_admin: false }) {
  vi.mocked(useMe).mockReturnValue({ data: user, isLoading: false } as never);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <MemoryRouter>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </MemoryRouter>
  );
  return render(<Insights />, { wrapper });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(insightsApi.skillGaps).mockResolvedValue(report() as never);
});

describe("access", () => {
  it("does not query on behalf of a signed-out visitor", () => {
    draw(null);
    expect(insightsApi.skillGaps).not.toHaveBeenCalled();
  });
});

describe("skill gaps", () => {
  it("shows the denominator alongside the counts", async () => {
    /* "missing in 9" is unreadable without it -- 9 of 10 and 9 of 400 are
     * opposite findings. */
    vi.mocked(insightsApi.skillGaps).mockResolvedValue(
      report({
        matches_analyzed: 40,
        gaps: [gap({ missing: 9, blocking: 9 })],
      }) as never,
    );
    draw();

    expect(await screen.findByText(/blocks 9 of 40/i)).toBeInTheDocument();
  });

  it("renders every bucket, not only the bad news", async () => {
    /* A skill satisfied in half the postings is a different problem from one
     * satisfied in none, and a missing-only view cannot tell them apart. */
    vi.mocked(insightsApi.skillGaps).mockResolvedValue(
      report({ gaps: [gap({ missing: 3, partial: 2, matched: 5 })] }) as never,
    );
    draw();

    expect(await screen.findByText("3 missing")).toBeInTheDocument();
    expect(screen.getByText("2 partial")).toBeInTheDocument();
    expect(screen.getByText("5 matched")).toBeInTheDocument();
  });

  it("states what the ranking is by", async () => {
    /* Left implicit, a reader assumes raw frequency and misreads row two. */
    vi.mocked(insightsApi.skillGaps).mockResolvedValue(
      report({ gaps: [gap({ missing: 1, blocking: 1 })] }) as never,
    );
    draw();

    expect(await screen.findByText(/list the skill as/i)).toBeInTheDocument();
  });

  it("omits the blocking badge when nothing is blocked", async () => {
    vi.mocked(insightsApi.skillGaps).mockResolvedValue(
      report({ gaps: [gap({ partial: 4, blocking: 0 })] }) as never,
    );
    draw();

    expect(await screen.findByText("4 partial")).toBeInTheDocument();
    expect(screen.queryByText(/blocks/i)).not.toBeInTheDocument();
  });

  it("distinguishes nothing-scored-yet from nothing-missing", async () => {
    /* Two empty states with opposite meanings. Telling somebody with 40
     * matches to go upload a resume would be nonsense. */
    vi.mocked(insightsApi.skillGaps).mockResolvedValue(
      report({ matches_analyzed: 0, gaps: [] }) as never,
    );
    draw();
    expect(await screen.findByText(/nothing scored yet/i)).toBeInTheDocument();
  });

  it("says so when the feed has matches but no gaps", async () => {
    vi.mocked(insightsApi.skillGaps).mockResolvedValue(
      report({ matches_analyzed: 12, gaps: [] }) as never,
    );
    draw();
    expect(await screen.findByText(/every requirement/i)).toBeInTheDocument();
  });
});
