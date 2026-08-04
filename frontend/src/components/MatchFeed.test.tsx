import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  matchApi: {
    list: vi.fn(),
    feedback: vi.fn(),
    recommend: vi.fn(),
  },
  errorMessage: (_: unknown, fallback = "Something went wrong.") => fallback,
}));

import { matchApi } from "../api/client";
import { MatchFeed } from "./MatchFeed";

/* --- fixtures --------------------------------------------------------- */

const skill = (over: Partial<Record<string, unknown>> = {}) => ({
  name: "Python",
  necessity: "required",
  bucket: "matched",
  required_years: null,
  candidate_years: null,
  evidence: null,
  ...over,
});

const match = (over: Partial<Record<string, unknown>> = {}) => ({
  id: 1,
  profile_id: 1,
  job_posting_id: 10,
  semantic_score: 0.62,
  skill_score: 0.8,
  final_score: 0.73,
  origin: "recommendation",
  scorer_version: 3,
  scored_at: "2026-08-01T10:00:00Z",
  counts: { matched: 4, partial: 1, missing: 2, missing_required: 0 },
  skills: [],
  weights: { semantic: 0.4, skill: 0.6 },
  extraction_failed: false,
  posting_title: "Senior Backend Engineer",
  posting_company: "Acme",
  posting_url: "https://boards.example.com/jobs/1",
  ...over,
});

function draw(profileId = 1) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  return render(<MatchFeed profileId={profileId} />, { wrapper });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(matchApi.list).mockResolvedValue([] as never);
});

/* --- the explanation --------------------------------------------------- */

describe("the score, and what it was made of", () => {
  /* A blended number on its own is unfalsifiable. Everything here is about
   * the feed staying inspectable -- if these break, the product degrades to
   * a percentage nobody can argue with. */

  it("shows both sub-scores alongside the blend, not just the blend", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([match()] as never);
    draw();

    expect(await screen.findByText("73%")).toBeInTheDocument();
    const subs = screen.getByText(/semantic/i);
    expect(subs).toHaveTextContent("62%");
    expect(subs).toHaveTextContent("80%");
  });

  it("states the blend weights rather than leaving 73% unexplained", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([match()] as never);
    draw();

    expect(await screen.findByText(/weighted 0.4 \/ 0.6/i)).toBeInTheDocument();
  });

  it("counts the three buckets", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([match()] as never);
    draw();

    expect(await screen.findByText("4 matched")).toBeInTheDocument();
    expect(screen.getByText("1 partial")).toBeInTheDocument();
    expect(screen.getByText("2 missing")).toBeInTheDocument();
  });

  it("calls out missing *required* skills separately from missing ones", async () => {
    /* The blocker chip is the difference between "you are short a nice-to-have"
     * and "you do not qualify". Folding it into the missing count loses that. */
    vi.mocked(matchApi.list).mockResolvedValue([
      match({ counts: { matched: 1, partial: 0, missing: 3, missing_required: 2 } }),
    ] as never);
    draw();

    expect(await screen.findByText("2 required missing")).toBeInTheDocument();
  });

  it("omits the blocker chip when nothing required is missing", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([match()] as never);
    draw();

    await screen.findByText("4 matched");
    expect(screen.queryByText(/required missing/i)).not.toBeInTheDocument();
  });

  it("says when a score is semantic-only rather than letting it pass as normal", async () => {
    /* Without this an unparseable posting yields a score that looks like any
     * other one that happened to find no skill overlap. */
    vi.mocked(matchApi.list).mockResolvedValue([
      match({ extraction_failed: true }),
    ] as never);
    draw();

    expect(
      await screen.findByText(/could not be parsed into structured requirements/i),
    ).toBeInTheDocument();
  });
});

describe("the breakdown", () => {
  const withSkills = () =>
    match({
      skills: [
        skill({ name: "Kubernetes", bucket: "matched", necessity: "preferred" }),
        skill({
          name: "Go",
          bucket: "missing",
          necessity: "required",
          evidence: "5+ years of production Go",
        }),
        skill({
          name: "React",
          bucket: "partial",
          necessity: "required",
          required_years: 5,
          candidate_years: 2,
        }),
      ],
    });

  it("is collapsed until asked for, and names how much is behind it", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([withSkills()] as never);
    draw();

    const toggle = await screen.findByRole("button", { name: /why this score \(3\)/i });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Kubernetes")).not.toBeInTheDocument();
  });

  it("opens and closes", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([withSkills()] as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /why this score/i }));
    expect(screen.getByText("Kubernetes")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /hide breakdown/i }));
    expect(screen.queryByText("Kubernetes")).not.toBeInTheDocument();
  });

  it("puts a missing required skill first, whatever order the API sent", async () => {
    /* The ordering *is* the advice. A missing requirement is the thing that
     * decides, so burying it under satisfied nice-to-haves inverts the point
     * of showing a breakdown at all. The fixture deliberately lists it second. */
    vi.mocked(matchApi.list).mockResolvedValue([withSkills()] as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /why this score/i }));

    const names = screen
      .getAllByRole("listitem")
      .filter((li) => li.className.includes("mf-skill"))
      .map((li) => li.querySelector(".mf-skill-name")?.textContent);

    expect(names[0]).toMatch(/^Go/);
    // Partial before satisfied; the matched preferred one comes last.
    expect(names[1]).toMatch(/^React/);
    expect(names[2]).toMatch(/^Kubernetes/);
  });

  it("shows years as a shortfall when the posting states a requirement", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([withSkills()] as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /why this score/i }));

    expect(screen.getByText("2 of 5 yrs")).toBeInTheDocument();
  });

  it("quotes the posting's own words for a skill", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([withSkills()] as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /why this score/i }));

    expect(screen.getByText("5+ years of production Go")).toBeInTheDocument();
  });

  it("offers no toggle when there is nothing to explain", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([match({ skills: [] })] as never);
    draw();

    await screen.findByText("4 matched");
    expect(screen.queryByRole("button", { name: /why this score/i })).not.toBeInTheDocument();
  });
});

describe("posting identity", () => {
  it("links to the posting when there is a URL", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([match()] as never);
    draw();

    const link = await screen.findByRole("link", { name: /senior backend engineer/i });
    expect(link).toHaveAttribute("href", "https://boards.example.com/jobs/1");
    // noopener matters: target=_blank without it hands window.opener to a
    // third-party page we did not write.
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });

  it("does not render a dead link when the posting has no URL", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([match({ posting_url: null })] as never);
    draw();

    expect(await screen.findByText(/senior backend engineer/i)).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /senior backend/i })).not.toBeInTheDocument();
  });

  it("falls back to a placeholder rather than rendering an empty heading", async () => {
    /* Greenhouse postings had no board title for a while and extraction can
     * still return null, so this path is reachable in real data. */
    vi.mocked(matchApi.list).mockResolvedValue([
      match({ posting_title: null, posting_company: null }),
    ] as never);
    draw();

    expect(await screen.findByText("Untitled posting")).toBeInTheDocument();
  });
});

/* --- filters ----------------------------------------------------------- */

describe("the filter bar", () => {
  it("passes a chosen origin to the API", async () => {
    draw();
    await waitFor(() => expect(matchApi.list).toHaveBeenCalled());

    await userEvent.selectOptions(screen.getByLabelText(/source/i), "recommendation");

    await waitFor(() =>
      expect(matchApi.list).toHaveBeenLastCalledWith(1, 25, { origin: "recommendation" }),
    );
  });

  it("sends seniority as a list, which is what the endpoint reads", async () => {
    /* FastAPI declares this as list[str]. A bare string arrives as a
     * differently-shaped parameter and is silently ignored -- the filter
     * would appear to do nothing. */
    draw();
    await waitFor(() => expect(matchApi.list).toHaveBeenCalled());

    await userEvent.selectOptions(screen.getByLabelText(/seniority/i), "senior");

    await waitFor(() =>
      expect(matchApi.list).toHaveBeenLastCalledWith(1, 25, { seniority: ["senior"] }),
    );
  });

  it("clears a filter to undefined rather than to an empty string", async () => {
    /* `origin: ""` is a filter for postings whose origin is the empty string,
     * which is none of them -- it would read as "no results" rather than "no
     * filter". Undefined is the only thing that means "any".
     *
     * Asserted on the empty-state copy rather than on a fresh request,
     * because clearing is the point: `{origin: undefined}` hashes to the same
     * query key as `{}`, so React Query correctly serves the unfiltered rows
     * it already has instead of asking again. */
    draw();
    await waitFor(() => expect(matchApi.list).toHaveBeenCalled());

    const select = screen.getByLabelText(/source/i);
    await userEvent.selectOptions(select, "recommendation");
    await screen.findByText(/widen them to see more/i);

    await userEvent.selectOptions(select, "");

    expect(await screen.findByText(/submit a posting url above/i)).toBeInTheDocument();
    for (const call of vi.mocked(matchApi.list).mock.calls) {
      expect(call[2]).not.toHaveProperty("origin", "");
    }
  });

  it("carries the toggles through", async () => {
    draw();
    await waitFor(() => expect(matchApi.list).toHaveBeenCalled());

    await userEvent.click(screen.getByLabelText(/remote only/i));

    await waitFor(() =>
      expect(matchApi.list).toHaveBeenLastCalledWith(1, 25, { remote_only: true }),
    );
  });

  it("hides closed postings by default and keeps them reachable", async () => {
    /* A filled posting is still a true record of what was recommended, so it
     * is hidden rather than dropped -- but it must not be presented as live
     * without the user asking. */
    draw();
    await waitFor(() => expect(matchApi.list).toHaveBeenCalledWith(1, 25, {}));

    await userEvent.click(screen.getByLabelText(/include closed/i));

    await waitFor(() =>
      expect(matchApi.list).toHaveBeenLastCalledWith(1, 25, { include_closed: true }),
    );
  });

  it("keeps filters in the query key so the list cannot contradict the controls", async () => {
    /* Without the filters in the key, React Query serves the previous
     * filter's rows from cache while the new request is in flight -- so the
     * feed briefly shows rows the visible controls exclude. */
    vi.mocked(matchApi.list).mockResolvedValue([match()] as never);
    draw();
    await screen.findByText(/senior backend engineer/i);

    vi.mocked(matchApi.list).mockResolvedValue([
      match({ id: 2, posting_title: "Staff Platform Engineer" }),
    ] as never);
    await userEvent.selectOptions(screen.getByLabelText(/source/i), "user_submission");

    expect(await screen.findByText(/staff platform engineer/i)).toBeInTheDocument();
  });
});

describe("the empty feed", () => {
  it("tells a filtered user to widen, not to go submit something", async () => {
    /* Somebody with fifty matches who ticked "remote only" does not have the
     * problem the onboarding copy describes. */
    draw();
    await waitFor(() => expect(matchApi.list).toHaveBeenCalled());

    await userEvent.click(screen.getByLabelText(/remote only/i));

    expect(await screen.findByText(/widen them to see more/i)).toBeInTheDocument();
  });

  it("gives onboarding advice when nothing is filtered", async () => {
    draw();

    expect(await screen.findByText(/submit a posting url above/i)).toBeInTheDocument();
  });
});

/* --- feedback ---------------------------------------------------------- */

describe("feedback capture", () => {
  beforeEach(() => {
    vi.mocked(matchApi.list).mockResolvedValue([match()] as never);
    vi.mocked(matchApi.feedback).mockResolvedValue({ id: 1 } as never);
  });

  it("records the verdict against the match", async () => {
    draw();

    await userEvent.click(await screen.findByRole("button", { name: "Interested" }));

    await waitFor(() => expect(matchApi.feedback).toHaveBeenCalledWith(1, "interested"));
  });

  it("acknowledges rather than leaving the click unanswered", async () => {
    draw();

    await userEvent.click(await screen.findByRole("button", { name: "Applied" }));

    expect(await screen.findByRole("status")).toHaveTextContent(/recorded/i);
  });

  it("records a second, different verdict instead of treating these as radio buttons", async () => {
    /* The table is append-only and interested-then-applied is a funnel worth
     * keeping. Mutually-exclusive controls would throw that sequence away in
     * the UI even though the API preserves it. */
    draw();

    await userEvent.click(await screen.findByRole("button", { name: "Interested" }));
    await waitFor(() => expect(matchApi.feedback).toHaveBeenCalledWith(1, "interested"));

    await userEvent.click(screen.getByRole("button", { name: "Applied" }));

    await waitFor(() => expect(matchApi.feedback).toHaveBeenCalledWith(1, "applied"));
    expect(matchApi.feedback).toHaveBeenCalledTimes(2);
  });

  it("reverts the confirmation when the label never reached the server", async () => {
    /* A verdict that looks recorded and was not is exactly the data loss this
     * feature exists to prevent. */
    vi.mocked(matchApi.feedback).mockRejectedValue(new Error("500"));
    draw();

    await userEvent.click(await screen.findByRole("button", { name: "Interested" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/could not record that/i);
    expect(screen.queryByText(/recorded — thanks/i)).not.toBeInTheDocument();
  });

  it("does not promise the feedback improves ranking, because nothing reads it", async () => {
    draw();

    expect(await screen.findByText(/was this a good match\?/i)).toBeInTheDocument();
  });
});

/* --- building the feed ------------------------------------------------- */

describe("find matches for me", () => {
  it("does not build a feed just because somebody opened the page", async () => {
    /* A build is a recall plus 200 reranks. Firing it on mount turns the
     * spec's lazy strategy into an eager one for every profile clicked. */
    draw();
    await waitFor(() => expect(matchApi.list).toHaveBeenCalled());

    expect(matchApi.recommend).not.toHaveBeenCalled();
  });

  it("asks the server to build one when clicked", async () => {
    vi.mocked(matchApi.recommend).mockResolvedValue({ status: "queued" } as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /find matches for me/i }));

    await waitFor(() => expect(matchApi.recommend).toHaveBeenCalledWith(1));
  });

  it("distinguishes queued from already-current", async () => {
    /* Both mean "nothing is coming right now" for different reasons, and
     * saying them the same way leaves someone waiting on a build that will
     * never arrive. */
    vi.mocked(matchApi.recommend).mockResolvedValue({ status: "already_current" } as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /find matches/i }));

    expect(await screen.findByText(/already up to date with the current scorer/i)).toBeInTheDocument();
  });

  it("explains a profile that is not ready yet", async () => {
    vi.mocked(matchApi.recommend).mockResolvedValue({ status: "profile_not_ready" } as never);
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /find matches/i }));

    expect(await screen.findByText(/still being processed/i)).toBeInTheDocument();
  });

  it("reports a failure rather than looking like the button did nothing", async () => {
    vi.mocked(matchApi.recommend).mockRejectedValue(new Error("503"));
    draw();

    await userEvent.click(await screen.findByRole("button", { name: /find matches/i }));

    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});

/* --- scorer generations ------------------------------------------------ */

describe("scorer versions", () => {
  it("warns when the feed mixes two generations", async () => {
    /* Scores from different scorers are not comparable, so a list sorted
     * across two of them is silently wrong -- position 3 beats position 4
     * because the rules differed, not because it fits better. */
    vi.mocked(matchApi.list).mockResolvedValue([
      match({ id: 1, scorer_version: 3 }),
      match({ id: 2, scorer_version: 4, posting_title: "Other" }),
    ] as never);
    draw();

    expect(
      await screen.findByText(/2 different scorer versions and are not directly comparable/i),
    ).toBeInTheDocument();
  });

  it("stays quiet when every row came from one scorer", async () => {
    vi.mocked(matchApi.list).mockResolvedValue([
      match({ id: 1 }),
      match({ id: 2, posting_title: "Other" }),
    ] as never);
    draw();

    await screen.findByText(/other/i);
    expect(screen.queryByText(/different scorer versions/i)).not.toBeInTheDocument();
  });
});

/* --- ranking ----------------------------------------------------------- */

describe("ranking", () => {
  it("numbers rows in the order the server returned them", async () => {
    /* The server sorts by final_score. Re-sorting here would be a second
     * ranking rule that can disagree with the one the scores came from. */
    vi.mocked(matchApi.list).mockResolvedValue([
      match({ id: 1, posting_title: "First", final_score: 0.9 }),
      match({ id: 2, posting_title: "Second", final_score: 0.4 }),
    ] as never);
    draw();

    // "#1", not /first/i -- the panel's own subtitle ends "best first", so a
    // loose matcher resolves against the header before any row has loaded.
    await screen.findByText("#1");

    const cards = screen.getAllByRole("listitem").filter((li) => li.className.includes("mf-card"));
    expect(within(cards[0]).getByText("#1")).toBeInTheDocument();
    expect(within(cards[0]).getByText(/First/)).toBeInTheDocument();
    expect(within(cards[1]).getByText("#2")).toBeInTheDocument();
  });

  it("surfaces a load failure instead of rendering an empty feed", async () => {
    vi.mocked(matchApi.list).mockRejectedValue(new Error("500"));
    draw();

    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });
});
