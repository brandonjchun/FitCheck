import { fireEvent, renderHook, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { QueueHealth } from "../api/client";
import {
  HISTORY_POINTS,
  trendFor,
  useDepthHistory,
  useSeriesSlots,
  type DepthSample,
} from "../hooks/useQueueHistory";
import { QueueDepthChart, QueueSparkline } from "./QueueDepthChart";

/* --- fixtures --------------------------------------------------------- */

const queue = (name: string, depth: number): QueueHealth =>
  ({
    name,
    depth,
    declared: true,
    started: 0,
    failed: 0,
    deferred: 0,
    scheduled: 0,
    worker_count: 1,
  }) as QueueHealth;

/** `n` samples, `depths` applied to every one unless a builder is given. */
const samplesOf = (
  rows: Array<Record<string, number>>,
  startAt = 1_000_000,
): DepthSample[] =>
  rows.map((depths, i) => ({ t: startAt + i * 3000, depths }));

/**
 * Wraps the chart in the real slot registry rather than a stub map.
 *
 * The registry is the thing under test in the identity cases below -- it is what
 * makes a queue keep its colour across a reordering -- so a fixture that handed
 * over a pre-built map would assert on the fixture instead of on the component.
 */
function Harness({
  queues,
  samples,
}: {
  queues: QueueHealth[];
  samples: DepthSample[];
}) {
  const slots = useSeriesSlots(queues);
  return (
    <QueueDepthChart
      queues={queues}
      samples={samples}
      pollMs={3000}
      slots={slots}
    />
  );
}

function draw(queues: QueueHealth[], samples: DepthSample[]) {
  return render(<Harness queues={queues} samples={samples} />);
}

/** The rendered stroke colour of each series line, in DOM order. */
function lineColors(container: HTMLElement): string[] {
  return Array.from(container.querySelectorAll(".qdc-line")).map(
    (el) => (el.parentElement as HTMLElement).style.color,
  );
}

/* --- the sample buffer ------------------------------------------------- */

describe("useDepthHistory", () => {
  it("appends one sample per completed poll", () => {
    const { result, rerender } = renderHook(
      ({ queues, at }: { queues: QueueHealth[]; at: number }) =>
        useDepthHistory(queues, at),
      { initialProps: { queues: [queue("ingest", 3)], at: 100 } },
    );

    expect(result.current).toHaveLength(1);

    rerender({ queues: [queue("ingest", 7)], at: 200 });

    expect(result.current).toHaveLength(2);
    expect(result.current[1].depths).toEqual({ ingest: 7 });
  });

  it("ignores a re-render that carried no new poll", () => {
    /* `queues` is a fresh array on every render, so keying the effect on it
     * would append a duplicate sample every time the parent re-rendered --
     * flattening the chart's time axis into whatever React felt like doing. */
    const { result, rerender } = renderHook(
      ({ queues, at }: { queues: QueueHealth[]; at: number }) =>
        useDepthHistory(queues, at),
      { initialProps: { queues: [queue("ingest", 3)], at: 100 } },
    );

    rerender({ queues: [queue("ingest", 3)], at: 100 });
    rerender({ queues: [queue("ingest", 3)], at: 100 });

    expect(result.current).toHaveLength(1);
  });

  it("holds the window at its cap instead of growing without bound", () => {
    /* A page left open on a wall display runs for days. Without the cap this
     * is an unbounded array and an ever-slower render. */
    const { result, rerender } = renderHook(
      ({ at }: { at: number }) => useDepthHistory([queue("ingest", at)], at),
      { initialProps: { at: 1 } },
    );

    for (let i = 2; i <= HISTORY_POINTS + 25; i++) rerender({ at: i });

    expect(result.current).toHaveLength(HISTORY_POINTS);
    /* The newest sample survived and the oldest was dropped, i.e. it discards
     * from the correct end. */
    expect(result.current[HISTORY_POINTS - 1].t).toBe(HISTORY_POINTS + 25);
  });

  it("records nothing before the first poll resolves", () => {
    const { result } = renderHook(() => useDepthHistory(undefined, undefined));
    expect(result.current).toEqual([]);
  });
});

/* --- priming ----------------------------------------------------------- */

describe("before there is anything to plot", () => {
  it("says what it is waiting for rather than showing an empty plot", () => {
    /* Two points are the minimum for a line. An empty axis with no explanation
     * reads as "no queues", which is a different and alarming statement. */
    draw([queue("ingest", 0)], samplesOf([{ ingest: 0 }]));

    expect(screen.getByText(/collecting samples/i)).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("still lists the queues and their current depth", () => {
    /* The legend is not gated on having history: the depths are known from the
     * first poll, and hiding them for the first three seconds is a regression
     * against the cards this sits above. */
    draw([queue("ingest", 42)], samplesOf([{ ingest: 42 }]));

    expect(screen.getByText("42")).toBeInTheDocument();
  });
});

/* --- the plot ---------------------------------------------------------- */

describe("plotting", () => {
  const queues = [queue("ingest", 5), queue("scoring", 2)];
  const samples = samplesOf([
    { ingest: 1, scoring: 0 },
    { ingest: 3, scoring: 1 },
    { ingest: 5, scoring: 2 },
  ]);

  it("draws one line per queue", () => {
    const { container } = draw(queues, samples);

    expect(container.querySelectorAll(".qdc-line")).toHaveLength(2);
  });

  it("plots a point per sample", () => {
    const { container } = draw(queues, samples);

    const points = container
      .querySelector(".qdc-line")!
      .getAttribute("points")!
      .trim()
      .split(/\s+/);
    expect(points).toHaveLength(3);
  });

  it("gives no coordinate a NaN", () => {
    /* One NaN in a `points` list silently drops the whole polyline, so the
     * chart renders as blank rather than as broken -- the failure mode most
     * likely to ship unnoticed. */
    const { container } = draw(queues, samples);

    for (const line of Array.from(container.querySelectorAll(".qdc-line"))) {
      expect(line.getAttribute("points")).not.toMatch(/NaN/);
    }
  });

  it("scales every series against one shared axis", () => {
    /* Never two y-scales. A second axis makes the alignment of the two scales
     * arbitrary, so the chart invents a relationship the data does not have.
     * Here: equal depths must land on an equal y. */
    const { container } = draw(
      [queue("a", 4), queue("b", 4)],
      samplesOf([
        { a: 0, b: 0 },
        { a: 4, b: 4 },
      ]),
    );

    const [first, second] = Array.from(container.querySelectorAll(".qdc-line")).map(
      (l) => l.getAttribute("points"),
    );
    expect(first).toBe(second);
  });

  it("describes itself for a reader who cannot see it", () => {
    draw(queues, samples);

    const label = screen.getByRole("img").getAttribute("aria-label")!;
    expect(label).toMatch(/ingest now 5, peak 5/);
    expect(label).toMatch(/scoring now 2, peak 2/);
  });
});

/* --- colour follows the entity ---------------------------------------- */

describe("series identity", () => {
  it("keeps a queue's colour when the depth ordering changes", () => {
    /* The property that makes the chart readable over time. An operator who
     * has learned that ingest is the second colour must not find it repainted
     * because scoring overtook it -- colour follows the entity, never its
     * rank. */
    const { container, rerender } = render(
      <Harness
        queues={[queue("ingest", 90), queue("scoring", 1)]}
        samples={samplesOf([
          { ingest: 80, scoring: 0 },
          { ingest: 90, scoring: 1 },
        ])}
      />,
    );

    const before = lineColors(container);

    // ingest drains to nothing while scoring takes over the backlog.
    rerender(
      <Harness
        queues={[queue("scoring", 120), queue("ingest", 0)]}
        samples={samplesOf([
          { ingest: 80, scoring: 0 },
          { ingest: 0, scoring: 120 },
        ])}
      />,
    );

    expect(lineColors(container)).toEqual(before);
  });

  it("gives a queue that appears later its own unused colour", () => {
    /* An undeclared queue showing up mid-session is one of the two states this
     * whole page exists to catch, so it has to be visible and distinct rather
     * than sharing a hue with a queue already on screen. */
    const { container, rerender } = render(
      <Harness
        queues={[queue("ingest", 1)]}
        samples={samplesOf([{ ingest: 1 }, { ingest: 1 }])}
      />,
    );

    rerender(
      <Harness
        queues={[queue("ingest", 1), queue("defualt", 9)]}
        samples={samplesOf([
          { ingest: 1, defualt: 0 },
          { ingest: 1, defualt: 9 },
        ])}
      />,
    );

    const colors = lineColors(container);
    expect(colors).toHaveLength(2);
    expect(new Set(colors).size).toBe(2);
  });

  it("declines to plot a fifth queue instead of inventing a colour", () => {
    /* The cap is four because overlaid lines all share one plot box, so every
     * pair of lines has to be separable -- and only two 4-slot combinations of
     * the series palette clear that in both themes, with no combination of five
     * clearing it at all. What matters as much as the cap is that the omission
     * is *stated*: a silent truncation reads as "these four are all of them",
     * on the one page whose job is to notice a queue nobody is consuming. */
    const many = Array.from({ length: 6 }, (_, i) => queue(`q${i}`, i));
    const row: Record<string, number> = {};
    for (const q of many) row[q.name] = q.depth;

    const { container } = draw(many, samplesOf([row, row]));

    expect(container.querySelectorAll(".qdc-line")).toHaveLength(4);
    expect(screen.getByText(/2 further queues not plotted/i)).toBeInTheDocument();
    expect(screen.getByText(/q4, q5/)).toBeInTheDocument();
  });

  it("uses only the slots that pass every colour gate for overlaid lines", () => {
    /* Pins the safe set. Slots 2 and 4 together measure ΔE 4.8 under simulated
     * deuteranopia and 10.6 with full colour vision on the dark surface, against
     * floors of 8 and 15 -- so a well-meaning "just use 1,2,3,4 like every other
     * chart" is a regression a reader cannot report, because to them the two
     * lines were always the same colour. */
    const { container } = draw(
      [queue("a", 1), queue("b", 1), queue("c", 1), queue("d", 1)],
      samplesOf([
        { a: 1, b: 1, c: 1, d: 1 },
        { a: 1, b: 1, c: 1, d: 1 },
      ]),
    );

    expect(lineColors(container)).toEqual([
      "var(--series-1)",
      "var(--series-4)",
      "var(--series-5)",
      "var(--series-6)",
    ]);
  });
});

/* --- reading a value --------------------------------------------------- */

describe("the readout", () => {
  const queues = [queue("ingest", 5), queue("scoring", 2)];
  const samples = samplesOf([
    { ingest: 1, scoring: 0 },
    { ingest: 3, scoring: 1 },
    { ingest: 5, scoring: 2 },
  ]);

  it("shows every series at the hovered moment, not just the line under the pointer", () => {
    const { container } = draw(queues, samples);

    const hit = container.querySelector(".qdc-hit")!;
    // jsdom reports a zero-sized box, which clamps to the oldest sample.
    fireEvent.pointerMove(hit, { clientX: 0 });

    const tip = screen.getByRole("status");
    expect(within(tip).getByText("ingest")).toBeInTheDocument();
    expect(within(tip).getByText("scoring")).toBeInTheDocument();
    expect(within(tip).getByText("1")).toBeInTheDocument();
  });

  it("moves the same crosshair from the keyboard", async () => {
    /* Keyboard parity, not a lesser version: a tooltip that only opens on
     * hover makes every value it carries unreachable without a pointer. */
    const user = userEvent.setup();
    draw(queues, samples);

    const svg = screen.getByRole("img");
    svg.focus();
    await user.keyboard("{ArrowLeft}");

    const tip = screen.getByRole("status");
    // One step back from the newest sample: ingest was 3 there, not 5.
    expect(within(tip).getByText("3")).toBeInTheDocument();
  });

  it("closes the readout on Escape", async () => {
    const user = userEvent.setup();
    draw(queues, samples);

    screen.getByRole("img").focus();
    await user.keyboard("{ArrowLeft}");
    expect(screen.getByRole("status")).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});

/* --- the table twin --------------------------------------------------- */

describe("table view", () => {
  it("carries every plotted value without needing a pointer", async () => {
    /* Not a nicety. Three of the light-mode series colours sit below 3:1
     * against the surface, which obliges a non-colour route to every value --
     * and this is also the only way to read an exact number off the chart. */
    const user = userEvent.setup();
    draw(
      [queue("ingest", 5)],
      samplesOf([{ ingest: 1 }, { ingest: 3 }, { ingest: 5 }]),
    );

    await user.click(screen.getByRole("button", { name: /table/i }));

    const table = screen.getByRole("table");
    expect(within(table).getByText("1")).toBeInTheDocument();
    expect(within(table).getByText("3")).toBeInTheDocument();
    expect(within(table).getByText("5")).toBeInTheDocument();
  });

  it("says how full the window is", async () => {
    const user = userEvent.setup();
    draw([queue("ingest", 1)], samplesOf([{ ingest: 1 }, { ingest: 1 }]));

    await user.click(screen.getByRole("button", { name: /table/i }));

    expect(
      screen.getByText(new RegExp(`2 of ${HISTORY_POINTS} samples`, "i")),
    ).toBeInTheDocument();
  });

  it("lists the newest sample first", async () => {
    const user = userEvent.setup();
    draw([queue("ingest", 9)], samplesOf([{ ingest: 1 }, { ingest: 9 }]));

    await user.click(screen.getByRole("button", { name: /table/i }));

    const rows = screen.getAllByRole("row");
    // rows[0] is the header.
    expect(rows[1]).toHaveTextContent("now");
    expect(rows[1]).toHaveTextContent("9");
  });
});

/* --- the per-card sparkline ------------------------------------------- */

describe("card sparkline", () => {
  const sparkline = (values: number[], slot: number | undefined = 1) =>
    render(<QueueSparkline name="discovery" values={values} slot={slot} />);

  it("scales to the queue's own peak, not a shared one", () => {
    /* The whole reason it exists. On the shared axis above, a queue peaking at
     * 14 beside one peaking at 400 is a flat line on the baseline and shows no
     * motion at all -- so three of four queues were unreadable while a backlog
     * drained. Here the peak must reach the top of its own box. */
    const { container } = sparkline([0, 7, 14]);

    const ys = container
      .querySelector(".qs-line")!
      .getAttribute("points")!
      .split(" ")
      .map((p) => Number(p.split(",")[1]));

    // Lowest value sits at the bottom, the peak at the top, both inside the box.
    expect(Math.max(...ys)).toBeGreaterThan(Math.min(...ys));
    expect(Math.min(...ys)).toBeLessThan(6);
  });

  it("states the scale rather than implying it", () => {
    /* Independent y-scales per panel mislead unless the reader is told: without
     * the peak printed beside it, discovery's 14 and ingest's 400 draw the
     * identical shape. */
    sparkline([0, 7, 14]);

    expect(screen.getByText("peak 14")).toBeInTheDocument();
  });

  it("does not divide by zero on a queue that never had work", () => {
    /* An idle queue is the common case, not the edge case -- and a zero
     * denominator renders NaN into every coordinate, which drops the polyline
     * silently rather than erroring. */
    const { container } = sparkline([0, 0, 0]);

    expect(container.querySelector(".qs-line")!.getAttribute("points")).not.toMatch(
      /NaN/,
    );
    expect(screen.getByText("peak 0")).toBeInTheDocument();
  });

  it("echoes the chart's colour for a plotted queue", () => {
    /* Same queue, same colour in both views. Two different colours for one
     * queue is worse than no colour, because the reader has to work out that
     * the two shapes are the same thing. */
    const { container } = sparkline([0, 1], 5);

    expect((container.querySelector(".qs-line") as HTMLElement).style.color).toBe(
      "var(--series-5)",
    );
  });

  it("falls back to de-emphasis ink for a queue the chart could not plot", () => {
    /* Past the four-colour cap there is no identity colour to echo, and reusing
     * one would claim a relationship to the wrong line. Rendered directly rather
     * than through the helper, whose default parameter would swallow an explicit
     * `undefined` and quietly assert the opposite of the intent. */
    const { container } = render(
      <QueueSparkline name="q9" values={[0, 1]} slot={undefined} />,
    );

    expect((container.querySelector(".qs-line") as HTMLElement).style.color).toBe(
      "var(--text-faint)",
    );
  });

  it("says it is collecting rather than drawing a line from one point", () => {
    sparkline([3]);

    expect(screen.getByText(/collecting/i)).toBeInTheDocument();
  });

  it("summarises itself for a screen reader", () => {
    sparkline([0, 7, 14]);

    expect(screen.getByRole("img").getAttribute("aria-label")).toMatch(
      /discovery trend: now 14, peak 14/,
    );
  });
});

/* --- shared helpers --------------------------------------------------- */

describe("trendFor", () => {
  it("reads a missing queue as zero rather than a gap", () => {
    /* A queue that appeared mid-window has no entry in the earlier samples.
     * Treating that as undefined puts NaN in the path; zero is also the truth --
     * the queue held nothing, because it did not exist. */
    expect(
      trendFor("late", [{ t: 1, depths: {} }, { t: 2, depths: { late: 4 } }]),
    ).toEqual([0, 4]);
  });
});

/* --- honesty about the window ---------------------------------------- */

describe("what it claims", () => {
  it("admits the history is client-side", () => {
    /* Nothing is stored server-side, so a reload starts over. An operator who
     * assumed otherwise would read a fresh window as "the backlog cleared". */
    draw([queue("ingest", 1)], samplesOf([{ ingest: 1 }, { ingest: 1 }]));

    expect(screen.getByText(/starts over on reload/i)).toBeInTheDocument();
  });

  it("states the sampling interval and the window length", () => {
    draw([queue("ingest", 1)], samplesOf([{ ingest: 1 }, { ingest: 1 }]));

    expect(screen.getByText(/every 3s/i)).toBeInTheDocument();
    expect(screen.getByText(/last 3 minutes/i)).toBeInTheDocument();
  });
});
