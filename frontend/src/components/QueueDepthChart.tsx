import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { QueueHealth } from "../api/client";
import {
  HISTORY_POINTS,
  SERIES_SLOTS,
  trendFor,
  type DepthSample,
} from "../hooks/useQueueHistory";
import "./QueueDepthChart.css";

/**
 * Queue depth over time, plotted from what the page is already polling.
 *
 * The cards beside this chart answer "what is the state right now". They cannot
 * answer the question an operator actually has while watching a crawl drain,
 * which is "is this going up or down, and how fast" -- a card showing 340 looks
 * identical whether it was 12 a minute ago or 900. That difference is the whole
 * reason this exists, and it is why the chart sits *above* the cards rather than
 * replacing them: two different questions about one dataset.
 *
 * See `hooks/useQueueHistory` for where the samples come from and why the series
 * palette is four colours rather than eight.
 */

/** Plot box, in px. The container adds the axis band below this. */
const PLOT_HEIGHT = 168;
const X_AXIS_HEIGHT = 22;
const PAD_LEFT = 44;
const PAD_RIGHT = 16;
const PAD_TOP = 12;

/**
 * Gap between zero and the axis rule.
 *
 * Most queues are empty most of the time, so several series sit on zero at once
 * -- and drawn flush to the baseline their 2px stroke merges into the 1px axis
 * rule, which is how a screenshot showed three queues that looked like one
 * slightly thick axis. Lifting zero off the rule costs 6px and is the difference
 * between "these queues are idle" and "these queues are missing".
 */
const ZERO_INSET = 6;

/** Fallback width for jsdom and the first paint before the observer fires. */
const DEFAULT_WIDTH = 760;

/**
 * One queue's depth over the same window, scaled to its own peak.
 *
 * The companion to the overlaid chart, and the answer to its one real
 * shortcoming: on a shared axis a queue that peaks at 14 next to one that peaks
 * at 400 is a flat line on the baseline, so three of four queues showed no
 * motion at all while a backlog drained. Here each queue gets its own scale, so
 * every one of them is legible -- at the cost of cross-queue comparison, which
 * is exactly what the chart above is for.
 *
 * **The scale is stated, not implied.** Panels with independent y-scales are
 * misleading unless the reader is told, so the peak is printed beside the line.
 * Without it, discovery's 14 and ingest's 400 draw the same shape.
 */
export function QueueSparkline({
  name,
  values,
  slot,
}: {
  name: string;
  values: number[];
  slot: number | undefined;
}) {
  const width = 132;
  const height = 30;
  const inset = 3;
  const peak = values.length ? Math.max(...values) : 0;
  const scale = Math.max(1, peak);

  if (values.length < 2) {
    return <p className="qs-empty">collecting…</p>;
  }

  const points = values
    .map((v, i) => {
      const x = (i / (values.length - 1)) * width;
      const y = height - inset - (v / scale) * (height - inset * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <div className="qs">
      <svg
        className="qs-svg"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        /* The values themselves stay reachable in the chart's table view; this
         * is the summary a screen reader needs to know whether to care. */
        aria-label={`${name} trend: now ${values[values.length - 1]}, peak ${peak} over the window.`}
      >
        <polyline
          className="qs-line"
          points={points}
          /* Undefined slot means this queue is past the chart's four-colour cap,
           * so it has no identity colour to echo. The de-emphasis ink is the
           * honest answer and doubles as a hint that it is not in the chart. */
          style={{ color: slot ? `var(--series-${slot})` : "var(--text-faint)" }}
        />
      </svg>
      {/* A visible direct label rather than a tooltip. The sparkline has no axis,
        * so without this the shape has no scale attached to it -- and a direct
        * label beats a hover the reader has to discover. */}
      <span className="qs-peak">peak {peak.toLocaleString()}</span>
    </div>
  );
}

/** Round a y-axis maximum up to something a human reads. */
function niceMax(value: number): number {
  if (value <= 4) return 4;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 2, 2.5, 5, 10]) {
    const candidate = step * magnitude;
    if (candidate >= value) return candidate;
  }
  return 10 * magnitude;
}

/** Measure the container so strokes stay 2px instead of being viewBox-scaled. */
function useMeasuredWidth() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(DEFAULT_WIDTH);

  useLayoutEffect(() => {
    const node = ref.current;
    /* Guarded rather than assumed: jsdom has no ResizeObserver, and a chart
     * that throws in the test environment is a chart nobody can test. */
    if (!node || typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width ?? 0;
      /* Zero happens while the element is display:none in a collapsed
       * container; keeping the last good width avoids a divide-by-zero
       * geometry pass that renders NaN into every path. */
      if (next > 0) setWidth(next);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return { ref, width };
}

type Series = {
  name: string;
  slot: number;
  values: number[];
  latest: number;
  peak: number;
};

export function QueueDepthChart({
  queues,
  samples,
  pollMs,
  slots,
}: {
  queues: QueueHealth[];
  samples: DepthSample[];
  pollMs: number;
  slots: Map<string, number>;
}) {
  const { ref, width } = useMeasuredWidth();
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);

  const series: Series[] = useMemo(() => {
    const named = queues
      .filter((q) => slots.has(q.name))
      .map((q) => {
        const values = trendFor(q.name, samples);
        return {
          name: q.name,
          slot: slots.get(q.name) as number,
          values,
          latest: q.depth,
          peak: values.length ? Math.max(...values) : q.depth,
        };
      });
    /* Ordered by the slot's position in SERIES_SLOTS, so the legend reads in the
     * order the colours were handed out -- not by depth, which would reshuffle
     * the legend every poll even though the colours correctly stayed put. */
    return named.sort(
      (a, b) => SERIES_SLOTS.indexOf(a.slot as 1) - SERIES_SLOTS.indexOf(b.slot as 1),
    );
  }, [queues, samples, slots]);

  /* Queues past the cap are not plotted, and the note below says so. A silent
   * truncation would read as "these four are all of them". */
  const unplotted = queues.filter((q) => !slots.has(q.name));

  const plotWidth = Math.max(120, width - PAD_LEFT - PAD_RIGHT);
  const yMax = niceMax(
    Math.max(1, ...series.flatMap((s) => (s.values.length ? s.values : [s.latest]))),
  );
  const usableHeight = PLOT_HEIGHT - PAD_TOP - ZERO_INSET;

  const xAt = useCallback(
    (index: number) => {
      if (samples.length <= 1) return PAD_LEFT + plotWidth;
      return PAD_LEFT + (index / (samples.length - 1)) * plotWidth;
    },
    [samples.length, plotWidth],
  );
  const yAt = useCallback(
    (value: number) => PAD_TOP + usableHeight - (value / yMax) * usableHeight,
    [usableHeight, yMax],
  );

  const ticks = useMemo(() => [0, yMax / 2, yMax], [yMax]);

  /* Nearest-sample snapping. The reader aims at a moment in time, never at a
   * 2px line, so the whole plot is one hit target and the crosshair goes to
   * whichever sample is closest. */
  const indexFromPointer = useCallback(
    (clientX: number, box: DOMRect) => {
      if (samples.length === 0) return null;
      const local = clientX - box.left - PAD_LEFT;
      const ratio = plotWidth === 0 ? 0 : local / plotWidth;
      const index = Math.round(ratio * (samples.length - 1));
      return Math.min(samples.length - 1, Math.max(0, index));
    },
    [samples.length, plotWidth],
  );

  const onPointerMove = (event: React.PointerEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    setHoverIndex(indexFromPointer(event.clientX, box));
  };

  /* Keyboard parity with hover, not a lesser version of it: the arrow keys
   * move the same crosshair and open the same readout. */
  const onKeyDown = (event: React.KeyboardEvent<SVGSVGElement>) => {
    if (samples.length === 0) return;
    const last = samples.length - 1;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const from = hoverIndex ?? last;
      setHoverIndex(
        event.key === "ArrowLeft" ? Math.max(0, from - 1) : Math.min(last, from + 1),
      );
    } else if (event.key === "Home") {
      event.preventDefault();
      setHoverIndex(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setHoverIndex(last);
    } else if (event.key === "Escape") {
      setHoverIndex(null);
    }
  };

  const windowSeconds = Math.round((HISTORY_POINTS * pollMs) / 1000);
  const priming = samples.length < 2;

  /* Age of a sample, relative to the newest one. A rolling live window reads
   * better as "40s ago" than as a wall-clock time nobody is comparing against
   * anything. */
  const agoLabel = (index: number) => {
    const newest = samples[samples.length - 1]?.t ?? 0;
    const seconds = Math.round((newest - (samples[index]?.t ?? 0)) / 1000);
    if (seconds <= 0) return "now";
    if (seconds < 60) return `${seconds}s ago`;
    return `${Math.round(seconds / 60)}m ago`;
  };

  const hovered = hoverIndex === null ? null : samples[hoverIndex];

  return (
    <figure className="qdc" ref={ref}>
      <figcaption className="qdc-head">
        <div>
          <h3 className="qdc-title">Depth over time</h3>
          <p className="qdc-sub">
            Jobs waiting per queue, sampled every {pollMs / 1000}s over the last{" "}
            {windowSeconds >= 60
              ? `${Math.round(windowSeconds / 60)} minutes`
              : `${windowSeconds} seconds`}
            . Held in the browser, so it starts over on reload.
          </p>
        </div>
        <button
          type="button"
          className="qdc-toggle"
          onClick={() => setShowTable((v) => !v)}
          aria-expanded={showTable}
        >
          {showTable ? "Hide table" : "Table"}
        </button>
      </figcaption>

      {priming ? (
        /* Not a spinner. Two points are the minimum for a line, so this says
         * what it is waiting for and roughly how long that takes. */
        <p className="qdc-priming">
          Collecting samples — the first line appears in about{" "}
          {Math.max(1, Math.round(pollMs / 1000))}s.
        </p>
      ) : (
        <div className="qdc-plot-wrap">
          <svg
            className="qdc-svg"
            width={width}
            height={PLOT_HEIGHT + X_AXIS_HEIGHT}
            viewBox={`0 0 ${width} ${PLOT_HEIGHT + X_AXIS_HEIGHT}`}
            role="img"
            tabIndex={0}
            aria-label={`Queue depth over the last ${windowSeconds} seconds. ${series
              .map((s) => `${s.name} now ${s.latest}, peak ${s.peak}`)
              .join(". ")}`}
            onKeyDown={onKeyDown}
            onBlur={() => setHoverIndex(null)}
          >
            {/* Gridlines: solid hairlines one step off the surface. Dashed
              * would read as a threshold, which none of these are. */}
            {ticks.map((tick) => (
              <g key={tick}>
                <line
                  className="qdc-grid"
                  x1={PAD_LEFT}
                  x2={PAD_LEFT + plotWidth}
                  y1={yAt(tick)}
                  y2={yAt(tick)}
                />
                <text className="qdc-tick" x={PAD_LEFT - 8} y={yAt(tick) + 4}>
                  {Math.round(tick).toLocaleString()}
                </text>
              </g>
            ))}

            <line
              className="qdc-axis"
              x1={PAD_LEFT}
              x2={PAD_LEFT + plotWidth}
              y1={yAt(0) + ZERO_INSET}
              y2={yAt(0) + ZERO_INSET}
            />

            {hovered && (
              <line
                className="qdc-crosshair"
                x1={xAt(hoverIndex as number)}
                x2={xAt(hoverIndex as number)}
                y1={PAD_TOP}
                y2={yAt(0) + ZERO_INSET}
              />
            )}

            {series.map((s) => (
              <g key={s.name} style={{ color: `var(--series-${s.slot})` }}>
                <polyline
                  className="qdc-line"
                  points={s.values.map((v, i) => `${xAt(i)},${yAt(v)}`).join(" ")}
                />
                {/* End marker with a surface-coloured ring, so two queues
                  * sitting on zero stay countable where they overlap. */}
                <circle
                  className="qdc-end"
                  cx={xAt(s.values.length - 1)}
                  cy={yAt(s.values[s.values.length - 1] ?? 0)}
                  r={4}
                />
                {hovered && (
                  <circle
                    className="qdc-dot"
                    cx={xAt(hoverIndex as number)}
                    cy={yAt(s.values[hoverIndex as number] ?? 0)}
                    r={4}
                  />
                )}
              </g>
            ))}

            <text className="qdc-xlab" x={PAD_LEFT} y={PLOT_HEIGHT + 14}>
              {agoLabel(0)}
            </text>
            <text
              className="qdc-xlab qdc-xlab-end"
              x={PAD_LEFT + plotWidth}
              y={PLOT_HEIGHT + 14}
            >
              now
            </text>

            {/* Last, so it takes the pointer from everything beneath it. */}
            <rect
              className="qdc-hit"
              x={PAD_LEFT}
              y={PAD_TOP}
              width={plotWidth}
              height={usableHeight + ZERO_INSET}
              onPointerMove={onPointerMove}
              onPointerLeave={() => setHoverIndex(null)}
            />
          </svg>

          {hovered && (
            <div
              className="qdc-tip"
              style={{
                /* Flipped to the left of the crosshair past the midpoint so the
                 * readout never hangs off the card. */
                left: xAt(hoverIndex as number),
                transform:
                  xAt(hoverIndex as number) > PAD_LEFT + plotWidth / 2
                    ? "translateX(calc(-100% - 12px))"
                    : "translateX(12px)",
              }}
              role="status"
            >
              <span className="qdc-tip-when">{agoLabel(hoverIndex as number)}</span>
              <ul className="qdc-tip-rows">
                {series.map((s) => (
                  <li key={s.name}>
                    <span
                      className="qdc-key"
                      style={{ background: `var(--series-${s.slot})` }}
                      aria-hidden="true"
                    />
                    {/* Value first and heavier: the reader already knows which
                      * series they are looking at and wants the number. */}
                    <span className="qdc-tip-value">
                      {(hovered.depths[s.name] ?? 0).toLocaleString()}
                    </span>
                    <span className="qdc-tip-name">{s.name}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Always present, never optional for two or more series: colour is the
        * fast channel and the legend is the dependable one. */}
      <ul className="qdc-legend">
        {series.map((s) => (
          <li key={s.name}>
            <span
              className="qdc-key qdc-key-line"
              style={{ background: `var(--series-${s.slot})` }}
              aria-hidden="true"
            />
            <span className="qdc-legend-name">{s.name}</span>
            <span className="qdc-legend-value">{s.latest.toLocaleString()}</span>
          </li>
        ))}
      </ul>

      {unplotted.length > 0 && (
        <p className="qdc-note">
          {unplotted.length} further queue{unplotted.length === 1 ? "" : "s"} not
          plotted ({unplotted.map((q) => q.name).join(", ")}) — overlaid lines only
          hold {SERIES_SLOTS.length} colours that stay distinguishable to a
          colourblind reader. Their depth is in the table and in the cards below.
        </p>
      )}

      {showTable && (
        /* The chart's WCAG-clean twin, and not a nicety: three of the light
         * series steps sit below 3:1 against white, which obliges a
         * non-colour route to every value. It is also the only way to read an
         * exact number without a pointer. */
        <div className="qdc-table-wrap">
          <table className="qdc-table">
            <caption>
              Queue depth by sample, newest first. {samples.length} of{" "}
              {HISTORY_POINTS} samples held.
            </caption>
            <thead>
              <tr>
                <th scope="col">When</th>
                {series.map((s) => (
                  <th scope="col" key={s.name}>
                    {s.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {samples
                .map((sample, index) => ({ sample, index }))
                .reverse()
                .map(({ sample, index }) => (
                  <tr key={sample.t}>
                    <th scope="row">{agoLabel(index)}</th>
                    {series.map((s) => (
                      <td key={s.name}>{(sample.depths[s.name] ?? 0).toLocaleString()}</td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </figure>
  );
}
