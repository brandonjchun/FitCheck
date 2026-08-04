/* The data layer behind the ops dashboard's queue charts.
 *
 * Separate from the components that draw it for two reasons. The dull one is
 * that mixing hooks and components in one module breaks Vite's fast refresh. The
 * useful one is that sampling and colour assignment are the parts with rules
 * attached -- how often a sample is taken, what happens at the cap, which queue
 * owns which colour -- and they are worth reading and testing without an SVG in
 * the way.
 *
 * **History is client-side and deliberately so.** Nothing is stored server-side
 * and nothing survives a reload: the buffer fills from the same poll the rest of
 * the page already runs. The alternative is a time series in Redis, which means
 * a retention policy, a write on every sample, and a new endpoint -- to support a
 * window that is only ever read by somebody actively watching the screen. When
 * this has to answer "what happened at 3am" it will need that storage; while it
 * answers "what is happening now", the poll is the source.
 *
 * The consequence is stated in the UI rather than hidden: the chart says how long
 * its window is and admits when it is still filling.
 */

import { useEffect, useRef, useState } from "react";
import type { QueueHealth } from "../api/client";

/** Samples retained. At the dashboard's 3s poll this is a 3-minute window. */
export const HISTORY_POINTS = 60;

/**
 * Which `--series-N` tokens the overlaid chart may use, in the order it hands
 * them out.
 *
 * **Not 1,2,3,4, and the reason is the chart form.** The eight series tokens are
 * ordered to be safe for *neighbouring* marks -- stacked segments and adjacent
 * bars, where only touching pairs ever need separating. Overlaid lines are not
 * that: they all share one plot box, so any two of them can end up running
 * alongside each other, and the pair that has to be separable is *every* pair.
 *
 * Measured against this project's own surfaces, slots 2 and 4 fail that test
 * outright -- ΔE 4.8 under simulated deuteranopia and 10.6 with full colour
 * vision on the dark surface, against floors of 8 and 15. It is not a marginal
 * call and it was not theory: rendered side by side, `scoring` and `interactive`
 * were the same amber line. Of every 4-slot combination only {1,4,5,6} and
 * {4,5,6,7} clear all gates in both themes, and no combination of five does.
 *
 * So: four series, these four tokens. A fifth queue is named in a note and left
 * to the table rather than handed a fifth colour, because there is no fifth
 * colour here that a red-green colourblind reader -- roughly one man in twelve --
 * could tell apart from the four above it.
 */
export const SERIES_SLOTS = [1, 4, 5, 6] as const;

export type DepthSample = {
  /** react-query's `dataUpdatedAt` for the poll that produced this sample. */
  t: number;
  depths: Record<string, number>;
};

/**
 * Accumulate one sample per completed poll.
 *
 * Keyed on `updatedAt` rather than on the array identity of `queues`, which is
 * new on every render and would append a duplicate every time the parent
 * re-rendered -- flattening the time axis into whatever React happened to do.
 * react-query only moves `dataUpdatedAt` when a fetch actually resolves, so this
 * samples real polls and ignores renders.
 */
export function useDepthHistory(
  queues: QueueHealth[] | undefined,
  updatedAt: number | undefined,
): DepthSample[] {
  const [samples, setSamples] = useState<DepthSample[]>([]);
  const lastStamp = useRef<number>(0);

  useEffect(() => {
    if (!queues || !updatedAt || updatedAt === lastStamp.current) return;
    lastStamp.current = updatedAt;

    const depths: Record<string, number> = {};
    for (const q of queues) depths[q.name] = q.depth;

    setSamples((prev) => {
      const next = [...prev, { t: updatedAt, depths }];
      /* A ring buffer by slicing. At 60 entries the copy is free, and the
       * alternative -- mutating a fixed array in place -- would not give React a
       * new reference to re-render from. The cap matters because this page is
       * the kind that gets left open on a wall display for days. */
      return next.length > HISTORY_POINTS ? next.slice(-HISTORY_POINTS) : next;
    });
  }, [queues, updatedAt]);

  return samples;
}

/**
 * The name → colour-slot registry, owned above every view that reads it.
 *
 * Two properties, both of which a simpler implementation loses:
 *
 *   **Colour follows the entity, never its rank.** An operator who has learned
 *   that ingest is the blue line must not find it repainted because scoring
 *   overtook it. Slots are handed out on first sighting and never reassigned, so
 *   a queue keeps its colour through every reordering the poll produces.
 *
 *   **One registry, both consumers.** The card sparklines echo the chart's
 *   colours, and a queue drawn blue in the chart and amber on its own card is
 *   worse than no colour at all -- the reader has to work out that the two
 *   shapes are the same thing.
 *
 * A ref rather than state, written during render: assigning a slot must not
 * cause a render, and the assignment is idempotent so repeating it costs nothing.
 */
export function useSeriesSlots(queues: QueueHealth[]): Map<string, number> {
  const slots = useRef<Map<string, number>>(new Map());
  for (const q of queues) {
    if (!slots.current.has(q.name) && slots.current.size < SERIES_SLOTS.length) {
      slots.current.set(q.name, SERIES_SLOTS[slots.current.size]);
    }
  }
  return slots.current;
}

/**
 * One queue's depth across the window.
 *
 * A queue absent from an earlier sample reads as zero rather than as a gap.
 * Undefined would put NaN in the path -- which drops the whole line silently
 * instead of erroring -- and zero is also the truth: the queue held nothing,
 * because it did not yet exist.
 */
export function trendFor(name: string, samples: DepthSample[]): number[] {
  return samples.map((s) => s.depths[name] ?? 0);
}
