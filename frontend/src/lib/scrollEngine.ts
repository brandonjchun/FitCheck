/**
 * A single scroll loop shared by every parallax layer on the page.
 *
 * The naive approach is one `scroll` listener per animated element. With a
 * dozen layers that is a dozen handlers competing on the same event, each
 * reading layout and each writing style -- the classic layout-thrash shape,
 * where a read after a write forces the browser to recompute geometry
 * mid-frame.
 *
 * This module inverts it. One passive listener marks the position dirty; one
 * rAF loop then does *all* reads, then *all* writes, once per frame. Layers
 * subscribe and receive a normalized progress value; they never touch the
 * scroll position themselves.
 *
 * The loop is also demand-driven: it stops entirely when nothing is
 * subscribed or nothing is on screen, so an idle page costs zero frames.
 */

export type ParallaxTarget = {
  el: HTMLElement;
  /** Called once per frame with the element's viewport progress. */
  onFrame: (progress: number, viewportH: number) => void;
  /** Set by the engine's IntersectionObserver; skipped when false. */
  visible: boolean;
};

const targets = new Set<ParallaxTarget>();

let rafId: number | null = null;
let observer: IntersectionObserver | null = null;

function getObserver(): IntersectionObserver {
  if (observer) return observer;
  observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        for (const t of targets) {
          if (t.el === entry.target) t.visible = entry.isIntersecting;
        }
      }
      // Visibility changes can be the thing that makes work necessary again.
      ensureRunning();
    },
    // Generous margin so a layer is already in its correct transformed
    // position by the time it scrolls into view, rather than snapping.
    { rootMargin: "180px 0px 180px 0px", threshold: 0 },
  );
  return observer;
}

/**
 * Progress of an element through the viewport, normalized to 0..1.
 *
 * 0 means the element's top edge has just reached the bottom of the
 * viewport; 1 means its bottom edge has just passed the top. The midpoint,
 * 0.5, is the element centred -- which is the value most layers treat as
 * their rest position, so that a layer is undisplaced when it is the thing
 * you are actually looking at.
 */
function progressOf(rect: DOMRect, viewportH: number): number {
  const total = viewportH + rect.height;
  const travelled = viewportH - rect.top;
  return Math.min(1, Math.max(0, travelled / total));
}

function frame() {
  rafId = null;

  const viewportH = window.innerHeight;

  // Phase 1: read. Every getBoundingClientRect happens before any style is
  // written, so the browser computes layout at most once for the batch.
  const work: Array<{ t: ParallaxTarget; progress: number }> = [];
  for (const t of targets) {
    if (!t.visible) continue;
    work.push({ t, progress: progressOf(t.el.getBoundingClientRect(), viewportH) });
  }

  // Phase 2: write.
  for (const { t, progress } of work) {
    t.onFrame(progress, viewportH);
  }

  if (work.length > 0) ensureRunning();
}

function ensureRunning() {
  if (rafId !== null) return;
  if (targets.size === 0) return;
  rafId = requestAnimationFrame(frame);
}

function onScroll() {
  ensureRunning();
}

let listening = false;

function startListening() {
  if (listening) return;
  listening = true;
  // Passive: this handler never calls preventDefault, and saying so lets the
  // browser scroll without waiting to find out.
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll, { passive: true });
}

function stopListening() {
  if (!listening) return;
  listening = false;
  window.removeEventListener("scroll", onScroll);
  window.removeEventListener("resize", onScroll);
  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

/** Register a layer. Returns an unsubscribe function. */
export function addParallaxTarget(
  el: HTMLElement,
  onFrame: ParallaxTarget["onFrame"],
): () => void {
  const target: ParallaxTarget = { el, onFrame, visible: true };
  targets.add(target);
  getObserver().observe(el);
  startListening();
  ensureRunning();

  return () => {
    targets.delete(target);
    observer?.unobserve(el);
    if (targets.size === 0) stopListening();
  };
}
