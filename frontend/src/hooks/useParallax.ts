import { useEffect, useRef } from "react";
import { addParallaxTarget } from "../lib/scrollEngine";
import { useReducedMotion } from "./useReducedMotion";

export type ParallaxOptions = {
  /**
   * Vertical drift in pixels across the full scroll pass. Positive values
   * trail the scroll (the layer appears further away); negative values lead
   * it (nearer than the page). Depth is this number and nothing else --
   * stacking several layers with different speeds is what produces the
   * layered effect.
   */
  speed?: number;
  /**
   * Scale delta across the pass. `0.18` means the layer travels from 1.18x
   * down to 1.0x as it crosses centre -- the "zoom parallax" look. Applied
   * about the element's own centre.
   */
  zoom?: number;
  /** Fade amount at the extremes; 0 disables. */
  fade?: number;
  /** Horizontal drift in pixels, for layers that should also slide. */
  drift?: number;
};

/**
 * Attach an element to the shared scroll engine.
 *
 * Transforms are written directly to `element.style`, never through React
 * state. A setState per frame would re-render the subtree sixty times a
 * second and turn a compositor-only animation into a full render pass; the
 * DOM write here touches only `transform` and `opacity`, both of which the
 * compositor can handle without layout or paint.
 */
export function useParallax<T extends HTMLElement = HTMLDivElement>(
  options: ParallaxOptions = {},
) {
  const ref = useRef<T | null>(null);
  const reduced = useReducedMotion();

  // Held in a ref so that changing options never re-subscribes mid-scroll.
  const optsRef = useRef(options);
  optsRef.current = options;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (reduced) {
      // Explicitly cleared rather than merely left alone: the preference can
      // flip while the page is open, and a stale transform would strand the
      // layer wherever the last frame happened to put it.
      el.style.transform = "";
      el.style.opacity = "";
      return;
    }

    // Hint the compositor once, at subscribe time. Setting will-change per
    // frame is worse than not setting it at all -- it invalidates the very
    // cache it exists to create.
    el.style.willChange = "transform, opacity";

    const unsubscribe = addParallaxTarget(el, (progress) => {
      const { speed = 0, zoom = 0, fade = 0, drift = 0 } = optsRef.current;

      // Centred progress: -1 entering, 0 at centre, +1 leaving. Every effect
      // below is expressed against this so that "at rest in the middle of
      // the viewport" is the natural default.
      const centred = (progress - 0.5) * 2;

      const translateY = centred * speed;
      const translateX = centred * drift;
      const scale = 1 + Math.abs(centred) * zoom;

      el.style.transform = `translate3d(${translateX.toFixed(2)}px, ${translateY.toFixed(2)}px, 0) scale(${scale.toFixed(4)})`;

      if (fade > 0) {
        const opacity = 1 - Math.min(1, Math.abs(centred)) * fade;
        el.style.opacity = opacity.toFixed(3);
      }
    });

    return () => {
      unsubscribe();
      el.style.willChange = "";
    };
  }, [reduced]);

  return ref;
}
