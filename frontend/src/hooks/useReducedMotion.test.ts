import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useReducedMotion } from "./useReducedMotion";
import { emitMediaChange, setMatchMedia } from "../test/setup";

const REDUCE = "(prefers-reduced-motion: reduce)";

describe("useReducedMotion", () => {
  it("reports the preference on first render", () => {
    /* Read synchronously in the initializer rather than in an effect. A
     * component that animates on mount would otherwise run one frame of
     * motion before finding out it should not have. */
    setMatchMedia((query) => query === REDUCE);

    const { result } = renderHook(() => useReducedMotion());

    expect(result.current).toBe(true);
  });

  it("defaults to false when nothing is set", () => {
    expect(renderHook(() => useReducedMotion()).result.current).toBe(false);
  });

  it("follows a change made while the page is open", () => {
    /* Subscribed rather than read once, because the preference is a live OS
     * setting. Reading it a single time means a user who turns it on watches
     * the animation keep running until they reload -- and the CSS media
     * query cannot help, since what this gates is a requestAnimationFrame
     * loop writing transforms in JavaScript. */
    const { result } = renderHook(() => useReducedMotion());
    expect(result.current).toBe(false);

    act(() => emitMediaChange(REDUCE, true));

    expect(result.current).toBe(true);
  });

  it("follows the preference back off", () => {
    setMatchMedia((query) => query === REDUCE);
    const { result } = renderHook(() => useReducedMotion());

    act(() => emitMediaChange(REDUCE, false));

    expect(result.current).toBe(false);
  });

  it("unsubscribes on unmount", () => {
    /* A listener that outlives its component calls setState on an unmounted
     * hook, which React warns about and which leaks one listener per mount
     * across a session's worth of navigation. */
    const { result, unmount } = renderHook(() => useReducedMotion());

    unmount();
    act(() => emitMediaChange(REDUCE, true));

    expect(result.current).toBe(false);
  });
});
