import { useEffect, useState } from "react";

/**
 * Whether the user has asked for reduced motion.
 *
 * The CSS media query in global.css handles declarative animation, but it
 * cannot stop JavaScript: a requestAnimationFrame loop writing transforms
 * runs regardless of what CSS thinks. Components that animate in JS branch
 * on this and skip the work entirely -- which is both the accessible outcome
 * and the cheaper one.
 *
 * Subscribed rather than read once, because the preference can change while
 * the page is open (an OS-level toggle, or a display switching modes).
 */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  });

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return reduced;
}
