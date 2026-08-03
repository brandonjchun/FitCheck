import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

/**
 * `matchMedia`, which jsdom does not implement.
 *
 * Not a detail: three separate pieces of this app branch on it --
 * `useReducedMotion`, `useTheme`'s toggle, and the canvas hero -- so without a
 * stub the tests fail on a missing global rather than on anything real.
 *
 * The stub defaults every query to *not* matching and is settable per test via
 * `setMatchMedia`. Defaulting to false rather than true matters: it means a
 * test that forgets to configure the preference sees the ordinary path, not
 * the reduced-motion one, so a component that silently stopped animating
 * cannot pass by accident.
 */
type Listener = (event: MediaQueryListEvent) => void;

const listeners = new Map<string, Set<Listener>>();
let matches: (query: string) => boolean = () => false;

export function setMatchMedia(fn: (query: string) => boolean) {
  matches = fn;
}

/** Fire a preference change, as an OS-level toggle would. */
export function emitMediaChange(query: string, value: boolean) {
  const event = { matches: value, media: query } as MediaQueryListEvent;
  listeners.get(query)?.forEach((listener) => listener(event));
}

function install() {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    configurable: true,
    value: (query: string) => ({
      media: query,
      get matches() {
        return matches(query);
      },
      onchange: null,
      addEventListener: (_: string, listener: Listener) => {
        if (!listeners.has(query)) listeners.set(query, new Set());
        listeners.get(query)!.add(listener);
      },
      removeEventListener: (_: string, listener: Listener) => {
        listeners.get(query)?.delete(listener);
      },
      // Deprecated pair, still called by some libraries.
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

install();

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  localStorage.clear();
  listeners.clear();
  matches = () => false;
  // Themes are applied to the root element, which outlives cleanup() --
  // it is not part of any component's tree. Left in place, one test's
  // choice becomes the next test's starting state.
  document.documentElement.removeAttribute("data-theme");
});
