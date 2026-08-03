import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { useTheme } from "./useTheme";
import { setMatchMedia } from "../test/setup";

const STORAGE_KEY = "fitcheck-theme";
const DARK = "(prefers-color-scheme: dark)";

/** What the page is actually showing, as opposed to what state says. */
const applied = () => document.documentElement.getAttribute("data-theme");

describe("useTheme", () => {
  beforeEach(() => setMatchMedia(() => false));

  describe("applying the choice", () => {
    it("writes an explicit choice to the root element", () => {
      const { result } = renderHook(() => useTheme());

      act(() => result.current.setTheme("dark"));

      expect(applied()).toBe("dark");
    });

    it("removes the attribute for system rather than writing 'system'", () => {
      /* The CSS matches [data-theme="light"], [data-theme="dark"], or the
       * media query when neither is present. data-theme="system" matches no
       * selector at all, which strands the page on the light default and
       * looks like the OS preference being ignored. */
      const { result } = renderHook(() => useTheme());
      act(() => result.current.setTheme("dark"));

      act(() => result.current.setTheme("system"));

      expect(applied()).toBeNull();
    });
  });

  describe("persistence", () => {
    it("survives a remount", () => {
      const first = renderHook(() => useTheme());
      act(() => first.result.current.setTheme("dark"));

      const second = renderHook(() => useTheme());

      expect(second.result.current.theme).toBe("dark");
      expect(applied()).toBe("dark");
    });

    it("clears storage for system instead of storing the word", () => {
      /* Storing "system" and reading it back are equivalent here, so this is
       * about not leaving a value that outlives a future change of default. */
      const { result } = renderHook(() => useTheme());
      act(() => result.current.setTheme("light"));

      act(() => result.current.setTheme("system"));

      expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    });
  });

  describe("toggle from system", () => {
    /* The whole reason toggle() is not a two-line flip.
     *
     * Cycling system -> light for someone whose OS is already light is a
     * click that changes nothing on screen. The user concludes the button is
     * broken, and nothing in the code looks wrong -- state did change. So the
     * current *appearance* is resolved first, then inverted. */

    it("goes to light when the OS is dark", () => {
      setMatchMedia((query) => query === DARK);
      const { result } = renderHook(() => useTheme());

      act(() => result.current.toggle());

      expect(result.current.theme).toBe("light");
      expect(applied()).toBe("light");
    });

    it("goes to dark when the OS is light", () => {
      setMatchMedia(() => false);
      const { result } = renderHook(() => useTheme());

      act(() => result.current.toggle());

      expect(result.current.theme).toBe("dark");
    });

    it("always changes what is on screen", () => {
      /* Stated as the property rather than the mechanism, so it still holds
       * if the resolution strategy is rewritten. */
      for (const osIsDark of [true, false]) {
        localStorage.clear();
        document.documentElement.removeAttribute("data-theme");
        setMatchMedia((query) => (query === DARK ? osIsDark : false));

        const { result } = renderHook(() => useTheme());
        const before = osIsDark ? "dark" : "light";

        act(() => result.current.toggle());

        expect(applied()).not.toBe(before);
      }
    });
  });

  describe("toggle from an explicit theme", () => {
    it("flips without consulting the OS", () => {
      setMatchMedia((query) => query === DARK);
      const { result } = renderHook(() => useTheme());
      act(() => result.current.setTheme("light"));

      act(() => result.current.toggle());

      expect(result.current.theme).toBe("dark");
    });

    it("round-trips", () => {
      const { result } = renderHook(() => useTheme());
      act(() => result.current.setTheme("dark"));

      act(() => result.current.toggle());
      act(() => result.current.toggle());

      expect(result.current.theme).toBe("dark");
    });
  });
});
