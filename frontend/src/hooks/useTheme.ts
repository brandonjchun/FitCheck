import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "fitcheck-theme";

function apply(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") {
    // Removing the attribute hands control back to the media query in
    // tokens.css. Writing data-theme="system" instead would match neither
    // selector and strand the page on light.
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", theme);
  }
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(() => {
    if (typeof window === "undefined") return "system";
    return (localStorage.getItem(STORAGE_KEY) as Theme) ?? "system";
  });

  useEffect(() => {
    apply(theme);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    if (next === "system") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, next);
  }, []);

  const toggle = useCallback(() => {
    // Resolve "system" to whatever it is currently showing before flipping,
    // so the first click always visibly changes something. Cycling
    // system -> light would look like a no-op to anyone already on a light OS.
    const resolved =
      theme === "system"
        ? window.matchMedia("(prefers-color-scheme: dark)").matches
          ? "dark"
          : "light"
        : theme;
    setTheme(resolved === "dark" ? "light" : "dark");
  }, [theme, setTheme]);

  return { theme, setTheme, toggle };
}
