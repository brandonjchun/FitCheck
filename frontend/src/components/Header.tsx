import { useEffect, useState } from "react";
import { Link, NavLink } from "react-router-dom";
import { useTheme } from "../hooks/useTheme";
import { useLogout, useMe } from "../hooks/useAuth";
import "./Header.css";

function Logo() {
  return (
    <Link to="/" className="logo" aria-label="FitCheck home">
      <span className="logo-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32" width="26" height="26" fill="none">
          <defs>
            <linearGradient id="fc-logo" x1="0" y1="0" x2="32" y2="32">
              <stop offset="0%" stopColor="var(--hero-1)" />
              <stop offset="100%" stopColor="var(--hero-2)" />
            </linearGradient>
          </defs>
          <rect width="32" height="32" rx="9" fill="url(#fc-logo)" />
          <path
            d="M9 16.5l4.6 4.6L23 11.7"
            stroke="white"
            strokeWidth="2.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
      <span className="logo-word">FitCheck</span>
    </Link>
  );
}

function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const label = theme === "dark" ? "Switch to light theme" : "Switch to dark theme";

  return (
    <button className="icon-btn" onClick={toggle} aria-label={label} title={label}>
      <svg viewBox="0 0 24 24" width="18" height="18" fill="none" aria-hidden="true">
        <path
          d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </button>
  );
}

export function Header() {
  const [scrolled, setScrolled] = useState(false);
  const { data: user } = useMe();
  const logout = useLogout();

  useEffect(() => {
    // Threshold rather than a continuous value: the header only has two
    // states, so recomputing a style on every pixel would be work with no
    // visible result.
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className={`site-header ${scrolled ? "is-scrolled" : ""}`}>
      <div className="header-inner">
        <Logo />

        <nav className="header-nav" aria-label="Main">
          <a href="/#how">How it works</a>
          <a href="/#architecture">Architecture</a>
          <a href="/#scoring">Scoring</a>
        </nav>

        <div className="header-actions">
          <ThemeToggle />
          {user ? (
            <>
              <NavLink to="/app" className="btn btn-ghost">
                Workspace
              </NavLink>
              <NavLink to="/ops" className="btn btn-ghost">
                Ops
              </NavLink>
              <button
                className="btn btn-ghost"
                onClick={() => logout.mutate()}
                disabled={logout.isPending}
              >
                {logout.isPending ? "Signing out…" : "Sign out"}
              </button>
            </>
          ) : (
            <>
              <NavLink to="/signin" className="header-link">
                Sign in
              </NavLink>
              <NavLink to="/signin?mode=register" className="btn btn-primary">
                Get started
              </NavLink>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
