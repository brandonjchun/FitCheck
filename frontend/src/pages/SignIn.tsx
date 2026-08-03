import { useState, type FormEvent } from "react";
import { Link, Navigate, useSearchParams } from "react-router-dom";
import { errorMessage } from "../api/client";
import { useLogin, useMe, useRegister } from "../hooks/useAuth";
import { HeroCanvas } from "../components/HeroCanvas";
import "./SignIn.css";

/** Mirrors the backend's RegisterRequest floor (min_length=12). */
const MIN_PASSWORD = 12;

export function SignIn() {
  const [params, setParams] = useSearchParams();
  const isRegister = params.get("mode") === "register";

  const { data: user, isLoading } = useMe();
  const login = useLogin();
  const register = useRegister();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const active = isRegister ? register : login;

  // `replace` so the back button returns to wherever they came from rather
  // than stepping through every toggle of this switch.
  const setMode = (next: boolean) => {
    setParams(next ? { mode: "register" } : {}, { replace: true });
    active.reset();
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    active.mutate({ email, password });
  };

  if (isLoading) return null;
  if (user) return <Navigate to="/app" replace />;

  const tooShort = isRegister && password.length > 0 && password.length < MIN_PASSWORD;

  return (
    <main id="main" className="auth">
      <div className="auth-bg" aria-hidden="true">
        <HeroCanvas />
      </div>

      <div className="auth-card card">
        <Link to="/" className="auth-back">
          ← Back
        </Link>

        <h1 className="auth-title">
          {isRegister ? "Create your account" : "Welcome back"}
        </h1>
        <p className="auth-sub">
          {isRegister
            ? "One resume, scored against every posting you check."
            : "Sign in to your workspace."}
        </p>

        <form onSubmit={onSubmit} className="auth-form">
          <div className="field">
            <label className="label" htmlFor="email">
              Email
            </label>
            <input
              id="email"
              className="input"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
            />
          </div>

          <div className="field">
            <label className="label" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              className="input"
              type="password"
              // Tells a password manager whether to offer a saved credential
              // or generate a new one. Wrong value here is why managers
              // sometimes fill a login into a signup form.
              autoComplete={isRegister ? "new-password" : "current-password"}
              required
              minLength={isRegister ? MIN_PASSWORD : undefined}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={isRegister ? `At least ${MIN_PASSWORD} characters` : "••••••••••••"}
              aria-describedby={isRegister ? "pw-hint" : undefined}
            />
            {isRegister && (
              <p id="pw-hint" className={`auth-hint ${tooShort ? "is-warn" : ""}`}>
                {MIN_PASSWORD} characters minimum. Length beats symbols —
                there's no "must contain a special character" rule here.
              </p>
            )}
          </div>

          {active.isError && (
            <p className="form-error" role="alert">
              {errorMessage(active.error, "Could not sign you in.")}
            </p>
          )}

          <button
            type="submit"
            className="btn btn-primary btn-lg auth-submit"
            disabled={active.isPending || tooShort}
          >
            {active.isPending
              ? "Working…"
              : isRegister
                ? "Create account"
                : "Sign in"}
          </button>
        </form>

        <p className="auth-switch">
          {isRegister ? "Already have an account?" : "No account yet?"}{" "}
          <button type="button" onClick={() => setMode(!isRegister)}>
            {isRegister ? "Sign in" : "Create one"}
          </button>
        </p>
      </div>
    </main>
  );
}
