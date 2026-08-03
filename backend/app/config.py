"""Application settings, read from the environment.

Secrets and per-environment values never get hardcoded. pydantic-settings
reads them from a .env file in development and from real environment
variables in production, validating types on the way in -- a malformed
DATABASE_URL fails at startup with a clear error rather than on the first
query.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every value the app needs from its environment.

    Field names map to env vars case-insensitively: `database_url` is
    populated by DATABASE_URL.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Fail loudly if .env contains keys this class does not declare --
        # usually a typo, and silently ignoring it means debugging a setting
        # that "isn't being applied" for an hour.
        extra="forbid",
    )

    # postgresql+psycopg:// selects the psycopg 3 driver. The bare
    # postgresql:// prefix means psycopg2, which is not installed.
    database_url: str = "postgresql+psycopg://fitcheck:devpassword@localhost:5432/fitcheck"
    redis_url: str = "redis://localhost:6379/0"

    # 2 MB. Enforced by BodySizeLimitMiddleware.
    max_upload_bytes: int = 2 * 1024 * 1024

    # Connection pool sizing. Postgres defaults to max_connections = 100, and
    # every API process and worker process holds its own pool, so the ceiling
    # is (processes x (pool_size + max_overflow)) -- not one number.
    #
    # These defaults are sized for the API tier, which serves many concurrent
    # requests per process and genuinely wants headroom. Workers do not: each
    # one runs a single job at a time, and app.workers.tasks opens its
    # sessions sequentially rather than concurrently, so one connection is
    # usually the true working set. docker-compose.yml overrides both values
    # down for the worker service -- scaling workers is meant to add capacity,
    # not to march toward max_connections.
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # --- LLM extraction (M2) ---
    # Which provider backs structured extraction. Both are implemented; this
    # decides which one get_provider() returns. Flip it in .env, restart, done
    # -- no code change, which is the point of the abstraction.
    llm_provider: Literal["gemini", "ollama"] = "gemini"

    # Gemini: free tier key from aistudio.google.com. Optional so the app can
    # start on the ollama path without one.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"

    # Ollama: local daemon, no key. Default host is the standard local port.
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # --- Sessions and auth (M3) ---
    # Signs the session cookie. The signature is not what keeps a session
    # secret -- the session id is 256 bits of randomness and the body lives in
    # Redis -- but it lets a tampered or foreign cookie be rejected without a
    # Redis round trip, and it stops a client treating the cookie as a value
    # it can edit.
    #
    # The default is a development convenience and nothing more. Rotating it
    # invalidates every outstanding cookie, which is the intended behaviour on
    # a suspected leak. Production sets SESSION_SECRET to a real random value;
    # a deployment that forgets is running with a published key.
    session_secret: str = "dev-only-session-secret-change-me"

    # Name of the cookie carrying the signed session id.
    session_cookie_name: str = "fc_session"

    # How long a session lives, in seconds. Applied twice on purpose: as the
    # signature's max_age and as the Redis TTL. The signature bound means an
    # expired cookie is rejected before touching Redis; the Redis bound is the
    # authoritative one, and is what makes logout and revocation immediate.
    session_ttl_seconds: int = 7 * 24 * 60 * 60

    # Sets the cookie's Secure flag, which tells the browser to send it over
    # HTTPS only. Off by default because local development is plain http on
    # localhost and a Secure cookie would simply never be sent. Production
    # must turn this on -- without it the session id crosses the network in
    # the clear on any downgraded request.
    session_cookie_secure: bool = False

    # Origins allowed to make credentialed cross-origin requests. The Vite dev
    # server default is listed because the frontend will run there while the
    # API runs on 8000, and a session cookie is not sent cross-origin without
    # this. Note that "*" is invalid alongside credentials -- browsers reject
    # that pairing, correctly, since it would let any site spend a user's
    # session.
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


# Imported as `from app.config import settings` -- constructed once at import
# time, so validation failures surface at startup.
settings = Settings()
