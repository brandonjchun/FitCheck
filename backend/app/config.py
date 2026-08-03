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


# Imported as `from app.config import settings` -- constructed once at import
# time, so validation failures surface at startup.
settings = Settings()
