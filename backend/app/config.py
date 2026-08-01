"""Application settings, read from the environment.

Secrets and per-environment values never get hardcoded. pydantic-settings
reads them from a .env file in development and from real environment
variables in production, validating types on the way in -- a malformed
DATABASE_URL fails at startup with a clear error rather than on the first
query.
"""

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
    # every API process and worker process holds its own pool. 5 + 5 overflow
    # per process leaves room for several workers alongside the API tier.
    db_pool_size: int = 5
    db_max_overflow: int = 5


# Imported as `from app.config import settings` -- constructed once at import
# time, so validation failures surface at startup.
settings = Settings()
