"""Application settings, read from the environment.

Secrets and per-environment values never get hardcoded. pydantic-settings
reads them from a .env file in development and from real environment
variables in production, validating types on the way in -- a malformed
DATABASE_URL fails at startup with a clear error rather than on the first
query.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env, resolved from this file rather than from the working
# directory.
#
# A bare ".env" is CWD-relative, which means the app reads a *different* file
# depending on where it was launched from: backend/.env under
# `cd backend && uvicorn ...`, and the repository root's .env under
# `pytest backend/tests` from the top level. That root file exists and holds
# the compose credentials (POSTGRES_PASSWORD, REDIS_PASSWORD), which this
# class does not declare -- so with extra="forbid" the app simply refused to
# start, and only from some directories.
#
# Anchoring to __file__ makes the answer the same everywhere.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Every value the app needs from its environment.

    Field names map to env vars case-insensitively: `database_url` is
    populated by DATABASE_URL.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        # Fail loudly if .env contains keys this class does not declare --
        # usually a typo, and silently ignoring it means debugging a setting
        # that "isn't being applied" for an hour.
        #
        # Note this only governs the *file* named above. Real environment
        # variables that happen to share a name with an undeclared field are
        # ignored rather than rejected, which is what lets the compose stack
        # export POSTGRES_PASSWORD into a worker container without the worker
        # refusing to boot.
        extra="forbid",
    )

    # postgresql+psycopg:// selects the psycopg 3 driver. The bare
    # postgresql:// prefix means psycopg2, which is not installed.
    #
    # These defaults carry a placeholder rather than a real password on
    # purpose. A working credential committed here is a credential in every
    # clone and every fork, and the one that used to live on this line was
    # also the one protecting the database. Real values come from .env, which
    # is gitignored; the placeholder fails authentication loudly rather than
    # letting a misconfigured process quietly reach a database it should not.
    #
    # The empty user in the Redis URL is not a typo -- Redis authenticates
    # with a password and no username unless ACLs are configured.
    #
    # 127.0.0.1 rather than localhost, and the difference is two seconds per
    # connection. Compose publishes these on the IPv4 loopback only, while
    # `localhost` on Windows resolves to ::1 first -- so every connect tries
    # IPv6, waits for the refusal, then falls back to IPv4. Measured at 2.05s
    # via localhost against 0.01s via 127.0.0.1, which across a test suite is
    # the difference between eight seconds and several minutes.
    database_url: str = "postgresql+psycopg://fitcheck:SET_IN_DOTENV@127.0.0.1:5432/fitcheck"
    redis_url: str = "redis://:SET_IN_DOTENV@127.0.0.1:6379/0"

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

    # --- LLM extraction (M2, split per task at M8) ---
    # Which provider backs each kind of extraction. Two settings rather than
    # one, because the two tasks have opposite requirements and one global
    # value has to be wrong for one of them.
    #
    # Resumes stay on Gemini: roughly one call per signup, and the quality gap
    # is real -- 10 skills with inferred years against a local model's 6, and
    # a seniority of "mid" where local returns "unknown". Every downstream
    # score reads this, so it is the wrong place to economise.
    llm_provider_profile: Literal["gemini", "ollama"] = "gemini"

    # Postings go local. Measured on a real posting, llama3.1:8b returned the
    # same five skills as Gemini with 5/5 verbatim evidence, in 11s against
    # 20s. And the volume is what forces it: the Gemini free tier is 20
    # requests per day, so a single 300-posting crawl would take two weeks.
    llm_provider_posting: Literal["gemini", "ollama"] = "ollama"

    # Where work goes when the provider above reports its quota exhausted.
    # None disables the fallback entirely, and a job then fails and retries
    # against the same exhausted provider -- correct if you would rather wait
    # for quality than proceed at lower quality, which is a real preference
    # and not the default one.
    llm_fallback_provider: Literal["gemini", "ollama"] | None = "ollama"

    # How long to route around a provider when it gives no estimate of its
    # own. Four hours is a compromise between the two limits Gemini's free
    # tier enforces: a per-minute cap that clears in under a minute, and a
    # per-day cap that clears at midnight Pacific. When the error names a
    # retry delay -- and Gemini's does -- that number is used instead and
    # this is never consulted.
    llm_quota_cooldown_seconds: int = 4 * 60 * 60

    # Nothing routes around a provider for longer than this, however large an
    # estimate it returns. A provider answering "retry in 30 days" would
    # otherwise silently become permanent, and a config value nobody
    # remembers setting is a worse outage than a daily one.
    llm_quota_max_cooldown_seconds: int = 24 * 60 * 60

    # Gemini: free tier key from aistudio.google.com. Optional so the app can
    # start on the ollama path without one.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.6-flash"

    # Ollama: local daemon, no key.
    #
    # `host.docker.internal` rather than `localhost`, because the workers that
    # do the extracting run in containers and `localhost` there is the
    # container itself. Pointing at localhost from inside one produces
    # connection-refused, which app/providers/ollama.py classifies as
    # transient -- so it retries with backoff, forever, against nothing.
    # Docker Desktop resolves this name to the host; on Linux the compose
    # file adds the corresponding extra_hosts entry.
    ollama_host: str = "http://host.docker.internal:11434"

    # 8B rather than the 14B also installed. Measured on the same resume,
    # llama3.1:8b quoted verbatim evidence for 29 of 33 skills where
    # qwen2.5:14b managed 2 of 16 -- nearly twice the parameters and far worse
    # at the single instruction that makes an extraction auditable. Parameter
    # count is not the axis that matters here.
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"

    # --- Outbound fetching (M5) ---
    # Identify the crawler honestly, with a way to be contacted. Impersonating
    # a browser to evade detection is the thing that gets a project blocked
    # and is not defensible in an interview; a real UA means an annoyed site
    # operator can email rather than firewall.
    fetch_user_agent: str = (
        "FitCheckBot/0.1 (+https://github.com/brandonjchun/FitCheck; "
        "student project; contact via repository issues)"
    )

    # USAJOBS is the one board kind that requires a credential, and it is free
    # and instant from developer.usajobs.gov. Empty by default so the rest of
    # the app starts without it -- `enumerate_usajobs` raises a named error when
    # a source of that kind is crawled unconfigured, which is a better failure
    # than a startup that refuses to boot over a board nobody seeded.
    #
    # Two values rather than one because their terms ask that the User-Agent be
    # the email the key was registered to. Reusing `fetch_user_agent` would send
    # them a string they cannot tie to an account.
    usajobs_api_key: str = ""
    usajobs_email: str = ""

    # Split rather than one number. Connect failures should surface fast --
    # a host that will not accept a TCP connection in 5s is down, and waiting
    # 20 to learn that wastes a worker slot. Read gets longer because a slow
    # server is still a working one.
    fetch_connect_timeout: float = 5.0
    fetch_read_timeout: float = 20.0

    # Abort a response past this many bytes. Enforced while streaming, not
    # after: checking Content-Length alone is trivially defeated by a server
    # that omits it or lies, and by then the bytes are already in memory.
    # 5 MB is far past any job posting and well short of trouble.
    fetch_max_bytes: int = 5 * 1024 * 1024

    # Redirects are followed, but not indefinitely -- a redirect loop would
    # otherwise consume the whole job timeout.
    fetch_max_redirects: int = 5

    # Per-domain rate limiting, enforced in Redis across every worker process.
    # In-process limiting is useless here: four workers each politely holding
    # themselves to 1 rps produce 4 rps at the target, which is how projects
    # get IP-banned and then blame the code.
    #
    # One request per second sustained, with a small burst so a handful of
    # URLs from the same board do not each wait a full second.
    fetch_rate_per_second: float = 1.0
    fetch_rate_burst: int = 3

    # How long a worker will wait for a rate-limit token before giving up and
    # letting the job retry. Bounded because a worker blocked on a token is a
    # worker doing nothing -- past this it is better to release the slot and
    # let backoff reschedule the work.
    fetch_rate_max_wait_seconds: float = 10.0

    # robots.txt is cached per host for this long. Re-fetching it before each
    # of 400 postings on one board is both wasteful and rude.
    robots_cache_seconds: int = 3600

    # --- Batch URL upload (M4) ---
    # The cap that matters, and it is a count rather than a size. The 2 MB
    # body limit above bounds *bytes*, and 2 MB of text is roughly 40,000
    # URLs -- so the upload limit looks like a bound on work and is not one.
    # Each accepted URL becomes an outbound HTTP request to somebody else's
    # server, which is what actually needs bounding.
    #
    # 500 is chosen to be comfortably larger than a real hand-collected list
    # and far smaller than a denial-of-service. Lines past it are reported
    # back as rejected rather than silently dropped.
    max_urls_per_batch: int = 500

    # How many batches one user may have running at once. Without it the
    # per-batch cap is trivially bypassed by uploading the same file twenty
    # times, which is the usual way a per-request limit turns out not to be a
    # limit at all.
    max_open_batches_per_user: int = 3

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

    # --- Logging --------------------------------------------------------
    #
    # Two knobs rather than one "debug" flag, because the level and the shape
    # of a log line are independent choices: chasing a bug in production means
    # turning the level down to DEBUG while keeping JSON, and a boolean would
    # force those together.
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # `console` is the coloured, human-aligned renderer; `json` is one object
    # per line for a log shipper to parse.
    #
    # Default is `console` because the common case for this project is a
    # person reading `docker compose logs`, and JSON is materially worse to
    # read unaided. Anything collecting these centrally should set `json` --
    # that is the whole reason the renderer is configurable rather than
    # chosen once at import.
    log_format: Literal["console", "json"] = "console"


# Imported as `from app.config import settings` -- constructed once at import
# time, so validation failures surface at startup.
settings = Settings()
