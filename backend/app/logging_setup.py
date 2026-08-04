"""One logging configuration, applied by every entry point.

Spec section 12 counts observability, and the project's own limitations doc
recorded `structlog` as installed and unused. This is that gap closed.

**The design constraint that shapes everything here: the existing code stays
untouched.** There are on the order of a hundred `logger.info("thing: %s", x)`
calls across routers, workers, and providers. Rewriting them into structlog's
keyword style would be a large diff whose only effect is churn, and it would
have to be repeated by anyone adding a log line who forgot which style this
file expects. So stdlib logging remains the interface, and structlog is
installed *underneath* it as the renderer via `ProcessorFormatter`. Every
existing call site keeps working and gains structure for free.

**What structure buys, concretely.** Two things a plain string cannot:

- *Correlation.* `bind_context(job_id=...)` attaches a key to every line a
  worker emits for the rest of that job, including lines from libraries that
  know nothing about jobs. Reconstructing one job's history out of interleaved
  output from four workers is otherwise guesswork.
- *Machine-readability.* Under `log_format=json` each line is one object, so
  "how many extractions failed permanently last night" is a query rather than
  a grep with a regex that breaks on the first message someone rewords.

**Idempotent by design.** `configure_logging()` may be called more than once
-- the API calls it at startup, tests import the app repeatedly, and a worker
that imports the app package gets it too. Re-running it replaces handlers
rather than appending, because the failure mode of the naive version is every
line printed N times, which looks like a retry bug.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import settings

# Keys structlog puts on the event dict that a console reader does not need to
# see repeated on every line.
_CONSOLE_NOISE = ("thread_name",)

_configured = False


def _shared_processors() -> list[Any]:
    """Processors applied to both structlog and stdlib records.

    Order matters and is not arbitrary: context has to be merged before
    anything renders, the timestamp has to exist before a renderer looks for
    it, and exception formatting has to come last so it sees the final event
    dict.
    """
    return [
        # Pulls in whatever `bind_context` put on the current context var.
        # This is what makes correlation ids appear on lines whose call site
        # knows nothing about them.
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        # UTC, ISO 8601. Not local time: these are read next to Postgres and
        # Redis output, and three timezones in one terminal is how an
        # ordering bug gets misdiagnosed.
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]


def configure_logging() -> None:
    """Install the logging configuration. Safe to call repeatedly."""
    global _configured

    shared = _shared_processors()
    as_json = settings.log_format == "json"

    if as_json:
        renderer: Any = structlog.processors.JSONRenderer()
        # ConsoleRenderer formats exc_info itself; JSONRenderer does not, and
        # without this an exception serialises as the repr of a traceback
        # object -- `"<traceback object at 0x...>"`. That is worse than no
        # traceback, because it looks like one was captured.
        exception_processors: list[Any] = [structlog.processors.format_exc_info]
    else:
        exception_processors = []
        # `colors` is left on: Docker Compose logs render ANSI fine, and a
        # developer reading them is the case this format exists for.
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared,
            # Hands off to the stdlib formatter below rather than rendering
            # here. Without this, structlog output and stdlib output would be
            # formatted by two different code paths and drift apart.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        # Caching binds the wrapper class on first use. Fine in production and
        # wrong under tests, which reconfigure between cases -- so it is
        # disabled and the cost is a dictionary lookup per call.
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # `foreign_pre_chain` is what makes a plain `logging.getLogger(...)`
        # call -- every existing line in this codebase, plus uvicorn's and
        # SQLAlchemy's -- pass through the same processors as a structlog one.
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *([_drop_console_noise] if not as_json else []),
            *exception_processors,
            renderer,
        ],
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    # Marked so a second call can find and replace *this* handler rather than
    # every handler. Appending blindly would print each line once per call to
    # configure_logging, which reads as duplicated work rather than duplicated
    # output -- and clearing blindly would remove handlers this module did not
    # install, which is how calling a logging setup function silently breaks
    # pytest's caplog and any sidecar the host process attached.
    handler._fitcheck_handler = True  # type: ignore[attr-defined]

    root = logging.getLogger()
    root.handlers = [
        h for h in root.handlers if not getattr(h, "_fitcheck_handler", False)
    ]
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # SQLAlchemy echoes every statement at INFO when enabled, and uvicorn logs
    # an access line per request. Both are useful deliberately and noise by
    # default, so they are pinned above the root level rather than silenced --
    # raising the root to DEBUG should not drown the output in SQL.
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # httpx logs an INFO line for every outbound request. During a 500-posting
    # crawl that is 500 lines saying what the crawl already reports.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _configured = True


def _drop_console_noise(_logger, _name, event_dict: dict) -> dict:
    for key in _CONSOLE_NOISE:
        event_dict.pop(key, None)
    return event_dict


def bind_context(**kwargs: Any) -> None:
    """Attach keys to every subsequent log line on this context.

    Scoped to the current contextvar, which means it follows an async request
    through awaits and does not leak between concurrently-handled requests.
    In a worker, one job is one context.
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Drop everything `bind_context` attached.

    Called at the end of a request or job. Skipping it in a worker is how a
    job id from a previous job ends up stamped on an unrelated one, which is
    worse than no correlation id at all -- it is a confidently wrong one.
    """
    structlog.contextvars.clear_contextvars()


def get_logger(name: str | None = None):
    """A structlog logger, for new code that wants the keyword style.

    Existing modules keep using `logging.getLogger(__name__)` and lose
    nothing; both end up in the same pipeline.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
