"""Logging configuration: the parts whose failure is silent.

Every assertion here is about a defect that produces *plausible* output. A
duplicated handler prints each line twice, which reads as retried work. A
leaked context stamps the wrong job id on a line, which reads as a correlation
you can trust. Neither raises, so neither is caught by anything else.
"""

import json
import logging

import pytest
import structlog

from app.logging_setup import (
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def restore_logging():
    """Put the root logger back however the suite had it.

    Without this, configuring logging in one test changes what every later
    test's output looks like -- and under random ordering that is a different
    set of tests each run.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers, root.level = saved_handlers, saved_level
    structlog.contextvars.clear_contextvars()


def _capture(caplog, fn):
    with caplog.at_level(logging.INFO):
        fn()
    return caplog.records


class TestConfiguration:
    def test_repeated_configuration_does_not_duplicate_handlers(self):
        """The naive version appends, and every line prints N times."""
        configure_logging()
        configure_logging()
        configure_logging()

        ours = [
            h
            for h in logging.getLogger().handlers
            if getattr(h, "_fitcheck_handler", False)
        ]
        assert len(ours) == 1

    def test_foreign_handlers_are_left_alone(self):
        """Configuring logging must not detach handlers it did not install.

        Clearing the root wholesale is the tempting version and it silently
        breaks pytest's caplog, plus anything the host process attached.
        """
        root = logging.getLogger()
        sentinel = logging.NullHandler()
        root.addHandler(sentinel)
        try:
            configure_logging()
            assert sentinel in root.handlers
        finally:
            root.removeHandler(sentinel)

    def test_a_plain_stdlib_logger_still_works(self, caplog):
        """The whole point of the ProcessorFormatter approach.

        Roughly a hundred existing call sites use `logger.info("x: %s", y)`.
        If those stopped working, this configuration would be a rewrite rather
        than a configuration.
        """
        configure_logging()
        records = _capture(
            caplog, lambda: logging.getLogger("legacy").info("value is %s", 42)
        )
        assert any(r.getMessage() == "value is 42" for r in records)

    def test_noisy_libraries_are_pinned_above_the_root_level(self):
        """Raising the root to DEBUG must not drown the output in SQL."""
        configure_logging()
        assert logging.getLogger("sqlalchemy.engine").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING


class TestContextBinding:
    """Asserted on rendered output, not on the LogRecord.

    Context keys are merged by the formatter at render time, so they are
    deliberately *not* attributes on the record -- which means `caplog` cannot
    see them and a test written against it would pass whether or not binding
    worked.
    """

    def test_bound_keys_reach_the_output(self, capsys):
        configure_logging()

        bind_context(job_id="abc123")
        logging.getLogger("worker").info("working")

        assert "job_id=abc123" in capsys.readouterr().out

    def test_clearing_context_actually_clears_it(self, capsys):
        """A leaked id is worse than no id: it is a confident wrong answer."""
        configure_logging()

        bind_context(job_id="first")
        logging.getLogger("worker").info("first job")
        clear_context()
        logging.getLogger("worker").info("second job")

        lines = capsys.readouterr().out.strip().splitlines()
        second = [line for line in lines if "second job" in line]

        assert second, "the second line should have been emitted"
        assert "first" not in second[0]


class TestJsonRendering:
    def test_json_format_emits_one_object_per_line(self, capsys, monkeypatch):
        from app.config import settings

        monkeypatch.setattr(settings, "log_format", "json")
        configure_logging()

        get_logger("t").info("structured", widget="sprocket")

        line = capsys.readouterr().out.strip().splitlines()[-1]
        payload = json.loads(line)

        assert payload["event"] == "structured"
        assert payload["widget"] == "sprocket"
        assert payload["level"] == "info"
        assert payload["timestamp"].endswith("Z")

    def test_json_exceptions_carry_a_real_traceback(self, capsys, monkeypatch):
        """Without format_exc_info this serialises as `<traceback object at 0x..>`,
        which is worse than nothing -- it looks like a traceback was captured."""
        from app.config import settings

        monkeypatch.setattr(settings, "log_format", "json")
        configure_logging()

        try:
            raise ValueError("boom")
        except ValueError:
            logging.getLogger("t").exception("failed")

        line = capsys.readouterr().out.strip().splitlines()[-1]
        payload = json.loads(line)

        assert "Traceback" in payload["exception"]
        assert "ValueError: boom" in payload["exception"]
