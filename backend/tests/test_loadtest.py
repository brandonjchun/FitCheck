"""The load generator is code, and the ways it can be wrong are quiet ones.

A locustfile that imports cleanly and produces confident numbers about the
wrong thing is worse than one that crashes. Two failures are worth guarding
and neither is visible in a results table:

- **It points at a real host.** Then the p95 is somebody else's server, the
  experiment is invalid, and a burst generator has been aimed at a third
  party. `.invalid` is what prevents that, and it is one careless edit away.
- **Its cleanup misses a table.** Then the ops dashboard's dead-letter list
  fills with load-test rows that mask real failures, and `users` stops being
  a number anyone can quote.

**These read the locustfile rather than importing it, deliberately.** Locust
calls `gevent.monkey.patch_all()` at import, which rewrites `socket`, `ssl`,
and `threading` for the whole process -- so importing it here would patch SSL
underneath httpx and anyio for the other 755 tests in the session, and gevent
warns that the result can be silently wrong rather than loudly broken. The
constants live in `loadtest.config`, which is plain Python, and the structure
is checked by parsing. The harness's real behaviour is verified by running it;
the numbers that produced are in the README.
"""

import ast
import inspect
from pathlib import Path

import pytest

from loadtest import cleanup
from loadtest.config import EMAIL_PREFIX, PASSWORD, UNREACHABLE

LOCUSTFILE = Path(__file__).resolve().parent.parent / "loadtest" / "locustfile.py"
SOURCE = LOCUSTFILE.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

USER_CLASSES = {
    node.name: node
    for node in TREE.body
    if isinstance(node, ast.ClassDef)
    and any(getattr(base, "id", "") == "HttpUser" for base in node.bases)
}


def _class_weight(node: ast.ClassDef) -> int:
    for statement in node.body:
        if (
            isinstance(statement, ast.Assign)
            and getattr(statement.targets[0], "id", "") == "weight"
        ):
            return statement.value.value
    raise AssertionError(f"{node.name} declares no weight")


def _has_decorated(node: ast.ClassDef, name: str) -> bool:
    return any(
        isinstance(item, ast.FunctionDef)
        and any(
            getattr(d, "id", "") == name or getattr(getattr(d, "func", None), "id", "") == name
            for d in item.decorator_list
        )
        for item in node.body
    )


class TestItCannotTouchAThirdParty:
    def test_the_submit_host_is_a_reserved_tld(self) -> None:
        """RFC 2606 reserves `.invalid` precisely so it can never resolve.

        The submit path is still exercised in full -- validation, row, enqueue,
        worker pickup -- and the fetch dies at DNS with no packet leaving the
        machine.
        """
        assert UNREACHABLE.endswith(".invalid")
        assert UNREACHABLE.startswith("https://")

    @pytest.mark.parametrize(
        "host", ["greenhouse", "lever.co", "ashbyhq", "workable", "indeed", "linkedin"]
    )
    def test_no_real_board_appears_anywhere_in_the_harness(self, host: str) -> None:
        """A pasted real URL turns a load test into an outbound burst against
        somebody else's rate limit."""
        assert host not in SOURCE.lower()

    def test_the_only_submitted_host_is_the_unreachable_one(self) -> None:
        """Every URL the generator submits is built from UNREACHABLE. A second
        literal host would slip past the constant."""
        literals = [
            node.value
            for node in ast.walk(TREE)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ]
        for value in literals:
            if value.startswith(("http://", "https://")):
                assert value == UNREACHABLE, f"unexpected host literal {value!r}"


class TestTheMixIsHonest:
    def test_all_three_paths_are_represented(self) -> None:
        """M11 asks for a burst on all the paths, not the convenient one."""
        assert set(USER_CLASSES) == {"FeedReader", "UrlSubmitter", "BatchSubmitter"}

    def test_reads_outweigh_writes(self) -> None:
        """§6.1 puts a human in front of the read path and nobody in front of
        the bulk one. Weighted the other way, the headline p95 would be
        dominated by the path with the loosest latency budget."""
        assert (
            _class_weight(USER_CLASSES["FeedReader"])
            > _class_weight(USER_CLASSES["UrlSubmitter"])
            > _class_weight(USER_CLASSES["BatchSubmitter"])
        )

    @pytest.mark.parametrize(
        "name", ["FeedReader", "UrlSubmitter", "BatchSubmitter"]
    )
    def test_each_class_authenticates_before_it_measures(self, name: str) -> None:
        """An unauthenticated run measures the 401 path: fast, uniform, and
        meaningless."""
        body = ast.get_source_segment(SOURCE, USER_CLASSES[name])
        assert "on_start" in body
        assert "_register_and_login" in body

    @pytest.mark.parametrize(
        "name", ["FeedReader", "UrlSubmitter", "BatchSubmitter"]
    )
    def test_each_class_has_at_least_one_task(self, name: str) -> None:
        assert _has_decorated(USER_CLASSES[name], "task"), f"{name} would idle"

    def test_the_password_meets_the_registration_rule(self) -> None:
        """A password the API rejects makes every simulated user a 422 and the
        whole run a measurement of the validation path."""
        assert len(PASSWORD) >= 12


class TestQueueDepthIsSampled:
    def test_the_run_records_depth_as_well_as_latency(self) -> None:
        """Half of M11's definition of done. A p95 of 12 ms on submit says
        nothing if `ingest` is 40,000 deep behind it."""
        assert "queue_depths" in SOURCE
        assert "test_start" in SOURCE and "test_stop" in SOURCE

    def test_sampling_failure_cannot_fail_the_run(self) -> None:
        """Instrumentation that can take down the experiment it is measuring
        is worse than no instrumentation."""
        assert "except Exception" in SOURCE


class TestCleanupCoversWhatTheRunCreates:
    def test_it_shares_the_constants_the_run_registers_under(self) -> None:
        """Two copies would drift, and the cleanup would report success while
        removing nothing."""
        assert cleanup.EMAIL_PREFIX == EMAIL_PREFIX
        assert cleanup.UNREACHABLE == UNREACHABLE

    def test_it_reaches_the_table_the_cascade_does_not(self) -> None:
        """`job_postings` is deliberately unowned, so deleting users does not
        remove it -- the exact table a cleanup forgets."""
        source = inspect.getsource(cleanup.main)
        assert "job_postings" in source
        assert "users" in source
        assert "ingest_jobs" in source

    def test_it_does_not_import_locust(self) -> None:
        """Cleanup runs against the database; pulling in a monkey-patching
        load generator to read two strings would be a strange thing to do to a
        maintenance script."""
        assert "locustfile" not in inspect.getsource(cleanup)
