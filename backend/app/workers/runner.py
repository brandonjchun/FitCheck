"""The worker class the compose services run.

`rq worker` has no startup hook, so without this there is nowhere for a worker
process to call `configure_logging()` -- its output would go through whatever
logging configuration happened to exist by accident, which in practice means
unformatted lines with no timestamps.

The second job is correlation, and it is the one that matters. A worker runs
one job at a time but four workers run concurrently, and their output is
interleaved in `docker compose logs`. Every line below carries `job_id`,
`job_func`, and `queue`, so reconstructing one job's history is a filter
rather than an exercise in reading timestamps and guessing.

**Why a Worker subclass rather than RQ's `exception_handlers` or a decorator
on each task.** The handler hooks only fire on failure, and the interesting
lines are usually the ones before it. A decorator would have to be remembered
on every new task, and the one that gets forgotten is the one being debugged
at 2am. Binding in `perform_job` covers every task automatically, including
ones that do not exist yet.
"""

from __future__ import annotations

import rq.logutils
import rq.worker
from rq import Worker
from rq.job import Job
from rq.queue import Queue

from app.logging_setup import bind_context, clear_context, configure_logging


def _no_op_loghandlers(*_args, **_kwargs) -> None:
    """Stop RQ installing its own handlers on `rq.worker` and `rq.job`.

    RQ calls `setup_loghandlers(name='rq.worker')` during bootstrap and again
    as the worker starts listening. Each call attaches a handler directly to
    that logger, and because those loggers also propagate to the root -- where
    this app's handler lives -- every RQ line is emitted twice: once in RQ's
    format, once in ours. A queue that appears to log everything twice reads
    as a queue that is doing everything twice.

    Neutered at the module level rather than by overriding a method, because
    `Worker.bootstrap` calls the function through its own module namespace;
    a subclass override is simply not consulted. Stripping the handlers after
    the fact was tried first and loses to the second call.

    The lines are not lost. Both loggers keep `propagate = True`, so every one
    of them arrives at the root handler and is rendered in this app's format,
    with the job correlation `LoggingWorker` adds.
    """
    return None


rq.logutils.setup_loghandlers = _no_op_loghandlers
# `rq.worker` imported the symbol directly (`from rq.logutils import
# setup_loghandlers`), so patching the source module alone leaves the worker's
# own reference pointing at the original.
rq.worker.setup_loghandlers = _no_op_loghandlers


class LoggingWorker(Worker):
    """An RQ worker whose every log line names the job that produced it."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Configured once per process rather than per job. Per job would be
        # correct and wasteful -- it rebuilds the handler chain thousands of
        # times a day for output that never changes.
        configure_logging()

        # The `rq` CLI calls `setup_loghandlers_from_args` before it ever
        # constructs a worker, and that attaches a handler to the `rq.worker`
        # logger specifically -- not to the root. So `configure_logging`'s
        # root-level handling never sees it, `rq.worker` keeps propagating to
        # the root as well, and every worker line is emitted twice in two
        # different formats. A queue that appears to log everything twice
        # reads as a queue that is doing everything twice.
        #
        # Note the ordering that makes this unavoidable: RQ installs its
        # handler only `if not _has_effective_handler(logger)`, and at CLI
        # time there is none, because this app's handler is not installed
        # until the line above. Being polite about foreign handlers cannot
        # help here -- the handler did not exist yet when the policy ran.
        #
        # Cleared rather than `propagate = False`, because the goal is one
        # copy *in this app's format* carrying the job correlation added
        # below. Silencing propagation would keep RQ's format and lose that.
        #
        # Scoped to this entry point rather than pushed into
        # `configure_logging`: a worker process is a place where "exactly one
        # log format" is safe to insist on, since nothing else owns its
        # stdout. Elsewhere -- tests, an embedding host process -- it is not.
        import logging as _logging

        for name in ("rq.worker", "rq.job", "rq.scheduler"):
            rq_logger = _logging.getLogger(name)
            rq_logger.handlers = []
            rq_logger.propagate = True

    def perform_job(self, job: Job, queue: Queue) -> bool:
        bind_context(
            job_id=job.id,
            # The dotted path, not the callable: a worker resolves the
            # function lazily, and touching `job.func` here would import the
            # task module just to write a log line -- and raise inside the
            # logging path if that import fails, turning a task bug into a
            # worker crash.
            job_func=job.func_name,
            queue=queue.name,
        )
        try:
            return super().perform_job(job, queue)
        finally:
            # In a `finally` because RQ reuses this process for the next job.
            # A leaked job_id would stamp the previous job's id onto the next
            # one's output, which is worse than no id: it is a wrong answer
            # that looks authoritative.
            clear_context()
