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

from rq import Worker
from rq.job import Job
from rq.queue import Queue

from app.logging_setup import bind_context, clear_context, configure_logging


class LoggingWorker(Worker):
    """An RQ worker whose every log line names the job that produced it."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Configured once per process rather than per job. Per job would be
        # correct and wasteful -- it rebuilds the handler chain thousands of
        # times a day for output that never changes.
        configure_logging()

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
