# Worker image. The API tier still runs on the host during development; this
# exists so workers run on Linux, where RQ's default forking model works.
#
# RQ forks a child process per job so a crashed job cannot take the worker
# down with it. fork() does not exist on Windows, which is why running a
# plain `rq worker` there requires SimpleWorker -- a fallback that executes
# jobs in the worker's own process and therefore loses that isolation. A
# Linux container gets the real behaviour and matches how this would deploy.

FROM python:3.13-slim

# Fail fast and log immediately: unbuffered output means container logs
# appear in real time rather than when a buffer happens to flush.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies in their own layer so that editing application code does not
# reinstall 63 packages on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Non-root. The worker executes model output and parses untrusted documents;
# there is no reason for it to be able to write outside its own workspace.
RUN useradd --create-home --uid 1000 worker && chown -R worker:worker /app
USER worker

CMD ["rq", "worker", "--url", "redis://redis:6379/0", "default"]
