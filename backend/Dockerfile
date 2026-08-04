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
# reinstall every package on every rebuild.
#
# The PyTorch CPU index is not optional. sentence-transformers depends on
# torch, and the default PyPI wheel is the CUDA build -- roughly 2.5 GB of
# GPU runtime, in an image whose container has no GPU and never will. The
# CPU wheel is a fraction of that and runs MiniLM in ~10ms, which is far
# below the network time of the fetch that precedes it.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Bake the embedding model into the image rather than downloading it on the
# first job.
#
# Three reasons, in order of how badly they bite. A cold worker would
# otherwise spend its first job downloading 90 MB, so the first user request
# after every deploy is slow for a reason nothing reports. Scaling to N
# workers means N downloads of the same file. And it makes the worker
# depend on huggingface.co being reachable at *runtime* -- a build-time
# dependency is one you find out about in CI, a runtime one is an outage.
ENV HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && chmod -R a+rX /opt/hf

COPY app ./app
# The scheduler and the operational one-offs (seeding, backfill, benchmark)
# run inside this image, so they have to be in it. Copied separately from
# `app` to keep the layer cache useful: these change on a different rhythm
# than application code.
COPY scripts ./scripts
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

# Non-root. The worker executes model output and parses untrusted documents;
# there is no reason for it to be able to write outside its own workspace.
RUN useradd --create-home --uid 1000 worker && chown -R worker:worker /app
USER worker

# Overridden by docker-compose.yml, which names the real queues. The default
# here names `interactive` rather than the long-gone `default` queue, so a
# bare `docker run` of this image consumes something that exists.
CMD ["rq", "worker", "--url", "redis://redis:6379/0", "interactive"]
