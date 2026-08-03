"""Pydantic models -- the API contract.

Deliberately separate from models.py (SQLAlchemy ORM models, the storage
shape) and extraction.py (the LLM output contract). All three describe a
profile; all three change for different reasons.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, HttpUrl

from app.extraction import Seniority, SkillSource
from app.models import JobStatus


class ExtractedSkill(BaseModel):
    """A skill as surfaced to clients -- canonical name plus its evidence."""

    name: str
    years: float | None = None
    evidence: str | None = None

    # Optional here even though the LLM contract makes it required, because
    # this model is built from stored JSONB rather than from a fresh
    # extraction. Profiles written before extraction version 3 have no
    # `source` key, and a required field would turn every one of them into a
    # 500 on read. Null means "extracted before this was captured", which is
    # information the client can act on; a re-extraction fills it in.
    source: SkillSource | None = None


class ProfileUploadResponse(BaseModel):
    """What POST /api/profiles returns once a resume has been stored.

    from_attributes lets FastAPI build this from a SQLAlchemy Profile object
    by reading attributes rather than requiring a dict.

    `extraction_ok` is deliberately explicit. Text extraction and LLM
    extraction fail independently: a resume can parse perfectly and still
    come back with no structured profile if the provider was down. Rather
    than failing the upload -- which would discard raw_text we successfully
    parsed -- the row is saved with null extraction and this flag tells the
    client which happened. A client seeing skills=[] otherwise cannot
    distinguish "no skills found" from "extraction never ran".
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    characters: int
    created_at: datetime

    extraction_ok: bool
    seniority: Seniority | None = None
    years_experience: float | None = None
    skills: list[ExtractedSkill] = []

    raw_text: str


class RegisterRequest(BaseModel):
    """Body of POST /api/auth/register.

    `EmailStr` rejects a malformed address with a field-level 422 before any
    row is written. The minimum length is deliberately a floor rather than a
    composition rule -- no "must contain a symbol", because those rules push
    people toward `Password1!` and measurably reduce entropy compared with
    simply requiring more characters.
    """

    email: EmailStr
    password: str = Field(min_length=12, max_length=1024)


class LoginRequest(BaseModel):
    """Body of POST /api/auth/login.

    No length constraints. Validating the shape of a *submitted* password
    tells an attacker the policy and rejects legacy passwords that predate a
    rule change; the only question at login is whether it matches.
    """

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """A user as returned to clients. Note what is absent: password_hash.

    This is the concrete reason ORM models are not reused as API responses.
    Returning a `User` directly would serialize every column it has, and the
    first person to notice would be whoever reads the hash out of the
    response body.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    created_at: datetime

    # Exposed so the client can avoid offering a link that would only 403.
    # This is presentation, not enforcement -- the server checks the same flag
    # on every ops request, and a client that lies to itself about this gains
    # nothing but a different error message.
    is_admin: bool = False


class BatchCreateResponse(BaseModel):
    """What POST /api/batches returns once a URL list has been accepted.

    Every line the user sent is accounted for: `accepted + rejected +
    duplicates` equals the non-blank line count of their file. A batch that
    quietly ingests 500 of someone's 4,000 lines is worse than one that
    refuses, because they cannot tell which 3,500 are missing.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    profile_id: int
    filename: str
    accepted: int
    rejected: int
    duplicates: int
    created_at: datetime


class BatchStatusResponse(BaseModel):
    """Aggregate progress for one batch. The endpoint a client polls.

    One request covers the whole batch rather than N requests for N jobs --
    polling 500 individual job endpoints every two seconds is how a progress
    view becomes the heaviest thing in the system.

    The counts are derived by grouping jobs on `batch_id`, never read from a
    stored counter. See models.UrlBatch for why.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    profile_id: int
    filename: str
    total: int
    rejected: int
    duplicates: int
    created_at: datetime

    # Keyed by job status: queued, running, succeeded, failed, dead. Absent
    # keys mean zero -- the client sums what it gets rather than assuming a
    # fixed set, so adding a lifecycle state later is not a breaking change.
    counts: dict[str, int]

    # True once no job is still queued or running. Lets the client stop
    # polling without hardcoding which statuses are terminal, the same way
    # JobResponse.is_terminal does for a single job.
    is_complete: bool


class JobSubmitRequest(BaseModel):
    """Body of POST /api/jobs.

    `HttpUrl` is doing real work here: a malformed URL is rejected with a
    field-level 422 before any row is written or any job enqueued. Without
    it, the bad value reaches a worker minutes later and fails there, which
    is a far worse place to discover a typo.
    """

    url: HttpUrl
    profile_id: int
    notes: str | None = Field(default=None, max_length=500)


class JobResponse(BaseModel):
    """A job's current state. This is what the frontend polls."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    profile_id: int
    url: str
    status: JobStatus
    attempts: int
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime

    # Lets the client stop polling without hardcoding which statuses are
    # terminal. Duplicating that set in the frontend guarantees the two drift
    # the first time a state is added.
    is_terminal: bool


# --- Ops dashboard ------------------------------------------------------
#
# System-wide views, as opposed to the per-user contracts above. See
# routers/ops.py for the authorization caveat.


class WorkerInfo(BaseModel):
    """One RQ worker process and the queues it drains."""

    name: str
    state: str
    # Ordered, because it is the *priority* order RQ checks them in, not a set.
    queues: list[str]
    current_job_id: str | None = None
    successful_jobs: int
    failed_jobs: int


class QueueHealth(BaseModel):
    """One queue's depth and the registries around it."""

    name: str
    depth: int
    started: int
    failed: int
    deferred: int
    scheduled: int

    # False when the queue exists in Redis but not in QUEUE_NAMES -- work
    # left behind by a rename, which nothing will ever consume.
    declared: bool
    # Zero here alongside a non-zero depth is the stranded-queue signature:
    # jobs waiting, nobody listening.
    worker_count: int


class StatusCount(BaseModel):
    status: str
    count: int


class OpsOverview(BaseModel):
    """Everything the dashboard polls, in one response."""

    queues: list[QueueHealth]
    workers: list[WorkerInfo]
    jobs_by_status: list[StatusCount]
    job_timeout_seconds: int
    result_ttl_seconds: int
    failure_ttl_seconds: int


class DeadLetterItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    url: str
    status: str
    attempts: int
    last_error: str | None = None
    updated_at: datetime


class RequeueResponse(BaseModel):
    id: int
    status: str
    queue: str
