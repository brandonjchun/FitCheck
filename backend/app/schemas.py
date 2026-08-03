"""Pydantic models -- the API contract.

Deliberately separate from models.py (SQLAlchemy ORM models, the storage
shape) and extraction.py (the LLM output contract). All three describe a
profile; all three change for different reasons.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

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
