"""Pydantic models -- the API contract.

Deliberately separate from models.py (SQLAlchemy ORM models, the storage
shape) and extraction.py (the LLM output contract). All three describe a
profile; all three change for different reasons.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.extraction import Seniority


class ExtractedSkill(BaseModel):
    """A skill as surfaced to clients -- canonical name plus its evidence."""

    name: str
    years: float | None = None
    evidence: str | None = None


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
