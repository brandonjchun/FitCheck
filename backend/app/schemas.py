"""Pydantic models -- the API contract.

Deliberately separate from models.py (SQLAlchemy ORM models, the storage
shape). Conflating them means a database column rename becomes a breaking
API change, and internal columns leak to clients by default.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ProfileUploadResponse(BaseModel):
    """What POST /api/profiles returns once a resume has been stored.

    from_attributes lets FastAPI build this from a SQLAlchemy Profile object
    by reading attributes, rather than requiring a dict. Note what is absent:
    `extracted`, `seniority`, and `years_experience` exist as columns but are
    not exposed here. That is the separation earning its keep -- the storage
    shape and the API contract move independently.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    characters: int
    raw_text: str
    created_at: datetime
