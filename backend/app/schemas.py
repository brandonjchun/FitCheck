"""Pydantic models -- the API contract.

Deliberately separate from models.py (SQLAlchemy ORM models, the storage
shape). Conflating them means a database column rename becomes a breaking
API change, and internal columns leak to clients by default.
"""

from pydantic import BaseModel


class ProfileUploadResponse(BaseModel):
    """What POST /api/profiles returns once a resume has been parsed."""

    filename: str
    characters: int
    raw_text: str
