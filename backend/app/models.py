"""SQLAlchemy ORM models -- the storage shape.

Deliberately separate from schemas.py (Pydantic, the API contract). A column
rename here should not be a breaking API change, and internal columns should
not leak to clients by default.
"""

from datetime import datetime
from decimal import Decimal

from sqlalchemy import BigInteger, DateTime, Numeric, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Profile(Base):
    """One uploaded resume and everything derived from it."""

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    # The hybrid from spec section 3.3: the full LLM extraction blob lives in
    # JSONB, and the two fields we actually filter on are promoted to real
    # columns. Both are nullable because M1 only stores text -- M2 populates
    # them.
    extracted: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    seniority: Mapped[str | None] = mapped_column(Text, nullable=True)
    years_experience: Mapped[Decimal | None] = mapped_column(
        Numeric(4, 1), nullable=True
    )

    # embedding: vector(384) is added at M6, together with the pgvector
    # SQLAlchemy type. The extension is available in the image already.

    # server_default means Postgres fills this in, not Python -- so rows
    # inserted by a migration or by psql get a timestamp too. timezone=True
    # stores timestamptz; storing naive local times is a bug you find in
    # October.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    @property
    def filename(self) -> str:
        """Alias for the API contract, which calls this `filename`."""
        return self.original_filename

    @property
    def characters(self) -> int:
        """Derived, not stored -- it is len(raw_text) and would go stale."""
        return len(self.raw_text)

    @property
    def extraction_ok(self) -> bool:
        """Whether structured extraction ran successfully for this profile.

        Distinguishes "the LLM found no skills" from "the LLM never ran",
        which an empty skills list alone cannot.
        """
        return self.extracted is not None

    @property
    def skills(self) -> list[dict]:
        """Skills from the extraction blob, for the API response.

        Reads out of JSONB rather than a promoted column: skills are a list
        we display but never filter or join on, so denormalizing them into
        their own table would add a join for no query benefit. That changes
        at M6, when scoring needs set operations over them.
        """
        if not self.extracted:
            return []
        return self.extracted.get("skills", [])

    def __repr__(self) -> str:
        return f"<Profile id={self.id} file={self.original_filename!r}>"
