"""SQLAlchemy ORM models -- the storage shape.

Deliberately separate from schemas.py (Pydantic, the API contract). A column
rename here should not be a breaking API change, and internal columns should
not leak to clients by default.
"""

import hashlib
from datetime import datetime
from decimal import Decimal
from typing import Literal

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

# The job lifecycle from spec section 6.3. Stored as text rather than a
# Postgres enum: adding a state to a text column is a no-op, while adding one
# to an enum type requires ALTER TYPE and a migration that cannot run inside a
# transaction on older Postgres. The tradeoff is that the database will not
# reject a typo -- the Literal and the API schema are what enforce it.
JobStatus = Literal["queued", "running", "succeeded", "failed", "dead"]

TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "dead"})


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


def hash_url(url: str) -> str:
    """Stable dedupe key for a URL.

    Hashed rather than storing the raw URL in the unique index because URLs
    can exceed the ~2704-byte limit of a btree index entry, and a fixed-width
    key keeps the index small. SHA-256 rather than MD5 only because there is
    no reason to pick the weaker one; collision resistance is not really the
    property being relied on here.

    Deliberately NOT normalized (no lowercasing, no query-param sorting, no
    trailing-slash stripping). Two URLs differing only in a tracking param
    will be treated as different jobs. Normalizing correctly is genuinely
    hard -- ?page=2 matters, ?utm_source=x does not, and no generic rule
    tells them apart -- so this errs toward re-fetching rather than toward
    silently returning the wrong cached posting.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class Job(Base):
    """One submitted job-posting URL: the unit of asynchronous work.

    Separate from JobPosting on purpose (spec section 5.2). This is a *work
    record* -- it exists the instant a URL is submitted and survives every
    attempt failing. JobPosting is a *result record* and exists only on
    success. Collapsing them would mean one table with half its columns null
    most of the time, and would make "show me everything that failed" a query
    against a table that is conceptually about postings.

    This separation is also why the ops dashboard in M8 is cheap to build:
    this table IS the audit log.
    """

    __tablename__ = "jobs"
    __table_args__ = (
        # Enforced in the database, not the application. An application-level
        # "does this already exist?" check races: two concurrent submissions
        # both read "no", both insert, and you have scraped the same page
        # twice. The database is the only place this can be decided.
        UniqueConstraint("profile_id", "url_hash", name="jobs_profile_url_uniq"),
        # Postgres does NOT index foreign keys automatically (unlike primary
        # keys). Without this, "all jobs for this profile" is a seq scan.
        Index("ix_jobs_profile_id", "profile_id"),
        # The ops dashboard's hot path: count/filter by state.
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    profile_id: Mapped[int] = mapped_column(
        BigInteger,
        # CASCADE: a job is meaningless without the candidate it was submitted
        # for, so deleting a profile should not leave orphaned work records.
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
    )

    url: Mapped[str] = mapped_column(Text, nullable=False)
    url_hash: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # Truncated exception detail. Text rather than JSONB because nothing
    # queries into it -- it is read by a human staring at a failed job.
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Correlates this row with RQ's own job record in Redis. Without it there
    # is no way to go from "this database row is stuck" to "here is what the
    # queue thinks is happening".
    rq_job_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    @property
    def is_terminal(self) -> bool:
        """Whether this job has reached a state it will never leave.

        The frontend polls until this is true (spec section 7.2).
        """
        return self.status in TERMINAL_STATUSES

    def __repr__(self) -> str:
        return f"<Job id={self.id} status={self.status!r} url={self.url!r}>"
