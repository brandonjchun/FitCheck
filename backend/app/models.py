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
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
    text,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, REAL
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.embeddings import EMBEDDING_DIM
from app.extraction import POSTING_EXTRACTION_VERSION, PROFILE_EXTRACTION_VERSION
from app.skills import normalize_skill_items

# The job lifecycle from spec section 6.3. Stored as text rather than a
# Postgres enum: adding a state to a text column is a no-op, while adding one
# to an enum type requires ALTER TYPE and a migration that cannot run inside a
# transaction on older Postgres. The tradeoff is that the database will not
# reject a typo -- the Literal and the API schema are what enforce it.
JobStatus = Literal["queued", "running", "succeeded", "failed", "dead"]

TERMINAL_STATUSES: frozenset[str] = frozenset({"succeeded", "failed", "dead"})

# Consecutive failed crawls before a source is rested. Spec section 9 item 7.
#
# Five rather than one, because boards have bad minutes: a single 503 is
# noise and tripping on it would disable a healthy board. Five in a row at a
# daily interval is five days of a board being genuinely broken, which is
# past the point where continuing to ask is useful to anyone.
MAX_CONSECUTIVE_FAILURES = 5

# What a discover job produces versus what fetches one page. Carried on the
# job row so the ops dashboard can tell a crawl tick apart from the hundreds
# of fetches it fans out into -- without it, a board being enumerated and a
# board being scraped look identical in the queue.
JobKind = Literal["ingest_posting", "discover"]


class User(Base):
    """One account. Identity for everything a person owns.

    Deliberately minimal: email verification, password reset, and profile
    fields are all out of scope, and adding columns for them now would be
    guessing at flows that do not exist yet.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    # citext at the database level (see the migration), so Bob@x.com and
    # bob@x.com are one account rather than two. Doing this in Python instead
    # -- lowercasing before every insert and lookup -- works right up until
    # one code path forgets, and then a user has a duplicate account they
    # cannot explain. Uniqueness enforced in the database for the same reason.
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    # Argon2id encoded string: algorithm, parameters, salt, and digest in one
    # field. That format is why no separate salt column exists, and why
    # changing the work factor later does not invalidate old hashes -- each
    # row records the parameters it was made with.
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)

    # Operator flag, not a role table. One boolean is the honest shape for a
    # system with exactly two levels of access: everyone, and whoever may read
    # system-wide state. A `roles` table plus a join would be more general and
    # would encode no more information than this does -- add it when a third
    # level actually exists, not in anticipation of one.
    #
    # Granted out of band. There is deliberately no endpoint that sets this:
    # an API that can promote an account is a privilege-escalation target
    # bought for the price of saving one UPDATE. Promote with SQL.
    #
    # NOT NULL with a false default, so a new account is never ambiguously
    # privileged. A nullable flag has to be interpreted somewhere, and the
    # interpretation written by accident is always the permissive one.
    is_admin: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} admin={self.is_admin}>"


class Profile(Base):
    """One uploaded resume and everything derived from it."""

    __tablename__ = "profiles"
    __table_args__ = (
        Index("ix_profiles_user_id", "user_id"),
        # Exactly one active resume per user, enforced by a partial unique
        # index rather than by application code. Two concurrent "make this
        # one active" requests both read "no active profile", both write, and
        # only the database can reject the second. WHERE is_active means the
        # constraint applies solely to active rows, so a user may keep any
        # number of inactive ones.
        Index(
            "profiles_one_active_per_user",
            "user_id",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    # CASCADE: deleting an account removes the resumes it owns. A profile
    # without an owner is unreachable by every query in the app, so leaving
    # one behind is a leak rather than a feature.
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Which resume drives this user's feed. A user may upload several
    # versions; exactly one is current.
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

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

    # Which generation of the prompt and schema produced `extracted`. Null
    # until extraction succeeds, and set in the same commit as the blob so the
    # two can never disagree about which rules built it.
    #
    # A content hash cannot detect a prompt change, so without this a better
    # prompt would leave every stored profile on the old behaviour with no way
    # to find them. See extraction.PROFILE_EXTRACTION_VERSION.
    extraction_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # The resume as a point in meaning-space, for the semantic half of
    # scoring. 384 dimensions because that is what all-MiniLM-L6-v2 emits;
    # the width is baked into the column, so changing model means rewriting
    # this column and rebuilding every index on it.
    #
    # Not derived from `raw_text` directly -- see embeddings.embed_text. The
    # model's context is 256 word pieces and a resume is several times that,
    # so the text is chunked and pooled rather than silently truncated to its
    # first fifth.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

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

        That distinction only holds because a document with no extractable
        text now raises rather than storing an empty result -- see
        workers.extract.EmptyDocumentError. While that path returned a
        populated-but-empty profile, this reported true for a scanned PDF the
        model was never sent, which is precisely the case it exists to rule
        out.
        """
        return self.extracted is not None

    @property
    def extraction_is_current(self) -> bool:
        """Whether `extracted` was produced by the current prompt and schema.

        False for a profile extracted under an older generation. This is the
        row-level form of the `WHERE extraction_version < CURRENT` sweep, and
        what lets a re-extraction request tell "already done" apart from
        "done, but by rules we have since replaced".
        """
        return (
            self.extracted is not None
            and self.extraction_version == PROFILE_EXTRACTION_VERSION
        )

    @property
    def skills(self) -> list[dict]:
        """Skills from the extraction blob, canonicalized for the caller.

        Reads out of JSONB rather than a promoted column: skills are a list
        we display but never filter or join on, so denormalizing them into
        their own table would add a join for no query benefit. That changes
        at M7, when scoring needs set operations over them.

        Normalization happens here rather than before the write, so the
        column keeps the model's original spellings and an addition to the
        alias map applies to every existing profile immediately -- no
        backfill, no re-running the LLM. The cost is a dict lookup per skill
        per read, against a list that is a few dozen entries long.
        """
        if not self.extracted:
            return []
        return normalize_skill_items(self.extracted.get("skills", []))

    def __repr__(self) -> str:
        return f"<Profile id={self.id} file={self.original_filename!r}>"


class UrlBatch(Base):
    """One uploaded list of job-posting URLs.

    Groups the jobs it fanned out into, so a client can poll one endpoint for
    aggregate progress instead of N individual ones.

    Note what this table does *not* have: a `completed_count`. Progress is
    derived with a GROUP BY over the jobs pointing here. A counter column
    would be incremented by N concurrent workers, which is the
    `counter = counter + 1` double-count that at-least-once delivery
    guarantees will eventually happen -- and a summary that disagrees with
    the rows it summarizes is worse than no summary.
    """

    __tablename__ = "url_batches"
    __table_args__ = (Index("ix_url_batches_user_id", "user_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    profile_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
    )

    original_filename: Mapped[str] = mapped_column(Text, nullable=False)

    # What happened to the lines the user sent. Stored rather than derived
    # because the rejected ones never became jobs -- there is no row to count
    # later, and "you sent 4,000 lines and we took 500" is exactly what the
    # user needs to see.
    total_urls: Mapped[int] = mapped_column(Integer, nullable=False)
    rejected_urls: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    duplicate_urls: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<UrlBatch id={self.id} urls={self.total_urls}>"


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


# --- dedupe keys -------------------------------------------------------
#
# What makes two in-flight jobs "the same work". Built by the caller rather
# than derived from the row, because only the caller knows: two people
# submitting one URL are two legitimate jobs, while two crawl ticks finding
# one posting are not. The partial unique index on (kind, dedupe_key) is what
# enforces whatever these say.


def dedupe_key_for_submission(profile_id: int, url_hash: str) -> str:
    """A user submitting a URL for one of their profiles.

    Scoped to the profile, so two candidates submitting the same posting each
    get their own fetch and their own score -- which is correct, because a
    match is per-profile and one job cannot produce two of them.
    """
    return f"profile:{profile_id}:{url_hash}"


def dedupe_key_for_crawl(canonical_key: str) -> str:
    """The crawler finding a posting on a board.

    Scoped globally rather than per-profile, and that is the whole difference
    from the above: a crawled posting enters the shared catalog, so two
    sources listing the same job must collapse to one fetch. Scoring against
    individual profiles happens later, from the catalog, by a different job.
    """
    return f"posting:{canonical_key}"


def dedupe_key_for_discover(source_id: int) -> str:
    """One crawl tick for one board.

    This is the key that makes overlapping ticks safe. A daily schedule that
    fires while yesterday's crawl is still draining would otherwise enumerate
    the board twice and double the fan-out.
    """
    return f"discover:{source_id}"


def canonical_key_for_url(url: str) -> str:
    """Global dedupe key for a user-submitted one-off URL.

    Namespaced `url:` because crawled postings will key as
    `{source_kind}:{board_token}:{external_id}` at M8. Keeping both in one
    column with distinct prefixes is what lets a posting submitted by hand
    and the same posting found by the crawler collapse onto one row.

    Hashed rather than stored raw: URLs can exceed the btree index entry
    limit, and a fixed-width key keeps the unique index small.

    Note the difference from `hash_url`. That one deliberately does not
    normalize, which was right when the key was per-profile -- erring toward
    re-fetching rather than serving the wrong cached posting. This key is
    global, so un-normalized would mean the same posting stored twice under
    two tracking URLs and extracted twice at full LLM cost. The reasoning
    inverts with the scope of the key.
    """
    from app.urls import normalize_url

    normalized = normalize_url(url) or url.strip()
    return f"url:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"


class JobPosting(Base):
    """One job posting: the thing a person applies to.

    A *result* record, as opposed to IngestJob's *work* record (spec section 5.6).
    It exists only on a successful fetch, and it outlives any individual
    attempt -- at M8 one posting is re-crawled dozens of times over its life,
    each crawl a new IngestJob row against this same posting.

    Decoupled from any user on purpose. Two people submitting the same
    Greenhouse posting must land on one row, or the crawler creates a third.
    """

    __tablename__ = "job_postings"
    __table_args__ = (
        # Global, not per-user. This is the constraint that makes the catalog
        # a catalog rather than a pile of per-profile copies.
        UniqueConstraint("canonical_key", name="job_postings_canonical_uniq"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    canonical_key: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)

    # SHA-256 of the normalized text. The M8 content-hash gate compares this
    # to decide whether a re-crawl needs to pay for extraction and embedding
    # again, which is what makes a daily crawl cheap. Stored from M5 so the
    # gate has history to compare against the first time it runs.
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)

    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Populated by extraction at M7. Present now so the fetch has somewhere to
    # write and the M7 diff is scoring rather than a schema change.
    extracted: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    extraction_version: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Promoted out of the `extracted` blob, per spec section 3.3: anything the
    # feed filters, sorts, or indexes on becomes a real column, and the long
    # tail stays in JSONB. A JSONB probe cannot use a plain btree index, so
    # `WHERE extracted->>'seniority' = 'senior'` is a sequential scan where
    # `WHERE seniority = 'senior'` is a lookup.
    #
    # All null until M7's extraction fills them. They exist now because adding
    # a nullable column is a catalog write -- instant at any table size --
    # while *changing* a column's type rewrites every row and rebuilds every
    # index on it. These five have a known shape, so there is nothing to guess
    # and no reason to make M7 ship a migration alongside its scoring code.
    #
    # Which board this came from, or NULL for a user-submitted one-off URL.
    source_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("sources.id", ondelete="SET NULL"), nullable=True
    )

    # The board's own last-changed timestamp, for boards that publish one.
    #
    # The cheap half of re-crawl economics. A content hash can only tell you
    # a posting is unchanged *after* you have fetched it; this tells you
    # before, so a daily crawl of a stable board fetches almost nothing
    # instead of everything. NULL for boards that do not offer it (Lever,
    # Ashby) -- which is fine, because those hand back the full description
    # in the listing and their hash is free.
    source_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    company: Mapped[str | None] = mapped_column(Text, nullable=True)
    location: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Free text rather than an enum, matching how `seniority` is stored on
    # Profile. The vocabulary comes out of an LLM, so pinning it to a Postgres
    # enum now would mean an ALTER TYPE the first time a posting says
    # "flexible" instead of "hybrid".
    remote_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    seniority: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Numeric(4,1) mirrors profiles.years_experience, so a comparison between
    # what a posting demands and what a candidate has needs no cast.
    min_years: Mapped[Decimal | None] = mapped_column(Numeric(4, 1), nullable=True)

    # The posting as a point in the same space as profiles.embedding. It has
    # to be the same model and the same width or the two are not comparable
    # -- vectors from different models are coordinates from different maps,
    # and the arithmetic still runs, which is what makes it dangerous.
    #
    # The HNSW index over this column is M9's, not M7's, and it wants
    # `WHERE closed_at IS NULL` baked into the index definition so closed
    # postings are never retrieved rather than filtered afterwards. Building
    # it now against an empty table would tune it on nothing.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Updated on every successful re-fetch, including one that skips
    # extraction because the hash matched. This is the heartbeat that closure
    # detection reads at M8.
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Tombstone. Set when a posting is absent from a *complete* enumeration of
    # its source (M8), never by deleting the row -- `matches` will reference
    # postings, so a delete either cascades away a user's history or raises a
    # foreign key error mid-crawl. A tombstone also gives the UI an honest
    # state ("this role appears to have been filled") rather than the row
    # silently vanishing.
    #
    # Present now because it is the predicate in M9's partial vector index
    # (`WHERE closed_at IS NULL`), which is what keeps closed postings out of
    # ANN retrieval entirely rather than filtering them after the fact.
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    @property
    def is_open(self) -> bool:
        """Whether this posting is still live.

        A property rather than a stored boolean, so it cannot drift from
        `closed_at`. Two columns encoding one fact is two things to keep in
        sync and one of them will eventually be wrong.
        """
        return self.closed_at is None

    @property
    def skills(self) -> list[dict]:
        """Requirements with canonical skill names, normalized on read.

        The posting-side mirror of Profile.skills, and normalized the same
        way for one reason above all: the scorer compares these two lists by
        name. If a resume says "JS" and a posting says "JavaScript", the
        overlap is only visible when both have passed through the same alias
        map -- so normalizing one side and not the other would silently
        report every such pair as a missing requirement.

        `necessity`, `min_years`, and `evidence` carry through untouched.
        """
        if not self.extracted:
            return []
        return normalize_skill_items(self.extracted.get("skills", []))

    @property
    def extraction_ok(self) -> bool:
        return self.extracted is not None

    @property
    def extraction_is_current(self) -> bool:
        """Whether `extracted` was produced by the current posting prompt.

        Keyed on the version rather than on presence, matching the resume
        path: a posting extracted under an older prompt *has* an extraction
        and still needs a new one, so a presence check would make a version
        bump unactionable.
        """
        return (
            self.extracted is not None
            and self.extraction_version == POSTING_EXTRACTION_VERSION
        )

    def __repr__(self) -> str:
        return f"<JobPosting id={self.id} url={self.url!r}>"


class IngestJob(Base):
    """One submitted job-posting URL: the unit of asynchronous work.

    Named `ingest_jobs` rather than `jobs` because a job *catalog* now exists
    (spec section 5.1). With both tables present, "job" means two different
    things and every sentence about the system needs a disambiguating clause
    -- which is exactly the situation the rename was written to prevent, and
    the reason it happens now rather than after M8 doubles the call sites.

    Separate from JobPosting on purpose (spec section 5.2). This is a *work
    record* -- it exists the instant a URL is submitted and survives every
    attempt failing. JobPosting is a *result record* and exists only on
    success. Collapsing them would mean one table with half its columns null
    most of the time, and would make "show me everything that failed" a query
    against a table that is conceptually about postings.

    This separation is also why the ops dashboard in M10 is cheap to build:
    this table IS the audit log.
    """

    __tablename__ = "ingest_jobs"
    __table_args__ = (
        # Enforced in the database, not the application. An application-level
        # "does this already exist?" check races: two concurrent submissions
        # both read "no", both insert, and you have scraped the same page
        # twice. The database is the only place this can be decided.
        #
        # **Partial, on in-flight work only** (spec section 5.4). The earlier
        # form was a plain UNIQUE (profile_id, url_hash), which was right
        # while every job came from a person and wrong the moment a crawler
        # exists: a daily re-crawl of the same board submits the same URLs
        # every day, and a total constraint would reject every tick after the
        # first. Restricting it to `queued` and `running` means two crawl
        # ticks overlapping still collapse to one job, while yesterday's
        # completed row stays as the audit log without blocking today.
        #
        # Keyed on (kind, dedupe_key) rather than on profile_id, because a
        # crawler job has no profile. Postgres treats NULLs as distinct in a
        # unique index, so a nullable profile_id in the old constraint would
        # have silently stopped constraining anything at all -- the failure
        # mode being duplicate outbound fetches, with the index still present
        # and looking like it worked.
        Index(
            "ingest_jobs_inflight_uniq",
            "kind",
            "dedupe_key",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
        ),
        # Postgres does NOT index foreign keys automatically (unlike primary
        # keys). Without this, "all jobs for this profile" is a seq scan.
        Index("ix_ingest_jobs_profile_id", "profile_id"),
        Index("ix_ingest_jobs_source_id", "source_id"),
        # The ops dashboard's hot path: count/filter by state.
        Index("ix_ingest_jobs_status", "status"),
        Index("ix_ingest_jobs_created_at", "created_at"),
        # Batch progress groups on this, on every poll while a batch runs.
        Index("ix_ingest_jobs_batch_id", "batch_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    # What this job does. `ingest_posting` fetches one page; `discover`
    # enumerates a whole board and fans out into hundreds of the former.
    #
    # Needed on the row rather than inferred, because the two are
    # indistinguishable from outside: both are ingest_jobs, both have a URL,
    # and an ops dashboard showing 400 rows cannot otherwise tell one crawl
    # tick from the 399 fetches it produced.
    kind: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="ingest_posting"
    )

    # What makes this job a duplicate of another in-flight one. Built by the
    # caller, because only the caller knows what "the same work" means:
    # `profile:{id}:{url_hash}` for a user submission, since two people
    # submitting one URL are two legitimate jobs; `posting:{canonical_key}`
    # for a crawled page, since two boards listing one posting are not;
    # `discover:{source_id}` for a crawl tick.
    # NOT NULL with no default on purpose. An empty-string default would let
    # a caller that forgets this insert successfully, and then collide with
    # every *other* caller that forgot -- surfacing as a unique violation on a
    # key nobody set, in whichever unrelated code path ran second. Requiring
    # it moves the failure to the line that omitted it.
    dedupe_key: Mapped[str] = mapped_column(Text, nullable=False)

    # NULL for crawler-originated work, which belongs to no candidate.
    #
    # This is the column M8 forced open. It was NOT NULL while every job came
    # from somebody clicking submit, and Path B's whole premise is a catalog
    # built ahead of anyone asking for it -- a discovered posting is scored
    # against profiles later, by a different job, rather than being submitted
    # on behalf of one.
    profile_id: Mapped[int | None] = mapped_column(
        BigInteger,
        # CASCADE: a job is meaningless without the candidate it was submitted
        # for, so deleting a profile should not leave orphaned work records.
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=True,
    )

    # Which board produced this job, or NULL for a user submission. SET NULL
    # rather than CASCADE: removing a source should stop future crawls, not
    # erase the record of what was already fetched from it.
    source_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("sources.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Which upload produced this job, or NULL for a single URL submission.
    # SET NULL rather than CASCADE on delete: the job is a real work record
    # with its own history, and discarding a batch should orphan the grouping
    # rather than destroy the audit trail of what was fetched.
    batch_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("url_batches.id", ondelete="SET NULL"),
        nullable=True,
    )

    url: Mapped[str] = mapped_column(Text, nullable=False)
    url_hash: Mapped[str] = mapped_column(Text, nullable=False)

    # The posting this work produced, set on success. Null while queued, and
    # null forever for a job that never succeeded -- which is precisely the
    # work-record/result-record split: the attempt is recorded either way,
    # the posting only exists if there was something to record.
    job_posting_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("job_postings.id", ondelete="SET NULL"),
        nullable=True,
    )

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
        return f"<IngestJob id={self.id} status={self.status!r} url={self.url!r}>"


# Where a match came from. Both paths write the same table with the same
# scorer -- the difference is only who asked. A user submitting a URL gets
# `user_submission`; M9's scheduled scoring gets `recommendation`.
#
# Worth keeping distinct even though nothing branches on it yet: "postings
# you asked about" and "postings we suggested" are different products from
# the user's side, and a feed that silently mixes them reads as the system
# recommending something the user already found themselves.
MatchOrigin = Literal["user_submission", "recommendation"]


class Match(Base):
    """One scored (profile, posting) pair, with the reasoning kept.

    The row every path converges on. Path A writes one when a user submits a
    URL; M9's recommender writes the top 50 per profile from the same scorer
    over the same columns.

    `breakdown` is not decoration. A ranked feed whose only output is a
    number cannot be argued with, and spec section 8.4 requires both
    sub-scores and the skill accounting to be surfaced -- so the thing that
    justifies the rank is stored beside it rather than recomputed on read,
    which would silently change as the alias map grows.
    """

    __tablename__ = "matches"
    __table_args__ = (
        # One score per pair. Re-scoring updates in place rather than
        # appending, so the feed cannot show the same posting twice at two
        # different ranks -- which is what an append-only history would do
        # the first time anything re-scored.
        UniqueConstraint(
            "profile_id", "job_posting_id", name="matches_profile_posting_uniq"
        ),
        # The feed query, exactly: this profile's matches, best first. A
        # composite index in this order serves both the filter and the sort,
        # so the planner never sorts the result set.
        Index("matches_feed_idx", "profile_id", text("final_score DESC")),
        Index("ix_matches_job_posting_id", "job_posting_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    profile_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False
    )
    # CASCADE here too, but note it should almost never fire: a closed posting
    # is tombstoned with `closed_at`, never deleted, precisely so a user's
    # match history survives the role being filled.
    job_posting_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False
    )

    # `real` per the spec rather than double precision. These are similarity
    # scores in [0, 1] from a model whose own output is float32 -- storing
    # them at double width would be four bytes of false precision per column
    # on the table that grows fastest (profiles x postings).
    semantic_score: Mapped[float] = mapped_column(REAL, nullable=False)
    skill_score: Mapped[float] = mapped_column(REAL, nullable=False)
    final_score: Mapped[float] = mapped_column(REAL, nullable=False)

    breakdown: Mapped[dict] = mapped_column(JSONB, nullable=False)

    origin: Mapped[str] = mapped_column(Text, nullable=False)

    # Which rules produced these numbers. A feed sorted across two generations
    # is silently wrong -- position 3 beats position 4 because it was scored
    # under different weights, not because it fits better. See
    # scoring.SCORER_VERSION.
    scorer_version: Mapped[int] = mapped_column(Integer, nullable=False)

    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<Match profile={self.profile_id} posting={self.job_posting_id} "
            f"score={self.final_score:.3f}>"
        )


# What kind of board a source is, which decides which adapter enumerates it.
#
# Text rather than an enum for the same reason `status` is: adding a kind is
# a no-op on a text column and an ALTER TYPE on an enum, and the set of job
# boards worth crawling is exactly the sort of thing that grows.
SourceKind = Literal["greenhouse", "lever", "ashby", "careers_page"]


class Source(Base):
    """One board we crawl on a schedule.

    Path B's input. A source names a company's job board and carries the
    state a crawler needs to be a good citizen about it: whether it is
    enabled, how often to look, when it was last *attempted*, and when it
    last *succeeded*.

    Those last two are deliberately separate columns rather than one, and the
    distinction is load-bearing rather than bookkeeping -- see
    `last_success_at`.
    """

    __tablename__ = "sources"
    __table_args__ = (
        # One row per board. Re-seeding is then an upsert rather than a
        # careful check, and a duplicated source would double every crawl
        # against somebody else's server.
        UniqueConstraint("kind", "board_token", name="sources_kind_token_uniq"),
        # The scheduler's query: which sources are due. Partial on `enabled`
        # so a disabled board is not merely filtered out but never visited by
        # the index at all.
        Index(
            "ix_sources_due",
            "last_crawled_at",
            postgresql_where=text("enabled"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    kind: Mapped[str] = mapped_column(Text, nullable=False)

    # The company slug in the board URL -- `anthropic` in
    # boards-api.greenhouse.io/v1/boards/anthropic/jobs. Stored rather than a
    # full URL because the adapter owns the URL shape: a board that moves
    # endpoints is then one change in one adapter instead of a data migration
    # across every row.
    board_token: Mapped[str] = mapped_column(Text, nullable=False)

    display_name: Mapped[str] = mapped_column(Text, nullable=False)

    # Kill switch, no deploy required. Spec section 9 item 7 -- one board that
    # starts 403ing should not consume the retry budget for a week, and the
    # fix has to be available at 2am without a release.
    enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )

    crawl_interval_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="86400"
    )

    # When a crawl was last *attempted*.
    last_crawled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # When a crawl last *completed a full enumeration without error*.
    #
    # Separate from `last_crawled_at`, and the separation is the whole reason
    # closure detection is safe. "This posting is gone, so it must be closed"
    # is only sound if the list we compared against was complete. If
    # enumeration returned three pages and then timed out, the postings on
    # page four are absent from our view and present in reality -- and running
    # the closure UPDATE would tombstone an entire board out of every user's
    # feed, silently, with no error anywhere.
    #
    # Two columns make "attempted but did not finish" representable. One
    # column cannot express it, which is how that bug gets written.
    last_success_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Circuit breaker state. Reset to zero on any success, so this counts
    # *consecutive* failures rather than lifetime ones -- a board that fails
    # once a week forever is healthy, and a board that has failed five times
    # in a row is not.
    consecutive_failures: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    @property
    def circuit_open(self) -> bool:
        """Whether this source is being rested after repeated failures.

        A property rather than a stored boolean so it cannot drift from the
        counter it is derived from -- the same reasoning as `is_open` on
        JobPosting. Two columns encoding one fact is one of them eventually
        being wrong.
        """
        return self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    def __repr__(self) -> str:
        return f"<Source {self.kind}:{self.board_token} enabled={self.enabled}>"
