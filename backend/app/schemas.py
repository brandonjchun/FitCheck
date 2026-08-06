"""Pydantic models -- the API contract.

Deliberately separate from models.py (SQLAlchemy ORM models, the storage
shape) and extraction.py (the LLM output contract). All three describe a
profile; all three change for different reasons.
"""

from datetime import datetime
from typing import Literal

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

    # Which resume currently drives this user's feed. Exposed so a client that
    # just uploaded knows whether it changed anything: upload deliberately does
    # not promote, so a second upload comes back `is_active: false` and the UI
    # can say so rather than implying the new file took over.
    is_active: bool = False

    raw_text: str


class ProfileSummary(BaseModel):
    """One resume in the list of a user's uploads.

    Separate from ProfileUploadResponse because of what it leaves out:
    `raw_text` and `skills`. A version picker renders a filename, a date, and
    a badge -- shipping the full text of every resume a user has ever uploaded
    to draw that list is the kind of payload that is fine with three rows and
    absurd with thirty.

    `characters` survives the cut because it is the one size signal worth
    showing, and it is derived from raw_text rather than stored -- see
    models.Profile.characters.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    characters: int
    created_at: datetime

    is_active: bool
    extraction_ok: bool
    seniority: Seniority | None = None
    years_experience: float | None = None

    # A count rather than the list. The picker shows "18 skills" as a quality
    # signal; the skills themselves belong to whichever profile is open.
    skill_count: int


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


class SourceFreshness(BaseModel):
    """How current one board's slice of the catalog is.

    `last_success_at` is the honest freshness figure and `last_crawled_at` is
    not: a source that has been attempted every hour and succeeded once last
    week looks perfectly healthy on the second column alone. Both are exposed
    so the gap between them is visible, because that gap *is* the failure --
    a crawler that runs on schedule and returns nothing produces a catalog
    that ages while every dashboard says it is being tended.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    kind: str
    board_token: str
    display_name: str
    enabled: bool
    crawl_interval_seconds: int

    last_crawled_at: datetime | None = None
    last_success_at: datetime | None = None
    consecutive_failures: int
    circuit_open: bool

    # Derived rather than stored: seconds since the last *successful* crawl,
    # and whether that exceeds this source's own interval. Computed server-side
    # so every client agrees on what "stale" means instead of each reimplementing
    # the comparison against its own clock.
    seconds_since_success: float | None = None
    is_stale: bool

    open_postings: int


class GateStats(BaseModel):
    """Content-hash gate effectiveness.

    The measurement spec section 6.7 asks to be *reported* rather than
    asserted. Counters are process-lifetime and unpartitioned, so this mixes a
    board's first crawl -- where every posting is necessarily a miss -- with
    the steady-state re-crawls the number is actually about. Read it as a
    floor on the gate's value, not as the steady-state rate.
    """

    hits: int
    misses: int
    total: int
    hit_rate: float


class OpsOverview(BaseModel):
    """Everything the dashboard polls, in one response."""

    queues: list[QueueHealth]
    workers: list[WorkerInfo]
    jobs_by_status: list[StatusCount]
    job_timeout_seconds: int
    result_ttl_seconds: int
    failure_ttl_seconds: int

    sources: list[SourceFreshness]
    gate: GateStats


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


class MatchSkill(BaseModel):
    """One posting requirement, judged against the candidate.

    The unit the UI renders as a green / amber / red row. `bucket` carries
    the verdict and `evidence` carries the posting's own words for it, so a
    user can see what the score was reacting to rather than being told a
    number.
    """

    name: str
    necessity: Literal["required", "preferred", "unknown"]
    bucket: Literal["matched", "partial", "missing"]
    required_years: float | None = None
    candidate_years: float | None = None
    evidence: str | None = None


class MatchCounts(BaseModel):
    matched: int
    partial: int
    missing: int
    # Broken out because it is the only count that decides anything. A
    # candidate missing four preferred skills is a fine applicant; one
    # missing a single required skill usually is not, and a UI that shows
    # only a total cannot tell those apart.
    missing_required: int


class MatchSeniority(BaseModel):
    """The level gap between the candidate and the role.

    The counterpart to `MatchCounts` for the other half of "why is this here".
    Skills answer whether the candidate can do the work; this answers whether
    the role is pitched at them.

    `direction` is what a UI should branch on, and it is carried rather than
    derived because the sign of `steps` cannot express all four cases: 0 means
    a match and None means nobody could tell, and a client testing `steps < 0`
    reads both as "not under". The levels are carried alongside so the copy can
    name which side was missing.
    """

    profile_level: str | None = None
    posting_level: str | None = None
    # Signed from the candidate's side: negative means the role sits above
    # them. Same orientation as `years_gap`, deliberately.
    steps: int | None = None
    direction: Literal["match", "under", "over", "unknown"]
    candidate_years: float | None = None
    required_years: float | None = None
    years_gap: float | None = None


class MatchResponse(BaseModel):
    """One scored (profile, posting) pair, with the reasoning attached.

    Both sub-scores travel with the final one, and the skill breakdown
    travels with both. Spec section 8.4 requires it, and the reason is that
    a blended number alone is unfalsifiable -- 0.35 says nothing a user can
    act on, while "you match 2 of 5 requirements and are short on React
    years" does.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    profile_id: int
    job_posting_id: int

    semantic_score: float
    skill_score: float
    final_score: float

    origin: str
    # Surfaced deliberately. A client comparing two matches has no other way
    # to know they were produced by different rules and are not comparable.
    scorer_version: int
    scored_at: datetime

    # Flattened out of the stored breakdown blob rather than exposing it raw,
    # so the JSONB shape stays an internal detail that can change without
    # breaking the contract.
    counts: MatchCounts
    skills: list[MatchSkill]
    weights: dict[str, float]
    extraction_failed: bool

    # None for a match scored before the delta existed. Optional rather than
    # defaulted to an empty gap, because a client has to be able to tell "this
    # match predates the field" from "we compared and could not tell" -- the
    # latter is a `MatchSeniority` whose direction is "unknown", and rendering
    # that copy over an old row would be asserting something never computed.
    seniority: MatchSeniority | None = None

    # Denormalized from the posting for display. A feed showing 50 matches
    # would otherwise need 50 extra requests, or a join the client has to
    # know to ask for.
    posting_url: str | None = None
    posting_title: str | None = None
    posting_company: str | None = None


class RecommendationRun(BaseModel):
    """The outcome of asking for a feed to be built.

    `status` rather than a bare boolean because the three cases call for three
    different things from the UI: `queued` means show a building state and keep
    polling, `already_current` means the feed on screen is the answer, and
    `profile_not_ready` means wait for extraction rather than for scoring. A
    boolean would collapse the last two into "nothing happened", and the user
    would be told to wait for a job that is never going to run.
    """

    profile_id: int
    status: Literal["queued", "already_current", "profile_not_ready"]
    queued: bool


class SkillGap(BaseModel):
    """How one requirement fared across a user's whole feed.

    All four buckets travel together rather than just the bad news, because
    the ratio is the point: "missing in 3 of 40" is noise and "missing in 30
    of 40" is the next thing to learn, and the raw count alone cannot tell
    those apart.
    """

    name: str
    missing: int
    partial: int
    matched: int
    # Missing *and* listed as required -- the subset that actually
    # disqualifies, which is what the ranking is by.
    blocking: int


class SkillGapReport(BaseModel):
    profile_id: int | None = None
    matches_analyzed: int
    gaps: list[SkillGap]


class SavedMatch(BaseModel):
    """A match the user reacted to, with the reaction attached."""

    model_config = ConfigDict(from_attributes=True)

    match_id: int
    profile_id: int
    job_posting_id: int
    final_score: float
    verdict: str
    verdict_at: datetime

    posting_url: str | None = None
    posting_title: str | None = None
    posting_company: str | None = None
    posting_closed: bool = False


class FeedbackCreate(BaseModel):
    """A user's judgment on one recommendation.

    The verdict is a Literal rather than a plain string so an unknown value is
    a 422 naming the alternatives, produced before the request reaches a
    handler. The database column is deliberately unconstrained text -- see
    `MatchFeedback` -- which puts the vocabulary in exactly one place that can
    explain itself to a caller.
    """

    verdict: Literal["interested", "not_interested", "applied"]


class FeedbackResponse(BaseModel):
    """A recorded label.

    Returns the row rather than an empty 201 so the client has the id and the
    server's `created_at`, which is what orders the funnel. A client that
    stamped its own time would be recording clock skew as sequence.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    match_id: int
    verdict: str
    created_at: datetime
