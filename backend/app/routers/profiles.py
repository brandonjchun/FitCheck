"""Routes for resume upload and candidate profiles."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.documents import DocumentError, extract_text
from app.models import Profile, User
from app.queues import FAILURE_TTL, JOB_TIMEOUT, QUEUE_SCORING, RESULT_TTL, get_queue
from app.schemas import ProfileSummary, ProfileUploadResponse
from app.security import current_user, owned_profile

logger = logging.getLogger(__name__)

# A router is a mountable group of routes. main.py calls
# app.include_router(profiles.router), which is what makes these paths live.
router = APIRouter(prefix="/api/profiles", tags=["profiles"])


def _summarize(profile: Profile) -> ProfileSummary:
    """Build the list-view shape for one profile.

    Constructed field by field rather than with from_attributes because
    `skill_count` has no matching attribute on the model -- it is the length
    of a property that decodes JSONB. Counting here keeps that derivation out
    of the ORM, where a column-shaped name that is really a computed list
    would invite someone to filter on it.
    """
    return ProfileSummary(
        id=profile.id,
        filename=profile.filename,
        characters=profile.characters,
        created_at=profile.created_at,
        is_active=profile.is_active,
        extraction_ok=profile.extraction_ok,
        seniority=profile.seniority,
        years_experience=profile.years_experience,
        skill_count=len(profile.skills),
    )


@router.post("", response_model=ProfileUploadResponse, status_code=201)
def upload_resume(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Profile:
    """Accept a resume upload, store its text, and queue structured extraction.

    Text extraction is synchronous -- it is local CPU work measured in
    milliseconds. LLM extraction is not: it measured 26s on Gemini and 57s on
    a local model, and it used to run right here, holding the request open
    the entire time. It is now enqueued instead.

    So this returns 201, not 202. A profile really is created and immediately
    readable at GET /api/profiles/{id}; only its structured fields arrive
    later. `extraction_ok` is false until the worker finishes, and the client
    polls that endpoint the same way it polls a job.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = file.file.read()
    file_name = file.filename

    try:
        raw_text = extract_text(content, file_name)
    except DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # A file can parse cleanly and still yield nothing: a PDF exported by a
    # phone scanner app is images all the way down, and pdfplumber returns ""
    # for it without erroring. 422 rather than 400 -- the request was
    # well-formed and the file was a real PDF, the content is just unusable.
    #
    # Rejecting here rather than storing the row is what keeps `extraction_ok`
    # meaningful downstream. The alternative was a profile with zero skills,
    # no error, and no way for the client to tell that apart from a resume the
    # model genuinely found nothing in.
    if not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be read from this document. If it is a scanned "
                "or photographed resume, it has no text layer -- export a "
                "text-based PDF or upload a DOCX instead."
            ),
        )

    # First resume becomes the active one; later uploads land inactive and an
    # explicit switch promotes them. Auto-promoting the newest upload would
    # mean uploading a draft silently swaps out whatever is driving the feed,
    # which is a surprising thing for an upload to do.
    has_active = db.scalar(
        select(Profile.id).where(Profile.user_id == user.id, Profile.is_active)
    )

    profile = Profile(
        user_id=user.id,
        original_filename=file_name,
        raw_text=raw_text,
        is_active=has_active is None,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)

    # Enqueue after commit, for the same reason as job submission: a worker
    # must never look up a row that has not been written yet.
    #
    # `scoring`, not `interactive`: extraction is CPU-and-API bound and takes
    # tens of seconds, so putting it on the interactive queue would let one
    # upload delay a URL submission that takes milliseconds to accept. It sits
    # above `ingest` in the worker's drain order because a stale profile is
    # annoying and a hung spinner is worse.
    try:
        get_queue(QUEUE_SCORING).enqueue(
            "app.workers.tasks.extract_profile_task",
            profile.id,
            job_timeout=JOB_TIMEOUT,
            result_ttl=RESULT_TTL,
            failure_ttl=FAILURE_TTL,
        )
    except Exception as exc:
        # Redis down must not cost the user their upload. The row persists
        # with extracted=NULL, which is the same state a failed extraction
        # leaves behind and which a re-extraction sweep can find.
        logger.error("upload_resume: enqueue failed for profile %s: %s", profile.id, exc)

    return profile


@router.get("", response_model=list[ProfileSummary])
def list_profiles(
    response: Response,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> list[ProfileSummary]:
    """Every resume this user has uploaded, newest first.

    The endpoint that makes an account durable rather than a single session.
    Without it a profile was reachable only through an id the client happened
    to still be holding, so a page refresh stranded every upload -- the rows
    were never lost, there was simply no way left to ask for them.

    Ordered by created_at with the id as a tiebreak. Two uploads can share a
    timestamp, and an unstable sort makes a list that reshuffles itself
    between renders for no reason the user can see.
    """
    response.headers["Cache-Control"] = "no-store"

    profiles = db.scalars(
        select(Profile)
        .where(Profile.user_id == user.id)
        .order_by(Profile.created_at.desc(), Profile.id.desc())
    ).all()

    return [_summarize(profile) for profile in profiles]


@router.get("/{profile_id}", response_model=ProfileUploadResponse)
def get_profile(
    response: Response, profile: Profile = Depends(owned_profile)
) -> Profile:
    """Fetch one profile. Polled until `extraction_ok` turns true.

    Takes `owned_profile` rather than an id plus a lookup, so the ownership
    predicate is structural. There is no version of this handler that
    accidentally serves someone else's resume, because it never had the
    chance to load one.
    """
    response.headers["Cache-Control"] = "no-store"
    return profile


@router.post("/{profile_id}/extract", response_model=ProfileUploadResponse, status_code=202)
def reextract_profile(profile: Profile = Depends(owned_profile)) -> Profile:
    """Re-run extraction for a profile whose earlier attempt failed.

    Closes the gap where a provider outage left a profile permanently
    un-scoreable with no way to retry it. Safe to call on an
    already-extracted profile: the task checks and returns early rather than
    spending a second API call.

    Ownership matters more here than on a read. This endpoint spends an LLM
    call, so without the predicate anyone could run up the bill on someone
    else's profile ids.
    """
    profile_id = profile.id

    try:
        get_queue(QUEUE_SCORING).enqueue(
            "app.workers.tasks.extract_profile_task",
            profile.id,
            job_timeout=JOB_TIMEOUT,
            result_ttl=RESULT_TTL,
            failure_ttl=FAILURE_TTL,
        )
    except Exception as exc:
        logger.error("reextract_profile: enqueue failed for %s: %s", profile_id, exc)
        raise HTTPException(
            status_code=503, detail="Queue unavailable; try again shortly"
        ) from exc

    return profile


@router.post("/{profile_id}/activate", response_model=ProfileUploadResponse)
def activate_profile(
    profile: Profile = Depends(owned_profile),
    db: Session = Depends(get_db),
) -> Profile:
    """Make this resume the active one.

    The counterpart to upload's deliberate refusal to auto-promote. Uploading
    a draft must not silently swap out whatever drives the feed, so promotion
    is an explicit act -- and this is where it happens.

    Two statements, one transaction, and the order is not negotiable. The
    partial unique index `profiles_one_active_per_user` permits exactly one
    active row per user and is enforced as each statement runs, so promoting
    before demoting collides with the row being replaced. Clearing first
    leaves the account momentarily with no active profile; the index allows
    that, and no other transaction can observe it.
    """
    if profile.is_active:
        # Idempotent. A double-click is not an error and there is nothing to
        # write -- returning 409 here would make the UI apologise for a
        # no-op.
        return profile

    db.execute(
        update(Profile)
        .where(Profile.user_id == profile.user_id, Profile.is_active)
        .values(is_active=False)
    )
    profile.is_active = True

    try:
        db.commit()
    except IntegrityError:
        # Lost a race with a concurrent activation for the same account: both
        # transactions cleared the flag, both set it. The index decides, the
        # same way it decides duplicate registration in auth.register. A
        # retry succeeds because by then the winner is visible.
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Another activation for this account is in flight; try again",
        ) from None

    db.refresh(profile)
    return profile


@router.delete("/{profile_id}", status_code=204)
def delete_profile(
    profile: Profile = Depends(owned_profile),
    db: Session = Depends(get_db),
) -> None:
    """Delete a resume and everything derived from it.

    The cascade is declared by the foreign keys rather than performed here:
    `ingest_jobs.profile_id` and `url_batches.profile_id` are both ON DELETE
    CASCADE, so the work submitted against this resume goes with it. That is
    the honest shape -- a job scored against a resume that no longer exists
    can be neither re-scored nor explained, and keeping it would leave the
    ops dashboard counting work whose subject is gone.

    Deleting the *active* resume promotes the newest survivor instead of
    leaving the account with none. An account holding three resumes and
    driving its feed from zero of them is a state with no honest UI, and its
    only exit would be an activate call the client has to know to make.

    204 with no body. There is no meaningful representation of a row that was
    just removed, and returning the deleted object invites a client to render
    it.
    """
    was_active = profile.is_active
    user_id = profile.user_id

    db.delete(profile)
    # Flushed, not committed. The DELETE has to reach the database before the
    # promotion below, or the unique index sees two active rows -- the one
    # being deleted and its replacement -- and rejects the second.
    db.flush()

    if was_active:
        replacement = db.scalar(
            select(Profile)
            .where(Profile.user_id == user_id)
            .order_by(Profile.created_at.desc(), Profile.id.desc())
            .limit(1)
        )
        if replacement is not None:
            replacement.is_active = True

    db.commit()
