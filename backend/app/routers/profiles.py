"""Routes for resume upload and candidate profiles."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.documents import DocumentError, extract_text
from app.models import Profile
from app.queue import FAILURE_TTL, JOB_TIMEOUT, RESULT_TTL, get_queue
from app.schemas import ProfileUploadResponse

logger = logging.getLogger(__name__)

# A router is a mountable group of routes. main.py calls
# app.include_router(profiles.router), which is what makes these paths live.
router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.post("", response_model=ProfileUploadResponse, status_code=201)
def upload_resume(
    file: UploadFile = File(...),
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

    profile = Profile(original_filename=file_name, raw_text=raw_text)
    db.add(profile)
    db.commit()
    db.refresh(profile)

    # Enqueue after commit, for the same reason as job submission: a worker
    # must never look up a row that has not been written yet.
    try:
        get_queue().enqueue(
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


@router.get("/{profile_id}", response_model=ProfileUploadResponse)
def get_profile(
    profile_id: int, response: Response, db: Session = Depends(get_db)
) -> Profile:
    """Fetch one profile. Polled until `extraction_ok` turns true."""
    profile = db.get(Profile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

    response.headers["Cache-Control"] = "no-store"
    return profile


@router.post("/{profile_id}/extract", response_model=ProfileUploadResponse, status_code=202)
def reextract_profile(profile_id: int, db: Session = Depends(get_db)) -> Profile:
    """Re-run extraction for a profile whose earlier attempt failed.

    Closes the gap where a provider outage left a profile permanently
    un-scoreable with no way to retry it. Safe to call on an
    already-extracted profile: the task checks and returns early rather than
    spending a second API call.
    """
    profile = db.get(Profile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

    try:
        get_queue().enqueue(
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
