"""Routes for resume upload and candidate profiles."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.documents import DocumentError, extract_text
from app.extraction import ExtractedProfile
from app.models import Profile
from app.providers import LLMError
from app.schemas import ProfileUploadResponse
from app.workers.extract import extract_profile

logger = logging.getLogger(__name__)

# A router is a mountable group of routes. main.py calls
# app.include_router(profiles.router), which is what makes these paths live.
# The prefix and tags apply to every route defined below; tags become the
# section headings in the Swagger UI at /docs.
router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.post("", response_model=ProfileUploadResponse, status_code=201)
def upload_resume(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Profile:
    """Accept a resume upload, extract its text and structure, and store it.

    Returns 201 with the new row's id. Structured extraction is best-effort:
    if the LLM provider fails, the profile is still persisted with its raw
    text and `extraction_ok: false`, and can be re-extracted later. Losing a
    successfully parsed document because a third party was unavailable would
    be the wrong trade.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = file.file.read()
    file_name = file.filename

    try:
        raw_text = extract_text(content, file_name)
    except DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    extracted: ExtractedProfile | None = None
    try:
        extracted = extract_profile(raw_text)
    except LLMError as e:
        # Both transient and permanent land here. Neither should cost the
        # user their upload -- the text parsed fine, and that is the
        # expensive, irreplaceable part.
        logger.warning("extraction failed for %s: %s", file_name, e)

    profile = Profile(
        original_filename=file_name,
        raw_text=raw_text,
        # model_dump(mode="json") because JSONB cannot store Python floats
        # inside Pydantic objects directly -- this produces plain dicts,
        # lists, and scalars that psycopg can serialize.
        extracted=extracted.model_dump(mode="json") if extracted else None,
        seniority=extracted.seniority if extracted else None,
        years_experience=extracted.total_years_experience if extracted else None,
    )

    db.add(profile)
    db.commit()
    db.refresh(profile)

    return profile
