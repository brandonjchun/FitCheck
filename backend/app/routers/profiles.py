"""Routes for resume upload and candidate profiles."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.documents import DocumentError, extract_text
from app.schemas import ProfileUploadResponse

# A router is a mountable group of routes. main.py calls
# app.include_router(profiles.router), which is what makes these paths live.
# The prefix and tags apply to every route defined below; tags become the
# section headings in the Swagger UI at /docs.
router = APIRouter(prefix="/api/profiles", tags=["profiles"])


@router.post("", response_model=ProfileUploadResponse)
def upload_resume(file: UploadFile = File(...)) -> ProfileUploadResponse:
    """
    Accept a resume upload and return its extracted text.
    """
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = file.file.read()
    file_name = file.filename

    try:
        raw_text = extract_text(content, file_name)
    except DocumentError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ProfileUploadResponse(
        filename=file_name,
        characters=len(raw_text),
        raw_text=raw_text,
    )
