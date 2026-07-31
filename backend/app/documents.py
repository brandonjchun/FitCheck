"""
Raw text extraction from uploaded resume files.

Pure functions: bytes in, text out. No database, no network, no framework.
"""

from io import BytesIO
from pathlib import Path

import pdfplumber
import docx


class UnsupportedFileTypeError(Exception):
    """Raised when the uploaded file is not a format we can extract text from."""


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from an uploaded resume.

    Args:
        file_bytes: The full contents of the uploaded file.
        filename: The original filename, e.g. "brandon_resume.pdf".

    Returns:
        The document's text content as a single string.

    Raises:
        UnsupportedFileTypeError: If the file is not a PDF or DOCX.
    """
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return _extract_pdf(file_bytes)
    if extension == ".docx":
        return _extract_docx(file_bytes)

    raise UnsupportedFileTypeError(
        f"Cannot extract text from '{extension or 'no extension'}': {filename}"
    )


def _extract_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF bytes.
    """
    # pdfplumber needs a file-like object -- something with .read/.seek/.tell.
    # BytesIO wraps the bytes we already have in exactly that interface,
    # backed by memory instead of disk. Nothing touches the filesystem.
    buffer = BytesIO(file_bytes)

    page_texts: list[str] = []

    with pdfplumber.open(buffer) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            # Pages with no text layer (scanned images) return None, not "".
            # str.join raises TypeError on a None element, so filter here.
            if text:
                page_texts.append(text)

    return "\n\n".join(page_texts)


def _extract_docx(file_bytes: bytes) -> str:
    """
    Extract text from DOCX bytes.

    Known Limitation: Won't pull information from tables.
    """
    buffer = BytesIO(file_bytes)

    paragraph_texts: list[str] = []

    document = docx.Document(buffer)
    for paragraph in document.paragraphs:
        text = paragraph.text
        if text:
            paragraph_texts.append(text)
    
    return "\n\n".join(paragraph_texts)
