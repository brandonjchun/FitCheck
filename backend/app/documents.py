"""
Raw text extraction from uploaded resume files.

Pure functions: bytes in, text out. No database, no network, no framework.
"""

from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import docx
import pdfplumber
from pdfplumber.utils.exceptions import PdfminerException


class DocumentError(Exception):
    """Base for every failure caused by the *input document*.

    The point of a common base is that the caller can write one `except
    DocumentError` and map the whole category to 400, without importing
    pdfplumber or zipfile to know what a bad upload looks like. Anything that
    is not a DocumentError is a bug in our code and should surface as a 500.

    This is the same error-classification idea the queue uses in M5, where
    failures split into "retry this" and "never retry this" -- classify by
    what the caller should *do*, not by where the error happened.
    """


class UnsupportedFileTypeError(DocumentError):
    """Raised when the uploaded file is not a format we can extract text from."""


class CorruptDocumentError(DocumentError):
    """Raised when the format is supported but the bytes cannot be parsed.

    Truncated uploads, garbage renamed to .pdf, a .docx that is not really a
    Word file. The client sent something unreadable -- that is their problem
    to fix, not a server fault.
    """


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

    Raises:
        CorruptDocumentError: If the bytes cannot be parsed as a PDF.
    """
    # pdfplumber needs a file-like object -- something with .read/.seek/.tell.
    # BytesIO wraps the bytes we already have in exactly that interface,
    # backed by memory instead of disk. Nothing touches the filesystem.
    buffer = BytesIO(file_bytes)

    # Only open() is inside the try. pdfminer validates the header and the
    # cross-reference table here, which is where every malformed PDF fails.
    # Keeping the loop outside means a bug in our own extraction code
    # surfaces as itself instead of being mislabelled a corrupt upload.
    try:
        pdf = pdfplumber.open(buffer)
    except PdfminerException as exc:
        raise CorruptDocumentError(
            "This file could not be read as a PDF. It may be corrupt, "
            "incomplete, or not actually a PDF."
        ) from exc

    page_texts: list[str] = []

    with pdf:
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

    Raises:
        CorruptDocumentError: If the bytes cannot be parsed as a DOCX.
    """
    buffer = BytesIO(file_bytes)

    # Two failure modes, both from this one call:
    #   BadZipFile -- the bytes are not a zip at all (a DOCX is a zip).
    #   KeyError   -- a valid zip missing the parts a Word document needs,
    #                 which python-docx surfaces as a failed lookup.
    # Catching a builtin as broad as KeyError is only safe because the try
    # covers exactly one call; widen it and this starts swallowing our bugs.
    try:
        document = docx.Document(buffer)
    except (BadZipFile, KeyError) as exc:
        raise CorruptDocumentError(
            "This file could not be read as a Word document. It may be "
            "corrupt, incomplete, or not actually a .docx."
        ) from exc

    paragraph_texts: list[str] = []

    for paragraph in document.paragraphs:
        text = paragraph.text
        if text:
            paragraph_texts.append(text)

    return "\n\n".join(paragraph_texts)
