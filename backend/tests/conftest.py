"""Shared test fixtures: document builders.

Every test document is constructed in memory at test time rather than
committed as a binary fixture. Two reasons -- .gitignore deliberately keeps
resumes out of this repo because they contain personal data, and a generated
document makes its own defect obvious. A committed `corrupt.pdf` is opaque;
`truncated_pdf()` says exactly what is wrong with it.
"""

import zipfile
from io import BytesIO

import docx
import pytest
from PIL import Image


def _escape_pdf_text(text: str) -> bytes:
    """Escape the three characters that are syntax inside a PDF string."""
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    return escaped.encode("ascii")


def make_pdf(text: str = "Hello") -> bytes:
    """Build a minimal single-page PDF with a real text layer.

    Hand-assembled rather than pulled from a library because nothing in
    requirements.txt writes PDFs, and adding a dependency to produce five
    test objects is a poor trade. The byte offsets in the cross-reference
    table are computed as the file is built -- pdfminer validates that table
    on open(), so they have to be right or every test here would exercise the
    corrupt-document path by accident.
    """
    stream = b"BT /F1 24 Tf 72 700 Td (" + _escape_pdf_text(text) + b") Tj ET"

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_offset = len(out)
    size = len(objects) + 1

    out += b"xref\n0 " + str(size).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")

    out += b"trailer\n<< /Size " + str(size).encode() + b" /Root 1 0 R >>\n"
    out += b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF\n"

    return bytes(out)


def make_scanned_pdf() -> bytes:
    """Build a PDF whose only content is an image -- no text layer at all.

    This is what a photographed or scanned resume looks like to pdfplumber:
    a structurally valid PDF where every page's extract_text() returns None.
    Pillow is already a pdfplumber dependency, so this costs nothing.
    """
    buffer = BytesIO()
    Image.new("RGB", (200, 200), "white").save(buffer, format="PDF")
    return buffer.getvalue()


def make_docx(paragraphs: list[str]) -> bytes:
    """Build a real .docx containing the given paragraphs."""
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_zip_without_word_parts() -> bytes:
    """A structurally valid zip that is not a Word document.

    A .docx *is* a zip, so this passes the BadZipFile check and fails later,
    when python-docx looks for the document parts that are not there. It is
    the second of the two failure modes _extract_docx catches.
    """
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("not-a-word-document.txt", "hello")
    return buffer.getvalue()


@pytest.fixture
def anyio_backend() -> str:
    """Run @pytest.mark.anyio tests on asyncio only.

    anyio's pytest plugin parameterises over asyncio and trio by default;
    trio is not installed and this app only ever runs on asyncio.
    """
    return "asyncio"
