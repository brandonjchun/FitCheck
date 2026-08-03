"""Shared test fixtures: document builders.

Every test document is constructed in memory at test time rather than
committed as a binary fixture. Two reasons -- .gitignore deliberately keeps
resumes out of this repo because they contain personal data, and a generated
document makes its own defect obvious. A committed `corrupt.pdf` is opaque;
`truncated_pdf()` says exactly what is wrong with it.
"""

import zipfile
from io import BytesIO
from uuid import uuid4

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


def make_docx_with_table(
    rows: list[list[str]],
    before: list[str] | None = None,
    after: list[str] | None = None,
) -> bytes:
    """Build a .docx with a table, optionally surrounded by paragraphs.

    `before` and `after` exist to pin document *order*. A table appended to the
    end of the file cannot distinguish an implementation that walks the body in
    sequence from one that reads paragraphs and tables separately and
    concatenates -- both pass. Content on either side of the table is what
    makes those two behaviours produce different output.
    """
    document = docx.Document()

    for text in before or []:
        document.add_paragraph(text)

    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    for row_index, row in enumerate(rows):
        for cell_index, text in enumerate(row):
            # Cells start life holding one empty paragraph, so assigning .text
            # replaces it rather than appending a second.
            table.cell(row_index, cell_index).text = text

    for text in after or []:
        document.add_paragraph(text)

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_docx_with_merged_row(header: str, body: list[str]) -> bytes:
    """Build a table whose first row is one cell merged across every column.

    The layout a resume uses for a section banner above its columns. python-docx
    reports a merged cell once per grid column it spans, so this is the fixture
    that catches an implementation emitting the header N times.
    """
    document = docx.Document()

    table = document.add_table(rows=2, cols=len(body))
    merged = table.cell(0, 0)
    for column in range(1, len(body)):
        merged = merged.merge(table.cell(0, column))
    merged.text = header

    for column, text in enumerate(body):
        table.cell(1, column).text = text

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_docx_with_nested_table(outer: str, inner: list[str]) -> bytes:
    """Build a table containing another table -- a cell holding a sub-grid.

    Rare in hand-written resumes and routine in ones exported from a template.
    Either the extractor recurses or the inner content vanishes silently.
    """
    document = docx.Document()

    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = outer

    host = table.cell(0, 1)
    nested = host.add_table(rows=1, cols=len(inner))
    for column, text in enumerate(inner):
        nested.cell(0, column).text = text

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
def make_user():
    """Create real user rows, cleaned up afterwards.

    Deleting the user cascades to their profiles and, through those, to their
    jobs -- so tests that create data under a user need no cleanup of their
    own.

    Emails are randomised because the unique index is case-insensitive and
    real: a fixed address would make the second test in a session collide
    with leftovers from the first.

    The domain is .dev rather than .local or .test on purpose -- those are
    IANA special-use domains and EmailStr rejects them, so a fixture using
    one would create accounts the API itself would refuse to register.
    """
    from app.db import SessionLocal
    from app.models import User
    from app.security import hash_password

    created: list[int] = []

    def _make(email: str | None = None, password: str = "testpassword123"):
        db = SessionLocal()
        try:
            user = User(
                email=email or f"t{uuid4().hex[:12]}@fitcheck.dev",
                password_hash=hash_password(password),
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            created.append(user.id)
            db.expunge(user)
            return user
        finally:
            db.close()

    yield _make

    db = SessionLocal()
    try:
        for user_id in created:
            obj = db.get(User, user_id)
            if obj is not None:
                db.delete(obj)
        db.commit()
    finally:
        db.close()


@pytest.fixture
def as_user():
    """Authenticate the test client as a given user.

    Overrides the `current_user` dependency rather than performing a real
    login, so tests that are about something else do not each need Redis and
    a cookie jar. `owned_profile` is deliberately *not* overridden -- it
    depends on `current_user`, so the real ownership predicate still runs and
    is still what the authorization tests exercise.

    The genuine cookie flow is covered end to end in test_auth.py.
    """
    from app.main import app
    from app.security import current_user

    def _as(user):
        app.dependency_overrides[current_user] = lambda: user
        return user

    yield _as

    app.dependency_overrides.clear()


@pytest.fixture
def anyio_backend() -> str:
    """Run @pytest.mark.anyio tests on asyncio only.

    anyio's pytest plugin parameterises over asyncio and trio by default;
    trio is not installed and this app only ever runs on asyncio.
    """
    return "asyncio"
