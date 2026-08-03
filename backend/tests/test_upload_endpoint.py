"""Tests for the real application's wiring.

Every case here is chosen so it resolves *before* the handler reaches the
database or the LLM provider: a rejected upload, a validation failure, or an
unparseable document never gets that far. So this module needs neither
Postgres nor Redis running -- SQLAlchemy's engine is created at import time
but does not open a connection until a query is issued.

Kept separate from test_middleware.py because this module imports app.main,
which transitively imports the extraction and provider modules. A failure
here points at wiring; a failure there points at the middleware itself.
"""

import pytest
from conftest import make_docx, make_pdf, make_scanned_pdf
from fastapi.testclient import TestClient

from app.main import app
from app.middleware import MAX_UPLOAD_BYTES
from app.models import User

UPLOAD_URL = "/api/profiles"


@pytest.fixture
def client(as_user) -> TestClient:
    """An authenticated client that still touches no database.

    Every case in this module resolves before the handler reaches storage, so
    the identity only has to exist, not to be persisted -- an unsaved User
    with an id is enough. Overriding the dependency keeps this module's
    "needs neither Postgres nor Redis" property intact now that the endpoints
    require a session.
    """
    as_user(User(id=1, email="t@test.local", password_hash="x"))
    return TestClient(app, raise_server_exceptions=False)


def test_upload_requires_authentication() -> None:
    """The one case here that must NOT be authenticated.

    Uses its own unauthenticated client, because the module-level fixture
    exists precisely to bypass this.
    """
    anonymous = TestClient(app, raise_server_exceptions=False)

    response = anonymous.post(
        UPLOAD_URL, files={"file": ("resume.pdf", make_pdf("hi"), "application/pdf")}
    )

    assert response.status_code == 401


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class TestSizeLimitWiring:
    """The middleware is mounted and its exception is mapped.

    test_middleware.py proves the middleware works. This proves main.py
    actually installed it on the app the server runs, and that the
    RequestBodyTooLarge raised from the receive channel becomes a 413 rather
    than the 500 it would be without the registered handler.
    """

    def test_oversized_upload_rejected_with_413(self, client: TestClient) -> None:
        # Rejected on the Content-Length pre-check, before multipart parsing
        # allocates anything and before any dependency is resolved.
        response = client.post(UPLOAD_URL, content=b"x" * (MAX_UPLOAD_BYTES + 1))

        assert response.status_code == 413

    def test_oversized_chunked_upload_rejected_with_413(
        self, client: TestClient
    ) -> None:
        """No Content-Length, so only the streaming counter can catch this.

        The body has to be well-formed multipart, not just oversized bytes.
        A body FastAPI cannot parse as a form is rejected with 422 before
        enough of it has been read to cross the cap -- so an unstructured
        payload here would pass for the wrong reason and prove nothing.

        Without the @app.exception_handler in main.py this returns 500.
        """
        boundary = "----fitcheck-test-boundary"

        def chunks():
            yield (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="file"; '
                'filename="resume.pdf"\r\n'
                "Content-Type: application/pdf\r\n\r\n"
            ).encode()
            # 2.25 MB of payload against a 2 MB cap.
            for _ in range(9):
                yield b"x" * (256 * 1024)
            yield f"\r\n--{boundary}--\r\n".encode()

        # Passing an iterator makes httpx use chunked transfer encoding, so
        # the Content-Length pre-check has nothing to inspect.
        response = client.post(
            UPLOAD_URL,
            content=chunks(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

        assert response.status_code == 413


class TestRequestValidation:
    def test_missing_file_field_returns_422(self, client: TestClient) -> None:
        # FastAPI's own validation, from the File(...) annotation. Worth a test
        # because it is easy to break by making the parameter optional.
        response = client.post(UPLOAD_URL, files={})

        assert response.status_code == 422

    def test_wrong_field_name_returns_422(self, client: TestClient) -> None:
        response = client.post(
            UPLOAD_URL, files={"resume": ("resume.pdf", make_pdf(), "application/pdf")}
        )

        assert response.status_code == 422


class TestDocumentErrorsBecome400:
    """DocumentError is the router's client-error contract.

    A bad upload is the user's problem to fix and must never read as a server
    fault -- a 500 here would page someone for a corrupt PDF.
    """

    def test_unsupported_file_type_returns_400(self, client: TestClient) -> None:
        response = client.post(
            UPLOAD_URL, files={"file": ("resume.txt", b"plain text", "text/plain")}
        )

        assert response.status_code == 400
        assert ".txt" in response.json()["detail"]

    def test_corrupt_pdf_returns_400(self, client: TestClient) -> None:
        response = client.post(
            UPLOAD_URL,
            files={"file": ("resume.pdf", b"not a pdf", "application/pdf")},
        )

        assert response.status_code == 400

    def test_truncated_docx_returns_400(self, client: TestClient) -> None:
        full = make_docx(["Brandon Chun"])

        response = client.post(
            UPLOAD_URL,
            files={
                "file": (
                    "resume.docx",
                    full[: len(full) // 2],
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        assert response.status_code == 400

    def test_content_type_header_is_not_trusted(self, client: TestClient) -> None:
        """Routing is by extension, and the declared MIME type is ignored.

        A browser will happily label a .txt as application/pdf. The bytes are
        what matter, and they are not a PDF.
        """
        response = client.post(
            UPLOAD_URL, files={"file": ("resume.txt", b"plain text", "application/pdf")}
        )

        assert response.status_code == 400


class TestEmptyDocumentRejected:
    """A file can parse cleanly and still contain nothing usable.

    422 rather than 400: the request was well-formed and the file really was
    a valid PDF. Only the content is unusable, which is a different failure
    from a corrupt upload.
    """

    def test_scanned_pdf_returns_422(self, client: TestClient) -> None:
        """The common case -- resumes exported from phone scanner apps.

        pdfplumber returns "" for a PDF that is images all the way down, with
        no error. Before this check the upload succeeded and produced a
        profile with zero skills, no error, and nothing to distinguish it
        from a resume the model genuinely found nothing in.
        """
        response = client.post(
            UPLOAD_URL,
            files={"file": ("scan.pdf", make_scanned_pdf(), "application/pdf")},
        )

        assert response.status_code == 422

    def test_rejection_explains_what_to_do(self, client: TestClient) -> None:
        """"Unprocessable entity" alone leaves the user with no next step.

        They uploaded a file that looks fine to them; the message has to name
        the actual problem and a way out.
        """
        response = client.post(
            UPLOAD_URL,
            files={"file": ("scan.pdf", make_scanned_pdf(), "application/pdf")},
        )

        detail = response.json()["detail"]
        assert "no text layer" in detail
        assert "DOCX" in detail

    def test_whitespace_only_docx_returns_422(self, client: TestClient) -> None:
        """Not just scans: an empty DOCX reaches the same dead end."""
        response = client.post(
            UPLOAD_URL,
            files={
                "file": (
                    "empty.docx",
                    make_docx(["   ", "\t"]),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        assert response.status_code == 422
