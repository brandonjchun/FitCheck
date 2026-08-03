"""Tests for app.documents -- extraction and its error classification.

These are the cheapest tests in the project to write and the most valuable to
have: extract_text is a pure function, so every case here is bytes in, string
or exception out. No database, no network, no app.
"""

import pytest
from conftest import (
    make_docx,
    make_pdf,
    make_scanned_pdf,
    make_zip_without_word_parts,
)

from app.documents import (
    CorruptDocumentError,
    DocumentError,
    UnsupportedFileTypeError,
    extract_text,
)


class TestDispatch:
    """extract_text picks a parser from the filename extension."""

    def test_unsupported_extension_rejected(self) -> None:
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            extract_text(b"plain text", "resume.txt")

        # The message names the offending extension -- this string reaches the
        # user as a 400 detail, so it has to be actionable on its own.
        assert ".txt" in str(exc_info.value)

    def test_missing_extension_rejected(self) -> None:
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            extract_text(b"plain text", "resume")

        assert "no extension" in str(exc_info.value)

    def test_extension_matching_is_case_insensitive(self) -> None:
        # Windows and email clients both produce .PDF often enough that a
        # case-sensitive check would reject real resumes.
        assert extract_text(make_pdf("Brandon"), "RESUME.PDF").strip() == "Brandon"

    def test_dispatch_uses_last_extension_only(self) -> None:
        # "my.resume.v2.docx" must route on .docx, not on .resume or .v2.
        text = extract_text(make_docx(["Berkeley"]), "my.resume.v2.docx")
        assert "Berkeley" in text


class TestPdf:
    def test_extracts_text_layer(self) -> None:
        assert "Hello" in extract_text(make_pdf("Hello"), "resume.pdf")

    def test_garbage_renamed_to_pdf(self) -> None:
        with pytest.raises(CorruptDocumentError):
            extract_text(b"this is not a pdf at all", "resume.pdf")

    def test_truncated_pdf(self) -> None:
        # Cutting the file in half destroys the cross-reference table, which
        # is what an interrupted upload actually produces.
        full = make_pdf("Hello")
        with pytest.raises(CorruptDocumentError):
            extract_text(full[: len(full) // 2], "resume.pdf")

    def test_empty_bytes(self) -> None:
        with pytest.raises(CorruptDocumentError):
            extract_text(b"", "resume.pdf")

    def test_scanned_pdf_returns_empty_string(self) -> None:
        """A PDF with no text layer is valid input that yields nothing.

        This is the case worth being deliberate about. Pages without a text
        layer return None from extract_text(), and str.join raises TypeError
        on a None element -- so the filter in _extract_pdf is load-bearing,
        and this test is what keeps it there.

        Returning "" rather than raising is the right call at this layer:
        the document parsed fine, it simply has no text. Deciding that an
        empty resume is a client error belongs to the endpoint, not here.
        """
        assert extract_text(make_scanned_pdf(), "scan.pdf") == ""


class TestDocx:
    def test_extracts_paragraphs(self) -> None:
        text = extract_text(make_docx(["Brandon Chun", "Berkeley"]), "resume.docx")

        assert "Brandon Chun" in text
        assert "Berkeley" in text

    def test_paragraphs_are_blank_line_separated(self) -> None:
        text = extract_text(make_docx(["First", "Second"]), "resume.docx")
        assert text == "First\n\nSecond"

    def test_empty_paragraphs_dropped(self) -> None:
        # Word documents are full of empty paragraphs used as spacing. Keeping
        # them would hand the LLM a document that is mostly blank lines.
        text = extract_text(make_docx(["First", "", "   ", "Second"]), "resume.docx")
        assert text == "First\n\n   \n\nSecond"

    def test_not_a_zip(self) -> None:
        with pytest.raises(CorruptDocumentError):
            extract_text(b"this is not a docx at all", "resume.docx")

    def test_valid_zip_missing_word_parts(self) -> None:
        # Passes the BadZipFile check and fails on the KeyError path instead.
        with pytest.raises(CorruptDocumentError):
            extract_text(make_zip_without_word_parts(), "resume.docx")

    def test_pdf_renamed_to_docx(self) -> None:
        with pytest.raises(CorruptDocumentError):
            extract_text(make_pdf("Hello"), "resume.docx")


class TestErrorHierarchy:
    """The base class is the contract the router depends on.

    routers/profiles.py catches DocumentError alone and maps it to 400. If a
    future extraction path raises something outside this hierarchy, that
    handler misses it and the user gets a 500 for their own bad file.
    """

    @pytest.mark.parametrize(
        ("file_bytes", "filename"),
        [
            (b"plain text", "resume.txt"),
            (b"not a pdf", "resume.pdf"),
            (b"not a docx", "resume.docx"),
        ],
        ids=["unsupported", "corrupt-pdf", "corrupt-docx"],
    )
    def test_every_input_failure_is_a_document_error(
        self, file_bytes: bytes, filename: str
    ) -> None:
        with pytest.raises(DocumentError):
            extract_text(file_bytes, filename)

    def test_subclasses_share_the_base(self) -> None:
        assert issubclass(UnsupportedFileTypeError, DocumentError)
        assert issubclass(CorruptDocumentError, DocumentError)
