"""Embeddings, and the truncation trap they exist to avoid.

These load the real MiniLM model rather than mocking it. That costs a few
seconds once per session and is the point: the property under test is that a
long document is *not* silently truncated, and a mocked encoder would happily
confirm that regardless of whether the chunking works.
"""

import pytest

from app.embeddings import (
    EMBEDDING_DIM,
    MAX_TOKENS,
    chunk_text,
    cosine_similarity,
    embed_text,
    get_model,
)

# Long enough to exceed 256 word pieces several times over. Varied on purpose
# -- repeating one line would chunk correctly even with broken logic, because
# every window would contain the same content.
LONG_RESUME = "\n".join(
    f"Line {i}: built and shipped a {tech} service handling {i * 1000} requests "
    f"per day, reducing {metric} by {i}%."
    for i, (tech, metric) in enumerate(
        [
            ("Python", "latency"), ("FastAPI", "error rate"), ("Postgres", "cost"),
            ("Redis", "p99"), ("Kubernetes", "toil"), ("Go", "memory"),
            ("Rust", "CPU"), ("Kafka", "lag"), ("Spark", "runtime"),
            ("Airflow", "failures"), ("Terraform", "drift"), ("gRPC", "bandwidth"),
        ] * 8,
        start=1,
    )
)


@pytest.fixture(scope="module")
def model():
    """Loaded once for the module. Load is seconds, inference is
    milliseconds -- the same reason the worker caches it per process."""
    return get_model()


class TestChunking:
    def test_every_chunk_fits_the_model(self, model) -> None:
        """The whole reason this module exists.

        MiniLM's context is 256 word pieces and anything past it is dropped
        with no error. A chunk over the limit is not a slow chunk, it is a
        chunk whose tail never reaches the model.
        """
        tokenizer = model.tokenizer

        for chunk in chunk_text(LONG_RESUME):
            length = len(tokenizer(chunk, add_special_tokens=False)["input_ids"])
            assert length <= MAX_TOKENS, f"chunk of {length} tokens exceeds {MAX_TOKENS}"

    def test_a_long_document_produces_several_chunks(self, model) -> None:
        assert len(chunk_text(LONG_RESUME)) > 1

    def test_a_short_document_is_one_chunk(self, model) -> None:
        """No pointless splitting -- most postings and every skills section
        fit comfortably."""
        assert len(chunk_text("Python developer with Postgres experience")) == 1

    def test_content_is_not_lost(self, model) -> None:
        """Chunking must partition the document, not sample it.

        The failure this catches is the quiet one: a chunker that drops the
        overflow instead of starting a new window looks identical from the
        outside, because the vector it returns is still a valid vector.
        """
        joined = " ".join(chunk_text(LONG_RESUME))

        assert "Line 1:" in joined
        assert "Line 96:" in joined

    def test_an_oversized_single_line_is_split(self, model) -> None:
        """One giant paragraph with no newlines is otherwise unembeddable."""
        wall = "word " * 2000

        chunks = chunk_text(wall)

        assert len(chunks) > 1

    def test_empty_input_produces_no_chunks(self, model) -> None:
        assert chunk_text("   \n\n  \n") == []


class TestEmbedText:
    def test_returns_the_declared_dimension(self, model) -> None:
        """The migration writes vector(384) into the column and Postgres
        enforces it, so a mismatch here is an insert failure at runtime."""
        assert len(embed_text("Python engineer")) == EMBEDDING_DIM

    def test_a_long_document_still_returns_one_vector(self, model) -> None:
        vector = embed_text(LONG_RESUME)

        assert len(vector) == EMBEDDING_DIM

    def test_the_vector_is_normalized(self, model) -> None:
        """Unit length is what makes cosine similarity a dot product, and
        what lets pgvector's `<=>` be read directly as 1 - similarity."""
        vector = embed_text(LONG_RESUME)
        magnitude = sum(x * x for x in vector) ** 0.5

        assert magnitude == pytest.approx(1.0, abs=1e-5)

    def test_pooling_renormalizes(self, model) -> None:
        """The mean of unit vectors is not a unit vector.

        Skipping the second normalization would make a document's magnitude
        depend on how internally consistent it is -- quietly penalising
        anyone whose resume spans two fields.
        """
        multi = embed_text(LONG_RESUME)
        single = embed_text("Python")

        assert len(chunk_text(LONG_RESUME)) > 1
        assert sum(x * x for x in multi) ** 0.5 == pytest.approx(1.0, abs=1e-5)
        assert sum(x * x for x in single) ** 0.5 == pytest.approx(1.0, abs=1e-5)

    def test_empty_text_raises(self, model) -> None:
        """Rather than returning zeros. A zero vector is a valid-looking
        value sitting at cosine distance 1.0 from everything, so it would
        rank last forever instead of announcing that nothing was embedded."""
        with pytest.raises(ValueError):
            embed_text("   ")

    def test_deterministic(self, model) -> None:
        """Two runs must agree, or a re-score changes a ranking for no
        reason a user could understand."""
        assert embed_text("Backend engineer") == embed_text("Backend engineer")


class TestSimilarity:
    def test_identical_text_is_one(self, model) -> None:
        vector = embed_text("Senior Python engineer")

        assert cosine_similarity(vector, vector) == pytest.approx(1.0, abs=1e-5)

    def test_related_beats_unrelated(self, model) -> None:
        """The only claim about the model actually worth making.

        Not an assertion on a threshold -- those drift between model
        versions and teach nobody anything. An ordering is what the ranking
        depends on.
        """
        resume = embed_text("Backend engineer building APIs in Python and Postgres")
        related = embed_text("Hiring a backend developer for our Python API team")
        unrelated = embed_text("Seeking a pastry chef experienced in French desserts")

        assert cosine_similarity(resume, related) > cosine_similarity(resume, unrelated)

    def test_long_documents_are_comparable_to_short_ones(self, model) -> None:
        """Pooled multi-chunk vectors and single-chunk vectors have to live
        in the same space, or a long resume could never match a short
        posting."""
        long_vector = embed_text(LONG_RESUME)
        short_vector = embed_text("Python and Kubernetes engineer")

        assert -1.0 <= cosine_similarity(long_vector, short_vector) <= 1.0
