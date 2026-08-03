"""Turning documents into vectors, locally.

`all-MiniLM-L6-v2`, 384 dimensions, running on CPU inside the worker
container. Local rather than an API for one reason above the others:
re-embedding. The catalog gets re-embedded whenever the model changes, the
input text changes, or a bug is fixed, and at 800 postings that is a
ten-second loop offline versus 800 quota-consuming calls. A cost you notice
is a cost you avoid paying, and "I stopped iterating because it burned
quota" is how a scorer stays bad.

**The model is loaded once per process.** Load is seconds, inference is
milliseconds, so loading per job would make ingest roughly fifty times
slower for no benefit. It is lazy rather than at import so that importing
this module -- which the API process does transitively -- does not pull
90 MB of weights into a process that never embeds anything.

**Chunking is not optional here, and that is the non-obvious part.**
MiniLM's context is 256 word pieces. Anything longer is *silently truncated*
-- no error, no warning, just a vector describing the first fifth of the
document. Measured on 4,000 characters of ordinary prose: 1,119 word pieces,
of which the model would see 256. That is 23%.

For a resume, the first 23% is the contact block and the summary line. The
experience section -- the entire reason the document exists -- would never
reach the model, and every resume would embed to roughly "a person, with an
email address, who is looking for work". The scores would be plausible,
stable, and meaningless.

So documents are split into windows that fit, each window is embedded, and
the results are averaged. Mean pooling over chunks is the standard treatment
and it is a real approximation: a long document's vector drifts toward its
average topic, so a resume covering backend and ML lands between them rather
than in either. That is a known limitation of document-level embedding
rather than a defect here, and section 8.5's rerank is what recovers the
specifics.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# The model, and the dimension the schema is built around. Changing either
# means re-embedding every row -- and changing the dimension additionally
# means rewriting the column type and rebuilding the index, which is why
# spec section 8.2 says decide before writing the migration.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Word pieces, matching the model's own limit. Kept slightly under 256 so the
# [CLS] and [SEP] the tokenizer adds cannot push a chunk over and trigger the
# truncation this exists to prevent.
MAX_TOKENS = 250

# Chunks overlap by this many tokens so a sentence split across a boundary is
# whole in at least one of them. Without it, the phrase that carries a
# skill's context can land half in each chunk and be well represented in
# neither.
CHUNK_OVERLAP_TOKENS = 32

_model = None
_model_lock = threading.Lock()


def get_model():
    """The shared SentenceTransformer for this process.

    Double-checked locking rather than a bare module global: an RQ worker
    forks per job but the API process serves requests on a threadpool, and
    two threads racing here would load the weights twice and keep whichever
    finished last.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Imported inside the function so that `import app.embeddings`
                # stays cheap. sentence_transformers pulls in torch, which is
                # seconds of import time and hundreds of MB of RSS -- real
                # cost for any process that does not embed.
                from sentence_transformers import SentenceTransformer

                logger.info("loading embedding model %s", MODEL_NAME)
                _model = SentenceTransformer(MODEL_NAME)
    return _model


def chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Split `text` into pieces that fit the model's context.

    Splits on line boundaries first, because both inputs arrive as lines --
    `documents.extract_text` emits one per paragraph or table row, and
    `fetch.html_to_text` one per block element. Breaking there keeps
    semantically whole units together far more often than a fixed character
    window would.

    A single line longer than the budget is split on word boundaries, since
    one 900-token paragraph is otherwise unembeddable.
    """
    tokenizer = get_model().tokenizer

    def token_count(value: str) -> int:
        return len(tokenizer(value, add_special_tokens=False)["input_ids"])

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    # Pre-split any line that cannot fit on its own.
    units: list[str] = []
    for line in lines:
        if token_count(line) <= max_tokens:
            units.append(line)
            continue
        words, current = line.split(), []
        for word in words:
            current.append(word)
            if token_count(" ".join(current)) > max_tokens:
                current.pop()
                if current:
                    units.append(" ".join(current))
                current = [word]
        if current:
            units.append(" ".join(current))

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        size = token_count(unit)
        if current and current_tokens + size > max_tokens:
            chunks.append("\n".join(current))
            # Carry the tail forward so a boundary does not sever context.
            overlap: list[str] = []
            overlap_tokens = 0
            for previous in reversed(current):
                previous_size = token_count(previous)
                if overlap_tokens + previous_size > CHUNK_OVERLAP_TOKENS:
                    break
                overlap.insert(0, previous)
                overlap_tokens += previous_size
            current, current_tokens = overlap, overlap_tokens

        current.append(unit)
        current_tokens += size

    if current:
        chunks.append("\n".join(current))

    return chunks


def embed_text(text: str) -> list[float]:
    """Embed a document of any length into one unit vector.

    Returns a 384-float list, ready for a pgvector column. The vector is
    L2-normalized, which makes cosine similarity a plain dot product and lets
    pgvector's `<=>` operator be read directly as `1 - similarity`.

    Raises:
        ValueError: `text` has no content to embed. Silently returning a zero
            vector would be worse -- it is a valid-looking value at cosine
            distance 1.0 from everything, so it would rank last forever
            rather than announcing that nothing was embedded.
    """
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("cannot embed empty text")

    model = get_model()

    # Encoded in one batched call rather than a loop. The batching is where
    # the speed is: per-chunk calls pay the framework overhead every time.
    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    if len(vectors) == 1:
        return vectors[0].tolist()

    # Mean-pool, then renormalize. The mean of unit vectors is not itself a
    # unit vector, and skipping the second normalization would make a
    # document's magnitude depend on how internally consistent it is --
    # quietly penalising anyone whose resume covers two fields.
    import numpy as np

    pooled = np.asarray(vectors).mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm == 0.0:
        # Only reachable if chunks cancel out exactly, which needs adversarial
        # input. Returning the first chunk is a defensible answer; a zero
        # vector is not, for the reason in the docstring.
        return vectors[0].tolist()

    return (pooled / norm).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarity between two vectors from this module.

    A plain dot product, valid only because `embed_text` returns unit
    vectors. Used by the scorer when both vectors are already in hand;
    retrieval over the catalog uses pgvector's `<=>` in SQL instead, because
    the point of storing them in Postgres is not to pull 10,000 rows into
    Python to compare them.
    """
    return float(sum(x * y for x, y in zip(a, b)))
