"""Two-stage retrieval: what recall is allowed to return, and how it ranks.

The recall stage is guarded more heavily than its size suggests, because its
failures are all silent. It returns rows; the question is always whether it
returned the *right* rows, and a feed full of wrong ones looks exactly like a
feed full of right ones until somebody reads it.
"""

import pytest
from sqlalchemy import select, text

from app.db import SessionLocal
from app.embeddings import EMBEDDING_DIM
from app.extraction import POSTING_EXTRACTION_VERSION
from app.models import JobPosting, Match, Profile
from app.retrieval import FeedFilters, recall_candidates
from app.scoring import SCORER_VERSION
from app.workers import tasks


def _vec(*leading: float) -> list[float]:
    """A unit-ish vector whose first coordinates are the interesting ones."""
    values = list(leading) + [0.0] * (EMBEDDING_DIM - len(leading))
    return values[:EMBEDDING_DIM]


def _skills(*names: str) -> dict:
    return {"skills": [{"name": n, "necessity": "required"} for n in names]}


class Catalog:
    """Postings this test created, and only those.

    The dev database is a real one with a crawled catalog in it, so a bare
    `recall_candidates` call returns other people's rows alongside the
    fixture's. Every assertion here is therefore scoped through `mine`:
    tests state what recall does with *these* postings and stay silent about
    the rest, which is the only form that holds whether the database is empty
    or has ten thousand rows in it.
    """

    def __init__(self):
        self.ids: list[int] = []

    def mine(self, posting_ids) -> list[int]:
        owned = set(self.ids)
        return [pid for pid in posting_ids if pid in owned]


@pytest.fixture
def catalog():
    """A handful of postings with known vectors, cleaned up afterwards."""
    registry = Catalog()
    created = registry.ids

    def _make(
        key: str,
        embedding: list[float] | None = None,
        *,
        extracted: dict | None = None,
        closed: bool = False,
        remote_type: str | None = None,
        seniority: str | None = None,
        min_years: float | None = None,
    ) -> int:
        db = SessionLocal()
        try:
            posting = JobPosting(
                canonical_key=f"test:{key}:{len(created)}",
                url=f"https://example.com/{key}",
                content_hash=f"hash-{key}-{len(created)}",
                raw_text="A job posting body.",
                embedding=embedding,
                extracted=extracted,
                extraction_version=POSTING_EXTRACTION_VERSION if extracted else None,
                remote_type=remote_type,
                seniority=seniority,
                min_years=min_years,
            )
            db.add(posting)
            db.commit()
            db.refresh(posting)
            pid = posting.id
            if closed:
                db.execute(
                    text("UPDATE job_postings SET closed_at = now() WHERE id = :i"),
                    {"i": pid},
                )
                db.commit()
        finally:
            db.close()
        created.append(pid)
        return pid

    registry.make = _make
    yield registry

    db = SessionLocal()
    try:
        db.query(Match).filter(Match.job_posting_id.in_(created)).delete(
            synchronize_session=False
        )
        db.query(JobPosting).filter(JobPosting.id.in_(created)).delete(
            synchronize_session=False
        )
        db.commit()
    finally:
        db.close()


@pytest.fixture
def scored_profile(make_user):
    """A profile with an embedding and one extracted skill."""
    user = make_user()
    db = SessionLocal()
    try:
        profile = Profile(
            user_id=user.id,
            original_filename="t.pdf",
            raw_text="Brandon uses Python.",
            embedding=_vec(1.0),
            extracted={"skills": [{"name": "Python", "years": 5}]},
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        pid = profile.id
    finally:
        db.close()
    return pid


def _recall_ids(vector, **kwargs) -> list[int]:
    db = SessionLocal()
    try:
        return [c.posting_id for c in recall_candidates(db, vector, **kwargs)]
    finally:
        db.close()


def _matches_for(profile_id: int) -> dict[int, Match]:
    db = SessionLocal()
    try:
        return {
            m.job_posting_id: m
            for m in db.execute(
                select(Match).where(Match.profile_id == profile_id)
            ).scalars()
        }
    finally:
        db.close()


class TestRecallEligibility:
    def test_closed_postings_are_never_recalled(self, catalog):
        live = catalog.make("live", _vec(1.0), extracted=_skills("Python"))
        catalog.make("dead", _vec(1.0), extracted=_skills("Python"), closed=True)

        assert catalog.mine(_recall_ids(_vec(1.0))) == [live]

    def test_unembedded_postings_are_never_recalled(self, catalog):
        catalog.make("no-vector", None, extracted=_skills("Python"))
        assert catalog.mine(_recall_ids(_vec(1.0))) == []

    def test_unextracted_postings_are_never_recalled(self, catalog):
        """The ranking-poisoning guard.

        `score_skills` scores an empty requirement list as 1.0, and an
        unextracted posting presents an empty requirement list -- so without
        this filter a posting nothing is known about outranks every posting
        that was actually assessed.

        The fixture writes `extracted=None`, which SQLAlchemy persists as the
        JSON value `null` rather than SQL NULL. That is deliberate here: it is
        the case an `IS NOT NULL` guard fails to catch, so this test only
        passes against a predicate that inspects the JSON type.
        """
        catalog.make("raw", _vec(1.0), extracted=None)
        assert catalog.mine(_recall_ids(_vec(1.0))) == []

    def test_ordering_is_by_similarity(self, catalog):
        """All three stay strongly aligned with the query, differing by degree.

        An orthogonal "far" posting would be the natural way to write this and
        is quietly wrong against a real database: its similarity is 0, real
        catalog postings score small positive values, and once the catalog
        exceeds RECALL_LIMIT the orthogonal one is pushed out of the window
        entirely. The test would then fail for a reason that has nothing to do
        with ordering.
        """
        near = catalog.make("near", _vec(1.0, 0.0), extracted=_skills("Python"))
        mid = catalog.make("mid", _vec(0.95, 0.31), extracted=_skills("Python"))
        far = catalog.make("far", _vec(0.9, 0.44), extracted=_skills("Python"))

        assert catalog.mine(_recall_ids(_vec(1.0, 0.0))) == [near, mid, far]

    def test_limit_bounds_the_shortlist(self, catalog):
        for i in range(5):
            catalog.make(f"p{i}", _vec(1.0), extracted=_skills("Python"))
        assert len(_recall_ids(_vec(1.0), limit=3)) == 3

    def test_semantic_score_is_similarity_not_distance(self, catalog):
        mine = catalog.make("same", _vec(1.0), extracted=_skills("Python"))
        db = SessionLocal()
        try:
            scores = {
                c.posting_id: c.semantic_score
                for c in recall_candidates(db, _vec(1.0))
            }
        finally:
            db.close()
        # Identical unit vectors: cosine similarity 1, cosine distance 0.
        assert scores[mine] == pytest.approx(1.0, abs=1e-6)


class TestRecallFilters:
    def test_remote_only(self, catalog):
        remote = catalog.make(
            "r", _vec(1.0), extracted=_skills("Python"), remote_type="remote"
        )
        catalog.make("o", _vec(1.0), extracted=_skills("Python"), remote_type="onsite")

        ids = _recall_ids(_vec(1.0), filters=FeedFilters(remote_only=True))
        assert catalog.mine(ids) == [remote]

    def test_seniority_band(self, catalog):
        senior = catalog.make(
            "s", _vec(1.0), extracted=_skills("Python"), seniority="senior"
        )
        catalog.make("j", _vec(1.0), extracted=_skills("Python"), seniority="junior")

        ids = _recall_ids(_vec(1.0), filters=FeedFilters(seniority=("senior",)))
        assert catalog.mine(ids) == [senior]

    def test_unstated_years_requirement_survives_a_years_filter(self, catalog):
        """NULL min_years means "never said", which is not "demands zero".

        Excluding these would silently drop every posting whose extraction was
        thin, which correlates with nothing a user cares about.
        """
        unstated = catalog.make(
            "u", _vec(1.0), extracted=_skills("Python"), min_years=None
        )
        low = catalog.make("l", _vec(1.0), extracted=_skills("Python"), min_years=2)
        catalog.make("h", _vec(1.0), extracted=_skills("Python"), min_years=10)

        ids = _recall_ids(_vec(1.0), filters=FeedFilters(max_min_years=3))
        assert set(catalog.mine(ids)) == {unstated, low}


class TestScoreProfile:
    def test_persists_recommendations_for_the_best_candidates(
        self, catalog, scored_profile
    ):
        good = catalog.make("good", _vec(1.0), extracted=_skills("Python"))
        # Aligned with the query but demanding skills the profile lacks, so
        # the two are separated by the *rerank* rather than by recall. Making
        # it orthogonal instead would push it out of the recall window on a
        # populated catalog and the test would pass without reranking anything.
        bad = catalog.make("bad", _vec(0.93, 0.37), extracted=_skills("Fortran", "COBOL"))

        assert tasks.score_profile(scored_profile) == "scored"

        matches = _matches_for(scored_profile)

        assert {good, bad} <= set(matches)
        assert matches[good].final_score > matches[bad].final_score
        assert matches[good].origin == "recommendation"
        assert matches[good].scorer_version == SCORER_VERSION
        assert matches[good].breakdown["counts"]["matched"] == 1

    def test_limit_caps_what_reaches_the_feed(self, catalog, scored_profile):
        for i in range(5):
            catalog.make(f"p{i}", _vec(1.0, i / 10), extracted=_skills("Python"))

        tasks.score_profile(scored_profile, limit=2)
        assert len(_matches_for(scored_profile)) == 2

    def test_rescoring_a_user_submission_does_not_relabel_its_origin(
        self, catalog, scored_profile
    ):
        """`origin` records how a pair first came to be scored.

        A later recommender pass has no standing to rewrite that history, and
        the feed UI reads it to tell "you asked about this" from "we suggested
        this".
        """
        posting = catalog.make("submitted", _vec(1.0), extracted=_skills("Python"))
        tasks.score_posting_for_profile(scored_profile, posting)

        tasks.score_profile(scored_profile)

        assert _matches_for(scored_profile)[posting].origin == "user_submission"

    def test_is_idempotent(self, catalog, scored_profile):
        catalog.make("p", _vec(1.0), extracted=_skills("Python"))

        tasks.score_profile(scored_profile)
        first = _matches_for(scored_profile)
        tasks.score_profile(scored_profile)
        second = _matches_for(scored_profile)

        assert set(first) == set(second), "re-running must upsert rather than append"

    def test_profile_without_an_embedding_is_reported_not_scored(
        self, catalog, make_user
    ):
        """Scoring anyway would rank on skills alone and look like it worked."""
        catalog.make("p", _vec(1.0), extracted=_skills("Python"))
        user = make_user()
        db = SessionLocal()
        try:
            profile = Profile(
                user_id=user.id,
                original_filename="t.pdf",
                raw_text="",
                embedding=None,
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)
            pid = profile.id
        finally:
            db.close()

        # _embed_profile is what would otherwise fill this in; an empty
        # document is the case that genuinely cannot produce a vector.
        assert tasks.score_profile(pid) in {"no_embedding", "no_candidates"}

    def test_missing_profile_is_not_an_error(self):
        assert tasks.score_profile(2**40) == "profile_missing"

    def test_empty_catalog_returns_cleanly(self, scored_profile):
        assert tasks.score_profile(scored_profile) in {"no_candidates", "scored"}
