"""Operator gating on the ops endpoints.

The sibling of test_authorization.py, and the contrast between them is the
point. There, the rule is **404** -- a profile you do not own must be
indistinguishable from one that never existed, because a 403 would confirm
the row and turn the endpoint into an id-enumeration oracle.

Here the rule is **403**, and for the opposite reason: `/api/ops/overview` is
a fixed path published in the OpenAPI document, so there is no existence to
conceal. Answering 404 would hide nothing and would mislead a legitimate
operator into debugging a missing endpoint when their account is simply
missing a flag.

`current_user` is overridden so a test can act as a chosen account, but
`require_admin` is the real one -- it depends on `current_user`, so the
actual flag check still runs and is what these assert.
"""

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal
from app.main import app
from app.models import User

OPS_READ_ENDPOINTS = ["/api/ops/overview", "/api/ops/dead-letter"]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def promote():
    """Grant operator access to an existing user.

    Writes the row *and* updates the detached instance the fixture handed
    back, so a test asserting through the dependency override and a test
    asserting through the database agree about the same account.
    """

    def _promote(user: User) -> User:
        db = SessionLocal()
        try:
            row = db.get(User, user.id)
            row.is_admin = True
            db.commit()
        finally:
            db.close()
        user.is_admin = True
        return user

    return _promote


class TestNonAdminIsRefused:
    @pytest.mark.parametrize("path", OPS_READ_ENDPOINTS)
    def test_read_endpoints_are_403(self, client, as_user, make_user, path):
        as_user(make_user())
        assert client.get(path).status_code == 403

    def test_requeue_is_403(self, client, as_user, make_user):
        as_user(make_user())
        assert client.post("/api/ops/jobs/1/requeue").status_code == 403

    def test_403_not_404(self, client, as_user, make_user):
        """The deliberate divergence from owned_profile's 404.

        A 404 here would tell an operator whose account lacks the flag that
        the endpoint does not exist, sending them to look for a deployment
        problem that is not there.
        """
        as_user(make_user())
        status = client.get("/api/ops/overview").status_code
        assert status == 403
        assert status != 404

    def test_refused_before_the_handler_runs(self, client, as_user, make_user):
        """Requeue of a nonexistent job still 403s rather than 404ing.

        Proves the gate is a router-level dependency rather than a check
        inside each handler: if the handler ran at all, job 999999 would
        produce the 404 it raises for a missing row. A non-admin must not be
        able to probe which job ids exist.
        """
        as_user(make_user())
        assert client.post("/api/ops/jobs/999999/requeue").status_code == 403


class TestAdminIsAllowed:
    def test_overview_is_200(self, client, as_user, make_user, promote):
        as_user(promote(make_user()))
        response = client.get("/api/ops/overview")
        assert response.status_code == 200
        assert "queues" in response.json()

    def test_dead_letter_is_200(self, client, as_user, make_user, promote):
        as_user(promote(make_user()))
        response = client.get("/api/ops/dead-letter")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_requeue_reaches_the_handler(self, client, as_user, make_user, promote):
        """404 here is the *success* signal for this test.

        An admin requeueing a job id that does not exist should get the
        handler's own "not found", which is only reachable once the gate has
        let them through. Contrast with the non-admin case above, which stops
        at 403 on the same URL.
        """
        as_user(promote(make_user()))
        assert client.post("/api/ops/jobs/999999/requeue").status_code == 404


class TestAnonymousIsUnauthenticated:
    @pytest.mark.parametrize("path", OPS_READ_ENDPOINTS)
    def test_401_not_403(self, path):
        """401, not 403, with no session at all.

        The two are not interchangeable and the distinction is actionable:
        401 means "authenticate and retry", 403 means "retrying will not
        help". Because `require_admin` layers on `current_user`, the missing
        session is caught first and the flag check is never reached.

        A bare client is used rather than the `as_user` fixture, since that
        fixture's whole job is to install the override this test needs absent.
        """
        anonymous = TestClient(app, raise_server_exceptions=False)
        assert anonymous.get(path).status_code == 401


class TestDefaultIsRestrictive:
    def test_new_accounts_are_not_admin(self, make_user):
        """The column's default decides who is privileged by accident.

        A nullable or true-defaulted flag would mean every account created
        before an operator thought about access control silently had it.
        """
        assert make_user().is_admin is False

    def test_flag_round_trips(self, make_user, promote):
        user = promote(make_user())
        db = SessionLocal()
        try:
            assert db.get(User, user.id).is_admin is True
        finally:
            db.close()
