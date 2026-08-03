"""Registration, login, logout, and session behaviour.

Unlike most of the suite these go through the real cookie flow rather than
overriding `current_user` -- signing, Redis, and expiry are the things under
test, so faking them would leave nothing. Needs Postgres and Redis.
"""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.db import SessionLocal
from app.main import app
from app.models import User
from app.queue import get_redis
from app.security import hash_password, verify_password

PASSWORD = "correct-horse-battery"


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def email() -> str:
    """A .dev address, not .local or .test.

    Those are IANA special-use domains and email-validator rejects them, so
    an address ending .local cannot be registered *or* logged in with -- a
    seed or fixture using one produces an account that exists and cannot be
    used.
    """
    return f"t{uuid4().hex[:12]}@fitcheck.dev"


@pytest.fixture(autouse=True)
def _cleanup():
    """Remove accounts these tests create.

    Registration goes through the API, so make_user's bookkeeping does not
    see them and cleanup has to be by address pattern.
    """
    yield
    db = SessionLocal()
    try:
        for user in db.query(User).filter(User.email.like("t%@fitcheck.dev")).all():
            db.delete(user)
        db.commit()
    finally:
        db.close()


class TestRegister:
    def test_creates_an_account_and_logs_in(self, client, email):
        response = client.post(
            "/api/auth/register", json={"email": email, "password": PASSWORD}
        )

        assert response.status_code == 201
        assert response.json()["email"] == email
        # Registering signs you in. Making someone retype credentials they
        # just chose is friction with no security value.
        assert settings.session_cookie_name in response.cookies

    def test_never_returns_the_password_hash(self, client, email):
        """The concrete reason ORM models are not reused as API responses."""
        response = client.post(
            "/api/auth/register", json={"email": email, "password": PASSWORD}
        )

        assert "password_hash" not in response.json()
        assert "password" not in response.json()

    def test_duplicate_email_returns_409(self, client, email):
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD})

        second = client.post(
            "/api/auth/register", json={"email": email, "password": PASSWORD}
        )

        assert second.status_code == 409

    def test_email_is_case_insensitive(self, client, email):
        """citext, so Bob@x.com and bob@x.com are one account.

        Enforced by the column type rather than by lowercasing in Python,
        which holds only until one code path forgets.
        """
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD})

        second = client.post(
            "/api/auth/register",
            json={"email": email.upper(), "password": PASSWORD},
        )

        assert second.status_code == 409

    def test_short_password_rejected(self, client, email):
        response = client.post(
            "/api/auth/register", json={"email": email, "password": "short"}
        )

        assert response.status_code == 422

    def test_malformed_email_rejected(self, client):
        response = client.post(
            "/api/auth/register", json={"email": "not-an-email", "password": PASSWORD}
        )

        assert response.status_code == 422


class TestLogin:
    @pytest.fixture
    def registered(self, client, email):
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
        client.cookies.clear()
        return email

    def test_correct_credentials_set_a_session(self, client, registered):
        response = client.post(
            "/api/auth/login", json={"email": registered, "password": PASSWORD}
        )

        assert response.status_code == 200
        assert settings.session_cookie_name in response.cookies

    def test_wrong_password_is_401(self, client, registered):
        response = client.post(
            "/api/auth/login", json={"email": registered, "password": "wrong-password"}
        )

        assert response.status_code == 401

    def test_unknown_and_wrong_password_are_indistinguishable(
        self, client, registered
    ):
        """Otherwise the login form is an account-existence oracle.

        Submit a list of addresses, note which ones say "no such user", and
        you have learned who is registered -- worth something alone and worth
        more alongside a password dump from somewhere else.
        """
        wrong_password = client.post(
            "/api/auth/login", json={"email": registered, "password": "wrong-password"}
        )
        no_such_user = client.post(
            "/api/auth/login",
            json={"email": "nobody@fitcheck.dev", "password": PASSWORD},
        )

        assert wrong_password.status_code == no_such_user.status_code == 401
        assert wrong_password.json()["detail"] == no_such_user.json()["detail"]

    def test_login_issues_a_new_session_id(self, client, registered):
        """Session fixation: the id must not be one the client supplied.

        An attacker who can plant a cookie before login would otherwise
        inherit the session once the victim authenticates.
        """
        client.cookies.set(settings.session_cookie_name, "planted-value")

        response = client.post(
            "/api/auth/login", json={"email": registered, "password": PASSWORD}
        )

        assert response.cookies[settings.session_cookie_name] != "planted-value"

    def test_cookie_is_httponly_and_lax(self, client, registered):
        """HttpOnly is the flag that survives an XSS."""
        response = client.post(
            "/api/auth/login", json={"email": registered, "password": PASSWORD}
        )

        header = response.headers["set-cookie"].lower()
        assert "httponly" in header
        assert "samesite=lax" in header


class TestMe:
    def test_returns_the_current_user(self, client, email):
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD})

        response = client.get("/api/auth/me")

        assert response.status_code == 200
        assert response.json()["email"] == email
        # A cached identity is how one user gets shown another's account.
        assert response.headers["cache-control"] == "no-store"

    def test_401_without_a_session(self, client):
        assert client.get("/api/auth/me").status_code == 401

    def test_401_with_a_tampered_cookie(self, client, email):
        """The signature's job: reject a forged id without touching Redis."""
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
        client.cookies.set(settings.session_cookie_name, "forged-session-id")

        assert client.get("/api/auth/me").status_code == 401


class TestLogout:
    def test_clears_the_session(self, client, email):
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
        assert client.get("/api/auth/me").status_code == 200

        assert client.post("/api/auth/logout").status_code == 204
        assert client.get("/api/auth/me").status_code == 401

    def test_revocation_is_server_side(self, client, email):
        """The property JWTs would not give without extra machinery.

        Replaying the exact cookie after logout fails, because the authority
        is the Redis key rather than anything the cookie itself asserts. A
        stateless token would still verify here.
        """
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
        stolen = client.cookies[settings.session_cookie_name]

        client.post("/api/auth/logout")

        replay = TestClient(app, raise_server_exceptions=False)
        replay.cookies.set(settings.session_cookie_name, stolen)

        assert replay.get("/api/auth/me").status_code == 401

    def test_logout_without_a_session_is_not_an_error(self, client):
        """The caller asked for "not logged in" and that is the end state."""
        assert client.post("/api/auth/logout").status_code == 204

    def test_session_key_is_deleted_from_redis(self, client, email):
        client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
        before = len(get_redis().keys("session:*"))

        client.post("/api/auth/logout")

        assert len(get_redis().keys("session:*")) == before - 1


class TestPasswordHashing:
    def test_hash_is_argon2id(self):
        assert hash_password(PASSWORD).startswith("$argon2id$")

    def test_same_password_hashes_differently(self):
        """Per-hash salt. Identical passwords must not produce identical rows.

        Without it, a stolen table shows at a glance which accounts share a
        password, and one cracked hash unlocks all of them.
        """
        assert hash_password(PASSWORD) != hash_password(PASSWORD)

    def test_verify_accepts_the_right_password(self):
        assert verify_password(hash_password(PASSWORD), PASSWORD) is True

    def test_verify_rejects_the_wrong_password(self):
        assert verify_password(hash_password(PASSWORD), "not-it") is False

    def test_verify_rejects_a_malformed_hash_instead_of_raising(self):
        """A corrupt column value must read as "wrong password", not a 500."""
        assert verify_password("not-a-hash", PASSWORD) is False
