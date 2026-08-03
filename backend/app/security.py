"""Password hashing, sessions, and the dependencies that enforce ownership.

Three things live here, in increasing order of how often they are got wrong.

**Hashing.** Argon2id, via argon2-cffi. Not SHA-256, not SHA-256 with a salt,
not bcrypt: a password hash needs to be deliberately slow and memory-hard, and
the fast hashes are fast for the attacker too.

**Sessions.** An opaque random id in an HttpOnly cookie, with the body in
Redis. The alternative -- a JWT -- is stateless right up to the first time a
session has to be killed early (logout everywhere, password change, stolen
laptop), at which point you build a server-side blocklist and have rebuilt
sessions with a larger cookie and worse ergonomics. Revocation here is a DEL.

**Authorization.** Authentication proves who; authorization proves allowed.
The gap between them is where IDOR lives, and the mitigation is that ownership
is a dependency rather than a line a handler has to remember to write.
"""

import logging
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Profile, User
from app.queues import get_redis

logger = logging.getLogger(__name__)

# Default parameters, which argon2-cffi keeps current with the RFC 9106
# recommendations. Tuning them is a deployment concern rather than a code one,
# and the encoded hash records whichever values produced it -- so raising the
# work factor later does not invalidate existing passwords.
_hasher = PasswordHasher()

_signer = TimestampSigner(settings.session_secret)

# Redis key prefix. Namespaced so session keys cannot collide with RQ's own
# keys in the same database.
_SESSION_PREFIX = "session:"

# A valid Argon2 hash of a throwaway value, used to spend the same work on a
# login for an address that does not exist as one that does. Without it,
# "unknown email" returns in microseconds and "wrong password" takes ~50ms,
# and that difference is a usable oracle for enumerating accounts. Computed
# once at import rather than per request.
_DUMMY_HASH = _hasher.hash("verify-against-this-when-no-user-exists")


def hash_password(password: str) -> str:
    """Return an Argon2id encoded hash for `password`."""
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Whether `password` matches `password_hash`.

    Uses the library's constant-time verify rather than `==`. A plain string
    comparison exits at the first differing byte, which leaks how much of a
    guess was right.
    """
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def verify_dummy_password(password: str) -> None:
    """Burn the same CPU a real verification would, then discard the result.

    Called on the no-such-user branch of login so the two paths take
    comparable time. See _DUMMY_HASH.
    """
    try:
        _hasher.verify(_DUMMY_HASH, password)
    except (VerifyMismatchError, InvalidHashError):
        pass


def create_session(user_id: int, response: Response) -> str:
    """Start a session for `user_id` and attach its cookie to `response`.

    The id is 256 bits from `secrets`, not `random` -- the latter is a
    Mersenne twister seeded predictably enough that observing a few outputs
    reveals the rest, which for session ids means forging them.
    """
    session_id = secrets.token_urlsafe(32)

    # `ex=` rather than setex, which redis-py deprecated in 2.6.12. The TTL is
    # set in the same command as the value on purpose: writing the key and
    # then expiring it separately leaves a window where a crash in between
    # produces a session that never expires.
    get_redis().set(
        f"{_SESSION_PREFIX}{session_id}",
        str(user_id),
        ex=settings.session_ttl_seconds,
    )

    response.set_cookie(
        key=settings.session_cookie_name,
        value=_signer.sign(session_id).decode(),
        max_age=settings.session_ttl_seconds,
        # Unreadable from JavaScript, so an XSS that can run script in the
        # page still cannot exfiltrate the session. This is the single
        # highest-value flag on the cookie.
        httponly=True,
        # HTTPS only. Off in development, where there is no TLS to use.
        secure=settings.session_cookie_secure,
        # Lax rather than Strict: the cookie is withheld on cross-site
        # subrequests (which is the CSRF-relevant case) but still sent when a
        # user follows a link into the app, so arriving from an external link
        # does not appear as a logout.
        samesite="lax",
        path="/",
    )

    return session_id


def destroy_session(request: Request, response: Response) -> None:
    """End the current session and clear its cookie.

    Deleting the Redis key is what actually logs the user out. Clearing the
    cookie is housekeeping: a client that kept a copy of the old value gets
    nothing for it, because the key it names is gone.
    """
    session_id = _read_session_id(request)
    if session_id is not None:
        get_redis().delete(f"{_SESSION_PREFIX}{session_id}")

    response.delete_cookie(
        key=settings.session_cookie_name,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        path="/",
    )


def _read_session_id(request: Request) -> str | None:
    """Extract and unsign the session id from the request cookie.

    Returns None for absent, tampered, and expired cookies alike. The caller
    has no reason to tell those apart -- all three mean "not authenticated" --
    and reporting which one it was tells an attacker whether a forged
    signature was close.
    """
    raw = request.cookies.get(settings.session_cookie_name)
    if not raw:
        return None

    try:
        return _signer.unsign(
            raw, max_age=settings.session_ttl_seconds
        ).decode()
    except (BadSignature, SignatureExpired):
        return None


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """The authenticated user, or 401.

    Redis is the authority. An unexpired, correctly-signed cookie whose key
    has been deleted is not a valid session -- that is exactly what makes
    logout and forced revocation take effect immediately rather than whenever
    the cookie happens to expire.
    """
    session_id = _read_session_id(request)
    if session_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    raw_user_id = get_redis().get(f"{_SESSION_PREFIX}{session_id}")
    if raw_user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    user = db.get(User, int(raw_user_id))
    if user is None:
        # The account was deleted while a session was live. Clean up the
        # dangling key so it stops being looked up on every request.
        get_redis().delete(f"{_SESSION_PREFIX}{session_id}")
        raise HTTPException(status_code=401, detail="Not authenticated")

    return user


def owned_profile(
    profile_id: int,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
) -> Profile:
    """The named profile, if this user owns it. Otherwise 404.

    Every profile-scoped endpoint takes this instead of loading by id, so the
    ownership predicate cannot be left out of one handler. Writing
    `db.get(Profile, profile_id)` and remembering to also check `user_id`
    works until the day someone adds an endpoint and forgets, which is the
    single most common vulnerability in applications shaped like this one.

    404 rather than 403 on purpose. 403 confirms the row exists, which turns
    the endpoint into an oracle for enumerating other people's profile ids.
    A user who does not own it should be unable to distinguish it from one
    that was never there.
    """
    profile = db.scalar(
        select(Profile).where(
            Profile.id == profile_id,
            Profile.user_id == user.id,
        )
    )
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

    return profile


def require_admin(user: User = Depends(current_user)) -> User:
    """The authenticated user, if they are an operator. Otherwise 403.

    **403 here, where `owned_profile` returns 404.** The two look like the
    same decision and are not. Returning 404 for a profile you do not own
    hides *whether that row exists*, because confirming existence is an
    enumeration oracle -- walk the ids and learn which are real. No
    equivalent secret exists here: `/api/ops/overview` is a fixed path
    published in the OpenAPI document, so a 404 would conceal nothing while
    telling a legitimate operator their deployment is missing an endpoint
    when the truth is their account is missing a flag.

    Layered on `current_user`, so an anonymous request still fails at 401
    before reaching this check. That distinction is the useful part for a
    client: 401 means "authenticate and retry", 403 means "retrying will not
    help".
    """
    if not user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="This endpoint requires operator access",
        )

    return user
