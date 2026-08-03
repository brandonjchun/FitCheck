"""Registration, login, logout, and the current-user probe."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.schemas import LoginRequest, RegisterRequest, UserResponse
from app.security import (
    create_session,
    current_user,
    destroy_session,
    hash_password,
    verify_dummy_password,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Returned for both "no such account" and "wrong password". Telling them apart
# turns the login form into an account-existence oracle: an attacker submits a
# list of addresses and learns which ones are registered, which is worth
# something on its own and worth more combined with a password dump from
# elsewhere.
_INVALID_CREDENTIALS = "Incorrect email or password"


@router.post("/register", response_model=UserResponse, status_code=201)
def register(
    payload: RegisterRequest, response: Response, db: Session = Depends(get_db)
) -> User:
    """Create an account and start a session for it.

    Registration necessarily reveals whether an address is taken -- there is
    no way to both refuse a duplicate and conceal that it exists. Login is
    where the disclosure actually matters, and that path stays silent.
    """
    user = User(
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    db.add(user)

    try:
        db.commit()
    except IntegrityError:
        # Lost a race against a concurrent registration for the same address,
        # or the address was already taken. The unique index is what decides
        # this -- a SELECT-then-INSERT check would let two requests both pass
        # and both insert.
        db.rollback()
        raise HTTPException(
            status_code=409, detail="An account with this email already exists"
        ) from None

    db.refresh(user)

    # Log in immediately. Making someone type the credentials they just chose
    # is friction with no security value.
    create_session(user.id, response)

    return user


@router.post("/login", response_model=UserResponse)
def login(
    payload: LoginRequest, response: Response, db: Session = Depends(get_db)
) -> User:
    """Exchange credentials for a session cookie."""
    # Compared case-insensitively. The column is citext, so this matches the
    # database's own notion of equality rather than layering a second,
    # different rule on top of it.
    user = db.scalar(
        select(User).where(func.lower(User.email) == payload.email.lower())
    )

    if user is None:
        # Spend the same work a real verification would, so the two branches
        # take comparable time. Returning here immediately would make an
        # unregistered address answer in microseconds while a registered one
        # takes ~50ms -- a difference that is trivially measurable.
        verify_dummy_password(payload.password)
        raise HTTPException(status_code=401, detail=_INVALID_CREDENTIALS)

    if not verify_password(user.password_hash, payload.password):
        raise HTTPException(status_code=401, detail=_INVALID_CREDENTIALS)

    # A new id per login, rather than reusing any session already present on
    # the request. Accepting a client-supplied id here is session fixation:
    # an attacker plants a known cookie, waits for the victim to authenticate,
    # and inherits the now-privileged session.
    create_session(user.id, response)

    return user


@router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> None:
    """End the current session.

    Deliberately not requiring `current_user`. Logging out with an already
    invalid session is not an error -- it is the state the caller was asking
    for, and answering 401 would leave a confused client apparently unable to
    log out.
    """
    destroy_session(request, response)


@router.get("/me", response_model=UserResponse)
def me(response: Response, user: User = Depends(current_user)) -> User:
    """Who the caller is. 401 when nobody.

    This is the client's auth state: a query, not something mirrored into
    application state that then has to be kept in sync. `no-store` because a
    cached identity is how one user is shown another's account.
    """
    response.headers["Cache-Control"] = "no-store"
    return user
