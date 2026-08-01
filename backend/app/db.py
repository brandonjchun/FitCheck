"""Database engine, session factory, and the per-request session dependency."""

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# The engine owns the connection pool. One per process, created at import
# time -- it is not a connection, it is a factory that manages a set of them.
engine = create_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    # Sends a cheap SELECT 1 before handing out a pooled connection. Docker
    # restarts and laptop sleeps silently kill connections the pool still
    # believes are live; without this you get a confusing OperationalError on
    # the first query after resume.
    pool_pre_ping=True,
)

# A factory that produces Session objects. Sessions are cheap and
# short-lived -- one per request, never shared across requests or threads.
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class every ORM model inherits from.

    SQLAlchemy collects table definitions on Base.metadata, which is what
    Alembic autogenerate compares against the live database to work out what
    changed.
    """


def get_db() -> Iterator[Session]:
    """Yield a database session for one request, then close it.

    A FastAPI dependency (spec section 4.5). A route declares
    `db: Session = Depends(get_db)` and receives the yielded Session.

    The `finally` is the point. Everything after the `yield` runs once the
    response has been sent -- including when the handler raised. Connections
    are a hard-limited resource (pool_size + max_overflow), so a session that
    is not returned on the error path leaks one permanently. Enough of those
    and every request blocks forever waiting for a connection nobody will
    give back, which looks exactly like the database being down.

    Transaction control is deliberately left to the handler: only the handler
    knows what a complete unit of work is. This function owns the session's
    lifetime, not its semantics.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
