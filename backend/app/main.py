"""FastAPI application entry point.

Creates the app object, registers routers, and exposes a liveness probe.
Business logic does not live here -- this module wires things together.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.middleware import BodySizeLimitMiddleware
from app.routers import auth, batches, jobs, ops, profiles

app = FastAPI(
    title="FitCheck",
    description="Resume/JD matching pipeline with explainable scoring.",
    version="0.1.0",
)

# Caps request bodies at the ASGI layer, before any handler sees them. This
# bounds *input* size only -- it does not bound parse cost. A valid DOCX is a
# zip and can decompress far past the cap; that gets handled by worker job
# timeouts once parsing moves off the request path.
app.add_middleware(BodySizeLimitMiddleware)

# The frontend runs on a different origin from the API in development, and a
# session cookie is not sent cross-origin without this.
#
# allow_credentials=True is the part that matters, and it is why the origin
# list is explicit. Browsers reject "*" alongside credentials -- correctly,
# because the pairing would let any site on the internet issue authenticated
# requests with a visitor's cookie and read the responses.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# There is deliberately no exception handler for RequestBodyTooLarge here.
# An earlier version registered one, on the assumption that the exception
# raised mid-stream would reach it. It does not: FastAPI wraps form parsing
# in a blanket `except Exception` and rewrites anything it catches as a
# 400, and every upload this app takes is multipart. The middleware writes
# the 413 to the send channel itself, so it owns both paths end to end.


# Plain `def`, not `async def`. This handler does no I/O, so either would work
# here -- but the rest of this app uses synchronous SQLAlchemy, and a blocking
# DB call inside an `async def` handler blocks the whole event loop. Staying
# consistent with `def` means that mistake has no place to happen.
@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 if the process is up and serving."""
    return {"status": "ok"}


app.include_router(auth.router)
app.include_router(profiles.router)
app.include_router(jobs.router)
app.include_router(batches.router)
app.include_router(ops.router)
