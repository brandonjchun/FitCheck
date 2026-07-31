"""FastAPI application entry point.

Creates the app object, registers routers, and exposes a liveness probe.
Business logic does not live here -- this module wires things together.
"""

from fastapi import FastAPI

app = FastAPI(
    title="FitCheck",
    description="Resume/JD matching pipeline with explainable scoring.",
    version="0.1.0",
)


# Plain `def`, not `async def`. This handler does no I/O, so either would work
# here -- but the rest of this app uses synchronous SQLAlchemy, and a blocking
# DB call inside an `async def` handler blocks the whole event loop. Staying
# consistent with `def` means that mistake has no place to happen.
@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 if the process is up and serving."""
    return {"status": "ok"}


# Routers get registered here as they are written, e.g.
#   from app.routers import profiles
#   app.include_router(profiles.router)
