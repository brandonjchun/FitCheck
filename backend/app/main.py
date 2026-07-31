"""FastAPI application entry point.

Creates the app object, registers routers, and exposes a liveness probe.
Business logic does not live here -- this module wires things together.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.middleware import BodySizeLimitMiddleware, RequestBodyTooLarge

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


# The middleware's Content-Length pre-check returns 413 on its own. But a
# client using chunked encoding sends no Content-Length, so the cap is
# enforced mid-stream by raising from the receive channel -- which surfaces
# as a 500 unless it is mapped back to a real response here.
@app.exception_handler(RequestBodyTooLarge)
async def request_body_too_large(
    request: Request, exc: RequestBodyTooLarge
) -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": str(exc)})


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
