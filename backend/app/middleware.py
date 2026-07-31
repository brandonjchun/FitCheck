"""ASGI middleware.

Written at the raw ASGI layer rather than as a Starlette BaseHTTPMiddleware,
because BaseHTTPMiddleware buffers the request body before handing it on --
which is exactly the thing a size cap exists to prevent.
"""

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# 2 MB. A text PDF resume is well under 1 MB; 2 leaves headroom for one with
# a photo or embedded fonts without letting anyone stream a movie at the
# parser. Move this to config.py once pydantic-settings is wired up.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


class RequestBodyTooLarge(Exception):
    """Raised mid-stream when a request body exceeds the configured cap."""


class BodySizeLimitMiddleware:
    """Reject request bodies larger than `max_bytes`.

    Two layers, because either alone is insufficient:

    1. Content-Length pre-check. Rejects before the app is ever invoked and
       before the body is read. Covers every well-behaved client -- browsers
       and HTTP libraries always set Content-Length on a file upload.

    2. Streaming byte count. A client can omit Content-Length entirely
       (chunked transfer encoding) or simply lie about it. So we also count
       what actually arrives and abort once the cap is crossed, rather than
       trusting a number the client supplied.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Lifespan and websocket scopes have no request body to police.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Layer 1: trust, but verify. Headers are a list of (bytes, bytes).
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break
                if declared > self.max_bytes:
                    await self._send_413(send, declared)
                    return
                break

        # Layer 2: count what actually arrives.
        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge(
                        f"Request body exceeded {self.max_bytes} bytes"
                    )
            return message

        await self.app(scope, counting_receive, send)

    async def _send_413(self, send: Send, declared: int) -> None:
        body = (
            b'{"detail":"Upload too large. Limit is '
            + str(self.max_bytes).encode()
            + b' bytes, got '
            + str(declared).encode()
            + b'."}'
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
