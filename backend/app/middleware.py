"""ASGI middleware.

Written at the raw ASGI layer rather than as a Starlette BaseHTTPMiddleware,
because BaseHTTPMiddleware buffers the request body before handing it on --
which is exactly the thing a size cap exists to prevent.
"""

from uuid import uuid4

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

    Both layers answer 413 from this middleware rather than by raising into
    the application, because the application does not reliably let the
    exception through -- see guarded_send.
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
        exceeded = False
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # The flag is what the 413 is actually driven by. Raising
                    # is only to stop the read immediately -- see below for
                    # why the exception cannot be relied on by itself.
                    exceeded = True
                    raise RequestBodyTooLarge(
                        f"Request body exceeded {self.max_bytes} bytes"
                    )
            return message

        async def guarded_send(message: Message) -> None:
            # Discard whatever the app decided to answer with once the cap has
            # been crossed. FastAPI wraps `await request.form()` in a blanket
            # `except Exception` and turns anything it catches into
            # `400 "There was an error parsing the body"` -- including the
            # exception above. Since every upload here is multipart, the raise
            # alone would never reach the app's exception handler, and an
            # oversized upload would report as a parse error rather than as
            # what it is. Dropping that response lets the real 413 be sent.
            nonlocal response_started
            if exceeded and not response_started:
                return
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except RequestBodyTooLarge:
            # Reaches here only when nothing swallowed it -- a handler reading
            # the raw body rather than a form. Either way the response is
            # written below, so there is one 413 path instead of two.
            pass

        if exceeded and not response_started:
            await self._send_413(send, received)

    async def _send_413(self, send: Send, size: int) -> None:
        """Write a 413 straight to the send channel.

        Built by hand rather than with a Response class because layer 1 runs
        before the application exists in this call stack -- there is no
        request object to build a response against yet.

        `size` is the declared Content-Length on the pre-check path, and the
        number of bytes counted before aborting on the streaming path. Both
        tell the client the same useful thing: what it sent was over the cap.
        """
        body = (
            b'{"detail":"Upload too large. Limit is '
            + str(self.max_bytes).encode()
            + b' bytes, got '
            + str(size).encode()
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


class RequestContextMiddleware:
    """Bind a request id to the logging context for the life of one request.

    The point is correlation. Without it, three log lines from three modules
    handling the same request are indistinguishable from three lines handling
    three different requests -- and under any concurrency at all, they arrive
    interleaved. With it, the whole request is one filter away.

    **The id is accepted from the client when offered.** A load balancer or
    the frontend may already have stamped `X-Request-ID`, and generating a
    fresh one here would break the chain at exactly the boundary the header
    exists to cross. The value is length-capped and stripped of anything
    non-printable before use: it goes into log output, and an unbounded
    client-controlled string in a log file is how log injection works.

    **Cleared in a `finally`.** contextvars are reused across requests by the
    event loop, so a leaked id would be stamped on an unrelated request --
    a confidently wrong correlation, which is worse than none.

    Raw ASGI rather than BaseHTTPMiddleware for the reason in the module
    docstring: BaseHTTPMiddleware buffers the body, and it sits in front of
    the size cap.
    """

    # Long enough for a UUID or a trace id, short enough that it cannot be
    # used to write a paragraph into the logs.
    MAX_ID_LENGTH = 64

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from app.logging_setup import bind_context, clear_context

        request_id = self._incoming_id(scope) or uuid4().hex

        bind_context(
            request_id=request_id,
            method=scope.get("method", ""),
            path=scope.get("path", ""),
        )

        async def send_with_id(message: Message) -> None:
            # Echoed back so a caller can quote the id when reporting a
            # problem, which is the difference between "it broke" and a
            # one-query lookup.
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", request_id.encode("ascii", "ignore")))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            clear_context()

    def _incoming_id(self, scope: Scope) -> str | None:
        for name, value in scope.get("headers", []):
            if name == b"x-request-id":
                candidate = value.decode("latin-1", "ignore").strip()
                # Printable ASCII only. A newline here would let a client
                # forge whole log lines.
                cleaned = "".join(
                    c for c in candidate if c.isprintable() and c.isascii()
                )
                return cleaned[: self.MAX_ID_LENGTH] or None
        return None
