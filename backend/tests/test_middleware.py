"""Tests for BodySizeLimitMiddleware in isolation.

Deliberately mounted on a throwaway app rather than app.main, so these tests
describe the middleware's behaviour and nothing else. The wiring into the real
application is tested separately in test_upload_limits.py.

The middleware is instantiated with a small max_bytes throughout. The cap
being 2 MB in production is a configuration detail; the logic under test is
"is the limit enforced on both paths", which reads better at 100 bytes.
"""

import json
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from app.middleware import MAX_UPLOAD_BYTES, BodySizeLimitMiddleware

LIMIT = 100


@pytest.fixture
def client() -> TestClient:
    """An app wired exactly the way main.py wires the real one.

    Note there is no exception handler for RequestBodyTooLarge -- the
    middleware answers 413 itself, and main.py registers none either.
    """
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=LIMIT)

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        body = await request.body()
        return {"received": len(body)}

    # raise_server_exceptions=False so an unhandled exception surfaces as the
    # 500 a real client would see, instead of being re-raised into the test.
    # Otherwise a regression that stops mapping RequestBodyTooLarge to a 413
    # shows up as an error rather than as the wrong status code.
    return TestClient(app, raise_server_exceptions=False)


class TestContentLengthPreCheck:
    """Layer 1: reject before the application is ever invoked."""

    def test_body_under_limit_passes(self, client: TestClient) -> None:
        response = client.post("/echo", content=b"x" * (LIMIT - 1))

        assert response.status_code == 200
        assert response.json() == {"received": LIMIT - 1}

    def test_body_exactly_at_limit_passes(self, client: TestClient) -> None:
        # The check is `> max_bytes`, not `>=`. A file of exactly the limit is
        # within it -- worth pinning so nobody "fixes" the comparison later.
        response = client.post("/echo", content=b"x" * LIMIT)

        assert response.status_code == 200
        assert response.json() == {"received": LIMIT}

    def test_one_byte_over_limit_rejected(self, client: TestClient) -> None:
        response = client.post("/echo", content=b"x" * (LIMIT + 1))

        assert response.status_code == 413

    def test_rejection_body_is_json_naming_both_numbers(
        self, client: TestClient
    ) -> None:
        response = client.post("/echo", content=b"x" * (LIMIT + 50))

        assert response.headers["content-type"] == "application/json"
        detail = response.json()["detail"]
        # The client needs to know the limit and what it sent to act on this.
        assert str(LIMIT) in detail
        assert str(LIMIT + 50) in detail


class TestStreamingCount:
    """Layer 2: count what actually arrives, for clients that send no
    Content-Length or send a false one."""

    def test_chunked_body_over_limit_rejected(self, client: TestClient) -> None:
        # Passing an iterator makes httpx use chunked transfer encoding, so no
        # Content-Length header exists and layer 1 cannot fire. This is the
        # path the exception handler in main.py exists for.
        def chunks() -> Any:
            for _ in range(10):
                yield b"x" * 50

        response = client.post("/echo", content=chunks())

        assert response.status_code == 413

    def test_chunked_body_under_limit_passes(self, client: TestClient) -> None:
        def chunks() -> Any:
            yield b"x" * 10
            yield b"x" * 10

        response = client.post("/echo", content=chunks())

        assert response.status_code == 200
        assert response.json() == {"received": 20}


def draining_app() -> tuple[Any, list[dict[str, Any]]]:
    """An inner ASGI app that reads the whole body and records its arguments."""
    calls: list[dict[str, Any]] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        calls.append({"scope": scope, "receive": receive, "send": send})
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break

    return app, calls


def swallowing_app(status: int = 400) -> Any:
    """An inner ASGI app that eats every exception and answers anyway.

    This is FastAPI's actual behaviour reduced to its essentials: form
    parsing is wrapped in a blanket `except Exception` that rewrites whatever
    it catches as a 400. Any middleware that signals by raising has to
    survive an application like this one.
    """

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        try:
            while True:
                message = await receive()
                if not message.get("more_body", False):
                    break
        except Exception:  # noqa: BLE001 -- deliberately mirroring FastAPI
            pass

        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b'{"detail":"There was an error parsing the body"}',
            }
        )

    return app


async def drive(
    inner: Any, scope: Scope, chunks: list[bytes], max_bytes: int = LIMIT
) -> list[Message]:
    """Run the middleware over a scripted body and collect what it sends."""
    middleware = BodySizeLimitMiddleware(inner, max_bytes=max_bytes)

    pending = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1}
        for index, chunk in enumerate(chunks)
    ]
    messages: list[Message] = []

    async def receive() -> Message:
        return pending.pop(0)

    async def send(message: Message) -> None:
        messages.append(message)

    await middleware(scope, receive, send)
    return messages


def http_scope(headers: list[tuple[bytes, bytes]] | None = None) -> Scope:
    return {
        "type": "http",
        "method": "POST",
        "path": "/echo",
        "headers": headers or [],
    }


def assert_is_413(messages: list[Message]) -> None:
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 413
    assert json.loads(messages[1]["body"])["detail"].startswith("Upload too large")


class TestAsgiEdgeCases:
    """Cases that cannot be produced through an HTTP client.

    httpx computes Content-Length itself and will not send a malformed or
    dishonest one, so these drive the middleware at the ASGI layer directly.
    """

    @pytest.mark.anyio
    async def test_lying_content_length_still_rejected(self) -> None:
        """A client declaring 10 bytes and sending 500 does not get through.

        This is the reason layer 2 exists. Trusting the header alone makes the
        cap advisory -- anyone willing to lie bypasses it entirely.
        """
        inner, _ = draining_app()

        messages = await drive(
            inner,
            http_scope([(b"content-length", b"10")]),
            [b"x" * 500],
        )

        assert_is_413(messages)

    @pytest.mark.anyio
    async def test_accumulates_across_chunks(self) -> None:
        """No single chunk exceeds the cap; their sum does."""
        inner, _ = draining_app()

        messages = await drive(inner, http_scope(), [b"x" * 60, b"x" * 60])

        assert_is_413(messages)

    @pytest.mark.anyio
    async def test_malformed_content_length_falls_through_to_counting(self) -> None:
        """A non-numeric Content-Length must not crash the pre-check.

        int() raises ValueError, the loop breaks, and enforcement falls to
        layer 2 -- which still rejects the oversized body.
        """
        inner, _ = draining_app()

        messages = await drive(
            inner,
            http_scope([(b"content-length", b"not-a-number")]),
            [b"x" * 500],
        )

        assert_is_413(messages)

    @pytest.mark.anyio
    async def test_body_under_limit_reaches_the_app_untouched(self) -> None:
        inner, calls = draining_app()

        messages = await drive(inner, http_scope(), [b"x" * 10])

        assert len(calls) == 1
        assert messages == []


class TestApplicationCannotMaskThe413:
    """The regression guard for the defect this suite was written to find.

    RequestBodyTooLarge is raised from inside the receive channel, which means
    it surfaces inside whatever code is reading the body. FastAPI reads the
    body inside a blanket `except Exception` and answers 400. Signalling by
    exception alone therefore loses the 413 on every multipart upload -- which
    is the only kind this API accepts.
    """

    @pytest.mark.anyio
    async def test_app_response_is_discarded_after_the_cap_is_crossed(self) -> None:
        messages = await drive(swallowing_app(400), http_scope(), [b"x" * 500])

        assert_is_413(messages)
        # Exactly one response: start + body, with nothing from the app.
        assert len(messages) == 2

    @pytest.mark.anyio
    async def test_app_response_is_kept_when_the_body_is_within_the_cap(self) -> None:
        """The suppression is conditional, not unconditional.

        A middleware that always discarded the app's response would pass the
        test above and break every successful upload.
        """
        messages = await drive(swallowing_app(400), http_scope(), [b"x" * 10])

        assert messages[0]["status"] == 400


class TestPreCheckShortCircuit:
    """Layer 1 answers without the application running at all."""

    @staticmethod
    async def _reject(headers: list[tuple[bytes, bytes]]) -> tuple[Any, list[Message]]:
        inner, calls = draining_app()
        middleware = BodySizeLimitMiddleware(inner, max_bytes=LIMIT)

        messages: list[Message] = []

        async def receive() -> Message:
            raise AssertionError("body must not be read after a 413")

        async def send(message: Message) -> None:
            messages.append(message)

        await middleware(http_scope(headers), receive, send)
        return calls, messages

    @pytest.mark.anyio
    async def test_413_is_sent_without_invoking_the_app(self) -> None:
        """An oversized upload costs no handler execution, no dependency
        resolution, and no database session."""
        calls, messages = await self._reject(
            [(b"content-length", str(LIMIT + 1).encode())]
        )

        assert calls == []
        assert_is_413(messages)

    @pytest.mark.anyio
    async def test_413_declares_an_accurate_content_length(self) -> None:
        """The response's own Content-Length must match its body.

        It is computed by hand rather than by a response class, and a wrong
        value makes the client hang waiting for bytes that never come.
        """
        _, messages = await self._reject([(b"content-length", b"999999")])

        headers = dict(messages[0]["headers"])
        assert int(headers[b"content-length"]) == len(messages[1]["body"])


class TestScopeFiltering:
    @pytest.mark.anyio
    async def test_non_http_scope_passes_through_untouched(self) -> None:
        """Lifespan and websocket scopes have no body to police.

        The inner app must receive the *original* receive callable, not the
        counting wrapper -- a lifespan message has no "body" key, and wrapping
        it would be pointless work on every startup and shutdown event.
        """
        inner, calls = draining_app()
        middleware = BodySizeLimitMiddleware(inner, max_bytes=LIMIT)

        async def receive() -> Message:
            return {"type": "lifespan.shutdown", "more_body": False}

        async def send(message: Message) -> None:
            pass

        await middleware({"type": "lifespan"}, receive, send)

        assert len(calls) == 1
        assert calls[0]["receive"] is receive
        assert calls[0]["send"] is send


def test_default_limit_is_two_megabytes() -> None:
    """The default is a product decision, not an accident.

    A text resume is well under 1 MB; 2 leaves room for embedded fonts and a
    photo. Pinning it means a change to the number is a deliberate edit to a
    failing test rather than a silent one.
    """
    assert MAX_UPLOAD_BYTES == 2 * 1024 * 1024
