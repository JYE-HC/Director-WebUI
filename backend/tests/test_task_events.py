from __future__ import annotations

import asyncio
from typing import Any

from directordeck.app import PromptTerminalEvents


async def test_prompt_terminal_events_notify_wakes_registered_waiter():
    registry = PromptTerminalEvents()
    event = registry.register("prompt-1")
    assert not event.is_set()

    registry.notify("prompt-1")
    assert event.is_set()


async def test_prompt_terminal_events_unregister_forgets_waiter():
    registry = PromptTerminalEvents()
    first = registry.register("prompt-1")
    registry.unregister("prompt-1")

    second = registry.register("prompt-1")
    assert second is not first
    assert not second.is_set()


async def test_prompt_terminal_events_notify_without_waiter_is_noop():
    registry = PromptTerminalEvents()
    # Must not raise and must not leave a stale entry behind.
    registry.notify("missing")
    assert registry.register("missing") is not None


async def test_task_events_streams_refresh_on_change(client):
    # httpx ASGITransport buffers a response until the body completes, which an
    # SSE stream never does, and Starlette's pre-2.4 streaming path spawns a
    # receive loop that never observes a disconnect from a synthetic receive.
    # Drive the ASGI app directly with an ASGI spec_version >= 2.4 so the
    # streaming response sends body messages without a disconnect-listener.
    app = client.director_app
    app.state.task_change_event.set()

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.6"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/tasks/events",
        "raw_path": b"/api/tasks/events",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 0),
        "server": ("testserver", 80),
    }

    def body_text() -> str:
        return "".join(
            message.get("body", b"").decode("utf-8", "replace")
            for message in sent
            if message.get("type") == "http.response.body"
        )

    task = asyncio.create_task(app(scope, receive, send))
    try:
        deadline = asyncio.get_running_loop().time() + 5.0
        text = ""
        while asyncio.get_running_loop().time() < deadline:
            text = body_text()
            if "event: refresh" in text:
                break
            await asyncio.sleep(0.01)
        assert "event: refresh" in text
        assert "data: {}" in text

        start = next(
            message for message in sent
            if message.get("type") == "http.response.start"
        )
        headers = dict(start.get("headers", []))
        content_type = headers.get(b"content-type", b"").decode("latin-1")
        assert "text/event-stream" in content_type
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_task_events_disconnect_finishes_asgi_response(client):
    app = client.director_app
    app.state.task_change_event.set()
    received: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    await received.put({"type": "http.request", "body": b"", "more_body": False})
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def receive() -> dict[str, Any]:
        return await received.get()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/tasks/events",
        "raw_path": b"/api/tasks/events",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 0),
        "server": ("testserver", 80),
    }

    task = asyncio.create_task(app(scope, receive, send))
    try:
        deadline = asyncio.get_running_loop().time() + 1.0
        while (
            not any(message.get("type") == "http.response.start" for message in sent)
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.01)
        assert any(message.get("type") == "http.response.start" for message in sent)

        await received.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, timeout=1.0)
        assert task.done()
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def test_task_events_connection_lifetime_is_bounded(client):
    app = client.director_app
    original_lifetime = app.state.task_events_max_lifetime_seconds
    app.state.task_events_max_lifetime_seconds = 0.02
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.6"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/tasks/events",
        "raw_path": b"/api/tasks/events",
        "query_string": b"",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 0),
        "server": ("testserver", 80),
    }

    try:
        await asyncio.wait_for(app(scope, receive, send), timeout=1.0)
    finally:
        app.state.task_events_max_lifetime_seconds = original_lifetime

    start = next(
        message for message in sent if message.get("type") == "http.response.start"
    )
    assert start["status"] == 200
    assert sent[-1].get("type") == "http.response.body"
    assert sent[-1].get("more_body") is False
