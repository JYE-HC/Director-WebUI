from __future__ import annotations

import asyncio

import httpx
import pytest
from starlette.requests import ClientDisconnect

from directordeck.app import _media_response, _proxy_comfy_media
from directordeck.comfy import ComfyClient, ComfyMediaStream


def _upstream(content: bytes = b"0123456789") -> httpx.Response:
    return httpx.Response(
        200,
        content=content,
        headers={"content-type": "video/mp4"},
        request=httpx.Request("GET", "http://comfy.test/view"),
    )


def test_media_response_supports_explicit_and_open_byte_ranges() -> None:
    explicit = _media_response(
        _upstream(), filename="take.mp4", byte_range="bytes=2-5"
    )
    assert explicit.status_code == 206
    assert explicit.body == b"2345"
    assert explicit.headers["accept-ranges"] == "bytes"
    assert explicit.headers["content-range"] == "bytes 2-5/10"
    assert explicit.headers["content-length"] == "4"

    open_ended = _media_response(
        _upstream(), filename="take.mp4", byte_range="bytes=7-"
    )
    assert open_ended.status_code == 206
    assert open_ended.body == b"789"
    assert open_ended.headers["content-range"] == "bytes 7-9/10"


def test_media_response_supports_suffix_ranges_and_rejects_invalid_ranges() -> None:
    suffix = _media_response(
        _upstream(), filename="take.mp4", byte_range="bytes=-3"
    )
    assert suffix.status_code == 206
    assert suffix.body == b"789"
    assert suffix.headers["content-range"] == "bytes 7-9/10"

    for value in ("items=0-1", "bytes=1-2,4-5", "bytes=10-", "bytes=5-2"):
        rejected = _media_response(
            _upstream(), filename="take.mp4", byte_range=value
        )
        assert rejected.status_code == 416
        assert rejected.body == b""
        assert rejected.headers["content-range"] == "bytes */10"
        assert rejected.headers["accept-ranges"] == "bytes"

    huge = _media_response(
        _upstream(), filename="take.mp4", byte_range=f"bytes={'9' * 5000}-"
    )
    assert huge.status_code == 416


async def test_comfy_media_proxy_forwards_range_and_streams_partial_body() -> None:
    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("range"))
        return httpx.Response(
            206,
            content=b"2345",
            headers={
                "content-type": "video/mp4",
                "content-range": "bytes 2-5/10",
                "content-length": "4",
            },
        )

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )
    response = await _proxy_comfy_media(
        client,
        {"filename": "take.mp4", "subfolder": "", "type": "output"},
        filename="take.mp4",
        byte_range="bytes=2-5",
    )
    body = b"".join([chunk async for chunk in response.body_iterator])

    assert seen_range == ["bytes=2-5"]
    assert response.status_code == 206
    assert response.headers["content-range"] == "bytes 2-5/10"
    assert body == b"2345"


async def test_comfy_media_proxy_rejects_huge_range_without_upstream_io() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b"unexpected")

    response = await _proxy_comfy_media(
        ComfyClient("http://comfy.test", transport=httpx.MockTransport(handler)),
        {"filename": "take.mp4", "subfolder": "", "type": "output"},
        filename="take.mp4",
        byte_range=f"bytes={'9' * 5000}-",
    )

    assert response.status_code == 416
    assert calls == 0


class _TrackedBody(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        for chunk in (b"first", b"second"):
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _TrackedStreamClient:
    def __init__(self) -> None:
        self.body = _TrackedBody()
        self.http = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200))
        )

    async def view_stream(
        self,
        _params: dict[str, str],
        *,
        byte_range: str | None = None,
    ) -> ComfyMediaStream:
        del byte_range
        return ComfyMediaStream(
            response=httpx.Response(
                200,
                stream=self.body,
                headers={"content-type": "video/mp4"},
            ),
            client=self.http,
        )


async def test_stream_closes_after_asgi_23_disconnect_following_first_chunk() -> None:
    client = _TrackedStreamClient()
    response = await _proxy_comfy_media(
        client,  # type: ignore[arg-type]
        {"filename": "take.mp4", "subfolder": "", "type": "output"},
        filename="take.mp4",
        byte_range=None,
    )
    first_chunk_sent = asyncio.Event()

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk_sent.set()
            # Keep the streaming task at a cancellation point until the ASGI
            # 2.3 disconnect watcher observes the simulated browser close.
            await asyncio.sleep(1)

    async def receive() -> dict:
        await first_chunk_sent.wait()
        return {"type": "http.disconnect"}

    await response(
        {"type": "http", "asgi": {"spec_version": "2.3"}},
        receive,
        send,
    )

    assert client.body.closed is True
    assert client.http.is_closed is True


async def test_stream_closes_after_asgi_24_send_failure() -> None:
    client = _TrackedStreamClient()
    response = await _proxy_comfy_media(
        client,  # type: ignore[arg-type]
        {"filename": "take.mp4", "subfolder": "", "type": "output"},
        filename="take.mp4",
        byte_range=None,
    )

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            raise OSError("downstream socket closed")

    async def receive() -> dict:
        return {"type": "http.request"}

    with pytest.raises(ClientDisconnect):
        await response(
            {"type": "http", "asgi": {"spec_version": "2.4"}},
            receive,
            send,
        )

    assert client.body.closed is True
    assert client.http.is_closed is True
