from __future__ import annotations

import json

import httpx
import pytest

from director.comfy import ComfyClient, ComfyError


async def test_comfy_submit_requests_only_server_owned_metadata_previews() -> None:
    submitted: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        submitted.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"prompt_id": "fixed-id", "number": 1, "node_errors": {}},
        )

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    await client.submit(
        {"1": {"class_type": "UNETLoader", "inputs": {}}},
        "director-client",
        "fixed-id",
    )

    assert submitted == {
        "prompt": {"1": {"class_type": "UNETLoader", "inputs": {}}},
        "client_id": "director-client",
        "extra_data": {"preview_method": "latent2rgb"},
        "prompt_id": "fixed-id",
    }


async def test_comfy_submit_prompt_id_mismatch_atomically_cancels_actual_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/prompt":
            return httpx.Response(
                200,
                json={"prompt_id": "actual", "number": 1, "node_errors": {}},
            )
        return httpx.Response(200, json={"cancelled": True})

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(
        ComfyError, match="different prompt id.*atomically cancelled"
    ) as caught:
        await client.submit({}, "director-client", "requested")

    assert caught.value.detail["requested_prompt_id"] == "requested"
    assert caught.value.detail["actual_prompt_id"] == "actual"
    assert caught.value.detail["cleanup_response"] == {"cancelled": True}
    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/prompt"),
        ("POST", "/api/jobs/actual/cancel"),
    ]


@pytest.mark.parametrize(
    ("cleanup_status", "cleanup_body", "message"),
    [
        (404, {"error": "route not found"}, "no atomic cleanup endpoint"),
        (200, {"cancelled": False}, "atomic cleanup was not confirmed"),
        (200, {"cancelled": "yes"}, "invalid atomic cleanup response"),
        (500, {"error": "broken"}, "atomic cleanup errored"),
    ],
)
async def test_comfy_submit_prompt_id_mismatch_cleanup_fails_closed(
    cleanup_status: int,
    cleanup_body: dict,
    message: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/prompt":
            return httpx.Response(
                200,
                json={"prompt_id": "actual", "number": 1, "node_errors": {}},
            )
        return httpx.Response(cleanup_status, json=cleanup_body)

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(ComfyError, match=message) as caught:
        await client.submit({}, "director-client", "requested")

    assert "may still be queued or running" in str(caught.value)
    assert caught.value.detail["actual_prompt_id"] == "actual"
    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/prompt"),
        ("POST", "/api/jobs/actual/cancel"),
    ]


async def test_comfy_bulk_history_is_bounded_by_server_owned_query() -> None:
    request_url: httpx.URL | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_url
        request_url = request.url
        return httpx.Response(200, json={"prompt-1": {"status": {}}})

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    history = await client.history(max_items=256)

    assert set(history) == {"prompt-1"}
    assert request_url is not None
    assert request_url.path == "/history"
    assert dict(request_url.params) == {"max_items": "256"}


@pytest.mark.parametrize("endpoint", ["/queue", "/history/prompt-1"])
async def test_comfy_status_reads_reject_non_object_contracts(endpoint: str) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(ComfyError, match="returned an invalid object"):
        if endpoint == "/queue":
            await client.queue()
        else:
            await client.history("prompt-1")


async def test_comfy_cancel_prefers_the_native_atomic_jobs_endpoint() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"cancelled": True})

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    cancelled = await client.cancel("prompt-123")

    assert cancelled is True
    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/api/jobs/prompt-123/cancel")
    ]


async def test_comfy_cancel_fallback_refuses_racy_legacy_pending_delete() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/jobs/prompt-123/cancel":
            return httpx.Response(404, json={"error": "route not found"})
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(
                200,
                json={"queue_running": [], "queue_pending": [[7, "prompt-123"]]},
            )
        return httpx.Response(200, json={})

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(ComfyError, match="atomic job cancellation"):
        await client.cancel("prompt-123")

    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/api/jobs/prompt-123/cancel"),
        ("GET", "/queue"),
    ]


async def test_comfy_cancel_fallback_refuses_unsafe_legacy_running_interrupt() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/jobs/prompt-123/cancel":
            return httpx.Response(404, json={"error": "route not found"})
        if request.method == "GET" and request.url.path == "/queue":
            return httpx.Response(
                200,
                json={"queue_running": [[7, "prompt-123"]], "queue_pending": []},
            )
        return httpx.Response(200, json={})

    client = ComfyClient("http://comfy.test", transport=httpx.MockTransport(handler))

    with pytest.raises(ComfyError, match="atomic running-job cancellation"):
        await client.cancel("prompt-123")

    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/api/jobs/prompt-123/cancel"),
        ("GET", "/queue"),
    ]


async def test_comfy_cancel_fallback_does_nothing_when_prompt_is_absent() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/jobs/prompt-123/cancel":
            return httpx.Response(404, json={"error": "route not found"})
        return httpx.Response(
            200,
            json={"queue_running": [], "queue_pending": [[8, "another-prompt"]]},
        )

    client = ComfyClient("http://comfy.test", transport=httpx.MockTransport(handler))

    cancelled = await client.cancel("prompt-123")

    assert cancelled is False
    assert [(request.method, request.url.path) for request in requests] == [
        ("POST", "/api/jobs/prompt-123/cancel"),
        ("GET", "/queue"),
    ]


async def test_comfy_cancel_does_not_fallback_for_other_atomic_errors() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, json={"error": "broken"})

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(ComfyError, match="HTTP 500"):
        await client.cancel("prompt-123")

    assert len(requests) == 1
