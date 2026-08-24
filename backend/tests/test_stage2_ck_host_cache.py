from __future__ import annotations

import asyncio


def _complete_object_info() -> dict[str, object]:
    return {
        "ModelAttentionBackend": {
            "input": {
                "required": {"attention": [["comfy kitchen attention"]]}
            }
        },
        "DirectorDeckRayInitializerAdvanced": {
            "input": {
                "required": {"XFuser_attention": [["COMFY_KITCHEN_INT8"]]}
            }
        },
    }


async def test_concurrent_requests_share_one_host_observation(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    object_info_calls = 0
    stats_calls = 0

    async def object_info(_class_types: tuple[str, ...]):
        nonlocal object_info_calls
        object_info_calls += 1
        started.set()
        await release.wait()
        return _complete_object_info()

    async def system_stats():
        nonlocal stats_calls
        stats_calls += 1
        return {"devices": [{"type": "cuda", "index": 0}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    first = asyncio.create_task(
        client.get("/api/capabilities/comfy-kitchen-attention")
    )
    await started.wait()
    second = asyncio.create_task(
        client.get("/api/capabilities/comfy-kitchen-attention")
    )
    await asyncio.sleep(0)
    release.set()
    responses = await asyncio.gather(first, second)

    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json() == responses[1].json()
    assert object_info_calls == 1
    assert stats_calls == 1


async def test_partial_observation_retries_only_unknown_fact(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    requested: list[tuple[str, ...]] = []
    stats_calls = 0

    async def object_info(class_types: tuple[str, ...]):
        requested.append(class_types)
        if class_types == (
            "ModelAttentionBackend",
            "DirectorDeckRayInitializerAdvanced",
        ):
            value = _complete_object_info()
            value["DirectorDeckRayInitializerAdvanced"] = {}
            return value
        assert class_types == ("DirectorDeckRayInitializerAdvanced",)
        return {
            "DirectorDeckRayInitializerAdvanced": _complete_object_info()[
                "DirectorDeckRayInitializerAdvanced"
            ]
        }

    async def system_stats():
        nonlocal stats_calls
        stats_calls += 1
        return {"devices": [{"type": "cuda", "index": 0}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    first = await client.get("/api/capabilities/comfy-kitchen-attention")
    second = await client.get("/api/capabilities/comfy-kitchen-attention")
    third = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert first.json()["state"] == "available"
    assert second.json()["state"] == "available"
    assert third.json() == second.json()
    assert requested == [
        ("ModelAttentionBackend", "DirectorDeckRayInitializerAdvanced"),
        ("DirectorDeckRayInitializerAdvanced",),
    ]
    assert stats_calls == 1


async def test_stats_failure_retries_only_device_inventory(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    object_info_calls = 0
    stats_calls = 0

    async def object_info(_class_types: tuple[str, ...]):
        nonlocal object_info_calls
        object_info_calls += 1
        return _complete_object_info()

    async def system_stats():
        nonlocal stats_calls
        stats_calls += 1
        if stats_calls == 1:
            raise OSError("stats unavailable")
        return {"devices": [{"type": "cuda", "index": 0}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    unknown = await client.get("/api/capabilities/comfy-kitchen-attention")
    recovered = await client.get("/api/capabilities/comfy-kitchen-attention")
    cached = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert unknown.json()["state"] == "unknown"
    assert recovered.json()["state"] == "available"
    assert cached.json() == recovered.json()
    assert object_info_calls == 1
    assert stats_calls == 2
