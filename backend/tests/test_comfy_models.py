from __future__ import annotations

import re

import httpx
import pytest

from directordeck.comfy import ComfyClient


def _director_raylight_initializer_info() -> dict[str, object]:
    return {
        "python_module": "custom_nodes.DirectorDeck-RayLight",
        "input": {
            "required": {
                "XFuser_attention": [
                    ["SAGE_ATTN", "COMFY_KITCHEN_INT8", "TORCH_FLASH"],
                    {"default": "TORCH_FLASH"},
                ]
            },
            "optional": {
                "driver_cleanup_policy": [
                    ["legacy_all", "ray_devices"],
                    {"default": "legacy_all"},
                ],
                "ram_cache_max_models": ["INT", {"default": 2, "min": 0}],
            },
        },
    }


async def test_capabilities_probes_native_atomic_cancel_without_queue_mutation() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/object_info":
            return httpx.Response(200, json={})
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"error": "old server"})

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()

    assert report["supports_cancel"] is False
    assert report["supported_modes"] == []
    assert report["native_timeline"]["modes"] == ["fl2va", "ref2va"]
    assert requests[-1][0] == "POST"
    assert re.fullmatch(
        r"/api/jobs/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/cancel",
        requests[-1][1],
    )
    assert all(path not in {"/queue", "/interrupt"} for _, path in requests)


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (401, {"cancelled": False}),
        (405, {"cancelled": False}),
        (500, {"cancelled": False}),
        (200, {"cancelled": "false"}),
        (200, {"cancelled": False, "unexpected": True}),
        (200, None),
    ],
)
async def test_capabilities_fail_closed_for_invalid_atomic_cancel_contract(
    status_code: int, payload: object
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json={})
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        if payload is None:
            return httpx.Response(status_code, content=b"not-json")
        return httpx.Response(status_code, json=payload)

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()

    assert report["connection"] == "online"
    assert report["supports_cancel"] is False


async def test_capabilities_accepts_the_exact_atomic_cancel_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json={})
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"cancelled": False})

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()

    assert report["supports_cancel"] is True


async def test_capabilities_keep_legacy_recipes_separate_from_timeline_families() -> None:
    available = {
        node: {"python_module": "test"}
        for node in ComfyClient.STANDARD_REQUIRED_NODES
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=available)
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"cancelled": False})

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()

    assert report["supported_modes"] == [
        "t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"
    ]
    assert report["native_timeline"] == {
        "supported": True,
        "modes": ["fl2va", "ref2va"],
        "continuity": False,
    }


async def test_continuity_capability_requires_optional_nodes_but_not_provenance() -> None:
    available = {
        node: {"python_module": "test"}
        for node in ComfyClient.STANDARD_REQUIRED_NODES
    }
    available.update(
        {
            node: {"python_module": module}
            for node, module in (
                (node, "custom_nodes.user-managed")
                for node in ComfyClient.CONTINUITY_REQUIRED_NODES
            )
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=available)
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"cancelled": False})

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()
    assert report["native_timeline"]["supported"] is True
    assert report["native_timeline"]["continuity"] is True
    assert report["missing_nodes"] == []

    available["MiniMaxH3AddGuide"]["python_module"] = "custom_nodes.lookalike"
    spoofed = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()
    assert spoofed["native_timeline"]["supported"] is True
    assert spoofed["native_timeline"]["continuity"] is True
    assert spoofed["missing_nodes"] == []


async def test_missing_continuity_node_does_not_disable_non_continuity_timeline() -> None:
    available = {
        node: {"python_module": "test"}
        for node in ComfyClient.STANDARD_REQUIRED_NODES
    }
    available.update(
        {
            node: {"python_module": "custom_nodes.user-managed"}
            for node in ComfyClient.CONTINUITY_REQUIRED_NODES
            if node != "TrimAudioDuration"
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=available)
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"cancelled": False})

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()
    assert report["native_timeline"]["supported"] is True
    assert report["native_timeline"]["continuity"] is False
    assert report["execution_backends"]["standard"]["available"] is True


async def test_raylight_without_lora_does_not_require_ray_lora_loader() -> None:
    available = {
        node: {"python_module": "test"}
        for node in (
            *ComfyClient.STANDARD_REQUIRED_NODES,
            *ComfyClient.RAYLIGHT_REQUIRED_NODES,
        )
    }
    available["DirectorDeckRayInitializerAdvanced"] = _director_raylight_initializer_info()
    assert "DirectorDeckRayLoraLoader" not in available

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=available)
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"cancelled": False})

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()

    assert report["execution_backends"]["raylight"] == {
        "available": True,
        "missing_nodes": [],
        "contract_issues": [],
        "conditional_requirements": {
            "lora": {
                "available": False,
                "missing_nodes": ["DirectorDeckRayLoraLoader"],
            }
        },
    }
    assert "DirectorDeckRayLoraLoader" not in report["missing_nodes"]


async def test_raylight_capability_reports_present_conditional_lora_loader() -> None:
    available = {
        node: {"python_module": "test"}
        for node in (
            *ComfyClient.STANDARD_REQUIRED_NODES,
            *ComfyClient.RAYLIGHT_REQUIRED_NODES,
            *ComfyClient.RAYLIGHT_LORA_REQUIRED_NODES,
        )
    }
    available["DirectorDeckRayInitializerAdvanced"] = _director_raylight_initializer_info()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=available)
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"cancelled": False})

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()

    assert report["execution_backends"]["raylight"]["available"] is True
    assert report["execution_backends"]["raylight"]["conditional_requirements"] == {
        "lora": {"available": True, "missing_nodes": []}
    }


async def test_raylight_capability_reports_initializer_schema_as_advisory() -> None:
    available = {
        node: {"python_module": "test"}
        for node in ComfyClient.RAYLIGHT_REQUIRED_NODES
    }
    available["DirectorDeckRayInitializerAdvanced"] = {
        "python_module": "custom_nodes.DirectorDeck-RayLight",
        "input": {
            "required": {
                "XFuser_attention": [
                    ["SAGE_ATTN", "TORCH_FLASH"],
                    {"default": "TORCH_FLASH"},
                ]
            },
            "optional": {},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return httpx.Response(200, json=available)
        if request.url.path == "/features":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"cancelled": False})

    report = await ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    ).capabilities()

    raylight = report["execution_backends"]["raylight"]
    assert raylight["available"] is True
    assert raylight["missing_nodes"] == []
    assert raylight["contract_issues"] == [
        "XFuser_attention must offer COMFY_KITCHEN_INT8 and TORCH_FLASH",
        "driver_cleanup_policy must offer legacy_all and ray_devices",
        "ram_cache_max_models must be optional INT with default 2 and min 0",
    ]


def test_raylight_base_and_conditional_lora_node_sets_are_disjoint() -> None:
    assert "DirectorDeckRayLoraLoader" not in ComfyClient.RAYLIGHT_REQUIRED_NODES
    assert ComfyClient.RAYLIGHT_LORA_REQUIRED_NODES == ("DirectorDeckRayLoraLoader",)


async def test_both_diffusion_slots_receive_the_complete_comfy_inventory() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        payloads = {
            "/models/diffusion_models": [
                "z_ref2va.safetensors",
                "generic_video_model.safetensors",
                "a_fl2va.safetensors",
            ],
            "/models/text_encoders": ["encoder.safetensors"],
            "/models/vae": ["video_vae.safetensors", "audio_vae.safetensors"],
            "/models/loras": ["z_style.safetensors", "a_turbo.safetensors"],
        }
        return httpx.Response(200, json=payloads[request.url.path])

    client = ComfyClient(
        "http://comfy.test", transport=httpx.MockTransport(handler)
    )

    inventory = await client.models()

    expected_diffusion = [
        "a_fl2va.safetensors",
        "generic_video_model.safetensors",
        "z_ref2va.safetensors",
    ]
    assert inventory["fl2va"] == expected_diffusion
    assert inventory["ref2va"] == expected_diffusion
    assert inventory["loras"] == ["a_turbo.safetensors", "z_style.safetensors"]
    assert requests == [
        ("GET", "/models/diffusion_models"),
        ("GET", "/models/text_encoders"),
        ("GET", "/models/vae"),
        ("GET", "/models/loras"),
    ]
