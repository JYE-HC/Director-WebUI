from __future__ import annotations

import httpx
import pytest

from directordeck.schemas import RuntimeSettingsV3


async def _standard_ck_object_info(_class_types: tuple[str, ...]):
    return {
        "ModelAttentionBackend": {
            "input": {
                "required": {"attention": [["comfy kitchen attention"]]}
            }
        }
    }


def _settings(client, *, multi_gpu: bool, device: str = "gpu:0") -> None:
    database = client.director_app.state.database
    current, authority = database.get_settings_authority()
    document = current.model_dump(mode="json")
    document["multi_gpu_enabled"] = multi_gpu
    document["placement"]["fl2va"]["device"] = device
    document["placement"]["ref2va"]["device"] = device
    database.put_settings(
        RuntimeSettingsV3.model_validate(document),
        expected_authority_token=authority,
        schema_version=3,
    )


async def test_standard_ck_capability_is_active_cached_and_needs_no_authority_token(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False)
    calls = 0

    async def object_info(class_types: tuple[str, ...]):
        nonlocal calls
        calls += 1
        assert class_types == (
            "ModelAttentionBackend",
            "DirectorDeckRayInitializerAdvanced",
        )
        return {
            "ModelAttentionBackend": {
                "input": {
                    "required": {
                        "attention": [["comfy kitchen attention"]]
                    }
                }
            }
        }

    async def system_stats():
        return {"devices": [{"type": "cuda", "index": 0}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    first = await client.get("/api/capabilities/comfy-kitchen-attention")
    second = await client.get("/api/capabilities/comfy-kitchen-attention")
    monkeypatch.setattr("directordeck.app.time.monotonic", lambda: 1_000_000_000.0)
    after_old_ttl = await client.get("/api/capabilities/comfy-kitchen-attention")
    database = client.director_app.state.database
    current, authority = database.get_settings_authority()
    unrelated = current.model_copy(update={"client_id": "another-client"})
    database.put_settings(
        unrelated,
        expected_authority_token=authority,
        schema_version=3,
    )
    after_unrelated_settings_write = await client.get(
        "/api/capabilities/comfy-kitchen-attention"
    )

    assert first.status_code == 200, first.text
    assert first.json() == second.json()
    assert first.json() == after_old_ttl.json()
    assert first.json() == after_unrelated_settings_write.json()
    assert first.json()["backend"] == "standard"
    assert first.json()["state"] == "available"
    assert first.json()["reasons"] == []
    assert calls == 1


@pytest.mark.parametrize("backend", ("xpu", "mps"))
async def test_standard_ck_uses_observed_device_backend(
    client,
    fake_comfy,
    monkeypatch,
    backend: str,
) -> None:
    _settings(client, multi_gpu=False)

    async def object_info(_class_types: tuple[str, ...]):
        return {
            "ModelAttentionBackend": {
                "input": {
                    "required": {
                        "attention": [["comfy kitchen attention"]]
                    }
                }
            }
        }

    async def system_stats():
        return {"devices": [{"type": backend, "index": 0}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    response = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "unavailable"
    assert response.json()["reasons"][0]["code"] == "target_device_not_cuda"


async def test_standard_ck_default_device_uses_observed_primary_device(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False, device="default")

    async def object_info(_class_types: tuple[str, ...]):
        return {
            "ModelAttentionBackend": {
                "input": {
                    "required": {
                        "attention": [["comfy kitchen attention"]]
                    }
                }
            }
        }

    async def system_stats():
        return {"devices": [{"type": "cuda", "index": 0}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    response = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "available"
    assert response.json()["reasons"] == []


@pytest.mark.parametrize(
    ("backend", "index"),
    (("cpu", None), ("xpu", 0), ("mps", None)),
)
async def test_standard_ck_default_device_reports_known_non_cuda_primary(
    client,
    fake_comfy,
    monkeypatch,
    backend: str,
    index: int | None,
) -> None:
    _settings(client, multi_gpu=False, device="default")

    async def object_info(_class_types: tuple[str, ...]):
        return {
            "ModelAttentionBackend": {
                "input": {
                    "required": {
                        "attention": [["comfy kitchen attention"]]
                    }
                }
            }
        }

    async def system_stats():
        return {"devices": [{"type": backend, "index": index}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    response = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "unavailable"
    assert response.json()["reasons"][0]["code"] == "target_device_not_cuda"


async def test_standard_ck_default_device_uses_primary_order_not_lowest_index(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False, device="default")

    async def object_info(_class_types: tuple[str, ...]):
        return {
            "ModelAttentionBackend": {
                "input": {
                    "required": {
                        "attention": [["comfy kitchen attention"]]
                    }
                }
            }
        }

    async def system_stats():
        return {
            "devices": [
                {"type": "cuda", "index": 1},
                {"type": "xpu", "index": 0},
            ]
        }

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    response = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "available"


async def test_standard_ck_default_device_stays_advisory_when_stats_fail(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False, device="default")

    async def object_info(_class_types: tuple[str, ...]):
        return {
            "ModelAttentionBackend": {
                "input": {
                    "required": {
                        "attention": [["comfy kitchen attention"]]
                    }
                }
            }
        }

    async def system_stats():
        raise httpx.ConnectError("stats unavailable")

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    response = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "unknown"
    assert response.json()["reasons"][0]["code"] == "target_device_not_cuda"


async def test_standard_ck_known_non_cuda_family_wins_over_unknown_family(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False, device="default")
    database = client.director_app.state.database
    current, authority = database.get_settings_authority()
    document = current.model_dump(mode="json")
    document["placement"]["ref2va"]["device"] = "cpu"
    database.put_settings(
        RuntimeSettingsV3.model_validate(document),
        expected_authority_token=authority,
        schema_version=3,
    )

    async def system_stats():
        raise httpx.ConnectError("stats unavailable")

    monkeypatch.setattr(
        fake_comfy,
        "object_info",
        _standard_ck_object_info,
        raising=False,
    )
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    response = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "unavailable"
    assert response.json()["reasons"][0]["code"] == "target_device_not_cuda"


async def test_standard_ck_explicit_gpu_uses_reported_index_not_primary_order(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False, device="gpu:0")

    async def system_stats():
        return {
            "devices": [
                {"type": "cuda", "index": 1},
                {"type": "xpu", "index": 0},
            ]
        }

    monkeypatch.setattr(
        fake_comfy,
        "object_info",
        _standard_ck_object_info,
        raising=False,
    )
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    response = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "unavailable"
    assert response.json()["reasons"][0]["code"] == "target_device_not_cuda"


async def test_standard_ck_rejects_partially_malformed_device_inventory(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False, device="default")

    async def system_stats():
        return {
            "devices": [
                {"type": "cuda", "index": 1},
                {"type": "cuda", "index": "invalid"},
            ]
        }

    monkeypatch.setattr(
        fake_comfy,
        "object_info",
        _standard_ck_object_info,
        raising=False,
    )
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    response = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert response.status_code == 200, response.text
    assert response.json()["state"] == "unknown"
    assert response.json()["reasons"][0]["code"] == "target_device_not_cuda"


async def test_standard_ck_only_checks_requested_project_families(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False)
    database = client.director_app.state.database
    current, authority = database.get_settings_authority()
    document = current.model_dump(mode="json")
    document["placement"]["ref2va"]["device"] = "cpu"
    database.put_settings(
        RuntimeSettingsV3.model_validate(document),
        expected_authority_token=authority,
        schema_version=3,
    )

    calls = 0

    async def object_info(_class_types: tuple[str, ...]):
        nonlocal calls
        calls += 1
        return {
            "ModelAttentionBackend": {
                "input": {
                    "required": {
                        "attention": [["comfy kitchen attention"]]
                    }
                }
            }
        }

    async def system_stats():
        return {"devices": [{"type": "cuda", "index": 0}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)

    fl2va = await client.get(
        "/api/capabilities/comfy-kitchen-attention?family=fl2va"
    )
    ref2va = await client.get(
        "/api/capabilities/comfy-kitchen-attention?family=ref2va"
    )

    assert fl2va.status_code == 200, fl2va.text
    assert fl2va.json()["state"] == "available"
    assert ref2va.status_code == 200, ref2va.text
    assert ref2va.json()["state"] == "unavailable"
    assert ref2va.json()["reasons"][0]["code"] == "target_device_not_cuda"
    assert fl2va.json()["context_revision"] != ref2va.json()["context_revision"]
    assert calls == 1


async def test_runtime_path_change_projects_cached_host_without_reprobe(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    calls = 0

    async def object_info(class_types: tuple[str, ...]):
        nonlocal calls
        calls += 1
        assert class_types == (
            "ModelAttentionBackend",
            "DirectorDeckRayInitializerAdvanced",
        )
        return {
            "ModelAttentionBackend": {
                "input": {
                    "required": {
                        "attention": [["comfy kitchen attention"]]
                    }
                }
            },
            "DirectorDeckRayInitializerAdvanced": {
                "input": {
                    "required": {
                        "XFuser_attention": [["COMFY_KITCHEN_INT8"]]
                    }
                }
            }
        }

    async def system_stats():
        return {"devices": [{"type": "cuda", "index": 0}]}

    monkeypatch.setattr(fake_comfy, "object_info", object_info, raising=False)
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)
    _settings(client, multi_gpu=False)

    standard = await client.get("/api/capabilities/comfy-kitchen-attention")
    _settings(client, multi_gpu=True)
    raylight = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert standard.status_code == 200, standard.text
    assert standard.json()["backend"] == "standard"
    assert raylight.status_code == 200, raylight.text
    assert raylight.json()["backend"] == "raylight"
    assert raylight.json()["state"] == "available"
    assert calls == 1


async def test_definite_absence_is_unavailable_but_transport_failure_is_unknown(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _settings(client, multi_gpu=False)

    calls = 0

    async def recover_after_failure(_class_types: tuple[str, ...]):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("offline")
        return {}

    async def system_stats():
        return {"devices": [{"type": "cuda", "index": 0}]}

    monkeypatch.setattr(
        fake_comfy,
        "object_info",
        recover_after_failure,
        raising=False,
    )
    monkeypatch.setattr(fake_comfy, "system_stats", system_stats)
    unknown = await client.get("/api/capabilities/comfy-kitchen-attention")
    unavailable = await client.get("/api/capabilities/comfy-kitchen-attention")
    cached = await client.get("/api/capabilities/comfy-kitchen-attention")

    assert unknown.status_code == 200, unknown.text
    assert unknown.json()["state"] == "unknown"
    assert unknown.json()["reasons"][0]["code"] == "host_not_connected"
    assert unavailable.json()["state"] == "unavailable"
    assert unavailable.json()["reasons"][0]["code"] == "standard_ck_node_not_observed"
    assert cached.json() == unavailable.json()
    assert calls == 2
