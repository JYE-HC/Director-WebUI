from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from dataclasses import dataclass

import pytest

from directordeck.capabilities import feature_catalog_etag, quote_feature_catalog_etag
from directordeck.comfy import ComfyError
from directordeck.workflow.contracts import HostCapabilitySnapshot
from directordeck.workflow.interpreters import V4BuiltinInterpreter

from . import extensible_workflow_v0_fixture_builder as fixture_builder
from .conftest import (
    adapt_legacy_workflow_requests,
    legacy_settings_document,
    save_legacy_settings_document,
    v5_timeline_document,
)


@pytest.fixture(autouse=True)
def _stage6_v5_request_adapter(client, monkeypatch) -> None:
    adapt_legacy_workflow_requests(client, monkeypatch)


@dataclass
class _MutableHostCapabilityProvider:
    current: HostCapabilitySnapshot
    calls: int = 0

    def snapshot(self) -> HostCapabilitySnapshot:
        self.calls += 1
        return self.current


def _v4_t2v_request() -> dict[str, object]:
    return {"config": fixture_builder._draft("t2v").model_dump(mode="json")}


def _without_node(
    snapshot: HostCapabilitySnapshot,
    class_type: str,
) -> HostCapabilitySnapshot:
    document = snapshot.model_dump(mode="json")
    document["node_registry"].pop(class_type)
    document["object_info_slices"].pop(class_type)
    return HostCapabilitySnapshot.model_validate_json(json.dumps(document))


def _with_module_fingerprint_drift(
    snapshot: HostCapabilitySnapshot,
    class_type: str,
) -> HostCapabilitySnapshot:
    document = snapshot.model_dump(mode="json")
    module = document["node_registry"][class_type]
    document["module_fingerprints"][module] = "sha256:" + "f" * 64
    return HostCapabilitySnapshot.model_validate_json(json.dumps(document))


def _without_media_tool(
    snapshot: HostCapabilitySnapshot,
    tool: str,
) -> HostCapabilitySnapshot:
    document = snapshot.model_dump(mode="json")
    document["media_tool_status"][tool] = {
        "available": False,
        "version": None,
    }
    return HostCapabilitySnapshot.model_validate_json(json.dumps(document))


def _with_gpu_backend(
    snapshot: HostCapabilitySnapshot,
    backend: str,
    *,
    count: int = 1,
) -> HostCapabilitySnapshot:
    document = snapshot.model_dump(mode="json")
    document["gpu_inventory"] = [
        {
            "logical_index": index,
            "backend": backend,
            "total_memory_mb": 16_384,
        }
        for index in range(count)
    ]
    return HostCapabilitySnapshot.model_validate_json(json.dumps(document))


def _reason_semantics(reason: dict[str, object]) -> tuple[object, ...]:
    return tuple(
        reason[key]
        for key in ("code", "rule", "message", "remediation", "safe_details")
    )


def _provider_snapshot(client) -> HostCapabilitySnapshot:
    provider = client.director_app.state.host_capability_provider
    assert provider is not None
    return provider.snapshot()


async def _enable_unmapped_standard_lora(client, filename: str = "style.safetensors") -> None:
    settings = await legacy_settings_document(client)
    binding = settings["models"]["fl2va"]
    binding["lora_name"] = filename
    binding["lora_strength"] = 1.0
    binding["standard_lora_loader_override"] = None
    response = await save_legacy_settings_document(client, settings)
    assert response.status_code == 200, response.text


async def _enable_mapped_standard_lora(
    client,
    *,
    filename: str = "style.safetensors",
    adapter_id: str = "model_only",
) -> None:
    settings = await legacy_settings_document(client)
    binding = settings["models"]["fl2va"]
    binding["lora_name"] = filename
    binding["lora_strength"] = 1.0
    binding["standard_lora_loader_override"] = {
        "loader": adapter_id,
        "model_filename": binding["filename"],
        "lora_name": filename,
    }
    response = await save_legacy_settings_document(client, settings)
    assert response.status_code == 200, response.text


def _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy) -> None:
    async def forbidden_capabilities() -> dict[str, object]:
        raise AssertionError(
            "Stage-5 preflight/compile must not call the legacy capability "
            "transport or its cancellation probe"
        )

    monkeypatch.setattr(fake_comfy, "capabilities", forbidden_capabilities)


def _error_codes(document: dict[str, object]) -> set[str]:
    errors = document.get("errors")
    assert isinstance(errors, list)
    return {
        str(error["code"])
        for error in errors
        if isinstance(error, dict) and "code" in error
    }


async def test_feature_catalog_has_strong_stable_etag_and_304_is_bodyless(
    client,
    fake_comfy,
) -> None:
    first = await client.get("/api/features/catalog")

    assert first.status_code == 200, first.text
    catalog = first.json()
    etag = first.headers["etag"]
    assert re.fullmatch(r'"sha256:[0-9a-f]{64}"', etag)
    assert not etag.startswith("W/")
    assert etag == quote_feature_catalog_etag(
        feature_catalog_etag(
            template_bundle_version=catalog["template_bundle_version"],
            host_capability_revision=catalog["host_capability_revision"],
        )
    )

    database = client.director_app.state.database
    database.put_raylight_runtime_state(
        {
            "version": 2,
            "epoch": 9,
            "current": {
                "family": "fl2va",
                "runtime_namespace": "stage5-catalog-e9",
            },
            "tail_prompt_id": "stage5-busy-prompt",
            "tail_action": "ray_unit",
            "tainted": True,
        }
    )
    ray_before = deepcopy(database.get_raylight_runtime_state())
    fake_comfy.running = [[0, "stage5-running-prompt"]]
    fake_comfy.pending = [[1, "stage5-pending-prompt"]]
    queue_before = (deepcopy(fake_comfy.running), deepcopy(fake_comfy.pending))
    busy_tail = asyncio.get_running_loop().create_future()
    client.director_app.state.submission_tails["stage5-busy"] = busy_tail

    try:
        busy = await client.get("/api/features/catalog")
        cached = await client.get(
            "/api/features/catalog",
            headers={"If-None-Match": etag},
        )

        assert busy.status_code == 200, busy.text
        assert busy.headers["etag"] == etag
        assert cached.status_code == 304
        assert cached.content == b""
        assert cached.headers["etag"] == etag
        assert database.get_raylight_runtime_state() == ray_before
        assert (fake_comfy.running, fake_comfy.pending) == queue_before
        assert fake_comfy.queue_requests == 0
        assert client.director_app.state.submission_tails["stage5-busy"] is busy_tail
        assert not busy_tail.done()
    finally:
        client.director_app.state.submission_tails.pop("stage5-busy", None)
        busy_tail.cancel()


async def test_feature_preflight_accepts_valid_v4_without_emit_or_writes(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)

    def forbidden_emit(*_args, **_kwargs):
        raise AssertionError("feature preflight must stop before emit")

    monkeypatch.setattr(V4BuiltinInterpreter, "emit", forbidden_emit)
    database = client.director_app.state.database
    jobs_before = database.list_jobs()
    ray_before = deepcopy(database.get_raylight_runtime_state())

    response = await client.post(
        "/api/features/preflight",
        json=_v4_t2v_request(),
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["effective_by_segment"]
    assert database.list_jobs() == jobs_before
    assert database.get_raylight_runtime_state() == ray_before
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []


async def test_feature_preflight_reports_missing_model_without_side_effects(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)
    fake_comfy.model_inventory["fl2va"] = []
    database = client.director_app.state.database
    jobs_before = database.list_jobs()

    response = await client.post(
        "/api/features/preflight",
        json=_v4_t2v_request(),
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["valid"] is False
    assert "model_binding_unavailable" in _error_codes(report)
    assert database.list_jobs() == jobs_before
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []


@pytest.mark.parametrize("endpoint", ("/api/features/preflight", "/api/timeline/compile"))
async def test_unused_audio_vae_does_not_block_fl2va_mute(
    client,
    fake_comfy,
    endpoint: str,
) -> None:
    fake_comfy.model_inventory["audio_vae"] = []
    body = {
        "config": fixture_builder._draft(
            "t2v",
            audio_mode="mute",
        ).model_dump(mode="json")
    }

    response = await client.post(endpoint, json=body)

    assert response.status_code == 200, response.text
    if endpoint.endswith("preflight"):
        assert response.json()["valid"] is True


async def test_generated_audio_still_requires_audio_vae(
    client,
    fake_comfy,
) -> None:
    fake_comfy.model_inventory["audio_vae"] = []

    response = await client.post(
        "/api/features/preflight",
        json=_v4_t2v_request(),
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["valid"] is False
    reason = next(
        item
        for item in report["errors"]
        if item["code"] == "model_binding_unavailable"
    )
    assert reason["safe_details"]["bindings"] == [
        "audio_vae"
    ]


@pytest.mark.parametrize("backend", ("xpu", "mps"))
async def test_standard_gpu_placement_uses_snapshot_backend_neutral_namespace(
    client,
    fake_comfy,
    backend: str,
) -> None:
    snapshot = _with_gpu_backend(_provider_snapshot(client), backend)
    client.director_app.state.host_capability_provider = (
        _MutableHostCapabilityProvider(snapshot)
    )
    authority = (await client.get("/api/settings/authority")).json()
    settings = authority["settings"]
    settings["placement"]["fl2va"]["device"] = "gpu:0"
    settings["placement"]["clip_device"] = "gpu:0"
    settings["placement"]["video_vae_device"] = "gpu:0"
    saved = await client.put(
        "/api/settings/authority",
        json={
            "document": settings,
            "expected_authority_token": authority["authority_token"],
            "schema_version": 3,
        },
    )
    assert saved.status_code == 200, saved.text

    response = await client.post(
        "/api/features/preflight",
        json={
            "config": fixture_builder._draft(
                "t2v",
                audio_mode="mute",
            ).model_dump(mode="json")
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["valid"] is True


async def test_feature_preflight_fails_closed_on_host_contract_drift(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)
    snapshot = _without_node(_provider_snapshot(client), "SaveVideo")
    client.director_app.state.host_capability_provider = (
        _MutableHostCapabilityProvider(snapshot)
    )

    response = await client.post(
        "/api/features/preflight",
        json=_v4_t2v_request(),
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["valid"] is False
    assert "node_unavailable" in _error_codes(report)
    assert report["host_capability_revision"] == (
        snapshot.host_capability_revision()
    )
    assert client.director_app.state.database.list_jobs() == []
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []


async def test_feature_preflight_ignores_interface_node_fingerprint_drift(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)
    snapshot = _with_module_fingerprint_drift(
        _provider_snapshot(client),
        "SaveVideo",
    )
    client.director_app.state.host_capability_provider = (
        _MutableHostCapabilityProvider(snapshot)
    )

    response = await client.post(
        "/api/features/preflight",
        json=_v4_t2v_request(),
    )

    assert response.status_code == 200, response.text
    assert response.json()["valid"] is True
    assert fake_comfy.prompts == []


@pytest.mark.parametrize(
    ("tool", "draft_factory", "catalog_state"),
    (
        ("ffprobe", lambda: fixture_builder._draft("t2v"), "unavailable"),
        (
            "ffmpeg",
            lambda: fixture_builder._continuity_draft().model_copy(
                update={"export_mode": "all"}
            ),
            "conditional",
        ),
    ),
    ids=("take-observation-ffprobe", "timeline-assembly-ffmpeg"),
)
async def test_media_capability_reason_is_shared_by_catalog_preflight_compile_and_submit(
    client,
    fake_comfy,
    monkeypatch,
    tool: str,
    draft_factory,
    catalog_state: str,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)
    snapshot = _without_media_tool(_provider_snapshot(client), tool)
    client.director_app.state.host_capability_provider = (
        _MutableHostCapabilityProvider(snapshot)
    )
    request_body = {
        "config": draft_factory().model_dump(mode="json"),
    }

    catalog_response = await client.get("/api/features/catalog")
    assert catalog_response.status_code == 200, catalog_response.text
    save_take = next(
        entry
        for entry in catalog_response.json()["entries"]
        if entry["id"] == "save_take"
    )
    assert save_take["availability"]["state"] == catalog_state

    preflight = await client.post(
        "/api/features/preflight",
        json=request_body,
    )
    assert preflight.status_code == 200, preflight.text
    preflight_reasons = [
        reason
        for reason in preflight.json()["errors"]
        if reason["code"] == "media_tool_unavailable"
        and reason["safe_details"] == {"tool": tool}
    ]
    assert preflight_reasons
    if catalog_state == "unavailable":
        catalog_reason = next(
            reason
            for reason in save_take["availability"]["reasons"]
            if reason["code"] == "media_tool_unavailable"
            and reason["backend"] == "standard"
        )
        assert _reason_semantics(catalog_reason) == _reason_semantics(
            preflight_reasons[0]
        )

    database = client.director_app.state.database
    jobs_before = database.list_jobs()
    for endpoint in ("/api/timeline/compile", "/api/timeline/jobs"):
        rejected = await client.post(endpoint, json=request_body)
        assert rejected.status_code == 422, rejected.text
        detail = rejected.json()["detail"]
        assert detail["code"] == "media_tool_unavailable"
        endpoint_reason = next(
            reason
            for reason in detail["reasons"]
            if reason["safe_details"] == {"tool": tool}
        )
        assert _reason_semantics(endpoint_reason) == _reason_semantics(
            preflight_reasons[0]
        )

    assert database.list_jobs() == jobs_before
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []


async def test_feature_preflight_reports_operational_readiness_without_feature_gate(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)
    database = client.director_app.state.database
    database.put_raylight_runtime_state(
        {
            "version": 1,
            "epoch": 3,
            "current": None,
        }
    )
    ray_before = deepcopy(database.get_raylight_runtime_state())

    response = await client.post(
        "/api/features/preflight",
        json=_v4_t2v_request(),
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["valid"] is True
    assert report["operational_readiness"]["submission_allowed"] is False
    assert report["operational_readiness"]["ray_recovery_required"] is True
    assert "ray_recovery_required" not in _error_codes(report)
    assert database.get_raylight_runtime_state() == ray_before
    assert database.list_jobs() == []
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []


@pytest.mark.parametrize("backend", ("xpu", "mps"))
async def test_persisted_ray_ledger_uses_cuda_only_readiness_namespace(
    client,
    backend: str,
) -> None:
    snapshot = _with_gpu_backend(
        _provider_snapshot(client),
        backend,
        count=2,
    )
    client.director_app.state.host_capability_provider = (
        _MutableHostCapabilityProvider(snapshot)
    )
    database = client.director_app.state.database
    database.put_raylight_runtime_state(
        {
            "version": 2,
            "epoch": 4,
            "current": {
                "version": 2,
                "initializer_node_id": "initializer",
                "loader_subgraph": {
                    "initializer": {
                        "class_type": "DirectorDeckRayInitializerAdvanced",
                        "inputs": {"GPU": 2, "GPU_SELECT": "0,1"},
                    }
                },
            },
            "tail_prompt_id": None,
            "tail_action": None,
            "tainted": False,
        }
    )

    response = await client.post(
        "/api/features/preflight",
        json={
            "config": fixture_builder._draft(
                "t2v",
                audio_mode="mute",
            ).model_dump(mode="json")
        },
    )

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["valid"] is True
    assert report["operational_readiness"][
        "invalid_runtime_gpu_indices"
    ] == [0, 1]
    assert "invalid_runtime_gpu_indices" not in _error_codes(report)


async def test_preflight_and_compile_never_use_legacy_capability_cancel_probe(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)
    cancel_before = (
        list(fake_comfy.cancelled),
        list(fake_comfy.pending_cancelled),
        list(fake_comfy.interrupted),
        fake_comfy.queue_requests,
        list(fake_comfy.history_requests),
    )

    preflight = await client.post(
        "/api/features/preflight",
        json=_v4_t2v_request(),
    )
    compiled = await client.post(
        "/api/timeline/compile",
        json=_v4_t2v_request(),
    )

    assert preflight.status_code == 200, preflight.text
    assert preflight.json()["valid"] is True
    assert compiled.status_code == 200, compiled.text
    compile_report = compiled.json()
    assert set(
        (
            "template_bundle_version",
            "host_capability_revision",
            "plans",
            "node_policy",
            "features",
            "effective_execution_digest",
        )
    ) <= set(compile_report)
    assert compile_report["features"]["requested"][
        "template_bundle_version"
    ] == compile_report["template_bundle_version"]
    assert set(compile_report["features"]["effective_by_segment"]) == {
        plan["segment_id"] for plan in compile_report["plans"]
    }
    assert compile_report["features"]["resolutions"]
    resolution = compile_report["features"]["resolutions"][0]
    assert set(resolution) == {
        "segment_id",
        "unit_id",
        "feature_id",
        "version",
        "backend",
        "family",
        "template_id",
        "resolution",
        "adapter_fingerprint",
        "capability",
    }
    assert set(resolution["resolution"]) == {
        "state",
        "implementations",
        "resolution_details",
    }
    assert resolution["resolution"]["implementations"]
    assert compile_report["features"]["notices"] == []
    assert compile_report["effective_execution_digest"]["algorithm"] == (
        "sha256-canonical-json-v1"
    )
    assert re.fullmatch(
        r"sha256-[0-9a-f]{64}",
        compile_report["effective_execution_digest"]["value"],
    )
    assert fake_comfy.prompts == []
    assert (
        fake_comfy.cancelled,
        fake_comfy.pending_cancelled,
        fake_comfy.interrupted,
        fake_comfy.queue_requests,
        fake_comfy.history_requests,
    ) == cancel_before


async def test_host_context_observation_failure_has_one_safe_entrypoint_contract(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    async def unavailable_models() -> dict[str, list[str]]:
        raise ComfyError("private upstream detail must not cross the API")

    monkeypatch.setattr(fake_comfy, "models", unavailable_models)
    expected_semantics: tuple[object, ...] | None = None
    for endpoint in (
        "/api/features/preflight",
        "/api/timeline/compile",
        "/api/timeline/jobs",
    ):
        response = await client.post(endpoint, json=_v4_t2v_request())
        if endpoint.endswith("preflight"):
            assert response.status_code == 200, response.text
            reason = response.json()["errors"][0]
        else:
            assert response.status_code == 502, response.text
            detail = response.json()["detail"]
            assert detail["code"] == "host_context_unavailable"
            reason = detail["reasons"][0]
        semantics = _reason_semantics(reason)
        expected_semantics = expected_semantics or semantics
        assert semantics == expected_semantics
        assert "private upstream detail" not in response.text

    assert client.director_app.state.database.list_jobs() == []
    assert fake_comfy.prompts == []


async def test_segment_selection_reason_is_identical_and_does_not_echo_ids(
    client,
    fake_comfy,
) -> None:
    secret_id = "missing-secret-segment"
    body = {
        **_v4_t2v_request(),
        "segment_ids": [secret_id],
    }
    expected_reason: dict[str, object] | None = None

    for endpoint in (
        "/api/features/preflight",
        "/api/timeline/compile",
        "/api/timeline/jobs",
    ):
        response = await client.post(endpoint, json=body)
        if endpoint.endswith("preflight"):
            assert response.status_code == 200, response.text
            report = response.json()
            assert report["valid"] is False
            reason = report["errors"][0]
        else:
            assert response.status_code == 422, response.text
            reason = response.json()["detail"]["reasons"][0]
        assert reason["code"] == "segment_selection_invalid"
        expected_reason = expected_reason or reason
        assert reason == expected_reason
        assert secret_id not in response.text

    assert client.director_app.state.database.list_jobs() == []
    assert fake_comfy.prompts == []


async def test_missing_asset_has_one_safe_entrypoint_contract(
    client,
    fake_comfy,
) -> None:
    secret_id = "unregistered-private-asset"
    document = fixture_builder._draft("i2v").model_dump(mode="json")
    document["segments"][0]["first_image"]["id"] = secret_id
    body = {"config": document}
    expected_reason: dict[str, object] | None = None

    for endpoint in (
        "/api/features/preflight",
        "/api/timeline/compile",
        "/api/timeline/jobs",
    ):
        response = await client.post(endpoint, json=body)
        if endpoint.endswith("preflight"):
            assert response.status_code == 200, response.text
            reason = response.json()["errors"][0]
        else:
            assert response.status_code == 422, response.text
            reason = response.json()["detail"]["reasons"][0]
        assert reason["code"] == "asset_unavailable"
        expected_reason = expected_reason or reason
        assert reason == expected_reason
        assert secret_id not in response.text

    assert client.director_app.state.database.list_jobs() == []
    assert fake_comfy.prompts == []


@pytest.mark.parametrize("missing_model", (False, True), ids=("creative-only", "creative-and-host"))
async def test_unmapped_standard_lora_uses_default_loader_without_hiding_host_errors(
    client,
    fake_comfy,
    missing_model: bool,
) -> None:
    await _enable_unmapped_standard_lora(client)
    if missing_model:
        fake_comfy.model_inventory["fl2va"] = []
    body = _v4_t2v_request()

    preflight = await client.post("/api/features/preflight", json=body)
    assert preflight.status_code == 200, preflight.text
    expected_reasons = preflight.json()["errors"]
    assert [reason["code"] for reason in expected_reasons] == (
        ["model_binding_unavailable"] if missing_model else []
    )

    compiled = await client.post("/api/timeline/compile", json=body)
    if missing_model:
        assert compiled.status_code == 422, compiled.text
        assert compiled.json()["detail"]["reasons"] == expected_reasons
    else:
        assert compiled.status_code == 200, compiled.text
        lora_resolution = next(
            resolution
            for resolution in compiled.json()["features"]["resolutions"]
            if resolution["feature_id"] == "lora"
        )
        assert lora_resolution["resolution"]["resolution_details"] == {
            "source": "legacy_v4_exact_native_fragment",
            "backend": "standard",
            "family": "fl2va",
            "adapter_id": "model_only",
            "binding": {
                "family": "fl2va",
                "model_filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "lora_filename": "style.safetensors",
            },
            "mapping_source": "factory_default",
            "strength": 1.0,
            "loader_options": {},
        }
        assert [
            implementation["class_type"]
            for implementation in lora_resolution["resolution"]["implementations"]
        ] == ["LoraLoaderModelOnly"]


async def test_standard_lora_adapter_fingerprint_is_shared_by_catalog_preflight_and_compile(
    client,
    fake_comfy,
) -> None:
    await _enable_mapped_standard_lora(client)
    body = _v4_t2v_request()

    catalog_response = await client.get("/api/features/catalog")
    preflight_response = await client.post("/api/features/preflight", json=body)
    compile_response = await client.post("/api/timeline/compile", json=body)

    assert catalog_response.status_code == 200, catalog_response.text
    assert preflight_response.status_code == 200, preflight_response.text
    assert preflight_response.json()["valid"] is True
    assert compile_response.status_code == 200, compile_response.text

    lora_catalog = next(
        entry
        for entry in catalog_response.json()["entries"]
        if entry["id"] == "lora"
    )
    catalog_option = next(
            option
            for option in lora_catalog["adapter_options"]
            if option["backend"] == "standard"
            and "fl2va" in option["supported_families"]
            and option["adapter_id"] == "model_only"
    )
    preflight_lora = next(
        feature
        for segment in preflight_response.json()["effective_by_segment"].values()
        for feature in segment["features"]
        if feature["id"] == "lora"
    )
    compiled_lora = next(
        resolution
        for resolution in compile_response.json()["features"]["resolutions"]
        if resolution["feature_id"] == "lora"
    )

    assert catalog_option["adapter_fingerprint"] == (
        preflight_lora["adapter_fingerprint"]
    ) == compiled_lora["adapter_fingerprint"]
    assert catalog_option["capability"] == preflight_lora["capability"]
    assert catalog_option["capability"] == compiled_lora["capability"]


async def test_standard_lora_adapter_reason_is_shared_by_catalog_preflight_and_compile(
    client,
) -> None:
    await _enable_mapped_standard_lora(client)
    snapshot = _without_node(
        _provider_snapshot(client),
        "LoraLoaderModelOnly",
    )
    client.director_app.state.host_capability_provider = (
        _MutableHostCapabilityProvider(snapshot)
    )
    body = _v4_t2v_request()

    catalog_response = await client.get("/api/features/catalog")
    preflight_response = await client.post("/api/features/preflight", json=body)
    compile_response = await client.post("/api/timeline/compile", json=body)

    assert catalog_response.status_code == 200, catalog_response.text
    assert preflight_response.status_code == 200, preflight_response.text
    assert preflight_response.json()["valid"] is False
    assert compile_response.status_code == 422, compile_response.text

    lora_catalog = next(
        entry
        for entry in catalog_response.json()["entries"]
        if entry["id"] == "lora"
    )
    catalog_option = next(
            option
            for option in lora_catalog["adapter_options"]
            if option["backend"] == "standard"
            and "fl2va" in option["supported_families"]
            and option["adapter_id"] == "model_only"
    )
    preflight_lora = next(
        feature
        for segment in preflight_response.json()["effective_by_segment"].values()
        for feature in segment["features"]
        if feature["id"] == "lora"
    )
    catalog_reason = catalog_option["capability"]["reasons"][0]
    preflight_reason = preflight_lora["capability"]["reasons"][0]
    compile_reason = next(
        reason
        for reason in compile_response.json()["detail"]["reasons"]
        if reason["feature_id"] == "lora"
        and reason["code"] == "node_unavailable"
    )

    assert catalog_option["adapter_fingerprint"] == (
        preflight_lora["adapter_fingerprint"]
    )
    assert _reason_semantics(catalog_reason) == _reason_semantics(
        preflight_reason
    ) == _reason_semantics(compile_reason)
    assert client.director_app.state.database.list_jobs() == []


@pytest.mark.parametrize(
    "private_filename",
    (
        "/" + "home/alice/private/model.safetensors",
        "C:\\" + "Users\\Alice\\private\\model.safetensors",
        r"\\server\share\private\model.safetensors",
        "file:///" + "home/alice/private/model.safetensors",
    ),
    ids=("posix", "windows-drive", "windows-unc", "file-uri"),
)
async def test_missing_model_reason_never_echoes_private_filename(
    client,
    fake_comfy,
    private_filename: str,
) -> None:
    authority = (await client.get("/api/timeline/authority")).json()
    authority["document"]["model_stack"]["fl2va"]["filename"] = private_filename
    saved = await client.put(
        "/api/timeline/authority",
        json={
            "document": authority["document"],
            "expected_revision": authority["revision"],
        },
    )
    assert saved.status_code == 200, saved.text
    expected_reasons: list[dict[str, object]] | None = None

    for endpoint in (
        "/api/features/preflight",
        "/api/timeline/compile",
        "/api/timeline/jobs",
    ):
        response = await client.post(endpoint, json=_v4_t2v_request())
        if endpoint.endswith("preflight"):
            assert response.status_code == 200, response.text
            reasons = response.json()["errors"]
        else:
            assert response.status_code == 422, response.text
            reasons = response.json()["detail"]["reasons"]
        reason = next(
            item for item in reasons if item["code"] == "model_binding_unavailable"
        )
        assert reason["safe_details"] == {"bindings": ["fl2va"]}
        expected_reasons = expected_reasons or reasons
        assert reasons == expected_reasons
        assert private_filename not in response.text

    assert client.director_app.state.database.list_jobs() == []
    assert fake_comfy.prompts == []


async def test_missing_project_is_rejected_consistently_before_execution(
    client,
    fake_comfy,
) -> None:
    private_project_id = "missing-private-project"
    body = {
        "config": await v5_timeline_document(
            client,
            fixture_builder._draft("t2v").model_dump(mode="json"),
        )
    }
    preflight = await client.post(
        "/api/features/preflight",
        json={**body, "project_id": private_project_id},
    )

    assert preflight.status_code == 200, preflight.text
    assert preflight.json()["valid"] is False
    expected_reason = preflight.json()["errors"][0]
    assert expected_reason["code"] == "project_not_found"
    assert private_project_id not in preflight.text

    for endpoint in (
        f"/api/projects/{private_project_id}/compile",
        f"/api/projects/{private_project_id}/jobs",
    ):
        rejected = await client.post(endpoint, json=body)
        assert rejected.status_code == 404, rejected.text
        detail = rejected.json()["detail"]
        assert detail["code"] == "project_not_found"
        assert detail["reasons"] == [expected_reason]
        assert private_project_id not in rejected.text

    assert client.director_app.state.database.list_jobs() == []
    assert fake_comfy.prompts == []


@pytest.mark.parametrize(
    "endpoint",
    (
        "/api/projects/missing-private-project/compile",
        "/api/projects/missing-private-project/jobs",
    ),
)
async def test_project_execution_requires_explicit_v5_snapshot_before_lookup(
    client,
    endpoint: str,
) -> None:
    rejected = await client.post(endpoint, json={})

    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["detail"][0]["loc"] == ["body", "config"]


@pytest.mark.parametrize(
    ("case", "expected_code"),
    (
        ("frame-limit", "segment_frame_limit_exceeded"),
        ("ref2va-input", "ref2va_input_required"),
        ("missing-asset", "asset_unavailable"),
    ),
)
async def test_safe_creative_failure_is_identical_across_execution_entrypoints(
    client,
    fake_comfy,
    case: str,
    expected_code: str,
) -> None:
    if case == "frame-limit":
        document = fixture_builder._draft("t2v").model_dump(mode="json")
        document["segments"][0]["duration_seconds"] = 30.0
        private_value = ""
    elif case == "ref2va-input":
        document = fixture_builder._draft("r2v").model_dump(mode="json")
        segment = document["segments"][0]
        segment.update(
            prompt="Generate an unfinished Ref2VA segment.",
            source_video=None,
            reference_images=[],
            reference_audios=[],
            reference_videos=[],
        )
        private_value = str(segment["id"])
    else:
        document = fixture_builder._draft("i2v").model_dump(mode="json")
        private_value = str(document["segments"][0]["first_image"]["id"])

    body = {"config": document}
    preflight = await client.post("/api/features/preflight", json=body)
    assert preflight.status_code == 200, preflight.text
    expected_reasons = preflight.json()["errors"]
    assert expected_reasons[0]["code"] == expected_code

    for endpoint in ("/api/timeline/compile", "/api/timeline/jobs"):
        rejected = await client.post(endpoint, json=body)
        assert rejected.status_code == 422, rejected.text
        assert rejected.json()["detail"]["reasons"] == expected_reasons
        if private_value:
            assert private_value not in rejected.text

    if private_value:
        assert private_value not in preflight.text
    assert client.director_app.state.database.list_jobs() == []
    assert fake_comfy.prompts == []


@pytest.mark.parametrize(
    "endpoint",
    ("/api/timeline/compile", "/api/timeline/jobs"),
    ids=("compile", "submit"),
)
async def test_compile_and_submit_recapture_drift_before_emit_or_persistence(
    client,
    fake_comfy,
    monkeypatch,
    endpoint: str,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)
    provider = _MutableHostCapabilityProvider(_provider_snapshot(client))
    client.director_app.state.host_capability_provider = provider

    advisory = await client.post(
        "/api/features/preflight",
        json=_v4_t2v_request(),
    )
    assert advisory.status_code == 200, advisory.text
    assert advisory.json()["valid"] is True
    assert provider.calls == 1

    # CLIPLoader belongs to the first active feature. If the endpoint reuses
    # the advisory result, or evaluates after graph emission, this assertion
    # catches the authority violation before any job row can be accepted.
    provider.current = _without_node(provider.current, "CLIPLoader")
    emitted_features: list[str] = []
    original_emit = V4BuiltinInterpreter.emit

    def tracking_emit(self, *args, **kwargs):
        emitted_features.append(self.id)
        return original_emit(self, *args, **kwargs)

    monkeypatch.setattr(V4BuiltinInterpreter, "emit", tracking_emit)
    database = client.director_app.state.database
    jobs_before = database.list_jobs()

    rejected = await client.post(endpoint, json=_v4_t2v_request())

    assert rejected.status_code == 422, rejected.text
    assert rejected.json()["detail"]["code"] == "node_unavailable"
    assert provider.calls == 2
    assert emitted_features == []
    assert database.list_jobs() == jobs_before
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []


@pytest.mark.parametrize(
    "invalid_inventory",
    (
        None,
        {
            "fl2va": None,
            "ref2va": [],
            "clip": [],
            "video_vae": [],
            "audio_vae": [],
            "loras": [],
        },
    ),
    ids=("not-a-mapping", "category-not-a-list"),
)
async def test_invalid_model_inventory_has_one_safe_entrypoint_contract(
    client,
    fake_comfy,
    monkeypatch,
    invalid_inventory,
) -> None:
    async def invalid_models():
        return invalid_inventory

    monkeypatch.setattr(fake_comfy, "models", invalid_models)
    expected_reason: dict[str, object] | None = None

    for endpoint in (
        "/api/features/preflight",
        "/api/timeline/compile",
        "/api/timeline/jobs",
    ):
        response = await client.post(endpoint, json=_v4_t2v_request())
        if endpoint.endswith("preflight"):
            assert response.status_code == 200, response.text
            reason = response.json()["errors"][0]
        else:
            assert response.status_code == 502, response.text
            reason = response.json()["detail"]["reasons"][0]
        assert reason["code"] == "host_context_unavailable"
        expected_reason = expected_reason or reason
        assert reason == expected_reason

    assert client.director_app.state.database.list_jobs() == []
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []


@pytest.mark.parametrize(
    ("method", "endpoint", "body"),
    (
        ("get", "/api/features/catalog", None),
        ("post", "/api/features/preflight", _v4_t2v_request()),
        ("post", "/api/timeline/compile", _v4_t2v_request()),
        ("post", "/api/timeline/jobs", _v4_t2v_request()),
    ),
    ids=("catalog", "preflight", "compile", "submit"),
)
async def test_missing_host_capability_provider_fails_closed(
    client,
    fake_comfy,
    monkeypatch,
    method: str,
    endpoint: str,
    body: dict[str, object] | None,
) -> None:
    _install_forbidden_legacy_capabilities(monkeypatch, fake_comfy)
    client.director_app.state.host_capability_provider = None
    database = client.director_app.state.database
    jobs_before = database.list_jobs()

    if body is not None and isinstance(body.get("config"), dict):
        body = {
            **body,
            "config": await v5_timeline_document(client, body["config"]),
        }

    response = await client.request(method, endpoint, json=body)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == (
        "host_capability_provider_unavailable"
    )
    assert database.list_jobs() == jobs_before
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []
