from __future__ import annotations

import asyncio
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

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


def _bundle6_feature_use(response, feature_id: str) -> dict[str, Any]:
    document = response.json()
    assert document["template_bundle_version"] == 6
    matches = [
        use
        for use in document["features"]["uses"]
        if use["feature_id"] == feature_id
    ]
    assert len(matches) == 1
    return matches[0]


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


async def _wait_until(predicate) -> None:
    async def ready() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(ready(), timeout=2.0)


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
async def test_media_capability_reason_remains_diagnostic_only(
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
    job_ids_before = {job["id"] for job in database.list_jobs()}
    compiled = await client.post("/api/timeline/compile", json=request_body)
    assert compiled.status_code == 200, compiled.text
    assert {job["id"] for job in database.list_jobs()} == job_ids_before

    submitted = await client.post("/api/timeline/jobs", json=request_body)
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["id"]
    await _wait_until(lambda: bool(fake_comfy.prompts))
    stored = database.get_job(job_id)
    assert stored is not None
    assert stored["status"] in {"preparing", "queued", "running", "succeeded"}
    assert fake_comfy.prompts
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
    assert compile_report["features"]["effective_by_segment"] == {}
    assert compile_report["features"]["resolutions"] == []
    assert {
        use["segment_id"] for use in compile_report["features"]["uses"]
    } == {
        plan["segment_id"] for plan in compile_report["plans"]
    }
    assert {
        use["feature_id"] for use in compile_report["features"]["uses"]
    } == {
        "auxiliary_models",
        "diffusion_model",
        "execution_strategy",
        "lora",
        "comfy_kitchen_attention",
        "sigma_schedule",
        "multimodal_conditioning",
        "continuity",
        "sampling_pipeline",
        "video_decode",
        "audio_output",
        "save_take",
    }
    feature_use = next(
        use
        for use in compile_report["features"]["uses"]
        if use["feature_id"] == "auxiliary_models"
    )
    assert set(feature_use) == {
        "segment_id",
        "unit_id",
        "feature_id",
        "version",
        "backend",
        "family",
        "template_id",
        "state",
        "config_source",
        "reason_code",
        "implementation",
        "execution_identity",
        "runtime_pool_identity",
        "node_emissions",
    }
    assert feature_use["state"] == "applicable"
    assert feature_use["implementation"]
    assert feature_use["node_emissions"]
    assert compile_report["features"]["notices"] == []
    assert compile_report["features"]["advisories"] == []
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


async def test_host_context_observation_failure_does_not_gate_production(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    async def unavailable_models() -> dict[str, list[str]]:
        raise ComfyError("private upstream detail must not cross the API")

    monkeypatch.setattr(fake_comfy, "models", unavailable_models)
    preflight = await client.post(
        "/api/features/preflight", json=_v4_t2v_request()
    )
    assert preflight.status_code == 200, preflight.text
    assert preflight.json()["errors"][0]["code"] == "host_context_unavailable"
    assert "private upstream detail" not in preflight.text

    compiled = await client.post("/api/timeline/compile", json=_v4_t2v_request())
    assert compiled.status_code == 200, compiled.text
    submitted = await client.post("/api/timeline/jobs", json=_v4_t2v_request())
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["id"]
    await _wait_until(
        lambda: job_id not in client.director_app.state.submission_jobs
    )
    stored = client.director_app.state.database.get_job(job_id)
    assert stored is not None and stored["status"] == "queued"
    assert fake_comfy.prompts


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
async def test_unmapped_standard_lora_uses_default_loader_with_advisory_host_errors(
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
    assert compiled.status_code == 200, compiled.text
    lora_use = _bundle6_feature_use(compiled, "lora")
    assert lora_use["state"] == "applicable"
    assert lora_use["implementation"] == {
        "implementation_id": "directordeck.v6.lora.standard",
        "implementation_version": 1,
        "carrier_kind": "comfy_node",
        "responsibility": "host_user",
        "class_types": ["LoraLoaderModelOnly"],
        "binding_key": "model_only",
    }
    assert lora_use["execution_identity"]["details"]["config"] == {
        "backend": "standard",
        "family": "fl2va",
        "model_filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "lora_filename": "style.safetensors",
        "strength": 1.0,
        "adapter_id": "model_only",
        "class_type": "LoraLoaderModelOnly",
        "input_contract": "model_only",
        "source": "factory_default",
        "options": {},
    }
    assert [
        emission["class_type"] for emission in lora_use["node_emissions"]
    ] == ["LoraLoaderModelOnly"]


async def test_standard_lora_adapter_fingerprint_is_stable_without_compile_host_verification(
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
    compiled_lora = _bundle6_feature_use(compile_response, "lora")

    assert catalog_option["adapter_fingerprint"] == (
        preflight_lora["adapter_fingerprint"]
    )
    assert catalog_option["capability"] == preflight_lora["capability"]
    assert compiled_lora["implementation"] == {
        "implementation_id": "directordeck.v6.lora.standard",
        "implementation_version": 1,
        "carrier_kind": "comfy_node",
        "responsibility": "host_user",
        "class_types": ["LoraLoaderModelOnly"],
        "binding_key": "model_only",
    }
    assert "runtime_fingerprint" not in json.dumps(compiled_lora, sort_keys=True)


async def test_standard_lora_adapter_reason_is_diagnostic_without_gating_compile(
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
    catalog_reason = catalog_option["capability"]["reasons"][0]
    preflight_reason = preflight_lora["capability"]["reasons"][0]
    compiled_lora = _bundle6_feature_use(compile_response, "lora")

    assert catalog_option["adapter_fingerprint"] == (
        preflight_lora["adapter_fingerprint"]
    )
    assert _reason_semantics(catalog_reason) == _reason_semantics(
        preflight_reason
    )
    assert [
        emission["class_type"]
        for emission in compiled_lora["node_emissions"]
    ] == ["LoraLoaderModelOnly"]
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
async def test_missing_model_reason_is_advisory_and_privacy_safe(
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
    preflight = await client.post(
        "/api/features/preflight", json=_v4_t2v_request()
    )
    assert preflight.status_code == 200, preflight.text
    reason = next(
        item
        for item in preflight.json()["errors"]
        if item["code"] == "model_binding_unavailable"
    )
    assert reason["safe_details"] == {"bindings": ["fl2va"]}
    assert private_filename not in preflight.text

    compiled = await client.post("/api/timeline/compile", json=_v4_t2v_request())
    assert compiled.status_code == 200, compiled.text
    submitted = await client.post("/api/timeline/jobs", json=_v4_t2v_request())
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["id"]
    await _wait_until(
        lambda: job_id not in client.director_app.state.submission_jobs
    )
    stored = client.director_app.state.database.get_job(job_id)
    assert stored is not None and stored["status"] == "queued"
    assert fake_comfy.prompts


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
async def test_safe_creative_failure_closes_before_or_after_job_admission(
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

    rejected_compile = await client.post("/api/timeline/compile", json=body)
    assert rejected_compile.status_code == 422, rejected_compile.text
    assert rejected_compile.json()["detail"]["reasons"] == expected_reasons

    submitted = await client.post("/api/timeline/jobs", json=body)
    if case == "frame-limit":
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["id"]
        await _wait_until(
            lambda: job_id not in client.director_app.state.submission_jobs
        )
        stored = client.director_app.state.database.get_job(job_id)
        assert stored is not None and stored["status"] == "failed"
        assert expected_code in stored["error"]
    else:
        assert submitted.status_code == 422, submitted.text
        assert submitted.json()["detail"]["reasons"] == expected_reasons
        if private_value:
            assert private_value not in submitted.text

    if private_value:
        assert private_value not in preflight.text
        assert private_value not in rejected_compile.text
    assert len(client.director_app.state.database.list_jobs()) == (
        1 if case == "frame-limit" else 0
    )
    assert fake_comfy.prompts == []


@pytest.mark.parametrize(
    "endpoint",
    ("/api/timeline/compile", "/api/timeline/jobs"),
    ids=("compile", "submit"),
)
async def test_compile_and_submit_do_not_authorize_from_host_observation(
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

    # The diagnostic endpoint still reports its captured observation. Compile
    # and submit deliberately do not recapture or authorize from that result.
    provider.current = _without_node(provider.current, "CLIPLoader")
    monkeypatch.setattr(
        "directordeck.app.project_v5_contextual_host_authority",
        lambda *_args, **_kwargs: pytest.fail(
            "compile/submit must not run the advisory contextual projection"
        ),
    )
    database = client.director_app.state.database
    job_ids_before = {job["id"] for job in database.list_jobs()}

    response = await client.post(endpoint, json=_v4_t2v_request())

    assert response.status_code == 200, response.text
    assert provider.calls == 1
    if endpoint.endswith("compile"):
        uses = response.json()["features"]["uses"]
        assert uses
        assert any(
            use["feature_id"] == "auxiliary_models"
            and any(
                emission["class_type"] == "CLIPLoader"
                for emission in use["node_emissions"]
            )
            for use in uses
        )
        assert {job["id"] for job in database.list_jobs()} == job_ids_before
        assert fake_comfy.prompts == []
    else:
        job_id = response.json()["id"]
        await _wait_until(
            lambda: job_id not in client.director_app.state.submission_jobs
        )
        stored = database.get_job(job_id)
        assert stored is not None and stored["status"] == "queued"
        plan = database.get_job_execution_plan(job_id)
        assert plan is not None and plan.version == 3
        assert any(
            use.feature_id == "auxiliary_models"
            and any(
                emission.class_type == "CLIPLoader"
                for emission in use.node_emissions
            )
            for use in plan.compile_report.feature_resolutions
        )
        assert len(fake_comfy.prompts) == 1
        assert any(
            node.get("class_type") == "CLIPLoader"
            for node in fake_comfy.prompts[0]["prompt"].values()
        )
    assert fake_comfy.cancelled == []


async def test_submit_compile_failure_is_persisted_on_the_admitted_job(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    def fail_compile(*_args, **_kwargs):
        from directordeck.native_templates import NativeTemplateError

        raise NativeTemplateError("synthetic internal compile failure")

    monkeypatch.setattr(
        "directordeck.app.compile_project_execution_plan",
        fail_compile,
    )

    response = await client.post("/api/timeline/jobs", json=_v4_t2v_request())

    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    assert response.json()["status"] == "preparing"
    await _wait_until(
        lambda: job_id not in client.director_app.state.submission_jobs
    )
    stored = client.director_app.state.database.get_job(job_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["stage"] == "compile_failed"
    assert "creative_configuration_invalid" in stored["error"]
    assert fake_comfy.prompts == []


async def test_submit_child_materialization_failure_closes_the_admitted_job(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    database = client.director_app.state.database

    def fail_child_materialization(_child):
        raise RuntimeError("synthetic child persistence failure")

    monkeypatch.setattr(
        database,
        "create_job_child",
        fail_child_materialization,
    )

    response = await client.post("/api/timeline/jobs", json=_v4_t2v_request())

    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    await _wait_until(
        lambda: job_id not in client.director_app.state.submission_jobs
    )
    stored = database.get_job(job_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["stage"] == "preflight_failed"
    assert "execution_plan_invariant_failed" in stored["error"]
    assert fake_comfy.prompts == []


async def test_submit_planning_failure_is_persisted_with_its_local_reason(
    client,
    fake_comfy,
    monkeypatch,
) -> None:
    def fail_planning(*_args, **_kwargs):
        from directordeck.execution.submission import SubmissionPlanningError

        raise SubmissionPlanningError("synthetic locked-plan failure")

    monkeypatch.setattr(
        "directordeck.app.LockedSubmissionPlanner.build_wave",
        fail_planning,
    )

    response = await client.post("/api/timeline/jobs", json=_v4_t2v_request())

    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    await _wait_until(
        lambda: job_id not in client.director_app.state.submission_jobs
    )
    stored = client.director_app.state.database.get_job(job_id)
    assert stored is not None
    assert stored["status"] == "failed"
    assert stored["stage"] == "preflight_failed"
    assert "submission_plan_invalid" in stored["error"]
    assert "synthetic locked-plan failure" in stored["error"]
    assert fake_comfy.prompts == []
    assert client.director_app.state.submission_tails == {}
    assert all(
        not lock.locked()
        for lock in client.director_app.state.submission_locks.values()
    )


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
async def test_invalid_model_inventory_does_not_gate_production(
    client,
    fake_comfy,
    monkeypatch,
    invalid_inventory,
) -> None:
    async def invalid_models():
        return invalid_inventory

    monkeypatch.setattr(fake_comfy, "models", invalid_models)
    preflight = await client.post(
        "/api/features/preflight", json=_v4_t2v_request()
    )
    assert preflight.status_code == 200, preflight.text
    assert preflight.json()["errors"][0]["code"] == "host_context_unavailable"

    compiled = await client.post("/api/timeline/compile", json=_v4_t2v_request())
    assert compiled.status_code == 200, compiled.text
    submitted = await client.post("/api/timeline/jobs", json=_v4_t2v_request())
    assert submitted.status_code == 200, submitted.text
    job_id = submitted.json()["id"]
    await _wait_until(
        lambda: job_id not in client.director_app.state.submission_jobs
    )
    stored = client.director_app.state.database.get_job(job_id)
    assert stored is not None and stored["status"] == "queued"
    assert fake_comfy.prompts
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
async def test_missing_host_capability_provider_only_disables_diagnostics(
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
    job_ids_before = {job["id"] for job in database.list_jobs()}

    if body is not None and isinstance(body.get("config"), dict):
        body = {
            **body,
            "config": await v5_timeline_document(client, body["config"]),
        }

    response = await client.request(method, endpoint, json=body)

    if endpoint in {"/api/features/catalog", "/api/features/preflight"}:
        assert response.status_code == 503, response.text
        assert response.json()["detail"]["code"] == (
            "host_capability_provider_unavailable"
        )
        assert {job["id"] for job in database.list_jobs()} == job_ids_before
        assert fake_comfy.prompts == []
    elif endpoint.endswith("compile"):
        assert response.status_code == 200, response.text
        assert {job["id"] for job in database.list_jobs()} == job_ids_before
        assert fake_comfy.prompts == []
    else:
        assert response.status_code == 200, response.text
        job_id = response.json()["id"]
        await _wait_until(
            lambda: job_id not in client.director_app.state.submission_jobs
        )
        stored = database.get_job(job_id)
        assert stored is not None and stored["status"] == "queued"
        assert fake_comfy.prompts
    assert fake_comfy.cancelled == []
