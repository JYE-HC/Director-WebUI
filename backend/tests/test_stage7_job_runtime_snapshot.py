from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock

import pytest

from directordeck.app import _v5_generation_runtime_details, create_app
from directordeck.execution.submission import LockedSubmissionPlanner
from directordeck.schemas import (
    RuntimeSettings,
    RuntimeSettingsV2,
    RuntimeSettingsV3,
    UnifiedTimelineDraftV5,
    default_settings,
    default_model_stack,
    default_runtime_settings_v3,
    default_runtime_settings_v2,
    default_timeline_draft_v5,
)
from directordeck.workflow.execution import CompiledExecutionPlan, OutputDescriptor
from directordeck.workflow.project_compiler import compile_project_execution_plan
from directordeck.workflow.runtime_snapshot import (
    JobRuntimeSnapshotV1,
    build_job_runtime_snapshot,
)
from directordeck.workflow.v5_compat import (
    compile_v5_execution_plan,
    project_v5_runtime_currentness,
)
from directordeck.workflow.v6_projection import project_v5_authority_to_v6

from .test_execution_evidence_database import _persist_observed_success
from .test_workflow_execution_contracts import endpoint_identity


_MAPPED_STANDARD_MODEL = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
_MAPPED_STANDARD_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"


def _active_mapped_lora_draft() -> UnifiedTimelineDraftV5:
    document = default_timeline_draft_v5(
        default_model_stack()
    ).model_dump(mode="json")
    document["model_stack"]["fl2va"]["filename"] = _MAPPED_STANDARD_MODEL
    document["features"]["project"]["lora"] = {
        "enabled": True,
        "params": {
            "by_family": {
                "fl2va": {
                    "enabled": True,
                    "filename": _MAPPED_STANDARD_LORA,
                    "strength": 0.75,
                },
                "ref2va": {
                    "enabled": False,
                    "filename": None,
                    "strength": 1.0,
                },
            }
        },
    }
    document["segments"] = [
        {
            "id": "stage7-fl-segment",
            "title": "Stage 7 FL",
            "mode": "fl2va",
            "prompt": "A camera moves through a storm.",
            "duration_seconds": 5.0,
            "enabled": True,
            "ref_image_size": "match",
            "audio_mode": "generate",
        }
    ]
    return UnifiedTimelineDraftV5.model_validate(document)


def _settings_with_overrides(
    settings: RuntimeSettingsV3,
    overrides: list[dict[str, object]],
) -> RuntimeSettingsV3:
    document = settings.model_dump(mode="json")
    document["lora_loader_overrides"] = overrides
    return RuntimeSettingsV3.model_validate(document)


def _raylight_settings(settings: RuntimeSettingsV3) -> RuntimeSettingsV3:
    document = settings.model_dump(mode="json")
    document["multi_gpu_enabled"] = True
    document["placement"]["fl2va"]["raylight"].update(
        {
            "gpu_select": [0, 1],
            "ulysses_degree": 2,
            "ring_degree": 1,
        }
    )
    return RuntimeSettingsV3.model_validate(document)


def _changed_active_lora(
    draft: UnifiedTimelineDraftV5,
    *,
    filename: str | None = None,
    strength: float | None = None,
) -> UnifiedTimelineDraftV5:
    document = draft.model_dump(mode="json")
    slot = document["features"]["project"]["lora"]["params"]["by_family"][
        "fl2va"
    ]
    if filename is not None:
        slot["filename"] = filename
    if strength is not None:
        slot["strength"] = strength
    return UnifiedTimelineDraftV5.model_validate(document)


class _PoisonStandardOverrideTable(list[object]):
    """Prove that a Ray-only read never enumerates Standard host knowledge."""

    def __iter__(self):
        raise AssertionError("RayLight currentness read the Standard mapping table")


def _user_mapping_record(adapter_id: str) -> dict[str, object]:
    current_adapter_id = (
        "minimax_h3_turbo" if adapter_id == "dedicated" else adapter_id
    )
    return {
        "lora_filename": _MAPPED_STANDARD_LORA,
        "adapter_id": current_adapter_id,
        "options": (
            {"low_vram": False}
            if current_adapter_id == "minimax_h3_turbo"
            else {}
        ),
    }


def _pathful_historical_v3_pair() -> tuple[
    UnifiedTimelineDraftV5, RuntimeSettingsV3
]:
    mapping = _user_mapping_record("dedicated")
    model_basename = _MAPPED_STANDARD_MODEL.replace("\\", "/").split("/")[-1]
    lora_basename = str(mapping["lora_filename"]).replace("\\", "/").split("/")[-1]
    model_filename = f"/private/history/models/{model_basename}"
    lora_filename = f"C:\\private\\history\\loras\\{lora_basename}"

    draft_document = _active_mapped_lora_draft().model_dump(mode="json")
    draft_document["model_stack"] = {
        "fl2va": {"filename": model_filename},
        "ref2va": {"filename": "/private/history/models/ref2va.safetensors"},
        "clip": {"filename": "C:\\private\\history\\clip.safetensors"},
        "video_vae": {"filename": "/private/history/vae/video.safetensors"},
        "audio_vae": {"filename": "\\\\private-host\\audio\\audio.safetensors"},
    }
    draft_document["features"]["project"]["lora"]["params"]["by_family"][
        "fl2va"
    ]["filename"] = lora_filename
    draft = UnifiedTimelineDraftV5.model_validate(draft_document)

    settings = RuntimeSettingsV3.model_validate(
        {
            "schema_version": 3,
            "client_id": "historical-private-client",
            "placement": {
                "fl2va": {"device": "gpu:1"},
                "clip_device": "cpu",
                "video_vae_device": "gpu:2",
                "audio_vae_device": "gpu:3",
            },
            "lora_loader_overrides": [
                {
                    "lora_filename": lora_filename,
                    "adapter_id": "minimax_h3_turbo",
                    "options": {"low_vram": False},
                },
                {
                    "lora_filename": "/private/history/mapping-secret-lora.safetensors",
                    "adapter_id": "model_only",
                    "options": {},
                },
            ],
        }
    )
    return draft, settings


def _seed_readable_job(database, draft, settings: RuntimeSettingsV3) -> str:
    now = "2026-08-22T12:00:00+00:00"
    job_id = "stage7-runtime-snapshot-job"
    segment_id = "stage7-fl-segment"
    plan = compile_project_execution_plan(
        draft,
        settings,
        job_id,
        [segment_id],
    )
    runtime_snapshot = build_job_runtime_snapshot(
        draft,
        [segment_id],
        settings,
        plan,
    )
    database.create_job(
        {
            "id": job_id,
            "mode": "timeline",
            "status": "preparing",
            "progress": 0.0,
            "stage": "preflight",
            "prompt_id": None,
            "project_id": database.LEGACY_DEFAULT_PROJECT_ID,
            "outputs": [],
            "error": None,
            "config_snapshot": {
                "timeline": draft.model_dump(mode="json"),
                "segment_ids": [segment_id],
            },
            "settings_snapshot": runtime_snapshot.model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    database.create_job_execution_plan(job_id, plan)
    planner = LockedSubmissionPlanner(endpoint_identity(endpoint_key="embedded"))
    locked = planner.build_wave(
        plan,
        source_unit_ordinal=0,
        segment_child_id="stage7-runtime-snapshot-child",
    )
    database.persist_job_child_submission_intent(
        job_id,
        locked_plan=locked,
        exact_snapshot=planner.exact_snapshot(locked, locked.units[0]),
    )
    _persist_observed_success(
        database,
        "stage7-runtime-snapshot-child",
        output=OutputDescriptor(
            filename="stage7-currentness.mp4",
            subfolder="segments",
        ),
        has_audio=True,
    )
    database.update_job(
        job_id,
        status="succeeded",
        progress=1.0,
        stage="completed",
        started_at=now,
        completed_at=now,
    )
    return job_id


def _install_retained_v5_authority(database, draft: UnifiedTimelineDraftV5) -> None:
    """Model one project intentionally retained on its frozen Bundle-5 path."""

    with database.connect() as connection:
        connection.execute(
            "UPDATE unified_timeline SET document = ?, revision = revision + 1 "
            "WHERE singleton = 1",
            (draft.model_dump_json(),),
        )


def test_new_job_snapshot_contains_only_actual_adapter_not_mapping_table() -> None:
    draft = _active_mapped_lora_draft()
    baseline = RuntimeSettingsV3.model_validate(
        {
            "schema_version": 3,
            "client_id": "directordeck",
            "lora_loader_overrides": [],
        }
    )
    settings = _settings_with_overrides(
        baseline,
        [
            _user_mapping_record("dedicated"),
            {
                "lora_filename": "unrelated-lora.safetensors",
                "adapter_id": "model_only",
                "options": {},
            },
        ],
    )
    plan = compile_v5_execution_plan(
        draft,
        settings,
        "stage7-bounded-snapshot",
        ["stage7-fl-segment"],
    )
    snapshot = build_job_runtime_snapshot(
        draft,
        ["stage7-fl-segment"],
        settings,
        plan,
    )
    document = snapshot.model_dump(mode="json")
    serialized = json.dumps(document, sort_keys=True)

    assert "lora_loader_overrides" not in serialized
    assert "unrelated-lora.safetensors" not in serialized
    assert document["resolved_lora_adapters"] == [
        {
            "family": "fl2va",
            "backend": "standard",
            "adapter_id": "minimax_h3_turbo",
            "binding": {
                "family": "fl2va",
                "model_filename": _MAPPED_STANDARD_MODEL,
                "lora_filename": _MAPPED_STANDARD_LORA,
            },
            "class_type": "MiniMaxH3TurboLoRA",
            "node_contract_id": "directordeck.node.MiniMaxH3TurboLoRA",
            "semantic_version": "1.0.0",
            "options": {"low_vram": False},
            "runtime_fingerprint": document["resolved_lora_adapters"][0][
                "runtime_fingerprint"
            ],
        }
    ]
    assert "source" not in document["resolved_lora_adapters"][0]
    assert document["control_evidence"] == {
        "progress_client_id": settings.client_id,
    }


def _insert_active_job_with_settings_snapshot(
    database,
    *,
    job_id: str,
    settings_snapshot: dict[str, object],
) -> None:
    now = "2026-08-22T12:00:00+00:00"
    database.create_job(
        {
            "id": job_id,
            "mode": "timeline",
            "status": "running",
            "progress": 0.5,
            "stage": "sampling",
            "prompt_id": f"prompt-{job_id}",
            "project_id": database.LEGACY_DEFAULT_PROJECT_ID,
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": settings_snapshot,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": None,
        }
    )


async def test_lifespan_uses_captured_control_evidence_after_client_setting_changes(
    tmp_path,
    monkeypatch,
    fake_comfy,
) -> None:
    app = create_app(
        comfy_url="http://comfy.test:8188",
        database_path=tmp_path / "stage7-control-evidence.sqlite3",
        comfy_factory=lambda _url: fake_comfy,
    )
    database = app.state.database
    database.initialize()
    captured = _settings_with_overrides(
        database.get_settings(),
        [_user_mapping_record("dedicated")],
    ).model_copy(update={"client_id": "submitted-client"})
    draft = _active_mapped_lora_draft()
    segment_ids = ["stage7-fl-segment"]
    plan = compile_v5_execution_plan(
        draft,
        captured,
        "stage7-active-control-evidence",
        segment_ids,
    )
    bounded = build_job_runtime_snapshot(
        draft,
        segment_ids,
        captured,
        plan,
    )
    _insert_active_job_with_settings_snapshot(
        database,
        job_id="stage7-active-control-evidence",
        settings_snapshot=bounded.model_dump(mode="json"),
    )

    # Historical full-settings snapshots remain supported across all three
    # settings generations while new jobs use bounded control evidence.
    legacy_snapshots: tuple[
        tuple[str, RuntimeSettings | RuntimeSettingsV2 | RuntimeSettingsV3], ...
    ] = (
        (
            "legacy-v1",
            default_settings().model_copy(update={"client_id": "legacy-v1-client"}),
        ),
        (
            "legacy-v2",
            default_runtime_settings_v2().model_copy(
                update={"client_id": "legacy-v2-client"}
            ),
        ),
        (
            "legacy-v3",
            captured.model_copy(update={"client_id": "legacy-v3-client"}),
        ),
    )
    for job_id, settings in legacy_snapshots:
        _insert_active_job_with_settings_snapshot(
            database,
            job_id=job_id,
            settings_snapshot=settings.model_dump(mode="json"),
        )

    current = captured.model_copy(update={"client_id": "current-client"})
    _, authority = database.get_settings_authority()
    database.put_settings_v3_authority(
        current,
        expected_authority_token=authority,
        schema_version=3,
    )
    ensure = Mock()
    close = AsyncMock()
    monkeypatch.setattr(app.state.progress_manager, "ensure", ensure)
    monkeypatch.setattr(app.state.progress_manager, "close", close)

    async with app.router.lifespan_context(app):
        pass

    monitored_client_ids = {call.args[1] for call in ensure.call_args_list}
    assert monitored_client_ids == {
        "current-client",
        "submitted-client",
        "legacy-v1-client",
        "legacy-v2-client",
        "legacy-v3-client",
    }
    close.assert_awaited_once_with()
    persisted = database.get_job("stage7-active-control-evidence")
    assert persisted is not None
    assert JobRuntimeSnapshotV1.model_validate(persisted["settings_snapshot"]) == bounded


@pytest.mark.parametrize("backend", ("standard", "raylight"))
@pytest.mark.parametrize("drift", ("binding", "strength"))
def test_runtime_snapshot_rejects_plan_and_creative_lora_drift(
    backend: str,
    drift: str,
) -> None:
    original = _active_mapped_lora_draft()
    changed_filename = "styles/minimax_h3_turbo_stage7-different.safetensors"
    settings = _settings_with_overrides(
        RuntimeSettingsV3.model_validate(
            {
                "schema_version": 3,
                "client_id": "directordeck",
                "lora_loader_overrides": [],
            }
        ),
        [
            _user_mapping_record("dedicated"),
            {
                **_user_mapping_record("dedicated"),
                "lora_filename": changed_filename,
            },
        ],
    )
    if backend == "raylight":
        settings = _raylight_settings(settings)
    plan = compile_v5_execution_plan(
        original,
        settings,
        f"stage7-{backend}-{drift}-drift",
        ["stage7-fl-segment"],
    )
    changed = _changed_active_lora(
        original,
        filename=(changed_filename if drift == "binding" else None),
        strength=(0.5 if drift == "strength" else None),
    )

    with pytest.raises(ValueError, match=f"compiled LoRA {drift} drifted"):
        build_job_runtime_snapshot(
            changed,
            ["stage7-fl-segment"],
            settings,
            plan,
        )


async def test_currentness_ignores_unrelated_mapping_records_but_tracks_adapter(
    client,
) -> None:
    database = client.director_app.state.database
    draft = _active_mapped_lora_draft()
    _install_retained_v5_authority(database, draft)
    captured = _settings_with_overrides(
        database.get_settings(),
        [_user_mapping_record("dedicated")],
    )
    _, token = database.get_settings_authority()
    database.put_settings_v3_authority(
        captured,
        expected_authority_token=token,
        schema_version=3,
    )
    job_id = _seed_readable_job(database, draft, captured)
    raw_job = database.get_job(job_id)
    assert raw_job is not None
    parsed_snapshot = JobRuntimeSnapshotV1.model_validate(
        raw_job["settings_snapshot"]
    )
    projected_runtime = project_v5_runtime_currentness(
        draft,
        ["stage7-fl-segment"],
        captured,
    )
    assert projected_runtime.families[0].family == "fl2va"
    assert parsed_snapshot.runtime_projection.families[0].family == "fl2va"

    stored_plan = database.get_job_execution_plan(job_id)
    assert stored_plan is not None
    recompiled = compile_v5_execution_plan(
        draft,
        captured,
        job_id,
        ["stage7-fl-segment"],
    )
    assert (
        recompiled.effective_execution_digest
        == stored_plan.effective_execution_digest
    )
    assert build_job_runtime_snapshot(
        draft,
        ["stage7-fl-segment"],
        captured,
        recompiled,
    ) == build_job_runtime_snapshot(
        draft,
        ["stage7-fl-segment"],
        captured,
        stored_plan,
    )

    current = await client.get(f"/api/jobs/{job_id}")
    assert current.status_code == 200, current.text
    assert current.json()["segment_results"][0]["current_snapshot"] is True, current.json()

    same_adapter_with_unrelated_record = _settings_with_overrides(
        captured,
        [
            _user_mapping_record("dedicated"),
            {
                "lora_filename": "unrelated-lora.safetensors",
                "adapter_id": "model_only",
                "options": {},
            },
        ],
    ).model_copy(update={"client_id": "stage7-new-monitor-client"})
    _, token = database.get_settings_authority()
    database.put_settings_v3_authority(
        same_adapter_with_unrelated_record,
        expected_authority_token=token,
        schema_version=3,
    )
    still_current = await client.get(f"/api/jobs/{job_id}")
    assert still_current.status_code == 200, still_current.text
    assert (
        still_current.json()["segment_results"][0]["current_snapshot"] is True
    )

    unrelated_mapping = same_adapter_with_unrelated_record.lora_loader_overrides[
        1
    ].model_dump(mode="json")
    changed_adapter = _settings_with_overrides(
        same_adapter_with_unrelated_record,
        [
            _user_mapping_record("model_only"),
            unrelated_mapping,
        ],
    )
    _, token = database.get_settings_authority()
    database.put_settings_v3_authority(
        changed_adapter,
        expected_authority_token=token,
        schema_version=3,
    )
    stale = await client.get(f"/api/jobs/{job_id}")
    assert stale.status_code == 200, stale.text
    assert stale.json()["segment_results"][0]["current_snapshot"] is False

    details = await client.get(f"/api/jobs/{job_id}/generation-details")
    assert details.status_code == 200, details.text
    payload = details.json()
    assert payload["runtime_snapshot_available"] is True
    assert payload["models"] == [
        {
            "family": "fl2va",
            "filename": draft.model_stack.fl2va.filename,
            "device": captured.placement.fl2va.device,
            "lora_name": _user_mapping_record("dedicated")["lora_filename"],
            "lora_strength": 0.75,
            "backends": ["standard"],
            "logical_gpu_indices": [],
            "ulysses_degree": None,
            "ring_degree": None,
        }
    ]
    assert {item["role"] for item in payload["shared_models"]} == {
        "clip",
        "video_vae",
        "audio_vae",
    }

    exported = await client.get(f"/api/jobs/{job_id}/project")
    assert exported.status_code == 200, exported.text
    assert exported.json()["project"] == draft.model_dump(mode="json")
    assert exported.json()["segment_ids"] == ["stage7-fl-segment"]


async def test_bundle6_currentness_uses_v3_plan_and_ck_authority(client) -> None:
    database = client.director_app.state.database
    draft = project_v5_authority_to_v6(_active_mapped_lora_draft()).draft
    _, revision = database.get_timeline_authority()
    database.validate_and_put_timeline_authority(
        draft,
        expected_revision=revision,
    )
    settings = _settings_with_overrides(
        database.get_settings(),
        [_user_mapping_record("dedicated")],
    )
    _, token = database.get_settings_authority()
    database.put_settings_v3_authority(
        settings,
        expected_authority_token=token,
        schema_version=3,
    )
    job_id = _seed_readable_job(database, draft, settings)
    stored_plan = database.get_job_execution_plan(job_id)
    assert stored_plan is not None
    assert stored_plan.version == 3

    current = await client.get(f"/api/jobs/{job_id}")
    assert current.status_code == 200, current.text
    assert current.json()["segment_results"][0]["current_snapshot"] is True

    document = draft.model_dump(mode="json")
    document["features"]["project"]["comfy_kitchen_attention"]["enabled"] = True
    changed = UnifiedTimelineDraftV5.model_validate(document)
    _, revision = database.get_timeline_authority()
    database.validate_and_put_timeline_authority(
        changed,
        expected_revision=revision,
    )
    stale = await client.get(f"/api/jobs/{job_id}")
    assert stale.status_code == 200, stale.text
    assert stale.json()["segment_results"][0]["current_snapshot"] is False


def test_bundle6_runtime_snapshot_rejects_lora_strength_drift() -> None:
    draft = project_v5_authority_to_v6(_active_mapped_lora_draft()).draft
    settings = _settings_with_overrides(
        default_runtime_settings_v3(),
        [_user_mapping_record("dedicated")],
    )
    plan = compile_project_execution_plan(draft, settings, "v6-strength-drift")
    document = plan.model_dump(mode="json")
    lora_use = next(
        use
        for use in document["compile_report"]["feature_resolutions"]
        if use["feature_id"] == "lora" and use["state"] == "applicable"
    )
    lora_use["execution_identity"]["details"]["config"]["strength"] = 0.25
    drifted = CompiledExecutionPlan.model_validate_json(json.dumps(document))

    with pytest.raises(ValueError, match="LoRA evidence drifted"):
        build_job_runtime_snapshot(draft, None, settings, drifted)


async def test_active_raylight_currentness_never_reads_standard_mapping(
    client,
    monkeypatch,
) -> None:
    database = client.director_app.state.database
    draft = _active_mapped_lora_draft()
    _install_retained_v5_authority(database, draft)
    settings = _raylight_settings(
        _settings_with_overrides(
            database.get_settings(),
            [_user_mapping_record("model_only")],
        )
    )
    _, token = database.get_settings_authority()
    database.put_settings_v3_authority(
        settings,
        expected_authority_token=token,
        schema_version=3,
    )
    job_id = _seed_readable_job(database, draft, settings)
    raw_job = database.get_job(job_id)
    assert raw_job is not None
    captured = JobRuntimeSnapshotV1.model_validate(raw_job["settings_snapshot"])
    assert [
        (item.backend, item.adapter_id, item.binding)
        for item in captured.resolved_lora_adapters
    ] == [("raylight", "ray_lora", None)]

    poisoned = settings.model_copy(deep=True)
    object.__setattr__(
        poisoned,
        "lora_loader_overrides",
        _PoisonStandardOverrideTable(settings.lora_loader_overrides),
    )
    monkeypatch.setattr(database, "get_settings", lambda: poisoned)

    response = await client.get(f"/api/jobs/{job_id}")

    assert response.status_code == 200, response.text
    assert response.json()["segment_results"][0]["current_snapshot"] is True


async def test_historical_v5_runtime_v2_snapshot_remains_viewable_and_exportable(
    client,
) -> None:
    database = client.director_app.state.database
    draft = _active_mapped_lora_draft()
    _install_retained_v5_authority(database, draft)
    mapped_settings = _settings_with_overrides(
        database.get_settings(),
        [_user_mapping_record("dedicated")],
    )
    job_id = _seed_readable_job(database, draft, mapped_settings)
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET settings_snapshot = ? WHERE id = ?",
            (default_runtime_settings_v2().model_dump_json(), job_id),
        )

    details = await client.get(f"/api/jobs/{job_id}/generation-details")
    assert details.status_code == 200, details.text
    payload = details.json()
    assert payload["runtime_snapshot_available"] is True
    assert payload["models"][0]["filename"] == draft.model_stack.fl2va.filename
    assert payload["models"][0]["lora_name"] == _user_mapping_record(
        "dedicated"
    )["lora_filename"]
    assert payload["models"][0]["backends"] == ["standard"]

    exported = await client.get(f"/api/jobs/{job_id}/project")
    assert exported.status_code == 200, exported.text
    assert exported.json()["project"] == draft.model_dump(mode="json")


def test_historical_v5_runtime_v3_projection_is_discriminated_and_fail_closed() -> None:
    draft, settings = _pathful_historical_v3_pair()
    job = {
        "config_snapshot": {"segment_ids": ["stage7-fl-segment"]},
        "settings_snapshot": settings.model_dump(mode="json"),
    }

    projected = _v5_generation_runtime_details(job, draft, ["fl2va"])

    assert projected is not None
    models, shared_models = projected
    assert models == [
        {
            "family": "fl2va",
            "filename": str(draft.model_stack.fl2va.filename)
            .replace("\\", "/")
            .split("/")[-1],
            "device": "gpu:1",
                "lora_name": _MAPPED_STANDARD_LORA,
            "lora_strength": 0.75,
            "backends": ["standard"],
            "logical_gpu_indices": [],
            "ulysses_degree": None,
            "ring_degree": None,
        }
    ]
    assert shared_models == [
        {"role": "clip", "filename": "clip.safetensors", "device": "cpu"},
        {
            "role": "video_vae",
            "filename": "video.safetensors",
            "device": "gpu:2",
        },
        {
            "role": "audio_vae",
            "filename": "audio.safetensors",
            "device": "gpu:3",
        },
    ]
    serialized = json.dumps(projected, sort_keys=True)
    assert settings.client_id not in serialized
    assert "lora_loader_overrides" not in serialized
    assert "mapping-secret" not in serialized
    assert "/private/history" not in serialized
    assert "C:\\private\\history" not in serialized

    corrupt = settings.model_dump(mode="json")
    corrupt["placement"]["fl2va"]["device"] = "/private/history/gpu"
    assert (
        _v5_generation_runtime_details(
            {**job, "settings_snapshot": corrupt},
            draft,
            ["fl2va"],
        )
        is None
    )


async def test_historical_v5_runtime_v3_details_are_path_safe_and_immutable(
    client,
    monkeypatch,
) -> None:
    database = client.director_app.state.database
    draft, settings = _pathful_historical_v3_pair()
    job_id = _seed_readable_job(database, draft, settings)
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET settings_snapshot = ? WHERE id = ?",
            (settings.model_dump_json(), job_id),
        )

    def reject_mutable_settings_read() -> None:
        raise AssertionError("generation details read mutable runtime settings")

    monkeypatch.setattr(database, "get_settings", reject_mutable_settings_read)
    details = await client.get(f"/api/jobs/{job_id}/generation-details")

    assert details.status_code == 200, details.text
    payload = details.json()
    assert payload["runtime_snapshot_available"] is True
    assert payload["models"] == [
        {
            "family": "fl2va",
            "filename": str(draft.model_stack.fl2va.filename)
            .replace("\\", "/")
            .split("/")[-1],
            "device": "gpu:1",
            "lora_name": _MAPPED_STANDARD_LORA,
            "lora_strength": 0.75,
            "backends": ["standard"],
            "logical_gpu_indices": [],
            "ulysses_degree": None,
            "ring_degree": None,
        }
    ]
    assert payload["shared_models"] == [
        {"role": "clip", "filename": "clip.safetensors", "device": "cpu"},
        {
            "role": "video_vae",
            "filename": "video.safetensors",
            "device": "gpu:2",
        },
        {
            "role": "audio_vae",
            "filename": "audio.safetensors",
            "device": "gpu:3",
        },
    ]
    wire = details.text
    assert settings.client_id not in wire
    assert "lora_loader_overrides" not in wire
    assert "mapping-secret" not in wire
    assert "/private/history" not in wire
    assert settings.lora_loader_overrides[0].lora_filename not in str(payload)

    corrupt = settings.model_dump(mode="json")
    corrupt["lora_loader_overrides"][0]["adapter_id"] = "not-an-adapter"
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET settings_snapshot = ? WHERE id = ?",
            (json.dumps(corrupt), job_id),
        )
    rejected = await client.get(f"/api/jobs/{job_id}/generation-details")
    assert rejected.status_code == 200, rejected.text
    # Creative details remain useful, but corrupt runtime evidence must not be
    # guessed or projected into the public response.
    assert rejected.json()["runtime_snapshot_available"] is False
    assert rejected.json()["models"] == []
    assert rejected.json()["shared_models"] == []


def test_raylight_runtime_snapshot_restores_json_array_tuples() -> None:
    snapshot = JobRuntimeSnapshotV1.model_validate(
        {
            "snapshot_schema_version": 1,
            "runtime_projection": {
                "memory_policy": "keep_resident",
                "raylight_residency_policy": "keep_until_switch",
                "multi_gpu_enabled": True,
                "families": [
                    {
                        "family": "fl2va",
                        "backend": "raylight",
                        "device": "default",
                        "raylight_profile": {
                            "gpu_select": [0, 1],
                            "ulysses_degree": 2,
                            "ring_degree": 1,
                            "cfg_degree": 1,
                            "dp_degree": 1,
                            "fsdp": False,
                            "cpu_offload": False,
                        },
                    }
                ],
                "clip_device": "default",
                "video_vae_device": "default",
                "audio_vae_device": None,
            },
            "resolved_lora_adapters": [],
        }
    )

    assert snapshot.runtime_projection.families[0].raylight_profile is not None
    assert snapshot.runtime_projection.families[
        0
    ].raylight_profile.gpu_select == (0, 1)
