from __future__ import annotations

import json

from directordeck.schemas import (
    RuntimeSettingsV3,
    UnifiedTimelineDraftV4,
    UnifiedTimelineDraftV5,
    default_runtime_settings_v3,
    default_settings,
    default_timeline_draft,
)
from directordeck.execution.submission import LockedSubmissionPlanner
from directordeck.workflow.execution import OutputDescriptor
from directordeck.workflow.effective_features import (
    migrate_timeline_feature_authority_to_v5,
)
from directordeck.workflow.v5_compat import (
    compile_v5_execution_plan,
    project_v5_runtime_currentness,
)
from directordeck.workflow.runtime_snapshot import build_job_runtime_snapshot

from .test_execution_evidence_database import _persist_observed_success
from .test_workflow_execution_contracts import endpoint_identity


def _complete_v5_draft() -> UnifiedTimelineDraftV5:
    legacy = default_timeline_draft().model_dump(mode="json")
    model_settings = default_settings().models
    legacy.update(
        {
            "version": 5,
            "model_stack": {
                role: {"filename": getattr(model_settings, role).filename}
                for role in (
                    "fl2va",
                    "ref2va",
                    "clip",
                    "video_vae",
                    "audio_vae",
                )
            },
            "features": {
                "template_bundle_version": 4,
                "project": {
                    "lora": {
                        "enabled": False,
                        "params": {
                            "by_family": {
                                family: {
                                    "enabled": False,
                                    "filename": None,
                                    "strength": 1.0,
                                }
                                for family in ("fl2va", "ref2va")
                            }
                        },
                    }
                },
                "by_segment": {},
            },
            "segments": [
                {
                    "id": "fl-segment",
                    "title": "FL",
                    "mode": "fl2va",
                    "prompt": "FL motion",
                    "duration_seconds": 5.0,
                    "enabled": True,
                    "ref_image_size": "match",
                    "audio_mode": "mute",
                },
                {
                    "id": "ref-segment",
                    "title": "Ref",
                    "mode": "ref2va",
                    "prompt": "Ref identity",
                    "duration_seconds": 5.0,
                    "enabled": True,
                    "ref_image_size": "match",
                    "audio_mode": "mute",
                },
            ],
        }
    )
    return UnifiedTimelineDraftV5.model_validate(legacy)


def _runtime_copy(
    settings: RuntimeSettingsV3,
    mutate,
) -> RuntimeSettingsV3:
    document = settings.model_dump(mode="json")
    mutate(document)
    return RuntimeSettingsV3.model_validate(document)


def _install_frozen_v5_authority(database, draft: UnifiedTimelineDraftV5) -> None:
    """Install historical authority bytes without weakening downgrade guards."""

    current = migrate_timeline_feature_authority_to_v5(draft)
    with database.connect() as connection:
        connection.execute(
            "UPDATE unified_timeline SET document = ?, revision = revision + 1 "
            "WHERE singleton = 1",
            (current.model_dump_json(),),
        )


def test_runtime_currentness_ignores_unselected_family_and_unused_runtime_fields() -> None:
    draft = _complete_v5_draft()
    baseline = default_runtime_settings_v3()
    expected = project_v5_runtime_currentness(
        draft,
        ["fl-segment"],
        baseline,
    )

    def mutate_unrelated(document: dict) -> None:
        document["client_id"] = "another-monitor-client"
        document["placement"]["ref2va"] = {
            "device": "gpu:7",
            "raylight": {
                "gpu_select": [2, 3],
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_degree": 1,
                "dp_degree": 1,
                "fsdp": False,
                "cpu_offload": False,
            },
        }
        document["placement"]["audio_vae_device"] = "gpu:6"
        document["lora_loader_overrides"] = [
            {
                "family": "ref2va",
                "model_filename": "unselected-model.safetensors",
                "lora_filename": "unselected-lora.safetensors",
                "adapter_id": "model_only",
            }
        ]

    unrelated = _runtime_copy(baseline, mutate_unrelated)
    assert (
        project_v5_runtime_currentness(draft, ["fl-segment"], unrelated)
        == expected
    )

    # A one-GPU RayLight profile still selects Standard; its inactive GPU
    # index is not part of the emitted graph or placement contract.
    inactive_standard_profile = _runtime_copy(
        baseline,
        lambda document: document["placement"]["fl2va"]["raylight"].update(
            gpu_select=[5]
        ),
    )
    assert (
        project_v5_runtime_currentness(
            draft,
            ["fl-segment"],
            inactive_standard_profile,
        )
        == expected
    )


def test_runtime_currentness_tracks_selected_placement_changes() -> None:
    draft = _complete_v5_draft()
    baseline = default_runtime_settings_v3()
    expected = project_v5_runtime_currentness(
        draft,
        ["fl-segment"],
        baseline,
    )

    changes = (
        lambda document: document["placement"]["fl2va"].update(device="gpu:1"),
        lambda document: document["placement"].update(clip_device="gpu:1"),
        lambda document: document["placement"].update(video_vae_device="gpu:1"),
        lambda document: document["placement"]["fl2va"]["raylight"].update(
            gpu_select=[0, 1],
            ulysses_degree=2,
        ),
    )
    for change in changes:
        changed = _runtime_copy(baseline, change)
        assert (
            project_v5_runtime_currentness(draft, ["fl-segment"], changed)
            != expected
        )


def test_runtime_currentness_gates_global_raylight_controls_by_selected_route() -> None:
    draft = _complete_v5_draft()
    standard = default_runtime_settings_v3()
    changed_standard = _runtime_copy(
        standard,
        lambda document: document.update(
            multi_gpu_enabled=True,
            raylight_residency_policy="release_after_sampling",
        ),
    )

    # A one-GPU profile emits the Standard graph, so Ray process controls are
    # unrelated to this take even when their settings values change.
    assert project_v5_runtime_currentness(
        draft, ["fl-segment"], standard
    ) == project_v5_runtime_currentness(
        draft, ["fl-segment"], changed_standard
    )

    def select_raylight(document: dict) -> None:
        document["multi_gpu_enabled"] = True
        document["placement"]["fl2va"]["raylight"].update(
            gpu_select=[0, 1],
            ulysses_degree=2,
        )

    raylight = _runtime_copy(standard, select_raylight)
    raylight_residency_changed = _runtime_copy(
        raylight,
        lambda document: document.update(
            raylight_residency_policy="release_after_sampling"
        ),
    )
    raylight_gate_changed = _runtime_copy(
        raylight,
        lambda document: document.update(multi_gpu_enabled=False),
    )
    expected = project_v5_runtime_currentness(
        draft, ["fl-segment"], raylight
    )

    assert project_v5_runtime_currentness(
        draft, ["fl-segment"], raylight_residency_changed
    ) != expected
    assert project_v5_runtime_currentness(
        draft, ["fl-segment"], raylight_gate_changed
    ) != expected


def test_runtime_currentness_includes_audio_vae_only_when_route_uses_it() -> None:
    draft = _complete_v5_draft()
    baseline = default_runtime_settings_v3()
    changed = _runtime_copy(
        baseline,
        lambda document: document["placement"].update(
            audio_vae_device="gpu:4"
        ),
    )

    # Muted FL2VA does not load an audio VAE.
    assert project_v5_runtime_currentness(
        draft, ["fl-segment"], baseline
    ) == project_v5_runtime_currentness(draft, ["fl-segment"], changed)

    generated_audio = draft.model_dump(mode="json")
    generated_audio["segments"][0]["audio_mode"] = "generate"
    generated_audio_draft = UnifiedTimelineDraftV5.model_validate(generated_audio)
    assert project_v5_runtime_currentness(
        generated_audio_draft, ["fl-segment"], baseline
    ) != project_v5_runtime_currentness(
        generated_audio_draft, ["fl-segment"], changed
    )

    # Ref2VA's shared model route requires the audio VAE even when its output
    # soundtrack is muted.
    assert project_v5_runtime_currentness(
        draft, ["ref-segment"], baseline
    ) != project_v5_runtime_currentness(draft, ["ref-segment"], changed)


def test_runtime_projection_omits_complete_lora_mapping_table() -> None:
    document = _complete_v5_draft().model_dump(mode="json")
    fl_model = document["model_stack"]["fl2va"]["filename"]
    document["features"]["project"]["lora"] = {
        "enabled": True,
        "params": {
            "by_family": {
                "fl2va": {
                    "enabled": True,
                    "filename": "active-fl.safetensors",
                    "strength": 0.75,
                },
                "ref2va": {
                    "enabled": False,
                    "filename": "parked-ref.safetensors",
                    "strength": 1.0,
                },
            }
        },
    }
    draft = UnifiedTimelineDraftV5.model_validate(document)
    baseline = _runtime_copy(
        default_runtime_settings_v3(),
        lambda runtime: runtime.update(
            lora_loader_overrides=[
                {
                    "family": "fl2va",
                    "model_filename": fl_model,
                    "lora_filename": "active-fl.safetensors",
                    "adapter_id": "model_only",
                }
            ]
        ),
    )
    expected = project_v5_runtime_currentness(
        draft,
        ["fl-segment"],
        baseline,
    )
    assert not hasattr(expected, "lora_loader_overrides")

    unrelated = _runtime_copy(
        baseline,
        lambda runtime: runtime.update(
            lora_loader_overrides=[
                *runtime["lora_loader_overrides"],
                {
                    "family": "ref2va",
                    "model_filename": "other-model.safetensors",
                    "lora_filename": "other-lora.safetensors",
                    "adapter_id": "dedicated",
                },
            ]
        ),
    )
    assert (
        project_v5_runtime_currentness(draft, ["fl-segment"], unrelated)
        == expected
    )

    matching_adapter_changed = _runtime_copy(
        baseline,
        lambda runtime: runtime["lora_loader_overrides"][0].update(
            adapter_id="dedicated"
        ),
    )
    # Mapping resolution and adapter identity are captured separately by the
    # bounded job snapshot; the placement projection never copies mappings.
    assert (
        project_v5_runtime_currentness(
            draft,
            ["fl-segment"],
            matching_adapter_changed,
        )
        == expected
    )


def _seed_readable_v5_job(database, draft, settings: RuntimeSettingsV3) -> str:
    now = "2026-08-21T12:00:00+00:00"
    job_id = "v5-runtime-currentness-job"
    segment_id = "fl-segment"
    plan = compile_v5_execution_plan(
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
        segment_child_id="v5-runtime-currentness-child",
    )
    database.persist_job_child_submission_intent(
        job_id,
        locked_plan=locked,
        exact_snapshot=planner.exact_snapshot(locked, locked.units[0]),
    )
    _persist_observed_success(
        database,
        "v5-runtime-currentness-child",
        output=OutputDescriptor(
            filename="currentness.mp4",
            subfolder="segments",
        ),
        has_audio=False,
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


async def test_job_currentness_uses_relevant_v5_runtime_projection(client) -> None:
    database = client.director_app.state.database
    draft = _complete_v5_draft()
    _install_frozen_v5_authority(database, draft)
    captured = database.get_settings()
    job_id = _seed_readable_v5_job(database, draft, captured)

    current = await client.get(f"/api/jobs/{job_id}")
    assert current.status_code == 200, current.text
    assert current.json()["segment_results"][0]["current_snapshot"] is True

    unrelated = _runtime_copy(
        captured,
        lambda document: document["placement"]["ref2va"].update(
            device="gpu:7"
        ),
    )
    _, token = database.get_settings_authority()
    database.put_settings_v3_authority(
        unrelated,
        expected_authority_token=token,
        schema_version=3,
    )
    still_current = await client.get(f"/api/jobs/{job_id}")
    assert still_current.status_code == 200, still_current.text
    assert (
        still_current.json()["segment_results"][0]["current_snapshot"] is True
    )

    standard_globals = _runtime_copy(
        unrelated,
        lambda document: document.update(
            multi_gpu_enabled=True,
            raylight_residency_policy="release_after_sampling",
        ),
    )
    _, token = database.get_settings_authority()
    database.put_settings_v3_authority(
        standard_globals,
        expected_authority_token=token,
        schema_version=3,
    )
    still_current = await client.get(f"/api/jobs/{job_id}")
    assert still_current.status_code == 200, still_current.text
    assert (
        still_current.json()["segment_results"][0]["current_snapshot"] is True
    )

    relevant = _runtime_copy(
        standard_globals,
        lambda document: document["placement"]["fl2va"].update(
            device="gpu:1"
        ),
    )
    _, token = database.get_settings_authority()
    database.put_settings_v3_authority(
        relevant,
        expected_authority_token=token,
        schema_version=3,
    )
    stale = await client.get(f"/api/jobs/{job_id}")
    assert stale.status_code == 200, stale.text
    assert stale.json()["segment_results"][0]["current_snapshot"] is False


async def test_historical_v4_job_currentness_remains_fail_closed(client) -> None:
    database = client.director_app.state.database
    draft = _complete_v5_draft()
    _install_frozen_v5_authority(database, draft)
    job_id = _seed_readable_v5_job(database, draft, database.get_settings())

    legacy_document = draft.model_dump(
        mode="json",
        exclude={"model_stack", "features"},
    )
    legacy_document["version"] = 4
    legacy = UnifiedTimelineDraftV4.model_validate(legacy_document)
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET config_snapshot = ?, settings_snapshot = ? "
            "WHERE id = ?",
            (
                json.dumps(
                    {
                        "timeline": legacy.model_dump(mode="json"),
                        "segment_ids": ["fl-segment"],
                    },
                    ensure_ascii=False,
                ),
                default_settings().model_dump_json(),
                job_id,
            ),
        )

    historical = await client.get(f"/api/jobs/{job_id}")
    assert historical.status_code == 200, historical.text
    # The historical resolver can prove this v4 creative snapshot describes
    # the same project, but it cannot mint a v5 runtime-currentness authority.
    assert historical.json()["current_project"] is True
    assert historical.json()["segment_results"][0]["current_snapshot"] is False
