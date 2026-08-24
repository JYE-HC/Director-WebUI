from __future__ import annotations

import pytest

from directordeck.execution.submission import LockedSubmissionPlanner
from directordeck.schemas import RuntimeSettingsV3, UnifiedTimelineDraftV5
from directordeck.workflow.audit import GraphAuditError, validate_bound_graph
from directordeck.workflow.execution import CompiledExecutionPlan
from directordeck.workflow.node_contracts import V6_NODE_CONTRACT_REGISTRY
from directordeck.workflow.runtime_snapshot import (
    JobRuntimeSnapshotV1,
    build_job_runtime_snapshot,
)
from directordeck.workflow.v5_compat import compile_v5_execution_plan
from directordeck.workflow.v6_execution_adapter import compile_v6_execution_plan
from directordeck.workflow.v6_projection import project_v5_authority_to_v6

from . import extensible_workflow_v0_fixture_builder as fixture_builder
from .test_stage1a_v6_domain import _pair
from .test_workflow_execution_contracts import endpoint_identity
from .test_workflow_v5_compat import _v4_pair, _v5_pair


def _bundle5_pair() -> tuple[UnifiedTimelineDraftV5, RuntimeSettingsV3]:
    v4, settings_v1 = _v4_pair()
    draft, settings = _v5_pair(v4, settings_v1)
    document = draft.model_dump(mode="json")
    document["features"]["template_bundle_version"] = 5
    return UnifiedTimelineDraftV5.model_validate(document), settings


def _ray_settings(settings: RuntimeSettingsV3) -> RuntimeSettingsV3:
    document = settings.model_dump(mode="json")
    document["multi_gpu_enabled"] = True
    for family in ("fl2va", "ref2va"):
        document["placement"][family]["raylight"].update(
            {
                "gpu_select": [0, 1],
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_degree": 1,
                "dp_degree": 1,
                "fsdp": False,
                "cpu_offload": False,
            }
        )
    return RuntimeSettingsV3.model_validate(document)


def _bundle6_recipe_pair(
    recipe: str,
    *,
    audio_mode: str,
) -> tuple[UnifiedTimelineDraftV5, RuntimeSettingsV3]:
    draft, settings = _v5_pair(
        fixture_builder._draft(recipe, audio_mode=audio_mode),
        fixture_builder._settings("standard"),
    )
    document = draft.model_dump(mode="json")
    document["features"]["template_bundle_version"] = 6
    document["features"]["project"]["comfy_kitchen_attention"] = {
        "enabled": False,
        "params": {},
    }
    return UnifiedTimelineDraftV5.model_validate(document), settings


def _with_audio_vae(
    draft: UnifiedTimelineDraftV5,
    filename: str,
) -> UnifiedTimelineDraftV5:
    document = draft.model_dump(mode="json")
    document["model_stack"]["audio_vae"]["filename"] = filename
    return UnifiedTimelineDraftV5.model_validate(document)


def _progress_semantics(unit: object) -> list[tuple[object, ...]]:
    progress = unit.progress_spec  # type: ignore[attr-defined]
    return [
        (phase.kind, phase.label, phase.node_id, phase.weight)
        for phase in progress.phases
    ]


def test_default_standard_shadow_compile_preserves_exact_graph_and_outputs() -> None:
    source, settings = _bundle5_pair()
    target = project_v5_authority_to_v6(source).draft

    v5 = compile_v5_execution_plan(source, settings, "shadow-v5")
    v6 = compile_v6_execution_plan(target, settings, "shadow-v6")

    assert len(v5.segment_units) == len(v6.segment_units) == 1
    old, new = v5.segment_units[0], v6.segment_units[0]
    assert new.prompt_base == old.prompt_base
    assert new.expected_output_spec == old.expected_output_spec
    assert v6.compile_report.manifest["submission_order"] == (
        v5.compile_report.manifest["submission_order"]
    )
    assert _progress_semantics(new) == _progress_semantics(old)
    assert not any(
        node["class_type"] == "ModelAttentionBackend"
        for node in new.prompt_base.values()
    )


def test_plan_v3_and_report_v3_round_trip_without_reinterpretation() -> None:
    draft, settings = _pair()
    plan = compile_v6_execution_plan(draft, settings, "v6-plan-round-trip")
    serialized = plan.model_dump_json()

    restored = CompiledExecutionPlan.model_validate_json(serialized)

    assert restored == plan
    assert restored.version == 3
    assert restored.template_bundle_version == 6
    assert restored.compile_report.source == "bundle6_native_compile_v3"
    assert len(restored.compile_report.feature_resolutions) == 12
    assert '"python_module"' not in serialized
    assert '"runtime_fingerprint"' not in serialized


def test_bundle6_standard_plan_materializes_with_bundle6_registry() -> None:
    draft, settings = _pair()
    plan = compile_v6_execution_plan(draft, settings, "v6-standard-lock")
    planner = LockedSubmissionPlanner(endpoint_identity(endpoint_key="embedded"))

    locked = planner.build_wave(
        plan,
        source_unit_ordinal=0,
        segment_child_id="v6-standard-child",
    )

    assert len(locked.units) == 1
    assert locked.units[0].exact_prompt == plan.segment_units[0].prompt_base
    assert planner.exact_snapshot(locked, locked.units[0]).template_revision == 6


def test_bundle6_adapter_contract_digest_is_internal_graph_evidence() -> None:
    draft, settings = _pair()
    unit = compile_v6_execution_plan(
        draft, settings, "v6-adapter-contract"
    ).segment_units[0]
    snapshot = dict(unit.graph_audit_spec.node_contract_snapshot)
    node_id = next(iter(snapshot))
    snapshot[node_id] = snapshot[node_id].model_copy(
        update={"adapter_contract_digest": "sha256:" + "0" * 64}
    )
    drifted = unit.graph_audit_spec.model_copy(
        update={"node_contract_snapshot": snapshot}
    )

    with pytest.raises(GraphAuditError, match="Director adapter contract digest"):
        validate_bound_graph(
            prompt_base=unit.prompt_base,
            bound_prompt=unit.prompt_base,
            spec=drifted,
            node_contract_registry=V6_NODE_CONTRACT_REGISTRY,
            model_family=unit.family,
            backend=unit.backend,
            enforce_runtime_effects=False,
        )


def test_bundle6_raylight_plan_materializes_native_attention_descriptor() -> None:
    draft, base_settings = _pair()
    settings = _ray_settings(base_settings)
    plan = compile_v6_execution_plan(draft, settings, "v6-ray-lock")
    planner = LockedSubmissionPlanner(endpoint_identity(endpoint_key="embedded"))

    locked = planner.build_wave(
        plan,
        source_unit_ordinal=0,
        segment_child_id="v6-ray-child",
    )

    assert len(locked.units) == 1
    initializer_id, initializer = next(
        (node_id, node)
        for node_id, node in locked.units[0].exact_prompt.items()
        if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
    )
    assert initializer["inputs"]["XFuser_attention"] == "TORCH_FLASH"
    pointer = f"/{initializer_id}/inputs/ray_cluster_namespace"
    assert initializer["inputs"]["ray_cluster_namespace"] == (
        locked.units[0].late_bound_values[pointer]
    )
    assert initializer["inputs"]["ray_cluster_namespace"].endswith("-e1")
    assert plan.segment_units[0].runtime_pool_identity.fixed_parameters[
        "attention_mode"
    ] == "TORCH_FLASH"


def test_bundle6_runtime_snapshot_round_trips_without_v5_projection() -> None:
    draft, settings = _pair()
    plan = compile_v6_execution_plan(draft, settings, "v6-runtime-snapshot")

    snapshot = build_job_runtime_snapshot(draft, None, settings, plan)
    restored = JobRuntimeSnapshotV1.model_validate_json(
        snapshot.model_dump_json()
    )

    assert restored == snapshot
    assert restored.runtime_projection.families[0].backend == "standard"
    assert restored.resolved_lora_adapters == ()

    changed = settings.model_copy(update={"client_id": "other-client"})
    changed_snapshot = build_job_runtime_snapshot(draft, None, changed, plan)
    assert snapshot.has_same_execution_identity(changed_snapshot)


@pytest.mark.parametrize(
    ("recipe", "audio_mode", "audio_vae_is_active"),
    (
        ("t2v", "mute", False),
        ("t2v", "generate", True),
        ("r2v", "mute", True),
    ),
)
def test_bundle6_digest_includes_audio_vae_only_when_the_unit_uses_it(
    recipe: str,
    audio_mode: str,
    audio_vae_is_active: bool,
) -> None:
    draft, settings = _bundle6_recipe_pair(recipe, audio_mode=audio_mode)
    changed = _with_audio_vae(draft, "changed-audio-vae.safetensors")

    baseline_digest = compile_v6_execution_plan(
        draft, settings, "audio-vae-active-only"
    ).segment_units[0].effective_execution_digest
    changed_digest = compile_v6_execution_plan(
        changed, settings, "audio-vae-active-only"
    ).segment_units[0].effective_execution_digest

    assert (baseline_digest != changed_digest) is audio_vae_is_active
