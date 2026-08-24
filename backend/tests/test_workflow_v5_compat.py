from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from directordeck.native_templates import NativeTemplateError
from directordeck.schemas import (
    RuntimeSettingsV1,
    RuntimeSettingsV3,
    UnifiedTimelineDraftV4,
    UnifiedTimelineDraftV5,
    default_settings,
    default_timeline_draft,
)
from directordeck.workflow.v4_execution_adapter import (
    V4ExecutionAdapterError,
    compile_v4_execution_plan,
)
from directordeck.workflow.v5_compat import (
    V5CreativeAuthorityError,
    compile_v5_execution_plan,
    project_v5_compile_authority,
)

from . import extensible_workflow_v0_fixture_builder as fixture_builder


def _v4_pair() -> tuple[UnifiedTimelineDraftV4, RuntimeSettingsV1]:
    draft = default_timeline_draft()
    draft.segments[0].prompt = "A quiet lake at sunrise"
    return draft, default_settings()


def _v5_pair(
    draft: UnifiedTimelineDraftV4,
    settings: RuntimeSettingsV1,
) -> tuple[UnifiedTimelineDraftV5, RuntimeSettingsV3]:
    document = draft.model_dump(mode="json")
    document.update(
        {
            "version": 5,
            "model_stack": {
                role: {"filename": getattr(settings.models, role).filename}
                for role in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae")
            },
            "features": {
                "template_bundle_version": 4,
                "project": {
                    "lora": {
                        "enabled": any(
                            getattr(settings.models, family).lora_name is not None
                            for family in ("fl2va", "ref2va")
                        ),
                        "params": {
                            "by_family": {
                                family: {
                                    "enabled": (
                                        getattr(settings.models, family).lora_name
                                        is not None
                                    ),
                                    "filename": getattr(
                                        settings.models, family
                                    ).lora_name,
                                    "strength": getattr(
                                        settings.models, family
                                    ).lora_strength,
                                }
                                for family in ("fl2va", "ref2va")
                            }
                        },
                    }
                },
                "by_segment": {},
            },
        }
    )
    overrides = []
    for family in ("fl2va", "ref2va"):
        binding = getattr(settings.models, family)
        override = binding.standard_lora_loader_override
        if override is not None:
            overrides.append(
                {
                    "family": family,
                    "model_filename": override.model_filename,
                    "lora_filename": override.lora_name,
                    "adapter_id": override.loader,
                }
            )
    runtime = RuntimeSettingsV3.model_validate(
        {
            "schema_version": 3,
            "client_id": settings.client_id,
            "memory_policy": settings.memory_policy,
            "raylight_residency_policy": settings.raylight_residency_policy,
            "multi_gpu_enabled": settings.multi_gpu_enabled,
            "placement": {
                "fl2va": {
                    "device": settings.models.fl2va.device,
                    "raylight": settings.models.fl2va.raylight.model_dump(mode="json"),
                },
                "ref2va": {
                    "device": settings.models.ref2va.device,
                    "raylight": settings.models.ref2va.raylight.model_dump(mode="json"),
                },
                "clip_device": settings.models.clip.device,
                "video_vae_device": settings.models.video_vae.device,
                "audio_vae_device": settings.models.audio_vae.device,
            },
            "lora_loader_overrides": overrides,
        }
    )
    return UnifiedTimelineDraftV5.model_validate(document), runtime


def test_v5_projection_reconstructs_exact_v4_authorities() -> None:
    v4, settings_v1 = _v4_pair()
    v5, settings_v2 = _v5_pair(v4, settings_v1)

    projection = project_v5_compile_authority(v5, settings_v2)

    assert projection.draft.model_dump(mode="json") == v4.model_dump(mode="json")
    assert projection.settings.model_dump(mode="json") == settings_v1.model_dump(
        mode="json"
    )


def test_v5_projection_preserves_exact_manual_lora_override() -> None:
    v4, settings_v1 = _v4_pair()
    raw_settings = settings_v1.model_dump(mode="json")
    raw_settings["models"]["fl2va"].update(
        {
            "lora_name": "nested/custom.safetensors",
            "lora_strength": 0.75,
            "standard_lora_loader_override": {
                "loader": "model_only",
                "lora_name": "nested/custom.safetensors",
                "model_filename": raw_settings["models"]["fl2va"]["filename"],
            },
        }
    )
    settings_v1 = RuntimeSettingsV1.model_validate(raw_settings)
    v5, settings_v2 = _v5_pair(v4, settings_v1)

    projection = project_v5_compile_authority(v5, settings_v2)

    assert projection.settings.model_dump(mode="json") == settings_v1.model_dump(
        mode="json"
    )


def test_v5_default_graph_is_v4_compatible_but_uses_current_bundle_identity() -> None:
    v4, settings_v1 = _v4_pair()
    v5, settings_v2 = _v5_pair(v4, settings_v1)

    legacy = compile_v4_execution_plan(v4, settings_v1, "same-job")
    migrated = compile_v5_execution_plan(v5, settings_v2, "same-job")

    assert legacy.template_bundle_version == 4
    assert migrated.template_bundle_version == 5
    assert tuple(
        unit.prompt_base
        for unit in migrated.segment_units
    ) == tuple(
        unit.prompt_base
        for unit in legacy.segment_units
    )
    assert tuple(
        unit.graph_audit_spec.model_dump(mode="json")
        for unit in migrated.segment_units
    ) == tuple(
        unit.graph_audit_spec.model_dump(mode="json")
        for unit in legacy.segment_units
    )
    assert migrated.effective_execution_digest == legacy.effective_execution_digest
    migrated_progress = migrated.segment_units[0].progress_spec
    legacy_progress = legacy.segment_units[0].progress_spec
    assert migrated_progress is not None
    assert legacy_progress is not None
    assert tuple(
        phase for phase in migrated_progress.phases if phase.kind != "stage"
    ) == legacy_progress.phases
    stage_hints = tuple(
        phase for phase in migrated_progress.phases if phase.kind == "stage"
    )
    assert stage_hints
    assert all(phase.weight == 0.0 for phase in stage_hints)
    assert migrated.segment_units[0].preview_spec == (
        legacy.segment_units[0].preview_spec
    )


def test_v5_strict_feature_does_not_read_the_moving_current_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import directordeck.workflow.node_contracts as node_contracts

    v4, settings_v1 = _v4_pair()
    v5, settings_v3 = _v5_pair(v4, settings_v1)
    document = v5.model_dump(mode="json")
    document["features"]["template_bundle_version"] = 5
    document["features"]["project"]["attention_backend_override"] = {
        "enabled": True,
        "params": {"mode": "pytorch"},
    }

    # Simulate a later default-bundle cutover. Bundle-5 compilation must still
    # resolve its strict node from the frozen V5 registry.
    monkeypatch.setattr(
        node_contracts,
        "CURRENT_NODE_CONTRACT_REGISTRY",
        node_contracts.V4_NODE_CONTRACT_REGISTRY,
    )
    plan = compile_v5_execution_plan(
        UnifiedTimelineDraftV5.model_validate(document),
        settings_v3,
        "frozen-v5-registry",
    )

    strict_nodes = [
        node
        for node in plan.segment_units[0].prompt_base.values()
        if node["class_type"] == "DirectorStrictModelAttentionBackend"
    ]
    assert len(strict_nodes) == 1
    assert strict_nodes[0]["inputs"]["mode"] == "pytorch"


def test_v5_adapter_never_falls_back_when_explicit_execution_specs_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import directordeck.workflow.v5_compat as v5_compat

    v4, settings_v1 = _v4_pair()
    v5, settings_v3 = _v5_pair(v4, settings_v1)
    actual_compile = v5_compat.compile_projected_v5_timeline

    def compile_without_execution_specs(*args: object, **kwargs: object):
        result = actual_compile(*args, **kwargs)
        return replace(
            result,
            workflows=tuple(
                replace(unit, progress_spec=None, preview_spec=None)
                for unit in result.workflows
            ),
        )

    monkeypatch.setattr(
        v5_compat,
        "compile_projected_v5_timeline",
        compile_without_execution_specs,
    )

    with pytest.raises(V4ExecutionAdapterError, match="no explicit"):
        v5_compat.compile_v5_execution_plan(v5, settings_v3, "missing-specs")


@pytest.mark.parametrize(
    ("lora_name", "override"),
    (
        ("minimax_h3_turbo_v4_step600_ema.safetensors", "dedicated"),
        (
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "model_only",
        ),
    ),
    ids=("minimax_h3_turbo", "model_only"),
)
def test_v5_active_standard_lora_preserves_v4_graph_and_refreshes_identity(
    lora_name: str,
    override: str | None,
) -> None:
    v4 = fixture_builder._draft("t2v")
    settings_v1 = fixture_builder._settings(
        "standard",
        lora_family="fl2va",
        lora_name=lora_name,
        lora_strength=0.75,
        standard_override=override,
    )
    v5, settings_v3 = _v5_pair(v4, settings_v1)
    projection = project_v5_compile_authority(v5, settings_v3)

    legacy = compile_v4_execution_plan(
        v4,
        settings_v1,
        "same-lora-job",
        resolved_lora_adapters=projection.lora_adapter_map(),
    )
    current = compile_v5_execution_plan(v5, settings_v3, "same-lora-job")

    assert tuple(unit.prompt_base for unit in current.segment_units) == tuple(
        unit.prompt_base for unit in legacy.segment_units
    )
    assert tuple(
        unit.graph_audit_spec.model_dump(mode="json")
        for unit in current.segment_units
    ) == tuple(
        unit.graph_audit_spec.model_dump(mode="json")
        for unit in legacy.segment_units
    )
    # Current execution identity records the configured loader options.  The
    # frozen v4 identity predates that evidence, even when both prompt graphs
    # happen to be byte-for-byte equal.
    assert current.effective_execution_digest != legacy.effective_execution_digest


def test_v5_minimax_lora_low_vram_is_compiled_and_changes_execution_identity() -> None:
    lora_name = "minimax_h3_turbo_v4_step600_ema.safetensors"
    v4 = fixture_builder._draft("t2v")
    settings_v1 = fixture_builder._settings(
        "standard",
        lora_family="fl2va",
        lora_name=lora_name,
        lora_strength=0.75,
        standard_override="dedicated",
    )
    v5, settings_v3 = _v5_pair(v4, settings_v1)

    def compile_with(low_vram: bool):
        raw = settings_v3.model_dump(mode="json")
        raw["lora_loader_overrides"] = [
            {
                "lora_filename": lora_name,
                "adapter_id": "minimax_h3_turbo",
                "options": {"low_vram": low_vram},
            }
        ]
        return compile_v5_execution_plan(
            v5,
            RuntimeSettingsV3.model_validate(raw),
            "same-lora-options-job",
        )

    normal = compile_with(False)
    low_memory = compile_with(True)

    def loader_input(plan, name: str) -> object:
        node = next(
            node
            for unit in plan.segment_units
            for node in unit.prompt_base.values()
            if node["class_type"] == "MiniMaxH3TurboLoRA"
        )
        return node["inputs"][name]

    assert loader_input(normal, "low_vram") is False
    assert loader_input(low_memory, "low_vram") is True
    assert normal.effective_execution_digest != low_memory.effective_execution_digest


def test_v5_raylight_pool_revision_explicitly_invalidates_v4_identity() -> None:
    v4, settings_v1 = _v4_pair()
    raw = settings_v1.model_dump(mode="json")
    raw["multi_gpu_enabled"] = True
    raw["models"]["fl2va"]["raylight"].update(
        gpu_select=[0, 1],
        ulysses_degree=2,
        ring_degree=1,
        cfg_degree=1,
        dp_degree=1,
        fsdp=False,
        cpu_offload=False,
    )
    settings_v1 = RuntimeSettingsV1.model_validate(raw)
    v5, settings_v3 = _v5_pair(v4, settings_v1)

    legacy = compile_v4_execution_plan(v4, settings_v1, "same-ray-job")
    current = compile_v5_execution_plan(v5, settings_v3, "same-ray-job")

    assert legacy.segment_units[0].backend == "raylight"
    assert current.segment_units[0].backend == "raylight"
    assert current.effective_execution_digest != legacy.effective_execution_digest
    legacy_pool = legacy.segment_units[0].runtime_pool_identity
    current_pool = current.segment_units[0].runtime_pool_identity
    assert legacy_pool is not None and current_pool is not None
    assert legacy_pool.active_feature_pool_identities[0].feature == (
        "raylight_pool_intent@1"
    )
    assert current_pool.active_feature_pool_identities[0].feature == (
        "raylight_pool_intent@2"
    )
    assert current_pool.active_feature_pool_identities[0].identity[
        "attention"
    ] == "ck_int8"


def test_raylight_attention_topology_guard_is_v5_only() -> None:
    v4, settings_v1 = _v4_pair()
    raw_settings = settings_v1.model_dump(mode="json")
    raw_settings["multi_gpu_enabled"] = True
    raw_settings["models"]["fl2va"]["raylight"].update(
        gpu_select=[0, 1],
        ulysses_degree=1,
        ring_degree=2,
        cfg_degree=1,
        dp_degree=1,
        fsdp=False,
        cpu_offload=False,
    )
    settings_v1 = RuntimeSettingsV1.model_validate(raw_settings)
    v5, settings_v3 = _v5_pair(v4, settings_v1)

    legacy = compile_v4_execution_plan(v4, settings_v1, "legacy-ring-ck")
    legacy_initializer = next(
        node
        for node in legacy.segment_units[0].prompt_base.values()
        if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
    )
    assert legacy_initializer["inputs"]["XFuser_attention"] == (
        "COMFY_KITCHEN_INT8"
    )

    with pytest.raises(NativeTemplateError, match="requires ring_degree=1"):
        compile_v5_execution_plan(v5, settings_v3, "current-ring-ck")

    raw_v5 = v5.model_dump(mode="json")
    raw_v5["features"]["template_bundle_version"] = 5
    raw_v5["features"]["project"]["raylight_pool_intent"] = {
        "enabled": True,
        "params": {"attention": "torch_flash"},
    }
    flash = compile_v5_execution_plan(
        UnifiedTimelineDraftV5.model_validate(raw_v5),
        settings_v3,
        "current-ring-flash",
    )
    flash_initializer = next(
        node
        for node in flash.segment_units[0].prompt_base.values()
        if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
    )
    assert flash_initializer["inputs"]["XFuser_attention"] == "TORCH_FLASH"


def test_v5_incomplete_model_stack_fails_closed() -> None:
    v4, settings_v1 = _v4_pair()
    v5, settings_v2 = _v5_pair(v4, settings_v1)
    value = v5.model_dump(mode="json")
    value["model_stack"]["clip"]["filename"] = None

    with pytest.raises(V5CreativeAuthorityError) as caught:
        project_v5_compile_authority(
            UnifiedTimelineDraftV5.model_validate(value),
            settings_v2,
        )

    assert caught.value.code == "model_binding_required"
    assert caught.value.safe_details == {"bindings": ["clip"]}


def test_v5_enabled_lora_slot_requires_filename() -> None:
    v4, settings_v1 = _v4_pair()
    v5, settings_v2 = _v5_pair(v4, settings_v1)
    value = v5.model_dump(mode="json")
    value["features"]["project"]["lora"]["enabled"] = True
    value["features"]["project"]["lora"]["params"]["by_family"]["fl2va"] = {
        "enabled": True,
        "filename": None,
        "strength": 1.0,
    }

    with pytest.raises(V5CreativeAuthorityError) as caught:
        project_v5_compile_authority(
            UnifiedTimelineDraftV5.model_validate(value),
            settings_v2,
        )

    assert caught.value.code == "lora_binding_required"
    assert caught.value.safe_details == {"family": "fl2va"}


def test_v5_active_segment_lora_override_is_not_silently_applied() -> None:
    v4, settings_v1 = _v4_pair()
    v5, settings_v2 = _v5_pair(v4, settings_v1)
    value = v5.model_dump(mode="json")
    value["features"]["by_segment"][v5.segments[0].id] = {
        "lora": copy.deepcopy(value["features"]["project"]["lora"])
    }
    value["features"]["by_segment"][v5.segments[0].id]["lora"][
        "enabled"
    ] = True

    with pytest.raises(V5CreativeAuthorityError) as caught:
        project_v5_compile_authority(
            UnifiedTimelineDraftV5.model_validate(value),
            settings_v2,
        )

    assert caught.value.code == "feature_scope_unsupported"
    assert caught.value.segment_id == v5.segments[0].id


def test_v5_unknown_feature_is_rejected() -> None:
    v4, settings_v1 = _v4_pair()
    v5, settings_v2 = _v5_pair(v4, settings_v1)
    value = v5.model_dump(mode="json")
    value["features"]["project"]["future"] = {
        "enabled": False,
        "params": {},
    }

    with pytest.raises(V5CreativeAuthorityError) as caught:
        project_v5_compile_authority(
            UnifiedTimelineDraftV5.model_validate(value),
            settings_v2,
        )

    assert caught.value.code == "unknown_feature"
