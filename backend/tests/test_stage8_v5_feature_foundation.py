from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone

import pytest

from directordeck.capabilities import (
    CapabilityEvaluator,
    build_feature_catalog,
    build_operational_readiness,
    preflight_projected_v5_timeline,
)
from directordeck.native_templates import NativeTemplateError
from directordeck.schemas import (
    RuntimeSettingsV3,
    UnifiedTimelineDraftV5,
    default_timeline_draft_v5,
)
from directordeck.workflow.contracts import (
    HostCapabilitySnapshot,
    LogicalGpuCapability,
    MediaToolCapability,
    PackageCapability,
    RayLightInstallation,
    RuntimeProbeEvidence,
)
from directordeck.workflow.effective_features import (
    EffectiveFeatureConfiguration,
    V5FeatureConfigurationError,
    migrate_feature_configuration_to_v5,
    resolve_v5_effective_features,
)
from directordeck.workflow.templates import (
    V4_TEMPLATE_BUNDLE,
    V5_TEMPLATE_BUNDLE,
)
from directordeck.workflow.node_contracts import CURRENT_NODE_CONTRACT_REGISTRY
from directordeck.workflow.v5_compat import (
    compile_v5_execution_plan,
    project_v5_compile_authority,
)
from directordeck.workflow.v4_compiler import (
    EffectiveFeatureResolutionMismatch,
    compile_projected_v5_timeline,
)

from .test_workflow_v5_compat import _v4_pair, _v5_pair


_V4_TEMPLATE_CANONICAL_SHA256 = (
    "ba13e513fea9d1a1fad6de5765ca5d63040efed26a8f6a538f42cbd98178ef24"
)


def _document(*, bundle_version: int = 5) -> dict:
    value = default_timeline_draft_v5().model_dump(mode="json")
    value["features"]["template_bundle_version"] = bundle_version
    return value


def _draft(value: dict) -> UnifiedTimelineDraftV5:
    return UnifiedTimelineDraftV5.model_validate(value)


def _resolve(
    value: dict,
    *,
    backend: str = "standard",
):
    draft = _draft(value)
    return resolve_v5_effective_features(
        draft,
        selected_segment_ids=(draft.segments[0].id,),
        backend_by_family={"fl2va": backend, "ref2va": backend},
    )


def _raylight_ring_pair(
    attention: str,
) -> tuple[UnifiedTimelineDraftV5, RuntimeSettingsV3]:
    v4, settings_v1 = _v4_pair()
    v5, settings_v3 = _v5_pair(v4, settings_v1)
    raw_v5 = v5.model_dump(mode="json")
    raw_v5["features"]["template_bundle_version"] = 5
    raw_v5["features"]["project"]["raylight_pool_intent"] = {
        "enabled": True,
        "params": {"attention": attention},
    }
    raw_settings = settings_v3.model_dump(mode="json")
    raw_settings["multi_gpu_enabled"] = True
    raw_settings["placement"]["fl2va"]["raylight"].update(
        gpu_select=[0, 1],
        ulysses_degree=1,
        ring_degree=2,
        cfg_degree=1,
        dp_degree=1,
        fsdp=False,
        cpu_offload=False,
    )
    return (
        UnifiedTimelineDraftV5.model_validate(raw_v5),
        RuntimeSettingsV3.model_validate(raw_settings),
    )


def _current_snapshot(*, strict_runtime: bool = False) -> HostCapabilitySnapshot:
    node_registry: dict[str, str] = {}
    object_info: dict[str, object] = {}
    module_fingerprints: dict[str, str] = {}
    for contract in CURRENT_NODE_CONTRACT_REGISTRY.contracts.values():
        module = contract.allowed_python_modules[0]
        node_registry[contract.class_type] = module
        object_info[contract.class_type] = contract.object_info_contract
        fingerprint = contract.supported_runtime_fingerprints[0]
        previous = module_fingerprints.setdefault(module, fingerprint)
        assert previous == fingerprint
    return HostCapabilitySnapshot(
        schema_version=2 if strict_runtime else 1,
        generated_at=datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc),
        node_registry=node_registry,
        object_info_slices=object_info,
        module_fingerprints=module_fingerprints,
        importable_packages={
            name: PackageCapability(importable=True, version="1.0.0")
            for name in ("ray", "xfuser")
        },
        gpu_inventory=tuple(
            LogicalGpuCapability(logical_index=index, backend="cuda")
            for index in range(4)
        ),
        raylight_installation=RayLightInstallation(
            installed=True,
            node_contracts_available=True,
            reason_codes=(),
        ),
        media_tool_status={
            name: MediaToolCapability(available=True, version="7.1")
            for name in ("ffmpeg", "ffprobe")
        },
        runtime_probe_evidence=(
            {
                key: RuntimeProbeEvidence(available=True, code="available")
                for key in (
                    "strict_attention.pytorch",
                    "strict_attention.ck_int8.any",
                    "strict_attention.ck_int8.default",
                    "strict_h3_sage.any",
                    "strict_h3_sage.default",
                )
            }
            if strict_runtime
            else {}
        ),
    )


def test_v4_template_contract_remains_byte_frozen_while_v5_is_separate() -> None:
    payload = V4_TEMPLATE_BUNDLE.model_dump_json().encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == _V4_TEMPLATE_CANONICAL_SHA256
    assert V4_TEMPLATE_BUNDLE.version == 4
    assert V5_TEMPLATE_BUNDLE.version == 5
    assert V5_TEMPLATE_BUNDLE is not V4_TEMPLATE_BUNDLE
    assert V5_TEMPLATE_BUNDLE.segment_templates.standard.revision == 1


def test_v5_catalog_contract_has_stable_visibility_scopes_and_param_defaults() -> None:
    entries = {
        entry.id: entry
        for template in (
            V5_TEMPLATE_BUNDLE.segment_templates.standard,
            V5_TEMPLATE_BUNDLE.segment_templates.raylight,
        )
        for entry in template.entries
    }
    assert entries["attention_backend_override"].defaults == {
        "mode": "pytorch"
    }
    assert entries["attention_backend_override"].ui == {"visibility": "user"}
    assert entries["attention_backend_override"].scopes == (
        "project",
        "segment",
    )
    assert entries["h3_low_vram_attention"].defaults == {}
    assert entries["attention_backend_override"].conflicts == (
        "h3_low_vram_attention",
    )
    assert entries["h3_low_vram_attention"].conflicts == (
        "attention_backend_override",
    )
    assert entries["lora"].scopes == ("project",)
    ray_pool = entries["raylight_pool_intent"]
    assert ray_pool.version == 2
    assert ray_pool.mode == "needed"
    assert ray_pool.defaults == {"attention": "ck_int8"}
    assert ray_pool.params_schema["properties"]["attention"]["enum"] == (
        "ck_int8",
        "torch_flash",
    )
    assert "raylight_attention_parameter" not in entries
    assert {
        entry.ui["visibility"]
        for template in (
            V5_TEMPLATE_BUNDLE.segment_templates.standard,
            V5_TEMPLATE_BUNDLE.segment_templates.raylight,
        )
        for entry in template.entries
    } == {"user", "internal"}

    catalog = build_feature_catalog(
        _current_snapshot(strict_runtime=True),
        template_bundle=V5_TEMPLATE_BUNDLE,
    )
    catalog_entries = {entry.id: entry for entry in catalog.entries}
    assert catalog_entries["attention_backend_override"].availability.state == (
        "conditional"
    )
    assert catalog_entries["h3_low_vram_attention"].availability.state == (
        "conditional"
    )


def test_bundle4_migration_is_detached_deterministic_and_injects_no_user_record() -> None:
    value = _document(bundle_version=4)
    source = _draft(value)
    before = source.model_dump_json()

    first = migrate_feature_configuration_to_v5(source.features)
    second = migrate_feature_configuration_to_v5(source.features)

    assert source.model_dump_json() == before
    assert first.source_template_bundle_version == 4
    assert first.configuration.template_bundle_version == 5
    assert first.configuration.model_dump(mode="json") == second.configuration.model_dump(
        mode="json"
    )
    assert set(first.configuration.project) == {"lora"}
    assert "attention_backend_override" not in first.configuration.project

    effective = _resolve(value)
    segment = next(iter(effective.effective_by_segment.values()))
    by_id = {feature.id: feature for feature in segment.features}
    assert by_id["attention_backend_override"].active is False
    assert by_id["h3_low_vram_attention"].active is False
    assert by_id["lora"].active is False


def test_segment_selection_wholly_replaces_project_selection() -> None:
    value = _document()
    segment_id = value["segments"][0]["id"]
    value["features"]["project"]["attention_backend_override"] = {
        "enabled": True,
        "params": {"mode": "pytorch"},
    }
    value["features"]["by_segment"][segment_id] = {
        "attention_backend_override": {
            "enabled": True,
            "params": {"mode": "ck_int8"},
        }
    }

    effective = _resolve(value)
    feature = next(
        feature
        for feature in effective.effective_by_segment[segment_id].features
        if feature.id == "attention_backend_override"
    )
    assert feature.source == "segment"
    assert feature.active is True
    assert feature.params == {"mode": "ck_int8"}

    # A deep merge would inherit the project's mode and accept this document.
    # Whole replacement validates the segment object on its own and rejects it.
    invalid = copy.deepcopy(value)
    invalid["features"]["by_segment"][segment_id][
        "attention_backend_override"
    ]["params"] = {}
    with pytest.raises(V5FeatureConfigurationError) as caught:
        _resolve(invalid)
    assert caught.value.code == "feature_params_invalid"
    assert caught.value.segment_id == segment_id


def test_disabled_switches_have_no_active_cache_projection() -> None:
    effective = _resolve(_document())
    segment = next(iter(effective.effective_by_segment.values()))
    inactive = {
        feature.id
        for feature in segment.features
        if feature.active_cache_projection() is None
    }
    active = tuple(
        feature.active_cache_projection()
        for feature in segment.features
        if feature.active_cache_projection() is not None
    )
    assert {
        "attention_backend_override",
        "h3_low_vram_attention",
        "lora",
        "continuity",
    } <= inactive
    assert all(item is not None for item in active)
    assert all(
        item["feature"].split("@", 1)[0] not in inactive
        for item in active
        if item is not None
    )


def test_continuity_effective_state_matches_the_resolved_predecessor_route() -> None:
    v4, settings_v1 = _v4_pair()
    continuity = v4.segments[0].continuity.model_copy(
        update={"enabled": True, "overlap_frames": 5}
    )
    root = v4.segments[0].model_copy(
        update={"id": "continuity-root", "continuity": continuity},
        deep=True,
    )
    successor = v4.segments[0].model_copy(
        update={
            "id": "continuity-successor",
            "title": "continuity-successor",
            "prompt": "Continue the same scene",
            "continuity": continuity,
        },
        deep=True,
    )
    v5, settings_v3 = _v5_pair(
        v4.model_copy(update={"segments": [root, successor]}, deep=True),
        settings_v1,
    )

    projection = project_v5_compile_authority(v5, settings_v3)
    continuity_active = {
        segment_id: next(
            feature.active
            for feature in effective.features
            if feature.id == "continuity"
        )
        for segment_id, effective in (
            projection.effective_features.effective_by_segment.items()
        )
    }

    assert continuity_active == {
        "continuity-root": False,
        "continuity-successor": True,
    }
    plan = compile_v5_execution_plan(v5, settings_v3, "continuity-routing")
    assert tuple(
        unit.continuity_dependency["predecessor_segment_id"]
        if unit.continuity_dependency is not None
        else None
        for unit in plan.segment_units
    ) == (None, "continuity-root")


@pytest.mark.parametrize("project_feature", (
    "attention_backend_override",
    "h3_low_vram_attention",
))
def test_project_and_segment_active_attention_patches_conflict(
    project_feature: str,
) -> None:
    value = _document()
    segment_id = value["segments"][0]["id"]
    other = (
        "h3_low_vram_attention"
        if project_feature == "attention_backend_override"
        else "attention_backend_override"
    )
    params = (
        {"mode": "pytorch"}
        if project_feature == "attention_backend_override"
        else {}
    )
    other_params = (
        {"mode": "ck_int8"}
        if other == "attention_backend_override"
        else {}
    )
    value["features"]["project"][project_feature] = {
        "enabled": True,
        "params": params,
    }
    value["features"]["by_segment"][segment_id] = {
        other: {"enabled": True, "params": other_params}
    }

    with pytest.raises(V5FeatureConfigurationError) as caught:
        _resolve(value)
    assert caught.value.code == "feature_conflict"
    assert caught.value.segment_id == segment_id
    assert {
        caught.value.feature_id,
        caught.value.safe_details["conflicting_feature_id"],
    } == {"attention_backend_override", "h3_low_vram_attention"}


@pytest.mark.parametrize(
    ("mutate", "code"),
    (
        (
            lambda value: value["features"]["project"].__setitem__(
                "unknown_feature", {"enabled": False, "params": {}}
            ),
            "unknown_feature",
        ),
        (
            lambda value: value["features"]["by_segment"].__setitem__(
                value["segments"][0]["id"],
                {
                    "lora": copy.deepcopy(
                        value["features"]["project"]["lora"]
                    )
                },
            ),
            "feature_scope_unsupported",
        ),
        (
            lambda value: value["features"]["project"].__setitem__(
                "attention_backend_override",
                {
                    "enabled": True,
                    "params": {"mode": "pytorch", "unknown": True},
                },
            ),
            "feature_params_invalid",
        ),
    ),
)
def test_unknown_scope_and_unknown_params_fail_closed(mutate, code: str) -> None:
    value = _document()
    mutate(value)
    with pytest.raises(V5FeatureConfigurationError) as caught:
        _resolve(value)
    assert caught.value.code == code


def test_needed_feature_cannot_be_disabled_and_active_wrong_backend_fails() -> None:
    ray = _document()
    ray["features"]["project"]["raylight_pool_intent"] = {
        "enabled": False,
        "params": {"attention": "ck_int8"},
    }
    with pytest.raises(V5FeatureConfigurationError) as caught:
        _resolve(ray, backend="raylight")
    assert caught.value.code == "needed_feature_disabled"
    assert caught.value.feature_id == "raylight_pool_intent"

    unsupported = _document()
    unsupported["features"]["project"]["attention_backend_override"] = {
        "enabled": True,
        "params": {"mode": "pytorch"},
    }
    with pytest.raises(V5FeatureConfigurationError) as caught:
        _resolve(unsupported, backend="raylight")
    assert caught.value.code == "feature_backend_unsupported"
    assert caught.value.backend == "raylight"


def test_catalog_preflight_and_compile_share_bundle5_resolution() -> None:
    v4, settings_v1 = _v4_pair()
    v5, settings_v3 = _v5_pair(v4, settings_v1)
    projection = project_v5_compile_authority(v5, settings_v3)
    snapshot = _current_snapshot()
    readiness = build_operational_readiness(
        endpoint_online=True,
        available_logical_gpu_count=4,
    )

    catalog = build_feature_catalog(snapshot, template_bundle=V5_TEMPLATE_BUNDLE)
    preflight = preflight_projected_v5_timeline(
        draft=projection.draft,
        settings=projection.settings,
        effective_features=projection.effective_features,
        snapshot=snapshot,
        readiness=readiness,
        resolved_lora_adapters=projection.lora_adapter_map(),
    )
    compiled = compile_v5_execution_plan(
        v5,
        settings_v3,
        "same-v5-resolution",
        host_capability_snapshot=snapshot,
        operational_readiness=readiness,
        capability_evaluator=CapabilityEvaluator(
            CURRENT_NODE_CONTRACT_REGISTRY
        ),
    )

    assert catalog.template_bundle_version == 5
    catalog_ids = {entry.id for entry in catalog.entries}
    assert {
        "attention_backend_override",
        "h3_low_vram_attention",
        "raylight_pool_intent",
        "lora",
    } <= catalog_ids
    assert preflight.valid is True
    assert preflight.template_bundle_version == 5
    assert compiled.template_bundle_version == 5

    segment_id = v5.segments[0].id
    effective = projection.effective_features.effective_by_segment[segment_id]
    preflight_segment = preflight.effective_by_segment[segment_id]
    expected = tuple(
        (feature.id, feature.version, "active" if feature.active else "noop")
        for feature in effective.features
    )
    observed_preflight = tuple(
        (feature.id, feature.version, feature.state)
        for feature in preflight_segment.features
    )
    observed_compile = tuple(
        (feature.feature_id, feature.version, feature.resolution.state)
        for feature in compiled.compile_report.feature_resolutions
        if feature.segment_id == segment_id
    )
    assert observed_preflight == expected
    assert observed_compile == expected
    assert {
        feature.id: feature.adapter_fingerprint
        for feature in preflight_segment.features
    } == {
        feature.feature_id: feature.adapter_fingerprint
        for feature in compiled.compile_report.feature_resolutions
        if feature.segment_id == segment_id
    }


def test_raylight_attention_preflight_and_compile_share_effective_route_mode() -> None:
    snapshot = _current_snapshot()
    readiness = build_operational_readiness(
        endpoint_online=True,
        available_logical_gpu_count=4,
    )
    evaluator = CapabilityEvaluator(CURRENT_NODE_CONTRACT_REGISTRY)

    flash_v5, settings_v3 = _raylight_ring_pair("torch_flash")
    flash_projection = project_v5_compile_authority(flash_v5, settings_v3)
    flash_preflight = preflight_projected_v5_timeline(
        draft=flash_projection.draft,
        settings=flash_projection.settings,
        effective_features=flash_projection.effective_features,
        snapshot=snapshot,
        readiness=readiness,
        resolved_lora_adapters=flash_projection.lora_adapter_map(),
        evaluator=evaluator,
    )
    flash_plan = compile_v5_execution_plan(
        flash_v5,
        settings_v3,
        "raylight-ring-flash",
        host_capability_snapshot=snapshot,
        operational_readiness=readiness,
        capability_evaluator=evaluator,
    )

    assert flash_preflight.valid is True
    initializer = next(
        node
        for node in flash_plan.segment_units[0].prompt_base.values()
        if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
    )
    assert initializer["inputs"]["XFuser_attention"] == "TORCH_FLASH"

    ck_v5, settings_v3 = _raylight_ring_pair("ck_int8")
    ck_projection = project_v5_compile_authority(ck_v5, settings_v3)
    with pytest.raises(NativeTemplateError) as preflight_error:
        preflight_projected_v5_timeline(
            draft=ck_projection.draft,
            settings=ck_projection.settings,
            effective_features=ck_projection.effective_features,
            snapshot=snapshot,
            readiness=readiness,
            resolved_lora_adapters=ck_projection.lora_adapter_map(),
            evaluator=evaluator,
        )
    with pytest.raises(NativeTemplateError) as compile_error:
        compile_v5_execution_plan(
            ck_v5,
            settings_v3,
            "raylight-ring-ck",
            host_capability_snapshot=snapshot,
            operational_readiness=readiness,
            capability_evaluator=evaluator,
        )
    assert str(preflight_error.value) == str(compile_error.value)
    assert "requires ring_degree=1" in str(preflight_error.value)


def test_preflight_and_compile_reject_tampered_contextual_effective_state() -> None:
    v4, settings_v1 = _v4_pair()
    v5, settings_v3 = _v5_pair(v4, settings_v1)
    projection = project_v5_compile_authority(v5, settings_v3)
    raw = projection.effective_features.model_dump(mode="json")
    segment_id = v5.segments[0].id
    lora = next(
        item
        for item in raw["effective_by_segment"][segment_id]["features"]
        if item["id"] == "lora"
    )
    assert lora["active"] is False
    lora["active"] = True
    tampered = EffectiveFeatureConfiguration.model_validate_json(
        json.dumps(raw)
    )
    snapshot = _current_snapshot()
    readiness = build_operational_readiness(
        endpoint_online=True,
        available_logical_gpu_count=4,
    )

    with pytest.raises(EffectiveFeatureResolutionMismatch):
        preflight_projected_v5_timeline(
            draft=projection.draft,
            settings=projection.settings,
            effective_features=tampered,
            snapshot=snapshot,
            readiness=readiness,
            resolved_lora_adapters=projection.lora_adapter_map(),
        )
    with pytest.raises(EffectiveFeatureResolutionMismatch):
        compile_projected_v5_timeline(
            projection.draft,
            projection.settings,
            "tampered-effective",
            tampered,
            resolved_lora_adapters=projection.lora_adapter_map(),
        )


def test_disabled_attention_params_do_not_change_effective_execution_identity() -> None:
    v4, settings_v1 = _v4_pair()
    source, settings_v3 = _v5_pair(v4, settings_v1)
    baseline_raw = source.model_dump(mode="json")
    baseline_raw["features"]["template_bundle_version"] = 5
    baseline = UnifiedTimelineDraftV5.model_validate(baseline_raw)
    changed_raw = copy.deepcopy(baseline_raw)
    changed_raw["features"]["project"]["attention_backend_override"] = {
        "enabled": False,
        "params": {"mode": "ck_int8"},
    }
    changed = UnifiedTimelineDraftV5.model_validate(changed_raw)

    baseline_plan = compile_v5_execution_plan(
        baseline,
        settings_v3,
        "disabled-identity",
    )
    changed_plan = compile_v5_execution_plan(
        changed,
        settings_v3,
        "disabled-identity",
    )

    assert changed_plan.effective_execution_digest == (
        baseline_plan.effective_execution_digest
    )
    assert tuple(unit.prompt_base for unit in changed_plan.segment_units) == tuple(
        unit.prompt_base for unit in baseline_plan.segment_units
    )


def test_active_attention_mode_emits_exact_current_node_and_changes_identity() -> None:
    v4, settings_v1 = _v4_pair()
    source, settings_v3 = _v5_pair(v4, settings_v1)
    raw = source.model_dump(mode="json")
    raw["features"]["template_bundle_version"] = 5
    raw["features"]["project"]["attention_backend_override"] = {
        "enabled": True,
        "params": {"mode": "pytorch"},
    }
    without_runtime_probe = compile_v5_execution_plan(
        UnifiedTimelineDraftV5.model_validate(raw),
        settings_v3,
        "active-attention-missing-runtime",
        host_capability_snapshot=_current_snapshot(),
        operational_readiness=build_operational_readiness(
            endpoint_online=True,
            available_logical_gpu_count=4,
        ),
        capability_evaluator=CapabilityEvaluator(
            CURRENT_NODE_CONTRACT_REGISTRY
        ),
    )
    snapshot = _current_snapshot(strict_runtime=True)
    readiness = build_operational_readiness(
        endpoint_online=True,
        available_logical_gpu_count=4,
    )
    evaluator = CapabilityEvaluator(CURRENT_NODE_CONTRACT_REGISTRY)
    pytorch = compile_v5_execution_plan(
        UnifiedTimelineDraftV5.model_validate(raw),
        settings_v3,
        "active-attention",
        host_capability_snapshot=snapshot,
        operational_readiness=readiness,
        capability_evaluator=evaluator,
    )
    assert without_runtime_probe.effective_execution_digest == (
        pytorch.effective_execution_digest
    )
    raw["features"]["project"]["attention_backend_override"]["params"] = {
        "mode": "ck_int8"
    }
    ck_int8 = compile_v5_execution_plan(
        UnifiedTimelineDraftV5.model_validate(raw),
        settings_v3,
        "active-attention",
        host_capability_snapshot=snapshot,
        operational_readiness=readiness,
        capability_evaluator=evaluator,
    )

    for plan, mode in ((pytorch, "pytorch"), (ck_int8, "ck_int8")):
        prompt = plan.segment_units[0].prompt_base
        strict_nodes = [
            node
            for node in prompt.values()
            if node["class_type"] == "DirectorStrictModelAttentionBackend"
        ]
        assert len(strict_nodes) == 1
        assert strict_nodes[0]["inputs"]["mode"] == mode
        model_edge = strict_nodes[0]["inputs"]["model"]
        assert prompt[model_edge[0]]["class_type"] == "MiniMaxH3SigmaShift"
    assert pytorch.effective_execution_digest != ck_int8.effective_execution_digest


def test_active_h3_low_vram_emits_exact_current_model_replacement() -> None:
    v4, settings_v1 = _v4_pair()
    source, settings_v3 = _v5_pair(v4, settings_v1)
    raw = source.model_dump(mode="json")
    raw["features"]["template_bundle_version"] = 5
    raw["features"]["project"]["h3_low_vram_attention"] = {
        "enabled": True,
        "params": {},
    }

    plan = compile_v5_execution_plan(
        UnifiedTimelineDraftV5.model_validate(raw),
        settings_v3,
        "active-low-vram",
        host_capability_snapshot=_current_snapshot(strict_runtime=True),
        operational_readiness=build_operational_readiness(
            endpoint_online=True,
            available_logical_gpu_count=4,
        ),
        capability_evaluator=CapabilityEvaluator(
            CURRENT_NODE_CONTRACT_REGISTRY
        ),
    )
    prompt = plan.segment_units[0].prompt_base
    strict_nodes = [
        node
        for node in prompt.values()
        if node["class_type"] == "DirectorStrictH3LowVramSagePatch"
    ]
    assert len(strict_nodes) == 1
    assert set(strict_nodes[0]["inputs"]) == {"model"}
    model_edge = strict_nodes[0]["inputs"]["model"]
    assert prompt[model_edge[0]]["class_type"] == "MiniMaxH3SigmaShift"
