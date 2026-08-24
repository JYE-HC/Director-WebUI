from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import inspect
from typing import Any

import pytest

import directordeck.workflow.feature_config as feature_config
import directordeck.workflow.segment_compiler as segment_compiler
from directordeck.native_templates import _UNBOUND_PREDECESSOR_OUTPUT
from directordeck.schemas import RuntimeSettingsV3, UnifiedTimelineDraftV5
from directordeck.workflow.segment_compiler import compile_v6_timeline
from directordeck.workflow.templates_v6 import (
    V6_RAYLIGHT_SEGMENT_TEMPLATE,
    V6_STANDARD_SEGMENT_TEMPLATE,
)

from . import extensible_workflow_v0_fixture_builder as fixture_builder
from .test_stage1a_v6_domain import _pair, _with_lora
from .test_workflow_v5_compat import _v5_pair


def _with_ck(
    draft: UnifiedTimelineDraftV5,
    enabled: bool,
) -> UnifiedTimelineDraftV5:
    document = draft.model_dump(mode="json")
    document["features"]["project"]["comfy_kitchen_attention"] = {
        "enabled": enabled,
        "params": {},
    }
    return UnifiedTimelineDraftV5.model_validate(document)


def _ray_settings(settings: RuntimeSettingsV3) -> RuntimeSettingsV3:
    document = settings.model_dump(mode="json")
    document["multi_gpu_enabled"] = True
    document["placement"]["fl2va"]["raylight"].update(
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


def _feature(unit: Any, feature_id: str) -> Any:
    matches = [
        use
        for use in unit.compile_feature_uses
        if use.feature_id == feature_id
    ]
    assert len(matches) == 1
    return matches[0]


def _node(unit: Any, class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (node_id, node)
        for node_id, node in unit.prompt.items()
        if node["class_type"] == class_type
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_twelve_uses_and_unique_node_owner(unit: Any) -> None:
    template = (
        V6_RAYLIGHT_SEGMENT_TEMPLATE
        if unit.backend == "raylight"
        else V6_STANDARD_SEGMENT_TEMPLATE
    )
    assert len(unit.compile_feature_uses) == 12
    assert tuple(
        use.feature_id for use in unit.compile_feature_uses
    ) == tuple(entry.feature_id for entry in template.entries)

    emitted = [
        evidence
        for use in unit.compile_feature_uses
        for evidence in use.node_emissions
    ]
    owners = Counter(evidence.node_id for evidence in emitted)
    assert set(owners) == set(unit.prompt)
    assert set(owners.values()) == {1}
    for evidence in emitted:
        assert unit.prompt[evidence.node_id]["class_type"] == evidence.class_type
        assert _feature(unit, evidence.feature_id).feature_id == evidence.feature_id
    for use in unit.compile_feature_uses:
        if use.state != "applicable":
            assert use.implementation is None
            continue
        assert use.implementation is not None
        assert use.implementation.class_types == tuple(
            dict.fromkeys(item.class_type for item in use.node_emissions)
        )

    assert unit.graph_audit_traces == ()
    snapshot = unit.graph_audit_spec.node_contract_snapshot
    assert set(snapshot) == set(unit.prompt)
    for evidence in snapshot.values():
        document = evidence.model_dump(mode="json")
        assert document["evidence_kind"] == "director_adapter"
        assert "python_module" not in document
        assert "runtime_fingerprint" not in document
    assert set(unit.graph_audit_spec.node_contract_snapshot) == set(unit.prompt)


def _assert_progress_and_preview(unit: Any) -> None:
    progress = unit.progress_spec
    preview = unit.preview_spec
    assert progress is not None
    assert preview is not None
    phases = {phase.id: phase for phase in progress.phases}
    assert {
        "sampling",
        "decode_video",
        "assemble_media",
        "persist_take",
    } <= set(phases)
    assert phases["sampling"].kind == "fractional"
    assert phases["sampling"].weight == pytest.approx(0.70)
    assert phases["decode_video"].weight == pytest.approx(0.15)
    assert phases["assemble_media"].weight == pytest.approx(0.10)
    assert phases["persist_take"].weight == pytest.approx(0.05)
    assert sum(
        phase.weight for phase in progress.phases if phase.kind != "stage"
    ) == pytest.approx(1.0)
    assert all(phase.node_id in unit.prompt for phase in progress.phases)
    assert len(preview.sources) == 1
    source = preview.sources[0]
    assert source.node_id == phases["sampling"].node_id
    assert source.phase_id == "sampling"
    assert source.publish is True
    assert unit.prompt[source.node_id]["class_type"] in {
        "SamplerCustomAdvanced",
        "DirectorDeckRayXFuserSamplerCustomAdvanced",
    }


@pytest.mark.parametrize("ck_enabled", (False, True), ids=("ck_off", "ck_on"))
def test_standard_ck_is_after_the_actual_lora_factory_path(
    monkeypatch: pytest.MonkeyPatch,
    ck_enabled: bool,
) -> None:
    draft, settings = _pair()
    lora_filename = "nested/style.safetensors"
    draft = _with_ck(_with_lora(draft, lora_filename), ck_enabled)
    actual_factory = feature_config.resolve_standard_lora_adapter
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def factory_spy(binding: Any, overrides: Any):
        records = tuple(overrides)
        calls.append(
            (
                binding.model_filename,
                binding.lora_filename,
                tuple(record.adapter_id for record in records),
            )
        )
        return actual_factory(binding, records)

    monkeypatch.setattr(
        feature_config,
        "resolve_standard_lora_adapter",
        factory_spy,
    )
    result = compile_v6_timeline(draft, settings, f"standard-{ck_enabled}")
    unit = result.workflows[0]

    assert calls == [
        (
            draft.model_stack.fl2va.filename,
            lora_filename,
            (),
        )
    ]
    lora_id, _lora = _node(unit, "LoraLoaderModelOnly")
    sigma_id, sigma = _node(unit, "MiniMaxH3SigmaShift")
    assert int(lora_id) < int(sigma_id)
    ck_nodes = [
        (node_id, node)
        for node_id, node in unit.prompt.items()
        if node["class_type"] == "ModelAttentionBackend"
    ]
    ck_use = _feature(unit, "comfy_kitchen_attention")
    if ck_enabled:
        assert ck_use.state == "applicable"
        assert len(ck_use.node_emissions) == 1
        assert len(ck_nodes) == 1
        ck_id, ck = ck_nodes[0]
        assert int(lora_id) < int(ck_id) < int(sigma_id)
        assert ck["inputs"] == {
            "model": [lora_id, 0],
            "attention": "comfy kitchen attention",
        }
        assert sigma["inputs"]["model"] == [ck_id, 0]
        assert any(
            phase.label == "配置 CK Attention"
            for phase in unit.progress_spec.phases  # type: ignore[union-attr]
        )
    else:
        assert ck_use.state == "inactive"
        assert ck_use.reason_code == "disabled"
        assert ck_nodes == []
        assert sigma["inputs"]["model"] == [lora_id, 0]

    assert result.manifest["lora_resolution"]["fl2va"]["adapter_id"] == (
        "model_only"
    )
    assert result.manifest["lora_resolution"]["fl2va"]["source"] == (
        "factory_default"
    )
    _assert_twelve_uses_and_unique_node_owner(unit)
    _assert_progress_and_preview(unit)


def test_raylight_ck_is_zero_node_identity_and_execution_descriptor_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_draft, base_settings = _pair()
    base_draft = _with_lora(base_draft, "nested/style.safetensors")
    settings = _ray_settings(base_settings)
    actual_factory = feature_config.resolve_raylight_lora_adapter
    calls: list[str] = []

    def factory_spy(family: str):
        calls.append(family)
        return actual_factory(family)  # type: ignore[arg-type]

    monkeypatch.setattr(
        feature_config,
        "resolve_raylight_lora_adapter",
        factory_spy,
    )
    compiled = {
        enabled: compile_v6_timeline(
            _with_ck(base_draft, enabled),
            settings,
            f"ray-{enabled}",
        )
        for enabled in (False, True)
    }
    assert calls == ["fl2va", "fl2va"]

    namespaces: dict[bool, str] = {}
    for enabled, result in compiled.items():
        unit = result.workflows[0]
        assert unit.backend == "raylight"
        assert "ModelAttentionBackend" not in {
            node["class_type"] for node in unit.prompt.values()
        }
        assert _node(unit, "DirectorDeckRayLoraLoader")
        ck_use = _feature(unit, "comfy_kitchen_attention")
        assert ck_use.node_emissions == ()
        assert ck_use.state == ("applicable" if enabled else "inactive")
        if enabled:
            assert ck_use.implementation is not None
            assert ck_use.implementation.carrier_kind == "director_runtime"
            assert ck_use.implementation.class_types == ()

        pool_owners = [
            use
            for use in unit.compile_feature_uses
            if use.runtime_pool_identity is not None
        ]
        assert [use.feature_id for use in pool_owners] == [
            "execution_strategy"
        ]
        descriptor = dict(pool_owners[0].runtime_pool_identity)
        expected_attention = (
            "COMFY_KITCHEN_INT8" if enabled else "TORCH_FLASH"
        )
        assert descriptor["attention_mode"] == expected_attention
        initializer_id, initializer = _node(
            unit,
            "DirectorDeckRayInitializerAdvanced",
        )
        assert initializer["inputs"]["XFuser_attention"] == expected_attention
        assert initializer["inputs"]["ray_cluster_namespace"] == descriptor[
            "namespace"
        ]
        namespaces[enabled] = descriptor["namespace"]
        late_bound = unit.graph_audit_spec.allowed_late_bound_inputs
        assert any(
            item.input_pointer
            == f"/{initializer_id}/inputs/ray_cluster_namespace"
            and item.source_kind == "runtime_epoch"
            for item in late_bound
        )
        _assert_twelve_uses_and_unique_node_owner(unit)
        _assert_progress_and_preview(unit)

    assert len(compiled[False].workflows[0].prompt) == len(
        compiled[True].workflows[0].prompt
    )
    assert namespaces[False] != namespaces[True]


def test_segment_compiler_has_no_v4_v5_compiler_or_registry_delegate() -> None:
    tree = ast.parse(inspect.getsource(segment_compiler))
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    assert not imported_modules & {
        "directordeck.workflow.v4_compiler",
        "directordeck.workflow.v5_compat",
        "directordeck.workflow.v5_registry",
        "directordeck.workflow.interpreters.builtin",
        "v4_compiler",
        "v5_compat",
        "v5_registry",
        "interpreters.builtin",
    }
    assert not called_names & {
        "compile_v4_timeline",
        "compile_projected_v5_timeline",
        "compile_v4_execution_plan",
        "compile_v5_execution_plan",
        "builtin_interpreters",
    }


def test_same_run_continuity_keeps_exact_placeholder_and_owner() -> None:
    draft, settings = _pair()
    document = _with_ck(draft, False).model_dump(mode="json")
    second = deepcopy(document["segments"][0])
    second["id"] = "segment-2"
    second["title"] = "Segment 2"
    second["continuity"] = {"enabled": True, "overlap_frames": 22}
    document["segments"].append(second)
    timeline = UnifiedTimelineDraftV5.model_validate(document)

    result = compile_v6_timeline(timeline, settings, "continuity")

    assert len(result.workflows) == 2
    successor = result.workflows[1]
    dependency = successor.continuity
    assert dependency is not None
    assert dependency.predecessor_segment_id == timeline.segments[0].id
    assert dependency.source == "same_run"
    assert successor.prompt[dependency.load_video_node_id]["inputs"]["file"] == (
        _UNBOUND_PREDECESSOR_OUTPUT
    )
    assert _feature(successor, "continuity").state == "applicable"
    _assert_twelve_uses_and_unique_node_owner(successor)


def test_ref2va_segment_smoke_uses_the_native_v6_path() -> None:
    v4 = fixture_builder._draft("r2v")
    settings_v1 = fixture_builder._settings("standard")
    draft, settings = _v5_pair(v4, settings_v1)
    document = draft.model_dump(mode="json")
    document["features"]["template_bundle_version"] = 6
    document["features"]["project"]["comfy_kitchen_attention"] = {
        "enabled": False,
        "params": {},
    }
    draft = UnifiedTimelineDraftV5.model_validate(document)

    result = compile_v6_timeline(draft, settings, "ref2va-smoke")
    unit = result.workflows[0]

    assert result.families == ("ref2va",)
    assert unit.family == "ref2va"
    _node(unit, "MiniMaxH3ReferenceToVideo")
    _assert_twelve_uses_and_unique_node_owner(unit)
    _assert_progress_and_preview(unit)


def test_fl2va_mute_does_not_require_or_emit_an_audio_vae() -> None:
    draft, settings = _pair()
    document = _with_ck(draft, False).model_dump(mode="json")
    document["segments"][0]["audio_mode"] = "mute"
    document["model_stack"]["audio_vae"]["filename"] = None
    timeline = UnifiedTimelineDraftV5.model_validate(document)

    unit = compile_v6_timeline(timeline, settings, "fl2va-mute").workflows[0]

    assert sum(
        node["class_type"] == "VAELoader" for node in unit.prompt.values()
    ) == 1
    assert sum(
        node["class_type"] == "SelectVAEDevice" for node in unit.prompt.values()
    ) == 1
    auxiliary = _feature(unit, "auxiliary_models")
    assert auxiliary.execution_identity["details"]["config"]["audio_vae_filename"] is None
    _assert_twelve_uses_and_unique_node_owner(unit)
