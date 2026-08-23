from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from copy import deepcopy
from pathlib import Path

import pytest

from directordeck.native_templates import (
    bind_raylight_runtime_epoch,
    build_raylight_shutdown_unit,
    raylight_runtime_descriptor,
    validate_native_workflow_ready,
)
from directordeck.workflow.audit import (
    GraphAuditError,
    validate_graph_audit_spec,
)
from directordeck.workflow.builder import PromptGraphBuilder
from directordeck.workflow.contracts import ListRef, RecordRef, ResourcePool
from directordeck.schemas import LoraLoaderOverrideRecord
from directordeck.workflow.lora_factory import (
    LoraLoaderBindingKey,
    ResolvedLoraAdapter,
    resolve_standard_lora_adapter,
)
from directordeck.workflow.node_contracts import V4_NODE_CONTRACT_REGISTRY
from directordeck.workflow.v4_compiler import (
    V4_INTERPRETER_REGISTRY,
    V4_VALIDATED_TEMPLATES,
    _scope_public_reads,
    compile_v4_timeline,
)

from . import extensible_workflow_v0_fixture_builder as fixture_builder


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "extensible_workflow_v0"
    / "native_prompt_goldens.json"
)
_STAGE7_STANDARD_LORA_DELTA_CASES = frozenset(
    {
        "standard-lora-dedicated",
        "standard-lora-model-only",
        "standard-lora-bypass-model-only",
    }
)
_STAGE7_LORA_DELTA_CASES = (
    _STAGE7_STANDARD_LORA_DELTA_CASES | {"raylight-lora"}
)
_CURRENT_TO_FROZEN_RAYLIGHT_CLASS = {
    "DirectorDeckRayInitializerAdvanced": "RayInitializerAdvanced",
    "DirectorDeckRayLoraLoader": "RayLoraLoader",
    "DirectorDeckRayUNETLoader": "RayUNETLoader",
    "DirectorDeckRayMiniMaxH3SigmaShift": "RayMiniMaxH3SigmaShift",
    "DirectorDeckRayBasicGuider": "RayBasicGuider",
    "DirectorDeckRayBasicScheduler": "RayBasicScheduler",
    "DirectorDeckRayXFuserSamplerCustomAdvanced": "XFuserSamplerCustomAdvanced",
    "DirectorDeckRayKill": "RayKill",
}


def _with_frozen_raylight_class_ids(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            normalized_key = _CURRENT_TO_FROZEN_RAYLIGHT_CLASS.get(key, key)
            normalized_item = _with_frozen_raylight_class_ids(item)
            if normalized_key == "allowed_nodes" and isinstance(
                normalized_item, list
            ):
                normalized_item = sorted(normalized_item)
            elif normalized_key == "runtime_key" and isinstance(
                normalized_item, str
            ):
                normalized_item = "<raylight-runtime-key>"
            normalized[normalized_key] = normalized_item
        return normalized
    if isinstance(value, list):
        return [_with_frozen_raylight_class_ids(item) for item in value]
    if isinstance(value, str):
        return _CURRENT_TO_FROZEN_RAYLIGHT_CLASS.get(value, value)
    return value


def test_v4_compiler_matches_the_22_unchanged_phase0_cases(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fixture_builder,
        "compile_native_timeline",
        compile_v4_timeline,
    )

    actual = fixture_builder.build_native_prompt_goldens(
        skip_case_ids=_STAGE7_LORA_DELTA_CASES,
    )
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    expected["cases"] = [
        case
        for case in expected["cases"]
        if case["id"] not in _STAGE7_LORA_DELTA_CASES
    ]

    normalized_actual = _with_frozen_raylight_class_ids(actual)
    normalized_expected = _with_frozen_raylight_class_ids(expected)
    assert len(normalized_actual["cases"]) == len(normalized_expected["cases"]) == 22
    assert normalized_actual == normalized_expected


def test_raylight_lora_graph_is_unchanged_with_bounded_stage7_manifest_delta(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        fixture_builder,
        "compile_native_timeline",
        compile_v4_timeline,
    )
    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frozen_by_id = {case["id"]: case for case in frozen["cases"]}
    actual = fixture_builder.build_native_prompt_goldens(
        skip_case_ids=set(frozen_by_id) - {"raylight-lora"},
    )["cases"][0]
    expected = frozen_by_id["raylight-lora"]

    for field in (
        "id",
        "kind",
        "description",
        "bound_raylight_epoch",
        "families",
        "node_policy",
        "plans",
        "units",
    ):
        assert _with_frozen_raylight_class_ids(actual[field]) == (
            _with_frozen_raylight_class_ids(expected[field])
        )

    actual_manifest = dict(_with_frozen_raylight_class_ids(actual["manifest"]))
    expected_manifest = dict(
        _with_frozen_raylight_class_ids(expected["manifest"])
    )
    actual_lora = actual_manifest.pop("lora_resolution")["fl2va"]
    expected_lora = expected_manifest.pop("lora_resolution")["fl2va"]
    assert actual_manifest == expected_manifest
    assert {
        key: actual_lora[key]
        for key in (
            "lora_name",
            "model_filename",
            "backend",
            "loader_node",
        )
    } == {
        key: expected_lora[key]
        for key in (
            "lora_name",
            "model_filename",
            "backend",
            "loader_node",
        )
    }
    assert set(actual_lora) == set(expected_lora) | {"adapter_id", "binding"}
    assert expected_lora["source"] == "raylight"
    assert actual_lora["adapter_id"] == "ray_lora"
    assert actual_lora["binding"] is None
    assert actual_lora["source"] == "backend_fixed"


def test_production_registry_is_frozen_and_binds_both_exact_templates() -> None:
    assert V4_INTERPRETER_REGISTRY.frozen is True
    assert set(V4_VALIDATED_TEMPLATES) == {
        "h3_standard_segment",
        "h3_raylight_segment",
    }
    for validated in V4_VALIDATED_TEMPLATES.values():
        assert tuple(
            (binding.id, binding.version) for binding in validated.bindings
        ) == tuple(
            (entry.id, entry.version) for entry in validated.template.entries
        )


def test_production_public_read_adapter_owns_one_composite_resource() -> None:
    graph = PromptGraphBuilder()
    producer = graph.begin_scope("bundle_producer")
    image_node = producer.add_node("LoadImage", {"image": "one.png"})
    audio_node = producer.add_node("LoadAudio", {"audio": "one.wav"})
    value = RecordRef(
        fields={
            "image": producer.edge(image_node),
            "audio_tracks": ListRef(items=(producer.edge(audio_node),)),
        }
    )
    pool = producer.commit(
        public_outputs={"reference_bundle": value},
        resource_transaction=ResourcePool().begin().define(
            name="reference_bundle",
            type="REFERENCE_BUNDLE",
            value=value,
            source_feature_id="bundle_producer",
            producer_node_ids=(image_node, audio_node),
        ),
    )
    assert pool is not None

    consumer = graph.begin_scope("bundle_consumer")
    consumer_node = consumer.add_node(
        "ReferenceBundleConsumer",
        {
            "bundle": pool.read_required(
                "reference_bundle", expected_type="REFERENCE_BUNDLE"
            ).value
        },
    )
    reads = _scope_public_reads(
        inputs={"reference_bundle": pool.resources["reference_bundle"]},
        scope=consumer,
    )

    assert len(reads) == 1
    assert reads[0].resource_name == "reference_bundle"
    assert reads[0].input_pointer == f"/{consumer_node}/inputs/bundle"
    assert reads[0].value == value
    consumer.rollback()


class _SingleReadResolution(Mapping[str, ResolvedLoraAdapter]):
    def __init__(self, resolution: ResolvedLoraAdapter) -> None:
        self.resolution = resolution
        self.reads = 0

    def __getitem__(self, key: str) -> ResolvedLoraAdapter:
        if key != "fl2va":
            raise KeyError(key)
        self.reads += 1
        if self.reads > 1:
            raise AssertionError("LoRA evidence was resolved more than once")
        return self.resolution

    def __iter__(self) -> Iterator[str]:
        return iter(("fl2va",))

    def __len__(self) -> int:
        return 1


def test_compile_consumes_one_immutable_lora_resolution() -> None:
    settings = fixture_builder._settings(
        "standard",
        lora_family="fl2va",
        lora_name="renamed_generic.safetensors",
    )
    binding = LoraLoaderBindingKey(
        family="fl2va",
        model_filename=settings.models.fl2va.filename,
        lora_filename="renamed_generic.safetensors",
    )
    resolutions = _SingleReadResolution(
        resolve_standard_lora_adapter(
            binding,
            (
                LoraLoaderOverrideRecord(
                    family="fl2va",
                    model_filename=binding.model_filename,
                    lora_filename=binding.lora_filename,
                    adapter_id="model_only",
                ),
            ),
        )
    )

    result = compile_v4_timeline(
        fixture_builder._draft("t2v"),
        settings,
        "single-resolution",
        resolved_lora_adapters=resolutions,
    )

    assert resolutions.reads == 1
    assert "LoraLoaderModelOnly" in {
        node["class_type"] for node in result.workflows[0].prompt.values()
    }
    assert result.manifest["lora_resolution"]["fl2va"]["loader_node"] == (
        "LoraLoaderModelOnly"
    )
    assert result.manifest["lora_resolution"]["fl2va"]["source"] == (
        "user_override"
    )


def test_production_unit_carries_exact_structural_audit_and_rejects_old_edge() -> None:
    unit = compile_v4_timeline(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "audited-standard",
    ).workflows[0]

    assert unit.graph_audit_spec is not None
    assert unit.graph_audit_traces
    validate_graph_audit_spec(
        prompt=unit.prompt,
        spec=unit.graph_audit_spec,
        node_contract_registry=V4_NODE_CONTRACT_REGISTRY,
        feature_traces=unit.graph_audit_traces,
        model_family=unit.family,
        backend=unit.backend,
        enforce_runtime_effects=False,
    )
    validate_native_workflow_ready(unit)

    tampered = deepcopy(unit.prompt)
    unet_id = next(
        node_id
        for node_id, node in tampered.items()
        if node["class_type"] == "UNETLoader"
    )
    sigma = next(
        node
        for node in tampered.values()
        if node["class_type"] == "MiniMaxH3SigmaShift"
    )
    sigma["inputs"]["model"] = [unet_id, 0]
    with pytest.raises(GraphAuditError, match="exact edge|latest exact revision"):
        validate_graph_audit_spec(
            prompt=tampered,
            spec=unit.graph_audit_spec,
            node_contract_registry=V4_NODE_CONTRACT_REGISTRY,
            feature_traces=unit.graph_audit_traces,
            model_family=unit.family,
            backend=unit.backend,
            enforce_runtime_effects=False,
        )


def test_production_audit_rejects_nested_edge_in_scalar_guider_port() -> None:
    unit = compile_v4_timeline(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "nested-guider-edge",
    ).workflows[0]
    assert unit.graph_audit_spec is not None
    tampered = deepcopy(unit.prompt)
    sampler = next(
        node
        for node in tampered.values()
        if node["class_type"] == "SamplerCustomAdvanced"
    )
    sampler["inputs"]["guider"] = {"nested": sampler["inputs"]["guider"]}

    with pytest.raises(GraphAuditError, match="does not match port type 'GUIDER'"):
        validate_graph_audit_spec(
            prompt=tampered,
            spec=unit.graph_audit_spec,
            node_contract_registry=V4_NODE_CONTRACT_REGISTRY,
            feature_traces=unit.graph_audit_traces,
            model_family=unit.family,
            backend=unit.backend,
            enforce_runtime_effects=False,
        )


def test_raylight_segment_audit_rejects_legacy_attention_fallback() -> None:
    unit = compile_v4_timeline(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "legacy-attention-segment",
    ).workflows[0]
    assert unit.graph_audit_spec is not None
    tampered = deepcopy(unit.prompt)
    initializer = next(
        node
        for node in tampered.values()
        if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
    )
    initializer["inputs"]["XFuser_attention"] = "TORCH_FLASH"

    with pytest.raises(GraphAuditError, match="require COMFY_KITCHEN_INT8"):
        validate_graph_audit_spec(
            prompt=tampered,
            spec=unit.graph_audit_spec,
            node_contract_registry=V4_NODE_CONTRACT_REGISTRY,
            feature_traces=unit.graph_audit_traces,
            model_family=unit.family,
            backend=unit.backend,
            enforce_runtime_effects=False,
        )


def test_exact_mapped_user_host_lora_is_submission_ready() -> None:
    settings = fixture_builder._settings(
        "standard",
        lora_family="fl2va",
        lora_name="renamed_generic.safetensors",
    )
    binding = LoraLoaderBindingKey(
        family="fl2va",
        model_filename=settings.models.fl2va.filename,
        lora_filename="renamed_generic.safetensors",
    )
    resolved = resolve_standard_lora_adapter(
        binding,
        (
            LoraLoaderOverrideRecord(
                family="fl2va",
                model_filename=binding.model_filename,
                lora_filename=binding.lora_filename,
                adapter_id="model_only",
            ),
        ),
    )
    unit = compile_v4_timeline(
        fixture_builder._draft("t2v"),
        settings,
        "mapped-host-lora",
        resolved_lora_adapters={"fl2va": resolved},
    ).workflows[0]

    assert unit.graph_audit_spec is not None
    emitted = {node["class_type"] for node in unit.prompt.values()}
    assert "LoraLoaderModelOnly" in emitted
    validate_native_workflow_ready(unit)


@pytest.mark.parametrize(
    ("backend", "lora_name", "expected_classes"),
    (
        (
            "standard",
            "minimax_h3_turbo_v4_step600_ema.safetensors",
            {"MiniMaxH3TurboLoRA"},
        ),
        (
            "raylight",
            "baseline-ray-style.safetensors",
            {"DirectorDeckRayLoraLoader", "DirectorDeckRayUNETLoader"},
        ),
    ),
)
def test_mapped_standard_and_owned_raylight_lora_paths_are_submission_ready(
    backend: str,
    lora_name: str,
    expected_classes: set[str],
) -> None:
    settings = fixture_builder._settings(
        backend,
        lora_family="fl2va",
        lora_name=lora_name,
        lora_strength=0.75,
    )
    resolved_lora_adapters = None
    if backend == "standard":
        binding = LoraLoaderBindingKey(
            family="fl2va",
            model_filename=settings.models.fl2va.filename,
            lora_filename=lora_name,
        )
        resolved_lora_adapters = {
            "fl2va": resolve_standard_lora_adapter(
                binding,
                (
                    LoraLoaderOverrideRecord(
                        family="fl2va",
                        model_filename=binding.model_filename,
                        lora_filename=binding.lora_filename,
                        adapter_id="dedicated",
                    ),
                ),
            )
        }
    unit = compile_v4_timeline(
        fixture_builder._draft("t2v"),
        settings,
        f"mapped-{backend}-lora",
        resolved_lora_adapters=resolved_lora_adapters,
    ).workflows[0]
    if backend == "raylight":
        unit = bind_raylight_runtime_epoch(unit, 7)

    emitted_classes = {node["class_type"] for node in unit.prompt.values()}
    assert expected_classes <= emitted_classes
    validate_native_workflow_ready(unit)


def test_raylight_kill_control_has_terminal_without_persistent_artifact() -> None:
    source = compile_v4_timeline(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "audited-ray-control-source",
    ).workflows[0]
    bound = bind_raylight_runtime_epoch(source, 3)
    descriptor = raylight_runtime_descriptor(bound)
    assert descriptor is not None

    control = build_raylight_shutdown_unit(
        descriptor,
        unit_id="audited-ray-control",
    )

    assert control.graph_audit_spec is not None
    assert control.graph_audit_spec.unit_kind == "control"
    assert control.graph_audit_spec.control_kind == "ray_kill"
    assert control.graph_audit_spec.take_node_id is None
    assert all(
        evidence.persistent_artifact_role is None
        for evidence in control.graph_audit_spec.node_contract_snapshot.values()
    )
    validate_native_workflow_ready(control)
