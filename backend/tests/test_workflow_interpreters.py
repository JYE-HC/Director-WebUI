from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any

import pytest

from directordeck.native_templates import (
    NativeTemplateError,
    NativeWorkflowUnit,
    _align_h3_frame_count,
    _align_h3_frames,
    _raylight_namespace,
    bind_raylight_runtime_epoch,
)
from directordeck.schemas import LoraLoaderOverrideRecord
from directordeck.workflow.builder import GraphBuilderError, PromptGraphBuilder
from directordeck.workflow.contracts import EdgeRef, ResourcePool, TerminalRef
from directordeck.workflow.interpreters import (
    RAYLIGHT_INTERPRETER_IDS,
    STANDARD_INTERPRETER_IDS,
    ScopedBuilderEmitter,
    V4BuiltinContext,
    V4BuiltinParams,
    builtin_interpreter_map,
    builtin_interpreters,
    emit_raylight_model_load,
    emit_raylight_sampling,
    emit_standard_model_load,
    emit_standard_sampling,
)
from directordeck.workflow.registry import FeatureInterpreterRegistry
from directordeck.workflow.lora_factory import (
    LoraLoaderBindingKey,
    resolve_raylight_lora_adapter,
    resolve_standard_lora_adapter,
)
from directordeck.workflow.templates import (
    V4_RAYLIGHT_SEGMENT_TEMPLATE,
    V4_STANDARD_SEGMENT_TEMPLATE,
)
from .extensible_workflow_v0_fixture_builder import (
    FIXTURE_DIR,
    RAY_EPOCH,
    _continuity_draft,
    _draft,
    _maximum_reference_draft,
    _settings,
    render_json,
)


_GOLDENS = json.loads(
    (FIXTURE_DIR / "native_prompt_goldens.json").read_text(encoding="utf-8")
)
_GOLDEN_BY_ID = {case["id"]: case for case in _GOLDENS["cases"]}
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


def _with_frozen_raylight_class_ids(prompt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        node_id: {
            **node,
            "class_type": _CURRENT_TO_FROZEN_RAYLIGHT_CLASS.get(
                node["class_type"], node["class_type"]
            ),
        }
        for node_id, node in prompt.items()
    }


def _registry() -> FeatureInterpreterRegistry:
    registry = FeatureInterpreterRegistry()
    for interpreter in builtin_interpreters():
        registry.register(interpreter)
    return registry.freeze()


def _input_resources(entry: Any, pool: ResourcePool) -> dict[str, Any]:
    return {
        declaration.name: pool.resources[declaration.name]
        for declaration in entry.reads
        if declaration.name in pool.resources
    }


def _commit_emission(
    *,
    scope: Any,
    entry: Any,
    pool: ResourcePool,
    emission: Any,
) -> ResourcePool:
    declarations = {item.name: item for item in entry.writes}
    transaction = pool.begin()
    for name, value in emission.outputs.items():
        declaration = declarations[name]
        assert isinstance(value, (EdgeRef, TerminalRef))
        kwargs = {
            "name": name,
            "value": value,
            "source_feature_id": entry.id,
            "producer_node_ids": (value.node_id,),
        }
        if declaration.operation == "define":
            transaction = transaction.define(type=declaration.type, **kwargs)
        else:
            transaction = transaction.replace(
                expected_type=declaration.type,
                **kwargs,
            )
    return scope.commit_emission(emission, transaction)


def _compile_with_interpreters(
    *,
    draft: Any,
    settings: Any,
    job_id: str,
    segment_index: int = 0,
    continuity_prefix_frames: int = 0,
) -> tuple[dict[str, Any], str]:
    segment = draft.segments[segment_index]
    family = segment.mode
    binding = getattr(settings.models, family)
    backend = "raylight" if len(binding.raylight.gpu_select) >= 2 else "standard"
    sampling = getattr(draft.sampling, family)
    visible_frames = _align_h3_frames(segment.duration_seconds, draft.render.fps)
    sample_frames = _align_h3_frame_count(
        visible_frames + continuity_prefix_frames
    )
    resolved_lora = None
    if binding.lora_name is not None:
        if backend == "raylight":
            resolved_lora = resolve_raylight_lora_adapter(family)
        else:
            exact_binding = LoraLoaderBindingKey(
                family=family,
                model_filename=binding.filename,
                lora_filename=binding.lora_name,
            )
            override = binding.standard_lora_loader_override
            overrides = (
                ()
                if override is None
                else (
                    LoraLoaderOverrideRecord(
                        lora_filename=override.lora_name,
                        adapter_id=(
                            "minimax_h3_turbo"
                            if override.loader in {"dedicated", "minimax_h3_turbo"}
                            else "model_only"
                        ),
                        options=(
                            {"low_vram": binding.lora_low_vram}
                            if override.loader in {"dedicated", "minimax_h3_turbo"}
                            else {}
                        ),
                    ),
                )
            )
            resolved_lora = resolve_standard_lora_adapter(
                exact_binding,
                overrides,
            )
    context = V4BuiltinContext(
        backend=backend,
        family=family,
        template_bundle_version=4,
        settings=settings,
        draft=draft,
        segment=segment,
        binding=binding,
        sampling=sampling,
        job_id=job_id,
        visible_frames=visible_frames,
        sample_frames=sample_frames,
        continuity_prefix_frames=continuity_prefix_frames,
        lora_loader_node=(
            resolved_lora.adapter.class_type
            if resolved_lora is not None
            else None
        ),
        lora_adapter_id=(
            resolved_lora.adapter.adapter_id
            if resolved_lora is not None
            else None
        ),
        lora_loader_binding=(
            resolved_lora.binding if resolved_lora is not None else None
        ),
        lora_resolution_source=(
            resolved_lora.source if resolved_lora is not None else None
        ),
        lora_adapter_options=(
            resolved_lora.options if resolved_lora is not None else None
        ),
        raylight_namespace=(
            _raylight_namespace(family, binding)
            if backend == "raylight"
            else None
        ),
        clear_raylight_vram_after_sampling=(
            settings.raylight_residency_policy != "keep_until_switch"
        ),
    )
    template = (
        V4_RAYLIGHT_SEGMENT_TEMPLATE
        if backend == "raylight"
        else V4_STANDARD_SEGMENT_TEMPLATE
    )
    registry = _registry()
    graph = PromptGraphBuilder()
    pool = ResourcePool()
    params = V4BuiltinParams()
    output_node_id: str | None = None

    for entry in template.entries:
        if entry.id == "lora" and binding.lora_name is None:
            continue
        if entry.id == "continuity" and continuity_prefix_frames == 0:
            continue
        interpreter = registry.require(entry.id, entry.version)
        resolution = interpreter.resolve(params, context)
        with graph.begin_scope(entry.id) as scope:
            emission = interpreter.emit(
                scope,
                _input_resources(entry, pool),
                params,
                context,
                resolution,
            )
            pool = _commit_emission(
                scope=scope,
                entry=entry,
                pool=pool,
                emission=emission,
            )
            if entry.id == "save_take":
                take = emission.outputs["take_output"]
                assert isinstance(take, TerminalRef)
                output_node_id = take.node_id

    assert output_node_id is not None
    prompt = graph.prompt
    if backend == "raylight":
        unit = NativeWorkflowUnit(
            id=f"raylight-{family}-{segment_index:03d}",
            family=family,
            backend="raylight",
            segment_ids=(segment.id,),
            prompt=prompt,
            output_nodes={segment.id: output_node_id},
        )
        prompt = bind_raylight_runtime_epoch(unit, RAY_EPOCH).prompt
    return prompt, output_node_id


@pytest.mark.parametrize("backend", ("standard", "raylight"))
@pytest.mark.parametrize("recipe", ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"))
def test_real_scoped_builder_matches_frozen_core_prompt_goldens(
    backend: str,
    recipe: str,
) -> None:
    case_id = f"{backend}-{recipe}"
    actual, output_node_id = _compile_with_interpreters(
        draft=_draft(recipe),
        settings=_settings(backend),
        job_id=f"fixture-{case_id}",
    )
    expected_unit = _GOLDEN_BY_ID[case_id]["units"][0]

    assert render_json(_with_frozen_raylight_class_ids(actual)).encode("utf-8") == render_json(
        expected_unit["prompt"]
    ).encode("utf-8")
    assert output_node_id == expected_unit["output_nodes"][
        next(iter(expected_unit["output_nodes"]))
    ]


@pytest.mark.parametrize(
    ("case_id", "lora_name", "override", "expected_class"),
    (
        (
            "standard-lora-minimax-h3-turbo",
            "minimax_h3_turbo_v4_step600_ema.safetensors",
            "minimax_h3_turbo",
            "MiniMaxH3TurboLoRA",
        ),
        (
            "standard-lora-model-only",
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "model_only",
            "LoraLoaderModelOnly",
        ),
    ),
)
def test_standard_lora_fragments_use_exact_stage7_adapters(
    case_id: str,
    lora_name: str,
    override: str | None,
    expected_class: str,
) -> None:
    actual, _ = _compile_with_interpreters(
        draft=_draft("t2v", segment_id=f"baseline-{case_id}"),
        settings=_settings(
            "standard",
            lora_family="fl2va",
            lora_name=lora_name,
            lora_strength=0.75,
            standard_override=override,
        ),
        job_id=f"fixture-{case_id}",
    )
    classes = {node["class_type"] for node in actual.values()}
    assert expected_class in classes
    assert "LoraLoaderBypassModelOnly" not in classes


def test_raylight_lora_fragment_matches_frozen_prompt_golden() -> None:
    case_id = "raylight-lora"
    actual, _ = _compile_with_interpreters(
        draft=_draft("t2v", segment_id="baseline-raylight-lora"),
        settings=_settings(
            "raylight",
            lora_family="fl2va",
            lora_name="baseline-ray-style.safetensors",
            lora_strength=0.875,
        ),
        job_id=f"fixture-{case_id}",
    )
    expected = _GOLDEN_BY_ID[case_id]["units"][0]["prompt"]
    assert render_json(_with_frozen_raylight_class_ids(actual)) == render_json(expected)


@pytest.mark.parametrize(
    ("audio_mode", "recipe"),
    (("generate", "r2v"), ("source", "v2v"), ("mute", "t2v")),
)
def test_audio_output_fragments_match_frozen_prompt_goldens(
    audio_mode: str,
    recipe: str,
) -> None:
    case_id = f"audio-{audio_mode}"
    actual, _ = _compile_with_interpreters(
        draft=_draft(
            recipe,
            segment_id=f"baseline-audio-{audio_mode}",
            audio_mode=audio_mode,
        ),
        settings=_settings("standard"),
        job_id=f"fixture-{case_id}",
    )
    expected = _GOLDEN_BY_ID[case_id]["units"][0]["prompt"]
    assert render_json(actual) == render_json(expected)


def test_continuity_fragment_matches_real_builder_frozen_golden() -> None:
    case_id = "standard-continuity-same-run"
    draft = _continuity_draft()
    actual, _ = _compile_with_interpreters(
        draft=draft,
        settings=_settings("standard"),
        job_id=f"fixture-{case_id}",
        segment_index=1,
        continuity_prefix_frames=draft.segments[1].continuity.overlap_frames,
    )
    expected = _GOLDEN_BY_ID[case_id]["units"][1]["prompt"]
    assert render_json(actual) == render_json(expected)


def test_maximum_reference_autogrow_fragment_matches_frozen_golden() -> None:
    case_id = "standard-maximum-reference-slots"
    actual, _ = _compile_with_interpreters(
        draft=_maximum_reference_draft(),
        settings=_settings("standard"),
        job_id=f"fixture-{case_id}",
    )
    expected = _GOLDEN_BY_ID[case_id]["units"][0]["prompt"]
    assert render_json(actual) == render_json(expected)


def test_registry_binds_both_frozen_templates_in_exact_order() -> None:
    registry = _registry()
    standard = registry.validate_template(V4_STANDARD_SEGMENT_TEMPLATE)
    raylight = registry.validate_template(V4_RAYLIGHT_SEGMENT_TEMPLATE)

    assert tuple(binding.id for binding in standard.bindings) == STANDARD_INTERPRETER_IDS
    assert tuple(binding.id for binding in raylight.bindings) == RAYLIGHT_INTERPRETER_IDS
    assert len(registry.identities) == len(set(registry.identities)) == 15
    modes = {
        interpreter.id: interpreter.mode for interpreter in builtin_interpreters()
    }
    assert {name for name, mode in modes.items() if mode == "switch"} == {
        "lora",
        "continuity",
    }


def test_scoped_adapter_recursively_types_edges_and_rejects_unknown_nodes() -> None:
    graph = PromptGraphBuilder()
    with graph.begin_scope("source") as scope:
        source = scope.add_node("Source", {"value": 1})
        scope.commit()

    with graph.begin_scope("nested") as scope:
        emitter = ScopedBuilderEmitter(scope)
        emitter.add(
            "NestedConsumer",
            payload={
                "direct": [source, 0],
                "items": [[source, 1], {"again": [source, 2]}],
            },
        )
        scope.commit()
    assert graph.prompt["2"]["inputs"]["payload"] == {
        "direct": ["1", 0],
        "items": [["1", 1], {"again": ["1", 2]}],
    }

    with graph.begin_scope("unknown") as scope:
        with pytest.raises(GraphBuilderError, match="unknown node"):
            ScopedBuilderEmitter(scope).add("Consumer", model=["999", 0])


def test_backend_specific_model_and_sampler_apis_have_no_backend_switch() -> None:
    for function in (
        emit_standard_model_load,
        emit_raylight_model_load,
        emit_standard_sampling,
        emit_raylight_sampling,
    ):
        assert "backend" not in inspect.signature(function).parameters


def test_model_interpreters_fail_closed_on_the_wrong_backend() -> None:
    graph = PromptGraphBuilder()
    ray_binding = _settings("raylight").models.fl2va
    with graph.begin_scope("standard_model_load") as scope:
        with pytest.raises(NativeTemplateError, match="RayLight binding"):
            emit_standard_model_load(ScopedBuilderEmitter(scope), ray_binding)
    assert graph.prompt == {}

    standard_binding = _settings("standard").models.fl2va
    with graph.begin_scope("raylight_model_load") as scope:
        with pytest.raises(NativeTemplateError, match="Standard binding"):
            emit_raylight_model_load(
                ScopedBuilderEmitter(scope),
                ["missing", 0],
                standard_binding,
            )
    assert graph.prompt == {}
