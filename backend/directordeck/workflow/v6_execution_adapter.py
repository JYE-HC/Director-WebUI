from __future__ import annotations

"""Project one native Bundle-6 compile into immutable plan version 3."""

from collections.abc import Mapping
from typing import Any

from ..native_templates import (
    NativeCompileResult,
    NativeTemplateError,
    NativeWorkflowUnit,
    raylight_runtime_descriptor,
    raylight_workflow_logical_gpu_indices,
)
from ..schemas import RuntimeSettingsV3, UnifiedTimelineDraftV5
from .compile_report import CompiledExecutionReportV3
from .contracts import GraphNodeContractEvidence
from .execution import (
    CompiledExecutionPlan,
    DocumentDigest,
    ExpectedOutputSpec,
    FeatureExecutionIdentity,
    PreparedSegmentUnit,
    RayRuntimeIdentity,
    RuntimePoolIdentityContribution,
    RuntimeRequirements,
    SegmentExecutionIdentity,
    effective_execution_digest,
    sha256_document_digest,
)
from .segment_compiler import compile_v6_timeline
from .templates_v6 import V6_RAYLIGHT_SEGMENT_TEMPLATE, V6_STANDARD_SEGMENT_TEMPLATE


_AUDIO_MODE = {"generate": "generated", "source": "source", "mute": "none"}
_STANDARD_DEVICE_CLASSES = frozenset(
    {"SelectModelDevice", "SelectCLIPDevice", "SelectVAEDevice"}
)


class V6ExecutionAdapterError(ValueError):
    pass


def _template(unit: NativeWorkflowUnit):
    return (
        V6_RAYLIGHT_SEGMENT_TEMPLATE
        if unit.backend == "raylight"
        else V6_STANDARD_SEGMENT_TEMPLATE
    )


def _standard_gpu_indices(unit: NativeWorkflowUnit) -> tuple[int, ...]:
    indices: list[int] = []
    for node in unit.prompt.values():
        if node.get("class_type") not in _STANDARD_DEVICE_CLASSES:
            continue
        device = node["inputs"].get("device")
        if isinstance(device, str) and device.startswith("gpu:"):
            index = int(device.removeprefix("gpu:"))
            if index not in indices:
                indices.append(index)
    return tuple(indices)


def _feature_use(unit: NativeWorkflowUnit, feature_id: str):
    matches = [
        use
        for use in unit.compile_feature_uses
        if use.feature_id == feature_id and use.state == "applicable"
    ]
    if len(matches) != 1:
        raise V6ExecutionAdapterError(
            f"unit {unit.id!r} requires one active {feature_id!r} identity"
        )
    return matches[0]


def _runtime_parts(
    unit: NativeWorkflowUnit,
    settings: RuntimeSettingsV3,
    endpoint_key: str,
) -> tuple[RuntimeRequirements, RayRuntimeIdentity | None]:
    if unit.backend == "standard":
        if raylight_runtime_descriptor(unit) is not None:
            raise V6ExecutionAdapterError("Standard unit contains Ray runtime state")
        return (
            RuntimeRequirements(
                endpoint_key=endpoint_key,
                backend="standard",
                logical_gpu_indices=_standard_gpu_indices(unit),
                requires_standard_driver_access=True,
            ),
            None,
        )
    try:
        descriptor = raylight_runtime_descriptor(unit)
        logical_gpus = raylight_workflow_logical_gpu_indices(unit)
    except NativeTemplateError as exc:
        raise V6ExecutionAdapterError(str(exc)) from exc
    if descriptor is None:
        raise V6ExecutionAdapterError("RayLight unit has no runtime descriptor")
    execution = _feature_use(unit, "execution_strategy")
    sigma = _feature_use(unit, "sigma_schedule")
    if execution.runtime_pool_identity is None or sigma.execution_identity is None:
        raise V6ExecutionAdapterError("Ray runtime identity is incomplete")
    expected_residency = settings.raylight_residency_policy
    requirements = RuntimeRequirements(
        endpoint_key=endpoint_key,
        backend="raylight",
        logical_gpu_indices=logical_gpus,
        ray_compatibility_key=str(descriptor["compatibility_key"]),
        ray_runtime_key=str(descriptor["runtime_key"]),
        requires_standard_driver_access=False,
        expected_residency_policy=expected_residency,
    )
    runtime = RayRuntimeIdentity(
        schema_version=1,
        placement={"logical_gpu_indices": list(logical_gpus)},
        fixed_parameters={
            "compatibility_key": str(descriptor["compatibility_key"]),
            "runtime_key": str(descriptor["runtime_key"]),
            "attention_mode": execution.runtime_pool_identity.get("attention_mode"),
        },
        active_feature_pool_identities=(
            RuntimePoolIdentityContribution(
                feature="execution_strategy@1",
                identity=execution.runtime_pool_identity,
            ),
            RuntimePoolIdentityContribution(
                feature="sigma_schedule@1",
                identity=sigma.execution_identity,
            ),
        ),
    )
    return requirements, runtime


def _continuity_identity(unit: NativeWorkflowUnit) -> dict[str, Any] | None:
    dependency = unit.continuity
    if dependency is None:
        return None
    return {
        "predecessor_segment_id": dependency.predecessor_segment_id,
        "overlap_frames": dependency.overlap_frames,
        "input_pointer": f"/{dependency.load_video_node_id}/inputs/file",
        "source": dependency.source,
        "historical_take_id": dependency.historical_take_id,
        "resolved": dependency.resolved,
        "bound_file": dependency.bound_file,
    }


def _feature_identities(unit: NativeWorkflowUnit) -> tuple[FeatureExecutionIdentity, ...]:
    return tuple(
        FeatureExecutionIdentity(
            feature=f"{use.feature_id}@{use.version}",
            effective_cache_params=use.execution_identity,
            resolved_implementations=(use.implementation,),
        )
        for use in unit.compile_feature_uses
        if use.state == "applicable"
        and use.execution_identity is not None
        and use.implementation is not None
    )


def _node_contracts(unit: NativeWorkflowUnit) -> tuple[GraphNodeContractEvidence, ...]:
    if unit.graph_audit_spec is None:
        raise V6ExecutionAdapterError("compiled unit has no graph audit")
    return tuple(
        unit.graph_audit_spec.node_contract_snapshot[node_id]
        for node_id in sorted(unit.graph_audit_spec.node_contract_snapshot)
    )


def _model_stack_projection(
    unit: NativeWorkflowUnit,
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
) -> dict[str, Any]:
    projection = {
        "family": unit.family,
        "diffusion": {
            **getattr(draft.model_stack, unit.family).model_dump(mode="json"),
            "device": getattr(settings.placement, unit.family).device,
        },
        "clip": draft.model_stack.clip.model_dump(mode="json"),
        "video_vae": draft.model_stack.video_vae.model_dump(mode="json"),
    }
    auxiliary_identity = _feature_use(unit, "auxiliary_models").execution_identity
    details = auxiliary_identity.get("details") if auxiliary_identity else None
    config = details.get("config") if isinstance(details, Mapping) else None
    if isinstance(config, Mapping) and config.get("audio_vae_filename") is not None:
        projection["audio_vae"] = draft.model_stack.audio_vae.model_dump(mode="json")
    return projection


def _validate_native_result(native: NativeCompileResult) -> None:
    if not native.workflows or len(native.workflows) != len(native.plans):
        raise V6ExecutionAdapterError("native workflows and plans must be non-empty pairs")
    if native.manifest.get("submission_order") != [unit.id for unit in native.workflows]:
        raise V6ExecutionAdapterError("native manifest order differs from workflows")
    expected_families = tuple(
        family
        for family in ("fl2va", "ref2va")
        if any(unit.family == family for unit in native.workflows)
    )
    if native.families != expected_families:
        raise V6ExecutionAdapterError("native family summary differs from workflows")


def adapt_v6_compile_result(
    native: NativeCompileResult,
    *,
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    endpoint_key: str = "embedded",
) -> CompiledExecutionPlan:
    if draft.version != 5 or draft.features.template_bundle_version != 6:
        raise V6ExecutionAdapterError("Bundle-6 adapter requires a Bundle-6 authority")
    if not endpoint_key:
        raise V6ExecutionAdapterError("endpoint key must be non-empty")
    _validate_native_result(native)
    segments = {segment.id: segment for segment in draft.segments}
    prepared, digests = [], []
    for unit, raw_plan in zip(native.workflows, native.plans, strict=True):
        if unit.segment_ids != (raw_plan.get("segment_id"),):
            raise V6ExecutionAdapterError("native plan owner differs from workflow")
        segment = segments.get(unit.segment_ids[0])
        if segment is None or segment.mode != unit.family:
            raise V6ExecutionAdapterError("native workflow differs from captured segment")
        output_id = unit.output_nodes.get(segment.id)
        if (
            output_id is None
            or unit.prompt.get(output_id, {}).get("class_type") != "SaveVideo"
            or unit.graph_audit_spec is None
            or unit.progress_spec is None
            or unit.preview_spec is None
        ):
            raise V6ExecutionAdapterError("native workflow execution evidence is incomplete")
        expected = ExpectedOutputSpec(
            segment_id=segment.id,
            node_id=output_id,
            width=draft.render.width,
            height=draft.render.height,
            fps=draft.render.fps,
            visible_frame_count=int(raw_plan["visible_frame_count"]),
            expected_audio_mode=_AUDIO_MODE[segment.audio_mode],
        )
        requirements, runtime = _runtime_parts(unit, settings, endpoint_key)
        identity = SegmentExecutionIdentity(
            schema_version=2,
            segment_creative_input=segment.model_dump(
                mode="json", exclude={"title", "enabled", "continuity"}
            ),
            render=draft.render.model_dump(mode="json"),
            family_sampling=getattr(draft.sampling, unit.family).model_dump(mode="json"),
            model_stack_projection=_model_stack_projection(unit, draft, settings),
            runtime_placement_projection=(
                {
                    "logical_gpu_indices": list(requirements.logical_gpu_indices),
                    "ray_compatibility_key": requirements.ray_compatibility_key,
                    "ray_runtime_key": requirements.ray_runtime_key,
                    "expected_residency_policy": requirements.expected_residency_policy,
                    "runtime_pool_identity": runtime.model_dump(mode="json"),
                }
                if runtime is not None
                else None
            ),
            feature_execution_identities=_feature_identities(unit),
            continuity_input_identity=_continuity_identity(unit),
            expected_output_geometry=expected.geometry,
        )
        template = _template(unit)
        digest = effective_execution_digest(
            identity,
            template_id=template.id,
            template_revision=template.revision,
            resolved_node_contract_identities=_node_contracts(unit),
        )
        prepared.append(
            PreparedSegmentUnit(
                id=unit.id,
                owner_segment_id=segment.id,
                family=unit.family,
                backend=unit.backend,
                template_id=template.id,
                template_revision=template.revision,
                prompt_base=unit.prompt,
                graph_audit_spec=unit.graph_audit_spec,
                expected_output_spec=expected,
                progress_spec=unit.progress_spec,
                preview_spec=unit.preview_spec,
                continuity_dependency=_continuity_identity(unit),
                runtime_requirements=requirements,
                runtime_pool_identity=runtime,
                effective_execution_digest=digest,
            )
        )
        digests.append((unit.id, digest))

    plan_digest = sha256_document_digest(
        {
            "schema_version": 2,
            "template_bundle_version": 6,
            "units": [
                {
                    "unit_id": unit_id,
                    "effective_execution_digest": digest.model_dump(mode="json"),
                }
                for unit_id, digest in digests
            ],
        }
    )
    report = CompiledExecutionReportV3(
        manifest=native.manifest,
        plans=native.plans,
        families=native.families,
        unit_effective_execution_digests=tuple(
            {"unit_id": unit_id, "digest": digest.model_dump(mode="json")}
            for unit_id, digest in digests
        ),
        feature_resolutions=tuple(
            use for unit in native.workflows for use in unit.compile_feature_uses
        ),
        notices=tuple(
            notice for unit in native.workflows for notice in unit.compile_feature_notices
        ),
    )
    return CompiledExecutionPlan(
        version=3,
        template_bundle_version=6,
        segment_units=tuple(prepared),
        compile_report=report,
        node_policy=native.node_policy,
        effective_execution_digest=plan_digest,
    )


def compile_v6_execution_plan(
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    job_id: str,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, Any] | None = None,
    endpoint_key: str = "embedded",
) -> CompiledExecutionPlan:
    native = compile_v6_timeline(
        draft,
        settings,
        job_id,
        segment_ids,
        historical_takes=historical_takes,
    )
    return adapt_v6_compile_result(
        native,
        draft=draft,
        settings=settings,
        endpoint_key=endpoint_key,
    )


__all__ = [
    "V6ExecutionAdapterError",
    "adapt_v6_compile_result",
    "compile_v6_execution_plan",
]
