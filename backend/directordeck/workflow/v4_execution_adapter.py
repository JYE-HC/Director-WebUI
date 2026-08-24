from __future__ import annotations

"""Stage-4 execution-plan adapter for the stable v4 native compiler.

The v4 compiler remains the sole prompt constructor during the migration.  This
module projects that *same* compile result, captured timeline and captured
settings into the immutable execution contracts needed by the submission
layer.  It neither recompiles a graph nor invokes a feature interpreter.
"""

from collections.abc import Mapping
from typing import Any

from ..native_templates import (
    NativeCompileResult,
    NativeHistoricalTake,
    NativeTemplateError,
    NativeWorkflowUnit,
    _align_h3_frames,
    raylight_runtime_descriptor,
    raylight_workflow_logical_gpu_indices,
    resolve_execution_backend,
    validate_native_workflow_runtime_effects,
)
from ..schemas import (
    RuntimeSettings,
    UnifiedTimelineDraft,
    UnifiedTimelineSegment,
    timeline_segment_recipe,
)
from .audit import FeatureAuditTrace
from .compile_report import CompiledExecutionReportV2
from .contracts import (
    HostCapabilitySnapshot,
    ModelFamily,
    NodeContractEvidence,
    OperationalReadiness,
    TemplateBundle,
)
from .execution import (
    CompiledExecutionPlan,
    DocumentDigest,
    ExpectedOutputSpec,
    FeatureExecutionIdentity,
    PreparedSegmentUnit,
    PreviewSource,
    PreviewSpec,
    ProgressPhase,
    ProgressSpec,
    RayRuntimeIdentity,
    RuntimePoolIdentityContribution,
    RuntimeRequirements,
    SegmentExecutionIdentity,
    effective_execution_digest,
    sha256_document_digest,
)
from .lora_factory import ResolvedLoraAdapter
from .node_contracts import node_contract_registry_for_bundle
from .templates import V4_TEMPLATE_BUNDLE
from .v4_compiler import compile_v4_timeline


_AUDIO_MODE = {
    "generate": "generated",
    "source": "source",
    "mute": "none",
}
_SAMPLER_CLASSES = frozenset(
    {"SamplerCustomAdvanced", "DirectorDeckRayXFuserSamplerCustomAdvanced"}
)
_STANDARD_DEVICE_CLASSES = frozenset(
    {"SelectModelDevice", "SelectCLIPDevice", "SelectVAEDevice"}
)
class V4ExecutionAdapterError(ValueError):
    """The native result and its captured v4 authorities disagree."""


def _template_for(
    unit: NativeWorkflowUnit,
    template_bundle: TemplateBundle = V4_TEMPLATE_BUNDLE,
):
    templates = {
        "standard": template_bundle.segment_templates.standard,
        "raylight": template_bundle.segment_templates.raylight,
    }
    try:
        template = templates[unit.backend]
    except KeyError as exc:  # pragma: no cover - NativeWorkflowUnit is typed.
        raise V4ExecutionAdapterError(
            f"unsupported v4 execution backend {unit.backend!r}"
        ) from exc
    expected_id = f"h3_{unit.backend}_segment"
    if template.id != expected_id:
        raise AssertionError("v4 template bundle backend ordering drifted")
    return template


def _node_ids_for_classes(
    unit: NativeWorkflowUnit,
    class_types: frozenset[str],
    *,
    role: str,
) -> tuple[str, ...]:
    matches = tuple(
        str(node_id)
        for node_id, node in unit.prompt.items()
        if isinstance(node, Mapping) and node.get("class_type") in class_types
    )
    if len(matches) != 1:
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} must contain exactly one {role} node"
        )
    return matches


def _progress_and_preview_specs(
    unit: NativeWorkflowUnit,
    *,
    template_bundle_version: int,
) -> tuple[ProgressSpec, PreviewSpec]:
    if unit.progress_spec is not None or unit.preview_spec is not None:
        if unit.progress_spec is None or unit.preview_spec is None:
            raise V4ExecutionAdapterError(
                f"native unit {unit.id!r} has an incomplete explicit "
                "progress/preview authority"
            )
        return unit.progress_spec, unit.preview_spec

    # Frozen bundle-4 compatibility only. Current compilers attach an explicit
    # pair derived from interpreter emissions and must never mix that authority
    # with this historical class-type scan.
    if template_bundle_version != V4_TEMPLATE_BUNDLE.version:
        raise V4ExecutionAdapterError(
            f"native unit {unit.id!r} has no explicit progress/preview authority"
        )
    sampler_id = _node_ids_for_classes(
        unit,
        _SAMPLER_CLASSES,
        role="sampler",
    )[0]
    decode_id = _node_ids_for_classes(
        unit,
        frozenset({"VAEDecode"}),
        role="video decode",
    )[0]
    create_video_id = _node_ids_for_classes(
        unit,
        frozenset({"CreateVideo"}),
        role="video assembly",
    )[0]
    save_id = _node_ids_for_classes(
        unit,
        frozenset({"SaveVideo"}),
        role="take persistence",
    )[0]
    progress = ProgressSpec(
        version=1,
        phases=(
            ProgressPhase(
                id="sampling",
                label="采样中",
                node_id=sampler_id,
                kind="fractional",
                weight=0.70,
            ),
            ProgressPhase(
                id="decode_video",
                label="解码视频画面",
                node_id=decode_id,
                kind="milestone",
                weight=0.15,
            ),
            ProgressPhase(
                id="assemble_media",
                label="封装音视频",
                node_id=create_video_id,
                kind="milestone",
                weight=0.10,
            ),
            ProgressPhase(
                id="persist_take",
                label="写入视频文件",
                node_id=save_id,
                kind="milestone",
                weight=0.05,
            ),
        ),
    )
    preview = PreviewSpec(
        version=1,
        sources=(
            PreviewSource(
                node_id=sampler_id,
                phase_id="sampling",
                publish=True,
                priority=100,
            ),
        ),
    )
    return progress, preview


def _explicit_standard_gpu_indices(unit: NativeWorkflowUnit) -> tuple[int, ...]:
    indices: list[int] = []
    for node in unit.prompt.values():
        if (
            not isinstance(node, Mapping)
            or node.get("class_type") not in _STANDARD_DEVICE_CLASSES
            or not isinstance(node.get("inputs"), Mapping)
        ):
            continue
        device = node["inputs"].get("device")
        if not isinstance(device, str) or not device.startswith("gpu:"):
            continue
        index = int(device.removeprefix("gpu:"))
        if index not in indices:
            indices.append(index)
    return tuple(indices)


def _ray_runtime_parts(
    unit: NativeWorkflowUnit,
    *,
    endpoint_key: str,
    residency_policy: str,
    template_bundle: TemplateBundle,
) -> tuple[RuntimeRequirements, RayRuntimeIdentity]:
    try:
        descriptor = raylight_runtime_descriptor(unit)
        logical_gpu_indices = raylight_workflow_logical_gpu_indices(unit)
    except NativeTemplateError as exc:
        raise V4ExecutionAdapterError(str(exc)) from exc
    if descriptor is None:
        raise V4ExecutionAdapterError(
            f"RayLight unit {unit.id!r} has no runtime descriptor"
        )
    expected_clear_after_sampling = residency_policy == "release_after_sampling"
    if (
        descriptor.get("clear_vram_after_sampling")
        is not expected_clear_after_sampling
    ):
        raise V4ExecutionAdapterError(
            f"RayLight unit {unit.id!r} residency policy disagrees with "
            "captured settings"
        )
    initializer_id = str(descriptor["initializer_node_id"])
    initializer = unit.prompt.get(initializer_id)
    if not isinstance(initializer, Mapping) or not isinstance(
        initializer.get("inputs"), Mapping
    ):
        raise V4ExecutionAdapterError(
            f"RayLight unit {unit.id!r} has invalid initializer evidence"
        )
    initializer_inputs = {
        str(key): value
        for key, value in initializer["inputs"].items()
        if key != "ray_cluster_namespace"
    }
    sigma_id = _node_ids_for_classes(
        unit,
        frozenset({"DirectorDeckRayMiniMaxH3SigmaShift"}),
        role="RayLight sigma mutation",
    )[0]
    sigma = unit.prompt[sigma_id]
    if not isinstance(sigma, Mapping) or not isinstance(
        sigma.get("inputs"), Mapping
    ):
        raise V4ExecutionAdapterError(
            f"RayLight unit {unit.id!r} has invalid sigma evidence"
        )
    sigma_identity = {
        "shift_video": sigma["inputs"].get("shift_video"),
        "shift_audio": sigma["inputs"].get("shift_audio"),
    }
    template_versions = {
        entry.id: entry.version
        for entry in _template_for(unit, template_bundle).entries
    }
    pool_feature = (
        f"raylight_pool_intent@{template_versions['raylight_pool_intent']}"
    )
    sigma_feature = (
        f"raylight_sigma_shift@{template_versions['raylight_sigma_shift']}"
    )
    requirements = RuntimeRequirements(
        endpoint_key=endpoint_key,
        backend="raylight",
        logical_gpu_indices=logical_gpu_indices,
        ray_compatibility_key=str(descriptor["compatibility_key"]),
        ray_runtime_key=str(descriptor["runtime_key"]),
        requires_standard_driver_access=False,
        expected_residency_policy=residency_policy,
    )
    pool_identity: Mapping[str, Any] = initializer_inputs
    if template_bundle.version >= 5:
        evidence = _current_feature_identity_evidence(unit)
        pool_evidence = evidence.get("raylight_pool_intent")
        if (
            pool_evidence is None
            or not isinstance(pool_evidence.runtime_pool_identity, Mapping)
        ):
            raise V4ExecutionAdapterError(
                f"current RayLight unit {unit.id!r} has no exact pool identity"
            )
        pool_identity = pool_evidence.runtime_pool_identity
    runtime_identity = RayRuntimeIdentity(
        schema_version=1,
        placement={"logical_gpu_indices": list(logical_gpu_indices)},
        fixed_parameters={
            "compatibility_key": str(descriptor["compatibility_key"]),
            "initializer_inputs": initializer_inputs,
            "runtime_mutations": {
                "DirectorDeckRayMiniMaxH3SigmaShift": sigma_identity,
            },
        },
        active_feature_pool_identities=(
            RuntimePoolIdentityContribution(
                feature=pool_feature,
                identity=dict(pool_identity),
            ),
            RuntimePoolIdentityContribution(
                feature=sigma_feature,
                identity=sigma_identity,
            ),
        ),
    )
    return requirements, runtime_identity


def _runtime_parts(
    unit: NativeWorkflowUnit,
    *,
    endpoint_key: str,
    captured_settings: RuntimeSettings,
    template_bundle: TemplateBundle,
) -> tuple[RuntimeRequirements, RayRuntimeIdentity | None]:
    if unit.backend == "raylight":
        return _ray_runtime_parts(
            unit,
            endpoint_key=endpoint_key,
            residency_policy=captured_settings.raylight_residency_policy,
            template_bundle=template_bundle,
        )
    if raylight_runtime_descriptor(unit) is not None:
        raise V4ExecutionAdapterError(
            f"Standard unit {unit.id!r} unexpectedly carries Ray runtime state"
        )
    return (
        RuntimeRequirements(
            endpoint_key=endpoint_key,
            backend="standard",
            logical_gpu_indices=_explicit_standard_gpu_indices(unit),
            requires_standard_driver_access=True,
        ),
        None,
    )


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


def _feature_identities(
    unit: NativeWorkflowUnit,
    template_bundle: TemplateBundle,
) -> tuple[FeatureExecutionIdentity, ...]:
    template = _template_for(unit, template_bundle)
    versions = {entry.id: entry.version for entry in template.entries}
    feature_ids = tuple(trace.feature_id for trace in unit.graph_audit_traces)
    if not feature_ids or len(feature_ids) != len(set(feature_ids)):
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} must carry unique active feature traces"
        )
    current_evidence = (
        _current_feature_identity_evidence(unit)
        if template_bundle.version >= 5
        else None
    )
    if template_bundle.version < 5 and unit.feature_identity_evidence:
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} unexpectedly carries current identity evidence"
        )
    identities: list[FeatureExecutionIdentity] = []
    for trace in unit.graph_audit_traces:
        if not isinstance(trace, FeatureAuditTrace):
            raise V4ExecutionAdapterError(
                f"v4 unit {unit.id!r} has invalid feature audit evidence"
            )
        try:
            version = versions[trace.feature_id]
        except KeyError as exc:
            raise V4ExecutionAdapterError(
                f"v4 unit {unit.id!r} contains unknown feature trace "
                f"{trace.feature_id!r}"
            ) from exc
        cache_params: dict[str, Any]
        if current_evidence is not None:
            exact_cache_identity = current_evidence[
                trace.feature_id
            ].cache_identity
            if not isinstance(exact_cache_identity, Mapping):
                raise V4ExecutionAdapterError(
                    f"current feature {trace.feature_id!r} has a non-object cache identity"
                )
            cache_params = dict(exact_cache_identity)
        else:
            cache_params = {
                "authority": "resolved_feature_execution_identity",
                "feature_id": trace.feature_id,
                "backend": unit.backend,
                "family": unit.family,
                "implementations": [
                    implementation.runtime_fingerprint
                    for implementation in trace.resolution.implementations
                ],
            }
        if trace.feature_id == "lora" and current_evidence is None:
            details = trace.resolution.resolution_details
            adapter_id = details.get("adapter_id")
            binding = details.get("binding")
            strength = details.get("strength")
            loader_options = details.get("loader_options", {})
            if (
                not isinstance(adapter_id, str)
                or not adapter_id
                or not isinstance(binding, Mapping)
                or not isinstance(strength, (int, float))
                or isinstance(strength, bool)
                or not isinstance(loader_options, Mapping)
            ):
                raise V4ExecutionAdapterError(
                    f"v4 unit {unit.id!r} has incomplete LoRA identity evidence"
                )
            # mapping_source is deliberately report-only. Historical snapshots
            # may still say factory_default; when their adapter and binding are
            # identical, they retain one execution identity/currentness digest.
            cache_params.update(
                adapter_id=adapter_id,
                binding=dict(binding),
                strength=strength,
                loader_options=dict(loader_options),
            )
        identities.append(
            FeatureExecutionIdentity(
                feature=f"{trace.feature_id}@{version}",
                effective_cache_params=cache_params,
                resolved_implementations=trace.resolution.implementations,
            )
        )
    return tuple(identities)


def _current_feature_identity_evidence(unit: NativeWorkflowUnit) -> dict[str, Any]:
    """Require exact one-to-one active trace/cache identity evidence."""

    trace_ids = tuple(trace.feature_id for trace in unit.graph_audit_traces)
    evidence_ids = tuple(
        item.feature_id for item in unit.feature_identity_evidence
    )
    if evidence_ids != trace_ids or len(evidence_ids) != len(set(evidence_ids)):
        raise V4ExecutionAdapterError(
            f"current unit {unit.id!r} cache identity keys differ from active traces"
        )
    return {item.feature_id: item for item in unit.feature_identity_evidence}


def _model_stack_projection(
    unit: NativeWorkflowUnit,
    captured_settings: RuntimeSettings,
) -> dict[str, Any]:
    binding = getattr(captured_settings.models, unit.family)
    projection: dict[str, Any] = {
        "family": unit.family,
        "diffusion": {
            "filename": binding.filename,
            "device": binding.device,
        },
        "clip": captured_settings.models.clip.model_dump(mode="json"),
        "video_vae": captured_settings.models.video_vae.model_dump(mode="json"),
    }
    if binding.lora_name is not None:
        projection["diffusion"].update(
            {
                "lora_name": binding.lora_name,
                "lora_strength": binding.lora_strength,
                "lora_low_vram": binding.lora_low_vram,
            }
        )
    loaded_vae_names = {
        node["inputs"].get("vae_name")
        for node in unit.prompt.values()
        if isinstance(node, Mapping)
        and node.get("class_type") == "VAELoader"
        and isinstance(node.get("inputs"), Mapping)
    }
    if captured_settings.models.audio_vae.filename in loaded_vae_names:
        projection["audio_vae"] = captured_settings.models.audio_vae.model_dump(
            mode="json"
        )
    return projection


def _node_contract_evidence(unit: NativeWorkflowUnit) -> tuple[NodeContractEvidence, ...]:
    audit = unit.graph_audit_spec
    if audit is None:
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} has no GraphAuditSpec"
        )
    return tuple(
        audit.node_contract_snapshot[node_id]
        for node_id in sorted(audit.node_contract_snapshot)
    )


def _segment_identity(
    unit: NativeWorkflowUnit,
    *,
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    captured_settings: RuntimeSettings,
    expected_output: ExpectedOutputSpec,
    runtime_requirements: RuntimeRequirements,
    runtime_pool_identity: RayRuntimeIdentity | None,
    template_bundle: TemplateBundle,
) -> SegmentExecutionIdentity:
    creative_input = segment.model_dump(
        mode="json",
        exclude={"title", "enabled", "continuity"},
    )
    return SegmentExecutionIdentity(
        schema_version=1,
        segment_creative_input=creative_input,
        render=draft.render.model_dump(mode="json"),
        family_sampling=getattr(draft.sampling, unit.family).model_dump(
            mode="json"
        ),
        model_stack_projection=_model_stack_projection(unit, captured_settings),
        runtime_placement_projection=(
            {
                "logical_gpu_indices": list(
                    runtime_requirements.logical_gpu_indices
                ),
                "ray_compatibility_key": (
                    runtime_requirements.ray_compatibility_key
                ),
                "ray_runtime_key": runtime_requirements.ray_runtime_key,
                "expected_residency_policy": (
                    runtime_requirements.expected_residency_policy
                ),
                "runtime_pool_identity": runtime_pool_identity.model_dump(
                    mode="json"
                ),
            }
            if runtime_pool_identity is not None
            else None
        ),
        feature_execution_identities=_feature_identities(unit, template_bundle),
        continuity_input_identity=_continuity_identity(unit),
        expected_output_geometry=expected_output.geometry,
    )


def _validate_pair(
    unit: NativeWorkflowUnit,
    plan: Mapping[str, Any],
    *,
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    captured_settings: RuntimeSettings,
) -> str:
    if unit.segment_ids != (segment.id,):
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} must own exactly segment {segment.id!r}"
        )
    if unit.family != segment.mode:
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} family disagrees with captured segment"
        )
    captured_backend = resolve_execution_backend(
        getattr(captured_settings.models, unit.family)
    )
    if captured_backend != unit.backend:
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} backend disagrees with captured settings"
        )
    expected_plan = {
        "segment_id": segment.id,
        "mode": segment.mode,
        "recipe": timeline_segment_recipe(segment),
        "model_family": unit.family,
        "backend": unit.backend,
        "visible_frame_count": _align_h3_frames(
            segment.duration_seconds,
            draft.render.fps,
        ),
    }
    for field, expected in expected_plan.items():
        if plan.get(field) != expected:
            raise V4ExecutionAdapterError(
                f"v4 unit {unit.id!r} plan field {field!r} disagrees with "
                "captured compile authorities"
            )
    if set(unit.output_nodes) != {segment.id}:
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} must declare one owner output"
        )
    output_node_id = unit.output_nodes[segment.id]
    output_node = unit.prompt.get(output_node_id)
    if (
        not isinstance(output_node_id, str)
        or not isinstance(output_node, Mapping)
        or output_node.get("class_type") != "SaveVideo"
    ):
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} output must be one SaveVideo node"
        )
    if (
        unit.graph_audit_spec is None
        or unit.graph_audit_spec.take_node_id != output_node_id
    ):
        raise V4ExecutionAdapterError(
            f"v4 unit {unit.id!r} output disagrees with GraphAuditSpec"
        )
    return output_node_id


def adapt_v4_compile_result(
    native_result: NativeCompileResult,
    *,
    draft: UnifiedTimelineDraft,
    captured_settings: RuntimeSettings,
    endpoint_key: str = "embedded",
    host_capability_revision: str | None = None,
    template_bundle: TemplateBundle = V4_TEMPLATE_BUNDLE,
) -> CompiledExecutionPlan:
    """Project one already-produced v4 result into immutable execution units."""

    if draft.version != 4:
        raise V4ExecutionAdapterError("v4 execution adapter requires timeline v4")
    if not endpoint_key:
        raise V4ExecutionAdapterError("endpoint key must be non-empty")
    if len(native_result.workflows) != len(native_result.plans):
        raise V4ExecutionAdapterError(
            "native workflows and compile plans must have identical cardinality"
        )
    if not native_result.workflows:
        raise V4ExecutionAdapterError("native compile result contains no workflows")
    expected_families = tuple(
        family
        for family in ("fl2va", "ref2va")
        if any(unit.family == family for unit in native_result.workflows)
    )
    if native_result.families != expected_families:
        raise V4ExecutionAdapterError(
            "native family summary disagrees with workflows"
        )
    manifest_order = native_result.manifest.get("submission_order")
    if manifest_order != [unit.id for unit in native_result.workflows]:
        raise V4ExecutionAdapterError(
            "native manifest submission order disagrees with workflows"
        )
    segments = {segment.id: segment for segment in draft.segments}
    prepared_units: list[PreparedSegmentUnit] = []
    unit_digests: list[tuple[str, DocumentDigest]] = []
    for unit, plan in zip(
        native_result.workflows,
        native_result.plans,
        strict=True,
    ):
        # The native graph is a compiler-local compatibility projection.  Run
        # its strict runtime-effect gate here, before it is discarded, so the
        # production caller receives one complete execution authority instead
        # of retaining a second workflow representation for preflight.
        validate_native_workflow_runtime_effects(
            unit,
            node_contract_registry=node_contract_registry_for_bundle(
                template_bundle.version
            ),
        )
        if len(unit.segment_ids) != 1 or unit.segment_ids[0] not in segments:
            raise V4ExecutionAdapterError(
                f"v4 unit {unit.id!r} does not map to one captured segment"
            )
        segment = segments[unit.segment_ids[0]]
        output_node_id = _validate_pair(
            unit,
            plan,
            segment=segment,
            draft=draft,
            captured_settings=captured_settings,
        )
        visible_frame_count = int(plan["visible_frame_count"])
        expected_output = ExpectedOutputSpec(
            segment_id=segment.id,
            node_id=output_node_id,
            width=draft.render.width,
            height=draft.render.height,
            fps=draft.render.fps,
            visible_frame_count=visible_frame_count,
            expected_audio_mode=_AUDIO_MODE[segment.audio_mode],
        )
        progress_spec, preview_spec = _progress_and_preview_specs(
            unit,
            template_bundle_version=template_bundle.version,
        )
        runtime_requirements, runtime_pool_identity = _runtime_parts(
            unit,
            endpoint_key=endpoint_key,
            captured_settings=captured_settings,
            template_bundle=template_bundle,
        )
        template = _template_for(unit, template_bundle)
        if unit.graph_audit_spec is None:  # Kept local for type narrowing.
            raise V4ExecutionAdapterError(
                f"v4 unit {unit.id!r} has no GraphAuditSpec"
            )
        segment_identity = _segment_identity(
            unit,
            segment=segment,
            draft=draft,
            captured_settings=captured_settings,
            expected_output=expected_output,
            runtime_requirements=runtime_requirements,
            runtime_pool_identity=runtime_pool_identity,
            template_bundle=template_bundle,
        )
        unit_digest = effective_execution_digest(
            segment_identity,
            template_id=template.id,
            template_revision=template.revision,
            resolved_node_contract_identities=_node_contract_evidence(unit),
        )
        prepared = PreparedSegmentUnit(
            id=unit.id,
            owner_segment_id=segment.id,
            family=unit.family,
            backend=unit.backend,
            template_id=template.id,
            template_revision=template.revision,
            prompt_base=unit.prompt,
            graph_audit_spec=unit.graph_audit_spec,
            expected_output_spec=expected_output,
            progress_spec=progress_spec,
            preview_spec=preview_spec,
            continuity_dependency=_continuity_identity(unit),
            runtime_requirements=runtime_requirements,
            runtime_pool_identity=runtime_pool_identity,
            effective_execution_digest=unit_digest,
        )
        prepared_units.append(prepared)
        unit_digests.append((unit.id, unit_digest))

    plan_digest = sha256_document_digest(
        {
            "schema_version": 1,
            # Bundle/catalog revisions are not output identity by themselves.
            # Keep the frozen v4 domain marker so bundle 5 can add disabled
            # descriptors without invalidating otherwise identical Standard
            # takes. Active feature version/params/implementation and graph
            # contract changes are already captured by each unit digest.
            "template_bundle_version": V4_TEMPLATE_BUNDLE.version,
            "units": [
                {
                    "unit_id": unit_id,
                    "effective_execution_digest": digest.model_dump(mode="json"),
                }
                for unit_id, digest in unit_digests
            ],
        }
    )
    compile_report = CompiledExecutionReportV2(
        source="v4_native_compile_adapter_v2",
        host_capability_revision=host_capability_revision,
        manifest=native_result.manifest,
        plans=native_result.plans,
        families=native_result.families,
        unit_effective_execution_digests=tuple(
            {
                "unit_id": unit_id,
                "digest": digest.model_dump(mode="json"),
            }
            for unit_id, digest in unit_digests
        ),
        feature_resolutions=tuple(
            resolution
            for unit in native_result.workflows
            for resolution in unit.compile_feature_resolutions
        ),
        notices=tuple(
            notice
            for unit in native_result.workflows
            for notice in unit.compile_feature_notices
        ),
    )
    return CompiledExecutionPlan(
        version=2,
        template_bundle_version=template_bundle.version,
        segment_units=tuple(prepared_units),
        compile_report=compile_report,
        node_policy=native_result.node_policy,
        effective_execution_digest=plan_digest,
    )


def compile_v4_execution_plan(
    draft: UnifiedTimelineDraft,
    captured_settings: RuntimeSettings,
    job_id: str,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, NativeHistoricalTake] | None = None,
    resolved_lora_adapters: Mapping[
        ModelFamily, ResolvedLoraAdapter
    ]
    | None = None,
    endpoint_key: str = "embedded",
    host_capability_snapshot: HostCapabilitySnapshot | None = None,
    operational_readiness: OperationalReadiness | None = None,
    capability_evaluator: Any | None = None,
) -> CompiledExecutionPlan:
    """Compile once and expose only the immutable production execution plan.

    ``NativeCompileResult`` is deliberately confined to this adapter.  Callers
    that need migration-fixture inspection can invoke
    :func:`adapt_v4_compile_result` explicitly with a test-owned native result;
    production submission never receives both representations.
    """

    native_result = compile_v4_timeline(
        draft,
        captured_settings,
        job_id,
        segment_ids,
        historical_takes=historical_takes,
        resolved_lora_adapters=resolved_lora_adapters,
        host_capability_snapshot=host_capability_snapshot,
        operational_readiness=operational_readiness,
        capability_evaluator=capability_evaluator,
    )
    return adapt_v4_compile_result(
        native_result,
        draft=draft,
        captured_settings=captured_settings,
        endpoint_key=endpoint_key,
        host_capability_revision=(
            host_capability_snapshot.host_capability_revision()
            if host_capability_snapshot is not None
            else None
        ),
    )


__all__ = [
    "V4ExecutionAdapterError",
    "adapt_v4_compile_result",
    "compile_v4_execution_plan",
]
