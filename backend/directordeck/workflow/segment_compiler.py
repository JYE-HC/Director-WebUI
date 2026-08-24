from __future__ import annotations

"""Native Bundle-6 segment compiler built from semantic feature uses."""

from collections.abc import Mapping
from typing import Any

from ..native_templates import (
    NativeCompileResult,
    NativeContinuityDependency,
    NativeHistoricalTake,
    NativeTemplateError,
    NativeWorkflowUnit,
    _UNBOUND_PREDECESSOR_OUTPUT,
    bind_native_workflow_predecessor_output,
)
from ..schemas import RuntimeSettingsV3, UnifiedTimelineDraftV5, timeline_segment_recipe
from .audit import build_graph_audit_spec
from .builder import PromptGraphBuilder
from .compile_report import CompiledFeatureNotice, CompiledFeatureUseV3, NodeEmissionEvidenceV3
from .contracts import AllowedLateBoundInput, ResourcePool, TerminalRef
from .execution import derive_feature_execution_specs
from .feature_compiler_support import (
    audit_trace,
    commit_emission,
    node_contract_snapshot,
    public_reads,
    public_writes,
    read_resources,
)
from .feature_config import LoraConfigV1
from .node_contracts import (
    V4_OUTPUT_NEUTRAL_NODE_CLASSES,
    V6_NODE_CONTRACT_REGISTRY,
    v6_provenance_policy,
)
from .templates_v6 import (
    V6_RAYLIGHT_SEGMENT_TEMPLATE,
    V6_STANDARD_SEGMENT_TEMPLATE,
    SegmentTemplateV6,
)
from .v6_implementations import V6_FEATURE_REGISTRY
from .v6_projection import V6ResolvedRoute, project_v6_compile_authority
from .v6_registry import FeatureResolutionV6, ResolvedEarlierFeatures


class V6SegmentCompilerError(NativeTemplateError):
    pass


def _template(route: V6ResolvedRoute) -> SegmentTemplateV6:
    return (
        V6_RAYLIGHT_SEGMENT_TEMPLATE
        if route.backend == "raylight"
        else V6_STANDARD_SEGMENT_TEMPLATE
    )


def _continuity_load_node_id(prompt: Mapping[str, Any]) -> str:
    matches = [
        str(node_id)
        for node_id, node in prompt.items()
        if node.get("class_type") == "LoadVideo"
        and isinstance(node.get("inputs"), Mapping)
        and node["inputs"].get("file") == _UNBOUND_PREDECESSOR_OUTPUT
    ]
    if len(matches) != 1:
        raise V6SegmentCompilerError(
            "continuity must emit exactly one predecessor placeholder"
        )
    return matches[0]


def _late_bindings(
    prompt: Mapping[str, Any],
    continuity: NativeContinuityDependency | None,
    backend: str,
) -> tuple[AllowedLateBoundInput, ...]:
    result: list[AllowedLateBoundInput] = []
    if continuity is not None:
        result.append(
            AllowedLateBoundInput(
                input_pointer=f"/{continuity.load_video_node_id}/inputs/file",
                value_kind="string",
                source_kind="continuity",
            )
        )
    if backend == "raylight":
        initializers = [
            str(node_id)
            for node_id, node in prompt.items()
            if node.get("class_type") == "DirectorDeckRayInitializerAdvanced"
        ]
        if len(initializers) != 1:
            raise V6SegmentCompilerError("RayLight requires one bundled initializer")
        result.append(
            AllowedLateBoundInput(
                input_pointer=f"/{initializers[0]}/inputs/ray_cluster_namespace",
                value_kind="string",
                source_kind="runtime_epoch",
            )
        )
    return tuple(result)


def _plan(route: V6ResolvedRoute, context: Any, prompt: Mapping[str, Any]) -> dict[str, Any]:
    sampling = getattr(context.draft.sampling, route.family)
    return {
        "segment_id": route.segment.id,
        "mode": route.segment.mode,
        "recipe": timeline_segment_recipe(route.segment),
        "model_family": route.family,
        "backend": route.backend,
        "frame_count": context.visible_frames,
        "visible_frame_count": context.visible_frames,
        "sample_frame_count": context.sample_frames,
        "continuity_context_frames": context.continuity_prefix_frames,
        "alignment_tail_frame_count": (
            context.sample_frames
            - context.visible_frames
            - context.continuity_prefix_frames
        ),
        "predecessor_segment_id": route.predecessor_segment_id,
        "continuity_source": route.continuity_source,
        "historical_take_id": context.historical_take_id,
        "anchor_reset": route.anchor_reset,
        "seed_mode": "random" if sampling.random_seed else "fixed",
        "seed": sampling.seed,
        "conditioning_node": (
            "MiniMaxH3ImageToVideo"
            if route.family == "fl2va"
            else "MiniMaxH3ReferenceToVideo"
        ),
        "node_classes": list(dict.fromkeys(node["class_type"] for node in prompt.values())),
    }


def _compile_route(
    route: V6ResolvedRoute,
    *,
    projection: Any,
    job_id: str,
) -> tuple[NativeWorkflowUnit, dict[str, Any], LoraConfigV1 | None]:
    context = projection.context(route, job_id)
    template = _template(route)
    V6_FEATURE_REGISTRY.validate_template(template)
    graph = PromptGraphBuilder()
    pool = ResourcePool()
    resolutions: dict[str, FeatureResolutionV6 | None] = {}
    feature_uses: list[CompiledFeatureUseV3] = []
    notices: list[CompiledFeatureNotice] = []
    reads, writes, traces, scoped = [], [], [], []
    lora_config: LoraConfigV1 | None = None

    for use in template.entries:
        registration = V6_FEATURE_REGISTRY.require_feature(
            use.feature_id, use.feature_version
        )
        effective = registration.resolver.resolve(context)
        if (effective.feature_id, effective.feature_version) != (
            use.feature_id,
            use.feature_version,
        ):
            raise AssertionError("feature resolver identity drifted")
        if effective.state == "inactive":
            resolutions[use.feature_id] = None
            feature_uses.append(
                CompiledFeatureUseV3(
                    segment_id=route.segment.id,
                    unit_id=route.unit_id,
                    feature_id=use.feature_id,
                    version=use.feature_version,
                    backend=route.backend,
                    family=route.family,
                    template_id=template.id,
                    state="inactive",
                    config_source=effective.source,
                    reason_code=effective.reason_code,
                )
            )
            continue

        assert effective.config is not None
        dependencies = ResolvedEarlierFeatures(use.dependencies, resolutions)
        for dependency in use.dependencies:
            if dependency.required:
                dependencies.required(dependency.feature_id)
        implementation = V6_FEATURE_REGISTRY.require_implementation(
            use.feature_id,
            use.feature_version,
            route.backend,
            route.family,
        )
        resolution = implementation.resolve(
            effective.config,
            dependencies,
            context,
        )
        inputs = read_resources(pool, use)
        before = pool
        with graph.begin_scope(use.feature_id) as scope:
            emission = implementation.emit(
                scope,
                inputs,
                effective.config,
                resolution,
                dependencies,
                context,
            )
            pool = commit_emission(
                pool=pool,
                owner_id=use.feature_id,
                use=use,
                emission=emission,
                scope=scope,
            )
        resolution = resolution.model_copy(
            update={
                "implementation": resolution.implementation.model_copy(
                    update={
                        "class_types": tuple(
                            dict.fromkeys(
                                node["class_type"]
                                for node in scope.prompt_fragment.values()
                            )
                        )
                    }
                )
            }
        )
        resolutions[use.feature_id] = resolution
        reads.extend(public_reads(inputs, scope))
        writes.extend(public_writes(before, pool, use, emission))
        trace = audit_trace(
            feature_id=use.feature_id,
            scope=scope,
            registry=V6_NODE_CONTRACT_REGISTRY,
            structural_influence=bool(emission.outputs),
            output_neutral_classes=V4_OUTPUT_NEUTRAL_NODE_CLASSES,
        )
        if trace is not None:
            traces.append(trace)
        scoped.append((scope.emitted_node_ids, emission))
        notices.extend(
            CompiledFeatureNotice(
                segment_id=route.segment.id,
                unit_id=route.unit_id,
                feature_id=use.feature_id,
                message=message,
            )
            for message in emission.notices
        )
        feature_uses.append(
            CompiledFeatureUseV3(
                segment_id=route.segment.id,
                unit_id=route.unit_id,
                feature_id=use.feature_id,
                version=use.feature_version,
                backend=route.backend,
                family=route.family,
                template_id=template.id,
                state="applicable",
                config_source=effective.source,
                implementation=resolution.implementation,
                execution_identity=implementation.execution_identity(resolution, context),
                runtime_pool_identity=implementation.runtime_pool_identity(resolution, context),
                node_emissions=tuple(
                    NodeEmissionEvidenceV3(
                        node_id=node_id,
                        class_type=node["class_type"],
                        feature_id=use.feature_id,
                        implementation_id=resolution.implementation.implementation_id,
                    )
                    for node_id, node in scope.prompt_fragment.items()
                ),
            )
        )
        if isinstance(effective.config, LoraConfigV1):
            lora_config = effective.config

    take = pool.read_required("take_output", expected_type="TAKE", allow_terminal=True)
    if not isinstance(take.value, TerminalRef):
        raise AssertionError("save_take must publish a terminal")
    prompt = graph.prompt
    continuity = None
    if route.predecessor_segment_id is not None:
        continuity = NativeContinuityDependency(
            predecessor_segment_id=route.predecessor_segment_id,
            overlap_frames=context.continuity_prefix_frames,
            load_video_node_id=_continuity_load_node_id(prompt),
            source=route.continuity_source or "same_run",
            historical_take_id=context.historical_take_id,
        )
    audit = build_graph_audit_spec(
        prompt=prompt,
        node_contract_registry=V6_NODE_CONTRACT_REGISTRY,
        node_contract_snapshot=node_contract_snapshot(
            prompt,
            V6_NODE_CONTRACT_REGISTRY,
            director_adapter_only=True,
        ),
        public_writes=writes,
        public_reads=reads,
        feature_traces=traces,
        model_family=route.family,
        backend=route.backend,
        allowed_late_bound_inputs=_late_bindings(prompt, continuity, route.backend),
        unit_kind="segment",
        take_node_id=take.value.node_id,
        enforce_runtime_effects=False,
    )
    progress, preview = derive_feature_execution_specs(scoped)
    unit = NativeWorkflowUnit(
        id=route.unit_id,
        family=route.family,
        backend=route.backend,
        segment_ids=(route.segment.id,),
        prompt=prompt,
        output_nodes={route.segment.id: take.value.node_id},
        continuity=continuity,
        graph_audit_spec=audit,
        compile_feature_uses=tuple(feature_uses),
        compile_feature_notices=tuple(notices),
        progress_spec=progress,
        preview_spec=preview,
    )
    if route.historical_take is not None:
        unit = bind_native_workflow_predecessor_output(
            unit,
            route.historical_take.output,
            node_contract_registry=V6_NODE_CONTRACT_REGISTRY,
        )
    return unit, _plan(route, context, prompt), lora_config


def _manifest(
    projection: Any,
    workflows: list[NativeWorkflowUnit],
    loras: Mapping[str, LoraConfigV1],
) -> dict[str, Any]:
    return {
        "version": 3,
        "graph_source": "server",
        "accepts_client_workflow": False,
        "continuity": {
            "boundaries": [
                {
                    "segment_id": route.segment.id,
                    "predecessor_segment_id": route.predecessor_segment_id,
                    "overlap_frames": route.segment.continuity.overlap_frames,
                    "source": route.continuity_source,
                    "historical_take_id": (
                        route.historical_take.id if route.historical_take else None
                    ),
                }
                for route in projection.routes
                if route.predecessor_segment_id is not None
            ]
        },
        "submission_order": [unit.id for unit in workflows],
        "raylight_exclusive": any(unit.backend == "raylight" for unit in workflows),
        "lora_resolution": {
            family: config.model_dump(mode="json") for family, config in loras.items()
        },
        "units": [
            {
                "id": unit.id,
                "family": unit.family,
                "backend": unit.backend,
                "segment_ids": list(unit.segment_ids),
                "output_nodes": dict(unit.output_nodes),
            }
            for unit in workflows
        ],
    }


def compile_v6_timeline(
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    job_id: str,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, NativeHistoricalTake] | None = None,
) -> NativeCompileResult:
    """Compile normalized Bundle-6 authority without a legacy delegate."""

    try:
        projection = project_v6_compile_authority(
            draft,
            settings,
            segment_ids,
            historical_takes=historical_takes,
        )
        workflows, plans, loras = [], [], {}
        for route in projection.routes:
            unit, plan, lora = _compile_route(
                route,
                projection=projection,
                job_id=job_id,
            )
            workflows.append(unit)
            plans.append(plan)
            if lora is not None:
                loras.setdefault(route.family, lora)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, NativeTemplateError):
            raise
        raise V6SegmentCompilerError(str(exc)) from exc

    class_types = sorted(
        {node["class_type"] for unit in workflows for node in unit.prompt.values()}
    )
    provenance = dict(v6_provenance_policy(class_types))
    custom = sorted(
        class_type
        for class_type, source in provenance.items()
        if source in {"raylight", "lora-custom"}
    )
    return NativeCompileResult(
        workflows=tuple(workflows),
        manifest=_manifest(projection, workflows, loras),
        plans=tuple(plans),
        families=projection.families,
        node_policy={
            "graph_source": "server",
            "accepts_client_workflow": False,
            "allowed_nodes": class_types,
            "custom_nodes": custom,
            "provenance": provenance,
        },
    )


__all__ = ["V6SegmentCompilerError", "compile_v6_timeline"]
