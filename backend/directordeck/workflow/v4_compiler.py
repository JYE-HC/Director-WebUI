from __future__ import annotations

"""Behavior-preserving v4 compiler assembled from registered feature scopes.

Stage 2 changes the internal construction boundary, not Director's public
workflow behavior.  The resolver remains the sole creative-input authority;
each immutable template entry is bound to an exact interpreter and commits its
graph fragment together with the matching typed resource-pool delta.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..native_templates import (
    NativeCompileResult,
    NativeContinuityDependency,
    NativeFeatureIdentityEvidence,
    NativeHistoricalTake,
    NativeTemplateError,
    NativeWorkflowUnit,
    RaylightAttentionMode,
    _UNBOUND_PREDECESSOR_OUTPUT,
    _align_h3_frame_count,
    _align_h3_frames,
    _raylight_namespace,
    bind_native_workflow_predecessor_output,
    raylight_runtime_namespace,
)
from ..schemas import RuntimeSettings, UnifiedTimelineDraft, timeline_segment_recipe
from .audit import (
    FeatureAuditTrace,
    ResolvedNodeEmission,
    build_graph_audit_spec,
)
from .builder import PromptGraphBuilder
from .compile_report import (
    CompiledCapabilityEvaluation,
    CompiledFeatureNotice,
    CompiledFeatureResolution,
)
from .contracts import (
    AllowedLateBoundInput,
    CapabilitySet,
    EdgeRef,
    FeatureEmission,
    FeatureResolution,
    FeatureTemplateEntry,
    FeatureInterpreter,
    HostCapabilitySnapshot,
    ListRef,
    NodeContractRegistry,
    NodeContractEvidence,
    PublicResourceRead,
    PublicResourceWrite,
    PublishedValueRef,
    RecordRef,
    Resource,
    ResourcePool,
    OperationalReadiness,
    TerminalRef,
    canonical_sha256,
)
from .interpreters import (
    V4BuiltinContext,
    V4BuiltinParams,
    builtin_interpreters,
)
from .node_contracts import (
    V4_NODE_CONTRACT_REGISTRY,
    V4_OUTPUT_NEUTRAL_NODE_CLASSES,
    current_provenance_policy,
    native_provenance_policy,
)
from .lora_factory import ResolvedLoraAdapter
from .effective_features import (
    EffectiveFeatureConfiguration,
    EffectiveSegmentFeatures,
    feature_parameter_model,
)
from .execution import derive_feature_execution_specs
from .registry import FeatureInterpreterRegistry, ValidatedFeatureTemplate
from .templates import (
    V4_RAYLIGHT_SEGMENT_TEMPLATE,
    V4_STANDARD_SEGMENT_TEMPLATE,
    V4_TEMPLATE_BUNDLE,
)
from .v4_resolver import (
    CreativeCompileInputError,
    CreativeCompileInputResolver,
    V4CreativeCompileInput,
    V4ResolvedSegmentRoute,
)


def _build_registry() -> FeatureInterpreterRegistry:
    registry = FeatureInterpreterRegistry()
    for interpreter in builtin_interpreters():
        registry.register(interpreter)
    return registry.freeze()


V4_INTERPRETER_REGISTRY = _build_registry()
V4_VALIDATED_TEMPLATES: dict[str, ValidatedFeatureTemplate] = {
    template.id: V4_INTERPRETER_REGISTRY.validate_template(template)
    for template in (
        V4_STANDARD_SEGMENT_TEMPLATE,
        V4_RAYLIGHT_SEGMENT_TEMPLATE,
    )
}


def _plain_feature_params(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_feature_params(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_feature_params(item) for item in value]
    return value


def _published_node_ids(value: PublishedValueRef) -> tuple[str, ...]:
    if isinstance(value, (EdgeRef, TerminalRef)):
        return (value.node_id,)
    children = value.items if isinstance(value, ListRef) else value.fields.values()
    ordered: list[str] = []
    for child in children:
        for node_id in _published_node_ids(child):
            if node_id not in ordered:
                ordered.append(node_id)
    return tuple(ordered)


def _plain_published_ref(value: PublishedValueRef) -> Any:
    if isinstance(value, EdgeRef):
        return [value.node_id, value.output_slot]
    if isinstance(value, TerminalRef):
        return {"kind": "terminal", "node_id": value.node_id}
    if isinstance(value, ListRef):
        return [_plain_published_ref(item) for item in value.items]
    return {
        key: _plain_published_ref(item) for key, item in value.fields.items()
    }


def _published_edge_pointers(
    value: PublishedValueRef,
    *,
    pointer: str,
) -> tuple[tuple[str, EdgeRef], ...]:
    if isinstance(value, EdgeRef):
        return ((pointer, value),)
    if isinstance(value, TerminalRef):
        raise AssertionError("terminal resources cannot be consumed by graph inputs")
    if isinstance(value, ListRef):
        return tuple(
            leaf
            for index, item in enumerate(value.items)
            for leaf in _published_edge_pointers(
                item,
                pointer=f"{pointer}/{index}",
            )
        )
    return tuple(
        leaf
        for key, item in value.fields.items()
        for leaf in _published_edge_pointers(
            item,
            pointer=(
                f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}"
            ),
        )
    )


def _read_resources(
    pool: ResourcePool,
    entry: FeatureTemplateEntry,
) -> dict[str, Any]:
    inputs: dict[str, Any] = {}
    for declaration in entry.reads:
        resource = (
            pool.read_required(
                declaration.name,
                expected_type=declaration.type,
            )
            if declaration.required
            else pool.read_optional(
                declaration.name,
                expected_type=declaration.type,
            )
        )
        if resource is not None:
            inputs[declaration.name] = resource
    return inputs


def _scope_public_reads(
    *,
    inputs: Mapping[str, Resource],
    scope: Any,
) -> tuple[PublicResourceRead, ...]:
    """Explain every cross-scope input, including composite resources, once."""

    local_nodes = set(scope.emitted_node_ids)
    external_evidence = tuple(
        item
        for item in scope.input_edge_evidence
        if item.value.node_id not in local_nodes
    )
    evidence_by_key = {
        (item.consumer_node_id, item.input_pointer, item.value): item
        for item in external_evidence
    }
    if len(evidence_by_key) != len(external_evidence):
        raise AssertionError(
            f"feature {scope.feature_id!r} produced duplicate input-edge evidence"
        )

    reads: list[PublicResourceRead] = []
    claimed: set[tuple[str, str, EdgeRef]] = set()
    for consumer_node_id, node in scope.prompt_fragment.items():
        for input_name, actual_value in node["inputs"].items():
            pointer = (
                f"/{consumer_node_id}/inputs/"
                f"{input_name.replace('~', '~0').replace('/', '~1')}"
            )
            matching_resources = [
                resource
                for resource in inputs.values()
                if _plain_published_ref(resource.value) == actual_value
            ]
            related_evidence = tuple(
                item
                for item in external_evidence
                if item.consumer_node_id == consumer_node_id
                and (
                    item.input_pointer == pointer
                    or item.input_pointer.startswith(pointer + "/")
                )
            )
            if not related_evidence:
                continue
            if len(matching_resources) != 1:
                raise AssertionError(
                    f"feature {scope.feature_id!r} cross-scope input at "
                    f"{pointer!r} must match exactly one typed resource"
                )
            resource = matching_resources[0]
            expected = set(
                (consumer_node_id, leaf_pointer, value)
                for leaf_pointer, value in _published_edge_pointers(
                    resource.value,
                    pointer=pointer,
                )
            )
            observed = {
                (item.consumer_node_id, item.input_pointer, item.value)
                for item in related_evidence
            }
            if observed != expected:
                raise AssertionError(
                    f"feature {scope.feature_id!r} composite resource at "
                    f"{pointer!r} does not match its exact typed leaves"
                )
            if claimed & expected:
                raise AssertionError(
                    f"feature {scope.feature_id!r} cross-scope evidence has "
                    "more than one resource owner"
                )
            claimed.update(expected)
            reads.append(
                PublicResourceRead(
                    resource_name=resource.name,
                    type=resource.type,
                    revision=resource.revision,
                    consumer_node_id=consumer_node_id,
                    input_pointer=pointer,
                    value=resource.value,
                )
            )

    unclaimed = set(evidence_by_key) - claimed
    if unclaimed:
        _consumer, pointer, _value = min(
            unclaimed,
            key=lambda item: (item[0], item[1], item[2].node_id, item[2].output_slot),
        )
        raise AssertionError(
            f"feature {scope.feature_id!r} cross-scope edge at {pointer!r} "
            "must match exactly one whole typed resource"
        )
    return tuple(reads)


def _scope_public_writes(
    *,
    before: ResourcePool,
    after: ResourcePool,
    entry: FeatureTemplateEntry,
    emission: FeatureEmission,
) -> tuple[PublicResourceWrite, ...]:
    emitted_names = set(emission.outputs)
    writes: list[PublicResourceWrite] = []
    for declaration in entry.writes:
        if declaration.name not in emitted_names:
            continue
        resource = after.resources[declaration.name]
        previous = before.resources.get(declaration.name)
        writes.append(
            PublicResourceWrite(
                operation=declaration.operation,
                resource=resource,
                previous_revision=(
                    previous.revision if previous is not None else None
                ),
            )
        )
    return tuple(writes)


def _scope_trace_parts(
    *,
    entry: FeatureTemplateEntry,
    resolution: FeatureResolution,
    scope: Any,
) -> tuple[str, FeatureResolution, tuple[ResolvedNodeEmission, ...]]:
    implementation_by_class = {
        implementation.class_type: implementation
        for implementation in resolution.implementations
    }
    if len(implementation_by_class) != len(resolution.implementations):
        raise AssertionError(
            f"feature {entry.id!r} resolved duplicate class implementations"
        )
    emitted: list[ResolvedNodeEmission] = []
    fragment = scope.prompt_fragment
    for node_id, node in fragment.items():
        class_type = node["class_type"]
        implementation = implementation_by_class.get(class_type)
        if implementation is None:
            raise AssertionError(
                f"feature {entry.id!r} emitted unresolved node {class_type!r}"
            )
        emitted.append(
            ResolvedNodeEmission(
                node_id=node_id,
                implementation_binding_key=implementation.binding_key,
                output_affecting=(
                    class_type not in V4_OUTPUT_NEUTRAL_NODE_CLASSES
                ),
            )
        )
    if set(implementation_by_class) != {
        fragment[item.node_id]["class_type"] for item in emitted
    }:
        raise AssertionError(
            f"feature {entry.id!r} resolution contains an implementation "
            "which emitted no node"
        )
    return entry.id, resolution, tuple(emitted)


def _node_contract_snapshot(
    prompt: Mapping[str, Any],
    node_contract_registry: NodeContractRegistry = V4_NODE_CONTRACT_REGISTRY,
) -> dict[str, NodeContractEvidence]:
    snapshot: dict[str, NodeContractEvidence] = {}
    for node_id, node in prompt.items():
        contract = node_contract_registry.require(node["class_type"])
        if len(contract.allowed_python_modules) != 1:
            raise AssertionError(
                f"v4 node contract must select one module: {contract.class_type}"
            )
        snapshot[str(node_id)] = NodeContractEvidence(
            contract_id=contract.contract_id,
            semantic_version=contract.semantic_version,
            class_type=contract.class_type,
            python_module=contract.allowed_python_modules[0],
            runtime_fingerprint=contract.supported_runtime_fingerprints[0],
            execution_terminal_role=contract.execution_terminal_role,
            persistent_artifact_role=contract.persistent_artifact_role,
        )
    return snapshot


def _allowed_late_bindings(
    *,
    prompt: Mapping[str, Any],
    continuity: NativeContinuityDependency | None,
    backend: str,
) -> tuple[AllowedLateBoundInput, ...]:
    allowed: list[AllowedLateBoundInput] = []
    if continuity is not None:
        allowed.append(
            AllowedLateBoundInput(
                input_pointer=(
                    f"/{continuity.load_video_node_id}/inputs/file"
                ),
                value_kind="string",
                source_kind="continuity",
            )
        )
    if backend == "raylight":
        initializer_ids = [
            str(node_id)
            for node_id, node in prompt.items()
            if node.get("class_type") == "DirectorDeckRayInitializerAdvanced"
        ]
        if len(initializer_ids) != 1:
            raise AssertionError("RayLight segment must contain one initializer")
        allowed.append(
            AllowedLateBoundInput(
                input_pointer=(
                    f"/{initializer_ids[0]}/inputs/ray_cluster_namespace"
                ),
                value_kind="string",
                source_kind="runtime_epoch",
            )
        )
    return tuple(allowed)


def _commit_emission(
    *,
    pool: ResourcePool,
    entry: FeatureTemplateEntry,
    emission: FeatureEmission,
    scope: Any,
) -> ResourcePool:
    writes = {declaration.name: declaration for declaration in entry.writes}
    unexpected = set(emission.outputs) - set(writes)
    if unexpected:
        raise AssertionError(
            f"feature {entry.id!r} emitted undeclared resources: "
            + ", ".join(sorted(unexpected))
        )
    missing = {
        name
        for name, declaration in writes.items()
        if declaration.required and name not in emission.outputs
    }
    if missing:
        raise AssertionError(
            f"feature {entry.id!r} omitted declared resources: "
            + ", ".join(sorted(missing))
        )

    transaction = pool.begin()
    for name, value in emission.outputs.items():
        declaration = writes[name]
        producers = _published_node_ids(value)
        if declaration.operation == "define":
            transaction = transaction.define(
                name=name,
                type=declaration.type,
                value=value,
                source_feature_id=entry.id,
                producer_node_ids=producers,
            )
        else:
            transaction = transaction.replace(
                name=name,
                value=value,
                source_feature_id=entry.id,
                producer_node_ids=producers,
                expected_type=declaration.type,
            )
    return scope.commit_emission(emission, transaction)


def _entry_enabled(
    entry: FeatureTemplateEntry,
    route: V4ResolvedSegmentRoute,
) -> bool:
    return _route_feature_enabled(entry.id, route)


def _route_feature_enabled(
    entry_id: str,
    route: V4ResolvedSegmentRoute,
) -> bool:
    if entry_id == "lora":
        return route.lora_resolution is not None
    if entry_id == "continuity":
        return route.predecessor_segment_id is not None
    return True


def _continuity_load_node_id(prompt: Mapping[str, Any]) -> str:
    matches = [
        str(node_id)
        for node_id, node in prompt.items()
        if isinstance(node, Mapping)
        and node.get("class_type") == "LoadVideo"
        and isinstance(node.get("inputs"), Mapping)
        and node["inputs"].get("file") == _UNBOUND_PREDECESSOR_OUTPUT
    ]
    if len(matches) != 1:
        raise AssertionError(
            "continuity feature must emit exactly one predecessor placeholder"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class V4ResolvedFeatureBinding:
    """One immutable interpreter resolution shared by evaluate and emit."""

    interpreter: FeatureInterpreter
    resolution: FeatureResolution
    required_capabilities: CapabilitySet
    cache_identity: Any
    runtime_pool_identity: Any | None


class V4CapabilityEvaluationError(NativeTemplateError):
    """A live-host evaluator rejected an otherwise valid feature binding."""

    def __init__(self, evaluation: Any) -> None:
        reasons = tuple(getattr(evaluation, "reasons", ()))
        first = reasons[0] if reasons else None
        code = str(getattr(first, "code", "capability_unavailable"))
        message = str(
            getattr(first, "message", "The selected feature is unavailable.")
        )
        super().__init__(f"{code}: {message}")
        self.evaluation = evaluation


class EffectiveFeatureResolutionMismatch(NativeTemplateError):
    """A supplied effective snapshot disagrees with its immutable route."""

    def __init__(
        self,
        *,
        feature_id: str,
        segment_id: str,
        effective_active: bool,
        route_active: bool,
    ) -> None:
        super().__init__(
            "effective feature resolution mismatch for "
            f"{feature_id!r} on segment {segment_id!r}: "
            f"effective={effective_active}, route={route_active}"
        )
        self.feature_id = feature_id
        self.segment_id = segment_id
        self.effective_active = effective_active
        self.route_active = route_active


def require_effective_route_activation_match(
    *,
    entry_id: str,
    route: V4ResolvedSegmentRoute,
    effective_active: bool,
) -> None:
    """Cross-check contextual effective state without silently recomputing it."""

    if entry_id not in {"continuity", "lora"}:
        return
    route_active = _route_feature_enabled(entry_id, route)
    if effective_active != route_active:
        raise EffectiveFeatureResolutionMismatch(
            feature_id=entry_id,
            segment_id=route.segment_id,
            effective_active=effective_active,
            route_active=route_active,
        )


def resolve_effective_raylight_attention_mode(
    *,
    route: V4ResolvedSegmentRoute,
    effective_segment: EffectiveSegmentFeatures | None,
) -> RaylightAttentionMode:
    """Resolve the exact RayLight pool mode shared by preflight and compile."""

    if effective_segment is None or route.backend != "raylight":
        return "ck_int8"
    pool_feature = next(
        (
            feature
            for feature in effective_segment.features
            if feature.id == "raylight_pool_intent"
        ),
        None,
    )
    if pool_feature is None or not pool_feature.active:
        raise NativeTemplateError(
            "effective RayLight route requires active raylight_pool_intent"
        )
    pool_params = feature_parameter_model(
        pool_feature.id,
        pool_feature.version,
    ).model_validate(_plain_feature_params(pool_feature.params))
    attention = getattr(pool_params, "attention", None)
    if attention not in {"ck_int8", "torch_flash"}:
        raise NativeTemplateError(
            "effective RayLight pool intent has invalid attention mode"
        )
    return attention


def build_v4_route_context(
    route: V4ResolvedSegmentRoute,
    *,
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    job_id: str,
    timeline_assembly_required: bool = False,
    template_bundle_version: int = V4_TEMPLATE_BUNDLE.version,
    raylight_attention_mode: RaylightAttentionMode = "ck_int8",
) -> V4BuiltinContext:
    """Build the sole v4 graph-local context used by preflight and compile."""

    segment = route.materialize_segment()
    binding = getattr(settings.models, route.family)
    sampling = getattr(draft.sampling, route.family)
    visible_frames = _align_h3_frames(
        segment.duration_seconds,
        draft.render.fps,
    )
    context_frames = (
        route.continuity_overlap_frames
        if route.predecessor_segment_id is not None
        else 0
    )
    return V4BuiltinContext(
        backend=route.backend,
        family=route.family,
        template_bundle_version=template_bundle_version,
        settings=settings,
        draft=draft,
        segment=segment,
        binding=binding,
        sampling=sampling,
        job_id=job_id,
        visible_frames=visible_frames,
        sample_frames=_align_h3_frame_count(visible_frames + context_frames),
        continuity_prefix_frames=context_frames,
        lora_loader_node=(
            route.lora_resolution.loader_node
            if route.lora_resolution is not None
            else None
        ),
        lora_adapter_id=(
            route.lora_resolution.adapter_id
            if route.lora_resolution is not None
            else None
        ),
        lora_loader_binding=(
            route.lora_resolution.binding
            if route.lora_resolution is not None
            else None
        ),
        lora_resolution_source=(
            route.lora_resolution.source
            if route.lora_resolution is not None
            else None
        ),
        lora_adapter_options=(
            route.lora_resolution.options
            if route.lora_resolution is not None
            else None
        ),
        raylight_namespace=(
            (
                _raylight_namespace(route.family, binding)
                if template_bundle_version == V4_TEMPLATE_BUNDLE.version
                else raylight_runtime_namespace(
                    binding,
                    attention_mode=raylight_attention_mode,
                    enforce_attention_topology=True,
                )
            )
            if route.backend == "raylight"
            else None
        ),
        raylight_attention_mode=raylight_attention_mode,
        clear_raylight_vram_after_sampling=(
            route.clear_raylight_vram_after_sampling
        ),
        timeline_assembly_required=timeline_assembly_required,
    )


def resolve_v4_active_feature(
    *,
    entry: FeatureTemplateEntry,
    template: ValidatedFeatureTemplate,
    params: BaseModel,
    context: V4BuiltinContext,
) -> V4ResolvedFeatureBinding:
    """Run all pure interpreter contracts once, stopping before ``emit``."""

    interpreter = template.interpreter_for(entry)
    interpreter.validate_params(params, context)
    resolution = interpreter.resolve(params, context)
    if resolution.state != "active":
        raise AssertionError(
            f"active v4 template feature {entry.id!r} resolved as noop"
        )
    required = interpreter.required_capabilities(params, context, resolution)
    cache_identity = interpreter.cache_identity(params, context, resolution)
    runtime_pool_identity = interpreter.runtime_pool_identity(
        params,
        context,
        resolution,
    )
    return V4ResolvedFeatureBinding(
        interpreter=interpreter,
        resolution=resolution,
        required_capabilities=required,
        cache_identity=cache_identity,
        runtime_pool_identity=runtime_pool_identity,
    )


def _compile_route(
    *,
    route: V4ResolvedSegmentRoute,
    compile_input: V4CreativeCompileInput,
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    job_id: str,
    host_capability_snapshot: HostCapabilitySnapshot | None = None,
    operational_readiness: OperationalReadiness | None = None,
    capability_evaluator: Any | None = None,
    validated_templates: Mapping[str, ValidatedFeatureTemplate] = V4_VALIDATED_TEMPLATES,
    template_bundle_version: int = V4_TEMPLATE_BUNDLE.version,
    effective_segment: EffectiveSegmentFeatures | None = None,
    node_contract_registry: NodeContractRegistry = V4_NODE_CONTRACT_REGISTRY,
) -> tuple[NativeWorkflowUnit, dict[str, Any]]:
    raylight_attention_mode = resolve_effective_raylight_attention_mode(
        route=route,
        effective_segment=effective_segment,
    )
    context = build_v4_route_context(
        route,
        draft=draft,
        settings=settings,
        job_id=job_id,
        timeline_assembly_required=(
            compile_input.requires_timeline_assembly()
        ),
        template_bundle_version=template_bundle_version,
        raylight_attention_mode=raylight_attention_mode,
    )
    segment = context.segment
    sampling = context.sampling
    visible_frames = context.visible_frames
    context_frames = context.continuity_prefix_frames
    sample_frames = context.sample_frames
    template = validated_templates[route.template_id]
    graph = PromptGraphBuilder()
    pool = ResourcePool()
    effective_by_id = (
        {feature.id: feature for feature in effective_segment.features}
        if effective_segment is not None
        else None
    )
    if effective_segment is not None:
        if (
            effective_segment.segment_id != route.segment_id
            or effective_segment.backend != route.backend
            or effective_segment.family != route.family
            or effective_segment.template_id != route.template_id
            or tuple(effective_by_id or ())
            != tuple(entry.id for entry in template.template.entries)
        ):
            raise NativeTemplateError(
                "effective feature resolution does not match the compile route"
            )
    public_writes: list[PublicResourceWrite] = []
    public_reads: list[PublicResourceRead] = []
    trace_parts: list[
        tuple[str, FeatureResolution, tuple[ResolvedNodeEmission, ...]]
    ] = []
    compile_feature_resolutions: list[CompiledFeatureResolution] = []
    compile_feature_notices: list[CompiledFeatureNotice] = []
    feature_identity_evidence: list[NativeFeatureIdentityEvidence] = []
    scoped_feature_emissions: list[
        tuple[tuple[str, ...], FeatureEmission]
    ] = []

    for entry in template.template.entries:
        effective = (
            effective_by_id[entry.id]
            if effective_by_id is not None
            else None
        )
        enabled = (
            effective.active
            if effective is not None
            else _entry_enabled(entry, route)
        )
        # Continuity and LoRA have route-local activation conditions which are
        # already proven by the legacy route resolver.  Cross-check instead of
        # trusting an earlier advisory projection.
        if entry.id in {"continuity", "lora"}:
            route_enabled = _entry_enabled(entry, route)
            if effective is not None:
                require_effective_route_activation_match(
                    entry_id=entry.id,
                    route=route,
                    effective_active=effective.active,
                )
            enabled = route_enabled
        if not enabled:
            resolution = FeatureResolution(
                state="noop",
                implementations=(),
                resolution_details={
                    "reason": (
                        "disabled_by_v5_effective_config"
                        if effective is not None
                        else "disabled_by_v4_context"
                    )
                },
            )
            compile_feature_resolutions.append(
                CompiledFeatureResolution(
                    segment_id=route.segment_id,
                    unit_id=route.unit_id,
                    feature_id=entry.id,
                    version=entry.version,
                    backend=route.backend,
                    family=route.family,
                    template_id=route.template_id,
                    resolution=resolution,
                    adapter_fingerprint=canonical_sha256(
                        {
                            "schema_version": 1,
                            "feature_id": entry.id,
                            "feature_version": entry.version,
                            "backend": context.backend,
                            "family": context.family,
                            "implementations": (),
                        }
                    ),
                    capability=CompiledCapabilityEvaluation(available=True),
                )
            )
            continue
        params: BaseModel = (
            feature_parameter_model(entry.id, entry.version).model_validate(
                _plain_feature_params(effective.params)
            )
            if effective is not None
            else V4BuiltinParams()
        )
        feature_binding = resolve_v4_active_feature(
            entry=entry,
            template=template,
            params=params,
            context=context,
        )
        interpreter = feature_binding.interpreter
        resolution = feature_binding.resolution
        if effective is not None:
            feature_identity_evidence.append(
                NativeFeatureIdentityEvidence(
                    feature_id=entry.id,
                    cache_identity=feature_binding.cache_identity,
                    runtime_pool_identity=(
                        feature_binding.runtime_pool_identity
                    ),
                )
            )
        if entry.id == "lora":
            assert route.lora_resolution is not None
            if route.lora_resolution.loader_node not in {
                implementation.class_type
                for implementation in resolution.implementations
            }:
                raise AssertionError("resolved LoRA evidence drifted before emission")
        capability = CompiledCapabilityEvaluation(available=True)
        if capability_evaluator is not None:
            evaluation = capability_evaluator.evaluate(
                feature_id=entry.id,
                ctx=context,
                resolution=resolution,
                required_capabilities=feature_binding.required_capabilities,
                snapshot=host_capability_snapshot,
                readiness=operational_readiness,
                segment_id=route.segment_id,
                unit_id=route.unit_id,
            )
            if not evaluation.available:
                raise V4CapabilityEvaluationError(evaluation)
            capability = CompiledCapabilityEvaluation.model_validate_json(
                evaluation.model_dump_json()
            )
        compile_feature_resolutions.append(
            CompiledFeatureResolution(
                segment_id=route.segment_id,
                unit_id=route.unit_id,
                feature_id=entry.id,
                version=entry.version,
                backend=route.backend,
                family=route.family,
                template_id=route.template_id,
                resolution=resolution,
                adapter_fingerprint=canonical_sha256(
                    {
                        "schema_version": 1,
                        "feature_id": entry.id,
                        "feature_version": entry.version,
                        "backend": context.backend,
                        "family": context.family,
                        "implementations": tuple(
                            implementation.model_dump(mode="json")
                            for implementation in resolution.implementations
                        ),
                    }
                ),
                capability=capability,
            )
        )
        inputs = _read_resources(pool, entry)
        before_pool = pool
        with graph.begin_scope(entry.id) as scope:
            emission = interpreter.emit(
                scope,
                inputs,
                params,
                context,
                resolution,
            )
            pool = _commit_emission(
                pool=pool,
                entry=entry,
                emission=emission,
                scope=scope,
            )
        compile_feature_notices.extend(
            CompiledFeatureNotice(
                segment_id=route.segment_id,
                unit_id=route.unit_id,
                feature_id=entry.id,
                message=notice,
            )
            for notice in emission.notices
        )
        scoped_feature_emissions.append((scope.emitted_node_ids, emission))
        public_reads.extend(_scope_public_reads(inputs=inputs, scope=scope))
        public_writes.extend(
            _scope_public_writes(
                before=before_pool,
                after=pool,
                entry=entry,
                emission=emission,
            )
        )
        trace_parts.append(
            _scope_trace_parts(
                entry=entry,
                resolution=resolution,
                scope=scope,
            )
        )

    take = pool.read_required(
        "take_output",
        expected_type="TAKE",
        allow_terminal=True,
    )
    if not isinstance(take.value, TerminalRef):
        raise AssertionError("v4 save_take must publish one terminal output")

    prompt = graph.prompt
    continuity: NativeContinuityDependency | None = None
    if route.predecessor_segment_id is not None:
        continuity = NativeContinuityDependency(
            predecessor_segment_id=route.predecessor_segment_id,
            overlap_frames=context_frames,
            load_video_node_id=_continuity_load_node_id(prompt),
            source=route.continuity_source or "same_run",
            historical_take_id=(
                route.historical_take.id
                if route.historical_take is not None
                else None
            ),
        )
    audited_writes = tuple(public_writes)
    structural_features = {
        write.resource.source_feature_id for write in audited_writes
    }
    feature_traces = tuple(
        FeatureAuditTrace(
            feature_id=feature_id,
            resolution=resolution,
            emitted_nodes=emitted_nodes,
            structural_influence=feature_id in structural_features,
        )
        for feature_id, resolution, emitted_nodes in trace_parts
    )
    graph_audit_spec = build_graph_audit_spec(
        prompt=prompt,
        node_contract_registry=node_contract_registry,
        node_contract_snapshot=_node_contract_snapshot(
            prompt,
            node_contract_registry,
        ),
        public_writes=audited_writes,
        public_reads=public_reads,
        feature_traces=feature_traces,
        model_family=route.family,
        backend=route.backend,
        allowed_late_bound_inputs=_allowed_late_bindings(
            prompt=prompt,
            continuity=continuity,
            backend=route.backend,
        ),
        unit_kind="segment",
        take_node_id=take.value.node_id,
        enforce_runtime_effects=False,
    )
    progress_spec, preview_spec = derive_feature_execution_specs(
        scoped_feature_emissions
    )
    unit = NativeWorkflowUnit(
        id=route.unit_id,
        family=route.family,
        backend=route.backend,
        segment_ids=(segment.id,),
        prompt=prompt,
        output_nodes={segment.id: take.value.node_id},
        continuity=continuity,
        graph_audit_spec=graph_audit_spec,
        graph_audit_traces=feature_traces,
        compile_feature_resolutions=tuple(compile_feature_resolutions),
        compile_feature_notices=tuple(compile_feature_notices),
        feature_identity_evidence=tuple(feature_identity_evidence),
        progress_spec=progress_spec,
        preview_spec=preview_spec,
    )
    if route.historical_take is not None:
        unit = bind_native_workflow_predecessor_output(
            unit,
            route.historical_take.materialize_output(),
        )

    node_classes = tuple(
        dict.fromkeys(node["class_type"] for node in prompt.values())
    )
    plan = {
        "segment_id": segment.id,
        "mode": segment.mode,
        "recipe": timeline_segment_recipe(segment),
        "model_family": route.family,
        "backend": route.backend,
        "frame_count": visible_frames,
        "visible_frame_count": visible_frames,
        "sample_frame_count": sample_frames,
        "continuity_context_frames": context_frames,
        "alignment_tail_frame_count": (
            sample_frames - visible_frames - context_frames
        ),
        "predecessor_segment_id": route.predecessor_segment_id,
        "continuity_source": route.continuity_source,
        "historical_take_id": (
            route.historical_take.id if route.historical_take is not None else None
        ),
        "anchor_reset": route.anchor_reset,
        "seed_mode": "random" if sampling.random_seed else "fixed",
        "seed": sampling.seed,
        "conditioning_node": (
            "MiniMaxH3ImageToVideo"
            if segment.mode == "fl2va"
            else "MiniMaxH3ReferenceToVideo"
        ),
        "node_classes": list(node_classes),
    }
    return unit, plan


def _manifest(
    *,
    compile_input: V4CreativeCompileInput,
    workflows: list[NativeWorkflowUnit],
) -> dict[str, Any]:
    lora_resolution: dict[str, dict[str, Any]] = {}
    for family in compile_input.families:
        resolution = next(
            (
                route.lora_resolution
                for route in compile_input.routes
                if route.family == family and route.lora_resolution is not None
            ),
            None,
        )
        if resolution is None:
            continue
        lora_resolution[family] = {
            "lora_name": resolution.lora_name,
            "model_filename": resolution.model_filename,
            "backend": resolution.backend,
            "adapter_id": resolution.adapter_id,
            "binding": (
                resolution.binding.model_dump(mode="json")
                if resolution.binding is not None
                else None
            ),
            "loader_node": resolution.loader_node,
            "source": resolution.source,
        }

    return {
        "version": 2,
        "graph_source": "server",
        "accepts_client_workflow": False,
        "continuity": {
            "boundaries": [
                {
                    "segment_id": route.segment_id,
                    "predecessor_segment_id": route.predecessor_segment_id,
                    "overlap_frames": route.continuity_overlap_frames,
                    "source": route.continuity_source,
                    "historical_take_id": (
                        route.historical_take.id
                        if route.historical_take is not None
                        else None
                    ),
                }
                for route in compile_input.routes
                if route.predecessor_segment_id is not None
            ],
        },
        "submission_order": [unit.id for unit in workflows],
        "raylight_exclusive": any(
            unit.backend == "raylight" for unit in workflows
        ),
        "lora_resolution": lora_resolution,
        "resident_cache_scope": {
            "boundary": "comfy_endpoint",
            "standard": "family+model_loader_inputs",
            "prompt_partition": "one_segment",
            "raylight_initializer": "gpu_pool+topology",
            "raylight_model": "worker_ram_cache(model+lora+weight_dtype)",
            "raylight_cuda_residency": (
                "kept_for_compatible_key_until_explicit_switch"
                if compile_input.keep_raylight_resident
                else "released_after_each_sampler"
            ),
            "raylight_residency_reason": (
                "explicit_keyed_switch_policy"
                if compile_input.keep_raylight_resident
                else "shared_endpoint_safe_default"
            ),
            "raylight_resident_family": None,
        },
        "units": [
            {
                "id": unit.id,
                "family": unit.family,
                "backend": unit.backend,
                "segment_ids": list(unit.segment_ids),
                "output_nodes": dict(unit.output_nodes),
                "continuity": (
                    {
                        "predecessor_segment_id": (
                            unit.continuity.predecessor_segment_id
                        ),
                        "overlap_frames": unit.continuity.overlap_frames,
                        "load_video_node_id": unit.continuity.load_video_node_id,
                        "source": unit.continuity.source,
                        "historical_take_id": unit.continuity.historical_take_id,
                        "resolved": unit.continuity.resolved,
                    }
                    if unit.continuity is not None
                    else None
                ),
            }
            for unit in workflows
        ],
    }


def _compile_resolved_input(
    *,
    compile_input: V4CreativeCompileInput,
    resolved_draft: UnifiedTimelineDraft,
    resolved_settings: RuntimeSettings,
    job_id: str,
    host_capability_snapshot: HostCapabilitySnapshot | None,
    operational_readiness: OperationalReadiness | None,
    capability_evaluator: Any | None,
    validated_templates: Mapping[str, ValidatedFeatureTemplate],
    template_bundle_version: int,
    effective_features: EffectiveFeatureConfiguration | None = None,
    node_contract_registry: NodeContractRegistry = V4_NODE_CONTRACT_REGISTRY,
) -> NativeCompileResult:
    if effective_features is not None:
        route_ids = {route.segment_id for route in compile_input.routes}
        if (
            effective_features.template_bundle_version != template_bundle_version
            or set(effective_features.effective_by_segment) != route_ids
        ):
            raise NativeTemplateError(
                "effective feature resolution does not exactly cover compile routes"
            )

    workflows: list[NativeWorkflowUnit] = []
    plans: list[dict[str, Any]] = []
    for route in compile_input.routes:
        unit, plan = _compile_route(
            route=route,
            compile_input=compile_input,
            draft=resolved_draft,
            settings=resolved_settings,
            job_id=job_id,
            host_capability_snapshot=host_capability_snapshot,
            operational_readiness=operational_readiness,
            capability_evaluator=capability_evaluator,
            validated_templates=validated_templates,
            template_bundle_version=template_bundle_version,
            effective_segment=(
                effective_features.effective_by_segment[route.segment_id]
                if effective_features is not None
                else None
            ),
            node_contract_registry=node_contract_registry,
        )
        workflows.append(unit)
        plans.append(plan)

    all_nodes = sorted(
        {
            node["class_type"]
            for unit in workflows
            for node in unit.prompt.values()
        }
    )
    provenance = dict(
        native_provenance_policy(all_nodes)
        if node_contract_registry is V4_NODE_CONTRACT_REGISTRY
        else current_provenance_policy(all_nodes)
    )
    custom_nodes = sorted(
        node
        for node in all_nodes
        if provenance[node] in {
            "raylight",
            "lora-custom",
            "director-owned-strict-attention",
            "director-owned-strict-h3",
        }
    )
    return NativeCompileResult(
        workflows=tuple(workflows),
        manifest=_manifest(
            compile_input=compile_input,
            workflows=workflows,
        ),
        plans=tuple(plans),
        families=compile_input.families,
        node_policy={
            "graph_source": "server",
            "accepts_client_workflow": False,
            "allowed_nodes": all_nodes,
            "custom_nodes": custom_nodes,
            "provenance": {node: provenance[node] for node in all_nodes},
        },
    )


def compile_v4_timeline(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    job_id: str,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, NativeHistoricalTake] | None = None,
    resolved_lora_adapters: Mapping[str, ResolvedLoraAdapter] | None = None,
    host_capability_snapshot: HostCapabilitySnapshot | None = None,
    operational_readiness: OperationalReadiness | None = None,
    capability_evaluator: Any | None = None,
) -> NativeCompileResult:
    """Compile exact v4 prompts through validated templates/interpreters."""

    capability_inputs = (
        host_capability_snapshot,
        operational_readiness,
        capability_evaluator,
    )
    if any(item is not None for item in capability_inputs) and not all(
        item is not None for item in capability_inputs
    ):
        raise TypeError(
            "capability snapshot, readiness, and evaluator must be supplied together"
        )

    try:
        compile_input = CreativeCompileInputResolver.resolve_v4(
            draft,
            settings,
            segment_ids,
            historical_takes,
            resolved_lora_adapters,
        )
    except CreativeCompileInputError as exc:
        raise NativeTemplateError(str(exc)) from exc

    resolved_draft = compile_input.materialize_draft()
    resolved_settings = compile_input.materialize_settings()
    return _compile_resolved_input(
        compile_input=compile_input,
        resolved_draft=resolved_draft,
        resolved_settings=resolved_settings,
        job_id=job_id,
        host_capability_snapshot=host_capability_snapshot,
        operational_readiness=operational_readiness,
        capability_evaluator=capability_evaluator,
        validated_templates=V4_VALIDATED_TEMPLATES,
        template_bundle_version=V4_TEMPLATE_BUNDLE.version,
        node_contract_registry=V4_NODE_CONTRACT_REGISTRY,
    )


def compile_projected_v5_timeline(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    job_id: str,
    effective_features: EffectiveFeatureConfiguration,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, NativeHistoricalTake] | None = None,
    resolved_lora_adapters: Mapping[str, ResolvedLoraAdapter] | None = None,
    host_capability_snapshot: HostCapabilitySnapshot | None = None,
    operational_readiness: OperationalReadiness | None = None,
    capability_evaluator: Any | None = None,
) -> NativeCompileResult:
    """Compile a v5 authority projection through the current exact bundle."""

    capability_inputs = (
        host_capability_snapshot,
        operational_readiness,
        capability_evaluator,
    )
    if any(item is not None for item in capability_inputs) and not all(
        item is not None for item in capability_inputs
    ):
        raise TypeError(
            "capability snapshot, readiness, and evaluator must be supplied together"
        )
    try:
        compile_input = CreativeCompileInputResolver.resolve_v4(
            draft,
            settings,
            segment_ids,
            historical_takes,
            resolved_lora_adapters,
        )
    except CreativeCompileInputError as exc:
        raise NativeTemplateError(str(exc)) from exc
    from .node_contracts import CURRENT_NODE_CONTRACT_REGISTRY
    from .v5_registry import V5_VALIDATED_TEMPLATES

    return _compile_resolved_input(
        compile_input=compile_input,
        resolved_draft=compile_input.materialize_draft(),
        resolved_settings=compile_input.materialize_settings(),
        job_id=job_id,
        host_capability_snapshot=host_capability_snapshot,
        operational_readiness=operational_readiness,
        capability_evaluator=capability_evaluator,
        validated_templates=V5_VALIDATED_TEMPLATES,
        template_bundle_version=effective_features.template_bundle_version,
        effective_features=effective_features,
        node_contract_registry=CURRENT_NODE_CONTRACT_REGISTRY,
    )


__all__ = [
    "EffectiveFeatureResolutionMismatch",
    "V4CapabilityEvaluationError",
    "V4_INTERPRETER_REGISTRY",
    "V4ResolvedFeatureBinding",
    "V4_VALIDATED_TEMPLATES",
    "build_v4_route_context",
    "compile_projected_v5_timeline",
    "compile_v4_timeline",
    "resolve_effective_raylight_attention_mode",
    "resolve_v4_active_feature",
    "require_effective_route_activation_match",
]
