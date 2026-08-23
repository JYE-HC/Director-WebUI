from __future__ import annotations

"""Behavior-preserving views over the pre-extensible workflow records.

These adapters are deliberately evidence-only.  They freeze the fields which
the native v4 compiler or legacy SQLite rows actually own, and describe absent
new-contract evidence explicitly.  They do not manufacture prepared units,
exact prompt snapshots, prompt ownership, locked plans, or observed media.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BeforeValidator, Field, ValidationError, model_validator

from ..native_templates import (
    NativeCompileResult,
    NativeContinuityDependency,
    NativeWorkflowUnit,
)
from ..schemas import JobRead
from .audit import FeatureAuditTrace
from .contracts import (
    ContractModel,
    FrozenMap,
    GraphAuditSpec,
    HostCapabilitySnapshot,
    JsonObject,
    MAX_JSON_CONTAINER_ITEMS,
    MAX_JSON_DEPTH,
    NodeContractRegistry,
    PublishedValueRef,
    ResourcePool,
    RuntimeEffectContract,
    TemplateBundle,
)
from .execution import (
    CompiledExecutionPlan,
    ComfyNodeCacheIdentity,
    DocumentDigest,
    ExactPromptSnapshot,
    ExpectedOutputSpec,
    HistoricalTakeGeometryIdentity,
    LegacyOutputLocator,
    LockedSubmissionPlan,
    ObservedArtifactSpec,
    OutputDescriptor,
    PreparedControlUnit,
    PreparedSegmentUnit,
    PreviewSpec,
    ProgressSpec,
    PromptOwnership,
    RayRuntimeIdentity,
    RuntimeRequirements,
    SegmentExecutionIdentity,
    comfy_node_cache_identity,
    sha256_document_digest,
)


def _coerce_json_containers(
    value: Any,
    *,
    _depth: int = 1,
    _ancestors: set[int] | None = None,
) -> Any:
    """Copy JSON containers into the immutable contract representation.

    ``JsonObject`` intentionally uses tuples internally.  A before-validator is
    required here because legacy prompt/database documents arrive as ordinary
    JSON lists, including when a frozen envelope is read back from JSON.
    """

    if _depth > MAX_JSON_DEPTH:
        raise ValueError(f"legacy JSON exceeds maximum depth {MAX_JSON_DEPTH}")
    ancestors = _ancestors if _ancestors is not None else set()
    if isinstance(value, (Mapping, list, tuple)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("legacy JSON cannot contain reference cycles")
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ValueError("legacy JSON container exceeds maximum length")
        ancestors.add(identity)
        try:
            if isinstance(value, Mapping):
                return {
                    str(key): _coerce_json_containers(
                        item,
                        _depth=_depth + 1,
                        _ancestors=ancestors,
                    )
                    for key, item in value.items()
                }
            return tuple(
                _coerce_json_containers(
                    item,
                    _depth=_depth + 1,
                    _ancestors=ancestors,
                )
                for item in value
            )
        finally:
            ancestors.remove(identity)
    return value


LegacyJsonObject: TypeAlias = Annotated[
    JsonObject,
    BeforeValidator(_coerce_json_containers),
]


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


LegacyUnavailableReason = Literal[
    "not_persisted",
    "legacy_value_overwritten",
    "ambiguous_legacy_shape",
    "unversioned_snapshot",
    "no_media_probe",
    "implicit_runtime_fallback",
]


class LegacyUnavailableEvidence(ContractModel):
    """One new-contract field which a legacy source cannot prove."""

    field_path: Annotated[str, Field(min_length=1, max_length=512)]
    reason: LegacyUnavailableReason
    source_fields: tuple[Annotated[str, Field(min_length=1, max_length=256)], ...] = ()
    detail: Annotated[str, Field(min_length=1, max_length=1_024)] | None = None


def _unavailable(
    field_path: str,
    reason: LegacyUnavailableReason,
    *source_fields: str,
    detail: str | None = None,
) -> LegacyUnavailableEvidence:
    return LegacyUnavailableEvidence(
        field_path=field_path,
        reason=reason,
        source_fields=tuple(source_fields),
        detail=detail,
    )


class LegacyNativeContinuitySnapshot(ContractModel):
    predecessor_segment_id: str
    overlap_frames: int
    load_video_node_id: str
    source: Literal["same_run", "historical_take"]
    historical_take_id: str | None = None
    resolved: bool = False
    bound_file: str | None = None


class LegacyNativeWorkflowUnitSnapshot(ContractModel):
    """Frozen, lossless projection of the current ``NativeWorkflowUnit``."""

    schema_version: Literal[1] = 1
    id: str
    family: Literal["fl2va", "ref2va"]
    backend: Literal["standard", "raylight"]
    segment_ids: tuple[str, ...]
    prompt: LegacyJsonObject
    output_nodes: FrozenMap[str, str]
    continuity: LegacyNativeContinuitySnapshot | None = None
    graph_audit_spec: GraphAuditSpec | None = None
    graph_audit_traces: tuple[FeatureAuditTrace, ...] = ()
    raylight_runtime_epoch: int | None = None
    raylight_runtime_namespace: str | None = None


class LegacyNativeCompileSnapshot(ContractModel):
    """Frozen, lossless projection of the current ``NativeCompileResult``."""

    schema_version: Literal[1] = 1
    workflows: tuple[LegacyNativeWorkflowUnitSnapshot, ...]
    manifest: LegacyJsonObject
    plans: tuple[LegacyJsonObject, ...]
    families: tuple[Literal["fl2va", "ref2va"], ...]
    node_policy: LegacyJsonObject


class LegacyNodePolicySnapshot(ContractModel):
    """The fields the old compile response actually declared about nodes."""

    graph_source: str
    accepts_client_workflow: bool
    allowed_nodes: tuple[str, ...]
    custom_nodes: tuple[str, ...]
    provenance: FrozenMap[str, str]


T = TypeVar("T")


class LegacyAvailableContract(ContractModel, Generic[T]):
    status: Literal["available"] = "available"
    value: T
    provenance: Literal["legacy_exact", "derived_from_frozen_legacy"]
    limitations: tuple[LegacyUnavailableEvidence, ...] = ()


class LegacyUnavailableContract(ContractModel):
    status: Literal["unavailable"] = "unavailable"
    evidence: Annotated[
        tuple[LegacyUnavailableEvidence, ...],
        Field(min_length=1),
    ]


class LegacyUnitCacheIdentity(ContractModel):
    unit_id: str
    identity: ComfyNodeCacheIdentity


class LegacyValidatedTakeDescriptor(ContractModel):
    """A path-safe descriptor, not a claim about the output media contents."""

    source_take_id: str | None = None
    descriptor: OutputDescriptor


class LegacyStage1AdapterResult(ContractModel):
    """Complete Stage-1 coverage for one set of legacy compiler/database data."""

    schema_version: Literal[1] = 1
    compile_snapshot: LegacyNativeCompileSnapshot
    node_policy_snapshot: LegacyNodePolicySnapshot
    database_rows: LegacyExecutionRowsEnvelope
    template_bundle: LegacyAvailableContract[TemplateBundle] | LegacyUnavailableContract
    node_contract_registry: (
        LegacyAvailableContract[NodeContractRegistry] | LegacyUnavailableContract
    )
    host_capability_snapshot: (
        LegacyAvailableContract[HostCapabilitySnapshot] | LegacyUnavailableContract
    )
    resource_pool: LegacyAvailableContract[ResourcePool] | LegacyUnavailableContract
    published_value_refs: (
        LegacyAvailableContract[tuple[PublishedValueRef, ...]]
        | LegacyUnavailableContract
    )
    graph_audit_specs: (
        LegacyAvailableContract[tuple[GraphAuditSpec, ...]]
        | LegacyUnavailableContract
    )
    runtime_effect_contracts: (
        LegacyAvailableContract[tuple[RuntimeEffectContract, ...]]
        | LegacyUnavailableContract
    )
    runtime_requirements: (
        LegacyAvailableContract[tuple[RuntimeRequirements, ...]]
        | LegacyUnavailableContract
    )
    expected_output_specs: (
        LegacyAvailableContract[tuple[ExpectedOutputSpec, ...]]
        | LegacyUnavailableContract
    )
    observed_artifact_specs: (
        LegacyAvailableContract[tuple[ObservedArtifactSpec, ...]]
        | LegacyUnavailableContract
    )
    progress_specs: (
        LegacyAvailableContract[tuple[ProgressSpec, ...]]
        | LegacyUnavailableContract
    )
    preview_specs: (
        LegacyAvailableContract[tuple[PreviewSpec, ...]]
        | LegacyUnavailableContract
    )
    prepared_segment_units: (
        LegacyAvailableContract[tuple[PreparedSegmentUnit, ...]]
        | LegacyUnavailableContract
    )
    prepared_control_units: (
        LegacyAvailableContract[tuple[PreparedControlUnit, ...]]
        | LegacyUnavailableContract
    )
    compiled_execution_plan: (
        LegacyAvailableContract[CompiledExecutionPlan] | LegacyUnavailableContract
    )
    locked_submission_plan: (
        LegacyAvailableContract[LockedSubmissionPlan] | LegacyUnavailableContract
    )
    exact_prompt_snapshots: (
        LegacyAvailableContract[tuple[ExactPromptSnapshot, ...]]
        | LegacyUnavailableContract
    )
    prompt_ownership: (
        LegacyAvailableContract[tuple[PromptOwnership, ...]]
        | LegacyUnavailableContract
    )
    source_document_digest: LegacyAvailableContract[DocumentDigest]
    comfy_node_cache_identities: (
        LegacyAvailableContract[tuple[LegacyUnitCacheIdentity, ...]]
        | LegacyUnavailableContract
    )
    segment_execution_identities: (
        LegacyAvailableContract[tuple[SegmentExecutionIdentity, ...]]
        | LegacyUnavailableContract
    )
    historical_take_geometry_identities: (
        LegacyAvailableContract[tuple[HistoricalTakeGeometryIdentity, ...]]
        | LegacyUnavailableContract
    )
    ray_runtime_identities: (
        LegacyAvailableContract[tuple[RayRuntimeIdentity, ...]]
        | LegacyUnavailableContract
    )
    effective_execution_digest: (
        LegacyAvailableContract[DocumentDigest] | LegacyUnavailableContract
    )
    legacy_output_locators: (
        LegacyAvailableContract[tuple[LegacyOutputLocator, ...]]
        | LegacyUnavailableContract
    )
    validated_output_descriptors: (
        LegacyAvailableContract[tuple[LegacyValidatedTakeDescriptor, ...]]
        | LegacyUnavailableContract
    )


def freeze_native_workflow_unit(
    unit: NativeWorkflowUnit,
) -> LegacyNativeWorkflowUnitSnapshot:
    continuity = unit.continuity
    return LegacyNativeWorkflowUnitSnapshot(
        id=unit.id,
        family=unit.family,
        backend=unit.backend,
        segment_ids=unit.segment_ids,
        prompt=unit.prompt,
        output_nodes=unit.output_nodes,
        graph_audit_spec=unit.graph_audit_spec,
        graph_audit_traces=unit.graph_audit_traces,
        raylight_runtime_epoch=unit.raylight_runtime_epoch,
        raylight_runtime_namespace=unit.raylight_runtime_namespace,
        continuity=(
            LegacyNativeContinuitySnapshot(
                predecessor_segment_id=continuity.predecessor_segment_id,
                overlap_frames=continuity.overlap_frames,
                load_video_node_id=continuity.load_video_node_id,
                source=continuity.source,
                historical_take_id=continuity.historical_take_id,
                resolved=continuity.resolved,
                bound_file=continuity.bound_file,
            )
            if continuity is not None
            else None
        ),
    )


def restore_native_workflow_unit(
    snapshot: LegacyNativeWorkflowUnitSnapshot,
) -> NativeWorkflowUnit:
    continuity = snapshot.continuity
    return NativeWorkflowUnit(
        id=snapshot.id,
        family=snapshot.family,
        backend=snapshot.backend,
        segment_ids=snapshot.segment_ids,
        prompt=_thaw_json(snapshot.prompt),
        output_nodes=dict(snapshot.output_nodes),
        graph_audit_spec=snapshot.graph_audit_spec,
        graph_audit_traces=snapshot.graph_audit_traces,
        raylight_runtime_epoch=snapshot.raylight_runtime_epoch,
        raylight_runtime_namespace=snapshot.raylight_runtime_namespace,
        continuity=(
            NativeContinuityDependency(
                predecessor_segment_id=continuity.predecessor_segment_id,
                overlap_frames=continuity.overlap_frames,
                load_video_node_id=continuity.load_video_node_id,
                source=continuity.source,
                historical_take_id=continuity.historical_take_id,
                resolved=continuity.resolved,
                bound_file=continuity.bound_file,
            )
            if continuity is not None
            else None
        ),
    )


def native_workflow_unit_projection(
    snapshot: LegacyNativeWorkflowUnitSnapshot,
) -> dict[str, Any]:
    unit = restore_native_workflow_unit(snapshot)
    continuity = unit.continuity
    return {
        "id": unit.id,
        "family": unit.family,
        "backend": unit.backend,
        "segment_ids": list(unit.segment_ids),
        "prompt": unit.prompt,
        "output_nodes": dict(unit.output_nodes),
        "continuity": (
            {
                "predecessor_segment_id": continuity.predecessor_segment_id,
                "overlap_frames": continuity.overlap_frames,
                "load_video_node_id": continuity.load_video_node_id,
                "source": continuity.source,
                "historical_take_id": continuity.historical_take_id,
                "resolved": continuity.resolved,
            }
            if continuity is not None
            else None
        ),
    }


def freeze_native_compile_result(
    result: NativeCompileResult,
) -> LegacyNativeCompileSnapshot:
    return LegacyNativeCompileSnapshot(
        workflows=tuple(
            freeze_native_workflow_unit(unit) for unit in result.workflows
        ),
        manifest=result.manifest,
        plans=tuple(result.plans),
        families=result.families,
        node_policy=result.node_policy,
    )


def restore_native_compile_result(
    snapshot: LegacyNativeCompileSnapshot,
) -> NativeCompileResult:
    return NativeCompileResult(
        workflows=tuple(
            restore_native_workflow_unit(unit) for unit in snapshot.workflows
        ),
        manifest=_thaw_json(snapshot.manifest),
        plans=tuple(_thaw_json(plan) for plan in snapshot.plans),
        families=snapshot.families,
        node_policy=_thaw_json(snapshot.node_policy),
    )


def native_compile_result_projection(
    snapshot: LegacyNativeCompileSnapshot,
) -> dict[str, Any]:
    result = restore_native_compile_result(snapshot)
    return {
        "workflows": [
            native_workflow_unit_projection(unit) for unit in snapshot.workflows
        ],
        "manifest": result.manifest,
        "plans": list(result.plans),
        "families": list(result.families),
        "node_policy": result.node_policy,
    }


class LegacyTimelineCompileRead(ContractModel):
    """Frozen pre-Stage-6 HTTP projection; never used by the live v5 API."""

    execution_strategy: Literal["native_segment_graph_v1"] = (
        "native_segment_graph_v1"
    )
    model_families: tuple[Literal["fl2va", "ref2va"], ...]
    plans: tuple[JsonObject, ...]
    node_policy: JsonObject


def legacy_compile_public_read(
    snapshot: LegacyNativeCompileSnapshot,
) -> LegacyTimelineCompileRead:
    """Recreate the existing compile API projection, without new semantics."""

    restored = restore_native_compile_result(snapshot)
    return LegacyTimelineCompileRead(
        model_families=restored.families,
        plans=restored.plans,
        node_policy=restored.node_policy,
    )


LegacyPromptSnapshotKind = Literal[
    "absent",
    "native_prompt",
    "compile_manifest",
    "planned_manifest",
    "unknown",
]


class LegacyPromptSnapshotEvidence(ContractModel):
    kind: LegacyPromptSnapshotKind
    document: LegacyJsonObject | None = None

    @model_validator(mode="after")
    def validate_document(self) -> "LegacyPromptSnapshotEvidence":
        if self.kind == "absent" and self.document is not None:
            raise ValueError("an absent legacy prompt snapshot cannot carry a document")
        if self.kind not in {"absent", "unknown"} and self.document is None:
            raise ValueError("a recognized legacy prompt snapshot requires a document")
        return self


def classify_legacy_prompt_snapshot(value: Any) -> LegacyPromptSnapshotEvidence:
    if value is None:
        return LegacyPromptSnapshotEvidence(kind="absent", document=None)
    if not isinstance(value, Mapping):
        return LegacyPromptSnapshotEvidence(kind="unknown", document=None)
    if value and all(
        isinstance(node, Mapping)
        and isinstance(node.get("class_type"), str)
        and bool(node.get("class_type"))
        and isinstance(node.get("inputs"), Mapping)
        for node in value.values()
    ):
        kind: LegacyPromptSnapshotKind = "native_prompt"
    elif (
        value.get("graph_source") == "server"
        and isinstance(value.get("units"), (list, tuple))
        and isinstance(value.get("submission_order"), (list, tuple))
    ):
        kind = (
            "planned_manifest"
            if "runtime_epoch" in value or "runtime_transitions" in value
            else "compile_manifest"
        )
    else:
        kind = "unknown"
    return LegacyPromptSnapshotEvidence(kind=kind, document=value)


class LegacyJobRowEnvelope(ContractModel):
    schema_version: Literal[1] = 1
    row: LegacyJsonObject
    prompt_snapshot: LegacyPromptSnapshotEvidence
    unavailable_evidence: tuple[LegacyUnavailableEvidence, ...]


class LegacyJobChildRowEnvelope(ContractModel):
    schema_version: Literal[1] = 1
    row: LegacyJsonObject
    prompt_snapshot: LegacyPromptSnapshotEvidence
    output_locators: tuple[LegacyOutputLocator, ...]
    unavailable_evidence: tuple[LegacyUnavailableEvidence, ...]


class LegacySegmentTakeRowEnvelope(ContractModel):
    schema_version: Literal[1] = 1
    row: LegacyJsonObject
    validated_output_descriptor: OutputDescriptor | None
    legacy_has_audio: bool | None
    legacy_has_audio_provenance: Literal[
        "authored_audio_mode_inference"
    ] | None
    observed_artifact: None = None
    unavailable_evidence: tuple[LegacyUnavailableEvidence, ...]


class LegacyRayLedgerEnvelope(ContractModel):
    schema_version: Literal[1] = 1
    ledger: LegacyJsonObject | None
    unavailable_evidence: tuple[LegacyUnavailableEvidence, ...]


class LegacyExecutionRowsEnvelope(ContractModel):
    schema_version: Literal[1] = 1
    job: LegacyJobRowEnvelope
    children: tuple[LegacyJobChildRowEnvelope, ...]
    takes: tuple[LegacySegmentTakeRowEnvelope, ...]
    ray_ledger: LegacyRayLedgerEnvelope


def legacy_output_locators_from_child_row(
    child: Mapping[str, Any],
) -> tuple[tuple[LegacyOutputLocator, ...], tuple[LegacyUnavailableEvidence, ...]]:
    segment_ids = child.get("segment_ids")
    output_nodes = child.get("output_nodes")
    if not isinstance(segment_ids, (list, tuple)) or any(
        not isinstance(segment_id, str) or not segment_id for segment_id in segment_ids
    ):
        return (), (
            _unavailable(
                "output_locators",
                "ambiguous_legacy_shape",
                "segment_ids",
                "output_nodes",
                detail="segment_ids is not an array of non-empty strings",
            ),
        )
    if len(set(segment_ids)) != len(segment_ids):
        return (), (
            _unavailable(
                "output_locators",
                "ambiguous_legacy_shape",
                "segment_ids",
                detail="segment_ids contains duplicate identities",
            ),
        )
    if not segment_ids:
        return (), ()
    if not isinstance(output_nodes, Mapping):
        return (), (
            _unavailable(
                "output_locators",
                "ambiguous_legacy_shape",
                "output_nodes",
                detail="output_nodes is not an object",
            ),
        )

    candidate_nodes: dict[str, str] = {}
    gaps: list[LegacyUnavailableEvidence] = []
    declared = set(segment_ids)
    if any(key not in declared for key in output_nodes):
        gaps.append(
            _unavailable(
                "output_nodes",
                "ambiguous_legacy_shape",
                "segment_ids",
                "output_nodes",
                detail="output_nodes contains undeclared segment identities",
            )
        )
    for segment_id in segment_ids:
        node_id = output_nodes.get(segment_id)
        if not isinstance(node_id, str) or not node_id:
            gaps.append(
                _unavailable(
                    "output_locators",
                    "not_persisted",
                    "output_nodes",
                    detail="the declared segment has no non-empty output node",
                )
            )
            continue
        candidate_nodes[segment_id] = node_id

    duplicate_nodes = {
        node_id
        for node_id in candidate_nodes.values()
        if tuple(candidate_nodes.values()).count(node_id) > 1
    }
    locators: list[LegacyOutputLocator] = []
    for segment_id in segment_ids:
        node_id = candidate_nodes.get(segment_id)
        if node_id is None:
            continue
        if node_id in duplicate_nodes:
            gaps.append(
                _unavailable(
                    "output_locators",
                    "ambiguous_legacy_shape",
                    "output_nodes",
                    detail="one output node is assigned to multiple segments",
                )
            )
            continue
        try:
            locators.append(
                LegacyOutputLocator(segment_id=segment_id, node_id=node_id)
            )
        except ValidationError:
            gaps.append(
                _unavailable(
                    "output_locators",
                    "ambiguous_legacy_shape",
                    "segment_ids",
                    "output_nodes",
                    detail="the persisted identities do not satisfy the locator contract",
                )
            )
    return tuple(locators), tuple(gaps)


def freeze_legacy_job_row(row: Mapping[str, Any]) -> LegacyJobRowEnvelope:
    prompt = classify_legacy_prompt_snapshot(row.get("prompt_snapshot"))
    unavailable = (
        _unavailable(
            "compiled_execution_plan",
            "not_persisted",
            "prompt_snapshot",
            detail="the parent snapshot is overloaded and is not a versioned plan",
        ),
        _unavailable("locked_submission_plan", "not_persisted"),
        _unavailable("endpoint_identity", "not_persisted", "settings_snapshot"),
        _unavailable(
            "prompt_ownership.requested_prompt_id",
            "legacy_value_overwritten",
            "prompt_id",
        ),
        _unavailable(
            "prompt_ownership.actual_prompt_id",
            "legacy_value_overwritten",
            "prompt_id",
        ),
        _unavailable("prompt_ownership.ownership_revision", "not_persisted"),
        _unavailable("prompt_ownership.cleanup_certificate", "not_persisted"),
    )
    return LegacyJobRowEnvelope(
        row=row,
        prompt_snapshot=prompt,
        unavailable_evidence=unavailable,
    )


def freeze_legacy_job_child_row(
    row: Mapping[str, Any],
) -> LegacyJobChildRowEnvelope:
    prompt = classify_legacy_prompt_snapshot(row.get("prompt_snapshot"))
    locators, locator_gaps = legacy_output_locators_from_child_row(row)
    unavailable = [
        _unavailable("exact_prompt_snapshot.unit_id", "not_persisted", "id"),
        _unavailable("exact_prompt_snapshot.template_id", "not_persisted"),
        _unavailable("exact_prompt_snapshot.template_revision", "not_persisted"),
        _unavailable("exact_prompt_snapshot.endpoint_identity", "not_persisted"),
        _unavailable("exact_prompt_snapshot.graph_audit_spec", "not_persisted"),
        _unavailable("exact_prompt_snapshot.expected_output_spec", "not_persisted"),
        _unavailable(
            "exact_prompt_snapshot.progress_spec",
            "implicit_runtime_fallback",
            "prompt_snapshot",
        ),
        _unavailable(
            "exact_prompt_snapshot.preview_spec",
            "implicit_runtime_fallback",
            "prompt_snapshot",
        ),
        _unavailable("exact_prompt_snapshot.effective_execution_digest", "not_persisted"),
        _unavailable(
            "prompt_ownership.requested_prompt_id",
            "legacy_value_overwritten",
            "prompt_id",
        ),
        _unavailable(
            "prompt_ownership.actual_prompt_id",
            "legacy_value_overwritten",
            "prompt_id",
        ),
        _unavailable("prompt_ownership.ownership_revision", "not_persisted"),
        _unavailable("prompt_ownership.cleanup_certificate", "not_persisted"),
    ]
    if prompt.kind != "native_prompt":
        unavailable.append(
            _unavailable(
                "exact_prompt_snapshot.exact_prompt",
                "unversioned_snapshot",
                "prompt_snapshot",
            )
        )
    else:
        unavailable.append(
            _unavailable(
                "exact_prompt_snapshot.submission_binding",
                "unversioned_snapshot",
                "prompt_snapshot",
                "prompt_id",
                detail="the row does not prove whether this is base or submitted prompt",
            )
        )
    unavailable.extend(locator_gaps)
    return LegacyJobChildRowEnvelope(
        row=row,
        prompt_snapshot=prompt,
        output_locators=locators,
        unavailable_evidence=tuple(unavailable),
    )


def freeze_legacy_segment_take_row(
    row: Mapping[str, Any],
) -> LegacySegmentTakeRowEnvelope:
    descriptor = row.get("output_descriptor", row.get("output"))
    validated_output_descriptor: OutputDescriptor | None = None
    if isinstance(descriptor, Mapping):
        try:
            validated_output_descriptor = OutputDescriptor.model_validate(descriptor)
        except ValidationError:
            pass
    raw_has_audio = row.get("has_audio")
    if isinstance(raw_has_audio, bool):
        legacy_has_audio = raw_has_audio
    elif type(raw_has_audio) is int and raw_has_audio in {0, 1}:
        legacy_has_audio = bool(raw_has_audio)
    else:
        legacy_has_audio = None
    unavailable = [
        _unavailable(
            f"observed_artifact.{field}",
            "no_media_probe",
            "output_descriptor",
        )
        for field in (
            "width",
            "height",
            "fps",
            "frame_count",
            "duration_seconds",
            "has_audio",
            "media_probe_version",
            "content_hash",
        )
    ]
    if validated_output_descriptor is None:
        unavailable.append(
            _unavailable(
                "validated_output_descriptor",
                "ambiguous_legacy_shape",
                "output_descriptor",
                "output",
                detail="the legacy descriptor is absent or fails current path safety",
            )
        )
    if legacy_has_audio is None:
        unavailable.append(
            _unavailable(
                "legacy_has_audio",
                "not_persisted",
                "has_audio",
            )
        )
    return LegacySegmentTakeRowEnvelope(
        row=row,
        validated_output_descriptor=validated_output_descriptor,
        legacy_has_audio=legacy_has_audio,
        legacy_has_audio_provenance=(
            "authored_audio_mode_inference"
            if legacy_has_audio is not None
            else None
        ),
        observed_artifact=None,
        unavailable_evidence=tuple(unavailable),
    )


def freeze_legacy_ray_ledger(
    ledger: Mapping[str, Any] | None,
) -> LegacyRayLedgerEnvelope:
    unavailable = [
        _unavailable("locked_submission_plan.ray_ledger_before", "not_persisted"),
        _unavailable(
            "locked_submission_plan.ray_ledger_after_intent",
            "not_persisted",
        ),
        _unavailable(
            "locked_submission_plan.source_compiled_plan_digest",
            "not_persisted",
        ),
        _unavailable("locked_submission_plan.endpoint_identity", "not_persisted"),
    ]
    if ledger is None:
        unavailable.append(_unavailable("ray_runtime_ledger", "not_persisted"))
    return LegacyRayLedgerEnvelope(
        ledger=ledger,
        unavailable_evidence=tuple(unavailable),
    )


def freeze_legacy_execution_rows(
    *,
    job: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    takes: Sequence[Mapping[str, Any]],
    ray_ledger: Mapping[str, Any] | None,
) -> LegacyExecutionRowsEnvelope:
    return LegacyExecutionRowsEnvelope(
        job=freeze_legacy_job_row(job),
        children=tuple(freeze_legacy_job_child_row(row) for row in children),
        takes=tuple(freeze_legacy_segment_take_row(row) for row in takes),
        ray_ledger=freeze_legacy_ray_ledger(ray_ledger),
    )


def freeze_legacy_node_policy(
    node_policy: Mapping[str, Any],
) -> LegacyNodePolicySnapshot:
    allowed_nodes = node_policy.get("allowed_nodes")
    custom_nodes = node_policy.get("custom_nodes")
    provenance = node_policy.get("provenance")
    graph_source = node_policy.get("graph_source")
    accepts_client_workflow = node_policy.get("accepts_client_workflow")
    if (
        not isinstance(graph_source, str)
        or not isinstance(accepts_client_workflow, bool)
        or not isinstance(allowed_nodes, (list, tuple))
        or any(not isinstance(item, str) for item in allowed_nodes)
        or not isinstance(custom_nodes, (list, tuple))
        or any(not isinstance(item, str) for item in custom_nodes)
        or not isinstance(provenance, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in provenance.items()
        )
    ):
        raise ValueError("legacy node_policy has an invalid compatibility shape")
    return LegacyNodePolicySnapshot(
        graph_source=graph_source,
        accepts_client_workflow=accepts_client_workflow,
        allowed_nodes=tuple(allowed_nodes),
        custom_nodes=tuple(custom_nodes),
        provenance=dict(provenance),
    )


def _unavailable_contract(
    field_path: str,
    reason: LegacyUnavailableReason,
    *source_fields: str,
    detail: str,
) -> LegacyUnavailableContract:
    return LegacyUnavailableContract(
        evidence=(
            _unavailable(
                field_path,
                reason,
                *source_fields,
                detail=detail,
            ),
        )
    )


def adapt_legacy_stage1_sources(
    compile_result: NativeCompileResult,
    *,
    job: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    takes: Sequence[Mapping[str, Any]],
    ray_ledger: Mapping[str, Any] | None,
) -> LegacyStage1AdapterResult:
    """Map every Stage-1 concept to typed evidence or explicit unavailability."""

    compile_snapshot = freeze_native_compile_result(compile_result)
    node_policy_snapshot = freeze_legacy_node_policy(compile_result.node_policy)
    database_rows = freeze_legacy_execution_rows(
        job=job,
        children=children,
        takes=takes,
        ray_ledger=ray_ledger,
    )
    source_document_digest = sha256_document_digest(
        {
            "compile_snapshot": compile_snapshot.model_dump(mode="json"),
            "database_rows": database_rows.model_dump(mode="json"),
        }
    )

    cache_identities: list[LegacyUnitCacheIdentity] = []
    cache_limitations: list[LegacyUnavailableEvidence] = []
    for unit in compile_result.workflows:
        try:
            cache_identities.append(
                LegacyUnitCacheIdentity(
                    unit_id=unit.id,
                    identity=comfy_node_cache_identity(
                        _coerce_json_containers(unit.prompt)
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            cache_limitations.append(
                _unavailable(
                    "comfy_node_cache_identities",
                    "ambiguous_legacy_shape",
                    "NativeWorkflowUnit.prompt",
                    detail=(
                        "one prompt cannot form a node cache identity: "
                        f"{str(exc)[:900]}"
                    ),
                )
            )
    cache_slot: (
        LegacyAvailableContract[tuple[LegacyUnitCacheIdentity, ...]]
        | LegacyUnavailableContract
    )
    if cache_identities:
        cache_slot = LegacyAvailableContract(
            value=tuple(cache_identities),
            provenance="derived_from_frozen_legacy",
            limitations=tuple(cache_limitations),
        )
    else:
        cache_slot = LegacyUnavailableContract(
            evidence=tuple(cache_limitations)
            or (
                _unavailable(
                    "comfy_node_cache_identities",
                    "not_persisted",
                    "NativeWorkflowUnit.prompt",
                ),
            )
        )

    locators = tuple(
        locator
        for child in database_rows.children
        for locator in child.output_locators
    )
    locator_limitations = tuple(
        gap
        for child in database_rows.children
        for gap in child.unavailable_evidence
        if gap.field_path.startswith("output_locator")
        or gap.field_path == "output_nodes"
    )
    locator_slot: (
        LegacyAvailableContract[tuple[LegacyOutputLocator, ...]]
        | LegacyUnavailableContract
    )
    if locators:
        locator_slot = LegacyAvailableContract(
            value=locators,
            provenance="legacy_exact",
            limitations=locator_limitations,
        )
    else:
        locator_slot = LegacyUnavailableContract(
            evidence=locator_limitations
            or (
                _unavailable(
                    "legacy_output_locators",
                    "not_persisted",
                    "job_children.output_nodes",
                ),
            )
        )

    descriptors: list[LegacyValidatedTakeDescriptor] = []
    descriptor_limitations: list[LegacyUnavailableEvidence] = []
    for take in database_rows.takes:
        if take.validated_output_descriptor is not None:
            raw_take = _thaw_json(take.row)
            raw_take_id = raw_take.get("id")
            descriptors.append(
                LegacyValidatedTakeDescriptor(
                    source_take_id=(raw_take_id if isinstance(raw_take_id, str) else None),
                    descriptor=take.validated_output_descriptor,
                )
            )
        descriptor_limitations.extend(
            gap
            for gap in take.unavailable_evidence
            if gap.field_path == "validated_output_descriptor"
        )
    descriptor_slot: (
        LegacyAvailableContract[tuple[LegacyValidatedTakeDescriptor, ...]]
        | LegacyUnavailableContract
    )
    if descriptors:
        descriptor_slot = LegacyAvailableContract(
            value=tuple(descriptors),
            provenance="legacy_exact",
            limitations=tuple(descriptor_limitations),
        )
    else:
        descriptor_slot = LegacyUnavailableContract(
            evidence=tuple(descriptor_limitations)
            or (
                _unavailable(
                    "validated_output_descriptors",
                    "not_persisted",
                    "segment_takes.output_descriptor",
                ),
            )
        )

    return LegacyStage1AdapterResult(
        compile_snapshot=compile_snapshot,
        node_policy_snapshot=node_policy_snapshot,
        database_rows=database_rows,
        template_bundle=_unavailable_contract(
            "template_bundle",
            "not_persisted",
            "NativeCompileResult.manifest",
            detail="the monolithic compiler did not persist a TemplateBundle",
        ),
        node_contract_registry=_unavailable_contract(
            "node_contract_registry",
            "not_persisted",
            "NativeCompileResult.node_policy",
            detail="node_policy lacks ObjectInfo, ports, effects and fingerprints",
        ),
        host_capability_snapshot=_unavailable_contract(
            "host_capability_snapshot",
            "not_persisted",
            "NativeCompileResult.node_policy",
            detail="allowed node names are not a versioned host capability snapshot",
        ),
        resource_pool=_unavailable_contract(
            "resource_pool",
            "not_persisted",
            "NativeWorkflowUnit.prompt",
            detail="the legacy graph did not retain typed resource transactions",
        ),
        published_value_refs=_unavailable_contract(
            "published_value_refs",
            "not_persisted",
            "NativeWorkflowUnit.prompt",
            detail="raw Comfy edges do not prove typed published values",
        ),
        graph_audit_specs=_unavailable_contract(
            "graph_audit_specs",
            "not_persisted",
            "NativeCompileResult.node_policy",
            detail="legacy policy has no per-node contract evidence",
        ),
        runtime_effect_contracts=_unavailable_contract(
            "runtime_effect_contracts",
            "not_persisted",
            "NativeCompileResult.node_policy",
            detail="legacy provenance labels do not describe runtime effects",
        ),
        runtime_requirements=_unavailable_contract(
            "runtime_requirements",
            "not_persisted",
            "NativeWorkflowUnit.backend",
            "raylight_runtime_state",
            detail="backend and Ray ledger fragments do not prove full requirements",
        ),
        expected_output_specs=_unavailable_contract(
            "expected_output_specs",
            "not_persisted",
            "NativeCompileResult.plans",
            "job_children.output_nodes",
            detail="legacy locators do not prove a complete expected output spec",
        ),
        observed_artifact_specs=_unavailable_contract(
            "observed_artifact_specs",
            "no_media_probe",
            "segment_takes",
            detail="legacy takes contain no trusted media probe",
        ),
        progress_specs=_unavailable_contract(
            "progress_specs",
            "implicit_runtime_fallback",
            "NativeWorkflowUnit.prompt",
            detail="legacy progress is class-type fallback rather than a spec",
        ),
        preview_specs=_unavailable_contract(
            "preview_specs",
            "implicit_runtime_fallback",
            "NativeWorkflowUnit.prompt",
            detail="legacy preview ownership is sampler-count fallback",
        ),
        prepared_segment_units=_unavailable_contract(
            "prepared_segment_units",
            "not_persisted",
            "NativeWorkflowUnit",
            detail="strict prepared units require unavailable audit and user specs",
        ),
        prepared_control_units=_unavailable_contract(
            "prepared_control_units",
            "not_persisted",
            "job_children",
            detail="legacy control rows do not preserve template and audit evidence",
        ),
        compiled_execution_plan=_unavailable_contract(
            "compiled_execution_plan",
            "not_persisted",
            "NativeCompileResult",
            detail="the legacy manifest cannot satisfy the strict plan contract",
        ),
        locked_submission_plan=_unavailable_contract(
            "locked_submission_plan",
            "not_persisted",
            "jobs",
            "raylight_runtime_state",
            detail="legacy rows do not preserve one atomic locked plan",
        ),
        exact_prompt_snapshots=_unavailable_contract(
            "exact_prompt_snapshots",
            "unversioned_snapshot",
            "job_children.prompt_snapshot",
            detail="base and submitted prompts are not durably distinguished",
        ),
        prompt_ownership=_unavailable_contract(
            "prompt_ownership",
            "legacy_value_overwritten",
            "job_children.prompt_id",
            detail="requested and actual IDs share one overwriteable column",
        ),
        source_document_digest=LegacyAvailableContract(
            value=source_document_digest,
            provenance="derived_from_frozen_legacy",
        ),
        comfy_node_cache_identities=cache_slot,
        segment_execution_identities=_unavailable_contract(
            "segment_execution_identities",
            "not_persisted",
            "jobs.config_snapshot",
            "jobs.settings_snapshot",
            detail="legacy rows lack resolved feature and implementation identities",
        ),
        historical_take_geometry_identities=_unavailable_contract(
            "historical_take_geometry_identities",
            "no_media_probe",
            "segment_takes",
            detail="legacy take geometry was not observed and versioned",
        ),
        ray_runtime_identities=_unavailable_contract(
            "ray_runtime_identities",
            "not_persisted",
            "raylight_runtime_state",
            detail="the v2 ledger descriptor is not the Stage-1 Ray identity model",
        ),
        effective_execution_digest=_unavailable_contract(
            "effective_execution_digest",
            "not_persisted",
            "NativeCompileResult",
            detail="a source digest must not be presented as an effective digest",
        ),
        legacy_output_locators=locator_slot,
        validated_output_descriptors=descriptor_slot,
    )


LegacyRowEnvelope: TypeAlias = (
    LegacyJobRowEnvelope
    | LegacyJobChildRowEnvelope
    | LegacySegmentTakeRowEnvelope
)


def legacy_row_projection(envelope: LegacyRowEnvelope) -> dict[str, Any]:
    """Return the exact input row shape; evidence never mutates public data."""

    return _thaw_json(envelope.row)


def legacy_ray_ledger_projection(
    envelope: LegacyRayLedgerEnvelope,
) -> dict[str, Any] | None:
    return _thaw_json(envelope.ledger) if envelope.ledger is not None else None


class LegacyPublicJobEnvelope(ContractModel):
    """Deeply frozen input and output of the existing public job projector."""

    schema_version: Literal[1] = 1
    source_job: LegacyJsonObject
    source_children: tuple[LegacyJsonObject, ...]
    source_takes: tuple[LegacyJsonObject, ...]
    public_job: LegacyJsonObject


LegacyPublicJobProjector: TypeAlias = Callable[
    [dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]],
    JobRead | Mapping[str, Any],
]


def freeze_legacy_public_job_from_rows(
    *,
    job: Mapping[str, Any],
    children: Sequence[Mapping[str, Any]],
    takes: Sequence[Mapping[str, Any]],
    projector: LegacyPublicJobProjector,
) -> LegacyPublicJobEnvelope:
    """Run the current projector over isolated row copies, then freeze JSON.

    The adapter does not reproduce any URL, filtering, naming, output-count or
    segment-result logic.  Validation as ``JobRead`` is intentionally deferred
    until restore so this envelope remains a faithful oracle capture.
    """

    isolated_job = _thaw_json(_coerce_json_containers(job))
    isolated_children = tuple(
        _thaw_json(_coerce_json_containers(child)) for child in children
    )
    isolated_takes = tuple(
        _thaw_json(_coerce_json_containers(take)) for take in takes
    )
    projected = projector(isolated_job, isolated_children, isolated_takes)
    public_document = (
        projected.model_dump(mode="json")
        if isinstance(projected, JobRead)
        else projected
    )
    if not isinstance(public_document, Mapping):
        raise TypeError("legacy public job projector must return a JSON object")
    return LegacyPublicJobEnvelope(
        source_job=job,
        source_children=tuple(children),
        source_takes=tuple(takes),
        public_job=public_document,
    )


def restore_legacy_public_job(envelope: LegacyPublicJobEnvelope) -> JobRead:
    return JobRead.model_validate(_thaw_json(envelope.public_job))


LegacyStage1AdapterResult.model_rebuild(_types_namespace=globals())


__all__ = [
    "LegacyExecutionRowsEnvelope",
    "LegacyJobChildRowEnvelope",
    "LegacyJobRowEnvelope",
    "LegacyNativeCompileSnapshot",
    "LegacyNativeContinuitySnapshot",
    "LegacyNativeWorkflowUnitSnapshot",
    "LegacyPromptSnapshotEvidence",
    "LegacyPublicJobEnvelope",
    "LegacyRayLedgerEnvelope",
    "LegacySegmentTakeRowEnvelope",
    "LegacyUnavailableEvidence",
    "LegacyAvailableContract",
    "LegacyNodePolicySnapshot",
    "LegacyStage1AdapterResult",
    "LegacyUnavailableContract",
    "LegacyUnitCacheIdentity",
    "LegacyValidatedTakeDescriptor",
    "adapt_legacy_stage1_sources",
    "classify_legacy_prompt_snapshot",
    "freeze_legacy_execution_rows",
    "freeze_legacy_job_child_row",
    "freeze_legacy_job_row",
    "freeze_legacy_public_job_from_rows",
    "freeze_legacy_ray_ledger",
    "freeze_legacy_segment_take_row",
    "freeze_native_compile_result",
    "freeze_native_workflow_unit",
    "freeze_legacy_node_policy",
    "legacy_compile_public_read",
    "legacy_output_locators_from_child_row",
    "legacy_ray_ledger_projection",
    "legacy_row_projection",
    "native_compile_result_projection",
    "native_workflow_unit_projection",
    "restore_legacy_public_job",
    "restore_native_compile_result",
    "restore_native_workflow_unit",
]
