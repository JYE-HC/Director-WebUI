"""Immutable execution contracts for the extensible workflow architecture.

This module deliberately contains data models and pure functions only.  It does
not compile, submit, execute, persist, or recover a prompt.  Production paths
continue to use the legacy implementation until a later migration phase binds
these contracts explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from typing import Annotated, Any, Final, Literal, TypeAlias

from pydantic import (
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
)

from .canonical import (
    MAX_SAFE_JSON_INTEGER,
    canonical_json,
    canonical_json_bytes,
    canonical_values_equal,
    fnv1a32_utf16,
    javascript_json_stringify,
    utf16_sort_key,
)
from .compile_report import CompiledExecutionReportV2, CompiledExecutionReportV3
from .contracts import (
    Backend,
    ContractModel,
    FeatureEmission,
    GraphAuditSpec,
    GraphNodeContractEvidence,
    JsonObject,
    ModelFamily,
    NodeContractRegistry,
    ResolvedFeatureImplementation,
    ResolvedImplementationIdentity,
)


DocumentDigestAlgorithm: TypeAlias = Literal[
    "fnv1a32-json-stringify-v1",
    "sha256-canonical-json-v1",
]
AudioMode: TypeAlias = Literal["generated", "source", "none"]
PromptOwnershipState: TypeAlias = Literal[
    "prepared",
    "submitting",
    "owned_requested_id",
    "owned_actual_id",
    "cancel_pending",
    "cleanup_confirmed",
    "terminal_confirmed",
    "unconfirmed",
]

_UNSET: Final = object()
_FNV_PATTERN: Final = re.compile(r"^fnv1a-[0-9a-f]{8}$")
_SHA256_PATTERN: Final = re.compile(r"^sha256-[0-9a-f]{64}$")
_FEATURE_PATTERN: Final = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.+_-]*$"
)


NonEmptyString: TypeAlias = Annotated[str, Field(min_length=1, max_length=1024)]
Identifier: TypeAlias = Annotated[str, Field(min_length=1, max_length=256)]
# ComfyUI owns the receipt value and older/custom hosts are not required to
# use Director's shorter identifiers.  Keep the opaque transport value
# bounded, but make every durable ownership/release contract accept the same
# domain as ComfyClient so a mismatched receipt can always be recovered.
PromptIdentifier: TypeAlias = Annotated[str, Field(min_length=1, max_length=512)]
NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0, le=MAX_SAFE_JSON_INTEGER)]
PositiveInt: TypeAlias = Annotated[int, Field(ge=1, le=MAX_SAFE_JSON_INTEGER)]
NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0, allow_inf_nan=False)]
PositiveFloat: TypeAlias = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class FrozenContractModel(ContractModel):
    """Strict, immutable base for persisted or transmitted contracts."""


class DocumentDigest(FrozenContractModel):
    algorithm: DocumentDigestAlgorithm
    value: NonEmptyString

    @model_validator(mode="after")
    def validate_algorithm_value_pair(self) -> "DocumentDigest":
        pattern = (
            _FNV_PATTERN
            if self.algorithm == "fnv1a32-json-stringify-v1"
            else _SHA256_PATTERN
        )
        if pattern.fullmatch(self.value) is None:
            raise ValueError(
                f"digest value does not match algorithm {self.algorithm}"
            )
        return self


class CompiledPlanDigest(DocumentDigest):
    """A SHA-256 digest branded as the exact serialized compiled plan."""

    brand: Literal["compiled-execution-plan-v1"] = "compiled-execution-plan-v1"
    algorithm: Literal["sha256-canonical-json-v1"] = "sha256-canonical-json-v1"


def sha256_document_digest(value: Any) -> DocumentDigest:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return DocumentDigest(
        algorithm="sha256-canonical-json-v1",
        value=f"sha256-{digest}",
    )


def legacy_fnv1a32_document_digest(value: Any) -> DocumentDigest:
    serialized = javascript_json_stringify(value)
    hash_value = fnv1a32_utf16(serialized)
    return DocumentDigest(
        algorithm="fnv1a32-json-stringify-v1",
        value=f"fnv1a-{hash_value:08x}",
    )


def digest_document(
    value: Any,
    *,
    algorithm: DocumentDigestAlgorithm = "sha256-canonical-json-v1",
) -> DocumentDigest:
    if algorithm == "sha256-canonical-json-v1":
        return sha256_document_digest(value)
    if algorithm == "fnv1a32-json-stringify-v1":
        return legacy_fnv1a32_document_digest(value)
    raise ValueError(f"unsupported document digest algorithm {algorithm!r}")


class OutputDescriptor(FrozenContractModel):
    filename: Annotated[str, Field(min_length=1, max_length=512)]
    subfolder: Annotated[str, Field(max_length=512)] = ""
    type: Literal["output"] = "output"

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if (
            value != value.strip()
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "[" in value
            or "]" in value
            or any(
                ord(character) < 32
                or ord(character) == 127
                or 0xD800 <= ord(character) <= 0xDFFF
                for character in value
            )
        ):
            raise ValueError("output filename is unsafe")
        return value

    @field_validator("subfolder")
    @classmethod
    def validate_subfolder(cls, value: str) -> str:
        if (
            value != value.strip()
            or "\\" in value
            or "[" in value
            or "]" in value
            or any(
                ord(character) < 32
                or ord(character) == 127
                or 0xD800 <= ord(character) <= 0xDFFF
                for character in value
            )
        ):
            raise ValueError("output subfolder is unsafe")
        if value:
            if value in {".", ".."} or any(
                part in {"", ".", ".."} for part in value.split("/")
            ):
                raise ValueError("output subfolder is unsafe")
            folder = PurePosixPath(value)
            if folder.is_absolute() or any(
                part in {"", ".", ".."} for part in folder.parts
            ):
                raise ValueError("output subfolder is unsafe")
        return value


class ContinuityLateBindingEvidence(FrozenContractModel):
    """Typed authority for one predecessor take materialized under the lock."""

    source_kind: Literal["continuity"] = "continuity"
    input_pointer: NonEmptyString
    predecessor_segment_id: Identifier
    dependency_source: Literal["same_run", "historical_take"]
    historical_take_id: Identifier | None = None
    output: OutputDescriptor

    @model_validator(mode="after")
    def validate_dependency_source(self) -> "ContinuityLateBindingEvidence":
        if not self.input_pointer.startswith("/"):
            raise ValueError("continuity evidence pointer must be absolute")
        if self.dependency_source == "historical_take":
            if self.historical_take_id is None:
                raise ValueError(
                    "historical continuity evidence requires a take id"
                )
        elif self.historical_take_id is not None:
            raise ValueError(
                "same-run continuity evidence cannot carry a historical take id"
            )
        return self

    @property
    def bound_value(self) -> str:
        relative = (
            PurePosixPath(self.output.filename)
            if not self.output.subfolder
            else PurePosixPath(self.output.subfolder) / self.output.filename
        )
        return f"{relative.as_posix()} [output]"


class RuntimeEpochLateBindingEvidence(FrozenContractModel):
    """Typed authority for a Ray namespace derived from the locked ledger epoch."""

    source_kind: Literal["runtime_epoch"] = "runtime_epoch"
    input_pointer: NonEmptyString
    epoch: PositiveInt

    @field_validator("input_pointer")
    @classmethod
    def validate_pointer(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("runtime epoch evidence pointer must be absolute")
        return value


LateBindingEvidence: TypeAlias = Annotated[
    ContinuityLateBindingEvidence | RuntimeEpochLateBindingEvidence,
    Field(discriminator="source_kind"),
]


class ExpectedOutputGeometry(FrozenContractModel):
    width: PositiveInt
    height: PositiveInt
    fps: PositiveFloat
    visible_frame_count: PositiveInt
    expected_audio_mode: AudioMode


class ExpectedOutputSpec(ExpectedOutputGeometry):
    segment_id: Identifier
    node_id: Identifier
    kind: Literal["video"] = "video"
    role: Literal["take"] = "take"

    @property
    def geometry(self) -> ExpectedOutputGeometry:
        return ExpectedOutputGeometry(
            width=self.width,
            height=self.height,
            fps=self.fps,
            visible_frame_count=self.visible_frame_count,
            expected_audio_mode=self.expected_audio_mode,
        )


class ObservedArtifactSpec(FrozenContractModel):
    segment_id: Identifier
    child_id: Identifier
    output_descriptor: OutputDescriptor
    width: PositiveInt
    height: PositiveInt
    fps: PositiveFloat
    frame_count: PositiveInt
    duration_seconds: PositiveFloat
    has_audio: bool
    media_probe_version: NonEmptyString
    content_hash: Annotated[str, Field(min_length=1, max_length=256)] | None = None


class AssemblySourceArtifactRef(FrozenContractModel):
    """Digest-pinned observed take consumed by one timeline assembly."""

    segment_id: Identifier
    child_id: Identifier
    observed_artifact_digest: DocumentDigest

    @model_validator(mode="after")
    def validate_digest(self) -> "AssemblySourceArtifactRef":
        if self.observed_artifact_digest.algorithm != "sha256-canonical-json-v1":
            raise ValueError(
                "assembly source artifact digest must use canonical SHA-256"
            )
        return self


class ObservedAssemblyArtifactSpec(FrozenContractModel):
    """Immutable authority for the parent artifact assembled from segment takes."""

    schema_version: Literal[1] = 1
    job_id: Identifier
    kind: Literal["video"] = "video"
    role: Literal["timeline_assembly"] = "timeline_assembly"
    source_compiled_plan_digest: CompiledPlanDigest
    source_artifacts: Annotated[
        tuple[AssemblySourceArtifactRef, ...], Field(min_length=2, max_length=128)
    ]
    output_descriptor: OutputDescriptor
    width: PositiveInt
    height: PositiveInt
    fps: PositiveFloat
    frame_count: PositiveInt
    duration_seconds: PositiveFloat
    has_audio: bool
    assembly_method_version: Literal["directordeck_timeline_concat_v1"] = (
        "directordeck_timeline_concat_v1"
    )
    media_probe_version: NonEmptyString
    content_hash: Annotated[str, Field(min_length=1, max_length=256)] | None = None

    @model_validator(mode="after")
    def validate_source_identities(self) -> "ObservedAssemblyArtifactSpec":
        segment_ids = [source.segment_id for source in self.source_artifacts]
        child_ids = [source.child_id for source in self.source_artifacts]
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("assembly source segment ids must be unique")
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("assembly source child ids must be unique")
        return self


class LegacyOutputLocator(FrozenContractModel):
    segment_id: Identifier
    node_id: Identifier


class ProgressPhase(FrozenContractModel):
    id: Identifier
    label: NonEmptyString
    node_id: Identifier
    kind: Literal["stage", "milestone", "fractional"]
    weight: NonNegativeFloat

    @model_validator(mode="after")
    def validate_weight(self) -> "ProgressPhase":
        if self.kind == "stage":
            if self.weight != 0:
                raise ValueError("stage-only progress phases must have zero weight")
        elif self.weight <= 0:
            raise ValueError("weighted progress phases must have positive weight")
        return self


class ProgressSpec(FrozenContractModel):
    version: PositiveInt
    phases: Annotated[tuple[ProgressPhase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_phases(self) -> "ProgressSpec":
        phase_ids = [phase.id for phase in self.phases]
        node_ids = [phase.node_id for phase in self.phases]
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("progress phase ids must be unique")
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("progress node ids must be unique")
        if not math.isclose(
            math.fsum(phase.weight for phase in self.phases),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("progress phase weights must sum to 1")
        return self


class PreviewSource(FrozenContractModel):
    node_id: Identifier
    phase_id: Identifier
    publish: bool
    priority: int
    supersedes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def validate_supersedes(self) -> "PreviewSource":
        if len(set(self.supersedes)) != len(self.supersedes):
            raise ValueError("preview supersedes entries must be unique")
        if self.node_id in self.supersedes:
            raise ValueError("preview source cannot supersede itself")
        return self


class PreviewSpec(FrozenContractModel):
    version: PositiveInt
    sources: tuple[PreviewSource, ...]

    @model_validator(mode="after")
    def validate_sources(self) -> "PreviewSpec":
        source_ids = [source.node_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("preview source node ids must be unique")
        known = set(source_ids)
        for source in self.sources:
            unknown = set(source.supersedes) - known
            if unknown:
                raise ValueError(
                    f"preview source {source.node_id!r} supersedes unknown nodes"
                )

        edges = {source.node_id: source.supersedes for source in self.sources}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("preview supersedes relation must be acyclic")
            if node_id in visited:
                return
            visiting.add(node_id)
            for predecessor in edges[node_id]:
                visit(predecessor)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in source_ids:
            visit(node_id)
        return self


def derive_feature_execution_specs(
    scoped_emissions: Sequence[tuple[Sequence[str], FeatureEmission]],
) -> tuple[ProgressSpec | None, PreviewSpec | None]:
    """Derive the complete execution UX authority from feature emissions.

    ``scoped_emissions`` must retain template order and, within each emission,
    hint order.  Hints are deliberately scoped to the nodes emitted by the
    feature which declared them: an interpreter cannot claim progress or
    preview ownership for another feature's private graph.

    The first hint switches the whole unit to explicit authority.  Progress is
    mandatory for preview ownership; progress without preview sources derives
    an explicit empty ``PreviewSpec``.  Callers must never fill either case
    with the legacy class-type scan.
    """

    progress_phases: list[ProgressPhase] = []
    preview_sources: list[PreviewSource] = []
    known_scope_nodes: set[str] = set()

    for scope_node_ids, emission in scoped_emissions:
        if isinstance(scope_node_ids, (str, bytes)):
            raise ValueError("feature execution scope must be a node-id sequence")
        local_nodes = tuple(scope_node_ids)
        if any(not isinstance(node_id, str) or not node_id for node_id in local_nodes):
            raise ValueError(
                "feature execution scope node ids must be non-empty strings"
            )
        if len(set(local_nodes)) != len(local_nodes):
            raise ValueError("feature execution scope node ids must be unique")
        overlap = known_scope_nodes.intersection(local_nodes)
        if overlap:
            raise ValueError("feature execution scopes must not share node ids")
        known_scope_nodes.update(local_nodes)
        local_node_set = set(local_nodes)

        for raw_hint in emission.progress_hints:
            if not isinstance(raw_hint, Mapping):
                raise ValueError("progress hint must be an object")
            try:
                phase = ProgressPhase.model_validate(dict(raw_hint))
            except ValueError as exc:
                raise ValueError("progress hint is invalid") from exc
            if phase.node_id not in local_node_set:
                raise ValueError(
                    "progress hint node must belong to its feature emission scope"
                )
            progress_phases.append(phase)

        for raw_hint in emission.preview_hints:
            if not isinstance(raw_hint, Mapping):
                raise ValueError("preview hint must be an object")
            try:
                source = PreviewSource.model_validate(dict(raw_hint))
            except ValueError as exc:
                raise ValueError("preview hint is invalid") from exc
            if source.node_id not in local_node_set:
                raise ValueError(
                    "preview hint node must belong to its feature emission scope"
                )
            preview_sources.append(source)

    if not progress_phases and not preview_sources:
        return None, None
    if not progress_phases:
        raise ValueError("preview hints require explicit progress hints")

    # These constructors enforce positive normalized weights, unique phase and
    # node ids, unique preview sources, known supersedes targets and an acyclic
    # supersedes relation.
    progress_spec = ProgressSpec(version=1, phases=tuple(progress_phases))
    preview_spec = PreviewSpec(version=1, sources=tuple(preview_sources))
    phase_ids = {phase.id for phase in progress_spec.phases}
    if any(source.phase_id not in phase_ids for source in preview_spec.sources):
        raise ValueError("preview hint references an unknown progress phase")
    return progress_spec, preview_spec


def _prompt_node_ids(prompt: Mapping[str, Any]) -> frozenset[str]:
    if not prompt:
        raise ValueError("prompt must contain at least one node")
    node_ids: set[str] = set()
    for node_id, node in prompt.items():
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("prompt node ids must be non-empty strings")
        if not isinstance(node, Mapping):
            raise ValueError(f"prompt node {node_id!r} must be an object")
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(class_type, str) or not class_type:
            raise ValueError(f"prompt node {node_id!r} has no class_type")
        if not isinstance(inputs, Mapping):
            raise ValueError(f"prompt node {node_id!r} inputs must be an object")
        node_ids.add(node_id)
    return frozenset(node_ids)


def _validate_prompt_specs(
    *,
    prompt: Mapping[str, Any],
    expected_output_spec: ExpectedOutputSpec | None,
    progress_spec: ProgressSpec | None,
    preview_spec: PreviewSpec | None,
) -> None:
    node_ids = _prompt_node_ids(prompt)
    if expected_output_spec is not None and expected_output_spec.node_id not in node_ids:
        raise ValueError("expected output node does not exist in prompt")
    if progress_spec is not None:
        missing_progress = {
            phase.node_id for phase in progress_spec.phases if phase.node_id not in node_ids
        }
        if missing_progress:
            raise ValueError("progress spec references nodes absent from prompt")
    if preview_spec is not None:
        missing_preview = {
            source.node_id for source in preview_spec.sources if source.node_id not in node_ids
        }
        if missing_preview:
            raise ValueError("preview spec references nodes absent from prompt")
        if progress_spec is None and preview_spec.sources:
            raise ValueError("preview sources require a progress spec")
        phase_ids = (
            {phase.id for phase in progress_spec.phases}
            if progress_spec is not None
            else set()
        )
        if any(source.phase_id not in phase_ids for source in preview_spec.sources):
            raise ValueError("preview spec references an unknown progress phase")


def _validate_prompt_audit(
    *,
    prompt: Mapping[str, Any],
    graph_audit_spec: GraphAuditSpec,
    unit_kind: Literal["segment", "control"],
    expected_output_spec: ExpectedOutputSpec | None,
) -> None:
    node_ids = _prompt_node_ids(prompt)
    if graph_audit_spec.unit_kind != unit_kind:
        raise ValueError("graph audit unit kind must match the execution unit")
    if set(graph_audit_spec.node_contract_snapshot) != node_ids:
        raise ValueError("graph audit node snapshot must exactly cover the prompt")
    for node_id, evidence in graph_audit_spec.node_contract_snapshot.items():
        if prompt[node_id]["class_type"] != evidence.class_type:
            raise ValueError(
                f"prompt node {node_id!r} class_type does not match graph audit"
            )
    if unit_kind == "segment":
        if expected_output_spec is None:
            raise ValueError("segment graph audit requires an expected output")
        if graph_audit_spec.take_node_id != expected_output_spec.node_id:
            raise ValueError("graph audit take node must match expected output node")


class EndpointIdentity(FrozenContractModel):
    schema_version: PositiveInt = 1
    endpoint_key: Identifier
    runtime_instance_id: Identifier


class RuntimeRequirements(FrozenContractModel):
    endpoint_key: Identifier
    backend: Backend
    logical_gpu_indices: Annotated[
        tuple[Annotated[int, Field(ge=0, le=255)], ...], Field(max_length=8)
    ]
    ray_compatibility_key: NonEmptyString | None = None
    ray_runtime_key: NonEmptyString | None = None
    requires_standard_driver_access: bool
    expected_residency_policy: Literal[
        "keep_until_switch", "release_after_sampling"
    ] | None = None

    @model_validator(mode="after")
    def validate_backend_requirements(self) -> "RuntimeRequirements":
        if len(set(self.logical_gpu_indices)) != len(self.logical_gpu_indices):
            raise ValueError("logical GPU indices must be unique")
        if self.backend == "raylight":
            if len(self.logical_gpu_indices) < 2:
                raise ValueError("RayLight requires at least two logical GPUs")
            if self.ray_compatibility_key is None or self.ray_runtime_key is None:
                raise ValueError("RayLight requires compatibility and runtime keys")
            if self.expected_residency_policy is None:
                raise ValueError("RayLight requires an expected residency policy")
        elif any(
            value is not None
            for value in (
                self.ray_compatibility_key,
                self.ray_runtime_key,
                self.expected_residency_policy,
            )
        ):
            raise ValueError("Standard requirements cannot carry Ray runtime fields")
        return self


class ComfyNodeIdentity(FrozenContractModel):
    node_id: Identifier
    class_type: NonEmptyString
    inputs: JsonObject


class ComfyNodeCacheIdentity(FrozenContractModel):
    schema_version: PositiveInt = 1
    nodes: Annotated[tuple[ComfyNodeIdentity, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_node_ids(self) -> "ComfyNodeCacheIdentity":
        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Comfy node cache identity node ids must be unique")
        return self


def comfy_node_cache_identity(prompt: Mapping[str, Any]) -> ComfyNodeCacheIdentity:
    _prompt_node_ids(prompt)
    return ComfyNodeCacheIdentity(
        nodes=tuple(
            ComfyNodeIdentity(
                node_id=node_id,
                class_type=str(prompt[node_id]["class_type"]),
                inputs=prompt[node_id]["inputs"],
            )
            for node_id in sorted(prompt, key=utf16_sort_key)
        )
    )


class FeatureExecutionIdentity(FrozenContractModel):
    feature: Annotated[str, Field(min_length=3, max_length=256)]
    effective_cache_params: JsonObject
    resolved_implementations: Annotated[
        tuple[
            ResolvedImplementationIdentity | ResolvedFeatureImplementation,
            ...,
        ],
        Field(min_length=1),
    ]

    @field_validator("feature")
    @classmethod
    def validate_feature(cls, value: str) -> str:
        if _FEATURE_PATTERN.fullmatch(value) is None:
            raise ValueError("feature identity must use id@version")
        return value

    @model_validator(mode="after")
    def validate_implementations(self) -> "FeatureExecutionIdentity":
        bindings = [
            implementation.binding_key
            for implementation in self.resolved_implementations
        ]
        if len(set(bindings)) != len(bindings):
            raise ValueError("resolved implementation bindings must be unique")
        return self


class SegmentExecutionIdentity(FrozenContractModel):
    schema_version: PositiveInt
    segment_creative_input: JsonObject
    render: JsonObject
    family_sampling: JsonObject
    model_stack_projection: JsonObject
    runtime_placement_projection: JsonObject | None = None
    feature_execution_identities: tuple[FeatureExecutionIdentity, ...]
    continuity_input_identity: JsonObject | None
    expected_output_geometry: ExpectedOutputGeometry

    @model_validator(mode="after")
    def validate_features(self) -> "SegmentExecutionIdentity":
        features = [identity.feature for identity in self.feature_execution_identities]
        if len(set(features)) != len(features):
            raise ValueError("segment feature execution identities must be unique")
        return self


class HistoricalTakeGeometryIdentity(FrozenContractModel):
    project_id: Identifier
    segment_id: Identifier
    width: PositiveInt
    height: PositiveInt
    fps: PositiveFloat
    visible_frame_count: PositiveInt
    required_audio_capability: bool


class RuntimePoolIdentityContribution(FrozenContractModel):
    feature: Annotated[str, Field(min_length=3, max_length=256)]
    identity: JsonObject

    @field_validator("feature")
    @classmethod
    def validate_feature(cls, value: str) -> str:
        if _FEATURE_PATTERN.fullmatch(value) is None:
            raise ValueError("runtime pool feature must use id@version")
        return value


class RayRuntimeIdentity(FrozenContractModel):
    schema_version: PositiveInt
    backend: Literal["raylight"] = "raylight"
    placement: JsonObject
    fixed_parameters: JsonObject
    active_feature_pool_identities: tuple[RuntimePoolIdentityContribution, ...]

    @model_validator(mode="after")
    def validate_features(self) -> "RayRuntimeIdentity":
        features = [item.feature for item in self.active_feature_pool_identities]
        if len(set(features)) != len(features):
            raise ValueError("Ray runtime feature pool identities must be unique")
        return self


def effective_execution_digest(
    segment_identity: SegmentExecutionIdentity,
    *,
    template_id: str,
    template_revision: int,
    resolved_node_contract_identities: Sequence[GraphNodeContractEvidence],
) -> DocumentDigest:
    if not template_id or template_revision < 1:
        raise ValueError("template id must be non-empty and revision must be positive")
    if not resolved_node_contract_identities:
        raise ValueError("effective execution identity requires node contracts")
    payload = {
        "segment_execution_identity": segment_identity.model_dump(mode="json"),
        "template": {"id": template_id, "revision": template_revision},
        "resolved_node_contract_identities": [
            identity.model_dump(mode="json")
            for identity in resolved_node_contract_identities
        ],
    }
    return sha256_document_digest(payload)


class PreparedSegmentUnit(FrozenContractModel):
    id: Identifier
    owner_segment_id: Identifier
    family: ModelFamily
    backend: Backend
    template_id: Identifier
    template_revision: PositiveInt
    prompt_base: JsonObject
    graph_audit_spec: GraphAuditSpec
    expected_output_spec: ExpectedOutputSpec
    progress_spec: ProgressSpec
    preview_spec: PreviewSpec
    continuity_dependency: JsonObject | None
    runtime_requirements: RuntimeRequirements
    runtime_pool_identity: RayRuntimeIdentity | None
    effective_execution_digest: DocumentDigest

    @model_validator(mode="after")
    def validate_unit(self) -> "PreparedSegmentUnit":
        if self.expected_output_spec.segment_id != self.owner_segment_id:
            raise ValueError("expected output segment must match unit owner")
        if self.runtime_requirements.backend != self.backend:
            raise ValueError("runtime requirements backend must match unit backend")
        if self.backend == "raylight" and self.runtime_pool_identity is None:
            raise ValueError("RayLight unit requires a runtime pool identity")
        if self.backend == "standard" and self.runtime_pool_identity is not None:
            raise ValueError("Standard unit cannot carry a Ray runtime pool identity")
        if self.effective_execution_digest.algorithm != "sha256-canonical-json-v1":
            raise ValueError("segment execution digest must use canonical SHA-256")
        _validate_prompt_specs(
            prompt=self.prompt_base,
            expected_output_spec=self.expected_output_spec,
            progress_spec=self.progress_spec,
            preview_spec=self.preview_spec,
        )
        _validate_prompt_audit(
            prompt=self.prompt_base,
            graph_audit_spec=self.graph_audit_spec,
            unit_kind="segment",
            expected_output_spec=self.expected_output_spec,
        )
        return self


class LockedSegmentUnit(PreparedSegmentUnit):
    kind: Literal["segment"] = "segment"
    child_id: Identifier
    requested_prompt_id: Identifier
    group_index: PositiveInt
    exact_prompt: JsonObject
    late_binding_evidence: tuple[LateBindingEvidence, ...] = ()
    late_bound_values: JsonObject

    @model_validator(mode="after")
    def validate_exact_prompt(self) -> "LockedSegmentUnit":
        evidence_pointers = [
            evidence.input_pointer for evidence in self.late_binding_evidence
        ]
        if len(set(evidence_pointers)) != len(evidence_pointers):
            raise ValueError("late-binding evidence pointers must be unique")
        expected_values: dict[str, Any] = {}
        for evidence in self.late_binding_evidence:
            if isinstance(evidence, ContinuityLateBindingEvidence):
                expected_values[evidence.input_pointer] = evidence.bound_value
                continue
            compatibility_key = self.runtime_requirements.ray_compatibility_key
            if self.backend != "raylight" or compatibility_key is None:
                raise ValueError(
                    "runtime epoch evidence requires RayLight runtime identity"
                )
            expected_values[evidence.input_pointer] = (
                f"{compatibility_key}-e{evidence.epoch}"
            )
        if canonical_json_bytes(expected_values) != canonical_json_bytes(
            self.late_bound_values
        ):
            raise ValueError(
                "late-bound values must be derived from typed binding evidence"
            )
        _validate_prompt_specs(
            prompt=self.exact_prompt,
            expected_output_spec=self.expected_output_spec,
            progress_spec=self.progress_spec,
            preview_spec=self.preview_spec,
        )
        _validate_prompt_audit(
            prompt=self.exact_prompt,
            graph_audit_spec=self.graph_audit_spec,
            unit_kind="segment",
            expected_output_spec=self.expected_output_spec,
        )
        return self

    def validate_materialized_prompt(
        self,
        *,
        node_contract_registry: NodeContractRegistry,
    ) -> "LockedSegmentUnit":
        """Prove exact materialization against the prepared graph and evidence."""

        from .audit import validate_bound_graph

        validate_bound_graph(
            prompt_base=self.prompt_base,
            bound_prompt=self.exact_prompt,
            spec=self.graph_audit_spec,
            node_contract_registry=node_contract_registry,
            model_family=self.family,
            backend=self.backend,
            expected_late_bound_values=self.late_bound_values,
            enforce_runtime_effects=False,
        )
        return self


class PreparedControlUnit(FrozenContractModel):
    id: Identifier
    kind: Literal["control"] = "control"
    control_kind: Literal["ray_kill"] = "ray_kill"
    owner_segment_id: None = None
    family: ModelFamily
    backend: Literal["raylight"] = "raylight"
    template_id: Literal["raylight_kill_control"] = "raylight_kill_control"
    template_revision: PositiveInt
    child_id: Identifier
    requested_prompt_id: Identifier
    group_index: NonNegativeInt
    prompt_base: JsonObject
    graph_audit_spec: GraphAuditSpec
    runtime_descriptor_digest: DocumentDigest
    effective_execution_digest: DocumentDigest
    preceding_unit_id: Identifier
    expected_output_spec: None = None
    progress_spec: None = None
    preview_spec: None = None

    @model_validator(mode="after")
    def validate_control(self) -> "PreparedControlUnit":
        if self.preceding_unit_id == self.id:
            raise ValueError("control unit cannot precede itself")
        if self.runtime_descriptor_digest.algorithm != "sha256-canonical-json-v1":
            raise ValueError("control runtime descriptor must use canonical SHA-256")
        if self.effective_execution_digest.algorithm != "sha256-canonical-json-v1":
            raise ValueError("control execution digest must use canonical SHA-256")
        _validate_prompt_specs(
            prompt=self.prompt_base,
            expected_output_spec=None,
            progress_spec=None,
            preview_spec=None,
        )
        _validate_prompt_audit(
            prompt=self.prompt_base,
            graph_audit_spec=self.graph_audit_spec,
            unit_kind="control",
            expected_output_spec=None,
        )
        return self


def _coerce_compiled_execution_report(value: Any) -> Any:
    if isinstance(value, (CompiledExecutionReportV2, CompiledExecutionReportV3)):
        return value
    if (
        isinstance(value, dict)
        and value.get("source") == "v4_native_compile_adapter_v2"
    ):
        return CompiledExecutionReportV2.model_validate_json(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    if isinstance(value, dict) and value.get("source") == "bundle6_native_compile_v3":
        return CompiledExecutionReportV3.model_validate_json(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    return value


class CompiledExecutionPlan(FrozenContractModel):
    version: PositiveInt
    template_bundle_version: PositiveInt
    segment_units: Annotated[tuple[PreparedSegmentUnit, ...], Field(min_length=1)]
    # V2 is a dedicated bounded contract because a legal 128-segment plan can
    # contain more than the generic JsonObject container limit of 256 feature
    # resolution records. JsonObject remains only for frozen historical V1.
    compile_report: Annotated[
        CompiledExecutionReportV2 | CompiledExecutionReportV3 | JsonObject,
        BeforeValidator(_coerce_compiled_execution_report),
    ]
    node_policy: JsonObject
    effective_execution_digest: DocumentDigest

    @property
    def plan_execution_digest(self) -> DocumentDigest:
        """Explicit name for the compatibility-named plan aggregate digest."""

        return self.effective_execution_digest

    @model_validator(mode="after")
    def validate_plan(self) -> "CompiledExecutionPlan":
        if any(type(unit) is not PreparedSegmentUnit for unit in self.segment_units):
            raise ValueError("compiled plan may contain prepared segment units only")
        unit_ids = [unit.id for unit in self.segment_units]
        owners = [unit.owner_segment_id for unit in self.segment_units]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("compiled segment unit ids must be unique")
        if len(set(owners)) != len(owners):
            raise ValueError("compiled segment owners must be unique")
        if self.effective_execution_digest.algorithm != "sha256-canonical-json-v1":
            raise ValueError("compiled plan digest must use canonical SHA-256")
        if self.version == 1:
            # Frozen historical Stage-4 plans predate typed compile evidence.
            if isinstance(
                self.compile_report,
                (CompiledExecutionReportV2, CompiledExecutionReportV3),
            ):
                raise ValueError("historical plan cannot carry a v2 compile report")
            return self
        if self.version not in {2, 3}:
            raise ValueError("compiled execution plan version is unsupported")
        report_type = (
            CompiledExecutionReportV2
            if self.version == 2
            else CompiledExecutionReportV3
        )
        if not isinstance(self.compile_report, report_type):
            raise ValueError(f"v{self.version} plan requires its typed compile report")
        if self.version == 3 and self.template_bundle_version != 6:
            raise ValueError("v3 plan requires Bundle 6")
        report = self.compile_report
        units_by_id = {unit.id: unit for unit in self.segment_units}
        digest_unit_ids = tuple(
            item.unit_id for item in report.unit_effective_execution_digests
        )
        if digest_unit_ids != tuple(unit.id for unit in self.segment_units):
            raise ValueError("compiled report unit digest order differs from plan")
        resolution_unit_ids = {item.unit_id for item in report.feature_resolutions}
        if resolution_unit_ids != set(units_by_id):
            raise ValueError("compiled report resolutions must cover every plan unit")
        for item in report.feature_resolutions:
            unit = units_by_id[item.unit_id]
            if (
                item.segment_id != unit.owner_segment_id
                or item.backend != unit.backend
                or item.family != unit.family
                or item.template_id != unit.template_id
            ):
                raise ValueError(
                    "compiled feature resolution differs from its prepared unit"
                )
        plan_segment_ids = tuple(
            item.get("segment_id") for item in report.plans
        )
        if plan_segment_ids != tuple(
            unit.owner_segment_id for unit in self.segment_units
        ):
            raise ValueError("compiled report plans differ from prepared units")
        return self


def compiled_execution_plan_digest(
    plan: CompiledExecutionPlan,
) -> CompiledPlanDigest:
    digest = sha256_document_digest(plan.model_dump(mode="json"))
    return CompiledPlanDigest(value=digest.value)


def compiled_execution_plan_digest_from_canonical_json(
    canonical_plan_json: str,
) -> CompiledPlanDigest:
    """Hash already-canonical persisted plan bytes without re-encoding JSON.

    This narrow boundary is for storage code that owns the exact output of
    :func:`canonical_json`.  It deliberately does not accept arbitrary JSON:
    callers must first establish that these are the immutable canonical bytes
    whose digest is being verified.
    """

    if not isinstance(canonical_plan_json, str):
        raise TypeError("canonical compiled plan must be a string")
    digest = hashlib.sha256(canonical_plan_json.encode("utf-8")).hexdigest()
    return CompiledPlanDigest(value=f"sha256-{digest}")


def ordered_compiled_segment_units(
    plan: CompiledExecutionPlan,
    config_snapshot: Mapping[str, Any],
) -> tuple[PreparedSegmentUnit, ...]:
    """Resolve immutable compiled units in authored timeline output order.

    ``segment_units`` intentionally follows the endpoint submission order so
    compatible model families can remain resident.  The final movie must
    instead follow the authored timeline.  The job's captured creative
    snapshot is the durable ordering authority; this helper requires it to
    name exactly the same owner set as the compiled plan.
    """

    timeline = config_snapshot.get("timeline")
    if not isinstance(timeline, Mapping):
        raise ValueError("timeline output order requires a captured timeline")
    segments = timeline.get("segments")
    if not isinstance(segments, (list, tuple)):
        raise ValueError("captured timeline has no segment order")
    requested = config_snapshot.get("segment_ids")
    if requested is None:
        selected: set[str] | None = None
    elif isinstance(requested, (list, tuple)) and all(
        isinstance(item, str) and item for item in requested
    ):
        if len(set(requested)) != len(requested):
            raise ValueError("captured segment selection contains duplicates")
        selected = set(requested)
    else:
        raise ValueError("captured segment selection is invalid")

    ordered_ids: list[str] = []
    seen_timeline_ids: set[str] = set()
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("captured timeline contains an invalid segment")
        segment_id = segment.get("id")
        if not isinstance(segment_id, str) or not segment_id:
            raise ValueError("captured timeline segment id is invalid")
        if segment_id in seen_timeline_ids:
            raise ValueError("captured timeline segment ids must be unique")
        seen_timeline_ids.add(segment_id)
        if segment.get("enabled", True) is not True:
            continue
        if selected is None or segment_id in selected:
            ordered_ids.append(segment_id)

    units_by_owner = {unit.owner_segment_id: unit for unit in plan.segment_units}
    if (
        len(units_by_owner) != len(plan.segment_units)
        or len(ordered_ids) != len(plan.segment_units)
        or set(ordered_ids) != set(units_by_owner)
    ):
        raise ValueError(
            "captured timeline output order differs from compiled segment owners"
        )
    return tuple(units_by_owner[segment_id] for segment_id in ordered_ids)


LockedSubmissionUnit: TypeAlias = LockedSegmentUnit | PreparedControlUnit


class ControlContinuationDependency(FrozenContractModel):
    """Frozen identity of the RayKill proof required by a continuation.

    A post-control plan contains only the segment that is about to cross the
    network. Without this identity, deleting the preceding control row could
    make that plan indistinguishable from a segment that never needed a
    barrier. The dependency commits both the original two-unit wave and its
    exact control snapshot before the continuation is persisted.
    """

    schema_version: Literal[1] = 1
    control_child_id: Identifier
    control_unit_id: Identifier
    control_requested_prompt_id: PromptIdentifier
    control_group_index: NonNegativeInt
    original_locked_plan_digest: DocumentDigest
    control_exact_prompt_snapshot_digest: DocumentDigest

    @model_validator(mode="after")
    def validate_digests(self) -> "ControlContinuationDependency":
        for label, digest in (
            ("original locked plan", self.original_locked_plan_digest),
            ("control exact prompt snapshot", self.control_exact_prompt_snapshot_digest),
        ):
            if digest.algorithm != "sha256-canonical-json-v1":
                raise ValueError(f"{label} dependency requires canonical SHA-256")
        return self


class LockedSubmissionPlan(FrozenContractModel):
    version: PositiveInt
    endpoint_identity: EndpointIdentity
    units: Annotated[tuple[LockedSubmissionUnit, ...], Field(min_length=1)]
    source_compiled_plan_digest: CompiledPlanDigest
    source_unit_id: Identifier
    source_unit_ordinal: NonNegativeInt
    ray_ledger_before: JsonObject | None
    ray_ledger_after_intent: JsonObject | None
    control_dependency: ControlContinuationDependency | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> "LockedSubmissionPlan":
        unit_ids = [unit.id for unit in self.units]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("locked submission unit ids must be unique")
        child_ids = [unit.child_id for unit in self.units]
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("locked submission child ids must be unique")
        prompt_ids = [unit.requested_prompt_id for unit in self.units]
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("locked submission requested prompt ids must be unique")
        if len(self.units) not in {1, 2}:
            raise ValueError("locked submission wave must contain one or two units")
        segment = self.units[-1]
        if not isinstance(segment, LockedSegmentUnit):
            raise ValueError("locked submission wave must end with one segment")
        if len(self.units) == 2:
            if self.control_dependency is not None:
                raise ValueError(
                    "an original control wave cannot depend on another control"
                )
            control = self.units[0]
            if not isinstance(control, PreparedControlUnit):
                raise ValueError("two-unit submission wave must start with a control")
            if control.preceding_unit_id != segment.id:
                raise ValueError(
                    "control unit must immediately precede its declared segment"
                )
        elif self.control_dependency is not None:
            dependency = self.control_dependency
            if dependency.control_child_id == segment.child_id:
                raise ValueError("control dependency child must differ from segment child")
            if dependency.control_unit_id == segment.id:
                raise ValueError("control dependency unit must differ from segment unit")
            if dependency.control_group_index + 1 != segment.group_index:
                raise ValueError(
                    "control dependency group must immediately precede its segment"
                )
        group_indices = [unit.group_index for unit in self.units]
        if any(
            current + 1 != following
            for current, following in zip(group_indices, group_indices[1:])
        ):
            raise ValueError(
                "locked submission group indices must be globally adjacent"
            )
        expected_segment_group = self.source_unit_ordinal * 2 + 1
        if segment.group_index != expected_segment_group:
            raise ValueError(
                "locked segment group index must match its global source ordinal"
            )
        if len(self.units) == 2 and self.units[0].group_index != (
            expected_segment_group - 1
        ):
            raise ValueError(
                "locked control group index must immediately precede segment globally"
            )
        if segment.id != self.source_unit_id:
            raise ValueError("locked segment must match the declared source unit")
        if (
            segment.runtime_requirements.endpoint_key
            != self.endpoint_identity.endpoint_key
        ):
            raise ValueError(
                "locked segment endpoint key must match endpoint identity"
            )
        return self

    def validate_source_compiled_plan(
        self,
        compiled_plan: CompiledExecutionPlan,
    ) -> "LockedSubmissionPlan":
        expected_digest = compiled_execution_plan_digest(compiled_plan)
        if self.source_unit_ordinal >= len(compiled_plan.segment_units):
            raise ValueError("locked plan source unit ordinal is out of range")
        prepared = compiled_plan.segment_units[self.source_unit_ordinal]
        return self.validate_source_prepared_unit(
            prepared,
            verified_source_compiled_plan_digest=expected_digest,
        )

    def validate_source_prepared_unit(
        self,
        prepared: PreparedSegmentUnit,
        *,
        verified_source_compiled_plan_digest: CompiledPlanDigest,
    ) -> "LockedSubmissionPlan":
        """Cross-link one strict prepared unit after its plan bytes were verified.

        The caller must independently authenticate the complete compiled-plan
        bytes against ``verified_source_compiled_plan_digest``.  This split lets
        transactional readers validate the current ordinal without repeatedly
        canonicalizing every sibling unit.
        """

        if (
            self.source_compiled_plan_digest
            != verified_source_compiled_plan_digest
        ):
            raise ValueError("locked plan source digest does not match compiled plan")
        locked = self.units[-1]
        assert isinstance(locked, LockedSegmentUnit)
        if prepared.id != self.source_unit_id or locked.id != prepared.id:
            raise ValueError("locked segment does not match compiled source unit")
        stable_fields = (
            "id",
            "owner_segment_id",
            "family",
            "backend",
            "template_id",
            "template_revision",
            "prompt_base",
            "continuity_dependency",
            "graph_audit_spec",
            "expected_output_spec",
            "progress_spec",
            "preview_spec",
            "runtime_requirements",
            "runtime_pool_identity",
            "effective_execution_digest",
        )
        locked_projection = locked.model_dump(mode="json", include=set(stable_fields))
        prepared_projection = prepared.model_dump(
            mode="json", include=set(stable_fields)
        )
        if not canonical_values_equal(locked_projection, prepared_projection):
            raise ValueError("locked segment identity drifted from compiled plan")

        return self


def locked_submission_plan_from_compiled(
    compiled_plan: CompiledExecutionPlan,
    *,
    endpoint_identity: EndpointIdentity,
    units: tuple[LockedSubmissionUnit, ...],
    source_unit_id: str,
    source_unit_ordinal: int,
    ray_ledger_before: Mapping[str, Any] | None,
    ray_ledger_after_intent: Mapping[str, Any] | None,
    version: int = 1,
) -> LockedSubmissionPlan:
    """Build and verify one locked plan against its exact compiled source."""

    plan = LockedSubmissionPlan(
        version=version,
        endpoint_identity=endpoint_identity,
        units=units,
        source_compiled_plan_digest=compiled_execution_plan_digest(compiled_plan),
        source_unit_id=source_unit_id,
        source_unit_ordinal=source_unit_ordinal,
        ray_ledger_before=ray_ledger_before,
        ray_ledger_after_intent=ray_ledger_after_intent,
    )
    return plan.validate_source_compiled_plan(compiled_plan)


class ExactPromptSnapshot(FrozenContractModel):
    schema_version: PositiveInt
    unit_id: Identifier
    unit_kind: Literal["segment", "control"]
    owner_segment_id: Identifier | None
    control_kind: Literal["ray_kill"] | None
    family: ModelFamily
    backend: Backend
    template_id: Identifier
    template_revision: PositiveInt
    endpoint_identity: EndpointIdentity
    exact_prompt: JsonObject
    graph_audit_spec: GraphAuditSpec
    expected_output_spec: ExpectedOutputSpec | None
    progress_spec: ProgressSpec | None
    preview_spec: PreviewSpec | None
    effective_execution_digest: DocumentDigest

    @model_validator(mode="after")
    def validate_snapshot(self) -> "ExactPromptSnapshot":
        if self.effective_execution_digest.algorithm != "sha256-canonical-json-v1":
            raise ValueError("exact prompt digest must use canonical SHA-256")
        if self.unit_kind == "segment":
            if self.owner_segment_id is None or self.control_kind is not None:
                raise ValueError("segment snapshot requires an owner and no control kind")
            if (
                self.expected_output_spec is None
                or self.progress_spec is None
                or self.preview_spec is None
            ):
                raise ValueError("segment snapshot requires output, progress and preview specs")
            if self.expected_output_spec.segment_id != self.owner_segment_id:
                raise ValueError("snapshot output segment must match owner")
        else:
            if self.owner_segment_id is not None or self.control_kind != "ray_kill":
                raise ValueError("control snapshot must be an ownerless RayKill")
            if self.backend != "raylight" or self.template_id != "raylight_kill_control":
                raise ValueError("control snapshot must use the RayLight kill template")
            if any(
                spec is not None
                for spec in (
                    self.expected_output_spec,
                    self.progress_spec,
                    self.preview_spec,
                )
            ):
                raise ValueError("control snapshot cannot carry user-facing specs")
        _validate_prompt_specs(
            prompt=self.exact_prompt,
            expected_output_spec=self.expected_output_spec,
            progress_spec=self.progress_spec,
            preview_spec=self.preview_spec,
        )
        _validate_prompt_audit(
            prompt=self.exact_prompt,
            graph_audit_spec=self.graph_audit_spec,
            unit_kind=self.unit_kind,
            expected_output_spec=self.expected_output_spec,
        )
        return self


def _normalize_evidence_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("ownership evidence timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


class HistoryTerminalEvidence(FrozenContractModel):
    kind: Literal["history_terminal"] = "history_terminal"
    prompt_id: PromptIdentifier
    terminal_status: Literal["succeeded", "failed", "cancelled"]
    history_digest: DocumentDigest
    observed_at: datetime

    _normalize_observed_at = field_validator("observed_at")(
        _normalize_evidence_timestamp
    )

    @model_validator(mode="after")
    def validate_history_digest(self) -> "HistoryTerminalEvidence":
        if self.history_digest.algorithm != "sha256-canonical-json-v1":
            raise ValueError("history terminal evidence requires canonical SHA-256")
        return self


class OutputObservationReceipt(FrozenContractModel):
    """Immutable bridge from exact terminal history to local media probing.

    The receipt deliberately contains no filesystem path.  It commits the
    exact history descriptor to the immutable prompt snapshot and permits a
    later process to finish media observation without querying ComfyUI history
    again or re-running the compiler.
    """

    schema_version: Literal[1] = 1
    child_id: Identifier
    segment_id: Identifier
    node_id: Identifier
    output_descriptor: OutputDescriptor
    exact_prompt_snapshot_digest: DocumentDigest
    expected_output_spec_digest: DocumentDigest
    history_evidence: HistoryTerminalEvidence

    @model_validator(mode="after")
    def validate_receipt(self) -> "OutputObservationReceipt":
        for label, digest in (
            ("exact prompt snapshot", self.exact_prompt_snapshot_digest),
            ("expected output spec", self.expected_output_spec_digest),
        ):
            if digest.algorithm != "sha256-canonical-json-v1":
                raise ValueError(f"{label} digest requires canonical SHA-256")
        if self.history_evidence.terminal_status != "succeeded":
            raise ValueError("an output observation receipt requires successful history")
        return self


class ExactCancelConfirmedEvidence(FrozenContractModel):
    kind: Literal["exact_cancel_confirmed"] = "exact_cancel_confirmed"
    prompt_id: PromptIdentifier
    confirmation_id: NonEmptyString
    confirmed_at: datetime

    _normalize_confirmed_at = field_validator("confirmed_at")(
        _normalize_evidence_timestamp
    )


class EndpointRestartCertificate(FrozenContractModel):
    kind: Literal["endpoint_restart_certificate"] = "endpoint_restart_certificate"
    certificate_version: Literal[1]
    prompt_id: PromptIdentifier
    endpoint_identity: EndpointIdentity
    restart_id: Identifier
    queue_and_history_cleared: Literal[True]
    confirmed_at: datetime

    _normalize_confirmed_at = field_validator("confirmed_at")(
        _normalize_evidence_timestamp
    )


PromptReleaseEvidence: TypeAlias = Annotated[
    HistoryTerminalEvidence
    | ExactCancelConfirmedEvidence
    | EndpointRestartCertificate,
    Field(discriminator="kind"),
]


class PromptOwnership(FrozenContractModel):
    requested_prompt_id: Identifier
    actual_prompt_id: PromptIdentifier | None = None
    state: PromptOwnershipState
    ownership_revision: Annotated[int, Field(ge=0, le=MAX_SAFE_JSON_INTEGER)]
    cleanup_certificate: PromptReleaseEvidence | None = None
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ownership timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_state_evidence(self) -> "PromptOwnership":
        if self.state in {"prepared", "submitting", "owned_requested_id"}:
            if self.actual_prompt_id is not None:
                raise ValueError(f"ownership state {self.state} cannot carry actual id")
        if self.state == "owned_actual_id" and self.actual_prompt_id is None:
            raise ValueError("owned_actual_id requires the actual prompt id")
        confirmed_states = {"cleanup_confirmed", "terminal_confirmed"}
        if self.state in confirmed_states and self.cleanup_certificate is None:
            raise ValueError(f"{self.state} requires structured release evidence")
        if self.cleanup_certificate is not None and self.state not in confirmed_states:
            raise ValueError("release evidence requires a confirmed ownership state")
        if (
            self.cleanup_certificate is not None
            and self.cleanup_certificate.prompt_id != self.effective_prompt_id
        ):
            raise ValueError("release evidence must match the effective prompt id")
        if self.state == "terminal_confirmed" and not isinstance(
            self.cleanup_certificate, HistoryTerminalEvidence
        ):
            raise ValueError("terminal_confirmed requires history terminal evidence")
        if self.state == "cleanup_confirmed" and isinstance(
            self.cleanup_certificate, HistoryTerminalEvidence
        ):
            raise ValueError(
                "cleanup_confirmed requires cancel or endpoint restart evidence"
            )
        return self

    @property
    def effective_prompt_id(self) -> str:
        return self.actual_prompt_id or self.requested_prompt_id


class OwnershipRevisionConflict(ValueError):
    """The caller attempted an ownership update from a stale revision."""


class InvalidOwnershipTransition(ValueError):
    """The caller attempted a non-monotonic ownership transition."""


_OWNERSHIP_TRANSITIONS: Final[dict[str, frozenset[str]]] = {
    "prepared": frozenset({"prepared", "submitting", "unconfirmed"}),
    "submitting": frozenset(
        {
            "submitting",
            "owned_requested_id",
            "owned_actual_id",
            "cancel_pending",
            "cleanup_confirmed",
            "terminal_confirmed",
            "unconfirmed",
        }
    ),
    "owned_requested_id": frozenset(
        {
            "owned_requested_id",
            "owned_actual_id",
            "cancel_pending",
            "cleanup_confirmed",
            "terminal_confirmed",
            "unconfirmed",
        }
    ),
    "owned_actual_id": frozenset(
        {
            "owned_actual_id",
            "cancel_pending",
            "cleanup_confirmed",
            "terminal_confirmed",
            "unconfirmed",
        }
    ),
    "cancel_pending": frozenset(
        {
            "cancel_pending",
            "cleanup_confirmed",
            "terminal_confirmed",
            "unconfirmed",
        }
    ),
    "unconfirmed": frozenset(
        {
            "unconfirmed",
            "owned_requested_id",
            "owned_actual_id",
            "cancel_pending",
            "cleanup_confirmed",
            "terminal_confirmed",
        }
    ),
    "cleanup_confirmed": frozenset({"cleanup_confirmed"}),
    "terminal_confirmed": frozenset({"terminal_confirmed"}),
}


def effective_prompt_id(ownership: PromptOwnership) -> str:
    return ownership.effective_prompt_id


def can_transition_prompt_ownership(
    current: PromptOwnership,
    next_state: PromptOwnershipState,
) -> bool:
    return next_state in _OWNERSHIP_TRANSITIONS[current.state]


def transition_prompt_ownership(
    current: PromptOwnership,
    *,
    expected_revision: int,
    state: PromptOwnershipState,
    updated_at: datetime,
    actual_prompt_id: str | None | object = _UNSET,
    cleanup_certificate: PromptReleaseEvidence | Mapping[str, Any] | None | object = _UNSET,
) -> PromptOwnership:
    """Apply one CAS-guarded, monotonic ownership transition."""

    if expected_revision != current.ownership_revision:
        raise OwnershipRevisionConflict(
            f"expected ownership revision {expected_revision}, "
            f"found {current.ownership_revision}"
        )
    if not can_transition_prompt_ownership(current, state):
        raise InvalidOwnershipTransition(
            f"ownership cannot transition from {current.state} to {state}"
        )

    next_actual = (
        current.actual_prompt_id if actual_prompt_id is _UNSET else actual_prompt_id
    )
    if next_actual is not None and not isinstance(next_actual, str):
        raise TypeError("actual prompt id must be a string or None")
    if current.actual_prompt_id is not None and next_actual != current.actual_prompt_id:
        raise InvalidOwnershipTransition("actual prompt id cannot be cleared or changed")

    next_certificate = (
        current.cleanup_certificate
        if cleanup_certificate is _UNSET
        else cleanup_certificate
    )
    if next_certificate is not None and not isinstance(
        next_certificate,
        (
            Mapping,
            HistoryTerminalEvidence,
            ExactCancelConfirmedEvidence,
            EndpointRestartCertificate,
        ),
    ):
        raise TypeError("cleanup certificate must be structured release evidence")
    if current.cleanup_certificate is not None:
        current_evidence = current.cleanup_certificate.model_dump(mode="json")
        next_evidence = (
            next_certificate.model_dump(mode="json")
            if isinstance(next_certificate, FrozenContractModel)
            else next_certificate
        )
        if next_evidence is None or canonical_json_bytes(
            next_evidence
        ) != canonical_json_bytes(current_evidence):
            raise InvalidOwnershipTransition(
                "cleanup certificate cannot be cleared or changed"
            )
    if updated_at.tzinfo is None or updated_at.utcoffset() is None:
        raise ValueError("ownership timestamp must be timezone-aware")
    if updated_at.astimezone(timezone.utc) < current.updated_at:
        raise InvalidOwnershipTransition("ownership timestamp cannot move backward")

    return PromptOwnership(
        requested_prompt_id=current.requested_prompt_id,
        actual_prompt_id=next_actual,
        state=state,
        ownership_revision=current.ownership_revision + 1,
        cleanup_certificate=next_certificate,
        updated_at=updated_at,
    )


__all__ = [
    "AssemblySourceArtifactRef",
    "AudioMode",
    "Backend",
    "CompiledPlanDigest",
    "CompiledExecutionPlan",
    "ComfyNodeCacheIdentity",
    "ComfyNodeIdentity",
    "ControlContinuationDependency",
    "ContinuityLateBindingEvidence",
    "DocumentDigest",
    "DocumentDigestAlgorithm",
    "EndpointIdentity",
    "EndpointRestartCertificate",
    "ExactCancelConfirmedEvidence",
    "ExactPromptSnapshot",
    "ExpectedOutputGeometry",
    "ExpectedOutputSpec",
    "FeatureExecutionIdentity",
    "HistoricalTakeGeometryIdentity",
    "HistoryTerminalEvidence",
    "InvalidOwnershipTransition",
    "LegacyOutputLocator",
    "LateBindingEvidence",
    "LockedSegmentUnit",
    "LockedSubmissionPlan",
    "ObservedArtifactSpec",
    "ObservedAssemblyArtifactSpec",
    "OutputObservationReceipt",
    "OutputDescriptor",
    "OwnershipRevisionConflict",
    "PreparedControlUnit",
    "PreparedSegmentUnit",
    "PreviewSource",
    "PreviewSpec",
    "ProgressPhase",
    "ProgressSpec",
    "PromptOwnership",
    "PromptOwnershipState",
    "PromptReleaseEvidence",
    "RayRuntimeIdentity",
    "ResolvedImplementationIdentity",
    "RuntimePoolIdentityContribution",
    "RuntimeEpochLateBindingEvidence",
    "RuntimeRequirements",
    "SegmentExecutionIdentity",
    "can_transition_prompt_ownership",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_values_equal",
    "comfy_node_cache_identity",
    "compiled_execution_plan_digest",
    "compiled_execution_plan_digest_from_canonical_json",
    "digest_document",
    "derive_feature_execution_specs",
    "effective_execution_digest",
    "effective_prompt_id",
    "legacy_fnv1a32_document_digest",
    "locked_submission_plan_from_compiled",
    "ordered_compiled_segment_units",
    "sha256_document_digest",
    "transition_prompt_ownership",
]
