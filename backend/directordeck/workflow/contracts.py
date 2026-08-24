from __future__ import annotations

"""Pure, immutable contracts for the extensible workflow architecture.

This module deliberately has no dependency on the current compiler, database,
or ComfyUI transport.  Stage 1 consumers may construct and serialize these
contracts, but production prompt submission continues to use the legacy path.
"""

import hashlib
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from functools import lru_cache
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    get_args,
    runtime_checkable,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic_core import core_schema

from .canonical import MAX_SAFE_JSON_INTEGER, canonical_json_bytes


MAX_JSON_DEPTH = 12
MAX_JSON_CONTAINER_ITEMS = 256
MAX_JSON_STRING_LENGTH = 65_536
MAX_PUBLISHED_REF_DEPTH = 8
MAX_PUBLISHED_LIST_ITEMS = 128
MAX_PUBLISHED_RECORD_FIELDS = 128
MAX_REGISTRY_ITEMS = 4_096

_IDENTIFIER_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"
_RESOURCE_NAME_PATTERN = r"^[a-z][a-z0-9_.:-]{0,127}$"
_NODE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$"
_MODULE_PATTERN = r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,255}$"
_SEMVER_PATTERN = r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$"
_FINGERPRINT_PATTERN = r"^sha256:[0-9a-f]{64}$"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_RECORD_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_FILE_URI = re.compile(r"^file:", re.I)
_CREDENTIAL_URL = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@\s:]+:[^/@\s]+@", re.I)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.I),
)
_SENSITIVE_INPUT_NAME = re.compile(
    r"(?:^|[_-])(?:access[_-]?token|refresh[_-]?token|api[_-]?key|password|secret|authorization)(?:$|[_-])",
    re.I,
)
_FORBIDDEN_FEATURE_ASSET_KEYS = frozenset(
    {
        "asset",
        "asset_id",
        "asset_ids",
        "asset_reference",
        "asset_references",
        "asset_path",
        "comfy_path",
        "comfyui_path",
        "input_path",
        "media_path",
    }
)


class ContractModel(BaseModel):
    """Strict JSON contract base; instances cannot be reassigned or extended."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )


K = TypeVar("K")
V = TypeVar("V")


class FrozenMap(Mapping[K, V], Generic[K, V]):
    """An insertion-ordered immutable mapping with ordinary JSON encoding."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        self._data = dict(value)

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMap({self._data!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self) == dict(other)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> core_schema.CoreSchema:
        arguments = get_args(source_type)
        if len(arguments) != 2:
            raise TypeError("FrozenMap requires key and value type parameters")
        key_schema = handler.generate_schema(arguments[0])
        value_schema = handler.generate_schema(arguments[1])
        dictionary_schema = core_schema.dict_schema(key_schema, value_schema)
        return core_schema.no_info_after_validator_function(
            cls,
            dictionary_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: dict(value),
                return_schema=dictionary_schema,
            ),
        )


Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)]
ResourceName = Annotated[
    str,
    Field(min_length=1, max_length=128, pattern=_RESOURCE_NAME_PATTERN),
]
NodeId = Annotated[str, Field(min_length=1, max_length=128, pattern=_NODE_ID_PATTERN)]
ClassType = Annotated[str, Field(min_length=1, max_length=128)]
PortType = Annotated[str, Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)]
ModuleIdentity = Annotated[
    str,
    Field(min_length=1, max_length=256, pattern=_MODULE_PATTERN),
]
SemanticVersion = Annotated[str, Field(pattern=_SEMVER_PATTERN, max_length=128)]
RuntimeFingerprint = Annotated[str, Field(pattern=_FINGERPRINT_PATTERN)]
Sha256Digest = Annotated[str, Field(pattern=_DIGEST_PATTERN)]
PositiveVersion = Annotated[int, Field(ge=1, le=2_147_483_647)]
Revision = Annotated[int, Field(ge=1, le=9_007_199_254_740_991)]
GraphPhase = Literal[
    "bootstrap",
    "model_load",
    "model_prepare",
    "model_patch",
    "conditioning",
    "sampling",
    "decode",
    "postprocess",
    "persist",
]
Backend = Literal["standard", "raylight"]
ModelFamily = Literal["fl2va", "ref2va"]

GRAPH_PHASE_ORDER: tuple[GraphPhase, ...] = (
    "bootstrap",
    "model_load",
    "model_prepare",
    "model_patch",
    "conditioning",
    "sampling",
    "decode",
    "postprocess",
    "persist",
)
_GRAPH_PHASE_INDEX = {phase: index for index, phase in enumerate(GRAPH_PHASE_ORDER)}


type JsonScalar = None | bool | int | float | str
type JsonValue = (
    JsonScalar | tuple[JsonValue, ...] | FrozenMap[str, JsonValue]
)


def _normalize_json_input(
    value: Any,
    *,
    depth: int = 1,
    ancestors: set[int] | None = None,
) -> Any:
    """Convert ordinary JSON lists/maps to immutable-schema inputs."""

    if ancestors is None:
        ancestors = set()
    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"JSON value exceeds maximum depth {MAX_JSON_DEPTH}")
    if value is None or isinstance(value, (bool, int, float, str)):
        _validate_json_value(value, depth=depth)
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("JSON value cannot contain reference cycles")
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ValueError("JSON object exceeds maximum length")
        ancestors.add(identity)
        try:
            return {
                key: _normalize_json_input(
                    item,
                    depth=depth + 1,
                    ancestors=ancestors,
                )
                for key, item in value.items()
            }
        finally:
            ancestors.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in ancestors:
            raise ValueError("JSON value cannot contain reference cycles")
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ValueError("JSON array exceeds maximum length")
        ancestors.add(identity)
        try:
            return tuple(
                _normalize_json_input(
                    item,
                    depth=depth + 1,
                    ancestors=ancestors,
                )
                for item in value
            )
        finally:
            ancestors.remove(identity)
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


def _validate_json_value(value: JsonValue, *, depth: int = 1) -> JsonValue:
    if depth > MAX_JSON_DEPTH:
        raise ValueError(f"JSON value exceeds maximum depth {MAX_JSON_DEPTH}")
    if value is None or isinstance(value, (bool, int)):
        if isinstance(value, int) and not isinstance(value, bool):
            if abs(value) > MAX_SAFE_JSON_INTEGER:
                raise ValueError("JSON integer exceeds the interoperable safe range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        if value == 0 and math.copysign(1.0, value) < 0:
            raise ValueError("JSON numbers must not use negative zero")
        return value
    if isinstance(value, str):
        if len(value) > MAX_JSON_STRING_LENGTH:
            raise ValueError("JSON string exceeds maximum length")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("JSON string contains an unpaired UTF-16 surrogate")
        return value
    if isinstance(value, tuple):
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ValueError("JSON array exceeds maximum length")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return value
    if isinstance(value, FrozenMap):
        if len(value) > MAX_JSON_CONTAINER_ITEMS:
            raise ValueError("JSON object exceeds maximum length")
        for key, item in value.items():
            if (
                not key
                or len(key) > 256
                or any(
                    ord(character) < 0x20
                    or 0xD800 <= ord(character) <= 0xDFFF
                    for character in key
                )
            ):
                raise ValueError("JSON object key is empty, too long, or contains controls")
            _validate_json_value(item, depth=depth + 1)
        return value
    raise ValueError(f"unsupported JSON value: {type(value).__name__}")


BoundedJsonValue = Annotated[
    JsonValue,
    BeforeValidator(_normalize_json_input),
    AfterValidator(_validate_json_value),
]


def _validate_json_object(value: FrozenMap[str, JsonValue]) -> FrozenMap[str, JsonValue]:
    _validate_json_value(value)
    return value


def _normalize_json_object(value: Any) -> dict[str, Any]:
    normalized = _normalize_json_input(value)
    if not isinstance(normalized, dict):
        raise ValueError("value must be a JSON object")
    return normalized


JsonObject = Annotated[
    FrozenMap[str, JsonValue],
    BeforeValidator(_normalize_json_object),
    AfterValidator(_validate_json_object),
]


def canonical_sha256(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def _require_unique(values: tuple[Any, ...], label: str) -> None:
    encoded = [canonical_json_bytes(value) for value in values]
    if len(encoded) != len(set(encoded)):
        raise ValueError(f"{label} must contain unique values")


def _reject_feature_asset_fields(value: JsonValue) -> None:
    if isinstance(value, tuple):
        for item in value:
            _reject_feature_asset_fields(item)
        return
    if not isinstance(value, FrozenMap):
        return
    lowered = {key.lower() for key in value}
    if lowered & _FORBIDDEN_FEATURE_ASSET_KEYS:
        raise ValueError("feature parameters cannot own asset references or paths")
    if {"name", "subfolder", "type", "kind"} <= lowered:
        raise ValueError("feature parameters cannot contain an AssetReference shape")
    for item in value.values():
        _reject_feature_asset_fields(item)


class ResourceReadDeclaration(ContractModel):
    name: ResourceName
    type: PortType
    required: bool = True


class ResourceWriteDeclaration(ContractModel):
    name: ResourceName
    type: PortType
    operation: Literal["define", "replace"]
    required: bool = True


class FeatureTemplateEntry(ContractModel):
    id: Identifier
    version: PositiveVersion
    title: Annotated[str, Field(min_length=1, max_length=256)]
    description: Annotated[str, Field(min_length=1, max_length=4_096)]
    mode: Literal["switch", "needed"]
    layer: Literal["graph"] = "graph"
    graph_phase: GraphPhase
    reads: tuple[ResourceReadDeclaration, ...] = ()
    writes: tuple[ResourceWriteDeclaration, ...] = ()
    params_schema: JsonObject = Field(default_factory=dict)
    defaults: JsonObject = Field(default_factory=dict)
    cache_policy: JsonObject = Field(default_factory=dict)
    backends: tuple[Backend, ...]
    families: tuple[ModelFamily, ...]
    conflicts: tuple[Identifier, ...] = ()
    requires: tuple[Identifier, ...] = ()
    scopes: tuple[Identifier, ...]
    ui: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_entry(self) -> FeatureTemplateEntry:
        for label, values in (
            ("reads", tuple(item.name for item in self.reads)),
            ("writes", tuple(item.name for item in self.writes)),
            ("backends", self.backends),
            ("families", self.families),
            ("conflicts", self.conflicts),
            ("requires", self.requires),
            ("scopes", self.scopes),
        ):
            _require_unique(values, label)
        if not self.backends or not self.families or not self.scopes:
            raise ValueError("backends, families, and scopes must be non-empty")
        if self.id in self.conflicts or self.id in self.requires:
            raise ValueError("feature cannot conflict with or require itself")
        overlap = set(self.conflicts) & set(self.requires)
        if overlap:
            raise ValueError("conflicts and requires must be disjoint")
        _reject_feature_asset_fields(self.params_schema)
        _reject_feature_asset_fields(self.defaults)
        return self


class SegmentTemplate(ContractModel):
    id: Literal["h3_standard_segment", "h3_raylight_segment"]
    revision: PositiveVersion
    entries: tuple[FeatureTemplateEntry, ...]

    @property
    def backend(self) -> Backend:
        return "standard" if self.id == "h3_standard_segment" else "raylight"

    @model_validator(mode="after")
    def _validate_entries(self) -> SegmentTemplate:
        if not self.entries:
            raise ValueError("segment template must contain at least one entry")
        _require_unique(tuple(entry.id for entry in self.entries), "template entry ids")
        previous = -1
        for entry in self.entries:
            current = _GRAPH_PHASE_INDEX[entry.graph_phase]
            if current < previous:
                raise ValueError("feature graph phases must be monotonic")
            if self.backend not in entry.backends:
                raise ValueError(
                    f"feature {entry.id!r} does not support template backend {self.backend!r}"
                )
            previous = current
        return self


class ControlTemplate(ContractModel):
    id: Literal["raylight_kill_control"]
    revision: PositiveVersion


class SegmentTemplateSet(ContractModel):
    standard: SegmentTemplate
    raylight: SegmentTemplate

    @model_validator(mode="after")
    def _validate_identities(self) -> SegmentTemplateSet:
        if self.standard.id != "h3_standard_segment":
            raise ValueError("standard template has the wrong identity")
        if self.raylight.id != "h3_raylight_segment":
            raise ValueError("raylight template has the wrong identity")
        return self


class ControlTemplateSet(ContractModel):
    ray_kill: ControlTemplate


class TemplateBundle(ContractModel):
    version: PositiveVersion
    segment_templates: SegmentTemplateSet
    control_templates: ControlTemplateSet


class EdgeRef(ContractModel):
    kind: Literal["edge"] = "edge"
    node_id: NodeId
    output_slot: Annotated[int, Field(ge=0, le=255)]


class TerminalRef(ContractModel):
    kind: Literal["terminal"] = "terminal"
    node_id: NodeId


class ListRef(ContractModel):
    kind: Literal["list"] = "list"
    items: tuple[PublishedValueRef, ...]

    @model_validator(mode="after")
    def _validate_items(self) -> ListRef:
        if len(self.items) > MAX_PUBLISHED_LIST_ITEMS:
            raise ValueError("published list exceeds maximum length")
        if _published_ref_depth(self) > MAX_PUBLISHED_REF_DEPTH:
            raise ValueError("published value exceeds maximum recursion depth")
        return self


class RecordRef(ContractModel):
    kind: Literal["record"] = "record"
    fields: FrozenMap[str, PublishedValueRef]

    @model_validator(mode="after")
    def _validate_fields(self) -> RecordRef:
        if len(self.fields) > MAX_PUBLISHED_RECORD_FIELDS:
            raise ValueError("published record exceeds maximum length")
        for key in self.fields:
            if _RECORD_KEY_PATTERN.fullmatch(key) is None:
                raise ValueError(f"published record key is not JSON-safe: {key!r}")
        if _published_ref_depth(self) > MAX_PUBLISHED_REF_DEPTH:
            raise ValueError("published value exceeds maximum recursion depth")
        return self


PublishedValueRef = Annotated[
    EdgeRef | TerminalRef | ListRef | RecordRef,
    Field(discriminator="kind"),
]


def _published_ref_depth(value: PublishedValueRef) -> int:
    if isinstance(value, (EdgeRef, TerminalRef)):
        return 1
    children: Iterable[PublishedValueRef]
    if isinstance(value, ListRef):
        children = value.items
    else:
        children = value.fields.values()
    return 1 + max((_published_ref_depth(child) for child in children), default=0)


def _published_ref_node_ids(value: PublishedValueRef) -> tuple[str, ...]:
    if isinstance(value, (EdgeRef, TerminalRef)):
        return (value.node_id,)
    children = value.items if isinstance(value, ListRef) else value.fields.values()
    result: list[str] = []
    for child in children:
        for node_id in _published_ref_node_ids(child):
            if node_id not in result:
                result.append(node_id)
    return tuple(result)


def _published_ref_contains_terminal(value: PublishedValueRef) -> bool:
    if isinstance(value, TerminalRef):
        return True
    if isinstance(value, EdgeRef):
        return False
    children = value.items if isinstance(value, ListRef) else value.fields.values()
    return any(_published_ref_contains_terminal(child) for child in children)


class Resource(ContractModel):
    name: ResourceName
    type: PortType
    value: PublishedValueRef
    source_feature_id: Identifier
    revision: Revision
    producer_node_ids: tuple[NodeId, ...]

    @model_validator(mode="after")
    def _validate_producers(self) -> Resource:
        if not self.producer_node_ids:
            raise ValueError("resource must name at least one producer node")
        _require_unique(self.producer_node_ids, "producer_node_ids")
        if set(self.producer_node_ids) != set(_published_ref_node_ids(self.value)):
            raise ValueError("producer_node_ids must exactly match the published value")
        return self


class ResourcePool(ContractModel):
    schema_version: PositiveVersion = 1
    resources: FrozenMap[ResourceName, Resource] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_resources(self) -> ResourcePool:
        if len(self.resources) > MAX_REGISTRY_ITEMS:
            raise ValueError("resource pool exceeds maximum size")
        for name, resource in self.resources.items():
            if name != resource.name:
                raise ValueError("resource map key must match resource.name")
        return self

    def define(
        self,
        *,
        name: str,
        type: str,
        value: PublishedValueRef,
        source_feature_id: str,
        producer_node_ids: Iterable[str],
    ) -> ResourcePool:
        if name in self.resources:
            raise ValueError(f"resource {name!r} is already defined")
        resource = Resource(
            name=name,
            type=type,
            value=value,
            source_feature_id=source_feature_id,
            revision=1,
            producer_node_ids=tuple(producer_node_ids),
        )
        updated = dict(self.resources)
        updated[name] = resource
        return ResourcePool(schema_version=self.schema_version, resources=updated)

    def replace(
        self,
        *,
        name: str,
        value: PublishedValueRef,
        source_feature_id: str,
        producer_node_ids: Iterable[str],
        expected_type: str,
        expected_revision: int | None = None,
    ) -> ResourcePool:
        current = self.read_required(
            name,
            expected_type=expected_type,
            expected_revision=expected_revision,
            allow_terminal=False,
        )
        resource = Resource(
            name=name,
            type=current.type,
            value=value,
            source_feature_id=source_feature_id,
            revision=current.revision + 1,
            producer_node_ids=tuple(producer_node_ids),
        )
        updated = dict(self.resources)
        updated[name] = resource
        return ResourcePool(schema_version=self.schema_version, resources=updated)

    def read_required(
        self,
        name: str,
        *,
        expected_type: str,
        expected_revision: int | None = None,
        allow_terminal: bool = False,
    ) -> Resource:
        resource = self.resources.get(name)
        if resource is None:
            raise KeyError(f"required resource is missing: {name}")
        self._validate_read(
            resource,
            expected_type=expected_type,
            expected_revision=expected_revision,
            allow_terminal=allow_terminal,
        )
        return resource

    def read_optional(
        self,
        name: str,
        *,
        expected_type: str,
        expected_revision: int | None = None,
        allow_terminal: bool = False,
    ) -> Resource | None:
        resource = self.resources.get(name)
        if resource is None:
            return None
        self._validate_read(
            resource,
            expected_type=expected_type,
            expected_revision=expected_revision,
            allow_terminal=allow_terminal,
        )
        return resource

    @staticmethod
    def _validate_read(
        resource: Resource,
        *,
        expected_type: str,
        expected_revision: int | None,
        allow_terminal: bool,
    ) -> None:
        if resource.type != expected_type:
            raise TypeError(
                f"resource {resource.name!r} has type {resource.type!r}, "
                f"expected {expected_type!r}"
            )
        if expected_revision is not None and resource.revision != expected_revision:
            raise ValueError(
                f"resource {resource.name!r} has revision {resource.revision}, "
                f"expected {expected_revision}"
            )
        if not allow_terminal and _published_ref_contains_terminal(resource.value):
            raise TypeError("terminal references cannot be consumed as data edges")

    def begin(self) -> ResourcePoolTransaction:
        return ResourcePoolTransaction(base=self, staged=self)

    def snapshot_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_snapshot_json(cls, payload: str | bytes) -> ResourcePool:
        return cls.model_validate_json(payload)


class ResourcePoolTransaction(ContractModel):
    """Persistent transaction: each operation returns a new staged value."""

    base: ResourcePool
    staged: ResourcePool

    @model_validator(mode="after")
    def _validate_lineage(self) -> ResourcePoolTransaction:
        if self.base.schema_version != self.staged.schema_version:
            raise ValueError("transaction cannot change resource pool schema version")
        for name, original in self.base.resources.items():
            current = self.staged.resources.get(name)
            if current is None or current.revision < original.revision:
                raise ValueError("transaction cannot delete or rewind a base resource")
            if current.revision == original.revision and current != original:
                raise ValueError("same resource revision must preserve exact content")
            if current.revision > original.revision + 1:
                raise ValueError("transaction cannot skip a resource revision")
            if current.name != original.name or current.type != original.type:
                raise ValueError("transaction replace cannot change resource name or type")
        for name, current in self.staged.resources.items():
            if name not in self.base.resources and current.revision != 1:
                raise ValueError("transaction define must create revision 1")
        return self

    def define(
        self,
        *,
        name: str,
        type: str,
        value: PublishedValueRef,
        source_feature_id: str,
        producer_node_ids: Iterable[str],
    ) -> ResourcePoolTransaction:
        staged = self.staged.define(
            name=name,
            type=type,
            value=value,
            source_feature_id=source_feature_id,
            producer_node_ids=producer_node_ids,
        )
        return ResourcePoolTransaction(base=self.base, staged=staged)

    def replace(
        self,
        *,
        name: str,
        value: PublishedValueRef,
        source_feature_id: str,
        producer_node_ids: Iterable[str],
        expected_type: str,
        expected_revision: int | None = None,
    ) -> ResourcePoolTransaction:
        staged = self.staged.replace(
            name=name,
            value=value,
            source_feature_id=source_feature_id,
            producer_node_ids=producer_node_ids,
            expected_type=expected_type,
            expected_revision=expected_revision,
        )
        return ResourcePoolTransaction(base=self.base, staged=staged)

    def read_required(
        self,
        name: str,
        *,
        expected_type: str,
        expected_revision: int | None = None,
        allow_terminal: bool = False,
    ) -> Resource:
        return self.staged.read_required(
            name,
            expected_type=expected_type,
            expected_revision=expected_revision,
            allow_terminal=allow_terminal,
        )

    def read_optional(
        self,
        name: str,
        *,
        expected_type: str,
        expected_revision: int | None = None,
        allow_terminal: bool = False,
    ) -> Resource | None:
        return self.staged.read_optional(
            name,
            expected_type=expected_type,
            expected_revision=expected_revision,
            allow_terminal=allow_terminal,
        )

    def commit(self) -> ResourcePool:
        return self.staged

    def rollback(self) -> ResourcePool:
        return self.base


class ResolvedImplementationIdentity(ContractModel):
    role: Identifier
    class_type: ClassType
    implementation_id: Identifier
    semantic_version: SemanticVersion
    runtime_fingerprint: RuntimeFingerprint
    binding_key: Identifier


class ResolvedFeatureImplementation(ContractModel):
    """Bundle-6 adapter identity, independent from live host fingerprints."""

    implementation_id: Identifier
    implementation_version: PositiveVersion
    carrier_kind: Literal[
        "host_runtime",
        "comfy_node",
        "private_subgraph",
        "director_runtime",
    ]
    responsibility: Literal["director", "host_user"]
    class_types: tuple[ClassType, ...] = ()
    binding_key: Identifier | None = None

    @model_validator(mode="after")
    def unique_classes(self) -> "ResolvedFeatureImplementation":
        _require_unique(self.class_types, "implementation class types")
        return self


class FeatureResolution(ContractModel):
    state: Literal["active", "noop"]
    implementations: tuple[ResolvedImplementationIdentity, ...]
    resolution_details: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_resolution(self) -> FeatureResolution:
        _require_unique(
            tuple(item.binding_key for item in self.implementations),
            "implementation binding keys",
        )
        if self.state == "active" and not self.implementations:
            raise ValueError("active resolution requires an implementation")
        if self.state == "noop":
            if self.implementations:
                raise ValueError("noop resolution cannot claim implementations")
            reason = self.resolution_details.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("noop resolution must provide a non-empty reason")
        return self


class CapabilitySet(ContractModel):
    """Ordered, namespaced requirements declared by one feature resolution.

    The contract deliberately accepts future namespaces so a newer feature
    document can still be parsed by an older backend.  Availability is always
    decided by ``CapabilityEvaluator``; an unregistered namespace or member is
    therefore rejected with the same fail-closed reason as every other Stage-5
    consumer instead of being mistaken for an optional capability.
    """

    ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def _validate_ids(self) -> CapabilitySet:
        _require_unique(self.ids, "capability ids")
        if any("." not in capability_id for capability_id in self.ids):
            raise ValueError("capability ids must be namespaced")
        return self

    def in_namespace(self, namespace: str) -> tuple[str, ...]:
        if not namespace or "." in namespace:
            raise ValueError("capability namespace must be one identifier part")
        prefix = f"{namespace}."
        return tuple(
            capability_id
            for capability_id in self.ids
            if capability_id.startswith(prefix)
        )


class FeatureEmission(ContractModel):
    outputs: FrozenMap[ResourceName, PublishedValueRef] = Field(default_factory=dict)
    progress_hints: tuple[BoundedJsonValue, ...] = ()
    preview_hints: tuple[BoundedJsonValue, ...] = ()
    notices: tuple[Annotated[str, Field(min_length=1, max_length=4_096)], ...] = ()
    emission_details: JsonObject = Field(default_factory=dict)


@runtime_checkable
class CompileContext(Protocol):
    backend: Backend
    family: ModelFamily
    template_bundle_version: int


@runtime_checkable
class ScopedGraphBuilderProtocol(Protocol):
    @property
    def emitted_node_ids(self) -> tuple[str, ...]: ...

    @property
    def prompt_fragment(self) -> Mapping[str, Mapping[str, JsonValue]]: ...

    def add_node(self, class_type: str, inputs: Mapping[str, JsonValue]) -> str: ...

    def edge(self, node_id: str, output_slot: int) -> EdgeRef: ...

    def terminal(self, node_id: str) -> TerminalRef: ...


@runtime_checkable
class FeatureInterpreter(Protocol):
    id: str
    version: int

    def validate_params(self, params: BaseModel, ctx: CompileContext) -> None: ...

    def resolve(self, params: BaseModel, ctx: CompileContext) -> FeatureResolution: ...

    def required_capabilities(
        self,
        params: BaseModel,
        ctx: CompileContext,
        resolution: FeatureResolution,
    ) -> CapabilitySet: ...

    def cache_identity(
        self,
        params: BaseModel,
        ctx: CompileContext,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue: ...

    def runtime_pool_identity(
        self,
        params: BaseModel,
        ctx: CompileContext,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue | None: ...

    def emit(
        self,
        builder: ScopedGraphBuilderProtocol,
        inputs: Mapping[str, Resource],
        params: BaseModel,
        ctx: CompileContext,
        resolution: FeatureResolution,
    ) -> FeatureEmission: ...


class ObjectInfoInputContract(ContractModel):
    port_type: PortType
    enum_values: tuple[JsonScalar, ...] = ()
    has_director_default: bool = False
    director_default: JsonScalar = None

    @model_validator(mode="after")
    def _validate_default(self) -> ObjectInfoInputContract:
        _require_unique(self.enum_values, "object-info enum values")
        if not self.has_director_default and self.director_default is not None:
            raise ValueError("director_default requires has_director_default=true")
        if (
            self.has_director_default
            and self.enum_values
            and canonical_json_bytes(self.director_default)
            not in {canonical_json_bytes(item) for item in self.enum_values}
        ):
            raise ValueError("director_default is not present in enum_values")
        return self


class ObjectInfoOutputContract(ContractModel):
    index: Annotated[int, Field(ge=0, le=255)]
    port_type: PortType
    name: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    is_list: bool = False


class ObjectInfoContract(ContractModel):
    normalization_version: PositiveVersion
    required_inputs: FrozenMap[Identifier, ObjectInfoInputContract] = Field(
        default_factory=dict
    )
    optional_inputs: FrozenMap[Identifier, ObjectInfoInputContract] = Field(
        default_factory=dict
    )
    director_supplied_inputs: tuple[Identifier, ...] = ()
    outputs: tuple[ObjectInfoOutputContract, ...] = ()
    output_node: bool = False

    @model_validator(mode="after")
    def _validate_shape(self) -> ObjectInfoContract:
        overlap = set(self.required_inputs) & set(self.optional_inputs)
        if overlap:
            raise ValueError("required and optional inputs must be disjoint")
        _require_unique(self.director_supplied_inputs, "director supplied inputs")
        known_inputs = set(self.required_inputs) | set(self.optional_inputs)
        unknown = set(self.director_supplied_inputs) - known_inputs
        if unknown:
            raise ValueError("Director-supplied input is absent from object-info contract")
        if tuple(output.index for output in self.outputs) != tuple(range(len(self.outputs))):
            raise ValueError("object-info output indices must be contiguous from zero")
        return self


class NodeOutputContract(ContractModel):
    slots: tuple[ObjectInfoOutputContract, ...] = ()

    @model_validator(mode="after")
    def _validate_slots(self) -> NodeOutputContract:
        if tuple(slot.index for slot in self.slots) != tuple(range(len(self.slots))):
            raise ValueError("output contract indices must be contiguous from zero")
        return self


class RuntimeEffectContract(ContractModel):
    policy: Literal["strict_transform", "identity_allowed", "side_effect_only"]
    unsupported_behavior: Literal["raise", "identity", "fallback"]
    validation_method: Literal[
        "node_contract",
        "strict_wrapper",
        "director_owned_implementation",
        "user_assumed",
    ]
    verified_model_families: tuple[ModelFamily, ...] = ()
    verified_backends: tuple[Backend, ...] = ()
    notes: tuple[Annotated[str, Field(min_length=1, max_length=4_096)], ...] = ()

    @model_validator(mode="after")
    def _validate_policy(self) -> RuntimeEffectContract:
        _require_unique(self.verified_model_families, "verified model families")
        _require_unique(self.verified_backends, "verified backends")
        if self.policy == "strict_transform":
            if self.unsupported_behavior != "raise":
                raise ValueError("strict_transform must raise when unsupported")
            if not self.verified_model_families or not self.verified_backends:
                raise ValueError("strict_transform must name verified families and backends")
        if self.policy == "side_effect_only" and self.unsupported_behavior == "identity":
            raise ValueError("side_effect_only cannot report identity behavior")
        return self


ExecutionTerminalRole = Literal["take", "ray_kill"]
PersistentArtifactRole = Literal["take"]


class NodeContract(ContractModel):
    contract_id: Identifier
    semantic_version: SemanticVersion
    class_type: ClassType
    allowed_python_modules: tuple[ModuleIdentity, ...]
    object_info_contract: ObjectInfoContract
    output_contract: NodeOutputContract
    execution_terminal_role: ExecutionTerminalRole | None = None
    persistent_artifact_role: PersistentArtifactRole | None = None
    runtime_effect_contract: RuntimeEffectContract
    supported_runtime_fingerprints: tuple[RuntimeFingerprint, ...]

    @model_validator(mode="after")
    def _validate_contract(self) -> NodeContract:
        if not self.allowed_python_modules:
            raise ValueError("node contract must allow at least one normalized module")
        _require_unique(self.allowed_python_modules, "allowed python modules")
        if not self.supported_runtime_fingerprints:
            raise ValueError("node contract must support at least one runtime fingerprint")
        _require_unique(
            self.supported_runtime_fingerprints,
            "supported runtime fingerprints",
        )
        if self.object_info_contract.outputs != self.output_contract.slots:
            raise ValueError("object-info outputs and output contract must match exactly")
        if self.execution_terminal_role == "take":
            if self.persistent_artifact_role != "take":
                raise ValueError("take terminal must be the take persistent artifact")
        elif self.persistent_artifact_role is not None:
            raise ValueError("persistent take artifact requires a take execution terminal")
        if self.execution_terminal_role == "ray_kill":
            if self.persistent_artifact_role is not None:
                raise ValueError("RayKill cannot create a persistent artifact")
            if self.runtime_effect_contract.policy != "side_effect_only":
                raise ValueError("RayKill must use a side_effect_only runtime contract")
        return self


class NodeContractRegistry(ContractModel):
    schema_version: PositiveVersion = 1
    contracts: FrozenMap[ClassType, NodeContract] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_registry(self) -> NodeContractRegistry:
        if len(self.contracts) > MAX_REGISTRY_ITEMS:
            raise ValueError("node contract registry exceeds maximum size")
        for class_type, contract in self.contracts.items():
            if class_type != contract.class_type:
                raise ValueError("registry key must match NodeContract.class_type")
        return self

    def register(self, contract: NodeContract) -> NodeContractRegistry:
        if contract.class_type in self.contracts:
            raise ValueError(f"node contract already registered: {contract.class_type}")
        updated = dict(self.contracts)
        updated[contract.class_type] = contract
        return NodeContractRegistry(schema_version=self.schema_version, contracts=updated)

    def require(self, class_type: str) -> NodeContract:
        try:
            return self.contracts[class_type]
        except KeyError as exc:
            raise KeyError(f"unknown node contract: {class_type}") from exc

    def validate_implementation(
        self,
        implementation: ResolvedImplementationIdentity,
        *,
        output_affecting: bool,
        model_family: ModelFamily,
        backend: Backend,
    ) -> NodeContract:
        contract = self.require(implementation.class_type)
        if implementation.implementation_id != contract.contract_id:
            raise ValueError("implementation id does not match node contract")
        if implementation.semantic_version != contract.semantic_version:
            raise ValueError("implementation semantic version does not match node contract")
        if implementation.runtime_fingerprint not in contract.supported_runtime_fingerprints:
            raise ValueError("implementation adapter identity does not match node contract")
        effect = contract.runtime_effect_contract
        if (
            effect.verified_model_families
            and model_family not in effect.verified_model_families
        ):
            raise ValueError(
                f"implementation is not verified for model family {model_family!r}"
            )
        if effect.verified_backends and backend not in effect.verified_backends:
            raise ValueError(
                f"implementation is not verified for backend {backend!r}"
            )
        if output_affecting:
            director_verified = (
                effect.policy == "strict_transform"
                and effect.unsupported_behavior == "raise"
            )
            user_assumed = (
                effect.validation_method == "user_assumed"
            )
            if not (director_verified or user_assumed):
                raise ValueError(
                    "output-affecting implementation requires either a fail-closed "
                    "strict_transform or an explicit user-assumed interface contract"
                )
        return contract

    def validate_resolution(
        self,
        resolution: FeatureResolution,
        *,
        output_affecting: bool,
        model_family: ModelFamily,
        backend: Backend,
    ) -> tuple[NodeContract, ...]:
        if resolution.state == "noop":
            return ()
        return tuple(
            self.validate_implementation(
                item,
                output_affecting=output_affecting,
                model_family=model_family,
                backend=backend,
            )
            for item in resolution.implementations
        )


class NodeContractEvidence(ContractModel):
    contract_id: Identifier
    semantic_version: SemanticVersion
    class_type: ClassType
    python_module: ModuleIdentity
    runtime_fingerprint: RuntimeFingerprint
    execution_terminal_role: ExecutionTerminalRole | None = None
    persistent_artifact_role: PersistentArtifactRole | None = None


class DirectorAdapterContractEvidence(ContractModel):
    """Bundle-6 graph evidence for Director's adapter contract only.

    This deliberately contains no host module, source, package, or runtime
    fingerprint.  It identifies the immutable contract Director compiled
    against; ComfyUI remains responsible for accepting the emitted class type.
    """

    evidence_kind: Literal["director_adapter"] = "director_adapter"
    contract_id: Identifier
    semantic_version: SemanticVersion
    class_type: ClassType
    adapter_contract_digest: Sha256Digest
    execution_terminal_role: ExecutionTerminalRole | None = None
    persistent_artifact_role: PersistentArtifactRole | None = None


@lru_cache(maxsize=256)
def _director_adapter_contract_digest(serialized: str) -> Sha256Digest:
    contract = NodeContract.model_validate_json(serialized)
    return canonical_sha256(
        {
            "contract_id": contract.contract_id,
            "semantic_version": contract.semantic_version,
            "class_type": contract.class_type,
            "object_info_contract": contract.object_info_contract.model_dump(
                mode="json"
            ),
            "output_contract": contract.output_contract.model_dump(mode="json"),
            "execution_terminal_role": contract.execution_terminal_role,
            "persistent_artifact_role": contract.persistent_artifact_role,
            "runtime_effect_contract": contract.runtime_effect_contract.model_dump(
                mode="json"
            ),
        }
    )


def director_adapter_contract_digest(contract: NodeContract) -> Sha256Digest:
    """Hash immutable Director graph semantics once per distinct contract."""

    return _director_adapter_contract_digest(contract.model_dump_json())


GraphNodeContractEvidence = DirectorAdapterContractEvidence | NodeContractEvidence


class PublicResourceWrite(ContractModel):
    operation: Literal["define", "replace"]
    resource: Resource
    previous_revision: Revision | None = None

    @model_validator(mode="after")
    def _validate_revision(self) -> PublicResourceWrite:
        if self.operation == "define":
            if self.resource.revision != 1 or self.previous_revision is not None:
                raise ValueError("define must create revision 1 without a previous revision")
        else:
            if self.previous_revision is None:
                raise ValueError("replace must name its previous revision")
            if self.resource.revision != self.previous_revision + 1:
                raise ValueError("replace must increment the previous revision exactly once")
        return self


class PublicResourceRead(ContractModel):
    resource_name: ResourceName
    type: PortType
    revision: Revision
    consumer_node_id: NodeId
    input_pointer: Annotated[str, Field(min_length=2, max_length=512, pattern=r"^/.*")]
    value: PublishedValueRef

    @model_validator(mode="after")
    def _reject_terminal(self) -> PublicResourceRead:
        if _published_ref_contains_terminal(self.value):
            raise ValueError("public data read cannot consume a terminal reference")
        return self


class AllowedLateBoundInput(ContractModel):
    input_pointer: Annotated[str, Field(min_length=2, max_length=512, pattern=r"^/.*")]
    value_kind: Identifier
    source_kind: Literal["resource", "continuity", "runtime_epoch"]
    resource_name: ResourceName | None = None
    revision: Revision | None = None
    base_value_digest: Sha256Digest | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> AllowedLateBoundInput:
        paired = self.resource_name is not None and self.revision is not None
        empty = self.resource_name is None and self.revision is None
        if not (paired or empty):
            raise ValueError("late-bound resource name and revision must be paired")
        if self.source_kind == "resource" and not paired:
            raise ValueError("resource late binding requires resource name and revision")
        if self.source_kind != "resource" and not empty:
            raise ValueError("continuity/runtime epoch cannot claim a resource revision")
        return self


class GraphAuditSpec(ContractModel):
    version: PositiveVersion
    unit_kind: Literal["segment", "control"]
    control_kind: Literal["ray_kill"] | None = None
    take_node_id: NodeId | None = None
    node_contract_snapshot: FrozenMap[NodeId, GraphNodeContractEvidence]
    public_writes: tuple[PublicResourceWrite, ...] = ()
    public_reads: tuple[PublicResourceRead, ...] = ()
    allowed_late_bound_inputs: tuple[AllowedLateBoundInput, ...] = ()
    structural_influence_features: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def _validate_audit_shape(self) -> GraphAuditSpec:
        if not self.node_contract_snapshot:
            raise ValueError("graph audit must contain node contract evidence")
        for node_id, evidence in self.node_contract_snapshot.items():
            if node_id == "" or evidence.class_type == "":
                raise ValueError("node contract evidence is incomplete")
        terminal_nodes = {
            node_id: evidence
            for node_id, evidence in self.node_contract_snapshot.items()
            if evidence.execution_terminal_role is not None
        }
        persistent_nodes = {
            node_id: evidence
            for node_id, evidence in self.node_contract_snapshot.items()
            if evidence.persistent_artifact_role is not None
        }
        if self.unit_kind == "segment":
            if self.control_kind is not None or self.take_node_id is None:
                raise ValueError("segment audit requires take_node_id and no control_kind")
            evidence = terminal_nodes.get(self.take_node_id)
            if (
                len(terminal_nodes) != 1
                or evidence is None
                or evidence.execution_terminal_role != "take"
                or len(persistent_nodes) != 1
                or evidence.persistent_artifact_role != "take"
            ):
                raise ValueError("segment must have one matching take terminal/artifact")
        else:
            if self.control_kind != "ray_kill" or self.take_node_id is not None:
                raise ValueError("control audit requires ray_kill and no take_node_id")
            if (
                len(terminal_nodes) != 1
                or next(iter(terminal_nodes.values())).execution_terminal_role != "ray_kill"
                or persistent_nodes
            ):
                raise ValueError("control must have one RayKill terminal and no artifact")

        node_ids = set(self.node_contract_snapshot)
        writes_by_revision: dict[tuple[str, int], Resource] = {}
        for write in self.public_writes:
            if not set(write.resource.producer_node_ids) <= node_ids:
                raise ValueError("public write references a producer absent from node snapshot")
            key = (write.resource.name, write.resource.revision)
            if key in writes_by_revision:
                raise ValueError("public resource revision is written more than once")
            writes_by_revision[key] = write.resource
        for read in self.public_reads:
            if read.consumer_node_id not in node_ids:
                raise ValueError("public read consumer is absent from node snapshot")
            resource = writes_by_revision.get((read.resource_name, read.revision))
            if resource is None:
                raise ValueError("public read does not match a declared resource revision")
            if resource.type != read.type or resource.value != read.value:
                raise ValueError("public read must use the exact declared type and value")
        for late_bound in self.allowed_late_bound_inputs:
            if (
                late_bound.source_kind == "resource"
                and (late_bound.resource_name, late_bound.revision)
                not in writes_by_revision
            ):
                raise ValueError(
                    "resource late binding must match a declared public write revision"
                )
        _require_unique(
            tuple(item.input_pointer for item in self.allowed_late_bound_inputs),
            "late-bound input pointers",
        )
        _require_unique(self.structural_influence_features, "structural feature ids")
        writers = {write.resource.source_feature_id for write in self.public_writes}
        if not set(self.structural_influence_features) <= writers:
            raise ValueError("structural influence feature must own a public write")
        return self


class PackageCapability(ContractModel):
    importable: bool
    version: Annotated[str, Field(min_length=1, max_length=128)] | None = None


class LogicalGpuCapability(ContractModel):
    logical_index: Annotated[int, Field(ge=0, le=255)]
    backend: Literal["cuda", "xpu", "mps"]
    total_memory_mb: Annotated[int, Field(ge=1, le=16_777_216)] | None = None


class RayLightInstallation(ContractModel):
    installed: bool
    package_version: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    node_contracts_available: bool = False
    reason_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def _validate_installation(self) -> RayLightInstallation:
        _require_unique(self.reason_codes, "RayLight reason codes")
        if not self.installed and (self.package_version is not None or self.node_contracts_available):
            raise ValueError("uninstalled RayLight cannot expose package/contracts")
        return self


class MediaToolCapability(ContractModel):
    available: bool
    version: Annotated[str, Field(min_length=1, max_length=128)] | None = None


class RuntimeProbeEvidence(ContractModel):
    """One exact, privacy-safe in-process runtime capability observation."""

    available: bool
    code: Identifier
    architecture: Identifier | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> RuntimeProbeEvidence:
        if self.available != (self.code == "available"):
            raise ValueError(
                "runtime probe available flag must agree with its stable code"
            )
        return self


def _iter_nested_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_nested_strings(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_nested_strings(item)


class HostCapabilitySnapshot(ContractModel):
    schema_version: PositiveVersion
    generated_at: datetime
    node_registry: FrozenMap[ClassType, ModuleIdentity]
    object_info_slices: FrozenMap[ClassType, ObjectInfoContract]
    module_fingerprints: FrozenMap[ModuleIdentity, RuntimeFingerprint]
    importable_packages: FrozenMap[Identifier, PackageCapability]
    gpu_inventory: tuple[LogicalGpuCapability, ...]
    raylight_installation: RayLightInstallation
    media_tool_status: FrozenMap[Identifier, MediaToolCapability]
    runtime_probe_evidence: FrozenMap[Identifier, RuntimeProbeEvidence] = Field(
        default_factory=dict
    )

    _STATIC_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_version",
            "node_registry",
            "object_info_slices",
            "module_fingerprints",
            "importable_packages",
            "gpu_inventory",
            "raylight_installation",
            "media_tool_status",
            "runtime_probe_evidence",
        }
    )

    @model_validator(mode="after")
    def _validate_snapshot(self) -> HostCapabilitySnapshot:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        if len(self.node_registry) > MAX_REGISTRY_ITEMS:
            raise ValueError("host node registry exceeds maximum size")
        if len(self.runtime_probe_evidence) > MAX_REGISTRY_ITEMS:
            raise ValueError("host runtime probe evidence exceeds maximum size")
        if self.schema_version == 1 and self.runtime_probe_evidence:
            raise ValueError(
                "host capability schema 1 cannot contain runtime probe evidence"
            )
        if not set(self.object_info_slices) <= set(self.node_registry):
            raise ValueError("object-info slice references an absent node")
        # Module fingerprints are advisory observations and may be sparse or
        # temporarily stale relative to the live class registry.  They must not
        # make an otherwise usable host snapshot invalid.
        indices = tuple(item.logical_index for item in self.gpu_inventory)
        if indices != tuple(range(len(indices))):
            raise ValueError("logical GPU inventory must be contiguous and ordered")
        payload = self.model_dump(mode="json")
        for value in _iter_nested_strings(payload):
            if (
                value.startswith(("/", "\\"))
                or _ABSOLUTE_WINDOWS_PATH.match(value)
                or _FILE_URI.match(value)
            ):
                raise ValueError("host capability snapshot cannot contain absolute paths")
            if _CREDENTIAL_URL.match(value):
                raise ValueError("host capability snapshot cannot contain URL credentials")
            if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
                raise ValueError("host capability snapshot cannot contain credentials")
        for object_info in self.object_info_slices.values():
            for name, input_contract in (
                *object_info.required_inputs.items(),
                *object_info.optional_inputs.items(),
            ):
                if (
                    _SENSITIVE_INPUT_NAME.search(name)
                    and input_contract.has_director_default
                    and input_contract.director_default not in (None, "")
                ):
                    raise ValueError(
                        "host capability snapshot cannot contain sensitive defaults"
                    )
        return self

    def host_capability_revision(self) -> str:
        payload = self.model_dump(mode="json", include=self._STATIC_FIELDS)
        return canonical_sha256(payload)


class OperationalReadiness(ContractModel):
    endpoint_online: bool
    submission_allowed: bool
    ray_recovery_required: bool
    ray_tainted: bool
    invalid_runtime_gpu_indices: tuple[
        Annotated[int, Field(ge=0, le=255)], ...
    ] = ()
    blocking_reason_codes: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def _validate_readiness(self) -> OperationalReadiness:
        _require_unique(
            self.invalid_runtime_gpu_indices,
            "invalid runtime GPU indices",
        )
        _require_unique(self.blocking_reason_codes, "blocking reason codes")
        intrinsically_blocked = (
            not self.endpoint_online
            or self.ray_recovery_required
            or bool(self.invalid_runtime_gpu_indices)
        )
        if self.submission_allowed and intrinsically_blocked:
            raise ValueError("submission cannot be allowed while runtime is blocked")
        if self.submission_allowed and self.blocking_reason_codes:
            raise ValueError("allowed submission cannot have blocking reasons")
        if not self.submission_allowed and not self.blocking_reason_codes:
            raise ValueError("blocked submission must provide a reason code")
        return self


@runtime_checkable
class HostCapabilityProvider(Protocol):
    def snapshot(self) -> HostCapabilitySnapshot: ...


ListRef.model_rebuild(_types_namespace=globals())
RecordRef.model_rebuild(_types_namespace=globals())
