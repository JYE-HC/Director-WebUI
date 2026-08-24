from __future__ import annotations

"""Pure construction and validation for :class:`GraphAuditSpec`.

The compiler supplies immutable evidence collected while feature scopes are
committed.  This module never calls an interpreter and never mutates a prompt
or resource pool.  The same validator can therefore be run both immediately
after compilation and again after the exact prompt has been materialized.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from .contracts import (
    AllowedLateBoundInput,
    Backend,
    canonical_sha256,
    ContractModel,
    DirectorAdapterContractEvidence,
    EdgeRef,
    FeatureResolution,
    GraphAuditSpec,
    GraphNodeContractEvidence,
    Identifier,
    ListRef,
    ModelFamily,
    NodeContractRegistry,
    NodeId,
    ObjectInfoInputContract,
    PublicResourceRead,
    PublicResourceWrite,
    PublishedValueRef,
    RecordRef,
    TerminalRef,
    director_adapter_contract_digest,
)


class GraphAuditError(ValueError):
    """The prompt and its immutable compilation evidence disagree."""


class ResolvedNodeEmission(ContractModel):
    """Bind one emitted node to the resolution implementation that emitted it.

    A binding may own more than one node of the same class (for example two VAE
    loader instances), but every emitted node must be bound and every resolved
    implementation must be observed at least once.
    """

    node_id: NodeId
    implementation_binding_key: Identifier
    output_affecting: bool = False


class FeatureAuditTrace(ContractModel):
    """Ordered, immutable evidence for one active graph feature invocation."""

    feature_id: Identifier
    resolution: FeatureResolution
    emitted_nodes: Annotated[tuple[ResolvedNodeEmission, ...], Field(min_length=1)]
    structural_influence: bool = False

    @model_validator(mode="after")
    def _validate_trace(self) -> FeatureAuditTrace:
        if self.resolution.state != "active":
            raise ValueError("graph audit traces may contain only active features")
        node_ids = tuple(item.node_id for item in self.emitted_nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("feature audit emitted node ids must be unique")
        known_bindings = {
            item.binding_key for item in self.resolution.implementations
        }
        observed_bindings = {
            item.implementation_binding_key for item in self.emitted_nodes
        }
        if observed_bindings != known_bindings:
            raise ValueError(
                "feature audit node bindings must exactly cover the resolution"
            )
        effects_by_binding: dict[str, bool] = {}
        for emitted in self.emitted_nodes:
            previous = effects_by_binding.setdefault(
                emitted.implementation_binding_key,
                emitted.output_affecting,
            )
            if previous != emitted.output_affecting:
                raise ValueError(
                    "one resolution binding cannot mix output-affecting roles"
                )
        return self


class PromptInputEdge(ContractModel):
    """One typed Comfy edge discovered at an exact RFC 6901 input pointer."""

    source_node_id: NodeId
    output_slot: Annotated[int, Field(ge=0, le=255)]
    consumer_node_id: NodeId
    input_pointer: Annotated[str, Field(min_length=2, max_length=512, pattern=r"^/.*")]


_LateBoundKind = Literal[
    "edge",
    "string",
    "integer",
    "number",
    "boolean",
    "list",
    "record",
    "json",
]
_SUPPORTED_LATE_BOUND_KINDS: frozenset[str] = frozenset(
    ("edge", "string", "integer", "number", "boolean", "list", "record", "json")
)


def _escape_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _decode_pointer(pointer: str) -> tuple[str, ...]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise GraphAuditError("JSON pointer must start with '/'")
    result: list[str] = []
    for raw_token in pointer[1:].split("/"):
        token: list[str] = []
        index = 0
        while index < len(raw_token):
            character = raw_token[index]
            if character != "~":
                token.append(character)
                index += 1
                continue
            if index + 1 >= len(raw_token) or raw_token[index + 1] not in "01":
                raise GraphAuditError(f"invalid JSON pointer escape in {pointer!r}")
            token.append("~" if raw_token[index + 1] == "0" else "/")
            index += 2
        result.append("".join(token))
    return tuple(result)


def _resolve_pointer(document: Any, pointer: str) -> Any:
    current = document
    for token in _decode_pointer(pointer):
        if isinstance(current, Mapping):
            if token not in current:
                raise GraphAuditError(f"JSON pointer does not exist: {pointer}")
            current = current[token]
            continue
        if isinstance(current, (list, tuple)):
            if token == "0":
                index = 0
            elif token.isdigit() and not token.startswith("0"):
                index = int(token)
            else:
                raise GraphAuditError(f"JSON pointer has invalid array index: {pointer}")
            if index >= len(current):
                raise GraphAuditError(f"JSON pointer does not exist: {pointer}")
            current = current[index]
            continue
        raise GraphAuditError(f"JSON pointer traverses a scalar: {pointer}")
    return current


def _validate_prompt_shape(prompt: Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(prompt, Mapping) or not prompt:
        raise GraphAuditError("prompt must be a non-empty mapping")
    node_ids: list[str] = []
    for node_id, node in prompt.items():
        if not isinstance(node_id, str) or not node_id:
            raise GraphAuditError("prompt node ids must be non-empty strings")
        if not isinstance(node, Mapping):
            raise GraphAuditError(f"prompt node {node_id!r} must be a mapping")
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(class_type, str) or not class_type:
            raise GraphAuditError(
                f"prompt node {node_id!r} must have a non-empty class_type"
            )
        if not isinstance(inputs, Mapping):
            raise GraphAuditError(f"prompt node {node_id!r} inputs must be a mapping")
        node_ids.append(node_id)
    return tuple(node_ids)


def collect_prompt_input_edges(prompt: Mapping[str, Any]) -> tuple[PromptInputEdge, ...]:
    """Discover every Comfy data edge beneath a prompt node's ``inputs`` map."""

    node_ids = set(_validate_prompt_shape(prompt))
    result: list[PromptInputEdge] = []

    def walk(value: Any, *, consumer: str, tokens: tuple[str, ...]) -> None:
        if isinstance(value, (list, tuple)):
            if (
                len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
                and not isinstance(value[1], bool)
            ):
                if value[0] not in node_ids:
                    raise GraphAuditError(
                        f"edge at /{'/'.join(tokens)} references unknown node {value[0]!r}"
                    )
                try:
                    result.append(
                        PromptInputEdge(
                            source_node_id=value[0],
                            output_slot=value[1],
                            consumer_node_id=consumer,
                            input_pointer="/" + "/".join(
                                _escape_pointer_token(token) for token in tokens
                            ),
                        )
                    )
                except ValueError as exc:
                    raise GraphAuditError(
                        f"edge at /{'/'.join(tokens)} has an invalid output slot"
                    ) from exc
                return
            for index, item in enumerate(value):
                walk(item, consumer=consumer, tokens=(*tokens, str(index)))
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise GraphAuditError("prompt input object keys must be strings")
                walk(item, consumer=consumer, tokens=(*tokens, key))

    for node_id, node in prompt.items():
        for input_name, value in node["inputs"].items():
            if not isinstance(input_name, str):
                raise GraphAuditError("prompt input names must be strings")
            walk(value, consumer=node_id, tokens=(node_id, "inputs", input_name))
    return tuple(result)


def _plain_published_ref(value: PublishedValueRef) -> Any:
    if isinstance(value, EdgeRef):
        return [value.node_id, value.output_slot]
    if isinstance(value, TerminalRef):
        return {"kind": "terminal", "node_id": value.node_id}
    if isinstance(value, ListRef):
        return [_plain_published_ref(item) for item in value.items]
    assert isinstance(value, RecordRef)
    return {key: _plain_published_ref(item) for key, item in value.fields.items()}


def _leaf_refs(value: PublishedValueRef) -> tuple[EdgeRef | TerminalRef, ...]:
    if isinstance(value, (EdgeRef, TerminalRef)):
        return (value,)
    children = value.items if isinstance(value, ListRef) else value.fields.values()
    return tuple(ref for child in children for ref in _leaf_refs(child))


def _same_exact_value(left: Any, right: Any) -> bool:
    # JSON has one array kind. Contract models freeze arrays as tuples while
    # freshly compiled prompts use lists, so representation must not invent a
    # semantic change between those two in-memory containers.
    if not (
        isinstance(left, (list, tuple))
        and isinstance(right, (list, tuple))
    ) and type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if tuple(left) != tuple(right):
            return False
        return all(_same_exact_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_exact_value(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float) and math.isnan(left):
        return False
    return left == right


def _validate_object_info_literal(
    *,
    node_id: str,
    input_name: str,
    value: Any,
    input_contract: ObjectInfoInputContract,
    check_enum: bool = True,
    require_literal_type: bool = True,
) -> None:
    """Validate a materialized scalar against the frozen ObjectInfo slice.

    Prepared prompts may carry a non-runtime placeholder at an explicitly
    whitelisted late-bound pointer.  Once that pointer is materialized, enum
    membership must be checked just like an ordinary Director-supplied input;
    ``value_kind=string`` alone is not enough for a COMBO port.
    """

    if check_enum and input_contract.enum_values and not any(
        _same_exact_value(value, candidate)
        for candidate in input_contract.enum_values
    ):
        raise GraphAuditError(
            f"node {node_id!r} input {input_name!r} is outside its enum"
        )
    if not require_literal_type:
        return
    port_type = input_contract.port_type
    valid_literal = (
        (
            (port_type == "STRING" or port_type.endswith("COMBO"))
            and isinstance(value, str)
        )
        or (
            port_type == "INT"
            and isinstance(value, int)
            and not isinstance(value, bool)
        )
        or (
            port_type == "FLOAT"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
        or (port_type == "BOOLEAN" and isinstance(value, bool))
    )
    if not valid_literal:
        raise GraphAuditError(
            f"node {node_id!r} input {input_name!r} does not match "
            f"port type {port_type!r}"
        )


def _validate_node_contracts(
    *,
    prompt: Mapping[str, Any],
    spec: GraphAuditSpec,
    registry: NodeContractRegistry,
) -> None:
    prompt_node_ids = set(prompt)
    if set(spec.node_contract_snapshot) != prompt_node_ids:
        raise GraphAuditError(
            "node contract snapshot must exactly cover the prompt node set"
        )
    for node_id, evidence in spec.node_contract_snapshot.items():
        class_type = prompt[node_id]["class_type"]
        if class_type != evidence.class_type:
            raise GraphAuditError(
                f"node {node_id!r} class_type differs from its contract evidence"
            )
        try:
            contract = registry.require(class_type)
        except KeyError as exc:
            raise GraphAuditError(f"node {node_id!r} has no registered contract") from exc
        if evidence.contract_id != contract.contract_id:
            raise GraphAuditError(f"node {node_id!r} contract id does not match registry")
        if evidence.semantic_version != contract.semantic_version:
            raise GraphAuditError(
                f"node {node_id!r} contract version does not match registry"
            )
        if isinstance(evidence, DirectorAdapterContractEvidence):
            if evidence.adapter_contract_digest != director_adapter_contract_digest(
                contract
            ):
                raise GraphAuditError(
                    f"node {node_id!r} Director adapter contract digest differs "
                    "from its graph contract"
                )
        else:
            if evidence.python_module not in contract.allowed_python_modules:
                raise GraphAuditError(
                    f"node {node_id!r} compiler module differs from its graph contract"
                )
            if (
                evidence.runtime_fingerprint
                not in contract.supported_runtime_fingerprints
            ):
                raise GraphAuditError(
                    f"node {node_id!r} compiler adapter identity differs from "
                    "its graph contract"
                )
        if evidence.execution_terminal_role != contract.execution_terminal_role:
            raise GraphAuditError(
                f"node {node_id!r} execution terminal role differs from registry"
            )
        if evidence.persistent_artifact_role != contract.persistent_artifact_role:
            raise GraphAuditError(
                f"node {node_id!r} persistent artifact role differs from registry"
            )
        inputs = prompt[node_id]["inputs"]
        object_info = contract.object_info_contract
        required = set(object_info.required_inputs)
        optional = set(object_info.optional_inputs)
        supplied = set(object_info.director_supplied_inputs)
        actual_names = set(inputs)
        missing = required - actual_names
        if missing:
            raise GraphAuditError(
                f"node {node_id!r} is missing required inputs: "
                + ", ".join(sorted(missing))
            )
        unknown = actual_names - required - optional
        if unknown:
            raise GraphAuditError(
                f"node {node_id!r} has undeclared inputs: "
                + ", ".join(sorted(unknown))
            )
        unsupplied = actual_names - supplied
        if unsupplied:
            raise GraphAuditError(
                f"node {node_id!r} uses inputs Director is not allowed to supply: "
                + ", ".join(sorted(unsupplied))
            )
        input_contracts = {
            **object_info.required_inputs,
            **object_info.optional_inputs,
        }
        late_bound_pointers = {
            item.input_pointer for item in spec.allowed_late_bound_inputs
        }
        composite_reads = {
            item.input_pointer: item
            for item in spec.public_reads
            if isinstance(item.value, (ListRef, RecordRef))
        }
        for input_name, value in inputs.items():
            input_contract = input_contracts[input_name]
            input_pointer = (
                f"/{_escape_pointer_token(node_id)}/inputs/"
                f"{_escape_pointer_token(input_name)}"
            )
            if _is_prompt_edge_value(value):
                continue
            if input_pointer in late_bound_pointers:
                continue
            composite_read = composite_reads.get(input_pointer)
            if composite_read is not None:
                if not _same_exact_value(
                    value,
                    _plain_published_ref(composite_read.value),
                ):
                    raise GraphAuditError(
                        f"node {node_id!r} input {input_name!r} does not match "
                        "its exact composite public resource"
                    )
                continue
            _validate_object_info_literal(
                node_id=node_id,
                input_name=input_name,
                value=value,
                input_contract=input_contract,
            )
        if (
            spec.unit_kind == "segment"
            and class_type == "DirectorDeckRayInitializerAdvanced"
            and inputs.get("XFuser_attention") != "COMFY_KITCHEN_INT8"
            # The frozen v4 registry predates the versioned Ray attention
            # semantic.  CURRENT explicitly supports the bundle-5 enum and
            # its interpreter/topology guard, including TORCH_FLASH.
            and "DirectorStrictModelAttentionBackend" not in registry.contracts
            # Bundle 6 owns the selected attention through execution_strategy;
            # CK off intentionally compiles the bundled TORCH_FLASH baseline.
            and "ModelAttentionBackend" not in registry.contracts
        ):
            raise GraphAuditError(
                "RayLight segment graphs require COMFY_KITCHEN_INT8 attention; "
                "legacy TORCH_FLASH is allowed only in a DirectorDeckRayKill control graph"
            )

    composite_read_paths = tuple(
        _decode_pointer(read.input_pointer)
        for read in spec.public_reads
        if isinstance(read.value, (ListRef, RecordRef))
    )
    for edge in collect_prompt_input_edges(prompt):
        source_contract = registry.require(prompt[edge.source_node_id]["class_type"])
        slots = {
            slot.index: slot for slot in source_contract.output_contract.slots
        }
        if edge.output_slot not in slots:
            raise GraphAuditError(
                f"edge at {edge.input_pointer} uses undeclared output slot "
                f"{edge.output_slot} of node {edge.source_node_id!r}"
            )
        tokens = _decode_pointer(edge.input_pointer)
        input_name = tokens[2]
        consumer_contract = registry.require(
            prompt[edge.consumer_node_id]["class_type"]
        ).object_info_contract
        input_contract = (
            consumer_contract.required_inputs.get(input_name)
            or consumer_contract.optional_inputs.get(input_name)
        )
        if input_contract is None:
            raise GraphAuditError(
                f"edge at {edge.input_pointer} targets an undeclared input"
            )
        edge_path = _decode_pointer(edge.input_pointer)
        nested_in_composite_resource = any(
            len(edge_path) > len(read_path)
            and edge_path[: len(read_path)] == read_path
            for read_path in composite_read_paths
        )
        if (
            not nested_in_composite_resource
            and slots[edge.output_slot].port_type != input_contract.port_type
        ):
            raise GraphAuditError(
                f"edge at {edge.input_pointer} has port type "
                f"{slots[edge.output_slot].port_type!r}, expected "
                f"{input_contract.port_type!r}"
            )


def _validate_feature_traces(
    *,
    prompt: Mapping[str, Any],
    spec: GraphAuditSpec,
    traces: Sequence[FeatureAuditTrace],
    registry: NodeContractRegistry,
    model_family: ModelFamily,
    backend: Backend,
    enforce_runtime_effects: bool,
) -> dict[str, str]:
    feature_ids = [trace.feature_id for trace in traces]
    if len(feature_ids) != len(set(feature_ids)):
        raise GraphAuditError("feature audit traces must have unique feature ids")
    owner_by_node: dict[str, str] = {}
    for trace in traces:
        implementations = {
            item.binding_key: item for item in trace.resolution.implementations
        }
        for emitted in trace.emitted_nodes:
            if emitted.node_id in owner_by_node:
                raise GraphAuditError(
                    f"node {emitted.node_id!r} is claimed by more than one feature"
                )
            node = prompt.get(emitted.node_id)
            if node is None:
                raise GraphAuditError(
                    f"feature {trace.feature_id!r} claims a node absent from prompt"
                )
            implementation = implementations[emitted.implementation_binding_key]
            if node["class_type"] != implementation.class_type:
                raise GraphAuditError(
                    f"feature {trace.feature_id!r} emitted class "
                    f"{node['class_type']!r} for resolution binding "
                    f"{emitted.implementation_binding_key!r}, expected "
                    f"{implementation.class_type!r}"
                )
            evidence = spec.node_contract_snapshot.get(emitted.node_id)
            if evidence is None or evidence.class_type != implementation.class_type:
                raise GraphAuditError(
                    f"feature {trace.feature_id!r} resolution and provenance disagree"
                )
            if (
                not isinstance(evidence, DirectorAdapterContractEvidence)
                and evidence.runtime_fingerprint != implementation.runtime_fingerprint
            ):
                raise GraphAuditError(
                    f"feature {trace.feature_id!r} resolution fingerprint and "
                    "emission evidence disagree"
                )
            try:
                registry.validate_implementation(
                    implementation,
                    output_affecting=(
                        enforce_runtime_effects and emitted.output_affecting
                    ),
                    model_family=model_family,
                    backend=backend,
                )
            except (KeyError, ValueError) as exc:
                raise GraphAuditError(
                    f"resolution binding {emitted.implementation_binding_key!r} "
                    f"does not satisfy its node contract: {exc}"
                ) from exc
            owner_by_node[emitted.node_id] = trace.feature_id
    if set(owner_by_node) != set(prompt):
        missing = sorted(set(prompt) - set(owner_by_node))
        raise GraphAuditError(
            "feature audit traces do not cover the prompt node set: " + ", ".join(missing)
        )

    expected_structural = tuple(
        trace.feature_id for trace in traces if trace.structural_influence
    )
    if spec.structural_influence_features != expected_structural:
        raise GraphAuditError(
            "structural influence features differ from ordered feature traces"
        )
    return owner_by_node


def _validate_read_pointer(
    *,
    prompt: Mapping[str, Any],
    read: PublicResourceRead,
    allowed_resource_pointers: frozenset[str],
) -> None:
    tokens = _decode_pointer(read.input_pointer)
    if len(tokens) < 3 or tokens[0] != read.consumer_node_id or tokens[1] != "inputs":
        raise GraphAuditError(
            f"public read pointer {read.input_pointer!r} does not identify its consumer input"
        )
    actual = _resolve_pointer(prompt, read.input_pointer)
    if read.input_pointer not in allowed_resource_pointers:
        expected = _plain_published_ref(read.value)
        if not _same_exact_value(actual, expected):
            raise GraphAuditError(
                f"public read pointer {read.input_pointer!r} does not contain its exact edge"
            )


def _validate_resource_history(
    *,
    prompt: Mapping[str, Any],
    spec: GraphAuditSpec,
    traces: Sequence[FeatureAuditTrace],
    owner_by_node: Mapping[str, str],
) -> None:
    trace_index = {trace.feature_id: index for index, trace in enumerate(traces)}
    writes_by_feature: dict[str, list[PublicResourceWrite]] = {
        trace.feature_id: [] for trace in traces
    }
    for write in spec.public_writes:
        feature_id = write.resource.source_feature_id
        if feature_id not in writes_by_feature:
            raise GraphAuditError(
                f"public write source feature {feature_id!r} has no audit trace"
            )
        for producer in write.resource.producer_node_ids:
            if owner_by_node.get(producer) != feature_id:
                raise GraphAuditError(
                    f"resource {write.resource.name!r} producer is not owned by its feature"
                )
        leaf_refs = _leaf_refs(write.resource.value)
        if any(isinstance(ref, TerminalRef) for ref in leaf_refs) and not isinstance(
            write.resource.value, TerminalRef
        ):
            raise GraphAuditError(
                "a terminal resource must be one direct TerminalRef, not a composite"
            )
        for ref in leaf_refs:
            evidence = spec.node_contract_snapshot[ref.node_id]
            if isinstance(ref, EdgeRef):
                # Output slots used only as public values still need validation;
                # they may not appear in a downstream prompt edge yet.
                # Registry slot validation happens in the caller below.
                if evidence.execution_terminal_role is not None:
                    raise GraphAuditError("execution terminals cannot publish data edges")
            elif evidence.execution_terminal_role is None:
                raise GraphAuditError("TerminalRef must name an execution terminal")
        writes_by_feature[feature_id].append(write)

    reads_by_feature: dict[str, list[PublicResourceRead]] = {
        trace.feature_id: [] for trace in traces
    }
    allowed_resource_pointers = frozenset(
        item.input_pointer
        for item in spec.allowed_late_bound_inputs
        if item.source_kind == "resource"
    )
    read_pointers: list[tuple[str, ...]] = []
    for read in spec.public_reads:
        owner = owner_by_node.get(read.consumer_node_id)
        if owner is None:
            raise GraphAuditError("public read consumer has no feature owner")
        _validate_read_pointer(
            prompt=prompt,
            read=read,
            allowed_resource_pointers=allowed_resource_pointers,
        )
        tokens = _decode_pointer(read.input_pointer)
        if any(
            tokens[: len(existing)] == existing
            or existing[: len(tokens)] == tokens
            for existing in read_pointers
        ):
            raise GraphAuditError("public read pointers must not overlap")
        read_pointers.append(tokens)
        reads_by_feature[owner].append(read)

    # Public writes are a compile-order ledger, not an unordered inventory.
    write_order = [
        trace_index[write.resource.source_feature_id] for write in spec.public_writes
    ]
    if write_order != sorted(write_order):
        raise GraphAuditError("public writes are not ordered by feature execution")

    current: dict[str, Any] = {}
    consumed_revisions: set[tuple[str, int]] = set()
    reverse_edges = _reverse_dependency_edges(prompt=prompt, spec=spec)

    def dependency_cone(node_ids: Sequence[str]) -> set[str]:
        cone: set[str] = set()
        stack = list(node_ids)
        while stack:
            node_id = stack.pop()
            if node_id in cone:
                continue
            cone.add(node_id)
            stack.extend(reverse_edges[node_id])
        return cone

    for trace in traces:
        feature_reads = reads_by_feature[trace.feature_id]
        for read in feature_reads:
            resource = current.get(read.resource_name)
            if resource is None:
                raise GraphAuditError(
                    f"feature {trace.feature_id!r} reads undefined resource "
                    f"{read.resource_name!r}"
                )
            if (
                resource.type != read.type
                or resource.revision != read.revision
                or resource.value != read.value
            ):
                raise GraphAuditError(
                    f"feature {trace.feature_id!r} does not read the latest exact "
                    f"revision of resource {read.resource_name!r}"
                )
            consumed_revisions.add((read.resource_name, read.revision))

        for write in writes_by_feature[trace.feature_id]:
            resource = write.resource
            previous = current.get(resource.name)
            if write.operation == "define":
                if previous is not None or resource.revision != 1:
                    raise GraphAuditError(
                        f"resource {resource.name!r} define is not revision 1"
                    )
            else:
                if previous is None:
                    raise GraphAuditError(
                        f"resource {resource.name!r} replace has no previous revision"
                    )
                if (
                    previous.type != resource.type
                    or previous.revision + 1 != resource.revision
                    or write.previous_revision != previous.revision
                ):
                    raise GraphAuditError(
                        f"resource {resource.name!r} replace does not advance the latest revision"
                    )
                if not any(
                    read.resource_name == resource.name
                    and read.revision == previous.revision
                    and read.value == previous.value
                    for read in feature_reads
                ):
                    raise GraphAuditError(
                        f"resource {resource.name!r} replace does not consume its previous revision"
                    )
                previous_reads = [
                    read
                    for read in feature_reads
                    if read.resource_name == resource.name
                    and read.revision == previous.revision
                    and read.value == previous.value
                ]
                producer_cone = dependency_cone(resource.producer_node_ids)
                if not set(previous.producer_node_ids) <= producer_cone:
                    raise GraphAuditError(
                        f"resource {resource.name!r} replacement producers do not "
                        "depend on the previous revision"
                    )
                if not {
                    read.consumer_node_id for read in previous_reads
                } <= producer_cone:
                    raise GraphAuditError(
                        f"resource {resource.name!r} previous-revision read does not "
                        "influence the replacement producers"
                    )
            current[resource.name] = resource

    # Every data revision is consumed explicitly.  A terminal resource is the
    # final execution product and is intentionally not a downstream data read.
    for write in spec.public_writes:
        key = (write.resource.name, write.resource.revision)
        if any(isinstance(ref, TerminalRef) for ref in _leaf_refs(write.resource.value)):
            continue
        if key not in consumed_revisions:
            raise GraphAuditError(
                f"dead public write: resource {write.resource.name!r} revision "
                f"{write.resource.revision} has no downstream public read"
            )

    # Every edge crossing a feature scope is explained by exactly one public
    # resource read.  Private values therefore cannot leak between scopes.
    decoded_reads = [
        (_decode_pointer(read.input_pointer), read)
        for read in spec.public_reads
    ]
    typed_dependency_paths = tuple(
        _decode_pointer(item.input_pointer)
        for item in spec.allowed_late_bound_inputs
        if item.source_kind != "resource"
    )
    for edge in collect_prompt_input_edges(prompt):
        source_owner = owner_by_node[edge.source_node_id]
        consumer_owner = owner_by_node[edge.consumer_node_id]
        if source_owner == consumer_owner:
            continue
        edge_tokens = _decode_pointer(edge.input_pointer)
        if any(
            edge_tokens[: len(dependency_path)] == dependency_path
            for dependency_path in typed_dependency_paths
        ):
            # Continuity/runtime dependencies have their own typed evidence and
            # are verified against the exact prepared value at late binding.
            continue
        matches = [
            read
            for tokens, read in decoded_reads
            if edge.consumer_node_id == read.consumer_node_id
            and edge_tokens[: len(tokens)] == tokens
        ]
        if len(matches) != 1:
            raise GraphAuditError(
                f"cross-feature edge at {edge.input_pointer} is not owned by one public read"
            )
        resource_ref_nodes = {
            ref.node_id for ref in _leaf_refs(matches[0].value)
        }
        if edge.source_node_id not in resource_ref_nodes:
            raise GraphAuditError(
                f"cross-feature edge at {edge.input_pointer} differs from resource evidence"
            )


def _reverse_dependency_edges(
    *,
    prompt: Mapping[str, Any],
    spec: GraphAuditSpec,
) -> dict[str, set[str]]:
    reverse_edges: dict[str, set[str]] = {node_id: set() for node_id in prompt}
    for edge in collect_prompt_input_edges(prompt):
        reverse_edges[edge.consumer_node_id].add(edge.source_node_id)
    # A resource pointer may deliberately contain an unbound placeholder in
    # prompt_base.  Its exact typed resource ref is still part of the audit and
    # therefore participates as a virtual dependency until materialization.
    resource_late_pointers = {
        item.input_pointer
        for item in spec.allowed_late_bound_inputs
        if item.source_kind == "resource"
    }
    for read in spec.public_reads:
        if read.input_pointer not in resource_late_pointers:
            continue
        for ref in _leaf_refs(read.value):
            if isinstance(ref, TerminalRef):
                raise GraphAuditError("late-bound data dependency cannot be a terminal")
            reverse_edges[read.consumer_node_id].add(ref.node_id)
    return reverse_edges


def _validate_acyclic_and_cone(
    *,
    prompt: Mapping[str, Any],
    spec: GraphAuditSpec,
) -> None:
    reverse_edges = _reverse_dependency_edges(prompt=prompt, spec=spec)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise GraphAuditError("prompt data-edge graph must be acyclic")
        if node_id in visited:
            return
        visiting.add(node_id)
        for predecessor in reverse_edges[node_id]:
            visit(predecessor)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in prompt:
        visit(node_id)

    if spec.unit_kind == "control":
        if spec.public_writes or spec.public_reads or spec.structural_influence_features:
            raise GraphAuditError(
                "control graph cannot publish data resources or structural influence"
            )
        return

    assert spec.take_node_id is not None
    terminal_writes = [
        write
        for write in spec.public_writes
        if isinstance(write.resource.value, TerminalRef)
    ]
    if (
        len(terminal_writes) != 1
        or terminal_writes[0].resource.name != "take_output"
        or terminal_writes[0].resource.value.node_id != spec.take_node_id
    ):
        raise GraphAuditError(
            "segment graph must publish exactly one take_output terminal"
        )
    cone: set[str] = set()
    stack = [spec.take_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in cone:
            continue
        cone.add(node_id)
        stack.extend(reverse_edges[node_id])

    for write in spec.public_writes:
        missing = set(write.resource.producer_node_ids) - cone
        if missing:
            raise GraphAuditError(
                f"dead public write: resource {write.resource.name!r} producers "
                "are outside the unique take dependency cone"
            )


def _validate_late_bound_declarations(
    *,
    prompt: Mapping[str, Any],
    spec: GraphAuditSpec,
    registry: NodeContractRegistry,
) -> None:
    token_paths: list[tuple[str, ...]] = []
    for allowed in spec.allowed_late_bound_inputs:
        if allowed.base_value_digest is None:
            raise GraphAuditError(
                f"late-bound pointer {allowed.input_pointer!r} has no immutable "
                "base-value digest"
            )
        if allowed.value_kind not in _SUPPORTED_LATE_BOUND_KINDS:
            raise GraphAuditError(
                f"unsupported late-bound value kind {allowed.value_kind!r}"
            )
        tokens = _decode_pointer(allowed.input_pointer)
        if len(tokens) != 3 or tokens[0] not in prompt or tokens[1] != "inputs":
            raise GraphAuditError(
                "late-bound pointers must identify one direct prompt input"
            )
        _resolve_pointer(prompt, allowed.input_pointer)
        if any(
            tokens[: len(existing)] == existing
            or existing[: len(tokens)] == tokens
            for existing in token_paths
        ):
            raise GraphAuditError("late-bound pointers must be disjoint")
        token_paths.append(tokens)

        node_contract = registry.require(prompt[tokens[0]]["class_type"])
        input_contract = (
            node_contract.object_info_contract.required_inputs.get(tokens[2])
            or node_contract.object_info_contract.optional_inputs.get(tokens[2])
        )
        if input_contract is None:
            raise GraphAuditError(
                f"late-bound pointer {allowed.input_pointer!r} targets an "
                "undeclared node input"
            )
        expected_literal_kind = {
            "STRING": "string",
            "INT": "integer",
            "FLOAT": "number",
            "BOOLEAN": "boolean",
            "LIST": "list",
            "RECORD": "record",
            "JSON": "json",
        }.get(input_contract.port_type)
        if input_contract.port_type.endswith("COMBO"):
            expected_literal_kind = "string"

        if allowed.source_kind == "resource":
            resource = next(
                (
                    write.resource
                    for write in spec.public_writes
                    if write.resource.name == allowed.resource_name
                    and write.resource.revision == allowed.revision
                ),
                None,
            )
            if resource is None:
                raise GraphAuditError(
                    f"late-bound resource {allowed.resource_name!r} revision "
                    f"{allowed.revision!r} has no public write"
                )
            expected_resource_kind = (
                "edge"
                if isinstance(resource.value, EdgeRef)
                else "list"
                if isinstance(resource.value, ListRef)
                else "record"
                if isinstance(resource.value, RecordRef)
                else None
            )
            if expected_resource_kind is None:
                raise GraphAuditError("terminal resources cannot be late-bound inputs")
            if allowed.value_kind != expected_resource_kind:
                raise GraphAuditError(
                    f"late-bound resource at {allowed.input_pointer!r} requires "
                    f"value kind {expected_resource_kind!r}"
                )
        elif allowed.value_kind != "edge":
            if (
                expected_literal_kind is None
                or allowed.value_kind != expected_literal_kind
            ):
                raise GraphAuditError(
                    f"late-bound value kind {allowed.value_kind!r} is incompatible "
                    f"with input port type {input_contract.port_type!r}"
                )


def _validate_public_ref_output_slots(
    *,
    spec: GraphAuditSpec,
    registry: NodeContractRegistry,
) -> None:
    for write in spec.public_writes:
        for ref in _leaf_refs(write.resource.value):
            if isinstance(ref, TerminalRef):
                if write.resource.type != "TAKE":
                    raise GraphAuditError(
                        f"terminal resource {write.resource.name!r} must use type 'TAKE'"
                    )
                continue
            evidence = spec.node_contract_snapshot[ref.node_id]
            contract = registry.require(evidence.class_type)
            slots = {
                slot.index: slot for slot in contract.output_contract.slots
            }
            if ref.output_slot not in slots:
                raise GraphAuditError(
                    f"resource {write.resource.name!r} uses undeclared output slot "
                    f"{ref.output_slot} of node {ref.node_id!r}"
                )
            if (
                isinstance(write.resource.value, EdgeRef)
                and slots[ref.output_slot].port_type != write.resource.type
            ):
                raise GraphAuditError(
                    f"resource {write.resource.name!r} declares type "
                    f"{write.resource.type!r}, but its producer slot is "
                    f"{slots[ref.output_slot].port_type!r}"
                )
    for read in spec.public_reads:
        tokens = _decode_pointer(read.input_pointer)
        evidence = spec.node_contract_snapshot[read.consumer_node_id]
        contract = registry.require(evidence.class_type).object_info_contract
        input_name = tokens[2] if len(tokens) >= 3 else ""
        input_contract = (
            contract.required_inputs.get(input_name)
            or contract.optional_inputs.get(input_name)
        )
        if input_contract is None or input_contract.port_type != read.type:
            raise GraphAuditError(
                f"public read at {read.input_pointer} has resource type "
                f"{read.type!r} incompatible with its consumer input"
            )


def validate_graph_audit_spec(
    *,
    prompt: Mapping[str, Any],
    spec: GraphAuditSpec,
    node_contract_registry: NodeContractRegistry,
    feature_traces: Sequence[FeatureAuditTrace],
    model_family: ModelFamily,
    backend: Backend,
    enforce_runtime_effects: bool = True,
) -> None:
    """Validate one immutable audit against its exact prompt representation.

    ``enforce_runtime_effects=False`` is a compile-only compatibility gate.  It
    skips only the fail-closed effect policy check; resolution, provenance,
    fingerprint and every structural rule remain mandatory.
    """

    _validate_prompt_shape(prompt)
    _validate_node_contracts(
        prompt=prompt,
        spec=spec,
        registry=node_contract_registry,
    )
    owner_by_node = _validate_feature_traces(
        prompt=prompt,
        spec=spec,
        traces=feature_traces,
        registry=node_contract_registry,
        model_family=model_family,
        backend=backend,
        enforce_runtime_effects=enforce_runtime_effects,
    )
    _validate_late_bound_declarations(
        prompt=prompt,
        spec=spec,
        registry=node_contract_registry,
    )
    _validate_public_ref_output_slots(spec=spec, registry=node_contract_registry)
    _validate_resource_history(
        prompt=prompt,
        spec=spec,
        traces=feature_traces,
        owner_by_node=owner_by_node,
    )
    _validate_acyclic_and_cone(prompt=prompt, spec=spec)


def build_graph_audit_spec(
    *,
    prompt: Mapping[str, Any],
    node_contract_registry: NodeContractRegistry,
    node_contract_snapshot: Mapping[str, GraphNodeContractEvidence],
    public_writes: Sequence[PublicResourceWrite],
    public_reads: Sequence[PublicResourceRead],
    feature_traces: Sequence[FeatureAuditTrace],
    model_family: ModelFamily,
    backend: Backend,
    allowed_late_bound_inputs: Sequence[AllowedLateBoundInput] = (),
    unit_kind: Literal["segment", "control"],
    take_node_id: str | None = None,
    control_kind: Literal["ray_kill"] | None = None,
    version: int = 1,
    enforce_runtime_effects: bool = True,
) -> GraphAuditSpec:
    """Construct a :class:`GraphAuditSpec` and immediately prove it valid.

    The runtime-effect switch has the same narrow compile-only semantics as
    :func:`validate_graph_audit_spec` and defaults to strict enforcement.
    """

    anchored_late_bound_inputs: list[AllowedLateBoundInput] = []
    for declaration in allowed_late_bound_inputs:
        try:
            base_value_digest = canonical_sha256(
                _resolve_pointer(prompt, declaration.input_pointer)
            )
        except (GraphAuditError, TypeError, ValueError) as exc:
            raise GraphAuditError(
                f"cannot anchor late-bound pointer "
                f"{declaration.input_pointer!r}: {exc}"
            ) from exc
        if (
            declaration.base_value_digest is not None
            and declaration.base_value_digest != base_value_digest
        ):
            raise GraphAuditError(
                f"late-bound pointer {declaration.input_pointer!r} supplied an "
                "incorrect base-value digest"
            )
        anchored_late_bound_inputs.append(
            declaration.model_copy(
                update={"base_value_digest": base_value_digest}
            )
        )

    structural_features = tuple(
        trace.feature_id for trace in feature_traces if trace.structural_influence
    )
    try:
        spec = GraphAuditSpec(
            version=version,
            unit_kind=unit_kind,
            control_kind=control_kind,
            take_node_id=take_node_id,
            node_contract_snapshot=dict(node_contract_snapshot),
            public_writes=tuple(public_writes),
            public_reads=tuple(public_reads),
            allowed_late_bound_inputs=tuple(anchored_late_bound_inputs),
            structural_influence_features=structural_features,
        )
    except ValueError as exc:
        raise GraphAuditError(f"invalid GraphAuditSpec ingredients: {exc}") from exc
    validate_graph_audit_spec(
        prompt=prompt,
        spec=spec,
        node_contract_registry=node_contract_registry,
        feature_traces=feature_traces,
        model_family=model_family,
        backend=backend,
        enforce_runtime_effects=enforce_runtime_effects,
    )
    return spec


def _assert_only_allowed_changes(
    base: Any,
    bound: Any,
    *,
    allowed_paths: frozenset[tuple[str, ...]],
    path: tuple[str, ...] = (),
) -> None:
    if path in allowed_paths:
        return
    if isinstance(base, Mapping):
        if not isinstance(bound, Mapping) or tuple(base) != tuple(bound):
            raise GraphAuditError(
                "late binding changed a mapping type, key set, or key order outside whitelist"
            )
        for key in base:
            _assert_only_allowed_changes(
                base[key],
                bound[key],
                allowed_paths=allowed_paths,
                path=(*path, key),
            )
        return
    if isinstance(base, (list, tuple)):
        if not isinstance(bound, (list, tuple)) or len(base) != len(bound):
            raise GraphAuditError(
                "late binding changed a sequence type or length outside whitelist"
            )
        for index, (base_item, bound_item) in enumerate(
            zip(base, bound, strict=True)
        ):
            _assert_only_allowed_changes(
                base_item,
                bound_item,
                allowed_paths=allowed_paths,
                path=(*path, str(index)),
            )
        return
    if not _same_exact_value(base, bound):
        raise GraphAuditError("late binding changed a non-whitelisted prompt value")


def _validate_value_kind(kind: _LateBoundKind | str, value: Any) -> None:
    valid = False
    if kind == "edge":
        valid = (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
            and 0 <= value[1] <= 255
        )
    elif kind == "string":
        valid = isinstance(value, str)
    elif kind == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "number":
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    elif kind == "boolean":
        valid = isinstance(value, bool)
    elif kind == "list":
        valid = isinstance(value, (list, tuple))
    elif kind == "record":
        valid = isinstance(value, Mapping)
    elif kind == "json":
        valid = value is None or isinstance(
            value, (bool, int, float, str, list, tuple, Mapping)
        )
    if not valid:
        raise GraphAuditError(f"late-bound value does not match declared kind {kind!r}")


def _is_plain_edge(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _is_prompt_edge_value(value: Any) -> bool:
    return _is_plain_edge(value) and 0 <= value[1] <= 255


def _assert_existing_output_slots_unchanged(base: Any, bound: Any) -> None:
    """Reject changing the slot of an edge already present in prompt_base."""

    if _is_plain_edge(base):
        if _is_plain_edge(bound) and base[1] != bound[1]:
            raise GraphAuditError("late binding changed an existing edge output slot")
        return
    if isinstance(base, Mapping) and isinstance(bound, Mapping):
        for key in set(base) & set(bound):
            _assert_existing_output_slots_unchanged(base[key], bound[key])
        return
    if isinstance(base, (list, tuple)) and isinstance(bound, (list, tuple)):
        for base_item, bound_item in zip(base, bound):
            _assert_existing_output_slots_unchanged(base_item, bound_item)


def validate_bound_graph(
    *,
    prompt_base: Mapping[str, Any],
    bound_prompt: Mapping[str, Any],
    spec: GraphAuditSpec,
    node_contract_registry: NodeContractRegistry,
    model_family: ModelFamily,
    backend: Backend,
    feature_traces: Sequence[FeatureAuditTrace] | None = None,
    expected_late_bound_values: Mapping[str, Any] | None = None,
    enforce_runtime_effects: bool = True,
) -> None:
    """Prove that materialization changed only exact declared input pointers.

    Resource-sourced values are derived from the referenced public write.
    Continuity and runtime-epoch values must be supplied from their prepared,
    typed dependency evidence through ``expected_late_bound_values``.
    Runtime effects remain strict by default; structural-only validation
    without transient traces requires an explicit ``False`` compatibility gate.
    """

    if enforce_runtime_effects and feature_traces is None:
        raise GraphAuditError(
            "strict runtime-effect validation requires feature audit traces"
        )
    if feature_traces is not None:
        validate_graph_audit_spec(
            prompt=prompt_base,
            spec=spec,
            node_contract_registry=node_contract_registry,
            feature_traces=feature_traces,
            model_family=model_family,
            backend=backend,
            enforce_runtime_effects=enforce_runtime_effects,
        )
    else:
        # Prepared/locked units persist GraphAuditSpec, not the compiler's
        # transient per-interpreter trace.  The base was fully proven when the
        # spec was built; submission-time validation rechecks every property
        # that a whitelisted value replacement could affect.
        _validate_prompt_shape(prompt_base)
        _validate_node_contracts(
            prompt=prompt_base,
            spec=spec,
            registry=node_contract_registry,
        )
        _validate_late_bound_declarations(
            prompt=prompt_base,
            spec=spec,
            registry=node_contract_registry,
        )
        _validate_public_ref_output_slots(
            spec=spec,
            registry=node_contract_registry,
        )
        _validate_acyclic_and_cone(prompt=prompt_base, spec=spec)
    expected = dict(expected_late_bound_values or {})
    allowed_by_pointer = {
        item.input_pointer: item for item in spec.allowed_late_bound_inputs
    }
    unknown_expected = set(expected) - set(allowed_by_pointer)
    if unknown_expected:
        raise GraphAuditError("late-bound expected values include unknown pointers")

    allowed_paths = frozenset(
        _decode_pointer(pointer) for pointer in allowed_by_pointer
    )
    _assert_only_allowed_changes(
        prompt_base,
        bound_prompt,
        allowed_paths=allowed_paths,
    )

    writes = {
        (write.resource.name, write.resource.revision): write.resource
        for write in spec.public_writes
    }
    for pointer, declaration in allowed_by_pointer.items():
        base_value = _resolve_pointer(prompt_base, pointer)
        actual = _resolve_pointer(bound_prompt, pointer)
        changed = not _same_exact_value(base_value, actual)
        base_is_original = (
            canonical_sha256(base_value) == declaration.base_value_digest
        )
        actual_is_original = (
            canonical_sha256(actual) == declaration.base_value_digest
        )
        _assert_existing_output_slots_unchanged(base_value, actual)
        if declaration.source_kind == "resource":
            resource = writes[(declaration.resource_name, declaration.revision)]
            wanted = _plain_published_ref(resource.value)
            if pointer in expected and not _same_exact_value(expected[pointer], wanted):
                raise GraphAuditError(
                    "caller-supplied resource late-bound value disagrees with audit"
                )
        else:
            needs_typed_evidence = (
                changed
                or not base_is_original
                or not actual_is_original
                or pointer in expected
            )
            if not needs_typed_evidence:
                continue
            if pointer not in expected:
                raise GraphAuditError(
                    f"missing typed expected value for late-bound pointer {pointer!r}"
                )
            wanted = expected[pointer]
        if not base_is_original:
            _validate_value_kind(declaration.value_kind, base_value)
            if not _same_exact_value(base_value, wanted):
                raise GraphAuditError(
                    f"late-bound pointer {pointer!r} base differs from its "
                    "immutable placeholder without matching typed evidence"
                )
        elif not changed and pointer not in expected:
            # An untouched original placeholder consumes no dependency evidence.
            continue
        _validate_value_kind(declaration.value_kind, actual)
        _validate_value_kind(declaration.value_kind, wanted)
        pointer_tokens = _decode_pointer(pointer)
        bound_contract = node_contract_registry.require(
            bound_prompt[pointer_tokens[0]]["class_type"]
        )
        bound_input_contract = (
            bound_contract.object_info_contract.required_inputs.get(
                pointer_tokens[2]
            )
            or bound_contract.object_info_contract.optional_inputs.get(
                pointer_tokens[2]
            )
        )
        assert bound_input_contract is not None
        scalar_kind = declaration.value_kind in {
            "string",
            "integer",
            "number",
            "boolean",
        }
        check_enum = declaration.value_kind not in {"edge", "list", "record"}
        _validate_object_info_literal(
            node_id=pointer_tokens[0],
            input_name=pointer_tokens[2],
            value=actual,
            input_contract=bound_input_contract,
            check_enum=check_enum,
            require_literal_type=scalar_kind,
        )
        _validate_object_info_literal(
            node_id=pointer_tokens[0],
            input_name=pointer_tokens[2],
            value=wanted,
            input_contract=bound_input_contract,
            check_enum=check_enum,
            require_literal_type=scalar_kind,
        )
        if not _same_exact_value(actual, wanted):
            raise GraphAuditError(
                f"late-bound pointer {pointer!r} does not equal its typed dependency"
            )

    # Running the same pure structural kernels again catches node-set, class,
    # output-slot, resource-edge, terminal and dependency-cone drift in the
    # exact materialized prompt.  Full transient resolution/history evidence is
    # also rerun when the compiler still has it.
    if feature_traces is not None:
        validate_graph_audit_spec(
            prompt=bound_prompt,
            spec=spec,
            node_contract_registry=node_contract_registry,
            feature_traces=feature_traces,
            model_family=model_family,
            backend=backend,
            enforce_runtime_effects=enforce_runtime_effects,
        )
    else:
        _validate_prompt_shape(bound_prompt)
        _validate_node_contracts(
            prompt=bound_prompt,
            spec=spec,
            registry=node_contract_registry,
        )
        _validate_late_bound_declarations(
            prompt=bound_prompt,
            spec=spec,
            registry=node_contract_registry,
        )
        _validate_public_ref_output_slots(
            spec=spec,
            registry=node_contract_registry,
        )
        for read in spec.public_reads:
            _validate_read_pointer(
                prompt=bound_prompt,
                read=read,
                allowed_resource_pointers=frozenset(),
            )
        _validate_acyclic_and_cone(prompt=bound_prompt, spec=spec)


__all__ = [
    "FeatureAuditTrace",
    "GraphAuditError",
    "PromptInputEdge",
    "ResolvedNodeEmission",
    "build_graph_audit_spec",
    "collect_prompt_input_edges",
    "validate_bound_graph",
    "validate_graph_audit_spec",
]
