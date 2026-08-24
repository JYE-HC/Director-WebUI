from __future__ import annotations

"""Small graph/resource helpers shared by native semantic compilers."""

import re
from collections.abc import Mapping
from typing import Any, Protocol

from .audit import FeatureAuditTrace, ResolvedNodeEmission
from .contracts import (
    DirectorAdapterContractEvidence,
    EdgeRef,
    FeatureEmission,
    FeatureResolution,
    GraphNodeContractEvidence,
    NodeContractEvidence,
    NodeContractRegistry,
    PublicResourceRead,
    PublicResourceWrite,
    PublishedValueRef,
    ResolvedImplementationIdentity,
    Resource,
    ResourcePool,
    TerminalRef,
    director_adapter_contract_digest,
)


class FeatureUseLike(Protocol):
    reads: tuple[Any, ...]
    writes: tuple[Any, ...]


def published_node_ids(value: PublishedValueRef) -> tuple[str, ...]:
    if isinstance(value, (EdgeRef, TerminalRef)):
        return (value.node_id,)
    children = (
        value.items
        if hasattr(value, "items") and not hasattr(value, "fields")
        else value.fields.values()
    )
    ordered: list[str] = []
    for child in children:
        for node_id in published_node_ids(child):
            if node_id not in ordered:
                ordered.append(node_id)
    return tuple(ordered)


def _plain_ref(value: PublishedValueRef) -> Any:
    if isinstance(value, EdgeRef):
        return [value.node_id, value.output_slot]
    if isinstance(value, TerminalRef):
        return {"kind": "terminal", "node_id": value.node_id}
    if hasattr(value, "items") and not hasattr(value, "fields"):
        return [_plain_ref(item) for item in value.items]
    return {key: _plain_ref(item) for key, item in value.fields.items()}


def _edge_pointers(
    value: PublishedValueRef,
    *,
    pointer: str,
) -> tuple[tuple[str, EdgeRef], ...]:
    if isinstance(value, EdgeRef):
        return ((pointer, value),)
    if isinstance(value, TerminalRef):
        raise AssertionError("terminal resources cannot be graph inputs")
    if hasattr(value, "items") and not hasattr(value, "fields"):
        return tuple(
            leaf
            for index, item in enumerate(value.items)
            for leaf in _edge_pointers(item, pointer=f"{pointer}/{index}")
        )
    return tuple(
        leaf
        for key, item in value.fields.items()
        for leaf in _edge_pointers(
            item,
            pointer=f"{pointer}/{key.replace('~', '~0').replace('/', '~1')}",
        )
    )


def read_resources(pool: ResourcePool, use: FeatureUseLike) -> dict[str, Resource]:
    inputs: dict[str, Resource] = {}
    for declaration in use.reads:
        resource = (
            pool.read_required(declaration.name, expected_type=declaration.type)
            if declaration.required
            else pool.read_optional(declaration.name, expected_type=declaration.type)
        )
        if resource is not None:
            inputs[declaration.name] = resource
    return inputs


def commit_emission(
    *,
    pool: ResourcePool,
    owner_id: str,
    use: FeatureUseLike,
    emission: FeatureEmission,
    scope: Any,
) -> ResourcePool:
    writes = {declaration.name: declaration for declaration in use.writes}
    unexpected = set(emission.outputs) - set(writes)
    missing = {
        name
        for name, declaration in writes.items()
        if declaration.required and name not in emission.outputs
    }
    if unexpected or missing:
        raise AssertionError(
            f"feature {owner_id!r} resource contract mismatch: "
            f"unexpected={sorted(unexpected)!r}, missing={sorted(missing)!r}"
        )
    transaction = pool.begin()
    for name, value in emission.outputs.items():
        declaration = writes[name]
        producers = published_node_ids(value)
        if declaration.operation == "define":
            transaction = transaction.define(
                name=name,
                type=declaration.type,
                value=value,
                source_feature_id=owner_id,
                producer_node_ids=producers,
            )
        else:
            transaction = transaction.replace(
                name=name,
                value=value,
                source_feature_id=owner_id,
                producer_node_ids=producers,
                expected_type=declaration.type,
            )
    return scope.commit_emission(emission, transaction)


def public_reads(
    inputs: Mapping[str, Resource],
    scope: Any,
) -> tuple[PublicResourceRead, ...]:
    local_nodes = set(scope.emitted_node_ids)
    external = tuple(
        item for item in scope.input_edge_evidence if item.value.node_id not in local_nodes
    )
    evidence_keys = {
        (item.consumer_node_id, item.input_pointer, item.value) for item in external
    }
    if len(evidence_keys) != len(external):
        raise AssertionError(
            f"feature {scope.feature_id!r} produced duplicate input-edge evidence"
        )
    reads: list[PublicResourceRead] = []
    claimed: set[tuple[str, str, EdgeRef]] = set()
    for consumer, node in scope.prompt_fragment.items():
        for input_name, actual in node["inputs"].items():
            pointer = (
                f"/{consumer}/inputs/"
                f"{input_name.replace('~', '~0').replace('/', '~1')}"
            )
            matches = [
                resource
                for resource in inputs.values()
                if _plain_ref(resource.value) == actual
            ]
            related = tuple(
                item
                for item in external
                if item.consumer_node_id == consumer
                and (
                    item.input_pointer == pointer
                    or item.input_pointer.startswith(pointer + "/")
                )
            )
            if not related:
                continue
            if len(matches) != 1:
                raise AssertionError("cross-feature edge must match one typed resource")
            resource = matches[0]
            expected = {
                (consumer, leaf_pointer, edge)
                for leaf_pointer, edge in _edge_pointers(resource.value, pointer=pointer)
            }
            observed = {
                (item.consumer_node_id, item.input_pointer, item.value) for item in related
            }
            if observed != expected or claimed & expected:
                raise AssertionError("cross-feature edge evidence is not exact")
            claimed.update(expected)
            reads.append(
                PublicResourceRead(
                    resource_name=resource.name,
                    type=resource.type,
                    revision=resource.revision,
                    consumer_node_id=consumer,
                    input_pointer=pointer,
                    value=resource.value,
                )
            )
    if evidence_keys != claimed:
        raise AssertionError("cross-feature edge evidence is unclaimed")
    return tuple(reads)


def public_writes(
    before: ResourcePool,
    after: ResourcePool,
    use: FeatureUseLike,
    emission: FeatureEmission,
) -> tuple[PublicResourceWrite, ...]:
    writes: list[PublicResourceWrite] = []
    for declaration in use.writes:
        if declaration.name not in emission.outputs:
            continue
        resource = after.resources[declaration.name]
        previous = before.resources.get(declaration.name)
        writes.append(
            PublicResourceWrite(
                operation=declaration.operation,
                resource=resource,
                previous_revision=(previous.revision if previous else None),
            )
        )
    return tuple(writes)


def node_contract_snapshot(
    prompt: Mapping[str, Any],
    registry: NodeContractRegistry,
    *,
    director_adapter_only: bool = False,
) -> dict[str, GraphNodeContractEvidence]:
    snapshot: dict[str, GraphNodeContractEvidence] = {}
    for node_id, node in prompt.items():
        contract = registry.require(node["class_type"])
        if director_adapter_only:
            snapshot[str(node_id)] = DirectorAdapterContractEvidence(
                contract_id=contract.contract_id,
                semantic_version=contract.semantic_version,
                class_type=contract.class_type,
                adapter_contract_digest=director_adapter_contract_digest(contract),
                execution_terminal_role=contract.execution_terminal_role,
                persistent_artifact_role=contract.persistent_artifact_role,
            )
            continue
        if (
            len(contract.allowed_python_modules) != 1
            or len(contract.supported_runtime_fingerprints) != 1
        ):
            raise AssertionError(
                f"legacy node contract must select one adapter: {contract.class_type}"
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


def audit_trace(
    *,
    feature_id: str,
    scope: Any,
    registry: NodeContractRegistry,
    structural_influence: bool,
    output_neutral_classes: frozenset[str],
) -> FeatureAuditTrace | None:
    if not scope.emitted_node_ids:
        return None
    class_types = tuple(
        dict.fromkeys(node["class_type"] for node in scope.prompt_fragment.values())
    )
    identities = []
    binding_by_class: dict[str, str] = {}
    for class_type in class_types:
        contract = registry.require(class_type)
        binding = re.sub(r"[^A-Za-z0-9_.:-]", "_", f"{feature_id}.{class_type}")
        binding_by_class[class_type] = binding
        identities.append(
            ResolvedImplementationIdentity(
                role="node",
                class_type=class_type,
                implementation_id=contract.contract_id,
                semantic_version=contract.semantic_version,
                runtime_fingerprint=contract.supported_runtime_fingerprints[0],
                binding_key=binding,
            )
        )
    resolution = FeatureResolution(
        state="active",
        implementations=tuple(identities),
        resolution_details={"source": "bundle6_graph_audit_compat"},
    )
    return FeatureAuditTrace(
        feature_id=feature_id,
        resolution=resolution,
        emitted_nodes=tuple(
            ResolvedNodeEmission(
                node_id=node_id,
                implementation_binding_key=binding_by_class[node["class_type"]],
                output_affecting=node["class_type"] not in output_neutral_classes,
            )
            for node_id, node in scope.prompt_fragment.items()
        ),
        structural_influence=structural_influence,
    )


__all__ = [
    "audit_trace",
    "commit_emission",
    "node_contract_snapshot",
    "public_reads",
    "public_writes",
    "read_resources",
]
