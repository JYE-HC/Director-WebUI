from __future__ import annotations

"""Transactional graph construction primitives for workflow interpreters.

The builder deliberately keeps ComfyUI's API prompt representation as ordinary
``dict``/``list`` JSON.  Typed references exist at the interpreter boundary and
are lowered only while a node is staged.  A feature scope is the atomic unit:
its nodes, node-id counter and resource-pool transaction either commit together
or are all discarded.
"""

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .canonical import MAX_SAFE_JSON_INTEGER
from .contracts import (
    EdgeRef,
    FeatureEmission,
    FrozenMap,
    ListRef,
    PublishedValueRef,
    RecordRef,
    ResourcePool,
    ResourcePoolTransaction,
    TerminalRef,
)


MAX_GRAPH_INPUT_DEPTH = 16
MAX_GRAPH_CONTAINER_ITEMS = 4_096
MAX_GRAPH_STRING_LENGTH = 65_536


class GraphBuilderError(ValueError):
    """A feature attempted an invalid or non-atomic graph mutation."""


@dataclass(frozen=True, slots=True)
class InputEdgeEvidence:
    """One exact prompt pointer at which a scope consumed a typed edge."""

    feature_id: str
    consumer_node_id: str
    input_pointer: str
    value: EdgeRef


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _published_refs(value: PublishedValueRef) -> tuple[EdgeRef | TerminalRef, ...]:
    if isinstance(value, (EdgeRef, TerminalRef)):
        return (value,)
    children = value.items if isinstance(value, ListRef) else value.fields.values()
    return tuple(ref for child in children for ref in _published_refs(child))


def _plain_json(
    value: Any,
    *,
    scope: ScopedGraphBuilder,
    consumer_node_id: str,
    input_pointer: str,
    depth: int = 1,
    ancestors: set[int] | None = None,
) -> Any:
    """Lower typed edges and validate a bounded ordinary JSON input value."""

    if depth > MAX_GRAPH_INPUT_DEPTH:
        raise GraphBuilderError(
            f"node input exceeds maximum depth {MAX_GRAPH_INPUT_DEPTH}"
        )
    if ancestors is None:
        ancestors = set()

    if isinstance(value, EdgeRef):
        scope._record_input_edge(
            value,
            consumer_node_id=consumer_node_id,
            input_pointer=input_pointer,
        )
        return [value.node_id, value.output_slot]
    if isinstance(value, TerminalRef):
        raise GraphBuilderError("terminal references cannot be consumed downstream")
    if isinstance(value, ListRef):
        return [
            _plain_json(
                item,
                scope=scope,
                consumer_node_id=consumer_node_id,
                input_pointer=f"{input_pointer}/{index}",
                depth=depth + 1,
                ancestors=ancestors,
            )
            for index, item in enumerate(value.items)
        ]
    if isinstance(value, RecordRef):
        return {
            key: _plain_json(
                item,
                scope=scope,
                consumer_node_id=consumer_node_id,
                input_pointer=f"{input_pointer}/{_json_pointer_token(key)}",
                depth=depth + 1,
                ancestors=ancestors,
            )
            for key, item in value.fields.items()
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, int) and not isinstance(value, bool):
            if abs(value) > MAX_SAFE_JSON_INTEGER:
                raise GraphBuilderError("node input integer exceeds JSON safe range")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise GraphBuilderError("node input number must be finite")
            if value == 0 and math.copysign(1.0, value) < 0:
                raise GraphBuilderError("node input number cannot be negative zero")
        elif isinstance(value, str):
            if len(value) > MAX_GRAPH_STRING_LENGTH:
                raise GraphBuilderError("node input string exceeds maximum length")
            if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
                raise GraphBuilderError("node input string contains a lone surrogate")
        return value

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise GraphBuilderError("node input cannot contain reference cycles")
        if len(value) > MAX_GRAPH_CONTAINER_ITEMS:
            raise GraphBuilderError("node input object exceeds maximum length")
        ancestors.add(identity)
        try:
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str) or not key:
                    raise GraphBuilderError(
                        "node input object keys must be non-empty strings"
                    )
                if len(key) > 256 or any(
                    ord(character) < 0x20
                    or 0xD800 <= ord(character) <= 0xDFFF
                    for character in key
                ):
                    raise GraphBuilderError("node input object key is not JSON-safe")
                result[key] = _plain_json(
                    item,
                    scope=scope,
                    consumer_node_id=consumer_node_id,
                    input_pointer=(
                        f"{input_pointer}/{_json_pointer_token(key)}"
                    ),
                    depth=depth + 1,
                    ancestors=ancestors,
                )
            return result
        finally:
            ancestors.remove(identity)

    if isinstance(value, (list, tuple)):
        # A raw Comfy edge would bypass terminal/provenance checks.  Interpreters
        # must use EdgeRef even though the committed prompt remains a plain list.
        if (
            len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
        ):
            raise GraphBuilderError(
                "raw edge-shaped node input is forbidden; use builder.edge()"
            )
        identity = id(value)
        if identity in ancestors:
            raise GraphBuilderError("node input cannot contain reference cycles")
        if len(value) > MAX_GRAPH_CONTAINER_ITEMS:
            raise GraphBuilderError("node input array exceeds maximum length")
        ancestors.add(identity)
        try:
            return [
                _plain_json(
                    item,
                    scope=scope,
                    consumer_node_id=consumer_node_id,
                    input_pointer=f"{input_pointer}/{index}",
                    depth=depth + 1,
                    ancestors=ancestors,
                )
                for index, item in enumerate(value)
            ]
        finally:
            ancestors.remove(identity)

    raise GraphBuilderError(
        f"node input is not JSON or a typed edge: {type(value).__name__}"
    )


class PromptGraphBuilder:
    """Own a prompt and allocate globally continuous decimal node ids."""

    __slots__ = (
        "_active_scope",
        "_counter",
        "_input_edge_evidence",
        "_node_feature_ids",
        "_prompt",
        "_terminal_node_ids",
    )

    def __init__(self) -> None:
        self._prompt: dict[str, dict[str, Any]] = {}
        self._counter = 0
        self._input_edge_evidence: list[InputEdgeEvidence] = []
        self._node_feature_ids: dict[str, str] = {}
        self._terminal_node_ids: set[str] = set()
        self._active_scope: ScopedGraphBuilder | None = None

    @property
    def prompt(self) -> dict[str, dict[str, Any]]:
        """Return an ordinary, detached API-format prompt snapshot."""

        return deepcopy(self._prompt)

    @property
    def node_count(self) -> int:
        return len(self._prompt)

    @property
    def terminal_node_ids(self) -> tuple[str, ...]:
        return tuple(
            node_id for node_id in self._prompt if node_id in self._terminal_node_ids
        )

    @property
    def input_edge_evidence(self) -> tuple[InputEdgeEvidence, ...]:
        return tuple(self._input_edge_evidence)

    @property
    def node_feature_ids(self) -> dict[str, str]:
        return dict(self._node_feature_ids)

    def begin_scope(self, feature_id: str) -> ScopedGraphBuilder:
        if self._active_scope is not None:
            raise GraphBuilderError("a graph feature scope is already active")
        if (
            not isinstance(feature_id, str)
            or not feature_id
            or len(feature_id) > 128
        ):
            raise GraphBuilderError("feature_id must be a non-empty bounded string")
        scope = ScopedGraphBuilder(parent=self, feature_id=feature_id)
        self._active_scope = scope
        return scope

    def _contains_node(self, node_id: str) -> bool:
        return node_id in self._prompt

    def _is_terminal(self, node_id: str) -> bool:
        return node_id in self._terminal_node_ids

    def _commit(self, scope: ScopedGraphBuilder) -> None:
        if self._active_scope is not scope:
            raise GraphBuilderError("feature scope is not the active graph scope")
        if scope._base_counter != self._counter:
            raise GraphBuilderError("graph node counter changed during feature scope")
        if set(scope._prompt_fragment) & set(self._prompt):
            raise GraphBuilderError("feature scope node ids collide with the graph")
        self._prompt.update(deepcopy(scope._prompt_fragment))
        self._node_feature_ids.update(
            {node_id: scope.feature_id for node_id in scope._prompt_fragment}
        )
        self._input_edge_evidence.extend(scope._input_edge_evidence)
        self._terminal_node_ids.update(scope._terminal_node_ids)
        self._counter = scope._counter
        self._active_scope = None

    def _rollback(self, scope: ScopedGraphBuilder) -> None:
        if self._active_scope is scope:
            self._active_scope = None


class ScopedGraphBuilder:
    """Stage the private subgraph emitted by exactly one feature interpreter."""

    __slots__ = (
        "_base_counter",
        "_closed",
        "_committed",
        "_consumed_node_ids",
        "_counter",
        "_emitted_edges",
        "_input_edges",
        "_input_edge_evidence",
        "_parent",
        "_prompt_fragment",
        "_terminal_node_ids",
        "feature_id",
    )

    def __init__(self, *, parent: PromptGraphBuilder, feature_id: str) -> None:
        self._parent = parent
        self.feature_id = feature_id
        self._base_counter = parent._counter
        self._counter = parent._counter
        self._prompt_fragment: dict[str, dict[str, Any]] = {}
        self._terminal_node_ids: set[str] = set()
        self._consumed_node_ids: set[str] = set()
        self._emitted_edges: list[EdgeRef] = []
        self._input_edges: list[EdgeRef] = []
        self._input_edge_evidence: list[InputEdgeEvidence] = []
        self._closed = False
        self._committed = False

    def __enter__(self) -> ScopedGraphBuilder:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._closed:
            self.rollback()

    @property
    def emitted_node_ids(self) -> tuple[str, ...]:
        return tuple(self._prompt_fragment)

    @property
    def emitted_edges(self) -> tuple[EdgeRef, ...]:
        return tuple(self._emitted_edges)

    @property
    def input_edges(self) -> tuple[EdgeRef, ...]:
        return tuple(self._input_edges)

    @property
    def input_edge_evidence(self) -> tuple[InputEdgeEvidence, ...]:
        return tuple(self._input_edge_evidence)

    @property
    def prompt_fragment(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._prompt_fragment)

    @property
    def committed(self) -> bool:
        return self._committed

    def _require_open(self) -> None:
        if self._closed:
            raise GraphBuilderError("feature graph scope is already closed")

    def _node_exists(self, node_id: str) -> bool:
        return node_id in self._prompt_fragment or self._parent._contains_node(node_id)

    def _is_terminal(self, node_id: str) -> bool:
        return (
            node_id in self._terminal_node_ids
            or self._parent._is_terminal(node_id)
        )

    def _record_input_edge(
        self,
        edge: EdgeRef,
        *,
        consumer_node_id: str,
        input_pointer: str,
    ) -> None:
        if not self._node_exists(edge.node_id):
            raise GraphBuilderError(f"edge references unknown node {edge.node_id!r}")
        if self._is_terminal(edge.node_id):
            raise GraphBuilderError("terminal node output cannot be consumed downstream")
        self._consumed_node_ids.add(edge.node_id)
        self._input_edges.append(edge)
        self._input_edge_evidence.append(
            InputEdgeEvidence(
                feature_id=self.feature_id,
                consumer_node_id=consumer_node_id,
                input_pointer=input_pointer,
                value=edge,
            )
        )

    def add_node(
        self,
        class_type: str,
        inputs: Mapping[str, Any] | None = None,
        **input_kwargs: Any,
    ) -> str:
        """Stage one node without exposing a raw Comfy edge to the caller."""

        self._require_open()
        if (
            not isinstance(class_type, str)
            or not class_type
            or len(class_type) > 128
        ):
            raise GraphBuilderError("class_type must be a non-empty bounded string")
        if inputs is not None and input_kwargs:
            raise GraphBuilderError("pass node inputs as a mapping or keywords, not both")
        source_inputs: Mapping[str, Any] = inputs if inputs is not None else input_kwargs
        if not isinstance(source_inputs, Mapping):
            raise GraphBuilderError("node inputs must be a mapping")
        consumed_before = set(self._consumed_node_ids)
        evidence_length_before = len(self._input_edges)
        pointer_evidence_length_before = len(self._input_edge_evidence)
        candidate_node_id = str(self._counter + 1)
        try:
            plain_inputs = _plain_json(
                source_inputs,
                scope=self,
                consumer_node_id=candidate_node_id,
                input_pointer=f"/{candidate_node_id}/inputs",
            )
        except Exception:
            self._consumed_node_ids = consumed_before
            del self._input_edges[evidence_length_before:]
            del self._input_edge_evidence[pointer_evidence_length_before:]
            raise
        assert isinstance(plain_inputs, dict)
        self._counter += 1
        node_id = str(self._counter)
        assert node_id == candidate_node_id
        self._prompt_fragment[node_id] = {
            "class_type": class_type,
            "inputs": plain_inputs,
        }
        return node_id

    def edge(self, node_id: str, output_slot: int = 0) -> EdgeRef:
        self._require_open()
        if not self._node_exists(node_id):
            raise GraphBuilderError(f"edge references unknown node {node_id!r}")
        if self._is_terminal(node_id):
            raise GraphBuilderError("terminal node cannot publish a data edge")
        edge = EdgeRef(node_id=node_id, output_slot=output_slot)
        self._emitted_edges.append(edge)
        return edge

    def terminal(self, node_id: str) -> TerminalRef:
        self._require_open()
        if node_id not in self._prompt_fragment:
            raise GraphBuilderError(
                "a feature can declare terminals only for nodes in its own scope"
            )
        if node_id in self._consumed_node_ids:
            raise GraphBuilderError(
                "a node already consumed downstream cannot become a terminal"
            )
        self._terminal_node_ids.add(node_id)
        return TerminalRef(node_id=node_id)

    def owns_public_value(self, value: PublishedValueRef) -> bool:
        return all(
            ref.node_id in self._prompt_fragment for ref in _published_refs(value)
        )

    def validate_public_outputs(
        self,
        outputs: Mapping[str, PublishedValueRef],
    ) -> None:
        self._require_open()
        for name, value in outputs.items():
            if not isinstance(name, str) or not name:
                raise GraphBuilderError("public output names must be non-empty strings")
            if not isinstance(value, (EdgeRef, TerminalRef, ListRef, RecordRef)):
                raise GraphBuilderError(
                    f"public output {name!r} is not a PublishedValueRef"
                )
            refs = _published_refs(value)
            if not refs:
                raise GraphBuilderError(
                    f"public output {name!r} must contain at least one producer"
                )
            if any(ref.node_id not in self._prompt_fragment for ref in refs):
                raise GraphBuilderError(
                    f"public output {name!r} does not originate in this feature scope"
                )
            for ref in refs:
                if isinstance(ref, TerminalRef):
                    if ref.node_id not in self._terminal_node_ids:
                        raise GraphBuilderError(
                            f"public output {name!r} uses an undeclared terminal"
                        )
                elif ref.node_id in self._terminal_node_ids:
                    raise GraphBuilderError(
                        f"terminal node in public output {name!r} must use TerminalRef"
                    )

    def commit(
        self,
        *,
        public_outputs: Mapping[str, PublishedValueRef] | None = None,
        resource_transaction: ResourcePoolTransaction | None = None,
    ) -> ResourcePool | None:
        """Atomically commit this scope and its matching resource transaction.

        When a transaction is supplied, its complete delta must exactly match
        ``public_outputs``.  This prevents callers from committing graph nodes
        without their resources, or resources whose producers were rolled back.
        """

        self._require_open()
        outputs = dict(public_outputs or {})
        try:
            self.validate_public_outputs(outputs)
            committed_pool: ResourcePool | None = None
            if resource_transaction is None:
                if outputs:
                    raise GraphBuilderError(
                        "public outputs require a resource-pool transaction"
                    )
            else:
                changed = {
                    name: resource
                    for name, resource in resource_transaction.staged.resources.items()
                    if resource_transaction.base.resources.get(name) != resource
                }
                if set(changed) != set(outputs):
                    raise GraphBuilderError(
                        "resource transaction delta must exactly match public outputs"
                    )
                for name, value in outputs.items():
                    resource = changed[name]
                    if resource.value != value:
                        raise GraphBuilderError(
                            f"resource {name!r} does not contain its public output"
                        )
                    if resource.source_feature_id != self.feature_id:
                        raise GraphBuilderError(
                            f"resource {name!r} has the wrong source feature"
                        )
                committed_pool = resource_transaction.commit()
            self._parent._commit(self)
        except Exception:
            self.rollback()
            raise
        self._closed = True
        self._committed = True
        return committed_pool

    def commit_emission(
        self,
        emission: FeatureEmission,
        resource_transaction: ResourcePoolTransaction,
    ) -> ResourcePool:
        result = self.commit(
            public_outputs=emission.outputs,
            resource_transaction=resource_transaction,
        )
        assert result is not None
        return result

    def rollback(self) -> None:
        if self._closed:
            return
        self._parent._rollback(self)
        self._closed = True


__all__ = [
    "GraphBuilderError",
    "InputEdgeEvidence",
    "PromptGraphBuilder",
    "ScopedGraphBuilder",
]
