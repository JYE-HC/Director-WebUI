from __future__ import annotations

"""Small bridge between the frozen native graph helpers and Stage-2 builders."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

from ..contracts import ScopedGraphBuilderProtocol


NativeEdge: TypeAlias = list[Any]


class NativeNodeEmitter(Protocol):
    """The ordered node append operation used by native fragment emitters."""

    def add(self, class_type: str, **inputs: Any) -> str: ...


@dataclass(frozen=True, slots=True)
class ScopedBuilderEmitter:
    """Adapt the Stage-1 builder protocol without weakening its ownership."""

    builder: ScopedGraphBuilderProtocol

    def add(self, class_type: str, **inputs: Any) -> str:
        return self.builder.add_node(
            class_type,
            {name: self._typed_input(value) for name, value in inputs.items()},
        )

    def _typed_input(self, value: Any) -> Any:
        """Translate every legacy JSON edge before the strict builder sees it."""

        if (
            isinstance(value, (list, tuple))
            and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], int)
            and not isinstance(value[1], bool)
        ):
            # ``builder.edge`` owns both unknown-node and terminal rejection.
            return self.builder.edge(value[0], value[1])
        if isinstance(value, Mapping):
            return {
                name: self._typed_input(item)
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [self._typed_input(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._typed_input(item) for item in value)
        return value


def edge(node_id: str, output_slot: int = 0) -> NativeEdge:
    return [node_id, output_slot]
