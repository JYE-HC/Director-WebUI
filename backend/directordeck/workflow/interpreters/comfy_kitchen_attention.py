from __future__ import annotations

"""The sole Bundle-6 attention feature carriers."""

from ...schemas import StrictModel
from ..contracts import EdgeRef, FeatureEmission, Resource, ScopedGraphBuilderProtocol
from ..v6_registry import (
    FeatureResolutionV6,
    ResolvedFeatureImplementation,
)


def resolve_comfy_kitchen_attention(
    *,
    backend: str,
    implementation_id: str,
    implementation_version: int,
) -> FeatureResolutionV6:
    if backend == "standard":
        carrier = "comfy_node"
        responsibility = "host_user"
        class_types = ("ModelAttentionBackend",)
        details = {
            "backend": "standard",
            "prompt_choice": "comfy kitchen attention",
        }
    elif backend == "raylight":
        carrier = "director_runtime"
        responsibility = "director"
        class_types = ()
        details = {
            "backend": "raylight",
            "initializer_attention": "COMFY_KITCHEN_INT8",
        }
    else:  # pragma: no cover - exact registry construction prevents this.
        raise ValueError(f"unsupported CK backend: {backend}")
    return FeatureResolutionV6(
        implementation=ResolvedFeatureImplementation(
            implementation_id=implementation_id,
            implementation_version=implementation_version,
            carrier_kind=carrier,
            responsibility=responsibility,
            class_types=class_types,
        ),
        details=details,
    )


def emit_standard_comfy_kitchen_attention(
    builder: ScopedGraphBuilderProtocol,
    inputs: dict[str, Resource],
    _config: StrictModel,
) -> FeatureEmission:
    resource = inputs.get("model")
    if resource is None or not isinstance(resource.value, EdgeRef):
        raise TypeError("Standard CK requires one MODEL edge")
    node_id = builder.add_node(
        "ModelAttentionBackend",
        {
            "model": resource.value,
            "attention": "comfy kitchen attention",
        },
    )
    return FeatureEmission(outputs={"model": builder.edge(node_id, 0)})


def emit_raylight_comfy_kitchen_attention() -> FeatureEmission:
    """Ray CK is a typed dependency consumed by execution_strategy."""

    return FeatureEmission()


__all__ = [
    "emit_raylight_comfy_kitchen_attention",
    "emit_standard_comfy_kitchen_attention",
    "resolve_comfy_kitchen_attention",
]
