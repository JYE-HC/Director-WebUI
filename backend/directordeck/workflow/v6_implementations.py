from __future__ import annotations

"""Bundle-6 implementation identities and exact route matrix."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..schemas import StrictModel
from .contracts import Resource, ScopedGraphBuilderProtocol
from .feature_config import LoraConfigV1, V6ConfigContext
from .feature_definitions import BUNDLE6_FEATURE_DEFINITIONS
from .interpreters.comfy_kitchen_attention import resolve_comfy_kitchen_attention
from .interpreters.execution_strategy import resolve_execution_strategy
from .interpreters.v6_native import emit_v6_native_feature
from .v6_registry import (
    CarrierKind,
    FeatureResolutionV6,
    ResolvedEarlierFeatures,
    ResolvedFeatureImplementation,
    Responsibility,
    V6FeatureRegistry,
    bundle6_feature_registrations,
    implementation_matrix,
)


@dataclass(frozen=True, slots=True)
class NativeFeatureImplementationV6:
    feature_id: str
    backend: str
    carrier_kind: CarrierKind
    responsibility: Responsibility
    feature_version: int = 1
    implementation_version: int = 1

    @property
    def implementation_id(self) -> str:
        return f"directordeck.v6.{self.feature_id}.{self.backend}"

    def resolve(
        self,
        config: StrictModel,
        dependencies: ResolvedEarlierFeatures,
        context: V6ConfigContext,
    ) -> FeatureResolutionV6:
        if self.feature_id == "comfy_kitchen_attention":
            return resolve_comfy_kitchen_attention(
                backend=self.backend,
                implementation_id=self.implementation_id,
                implementation_version=self.implementation_version,
            )
        if self.feature_id == "execution_strategy":
            return resolve_execution_strategy(
                config=config,  # type: ignore[arg-type]
                dependencies=dependencies,
                context=context,
                implementation_id=self.implementation_id,
                implementation_version=self.implementation_version,
            )
        lora = LoraConfigV1.model_validate(config) if self.feature_id == "lora" else None
        return FeatureResolutionV6(
            implementation=ResolvedFeatureImplementation(
                implementation_id=self.implementation_id,
                implementation_version=self.implementation_version,
                carrier_kind=self.carrier_kind,
                responsibility=self.responsibility,
                class_types=(),
                binding_key=lora.adapter_id if lora is not None else None,
            ),
            details={
                "backend": context.backend,
                "family": context.family,
                "config": config.model_dump(mode="json"),
            },
        )

    def execution_identity(
        self,
        resolution: FeatureResolutionV6,
        _context: V6ConfigContext,
    ) -> dict[str, Any]:
        return {
            "implementation": resolution.implementation.model_dump(mode="json"),
            "details": dict(resolution.details),
        }

    def runtime_pool_identity(
        self,
        resolution: FeatureResolutionV6,
        _context: V6ConfigContext,
    ) -> dict[str, Any] | None:
        if self.feature_id != "execution_strategy" or self.backend != "raylight":
            return None
        descriptor = resolution.details.get("runtime_descriptor")
        return dict(descriptor) if isinstance(descriptor, Mapping) else None

    def emit(
        self,
        builder: ScopedGraphBuilderProtocol,
        inputs: Mapping[str, Resource],
        config: StrictModel,
        resolution: FeatureResolutionV6,
        dependencies: ResolvedEarlierFeatures,
        context: V6ConfigContext,
    ) -> Any:
        return emit_v6_native_feature(
            self.feature_id,
            self.backend,
            builder,
            inputs,
            config,
            resolution,
            dependencies,
            context,
        )


def _adapter(feature_id: str, backend: str) -> NativeFeatureImplementationV6:
    director = backend == "raylight" and feature_id in {
        "diffusion_model",
        "execution_strategy",
        "lora",
        "sigma_schedule",
        "sampling_pipeline",
        "comfy_kitchen_attention",
    }
    carrier: CarrierKind = (
        "private_subgraph"
        if feature_id in {
            "auxiliary_models",
            "multimodal_conditioning",
            "continuity",
            "sampling_pipeline",
            "video_decode",
            "audio_output",
        }
        else "comfy_node"
    )
    if backend == "raylight" and feature_id in {
        "execution_strategy",
        "comfy_kitchen_attention",
    }:
        carrier = "director_runtime"
    return NativeFeatureImplementationV6(
        feature_id=feature_id,
        backend=backend,
        carrier_kind=carrier,
        responsibility="director" if director else "host_user",
    )


V6_FEATURE_REGISTRY = V6FeatureRegistry(
    bundle6_feature_registrations(),
    implementation_matrix(
        {
            (definition.id, backend): _adapter(definition.id, backend)
            for definition in BUNDLE6_FEATURE_DEFINITIONS
            for backend in ("standard", "raylight")
        }
    ),
)


__all__ = ["NativeFeatureImplementationV6", "V6_FEATURE_REGISTRY"]
