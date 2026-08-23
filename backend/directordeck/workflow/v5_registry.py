from __future__ import annotations

"""Exact bundle-5 interpreter registry and bounded extension seam.

Existing graph fragments are delegated without copying their implementation.
New Stage-8 switches are registered fail-closed until their reviewed runtime
effect contracts are installed; a disabled switch never calls them.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..native_templates import NativeTemplateError
from .contracts import (
    BoundedJsonValue,
    CapabilitySet,
    EdgeRef,
    FeatureEmission,
    FeatureResolution,
    Resource,
    ResolvedImplementationIdentity,
    ScopedGraphBuilderProtocol,
)
from .effective_features import (
    AttentionBackendOverrideParams,
    EmptyFeatureParams,
    RayLightPoolIntentParams,
    feature_parameter_model,
)
from .interpreters import V4BuiltinInterpreter, builtin_interpreter_map
from .node_contracts import require_current_node_contract
from .registry import FeatureInterpreterRegistry, ValidatedFeatureTemplate
from .templates import V5_TEMPLATE_BUNDLE


_STRICT_ATTENTION_CLASS = "DirectorStrictModelAttentionBackend"
_STRICT_H3_LOW_VRAM_CLASS = "DirectorStrictH3LowVramSagePatch"


def _current_node_identity(
    feature_id: str,
    class_type: str,
) -> ResolvedImplementationIdentity:
    contract = require_current_node_contract(class_type)
    return ResolvedImplementationIdentity(
        role="node",
        class_type=class_type,
        implementation_id=contract.contract_id,
        semantic_version=contract.semantic_version,
        runtime_fingerprint=contract.supported_runtime_fingerprints[0],
        binding_key=f"{feature_id}.{class_type}",
    )


def _require_standard_v5_context(ctx: Any) -> None:
    if getattr(ctx, "template_bundle_version", None) != 5:
        raise NativeTemplateError("strict feature requires template bundle 5")
    if getattr(ctx, "backend", None) != "standard":
        raise NativeTemplateError("strict feature requires the Standard backend")
    if getattr(ctx, "family", None) not in {"fl2va", "ref2va"}:
        raise NativeTemplateError("strict feature requires an H3 model family")
    if getattr(ctx, "binding", None) is None:
        raise NativeTemplateError("strict feature requires an exact model binding")


def _model_input(
    inputs: Mapping[str, Resource],
) -> EdgeRef:
    if set(inputs) != {"model"}:
        raise NativeTemplateError("strict model feature requires exactly one model input")
    resource = inputs["model"]
    if resource.type != "MODEL" or not isinstance(resource.value, EdgeRef):
        raise NativeTemplateError("strict model feature received an invalid MODEL edge")
    return resource.value


@dataclass(frozen=True, slots=True)
class _AttentionBackendInterpreter:
    id: str = "attention_backend_override"
    version: int = 1

    def validate_params(self, params: BaseModel, ctx: Any) -> None:
        if not isinstance(params, AttentionBackendOverrideParams):
            raise TypeError(
                "attention_backend_override@1 requires "
                "AttentionBackendOverrideParams"
            )
        _require_standard_v5_context(ctx)

    def resolve(self, params: BaseModel, ctx: Any) -> FeatureResolution:
        self.validate_params(params, ctx)
        assert isinstance(params, AttentionBackendOverrideParams)
        return FeatureResolution(
            state="active",
            implementations=(
                _current_node_identity(self.id, _STRICT_ATTENTION_CLASS),
            ),
            resolution_details={
                "source": "bundle5_strict_attention_backend",
                "mode": params.mode,
                "backend": ctx.backend,
                "family": ctx.family,
            },
        )

    def _require_resolution(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> AttentionBackendOverrideParams:
        self.validate_params(params, ctx)
        if resolution != self.resolve(params, ctx):
            raise NativeTemplateError(
                "attention backend resolution does not match params/context"
            )
        assert isinstance(params, AttentionBackendOverrideParams)
        return params

    def required_capabilities(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> CapabilitySet:
        exact = self._require_resolution(params, ctx, resolution)
        from ..capabilities.evaluator import (
            STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
            STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE,
            contextual_runtime_capability_id,
        )

        probe = (
            STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE
            if exact.mode == "ck_int8"
            else STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE
        )
        return CapabilitySet(
            ids=(
                f"node.{_STRICT_ATTENTION_CLASS}",
                contextual_runtime_capability_id(probe, ctx),
            )
        )

    def cache_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue:
        exact = self._require_resolution(params, ctx, resolution)
        return {
            "authority": "resolved_feature_execution_identity",
            "feature_id": self.id,
            "feature_version": self.version,
            "backend": ctx.backend,
            "family": ctx.family,
            "mode": exact.mode,
            "implementation": resolution.implementations[0].model_dump(
                mode="json"
            ),
        }

    def runtime_pool_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> None:
        self._require_resolution(params, ctx, resolution)
        return None

    def emit(
        self,
        builder: ScopedGraphBuilderProtocol,
        inputs: Mapping[str, Resource],
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> FeatureEmission:
        exact = self._require_resolution(params, ctx, resolution)
        node_id = builder.add_node(
            _STRICT_ATTENTION_CLASS,
            {"model": _model_input(inputs), "mode": exact.mode},
        )
        return FeatureEmission(
            outputs={"model": builder.edge(node_id, 0)},
            emission_details={
                "source": "bundle5_strict_attention_backend",
                "mode": exact.mode,
            },
        )


@dataclass(frozen=True, slots=True)
class _H3LowVramAttentionInterpreter:
    id: str = "h3_low_vram_attention"
    version: int = 1

    def validate_params(self, params: BaseModel, ctx: Any) -> None:
        if not isinstance(params, EmptyFeatureParams):
            raise TypeError(
                "h3_low_vram_attention@1 requires EmptyFeatureParams"
            )
        _require_standard_v5_context(ctx)

    def resolve(self, params: BaseModel, ctx: Any) -> FeatureResolution:
        self.validate_params(params, ctx)
        return FeatureResolution(
            state="active",
            implementations=(
                _current_node_identity(self.id, _STRICT_H3_LOW_VRAM_CLASS),
            ),
            resolution_details={
                "source": "bundle5_strict_h3_low_vram_attention",
                "backend": ctx.backend,
                "family": ctx.family,
            },
        )

    def _require_resolution(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> None:
        self.validate_params(params, ctx)
        if resolution != self.resolve(params, ctx):
            raise NativeTemplateError(
                "H3 low-VRAM resolution does not match params/context"
            )

    def required_capabilities(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> CapabilitySet:
        self._require_resolution(params, ctx, resolution)
        from ..capabilities.evaluator import (
            STRICT_H3_SAGE_RUNTIME_PROBE,
            contextual_runtime_capability_id,
        )

        return CapabilitySet(
            ids=(
                f"node.{_STRICT_H3_LOW_VRAM_CLASS}",
                contextual_runtime_capability_id(
                    STRICT_H3_SAGE_RUNTIME_PROBE,
                    ctx,
                ),
            )
        )

    def cache_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue:
        self._require_resolution(params, ctx, resolution)
        return {
            "authority": "resolved_feature_execution_identity",
            "feature_id": self.id,
            "feature_version": self.version,
            "backend": ctx.backend,
            "family": ctx.family,
            "implementation": resolution.implementations[0].model_dump(
                mode="json"
            ),
        }

    def runtime_pool_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> None:
        self._require_resolution(params, ctx, resolution)
        return None

    def emit(
        self,
        builder: ScopedGraphBuilderProtocol,
        inputs: Mapping[str, Resource],
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> FeatureEmission:
        self._require_resolution(params, ctx, resolution)
        node_id = builder.add_node(
            _STRICT_H3_LOW_VRAM_CLASS,
            {"model": _model_input(inputs)},
        )
        return FeatureEmission(
            outputs={"model": builder.edge(node_id, 0)},
            emission_details={
                "source": "bundle5_strict_h3_low_vram_attention",
            },
        )


@dataclass(frozen=True, slots=True)
class _V5BuiltinCompatibilityInterpreter:
    id: str
    version: int
    delegate: V4BuiltinInterpreter

    def _legacy_params(self, params: BaseModel) -> BaseModel:
        expected = feature_parameter_model(self.id, self.version)
        if not isinstance(params, expected):
            raise TypeError(
                f"{self.id}@{self.version} requires {expected.__name__}"
            )
        return self.delegate_params()

    @staticmethod
    def delegate_params() -> BaseModel:
        from .interpreters import V4BuiltinParams

        return V4BuiltinParams()

    def validate_params(self, params: BaseModel, ctx: Any) -> None:
        self.delegate.validate_params(self._legacy_params(params), ctx)

    def resolve(self, params: BaseModel, ctx: Any) -> FeatureResolution:
        return self.delegate.resolve(self._legacy_params(params), ctx)

    def required_capabilities(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> CapabilitySet:
        return self.delegate.required_capabilities(
            self._legacy_params(params),
            ctx,
            resolution,
        )

    def cache_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue:
        # Preserve the frozen adapter's established effective-identity shape
        # for unchanged active features while making this interpreter return
        # the exact value consumed by the bundle-5 adapter.  Merely adding
        # disabled descriptors therefore cannot stale Standard takes.
        self.delegate.cache_identity(
            self._legacy_params(params),
            ctx,
            resolution,
        )
        identity: dict[str, Any] = {
            "authority": "resolved_feature_execution_identity",
            "feature_id": self.id,
            "backend": ctx.backend,
            "family": ctx.family,
            "implementations": [
                implementation.runtime_fingerprint
                for implementation in resolution.implementations
            ],
        }
        if self.id == "lora":
            details = resolution.resolution_details
            binding = details.get("binding")
            identity.update(
                adapter_id=details.get("adapter_id"),
                binding=(dict(binding) if isinstance(binding, Mapping) else binding),
                strength=details.get("strength"),
            )
        if self.id == "raylight_pool_intent":
            assert isinstance(params, RayLightPoolIntentParams)
            identity["attention"] = params.attention
        return identity

    def runtime_pool_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue | None:
        identity = self.delegate.runtime_pool_identity(
            self._legacy_params(params),
            ctx,
            resolution,
        )
        if self.id != "raylight_pool_intent":
            return identity
        assert isinstance(params, RayLightPoolIntentParams)
        return {"compatibility": identity, "attention": params.attention}

    def emit(
        self,
        builder: ScopedGraphBuilderProtocol,
        inputs: Mapping[str, Resource],
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> FeatureEmission:
        return self.delegate.emit(
            builder,
            inputs,
            self._legacy_params(params),
            ctx,
            resolution,
        )


@dataclass(frozen=True, slots=True)
class _UnavailableV5Interpreter:
    id: str
    version: int

    def validate_params(self, params: BaseModel, ctx: Any) -> None:
        del ctx
        expected = feature_parameter_model(self.id, self.version)
        if not isinstance(params, expected):
            raise TypeError(
                f"{self.id}@{self.version} requires {expected.__name__}"
            )

    def _unavailable(self, params: BaseModel, ctx: Any) -> None:
        self.validate_params(params, ctx)
        raise NativeTemplateError(
            f"exact feature implementation is unavailable: {self.id}@{self.version}"
        )

    def resolve(self, params: BaseModel, ctx: Any) -> FeatureResolution:
        self._unavailable(params, ctx)
        raise AssertionError("unreachable")

    def required_capabilities(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> CapabilitySet:
        del resolution
        self._unavailable(params, ctx)
        raise AssertionError("unreachable")

    def cache_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue:
        del resolution
        self._unavailable(params, ctx)
        raise AssertionError("unreachable")

    def runtime_pool_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue | None:
        del resolution
        self._unavailable(params, ctx)
        raise AssertionError("unreachable")

    def emit(
        self,
        builder: ScopedGraphBuilderProtocol,
        inputs: Mapping[str, Resource],
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> FeatureEmission:
        del builder, inputs, resolution
        self._unavailable(params, ctx)
        raise AssertionError("unreachable")


def _build_v5_registry() -> FeatureInterpreterRegistry:
    legacy = builtin_interpreter_map()
    identities = OrderedIdentity.from_bundle()
    registry = FeatureInterpreterRegistry()
    for feature_id, version in identities:
        delegate = legacy.get(feature_id)
        if (feature_id, version) == ("attention_backend_override", 1):
            interpreter: Any = _AttentionBackendInterpreter()
        elif (feature_id, version) == ("h3_low_vram_attention", 1):
            interpreter = _H3LowVramAttentionInterpreter()
        elif delegate is not None:
            interpreter = _V5BuiltinCompatibilityInterpreter(
                id=feature_id,
                version=version,
                delegate=delegate,
            )
        else:
            interpreter = _UnavailableV5Interpreter(
                id=feature_id,
                version=version,
            )
        registry.register(interpreter)
    return registry.freeze()


class OrderedIdentity:
    @staticmethod
    def from_bundle() -> tuple[tuple[str, int], ...]:
        return tuple(
            dict.fromkeys(
                (entry.id, entry.version)
                for template in (
                    V5_TEMPLATE_BUNDLE.segment_templates.standard,
                    V5_TEMPLATE_BUNDLE.segment_templates.raylight,
                )
                for entry in template.entries
            )
        )


V5_INTERPRETER_REGISTRY = _build_v5_registry()
V5_VALIDATED_TEMPLATES: dict[str, ValidatedFeatureTemplate] = {
    template.id: V5_INTERPRETER_REGISTRY.validate_template(template)
    for template in (
        V5_TEMPLATE_BUNDLE.segment_templates.standard,
        V5_TEMPLATE_BUNDLE.segment_templates.raylight,
    )
}


__all__ = [
    "V5_INTERPRETER_REGISTRY",
    "V5_VALIDATED_TEMPLATES",
]
