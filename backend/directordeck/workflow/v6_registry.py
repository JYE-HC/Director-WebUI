from __future__ import annotations

"""Exact Bundle-6 semantic/config/implementation registry."""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from pydantic import Field, model_validator

from ..schemas import LoraFeatureParams, StrictModel
from .contracts import (
    Backend,
    ContractModel,
    Identifier,
    JsonObject,
    ModelFamily,
    PositiveVersion,
    ResolvedFeatureImplementation,
)
from .feature_config import (
    ComfyKitchenAttentionParamsV1,
    EffectiveFeatureUse,
    EmptyFeatureParams,
    FeatureConfigResolver,
    V6_CONFIG_RESOLVERS,
    default_lora_params,
)
from .feature_definitions import (
    BUNDLE6_FEATURE_DEFINITIONS,
    FeatureDefinition,
)
from .templates_v6 import FeatureDependency, FeatureUse, SegmentTemplateV6


CarrierKind = Literal[
    "host_runtime",
    "comfy_node",
    "private_subgraph",
    "director_runtime",
]
Responsibility = Literal["director", "host_user"]


class FeatureResolutionV6(ContractModel):
    implementation: ResolvedFeatureImplementation
    details: JsonObject = Field(default_factory=dict)


class FeatureImplementation(Protocol):
    feature_id: str
    feature_version: int
    implementation_id: str
    implementation_version: int
    carrier_kind: CarrierKind
    responsibility: Responsibility

    def resolve(
        self,
        config: StrictModel,
        dependencies: "ResolvedEarlierFeatures",
        context: Any,
    ) -> FeatureResolutionV6: ...

    def execution_identity(
        self,
        resolution: FeatureResolutionV6,
        context: Any,
    ) -> JsonObject: ...

    def runtime_pool_identity(
        self,
        resolution: FeatureResolutionV6,
        context: Any,
    ) -> JsonObject | None: ...

    def emit(
        self,
        builder: Any,
        inputs: Any,
        config: StrictModel,
        resolution: FeatureResolutionV6,
        dependencies: "ResolvedEarlierFeatures",
        context: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class FeatureRegistration:
    definition: FeatureDefinition
    config_model: type[StrictModel]
    default_factory: Callable[[], StrictModel]
    resolver: FeatureConfigResolver

    def __post_init__(self) -> None:
        default = self.default_factory()
        if not isinstance(default, self.config_model):
            raise TypeError("feature default factory returned the wrong model")


@dataclass(frozen=True, slots=True)
class ImplementationRegistration:
    feature_id: str
    feature_version: int
    backend: Backend
    family: ModelFamily
    implementation: FeatureImplementation

    def __post_init__(self) -> None:
        identity = (
            self.implementation.feature_id,
            self.implementation.feature_version,
        )
        if identity != (self.feature_id, self.feature_version):
            raise ValueError("implementation registration identity drifted")


class V6RegistryError(ValueError):
    pass


class ResolvedEarlierFeatures:
    __slots__ = ("_dependencies", "_resolutions")

    def __init__(
        self,
        dependencies: tuple[FeatureDependency, ...],
        resolutions: dict[str, FeatureResolutionV6 | None],
    ) -> None:
        self._dependencies = {item.feature_id: item for item in dependencies}
        self._resolutions = resolutions

    def _declared(self, feature_id: str) -> FeatureDependency:
        try:
            return self._dependencies[feature_id]
        except KeyError as exc:
            raise V6RegistryError(
                f"feature dependency was not declared: {feature_id}"
            ) from exc

    def required(self, feature_id: str) -> FeatureResolutionV6:
        dependency = self._declared(feature_id)
        value = self._resolutions.get(feature_id)
        if value is None:
            raise V6RegistryError(
                f"required feature dependency is inactive: {feature_id}"
            )
        if not dependency.required:
            # Optional declarations may still be read through required() when
            # the concrete implementation genuinely needs the active branch.
            return value
        return value

    def optional(self, feature_id: str) -> FeatureResolutionV6 | None:
        self._declared(feature_id)
        return self._resolutions.get(feature_id)


class V6FeatureRegistry:
    __slots__ = ("_features", "_implementations")

    def __init__(
        self,
        features: tuple[FeatureRegistration, ...],
        implementations: tuple[ImplementationRegistration, ...],
    ) -> None:
        feature_map: dict[tuple[str, int], FeatureRegistration] = {}
        for registration in features:
            key = (registration.definition.id, registration.definition.version)
            if key in feature_map:
                raise V6RegistryError(f"duplicate feature registration: {key}")
            feature_map[key] = registration
        implementation_map: dict[
            tuple[str, int, Backend, ModelFamily],
            FeatureImplementation,
        ] = {}
        for registration in implementations:
            key = (
                registration.feature_id,
                registration.feature_version,
                registration.backend,
                registration.family,
            )
            if key in implementation_map:
                raise V6RegistryError(f"duplicate implementation registration: {key}")
            if key[:2] not in feature_map:
                raise V6RegistryError("implementation has no semantic registration")
            implementation_map[key] = registration.implementation
        self._features = feature_map
        self._implementations = implementation_map

    @property
    def feature_identities(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._features)

    @property
    def implementation_identities(
        self,
    ) -> tuple[tuple[str, int, Backend, ModelFamily], ...]:
        return tuple(self._implementations)

    def require_feature(self, feature_id: str, version: int) -> FeatureRegistration:
        try:
            return self._features[(feature_id, version)]
        except KeyError as exc:
            raise V6RegistryError(
                f"unknown exact feature registration: {feature_id}@{version}"
            ) from exc

    def require_implementation(
        self,
        feature_id: str,
        version: int,
        backend: Backend,
        family: ModelFamily,
    ) -> FeatureImplementation:
        try:
            return self._implementations[(feature_id, version, backend, family)]
        except KeyError as exc:
            raise V6RegistryError(
                "unknown exact feature implementation: "
                f"{feature_id}@{version}/{backend}/{family}"
            ) from exc

    def validate_template(self, template: SegmentTemplateV6) -> None:
        positions = {entry.feature_id: index for index, entry in enumerate(template.entries)}
        for index, use in enumerate(template.entries):
            self.require_feature(use.feature_id, use.feature_version)
            for dependency in use.dependencies:
                position = positions.get(dependency.feature_id)
                if position is None or position >= index:
                    raise V6RegistryError("dependency must reference an earlier feature")
            for family in ("fl2va", "ref2va"):
                self.require_implementation(
                    use.feature_id,
                    use.feature_version,
                    template.backend,
                    family,
                )


def _empty_default() -> EmptyFeatureParams:
    return EmptyFeatureParams()


def _ck_default() -> ComfyKitchenAttentionParamsV1:
    return ComfyKitchenAttentionParamsV1()


def bundle6_feature_registrations() -> tuple[FeatureRegistration, ...]:
    registrations: list[FeatureRegistration] = []
    for definition in BUNDLE6_FEATURE_DEFINITIONS:
        if definition.id == "lora":
            model: type[StrictModel] = LoraFeatureParams
            factory: Callable[[], StrictModel] = default_lora_params
        elif definition.id == "comfy_kitchen_attention":
            model = ComfyKitchenAttentionParamsV1
            factory = _ck_default
        else:
            model = EmptyFeatureParams
            factory = _empty_default
        registrations.append(
            FeatureRegistration(
                definition=definition,
                config_model=model,
                default_factory=factory,
                resolver=V6_CONFIG_RESOLVERS[definition.id],
            )
        )
    return tuple(registrations)


def implementation_matrix(
    implementations: dict[tuple[str, Backend], FeatureImplementation],
) -> tuple[ImplementationRegistration, ...]:
    registrations: list[ImplementationRegistration] = []
    for definition in BUNDLE6_FEATURE_DEFINITIONS:
        for backend in definition.backends:
            implementation = implementations[(definition.id, backend)]
            for family in definition.families:
                registrations.append(
                    ImplementationRegistration(
                        feature_id=definition.id,
                        feature_version=definition.version,
                        backend=backend,
                        family=family,
                        implementation=implementation,
                    )
                )
    return tuple(registrations)


__all__ = [
    "CarrierKind",
    "FeatureImplementation",
    "FeatureRegistration",
    "FeatureResolutionV6",
    "ImplementationRegistration",
    "ResolvedEarlierFeatures",
    "ResolvedFeatureImplementation",
    "Responsibility",
    "V6FeatureRegistry",
    "V6RegistryError",
    "bundle6_feature_registrations",
    "implementation_matrix",
]
