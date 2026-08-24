from __future__ import annotations

"""Static feature-catalog projection and identity."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from ..workflow.contracts import (
    Backend,
    CapabilitySet,
    ContractModel,
    FeatureResolution,
    FeatureTemplateEntry,
    HostCapabilitySnapshot,
    Identifier,
    JsonObject,
    ModelFamily,
    NodeContractRegistry,
    PositiveVersion,
    ResolvedImplementationIdentity,
    Sha256Digest,
    TemplateBundle,
    canonical_sha256,
)
from ..workflow.interpreters import (
    builtin_implementation_identity,
    builtin_required_capability_ids,
    catalog_implementation_alternatives,
)
from ..config_manager import get_directordeck_config
from ..workflow.lora_factory import standard_lora_adapters
from ..workflow.node_contracts import (
    V4_NODE_CONTRACT_REGISTRY,
    V5_NODE_CONTRACT_REGISTRY,
)
from ..workflow.templates import V4_TEMPLATE_BUNDLE
from .evaluator import (
    CapabilityEvaluation,
    CapabilityEvaluator,
    CapabilityReason,
    resolution_adapter_fingerprint,
)


@dataclass(slots=True)
class _MergedCatalogEntry:
    entry: FeatureTemplateEntry
    backends: list[Backend]


@dataclass(frozen=True, slots=True)
class _CatalogCompileContext:
    backend: Backend
    family: str
    template_bundle_version: int


class CatalogAvailability(ContractModel):
    state: Literal["available", "unavailable", "conditional"]
    reasons: tuple[CapabilityReason, ...] = ()

    @model_validator(mode="after")
    def _validate_reasons(self) -> "CatalogAvailability":
        if self.state == "unavailable" and not self.reasons:
            raise ValueError("unavailable catalog entry requires a reason")
        if self.state == "available" and self.reasons:
            raise ValueError("available catalog entry cannot contain reasons")
        return self


class CatalogAdapterConfigurationOption(ContractModel):
    id: Identifier
    type: Literal["boolean"]
    label: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=512)
    default: bool


class CatalogAdapterOption(ContractModel):
    """One configured loader choice plus advisory host observation."""

    adapter_id: Identifier
    display_name: str = Field(min_length=1, max_length=128)
    class_type: str = Field(min_length=1, max_length=256)
    is_default: bool
    backend: Literal["standard"]
    supported_families: tuple[ModelFamily, ...]
    configuration_options: tuple[CatalogAdapterConfigurationOption, ...] = ()
    adapter_fingerprint: Sha256Digest
    capability: CapabilityEvaluation


class FeatureCatalogEntry(ContractModel):
    id: Identifier
    version: PositiveVersion
    title: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=4_096)
    mode: Literal["switch", "needed"]
    layer: Literal["graph"]
    scopes: tuple[Identifier, ...]
    params_schema: JsonObject = Field(default_factory=dict)
    defaults: JsonObject = Field(default_factory=dict)
    backends: tuple[Backend, ...]
    availability: CatalogAvailability
    adapter_options: tuple[CatalogAdapterOption, ...] = ()
    ui: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_adapter_options(self) -> "FeatureCatalogEntry":
        identities = tuple(
            (option.backend, option.adapter_id)
            for option in self.adapter_options
        )
        if len(identities) != len(set(identities)):
            raise ValueError("catalog adapter options must be unique")
        expects_standard_options = (
            self.id == "lora" and "standard" in self.backends
        )
        if expects_standard_options != bool(self.adapter_options):
            raise ValueError(
                "only a Standard LoRA catalog entry may expose adapter options"
            )
        return self


class FeatureCatalog(ContractModel):
    template_bundle_version: PositiveVersion
    host_capability_revision: Sha256Digest
    entries: tuple[FeatureCatalogEntry, ...]


def feature_catalog_etag(
    *,
    template_bundle_version: int,
    host_capability_revision: str,
) -> str:
    """Return the opaque strong ETag value without HTTP quote characters."""

    return canonical_sha256(
        {
            "template_bundle_version": template_bundle_version,
            "host_capability_revision": host_capability_revision,
        }
    )


def quote_feature_catalog_etag(etag: str) -> str:
    """Format a validated digest as a strong HTTP ETag header value."""

    if not (
        isinstance(etag, str)
        and etag.startswith("sha256:")
        and len(etag) == 71
        and all(character in "0123456789abcdef" for character in etag[7:])
    ):
        raise ValueError("catalog ETag must be a canonical sha256 digest")
    return f'"{etag}"'


def _static_availability(
    *,
    feature_id: str,
    backends: tuple[Backend, ...],
    snapshot: HostCapabilitySnapshot,
    evaluator: CapabilityEvaluator,
    template_bundle_version: int,
    node_contract_registry: NodeContractRegistry,
) -> CatalogAvailability:
    failures: list[CapabilityReason] = []
    for backend in backends:
        try:
            alternatives = catalog_implementation_alternatives(
                feature_id,
                backend,
            )
        except KeyError:
            failures.append(
                CapabilityReason(
                    code="feature_interpreter_unavailable",
                    feature_id=feature_id,
                    segment_id=None,
                    unit_id=None,
                    backend=backend,
                    rule="exact_feature_interpreter_registration",
                    message="The feature implementation is not installed in this build.",
                    remediation="Install a DirectorDeck release that provides this exact feature version.",
                    safe_details={"template_bundle_version": template_bundle_version},
                )
            )
            continue
        for family, class_types in alternatives:
            implementations: list[ResolvedImplementationIdentity] = []
            for index, class_type in enumerate(class_types):
                contract = node_contract_registry.require(class_type)
                implementations.append(
                    ResolvedImplementationIdentity(
                        role="node",
                        class_type=class_type,
                        implementation_id=contract.contract_id,
                        semantic_version=contract.semantic_version,
                        runtime_fingerprint=(
                            contract.supported_runtime_fingerprints[0]
                        ),
                        binding_key=(
                            f"catalog.{index}."
                            + class_type.replace(" ", "_")
                        ),
                    )
                )
            evaluation = evaluator.evaluate(
                feature_id=feature_id,
                ctx=_CatalogCompileContext(
                    backend=backend,
                    family=family,
                    template_bundle_version=template_bundle_version,
                ),
                resolution=FeatureResolution(
                    state="active",
                    implementations=tuple(implementations),
                ),
                required_capabilities=CapabilitySet(
                    ids=builtin_required_capability_ids(
                        feature_id=feature_id,
                        class_types=class_types,
                        # Catalog asks whether *any* supported context can
                        # work. Timeline assembly is contextual, so the
                        # conservative static alternative is the native-take
                        # path that needs only ffprobe.
                        timeline_assembly_required=False,
                    )
                ),
                snapshot=snapshot,
                readiness=None,
            )
            if evaluation.available:
                return CatalogAvailability(
                    state="conditional",
                    reasons=(
                        CapabilityReason(
                            code="context_required",
                            feature_id=feature_id,
                            segment_id=None,
                            unit_id=None,
                            backend=None,
                            rule="contextual_resolution",
                            message="Availability depends on the selected model and segment context.",
                            remediation="Run feature preflight for the current timeline.",
                            safe_details={},
                        ),
                    ),
                )
            failures.extend(evaluation.reasons)

    unique_failures: list[CapabilityReason] = []
    seen: set[str] = set()
    for reason in failures:
        identity = reason.model_dump_json()
        if identity in seen:
            continue
        seen.add(identity)
        unique_failures.append(reason)
    return CatalogAvailability(
        state="unavailable",
        reasons=tuple(unique_failures),
    )


def _standard_lora_adapter_options(
    *,
    snapshot: HostCapabilitySnapshot,
    evaluator: CapabilityEvaluator,
    template_bundle_version: int,
    feature_version: int,
) -> tuple[CatalogAdapterOption, ...]:
    """Project Standard registry entries through the sole live evaluator.

    This projection intentionally excludes mutable mapping records and model
    inventory. It reports whether the mapped class_type is present; it does
    not approve or certify the user's implementation. Contextual preflight
    adds creative/model observations later while reusing the same compiler
    adapter identity.
    """

    options: list[CatalogAdapterOption] = []
    default_loader_id = get_directordeck_config().default_loader_id
    for adapter in standard_lora_adapters():
        family = adapter.supported_families[0]
        context = _CatalogCompileContext(
            backend="standard",
            family=family,
            template_bundle_version=template_bundle_version,
        )
        resolution = FeatureResolution(
            state="active",
            implementations=(
                builtin_implementation_identity(
                    "lora",
                    adapter.class_type,
                ),
            ),
            resolution_details={
                "adapter_id": adapter.adapter_id,
                "source": "catalog_adapter_registry",
            },
        )
        capability = evaluator.evaluate(
            feature_id="lora",
            ctx=context,
            resolution=resolution,
            required_capabilities=CapabilitySet(
                ids=builtin_required_capability_ids(
                    feature_id="lora",
                    class_types=(adapter.class_type,),
                    timeline_assembly_required=False,
                )
            ),
            snapshot=snapshot,
            readiness=None,
        )
        options.append(
            CatalogAdapterOption(
                adapter_id=adapter.adapter_id,
                display_name=adapter.display_name,
                class_type=adapter.class_type,
                is_default=adapter.adapter_id == default_loader_id,
                backend="standard",
                supported_families=adapter.supported_families,
                configuration_options=tuple(
                    CatalogAdapterConfigurationOption(
                        id=definition.id,
                        type=definition.type,
                        label=definition.label,
                        description=definition.description,
                        default=definition.default,
                    )
                    for definition in adapter.option_definitions
                ),
                adapter_fingerprint=resolution_adapter_fingerprint(
                    feature_id="lora",
                    feature_version=feature_version,
                    ctx=context,
                    resolution=resolution,
                ),
                capability=capability,
            )
        )
    return tuple(options)


def build_feature_catalog(
    snapshot: HostCapabilitySnapshot,
    *,
    template_bundle: TemplateBundle = V4_TEMPLATE_BUNDLE,
) -> FeatureCatalog:
    """Project immutable templates and a static host snapshot into the catalog.

    Duplicate feature identities shared by Standard and RayLight templates are
    merged in first-template order.  Creative context and operational readiness
    are intentionally absent from this function.
    """

    if not isinstance(snapshot, HostCapabilitySnapshot):
        raise TypeError("catalog requires HostCapabilitySnapshot")
    if not isinstance(template_bundle, TemplateBundle):
        raise TypeError("catalog requires TemplateBundle")

    merged: OrderedDict[tuple[str, int], _MergedCatalogEntry] = OrderedDict()
    templates = (
        template_bundle.segment_templates.standard,
        template_bundle.segment_templates.raylight,
    )
    for template in templates:
        for entry in template.entries:
            key = (entry.id, entry.version)
            existing = merged.get(key)
            if existing is None:
                merged[key] = _MergedCatalogEntry(
                    entry=entry,
                    backends=list(entry.backends),
                )
                continue
            original = existing.entry
            if (
                (original.title, original.description)
                != (entry.title, entry.description)
                and key != ("lora", 1)
            ):
                raise ValueError(
                    f"catalog feature {entry.id}@{entry.version} has conflicting metadata"
                )
            for field in (
                "mode",
                "layer",
                "scopes",
                "params_schema",
                "defaults",
                "ui",
            ):
                if getattr(original, field) != getattr(entry, field):
                    raise ValueError(
                        f"catalog feature {entry.id}@{entry.version} has conflicting {field}"
                    )
            for backend in entry.backends:
                if backend not in existing.backends:
                    existing.backends.append(backend)

    entries: list[FeatureCatalogEntry] = []
    node_contract_registry = (
        V5_NODE_CONTRACT_REGISTRY
        if template_bundle.version >= 5
        else V4_NODE_CONTRACT_REGISTRY
    )
    evaluator = CapabilityEvaluator(node_contract_registry)
    for value in merged.values():
        entry = value.entry
        backends = tuple(value.backends)
        title, description = (
            (
                "LoRA",
                "Apply the selected fail-closed LoRA adapter.",
            )
            if (entry.id, entry.version) == ("lora", 1)
            and len(backends) > 1
            else (entry.title, entry.description)
        )
        entries.append(
            FeatureCatalogEntry(
                id=entry.id,
                version=entry.version,
                title=title,
                description=description,
                mode=entry.mode,
                layer=entry.layer,
                scopes=entry.scopes,
                params_schema=entry.params_schema,
                defaults=entry.defaults,
                backends=backends,
                availability=_static_availability(
                    feature_id=entry.id,
                    backends=backends,
                    snapshot=snapshot,
                    evaluator=evaluator,
                    template_bundle_version=template_bundle.version,
                    node_contract_registry=node_contract_registry,
                ),
                adapter_options=(
                    _standard_lora_adapter_options(
                        snapshot=snapshot,
                        evaluator=evaluator,
                        template_bundle_version=template_bundle.version,
                        feature_version=entry.version,
                    )
                    if entry.id == "lora" and "standard" in backends
                    else ()
                ),
                ui=entry.ui,
            )
        )

    return FeatureCatalog(
        template_bundle_version=template_bundle.version,
        host_capability_revision=snapshot.host_capability_revision(),
        entries=tuple(entries),
    )


__all__ = [
    "CatalogAdapterOption",
    "CatalogAdapterConfigurationOption",
    "CatalogAvailability",
    "FeatureCatalog",
    "FeatureCatalogEntry",
    "build_feature_catalog",
    "feature_catalog_etag",
    "quote_feature_catalog_etag",
]
