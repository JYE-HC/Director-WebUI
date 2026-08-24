from __future__ import annotations

"""Read-only contextual feature preflight for the frozen v4 authority."""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from ..schemas import RuntimeSettings, UnifiedTimelineDraft
from ..workflow.contracts import (
    Backend,
    ContractModel,
    FeatureResolution,
    FrozenMap,
    HostCapabilitySnapshot,
    ModelFamily,
    OperationalReadiness,
    PositiveVersion,
    Sha256Digest,
)
from ..workflow.interpreters import V4BuiltinParams
from ..workflow.effective_features import (
    EffectiveFeatureConfiguration,
    feature_parameter_model,
)
from ..workflow.node_contracts import (
    V4_NODE_CONTRACT_REGISTRY,
    V5_NODE_CONTRACT_REGISTRY,
)
from ..workflow.lora_factory import ResolvedLoraAdapter
from ..workflow.templates import V4_TEMPLATE_BUNDLE
from ..workflow.v4_compiler import (
    V4_VALIDATED_TEMPLATES,
    build_v4_route_context,
    require_effective_route_activation_match,
    resolve_effective_raylight_attention_mode,
    resolve_v4_active_feature,
)
from ..workflow.v4_resolver import (
    CreativeCompileInputError,
    CreativeCompileInputResolver,
    HistoricalTakeLike,
    V4ResolvedSegmentRoute,
)
from .evaluator import (
    CapabilityEvaluation,
    CapabilityEvaluator,
    CapabilityReason,
    resolution_adapter_fingerprint,
)


class EffectiveFeaturePreflight(ContractModel):
    id: str = Field(min_length=1, max_length=128)
    version: PositiveVersion
    state: Literal["active", "noop"]
    adapter_fingerprint: Sha256Digest
    capability: CapabilityEvaluation


class EffectiveSegmentPreflight(ContractModel):
    unit_id: Annotated[str, Field(min_length=1, max_length=256)]
    backend: Backend
    family: ModelFamily
    template_id: Literal["h3_standard_segment", "h3_raylight_segment"]
    features: tuple[EffectiveFeaturePreflight, ...]

    @model_validator(mode="after")
    def _validate_feature_ids(self) -> "EffectiveSegmentPreflight":
        identities = tuple((item.id, item.version) for item in self.features)
        if len(identities) != len(set(identities)):
            raise ValueError("effective feature identities must be unique")
        return self


class FeaturePreflightReport(ContractModel):
    template_bundle_version: PositiveVersion
    host_capability_revision: Sha256Digest
    operational_readiness: OperationalReadiness
    valid: bool
    errors: tuple[CapabilityReason, ...]
    effective_by_segment: FrozenMap[str, EffectiveSegmentPreflight] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _validate_validity(self) -> "FeaturePreflightReport":
        if self.valid == bool(self.errors):
            raise ValueError("preflight validity must agree with errors")
        return self


def _route_entry_enabled(entry_id: str, route: V4ResolvedSegmentRoute) -> bool:
    if entry_id == "lora":
        return route.lora_resolution is not None
    if entry_id == "continuity":
        return route.predecessor_segment_id is not None
    return True


def _invalid_report(
    *,
    snapshot: HostCapabilitySnapshot,
    readiness: OperationalReadiness,
    error: CapabilityReason,
) -> FeaturePreflightReport:
    return FeaturePreflightReport(
        template_bundle_version=V4_TEMPLATE_BUNDLE.version,
        host_capability_revision=snapshot.host_capability_revision(),
        operational_readiness=readiness,
        valid=False,
        errors=(error,),
        effective_by_segment={},
    )


def preflight_v4_timeline(
    *,
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    snapshot: HostCapabilitySnapshot,
    readiness: OperationalReadiness,
    segment_ids: list[str] | tuple[str, ...] | None = None,
    historical_takes: Mapping[str, HistoricalTakeLike] | None = None,
    resolved_lora_adapters: Mapping[
        ModelFamily, ResolvedLoraAdapter
    ]
    | None = None,
    evaluator: CapabilityEvaluator | None = None,
) -> FeaturePreflightReport:
    """Resolve and evaluate v4 features without graph emission or I/O.

    The caller supplies already-read settings, host snapshot, readiness, and
    optional historical/LoRA observations.  The function never reads a
    database, calls ComfyUI, mutates Ray state, or invokes ``emit()``.  Its
    result is advisory only; submission must call resolution/evaluation again.
    """

    if not isinstance(draft, UnifiedTimelineDraft) or draft.version != 4:
        raise TypeError("v4 preflight requires a validated timeline v4 draft")
    if not isinstance(settings, RuntimeSettings):
        raise TypeError("v4 preflight requires validated RuntimeSettings")
    if not isinstance(snapshot, HostCapabilitySnapshot):
        raise TypeError("v4 preflight requires HostCapabilitySnapshot")
    if not isinstance(readiness, OperationalReadiness):
        raise TypeError("v4 preflight requires OperationalReadiness")
    capability_evaluator = evaluator or CapabilityEvaluator(
        V4_NODE_CONTRACT_REGISTRY
    )

    try:
        compile_input = CreativeCompileInputResolver.resolve_v4(
            draft,
            settings,
            segment_ids,
            historical_takes,
            resolved_lora_adapters,
        )
    except CreativeCompileInputError as exc:
        return _invalid_report(
            snapshot=snapshot,
            readiness=readiness,
            error=CapabilityReason(
                code=exc.code,
                feature_id=exc.feature_id,
                segment_id=exc.segment_id,
                unit_id=None,
                backend=exc.backend,
                rule=exc.rule,
                message=exc.public_message,
                remediation=exc.remediation,
                safe_details=exc.safe_details,
            ),
        )

    resolved_draft = compile_input.materialize_draft()
    resolved_settings = compile_input.materialize_settings()
    effective_by_segment: dict[str, EffectiveSegmentPreflight] = {}
    errors: list[CapabilityReason] = []
    params = V4BuiltinParams()

    for route in compile_input.routes:
        context = build_v4_route_context(
            route,
            draft=resolved_draft,
            settings=resolved_settings,
            job_id="feature-preflight",
            timeline_assembly_required=(
                compile_input.requires_timeline_assembly()
            ),
        )
        template = V4_VALIDATED_TEMPLATES[route.template_id]
        effective_features: list[EffectiveFeaturePreflight] = []
        for entry in template.template.entries:
            try:
                if not _route_entry_enabled(entry.id, route):
                    resolution = FeatureResolution(
                        state="noop",
                        implementations=(),
                        resolution_details={"reason": "disabled_by_v4_context"},
                    )
                    # v4 built-ins require their active resolution when their
                    # method is called. A disabled switch does not call the
                    # interpreter at all and therefore declares no capability.
                    evaluation = CapabilityEvaluation(available=True)
                else:
                    feature_binding = resolve_v4_active_feature(
                        entry=entry,
                        template=template,
                        params=params,
                        context=context,
                    )
                    resolution = feature_binding.resolution
                    required = feature_binding.required_capabilities
                    evaluation = capability_evaluator.evaluate(
                        feature_id=entry.id,
                        ctx=context,
                        resolution=resolution,
                        required_capabilities=required,
                        snapshot=snapshot,
                        readiness=readiness,
                        segment_id=route.segment_id,
                        unit_id=route.unit_id,
                    )
            except (KeyError, TypeError, ValueError):
                failure = CapabilityReason(
                    code="feature_resolution_failed",
                    feature_id=entry.id,
                    segment_id=route.segment_id,
                    unit_id=route.unit_id,
                    backend=route.backend,
                    rule="feature_interpreter_resolution",
                    message="A feature could not be resolved for this segment.",
                    remediation="Correct the feature context or install the supported adapter.",
                    safe_details={},
                )
                errors.append(failure)
                effective_features.append(
                    EffectiveFeaturePreflight(
                        id=entry.id,
                        version=entry.version,
                        state="noop",
                        adapter_fingerprint=resolution_adapter_fingerprint(
                            feature_id=entry.id,
                            feature_version=entry.version,
                            ctx=context,
                            resolution=FeatureResolution(
                                state="noop",
                                implementations=(),
                                resolution_details={"reason": "resolution_failed"},
                            ),
                        ),
                        capability=CapabilityEvaluation(
                            available=False,
                            reasons=(failure,),
                        ),
                    )
                )
                continue

            errors.extend(evaluation.reasons)
            effective_features.append(
                EffectiveFeaturePreflight(
                    id=entry.id,
                    version=entry.version,
                    state=resolution.state,
                    adapter_fingerprint=resolution_adapter_fingerprint(
                        feature_id=entry.id,
                        feature_version=entry.version,
                        ctx=context,
                        resolution=resolution,
                    ),
                    capability=evaluation,
                )
            )

        effective_by_segment[route.segment_id] = EffectiveSegmentPreflight(
            unit_id=route.unit_id,
            backend=route.backend,
            family=route.family,
            template_id=route.template_id,
            features=tuple(effective_features),
        )

    return FeaturePreflightReport(
        template_bundle_version=V4_TEMPLATE_BUNDLE.version,
        host_capability_revision=snapshot.host_capability_revision(),
        operational_readiness=readiness,
        valid=not errors,
        errors=tuple(errors),
        effective_by_segment=effective_by_segment,
    )


def _plain_feature_params(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_feature_params(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_feature_params(item) for item in value]
    return value


def preflight_projected_v5_timeline(
    *,
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    effective_features: EffectiveFeatureConfiguration,
    snapshot: HostCapabilitySnapshot,
    readiness: OperationalReadiness,
    segment_ids: list[str] | tuple[str, ...] | None = None,
    historical_takes: Mapping[str, HistoricalTakeLike] | None = None,
    resolved_lora_adapters: Mapping[ModelFamily, ResolvedLoraAdapter] | None = None,
    evaluator: CapabilityEvaluator | None = None,
) -> FeaturePreflightReport:
    """Evaluate the exact bundle-5 selections and stop before graph emission."""

    from ..workflow.templates import V5_TEMPLATE_BUNDLE
    from ..workflow.v5_registry import V5_VALIDATED_TEMPLATES

    if effective_features.template_bundle_version != V5_TEMPLATE_BUNDLE.version:
        raise TypeError("v5 preflight requires the current effective feature bundle")
    capability_evaluator = evaluator or CapabilityEvaluator(
        V5_NODE_CONTRACT_REGISTRY
    )
    try:
        compile_input = CreativeCompileInputResolver.resolve_v4(
            draft,
            settings,
            segment_ids,
            historical_takes,
            resolved_lora_adapters,
        )
    except CreativeCompileInputError as exc:
        return FeaturePreflightReport(
            template_bundle_version=V5_TEMPLATE_BUNDLE.version,
            host_capability_revision=snapshot.host_capability_revision(),
            operational_readiness=readiness,
            valid=False,
            errors=(
                CapabilityReason(
                    code=exc.code,
                    feature_id=exc.feature_id,
                    segment_id=exc.segment_id,
                    unit_id=None,
                    backend=exc.backend,
                    rule=exc.rule,
                    message=exc.public_message,
                    remediation=exc.remediation,
                    safe_details=exc.safe_details,
                ),
            ),
            effective_by_segment={},
        )

    route_ids = {route.segment_id for route in compile_input.routes}
    if set(effective_features.effective_by_segment) != route_ids:
        raise ValueError("effective feature resolution must exactly cover v5 routes")
    resolved_draft = compile_input.materialize_draft()
    resolved_settings = compile_input.materialize_settings()
    effective_by_segment: dict[str, EffectiveSegmentPreflight] = {}
    errors: list[CapabilityReason] = []

    for route in compile_input.routes:
        effective_segment = effective_features.effective_by_segment[route.segment_id]
        raylight_attention_mode = resolve_effective_raylight_attention_mode(
            route=route,
            effective_segment=effective_segment,
        )
        context = build_v4_route_context(
            route,
            draft=resolved_draft,
            settings=resolved_settings,
            job_id="feature-preflight",
            timeline_assembly_required=compile_input.requires_timeline_assembly(),
            template_bundle_version=V5_TEMPLATE_BUNDLE.version,
            raylight_attention_mode=raylight_attention_mode,
        )
        template = V5_VALIDATED_TEMPLATES[route.template_id]
        configured = {item.id: item for item in effective_segment.features}
        if tuple(configured) != tuple(entry.id for entry in template.template.entries):
            raise ValueError("effective feature order differs from current template")
        projected: list[EffectiveFeaturePreflight] = []
        for entry in template.template.entries:
            selection = configured[entry.id]
            active = selection.active
            require_effective_route_activation_match(
                entry_id=entry.id,
                route=route,
                effective_active=active,
            )
            if not active:
                resolution = FeatureResolution(
                    state="noop",
                    implementations=(),
                    resolution_details={"reason": "disabled_by_v5_effective_config"},
                )
                evaluation = CapabilityEvaluation(available=True)
            else:
                try:
                    params = feature_parameter_model(
                        entry.id,
                        entry.version,
                    ).model_validate(_plain_feature_params(selection.params))
                    binding = resolve_v4_active_feature(
                        entry=entry,
                        template=template,
                        params=params,
                        context=context,
                    )
                    resolution = binding.resolution
                    evaluation = capability_evaluator.evaluate(
                        feature_id=entry.id,
                        ctx=context,
                        resolution=resolution,
                        required_capabilities=binding.required_capabilities,
                        snapshot=snapshot,
                        readiness=readiness,
                        segment_id=route.segment_id,
                        unit_id=route.unit_id,
                    )
                except (KeyError, TypeError, ValueError):
                    failure = CapabilityReason(
                        code="feature_resolution_failed",
                        feature_id=entry.id,
                        segment_id=route.segment_id,
                        unit_id=route.unit_id,
                        backend=route.backend,
                        rule="feature_interpreter_resolution",
                        message="A feature could not be resolved for this segment.",
                        remediation="Correct the feature context or install the exact feature implementation.",
                        safe_details={},
                    )
                    resolution = FeatureResolution(
                        state="noop",
                        implementations=(),
                        resolution_details={"reason": "resolution_failed"},
                    )
                    evaluation = CapabilityEvaluation(
                        available=False,
                        reasons=(failure,),
                    )
            errors.extend(evaluation.reasons)
            projected.append(
                EffectiveFeaturePreflight(
                    id=entry.id,
                    version=entry.version,
                    state=resolution.state,
                    adapter_fingerprint=resolution_adapter_fingerprint(
                        feature_id=entry.id,
                        feature_version=entry.version,
                        ctx=context,
                        resolution=resolution,
                    ),
                    capability=evaluation,
                )
            )
        effective_by_segment[route.segment_id] = EffectiveSegmentPreflight(
            unit_id=route.unit_id,
            backend=route.backend,
            family=route.family,
            template_id=route.template_id,
            features=tuple(projected),
        )

    return FeaturePreflightReport(
        template_bundle_version=V5_TEMPLATE_BUNDLE.version,
        host_capability_revision=snapshot.host_capability_revision(),
        operational_readiness=readiness,
        valid=not errors,
        errors=tuple(errors),
        effective_by_segment=effective_by_segment,
    )


__all__ = [
    "EffectiveFeaturePreflight",
    "EffectiveSegmentPreflight",
    "FeaturePreflightReport",
    "preflight_projected_v5_timeline",
    "preflight_v4_timeline",
]
