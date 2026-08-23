from __future__ import annotations

"""v5 creative authority projection into the graph-stable compiler core.

The v5 migration moves model and LoRA choices out of runtime settings. Stage 7
resolves the active LoRA through exact user/factory mappings here, before the
compatibility-shaped graph compiler sees the request. No live-settings
creative fallback, filename inference, or metadata probe is permitted.

The projection is pure: it reads one already validated v5 draft and one
captured RuntimeSettingsV3 document, never reads live project/settings state,
and never mutates either input.
"""

from dataclasses import dataclass
from typing import Any, Literal

from ..compiler import unified_continuity_predecessors
from ..schemas import (
    DiffusionModelBinding,
    LoraFeatureParams,
    RuntimeDiffusionPlacement,
    RuntimeSettingsV1,
    RuntimeSettingsV3,
    StandardLoraLoaderOverride,
    UnifiedTimelineDraftV4,
    UnifiedTimelineDraftV5,
    UnifiedTimelineSegment,
)
from .contracts import HostCapabilitySnapshot, ModelFamily, OperationalReadiness
from .execution import CompiledExecutionPlan
from .effective_features import (
    EffectiveFeatureConfiguration,
    V5FeatureConfigurationError,
    migrate_feature_configuration_to_v5,
    resolve_v5_effective_features,
)
from .lora_factory import (
    LoraAdapterResolutionError,
    LoraLoaderBindingKey,
    ResolvedLoraAdapter,
    resolve_raylight_lora_adapter,
    resolve_standard_lora_adapter,
)
from .templates import V5_TEMPLATE_BUNDLE
from .v4_compiler import compile_projected_v5_timeline
from .v4_execution_adapter import adapt_v4_compile_result


class V5CreativeAuthorityError(ValueError):
    """The captured v5 authorities cannot feed the frozen compiler safely."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        feature_id: str | None = None,
        segment_id: str | None = None,
        safe_details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.feature_id = feature_id
        self.segment_id = segment_id
        self.safe_details = dict(safe_details or {})


@dataclass(frozen=True, slots=True)
class V5LegacyCompileProjection:
    """One immutable compatibility view used for a single compile request."""

    draft: UnifiedTimelineDraftV4
    settings: RuntimeSettingsV1
    resolved_lora_adapters: tuple["V5ResolvedFamilyLora", ...]
    effective_features: EffectiveFeatureConfiguration

    def lora_adapter_map(self) -> dict[ModelFamily, ResolvedLoraAdapter]:
        return {item.family: item.resolution for item in self.resolved_lora_adapters}


@dataclass(frozen=True, slots=True)
class V5ContextualHostProjection:
    """Legacy-shaped inventory/placement view without adapter resolution.

    This projection is intentionally insufficient for preflight or compile.
    It exists only so all public entrypoints can collect safe host model and
    placement diagnostics even when the exact Standard LoRA mapping is absent.
    """

    draft: UnifiedTimelineDraftV4
    settings: RuntimeSettingsV1


@dataclass(frozen=True, slots=True)
class V5ResolvedFamilyLora:
    """One active family and its already-resolved immutable adapter."""

    family: ModelFamily
    resolution: ResolvedLoraAdapter


@dataclass(frozen=True, slots=True)
class V5RayLightRuntimeProjection:
    """Immutable RayLight topology used by a selected RayLight route."""

    gpu_select: tuple[int, ...]
    ulysses_degree: int
    ring_degree: int
    cfg_degree: int
    dp_degree: int
    fsdp: bool
    cpu_offload: bool


@dataclass(frozen=True, slots=True)
class V5RuntimeFamilyProjection:
    """The runtime placement fields reachable by one selected family route."""

    family: ModelFamily
    backend: Literal["standard", "raylight"]
    device: str
    # A one-GPU RayLight profile selects the Standard route, so its GPU index
    # and topology are not execution inputs.  Preserve the exact profile only
    # when the selected route actually emits RayLight nodes.
    raylight_profile: V5RayLightRuntimeProjection | None


@dataclass(frozen=True, slots=True)
class V5RuntimeCurrentnessProjection:
    """Bounded runtime authority that can affect one v5 job's execution.

    ``RuntimeSettingsV3`` also contains placement and host mappings for other
    model families.  Copying that entire document into currentness
    would make an FL2VA take stale after an unrelated Ref2VA edit.  This value
    contains only fields reachable from the job's captured segment selection.
    """

    memory_policy: str
    # These process-wide Ray controls affect execution only when at least one
    # selected family actually resolves to the RayLight route.  Keeping them
    # out of Standard-only projections prevents unrelated runtime toggles from
    # invalidating otherwise identical Standard takes.
    raylight_residency_policy: str | None
    multi_gpu_enabled: bool | None
    families: tuple[V5RuntimeFamilyProjection, ...]
    clip_device: str
    video_vae_device: str
    audio_vae_device: str | None


def _reject_unsupported_features(draft: UnifiedTimelineDraftV5) -> None:
    try:
        migrate_feature_configuration_to_v5(draft.features)
    except V5FeatureConfigurationError as exc:
        raise V5CreativeAuthorityError(
            str(exc),
            code=exc.code,
            feature_id=exc.feature_id,
            segment_id=exc.segment_id,
            safe_details=exc.safe_details,
        ) from exc


def _effective_features(
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    selected: tuple[UnifiedTimelineSegment, ...],
) -> EffectiveFeatureConfiguration:
    backend_by_family: dict[ModelFamily, Literal["standard", "raylight"]] = {}
    for family in ("fl2va", "ref2va"):
        placement = getattr(settings.placement, family)
        backend_by_family[family] = (
            "raylight" if len(placement.raylight.gpu_select) >= 2 else "standard"
        )
    continuity_consumers = unified_continuity_predecessors(draft)
    contextual = {
        segment.id: (
            frozenset({"continuity"})
            if segment.id in continuity_consumers
            else frozenset()
        )
        for segment in selected
    }
    try:
        return resolve_v5_effective_features(
            draft,
            selected_segment_ids=tuple(segment.id for segment in selected),
            backend_by_family=backend_by_family,
            contextual_switches=contextual,
        )
    except V5FeatureConfigurationError as exc:
        raise V5CreativeAuthorityError(
            str(exc),
            code=exc.code,
            feature_id=exc.feature_id,
            segment_id=exc.segment_id,
            safe_details=exc.safe_details,
        ) from exc


def _required_filename(
    draft: UnifiedTimelineDraftV5,
    role: str,
) -> str:
    selection = getattr(draft.model_stack, role)
    if selection.filename is None:
        raise V5CreativeAuthorityError(
            "The project is incomplete and cannot be compiled until all model bindings are selected.",
            code="model_binding_required",
            safe_details={"bindings": [role]},
        )
    return selection.filename


def _lora_params(draft: UnifiedTimelineDraftV5) -> LoraFeatureParams | None:
    selection = draft.features.project.get("lora")
    if selection is None or not selection.enabled:
        return None
    return LoraFeatureParams.model_validate(selection.params)


def _selected_v5_segments(
    draft: UnifiedTimelineDraftV5,
    segment_ids: list[str] | None,
) -> tuple[UnifiedTimelineSegment, ...]:
    """Resolve one bounded job selection without widening stale snapshots."""

    enabled = tuple(segment for segment in draft.segments if segment.enabled)
    if segment_ids is None:
        selected = enabled
    else:
        if (
            not isinstance(segment_ids, list)
            or not 1 <= len(segment_ids) <= 128
            or any(
                not isinstance(segment_id, str)
                or not 1 <= len(segment_id) <= 128
                for segment_id in segment_ids
            )
            or len(set(segment_ids)) != len(segment_ids)
        ):
            raise V5CreativeAuthorityError(
                "The captured segment selection is invalid.",
                code="segment_selection_invalid",
            )
        selected_ids = set(segment_ids)
        enabled_ids = {segment.id for segment in enabled}
        if not selected_ids <= enabled_ids:
            raise V5CreativeAuthorityError(
                "The captured segment selection is empty, disabled, or stale.",
                code="segment_selection_invalid",
            )
        selected = tuple(
            segment for segment in enabled if segment.id in selected_ids
        )
    if not selected:
        raise V5CreativeAuthorityError(
            "The captured segment selection contains no enabled segments.",
            code="segment_selection_invalid",
        )
    return selected


def _raylight_runtime_projection(
    placement: RuntimeDiffusionPlacement,
) -> V5RayLightRuntimeProjection:
    profile = placement.raylight
    return V5RayLightRuntimeProjection(
        gpu_select=tuple(profile.gpu_select),
        ulysses_degree=profile.ulysses_degree,
        ring_degree=profile.ring_degree,
        cfg_degree=profile.cfg_degree,
        dp_degree=profile.dp_degree,
        fsdp=profile.fsdp,
        cpu_offload=profile.cpu_offload,
    )


def project_v5_runtime_currentness(
    draft: UnifiedTimelineDraftV5,
    segment_ids: list[str] | None,
    settings: RuntimeSettingsV3,
) -> V5RuntimeCurrentnessProjection:
    """Project only runtime settings reachable by one captured v5 job.

    Creative fields remain represented by ``draft`` itself.  The projection
    intentionally omits ``client_id``, unrelated family placement, an unused
    audio-VAE device, and the complete LoRA mapping table. Resolved adapter
    identity is captured separately by :func:`resolve_v5_lora_adapters`.
    """

    if draft.version != 5 or settings.schema_version != 3:
        raise V5CreativeAuthorityError(
            "runtime currentness requires timeline schema 5 and runtime settings schema 3.",
            code="creative_authority_schema_mismatch",
        )
    _reject_unsupported_features(draft)
    selected = _selected_v5_segments(draft, segment_ids)
    families = tuple(
        family
        for family in ("fl2va", "ref2va")
        if any(segment.mode == family for segment in selected)
    )
    family_projections: list[V5RuntimeFamilyProjection] = []
    for family in families:
        placement = getattr(settings.placement, family)
        backend: Literal["standard", "raylight"] = (
            "raylight" if len(placement.raylight.gpu_select) >= 2 else "standard"
        )
        family_projections.append(
            V5RuntimeFamilyProjection(
                family=family,
                backend=backend,
                device=placement.device,
                raylight_profile=(
                    _raylight_runtime_projection(placement)
                    if backend == "raylight"
                    else None
                ),
            )
        )

    needs_audio_vae = any(
        segment.mode == "ref2va" or segment.audio_mode == "generate"
        for segment in selected
    )
    uses_raylight = any(
        projection.backend == "raylight" for projection in family_projections
    )
    return V5RuntimeCurrentnessProjection(
        memory_policy=settings.memory_policy,
        raylight_residency_policy=(
            settings.raylight_residency_policy if uses_raylight else None
        ),
        multi_gpu_enabled=(settings.multi_gpu_enabled if uses_raylight else None),
        families=tuple(family_projections),
        clip_device=settings.placement.clip_device,
        video_vae_device=settings.placement.video_vae_device,
        audio_vae_device=(
            settings.placement.audio_vae_device if needs_audio_vae else None
        ),
    )


def resolve_v5_lora_adapters(
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    segment_ids: list[str] | None,
) -> tuple[V5ResolvedFamilyLora, ...]:
    """Resolve only active, selected family slots through Stage-7 rules."""

    if draft.version != 5 or settings.schema_version != 3:
        raise V5CreativeAuthorityError(
            "LoRA resolution requires timeline schema 5 and runtime settings schema 3.",
            code="creative_authority_schema_mismatch",
        )
    _reject_unsupported_features(draft)
    selected = _selected_v5_segments(draft, segment_ids)
    families = tuple(
        family
        for family in ("fl2va", "ref2va")
        if any(segment.mode == family for segment in selected)
    )
    lora = _lora_params(draft)
    if lora is None:
        return ()

    resolved: list[V5ResolvedFamilyLora] = []
    for family in families:
        family_lora = lora.by_family[family]
        if not family_lora.enabled:
            continue
        if family_lora.filename is None:
            raise V5CreativeAuthorityError(
                "An enabled LoRA family slot requires a selected file.",
                code="lora_binding_required",
                feature_id="lora",
                safe_details={"family": family},
            )
        if family_lora.strength == 0.0:
            raise V5CreativeAuthorityError(
                "An enabled LoRA family slot requires a non-zero strength.",
                code="lora_strength_invalid",
                feature_id="lora",
                safe_details={"family": family},
            )
        placement = getattr(settings.placement, family)
        backend: Literal["standard", "raylight"] = (
            "raylight" if len(placement.raylight.gpu_select) >= 2 else "standard"
        )
        try:
            if backend == "raylight":
                resolution = resolve_raylight_lora_adapter(family)
            else:
                resolution = resolve_standard_lora_adapter(
                    LoraLoaderBindingKey(
                        family=family,
                        model_filename=_required_filename(draft, family),
                        lora_filename=family_lora.filename,
                    ),
                    settings.lora_loader_overrides,
                )
        except LoraAdapterResolutionError as exc:
            details: dict[str, Any] = {"family": family}
            if exc.adapter_id is not None:
                details["adapter_id"] = exc.adapter_id
            raise V5CreativeAuthorityError(
                str(exc),
                code=exc.code,
                feature_id="lora",
                safe_details=details,
            ) from exc
        resolved.append(
            V5ResolvedFamilyLora(family=family, resolution=resolution)
        )
    return tuple(resolved)


def _diffusion_binding(
    *,
    family: ModelFamily,
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    lora: LoraFeatureParams | None,
    resolved_lora: ResolvedLoraAdapter | None,
) -> DiffusionModelBinding:
    model_filename = _required_filename(draft, family)
    family_lora = lora.by_family[family] if lora is not None else None
    lora_filename = (
        family_lora.filename
        if family_lora is not None and family_lora.enabled
        else None
    )
    if family_lora is not None and family_lora.enabled and lora_filename is None:
        raise V5CreativeAuthorityError(
            "An enabled LoRA family slot requires a selected file.",
            code="lora_binding_required",
            feature_id="lora",
            safe_details={"family": family},
        )
    if lora_filename is not None and resolved_lora is None:
        raise V5CreativeAuthorityError(
            "An active LoRA slot has no immutable adapter resolution.",
            code="lora_loader_mapping_required",
            feature_id="lora",
            safe_details={"family": family},
        )
    standard_override = None
    if resolved_lora is not None and resolved_lora.adapter.backend == "standard":
        if resolved_lora.binding != LoraLoaderBindingKey(
            family=family,
            model_filename=model_filename,
            lora_filename=lora_filename or "",
        ):
            raise V5CreativeAuthorityError(
                "The resolved LoRA adapter binding drifted before compilation.",
                code="lora_adapter_incompatible",
                feature_id="lora",
                safe_details={"family": family},
            )
        standard_override = StandardLoraLoaderOverride(
            loader=resolved_lora.adapter.adapter_id,
            lora_name=lora_filename,
            model_filename=model_filename,
        )
    placement = getattr(settings.placement, family)
    return DiffusionModelBinding(
        filename=model_filename,
        device=placement.device,
        lora_name=lora_filename,
        lora_strength=(family_lora.strength if family_lora is not None else 1.0),
        lora_loader="auto",
        standard_lora_loader_override=standard_override,
        lora_low_vram=bool(
            resolved_lora.options.get("low_vram", False)
            if resolved_lora is not None
            else False
        ),
        backend="auto",
        raylight=placement.raylight,
    )


def _contextual_diffusion_binding(
    *,
    family: ModelFamily,
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    lora: LoraFeatureParams | None,
) -> DiffusionModelBinding:
    """Build only the fields needed by host inventory and placement checks."""

    family_lora = lora.by_family[family] if lora is not None else None
    lora_filename = (
        family_lora.filename
        if family_lora is not None and family_lora.enabled
        else None
    )
    if family_lora is not None and family_lora.enabled and lora_filename is None:
        raise V5CreativeAuthorityError(
            "An enabled LoRA family slot requires a selected file.",
            code="lora_binding_required",
            feature_id="lora",
            safe_details={"family": family},
        )
    placement = getattr(settings.placement, family)
    return DiffusionModelBinding(
        filename=_required_filename(draft, family),
        device=placement.device,
        lora_name=lora_filename,
        lora_strength=(family_lora.strength if family_lora is not None else 1.0),
        lora_loader="auto",
        standard_lora_loader_override=None,
        lora_low_vram=False,
        backend="auto",
        raylight=placement.raylight,
    )


def _legacy_draft(draft: UnifiedTimelineDraftV5) -> UnifiedTimelineDraftV4:
    raw_draft = draft.model_dump(
        mode="json",
        exclude={"model_stack", "features"},
    )
    raw_draft["version"] = 4
    return UnifiedTimelineDraftV4.model_validate(raw_draft)


def _legacy_runtime_settings(
    draft: UnifiedTimelineDraftV5,
    captured_settings: RuntimeSettingsV3,
    *,
    fl2va: DiffusionModelBinding,
    ref2va: DiffusionModelBinding,
) -> RuntimeSettingsV1:
    return RuntimeSettingsV1.model_validate(
        {
            "client_id": captured_settings.client_id,
            "memory_policy": captured_settings.memory_policy,
            "raylight_residency_policy": (
                captured_settings.raylight_residency_policy
            ),
            "multi_gpu_enabled": captured_settings.multi_gpu_enabled,
            "models": {
                "fl2va": fl2va.model_dump(mode="json"),
                "ref2va": ref2va.model_dump(mode="json"),
                "clip": {
                    "filename": _required_filename(draft, "clip"),
                    "device": captured_settings.placement.clip_device,
                },
                "video_vae": {
                    "filename": _required_filename(draft, "video_vae"),
                    "device": captured_settings.placement.video_vae_device,
                },
                "audio_vae": {
                    "filename": _required_filename(draft, "audio_vae"),
                    "device": captured_settings.placement.audio_vae_device,
                },
            },
        }
    )


def project_v5_contextual_host_authority(
    draft: UnifiedTimelineDraftV5,
    captured_settings: RuntimeSettingsV3,
    segment_ids: list[str] | None = None,
) -> V5ContextualHostProjection:
    """Project host-check inputs without requiring an exact LoRA mapping."""

    if draft.version != 5 or captured_settings.schema_version != 3:
        raise V5CreativeAuthorityError(
            "v5 host context requires timeline schema 5 and runtime settings schema 3.",
            code="creative_authority_schema_mismatch",
        )
    _reject_unsupported_features(draft)
    lora = _lora_params(draft)
    selected = _selected_v5_segments(draft, segment_ids)
    _effective_features(draft, captured_settings, selected)
    selected_families = {segment.mode for segment in selected}
    fl2va = _contextual_diffusion_binding(
        family="fl2va",
        draft=draft,
        settings=captured_settings,
        lora=(lora if "fl2va" in selected_families else None),
    )
    ref2va = _contextual_diffusion_binding(
        family="ref2va",
        draft=draft,
        settings=captured_settings,
        lora=(lora if "ref2va" in selected_families else None),
    )
    return V5ContextualHostProjection(
        draft=_legacy_draft(draft),
        settings=_legacy_runtime_settings(
            draft,
            captured_settings,
            fl2va=fl2va,
            ref2va=ref2va,
        ),
    )


def project_v5_compile_authority(
    draft: UnifiedTimelineDraftV5,
    captured_settings: RuntimeSettingsV3,
    segment_ids: list[str] | None = None,
) -> V5LegacyCompileProjection:
    """Reconstruct the exact legacy compiler view from two captured inputs."""

    if draft.version != 5 or captured_settings.schema_version != 3:
        raise V5CreativeAuthorityError(
            "v5 compilation requires timeline schema 5 and runtime settings schema 3.",
            code="creative_authority_schema_mismatch",
        )
    _reject_unsupported_features(draft)
    lora = _lora_params(draft)
    selected = _selected_v5_segments(draft, segment_ids)
    effective_features = _effective_features(draft, captured_settings, selected)
    selected_families = {
        segment.mode for segment in selected
    }
    resolved_lora_adapters = resolve_v5_lora_adapters(
        draft,
        captured_settings,
        segment_ids,
    )
    resolved_by_family = {
        item.family: item.resolution for item in resolved_lora_adapters
    }

    legacy_draft = _legacy_draft(draft)

    fl2va = _diffusion_binding(
        family="fl2va",
        draft=draft,
        settings=captured_settings,
        lora=(lora if "fl2va" in selected_families else None),
        resolved_lora=resolved_by_family.get("fl2va"),
    )
    ref2va = _diffusion_binding(
        family="ref2va",
        draft=draft,
        settings=captured_settings,
        lora=(lora if "ref2va" in selected_families else None),
        resolved_lora=resolved_by_family.get("ref2va"),
    )
    legacy_settings = _legacy_runtime_settings(
        draft,
        captured_settings,
        fl2va=fl2va,
        ref2va=ref2va,
    )
    return V5LegacyCompileProjection(
        draft=legacy_draft,
        settings=legacy_settings,
        resolved_lora_adapters=resolved_lora_adapters,
        effective_features=effective_features,
    )


def compile_v5_execution_plan(
    draft: UnifiedTimelineDraftV5,
    captured_settings: RuntimeSettingsV3,
    job_id: str,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Any = None,
    endpoint_key: str = "embedded",
    host_capability_snapshot: HostCapabilitySnapshot | None = None,
    operational_readiness: OperationalReadiness | None = None,
    capability_evaluator: Any | None = None,
) -> CompiledExecutionPlan:
    """Compile a v5 snapshot through the current bundle-5 feature order."""

    projection = project_v5_compile_authority(
        draft,
        captured_settings,
        segment_ids,
    )
    native_result = compile_projected_v5_timeline(
        projection.draft,
        projection.settings,
        job_id,
        projection.effective_features,
        segment_ids,
        historical_takes=historical_takes,
        resolved_lora_adapters=projection.lora_adapter_map(),
        host_capability_snapshot=host_capability_snapshot,
        operational_readiness=operational_readiness,
        capability_evaluator=capability_evaluator,
    )
    return adapt_v4_compile_result(
        native_result,
        draft=projection.draft,
        captured_settings=projection.settings,
        endpoint_key=endpoint_key,
        host_capability_revision=(
            host_capability_snapshot.host_capability_revision()
            if host_capability_snapshot is not None
            else None
        ),
        template_bundle=V5_TEMPLATE_BUNDLE,
    )


__all__ = [
    "V5ContextualHostProjection",
    "V5CreativeAuthorityError",
    "V5LegacyCompileProjection",
    "V5RayLightRuntimeProjection",
    "V5ResolvedFamilyLora",
    "V5RuntimeCurrentnessProjection",
    "V5RuntimeFamilyProjection",
    "compile_v5_execution_plan",
    "project_v5_contextual_host_authority",
    "project_v5_compile_authority",
    "project_v5_runtime_currentness",
    "resolve_v5_lora_adapters",
]
