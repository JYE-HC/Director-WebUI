from __future__ import annotations

"""Pure Bundle-5 authority migration and Bundle-6 compile-route projection."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..compiler import (
    DraftNotRunnable,
    unified_continuity_predecessors,
    validate_unified_runnable,
)
from ..native_templates import NativeHistoricalTake, _align_h3_frame_count
from ..schemas import (
    FeatureConfiguration,
    FeatureSelection,
    RuntimeSettingsV3,
    UnifiedFL2VASegment,
    UnifiedTimelineDraftV5,
    UnifiedTimelineSegment,
)
from .contracts import Backend, ModelFamily
from .feature_config import V6FeatureConfigurationError


class V5V6ProjectionError(ValueError):
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
class V5V6AuthorityProjection:
    draft: UnifiedTimelineDraftV5
    notices: tuple[str, ...] = ()


_LEGACY_FEATURES = frozenset(
    {
        "lora",
        "attention_backend_override",
        "h3_low_vram_attention",
        "raylight_pool_intent",
    }
)
_V6_FEATURES = frozenset({"lora", "comfy_kitchen_attention"})


def _selection(
    draft: UnifiedTimelineDraftV5,
    segment_id: str,
    feature_id: str,
) -> tuple[FeatureSelection | None, bool]:
    segment = draft.features.by_segment.get(segment_id, {})
    if feature_id in segment:
        return segment[feature_id], True
    if feature_id in draft.features.project:
        return draft.features.project[feature_id], True
    return None, False


def _validate_known_features(
    draft: UnifiedTimelineDraftV5,
    allowed: frozenset[str],
) -> None:
    for segment_id, selections in ((None, draft.features.project), *draft.features.by_segment.items()):
        unknown = sorted(set(selections) - allowed)
        if unknown:
            raise V5V6ProjectionError(
                "The project contains a feature unknown to this bundle migration.",
                code="unknown_feature",
                feature_id=unknown[0],
                segment_id=segment_id,
            )


def _standard_ck_intent(
    selection: FeatureSelection | None,
    *,
    segment_id: str,
) -> bool:
    if selection is None or not selection.enabled:
        return False
    if set(selection.params) != {"mode"}:
        raise V5V6ProjectionError(
            "The legacy Standard attention selection is invalid.",
            code="feature_params_invalid",
            feature_id="attention_backend_override",
            segment_id=segment_id,
        )
    mode = selection.params["mode"]
    if mode == "ck_int8":
        return True
    if mode == "pytorch":
        raise V5V6ProjectionError(
            "Explicit PyTorch attention has no automatic Bundle 6 equivalent.",
            code="attention_migration_conflict",
            feature_id="attention_backend_override",
            segment_id=segment_id,
        )
    raise V5V6ProjectionError(
        "The legacy Standard attention mode is invalid.",
        code="feature_params_invalid",
        feature_id="attention_backend_override",
        segment_id=segment_id,
    )


def _ray_ck_intent(
    selection: FeatureSelection | None,
    explicit: bool,
    *,
    segment_id: str,
) -> tuple[bool | None, bool]:
    if not explicit:
        return None, False
    assert selection is not None
    if not selection.enabled:
        return False, True
    if set(selection.params) != {"attention"}:
        raise V5V6ProjectionError(
            "The legacy RayLight attention selection is invalid.",
            code="feature_params_invalid",
            feature_id="raylight_pool_intent",
            segment_id=segment_id,
        )
    attention = selection.params["attention"]
    if attention == "ck_int8":
        return True, False
    if attention == "torch_flash":
        return False, False
    raise V5V6ProjectionError(
        "The legacy RayLight attention mode is invalid.",
        code="feature_params_invalid",
        feature_id="raylight_pool_intent",
        segment_id=segment_id,
    )


def project_v5_authority_to_v6(
    draft: UnifiedTimelineDraftV5,
) -> V5V6AuthorityProjection:
    """Map only unambiguous authoring semantics; never read runtime capability."""

    source = draft.features.template_bundle_version
    if source == 6:
        _validate_known_features(draft, _V6_FEATURES)
        if draft.features.by_segment:
            raise V5V6ProjectionError(
                "Bundle 6 feature selections are project-only in this release.",
                code="feature_scope_unsupported",
            )
        ck = draft.features.project.get("comfy_kitchen_attention")
        if ck is None or set(ck.params) or not isinstance(ck.enabled, bool):
            raise V5V6ProjectionError(
                "Bundle 6 requires the canonical CK project selection.",
                code="feature_params_invalid",
                feature_id="comfy_kitchen_attention",
            )
        return V5V6AuthorityProjection(draft=draft.model_copy(deep=True))
    if source != 5:
        raise V5V6ProjectionError(
            "Only Bundle 5 authorities can be projected to Bundle 6.",
            code="template_bundle_version_unsupported",
        )

    _validate_known_features(draft, _LEGACY_FEATURES)
    if any("lora" in selections for selections in draft.features.by_segment.values()):
        raise V5V6ProjectionError(
            "Segment-scoped LoRA has no automatic Bundle 6 projection.",
            code="feature_scope_unsupported",
            feature_id="lora",
        )

    notices: list[str] = []
    normalized: list[bool] = []
    standard_carrier_changed = False
    for segment in draft.segments:
        low_vram, _ = _selection(draft, segment.id, "h3_low_vram_attention")
        if low_vram is not None and low_vram.enabled:
            raise V5V6ProjectionError(
                "Low-VRAM attention has no automatic Bundle 6 equivalent.",
                code="attention_migration_conflict",
                feature_id="h3_low_vram_attention",
                segment_id=segment.id,
            )
        standard, _ = _selection(
            draft,
            segment.id,
            "attention_backend_override",
        )
        standard_ck = _standard_ck_intent(standard, segment_id=segment.id)
        ray, ray_explicit = _selection(
            draft,
            segment.id,
            "raylight_pool_intent",
        )
        ray_ck, invalid_needed = _ray_ck_intent(
            ray,
            ray_explicit,
            segment_id=segment.id,
        )
        if invalid_needed and "Legacy disabled RayLight pool intent was normalized." not in notices:
            notices.append("Legacy disabled RayLight pool intent was normalized.")
        if standard_ck:
            if ray_ck is False:
                raise V5V6ProjectionError(
                    "Standard and RayLight attention intentions conflict.",
                    code="attention_migration_conflict",
                    segment_id=segment.id,
                )
            normalized.append(True)
            standard_carrier_changed = True
        else:
            if ray_ck is True:
                raise V5V6ProjectionError(
                    "A Ray-only CK request cannot be expanded to all backends automatically.",
                    code="attention_migration_conflict",
                    segment_id=segment.id,
                )
            normalized.append(False)

    if len(set(normalized)) != 1:
        raise V5V6ProjectionError(
            "Segment attention intentions cannot be folded into one project switch.",
            code="attention_migration_conflict",
        )
    if standard_carrier_changed:
        notices.append(
            "Standard CK now uses ComfyUI's official ModelAttentionBackend carrier."
        )

    project: dict[str, FeatureSelection] = {}
    if "lora" in draft.features.project:
        project["lora"] = draft.features.project["lora"].model_copy(deep=True)
    project["comfy_kitchen_attention"] = FeatureSelection(
        enabled=normalized[0],
        params={},
    )
    migrated = draft.model_copy(
        update={
            "features": FeatureConfiguration(
                template_bundle_version=6,
                project=project,
                by_segment={},
            )
        },
        deep=True,
    )
    return V5V6AuthorityProjection(draft=migrated, notices=tuple(notices))


@dataclass(frozen=True, slots=True)
class V6ResolvedRoute:
    timeline_index: int
    segment: UnifiedTimelineSegment
    backend: Backend
    family: ModelFamily
    unit_id: str
    predecessor_segment_id: str | None
    continuity_source: Literal["same_run", "historical_take"] | None
    historical_take: NativeHistoricalTake | None
    anchor_reset: bool
    clear_raylight_vram_after_sampling: bool


@dataclass(frozen=True, slots=True)
class V6SegmentCompileContext:
    draft: UnifiedTimelineDraftV5
    settings: RuntimeSettingsV3
    segment: UnifiedTimelineSegment
    backend: Backend
    family: ModelFamily
    job_id: str
    unit_id: str
    visible_frames: int
    sample_frames: int
    continuity_prefix_frames: int
    predecessor_segment_id: str | None
    continuity_source: Literal["same_run", "historical_take"] | None
    historical_take_id: str | None
    clear_raylight_vram_after_sampling: bool
    timeline_assembly_required: bool


@dataclass(frozen=True, slots=True)
class V6CompileProjection:
    draft: UnifiedTimelineDraftV5
    settings: RuntimeSettingsV3
    routes: tuple[V6ResolvedRoute, ...]
    families: tuple[ModelFamily, ...]
    selected_segment_ids: tuple[str, ...]
    timeline_assembly_required: bool

    def context(self, route: V6ResolvedRoute, job_id: str) -> V6SegmentCompileContext:
        visible = max(5, int(round(route.segment.duration_seconds * self.draft.render.fps)))
        visible += (5 - visible % 17) % 17
        prefix = (
            route.segment.continuity.overlap_frames
            if route.predecessor_segment_id is not None
            else 0
        )
        return V6SegmentCompileContext(
            draft=self.draft,
            settings=self.settings,
            segment=route.segment,
            backend=route.backend,
            family=route.family,
            job_id=job_id,
            unit_id=route.unit_id,
            visible_frames=visible,
            sample_frames=_align_h3_frame_count(visible + prefix),
            continuity_prefix_frames=prefix,
            predecessor_segment_id=route.predecessor_segment_id,
            continuity_source=route.continuity_source,
            historical_take_id=(route.historical_take.id if route.historical_take else None),
            clear_raylight_vram_after_sampling=route.clear_raylight_vram_after_sampling,
            timeline_assembly_required=self.timeline_assembly_required,
        )


def project_v6_compile_authority(
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, NativeHistoricalTake] | None = None,
) -> V6CompileProjection:
    if draft.features.template_bundle_version != 6:
        raise V5V6ProjectionError(
            "Bundle 6 compilation requires Bundle 6 feature authority.",
            code="template_bundle_version_unsupported",
        )
    # The idempotent V6 branch validates the current feature authority without
    # reading runtime state or rewriting user choices.
    draft = project_v5_authority_to_v6(draft).draft
    try:
        enabled = validate_unified_runnable(draft, segment_ids=segment_ids)  # type: ignore[arg-type]
    except DraftNotRunnable as exc:
        raise V5V6ProjectionError(
            str(exc),
            code="creative_configuration_invalid",
        ) from exc
    requested = set(segment_ids) if segment_ids is not None else None
    selected = [
        segment for segment in enabled
        if requested is None or segment.id in requested
    ]
    selected = sorted(
        selected,
        key=lambda item: next(index for index, segment in enumerate(draft.segments) if segment.id == item.id),
    )
    required_bindings = ["clip", "video_vae"]
    if any(
        segment.mode == "ref2va" or segment.audio_mode == "generate"
        for segment in selected
    ):
        required_bindings.append("audio_vae")
    required_bindings.extend(
        family
        for family in ("fl2va", "ref2va")
        if any(segment.mode == family for segment in selected)
    )
    missing_bindings = [
        role
        for role in required_bindings
        if getattr(draft.model_stack, role).filename is None
    ]
    if missing_bindings:
        missing_auxiliary = any(
            role in {"clip", "video_vae", "audio_vae"}
            for role in missing_bindings
        )
        missing_diffusion = any(
            role in {"fl2va", "ref2va"} for role in missing_bindings
        )
        raise V6FeatureConfigurationError(
            "One or more required model bindings are incomplete.",
            code="model_binding_required",
            feature_id=(
                None
                if missing_auxiliary and missing_diffusion
                else "auxiliary_models" if missing_auxiliary else "diffusion_model"
            ),
            safe_details={"bindings": missing_bindings},
        )
    selected_ids = {segment.id for segment in selected}
    predecessors = unified_continuity_predecessors(draft)  # type: ignore[arg-type]
    previous = {
        segment.id: (enabled[index - 1] if index else None)
        for index, segment in enumerate(enabled)
    }
    clear_after_sampling = settings.raylight_residency_policy != "keep_until_switch"
    routes: list[V6ResolvedRoute] = []
    for segment in selected:
        index = next(i for i, item in enumerate(draft.segments) if item.id == segment.id)
        family: ModelFamily = segment.mode
        backend: Backend = (
            "raylight" if settings.multi_gpu_enabled else "standard"
        )
        predecessor = predecessors.get(segment.id)
        source: Literal["same_run", "historical_take"] | None = None
        historical: NativeHistoricalTake | None = None
        if predecessor is not None:
            source = "same_run" if predecessor.id in selected_ids else "historical_take"
            if source == "historical_take":
                historical = None if historical_takes is None else historical_takes.get(segment.id)
                if historical is None or historical.segment_id != predecessor.id:
                    raise V5V6ProjectionError(
                        "Continuity requires the exact predecessor take.",
                        code="historical_take_required",
                        feature_id="continuity",
                        segment_id=segment.id,
                    )
        anchor_reset = bool(
            segment.continuity.enabled
            and predecessor is None
            and (
                previous[segment.id] is None
                or (isinstance(segment, UnifiedFL2VASegment) and segment.first_image is not None)
            )
        )
        routes.append(
            V6ResolvedRoute(
                timeline_index=index,
                segment=segment.model_copy(deep=True),
                backend=backend,
                family=family,
                unit_id=f"{backend}-{family}-{index:03d}",
                predecessor_segment_id=(predecessor.id if predecessor else None),
                continuity_source=source,
                historical_take=historical,
                anchor_reset=anchor_reset,
                clear_raylight_vram_after_sampling=clear_after_sampling,
            )
        )
    has_continuity = any(route.predecessor_segment_id for route in routes)
    ordered = (
        sorted(routes, key=lambda route: route.timeline_index)
        if has_continuity
        else sorted(
            routes,
            key=lambda route: (
                ("standard", "raylight").index(route.backend),
                ("fl2va", "ref2va").index(route.family),
                route.timeline_index,
            ),
        )
    )
    selected_segment_ids = tuple(segment.id for segment in selected)
    enabled_ids = {segment.id for segment in enabled}
    assembly = (
        draft.export_mode == "all"
        and len(selected_segment_ids) > 1
        and set(selected_segment_ids) == enabled_ids
    )
    families = tuple(
        family for family in ("fl2va", "ref2va") if any(route.family == family for route in ordered)
    )
    return V6CompileProjection(
        draft=draft.model_copy(deep=True),
        settings=settings.model_copy(deep=True),
        routes=tuple(ordered),
        families=families,
        selected_segment_ids=selected_segment_ids,
        timeline_assembly_required=assembly,
    )


__all__ = [
    "V5V6AuthorityProjection",
    "V5V6ProjectionError",
    "V6CompileProjection",
    "V6ResolvedRoute",
    "V6SegmentCompileContext",
    "project_v5_authority_to_v6",
    "project_v6_compile_authority",
]
