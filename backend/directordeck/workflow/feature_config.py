from __future__ import annotations

"""Typed, pure Bundle-6 feature configuration projection."""

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from ..schemas import (
    LoraFamilyFeatureSelection,
    LoraFeatureParams,
    RuntimeSettingsV3,
    StrictModel,
    UnifiedTimelineDraftV5,
    UnifiedTimelineSegment,
    timeline_segment_recipe,
)
from .contracts import Backend, ModelFamily
from .feature_config_models import (
    AudioOutputConfigV1,
    AuxiliaryModelsConfigV1,
    ComfyKitchenAttentionParamsV1,
    ConditioningConfigV1,
    ContinuityConfigV1,
    DiffusionModelConfigV1,
    EmptyFeatureParams,
    ExecutionStrategyConfigV1,
    LoraConfigV1,
    RayExecutionConfigV1,
    SamplingPipelineConfigV1,
    SaveTakeConfigV1,
    SigmaScheduleConfigV1,
    VideoDecodeConfigV1,
)
from .lora_factory import (
    LoraAdapterResolutionError,
    LoraLoaderBindingKey,
    resolve_raylight_lora_adapter,
    resolve_standard_lora_adapter,
)


class V6FeatureConfigurationError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        feature_id: str | None,
        segment_id: str | None = None,
        safe_details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.feature_id = feature_id
        self.segment_id = segment_id
        self.safe_details = dict(safe_details or {})


class V6ConfigContext(Protocol):
    draft: UnifiedTimelineDraftV5
    settings: RuntimeSettingsV3
    segment: UnifiedTimelineSegment
    backend: Backend
    family: ModelFamily
    job_id: str
    visible_frames: int
    sample_frames: int
    continuity_prefix_frames: int
    predecessor_segment_id: str | None
    continuity_source: Literal["same_run", "historical_take"] | None
    historical_take_id: str | None
    clear_raylight_vram_after_sampling: bool


class FeatureConfigResolver(Protocol):
    id: str

    def resolve(self, context: V6ConfigContext) -> "EffectiveFeatureUse": ...


@dataclass(frozen=True, slots=True)
class EffectiveFeatureUse:
    feature_id: str
    feature_version: int
    state: Literal["inactive", "applicable"]
    source: Literal["definition_default", "project", "segment", "context"]
    config: StrictModel | None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if self.state == "inactive":
            if self.config is not None or not self.reason_code:
                raise ValueError("inactive feature use requires only a reason")
        elif self.config is None or self.reason_code is not None:
            raise ValueError("applicable feature use requires typed config")


def _required_filename(draft: UnifiedTimelineDraftV5, role: str) -> str:
    filename = getattr(draft.model_stack, role).filename
    if filename is None:
        raise V6FeatureConfigurationError(
            "The selected model binding is incomplete.",
            code="model_binding_required",
            feature_id=("diffusion_model" if role in {"fl2va", "ref2va"} else "auxiliary_models"),
            safe_details={"bindings": [role]},
        )
    return filename


def _applicable(
    feature_id: str,
    config: StrictModel,
    source: Literal["definition_default", "project", "segment", "context"],
) -> EffectiveFeatureUse:
    return EffectiveFeatureUse(feature_id, 1, "applicable", source, config)


def _inactive(
    feature_id: str,
    reason: str,
    source: Literal["definition_default", "project", "segment", "context"] = "definition_default",
) -> EffectiveFeatureUse:
    return EffectiveFeatureUse(
        feature_id,
        1,
        "inactive",
        source,
        None,
        reason,
    )


def _auxiliary(context: V6ConfigContext) -> EffectiveFeatureUse:
    needs_audio_vae = (
        context.family == "ref2va" or context.segment.audio_mode == "generate"
    )
    return _applicable(
        "auxiliary_models",
        AuxiliaryModelsConfigV1(
            clip_filename=_required_filename(context.draft, "clip"),
            clip_device=context.settings.placement.clip_device,
            video_vae_filename=_required_filename(context.draft, "video_vae"),
            video_vae_device=context.settings.placement.video_vae_device,
            audio_vae_filename=(
                _required_filename(context.draft, "audio_vae")
                if needs_audio_vae
                else None
            ),
            audio_vae_device=(
                context.settings.placement.audio_vae_device
                if needs_audio_vae
                else None
            ),
        ),
        "project",
    )


def _diffusion(context: V6ConfigContext) -> EffectiveFeatureUse:
    placement = getattr(context.settings.placement, context.family)
    return _applicable(
        "diffusion_model",
        DiffusionModelConfigV1(
            family=context.family,
            backend=context.backend,
            filename=_required_filename(context.draft, context.family),
            device=placement.device,
        ),
        "project",
    )


def _execution(context: V6ConfigContext) -> EffectiveFeatureUse:
    placement = getattr(context.settings.placement, context.family)
    raylight = None
    if context.backend == "raylight":
        if not context.settings.multi_gpu_enabled:
            raise V6FeatureConfigurationError(
                "Multi-GPU inference is disabled for the selected RayLight route.",
                code="multi_gpu_disabled",
                feature_id="execution_strategy",
                segment_id=context.segment.id,
            )
        profile = placement.raylight
        raylight = RayExecutionConfigV1(
            gpu_select=tuple(profile.gpu_select),
            ulysses_degree=profile.ulysses_degree,
            ring_degree=profile.ring_degree,
            cfg_degree=profile.cfg_degree,
            dp_degree=profile.dp_degree,
            fsdp=profile.fsdp,
            cpu_offload=profile.cpu_offload,
            clear_vram_after_sampling=context.clear_raylight_vram_after_sampling,
        )
    return _applicable(
        "execution_strategy",
        ExecutionStrategyConfigV1(
            backend=context.backend,
            device=placement.device,
            raylight=raylight,
        ),
        "context",
    )


def _lora(context: V6ConfigContext) -> EffectiveFeatureUse:
    selection = context.draft.features.project.get("lora")
    if selection is None:
        return _inactive("lora", "disabled")
    if not selection.enabled:
        return _inactive("lora", "disabled", "project")
    params = LoraFeatureParams.model_validate(selection.params)
    family = params.by_family[context.family]
    if not family.enabled:
        return _inactive("lora", "family_disabled", "project")
    if family.filename is None or family.strength == 0:
        raise V6FeatureConfigurationError(
            "The active LoRA requires a filename and non-zero strength.",
            code=("lora_binding_required" if family.filename is None else "lora_strength_invalid"),
            feature_id="lora",
            segment_id=context.segment.id,
        )
    model_filename = _required_filename(context.draft, context.family)
    try:
        resolution = (
            resolve_raylight_lora_adapter(context.family)
            if context.backend == "raylight"
            else resolve_standard_lora_adapter(
                LoraLoaderBindingKey(
                    family=context.family,
                    model_filename=model_filename,
                    lora_filename=family.filename,
                ),
                context.settings.lora_loader_overrides,
            )
        )
    except LoraAdapterResolutionError as exc:
        raise V6FeatureConfigurationError(
            str(exc),
            code=exc.code,
            feature_id="lora",
            segment_id=context.segment.id,
        ) from exc
    adapter = resolution.adapter
    return _applicable(
        "lora",
        LoraConfigV1(
            family=context.family,
            backend=context.backend,
            model_filename=model_filename,
            lora_filename=family.filename,
            strength=family.strength,
            adapter_id=adapter.adapter_id,
            class_type=adapter.class_type,
            input_contract=adapter.input_contract,
            source=resolution.source,
            options=dict(resolution.options),
        ),
        "project",
    )


def _sigma(context: V6ConfigContext) -> EffectiveFeatureUse:
    sampling = getattr(context.draft.sampling, context.family)
    return _applicable(
        "sigma_schedule",
        SigmaScheduleConfigV1(
            shift_video=sampling.shift,
            shift_audio=sampling.audio_shift,
        ),
        "project",
    )


def _ck(context: V6ConfigContext) -> EffectiveFeatureUse:
    if "comfy_kitchen_attention" in context.draft.features.by_segment.get(context.segment.id, {}):
        raise V6FeatureConfigurationError(
            "Comfy Kitchen Attention is project-only.",
            code="feature_scope_unsupported",
            feature_id="comfy_kitchen_attention",
            segment_id=context.segment.id,
        )
    selection = context.draft.features.project.get("comfy_kitchen_attention")
    if selection is None:
        return _inactive("comfy_kitchen_attention", "disabled")
    if not selection.enabled:
        return _inactive("comfy_kitchen_attention", "disabled", "project")
    ComfyKitchenAttentionParamsV1.model_validate(selection.params)
    return _applicable(
        "comfy_kitchen_attention",
        ComfyKitchenAttentionParamsV1(),
        "project",
    )


def _conditioning(context: V6ConfigContext) -> EffectiveFeatureUse:
    return _applicable(
        "multimodal_conditioning",
        ConditioningConfigV1(
            segment_id=context.segment.id,
            family=context.family,
            recipe=timeline_segment_recipe(context.segment),
            sample_frames=context.sample_frames,
        ),
        "segment",
    )


def _continuity(context: V6ConfigContext) -> EffectiveFeatureUse:
    if context.predecessor_segment_id is None or context.continuity_source is None:
        return _inactive("continuity", "no_predecessor", "context")
    return _applicable(
        "continuity",
        ContinuityConfigV1(
            predecessor_segment_id=context.predecessor_segment_id,
            source=context.continuity_source,
            overlap_frames=context.continuity_prefix_frames,
            historical_take_id=context.historical_take_id,
        ),
        "context",
    )


def _sampling(context: V6ConfigContext) -> EffectiveFeatureUse:
    sampling = getattr(context.draft.sampling, context.family)
    return _applicable(
        "sampling_pipeline",
        SamplingPipelineConfigV1(
            steps=sampling.steps,
            seed=sampling.seed,
            random_seed=sampling.random_seed,
            sampler=sampling.sampler,
            scheduler=sampling.scheduler,
        ),
        "project",
    )


def _decode(context: V6ConfigContext) -> EffectiveFeatureUse:
    return _applicable(
        "video_decode",
        VideoDecodeConfigV1(
            visible_frames=context.visible_frames,
            continuity_prefix_frames=context.continuity_prefix_frames,
        ),
        "project",
    )


def _audio(context: V6ConfigContext) -> EffectiveFeatureUse:
    return _applicable(
        "audio_output",
        AudioOutputConfigV1(
            mode=context.segment.audio_mode,
            visible_frames=context.visible_frames,
            continuity_prefix_frames=context.continuity_prefix_frames,
        ),
        "segment",
    )


def _save(context: V6ConfigContext) -> EffectiveFeatureUse:
    return _applicable(
        "save_take",
        SaveTakeConfigV1(job_id=context.job_id, segment_id=context.segment.id),
        "context",
    )


@dataclass(frozen=True, slots=True)
class FunctionFeatureConfigResolver:
    id: str
    function: Any

    def resolve(self, context: V6ConfigContext) -> EffectiveFeatureUse:
        result = self.function(context)
        if result.feature_id != self.id:
            raise AssertionError("feature config resolver returned the wrong owner")
        return result


V6_CONFIG_RESOLVERS: dict[str, FeatureConfigResolver] = {
    feature_id: FunctionFeatureConfigResolver(feature_id, function)
    for feature_id, function in (
        ("auxiliary_models", _auxiliary),
        ("diffusion_model", _diffusion),
        ("execution_strategy", _execution),
        ("lora", _lora),
        ("sigma_schedule", _sigma),
        ("comfy_kitchen_attention", _ck),
        ("multimodal_conditioning", _conditioning),
        ("continuity", _continuity),
        ("sampling_pipeline", _sampling),
        ("video_decode", _decode),
        ("audio_output", _audio),
        ("save_take", _save),
    )
}


def default_lora_params() -> LoraFeatureParams:
    return LoraFeatureParams(
        by_family={
            "fl2va": LoraFamilyFeatureSelection(),
            "ref2va": LoraFamilyFeatureSelection(),
        }
    )


__all__ = [
    "AudioOutputConfigV1",
    "AuxiliaryModelsConfigV1",
    "ComfyKitchenAttentionParamsV1",
    "ConditioningConfigV1",
    "ContinuityConfigV1",
    "DiffusionModelConfigV1",
    "EffectiveFeatureUse",
    "EmptyFeatureParams",
    "ExecutionStrategyConfigV1",
    "FeatureConfigResolver",
    "LoraConfigV1",
    "SamplingPipelineConfigV1",
    "SaveTakeConfigV1",
    "SigmaScheduleConfigV1",
    "V6_CONFIG_RESOLVERS",
    "V6ConfigContext",
    "V6FeatureConfigurationError",
    "VideoDecodeConfigV1",
    "default_lora_params",
]
