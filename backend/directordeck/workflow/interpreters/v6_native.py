from __future__ import annotations

"""Thin Bundle-6 adapters over the existing reviewed native emitters."""

from collections.abc import Callable, Mapping
from typing import Any

from ...schemas import StrictModel
from ..contracts import (
    EdgeRef,
    FeatureEmission,
    Resource,
    ScopedGraphBuilderProtocol,
    TerminalRef,
)
from ..feature_config import (
    AuxiliaryModelsConfigV1,
    LoraConfigV1,
    V6ConfigContext,
)
from ..v6_registry import FeatureResolutionV6, ResolvedEarlierFeatures
from ._emitter import NativeEdge, ScopedBuilderEmitter
from ._types import SharedModelOutputs
from .audio_output import emit_audio_output
from .comfy_kitchen_attention import (
    emit_raylight_comfy_kitchen_attention,
    emit_standard_comfy_kitchen_attention,
)
from .conditioning import emit_family_conditioning
from .continuity import emit_continuity
from .decode_video import emit_decode_video
from .execution_strategy import (
    diffusion_binding,
    emit_execution_strategy,
)
from .lora import emit_raylight_lora, emit_standard_lora
from .raylight_model_path import emit_raylight_model_load, emit_raylight_sigma_shift
from .sampling_raylight import emit_raylight_sampling
from .sampling_standard import emit_standard_sampling
from .save_take import emit_save_take
from .shared_models import emit_auxiliary_models
from .standard_model_path import emit_standard_model_load, emit_standard_sigma_shift
from .v6_execution_hints import attach_v6_execution_hints


EntryOutputs = dict[str, EdgeRef | TerminalRef]
Emitter = Callable[
    [
        ScopedGraphBuilderProtocol,
        Mapping[str, Resource],
        StrictModel,
        FeatureResolutionV6,
        ResolvedEarlierFeatures,
        V6ConfigContext,
    ],
    FeatureEmission,
]


def _edge(inputs: Mapping[str, Resource], name: str, *, optional: bool = False) -> NativeEdge | None:
    resource = inputs.get(name)
    if resource is None:
        if optional:
            return None
        raise KeyError(f"required feature resource is missing: {name}")
    if not isinstance(resource.value, EdgeRef):
        raise TypeError(f"feature resource must contain an edge: {name}")
    return [resource.value.node_id, resource.value.output_slot]


def _publish(builder: ScopedGraphBuilderProtocol, value: NativeEdge) -> EdgeRef:
    return builder.edge(str(value[0]), int(value[1]))


def _shared(inputs: Mapping[str, Resource], *, require_audio: bool) -> SharedModelOutputs:
    clip = _edge(inputs, "clip")
    video = _edge(inputs, "video_vae")
    audio = _edge(inputs, "audio_vae", optional=True)
    if clip is None or video is None or (require_audio and audio is None):
        raise KeyError("feature requires the selected auxiliary models")
    return SharedModelOutputs(clip, video, audio)


def _auxiliary(
    builder: ScopedGraphBuilderProtocol,
    _inputs: Mapping[str, Resource],
    config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    typed = AuxiliaryModelsConfigV1.model_validate(config)
    emitted = emit_auxiliary_models(
        ScopedBuilderEmitter(builder),
        **typed.model_dump(mode="python"),
    )
    outputs: EntryOutputs = {
        "clip": _publish(builder, emitted.clip),
        "video_vae": _publish(builder, emitted.video_vae),
    }
    if emitted.audio_vae is not None:
        outputs["audio_vae"] = _publish(builder, emitted.audio_vae)
    return FeatureEmission(outputs=outputs)


def _diffusion(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    _config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    emitter = ScopedBuilderEmitter(builder)
    binding = diffusion_binding(context)
    if context.backend == "standard":
        model = emit_standard_model_load(emitter, binding)
    else:
        actors = _edge(inputs, "ray_actors_init")
        assert actors is not None
        model = emit_raylight_model_load(
            emitter,
            actors,
            binding,
            lora=_edge(inputs, "ray_lora", optional=True),
        )
    return FeatureEmission(outputs={"model": _publish(builder, model)})


def _execution(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    config: StrictModel,
    resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    return emit_execution_strategy(builder, inputs, config, resolution, context)


def _lora(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    typed = LoraConfigV1.model_validate(config)
    binding = diffusion_binding(context, typed)
    emitter = ScopedBuilderEmitter(builder)
    if context.backend == "standard":
        model = _edge(inputs, "model")
        assert model is not None
        emitted = emit_standard_lora(
            emitter,
            model,
            binding,
            adapter_id=typed.adapter_id,
            loader_node=typed.class_type,
            adapter_options=typed.options,
        )
        return FeatureEmission(outputs={"model": _publish(builder, emitted)})
    emitted = emit_raylight_lora(emitter, binding)
    return FeatureEmission(outputs={"ray_lora": _publish(builder, emitted)})


def _ck(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    if context.backend == "standard":
        return emit_standard_comfy_kitchen_attention(builder, dict(inputs), config)
    return emit_raylight_comfy_kitchen_attention()


def _sigma(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    _config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    model = _edge(inputs, "model")
    assert model is not None
    sampling = getattr(context.draft.sampling, context.family)
    emitter = ScopedBuilderEmitter(builder)
    emitted = (
        emit_standard_sigma_shift(emitter, model, sampling)
        if context.backend == "standard"
        else emit_raylight_sigma_shift(emitter, model, sampling)
    )
    return FeatureEmission(outputs={"model": _publish(builder, emitted)})


def _conditioning(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    _config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    emitted = emit_family_conditioning(
        ScopedBuilderEmitter(builder),
        context.segment,
        context.draft,  # type: ignore[arg-type]
        _shared(inputs, require_audio=context.family == "ref2va"),
        frames=context.sample_frames,
    )
    outputs: EntryOutputs = {
        "conditioning": _publish(builder, emitted.conditioning),
        "latent": _publish(builder, emitted.latent),
    }
    if emitted.source_audio is not None and context.segment.audio_mode == "source":
        outputs["source_audio"] = _publish(builder, emitted.source_audio)
    return FeatureEmission(outputs=outputs)


def _continuity(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    _config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    conditioning = _edge(inputs, "conditioning")
    latent = _edge(inputs, "latent")
    video_vae = _edge(inputs, "video_vae")
    assert conditioning is not None and latent is not None and video_vae is not None
    emitted = emit_continuity(
        ScopedBuilderEmitter(builder),
        conditioning=conditioning,
        latent=latent,
        segment=context.segment,
        draft=context.draft,  # type: ignore[arg-type]
        video_vae=video_vae,
        audio_vae=_edge(inputs, "audio_vae", optional=True),
        visible_frames=context.visible_frames,
        overlap_frames=context.continuity_prefix_frames,
    )
    return FeatureEmission(outputs={"conditioning": _publish(builder, emitted.conditioning)})


def _sampling(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    _config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    model, conditioning, latent = (
        _edge(inputs, "model"),
        _edge(inputs, "conditioning"),
        _edge(inputs, "latent"),
    )
    assert model is not None and conditioning is not None and latent is not None
    sampling = getattr(context.draft.sampling, context.family)
    emitter = ScopedBuilderEmitter(builder)
    emitted = (
        emit_standard_sampling(
            emitter,
            model=model,
            conditioning=conditioning,
            latent=latent,
            sampling=sampling,
            seed=sampling.seed,
        )
        if context.backend == "standard"
        else emit_raylight_sampling(
            emitter,
            ray_actors=model,
            conditioning=conditioning,
            latent=latent,
            sampling=sampling,
            seed=sampling.seed,
        )
    )
    return FeatureEmission(outputs={"samples": _publish(builder, emitted)})


def _decode(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    _config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    samples, video_vae = _edge(inputs, "samples"), _edge(inputs, "video_vae")
    assert samples is not None and video_vae is not None
    emitted = emit_decode_video(
        ScopedBuilderEmitter(builder),
        samples=samples,
        video_vae=video_vae,
        visible_frames=context.visible_frames,
        continuity_prefix_frames=context.continuity_prefix_frames,
    )
    return FeatureEmission(outputs={"frames": _publish(builder, emitted)})


def _audio(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    _config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    frames, samples = _edge(inputs, "frames"), _edge(inputs, "samples")
    assert frames is not None and samples is not None
    emitted = emit_audio_output(
        ScopedBuilderEmitter(builder),
        images=frames,
        samples=samples,
        source_audio=_edge(inputs, "source_audio", optional=True),
        draft=context.draft,  # type: ignore[arg-type]
        audio_vae=_edge(inputs, "audio_vae", optional=True),
        segment=context.segment,
        visible_frames=context.visible_frames,
        continuity_prefix_frames=context.continuity_prefix_frames,
    )
    return FeatureEmission(outputs={"video": _publish(builder, emitted.video)})


def _save(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    _config: StrictModel,
    _resolution: FeatureResolutionV6,
    _dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    video = _edge(inputs, "video")
    assert video is not None
    node_id = emit_save_take(
        ScopedBuilderEmitter(builder),
        video=video,
        job_id=context.job_id,
        segment=context.segment,
    )
    return FeatureEmission(outputs={"take_output": builder.terminal(node_id)})


_HANDLERS: dict[tuple[str, str], Emitter] = {
    (feature, backend): handler
    for feature, handler in (
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
    for backend in ("standard", "raylight")
}


def emit_v6_native_feature(
    feature_id: str,
    backend: str,
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    config: StrictModel,
    resolution: FeatureResolutionV6,
    dependencies: ResolvedEarlierFeatures,
    context: V6ConfigContext,
) -> FeatureEmission:
    emission = _HANDLERS[(feature_id, backend)](
        builder,
        inputs,
        config,
        resolution,
        dependencies,
        context,
    )
    return attach_v6_execution_hints(feature_id, builder, emission)


__all__ = ["emit_v6_native_feature"]
