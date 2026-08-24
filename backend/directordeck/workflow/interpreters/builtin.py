from __future__ import annotations

"""Registrable Stage-2 wrappers around the exact v4 fragment emitters."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel

from ...native_templates import NativeTemplateError, RaylightAttentionMode
from ...schemas import (
    DiffusionModelBinding,
    RuntimeSettings,
    SamplingConfig,
    UnifiedFL2VASegment,
    UnifiedRef2VASegment,
    UnifiedTimelineDraft,
    UnifiedTimelineSegment,
)
from ..contracts import (
    Backend,
    BoundedJsonValue,
    CapabilitySet,
    ContractModel,
    EdgeRef,
    FeatureEmission,
    FeatureResolution,
    FrozenMap,
    JsonValue,
    ModelFamily,
    Resource,
    ResolvedImplementationIdentity,
    ScopedGraphBuilderProtocol,
    TerminalRef,
)
from ..execution_hints import build_feature_execution_hints
from ..node_contracts import (
    require_native_node_contract,
    require_v5_node_contract,
)
from ..lora_factory import (
    LoraAdapterResolutionSource,
    LoraLoaderBindingKey,
)
from ._emitter import NativeEdge, ScopedBuilderEmitter
from ._types import SharedModelOutputs
from .audio_output import emit_audio_output
from .conditioning import emit_family_conditioning
from .continuity import emit_continuity
from .decode_video import emit_decode_video
from .lora import emit_raylight_lora, emit_standard_lora
from .raylight_model_path import (
    emit_raylight_model_load,
    emit_raylight_pool_intent,
    emit_raylight_sigma_shift,
)
from .sampling_raylight import emit_raylight_sampling
from .sampling_standard import emit_standard_sampling
from .save_take import emit_save_take
from .shared_models import emit_shared_models
from .standard_model_path import (
    emit_standard_model_device,
    emit_standard_model_load,
    emit_standard_sigma_shift,
)


class V4BuiltinParams(ContractModel):
    """The v4 templates carry no feature-owned parameter document."""


@dataclass(frozen=True, slots=True)
class V4BuiltinContext:
    """Complete graph-local input for all built-in v4 interpreters."""

    backend: Backend
    family: ModelFamily
    template_bundle_version: int
    settings: RuntimeSettings
    draft: UnifiedTimelineDraft
    segment: UnifiedTimelineSegment
    binding: DiffusionModelBinding
    sampling: SamplingConfig
    job_id: str
    visible_frames: int
    sample_frames: int
    continuity_prefix_frames: int
    lora_loader_node: str | None
    lora_adapter_id: str | None
    lora_loader_binding: LoraLoaderBindingKey | None
    lora_resolution_source: LoraAdapterResolutionSource | None
    lora_adapter_options: FrozenMap[str, JsonValue] | None = None
    raylight_namespace: str | None = None
    raylight_attention_mode: RaylightAttentionMode = "ck_int8"
    clear_raylight_vram_after_sampling: bool = False
    timeline_assembly_required: bool = False


_STATIC_IMPLEMENTATION_CLASSES: dict[str, tuple[str, ...]] = {
    "shared_models": (
        "CLIPLoader",
        "SelectCLIPDevice",
        "VAELoader",
        "SelectVAEDevice",
    ),
    "standard_model_load": ("UNETLoader",),
    "standard_model_device": ("SelectModelDevice",),
    "standard_sigma_shift": ("MiniMaxH3SigmaShift",),
    "attention_backend_override": ("DirectorStrictModelAttentionBackend",),
    "h3_low_vram_attention": ("DirectorStrictH3LowVramSagePatch",),
    "raylight_pool_intent": ("DirectorDeckRayInitializerAdvanced",),
    "raylight_model_load": ("DirectorDeckRayUNETLoader",),
    "raylight_sigma_shift": ("DirectorDeckRayMiniMaxH3SigmaShift",),
    "standard_sampling": (
        "BasicGuider",
        "BasicScheduler",
        "KSamplerSelect",
        "RandomNoise",
        "SamplerCustomAdvanced",
    ),
    "raylight_sampling": (
        "DirectorDeckRayBasicGuider",
        "DirectorDeckRayBasicScheduler",
        "KSamplerSelect",
        "DirectorDeckRayXFuserSamplerCustomAdvanced",
    ),
    "save_take": ("SaveVideo",),
}

# Bundle-5 progress is declared by the feature which owns each private node.
# These stage-only hints deliberately carry no numeric weight: ComfyUI exposes
# an exact current node but no byte/frame total while loading models, reading
# references or building conditioning. The shared hint builder owns labels so
# Bundle 5 and Bundle 6 cannot silently drift.
_PRE_SAMPLING_STAGE_FEATURES = frozenset(
    {
        "shared_models",
        "standard_model_load",
        "standard_model_device",
        "lora",
        "standard_sigma_shift",
        "raylight_pool_intent",
        "raylight_model_load",
        "raylight_sigma_shift",
        "family_conditioning",
        "continuity",
        "standard_sampling",
        "raylight_sampling",
    }
)
_SAMPLING_STAGE_FEATURES = frozenset(
    {"standard_sampling", "raylight_sampling"}
)


def _require_context(ctx: Any) -> V4BuiltinContext:
    if not isinstance(ctx, V4BuiltinContext):
        raise TypeError("v4 built-in interpreter requires V4BuiltinContext")
    if ctx.template_bundle_version not in {4, 5}:
        raise NativeTemplateError(
            "built-in compatibility fragments require template bundle version 4 or 5"
        )
    if ctx.segment.mode != ctx.family:
        raise NativeTemplateError("interpreter context family does not match segment")
    if ctx.backend == "raylight" and ctx.raylight_namespace is None:
        raise NativeTemplateError("RayLight interpreter context requires a namespace")
    if (ctx.binding.lora_name is None) != (ctx.lora_loader_node is None):
        raise NativeTemplateError(
            "LoRA binding and immutable loader resolution must agree"
        )
    lora_resolution_parts = (
        ctx.lora_loader_node,
        ctx.lora_adapter_id,
        ctx.lora_resolution_source,
    )
    if any(item is None for item in lora_resolution_parts) != all(
        item is None for item in lora_resolution_parts
    ):
        raise NativeTemplateError("LoRA adapter resolution is incomplete")
    if ctx.backend == "standard" and ctx.lora_loader_node is not None:
        if ctx.lora_loader_binding is None:
            raise NativeTemplateError("Standard LoRA requires an exact binding key")
    elif ctx.lora_loader_binding is not None:
        raise NativeTemplateError("RayLight/disabled LoRA cannot own a Standard key")
    if ctx.lora_loader_node == "DirectorDeckRayLoraLoader" and ctx.backend != "raylight":
        raise NativeTemplateError("Standard context cannot use RayLight LoRA loader")
    if (
        ctx.lora_loader_node is not None
        and ctx.lora_loader_node != "DirectorDeckRayLoraLoader"
        and ctx.backend != "standard"
    ):
        raise NativeTemplateError("RayLight context cannot use Standard LoRA loader")
    return ctx


def _resource_edge(
    inputs: Mapping[str, Resource],
    name: str,
    *,
    optional: bool = False,
) -> NativeEdge | None:
    resource = inputs.get(name)
    if resource is None:
        if optional:
            return None
        raise KeyError(f"required interpreter resource is missing: {name}")
    if not isinstance(resource.value, EdgeRef):
        raise TypeError(f"interpreter resource {name!r} must contain EdgeRef")
    return [resource.value.node_id, resource.value.output_slot]


def _publish_edge(
    builder: ScopedGraphBuilderProtocol,
    value: NativeEdge,
) -> EdgeRef:
    return builder.edge(str(value[0]), int(value[1]))


def _shared_from_inputs(
    inputs: Mapping[str, Resource],
    *,
    require_audio_vae: bool,
) -> SharedModelOutputs:
    clip = _resource_edge(inputs, "clip")
    video_vae = _resource_edge(inputs, "video_vae")
    audio_vae = _resource_edge(inputs, "audio_vae", optional=True)
    assert clip is not None and video_vae is not None
    if require_audio_vae and audio_vae is None:
        raise KeyError("required interpreter resource is missing: 'audio_vae'")
    return SharedModelOutputs(clip, video_vae, audio_vae)


def _unique_class_types(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _implementation_class_types(
    feature_id: str,
    ctx: V4BuiltinContext,
) -> tuple[str, ...]:
    """Purely resolve every unique node implementation one feature may emit."""

    if feature_id == "lora":
        if ctx.lora_loader_node is None:
            raise NativeTemplateError("disabled LoRA must not be resolved")
        return (ctx.lora_loader_node,)
    if feature_id == "family_conditioning":
        classes: list[str] = []
        if isinstance(ctx.segment, UnifiedFL2VASegment):
            if ctx.segment.first_image is not None:
                classes.append("LoadImage")
            if ctx.segment.last_image is not None:
                classes.append("LoadImage")
            classes.append("MiniMaxH3ImageToVideo")
            return _unique_class_types(classes)
        if not isinstance(ctx.segment, UnifiedRef2VASegment):
            raise NativeTemplateError(
                f"unsupported conditioning segment: {type(ctx.segment).__name__}"
            )
        if ctx.segment.source_video is not None:
            classes.extend(("LoadVideo", "Video Slice", "GetVideoComponents"))
        if ctx.segment.reference_images:
            classes.append("LoadImage")
        if ctx.segment.reference_videos:
            classes.extend(("LoadVideo", "GetVideoComponents"))
        if ctx.segment.reference_audios:
            classes.append("LoadAudio")
        classes.append("MiniMaxH3ReferenceToVideo")
        return _unique_class_types(classes)
    if feature_id == "continuity":
        classes = ["LoadVideo", "GetVideoComponents", "ImageFromBatch"]
        if ctx.segment.audio_mode == "generate":
            classes.append("TrimAudioDuration")
        classes.append("MiniMaxH3AddGuide")
        if (
            isinstance(ctx.segment, UnifiedFL2VASegment)
            and ctx.segment.last_image is not None
        ):
            classes.extend(("LoadImage", "MiniMaxH3AddGuide"))
        return _unique_class_types(classes)
    if feature_id == "decode_video":
        classes = ["VAEDecode"]
        if ctx.continuity_prefix_frames:
            classes.append("ImageFromBatch")
        return tuple(classes)
    if feature_id == "audio_output":
        classes = []
        if ctx.segment.audio_mode == "generate":
            classes.append("VAEDecodeAudio")
            if ctx.continuity_prefix_frames:
                classes.append("TrimAudioDuration")
        classes.append("CreateVideo")
        return tuple(classes)
    try:
        return _STATIC_IMPLEMENTATION_CLASSES[feature_id]
    except KeyError as exc:
        raise AssertionError(f"unknown built-in feature: {feature_id}") from exc


def catalog_implementation_alternatives(
    feature_id: str,
    backend: Backend,
) -> tuple[tuple[ModelFamily, tuple[str, ...]], ...]:
    """Return conservative live-host alternatives for static catalog checks.

    Catalog has no segment/model context, so it may only declare a feature
    unavailable when every supported family/adapter alternative is impossible.
    Contextual preflight still performs the exact resolution.
    """

    if feature_id == "lora":
        class_alternatives = (
            (
                "MiniMaxH3TurboLoRA",
            ),
            ("LoraLoaderBypassModelOnly",),
            ("LoraLoaderModelOnly",),
        ) if backend == "standard" else (("DirectorDeckRayLoraLoader",),)
    elif feature_id == "family_conditioning":
        return (
            ("fl2va", ("MiniMaxH3ImageToVideo",)),
            ("ref2va", ("MiniMaxH3ReferenceToVideo",)),
        )
    elif feature_id == "continuity":
        class_alternatives = (
            ("LoadVideo", "GetVideoComponents", "ImageFromBatch", "MiniMaxH3AddGuide"),
        )
    elif feature_id == "decode_video":
        class_alternatives = (("VAEDecode",),)
    elif feature_id == "audio_output":
        class_alternatives = (("CreateVideo",),)
    else:
        try:
            class_alternatives = (_STATIC_IMPLEMENTATION_CLASSES[feature_id],)
        except KeyError as exc:
            raise KeyError(f"no static catalog alternatives for {feature_id!r}") from exc
    return tuple(
        (family, classes)
        for family in ("fl2va", "ref2va")
        for classes in class_alternatives
    )


def _binding_key(feature_id: str, class_type: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]", "_", class_type)
    return f"{feature_id}.{normalized}"


def builtin_implementation_identity(
    feature_id: str,
    class_type: str,
) -> ResolvedImplementationIdentity:
    """Return the release-owned implementation identity for a built-in.

    Catalog projections must use the same binding-key construction as the
    contextual interpreter.  Otherwise two views of the same adapter produce
    different adapter fingerprints even though they name the same node.
    """

    contract = require_native_node_contract(class_type)
    return ResolvedImplementationIdentity(
        role="node",
        class_type=class_type,
        implementation_id=contract.contract_id,
        semantic_version=contract.semantic_version,
        runtime_fingerprint=contract.supported_runtime_fingerprints[0],
        binding_key=_binding_key(feature_id, class_type),
    )


def builtin_required_capability_ids(
    *,
    feature_id: str,
    class_types: tuple[str, ...],
    timeline_assembly_required: bool,
) -> tuple[str, ...]:
    """Declare exact node and non-node requirements for one v4 feature.

    ``ffprobe`` authenticates every completed take.  ``ffmpeg`` is only used
    by Director's post-run timeline assembly, which is skipped for per-segment
    export, partial selection, and a one-segment ``all`` export.  Ray package
    requirements are owned only by adapters from the reviewed RayLight pack.
    """

    capability_ids = [
        "node." + re.sub(r"[^A-Za-z0-9_.:-]", "_", class_type)
        for class_type in class_types
    ]
    def contract_for(class_type: str):
        try:
            return require_native_node_contract(class_type)
        except KeyError:
            return require_v5_node_contract(class_type)

    raylight_classes = tuple(
        class_type
        for class_type in class_types
        if "custom_nodes.DirectorDeck-RayLight"
        in contract_for(class_type).allowed_python_modules
    )
    if raylight_classes:
        capability_ids.extend(
            (
                "raylight.installation",
                "raylight.cleanup",
                "package.ray",
            )
        )
    if "DirectorDeckRayXFuserSamplerCustomAdvanced" in raylight_classes:
        capability_ids.append("package.xfuser")
    if feature_id == "save_take":
        capability_ids.append("media.ffprobe")
        if timeline_assembly_required:
            capability_ids.append("media.ffmpeg")
    elif feature_id == "attention_backend_override":
        # Catalog evaluates the default authorable selection. Exact selected
        # modes are re-evaluated by the v5 interpreter during preflight.
        capability_ids.append("runtime.strict_attention.pytorch")
    elif feature_id == "h3_low_vram_attention":
        # Catalog has no saved placement binding, so it consumes the provider's
        # derived any-device evidence; preflight/compile use the exact device.
        capability_ids.append("runtime.strict_h3_sage.any")
    return tuple(dict.fromkeys(capability_ids))


EntryOutputs = dict[str, EdgeRef | TerminalRef]
EntryEmitter = Callable[
    [ScopedGraphBuilderProtocol, Mapping[str, Resource], V4BuiltinContext],
    EntryOutputs,
]


def _execution_hints(
    *,
    feature_id: str,
    outputs: EntryOutputs,
    builder: ScopedGraphBuilderProtocol,
    context: V4BuiltinContext,
) -> tuple[tuple[BoundedJsonValue, ...], tuple[BoundedJsonValue, ...]]:
    """Publish the bundle-5 equivalent of the frozen legacy UX scan."""

    if context.template_bundle_version < 5:
        return (), ()
    return build_feature_execution_hints(
        feature_id=feature_id,
        outputs=outputs,
        builder=builder,
        pre_sampling_features=_PRE_SAMPLING_STAGE_FEATURES,
        sampling_features=_SAMPLING_STAGE_FEATURES,
    )


def _emit_shared_models_entry(
    builder: ScopedGraphBuilderProtocol,
    _inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    shared = emit_shared_models(ScopedBuilderEmitter(builder), context.settings)
    outputs = {
        "clip": _publish_edge(builder, shared.clip),
        "video_vae": _publish_edge(builder, shared.video_vae),
    }
    if context.family == "ref2va" or context.segment.audio_mode == "generate":
        outputs["audio_vae"] = _publish_edge(builder, shared.audio_vae)
    return outputs


def _emit_standard_model_load_entry(
    builder: ScopedGraphBuilderProtocol,
    _inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    model = emit_standard_model_load(
        ScopedBuilderEmitter(builder), context.binding
    )
    return {"model": _publish_edge(builder, model)}


def _emit_standard_model_device_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    model = _resource_edge(inputs, "model")
    assert model is not None
    emitted = emit_standard_model_device(
        ScopedBuilderEmitter(builder), model, context.binding
    )
    return {"model": _publish_edge(builder, emitted)}


def _emit_lora_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    emitter = ScopedBuilderEmitter(builder)
    if context.backend == "standard":
        model = _resource_edge(inputs, "model")
        assert model is not None
        assert context.lora_loader_node is not None
        assert context.lora_adapter_id is not None
        emitted = emit_standard_lora(
            emitter,
            model,
            context.binding,
            adapter_id=context.lora_adapter_id,
            loader_node=context.lora_loader_node,
            adapter_options=context.lora_adapter_options or FrozenMap(),
        )
        return {"model": _publish_edge(builder, emitted)}
    emitted = emit_raylight_lora(emitter, context.binding)
    return {"ray_lora": _publish_edge(builder, emitted)}


def _emit_standard_sigma_shift_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    model = _resource_edge(inputs, "model")
    assert model is not None
    emitted = emit_standard_sigma_shift(
        ScopedBuilderEmitter(builder), model, context.sampling
    )
    return {"model": _publish_edge(builder, emitted)}


def _emit_raylight_pool_intent_entry(
    builder: ScopedGraphBuilderProtocol,
    _inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    assert context.raylight_namespace is not None
    emitted = emit_raylight_pool_intent(
        ScopedBuilderEmitter(builder),
        context.binding,
        namespace=context.raylight_namespace,
        clear_vram_after_sampling=context.clear_raylight_vram_after_sampling,
        attention_mode=context.raylight_attention_mode,
        enforce_attention_topology=context.template_bundle_version >= 5,
    )
    return {"ray_actors_init": _publish_edge(builder, emitted)}


def _emit_raylight_model_load_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    pool_intent = _resource_edge(inputs, "ray_actors_init")
    lora = _resource_edge(inputs, "ray_lora", optional=True)
    assert pool_intent is not None
    emitted = emit_raylight_model_load(
        ScopedBuilderEmitter(builder),
        pool_intent,
        context.binding,
        lora=lora,
    )
    return {"model": _publish_edge(builder, emitted)}


def _emit_raylight_sigma_shift_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    model = _resource_edge(inputs, "model")
    assert model is not None
    emitted = emit_raylight_sigma_shift(
        ScopedBuilderEmitter(builder), model, context.sampling
    )
    return {"model": _publish_edge(builder, emitted)}


def _emit_family_conditioning_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    emitted = emit_family_conditioning(
        ScopedBuilderEmitter(builder),
        context.segment,
        context.draft,
        _shared_from_inputs(
            inputs,
            require_audio_vae=context.family == "ref2va",
        ),
        frames=context.sample_frames,
    )
    outputs: EntryOutputs = {
        "conditioning": _publish_edge(builder, emitted.conditioning),
        "latent": _publish_edge(builder, emitted.latent),
    }
    if emitted.source_audio is not None and context.segment.audio_mode == "source":
        outputs["source_audio"] = _publish_edge(builder, emitted.source_audio)
    return outputs


def _emit_continuity_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    conditioning = _resource_edge(inputs, "conditioning")
    latent = _resource_edge(inputs, "latent")
    video_vae = _resource_edge(inputs, "video_vae")
    audio_vae = _resource_edge(inputs, "audio_vae", optional=True)
    assert conditioning is not None and latent is not None
    assert video_vae is not None
    if context.segment.audio_mode == "generate" and audio_vae is None:
        raise KeyError("required interpreter resource is missing: 'audio_vae'")
    emitted = emit_continuity(
        ScopedBuilderEmitter(builder),
        conditioning=conditioning,
        latent=latent,
        segment=context.segment,
        draft=context.draft,
        video_vae=video_vae,
        audio_vae=audio_vae,
        visible_frames=context.visible_frames,
        overlap_frames=context.continuity_prefix_frames,
    )
    return {"conditioning": _publish_edge(builder, emitted.conditioning)}


def _emit_standard_sampling_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    model = _resource_edge(inputs, "model")
    conditioning = _resource_edge(inputs, "conditioning")
    latent = _resource_edge(inputs, "latent")
    assert model is not None and conditioning is not None and latent is not None
    emitted = emit_standard_sampling(
        ScopedBuilderEmitter(builder),
        model=model,
        conditioning=conditioning,
        latent=latent,
        sampling=context.sampling,
        seed=context.sampling.seed,
    )
    return {"samples": _publish_edge(builder, emitted)}


def _emit_raylight_sampling_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    model = _resource_edge(inputs, "model")
    conditioning = _resource_edge(inputs, "conditioning")
    latent = _resource_edge(inputs, "latent")
    assert model is not None and conditioning is not None and latent is not None
    emitted = emit_raylight_sampling(
        ScopedBuilderEmitter(builder),
        ray_actors=model,
        conditioning=conditioning,
        latent=latent,
        sampling=context.sampling,
        seed=context.sampling.seed,
    )
    return {"samples": _publish_edge(builder, emitted)}


def _emit_decode_video_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    samples = _resource_edge(inputs, "samples")
    video_vae = _resource_edge(inputs, "video_vae")
    assert samples is not None and video_vae is not None
    emitted = emit_decode_video(
        ScopedBuilderEmitter(builder),
        samples=samples,
        video_vae=video_vae,
        visible_frames=context.visible_frames,
        continuity_prefix_frames=context.continuity_prefix_frames,
    )
    return {"frames": _publish_edge(builder, emitted)}


def _emit_audio_output_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    frames = _resource_edge(inputs, "frames")
    samples = _resource_edge(inputs, "samples")
    audio_vae = _resource_edge(inputs, "audio_vae", optional=True)
    assert frames is not None and samples is not None
    if context.segment.audio_mode == "generate" and audio_vae is None:
        raise KeyError("required interpreter resource is missing: 'audio_vae'")
    emitted = emit_audio_output(
        ScopedBuilderEmitter(builder),
        images=frames,
        samples=samples,
        source_audio=_resource_edge(inputs, "source_audio", optional=True),
        draft=context.draft,
        audio_vae=audio_vae,
        segment=context.segment,
        visible_frames=context.visible_frames,
        continuity_prefix_frames=context.continuity_prefix_frames,
    )
    return {"video": _publish_edge(builder, emitted.video)}


def _emit_save_take_entry(
    builder: ScopedGraphBuilderProtocol,
    inputs: Mapping[str, Resource],
    context: V4BuiltinContext,
) -> EntryOutputs:
    video = _resource_edge(inputs, "video")
    assert video is not None
    node_id = emit_save_take(
        ScopedBuilderEmitter(builder),
        video=video,
        job_id=context.job_id,
        segment=context.segment,
    )
    return {"take_output": builder.terminal(node_id)}


_ENTRY_EMITTERS: dict[str, EntryEmitter] = {
    "shared_models": _emit_shared_models_entry,
    "standard_model_load": _emit_standard_model_load_entry,
    "standard_model_device": _emit_standard_model_device_entry,
    "lora": _emit_lora_entry,
    "standard_sigma_shift": _emit_standard_sigma_shift_entry,
    "raylight_pool_intent": _emit_raylight_pool_intent_entry,
    "raylight_model_load": _emit_raylight_model_load_entry,
    "raylight_sigma_shift": _emit_raylight_sigma_shift_entry,
    "family_conditioning": _emit_family_conditioning_entry,
    "continuity": _emit_continuity_entry,
    "standard_sampling": _emit_standard_sampling_entry,
    "raylight_sampling": _emit_raylight_sampling_entry,
    "decode_video": _emit_decode_video_entry,
    "audio_output": _emit_audio_output_entry,
    "save_take": _emit_save_take_entry,
}


@dataclass(frozen=True, slots=True)
class V4BuiltinInterpreter:
    """One exact, registry-compatible entry backed by a single-duty emitter."""

    id: str
    version: int = 1
    mode: Literal["needed", "switch"] = "needed"

    def validate_params(self, params: BaseModel, ctx: Any) -> None:
        if not isinstance(params, V4BuiltinParams):
            raise TypeError("v4 built-ins require V4BuiltinParams")
        _require_context(ctx)

    def resolve(self, params: BaseModel, ctx: Any) -> FeatureResolution:
        self.validate_params(params, ctx)
        context = _require_context(ctx)
        class_types = _implementation_class_types(self.id, context)
        resolution_details: dict[str, Any] = {
            "source": "legacy_v4_exact_native_fragment",
            "backend": context.backend,
            "family": context.family,
        }
        if self.id == "lora":
            if (
                context.lora_adapter_id is None
                or context.lora_resolution_source is None
            ):
                raise NativeTemplateError("LoRA adapter evidence is incomplete")
            resolution_details.update(
                adapter_id=context.lora_adapter_id,
                binding=(
                    context.lora_loader_binding.model_dump(mode="json")
                    if context.lora_loader_binding is not None
                    else {
                        "backend": "raylight",
                        "family": context.family,
                        "lora_filename": context.binding.lora_name,
                    }
                ),
                mapping_source=context.lora_resolution_source,
                strength=context.binding.lora_strength,
                loader_options=dict(context.lora_adapter_options or {}),
            )
        return FeatureResolution(
            state="active",
            implementations=tuple(
                builtin_implementation_identity(self.id, class_type)
                for class_type in class_types
            ),
            resolution_details=resolution_details,
        )

    def required_capabilities(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> CapabilitySet:
        self.validate_params(params, ctx)
        if resolution != self.resolve(params, ctx):
            raise NativeTemplateError("feature resolution does not match context")
        context = _require_context(ctx)
        return CapabilitySet(
            ids=builtin_required_capability_ids(
                feature_id=self.id,
                class_types=tuple(
                    item.class_type for item in resolution.implementations
                ),
                timeline_assembly_required=(
                    context.timeline_assembly_required
                ),
            )
        )

    def cache_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue:
        self.validate_params(params, ctx)
        if resolution != self.resolve(params, ctx):
            raise NativeTemplateError("feature resolution does not match context")
        context = _require_context(ctx)
        # Stage 2 retains the legacy whole-take cache authority. This identity
        # records only the exact built-in selection and is not a new cache key.
        return {
            "authority": "legacy_v4_take_fingerprint",
            "feature_id": self.id,
            "backend": context.backend,
            "family": context.family,
            "implementations": [
                item.runtime_fingerprint for item in resolution.implementations
            ],
        }

    def runtime_pool_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> BoundedJsonValue | None:
        self.validate_params(params, ctx)
        if resolution != self.resolve(params, ctx):
            raise NativeTemplateError("feature resolution does not match context")
        context = _require_context(ctx)
        if self.id == "raylight_pool_intent":
            profile = context.binding.raylight
            return {
                "namespace": context.raylight_namespace,
                "gpu_select": list(profile.gpu_select),
                "ulysses_degree": profile.ulysses_degree,
                "ring_degree": profile.ring_degree,
                "cfg_degree": profile.cfg_degree,
                "dp_degree": profile.dp_degree,
                "fsdp": profile.fsdp,
                "cpu_offload": profile.cpu_offload,
                "clear_vram_after_sampling": (
                    context.clear_raylight_vram_after_sampling
                ),
            }
        if self.id == "raylight_sigma_shift":
            return {
                "shift_video": context.sampling.shift,
                "shift_audio": context.sampling.audio_shift,
            }
        return None

    def emit(
        self,
        builder: ScopedGraphBuilderProtocol,
        inputs: Mapping[str, Resource],
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> FeatureEmission:
        self.validate_params(params, ctx)
        if resolution != self.resolve(params, ctx):
            raise NativeTemplateError("feature resolution does not match context")
        context = _require_context(ctx)
        handler = _ENTRY_EMITTERS.get(self.id)
        if handler is None:
            raise AssertionError(f"unknown built-in interpreter: {self.id}")
        outputs = handler(builder, inputs, context)
        progress_hints, preview_hints = _execution_hints(
            feature_id=self.id,
            outputs=outputs,
            builder=builder,
            context=context,
        )

        return FeatureEmission(
            outputs=outputs,
            progress_hints=progress_hints,
            preview_hints=preview_hints,
            emission_details={
                "source": "legacy_v4_exact_native_fragment",
                "feature_id": self.id,
            },
        )


_BUILTIN_IDS = (
    "shared_models",
    "standard_model_load",
    "standard_model_device",
    "lora",
    "standard_sigma_shift",
    "family_conditioning",
    "continuity",
    "standard_sampling",
    "decode_video",
    "audio_output",
    "save_take",
    "raylight_pool_intent",
    "raylight_model_load",
    "raylight_sigma_shift",
    "raylight_sampling",
)

_SWITCH_IDS = frozenset({"lora", "continuity"})


def builtin_interpreters() -> tuple[V4BuiltinInterpreter, ...]:
    return tuple(
        V4BuiltinInterpreter(
            id=feature_id,
            mode="switch" if feature_id in _SWITCH_IDS else "needed",
        )
        for feature_id in _BUILTIN_IDS
    )


def builtin_interpreter_map() -> dict[str, V4BuiltinInterpreter]:
    return {interpreter.id: interpreter for interpreter in builtin_interpreters()}


__all__ = [
    "V4BuiltinContext",
    "V4BuiltinInterpreter",
    "V4BuiltinParams",
    "builtin_required_capability_ids",
    "builtin_interpreter_map",
    "builtin_interpreters",
    "catalog_implementation_alternatives",
]
