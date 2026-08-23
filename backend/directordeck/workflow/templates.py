from __future__ import annotations

"""Immutable v4 template identities used by the behavior-preserving compiler.

The order below follows the nodes emitted by the legacy v4 compiler.  It is
intentionally not the aspirational order in the architecture prose: moving
Standard LoRA after SigmaShift, or DirectorDeckRayLoraLoader after
DirectorDeckRayUNETLoader, would
change the exact prompt and violate the Stage-2 migration gate.
"""

from collections.abc import Iterable

from .contracts import (
    Backend,
    ControlTemplate,
    ControlTemplateSet,
    FeatureTemplateEntry,
    GraphPhase,
    ModelFamily,
    ResourceReadDeclaration,
    ResourceWriteDeclaration,
    SegmentTemplate,
    SegmentTemplateSet,
    TemplateBundle,
)


_FAMILIES: tuple[ModelFamily, ...] = ("fl2va", "ref2va")


def _read(
    name: str,
    type: str,
    *,
    required: bool = True,
) -> ResourceReadDeclaration:
    return ResourceReadDeclaration(name=name, type=type, required=required)


def _write(
    name: str,
    type: str,
    operation: str = "define",
    *,
    required: bool = True,
) -> ResourceWriteDeclaration:
    return ResourceWriteDeclaration(
        name=name,
        type=type,
        operation=operation,
        required=required,
    )


def _entry(
    feature_id: str,
    *,
    title: str,
    description: str,
    mode: str,
    phase: GraphPhase,
    backend: Backend,
    reads: Iterable[ResourceReadDeclaration] = (),
    writes: Iterable[ResourceWriteDeclaration] = (),
) -> FeatureTemplateEntry:
    return FeatureTemplateEntry(
        id=feature_id,
        version=1,
        title=title,
        description=description,
        mode=mode,
        graph_phase=phase,
        reads=tuple(reads),
        writes=tuple(writes),
        params_schema={"type": "object", "additionalProperties": False},
        defaults={},
        cache_policy={"identity": "legacy_v4_exact_inputs"},
        backends=(backend,),
        families=_FAMILIES,
        scopes=("segment",),
        ui={"visibility": "internal_v4"},
    )


_SHARED_WRITES = (
    _write("clip", "CLIP"),
    _write("video_vae", "VAE"),
    _write("audio_vae", "VAE", required=False),
)

_CONDITIONING_READS = (
    _read("clip", "CLIP"),
    _read("video_vae", "VAE"),
    _read("audio_vae", "VAE", required=False),
)
_CONDITIONING_WRITES = (
    _write("conditioning", "CONDITIONING"),
    _write("latent", "LATENT"),
    # Only Ref2VA source-video routes publish this declared optional resource.
    # A later optional read is deterministic when the resource is absent.
    _write("source_audio", "AUDIO", required=False),
)

_CONTINUITY_READS = (
    _read("conditioning", "CONDITIONING"),
    _read("latent", "LATENT"),
    _read("video_vae", "VAE"),
    _read("audio_vae", "VAE", required=False),
)
_CONTINUITY_WRITES = (
    _write("conditioning", "CONDITIONING", "replace"),
)

_DECODE_READS = (
    _read("samples", "LATENT"),
    _read("video_vae", "VAE"),
)
_AUDIO_OUTPUT_READS = (
    _read("frames", "IMAGE"),
    _read("samples", "LATENT"),
    _read("audio_vae", "VAE", required=False),
    _read("source_audio", "AUDIO", required=False),
)


V4_STANDARD_SEGMENT_TEMPLATE = SegmentTemplate(
    id="h3_standard_segment",
    revision=1,
    entries=(
        _entry(
            "shared_models",
            title="Shared CLIP and VAEs",
            description="Load and place the shared CLIP, video VAE and audio VAE.",
            mode="needed",
            phase="bootstrap",
            backend="standard",
            writes=_SHARED_WRITES,
        ),
        _entry(
            "standard_model_load",
            title="Standard model load",
            description="Load the selected H3 diffusion model with UNETLoader.",
            mode="needed",
            phase="model_load",
            backend="standard",
            writes=(_write("model", "MODEL"),),
        ),
        _entry(
            "standard_model_device",
            title="Standard model placement",
            description="Apply the authoritative Standard model device placement.",
            mode="needed",
            phase="model_prepare",
            backend="standard",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
        ),
        # Legacy v4 emits its selected LoRA loader before SigmaShift.
        _entry(
            "lora",
            title="Standard LoRA",
            description="Apply the current fail-closed Standard LoRA adapter.",
            mode="switch",
            phase="model_prepare",
            backend="standard",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
        ),
        _entry(
            "standard_sigma_shift",
            title="Standard Sigma shift",
            description="Apply the current H3 video and audio sigma shifts.",
            mode="needed",
            phase="model_prepare",
            backend="standard",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
        ),
        _entry(
            "family_conditioning",
            title="H3 family conditioning",
            description="Build the FL2VA or Ref2VA conditioning and latent inputs.",
            mode="needed",
            phase="conditioning",
            backend="standard",
            reads=_CONDITIONING_READS,
            writes=_CONDITIONING_WRITES,
        ),
        _entry(
            "continuity",
            title="Segment continuity",
            description="Apply the authored predecessor-tail guide when enabled.",
            mode="switch",
            phase="conditioning",
            backend="standard",
            reads=_CONTINUITY_READS,
            writes=_CONTINUITY_WRITES,
        ),
        _entry(
            "standard_sampling",
            title="Standard sampling",
            description="Run the current Standard H3 sampler subgraph.",
            mode="needed",
            phase="sampling",
            backend="standard",
            reads=(
                _read("model", "MODEL"),
                _read("conditioning", "CONDITIONING"),
                _read("latent", "LATENT"),
            ),
            writes=(_write("samples", "LATENT"),),
        ),
        _entry(
            "decode_video",
            title="Decode video frames",
            description="Decode the sampled video latent and crop continuity context.",
            mode="needed",
            phase="decode",
            backend="standard",
            reads=_DECODE_READS,
            writes=(_write("frames", "IMAGE"),),
        ),
        _entry(
            "audio_output",
            title="Select audio output",
            description="Generate, retain or mute audio and create the video object.",
            mode="needed",
            phase="decode",
            backend="standard",
            reads=_AUDIO_OUTPUT_READS,
            writes=(_write("video", "VIDEO"),),
        ),
        _entry(
            "save_take",
            title="Save take",
            description="Persist the unit's sole final take with SaveVideo.",
            mode="needed",
            phase="persist",
            backend="standard",
            reads=(_read("video", "VIDEO"),),
            writes=(_write("take_output", "TAKE"),),
        ),
    ),
)


V4_RAYLIGHT_SEGMENT_TEMPLATE = SegmentTemplate(
    id="h3_raylight_segment",
    revision=1,
    entries=(
        _entry(
            "shared_models",
            title="Shared CLIP and VAEs",
            description="Load and place the shared CLIP, video VAE and audio VAE.",
            mode="needed",
            phase="bootstrap",
            backend="raylight",
            writes=_SHARED_WRITES,
        ),
        _entry(
            "raylight_pool_intent",
            title="RayLight pool intent",
            description="Create the exact v4 RayLight initializer and pool inputs.",
            mode="needed",
            phase="bootstrap",
            backend="raylight",
            writes=(_write("ray_actors_init", "RAY_ACTORS_INIT"),),
        ),
        # Legacy v4 creates DirectorDeckRayLoraLoader before
        # DirectorDeckRayUNETLoader.
        _entry(
            "lora",
            title="RayLight LoRA",
            description="Build the optional fixed RayLight LoRA descriptor.",
            mode="switch",
            phase="model_load",
            backend="raylight",
            writes=(_write("ray_lora", "RAY_LORA"),),
        ),
        _entry(
            "raylight_model_load",
            title="RayLight model load",
            description="Load the H3 diffusion model into the current Ray actor pool.",
            mode="needed",
            phase="model_load",
            backend="raylight",
            reads=(
                _read("ray_actors_init", "RAY_ACTORS_INIT"),
                _read("ray_lora", "RAY_LORA", required=False),
            ),
            writes=(_write("model", "RAY_ACTORS"),),
        ),
        _entry(
            "raylight_sigma_shift",
            title="RayLight Sigma shift",
            description="Apply the current RayLight video and audio sigma shifts.",
            mode="needed",
            phase="model_prepare",
            backend="raylight",
            reads=(_read("model", "RAY_ACTORS"),),
            writes=(_write("model", "RAY_ACTORS", "replace"),),
        ),
        _entry(
            "family_conditioning",
            title="H3 family conditioning",
            description="Build the FL2VA or Ref2VA conditioning and latent inputs.",
            mode="needed",
            phase="conditioning",
            backend="raylight",
            reads=_CONDITIONING_READS,
            writes=_CONDITIONING_WRITES,
        ),
        _entry(
            "continuity",
            title="Segment continuity",
            description="Apply the authored predecessor-tail guide when enabled.",
            mode="switch",
            phase="conditioning",
            backend="raylight",
            reads=_CONTINUITY_READS,
            writes=_CONTINUITY_WRITES,
        ),
        _entry(
            "raylight_sampling",
            title="RayLight sampling",
            description="Run the current RayLight H3 sampler subgraph.",
            mode="needed",
            phase="sampling",
            backend="raylight",
            reads=(
                _read("model", "RAY_ACTORS"),
                _read("conditioning", "CONDITIONING"),
                _read("latent", "LATENT"),
            ),
            writes=(_write("samples", "LATENT"),),
        ),
        _entry(
            "decode_video",
            title="Decode video frames",
            description="Decode the sampled video latent and crop continuity context.",
            mode="needed",
            phase="decode",
            backend="raylight",
            reads=_DECODE_READS,
            writes=(_write("frames", "IMAGE"),),
        ),
        _entry(
            "audio_output",
            title="Select audio output",
            description="Generate, retain or mute audio and create the video object.",
            mode="needed",
            phase="decode",
            backend="raylight",
            reads=_AUDIO_OUTPUT_READS,
            writes=(_write("video", "VIDEO"),),
        ),
        _entry(
            "save_take",
            title="Save take",
            description="Persist the unit's sole final take with SaveVideo.",
            mode="needed",
            phase="persist",
            backend="raylight",
            reads=(_read("video", "VIDEO"),),
            writes=(_write("take_output", "TAKE"),),
        ),
    ),
)


V4_TEMPLATE_BUNDLE = TemplateBundle(
    version=4,
    segment_templates=SegmentTemplateSet(
        standard=V4_STANDARD_SEGMENT_TEMPLATE,
        raylight=V4_RAYLIGHT_SEGMENT_TEMPLATE,
    ),
    control_templates=ControlTemplateSet(
        ray_kill=ControlTemplate(id="raylight_kill_control", revision=1),
    ),
)


# ``V4_TEMPLATE_BUNDLE`` above is a historical, byte-frozen contract.  Current
# v5 projects compile against a separate bundle so adding an authorable feature
# can never mutate the compatibility fixture or the old graph ordering.
_EMPTY_PARAMS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {},
}
_LORA_PARAMS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["by_family"],
    "properties": {
        "by_family": {
            "type": "object",
            "additionalProperties": False,
            "required": ["fl2va", "ref2va"],
            "properties": {
                family: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["enabled", "filename", "strength"],
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "filename": {
                            "type": ["string", "null"],
                            "minLength": 1,
                            "maxLength": 1024,
                        },
                        "strength": {
                            "type": "number",
                            "minimum": -10,
                            "maximum": 10,
                        },
                    },
                }
                for family in _FAMILIES
            },
        }
    },
}
_LORA_DEFAULT_PARAMS = {
    "by_family": {
        family: {"enabled": False, "filename": None, "strength": 1.0}
        for family in _FAMILIES
    },
}


def _v5_entry(
    feature_id: str,
    *,
    title: str,
    description: str,
    version: int = 1,
    mode: str,
    phase: GraphPhase,
    backend: Backend,
    reads: Iterable[ResourceReadDeclaration] = (),
    writes: Iterable[ResourceWriteDeclaration] = (),
    scopes: tuple[str, ...] = ("segment",),
    visibility: str = "internal",
    params_schema: dict[str, object] | None = None,
    defaults: dict[str, object] | None = None,
    conflicts: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
) -> FeatureTemplateEntry:
    if visibility not in {"user", "internal"}:
        raise ValueError("v5 feature visibility must be user or internal")
    return FeatureTemplateEntry(
        id=feature_id,
        version=version,
        title=title,
        description=description,
        mode=mode,
        graph_phase=phase,
        reads=tuple(reads),
        writes=tuple(writes),
        params_schema=params_schema or _EMPTY_PARAMS_SCHEMA,
        defaults=defaults or {},
        cache_policy={"identity": "effective_active_selection_v1"},
        backends=(backend,),
        families=_FAMILIES,
        conflicts=conflicts,
        requires=requires,
        scopes=scopes,
        ui={"visibility": visibility},
    )


_ATTENTION_PARAMS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mode"],
    "properties": {
        "mode": {"type": "string", "enum": ["pytorch", "ck_int8"]},
    },
}
_RAYLIGHT_POOL_PARAMS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["attention"],
    "properties": {
        "attention": {
            "type": "string",
            "enum": ["ck_int8", "torch_flash"],
        },
    },
}


V5_STANDARD_SEGMENT_TEMPLATE = SegmentTemplate(
    id="h3_standard_segment",
    # Adding default-disabled descriptors does not change an active graph.
    # Bundle version 5 freezes the expanded catalog; the effective graph and
    # active identities decide cache/currentness changes.
    revision=1,
    entries=(
        _v5_entry(
            "shared_models",
            title="Shared CLIP and VAEs",
            description="Load and place the shared CLIP, video VAE and audio VAE.",
            mode="needed",
            phase="bootstrap",
            backend="standard",
            writes=_SHARED_WRITES,
        ),
        _v5_entry(
            "standard_model_load",
            title="Standard model load",
            description="Load the selected H3 diffusion model with UNETLoader.",
            mode="needed",
            phase="model_load",
            backend="standard",
            writes=(_write("model", "MODEL"),),
        ),
        _v5_entry(
            "standard_model_device",
            title="Standard model placement",
            description="Apply the authoritative Standard model device placement.",
            mode="needed",
            phase="model_prepare",
            backend="standard",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
        ),
        _v5_entry(
            "lora",
            title="LoRA",
            description="Apply the selected fail-closed LoRA adapter.",
            mode="switch",
            phase="model_prepare",
            backend="standard",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
            scopes=("project",),
            visibility="user",
            params_schema=_LORA_PARAMS_SCHEMA,
            defaults=_LORA_DEFAULT_PARAMS,
        ),
        _v5_entry(
            "standard_sigma_shift",
            title="Standard Sigma shift",
            description="Apply the current H3 video and audio sigma shifts.",
            mode="needed",
            phase="model_prepare",
            backend="standard",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
        ),
        _v5_entry(
            "attention_backend_override",
            title="Attention backend override",
            description="Explicitly select the strict Standard H3 attention backend.",
            mode="switch",
            phase="model_patch",
            backend="standard",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
            scopes=("project", "segment"),
            visibility="user",
            params_schema=_ATTENTION_PARAMS_SCHEMA,
            defaults={"mode": "pytorch"},
            conflicts=("h3_low_vram_attention",),
        ),
        _v5_entry(
            "h3_low_vram_attention",
            title="H3 low-VRAM attention",
            description="Apply the strict all-block H3 low-VRAM attention patch.",
            mode="switch",
            phase="model_patch",
            backend="standard",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
            scopes=("project", "segment"),
            visibility="user",
            conflicts=("attention_backend_override",),
        ),
        _v5_entry(
            "family_conditioning",
            title="H3 family conditioning",
            description="Build the FL2VA or Ref2VA conditioning and latent inputs.",
            mode="needed",
            phase="conditioning",
            backend="standard",
            reads=_CONDITIONING_READS,
            writes=_CONDITIONING_WRITES,
        ),
        _v5_entry(
            "continuity",
            title="Segment continuity",
            description="Apply the authored predecessor-tail guide when enabled.",
            mode="switch",
            phase="conditioning",
            backend="standard",
            reads=_CONTINUITY_READS,
            writes=_CONTINUITY_WRITES,
        ),
        _v5_entry(
            "standard_sampling",
            title="Standard sampling",
            description="Run the current Standard H3 sampler subgraph.",
            mode="needed",
            phase="sampling",
            backend="standard",
            reads=(
                _read("model", "MODEL"),
                _read("conditioning", "CONDITIONING"),
                _read("latent", "LATENT"),
            ),
            writes=(_write("samples", "LATENT"),),
        ),
        _v5_entry(
            "decode_video",
            title="Decode video frames",
            description="Decode the sampled video latent and crop continuity context.",
            mode="needed",
            phase="decode",
            backend="standard",
            reads=_DECODE_READS,
            writes=(_write("frames", "IMAGE"),),
        ),
        _v5_entry(
            "audio_output",
            title="Select audio output",
            description="Generate, retain or mute audio and create the video object.",
            mode="needed",
            phase="decode",
            backend="standard",
            reads=_AUDIO_OUTPUT_READS,
            writes=(_write("video", "VIDEO"),),
        ),
        _v5_entry(
            "save_take",
            title="Save take",
            description="Persist the unit's sole final take with SaveVideo.",
            mode="needed",
            phase="persist",
            backend="standard",
            reads=(_read("video", "VIDEO"),),
            writes=(_write("take_output", "TAKE"),),
        ),
    ),
)


V5_RAYLIGHT_SEGMENT_TEMPLATE = SegmentTemplate(
    id="h3_raylight_segment",
    revision=1,
    entries=(
        _v5_entry(
            "shared_models",
            title="Shared CLIP and VAEs",
            description="Load and place the shared CLIP, video VAE and audio VAE.",
            mode="needed",
            phase="bootstrap",
            backend="raylight",
            writes=_SHARED_WRITES,
        ),
        _v5_entry(
            "raylight_pool_intent",
            title="RayLight pool intent",
            description="Create the exact RayLight initializer, pool and attention inputs.",
            version=2,
            mode="needed",
            phase="bootstrap",
            backend="raylight",
            writes=(_write("ray_actors_init", "RAY_ACTORS_INIT"),),
            scopes=("project", "segment"),
            visibility="user",
            params_schema=_RAYLIGHT_POOL_PARAMS_SCHEMA,
            defaults={"attention": "ck_int8"},
        ),
        # DirectorDeckRayLoraLoader creates a descriptor consumed by
        # DirectorDeckRayUNETLoader; this
        # dependency keeps the existing audited order until the external pack
        # exposes a model-patch contract that can run after model load.
        _v5_entry(
            "lora",
            title="LoRA",
            description="Apply the selected fail-closed LoRA adapter.",
            mode="switch",
            phase="model_load",
            backend="raylight",
            writes=(_write("ray_lora", "RAY_LORA"),),
            scopes=("project",),
            visibility="user",
            params_schema=_LORA_PARAMS_SCHEMA,
            defaults=_LORA_DEFAULT_PARAMS,
        ),
        _v5_entry(
            "raylight_model_load",
            title="RayLight model load",
            description="Load the H3 model into the current Ray actor pool.",
            mode="needed",
            phase="model_load",
            backend="raylight",
            reads=(
                _read("ray_actors_init", "RAY_ACTORS_INIT"),
                _read("ray_lora", "RAY_LORA", required=False),
            ),
            writes=(_write("model", "RAY_ACTORS"),),
        ),
        _v5_entry(
            "raylight_sigma_shift",
            title="RayLight Sigma shift",
            description="Apply the current RayLight video and audio sigma shifts.",
            mode="needed",
            phase="model_prepare",
            backend="raylight",
            reads=(_read("model", "RAY_ACTORS"),),
            writes=(_write("model", "RAY_ACTORS", "replace"),),
        ),
        _v5_entry(
            "family_conditioning",
            title="H3 family conditioning",
            description="Build the FL2VA or Ref2VA conditioning and latent inputs.",
            mode="needed",
            phase="conditioning",
            backend="raylight",
            reads=_CONDITIONING_READS,
            writes=_CONDITIONING_WRITES,
        ),
        _v5_entry(
            "continuity",
            title="Segment continuity",
            description="Apply the authored predecessor-tail guide when enabled.",
            mode="switch",
            phase="conditioning",
            backend="raylight",
            reads=_CONTINUITY_READS,
            writes=_CONTINUITY_WRITES,
        ),
        _v5_entry(
            "raylight_sampling",
            title="RayLight sampling",
            description="Run the current RayLight H3 sampler subgraph.",
            mode="needed",
            phase="sampling",
            backend="raylight",
            reads=(
                _read("model", "RAY_ACTORS"),
                _read("conditioning", "CONDITIONING"),
                _read("latent", "LATENT"),
            ),
            writes=(_write("samples", "LATENT"),),
        ),
        _v5_entry(
            "decode_video",
            title="Decode video frames",
            description="Decode the sampled video latent and crop continuity context.",
            mode="needed",
            phase="decode",
            backend="raylight",
            reads=_DECODE_READS,
            writes=(_write("frames", "IMAGE"),),
        ),
        _v5_entry(
            "audio_output",
            title="Select audio output",
            description="Generate, retain or mute audio and create the video object.",
            mode="needed",
            phase="decode",
            backend="raylight",
            reads=_AUDIO_OUTPUT_READS,
            writes=(_write("video", "VIDEO"),),
        ),
        _v5_entry(
            "save_take",
            title="Save take",
            description="Persist the unit's sole final take with SaveVideo.",
            mode="needed",
            phase="persist",
            backend="raylight",
            reads=(_read("video", "VIDEO"),),
            writes=(_write("take_output", "TAKE"),),
        ),
    ),
)


V5_TEMPLATE_BUNDLE = TemplateBundle(
    version=5,
    segment_templates=SegmentTemplateSet(
        standard=V5_STANDARD_SEGMENT_TEMPLATE,
        raylight=V5_RAYLIGHT_SEGMENT_TEMPLATE,
    ),
    control_templates=ControlTemplateSet(
        ray_kill=ControlTemplate(id="raylight_kill_control", revision=1),
    ),
)

CURRENT_TEMPLATE_BUNDLE = V5_TEMPLATE_BUNDLE


__all__ = [
    "CURRENT_TEMPLATE_BUNDLE",
    "V4_RAYLIGHT_SEGMENT_TEMPLATE",
    "V4_STANDARD_SEGMENT_TEMPLATE",
    "V4_TEMPLATE_BUNDLE",
    "V5_RAYLIGHT_SEGMENT_TEMPLATE",
    "V5_STANDARD_SEGMENT_TEMPLATE",
    "V5_TEMPLATE_BUNDLE",
]
