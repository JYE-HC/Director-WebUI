from __future__ import annotations

"""Ordered Bundle-6 feature uses; semantic definitions remain graph-free."""

from typing import Literal

from pydantic import model_validator

from .contracts import (
    Backend,
    ContractModel,
    GRAPH_PHASE_ORDER,
    GraphPhase,
    Identifier,
    PositiveVersion,
    ResourceReadDeclaration,
    ResourceWriteDeclaration,
)
from .feature_definitions import BUNDLE6_FEATURE_DEFINITIONS_BY_ID


class FeatureDependency(ContractModel):
    feature_id: Identifier
    feature_version: PositiveVersion = 1
    required: bool = True


class FeatureUse(ContractModel):
    feature_id: Identifier
    feature_version: PositiveVersion
    graph_phase: GraphPhase
    reads: tuple[ResourceReadDeclaration, ...] = ()
    writes: tuple[ResourceWriteDeclaration, ...] = ()
    dependencies: tuple[FeatureDependency, ...] = ()

    @model_validator(mode="after")
    def validate_declarations(self) -> "FeatureUse":
        for label, values in (
            ("reads", tuple(item.name for item in self.reads)),
            ("writes", tuple(item.name for item in self.writes)),
            ("dependencies", tuple(item.feature_id for item in self.dependencies)),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"feature use {label} must be unique")
        if any(item.feature_id == self.feature_id for item in self.dependencies):
            raise ValueError("feature use cannot depend on itself")
        return self


class SegmentTemplateV6(ContractModel):
    id: Literal["h3_standard_segment", "h3_raylight_segment"]
    revision: PositiveVersion
    entries: tuple[FeatureUse, ...]

    @property
    def backend(self) -> Backend:
        return "standard" if self.id == "h3_standard_segment" else "raylight"

    @model_validator(mode="after")
    def validate_entries(self) -> "SegmentTemplateV6":
        if not self.entries:
            raise ValueError("segment template must contain feature uses")
        ids = tuple(entry.feature_id for entry in self.entries)
        if len(ids) != len(set(ids)):
            raise ValueError("segment template feature uses must be unique")
        phase_index = {phase: index for index, phase in enumerate(GRAPH_PHASE_ORDER)}
        positions: dict[str, int] = {}
        resources: dict[str, tuple[str, bool]] = {}
        previous_phase = -1
        for index, entry in enumerate(self.entries):
            definition = BUNDLE6_FEATURE_DEFINITIONS_BY_ID.get(entry.feature_id)
            if definition is None or definition.version != entry.feature_version:
                raise ValueError("feature use has no exact Bundle 6 definition")
            if self.backend not in definition.backends:
                raise ValueError("feature use does not support template backend")
            current_phase = phase_index[entry.graph_phase]
            if current_phase < previous_phase:
                raise ValueError("feature graph phases must be monotonic")
            previous_phase = current_phase
            for dependency in entry.dependencies:
                position = positions.get(dependency.feature_id)
                if position is None or position >= index:
                    raise ValueError("feature dependency must be an earlier use")
                depended = self.entries[position]
                if depended.feature_version != dependency.feature_version:
                    raise ValueError("feature dependency version does not match")
            for read in entry.reads:
                known = resources.get(read.name)
                if known is None:
                    if read.required:
                        raise ValueError(f"required resource is undefined: {read.name}")
                    continue
                if known[0] != read.type or (read.required and not known[1]):
                    raise ValueError(f"resource read is incompatible: {read.name}")
            for write in entry.writes:
                known = resources.get(write.name)
                if write.operation == "define":
                    if known is not None:
                        raise ValueError(f"resource is already defined: {write.name}")
                    resources[write.name] = (write.type, write.required)
                else:
                    if known is None or known[0] != write.type:
                        raise ValueError(f"resource replace is incompatible: {write.name}")
                    resources[write.name] = (known[0], known[1])
            positions[entry.feature_id] = index
        return self


class SegmentTemplateSetV6(ContractModel):
    standard: SegmentTemplateV6
    raylight: SegmentTemplateV6


class TemplateBundleV6(ContractModel):
    version: Literal[6] = 6
    segment_templates: SegmentTemplateSetV6


def _read(name: str, port_type: str, *, required: bool = True) -> ResourceReadDeclaration:
    return ResourceReadDeclaration(name=name, type=port_type, required=required)


def _write(
    name: str,
    port_type: str,
    operation: Literal["define", "replace"],
    *,
    required: bool = True,
) -> ResourceWriteDeclaration:
    return ResourceWriteDeclaration(
        name=name,
        type=port_type,
        operation=operation,
        required=required,
    )


def _use(
    feature_id: str,
    phase: GraphPhase,
    *,
    reads: tuple[ResourceReadDeclaration, ...] = (),
    writes: tuple[ResourceWriteDeclaration, ...] = (),
    dependencies: tuple[FeatureDependency, ...] = (),
) -> FeatureUse:
    return FeatureUse(
        feature_id=feature_id,
        feature_version=1,
        graph_phase=phase,
        reads=reads,
        writes=writes,
        dependencies=dependencies,
    )


_SHARED = (
    _use(
        "auxiliary_models",
        "bootstrap",
        writes=(
            _write("clip", "CLIP", "define"),
            _write("video_vae", "VAE", "define"),
            _write("audio_vae", "VAE", "define", required=False),
        ),
    ),
)
_CONDITIONING = (
    _use(
        "multimodal_conditioning",
        "conditioning",
        reads=(
            _read("clip", "CLIP"),
            _read("video_vae", "VAE"),
            _read("audio_vae", "VAE", required=False),
        ),
        writes=(
            _write("conditioning", "CONDITIONING", "define"),
            _write("latent", "LATENT", "define"),
            _write("source_audio", "AUDIO", "define", required=False),
        ),
    ),
    _use(
        "continuity",
        "conditioning",
        reads=(
            _read("conditioning", "CONDITIONING"),
            _read("latent", "LATENT"),
            _read("video_vae", "VAE"),
            _read("audio_vae", "VAE", required=False),
        ),
        writes=(_write("conditioning", "CONDITIONING", "replace", required=False),),
    ),
)
_OUTPUT = (
    _use(
        "video_decode",
        "decode",
        reads=(_read("samples", "LATENT"), _read("video_vae", "VAE")),
        writes=(_write("frames", "IMAGE", "define"),),
    ),
    _use(
        "audio_output",
        "decode",
        reads=(
            _read("frames", "IMAGE"),
            _read("samples", "LATENT"),
            _read("source_audio", "AUDIO", required=False),
            _read("audio_vae", "VAE", required=False),
        ),
        writes=(_write("video", "VIDEO", "define"),),
    ),
    _use(
        "save_take",
        "persist",
        reads=(_read("video", "VIDEO"),),
        writes=(_write("take_output", "TAKE", "define"),),
    ),
)


V6_STANDARD_SEGMENT_TEMPLATE = SegmentTemplateV6(
    id="h3_standard_segment",
    revision=6,
    entries=(
        *_SHARED,
        _use("diffusion_model", "model_load", writes=(_write("model", "MODEL", "define"),)),
        _use(
            "execution_strategy",
            "model_prepare",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
        ),
        _use(
            "lora",
            "model_prepare",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace", required=False),),
        ),
        _use(
            "comfy_kitchen_attention",
            "model_patch",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace", required=False),),
        ),
        _use(
            "sigma_schedule",
            "model_patch",
            reads=(_read("model", "MODEL"),),
            writes=(_write("model", "MODEL", "replace"),),
        ),
        *_CONDITIONING,
        _use(
            "sampling_pipeline",
            "sampling",
            reads=(
                _read("model", "MODEL"),
                _read("conditioning", "CONDITIONING"),
                _read("latent", "LATENT"),
            ),
            writes=(_write("samples", "LATENT", "define"),),
        ),
        *_OUTPUT,
    ),
)

V6_RAYLIGHT_SEGMENT_TEMPLATE = SegmentTemplateV6(
    id="h3_raylight_segment",
    revision=6,
    entries=(
        *_SHARED,
        _use("comfy_kitchen_attention", "bootstrap"),
        _use(
            "execution_strategy",
            "bootstrap",
            writes=(_write("ray_actors_init", "RAY_ACTORS_INIT", "define"),),
            dependencies=(
                FeatureDependency(
                    feature_id="comfy_kitchen_attention",
                    required=False,
                ),
            ),
        ),
        _use("lora", "model_load", writes=(_write("ray_lora", "RAY_LORA", "define", required=False),)),
        _use(
            "diffusion_model",
            "model_load",
            reads=(
                _read("ray_actors_init", "RAY_ACTORS_INIT"),
                _read("ray_lora", "RAY_LORA", required=False),
            ),
            writes=(_write("model", "RAY_ACTORS", "define"),),
        ),
        _use(
            "sigma_schedule",
            "model_prepare",
            reads=(_read("model", "RAY_ACTORS"),),
            writes=(_write("model", "RAY_ACTORS", "replace"),),
        ),
        *_CONDITIONING,
        _use(
            "sampling_pipeline",
            "sampling",
            reads=(
                _read("model", "RAY_ACTORS"),
                _read("conditioning", "CONDITIONING"),
                _read("latent", "LATENT"),
            ),
            writes=(_write("samples", "LATENT", "define"),),
        ),
        *_OUTPUT,
    ),
)

V6_TEMPLATE_BUNDLE = TemplateBundleV6(
    segment_templates=SegmentTemplateSetV6(
        standard=V6_STANDARD_SEGMENT_TEMPLATE,
        raylight=V6_RAYLIGHT_SEGMENT_TEMPLATE,
    )
)


__all__ = [
    "FeatureDependency",
    "FeatureUse",
    "SegmentTemplateV6",
    "TemplateBundleV6",
    "V6_RAYLIGHT_SEGMENT_TEMPLATE",
    "V6_STANDARD_SEGMENT_TEMPLATE",
    "V6_TEMPLATE_BUNDLE",
]
