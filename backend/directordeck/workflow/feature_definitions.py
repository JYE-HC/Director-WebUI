from __future__ import annotations

"""Stable Bundle-6 workflow semantics, deliberately free of graph and UI data."""

from typing import Literal

from pydantic import Field, model_validator

from .contracts import (
    Backend,
    ContractModel,
    Identifier,
    ModelFamily,
    PositiveVersion,
)


class FeatureDefinition(ContractModel):
    id: Identifier
    version: PositiveVersion
    title: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=4_096)
    activation: Literal["needed", "switch", "contextual"]
    authoring: Literal["none", "project", "segment", "project_segment"]
    backends: tuple[Backend, ...]
    families: tuple[ModelFamily, ...]

    @model_validator(mode="after")
    def validate_routes(self) -> "FeatureDefinition":
        if not self.backends or len(set(self.backends)) != len(self.backends):
            raise ValueError("feature backends must be non-empty and unique")
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("feature families must be non-empty and unique")
        return self


_ALL_BACKENDS: tuple[Backend, ...] = ("standard", "raylight")
_ALL_FAMILIES: tuple[ModelFamily, ...] = ("fl2va", "ref2va")


def _definition(
    feature_id: str,
    title: str,
    description: str,
    *,
    activation: Literal["needed", "switch", "contextual"] = "needed",
    authoring: Literal["none", "project", "segment", "project_segment"] = "none",
) -> FeatureDefinition:
    return FeatureDefinition(
        id=feature_id,
        version=1,
        title=title,
        description=description,
        activation=activation,
        authoring=authoring,
        backends=_ALL_BACKENDS,
        families=_ALL_FAMILIES,
    )


BUNDLE6_FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    _definition(
        "auxiliary_models",
        "Auxiliary models",
        "Prepare CLIP and the video/audio VAE resources.",
        authoring="project",
    ),
    _definition(
        "diffusion_model",
        "Diffusion model",
        "Load the selected family diffusion model for the active backend.",
        authoring="project",
    ),
    _definition(
        "execution_strategy",
        "Execution strategy",
        "Prepare Standard device placement or the bundled RayLight pool.",
    ),
    _definition(
        "lora",
        "LoRA",
        "Apply the configured LoRA through the product-selected loader adapter.",
        activation="switch",
        authoring="project",
    ),
    _definition(
        "sigma_schedule",
        "Sigma schedule",
        "Apply the family video and audio sigma shifts.",
        authoring="project",
    ),
    _definition(
        "comfy_kitchen_attention",
        "Comfy Kitchen Attention",
        "Request ComfyUI CK attention on the currently selected runtime path.",
        activation="switch",
        authoring="project",
    ),
    _definition(
        "multimodal_conditioning",
        "Multimodal conditioning",
        "Build family-specific text, image, video and audio conditioning.",
        authoring="segment",
    ),
    _definition(
        "continuity",
        "Continuity",
        "Condition the segment from its authored predecessor when applicable.",
        activation="contextual",
        authoring="segment",
    ),
    _definition(
        "sampling_pipeline",
        "Sampling pipeline",
        "Run the selected family sampler and scheduler as one semantic step.",
        authoring="project",
    ),
    _definition(
        "video_decode",
        "Video decode",
        "Decode sampled latents and remove any continuity prefix.",
        authoring="project",
    ),
    _definition(
        "audio_output",
        "Audio output",
        "Apply generate, source or mute audio policy and create the video.",
        authoring="segment",
    ),
    _definition(
        "save_take",
        "Save take",
        "Persist the unit's single final take.",
    ),
)

BUNDLE6_FEATURE_DEFINITIONS_BY_ID = {
    definition.id: definition for definition in BUNDLE6_FEATURE_DEFINITIONS
}
if len(BUNDLE6_FEATURE_DEFINITIONS_BY_ID) != len(BUNDLE6_FEATURE_DEFINITIONS):
    raise AssertionError("Bundle 6 feature definitions must be unique")


__all__ = [
    "BUNDLE6_FEATURE_DEFINITIONS",
    "BUNDLE6_FEATURE_DEFINITIONS_BY_ID",
    "FeatureDefinition",
]
