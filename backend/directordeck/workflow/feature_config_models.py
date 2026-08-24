from __future__ import annotations

"""Small typed value objects produced by Bundle-6 config resolvers."""

from typing import Literal

from pydantic import Field

from ..schemas import StrictModel
from .contracts import Backend, ModelFamily


class EmptyFeatureParams(StrictModel):
    pass


class ComfyKitchenAttentionParamsV1(StrictModel):
    pass


class AuxiliaryModelsConfigV1(StrictModel):
    clip_filename: str
    clip_device: str
    video_vae_filename: str
    video_vae_device: str
    audio_vae_filename: str | None = None
    audio_vae_device: str | None = None


class DiffusionModelConfigV1(StrictModel):
    family: ModelFamily
    backend: Backend
    filename: str
    device: str


class RayExecutionConfigV1(StrictModel):
    gpu_select: tuple[int, ...]
    ulysses_degree: int
    ring_degree: int
    cfg_degree: int
    dp_degree: int
    fsdp: bool
    cpu_offload: bool
    clear_vram_after_sampling: bool


class ExecutionStrategyConfigV1(StrictModel):
    backend: Backend
    device: str
    raylight: RayExecutionConfigV1 | None = None


class LoraConfigV1(StrictModel):
    family: ModelFamily
    backend: Backend
    model_filename: str
    lora_filename: str
    strength: float = Field(ge=-10, le=10, allow_inf_nan=False)
    adapter_id: str
    class_type: str
    input_contract: str
    source: Literal["user_override", "factory_default", "backend_fixed"]
    options: dict[str, bool] = Field(default_factory=dict)


class SigmaScheduleConfigV1(StrictModel):
    shift_video: float
    shift_audio: float


class ConditioningConfigV1(StrictModel):
    segment_id: str
    family: ModelFamily
    recipe: Literal["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"]
    sample_frames: int


class ContinuityConfigV1(StrictModel):
    predecessor_segment_id: str
    source: Literal["same_run", "historical_take"]
    overlap_frames: int
    historical_take_id: str | None = None


class SamplingPipelineConfigV1(StrictModel):
    steps: int
    seed: int
    random_seed: bool
    sampler: str
    scheduler: str


class VideoDecodeConfigV1(StrictModel):
    visible_frames: int
    continuity_prefix_frames: int


class AudioOutputConfigV1(StrictModel):
    mode: Literal["generate", "source", "mute"]
    visible_frames: int
    continuity_prefix_frames: int


class SaveTakeConfigV1(StrictModel):
    job_id: str
    segment_id: str


__all__ = [
    "AudioOutputConfigV1",
    "AuxiliaryModelsConfigV1",
    "ComfyKitchenAttentionParamsV1",
    "ConditioningConfigV1",
    "ContinuityConfigV1",
    "DiffusionModelConfigV1",
    "EmptyFeatureParams",
    "ExecutionStrategyConfigV1",
    "LoraConfigV1",
    "RayExecutionConfigV1",
    "SamplingPipelineConfigV1",
    "SaveTakeConfigV1",
    "SigmaScheduleConfigV1",
    "VideoDecodeConfigV1",
]
