from __future__ import annotations

from dataclasses import dataclass

from ._emitter import NativeEdge


@dataclass(frozen=True, slots=True)
class SharedModelOutputs:
    clip: NativeEdge
    video_vae: NativeEdge
    audio_vae: NativeEdge | None

    def as_legacy_mapping(self) -> dict[str, NativeEdge | None]:
        return {
            "clip": self.clip,
            "video_vae": self.video_vae,
            "audio_vae": self.audio_vae,
        }


@dataclass(frozen=True, slots=True)
class ConditioningOutputs:
    conditioning: NativeEdge
    latent: NativeEdge
    source_audio: NativeEdge | None


@dataclass(frozen=True, slots=True)
class ContinuityOutputs:
    conditioning: NativeEdge
    load_video_node_id: str


@dataclass(frozen=True, slots=True)
class AudioOutput:
    video: NativeEdge
    audio: NativeEdge | None
