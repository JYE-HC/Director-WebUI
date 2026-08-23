from __future__ import annotations

from ...native_templates import NativeTemplateError
from ...schemas import (
    UnifiedRef2VASegment,
    UnifiedTimelineDraft,
    UnifiedTimelineSegment,
)
from ._emitter import NativeEdge, NativeNodeEmitter, edge
from ._types import AudioOutput, SharedModelOutputs


INTERPRETER_ID = "audio_output"
INTERPRETER_VERSION = 1


def emit_audio_output(
    emitter: NativeNodeEmitter,
    *,
    images: NativeEdge,
    samples: NativeEdge,
    source_audio: NativeEdge | None,
    draft: UnifiedTimelineDraft,
    shared: SharedModelOutputs | None = None,
    audio_vae: NativeEdge | None = None,
    segment: UnifiedTimelineSegment,
    visible_frames: int,
    continuity_prefix_frames: int,
) -> AudioOutput:
    if shared is not None:
        audio_vae = shared.audio_vae
    video_inputs = {
        "images": images,
        "fps": draft.render.fps,
        "bit_depth": 8,
    }
    selected_audio: NativeEdge | None = None
    if segment.audio_mode == "generate":
        if audio_vae is None:
            raise TypeError("generated audio requires the exact audio VAE edge")
        selected_audio = edge(
            emitter.add(
                "VAEDecodeAudio",
                samples=samples,
                vae=audio_vae,
            )
        )
        if continuity_prefix_frames:
            selected_audio = edge(
                emitter.add(
                    "TrimAudioDuration",
                    audio=selected_audio,
                    start_index=continuity_prefix_frames / draft.render.fps,
                    duration=visible_frames / draft.render.fps,
                )
            )
        video_inputs["audio"] = selected_audio
    elif segment.audio_mode == "source":
        if source_audio is None:
            raise NativeTemplateError(
                "source audio is available only for v2v/rv2v segments"
            )
        if (
            not isinstance(segment, UnifiedRef2VASegment)
            or segment.source_video is None
        ):
            raise NativeTemplateError(
                "source audio is available only for v2v/rv2v segments"
            )
        assert segment.source_video.metadata is not None
        if not segment.source_video.metadata.has_audio:
            raise NativeTemplateError(
                f"segment '{segment.id}' cannot use audio_mode='source': "
                "the server-probed source video has no audio stream"
            )
        selected_audio = source_audio
        video_inputs["audio"] = selected_audio
    video = edge(emitter.add("CreateVideo", **video_inputs))
    return AudioOutput(video=video, audio=selected_audio)
