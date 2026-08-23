from __future__ import annotations

from ...native_templates import _add_continuity_guides
from ...schemas import UnifiedTimelineDraft, UnifiedTimelineSegment
from ._emitter import NativeEdge, NativeNodeEmitter
from ._types import ContinuityOutputs, SharedModelOutputs


INTERPRETER_ID = "continuity"
INTERPRETER_VERSION = 1


def emit_continuity(
    emitter: NativeNodeEmitter,
    *,
    conditioning: NativeEdge,
    latent: NativeEdge,
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    shared: SharedModelOutputs | None = None,
    video_vae: NativeEdge | None = None,
    audio_vae: NativeEdge | None = None,
    visible_frames: int,
    overlap_frames: int,
) -> ContinuityOutputs:
    if shared is not None:
        video_vae = shared.video_vae
        audio_vae = shared.audio_vae
    if video_vae is None:
        raise TypeError("continuity requires the exact video VAE edge")
    if segment.audio_mode == "generate" and audio_vae is None:
        raise TypeError("generated-audio continuity requires the exact audio VAE edge")
    updated, load_video_node_id = _add_continuity_guides(
        emitter,  # type: ignore[arg-type]
        conditioning=conditioning,
        latent=latent,
        segment=segment,
        draft=draft,
        shared={"video_vae": video_vae, "audio_vae": audio_vae},
        visible_frames=visible_frames,
        overlap_frames=overlap_frames,
    )
    return ContinuityOutputs(updated, load_video_node_id)
