from __future__ import annotations

from ...native_templates import NativeTemplateError
from ._emitter import NativeEdge, NativeNodeEmitter, edge


INTERPRETER_ID = "decode_video"
INTERPRETER_VERSION = 1


def emit_decode_video(
    emitter: NativeNodeEmitter,
    *,
    samples: NativeEdge,
    video_vae: NativeEdge,
    visible_frames: int,
    continuity_prefix_frames: int,
) -> NativeEdge:
    if visible_frames <= 0 or continuity_prefix_frames < 0:
        raise NativeTemplateError("decoded frame counts must be non-negative")
    images = edge(emitter.add("VAEDecode", samples=samples, vae=video_vae))
    if continuity_prefix_frames:
        images = edge(
            emitter.add(
                "ImageFromBatch",
                image=images,
                batch_index=continuity_prefix_frames,
                length=visible_frames,
            )
        )
    return images
