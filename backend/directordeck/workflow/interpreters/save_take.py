from __future__ import annotations

import re

from ...schemas import UnifiedTimelineSegment
from ._emitter import NativeEdge, NativeNodeEmitter


INTERPRETER_ID = "save_take"
INTERPRETER_VERSION = 1


def emit_save_take(
    emitter: NativeNodeEmitter,
    *,
    video: NativeEdge,
    job_id: str,
    segment: UnifiedTimelineSegment,
) -> str:
    safe_segment = re.sub(r"[^A-Za-z0-9_-]+", "_", segment.id)[:64]
    return emitter.add(
        "SaveVideo",
        video=video,
        filename_prefix=f"video/DirectorDeck_timeline_{job_id[:8]}_{safe_segment}",
        format="auto",
        codec="auto",
    )
