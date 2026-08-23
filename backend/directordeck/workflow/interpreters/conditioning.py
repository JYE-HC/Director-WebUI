from __future__ import annotations

from ...native_templates import NativeTemplateError, _conditioning
from ...schemas import (
    UnifiedFL2VASegment,
    UnifiedRef2VASegment,
    UnifiedTimelineDraft,
    UnifiedTimelineSegment,
)
from ._emitter import NativeNodeEmitter
from ._types import ConditioningOutputs, SharedModelOutputs


INTERPRETER_ID = "family_conditioning"
INTERPRETER_VERSION = 1


def _emit_conditioning(
    emitter: NativeNodeEmitter,
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    shared: SharedModelOutputs,
    *,
    frames: int,
) -> ConditioningOutputs:
    conditioning, latent, source_audio = _conditioning(
        emitter,  # type: ignore[arg-type]
        segment,
        draft,
        shared.as_legacy_mapping(),
        frames=frames,
    )
    return ConditioningOutputs(conditioning, latent, source_audio)


def emit_fl2va_conditioning(
    emitter: NativeNodeEmitter,
    segment: UnifiedFL2VASegment,
    draft: UnifiedTimelineDraft,
    shared: SharedModelOutputs,
    *,
    frames: int,
) -> ConditioningOutputs:
    if not isinstance(segment, UnifiedFL2VASegment):
        raise NativeTemplateError("FL2VA conditioning requires an FL2VA segment")
    return _emit_conditioning(emitter, segment, draft, shared, frames=frames)


def emit_ref2va_conditioning(
    emitter: NativeNodeEmitter,
    segment: UnifiedRef2VASegment,
    draft: UnifiedTimelineDraft,
    shared: SharedModelOutputs,
    *,
    frames: int,
) -> ConditioningOutputs:
    if not isinstance(segment, UnifiedRef2VASegment):
        raise NativeTemplateError("Ref2VA conditioning requires a Ref2VA segment")
    return _emit_conditioning(emitter, segment, draft, shared, frames=frames)


def emit_family_conditioning(
    emitter: NativeNodeEmitter,
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    shared: SharedModelOutputs,
    *,
    frames: int,
) -> ConditioningOutputs:
    if isinstance(segment, UnifiedFL2VASegment):
        return emit_fl2va_conditioning(
            emitter, segment, draft, shared, frames=frames
        )
    if isinstance(segment, UnifiedRef2VASegment):
        return emit_ref2va_conditioning(
            emitter, segment, draft, shared, frames=frames
        )
    raise NativeTemplateError(f"unsupported segment mode: {segment.mode}")
