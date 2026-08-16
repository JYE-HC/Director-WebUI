from __future__ import annotations

import json
import hashlib
import re
from typing import Any, Literal

from .schemas import (
    AssetReference,
    FL2VDraft,
    GenerationMode,
    I2VDraft,
    ModeDraft,
    R2VDraft,
    RV2VDraft,
    RuntimeSettings,
    T2VDraft,
    UnifiedFL2VASegment,
    UnifiedRef2VASegment,
    UnifiedTimelineDraft,
    UnifiedTimelineSegment,
    V2VDraft,
    mode_draft_to_timeline,
    timeline_segment_recipe,
)


MODE_LABELS: dict[GenerationMode, str] = {
    "t2v": "t2v — 文生视频(Text to Video)",
    "i2v": "i2v — 图生视频(Image to Video)",
    "fl2v": "fl2v — 首尾帧生视频(First-Last Frame)",
    "r2v": "r2v — 参考主体生视频(Reference to Video)",
    "v2v": "v2v — 视频转视频(Video to Video)",
    "rv2v": "rv2v — 参考素材改视频(Reference Video Edit)",
}


class DraftNotRunnable(ValueError):
    pass


MAX_SEGMENT_FRAMES = 512
MAX_TIMELINE_FRAMES = 100_000
_REFERENCE_TAG = re.compile(
    r"<\s*(Picture|Audio|Video)\s+([0-9]+)\s*>", re.IGNORECASE
)


def timeline_segment_take_fingerprint(
    draft: UnifiedTimelineDraft,
    segment: UnifiedTimelineSegment,
) -> str:
    """Fingerprint only the saved video's structural continuation contract.

    A historical take is an already-rendered media artifact, not a cache entry
    for the current prompt.  Its prompt, recipe/model family, anchors,
    references, source trims, models, LoRA and sampling settings therefore do
    not participate.  Stable ``segment.id`` and ComfyUI origin remain separate
    lookup keys; this digest only proves that the output canvas and actual H3
    visible-frame geometry still match the current predecessor slot.
    """

    payload: dict[str, Any] = {
        "schema": "director-segment-take-geometry-v1",
        "width": draft.render.width,
        "height": draft.render.height,
        "fps": draft.render.fps,
        "visible_frame_count": align_h3_frames(
            segment.duration_seconds, draft.render.fps
        ),
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"take-geometry-v1:sha256:{hashlib.sha256(canonical).hexdigest()}"


def align_h3_frames(duration_seconds: float, fps: float) -> int:
    raw = max(5, int(round(duration_seconds * fps)))
    aligned = raw + ((5 - raw % 17) % 17)
    if aligned > MAX_SEGMENT_FRAMES:
        raise DraftNotRunnable(
            f"one shot compiles to {aligned} frames; MiniMax H3 Director allows at most "
            f"{MAX_SEGMENT_FRAMES} frames per shot"
        )
    return aligned


def _validate_total_frames(total_frames: int) -> None:
    if total_frames > MAX_TIMELINE_FRAMES:
        raise DraftNotRunnable(
            f"timeline compiles to {total_frames} frames; Director allows at most "
            f"{MAX_TIMELINE_FRAMES}"
        )


def _validate_reference_tags(draft: ModeDraft, shots: list[Any]) -> None:
    """Reject stale explicit tags before Director silently degrades them.

    FL2VA keyframes are presented to the text encoder as one-based pictures in
    first/last input order. Director stores Ref2VA reference slots zero-based
    but exposes those inputs through the same one-based prompt protocol. V2V
    and RV2V reserve ``<Video 1>`` for the source timeline rather than a
    reference-video slot.
    """

    if not isinstance(
        draft, (I2VDraft, FL2VDraft, R2VDraft, V2VDraft, RV2VDraft)
    ):
        return
    for shot in shots:
        effective_prompt = shot.prompt.strip() or draft.prompt.strip()
        if not effective_prompt:
            continue
        available: dict[str, set[int]]
        if isinstance(draft, I2VDraft):
            available = {
                "Picture": {1} if shot.first_image is not None else set(),
                "Audio": set(),
                "Video": set(),
            }
        elif isinstance(draft, FL2VDraft):
            picture_count = int(shot.first_image is not None) + int(
                shot.last_image is not None
            )
            available = {
                "Picture": set(range(1, picture_count + 1)),
                "Audio": set(),
                "Video": set(),
            }
        elif isinstance(draft, R2VDraft):
            available = {
                "Picture": {asset.slot + 1 for asset in shot.reference_images},
                "Audio": {asset.slot + 1 for asset in shot.reference_audios},
                "Video": {asset.slot + 1 for asset in shot.reference_videos},
            }
        elif isinstance(draft, RV2VDraft):
            available = {
                "Picture": {asset.slot + 1 for asset in shot.reference_images},
                "Audio": {asset.slot + 1 for asset in shot.reference_audios},
                "Video": {1},
            }
        else:
            available = {"Picture": set(), "Audio": set(), "Video": {1}}
        for match in _REFERENCE_TAG.finditer(effective_prompt):
            kind = match.group(1).title()
            number = int(match.group(2))
            if number not in available[kind]:
                raise DraftNotRunnable(
                    f"shot '{shot.id}' prompt references <{kind} {number}>, "
                    f"but that reference slot does not exist"
                )


def validate_runnable(draft: ModeDraft) -> None:
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for shot in draft.shots:
        if shot.id in seen_ids:
            duplicate_ids.add(shot.id)
        seen_ids.add(shot.id)
    if duplicate_ids:
        raise DraftNotRunnable(
            f"shot ids must be unique; duplicates: {', '.join(sorted(duplicate_ids))}"
        )
    shots = [shot for shot in draft.shots if shot.enabled]
    if not shots:
        raise DraftNotRunnable("at least one enabled shot is required")
    if not draft.prompt.strip() and any(not shot.prompt.strip() for shot in shots):
        raise DraftNotRunnable("every enabled shot needs a prompt when the shared prompt is empty")
    if isinstance(draft, I2VDraft):
        missing = [shot.id for shot in shots if shot.first_image is None]
        if missing:
            raise DraftNotRunnable(f"i2v shots missing first_image: {', '.join(missing)}")
    elif isinstance(draft, FL2VDraft):
        missing = [shot.id for shot in shots if shot.first_image is None and shot.last_image is None]
        if missing:
            raise DraftNotRunnable(f"fl2v shots need first_image and/or last_image: {', '.join(missing)}")
    elif isinstance(draft, R2VDraft):
        missing = [
            shot.id
            for shot in shots
            if not (shot.reference_images or shot.reference_audios or shot.reference_videos)
        ]
        if missing:
            raise DraftNotRunnable(f"r2v shots need reference media: {', '.join(missing)}")
    elif isinstance(draft, V2VDraft) and not isinstance(draft, RV2VDraft):
        missing = [shot.id for shot in shots if shot.source_video is None]
        if missing:
            raise DraftNotRunnable(f"v2v shots missing source_video: {', '.join(missing)}")
    elif isinstance(draft, RV2VDraft):
        no_source = [shot.id for shot in shots if shot.source_video is None]
        if no_source:
            raise DraftNotRunnable(f"rv2v shots missing source_video: {', '.join(no_source)}")
    _validate_reference_tags(draft, shots)


def _asset_entry(asset: AssetReference, field: str, *, index: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        field: asset.comfy_path,
        "fileName": asset.name,
        "subfolder": asset.subfolder,
        "type": "input",
    }
    if index is not None:
        value["index"] = index
    return value


def _base_timeline(draft: ModeDraft, *, version: int, timeline_mode: str) -> dict[str, Any]:
    render = draft.render
    return {
        "version": version,
        "editMode": "segment",
        "timelineMode": timeline_mode,
        "totalFrames": 0,
        "frameRate": render.fps,
        "width": render.width,
        "height": render.height,
        "refMaxSize": max(render.width, render.height),
        "refImageSize": draft.ref_image_size,
        "output": {
            "mode": "fixed",
            "longEdge": max(render.width, render.height),
            "width": render.width,
            "height": render.height,
            "maxExportFrames": 0,
            "exportMode": "all",
            "audioMode": "generate",
            "continuityEnabled": False,
            "continuityOverlapFrames": 9,
        },
        "videoClips": [],
        "video": {
            "fileName": "",
            "videoFile": "",
            "subfolder": "",
            "type": "input",
            "frames": [],
            "frameMap": [],
        },
        "global": {
            "taskType": MODE_LABELS[draft.mode],
            "prompt": draft.prompt,
            "refs": [],
            "refAudios": [],
            "referenceVideo": {},
            "continuousReference": False,
            "genImage": {"imageFile": ""},
            "commonEnabled": False,
        },
        "segments": [],
        "gen": {"defaultFrameCount": 124},
        "runSelectEnabled": False,
        "runSelection": [],
    }


def _segment_common(draft: ModeDraft, shot: Any, start: int, frames: int) -> dict[str, Any]:
    return {
        "id": shot.id,
        "start": start,
        "length": frames,
        "frameCount": frames,
        "durationSec": shot.duration_seconds,
        "prompt": shot.prompt or draft.prompt,
        "taskType": "",
    }


def _build_prompt_batch(draft: T2VDraft | I2VDraft | R2VDraft) -> dict[str, Any]:
    timeline = _base_timeline(draft, version=5, timeline_mode="prompt_batch")
    cursor = 0
    segments: list[dict[str, Any]] = []
    for shot in draft.shots:
        if not shot.enabled:
            continue
        frames = align_h3_frames(shot.duration_seconds, draft.render.fps)
        segment = _segment_common(draft, shot, cursor, frames)
        segment.update(refs=[], refAudios=[], refVideos=[], genImage={"imageFile": ""})
        if isinstance(draft, I2VDraft):
            assert shot.first_image is not None
            segment["genImage"] = _asset_entry(shot.first_image, "imageFile")
        elif isinstance(draft, R2VDraft):
            segment["refs"] = [
                _asset_entry(asset, "imageFile", index=asset.slot)
                for asset in shot.reference_images
            ]
            segment["refAudios"] = [
                _asset_entry(asset, "audioFile", index=asset.slot)
                for asset in shot.reference_audios
            ]
            segment["refVideos"] = [
                _asset_entry(asset, "videoFile", index=asset.slot)
                for asset in shot.reference_videos
            ]
        segments.append(segment)
        cursor += frames
    timeline["segments"] = segments
    timeline["totalFrames"] = cursor
    _validate_total_frames(cursor)
    timeline["gen"]["defaultFrameCount"] = segments[0]["frameCount"]
    return timeline


def _build_fl2v(draft: FL2VDraft) -> dict[str, Any]:
    timeline = _base_timeline(draft, version=5, timeline_mode="fl2v")
    cursor = 0
    shots: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for shot in draft.shots:
        if not shot.enabled:
            continue
        frames = align_h3_frames(shot.duration_seconds, draft.render.fps)
        start_image = _asset_entry(shot.first_image, "imageFile") if shot.first_image else None
        end_image = _asset_entry(shot.last_image, "imageFile") if shot.last_image else None
        shots.append(
            {
                "id": shot.id,
                "durationSec": shot.duration_seconds,
                "prompt": shot.prompt or draft.prompt,
                "startImage": start_image,
                "endImage": end_image,
            }
        )
        segment = _segment_common(draft, shot, cursor, frames)
        segment.update(
            refs=[],
            isStartFrame=start_image is not None,
            isEndFrame=end_image is not None,
            genImage=start_image or {"imageFile": ""},
            endImage=end_image,
        )
        segments.append(segment)
        cursor += frames
    timeline["shots"] = shots
    timeline["segments"] = segments
    timeline["totalFrames"] = cursor
    _validate_total_frames(cursor)
    timeline["gen"]["defaultFrameCount"] = segments[0]["frameCount"]
    return timeline


def _build_video_timeline(draft: V2VDraft | RV2VDraft) -> dict[str, Any]:
    timeline = _base_timeline(draft, version=4, timeline_mode="video")
    cursor = 0
    segments: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []
    frame_map: list[dict[str, int]] = []
    for clip_index, shot in enumerate(shot for shot in draft.shots if shot.enabled):
        assert shot.source_video is not None
        output_frames = align_h3_frames(shot.duration_seconds, draft.render.fps)
        asset = shot.source_video
        assert asset.metadata is not None
        metadata = asset.metadata
        source_frame_count = max(1, int(round(metadata.duration * draft.render.fps)))
        source_start = min(
            source_frame_count - 1,
            max(0, int(round(shot.source_start_seconds * draft.render.fps))),
        )
        source_end = min(
            source_frame_count,
            max(
                source_start + 1,
                int(
                    round(
                        (shot.source_start_seconds + shot.source_duration_seconds)
                        * draft.render.fps
                    )
                ),
            ),
        )
        source_frames = source_end - source_start
        clip = _asset_entry(asset, "videoFile")
        clip.update(
            duration=metadata.duration,
            nativeFps=metadata.native_fps,
            nativeFrameCount=metadata.frame_count,
            width=metadata.width,
            height=metadata.height,
            sourceFrameCount=source_frame_count,
            logicalStart=cursor,
            logicalEnd=cursor + output_frames,
            longEdge=max(draft.render.width, draft.render.height),
        )
        clips.append(clip)
        for index in range(output_frames):
            if output_frames <= 1:
                source_index = source_start
            else:
                source_index = source_start + round(index * (source_frames - 1) / (output_frames - 1))
            source_index = min(source_frame_count - 1, max(0, source_index))
            frame_map.append({"clip": clip_index, "frame": source_index})
        segment = _segment_common(draft, shot, cursor, output_frames)
        segment.update(refs=[], refAudios=[], referenceVideo={})
        if isinstance(draft, RV2VDraft):
            segment["refs"] = [
                _asset_entry(asset, "imageFile", index=asset.slot)
                for asset in shot.reference_images
            ]
            segment["refAudios"] = [
                _asset_entry(asset, "audioFile", index=asset.slot)
                for asset in shot.reference_audios
            ]
        segments.append(segment)
        cursor += output_frames
    timeline["videoClips"] = clips
    timeline["video"] = {
        **clips[0],
        "frames": [],
        "frameMap": frame_map,
    }
    timeline["segments"] = segments
    timeline["totalFrames"] = cursor
    _validate_total_frames(cursor)
    return timeline


def build_timeline(draft: ModeDraft) -> dict[str, Any]:
    validate_runnable(draft)
    if isinstance(draft, FL2VDraft):
        return _build_fl2v(draft)
    if isinstance(draft, (V2VDraft, RV2VDraft)):
        return _build_video_timeline(draft)
    if isinstance(draft, (T2VDraft, I2VDraft, R2VDraft)):
        return _build_prompt_batch(draft)
    raise TypeError(f"unsupported draft type: {type(draft).__name__}")


def unified_model_families(
    draft: UnifiedTimelineDraft,
    *,
    segment_ids: list[str] | None = None,
) -> tuple[Literal["fl2va", "ref2va"], ...]:
    selected = set(segment_ids) if segment_ids is not None else None
    enabled_families = {
        segment.mode
        for segment in draft.segments
        if segment.enabled and (selected is None or segment.id in selected)
    }
    families: list[Literal["fl2va", "ref2va"]] = []
    if "fl2va" in enabled_families:
        families.append("fl2va")
    if "ref2va" in enabled_families:
        families.append("ref2va")
    return tuple(families)


def _unified_reference_slots(segment: UnifiedTimelineSegment) -> dict[str, set[int]]:
    if isinstance(segment, UnifiedFL2VASegment):
        picture_count = int(segment.first_image is not None) + int(
            segment.last_image is not None
        )
        return {
            "Picture": set(range(1, picture_count + 1)),
            "Audio": set(),
            "Video": set(),
        }
    if isinstance(segment, UnifiedRef2VASegment):
        paired_source_audio = (
            segment.source_video is not None
            and segment.source_audio_as_reference
        )
        audio_offset = 1 if paired_source_audio else 0
        video_offset = 1 if segment.source_video is not None else 0
        return {
            "Picture": {asset.slot + 1 for asset in segment.reference_images},
            "Audio": (
                ({1} if paired_source_audio else set())
                | {
                    asset.slot + 1 + audio_offset
                    for asset in segment.reference_audios
                }
            ),
            "Video": (
                ({1} if segment.source_video is not None else set())
                | {
                    asset.slot + 1 + video_offset
                    for asset in segment.reference_videos
                }
            ),
        }
    return {"Picture": set(), "Audio": set(), "Video": set()}


def _require_dense_reference_slots(segment: UnifiedTimelineSegment) -> None:
    """Match the official H3 autogrow presentation order exactly.

    ``MiniMaxH3ReferenceToVideo`` labels connected inputs densely in the order
    they are presented. Sparse persisted slot numbers would therefore make a
    prompt's ``<Picture N>/<Video N>/<Audio N>`` refer to a different asset.
    Native graph v1 rejects that ambiguity rather than silently renumbering it.
    """

    fields: list[tuple[str, list[Any]]] = []
    if isinstance(segment, UnifiedRef2VASegment):
        fields.extend(
            [
                ("reference_images", list(segment.reference_images)),
                ("reference_audios", list(segment.reference_audios)),
                ("reference_videos", list(segment.reference_videos)),
            ]
        )
    for field, values in fields:
        slots = sorted(item.slot for item in values)
        expected = list(range(len(slots)))
        if slots != expected:
            raise DraftNotRunnable(
                f"segment '{segment.id}' {field} slots must be dense 0..N-1 for "
                "the official MiniMax H3 reference autogrow inputs; "
                f"received {slots}"
            )


def unified_continuity_predecessors(
    draft: UnifiedTimelineDraft,
) -> dict[str, UnifiedTimelineSegment]:
    """Return the authored predecessor for every segment that consumes context.

    Adjacency is defined by the complete enabled timeline, never by a partial
    run selection.  An explicit FL2VA first image starts a new authored chain;
    it must not be combined silently with the preceding take at frame zero.
    """

    predecessors: dict[str, UnifiedTimelineSegment] = {}
    previous: UnifiedTimelineSegment | None = None
    for segment in (item for item in draft.segments if item.enabled):
        anchor_reset = (
            isinstance(segment, UnifiedFL2VASegment)
            and segment.first_image is not None
        )
        if (
            segment.continuity.enabled
            and previous is not None
            and not anchor_reset
        ):
            predecessors[segment.id] = previous
        previous = segment
    return predecessors


def _validate_unified_continuity_selection(
    draft: UnifiedTimelineDraft,
    execution: list[UnifiedTimelineSegment],
) -> None:
    predecessors = unified_continuity_predecessors(draft)
    for segment in execution:
        predecessor = predecessors.get(segment.id)
        if predecessor is None:
            continue
        overlap = segment.continuity.overlap_frames
        predecessor_frames = align_h3_frames(
            predecessor.duration_seconds, draft.render.fps
        )
        if predecessor_frames < overlap:
            raise DraftNotRunnable(
                f"segment '{predecessor.id}' has {predecessor_frames} visible frames, "
                f"fewer than the requested {overlap} continuity tail frames"
            )
        visible_frames = align_h3_frames(
            segment.duration_seconds, draft.render.fps
        )
        requested = visible_frames + overlap
        sample_frames = requested + ((5 - requested % 17) % 17)
        if sample_frames > MAX_SEGMENT_FRAMES:
            raise DraftNotRunnable(
                f"segment '{segment.id}' needs {sample_frames} internal sample frames "
                f"({visible_frames} visible + {overlap} predecessor-tail context, H3 "
                f"aligned), above the {MAX_SEGMENT_FRAMES}-frame limit"
            )


def validate_unified_runnable(
    draft: UnifiedTimelineDraft,
    *,
    segment_ids: list[str] | None = None,
) -> list[UnifiedTimelineSegment]:
    enabled = [segment for segment in draft.segments if segment.enabled]
    if not enabled:
        raise DraftNotRunnable("at least one enabled timeline segment is required")
    by_id = {segment.id: segment for segment in enabled}
    if segment_ids is not None:
        missing = [segment_id for segment_id in segment_ids if segment_id not in by_id]
        if missing:
            raise DraftNotRunnable(
                "segment_ids must name enabled timeline segments; unknown or disabled: "
                + ", ".join(missing)
            )
    execution = (
        [by_id[segment_id] for segment_id in segment_ids]
        if segment_ids is not None
        else enabled
    )
    if any(not segment.prompt.strip() for segment in execution):
        raise DraftNotRunnable("every enabled timeline segment needs a prompt")
    if draft.render.fps != 24.0:
        raise DraftNotRunnable(
            "native MiniMax H3 temporal and reference-video contracts are fixed at "
            "24 fps; render.fps must equal 24"
        )
    _validate_unified_continuity_selection(draft, execution)
    missing_ref2va = [
        segment.id
        for segment in execution
        if isinstance(segment, UnifiedRef2VASegment)
        and segment.source_video is None
        and not (
            segment.reference_images
            or segment.reference_audios
            or segment.reference_videos
        )
    ]
    if missing_ref2va:
        raise DraftNotRunnable(
            "Ref2VA segments need source_video or independent reference media: "
            + ", ".join(missing_ref2va)
        )
    silent_source_references = [
        segment.id
        for segment in execution
        if isinstance(segment, UnifiedRef2VASegment)
        and segment.source_audio_as_reference
        and (
            segment.source_video is None
            or segment.source_video.metadata is None
            or not segment.source_video.metadata.has_audio
        )
    ]
    if silent_source_references:
        raise DraftNotRunnable(
            "source_audio_as_reference requires a server-probed source video "
            "with an audio stream: " + ", ".join(silent_source_references)
        )
    for segment in execution:
        _require_dense_reference_slots(segment)
        prompt = segment.prompt.strip()
        available = _unified_reference_slots(segment)
        for match in _REFERENCE_TAG.finditer(prompt):
            kind = match.group(1).title()
            number = int(match.group(2))
            if number not in available[kind]:
                raise DraftNotRunnable(
                    f"segment '{segment.id}' prompt references <{kind} {number}>, "
                    "but that reference slot does not exist for its mode"
                )
    source_audio_segments = [
        segment for segment in execution if segment.audio_mode == "source"
    ]
    if any(
        not isinstance(segment, UnifiedRef2VASegment)
        or segment.source_video is None
        for segment in source_audio_segments
    ):
        raise DraftNotRunnable(
            "audio_mode='source' requires that segment to be Ref2VA "
            "with source_video"
        )
    for segment in source_audio_segments:
        assert isinstance(segment, UnifiedRef2VASegment)
        assert segment.source_video is not None
        assert segment.source_video.metadata is not None
        if not segment.source_video.metadata.has_audio:
            raise DraftNotRunnable(
                f"segment '{segment.id}' cannot use audio_mode='source': "
                "the server-probed source video has no audio stream"
            )
        fps = draft.render.fps
        full_frames = max(
            1, int(round(segment.source_video.metadata.duration * fps))
        )
        source_start = min(
            full_frames - 1,
            max(0, int(round(segment.source_start_seconds * fps))),
        )
        source_end = min(
            full_frames,
            max(
                source_start + 1,
                int(
                    round(
                        (
                            segment.source_start_seconds
                            + segment.source_duration_seconds
                        )
                        * fps
                    )
                ),
            ),
        )
        output_frames = align_h3_frames(segment.duration_seconds, fps)
        if source_end - source_start != output_frames:
            raise DraftNotRunnable(
                f"segment '{segment.id}' source audio cannot be time-stretched: "
                f"source trim is {source_end - source_start} frames but output is "
                f"{output_frames} frames; make source_duration_seconds match the "
                f"aligned output duration ({output_frames / fps:.6g} seconds)"
            )
    return enabled


def _source_video_entry(
    segment: UnifiedRef2VASegment,
    *,
    fps: float,
    output_frames: int,
    clip_index: int,
    logical_start: int,
    long_edge: int,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    assert segment.source_video is not None
    asset = segment.source_video
    assert asset.metadata is not None
    metadata = asset.metadata
    full_source_frames = max(1, int(round(metadata.duration * fps)))
    source_start = min(
        full_source_frames - 1,
        max(0, int(round(segment.source_start_seconds * fps))),
    )
    source_end = min(
        full_source_frames,
        max(
            source_start + 1,
            int(
                round(
                    (segment.source_start_seconds + segment.source_duration_seconds)
                    * fps
                )
            ),
        ),
    )
    source_count = source_end - source_start
    entry = _asset_entry(asset, "videoFile")
    entry.update(
        {
            "clipIndex": clip_index,
            "duration": metadata.duration,
            "nativeFps": metadata.native_fps,
            "nativeFrameCount": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
            "sourceTotalFrames": full_source_frames,
            "sourceFrameCount": source_count,
            "sourceStartFrame": source_start,
            "sourceEndFrame": source_end,
            "logicalStart": logical_start,
            "logicalEnd": logical_start + output_frames,
            "longEdge": long_edge,
        }
    )
    mapping: list[dict[str, int]] = []
    for index in range(output_frames):
        if output_frames <= 1:
            source_index = source_start
        else:
            source_index = source_start + round(
                index * (source_count - 1) / (output_frames - 1)
            )
        mapping.append(
            {
                "clip": clip_index,
                "frame": min(full_source_frames - 1, max(0, source_index)),
            }
        )
    return entry, mapping


def _unified_cache_key(
    draft: UnifiedTimelineDraft,
    segment: UnifiedTimelineSegment,
    settings: RuntimeSettings,
    *,
    predecessor_cache_key: str | None,
    successor_signature: dict[str, Any] | None,
) -> str:
    family = segment.mode
    payload = {
        # Timeline v4 also moved media policies onto each segment.
        "schema": "director-web-segment-cache-v4",
        "segment": segment.model_dump(mode="json"),
        "render": draft.render.model_dump(mode="json"),
        # Cache identity follows the exact family branch used by this segment;
        # changing Ref2VA controls must not invalidate an FL2VA take and vice
        # versa. The concrete visible seed participates directly.
        "sampling": getattr(draft.sampling, family).model_dump(mode="json"),
        "family": family,
        "family_model": getattr(settings.models, family).model_dump(mode="json"),
        "clip": settings.models.clip.model_dump(mode="json"),
        "video_vae": settings.models.video_vae.model_dump(mode="json"),
        "audio_vae": settings.models.audio_vae.model_dump(mode="json"),
        "comfy_origin": str(settings.comfy_url).rstrip("/"),
        "predecessor_cache_key": predecessor_cache_key,
        # Continuity phase alignment may trim and rewrite this segment after
        # observing the next segment. Include that boundary in the physical
        # cache identity so a changed successor can never reuse/overwrite the
        # former predecessor render.
        "successor": successor_signature,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def build_unified_timeline(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    *,
    segment_ids: list[str] | None = None,
) -> dict[str, Any]:
    enabled = validate_unified_runnable(draft, segment_ids=segment_ids)
    continuity_predecessors = unified_continuity_predecessors(draft)
    selected_ids = set(segment_ids) if segment_ids is not None else None
    selected_indices = (
        [index for index, segment in enumerate(enabled) if segment.id in selected_ids]
        if selected_ids is not None
        else list(range(len(enabled)))
    )
    partial_selection = len(selected_indices) != len(enabled)
    render = draft.render
    long_edge = max(render.width, render.height)
    first_recipe = (
        timeline_segment_recipe(
            next(segment for segment in enabled if segment.id in selected_ids)
        )
        if selected_ids is not None
        else timeline_segment_recipe(enabled[0])
    )
    timeline: dict[str, Any] = {
        "version": 6,
        "timelineMode": "mixed",
        "editMode": "segment",
        "totalFrames": 0,
        "frameRate": render.fps,
        "width": render.width,
        "height": render.height,
        "refMaxSize": long_edge,
        "refImageSize": enabled[0].ref_image_size,
        "refImageSizeBySegment": {
            segment.id: segment.ref_image_size for segment in enabled
        },
        "output": {
            "mode": "fixed",
            "longEdge": long_edge,
            "width": render.width,
            "height": render.height,
            "maxExportFrames": 0,
            # Partial execution is intentionally segment-only.  An all-export
            # merge could otherwise splice stale cache or omit uncached gen
            # segments and then pad them into a misleading long video.
            "exportMode": "segments" if partial_selection else draft.export_mode,
            "audioMode": enabled[0].audio_mode,
            "audioModeBySegment": {
                segment.id: segment.audio_mode for segment in enabled
            },
            # The retired mixed-builder format had one project-wide
            # continuity switch. V3 cannot be represented by that lossy pair,
            # so expose the exact incoming setting keyed by stable segment ID.
            "continuityBySegment": {
                segment.id: segment.continuity.model_dump(mode="json")
                for segment in enabled
            },
        },
        "videoClips": [],
        "video": {
            "fileName": "",
            "videoFile": "",
            "subfolder": "",
            "type": "input",
            "frames": [],
            "frameMap": [],
        },
        "global": {
            # The mixed builder treats this only as a compatibility fallback;
            # every segment below carries its authoritative full task label.
            "taskType": MODE_LABELS[first_recipe],
            "prompt": "",
            "refs": [],
            "refAudios": [],
            "referenceVideo": {},
            "continuousReference": False,
            "genImage": {"imageFile": ""},
            "commonEnabled": False,
        },
        "segments": [],
        "gen": {"defaultFrameCount": 124},
        "runSelectEnabled": partial_selection,
        "runSelection": selected_indices if partial_selection else [],
    }
    cursor = 0
    source_clips: list[dict[str, Any]] = []
    source_mappings: list[list[dict[str, int]] | None] = []
    compiled_segments: list[dict[str, Any]] = []
    previous_cache_key: str | None = None
    for enabled_index, segment in enumerate(enabled):
        frames = align_h3_frames(segment.duration_seconds, render.fps)
        compiled: dict[str, Any] = {
            "id": segment.id,
            "start": cursor,
            "length": frames,
            "frameCount": frames,
            "durationSec": segment.duration_seconds,
            "prompt": segment.prompt,
            "taskType": MODE_LABELS[timeline_segment_recipe(segment)],
            "refs": [],
            "refAudios": [],
            "refVideos": [],
            "genImage": {"imageFile": ""},
            "endImage": None,
            "sourceVideo": {},
            "continuity": segment.continuity.model_dump(mode="json"),
        }
        predecessor_cache_key = (
            previous_cache_key
            if segment.id in continuity_predecessors
            and previous_cache_key is not None
            else None
        )
        successor = (
            enabled[enabled_index + 1]
            if enabled_index + 1 < len(enabled)
            else None
        )
        cache_key = _unified_cache_key(
            draft,
            segment,
            settings,
            predecessor_cache_key=predecessor_cache_key,
            successor_signature=(
                {
                    "segment": successor.model_dump(mode="json"),
                    "family": successor.mode,
                    "family_model": getattr(
                        settings.models,
                        successor.mode,
                    ).model_dump(mode="json"),
                }
                if successor is not None
                and continuity_predecessors.get(successor.id) is not None
                and continuity_predecessors[successor.id].id == segment.id
                else None
            ),
        )
        compiled["cacheKey"] = cache_key
        if predecessor_cache_key is not None:
            compiled["predecessorCacheKey"] = predecessor_cache_key
        if isinstance(segment, UnifiedFL2VASegment):
            if segment.first_image is not None:
                compiled["genImage"] = _asset_entry(
                    segment.first_image, "imageFile"
                )
            if segment.last_image is not None:
                compiled["endImage"] = _asset_entry(
                    segment.last_image, "imageFile"
                )
        elif isinstance(segment, UnifiedRef2VASegment):
            compiled["refs"] = [
                _asset_entry(asset, "imageFile", index=asset.slot)
                for asset in segment.reference_images
            ]
            compiled["refAudios"] = [
                _asset_entry(asset, "audioFile", index=asset.slot)
                for asset in segment.reference_audios
            ]
            compiled["refVideos"] = [
                _asset_entry(
                    asset,
                    "videoFile",
                    index=asset.slot + (1 if segment.source_video is not None else 0),
                )
                for asset in segment.reference_videos
            ]
            # A non-selected unfinished source segment is a valid editing
            # state. The mixed node does not inspect it when runSelection
            # excludes the segment, so only compile a source descriptor when
            # the user has actually supplied one.
            if segment.source_video is not None:
                source_entry, mapping = _source_video_entry(
                    segment,
                    fps=render.fps,
                    output_frames=frames,
                    clip_index=len(source_clips),
                    logical_start=cursor,
                    long_edge=long_edge,
                )
                compiled["sourceVideo"] = dict(source_entry)
                root_clip = dict(source_entry)
                # Root ``videoClips`` follows the legacy video-IO contract where
                # sourceFrameCount is the complete resampled file length.  Nested
                # ``sourceVideo`` follows the mixed-builder contract where it is
                # the selected end-exclusive range length.
                root_clip["sourceFrameCount"] = source_entry["sourceTotalFrames"]
                source_clips.append(root_clip)
                source_mappings.append(mapping)
            else:
                source_mappings.append(None)
        if not isinstance(segment, UnifiedRef2VASegment):
            source_mappings.append(None)
        compiled_segments.append(compiled)
        previous_cache_key = cache_key
        cursor += frames
    _validate_total_frames(cursor)
    timeline["segments"] = compiled_segments
    timeline["totalFrames"] = cursor
    timeline["gen"]["defaultFrameCount"] = compiled_segments[0]["frameCount"]
    timeline["videoClips"] = source_clips
    if source_clips:
        timeline["video"] = {**source_clips[0], "frames": [], "frameMap": []}
    # Source-audio extraction indexes root frameMap by absolute logical frame.
    # For partial segment export, unselected generation gaps receive inert map
    # entries; they are never decoded, while selected source ranges retain the
    # same end-exclusive coordinates as nested sourceVideo.
    if source_clips:
        placeholder = {"clip": 0, "frame": 0}
        dense_map = [
            entry
            for segment, mapping in zip(enabled, source_mappings, strict=True)
            for entry in (
                mapping
                if mapping is not None
                else [placeholder] * align_h3_frames(
                    segment.duration_seconds, render.fps
                )
            )
        ]
        if len(dense_map) != cursor:
            raise AssertionError("source frame map must cover the complete timeline")
        timeline["video"]["frameMap"] = dense_map
    return timeline


def compile_unified_prompt(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    *,
    job_id: str,
    segment_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Literal["fl2va", "ref2va"], ...]]:
    """Compatibility wrapper for a single native workflow unit.

    Mixed-family timelines necessarily compile to multiple parent-owned child
    prompts and therefore cannot be represented by this historical return
    type. New code must use ``compile_native_timeline`` directly.
    """

    from .native_templates import NativeTemplateError, compile_native_timeline

    try:
        compiled = compile_native_timeline(
            draft, settings, job_id, segment_ids=segment_ids
        )
    except NativeTemplateError as exc:
        raise DraftNotRunnable(str(exc)) from exc
    if len(compiled.workflows) != 1:
        raise DraftNotRunnable(
            "mixed native timelines compile to multiple workflow units; use "
            "compile_native_timeline and the parent/child job orchestrator"
        )
    return (
        compiled.workflows[0].prompt,
        compiled.manifest,
        compiled.families,
    )


def compile_prompt(
    draft: ModeDraft,
    settings: RuntimeSettings,
    *,
    job_id: str,
) -> dict[str, Any]:
    """Compile a legacy six-mode draft through the native template builder."""

    from .native_templates import NativeTemplateError, compile_native_timeline

    # Keep the caller's explicit legacy recipe contract before widening it to
    # a v2 family shape whose recipe is inferred from present assets.
    validate_runnable(draft)
    try:
        compiled = compile_native_timeline(
            mode_draft_to_timeline(draft), settings, job_id
        )
    except NativeTemplateError as exc:
        raise DraftNotRunnable(str(exc)) from exc
    if len(compiled.workflows) != 1:
        raise DraftNotRunnable("one legacy mode must compile to one native workflow unit")
    return compiled.workflows[0].prompt
