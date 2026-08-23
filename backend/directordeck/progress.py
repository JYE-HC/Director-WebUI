from __future__ import annotations

"""ComfyUI-native progress streaming for server-owned workflow units.

ComfyUI sends prompt/node lifecycle and sampler progress to the websocket
belonging to the ``client_id`` that submitted a prompt. The browser never
connects to that socket directly: the backend owns the client id, consumes the
standard ``execution_start``/``executing``/``progress`` events, and persists a
compact child-job snapshot for normal API polling.

Queue/history remain the lifecycle authority.  Losing this optional websocket
therefore loses only live stage/step detail, never the durable
queued/running/terminal state of a job.
"""

import asyncio
import json
import ssl
import struct
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import websockets

from .workflow.execution import PreviewSpec, ProgressSpec


_SAMPLER_CLASSES = frozenset(
    {
        "SamplerCustomAdvanced",
        "DirectorDeckRayXFuserSamplerCustomAdvanced",
        # Historical task display only; new submissions use the Director alias.
        "XFuserSamplerCustomAdvanced",
    }
)

# ``executing`` is the only standard ComfyUI event that identifies long-running
# nodes which do not publish their own numeric progress.  Keep the labels tied
# to the server-owned native templates: this is deliberately not a generic
# workflow/node-name renderer.
_EXECUTION_STAGES: dict[str, str] = {
    "CLIPLoader": "加载文本编码器",
    "SelectCLIPDevice": "分配文本编码器设备",
    "VAELoader": "加载 VAE",
    "SelectVAEDevice": "分配 VAE 设备",
    "UNETLoader": "加载生成模型",
    "SelectModelDevice": "分配生成模型设备",
    "LoraLoaderModelOnly": "加载 LoRA",
    "LoraLoaderBypassModelOnly": "加载 LoRA",
    "MiniMaxH3TurboLoRA": "加载 H3 Turbo LoRA",
    "MiniMaxH3SigmaShift": "配置生成模型",
    "RayInitializerAdvanced": "初始化 RayLight 多卡",
    "DirectorDeckRayInitializerAdvanced": "初始化 RayLight 多卡",
    "RayLoraLoader": "加载 RayLight LoRA",
    "DirectorDeckRayLoraLoader": "加载 RayLight LoRA",
    "RayUNETLoader": "加载 RayLight 生成模型",
    "DirectorDeckRayUNETLoader": "加载 RayLight 生成模型",
    "RayMiniMaxH3SigmaShift": "配置 RayLight 生成模型",
    "DirectorDeckRayMiniMaxH3SigmaShift": "配置 RayLight 生成模型",
    "LoadImage": "读取参考图",
    "LoadVideo": "读取参考视频",
    "Video Slice": "裁剪参考视频",
    "GetVideoComponents": "解析参考视频",
    "LoadAudio": "读取参考音频",
    "ImageFromBatch": "处理参考图",
    "MiniMaxH3ImageToVideo": "构建画面条件",
    "MiniMaxH3ReferenceToVideo": "构建多模态条件",
    "BasicGuider": "准备采样引导",
    "RayBasicGuider": "准备 RayLight 采样引导",
    "DirectorDeckRayBasicGuider": "准备 RayLight 采样引导",
    "BasicScheduler": "生成采样计划",
    "RayBasicScheduler": "生成 RayLight 采样计划",
    "DirectorDeckRayBasicScheduler": "生成 RayLight 采样计划",
    "KSamplerSelect": "选择采样器",
    "RandomNoise": "生成初始噪声",
    "SamplerCustomAdvanced": "采样中",
    "XFuserSamplerCustomAdvanced": "RayLight 采样中",
    "DirectorDeckRayXFuserSamplerCustomAdvanced": "RayLight 采样中",
    "VAEDecode": "解码视频画面",
    "VAEDecodeAudio": "解码音频",
    "CreateVideo": "封装音视频",
    "SaveVideo": "写入视频文件",
}

# Percentages outside sampling are coarse phase landmarks. ComfyUI reports no
# byte/frame total for model loading, conditioning, VAE decode or SaveVideo, so
# inventing a continuously moving counter would be misleading. Sampler events
# occupy the middle window and retain their exact step/total label.
_PREPARING_PROGRESS = 0.10
_SAMPLING_START = 0.15
_SAMPLING_END = 0.85
_POSTPROCESS_PROGRESS: dict[str, float] = {
    "VAEDecode": 0.90,
    "VAEDecodeAudio": 0.90,
    "CreateVideo": 0.95,
    "SaveVideo": 0.98,
}
# Queue/history reconciliation uses 1% only as lifecycle evidence that ComfyUI
# has started the prompt.  It is not a real workflow-phase position.  Explicit
# zero-boundary node events must therefore be allowed to replace the stale
# stage label while retaining this numeric floor.
_RUNNING_LIFECYCLE_PROGRESS_FLOOR = 0.01
PREVIEW_EVENT_WITH_METADATA = 4
MAX_PREVIEW_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_PREVIEW_METADATA_BYTES = 64 * 1024
MAX_PREVIEW_PHASE_WATERMARKS = 4096
_PROGRESS_BOUNDARY_EPSILON = 1e-12


@dataclass(frozen=True)
class ComfyProgressEvent:
    prompt_id: str
    node_id: str
    value: float
    maximum: float
    from_progress_state: bool = False


@dataclass(frozen=True)
class ComfyExecutionEvent:
    """A standard ComfyUI prompt/node lifecycle event.

    ``node_id=None`` represents ``execution_start``. A terminal
    ``executing(node=None)`` event is intentionally not represented because
    queue/history, rather than the optional websocket, owns completion.
    """

    prompt_id: str
    node_id: str | None


@dataclass(frozen=True)
class ComfyReconcileHint:
    """A WebSocket hint that queue/history authority may have changed.

    These events deliberately carry no lifecycle decision.  Some arrive before
    ComfyUI commits prompt history, and ``status`` does not identify a prompt at
    all, so the only safe consumer action is to wake normal reconciliation.
    """

    event_type: str
    prompt_id: str | None


@dataclass(frozen=True)
class ChildProgress:
    progress: float
    stage: str


@dataclass(frozen=True)
class ComfyPreviewEvent:
    prompt_id: str
    node_id: str
    mime_type: str
    content: bytes


@dataclass(frozen=True)
class LivePreview:
    job_id: str
    child_id: str
    segment_id: str
    prompt_id: str
    node_id: str
    mime_type: str
    content: bytes
    stored_at: float
    source_phase_id: str | None = None
    source_phase_index: int | None = None
    source_priority: int | None = None
    source_supersedes: tuple[str, ...] = ()
    source_group_index: int | None = None


@dataclass(frozen=True)
class ResolvedPreviewSource:
    """One child-owned preview source resolved from persisted execution evidence."""

    segment_id: str
    node_id: str
    phase_id: str
    phase_index: int | None
    priority: int
    supersedes: tuple[str, ...]
    group_index: int | None
    persisted: bool


def websocket_url(base_url: str, client_id: str) -> str:
    """Build ComfyUI's websocket URL while preserving a reverse-proxy prefix."""

    parsed = urlsplit(base_url.rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("ComfyUI websocket requires an absolute HTTP(S) URL")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = f"{parsed.path.rstrip('/')}/ws"
    return urlunsplit((scheme, parsed.netloc, path, urlencode({"clientId": client_id}), ""))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _event_from_data(data: Any) -> ComfyProgressEvent | None:
    if not isinstance(data, Mapping):
        return None
    prompt_id = data.get("prompt_id")
    node_id = data.get("node") or data.get("node_id")
    value = _finite_number(data.get("value"))
    maximum = _finite_number(data.get("max"))
    if (
        not isinstance(prompt_id, str)
        or not prompt_id
        or not isinstance(node_id, (str, int))
        or value is None
        or maximum is None
        or maximum <= 0
    ):
        return None
    return ComfyProgressEvent(
        prompt_id=prompt_id,
        node_id=str(node_id),
        value=max(0.0, min(value, maximum)),
        maximum=maximum,
    )


def parse_progress_message(message: str | bytes) -> list[ComfyProgressEvent]:
    """Parse only ComfyUI's standard JSON progress messages.

    Binary preview frames are deliberately ignored.  Both the classic
    ``progress`` event and the newer aggregate ``progress_state`` event are
    accepted so this remains compatible across ComfyUI frontend generations.
    """

    if isinstance(message, bytes):
        return []
    try:
        payload = json.loads(message)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, Mapping):
        return []
    event_type = payload.get("type")
    data = payload.get("data")
    if event_type == "progress":
        event = _event_from_data(data)
        return [event] if event is not None else []
    if event_type != "progress_state" or not isinstance(data, Mapping):
        return []
    prompt_id = data.get("prompt_id")
    nodes = data.get("nodes")
    if not isinstance(prompt_id, str) or not isinstance(nodes, Mapping):
        return []
    events: list[ComfyProgressEvent] = []
    for node_id, state in nodes.items():
        if not isinstance(state, Mapping) or state.get("state") != "running":
            continue
        event = _event_from_data(
            {
                "prompt_id": prompt_id,
                "node": state.get("node_id", node_id),
                "value": state.get("value"),
                "max": state.get("max"),
            }
        )
        if event is not None:
            events.append(
                ComfyProgressEvent(
                    prompt_id=event.prompt_id,
                    node_id=event.node_id,
                    value=event.value,
                    maximum=event.maximum,
                    from_progress_state=True,
                )
            )
    return events


def parse_execution_message(message: str | bytes) -> ComfyExecutionEvent | None:
    """Parse prompt start and per-node ``executing`` websocket events.

    A reconnect notification from ComfyUI contains only the current node and
    omits ``prompt_id``. It cannot be attributed safely when the same client id
    owns multiple queued prompts, so it is deliberately ignored.
    """

    if isinstance(message, bytes):
        return None
    try:
        payload = json.loads(message)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    event_type = payload.get("type")
    data = payload.get("data")
    if event_type not in {"execution_start", "executing"} or not isinstance(
        data, Mapping
    ):
        return None
    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        return None
    if event_type == "execution_start":
        return ComfyExecutionEvent(prompt_id=prompt_id, node_id=None)
    node_id = data.get("node")
    if (
        node_id is None
        or isinstance(node_id, bool)
        or not isinstance(node_id, (str, int))
    ):
        # ComfyUI sends executing(node=None) after prompt execution. History is
        # the durable success/failure authority and will close the child.
        return None
    return ComfyExecutionEvent(prompt_id=prompt_id, node_id=str(node_id))


def parse_reconcile_message(message: str | bytes) -> ComfyReconcileHint | None:
    """Parse queue/terminal WebSocket events only as reconciliation wake hints."""

    if isinstance(message, bytes):
        return None
    try:
        payload = json.loads(message)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    event_type = payload.get("type")
    data = payload.get("data")
    if event_type == "status" and isinstance(data, Mapping):
        return ComfyReconcileHint(event_type="status", prompt_id=None)
    if event_type not in {
        "execution_success",
        "execution_error",
        "execution_interrupted",
        "executing",
    } or not isinstance(data, Mapping):
        return None
    prompt_id = data.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        return None
    if event_type == "executing" and data.get("node", object()) is not None:
        return None
    return ComfyReconcileHint(event_type=event_type, prompt_id=prompt_id)


def parse_preview_message(message: str | bytes) -> ComfyPreviewEvent | None:
    """Parse only ComfyUI's metadata-bearing preview event (binary type 4).

    Legacy binary previews do not identify their prompt or sampler node and
    can be misattributed when multiple jobs share an endpoint. They are always
    rejected. The whole frame, metadata and encoded image included, is bounded
    before JSON parsing or allocation into the live cache.
    """

    if not isinstance(message, bytes):
        return None
    if len(message) < 9 or len(message) > MAX_PREVIEW_MESSAGE_BYTES:
        return None
    event_type = struct.unpack(">I", message[:4])[0]
    if event_type != PREVIEW_EVENT_WITH_METADATA:
        return None
    metadata_length = struct.unpack(">I", message[4:8])[0]
    if (
        metadata_length <= 0
        or metadata_length > MAX_PREVIEW_METADATA_BYTES
        or 8 + metadata_length >= len(message)
    ):
        return None
    try:
        metadata = json.loads(message[8 : 8 + metadata_length].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(metadata, Mapping):
        return None
    prompt_id = metadata.get("prompt_id")
    node_id = metadata.get("node_id")
    mime_type = metadata.get("image_type")
    if (
        not isinstance(prompt_id, str)
        or not prompt_id
        or not isinstance(node_id, (str, int))
        or isinstance(node_id, bool)
        or mime_type not in {"image/jpeg", "image/png"}
    ):
        return None
    content = message[8 + metadata_length :]
    if mime_type == "image/png":
        valid_signature = content.startswith(b"\x89PNG\r\n\x1a\n")
    else:
        valid_signature = content.startswith(b"\xff\xd8\xff")
    if not valid_signature:
        return None
    return ComfyPreviewEvent(
        prompt_id=prompt_id,
        node_id=str(node_id),
        mime_type=mime_type,
        content=content,
    )


_ABSENT = object()


def _child_exact_prompt_snapshot(
    child: Mapping[str, Any],
) -> tuple[bool, Any]:
    """Return persisted exact evidence without confusing absence with corruption."""

    if "exact_prompt_snapshot" in child:
        return True, child.get("exact_prompt_snapshot")
    evidence = child.get("execution_evidence")
    if isinstance(evidence, Mapping):
        if "exact_prompt_snapshot" in evidence:
            return True, evidence.get("exact_prompt_snapshot")
        # Any execution-evidence marker means this is a typed child.  An
        # invalid sentinel or incomplete row must disable every legacy
        # class/output-node fallback instead of authorizing mutable columns.
        return True, None
    elif evidence is not None:
        value = getattr(evidence, "exact_prompt_snapshot", _ABSENT)
        return True, None if value is _ABSENT else value
    return False, None


def _snapshot_field(snapshot: Any, name: str) -> Any:
    if isinstance(snapshot, Mapping):
        return snapshot.get(name, _ABSENT)
    return getattr(snapshot, name, _ABSENT)


def _persisted_progress_spec(
    child: Mapping[str, Any],
) -> tuple[bool, ProgressSpec | None]:
    present, snapshot = _child_exact_prompt_snapshot(child)
    if not present:
        return False, None
    raw = _snapshot_field(snapshot, "progress_spec")
    if raw is _ABSENT or raw is None:
        return True, None
    try:
        if isinstance(raw, ProgressSpec):
            spec = raw
        elif isinstance(raw, Mapping):
            spec = ProgressSpec.model_validate_json(json.dumps(raw))
        else:
            spec = ProgressSpec.model_validate(raw)
    except (TypeError, ValueError):
        # Persisted-but-invalid evidence must fail closed. Falling back to node
        # class guesses would silently replace the Stage-4 authority.
        return True, None
    exact_prompt = _snapshot_field(snapshot, "exact_prompt")
    if not isinstance(exact_prompt, Mapping) or any(
        phase.node_id not in exact_prompt for phase in spec.phases
    ):
        return True, None
    return True, spec


def _persisted_preview_spec(
    child: Mapping[str, Any],
) -> tuple[bool, Any, PreviewSpec | None]:
    present, snapshot = _child_exact_prompt_snapshot(child)
    if not present:
        return False, None, None
    raw = _snapshot_field(snapshot, "preview_spec")
    if raw is _ABSENT or raw is None:
        return True, snapshot, None
    try:
        if isinstance(raw, PreviewSpec):
            preview_spec = raw
        elif isinstance(raw, Mapping):
            preview_spec = PreviewSpec.model_validate_json(json.dumps(raw))
        else:
            preview_spec = PreviewSpec.model_validate(raw)
    except (TypeError, ValueError):
        return True, snapshot, None

    progress_present, progress_spec = _persisted_progress_spec(child)
    if not progress_present or progress_spec is None:
        return True, snapshot, None
    phase_ids = {phase.id for phase in progress_spec.phases}
    exact_prompt = _snapshot_field(snapshot, "exact_prompt")
    if not isinstance(exact_prompt, Mapping) or any(
        source.phase_id not in phase_ids or source.node_id not in exact_prompt
        for source in preview_spec.sources
    ):
        return True, snapshot, None
    return True, snapshot, preview_spec


def _legacy_sampler_segment_for_node(
    child: Mapping[str, Any], node_id: str
) -> str | None:
    prompt = child.get("prompt_snapshot")
    segment_ids = child.get("segment_ids")
    if not isinstance(prompt, Mapping) or not isinstance(segment_ids, list):
        return None
    sampler_ids = [
        str(candidate_id)
        for candidate_id, node in prompt.items()
        if isinstance(node, Mapping) and node.get("class_type") in _SAMPLER_CLASSES
    ]
    if node_id not in sampler_ids or len(sampler_ids) != len(segment_ids):
        return None
    index = sampler_ids.index(node_id)
    segment_id = segment_ids[index]
    return str(segment_id) if isinstance(segment_id, str) and segment_id else None


def _snapshot_segment_id(snapshot: Any, child: Mapping[str, Any]) -> str | None:
    owner = _snapshot_field(snapshot, "owner_segment_id")
    expected = _snapshot_field(snapshot, "expected_output_spec")
    expected_segment = _snapshot_field(expected, "segment_id")
    candidates = tuple(
        value
        for value in (owner, expected_segment)
        if isinstance(value, str) and value
    )
    if not candidates or len(set(candidates)) != 1:
        return None
    segment_id = candidates[0]
    child_segments = child.get("segment_ids")
    if (
        not isinstance(child_segments, list)
        or len(child_segments) != 1
        or child_segments[0] != segment_id
    ):
        return None
    return segment_id


def _child_group_index(child: Mapping[str, Any]) -> int | None:
    value = child.get("group_index")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def preview_source_for_node(
    child: Mapping[str, Any], node_id: str
) -> ResolvedPreviewSource | None:
    """Resolve preview ownership from persisted spec, with legacy-only fallback."""

    persisted, snapshot, preview_spec = _persisted_preview_spec(child)
    if persisted:
        if preview_spec is None:
            return None
        exact_prompt = _snapshot_field(snapshot, "exact_prompt")
        if not isinstance(exact_prompt, Mapping) or node_id not in exact_prompt:
            return None
        source = next(
            (candidate for candidate in preview_spec.sources if candidate.node_id == node_id),
            None,
        )
        if source is None or source.publish is not True:
            return None
        progress_present, progress_spec = _persisted_progress_spec(child)
        if not progress_present or progress_spec is None:
            return None
        phase_index = next(
            (
                index
                for index, phase in enumerate(progress_spec.phases)
                if phase.id == source.phase_id
            ),
            None,
        )
        if phase_index is None:
            return None
        segment_id = _snapshot_segment_id(snapshot, child)
        if segment_id is None:
            return None
        return ResolvedPreviewSource(
            segment_id=segment_id,
            node_id=source.node_id,
            phase_id=source.phase_id,
            phase_index=phase_index,
            priority=source.priority,
            supersedes=source.supersedes,
            group_index=_child_group_index(child),
            persisted=True,
        )

    segment_id = _legacy_sampler_segment_for_node(child, node_id)
    if segment_id is None:
        return None
    return ResolvedPreviewSource(
        segment_id=segment_id,
        node_id=node_id,
        phase_id="legacy",
        phase_index=None,
        priority=0,
        supersedes=(),
        group_index=_child_group_index(child),
        persisted=False,
    )


def sampler_segment_for_node(
    child: Mapping[str, Any], node_id: str
) -> str | None:
    """Resolve an explicitly published source, or a sampler for legacy rows."""

    source = preview_source_for_node(child, node_id)
    return source.segment_id if source is not None else None


def _linked_node_ids(value: Any, prompt: Mapping[str, Any]) -> set[str]:
    """Collect graph edges from one API-prompt input value."""

    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and not isinstance(value[0], bool)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
        and str(value[0]) in prompt
    ):
        return {str(value[0])}
    if isinstance(value, Mapping):
        linked: set[str] = set()
        for nested in value.values():
            linked.update(_linked_node_ids(nested, prompt))
        return linked
    if isinstance(value, list):
        linked = set()
        for nested in value:
            linked.update(_linked_node_ids(nested, prompt))
        return linked
    return set()


def _output_ancestors(prompt: Mapping[str, Any], output_node_id: str) -> set[str]:
    ancestors: set[str] = set()
    pending = [output_node_id]
    while pending:
        node_id = pending.pop()
        if node_id in ancestors:
            continue
        node = prompt.get(node_id)
        if not isinstance(node, Mapping):
            continue
        ancestors.add(node_id)
        inputs = node.get("inputs")
        if isinstance(inputs, Mapping):
            for value in inputs.values():
                pending.extend(_linked_node_ids(value, prompt) - ancestors)
    return ancestors


def _node_segment_owners(
    child: Mapping[str, Any], node_id: str
) -> tuple[int, ...]:
    """Resolve the segment output branches that depend on a prompt node."""

    prompt = child.get("prompt_snapshot")
    segment_ids = child.get("segment_ids")
    output_nodes = child.get("output_nodes")
    if (
        not isinstance(prompt, Mapping)
        or node_id not in prompt
        or not isinstance(segment_ids, list)
        or not segment_ids
    ):
        return ()
    if len(segment_ids) == 1:
        return (0,)
    if not isinstance(output_nodes, Mapping):
        return ()
    owners: list[int] = []
    for index, segment_id in enumerate(segment_ids):
        output_node_id = output_nodes.get(segment_id)
        if not isinstance(output_node_id, (str, int)) or isinstance(
            output_node_id, bool
        ):
            continue
        if node_id in _output_ancestors(prompt, str(output_node_id)):
            owners.append(index)
    return tuple(owners)


def _progress_phase_bounds(
    spec: ProgressSpec, node_id: str
) -> tuple[Any, float, float] | None:
    start = 0.0
    for phase in spec.phases:
        end = min(1.0, start + phase.weight)
        if phase.node_id == node_id:
            return phase, start, end
        start = end
    return None


def preview_phase_index_for_event(
    child: Mapping[str, Any],
    event: ComfyProgressEvent | ComfyExecutionEvent,
) -> int | None:
    """Resolve a typed event's ordered phase without enabling legacy guesses."""

    if event.node_id is None:
        return None
    persisted, spec = _persisted_progress_spec(child)
    if not persisted or spec is None:
        return None
    return next(
        (
            index
            for index, phase in enumerate(spec.phases)
            if phase.node_id == event.node_id
        ),
        None,
    )


def durable_preview_phase_watermark(child: Mapping[str, Any]) -> int | None:
    """Recover a conservative preview phase floor from durable child progress.

    Live preview frames and their exact websocket ordering are deliberately
    process-local.  The monotonic child progress is durable, however, so after
    a restart it can safely reject sources from phases whose boundary has
    already been reached.  Equality is treated as the later phase having
    started: dropping one boundary frame is preferable to visibly rewinding a
    recovered preview.
    """

    persisted, spec = _persisted_progress_spec(child)
    if not persisted or spec is None:
        return None
    progress = _finite_number(child.get("progress"))
    if progress is None:
        return None
    bounded_progress = max(0.0, min(1.0, progress))
    phase_start = 0.0
    watermark: int | None = None
    for index, phase in enumerate(spec.phases):
        if phase_start <= bounded_progress + _PROGRESS_BOUNDARY_EPSILON:
            watermark = index
        else:
            break
        phase_start = min(1.0, phase_start + phase.weight)
    return watermark


def _persisted_execution_snapshot(
    child: Mapping[str, Any],
    event: ComfyExecutionEvent,
    spec: ProgressSpec,
) -> ChildProgress | None:
    current_progress = _finite_number(child.get("progress")) or 0.0
    if event.node_id is None:
        return ChildProgress(progress=current_progress, stage="开始执行")
    match = _progress_phase_bounds(spec, event.node_id)
    if match is None:
        return None
    phase, start, end = match
    mapped = end if phase.kind == "milestone" else start
    if mapped < current_progress:
        if (
            start <= _PROGRESS_BOUNDARY_EPSILON
            and current_progress
            <= _RUNNING_LIFECYCLE_PROGRESS_FLOOR + _PROGRESS_BOUNDARY_EPSILON
        ):
            # A queue poll may win the race with the sampler's ``executing``
            # frame and install the synthetic 1% running floor first.  Keep
            # that monotonic number, but let the exact later phase label take
            # effect immediately.  Once real fractional progress exceeds the
            # lifecycle floor, the normal no-rewind rule applies again.
            mapped = current_progress
        else:
            return None
    return ChildProgress(
        progress=max(0.0, min(1.0, mapped)),
        stage=phase.label,
    )


def _display_progress_number(value: float) -> int | float:
    return int(value) if value.is_integer() else round(value, 3)


def _persisted_fractional_snapshot(
    child: Mapping[str, Any],
    event: ComfyProgressEvent,
    spec: ProgressSpec,
) -> ChildProgress | None:
    match = _progress_phase_bounds(spec, event.node_id)
    if match is None:
        return None
    phase, start, end = match
    if phase.kind != "fractional":
        return None
    value = _finite_number(event.value)
    maximum = _finite_number(event.maximum)
    if value is None or maximum is None or maximum <= 0:
        return None
    if event.from_progress_state and maximum <= 1:
        return None
    bounded_value = max(0.0, min(value, maximum))
    fraction = bounded_value / maximum
    mapped = start + (end - start) * fraction
    current_progress = _finite_number(child.get("progress")) or 0.0
    if mapped < current_progress:
        return None
    return ChildProgress(
        progress=max(0.0, min(1.0, mapped)),
        stage=(
            f"{phase.label} · "
            f"{_display_progress_number(bounded_value)}/"
            f"{_display_progress_number(maximum)}"
        ),
    )


def child_execution_snapshot(
    child: Mapping[str, Any], event: ComfyExecutionEvent
) -> ChildProgress | None:
    """Project a standard ComfyUI node lifecycle event onto a child stage.

    Numeric progress for non-sampler nodes is intentionally a coarse phase
    landmark. Their standard API carries no byte/frame totals; the exact,
    operator-useful signal is the node-derived Chinese stage label.
    """

    persisted, progress_spec = _persisted_progress_spec(child)
    if persisted:
        if progress_spec is None:
            return None
        return _persisted_execution_snapshot(child, event, progress_spec)

    segment_ids = child.get("segment_ids")
    if not isinstance(segment_ids, list) or not segment_ids:
        return None
    current_progress = _finite_number(child.get("progress")) or 0.0
    if event.node_id is None:
        return ChildProgress(
            progress=max(current_progress, 0.01), stage="开始执行"
        )
    prompt = child.get("prompt_snapshot")
    if not isinstance(prompt, Mapping):
        return None
    node = prompt.get(event.node_id)
    if not isinstance(node, Mapping):
        return None
    class_type = node.get("class_type")
    if not isinstance(class_type, str) or not class_type:
        return None
    label = _EXECUTION_STAGES.get(class_type, "执行原生节点")
    local_progress = _POSTPROCESS_PROGRESS.get(class_type, _PREPARING_PROGRESS)
    if class_type in _SAMPLER_CLASSES:
        local_progress = _SAMPLING_START

    owners = _node_segment_owners(child, event.node_id)
    count = len(segment_ids)
    if len(owners) == 1:
        index = owners[0]
        return ChildProgress(
            progress=max(
                current_progress,
                max(0.0, min(1.0, (index + local_progress) / count)),
            ),
            stage=f"片段 {index + 1}/{count} · {label}",
        )
    if len(owners) > 1:
        # Shared loaders belong to the whole child rather than one segment.
        return ChildProgress(
            progress=max(
                current_progress,
                max(0.0, min(1.0, local_progress / count)),
            ),
            stage=f"准备执行 · {label}",
        )
    return None


def child_execution_start_snapshot(
    child: Mapping[str, Any], event: ComfyExecutionEvent
) -> ChildProgress | None:
    """Recognize execution of an exact-prompt node outside ``ProgressSpec``.

    A typed progress spec owns every numeric landmark and phase label, so an
    undeclared node must not fall back to class-name guesses.  It can still be
    the first websocket proof that a queued/submitting prompt has begun.  This
    narrow projection advances lifecycle without advancing numeric progress;
    malformed/missing persisted specs and non-exact node ids fail closed.
    """

    if child.get("status") not in {"preparing", "queued"}:
        return None
    persisted, progress_spec = _persisted_progress_spec(child)
    if not persisted or progress_spec is None:
        return None
    present, snapshot = _child_exact_prompt_snapshot(child)
    if not present:
        return None
    exact_prompt = _snapshot_field(snapshot, "exact_prompt")
    if not isinstance(exact_prompt, Mapping):
        return None
    if event.node_id is not None and event.node_id not in exact_prompt:
        return None
    current_progress = _finite_number(child.get("progress")) or 0.0
    return ChildProgress(progress=current_progress, stage="开始执行")


def child_progress_snapshot(
    child: Mapping[str, Any], event: ComfyProgressEvent
) -> ChildProgress | None:
    """Map a sampler node event to its segment within one native child graph."""

    persisted, progress_spec = _persisted_progress_spec(child)
    if persisted:
        if progress_spec is None:
            return None
        return _persisted_fractional_snapshot(child, event, progress_spec)

    prompt = child.get("prompt_snapshot")
    segment_ids = child.get("segment_ids")
    segment_id = sampler_segment_for_node(child, event.node_id)
    if (
        segment_id is None
        or not isinstance(prompt, Mapping)
        or not isinstance(segment_ids, list)
    ):
        return None
    # ComfyUI's aggregate progress state assigns every running node a generic
    # 0/1 lifecycle counter. RayLight's worker-side sampling callback is not
    # bridged to the main process, so presenting that placeholder as
    # "sampling 0/1" would be false precision. A real direct progress event
    # (including a future RayLight bridge) remains eligible even for one step.
    if event.from_progress_state and event.maximum <= 1:
        return None
    sampler_ids = [
        str(node_id)
        for node_id, node in prompt.items()
        if isinstance(node, Mapping) and node.get("class_type") in _SAMPLER_CLASSES
    ]
    index = sampler_ids.index(event.node_id)
    fraction = event.value / event.maximum
    sampling_progress = _SAMPLING_START + (
        _SAMPLING_END - _SAMPLING_START
    ) * fraction
    progress = (index + sampling_progress) / max(1, len(sampler_ids))
    step = int(event.value) if event.value.is_integer() else round(event.value, 3)
    maximum = (
        int(event.maximum)
        if event.maximum.is_integer()
        else round(event.maximum, 3)
    )
    return ChildProgress(
        progress=max(0.0, min(1.0, progress)),
        stage=f"片段 {index + 1}/{len(sampler_ids)} · 采样 {step}/{maximum}",
    )


ProgressSink = Callable[
    [str, ComfyProgressEvent | ComfyExecutionEvent], Awaitable[None]
]
PreviewSink = Callable[[str, ComfyPreviewEvent], Awaitable[None]]
ReconcileSink = Callable[[str, ComfyReconcileHint], Awaitable[None]]


class LivePreviewCache:
    """Bounded process-local latest-frame cache; previews never enter SQLite."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 60.0,
        max_total_bytes: int = 16 * 1024 * 1024,
        max_phase_watermarks: int = MAX_PREVIEW_PHASE_WATERMARKS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            ttl_seconds <= 0
            or max_total_bytes < MAX_PREVIEW_MESSAGE_BYTES
            or not isinstance(max_phase_watermarks, int)
            or isinstance(max_phase_watermarks, bool)
            or max_phase_watermarks <= 0
        ):
            raise ValueError("live preview cache limits are invalid")
        self.ttl_seconds = ttl_seconds
        self.max_total_bytes = max_total_bytes
        self.max_phase_watermarks = max_phase_watermarks
        self._clock = clock
        self._entries: OrderedDict[str, LivePreview] = OrderedDict()
        self._discarded: OrderedDict[str, None] = OrderedDict()
        self._phase_watermarks: OrderedDict[tuple[str, str, str], int] = (
            OrderedDict()
        )
        self._total_bytes = 0

    def _purge_expired(self, now: float) -> None:
        expired = [
            job_id
            for job_id, preview in self._entries.items()
            if now - preview.stored_at > self.ttl_seconds
        ]
        for job_id in expired:
            self._remove_entry(job_id)

    def _remove_entry(self, job_id: str) -> None:
        preview = self._entries.pop(job_id, None)
        if preview is not None:
            self._total_bytes -= len(preview.content)

    def _remove_job_watermarks(self, job_id: str) -> None:
        for key in tuple(self._phase_watermarks):
            if key[0] == job_id:
                self._phase_watermarks.pop(key, None)

    def advance_phase(
        self,
        *,
        job_id: str,
        child_id: str,
        prompt_id: str,
        phase_index: int,
    ) -> bool:
        """Advance one exact prompt's phase-start watermark monotonically."""

        if (
            job_id in self._discarded
            or not job_id
            or not child_id
            or not prompt_id
            or not isinstance(phase_index, int)
            or isinstance(phase_index, bool)
            or phase_index < 0
        ):
            return False
        key = (job_id, child_id, prompt_id)
        current = self._phase_watermarks.get(key)
        if current is None or phase_index > current:
            self._phase_watermarks[key] = phase_index
        self._phase_watermarks.move_to_end(key)
        while len(self._phase_watermarks) > self.max_phase_watermarks:
            self._phase_watermarks.popitem(last=False)
        return current is None or phase_index > current

    def _phase_watermark(
        self, *, job_id: str, child_id: str, prompt_id: str
    ) -> int | None:
        return self._phase_watermarks.get((job_id, child_id, prompt_id))

    @staticmethod
    def _may_replace(
        current: LivePreview,
        *,
        child_id: str,
        event: ComfyPreviewEvent,
        source: ResolvedPreviewSource | None,
    ) -> bool:
        if source is None:
            # A legacy frame may refresh another legacy frame, but cannot
            # overwrite a source selected by persisted PreviewSpec authority.
            return current.source_priority is None
        if current.source_priority is None:
            return True
        if (
            current.source_group_index is not None
            and source.group_index is not None
            and source.group_index != current.source_group_index
        ):
            return source.group_index > current.source_group_index
        if current.child_id != child_id:
            return True
        if current.node_id == event.node_id:
            return True
        if (
            current.source_phase_index is not None
            and source.phase_index is not None
            and source.phase_index != current.source_phase_index
        ):
            return source.phase_index > current.source_phase_index
        if current.node_id in source.supersedes:
            return True
        if event.node_id in current.source_supersedes:
            return False
        return source.priority > current.source_priority

    def put(
        self,
        *,
        job_id: str,
        child_id: str,
        segment_id: str,
        event: ComfyPreviewEvent,
        source: ResolvedPreviewSource | None = None,
        minimum_phase_index: int | None = None,
    ) -> bool:
        if job_id in self._discarded or len(event.content) > MAX_PREVIEW_MESSAGE_BYTES:
            return False
        if source is not None:
            if source.segment_id != segment_id or source.node_id != event.node_id:
                return False
            if not source.persisted:
                source = None
            elif source.phase_index is None:
                return False
        if source is not None:
            if minimum_phase_index is not None:
                if (
                    not isinstance(minimum_phase_index, int)
                    or isinstance(minimum_phase_index, bool)
                    or minimum_phase_index < 0
                ):
                    return False
                # The durable floor is a conservative restart fallback. While
                # this process has exact event ordering, retain that more
                # precise watermark so a milestone source can still publish
                # before the next phase actually starts.
                if self._phase_watermark(
                    job_id=job_id,
                    child_id=child_id,
                    prompt_id=event.prompt_id,
                ) is None:
                    self.advance_phase(
                        job_id=job_id,
                        child_id=child_id,
                        prompt_id=event.prompt_id,
                        phase_index=minimum_phase_index,
                    )
            # A preview frame itself proves that its declared phase has begun,
            # even if its executing event was missed during websocket startup.
            self.advance_phase(
                job_id=job_id,
                child_id=child_id,
                prompt_id=event.prompt_id,
                phase_index=source.phase_index,
            )
            watermark = self._phase_watermark(
                job_id=job_id,
                child_id=child_id,
                prompt_id=event.prompt_id,
            )
            if watermark is not None and source.phase_index < watermark:
                return False
        now = self._clock()
        self._purge_expired(now)
        current = self._entries.get(job_id)
        if current is not None and not self._may_replace(
            current,
            child_id=child_id,
            event=event,
            source=source,
        ):
            return False
        self._remove_entry(job_id)
        preview = LivePreview(
            job_id=job_id,
            child_id=child_id,
            segment_id=segment_id,
            prompt_id=event.prompt_id,
            node_id=event.node_id,
            mime_type=event.mime_type,
            content=event.content,
            stored_at=now,
            source_phase_id=source.phase_id if source is not None else None,
            source_phase_index=(
                source.phase_index if source is not None else None
            ),
            source_priority=source.priority if source is not None else None,
            source_supersedes=source.supersedes if source is not None else (),
            source_group_index=(
                source.group_index if source is not None else None
            ),
        )
        self._entries[job_id] = preview
        self._total_bytes += len(event.content)
        while self._total_bytes > self.max_total_bytes and self._entries:
            oldest = next(iter(self._entries))
            self._remove_entry(oldest)
        return job_id in self._entries

    def get(self, job_id: str) -> LivePreview | None:
        now = self._clock()
        self._purge_expired(now)
        preview = self._entries.get(job_id)
        if preview is not None:
            self._entries.move_to_end(job_id)
        return preview

    def discard(self, job_id: str) -> None:
        """Remove a job and reject a late in-flight frame after deletion."""

        self._remove_entry(job_id)
        self._remove_job_watermarks(job_id)
        self._discarded[job_id] = None
        self._discarded.move_to_end(job_id)
        while len(self._discarded) > 4096:
            self._discarded.popitem(last=False)

    def evict(self, job_id: str) -> None:
        """Remove a stale frame without tombstoning a still-live parent job.

        A parent can outlive the child whose sampler produced its latest frame
        (for example while another family child is still queued).  In that
        case a later valid child preview must remain eligible for the cache.
        Tombstones are reserved for deletion, where every late frame must lose.
        """

        self._remove_entry(job_id)

    def clear(self) -> None:
        self._entries.clear()
        self._discarded.clear()
        self._phase_watermarks.clear()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        self._purge_expired(self._clock())
        return self._total_bytes

    @property
    def phase_watermark_count(self) -> int:
        return len(self._phase_watermarks)


class NativeProgressMonitor:
    """One reconnecting standard ComfyUI websocket listener."""

    def __init__(
        self,
        base_url: str,
        client_id: str,
        sink: ProgressSink,
        preview_sink: PreviewSink | None = None,
        reconcile_sink: ReconcileSink | None = None,
        ready_event: asyncio.Event | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.sink = sink
        self.preview_sink = preview_sink
        self.reconcile_sink = reconcile_sink
        self.ready_event = ready_event
        self.ssl_context = ssl_context

    async def run(self) -> None:
        delay = 0.5
        url = websocket_url(self.base_url, self.client_id)
        while True:
            try:
                connect_kwargs: dict[str, Any] = {
                    "proxy": None,
                    "open_timeout": 10,
                    "ping_interval": 20,
                    "ping_timeout": 20,
                    "max_size": 2 * 1024 * 1024,
                }
                # Omitting ``ssl`` lets websockets construct its normal client
                # context for public-CA wss:// URLs. Explicit ssl=None is
                # rejected for wss://, while a context is invalid for ws://.
                if self.ssl_context is not None:
                    connect_kwargs["ssl"] = self.ssl_context
                async with websockets.connect(
                    url,
                    **connect_kwargs,
                ) as socket:
                    # Ask ComfyUI for metadata-bearing previews. Only bounded
                    # event-4 PNG/JPEG frames from registered sampler nodes are
                    # accepted by the preview sink; JSON progress is durable.
                    await socket.send(
                        json.dumps(
                            {
                                "type": "feature_flags",
                                "data": {"supports_preview_metadata": True},
                            }
                        )
                    )
                    if self.ready_event is not None:
                        self.ready_event.set()
                    delay = 0.5
                    try:
                        async for message in socket:
                            if isinstance(message, bytes):
                                event = parse_preview_message(message)
                                if event is not None and self.preview_sink is not None:
                                    await self.preview_sink(self.base_url, event)
                                continue
                            reconcile_hint = parse_reconcile_message(message)
                            if (
                                reconcile_hint is not None
                                and self.reconcile_sink is not None
                            ):
                                await self.reconcile_sink(
                                    self.base_url, reconcile_hint
                                )
                            execution_event = parse_execution_message(message)
                            if execution_event is not None:
                                await self.sink(self.base_url, execution_event)
                            for event in parse_progress_message(message):
                                await self.sink(self.base_url, event)
                    finally:
                        if self.ready_event is not None:
                            self.ready_event.clear()
            except asyncio.CancelledError:
                if self.ready_event is not None:
                    self.ready_event.clear()
                raise
            except Exception:
                if self.ready_event is not None:
                    self.ready_event.clear()
                # Queue/history polling remains authoritative while the socket
                # is unavailable. Reconnect with a bounded exponential delay.
                await asyncio.sleep(delay)
                delay = min(15.0, delay * 2)


class NativeProgressManager:
    """Own at most one listener for each ComfyUI origin/client-id pair."""

    def __init__(
        self,
        sink: ProgressSink,
        preview_sink: PreviewSink | None = None,
        reconcile_sink: ReconcileSink | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._sink = sink
        self._preview_sink = preview_sink
        self._reconcile_sink = reconcile_sink
        self._ssl_context = ssl_context
        self._tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._ready: dict[tuple[str, str], asyncio.Event] = {}
        self._closed = False

    def ensure(self, base_url: str, client_id: str) -> None:
        if self._closed or not base_url.strip() or not client_id.strip():
            return
        key = (base_url.rstrip("/"), client_id)
        task = self._tasks.get(key)
        if task is not None and not task.done():
            return
        ready_event = self._ready.setdefault(key, asyncio.Event())
        monitor = NativeProgressMonitor(
            key[0],
            key[1],
            self._sink,
            self._preview_sink,
            self._reconcile_sink,
            ready_event,
            self._ssl_context,
        )
        self._tasks[key] = asyncio.create_task(
            monitor.run(), name=f"comfy-progress:{key[0]}"
        )

    async def ensure_ready(
        self,
        base_url: str,
        client_id: str,
        *,
        timeout_seconds: float = 1.0,
    ) -> bool:
        """Start a listener and wait briefly for its first connected state.

        Missing the optional websocket must never block durable queue/history
        execution indefinitely, hence the bounded boolean result. Waiting
        before POST /prompt closes the usual first-node race on a newly saved
        endpoint without changing lifecycle authority.
        """

        self.ensure(base_url, client_id)
        key = (base_url.rstrip("/"), client_id)
        ready_event = self._ready.get(key)
        if ready_event is None:
            return False
        try:
            await asyncio.wait_for(ready_event.wait(), timeout_seconds)
        except asyncio.TimeoutError:
            return False
        return True

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._tasks.values())
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._ready.clear()
