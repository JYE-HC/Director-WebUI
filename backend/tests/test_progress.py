from __future__ import annotations

import asyncio
import json
import struct

from director.progress import (
    MAX_PREVIEW_MESSAGE_BYTES,
    ComfyExecutionEvent,
    ComfyPreviewEvent,
    ComfyProgressEvent,
    ComfyReconcileHint,
    LivePreviewCache,
    NativeProgressManager,
    child_execution_snapshot,
    child_progress_snapshot,
    parse_execution_message,
    parse_preview_message,
    parse_progress_message,
    parse_reconcile_message,
    sampler_segment_for_node,
    websocket_url,
)


def preview_message(
    *,
    event_type: int = 4,
    prompt_id: str = "prompt-1",
    node_id: str = "8",
    mime_type: str = "image/png",
    content: bytes = b"\x89PNG\r\n\x1a\npreview",
) -> bytes:
    metadata = json.dumps(
        {
            "prompt_id": prompt_id,
            "node_id": node_id,
            "image_type": mime_type,
        }
    ).encode()
    return (
        struct.pack(">I", event_type)
        + struct.pack(">I", len(metadata))
        + metadata
        + content
    )


def test_websocket_url_preserves_reverse_proxy_prefix() -> None:
    assert websocket_url("https://example.test/comfy/", "client/a") == (
        "wss://example.test/comfy/ws?clientId=client%2Fa"
    )


async def test_progress_manager_can_bounded_wait_for_websocket_handshake(
    monkeypatch,
) -> None:
    release = asyncio.Event()
    sent: list[str] = []

    class FakeSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, message: str) -> None:
            sent.append(message)

        def __aiter__(self):
            return self

        async def __anext__(self):
            await release.wait()
            raise StopAsyncIteration

    monkeypatch.setattr(
        "director.progress.websockets.connect", lambda *_args, **_kwargs: FakeSocket()
    )

    async def sink(_origin, _event) -> None:
        return None

    manager = NativeProgressManager(sink)
    try:
        assert await manager.ensure_ready(
            "http://comfy.test:8188", "director", timeout_seconds=0.5
        )
        assert sent and '"supports_preview_metadata": true' in sent[0]
    finally:
        release.set()
        await manager.close()


async def test_progress_manager_bounded_wait_times_out_before_handshake(
    monkeypatch,
) -> None:
    release = asyncio.Event()

    class StalledSocket:
        async def __aenter__(self):
            await release.wait()
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        "director.progress.websockets.connect",
        lambda *_args, **_kwargs: StalledSocket(),
    )

    async def sink(_origin, _event) -> None:
        return None

    manager = NativeProgressManager(sink)
    try:
        assert not await manager.ensure_ready(
            "http://comfy.test:8188", "director", timeout_seconds=0.01
        )
    finally:
        release.set()
        await manager.close()


async def test_progress_manager_forwards_reconcile_hints(monkeypatch) -> None:
    release = asyncio.Event()
    received: list[tuple[str, ComfyReconcileHint]] = []
    delivered = asyncio.Event()

    class FakeSocket:
        def __init__(self) -> None:
            self.messages = iter(
                [
                    '{"type":"status","data":{"status":{"exec_info":{"queue_remaining":1}}}}',
                    '{"type":"execution_success","data":{"prompt_id":"prompt-1"}}',
                    '{"type":"executing","data":{"prompt_id":"prompt-1","node":null}}',
                ]
            )

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, _message: str) -> None:
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                await release.wait()
                raise StopAsyncIteration

    monkeypatch.setattr(
        "director.progress.websockets.connect", lambda *_args, **_kwargs: FakeSocket()
    )

    async def progress_sink(_origin, _event) -> None:
        return None

    async def reconcile_sink(origin, event) -> None:
        received.append((origin, event))
        if len(received) == 3:
            delivered.set()

    manager = NativeProgressManager(
        progress_sink, reconcile_sink=reconcile_sink
    )
    try:
        manager.ensure("http://comfy.test:8188", "director")
        await asyncio.wait_for(delivered.wait(), timeout=0.5)
        assert received == [
            (
                "http://comfy.test:8188",
                ComfyReconcileHint(event_type="status", prompt_id=None),
            ),
            (
                "http://comfy.test:8188",
                ComfyReconcileHint(
                    event_type="execution_success", prompt_id="prompt-1"
                ),
            ),
            (
                "http://comfy.test:8188",
                ComfyReconcileHint(event_type="executing", prompt_id="prompt-1"),
            ),
        ]
    finally:
        release.set()
        await manager.close()


def test_parse_standard_progress_and_ignore_binary_preview() -> None:
    events = parse_progress_message(
        '{"type":"progress","data":{"prompt_id":"p1","node":"12",'
        '"value":3,"max":20}}'
    )
    assert events == [
        ComfyProgressEvent(
            prompt_id="p1", node_id="12", value=3.0, maximum=20.0
        )
    ]
    assert parse_progress_message(b"\x00preview") == []
    assert parse_progress_message('{"type":"progress","data":{"max":0}}') == []


def test_parse_standard_execution_events_without_guessing_reconnect_owner() -> None:
    assert parse_execution_message(
        '{"type":"execution_start","data":{"prompt_id":"p1"}}'
    ) == ComfyExecutionEvent(prompt_id="p1", node_id=None)
    assert parse_execution_message(
        '{"type":"executing","data":{"prompt_id":"p1","node":"12"}}'
    ) == ComfyExecutionEvent(prompt_id="p1", node_id="12")

    # Current ComfyUI emits node=None after the prompt finishes. History owns
    # that terminal transition, not the optional websocket.
    assert parse_execution_message(
        '{"type":"executing","data":{"prompt_id":"p1","node":null}}'
    ) is None
    # Reconnect snapshots omit prompt_id and cannot safely be attributed.
    assert parse_execution_message(
        '{"type":"executing","data":{"node":"12"}}'
    ) is None
    assert parse_execution_message(b"binary") is None


def test_parse_reconcile_hints_without_treating_node_execution_as_terminal() -> None:
    assert parse_reconcile_message(
        '{"type":"status","data":{"status":{"exec_info":{"queue_remaining":0}}}}'
    ) == ComfyReconcileHint(event_type="status", prompt_id=None)
    for event_type in (
        "execution_success",
        "execution_error",
        "execution_interrupted",
    ):
        assert parse_reconcile_message(
            json.dumps({"type": event_type, "data": {"prompt_id": "prompt-1"}})
        ) == ComfyReconcileHint(event_type=event_type, prompt_id="prompt-1")
    assert parse_reconcile_message(
        '{"type":"executing","data":{"prompt_id":"prompt-1","node":null}}'
    ) == ComfyReconcileHint(event_type="executing", prompt_id="prompt-1")
    assert (
        parse_reconcile_message(
            '{"type":"executing","data":{"prompt_id":"prompt-1","node":"12"}}'
        )
        is None
    )
    assert parse_reconcile_message(
        '{"type":"execution_success","data":{}}'
    ) is None
    assert parse_reconcile_message(b"binary") is None


def test_parse_only_metadata_bearing_bounded_png_or_jpeg_preview() -> None:
    assert parse_preview_message(preview_message()) == ComfyPreviewEvent(
        prompt_id="prompt-1",
        node_id="8",
        mime_type="image/png",
        content=b"\x89PNG\r\n\x1a\npreview",
    )
    jpeg = parse_preview_message(
        preview_message(
            mime_type="image/jpeg", content=b"\xff\xd8\xffjpeg-preview"
        )
    )
    assert jpeg is not None and jpeg.mime_type == "image/jpeg"

    assert parse_preview_message(preview_message(event_type=1)) is None
    assert parse_preview_message(preview_message(mime_type="image/webp")) is None
    assert parse_preview_message(preview_message(content=b"not-a-png")) is None
    assert parse_preview_message(b"\x00" * (MAX_PREVIEW_MESSAGE_BYTES + 1)) is None


def test_preview_parser_rejects_malformed_metadata_boundaries() -> None:
    message = preview_message()
    assert parse_preview_message(message[:4] + struct.pack(">I", 0) + message[8:]) is None
    assert (
        parse_preview_message(
            message[:4] + struct.pack(">I", 65 * 1024) + message[8:]
        )
        is None
    )
    invalid_json = struct.pack(">II", 4, 1) + b"{" + b"\x89PNG\r\n\x1a\n"
    assert parse_preview_message(invalid_json) is None


def test_parse_running_nodes_from_progress_state() -> None:
    events = parse_progress_message(
        '{"type":"progress_state","data":{"prompt_id":"p2","nodes":{'
        '"8":{"state":"finished","value":20,"max":20},'
        '"16":{"state":"running","value":7,"max":20}}}}'
    )
    assert events == [
        ComfyProgressEvent(
            prompt_id="p2",
            node_id="16",
            value=7.0,
            maximum=20.0,
            from_progress_state=True,
        )
    ]


def test_generic_progress_state_counter_is_not_presented_as_sampler_steps() -> None:
    child = {
        "segment_ids": ["shot-a"],
        "prompt_snapshot": {
            "16": {"class_type": "XFuserSamplerCustomAdvanced", "inputs": {}},
        },
    }
    aggregate = parse_progress_message(
        '{"type":"progress_state","data":{"prompt_id":"p-ray","nodes":{'
        '"16":{"state":"running","value":0,"max":1}}}}'
    )

    assert len(aggregate) == 1
    assert child_progress_snapshot(child, aggregate[0]) is None
    assert child_progress_snapshot(
        child,
        ComfyProgressEvent(
            prompt_id="p-ray", node_id="16", value=1.0, maximum=1.0
        ),
    ) is not None


def test_child_progress_maps_sampler_to_segment_and_step() -> None:
    child = {
        "segment_ids": ["shot-a", "shot-b"],
        "prompt_snapshot": {
            "1": {"class_type": "UNETLoader", "inputs": {}},
            "8": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
            "16": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
        },
    }

    snapshot = child_progress_snapshot(
        child,
        ComfyProgressEvent(
            prompt_id="p3", node_id="16", value=5.0, maximum=20.0
        ),
    )

    assert snapshot is not None
    assert snapshot.progress == 0.6625
    assert snapshot.stage == "片段 2/2 · 采样 5/20"


def test_execution_stage_covers_raylight_load_condition_decode_and_save() -> None:
    child = {
        "segment_ids": ["shot-a"],
        "output_nodes": {"shot-a": "8"},
        "prompt_snapshot": {
            "1": {"class_type": "RayInitializerAdvanced", "inputs": {}},
            "2": {
                "class_type": "RayUNETLoader",
                "inputs": {"ray_actors_init": ["1", 0]},
            },
            "3": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {},
            },
            "4": {
                "class_type": "XFuserSamplerCustomAdvanced",
                "inputs": {"guider": ["2", 0], "latent_image": ["3", 1]},
            },
            "5": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["4", 0]},
            },
            "6": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["5", 0]},
            },
            "8": {
                "class_type": "SaveVideo",
                "inputs": {"video": ["6", 0]},
            },
        },
    }

    expected = {
        None: (0.01, "开始执行"),
        "1": (0.10, "片段 1/1 · 初始化 RayLight 多卡"),
        "2": (0.10, "片段 1/1 · 加载 RayLight 生成模型"),
        "3": (0.10, "片段 1/1 · 构建多模态条件"),
        "4": (0.15, "片段 1/1 · RayLight 采样中"),
        "5": (0.90, "片段 1/1 · 解码视频画面"),
        "6": (0.95, "片段 1/1 · 封装音视频"),
        "8": (0.98, "片段 1/1 · 写入视频文件"),
    }
    for node_id, (progress, stage) in expected.items():
        snapshot = child_execution_snapshot(
            child, ComfyExecutionEvent(prompt_id="p-ray", node_id=node_id)
        )
        assert snapshot is not None
        assert snapshot.progress == progress
        assert snapshot.stage == stage


def test_execution_stage_rejects_unknown_node_and_marks_shared_loader() -> None:
    child = {
        "segment_ids": ["shot-a", "shot-b"],
        "output_nodes": {"shot-a": "4", "shot-b": "5"},
        "prompt_snapshot": {
            "1": {"class_type": "UNETLoader", "inputs": {}},
            "2": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
            "3": {"class_type": "VAEDecode", "inputs": {"samples": ["1", 0]}},
            "4": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0]}},
            "5": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0]}},
        },
    }
    shared = child_execution_snapshot(
        child, ComfyExecutionEvent(prompt_id="p", node_id="1")
    )
    second = child_execution_snapshot(
        child, ComfyExecutionEvent(prompt_id="p", node_id="3")
    )
    unknown = child_execution_snapshot(
        child, ComfyExecutionEvent(prompt_id="p", node_id="missing")
    )

    assert shared is not None
    assert shared.progress == 0.05
    assert shared.stage == "准备执行 · 加载生成模型"
    assert second is not None
    assert second.progress == 0.95
    assert second.stage == "片段 2/2 · 解码视频画面"
    assert unknown is None


def test_child_progress_rejects_unknown_or_ambiguous_sampler_mapping() -> None:
    child = {
        "segment_ids": ["shot-a", "shot-b"],
        "prompt_snapshot": {
            "8": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
        },
    }
    event = ComfyProgressEvent(
        prompt_id="p4", node_id="8", value=1.0, maximum=20.0
    )
    assert child_progress_snapshot(child, event) is None


def test_sampler_node_maps_one_to_one_to_stable_segment_id() -> None:
    child = {
        "segment_ids": ["shot-a", "shot-b"],
        "prompt_snapshot": {
            "8": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
            "16": {"class_type": "XFuserSamplerCustomAdvanced", "inputs": {}},
        },
    }
    assert sampler_segment_for_node(child, "8") == "shot-a"
    assert sampler_segment_for_node(child, "16") == "shot-b"
    assert sampler_segment_for_node(child, "missing") is None


def test_live_preview_cache_expires_evicts_and_tombstones_deleted_job() -> None:
    now = [10.0]
    cache = LivePreviewCache(
        ttl_seconds=5,
        max_total_bytes=MAX_PREVIEW_MESSAGE_BYTES,
        clock=lambda: now[0],
    )
    event = ComfyPreviewEvent(
        prompt_id="p1", node_id="8", mime_type="image/png", content=b"frame"
    )
    assert cache.put(
        job_id="job-1", child_id="child-1", segment_id="shot-1", event=event
    )
    assert cache.get("job-1") is not None
    now[0] = 16.0
    assert cache.get("job-1") is None
    assert cache.total_bytes == 0

    cache.discard("job-2")
    assert not cache.put(
        job_id="job-2", child_id="child-2", segment_id="shot-2", event=event
    )
    cache.clear()
    assert cache.total_bytes == 0


def test_live_preview_cache_enforces_total_lru_bound_without_tombstoning() -> None:
    cache = LivePreviewCache(max_total_bytes=MAX_PREVIEW_MESSAGE_BYTES)
    first = ComfyPreviewEvent(
        prompt_id="p1",
        node_id="8",
        mime_type="image/jpeg",
        content=b"a" * (MAX_PREVIEW_MESSAGE_BYTES // 2 + 1),
    )
    second = ComfyPreviewEvent(
        prompt_id="p2",
        node_id="9",
        mime_type="image/jpeg",
        content=b"b" * (MAX_PREVIEW_MESSAGE_BYTES // 2 + 1),
    )
    assert cache.put(
        job_id="job-1", child_id="child-1", segment_id="shot-1", event=first
    )
    assert cache.put(
        job_id="job-2", child_id="child-2", segment_id="shot-2", event=second
    )
    assert cache.get("job-1") is None
    assert cache.get("job-2") is not None
    assert cache.total_bytes == len(second.content)

    cache.evict("job-2")
    assert cache.put(
        job_id="job-2", child_id="child-2", segment_id="shot-2", event=second
    )
