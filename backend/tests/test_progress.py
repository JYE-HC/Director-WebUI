from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import replace

import pytest

from directordeck.progress import (
    MAX_PREVIEW_MESSAGE_BYTES,
    ComfyExecutionEvent,
    ComfyPreviewEvent,
    ComfyProgressEvent,
    ComfyReconcileHint,
    LivePreviewCache,
    NativeProgressManager,
    ResolvedPreviewSource,
    child_execution_start_snapshot,
    child_execution_snapshot,
    child_progress_snapshot,
    durable_preview_phase_watermark,
    parse_execution_message,
    parse_preview_message,
    parse_progress_message,
    parse_reconcile_message,
    preview_phase_index_for_event,
    preview_source_for_node,
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


def persisted_specs_child(*, nested: bool = False) -> dict:
    progress_spec = {
        "version": 1,
        "phases": [
            {
                "id": "prepare",
                "label": "准备模型",
                "node_id": "load",
                "kind": "milestone",
                "weight": 0.1,
            },
            {
                "id": "sample_early",
                "label": "第一次采样",
                "node_id": "sample-a",
                "kind": "fractional",
                "weight": 0.3,
            },
            {
                "id": "upscale",
                "label": "超分放大",
                "node_id": "upscale",
                "kind": "milestone",
                "weight": 0.2,
            },
            {
                "id": "sample_late",
                "label": "第二次采样",
                "node_id": "sample-b",
                "kind": "fractional",
                "weight": 0.3,
            },
            {
                "id": "persist",
                "label": "写入成片",
                "node_id": "save",
                "kind": "milestone",
                "weight": 0.1,
            },
        ],
    }
    preview_spec = {
        "version": 1,
        "sources": [
            {
                "node_id": "sample-a",
                "phase_id": "sample_early",
                "publish": True,
                "priority": 10,
                "supersedes": [],
            },
            {
                "node_id": "upscale",
                "phase_id": "upscale",
                "publish": False,
                "priority": 50,
                "supersedes": [],
            },
            {
                "node_id": "sample-b",
                "phase_id": "sample_late",
                "publish": True,
                "priority": 20,
                "supersedes": ["sample-a"],
            },
            {
                "node_id": "save",
                "phase_id": "persist",
                "publish": True,
                "priority": 30,
                "supersedes": [],
            },
        ],
    }
    snapshot = {
        "owner_segment_id": "shot-a",
        "expected_output_spec": {"segment_id": "shot-a"},
        "exact_prompt": {
            "load": {"class_type": "UNETLoader", "inputs": {}},
            "sample-a": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
            "upscale": {"class_type": "UpscaleModel", "inputs": {}},
            "sample-b": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
            "save": {"class_type": "SaveVideo", "inputs": {}},
            "legacy-sampler": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {},
            },
        },
        "progress_spec": progress_spec,
        "preview_spec": preview_spec,
    }
    child = {
        "segment_ids": ["shot-a"],
        "group_index": 3,
        "progress": 0.0,
        # A valid Stage-4 snapshot must prevent these legacy class guesses from
        # becoming a second, conflicting authority.
        "prompt_snapshot": snapshot["exact_prompt"],
    }
    if nested:
        child["execution_evidence"] = {"exact_prompt_snapshot": snapshot}
    else:
        child["exact_prompt_snapshot"] = snapshot
    return child


def test_websocket_url_preserves_reverse_proxy_prefix() -> None:
    assert websocket_url("https://example.test/comfy/", "client/a") == (
        "wss://example.test/comfy/ws?clientId=client%2Fa"
    )


@pytest.mark.parametrize(
    "base_url",
    ("http://comfy.test:8188", "https://comfy.test:8188"),
)
async def test_progress_manager_can_bounded_wait_for_direct_websocket_handshake(
    monkeypatch,
    base_url: str,
) -> None:
    release = asyncio.Event()
    sent: list[str] = []
    connect_calls: list[tuple[str, dict[str, object]]] = []

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

    def connect(url: str, **kwargs: object) -> FakeSocket:
        connect_calls.append((url, kwargs))
        return FakeSocket()

    monkeypatch.setattr("directordeck.progress.websockets.connect", connect)

    async def sink(_origin, _event) -> None:
        return None

    manager = NativeProgressManager(sink)
    try:
        assert await manager.ensure_ready(
            base_url, "director", timeout_seconds=0.5
        )
        assert sent and '"supports_preview_metadata": true' in sent[0]
        assert len(connect_calls) == 1
        assert connect_calls[0][1]["proxy"] is None
        assert "ssl" not in connect_calls[0][1]
    finally:
        release.set()
        await manager.close()


async def test_progress_manager_passes_explicit_tls_context_without_proxy(
    monkeypatch,
) -> None:
    release = asyncio.Event()
    connect_kwargs: dict[str, object] = {}
    tls_context = object()

    class FakeSocket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send(self, _message: str) -> None:
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            await release.wait()
            raise StopAsyncIteration

    def connect(_url: str, **kwargs: object) -> FakeSocket:
        connect_kwargs.update(kwargs)
        return FakeSocket()

    monkeypatch.setattr("directordeck.progress.websockets.connect", connect)

    async def sink(_origin, _event) -> None:
        return None

    manager = NativeProgressManager(sink, ssl_context=tls_context)  # type: ignore[arg-type]
    try:
        assert await manager.ensure_ready(
            "https://comfy.test:8188", "director", timeout_seconds=0.5
        )
        assert connect_kwargs["proxy"] is None
        assert connect_kwargs["ssl"] is tls_context
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
        "directordeck.progress.websockets.connect",
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
        "directordeck.progress.websockets.connect", lambda *_args, **_kwargs: FakeSocket()
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
            "16": {"class_type": "DirectorDeckRayXFuserSamplerCustomAdvanced", "inputs": {}},
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


@pytest.mark.parametrize("nested", [False, True])
def test_persisted_progress_spec_maps_ordered_weighted_phases_monotonically(
    nested: bool,
) -> None:
    child = persisted_specs_child(nested=nested)

    prepare = child_execution_snapshot(
        child,
        ComfyExecutionEvent(prompt_id="p-stage4", node_id="load"),
    )
    assert prepare is not None
    assert prepare.progress == pytest.approx(0.1)
    assert prepare.stage == "准备模型"

    child["progress"] = prepare.progress
    early = child_progress_snapshot(
        child,
        ComfyProgressEvent(
            prompt_id="p-stage4",
            node_id="sample-a",
            value=50.0,
            maximum=100.0,
        ),
    )
    assert early is not None
    assert early.progress == pytest.approx(0.25)
    assert early.stage == "第一次采样 · 50/100"

    child["progress"] = early.progress
    upscale = child_execution_snapshot(
        child,
        ComfyExecutionEvent(prompt_id="p-stage4", node_id="upscale"),
    )
    assert upscale is not None
    assert upscale.progress == pytest.approx(0.6)
    assert upscale.stage == "超分放大"

    child["progress"] = upscale.progress
    late = child_progress_snapshot(
        child,
        ComfyProgressEvent(
            prompt_id="p-stage4",
            node_id="sample-b",
            value=50.0,
            maximum=100.0,
        ),
    )
    assert late is not None
    assert late.progress == pytest.approx(0.75)
    assert late.stage == "第二次采样 · 50/100"

    child["progress"] = late.progress
    assert child_progress_snapshot(
        child,
        ComfyProgressEvent(
            prompt_id="p-stage4",
            node_id="sample-a",
            value=100.0,
            maximum=100.0,
        ),
    ) is None
    assert child_progress_snapshot(
        child,
        ComfyProgressEvent(
            prompt_id="p-stage4",
            node_id="legacy-sampler",
            value=100.0,
            maximum=100.0,
        ),
    ) is None
    assert child_execution_snapshot(
        child,
        ComfyExecutionEvent(prompt_id="p-stage4", node_id="legacy-sampler"),
    ) is None

    persisted = child_execution_snapshot(
        child,
        ComfyExecutionEvent(prompt_id="p-stage4", node_id="save"),
    )
    assert persisted is not None
    assert persisted.progress == pytest.approx(1.0)
    assert persisted.stage == "写入成片"


def test_persisted_stage_only_phase_updates_label_without_inventing_progress() -> None:
    child = persisted_specs_child()
    child["exact_prompt_snapshot"]["progress_spec"]["phases"].insert(
        0,
        {
            "id": "inspect",
            "label": "读取并编码输入",
            "node_id": "legacy-sampler",
            "kind": "stage",
            "weight": 0.0,
        },
    )

    snapshot = child_execution_snapshot(
        child,
        ComfyExecutionEvent(
            prompt_id="p-stage4",
            node_id="legacy-sampler",
        ),
    )

    assert snapshot is not None
    assert snapshot.progress == 0.0
    assert snapshot.stage == "读取并编码输入"


def test_persisted_missing_or_invalid_progress_spec_never_uses_legacy_fallback() -> None:
    child = persisted_specs_child()
    snapshot = child["exact_prompt_snapshot"]
    snapshot["progress_spec"] = None
    event = ComfyProgressEvent(
        prompt_id="p-stage4",
        node_id="sample-a",
        value=5.0,
        maximum=20.0,
    )
    assert child_progress_snapshot(child, event) is None
    assert child_execution_snapshot(
        child,
        ComfyExecutionEvent(prompt_id="p-stage4", node_id="sample-a"),
    ) is None

    snapshot["progress_spec"] = {
        "version": 1,
        "phases": [
            {
                "id": "broken",
                "label": "Broken",
                "node_id": "sample-a",
                "kind": "fractional",
                "weight": 0.5,
            }
        ],
    }
    assert child_progress_snapshot(child, event) is None


def test_invalid_execution_evidence_marker_disables_progress_and_preview_fallback() -> None:
    child = persisted_specs_child()
    snapshot = child.pop("exact_prompt_snapshot")
    child["execution_evidence"] = {"invalid": True}
    # These mutable fields would fully authorize the old fallback if the
    # typed marker were mistaken for evidence absence.
    child["prompt_snapshot"] = snapshot["exact_prompt"]
    child["output_nodes"] = {"shot-a": "save"}

    assert child_progress_snapshot(
        child,
        ComfyProgressEvent(
            prompt_id="p-stage4",
            node_id="legacy-sampler",
            value=50.0,
            maximum=100.0,
        ),
    ) is None
    assert child_execution_snapshot(
        child,
        ComfyExecutionEvent(
            prompt_id="p-stage4",
            node_id="legacy-sampler",
        ),
    ) is None
    assert preview_source_for_node(child, "legacy-sampler") is None


def test_exact_unlisted_node_can_start_lifecycle_without_progress_fallback() -> None:
    child = persisted_specs_child()
    child.update(status="preparing", stage="submitting", progress=0.0)
    event = ComfyExecutionEvent(
        prompt_id="p-stage4", node_id="legacy-sampler"
    )

    # The node is in the immutable exact prompt but intentionally absent from
    # ProgressSpec. It proves execution began without inheriting the legacy
    # sampler's numeric landmark or label.
    assert child_execution_snapshot(child, event) is None
    started = child_execution_start_snapshot(child, event)
    assert started is not None
    assert started.progress == 0.0
    assert started.stage == "开始执行"

    unknown = ComfyExecutionEvent(prompt_id="p-stage4", node_id="missing")
    assert child_execution_start_snapshot(child, unknown) is None

    child["exact_prompt_snapshot"]["progress_spec"] = None
    assert child_execution_start_snapshot(child, event) is None


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
            "1": {"class_type": "DirectorDeckRayInitializerAdvanced", "inputs": {}},
            "2": {
                "class_type": "DirectorDeckRayUNETLoader",
                "inputs": {"ray_actors_init": ["1", 0]},
            },
            "3": {
                "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {},
            },
            "4": {
                "class_type": "DirectorDeckRayXFuserSamplerCustomAdvanced",
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
            "16": {"class_type": "DirectorDeckRayXFuserSamplerCustomAdvanced", "inputs": {}},
        },
    }
    assert sampler_segment_for_node(child, "8") == "shot-a"
    assert sampler_segment_for_node(child, "16") == "shot-b"
    assert sampler_segment_for_node(child, "missing") is None


def test_persisted_preview_spec_filters_sources_and_rejects_late_old_phase() -> None:
    child = persisted_specs_child()
    early = preview_source_for_node(child, "sample-a")
    late = preview_source_for_node(child, "sample-b")
    highest = preview_source_for_node(child, "save")

    assert isinstance(early, ResolvedPreviewSource)
    assert isinstance(late, ResolvedPreviewSource)
    assert isinstance(highest, ResolvedPreviewSource)
    assert early.persisted and (early.phase_index, early.priority) == (1, 10)
    assert (late.phase_index, late.priority) == (3, 20)
    assert late.supersedes == ("sample-a",)
    assert (highest.phase_index, highest.priority) == (4, 30)
    assert preview_source_for_node(child, "upscale") is None
    assert preview_source_for_node(child, "legacy-sampler") is None
    assert sampler_segment_for_node(child, "sample-a") == "shot-a"
    assert sampler_segment_for_node(child, "upscale") is None

    cache = LivePreviewCache(max_total_bytes=MAX_PREVIEW_MESSAGE_BYTES)

    def event(node_id: str, content: bytes) -> ComfyPreviewEvent:
        return ComfyPreviewEvent(
            prompt_id="p-stage4",
            node_id=node_id,
            mime_type="image/png",
            content=content,
        )

    assert cache.put(
        job_id="job-stage4",
        child_id="child-stage4",
        segment_id="shot-a",
        event=event("sample-a", b"early"),
        source=early,
    )
    # A non-publishing intermediate phase is still a phase-start watermark.
    # Its execution must close the preceding sampler before the next sampler
    # has emitted a frame.
    assert cache.advance_phase(
        job_id="job-stage4",
        child_id="child-stage4",
        prompt_id="p-stage4",
        phase_index=2,
    )
    assert not cache.put(
        job_id="job-stage4",
        child_id="child-stage4",
        segment_id="shot-a",
        event=event("sample-a", b"delayed-before-next-preview"),
        source=early,
    )
    assert cache.get("job-stage4").content == b"early"  # type: ignore[union-attr]
    assert cache.put(
        job_id="job-stage4",
        child_id="child-stage4",
        segment_id="shot-a",
        event=event("sample-b", b"late"),
        source=late,
    )
    assert not cache.put(
        job_id="job-stage4",
        child_id="child-stage4",
        segment_id="shot-a",
        event=event("sample-a", b"delayed-old"),
        source=early,
    )
    assert cache.get("job-stage4").content == b"late"  # type: ignore[union-attr]

    # A higher-priority source need not duplicate every transitive supersedes
    # edge. Once selected, the lower-priority late frame cannot overwrite it.
    assert cache.put(
        job_id="job-stage4",
        child_id="child-stage4",
        segment_id="shot-a",
        event=event("save", b"highest"),
        source=highest,
    )
    assert not cache.put(
        job_id="job-stage4",
        child_id="child-stage4",
        segment_id="shot-a",
        event=event("sample-b", b"delayed-late"),
        source=late,
    )
    stored = cache.get("job-stage4")
    assert stored is not None
    assert stored.node_id == "save"
    assert stored.content == b"highest"
    assert stored.source_phase_id == "persist"
    assert stored.source_phase_index == 4
    assert stored.source_priority == 30


def test_phase_watermark_recovers_scopes_and_remains_bounded() -> None:
    child = persisted_specs_child()
    early = preview_source_for_node(child, "sample-a")
    late = preview_source_for_node(child, "sample-b")
    assert isinstance(early, ResolvedPreviewSource)
    assert isinstance(late, ResolvedPreviewSource)

    assert preview_phase_index_for_event(
        child,
        ComfyExecutionEvent(prompt_id="p-stage4", node_id="upscale"),
    ) == 2
    assert preview_phase_index_for_event(
        child,
        ComfyProgressEvent(
            prompt_id="p-stage4",
            node_id="sample-b",
            value=1.0,
            maximum=10.0,
        ),
    ) == 3
    assert preview_phase_index_for_event(
        child,
        ComfyExecutionEvent(prompt_id="p-stage4", node_id="legacy-sampler"),
    ) is None

    # At the exact 0.6 boundary the durable projection conservatively treats
    # sample_late as started. A fresh process-local cache therefore cannot
    # visibly rewind to sample_early after a restart.
    child["progress"] = 0.6
    recovered = durable_preview_phase_watermark(child)
    assert recovered == 3
    cache = LivePreviewCache(max_phase_watermarks=2)

    def event(prompt_id: str, node_id: str, content: bytes) -> ComfyPreviewEvent:
        return ComfyPreviewEvent(
            prompt_id=prompt_id,
            node_id=node_id,
            mime_type="image/png",
            content=content,
        )

    assert not cache.put(
        job_id="job-recovered",
        child_id="child-old",
        segment_id="shot-a",
        event=event("p-stage4", early.node_id, b"late-after-restart"),
        source=early,
        minimum_phase_index=recovered,
    )
    assert cache.get("job-recovered") is None
    assert cache.put(
        job_id="job-recovered",
        child_id="child-old",
        segment_id="shot-a",
        event=event("p-stage4", late.node_id, b"current"),
        source=late,
        minimum_phase_index=recovered,
    )

    # Phase order is authoritative even if a later source has a numerically
    # lower priority and omits an explicit supersedes edge.
    phase_order_cache = LivePreviewCache()
    assert phase_order_cache.put(
        job_id="job-phase-order",
        child_id="child-phase-order",
        segment_id="shot-a",
        event=event("p-order", early.node_id, b"early"),
        source=early,
    )
    assert phase_order_cache.put(
        job_id="job-phase-order",
        child_id="child-phase-order",
        segment_id="shot-a",
        event=event("p-order", late.node_id, b"later-low-priority"),
        source=replace(late, priority=-10, supersedes=()),
    )

    # Watermarks are monotonic and scoped to an exact child/prompt pair. A
    # later child can begin at its own first phase under the same parent.
    assert not cache.advance_phase(
        job_id="job-recovered",
        child_id="child-old",
        prompt_id="p-stage4",
        phase_index=1,
    )
    assert cache.advance_phase(
        job_id="job-recovered",
        child_id="child-next",
        prompt_id="p-next",
        phase_index=0,
    )
    assert cache.put(
        job_id="job-recovered",
        child_id="child-next",
        segment_id="shot-a",
        event=event("p-next", early.node_id, b"next-child"),
        source=early,
    )
    assert cache.get("job-recovered").content == b"next-child"  # type: ignore[union-attr]
    assert cache.phase_watermark_count == 2
    assert cache.advance_phase(
        job_id="job-third",
        child_id="child-third",
        prompt_id="p-third",
        phase_index=0,
    )
    assert cache.phase_watermark_count == 2

    cache.discard("job-recovered")
    assert cache.phase_watermark_count == 1
    assert not cache.advance_phase(
        job_id="job-recovered",
        child_id="child-next",
        prompt_id="p-next",
        phase_index=1,
    )
    cache.clear()
    assert cache.phase_watermark_count == 0


def test_legacy_and_malformed_typed_rows_do_not_enter_phase_watermark_policy() -> None:
    legacy = {
        "segment_ids": ["shot-a"],
        "prompt_snapshot": {
            "sampler": {"class_type": "SamplerCustomAdvanced", "inputs": {}},
        },
    }
    legacy_event = ComfyExecutionEvent(prompt_id="legacy-prompt", node_id="sampler")
    assert preview_phase_index_for_event(legacy, legacy_event) is None
    assert durable_preview_phase_watermark(legacy) is None

    source = preview_source_for_node(legacy, "sampler")
    assert isinstance(source, ResolvedPreviewSource)
    assert source.persisted is False and source.phase_index is None
    cache = LivePreviewCache()
    first = ComfyPreviewEvent(
        prompt_id="legacy-prompt",
        node_id="sampler",
        mime_type="image/png",
        content=b"first",
    )
    refreshed = ComfyPreviewEvent(
        prompt_id="legacy-prompt",
        node_id="sampler",
        mime_type="image/png",
        content=b"refreshed",
    )
    assert cache.put(
        job_id="legacy-job",
        child_id="legacy-child",
        segment_id="shot-a",
        event=first,
        source=source,
    )
    assert cache.put(
        job_id="legacy-job",
        child_id="legacy-child",
        segment_id="shot-a",
        event=refreshed,
        source=source,
    )
    assert cache.get("legacy-job").content == b"refreshed"  # type: ignore[union-attr]

    malformed = persisted_specs_child()
    malformed["exact_prompt_snapshot"]["progress_spec"] = {"broken": True}
    assert preview_phase_index_for_event(malformed, legacy_event) is None
    assert durable_preview_phase_watermark(malformed) is None
    assert preview_source_for_node(malformed, "sample-a") is None


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
