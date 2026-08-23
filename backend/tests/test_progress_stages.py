from __future__ import annotations

import asyncio

import pytest
from starlette.requests import Request

from directordeck.app import _sync_job
from directordeck.progress import (
    ComfyExecutionEvent,
    ComfyPreviewEvent,
    ComfyProgressEvent,
)
from directordeck.schemas import RuntimeSettings, default_settings

from .conftest import (
    adapt_legacy_workflow_requests,
    runnable_draft,
    save_database_legacy_settings,
)


@pytest.fixture(autouse=True)
def _stage6_v5_request_adapter(client, monkeypatch) -> None:
    adapt_legacy_workflow_requests(client, monkeypatch)


async def test_raylight_native_events_follow_persisted_progress_spec(
    client,
    monkeypatch,
) -> None:
    raw_settings = default_settings().model_dump(mode="json")
    raw_settings["multi_gpu_enabled"] = True
    raw_settings["models"]["fl2va"].update(
        {
            "backend": "raylight",
            "raylight": {
                "gpu_select": [0, 1],
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_degree": 1,
                "dp_degree": 1,
                "fsdp": False,
                "cpu_offload": False,
            },
        }
    )
    save_database_legacy_settings(
        client.director_app.state.database,
        RuntimeSettings.model_validate(raw_settings),
    )
    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )
    assert response.status_code == 200, response.text
    parent = response.json()
    database = client.director_app.state.database
    while client.director_app.state.submission_tasks:
        await asyncio.sleep(0)
    child = database.list_job_children(parent["id"])[0]
    # ComfyUI can start immediately after accepting the prompt, before its
    # HTTP response reaches Director. The first websocket event must still be
    # eligible while this row retains submission ownership.
    child = database.update_job_child(
        child["id"],
        status="preparing",
        progress=0.0,
        stage="submitting",
        outputs=[],
        error=None,
        started_at=None,
        completed_at=None,
    )
    prompt = child["prompt_snapshot"]

    def node_id(class_type: str) -> str:
        return next(
            str(candidate)
            for candidate, node in prompt.items()
            if node["class_type"] == class_type
        )

    sink = client.director_app.state.progress_manager._sink
    origin = "http://comfy.test:8188"
    prompt_id = child["prompt_id"]

    def full_parent_snapshot_is_forbidden(_job_id: str):
        raise AssertionError("native websocket events must use the status projection")

    monkeypatch.setattr(database, "get_job", full_parent_snapshot_is_forbidden)

    stages = [
        (None, 0.0, "开始执行"),
        (
            node_id("DirectorDeckRayInitializerAdvanced"),
            0.0,
            "初始化 RayLight 多卡",
        ),
        (
            node_id("DirectorDeckRayUNETLoader"),
            0.0,
            "加载 RayLight 生成模型",
        ),
    ]
    for current_node, expected_progress, expected_stage in stages:
        await sink(
            origin,
            ComfyExecutionEvent(prompt_id=prompt_id, node_id=current_node),
        )
        stored = database.get_job_child(child["id"])
        assert stored is not None
        assert stored["status"] == "running"
        assert stored["progress"] == pytest.approx(expected_progress)
        assert stored["stage"] == expected_stage

    sampler_id = node_id("DirectorDeckRayXFuserSamplerCustomAdvanced")
    await sink(
        origin,
        ComfyExecutionEvent(prompt_id=prompt_id, node_id=sampler_id),
    )
    ray_sampling = database.get_job_child(child["id"])
    assert ray_sampling is not None
    assert ray_sampling["stage"] == "采样中"
    # The installed RayLight worker normally does not bridge its inner step
    # callback to ComfyUI's main-process websocket. This synthetic standard
    # event verifies compatibility if a future RayLight version does so; the
    # executing-derived stage above is the current guaranteed signal.
    await sink(
        origin,
        ComfyProgressEvent(
            prompt_id=prompt_id,
            node_id=sampler_id,
            value=5.0,
            maximum=25.0,
        ),
    )
    sampling = database.get_job_child(child["id"])
    assert sampling is not None
    assert sampling["progress"] == pytest.approx(0.14)
    assert sampling["stage"] == "采样中 · 5/25"

    preview_sink = client.director_app.state.progress_manager._preview_sink
    assert preview_sink is not None
    initial_preview = b"\x89PNG\r\n\x1a\nstatus-projection-preview"
    await preview_sink(
        origin,
        ComfyPreviewEvent(
            prompt_id=prompt_id,
            node_id=sampler_id,
            mime_type="image/png",
            content=initial_preview,
        ),
    )
    assert client.director_app.state.live_preview_cache.get(parent["id"]) is not None

    for class_type, expected_progress, expected_stage in (
        ("VAEDecode", 0.85, "解码视频画面"),
        ("CreateVideo", 0.95, "封装音视频"),
        ("SaveVideo", 1.0, "写入视频文件"),
    ):
        await sink(
            origin,
            ComfyExecutionEvent(
                prompt_id=prompt_id, node_id=node_id(class_type)
            ),
        )
        stored = database.get_job_child(child["id"])
        assert stored is not None
        assert stored["progress"] == pytest.approx(expected_progress)
        assert stored["stage"] == expected_stage
        if class_type == "VAEDecode":
            # The decode execution event advances the cache watermark before
            # any later preview exists. A delayed sampler frame must not make
            # the live view move back to the sampling phase.
            await preview_sink(
                origin,
                ComfyPreviewEvent(
                    prompt_id=prompt_id,
                    node_id=sampler_id,
                    mime_type="image/png",
                    content=b"\x89PNG\r\n\x1a\nlate-sampler-preview",
                ),
            )
            preview = client.director_app.state.live_preview_cache.get(
                parent["id"]
            )
            assert preview is not None
            assert preview.content == initial_preview

    terminal = database.update_job_child(
        child["id"],
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        completed_at="2026-08-13T00:00:00+00:00",
    )
    await sink(
        origin,
        ComfyExecutionEvent(
            prompt_id=prompt_id,
            node_id=node_id("DirectorDeckRayInitializerAdvanced"),
        ),
    )
    assert database.get_job_child(child["id"]) == terminal


async def test_standard_native_events_publish_explicit_pre_sampling_stages(
    client,
    monkeypatch,
) -> None:
    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )
    assert response.status_code == 200, response.text
    parent = response.json()
    database = client.director_app.state.database
    while client.director_app.state.submission_tasks:
        await asyncio.sleep(0)
    child = database.list_job_children(parent["id"])[0]
    child = database.update_job_child(
        child["id"],
        status="preparing",
        progress=0.0,
        stage="submitting",
        outputs=[],
        error=None,
        started_at=None,
        completed_at=None,
    )
    prompt = child["prompt_snapshot"]

    def node_id(class_type: str) -> str:
        return next(
            str(candidate)
            for candidate, node in prompt.items()
            if node["class_type"] == class_type
        )

    monkeypatch.setattr(
        database,
        "get_job",
        lambda _job_id: (_ for _ in ()).throw(
            AssertionError("native websocket events must use the status projection")
        ),
    )
    sink = client.director_app.state.progress_manager._sink
    for class_type, expected_stage in (
        ("CLIPLoader", "加载文本编码器"),
        ("UNETLoader", "加载生成模型"),
        ("MiniMaxH3ImageToVideo", "构建画面条件"),
        ("BasicGuider", "准备采样引导"),
    ):
        await sink(
            "http://comfy.test:8188",
            ComfyExecutionEvent(
                prompt_id=child["prompt_id"],
                node_id=node_id(class_type),
            ),
        )
        stored = database.get_job_child(child["id"])
        assert stored is not None
        assert stored["status"] == "running"
        assert stored["progress"] == 0.0
        assert stored["stage"] == expected_stage

    # Queue reconciliation can observe the running prompt between the
    # conditioning and sampler websocket frames. Its synthetic 1% lifecycle
    # floor must not suppress the exact sampler-start stage.
    conditioned = database.get_job_child(child["id"])
    assert conditioned is not None
    conditioned = database.update_job_child(
        child["id"],
        status="running",
        progress=0.01,
        stage=conditioned["stage"],
    )
    sampler_id = node_id("SamplerCustomAdvanced")
    await sink(
        "http://comfy.test:8188",
        ComfyExecutionEvent(
            prompt_id=child["prompt_id"],
            node_id=sampler_id,
        ),
    )
    sampling_started = database.get_job_child(child["id"])
    assert sampling_started is not None
    assert sampling_started["progress"] == pytest.approx(0.01)
    assert sampling_started["stage"] == "采样中"

    await sink(
        "http://comfy.test:8188",
        ComfyProgressEvent(
            prompt_id=child["prompt_id"],
            node_id=sampler_id,
            value=1.0,
            maximum=4.0,
        ),
    )
    first_step = database.get_job_child(child["id"])
    assert first_step is not None
    assert first_step["progress"] == pytest.approx(0.175)
    assert first_step["stage"] == "采样中 · 1/4"


async def test_immediate_raylight_execution_before_submit_response_is_preserved(
    client, fake_comfy, monkeypatch
) -> None:
    raw_settings = default_settings().model_dump(mode="json")
    raw_settings["multi_gpu_enabled"] = True
    raw_settings["models"]["fl2va"].update(
        {
            "backend": "raylight",
            "raylight": {
                "gpu_select": [0, 1],
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_degree": 1,
                "dp_degree": 1,
                "fsdp": False,
                "cpu_offload": False,
            },
        }
    )
    save_database_legacy_settings(
        client.director_app.state.database,
        RuntimeSettings.model_validate(raw_settings),
    )
    fake_comfy.auto_complete_raylight = False
    original_submit = fake_comfy.submit
    receipt_states: list[str] = []

    async def execute_before_return(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        assert prompt_id is not None
        initializer = next(
            str(node_id)
            for node_id, node in prompt.items()
            if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
        )
        await client.director_app.state.progress_manager._sink(
            "http://comfy.test:8188",
            ComfyExecutionEvent(prompt_id=prompt_id, node_id=initializer),
        )
        database = client.director_app.state.database
        early_child = database.find_any_job_children_by_prompt_id(prompt_id)[0]
        assert early_child["status"] == "running", early_child
        submitted = await original_submit(
            prompt,
            client_id,
            prompt_id,
            on_receipt=on_receipt,
        )
        ownership = database.get_prompt_ownership(early_child["id"])
        assert ownership is not None
        receipt_states.append(ownership.state)
        return submitted

    monkeypatch.setattr(fake_comfy, "submit", execute_before_return)
    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )

    assert response.status_code == 200, response.text
    parent = response.json()
    assert parent["status"] == "preparing"
    database = client.director_app.state.database

    async def child_is_running() -> None:
        while database.list_job_children(parent["id"])[0]["status"] != "running":
            await asyncio.sleep(0)

    await asyncio.wait_for(child_is_running(), timeout=1.0)
    child = database.list_job_children(parent["id"])[0]
    assert child["status"] == "running"
    assert child["progress"] == pytest.approx(0.0)
    assert child["stage"] == "初始化 RayLight 多卡"
    assert receipt_states == ["owned_requested_id"]
    await client.post(f"/api/jobs/{parent['id']}/cancel")


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed"])
async def test_completion_before_submit_response_is_accepted_and_converges(
    client, fake_comfy, monkeypatch, terminal_status: str
) -> None:
    original_submit = fake_comfy.submit

    async def complete_before_return(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        assert prompt_id is not None
        sampler = next(
            str(node_id)
            for node_id, node in prompt.items()
            if node["class_type"] == "SamplerCustomAdvanced"
        )
        save = next(
            str(node_id)
            for node_id, node in prompt.items()
            if node["class_type"] == "SaveVideo"
        )
        await client.director_app.state.progress_manager._sink(
            "http://comfy.test:8188",
            ComfyExecutionEvent(prompt_id=prompt_id, node_id=sampler),
        )
        child = client.director_app.state.database.get_job_child(prompt_id)
        assert child is not None and child["status"] == "running"
        client.director_app.state.database.update_job_child(
            prompt_id,
            status=terminal_status,
            progress=1.0,
            stage="completed" if terminal_status == "succeeded" else "failed",
            outputs=(
                [
                    {
                        "node_id": save,
                        "filename": "fast.mp4",
                        "subfolder": "video",
                        "type": "output",
                    }
                ]
                if terminal_status == "succeeded"
                else []
            ),
            error="fast failure" if terminal_status == "failed" else None,
            completed_at="2026-08-13T00:00:00+00:00",
        )
        return await original_submit(
            prompt,
            client_id,
            prompt_id,
            on_receipt=on_receipt,
        )

    monkeypatch.setattr(fake_comfy, "submit", complete_before_return)
    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )
    assert response.status_code == 200, response.text
    submitted = response.json()
    assert submitted["status"] == "preparing"
    while client.director_app.state.submission_tasks:
        await asyncio.sleep(0)

    database = client.director_app.state.database
    submitted_child = database.list_job_children(submitted["id"])[0]
    assert submitted_child["status"] == terminal_status
    stored = database.get_job(submitted["id"])
    assert stored is not None
    reconciled = await _sync_job(
        Request({"type": "http", "app": client.director_app}), stored
    )
    if terminal_status == "succeeded":
        # A websocket/projection write is not typed result authority.  Even
        # when it races ahead of POST /prompt returning, a Stage-4 child may
        # publish success only through exact history + ObservedArtifactSpec.
        assert reconciled["status"] == "failed"
        assert reconciled["stage"] in {"segments_failed", "output_missing"}
        assert reconciled["outputs"] == []
    else:
        assert reconciled["status"] == terminal_status
    if terminal_status == "succeeded":
        assert reconciled["progress"] == 1.0
    else:
        assert reconciled["error"] == "fast failure"
