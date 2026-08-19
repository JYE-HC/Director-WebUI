from __future__ import annotations

import asyncio

import pytest
from starlette.requests import Request

from directordeck.app import _sync_job
from directordeck.progress import ComfyExecutionEvent, ComfyProgressEvent
from directordeck.schemas import RuntimeSettings, default_settings

from .conftest import runnable_draft


async def test_raylight_native_events_persist_all_visible_execution_stages(
    client,
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
    client.director_app.state.database.put_settings(
        RuntimeSettings.model_validate(raw_settings)
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

    stages = [
        (None, 0.01, "开始执行"),
        (
            node_id("RayInitializerAdvanced"),
            0.10,
            "片段 1/1 · 初始化 RayLight 多卡",
        ),
        (
            node_id("RayUNETLoader"),
            0.10,
            "片段 1/1 · 加载 RayLight 生成模型",
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

    sampler_id = node_id("XFuserSamplerCustomAdvanced")
    await sink(
        origin,
        ComfyExecutionEvent(prompt_id=prompt_id, node_id=sampler_id),
    )
    ray_sampling = database.get_job_child(child["id"])
    assert ray_sampling is not None
    assert ray_sampling["stage"] == "片段 1/1 · RayLight 采样中"
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
    assert sampling["progress"] == pytest.approx(0.29)
    assert sampling["stage"] == "片段 1/1 · 采样 5/25"

    for class_type, expected_progress, expected_stage in (
        ("VAEDecode", 0.90, "片段 1/1 · 解码视频画面"),
        ("CreateVideo", 0.95, "片段 1/1 · 封装音视频"),
        ("SaveVideo", 0.98, "片段 1/1 · 写入视频文件"),
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
            node_id=node_id("RayInitializerAdvanced"),
        ),
    )
    assert database.get_job_child(child["id"]) == terminal


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
    client.director_app.state.database.put_settings(
        RuntimeSettings.model_validate(raw_settings)
    )
    fake_comfy.auto_complete_raylight = False
    original_submit = fake_comfy.submit

    async def execute_before_return(prompt, client_id, prompt_id=None):
        assert prompt_id is not None
        initializer = next(
            str(node_id)
            for node_id, node in prompt.items()
            if node["class_type"] == "RayInitializerAdvanced"
        )
        await client.director_app.state.progress_manager._sink(
            "http://comfy.test:8188",
            ComfyExecutionEvent(prompt_id=prompt_id, node_id=initializer),
        )
        return await original_submit(prompt, client_id, prompt_id)

    monkeypatch.setattr(fake_comfy, "submit", execute_before_return)
    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )

    assert response.status_code == 200, response.text
    parent = response.json()
    assert parent["status"] == "preparing"
    database = client.director_app.state.database
    while database.list_job_children(parent["id"])[0]["status"] != "running":
        await asyncio.sleep(0)
    child = database.list_job_children(parent["id"])[0]
    assert child["status"] == "running"
    assert child["progress"] == pytest.approx(0.10)
    assert child["stage"] == "片段 1/1 · 初始化 RayLight 多卡"
    await client.post(f"/api/jobs/{parent['id']}/cancel")


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed"])
async def test_completion_before_submit_response_is_accepted_and_converges(
    client, fake_comfy, monkeypatch, terminal_status: str
) -> None:
    original_submit = fake_comfy.submit

    async def complete_before_return(prompt, client_id, prompt_id=None):
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
        return await original_submit(prompt, client_id, prompt_id)

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
    assert reconciled["status"] == terminal_status
    if terminal_status == "succeeded":
        assert reconciled["progress"] == 1.0
    else:
        assert reconciled["error"] == "fast failure"
