from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest
from starlette.requests import Request

from director.app import _sync_job
from director.comfy import ComfyError
from director.media import MediaToolError, VideoProxy, VideoProxyResult
from director.schemas import VideoMetadata

from .conftest import runnable_draft, wait_for_submission_tasks


async def _reconcile(client, job_id: str) -> dict:
    database = client.director_app.state.database
    job = database.get_job(job_id)
    assert job is not None
    return await _sync_job(
        Request({"type": "http", "app": client.director_app}), job
    )


async def _submitted_job(client, created: dict) -> dict:
    await wait_for_submission_tasks(client)
    response = await client.get(f"/api/jobs/{created['id']}")
    assert response.status_code == 200, response.text
    return response.json()


def media_bytes(filename: str) -> bytes:
    lowered = filename.lower()
    if lowered.endswith(".png"):
        return b"\x89PNG\r\n\x1a\nfixture"
    if lowered.endswith(".wav"):
        return b"RIFF\x04\x00\x00\x00WAVEfixture"
    if lowered.endswith(".mp4"):
        return b"\x00\x00\x00\x18ftypisomfixture"
    raise AssertionError(f"no test media signature for {filename}")


async def test_asset_upload_returns_stable_identity_and_preview(client, fake_comfy) -> None:
    response = await client.post(
        "/api/assets",
        data={"kind": "image"},
        files={"file": ("../hero image.png", media_bytes("hero image.png"), "image/png")},
    )
    assert response.status_code == 200, response.text
    asset = response.json()["asset"]
    assert asset["name"] == "hero_image.png"
    assert asset["path"] == "director-web/hero_image.png"
    assert asset["id"]
    assert asset["preview_url"] == f"/api/assets/{asset['id']}/preview"
    assert fake_comfy.uploads[0]["filename"] == "hero_image.png"

    preview = await client.get(asset["preview_url"])
    assert preview.status_code == 200
    assert preview.content == b"fake-video"
    assert preview.headers["content-type"].startswith("image/png")
    assert preview.headers["x-content-type-options"] == "nosniff"
    assert preview.headers["content-disposition"].startswith("inline;")


async def test_create_job_submits_api_prompt_and_syncs_history(client, fake_comfy) -> None:
    response = await client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})
    assert response.status_code == 200, response.text
    job = await _submitted_job(client, response.json())
    assert job["status"] == "queued"
    assert job["prompt_id"] == fake_comfy.prompts[0]["prompt_id"]
    submitted = fake_comfy.prompts[0]
    assert submitted["client_id"] == "director-web"
    save_id = next(
        node_id
        for node_id, node in submitted["prompt"].items()
        if node["class_type"] == "SaveVideo"
    )

    fake_comfy.pending = []
    fake_comfy.histories[job["prompt_id"]] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            save_id: {
                "videos": [
                    {"filename": "Director_t2v.mp4", "subfolder": "video", "type": "output"}
                ]
            }
        },
    }
    await _reconcile(client, job["id"])
    finished = await client.get(f"/api/jobs/{job['id']}")
    assert finished.status_code == 200
    assert finished.json()["status"] == "succeeded"
    assert finished.json()["progress"] == 1.0
    assert finished.json()["outputs"] == [f"/api/jobs/{job['id']}/outputs/0"]
    assert finished.json()["output_files"] == ["output/video/Director_t2v.mp4"]

    output = await client.get(finished.json()["outputs"][0])
    assert output.status_code == 200
    assert output.content == b"fake-video"
    assert output.headers["content-type"].startswith("video/mp4")
    assert output.headers["x-content-type-options"] == "nosniff"
    assert output.headers["content-disposition"].startswith("inline;")

    partial = await client.get(
        finished.json()["outputs"][0], headers={"Range": "bytes=1-3"}
    )
    assert partial.status_code == 206
    assert partial.content == b"ake"
    assert partial.headers["accept-ranges"] == "bytes"
    assert partial.headers["content-range"] == "bytes 1-3/10"


async def test_completed_job_output_can_be_safely_imported_as_input_asset(
    client, fake_comfy, monkeypatch
) -> None:
    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )
    job = await _submitted_job(client, response.json())
    submitted = fake_comfy.prompts[0]
    save_id = next(
        node_id
        for node_id, node in submitted["prompt"].items()
        if node["class_type"] == "SaveVideo"
    )
    fake_comfy.pending = []
    fake_comfy.histories[job["prompt_id"]] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            save_id: {
                "videos": [
                    {
                        "filename": "Director_t2v.mp4",
                        "subfolder": "video",
                        "type": "output",
                    }
                ]
            }
        },
    }
    await _reconcile(client, job["id"])
    monkeypatch.setattr(
        "director.task_management.create_24fps_proxy_bytes",
        lambda _content, _suffix: VideoProxy(
            content=b"normalized-24fps",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(
                {
                    "duration": 12.0,
                    "native_fps": 24.0,
                    "frame_count": 288,
                    "width": 1920,
                    "height": 1080,
                    "probe_method": "fake_ffprobe",
                    "has_audio": True,
                }
            ),
        ),
    )

    imported = await client.post(
        f"/api/jobs/{job['id']}/import-output", json={"output_index": 0}
    )

    assert imported.status_code == 200, imported.text
    asset = imported.json()["asset"]
    assert asset["type"] == "input"
    assert asset["kind"] == "video"
    assert asset["metadata"]["native_fps"] == 24
    assert fake_comfy.uploads[-1] == {
        "filename": "Director_t2v_24fps.mp4",
        "content": b"normalized-24fps",
        "content_type": "video/mp4",
        "kind": "video",
    }
    assets = (await client.get("/api/assets")).json()["assets"]
    assert any(candidate["id"] == asset["id"] for candidate in assets)


async def test_running_job_uses_native_child_progress(client, fake_comfy) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    fake_comfy.pending = []
    fake_comfy.running = [[0, created["prompt_id"]]]
    child = client.director_app.state.database.list_job_children(created["id"])[0]
    client.director_app.state.database.update_job_child(
        child["id"], status="running", progress=0.42, stage="采样 · 第 7/25 步"
    )

    await _reconcile(client, created["id"])
    running = (await client.get(f"/api/jobs/{created['id']}")).json()

    assert running["status"] == "running"
    assert running["progress"] == pytest.approx(0.42)
    assert running["stage"] == "采样 · 第 7/25 步"


async def test_running_progress_is_monotonic_and_falls_back_for_old_nodes(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    fake_comfy.pending = []
    fake_comfy.running = [[0, created["prompt_id"]]]
    child = client.director_app.state.database.list_job_children(created["id"])[0]
    client.director_app.state.database.update_job_child(
        child["id"], status="running", progress=0.6, stage="采样"
    )

    await _reconcile(client, created["id"])
    stale = (await client.get(f"/api/jobs/{created['id']}")).json()
    assert stale["progress"] == pytest.approx(0.6)

    fallback = (await client.get(f"/api/jobs/{created['id']}")).json()
    assert fallback["progress"] == pytest.approx(0.6)
    assert fallback["stage"] == "采样"


async def test_first_running_poll_without_progress_endpoint_does_not_leave_queued_stage(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    fake_comfy.pending = []
    fake_comfy.running = [[0, created["prompt_id"]]]

    await _reconcile(client, created["id"])
    running = (await client.get(f"/api/jobs/{created['id']}")).json()

    assert running["status"] == "running"
    assert running["progress"] == pytest.approx(0.01)
    assert running["stage"] == "sampling"


async def test_same_status_concurrent_progress_updates_cannot_rewind_job(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    database = client.director_app.state.database
    database.update_job(
        created["id"], status="running", progress=0.2, stage="采样 · 第 5/25 步"
    )
    baseline = database.get_job(created["id"])
    assert baseline is not None

    newest = database.update_job_progress_monotonic(
        created["id"],
        "running",
        0.8,
        stage="采样 · 第 20/25 步",
        started_at="2026-01-01T00:00:00+00:00",
        expected_updated_at=baseline["updated_at"],
    )
    stale = database.update_job_progress_monotonic(
        created["id"],
        "running",
        0.4,
        stage="采样 · 第 10/25 步",
        started_at="2026-01-01T00:00:00+00:00",
        expected_updated_at=baseline["updated_at"],
    )

    assert newest is not None and newest["progress"] == pytest.approx(0.8)
    assert stale is None
    authoritative = database.get_job(created["id"])
    assert authoritative is not None
    assert authoritative["progress"] == pytest.approx(0.8)
    assert authoritative["stage"] == "采样 · 第 20/25 步"


async def test_equal_progress_stale_phase_cannot_overwrite_newer_phase(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    database = client.director_app.state.database
    database.update_job(
        created["id"], status="running", progress=0.5, stage="采样 · 第 25/25 步"
    )
    baseline = database.get_job(created["id"])
    assert baseline is not None

    decode = database.update_job_progress_monotonic(
        created["id"],
        "running",
        0.5,
        stage="AV 解码",
        started_at="2026-01-01T00:00:00+00:00",
        expected_updated_at=baseline["updated_at"],
    )
    stale_sample = database.update_job_progress_monotonic(
        created["id"],
        "running",
        0.5,
        stage="采样 · 第 25/25 步",
        started_at="2026-01-01T00:00:00+00:00",
        expected_updated_at=baseline["updated_at"],
    )

    assert decode is not None and decode["stage"] == "AV 解码"
    assert stale_sample is None
    authoritative = database.get_job(created["id"])
    assert authoritative is not None and authoritative["stage"] == "AV 解码"


async def test_delete_terminal_job_forgets_only_the_director_record(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    save_id = next(
        node_id
        for node_id, node in fake_comfy.prompts[-1]["prompt"].items()
        if node["class_type"] == "SaveVideo"
    )
    fake_comfy.pending = []
    fake_comfy.histories[created["prompt_id"]] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            save_id: {
                "videos": [
                    {
                        "filename": "Director_t2v_delete.mp4",
                        "subfolder": "video",
                        "type": "output",
                    }
                ]
            }
        },
    }
    await _reconcile(client, created["id"])
    finished = (await client.get(f"/api/jobs/{created['id']}")).json()
    output_url = finished["outputs"][0]
    assert (await client.get(output_url)).status_code == 200

    deleted = await client.delete(f"/api/jobs/{created['id']}")

    assert deleted.status_code == 200
    assert deleted.json() == {
        "deleted_job_id": created["id"],
        "outputs_preserved": True,
    }
    assert (await client.get(f"/api/jobs/{created['id']}")).status_code == 404
    assert (await client.get(output_url)).status_code == 404
    # Deleting a Director row does not dispatch an output-file deletion to
    # ComfyUI; the upstream history/output remains owned by that server.
    assert created["prompt_id"] in fake_comfy.histories
    assert fake_comfy.view_content == b"fake-video"


async def test_delete_job_rejects_active_work_without_comfy_io(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)

    queue_requests = fake_comfy.queue_requests
    history_requests = list(fake_comfy.history_requests)
    active = await client.delete(f"/api/jobs/{created['id']}")

    assert active.status_code == 409
    assert "cancel" in active.json()["detail"]
    assert fake_comfy.queue_requests == queue_requests
    assert fake_comfy.history_requests == history_requests
    assert (await client.get(f"/api/jobs/{created['id']}")).status_code == 200
    assert (await client.delete("/api/jobs/missing-job")).status_code == 404


async def test_clear_jobs_removes_terminal_rows_and_keeps_active_work(client) -> None:
    jobs = []
    for mode in ("t2v", "i2v", "fl2v"):
        response = await client.post(
            "/api/jobs", json={"mode": mode, "config": runnable_draft(mode)}
        )
        assert response.status_code == 200, response.text
        jobs.append(response.json())
    await wait_for_submission_tasks(client)
    database = client.director_app.state.database  # type: ignore[attr-defined]
    database.update_job(jobs[0]["id"], status="succeeded", progress=1.0)
    database.update_job(jobs[1]["id"], status="failed", progress=1.0)

    cleared = await client.delete("/api/jobs")

    assert cleared.status_code == 200
    assert cleared.json() == {
        "deleted_count": 2,
        "active_count": 1,
        "outputs_preserved": True,
    }
    assert database.get_job(jobs[0]["id"]) is None
    assert database.get_job(jobs[1]["id"]) is None
    assert database.get_job(jobs[2]["id"])["status"] == "queued"


async def test_database_job_delete_is_compare_and_set(client) -> None:
    database = client.director_app.state.database  # type: ignore[attr-defined]
    now = "2026-08-12T00:00:00+00:00"
    job = database.create_job(
        {
            "id": "delete-cas",
            "mode": "t2v",
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": "prompt-delete-cas",
            "outputs": [],
            "error": None,
            "config_snapshot": runnable_draft("t2v"),
            "settings_snapshot": {},
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
        }
    )
    assert job["status"] == "succeeded"

    assert database.delete_job_if_status(job["id"], "failed") is False
    assert database.get_job(job["id"]) is not None
    assert database.delete_job_if_status(job["id"], "succeeded") is True
    with pytest.raises(KeyError):
        database.delete_job_if_status(job["id"], "succeeded")


@pytest.mark.parametrize(
    ("mode", "settings_slot", "selected_model"),
    [
        ("t2v", "fl2va", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
        ("r2v", "ref2va", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    ],
)
async def test_each_diffusion_slot_accepts_any_model_from_the_shared_inventory(
    client, fake_comfy, mode: str, settings_slot: str, selected_model: str
) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"][settings_slot]["filename"] = selected_model
    assert (await client.put("/api/settings", json=settings)).status_code == 200

    response = await client.post(
        "/api/jobs", json={"mode": mode, "config": runnable_draft(mode)}
    )

    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    loader = next(
        node
        for node in fake_comfy.prompts[-1]["prompt"].values()
        if node["class_type"] == "UNETLoader"
    )
    assert loader["inputs"]["unet_name"] == selected_model


async def test_job_automatically_uses_model_only_lora(client, fake_comfy) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"]["fl2va"].update(
        lora_name="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        lora_strength=0.5,
        lora_loader="dedicated",
    )
    assert (await client.put("/api/settings", json=settings)).status_code == 200

    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )

    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    lora = next(
        node
        for node in fake_comfy.prompts[-1]["prompt"].values()
        if node["class_type"] == "LoraLoaderModelOnly"
    )
    assert lora["inputs"]["lora_name"] == (
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    )
    assert lora["inputs"]["strength_model"] == 0.5


async def test_standard_ref2va_official_lora_compiles_without_raylight(
    client, fake_comfy
) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"]["ref2va"].update(
        lora_name="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        lora_strength=0.8,
    )
    assert (await client.put("/api/settings", json=settings)).status_code == 200

    response = await client.post(
        "/api/jobs", json={"mode": "r2v", "config": runnable_draft("r2v")}
    )

    assert response.status_code == 200, response.text
    assert response.json()["children"][0]["backend"] == "standard"
    await wait_for_submission_tasks(client)
    prompt = fake_comfy.prompts[-1]["prompt"]
    assert "LoraLoaderModelOnly" in {
        node["class_type"] for node in prompt.values()
    }
    assert "RayLoraLoader" not in {
        node["class_type"] for node in prompt.values()
    }


async def test_unknown_named_generic_lora_is_resolved_from_remote_metadata(
    client, fake_comfy
) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"]["fl2va"].update(
        lora_name="renamed_generic.safetensors",
        lora_strength=0.6,
    )
    assert (await client.put("/api/settings", json=settings)).status_code == 200

    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )

    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    assert fake_comfy.lora_metadata_requests == ["renamed_generic.safetensors"]
    prompt = fake_comfy.prompts[-1]["prompt"]
    assert "LoraLoaderModelOnly" in {
        node["class_type"] for node in prompt.values()
    }


async def test_scoped_standard_lora_override_handles_missing_metadata(
    client, fake_comfy
) -> None:
    settings = (await client.get("/api/settings")).json()
    binding = settings["models"]["fl2va"]
    binding["lora_name"] = "style.safetensors"
    binding["standard_lora_loader_override"] = {
        "loader": "model_only",
        "lora_name": "style.safetensors",
        "model_filename": binding["filename"],
        "comfy_origin": settings["comfy_url"],
    }
    saved = await client.put("/api/settings", json=settings)
    assert saved.status_code == 200, saved.text

    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )

    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    assert fake_comfy.lora_metadata_requests == []
    prompt = fake_comfy.prompts[-1]["prompt"]
    assert "LoraLoaderModelOnly" in {
        node["class_type"] for node in prompt.values()
    }


async def test_job_rejects_unavailable_lora_before_submission(client, fake_comfy) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"]["fl2va"].update(
        lora_name="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        lora_loader="model_only",
    )
    assert (await client.put("/api/settings", json=settings)).status_code == 200

    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )

    assert response.status_code == 409
    assert (
        "loras:fl2va:minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
        in response.text
    )
    assert fake_comfy.prompts == []


async def test_job_requires_the_automatically_derived_lora_loader_node(client, fake_comfy) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"]["fl2va"].update(
        lora_name="minimax_h3_fl2v_turbo_4step_v1.0_768p_10ErosMax_beta1_pruned_compat_v001_T8.safetensors",
        lora_loader="model_only",
    )
    assert (await client.put("/api/settings", json=settings)).status_code == 200
    fake_comfy.available_nodes.remove("LoraLoaderBypassModelOnly")

    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )

    # The typed graph is valid, but the configured ComfyUI runtime cannot
    # satisfy it. Treat this as a runtime capability conflict, consistently
    # with unavailable model inventory and logical devices.
    assert response.status_code == 409
    assert "LoraLoaderBypassModelOnly" in response.text
    assert fake_comfy.prompts == []


async def test_job_refuses_unknown_auto_lora_dialect(client, fake_comfy) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"]["fl2va"].update(
        lora_name="style.safetensors",
        lora_loader="auto",
    )
    assert (await client.put("/api/settings", json=settings)).status_code == 200

    response = await client.post(
        "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
    )

    # Auto could not derive a safe typed graph from this LoRA dialect, so this
    # is an invalid job configuration rather than a runtime inventory conflict.
    assert response.status_code == 422
    assert "cannot be inferred safely" in response.text
    assert fake_comfy.prompts == []


async def test_create_job_can_use_saved_mode_draft(client, fake_comfy) -> None:
    draft = runnable_draft("i2v")
    assert (await client.put("/api/drafts/i2v", json=draft)).status_code == 200
    response = await client.post("/api/jobs", json={"mode": "i2v"})
    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    load_image = next(
        node
        for node in fake_comfy.prompts[-1]["prompt"].values()
        if node["class_type"] == "LoadImage"
    )
    assert load_image["inputs"]["image"] == "director-web/first.png"


async def test_cancel_job_targets_comfy_prompt(client, fake_comfy) -> None:
    created = (await client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})).json()
    created = await _submitted_job(client, created)
    cancelled = await client.post(f"/api/jobs/{created['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert fake_comfy.cancelled == [created["prompt_id"]]
    assert fake_comfy.pending_cancelled == [created["prompt_id"]]
    assert fake_comfy.interrupted == []


async def test_cancel_running_job_uses_targeted_interrupt(client, fake_comfy) -> None:
    created = (await client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})).json()
    created = await _submitted_job(client, created)
    fake_comfy.pending = []
    fake_comfy.running = [[0, created["prompt_id"]]]
    cancelled = await client.post(f"/api/jobs/{created['id']}/cancel")
    assert cancelled.status_code == 200
    assert fake_comfy.interrupted == [created["prompt_id"]]
    assert fake_comfy.pending_cancelled == []


async def test_cancel_syncs_a_just_completed_job_before_mutating_comfy(client, fake_comfy) -> None:
    created = (await client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})).json()
    created = await _submitted_job(client, created)
    save_id = next(
        node_id
        for node_id, node in fake_comfy.prompts[0]["prompt"].items()
        if node["class_type"] == "SaveVideo"
    )
    fake_comfy.pending = []
    fake_comfy.histories[created["prompt_id"]] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            save_id: {
                "videos": [
                    {"filename": "completed.mp4", "subfolder": "video", "type": "output"}
                ]
            }
        },
    }

    response = await client.post(f"/api/jobs/{created['id']}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert fake_comfy.cancelled == []


async def test_explicit_cancel_wins_when_completion_arrives_during_cancel_dispatch(
    client, fake_comfy
) -> None:
    created = (await client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})).json()
    created = await _submitted_job(client, created)
    save_id = next(
        node_id
        for node_id, node in fake_comfy.prompts[0]["prompt"].items()
        if node["class_type"] == "SaveVideo"
    )
    fake_comfy.complete_on_cancel = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            save_id: {
                "videos": [
                    {"filename": "completed.mp4", "subfolder": "video", "type": "output"}
                ]
            }
        },
    }

    response = await client.post(f"/api/jobs/{created['id']}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert fake_comfy.cancelled == []


async def test_cancel_during_preflight_prevents_upstream_submission(client, fake_comfy) -> None:
    fake_comfy.preflight_started = asyncio.Event()
    fake_comfy.preflight_release = asyncio.Event()
    create_request = asyncio.create_task(
        client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})
    )
    await asyncio.wait_for(fake_comfy.preflight_started.wait(), timeout=1)
    jobs = (await client.get("/api/jobs")).json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "preparing"
    cancelled = await client.post(f"/api/jobs/{jobs[0]['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    fake_comfy.preflight_release.set()
    created = await asyncio.wait_for(create_request, timeout=1)

    assert created.status_code == 200
    assert created.json()["status"] == "cancelled"
    assert fake_comfy.prompts == []


async def test_cancel_during_submit_cancels_minted_prompt_without_reviving_job(
    client, fake_comfy
) -> None:
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    create_request = asyncio.create_task(
        client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})
    )
    await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
    jobs = (await client.get("/api/jobs")).json()["jobs"]
    assert len(jobs) == 1
    assert jobs[0]["status"] == "preparing"
    preallocated_prompt_id = jobs[0]["children"][0]["prompt_id"]

    cancelled = await client.post(f"/api/jobs/{jobs[0]['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelling"
    assert cancelled.json()["children"][0]["stage"] == "cancelling_during_submit"
    fake_comfy.submit_release.set()
    accepted = await asyncio.wait_for(create_request, timeout=1)
    await wait_for_submission_tasks(client)
    created = await client.get(f"/api/jobs/{jobs[0]['id']}")

    assert accepted.status_code == 200
    assert accepted.json()["status"] == "preparing"
    assert created.status_code == 200
    assert created.json()["status"] == "cancelled"
    assert created.json()["prompt_id"] == preallocated_prompt_id
    assert fake_comfy.cancelled == [preallocated_prompt_id]
    refreshed = await client.get(f"/api/jobs/{jobs[0]['id']}")
    assert refreshed.json()["status"] == "cancelled"


async def test_completion_during_post_submit_cancel_is_not_overwritten(
    client, fake_comfy
) -> None:
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    create_request = asyncio.create_task(
        client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})
    )
    await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
    job = (await client.get("/api/jobs")).json()["jobs"][0]
    preallocated_prompt_id = job["children"][0]["prompt_id"]
    cancelled = await client.post(f"/api/jobs/{job['id']}/cancel")
    assert cancelled.json()["status"] == "cancelling"
    # A successful native child must still provide its exact expected
    # SaveVideo output; completion without it is intentionally failed closed.
    save_id = next(
        node_id
        for node_id, node in fake_comfy.prompts[0]["prompt"].items()
        if node["class_type"] == "SaveVideo"
    ) if fake_comfy.prompts else None
    # submit is held before FakeComfy records the prompt, so derive the output
    # node from the durable child snapshot instead.
    if save_id is None:
        child = client.director_app.state.database.list_job_children(job["id"])[0]
        save_id = next(iter(child["output_nodes"].values()))
    fake_comfy.complete_on_cancel = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            save_id: {
                "videos": [
                    {"filename": "completed.mp4", "subfolder": "video", "type": "output"}
                ]
            }
        },
    }
    fake_comfy.submit_release.set()

    accepted = await asyncio.wait_for(create_request, timeout=1)
    await wait_for_submission_tasks(client)
    created = await client.get(f"/api/jobs/{job['id']}")

    assert accepted.status_code == 200
    assert accepted.json()["status"] == "preparing"
    assert created.status_code == 200
    assert created.json()["status"] == "cancelled"
    assert created.json()["prompt_id"] == preallocated_prompt_id


async def test_cancel_failure_after_inflight_submit_stays_reconcilable(
    client, fake_comfy
) -> None:
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    create_request = asyncio.create_task(
        client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})
    )
    await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
    job = (await client.get("/api/jobs")).json()["jobs"][0]
    preallocated_prompt_id = job["children"][0]["prompt_id"]
    cancelled = await client.post(f"/api/jobs/{job['id']}/cancel")
    assert cancelled.json()["status"] == "cancelling"

    fake_comfy.cancel_error = ComfyError("cancel endpoint unavailable")
    fake_comfy.submit_release.set()
    accepted = await asyncio.wait_for(create_request, timeout=1)
    await wait_for_submission_tasks(client)

    assert accepted.status_code == 200
    assert accepted.json()["status"] == "preparing"
    persisted = (await client.get(f"/api/jobs/{job['id']}")).json()
    assert persisted["status"] == "cancelling"
    assert persisted["stage"] == "cancel_failed"
    assert persisted["prompt_id"] == preallocated_prompt_id
    assert persisted["completed_at"] is None


async def test_cancel_transport_failure_stays_reconcilable(client, fake_comfy) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    fake_comfy.cancel_error = ComfyError("cancel endpoint unavailable")

    response = await client.post(f"/api/jobs/{created['id']}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelling"
    assert response.json()["stage"] == "cancel_failed"
    assert response.json()["completed_at"] is None

    child = client.director_app.state.database.list_job_children(created["id"])[0]
    save_id = next(iter(child["output_nodes"].values()))
    fake_comfy.cancel_error = None
    fake_comfy.pending = []
    fake_comfy.histories[created["prompt_id"]] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            save_id: {
                "videos": [
                    {"filename": "completed.mp4", "subfolder": "video", "type": "output"}
                ]
            }
        },
    }
    await _reconcile(client, created["id"])
    reconciled = await client.get(f"/api/jobs/{created['id']}")
    assert reconciled.json()["status"] == "cancelled"


async def test_explicit_cancel_retries_a_transient_timeline_dispatch_failure(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    prompt_id = created["children"][0]["prompt_id"]
    fake_comfy.cancel_error = ComfyError("cancel endpoint unavailable")

    first = await client.post(f"/api/jobs/{created['id']}/cancel")
    assert first.status_code == 200
    assert first.json()["status"] == "cancelling"
    assert first.json()["stage"] == "cancel_failed"

    fake_comfy.cancel_error = None
    second = await client.post(f"/api/jobs/{created['id']}/cancel")

    assert second.status_code == 200
    assert second.json()["status"] == "cancelled"
    assert second.json()["error"] is None
    assert fake_comfy.cancelled == [prompt_id]


async def test_interrupted_cancelling_job_recovers_as_cancelled(client, fake_comfy) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    fake_comfy.cancel_error = ComfyError("local write crashed after dispatch")
    failed_dispatch = await client.post(f"/api/jobs/{created['id']}/cancel")
    assert failed_dispatch.json()["status"] == "cancelling"

    fake_comfy.cancel_error = None
    fake_comfy.pending = []
    fake_comfy.histories[created["prompt_id"]] = {
        "status": {
            "status_str": "error",
            "completed": True,
            "messages": [["execution_interrupted", {"prompt_id": created["prompt_id"]}]],
        },
        "outputs": {},
    }
    await _reconcile(client, created["id"])
    reconciled = await client.get(f"/api/jobs/{created['id']}")
    assert reconciled.json()["status"] == "cancelled"
    assert reconciled.json()["error"] is None


async def test_cancelled_pending_prompt_recovers_when_absent_everywhere(client, fake_comfy) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    fake_comfy.cancel_error = ComfyError("simulated local crash")
    first = await client.post(f"/api/jobs/{created['id']}/cancel")
    assert first.json()["status"] == "cancelling"

    fake_comfy.cancel_error = None
    fake_comfy.pending = []
    await _reconcile(client, created["id"])
    recovered = await client.get(f"/api/jobs/{created['id']}")

    assert recovered.json()["status"] == "cancelled"


async def test_stale_sync_cannot_overwrite_a_terminal_job(client, fake_comfy) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    fake_comfy.history_started = asyncio.Event()
    fake_comfy.history_release = asyncio.Event()
    fake_comfy.history_response = {
        created["prompt_id"]: {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {},
        }
    }
    fake_comfy.pending = []
    stale_request = asyncio.create_task(_reconcile(client, created["id"]))
    await asyncio.wait_for(fake_comfy.history_started.wait(), timeout=1)

    app = client.director_app
    app.state.database.update_job(
        created["id"],
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        completed_at="2026-01-01T00:00:00+00:00",
    )
    fake_comfy.history_release.set()
    await asyncio.wait_for(stale_request, timeout=1)
    response = await client.get(f"/api/jobs/{created['id']}")

    assert response.json()["status"] == "cancelled"


async def test_sqlite_only_list_omits_a_deleted_job(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    database = client.director_app.state.database
    database.update_job(
        created["id"],
        status="succeeded",
        progress=1.0,
        stage="completed",
        completed_at="2026-01-01T00:00:00+00:00",
    )

    active = (
        await client.post(
            "/api/jobs", json={"mode": "i2v", "config": runnable_draft("i2v")}
        )
    ).json()
    active = await _submitted_job(client, active)
    deleted = await client.delete(f"/api/jobs/{created['id']}")
    assert deleted.status_code == 200
    response = await client.get("/api/jobs")

    assert response.status_code == 200
    assert [job["id"] for job in response.json()["jobs"]] == [active["id"]]


async def test_sqlite_only_list_handles_a_directly_deleted_row(client, fake_comfy) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    database = client.director_app.state.database
    database.update_job(
        created["id"],
        status="succeeded",
        progress=1.0,
        stage="completed",
        completed_at="2026-01-01T00:00:00+00:00",
    )
    assert database.delete_job_if_status(created["id"], "succeeded") is True
    response = await client.get("/api/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"] == []


async def test_single_job_get_returns_404_for_a_deleted_row(
    client, fake_comfy
) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    database = client.director_app.state.database
    database.update_job(
        created["id"],
        status="succeeded",
        progress=1.0,
        stage="completed",
        completed_at="2026-01-01T00:00:00+00:00",
    )
    assert database.delete_job_if_status(created["id"], "succeeded") is True
    response = await client.get(f"/api/jobs/{created['id']}")

    assert response.status_code == 404
    assert response.json()["detail"] == "job not found"


async def test_history_for_another_prompt_is_not_misattributed(client, fake_comfy) -> None:
    created = (
        await client.post(
            "/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")}
        )
    ).json()
    created = await _submitted_job(client, created)
    fake_comfy.pending = []
    fake_comfy.history_response = {
        "unrelated-prompt": {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {},
        }
    }

    refreshed = await client.get(f"/api/jobs/{created['id']}")

    assert refreshed.json()["status"] == "queued"


async def test_invalid_or_incomplete_job_does_not_submit(client, fake_comfy) -> None:
    draft = runnable_draft("i2v")
    draft["shots"][0]["first_image"] = None
    response = await client.post("/api/jobs", json={"mode": "i2v", "config": draft})
    assert response.status_code == 422
    assert not fake_comfy.prompts


async def test_success_history_with_progress_messages_is_not_failed(client, fake_comfy) -> None:
    created = (await client.post("/api/jobs", json={"mode": "t2v", "config": runnable_draft("t2v")})).json()
    created = await _submitted_job(client, created)
    child = client.director_app.state.database.list_job_children(created["id"])[0]
    save_id = next(iter(child["output_nodes"].values()))
    fake_comfy.pending = []
    fake_comfy.histories[created["prompt_id"]] = {
        "status": {
            "status_str": "success",
            "completed": True,
            "messages": [["execution_start", {"prompt_id": created["prompt_id"]}], ["execution_success", {}]],
        },
        "outputs": {
            save_id: {
                "videos": [
                    {"filename": "completed.mp4", "subfolder": "video", "type": "output"}
                ]
            }
        },
    }
    await _reconcile(client, created["id"])
    refreshed = await client.get(f"/api/jobs/{created['id']}")
    assert refreshed.status_code == 200
    assert refreshed.json()["status"] == "succeeded"
    assert refreshed.json()["error"] is None


@pytest.mark.parametrize(
    ("kind", "filename", "content_type"),
    [
        ("image", "still.PNG", "image/png"),
        ("audio", "dialogue.wav", "audio/x-wav"),
        ("video", "source.mp4", "video/mp4"),
    ],
)
async def test_upload_accepts_allowlisted_extension_mime_pairs(
    client, fake_comfy, monkeypatch, kind: str, filename: str, content_type: str
) -> None:
    proxy_metadata = VideoMetadata(
        duration=1.0,
        native_fps=24.0,
        frame_count=24,
        width=64,
        height=64,
        probe_method="test_proxy",
    )
    monkeypatch.setattr(
        "director.app.create_24fps_proxy_file",
        lambda source, destination: (
            destination.write_bytes(b"proxy-video"),
            VideoProxyResult(metadata=proxy_metadata, strategy="transcode"),
        )[1],
    )
    response = await client.post(
        "/api/assets",
        data={"kind": kind},
        files={"file": (filename, media_bytes(filename), content_type)},
    )

    assert response.status_code == 200, response.text
    assert response.json()["asset"]["kind"] == kind
    if kind == "video":
        assert fake_comfy.uploads[-1]["filename"] == "source_24fps.mp4"
        assert fake_comfy.uploads[-1]["content_type"] == "video/mp4"
        assert response.json()["asset"]["metadata"] == proxy_metadata.model_dump(mode="json")
    else:
        assert fake_comfy.uploads[-1]["content_type"] == content_type


async def test_upload_exposes_bounded_phase_metrics(client, fake_comfy) -> None:
    upload_id = "12345678-1234-4123-8123-123456789abc"
    response = await client.post(
        "/api/assets",
        data={"kind": "image", "upload_id": upload_id},
        files={"file": ("still.png", media_bytes("still.png"), "image/png")},
    )
    progress = await client.get(f"/api/uploads/{upload_id}")

    assert response.status_code == 200, response.text
    assert progress.status_code == 200
    assert progress.json() == {
        "stage": "complete",
        "input_bytes": len(media_bytes("still.png")),
        "output_bytes": len(media_bytes("still.png")),
        "strategy": "passthrough",
        "elapsed_seconds": pytest.approx(0, abs=1),
    }
    assert isinstance(fake_comfy.uploads[-1]["content"], bytes)


async def test_unknown_upload_progress_is_not_created_by_read(client) -> None:
    response = await client.get("/api/uploads/12345678-1234-4123-8123-123456789abc")
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("kind", "filename", "content_type"),
    [
        ("image", "payload.exe", "image/png"),
        ("image", "still.png", "video/mp4"),
        ("audio", "voice.wav", "audio/mpeg"),
        ("video", "source.mp4", "application/octet-stream"),
    ],
)
async def test_upload_rejects_unapproved_or_mismatched_media_metadata(
    client, fake_comfy, kind: str, filename: str, content_type: str
) -> None:
    response = await client.post(
        "/api/assets",
        data={"kind": kind},
        files={"file": (filename, b"not-forwarded", content_type)},
    )

    assert response.status_code == 422
    assert fake_comfy.uploads == []


async def test_upload_stops_at_the_per_kind_size_limit(client, fake_comfy, monkeypatch) -> None:
    import director.app as app_module

    monkeypatch.setitem(app_module._UPLOAD_LIMITS, "image", 4)
    response = await client.post(
        "/api/assets",
        data={"kind": "image"},
        files={"file": ("oversize.png", b"12345", "image/png")},
    )

    assert response.status_code == 413
    assert fake_comfy.uploads == []


async def test_failed_video_proxy_does_not_upload_or_register_asset(
    client, fake_comfy, tmp_path, monkeypatch
) -> None:
    with sqlite3.connect(tmp_path / "director.sqlite3") as database:
        before = database.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    def fail_proxy(source: Path, destination: Path) -> VideoProxyResult:
        raise MediaToolError("invalid video")

    monkeypatch.setattr("director.app.create_24fps_proxy_file", fail_proxy)

    response = await client.post(
        "/api/assets",
        data={"kind": "video"},
        files={"file": ("unprobeable.mp4", media_bytes("unprobeable.mp4"), "video/mp4")},
    )

    assert response.status_code == 422
    assert fake_comfy.uploads == []
    with sqlite3.connect(tmp_path / "director.sqlite3") as database:
        after = database.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    assert after == before


@pytest.mark.parametrize("mode", ["v2v", "rv2v"])
@pytest.mark.parametrize("endpoint", ["draft", "job"])
async def test_source_trim_cannot_exceed_probed_duration(
    client, fake_comfy, mode: str, endpoint: str
) -> None:
    draft = runnable_draft(mode)
    draft["shots"][0]["source_start_seconds"] = 8.0
    draft["shots"][0]["source_duration_seconds"] = 4.01
    if endpoint == "draft":
        response = await client.put(f"/api/drafts/{mode}", json=draft)
    else:
        response = await client.post("/api/jobs", json={"mode": mode, "config": draft})

    assert response.status_code == 422
    assert "metadata.duration" in response.text
    assert fake_comfy.prompts == []
