from __future__ import annotations

import copy
import json

from .conftest import wait_for_submission_tasks


async def _create_timeline_job(client, *, title: str | None = None) -> dict:
    project = copy.deepcopy((await client.get("/api/timeline")).json())
    project["segments"][0]["prompt"] = "A cinematic camera move"
    if title is not None:
        project["title"] = title
    else:
        saved = await client.put("/api/timeline", json=project)
        assert saved.status_code == 200, saved.text
    response = await client.post("/api/timeline/jobs", json={"config": project})
    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    created = response.json()
    refreshed = await client.get(f"/api/jobs/{created['id']}")
    assert refreshed.status_code == 200, refreshed.text
    return refreshed.json()


async def test_job_list_filters_sorts_summarizes_and_never_contacts_comfy(
    client, fake_comfy
) -> None:
    current = await _create_timeline_job(client)
    alpha = await _create_timeline_job(client, title="Alpha 失败项目")
    beta = await _create_timeline_job(client, title="Beta 完成项目")
    database = client.director_app.state.database
    database.update_job(
        current["id"],
        status="running",
        progress=0.5,
        stage="采样中",
        started_at="2026-08-13T12:00:00+00:00",
    )
    database.update_job(
        alpha["id"],
        status="failed",
        progress=1.0,
        stage="执行失败",
        error="token=private-value generation failed",
        started_at="2026-08-13T10:00:00+00:00",
        completed_at="2026-08-13T10:00:09+00:00",
    )
    database.update_job(
        beta["id"],
        status="succeeded",
        progress=1.0,
        stage="completed",
        outputs=[
            {
                "filename": "beta.mp4",
                "subfolder": "video",
                "type": "output",
            }
        ],
        started_at="2026-08-13T11:00:00+00:00",
        completed_at="2026-08-13T11:00:03+00:00",
    )

    queue_requests = fake_comfy.queue_requests
    history_requests = list(fake_comfy.history_requests)
    filtered = await client.get(
        "/api/jobs",
        params={
            "status": "failed",
            "q": "Alpha",
            "sort_by": "execution_duration",
            "sort_order": "desc",
            "limit": 1,
        },
    )

    assert filtered.status_code == 200, filtered.text
    payload = filtered.json()
    assert payload["total"] == 1
    assert payload["has_more"] is False
    assert payload["summary"] == {
        "total": 3,
        "active": 1,
        "queued": 0,
        "preparing": 0,
        "running": 1,
        "cancelling": 0,
        "succeeded": 1,
        "failed": 1,
        "cancelled": 0,
    }
    task = payload["jobs"][0]
    assert task["id"] == alpha["id"]
    assert task["display_name"] == "Alpha 失败项目"
    assert task["project_title"] == "Alpha 失败项目"
    assert task["execution_duration_seconds"] == 9.0
    assert task["error_summary"] == "token=[已隐藏] generation failed"
    assert task["current_project"] is False
    assert fake_comfy.queue_requests == queue_requests
    assert fake_comfy.history_requests == history_requests

    page = (await client.get("/api/jobs", params={"limit": 1})).json()
    assert page["total"] == 3
    assert page["has_more"] is True
    completed = (
        await client.get("/api/jobs", params={"status": "succeeded"})
    ).json()["jobs"][0]
    assert completed["output_count"] == 1
    current_task = (
        await client.get("/api/jobs", params={"status": "running"})
    ).json()["jobs"][0]
    assert current_task["id"] == current["id"]
    assert current_task["current_project"] is True


async def test_bulk_cancel_prevalidates_all_local_parent_ids(client, fake_comfy) -> None:
    first = await _create_timeline_job(client, title="待取消一")
    second = await _create_timeline_job(client, title="待取消二")
    database = client.director_app.state.database
    cancelled_before = list(fake_comfy.cancelled)

    rejected = await client.post(
        "/api/jobs/cancel",
        json={"job_ids": [first["id"], "not-a-local-parent"]},
    )

    assert rejected.status_code == 404
    assert fake_comfy.cancelled == cancelled_before
    assert database.get_job(first["id"])["cancel_requested"] == 0

    cancelled = await client.post(
        "/api/jobs/cancel",
        json={"job_ids": [first["id"], second["id"]]},
    )

    assert cancelled.status_code == 200, cancelled.text
    payload = cancelled.json()
    assert payload["requested_count"] == 2
    assert payload["terminal_count"] == 2
    assert [task["id"] for task in payload["jobs"]] == [first["id"], second["id"]]
    assert {task["status"] for task in payload["jobs"]} == {"cancelled"}
    assert set(fake_comfy.cancelled) >= {first["prompt_id"], second["prompt_id"]}


async def test_diagnostic_is_redacted_and_project_snapshot_is_typed(
    client, fake_comfy
) -> None:
    created = await _create_timeline_job(client, title="可恢复项目")
    database = client.director_app.state.database
    database.update_job(
        created["id"],
        status="failed",
        progress=1.0,
        stage="failed",
        error=(
            "token=private-token Authorization: Bearer private-bearer request to "
            "https://alice:private-password@comfy.test failed "
            "access_token=private-access refreshToken=private-refresh "
            "client_secret=private-client-secret apiKey=private-api-key "
            "\"Authorization\": \"Bearer private-json-bearer\" "
            "Authorization: 'Basic private-quoted-basic' "
            "password = \"private password with spaces\""
        ),
        completed_at="2026-08-13T12:00:00+00:00",
    )

    response = await client.get(f"/api/jobs/{created['id']}/diagnostic")

    assert response.status_code == 200, response.text
    diagnostic = response.json()
    serialized = json.dumps(diagnostic, ensure_ascii=False)
    assert diagnostic["settings_included"] is False
    assert diagnostic["workflow_included"] is False
    assert "private-token" not in serialized
    assert "private-bearer" not in serialized
    assert "private-password" not in serialized
    assert "private-access" not in serialized
    assert "private-refresh" not in serialized
    assert "private-client-secret" not in serialized
    assert "private-api-key" not in serialized
    assert "private-json-bearer" not in serialized
    assert "private-quoted-basic" not in serialized
    assert "private password with spaces" not in serialized
    assert "settings_snapshot" not in serialized
    assert "config_snapshot" not in serialized
    assert "prompt_snapshot" not in serialized
    assert "prompt_id" not in serialized

    project_response = await client.get(f"/api/jobs/{created['id']}/project")
    assert project_response.status_code == 200, project_response.text
    project = project_response.json()
    assert project["job_id"] == created["id"]
    assert project["project"]["title"] == "可恢复项目"
    assert project["project"]["version"] == 4
    assert "settings_snapshot" not in project
    assert "prompt_snapshot" not in project


async def test_generation_details_are_typed_lazy_and_exclude_runtime_secrets(
    client, fake_comfy
) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["client_id"] = "private-client-id"
    settings["models"]["fl2va"].update(
        lora_name=(
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_"
            "comfyui_bf16.safetensors"
        ),
        lora_strength=0.75,
        raylight={
            "gpu_select": [0, 1],
            "ulysses_degree": 2,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    saved_settings = await client.put("/api/settings", json=settings)
    assert saved_settings.status_code == 200, saved_settings.text

    project = copy.deepcopy((await client.get("/api/timeline")).json())
    project["title"] = "生成参数快照"
    project["render"] = {"width": 1280, "height": 704, "fps": 24.0}
    project["export_mode"] = "segments"
    project["sampling"]["fl2va"].update(
        steps=4,
        seed=123456789,
        random_seed=False,
        sampler="euler",
        scheduler="karras",
        shift=9.5,
        audio_shift=2.5,
    )
    project["segments"][0].update(
        title="暴风雪环境",
        prompt="POV walks through a blizzard\nThe red apple stays visible.",
        duration_seconds=10.0,
        ref_image_size="max",
        audio_mode="generate",
    )
    created_response = await client.post(
        "/api/timeline/jobs", json={"config": project}
    )
    assert created_response.status_code == 200, created_response.text
    await wait_for_submission_tasks(client)
    job_id = created_response.json()["id"]

    queue_requests = fake_comfy.queue_requests
    history_requests = list(fake_comfy.history_requests)
    response = await client.get(f"/api/jobs/{job_id}/generation-details")

    assert response.status_code == 200, response.text
    details = response.json()
    assert details["schema_version"] == 2
    assert details["project_title"] == "生成参数快照"
    assert details["render"] == {
        "width": 1280,
        "height": 704,
        "fps": 24.0,
        "export_mode": "segments",
        "total_duration_seconds": 10.0,
    }
    assert details["sampling"] == [
        {
            "family": "fl2va",
            "steps": 4,
            "seed": 123456789,
            "random_seed": False,
            "sampler": "euler",
            "scheduler": "karras",
            "shift": 9.5,
            "audio_shift": 2.5,
        }
    ]
    assert details["models"][0]["family"] == "fl2va"
    assert details["models"][0]["lora_strength"] == 0.75
    assert details["models"][0]["backends"] == ["raylight"]
    assert details["models"][0]["logical_gpu_indices"] == [0, 1]
    assert details["models"][0]["ulysses_degree"] == 2
    assert details["segments"][0]["recipe"] == "t2v"
    assert details["segments"][0]["duration_seconds"] == 10.0
    assert details["segments"][0]["ref_image_size"] == "max"
    assert details["segments"][0]["audio_mode"] == "generate"
    assert details["segments"][0]["prompt"].startswith("POV walks")
    serialized = json.dumps(details, ensure_ascii=False)
    assert "private-client-id" not in serialized
    assert "comfy_url" not in serialized
    assert "client_id" not in serialized
    assert "settings_snapshot" not in serialized
    assert "prompt_snapshot" not in serialized
    assert "prompt_id" not in serialized
    assert "source_start_seconds" not in serialized
    assert fake_comfy.queue_requests == queue_requests
    assert fake_comfy.history_requests == history_requests
