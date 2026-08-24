from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import re
import uuid
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from starlette.requests import Request

import directordeck.app as director_app_module
from directordeck.app import (
    _await_raylight_transition,
    _child_with_execution_evidence,
    _ordered_timeline_outputs,
    _refresh_raylight_runtime_tail,
    _recover_interrupted_submission,
    _sync_job,
    _sync_timeline_job,
    create_app,
)
from directordeck.comfy import ComfyError, ComfyPromptRejected
from directordeck.database import Database
from directordeck.host_artifacts import PermanentHostOutputProbeError
from directordeck.media import VideoProxy
from directordeck.native_templates import (
    bind_raylight_runtime_epoch,
    compile_native_timeline,
    raylight_runtime_descriptor,
)
from directordeck.progress import ComfyExecutionEvent, ComfyReconcileHint
from directordeck.schemas import (
    RuntimeSettings,
    UnifiedTimelineDraft,
    VideoMetadata,
    default_settings,
    default_timeline_draft,
    utc_now,
)
from directordeck.workflow.execution import (
    ExactCancelConfirmedEvidence,
    ObservedArtifactSpec,
    OutputDescriptor,
)

from .conftest import (
    VIDEO_METADATA,
    FakeHostCapabilityProvider,
    adapt_legacy_workflow_requests,
    asset,
    save_database_legacy_settings,
    save_legacy_settings_document,
    runtime_authority_headers,
    v5_timeline_fixture,
)


@pytest.fixture(autouse=True)
def _stage6_v5_request_adapter(client, monkeypatch) -> None:
    adapt_legacy_workflow_requests(client, monkeypatch)


def _background_request(app) -> Request:
    return Request({"type": "http", "app": app})


async def _reconcile(client, parent: dict, *, allow_assembly: bool = False) -> dict:
    database = client.director_app.state.database
    latest = database.get_job(parent["id"])
    assert latest is not None
    return await _sync_timeline_job(
        _background_request(client.director_app),
        latest,
        allow_assembly=allow_assembly,
    )


async def _exercise_cancel_recovery_terminal_race(
    client, fake_comfy, monkeypatch, *, paused_actor: str
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"cas-{paused_actor}"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(parent["id"])[0]
    database.update_job(
        parent["id"], status="cancelling", stage="restart_cancel_pending"
    )
    database.update_job_child(
        child["id"], status="cancelling", stage="restart_cancel_pending"
    )
    snapshot = database.get_job(parent["id"])
    assert snapshot is not None

    stale_snapshot_ready = asyncio.Event()
    winner_committed_terminal = asyncio.Event()
    original_cas = director_app_module._cas_active_child_update
    original_confirm = director_app_module._confirm_typed_exact_cancel
    paused = False

    async def interleaved_cas(database_arg, child_snapshot, **updates):
        nonlocal paused
        actor = asyncio.current_task().get_name()
        if (
            actor == paused_actor
            and not paused
            and updates.get("stage") == "restart_cancel_pending"
        ):
            paused = True
            stale_snapshot_ready.set()
            await winner_committed_terminal.wait()
        result = await original_cas(database_arg, child_snapshot, **updates)
        if (
            actor != paused_actor
            and result[1]
            and updates.get("status") == "cancelled"
        ):
            winner_committed_terminal.set()
        return result

    monkeypatch.setattr(
        director_app_module, "_cas_active_child_update", interleaved_cas
    )

    def interleaved_confirm(*args, **kwargs):
        result = original_confirm(*args, **kwargs)
        actor = asyncio.current_task().get_name()
        if result is not None and actor != paused_actor:
            winner_committed_terminal.set()
        return result

    monkeypatch.setattr(
        director_app_module,
        "_confirm_typed_exact_cancel",
        interleaved_confirm,
    )

    async def recover() -> None:
        await _recover_interrupted_submission(
            _background_request(client.director_app), snapshot
        )

    async def user_cancel():
        return await client.post(f"/api/jobs/{parent['id']}/cancel")

    actors = {
        "restart-recovery": recover,
        "user-cancel": user_cancel,
    }
    winner_actor = (
        "user-cancel" if paused_actor == "restart-recovery" else "restart-recovery"
    )
    paused_task = asyncio.create_task(actors[paused_actor](), name=paused_actor)
    await asyncio.wait_for(stale_snapshot_ready.wait(), timeout=1)
    winner_task = asyncio.create_task(actors[winner_actor](), name=winner_actor)
    paused_result, winner_result = await asyncio.wait_for(
        asyncio.gather(paused_task, winner_task), timeout=1
    )
    user_response = (
        paused_result if paused_actor == "user-cancel" else winner_result
    )
    assert user_response.status_code == 200
    assert user_response.json()["status"] == "cancelled"

    stored_parent = database.get_job(parent["id"])
    stored_child = database.get_job_child(child["id"])
    assert stored_parent is not None and stored_parent["status"] == "cancelled"
    assert stored_child is not None and stored_child["status"] == "cancelled"
    assert fake_comfy.cancelled == [child["prompt_id"]]

    deleted = await client.delete(f"/api/jobs/{parent['id']}")
    assert deleted.status_code == 200, deleted.text
    assert database.get_job(parent["id"]) is None
    assert database.get_job_child(child["id"]) is None


def _segment(identity: str, mode: str = "t2v") -> dict:
    value = {
        "id": identity,
        "title": identity,
        "mode": mode,
        "duration_seconds": 1.0,
        "prompt": f"Prompt for {identity}",
        "enabled": True,
    }
    if mode == "i2v":
        value["first_image"] = asset("first.png", "image")
    elif mode == "r2v":
        value.update(
            reference_images=[asset("reference.png", "image", slot=0)],
            reference_audios=[],
            reference_videos=[],
        )
    return value


def _timeline(*segments: dict, export_mode: str = "segments") -> dict:
    value = default_timeline_draft().model_dump(mode="json")
    for sampling in value["sampling"].values():
        sampling["seed"] = 42
        sampling["random_seed"] = False
    value["export_mode"] = export_mode
    value["segments"] = list(segments)
    value["version"] = 1
    return value


def _continuity_timeline(*segments: dict) -> dict:
    value = _timeline(*segments)
    value["continuity"] = {"enabled": True, "overlap_frames": 5}
    return value


def _success(child: dict) -> dict:
    segment_id = child["segment_ids"][0]
    node_id = child["output_nodes"][segment_id]
    return {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            node_id: {
                "videos": [
                    {
                        "filename": f"{segment_id}.mp4",
                        "subfolder": "segments",
                        "type": "output",
                    }
                ]
            }
        },
    }


def _failure(message: str = "segment exploded") -> dict:
    return {
        "status": {
            "status_str": "error",
            "completed": True,
            "messages": [["execution_error", {"exception_message": message}]],
        },
        "outputs": {},
    }


def test_per_segment_units_are_safe_ordered_and_reuse_stable_loader_nodes() -> None:
    draft = UnifiedTimelineDraft.model_validate(
        _timeline(
            _segment("unsafe / 中文", "t2v"),
            _segment("anchor", "i2v"),
            _segment("reference", "r2v"),
        )
    )
    settings = default_settings()

    result = compile_native_timeline(draft, settings, "job-stable")

    assert [unit.segment_ids for unit in result.workflows] == [
        ("unsafe / 中文",),
        ("anchor",),
        ("reference",),
    ]
    assert [unit.family for unit in result.workflows] == [
        "fl2va",
        "fl2va",
        "ref2va",
    ]
    assert len({unit.id for unit in result.workflows}) == 3
    assert all(re.fullmatch(r"[a-z0-9-]+", unit.id) for unit in result.workflows)
    assert result.manifest["resident_cache_scope"]["prompt_partition"] == (
        "one_segment"
    )

    stable_classes = {
        "CLIPLoader",
        "SelectCLIPDevice",
        "VAELoader",
        "SelectVAEDevice",
        "UNETLoader",
        "SelectModelDevice",
        "MiniMaxH3SigmaShift",
    }

    def stable_nodes(unit) -> dict:
        return {
            node_id: node
            for node_id, node in unit.prompt.items()
            if node["class_type"] in stable_classes
        }

    assert stable_nodes(result.workflows[0]) == stable_nodes(result.workflows[1])
    assert result.workflows[0].output_nodes == {
        "unsafe / 中文": next(iter(result.workflows[0].output_nodes.values()))
    }


async def test_128_segment_background_sync_uses_one_queue_and_one_bulk_history(
    client, fake_comfy
) -> None:
    segments = [_segment(f"segment-{index:03d}") for index in range(128)]
    created = await client.post(
        "/api/timeline/jobs", json={"config": _timeline(*segments)}
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 128)
    assert len(fake_comfy.prompts) == 128
    children = client.director_app.state.database.list_job_children(parent["id"])
    assert len(children) == 128
    assert all(len(child["segment_ids"]) == 1 for child in children)

    fake_comfy.queue_requests = 0
    fake_comfy.history_requests.clear()
    queued = await client.get(f"/api/jobs/{parent['id']}")
    assert queued.status_code == 200
    assert queued.json()["status"] == "queued"
    assert fake_comfy.queue_requests == 0
    assert fake_comfy.history_requests == []

    fake_comfy.pending = []
    fake_comfy.histories = {
        child["prompt_id"]: _success(child) for child in children
    }
    fake_comfy.queue_requests = 0
    fake_comfy.history_requests.clear()
    await _reconcile(client, parent)
    completed = await client.get(f"/api/jobs/{parent['id']}")

    assert completed.status_code == 200, completed.text
    assert completed.json()["status"] == "succeeded"
    assert completed.json()["stage"] == "segments_completed"
    assert len(completed.json()["segment_results"]) == 128
    assert fake_comfy.queue_requests == 1
    assert fake_comfy.history_requests == [(None, 256)]


async def test_job_list_and_details_are_sqlite_only_when_comfy_is_a_black_hole(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs", json={"config": _timeline(_segment("sqlite-only"))}
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    database = client.director_app.state.database
    now = utc_now()
    settings = database.get_settings().model_dump(mode="json")
    database.create_job(
        {
            "id": "legacy-black-hole",
            "mode": "t2v",
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "prompt_id": "legacy-prompt",
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": settings,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    calls = {"queue": 0, "history": 0}
    never = asyncio.Event()

    async def black_hole_queue():
        calls["queue"] += 1
        await never.wait()

    async def black_hole_history(*_args, **_kwargs):
        calls["history"] += 1
        await never.wait()

    monkeypatch.setattr(fake_comfy, "queue", black_hole_queue)
    monkeypatch.setattr(fake_comfy, "history", black_hole_history)

    listing, timeline_detail, legacy_detail = await asyncio.wait_for(
        asyncio.gather(
            client.get("/api/jobs"),
            client.get(f"/api/jobs/{parent['id']}"),
            client.get("/api/jobs/legacy-black-hole"),
        ),
        timeout=0.25,
    )

    assert listing.status_code == 200
    assert timeline_detail.status_code == 200
    assert legacy_detail.status_code == 200
    assert calls == {"queue": 0, "history": 0}


async def test_parent_background_sync_is_single_flight(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs", json={"config": _timeline(_segment("shared"))}
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    fake_comfy.pending = []
    fake_comfy.histories[child["prompt_id"]] = _success(child)
    fake_comfy.queue_requests = 0
    fake_comfy.history_requests.clear()
    fake_comfy.history_started = asyncio.Event()
    fake_comfy.history_release = asyncio.Event()

    detail = asyncio.create_task(_reconcile(client, parent))
    await asyncio.wait_for(fake_comfy.history_started.wait(), timeout=1)
    listing = asyncio.create_task(_reconcile(client, parent))
    await asyncio.sleep(0)
    fake_comfy.history_release.set()
    detail_response, list_response = await asyncio.gather(detail, listing)

    assert detail_response["status"] == "succeeded"
    assert list_response["status"] == "succeeded"
    output_response = await client.get(
        f"/api/jobs/{parent['id']}/segment-output",
        params={"segment_id": "shared"},
    )
    assert output_response.status_code == 200
    assert fake_comfy.queue_requests == 1
    assert fake_comfy.history_requests == [(None, 128)]


async def test_typed_result_assembly_and_import_ignore_mutable_child_outputs(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("observed-authority"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(parent["id"])[0]
    fake_comfy.video_probe_result = {
        **VIDEO_METADATA,
        "width": 1280,
        "height": 720,
        "frame_count": 25,
        "duration": 25 / 24,
        # The authored mode expects generated audio; the take ledger must use
        # the actual probe rather than that mutable expectation.
        "has_audio": False,
        "probe_method": "observed-test-ffprobe-v1",
    }
    _complete_fake_prompt(fake_comfy, child)

    synced = await _reconcile(client, parent)
    observed = database.get_observed_artifact(child["id"])
    assert observed is not None
    assert observed.output_descriptor.filename == "observed-authority.mp4"
    assert observed.content_hash is None
    assert fake_comfy.video_probes == [
        {
            "filename": "observed-authority.mp4",
            "subfolder": "segments",
            "type": "output",
        }
    ]
    with database.connect() as connection:
        take = connection.execute(
            "SELECT has_audio FROM segment_takes WHERE source_child_id = ?",
            (child["id"],),
        ).fetchone()
        assert take is not None and int(take["has_audio"]) == 0
        connection.execute(
            "UPDATE job_children SET output_nodes = ?, outputs = ? WHERE id = ?",
            (
                json.dumps({"observed-authority": "forged-node"}),
                json.dumps(
                    [
                        {
                            "node_id": "forged-node",
                            "filename": "forged.mp4",
                            "subfolder": "attacker",
                            "type": "output",
                        }
                    ]
                ),
                child["id"],
            ),
        )
        connection.execute(
            "UPDATE jobs SET outputs = ? WHERE id = ?",
            (
                json.dumps(
                    [
                        {
                            "node_id": "assembly",
                            "filename": "forged-parent.mp4",
                            "subfolder": "attacker",
                            "type": "output",
                        }
                    ]
                ),
                parent["id"],
            ),
        )

    stored_parent = database.get_job(parent["id"])
    stored_child = database.get_job_child(child["id"])
    assert stored_parent is not None and stored_child is not None
    enriched = _child_with_execution_evidence(database, stored_child)
    ordered = _ordered_timeline_outputs(stored_parent, [enriched])
    expected = enriched["execution_evidence"][
        "exact_prompt_snapshot"
    ].expected_output_spec
    assert expected is not None
    assert ordered == [
        {
            "node_id": expected.node_id,
            "filename": "observed-authority.mp4",
            "subfolder": "segments",
            "type": "output",
        }
    ]

    viewed: list[dict[str, str]] = []
    original_view = fake_comfy.view

    async def capture_view(params: dict[str, str]):
        viewed.append(dict(params))
        return await original_view(params)

    monkeypatch.setattr(fake_comfy, "view", capture_view)
    result = await client.get(f"/api/jobs/{parent['id']}")
    assert result.status_code == 200, result.text
    assert result.json()["output_files"] == [
        "output/segments/observed-authority.mp4"
    ]
    assert result.json()["segment_results"][0]["output_file"] == (
        "output/segments/observed-authority.mp4"
    )
    assert result.json()["children"][0]["outputs"] == [
        "output/segments/observed-authority.mp4"
    ]
    media = await client.get(
        f"/api/jobs/{parent['id']}/segment-output",
        params={"segment_id": "observed-authority"},
    )
    assert media.status_code == 200, media.text
    parent_media = await client.get(f"/api/jobs/{parent['id']}/outputs/0")
    assert parent_media.status_code == 200, parent_media.text
    monkeypatch.setattr(
        "directordeck.task_management.create_24fps_proxy_bytes",
        lambda _content, _suffix: VideoProxy(
            content=b"normalized-observed",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        ),
    )
    imported = await client.post(
        f"/api/jobs/{parent['id']}/import-output",
        json={"segment_id": "observed-authority"},
    )
    assert imported.status_code == 200, imported.text
    parent_imported = await client.post(
        f"/api/jobs/{parent['id']}/import-output",
        json={"output_index": 0},
    )
    assert parent_imported.status_code == 200, parent_imported.text
    assert viewed == [
        {
            "filename": "observed-authority.mp4",
            "subfolder": "segments",
            "type": "output",
        },
        {
            "filename": "observed-authority.mp4",
            "subfolder": "segments",
            "type": "output",
        },
        {
            "filename": "observed-authority.mp4",
            "subfolder": "segments",
            "type": "output",
        },
        {
            "filename": "observed-authority.mp4",
            "subfolder": "segments",
            "type": "output",
        },
    ]

    # Deleting only the exact row simulates partial/corrupt typed persistence.
    # Ownership, receipt and observed-take markers remain, so the public read
    # must stay typed and hide mutable compatibility outputs rather than
    # reinterpreting the forged child columns as legacy authority.
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM job_child_execution_evidence WHERE child_id = ?",
            (child["id"],),
        )
    incomplete = await client.get(f"/api/jobs/{parent['id']}")
    assert incomplete.status_code == 200, incomplete.text
    assert incomplete.json()["outputs"] == []
    assert incomplete.json()["output_files"] == []
    assert incomplete.json()["children"][0]["outputs"] == []
    assert incomplete.json()["segment_results"] == []
    refused_import = await client.post(
        f"/api/jobs/{parent['id']}/import-output",
        json={"segment_id": "observed-authority"},
    )
    assert refused_import.status_code == 409
    assert "可信的实际媒体探测结果" in refused_import.text
    assert (
        await client.get(f"/api/jobs/{parent['id']}/outputs/0")
    ).status_code == 404
    refused_parent_import = await client.post(
        f"/api/jobs/{parent['id']}/import-output",
        json={"output_index": 0},
    )
    assert refused_parent_import.status_code == 404
    assert len(viewed) == 4
    assert synced["status"] == "succeeded"


async def test_pending_output_receipt_recovers_without_history_or_recompile(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("probe-recovery"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(parent["id"])[0]
    _complete_fake_prompt(fake_comfy, child)
    fake_comfy.video_probe_result = None

    first = await _reconcile(client, parent)
    pending = database.get_job_child(child["id"])
    receipt = database.get_output_observation_receipt(child["id"])
    ownership = database.get_prompt_ownership(child["id"])
    assert pending is not None and pending["status"] == "running"
    assert pending["stage"] == "verifying_output"
    assert receipt is not None
    assert ownership is not None and ownership.state == "terminal_confirmed"
    assert database.get_observed_artifact(child["id"]) is None
    assert first["status"] == "running"
    unavailable = await client.post(
        f"/api/jobs/{parent['id']}/import-output",
        json={"segment_id": "probe-recovery"},
    )
    assert unavailable.status_code == 409
    assert "可信的实际媒体探测结果" in unavailable.text

    # A later process/reconciler needs only the durable receipt.  History may
    # already have been pruned; no compiler or mutable child output is reused.
    fake_comfy.histories.clear()
    fake_comfy.video_probe_result = dict(VIDEO_METADATA)
    recovered = await _reconcile(client, parent)
    assert recovered["status"] == "succeeded"
    assert database.get_observed_artifact(child["id"]) is not None
    assert len(fake_comfy.video_probes) == 2


async def test_permanent_host_probe_rejection_closes_typed_artifact_failure(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("permanent-probe-rejection"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(created.json()["id"])[0]

    def reject_permanently(_descriptor):
        raise PermanentHostOutputProbeError("unsupported video descriptor")

    monkeypatch.setattr(fake_comfy, "probe_output", reject_permanently)
    _complete_fake_prompt(fake_comfy, child)
    result = await _reconcile(client, created.json())

    failed = database.get_job_child(child["id"])
    ownership = database.get_prompt_ownership(child["id"])
    assert result["status"] == "failed"
    assert failed is not None and failed["status"] == "failed"
    assert failed["stage"] == "artifact_verification_failed"
    assert ownership is not None and ownership.state == "terminal_confirmed"
    assert database.get_output_observation_receipt(child["id"]) is not None
    assert database.get_observed_artifact(child["id"]) is None


@pytest.mark.parametrize("backend", ["standard", "raylight"])
async def test_continuity_waits_for_retryable_observation_before_successor(
    client, fake_comfy, monkeypatch, backend: str
) -> None:
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    if backend == "raylight":
        settings = _raylight_settings_document()
        assert (await save_legacy_settings_document(client, settings)).status_code == 200
        fake_comfy.auto_complete_raylight = False
    second_mode = "r2v" if backend == "standard" else "t2v"
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment(f"{backend}-probe-root", "t2v"),
                _segment(f"{backend}-probe-successor", second_mode),
            )
        },
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    first, second = database.list_job_children(created.json()["id"])
    fake_comfy.video_probe_result = None
    _complete_fake_prompt(fake_comfy, first)

    await _wait_until(
        lambda: database.get_output_observation_receipt(first["id"]) is not None
    )
    pending = database.get_job_child(first["id"])
    untouched_successor = database.get_job_child(second["id"])
    ownership = database.get_prompt_ownership(first["id"])
    assert pending is not None and pending["status"] == "running"
    assert pending["stage"] == "verifying_output"
    assert ownership is not None and ownership.state == "terminal_confirmed"
    assert untouched_successor is not None
    assert untouched_successor["prompt_id"] is None
    assert untouched_successor["status"] == "preparing"
    assert len(fake_comfy.prompts) == 1
    if backend == "raylight":
        ray_state = database.get_raylight_runtime_state()
        assert ray_state is not None and ray_state["tainted"] is False

    fake_comfy.video_probe_result = dict(VIDEO_METADATA)
    await _wait_for_prompt_count(fake_comfy, 2)
    submitted_successor = database.get_job_child(second["id"])
    assert submitted_successor is not None
    assert submitted_successor["prompt_id"] == fake_comfy.prompts[1]["prompt_id"]
    _complete_fake_prompt(fake_comfy, submitted_successor)
    await _wait_for_submission_jobs(client)


async def test_cancelled_background_sync_caller_does_not_cancel_shared_flight(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs", json={"config": _timeline(_segment("shielded"))}
    )
    parent = created.json()
    await _wait_for_submission_jobs(client)
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    fake_comfy.pending = []
    fake_comfy.histories[child["prompt_id"]] = _success(child)
    fake_comfy.queue_requests = 0
    fake_comfy.history_requests.clear()
    fake_comfy.history_started = asyncio.Event()
    fake_comfy.history_release = asyncio.Event()

    disconnected = asyncio.create_task(_reconcile(client, parent))
    await asyncio.wait_for(fake_comfy.history_started.wait(), timeout=1)
    disconnected.cancel()
    with pytest.raises(asyncio.CancelledError):
        await disconnected
    replacement = asyncio.create_task(_reconcile(client, parent))
    await asyncio.sleep(0)
    fake_comfy.history_release.set()
    response = await replacement

    assert response["status"] == "succeeded"
    assert fake_comfy.queue_requests == 1
    assert fake_comfy.history_requests == [(None, 128)]


async def test_degraded_queue_never_fans_out_exact_history_requests(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                *[_segment(f"segment-{index:03d}") for index in range(32)]
            )
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    fake_comfy.pending = []
    fake_comfy.queue_error = ComfyError("queue unavailable")
    fake_comfy.history_requests.clear()

    await _reconcile(client, parent)
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert fake_comfy.history_requests == [(None, 128)]


async def test_external_pending_dequeue_keeps_typed_ownership_unconfirmed(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("removed-in-comfy"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    parent = (await client.get(f"/api/jobs/{parent['id']}")).json()
    child = parent["children"][0]

    # ComfyUI's Web queue-delete operation removes a pending prompt without
    # writing a history tombstone or notifying Director directly.
    fake_comfy.pending = []
    fake_comfy.history_requests.clear()
    await _reconcile(client, parent)

    refreshed = (await client.get(f"/api/jobs/{parent['id']}")).json()
    assert refreshed["status"] == "queued"
    assert refreshed["children"][0]["status"] == "queued"
    ownership = client.director_app.state.database.get_prompt_ownership(child["id"])
    assert ownership is not None and ownership.state == "unconfirmed"
    assert fake_comfy.history_requests == [(None, 128), (child["prompt_id"], None)]


async def test_legacy_external_pending_dequeue_converges_after_second_history_read(
    client, fake_comfy
) -> None:
    database = client.director_app.state.database
    now = utc_now()
    database.create_job(
        {
            "id": "legacy-removed-in-comfy",
            "mode": "t2v",
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "prompt_id": "legacy-removed-prompt",
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": database.get_settings().model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    fake_comfy.pending = []
    fake_comfy.history_requests.clear()
    snapshot = database.get_job("legacy-removed-in-comfy")
    assert snapshot is not None

    await _sync_job(_background_request(client.director_app), snapshot)

    refreshed = database.get_job("legacy-removed-in-comfy")
    assert refreshed is not None
    assert refreshed["status"] == "cancelled"
    assert refreshed["stage"] == "ComfyUI 端任务已移除"
    assert fake_comfy.history_requests == [
        ("legacy-removed-prompt", None),
        ("legacy-removed-prompt", None),
    ]


async def test_legacy_second_history_read_wins_queue_handoff_race(
    client, fake_comfy, monkeypatch
) -> None:
    database = client.director_app.state.database
    now = utc_now()
    database.create_job(
        {
            "id": "legacy-history-handoff",
            "mode": "t2v",
            "status": "running",
            "progress": 0.5,
            "stage": "sampling",
            "prompt_id": "legacy-handoff-prompt",
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": database.get_settings().model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": None,
        }
    )
    fake_comfy.pending = []
    fake_comfy.running = []
    history_reads = 0

    async def completing_history(
        prompt_id: str | None = None, *, max_items: int | None = None
    ) -> dict:
        nonlocal history_reads
        assert prompt_id == "legacy-handoff-prompt"
        assert max_items is None
        history_reads += 1
        if history_reads == 1:
            return {}
        return {
            prompt_id: {
                "status": {
                    "status_str": "success",
                    "completed": True,
                    "messages": [],
                },
                "outputs": {},
            }
        }

    monkeypatch.setattr(fake_comfy, "history", completing_history)
    snapshot = database.get_job("legacy-history-handoff")
    assert snapshot is not None

    await _sync_job(_background_request(client.director_app), snapshot)

    refreshed = database.get_job("legacy-history-handoff")
    assert refreshed is not None
    assert refreshed["status"] == "succeeded"
    assert refreshed["stage"] == "completed"
    assert history_reads == 2


async def test_external_running_interrupt_history_is_cancelled(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("interrupted-in-comfy"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    parent = (await client.get(f"/api/jobs/{parent['id']}")).json()
    child = parent["children"][0]
    fake_comfy.pending = []
    fake_comfy.histories[child["prompt_id"]] = {
        "status": {
            "status_str": "error",
            "completed": True,
            "messages": [
                ["execution_interrupted", {"prompt_id": child["prompt_id"]}]
            ],
        },
        "outputs": {},
    }

    await _reconcile(client, parent)

    refreshed = (await client.get(f"/api/jobs/{parent['id']}")).json()
    assert refreshed["status"] == "cancelled"
    assert refreshed["stage"] == "ComfyUI 端任务已移除"
    assert refreshed["children"][0]["status"] == "cancelled"
    assert refreshed["children"][0]["stage"] == "ComfyUI 端已中断"
    assert refreshed["children"][0]["error"] is None


async def test_running_prompt_missing_from_queue_keeps_typed_owner_unconfirmed(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("missing-running-state"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    fake_comfy.pending = []
    fake_comfy.running = [[0, child["prompt_id"]]]
    await _reconcile(client, parent)

    fake_comfy.running = []
    await _reconcile(client, parent)

    refreshed = (await client.get(f"/api/jobs/{parent['id']}")).json()
    assert refreshed["status"] == "running"
    assert refreshed["children"][0]["status"] == "running"
    ownership = client.director_app.state.database.get_prompt_ownership(child["id"])
    assert ownership is not None and ownership.state == "unconfirmed"


async def test_partial_external_removal_keeps_parent_live_and_successful_take(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("completed-segment"),
                _segment("removed-segment"),
            )
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    completed_child, removed_child = (
        client.director_app.state.database.list_job_children(parent["id"])
    )
    fake_comfy.pending = []
    fake_comfy.histories[completed_child["prompt_id"]] = _success(completed_child)

    await _reconcile(client, parent)

    refreshed = (await client.get(f"/api/jobs/{parent['id']}")).json()
    assert refreshed["status"] == "running"
    statuses = {child["id"]: child["status"] for child in refreshed["children"]}
    assert statuses == {
        completed_child["id"]: "succeeded",
        removed_child["id"]: "queued",
    }
    assert [result["segment_id"] for result in refreshed["segment_results"]] == [
        "completed-segment"
    ]


async def test_successful_raylight_control_does_not_release_removed_segment_owners(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("removed-after-control-a"),
                _segment("removed-after-control-b"),
            )
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    _add_succeeded_raylight_control(database, parent["id"], group_index=100)

    # Both real generation prompts were removed in ComfyUI. A successful
    # internal DirectorDeckRayKill is durable orchestration history, not a third segment
    # that turns this all-removed result into a misleading partial failure.
    fake_comfy.pending = []
    await _reconcile(client, parent)

    refreshed = database.get_job(parent["id"])
    assert refreshed is not None
    assert refreshed["status"] == "queued"
    segment_ownership = [
        database.get_prompt_ownership(child["id"])
        for child in database.list_job_children(parent["id"])
        if child["segment_ids"]
    ]
    assert all(
        ownership is not None and ownership.state == "unconfirmed"
        for ownership in segment_ownership
    )


async def test_raylight_control_does_not_count_as_segment_or_start_generation(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("counted-segment-a"),
                _segment("counted-segment-b"),
            )
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    _add_succeeded_raylight_control(database, parent["id"], group_index=100)

    queued = await _reconcile(client, parent)
    assert queued["status"] == "queued"
    assert queued["stage"] == "native segments 0/2"
    assert queued["started_at"] is None

    first, _second = [
        child
        for child in database.list_job_children(parent["id"])
        if child["segment_ids"]
    ]
    fake_comfy.pending = [
        item for item in fake_comfy.pending if first["prompt_id"] not in item
    ]
    fake_comfy.histories[str(first["prompt_id"])] = _success(first)
    advanced = await _reconcile(client, parent)
    assert advanced["status"] == "running"
    assert advanced["stage"] == "native segments 1/2"
    assert advanced["started_at"] is not None


async def test_external_absence_is_not_inferred_when_exact_history_fails(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("history-outage"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    fake_comfy.pending = []
    original_history = fake_comfy.history

    async def exact_history_outage(
        prompt_id: str | None = None, *, max_items: int | None = None
    ) -> dict:
        if prompt_id is not None:
            raise ComfyError("exact history temporarily unavailable")
        return await original_history(prompt_id, max_items=max_items)

    monkeypatch.setattr(fake_comfy, "history", exact_history_outage)
    await _reconcile(client, parent)

    refreshed = (await client.get(f"/api/jobs/{parent['id']}")).json()
    assert refreshed["status"] == "queued"
    assert refreshed["children"][0]["status"] == "queued"


async def test_external_absence_is_not_inferred_from_malformed_exact_history(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("bad-history-contract"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    parent = (await client.get(f"/api/jobs/{parent['id']}")).json()
    prompt_id = parent["children"][0]["prompt_id"]
    fake_comfy.pending = []

    async def malformed_exact_history(
        requested_prompt_id: str | None = None, *, max_items: int | None = None
    ) -> dict:
        if requested_prompt_id is None:
            assert max_items == 128
            return {}
        assert requested_prompt_id == prompt_id
        return {prompt_id: "malformed-entry"}

    monkeypatch.setattr(fake_comfy, "history", malformed_exact_history)
    await _reconcile(client, parent)

    refreshed = (await client.get(f"/api/jobs/{parent['id']}")).json()
    assert refreshed["status"] == "queued"
    assert refreshed["children"][0]["status"] == "queued"


async def test_external_absence_is_not_inferred_from_malformed_queue_snapshot(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("bad-queue-contract"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    fake_comfy.pending = []

    async def malformed_queue() -> dict:
        return {"queue_running": []}

    monkeypatch.setattr(fake_comfy, "queue", malformed_queue)
    await _reconcile(client, parent)

    refreshed = (await client.get(f"/api/jobs/{parent['id']}")).json()
    assert refreshed["status"] == "queued"
    assert refreshed["children"][0]["status"] == "queued"
    assert fake_comfy.history_requests == [(None, 128)]


async def test_exact_history_fallback_is_capped_per_parent_poll(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                *[_segment(f"segment-{index:03d}") for index in range(32)]
            )
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    fake_comfy.pending = []
    fake_comfy.histories = {}
    fake_comfy.history_requests.clear()

    await _reconcile(client, parent)
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert fake_comfy.history_requests[0] == (None, 128)
    assert len(fake_comfy.history_requests) == 17
    assert all(
        prompt_id is not None and maximum is None
        for prompt_id, maximum in fake_comfy.history_requests[1:]
    )


async def test_exact_history_fallback_rotates_without_starving_later_children(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                *[_segment(f"segment-{index:03d}") for index in range(32)]
            )
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    fake_comfy.pending = []
    fake_comfy.histories = {}

    fake_comfy.history_requests.clear()
    await _reconcile(client, parent)
    first = await client.get(f"/api/jobs/{parent['id']}")
    assert first.status_code == 200
    first_exact = {
        prompt_id
        for prompt_id, maximum in fake_comfy.history_requests
        if prompt_id is not None and maximum is None
    }
    assert len(first_exact) == 16

    fake_comfy.history_requests.clear()
    await _reconcile(client, parent)
    second = await client.get(f"/api/jobs/{parent['id']}")
    assert second.status_code == 200
    second_exact = {
        prompt_id
        for prompt_id, maximum in fake_comfy.history_requests
        if prompt_id is not None and maximum is None
    }
    assert len(second_exact) == 16
    assert first_exact.isdisjoint(second_exact)


async def test_one_segment_failure_preserves_other_generated_takes(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("first"), _segment("failed"), _segment("last")
            )
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    children = client.director_app.state.database.list_job_children(parent["id"])
    fake_comfy.pending = []
    for child in children:
        segment_id = child["segment_ids"][0]
        fake_comfy.histories[child["prompt_id"]] = (
            _failure() if segment_id == "failed" else _success(child)
        )

    await _reconcile(client, parent)
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "failed"
    assert job["stage"] == "segments_failed"
    assert [take["segment_id"] for take in job["segment_results"]] == [
        "first",
        "last",
    ]
    assert job["outputs"] == []


async def test_parent_cancel_preserves_completed_take_and_targets_only_active_segments(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("completed"), _segment("queued-a"), _segment("queued-b")
            )
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_submission_jobs(client)
    children = client.director_app.state.database.list_job_children(parent["id"])
    completed_child = children[0]
    fake_comfy.histories[completed_child["prompt_id"]] = _success(completed_child)
    fake_comfy.pending = [
        item
        for item in fake_comfy.pending
        if completed_child["prompt_id"] not in item
    ]

    cancelled = await client.post(f"/api/jobs/{parent['id']}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    job = cancelled.json()
    assert job["status"] == "cancelled"
    assert [take["segment_id"] for take in job["segment_results"]] == [
        "completed"
    ]
    assert set(fake_comfy.cancelled) == {
        children[1]["prompt_id"],
        children[2]["prompt_id"],
    }


async def test_client_disconnect_during_second_submit_does_not_strand_batch(
    client, fake_comfy, monkeypatch
) -> None:
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    original_submit = fake_comfy.submit
    calls = 0

    async def block_second(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        nonlocal calls
        calls += 1
        if calls == 2:
            second_started.set()
            await second_release.wait()
        return await original_submit(
            prompt,
            client_id,
            prompt_id,
            on_receipt=on_receipt,
        )

    monkeypatch.setattr(fake_comfy, "submit", block_second)
    request = asyncio.create_task(
        client.post(
            "/api/timeline/jobs",
            json={
                "config": _timeline(
                    _segment("first"), _segment("second"), _segment("third")
                )
            },
        )
    )
    await asyncio.wait_for(second_started.wait(), timeout=1)
    request.cancel()
    response = await request
    assert response.status_code == 200
    assert response.json()["status"] == "preparing"
    second_release.set()
    for _ in range(100):
        if not client.director_app.state.submission_tasks:
            break
        await asyncio.sleep(0.01)

    assert client.director_app.state.submission_tasks == set()
    persisted = client.director_app.state.database.list_jobs()[0]
    children = client.director_app.state.database.list_job_children(persisted["id"])
    assert persisted["status"] == "queued"
    assert len(children) == 3
    assert all(child["status"] == "queued" for child in children)
    assert all(child["prompt_id"] for child in children)
    assert len(fake_comfy.prompts) == 3


async def test_submit_side_effect_then_response_error_targets_durable_prompt(
    client, fake_comfy
) -> None:
    fake_comfy.submit_error_after_side_effect = ComfyError(
        "response connection lost"
    )

    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ambiguous"))},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "preparing"
    await _wait_for_submission_jobs(client)
    assert len(fake_comfy.prompts) == 1
    prompt_id = fake_comfy.prompts[0]["prompt_id"]
    assert fake_comfy.cancelled == [prompt_id]
    parent = client.director_app.state.database.list_jobs()[0]
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    evidence = client.director_app.state.database.get_job_child_execution_evidence(
        child["id"]
    )
    assert evidence is not None
    assert fake_comfy.prompts[0]["prompt"] == evidence[
        "exact_prompt_snapshot"
    ].model_dump(mode="json")["exact_prompt"]
    assert parent["status"] == "failed"
    assert parent["stage"] == "submission_failed"
    assert child["status"] == "cancelled"
    assert child["stage"] == "cancelled_after_submission_failure"


async def test_ambiguous_cleanup_releases_ticket_before_exact_cancel_finishes(
    client, fake_comfy
) -> None:
    fake_comfy.submit_error_after_side_effect = ComfyError(
        "response connection lost"
    )
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def pause_before_exact_cancel(_job_id, _child_id):
        cleanup_started.set()
        await cleanup_release.wait()

    client.director_app.state.before_cleanup_cancel_request = (
        pause_before_exact_cancel
    )
    first = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cleanup-ticket-owner"))},
    )
    assert first.status_code == 200, first.text

    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        fake_comfy.submit_error_after_side_effect = None
        follower = await client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("cleanup-ticket-follower"))},
        )
        assert follower.status_code == 200, follower.text
        await _wait_for_prompt_count(fake_comfy, 2)

        database = client.director_app.state.database
        follower_child = database.list_job_children(follower.json()["id"])[0]
        assert fake_comfy.prompts[1]["prompt_id"] == follower_child["id"]
        assert first.json()["id"] in client.director_app.state.submission_jobs
        assert not cleanup_release.is_set()
    finally:
        cleanup_release.set()

    await _wait_for_submission_jobs(client)


async def test_prompt_validation_rejection_fails_locally_and_releases_next_submit(
    client, fake_comfy, monkeypatch
) -> None:
    original_submit = fake_comfy.submit
    calls = 0
    rejection_detail = {
        "error": {"type": "prompt_outputs_failed_validation"},
        "node_errors": {"7": {"class_type": "MissingLoader"}},
    }

    async def reject_first(prompt, client_id, prompt_id=None, *, on_receipt=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ComfyPromptRejected(
                "validation rejected",
                status_code=400,
                detail=rejection_detail,
            )
        return await original_submit(
            prompt,
            client_id,
            prompt_id,
            on_receipt=on_receipt,
        )

    monkeypatch.setattr(fake_comfy, "submit", reject_first)
    rejected = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("definite-rejection"))},
    )
    assert rejected.status_code == 200, rejected.text
    await _wait_for_submission_jobs(client)

    database = client.director_app.state.database
    rejected_parent = database.get_job(rejected.json()["id"])
    rejected_child = database.list_job_children(rejected.json()["id"])[0]
    assert rejected_parent is not None
    assert rejected_parent["status"] == "failed"
    assert rejected_parent["stage"] == "submission_failed"
    assert rejected_child["status"] == "failed"
    assert rejected_child["stage"] == "submission_failed"
    assert rejected_child["prompt_id"] is None
    assert database.get_job_child_execution_evidence(rejected_child["id"]) is None
    assert database.get_prompt_ownership(rejected_child["id"]) is None
    assert json.loads(rejected_child["error"])["detail"] == rejection_detail
    assert fake_comfy.prompts == []
    assert fake_comfy.cancelled == []
    deleted = await client.delete(f"/api/jobs/{rejected.json()['id']}")
    assert deleted.status_code == 200, deleted.text

    following = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("after-definite-rejection"))},
    )
    assert following.status_code == 200, following.text
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    following_child = database.list_job_children(following.json()["id"])[0]
    assert fake_comfy.prompts[0]["prompt_id"] == following_child["id"]


async def test_explicit_retry_rejects_an_unreleased_prompt_owner(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("retry-active-owner"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent_id = created.json()["id"]
    child = database.list_job_children(parent_id)[0]
    ownership = database.get_prompt_ownership(child["id"])
    assert ownership is not None
    assert ownership.state == "owned_requested_id"

    retry = await client.post(f"/api/jobs/{parent_id}/retry")

    assert retry.status_code == 409
    assert "still owns" in retry.json()["detail"]
    assert len(database.list_jobs()) == 1
    assert len(fake_comfy.prompts) == 1


async def test_explicit_retry_rejects_terminal_lifecycle_with_unreleased_owner(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("retry-forged-terminal-owner"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent_id = created.json()["id"]
    child = database.list_job_children(parent_id)[0]
    database.update_job_child(
        child["id"],
        status="succeeded",
        progress=1.0,
        stage="completed",
        completed_at=utc_now(),
    )
    database.update_job(
        parent_id,
        status="succeeded",
        progress=1.0,
        stage="segments_completed",
        completed_at=utc_now(),
    )

    retry = await client.post(f"/api/jobs/{parent_id}/retry")

    assert retry.status_code == 409
    assert "still owns" in retry.json()["detail"]
    assert len(database.list_jobs()) == 1
    assert len(fake_comfy.prompts) == 1


@pytest.mark.parametrize(
    ("status_str", "completed"),
    [("success", False), ("pending", True)],
)
async def test_standard_contradictory_history_never_releases_prompt_ownership(
    client, fake_comfy, status_str: str, completed: bool
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"contradictory-{status_str}"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(created.json()["id"])[0]
    ownership_before = database.get_prompt_ownership(child["id"])
    assert ownership_before is not None

    with pytest.raises(ComfyError, match="contradictory"):
        await director_app_module._sync_timeline_child(
            _background_request(client.director_app),
            child,
            parent_cancelling=False,
            history_entry={
                "status": {
                    "status_str": status_str,
                    "completed": completed,
                    "messages": [],
                },
                "outputs": {},
            },
            running=False,
            pending=False,
            confirmed_absent=False,
        )

    assert database.get_prompt_ownership(child["id"]) == ownership_before
    assert database.get_job_child(child["id"])["status"] not in {
        "succeeded",
        "failed",
        "cancelled",
    }


async def test_explicit_retry_creates_a_fresh_execution_lineage(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("retry-terminal-source"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    source_id = created.json()["id"]
    source_child = database.list_job_children(source_id)[0]
    fake_comfy.pending = []
    fake_comfy.histories[source_child["prompt_id"]] = _success(source_child)
    terminal = await _reconcile(client, created.json())
    assert terminal["status"] == "succeeded"
    source_before = database.get_job(source_id)
    source_child_before = database.get_job_child(source_child["id"])
    source_evidence = database.get_job_child_execution_evidence(
        source_child["id"]
    )

    retry = await client.post(f"/api/jobs/{source_id}/retry")

    assert retry.status_code == 200, retry.text
    retry_id = retry.json()["id"]
    assert retry_id != source_id
    await _wait_for_prompt_count(fake_comfy, 2)
    await _wait_for_submission_jobs(client)
    retry_child = database.list_job_children(retry_id)[0]
    assert retry_child["id"] != source_child["id"]
    assert retry_child["prompt_id"] != source_child["prompt_id"]
    assert database.get_job_execution_plan(retry_id) is not None
    assert database.get_job_child_execution_evidence(retry_child["id"]) is not None
    assert database.get_prompt_ownership(retry_child["id"]) is not None
    assert database.get_job(source_id) == source_before
    assert database.get_job_child(source_child["id"]) == source_child_before
    assert (
        database.get_job_child_execution_evidence(source_child["id"])
        == source_evidence
    )


async def test_user_cancel_cannot_release_an_old_endpoint_epoch(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancel-old-endpoint-epoch"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(created.json()["id"])[0]
    before_io = (
        list(fake_comfy.cancelled),
        fake_comfy.queue_requests,
        list(fake_comfy.history_requests),
    )
    old_endpoint = client.director_app.state.endpoint_identity
    client.director_app.state.endpoint_identity = old_endpoint.model_copy(
        update={"runtime_instance_id": "replacement-comfy-boot"}
    )

    cancelled = await client.post(f"/api/jobs/{created.json()['id']}/cancel")

    assert cancelled.status_code == 409
    assert "restart confirmation" in cancelled.json()["detail"]
    assert (
        list(fake_comfy.cancelled),
        fake_comfy.queue_requests,
        list(fake_comfy.history_requests),
    ) == before_io
    ownership = database.get_prompt_ownership(child["id"])
    persisted_child = database.get_job_child(child["id"])
    persisted_parent = database.get_job(created.json()["id"])
    assert ownership is not None and ownership.state == "unconfirmed"
    assert persisted_child is not None
    assert persisted_child["stage"] == "restart_certificate_required"
    assert persisted_parent is not None
    assert persisted_parent["stage"] == "restart_certificate_required"

    confirmed = await client.post(
        f"/api/jobs/{created.json()['id']}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )

    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["status"] == "cancelled"
    assert (
        confirmed.json()["stage"]
        == "cancelled_after_confirmed_comfy_restart"
    )
    released = database.get_prompt_ownership(child["id"])
    assert released is not None and released.state == "cleanup_confirmed"
    certificate = released.cleanup_certificate
    assert certificate is not None
    assert certificate.kind == "endpoint_restart_certificate"
    assert certificate.prompt_id == released.effective_prompt_id
    assert certificate.endpoint_identity == old_endpoint
    assert certificate.restart_id == "replacement-comfy-boot"
    assert certificate.queue_and_history_cleared is True
    assert (
        list(fake_comfy.cancelled),
        fake_comfy.queue_requests,
        list(fake_comfy.history_requests),
    ) == before_io

    retried = await client.post(
        f"/api/jobs/{created.json()['id']}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )
    assert retried.status_code == 200, retried.text
    assert retried.json()["updated_at"] == confirmed.json()["updated_at"]

    client.director_app.state.endpoint_identity = old_endpoint.model_copy(
        update={"runtime_instance_id": "later-comfy-boot"}
    )
    stale_certificate = await client.post(
        f"/api/jobs/{created.json()['id']}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )
    assert stale_certificate.status_code == 409
    assert "different replacement boot" in stale_certificate.json()["detail"]
    assert database.get_prompt_ownership(child["id"]) == released


async def test_startup_handoff_can_be_closed_by_restart_certificate_without_recancel(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("startup-restart-certificate"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent_id = created.json()["id"]
    child = database.list_job_children(parent_id)[0]
    database.update_job(parent_id, status="preparing", stage="submitting")
    database.update_job_child(child["id"], status="preparing", stage="submitting")

    assert database.prepare_interrupted_submissions_for_recovery() == 1
    handed_off = database.get_job(parent_id)
    assert handed_off is not None
    assert handed_off["status"] == "cancelling"
    assert handed_off["cancel_requested"] == 1
    before_io = (
        list(fake_comfy.cancelled),
        fake_comfy.queue_requests,
        list(fake_comfy.history_requests),
    )
    old_endpoint = client.director_app.state.endpoint_identity
    client.director_app.state.endpoint_identity = old_endpoint.model_copy(
        update={"runtime_instance_id": "startup-replacement-boot"}
    )

    await _recover_interrupted_submission(
        _background_request(client.director_app), handed_off
    )
    confirmed = await client.post(
        f"/api/jobs/{parent_id}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )

    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["status"] == "cancelled"
    ownership = database.get_prompt_ownership(child["id"])
    assert ownership is not None and ownership.state == "cleanup_confirmed"
    assert ownership.cleanup_certificate is not None
    assert ownership.cleanup_certificate.kind == "endpoint_restart_certificate"
    assert (
        list(fake_comfy.cancelled),
        fake_comfy.queue_requests,
        list(fake_comfy.history_requests),
    ) == before_io


async def test_user_cancel_terminal_cannot_be_revived_by_stale_submit_cleanup(
    client, fake_comfy
) -> None:
    fake_comfy.submit_error_after_side_effect = ComfyError(
        "response connection lost"
    )
    cleanup_claimed = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def pause_after_cleanup_claim(_job_id, _child_id):
        cleanup_claimed.set()
        await cleanup_release.wait()

    client.director_app.state.before_cleanup_cancel_request = (
        pause_after_cleanup_claim
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cleanup-cancel-race"))},
    )
    await asyncio.wait_for(cleanup_claimed.wait(), timeout=1)
    job_id = created.json()["id"]

    # Cleanup has claimed submission_cancel_pending. User cancellation gets the
    # first exact True acknowledgement and closes both rows; cleanup's later
    # duplicate cancellation therefore returns False and must not revive them.
    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    cleanup_release.set()
    await _wait_for_submission_jobs(client)

    parent = client.director_app.state.database.get_job(job_id)
    child = client.director_app.state.database.list_job_children(job_id)[0]
    assert parent is not None and parent["status"] == "cancelled"
    assert child["status"] == "cancelled"
    assert fake_comfy.cancelled == [child["prompt_id"]]


async def test_user_cancel_intent_wins_after_cleanup_parent_snapshot(
    client, fake_comfy
) -> None:
    fake_comfy.submit_error_after_side_effect = ComfyError(
        "response connection lost"
    )
    finalize_started = asyncio.Event()
    finalize_release = asyncio.Event()

    async def pause_before_finalize(_job_id):
        finalize_started.set()
        await finalize_release.wait()

    client.director_app.state.before_cleanup_finalize = pause_before_finalize
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cleanup-finalize-cancel-race"))},
    )
    await asyncio.wait_for(finalize_started.wait(), timeout=1)
    job_id = created.json()["id"]

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    finalize_release.set()
    await _wait_for_submission_jobs(client)

    parent = client.director_app.state.database.get_job(job_id)
    child = client.director_app.state.database.list_job_children(job_id)[0]
    assert parent is not None and parent["status"] == "cancelled"
    assert child["status"] == "cancelled"


async def test_user_cancel_wins_raylight_control_submission_cleanup(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    resident_timeline = _timeline(_segment("resident-before-control-cleanup", "t2v"))
    resident_timeline["sampling"]["fl2va"]["shift"] = 12.0
    resident = await client.post(
        "/api/timeline/jobs",
        json={"config": resident_timeline},
    )
    assert resident.status_code == 200
    await _wait_for_submission_jobs(client)

    fake_comfy.auto_complete_ray_kill = False
    fake_comfy.submit_error_after_side_effect = ComfyError(
        "DirectorDeckRayKill response connection lost"
    )
    finalize_started = asyncio.Event()
    finalize_release = asyncio.Event()

    async def pause_before_finalize(_job_id):
        finalize_started.set()
        await finalize_release.wait()

    client.director_app.state.before_cleanup_finalize = pause_before_finalize
    switched_timeline = _timeline(_segment("control-cleanup-target", "t2v"))
    switched_timeline["sampling"]["fl2va"]["shift"] = 8.0
    switched = await client.post(
        "/api/timeline/jobs",
        json={"config": switched_timeline},
    )
    assert switched.status_code == 200
    await asyncio.wait_for(finalize_started.wait(), timeout=1)

    cancelled = await client.post(f"/api/jobs/{switched.json()['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    finalize_release.set()
    await _wait_for_submission_jobs(client)

    parent = client.director_app.state.database.get_job(switched.json()["id"])
    children = client.director_app.state.database.list_job_children(
        switched.json()["id"]
    )
    assert parent is not None and parent["status"] == "cancelled"
    assert any(not child["segment_ids"] for child in children)
    assert all(child["status"] == "cancelled" for child in children)
    assert len(fake_comfy.prompts) == 2


@pytest.mark.parametrize(
    "inline_cleanup_detail",
    [
        {"cleanup_response": {"cancelled": False}},
        {"cleanup_error": "inline atomic cancel connection reset"},
    ],
)
async def test_prompt_id_mismatch_rebinds_actual_id_before_outer_cleanup(
    client, fake_comfy, monkeypatch, inline_cleanup_detail
) -> None:
    actual_prompt_id = "actual-comfy-prompt"
    inline_cancel_targets: list[str] = []

    async def mismatched_submit(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        assert isinstance(prompt_id, str) and prompt_id
        fake_comfy.prompts.append(
            {
                "prompt": prompt,
                "client_id": client_id,
                "prompt_id": actual_prompt_id,
            }
        )
        fake_comfy.pending.append([0, actual_prompt_id])
        inline_cancel_targets.append(actual_prompt_id)
        assert on_receipt is not None
        on_receipt(prompt_id, actual_prompt_id)
        raise ComfyError(
            "ComfyUI returned a different prompt id; inline cleanup was not confirmed",
            detail={
                "requested_prompt_id": prompt_id,
                "actual_prompt_id": actual_prompt_id,
                **inline_cleanup_detail,
            },
        )

    monkeypatch.setattr(fake_comfy, "submit", mismatched_submit)

    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("mismatched-id"))},
    )

    assert response.status_code == 200
    await _wait_for_submission_jobs(client)
    assert inline_cancel_targets == [actual_prompt_id]
    assert fake_comfy.cancelled == [actual_prompt_id]
    parent = client.director_app.state.database.list_jobs()[0]
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    assert parent["status"] == "failed"
    assert parent["prompt_id"] == actual_prompt_id
    assert child["status"] == "cancelled"
    assert child["prompt_id"] == actual_prompt_id


async def test_prompt_id_mismatch_with_confirmed_inline_cleanup_is_not_cancelled_twice(
    client, fake_comfy, monkeypatch
) -> None:
    actual_prompt_id = "actual-already-cancelled"
    outer_cancel_targets: list[str] = []

    async def mismatched_submit(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        assert isinstance(prompt_id, str) and prompt_id
        fake_comfy.prompts.append(
            {
                "prompt": prompt,
                "client_id": client_id,
                "prompt_id": actual_prompt_id,
            }
        )
        assert on_receipt is not None
        on_receipt(prompt_id, actual_prompt_id)
        # This is the shape produced by ComfyClient after its same-client
        # atomic cleanup has already removed the unexpected upstream prompt.
        raise ComfyError(
            "ComfyUI returned a different prompt id; unexpected job was cancelled",
            detail={
                "requested_prompt_id": prompt_id,
                "actual_prompt_id": actual_prompt_id,
                "cleanup_response": {"cancelled": True},
            },
        )

    async def record_outer_cancel(prompt_id: str) -> bool:
        outer_cancel_targets.append(prompt_id)
        return False

    monkeypatch.setattr(fake_comfy, "submit", mismatched_submit)
    monkeypatch.setattr(fake_comfy, "cancel", record_outer_cancel)

    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("inline-confirmed"))},
    )

    assert response.status_code == 200
    await _wait_for_submission_jobs(client)
    assert outer_cancel_targets == []
    database = client.director_app.state.database
    parent = database.list_jobs()[0]
    child = database.list_job_children(parent["id"])[0]
    assert parent["status"] == "failed"
    assert parent["stage"] == "submission_failed"
    assert parent["prompt_id"] == actual_prompt_id
    assert child["status"] == "cancelled"
    assert child["stage"] == "cancelled_after_submission_failure"
    assert child["prompt_id"] == actual_prompt_id


@pytest.mark.parametrize("inline_cancelled", [True, False])
async def test_same_prompt_id_receipt_failure_honors_inline_cleanup_once(
    client, fake_comfy, monkeypatch, inline_cancelled: bool
) -> None:
    outer_cancel_targets: list[str] = []
    requested_id: str | None = None

    async def receipt_failure(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        nonlocal requested_id
        assert isinstance(prompt_id, str) and prompt_id
        requested_id = prompt_id
        fake_comfy.prompts.append(
            {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
        )
        if not inline_cancelled:
            fake_comfy.pending.append([0, prompt_id])
        # ComfyClient has already called the hook and converted its failure to
        # this authenticated receipt detail.  The durable ownership row is
        # intentionally still in its pre-receipt submitting state.
        raise ComfyError(
            "ComfyUI submit receipt hook failed",
            detail={
                "requested_prompt_id": prompt_id,
                "actual_prompt_id": prompt_id,
                "receipt_hook_error": "ExecutionEvidenceConflict",
                "cleanup_response": {"cancelled": inline_cancelled},
            },
        )

    original_cancel = fake_comfy.cancel

    async def record_outer_cancel(prompt_id: str) -> bool:
        outer_cancel_targets.append(prompt_id)
        return await original_cancel(prompt_id)

    monkeypatch.setattr(fake_comfy, "submit", receipt_failure)
    monkeypatch.setattr(fake_comfy, "cancel", record_outer_cancel)

    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"same-id-receipt-{inline_cancelled}"))},
    )

    assert response.status_code == 200
    await _wait_for_submission_jobs(client)
    assert requested_id is not None
    assert outer_cancel_targets == ([] if inline_cancelled else [requested_id])
    database = client.director_app.state.database
    parent = database.list_jobs()[0]
    child = database.list_job_children(parent["id"])[0]
    ownership = database.get_prompt_ownership(child["id"])
    assert parent["status"] == "failed"
    assert child["status"] == "cancelled"
    assert ownership is not None and ownership.state == "cleanup_confirmed"
    assert ownership.effective_prompt_id == requested_id


@pytest.mark.parametrize(
    "untrusted_detail",
    ["wrong-requested", "invalid-actual", "stale-durable-requested"],
)
async def test_submit_error_detail_cannot_redirect_cleanup_without_durable_match(
    client, fake_comfy, monkeypatch, untrusted_detail
) -> None:
    observed_requested_id: str | None = None
    expected_cleanup_id: str | None = None

    async def untrusted_submit(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        nonlocal expected_cleanup_id, observed_requested_id
        assert isinstance(prompt_id, str) and prompt_id
        observed_requested_id = prompt_id
        expected_cleanup_id = prompt_id
        fake_comfy.prompts.append(
            {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
        )
        detail = {
            "requested_prompt_id": prompt_id,
            "actual_prompt_id": "untrusted-actual",
        }
        if untrusted_detail == "wrong-requested":
            detail["requested_prompt_id"] = "another-child"
        elif untrusted_detail == "invalid-actual":
            detail["actual_prompt_id"] = "   "
        else:
            # Another owner has already replaced the durable token. The stale
            # exception detail can no longer redirect cleanup away from it.
            expected_cleanup_id = "newer-durable-prompt"
            client.director_app.state.database.update_job_child(
                prompt_id, prompt_id=expected_cleanup_id
            )
        fake_comfy.pending.append([0, expected_cleanup_id])
        raise ComfyError("untrusted detail", detail=detail)

    monkeypatch.setattr(fake_comfy, "submit", untrusted_submit)

    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("untrusted-id"))},
    )

    assert response.status_code == 200
    await _wait_for_submission_jobs(client)
    assert observed_requested_id is not None
    assert expected_cleanup_id is not None
    database = client.director_app.state.database
    parent = database.list_jobs()[0]
    child = database.list_job_children(parent["id"])[0]
    assert child["prompt_id"] == expected_cleanup_id
    if untrusted_detail == "stale-durable-requested":
        # A mutable child projection that disagrees with immutable ownership
        # is corruption, not authority to cancel an arbitrary third-party id.
        assert fake_comfy.cancelled == []
        assert parent["status"] == "cancelling"
        assert child["stage"] == "submission_interrupted"
        ownership = database.get_prompt_ownership(child["id"])
        assert ownership is not None
        assert ownership.effective_prompt_id == observed_requested_id
    else:
        assert fake_comfy.cancelled == [expected_cleanup_id]


@pytest.mark.parametrize(
    "actual_prompt_id",
    ["actual-recovery-prompt", "a" * 300],
    ids=["ordinary-id", "server-id-over-256"],
)
async def test_prompt_id_mismatch_survives_outer_cancel_failure_for_recovery(
    client, fake_comfy, monkeypatch, actual_prompt_id: str
) -> None:
    outer_cancel_targets: list[str] = []
    original_cancel = fake_comfy.cancel

    async def mismatched_submit(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        assert isinstance(prompt_id, str) and prompt_id
        fake_comfy.prompts.append(
            {
                "prompt": prompt,
                "client_id": client_id,
                "prompt_id": actual_prompt_id,
            }
        )
        fake_comfy.pending.append([0, actual_prompt_id])
        assert on_receipt is not None
        on_receipt(prompt_id, actual_prompt_id)
        raise ComfyError(
            "ComfyUI returned a different prompt id; inline cleanup failed",
            detail={
                "requested_prompt_id": prompt_id,
                "actual_prompt_id": actual_prompt_id,
                "cleanup_error": "inline cancellation transport failed",
            },
        )

    async def failing_outer_cancel(prompt_id: str) -> bool:
        outer_cancel_targets.append(prompt_id)
        raise ComfyError("outer cancellation transport failed")

    monkeypatch.setattr(fake_comfy, "submit", mismatched_submit)
    monkeypatch.setattr(fake_comfy, "cancel", failing_outer_cancel)

    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("recover-actual-id"))},
    )

    assert response.status_code == 200
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent = database.list_jobs()[0]
    child = database.list_job_children(parent["id"])[0]
    assert outer_cancel_targets == [actual_prompt_id]
    assert parent["status"] == "cancelling"
    assert parent["prompt_id"] == actual_prompt_id
    assert child["status"] == "cancelling"
    assert child["stage"] == "submission_cancel_failed"
    assert child["prompt_id"] == actual_prompt_id

    # A later process-owned recovery pass must retain and target the rebound
    # id, never the caller-assigned id that this ComfyUI ignored.
    monkeypatch.setattr(fake_comfy, "cancel", original_cancel)
    app = client.director_app
    app.state.reconcile_interval_seconds = 0.01
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())
    async with app.router.lifespan_context(app):
        for _ in range(100):
            persisted = database.get_job(parent["id"])
            if persisted is not None and persisted["status"] == "cancelled":
                break
            await asyncio.sleep(0.01)

    persisted = database.get_job(parent["id"])
    child = database.get_job_child(child["id"])
    assert persisted is not None and persisted["status"] == "cancelled"
    assert child is not None and child["status"] == "cancelled"
    assert child["prompt_id"] == actual_prompt_id
    assert fake_comfy.cancelled == [actual_prompt_id]


async def test_status_sync_cannot_steal_live_submit_ownership(
    client, fake_comfy
) -> None:
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    request = asyncio.create_task(
        client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("owned-by-submit"))},
        )
    )
    await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
    database = client.director_app.state.database
    parent = database.list_jobs()[0]
    child = database.list_job_children(parent["id"])[0]
    assert child["stage"] == "submitting"
    fake_comfy.pending = []
    fake_comfy.histories[child["prompt_id"]] = _success(child)

    polled = await client.get(f"/api/jobs/{parent['id']}")
    assert polled.status_code == 200
    assert polled.json()["children"][0]["stage"] == "submitting"
    assert database.list_job_children(parent["id"])[0]["status"] == "preparing"

    fake_comfy.submit_release.set()
    submitted = await asyncio.wait_for(request, timeout=1)

    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["status"] == "preparing"
    await _wait_for_submission_jobs(client)
    submitted_job = await client.get(f"/api/jobs/{parent['id']}")
    assert submitted_job.json()["status"] in {"queued", "running", "succeeded"}
    assert submitted_job.json()["children"][0]["status"] in {
        "queued",
        "running",
        "succeeded",
    }


async def test_concurrent_cancel_requests_share_priority_reconciliation(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancel-once"))},
    )
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 1)
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    fake_comfy.pending = []
    fake_comfy.histories[str(child["prompt_id"])] = {
        "status": {
            "status_str": "error",
            "completed": True,
            "messages": [
                ["execution_interrupted", {"prompt_id": child["prompt_id"]}]
            ],
        },
        "outputs": {},
    }
    fake_comfy.history_started = asyncio.Event()
    fake_comfy.history_release = asyncio.Event()

    first = asyncio.create_task(client.post(f"/api/jobs/{parent['id']}/cancel"))
    await asyncio.wait_for(fake_comfy.history_started.wait(), timeout=1)
    second = asyncio.create_task(client.post(f"/api/jobs/{parent['id']}/cancel"))
    await asyncio.sleep(0)
    fake_comfy.history_release.set()
    first_response, second_response = await asyncio.gather(first, second)

    assert first_response.status_code == second_response.status_code == 200
    assert first_response.json()["status"] == "cancelled"
    assert second_response.json()["status"] == "cancelled"


async def test_cancel_replaces_existing_normal_no_assembly_sync_before_terminal_probe(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("normal-flight-before-cancel"))},
    )
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent = database.get_job(created.json()["id"])
    child = database.list_job_children(created.json()["id"])[0]
    assert parent is not None
    fake_comfy.pending = []
    fake_comfy.histories[str(child["prompt_id"])] = _success(child)
    ordinary_started = asyncio.Event()
    ordinary_release = asyncio.Event()

    async def ordinary_no_assembly_sync():
        ordinary_started.set()
        await ordinary_release.wait()
        latest = database.get_job(parent["id"])
        assert latest is not None
        return await director_app_module._sync_timeline_job_once(
            _background_request(client.director_app),
            latest,
            allow_assembly=False,
            honor_cancel_intent=True,
        )

    async with client.director_app.state.timeline_sync_lock:
        director_app_module._register_timeline_sync_task(
            _background_request(client.director_app),
            parent["id"],
            ordinary_no_assembly_sync(),
            allow_assembly=False,
            honor_cancel_intent=True,
        )
    await asyncio.wait_for(ordinary_started.wait(), timeout=1)

    cancellation = asyncio.create_task(
        client.post(f"/api/jobs/{parent['id']}/cancel")
    )
    await asyncio.sleep(0)
    ordinary_release.set()
    response = await asyncio.wait_for(cancellation, timeout=1)

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"
    assert str(child["prompt_id"]) not in fake_comfy.cancelled


@pytest.mark.parametrize(
    ("with_control", "segment_stage", "control_stage"),
    [
        (False, "cancelled", "cancelled"),
        (True, "cancelled", "cancelled"),
        (
            True,
            "restart_cancelled_not_submitted",
            "cancelled_after_restart",
        ),
    ],
)
async def test_interrupted_explicit_cancel_retry_preserves_cancelled_children(
    client,
    fake_comfy,
    with_control: bool,
    segment_stage: str,
    control_stage: str,
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"cancel-retry-{with_control}"))},
    )
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent = database.get_job(created.json()["id"])
    child = database.list_job_children(created.json()["id"])[0]
    assert parent is not None
    marked, first_claim = database.mark_job_cancel_requested(parent["id"])
    assert marked is not None and first_claim
    database.update_job(parent["id"], status="cancelling", stage="cancelling")
    ownership = database.get_prompt_ownership(child["id"])
    assert ownership is not None
    confirmed_at = datetime.now(timezone.utc)
    released = database.confirm_prompt_cleanup(
        child["id"],
        expected_revision=ownership.ownership_revision,
        evidence=ExactCancelConfirmedEvidence(
            prompt_id=ownership.effective_prompt_id,
            confirmation_id=(
                f"test-exact-cancel:{ownership.effective_prompt_id}"
            ),
            confirmed_at=confirmed_at,
        ),
        stage=segment_stage,
        updated_at=confirmed_at,
        completed_at=confirmed_at.isoformat(),
    )
    assert released is not None
    fake_comfy.pending = [
        item
        for item in fake_comfy.pending
        if ownership.effective_prompt_id not in item
    ]
    if with_control:
        now = utc_now()
        control_index = max(
            item["group_index"]
            for item in database.list_job_children(parent["id"])
        ) + 1
        database.create_job_child(
            {
                "id": f"retry-control-{parent['id']}",
                "job_id": parent["id"],
                "group_index": control_index,
                "family": "fl2va",
                "backend": "raylight",
                "segment_ids": [],
                "output_nodes": {},
                "status": "cancelled",
                "progress": 1.0,
                "stage": control_stage,
                "prompt_id": f"retry-control-{parent['id']}",
                "outputs": [],
                "error": None,
                "prompt_snapshot": {},
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": now,
            }
        )

    retried = await client.post(f"/api/jobs/{parent['id']}/cancel")
    assert retried.status_code == 200, retried.text
    assert retried.json()["status"] == "cancelled"
    # A third click is a terminal idempotent read and must not depend on a
    # newly-bound local variable in the first-claim branch.
    repeated = await client.post(f"/api/jobs/{parent['id']}/cancel")
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "cancelled"
    assert all(
        item["status"] == "cancelled"
        for item in database.list_job_children(parent["id"])
    )


async def test_control_closure_reloads_cancel_intent_after_upstream_wait(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("stale-control-cancel-intent"))},
    )
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent = database.get_job(created.json()["id"])
    segment = database.list_job_children(created.json()["id"])[0]
    assert parent is not None
    now = utc_now()
    database.update_job_child(
        segment["id"],
        status="succeeded",
        progress=1.0,
        stage="completed",
        outputs=[{"node_id": "20", "filename": "done.mp4", "subfolder": "", "type": "output"}],
        completed_at=now,
    )
    next_index = max(
        item["group_index"]
        for item in database.list_job_children(parent["id"])
    ) + 1
    control = database.create_job_child(
        {
            "id": f"stale-control-{parent['id']}",
            "job_id": parent["id"],
            "group_index": next_index,
            "family": "fl2va",
            "backend": "raylight",
            "segment_ids": [],
            "output_nodes": {},
            "status": "queued",
            "progress": 0.0,
            "stage": "queued",
            "prompt_id": f"stale-control-{parent['id']}",
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    target = database.create_job_child(
        {
            "id": f"stale-target-{parent['id']}",
            "job_id": parent["id"],
            "group_index": next_index + 1,
            "family": "ref2va",
            "backend": "raylight",
            "segment_ids": ["never-submitted"],
            "output_nodes": {"never-submitted": "20"},
            "status": "preparing",
            "progress": 0.0,
            "stage": "preflight",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    batch_waiting = asyncio.Event()
    batch_release = asyncio.Event()

    async def interleaved_batch(_request, _job, _children):
        batch_waiting.set()
        await batch_release.wait()
        return database.list_job_children(parent["id"])

    monkeypatch.setattr(
        director_app_module, "_sync_timeline_children_batch", interleaved_batch
    )
    reconciliation = asyncio.create_task(
        director_app_module._sync_timeline_job_once(
            _background_request(client.director_app), parent, allow_assembly=False
        )
    )
    await asyncio.wait_for(batch_waiting.wait(), timeout=1)
    marked, first_claim = database.mark_job_cancel_requested(parent["id"])
    assert marked is not None and first_claim
    database.update_job(parent["id"], status="cancelling", stage="cancelling")
    database.update_job_child(
        control["id"],
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        completed_at=utc_now(),
    )
    batch_release.set()
    reconciled = await asyncio.wait_for(reconciliation, timeout=1)

    assert reconciled["status"] == "cancelled"
    closed_target = database.get_job_child(target["id"])
    assert closed_target is not None
    assert closed_target["status"] == "cancelled"
    assert closed_target["stage"] == "cancelled"


async def test_restart_recovery_terminal_cannot_be_revived_by_stale_user_cancel(
    client, fake_comfy, monkeypatch
) -> None:
    await _exercise_cancel_recovery_terminal_race(
        client, fake_comfy, monkeypatch, paused_actor="user-cancel"
    )


async def test_user_cancel_terminal_cannot_be_revived_by_stale_restart_recovery(
    client, fake_comfy, monkeypatch
) -> None:
    await _exercise_cancel_recovery_terminal_race(
        client, fake_comfy, monkeypatch, paused_actor="restart-recovery"
    )


async def test_reconciler_rotates_exceptional_parents_past_bounded_batch(
    client, fake_comfy, monkeypatch
) -> None:
    parent_ids: list[str] = []
    for index in range(5):
        response = await client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment(f"fair-{index}"))},
        )
        assert response.status_code == 200, response.text
        parent_ids.append(response.json()["id"])

    # POST returns after durable acceptance while dispatch continues in a
    # managed task.  Let those writers settle so this test isolates bounded
    # reconciler rotation instead of racing submission-owned parent updates.
    await _wait_for_submission_jobs(client)

    seen: list[str] = []
    reached_fifth = asyncio.Event()

    async def poison_then_observe(_request, snapshot, **_kwargs):
        job_id = str(snapshot["id"])
        seen.append(job_id)
        if job_id in parent_ids[:4]:
            raise ValueError("poisoned parent")
        reached_fifth.set()
        return snapshot

    app = client.director_app
    app.state.reconcile_batch_size = 4
    app.state.reconcile_interval_seconds = 0.001
    monkeypatch.setattr("directordeck.app._sync_timeline_job", poison_then_observe)
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(reached_fifth.wait(), timeout=1)

    assert set(seen[:4]) == set(parent_ids[:4])
    assert parent_ids[4] in seen[4:]


async def test_lifespan_shutdown_cancels_and_clears_managed_sync_flights(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("shutdown"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    fake_comfy.pending = []
    fake_comfy.history_started = asyncio.Event()
    fake_comfy.history_release = asyncio.Event()
    app = client.director_app
    app.state.reconcile_interval_seconds = 0.001
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(fake_comfy.history_started.wait(), timeout=1)
        assert app.state.timeline_sync_tasks
        assert app.state.timeline_sync_all_tasks

    assert app.state.timeline_sync_tasks == {}
    assert app.state.timeline_sync_all_tasks == set()


@pytest.mark.parametrize(
    ("cancel_error", "expected_stage"),
    [
        (ComfyError("cancel endpoint unavailable"), "submission_cancel_failed"),
        (None, "submission_cancel_unconfirmed"),
    ],
)
@pytest.mark.parametrize("submit_error_kind", ["comfy", "timeout"])
async def test_ambiguous_submit_cleanup_never_claims_failure_without_confirmed_cancel(
    client, fake_comfy, cancel_error, expected_stage, submit_error_kind
) -> None:
    fake_comfy.submit_error_after_side_effect = (
        ComfyError("response connection lost")
        if submit_error_kind == "comfy"
        else httpx.ReadTimeout("prompt response timed out")
    )
    if cancel_error is not None:
        fake_comfy.cancel_error = cancel_error
    else:
        original_cancel = fake_comfy.cancel

        async def unconfirmed(prompt_id: str) -> bool:
            await original_cancel(prompt_id)
            # Model an atomic endpoint that cannot confirm ownership despite
            # the prompt still being potentially accepted upstream.
            return False

        fake_comfy.cancel = unconfirmed

    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ambiguous"))},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "preparing"
    await _wait_for_submission_jobs(client)
    parent = client.director_app.state.database.list_jobs()[0]
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    assert parent["status"] == "cancelling"
    assert parent["stage"] == "submission_cancel_pending"
    assert parent["completed_at"] is None
    assert child["status"] == "cancelling"
    assert child["stage"] == expected_stage
    assert child["completed_at"] is None


async def test_ambiguous_submit_uncertainty_stays_owned_until_late_prompt_is_cancelled(
    client, fake_comfy, monkeypatch
) -> None:
    fake_comfy.submit_error_after_side_effect = ComfyError(
        "response connection lost"
    )
    original_cancel = fake_comfy.cancel
    cancel_attempts = 0

    async def delayed_cancel(prompt_id: str) -> bool:
        nonlocal cancel_attempts
        cancel_attempts += 1
        if cancel_attempts == 1:
            # The ambiguous cleanup runs while the old /prompt side effect is
            # not yet visible to the atomic cancel endpoint.
            fake_comfy.pending = []
            return False
        return await original_cancel(prompt_id)

    monkeypatch.setattr(fake_comfy, "cancel", delayed_cancel)
    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("late-ambiguous"))},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "preparing"
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent = database.list_jobs()[0]
    child = database.list_job_children(parent["id"])[0]
    assert child["stage"] == "submission_cancel_unconfirmed"

    # A normal background pass sees queue/history absence but must respect the
    # recovery ownership marker instead of claiming a local terminal state.
    await _reconcile(client, parent)
    child = database.get_job_child(child["id"])
    assert child is not None and child["status"] == "cancelling"
    assert child["stage"] == "submission_cancel_unconfirmed"

    # Model the original /prompt reaching queue.put after that absent poll.
    fake_comfy.pending = [[0, child["prompt_id"]]]
    app = client.director_app
    app.state.reconcile_interval_seconds = 0.01
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())
    async with app.router.lifespan_context(app):
        for _ in range(100):
            parent = database.get_job(parent["id"])
            if parent is not None and parent["status"] == "cancelled":
                break
            await asyncio.sleep(0.01)

    parent = database.get_job(parent["id"])
    child = database.get_job_child(child["id"])
    assert parent is not None and parent["status"] == "cancelled"
    assert child is not None and child["status"] == "cancelled"
    assert cancel_attempts >= 2
    assert fake_comfy.cancelled == [child["prompt_id"]]


async def test_cancel_during_submit_cannot_make_parent_deletable_before_owner_returns(
    client, fake_comfy
) -> None:
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    request = asyncio.create_task(
        client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("owned"))},
        )
    )
    await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
    parent = client.director_app.state.database.list_jobs()[0]

    cancelled = await client.post(f"/api/jobs/{parent['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelling"
    child = cancelled.json()["children"][0]
    assert child["status"] == "cancelling"
    assert child["stage"] == "cancelling_during_submit"
    assert (await client.delete(f"/api/jobs/{parent['id']}")).status_code == 409
    cleared = (await client.delete("/api/jobs")).json()
    assert cleared["deleted_count"] == 0
    assert cleared["active_count"] == 1

    fake_comfy.submit_release.set()
    completed = await asyncio.wait_for(request, timeout=1)

    assert completed.status_code == 200
    assert completed.json()["status"] == "preparing"
    await _wait_for_submission_jobs(client)
    stored = await client.get(f"/api/jobs/{parent['id']}")
    assert stored.json()["status"] == "cancelled"
    assert client.director_app.state.database.get_job(parent["id"])["status"] == "cancelled"


async def test_cancel_during_submit_targets_the_late_actual_prompt_id(
    client, fake_comfy, monkeypatch
) -> None:
    submit_started = asyncio.Event()
    submit_release = asyncio.Event()
    actual_prompt_id = "late-actual-cancel-target"

    async def delayed_actual_submit(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        assert isinstance(prompt_id, str) and prompt_id
        submit_started.set()
        await submit_release.wait()
        fake_comfy.prompts.append(
            {
                "prompt": prompt,
                "client_id": client_id,
                "prompt_id": actual_prompt_id,
            }
        )
        fake_comfy.pending.append([0, actual_prompt_id])
        assert on_receipt is not None
        on_receipt(prompt_id, actual_prompt_id)
        return {"prompt_id": actual_prompt_id, "number": 1, "node_errors": {}}

    monkeypatch.setattr(fake_comfy, "submit", delayed_actual_submit)
    request = asyncio.create_task(
        client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("late-actual-owner"))},
        )
    )
    await asyncio.wait_for(submit_started.wait(), timeout=1)
    database = client.director_app.state.database
    parent = database.list_jobs()[0]

    cancelling = await client.post(f"/api/jobs/{parent['id']}/cancel")

    assert cancelling.status_code == 200
    assert cancelling.json()["status"] == "cancelling"
    assert fake_comfy.cancelled == []
    submit_release.set()
    assert (await asyncio.wait_for(request, timeout=1)).status_code == 200
    await _wait_for_submission_jobs(client)

    child = database.list_job_children(parent["id"])[0]
    ownership = database.get_prompt_ownership(child["id"])
    assert database.get_job(parent["id"])["status"] == "cancelled"
    assert child["status"] == "cancelled"
    assert child["prompt_id"] == actual_prompt_id
    assert fake_comfy.cancelled == [actual_prompt_id]
    assert ownership is not None and ownership.state == "cleanup_confirmed"
    assert ownership.effective_prompt_id == actual_prompt_id
    assert ownership.cleanup_certificate is not None
    assert ownership.cleanup_certificate.prompt_id == actual_prompt_id


@pytest.mark.parametrize("late_cleanup", ["false", "error"])
async def test_late_submit_response_never_revives_confirmed_terminal_cancel(
    client, fake_comfy, monkeypatch, late_cleanup: str
) -> None:
    submit_visible = asyncio.Event()
    submit_response_release = asyncio.Event()

    async def submit_before_response(
        prompt, client_id, prompt_id=None, *, on_receipt=None
    ):
        assert isinstance(prompt_id, str) and prompt_id
        fake_comfy.prompts.append(
            {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
        )
        fake_comfy.pending.append([0, prompt_id])
        submit_visible.set()
        await submit_response_release.wait()
        assert on_receipt is not None
        on_receipt(prompt_id, prompt_id)
        return {"prompt_id": prompt_id, "number": 1, "node_errors": {}}

    monkeypatch.setattr(fake_comfy, "submit", submit_before_response)
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"late-response-{late_cleanup}"))},
    )
    assert created.status_code == 200
    await asyncio.wait_for(submit_visible.wait(), timeout=1)
    database = client.director_app.state.database
    child = database.list_job_children(created.json()["id"])[0]
    prompt_id = str(child["prompt_id"])

    # Model caller-ID execution becoming observable before POST /prompt sends
    # its HTTP response. This removes the submission-ownership stage, so the
    # user's exact directed cancel can positively finish both lifecycle rows.
    await client.director_app.state.progress_manager._sink(
        "http://comfy.test:8188",
        ComfyExecutionEvent(prompt_id=prompt_id, node_id=None),
    )
    running = database.get_job_child(child["id"])
    assert running is not None and running["status"] == "running"

    cancelled = await asyncio.wait_for(
        client.post(f"/api/jobs/{created.json()['id']}/cancel"), timeout=3
    )
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    terminal_parent = database.get_job(created.json()["id"])
    terminal_child = database.get_job_child(child["id"])
    assert terminal_parent is not None and terminal_parent["status"] == "cancelled"
    assert terminal_child is not None and terminal_child["status"] == "cancelled"

    # The submitter performs one redundant exact cleanup after receiving its
    # delayed response. Neither a normal False nor a transport error may turn
    # the already-confirmed terminal rows back into cancelling.
    if late_cleanup == "error":
        fake_comfy.cancel_error = ComfyError("redundant cleanup unavailable")
    submit_response_release.set()
    await _wait_for_submission_jobs(client)

    stored_parent = database.get_job(created.json()["id"])
    stored_child = database.get_job_child(child["id"])
    assert stored_parent is not None and stored_parent["status"] == "cancelled"
    assert stored_child is not None and stored_child["status"] == "cancelled"
    assert stored_parent["updated_at"] == terminal_parent["updated_at"]
    assert stored_child["updated_at"] == terminal_child["updated_at"]


async def test_preflight_failure_closure_cannot_overwrite_concurrent_child_success(
    client, fake_comfy, monkeypatch
) -> None:
    database = client.director_app.state.database

    async def fail_before_claim(_job_id, _child_id):
        raise director_app_module.HTTPException(
            status_code=409, detail="forced preflight race"
        )

    client.director_app.state.before_submission_claim = fail_before_claim
    original_list_children = database.list_job_children
    injected_terminal = False
    preserved_output = {
        "node_id": "20",
        "filename": "completed-before-preflight-closure.mp4",
        "subfolder": "segments",
        "type": "output",
    }

    def list_children_with_concurrent_success(job_id):
        nonlocal injected_terminal
        snapshots = original_list_children(job_id)
        parent = database.get_job(job_id)
        if (
            not injected_terminal
            and parent is not None
            and parent["status"] == "failed"
            and snapshots
        ):
            injected_terminal = True
            database.update_job_child(
                snapshots[0]["id"],
                status="succeeded",
                progress=1.0,
                stage="completed",
                outputs=[preserved_output],
                error=None,
                completed_at=utc_now(),
            )
        # Return the deliberately stale snapshots. The failure closure must
        # use their row versions rather than overwriting the concurrent result.
        return snapshots

    monkeypatch.setattr(database, "list_job_children", list_children_with_concurrent_success)
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("preflight-closure-cas"))},
    )
    assert created.status_code == 200
    await _wait_for_submission_jobs(client)

    parent = database.get_job(created.json()["id"])
    child = original_list_children(created.json()["id"])[0]
    assert injected_terminal
    assert parent is not None and parent["status"] == "failed"
    assert child["status"] == "succeeded"
    assert child["stage"] == "completed"
    assert child["outputs"] == [preserved_output]


async def test_cancel_between_parent_read_and_atomic_child_claim_never_submits(
    client, fake_comfy, monkeypatch
) -> None:
    database = client.director_app.state.database
    claim_started = asyncio.Event()
    claim_release = asyncio.Event()

    async def paused_claim(_job_id, _child_id):
        claim_started.set()
        await claim_release.wait()

    client.director_app.state.before_submission_claim = paused_claim
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancel-before-claim", "t2v"))},
    )
    await asyncio.wait_for(claim_started.wait(), timeout=1)
    job_id = created.json()["id"]
    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    claim_release.set()
    await _wait_for_submission_jobs(client)
    assert fake_comfy.prompts == []
    parent = database.get_job(job_id)
    children = database.list_job_children(job_id)
    assert parent is not None and parent["status"] == "cancelled"
    assert children and all(child["status"] == "cancelled" for child in children)


async def test_cancel_failure_during_submit_keeps_owner_and_parent_recoverable(
    client, fake_comfy
) -> None:
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    fake_comfy.cancel_error = ComfyError("cancel endpoint unavailable")
    request = asyncio.create_task(
        client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("owned"))},
        )
    )
    await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
    parent = client.director_app.state.database.list_jobs()[0]

    cancelled = await client.post(f"/api/jobs/{parent['id']}/cancel")
    assert cancelled.json()["status"] == "cancelling"
    assert cancelled.json()["children"][0]["stage"] == "cancelling_during_submit"
    assert (await client.delete(f"/api/jobs/{parent['id']}")).status_code == 409

    fake_comfy.submit_release.set()
    completed = await asyncio.wait_for(request, timeout=1)

    assert completed.status_code == 200
    assert completed.json()["status"] == "preparing"
    await _wait_for_submission_jobs(client)
    persisted = client.director_app.state.database.get_job(parent["id"])
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    assert persisted is not None and persisted["status"] == "cancelling"
    assert persisted["completed_at"] is None
    assert child["status"] == "cancelling"
    assert child["completed_at"] is None


async def test_lifespan_reconciler_completes_segments_without_browser_get(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("background"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    child = client.director_app.state.database.list_job_children(parent["id"])[0]
    fake_comfy.pending = []
    fake_comfy.histories[child["prompt_id"]] = _success(child)
    app = client.director_app
    app.state.reconcile_interval_seconds = 0.01
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    close = AsyncMock()
    monkeypatch.setattr(app.state.progress_manager, "close", close)

    async with app.router.lifespan_context(app):
        for _ in range(100):
            stored = app.state.database.get_job(parent["id"])
            if stored is not None and stored["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)

    stored = app.state.database.get_job(parent["id"])
    assert stored is not None
    assert stored["status"] == "succeeded"
    assert stored["stage"] == "segments_completed"
    close.assert_awaited_once_with()


async def test_websocket_hint_wakes_reconciler_before_periodic_timeout(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("event-wake"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(parent["id"])[0]
    app = client.director_app
    app.state.reconcile_interval_seconds = 60.0
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())

    async with app.router.lifespan_context(app):
        for _ in range(100):
            if fake_comfy.queue_requests > 0:
                break
            await asyncio.sleep(0.005)
        assert fake_comfy.queue_requests > 0
        baseline_queue_requests = fake_comfy.queue_requests

        fake_comfy.pending = []
        fake_comfy.histories[child["prompt_id"]] = _success(child)
        await app.state.progress_manager._reconcile_sink(
            "http://comfy.test:8188",
            ComfyReconcileHint(
                event_type="executing", prompt_id=child["prompt_id"]
            ),
        )

        for _ in range(100):
            stored = database.get_job(parent["id"])
            if stored is not None and stored["status"] == "succeeded":
                break
            await asyncio.sleep(0.005)

    stored = database.get_job(parent["id"])
    assert stored is not None and stored["status"] == "succeeded"
    assert fake_comfy.queue_requests > baseline_queue_requests


async def test_restart_reconciler_resumes_full_timeline_assembly(
    client, fake_comfy, monkeypatch
) -> None:
    document = _timeline(
        _segment("first"), _segment("second"), export_mode="all"
    )
    created = await client.post("/api/timeline/jobs", json={"config": document})
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 2)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    children = database.list_job_children(parent["id"])
    for child in children:
        _complete_fake_prompt(fake_comfy, child)
    reconciled = await _reconcile(client, parent, allow_assembly=False)
    assert all(
        child["status"] == "succeeded" for child in reconciled["children"]
    )
    database.update_job(
        parent["id"],
        status="running",
        progress=1.0,
        # Simulate process death after the durable single-flight claim. Startup
        # must release this stale claim and reuse persisted evidence, never the
        # compiler, before retrying the assembly.
        stage="assembling",
        outputs=[],
        completed_at=None,
    )

    restarted = create_app(
        comfy_url="http://comfy.test:8188",
        database_path=database.path,
        comfy_factory=lambda _comfy_url: fake_comfy,
        host_output_probe=fake_comfy,
    )
    restarted.state.reconcile_interval_seconds = 0.01
    monkeypatch.setattr(restarted.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(restarted.state.progress_manager, "close", AsyncMock())
    assembled = VideoProxy(
        content=b"background-assembled",
        filename_suffix=".mp4",
        metadata=VideoMetadata.model_validate(VIDEO_METADATA),
    )
    assemble = Mock(return_value=assembled)
    monkeypatch.setattr("directordeck.app.assemble_video_bytes", assemble)
    compiler_recovery = Mock(
        side_effect=AssertionError(
            "restart assembly must use the persisted compiled plan"
        )
    )
    monkeypatch.setattr(
        director_app_module,
        "compile_project_execution_plan",
        compiler_recovery,
    )

    async with restarted.router.lifespan_context(restarted):
        for _ in range(100):
            stored = restarted.state.database.get_job(parent["id"])
            if stored is not None and stored["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)

    stored = restarted.state.database.get_job(parent["id"])
    assert stored is not None
    assert stored["status"] == "succeeded"
    assert stored["stage"] == "completed"
    assert len(stored["outputs"]) == 1
    assert stored["outputs"][0]["node_id"] == "assembly"
    evidence = restarted.state.database.get_observed_assembly_artifact(
        parent["id"]
    )
    assert evidence is not None
    assert evidence.output_descriptor.filename == stored["outputs"][0]["filename"]
    assert evidence.frame_count == VIDEO_METADATA["frame_count"]
    assert evidence.media_probe_version == VIDEO_METADATA["probe_method"]
    assert [source.segment_id for source in evidence.source_artifacts] == [
        "first",
        "second",
    ]
    assemble.assert_called_once()
    compiler_recovery.assert_not_called()


async def test_multi_all_parent_output_uses_assembly_evidence_for_every_api(
    client, fake_comfy, monkeypatch
) -> None:
    document = _timeline(
        _segment("authority-first"),
        _segment("authority-second"),
        export_mode="all",
    )
    created = await client.post("/api/timeline/jobs", json={"config": document})
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 2)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    for child in database.list_job_children(parent["id"]):
        _complete_fake_prompt(fake_comfy, child)
    monkeypatch.setattr(
        director_app_module,
        "assemble_video_bytes",
        Mock(
            return_value=VideoProxy(
                content=b"authority-assembly",
                filename_suffix=".mp4",
                metadata=VideoMetadata.model_validate(VIDEO_METADATA),
            )
        ),
    )

    reconciled = await _reconcile(client, parent, allow_assembly=True)
    assert reconciled["status"] == "succeeded"
    evidence = database.get_observed_assembly_artifact(parent["id"])
    assert evidence is not None
    trusted_filename = evidence.output_descriptor.filename
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET outputs = ? WHERE id = ?",
            (
                json.dumps(
                    [
                        {
                            "node_id": "assembly",
                            "filename": "forged-assembly.mp4",
                            "subfolder": "attacker",
                            "type": "output",
                        }
                    ]
                ),
                parent["id"],
            ),
        )

    viewed: list[dict[str, str]] = []
    original_view = fake_comfy.view

    async def capture_view(params: dict[str, str]):
        viewed.append(dict(params))
        return await original_view(params)

    monkeypatch.setattr(fake_comfy, "view", capture_view)
    detail = await client.get(f"/api/jobs/{parent['id']}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["output_files"] == [
        f"output/directordeck/timelines/{trusted_filename}"
    ]
    proxied = await client.get(f"/api/jobs/{parent['id']}/outputs/0")
    assert proxied.status_code == 200, proxied.text
    monkeypatch.setattr(
        "directordeck.task_management.create_24fps_proxy_bytes",
        lambda _content, _suffix: VideoProxy(
            content=b"normalized-assembly",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        ),
    )
    imported = await client.post(
        f"/api/jobs/{parent['id']}/import-output",
        json={"output_index": 0},
    )
    assert imported.status_code == 200, imported.text
    assert viewed == [
        evidence.output_descriptor.model_dump(mode="json"),
        evidence.output_descriptor.model_dump(mode="json"),
    ]

    with database.connect() as connection:
        connection.execute(
            "DROP TRIGGER job_observed_assembly_artifacts_immutable"
        )
        connection.execute(
            "UPDATE job_observed_assembly_artifacts "
            "SET observed_assembly_artifact_digest = ? WHERE job_id = ?",
            ("sha256-" + "0" * 64, parent["id"]),
        )
    hidden = await client.get(f"/api/jobs/{parent['id']}")
    assert hidden.status_code == 200, hidden.text
    assert hidden.json()["outputs"] == []
    assert hidden.json()["output_files"] == []
    assert (
        await client.get(f"/api/jobs/{parent['id']}/outputs/0")
    ).status_code == 404
    refused = await client.post(
        f"/api/jobs/{parent['id']}/import-output",
        json={"output_index": 0},
    )
    assert refused.status_code == 404
    assert len(viewed) == 2


async def test_assembly_finalize_cas_does_not_revive_cancelled_parent(
    client, fake_comfy, monkeypatch
) -> None:
    document = _timeline(
        _segment("cancel-assembly-first"),
        _segment("cancel-assembly-second"),
        export_mode="all",
    )
    created = await client.post("/api/timeline/jobs", json={"config": document})
    assert created.status_code == 200, created.text
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 2)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    for child in database.list_job_children(parent["id"]):
        _complete_fake_prompt(fake_comfy, child)
    ready = await _reconcile(client, parent, allow_assembly=False)
    assert ready["stage"] == "segments_ready"
    assembly_started = asyncio.Event()
    assembly_release = asyncio.Event()

    async def paused_assembly(_request, _claimed, artifacts):
        assert isinstance(artifacts, tuple)
        assert len(artifacts) == 2
        assert all(isinstance(item, ObservedArtifactSpec) for item in artifacts)
        assembly_started.set()
        await assembly_release.wait()
        return (
            OutputDescriptor(
                filename="cancel-race-assembly.mp4",
                subfolder="directordeck/timelines",
            ),
            VideoMetadata.model_validate(VIDEO_METADATA),
            "sha256:" + "b" * 64,
        )

    monkeypatch.setattr(
        director_app_module,
        "_assemble_timeline_output",
        paused_assembly,
    )
    flight = asyncio.create_task(
        _reconcile(client, ready, allow_assembly=True)
    )
    await asyncio.wait_for(assembly_started.wait(), timeout=1)
    marked, first_claim = database.mark_job_cancel_requested(parent["id"])
    assert marked is not None and first_claim
    assembly_release.set()
    settled = await asyncio.wait_for(flight, timeout=1)

    assert settled["status"] != "succeeded"
    assert settled["cancel_requested"] == 1
    assert database.get_observed_assembly_artifact(parent["id"]) is None
    durable = database.get_job(parent["id"])
    assert durable is not None
    assert durable["status"] != "succeeded"
    assert durable["outputs"] == []


async def test_runtime_recovery_selector_requires_explicit_parent_handoff(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("selector-owner"))},
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    database = client.director_app.state.database
    await _wait_until(lambda: bool(database.list_job_children(parent["id"])))
    child = database.list_job_children(parent["id"])[0]

    database.update_job(parent["id"], status="preparing", stage="submitting")
    database.update_job_child(
        child["id"], status="cancelling", stage="cancelling_during_submit"
    )
    assert database.list_interrupted_preparing_jobs() == []

    # Cleanup writes its child marker before publishing the parent handoff.
    # The persistent worker must not race that current-process cleanup owner.
    database.update_job_child(
        child["id"], status="cancelling", stage="submission_cancel_pending"
    )
    assert database.list_interrupted_preparing_jobs() == []

    database.update_job(
        parent["id"], status="cancelling", stage="submission_cancel_pending"
    )
    assert [row["id"] for row in database.list_interrupted_preparing_jobs()] == [
        parent["id"]
    ]


async def test_runtime_recovery_never_cancels_live_preflight_or_submit(
    tmp_path, fake_comfy, monkeypatch
) -> None:
    app = create_app(
        comfy_url="http://comfy.test:8188",
        database_path=tmp_path / "live-submit.sqlite3",
        comfy_factory=lambda _comfy_url: fake_comfy,
        host_capability_provider=FakeHostCapabilityProvider(fake_comfy),
        host_output_probe=fake_comfy,
    )
    database = app.state.database
    database.initialize()
    save_database_legacy_settings(database, default_settings())
    app.state.reconcile_interval_seconds = 0.01
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())
    compile_started = asyncio.Event()
    compile_release = asyncio.Event()
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    cancel_calls = 0
    original_cancel = fake_comfy.cancel

    async def count_cancel(prompt_id: str) -> bool:
        nonlocal cancel_calls
        cancel_calls += 1
        return await original_cancel(prompt_id)

    monkeypatch.setattr(fake_comfy, "cancel", count_cancel)

    async def before_timeline_compile(_job_id: str) -> None:
        compile_started.set()
        await compile_release.wait()

    monkeypatch.setattr(
        app.state,
        "before_timeline_compile",
        before_timeline_compile,
        raising=False,
    )

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as http:
            submission = asyncio.create_task(
                http.post(
                    "/api/timeline/jobs",
                    json={
                        "config": v5_timeline_fixture(
                            _timeline(_segment("live-owner"))
                        )
                    },
                )
            )
            await asyncio.wait_for(compile_started.wait(), timeout=1)
            await asyncio.sleep(0.04)
            parent = database.list_jobs()[0]
            assert parent["status"] == "preparing"
            assert cancel_calls == 0

            compile_release.set()
            await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
            await asyncio.sleep(0.04)
            parent = database.list_jobs()[0]
            child = database.list_job_children(parent["id"])[0]
            assert parent["status"] == "preparing"
            assert child["stage"] == "submitting"
            assert cancel_calls == 0

            fake_comfy.submit_release.set()
            response = await asyncio.wait_for(submission, timeout=1)
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "preparing"
            await _wait_until(lambda: not app.state.submission_tasks)
            stored = database.get_job(parent["id"])
            assert stored is not None and stored["status"] == "queued"
            assert cancel_calls == 0


async def test_lifespan_recovers_interrupted_submission_and_empty_parent(
    tmp_path, fake_comfy, monkeypatch
) -> None:
    app = create_app(
        comfy_url="http://comfy.test:8188",
        database_path=tmp_path / "restart.sqlite3",
        comfy_factory=lambda _comfy_url: fake_comfy,
    )
    database = app.state.database
    database.initialize()
    settings = default_settings()
    runtime_settings = save_database_legacy_settings(database, settings)
    now = utc_now()
    timeline = _timeline(_segment("bound"), _segment("not-submitted"))
    for parent_id in ("interrupted-parent", "empty-parent", "sigkill-parent"):
        database.create_job(
            {
                "id": parent_id,
                "mode": "timeline",
                "status": "preparing",
                "progress": 0.0,
                "stage": "submitting",
                "prompt_id": None,
                "outputs": [],
                "error": None,
                "config_snapshot": {"timeline": timeline, "segment_ids": None},
                "settings_snapshot": runtime_settings.model_dump(mode="json"),
                "prompt_snapshot": {"version": 1},
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
            }
        )
    database.update_job(
        "interrupted-parent",
        status="cancelling",
        stage="submission_interrupted",
        error="shutdown interrupted submission",
    )
    # This is the Stage-0 admission window: the public parent exists, but the
    # compiler has not yet filled its bounded runtime snapshot or child rows.
    database.update_job(
        "empty-parent",
        stage="compiling",
        settings_snapshot={},
        prompt_snapshot=None,
    )
    database.update_job(
        "sigkill-parent",
        status="cancelling",
        stage="cancel_failed",
        error="process killed before submit owner returned",
    )
    database.create_job_child(
        {
            "id": "bound-prompt",
            "job_id": "interrupted-parent",
            "group_index": 0,
            "family": "fl2va",
            "backend": "standard",
            "segment_ids": ["bound"],
            "output_nodes": {"bound": "20"},
            "status": "preparing",
            "progress": 0.0,
            "stage": "submitting",
            "prompt_id": "bound-prompt",
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    database.create_job_child(
        {
            "id": "not-submitted",
            "job_id": "interrupted-parent",
            "group_index": 1,
            "family": "fl2va",
            "backend": "standard",
            "segment_ids": ["not-submitted"],
            "output_nodes": {"not-submitted": "20"},
            "status": "preparing",
            "progress": 0.0,
            "stage": "preflight",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    database.create_job_child(
        {
            "id": "sigkill-prompt",
            "job_id": "sigkill-parent",
            "group_index": 0,
            "family": "fl2va",
            "backend": "standard",
            "segment_ids": ["bound"],
            "output_nodes": {"bound": "20"},
            "status": "cancelling",
            "progress": 0.0,
            "stage": "cancelling_during_submit",
            "prompt_id": "sigkill-prompt",
            "outputs": [],
            "error": "cancel failed before SIGKILL",
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    fake_comfy.pending = [[0, "bound-prompt"], [1, "sigkill-prompt"]]
    ensure = Mock()
    close = AsyncMock()
    monkeypatch.setattr(app.state.progress_manager, "ensure", ensure)
    monkeypatch.setattr(app.state.progress_manager, "close", close)
    app.state.reconcile_interval_seconds = 0.01

    async with app.router.lifespan_context(app):
        for _ in range(100):
            interrupted = database.get_job("interrupted-parent")
            sigkill = database.get_job("sigkill-parent")
            if (
                interrupted is not None
                and interrupted["status"] == "cancelled"
                and sigkill is not None
                and sigkill["status"] == "cancelled"
            ):
                break
            await asyncio.sleep(0.01)

    interrupted = database.get_job("interrupted-parent")
    empty = database.get_job("empty-parent")
    sigkill = database.get_job("sigkill-parent")
    assert interrupted is not None and interrupted["status"] == "cancelled"
    assert empty is not None and empty["status"] == "cancelled"
    assert sigkill is not None and sigkill["status"] == "cancelled"
    assert fake_comfy.cancelled == ["bound-prompt", "sigkill-prompt"]
    assert {
        child["status"]
        for child in database.list_job_children("interrupted-parent")
    } == {"cancelled"}
    assert database.list_job_children("sigkill-parent")[0]["status"] == "cancelled"
    close.assert_awaited_once()


async def test_lifespan_yields_before_128_child_black_hole_restart_cancellation(
    tmp_path, fake_comfy, monkeypatch
) -> None:
    app = create_app(
        comfy_url="http://comfy.test:8188",
        database_path=tmp_path / "restart-black-hole.sqlite3",
        comfy_factory=lambda _comfy_url: fake_comfy,
    )
    database = app.state.database
    database.initialize()
    settings = default_settings()
    runtime_settings = save_database_legacy_settings(database, settings)
    now = utc_now()
    segments = [_segment(f"restart-{index:03d}") for index in range(128)]
    timeline = _timeline(*segments)
    database.create_job(
        {
            "id": "restart-black-hole",
            "mode": "timeline",
            "status": "preparing",
            "progress": 0.0,
            "stage": "submitting",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {"timeline": timeline, "segment_ids": None},
            "settings_snapshot": runtime_settings.model_dump(mode="json"),
            "prompt_snapshot": {"version": 1},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    for index, segment in enumerate(segments):
        prompt_id = f"restart-prompt-{index:03d}"
        database.create_job_child(
            {
                "id": prompt_id,
                "job_id": "restart-black-hole",
                "group_index": index,
                "family": "fl2va",
                "backend": "standard",
                "segment_ids": [segment["id"]],
                "output_nodes": {segment["id"]: "20"},
                "status": "preparing",
                "progress": 0.0,
                "stage": "submitting",
                "prompt_id": prompt_id,
                "outputs": [],
                "error": None,
                "prompt_snapshot": {},
                "created_at": now,
                "updated_at": now,
                "started_at": None,
                "completed_at": None,
            }
        )
    cancel_started = asyncio.Event()
    never = asyncio.Event()
    cancel_calls = 0

    async def black_hole_cancel(_prompt_id: str) -> bool:
        nonlocal cancel_calls
        cancel_calls += 1
        cancel_started.set()
        await never.wait()
        return True

    monkeypatch.setattr(fake_comfy, "cancel", black_hole_cancel)
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())
    app.state.reconcile_interval_seconds = 60.0
    context = app.router.lifespan_context(app)

    await asyncio.wait_for(context.__aenter__(), timeout=0.25)
    try:
        await asyncio.wait_for(cancel_started.wait(), timeout=0.25)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as http:
            health = await asyncio.wait_for(http.get("/api/health"), timeout=0.25)
        assert health.json() == {"status": "ok"}
        children = database.list_job_children("restart-black-hole")
        assert len(children) == 128
        assert {child["stage"] for child in children} == {
            "restart_cancel_pending"
        }
        assert fake_comfy.queue_requests == 0
        assert fake_comfy.history_requests == []
        assert cancel_calls == 1
    finally:
        await asyncio.wait_for(context.__aexit__(None, None, None), timeout=0.25)


async def test_restart_cancel_false_does_not_lose_a_late_prompt_side_effect(
    tmp_path, fake_comfy, monkeypatch
) -> None:
    app = create_app(
        comfy_url="http://comfy.test:8188",
        database_path=tmp_path / "restart-late-prompt.sqlite3",
        comfy_factory=lambda _comfy_url: fake_comfy,
    )
    database = app.state.database
    database.initialize()
    settings = default_settings()
    runtime_settings = save_database_legacy_settings(database, settings)
    now = utc_now()
    timeline = _timeline(_segment("late"))
    database.create_job(
        {
            "id": "late-parent",
            "mode": "timeline",
            "status": "preparing",
            "progress": 0.0,
            "stage": "submitting",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {"timeline": timeline, "segment_ids": None},
            "settings_snapshot": runtime_settings.model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    database.create_job_child(
        {
            "id": "late-prompt",
            "job_id": "late-parent",
            "group_index": 0,
            "family": "fl2va",
            "backend": "standard",
            "segment_ids": ["late"],
            "output_nodes": {"late": "20"},
            "status": "preparing",
            "progress": 0.0,
            "stage": "submitting",
            "prompt_id": "late-prompt",
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    original_cancel = fake_comfy.cancel
    first_attempt = asyncio.Event()
    cancel_attempts = 0

    async def observed_cancel(prompt_id: str) -> bool:
        nonlocal cancel_attempts
        cancel_attempts += 1
        result = await original_cancel(prompt_id)
        if cancel_attempts == 1:
            first_attempt.set()
        return result

    monkeypatch.setattr(fake_comfy, "cancel", observed_cancel)
    monkeypatch.setattr(app.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(app.state.progress_manager, "close", AsyncMock())
    app.state.reconcile_interval_seconds = 0.01

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(first_attempt.wait(), timeout=0.25)
        first_child = database.get_job_child("late-prompt")
        assert first_child is not None
        assert first_child["status"] == "cancelling"
        assert first_child["stage"] == "restart_cancel_unconfirmed"
        assert database.get_job("late-parent")["status"] == "cancelling"

        # Model the previous process' /prompt handler reaching queue.put only
        # after the first restart cancel and exact-history read returned absent.
        fake_comfy.pending = [[0, "late-prompt"]]
        for _ in range(100):
            parent = database.get_job("late-parent")
            if parent is not None and parent["status"] == "cancelled":
                break
            await asyncio.sleep(0.01)

    parent = database.get_job("late-parent")
    child = database.get_job_child("late-prompt")
    assert parent is not None and parent["status"] == "cancelled"
    assert child is not None and child["status"] == "cancelled"
    assert cancel_attempts >= 2
    assert fake_comfy.cancelled == ["late-prompt"]


async def test_user_cancel_preserves_restart_ownership_until_directed_cancel_confirms(
    client, fake_comfy, monkeypatch
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("restart-user-cancel"))},
    )
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(parent["id"])[0]
    database.update_job(
        parent["id"], status="cancelling", stage="restart_cancel_unconfirmed"
    )
    database.update_job_child(
        child["id"], status="cancelling", stage="restart_cancel_unconfirmed"
    )
    fake_comfy.pending = []
    original_cancel = fake_comfy.cancel

    async def unconfirmed(_prompt_id: str) -> bool:
        return False

    monkeypatch.setattr(fake_comfy, "cancel", unconfirmed)
    first = await client.post(f"/api/jobs/{parent['id']}/cancel")

    assert first.status_code == 200
    assert first.json()["status"] == "cancelling"
    persisted_child = database.get_job_child(child["id"])
    assert persisted_child is not None
    assert persisted_child["status"] == "cancelling"
    assert persisted_child["stage"] == "restart_cancel_pending"

    # The old /prompt side effect appears after the unconfirmed attempt. A
    # later explicit directed cancel can now close it; the prior absent sync
    # never marked it terminal.
    fake_comfy.pending = [[0, child["prompt_id"]]]
    monkeypatch.setattr(fake_comfy, "cancel", original_cancel)
    second = await client.post(f"/api/jobs/{parent['id']}/cancel")

    assert second.status_code == 200
    assert second.json()["status"] == "cancelled"
    assert fake_comfy.cancelled == [child["prompt_id"]]


async def test_user_cancel_failure_keeps_restart_parent_in_recovery_selector(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("restart-failed-retry"))},
    )
    parent = created.json()
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(parent["id"])[0]
    database.update_job(
        parent["id"], status="cancelling", stage="restart_cancel_unconfirmed"
    )
    database.update_job_child(
        child["id"], status="cancelling", stage="restart_cancel_unconfirmed"
    )
    fake_comfy.cancel_error = ComfyError("temporary cancel outage")

    failed = await client.post(f"/api/jobs/{parent['id']}/cancel")

    assert failed.status_code == 200
    assert failed.json()["status"] == "cancelling"
    assert failed.json()["stage"] == "restart_cancel_failed"
    assert [row["id"] for row in database.list_interrupted_preparing_jobs()] == [
        parent["id"]
    ]

    fake_comfy.cancel_error = None
    retried = await client.post(f"/api/jobs/{parent['id']}/cancel")
    assert retried.status_code == 200
    assert retried.json()["status"] == "cancelled"


async def test_confirm_comfy_restart_atomically_rejects_mixed_typed_and_legacy_owners(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("confirmed-restart-take", "t2v"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent_id = created.json()["id"]
    succeeded_child = database.list_job_children(parent_id)[0]
    segment_id = succeeded_child["segment_ids"][0]
    output_node = succeeded_child["output_nodes"][segment_id]
    preserved_output = {
        "node_id": output_node,
        "filename": "preserved-take.mp4",
        "subfolder": "segments",
        "type": "output",
    }
    database.update_job_child(
        succeeded_child["id"],
        status="succeeded",
        progress=1.0,
        stage="完成",
        outputs=[preserved_output],
        completed_at=utc_now(),
    )
    recovery_prompt_id = "confirmed-restart-control-prompt"
    now = utc_now()
    recovery_group_index = max(
        child["group_index"] for child in database.list_job_children(parent_id)
    ) + 1
    recovery_child = database.create_job_child(
        {
            "id": "confirmed-restart-control-child",
            "job_id": parent_id,
            "group_index": recovery_group_index,
            "family": "fl2va",
            "backend": "raylight",
            "segment_ids": [],
            "output_nodes": {},
            "status": "cancelling",
            "progress": 0.0,
            "stage": "restart_cancel_failed",
            "prompt_id": recovery_prompt_id,
            "outputs": [],
            "error": "old transport error",
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    marked, first_claim = database.mark_job_cancel_requested(parent_id)
    assert marked is not None and first_claim
    database.update_job(
        parent_id,
        status="cancelling",
        stage="restart_cancel_failed",
        error="old recovery error",
    )

    current_descriptor = {"family": "fl2va", "runtime_namespace": "kept-e7"}
    database.put_raylight_runtime_state({
            "version": 2,
            "epoch": 7,
            "current": current_descriptor,
            "tail_prompt_id": recovery_prompt_id,
            "tail_action": "ray_unit",
            "tainted": False,
        },
    )
    upstream_before = (
        len(fake_comfy.cancelled),
        fake_comfy.queue_requests,
        list(fake_comfy.history_requests),
    )

    children_before = database.list_job_children(parent_id)
    runtime_before = database.get_raylight_runtime_state()
    rejected = await client.post(
        f"/api/jobs/{parent_id}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )

    assert rejected.status_code == 409, rejected.text
    assert "mix legacy prompt owners" in rejected.json()["detail"]
    assert database.get_job(parent_id)["status"] == "cancelling"
    assert database.list_job_children(parent_id) == children_before
    assert database.get_job_child(succeeded_child["id"])["outputs"] == [
        preserved_output
    ]
    assert database.get_job_child(recovery_child["id"])["status"] == "cancelling"
    assert database.get_raylight_runtime_state() == runtime_before
    assert (
        len(fake_comfy.cancelled),
        fake_comfy.queue_requests,
        list(fake_comfy.history_requests),
    ) == upstream_before


async def test_confirm_comfy_restart_requires_strict_certificate_and_owner(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("strict-restart-owner", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent_id = created.json()["id"]
    child = database.list_job_children(parent_id)[0]
    marked, first_claim = database.mark_job_cancel_requested(parent_id)
    assert marked is not None and first_claim
    database.update_job(
        parent_id, status="cancelling", stage="restart_cancel_unconfirmed"
    )
    database.update_job_child(
        child["id"], status="cancelling", stage="restart_cancel_unconfirmed"
    )

    for document in (
        {},
        {"confirmation": True},
        {"confirmation": "yes"},
        {
            "confirmation": "comfyui_process_restarted",
            "unexpected": "field",
        },
    ):
        rejected = await client.post(
            f"/api/jobs/{parent_id}/recovery/confirm-comfy-restart",
            json=document,
        )
        assert rejected.status_code == 422

    same_runtime = await client.post(
        f"/api/jobs/{parent_id}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )
    assert same_runtime.status_code == 409
    assert "runtime instance has not changed" in same_runtime.json()["detail"]
    assert database.get_job(parent_id)["status"] == "cancelling"
    assert database.get_job_child(child["id"])["status"] == "cancelling"

    old_endpoint = client.director_app.state.endpoint_identity
    client.director_app.state.endpoint_identity = old_endpoint.model_copy(
        update={"runtime_instance_id": "strict-certificate-replacement"}
    )
    database.update_job_child(
        child["id"], status="cancelling", stage="submission_cancel_unconfirmed"
    )
    invalid_stage = await client.post(
        f"/api/jobs/{parent_id}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )
    assert invalid_stage.status_code == 409
    assert "invalid restart-recovery" in invalid_stage.json()["detail"]

    database.update_job_child(
        child["id"],
        status="cancelling",
        stage="restart_cancel_unconfirmed",
        prompt_id=None,
    )
    missing_prompt = await client.post(
        f"/api/jobs/{parent_id}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )
    assert missing_prompt.status_code == 409
    assert "mix legacy prompt owners" in missing_prompt.json()["detail"]

    database.update_job(
        parent_id,
        status="succeeded",
        progress=1.0,
        stage="完成",
        completed_at=utc_now(),
    )
    terminal_wins = await client.post(
        f"/api/jobs/{parent_id}/recovery/confirm-comfy-restart",
        json={"confirmation": "comfyui_process_restarted"},
    )
    assert terminal_wins.status_code == 409
    assert database.get_job(parent_id)["status"] == "succeeded"
    assert database.get_job_child(child["id"])["status"] == "cancelling"


async def test_confirm_comfy_restart_rejects_live_director_dispatcher(
    client, fake_comfy
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("live-restart-owner", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent_id = created.json()["id"]
    child = database.list_job_children(parent_id)[0]
    marked, first_claim = database.mark_job_cancel_requested(parent_id)
    assert marked is not None and first_claim
    database.update_job(
        parent_id, status="cancelling", stage="restart_cancel_pending"
    )
    database.update_job_child(
        child["id"], status="cancelling", stage="restart_cancel_pending"
    )
    blocker = asyncio.Event()
    live_dispatcher = asyncio.create_task(blocker.wait())
    client.director_app.state.submission_jobs[parent_id] = live_dispatcher
    try:
        rejected = await client.post(
            f"/api/jobs/{parent_id}/recovery/confirm-comfy-restart",
            json={"confirmation": "comfyui_process_restarted"},
        )
        assert rejected.status_code == 409
        assert "still owned by this Director process" in rejected.json()["detail"]
        assert database.get_job(parent_id)["status"] == "cancelling"
        assert database.get_job_child(child["id"])["status"] == "cancelling"
    finally:
        live_dispatcher.cancel()
        await asyncio.gather(live_dispatcher, return_exceptions=True)
        client.director_app.state.submission_jobs.pop(parent_id, None)


def _raylight_settings_document() -> dict:
    document = default_settings().model_dump(mode="json")
    document["multi_gpu_enabled"] = True
    for family in ("fl2va", "ref2va"):
        document["models"][family].update(
            backend="raylight",
            device="default",
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
    document["raylight_residency_policy"] = "keep_until_switch"
    return document


def _install_stale_eight_gpu_runtime(client, fake_comfy) -> dict:
    database = client.director_app.state.database
    old_document = _raylight_settings_document()
    for family in ("fl2va", "ref2va"):
        old_document["models"][family]["raylight"].update(
            gpu_select=list(range(8)),
            ulysses_degree=8,
            ring_degree=1,
        )
    old_settings = RuntimeSettings.model_validate(old_document)
    unit = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(
            _timeline(_segment("stale-eight-gpu-runtime", "t2v"))
        ),
        old_settings,
        "stale-four-gpu-job",
    ).workflows[0]
    descriptor = raylight_runtime_descriptor(
        bind_raylight_runtime_epoch(unit, 36)
    )
    assert descriptor is not None
    state = {
        "version": 2,
        "epoch": 36,
        "current": descriptor,
        "tail_prompt_id": "old-eight-to-four-tail",
        "tail_action": "shutdown",
        "tainted": True,
    }
    database.put_raylight_runtime_state(state)
    current_document = _raylight_settings_document()
    for family in ("fl2va", "ref2va"):
        current_document["models"][family]["raylight"].update(
            gpu_select=[0, 1, 2, 3],
            ulysses_degree=4,
            ring_degree=1,
        )
    save_database_legacy_settings(
        database,
        RuntimeSettings.model_validate(current_document),
    )
    fake_comfy.system_devices = [
        {
            "index": index,
            "type": "cuda",
            "name": f"NVIDIA RTX A6000 {index}",
            "vram_total": 48_000,
            "vram_free": 40_000,
        }
        for index in range(4)
    ]
    fake_comfy.histories["old-eight-to-four-tail"] = {
        "status": {
            "status_str": "error",
            "completed": True,
            "messages": [["execution_error", {"node_id": "ray-kill"}]],
        },
        "outputs": {},
    }
    return state


async def test_stale_raylight_gpu_runtime_is_reported_and_never_submitted(
    client, fake_comfy
) -> None:
    before = _install_stale_eight_gpu_runtime(client, fake_comfy)

    status = await client.get(
        "/api/raylight/runtime",
        headers=await runtime_authority_headers(client),
    )
    assert status.status_code == 200, status.text
    status_document = status.json()
    recovery_token = status_document.pop("recovery_token")
    assert re.fullmatch(r"[0-9a-f]{64}", recovery_token)
    assert status_document == {
        "active": True,
        "recovery_required": True,
        "epoch": 36,
        "runtime_gpu_indexes": [0, 1, 2, 3, 4, 5, 6, 7],
        "available_gpu_indexes": [0, 1, 2, 3],
        "invalid_gpu_indexes": [4, 5, 6, 7],
        "tainted": True,
    }

    rejected = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("blocked-stale-runtime", "t2v"))},
    )
    assert rejected.status_code == 200, rejected.text
    await _wait_for_submission_jobs(client)
    stored = client.director_app.state.database.get_job(rejected.json()["id"])
    assert stored is not None and stored["status"] == "failed"
    assert stored["stage"] == "preflight_failed"
    assert "raylight_runtime_restart_confirmation_required" in stored["error"]
    assert fake_comfy.prompts == []
    assert client.director_app.state.database.get_raylight_runtime_state() == before


async def test_raylight_gpu_visibility_change_between_preflights_blocks_old_barrier(
    client, fake_comfy, monkeypatch
) -> None:
    before = _install_stale_eight_gpu_runtime(client, fake_comfy)
    stats_calls = 0

    async def changing_system_stats() -> dict:
        nonlocal stats_calls
        stats_calls += 1
        visible_count = 8 if stats_calls <= 2 else 4
        return {
            "devices": [
                {
                    "index": index,
                    "type": "cuda",
                    "name": f"GPU {index}",
                    "vram_total": 48_000,
                    "vram_free": 40_000,
                }
                for index in range(visible_count)
            ]
        }

    monkeypatch.setattr(fake_comfy, "system_stats", changing_system_stats)
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("visibility-race", "t2v"))},
    )
    assert created.status_code == 200, created.text
    await _wait_until(lambda: stats_calls >= 3)
    await _wait_for_submission_jobs(client)

    parent = client.director_app.state.database.get_job(created.json()["id"])
    assert parent is not None
    assert parent["status"] == "failed"
    assert parent["stage"] == "preflight_failed"
    assert "4, 5, 6, 7" in str(parent["error"])
    assert '"expected_epoch": 36' in str(parent["error"])
    assert '"recovery_token":' in str(parent["error"])
    assert stats_calls >= 2
    assert fake_comfy.prompts == []
    after = client.director_app.state.database.get_raylight_runtime_state()
    assert after is not None
    assert after["epoch"] == before["epoch"]
    assert after["current"] == before["current"]
    assert after["tail_prompt_id"] is None
    assert after["tail_action"] is None
    assert after["tainted"] is True


async def test_confirmed_raylight_restart_recovery_preserves_epoch_and_builds_new_pool(
    client, fake_comfy
) -> None:
    before = _install_stale_eight_gpu_runtime(client, fake_comfy)
    database = client.director_app.state.database
    blocked_status = (
        await client.get(
            "/api/raylight/runtime",
            headers=await runtime_authority_headers(client),
        )
    ).json()

    confirmed = await client.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        json={
            "confirmation": "comfyui_process_restarted",
            "expected_epoch": 36,
            "expected_recovery_token": blocked_status["recovery_token"],
        },
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json() == {
        "active": False,
        "recovery_required": False,
        "epoch": 36,
        "runtime_gpu_indexes": [],
        "available_gpu_indexes": [0, 1, 2, 3],
        "invalid_gpu_indexes": [],
        "tainted": False,
        "recovery_token": None,
    }
    assert database.get_raylight_runtime_state() == {
        "version": 2,
        "epoch": 36,
        "current": None,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": False,
    }
    backups = list(database.path.parent.glob(
        f"{database.path.stem}.before-raylight-recovery-e36-*.sqlite3"
    ))
    assert len(backups) == 1
    with Database(backups[0]).connect() as backup_connection:
        assert backup_connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert Database(backups[0]).get_raylight_runtime_state() == before
    assert not database.settle_raylight_runtime_prompt("old-eight-to-four-tail",
        succeeded=True,
    )
    repeated = await client.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        json={
            "confirmation": "comfyui_process_restarted",
            "expected_epoch": 36,
            "expected_recovery_token": blocked_status["recovery_token"],
        },
    )
    assert repeated.status_code == 200
    assert repeated.json() == confirmed.json()

    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("new-four-gpu-pool", "t2v"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[0]["prompt"].values()
    )
    initializer = next(
        node["inputs"]
        for node in fake_comfy.prompts[0]["prompt"].values()
        if node.get("class_type") == "DirectorDeckRayInitializerAdvanced"
    )
    assert initializer["GPU_SELECT"] == "0,1,2,3"
    assert initializer["ray_cluster_namespace"].endswith("-e37")


async def test_raylight_restart_recovery_is_strict_and_fails_closed(
    client, fake_comfy
) -> None:
    before = _install_stale_eight_gpu_runtime(client, fake_comfy)
    endpoint = "/api/raylight/runtime/recovery/confirm-comfy-restart"
    blocked_status = (
        await client.get(
            "/api/raylight/runtime",
            headers=await runtime_authority_headers(client),
        )
    ).json()
    base = {
        "confirmation": "comfyui_process_restarted",
        "expected_epoch": 36,
        "expected_recovery_token": blocked_status["recovery_token"],
    }

    assert (await client.post(endpoint, json={**base, "extra": True})).status_code == 422
    assert (await client.post(endpoint, json={**base, "expected_epoch": 35})).status_code == 409
    assert (await client.post(
        endpoint,
        json={**base, "expected_recovery_token": "0" * 64},
    )).status_code == 409
    fake_comfy.pending = [[0, "manual-comfy-prompt"]]
    queued = await client.post(endpoint, json=base)
    assert queued.status_code == 409
    assert "empty ComfyUI queue" in queued.json()["detail"]
    assert client.director_app.state.database.get_raylight_runtime_state() == before

    fake_comfy.pending = []
    submission_lock = client.director_app.state.submission_locks.setdefault(
        "embedded",
        director_app_module.anyio.Lock(),
    )
    await submission_lock.acquire()
    try:
        in_flight = await client.post(endpoint, json=base)
    finally:
        submission_lock.release()
    assert in_flight.status_code == 409
    assert in_flight.json()["detail"]["code"] == "raylight_recovery_in_flight"

    database = client.director_app.state.database
    now = utc_now()
    database.create_job(
        {
            "id": "active-during-raylight-recovery",
            "mode": "timeline",
            "status": "preparing",
            "progress": 0.0,
            "stage": "submitting",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": database.get_settings().model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    active = await client.post(endpoint, json=base)
    assert active.status_code == 409
    assert "all Director jobs to be terminal" in active.json()["detail"]
    assert database.get_raylight_runtime_state() == before
    database.update_job(
        "active-during-raylight-recovery",
        status="failed",
        progress=1.0,
        stage="failed",
        completed_at=utc_now(),
    )
    database.create_job_child(
        {
            "id": "active-child-during-raylight-recovery",
            "job_id": "active-during-raylight-recovery",
            "group_index": 0,
            "family": "fl2va",
            "backend": "raylight",
            "segment_ids": [],
            "output_nodes": {},
            "status": "preparing",
            "progress": 0.0,
            "stage": "submitting",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    active_child = await client.post(endpoint, json=base)
    assert active_child.status_code == 409
    assert "all Director jobs to be terminal" in active_child.json()["detail"]
    assert database.get_raylight_runtime_state() == before


async def test_raylight_restart_recovery_requires_exact_history_and_maps_stats_failure(
    client, fake_comfy, monkeypatch
) -> None:
    before = _install_stale_eight_gpu_runtime(client, fake_comfy)
    status = (
        await client.get(
            "/api/raylight/runtime",
            headers=await runtime_authority_headers(client),
        )
    ).json()
    body = {
        "confirmation": "comfyui_process_restarted",
        "expected_epoch": 36,
        "expected_recovery_token": status["recovery_token"],
    }
    fake_comfy.histories.clear()
    missing = await client.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        json=body,
    )
    assert missing.status_code == 409
    assert "exact terminal history" in missing.json()["detail"]
    database = client.director_app.state.database
    assert database.get_raylight_runtime_state() == before
    assert not list(database.path.parent.glob(
        f"{database.path.stem}.before-raylight-recovery-*.sqlite3"
    ))

    async def unavailable_stats() -> dict:
        raise ComfyError("system stats unavailable")

    monkeypatch.setattr(fake_comfy, "system_stats", unavailable_stats)
    unavailable = await client.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        json=body,
    )
    assert unavailable.status_code == 502
    assert database.get_raylight_runtime_state() == before


async def test_raylight_restart_recovery_uses_durable_exact_terminal_certificate(
    client, fake_comfy
) -> None:
    _before = _install_stale_eight_gpu_runtime(client, fake_comfy)
    database = client.director_app.state.database
    assert database.settle_raylight_runtime_prompt("old-eight-to-four-tail",
        succeeded=False,
        terminal_history_certified=True,
    )
    certified = database.get_raylight_runtime_state()
    assert certified is not None
    assert certified["tail_terminal_certificate"] == {
        "prompt_id": "old-eight-to-four-tail",
        "action": "shutdown",
        "succeeded": False,
    }

    # Official ComfyUI keeps history in process memory, so a real restart
    # removes the old tail. Director's exact, prompt-bound certificate remains
    # part of the token/CAS-protected ledger and prevents a recovery deadlock.
    fake_comfy.histories.clear()
    blocked_status = (
        await client.get(
            "/api/raylight/runtime",
            headers=await runtime_authority_headers(client),
        )
    ).json()
    confirmed = await client.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        json={
            "confirmation": "comfyui_process_restarted",
            "expected_epoch": 36,
            "expected_recovery_token": blocked_status["recovery_token"],
        },
    )

    assert confirmed.status_code == 200, confirmed.text
    recovered = database.get_raylight_runtime_state()
    assert recovered is not None
    assert recovered["epoch"] == 36
    assert recovered["current"] is None
    assert "tail_terminal_certificate" not in recovered


async def test_raylight_history_absence_never_mints_a_terminal_certificate(
    client, fake_comfy
) -> None:
    before = _install_stale_eight_gpu_runtime(client, fake_comfy)
    fake_comfy.histories.clear()

    refreshed = await _refresh_raylight_runtime_tail(
        fake_comfy,
        client.director_app.state.database,
        before,
    )

    assert refreshed["tainted"] is True
    assert "tail_terminal_certificate" not in refreshed
    assert "tail_terminal_certificate" not in (
        client.director_app.state.database.get_raylight_runtime_state() or {}
    )


async def test_raylight_restart_recovery_full_state_cas_preserves_late_tail(
    client, fake_comfy, monkeypatch
) -> None:
    before = _install_stale_eight_gpu_runtime(client, fake_comfy)
    status = (
        await client.get(
            "/api/raylight/runtime",
            headers=await runtime_authority_headers(client),
        )
    ).json()
    database = client.director_app.state.database
    original_confirm = database.confirm_raylight_runtime_restart
    late = {
        **before,
        "tail_prompt_id": "late-reconciler-tail",
        "tail_action": "ray_unit",
        "tainted": False,
    }

    def race_confirmation(*args, **kwargs):
        database.put_raylight_runtime_state(late)
        return original_confirm(*args, **kwargs)

    monkeypatch.setattr(
        database,
        "confirm_raylight_runtime_restart",
        race_confirmation,
    )
    response = await client.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        json={
            "confirmation": "comfyui_process_restarted",
            "expected_epoch": 36,
            "expected_recovery_token": status["recovery_token"],
        },
    )
    assert response.status_code == 409
    assert "runtime changed" in response.json()["detail"]
    assert database.get_raylight_runtime_state() == late
    assert not list(database.path.parent.glob(
        f"{database.path.stem}.before-raylight-recovery-*.sqlite3"
    ))


async def test_malformed_raylight_runtime_only_blocks_selected_raylight(
    client, fake_comfy
) -> None:
    database = client.director_app.state.database
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO raylight_runtime_state(singleton, descriptor, updated_at) "
            "VALUES(?, ?, ?)",
            (1, "{not-json", utc_now()),
        )

    status = await client.get(
        "/api/raylight/runtime",
        headers=await runtime_authority_headers(client),
    )
    assert status.status_code == 409
    assert status.json()["detail"]["code"] == "raylight_runtime_state_invalid"
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("malformed-runtime", "t2v"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_submission_jobs(client)
    standard_stored = database.get_job(created.json()["id"])
    assert standard_stored is not None and standard_stored["status"] == "queued", (
        None if standard_stored is None else standard_stored["error"]
    )
    assert len(fake_comfy.prompts) == 1

    assert (
        await save_legacy_settings_document(client, _raylight_settings_document())
    ).status_code == 200
    ray = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("malformed-runtime-ray", "t2v"))},
    )
    assert ray.status_code == 200, ray.text
    await _wait_for_submission_jobs(client)
    stored = database.get_job(ray.json()["id"])
    assert stored is not None and stored["status"] == "failed"
    assert stored["stage"] == "preflight_failed"
    assert "raylight_runtime_state_invalid" in stored["error"]
    assert len(fake_comfy.prompts) == 1


def _initializer_namespace(prompt: dict) -> str:
    return next(
        node["inputs"]["ray_cluster_namespace"]
        for node in prompt.values()
        if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
    )


async def _wait_for_prompt_count(fake_comfy, expected: int) -> None:
    async def ready() -> None:
        while len(fake_comfy.prompts) < expected:
            await asyncio.sleep(0)

    await asyncio.wait_for(ready(), timeout=2.0)


async def _wait_until(predicate) -> None:
    async def ready() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(ready(), timeout=2.0)


async def _wait_for_submission_jobs(client) -> None:
    await _wait_until(lambda: not client.director_app.state.submission_tasks)


async def test_create_returns_admitted_job_before_runtime_preflight(
    client, fake_comfy
) -> None:
    assert (
        await save_legacy_settings_document(client, _raylight_settings_document())
    ).status_code == 200
    fake_comfy.preflight_started = asyncio.Event()
    fake_comfy.preflight_release = asyncio.Event()

    try:
        response = await asyncio.wait_for(
            client.post(
                "/api/timeline/jobs",
                json={"config": _timeline(_segment("early-admission"))},
            ),
            timeout=0.5,
        )

        assert response.status_code == 200, response.text
        assert response.json()["status"] == "preparing"
        job_id = response.json()["id"]
        stored = client.director_app.state.database.get_job(job_id)
        assert stored is not None
        assert stored["stage"] in {"compiling", "preflight", "submitting"}
        await asyncio.wait_for(fake_comfy.preflight_started.wait(), timeout=1)
        assert fake_comfy.prompts == []
    finally:
        fake_comfy.preflight_release.set()

    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)


@pytest.mark.parametrize("first_compile_fails", [False, True])
async def test_blocked_first_compiler_keeps_later_prompt_behind_admission_ticket(
    client, fake_comfy, monkeypatch, first_compile_fails
) -> None:
    first_compile_started = asyncio.Event()
    release_first_compile = asyncio.Event()
    compile_order: list[str] = []

    async def before_timeline_compile(job_id: str) -> None:
        compile_order.append(job_id)
        if len(compile_order) == 1:
            first_compile_started.set()
            await release_first_compile.wait()
            if first_compile_fails:
                raise ValueError("first compiler failed")

    monkeypatch.setattr(
        client.director_app.state,
        "before_timeline_compile",
        before_timeline_compile,
        raising=False,
    )
    try:
        first = await client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("slow-first-compile"))},
        )
        assert first.status_code == 200, first.text
        await asyncio.wait_for(first_compile_started.wait(), timeout=1)

        second = await client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("fast-second-compile"))},
        )
        assert second.status_code == 200, second.text
        database = client.director_app.state.database
        await _wait_until(
            lambda: bool(database.list_job_children(second.json()["id"]))
        )
        assert fake_comfy.prompts == []

        release_first_compile.set()
        await _wait_for_prompt_count(fake_comfy, 1 if first_compile_fails else 2)
        second_child = database.list_job_children(second.json()["id"])[0]
        await _wait_for_submission_jobs(client)
        if first_compile_fails:
            failed_first = database.get_job(first.json()["id"])
            assert failed_first is not None and failed_first["status"] == "failed"
            assert [item["prompt_id"] for item in fake_comfy.prompts] == [
                second_child["id"]
            ]
        else:
            first_child = database.list_job_children(first.json()["id"])[0]
            assert [item["prompt_id"] for item in fake_comfy.prompts] == [
                first_child["id"],
                second_child["id"],
            ]
    finally:
        release_first_compile.set()


async def _make_ambiguous_standard_recovery_tail(client, fake_comfy, identity: str):
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(identity, "t2v"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    parent = database.get_job(created.json()["id"])
    child = database.list_job_children(created.json()["id"])[0]
    assert parent is not None and child["backend"] == "standard"
    fake_comfy.pending = []
    database.update_job(
        parent["id"], status="cancelling", stage="restart_cancel_pending"
    )
    database.update_job_child(
        child["id"], status="cancelling", stage="restart_cancel_pending"
    )
    return database.get_job(parent["id"]), database.get_job_child(child["id"])


def _add_succeeded_raylight_control(database, job_id: str, *, group_index: int) -> dict:
    now = utc_now()
    return database.create_job_child(
        {
            "id": f"control-{job_id}-{group_index}",
            "job_id": job_id,
            "group_index": group_index,
            "family": "fl2va",
            "backend": "raylight",
            "segment_ids": [],
            "output_nodes": {},
            "status": "succeeded",
            "progress": 1.0,
            "stage": "RayLight 安全切换完成",
            "prompt_id": f"control-prompt-{job_id}-{group_index}",
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
        }
    )


def _complete_fake_prompt(fake_comfy, child: dict) -> None:
    prompt_id = str(child["prompt_id"])
    fake_comfy.pending = [
        item for item in fake_comfy.pending if prompt_id not in item
    ]
    fake_comfy.running = [
        item for item in fake_comfy.running if prompt_id not in item
    ]
    fake_comfy.histories[prompt_id] = _success(child)


async def test_raylight_dispatcher_returns_preparing_and_terminal_gates_parents(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )

    first = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("first-a", "t2v"), _segment("second-a", "t2v")
            )
        },
    )
    second = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("other-parent-a", "t2v"))},
    )
    assert first.status_code == second.status_code == 200
    assert first.json()["status"] == second.json()["status"] == "preparing"
    await _wait_for_prompt_count(fake_comfy, 1)

    database = client.director_app.state.database
    first_children = database.list_job_children(first.json()["id"])
    await _wait_until(
        lambda: bool(database.list_job_children(second.json()["id"]))
    )
    second_children = database.list_job_children(second.json()["id"])
    assert first_children[0]["prompt_id"] == fake_comfy.prompts[0]["prompt_id"]
    assert first_children[1]["prompt_id"] is None
    assert second_children[0]["prompt_id"] is None
    # Yield repeatedly: neither the same-parent sibling nor a later parent may
    # become visible in Comfy while the first Ray generation is nonterminal.
    for _ in range(10):
        await asyncio.sleep(0)
    assert len(fake_comfy.prompts) == 1

    _complete_fake_prompt(fake_comfy, first_children[0])
    await _wait_for_prompt_count(fake_comfy, 2)
    first_children = database.list_job_children(first.json()["id"])
    assert first_children[1]["prompt_id"] == fake_comfy.prompts[1]["prompt_id"]
    assert database.list_job_children(second.json()["id"])[0]["prompt_id"] is None

    _complete_fake_prompt(fake_comfy, first_children[1])
    await _wait_for_prompt_count(fake_comfy, 3)
    assert (
        database.list_job_children(second.json()["id"])[0]["prompt_id"]
        == fake_comfy.prompts[2]["prompt_id"]
    )


async def test_ambiguous_old_standard_submission_does_not_gate_new_raylight(
    client, fake_comfy, monkeypatch
) -> None:
    old_parent, old_child = await _make_ambiguous_standard_recovery_tail(
        client, fake_comfy, "ambiguous-standard-before-ray"
    )
    assert old_parent is not None and old_child is not None
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )

    newer = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-after-ambiguous-standard", "t2v"))},
    )
    assert newer.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 2)
    assert any(
        node.get("class_type") == "DirectorDeckRayInitializerAdvanced"
        for node in fake_comfy.prompts[1]["prompt"].values()
    )
    assert client.director_app.state.database.get_job_child(old_child["id"])[
        "stage"
    ] == "restart_cancel_pending"
    assert fake_comfy.cancelled == []


async def test_cancel_new_parent_does_not_mutate_ambiguous_old_standard_recovery(
    client, fake_comfy, monkeypatch
) -> None:
    old_parent, old_child = await _make_ambiguous_standard_recovery_tail(
        client, fake_comfy, "ambiguous-standard-cancel-new"
    )
    assert old_parent is not None and old_child is not None
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    fake_comfy.preflight_started = asyncio.Event()
    fake_comfy.preflight_release = asyncio.Event()
    newer = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancelled-behind-old-standard", "t2v"))},
    )
    assert newer.status_code == 200
    await asyncio.wait_for(fake_comfy.preflight_started.wait(), timeout=1)
    assert len(fake_comfy.prompts) == 1

    cancelled = await client.post(f"/api/jobs/{newer.json()['id']}/cancel")
    assert cancelled.status_code == 200
    fake_comfy.preflight_release.set()
    await _wait_for_submission_jobs(client)
    assert client.director_app.state.database.get_job(newer.json()["id"])[
        "status"
    ] == "cancelled"
    assert len(fake_comfy.prompts) == 1
    # Cancelling the waiter does not mutate or falsely settle the old owner.
    preserved = client.director_app.state.database.get_job_child(old_child["id"])
    assert preserved is not None and preserved["stage"] == "restart_cancel_pending"


@pytest.mark.parametrize("terminal", ["failed", "externally_removed"])
async def test_raylight_failed_generation_kills_pool_before_same_parent_successor(
    client, fake_comfy, monkeypatch, terminal: str
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )

    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment(f"{terminal}-one", "t2v"),
                _segment(f"{terminal}-two", "t2v"),
            )
        },
    )
    assert created.status_code == 200
    assert created.json()["status"] == "preparing"
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    first_child = database.list_job_children(created.json()["id"])[0]
    prompt_id = str(first_child["prompt_id"])
    fake_comfy.pending = []
    if terminal == "failed":
        fake_comfy.histories[prompt_id] = _failure("ray actor failed")
    else:
        # Queue/history absence is not release evidence for a Stage-4 owner.
        # The old test expected a local "externally removed" classification;
        # the durable exact-prompt contract now deliberately blocks successors.
        await asyncio.sleep(0.05)
        assert len(fake_comfy.prompts) == 1
        ownership = database.get_prompt_ownership(first_child["id"])
        assert ownership is not None and ownership.state == "unconfirmed"
        children = database.list_job_children(created.json()["id"])
        second = next(
            child
            for child in children
            if child["segment_ids"] == ["externally_removed-two"]
        )
        assert second["prompt_id"] is None
        fake_comfy.pending = [[0, prompt_id]]
        cancelled = await client.post(
            f"/api/jobs/{created.json()['id']}/cancel"
        )
        assert cancelled.status_code == 200
        return

    # The successor can only appear after an exact DirectorDeckRayKill barrier, and the
    # invalid mutable actor handle must never reuse epoch 1.
    await _wait_for_prompt_count(fake_comfy, 3)
    submitted_types = [
        {node["class_type"] for node in item["prompt"].values()}
        for item in fake_comfy.prompts
    ]
    assert "DirectorDeckRayKill" not in submitted_types[0]
    assert "DirectorDeckRayKill" in submitted_types[1]
    assert "DirectorDeckRayKill" not in submitted_types[2]
    assert [
        _initializer_namespace(fake_comfy.prompts[index]["prompt"]).rsplit("-e", 1)[1]
        for index in (0, 2)
    ] == ["1", "2"]
    children = database.list_job_children(created.json()["id"])
    first = next(child for child in children if child["segment_ids"] == [f"{terminal}-one"])
    assert first["status"] == "failed"
    assert next(child for child in children if not child["segment_ids"])["status"] == "succeeded"


async def test_parent_cancel_during_raylight_gate_never_submits_successor(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("cancel-first", "t2v"),
                _segment("must-not-submit", "t2v"),
            )
        },
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    cancelled = await client.post(f"/api/jobs/{created.json()['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] in {"cancelling", "cancelled"}
    await _wait_until(
        lambda: not client.director_app.state.submission_tasks
    )
    assert len(fake_comfy.prompts) == 1
    children = client.director_app.state.database.list_job_children(
        created.json()["id"]
    )
    assert children[1]["prompt_id"] is None
    assert children[1]["status"] == "cancelled"


async def test_raylight_running_observation_is_durable_and_later_absence_blocks(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("running-lost", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    child = database.list_job_children(created.json()["id"])[0]
    prompt_id = str(child["prompt_id"])
    fake_comfy.pending = []
    fake_comfy.running = [[0, prompt_id]]
    await _wait_until(
        lambda: database.get_job_child(child["id"])["status"] == "running"
    )
    running = database.get_job_child(child["id"])
    assert running is not None and running["started_at"] is not None

    fake_comfy.running = []
    await _wait_until(
        lambda: database.get_prompt_ownership(child["id"]).state == "unconfirmed"
    )
    blocked = database.get_job_child(child["id"])
    assert blocked is not None and blocked["status"] == "running"
    # Explicit cancellation remains the only release path after ambiguous
    # queue/history absence.
    fake_comfy.pending = [[0, prompt_id]]
    cancelled = await client.post(f"/api/jobs/{created.json()['id']}/cancel")
    assert cancelled.status_code == 200


async def test_parent_cancel_while_waiting_for_raylight_barrier_closes_control(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    first = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("resident-before-cancel", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    fake_comfy.auto_complete_ray_kill = False
    switched_timeline = _timeline(_segment("barrier-cancel", "t2v"))
    switched_timeline["sampling"]["fl2va"]["shift"] = 8.0
    switched = await client.post(
        "/api/timeline/jobs",
        json={"config": switched_timeline},
    )
    await _wait_for_prompt_count(fake_comfy, 2)
    assert any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[1]["prompt"].values()
    )
    cancelled = await client.post(f"/api/jobs/{switched.json()['id']}/cancel")
    assert cancelled.status_code == 200
    await _wait_until(lambda: not client.director_app.state.submission_tasks)
    # No target generation was ever submitted after the cancelled barrier.
    assert len(fake_comfy.prompts) == 2
    children = client.director_app.state.database.list_job_children(
        switched.json()["id"]
    )
    stored_parent = client.director_app.state.database.get_job(
        switched.json()["id"]
    )
    assert stored_parent is not None and stored_parent["status"] == "cancelled"
    assert stored_parent["stage"] == "cancelled"
    assert all(child["status"] == "cancelled" for child in children)
    assert all(child.get("prompt_id") is None for child in children if child["segment_ids"])


async def test_failed_dynamic_raylight_barrier_is_orchestration_failure(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    first = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("resident-before-failed-kill", "t2v"))},
    )
    assert first.status_code == 200
    await _wait_for_submission_jobs(client)
    fake_comfy.auto_complete_ray_kill = False

    switched_timeline = _timeline(_segment("target-after-failed-kill", "t2v"))
    switched_timeline["sampling"]["fl2va"]["shift"] = 8.0
    switched = await client.post(
        "/api/timeline/jobs",
        json={"config": switched_timeline},
    )
    assert switched.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 2)
    barrier_prompt_id = str(fake_comfy.prompts[-1]["prompt_id"])
    assert any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[-1]["prompt"].values()
    )
    fake_comfy.pending = [
        item for item in fake_comfy.pending if barrier_prompt_id not in item
    ]
    fake_comfy.histories[barrier_prompt_id] = _failure("DirectorDeckRayKill exploded")
    await _wait_for_submission_jobs(client)

    database = client.director_app.state.database
    parent = database.get_job(switched.json()["id"])
    children = database.list_job_children(switched.json()["id"])
    control = next(child for child in children if not child["segment_ids"])
    target = next(child for child in children if child["segment_ids"])
    assert parent is not None and parent["status"] == "failed"
    assert parent["stage"] == "raylight_switch_failed"
    assert parent["cancel_requested"] == 0
    assert control["status"] == "failed"
    assert target["status"] == "failed"
    # The target generation must never cross the network after a failed kill.
    assert len(fake_comfy.prompts) == 2


async def test_raylight_barrier_failure_before_cancel_click_remains_failed(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    first = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("resident-before-click-race", "t2v"))},
    )
    assert first.status_code == 200
    await _wait_for_submission_jobs(client)
    fake_comfy.auto_complete_ray_kill = False
    switched_timeline = _timeline(_segment("target-after-click-race", "t2v"))
    switched_timeline["sampling"]["fl2va"]["shift"] = 8.0
    switched = await client.post(
        "/api/timeline/jobs",
        json={"config": switched_timeline},
    )
    await _wait_for_prompt_count(fake_comfy, 2)
    barrier_prompt_id = str(fake_comfy.prompts[-1]["prompt_id"])
    fake_comfy.pending = [
        item for item in fake_comfy.pending if barrier_prompt_id not in item
    ]
    fake_comfy.histories[barrier_prompt_id] = _failure(
        "DirectorDeckRayKill failed before cancel click"
    )

    cancelled = await client.post(f"/api/jobs/{switched.json()['id']}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "failed"
    assert cancelled.json()["stage"] == "raylight_switch_failed"
    assert barrier_prompt_id not in fake_comfy.cancelled
    await _wait_for_submission_jobs(client)
    assert len(fake_comfy.prompts) == 2


async def test_live_raylight_dispatcher_preserves_success_before_cancel_click(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.01
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-success-before-click", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    child = database.list_job_children(created.json()["id"])[0]
    _complete_fake_prompt(fake_comfy, child)

    cancelled = await client.post(f"/api/jobs/{created.json()['id']}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "succeeded"
    assert str(child["prompt_id"]) not in fake_comfy.cancelled
    await _wait_for_submission_jobs(client)


async def test_live_dispatcher_blocks_delete_and_bulk_clear_after_terminal_cas(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("blocked-submit", "t2v"))},
    )
    await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
    database = client.director_app.state.database
    child = database.list_job_children(created.json()["id"])[0]
    database.update_job_child(
        child["id"],
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        completed_at=utc_now(),
    )
    database.update_job(
        created.json()["id"],
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        completed_at=utc_now(),
    )

    rejected = await client.delete(f"/api/jobs/{created.json()['id']}")
    assert rejected.status_code == 409
    cleared = await client.delete("/api/jobs")
    assert cleared.status_code == 200
    assert cleared.json()["deleted_count"] == 0
    assert database.get_job(created.json()["id"]) is not None

    fake_comfy.submit_release.set()
    await _wait_until(lambda: not client.director_app.state.submission_tasks)
    deleted = await client.delete(f"/api/jobs/{created.json()['id']}")
    assert deleted.status_code == 200


async def test_cancelled_waiting_dispatcher_releases_ownership_for_immediate_delete(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False

    predecessor = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("holds-endpoint-ticket", "t2v"))},
    )
    assert predecessor.status_code == 200, predecessor.text
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    predecessor_child = database.list_job_children(predecessor.json()["id"])[0]

    waiting = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancel-before-ticket", "t2v"))},
    )
    assert waiting.status_code == 200, waiting.text
    waiting_id = waiting.json()["id"]
    await _wait_until(lambda: bool(database.list_job_children(waiting_id)))
    waiting_child = database.list_job_children(waiting_id)[0]
    assert waiting_child["prompt_id"] is None
    assert len(fake_comfy.prompts) == 1

    cancelled = await client.post(f"/api/jobs/{waiting_id}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    await _wait_until(
        lambda: waiting_id not in client.director_app.state.submission_jobs
    )
    assert len(fake_comfy.prompts) == 1
    deleted = await client.delete(f"/api/jobs/{waiting_id}")
    assert deleted.status_code == 200, deleted.text

    follower = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("still-waits-for-real-owner", "t2v"))},
    )
    assert follower.status_code == 200, follower.text
    await _wait_until(
        lambda: bool(database.list_job_children(follower.json()["id"]))
    )
    follower_child = database.list_job_children(follower.json()["id"])[0]
    assert follower_child["prompt_id"] is None
    assert len(fake_comfy.prompts) == 1

    _complete_fake_prompt(fake_comfy, predecessor_child)
    await _wait_for_prompt_count(fake_comfy, 2)
    follower_child = database.get_job_child(follower_child["id"])
    assert follower_child is not None and follower_child["prompt_id"] is not None
    _complete_fake_prompt(fake_comfy, follower_child)
    await _wait_for_submission_jobs(client)


async def test_terminal_cancel_retry_quiesces_waiting_dispatcher_before_delete(
    client, fake_comfy
) -> None:
    """A terminalized waiter must not keep its endpoint ticket or delete guard."""

    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False

    predecessor = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("terminal-retry-predecessor", "t2v"))},
    )
    assert predecessor.status_code == 200, predecessor.text
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    predecessor_child = database.list_job_children(predecessor.json()["id"])[0]

    waiting = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("terminal-retry-waiter", "t2v"))},
    )
    assert waiting.status_code == 200, waiting.text
    waiting_id = waiting.json()["id"]
    await _wait_until(lambda: bool(database.list_job_children(waiting_id)))
    waiting_child = database.list_job_children(waiting_id)[0]
    assert waiting_child["prompt_id"] is None

    follower = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("terminal-retry-follower", "t2v"))},
    )
    assert follower.status_code == 200, follower.text
    await _wait_until(
        lambda: bool(database.list_job_children(follower.json()["id"]))
    )
    follower_child = database.list_job_children(follower.json()["id"])[0]
    assert follower_child["prompt_id"] is None

    # Model the real race: another cancellation owner publishes the durable
    # terminal rows before this idempotent cancel request reaches its initial
    # sync.  The submission coroutine is still shielded behind the older
    # endpoint ticket and therefore remains in ``submission_jobs``.
    database.update_job_child(
        waiting_child["id"],
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        completed_at=utc_now(),
    )
    database.update_job(
        waiting_id,
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        cancel_requested=True,
        completed_at=utc_now(),
    )
    waiting_dispatcher = client.director_app.state.submission_jobs[waiting_id]
    assert not waiting_dispatcher.done()

    cancelled = await client.post(f"/api/jobs/{waiting_id}/cancel")

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert waiting_id not in client.director_app.state.submission_jobs
    deleted = await client.delete(f"/api/jobs/{waiting_id}")
    assert deleted.status_code == 200, deleted.text
    assert len(fake_comfy.prompts) == 1

    _complete_fake_prompt(fake_comfy, predecessor_child)
    await _wait_for_prompt_count(fake_comfy, 2)
    follower_child = database.get_job_child(follower_child["id"])
    assert follower_child is not None and follower_child["prompt_id"] is not None
    _complete_fake_prompt(fake_comfy, follower_child)
    await _wait_for_submission_jobs(client)


async def test_cancelled_middle_ticket_keeps_pending_predecessor_order(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    endpoint_key = "embedded"
    pending_predecessor = asyncio.get_running_loop().create_future()
    client.director_app.state.submission_tails[endpoint_key] = pending_predecessor

    try:
        first = await client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("ticket-proxy-first", "t2v"))},
        )
        middle = await client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("ticket-proxy-middle", "t2v"))},
        )
        assert first.status_code == middle.status_code == 200
        database = client.director_app.state.database
        await _wait_until(
            lambda: bool(database.list_job_children(first.json()["id"]))
        )
        first_child = database.list_job_children(first.json()["id"])[0]
        assert fake_comfy.prompts == []

        cancelled = await client.post(f"/api/jobs/{middle.json()['id']}/cancel")

        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
        assert middle.json()["id"] not in client.director_app.state.submission_jobs
        deleted = await client.delete(f"/api/jobs/{middle.json()['id']}")
        assert deleted.status_code == 200, deleted.text

        last = await client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("ticket-proxy-last", "t2v"))},
        )
        assert last.status_code == 200, last.text
        await _wait_until(
            lambda: bool(database.list_job_children(last.json()["id"]))
        )
        last_child = database.list_job_children(last.json()["id"])[0]
        await asyncio.sleep(0.02)
        assert fake_comfy.prompts == []

        pending_predecessor.set_result(None)
        await _wait_for_prompt_count(fake_comfy, 1)
        assert fake_comfy.prompts[0]["prompt_id"] == first_child["id"]
        first_child = database.get_job_child(first_child["id"])
        assert first_child is not None
        _complete_fake_prompt(fake_comfy, first_child)

        await _wait_for_prompt_count(fake_comfy, 2)
        assert fake_comfy.prompts[1]["prompt_id"] == last_child["id"]
        last_child = database.get_job_child(last_child["id"])
        assert last_child is not None
        _complete_fake_prompt(fake_comfy, last_child)
        await _wait_for_submission_jobs(client)
    finally:
        if not pending_predecessor.done():
            pending_predecessor.set_result(None)
        remaining = list(client.director_app.state.submission_tasks)
        for task in remaining:
            task.cancel()
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)


async def test_dispatcher_does_not_cancel_parent_advanced_by_reconciler(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("reconciled-first", "t2v"),
                _segment("reconciled-second", "t2v"),
            )
        },
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    children = database.list_job_children(created.json()["id"])
    _complete_fake_prompt(fake_comfy, children[0])
    await _wait_for_prompt_count(fake_comfy, 2)
    database.update_job(
        created.json()["id"],
        status="running",
        stage="native segments 1/2",
        started_at=utc_now(),
    )
    children = database.list_job_children(created.json()["id"])
    _complete_fake_prompt(fake_comfy, children[1])
    await _wait_until(lambda: not client.director_app.state.submission_tasks)
    await _reconcile(client, created.json())
    finished = await client.get(f"/api/jobs/{created.json()['id']}")
    assert finished.status_code == 200
    assert finished.json()["status"] == "succeeded"
    assert all(
        child["status"] == "succeeded" for child in finished.json()["children"]
    )


async def test_cancelled_ticket_waiter_closes_dynamically_created_barrier(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    holder = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ticket-holder", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    waiter = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancelled-waiter", "r2v"))},
    )
    cancelled = await client.post(f"/api/jobs/{waiter.json()['id']}/cancel")
    assert cancelled.status_code == 200

    database = client.director_app.state.database
    holder_child = database.list_job_children(holder.json()["id"])[0]
    _complete_fake_prompt(fake_comfy, holder_child)
    await _wait_until(lambda: not client.director_app.state.submission_tasks)
    # Cancellation may win before compilation materializes any child. If a
    # child already exists it must be terminal, and no DirectorDeckRayKill may
    # cross the network on the cancelled parent's behalf.
    assert len(fake_comfy.prompts) == 1
    children = database.list_job_children(waiter.json()["id"])
    assert all(child["status"] == "cancelled" for child in children)
    assert all(child.get("prompt_id") is None for child in children)


@pytest.mark.parametrize(
    "recovery_stage", ["submission_cancel_unconfirmed", "restart_cancel_pending"]
)
async def test_ambiguous_old_raylight_fails_only_the_selected_ray_job(
    client, fake_comfy, monkeypatch, recovery_stage: str
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    old = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"old-{recovery_stage}", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    old_child = database.list_job_children(old.json()["id"])[0]
    fake_comfy.pending = []
    database.update_job_child(
        old_child["id"], status="cancelling", stage=recovery_stage
    )
    database.update_job(
        old.json()["id"], status="cancelling", stage=recovery_stage
    )

    newer = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"new-{recovery_stage}", "t2v"))},
    )
    assert newer.status_code == 200
    await _wait_for_submission_jobs(client)
    assert len(fake_comfy.prompts) == 1
    failed = database.get_job(newer.json()["id"])
    assert failed is not None and failed["status"] == "failed"
    assert "raylight_transition_recovery_pending" in failed["error"]

    # The old job retains its own durable recovery ownership, but an unrelated
    # Standard job is admitted and reaches ComfyUI immediately.
    settings["multi_gpu_enabled"] = False
    settings["models"]["fl2va"].update(
        backend="standard",
        raylight={
            "gpu_select": [0],
            "ulysses_degree": 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    standard = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"standard-{recovery_stage}", "t2v"))},
    )
    assert standard.status_code == 200
    await _wait_for_submission_jobs(client)
    stored_standard = database.get_job(standard.json()["id"])
    assert stored_standard is not None and stored_standard["status"] == "queued", (
        None if stored_standard is None else stored_standard["error"]
    )
    await _wait_for_prompt_count(fake_comfy, 2)
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[1]["prompt"].values()
    )
    old_after = database.get_job_child(old_child["id"])
    assert old_after is not None
    assert old_after["status"] == "cancelling"
    assert old_after["prompt_id"] == old_child["prompt_id"]


async def test_cancel_new_parent_aborts_ambiguous_old_raylight_tail_gate(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    old = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("old-ambiguous-cancel-new", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    origin = "embedded"
    old_child = database.list_job_children(old.json()["id"])[0]
    old_prompt_id = str(old_child["prompt_id"])
    fake_comfy.pending = []
    database.update_job_child(
        old_child["id"], status="cancelling", stage="submission_cancel_unconfirmed"
    )
    database.update_job(
        old.json()["id"], status="cancelling", stage="submission_cancel_unconfirmed"
    )
    before_state = database.get_raylight_runtime_state()

    newer = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancel-this-new", "r2v"))},
    )
    cancelled = await client.post(f"/api/jobs/{newer.json()['id']}/cancel")
    assert cancelled.status_code == 200
    await _wait_until(
        lambda: newer.json()["id"] not in client.director_app.state.submission_jobs
    )
    assert len(fake_comfy.prompts) == 1
    assert database.get_raylight_runtime_state() == before_state
    assert database.get_raylight_runtime_state()["tail_prompt_id"] == old_prompt_id


async def test_cancel_new_parent_aborts_ambiguous_old_shutdown_gate(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    resident = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("shutdown-cancel-new-resident", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    origin = "embedded"
    state = database.get_raylight_runtime_state()
    assert state is not None and state["current"] is not None
    old_barrier_id = str(uuid.uuid4())
    old_parent_id = str(uuid.uuid4())
    now = utc_now()
    database.create_job(
        {
            "id": old_parent_id,
            "mode": "timeline",
            "status": "cancelling",
            "progress": 0.0,
            "stage": "submission_cancel_unconfirmed",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": settings,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    database.create_job_child(
        {
            "id": old_barrier_id,
            "job_id": old_parent_id,
            "group_index": 0,
            "family": "fl2va",
            "backend": "raylight",
            "segment_ids": [],
            "output_nodes": {},
            "status": "cancelling",
            "progress": 0.0,
            "stage": "submission_cancel_unconfirmed",
            "prompt_id": old_barrier_id,
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    before_state = {
        "version": 2,
        "epoch": state["epoch"],
        "current": state["current"],
        "tail_prompt_id": old_barrier_id,
        "tail_action": "shutdown",
        "tainted": True,
    }
    database.put_raylight_runtime_state(before_state)

    newer = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancel-after-old-shutdown", "r2v"))},
    )
    cancelled = await client.post(f"/api/jobs/{newer.json()['id']}/cancel")
    assert cancelled.status_code == 200
    await _wait_until(
        lambda: newer.json()["id"] not in client.director_app.state.submission_jobs
    )
    assert len(fake_comfy.prompts) == 1
    assert database.get_raylight_runtime_state() == before_state


async def test_standard_create_returns_preparing_behind_running_raylight(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    ray = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-holds-standard", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    ray_child = client.director_app.state.database.list_job_children(
        ray.json()["id"]
    )[0]

    settings["multi_gpu_enabled"] = False
    settings["models"]["fl2va"].update(
        backend="standard",
        raylight={
            "gpu_select": [0],
            "ulysses_degree": 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    standard = await asyncio.wait_for(
        client.post(
            "/api/timeline/jobs",
            json={"config": _timeline(_segment("standard-waits", "t2v"))},
        ),
        timeout=5,
    )
    assert standard.status_code == 200
    assert standard.json()["status"] == "preparing"
    assert len(fake_comfy.prompts) == 1
    # Admission now precedes compilation and child materialization.  The
    # accepted parent is immediately visible even while endpoint-local work is
    # serialized behind the running Ray task.
    assert not client.director_app.state.database.list_job_children(
        standard.json()["id"]
    )

    _complete_fake_prompt(fake_comfy, ray_child)
    await _wait_for_prompt_count(fake_comfy, 3)
    assert any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[1]["prompt"].values()
    )
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[2]["prompt"].values()
    )


@pytest.mark.parametrize(
    "recovery_stage", ["submission_cancel_unconfirmed", "restart_cancel_pending"]
)
async def test_ambiguous_old_raylight_shutdown_is_not_a_global_gate(
    client, fake_comfy, monkeypatch, recovery_stage: str
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    resident = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"shutdown-resident-{recovery_stage}", "t2v"))},
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    origin = "embedded"
    state = database.get_raylight_runtime_state()
    assert state is not None and state["current"] is not None

    old_barrier_id = str(uuid.uuid4())
    old_parent_id = str(uuid.uuid4())
    now = utc_now()
    database.create_job(
        {
            "id": old_parent_id,
            "mode": "timeline",
            "status": "cancelling",
            "progress": 0.0,
            "stage": recovery_stage,
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": settings,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    database.create_job_child(
        {
            "id": old_barrier_id,
            "job_id": old_parent_id,
            "group_index": 0,
            "family": "fl2va",
            "backend": "raylight",
            "segment_ids": [],
            "output_nodes": {},
            "status": "cancelling",
            "progress": 0.0,
            "stage": recovery_stage,
            "prompt_id": old_barrier_id,
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    database.put_raylight_runtime_state({
            "version": 2,
            "epoch": state["epoch"],
            "current": state["current"],
            "tail_prompt_id": old_barrier_id,
            "tail_action": "shutdown",
            "tainted": True,
        },
    )

    newer = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"after-shutdown-{recovery_stage}", "r2v"))},
    )
    assert newer.status_code == 200
    await _wait_for_submission_jobs(client)
    assert len(fake_comfy.prompts) == 1
    failed = database.get_job(newer.json()["id"])
    assert failed is not None and failed["status"] == "failed"
    assert "raylight_transition_recovery_pending" in failed["error"]

    # The selected Ray job fails locally, but the historical shutdown does not
    # become an endpoint-wide gate.  Standard submission proceeds without
    # taking ownership of or rewriting the ambiguous Ray ledger.
    settings["multi_gpu_enabled"] = False
    settings["models"]["fl2va"].update(
        backend="standard",
        raylight={
            "gpu_select": [0],
            "ulysses_degree": 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    standard = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment(f"standard-after-shutdown-{recovery_stage}", "t2v")
            )
        },
    )
    assert standard.status_code == 200
    await _wait_for_submission_jobs(client)
    stored_standard = database.get_job(standard.json()["id"])
    assert stored_standard is not None and stored_standard["status"] == "queued", (
        None if stored_standard is None else stored_standard["error"]
    )
    await _wait_for_prompt_count(fake_comfy, 2)
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[-1]["prompt"].values()
    )


async def test_missing_ambiguous_raylight_tail_child_does_not_gate_standard(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    resident = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("missing-tail-resident", "t2v"))},
    )
    assert resident.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    state = database.get_raylight_runtime_state()
    assert state is not None and state["current"] is not None
    database.put_raylight_runtime_state(
        {
            "version": 2,
            "epoch": state["epoch"],
            "current": state["current"],
            "tail_prompt_id": str(uuid.uuid4()),
            "tail_action": "shutdown",
            "tainted": True,
        }
    )

    settings["multi_gpu_enabled"] = False
    settings["models"]["fl2va"].update(
        backend="standard",
        raylight={
            "gpu_select": [0],
            "ulysses_degree": 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    standard = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("standard-after-missing-tail", "t2v"))},
    )

    assert standard.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 2)
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[-1]["prompt"].values()
    )


async def test_raylight_model_switch_reuses_pool_and_epoch_without_barrier(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200

    model_b_filename = "generic_h3_diffusion.safetensors"
    settings_b = {
        key: (
            value
            if key != "models"
            else {
                family: dict(model) for family, model in value.items()
            }
        )
        for key, value in settings.items()
    }
    settings_b["models"]["fl2va"]["filename"] = model_b_filename

    first = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("a-one", "t2v"))},
    )
    repeated = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("a-two", "t2v"))},
    )
    assert (await save_legacy_settings_document(client, settings_b)).status_code == 200
    switched = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("a-b", "t2v"))},
    )
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    switched_back = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("a-three", "t2v"))},
    )
    assert all(
        response.status_code == 200
        for response in (first, repeated, switched, switched_back)
    )
    await _wait_for_prompt_count(fake_comfy, 4)

    ray_prompts = [
        item["prompt"]
        for item in fake_comfy.prompts
        if any(
            node.get("class_type") == "DirectorDeckRayUNETLoader"
            for node in item["prompt"].values()
        )
        and not any(
            node.get("class_type") == "DirectorDeckRayKill"
            for node in item["prompt"].values()
        )
    ]
    unet_names = [
        next(
            node["inputs"]["unet_name"]
            for node in prompt.values()
            if node["class_type"] == "DirectorDeckRayUNETLoader"
        )
        for prompt in ray_prompts
    ]
    assert unet_names == [
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        model_b_filename,
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    ]
    # Model switches keep the same pool: one namespace, one epoch, no DirectorDeckRayKill.
    namespaces = [_initializer_namespace(prompt) for prompt in ray_prompts]
    assert namespaces[0] == namespaces[1] == namespaces[2] == namespaces[3]
    assert [namespace.rsplit("-e", 1)[1] for namespace in namespaces] == [
        "1",
        "1",
        "1",
        "1",
    ]
    barriers = [
        item["prompt"]
        for item in fake_comfy.prompts
        if any(
            node.get("class_type") == "DirectorDeckRayKill"
            for node in item["prompt"].values()
        )
    ]
    assert len(barriers) == 0


async def test_legacy_torch_flash_pool_crosses_compatibility_barrier_without_enabling_ck(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200

    legacy_unit = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(
            _timeline(_segment("legacy-torch-flash", "t2v"))
        ),
        RuntimeSettings.model_validate(settings),
        "persisted-legacy-torch-flash",
    ).workflows[0]
    legacy_descriptor = raylight_runtime_descriptor(
        bind_raylight_runtime_epoch(legacy_unit, 1)
    )
    assert legacy_descriptor is not None
    legacy_namespace = "director-legacy-torch-flash-e1"
    legacy_descriptor.update(
        compatibility_key="director-legacy-torch-flash",
        runtime_key="legacy-torch-flash-runtime-key",
        runtime_namespace=legacy_namespace,
    )
    legacy_initializer_id = legacy_descriptor["initializer_node_id"]
    legacy_initializer = legacy_descriptor["loader_subgraph"][
        legacy_initializer_id
    ]["inputs"]
    legacy_initializer.update(
        XFuser_attention="TORCH_FLASH",
        ray_cluster_namespace=legacy_namespace,
    )
    client.director_app.state.database.put_raylight_runtime_state(
        {
            "version": 2,
            "epoch": 1,
            "current": legacy_descriptor,
            "tail_prompt_id": None,
            "tail_action": None,
            "tainted": False,
        }
    )

    current = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("current-comfy-kitchen", "t2v"))},
    )
    assert current.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 2)

    barrier, current_prompt = [
        item["prompt"] for item in fake_comfy.prompts
    ]
    assert any(node.get("class_type") == "DirectorDeckRayKill" for node in barrier.values())
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill" for node in current_prompt.values()
    )

    def attention(prompt: dict) -> str:
        return next(
            node["inputs"]["XFuser_attention"]
            for node in prompt.values()
            if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
        )

    # DirectorDeckRayKill must replay the persisted old initializer verbatim instead of
    # silently changing its kernel while reconstructing the loader chain.
    assert attention(barrier) == "TORCH_FLASH"
    assert attention(current_prompt) == "TORCH_FLASH"
    assert _initializer_namespace(barrier) == legacy_namespace
    assert _initializer_namespace(current_prompt).endswith("-e2")
    assert _initializer_namespace(current_prompt) != legacy_namespace


async def test_raylight_sigma_shift_a_b_a_uses_three_actor_epochs(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200

    timelines = []
    for identity, shift in (("shift-12-a", 12.0), ("shift-8", 8.0), ("shift-12-b", 12.0)):
        timeline = _timeline(_segment(identity, "t2v"))
        timeline["sampling"]["fl2va"]["shift"] = shift
        timelines.append(timeline)
    responses = [
        await client.post("/api/timeline/jobs", json={"config": timeline})
        for timeline in timelines
    ]
    assert all(response.status_code == 200 for response in responses)
    await _wait_for_prompt_count(fake_comfy, 5)

    generation_prompts = [
        item["prompt"]
        for item in fake_comfy.prompts
        if any(node.get("class_type") == "DirectorDeckRayUNETLoader" for node in item["prompt"].values())
        and not any(node.get("class_type") == "DirectorDeckRayKill" for node in item["prompt"].values())
    ]
    assert [
        _initializer_namespace(prompt).rsplit("-e", 1)[1]
        for prompt in generation_prompts
    ] == ["1", "2", "3"]
    assert sum(
        any(node.get("class_type") == "DirectorDeckRayKill" for node in item["prompt"].values())
        for item in fake_comfy.prompts
    ) == 2


async def test_mixed_raylight_job_reuses_pool_across_families_without_barrier(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200

    response = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("fl", "t2v"), _segment("ref", "r2v")
            )
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["progress"] == 0.0
    assert response.json()["status"] == "preparing"
    await _wait_for_prompt_count(fake_comfy, 2)
    submitted_types = [
        {
            node["class_type"] for node in item["prompt"].values()
        }
        for item in fake_comfy.prompts
    ]
    # The two families share one pool now: the worker RAM cache swaps bases
    # in place, so no DirectorDeckRayKill control prompt separates them.
    assert not any("DirectorDeckRayKill" in types for types in submitted_types)
    children = client.director_app.state.database.list_job_children(
        response.json()["id"]
    )
    assert all(child["segment_ids"] for child in children)
    assert len(children) == 2

    fake_comfy.pending = []
    fake_comfy.histories.update(
        {child["prompt_id"]: _success(child) for child in children}
    )
    completed = await _reconcile(client, response.json())
    assert completed["status"] == "succeeded"
    assert completed["progress"] == 1.0


async def test_global_multi_gpu_setting_ignores_retired_per_family_backend(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    settings["models"]["fl2va"]["backend"] = "standard"
    assert (await save_legacy_settings_document(client, settings)).status_code == 200

    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("global-ray", "t2v"))},
    )

    assert response.status_code == 200, response.text
    await _wait_for_submission_jobs(client)
    stored = client.director_app.state.database.get_job(response.json()["id"])
    assert stored is not None and stored["error"] is None, (
        None if stored is None else stored["error"]
    )
    await _wait_for_prompt_count(fake_comfy, 1)
    children = client.director_app.state.database.list_job_children(
        response.json()["id"]
    )
    assert [child["backend"] for child in children] == ["raylight"]
    assert all(
        any(
            node.get("class_type") == "DirectorDeckRayUNETLoader"
            for node in submission["prompt"].values()
        )
        for submission in fake_comfy.prompts
    )
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for submission in fake_comfy.prompts
        for node in submission["prompt"].values()
    )


async def test_raylight_barrier_queue_wait_does_not_consume_execution_timeout(
    fake_comfy, monkeypatch
) -> None:
    prompt_id = "queued-ray-kill"
    fake_comfy.pending = [[0, prompt_id]]
    clock = 0.0

    class LoopClock:
        def time(self) -> float:
            return clock

    async def advance_without_waiting(_delay: float) -> None:
        nonlocal clock
        clock += 1.0
        if clock >= 3.0:
            fake_comfy.pending = []
            fake_comfy.histories[prompt_id] = {
                "status": {
                    "status_str": "success",
                    "completed": True,
                    "messages": [],
                },
                "outputs": {},
            }

    monkeypatch.setattr(director_app_module.asyncio, "get_running_loop", LoopClock)
    monkeypatch.setattr(director_app_module.asyncio, "sleep", advance_without_waiting)

    # Three seconds pending is longer than this one-second ambiguity timeout.
    # Queue visibility must keep resetting the deadline until exact successful
    # history appears.
    await _await_raylight_transition(
        fake_comfy, prompt_id, timeout_seconds=1.0
    )


async def test_raylight_barrier_rejects_contradictory_completed_history(
    fake_comfy
) -> None:
    prompt_id = "contradictory-ray-kill"
    fake_comfy.histories[prompt_id] = {
        "status": {
            "status_str": "running",
            "completed": True,
            "messages": [],
        },
        "outputs": {},
    }

    with pytest.raises(ComfyError, match="contradictory"):
        await _await_raylight_transition(
            fake_comfy,
            prompt_id,
            timeout_seconds=1.0,
        )


async def test_release_policy_reuses_live_actor_epoch_but_not_cuda_residency(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    settings["raylight_residency_policy"] = "release_after_sampling"
    assert (await save_legacy_settings_document(client, settings)).status_code == 200

    first = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("release-one", "t2v"))},
    )
    second = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("release-two", "t2v"))},
    )
    assert first.status_code == second.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 2)

    ray_prompts = [
        item["prompt"]
        for item in fake_comfy.prompts
        if any(node.get("class_type") == "DirectorDeckRayUNETLoader" for node in item["prompt"].values())
        and not any(node.get("class_type") == "DirectorDeckRayKill" for node in item["prompt"].values())
    ]
    namespaces = [_initializer_namespace(prompt) for prompt in ray_prompts]
    # The installed RayLight clear path keeps the ModelPatcher and actor
    # handles alive while unpatching/offloading CUDA weights. Comfy may cache
    # the loader output safely; the next sampler patches the same model back to
    # CUDA. This is different from keep policy only in CUDA residency, not the
    # actor epoch.
    assert [namespace.rsplit("-e", 1)[1] for namespace in namespaces] == ["1", "1"]
    assert all(
        next(
            node["inputs"]["clear_vram_after_sampling"]
            for node in prompt.values()
            if node["class_type"] == "DirectorDeckRayInitializerAdvanced"
        ) is True
        for prompt in ray_prompts
    )
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for item in fake_comfy.prompts
        for node in item["prompt"].values()
    )


async def test_switching_live_keep_pool_to_release_crosses_one_barrier_then_reuses_epoch(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    resident = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("keep-before-release", "t2v"))},
    )
    assert resident.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)

    settings["raylight_residency_policy"] = "release_after_sampling"
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    before = len(fake_comfy.prompts)
    switched = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("first-release", "t2v"))},
    )
    repeated = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("second-release", "t2v"))},
    )
    assert switched.status_code == repeated.status_code == 200
    await _wait_for_prompt_count(fake_comfy, before + 3)

    submitted = fake_comfy.prompts[before:]
    assert any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in submitted[0]["prompt"].values()
    )
    generations = [
        item["prompt"]
        for item in submitted
        if not any(
            node.get("class_type") == "DirectorDeckRayKill"
            for node in item["prompt"].values()
        )
    ]
    assert [
        _initializer_namespace(prompt).rsplit("-e", 1)[1]
        for prompt in generations
    ] == ["2", "2"]
    # A first-position transition occupies the reserved even index without
    # colliding with the already-created generation child.
    children = client.director_app.state.database.list_job_children(
        switched.json()["id"]
    )
    assert [child["group_index"] for child in children] == [0, 1]


async def test_raylight_to_standard_waits_for_full_loader_kill_barrier(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    ray = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-before-standard", "t2v"))},
    )
    assert ray.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)

    class ForbiddenProductionHostObservation:
        def snapshot(self):
            raise AssertionError(
                "RayLight transition submission must not capture host classes"
            )

    client.director_app.state.host_capability_provider = (
        ForbiddenProductionHostObservation()
    )

    settings["multi_gpu_enabled"] = False
    settings["models"]["fl2va"].update(
        backend="standard",  # obsolete value is deliberately ignored
        raylight={
            "gpu_select": [0],
            "ulysses_degree": 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    standard = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("standard-after-ray", "t2v"))},
    )
    assert standard.status_code == 200, standard.text
    await _wait_for_prompt_count(fake_comfy, 3)

    barrier = fake_comfy.prompts[-2]["prompt"]
    target = fake_comfy.prompts[-1]["prompt"]
    assert {
        "DirectorDeckRayInitializerAdvanced",
        "DirectorDeckRayUNETLoader",
        "DirectorDeckRayKill",
    } <= {node["class_type"] for node in barrier.values()}
    assert "UNETLoader" in {node["class_type"] for node in target.values()}
    assert "DirectorDeckRayKill" not in {node["class_type"] for node in target.values()}


@pytest.mark.parametrize("terminal", ["cancelled", "failed"])
async def test_cancelled_or_failed_raylight_tail_forces_barrier_before_reuse(
    client, fake_comfy, terminal: str
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"ray-{terminal}", "t2v"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    child = client.director_app.state.database.list_job_children(
        created.json()["id"]
    )[0]

    if terminal == "cancelled":
        cancelled = await client.post(f"/api/jobs/{created.json()['id']}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
    else:
        fake_comfy.pending = []
        fake_comfy.histories[child["prompt_id"]] = _failure("ray actor failed")
        failed = await _reconcile(client, created.json())
        assert failed["status"] == "failed"

    fake_comfy.auto_complete_raylight = True

    before = len(fake_comfy.prompts)
    retried = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"ray-{terminal}-retry", "t2v"))},
    )
    assert retried.status_code == 200, retried.text
    await _wait_for_prompt_count(fake_comfy, before + 2)
    submitted = fake_comfy.prompts[before:]
    assert any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in submitted[0]["prompt"].values()
    )
    retry_prompt = submitted[-1]["prompt"]
    assert _initializer_namespace(retry_prompt).endswith("-e2")


async def test_raylight_runtime_epoch_and_taint_survive_database_restart(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-before-restart", "t2v"))},
    )
    assert created.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    origin = "embedded"
    state = database.get_raylight_runtime_state()
    assert state is not None and state["epoch"] == 1
    state["tainted"] = True
    database.put_raylight_runtime_state(state)
    # Model a terminal actor failure whose exact history was pruned before the
    # restart. The durable child, rather than absence alone, certifies taint.
    fake_comfy.histories.pop(str(state["tail_prompt_id"]), None)
    child = database.list_job_children(created.json()["id"])[0]
    database.update_job_child(
        child["id"],
        status="failed",
        progress=1.0,
        stage="failed",
        error="ray actor failed before restart",
        completed_at=utc_now(),
    )
    database.update_job(
        created.json()["id"],
        status="failed",
        progress=1.0,
        stage="segments_failed",
        error="ray actor failed before restart",
        completed_at=utc_now(),
    )

    # initialize() is the startup migration path of a new backend process.
    database.initialize()
    restarted = database.get_raylight_runtime_state()
    assert restarted is not None
    assert restarted["epoch"] == 1
    assert restarted["tainted"] is True

    retried = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-after-restart", "t2v"))},
    )
    assert retried.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 3)
    assert any(
        node.get("class_type") == "DirectorDeckRayKill"
        for item in fake_comfy.prompts
        for node in item["prompt"].values()
    )


async def test_raylight_exact_terminal_result_is_persisted_with_the_runtime_tail(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-terminal-certificate", "t2v"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_submission_jobs(client)

    database = client.director_app.state.database
    origin = "embedded"
    state = database.get_raylight_runtime_state()
    assert state is not None
    assert state["tail_terminal_certificate"] == {
        "prompt_id": state["tail_prompt_id"],
        "action": "ray_unit",
        "succeeded": True,
    }


async def test_contradictory_raylight_generation_never_mints_terminal_certificate(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-contradictory-history", "t2v"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    child = client.director_app.state.database.list_job_children(
        created.json()["id"]
    )[0]
    fake_comfy.pending = []
    fake_comfy.histories[str(child["prompt_id"])] = {
        "status": {
            "status_str": "running",
            "completed": False,
            "messages": [["execution_interrupted", {"node_id": "sampler"}]],
        },
        "outputs": {},
    }
    await _wait_for_submission_jobs(client)

    database = client.director_app.state.database
    origin = "embedded"
    state = database.get_raylight_runtime_state()
    assert state is not None
    assert state["tainted"] is True
    assert "tail_terminal_certificate" not in state


async def test_legacy_direct_raylight_descriptor_requires_restart_certificate(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    database = client.director_app.state.database
    origin = "embedded"
    legacy = {
        "version": 1,
        "family": "fl2va",
        "compatibility_key": "legacy-direct-key",
        "runtime_namespace": "legacy-direct-key-e7",
    }
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO raylight_runtime_state(singleton, descriptor, updated_at) "
            "VALUES(?, ?, ?)",
            (1, json.dumps(legacy), utc_now()),
        )

    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("legacy-direct-next", "t2v"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_submission_jobs(client)
    parent = database.get_job(created.json()["id"])
    assert parent is not None
    assert parent["status"] == "failed"
    assert parent["stage"] == "preflight_failed"
    assert "raylight_runtime_restart_confirmation_required" in parent["error"]
    assert fake_comfy.prompts == []


@pytest.mark.parametrize("legacy_shape", ["direct", "envelope"])
async def test_legacy_unknown_raylight_pool_does_not_gate_standard(
    client, fake_comfy, legacy_shape: str
) -> None:
    database = client.director_app.state.database
    legacy_descriptor = {
        "version": 1,
        "family": "fl2va",
        "compatibility_key": "legacy-unknown-key",
        "runtime_namespace": "legacy-unknown-key-e7",
    }
    raw_state = (
        legacy_descriptor
        if legacy_shape == "direct"
        else {
            "version": 1,
            "epoch": 7,
            "current": legacy_descriptor,
            "tail_prompt_id": "unverified-legacy-tail",
            "tainted": False,
        }
    )
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO raylight_runtime_state(singleton, descriptor, updated_at) "
            "VALUES(?, ?, ?)",
            (1, json.dumps(raw_state), utc_now()),
        )

    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"legacy-{legacy_shape}-standard", "t2v"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_submission_jobs(client)
    parent = database.get_job(created.json()["id"])
    assert parent is not None and parent["status"] == "queued"
    assert len(fake_comfy.prompts) == 1
    assert database.get_raylight_runtime_state() is not None


async def test_standard_does_not_adopt_certified_old_shutdown_tail(
    client, fake_comfy
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    ray = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("ray-before-crashed-barrier", "t2v"))},
    )
    assert ray.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    origin = "embedded"
    state = database.get_raylight_runtime_state()
    assert state is not None and state["current"] is not None

    # Model the durable write made immediately before POST /prompt when the
    # backend then dies after DirectorDeckRayKill reaches successful history but before it
    # can clear the ledger synchronously.
    barrier_prompt_id = "barrier-completed-before-restart"
    database.put_raylight_runtime_state({
            "version": 2,
            "epoch": state["epoch"],
            "current": state["current"],
            "tail_prompt_id": barrier_prompt_id,
            "tail_action": "shutdown",
            "tainted": True,
        },
    )
    fake_comfy.histories[barrier_prompt_id] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {},
    }
    settings["multi_gpu_enabled"] = False
    settings["models"]["fl2va"].update(
        backend="standard",  # obsolete value is deliberately ignored
        raylight={
            "gpu_select": [0],
            "ulysses_degree": 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    assert (await save_legacy_settings_document(client, settings)).status_code == 200

    before = len(fake_comfy.prompts)
    standard = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("standard-after-restart", "t2v"))},
    )
    assert standard.status_code == 200, standard.text
    await _wait_for_prompt_count(fake_comfy, before + 1)
    submitted = fake_comfy.prompts[before:]
    assert len(submitted) == 1
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in submitted[0]["prompt"].values()
    )
    settled = database.get_raylight_runtime_state()
    assert settled is not None
    assert settled["current"] == state["current"]
    assert settled["tail_prompt_id"] == barrier_prompt_id
    assert settled["tail_action"] == "shutdown"
    assert settled["tainted"] is True


@pytest.mark.parametrize("old_barrier_result", ["failed", "absent"])
async def test_standard_does_not_adopt_dead_shutdown_tail(
    client, fake_comfy, old_barrier_result: str
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    ray = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"resident-{old_barrier_result}", "t2v"))},
    )
    assert ray.status_code == 200
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    origin = "embedded"
    state = database.get_raylight_runtime_state()
    assert state is not None and state["current"] is not None
    dead_prompt_id = f"dead-shutdown-{old_barrier_result}"
    database.put_raylight_runtime_state({
            "version": 2,
            "epoch": state["epoch"],
            "current": state["current"],
            "tail_prompt_id": dead_prompt_id,
            "tail_action": "shutdown",
            "tainted": True,
        },
    )
    if old_barrier_result == "failed":
        fake_comfy.histories[dead_prompt_id] = _failure("old DirectorDeckRayKill failed")

    settings["multi_gpu_enabled"] = False
    settings["models"]["fl2va"].update(
        backend="standard",
        raylight={
            "gpu_select": [0],
            "ulysses_degree": 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    before = len(fake_comfy.prompts)
    standard = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"standard-{old_barrier_result}", "t2v"))},
    )
    assert standard.status_code == 200
    await _wait_for_prompt_count(fake_comfy, before + 1)
    submitted = fake_comfy.prompts[before:]
    assert len(submitted) == 1
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in submitted[0]["prompt"].values()
    )
    preserved = database.get_raylight_runtime_state()
    assert preserved is not None
    assert preserved["tail_prompt_id"] == dead_prompt_id
    assert preserved["tail_action"] == "shutdown"
    assert preserved["tainted"] is True


@pytest.mark.parametrize("backend", ["standard", "raylight"])
async def test_continuity_terminal_gates_and_binds_exact_predecessor_output(
    client, fake_comfy, monkeypatch, backend: str
) -> None:
    if backend == "raylight":
        settings = _raylight_settings_document()
        assert (await save_legacy_settings_document(client, settings)).status_code == 200
        fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment(f"{backend}-continuity-first", "t2v"),
                _segment(
                    f"{backend}-continuity-second",
                    "r2v" if backend == "standard" else "t2v",
                ),
            )
        },
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    children = database.list_job_children(created.json()["id"])
    first, second = children
    if backend == "standard":
        assert (first["family"], second["family"]) == ("fl2va", "ref2va")
    assert first["prompt_id"] == fake_comfy.prompts[0]["prompt_id"]
    assert second["prompt_id"] is None

    _complete_fake_prompt(fake_comfy, first)
    await _wait_for_prompt_count(fake_comfy, 2)
    submitted_successor = fake_comfy.prompts[1]["prompt"]
    continuity_files = [
        node["inputs"]["file"]
        for node in submitted_successor.values()
        if node.get("class_type") == "LoadVideo"
        and isinstance(node.get("inputs"), dict)
        and str(node["inputs"].get("file") or "").endswith(" [output]")
    ]
    assert continuity_files == [
        f"segments/{backend}-continuity-first.mp4 [output]"
    ]
    stored_second = database.get_job_child(second["id"])
    assert stored_second is not None
    assert any(
        node.get("class_type") == "LoadVideo"
        and node.get("inputs", {}).get("file") == continuity_files[0]
        for node in stored_second["prompt_snapshot"].values()
    )


@pytest.mark.parametrize(
    ("status_str", "completed"),
    [("success", False), ("pending", True)],
)
async def test_standard_continuity_gate_rejects_contradictory_exact_history(
    client,
    fake_comfy,
    monkeypatch,
    status_str: str,
    completed: bool,
) -> None:
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment(f"contradictory-gate-{status_str}-root", "t2v"),
                _segment(f"contradictory-gate-{status_str}-child", "t2v"),
            )
        },
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    first, second = database.list_job_children(created.json()["id"])
    ownership_before = database.get_prompt_ownership(first["id"])
    assert ownership_before is not None

    fake_comfy.pending = []
    fake_comfy.histories[str(first["prompt_id"])] = {
        "status": {
            "status_str": status_str,
            "completed": completed,
            "messages": [],
        },
        "outputs": {},
    }
    await _wait_for_submission_jobs(client)

    assert len(fake_comfy.prompts) == 1
    ownership_after = database.get_prompt_ownership(first["id"])
    assert ownership_after is not None
    assert ownership_after.state != "terminal_confirmed"
    assert database.get_observed_artifact(first["id"]) is None
    stored_second = database.get_job_child(second["id"])
    assert stored_second is not None
    assert stored_second["prompt_id"] is None


@pytest.mark.parametrize(
    "terminal", ["failed", "cancelled", "missing_output", "ambiguous_output"]
)
async def test_continuity_predecessor_terminal_failure_marks_all_descendants_without_post(
    client, fake_comfy, monkeypatch, terminal: str
) -> None:
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment(f"{terminal}-root", "t2v"),
                _segment(f"{terminal}-child", "t2v"),
                _segment(f"{terminal}-grandchild", "t2v"),
            )
        },
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    root = database.list_job_children(created.json()["id"])[0]
    prompt_id = str(root["prompt_id"])
    fake_comfy.pending = []
    if terminal == "failed":
        fake_comfy.histories[prompt_id] = _failure("continuity root failed")
    elif terminal == "cancelled":
        fake_comfy.histories[prompt_id] = {
            "status": {
                "status_str": "error",
                "completed": True,
                "messages": [
                    ["execution_interrupted", {"prompt_id": prompt_id}]
                ],
            },
            "outputs": {},
        }
    elif terminal == "missing_output":
        fake_comfy.histories[prompt_id] = {
            "status": {
                "status_str": "success",
                "completed": True,
                "messages": [],
            },
            "outputs": {},
        }
    elif terminal == "ambiguous_output":
        output_node = root["output_nodes"][root["segment_ids"][0]]
        fake_comfy.histories[prompt_id] = {
            "status": {
                "status_str": "success",
                "completed": True,
                "messages": [],
            },
            "outputs": {
                output_node: {
                    "videos": [
                        {
                            "filename": "take-a.mp4",
                            "subfolder": "segments",
                            "type": "output",
                        },
                        {
                            "filename": "take-b.mp4",
                            "subfolder": "segments",
                            "type": "output",
                        },
                    ]
                }
            },
        }

    await _wait_for_submission_jobs(client)
    assert len(fake_comfy.prompts) == 1
    children = database.list_job_children(created.json()["id"])
    descendants = children[1:]
    assert [child["status"] for child in descendants] == ["failed", "failed"]
    assert [child["stage"] for child in descendants] == [
        "dependency_failed",
        "dependency_failed",
    ]
    assert all(child["prompt_id"] is None for child in descendants)
    assert all("continuity dependency failed" in child["error"] for child in descendants)


async def test_continuity_raylight_failure_never_posts_descendant_or_cleanup_barrier(
    client, fake_comfy, monkeypatch
) -> None:
    settings = _raylight_settings_document()
    assert (await save_legacy_settings_document(client, settings)).status_code == 200
    fake_comfy.auto_complete_raylight = False
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment("ray-continuity-failed-root", "t2v"),
                _segment("ray-continuity-never-posted", "t2v"),
            )
        },
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    root = database.list_job_children(created.json()["id"])[0]
    fake_comfy.pending = []
    fake_comfy.histories[str(root["prompt_id"])] = _failure("ray predecessor failed")

    await _wait_for_submission_jobs(client)
    assert len(fake_comfy.prompts) == 1
    assert not any(
        node.get("class_type") == "DirectorDeckRayKill"
        for node in fake_comfy.prompts[0]["prompt"].values()
    )
    descendant = database.list_job_children(created.json()["id"])[1]
    assert descendant["status"] == "failed"
    assert descendant["stage"] == "dependency_failed"
    assert descendant["prompt_id"] is None


async def test_continuity_selection_must_be_predecessor_closed_with_zero_posts(
    client, fake_comfy
) -> None:
    response = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment("selection-root", "t2v"),
                _segment("selection-successor", "t2v"),
            ),
            "segment_ids": ["selection-successor"],
        },
    )
    assert response.status_code == 200, response.text
    await _wait_for_submission_jobs(client)
    stored = client.director_app.state.database.get_job(response.json()["id"])
    assert stored is not None and stored["status"] == "failed"
    assert stored["stage"] == "compile_failed"
    assert "historical_take_required" in stored["error"]
    assert "selection-root" not in stored["error"]
    assert "selection-successor" not in stored["error"]
    assert fake_comfy.prompts == []


async def test_continuity_cancel_after_predecessor_success_closes_unclaimed_successor(
    client, fake_comfy, monkeypatch
) -> None:
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    successor_claim_started = asyncio.Event()
    release_successor_claim = asyncio.Event()
    successor_claim_finished = asyncio.Event()
    cancel_intent_marked = asyncio.Event()
    release_cancel_owner = asyncio.Event()

    async def pause_successor_claim(_job_id: str, child_id: str) -> None:
        child = client.director_app.state.database.get_job_child(child_id)
        if child is not None and child["segment_ids"] == ["cancel-successor"]:
            successor_claim_started.set()
            await release_successor_claim.wait()

    client.director_app.state.before_submission_claim = pause_successor_claim
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment("cancel-root", "t2v"),
                _segment("cancel-successor", "t2v"),
            )
        },
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    database = client.director_app.state.database
    root = database.list_job_children(created.json()["id"])[0]
    _complete_fake_prompt(fake_comfy, root)
    await asyncio.wait_for(successor_claim_started.wait(), timeout=1)

    original_claim = database.persist_job_child_submission_intent

    def observe_successor_claim(*args, **kwargs):
        locked_plan = kwargs["locked_plan"]
        segment_unit = locked_plan.units[-1]
        try:
            return original_claim(*args, **kwargs)
        finally:
            child = database.get_job_child(str(segment_unit.child_id))
            if child is not None and child["segment_ids"] == ["cancel-successor"]:
                successor_claim_finished.set()

    monkeypatch.setattr(
        database,
        "persist_job_child_submission_intent",
        observe_successor_claim,
    )
    original_cancel = director_app_module._cancel_timeline_job

    async def pause_after_cancel_intent(request, job, *, initial_cancel_claimed=False):
        cancel_intent_marked.set()
        await release_cancel_owner.wait()
        return await original_cancel(
            request,
            job,
            initial_cancel_claimed=initial_cancel_claimed,
        )

    monkeypatch.setattr(
        director_app_module,
        "_cancel_timeline_job",
        pause_after_cancel_intent,
    )

    cancel_request = asyncio.create_task(
        client.post(f"/api/jobs/{created.json()['id']}/cancel")
    )
    await asyncio.wait_for(cancel_intent_marked.wait(), timeout=1)
    marked_parent = database.get_job(created.json()["id"])
    assert marked_parent is not None
    assert marked_parent["status"] == "preparing"
    assert bool(marked_parent["cancel_requested"]) is True

    # Releasing the successor exactly after the durable intent bit is set but
    # before the cancel owner advances the parent must still make the atomic
    # child claim fail. No successor prompt may cross the network in this gap.
    release_successor_claim.set()
    await asyncio.wait_for(successor_claim_finished.wait(), timeout=1)
    successor = database.list_job_children(created.json()["id"])[1]
    assert successor["prompt_id"] is None
    assert len(fake_comfy.prompts) == 1

    release_cancel_owner.set()
    cancelled = await asyncio.wait_for(cancel_request, timeout=3)
    assert cancelled.status_code == 200
    await _wait_for_submission_jobs(client)
    assert len(fake_comfy.prompts) == 1
    successor = database.list_job_children(created.json()["id"])[1]
    assert successor["status"] == "cancelled"
    assert successor["prompt_id"] is None


async def test_continuity_restart_does_not_resume_an_unsubmitted_successor(
    client, fake_comfy, monkeypatch
) -> None:
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    successor_claim_started = asyncio.Event()
    release_dead_dispatcher = asyncio.Event()

    async def pause_successor_claim(_job_id: str, child_id: str) -> None:
        child = client.director_app.state.database.get_job_child(child_id)
        if child is not None and child["segment_ids"] == ["restart-successor"]:
            successor_claim_started.set()
            await release_dead_dispatcher.wait()

    client.director_app.state.before_submission_claim = pause_successor_claim
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment("restart-root", "t2v"),
                _segment("restart-successor", "t2v"),
            )
        },
    )
    assert created.status_code == 200, created.text
    database = client.director_app.state.database
    await _wait_for_prompt_count(fake_comfy, 1)
    root, successor = database.list_job_children(created.json()["id"])
    _complete_fake_prompt(fake_comfy, root)
    await asyncio.wait_for(successor_claim_started.wait(), timeout=1)

    root = database.get_job_child(root["id"])
    successor = database.get_job_child(successor["id"])
    assert root is not None and root["status"] == "succeeded"
    assert successor is not None and successor["status"] == "preparing"
    assert successor["prompt_id"] is None
    submitted_before_restart = len(fake_comfy.prompts)

    # The old dispatcher is deliberately left suspended, modelling the durable
    # SQLite shape after SIGKILL: the predecessor has a certified take while no
    # successor POST has been claimed. A new process must cancel that local row,
    # never reconstruct/bind the dependency and resume generation automatically.
    restarted = create_app(
        comfy_url="http://comfy.test:8188",
        database_path=database.path,
        comfy_factory=lambda _comfy_url: fake_comfy,
    )
    restarted.state.reconcile_interval_seconds = 0.01
    monkeypatch.setattr(restarted.state.progress_manager, "ensure", Mock())
    monkeypatch.setattr(restarted.state.progress_manager, "close", AsyncMock())
    try:
        async with restarted.router.lifespan_context(restarted):
            await _wait_until(
                lambda: (
                    (stored := restarted.state.database.get_job(created.json()["id"]))
                    is not None
                    and stored["status"] in {"failed", "cancelled"}
                )
            )

        recovered_parent = restarted.state.database.get_job(created.json()["id"])
        recovered_children = restarted.state.database.list_job_children(
            created.json()["id"]
        )
        assert recovered_parent is not None
        assert recovered_parent["status"] == "cancelled"
        assert recovered_children[0]["status"] == "succeeded"
        assert recovered_children[1]["status"] == "cancelled"
        assert (
            recovered_children[1]["stage"]
            == "restart_cancelled_not_submitted"
        )
        assert recovered_children[1]["prompt_id"] is None
        assert len(fake_comfy.prompts) == submitted_before_restart == 1
    finally:
        release_dead_dispatcher.set()
        await _wait_for_submission_jobs(client)

    assert len(fake_comfy.prompts) == 1


async def test_continuity_failure_propagation_stops_at_explicit_anchor_reset(
    client, fake_comfy, monkeypatch
) -> None:
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _continuity_timeline(
                _segment("failed-chain-root", "t2v"),
                _segment("failed-chain-child", "t2v"),
                _segment("fresh-anchor", "i2v"),
                _segment("fresh-successor", "t2v"),
            )
        },
    )
    assert created.status_code == 200, created.text
    database = client.director_app.state.database
    await _wait_for_prompt_count(fake_comfy, 1)
    root, failed_child, anchor, successor = database.list_job_children(
        created.json()["id"]
    )

    root_prompt_id = str(root["prompt_id"])
    fake_comfy.pending = [
        item for item in fake_comfy.pending if root_prompt_id not in item
    ]
    fake_comfy.histories[root_prompt_id] = _failure("first continuity chain failed")

    # B is a dependent failure, but C has an explicit first image and starts a
    # fresh chain. It must still be submitted instead of being swept into A's
    # transitive dependency failure set.
    await _wait_for_prompt_count(fake_comfy, 2)
    failed_child = database.get_job_child(failed_child["id"])
    anchor = database.get_job_child(anchor["id"])
    assert failed_child is not None
    assert failed_child["status"] == "failed"
    assert failed_child["stage"] == "dependency_failed"
    assert failed_child["prompt_id"] is None
    assert anchor is not None
    assert anchor["prompt_id"] == fake_comfy.prompts[1]["prompt_id"]

    _complete_fake_prompt(fake_comfy, anchor)
    await _wait_for_prompt_count(fake_comfy, 3)
    successor_prompt = fake_comfy.prompts[2]["prompt"]
    assert any(
        node.get("class_type") == "LoadVideo"
        and node.get("inputs", {}).get("file")
        == "segments/fresh-anchor.mp4 [output]"
        for node in successor_prompt.values()
    )

    successor = database.get_job_child(successor["id"])
    assert successor is not None and successor["prompt_id"] is not None
    _complete_fake_prompt(fake_comfy, successor)
    await _wait_for_submission_jobs(client)
    await _reconcile(client, created.json())

    children = database.list_job_children(created.json()["id"])
    assert [child["status"] for child in children] == [
        "failed",
        "failed",
        "succeeded",
        "succeeded",
    ]
    assert len(fake_comfy.prompts) == 3


async def test_continuity_dependency_failure_aggregates_parent_without_assembly(
    client, fake_comfy, monkeypatch
) -> None:
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    assemble = AsyncMock(
        side_effect=AssertionError("failed continuity jobs must not start assembly")
    )
    monkeypatch.setattr(director_app_module, "_assemble_timeline_output", assemble)
    document = _continuity_timeline(
        _segment("aggregate-root", "t2v"),
        _segment("aggregate-failed", "t2v"),
        _segment("aggregate-dependent", "t2v"),
    )
    document["export_mode"] = "all"
    created = await client.post(
        "/api/timeline/jobs", json={"config": document}
    )
    assert created.status_code == 200, created.text
    database = client.director_app.state.database
    await _wait_for_prompt_count(fake_comfy, 1)
    root, failed, dependent = database.list_job_children(created.json()["id"])
    _complete_fake_prompt(fake_comfy, root)

    await _wait_for_prompt_count(fake_comfy, 2)
    failed = database.get_job_child(failed["id"])
    assert failed is not None and failed["prompt_id"] is not None
    failed_prompt_id = str(failed["prompt_id"])
    fake_comfy.pending = [
        item for item in fake_comfy.pending if failed_prompt_id not in item
    ]
    fake_comfy.histories[failed_prompt_id] = _failure("middle segment failed")

    await _wait_for_submission_jobs(client)
    reconciled = await _reconcile(
        client, created.json(), allow_assembly=True
    )
    assert reconciled["status"] == "failed"
    assert reconciled["stage"] == "segments_failed"
    assert reconciled["outputs"] == []
    assemble.assert_not_awaited()
    assert len(fake_comfy.prompts) == 2

    dependent = database.get_job_child(dependent["id"])
    assert dependent is not None
    assert dependent["status"] == "failed"
    assert dependent["stage"] == "dependency_failed"
    assert dependent["prompt_id"] is None

    # Parent failure must not erase an already certified predecessor take. The
    # task drawer still exposes that exact segment result even though no full
    # movie can be assembled.
    response = await client.get(f"/api/jobs/{created.json()['id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert [result["segment_id"] for result in body["segment_results"]] == [
        "aggregate-root"
    ]
