from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from starlette.requests import Request

from directordeck.app import (
    _is_full_timeline_selection,
    _segment_results,
    _sync_timeline_job,
    create_app,
)
from directordeck.media import VideoProxy
from directordeck.progress import ComfyPreviewEvent, ComfyProgressEvent
from directordeck.schemas import (
    RuntimeSettings,
    VideoMetadata,
    default_settings,
    default_timeline_draft,
    utc_now,
)

from .conftest import VIDEO_METADATA, asset, wait_for_submission_tasks


def _background_request(app) -> Request:
    return Request({"type": "http", "app": app})


async def _reconcile_timeline(
    client, parent: dict, *, allow_assembly: bool = False
) -> dict:
    database = client.director_app.state.database
    latest = database.get_job(parent["id"])
    assert latest is not None
    return await _sync_timeline_job(
        _background_request(client.director_app),
        latest,
        allow_assembly=allow_assembly,
    )


def _segment(identity: str) -> dict:
    return {
        "id": identity,
        "title": identity,
        "mode": "t2v",
        "duration_seconds": 5.0,
        "prompt": f"Prompt for {identity}",
        "enabled": True,
    }


def _timeline(*segment_ids: str) -> dict:
    document = default_timeline_draft().model_dump(mode="json")
    for sampling in document["sampling"].values():
        sampling["seed"] = 42
        sampling["random_seed"] = False
    document["segments"] = [_segment(identity) for identity in segment_ids]
    document["version"] = 1
    return document


def test_full_run_is_the_explicit_set_of_every_enabled_segment() -> None:
    timeline = _timeline("first", "second", "disabled")
    timeline["segments"][2]["enabled"] = False

    assert _is_full_timeline_selection(
        {"timeline": timeline, "segment_ids": ["second", "first"]}
    )
    assert _is_full_timeline_selection({"timeline": timeline, "segment_ids": None})
    assert not _is_full_timeline_selection(
        {"timeline": timeline, "segment_ids": ["first"]}
    )
    assert not _is_full_timeline_selection(
        {"timeline": timeline, "segment_ids": ["first", "second", "disabled"]}
    )


async def _create_timeline_job(client, *segment_ids: str) -> tuple[dict, list[dict]]:
    response = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(*segment_ids)},
    )
    assert response.status_code == 200, response.text
    accepted = response.json()
    await wait_for_submission_tasks(client)
    refreshed = await client.get(f"/api/jobs/{accepted['id']}")
    assert refreshed.status_code == 200, refreshed.text
    parent = refreshed.json()
    children = client.director_app.state.database.list_job_children(parent["id"])
    assert len(children) == len(segment_ids)
    assert all(len(child["segment_ids"]) == 1 for child in children)
    return parent, children


def _success_history(
    child: dict,
    *,
    omit: set[str] | None = None,
    duplicate: set[str] | None = None,
    reverse: bool = False,
) -> dict:
    omitted = omit or set()
    duplicated = duplicate or set()
    pairs = list(child["output_nodes"].items())
    if reverse:
        pairs.reverse()
    outputs: dict[str, dict] = {}
    for segment_id, node_id in pairs:
        if segment_id in omitted:
            continue
        videos = [
            {
                "filename": f"{segment_id}.mp4",
                "subfolder": "segments",
                "type": "output",
            }
        ]
        if segment_id in duplicated:
            videos.append(
                {
                    "filename": f"{segment_id}-alternate.mp4",
                    "subfolder": "segments",
                    "type": "output",
                }
            )
        outputs[str(node_id)] = {"videos": videos}
    return {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": outputs,
    }


def test_segment_results_follow_timeline_selection_and_fail_closed_on_duplicates() -> None:
    job = {
        "id": "parent/one",
        "config_snapshot": {
            "timeline": _timeline("first", "second", "third"),
            # Selection order is an execution request detail; edit order is
            # authoritative when takes are presented back to the timeline.
            "segment_ids": ["third", "first"],
        },
        "children": [
            {
                "id": "child-ref",
                "segment_ids": ["third"],
                "output_nodes": {"third": "90"},
                "outputs": [
                    {
                        "node_id": "90",
                        "filename": "third.mp4",
                        "subfolder": "segments",
                        "type": "output",
                    }
                ],
            },
            {
                "id": "child-fl",
                "segment_ids": ["first"],
                "output_nodes": {"first": "20"},
                # Reverse/duplicate history order must not become identity.
                "outputs": [
                    {
                        "node_id": "20",
                        "filename": "first-b.mp4",
                        "subfolder": "segments",
                        "type": "output",
                    },
                    {
                        "node_id": "20",
                        "filename": "first-a.mp4",
                        "subfolder": "segments",
                        "type": "output",
                    },
                    {
                        "node_id": "unregistered",
                        "filename": "forged.mp4",
                        "subfolder": "segments",
                        "type": "output",
                    },
                ],
            },
        ],
    }

    results = _segment_results(job)

    # Ambiguous first is omitted; unselected second and unknown node output
    # are never surfaced. The unique selected third take remains usable.
    assert results == [
        {
            "segment_id": "third",
            "child_id": "child-ref",
            "output_url": "/api/jobs/parent%2Fone/segment-output?segment_id=third",
            "output_file": "output/segments/third.mp4",
            "current_snapshot": False,
        }
    ]


async def test_segment_candidate_currentness_requires_exact_authority_snapshots(
    client, fake_comfy
) -> None:
    draft = _timeline("current")
    assert (await client.put("/api/timeline", json=draft)).status_code == 200
    parent, children = await _create_timeline_job(client, "current")
    child = children[0]
    fake_comfy.pending = []
    fake_comfy.histories[child["prompt_id"]] = _success_history(child)

    await _reconcile_timeline(client, parent)
    completed = await client.get(f"/api/jobs/{parent['id']}")
    assert completed.status_code == 200, completed.text
    assert completed.json()["segment_results"][0]["current_snapshot"] is True

    edited = _timeline("current")
    edited["segments"][0]["prompt"] = "A changed prompt"
    assert (await client.put("/api/timeline", json=edited)).status_code == 200
    historical = await client.get(f"/api/jobs/{parent['id']}")
    assert historical.json()["segment_results"][0]["current_snapshot"] is False

    # Restoring the creative document is insufficient if any runtime setting
    # changed; exact model/endpoint/device selection is part of take identity.
    assert (await client.put("/api/timeline", json=draft)).status_code == 200
    settings = (await client.get("/api/settings")).json()
    settings["client_id"] = "candidate-settings-changed"
    assert (await client.put("/api/settings", json=settings)).status_code == 200
    stale_runtime = await client.get(f"/api/jobs/{parent['id']}")
    assert stale_runtime.json()["segment_results"][0]["current_snapshot"] is False


async def test_invalid_historical_snapshot_keeps_take_but_never_marks_current(
    client, fake_comfy
) -> None:
    draft = _timeline("legacy")
    assert (await client.put("/api/timeline", json=draft)).status_code == 200
    parent, children = await _create_timeline_job(client, "legacy")
    child = children[0]
    fake_comfy.pending = []
    fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    await _reconcile_timeline(client, parent)
    assert (await client.get(f"/api/jobs/{parent['id']}")).status_code == 200

    # Keep just enough historical structure to map the take while deliberately
    # making it invalid under today's strict UnifiedTimelineDraft schema.
    import json
    import sqlite3

    with sqlite3.connect(client.director_app.state.database.path) as connection:
        connection.execute(
            "UPDATE jobs SET config_snapshot = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "timeline": {"segments": draft["segments"]},
                        "segment_ids": None,
                    }
                ),
                parent["id"],
            ),
        )

    response = await client.get(f"/api/jobs/{parent['id']}")
    assert response.status_code == 200, response.text
    assert response.json()["segment_results"] == [
        {
            "segment_id": "legacy",
            "child_id": child["id"],
            "output_url": (
                f"/api/jobs/{parent['id']}/segment-output?segment_id=legacy"
            ),
            "output_file": "output/segments/legacy.mp4",
            "current_snapshot": False,
        }
    ]


async def test_job_list_prefetches_snapshot_authorities_once(client, monkeypatch) -> None:
    database = client.director_app.state.database
    now = utc_now()
    for identity in ("history-a", "history-b"):
        database.create_job(
            {
                "id": identity,
                "mode": "timeline",
                "status": "succeeded",
                "progress": 1.0,
                "stage": "completed",
                "prompt_id": None,
                "outputs": [],
                "error": None,
                "config_snapshot": {
                    "timeline": database.get_timeline().model_dump(mode="json"),
                    "segment_ids": None,
                },
                "settings_snapshot": database.get_settings().model_dump(mode="json"),
                "prompt_snapshot": {},
                "created_at": now,
                "updated_at": now,
                "started_at": now,
                "completed_at": now,
            }
        )
    get_timeline = Mock(wraps=database.get_timeline)
    get_settings = Mock(wraps=database.get_settings)
    monkeypatch.setattr(database, "get_timeline", get_timeline)
    monkeypatch.setattr(database, "get_settings", get_settings)

    response = await client.get("/api/jobs")

    assert response.status_code == 200, response.text
    assert get_timeline.call_count == 1
    assert get_settings.call_count == 1


async def test_full_multisegment_success_assembles_exact_timeline_order(
    client, fake_comfy, monkeypatch
) -> None:
    # Put Ref2VA before FL2VA in the edit timeline. Native workflow groups are
    # submitted FL2VA first, so this proves assembly follows segment order
    # across multiple child prompts instead of child/group order.
    draft = _timeline("ref-first", "fl-second")
    draft["segments"][0].update(
        mode="r2v",
        reference_images=[asset("reference.png", "image", slot=0)],
        reference_audios=[],
        reference_videos=[],
    )
    # Explicitly selecting every enabled segment is the only full-run
    # semantics the UI needs; omitting segment_ids is legacy wire shorthand.
    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": draft,
            "segment_ids": ["ref-first", "fl-second"],
        },
    )
    assert created.status_code == 200, created.text
    parent = created.json()
    await wait_for_submission_tasks(client)
    parent = (await client.get(f"/api/jobs/{parent['id']}")).json()
    children = client.director_app.state.database.list_job_children(parent["id"])
    assert [child["family"] for child in children] == ["fl2va", "ref2va"]
    for child in children:
        fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    fake_comfy.pending = []
    viewed: list[str] = []

    async def view(params: dict[str, str]) -> httpx.Response:
        filename = params["filename"]
        viewed.append(filename)
        return httpx.Response(
            200,
            content=filename.encode(),
            headers={"content-type": "video/mp4"},
            request=httpx.Request("GET", "http://comfy.test/view"),
        )

    monkeypatch.setattr(fake_comfy, "view", view)
    assembled = VideoProxy(
        content=b"assembled-long-video",
        filename_suffix=".mp4",
        metadata=VideoMetadata.model_validate(VIDEO_METADATA),
    )
    assemble = Mock(return_value=assembled)
    monkeypatch.setattr("directordeck.app.assemble_video_bytes", assemble)

    database = client.director_app.state.database
    background_result = await _sync_timeline_job(
        _background_request(client.director_app),
        database.get_job(parent["id"]),
        allow_assembly=True,
    )
    assert background_result["status"] == "succeeded"
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "succeeded"
    assert job["stage"] == "completed"
    assert job["outputs"] == [f"/api/jobs/{parent['id']}/outputs/0"]
    assert len(job["output_files"]) == 1
    assert job["output_files"][0].startswith(
        "output/directordeck/timelines/DirectorDeck_timeline_"
    )
    assert viewed == ["ref-first.mp4", "fl-second.mp4"]
    assemble.assert_called_once_with(
        [b"ref-first.mp4", b"fl-second.mp4"],
        fps=24.0,
        width=864,
        height=480,
    )
    assert fake_comfy.uploads[-1]["kind"] == "output"
    assert fake_comfy.uploads[-1]["content"] == b"assembled-long-video"
    stored = client.director_app.state.database.get_job(parent["id"])
    assert stored is not None
    assert stored["outputs"] == [
        {
            "node_id": "assembly",
            "filename": fake_comfy.uploads[-1]["filename"],
            "subfolder": "directordeck/timelines",
            "type": "output",
        }
    ]
    assert [item["segment_id"] for item in job["segment_results"]] == [
        "ref-first",
        "fl-second",
    ]
    expected_children = {
        segment_id: child["id"]
        for child in children
        for segment_id in child["segment_ids"]
    }
    assert {
        item["segment_id"]: item["child_id"]
        for item in job["segment_results"]
    } == expected_children
    assert all("workflow" not in item for item in job["segment_results"])

    take = await client.get(
        job["segment_results"][0]["output_url"]
    )
    assert take.status_code == 200
    assert take.content == b"ref-first.mp4"
    assert take.headers["content-type"].startswith("video/mp4")


async def test_background_status_pass_can_reconcile_without_starting_assembly(
    client, fake_comfy, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "first", "second")
    for child in children:
        fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    fake_comfy.pending = []
    assemble = AsyncMock(side_effect=AssertionError("HTTP read must not assemble"))
    monkeypatch.setattr("directordeck.app._assemble_timeline_output", assemble)

    await _reconcile_timeline(client, parent)
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "running"
    assert response.json()["stage"] == "segments_ready"
    assert [item["segment_id"] for item in response.json()["segment_results"]] == [
        "first",
        "second",
    ]
    assemble.assert_not_awaited()


async def test_single_segment_success_reuses_exact_native_output_without_assembly(
    client, fake_comfy, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "only")
    child = children[0]
    fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    fake_comfy.pending = []
    assemble = Mock(side_effect=AssertionError("single segment must not be re-encoded"))
    monkeypatch.setattr("directordeck.app.assemble_video_bytes", assemble)

    await _reconcile_timeline(client, parent)
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "succeeded"
    assert job["stage"] == "completed"
    assert job["output_files"] == ["output/segments/only.mp4"]
    assert assemble.call_count == 0
    stored = client.director_app.state.database.get_job(parent["id"])
    assert stored is not None
    assert stored["outputs"] == [
        {
            "node_id": child["output_nodes"]["only"],
            "filename": "only.mp4",
            "subfolder": "segments",
            "type": "output",
        }
    ]


@pytest.mark.parametrize(
    ("selected_ids", "export_mode", "expected_outputs"),
    [
        (["first"], "all", ["first.mp4"]),
        (["first", "second"], "segments", ["first.mp4", "second.mp4"]),
    ],
)
async def test_selection_or_output_policy_can_keep_results_as_individual_segments(
    client,
    fake_comfy,
    monkeypatch,
    selected_ids: list[str],
    export_mode: str,
    expected_outputs: list[str],
) -> None:
    draft = _timeline("first", "second")
    draft["export_mode"] = export_mode
    response = await client.post(
        "/api/timeline/jobs",
        json={"config": draft, "segment_ids": selected_ids},
    )
    assert response.status_code == 200, response.text
    parent = response.json()
    await wait_for_submission_tasks(client)
    children = client.director_app.state.database.list_job_children(parent["id"])
    assert [segment for child in children for segment in child["segment_ids"]] == selected_ids
    for child in children:
        fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    fake_comfy.pending = []
    assemble = Mock(side_effect=AssertionError("individual segments must not assemble"))
    monkeypatch.setattr("directordeck.app.assemble_video_bytes", assemble)

    result = await _sync_timeline_job(
        _background_request(client.director_app),
        client.director_app.state.database.get_job(parent["id"]),
        allow_assembly=True,
    )

    assert result["status"] == "succeeded"
    assert result["stage"] == "segments_completed"
    assert [output["filename"] for output in result["outputs"]] == expected_outputs
    assert assemble.call_count == 0


async def test_multisegment_missing_output_fails_before_assembly(
    client, fake_comfy, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "first", "second")
    for child in children:
        fake_comfy.histories[child["prompt_id"]] = _success_history(
            child, omit={"second"}
        )
    fake_comfy.pending = []
    assemble = Mock(side_effect=AssertionError("assembly must not start"))
    monkeypatch.setattr("directordeck.app.assemble_video_bytes", assemble)

    await _reconcile_timeline(client, parent)
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "failed"
    assert job["stage"] == "output_missing"
    assert "missing=['second']" in job["error"]
    assert job["outputs"] == []
    assert [item["segment_id"] for item in job["segment_results"]] == ["first"]
    assert assemble.call_count == 0


async def test_multisegment_duplicate_output_fails_before_assembly(
    client, fake_comfy, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "first", "second")
    for child in children:
        fake_comfy.histories[child["prompt_id"]] = _success_history(
            child, duplicate={"first"}
        )
    fake_comfy.pending = []
    assemble = Mock(side_effect=AssertionError("assembly must not start"))
    monkeypatch.setattr("directordeck.app.assemble_video_bytes", assemble)

    await _reconcile_timeline(client, parent)
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200, response.text
    job = response.json()
    assert job["status"] == "failed"
    assert job["stage"] == "output_missing"
    assert "duplicate output for segment 'first'" in job["error"]
    assert job["outputs"] == []
    assert [item["segment_id"] for item in job["segment_results"]] == ["second"]
    assert assemble.call_count == 0


async def test_http_reads_do_not_wait_for_background_timeline_assembly(
    client, fake_comfy, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "first", "second")
    for child in children:
        fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    fake_comfy.pending = []
    assembly_started = asyncio.Event()
    assembly_release = asyncio.Event()

    async def blocked_assembly(_request, _job, _outputs):
        assembly_started.set()
        await assembly_release.wait()
        return {
            "node_id": "assembly",
            "filename": "assembled.mp4",
            "subfolder": "directordeck/timelines",
            "type": "output",
        }

    monkeypatch.setattr("directordeck.app._assemble_timeline_output", blocked_assembly)
    database = client.director_app.state.database
    background = asyncio.create_task(
        _sync_timeline_job(
            _background_request(client.director_app),
            database.get_job(parent["id"]),
            allow_assembly=True,
        )
    )
    try:
        await asyncio.wait_for(assembly_started.wait(), timeout=1)
        listing, detail, take = await asyncio.wait_for(
            asyncio.gather(
                client.get("/api/jobs"),
                client.get(f"/api/jobs/{parent['id']}"),
                client.get(
                    f"/api/jobs/{parent['id']}/segment-output",
                    params={"segment_id": "first"},
                ),
            ),
            timeout=0.25,
        )

        assert listing.status_code == detail.status_code == take.status_code == 200
        listed = next(job for job in listing.json()["jobs"] if job["id"] == parent["id"])
        assert listed["stage"] == detail.json()["stage"] == "assembling"
        assert take.content == b"fake-video"
        assert not background.done()
    finally:
        assembly_release.set()

    assembled = await asyncio.wait_for(background, timeout=1)
    assert assembled["status"] == "succeeded"


async def test_unexpected_assembly_exception_releases_claim_for_one_later_retry(
    client, fake_comfy, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "first", "second")
    for child in children:
        fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    fake_comfy.pending = []
    assembled_output = {
        "node_id": "assembly",
        "filename": "retried.mp4",
        "subfolder": "directordeck/timelines",
        "type": "output",
    }
    assemble = AsyncMock(
        side_effect=[RuntimeError("unexpected assembler bug"), assembled_output]
    )
    monkeypatch.setattr("directordeck.app._assemble_timeline_output", assemble)

    with pytest.raises(RuntimeError, match="unexpected assembler bug"):
        await _reconcile_timeline(client, parent, allow_assembly=True)

    after_failure = client.director_app.state.database.get_job(parent["id"])
    assert after_failure is not None
    assert after_failure["status"] == "running"
    assert after_failure["stage"] == "assembly_retry"
    assert after_failure["outputs"] == []

    retried = await _reconcile_timeline(client, parent, allow_assembly=True)

    assert retried["status"] == "succeeded"
    assert retried["stage"] == "completed"
    assert retried["outputs"] == [assembled_output]
    assert assemble.await_count == 2


async def test_cancel_wins_while_multisegment_assembly_is_in_flight(
    client, fake_comfy, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "first", "second")
    for child in children:
        fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    fake_comfy.pending = []
    assembly_started = asyncio.Event()
    assembly_release = asyncio.Event()
    assembly_calls = 0

    async def blocked_assembly(_request, _job, _outputs):
        nonlocal assembly_calls
        assembly_calls += 1
        assembly_started.set()
        await assembly_release.wait()
        return {
            "node_id": "assembly",
            "filename": "orphaned-after-cancel.mp4",
            "subfolder": "directordeck/timelines",
            "type": "output",
        }

    monkeypatch.setattr("directordeck.app._assemble_timeline_output", blocked_assembly)
    database = client.director_app.state.database
    refresh = asyncio.create_task(
        _sync_timeline_job(
            _background_request(client.director_app),
            database.get_job(parent["id"]),
            allow_assembly=True,
        )
    )
    try:
        await asyncio.wait_for(assembly_started.wait(), timeout=1)
        claimed = client.director_app.state.database.get_job(parent["id"])
        assert claimed is not None
        assert (claimed["status"], claimed["stage"]) == ("running", "assembling")

        cancelled = await client.post(f"/api/jobs/{parent['id']}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
    finally:
        assembly_release.set()
    refreshed = await asyncio.wait_for(refresh, timeout=1)

    assert refreshed["status"] == "cancelled"
    assert refreshed["outputs"] == []
    assert assembly_calls == 1
    stored = client.director_app.state.database.get_job(parent["id"])
    assert stored is not None
    assert stored["status"] == "cancelled"
    assert stored["outputs"] == []


async def test_single_segment_success_cas_cannot_revive_cancelled_parent(
    client, fake_comfy, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "only")
    child = children[0]
    fake_comfy.histories[child["prompt_id"]] = _success_history(child)
    fake_comfy.pending = []
    database = client.director_app.state.database
    original = database.update_job_if_snapshot
    cancellation_won = False

    def cancel_immediately_before_success_cas(job_id: str, **kwargs):
        nonlocal cancellation_won
        if (
            not cancellation_won
            and job_id == parent["id"]
            and kwargs.get("status") == "succeeded"
            and kwargs.get("stage") == "completed"
        ):
            cancelled = database.update_job_if_status(
                job_id,
                kwargs["expected_status"],
                status="cancelled",
                progress=1.0,
                stage="cancelled",
                outputs=[],
                error=None,
                completed_at=utc_now(),
            )
            assert cancelled is not None
            cancellation_won = True
        return original(job_id, **kwargs)

    monkeypatch.setattr(
        database, "update_job_if_snapshot", cancel_immediately_before_success_cas
    )

    await _reconcile_timeline(client, parent)
    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200, response.text
    assert cancellation_won is True
    job = response.json()
    assert job["status"] == "cancelled"
    assert job["stage"] == "cancelled"
    assert job["outputs"] == []
    stored = database.get_job(parent["id"])
    assert stored is not None
    assert stored["status"] == "cancelled"
    assert stored["outputs"] == []


async def test_progress_sink_tolerates_job_deletion_between_lookup_and_write(
    client, monkeypatch
) -> None:
    parent, children = await _create_timeline_job(client, "only")
    child = children[0]
    sampler_id = next(
        str(node_id)
        for node_id, node in child["prompt_snapshot"].items()
        if node["class_type"] in {"SamplerCustomAdvanced", "XFuserSamplerCustomAdvanced"}
    )
    database = client.director_app.state.database
    original = database.update_job_child_progress_monotonic
    deletion_won = False

    def delete_immediately_before_progress_write(child_id: str, **kwargs):
        nonlocal deletion_won
        assert child_id == child["id"]
        terminal = database.update_job_if_status(
            parent["id"],
            "queued",
            status="cancelled",
            progress=1.0,
            stage="cancelled",
            outputs=[],
            error=None,
            completed_at=utc_now(),
        )
        assert terminal is not None
        assert database.delete_job_if_status(parent["id"], "cancelled") is True
        deletion_won = True
        # The child was removed by ON DELETE CASCADE. This raises KeyError;
        # the websocket sink must absorb that expected late-frame race.
        return original(child_id, **kwargs)

    monkeypatch.setattr(
        database,
        "update_job_child_progress_monotonic",
        delete_immediately_before_progress_write,
    )

    await client.director_app.state.progress_manager._sink(
        "http://comfy.test:8188",
        ComfyProgressEvent(
            prompt_id=child["prompt_id"],
            node_id=sampler_id,
            value=1.0,
            maximum=25.0,
        ),
    )

    assert deletion_won is True
    assert database.get_job(parent["id"]) is None
    assert database.get_job_child(child["id"]) is None


async def test_live_preview_accepts_only_registered_child_sampler_and_is_no_store(
    client,
) -> None:
    parent, children = await _create_timeline_job(client, "first", "second")
    child = children[1]
    sampler_ids = [
        str(node_id)
        for node_id, node in child["prompt_snapshot"].items()
        if node["class_type"] in {"SamplerCustomAdvanced", "XFuserSamplerCustomAdvanced"}
    ]
    assert len(sampler_ids) == 1
    cache = client.director_app.state.live_preview_cache
    preview_sink = client.director_app.state.progress_manager._preview_sink
    assert preview_sink is not None
    frame = b"\x89PNG\r\n\x1a\nlatest-frame"

    await preview_sink(
        "http://comfy.test:8188",
        ComfyPreviewEvent(
            prompt_id="forged-prompt",
            node_id=sampler_ids[0],
            mime_type="image/png",
            content=frame,
        ),
    )
    await preview_sink(
        "http://comfy.test:8188",
        ComfyPreviewEvent(
            prompt_id=child["prompt_id"],
            node_id="1",  # a loader, never an authenticated sampler preview
            mime_type="image/png",
            content=frame,
        ),
    )
    assert cache.get(parent["id"]) is None

    await preview_sink(
        "http://comfy.test:8188",
        ComfyPreviewEvent(
            prompt_id=child["prompt_id"],
            node_id=sampler_ids[0],
            mime_type="image/png",
            content=frame,
        ),
    )
    preview = cache.get(parent["id"])
    assert preview is not None
    assert preview.segment_id == "second"

    job = await client.get(f"/api/jobs/{parent['id']}")
    assert job.status_code == 200
    assert job.json()["live_preview_url"] == (
        f"/api/jobs/{parent['id']}/live-preview"
    )
    response = await client.get(job.json()["live_preview_url"])
    assert response.status_code == 200
    assert response.content == frame
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_live_preview_is_removed_on_job_delete_and_late_frame_cannot_restore(
    client,
) -> None:
    parent, children = await _create_timeline_job(client, "only")
    child = children[0]
    sampler_id = next(
        str(node_id)
        for node_id, node in child["prompt_snapshot"].items()
        if node["class_type"] in {"SamplerCustomAdvanced", "XFuserSamplerCustomAdvanced"}
    )
    event = ComfyPreviewEvent(
        prompt_id=child["prompt_id"],
        node_id=sampler_id,
        mime_type="image/jpeg",
        content=b"\xff\xd8\xffpreview",
    )
    preview_sink = client.director_app.state.progress_manager._preview_sink
    assert preview_sink is not None
    await preview_sink("http://comfy.test:8188", event)
    cache = client.director_app.state.live_preview_cache
    assert cache.get(parent["id"]) is not None

    database = client.director_app.state.database
    database.update_job_child(
        child["id"],
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        completed_at=utc_now(),
    )
    database.update_job(
        parent["id"],
        status="cancelled",
        progress=1.0,
        stage="cancelled",
        completed_at=utc_now(),
    )
    assert (await client.get(f"/api/jobs/{parent['id']}/live-preview")).status_code == 404

    deleted = await client.delete(f"/api/jobs/{parent['id']}")
    assert deleted.status_code == 200
    assert cache.get(parent["id"]) is None
    await preview_sink("http://comfy.test:8188", event)
    assert cache.get(parent["id"]) is None


async def test_job_read_hides_preview_from_terminal_child_while_parent_is_active(
    client,
) -> None:
    draft = _timeline("ref-child", "fl-child")
    draft["segments"][0].update(
        mode="r2v",
        reference_images=[asset("reference.png", "image", slot=0)],
        reference_audios=[],
        reference_videos=[],
    )
    created = await client.post("/api/timeline/jobs", json={"config": draft})
    assert created.status_code == 200, created.text
    parent = created.json()
    await wait_for_submission_tasks(client)
    parent = (await client.get(f"/api/jobs/{parent['id']}")).json()
    database = client.director_app.state.database
    children = database.list_job_children(parent["id"])
    assert len(children) == 2
    terminal_child, active_child = children
    sampler = next(
        str(node_id)
        for node_id, node in terminal_child["prompt_snapshot"].items()
        if node["class_type"]
        in {"SamplerCustomAdvanced", "XFuserSamplerCustomAdvanced"}
    )
    sink = client.director_app.state.progress_manager._preview_sink
    assert sink is not None
    await sink(
        "http://comfy.test:8188",
        ComfyPreviewEvent(
            prompt_id=terminal_child["prompt_id"],
            node_id=sampler,
            mime_type="image/png",
            content=b"\x89PNG\r\n\x1a\nterminal-child",
        ),
    )
    assert client.director_app.state.live_preview_cache.get(parent["id"]) is not None
    database.update_job_child(
        terminal_child["id"],
        status="succeeded",
        progress=1.0,
        stage="completed",
        outputs=[],
        completed_at=utc_now(),
    )

    response = await client.get(f"/api/jobs/{parent['id']}")

    assert response.status_code == 200, response.text
    assert response.json()["status"] in {"queued", "running"}
    assert response.json()["live_preview_url"] is None
    assert client.director_app.state.live_preview_cache.get(parent["id"]) is None

    # Stale eviction is not a deletion tombstone: the other live child may
    # immediately become the authoritative source of this parent's preview.
    active_sampler = next(
        str(node_id)
        for node_id, node in active_child["prompt_snapshot"].items()
        if node["class_type"]
        in {"SamplerCustomAdvanced", "XFuserSamplerCustomAdvanced"}
    )
    await sink(
        "http://comfy.test:8188",
        ComfyPreviewEvent(
            prompt_id=active_child["prompt_id"],
            node_id=active_sampler,
            mime_type="image/jpeg",
            content=b"\xff\xd8\xffactive-child",
        ),
    )
    assert client.director_app.state.live_preview_cache.get(parent["id"]) is not None


def _settings_for(client_id: str) -> RuntimeSettings:
    document = default_settings().model_dump(mode="json")
    document["client_id"] = client_id
    return RuntimeSettings.model_validate(document)


def _insert_lifecycle_job(database, job_id: str, status: str, settings: RuntimeSettings) -> None:
    now = utc_now()
    database.create_job(
        {
            "id": job_id,
            "mode": "timeline",
            "status": status,
            "progress": 0.0,
            "stage": status,
            "prompt_id": f"prompt-{job_id}",
            "outputs": [],
            "error": None,
            "config_snapshot": {"timeline": _timeline(job_id)},
            "settings_snapshot": settings.model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": now if status in {"succeeded", "failed", "cancelled"} else None,
        }
    )


async def test_lifespan_restores_progress_monitors_for_historical_active_endpoints(
    tmp_path, monkeypatch
) -> None:
    app = create_app(comfy_url="http://comfy.test:8188", database_path=tmp_path / "lifespan.sqlite3")
    database = app.state.database
    database.initialize()
    current = _settings_for("current-client")
    historical_a = _settings_for("client-a")
    historical_b = _settings_for("client-b")
    terminal = _settings_for("terminal-client")
    database.put_settings(current)
    _insert_lifecycle_job(database, "active-a", "queued", historical_a)
    _insert_lifecycle_job(database, "active-b", "running", historical_b)
    _insert_lifecycle_job(database, "finished", "succeeded", terminal)

    ensure = Mock()
    close = AsyncMock()
    monkeypatch.setattr(app.state.progress_manager, "ensure", ensure)
    monkeypatch.setattr(app.state.progress_manager, "close", close)

    async with app.router.lifespan_context(app):
        pass

    monitored = {
        (str(call.args[0]).rstrip("/"), call.args[1])
        for call in ensure.call_args_list
    }
    # The embedded app owns exactly one ComfyUI URL; historical jobs only add
    # their distinct client ids on that same host.
    assert (
        "http://comfy.test:8188",
        "current-client",
    ) in monitored
    assert (
        "http://comfy.test:8188",
        "client-a",
    ) in monitored
    assert (
        "http://comfy.test:8188",
        "client-b",
    ) in monitored
    assert all(
        client_id != "terminal-client" for _origin, client_id in monitored
    )
    close.assert_awaited_once_with()
