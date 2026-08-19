from __future__ import annotations

import asyncio
import sqlite3
import uuid
from copy import deepcopy
from pathlib import Path

import directordeck.app as director_app_module
from directordeck.compiler import timeline_segment_take_fingerprint
from directordeck.database import Database
from directordeck.schemas import UnifiedTimelineDraft, default_settings, default_timeline_draft

from .conftest import asset, wait_for_submission_tasks


def _segment(identity: str, *, prompt: str | None = None) -> dict:
    return {
        "id": identity,
        "title": identity,
        "mode": "fl2va",
        "duration_seconds": 1.0,
        "prompt": prompt or f"Prompt for {identity}",
        "enabled": True,
        "first_image": None,
        "last_image": None,
    }


def _timeline(
    *segments: dict,
    audio_mode: str = "generate",
    continuity: bool = False,
) -> UnifiedTimelineDraft:
    value = default_timeline_draft().model_dump(mode="json")
    value["segments"] = list(segments)
    for segment in value["segments"]:
        segment["ref_image_size"] = "match"
        segment["audio_mode"] = audio_mode
        segment["continuity"] = {
            "enabled": continuity,
            "overlap_frames": 5,
        }
    for sampling in value["sampling"].values():
        sampling["seed"] = 42
        sampling["random_seed"] = False
    return UnifiedTimelineDraft.model_validate(value)


def _take_fingerprint(
    timeline: UnifiedTimelineDraft, segment_id: str | None = None
) -> str:
    segment = next(
        item
        for item in timeline.segments
        if segment_id is None or item.id == segment_id
    )
    return timeline_segment_take_fingerprint(timeline, segment)


def _seed_successful_take(
    database: Database,
    timeline: UnifiedTimelineDraft,
    segment_id: str,
    *,
    filename: str | None = None,
    completed_at: str = "2026-08-13T12:00:00+00:00",
    duplicate_output: bool = False,
) -> tuple[str, str]:
    job_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    settings = default_settings()
    database.create_job(
        {
            "id": job_id,
            "mode": "timeline",
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {
                "timeline": timeline.model_dump(mode="json"),
                "segment_ids": [segment_id],
            },
            "settings_snapshot": settings.model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": completed_at,
            "updated_at": completed_at,
            "started_at": completed_at,
            "completed_at": completed_at,
        }
    )
    output = {
        "node_id": "save",
        "filename": filename or f"{segment_id}.mp4",
        "subfolder": "segments",
        "type": "output",
    }
    outputs = [output, dict(output, filename="duplicate.mp4")] if duplicate_output else [output]
    source_segment = next(
        segment for segment in timeline.segments if segment.id == segment_id
    )
    database.create_job_child(
        {
            "id": child_id,
            "job_id": job_id,
            "group_index": 1,
            "family": source_segment.mode,
            "backend": "standard",
            "segment_ids": [segment_id],
            "output_nodes": {segment_id: "save"},
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": child_id,
            "outputs": outputs,
            "error": None,
            "prompt_snapshot": {},
            "created_at": completed_at,
            "updated_at": completed_at,
            "started_at": completed_at,
            "completed_at": completed_at,
        }
    )
    return job_id, child_id


async def _wait_for_prompt_count(fake_comfy, count: int) -> None:
    async def ready() -> None:
        while len(fake_comfy.prompts) < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(ready(), timeout=2)


def _successful_child_history(child: dict, filename: str) -> dict:
    segment_id = child["segment_ids"][0]
    output_node_id = child["output_nodes"][segment_id]
    return {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            output_node_id: {
                "videos": [
                    {
                        "filename": filename,
                        "subfolder": "segments",
                        "type": "output",
                    }
                ]
            }
        },
    }


def test_take_fingerprint_tracks_output_geometry_only() -> None:
    timeline = _timeline(_segment("stable"))
    fingerprint = _take_fingerprint(timeline)

    cosmetic = timeline.model_copy(deep=True)
    cosmetic.segments[0].title = "renamed"
    cosmetic.segments[0].enabled = False
    cosmetic.segments[0].prompt = "changed while considering another prompt"
    assert _take_fingerprint(cosmetic) == fingerprint

    # Any raw duration that compiles to the same H3 visible frame count is the
    # same saved-video geometry. Crossing a frame lattice boundary is not.
    same_frames = timeline.model_copy(deep=True)
    same_frames.segments[0].duration_seconds = 1.1
    assert _take_fingerprint(same_frames) == fingerprint
    changed_frames = timeline.model_copy(deep=True)
    changed_frames.segments[0].duration_seconds = 2.0
    assert _take_fingerprint(changed_frames) != fingerprint

    changed_runtime = timeline.model_copy(deep=True)
    changed_runtime.sampling.fl2va.steps += 1
    changed_runtime.segments[0].continuity.enabled = True
    assert _take_fingerprint(changed_runtime) == fingerprint
    changed_runtime.render.width = 960
    assert _take_fingerprint(changed_runtime) != fingerprint


def test_take_fingerprint_ignores_recipe_prompt_and_media_conditioning() -> None:
    reference = {
        "id": "ref-content",
        "title": "ref-content",
        "mode": "ref2va",
        "duration_seconds": 1.0,
        "prompt": "Preserve the subject",
        "enabled": True,
        "source_video": asset("source.mp4", "video"),
        "source_start_seconds": 1.0,
        "source_duration_seconds": 1.0,
        "source_audio_as_reference": False,
        "reference_images": [
            asset("reference.png", "image", slot=0),
            asset("identity.png", "image", slot=1),
        ],
        "reference_audios": [asset("voice.wav", "audio", slot=0)],
        "reference_videos": [],
    }
    timeline = _timeline(reference, audio_mode="mute")
    fingerprint = _take_fingerprint(timeline)
    changed_document = timeline.model_dump(mode="json")
    changed_segment = changed_document["segments"][0]
    changed_segment["prompt"] = "a completely different reference prompt"
    changed_segment["reference_images"].reverse()
    changed_segment["source_start_seconds"] = 2.0
    changed_segment["source_duration_seconds"] = 2.0
    changed_segment["source_audio_as_reference"] = True
    changed_segment["reference_images"][0]["id"] = "another-reference-asset"
    changed = UnifiedTimelineDraft.model_validate(changed_document)
    assert _take_fingerprint(changed) == fingerprint

    # The generation recipe/family is not a property of the resulting video.
    fl = _timeline(_segment("ref-content"), audio_mode="mute")
    assert _take_fingerprint(fl) == fingerprint


def test_take_ledger_is_exact_audio_filtered_backfilled_and_not_job_cascaded(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "takes.sqlite3")
    database.initialize()
    timeline = _timeline(_segment("root"), audio_mode="generate")
    job_id, child_id = _seed_successful_take(database, timeline, "root")
    fingerprint = _take_fingerprint(timeline)

    take = database.find_latest_segment_take(
        "root",
        fingerprint,
        require_audio=True,
    )
    assert take is not None
    assert take["source_child_id"] == child_id
    assert take["output"] == {
        "filename": "root.mp4",
        "subfolder": "segments",
        "type": "output",
    }

    _seed_successful_take(
        database,
        timeline,
        "root",
        filename="newer-root.mp4",
        completed_at="2026-08-13T13:00:00+00:00",
    )
    newest = database.find_latest_segment_take(
        "root", fingerprint,
    )
    assert newest is not None
    assert newest["output"]["filename"] == "newer-root.mp4"

    _seed_successful_take(
        database,
        timeline,
        "root",
        filename="same-time-root.mp4",
        completed_at="2026-08-13T13:00:00+00:00",
    )
    with sqlite3.connect(database.path) as db:
        tied = db.execute(
            "SELECT id, output_descriptor FROM segment_takes "
            "WHERE segment_id = 'root' AND completed_at = ? ORDER BY id DESC",
            ("2026-08-13T13:00:00+00:00",),
        ).fetchall()
    newest = database.find_latest_segment_take(
        "root", fingerprint,
    )
    assert newest is not None
    assert newest["id"] == tied[0][0]

    with sqlite3.connect(database.path) as db:
        assert db.execute("PRAGMA foreign_key_list(segment_takes)").fetchall() == []
        db.execute("DELETE FROM segment_takes")
    assert database.find_latest_segment_take(
        "root", fingerprint,
    ) is None
    database.initialize()
    assert database.find_latest_segment_take(
        "root", fingerprint,
    ) is not None

    # Upgrade an already-registered authored-content row in place; the unique
    # source_child_id must not make the migration skip it.
    with sqlite3.connect(database.path) as db:
        db.execute(
            "UPDATE segment_takes SET content_fingerprint = 'sha256:legacy' "
            "WHERE source_child_id = ?",
            (child_id,),
        )
    database.initialize()
    assert database.find_latest_segment_take(
        "root", fingerprint,
    ) is not None

    assert database.delete_job_if_status(job_id, "succeeded") is True
    preserved = database.find_latest_segment_take(
        "root", fingerprint,
    )
    assert preserved is not None
    with sqlite3.connect(database.path) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM segment_takes WHERE source_child_id = ?",
            (child_id,),
        ).fetchone()[0] == 1


def test_mute_take_is_visual_only_and_ambiguous_success_is_not_registered(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "audio-takes.sqlite3")
    database.initialize()
    mute_timeline = _timeline(_segment("mute-root"), audio_mode="mute")
    _seed_successful_take(database, mute_timeline, "mute-root")
    fingerprint = _take_fingerprint(mute_timeline)
    assert database.find_latest_segment_take(
        "mute-root", fingerprint,
    ) is not None
    assert database.find_latest_segment_take(
        "mute-root",
        fingerprint,
        require_audio=True,
    ) is None

    bad_timeline = _timeline(_segment("ambiguous"))
    _seed_successful_take(
        database, bad_timeline, "ambiguous", duplicate_output=True
    )
    assert database.has_segment_take(
        "ambiguous",
    ) is False


async def test_compile_and_submit_reuse_server_resolved_historical_take_without_wait(
    client, fake_comfy
) -> None:
    database = client.director_app.state.database
    root = _segment("history-root")
    successor = _segment("history-successor")
    authored = _timeline(root, successor, continuity=True)
    _, source_child_id = _seed_successful_take(
        database,
        _timeline(root, audio_mode="generate"),
        "history-root",
        filename="certified-root.mp4",
    )
    # Runtime/model/LoRA/sampler/seed changes are not authored predecessor
    # content and must not invalidate its persisted take.
    live_settings = database.get_settings()
    live_settings.models.fl2va.filename = "generic_h3_diffusion.safetensors"
    live_settings.models.fl2va.lora_name = (
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    )
    live_settings.models.fl2va.lora_strength = 0.75
    database.put_settings(live_settings)
    authored.sampling.fl2va.steps = 31
    authored.sampling.fl2va.seed = 9_876_543

    preview = await client.post(
        "/api/timeline/compile",
        json={
            "config": authored.model_dump(mode="json"),
            "segment_ids": ["history-successor"],
        },
    )
    assert preview.status_code == 200, preview.text
    plan = preview.json()["plans"][0]
    assert plan["continuity_source"] == "historical_take"
    assert plan["predecessor_segment_id"] == "history-root"
    assert isinstance(plan["historical_take_id"], str)
    assert fake_comfy.prompts == []

    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": authored.model_dump(mode="json"),
            "segment_ids": ["history-successor"],
        },
    )
    assert created.status_code == 200, created.text
    await wait_for_submission_tasks(client)
    assert len(fake_comfy.prompts) == 1
    prompt = fake_comfy.prompts[0]["prompt"]
    assert any(
        node.get("class_type") == "LoadVideo"
        and node.get("inputs", {}).get("file")
        == "segments/certified-root.mp4 [output]"
        for node in prompt.values()
    )
    # No history result exists for the historical predecessor and none is
    # polled: only the newly submitted successor remains pending.
    assert all(request[0] != source_child_id for request in fake_comfy.history_requests)

    rejected = await client.post(
        "/api/timeline/jobs",
        json={
            "config": authored.model_dump(mode="json"),
            "segment_ids": ["history-successor"],
            "historical_take_id": plan["historical_take_id"],
        },
    )
    assert rejected.status_code == 422


async def test_historical_take_survives_recipe_family_prompt_and_reference_changes(
    client,
) -> None:
    database = client.director_app.state.database
    stable_id = "recipe-independent-root"
    original = _segment(stable_id, prompt="original T2V prompt")
    _seed_successful_take(
        database,
        _timeline(original, audio_mode="generate"),
        stable_id,
        filename="latest-structurally-compatible.mp4",
    )

    reference = asset("new-reference.png", "image", slot=0)
    database.put_asset(
        reference["id"],
        {key: value for key, value in reference.items() if key != "slot"},
    )
    edited_predecessor = {
        "id": stable_id,
        "title": "now edited as Ref2V",
        "mode": "ref2va",
        "duration_seconds": 1.1,
        "prompt": "A different recipe using <Picture 1>",
        "enabled": True,
        "source_video": None,
        "source_start_seconds": 0.0,
        "source_duration_seconds": 1.1,
        "source_audio_as_reference": False,
        "reference_images": [reference],
        "reference_audios": [],
        "reference_videos": [],
    }
    successor = _segment("mixed-family-successor", prompt="continue the shot")
    mixed = _timeline(
        edited_predecessor,
        successor,
        audio_mode="generate",
        continuity=True,
    )

    preview = await client.post(
        "/api/timeline/compile",
        json={
            "config": mixed.model_dump(mode="json"),
            "segment_ids": [successor["id"]],
        },
    )
    assert preview.status_code == 200, preview.text
    plan = preview.json()["plans"][0]
    assert plan["continuity_source"] == "historical_take"
    assert plan["predecessor_segment_id"] == stable_id
    assert plan["model_family"] == "fl2va"


async def test_historical_take_reports_missing_geometry_and_audio_incapable(client) -> None:
    database = client.director_app.state.database
    root = _segment("state-root")
    successor = _segment("state-successor")
    mute_source = _timeline(root, audio_mode="mute")
    _seed_successful_take(database, mute_source, "state-root")

    generate = _timeline(root, successor, audio_mode="generate", continuity=True)
    audio_error = await client.post(
        "/api/timeline/compile",
        json={
            "config": generate.model_dump(mode="json"),
            "segment_ids": ["state-successor"],
        },
    )
    assert audio_error.status_code == 422
    assert "不含生成音频接续所需的音轨" in audio_error.text

    muted = _timeline(root, successor, audio_mode="mute", continuity=True)
    assert (
        await client.post(
            "/api/timeline/compile",
            json={
                "config": muted.model_dump(mode="json"),
                "segment_ids": ["state-successor"],
            },
        )
    ).status_code == 200

    edited_document = muted.model_dump(mode="json")
    edited_document["segments"][0]["prompt"] = "predecessor prompt changed"
    edited = await client.post(
        "/api/timeline/compile",
        json={"config": edited_document, "segment_ids": ["state-successor"]},
    )
    assert edited.status_code == 200, edited.text

    mismatched_document = deepcopy(edited_document)
    mismatched_document["render"]["width"] = 960
    mismatch = await client.post(
        "/api/timeline/compile",
        json={"config": mismatched_document, "segment_ids": ["state-successor"]},
    )
    assert mismatch.status_code == 422
    assert "分辨率、帧率或可见帧数" in mismatch.text

    missing_document = deepcopy(edited_document)
    missing_document["segments"][0]["id"] = "never-rendered-root"
    missing = await client.post(
        "/api/timeline/compile",
        json={"config": missing_document, "segment_ids": ["state-successor"]},
    )
    assert missing.status_code == 422
    assert "没有可用的历史成功成片" in missing.text


async def test_historical_then_same_run_chain_waits_only_for_new_middle_take(
    client, fake_comfy, monkeypatch
) -> None:
    monkeypatch.setattr(
        director_app_module, "_RAYLIGHT_GENERATION_POLL_SECONDS", 0.001
    )
    database = client.director_app.state.database
    first = _segment("chain-a")
    middle = _segment("chain-b")
    last = _segment("chain-c")
    _, historical_child_id = _seed_successful_take(
        database,
        _timeline(first, audio_mode="generate"),
        "chain-a",
        filename="chain-a-history.mp4",
    )
    authored = _timeline(first, middle, last, continuity=True)

    created = await client.post(
        "/api/timeline/jobs",
        json={
            "config": authored.model_dump(mode="json"),
            "segment_ids": ["chain-b", "chain-c"],
        },
    )
    assert created.status_code == 200, created.text
    await _wait_for_prompt_count(fake_comfy, 1)
    first_prompt = fake_comfy.prompts[0]["prompt"]
    assert any(
        node.get("class_type") == "LoadVideo"
        and node.get("inputs", {}).get("file")
        == "segments/chain-a-history.mp4 [output]"
        for node in first_prompt.values()
    )

    children = database.list_job_children(created.json()["id"])
    middle_child = next(
        child for child in children if child["segment_ids"] == ["chain-b"]
    )
    middle_prompt_id = str(middle_child["prompt_id"])
    fake_comfy.pending = [
        item for item in fake_comfy.pending if middle_prompt_id not in item
    ]
    fake_comfy.histories[middle_prompt_id] = _successful_child_history(
        middle_child, "chain-b-current.mp4"
    )

    await _wait_for_prompt_count(fake_comfy, 2)
    await wait_for_submission_tasks(client)
    successor_prompt = fake_comfy.prompts[1]["prompt"]
    assert any(
        node.get("class_type") == "LoadVideo"
        and node.get("inputs", {}).get("file")
        == "segments/chain-b-current.mp4 [output]"
        for node in successor_prompt.values()
    )
    assert all(
        requested_prompt_id != historical_child_id
        for requested_prompt_id, _ in fake_comfy.history_requests
    )

    middle_segment = next(
        segment for segment in authored.segments if segment.id == "chain-b"
    )
    middle_take = database.find_latest_segment_take(
        "chain-b",
        timeline_segment_take_fingerprint(authored, middle_segment),
        require_audio=True,
    )
    assert middle_take is not None

    # A later edit to A cannot transitively invalidate the already-rendered B
    # artifact. C's current direct predecessor is B, so target-only reuse is
    # keyed only by B's stable id and saved-video geometry.
    changed_ancestor = authored.model_copy(deep=True)
    changed_ancestor.segments[0].prompt = "A changed after B was rendered"
    preview = await client.post(
        "/api/timeline/compile",
        json={
            "config": changed_ancestor.model_dump(mode="json"),
            "segment_ids": ["chain-c"],
        },
    )
    assert preview.status_code == 200, preview.text
    plan = preview.json()["plans"][0]
    assert plan["predecessor_segment_id"] == "chain-b"
    assert plan["continuity_source"] == "historical_take"
    assert plan["historical_take_id"] == middle_take["id"]


async def test_historical_resolution_uses_current_direct_predecessor_after_reorder(
    client,
) -> None:
    database = client.director_app.state.database
    first = _segment("order-a")
    other = _segment("order-x")
    target = _segment("order-target")
    _seed_successful_take(
        database, _timeline(first), "order-a", filename="order-a.mp4"
    )
    _seed_successful_take(
        database, _timeline(other), "order-x", filename="order-x.mp4"
    )
    first_take = database.find_latest_segment_take(
        "order-a",
        _take_fingerprint(_timeline(first)),
        require_audio=True,
    )
    other_take = database.find_latest_segment_take(
        "order-x",
        _take_fingerprint(_timeline(other)),
        require_audio=True,
    )
    assert first_take is not None and other_take is not None

    before = _timeline(first, other, target, continuity=True)
    before_preview = await client.post(
        "/api/timeline/compile",
        json={
            "config": before.model_dump(mode="json"),
            "segment_ids": ["order-target"],
        },
    )
    assert before_preview.status_code == 200, before_preview.text
    before_plan = before_preview.json()["plans"][0]
    assert before_plan["predecessor_segment_id"] == "order-x"
    assert before_plan["historical_take_id"] == other_take["id"]

    reordered = _timeline(other, first, target, continuity=True)
    after_preview = await client.post(
        "/api/timeline/compile",
        json={
            "config": reordered.model_dump(mode="json"),
            "segment_ids": ["order-target"],
        },
    )
    assert after_preview.status_code == 200, after_preview.text
    after_plan = after_preview.json()["plans"][0]
    assert after_plan["predecessor_segment_id"] == "order-a"
    assert after_plan["historical_take_id"] == first_take["id"]
