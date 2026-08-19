from __future__ import annotations

import threading
from typing import Any

import pytest

import directordeck.database as database_module
from directordeck.compiler import DraftNotRunnable, validate_unified_runnable
from directordeck.schemas import UnifiedTimelineDraft, default_timeline_draft

from .conftest import asset, runnable_draft


def timeline(*segments: dict[str, Any]) -> dict[str, Any]:
    document = default_timeline_draft().model_dump(mode="json")
    document["segments"] = list(segments)
    if any(
        item.get("mode") in {"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"}
        for item in segments
    ):
        document["version"] = 1
    else:
        for item in document["segments"]:
            item.setdefault("ref_image_size", "match")
            item.setdefault("audio_mode", "generate")
    return document


def segment(mode: str, identity: str, **values: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "id": identity,
        "title": f"title-{identity}",
        "mode": mode,
        "prompt": f"prompt-{identity}",
        "duration_seconds": 5.0,
        "enabled": True,
    }
    document.update(values)
    return document


def by_id(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in document["segments"]}


def assert_empty_fl2va_shape(value: dict[str, Any], identity: str) -> None:
    assert value == {
        "id": identity,
        "title": f"title-{identity}",
        "prompt": f"prompt-{identity}",
        "duration_seconds": 5.0,
        "enabled": True,
        "mode": "fl2va",
        "continuity": {"enabled": False, "overlap_frames": 22},
        "ref_image_size": "match",
        "audio_mode": "generate",
        "first_image": None,
        "last_image": None,
    }


def assert_empty_ref2va_shape(value: dict[str, Any], identity: str) -> None:
    assert value == {
        "id": identity,
        "title": f"title-{identity}",
        "prompt": f"prompt-{identity}",
        "duration_seconds": 5.0,
        "enabled": True,
        "mode": "ref2va",
        "continuity": {"enabled": False, "overlap_frames": 22},
        "ref_image_size": "match",
        "audio_mode": "generate",
        "source_video": None,
        "source_start_seconds": 0.0,
        "source_duration_seconds": 5.0,
        "source_audio_as_reference": False,
        "reference_images": [],
        "reference_audios": [],
        "reference_videos": [],
    }


async def test_cascade_delete_preserves_family_while_recipe_follows_remaining_media(
    client,
) -> None:
    first = asset("first.png", "image")
    last = asset("last.png", "image")
    document = timeline(
        segment("i2v", "i2v-first", first_image=first),
        segment(
            "fl2v",
            "fl2v-partial",
            first_image=first,
            last_image=last,
        ),
        segment(
            "fl2v",
            "fl2v-empty",
            first_image=first,
            last_image=None,
        ),
        segment(
            "r2v",
            "r2v-last-ref",
            reference_images=[asset("first.png", "image", slot=0)],
            reference_audios=[],
            reference_videos=[],
        ),
    )
    saved = await client.put("/api/timeline", json=document)
    assert saved.status_code == 200, saved.text

    deleted = await client.delete(
        f"/api/assets/{first['id']}", params={"cascade": "true"}
    )

    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["outputs_preserved"] is True
    assert set(deleted.json()["unbound_usages"]) == {
        "timeline.segments[0](i2v-first).first_image",
        "timeline.segments[1](fl2v-partial).first_image",
        "timeline.segments[2](fl2v-empty).first_image",
        "timeline.segments[3](r2v-last-ref).reference_images[0]",
    }
    assert client.director_app.state.database.get_asset(first["id"]) is None

    current = by_id((await client.get("/api/timeline")).json())
    assert_empty_fl2va_shape(current["i2v-first"], "i2v-first")
    assert current["fl2v-partial"]["mode"] == "fl2va"
    assert current["fl2v-partial"]["first_image"] is None
    assert current["fl2v-partial"]["last_image"]["id"] == last["id"]
    assert_empty_fl2va_shape(current["fl2v-empty"], "fl2v-empty")
    assert_empty_ref2va_shape(current["r2v-last-ref"], "r2v-last-ref")

    deleted_last = await client.delete(
        f"/api/assets/{last['id']}", params={"cascade": "true"}
    )

    assert deleted_last.status_code == 200, deleted_last.text
    assert deleted_last.json()["unbound_usages"] == [
        "timeline.segments[1](fl2v-partial).last_image"
    ]
    current = by_id((await client.get("/api/timeline")).json())
    assert_empty_fl2va_shape(current["fl2v-partial"], "fl2v-partial")


async def test_cascade_delete_rewrites_fl_picture_ordinals_without_touching_whitespace(
    client,
) -> None:
    first = asset("first.png", "image")
    last = asset("last.png", "image")
    prompt = "\tfirst <Picture 1>  \n\n  last <Picture 2>\t "
    document = timeline(
        segment(
            "fl2v",
            "fl-picture-labels",
            prompt=prompt,
            first_image=first,
            last_image=last,
        )
    )
    saved = await client.put("/api/timeline", json=document)
    assert saved.status_code == 200, saved.text

    deleted_first = await client.delete(
        f"/api/assets/{first['id']}", params={"cascade": "true"}
    )
    assert deleted_first.status_code == 200, deleted_first.text
    current = by_id((await client.get("/api/timeline")).json())[
        "fl-picture-labels"
    ]
    assert current["first_image"] is None
    assert current["last_image"]["id"] == last["id"]
    assert current["prompt"] == "\tfirst   \n\n  last <Picture 1>\t "

    deleted_last = await client.delete(
        f"/api/assets/{last['id']}", params={"cascade": "true"}
    )
    assert deleted_last.status_code == 200, deleted_last.text
    current = by_id((await client.get("/api/timeline")).json())[
        "fl-picture-labels"
    ]
    assert current["last_image"] is None
    assert current["prompt"] == "\tfirst   \n\n  last \t "


async def test_cascade_delete_source_preserves_ref2va_and_rewrites_reference_labels(
    client,
) -> None:
    source = asset("source.mp4", "video")
    document = timeline(
        segment(
            "v2v",
            "v2v-source",
            prompt="Edit <Video 1> with <Audio 1>",
            source_video=source,
            source_start_seconds=1.0,
            source_duration_seconds=5.0,
            source_audio_as_reference=True,
        ),
        segment(
            "rv2v",
            "rv2v-with-refs",
            prompt="Edit <Video 1>; source <Audio 1>; voice <Audio 2>",
            source_video=source,
            source_start_seconds=1.0,
            source_duration_seconds=5.0,
            source_audio_as_reference=True,
            reference_images=[asset("identity.png", "image", slot=0)],
            reference_audios=[asset("voice.wav", "audio", slot=0)],
        ),
        segment(
            "rv2v",
            "rv2v-without-refs",
            source_video=source,
            source_start_seconds=1.0,
            source_duration_seconds=5.0,
            reference_images=[],
            reference_audios=[],
        ),
    )
    saved = await client.put("/api/timeline", json=document)
    assert saved.status_code == 200, saved.text

    deleted = await client.delete(
        f"/api/assets/{source['id']}", params={"cascade": "true"}
    )

    assert deleted.status_code == 200, deleted.text
    assert set(deleted.json()["unbound_usages"]) == {
        "timeline.segments[0](v2v-source).source_video",
        "timeline.segments[1](rv2v-with-refs).source_video",
        "timeline.segments[2](rv2v-without-refs).source_video",
    }
    current = by_id((await client.get("/api/timeline")).json())
    assert current["v2v-source"]["mode"] == "ref2va"
    assert current["v2v-source"]["prompt"] == "Edit  with "
    assert current["v2v-source"]["source_video"] is None
    assert current["v2v-source"]["source_start_seconds"] == 0.0
    assert current["v2v-source"]["source_duration_seconds"] == 5.0
    assert current["v2v-source"]["source_audio_as_reference"] is False
    converted = current["rv2v-with-refs"]
    assert converted["mode"] == "ref2va"
    assert converted["reference_images"][0]["id"] == "fixture-image-identity.png"
    assert converted["reference_images"][0]["slot"] == 0
    assert converted["reference_audios"][0]["id"] == "fixture-audio-voice.wav"
    assert converted["reference_audios"][0]["slot"] == 0
    assert converted["reference_videos"] == []
    assert converted["prompt"] == "Edit ; source ; voice <Audio 1>"
    assert converted["source_video"] is None
    assert converted["source_start_seconds"] == 0.0
    assert converted["source_duration_seconds"] == 5.0
    assert_empty_ref2va_shape(
        current["rv2v-without-refs"], "rv2v-without-refs"
    )


async def test_v2_source_cascade_shifts_independent_video_and_audio_labels(
    client,
) -> None:
    source = asset("source.mp4", "video")
    document = timeline(
        segment(
            "ref2va",
            "source-with-independent-media",
            prompt=(
                "source <Video 1>; motion <Video 2>; source sound <Audio 1>; "
                "voice <Audio 2>"
            ),
            source_video=source,
            source_start_seconds=1.0,
            source_duration_seconds=5.0,
            source_audio_as_reference=True,
            reference_images=[],
            reference_videos=[asset("motion.mp4", "video", slot=0)],
            reference_audios=[asset("voice.wav", "audio", slot=0)],
        )
    )
    saved = await client.put("/api/timeline", json=document)
    assert saved.status_code == 200, saved.text

    deleted = await client.delete(
        f"/api/assets/{source['id']}", params={"cascade": "true"}
    )

    assert deleted.status_code == 200, deleted.text
    current = (await client.get("/api/timeline")).json()
    current_segment = current["segments"][0]
    assert current_segment["mode"] == "ref2va"
    assert current_segment["source_video"] is None
    assert current_segment["source_audio_as_reference"] is False
    assert current_segment["reference_videos"][0]["slot"] == 0
    assert current_segment["reference_audios"][0]["slot"] == 0
    assert current_segment["prompt"] == (
        "source ; motion <Video 1>; source sound ; voice <Audio 1>"
    )
    validate_unified_runnable(UnifiedTimelineDraft.model_validate(current))


async def test_cascade_compacts_reference_slots_and_rewrites_effective_prompt(
    client,
) -> None:
    top_level_prompt = "old=<Picture 1>; keep=<Picture 2>"
    document = timeline(
        segment(
            "r2v",
            "dense-after-delete",
            prompt="",
            reference_images=[
                asset("reference.png", "image", slot=0),
                asset("identity.png", "image", slot=1),
            ],
            reference_audios=[],
            reference_videos=[],
        )
    )
    document["prompt"] = top_level_prompt
    saved = await client.put("/api/timeline", json=document)
    assert saved.status_code == 200, saved.text

    removed = asset("reference.png", "image")
    deleted = await client.delete(
        f"/api/assets/{removed['id']}", params={"cascade": "true"}
    )

    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["unbound_usages"] == [
        "timeline.segments[0](dense-after-delete).reference_images[0]"
    ]
    current = (await client.get("/api/timeline")).json()
    current_segment = current["segments"][0]
    assert "prompt" not in current
    assert current_segment["mode"] == "ref2va"
    assert [
        (reference["id"], reference["slot"])
        for reference in current_segment["reference_images"]
    ] == [("fixture-image-identity.png", 0)]
    # The segment originally inherited the timeline prompt.  Cascade must
    # materialize that effective prompt before its per-segment tag rewrite:
    # the removed slot's tag disappears and the retained old slot 1 becomes 0
    # (<Picture 2> -> <Picture 1> in the one-based prompt protocol).
    assert current_segment["prompt"] == "old=; keep=<Picture 1>"
    assert current_segment["prompt"].count("<Picture 1>") == 1
    assert "<Picture 2>" not in current_segment["prompt"]

    validated = UnifiedTimelineDraft.model_validate(current)
    execution = validate_unified_runnable(
        validated, segment_ids=["dense-after-delete"]
    )
    assert [item.id for item in execution] == ["dense-after-delete"]
    compiled = await client.post(
        "/api/timeline/compile",
        json={"segment_ids": ["dense-after-delete"]},
    )
    assert compiled.status_code == 200, compiled.text


async def test_cascade_reference_rewrite_preserves_multiline_prompt_formatting(
    client,
) -> None:
    removed = asset("reference.png", "image", slot=0)
    retained = asset("identity.png", "image", slot=1)
    prompt = (
        "\n  subject_definitions:\n"
        "<Picture 1> 主体  \n\n"
        "visual_style:\n"
        "  保留 <Picture 2>\n\n"
        "overall_soundscape:\n"
        "N/A.\n"
    )
    document = timeline(segment(
        "r2v",
        "formatted-prompt",
        prompt=prompt,
        reference_images=[removed, retained],
        reference_audios=[],
        reference_videos=[],
    ))
    saved = await client.put("/api/timeline", json=document)
    assert saved.status_code == 200, saved.text

    deleted = await client.delete(
        f"/api/assets/{removed['id']}", params={"cascade": "true"}
    )

    assert deleted.status_code == 200, deleted.text
    current_prompt = (await client.get("/api/timeline")).json()["segments"][0]["prompt"]
    assert current_prompt == (
        "\n  subject_definitions:\n"
        " 主体  \n\n"
        "visual_style:\n"
        "  保留 <Picture 1>\n\n"
        "overall_soundscape:\n"
        "N/A.\n"
    )


async def test_cascade_normalizes_source_audio_when_enabled_source_is_removed(
    client,
) -> None:
    source = asset("source.mp4", "video")
    document = timeline(
        segment(
            "v2v",
            "source-audio-segment",
            source_video=source,
            source_start_seconds=1.0,
            source_duration_seconds=5.0,
        )
    )
    document["audio_mode"] = "source"
    saved = await client.put("/api/timeline", json=document)
    assert saved.status_code == 200, saved.text

    deleted = await client.delete(
        f"/api/assets/{source['id']}", params={"cascade": "true"}
    )

    assert deleted.status_code == 200, deleted.text
    current = (await client.get("/api/timeline")).json()
    assert "audio_mode" not in current
    assert current["segments"][0]["audio_mode"] == "generate"
    assert_empty_ref2va_shape(current["segments"][0], "source-audio-segment")
    with pytest.raises(DraftNotRunnable, match="Ref2VA segments need"):
        validate_unified_runnable(UnifiedTimelineDraft.model_validate(current))
    compiled = await client.post("/api/timeline/compile", json={})
    assert compiled.status_code == 422, compiled.text


async def test_cascade_unbinds_every_typed_reference_across_six_legacy_drafts(
    client,
) -> None:
    drafts = {mode: runnable_draft(mode) for mode in (
        "t2v",
        "i2v",
        "fl2v",
        "r2v",
        "v2v",
        "rv2v",
    )}
    drafts["t2v"]["prompt"] = "t2v-must-remain-identical"
    drafts["r2v"]["shots"][0]["reference_images"] = [
        asset("first.png", "image", slot=0)
    ]
    drafts["rv2v"]["shots"][0]["reference_images"] = [
        asset("first.png", "image", slot=0)
    ]
    for mode, draft in drafts.items():
        saved = await client.put(f"/api/drafts/{mode}", json=draft)
        assert saved.status_code == 200, saved.text

    first = asset("first.png", "image")
    deleted_image = await client.delete(
        f"/api/assets/{first['id']}", params={"cascade": "true"}
    )
    assert deleted_image.status_code == 200, deleted_image.text
    assert set(deleted_image.json()["unbound_usages"]) == {
        "drafts.fl2v.shots[0](fl2v-1).first_image",
        "drafts.i2v.shots[0](i2v-1).first_image",
        "drafts.r2v.shots[0](r2v-1).reference_images[0]",
        "drafts.rv2v.shots[0](rv2v-1).reference_images[0]",
    }

    for filename, kind, expected_usages in (
        (
            "voice.wav",
            "audio",
            {
                "drafts.r2v.shots[0](r2v-1).reference_audios[0]",
                "drafts.rv2v.shots[0](rv2v-1).reference_audios[0]",
            },
        ),
        (
            "motion.mp4",
            "video",
            {"drafts.r2v.shots[0](r2v-1).reference_videos[0]"},
        ),
        (
            "source.mp4",
            "video",
            {
                "drafts.rv2v.shots[0](rv2v-1).source_video",
                "drafts.v2v.shots[0](v2v-1).source_video",
            },
        ),
        (
            "last.png",
            "image",
            {"drafts.fl2v.shots[0](fl2v-1).last_image"},
        ),
    ):
        reference = asset(filename, kind)
        response = await client.delete(
            f"/api/assets/{reference['id']}", params={"cascade": "true"}
        )
        assert response.status_code == 200, response.text
        assert set(response.json()["unbound_usages"]) == expected_usages

    current = {
        mode: (await client.get(f"/api/drafts/{mode}")).json()
        for mode in drafts
    }
    expected_t2v = dict(drafts["t2v"])
    expected_t2v["sampling"] = dict(expected_t2v["sampling"])
    expected_t2v["sampling"].pop("cfg")
    expected_t2v["sampling"]["random_seed"] = False
    assert current["t2v"] == expected_t2v
    assert current["i2v"]["mode"] == "i2v"
    assert current["i2v"]["shots"][0]["first_image"] is None
    assert current["fl2v"]["mode"] == "fl2v"
    assert current["fl2v"]["shots"][0]["first_image"] is None
    assert current["fl2v"]["shots"][0]["last_image"] is None
    assert current["r2v"]["mode"] == "r2v"
    assert current["r2v"]["shots"][0]["reference_images"] == []
    assert current["r2v"]["shots"][0]["reference_audios"] == []
    assert current["r2v"]["shots"][0]["reference_videos"] == []
    assert current["v2v"]["mode"] == "v2v"
    assert current["v2v"]["shots"][0]["source_video"] is None
    assert current["rv2v"]["mode"] == "rv2v"
    assert current["rv2v"]["shots"][0]["source_video"] is None
    assert current["rv2v"]["shots"][0]["reference_images"] == []
    assert current["rv2v"]["shots"][0]["reference_audios"] == []


async def test_delete_without_cascade_still_returns_409_and_changes_nothing(
    client,
) -> None:
    first = asset("first.png", "image")
    timeline_before = timeline(
        segment("i2v", "timeline-i2v", first_image=first)
    )
    draft_before = runnable_draft("i2v")
    saved_timeline = await client.put("/api/timeline", json=timeline_before)
    saved_draft = await client.put("/api/drafts/i2v", json=draft_before)
    assert saved_timeline.status_code == 200
    assert saved_draft.status_code == 200

    refused = await client.delete(f"/api/assets/{first['id']}")

    assert refused.status_code == 409
    assert refused.json()["detail"]["message"] == (
        "asset is still referenced by saved drafts"
    )
    assert refused.json()["detail"]["outputs_preserved"] is True
    assert set(refused.json()["detail"]["usages"]) == {
        "timeline.segments[0](timeline-i2v).first_image",
        "drafts.i2v.shots[0](i2v-1).first_image",
    }
    assert client.director_app.state.database.get_asset(first["id"]) is not None
    assert (await client.get("/api/timeline")).json() == saved_timeline.json()
    assert (await client.get("/api/drafts/i2v")).json() == saved_draft.json()


@pytest.mark.parametrize("document_kind", ["timeline", "draft"])
def test_atomic_validate_and_put_cannot_resurrect_asset_after_cascade_delete(
    client, monkeypatch, document_kind
) -> None:
    database = client.director_app.state.database
    first = asset("first.png", "image")
    entered_validation = threading.Event()
    release_validation = threading.Event()
    original = database._validate_asset_iterator_in_connection

    def block_after_asset_lookup(connection, references):
        materialized = list(references)
        original(connection, materialized)
        entered_validation.set()
        assert release_validation.wait(timeout=2)

    monkeypatch.setattr(
        database, "_validate_asset_iterator_in_connection", block_after_asset_lookup
    )
    error: list[BaseException] = []

    def save() -> None:
        try:
            if document_kind == "timeline":
                value = UnifiedTimelineDraft.model_validate(
                    timeline(segment("i2v", "atomic", first_image=first))
                )
                database.validate_and_put_timeline(value)
            else:
                value = database_module.validate_mode_draft(
                    "i2v", runnable_draft("i2v")
                )
                database.validate_and_put_draft("i2v", value)
        except BaseException as exc:
            error.append(exc)

    saver = threading.Thread(target=save)
    saver.start()
    assert entered_validation.wait(timeout=2)

    deletion_result: list[list[str]] = []

    def delete() -> None:
        deletion_result.append(
            database.delete_asset_if_unused(first["id"], cascade=True)
        )

    deleter = threading.Thread(target=delete)
    deleter.start()
    # The delete is serialized behind the save's BEGIN IMMEDIATE transaction.
    deleter.join(timeout=0.05)
    assert deleter.is_alive()
    release_validation.set()
    saver.join(timeout=2)
    deleter.join(timeout=2)

    assert error == []
    assert not saver.is_alive() and not deleter.is_alive()
    assert deletion_result and deletion_result[0]
    assert database.get_asset(first["id"]) is None
    if document_kind == "timeline":
        saved = database.get_timeline().model_dump(mode="json")
        assert saved["segments"][0]["mode"] == "fl2va"
        assert saved["segments"][0]["first_image"] is None
        assert saved["segments"][0]["last_image"] is None
    else:
        saved = database.get_draft("i2v").model_dump(mode="json")
        assert saved["shots"][0]["first_image"] is None


def persisted_rows(database) -> tuple[Any, Any, Any]:
    with database.connect() as connection:
        timeline_row = connection.execute(
            "SELECT document, updated_at, revision FROM unified_timeline "
            "WHERE singleton = 1"
        ).fetchone()
        draft_rows = connection.execute(
            "SELECT mode, document, updated_at FROM mode_drafts ORDER BY mode"
        ).fetchall()
        asset_rows = connection.execute(
            "SELECT id, document, created_at FROM assets ORDER BY id"
        ).fetchall()
    assert timeline_row is not None
    return (
        tuple(timeline_row),
        tuple(tuple(row) for row in draft_rows),
        tuple(tuple(row) for row in asset_rows),
    )


async def test_cascade_validation_failure_rolls_back_asset_and_all_documents(
    client, monkeypatch
) -> None:
    first = asset("first.png", "image")
    timeline_before = timeline(
        segment("i2v", "rollback-i2v", first_image=first)
    )
    draft_before = runnable_draft("i2v")
    saved_timeline = await client.put("/api/timeline", json=timeline_before)
    saved_draft = await client.put("/api/drafts/i2v", json=draft_before)
    assert saved_timeline.status_code == 200
    assert saved_draft.status_code == 200
    database = client.director_app.state.database
    before = persisted_rows(database)
    original_validate_mode_draft = database_module.validate_mode_draft

    def fail_after_unbind(mode: str, value: Any):
        shots = value.get("shots", []) if isinstance(value, dict) else []
        if (
            mode == "i2v"
            and shots
            and isinstance(shots[0], dict)
            and shots[0].get("first_image") is None
        ):
            raise ValueError("forced cascade rollback")
        return original_validate_mode_draft(mode, value)

    monkeypatch.setattr(
        database_module, "validate_mode_draft", fail_after_unbind
    )

    with pytest.raises(ValueError, match="forced cascade rollback"):
        database.delete_asset_if_unused(first["id"], cascade=True)

    assert persisted_rows(database) == before
    assert database.get_asset(first["id"]) is not None
    assert database.get_timeline().model_dump(mode="json") == saved_timeline.json()
    assert database.get_draft("i2v").model_dump(mode="json") == saved_draft.json()
