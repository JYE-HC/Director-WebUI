from __future__ import annotations

import asyncio
import json
import sqlite3
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from directordeck.compiler import (
    DraftNotRunnable,
    build_unified_timeline,
    validate_unified_runnable,
)
from directordeck.database import Database
from directordeck.h3_capabilities import H3_REFERENCE_LIMITS
from directordeck.native_templates import NativeTemplateError, compile_native_timeline
from directordeck.schemas import (
    MINIMAX_H3_PROMPT_MAX_CHARACTERS,
    UnifiedTimelineDraft,
    default_settings,
    default_timeline_draft,
    iter_timeline_assets,
    timeline_segment_recipe,
)

from .conftest import (
    adapt_legacy_workflow_requests,
    asset,
    runnable_draft,
    save_timeline_document,
    v5_timeline_document,
    v5_timeline_fixture,
    wait_for_submission_tasks,
)


@pytest.fixture(autouse=True)
def _stage6_v5_request_adapter(client, monkeypatch) -> None:
    adapt_legacy_workflow_requests(client, monkeypatch)


def timeline(*segments: dict, **overrides) -> dict:
    value = default_timeline_draft().model_dump(mode="json")
    value.update(overrides)
    for sampling in value["sampling"].values():
        sampling["seed"] = 42
        sampling["random_seed"] = False
    value["segments"] = list(segments)
    if any(
        item.get("mode") in {"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"}
        for item in segments
    ):
        value["version"] = 1
    else:
        for item in value["segments"]:
            item.setdefault("ref_image_size", "match")
            item.setdefault("audio_mode", "generate")
    return value


def segment(mode: str, identity: str, **values) -> dict:
    value = {
        "id": identity,
        "title": identity,
        "mode": mode,
        "duration_seconds": 5.0,
        "prompt": f"Prompt for {identity}",
        "enabled": True,
    }
    if mode == "i2v":
        value["first_image"] = asset("first.png", "image")
    elif mode == "fl2v":
        value.update(
            first_image=asset("first.png", "image"),
            last_image=asset("last.png", "image"),
        )
    elif mode == "r2v":
        value.update(
            reference_images=[asset("reference.png", "image", slot=0)],
            reference_audios=[],
            reference_videos=[],
        )
    elif mode in {"v2v", "rv2v"}:
        value.update(
            source_video=asset("source.mp4", "video"),
            source_start_seconds=1.0,
            source_duration_seconds=5.0,
        )
        if mode == "rv2v":
            value.update(reference_images=[], reference_audios=[])
    value.update(values)
    return value


async def _submitted_job(client, accepted: dict) -> dict:
    await wait_for_submission_tasks(client)
    response = await client.get(f"/api/jobs/{accepted['id']}")
    assert response.status_code == 200, response.text
    return response.json()


def test_default_timeline_uses_the_segment_title_contract() -> None:
    assert default_timeline_draft().segments[0].title == "片段 01"


def test_h3_prompt_limit_counts_unicode_characters_not_utf8_bytes() -> None:
    value = default_timeline_draft().model_dump(mode="json")
    value["segments"][0]["prompt"] = "中" * MINIMAX_H3_PROMPT_MAX_CHARACTERS
    assert len(UnifiedTimelineDraft.model_validate(value).segments[0].prompt) == 7_000

    value["segments"][0]["prompt"] = "😀" * MINIMAX_H3_PROMPT_MAX_CHARACTERS
    assert len(UnifiedTimelineDraft.model_validate(value).segments[0].prompt) == 7_000

    value["segments"][0]["prompt"] += "中"
    with pytest.raises(ValidationError, match="at most 7000 characters"):
        UnifiedTimelineDraft.model_validate(value)


def test_unified_schema_is_discriminated_and_bounded() -> None:
    leaked = timeline(segment("t2v", "s1", first_image=asset("first.png", "image")))
    with pytest.raises(ValueError, match="first_image"):
        UnifiedTimelineDraft.model_validate(leaked)

    too_many = timeline(*(segment("t2v", f"s{i}") for i in range(129)))
    with pytest.raises(ValueError, match="at most 128"):
        UnifiedTimelineDraft.model_validate(too_many)

    source_audio = UnifiedTimelineDraft.model_validate(
        timeline(
            segment(
                "v2v",
                "source-audio",
                source_audio_as_reference=True,
            )
        )
    )
    assert source_audio.segments[0].source_audio_as_reference is True
    assert UnifiedTimelineDraft.model_validate(
        timeline(segment("rv2v", "default-source-audio"))
    ).segments[0].source_audio_as_reference is False
    with pytest.raises(ValueError, match="source_audio_as_reference"):
        UnifiedTimelineDraft.model_validate(
            timeline(
                segment(
                    "t2v",
                    "leaked-source-audio",
                    source_audio_as_reference=True,
                )
            )
        )

    explicit_v2_legacy = timeline(segment("t2v", "old-in-v2"))
    explicit_v2_legacy["version"] = 2
    with pytest.raises(ValueError, match="union tag|Input tag"):
        UnifiedTimelineDraft.model_validate(explicit_v2_legacy)

    explicit_v1_family = timeline(segment("fl2va", "new-in-v1"))
    explicit_v1_family["version"] = 1
    with pytest.raises(ValueError, match="version 1 accepts only"):
        UnifiedTimelineDraft.model_validate(explicit_v1_family)


def test_ref2va_rejects_duplicate_reference_asset_ids_and_source_alias() -> None:
    duplicate = asset("same.mp4", "video", slot=0)
    duplicate_second_slot = {**duplicate, "slot": 1}
    with pytest.raises(ValueError, match="asset ids must be unique"):
        UnifiedTimelineDraft.model_validate(
            timeline(
                segment(
                    "ref2va",
                    "duplicate-slots",
                    source_video=None,
                    source_start_seconds=0.0,
                    source_duration_seconds=5.0,
                    source_audio_as_reference=False,
                    reference_images=[],
                    reference_audios=[],
                    reference_videos=[duplicate, duplicate_second_slot],
                )
            )
        )

    source = asset("same.mp4", "video")
    with pytest.raises(ValueError, match="cannot also occupy"):
        UnifiedTimelineDraft.model_validate(
            timeline(
                segment(
                    "ref2va",
                    "source-alias",
                    source_video=source,
                    source_start_seconds=0.0,
                    source_duration_seconds=5.0,
                    source_audio_as_reference=False,
                    reference_images=[],
                    reference_audios=[],
                    reference_videos=[{**source, "slot": 0}],
                )
            )
        )


def test_legacy_shared_prompt_sampling_and_cfg_migrate_losslessly() -> None:
    value = default_timeline_draft().model_dump(mode="json")
    value["version"] = 1
    value["prompt"] = "legacy shared prompt"
    value["sampling"] = {
        "steps": 19,
        "cfg": 9.0,
        "seed": 314159,
        "sampler": "euler",
        "scheduler": "beta",
        "shift": 8.0,
        "audio_shift": 2.5,
    }
    value["segments"] = [
        segment("t2v", "inherits", prompt=""),
        segment("r2v", "specific", prompt="specific prompt"),
    ]

    migrated = UnifiedTimelineDraft.model_validate(value)
    dumped = migrated.model_dump(mode="json")

    assert "prompt" not in dumped
    assert [item.prompt for item in migrated.segments] == [
        "legacy shared prompt",
        "specific prompt",
    ]
    assert migrated.sampling.fl2va == migrated.sampling.ref2va
    assert migrated.sampling.fl2va.seed == 314159
    assert migrated.sampling.fl2va.random_seed is False
    assert migrated.sampling.fl2va.scheduler == "beta"
    assert migrated.sampling.ref2va.scheduler == "beta"
    assert "cfg" not in dumped["sampling"]["fl2va"]


def test_legacy_shared_random_seed_is_drawn_once_then_copied_to_both_families() -> None:
    value = default_timeline_draft().model_dump(mode="json")
    value["version"] = 1
    value["segments"] = [segment("t2v", "legacy")]
    value["sampling"] = {
        "steps": 25,
        "cfg": 1.0,
        "seed": -1,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "shift": 12.0,
        "audio_shift": 3.0,
    }

    migrated = UnifiedTimelineDraft.model_validate(value)

    assert migrated.sampling.fl2va == migrated.sampling.ref2va
    assert migrated.sampling.fl2va.random_seed is True
    assert 0 <= migrated.sampling.fl2va.seed <= 2**53 - 1


def test_v1_six_modes_migrate_losslessly_to_v3_families_and_recipes() -> None:
    original_modes = ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v")
    legacy_segments = [segment(mode, mode) for mode in original_modes]
    legacy_segments[-1]["reference_images"] = [
        asset("identity.png", "image", slot=0)
    ]
    value = timeline(*legacy_segments)
    value["version"] = 1

    migrated = UnifiedTimelineDraft.model_validate(value)
    dumped = migrated.model_dump(mode="json")

    assert migrated.version == 4
    assert [item.mode for item in migrated.segments] == [
        "fl2va",
        "fl2va",
        "fl2va",
        "ref2va",
        "ref2va",
        "ref2va",
    ]
    assert [timeline_segment_recipe(item) for item in migrated.segments] == list(
        original_modes
    )
    assert dumped["version"] == 4
    assert not ({"t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"} & {
        item["mode"] for item in dumped["segments"]
    })


async def test_legacy_timeline_put_requires_the_v5_revision_authority(
    client,
) -> None:
    legacy_segments = [
        segment(mode, f"api-{mode}")
        for mode in ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v")
    ]
    legacy_segments[-1]["reference_images"] = [
        asset("identity.png", "image", slot=0)
    ]

    response = await client.put(
        "/api/timeline",
        json=timeline(*legacy_segments),
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "timeline_authority_required"
    assert (await client.get("/api/timeline")).json()["version"] == 5


@pytest.mark.parametrize("enabled", [False, True])
def test_v2_project_continuity_migrates_to_every_v3_segment(enabled: bool) -> None:
    value = default_timeline_draft().model_dump(mode="json")
    value["version"] = 2
    value["continuity"] = {"enabled": enabled, "overlap_frames": 56}
    value["segments"].append({
        **value["segments"][0],
        "id": "disabled-segment",
        "enabled": False,
    })
    for item in value["segments"]:
        item.pop("continuity")
        item.pop("ref_image_size")
        item.pop("audio_mode")

    migrated = UnifiedTimelineDraft.model_validate(value)
    dumped = migrated.model_dump(mode="json")

    assert migrated.version == 4
    assert "continuity" not in dumped
    assert [item.continuity.model_dump(mode="json") for item in migrated.segments] == [
        {"enabled": enabled, "overlap_frames": 56},
        {"enabled": enabled, "overlap_frames": 56},
    ]

    value["segments"][0]["continuity"] = {
        "enabled": not enabled,
        "overlap_frames": 5,
    }
    with pytest.raises(ValueError, match="version 2 stores continuity only"):
        UnifiedTimelineDraft.model_validate(value)


def test_v3_project_media_policies_migrate_to_every_v4_segment() -> None:
    value = default_timeline_draft().model_dump(mode="json")
    value["version"] = 3
    value["ref_image_size"] = "max"
    value["audio_mode"] = "mute"
    value["segments"].append({
        **value["segments"][0],
        "id": "second-segment",
    })
    for item in value["segments"]:
        item.pop("ref_image_size")
        item.pop("audio_mode")

    migrated = UnifiedTimelineDraft.model_validate(value)

    assert migrated.version == 4
    assert all(item.ref_image_size == "max" for item in migrated.segments)
    assert all(item.audio_mode == "mute" for item in migrated.segments)
    dumped = migrated.model_dump(mode="json")
    assert "ref_image_size" not in dumped
    assert "audio_mode" not in dumped


@pytest.mark.parametrize(
    ("values", "recipe"),
    [
        ({}, "t2v"),
        ({"first_image": asset("first.png", "image")}, "i2v"),
        ({"last_image": asset("last.png", "image")}, "fl2v"),
        (
            {
                "first_image": asset("first.png", "image"),
                "last_image": asset("last.png", "image"),
            },
            "fl2v",
        ),
    ],
)
def test_fl2va_recipe_is_derived_from_first_and_last_anchors(
    values: dict, recipe: str
) -> None:
    draft = UnifiedTimelineDraft.model_validate(
        timeline(segment("fl2va", recipe, **values))
    )

    assert timeline_segment_recipe(draft.segments[0]) == recipe


@pytest.mark.parametrize(
    ("values", "recipe"),
    [
        (
            {"reference_images": [asset("reference.png", "image", slot=0)]},
            "r2v",
        ),
        (
            {
                "source_video": asset("source.mp4", "video"),
                "source_start_seconds": 1.0,
                "source_duration_seconds": 5.0,
            },
            "v2v",
        ),
        (
            {
                "source_video": asset("source.mp4", "video"),
                "source_start_seconds": 1.0,
                "source_duration_seconds": 5.0,
                "reference_audios": [asset("voice.wav", "audio", slot=0)],
            },
            "rv2v",
        ),
    ],
)
def test_ref2va_recipe_keeps_source_and_independent_references_distinct(
    values: dict, recipe: str
) -> None:
    draft = UnifiedTimelineDraft.model_validate(
        timeline(segment("ref2va", recipe, **values))
    )

    assert timeline_segment_recipe(draft.segments[0]) == recipe


def test_ref2va_source_and_reference_video_share_stock_three_video_limit() -> None:
    allowed = UnifiedTimelineDraft.model_validate(
        timeline(
            segment(
                "ref2va",
                "allowed",
                source_video=asset("source.mp4", "video"),
                source_start_seconds=1.0,
                source_duration_seconds=5.0,
                reference_videos=[
                    asset("motion.mp4", "video", slot=0),
                    asset("other.mp4", "video", slot=1),
                ],
            )
        )
    )
    assert timeline_segment_recipe(allowed.segments[0]) == "rv2v"

    too_many = allowed.model_dump(mode="json")
    too_many["segments"][0]["reference_videos"].append(
        asset("third.mp4", "video", slot=2)
    )
    with pytest.raises(
        ValueError,
        match="MiniMax H3 包括源视频在内最多支持 3 路参考视频",
    ):
        UnifiedTimelineDraft.model_validate(too_many)


def test_h3_reference_capacity_contract_matches_stock_node() -> None:
    assert H3_REFERENCE_LIMITS.reference_images == 9
    assert H3_REFERENCE_LIMITS.reference_video_channels == 3
    assert H3_REFERENCE_LIMITS.standalone_reference_audios == 3
    assert H3_REFERENCE_LIMITS.source_videos == 1
    assert H3_REFERENCE_LIMITS.paired_reference_video_audios == 3


@pytest.mark.parametrize("mode", ["r2v", "ref2va"])
@pytest.mark.parametrize(
    ("field", "kind", "accepted_count", "rejected_count", "message"),
    [
        ("reference_images", "image", 9, 10, "参考图片最多 9 张"),
        ("reference_audios", "audio", 3, 4, "独立参考音频最多 3 条"),
        ("reference_videos", "video", 3, 4, "参考视频通道最多 3 路"),
    ],
)
def test_h3_reference_boundaries_apply_to_v2_and_legacy_schemas(
    mode: str,
    field: str,
    kind: str,
    accepted_count: int,
    rejected_count: int,
    message: str,
) -> None:
    suffix = {"image": "png", "audio": "wav", "video": "mp4"}[kind]

    def references(count: int) -> list[dict]:
        return [
            asset(f"{field}-{index}.{suffix}", kind, slot=index)
            for index in range(count)
        ]

    accepted = UnifiedTimelineDraft.model_validate(
        timeline(
            segment(
                mode,
                f"{mode}-{field}-max",
                **{field: references(accepted_count)},
            )
        )
    )
    assert len(getattr(accepted.segments[0], field)) == accepted_count

    with pytest.raises(ValueError, match=message):
        UnifiedTimelineDraft.model_validate(
            timeline(
                segment(
                    mode,
                    f"{mode}-{field}-overflow",
                    **{field: references(rejected_count)},
                )
            )
        )


def test_source_soundtrack_does_not_reduce_standalone_audio_capacity() -> None:
    draft = UnifiedTimelineDraft.model_validate(
        timeline(
            segment(
                "ref2va",
                "source-and-three-audios",
                source_video=asset("source-with-audio.mp4", "video"),
                source_start_seconds=0.0,
                source_duration_seconds=5.0,
                source_audio_as_reference=True,
                reference_audios=[
                    asset(f"voice-{index}.wav", "audio", slot=index)
                    for index in range(3)
                ],
            )
        )
    )

    parsed = draft.segments[0]
    assert parsed.source_audio_as_reference is True
    assert len(parsed.reference_audios) == 3


def test_ref2va_editing_can_be_incomplete_but_compile_fails_closed() -> None:
    draft = UnifiedTimelineDraft.model_validate(
        timeline(segment("ref2va", "unfinished"))
    )

    assert timeline_segment_recipe(draft.segments[0]) == "r2v"
    with pytest.raises(DraftNotRunnable, match="Ref2VA segments need"):
        validate_unified_runnable(draft)


def test_fl2va_picture_tags_follow_present_keyframe_order() -> None:
    first = asset("first.png", "image")
    last = asset("last.png", "image")
    cases = [
        ({"first_image": first}, "Start from <Picture 1>.", "<Picture 2>"),
        ({"last_image": last}, "End on <Picture 1>.", "<Picture 2>"),
        (
            {"first_image": first, "last_image": last},
            "Move from <Picture 1> to <Picture 2>.",
            "<Picture 3>",
        ),
    ]

    for index, (anchors, prompt, invalid_tag) in enumerate(cases):
        draft = UnifiedTimelineDraft.model_validate(
            timeline(
                segment(
                    "fl2va",
                    f"fl-picture-{index}",
                    prompt=prompt,
                    **anchors,
                )
            )
        )
        validate_unified_runnable(draft)
        draft.segments[0].prompt = f"Invalid {invalid_tag}."
        with pytest.raises(DraftNotRunnable, match=invalid_tag.replace(" ", r"\s")):
            validate_unified_runnable(draft)


def test_complete_long_source_is_a_valid_saved_edit_state_but_not_runnable() -> None:
    source = asset("long-source.mp4", "video")
    source["metadata"].update(duration=180.0, frame_count=4_320)
    draft = UnifiedTimelineDraft.model_validate(
        timeline(
            segment(
                "ref2va",
                "long-source",
                duration_seconds=180.0,
                source_video=source,
                source_start_seconds=0.0,
                source_duration_seconds=180.0,
            )
        )
    )

    assert draft.segments[0].duration_seconds == 180.0
    assert draft.segments[0].source_duration_seconds == 180.0
    with pytest.raises(NativeTemplateError, match="native H3 template limit is 512"):
        compile_native_timeline(draft, default_settings(), "long-source-edit")


async def test_complete_long_source_autosave_succeeds_before_split(client) -> None:
    source = asset("long-source.mp4", "video")
    source["metadata"].update(duration=180.0, frame_count=4_320)
    client.director_app.state.database.put_asset(
        source["id"],
        source,
    )
    draft = timeline(
        segment(
            "ref2va",
            "long-source",
            duration_seconds=180.0,
            source_video=source,
            source_start_seconds=0.0,
            source_duration_seconds=180.0,
        )
    )

    saved = await save_timeline_document(client, draft)
    compile_response = await client.post("/api/timeline/compile", json={"config": draft})

    assert saved.status_code == 200, saved.text
    assert saved.json()["segments"][0]["duration_seconds"] == 180.0
    assert compile_response.status_code == 422, compile_response.text
    assert "512" in compile_response.text


async def test_empty_ref2va_can_be_saved_but_not_submitted(client, fake_comfy) -> None:
    draft = timeline(segment("ref2va", "unfinished"))

    saved = await save_timeline_document(client, draft)
    submitted = await client.post("/api/timeline/jobs", json={"config": draft})

    assert saved.status_code == 200, saved.text
    assert saved.json()["segments"][0]["mode"] == "ref2va"
    assert submitted.status_code == 422, submitted.text
    assert "Ref2VA segments need" in submitted.text
    assert fake_comfy.prompts == []


def test_v2_asset_iterator_covers_all_unified_media_slots() -> None:
    draft = UnifiedTimelineDraft.model_validate(
        timeline(
            segment(
                "fl2va",
                "fl",
                first_image=asset("first.png", "image"),
                last_image=asset("last.png", "image"),
            ),
            segment(
                "ref2va",
                "ref",
                source_video=asset("source.mp4", "video"),
                source_start_seconds=1.0,
                source_duration_seconds=5.0,
                reference_images=[asset("reference.png", "image", slot=0)],
                reference_audios=[asset("voice.wav", "audio", slot=0)],
                reference_videos=[asset("motion.mp4", "video", slot=0)],
            ),
        )
    )

    assert [location for location, _ in iter_timeline_assets(draft)] == [
        "segments[0](fl).first_image",
        "segments[0](fl).last_image",
        "segments[1](ref).source_video",
        "segments[1](ref).reference_images[0]",
        "segments[1](ref).reference_audios[0]",
        "segments[1](ref).reference_videos[0]",
    ]


def test_source_soundtrack_prompt_labels_are_mode_local_and_offset_rv2v_audio() -> None:
    v2v = UnifiedTimelineDraft.model_validate(
        timeline(
            segment(
                "v2v",
                "paired-v2v",
                prompt="Follow <Video 1> and <Audio 1>.",
                source_audio_as_reference=True,
            )
        )
    )
    validate_unified_runnable(v2v)

    disabled = v2v.model_copy(deep=True)
    disabled.segments[0].source_audio_as_reference = False
    with pytest.raises(DraftNotRunnable, match="<Audio 1>"):
        validate_unified_runnable(disabled)

    silent = v2v.model_copy(deep=True)
    assert silent.segments[0].source_video is not None
    assert silent.segments[0].source_video.metadata is not None
    silent.segments[0].source_video.metadata.has_audio = False
    with pytest.raises(DraftNotRunnable, match="with an audio stream"):
        validate_unified_runnable(silent)

    rv2v = UnifiedTimelineDraft.model_validate(
        timeline(
            segment(
                "rv2v",
                "paired-rv2v",
                prompt="Match source <Audio 1> to voice <Audio 2>.",
                source_audio_as_reference=True,
                reference_audios=[asset("voice.wav", "audio", slot=0)],
            )
        )
    )
    validate_unified_runnable(rv2v)
    rv2v.segments[0].prompt = "There is no <Audio 3>."
    with pytest.raises(DraftNotRunnable, match="<Audio 3>"):
        validate_unified_runnable(rv2v)


def test_source_output_audio_fails_closed_for_silent_or_historical_video() -> None:
    value = timeline(
        segment(
            "v2v",
            "silent-output",
            duration_seconds=56 / 24,
            source_duration_seconds=56 / 24,
        ),
        audio_mode="source",
    )
    value["segments"][0]["source_video"]["metadata"]["has_audio"] = False
    with pytest.raises(DraftNotRunnable, match="audio_mode='source'.*no audio stream"):
        validate_unified_runnable(UnifiedTimelineDraft.model_validate(value))


def test_mixed_timeline_compiles_to_two_server_owned_native_units() -> None:
    draft = UnifiedTimelineDraft.model_validate(
        timeline(segment("t2v", "opening"), segment("r2v", "hero"))
    )
    result = compile_native_timeline(draft, default_settings(), "mixed-job")

    assert result.families == ("fl2va", "ref2va")
    assert [unit.family for unit in result.workflows] == ["fl2va", "ref2va"]
    assert all(
        "MiniMaxH3Director"
        not in {node["class_type"] for node in unit.prompt.values()}
        for unit in result.workflows
    )
    assert result.node_policy["graph_source"] == "server"
    assert result.node_policy["accepts_client_workflow"] is False
    assert [(plan["mode"], plan["recipe"]) for plan in result.plans] == [
        ("fl2va", "t2v"),
        ("ref2va", "r2v"),
    ]


def test_family_sampling_branches_drive_exact_nodes_and_cache_keys() -> None:
    value = timeline(segment("t2v", "opening"), segment("r2v", "hero"))
    value["sampling"]["fl2va"].update(
        steps=11,
        seed=111,
        random_seed=False,
        sampler="euler",
        scheduler="normal",
        shift=7.0,
        audio_shift=2.0,
    )
    value["sampling"]["ref2va"].update(
        steps=33,
        seed=222,
        random_seed=True,
        sampler="dpmpp_2m",
        scheduler="beta",
        shift=17.0,
        audio_shift=4.0,
    )
    draft = UnifiedTimelineDraft.model_validate(value)
    settings = default_settings()
    result = compile_native_timeline(draft, settings, "family-sampling")

    expected = {
        "fl2va": (11, 111, "euler", "normal", 7.0, 2.0, "fixed"),
        "ref2va": (33, 222, "dpmpp_2m", "beta", 17.0, 4.0, "random"),
    }
    for unit, plan in zip(result.workflows, result.plans, strict=True):
        nodes = list(unit.prompt.values())
        scheduler = next(node for node in nodes if node["class_type"] == "BasicScheduler")
        noise = next(node for node in nodes if node["class_type"] == "RandomNoise")
        sampler = next(node for node in nodes if node["class_type"] == "KSamplerSelect")
        sigma = next(node for node in nodes if node["class_type"] == "MiniMaxH3SigmaShift")
        steps, seed, sampler_name, scheduler_name, shift, audio_shift, seed_mode = expected[unit.family]
        assert scheduler["inputs"]["steps"] == steps
        assert scheduler["inputs"]["scheduler"] == scheduler_name
        assert noise["inputs"]["noise_seed"] == seed
        assert sampler["inputs"]["sampler_name"] == sampler_name
        assert sigma["inputs"]["shift_video"] == shift
        assert sigma["inputs"]["shift_audio"] == audio_shift
        assert (plan["seed_mode"], plan["seed"]) == (seed_mode, seed)

    first = build_unified_timeline(draft, settings)
    first_keys = {
        item["id"]: item["cacheKey"] for item in first["segments"]
    }
    changed = draft.model_copy(deep=True)
    changed.sampling.ref2va.scheduler = "karras"
    second = build_unified_timeline(changed, settings)
    second_keys = {
        item["id"]: item["cacheKey"] for item in second["segments"]
    }
    assert second_keys["opening"] == first_keys["opening"]
    assert second_keys["hero"] != first_keys["hero"]


def test_native_global_constraints_fail_closed() -> None:
    value = timeline(segment("t2v", "shot"))
    value["render"].update(fps=30.0)
    with pytest.raises(DraftNotRunnable, match="must equal 24"):
        validate_unified_runnable(UnifiedTimelineDraft.model_validate(value))


def test_continuity_partial_run_keeps_authored_predecessor_constraints() -> None:
    value = timeline(segment("t2v", "opening"), segment("t2v", "continuation"))
    value["continuity"] = {"enabled": True, "overlap_frames": 22}
    draft = UnifiedTimelineDraft.model_validate(value)

    validate_unified_runnable(draft)
    validate_unified_runnable(draft, segment_ids=["continuation"])
    validate_unified_runnable(
        draft, segment_ids=["continuation", "opening"]
    )


def test_continuity_first_image_starts_a_new_chain() -> None:
    value = timeline(segment("t2v", "opening"), segment("i2v", "new-anchor"))
    value["continuity"] = {"enabled": True, "overlap_frames": 22}
    draft = UnifiedTimelineDraft.model_validate(value)

    validate_unified_runnable(draft, segment_ids=["new-anchor"])


def test_continuity_allows_cross_family_but_rejects_internal_frame_overflow() -> None:
    mixed = timeline(segment("t2v", "opening"), segment("r2v", "reference"))
    mixed["continuity"] = {"enabled": True, "overlap_frames": 22}
    validate_unified_runnable(UnifiedTimelineDraft.model_validate(mixed))

    oversized = timeline(
        segment("t2v", "opening"),
        segment("t2v", "continuation", duration_seconds=20.75),
    )
    oversized["continuity"] = {"enabled": True, "overlap_frames": 22}
    with pytest.raises(DraftNotRunnable, match="internal sample frames"):
        validate_unified_runnable(UnifiedTimelineDraft.model_validate(oversized))


def test_partial_run_ignores_unfinished_unselected_segment() -> None:
    unfinished = segment("r2v", "unfinished")
    unfinished["reference_images"] = []
    draft = UnifiedTimelineDraft.model_validate(
        timeline(segment("t2v", "ready"), unfinished)
    )

    validate_unified_runnable(draft, segment_ids=["ready"])
    result = compile_native_timeline(
        draft, default_settings(), "partial-job", segment_ids=["ready"]
    )
    assert result.families == ("fl2va",)
    assert result.workflows[0].segment_ids == ("ready",)


def test_initialize_imports_only_one_custom_legacy_draft(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    database = Database(path)
    database.initialize()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE unified_timeline")
        connection.execute(
            "UPDATE settings SET document = ? WHERE singleton = 1",
            (default_settings().model_dump_json(),),
        )
        custom = runnable_draft("i2v")
        custom["prompt"] = "Imported legacy prompt"
        connection.execute(
            "UPDATE mode_drafts SET document = ? WHERE mode = 'i2v'",
            (json.dumps(custom),),
        )
    database.initialize()
    imported = database.get_timeline()
    assert [item.mode for item in imported.segments] == ["fl2va"]
    assert imported.segments[0].prompt == "A cinematic camera move"
    assert imported.segments[0].first_image is not None


def test_initialize_does_not_invent_order_for_multiple_legacy_drafts(tmp_path) -> None:
    path = tmp_path / "legacy-multiple.sqlite3"
    database = Database(path)
    database.initialize()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE unified_timeline")
        connection.execute(
            "UPDATE settings SET document = ? WHERE singleton = 1",
            (default_settings().model_dump_json(),),
        )
        for mode in ("t2v", "r2v"):
            custom = runnable_draft(mode)
            custom["prompt"] = f"custom-{mode}"
            connection.execute(
                "UPDATE mode_drafts SET document = ? WHERE mode = ?",
                (json.dumps(custom), mode),
            )
    database.initialize()
    expected = v5_timeline_fixture(
        default_timeline_draft().model_dump(mode="json"),
        default_settings().model_dump(mode="json"),
    )
    expected["features"]["template_bundle_version"] = 5
    assert database.get_timeline().model_dump(mode="json") == expected
    with sqlite3.connect(path) as connection:
        notice = connection.execute(
            "SELECT message FROM migration_notices WHERE id = 'multiple-legacy-mode-drafts'"
        ).fetchone()
    assert notice is not None and "t2v" in notice[0] and "r2v" in notice[0]


def test_restart_releases_stale_assembly_claim_for_retry(tmp_path) -> None:
    path = tmp_path / "assembly-recovery.sqlite3"
    database = Database(path)
    database.initialize()
    now = "2026-08-12T00:00:00+00:00"
    database.create_job(
        {
            "id": "assembly-parent",
            "mode": "timeline",
            "status": "running",
            "progress": 1.0,
            "stage": "assembling",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {"timeline": timeline(segment("t2v", "a"))},
            "settings_snapshot": default_settings().model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": None,
        }
    )

    assert database.recover_interrupted_assemblies() == 1
    recovered = database.get_job("assembly-parent")
    assert recovered is not None
    assert recovered["status"] == "running"
    assert recovered["stage"] == "assembly_retry"
    assert database.recover_interrupted_assemblies() == 0


async def test_cancelled_parent_with_completed_children_does_not_start_assembly(
    client, fake_comfy, monkeypatch
) -> None:
    draft = timeline(segment("t2v", "first"), segment("t2v", "second"))
    created = (await client.post("/api/timeline/jobs", json={"config": draft})).json()
    created = await _submitted_job(client, created)
    child = client.director_app.state.database.list_job_children(created["id"])[0]
    # This fixture models one completed child while the remaining per-segment
    # prompt is still pending. Removing every pending prompt would instead be
    # a real external queue-clear, which reconciliation now reports explicitly.
    fake_comfy.pending = [
        entry for entry in fake_comfy.pending if child["prompt_id"] not in entry
    ]
    fake_comfy.histories[child["prompt_id"]] = {
        "status": {"status_str": "success", "completed": True, "messages": []},
        "outputs": {
            node_id: {
                "videos": [{
                    "filename": f"{segment_id}.mp4",
                    "subfolder": "video",
                    "type": "output",
                }]
            }
            for segment_id, node_id in child["output_nodes"].items()
        },
    }
    assemble = Mock(side_effect=AssertionError("assembly must not start"))
    monkeypatch.setattr("directordeck.app.assemble_video_bytes", assemble)

    cancelled = await client.post(f"/api/jobs/{created['id']}/cancel")

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert assemble.call_count == 0


async def test_concurrent_parent_submissions_do_not_interleave_family_units(
    client, fake_comfy
) -> None:
    fake_comfy.submit_started = asyncio.Event()
    fake_comfy.submit_release = asyncio.Event()
    mixed_request = asyncio.create_task(
        client.post(
            "/api/timeline/jobs",
            json={
                "config": timeline(
                    segment("t2v", "mixed-fl"), segment("r2v", "mixed-ref")
                )
            },
        )
    )
    await asyncio.wait_for(fake_comfy.submit_started.wait(), timeout=1)
    mixed_job = client.director_app.state.database.list_jobs()[0]
    mixed_child_ids = [
        child["id"]
        for child in client.director_app.state.database.list_job_children(
            mixed_job["id"]
        )
    ]

    single_request = asyncio.create_task(
        client.post(
            "/api/timeline/jobs",
            json={"config": timeline(segment("t2v", "single"))},
        )
    )
    await asyncio.sleep(0)
    fake_comfy.submit_release.set()
    mixed_response, single_response = await asyncio.gather(
        mixed_request, single_request
    )
    await wait_for_submission_tasks(client)

    assert mixed_response.status_code == single_response.status_code == 200
    assert [item["prompt_id"] for item in fake_comfy.prompts[:2]] == mixed_child_ids
    assert fake_comfy.prompts[2]["prompt_id"] not in mixed_child_ids


async def test_timeline_routes_are_redacted_and_submit_native_children(
    client, fake_comfy
) -> None:
    draft = timeline(segment("t2v", "ready"))
    saved = await save_timeline_document(client, draft)
    assert saved.status_code == 200, saved.text
    config = saved.json()

    response = await client.post(
        "/api/timeline/compile", json={"config": config}
    )
    assert response.status_code == 200, response.text
    compiled = response.json()
    assert compiled["execution_strategy"] == "native_segment_graph_v1"
    assert compiled["model_families"] == ["fl2va"]
    assert compiled["plans"][0]["seed_mode"] == "fixed"
    assert compiled["plans"][0]["seed"] == 42
    assert compiled["plans"][0]["mode"] == "fl2va"
    assert compiled["plans"][0]["recipe"] == "t2v"
    assert {"prompt", "workflow", "timeline_data"}.isdisjoint(compiled)
    assert compiled["node_policy"]["accepts_client_workflow"] is False

    response = await client.post(
        "/api/timeline/jobs", json={"config": config}
    )
    assert response.status_code == 200, response.text
    job = await _submitted_job(client, response.json())
    assert job["mode"] == "timeline"
    assert job["prompt_id"] == job["children"][0]["prompt_id"]
    assert len(job["children"]) == 1
    types = {node["class_type"] for node in fake_comfy.prompts[-1]["prompt"].values()}
    assert "MiniMaxH3Director" not in types
    assert "SaveVideo" in types


async def test_public_preflight_uses_compiled_execution_plan_only(
    client, monkeypatch
) -> None:
    def forbidden_legacy_compile(*_args, **_kwargs):
        raise AssertionError("public preflight must not call NativeCompileResult")

    monkeypatch.setattr(
        "directordeck.native_templates.compile_native_timeline",
        forbidden_legacy_compile,
    )
    response = await client.post(
        "/api/timeline/compile",
        json={"config": timeline(segment("t2v", "plan-only-preflight"))},
    )

    assert response.status_code == 200, response.text
    assert response.json()["plans"][0]["segment_id"] == "plan-only-preflight"


async def test_random_seed_report_and_submit_use_the_same_visible_safe_value(
    client, fake_comfy
) -> None:
    draft = timeline(segment("t2v", "random"))
    draft["sampling"]["fl2va"].update(seed=8_765_432_101, random_seed=True)

    preview = await client.post("/api/timeline/compile", json={"config": draft})

    assert preview.status_code == 200, preview.text
    assert preview.json()["plans"][0]["seed_mode"] == "random"
    assert preview.json()["plans"][0]["seed"] == 8_765_432_101

    submitted = await client.post("/api/timeline/jobs", json={"config": draft})
    assert submitted.status_code == 200, submitted.text
    await wait_for_submission_tasks(client)
    noise = next(
        node
        for node in fake_comfy.prompts[-1]["prompt"].values()
        if node["class_type"] == "RandomNoise"
    )
    assert noise["inputs"]["noise_seed"] == 8_765_432_101


async def test_fixed_seed_above_browser_safe_integer_is_rejected(client) -> None:
    draft = await v5_timeline_document(
        client,
        timeline(segment("t2v", "unsafe-seed")),
    )
    draft["sampling"]["fl2va"]["seed"] = 2**53

    response = await client.post("/api/timeline/compile", json={"config": draft})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-3:] == ["sampling", "fl2va", "seed"]
    assert response.json()["detail"][0]["ctx"]["le"] == 2**53 - 1


async def test_mixed_timeline_parent_has_two_authoritative_child_prompts(
    client, fake_comfy
) -> None:
    body = {"config": timeline(segment("t2v", "opening"), segment("r2v", "hero"))}
    response = await client.post("/api/timeline/jobs", json=body)
    assert response.status_code == 200, response.text
    job = await _submitted_job(client, response.json())
    assert job["prompt_id"] is None
    assert [child["family"] for child in job["children"]] == ["fl2va", "ref2va"]
    assert all(child["prompt_id"] for child in job["children"])
    assert len({child["prompt_id"] for child in job["children"]}) == 2


async def test_selected_compile_does_not_validate_unselected_stale_asset(client) -> None:
    stale = segment("i2v", "later")
    stale["first_image"] = {
        "name": "gone.png",
        "subfolder": "directordeck",
        "type": "input",
        "kind": "image",
        "id": "not-registered",
    }
    body = {
        "config": timeline(segment("t2v", "ready"), stale),
        "segment_ids": ["ready"],
    }
    assert (await client.post("/api/timeline/compile", json=body)).status_code == 200
    body["segment_ids"] = ["later"]
    blocked = await client.post("/api/timeline/compile", json=body)
    assert blocked.status_code == 422
    assert blocked.json()["detail"]["code"] == "asset_unavailable"
    assert "not-registered" not in blocked.text


async def test_timeline_submission_requires_every_selected_native_node(
    client, fake_comfy
) -> None:
    fake_comfy.available_nodes.remove("BasicGuider")
    response = await client.post(
        "/api/timeline/jobs", json={"config": timeline(segment("t2v", "ready"))}
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "node_unavailable"
    assert any(
        reason["safe_details"] == {"class_type": "BasicGuider"}
        for reason in detail["reasons"]
    )
    assert fake_comfy.prompts == []


async def test_timeline_submission_does_not_pin_host_owned_node_provenance(
    client, fake_comfy
) -> None:
    fake_comfy.node_provenance["UNETLoader"] = "custom_nodes.lookalike"
    response = await client.post(
        "/api/timeline/jobs", json={"config": timeline(segment("t2v", "ready"))}
    )
    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    assert fake_comfy.prompts


@pytest.mark.parametrize("raylight", [False, True])
async def test_timeline_preflight_rejects_unavailable_logical_gpu(
    client, fake_comfy, raylight: bool
) -> None:
    authority = (await client.get("/api/settings/authority")).json()
    settings = authority["settings"]
    if raylight:
        settings["multi_gpu_enabled"] = True
        settings["placement"]["fl2va"].update(
            device="default",
            raylight={
                "gpu_select": [0, 2],
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_degree": 1,
                "dp_degree": 1,
                "fsdp": False,
                "cpu_offload": False,
            },
        )
    else:
        settings["placement"]["fl2va"]["device"] = "gpu:3"
    saved = await client.put(
        "/api/settings/authority",
        json={
            "document": settings,
            "expected_authority_token": authority["authority_token"],
            "schema_version": 3,
        },
    )
    assert saved.status_code == 200, saved.text

    response = await client.post(
        "/api/timeline/jobs", json={"config": timeline(segment("t2v", "ready"))}
    )

    assert response.status_code == 422
    if raylight:
        detail = response.json()["detail"]
        assert detail["code"] == "invalid_runtime_gpu_indices"
        assert detail["reasons"][0]["safe_details"] == {
            "invalid_indices": [2]
        }
    else:
        detail = response.json()["detail"]
        assert detail["code"] == "runtime_placement_unavailable"
        assert detail["reasons"][0]["safe_details"] == {
            "devices": ["gpu:3"]
        }
    assert fake_comfy.prompts == []


async def test_asset_library_refuses_referenced_delete(
    client,
) -> None:
    database = client.director_app.state.database
    foreign = asset("foreign.png", "image")
    foreign["id"] = "foreign-origin"
    database.put_asset(foreign["id"], foreign)
    listed = await client.get("/api/assets?kind=image")
    ids = {item["id"] for item in listed.json()["assets"]}
    assert "fixture-image-first.png" in ids
    assert "foreign-origin" in ids

    draft = timeline(
        segment("i2v", "image-shot", first_image=asset("first.png", "image"))
    )
    assert (await save_timeline_document(client, draft)).status_code == 200
    blocked = await client.delete("/api/assets/fixture-image-first.png")
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["usages"] == [
        "timeline.segments[0](image-shot).first_image"
    ]

    await save_timeline_document(client, timeline(segment("t2v", "clear")))
    deleted = await client.delete("/api/assets/fixture-image-first.png")
    assert deleted.status_code == 200
    assert deleted.json()["outputs_preserved"] is True
