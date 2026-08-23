from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from directordeck.native_templates import NativeHistoricalTake
from directordeck.schemas import (
    LoraLoaderOverrideRecord,
    RuntimeSettings,
    UnifiedTimelineDraft,
    default_settings,
    default_timeline_draft,
)
from directordeck.workflow.lora_factory import (
    LoraLoaderBindingKey,
    resolve_standard_lora_adapter,
)
from directordeck.workflow.v4_resolver import (
    CreativeCompileInputError,
    CreativeCompileInputResolver,
    V4CreativeCompileInput,
)


def _image(name: str = "anchor.png", *, slot: int | None = None) -> dict:
    value = {
        "id": f"asset-{name}",
        "name": name,
        "subfolder": "resolver",
        "type": "input",
        "kind": "image",
    }
    if slot is not None:
        value["slot"] = slot
    return value


def _segment(
    segment_id: str,
    family: str,
    *,
    enabled: bool = True,
    continuity: bool = False,
    first_image: bool = False,
    reference_image_slot: int | None = None,
) -> dict:
    common = {
        "id": segment_id,
        "title": segment_id,
        "prompt": "",
        "duration_seconds": 5.0,
        "enabled": enabled,
        "mode": family,
        "continuity": {"enabled": continuity, "overlap_frames": 22},
        "ref_image_size": "match",
        "audio_mode": "generate",
    }
    if family == "fl2va":
        common.update(
            first_image=_image(f"{segment_id}.png") if first_image else None,
            last_image=None,
        )
    else:
        common.update(
            source_video=None,
            source_start_seconds=0.0,
            source_duration_seconds=5.0,
            source_audio_as_reference=False,
            reference_images=(
                []
                if reference_image_slot is None
                else [_image("reference.png", slot=reference_image_slot)]
            ),
            reference_audios=[],
            reference_videos=[],
        )
    return common


def _draft(*segments: dict) -> UnifiedTimelineDraft:
    value = default_timeline_draft().model_dump(mode="json")
    value["title"] = "v4 resolver test"
    value["segments"] = list(segments)
    return UnifiedTimelineDraft.model_validate(value)


def _settings(
    *,
    fl_backend: str = "standard",
    ref_backend: str = "standard",
    residency: str = "keep_until_switch",
) -> RuntimeSettings:
    value = default_settings().model_dump(mode="json")
    value["raylight_residency_policy"] = residency
    value["multi_gpu_enabled"] = "raylight" in {fl_backend, ref_backend}
    for family, backend in (("fl2va", fl_backend), ("ref2va", ref_backend)):
        raylight = backend == "raylight"
        value["models"][family]["raylight"] = {
            "gpu_select": [0, 1] if raylight else [0],
            "ulysses_degree": 2 if raylight else 1,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        }
    return RuntimeSettings.model_validate(value)


def _resolve(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings | None = None,
    segment_ids: list[str] | tuple[str, ...] | None = None,
    historical_takes: dict[str, NativeHistoricalTake] | None = None,
    resolved_lora_adapters: dict | None = None,
) -> V4CreativeCompileInput:
    return CreativeCompileInputResolver.resolve_v4(
        draft,
        settings or _settings(),
        segment_ids,
        historical_takes,
        resolved_lora_adapters,
    )


def test_v4_resolver_is_immutable_json_round_trip_and_does_not_mutate_inputs() -> None:
    draft = _draft(
        _segment("segment-a", "fl2va"),
        _segment("segment-b", "ref2va"),
    )
    settings = _settings()
    draft_before = deepcopy(draft.model_dump(mode="json"))
    settings_before = deepcopy(settings.model_dump(mode="json"))

    resolved = _resolve(draft, settings)

    assert draft.model_dump(mode="json") == draft_before
    assert settings.model_dump(mode="json") == settings_before
    restored = V4CreativeCompileInput.model_validate_json(
        resolved.model_dump_json()
    )
    assert restored == resolved
    assert restored.materialize_draft().model_dump(mode="json") == draft_before
    assert restored.materialize_settings().model_dump(mode="json") == settings_before
    assert [
        route.materialize_segment().model_dump(mode="json")
        for route in restored.routes
    ] == [draft_before["segments"][0], draft_before["segments"][1]]

    with pytest.raises(TypeError):
        resolved.draft_document["title"] = "mutated"  # type: ignore[index]
    with pytest.raises(ValidationError):
        resolved.routes[0].backend = "raylight"  # type: ignore[misc]

    draft.title = "source changed later"
    settings.models.fl2va.filename = "source-changed.safetensors"
    assert resolved.draft_document["title"] == draft_before["title"]
    assert (
        resolved.captured_legacy_settings["models"]["fl2va"]["filename"]
        == settings_before["models"]["fl2va"]["filename"]
    )


def test_selection_rejects_unknown_or_disabled_and_retains_timeline_order() -> None:
    draft = _draft(
        _segment("segment-a", "fl2va"),
        _segment("segment-b", "fl2va", enabled=False),
        _segment("segment-c", "fl2va"),
    )

    selected = _resolve(draft, segment_ids=["segment-c", "segment-a", "segment-c"])
    assert selected.requested_segment_ids == (
        "segment-c",
        "segment-a",
        "segment-c",
    )
    assert selected.selected_segment_ids == ("segment-a", "segment-c")
    assert [route.segment_id for route in selected.routes] == [
        "segment-a",
        "segment-c",
    ]

    with pytest.raises(CreativeCompileInputError) as disabled:
        _resolve(draft, segment_ids=["segment-b"])
    assert disabled.value.code == "segment_selection_invalid"
    assert disabled.value.rule == "segment_selection"
    assert "segment-b" not in disabled.value.public_message
    with pytest.raises(
        CreativeCompileInputError,
        match="at least one enabled timeline segment is required",
    ):
        _resolve(draft, segment_ids=[])


@pytest.mark.parametrize(
    ("segments", "settings", "expected"),
    [
        (
            (
                _segment("fl-ray", "fl2va"),
                _segment("ref-standard", "ref2va"),
            ),
            _settings(fl_backend="raylight", ref_backend="standard"),
            ("ref-standard", "fl-ray"),
        ),
        (
            (
                _segment("ref-first", "ref2va"),
                _segment("fl-second", "fl2va"),
            ),
            _settings(),
            ("fl-second", "ref-first"),
        ),
    ],
)
def test_independent_routes_keep_standard_then_raylight_and_fl_then_ref_order(
    segments: tuple[dict, ...],
    settings: RuntimeSettings,
    expected: tuple[str, ...],
) -> None:
    resolved = _resolve(_draft(*segments), settings)

    assert tuple(route.segment_id for route in resolved.routes) == expected
    assert resolved.submission_order == tuple(
        route.unit_id for route in resolved.routes
    )
    assert resolved.families == ("fl2va", "ref2va")
    for route in resolved.routes:
        assert route.unit_id == (
            f"{route.backend}-{route.family}-{route.timeline_index:03d}"
        )
        assert route.template_id == (
            "h3_standard_segment"
            if route.backend == "standard"
            else "h3_raylight_segment"
        )


def test_any_continuity_edge_forces_global_timeline_submission_order() -> None:
    draft = _draft(
        _segment("fl-ray", "fl2va", continuity=True),
        _segment("ref-standard", "ref2va", continuity=True),
    )

    resolved = _resolve(
        draft,
        _settings(fl_backend="raylight", ref_backend="standard"),
    )

    assert [route.segment_id for route in resolved.routes] == [
        "fl-ray",
        "ref-standard",
    ]
    first, second = resolved.routes
    assert first.anchor_reset is True
    assert first.predecessor_segment_id is None
    assert second.predecessor_segment_id == "fl-ray"
    assert second.continuity_source == "same_run"
    assert resolved.has_continuity_edges is True


def test_first_image_is_an_explicit_anchor_reset_and_starts_new_chain() -> None:
    resolved = _resolve(
        _draft(
            _segment("first", "fl2va"),
            _segment("reset", "fl2va", continuity=True, first_image=True),
            _segment("successor", "fl2va", continuity=True),
        )
    )

    reset = next(route for route in resolved.routes if route.segment_id == "reset")
    successor = next(
        route for route in resolved.routes if route.segment_id == "successor"
    )
    assert reset.anchor_reset is True
    assert reset.predecessor_segment_id is None
    assert reset.recipe == "i2v"
    assert successor.anchor_reset is False
    assert successor.predecessor_segment_id == "reset"


def test_partial_continuity_requires_exact_safe_historical_take() -> None:
    draft = _draft(
        _segment("predecessor", "fl2va"),
        _segment("target", "fl2va", continuity=True),
    )

    with pytest.raises(
        CreativeCompileInputError,
        match=(
            "continuity segment 'target' requires a server-resolved historical "
            "take for predecessor 'predecessor'"
        ),
    ):
        _resolve(draft, segment_ids=["target"])

    with pytest.raises(
        CreativeCompileInputError,
        match=(
            "historical take 'wrong-take' belongs to segment 'other', not "
            "current predecessor 'predecessor'"
        ),
    ):
        _resolve(
            draft,
            segment_ids=["target"],
            historical_takes={
                "target": NativeHistoricalTake(
                    id="wrong-take",
                    segment_id="other",
                    output={"filename": "take.mp4", "type": "output"},
                )
            },
        )

    take = NativeHistoricalTake(
        id="take-1",
        segment_id="predecessor",
        output={
            "filename": "take.mp4",
            "subfolder": "video/segments",
            "type": "output",
        },
    )
    resolved = _resolve(
        draft,
        segment_ids=["target"],
        historical_takes={"target": take},
    )
    route = resolved.routes[0]
    assert route.continuity_source == "historical_take"
    assert route.historical_take is not None
    assert route.historical_take.id == "take-1"
    assert (
        route.historical_take.annotated_output_path
        == "video/segments/take.mp4 [output]"
    )
    assert route.historical_take.materialize_output() == dict(take.output)

    with pytest.raises(
        CreativeCompileInputError,
        match="continuity predecessor must be a persisted ComfyUI output",
    ):
        _resolve(
            draft,
            segment_ids=["target"],
            historical_takes={
                "target": NativeHistoricalTake(
                    id="input-take",
                    segment_id="predecessor",
                    output={"filename": "take.mp4", "type": "input"},
                )
            },
        )


def test_fps_memory_and_reference_slot_policies_match_native_v4() -> None:
    wrong_fps = _draft(_segment("segment", "fl2va"))
    wrong_fps.render.fps = 30.0
    with pytest.raises(
        CreativeCompileInputError, match="render.fps must equal 24"
    ):
        _resolve(wrong_fps)

    settings = _settings().model_copy(
        update={"memory_policy": "clear_between_segments"}
    )
    with pytest.raises(
        CreativeCompileInputError,
        match="memory_policy='keep_resident'.*clear_between_segments",
    ):
        _resolve(_draft(_segment("segment", "fl2va")), settings)

    sparse = _draft(
        _segment("reference", "ref2va", reference_image_slot=2)
    )
    with pytest.raises(
        CreativeCompileInputError,
        match=(
            "segment 'reference' reference_images slots must be dense "
            "\\[0\\]; got \\[2\\]"
        ),
    ):
        _resolve(sparse)


def test_raylight_placement_is_derived_only_from_gpu_pool_and_fails_closed() -> None:
    disabled = _settings(fl_backend="raylight")
    disabled.multi_gpu_enabled = False
    with pytest.raises(
        CreativeCompileInputError, match="multi-GPU inference is disabled"
    ):
        _resolve(_draft(_segment("segment", "fl2va")), disabled)

    explicit_device = _settings(fl_backend="raylight")
    explicit_device.models.fl2va.device = "gpu:3"
    with pytest.raises(
        CreativeCompileInputError,
        match="fl2va.device must be 'default'.*authoritative logical GPU pool",
    ):
        _resolve(_draft(_segment("segment", "fl2va")), explicit_device)

    obsolete_backend = _settings()
    obsolete_backend.models.fl2va.backend = "raylight"
    resolved = _resolve(_draft(_segment("segment", "fl2va")), obsolete_backend)
    assert resolved.routes[0].backend == "standard"


@pytest.mark.parametrize(
    ("policy", "keep", "clear"),
    [
        ("keep_until_switch", True, False),
        ("release_after_sampling", False, True),
    ],
)
def test_raylight_residency_policy_is_captured_as_exact_clear_decision(
    policy: str, keep: bool, clear: bool
) -> None:
    resolved = _resolve(
        _draft(_segment("segment", "fl2va")),
        _settings(fl_backend="raylight", residency=policy),
    )
    assert resolved.keep_raylight_resident is keep
    assert resolved.clear_raylight_vram_after_sampling is clear
    assert resolved.routes[0].clear_raylight_vram_after_sampling is clear


def test_standard_lora_resolution_is_explicit_and_never_falls_back() -> None:
    value = _settings().model_dump(mode="json")
    value["models"]["fl2va"]["lora_name"] = "renamed.safetensors"
    settings = RuntimeSettings.model_validate(value)
    draft = _draft(_segment("segment", "fl2va"))

    with pytest.raises(CreativeCompileInputError) as unresolved:
        _resolve(draft, settings)
    assert unresolved.value.code == "lora_loader_mapping_required"
    assert unresolved.value.rule == "standard_lora_resolution"
    assert unresolved.value.feature_id == "lora"
    assert "renamed.safetensors" not in unresolved.value.public_message

    exact_binding = LoraLoaderBindingKey(
        family="fl2va",
        model_filename=settings.models.fl2va.filename,
        lora_filename="renamed.safetensors",
    )
    mapped = resolve_standard_lora_adapter(
        exact_binding,
        (
            LoraLoaderOverrideRecord(
                family="fl2va",
                model_filename=exact_binding.model_filename,
                lora_filename=exact_binding.lora_filename,
                adapter_id="model_only",
            ),
        ),
    )
    resolved = _resolve(
        draft,
        settings,
        resolved_lora_adapters={"fl2va": mapped},
    )
    assert resolved.routes[0].lora_resolution is not None
    assert resolved.routes[0].lora_resolution.loader_node == (
        "LoraLoaderModelOnly"
    )
    assert resolved.routes[0].lora_resolution.source == "user_override"

    value["models"]["fl2va"]["standard_lora_loader_override"] = {
        "loader": "bypass_model_only",
        "lora_name": "renamed.safetensors",
        "model_filename": value["models"]["fl2va"]["filename"],
    }
    explicit = _resolve(
        draft,
        RuntimeSettings.model_validate(value),
    )
    assert explicit.routes[0].lora_resolution is not None
    assert (
        explicit.routes[0].lora_resolution.loader_node
        == "LoraLoaderBypassModelOnly"
    )
    assert explicit.routes[0].lora_resolution.source == "user_override"


def test_raylight_lora_uses_fixed_loader_without_standard_metadata() -> None:
    value = _settings(fl_backend="raylight").model_dump(mode="json")
    value["models"]["fl2va"]["lora_name"] = "unknown-ray-lora.safetensors"

    resolved = _resolve(
        _draft(_segment("segment", "fl2va")),
        RuntimeSettings.model_validate(value),
    )

    assert resolved.routes[0].lora_resolution is not None
    assert resolved.routes[0].lora_resolution.loader_node == "DirectorDeckRayLoraLoader"
    assert resolved.routes[0].lora_resolution.source == "backend_fixed"


@pytest.mark.parametrize("strength", [0.0, -0.0])
@pytest.mark.parametrize("backend", ["standard", "raylight"])
def test_active_lora_rejects_positive_and_negative_zero_strength(
    backend: str,
    strength: float,
) -> None:
    settings = _settings(fl_backend=backend)
    value = settings.model_dump(mode="json")
    value["models"]["fl2va"].update(
        lora_name="minimax_h3_turbo_v4_step600.safetensors",
        lora_strength=strength,
    )

    with pytest.raises(
        CreativeCompileInputError,
        match="LoRA is enabled with zero strength",
    ):
        _resolve(
            _draft(_segment("segment", "fl2va")),
            RuntimeSettings.model_validate(value),
        )


def test_disabled_lora_may_preserve_legacy_zero_strength_without_emission() -> None:
    settings = _settings()
    value = settings.model_dump(mode="json")
    value["models"]["fl2va"].update(lora_name=None, lora_strength=0.0)

    resolved = _resolve(
        _draft(_segment("segment", "fl2va")),
        RuntimeSettings.model_validate(value),
    )

    assert resolved.routes[0].lora_resolution is None
