from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

import director.native_templates as native_templates_module
from director.native_templates import (
    EXPECTED_NATIVE_NODE_MODULES,
    NativeCompileResult,
    NativeHistoricalTake,
    NativeTemplateError,
    bind_native_workflow_predecessor_output,
    bind_raylight_runtime_epoch,
    build_raylight_shutdown_unit,
    compile_native_timeline,
    raylight_runtime_descriptor,
    raylight_runtime_logical_gpu_indices,
    raylight_workflow_logical_gpu_indices,
    validate_native_capabilities,
    validate_native_workflow_ready,
)
from director.schemas import RuntimeSettings, UnifiedTimelineDraft, default_settings


def _asset(kind: str, name: str, *, slot: int | None = None) -> dict:
    value = {
        "name": name,
        "subfolder": "director-web",
        "type": "input",
        "kind": kind,
        "id": f"asset-{name}",
        "content_hash": "sha256:" + "a" * 64,
    }
    if kind == "video":
        value["metadata"] = {
            "duration": 12.0,
            "native_fps": 24.0,
            "frame_count": 288,
            "width": 1920,
            "height": 1080,
            "probe_method": "test",
            "has_audio": True,
        }
    if slot is not None:
        value["slot"] = slot
    return value


def _segment(mode: str) -> dict:
    common = {
        "id": f"segment-{mode}",
        "title": mode,
        "mode": mode,
        "prompt": f"generate {mode}",
        "duration_seconds": 5.0,
        "enabled": True,
    }
    if mode == "i2v":
        common["first_image"] = _asset("image", "first.png")
    elif mode == "fl2v":
        common["first_image"] = _asset("image", "first.png")
        common["last_image"] = _asset("image", "last.png")
    elif mode == "r2v":
        common["prompt"] = "Use <Picture 1>, <Video 1>, and <Audio 1>."
        common["reference_images"] = [_asset("image", "ref.png", slot=0)]
        common["reference_videos"] = [_asset("video", "ref.mp4", slot=0)]
        common["reference_audios"] = [_asset("audio", "ref.wav", slot=0)]
    elif mode == "v2v":
        common["prompt"] = "Edit <Video 1>."
        common["source_video"] = _asset("video", "source.mp4")
        common["source_start_seconds"] = 1.0
        common["source_duration_seconds"] = 5.0
    elif mode == "rv2v":
        common["prompt"] = "Edit <Video 1> using <Picture 1> and <Audio 1>."
        common["source_video"] = _asset("video", "source.mp4")
        common["source_start_seconds"] = 1.0
        common["source_duration_seconds"] = 5.0
        common["reference_images"] = [_asset("image", "ref.png", slot=0)]
        common["reference_audios"] = [_asset("audio", "ref.wav", slot=0)]
    return common


def _draft(*modes: str, audio_mode: str = "generate") -> UnifiedTimelineDraft:
    return UnifiedTimelineDraft.model_validate(
        {
            "version": 1,
            "title": "native test",
            "prompt": "",
            "ref_image_size": "match",
            "render": {"width": 864, "height": 480, "fps": 24.0},
            "sampling": {
                "steps": 20,
                "cfg": 1.0,
                "seed": 1234,
                "sampler": "res_multistep",
                "scheduler": "simple",
                "shift": 12.0,
                "audio_shift": 3.0,
            },
            "continuity": {"enabled": False, "overlap_frames": 22},
            "audio_mode": audio_mode,
            "export_mode": "all",
            "segments": [_segment(mode) for mode in modes],
        }
    )


def _settings(
    *, fl_backend: str = "standard", ref_backend: str = "standard"
) -> RuntimeSettings:
    value = default_settings("http://comfy.test:8188").model_dump(mode="json")
    for family, backend in (("fl2va", fl_backend), ("ref2va", ref_backend)):
        raylight = backend == "raylight"
        value["models"][family]["backend"] = backend
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


def _continuity_draft(
    *modes: str,
    audio_mode: str = "generate",
    overlap_frames: int = 22,
) -> UnifiedTimelineDraft:
    value = _draft(modes[0], audio_mode=audio_mode).model_dump(mode="json")
    value["segments"] = [
        _draft(mode, audio_mode=audio_mode).segments[0].model_dump(mode="json")
        for mode in modes
    ]
    for index, segment in enumerate(value["segments"]):
        segment["id"] = f"segment-{index + 1}"
        segment["title"] = f"Segment {index + 1}"
        segment["continuity"] = {
            "enabled": True,
            "overlap_frames": overlap_frames,
        }
    return UnifiedTimelineDraft.model_validate(value)


def _types(prompt: dict) -> set[str]:
    return {node["class_type"] for node in prompt.values()}


@pytest.mark.parametrize("mode", ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"])
def test_standard_templates_cover_all_six_modes_without_director(mode: str) -> None:
    result = compile_native_timeline(
        _draft(mode), _settings(), f"job-standard-{mode}"
    )

    assert len(result.workflows) == 1
    unit = result.workflows[0]
    types = _types(unit.prompt)
    assert unit.backend == "standard"
    assert "MiniMaxH3Director" not in types
    assert {"UNETLoader", "SelectModelDevice", "MiniMaxH3SigmaShift"} <= types
    assert {
        "BasicGuider",
        "BasicScheduler",
        "KSamplerSelect",
        "RandomNoise",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "CreateVideo",
        "SaveVideo",
    } <= types
    conditioning = (
        "MiniMaxH3ImageToVideo"
        if mode in {"t2v", "i2v", "fl2v"}
        else "MiniMaxH3ReferenceToVideo"
    )
    assert conditioning in types
    assert result.plans[0]["mode"] == (
        "fl2va" if mode in {"t2v", "i2v", "fl2v"} else "ref2va"
    )
    assert result.plans[0]["recipe"] == mode
    if mode in {"v2v", "rv2v"}:
        assert {"LoadVideo", "Video Slice", "GetVideoComponents"} <= types


def test_reference_video_shorter_than_five_frames_is_rejected() -> None:
    value = _draft("r2v").model_dump(mode="json")
    video = value["segments"][0]["reference_videos"][0]
    video["metadata"].update(duration=4 / 24, frame_count=4)
    draft = UnifiedTimelineDraft.model_validate(value)

    with pytest.raises(NativeTemplateError, match="at least 5 frames"):
        compile_native_timeline(draft, _settings(), "job-short-reference")


def test_fl2va_end_only_anchor_derives_fl2v_and_connects_only_last_frame() -> None:
    value = _draft("t2v").model_dump(mode="json")
    value["segments"][0].update(
        mode="fl2va",
        first_image=None,
        last_image=_asset("image", "last.png"),
    )
    result = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(value), _settings(), "job-end-only"
    )
    prompt = result.workflows[0].prompt
    conditioning = next(
        node for node in prompt.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    )["inputs"]

    assert result.plans[0]["mode"] == "fl2va"
    assert result.plans[0]["recipe"] == "fl2v"
    assert "first_frame" not in conditioning
    assert "last_frame" in conditioning


@pytest.mark.parametrize("mode", ["v2v", "rv2v"])
def test_source_trim_shorter_than_five_frames_is_rejected(mode: str) -> None:
    value = _draft(mode).model_dump(mode="json")
    value["segments"][0]["source_duration_seconds"] = 4 / 24
    draft = UnifiedTimelineDraft.model_validate(value)

    with pytest.raises(NativeTemplateError, match="selects 4 frame"):
        compile_native_timeline(draft, _settings(), f"job-short-{mode}")


@pytest.mark.parametrize("mode", ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"])
def test_raylight_substitutes_only_model_and_sampler_path(mode: str) -> None:
    family = "fl2va" if mode in {"t2v", "i2v", "fl2v"} else "ref2va"
    settings = _settings(
        fl_backend="raylight" if family == "fl2va" else "standard",
        ref_backend="raylight" if family == "ref2va" else "standard",
    )
    result = compile_native_timeline(_draft(mode), settings, f"job-ray-{mode}")

    unit = result.workflows[0]
    types = _types(unit.prompt)
    assert unit.backend == "raylight"
    assert {
        "RayInitializerAdvanced",
        "RayUNETLoader",
        "RayMiniMaxH3SigmaShift",
        "RayBasicGuider",
        "RayBasicScheduler",
        "XFuserSamplerCustomAdvanced",
    } <= types
    assert {
        "UNETLoader",
        "SelectModelDevice",
        "MiniMaxH3SigmaShift",
        "BasicGuider",
        "BasicScheduler",
        "RandomNoise",
        "SamplerCustomAdvanced",
        "MiniMaxH3Director",
    }.isdisjoint(types)
    assert {"VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo"} <= types
    initializer = next(
        node for node in unit.prompt.values()
        if node["class_type"] == "RayInitializerAdvanced"
    )["inputs"]
    assert initializer["GPU"] == 2
    assert initializer["GPU_SELECT"] == "0,1"
    assert initializer["driver_cleanup_policy"] == "ray_devices"
    assert initializer["XFuser_attention"] == "COMFY_KITCHEN_INT8"
    assert initializer["ulysses_degree"] == 2
    assert initializer["FSDP"] is False
    assert initializer["clear_vram_after_sampling"] is False


@pytest.mark.parametrize(
    ("mode", "family", "backend", "scheduler_node"),
    [
        ("t2v", "fl2va", "standard", "BasicScheduler"),
        ("r2v", "ref2va", "standard", "BasicScheduler"),
        ("t2v", "fl2va", "raylight", "RayBasicScheduler"),
        ("r2v", "ref2va", "raylight", "RayBasicScheduler"),
    ],
)
def test_beta_scheduler_reaches_native_standard_and_raylight_prompts(
    mode: str,
    family: str,
    backend: str,
    scheduler_node: str,
) -> None:
    raw_draft = _draft(mode).model_dump(mode="json")
    raw_draft["sampling"][family]["scheduler"] = "beta"
    draft = UnifiedTimelineDraft.model_validate(raw_draft)
    settings = _settings(
        fl_backend=backend if family == "fl2va" else "standard",
        ref_backend=backend if family == "ref2va" else "standard",
    )

    result = compile_native_timeline(
        draft, settings, f"job-beta-{backend}-{family}"
    )
    scheduler = next(
        node
        for node in result.workflows[0].prompt.values()
        if node["class_type"] == scheduler_node
    )

    assert scheduler["inputs"]["scheduler"] == "beta"


def test_keyed_raylight_policy_keeps_compatible_workflows_resident() -> None:
    raw = _settings(
        fl_backend="raylight", ref_backend="raylight"
    ).model_dump(mode="json")
    raw["raylight_residency_policy"] = "keep_until_switch"
    settings = RuntimeSettings.model_validate(raw)
    result = compile_native_timeline(
        _draft("t2v", "i2v"), settings, "job-ray-resident"
    )

    initializers = [
        node["inputs"]
        for unit in result.workflows
        for node in unit.prompt.values()
        if node["class_type"] == "RayInitializerAdvanced"
    ]
    assert len(initializers) == 2
    assert all(
        initializer["clear_vram_after_sampling"] is False
        for initializer in initializers
    )
    assert all(
        initializer["driver_cleanup_policy"] == "ray_devices"
        for initializer in initializers
    )
    assert result.manifest["resident_cache_scope"]["raylight_cuda_residency"] == (
        "kept_for_compatible_key_until_explicit_switch"
    )
    assert result.manifest["resident_cache_scope"]["raylight_residency_reason"] == (
        "explicit_keyed_switch_policy"
    )
    assert result.manifest["resident_cache_scope"]["raylight_resident_family"] is None


def test_shared_endpoint_default_releases_raylight_worker_model() -> None:
    settings = _settings(fl_backend="standard", ref_backend="raylight")
    raw = settings.model_dump(mode="json")
    raw["raylight_residency_policy"] = "release_after_sampling"
    result = compile_native_timeline(
        _draft("r2v"),
        RuntimeSettings.model_validate(raw),
        "job-ray-shared-endpoint",
    )
    initializer = next(
        node["inputs"]
        for node in result.workflows[0].prompt.values()
        if node["class_type"] == "RayInitializerAdvanced"
    )

    assert initializer["clear_vram_after_sampling"] is True
    assert result.manifest["resident_cache_scope"]["raylight_cuda_residency"] == (
        "released_after_each_sampler"
    )
    assert result.manifest["resident_cache_scope"]["raylight_residency_reason"] == (
        "shared_endpoint_safe_default"
    )


def test_keyed_raylight_residency_accepts_both_families() -> None:
    settings = _settings(fl_backend="raylight", ref_backend="raylight")
    result = compile_native_timeline(
        _draft("t2v", "r2v"), settings, "job-keyed-two-families"
    )

    assert [(unit.family, unit.backend) for unit in result.workflows] == [
        ("fl2va", "raylight"),
        ("ref2va", "raylight"),
    ]


def test_raylight_epoch_changes_initializer_and_shutdown_barrier_is_forced() -> None:
    unit = compile_native_timeline(
        _draft("t2v"), _settings(fl_backend="raylight"), "job-epoch"
    ).workflows[0]
    first = bind_raylight_runtime_epoch(unit, 1)
    repeated = bind_raylight_runtime_epoch(unit, 1)
    switched_back = bind_raylight_runtime_epoch(unit, 3)

    first_descriptor = raylight_runtime_descriptor(first)
    repeated_descriptor = raylight_runtime_descriptor(repeated)
    switched_descriptor = raylight_runtime_descriptor(switched_back)
    assert first_descriptor is not None
    assert repeated_descriptor is not None
    assert switched_descriptor is not None
    assert first_descriptor["runtime_namespace"] == repeated_descriptor["runtime_namespace"]
    assert switched_descriptor["runtime_namespace"] != first_descriptor["runtime_namespace"]
    assert switched_descriptor["compatibility_key"] == first_descriptor["compatibility_key"]
    assert switched_descriptor["runtime_key"] == first_descriptor["runtime_key"]

    barrier = build_raylight_shutdown_unit(
        first_descriptor, unit_id="switch-test"
    )
    assert _types(barrier.prompt) == {
        "RayInitializerAdvanced",
        "RayUNETLoader",
        "RayKill",
    }
    barrier_initializer = next(
        node["inputs"]
        for node in barrier.prompt.values()
        if node["class_type"] == "RayInitializerAdvanced"
    )
    assert barrier_initializer["driver_cleanup_policy"] == "ray_devices"
    loader_id = next(
        node_id for node_id, node in barrier.prompt.items()
        if node["class_type"] == "RayUNETLoader"
    )
    kill = next(
        node for node in barrier.prompt.values()
        if node["class_type"] == "RayKill"
    )
    assert kill["inputs"]["ray_actors"] == [loader_id, 0]
    assert kill["inputs"]["kill_mode"] == "Kill Entire Cluster"


def test_raylight_runtime_gpu_identity_is_strict_and_lossless() -> None:
    unit = compile_native_timeline(
        _draft("t2v"), _settings(fl_backend="raylight"), "job-gpu-identity"
    ).workflows[0]
    descriptor = raylight_runtime_descriptor(unit)
    assert descriptor is not None
    assert raylight_runtime_logical_gpu_indices(descriptor) == (0, 1)
    assert raylight_workflow_logical_gpu_indices(unit) == (0, 1)

    initializer_id = descriptor["initializer_node_id"]
    inputs = descriptor["loader_subgraph"][initializer_id]["inputs"]
    inputs["GPU_SELECT"] = "0,0"
    with pytest.raises(NativeTemplateError, match="duplicates"):
        raylight_runtime_logical_gpu_indices(descriptor)

    inputs["GPU_SELECT"] = "0,one"
    with pytest.raises(NativeTemplateError, match="GPU_SELECT is invalid"):
        raylight_runtime_logical_gpu_indices(descriptor)

    inputs["GPU_SELECT"] = "0,1"
    inputs["GPU"] = 4
    with pytest.raises(NativeTemplateError, match="GPU count"):
        raylight_runtime_logical_gpu_indices(descriptor)


def test_raylight_runtime_key_includes_mutating_sigma_shift_inputs() -> None:
    settings = _settings(fl_backend="raylight")
    base_raw = _draft("t2v").model_dump(mode="json")
    changed_raw = deepcopy(base_raw)
    changed_raw["sampling"]["fl2va"]["shift"] = 8.0

    base = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(base_raw), settings, "job-shift-12"
    ).workflows[0]
    changed = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(changed_raw), settings, "job-shift-8"
    ).workflows[0]
    base_descriptor = raylight_runtime_descriptor(base)
    changed_descriptor = raylight_runtime_descriptor(changed)

    assert base_descriptor is not None
    assert changed_descriptor is not None
    assert base_descriptor["compatibility_key"] == changed_descriptor["compatibility_key"]
    assert base_descriptor["runtime_key"] != changed_descriptor["runtime_key"]


def test_raylight_runtime_key_includes_driver_cleanup_policy() -> None:
    unit = compile_native_timeline(
        _draft("t2v"), _settings(fl_backend="raylight"), "job-cleanup-policy"
    ).workflows[0]
    legacy_unit = deepcopy(unit)
    legacy_initializer = next(
        node
        for node in legacy_unit.prompt.values()
        if node["class_type"] == "RayInitializerAdvanced"
    )
    legacy_initializer["inputs"]["driver_cleanup_policy"] = "legacy_all"

    scoped_descriptor = raylight_runtime_descriptor(unit)
    legacy_descriptor = raylight_runtime_descriptor(legacy_unit)

    assert scoped_descriptor is not None
    assert legacy_descriptor is not None
    assert scoped_descriptor["runtime_key"] != legacy_descriptor["runtime_key"]
    initializer_id = scoped_descriptor["initializer_node_id"]
    assert scoped_descriptor["loader_subgraph"][initializer_id]["inputs"][
        "driver_cleanup_policy"
    ] == "ray_devices"


def test_raylight_attention_changes_namespace_and_runtime_identity_and_barrier_replays_old_value(
    monkeypatch,
) -> None:
    settings = _settings(fl_backend="raylight")
    ck_unit = compile_native_timeline(
        _draft("t2v"), settings, "job-attention-ck"
    ).workflows[0]
    ck_descriptor = raylight_runtime_descriptor(ck_unit)
    assert ck_descriptor is not None

    with monkeypatch.context() as context:
        context.setattr(
            native_templates_module,
            "_RAYLIGHT_XFUSER_ATTENTION",
            "TORCH_FLASH",
        )
        legacy_unit = compile_native_timeline(
            _draft("t2v"), settings, "job-attention-torch-flash"
        ).workflows[0]
    legacy_descriptor = raylight_runtime_descriptor(legacy_unit)
    assert legacy_descriptor is not None

    assert ck_descriptor["compatibility_key"] != legacy_descriptor["compatibility_key"]
    assert ck_descriptor["runtime_key"] != legacy_descriptor["runtime_key"]
    legacy_initializer_id = legacy_descriptor["initializer_node_id"]
    assert legacy_descriptor["loader_subgraph"][legacy_initializer_id]["inputs"][
        "XFuser_attention"
    ] == "TORCH_FLASH"

    barrier = build_raylight_shutdown_unit(
        legacy_descriptor, unit_id="switch-legacy-attention"
    )
    barrier_initializer = next(
        node
        for node in barrier.prompt.values()
        if node["class_type"] == "RayInitializerAdvanced"
    )
    assert barrier_initializer["inputs"]["XFuser_attention"] == "TORCH_FLASH"


def test_raylight_shutdown_barrier_preserves_optional_lora_loader_chain() -> None:
    raw = _settings(fl_backend="raylight").model_dump(mode="json")
    raw["models"]["fl2va"].update(
        lora_name="style.safetensors",
        lora_loader="auto",
    )
    unit = compile_native_timeline(
        _draft("t2v"), RuntimeSettings.model_validate(raw), "job-lora-barrier"
    ).workflows[0]
    bound = bind_raylight_runtime_epoch(unit, 4)
    descriptor = raylight_runtime_descriptor(bound)
    assert descriptor is not None

    barrier = build_raylight_shutdown_unit(descriptor, unit_id="switch-lora")
    assert _types(barrier.prompt) == {
        "RayInitializerAdvanced",
        "RayLoraLoader",
        "RayUNETLoader",
        "RayKill",
    }
    loader = next(
        node for node in barrier.prompt.values()
        if node["class_type"] == "RayUNETLoader"
    )
    lora_id = next(
        node_id for node_id, node in barrier.prompt.items()
        if node["class_type"] == "RayLoraLoader"
    )
    assert loader["inputs"]["lora"] == [lora_id, 0]


def test_mixed_timeline_is_split_into_family_units_and_redacted_manifest() -> None:
    result = compile_native_timeline(
        _draft("t2v", "r2v"),
        _settings(fl_backend="standard", ref_backend="raylight"),
        "job-mixed",
    )

    assert [(unit.family, unit.backend) for unit in result.workflows] == [
        ("fl2va", "standard"),
        ("ref2va", "raylight"),
    ]
    assert result.manifest["submission_order"] == [
        "standard-fl2va-000",
        "raylight-ref2va-001",
    ]
    assert result.manifest["raylight_exclusive"] is True
    assert "prompt" not in result.manifest
    assert all("prompt" not in plan for plan in result.plans)


def test_standard_unit_is_always_submitted_before_a_cleaning_raylight_unit() -> None:
    result = compile_native_timeline(
        _draft("t2v", "r2v"),
        _settings(fl_backend="raylight", ref_backend="standard"),
        "job-order",
    )

    assert [(unit.family, unit.backend) for unit in result.workflows] == [
        ("ref2va", "standard"),
        ("fl2va", "raylight"),
    ]


def test_autogrow_slots_preserve_dense_reference_ordinals() -> None:
    result = compile_native_timeline(_draft("r2v"), _settings(), "job-slots")
    prompt = result.workflows[0].prompt
    conditioning = next(
        node for node in prompt.values()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )["inputs"]

    assert conditioning["prompt"] == "Use <Picture 1>, <Video 1>, and <Audio 1>."
    assert "ref_images.ref_image_0" in conditioning
    assert "ref_videos.ref_video_0" in conditioning
    assert "ref_audios.ref_audio_0" in conditioning


def test_h3_reference_capacity_compiles_every_highest_stock_slot() -> None:
    value = _draft("v2v").model_dump(mode="json")
    segment = value["segments"][0]
    segment.update(
        prompt=(
            "Use <Picture 9>, <Video 3>, source <Audio 1>, and "
            "independent <Audio 4>."
        ),
        source_audio_as_reference=True,
        reference_images=[
            _asset("image", f"reference-{index}.png", slot=index)
            for index in range(9)
        ],
        reference_videos=[
            _asset("video", f"motion-{index}.mp4", slot=index)
            for index in range(2)
        ],
        reference_audios=[
            _asset("audio", f"voice-{index}.wav", slot=index)
            for index in range(3)
        ],
    )

    result = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(value),
        _settings(),
        "job-max-reference-slots",
    )
    prompt = result.workflows[0].prompt
    conditioning = next(
        node
        for node in prompt.values()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )["inputs"]

    assert "ref_images.ref_image_8" in conditioning
    assert "ref_videos.ref_video_2" in conditioning
    assert "ref_audios.ref_audio_2" in conditioning
    assert "ref_video_audios.ref_video_audio_0" in conditioning
    assert "<Audio 4>" in conditioning["prompt"]


def test_ref2va_source_and_independent_videos_compile_to_distinct_stock_slots() -> None:
    value = _draft("v2v").model_dump(mode="json")
    segment = value["segments"][0]
    segment["reference_videos"] = [
        _asset("video", "motion-a.mp4", slot=0),
        _asset("video", "motion-b.mp4", slot=1),
    ]
    segment["prompt"] = "Edit <Video 1> using <Video 2> and <Video 3>."

    result = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(value), _settings(), "job-source-and-refs"
    )
    prompt = result.workflows[0].prompt
    conditioning = next(
        node for node in prompt.values()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )["inputs"]

    assert result.plans[0]["mode"] == "ref2va"
    assert result.plans[0]["recipe"] == "rv2v"
    assert {
        "ref_videos.ref_video_0",
        "ref_videos.ref_video_1",
        "ref_videos.ref_video_2",
    } <= set(conditioning)
    source_component = conditioning["ref_videos.ref_video_0"][0]
    assert prompt[source_component]["class_type"] == "GetVideoComponents"
    source_slice = prompt[prompt[source_component]["inputs"]["video"][0]]
    assert source_slice["class_type"] == "Video Slice"
    for slot in (1, 2):
        component = conditioning[f"ref_videos.ref_video_{slot}"][0]
        assert prompt[component]["class_type"] == "GetVideoComponents"
        load = prompt[prompt[component]["inputs"]["video"][0]]
        assert load["class_type"] == "LoadVideo"


def test_sparse_autogrow_slots_fail_closed_instead_of_rewriting_prompt() -> None:
    value = _draft("r2v").model_dump(mode="json")
    value["segments"][0]["reference_images"][0]["slot"] = 2
    value["segments"][0]["prompt"] = "Use <Picture 3>."

    with pytest.raises(NativeTemplateError, match="slots must be dense"):
        compile_native_timeline(
            UnifiedTimelineDraft.model_validate(value), _settings(), "job-sparse"
        )


def test_source_audio_uses_trimmed_stock_video_audio_without_audio_decode() -> None:
    result = compile_native_timeline(
        _draft("v2v", audio_mode="source"), _settings(), "job-source-audio"
    )
    prompt = result.workflows[0].prompt
    types = _types(prompt)
    create = next(
        node for node in prompt.values() if node["class_type"] == "CreateVideo"
    )

    assert "VAEDecodeAudio" not in types
    assert "audio" in create["inputs"]
    components_id = create["inputs"]["audio"][0]
    assert prompt[components_id]["class_type"] == "GetVideoComponents"
    assert create["inputs"]["audio"][1] == 1


def test_media_policies_are_compiled_per_segment() -> None:
    draft = _draft("r2v", "rv2v")
    draft.segments[0].ref_image_size = "max"
    draft.segments[0].audio_mode = "mute"
    draft.segments[1].ref_image_size = "match"
    draft.segments[1].audio_mode = "generate"

    result = compile_native_timeline(draft, _settings(), "job-mixed-policies")
    first, second = (unit.prompt for unit in result.workflows)
    first_conditioning = next(
        node for node in first.values()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )
    second_conditioning = next(
        node for node in second.values()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )

    assert first_conditioning["inputs"]["ref_image_size"] == "max"
    assert second_conditioning["inputs"]["ref_image_size"] == "match"
    assert "VAEDecodeAudio" not in _types(first)
    assert "VAEDecodeAudio" in _types(second)


@pytest.mark.parametrize("mode", ["v2v", "rv2v"])
def test_source_soundtrack_conditioning_uses_stock_paired_audio_edge(mode: str) -> None:
    value = _draft(mode, audio_mode="generate").model_dump(mode="json")
    value["segments"][0]["source_audio_as_reference"] = True
    if mode == "rv2v":
        value["segments"][0]["prompt"] = (
            "Edit <Video 1> with source <Audio 1> and voice <Audio 2>."
        )
    else:
        value["segments"][0]["prompt"] = "Edit <Video 1> with <Audio 1>."
    result = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(value),
        _settings(),
        f"job-paired-source-{mode}",
    )
    prompt = result.workflows[0].prompt
    conditioning = next(
        node
        for node in prompt.values()
        if node["class_type"] == "MiniMaxH3ReferenceToVideo"
    )["inputs"]
    paired = conditioning["ref_video_audios.ref_video_audio_0"]

    assert conditioning["ref_videos.ref_video_0"][0] == paired[0]
    assert conditioning["ref_videos.ref_video_0"][1] == 0
    assert paired[1] == 1
    assert prompt[paired[0]]["class_type"] == "GetVideoComponents"
    if mode == "rv2v":
        assert "ref_audios.ref_audio_0" in conditioning
    create = next(
        node for node in prompt.values() if node["class_type"] == "CreateVideo"
    )
    # Conditioning is independent of the final output soundtrack policy:
    # generate still decodes H3's audio latent for CreateVideo.
    assert prompt[create["inputs"]["audio"][0]]["class_type"] == "VAEDecodeAudio"


def test_source_soundtrack_conditioning_fails_closed_for_silent_source() -> None:
    value = _draft("v2v").model_dump(mode="json")
    value["segments"][0]["source_audio_as_reference"] = True
    value["segments"][0]["source_video"]["metadata"]["has_audio"] = False
    draft = UnifiedTimelineDraft.model_validate(value)

    with pytest.raises(NativeTemplateError, match="has no audio stream"):
        compile_native_timeline(draft, _settings(), "job-silent-source")

    value = _draft("v2v", audio_mode="source").model_dump(mode="json")
    value["segments"][0]["source_video"]["metadata"]["has_audio"] = False
    with pytest.raises(NativeTemplateError, match="audio_mode='source'.*no audio stream"):
        compile_native_timeline(
            UnifiedTimelineDraft.model_validate(value),
            _settings(),
            "job-silent-source-output",
        )


def test_capability_preflight_reports_missing_fixed_node() -> None:
    result = compile_native_timeline(
        _draft("t2v"),
        _settings(fl_backend="raylight"),
        "job-capability",
    )
    available = set(result.node_policy["allowed_nodes"])
    available.remove("RayUNETLoader")

    with pytest.raises(NativeTemplateError, match="RayUNETLoader"):
        validate_native_capabilities(result, available)


def test_gpu_pool_is_the_only_backend_route_and_hidden_lora_flags_are_ignored() -> None:
    raw = _settings(fl_backend="raylight").model_dump(mode="json")
    raw["models"]["fl2va"]["device"] = "gpu:0"
    settings = RuntimeSettings.model_validate(raw)
    with pytest.raises(NativeTemplateError, match="gpu_select is the authoritative"):
        compile_native_timeline(_draft("t2v"), settings, "job-device")

    raw = deepcopy(_settings(fl_backend="raylight").model_dump(mode="json"))
    raw["models"]["fl2va"]["raylight"].update(
        gpu_select=[0], ulysses_degree=1
    )
    one_gpu = RuntimeSettings.model_validate(raw)
    one_gpu_result = compile_native_timeline(
        _draft("t2v"), one_gpu, "job-one-gpu"
    )
    assert one_gpu.models.fl2va.backend == "raylight"  # obsolete wire value
    assert one_gpu_result.workflows[0].backend == "standard"

    raw = deepcopy(_settings(fl_backend="raylight").model_dump(mode="json"))
    raw["models"]["fl2va"]["raylight"]["cpu_offload"] = True
    with pytest.raises(
        (ValidationError, NativeTemplateError), match="False|disabled"
    ):
        ignored_offload = RuntimeSettings.model_validate(raw)
        compile_native_timeline(
            _draft("t2v"), ignored_offload, "job-invalid-offload"
        )

    raw = deepcopy(_settings(fl_backend="raylight").model_dump(mode="json"))
    raw["models"]["fl2va"]["raylight"]["fsdp"] = True
    with pytest.raises((ValidationError, NativeTemplateError), match="False|disabled"):
        RuntimeSettings.model_validate(raw)

    raw = deepcopy(_settings(fl_backend="raylight").model_dump(mode="json"))
    raw["models"]["fl2va"].update(
        lora_name="minimax_h3_turbo_v4_step600.safetensors",
        lora_loader="dedicated",
        lora_low_vram=True,
    )
    unsupported_lora_option = RuntimeSettings.model_validate(raw)
    automatic_lora = compile_native_timeline(
        _draft("t2v"), unsupported_lora_option, "job-ray-lora-low-vram"
    )
    assert "RayLoraLoader" in _types(automatic_lora.workflows[0].prompt)


def test_auto_backend_and_explicit_standard_device_are_fail_closed() -> None:
    one_gpu = default_settings("http://comfy.test:8188")
    assert one_gpu.models.fl2va.backend == "auto"
    assert one_gpu.models.fl2va.raylight.fsdp is False
    standard = compile_native_timeline(_draft("t2v"), one_gpu, "job-auto-standard")
    assert standard.workflows[0].backend == "standard"

    raw = one_gpu.model_dump(mode="json")
    raw["models"]["fl2va"]["device"] = "gpu:3"
    explicit_device = compile_native_timeline(
        _draft("t2v"),
        RuntimeSettings.model_validate(raw),
        "job-standard-gpu",
    )
    selector = next(
        node
        for node in explicit_device.workflows[0].prompt.values()
        if node["class_type"] == "SelectModelDevice"
    )
    assert selector["inputs"]["device"] == "gpu:3"

    raw = one_gpu.model_dump(mode="json")
    raw["models"]["fl2va"]["raylight"].update(
        gpu_select=[0, 1], ulysses_degree=2
    )
    raylight = compile_native_timeline(
        _draft("t2v"), RuntimeSettings.model_validate(raw), "job-auto-raylight"
    )
    assert raylight.workflows[0].backend == "raylight"

    available = set(raylight.node_policy["allowed_nodes"])
    available.remove("RayUNETLoader")
    with pytest.raises(NativeTemplateError, match="RayUNETLoader"):
        validate_native_capabilities(raylight, available)


def test_runtime_capability_check_rejects_same_name_custom_node_override() -> None:
    result = compile_native_timeline(_draft("t2v"), _settings(), "job-provenance")
    provenance = {
        node: EXPECTED_NATIVE_NODE_MODULES[node]
        for node in result.node_policy["provenance"]
    }
    validate_native_capabilities(
        result, set(result.node_policy["allowed_nodes"]), provenance
    )

    provenance["UNETLoader"] = "custom_nodes.lookalike"
    with pytest.raises(NativeTemplateError, match="UNETLoader.*comfy-core"):
        validate_native_capabilities(
            result, set(result.node_policy["allowed_nodes"]), provenance
        )


def test_raylight_namespace_is_stable_per_topology_not_per_job() -> None:
    settings = _settings(fl_backend="raylight")
    first = compile_native_timeline(_draft("t2v"), settings, "job-first")
    second = compile_native_timeline(_draft("t2v"), settings, "job-second")

    def namespace(result: NativeCompileResult) -> str:
        unit = result.workflows[0]
        initializer = next(
            node
            for node in unit.prompt.values()
            if node["class_type"] == "RayInitializerAdvanced"
        )
        return initializer["inputs"]["ray_cluster_namespace"]

    assert namespace(first) == namespace(second)

    raw = settings.model_dump(mode="json")
    raw["models"]["fl2va"]["raylight"].update(
        gpu_select=[0, 1, 2, 3], ulysses_degree=4
    )
    changed = compile_native_timeline(
        _draft("t2v"), RuntimeSettings.model_validate(raw), "job-third"
    )
    assert namespace(changed) != namespace(first)
    assert first.manifest["resident_cache_scope"]["boundary"] == "comfy_endpoint"


def test_continuity_builds_unresolved_predecessor_graph_and_binds_output() -> None:
    result = compile_native_timeline(
        _continuity_draft("t2v", "t2v"),
        _settings(),
        "job-continuity",
    )

    first, successor = result.workflows
    assert [unit.segment_ids for unit in result.workflows] == [
        ("segment-1",),
        ("segment-2",),
    ]
    assert first.continuity is None
    validate_native_workflow_ready(first)
    assert successor.continuity is not None
    assert successor.continuity.predecessor_segment_id == "segment-1"
    assert successor.continuity.overlap_frames == 22
    assert successor.continuity.resolved is False
    with pytest.raises(NativeTemplateError, match="waiting for predecessor"):
        validate_native_workflow_ready(successor)

    prompt = successor.prompt
    dependency_loader = prompt[successor.continuity.load_video_node_id]
    assert dependency_loader["class_type"] == "LoadVideo"
    assert "UNBOUND_PREDECESSOR_OUTPUT" in dependency_loader["inputs"]["file"]
    image_slices = [
        node["inputs"]
        for node in prompt.values()
        if node["class_type"] == "ImageFromBatch"
    ]
    assert {entry["batch_index"] for entry in image_slices} == {-22, 22}
    assert {entry["length"] for entry in image_slices} == {22, 124}
    guides = [
        node["inputs"]
        for node in prompt.values()
        if node["class_type"] == "MiniMaxH3AddGuide"
    ]
    assert len(guides) == 1
    assert guides[0]["frame_idx"] == 0
    assert {node["inputs"]["start_index"] for node in prompt.values()
            if node["class_type"] == "TrimAudioDuration"} == {
        -(22 / 24),
        22 / 24,
    }
    conditioning = next(
        node
        for node in prompt.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    )
    assert conditioning["inputs"]["length"] == 158

    first_plan, successor_plan = result.plans
    assert first_plan["visible_frame_count"] == 124
    assert first_plan["sample_frame_count"] == 124
    assert first_plan["continuity_context_frames"] == 0
    assert first_plan["predecessor_segment_id"] is None
    assert first_plan["anchor_reset"] is True
    assert successor_plan["visible_frame_count"] == 124
    assert successor_plan["sample_frame_count"] == 158
    assert successor_plan["continuity_context_frames"] == 22
    assert successor_plan["alignment_tail_frame_count"] == 12
    assert successor_plan["predecessor_segment_id"] == "segment-1"
    assert successor_plan["anchor_reset"] is False
    assert result.manifest["units"][1]["continuity"] == {
        "predecessor_segment_id": "segment-1",
        "overlap_frames": 22,
        "load_video_node_id": successor.continuity.load_video_node_id,
        "source": "same_run",
        "historical_take_id": None,
        "resolved": False,
    }
    assert result.manifest["continuity"]["boundaries"] == [{
        "segment_id": "segment-2",
        "predecessor_segment_id": "segment-1",
        "overlap_frames": 22,
        "source": "same_run",
        "historical_take_id": None,
    }]

    available = set(result.node_policy["allowed_nodes"])
    provenance = {
        node: EXPECTED_NATIVE_NODE_MODULES[node] for node in available
    }
    # Registry preflight is valid while the dependency is intentionally
    # unresolved; readiness is a separate mandatory submission boundary.
    validate_native_capabilities(result, available, provenance)
    bound = bind_native_workflow_predecessor_output(
        successor,
        {
            "filename": "Director_timeline_00001_.mp4",
            "subfolder": "video",
            "type": "output",
        },
    )
    validate_native_workflow_ready(bound)
    assert bound.continuity is not None and bound.continuity.resolved is True
    assert (
        bound.prompt[bound.continuity.load_video_node_id]["inputs"]["file"]
        == "video/Director_timeline_00001_.mp4 [output]"
    )
    assert "UNBOUND_PREDECESSOR_OUTPUT" in dependency_loader["inputs"]["file"]


def test_raylight_epoch_binding_preserves_resolved_continuity_metadata() -> None:
    successor = compile_native_timeline(
        _continuity_draft("t2v", "t2v"),
        _settings(fl_backend="raylight"),
        "job-ray-continuity",
    ).workflows[1]
    bound = bind_native_workflow_predecessor_output(
        successor,
        {
            "filename": "take.mp4",
            "subfolder": "video",
            "type": "output",
        },
    )

    epoch_bound = bind_raylight_runtime_epoch(bound, 3)

    assert epoch_bound.continuity == bound.continuity
    validate_native_workflow_ready(epoch_bound)


@pytest.mark.parametrize(
    "output",
    [
        {"filename": "take.mp4", "subfolder": "video", "type": "input"},
        {"filename": "../take.mp4", "subfolder": "video", "type": "output"},
        {"filename": "take.mp4", "subfolder": "../video", "type": "output"},
        {"filename": "take.mp4", "subfolder": "/video", "type": "output"},
    ],
)
def test_continuity_output_binding_rejects_non_output_and_unsafe_paths(
    output: dict,
) -> None:
    successor = compile_native_timeline(
        _continuity_draft("t2v", "t2v"), _settings(), "job-bind-safety"
    ).workflows[1]

    with pytest.raises(NativeTemplateError):
        bind_native_workflow_predecessor_output(successor, output)


def test_continuity_first_image_resets_chain_without_predecessor_input() -> None:
    result = compile_native_timeline(
        _continuity_draft("t2v", "i2v", "t2v"),
        _settings(),
        "job-anchor-reset",
    )

    first, reset, successor = result.workflows
    assert first.continuity is None
    assert reset.continuity is None
    assert result.plans[1]["anchor_reset"] is True
    assert result.plans[1]["continuity_context_frames"] == 0
    assert result.plans[1]["sample_frame_count"] == 124
    reset_conditioning = next(
        node
        for node in reset.prompt.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    )
    assert "first_frame" in reset_conditioning["inputs"]
    assert successor.continuity is not None
    assert successor.continuity.predecessor_segment_id == "segment-2"


def test_continuity_switch_and_overlap_are_independent_per_target_segment() -> None:
    value = _continuity_draft("t2v", "t2v", "t2v", "t2v").model_dump(
        mode="json"
    )
    value["segments"][1]["continuity"] = {
        "enabled": True,
        "overlap_frames": 5,
    }
    value["segments"][2]["continuity"] = {
        "enabled": False,
        "overlap_frames": 39,
    }
    value["segments"][3]["continuity"] = {
        "enabled": True,
        "overlap_frames": 56,
    }

    result = compile_native_timeline(
        UnifiedTimelineDraft.model_validate(value),
        _settings(),
        "job-per-segment-continuity",
    )

    assert [plan["continuity_context_frames"] for plan in result.plans] == [
        0,
        5,
        0,
        56,
    ]
    assert result.workflows[1].continuity is not None
    assert result.workflows[1].continuity.predecessor_segment_id == "segment-1"
    assert result.workflows[2].continuity is None
    assert result.workflows[3].continuity is not None
    assert result.workflows[3].continuity.predecessor_segment_id == "segment-3"
    assert result.manifest["continuity"]["boundaries"] == [
        {
            "segment_id": "segment-2",
            "predecessor_segment_id": "segment-1",
            "overlap_frames": 5,
            "source": "same_run",
            "historical_take_id": None,
        },
        {
            "segment_id": "segment-4",
            "predecessor_segment_id": "segment-3",
            "overlap_frames": 56,
            "source": "same_run",
            "historical_take_id": None,
        },
    ]


def test_continuity_partial_selection_requires_server_resolved_take() -> None:
    draft = _continuity_draft("t2v", "t2v")
    with pytest.raises(NativeTemplateError, match="server-resolved historical take"):
        compile_native_timeline(
            draft, _settings(), "job-selection", segment_ids=["segment-2"]
        )

    result = compile_native_timeline(
        draft,
        _settings(),
        "job-selection",
        segment_ids=["segment-2"],
        historical_takes={
            "segment-2": NativeHistoricalTake(
                id="take-1",
                segment_id="segment-1",
                output={
                    "filename": "segment-1.mp4",
                    "subfolder": "segments",
                    "type": "output",
                },
            )
        },
    )
    dependency = result.workflows[0].continuity
    assert dependency is not None
    assert dependency.source == "historical_take"
    assert dependency.historical_take_id == "take-1"
    assert dependency.resolved is True
    assert result.plans[0]["continuity_source"] == "historical_take"
    assert result.plans[0]["historical_take_id"] == "take-1"
    validate_native_workflow_ready(result.workflows[0])

    mixed = compile_native_timeline(
        _continuity_draft("t2v", "r2v"),
        _settings(),
        "job-cross-family",
    )
    assert [unit.family for unit in mixed.workflows] == ["fl2va", "ref2va"]
    assert mixed.workflows[1].continuity is not None
    assert mixed.workflows[1].continuity.predecessor_segment_id == "segment-1"


def test_continuity_fl_last_image_keeps_picture_token_and_visible_anchor() -> None:
    value = _continuity_draft("t2v", "fl2v").model_dump(mode="json")
    value["segments"][1]["first_image"] = None
    value["segments"][1]["prompt"] = "End on <Picture 1>."
    draft = UnifiedTimelineDraft.model_validate(value)
    successor = compile_native_timeline(
        draft, _settings(), "job-visible-last-anchor"
    ).workflows[1]
    prompt = successor.prompt
    conditioning = next(
        node
        for node in prompt.values()
        if node["class_type"] == "MiniMaxH3ImageToVideo"
    )
    # Keep the image in ImageToVideo so Qwen receives its <Picture 1> vision
    # block. The same image is also guided at frame 145 because frame 157 is an
    # alignment-only tail that will be cropped from the saved segment.
    assert conditioning["inputs"]["prompt"] == "End on <Picture 1>."
    assert "last_frame" in conditioning["inputs"]
    assert sorted(
        node["inputs"]["frame_idx"]
        for node in prompt.values()
        if node["class_type"] == "MiniMaxH3AddGuide"
    ) == [0, 145]


@pytest.mark.parametrize("audio_mode", ["source", "mute"])
def test_continuity_source_and_mute_do_not_reuse_predecessor_audio(
    audio_mode: str,
) -> None:
    result = compile_native_timeline(
        _continuity_draft("v2v", "v2v", audio_mode=audio_mode),
        _settings(),
        f"job-continuity-{audio_mode}",
    )
    prompt = result.workflows[1].prompt
    assert "TrimAudioDuration" not in _types(prompt)
    create_video = next(
        node for node in prompt.values() if node["class_type"] == "CreateVideo"
    )
    if audio_mode == "source":
        assert "audio" in create_video["inputs"]
    else:
        assert "audio" not in create_video["inputs"]


def test_continuity_rejects_aligned_sample_above_h3_limit() -> None:
    value = _continuity_draft(
        "t2v", "t2v", overlap_frames=56
    ).model_dump(mode="json")
    value["segments"][1]["duration_seconds"] = 20.0
    with pytest.raises(NativeTemplateError, match="limit is 512"):
        compile_native_timeline(
            UnifiedTimelineDraft.model_validate(value),
            _settings(),
            "job-continuity-limit",
        )


def test_native_v1_ignores_obsolete_cfg_and_rejects_settings_it_cannot_honor() -> None:
    value = _draft("t2v").model_dump(mode="json")
    value["sampling"]["fl2va"]["cfg"] = 27.0
    migrated = UnifiedTimelineDraft.model_validate(value)
    assert "cfg" not in migrated.sampling.fl2va.model_dump(mode="json")
    compile_native_timeline(migrated, _settings(), "job-cfg")

    value = _draft("t2v").model_dump(mode="json")
    value["render"]["fps"] = 30.0
    with pytest.raises(NativeTemplateError, match="must equal 24"):
        compile_native_timeline(
            UnifiedTimelineDraft.model_validate(value), _settings(), "job-fps"
        )

    settings_value = _settings().model_dump(mode="json")
    settings_value["memory_policy"] = "clear_between_segments"
    with pytest.raises(ValidationError, match="keep_resident"):
        RuntimeSettings.model_validate(settings_value)

    settings_value = _settings().model_dump(mode="json")
    settings_value["models"]["fl2va"].update(
        lora_name="minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        lora_loader="dedicated",
        lora_low_vram=True,
    )
    automatic_standard_lora = compile_native_timeline(
        _draft("t2v"),
        RuntimeSettings.model_validate(settings_value),
        "job-low-vram-generic",
    )
    assert "LoraLoaderModelOnly" in _types(
        automatic_standard_lora.workflows[0].prompt
    )

    settings_value = _settings(fl_backend="raylight").model_dump(mode="json")
    settings_value["models"]["fl2va"].update(
        lora_name="minimax_h3_fl2v_turbo_4step_v1.0_768p_10ErosMax_beta1_pruned_compat_v001_T8.safetensors",
        lora_loader="model_only",
        lora_low_vram=True,
    )
    automatic_ray_lora = compile_native_timeline(
        _draft("t2v"),
        RuntimeSettings.model_validate(settings_value),
        "job-ray-explicit-loader",
    )
    assert "RayLoraLoader" in _types(automatic_ray_lora.workflows[0].prompt)


def test_standard_ref2va_official_lora_uses_core_model_only_loader() -> None:
    value = _settings().model_dump(mode="json")
    value["models"]["ref2va"].update(
        lora_name="minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        lora_strength=0.75,
    )

    result = compile_native_timeline(
        _draft("r2v"), RuntimeSettings.model_validate(value), "job-ref2v-lora"
    )

    prompt = result.workflows[0].prompt
    lora = next(
        node for node in prompt.values()
        if node["class_type"] == "LoraLoaderModelOnly"
    )
    assert lora["inputs"]["lora_name"] == value["models"]["ref2va"]["lora_name"]
    assert lora["inputs"]["strength_model"] == 0.75
    assert "RayLoraLoader" not in _types(prompt)
    assert result.manifest["lora_resolution"]["ref2va"]["source"] == "audited_profile"


def test_unknown_standard_lora_uses_remote_generic_metadata_or_scoped_override() -> None:
    value = _settings().model_dump(mode="json")
    value["models"]["fl2va"]["lora_name"] = "renamed_generic.safetensors"
    settings = RuntimeSettings.model_validate(value)

    inferred = compile_native_timeline(
        _draft("t2v"),
        settings,
        "job-metadata-lora",
        standard_lora_metadata={
            "fl2va": {
                "target_format": "ComfyUI generic LoRA",
                "source_format": "Diffusers PEFT LoRA",
            }
        },
    )
    assert "LoraLoaderModelOnly" in _types(inferred.workflows[0].prompt)
    assert inferred.manifest["lora_resolution"]["fl2va"]["source"] == "metadata"

    with pytest.raises(NativeTemplateError, match="choose a Standard LoRA loader"):
        compile_native_timeline(_draft("t2v"), settings, "job-unknown-lora")

    value["models"]["fl2va"]["standard_lora_loader_override"] = {
        "loader": "bypass_model_only",
        "lora_name": "renamed_generic.safetensors",
        "model_filename": value["models"]["fl2va"]["filename"],
        "comfy_origin": value["comfy_url"],
    }
    explicit = compile_native_timeline(
        _draft("t2v"),
        RuntimeSettings.model_validate(value),
        "job-explicit-lora",
    )
    assert "LoraLoaderBypassModelOnly" in _types(explicit.workflows[0].prompt)
    assert explicit.manifest["lora_resolution"]["fl2va"]["source"] == "manual"


def test_native_video_conditioning_requires_a_24fps_server_proxy() -> None:
    value = _draft("v2v").model_dump(mode="json")
    value["segments"][0]["source_video"]["metadata"]["native_fps"] = 30.0

    with pytest.raises(NativeTemplateError, match="server-created 24 fps proxy"):
        compile_native_timeline(
            UnifiedTimelineDraft.model_validate(value), _settings(), "job-proxy"
        )
