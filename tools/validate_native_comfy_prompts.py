#!/usr/bin/env python3
"""Validate the server-owned native H3 templates against installed ComfyUI.

This is a registry/prompt-structure test only.  It forces ComfyUI into CPU
mode, does not queue prompts, load model weights, or sample video.  It covers
all six standard templates, all six two-GPU RayLight templates, and dynamically
bound predecessor-output continuity prompts for both execution backends.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_COMFY_ROOT_VALUE = os.environ.get("COMFYUI_ROOT", "").strip()
COMFY_ROOT = (
    Path(_COMFY_ROOT_VALUE).expanduser().resolve()
    if _COMFY_ROOT_VALUE
    else PROJECT_ROOT / ".COMFYUI_ROOT-is-required"
)
RAYLIGHT_ROOT = Path(
    os.environ.get(
        "RAYLIGHT_NODE_ROOT", str(COMFY_ROOT / "custom_nodes" / "raylight")
    )
).resolve()
TURBO_LORA_ROOT = Path(
    os.environ.get(
        "MINIMAX_H3_TURBO_NODE_ROOT",
        str(COMFY_ROOT / "custom_nodes" / "ComfyUI-MiniMax-H3-Turbo"),
    )
).resolve()
TURBO_LORA = os.environ.get(
    "NATIVE_VALIDATION_TURBO_LORA",
    "minimax_h3_turbo_v4_step600_ema.safetensors",
)
VALIDATION_IMAGE = os.environ.get(
    "NATIVE_VALIDATION_IMAGE", "director-validation/image.png"
)
VALIDATION_VIDEO = os.environ.get(
    "NATIVE_VALIDATION_VIDEO", "director-validation/video.mp4"
)
VALIDATION_OUTPUT_VIDEO = os.environ.get("NATIVE_VALIDATION_OUTPUT_VIDEO", "")

sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(COMFY_ROOT))

from director.native_templates import (  # noqa: E402
    EXPECTED_NATIVE_NODE_MODULES,
    bind_native_workflow_predecessor_output,
    build_raylight_shutdown_unit,
    compile_native_timeline,
    raylight_runtime_descriptor,
    validate_native_workflow_ready,
)
from director.h3_capabilities import H3_REFERENCE_LIMITS  # noqa: E402
from director.schemas import (  # noqa: E402
    RuntimeSettings,
    UnifiedTimelineDraft,
    default_settings,
)


MODES = ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v")


def _asset(kind: str, path: str, *, slot: int | None = None) -> dict[str, Any]:
    normalized = path.replace("\\", "/").strip("/")
    parts = normalized.rsplit("/", 1)
    name = parts[-1]
    subfolder = parts[0] if len(parts) == 2 else ""
    value: dict[str, Any] = {
        "id": f"validation-{kind}-{slot if slot is not None else 'source'}",
        "name": name,
        "subfolder": subfolder,
        "type": "input",
        "kind": kind,
    }
    if kind == "video":
        # This fixture is a known 24fps validation proxy. LoadVideo performs
        # the authoritative path check; no video frames are decoded here.
        value["metadata"] = {
            "duration": 12.0,
            "native_fps": 24.0,
            "frame_count": 288,
            "width": 1920,
            "height": 1080,
            "probe_method": "native_prompt_validation_fixture",
            "has_audio": True,
        }
    if slot is not None:
        value["slot"] = slot
    return value


def _segment(mode: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": f"validation-{mode}",
        "title": f"Validate {mode}",
        "mode": mode,
        "prompt": "A cinematic scene with synchronized ambience.",
        "duration_seconds": 5.0,
        "enabled": True,
    }
    if mode == "i2v":
        value["prompt"] = "Begin from <Picture 1>."
        value["first_image"] = _asset("image", VALIDATION_IMAGE)
    elif mode == "fl2v":
        value.update(
            prompt="Move from <Picture 1> to <Picture 2>.",
            first_image=_asset("image", VALIDATION_IMAGE),
            last_image=_asset("image", VALIDATION_IMAGE),
        )
    elif mode == "r2v":
        value.update(
            prompt="Use <Picture 1>, <Video 1>, and <Audio 1>.",
            reference_images=[_asset("image", VALIDATION_IMAGE, slot=0)],
            reference_videos=[_asset("video", VALIDATION_VIDEO, slot=0)],
            # LoadAudio accepts a video container and extracts its soundtrack.
            reference_audios=[_asset("audio", VALIDATION_VIDEO, slot=0)],
        )
    elif mode == "v2v":
        value.update(
            prompt="Edit <Video 1>.",
            source_video=_asset("video", VALIDATION_VIDEO),
            source_start_seconds=0.0,
            source_duration_seconds=5.0,
        )
    elif mode == "rv2v":
        value.update(
            prompt="Edit <Video 1> using <Picture 1>.",
            source_video=_asset("video", VALIDATION_VIDEO),
            source_start_seconds=0.0,
            source_duration_seconds=5.0,
            reference_images=[_asset("image", VALIDATION_IMAGE, slot=0)],
            reference_audios=[],
        )
    return value


def _draft(mode: str, *, scheduler: str = "simple") -> UnifiedTimelineDraft:
    return UnifiedTimelineDraft.model_validate(
        {
            "version": 1,
            "title": "Native prompt validation",
            "prompt": "",
            "ref_image_size": "match",
            "render": {"width": 864, "height": 480, "fps": 24.0},
            "sampling": {
                "steps": 20,
                "cfg": 1.0,
                "seed": 42,
                "sampler": "res_multistep",
                "scheduler": scheduler,
                "shift": 12.0,
                "audio_shift": 3.0,
            },
            "continuity": {"enabled": False, "overlap_frames": 22},
            "audio_mode": "generate",
            "export_mode": "segments",
            "segments": [_segment(mode)],
        }
    )


def _continuity_draft(mode: str) -> UnifiedTimelineDraft:
    value = _draft(mode).model_dump(mode="json")
    predecessor = value["segments"][0]
    successor = dict(predecessor)
    predecessor["id"] = f"validation-{mode}-predecessor"
    predecessor["title"] = f"Validate {mode} predecessor"
    successor["id"] = f"validation-{mode}-successor"
    successor["title"] = f"Validate {mode} successor"
    # An explicit FL first image is a deliberate chain reset. The registry
    # prompt must exercise the dynamic predecessor path instead.
    if successor["mode"] == "fl2va":
        successor["first_image"] = None
        if successor.get("last_image") is not None:
            successor["prompt"] = "End on <Picture 1>."
    successor["continuity"] = {"enabled": True, "overlap_frames": 22}
    value["segments"] = [predecessor, successor]
    return UnifiedTimelineDraft.model_validate(value)


def _paired_source_audio_draft(mode: str) -> UnifiedTimelineDraft:
    if mode not in {"v2v", "rv2v"}:
        raise ValueError("paired source audio validation is v2v/rv2v only")
    value = _draft(mode).model_dump(mode="json")
    segment = value["segments"][0]
    segment["source_audio_as_reference"] = True
    segment["prompt"] = "Edit <Video 1> with paired <Audio 1>."
    return UnifiedTimelineDraft.model_validate(value)


def _maximum_reference_draft(*, with_source_video: bool) -> UnifiedTimelineDraft:
    """Exercise every last legal stock MiniMax H3 Autogrow input."""

    value = _draft("v2v" if with_source_video else "r2v").model_dump(mode="json")
    segment = value["segments"][0]
    independent_video_count = (
        H3_REFERENCE_LIMITS.independent_reference_video_capacity(
            has_source_video=with_source_video
        )
    )
    segment["reference_images"] = [
        _asset("image", VALIDATION_IMAGE, slot=slot)
        for slot in range(H3_REFERENCE_LIMITS.reference_images)
    ]
    segment["reference_videos"] = [
        _asset("video", VALIDATION_VIDEO, slot=slot)
        for slot in range(independent_video_count)
    ]
    segment["reference_audios"] = [
        _asset("audio", VALIDATION_VIDEO, slot=slot)
        for slot in range(H3_REFERENCE_LIMITS.standalone_reference_audios)
    ]
    if with_source_video:
        segment["source_audio_as_reference"] = True
    picture_tags = " ".join(
        f"<Picture {index}>"
        for index in range(1, H3_REFERENCE_LIMITS.reference_images + 1)
    )
    video_tags = " ".join(
        f"<Video {index}>"
        for index in range(1, H3_REFERENCE_LIMITS.reference_video_channels + 1)
    )
    audio_count = H3_REFERENCE_LIMITS.standalone_reference_audios + (
        1 if with_source_video else 0
    )
    audio_tags = " ".join(
        f"<Audio {index}>" for index in range(1, audio_count + 1)
    )
    segment["prompt"] = f"Use {picture_tags} {video_tags} {audio_tags}."
    return UnifiedTimelineDraft.model_validate(value)


def _raylight_settings() -> RuntimeSettings:
    value = default_settings("http://127.0.0.1:8188").model_dump(mode="json")
    for family in ("fl2va", "ref2va"):
        value["models"][family].update(
            backend="raylight",
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
    return RuntimeSettings.model_validate(value)


def _raylight_lora_settings() -> RuntimeSettings:
    value = _raylight_settings().model_dump(mode="json")
    value["models"]["fl2va"].update(
        lora_name=TURBO_LORA,
        lora_loader="auto",
        lora_low_vram=False,
    )
    return RuntimeSettings.model_validate(value)


def _dedicated_lora_settings() -> RuntimeSettings:
    value = _standard_settings().model_dump(mode="json")
    value["models"]["fl2va"].update(
        lora_name=TURBO_LORA,
        lora_loader="dedicated",
        lora_low_vram=False,
    )
    return RuntimeSettings.model_validate(value)


def _standard_lora_settings(loader: str) -> RuntimeSettings:
    value = _standard_settings().model_dump(mode="json")
    value["models"]["fl2va"].update(
        lora_name=TURBO_LORA,
        lora_loader=loader,
        lora_low_vram=False,
    )
    return RuntimeSettings.model_validate(value)


def _standard_settings() -> RuntimeSettings:
    """Exercise the official explicit-device selectors while backend=auto."""

    value = default_settings("http://127.0.0.1:8188").model_dump(mode="json")
    for family in ("fl2va", "ref2va"):
        value["models"][family]["device"] = "gpu:0"
    value["models"]["clip"]["device"] = "gpu:0"
    value["models"]["video_vae"]["device"] = "gpu:0"
    value["models"]["audio_vae"]["device"] = "gpu:0"
    return RuntimeSettings.model_validate(value)


def _require_fixture(path: str) -> None:
    target = COMFY_ROOT / "input" / path
    if not target.is_file():
        raise SystemExit(
            f"validation input is missing: {target}. Set NATIVE_VALIDATION_IMAGE "
            "or NATIVE_VALIDATION_VIDEO to a ComfyUI input-relative path."
        )


def _continuity_output_descriptor() -> dict[str, str]:
    """Resolve a real persisted output for LoadVideo prompt validation."""

    output_root = COMFY_ROOT / "output"
    if VALIDATION_OUTPUT_VIDEO:
        relative = Path(VALIDATION_OUTPUT_VIDEO.replace("\\", "/").strip("/"))
        target = output_root / relative
        candidates = [target]
    else:
        candidates = sorted(
            (
                candidate
                for suffix in ("*.mp4", "*.mov", "*.mkv", "*.webm")
                for candidate in output_root.rglob(suffix)
            ),
            key=lambda candidate: candidate.as_posix(),
        )
    target = next((candidate for candidate in candidates if candidate.is_file()), None)
    if target is None:
        raise SystemExit(
            "continuity validation needs a real ComfyUI output video. Set "
            "NATIVE_VALIDATION_OUTPUT_VIDEO to an output-relative path."
        )
    try:
        relative = target.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise SystemExit(
            f"continuity validation output escapes ComfyUI output: {target}"
        ) from exc
    return {
        "filename": relative.name,
        "subfolder": relative.parent.as_posix() if len(relative.parts) > 1 else "",
        "type": "output",
    }


async def _validate(
    execution: Any,
    prompt_id: str,
    prompt: dict[str, Any],
) -> bool:
    valid, error, outputs, node_errors = await execution.validate_prompt(
        prompt_id, prompt, None
    )
    if valid:
        print(f"PASS {prompt_id}: outputs={sorted(outputs)}")
        return True
    print(f"FAIL {prompt_id}: error={error!r} node_errors={node_errors!r}")
    return False


def _validate_registry_provenance(nodes: Any, results: list[Any]) -> list[str]:
    """Check the same registry metadata served by ComfyUI ``/object_info``."""

    failures: list[str] = []
    expected: dict[str, str] = {}
    for result in results:
        expected.update(result.node_policy["provenance"])
        for unit in result.workflows:
            if any(
                node["class_type"] == "MiniMaxH3Director"
                for node in unit.prompt.values()
            ):
                failures.append(f"{unit.id}: emitted forbidden MiniMaxH3Director")
    for node_name, provenance in sorted(expected.items()):
        node_class = nodes.NODE_CLASS_MAPPINGS.get(node_name)
        if node_class is None:
            failures.append(f"{node_name}: missing from registry")
            continue
        module = str(getattr(node_class, "__module__", ""))
        relative = str(getattr(node_class, "RELATIVE_PYTHON_MODULE", ""))
        source = relative or module
        matches = source == EXPECTED_NATIVE_NODE_MODULES[node_name]
        if not matches:
            failures.append(
                f"{node_name}: policy={provenance}, registry module={module!r}, "
                f"python_module={relative!r}"
            )
    if failures:
        for failure in failures:
            print(f"FAIL provenance: {failure}")
    else:
        print(
            f"PASS object_info provenance: {len(expected)} emitted node classes; "
            "no Director node"
        )
    return failures


def _validate_raylight_initializer_contract(nodes: Any) -> list[str]:
    """Verify the installed initializer exposes Director's exact inputs.

    ComfyUI ignores unknown prompt inputs during structural validation, so a
    prompt-only check could falsely pass against an older RayLight and then use
    its endpoint-global cleanup behavior or reject Director's attention choice.
    Inspect the live registered node schema as an additional compatibility
    boundary. RayLight keeps TORCH_FLASH as its third-party workflow default;
    Director selects COMFY_KITCHEN_INT8 explicitly in every generated prompt.
    """

    node_class = nodes.NODE_CLASS_MAPPINGS.get("RayInitializerAdvanced")
    if node_class is None:
        return ["RayInitializerAdvanced: missing from registry"]
    try:
        input_types = node_class.INPUT_TYPES()
    except Exception as exc:
        return [f"RayInitializerAdvanced: INPUT_TYPES failed: {exc}"]
    if not isinstance(input_types, dict):
        return ["RayInitializerAdvanced: INPUT_TYPES is not an object"]
    required = input_types.get("required")
    if not isinstance(required, dict):
        return ["RayInitializerAdvanced: required inputs are missing"]
    attention = required.get("XFuser_attention")
    if not isinstance(attention, tuple) or not attention:
        return [
            "RayInitializerAdvanced.XFuser_attention must be a required combo"
        ]
    attention_options = attention[0]
    attention_metadata = (
        attention[1]
        if len(attention) > 1 and isinstance(attention[1], dict)
        else {}
    )
    # Support both the legacy custom-node tuple and Comfy's newer COMBO shape.
    if attention_options == "COMBO":
        attention_options = attention_metadata.get("options")
    if (
        not isinstance(attention_options, (list, tuple))
        or "COMFY_KITCHEN_INT8" not in attention_options
        or "TORCH_FLASH" not in attention_options
    ):
        return [
            "RayInitializerAdvanced.XFuser_attention must offer "
            "COMFY_KITCHEN_INT8 and TORCH_FLASH"
        ]
    if attention_metadata.get("default") != "TORCH_FLASH":
        return [
            "RayInitializerAdvanced.XFuser_attention must retain "
            "TORCH_FLASH as the third-party workflow default"
        ]
    optional = input_types.get("optional")
    if not isinstance(optional, dict):
        return ["RayInitializerAdvanced: optional inputs are missing"]
    policy = optional.get("driver_cleanup_policy")
    if (
        not isinstance(policy, tuple)
        or not policy
        or not isinstance(policy[0], list)
        or "legacy_all" not in policy[0]
        or "ray_devices" not in policy[0]
    ):
        return [
            "RayInitializerAdvanced.driver_cleanup_policy must offer "
            "legacy_all and ray_devices"
        ]
    # The RAM-cache fork contract: Director explicitly sends this input, and
    # stock RayLight would silently ignore it (ComfyUI drops unknown prompt
    # inputs), leaving the model-switch RAM cache inactive. Fail the audit
    # unless the registered schema advertises it.
    ram_cache = optional.get("ram_cache_max_models")
    if (
        not isinstance(ram_cache, tuple)
        or not ram_cache
        or ram_cache[0] != "INT"
        or not isinstance(ram_cache[1], dict)
        or ram_cache[1].get("default") != 2
        or ram_cache[1].get("min") != 0
    ):
        return [
            "RayInitializerAdvanced.ram_cache_max_models must be an optional "
            "INT input with default 2 and min 0 (worker RAM-cache fork)"
        ]
    print(
        "PASS RayInitializerAdvanced contract: attention offers "
        "COMFY_KITCHEN_INT8/TORCH_FLASH with legacy default TORCH_FLASH; "
        "driver cleanup and RAM-cache inputs present"
    )
    return []


def _validate_compiled_raylight_attention(results: list[Any]) -> list[str]:
    """Require Director-owned RayLight prompts to select CK explicitly."""

    failures: list[str] = []
    checked = 0
    for result in results:
        for unit in result.workflows:
            if unit.backend != "raylight":
                continue
            initializers = [
                node
                for node in unit.prompt.values()
                if node.get("class_type") == "RayInitializerAdvanced"
            ]
            if len(initializers) != 1:
                failures.append(
                    f"{unit.id}: expected exactly one RayInitializerAdvanced"
                )
                continue
            checked += 1
            value = initializers[0].get("inputs", {}).get("XFuser_attention")
            if value != "COMFY_KITCHEN_INT8":
                failures.append(
                    f"{unit.id}: XFuser_attention must be explicitly "
                    "COMFY_KITCHEN_INT8"
                )
    if not failures:
        print(
            "PASS Director RayLight attention contract: "
            f"{checked} prompts explicitly select COMFY_KITCHEN_INT8"
        )
    return failures


def _validate_beta_scheduler_contract(nodes: Any) -> list[str]:
    """Verify both scheduler nodes advertise the exact ``beta`` wire value.

    Prompt validation normally catches invalid combo values, but checking the
    live schemas produces a direct compatibility error for either the stock
    ComfyUI node or RayLight's mirrored node after an upstream upgrade.
    """

    failures: list[str] = []
    for node_name in ("BasicScheduler", "RayBasicScheduler"):
        node_class = nodes.NODE_CLASS_MAPPINGS.get(node_name)
        if node_class is None:
            failures.append(f"{node_name}: missing from registry")
            continue
        try:
            input_types = node_class.INPUT_TYPES()
        except Exception as exc:
            failures.append(f"{node_name}: INPUT_TYPES failed: {exc}")
            continue
        required = input_types.get("required") if isinstance(input_types, dict) else None
        scheduler = required.get("scheduler") if isinstance(required, dict) else None
        options = scheduler[0] if isinstance(scheduler, tuple) and scheduler else None
        # ComfyUI's newer comfy_api nodes expose a V1 compatibility tuple as
        # ("COMBO", {"options": [...]}); legacy/custom nodes put the list in
        # tuple position zero.  Audit both registry shapes.
        if (
            options == "COMBO"
            and len(scheduler) > 1
            and isinstance(scheduler[1], dict)
        ):
            options = scheduler[1].get("options")
        if not isinstance(options, (list, tuple)) or "beta" not in options:
            failures.append(f"{node_name}.scheduler does not offer beta")
    if not failures:
        print("PASS beta scheduler contract: BasicScheduler, RayBasicScheduler")
    return failures


async def validate_all() -> int:
    if not _COMFY_ROOT_VALUE:
        raise SystemExit(
            "COMFYUI_ROOT is required; set it to the root of the ComfyUI checkout"
        )
    if not COMFY_ROOT.is_dir():
        raise SystemExit(f"ComfyUI root does not exist: {COMFY_ROOT}")
    if not RAYLIGHT_ROOT.is_dir():
        raise SystemExit(f"RayLight node root does not exist: {RAYLIGHT_ROOT}")
    if not TURBO_LORA_ROOT.is_dir():
        raise SystemExit(f"MiniMax H3 Turbo node root does not exist: {TURBO_LORA_ROOT}")
    if not (COMFY_ROOT / "models" / "loras" / TURBO_LORA).is_file():
        raise SystemExit(f"validation LoRA is missing: {TURBO_LORA}")
    _require_fixture(VALIDATION_IMAGE)
    _require_fixture(VALIDATION_VIDEO)
    continuity_output = _continuity_output_descriptor()

    # This must happen before importing execution/nodes, which import CUDA
    # model management as a side effect.
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

    import execution
    import nodes
    import server

    server.PromptServer(asyncio.get_running_loop())
    await nodes.init_public_apis()
    builtin_failures = await nodes.init_builtin_extra_nodes()
    if builtin_failures:
        print(f"Warning: unrelated built-in nodes failed to import: {builtin_failures}")

    # Auto + a one-device Ray pool resolves to standard. Explicit gpu:0
    # placements then exercise all three official Select*Device nodes.
    settings = _standard_settings()
    failures: list[str] = []
    compiled_results: list[Any] = []
    for mode in MODES:
        result = compile_native_timeline(
            _draft(mode), settings, f"validate-standard-{mode}"
        )
        compiled_results.append(result)
        if not await _validate(
            execution,
            f"native-standard-{mode}",
            result.workflows[0].prompt,
        ):
            failures.append(f"standard-{mode}")
    for mode in ("t2v", "r2v"):
        result = compile_native_timeline(
            _draft(mode, scheduler="beta"),
            settings,
            f"validate-standard-beta-{mode}",
        )
        compiled_results.append(result)
        if not await _validate(
            execution,
            f"native-standard-beta-{mode}",
            result.workflows[0].prompt,
        ):
            failures.append(f"standard-beta-{mode}")
    for mode in ("v2v", "rv2v"):
        result = compile_native_timeline(
            _paired_source_audio_draft(mode),
            settings,
            f"validate-standard-{mode}-paired-source-audio",
        )
        compiled_results.append(result)
        if not await _validate(
            execution,
            f"native-standard-{mode}-paired-source-audio",
            result.workflows[0].prompt,
        ):
            failures.append(f"standard-{mode}-paired-source-audio")

    standard_continuity = compile_native_timeline(
        _continuity_draft("fl2v"),
        settings,
        "validate-standard-continuity",
    )
    compiled_results.append(standard_continuity)
    bound_standard_continuity = bind_native_workflow_predecessor_output(
        standard_continuity.workflows[1], continuity_output
    )
    validate_native_workflow_ready(bound_standard_continuity)
    if not await _validate(
        execution,
        "native-standard-continuity",
        bound_standard_continuity.prompt,
    ):
        failures.append("standard-continuity")

    for label, draft in (
        ("r2v-max-references", _maximum_reference_draft(with_source_video=False)),
        ("rv2v-max-references", _maximum_reference_draft(with_source_video=True)),
    ):
        result = compile_native_timeline(draft, settings, f"validate-standard-{label}")
        compiled_results.append(result)
        if not await _validate(
            execution,
            f"native-standard-{label}",
            result.workflows[0].prompt,
        ):
            failures.append(f"standard-{label}")

    base_nodes = set(nodes.NODE_CLASS_MAPPINGS)
    raylight_loaded = await nodes.load_custom_node(
        str(RAYLIGHT_ROOT), base_nodes, module_parent="custom_nodes"
    )
    if not raylight_loaded:
        print("FAIL raylight-u2: RayLight failed to register")
        failures.append("raylight-u2")
    else:
        raylight_settings = _raylight_settings()
        raylight_without_lora = None
        for mode in MODES:
            family = "fl2va" if mode in {"t2v", "i2v", "fl2v"} else "ref2va"
            value = raylight_settings.model_dump(mode="json")
            value["models"][family]["backend"] = "raylight"
            selected_settings = RuntimeSettings.model_validate(value)
            result = compile_native_timeline(
                _draft(mode), selected_settings, f"validate-raylight-u2-{mode}"
            )
            compiled_results.append(result)
            if raylight_without_lora is None:
                raylight_without_lora = result.workflows[0]
            if not await _validate(
                execution,
                f"native-raylight-u2-{mode}",
                result.workflows[0].prompt,
            ):
                failures.append(f"raylight-u2-{mode}")

        for mode in ("t2v", "r2v"):
            family = "fl2va" if mode == "t2v" else "ref2va"
            value = raylight_settings.model_dump(mode="json")
            value["models"][family]["backend"] = "raylight"
            selected_settings = RuntimeSettings.model_validate(value)
            result = compile_native_timeline(
                _draft(mode, scheduler="beta"),
                selected_settings,
                f"validate-raylight-u2-beta-{mode}",
            )
            compiled_results.append(result)
            if not await _validate(
                execution,
                f"native-raylight-u2-beta-{mode}",
                result.workflows[0].prompt,
            ):
                failures.append(f"raylight-u2-beta-{mode}")

        raylight_continuity = compile_native_timeline(
            _continuity_draft("r2v"),
            raylight_settings,
            "validate-raylight-u2-continuity",
        )
        compiled_results.append(raylight_continuity)
        bound_raylight_continuity = bind_native_workflow_predecessor_output(
            raylight_continuity.workflows[1], continuity_output
        )
        validate_native_workflow_ready(bound_raylight_continuity)
        if not await _validate(
            execution,
            "native-raylight-u2-continuity",
            bound_raylight_continuity.prompt,
        ):
            failures.append("raylight-u2-continuity")

        ray_lora = compile_native_timeline(
            _draft("t2v"), _raylight_lora_settings(), "validate-raylight-lora"
        )
        compiled_results.append(ray_lora)
        if not await _validate(
            execution, "native-raylight-lora", ray_lora.workflows[0].prompt
        ):
            failures.append("raylight-lora")

        # The runtime switch graph is server-generated after ordinary compile,
        # so validate its exact RAY_ACTORS wiring and RayKill input contract
        # explicitly against the installed registry. Cover both forms of the
        # persisted loader chain: without and with the optional RayLoraLoader.
        barrier_sources = (
            ("raylight-kill", raylight_without_lora),
            ("raylight-lora-kill", ray_lora.workflows[0]),
        )
        for label, source_unit in barrier_sources:
            if source_unit is None:
                failures.append(f"{label}-source")
                continue
            descriptor = raylight_runtime_descriptor(source_unit)
            if descriptor is None:
                failures.append(f"{label}-descriptor")
                continue
            barrier = build_raylight_shutdown_unit(
                descriptor, unit_id=f"validate-{label}"
            )
            if not await _validate(execution, f"native-{label}", barrier.prompt):
                failures.append(label)

    turbo_loaded = await nodes.load_custom_node(
        str(TURBO_LORA_ROOT), set(nodes.NODE_CLASS_MAPPINGS), module_parent="custom_nodes"
    )
    if not turbo_loaded:
        print("FAIL standard-dedicated-lora: Turbo LoRA node failed to register")
        failures.append("standard-dedicated-lora")
    else:
        dedicated = compile_native_timeline(
            _draft("t2v"), _dedicated_lora_settings(), "validate-dedicated-lora"
        )
        compiled_results.append(dedicated)
        if not await _validate(
            execution, "native-standard-dedicated-lora", dedicated.workflows[0].prompt
        ):
            failures.append("standard-dedicated-lora")

    for loader, label in (
        ("model_only", "model-only"),
        ("bypass_model_only", "bypass-model-only"),
    ):
        standard_lora = compile_native_timeline(
            _draft("t2v"),
            _standard_lora_settings(loader),
            f"validate-{label}-lora",
        )
        compiled_results.append(standard_lora)
        if not await _validate(
            execution,
            f"native-standard-{label}-lora",
            standard_lora.workflows[0].prompt,
        ):
            failures.append(f"standard-{label}-lora")

    provenance_failures = _validate_registry_provenance(nodes, compiled_results)
    failures.extend(f"provenance:{item}" for item in provenance_failures)
    initializer_contract_failures = _validate_raylight_initializer_contract(nodes)
    failures.extend(
        f"raylight-initializer:{item}" for item in initializer_contract_failures
    )
    attention_failures = _validate_compiled_raylight_attention(compiled_results)
    failures.extend(f"raylight-attention:{item}" for item in attention_failures)
    beta_scheduler_failures = _validate_beta_scheduler_contract(nodes)
    failures.extend(
        f"beta-scheduler:{item}" for item in beta_scheduler_failures
    )

    if failures:
        print(f"Validation failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(
        "All six native standard prompts and all six RayLight U2 prompts passed "
        "ComfyUI validation, including dynamically bound continuity prompts "
        "for both backends (no model execution)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(validate_all()))
