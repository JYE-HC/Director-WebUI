from __future__ import annotations

"""Server-owned MiniMax H3 workflow templates.

The browser submits timeline data, never a ComfyUI prompt.  This module is the
only place that turns that data into API-format ``class_type + inputs`` graphs.
Conditioning, media IO and decode/save are stock ComfyUI nodes.  RayLight is an
approved, narrow substitution for the model/sampler path when a model family
is assigned a multi-GPU topology.
"""

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import Any, Literal

from .compiler import unified_continuity_predecessors
from .schemas import (
    AssetReference,
    DiffusionModelBinding,
    RuntimeSettings,
    SamplingConfig,
    StandardLoraLoader,
    UnifiedFL2VASegment,
    UnifiedRef2VASegment,
    UnifiedTimelineDraft,
    UnifiedTimelineSegment,
    timeline_segment_recipe,
)


ModelFamily = Literal["fl2va", "ref2va"]
ExecutionBackend = Literal["standard", "raylight"]
ContinuitySource = Literal["same_run", "historical_take"]

# Director always assigns RayLight an explicit local logical GPU pool.  Keep
# Comfy-managed CLIP/VAE models on every other logical GPU warm by asking the
# installed RayLight initializer to release driver models only on that pool.
# This is deliberately a fixed template input rather than another user-facing
# setting: the RayLight node's missing-input default remains ``legacy_all`` for
# old third-party workflows, while Director must never silently regain that
# endpoint-global cleanup behavior.
_RAYLIGHT_DRIVER_CLEANUP_POLICY = "ray_devices"
# Director's RayLight Ulysses templates use comfy-kitchen's INT8 attention
# adapter by default. Keep this as one server-owned value because it is also
# part of the actor namespace fingerprint below: changing attention backends
# must never reuse a pool initialized with a different kernel contract.
_RAYLIGHT_XFUSER_ATTENTION = "COMFY_KITCHEN_INT8"
# Per-worker LRU cap for base models offloaded to CPU RAM on a model switch
# (installed RayLight fork feature). 2 models at ~37GB per GPU on 48GB cards
# keeps both resident in RAM across A<->B switches; raise it only after
# accounting for world_size x model_bytes per cached model.
_RAYLIGHT_RAM_CACHE_MAX_MODELS = 2


class NativeTemplateError(ValueError):
    """The timeline cannot be represented by the validated native templates."""


_UNBOUND_PREDECESSOR_OUTPUT = (
    "__DIRECTOR_UNBOUND_PREDECESSOR_OUTPUT__.mp4 [output]"
)


@dataclass(frozen=True)
class NativeContinuityDependency:
    """One output edge which must be resolved after its predecessor succeeds."""

    predecessor_segment_id: str
    overlap_frames: int
    load_video_node_id: str
    source: ContinuitySource = "same_run"
    historical_take_id: str | None = None
    resolved: bool = False


@dataclass(frozen=True)
class NativeHistoricalTake:
    """Server-resolved persisted output for one authored predecessor."""

    id: str
    segment_id: str
    output: Mapping[str, Any]


@dataclass(frozen=True)
class NativeWorkflowUnit:
    """One independently submitted ComfyUI prompt.

    RayLight owns process-global Ray state, therefore the parent orchestrator
    must submit RayLight units serially and must not overlap them with another
    RayLight topology on the same ComfyUI endpoint.
    """

    id: str
    family: ModelFamily
    backend: ExecutionBackend
    segment_ids: tuple[str, ...]
    prompt: dict[str, Any]
    output_nodes: dict[str, str]
    continuity: NativeContinuityDependency | None = None


@dataclass(frozen=True)
class NativeCompileResult:
    workflows: tuple[NativeWorkflowUnit, ...]
    manifest: dict[str, Any]
    plans: tuple[dict[str, Any], ...]
    families: tuple[ModelFamily, ...]
    node_policy: dict[str, Any]


_H3_DEDICATED_TURBO_LORA = re.compile(
    r"^minimax_h3_turbo_v4_step600(?:_ema)?\.safetensors$", re.IGNORECASE
)
_H3_QUANTIZED_COMPAT_LORA = re.compile(
    r"^minimax_h3_fl2v_turbo_(?:8step_v1\.0|4step_v1\.0_768p)_"
    r"10erosmax_beta1_pruned_compat_v001_t8\.safetensors$",
    re.IGNORECASE,
)
_H3_COMFY_GENERIC_LORA = re.compile(
    r"^minimax_h3_(?:fl2v_turbo_(?:8step_v1\.0|4step_v1\.0_768p)|"
    r"ref2v_turbo_4step_v0\.1)_comfyui_bf16\.safetensors$",
    re.IGNORECASE,
)

_STANDARD_LORA_NODES: dict[StandardLoraLoader, str] = {
    "dedicated": "MiniMaxH3TurboLoRA",
    "bypass_model_only": "LoraLoaderBypassModelOnly",
    "model_only": "LoraLoaderModelOnly",
}

_PROVENANCE: dict[str, str] = {
    "UNETLoader": "comfy-core",
    "CLIPLoader": "comfy-core",
    "VAELoader": "comfy-core",
    "LoadImage": "comfy-core",
    "LoraLoaderModelOnly": "comfy-core",
    "VAEDecode": "comfy-core",
    "SelectModelDevice": "comfy-extras",
    "SelectCLIPDevice": "comfy-extras",
    "SelectVAEDevice": "comfy-extras",
    "LoadVideo": "comfy-extras",
    "Video Slice": "comfy-extras",
    "GetVideoComponents": "comfy-extras",
    "LoadAudio": "comfy-extras",
    "MiniMaxH3ImageToVideo": "comfy-core-official-minimax-h3",
    "MiniMaxH3ReferenceToVideo": "comfy-core-official-minimax-h3",
    "MiniMaxH3AddGuide": "comfy-core-official-minimax-h3",
    "MiniMaxH3SigmaShift": "comfy-core-official-minimax-h3",
    "BasicGuider": "comfy-extras",
    "BasicScheduler": "comfy-extras",
    "KSamplerSelect": "comfy-extras",
    "RandomNoise": "comfy-extras",
    "SamplerCustomAdvanced": "comfy-extras",
    "VAEDecodeAudio": "comfy-extras",
    "TrimAudioDuration": "comfy-extras",
    "CreateVideo": "comfy-extras",
    "SaveVideo": "comfy-extras",
    "ImageFromBatch": "comfy-extras",
    "MiniMaxH3TurboLoRA": "lora-custom",
    "LoraLoaderBypassModelOnly": "comfy-extras",
    "RayInitializerAdvanced": "raylight",
    "RayLoraLoader": "raylight",
    "RayUNETLoader": "raylight",
    "RayMiniMaxH3SigmaShift": "raylight",
    "RayBasicGuider": "raylight",
    "RayBasicScheduler": "raylight",
    "XFuserSamplerCustomAdvanced": "raylight",
    "RayKill": "raylight",
}

EXPECTED_NATIVE_NODE_MODULES: dict[str, str] = {
    "UNETLoader": "nodes",
    "CLIPLoader": "nodes",
    "VAELoader": "nodes",
    "LoadImage": "nodes",
    "LoraLoaderModelOnly": "nodes",
    "VAEDecode": "nodes",
    "SelectModelDevice": "comfy_extras.nodes_multigpu",
    "SelectCLIPDevice": "comfy_extras.nodes_multigpu",
    "SelectVAEDevice": "comfy_extras.nodes_multigpu",
    "LoadVideo": "comfy_extras.nodes_video",
    "Video Slice": "comfy_extras.nodes_video",
    "GetVideoComponents": "comfy_extras.nodes_video",
    "LoadAudio": "comfy_extras.nodes_audio",
    "MiniMaxH3ImageToVideo": "comfy_extras.nodes_minimax_h3",
    "MiniMaxH3ReferenceToVideo": "comfy_extras.nodes_minimax_h3",
    "MiniMaxH3AddGuide": "comfy_extras.nodes_minimax_h3",
    "MiniMaxH3SigmaShift": "comfy_extras.nodes_minimax_h3",
    "BasicGuider": "comfy_extras.nodes_custom_sampler",
    "BasicScheduler": "comfy_extras.nodes_custom_sampler",
    "KSamplerSelect": "comfy_extras.nodes_custom_sampler",
    "RandomNoise": "comfy_extras.nodes_custom_sampler",
    "SamplerCustomAdvanced": "comfy_extras.nodes_custom_sampler",
    "VAEDecodeAudio": "comfy_extras.nodes_audio",
    "TrimAudioDuration": "comfy_extras.nodes_audio",
    "CreateVideo": "comfy_extras.nodes_video",
    "SaveVideo": "comfy_extras.nodes_video",
    "ImageFromBatch": "comfy_extras.nodes_images",
    "MiniMaxH3TurboLoRA": "custom_nodes.ComfyUI-MiniMax-H3-Turbo",
    "LoraLoaderBypassModelOnly": "comfy_extras.nodes_lora_debug",
    "RayInitializerAdvanced": "custom_nodes.raylight",
    "RayLoraLoader": "custom_nodes.raylight",
    "RayUNETLoader": "custom_nodes.raylight",
    "RayMiniMaxH3SigmaShift": "custom_nodes.raylight",
    "RayBasicGuider": "custom_nodes.raylight",
    "RayBasicScheduler": "custom_nodes.raylight",
    "XFuserSamplerCustomAdvanced": "custom_nodes.raylight",
    "RayKill": "custom_nodes.raylight",
}

if set(EXPECTED_NATIVE_NODE_MODULES) != set(_PROVENANCE):
    raise AssertionError("native node provenance and exact module policies diverged")

_RAYLIGHT_REQUIRED = frozenset(
    {
        "RayInitializerAdvanced",
        "RayUNETLoader",
        "RayMiniMaxH3SigmaShift",
        "RayBasicGuider",
        "RayBasicScheduler",
        "XFuserSamplerCustomAdvanced",
    }
)


class _Graph:
    def __init__(self) -> None:
        self.prompt: dict[str, Any] = {}
        self._counter = 0

    def add(self, class_type: str, **inputs: Any) -> str:
        self._counter += 1
        node_id = str(self._counter)
        self.prompt[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id


def _edge(node_id: str, output: int = 0) -> list[Any]:
    return [node_id, output]


def _align_h3_frame_count(raw_frames: int) -> int:
    frames = max(5, raw_frames)
    frames += (5 - frames % 17) % 17
    if frames > 512:
        raise NativeTemplateError(
            f"segment compiles to {frames} frames; native H3 template limit is 512"
        )
    return frames


def _align_h3_frames(duration_seconds: float, fps: float) -> int:
    raw = max(5, int(round(duration_seconds * fps)))
    return _align_h3_frame_count(raw)


def _annotated_predecessor_output(output: Mapping[str, Any]) -> str:
    """Return a canonical Comfy ``[output]`` path from history metadata."""

    filename = output.get("filename")
    subfolder = output.get("subfolder", "")
    output_type = output.get("type")
    if output_type != "output":
        raise NativeTemplateError(
            "continuity predecessor must be a persisted ComfyUI output"
        )
    if not isinstance(filename, str) or not filename or len(filename) > 512:
        raise NativeTemplateError("continuity predecessor filename is invalid")
    if not isinstance(subfolder, str) or len(subfolder) > 512:
        raise NativeTemplateError("continuity predecessor subfolder is invalid")
    if (
        filename != filename.strip()
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "[" in filename
        or "]" in filename
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in filename
        )
    ):
        raise NativeTemplateError("continuity predecessor filename is unsafe")
    if (
        subfolder != subfolder.strip()
        or "\\" in subfolder
        or "[" in subfolder
        or "]" in subfolder
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in subfolder
        )
    ):
        raise NativeTemplateError("continuity predecessor subfolder is unsafe")
    folder = PurePosixPath(subfolder)
    if folder.is_absolute() or any(
        part in {"", ".", ".."} for part in folder.parts
    ):
        raise NativeTemplateError("continuity predecessor subfolder is unsafe")
    relative = PurePosixPath(filename) if not subfolder else folder / filename
    return f"{relative.as_posix()} [output]"


def bind_native_workflow_predecessor_output(
    unit: NativeWorkflowUnit,
    output: Mapping[str, Any],
) -> NativeWorkflowUnit:
    """Purely bind one successor graph to its predecessor's SaveVideo output.

    The caller must first resolve the unique history output belonging to the
    predecessor unit's declared SaveVideo node. This helper deliberately
    accepts only that single descriptor, never a free-form annotated path.
    """

    dependency = unit.continuity
    if dependency is None:
        raise NativeTemplateError(
            f"native workflow '{unit.id}' has no continuity predecessor"
        )
    if dependency.resolved:
        raise NativeTemplateError(
            f"native workflow '{unit.id}' continuity predecessor is already bound"
        )
    prompt = deepcopy(unit.prompt)
    node = prompt.get(dependency.load_video_node_id)
    if not isinstance(node, dict) or node.get("class_type") != "LoadVideo":
        raise NativeTemplateError(
            f"native workflow '{unit.id}' continuity LoadVideo node is invalid"
        )
    inputs = node.get("inputs")
    if (
        not isinstance(inputs, dict)
        or inputs.get("file") != _UNBOUND_PREDECESSOR_OUTPUT
    ):
        raise NativeTemplateError(
            f"native workflow '{unit.id}' continuity placeholder was modified"
        )
    inputs["file"] = _annotated_predecessor_output(output)
    return replace(
        unit,
        prompt=prompt,
        continuity=replace(dependency, resolved=True),
    )


def normalize_native_output_descriptor(
    output: Mapping[str, Any],
) -> dict[str, str]:
    """Validate and normalize one persistent ComfyUI output descriptor."""

    _annotated_predecessor_output(output)
    return {
        "filename": str(output["filename"]),
        "subfolder": str(output.get("subfolder") or ""),
        "type": "output",
    }


def validate_native_workflow_ready(unit: NativeWorkflowUnit) -> None:
    """Fail closed unless every dynamic predecessor input has been bound."""

    placeholder_nodes = [
        str(node_id)
        for node_id, node in unit.prompt.items()
        if isinstance(node, dict)
        and node.get("class_type") == "LoadVideo"
        and isinstance(node.get("inputs"), dict)
        and node["inputs"].get("file") == _UNBOUND_PREDECESSOR_OUTPUT
    ]
    dependency = unit.continuity
    if dependency is None:
        if placeholder_nodes:
            raise NativeTemplateError(
                f"native workflow '{unit.id}' contains an undeclared continuity input"
            )
        return
    if not dependency.resolved:
        raise NativeTemplateError(
            f"native workflow '{unit.id}' is waiting for predecessor segment "
            f"'{dependency.predecessor_segment_id}'"
        )
    node = unit.prompt.get(dependency.load_video_node_id)
    if (
        not isinstance(node, dict)
        or node.get("class_type") != "LoadVideo"
        or not isinstance(node.get("inputs"), dict)
        or not isinstance(node["inputs"].get("file"), str)
        or not node["inputs"]["file"].endswith(" [output]")
        or placeholder_nodes
    ):
        raise NativeTemplateError(
            f"native workflow '{unit.id}' has an invalid bound predecessor input"
        )


def _asset_path(asset: AssetReference) -> str:
    return asset.comfy_path


def _require_h3_video_asset(asset: AssetReference, *, usage: str) -> None:
    metadata = asset.metadata
    if metadata is None:
        raise NativeTemplateError(f"{usage} requires server-probed video metadata")
    if abs(metadata.native_fps - 24.0) > 0.01:
        raise NativeTemplateError(
            f"{usage} '{asset.name}' is {metadata.native_fps:.6g} fps; "
            "MiniMax H3 reference frames require a server-created 24 fps proxy"
        )
    if metadata.frame_count < 5:
        raise NativeTemplateError(
            f"{usage} '{asset.name}' contains {metadata.frame_count} frame(s); "
            "MiniMax H3 reference video requires at least 5 frames"
        )


def _require_h3_source_range(
    asset: AssetReference, *, start: float, duration: float, usage: str
) -> None:
    """Reject trims that stock ReferenceToVideo cannot condition on."""

    _require_h3_video_asset(asset, usage=usage)
    assert asset.metadata is not None
    full_frames = max(1, int(round(asset.metadata.duration * 24.0)))
    source_start = min(full_frames - 1, max(0, int(round(start * 24.0))))
    source_end = min(
        full_frames,
        max(source_start + 1, int(round((start + duration) * 24.0))),
    )
    selected_frames = source_end - source_start
    if selected_frames < 5:
        raise NativeTemplateError(
            f"{usage} '{asset.name}' selects {selected_frames} frame(s); "
            "MiniMax H3 reference video requires at least 5 frames"
        )


def _known_standard_lora_loader(basename: str) -> StandardLoraLoader | None:
    """Return the audited loader for a known upstream artifact basename."""

    if _H3_DEDICATED_TURBO_LORA.fullmatch(basename):
        return "dedicated"
    if _H3_QUANTIZED_COMPAT_LORA.fullmatch(basename):
        return "bypass_model_only"
    if _H3_COMFY_GENERIC_LORA.fullmatch(basename):
        return "model_only"
    return None


def _metadata_standard_lora_loader(
    binding: DiffusionModelBinding,
    basename: str,
    metadata: Mapping[str, Any] | None,
) -> StandardLoraLoader:
    """Infer a Standard loader from ComfyUI-served safetensors metadata.

    Director may run on a different host from ComfyUI, so it cannot inspect
    model files directly.  ComfyUI's read-only ``/view_metadata/loras`` route
    is the authority for an unknown artifact.  Only explicit, audited metadata
    contracts are accepted; an absent or ambiguous header remains fail-closed
    and can be resolved with the visible per-binding override.
    """

    if metadata is not None:
        target_format = metadata.get("target_format")
        if (
            isinstance(target_format, str)
            and target_format.strip().casefold() == "comfyui generic lora"
        ):
            compatible_main = metadata.get("compatible_main_file")
            if isinstance(compatible_main, str) and compatible_main.strip():
                selected_base = PurePosixPath(
                    binding.filename.replace("\\", "/")
                ).name
                compatible_base = PurePosixPath(
                    compatible_main.replace("\\", "/")
                ).name
                if selected_base != compatible_base:
                    raise NativeTemplateError(
                        f"LoRA '{basename}' declares exact compatibility with "
                        f"'{compatible_base}', not selected model "
                        f"'{selected_base}'"
                    )
                compatibility_scope = metadata.get("compatibility_scope")
                compatible_sha256 = metadata.get("compatible_main_sha256")
                if (
                    compatibility_scope == "exact_checkpoint_sha256_validated"
                    and isinstance(compatible_sha256, str)
                    and re.fullmatch(r"[0-9a-fA-F]{64}", compatible_sha256)
                ):
                    return "bypass_model_only"
                raise NativeTemplateError(
                    f"LoRA '{basename}' has checkpoint-specific compatibility "
                    "metadata but does not declare whether Standard should use "
                    "the generic or bypass loader; choose it explicitly in "
                    "Settings"
                )
            return "model_only"

        application = metadata.get("application")
        base_model = metadata.get("base_model")
        if (
            isinstance(application, str)
            and application.strip() == "W_eff = W + lora_B @ lora_A"
            and isinstance(base_model, str)
            and base_model.strip().casefold() == "minimax-h3"
        ):
            return "dedicated"

    raise NativeTemplateError(
        f"LoRA loader cannot be inferred safely from '{basename}': ComfyUI "
        "did not expose a supported safetensors metadata contract; choose a "
        "Standard LoRA loader explicitly in Settings"
    )


def _resolve_standard_lora(
    binding: DiffusionModelBinding,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    if binding.lora_name is None:
        return None
    if binding.standard_lora_loader_override is not None:
        return _STANDARD_LORA_NODES[
            binding.standard_lora_loader_override.loader
        ]
    basename = PurePosixPath(binding.lora_name.replace("\\", "/")).name
    loader = _known_standard_lora_loader(basename)
    if loader is None:
        loader = _metadata_standard_lora_loader(binding, basename, metadata)
    return _STANDARD_LORA_NODES[loader]


def resolve_execution_backend(binding: DiffusionModelBinding) -> ExecutionBackend:
    """Derive the sole execution route from the configured logical GPU pool.

    The legacy ``binding.backend`` value is intentionally ignored.  Keeping a
    hidden explicit value after removing its UI would let an old browser or
    database silently override the visible GPU selection.  Capability probing
    still verifies the exact derived graph before submission and never falls
    back to another backend when a required node is missing.
    """

    return "raylight" if len(binding.raylight.gpu_select) >= 2 else "standard"


def required_nodes_for_backend(
    backend: ExecutionBackend, *, has_lora: bool = False
) -> frozenset[str]:
    nodes = set(_RAYLIGHT_REQUIRED if backend == "raylight" else ())
    if backend == "raylight" and has_lora:
        nodes.add("RayLoraLoader")
    return frozenset(nodes)


def _shared_core(graph: _Graph, settings: RuntimeSettings) -> dict[str, list[Any]]:
    clip_loader = graph.add(
        "CLIPLoader",
        clip_name=settings.models.clip.filename,
        type="minimax",
        device="default",
    )
    clip = graph.add(
        "SelectCLIPDevice",
        clip=_edge(clip_loader),
        device=settings.models.clip.device,
    )
    video_vae_loader = graph.add(
        "VAELoader", vae_name=settings.models.video_vae.filename
    )
    video_vae = graph.add(
        "SelectVAEDevice",
        vae=_edge(video_vae_loader),
        device=settings.models.video_vae.device,
    )
    audio_vae_loader = graph.add(
        "VAELoader", vae_name=settings.models.audio_vae.filename
    )
    audio_vae = graph.add(
        "SelectVAEDevice",
        vae=_edge(audio_vae_loader),
        device=settings.models.audio_vae.device,
    )
    return {
        "clip": _edge(clip),
        "video_vae": _edge(video_vae),
        "audio_vae": _edge(audio_vae),
    }


def _standard_model(
    graph: _Graph,
    binding: DiffusionModelBinding,
    sampling: SamplingConfig,
    *,
    lora_metadata: Mapping[str, Any] | None = None,
) -> list[Any]:
    loader = graph.add(
        "UNETLoader", unet_name=binding.filename, weight_dtype="default"
    )
    selected = graph.add(
        "SelectModelDevice", model=_edge(loader), device=binding.device
    )
    model = _edge(selected)
    lora_node = _resolve_standard_lora(binding, lora_metadata)
    if lora_node is not None:
        assert binding.lora_name is not None
        if lora_node == "MiniMaxH3TurboLoRA":
            lora_inputs = {
                "model": model,
                "lora_name": binding.lora_name,
                "strength": binding.lora_strength,
                # Loader selection is automatic, so its old loader-specific
                # tuning flag cannot remain as an invisible behavior switch.
                "low_vram": False,
            }
        else:
            lora_inputs = {
                "model": model,
                "lora_name": binding.lora_name,
                "strength_model": binding.lora_strength,
            }
        model = _edge(graph.add(lora_node, **lora_inputs))
    shifted = graph.add(
        "MiniMaxH3SigmaShift",
        model=model,
        shift_video=sampling.shift,
        shift_audio=sampling.audio_shift,
    )
    return _edge(shifted)


def _raylight_model(
    graph: _Graph,
    binding: DiffusionModelBinding,
    sampling: SamplingConfig,
    *,
    namespace: str,
    clear_vram_after_sampling: bool,
) -> list[Any]:
    profile = binding.raylight
    if len(profile.gpu_select) < 2:
        raise NativeTemplateError("RayLight workflow needs at least two logical GPUs")
    if profile.fsdp or profile.cpu_offload:
        raise NativeTemplateError(
            "RayLight FSDP/CPU offload is disabled in native timeline v1 until "
            "its post-sampling CUDA cleanup is verified"
        )
    initializer = graph.add(
        "RayInitializerAdvanced",
        ray_cluster_address="local",
        ray_cluster_namespace=namespace,
        GPU=len(profile.gpu_select),
        GPU_SELECT=",".join(str(index) for index in profile.gpu_select),
        driver_cleanup_policy=_RAYLIGHT_DRIVER_CLEANUP_POLICY,
        ulysses_degree=profile.ulysses_degree,
        ring_degree=profile.ring_degree,
        # The installed RayLight workers have an exact model + LoRA cache key.
        # Keeping this false preserves their CUDA weights across stable
        # per-segment prompts. Director handles a later incompatible Ray or
        # Standard prompt with an explicit, awaited RayKill transition because
        # driver-side Comfy model management cannot evict allocations owned by
        # the separate Ray worker processes.
        clear_vram_after_sampling=clear_vram_after_sampling,
        ram_cache_max_models=_RAYLIGHT_RAM_CACHE_MAX_MODELS,
        cfg_degree=profile.cfg_degree,
        dp_degree=profile.dp_degree,
        sync_ulysses=False,
        FSDP=profile.fsdp,
        FSDP_CPU_OFFLOAD=profile.cpu_offload,
        XFuser_attention=_RAYLIGHT_XFUSER_ATTENTION,
        skip_comm_test=False,
        use_mmap=False,
    )
    lora: list[Any] | None = None
    if binding.lora_name is not None:
        lora = _edge(
            graph.add(
                "RayLoraLoader",
                lora_name=binding.lora_name,
                strength_model=binding.lora_strength,
            )
        )
    loader_inputs: dict[str, Any] = {
        "ray_actors_init": _edge(initializer),
        "unet_name": binding.filename,
        "weight_dtype": "default",
    }
    if lora is not None:
        loader_inputs["lora"] = lora
    actors = graph.add("RayUNETLoader", **loader_inputs)
    shifted = graph.add(
        "RayMiniMaxH3SigmaShift",
        ray_actors=_edge(actors),
        shift_video=sampling.shift,
        shift_audio=sampling.audio_shift,
    )
    return _edge(shifted)


def _raylight_namespace(
    family: ModelFamily, binding: DiffusionModelBinding
) -> str:
    """Return a stable actor namespace for one compatible resident pool.

    A job identifier must not participate in this key: doing so would force
    ``RayInitializerAdvanced`` to tear down and recreate the same actors for
    every take.  Only the GPU pool and topology participate.  Model file, LoRA
    and family deliberately do NOT participate: the installed worker RAM cache
    moves the outgoing model to CPU RAM on a key change and re-activates it
    from RAM when the key comes back, so a model switch keeps the pool alive
    and never re-reads the checkpoint.  The installed initializer still calls
    ``ray.shutdown()`` whenever this input changes (topology change), giving
    an explicit old-pool teardown for a genuinely incompatible pool.
    """

    profile = binding.raylight
    gpu_pool = "-".join(str(index) for index in profile.gpu_select)
    compatibility_document = {
        "backend": "raylight",
        "weight_dtype": "default",
        "gpu_select": profile.gpu_select,
        "driver_cleanup_policy": _RAYLIGHT_DRIVER_CLEANUP_POLICY,
        "ulysses_degree": profile.ulysses_degree,
        "ring_degree": profile.ring_degree,
        "cfg_degree": profile.cfg_degree,
        "dp_degree": profile.dp_degree,
        "fsdp": profile.fsdp,
        "cpu_offload": profile.cpu_offload,
        "attention": _RAYLIGHT_XFUSER_ATTENTION,
        "use_mmap": False,
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            compatibility_document,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    return (
        f"director-g{gpu_pool}"
        f"-u{profile.ulysses_degree}-r{profile.ring_degree}"
        f"-c{profile.cfg_degree}-d{profile.dp_degree}"
        f"-f{int(profile.fsdp)}-o{int(profile.cpu_offload)}"
        f"-k{fingerprint}"
    )


def raylight_runtime_descriptor(unit: NativeWorkflowUnit) -> dict[str, Any] | None:
    """Return the exact cached loader chain needed for a later kill.

    ``RayKill`` consumes ``RAY_ACTORS`` while ``RayInitializerAdvanced`` only
    produces ``RAY_ACTORS_INIT``.  Persisting just the initializer would make
    the transition graph type-invalid.  The descriptor therefore owns the
    complete, minimal initializer -> optional LoRA -> UNET-loader subgraph.
    Reusing the original node ids and inputs lets ComfyUI return the live
    loader output from cache; if that cache is gone, the same subgraph safely
    reconstructs a pool which the barrier immediately tears down.
    """

    matches = [
        (node_id, node)
        for node_id, node in unit.prompt.items()
        if node.get("class_type") == "RayInitializerAdvanced"
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' must contain exactly one initializer"
        )
    node_id, node = matches[0]
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' has invalid initializer inputs"
        )
    runtime_namespace = str(inputs.get("ray_cluster_namespace") or "")
    if not runtime_namespace:
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' has an empty runtime namespace"
        )
    compatibility_key = re.sub(r"-e[1-9][0-9]*$", "", runtime_namespace)

    loader_matches = [
        (candidate_id, candidate)
        for candidate_id, candidate in unit.prompt.items()
        if candidate.get("class_type") == "RayUNETLoader"
    ]
    if len(loader_matches) != 1:
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' must contain exactly one UNET loader"
        )
    loader_node_id, loader_node = loader_matches[0]
    loader_inputs = loader_node.get("inputs")
    if not isinstance(loader_inputs, dict):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' has invalid UNET loader inputs"
        )
    if loader_inputs.get("ray_actors_init") != _edge(str(node_id)):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' loader is not bound to its initializer"
        )

    dependency_ids = {str(node_id), str(loader_node_id)}
    lora_edge = loader_inputs.get("lora")
    if lora_edge is not None:
        if (
            not isinstance(lora_edge, list)
            or len(lora_edge) != 2
            or not isinstance(lora_edge[0], str)
        ):
            raise NativeTemplateError(
                f"RayLight unit '{unit.id}' has an invalid LoRA edge"
            )
        lora_node = unit.prompt.get(lora_edge[0])
        if not isinstance(lora_node, dict) or lora_node.get("class_type") != "RayLoraLoader":
            raise NativeTemplateError(
                f"RayLight unit '{unit.id}' loader has an invalid LoRA dependency"
            )
        dependency_ids.add(lora_edge[0])

    loader_subgraph = {
        dependency_id: deepcopy(unit.prompt[dependency_id])
        for dependency_id in dependency_ids
    }
    # The runtime key deliberately excludes the loader inputs (model file and
    # LoRA): the workers own a per-worker RAM cache and swap bases in place,
    # so a model switch keeps the pool and its epoch.  The key still crosses
    # the RayKill/epoch boundary on namespace (topology) changes and on sigma
    # mutations, which the workers cannot absorb safely.
    # RayMiniMaxH3SigmaShift mutates each worker's ModelPatcher in place. Its
    # cached output is therefore valid only for the exact shift pair: in
    # A(12) -> B(8) -> A(12), ComfyUI could otherwise return A's old output
    # while the shared actors still hold B's mutation. Treat a shift change as
    # an incompatible runtime key and cross the normal RayKill/epoch boundary.
    sigma_matches = [
        candidate
        for candidate in unit.prompt.values()
        if candidate.get("class_type") == "RayMiniMaxH3SigmaShift"
    ]
    # Shutdown units intentionally contain only the loader chain plus RayKill;
    # their descriptor was already persisted from a complete generation unit.
    is_shutdown_unit = any(
        candidate.get("class_type") == "RayKill"
        for candidate in unit.prompt.values()
    )
    if is_shutdown_unit:
        return None
    if len(sigma_matches) != 1 or not isinstance(sigma_matches[0].get("inputs"), dict):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' must contain exactly one sigma-shift node"
        )
    sigma_inputs = sigma_matches[0]["inputs"]
    # Pool-level inputs of the initializer participate (minus the namespace,
    # which is the canonical compatibility_key). This deliberately includes
    # XFuser_attention: a legacy TORCH_FLASH pool must cross RayKill before a
    # COMFY_KITCHEN_INT8 prompt can start. The model file and LoRA live in the
    # loader node and deliberately do not, so a model switch keeps the same
    # runtime key and reuses the pool via the worker RAM cache.
    runtime_identity = {
        "__initializer_inputs__": {
            key: value
            for key, value in inputs.items()
            if key != "ray_cluster_namespace"
        },
        "__runtime_mutations__": {
            "RayMiniMaxH3SigmaShift": {
                "shift_video": sigma_inputs.get("shift_video"),
                "shift_audio": sigma_inputs.get("shift_audio"),
            }
        },
    }
    runtime_key = hashlib.sha256(
        json.dumps(
            runtime_identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "version": 2,
        "family": unit.family,
        "compatibility_key": compatibility_key,
        "runtime_key": runtime_key,
        "runtime_namespace": runtime_namespace,
        "initializer_node_id": str(node_id),
        "loader_node_id": str(loader_node_id),
        "loader_subgraph": loader_subgraph,
        "clear_vram_after_sampling": bool(
            inputs.get("clear_vram_after_sampling", False)
        ),
    }


def _raylight_logical_gpu_indices(inputs: Mapping[str, Any]) -> tuple[int, ...]:
    """Parse Director's explicit logical GPU pool from one initializer.

    RayLight interprets ``GPU_SELECT`` inside the ComfyUI process CUDA
    namespace.  Persisted shutdown descriptors must keep that exact identity;
    silently dropping, remapping or deduplicating entries could target a
    different actor pool during a safety transition.
    """

    raw = inputs.get("GPU_SELECT")
    if not isinstance(raw, str) or not raw:
        raise NativeTemplateError(
            "persisted RayLight initializer GPU_SELECT is invalid"
        )
    indices: list[int] = []
    for token in raw.split(","):
        if not token or not token.isdecimal():
            raise NativeTemplateError(
                "persisted RayLight initializer GPU_SELECT is invalid"
            )
        index = int(token)
        if index in indices:
            raise NativeTemplateError(
                "persisted RayLight initializer GPU_SELECT contains duplicates"
            )
        indices.append(index)
    declared = inputs.get("GPU")
    if (
        not isinstance(declared, int)
        or isinstance(declared, bool)
        or declared != len(indices)
    ):
        raise NativeTemplateError(
            "persisted RayLight initializer GPU count does not match GPU_SELECT"
        )
    return tuple(indices)


def raylight_runtime_logical_gpu_indices(
    descriptor: Mapping[str, Any],
) -> tuple[int, ...]:
    """Return the exact logical GPU pool recorded in a runtime descriptor."""

    initializer_node_id = descriptor.get("initializer_node_id")
    loader_subgraph = descriptor.get("loader_subgraph")
    if (
        descriptor.get("version") != 2
        or not isinstance(initializer_node_id, str)
        or not isinstance(loader_subgraph, Mapping)
    ):
        raise NativeTemplateError("persisted RayLight loader identity is invalid")
    initializer = loader_subgraph.get(initializer_node_id)
    if (
        not isinstance(initializer, Mapping)
        or initializer.get("class_type") != "RayInitializerAdvanced"
        or not isinstance(initializer.get("inputs"), Mapping)
    ):
        raise NativeTemplateError("persisted RayLight initializer is invalid")
    return _raylight_logical_gpu_indices(initializer["inputs"])


def raylight_workflow_logical_gpu_indices(
    unit: NativeWorkflowUnit,
) -> tuple[int, ...]:
    """Return the explicit logical GPU pool in one RayLight workflow unit."""

    initializers = [
        node
        for node in unit.prompt.values()
        if isinstance(node, Mapping)
        and node.get("class_type") == "RayInitializerAdvanced"
    ]
    if len(initializers) != 1 or not isinstance(initializers[0].get("inputs"), Mapping):
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' must contain one valid initializer"
        )
    return _raylight_logical_gpu_indices(initializers[0]["inputs"])


def bind_raylight_runtime_epoch(
    unit: NativeWorkflowUnit, epoch: int
) -> NativeWorkflowUnit:
    """Bind a Ray unit to one persistent endpoint transition epoch."""

    if unit.backend != "raylight" or epoch < 1:
        raise NativeTemplateError("RayLight runtime epoch must be a positive integer")
    descriptor = raylight_runtime_descriptor(unit)
    if descriptor is None:
        raise NativeTemplateError(
            f"RayLight unit '{unit.id}' has no initializer to bind"
        )
    prompt = deepcopy(unit.prompt)
    initializer = prompt[str(descriptor["initializer_node_id"])]
    initializer["inputs"]["ray_cluster_namespace"] = (
        f"{descriptor['compatibility_key']}-e{epoch}"
    )
    return replace(unit, prompt=prompt)


def build_raylight_shutdown_unit(
    descriptor: dict[str, Any], *, unit_id: str
) -> NativeWorkflowUnit:
    """Build a forced RayKill barrier from a persisted resident descriptor.

    Reusing the original loader subgraph normally retrieves live ``RAY_ACTORS``
    from ComfyUI's cache. If that cache was cleared, running the initializer is
    still safe: RayLight first shuts down any current pool, creates a temporary
    compatible pool, loads the recorded model, and this prompt immediately
    kills it. The unique kill node id prevents ComfyUI from caching away the
    side effect.
    """

    if descriptor.get("version") != 2:
        raise NativeTemplateError("unsupported persisted RayLight runtime descriptor")
    family = descriptor.get("family")
    if family not in {"fl2va", "ref2va"}:
        raise NativeTemplateError("persisted RayLight runtime family is invalid")
    initializer_node_id = descriptor.get("initializer_node_id")
    loader_node_id = descriptor.get("loader_node_id")
    loader_subgraph = descriptor.get("loader_subgraph")
    if (
        not isinstance(initializer_node_id, str)
        or not initializer_node_id
        or not isinstance(loader_node_id, str)
        or not loader_node_id
        or not isinstance(loader_subgraph, dict)
    ):
        raise NativeTemplateError("persisted RayLight loader identity is invalid")
    node_types = {
        str(node.get("class_type") or "")
        for node in loader_subgraph.values()
        if isinstance(node, dict)
    }
    if (
        set(loader_subgraph) != {
            key for key in loader_subgraph if isinstance(key, str)
        }
        or initializer_node_id not in loader_subgraph
        or loader_node_id not in loader_subgraph
        or not node_types <= {
            "RayInitializerAdvanced",
            "RayLoraLoader",
            "RayUNETLoader",
        }
        or "RayInitializerAdvanced" not in node_types
        or "RayUNETLoader" not in node_types
    ):
        raise NativeTemplateError("persisted RayLight loader subgraph is invalid")
    initializer_node = loader_subgraph[initializer_node_id]
    loader_node = loader_subgraph[loader_node_id]
    if (
        not isinstance(initializer_node, dict)
        or initializer_node.get("class_type") != "RayInitializerAdvanced"
        or not isinstance(initializer_node.get("inputs"), dict)
        or not isinstance(loader_node, dict)
        or loader_node.get("class_type") != "RayUNETLoader"
        or not isinstance(loader_node.get("inputs"), dict)
        or loader_node["inputs"].get("ray_actors_init")
        != _edge(initializer_node_id)
    ):
        raise NativeTemplateError("persisted RayLight loader chain is invalid")
    kill_node_id = f"ray-kill-{unit_id}"
    if kill_node_id in loader_subgraph:
        raise NativeTemplateError("RayLight shutdown node id collides with loader graph")
    prompt = deepcopy(loader_subgraph)
    prompt[kill_node_id] = {
        "class_type": "RayKill",
        "inputs": {
            "ray_actors": _edge(loader_node_id),
            "kill_mode": "Kill Entire Cluster",
        },
    }
    return NativeWorkflowUnit(
        id=unit_id,
        family=family,
        backend="raylight",
        segment_ids=(),
        prompt=prompt,
        output_nodes={},
    )


def _load_image(graph: _Graph, asset: AssetReference) -> list[Any]:
    return _edge(graph.add("LoadImage", image=_asset_path(asset)))


def _load_video_components(
    graph: _Graph,
    asset: AssetReference,
    *,
    start: float | None = None,
    duration: float | None = None,
) -> tuple[list[Any], list[Any]]:
    if start is not None and duration is not None:
        _require_h3_source_range(
            asset,
            start=start,
            duration=duration,
            usage="native source-video conditioning",
        )
    else:
        _require_h3_video_asset(asset, usage="native video conditioning")
    video = graph.add("LoadVideo", file=_asset_path(asset))
    current = _edge(video)
    if start is not None and duration is not None:
        sliced = graph.add(
            "Video Slice",
            video=current,
            start_time=start,
            duration=duration,
            strict_duration=True,
        )
        current = _edge(sliced)
    components = graph.add("GetVideoComponents", video=current)
    return _edge(components, 0), _edge(components, 1)


def _load_audio(graph: _Graph, asset: AssetReference) -> list[Any]:
    return _edge(graph.add("LoadAudio", audio=_asset_path(asset)))


def _require_dense_slots(values: list[Any], *, field: str, segment_id: str) -> None:
    """Stock Autogrow assigns dense ordinals; never silently renumber prompts."""

    slots = sorted(value.slot for value in values)
    expected = list(range(len(values)))
    if slots != expected:
        raise NativeTemplateError(
            f"segment '{segment_id}' {field} slots must be dense {expected}; "
            f"got {slots}. Repair the slots and prompt tags explicitly."
        )


def _validate_native_reference_slots(segment: UnifiedTimelineSegment) -> None:
    if isinstance(segment, UnifiedRef2VASegment):
        _require_dense_slots(
            segment.reference_images,
            field="reference_images",
            segment_id=segment.id,
        )
        _require_dense_slots(
            segment.reference_videos,
            field="reference_videos",
            segment_id=segment.id,
        )
        _require_dense_slots(
            segment.reference_audios,
            field="reference_audios",
            segment_id=segment.id,
        )


def _conditioning(
    graph: _Graph,
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    shared: dict[str, list[Any]],
    *,
    frames: int,
) -> tuple[list[Any], list[Any], list[Any] | None]:
    prompt = segment.prompt.strip()
    source_audio: list[Any] | None = None
    common: dict[str, Any] = {
        "clip": shared["clip"],
        "vae": shared["video_vae"],
        "prompt": prompt,
        "width": draft.render.width,
        "height": draft.render.height,
        "length": frames,
    }
    if isinstance(segment, UnifiedFL2VASegment):
        if segment.first_image is not None:
            common["first_frame"] = _load_image(graph, segment.first_image)
        if segment.last_image is not None:
            common["last_frame"] = _load_image(graph, segment.last_image)
        node = graph.add("MiniMaxH3ImageToVideo", **common)
        return _edge(node, 0), _edge(node, 1), None

    common.update(
        audio_vae=shared["audio_vae"], ref_image_size=segment.ref_image_size
    )
    if not isinstance(segment, UnifiedRef2VASegment):
        raise NativeTemplateError(f"unsupported segment mode: {segment.mode}")
    if segment.source_video is None and not (
        segment.reference_images
        or segment.reference_audios
        or segment.reference_videos
    ):
        raise NativeTemplateError(
            f"Ref2VA segment '{segment.id}' requires source_video or independent "
            "reference media"
        )
    video_offset = 0
    if segment.source_video is not None:
        images, source_audio = _load_video_components(
            graph,
            segment.source_video,
            start=segment.source_start_seconds,
            duration=segment.source_duration_seconds,
        )
        common["ref_videos.ref_video_0"] = images
        video_offset = 1
        if segment.source_audio_as_reference:
            assert segment.source_video.metadata is not None
            if not segment.source_video.metadata.has_audio:
                raise NativeTemplateError(
                    f"Ref2VA segment '{segment.id}' enables "
                    "source_audio_as_reference, but its server-probed source "
                    "video has no audio stream"
                )
            # Stock MiniMaxH3ReferenceToVideo pairs this soundtrack with the
            # first reference video and assigns it <Audio 1> before any
            # independent ref_audios autogrow inputs.
            common["ref_video_audios.ref_video_audio_0"] = source_audio
    for dense, asset in enumerate(
        sorted(segment.reference_images, key=lambda item: item.slot)
    ):
        common[f"ref_images.ref_image_{dense}"] = _load_image(graph, asset)
    for dense, asset in enumerate(
        sorted(segment.reference_videos, key=lambda item: item.slot),
        start=video_offset,
    ):
        images, _audio = _load_video_components(graph, asset)
        common[f"ref_videos.ref_video_{dense}"] = images
    for dense, asset in enumerate(
        sorted(segment.reference_audios, key=lambda item: item.slot)
    ):
        common[f"ref_audios.ref_audio_{dense}"] = _load_audio(graph, asset)
    node = graph.add("MiniMaxH3ReferenceToVideo", **common)
    return _edge(node, 0), _edge(node, 1), source_audio


def _add_continuity_guides(
    graph: _Graph,
    *,
    conditioning: list[Any],
    latent: list[Any],
    segment: UnifiedTimelineSegment,
    draft: UnifiedTimelineDraft,
    shared: dict[str, list[Any]],
    visible_frames: int,
    overlap_frames: int,
) -> tuple[list[Any], str]:
    """Anchor the predecessor tail at frame zero of the successor sample."""

    predecessor_video = graph.add(
        "LoadVideo", file=_UNBOUND_PREDECESSOR_OUTPUT
    )
    components = graph.add(
        "GetVideoComponents", video=_edge(predecessor_video)
    )
    tail_images = graph.add(
        "ImageFromBatch",
        image=_edge(components, 0),
        batch_index=-overlap_frames,
        length=overlap_frames,
    )
    guide_inputs: dict[str, Any] = {
        "positive": conditioning,
        "latent": latent,
        "vae": shared["video_vae"],
        "image": _edge(tail_images),
        "frame_idx": 0,
    }
    if segment.audio_mode == "generate":
        tail_audio = graph.add(
            "TrimAudioDuration",
            audio=_edge(components, 1),
            start_index=-(overlap_frames / draft.render.fps),
            duration=overlap_frames / draft.render.fps,
        )
        guide_inputs.update(
            audio_vae=shared["audio_vae"],
            audio=_edge(tail_audio),
        )
    conditioning = _edge(graph.add("MiniMaxH3AddGuide", **guide_inputs))

    # ImageToVideo keeps last_frame in the Qwen presentation so FL2VA prompts
    # can resolve its <Picture N> label. A continuity sample also contains a
    # hidden aligned tail, so repeat the same image as a guide at the final
    # visible output frame; the node's implicit sample-end anchor is cropped.
    if isinstance(segment, UnifiedFL2VASegment) and segment.last_image is not None:
        conditioning = _edge(
            graph.add(
                "MiniMaxH3AddGuide",
                positive=conditioning,
                latent=latent,
                vae=shared["video_vae"],
                image=_load_image(graph, segment.last_image),
                frame_idx=overlap_frames + visible_frames - 1,
            )
        )
    return conditioning, predecessor_video


def _sample(
    graph: _Graph,
    *,
    backend: ExecutionBackend,
    model: list[Any],
    conditioning: list[Any],
    latent: list[Any],
    sampling: SamplingConfig,
    seed: int,
) -> list[Any]:
    if backend == "standard":
        guider = graph.add(
            "BasicGuider", model=model, conditioning=conditioning
        )
        scheduler = graph.add(
            "BasicScheduler",
            model=model,
            scheduler=sampling.scheduler,
            steps=sampling.steps,
            denoise=1.0,
        )
        sampler = graph.add(
            "KSamplerSelect", sampler_name=sampling.sampler
        )
        noise = graph.add("RandomNoise", noise_seed=seed)
        sampled = graph.add(
            "SamplerCustomAdvanced",
            noise=_edge(noise),
            guider=_edge(guider),
            sampler=_edge(sampler),
            sigmas=_edge(scheduler),
            latent_image=latent,
        )
        return _edge(sampled, 0)
    guider = graph.add(
        "RayBasicGuider", ray_actors=model, conditioning=conditioning
    )
    scheduler = graph.add(
        "RayBasicScheduler",
        ray_actors=model,
        scheduler=sampling.scheduler,
        steps=sampling.steps,
        denoise=1.0,
    )
    sampler = graph.add("KSamplerSelect", sampler_name=sampling.sampler)
    sampled = graph.add(
        "XFuserSamplerCustomAdvanced",
        add_noise=True,
        noise_seed=seed,
        guider=_edge(guider),
        sampler=_edge(sampler),
        sigmas=_edge(scheduler),
        latent_image=latent,
    )
    return _edge(sampled, 0)


def _decode_and_save(
    graph: _Graph,
    *,
    samples: list[Any],
    source_audio: list[Any] | None,
    draft: UnifiedTimelineDraft,
    shared: dict[str, list[Any]],
    job_id: str,
    segment: UnifiedTimelineSegment,
    visible_frames: int,
    continuity_prefix_frames: int,
) -> str:
    images = graph.add(
        "VAEDecode", samples=samples, vae=shared["video_vae"]
    )
    visible_images = _edge(images)
    if continuity_prefix_frames:
        visible_images = _edge(
            graph.add(
                "ImageFromBatch",
                image=visible_images,
                batch_index=continuity_prefix_frames,
                length=visible_frames,
            )
        )
    video_inputs: dict[str, Any] = {
        "images": visible_images,
        "fps": draft.render.fps,
        "bit_depth": 8,
    }
    if segment.audio_mode == "generate":
        audio = graph.add(
            "VAEDecodeAudio", samples=samples, vae=shared["audio_vae"]
        )
        visible_audio = _edge(audio)
        if continuity_prefix_frames:
            visible_audio = _edge(
                graph.add(
                    "TrimAudioDuration",
                    audio=visible_audio,
                    start_index=continuity_prefix_frames / draft.render.fps,
                    duration=visible_frames / draft.render.fps,
                )
            )
        video_inputs["audio"] = visible_audio
    elif segment.audio_mode == "source":
        if source_audio is None:
            raise NativeTemplateError(
                "source audio is available only for v2v/rv2v segments"
            )
        source_segment = segment
        if (
            not isinstance(source_segment, UnifiedRef2VASegment)
            or source_segment.source_video is None
        ):
            raise NativeTemplateError(
                "source audio is available only for v2v/rv2v segments"
            )
        assert source_segment.source_video is not None
        assert source_segment.source_video.metadata is not None
        if not source_segment.source_video.metadata.has_audio:
            raise NativeTemplateError(
                f"segment '{segment.id}' cannot use audio_mode='source': "
                "the server-probed source video has no audio stream"
            )
        video_inputs["audio"] = source_audio
    video = graph.add("CreateVideo", **video_inputs)
    safe_segment = re.sub(r"[^A-Za-z0-9_-]+", "_", segment.id)[:64]
    return graph.add(
        "SaveVideo",
        video=_edge(video),
        filename_prefix=(
            f"video/Director_timeline_{job_id[:8]}_{safe_segment}"
        ),
        format="auto",
        codec="auto",
    )


def _selected_segments(
    draft: UnifiedTimelineDraft, segment_ids: list[str] | None
) -> list[UnifiedTimelineSegment]:
    enabled = [segment for segment in draft.segments if segment.enabled]
    by_id = {segment.id: segment for segment in enabled}
    if segment_ids is None:
        selected = enabled
    else:
        missing = [segment_id for segment_id in segment_ids if segment_id not in by_id]
        if missing:
            raise NativeTemplateError(
                "segment selection contains unknown or disabled IDs: "
                + ", ".join(missing)
            )
        selected_set = set(segment_ids)
        # Timeline order, not request order, is authoritative.
        selected = [segment for segment in enabled if segment.id in selected_set]
    if not selected:
        raise NativeTemplateError("at least one enabled timeline segment is required")
    return selected


def standard_lora_metadata_requests(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    segment_ids: list[str] | None = None,
) -> dict[ModelFamily, str]:
    """Return unknown auto Standard LoRAs that ComfyUI must inspect.

    Known audited artifacts and explicit visible overrides compile without an
    upstream metadata request. RayLight has its own fixed loader and is never
    part of this Standard-only probe contract.
    """

    requests: dict[ModelFamily, str] = {}
    for segment in _selected_segments(draft, segment_ids):
        family = segment.mode
        if family in requests:
            continue
        binding = getattr(settings.models, family)
        if (
            resolve_execution_backend(binding) != "standard"
            or binding.lora_name is None
            or binding.standard_lora_loader_override is not None
        ):
            continue
        basename = PurePosixPath(binding.lora_name.replace("\\", "/")).name
        if _known_standard_lora_loader(basename) is None:
            requests[family] = binding.lora_name
    return requests


def _compile_unit(
    *,
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    job_id: str,
    family: ModelFamily,
    backend: ExecutionBackend,
    segments: list[UnifiedTimelineSegment],
    unit_id: str,
    clear_raylight_vram_after_sampling: bool,
    predecessor_segment_id: str | None,
    continuity_source: ContinuitySource | None,
    historical_take: NativeHistoricalTake | None,
    continuity_overlap_frames: int,
    anchor_reset: bool,
    standard_lora_metadata: Mapping[str, Any] | None,
) -> tuple[NativeWorkflowUnit, list[dict[str, Any]]]:
    if len(segments) != 1:
        raise AssertionError("native workflow units must contain exactly one segment")
    graph = _Graph()
    shared = _shared_core(graph, settings)
    binding = getattr(settings.models, family)
    sampling = getattr(draft.sampling, family)
    seed = sampling.seed
    namespace = _raylight_namespace(family, binding)
    if backend == "raylight":
        model = _raylight_model(
            graph,
            binding,
            sampling,
            namespace=namespace,
            clear_vram_after_sampling=clear_raylight_vram_after_sampling,
        )
    else:
        model = _standard_model(
            graph,
            binding,
            sampling,
            lora_metadata=standard_lora_metadata,
        )
    output_nodes: dict[str, str] = {}
    plans: list[dict[str, Any]] = []
    continuity_dependency: NativeContinuityDependency | None = None
    for segment in segments:
        visible_frames = _align_h3_frames(
            segment.duration_seconds, draft.render.fps
        )
        context_frames = (
            continuity_overlap_frames if predecessor_segment_id is not None else 0
        )
        sample_frames = _align_h3_frame_count(visible_frames + context_frames)
        conditioning, latent, source_audio = _conditioning(
            graph,
            segment,
            draft,
            shared,
            frames=sample_frames,
        )
        if predecessor_segment_id is not None:
            conditioning, load_video_node_id = _add_continuity_guides(
                graph,
                conditioning=conditioning,
                latent=latent,
                segment=segment,
                draft=draft,
                shared=shared,
                visible_frames=visible_frames,
                overlap_frames=context_frames,
            )
            continuity_dependency = NativeContinuityDependency(
                predecessor_segment_id=predecessor_segment_id,
                overlap_frames=context_frames,
                load_video_node_id=load_video_node_id,
                source=continuity_source or "same_run",
                historical_take_id=(
                    historical_take.id if historical_take is not None else None
                ),
            )
        samples = _sample(
            graph,
            backend=backend,
            model=model,
            conditioning=conditioning,
            latent=latent,
            sampling=sampling,
            seed=seed,
        )
        output_node = _decode_and_save(
            graph,
            samples=samples,
            source_audio=source_audio,
            draft=draft,
            shared=shared,
            job_id=job_id,
            segment=segment,
            visible_frames=visible_frames,
            continuity_prefix_frames=context_frames,
        )
        output_nodes[segment.id] = output_node
        node_classes = tuple(
            dict.fromkeys(
                node["class_type"] for node in graph.prompt.values()
            )
        )
        plans.append(
            {
                "segment_id": segment.id,
                "mode": segment.mode,
                "recipe": timeline_segment_recipe(segment),
                "model_family": family,
                "backend": backend,
                "frame_count": visible_frames,
                "visible_frame_count": visible_frames,
                "sample_frame_count": sample_frames,
                "continuity_context_frames": context_frames,
                "alignment_tail_frame_count": (
                    sample_frames - visible_frames - context_frames
                ),
                "predecessor_segment_id": predecessor_segment_id,
                "continuity_source": continuity_source,
                "historical_take_id": (
                    historical_take.id if historical_take is not None else None
                ),
                "anchor_reset": anchor_reset,
                # Randomness is resolved by the editor before submission. The
                # report always exposes the exact JSON-safe value that this
                # graph carries, even when the UI intends to re-roll it before
                # a later submit.
                "seed_mode": "random" if sampling.random_seed else "fixed",
                "seed": seed,
                "conditioning_node": (
                    "MiniMaxH3ImageToVideo"
                    if segment.mode == "fl2va"
                    else "MiniMaxH3ReferenceToVideo"
                ),
                "node_classes": list(node_classes),
            }
        )
    unit = NativeWorkflowUnit(
            id=unit_id,
            family=family,
            backend=backend,
            segment_ids=tuple(segment.id for segment in segments),
            prompt=graph.prompt,
            output_nodes=output_nodes,
            continuity=continuity_dependency,
        )
    if historical_take is not None:
        unit = bind_native_workflow_predecessor_output(
            unit, historical_take.output
        )
    return unit, plans


def compile_native_timeline(
    draft: UnifiedTimelineDraft,
    settings: RuntimeSettings,
    job_id: str,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, NativeHistoricalTake] | None = None,
    standard_lora_metadata: Mapping[
        ModelFamily, Mapping[str, Any] | None
    ] | None = None,
) -> NativeCompileResult:
    """Compile one server-owned prompt per selected timeline segment.

    Segment prompts deliberately repeat stable loader node IDs and inputs.
    ComfyUI can therefore reuse its endpoint-local model cache across prompts,
    while one failed branch cannot invalidate or cancel every take in a model
    family.  Submission order remains deterministic across recompilation.
    """

    if settings.memory_policy != "keep_resident":
        raise NativeTemplateError(
            "native segment workflows support memory_policy='keep_resident' only; "
            "clear_between_segments has no equivalent stock ComfyUI node"
        )
    if draft.render.fps != 24.0:
        raise NativeTemplateError(
            "MiniMax H3 native temporal and reference-video contracts are fixed "
            "at 24 fps; render.fps must equal 24"
        )
    selected = _selected_segments(draft, segment_ids)
    for segment in selected:
        _validate_native_reference_slots(segment)
    routed: list[
        tuple[
            int,
            UnifiedTimelineSegment,
            ModelFamily,
            ExecutionBackend,
            str | None,
            ContinuitySource | None,
            NativeHistoricalTake | None,
            bool,
        ]
    ] = []
    timeline_positions = {
        segment.id: index for index, segment in enumerate(draft.segments)
    }
    continuity_predecessors = unified_continuity_predecessors(draft)
    enabled_segments = [segment for segment in draft.segments if segment.enabled]
    previous_enabled = {
        segment.id: enabled_segments[index - 1] if index else None
        for index, segment in enumerate(enabled_segments)
    }
    selected_ids = {segment.id for segment in selected}
    for segment in selected:
        timeline_index = timeline_positions[segment.id]
        family = segment.mode
        binding = getattr(settings.models, family)
        backend = resolve_execution_backend(binding)
        if backend == "raylight" and binding.device != "default":
            raise NativeTemplateError(
                f"{family}.device must be 'default' when RayLight is enabled; "
                "raylight.gpu_select is the authoritative logical GPU pool"
            )
        predecessor_segment_id: str | None = None
        continuity_source: ContinuitySource | None = None
        historical_take: NativeHistoricalTake | None = None
        explicit_anchor_reset = (
            isinstance(segment, UnifiedFL2VASegment)
            and segment.first_image is not None
        )
        anchor_reset = False
        if segment.continuity.enabled:
            predecessor = continuity_predecessors.get(segment.id)
            anchor_reset = predecessor is None and (
                previous_enabled[segment.id] is None or explicit_anchor_reset
            )
            if predecessor is not None:
                predecessor_segment_id = predecessor.id
                if predecessor.id in selected_ids:
                    continuity_source = "same_run"
                else:
                    continuity_source = "historical_take"
                    historical_take = (historical_takes or {}).get(segment.id)
                    if historical_take is None:
                        raise NativeTemplateError(
                            f"continuity segment '{segment.id}' requires a "
                            f"server-resolved historical take for predecessor "
                            f"'{predecessor.id}'"
                        )
                    if historical_take.segment_id != predecessor.id:
                        raise NativeTemplateError(
                            f"historical take '{historical_take.id}' belongs to "
                            f"segment '{historical_take.segment_id}', not current "
                            f"predecessor '{predecessor.id}'"
                        )
        routed.append(
            (
                timeline_index,
                segment,
                family,
                backend,
                predecessor_segment_id,
                continuity_source,
                historical_take,
                anchor_reset,
            )
        )
    keep_raylight_resident = (
        settings.raylight_residency_policy == "keep_until_switch"
    )
    clear_raylight_vram_after_sampling = not keep_raylight_resident
    workflows: list[NativeWorkflowUnit] = []
    plans: list[dict[str, Any]] = []
    # Dependency chains must retain timeline order. Independent prompts keep
    # the established Standard-before-Ray grouping for stable cache behavior.
    has_continuity_edges = any(route[4] is not None for route in routed)
    ordered_routes = (
        sorted(routed, key=lambda route: route[0])
        if has_continuity_edges
        else sorted(
            routed,
            key=lambda route: (
                ("standard", "raylight").index(route[3]),
                ("fl2va", "ref2va").index(route[2]),
                route[0],
            ),
        )
    )
    for (
        timeline_index,
        segment,
        family,
        backend,
        predecessor_segment_id,
        continuity_source,
        historical_take,
        anchor_reset,
    ) in ordered_routes:
        unit, unit_plans = _compile_unit(
            draft=draft,
            settings=settings,
            job_id=job_id,
            family=family,
            backend=backend,
            segments=[segment],
            unit_id=f"{backend}-{family}-{timeline_index:03d}",
            clear_raylight_vram_after_sampling=(
                clear_raylight_vram_after_sampling
            ),
            predecessor_segment_id=predecessor_segment_id,
            continuity_source=continuity_source,
            historical_take=historical_take,
            continuity_overlap_frames=segment.continuity.overlap_frames,
            anchor_reset=anchor_reset,
            standard_lora_metadata=(standard_lora_metadata or {}).get(family),
        )
        workflows.append(unit)
        plans.extend(unit_plans)
    all_nodes = sorted(
        {
            node["class_type"]
            for unit in workflows
            for node in unit.prompt.values()
        }
    )
    unknown = sorted(set(all_nodes) - set(_PROVENANCE))
    if unknown:
        raise AssertionError(f"native template emitted unclassified nodes: {unknown}")
    custom_nodes = sorted(
        node
        for node in all_nodes
        if _PROVENANCE[node] in {"raylight", "lora-custom"}
    )
    families = tuple(
        family for family in ("fl2va", "ref2va") if any(
            unit.family == family for unit in workflows
        )
    )
    lora_resolution: dict[str, dict[str, Any]] = {}
    for family in families:
        binding = getattr(settings.models, family)
        if binding.lora_name is None:
            continue
        backend = resolve_execution_backend(binding)
        if backend == "raylight":
            loader_node = "RayLoraLoader"
            source = "raylight"
        else:
            loader_node = _resolve_standard_lora(
                binding, (standard_lora_metadata or {}).get(family)
            )
            basename = PurePosixPath(
                binding.lora_name.replace("\\", "/")
            ).name
            source = (
                "manual"
                if binding.standard_lora_loader_override is not None
                else "audited_profile"
                if _known_standard_lora_loader(basename) is not None
                else "metadata"
            )
        lora_resolution[family] = {
            "lora_name": binding.lora_name,
            "model_filename": binding.filename,
            "backend": backend,
            "loader_node": loader_node,
            "source": source,
        }
    manifest = {
        "version": 2,
        "graph_source": "server",
        "accepts_client_workflow": False,
        "continuity": {
            "boundaries": [
                {
                    "segment_id": segment.id,
                    "predecessor_segment_id": predecessor_segment_id,
                    "overlap_frames": segment.continuity.overlap_frames,
                    "source": continuity_source,
                    "historical_take_id": (
                        historical_take.id if historical_take is not None else None
                    ),
                }
                for (
                    _,
                    segment,
                    _,
                    _,
                    predecessor_segment_id,
                    continuity_source,
                    historical_take,
                    _,
                ) in ordered_routes
                if predecessor_segment_id is not None
            ],
        },
        "submission_order": [unit.id for unit in workflows],
        "raylight_exclusive": any(
            unit.backend == "raylight" for unit in workflows
        ),
        "lora_resolution": lora_resolution,
        "resident_cache_scope": {
            "boundary": "comfy_endpoint",
            "standard": "family+model_loader_inputs",
            "prompt_partition": "one_segment",
            "raylight_initializer": "gpu_pool+topology",
            "raylight_model": "worker_ram_cache(model+lora+weight_dtype)",
            "raylight_cuda_residency": (
                "kept_for_compatible_key_until_explicit_switch"
                if keep_raylight_resident
                else "released_after_each_sampler"
            ),
            "raylight_residency_reason": (
                "explicit_keyed_switch_policy"
                if keep_raylight_resident
                else "shared_endpoint_safe_default"
            ),
            "raylight_resident_family": None,
        },
        "units": [
            {
                "id": unit.id,
                "family": unit.family,
                "backend": unit.backend,
                "segment_ids": list(unit.segment_ids),
                "output_nodes": dict(unit.output_nodes),
                "continuity": (
                    {
                        "predecessor_segment_id": (
                            unit.continuity.predecessor_segment_id
                        ),
                        "overlap_frames": unit.continuity.overlap_frames,
                        "load_video_node_id": unit.continuity.load_video_node_id,
                        "source": unit.continuity.source,
                        "historical_take_id": (
                            unit.continuity.historical_take_id
                        ),
                        "resolved": unit.continuity.resolved,
                    }
                    if unit.continuity is not None
                    else None
                ),
            }
            for unit in workflows
        ],
    }
    return NativeCompileResult(
        workflows=tuple(workflows),
        manifest=manifest,
        plans=tuple(plans),
        families=families,
        node_policy={
            "graph_source": "server",
            "accepts_client_workflow": False,
            "allowed_nodes": all_nodes,
            "custom_nodes": custom_nodes,
            "provenance": {node: _PROVENANCE[node] for node in all_nodes},
        },
    )


def validate_native_capabilities(
    result: NativeCompileResult,
    available_nodes: list[str] | set[str],
    node_provenance: dict[str, str] | None = None,
) -> None:
    """Fail before queue submission when a fixed template is unavailable.

    ``class_type`` names alone are not a sufficient trust boundary: a custom
    extension can register the same name as a core node.  Real ComfyUI clients
    therefore pass the ``python_module`` values returned by ``/object_info``
    and every emitted class is checked against the server-owned policy.  The
    optional argument is retained for offline compiler tests that do not own a
    ComfyUI registry.
    """

    available = set(available_nodes)
    required = set(result.node_policy["allowed_nodes"])
    missing = sorted(required - available)
    if missing:
        raise NativeTemplateError(
            "ComfyUI is missing nodes required by the selected server template: "
            + ", ".join(missing)
        )
    if node_provenance is not None:
        invalid: list[str] = []
        expected = result.node_policy["provenance"]
        for node_name in sorted(required):
            source = node_provenance.get(node_name, "")
            policy = expected[node_name]
            expected_module = EXPECTED_NATIVE_NODE_MODULES[node_name]
            if source != expected_module:
                invalid.append(
                    f"{node_name} (expected {policy} from {expected_module}, "
                    f"got {source or 'unknown'})"
                )
        if invalid:
            raise NativeTemplateError(
                "ComfyUI node provenance does not match the fixed template policy: "
                + ", ".join(invalid)
            )
    forbidden = {
        node["class_type"]
        for unit in result.workflows
        for node in unit.prompt.values()
        if node["class_type"] == "MiniMaxH3Director"
    }
    if forbidden:
        raise AssertionError("native template must never emit MiniMaxH3Director")


def validate_native_workflow_unit_capabilities(
    unit: NativeWorkflowUnit,
    available_nodes: list[str] | set[str],
    node_provenance: dict[str, str] | None = None,
) -> None:
    """Apply the exact registry/provenance boundary to an internal barrier."""

    node_names = sorted(
        {str(node.get("class_type") or "") for node in unit.prompt.values()}
    )
    unknown = sorted(set(node_names) - set(_PROVENANCE))
    if unknown:
        raise NativeTemplateError(
            "native transition emitted unclassified nodes: " + ", ".join(unknown)
        )
    synthetic = NativeCompileResult(
        workflows=(unit,),
        manifest={},
        plans=(),
        families=(unit.family,),
        node_policy={
            "graph_source": "server",
            "accepts_client_workflow": False,
            "allowed_nodes": node_names,
            "custom_nodes": node_names,
            "provenance": {name: _PROVENANCE[name] for name in node_names},
        },
    )
    validate_native_capabilities(
        synthetic, available_nodes, node_provenance=node_provenance
    )
