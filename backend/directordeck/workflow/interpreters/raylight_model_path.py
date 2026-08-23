from __future__ import annotations

from ... import native_templates as _native_templates
from ...native_templates import (
    NativeTemplateError,
    RaylightAttentionMode,
    raylight_runtime_namespace,
    resolve_execution_backend,
    resolve_raylight_attention_backend,
)
from ...schemas import DiffusionModelBinding, SamplingConfig
from ._emitter import NativeEdge, NativeNodeEmitter, edge


POOL_INTENT_INTERPRETER_ID = "raylight_pool_intent"
MODEL_LOAD_INTERPRETER_ID = "raylight_model_load"
SIGMA_SHIFT_INTERPRETER_ID = "raylight_sigma_shift"
INTERPRETER_VERSION = 1


def _require_raylight(binding: DiffusionModelBinding) -> None:
    if resolve_execution_backend(binding) != "raylight":
        raise NativeTemplateError("RayLight model interpreter received a Standard binding")
    profile = binding.raylight
    if profile.fsdp or profile.cpu_offload:
        raise NativeTemplateError(
            "RayLight FSDP/CPU offload is disabled in native timeline v1 until "
            "its post-sampling CUDA cleanup is verified"
        )


def emit_raylight_pool_intent(
    emitter: NativeNodeEmitter,
    binding: DiffusionModelBinding,
    *,
    namespace: str,
    clear_vram_after_sampling: bool,
    attention_mode: RaylightAttentionMode = "ck_int8",
    enforce_attention_topology: bool = True,
) -> NativeEdge:
    _require_raylight(binding)
    profile = binding.raylight
    attention = resolve_raylight_attention_backend(
        attention_mode,
        binding=binding if enforce_attention_topology else None,
    )
    if namespace != raylight_runtime_namespace(
        binding,
        attention_mode=attention_mode,
        enforce_attention_topology=enforce_attention_topology,
    ):
        raise NativeTemplateError(
            "RayLight attention and runtime namespace do not match"
        )
    return edge(
        emitter.add(
            "DirectorDeckRayInitializerAdvanced",
            ray_cluster_address="local",
            ray_cluster_namespace=namespace,
            GPU=len(profile.gpu_select),
            GPU_SELECT=",".join(str(index) for index in profile.gpu_select),
            driver_cleanup_policy=(
                _native_templates._RAYLIGHT_DRIVER_CLEANUP_POLICY
            ),
            ulysses_degree=profile.ulysses_degree,
            ring_degree=profile.ring_degree,
            clear_vram_after_sampling=clear_vram_after_sampling,
            ram_cache_max_models=(
                _native_templates._RAYLIGHT_RAM_CACHE_MAX_MODELS
            ),
            cfg_degree=profile.cfg_degree,
            dp_degree=profile.dp_degree,
            sync_ulysses=False,
            FSDP=profile.fsdp,
            FSDP_CPU_OFFLOAD=profile.cpu_offload,
            XFuser_attention=attention,
            skip_comm_test=False,
            use_mmap=False,
        )
    )


def emit_raylight_model_load(
    emitter: NativeNodeEmitter,
    pool_intent: NativeEdge,
    binding: DiffusionModelBinding,
    *,
    lora: NativeEdge | None = None,
) -> NativeEdge:
    _require_raylight(binding)
    inputs = {
        "ray_actors_init": pool_intent,
        "unet_name": binding.filename,
        "weight_dtype": "default",
    }
    if lora is not None:
        inputs["lora"] = lora
    return edge(emitter.add("DirectorDeckRayUNETLoader", **inputs))


def emit_raylight_sigma_shift(
    emitter: NativeNodeEmitter,
    ray_actors: NativeEdge,
    sampling: SamplingConfig,
) -> NativeEdge:
    return edge(
        emitter.add(
            "DirectorDeckRayMiniMaxH3SigmaShift",
            ray_actors=ray_actors,
            shift_video=sampling.shift,
            shift_audio=sampling.audio_shift,
        )
    )
