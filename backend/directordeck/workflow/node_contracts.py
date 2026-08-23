from __future__ import annotations

"""Frozen v4 node contracts for Director-owned prompt compilation.

The registry in this module is the single class-type authority.  Compatibility
views for the pre-v5 ``node_policy`` shape are derived from it; they are not
parallel registration tables.

The hashes below are deterministic identities for Director's own compiler
adapter descriptions. They are never computed from, or compared with, live
ComfyUI/custom-node source and therefore do not authorize a user's runtime.
"""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import Field, model_validator

from .canonical import canonical_json_bytes
from .contracts import (
    Backend,
    ClassType,
    ContractModel,
    ExecutionTerminalRole,
    FrozenMap,
    ModelFamily,
    ModuleIdentity,
    NodeContract,
    NodeContractRegistry,
    NodeOutputContract,
    ObjectInfoContract,
    ObjectInfoInputContract,
    ObjectInfoOutputContract,
    PersistentArtifactRole,
    PositiveVersion,
    RuntimeFingerprint,
    RuntimeEffectContract,
    SemanticVersion,
    Sha256Digest,
)


NODE_CONTRACT_SCHEMA_VERSION: Final = 1
OBJECT_INFO_NORMALIZATION_VERSION: Final = 1
DIRECTOR_NODE_ADAPTER_SEMANTIC_VERSION: Final = "1.0.0"
RUNTIME_SUPPORT_FINGERPRINT_SCHEMA: Final = 1

# These nodes affect placement, persistence, or control flow without claiming
# to transform the creative value they pass or consume. Every other emitted
# v4 node must satisfy a fail-closed strict runtime-effect contract.
V4_OUTPUT_NEUTRAL_NODE_CLASSES: Final = frozenset(
    {
        "SelectModelDevice",
        "SelectCLIPDevice",
        "SelectVAEDevice",
        "SaveVideo",
        "DirectorDeckRayKill",
    }
)

_NO_DEFAULT = object()
_ALL_FAMILIES: tuple[ModelFamily, ...] = ("fl2va", "ref2va")
_ALL_BACKENDS: tuple[Backend, ...] = ("standard", "raylight")
_STANDARD_ONLY: tuple[Backend, ...] = ("standard",)
_RAYLIGHT_ONLY: tuple[Backend, ...] = ("raylight",)


class RuntimeFingerprintMaterial(ContractModel):
    """Canonical fingerprint input, without claiming where evidence came from."""

    schema_version: PositiveVersion = RUNTIME_SUPPORT_FINGERPRINT_SCHEMA
    normalized_module_identity: ModuleIdentity
    object_info_contract_slice: FrozenMap[ClassType, ObjectInfoContract]
    adapter_module_content_digest: Sha256Digest
    package_version: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    director_wrapper_semantic_version: SemanticVersion

    @model_validator(mode="after")
    def _require_contract_slice(self) -> RuntimeFingerprintMaterial:
        if not self.object_info_contract_slice:
            raise ValueError("runtime fingerprint requires an object-info slice")
        return self


def _input(
    port_type: str,
    *,
    enum_values: tuple[object, ...] = (),
    director_default: object = _NO_DEFAULT,
) -> ObjectInfoInputContract:
    values = tuple(enum_values)
    if director_default is _NO_DEFAULT:
        return ObjectInfoInputContract(
            port_type=port_type,
            enum_values=values,
        )
    return ObjectInfoInputContract(
        port_type=port_type,
        enum_values=values,
        has_director_default=True,
        director_default=director_default,
    )


def _output(
    index: int,
    port_type: str,
    name: str | None = None,
) -> ObjectInfoOutputContract:
    return ObjectInfoOutputContract(index=index, port_type=port_type, name=name)


def _object_info(
    *,
    required: Mapping[str, ObjectInfoInputContract],
    optional: Mapping[str, ObjectInfoInputContract] | None = None,
    outputs: tuple[ObjectInfoOutputContract, ...] = (),
    director_supplied_inputs: Iterable[str] | None = None,
    output_node: bool = False,
) -> ObjectInfoContract:
    optional_inputs = dict(optional or {})
    if director_supplied_inputs is None:
        supplied = (*required, *optional_inputs)
    else:
        supplied = tuple(director_supplied_inputs)
    return ObjectInfoContract(
        normalization_version=OBJECT_INFO_NORMALIZATION_VERSION,
        required_inputs=dict(required),
        optional_inputs=optional_inputs,
        director_supplied_inputs=tuple(supplied),
        outputs=outputs,
        output_node=output_node,
    )


def _strict(
    *,
    families: tuple[ModelFamily, ...] = _ALL_FAMILIES,
    backends: tuple[Backend, ...] = _ALL_BACKENDS,
    validation_method: Literal[
        "node_contract",
        "strict_wrapper",
        "director_owned_implementation",
        "user_assumed",
    ] = "node_contract",
    notes: tuple[str, ...] = (),
) -> RuntimeEffectContract:
    return RuntimeEffectContract(
        policy="strict_transform",
        unsupported_behavior="raise",
        validation_method=validation_method,
        verified_model_families=families,
        verified_backends=backends,
        notes=notes,
    )


def _identity_allowed(
    *,
    families: tuple[ModelFamily, ...] = _ALL_FAMILIES,
    backends: tuple[Backend, ...] = _ALL_BACKENDS,
    validation_method: Literal[
        "node_contract",
        "strict_wrapper",
        "director_owned_implementation",
    ] = "node_contract",
    unsupported_behavior: Literal["identity", "fallback"] = "identity",
    notes: tuple[str, ...],
) -> RuntimeEffectContract:
    return RuntimeEffectContract(
        policy="identity_allowed",
        unsupported_behavior=unsupported_behavior,
        validation_method=validation_method,
        verified_model_families=families,
        verified_backends=backends,
        notes=notes,
    )


def _side_effect(
    *,
    backends: tuple[Backend, ...] = _ALL_BACKENDS,
    notes: tuple[str, ...],
) -> RuntimeEffectContract:
    return RuntimeEffectContract(
        policy="side_effect_only",
        unsupported_behavior="raise",
        validation_method="node_contract",
        verified_model_families=_ALL_FAMILIES,
        verified_backends=backends,
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class _NodeSpec:
    class_type: str
    module: str
    object_info: ObjectInfoContract
    effect: RuntimeEffectContract
    execution_terminal_role: ExecutionTerminalRole | None = None
    persistent_artifact_role: PersistentArtifactRole | None = None


def _spec(
    class_type: str,
    module: str,
    object_info: ObjectInfoContract,
    effect: RuntimeEffectContract,
    *,
    execution_terminal_role: ExecutionTerminalRole | None = None,
    persistent_artifact_role: PersistentArtifactRole | None = None,
) -> _NodeSpec:
    return _NodeSpec(
        class_type=class_type,
        module=module,
        object_info=object_info,
        effect=effect,
        execution_terminal_role=execution_terminal_role,
        persistent_artifact_role=persistent_artifact_role,
    )


def _reference_inputs(
    prefix: str,
    item: str,
    count: int,
    port: str,
) -> dict[str, ObjectInfoInputContract]:
    return {
        f"{prefix}.{item}_{index}": _input(port)
        for index in range(count)
    }


_CORE = "nodes"
_MULTIGPU = "comfy_extras.nodes_multigpu"
_VIDEO = "comfy_extras.nodes_video"
_AUDIO = "comfy_extras.nodes_audio"
_H3 = "comfy_extras.nodes_minimax_h3"
_SAMPLER = "comfy_extras.nodes_custom_sampler"
_IMAGES = "comfy_extras.nodes_images"
_LORA_DEBUG = "comfy_extras.nodes_lora_debug"
_DIRECTOR_STRICT_ATTENTION = "custom_nodes.DirectorDeck-Strict-Attention"
_DIRECTOR_STRICT_H3 = "custom_nodes.DirectorDeck-Strict-H3"
_TURBO = "custom_nodes.ComfyUI-MiniMax-H3-Turbo"
_RAYLIGHT = "custom_nodes.DirectorDeck-RayLight"


_SELECT_IDENTITY_NOTE = (
    "The Comfy device selector is allowed to warn and pass through unchanged "
    "when the requested device or retarget operation is unavailable.",
)
_LORA_IDENTITY_NOTE = (
    "This implementation does not prove that every requested LoRA key changed "
    "the target model; it cannot carry a fail-closed output-affecting feature "
    "without a strict wrapper.",
)
_USER_ASSUMED_LORA_NOTE = (
    "The user-selected loader is checked only for the node interface Director "
    "actually invokes; the user is responsible for installing a compatible "
    "implementation and for its runtime LoRA effect.",
)
_DIRECTOR_STRICT_ATTENTION_NOTE = (
    "The Director wrapper accepts only the exact MiniMax H3 ModelPatcher, "
    "rejects an existing attention override, proves the selected ComfyUI "
    "backend registration and device capability before cloning, and reflects "
    "the exact installed callable without host fallback.",
    "CPU contract coverage proves the PyTorch path and fail-closed ck_int8 "
    "rejection; it is not a claim that a CUDA ck_int8 kernel is available.",
    "The bundled module exposes a privacy-safe exact runtime probe consumed "
    "by host capability schema 2 for default, CPU and logical-GPU preflight.",
    "A pre-existing strict H3 Sage block patch is rejected so mutually "
    "exclusive attention features cannot both claim runtime effect.",
)
_DIRECTOR_STRICT_H3_NOTE = (
    "The Director implementation requires the real ComfyUI ModelPatcher and "
    "exact audited clone/get/add methods, then proves the exact MiniMax H3 "
    "block and attention structure, identical top-level/core SageAttention "
    "callable, CUDA architecture and its exact compiled-flag/kernel evidence "
    "(or the exact SM86 Triton callable) before cloning.",
    "Kernel exceptions propagate without PyTorch fallback. CPU contract tests "
    "prove structural and fail-closed behavior only, not GPU availability.",
    "The bundled module exposes a privacy-safe per-device Sage kernel probe "
    "consumed by host capability schema 2 before graph emission.",
    "A pre-existing global optimized-attention override is rejected before "
    "clone so mutually exclusive attention features fail closed.",
)
_RAY_LORA_STRICT_NOTE = (
    "The bundled descriptor node rejects identity strengths; the paired "
    "RayUNET worker path validates complete H3 adapter resolution, source-key "
    "consumption and exact hook or patch installation.",
    "CPU contract tests cover non-FSDP, FSDP and quantized branches; this is "
    "not a claim of a real multi-GPU actor smoke test.",
)
_RAY_UNET_STRICT_NOTE = (
    "Base-model load failures propagate, and the optional bundled H3 LoRA "
    "worker chain is fail-closed across non-FSDP, FSDP and quantized paths.",
    "Director feature resolution supplies the H3 family constraint; the "
    "generic RayUNET node alone does not infer model family at runtime.",
    "CPU contract tests do not constitute a real multi-GPU actor smoke test.",
)


def _node_specs() -> tuple[_NodeSpec, ...]:
    ref_optional = {
        **_reference_inputs("ref_images", "ref_image", 9, "IMAGE"),
        **_reference_inputs("ref_videos", "ref_video", 3, "IMAGE"),
        **_reference_inputs("ref_video_audios", "ref_video_audio", 3, "AUDIO"),
        **_reference_inputs("ref_audios", "ref_audio", 3, "AUDIO"),
    }

    return (
        _spec(
            "UNETLoader",
            _CORE,
            _object_info(
                required={
                    "unet_name": _input("COMBO"),
                    "weight_dtype": _input(
                        "COMBO",
                        enum_values=(
                            "default",
                            "fp8_e4m3fn",
                            "fp8_e4m3fn_fast",
                            "fp8_e5m2",
                        ),
                        director_default="default",
                    ),
                },
                outputs=(_output(0, "MODEL"),),
            ),
            _strict(backends=_STANDARD_ONLY),
        ),
        _spec(
            "CLIPLoader",
            _CORE,
            _object_info(
                required={
                    "clip_name": _input("COMBO"),
                    "type": _input(
                        "COMBO", enum_values=("minimax",), director_default="minimax"
                    ),
                },
                optional={
                    "device": _input(
                        "COMBO",
                        enum_values=("default", "cpu"),
                        director_default="default",
                    )
                },
                outputs=(_output(0, "CLIP"),),
            ),
            _strict(),
        ),
        _spec(
            "VAELoader",
            _CORE,
            _object_info(
                required={"vae_name": _input("COMBO")},
                outputs=(_output(0, "VAE"),),
            ),
            _strict(),
        ),
        _spec(
            "LoadImage",
            _CORE,
            _object_info(
                required={"image": _input("COMBO")},
                outputs=(_output(0, "IMAGE"), _output(1, "MASK")),
            ),
            _strict(),
        ),
        _spec(
            "LoraLoaderModelOnly",
            _CORE,
            _object_info(
                required={
                    "model": _input("MODEL"),
                    "lora_name": _input("COMBO"),
                    "strength_model": _input("FLOAT", director_default=1.0),
                },
                outputs=(_output(0, "MODEL"),),
            ),
            _identity_allowed(
                backends=_STANDARD_ONLY,
                validation_method="user_assumed",
                notes=(*_LORA_IDENTITY_NOTE, *_USER_ASSUMED_LORA_NOTE),
            ),
        ),
        _spec(
            "VAEDecode",
            _CORE,
            _object_info(
                required={"samples": _input("LATENT"), "vae": _input("VAE")},
                outputs=(_output(0, "IMAGE"),),
            ),
            _strict(),
        ),
        _spec(
            "SelectModelDevice",
            _MULTIGPU,
            _object_info(
                required={
                    "model": _input("MODEL"),
                    "device": _input("COMBO", director_default="default"),
                },
                outputs=(_output(0, "MODEL"),),
            ),
            _identity_allowed(backends=_STANDARD_ONLY, notes=_SELECT_IDENTITY_NOTE),
        ),
        _spec(
            "SelectCLIPDevice",
            _MULTIGPU,
            _object_info(
                required={
                    "clip": _input("CLIP"),
                    "device": _input("COMBO", director_default="default"),
                },
                outputs=(_output(0, "CLIP"),),
            ),
            _identity_allowed(notes=_SELECT_IDENTITY_NOTE),
        ),
        _spec(
            "SelectVAEDevice",
            _MULTIGPU,
            _object_info(
                required={
                    "vae": _input("VAE"),
                    "device": _input("COMBO", director_default="default"),
                },
                outputs=(_output(0, "VAE"),),
            ),
            _identity_allowed(notes=_SELECT_IDENTITY_NOTE),
        ),
        _spec(
            "LoadVideo",
            _VIDEO,
            _object_info(
                required={"file": _input("COMBO")},
                outputs=(_output(0, "VIDEO"),),
            ),
            _strict(),
        ),
        _spec(
            "Video Slice",
            _VIDEO,
            _object_info(
                required={
                    "video": _input("VIDEO"),
                    "start_time": _input("FLOAT"),
                    "duration": _input("FLOAT"),
                    "strict_duration": _input("BOOLEAN", director_default=True),
                },
                outputs=(_output(0, "VIDEO"),),
            ),
            _strict(),
        ),
        _spec(
            "GetVideoComponents",
            _VIDEO,
            _object_info(
                required={"video": _input("VIDEO")},
                outputs=(
                    _output(0, "IMAGE", "images"),
                    _output(1, "AUDIO", "audio"),
                    _output(2, "FLOAT", "fps"),
                    _output(3, "INT", "bit_depth"),
                ),
            ),
            _strict(),
        ),
        _spec(
            "LoadAudio",
            _AUDIO,
            _object_info(
                required={"audio": _input("COMBO")},
                outputs=(_output(0, "AUDIO"),),
            ),
            _strict(),
        ),
        _spec(
            "MiniMaxH3ImageToVideo",
            _H3,
            _object_info(
                required={
                    "clip": _input("CLIP"),
                    "vae": _input("VAE"),
                    "prompt": _input("STRING"),
                    "width": _input("INT", director_default=864),
                    "height": _input("INT", director_default=480),
                    "length": _input("INT", director_default=124),
                },
                optional={
                    "first_frame": _input("IMAGE"),
                    "last_frame": _input("IMAGE"),
                },
                outputs=(
                    _output(0, "CONDITIONING", "positive"),
                    _output(1, "LATENT"),
                ),
            ),
            _strict(families=("fl2va",)),
        ),
        _spec(
            "MiniMaxH3ReferenceToVideo",
            _H3,
            _object_info(
                required={
                    "clip": _input("CLIP"),
                    "vae": _input("VAE"),
                    "audio_vae": _input("VAE"),
                    "prompt": _input("STRING"),
                    "width": _input("INT", director_default=864),
                    "height": _input("INT", director_default=480),
                    "length": _input("INT", director_default=124),
                    "ref_image_size": _input(
                        "COMBO",
                        enum_values=("match", "max"),
                        director_default="match",
                    ),
                },
                optional=ref_optional,
                outputs=(
                    _output(0, "CONDITIONING", "positive"),
                    _output(1, "LATENT"),
                ),
            ),
            _strict(families=("ref2va",)),
        ),
        _spec(
            "MiniMaxH3AddGuide",
            _H3,
            _object_info(
                required={
                    "positive": _input("CONDITIONING"),
                    "latent": _input("LATENT"),
                    "frame_idx": _input("INT", director_default=0),
                },
                optional={
                    "vae": _input("VAE"),
                    "audio_vae": _input("VAE"),
                    "image": _input("IMAGE"),
                    "audio": _input("AUDIO"),
                },
                outputs=(_output(0, "CONDITIONING", "positive"),),
            ),
            _strict(),
        ),
        _spec(
            "MiniMaxH3SigmaShift",
            _H3,
            _object_info(
                required={
                    "model": _input("MODEL"),
                    "shift_video": _input("FLOAT", director_default=12.0),
                    "shift_audio": _input("FLOAT", director_default=3.0),
                },
                outputs=(_output(0, "MODEL"),),
            ),
            _strict(backends=_STANDARD_ONLY),
        ),
        _spec(
            "BasicGuider",
            _SAMPLER,
            _object_info(
                required={
                    "model": _input("MODEL"),
                    "conditioning": _input("CONDITIONING"),
                },
                outputs=(_output(0, "GUIDER"),),
            ),
            _strict(backends=_STANDARD_ONLY),
        ),
        _spec(
            "BasicScheduler",
            _SAMPLER,
            _object_info(
                required={
                    "model": _input("MODEL"),
                    "scheduler": _input(
                        "COMBO",
                        enum_values=("simple", "normal", "karras", "beta"),
                        director_default="simple",
                    ),
                    "steps": _input("INT", director_default=25),
                    "denoise": _input("FLOAT", director_default=1.0),
                },
                outputs=(_output(0, "SIGMAS"),),
            ),
            _strict(backends=_STANDARD_ONLY),
        ),
        _spec(
            "KSamplerSelect",
            _SAMPLER,
            _object_info(
                required={
                    "sampler_name": _input(
                        "COMBO",
                        enum_values=("res_multistep", "euler", "dpmpp_2m"),
                        director_default="res_multistep",
                    )
                },
                outputs=(_output(0, "SAMPLER"),),
            ),
            _strict(),
        ),
        _spec(
            "RandomNoise",
            _SAMPLER,
            _object_info(
                required={"noise_seed": _input("INT", director_default=0)},
                outputs=(_output(0, "NOISE"),),
            ),
            _strict(backends=_STANDARD_ONLY),
        ),
        _spec(
            "SamplerCustomAdvanced",
            _SAMPLER,
            _object_info(
                required={
                    "noise": _input("NOISE"),
                    "guider": _input("GUIDER"),
                    "sampler": _input("SAMPLER"),
                    "sigmas": _input("SIGMAS"),
                    "latent_image": _input("LATENT"),
                },
                outputs=(
                    _output(0, "LATENT", "output"),
                    _output(1, "LATENT", "denoised_output"),
                ),
            ),
            _strict(backends=_STANDARD_ONLY),
        ),
        _spec(
            "VAEDecodeAudio",
            _AUDIO,
            _object_info(
                required={"samples": _input("LATENT"), "vae": _input("VAE")},
                outputs=(_output(0, "AUDIO"),),
            ),
            _strict(),
        ),
        _spec(
            "TrimAudioDuration",
            _AUDIO,
            _object_info(
                required={
                    "audio": _input("AUDIO"),
                    "start_index": _input("FLOAT"),
                    "duration": _input("FLOAT"),
                },
                outputs=(_output(0, "AUDIO"),),
            ),
            _strict(),
        ),
        _spec(
            "CreateVideo",
            _VIDEO,
            _object_info(
                required={
                    "images": _input("IMAGE"),
                    "fps": _input("FLOAT", director_default=24.0),
                },
                optional={
                    "audio": _input("AUDIO"),
                    "bit_depth": _input("INT", director_default=8),
                },
                outputs=(_output(0, "VIDEO"),),
            ),
            _strict(),
        ),
        _spec(
            "SaveVideo",
            _VIDEO,
            _object_info(
                required={
                    "video": _input("VIDEO"),
                    "filename_prefix": _input("STRING"),
                    "format": _input(
                        "COMBO", enum_values=("auto",), director_default="auto"
                    ),
                    "codec": _input(
                        "DYNAMIC_COMBO",
                        enum_values=("auto",),
                        director_default="auto",
                    ),
                },
                outputs=(_output(0, "VIDEO", "video"),),
                output_node=True,
            ),
            _side_effect(notes=("Persists the unique authored take artifact.",)),
            execution_terminal_role="take",
            persistent_artifact_role="take",
        ),
        _spec(
            "ImageFromBatch",
            _IMAGES,
            _object_info(
                required={
                    "image": _input("IMAGE"),
                    "batch_index": _input("INT"),
                    "length": _input("INT"),
                },
                outputs=(_output(0, "IMAGE"),),
            ),
            _strict(),
        ),
        _spec(
            "MiniMaxH3TurboLoRA",
            _TURBO,
            _object_info(
                required={
                    "model": _input("MODEL"),
                    "lora_name": _input("COMBO"),
                    "strength": _input("FLOAT", director_default=1.0),
                    "low_vram": _input("BOOLEAN", director_default=False),
                },
                outputs=(_output(0, "MODEL"),),
            ),
            _identity_allowed(
                backends=_STANDARD_ONLY,
                validation_method="user_assumed",
                notes=(*_LORA_IDENTITY_NOTE, *_USER_ASSUMED_LORA_NOTE),
            ),
        ),
        _spec(
            "LoraLoaderBypassModelOnly",
            _LORA_DEBUG,
            _object_info(
                required={
                    "model": _input("MODEL"),
                    "lora_name": _input("COMBO"),
                    "strength_model": _input("FLOAT", director_default=1.0),
                },
                outputs=(_output(0, "MODEL"),),
            ),
            _identity_allowed(
                backends=_STANDARD_ONLY,
                validation_method="user_assumed",
                notes=(*_LORA_IDENTITY_NOTE, *_USER_ASSUMED_LORA_NOTE),
            ),
        ),
        _spec(
            "DirectorDeckRayInitializerAdvanced",
            _RAYLIGHT,
            _object_info(
                required={
                    "ray_cluster_address": _input("STRING", director_default="local"),
                    "ray_cluster_namespace": _input(
                        "STRING", director_default="default"
                    ),
                    "GPU": _input("INT"),
                    "GPU_SELECT": _input("STRING"),
                    "ulysses_degree": _input("INT", director_default=1),
                    "ring_degree": _input("INT", director_default=1),
                    "clear_vram_after_sampling": _input(
                        "BOOLEAN", director_default=False
                    ),
                    "cfg_degree": _input("INT", director_default=1),
                    "dp_degree": _input("INT", director_default=1),
                    "sync_ulysses": _input("BOOLEAN", director_default=False),
                    "FSDP": _input("BOOLEAN", director_default=False),
                    "FSDP_CPU_OFFLOAD": _input(
                        "BOOLEAN", director_default=False
                    ),
                    "XFuser_attention": _input(
                        "COMBO",
                        enum_values=("COMFY_KITCHEN_INT8", "TORCH_FLASH"),
                        director_default="COMFY_KITCHEN_INT8",
                    ),
                    "skip_comm_test": _input("BOOLEAN", director_default=False),
                    "use_mmap": _input("BOOLEAN", director_default=False),
                },
                optional={
                    "ray_object_store_gb": _input("FLOAT"),
                    "ray_dashboard_address": _input("STRING"),
                    "torch_dist_address": _input("STRING"),
                    "driver_cleanup_policy": _input(
                        "COMBO",
                        enum_values=("legacy_all", "ray_devices"),
                        director_default="ray_devices",
                    ),
                    "ram_cache_max_models": _input("INT", director_default=2),
                },
                director_supplied_inputs=(
                    "ray_cluster_address",
                    "ray_cluster_namespace",
                    "GPU",
                    "GPU_SELECT",
                    "driver_cleanup_policy",
                    "ulysses_degree",
                    "ring_degree",
                    "clear_vram_after_sampling",
                    "ram_cache_max_models",
                    "cfg_degree",
                    "dp_degree",
                    "sync_ulysses",
                    "FSDP",
                    "FSDP_CPU_OFFLOAD",
                    "XFuser_attention",
                    "skip_comm_test",
                    "use_mmap",
                ),
                outputs=(_output(0, "RAY_ACTORS_INIT", "ray_actors_init"),),
            ),
            _strict(
                backends=_RAYLIGHT_ONLY,
                notes=(
                    "Director segment graphs select COMFY_KITCHEN_INT8 and the "
                    "supported RayLight fork raises without attention fallback; "
                    "control graphs may replay TORCH_FLASH only to terminate a "
                    "legacy resident pool.",
                ),
            ),
        ),
        _spec(
            "DirectorDeckRayLoraLoader",
            _RAYLIGHT,
            _object_info(
                required={
                    "lora_name": _input("COMBO"),
                    "strength_model": _input("FLOAT", director_default=1.0),
                },
                optional={"prev_ray_lora": _input("RAY_LORA")},
                director_supplied_inputs=("lora_name", "strength_model"),
                outputs=(_output(0, "RAY_LORA", "ray_lora"),),
            ),
            _strict(
                backends=_RAYLIGHT_ONLY,
                validation_method="director_owned_implementation",
                notes=_RAY_LORA_STRICT_NOTE,
            ),
        ),
        _spec(
            "DirectorDeckRayUNETLoader",
            _RAYLIGHT,
            _object_info(
                required={
                    "unet_name": _input("COMBO"),
                    "weight_dtype": _input(
                        "COMBO",
                        enum_values=(
                            "default",
                            "fp8_e4m3fn",
                            "fp8_e4m3fn_fast",
                            "fp8_e5m2",
                            "bf16",
                            "fp16",
                        ),
                        director_default="default",
                    ),
                    "ray_actors_init": _input("RAY_ACTORS_INIT"),
                },
                optional={"lora": _input("RAY_LORA")},
                outputs=(_output(0, "RAY_ACTORS", "ray_actors"),),
            ),
            _strict(
                backends=_RAYLIGHT_ONLY,
                validation_method="director_owned_implementation",
                notes=_RAY_UNET_STRICT_NOTE,
            ),
        ),
        _spec(
            "DirectorDeckRayMiniMaxH3SigmaShift",
            _RAYLIGHT,
            _object_info(
                required={
                    "ray_actors": _input("RAY_ACTORS"),
                    "shift_video": _input("FLOAT", director_default=12.0),
                    "shift_audio": _input("FLOAT", director_default=3.0),
                },
                outputs=(_output(0, "RAY_ACTORS", "ray_actors"),),
            ),
            _strict(backends=_RAYLIGHT_ONLY),
        ),
        _spec(
            "DirectorDeckRayBasicGuider",
            _RAYLIGHT,
            _object_info(
                required={
                    "ray_actors": _input("RAY_ACTORS"),
                    "conditioning": _input("CONDITIONING"),
                },
                outputs=(_output(0, "RAY_GUIDER", "guider"),),
            ),
            _strict(backends=_RAYLIGHT_ONLY),
        ),
        _spec(
            "DirectorDeckRayBasicScheduler",
            _RAYLIGHT,
            _object_info(
                required={
                    "ray_actors": _input("RAY_ACTORS"),
                    "scheduler": _input(
                        "COMBO",
                        enum_values=("simple", "normal", "karras", "beta"),
                        director_default="simple",
                    ),
                    "steps": _input("INT", director_default=25),
                    "denoise": _input("FLOAT", director_default=1.0),
                },
                outputs=(_output(0, "SIGMAS"),),
            ),
            _strict(backends=_RAYLIGHT_ONLY),
        ),
        _spec(
            "DirectorDeckRayXFuserSamplerCustomAdvanced",
            _RAYLIGHT,
            _object_info(
                required={
                    "add_noise": _input("BOOLEAN", director_default=True),
                    "noise_seed": _input("INT", director_default=0),
                    "guider": _input("RAY_GUIDER"),
                    "sampler": _input("SAMPLER"),
                    "sigmas": _input("SIGMAS"),
                    "latent_image": _input("LATENT"),
                },
                outputs=(
                    _output(0, "LATENT", "output"),
                    _output(1, "LATENT", "denoised_output"),
                    _output(2, "RAY_ACTORS", "ray_actors"),
                ),
            ),
            _strict(backends=_RAYLIGHT_ONLY),
        ),
        _spec(
            "DirectorDeckRayKill",
            _RAYLIGHT,
            _object_info(
                required={
                    "ray_actors": _input("RAY_ACTORS"),
                    "kill_mode": _input(
                        "COMBO",
                        enum_values=("Kill Workers Only", "Kill Entire Cluster"),
                        director_default="Kill Entire Cluster",
                    ),
                },
                output_node=True,
            ),
            _side_effect(
                backends=_RAYLIGHT_ONLY,
                notes=(
                    "Execution terminal that shuts down Ray workers or the "
                    "cluster and creates no persistent artifact.",
                ),
            ),
            execution_terminal_role="ray_kill",
        ),
    )


def _stage8_node_specs() -> tuple[_NodeSpec, ...]:
    """Current-only strict feature nodes; never mutate the frozen v4 set."""

    return (
        _spec(
            "DirectorStrictModelAttentionBackend",
            _DIRECTOR_STRICT_ATTENTION,
            _object_info(
                required={
                    "model": _input("MODEL"),
                    "mode": _input(
                        "COMBO",
                        enum_values=("pytorch", "ck_int8"),
                    ),
                },
                outputs=(_output(0, "MODEL", "model"),),
            ),
            _strict(
                backends=_STANDARD_ONLY,
                validation_method="strict_wrapper",
                notes=_DIRECTOR_STRICT_ATTENTION_NOTE,
            ),
        ),
        _spec(
            "DirectorStrictH3LowVramSagePatch",
            _DIRECTOR_STRICT_H3,
            _object_info(
                required={"model": _input("MODEL")},
                outputs=(_output(0, "MODEL", "model"),),
            ),
            _strict(
                backends=_STANDARD_ONLY,
                validation_method="director_owned_implementation",
                notes=_DIRECTOR_STRICT_H3_NOTE,
            ),
        ),
    )


def _sha256(payload: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _source_closure(*relative_files: str) -> tuple[str, ...]:
    """Return one canonical, duplicate-free reviewed source closure."""

    return tuple(sorted(set(relative_files)))


_DIRECTOR_GRAPH_BUILDER_SOURCE: Final = "workflow/builder.py"
_DIRECTOR_LEGACY_EMITTER_SOURCE: Final = "workflow/interpreters/_emitter.py"
_DIRECTOR_INTERPRETER_TYPES_SOURCE: Final = "workflow/interpreters/_types.py"

# Runtime support is scoped to the physical node module that a contract can
# actually reach.  This is intentionally a reviewed closure rather than a glob
# over every interpreter: adding an unrelated, default-off interpreter must not
# invalidate existing implementations.  Shared graph/emitter changes still
# invalidate every module whose adapter path executes that shared code.
DIRECTOR_NODE_ADAPTER_SOURCE_FILES: Final = FrozenMap(
    {
        _CORE: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            _DIRECTOR_INTERPRETER_TYPES_SOURCE,
            "native_templates.py",
            "workflow/interpreters/conditioning.py",
            "workflow/interpreters/decode_video.py",
            "workflow/interpreters/lora.py",
            "workflow/interpreters/shared_models.py",
            "workflow/interpreters/standard_model_path.py",
            "workflow/lora_factory.py",
        ),
        _MULTIGPU: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            _DIRECTOR_INTERPRETER_TYPES_SOURCE,
            "native_templates.py",
            "workflow/interpreters/shared_models.py",
            "workflow/interpreters/standard_model_path.py",
        ),
        _VIDEO: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            _DIRECTOR_INTERPRETER_TYPES_SOURCE,
            "native_templates.py",
            "workflow/interpreters/audio_output.py",
            "workflow/interpreters/conditioning.py",
            "workflow/interpreters/continuity.py",
            "workflow/interpreters/save_take.py",
        ),
        _AUDIO: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            _DIRECTOR_INTERPRETER_TYPES_SOURCE,
            "native_templates.py",
            "workflow/interpreters/audio_output.py",
            "workflow/interpreters/conditioning.py",
            "workflow/interpreters/continuity.py",
        ),
        _H3: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            _DIRECTOR_INTERPRETER_TYPES_SOURCE,
            "native_templates.py",
            "workflow/interpreters/conditioning.py",
            "workflow/interpreters/continuity.py",
            "workflow/interpreters/standard_model_path.py",
        ),
        _SAMPLER: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            "workflow/interpreters/sampling_raylight.py",
            "workflow/interpreters/sampling_standard.py",
        ),
        _IMAGES: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            _DIRECTOR_INTERPRETER_TYPES_SOURCE,
            "native_templates.py",
            "workflow/interpreters/continuity.py",
            "workflow/interpreters/decode_video.py",
        ),
        _LORA_DEBUG: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            "native_templates.py",
            "workflow/interpreters/lora.py",
            "workflow/lora_factory.py",
        ),
        _TURBO: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            "native_templates.py",
            "workflow/interpreters/lora.py",
            "workflow/lora_factory.py",
        ),
        _RAYLIGHT: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            _DIRECTOR_LEGACY_EMITTER_SOURCE,
            "native_templates.py",
            "workflow/interpreters/lora.py",
            "workflow/interpreters/raylight_model_path.py",
            "workflow/interpreters/sampling_raylight.py",
        ),
        _DIRECTOR_STRICT_ATTENTION: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            "workflow/v5_registry.py",
        ),
        _DIRECTOR_STRICT_H3: _source_closure(
            _DIRECTOR_GRAPH_BUILDER_SOURCE,
            "workflow/v5_registry.py",
        ),
    }
)


def _validate_director_adapter_source_files() -> None:
    required_modules = {
        item.module for item in (*_node_specs(), *_stage8_node_specs())
    }
    registered_modules = set(DIRECTOR_NODE_ADAPTER_SOURCE_FILES)
    if registered_modules != required_modules:
        missing = sorted(required_modules - registered_modules)
        unexpected = sorted(registered_modules - required_modules)
        raise RuntimeError(
            "Director node adapter source modules are incomplete: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )

    directordeck_root = Path(__file__).resolve().parents[1]
    for module, relative_files in DIRECTOR_NODE_ADAPTER_SOURCE_FILES.items():
        if not relative_files:
            raise RuntimeError(
                f"Director node adapter source closure is empty for {module}"
            )
        invalid = tuple(
            relative
            for relative in relative_files
            if Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or Path(relative).suffix != ".py"
        )
        if invalid:
            raise RuntimeError(
                f"Director node adapter source paths are invalid for {module}: "
                + ", ".join(invalid)
            )
        missing = tuple(
            relative
            for relative in relative_files
            if not (directordeck_root / relative).is_file()
        )
        if missing:
            raise RuntimeError(
                f"Director node adapter source closure is incomplete for {module}: "
                + ", ".join(missing)
            )


_validate_director_adapter_source_files()


def _director_adapter_content_digest(
    module: str,
    *,
    directordeck_root: Path | None = None,
) -> str:
    """Hash one reviewed adapter closure using only logical paths and bytes."""

    try:
        relative_files = DIRECTOR_NODE_ADAPTER_SOURCE_FILES[module]
    except KeyError as exc:
        raise KeyError(f"unknown Director node adapter module: {module}") from exc
    root = (
        Path(__file__).resolve().parents[1]
        if directordeck_root is None
        else directordeck_root
    )
    source_manifest = tuple(
        {
            "module_path": relative,
            "content_digest": "sha256:"
            + hashlib.sha256((root / relative).read_bytes()).hexdigest(),
        }
        for relative in relative_files
    )
    return _sha256(source_manifest)


DIRECTOR_NODE_ADAPTER_CONTENT_DIGESTS: Final = FrozenMap(
    {
        module: _director_adapter_content_digest(module)
        for module in DIRECTOR_NODE_ADAPTER_SOURCE_FILES
    }
)


def compute_runtime_fingerprint(
    material: RuntimeFingerprintMaterial,
) -> RuntimeFingerprint:
    """Hash Director's deterministic adapter identity material.

    This digest is compile/audit identity only. It is never compared with live
    ComfyUI source and does not authorize or certify a host implementation.
    """

    if not isinstance(material, RuntimeFingerprintMaterial):
        raise TypeError("runtime fingerprint material must be validated")
    return _sha256(material)


def _module_support_fingerprints(
    specs: tuple[_NodeSpec, ...],
) -> dict[str, str]:
    grouped: dict[str, list[_NodeSpec]] = defaultdict(list)
    for item in specs:
        grouped[item.module].append(item)

    result: dict[str, str] = {}
    for module, module_specs in grouped.items():
        ordered = sorted(module_specs, key=lambda item: item.class_type)
        adapter_binding = {
            item.class_type: {
                "contract_semantic_version": DIRECTOR_NODE_ADAPTER_SEMANTIC_VERSION,
                "director_supplied_inputs": item.object_info.director_supplied_inputs,
                "outputs": item.object_info.outputs,
                "execution_terminal_role": item.execution_terminal_role,
                "persistent_artifact_role": item.persistent_artifact_role,
                "runtime_effect_contract": item.effect,
            }
            for item in ordered
        }
        adapter_content_digest = _sha256(
            {
                "adapter_source_digest": DIRECTOR_NODE_ADAPTER_CONTENT_DIGESTS[
                    module
                ],
                "module_binding_contract": adapter_binding,
            }
        )
        support_material = RuntimeFingerprintMaterial(
            normalized_module_identity=module,
            object_info_contract_slice={
                item.class_type: item.object_info for item in ordered
            },
            adapter_module_content_digest=adapter_content_digest,
            package_version=None,
            director_wrapper_semantic_version=DIRECTOR_NODE_ADAPTER_SEMANTIC_VERSION,
        )
        result[module] = compute_runtime_fingerprint(support_material)
    return result


def _build_registry(specs: tuple[_NodeSpec, ...]) -> NodeContractRegistry:
    class_types = tuple(item.class_type for item in specs)
    if len(class_types) != len(set(class_types)):
        raise AssertionError("v4 node contract class types must be unique")

    fingerprints = _module_support_fingerprints(specs)
    registry = NodeContractRegistry(schema_version=NODE_CONTRACT_SCHEMA_VERSION)
    for item in specs:
        contract = NodeContract(
            contract_id=f"directordeck.node.{item.class_type.replace(' ', '_')}",
            semantic_version=DIRECTOR_NODE_ADAPTER_SEMANTIC_VERSION,
            class_type=item.class_type,
            allowed_python_modules=(item.module,),
            object_info_contract=item.object_info,
            output_contract=NodeOutputContract(slots=item.object_info.outputs),
            execution_terminal_role=item.execution_terminal_role,
            persistent_artifact_role=item.persistent_artifact_role,
            runtime_effect_contract=item.effect,
            supported_runtime_fingerprints=(fingerprints[item.module],),
        )
        registry = registry.register(contract)
    return registry


V4_NODE_CONTRACT_REGISTRY: Final[NodeContractRegistry] = _build_registry(
    _node_specs()
)
CURRENT_NODE_CONTRACT_REGISTRY: Final[NodeContractRegistry] = _build_registry(
    (*_node_specs(), *_stage8_node_specs())
)


def require_native_node_contract(class_type: str) -> NodeContract:
    """Return the frozen v4 contract for one prompt class type."""

    return V4_NODE_CONTRACT_REGISTRY.require(class_type)


def require_current_node_contract(class_type: str) -> NodeContract:
    """Return the current contract, including post-v4 strict feature nodes."""

    return CURRENT_NODE_CONTRACT_REGISTRY.require(class_type)


def release_supported_runtime_fingerprints(
    class_type: str,
) -> tuple[RuntimeFingerprint, ...]:
    """Return the frozen compiler-adapter identity for legacy callers."""

    return require_native_node_contract(class_type).supported_runtime_fingerprints


def current_release_supported_runtime_fingerprints(
    class_type: str,
) -> tuple[RuntimeFingerprint, ...]:
    """Return the current compiler-adapter identity for legacy callers."""

    return require_current_node_contract(class_type).supported_runtime_fingerprints


def _selected_contracts(
    class_types: Iterable[str] | None,
) -> tuple[NodeContract, ...]:
    if class_types is None:
        return tuple(V4_NODE_CONTRACT_REGISTRY.contracts.values())
    selected = tuple(dict.fromkeys(class_types))
    return tuple(require_native_node_contract(item) for item in selected)


def _selected_current_contracts(
    class_types: Iterable[str] | None,
) -> tuple[NodeContract, ...]:
    if class_types is None:
        return tuple(CURRENT_NODE_CONTRACT_REGISTRY.contracts.values())
    selected = tuple(dict.fromkeys(class_types))
    return tuple(require_current_node_contract(item) for item in selected)


def _module_policy(
    contracts: Iterable[NodeContract],
) -> FrozenMap[ClassType, ModuleIdentity]:
    policy: dict[str, str] = {}
    for contract in contracts:
        if len(contract.allowed_python_modules) != 1:
            raise AssertionError(
                "native node contract must select one exact module: "
                f"{contract.class_type}"
            )
        policy[contract.class_type] = contract.allowed_python_modules[0]
    return FrozenMap(policy)


def native_expected_module_policy(
    class_types: Iterable[str] | None = None,
) -> FrozenMap[ClassType, ModuleIdentity]:
    """Derive the legacy exact-module compatibility view from the registry."""

    return _module_policy(_selected_contracts(class_types))


def current_expected_module_policy(
    class_types: Iterable[str] | None = None,
) -> FrozenMap[ClassType, ModuleIdentity]:
    """Derive the current exact-module view without changing v4 policy."""

    return _module_policy(_selected_current_contracts(class_types))


def _provenance_for_module(module: str) -> str:
    if module == _CORE:
        return "comfy-core"
    if module == _H3:
        return "comfy-core-official-minimax-h3"
    if module.startswith("comfy_extras."):
        return "comfy-extras"
    if module == _DIRECTOR_STRICT_ATTENTION:
        return "director-owned-strict-attention"
    if module == _DIRECTOR_STRICT_H3:
        return "director-owned-strict-h3"
    if module == _TURBO:
        return "lora-custom"
    if module == _RAYLIGHT:
        return "raylight"
    raise ValueError(f"unclassified native module provenance: {module}")


def native_provenance_policy(
    class_types: Iterable[str] | None = None,
) -> FrozenMap[ClassType, str]:
    """Derive the legacy provenance labels without a class-type side table."""

    modules = native_expected_module_policy(class_types)
    return FrozenMap(
        (class_type, _provenance_for_module(module))
        for class_type, module in modules.items()
    )


def current_provenance_policy(
    class_types: Iterable[str] | None = None,
) -> FrozenMap[ClassType, str]:
    """Derive current provenance labels while preserving the v4 view."""

    modules = current_expected_module_policy(class_types)
    return FrozenMap(
        (class_type, _provenance_for_module(module))
        for class_type, module in modules.items()
    )


__all__ = [
    "CURRENT_NODE_CONTRACT_REGISTRY",
    "DIRECTOR_NODE_ADAPTER_SEMANTIC_VERSION",
    "DIRECTOR_NODE_ADAPTER_CONTENT_DIGESTS",
    "DIRECTOR_NODE_ADAPTER_SOURCE_FILES",
    "NODE_CONTRACT_SCHEMA_VERSION",
    "OBJECT_INFO_NORMALIZATION_VERSION",
    "RUNTIME_SUPPORT_FINGERPRINT_SCHEMA",
    "RuntimeFingerprintMaterial",
    "V4_NODE_CONTRACT_REGISTRY",
    "V4_OUTPUT_NEUTRAL_NODE_CLASSES",
    "compute_runtime_fingerprint",
    "current_expected_module_policy",
    "current_provenance_policy",
    "current_release_supported_runtime_fingerprints",
    "native_expected_module_policy",
    "native_provenance_policy",
    "release_supported_runtime_fingerprints",
    "require_current_node_contract",
    "require_native_node_contract",
]
