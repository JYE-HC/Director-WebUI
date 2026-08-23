from __future__ import annotations

"""Director-owned fail-closed MiniMax H3 Sage attention patch.

The stock ComfyUI Sage attention adapter deliberately falls back to PyTorch
when its kernel raises.  That behaviour is useful for an interactive host, but
it cannot prove the runtime effect of an output-affecting Director feature.
This node therefore calls the audited Sage entry point directly and patches
every MiniMax H3 DiT self-attention block only after the complete patch set has
been validated.

The package intentionally has no dependency on KJNodes.  SageAttention is a
GPU-specific host dependency and is probed in the running ComfyUI environment;
it is not installed implicitly by this package.
"""

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable

import torch

try:
    import sageattention as _SAGE_MODULE
except Exception:
    _SAGE_MODULE = None

try:
    import sageattention.core as _SAGE_CORE_MODULE
except Exception:
    _SAGE_CORE_MODULE = None

try:
    import comfy.model_management as _MODEL_MANAGEMENT
    import comfy.model_patcher as _MODEL_PATCHER_MODULE
    from comfy.ldm.minimax.model import (
        Attention as _MINIMAX_H3_ATTENTION_TYPE,
        DiTBlock as _MINIMAX_H3_BLOCK_TYPE,
        MiniMaxH3Model as _MINIMAX_H3_MODEL_TYPE,
    )
    from comfy.model_patcher import ModelPatcher as _MODEL_PATCHER_TYPE
    from comfy.quant_ops import ck as _CK
except Exception:
    _MODEL_MANAGEMENT = None
    _MODEL_PATCHER_MODULE = None
    _MODEL_PATCHER_TYPE = None
    _MINIMAX_H3_ATTENTION_TYPE = None
    _MINIMAX_H3_BLOCK_TYPE = None
    _MINIMAX_H3_MODEL_TYPE = None
    _CK = None


_PATCH_PREFIX = "diffusion_model.blocks"
_ATTENTION_OVERRIDE_KEY = "optimized_attention_override"
_SUPPORTED_CUDA_ARCHITECTURES = frozenset(
    {"sm80", "sm86", "sm89", "sm90", "sm120", "sm121"}
)
_SAGE_COMPILED_FLAG_KERNELS = {
    "sm80": ("SM80_ENABLED", "sageattn_qk_int8_pv_fp16_cuda"),
    "sm89": ("SM89_ENABLED", "sageattn_qk_int8_pv_fp8_cuda"),
    "sm90": ("SM90_ENABLED", "sageattn_qk_int8_pv_fp8_cuda_sm90"),
    # SageAttention dispatches Blackwell through its SM89-family extension.
    "sm120": ("SM89_ENABLED", "sageattn_qk_int8_pv_fp8_cuda"),
    "sm121": ("SM89_ENABLED", "sageattn_qk_int8_pv_fp8_cuda"),
}
_SAGE_CALLABLE_ONLY_KERNELS = {
    # SM86 is SageAttention's Triton route and has no compiled-extension flag.
    "sm86": "sageattn_qk_int8_pv_fp16_triton",
}
_ALLOWED_ATTENTION_DTYPES = frozenset({torch.float16, torch.bfloat16})
_ORIGINAL_H3_ATTENTION_FORWARD = (
    getattr(_MINIMAX_H3_ATTENTION_TYPE, "forward", None)
    if _MINIMAX_H3_ATTENTION_TYPE is not None
    else None
)
_OFFICIAL_MODEL_PATCHER_CLONE = getattr(_MODEL_PATCHER_TYPE, "clone", None)
_OFFICIAL_MODEL_PATCHER_GET_OBJECT = getattr(
    _MODEL_PATCHER_TYPE,
    "get_model_object",
    None,
)
_OFFICIAL_MODEL_PATCHER_ADD_OBJECT_PATCH = getattr(
    _MODEL_PATCHER_TYPE,
    "add_object_patch",
    None,
)


@dataclass(frozen=True, slots=True)
class RuntimeCapability:
    """Privacy-safe, stable result for catalog/preflight capability checks."""

    available: bool
    code: str
    architecture: str | None = None

    def as_dict(self) -> dict[str, bool | str | None]:
        return {
            "available": self.available,
            "code": self.code,
            "architecture": self.architecture,
        }


@dataclass(frozen=True, slots=True)
class _RuntimeBinding:
    sageattn: Callable[..., Any]
    architecture: str
    device_index: int


@dataclass(frozen=True, slots=True)
class _PatchTarget:
    key: str
    attention: Any


def _unavailable(code: str, architecture: str | None = None) -> RuntimeCapability:
    return RuntimeCapability(False, code, architecture)


def _sage_callable() -> Callable[..., Any] | None:
    candidate = getattr(_SAGE_MODULE, "sageattn", None)
    return candidate if callable(candidate) else None


def _sage_api_is_compatible(candidate: Callable[..., Any]) -> bool:
    try:
        # Capability evidence is collected from ``sageattention.core``.  The
        # function we actually execute must therefore be that exact function,
        # not an independently wrapped or monkey-patched top-level callable.
        if (
            _SAGE_CORE_MODULE is None
            or getattr(_SAGE_CORE_MODULE, "sageattn", None) is not candidate
        ):
            return False
        parameters = inspect.signature(candidate).parameters
    except (TypeError, ValueError):
        return False
    except Exception:
        return False
    names = tuple(parameters)
    if names[:3] != ("q", "k", "v"):
        return False
    return "tensor_layout" in parameters and "is_causal" in parameters


def _normalize_cuda_architecture(capability: Any) -> str | None:
    if (
        not isinstance(capability, Sequence)
        or isinstance(capability, (str, bytes))
        or len(capability) != 2
    ):
        return None
    major, minor = capability
    if type(major) is not int or type(minor) is not int:
        return None
    if not (0 <= major <= 99 and 0 <= minor <= 9):
        return None
    return f"sm{major}{minor}"


def _sage_kernel_is_compiled(architecture: str) -> bool:
    """Prove the exact Sage route without executing a GPU kernel.

    ``get_cuda_arch_versions()`` reports attached device capabilities; it says
    nothing about which optional SageAttention extension imported.  Supported
    SageAttention releases expose fail-closed ``SM*_ENABLED`` flags for that
    purpose.  A release without the corresponding exact flag remains
    unavailable: a Python/C-extension callable alone cannot prove that its
    binary contains code for the attached GPU architecture.  No exception
    detail or module path crosses the public capability boundary.
    """

    core = _SAGE_CORE_MODULE
    if core is None:
        return False

    try:
        callable_only = _SAGE_CALLABLE_ONLY_KERNELS.get(architecture)
        if callable_only is not None:
            return callable(getattr(core, callable_only, None))

        requirement = _SAGE_COMPILED_FLAG_KERNELS.get(architecture)
        if requirement is None:
            return False
        flag_name, kernel_name = requirement
        return getattr(core, flag_name, None) is True and callable(
            getattr(core, kernel_name, None)
        )
    except Exception:
        return False


def runtime_capability(device_index: int | None = None) -> RuntimeCapability:
    """Return whether the exact strict runtime can execute in this process.

    Every failure maps to a stable code.  Raw import, driver, and device errors
    are deliberately excluded from this public boundary.
    """

    if device_index == -1:
        return _unavailable("model_device_not_cuda")
    if device_index is not None and (
        type(device_index) is not int or device_index < 0
    ):
        return _unavailable("cuda_device_unavailable")

    if (
        _MODEL_PATCHER_MODULE is None
        or _MODEL_PATCHER_TYPE is None
        or getattr(_MODEL_PATCHER_MODULE, "ModelPatcher", None)
        is not _MODEL_PATCHER_TYPE
        or _OFFICIAL_MODEL_PATCHER_CLONE is None
        or getattr(_MODEL_PATCHER_TYPE, "clone", None)
        is not _OFFICIAL_MODEL_PATCHER_CLONE
        or _OFFICIAL_MODEL_PATCHER_GET_OBJECT is None
        or getattr(_MODEL_PATCHER_TYPE, "get_model_object", None)
        is not _OFFICIAL_MODEL_PATCHER_GET_OBJECT
        or _OFFICIAL_MODEL_PATCHER_ADD_OBJECT_PATCH is None
        or getattr(_MODEL_PATCHER_TYPE, "add_object_patch", None)
        is not _OFFICIAL_MODEL_PATCHER_ADD_OBJECT_PATCH
    ):
        return _unavailable("comfy_model_patcher_api_unavailable")

    if (
        _MINIMAX_H3_MODEL_TYPE is None
        or _MINIMAX_H3_BLOCK_TYPE is None
        or _MINIMAX_H3_ATTENTION_TYPE is None
        or _ORIGINAL_H3_ATTENTION_FORWARD is None
        or _MODEL_MANAGEMENT is None
        or _CK is None
        or not callable(getattr(_MODEL_MANAGEMENT, "cast_to", None))
        or not callable(getattr(_CK, "rms_rope_split_half", None))
        or not callable(getattr(_CK, "rms_rope_split_half_", None))
    ):
        return _unavailable("comfy_minimax_h3_api_unavailable")

    sageattn = _sage_callable()
    if sageattn is None:
        return _unavailable("sageattention_dependency_missing")
    if not _sage_api_is_compatible(sageattn):
        return _unavailable("sageattention_api_incompatible")

    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
    except Exception:
        return _unavailable("cuda_runtime_unavailable")
    if not cuda_available or device_count <= 0:
        return _unavailable("cuda_runtime_unavailable")

    try:
        resolved_device = (
            int(torch.cuda.current_device())
            if device_index is None
            else device_index
        )
        if type(resolved_device) is not int or not (
            0 <= resolved_device < device_count
        ):
            return _unavailable("cuda_device_unavailable")
        architecture = _normalize_cuda_architecture(
            torch.cuda.get_device_capability(resolved_device)
        )
    except Exception:
        return _unavailable("cuda_device_unavailable")
    if architecture is None or architecture not in _SUPPORTED_CUDA_ARCHITECTURES:
        return _unavailable("cuda_architecture_unsupported", architecture)

    if not _sage_kernel_is_compiled(architecture):
        return _unavailable("sageattention_kernel_unavailable", architecture)
    return RuntimeCapability(True, "available", architecture)


def director_runtime_capability(
    device_index: int | None = None,
) -> dict[str, bool | str | None]:
    """Stable module-level hook used by Director's in-process provider."""

    return runtime_capability(device_index=device_index).as_dict()


def _model_cuda_device_index(model: Any) -> int | None:
    load_device = getattr(model, "load_device", None)
    if load_device is None:
        return None
    if getattr(load_device, "type", None) != "cuda":
        raise RuntimeError(
            "Director strict H3 Sage runtime is unavailable: "
            "model_device_not_cuda"
        )
    index = getattr(load_device, "index", None)
    if index is None:
        return None
    if type(index) is not int or index < 0:
        raise RuntimeError(
            "Director strict H3 Sage runtime is unavailable: "
            "cuda_device_unavailable"
        )
    return index


def _require_runtime_binding(model: Any) -> _RuntimeBinding:
    device_index = _model_cuda_device_index(model)
    capability = runtime_capability(device_index=device_index)
    if not capability.available or capability.architecture is None:
        raise RuntimeError(
            "Director strict H3 Sage runtime is unavailable: "
            f"{capability.code}"
        )
    sageattn = _sage_callable()
    if sageattn is None or not _sage_api_is_compatible(sageattn):
        # Guard the tiny import/probe TOCTOU window without changing the public
        # failure vocabulary.
        raise RuntimeError(
            "Director strict H3 Sage runtime is unavailable: "
            "sageattention_api_incompatible"
        )
    resolved_index = (
        int(torch.cuda.current_device())
        if device_index is None
        else device_index
    )
    return _RuntimeBinding(
        sageattn=sageattn,
        architecture=capability.architecture,
        device_index=resolved_index,
    )


def _patcher_object_patches(model: Any, *, backup: bool = False) -> Mapping[str, Any]:
    attribute = "object_patches_backup" if backup else "object_patches"
    patches = getattr(model, attribute, None)
    if not isinstance(patches, Mapping):
        raise TypeError(
            "Director strict H3 Sage requires a ModelPatcher with "
            f"a mapping {attribute}"
        )
    return patches


def _same_identity_mapping(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return set(actual) == set(expected) and all(
        actual[key] is expected[key] for key in expected
    )


def _require_model_patcher_api(model: Any) -> None:
    if _MODEL_PATCHER_TYPE is None or not isinstance(model, _MODEL_PATCHER_TYPE):
        raise TypeError(
            "Director strict H3 Sage requires a real ComfyUI ModelPatcher"
        )
    expected_methods = (
        ("clone", _OFFICIAL_MODEL_PATCHER_CLONE),
        ("get_model_object", _OFFICIAL_MODEL_PATCHER_GET_OBJECT),
        ("add_object_patch", _OFFICIAL_MODEL_PATCHER_ADD_OBJECT_PATCH),
    )
    for method, expected in expected_methods:
        bound = getattr(model, method, None)
        if (
            expected is None
            or not callable(bound)
            or getattr(bound, "__func__", None) is not expected
        ):
            raise TypeError(
                "Director strict H3 Sage requires the audited ComfyUI "
                f"ModelPatcher.{method} implementation"
            )
    _patcher_object_patches(model)
    _patcher_object_patches(model, backup=True)


def _require_no_global_attention_override(model: Any) -> None:
    """Reject the mutually exclusive Standard attention feature contract."""

    model_options = getattr(model, "model_options", None)
    if type(model_options) is not dict:
        raise TypeError(
            "Director strict H3 Sage requires exact dict model_options"
        )
    transformer_options = model_options.get("transformer_options")
    if type(transformer_options) is not dict:
        raise TypeError(
            "Director strict H3 Sage requires exact dict transformer_options"
        )
    if _ATTENTION_OVERRIDE_KEY in transformer_options:
        raise RuntimeError(
            "Director strict H3 Sage refuses a MODEL with an existing "
            "optimized attention override"
        )


def _require_exact_attention_structure(attention: Any, *, index: int) -> None:
    if type(attention) is not _MINIMAX_H3_ATTENTION_TYPE:
        raise TypeError(
            "Director strict H3 Sage requires the exact MiniMax H3 Attention "
            f"type at block {index}"
        )
    if "forward" in getattr(attention, "__dict__", {}):
        raise RuntimeError(
            "Director strict H3 Sage found a pre-existing attention forward "
            f"override at block {index}"
        )
    if getattr(type(attention), "forward", None) is not _ORIGINAL_H3_ATTENTION_FORWARD:
        raise RuntimeError(
            "Director strict H3 Sage found a changed attention class contract "
            f"at block {index}"
        )

    heads = getattr(attention, "heads", None)
    head_dim = getattr(attention, "head_dim", None)
    if type(heads) is not int or heads <= 0 or type(head_dim) is not int or head_dim <= 0:
        raise ValueError(
            "Director strict H3 Sage found invalid attention dimensions at "
            f"block {index}"
        )
    for name in ("qkv_proj", "q_norm", "k_norm", "out_proj"):
        if not callable(getattr(attention, name, None)):
            raise TypeError(
                "Director strict H3 Sage found an incomplete attention module "
                f"at block {index}: {name}"
            )
    for name in ("q_norm", "k_norm"):
        norm = getattr(attention, name)
        if not hasattr(norm, "weight"):
            raise TypeError(
                "Director strict H3 Sage found an incomplete attention norm "
                f"at block {index}: {name}.weight"
            )
        epsilon = getattr(norm, "eps", None)
        if not isinstance(epsilon, (int, float)) or not math.isfinite(float(epsilon)):
            raise ValueError(
                "Director strict H3 Sage found an invalid attention norm "
                f"epsilon at block {index}: {name}"
            )


def _precompute_patch_targets(model: Any) -> tuple[Any, tuple[_PatchTarget, ...]]:
    _require_model_patcher_api(model)
    _require_no_global_attention_override(model)
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception as exc:
        raise TypeError(
            "Director strict H3 Sage could not read the diffusion model"
        ) from exc
    if type(diffusion_model) is not _MINIMAX_H3_MODEL_TYPE:
        actual = type(diffusion_model).__name__ if diffusion_model is not None else "None"
        raise TypeError(
            "Director strict H3 Sage requires the exact MiniMaxH3Model; "
            f"got {actual}"
        )

    blocks = getattr(diffusion_model, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) == 0:
        raise TypeError(
            "Director strict H3 Sage requires a non-empty MiniMax H3 ModuleList"
        )

    active_patches = _patcher_object_patches(model)
    backup_patches = _patcher_object_patches(model, backup=True)
    targets: list[_PatchTarget] = []
    seen_blocks: set[int] = set()
    seen_attentions: set[int] = set()
    for index, block in enumerate(blocks):
        if type(block) is not _MINIMAX_H3_BLOCK_TYPE:
            raise TypeError(
                "Director strict H3 Sage requires the exact MiniMax H3 DiTBlock "
                f"type at block {index}"
            )
        if id(block) in seen_blocks:
            raise RuntimeError(
                "Director strict H3 Sage found a repeated DiT block object"
            )
        seen_blocks.add(id(block))
        attention = getattr(block, "attn", None)
        _require_exact_attention_structure(attention, index=index)
        if id(attention) in seen_attentions:
            raise RuntimeError(
                "Director strict H3 Sage found a repeated attention object"
            )
        seen_attentions.add(id(attention))

        key = f"{_PATCH_PREFIX}.{index}.attn.forward"
        if key in active_patches or key in backup_patches:
            raise RuntimeError(
                "Director strict H3 Sage patch target is already owned: "
                f"{key}"
            )
        try:
            current_forward = model.get_model_object(key)
        except Exception as exc:
            raise TypeError(
                "Director strict H3 Sage could not resolve patch target: "
                f"{key}"
            ) from exc
        if (
            getattr(current_forward, "__self__", None) is not attention
            or getattr(current_forward, "__func__", None)
            is not _ORIGINAL_H3_ATTENTION_FORWARD
        ):
            raise RuntimeError(
                "Director strict H3 Sage patch target does not expose the "
                f"audited forward: {key}"
            )
        targets.append(_PatchTarget(key=key, attention=attention))

    if len(targets) != len(blocks) or len({target.key for target in targets}) != len(blocks):
        raise RuntimeError(
            "Director strict H3 Sage could not precompute one patch per block"
        )
    return diffusion_model, tuple(targets)


def _require_runtime_attention_tensor(
    value: Any,
    *,
    binding: _RuntimeBinding,
) -> None:
    if not torch.is_tensor(value) or value.ndim != 2:
        raise TypeError(
            "Director strict H3 Sage attention expects a rank-2 tensor"
        )
    if value.device.type != "cuda":
        raise RuntimeError(
            "Director strict H3 Sage attention requires a CUDA tensor"
        )
    if value.dtype not in _ALLOWED_ATTENTION_DTYPES:
        raise TypeError(
            "Director strict H3 Sage attention requires float16 or bfloat16"
        )
    index = value.device.index
    if index is None:
        try:
            index = int(torch.cuda.current_device())
        except Exception as exc:
            raise RuntimeError(
                "Director strict H3 Sage attention cannot resolve its CUDA device"
            ) from exc
    if index != binding.device_index:
        raise RuntimeError(
            "Director strict H3 Sage attention ran on an unapproved CUDA device"
        )
    try:
        architecture = _normalize_cuda_architecture(
            torch.cuda.get_device_capability(index)
        )
    except Exception as exc:
        raise RuntimeError(
            "Director strict H3 Sage attention cannot verify its CUDA architecture"
        ) from exc
    if architecture != binding.architecture:
        raise RuntimeError(
            "Director strict H3 Sage attention CUDA architecture changed"
        )


def _run_strict_h3_sage_forward(
    attention: Any,
    binding: _RuntimeBinding,
    x: Any,
    rope_freqs: Any = None,
    transformer_options: Any = None,
) -> Any:
    """Execute the exact H3 attention path with no exception fallback."""

    _require_runtime_attention_tensor(x, binding=binding)
    if transformer_options is not None and not isinstance(transformer_options, Mapping):
        raise TypeError("Director strict H3 Sage transformer_options must be a mapping")

    sequence_length = x.shape[0]
    heads = attention.heads
    head_dim = attention.head_dim
    inner = heads * head_dim
    qkv = attention.qkv_proj(x)
    if not torch.is_tensor(qkv) or tuple(qkv.shape) != (
        sequence_length,
        inner * 3,
    ):
        raise RuntimeError(
            "Director strict H3 Sage qkv projection violated its shape contract"
        )
    q, k, v = qkv.split(inner, dim=-1)
    q = q.view(1, sequence_length, heads, head_dim)
    k = k.view(1, sequence_length, heads, head_dim)
    v = v.view(1, sequence_length, heads, head_dim)

    if rope_freqs is not None:
        if not torch.is_tensor(rope_freqs) or rope_freqs.ndim < 3:
            raise TypeError(
                "Director strict H3 Sage rope_freqs contract is invalid"
            )
        cast_to = getattr(_MODEL_MANAGEMENT, "cast_to")
        q_weight = cast_to(attention.q_norm.weight, device=x.device)
        k_weight = cast_to(attention.k_norm.weight, device=x.device)
        rotation_dimension = rope_freqs.shape[-3] * 2
        if bool(getattr(_MODEL_MANAGEMENT, "in_training", False)):
            q, k = _CK.rms_rope_split_half(
                q,
                k,
                rope_freqs,
                q_weight,
                k_weight,
                epsilon=attention.q_norm.eps,
                rot_dim=rotation_dimension,
            )
        else:
            _CK.rms_rope_split_half_(
                q,
                k,
                rope_freqs,
                q_weight,
                k_weight,
                epsilon=attention.q_norm.eps,
                rot_dim=rotation_dimension,
            )
    else:
        q = attention.q_norm(q)
        k = attention.k_norm(k)

    # Do not wrap this call in try/except.  A Sage runtime error must terminate
    # the prompt instead of silently switching algorithms.
    output = binding.sageattn(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=False,
    )
    if not torch.is_tensor(output) or tuple(output.shape) != (
        1,
        sequence_length,
        heads,
        head_dim,
    ):
        raise RuntimeError(
            "Director strict H3 Sage kernel violated its output shape contract"
        )
    if output.device != x.device or output.dtype != x.dtype:
        raise RuntimeError(
            "Director strict H3 Sage kernel violated its output tensor contract"
        )
    return attention.out_proj(output.reshape(sequence_length, inner))


def _bound_strict_forward(
    attention: Any,
    binding: _RuntimeBinding,
) -> MethodType:
    def strict_forward(
        self: Any,
        x: Any,
        rope_freqs: Any = None,
        transformer_options: Any = None,
    ) -> Any:
        return _run_strict_h3_sage_forward(
            self,
            binding,
            x,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )

    strict_forward.__name__ = "director_strict_h3_sage_forward"
    strict_forward.__qualname__ = (
        "DirectorStrictH3LowVramSagePatch.director_strict_h3_sage_forward"
    )
    return MethodType(strict_forward, attention)


def _install_all_patches(
    model: Any,
    diffusion_model: Any,
    targets: tuple[_PatchTarget, ...],
    binding: _RuntimeBinding,
) -> Any:
    original_active = dict(_patcher_object_patches(model))
    original_backup = dict(_patcher_object_patches(model, backup=True))
    clone = model.clone()
    if clone is model:
        raise RuntimeError("Director strict H3 Sage ModelPatcher clone was not isolated")
    _require_model_patcher_api(clone)
    _require_no_global_attention_override(clone)
    if clone.get_model_object("diffusion_model") is not diffusion_model:
        raise RuntimeError(
            "Director strict H3 Sage ModelPatcher clone changed the diffusion model"
        )
    clone_active_mapping = _patcher_object_patches(clone)
    clone_backup_mapping = _patcher_object_patches(clone, backup=True)
    if clone_active_mapping is _patcher_object_patches(model) or (
        clone_backup_mapping is _patcher_object_patches(model, backup=True)
    ):
        raise RuntimeError(
            "Director strict H3 Sage ModelPatcher clone shared object patch state"
        )
    clone_active_before = dict(clone_active_mapping)
    clone_backup_before = dict(clone_backup_mapping)
    if not _same_identity_mapping(
        clone_active_before,
        original_active,
    ) or not _same_identity_mapping(clone_backup_before, original_backup):
        raise RuntimeError(
            "Director strict H3 Sage ModelPatcher clone changed existing object patches"
        )

    installed: dict[str, MethodType] = {}
    for target in targets:
        forward = _bound_strict_forward(target.attention, binding)
        clone.add_object_patch(target.key, forward)
        installed[target.key] = forward

    clone_active_after = _patcher_object_patches(clone)
    expected_keys = set(clone_active_before).union(installed)
    if set(clone_active_after) != expected_keys:
        raise RuntimeError(
            "Director strict H3 Sage patch installation changed the object patch set"
        )
    for key, forward in installed.items():
        if clone_active_after.get(key) is not forward:
            raise RuntimeError(
                "Director strict H3 Sage patch installation was incomplete: "
                f"{key}"
            )
        if clone.get_model_object(key) is not forward:
            raise RuntimeError(
                "Director strict H3 Sage patch reflection failed: "
                f"{key}"
            )
    if not _same_identity_mapping(
        _patcher_object_patches(model),
        original_active,
    ) or not _same_identity_mapping(
        _patcher_object_patches(model, backup=True),
        original_backup,
    ):
        raise RuntimeError(
            "Director strict H3 Sage patch installation mutated the input model"
        )
    return clone


class DirectorStrictH3LowVramSagePatch:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_strict_h3_sage"
    CATEGORY = "DirectorDeck/strict"
    DESCRIPTION = (
        "Fail-closed MiniMax H3 Sage attention patch. Requires a compatible "
        "SageAttention build and CUDA GPU; runtime kernel failures never fall "
        "back to another attention implementation."
    )

    def apply_strict_h3_sage(self, model: Any) -> tuple[Any]:
        binding = _require_runtime_binding(model)
        diffusion_model, targets = _precompute_patch_targets(model)
        return (
            _install_all_patches(
                model,
                diffusion_model,
                targets,
                binding,
            ),
        )


NODE_CLASS_MAPPINGS = {
    "DirectorStrictH3LowVramSagePatch": DirectorStrictH3LowVramSagePatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectorStrictH3LowVramSagePatch": (
        "Director Strict MiniMax H3 Low-VRAM Sage Patch"
    ),
}
