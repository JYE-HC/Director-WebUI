from __future__ import annotations

"""Fail-closed attention backend selection for Standard MiniMax H3 models.

ComfyUI's stock ``ModelAttentionBackend`` deliberately falls back to PyTorch
attention when a requested backend cannot be resolved.  That is convenient for
interactive workflows, but it cannot satisfy Director's output-affecting
runtime contract: a compiled ``ck_int8`` selection must either install that
exact implementation or fail before sampling.

This node therefore validates the exact H3/ModelPatcher structure, rejects an
existing backend override, proves the requested implementation and device
capability before cloning, then reflects the clone to prove that ComfyUI bound
the exact callable.  No branch returns the input MODEL as a successful no-op.
"""

import inspect
from typing import Any, Literal

import torch

import comfy.ldm.minimax.model as comfy_minimax_model
import comfy.ldm.modules.attention as comfy_attention
import comfy.model_patcher as comfy_model_patcher
from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.model_patcher import ModelPatcher


AttentionMode = Literal["pytorch", "ck_int8"]

_ATTENTION_OVERRIDE_KEY = "optimized_attention_override"
_STRICT_H3_PATCH_PREFIX = "diffusion_model.blocks."
_STRICT_H3_PATCH_SUFFIX = ".attn.forward"
_BACKEND_REGISTRY_KEYS: dict[str, str] = {
    "pytorch": "pytorch",
    "ck_int8": "comfy_kitchen_int8",
}
_OFFICIAL_CLONE = ModelPatcher.clone
_OFFICIAL_SET_MODEL_OPTIMIZED_ATTENTION = (
    ModelPatcher.set_model_optimized_attention
)


def _runtime_capability_result(
    available: bool,
    code: str,
    architecture: str | None = None,
) -> dict[str, bool | str | None]:
    """Return only stable, privacy-safe in-process probe evidence."""

    return {
        "available": available,
        "code": code,
        "architecture": architecture,
    }


def _runtime_api_contract_code() -> str | None:
    """Prove the exact reviewed ComfyUI types and methods are still active."""

    if (
        getattr(comfy_model_patcher, "ModelPatcher", None) is not ModelPatcher
        or not isinstance(ModelPatcher, type)
        or getattr(ModelPatcher, "clone", None) is not _OFFICIAL_CLONE
        or getattr(ModelPatcher, "set_model_optimized_attention", None)
        is not _OFFICIAL_SET_MODEL_OPTIMIZED_ATTENTION
    ):
        return "comfy_model_patcher_api_incompatible"
    if (
        getattr(comfy_minimax_model, "MiniMaxH3Model", None)
        is not MiniMaxH3Model
        or not isinstance(MiniMaxH3Model, type)
    ):
        return "comfy_minimax_h3_api_incompatible"
    if type(getattr(comfy_attention, "REGISTERED_ATTENTION_FUNCTIONS", None)) is not dict:
        return "attention_registry_unavailable"
    return None


def _cuda_architecture(device_index: int) -> str | None:
    try:
        capability = torch.cuda.get_device_capability(device_index)
    except Exception:
        return None
    if (
        not isinstance(capability, (tuple, list))
        or len(capability) != 2
        or type(capability[0]) is not int
        or type(capability[1]) is not int
        or capability[0] < 0
        or capability[1] < 0
    ):
        return None
    return f"sm{capability[0]}{capability[1]}"


def director_runtime_capability(
    mode: AttentionMode,
    device_index: int | None = None,
) -> dict[str, bool | str | None]:
    """Probe one exact backend without a MODEL or any graph mutation.

    ``device_index=None`` means the process default CUDA device and ``-1`` is
    the provider's explicit CPU-placement sentinel.  Every failure is reduced
    to a stable code; import, driver and third-party exception text never
    crosses the capability boundary.
    """

    try:
        selected_mode = _require_mode(mode)
    except ValueError:
        return _runtime_capability_result(False, "attention_mode_unsupported")

    api_code = _runtime_api_contract_code()
    if api_code is not None:
        return _runtime_capability_result(False, api_code)

    expected_name = (
        "attention_pytorch"
        if selected_mode == "pytorch"
        else "attention_comfy_kitchen_int8"
    )
    expected = getattr(comfy_attention, expected_name, None)
    if not callable(expected):
        return _runtime_capability_result(False, "attention_backend_unavailable")
    registry_key = _BACKEND_REGISTRY_KEYS[selected_mode]
    registry = comfy_attention.REGISTERED_ATTENTION_FUNCTIONS
    if registry.get(registry_key) is not expected:
        return _runtime_capability_result(
            False,
            "attention_backend_registration_mismatch",
        )
    if selected_mode == "pytorch":
        return _runtime_capability_result(True, "available")

    if device_index == -1:
        return _runtime_capability_result(False, "model_device_not_cuda")
    if device_index is not None and (
        type(device_index) is not int or device_index < 0
    ):
        return _runtime_capability_result(False, "cuda_device_unavailable")
    if (
        getattr(
            comfy_attention,
            "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
            False,
        )
        is not True
    ):
        return _runtime_capability_result(
            False,
            "comfy_kitchen_int8_dependency_unavailable",
        )
    kitchen = getattr(comfy_attention, "comfy_kitchen", None)
    probe = getattr(kitchen, "int8_attention_is_available", None)
    if not callable(probe):
        return _runtime_capability_result(
            False,
            "comfy_kitchen_int8_probe_unavailable",
        )
    try:
        if not bool(torch.cuda.is_available()):
            return _runtime_capability_result(False, "cuda_runtime_unavailable")
        device_count = int(torch.cuda.device_count())
    except Exception:
        return _runtime_capability_result(False, "cuda_runtime_unavailable")
    try:
        resolved_device = (
            int(torch.cuda.current_device())
            if device_index is None
            else device_index
        )
    except Exception:
        return _runtime_capability_result(False, "cuda_device_unavailable")
    if type(resolved_device) is not int or not (0 <= resolved_device < device_count):
        return _runtime_capability_result(False, "cuda_device_unavailable")
    device = torch.device("cuda", resolved_device)
    try:
        available = probe(device)
    except Exception:
        return _runtime_capability_result(
            False,
            "comfy_kitchen_int8_probe_failed",
        )
    architecture = _cuda_architecture(resolved_device)
    if available is not True:
        return _runtime_capability_result(
            False,
            "comfy_kitchen_int8_device_unavailable",
            architecture,
        )
    return _runtime_capability_result(True, "available", architecture)


def _require_exact_bound_method(
    instance: Any,
    name: str,
    expected: Any,
) -> None:
    method = getattr(instance, name, None)
    if not callable(method) or getattr(method, "__func__", None) is not expected:
        raise TypeError(
            "Director strict attention requires the audited ComfyUI "
            f"ModelPatcher.{name} implementation"
        )


def _require_h3_model_patcher(model: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(model, ModelPatcher):
        raise TypeError(
            "Director strict attention requires a ComfyUI ModelPatcher MODEL"
        )

    model_root = getattr(model, "model", None)
    diffusion_model = getattr(model_root, "diffusion_model", None)
    if type(diffusion_model) is not MiniMaxH3Model:
        actual = (
            type(diffusion_model).__name__
            if diffusion_model is not None
            else "None"
        )
        raise TypeError(
            "Director strict attention requires the exact ComfyUI "
            f"MiniMaxH3Model diffusion model; got {actual}"
        )

    _require_exact_bound_method(model, "clone", _OFFICIAL_CLONE)
    _require_exact_bound_method(
        model,
        "set_model_optimized_attention",
        _OFFICIAL_SET_MODEL_OPTIMIZED_ATTENTION,
    )

    model_options = getattr(model, "model_options", None)
    if type(model_options) is not dict:
        raise TypeError(
            "Director strict attention requires exact dict model_options"
        )
    transformer_options = model_options.get("transformer_options")
    if type(transformer_options) is not dict:
        raise TypeError(
            "Director strict attention requires exact dict transformer_options"
        )
    return model_root, transformer_options


def _require_no_attention_override(transformer_options: dict[str, Any]) -> None:
    if _ATTENTION_OVERRIDE_KEY in transformer_options:
        raise ValueError(
            "Director strict attention refuses a MODEL that already has an "
            "optimized attention override"
        )


def _require_no_strict_h3_sage_patch(model: ModelPatcher) -> None:
    for attribute in ("object_patches", "object_patches_backup"):
        patches = getattr(model, attribute, None)
        if not isinstance(patches, dict):
            raise TypeError(
                "Director strict attention requires exact ModelPatcher "
                f"{attribute} state"
            )
        if any(
            isinstance(key, str)
            and key.startswith(_STRICT_H3_PATCH_PREFIX)
            and key.endswith(_STRICT_H3_PATCH_SUFFIX)
            for key in patches
        ):
            raise RuntimeError(
                "Director strict attention refuses a MODEL with an existing "
                "strict H3 Sage attention patch"
            )


def _require_mode(mode: Any) -> AttentionMode:
    if type(mode) is not str or mode not in _BACKEND_REGISTRY_KEYS:
        raise ValueError(
            "Director strict attention mode must be exactly 'pytorch' or "
            "'ck_int8'"
        )
    return mode  # type: ignore[return-value]


def _registered_backend(registry_key: str) -> Any:
    registry = getattr(
        comfy_attention,
        "REGISTERED_ATTENTION_FUNCTIONS",
        None,
    )
    if type(registry) is not dict:
        raise RuntimeError(
            "Director strict attention cannot prove the ComfyUI attention "
            "registry"
        )
    return registry.get(registry_key)


def _require_backend(model: ModelPatcher, mode: AttentionMode) -> Any:
    registry_key = _BACKEND_REGISTRY_KEYS[mode]
    expected_name = (
        "attention_pytorch"
        if mode == "pytorch"
        else "attention_comfy_kitchen_int8"
    )
    expected = getattr(comfy_attention, expected_name, None)
    if not callable(expected):
        raise RuntimeError(
            f"Director strict attention backend '{mode}' is unavailable"
        )

    if mode == "ck_int8":
        load_device = getattr(model, "load_device", None)
        if not isinstance(load_device, torch.device) or load_device.type != "cuda":
            raise RuntimeError(
                "Director strict attention ck_int8 requires a CUDA model device"
            )
        if (
            getattr(
                comfy_attention,
                "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
                False,
            )
            is not True
        ):
            raise RuntimeError(
                "Director strict attention ck_int8 is unavailable in this "
                "ComfyUI runtime"
            )
        kitchen = getattr(comfy_attention, "comfy_kitchen", None)
        probe = getattr(kitchen, "int8_attention_is_available", None)
        if not callable(probe):
            raise RuntimeError(
                "Director strict attention cannot prove ck_int8 device "
                "capability"
            )
        try:
            available = probe(load_device)
        except Exception:
            raise RuntimeError(
                "Director strict attention could not verify ck_int8 device "
                "capability"
            ) from None
        if available is not True:
            raise RuntimeError(
                "Director strict attention ck_int8 is unavailable on the "
                "MODEL device"
            )

    registered = _registered_backend(registry_key)
    if registered is not expected:
        raise RuntimeError(
            f"Director strict attention cannot prove the exact '{mode}' "
            "backend registration"
        )
    return expected


def _require_exact_installed_override(
    transformer_options: dict[str, Any],
    backend: Any,
) -> None:
    override = transformer_options.get(_ATTENTION_OVERRIDE_KEY)
    if not callable(override):
        raise RuntimeError(
            "Director strict attention did not install an attention override"
        )
    try:
        closure = inspect.getclosurevars(override)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Director strict attention cannot inspect the installed override"
        ) from exc
    if closure.nonlocals != {"optimized_attention": backend}:
        raise RuntimeError(
            "Director strict attention installed an unproved or fallback "
            "backend"
        )

    backend_container = getattr(backend, "container_function", None)
    if backend_container is None:
        if hasattr(override, "container_function"):
            raise RuntimeError(
                "Director strict attention installed unexpected container "
                "semantics"
            )
    elif getattr(override, "container_function", None) is not backend_container:
        raise RuntimeError(
            "Director strict attention did not preserve exact container "
            "semantics"
        )


class DirectorStrictModelAttentionBackend:
    """Install one exact Standard H3 attention backend or fail closed."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["pytorch", "ck_int8"],),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, mode: Any) -> bool | str:
        try:
            _require_mode(mode)
        except ValueError as exc:
            return str(exc)
        return True

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "DirectorDeck/model/strict"

    def patch(self, model: Any, mode: Any) -> tuple[ModelPatcher]:
        selected_mode = _require_mode(mode)
        original_root, original_transformer_options = _require_h3_model_patcher(
            model
        )
        _require_no_attention_override(original_transformer_options)
        _require_no_strict_h3_sage_patch(model)

        # Resolve every dependency and device capability before clone(), so an
        # unavailable strict feature has no model-side effect at all.
        backend = _require_backend(model, selected_mode)
        original_top_level_keys = frozenset(model.model_options)
        original_transformer_keys = frozenset(original_transformer_options)

        cloned = model.clone()
        if cloned is model:
            raise RuntimeError(
                "Director strict attention clone returned the input MODEL"
            )
        cloned_root, cloned_transformer_options = _require_h3_model_patcher(cloned)
        if cloned_root is not original_root:
            raise RuntimeError(
                "Director strict attention clone changed the underlying H3 model"
            )
        if cloned.model_options is model.model_options:
            raise RuntimeError(
                "Director strict attention clone shares mutable model_options"
            )
        if cloned_transformer_options is original_transformer_options:
            raise RuntimeError(
                "Director strict attention clone shares mutable "
                "transformer_options"
            )
        _require_no_attention_override(cloned_transformer_options)
        _require_no_strict_h3_sage_patch(cloned)
        if frozenset(cloned.model_options) != original_top_level_keys:
            raise RuntimeError(
                "Director strict attention clone changed model option structure"
            )
        if frozenset(cloned_transformer_options) != original_transformer_keys:
            raise RuntimeError(
                "Director strict attention clone changed transformer option "
                "structure"
            )

        cloned.set_model_optimized_attention(backend)

        if _ATTENTION_OVERRIDE_KEY in original_transformer_options:
            raise RuntimeError(
                "Director strict attention mutated the input MODEL"
            )
        if frozenset(cloned.model_options) != original_top_level_keys:
            raise RuntimeError(
                "Director strict attention changed unexpected model options"
            )
        if frozenset(cloned_transformer_options) != (
            original_transformer_keys | {_ATTENTION_OVERRIDE_KEY}
        ):
            raise RuntimeError(
                "Director strict attention changed unexpected transformer "
                "options"
            )
        _require_exact_installed_override(cloned_transformer_options, backend)

        # Recheck live registration after mutation.  This catches a resolver
        # drift during node execution instead of returning unproved evidence.
        if _registered_backend(_BACKEND_REGISTRY_KEYS[selected_mode]) is not backend:
            raise RuntimeError(
                "Director strict attention backend registration changed during "
                "installation"
            )
        return (cloned,)


NODE_CLASS_MAPPINGS = {
    "DirectorStrictModelAttentionBackend": DirectorStrictModelAttentionBackend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DirectorStrictModelAttentionBackend": (
        "DirectorDeck Strict MiniMax H3 Attention Backend"
    ),
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "director_runtime_capability",
]
