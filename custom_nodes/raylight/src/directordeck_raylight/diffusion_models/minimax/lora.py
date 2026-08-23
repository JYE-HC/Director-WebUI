# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


_MINIMAX_H3_BARE_LORA_ROOTS = (
    "blocks.",
    "final_layer.",
    "token_refiner.",
)
_MINIMAX_H3_PREFIX = "diffusion_model."
_LORA_A_SUFFIX = ".lora_A.weight"
_LORA_B_SUFFIX = ".lora_B.weight"
_LORA_ALPHA_SUFFIX = ".alpha"


def _minimax_h3_parts(model_patcher: Any) -> tuple[Any, Any]:
    base_model = getattr(model_patcher, "model", None)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    return base_model, diffusion_model


def _is_minimax_h3_model_patcher(model_patcher: Any) -> bool:
    _, diffusion_model = _minimax_h3_parts(model_patcher)
    return type(diffusion_model).__name__ == "MiniMaxH3Model"


def _is_official_h3_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    if key.startswith(_MINIMAX_H3_PREFIX):
        key = key[len(_MINIMAX_H3_PREFIX) :]
    return key.startswith(_MINIMAX_H3_BARE_LORA_ROOTS)


def _resolve_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        if part.isdigit():
            value = value[int(part)]
        else:
            value = getattr(value, part)
    return value


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def _reduced_numeric_bool(value: Any, reducer: str) -> bool:
    reduce_fn = getattr(value, reducer, None)
    if not callable(reduce_fn):
        raise TypeError(f"tensor result does not support {reducer}()")
    reduced = reduce_fn()
    item_fn = getattr(reduced, "item", None)
    if not callable(item_fn):
        raise TypeError("tensor reduction does not produce a scalar")
    return bool(item_fn())


def _validate_finite_nonzero_tensor(value: Any, *, key: str) -> None:
    try:
        finite_fn = getattr(value, "isfinite", None)
        if not callable(finite_fn) or not _reduced_numeric_bool(finite_fn(), "all"):
            raise ValueError(f"MiniMax H3 LoRA tensor must contain only finite values: {key}")

        ne_fn = getattr(value, "ne", None)
        if not callable(ne_fn) or not _reduced_numeric_bool(ne_fn(0), "any"):
            raise ValueError(f"MiniMax H3 LoRA tensor must not be all zero: {key}")
    except ValueError:
        raise
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise ValueError(
            f"MiniMax H3 LoRA tensor cannot be validated numerically: {key}"
        ) from exc


def _validate_minimax_h3_lora(
    model_patcher: Any,
    lora_state_dict: Mapping[str, Any],
) -> tuple[str, ...]:
    base_model, diffusion_model = _minimax_h3_parts(model_patcher)
    h3_keys = [key for key in lora_state_dict if _is_official_h3_key(key)]

    if len(h3_keys) != len(lora_state_dict):
        unsupported = [key for key in lora_state_dict if key not in h3_keys]
        raise ValueError(
            "Unsupported key in MiniMax H3 LoRA state dict: "
            + ", ".join(str(key) for key in unsupported[:3])
        )

    if getattr(diffusion_model, "use_adaln_curves", False) and any(
        ".adaln_proj." in key for key in h3_keys
    ):
        raise RuntimeError(
            "Raylight cannot apply a MiniMax H3 LoRA containing AdaLN weights to "
            "a pruned/curve H3 base yet. Use the full H3 base, or use "
            "MiniMaxH3TurboLoRA outside Raylight until AdaLN E-grid injection is supported."
        )

    pairs: dict[str, dict[str, Any]] = {}
    auxiliary_modules: set[str] = set()
    for key in h3_keys:
        if key.endswith(_LORA_A_SUFFIX):
            pairs.setdefault(key[: -len(_LORA_A_SUFFIX)], {})["A"] = lora_state_dict[key]
        elif key.endswith(_LORA_B_SUFFIX):
            pairs.setdefault(key[: -len(_LORA_B_SUFFIX)], {})["B"] = lora_state_dict[key]
        elif key.endswith(_LORA_ALPHA_SUFFIX):
            auxiliary_modules.add(key[: -len(_LORA_ALPHA_SUFFIX)])
            alpha_shape = _shape(lora_state_dict[key])
            if alpha_shape != () or not callable(getattr(lora_state_dict[key], "item", None)):
                raise ValueError(
                    f"MiniMax H3 LoRA alpha must be scalar for {key}: shape={alpha_shape}"
                )
            try:
                alpha = float(lora_state_dict[key].item())
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    f"MiniMax H3 LoRA alpha must be a finite non-zero scalar for {key}"
                ) from exc
            if not math.isfinite(alpha) or alpha == 0.0:
                raise ValueError(
                    f"MiniMax H3 LoRA alpha must be a finite non-zero scalar for {key}"
                )
        else:
            raise ValueError(f"Unsupported MiniMax H3 LoRA adapter key: {key}")

    if not pairs:
        raise ValueError("MiniMax H3 LoRA contains no adapter A/B pairs")
    orphan_auxiliary = sorted(auxiliary_modules.difference(pairs))
    if orphan_auxiliary:
        raise ValueError(
            "MiniMax H3 LoRA auxiliary keys have no adapter pair: "
            + ", ".join(orphan_auxiliary[:3])
        )

    for module_key, pair in pairs.items():
        missing = {"A", "B"}.difference(pair)
        if missing:
            raise ValueError(
                f"Incomplete MiniMax H3 LoRA pair for {module_key}: missing {', '.join(sorted(missing))}"
            )

        a_shape = _shape(pair["A"])
        b_shape = _shape(pair["B"])
        if a_shape is None or b_shape is None:
            raise ValueError(
                f"MiniMax H3 LoRA tensors must expose shapes for {module_key}: "
                f"A={a_shape}, B={b_shape}"
            )
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
            raise ValueError(
                f"Invalid MiniMax H3 LoRA rank for {module_key}: A={a_shape}, B={b_shape}"
            )
        _validate_finite_nonzero_tensor(
            pair["A"], key=f"{module_key}{_LORA_A_SUFFIX}"
        )
        _validate_finite_nonzero_tensor(
            pair["B"], key=f"{module_key}{_LORA_B_SUFFIX}"
        )

        target_key = (
            module_key
            if module_key.startswith(_MINIMAX_H3_PREFIX)
            else f"{_MINIMAX_H3_PREFIX}{module_key}"
        )
        try:
            module = _resolve_path(base_model, target_key)
            base_shape = _shape(module.weight)
        except (AttributeError, IndexError, KeyError, TypeError):
            base_shape = None
        if base_shape is None:
            raise ValueError(f"MiniMax H3 LoRA target does not exist: {target_key}.weight")

        expected_shape = (b_shape[0], a_shape[1])
        if base_shape != expected_shape:
            raise ValueError(
                f"MiniMax H3 LoRA shape mismatch for {module_key}.weight: "
                f"base={base_shape}, A={a_shape}, B={b_shape}, expected_base={expected_shape}"
            )

    return tuple(
        f"{module_key}.weight"
        if module_key.startswith(_MINIMAX_H3_PREFIX)
        else f"{_MINIMAX_H3_PREFIX}{module_key}.weight"
        for module_key in sorted(pairs)
    )


def is_minimax_h3_model_patcher(model_patcher: Any) -> bool:
    """Public identity check used by the worker to select the strict H3 path."""

    return _is_minimax_h3_model_patcher(model_patcher)


def minimax_h3_lora_targets(
    model_patcher: Any,
    lora_state_dict: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """Return exact patch targets for H3, or ``None`` for a non-H3 model."""

    if not _is_minimax_h3_model_patcher(model_patcher):
        return None
    return _validate_minimax_h3_lora(model_patcher, lora_state_dict)


def normalize_minimax_h3_lora_keys(
    model_patcher: Any,
    lora_state_dict: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int]:
    """Validate and map official MiniMax H3 PEFT keys to ComfyUI's namespace.

    The H3 Turbo LoRA stores keys such as
    ``blocks.0.attn.qkv_proj.lora_A.weight``. ComfyUI exposes the matching
    patch target as ``diffusion_model.blocks.0.attn.qkv_proj``. Without this
    prefix Raylight's normal and FSDP loaders silently load zero adapters.

    The input mapping is never mutated. Ambiguous duplicate spellings and
    incompatible model/LoRA shapes fail before the first sampling forward.
    """

    if not _is_minimax_h3_model_patcher(model_patcher):
        if any(_is_official_h3_key(key) for key in lora_state_dict):
            _, diffusion_model = _minimax_h3_parts(model_patcher)
            raise TypeError(
                "MiniMax H3 LoRA keys require a MiniMaxH3Model diffusion model; "
                f"got {type(diffusion_model).__name__}"
            )
        return lora_state_dict, 0

    bare_keys = [
        key
        for key in lora_state_dict
        if isinstance(key, str)
        and not key.startswith(_MINIMAX_H3_PREFIX)
        and key.startswith(_MINIMAX_H3_BARE_LORA_ROOTS)
    ]
    conflicts = [key for key in bare_keys if f"{_MINIMAX_H3_PREFIX}{key}" in lora_state_dict]
    if conflicts:
        sample = ", ".join(conflicts[:3])
        raise ValueError(
            "MiniMax H3 LoRA contains both bare and diffusion_model-prefixed keys "
            f"for the same target: {sample}"
        )

    if bare_keys:
        normalized = dict(lora_state_dict)
        for key in bare_keys:
            normalized[f"{_MINIMAX_H3_PREFIX}{key}"] = normalized.pop(key)
    else:
        normalized = lora_state_dict

    _validate_minimax_h3_lora(model_patcher, normalized)
    return normalized, len(bare_keys)


def is_minimax_h3_fused_int8_fc2(model_patcher: Any, module_key: str) -> bool:
    """Return whether H3's fused INT8 MLP path bypasses this module's forward."""

    if not _is_minimax_h3_model_patcher(model_patcher):
        return False
    if not isinstance(module_key, str) or not module_key.endswith(".mlp.fc2"):
        return False

    base_model, _ = _minimax_h3_parts(model_patcher)
    try:
        weight = _resolve_path(base_model, module_key).weight
    except (AttributeError, IndexError, KeyError, TypeError):
        return False

    candidates = [weight]
    for attr in ("data", "_local_tensor"):
        value = getattr(weight, attr, None)
        if value is not None and value is not weight:
            candidates.append(value)

    for candidate in candidates:
        layout = getattr(candidate, "_layout_cls", None)
        layout_name = layout if isinstance(layout, str) else getattr(layout, "__name__", None)
        if layout_name != "TensorWiseINT8Layout":
            continue
        params = getattr(candidate, "_params", None)
        return not getattr(params, "transposed", False)
    return False
