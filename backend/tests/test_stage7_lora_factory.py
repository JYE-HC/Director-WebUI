from __future__ import annotations

from dataclasses import dataclass

import pytest

from directordeck.workflow.lora_factory import (
    LoraAdapterResolutionError,
    LoraLoaderBindingKey,
    resolve_raylight_lora_adapter,
    resolve_standard_lora_adapter,
)


@dataclass(frozen=True)
class _Override:
    lora_filename: str
    adapter_id: str
    options: dict[str, bool]


def _binding(
    lora_filename: str = "loras/style.safetensors",
) -> LoraLoaderBindingKey:
    return LoraLoaderBindingKey(
        family="fl2va",
        model_filename="models/minimax-h3.safetensors",
        lora_filename=lora_filename,
    )


@pytest.mark.parametrize(
    ("adapter_id", "class_type", "lora_filename"),
    (
        (
            "minimax_h3_turbo",
            "MiniMaxH3TurboLoRA",
            "nested/minimax_h3_turbo_v4.safetensors",
        ),
        ("model_only", "LoraLoaderModelOnly", "loras/style.safetensors"),
    ),
)
def test_lora_path_mapping_selects_each_supported_standard_adapter(
    adapter_id: str,
    class_type: str,
    lora_filename: str,
) -> None:
    binding = _binding(lora_filename)
    override = _Override(
        lora_filename=binding.lora_filename,
        adapter_id=adapter_id,
        options={"low_vram": True} if adapter_id == "minimax_h3_turbo" else {},
    )

    resolved = resolve_standard_lora_adapter(binding, (override,))

    assert resolved.source == "user_override"
    assert resolved.binding == binding
    assert resolved.adapter.adapter_id == adapter_id
    assert resolved.adapter.class_type == class_type
    assert dict(resolved.options) == (
        {"low_vram": True} if adapter_id == "minimax_h3_turbo" else {}
    )


def test_standard_binding_without_user_mapping_uses_model_only_default() -> None:
    binding = _binding()
    resolved = resolve_standard_lora_adapter(binding, ())

    assert resolved.source == "factory_default"
    assert resolved.binding == binding
    assert resolved.adapter.adapter_id == "model_only"
    assert resolved.adapter.class_type == "LoraLoaderModelOnly"
    assert dict(resolved.options) == {}


def test_regex_policy_selects_its_default_loader_without_a_user_mapping() -> None:
    binding = LoraLoaderBindingKey(
        family="fl2va",
        model_filename="models/minimax-h3.safetensors",
        lora_filename="nested/minimax_h3_turbo_v4_step600_ema.safetensors",
    )

    resolved = resolve_standard_lora_adapter(binding, ())

    assert resolved.source == "factory_default"
    assert resolved.adapter.adapter_id == "minimax_h3_turbo"
    assert resolved.adapter.class_type == "MiniMaxH3TurboLoRA"
    assert dict(resolved.options) == {"low_vram": False}


def test_exact_user_override_is_used_within_the_effective_policy() -> None:
    binding = _binding()
    override = _Override(
        lora_filename=binding.lora_filename,
        adapter_id="model_only",
        options={},
    )

    resolved = resolve_standard_lora_adapter(binding, (override,))

    assert resolved.source == "user_override"
    assert resolved.adapter.adapter_id == "model_only"


def test_removing_user_mapping_restores_the_policy_default_options() -> None:
    binding = _binding("nested/minimax_h3_turbo_v4.safetensors")
    override = _Override(
        lora_filename=binding.lora_filename,
        adapter_id="minimax_h3_turbo",
        options={"low_vram": True},
    )

    mapped = resolve_standard_lora_adapter(
        binding,
        (override,),
    )
    restored = resolve_standard_lora_adapter(
        binding,
        (),
    )
    assert mapped.adapter.adapter_id == "minimax_h3_turbo"
    assert dict(mapped.options) == {"low_vram": True}
    assert restored.adapter.adapter_id == "minimax_h3_turbo"
    assert dict(restored.options) == {"low_vram": False}


def test_same_lora_path_mapping_is_shared_across_family_and_model_context() -> None:
    fl_binding = _binding("nested/minimax_h3_turbo_v4.safetensors")
    overrides = (
        _Override(
            lora_filename=fl_binding.lora_filename,
            adapter_id="minimax_h3_turbo",
            options={"low_vram": True},
        ),
    )

    resolved = resolve_standard_lora_adapter(fl_binding, overrides)
    assert resolved.adapter.adapter_id == "minimax_h3_turbo"
    assert dict(resolved.options) == {"low_vram": True}


def test_dedicated_loader_is_rejected_outside_its_filename_policy() -> None:
    binding = _binding()
    override = _Override(
        lora_filename=binding.lora_filename,
        adapter_id="minimax_h3_turbo",
        options={"low_vram": False},
    )

    with pytest.raises(LoraAdapterResolutionError) as caught:
        resolve_standard_lora_adapter(binding, (override,))

    assert caught.value.code == "lora_loader_not_allowed_for_file"
    assert caught.value.adapter_id == "minimax_h3_turbo"


def test_unknown_user_adapter_fails_closed_for_the_exact_binding() -> None:
    binding = LoraLoaderBindingKey(
        family="fl2va",
        model_filename="models/H3.safetensors",
        lora_filename="styles/detail.safetensors",
    )
    override = _Override(
        lora_filename=binding.lora_filename,
        adapter_id="future_uninstalled_adapter",
        options={},
    )

    with pytest.raises(LoraAdapterResolutionError) as caught:
        resolve_standard_lora_adapter(binding, (override,))

    assert caught.value.code == "lora_adapter_unknown"
    assert caught.value.adapter_id == "future_uninstalled_adapter"


def test_duplicate_exact_user_records_fail_closed_even_for_corrupt_callers() -> None:
    binding = _binding()
    override = _Override(
        lora_filename=binding.lora_filename,
        adapter_id="minimax_h3_turbo",
        options={"low_vram": False},
    )

    with pytest.raises(LoraAdapterResolutionError) as caught:
        resolve_standard_lora_adapter(binding, (override, override))

    assert caught.value.code == "lora_loader_mapping_conflict"


@pytest.mark.parametrize("family", ("fl2va", "ref2va"))
def test_raylight_uses_fixed_adapter_without_a_standard_binding(family: str) -> None:
    resolved = resolve_raylight_lora_adapter(family)  # type: ignore[arg-type]

    assert resolved.source == "backend_fixed"
    assert resolved.binding is None
    assert resolved.adapter.adapter_id == "ray_lora"
    assert resolved.adapter.class_type == "DirectorDeckRayLoraLoader"
