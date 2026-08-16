from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


_VALIDATOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "validate_native_comfy_prompts.py"
)
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "director_native_prompt_validator", _VALIDATOR_PATH
)
assert _VALIDATOR_SPEC is not None and _VALIDATOR_SPEC.loader is not None
_VALIDATOR = importlib.util.module_from_spec(_VALIDATOR_SPEC)
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR)
_validate_compiled_raylight_attention = (
    _VALIDATOR._validate_compiled_raylight_attention
)
_validate_raylight_initializer_contract = (
    _VALIDATOR._validate_raylight_initializer_contract
)


def _nodes(
    *,
    attention_options: list[str] | None = None,
    attention_default: str = "TORCH_FLASH",
):
    options = attention_options or ["TORCH_FLASH", "COMFY_KITCHEN_INT8"]

    class Initializer:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "XFuser_attention": (
                        options,
                        {"default": attention_default},
                    )
                },
                "optional": {
                    "driver_cleanup_policy": (
                        ["legacy_all", "ray_devices"],
                        {"default": "legacy_all"},
                    ),
                    "ram_cache_max_models": (
                        "INT",
                        {"default": 2, "min": 0},
                    ),
                },
            }

    return SimpleNamespace(
        NODE_CLASS_MAPPINGS={"RayInitializerAdvanced": Initializer}
    )


def test_raylight_registry_contract_keeps_legacy_default_and_offers_director_ck() -> None:
    assert _validate_raylight_initializer_contract(_nodes()) == []


def test_raylight_registry_contract_rejects_missing_ck_or_changed_legacy_default() -> None:
    missing_ck = _validate_raylight_initializer_contract(
        _nodes(attention_options=["TORCH_FLASH"])
    )
    changed_default = _validate_raylight_initializer_contract(
        _nodes(attention_default="COMFY_KITCHEN_INT8")
    )

    assert any("must offer" in failure for failure in missing_ck)
    assert any("retain TORCH_FLASH" in failure for failure in changed_default)


def test_compiled_raylight_contract_requires_explicit_ck() -> None:
    def result(value: str):
        return SimpleNamespace(
            workflows=[
                SimpleNamespace(
                    id=f"unit-{value}",
                    backend="raylight",
                    prompt={
                        "1": {
                            "class_type": "RayInitializerAdvanced",
                            "inputs": {"XFuser_attention": value},
                        }
                    },
                )
            ]
        )

    assert _validate_compiled_raylight_attention(
        [result("COMFY_KITCHEN_INT8")]
    ) == []
    failures = _validate_compiled_raylight_attention([result("TORCH_FLASH")])
    assert any("must be explicitly COMFY_KITCHEN_INT8" in failure for failure in failures)
