from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from directordeck.native_templates import NativeTemplateError


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
_validate_compiled_attention_carriers = (
    _VALIDATOR._validate_compiled_attention_carriers
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
        NODE_CLASS_MAPPINGS={"DirectorDeckRayInitializerAdvanced": Initializer}
    )


def test_raylight_registry_contract_keeps_ck_off_baseline_and_offers_ck() -> None:
    assert _validate_raylight_initializer_contract(_nodes()) == []


def test_raylight_registry_contract_rejects_missing_ck_or_changed_off_baseline() -> None:
    missing_ck = _validate_raylight_initializer_contract(
        _nodes(attention_options=["TORCH_FLASH"])
    )
    changed_default = _validate_raylight_initializer_contract(
        _nodes(attention_default="COMFY_KITCHEN_INT8")
    )

    assert any("must offer" in failure for failure in missing_ck)
    assert any("retain TORCH_FLASH" in failure for failure in changed_default)


def _compiled_result(
    backend: str,
    *,
    ck_enabled: bool,
    ray_attention: str | None = None,
    standard_attention: str | None = None,
):
    prompt = {}
    if ray_attention is not None:
        prompt["1"] = {
            "class_type": "DirectorDeckRayInitializerAdvanced",
            "inputs": {"XFuser_attention": ray_attention},
        }
    if standard_attention is not None:
        prompt["2"] = {
            "class_type": "ModelAttentionBackend",
            "inputs": {"attention": standard_attention},
        }
    return SimpleNamespace(
        workflows=[
            SimpleNamespace(
                id=f"unit-{backend}-{'on' if ck_enabled else 'off'}",
                backend=backend,
                compile_feature_uses=[
                    SimpleNamespace(
                        feature_id="comfy_kitchen_attention",
                        state="applicable" if ck_enabled else "inactive",
                    )
                ],
                prompt=prompt,
            )
        ]
    )


def test_compiled_bundle6_attention_contract_accepts_exact_off_and_on_carriers() -> None:
    assert _validate_compiled_attention_carriers(
        [
            _compiled_result("standard", ck_enabled=False),
            _compiled_result(
                "standard",
                ck_enabled=True,
                standard_attention="comfy kitchen attention",
            ),
            _compiled_result(
                "raylight",
                ck_enabled=False,
                ray_attention="TORCH_FLASH",
            ),
            _compiled_result(
                "raylight",
                ck_enabled=True,
                ray_attention="COMFY_KITCHEN_INT8",
            ),
        ]
    ) == []


def test_compiled_bundle6_attention_contract_rejects_crossed_carriers() -> None:
    failures = _validate_compiled_attention_carriers(
        [
            _compiled_result(
                "standard",
                ck_enabled=False,
                standard_attention="comfy kitchen attention",
            ),
            _compiled_result("standard", ck_enabled=True),
            _compiled_result(
                "raylight",
                ck_enabled=False,
                ray_attention="COMFY_KITCHEN_INT8",
            ),
            _compiled_result(
                "raylight",
                ck_enabled=True,
                ray_attention="TORCH_FLASH",
            ),
        ]
    )

    assert any("Standard CK off requires 0" in failure for failure in failures)
    assert any("Standard CK on requires 1" in failure for failure in failures)
    assert any(
        "RayLight CK off must explicitly select TORCH_FLASH" in failure
        for failure in failures
    )
    assert any(
        "RayLight CK on must explicitly select COMFY_KITCHEN_INT8" in failure
        for failure in failures
    )


def test_bundle6_validator_fixtures_compile_all_four_attention_carriers() -> None:
    standard = _VALIDATOR._standard_settings()
    raylight = _VALIDATOR._raylight_settings()
    results = [
        _VALIDATOR._compile_bundle6_timeline(
            _VALIDATOR._draft("t2v"),
            settings,
            f"fixture-{settings.multi_gpu_enabled}-{ck_enabled}",
            ck_enabled=ck_enabled,
        )
        for settings, ck_enabled in (
            (standard, False),
            (standard, True),
            (raylight, False),
            (raylight, True),
        )
    ]

    assert _validate_compiled_attention_carriers(results) == []


def test_bundle6_validator_continuity_uses_bundle6_graph_audit() -> None:
    output = {"filename": "predecessor.mp4", "subfolder": "", "type": "output"}
    for settings in (_VALIDATOR._standard_settings(), _VALIDATOR._raylight_settings()):
        result = _VALIDATOR._compile_bundle6_timeline(
            _VALIDATOR._continuity_draft("fl2v"),
            settings,
            f"continuity-{settings.multi_gpu_enabled}",
        )
        successor = result.workflows[1]
        bound = _VALIDATOR.bind_native_workflow_predecessor_output(
            successor,
            output,
            node_contract_registry=_VALIDATOR.V6_NODE_CONTRACT_REGISTRY,
        )
        if settings.multi_gpu_enabled:
            bound = _VALIDATOR.bind_raylight_runtime_epoch(
                bound,
                1,
                node_contract_registry=_VALIDATOR.V6_NODE_CONTRACT_REGISTRY,
            )
            assert bound.raylight_runtime_epoch == 1
        assert bound.continuity is not None
        assert bound.continuity.resolved is True
        assert bound.continuity.bound_file == "predecessor.mp4 [output]"


def test_legacy_continuity_still_rejects_missing_feature_audit_traces() -> None:
    result = _VALIDATOR.compile_legacy_native_timeline(
        _VALIDATOR._continuity_draft("fl2v"),
        _VALIDATOR._standard_settings(),
        "legacy-missing-traces",
    )
    malformed = replace(result.workflows[1], graph_audit_traces=())

    with pytest.raises(
        NativeTemplateError,
        match="feature audit traces do not cover the prompt node set",
    ):
        _VALIDATOR.bind_native_workflow_predecessor_output(
            malformed,
            {"filename": "predecessor.mp4", "subfolder": "", "type": "output"},
        )


def test_validator_raylight_fixture_explicitly_enables_multi_gpu() -> None:
    settings = _VALIDATOR._raylight_settings()

    assert settings.multi_gpu_enabled is True
    assert settings.models.fl2va.raylight.gpu_select == [0, 1]
    assert settings.models.ref2va.raylight.gpu_select == [0, 1]


def test_validator_standard_lora_fixtures_use_visible_exact_overrides() -> None:
    for loader in ("model_only", "bypass_model_only"):
        settings = _VALIDATOR._standard_lora_settings(loader)
        binding = settings.models.fl2va
        override = binding.standard_lora_loader_override

        assert override is not None
        assert override.loader == loader
        assert override.lora_name == binding.lora_name
        assert override.model_filename == binding.filename
