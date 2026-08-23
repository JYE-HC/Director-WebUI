from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


NODE_PATH = Path(__file__).parents[1] / "__init__.py"
SPEC = importlib.util.spec_from_file_location(
    "directordeck_strict_attention_cpu_contract_node",
    NODE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
NODE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NODE)


def _model() -> NODE.ModelPatcher:
    return NODE.ModelPatcher(
        SimpleNamespace(diffusion_model=NODE.MiniMaxH3Model()),
        torch.device("cpu"),
    )


def test_cpu_gate_wire_contract_is_exact() -> None:
    assert NODE.NODE_CLASS_MAPPINGS == {
        "DirectorStrictModelAttentionBackend": (
            NODE.DirectorStrictModelAttentionBackend
        ),
    }
    assert NODE.DirectorStrictModelAttentionBackend.INPUT_TYPES() == {
        "required": {
            "model": ("MODEL",),
            "mode": (["pytorch", "ck_int8"],),
        }
    }
    assert NODE.DirectorStrictModelAttentionBackend.RETURN_TYPES == ("MODEL",)


def test_cpu_gate_runtime_probe_is_exact_and_never_claims_ck() -> None:
    assert NODE.director_runtime_capability("pytorch") == {
        "available": True,
        "code": "available",
        "architecture": None,
    }
    ck = NODE.director_runtime_capability("ck_int8")
    assert ck["available"] is False
    assert ck["code"] != "available"
    assert NODE.director_runtime_capability("ck_int8", -1)["code"] == (
        "model_device_not_cuda"
    )


def test_runtime_probe_requires_the_exact_registered_ck_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        NODE.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        True,
    )

    def probe(device: object) -> bool:
        calls.append(device)
        return device == torch.device("cuda", 1)

    monkeypatch.setattr(
        NODE.comfy_attention.comfy_kitchen,
        "int8_attention_is_available",
        probe,
    )
    monkeypatch.setattr(
        NODE.torch,
        "cuda",
        SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 2,
            current_device=lambda: 0,
            get_device_capability=lambda _index: (8, 0),
        ),
    )

    assert NODE.director_runtime_capability("ck_int8", 0)["available"] is False
    assert NODE.director_runtime_capability("ck_int8", 1) == {
        "available": True,
        "code": "available",
        "architecture": "sm80",
    }
    assert calls == [torch.device("cuda", 0), torch.device("cuda", 1)]


def test_cpu_gate_pytorch_binds_exact_backend_without_mutating_input() -> None:
    model = _model()

    result = NODE.DirectorStrictModelAttentionBackend().patch(
        model,
        "pytorch",
    )[0]

    assert result is not model
    assert result.model is model.model
    assert "optimized_attention_override" not in model.model_options[
        "transformer_options"
    ]
    installed = result.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    assert NODE.inspect.getclosurevars(installed).nonlocals == {
        "optimized_attention": NODE.comfy_attention.attention_pytorch,
    }


def test_cpu_gate_ck_int8_fails_before_clone() -> None:
    model = _model()

    with pytest.raises(RuntimeError, match="requires a CUDA model device"):
        NODE.DirectorStrictModelAttentionBackend().patch(model, "ck_int8")

    assert model.clone_count == 0


def test_cpu_gate_invalid_or_conflicting_override_fails_before_clone() -> None:
    model = _model()
    with pytest.raises(ValueError, match="must be exactly"):
        NODE.DirectorStrictModelAttentionBackend().patch(model, "auto")
    assert model.clone_count == 0

    model.model_options["transformer_options"][
        "optimized_attention_override"
    ] = lambda *_args, **_kwargs: None
    with pytest.raises(ValueError, match="already has"):
        NODE.DirectorStrictModelAttentionBackend().patch(model, "pytorch")
    assert model.clone_count == 0
