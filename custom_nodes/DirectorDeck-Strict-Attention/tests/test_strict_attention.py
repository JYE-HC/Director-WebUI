from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[4]
COMFY_ROOT = PROJECT_ROOT / "ComfyUI"
sys.path.insert(0, str(COMFY_ROOT))

# Import the checked-out ComfyUI on CPU.  The production process has already
# parsed its own arguments before loading custom nodes; this setup is specific
# to exercising the real ModelPatcher and attention APIs without a GPU.
import comfy.options

_ORIGINAL_ARGV = sys.argv[:]
try:
    comfy.options.enable_args_parsing()
    sys.argv[:] = [sys.argv[0], "--cpu"]
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    NODE_PATH = Path(__file__).parents[1] / "__init__.py"
    SPEC = importlib.util.spec_from_file_location(
        "directordeck_strict_attention_test_node",
        NODE_PATH,
    )
    assert SPEC is not None and SPEC.loader is not None
    NODE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(NODE)
finally:
    sys.argv[:] = _ORIGINAL_ARGV


class _NamedLikeH3:
    pass


_NamedLikeH3.__name__ = "MiniMaxH3Model"


def _bare_h3() -> object:
    # Exact checked-out class identity without allocating the multi-billion
    # parameter model.  The strict node only needs to establish type identity.
    return object.__new__(NODE.MiniMaxH3Model)


def _patcher(
    *,
    diffusion_model: object | None = None,
    device: str = "cpu",
) -> NODE.ModelPatcher:
    torch_device = torch.device(device)
    root = SimpleNamespace(
        diffusion_model=_bare_h3() if diffusion_model is None else diffusion_model,
        device=torch_device,
    )
    return NODE.ModelPatcher(
        root,
        torch_device,
        torch.device("cpu"),
        size=1,
    )


def _clone_counter(model: NODE.ModelPatcher) -> list[NODE.ModelPatcher]:
    clones: list[NODE.ModelPatcher] = []
    model.add_callback(
        "on_clone",
        lambda _source, cloned: clones.append(cloned),
    )
    return clones


def _apply(model: object, mode: object = "pytorch") -> NODE.ModelPatcher:
    return NODE.DirectorStrictModelAttentionBackend().patch(model, mode)[0]


def test_node_mapping_and_wire_contract_are_exact() -> None:
    assert NODE.NODE_CLASS_MAPPINGS == {
        "DirectorStrictModelAttentionBackend": NODE.DirectorStrictModelAttentionBackend,
    }
    assert "ModelAttentionBackend" not in NODE.NODE_CLASS_MAPPINGS
    assert NODE.DirectorStrictModelAttentionBackend.INPUT_TYPES() == {
        "required": {
            "model": ("MODEL",),
            "mode": (["pytorch", "ck_int8"],),
        }
    }
    assert NODE.DirectorStrictModelAttentionBackend.RETURN_TYPES == ("MODEL",)
    assert NODE.DirectorStrictModelAttentionBackend.FUNCTION == "patch"


@pytest.mark.parametrize(
    "mode",
    [None, 1, True, "", "inherit_host", "pytorch attention", "CK_INT8"],
)
def test_invalid_mode_fails_before_clone(mode: object) -> None:
    model = _patcher()
    clones = _clone_counter(model)

    with pytest.raises(ValueError, match="must be exactly"):
        _apply(model, mode)

    assert clones == []
    assert "optimized_attention_override" not in model.model_options[
        "transformer_options"
    ]


def test_non_model_patcher_and_name_only_h3_fail_before_clone() -> None:
    with pytest.raises(TypeError, match="requires a ComfyUI ModelPatcher"):
        _apply(SimpleNamespace())

    model = _patcher(diffusion_model=_NamedLikeH3())
    clones = _clone_counter(model)
    with pytest.raises(TypeError, match="exact ComfyUI MiniMaxH3Model"):
        _apply(model)
    assert clones == []


def test_existing_override_conflict_fails_before_clone() -> None:
    model = _patcher()
    existing = lambda *_args, **_kwargs: None
    model.model_options["transformer_options"][
        "optimized_attention_override"
    ] = existing
    clones = _clone_counter(model)

    with pytest.raises(ValueError, match="already has"):
        _apply(model)

    assert clones == []
    assert (
        model.model_options["transformer_options"][
            "optimized_attention_override"
        ]
        is existing
    )


def test_existing_strict_h3_sage_patch_conflict_fails_before_clone() -> None:
    model = _patcher()
    patch_key = "diffusion_model.blocks.0.attn.forward"
    existing = lambda *_args, **_kwargs: None
    model.object_patches[patch_key] = existing
    clones = _clone_counter(model)

    with pytest.raises(RuntimeError, match="existing strict H3 Sage"):
        _apply(model)

    assert clones == []
    assert model.object_patches[patch_key] is existing


@pytest.mark.parametrize(
    "model_options",
    [None, {}, {"transformer_options": None}, {"transformer_options": []}],
)
def test_malformed_model_options_fail_before_clone(model_options: object) -> None:
    model = _patcher()
    valid_options = model.model_options
    model.model_options = model_options

    try:
        with pytest.raises(TypeError, match="dict"):
            _apply(model)
    finally:
        # ModelPatcher.__del__ expects its own valid structure.  Restore it so
        # this negative input test does not manufacture an unrelated warning.
        model.model_options = valid_options


@pytest.mark.parametrize("method_name", ["clone", "set_model_optimized_attention"])
def test_model_patcher_api_replacement_fails_before_clone(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    model = _patcher()
    clones = _clone_counter(model)
    monkeypatch.setattr(
        NODE.ModelPatcher,
        method_name,
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(TypeError, match=f"ModelPatcher.{method_name}"):
        _apply(model)

    assert clones == []


def test_unregistered_or_replaced_pytorch_backend_fails_before_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _patcher()
    clones = _clone_counter(model)
    monkeypatch.setitem(
        NODE.comfy_attention.REGISTERED_ATTENTION_FUNCTIONS,
        "pytorch",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="exact 'pytorch' backend"):
        _apply(model)

    assert clones == []


def test_pytorch_uses_real_model_patcher_api_and_executes_on_cpu() -> None:
    model = _patcher()
    existing_marker = object()
    model.model_options["transformer_options"]["unrelated"] = existing_marker

    result = _apply(model, "pytorch")

    assert result is not model
    assert result.model is model.model
    assert result.parent is model
    assert "optimized_attention_override" not in model.model_options[
        "transformer_options"
    ]
    assert (
        result.model_options["transformer_options"]["unrelated"]
        is existing_marker
    )
    override = result.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    closure = NODE.inspect.getclosurevars(override)
    assert closure.nonlocals == {
        "optimized_attention": NODE.comfy_attention.attention_pytorch
    }

    # The installed override ignores the fallback callable supplied by ComfyUI
    # and executes the exact PyTorch backend with real CPU tensors.
    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("fallback must not execute")

    generator = torch.Generator(device="cpu").manual_seed(7)
    q = torch.randn((1, 3, 4), generator=generator)
    k = torch.randn((1, 3, 4), generator=generator)
    v = torch.randn((1, 3, 4), generator=generator)
    actual = override(forbidden_fallback, q, k, v, 1)
    expected = NODE.comfy_attention.attention_pytorch(q, k, v, 1)
    torch.testing.assert_close(actual, expected)


def test_clone_side_conflict_is_detected_before_installation() -> None:
    model = _patcher()
    injected = lambda *_args, **_kwargs: None

    def add_conflict(_source, cloned):
        cloned.model_options["transformer_options"][
            "optimized_attention_override"
        ] = injected

    model.add_callback("on_clone", add_conflict)

    with pytest.raises(ValueError, match="already has"):
        _apply(model)

    assert "optimized_attention_override" not in model.model_options[
        "transformer_options"
    ]


@pytest.mark.skipif(
    torch.cuda.is_available()
    or NODE.comfy_attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE,
    reason="this truthful host-capability assertion is specific to a CPU runtime",
)
def test_current_cpu_runtime_rejects_ck_int8_before_clone() -> None:
    model = _patcher(device="cpu")
    clones = _clone_counter(model)

    with pytest.raises(RuntimeError, match="requires a CUDA model device"):
        _apply(model, "ck_int8")

    assert clones == []


def test_current_cpu_runtime_capability_is_exact_and_fail_closed() -> None:
    assert NODE.director_runtime_capability("pytorch") == {
        "available": True,
        "code": "available",
        "architecture": None,
    }
    ck = NODE.director_runtime_capability("ck_int8")
    assert ck["available"] is False
    assert ck["code"] != "available"
    assert NODE.director_runtime_capability("ck_int8", -1) == {
        "available": False,
        "code": "model_device_not_cuda",
        "architecture": None,
    }


def test_runtime_capability_proves_exact_gpu_and_kitchen_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[torch.device] = []
    monkeypatch.setattr(
        NODE.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        True,
    )

    def probe(device: torch.device) -> bool:
        calls.append(device)
        return device == torch.device("cuda:1")

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
            get_device_capability=lambda index: (9, index),
        ),
    )

    unavailable = NODE.director_runtime_capability("ck_int8", 0)
    available = NODE.director_runtime_capability("ck_int8", 1)

    assert unavailable == {
        "available": False,
        "code": "comfy_kitchen_int8_device_unavailable",
        "architecture": "sm90",
    }
    assert available == {
        "available": True,
        "code": "available",
        "architecture": "sm91",
    }
    assert calls == [torch.device("cuda:0"), torch.device("cuda:1")]


def test_runtime_capability_rejects_api_drift_and_sanitizes_probe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(NODE.ModelPatcher, "clone", lambda *_args: None)
    assert NODE.director_runtime_capability("pytorch")["code"] == (
        "comfy_model_patcher_api_incompatible"
    )

    monkeypatch.undo()
    monkeypatch.setattr(
        NODE.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        NODE.torch,
        "cuda",
        SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            current_device=lambda: 0,
            get_device_capability=lambda _index: (8, 0),
        ),
    )

    def fail(_device: torch.device) -> bool:
        raise RuntimeError("/opt/sentinel-driver/detail")

    monkeypatch.setattr(
        NODE.comfy_attention.comfy_kitchen,
        "int8_attention_is_available",
        fail,
    )
    evidence = NODE.director_runtime_capability("ck_int8", 0)
    assert evidence["code"] == "comfy_kitchen_int8_probe_failed"
    assert "sentinel-driver" not in str(evidence)


def test_ck_int8_global_or_device_unavailability_fails_before_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _patcher(device="cuda:0")
    clones = _clone_counter(model)

    monkeypatch.setattr(
        NODE.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        False,
    )
    with pytest.raises(RuntimeError, match="unavailable in this ComfyUI runtime"):
        _apply(model, "ck_int8")
    assert clones == []

    monkeypatch.setattr(
        NODE.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        NODE.comfy_attention.comfy_kitchen,
        "int8_attention_is_available",
        lambda _device=None: False,
    )
    with pytest.raises(RuntimeError, match="unavailable on the MODEL device"):
        _apply(model, "ck_int8")
    assert clones == []


def test_ck_int8_probe_error_fails_closed_before_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _patcher(device="cuda:0")
    clones = _clone_counter(model)
    monkeypatch.setattr(
        NODE.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        True,
    )

    def fail_probe(_device=None):
        raise RuntimeError("driver detail that must not become fallback")

    monkeypatch.setattr(
        NODE.comfy_attention.comfy_kitchen,
        "int8_attention_is_available",
        fail_probe,
    )

    with pytest.raises(RuntimeError, match="could not verify"):
        _apply(model, "ck_int8")
    assert clones == []


def test_proved_ck_int8_binds_exact_function_and_container_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _patcher(device="cuda:0")
    monkeypatch.setattr(
        NODE.comfy_attention,
        "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        NODE.comfy_attention.comfy_kitchen,
        "int8_attention_is_available",
        lambda device=None: device == torch.device("cuda:0"),
    )
    monkeypatch.setitem(
        NODE.comfy_attention.REGISTERED_ATTENTION_FUNCTIONS,
        "comfy_kitchen_int8",
        NODE.comfy_attention.attention_comfy_kitchen_int8,
    )

    result = _apply(model, "ck_int8")

    override = result.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    assert NODE.inspect.getclosurevars(override).nonlocals == {
        "optimized_attention": (
            NODE.comfy_attention.attention_comfy_kitchen_int8
        )
    }
    assert (
        override.container_function
        is NODE.comfy_attention.attention_comfy_kitchen_int8.container_function
    )
    assert "optimized_attention_override" not in model.model_options[
        "transformer_options"
    ]
