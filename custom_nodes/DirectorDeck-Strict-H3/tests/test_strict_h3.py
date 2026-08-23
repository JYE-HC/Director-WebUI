from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


NODE_PATH = Path(__file__).parents[1] / "__init__.py"
SPEC = importlib.util.spec_from_file_location(
    "directordeck_strict_h3_test_node",
    NODE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
NODE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NODE
SPEC.loader.exec_module(NODE)


class _Projection:
    def __init__(self, *, copies: int = 1):
        self.copies = copies

    def __call__(self, value):
        if self.copies == 1:
            return value
        return torch.cat([value] * self.copies, dim=-1)


class _Norm:
    def __init__(self):
        self.weight = torch.ones(4, dtype=torch.float16)
        self.eps = 1e-5
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return value


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = _Projection(copies=3)
        self.q_norm = _Norm()
        self.k_norm = _Norm()
        self.out_proj = _Projection()

    def forward(self, x, rope_freqs=None, transformer_options=None):
        return x


class DiTBlock(torch.nn.Module):
    def __init__(self, attention=None):
        super().__init__()
        self.attn = attention or Attention()


class MiniMaxH3Model(torch.nn.Module):
    def __init__(self, count=3, blocks=None):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [DiTBlock() for _ in range(count)] if blocks is None else blocks
        )


class OtherModel(torch.nn.Module):
    pass


class _Root:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model


def _resolve(root, path):
    current = root
    for component in path.split("."):
        if component.isdigit():
            current = current[int(component)]
        else:
            current = getattr(current, component)
    return current


class _Patcher:
    def __init__(
        self,
        diffusion_model=None,
        *,
        object_patches=None,
        object_patches_backup=None,
        clone_behavior="normal",
        add_behavior="normal",
        load_device=None,
    ):
        self.model = _Root(diffusion_model or MiniMaxH3Model())
        self.object_patches = dict(object_patches or {})
        self.object_patches_backup = dict(object_patches_backup or {})
        self.clone_behavior = clone_behavior
        self.add_behavior = add_behavior
        self.load_device = load_device
        self.model_options = {"transformer_options": {}}
        self.clone_count = 0
        self.last_clone = None
        self.add_calls = []

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        return _resolve(self.model, name)

    def clone(self):
        self.clone_count += 1
        if self.clone_behavior == "same":
            return self
        if self.clone_behavior == "duck_clone":
            return SimpleNamespace(
                clone=lambda: None,
                get_model_object=lambda _name: None,
                add_object_patch=lambda _name, _value: None,
            )
        diffusion_model = self.model.diffusion_model
        if self.clone_behavior == "different_model":
            diffusion_model = MiniMaxH3Model(len(diffusion_model.blocks))
        clone = _Patcher(
            diffusion_model,
            object_patches=self.object_patches,
            object_patches_backup=self.object_patches_backup,
            clone_behavior=self.clone_behavior,
            add_behavior=self.add_behavior,
            load_device=self.load_device,
        )
        clone.model_options = {
            **self.model_options,
            "transformer_options": dict(
                self.model_options.get("transformer_options", {})
            ),
        }
        if self.clone_behavior == "shared_mapping":
            clone.object_patches = self.object_patches
            clone.object_patches_backup = self.object_patches_backup
        self.last_clone = clone
        return clone

    def add_object_patch(self, name, value):
        self.add_calls.append((name, value))
        if self.add_behavior == "raise_second" and len(self.add_calls) == 2:
            raise RuntimeError("synthetic add failure")
        if self.add_behavior == "drop_second" and len(self.add_calls) == 2:
            return
        if self.add_behavior == "wrong_second" and len(self.add_calls) == 2:
            self.object_patches[name] = object()
            return
        self.object_patches[name] = value
        if self.add_behavior == "extra_first" and len(self.add_calls) == 1:
            self.object_patches["unexpected.patch"] = object()


class _Cuda:
    def __init__(
        self,
        *,
        available=True,
        count=1,
        current=0,
        capability=(8, 0),
        failure=None,
    ):
        self.available = available
        self.count = count
        self.current = current
        self.capability = capability
        self.failure = failure

    def _maybe_raise(self):
        if self.failure is not None:
            raise self.failure

    def is_available(self):
        self._maybe_raise()
        return self.available

    def device_count(self):
        self._maybe_raise()
        return self.count

    def current_device(self):
        self._maybe_raise()
        return self.current

    def get_device_capability(self, device):
        self._maybe_raise()
        if isinstance(self.capability, dict):
            return self.capability[device]
        return self.capability


class _CK:
    def __init__(self):
        self.inplace_calls = []
        self.training_calls = []

    def rms_rope_split_half_(self, q, k, rope, qw, kw, **kwargs):
        self.inplace_calls.append((q, k, rope, qw, kw, kwargs))

    def rms_rope_split_half(self, q, k, rope, qw, kw, **kwargs):
        self.training_calls.append((q, k, rope, qw, kw, kwargs))
        return q, k


def _sage(q, k, v, tensor_layout="HND", is_causal=True):
    del q, k, tensor_layout, is_causal
    return v


def _configure_runtime(
    monkeypatch,
    *,
    sage=_sage,
    cuda=None,
    compiled_architectures=("sm80",),
    in_training=False,
):
    ck = _CK()
    cast_calls = []

    def cast_to(value, *, device):
        cast_calls.append((value, device))
        return value

    monkeypatch.setattr(NODE, "_MINIMAX_H3_MODEL_TYPE", MiniMaxH3Model)
    monkeypatch.setattr(NODE, "_MINIMAX_H3_BLOCK_TYPE", DiTBlock)
    monkeypatch.setattr(NODE, "_MINIMAX_H3_ATTENTION_TYPE", Attention)
    monkeypatch.setattr(NODE, "_ORIGINAL_H3_ATTENTION_FORWARD", Attention.forward)
    monkeypatch.setattr(NODE, "_MODEL_PATCHER_TYPE", _Patcher)
    monkeypatch.setattr(
        NODE,
        "_MODEL_PATCHER_MODULE",
        SimpleNamespace(ModelPatcher=_Patcher),
    )
    monkeypatch.setattr(NODE, "_OFFICIAL_MODEL_PATCHER_CLONE", _Patcher.clone)
    monkeypatch.setattr(
        NODE,
        "_OFFICIAL_MODEL_PATCHER_GET_OBJECT",
        _Patcher.get_model_object,
    )
    monkeypatch.setattr(
        NODE,
        "_OFFICIAL_MODEL_PATCHER_ADD_OBJECT_PATCH",
        _Patcher.add_object_patch,
    )
    monkeypatch.setattr(
        NODE,
        "_MODEL_MANAGEMENT",
        SimpleNamespace(cast_to=cast_to, in_training=in_training),
    )
    monkeypatch.setattr(NODE, "_CK", ck)
    monkeypatch.setattr(NODE, "_SAGE_MODULE", SimpleNamespace(sageattn=sage))
    monkeypatch.setattr(
        NODE,
        "_SAGE_CORE_MODULE",
        SimpleNamespace(
            sageattn=sage,
            SM80_ENABLED="sm80" in compiled_architectures,
            SM89_ENABLED=(
                "sm89" in compiled_architectures
                or "sm120" in compiled_architectures
                or "sm121" in compiled_architectures
            ),
            SM90_ENABLED="sm90" in compiled_architectures,
            sageattn_qk_int8_pv_fp16_cuda=lambda *args, **kwargs: None,
            sageattn_qk_int8_pv_fp16_triton=lambda *args, **kwargs: None,
            sageattn_qk_int8_pv_fp8_cuda=lambda *args, **kwargs: None,
            sageattn_qk_int8_pv_fp8_cuda_sm90=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setattr(NODE.torch, "cuda", cuda or _Cuda())
    return ck, cast_calls


def _apply(model):
    return NODE.DirectorStrictH3LowVramSagePatch().apply_strict_h3_sage(model)[0]


def _binding(sage=_sage, *, architecture="sm80", device_index=0):
    return NODE._RuntimeBinding(
        sageattn=sage,
        architecture=architecture,
        device_index=device_index,
    )


def test_node_mapping_and_schema_are_exact():
    assert NODE.NODE_CLASS_MAPPINGS == {
        "DirectorStrictH3LowVramSagePatch": (
            NODE.DirectorStrictH3LowVramSagePatch
        )
    }
    assert NODE.DirectorStrictH3LowVramSagePatch.INPUT_TYPES() == {
        "required": {"model": ("MODEL",)}
    }
    assert NODE.DirectorStrictH3LowVramSagePatch.RETURN_TYPES == ("MODEL",)


def test_current_process_capability_never_claims_an_unproved_runtime():
    capability = NODE.runtime_capability()
    if capability.available:
        assert capability.code == "available"
        assert capability.architecture in NODE._SUPPORTED_CUDA_ARCHITECTURES
    else:
        assert capability.code != "available"


def test_missing_sage_and_incompatible_api_are_unavailable(monkeypatch):
    _configure_runtime(monkeypatch)
    monkeypatch.setattr(NODE, "_SAGE_MODULE", None)
    assert NODE.director_runtime_capability() == {
        "available": False,
        "code": "sageattention_dependency_missing",
        "architecture": None,
    }

    def bad_signature(first, second, third):
        return first

    monkeypatch.setattr(NODE._SAGE_CORE_MODULE, "sageattn", bad_signature)
    monkeypatch.setattr(
        NODE,
        "_SAGE_MODULE",
        SimpleNamespace(sageattn=bad_signature),
    )
    assert NODE.runtime_capability().code == "sageattention_api_incompatible"

    def different_sage(q, k, v, tensor_layout="HND", is_causal=True):
        del q, k, tensor_layout, is_causal
        return v

    monkeypatch.setattr(
        NODE,
        "_SAGE_MODULE",
        SimpleNamespace(sageattn=different_sage),
    )
    assert NODE.runtime_capability().code == "sageattention_api_incompatible"


@pytest.mark.parametrize(
    ("cuda", "expected"),
    [
        (_Cuda(available=False), "cuda_runtime_unavailable"),
        (_Cuda(count=0), "cuda_runtime_unavailable"),
        (_Cuda(current=1), "cuda_device_unavailable"),
        (_Cuda(capability=(7, 5)), "cuda_architecture_unsupported"),
        (_Cuda(capability=(7, 0)), "cuda_architecture_unsupported"),
        (_Cuda(capability="sm80"), "cuda_architecture_unsupported"),
        (_Cuda(failure=RuntimeError("driver detail")), "cuda_runtime_unavailable"),
    ],
)
def test_cuda_gate_fails_closed_with_stable_codes(monkeypatch, cuda, expected):
    _configure_runtime(monkeypatch, cuda=cuda)
    capability = NODE.runtime_capability()
    assert capability.available is False
    assert capability.code == expected
    assert "driver detail" not in str(capability)


def test_supported_gpu_without_its_compiled_sage_kernel_is_unavailable(
    monkeypatch,
):
    _configure_runtime(monkeypatch, compiled_architectures=("sm90",))
    monkeypatch.setattr(
        NODE._SAGE_CORE_MODULE,
        "get_cuda_arch_versions",
        lambda: ["sm80"],
        raising=False,
    )
    capability = NODE.runtime_capability()
    assert capability == NODE.RuntimeCapability(
        False,
        "sageattention_kernel_unavailable",
        "sm80",
    )


@pytest.mark.parametrize(
    ("capability", "compiled_architecture"),
    [
        ((8, 0), "sm80"),
        ((8, 6), "sm86"),
        ((8, 9), "sm89"),
        ((9, 0), "sm90"),
    ],
)
def test_exact_compiled_flag_and_kernel_make_runtime_available(
    monkeypatch,
    capability,
    compiled_architecture,
):
    _configure_runtime(
        monkeypatch,
        cuda=_Cuda(capability=capability),
        compiled_architectures=(compiled_architecture,),
    )
    assert NODE.runtime_capability() == NODE.RuntimeCapability(
        True,
        "available",
        compiled_architecture,
    )


@pytest.mark.parametrize("compiled_flag", [False, None, 1, "true"])
def test_modern_compiled_flag_is_exact_and_fail_closed(
    monkeypatch,
    compiled_flag,
):
    _configure_runtime(monkeypatch)
    monkeypatch.setattr(NODE._SAGE_CORE_MODULE, "SM80_ENABLED", compiled_flag)
    assert NODE.runtime_capability() == NODE.RuntimeCapability(
        False,
        "sageattention_kernel_unavailable",
        "sm80",
    )


def test_compiled_flag_alone_does_not_replace_exact_kernel_export(monkeypatch):
    _configure_runtime(monkeypatch)
    monkeypatch.setattr(
        NODE._SAGE_CORE_MODULE,
        "sageattn_qk_int8_pv_fp16_cuda",
        None,
    )
    assert NODE.runtime_capability().code == "sageattention_kernel_unavailable"


def test_sm86_triton_route_requires_its_exact_callable(monkeypatch):
    _configure_runtime(monkeypatch, cuda=_Cuda(capability=(8, 6)))
    monkeypatch.setattr(
        NODE._SAGE_CORE_MODULE,
        "sageattn_qk_int8_pv_fp16_triton",
        None,
    )
    assert NODE.runtime_capability() == NODE.RuntimeCapability(
        False,
        "sageattention_kernel_unavailable",
        "sm86",
    )


@pytest.mark.parametrize(
    ("capability", "high_level_kernel"),
    [
        ((8, 0), "sageattn_qk_int8_pv_fp16_cuda"),
        ((8, 9), "sageattn_qk_int8_pv_fp8_cuda"),
        ((9, 0), "sageattn_qk_int8_pv_fp8_cuda_sm90"),
    ],
)
def test_flagless_legacy_extension_remains_unavailable(
    monkeypatch,
    capability,
    high_level_kernel,
):
    _configure_runtime(monkeypatch, cuda=_Cuda(capability=capability))
    architecture = f"sm{capability[0]}{capability[1]}"
    core = SimpleNamespace(
        sageattn=_sage,
        _qattn=SimpleNamespace(any_export=lambda *args: None),
        **{high_level_kernel: lambda *args, **kwargs: None},
    )
    monkeypatch.setattr(NODE, "_SAGE_CORE_MODULE", core)

    assert NODE.runtime_capability() == NODE.RuntimeCapability(
        False,
        "sageattention_kernel_unavailable",
        architecture,
    )


def test_kernel_probe_exception_is_private_and_fail_closed(monkeypatch):
    _configure_runtime(monkeypatch)

    class ExplodingCore:
        sageattn = staticmethod(_sage)

        def __getattr__(self, _name):
            raise RuntimeError("/opt/sentinel-sage-build/runtime")

    monkeypatch.setattr(NODE, "_SAGE_CORE_MODULE", ExplodingCore())
    capability = NODE.runtime_capability()
    assert capability == NODE.RuntimeCapability(
        False,
        "sageattention_kernel_unavailable",
        "sm80",
    )
    assert "sentinel-sage-build" not in str(capability)


def test_runtime_gate_runs_before_model_inspection_or_clone(monkeypatch):
    _configure_runtime(monkeypatch, cuda=_Cuda(available=False))
    model = _Patcher()
    with pytest.raises(RuntimeError, match="cuda_runtime_unavailable"):
        _apply(model)
    assert model.clone_count == 0


def test_explicit_non_cuda_model_placement_is_unavailable(monkeypatch):
    _configure_runtime(monkeypatch)
    model = _Patcher(load_device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="model_device_not_cuda"):
        _apply(model)
    assert model.clone_count == 0


def test_model_placement_selects_the_exact_gpu_capability(monkeypatch):
    cuda = _Cuda(count=2, current=0, capability={0: (8, 0), 1: (9, 0)})
    _configure_runtime(
        monkeypatch,
        cuda=cuda,
        compiled_architectures=("sm80", "sm90"),
    )
    model = _Patcher(
        MiniMaxH3Model(count=1),
        load_device=torch.device("cuda", 1),
    )

    result = _apply(model)

    installed = result.object_patches[
        "diffusion_model.blocks.0.attn.forward"
    ]
    assert installed.__func__.__closure__ is not None
    binding = next(
        cell.cell_contents
        for cell in installed.__func__.__closure__
        if isinstance(cell.cell_contents, NODE._RuntimeBinding)
    )
    assert binding.device_index == 1
    assert binding.architecture == "sm90"


def test_non_h3_model_fails_before_clone(monkeypatch):
    _configure_runtime(monkeypatch)
    model = _Patcher(OtherModel())
    with pytest.raises(TypeError, match="exact MiniMaxH3Model"):
        _apply(model)
    assert model.clone_count == 0


def test_duck_typed_model_patcher_is_rejected_before_clone(monkeypatch):
    _configure_runtime(monkeypatch)
    duck = SimpleNamespace(
        load_device=None,
        clone=lambda: None,
        get_model_object=lambda _name: MiniMaxH3Model(),
        add_object_patch=lambda _name, _value: None,
        object_patches={},
        object_patches_backup={},
        model_options={"transformer_options": {}},
    )

    with pytest.raises(TypeError, match="real ComfyUI ModelPatcher"):
        _apply(duck)


@pytest.mark.parametrize(
    "method_name",
    ("clone", "get_model_object", "add_object_patch"),
)
def test_model_patcher_subclass_method_override_is_rejected_before_clone(
    monkeypatch,
    method_name,
):
    _configure_runtime(monkeypatch)

    def replacement(*_args, **_kwargs):
        raise AssertionError("overridden ModelPatcher method must not execute")

    subclass = type(
        f"Overridden{method_name}",
        (_Patcher,),
        {method_name: replacement},
    )
    model = subclass()

    with pytest.raises(TypeError, match=f"ModelPatcher.{method_name}"):
        _apply(model)

    assert model.clone_count == 0


@pytest.mark.parametrize("missing", ["clone", "get_model_object", "add_object_patch"])
def test_incomplete_model_patcher_api_fails_before_clone(monkeypatch, missing):
    _configure_runtime(monkeypatch)
    model = _Patcher()
    setattr(model, missing, None)
    with pytest.raises(TypeError, match=missing):
        _apply(model)
    assert model.clone_count == 0


def test_empty_or_non_modulelist_blocks_fail_before_clone(monkeypatch):
    _configure_runtime(monkeypatch)
    empty = _Patcher(MiniMaxH3Model(count=0))
    with pytest.raises(TypeError, match="non-empty.*ModuleList"):
        _apply(empty)
    assert empty.clone_count == 0

    wrong = MiniMaxH3Model()
    wrong._modules.pop("blocks")
    wrong.__dict__["blocks"] = []
    model = _Patcher(wrong)
    with pytest.raises(TypeError, match="non-empty.*ModuleList"):
        _apply(model)
    assert model.clone_count == 0


def test_every_block_and_attention_must_have_exact_structure(monkeypatch):
    _configure_runtime(monkeypatch)

    class OtherBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    dm = MiniMaxH3Model(blocks=[DiTBlock(), OtherBlock()])
    model = _Patcher(dm)
    with pytest.raises(TypeError, match="exact MiniMax H3 DiTBlock.*block 1"):
        _apply(model)
    assert model.clone_count == 0

    class OtherAttention(Attention):
        pass

    dm = MiniMaxH3Model(blocks=[DiTBlock(OtherAttention())])
    model = _Patcher(dm)
    with pytest.raises(TypeError, match="exact MiniMax H3 Attention.*block 0"):
        _apply(model)
    assert model.clone_count == 0


def test_repeated_block_or_attention_objects_fail_before_clone(monkeypatch):
    _configure_runtime(monkeypatch)
    block = DiTBlock()
    model = _Patcher(MiniMaxH3Model(blocks=[block, block]))
    with pytest.raises(RuntimeError, match="repeated DiT block"):
        _apply(model)
    assert model.clone_count == 0

    attention = Attention()
    model = _Patcher(
        MiniMaxH3Model(blocks=[DiTBlock(attention), DiTBlock(attention)])
    )
    with pytest.raises(RuntimeError, match="repeated attention"):
        _apply(model)
    assert model.clone_count == 0


def test_instance_or_class_forward_override_fails_before_clone(monkeypatch):
    _configure_runtime(monkeypatch)
    attention = Attention()
    attention.forward = lambda value: value
    model = _Patcher(MiniMaxH3Model(blocks=[DiTBlock(attention)]))
    with pytest.raises(RuntimeError, match="pre-existing attention forward"):
        _apply(model)
    assert model.clone_count == 0

    original = Attention.forward
    monkeypatch.setattr(Attention, "forward", lambda self, value: value)
    model = _Patcher()
    with pytest.raises(RuntimeError, match="changed attention class contract"):
        _apply(model)
    assert model.clone_count == 0
    monkeypatch.setattr(Attention, "forward", original)


def test_incomplete_attention_api_fails_before_clone(monkeypatch):
    _configure_runtime(monkeypatch)
    attention = Attention()
    attention.heads = 0
    model = _Patcher(MiniMaxH3Model(blocks=[DiTBlock(attention)]))
    with pytest.raises(ValueError, match="invalid attention dimensions"):
        _apply(model)
    assert model.clone_count == 0

    attention = Attention()
    attention.qkv_proj = None
    model = _Patcher(MiniMaxH3Model(blocks=[DiTBlock(attention)]))
    with pytest.raises(TypeError, match="incomplete attention module.*qkv_proj"):
        _apply(model)
    assert model.clone_count == 0


@pytest.mark.parametrize("location", ["active", "backup"])
def test_patch_conflict_is_rejected_before_clone(monkeypatch, location):
    _configure_runtime(monkeypatch)
    key = "diffusion_model.blocks.1.attn.forward"
    kwargs = {
        "object_patches": {key: object()} if location == "active" else None,
        "object_patches_backup": {key: object()} if location == "backup" else None,
    }
    model = _Patcher(**kwargs)
    with pytest.raises(RuntimeError, match="already owned.*blocks.1"):
        _apply(model)
    assert model.clone_count == 0


def test_global_attention_override_conflict_is_rejected_before_clone(monkeypatch):
    _configure_runtime(monkeypatch)
    model = _Patcher()
    existing = lambda *_args, **_kwargs: None
    model.model_options["transformer_options"][
        "optimized_attention_override"
    ] = existing

    with pytest.raises(RuntimeError, match="existing optimized attention override"):
        _apply(model)

    assert model.clone_count == 0
    assert model.model_options["transformer_options"][
        "optimized_attention_override"
    ] is existing


def test_all_patch_keys_are_precomputed_and_installed_atomically(monkeypatch):
    _configure_runtime(monkeypatch)
    # A tensor value also proves preservation uses object identity rather than
    # tensor equality (whose truth value is ambiguous for multi-element data).
    unrelated = torch.ones(2)
    model = _Patcher(object_patches={"unrelated.patch": unrelated})

    result = _apply(model)

    expected = {
        "diffusion_model.blocks.0.attn.forward",
        "diffusion_model.blocks.1.attn.forward",
        "diffusion_model.blocks.2.attn.forward",
    }
    assert result is model.last_clone
    assert {key for key, _ in result.add_calls} == expected
    assert set(result.object_patches) == expected | {"unrelated.patch"}
    assert result.object_patches["unrelated.patch"] is unrelated
    assert model.object_patches == {"unrelated.patch": unrelated}
    for key in expected:
        installed = result.object_patches[key]
        assert result.get_model_object(key) is installed
        assert installed.__name__ == "director_strict_h3_sage_forward"


@pytest.mark.parametrize(
    ("behavior", "exception_type", "message"),
    [
        ("same", RuntimeError, "clone was not isolated"),
        ("duck_clone", TypeError, "real ComfyUI ModelPatcher"),
        ("different_model", RuntimeError, "clone changed the diffusion model"),
        ("shared_mapping", RuntimeError, "clone shared object patch state"),
    ],
)
def test_invalid_clone_is_rejected_without_input_mutation(
    monkeypatch,
    behavior,
    exception_type,
    message,
):
    _configure_runtime(monkeypatch)
    model = _Patcher(clone_behavior=behavior)
    with pytest.raises(exception_type, match=message):
        _apply(model)
    assert model.object_patches == {}


@pytest.mark.parametrize(
    ("behavior", "message"),
    [
        ("raise_second", "synthetic add failure"),
        ("drop_second", "changed the object patch set"),
        ("wrong_second", "installation was incomplete"),
        ("extra_first", "changed the object patch set"),
    ],
)
def test_partial_or_unverifiable_install_never_mutates_input(
    monkeypatch,
    behavior,
    message,
):
    _configure_runtime(monkeypatch)
    unrelated = object()
    model = _Patcher(
        object_patches={"unrelated.patch": unrelated},
        add_behavior=behavior,
    )
    with pytest.raises(RuntimeError, match=message):
        _apply(model)
    assert model.object_patches == {"unrelated.patch": unrelated}


def test_direct_runtime_uses_nhd_sage_and_never_host_attention(monkeypatch):
    calls = []

    def sage(q, k, v, tensor_layout="HND", is_causal=True):
        calls.append((q, k, v, tensor_layout, is_causal))
        return v

    _configure_runtime(monkeypatch, sage=sage)
    monkeypatch.setattr(
        NODE,
        "_require_runtime_attention_tensor",
        lambda _x, **_kwargs: None,
    )
    model = _Patcher(MiniMaxH3Model(count=1))
    result = _apply(model)
    forward = result.object_patches[
        "diffusion_model.blocks.0.attn.forward"
    ]
    value = torch.arange(12, dtype=torch.float16).reshape(3, 4)

    output = forward(value, transformer_options={})

    assert torch.equal(output, value)
    assert len(calls) == 1
    q, k, v, layout, causal = calls[0]
    assert q.shape == k.shape == v.shape == (1, 3, 2, 2)
    assert layout == "NHD"
    assert causal is False


def test_sage_runtime_exception_propagates_unchanged_without_fallback(monkeypatch):
    class KernelFailure(RuntimeError):
        pass

    failure = KernelFailure("sentinel kernel failure")
    calls = 0

    def sage(q, k, v, tensor_layout="HND", is_causal=True):
        nonlocal calls
        del q, k, v, tensor_layout, is_causal
        calls += 1
        raise failure

    _configure_runtime(monkeypatch, sage=sage)
    monkeypatch.setattr(
        NODE,
        "_require_runtime_attention_tensor",
        lambda _x, **_kwargs: None,
    )
    result = _apply(_Patcher(MiniMaxH3Model(count=1)))
    forward = result.object_patches[
        "diffusion_model.blocks.0.attn.forward"
    ]

    with pytest.raises(KernelFailure) as caught:
        forward(torch.ones((2, 4), dtype=torch.float16))
    assert caught.value is failure
    assert calls == 1


def test_rope_path_uses_exact_comfy_api_before_direct_sage(monkeypatch):
    calls = []

    def sage(q, k, v, tensor_layout="HND", is_causal=True):
        calls.append((tensor_layout, is_causal))
        return v

    ck, cast_calls = _configure_runtime(monkeypatch, sage=sage)
    monkeypatch.setattr(
        NODE,
        "_require_runtime_attention_tensor",
        lambda _x, **_kwargs: None,
    )
    result = _apply(_Patcher(MiniMaxH3Model(count=1)))
    forward = result.object_patches[
        "diffusion_model.blocks.0.attn.forward"
    ]
    value = torch.ones((2, 4), dtype=torch.float16)
    rope = torch.ones((2, 1, 2), dtype=torch.float16)

    assert torch.equal(forward(value, rope_freqs=rope), value)
    assert len(cast_calls) == 2
    assert len(ck.inplace_calls) == 1
    assert ck.inplace_calls[0][-1] == {"epsilon": 1e-5, "rot_dim": 4}
    assert ck.training_calls == []
    assert calls == [("NHD", False)]


def test_training_rope_path_uses_non_mutating_comfy_api(monkeypatch):
    ck, _ = _configure_runtime(monkeypatch, in_training=True)
    monkeypatch.setattr(
        NODE,
        "_require_runtime_attention_tensor",
        lambda _x, **_kwargs: None,
    )
    result = _apply(_Patcher(MiniMaxH3Model(count=1)))
    forward = result.object_patches[
        "diffusion_model.blocks.0.attn.forward"
    ]
    value = torch.ones((2, 4), dtype=torch.float16)
    rope = torch.ones((2, 1, 2), dtype=torch.float16)

    forward(value, rope_freqs=rope)

    assert len(ck.training_calls) == 1
    assert ck.inplace_calls == []


def test_runtime_input_qkv_and_kernel_output_contracts_fail_closed(monkeypatch):
    _configure_runtime(monkeypatch)
    with pytest.raises(RuntimeError, match="requires a CUDA tensor"):
        NODE._run_strict_h3_sage_forward(
            Attention(),
            _binding(),
            torch.ones((2, 4), dtype=torch.float16),
        )

    monkeypatch.setattr(
        NODE,
        "_require_runtime_attention_tensor",
        lambda _x, **_kwargs: None,
    )
    attention = Attention()
    attention.qkv_proj = _Projection(copies=2)
    with pytest.raises(RuntimeError, match="qkv projection"):
        NODE._run_strict_h3_sage_forward(
            attention,
            _binding(),
            torch.ones((2, 4), dtype=torch.float16),
        )

    def wrong_shape(q, k, v, tensor_layout="HND", is_causal=True):
        del q, k, tensor_layout, is_causal
        return v.reshape(2, 4)

    with pytest.raises(RuntimeError, match="output shape"):
        NODE._run_strict_h3_sage_forward(
            Attention(),
            _binding(wrong_shape),
            torch.ones((2, 4), dtype=torch.float16),
        )


def test_pyproject_does_not_implicitly_install_gpu_specific_sage():
    pyproject = (NODE_PATH.parent / "pyproject.toml").read_text()
    assert "dependencies = []" in pyproject
    assert "sageattention" not in pyproject.lower()
