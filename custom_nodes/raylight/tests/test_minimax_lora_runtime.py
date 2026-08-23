from __future__ import annotations

import logging
import sys
import types
from collections import OrderedDict
from unittest.mock import patch

import comfy.cli_args

comfy.cli_args.args.cpu = True

import torch


pipefusion = types.ModuleType("directordeck_raylight.diffusion_models.wan.pipefusion")
pipefusion.filter_wan_state_dict_for_stage = lambda *args, **kwargs: None
pipefusion.partition_wan_for_pipefusion = lambda *args, **kwargs: None
pipefusion.inject_wan21_pipefusion = lambda *args, **kwargs: None
sys.modules.setdefault(pipefusion.__name__, pipefusion)

from directordeck_raylight.comfy_dist import sd
from directordeck_raylight import nodes
from directordeck_raylight.diffusion_models.minimax.lora import normalize_minimax_h3_lora_keys
from directordeck_raylight.distributed_worker.ray_worker import RayWorker


TARGET = "diffusion_model.blocks.0.attn.qkv_proj.weight"
A_KEY = "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"
B_KEY = "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"
ALPHA_KEY = "diffusion_model.blocks.0.attn.qkv_proj.alpha"


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = torch.nn.Linear(8, 12, bias=False)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attention()


class MiniMaxH3Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])
        self.use_adaln_curves = False


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = MiniMaxH3Model()


class OtherModel(torch.nn.Module):
    pass


class _OtherBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = OtherModel()


class _OtherPatcher:
    def __init__(self):
        self.model = _OtherBase()

    def clone(self):
        return _OtherPatcher()


class ExternalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])


class _ExternalBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = ExternalModel()


class _Patcher:
    def __init__(self, *, patched_keys=None):
        self.model = _Base()
        self.patched_keys = patched_keys
        self.clone_count = 0
        self.injections = {}
        self.attachments = {}

    def clone(self):
        self.clone_count += 1
        clone = _Patcher(patched_keys=self.patched_keys)
        self.last_clone = clone
        return clone

    def add_patches(self, loaded, strength):
        del strength
        return list(loaded) if self.patched_keys is None else list(self.patched_keys)

    def set_injections(self, key, injections):
        self.injections[key] = injections

    def get_attachment(self, key):
        return self.attachments.get(key)

    def set_attachments(self, key, value):
        self.attachments[key] = value


class _ExternalPatcher(_Patcher):
    def __init__(self, *, patched_keys=None):
        super().__init__(patched_keys=patched_keys)
        self.model = _ExternalBase()

    def clone(self):
        self.clone_count += 1
        clone = _ExternalPatcher(patched_keys=self.patched_keys)
        self.last_clone = clone
        return clone


class _Adapter(sd.comfy.weight_adapter.WeightAdapterBase):
    name = "lora"

    def __init__(self, loaded_keys=None):
        self.loaded_keys = (
            {A_KEY, B_KEY, ALPHA_KEY} if loaded_keys is None else set(loaded_keys)
        )
        self.weights = (torch.zeros((2, 8)), torch.zeros((12, 2)), None)

    def h(self, x, base_out):
        del x
        return torch.zeros_like(base_out)


class _ShapeMismatchAdapter(_Adapter):
    def h(self, x, base_out):
        del x
        return torch.ones(
            (base_out.shape[-1] + 1,),
            device=base_out.device,
            dtype=base_out.dtype,
        )


class _Manager:
    hook_count = 1

    def __init__(self):
        self.adapters = []

    def add_adapter(self, key, adapter, strength):
        self.adapters.append((key, adapter, strength))

    def create_injections(self, model):
        del model
        return [object()]

    def get_hook_count(self):
        return self.hook_count


class _CapturingManager(_Manager):
    last_instance = None

    def __init__(self):
        super().__init__()
        type(self).last_instance = self


def _lora():
    return {
        A_KEY: torch.ones((2, 8)),
        B_KEY: torch.ones((12, 2)),
        ALPHA_KEY: torch.tensor(2.0),
    }


def _bare_worker(*, is_fsdp: bool, director_strict_h3: bool = True):
    worker = object.__new__(RayWorker)
    worker.parallel_dict = {
        "is_fsdp": is_fsdp,
        "is_quant": False,
        "use_mmap": False,
        "pipefusion_enabled": False,
        "director_strict_h3": director_strict_h3,
    }
    worker.model = None
    worker.state_dict = None
    worker.lora_list = None
    worker.ram_cache = OrderedDict()
    worker.base_patcher = None
    worker.current_base_key = None
    worker.active_request_key = None
    worker.cached_controlnet = None
    worker.vae_model = None
    worker.local_rank = 0
    worker.device_mesh = None
    worker.is_cpu_offload = False
    worker.xfuser_parallel = None
    worker._free_cached_aux_models = lambda: None
    return worker


def _assert_raises(exception_type, message, function):
    try:
        function()
    except exception_type as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected {exception_type.__name__}: {message}")


def _loader_patches(loaded, *, hook_count=1):
    class _ConfiguredManager(_Manager):
        pass

    _ConfiguredManager.hook_count = hook_count
    return (
        patch.object(sd.comfy.lora_convert, "convert_lora", side_effect=lambda value: value),
        patch.object(
            sd.comfy.lora,
            "model_lora_keys_unet",
            return_value={
                "diffusion_model.blocks.0.attn.qkv_proj": TARGET,
            },
        ),
        patch.object(sd.comfy.lora, "load_lora", return_value=loaded),
        patch.object(sd.comfy.weight_adapter, "BypassInjectionManager", _ConfiguredManager),
    )


def test_non_fsdp_zero_resolution_fails_before_clone():
    model = _Patcher()
    patches = _loader_patches({})
    with patches[0], patches[1], patches[2], patches[3]:
        _assert_raises(
            RuntimeError,
            "adapter resolution was incomplete",
            lambda: sd.load_minimax_h3_lora_for_models_strict(model, _lora(), 1.0),
        )
    assert model.clone_count == 0


def test_runtime_rejects_non_finite_or_identity_adapter_values():
    cases = (
        (ALPHA_KEY, torch.tensor(0.0), "alpha must be a finite non-zero scalar"),
        (ALPHA_KEY, torch.tensor(float("nan")), "alpha must be a finite non-zero scalar"),
        (A_KEY, torch.zeros((2, 8)), "must not be all zero"),
        (B_KEY, torch.zeros((12, 2)), "must not be all zero"),
        (A_KEY, torch.full((2, 8), float("inf")), "only finite values"),
        (B_KEY, torch.full((12, 2), float("nan")), "only finite values"),
    )
    for key, value, message in cases:
        lora = _lora()
        lora[key] = value
        _assert_raises(
            ValueError,
            message,
            lambda lora=lora: normalize_minimax_h3_lora_keys(_Patcher(), lora),
        )


def test_non_fsdp_partial_source_key_consumption_fails_before_clone():
    model = _Patcher()
    patches = _loader_patches({TARGET: _Adapter({A_KEY})})
    with patches[0], patches[1], patches[2], patches[3]:
        _assert_raises(
            RuntimeError,
            "did not consume every source key",
            lambda: sd.load_minimax_h3_lora_for_models_strict(model, _lora(), 1.0),
        )
    assert model.clone_count == 0


def test_non_fsdp_partial_hook_fails_before_injection_installation():
    model = _Patcher()
    patches = _loader_patches({TARGET: _Adapter()}, hook_count=0)
    with patches[0], patches[1], patches[2], patches[3]:
        _assert_raises(
            RuntimeError,
            "hook installation was incomplete",
            lambda: sd.load_minimax_h3_lora_for_models_strict(model, _lora(), 1.0),
        )
    assert not model.last_clone.injections


def test_non_fsdp_valid_path_installs_exact_bypass():
    model = _Patcher()
    patches = _loader_patches({TARGET: _Adapter()})
    with patches[0], patches[1], patches[2], patches[3]:
        result = sd.load_minimax_h3_lora_for_models_strict(model, _lora(), 1.0)
    assert result is model.last_clone
    assert len(result.injections["bypass_lora"]) == 1


def test_fsdp_merge_partial_patch_fails_closed():
    model = _Patcher(patched_keys=())
    with (
        patch.object(sd.comfy.lora_convert, "convert_lora", side_effect=lambda value: value),
        patch.object(sd.comfy.lora, "model_lora_keys_unet", return_value={}),
        patch.object(sd.comfy_dist.lora, "load_lora", return_value={TARGET: _Adapter()}),
    ):
        _assert_raises(
            RuntimeError,
            "patch installation was incomplete",
            lambda: sd.load_lora_for_models(model, _lora(), 1.0),
        )


def test_fsdp_merge_valid_path_applies_every_target():
    model = _Patcher()
    with (
        patch.object(sd.comfy.lora_convert, "convert_lora", side_effect=lambda value: value),
        patch.object(sd.comfy.lora, "model_lora_keys_unet", return_value={}),
        patch.object(sd.comfy_dist.lora, "load_lora", return_value={TARGET: _Adapter()}),
    ):
        result = sd.load_lora_for_models(model, _lora(), 1.0)
    assert result is model.last_clone


def test_quantized_partial_hook_fails_before_attachment_or_injection():
    model = _Patcher()
    class _NoHookManager(_Manager):
        hook_count = 0

    with (
        patch.object(sd.comfy.lora_convert, "convert_lora", side_effect=lambda value: value),
        patch.object(sd.comfy.lora, "model_lora_keys_unet", return_value={}),
        patch.object(sd.comfy_dist.lora, "load_lora", return_value={TARGET: _Adapter()}),
        patch.object(sd.comfy.weight_adapter, "BypassInjectionManager", _NoHookManager),
    ):
        _assert_raises(
            RuntimeError,
            "hook installation was incomplete",
            lambda: sd.load_lora_for_models_quantized(model, _lora(), 1.0),
        )
    clone = model.last_clone
    assert not clone.attachments
    assert not clone.injections


def test_quantized_valid_path_records_attachment_and_injection():
    model = _Patcher()
    with (
        patch.object(sd.comfy.lora_convert, "convert_lora", side_effect=lambda value: value),
        patch.object(sd.comfy.lora, "model_lora_keys_unet", return_value={}),
        patch.object(sd.comfy_dist.lora, "load_lora", return_value={TARGET: _Adapter()}),
        patch.object(sd.comfy.weight_adapter, "BypassInjectionManager", _Manager),
    ):
        result = sd.load_lora_for_models_quantized(model, _lora(), 1.0)
    assert result is model.last_clone
    assert sd.FSDP_LORA_SIDECAR_ATTACHMENT in result.attachments
    assert len(result.injections["quantized_lora_bypass"]) == 1


def _load_dynamic_sidecar_adapter(model):
    _CapturingManager.last_instance = None
    with (
        patch.object(sd.comfy.lora_convert, "convert_lora", side_effect=lambda value: value),
        patch.object(sd.comfy.lora, "model_lora_keys_unet", return_value={}),
        patch.object(
            sd.comfy_dist.lora,
            "load_lora",
            return_value={TARGET: _ShapeMismatchAdapter()},
        ),
        patch.object(
            sd.comfy.weight_adapter,
            "BypassInjectionManager",
            _CapturingManager,
        ),
    ):
        sd.load_lora_for_models_quantized(
            model,
            _lora(),
            1.0,
            dynamic_sidecar=True,
        )
    manager = _CapturingManager.last_instance
    assert manager is not None
    assert len(manager.adapters) == 1
    return manager.adapters[0][1]


def test_strict_h3_dynamic_sidecar_shape_mismatch_raises_not_identity(caplog):
    adapter = _load_dynamic_sidecar_adapter(_Patcher())
    x = torch.ones((2, 3, 8))
    base_out = torch.ones((2, 3, 12))

    caplog.set_level(logging.WARNING)
    _assert_raises(
        RuntimeError,
        "MiniMax H3 LoRA bypass output contract mismatch",
        lambda: adapter.h(x, base_out),
    )
    assert "BYPASS OFFSET SHAPE MISMATCH" not in caplog.text


def test_external_dynamic_sidecar_shape_mismatch_keeps_warning_identity(caplog):
    adapter = _load_dynamic_sidecar_adapter(_ExternalPatcher())
    x = torch.ones((2, 3, 8))
    base_out = torch.ones((2, 3, 12))

    caplog.set_level(logging.WARNING)
    delta = adapter.h(x, base_out)
    assert tuple(delta.shape) == tuple(base_out.shape)
    assert torch.count_nonzero(delta).item() == 0
    assert "BYPASS OFFSET SHAPE MISMATCH" in caplog.text


def test_ray_lora_loader_rejects_positive_and_negative_zero():
    loader = nodes.RayLoraLoader()
    for strength in (0.0, -0.0):
        _assert_raises(
            ValueError,
            "finite and non-zero",
            lambda strength=strength: loader.load_lora(
                "test.safetensors", strength, [{"existing": True}]
            ),
        )


def test_ray_lora_loader_preserves_valid_previous_descriptor_chain():
    previous = [{"path": "first.safetensors", "strength_model": 0.5}]
    with patch.object(
        nodes.folder_paths,
        "get_full_path_or_raise",
        return_value="second.safetensors",
    ):
        (result,) = nodes.RayLoraLoader().load_lora(
            "second.safetensors", -0.75, previous
        )
    assert result == [
        {"path": "first.safetensors", "strength_model": 0.5},
        {"path": "second.safetensors", "strength_model": -0.75},
    ]
    assert previous == [{"path": "first.safetensors", "strength_model": 0.5}]


def test_director_namespace_sets_strict_h3_worker_metadata_only_for_director():
    assert nodes._director_runtime_metadata("director-g123-e7") == {
        "director_strict_h3": True
    }
    assert nodes._director_runtime_metadata("default") == {
        "director_strict_h3": False
    }


def test_director_non_h3_lora_fails_before_every_generic_loader():
    for is_fsdp in (False, True):
        worker = _bare_worker(is_fsdp=is_fsdp)
        worker.model = _OtherPatcher()
        worker.lora_list = [{"path": "other.safetensors", "strength_model": 1.0}]
        with (
            patch.object(sd.comfy.sd, "load_lora_for_models") as comfy_generic,
            patch.object(sd, "load_lora_for_models") as fsdp_generic,
            patch.object(sd, "load_lora_for_models_quantized") as fsdp_quantized,
        ):
            _assert_raises(
                RuntimeError,
                "require an actual MiniMaxH3Model",
                worker.load_lora,
            )
        comfy_generic.assert_not_called()
        fsdp_generic.assert_not_called()
        fsdp_quantized.assert_not_called()


def test_director_model_gate_accepts_h3_for_fsdp_and_non_fsdp():
    for is_fsdp in (False, True):
        worker = _bare_worker(is_fsdp=is_fsdp)
        worker.model = _Patcher()
        worker._require_director_minimax_h3_model()


def test_external_non_h3_raylight_keeps_generic_lora_behavior():
    worker = _bare_worker(is_fsdp=False, director_strict_h3=False)
    worker.model = _OtherPatcher()
    worker.lora_list = [{"path": "other.safetensors", "strength_model": 0.5}]
    replacement = _OtherPatcher()
    with (
        patch.object(sd.comfy.utils, "load_torch_file", return_value={"other.lora": torch.ones(1)}),
        patch.object(
            sd.comfy.sd,
            "load_lora_for_models",
            return_value=(replacement, None),
        ) as comfy_generic,
    ):
        worker.load_lora()
    assert worker.model is replacement
    comfy_generic.assert_called_once()


def test_director_non_fsdp_without_lora_rejects_loaded_non_h3_model():
    worker = _bare_worker(is_fsdp=False)
    with patch.object(
        sd.comfy.sd, "load_diffusion_model", return_value=_OtherPatcher()
    ) as loader:
        _assert_raises(
            RuntimeError,
            "require an actual MiniMaxH3Model",
            lambda: worker.load_unet("other.ckpt", {}),
        )
    loader.assert_called_once()


def test_director_fsdp_without_lora_rejects_loaded_non_h3_model():
    worker = _bare_worker(is_fsdp=True)
    import comfy.model_management as comfy_model_management
    import comfy.model_patcher as comfy_model_patcher

    with (
        patch.object(torch.cuda, "synchronize"),
        patch.object(comfy_model_management, "soft_empty_cache"),
        patch.object(comfy_model_patcher, "LowVramPatch", create=True),
        patch.object(comfy_model_management, "cleanup_models_gc", create=True),
        patch.object(
            sd,
            "fsdp_load_diffusion_model",
            return_value=(_OtherPatcher(), {}),
        ) as loader,
    ):
        _assert_raises(
            RuntimeError,
            "require an actual MiniMaxH3Model",
            lambda: worker.load_unet("other.safetensors", {}),
        )
    loader.assert_called_once()
