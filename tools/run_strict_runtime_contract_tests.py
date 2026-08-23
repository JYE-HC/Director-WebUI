#!/usr/bin/env python3
"""Run all bundled strict-runtime contract tests without a GPU install.

The production node modules are imported and exercised by their real tests.  A
small deterministic host shim supplies only the ComfyUI, Ray and tensor surface
that the CPU contract tests need; no model weights are loaded, no sampling is
performed, and no mock result is treated as GPU availability.  This keeps the
release gate runnable from the root locked development environment on both
Linux and Windows.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAYLIGHT_SOURCE = PROJECT_ROOT / "custom_nodes" / "raylight" / "src"
RAYLIGHT_ATTENTION_TEST = (
    PROJECT_ROOT
    / "custom_nodes"
    / "raylight"
    / "tests"
    / "test_comfy_kitchen_attention.py"
)
STRICT_TESTS = (
    PROJECT_ROOT
    / "custom_nodes"
    / "DirectorDeck-Strict-Attention"
    / "tests"
    / "test_cpu_contract_gate.py",
    PROJECT_ROOT
    / "custom_nodes"
    / "DirectorDeck-Strict-H3"
    / "tests"
    / "test_cpu_contract_gate.py",
    PROJECT_ROOT / "custom_nodes" / "raylight" / "tests" / "test_minimax_lora.py",
    PROJECT_ROOT
    / "custom_nodes"
    / "raylight"
    / "tests"
    / "test_minimax_lora_runtime.py",
    f"{RAYLIGHT_ATTENTION_TEST}::test_choices_append_ck_without_changing_existing_enum_order",
    f"{RAYLIGHT_ATTENTION_TEST}::test_ck_rejects_ring_before_loading_kernel",
    f"{RAYLIGHT_ATTENTION_TEST}::test_ck_unavailable_on_worker_fails_without_fallback",
)
# The removed third-party H3/LoRA packs contributed 50 tests.  Keep the gate
# pinned to the complete Director-owned/RayLight contract suite that remains.
MINIMUM_CONTRACT_TESTS = 46
_HOST_TEMP: tempfile.TemporaryDirectory[str] | None = None


def _controlled_host_paths() -> tuple[Path, Path]:
    """Create the tiny filesystem identity that RayLight resolves at import."""

    global _HOST_TEMP
    if _HOST_TEMP is None:
        _HOST_TEMP = tempfile.TemporaryDirectory(prefix="directordeck-strict-runtime-")
        host_root = Path(_HOST_TEMP.name)
        comfy_root = host_root / "ComfyUI"
        raylight_root = host_root / "directordeck_raylight"
        comfy_root.mkdir()
        raylight_root.mkdir()
        (comfy_root / "main.py").write_text("", encoding="utf-8")
        (comfy_root / "execution.py").write_text("", encoding="utf-8")
        (comfy_root / "folder_paths.py").write_text("", encoding="utf-8")
        (raylight_root / "__init__.py").write_text("", encoding="utf-8")
        (host_root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    host_root = Path(_HOST_TEMP.name)
    return host_root / "ComfyUI", host_root / "directordeck_raylight"


def _module(name: str, *, package_path: Path | None = None) -> ModuleType:
    value = ModuleType(name)
    if package_path is not None:
        value.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules[name] = value
    return value


class _Tensor:
    """Shape-carrying CPU value used by the strict effect tests."""

    def __init__(self, shape: tuple[int, ...] | list[int] = (), value: Any = 0.0):
        self.shape = tuple(shape)
        self._value = value
        self.dtype = "float32"
        self.device = "cpu"

    def item(self) -> Any:
        return self._value

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to(self, *args: Any, **kwargs: Any) -> "_Tensor":
        del args, kwargs
        return self

    def float(self) -> "_Tensor":
        return self

    def reshape_as(self, other: "_Tensor") -> "_Tensor":
        return _Tensor(other.shape, self._value)

    def expand_as(self, other: "_Tensor") -> "_Tensor":
        return _Tensor(other.shape, self._value)

    def reshape(self, *_shape: int) -> "_Tensor":
        return self

    def isfinite(self) -> "_Tensor":
        try:
            finite = math.isfinite(float(self._value))
        except (TypeError, ValueError):
            finite = False
        return _Tensor(self.shape, finite)

    def ne(self, value: Any) -> "_Tensor":
        return _Tensor(self.shape, self._value != value)

    def all(self) -> "_Tensor":
        return _Tensor((), bool(self._value))

    def any(self) -> "_Tensor":
        return _Tensor((), bool(self._value))

    def __getitem__(self, _index: Any) -> "_Tensor":
        return self

    def __setitem__(self, _index: Any, value: Any) -> None:
        self._value = value

    def __mul__(self, other: Any) -> "_Tensor":
        del other
        return _Tensor(self.shape, self._value)

    __rmul__ = __mul__

    def __add__(self, other: Any) -> "_Tensor":
        value = getattr(other, "_value", other)
        try:
            result = self._value + value
        except TypeError:
            result = self._value
        return _Tensor(self.shape, result)

    __radd__ = __add__


class _Device(str):
    def __new__(cls, device_type: str, index: int | None = None):
        value = device_type if index is None else f"{device_type}:{index}"
        instance = super().__new__(cls, value)
        instance.type = device_type
        instance.index = index
        return instance


class _Module:
    def __init__(self) -> None:
        pass

    def state_dict(self) -> dict[str, _Tensor]:
        result: dict[str, _Tensor] = {}

        def visit(value: Any, prefix: str) -> None:
            if isinstance(value, _Tensor):
                result[prefix] = value
            elif isinstance(value, _Module):
                for name, child in vars(value).items():
                    visit(child, f"{prefix}.{name}" if prefix else name)
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    visit(child, f"{prefix}.{index}" if prefix else str(index))

        visit(self, "")
        return result


class _ModuleList(list):
    pass


class _Linear(_Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = _Tensor((out_features, in_features))
        self.bias = _Tensor((out_features,)) if bias else None


class _Conv(_Module):
    pass


class _MiniMaxH3Attention(_Module):
    def forward(self, value: Any, *_args: Any, **_kwargs: Any) -> Any:
        return value


class _MiniMaxH3DiTBlock(_Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _MiniMaxH3Attention()


class _MiniMaxH3Model(_Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = _ModuleList([_MiniMaxH3DiTBlock()])


class _ModelPatcher:
    def __init__(
        self,
        model: Any,
        load_device: Any = None,
        _offload_device: Any = None,
        **_kwargs: Any,
    ) -> None:
        self.model = model
        self.load_device = load_device
        self.model_options: dict[str, Any] = {"transformer_options": {}}
        self.object_patches: dict[str, Any] = {}
        self.object_patches_backup: dict[str, Any] = {}
        self.parent = None
        self.clone_count = 0

    def clone(self) -> "_ModelPatcher":
        self.clone_count += 1
        cloned = _ModelPatcher(self.model, self.load_device)
        cloned.parent = self
        cloned.model_options = {
            **self.model_options,
            "transformer_options": dict(
                self.model_options.get("transformer_options", {})
            ),
        }
        cloned.object_patches = dict(self.object_patches)
        cloned.object_patches_backup = dict(self.object_patches_backup)
        return cloned

    def set_model_optimized_attention(self, optimized_attention: Any) -> None:
        def optimized_attention_override(
            _fallback: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return optimized_attention(*args, **kwargs)

        container = getattr(optimized_attention, "container_function", None)
        if container is not None:
            optimized_attention_override.container_function = container
        self.model_options["transformer_options"][
            "optimized_attention_override"
        ] = optimized_attention_override

    def get_model_object(self, name: str) -> Any:
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        current = self.model
        for component in name.split("."):
            current = (
                current[int(component)]
                if component.isdigit()
                else getattr(current, component)
            )
        return current

    def add_object_patch(self, name: str, value: Any) -> None:
        self.object_patches[name] = value


def _install_torch_shim() -> ModuleType:
    torch = _module("torch")
    nn = _module("torch.nn")
    functional = _module("torch.nn.functional")
    distributed = _module("torch.distributed")
    distributed_fsdp = _module("torch.distributed.fsdp")

    nn.Module = _Module
    nn.ModuleList = _ModuleList
    nn.Linear = _Linear
    nn.Conv1d = _Conv
    nn.Conv2d = _Conv
    nn.Conv3d = _Conv
    nn.functional = functional

    functional.linear = lambda *_args, **_kwargs: _Tensor()
    functional.conv1d = lambda *_args, **_kwargs: _Tensor()
    functional.conv2d = lambda *_args, **_kwargs: _Tensor()
    functional.conv3d = lambda *_args, **_kwargs: _Tensor()
    functional.silu = lambda value: value

    torch.Tensor = _Tensor
    torch.nn = nn
    shape_tuple = lambda shape: (shape,) if isinstance(shape, int) else tuple(shape)
    torch.zeros = lambda shape, *args, **kwargs: _Tensor(shape_tuple(shape))
    torch.ones = lambda shape, *args, **kwargs: _Tensor(shape_tuple(shape), 1.0)
    torch.full = lambda shape, value, *args, **kwargs: _Tensor(
        shape_tuple(shape), value
    )
    torch.zeros_like = lambda value: _Tensor(value.shape)
    torch.tensor = lambda value, *args, **kwargs: _Tensor((), value)
    torch.cat = lambda values, dim=0: _Tensor(values[0].shape if values else ())
    torch.stack = lambda values, dim=0: _Tensor(
        (len(values), *(values[0].shape if values else ()))
    )
    torch.lerp = lambda start, end, weight: start
    torch.is_tensor = lambda value: isinstance(value, _Tensor)
    torch.isfinite = lambda value: value.isfinite()
    torch.count_nonzero = lambda value: _Tensor((), int(bool(value._value)))
    torch.no_grad = lambda: (lambda function: function)
    torch.float8_e4m3fn = "float8_e4m3fn"
    torch.float8_e5m2 = "float8_e5m2"
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.device = _Device
    torch.cuda = SimpleNamespace(
        device_count=lambda: 0,
        current_device=lambda: 0,
        get_device_capability=lambda: (0, 0),
        is_available=lambda: False,
        synchronize=lambda: None,
        empty_cache=lambda: None,
        ipc_collect=lambda: None,
    )
    distributed.ReduceOp = SimpleNamespace(SUM="sum")
    distributed.init_process_group = lambda *_args, **_kwargs: None
    distributed.destroy_process_group = lambda: None
    distributed.all_reduce = lambda *_args, **_kwargs: None
    distributed_fsdp.FSDPModule = type("FSDPModule", (), {})
    distributed.fsdp = distributed_fsdp
    torch.distributed = distributed
    return torch


class _WeightAdapterBase:
    pass


class _LoRAAdapter(_WeightAdapterBase):
    name = "lora"

    def __init__(self, loaded_keys: Any, weights: Any):
        # ComfyUI deliberately retains the one mutable set shared by every
        # adapter loaded from a file.  The strict-node regression suite relies
        # on this exact aliasing behavior.
        self.loaded_keys = loaded_keys
        self.weights = weights

    def h(self, _x: Any, base_out: _Tensor) -> _Tensor:
        return _Tensor(base_out.shape)


class _BypassInjectionManager:
    def __init__(self) -> None:
        self.adapters: list[tuple[str, Any, float]] = []

    def add_adapter(self, key: str, adapter: Any, strength: float) -> None:
        self.adapters.append((key, adapter, strength))

    def create_injections(self, _model: Any) -> list[object]:
        return [object() for _ in self.adapters]

    def get_hook_count(self) -> int:
        return len(self.adapters)


def _default_lora_loader(lora: dict[str, Any], key_map: dict[str, str], **_kwargs: Any):
    loaded: dict[str, _LoRAAdapter] = {}
    loaded_keys: set[str] = set()
    for module_name, target in key_map.items():
        alpha_name = f"{module_name}.alpha"
        alpha = None
        if alpha_name in lora:
            alpha_value = lora[alpha_name]
            alpha = alpha_value.item() if hasattr(alpha_value, "item") else alpha_value
            loaded_keys.add(alpha_name)
        for up_suffix, down_suffix in (
            (".lora_up.weight", ".lora_down.weight"),
            ("_lora.up.weight", "_lora.down.weight"),
            (".lora_B.weight", ".lora_A.weight"),
            (".lora.up.weight", ".lora.down.weight"),
            (".lora_B", ".lora_A"),
            (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
            (".lora_B.default.weight", ".lora_A.default.weight"),
        ):
            up_name = f"{module_name}{up_suffix}"
            if up_name not in lora:
                continue
            down_name = f"{module_name}{down_suffix}"
            loaded_keys.add(up_name)
            loaded_keys.add(down_name)
            loaded[target] = _LoRAAdapter(
                loaded_keys,
                (
                    lora[up_name],
                    lora[down_name],
                    alpha,
                    None,
                    None,
                    None,
                ),
            )
            break
    return loaded


def _install_comfy_shim() -> ModuleType:
    comfy_host, _ = _controlled_host_paths()
    comfy = _module("comfy", package_path=PROJECT_ROOT / ".strict-runtime-comfy")
    cli_args = _module("comfy.cli_args")
    cli_args.args = SimpleNamespace(
        cpu=False,
        disable_mmap=False,
        mmap_torch_files=False,
        disable_smart_memory=False,
        disable_async_offload=False,
        disable_dynamic_vram=False,
        enable_dynamic_vram=False,
        vram_headroom=None,
        force_non_blocking=False,
        deterministic=False,
        verbose=False,
        reserve_vram=None,
        fp16_intermediates=False,
        force_channels_last=False,
        supports_fp8_compute=False,
        enable_triton_backend=False,
        fast=(),
    )

    samplers = _module("comfy.samplers")
    samplers.KSAMPLER = lambda function: function
    samplers.KSampler = type(
        "KSampler",
        (),
        {"SAMPLERS": (), "SCHEDULERS": ()},
    )
    model_sampling = _module("comfy.model_sampling")
    model_sampling.ModelSamplingAV = type("ModelSamplingAV", (), {})

    lora = _module("comfy.lora")
    lora.load_lora = _default_lora_loader
    lora.model_lora_keys_unet = lambda _model, key_map: key_map
    lora_convert = _module("comfy.lora_convert")
    lora_convert.convert_lora = lambda value: value

    weight_adapter = _module("comfy.weight_adapter")
    weight_adapter.WeightAdapterBase = _WeightAdapterBase
    weight_adapter.LoRAAdapter = _LoRAAdapter
    weight_adapter.BypassInjectionManager = _BypassInjectionManager

    utils = _module("comfy.utils")
    utils.load_torch_file = lambda *_args, **_kwargs: {}

    def get_attr(root: Any, path: str) -> Any:
        value = root
        for part in path.split("."):
            value = value[int(part)] if part.isdigit() else getattr(value, part)
        return value

    utils.get_attr = get_attr
    patcher_extension = _module("comfy.patcher_extension")
    patcher_extension.WrappersMP = SimpleNamespace(
        DIFFUSION_MODEL="diffusion_model",
        OUTER_SAMPLE="outer_sample",
        PREDICT_NOISE="predict_noise",
    )
    patcher_extension.CallbacksMP = SimpleNamespace(ON_LOAD="on_load")

    sd = _module("comfy.sd")
    sd.model_detection_error_hint = lambda error: error
    sd.load_lora_for_models = lambda model, *_args, **_kwargs: (model, None)
    sd.load_diffusion_model = lambda *_args, **_kwargs: None
    sample = _module("comfy.sample")
    model_detection = _module("comfy.model_detection")
    model_management = _module("comfy.model_management")
    model_management.unload_all_models = lambda: None
    model_management.soft_empty_cache = lambda: None
    model_management.cast_to = lambda value, **_kwargs: value
    model_management.in_training = False
    memory_management = _module("comfy.memory_management")
    memory_management.aimdo_enabled = False
    memory_management.unload_all_models = lambda: None
    model_patcher = _module("comfy.model_patcher")
    model_patcher.LowVramPatch = type("LowVramPatch", (), {})
    model_patcher.ModelPatcher = _ModelPatcher

    ldm = _module("comfy.ldm", package_path=PROJECT_ROOT / ".strict-runtime-ldm")
    ldm_modules = _module(
        "comfy.ldm.modules",
        package_path=PROJECT_ROOT / ".strict-runtime-ldm" / "modules",
    )
    attention = _module("comfy.ldm.modules.attention")

    def attention_pytorch(q: Any, _k: Any, _v: Any, *_args: Any, **_kwargs: Any) -> Any:
        return q

    def attention_comfy_kitchen_int8(
        q: Any,
        _k: Any,
        _v: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        return q

    attention_comfy_kitchen_int8.container_function = object()
    attention.attention_pytorch = attention_pytorch
    attention.attention_comfy_kitchen_int8 = attention_comfy_kitchen_int8
    attention.REGISTERED_ATTENTION_FUNCTIONS = {
        "pytorch": attention_pytorch,
        "comfy_kitchen_int8": attention_comfy_kitchen_int8,
    }
    attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = False
    attention.comfy_kitchen = SimpleNamespace(
        int8_attention_is_available=lambda _device=None: False,
    )
    minimax = _module(
        "comfy.ldm.minimax",
        package_path=PROJECT_ROOT / ".strict-runtime-ldm" / "minimax",
    )
    minimax_model = _module("comfy.ldm.minimax.model")
    minimax_model.Attention = _MiniMaxH3Attention
    minimax_model.DiTBlock = _MiniMaxH3DiTBlock
    minimax_model.MiniMaxH3Model = _MiniMaxH3Model
    ldm.modules = ldm_modules
    ldm_modules.attention = attention
    ldm.minimax = minimax
    minimax.model = minimax_model

    quant_ops = _module(
        "comfy.quant_ops",
        package_path=PROJECT_ROOT / ".strict-runtime-quant-ops",
    )
    ck = _module("comfy.quant_ops.ck")
    ck.rms_rope_split_half = lambda q, k, *_args, **_kwargs: (q, k)
    ck.rms_rope_split_half_ = lambda *_args, **_kwargs: None
    quant_ops.ck = ck

    for name, value in {
        "cli_args": cli_args,
        "samplers": samplers,
        "model_sampling": model_sampling,
        "lora": lora,
        "lora_convert": lora_convert,
        "weight_adapter": weight_adapter,
        "utils": utils,
        "patcher_extension": patcher_extension,
        "sd": sd,
        "sample": sample,
        "model_detection": model_detection,
        "model_management": model_management,
        "memory_management": memory_management,
        "model_patcher": model_patcher,
        "ldm": ldm,
        "quant_ops": quant_ops,
    }.items():
        setattr(comfy, name, value)

    folder_paths = _module("folder_paths")
    folder_paths.get_filename_list = lambda _kind: []
    folder_paths.get_full_path = lambda _kind, name: name
    folder_paths.get_full_path_or_raise = lambda _kind, name: name
    folder_paths.__file__ = str(comfy_host / "folder_paths.py")
    return comfy


def _install_attention_dependency_shims() -> None:
    """Provide imports only; production RayLight owns all attention behavior."""

    xfuser = _module(
        "xfuser",
        package_path=PROJECT_ROOT / ".strict-runtime-xfuser",
    )
    xfuser_core = _module(
        "xfuser.core",
        package_path=PROJECT_ROOT / ".strict-runtime-xfuser" / "core",
    )
    long_context = _module("xfuser.core.long_ctx_attention")
    long_context.xFuserLongContextAttention = type(
        "xFuserLongContextAttention",
        (),
        {},
    )
    xfuser.core = xfuser_core
    xfuser_core.long_ctx_attention = long_context

    yunchang = _module(
        "yunchang",
        package_path=PROJECT_ROOT / ".strict-runtime-yunchang",
    )
    yunchang_comm = _module(
        "yunchang.comm",
        package_path=PROJECT_ROOT / ".strict-runtime-yunchang" / "comm",
    )
    all_to_all = _module("yunchang.comm.all_to_all")

    class SeqAllToAll4D:
        @staticmethod
        def apply(_group: Any, tensor: Any, *_args: Any) -> Any:
            return tensor

    all_to_all.SeqAllToAll4D = SeqAllToAll4D
    yunchang_globals = _module("yunchang.globals")
    yunchang_globals.PROCESS_GROUP = SimpleNamespace(ULYSSES_PG=None)
    kernels = _module("yunchang.kernels")

    class AttnType(Enum):
        TORCH_FLASH = "TORCH_FLASH"

    kernels.AttnType = AttnType
    yunchang.comm = yunchang_comm
    yunchang_comm.all_to_all = all_to_all
    yunchang.globals = yunchang_globals
    yunchang.kernels = kernels

    sage_patch = _module(
        "directordeck_raylight.distributed_modules.sageattention_hf_patch"
    )
    sage_patch.ensure_hf_fp8_cuda_kernel = lambda: None
    sage_patch.ensure_hf_sm90_kernel = lambda: None


def _install_raylight_shims() -> None:
    raylight_root = RAYLIGHT_SOURCE / "directordeck_raylight"
    _, raylight_host = _controlled_host_paths()
    directordeck_raylight = _module(
        "directordeck_raylight", package_path=raylight_root
    )
    directordeck_raylight.__file__ = str(raylight_host / "__init__.py")

    # Pytest derives the collection package name from the on-disk directory
    # ``custom_nodes/raylight/tests``.  The implementation itself deliberately
    # moved to ``directordeck_raylight``; without this inert collection package,
    # pytest executes the custom-node entrypoint merely to import its conftest.
    # Keep ``raylight`` as a test-container namespace only.  Its path does not
    # expose ``src/directordeck_raylight`` and therefore cannot alias or shadow
    # the private maintained implementation namespace.
    raylight_test_package = _module(
        "raylight",
        package_path=PROJECT_ROOT / "custom_nodes" / "raylight",
    )
    raylight_test_package.__file__ = str(
        PROJECT_ROOT / "custom_nodes" / "raylight" / "__init__.py"
    )

    comfy_dist = _module(
        "directordeck_raylight.comfy_dist",
        package_path=raylight_root / "comfy_dist",
    )
    comfy_dist_lora = _module("directordeck_raylight.comfy_dist.lora")
    comfy_dist_lora.load_lora = lambda *_args, **_kwargs: {}
    comfy_dist.lora = comfy_dist_lora
    directordeck_raylight.comfy_dist = comfy_dist

    gpu_visibility = _module("directordeck_raylight.gpu_visibility")
    gpu_visibility.resolve_cuda_visible_devices = (
        lambda selected, parent, visible_device_count=None: parent
    )

    pipefusion = _module("directordeck_raylight.distributed_modules.pipefusion")
    pipefusion.PipeFusionInjectRegistry = type(
        "PipeFusionInjectRegistry", (), {"inject": staticmethod(lambda *_args: None)}
    )
    pipefusion.pipefusion_diffusion_model_wrapper = lambda *_args, **_kwargs: None
    pipefusion.pipefusion_outer_sample_wrapper = lambda *_args, **_kwargs: None
    pipefusion.pipefusion_predict_noise_wrapper = lambda *_args, **_kwargs: None
    usp = _module("directordeck_raylight.distributed_modules.usp")
    usp.USPInjectRegistry = type("USPInjectRegistry", (), {})
    cfg = _module("directordeck_raylight.distributed_modules.cfg")
    cfg.CFGParallelInjectRegistry = type("CFGParallelInjectRegistry", (), {})

    pipefusion_state = _module("directordeck_raylight.distributed_worker.pipefusion_state")
    pipefusion_state.PIPEFUSION_RUNTIME_ATTACHMENT = "pipefusion_runtime"
    pipefusion_state.PIPEFUSION_WRAPPER_KEY = "pipefusion"
    pipefusion_state.PipeFusionRuntime = type("PipeFusionRuntime", (), {})
    parallel_group_manager = _module(
        "directordeck_raylight.distributed_worker.parallel_group_manager"
    )
    parallel_group_manager.initialize_xfuser_parallel = lambda *_args, **_kwargs: None
    parallel_group_manager.requires_xfuser_parallel = lambda *_args, **_kwargs: False
    ray_worker_controlnet = _module(
        "directordeck_raylight.distributed_worker.ray_worker_controlnet"
    )
    ray_worker_controlnet._prepare_control_models = lambda *_args, **_kwargs: None
    ray_worker_controlnet._remap_conditioning_devices = lambda *_args, **_kwargs: None
    ray_worker_controlnet._restore_controlnet_refs = lambda *_args, **_kwargs: None
    ray_worker_vae = _module("directordeck_raylight.distributed_worker.ray_worker_vae")
    ray_worker_vae.combine_dist_vae_partials = lambda values: values
    ray_worker_vae.combine_seedvr2_vae_partials = lambda values: values
    ray_worker_vae.load_vae_model = lambda *_args, **_kwargs: None
    ray_worker_vae.ray_vae_decode_finalize_impl = lambda *_args, **_kwargs: None
    ray_worker_vae.ray_vae_decode_partial_impl = lambda *_args, **_kwargs: None
    ray_worker_vae.ray_seedvr2_vae_decode_partial_impl = lambda *_args, **_kwargs: None
    worker_utils = _module("directordeck_raylight.distributed_worker.utils")
    worker_utils.Noise_EmptyNoise = type("Noise_EmptyNoise", (), {})
    worker_utils.Noise_RandomNoise = type("Noise_RandomNoise", (), {})
    worker_utils.patch_ray_tqdm = lambda *_args, **_kwargs: None

    quant_ops = _module("directordeck_raylight.comfy_dist.quant_ops")
    quant_ops.patch_temp_fix_ck_ops = lambda *_args, **_kwargs: None
    dist_model_management = _module("directordeck_raylight.comfy_dist.model_management")
    dist_model_management.cleanup_models_gc = lambda *_args, **_kwargs: None
    dist_model_patcher = _module("directordeck_raylight.comfy_dist.model_patcher")
    dist_model_patcher.LowVramPatch = type("LowVramPatch", (), {})

    driver_memory = _module("directordeck_raylight.driver_memory")
    driver_memory.DRIVER_CLEANUP_POLICIES = ()
    driver_memory.build_driver_cleanup_metadata = lambda *_args, **_kwargs: {}
    driver_memory.cleanup_driver_models_for_ray = lambda *_args, **_kwargs: None
    driver_memory.clear_ray_worker_vram_after_sampling = lambda *_args, **_kwargs: None

    ray = _module("ray", package_path=PROJECT_ROOT / ".strict-runtime-ray")
    ray.exceptions = SimpleNamespace(RayActorError=RuntimeError)
    ray_exceptions = _module("ray.exceptions")
    ray_exceptions.RayActorError = RuntimeError
    ray.get = lambda value: value
    ray.shutdown = lambda: None
    ray.init = lambda *_args, **_kwargs: None
    ray.remote = lambda value: value
    ray.actor = SimpleNamespace(exit_actor=lambda: None)

    tqdm = _module("tqdm", package_path=PROJECT_ROOT / ".strict-runtime-tqdm")
    tqdm_auto = _module("tqdm.auto")
    tqdm_auto.trange = range
    tqdm.auto = tqdm_auto


def install_controlled_cpu_runtime() -> None:
    """Install the deterministic host surface before pytest imports the nodes."""

    _install_torch_shim()
    _install_comfy_shim()
    _install_attention_dependency_shims()
    _install_raylight_shims()
    sys.path.insert(0, str(RAYLIGHT_SOURCE))


class _ContractCountGate:
    def __init__(self) -> None:
        self.collected = 0
        self.passed = 0

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = len(session.items)

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "call" and report.passed:
            self.passed += 1


def main() -> int:
    missing = [
        str(selection)
        for selection in STRICT_TESTS
        if not Path(str(selection).partition("::")[0]).is_file()
    ]
    if missing:
        print("Missing strict runtime contract tests:", *missing, sep="\n- ")
        return 2

    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    install_controlled_cpu_runtime()
    pytest_config = Path(_HOST_TEMP.name) / "pytest.ini" if _HOST_TEMP else None

    import pytest

    gate = _ContractCountGate()
    exit_code = int(
        pytest.main(
            [
                "-q",
                "-c",
                str(pytest_config),
                "--rootdir",
                str(PROJECT_ROOT),
                "--import-mode=importlib",
                *(str(path) for path in STRICT_TESTS),
            ],
            plugins=[gate],
        )
    )
    if exit_code == 0 and gate.passed >= MINIMUM_CONTRACT_TESTS:
        print(
            "Strict runtime contract gate passed: "
            f"{gate.passed}/{gate.collected} tests"
        )
        return 0
    if exit_code == 0:
        print(
            "Strict runtime contract collection shrank below "
            f"{MINIMUM_CONTRACT_TESTS}: collected {gate.collected}, "
            f"passed {gate.passed}"
        )
    return exit_code or 1


if __name__ == "__main__":
    raise SystemExit(main())
