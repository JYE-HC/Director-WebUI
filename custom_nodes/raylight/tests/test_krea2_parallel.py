import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).parents[3]
RAYLIGHT = ROOT / "custom_nodes/raylight/src/raylight"


def _function(path, function_name, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name is not None:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in nodes if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == function_name)


def _signature(function):
    args = function.args
    parameters = []

    def add(kind, argument, default=None):
        parameters.append((kind, argument.arg, ast.dump(default, include_attributes=False) if default else None))

    for argument in args.posonlyargs:
        add("positional-only", argument)
    positional = args.args
    positional_defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for argument, default in zip(positional, positional_defaults):
        add("positional", argument, default)
    if args.vararg:
        add("var-positional", args.vararg)
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        add("keyword-only", argument, default)
    if args.kwarg:
        add("var-keyword", args.kwarg)
    return parameters


def _assert_matching_signatures(core_path, core_name, raylight_path, raylight_name, core_class=None):
    core = _signature(_function(core_path, core_name, core_class))
    raylight = _signature(_function(raylight_path, raylight_name))
    assert raylight == core


def _source_defines(path, name, seen=None):
    seen = set() if seen is None else seen
    if path in seen:
        return False
    seen.add(path)
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return True
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 2:
            continue
        module_parts = node.module.split(".")
        imported_names = {item.name for item in node.names}
        if name not in imported_names:
            continue
        imported_path = RAYLIGHT.joinpath(*module_parts).with_suffix(".py")
        if _source_defines(imported_path, name, seen):
            return True
    return False


def test_krea2_forward_signature_matches_usp_forward():
    _assert_matching_signatures(
        ROOT / "comfy/ldm/krea2/model.py",
        "_forward",
        RAYLIGHT / "diffusion_models/krea2/xdit_context_parallel.py",
        "usp_dit_forward",
        "SingleStreamDiT",
    )


def test_hunyuan3d_forward_signature_matches_usp_forward():
    _assert_matching_signatures(
        ROOT / "comfy/ldm/hunyuan3d/model.py",
        "_forward",
        RAYLIGHT / "diffusion_models/hunyuan3d/xdit_context_parallel.py",
        "usp_dit_forward",
        "Hunyuan3Dv2",
    )


def test_chroma_single_stream_signature_matches_flux_block():
    _assert_matching_signatures(
        ROOT / "comfy/ldm/flux/layers.py",
        "forward",
        RAYLIGHT / "diffusion_models/chroma/xdit_context_parallel.py",
        "usp_single_stream_forward",
        "SingleStreamBlock",
    )


def test_chroma_double_stream_signature_matches_flux_block():
    _assert_matching_signatures(
        ROOT / "comfy/ldm/flux/layers.py",
        "forward",
        RAYLIGHT / "diffusion_models/chroma/xdit_context_parallel.py",
        "usp_double_stream_forward",
        "DoubleStreamBlock",
    )


def test_scail_forward_signature_matches_usp_forward():
    _assert_matching_signatures(
        ROOT / "comfy/ldm/wan/model.py",
        "forward_orig",
        RAYLIGHT / "diffusion_models/wan/xdit_context_parallel.py",
        "usp_scail_dit_forward",
        "SCAILWanModel",
    )


def test_boogu_injector_imports_exist_in_source_files():
    usp_path = RAYLIGHT / "distributed_modules/usp.py"
    tree = ast.parse(usp_path.read_text())
    injector = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_inject_boogu")
    imports = [node for node in ast.walk(injector) if isinstance(node, ast.ImportFrom)]

    assert imports
    for import_node in imports:
        assert import_node.level == 2
        module_parts = import_node.module.split(".")
        source_path = RAYLIGHT.joinpath(*module_parts).with_suffix(".py")
        for imported in import_node.names:
            assert _source_defines(source_path, imported.name), f"{imported.name} is missing from {source_path}"


def _load_cfg_utils(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    nested_tensor = types.ModuleType("comfy.nested_tensor")

    class NestedTensor:
        pass

    nested_tensor.NestedTensor = NestedTensor

    xfuser = types.ModuleType("xfuser")
    xfuser.__path__ = []
    xfuser_core = types.ModuleType("xfuser.core")
    xfuser_core.__path__ = []
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_classifier_free_guidance_rank = lambda: 1
    distributed.get_classifier_free_guidance_world_size = lambda: 2

    class Group:
        @staticmethod
        def all_gather(value, dim):
            return value

    distributed.get_cfg_group = Group

    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested_tensor)
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", xfuser_core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)

    spec = importlib.util.spec_from_file_location("cfg_utils_test", RAYLIGHT / "distributed_modules/cfg_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cfg_parallel_chunks_new_batch_kwargs(monkeypatch):
    cfg_utils = _load_cfg_utils(monkeypatch)

    def original(x, **kwargs):
        return x

    class Executor:
        def __init__(self):
            self.original = original
            self.args = None
            self.kwargs = None

        def __call__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return self.original(*args, **kwargs)

    executor = Executor()
    cfg_utils.cfg_parallel_forward(
        executor,
        torch.tensor([[1], [2]]),
        chunk_names=("x",),
        control={"input": torch.tensor([[3], [4]])},
        new_conditioning=torch.tensor([[5], [6]]),
        transformer_options={"static": torch.tensor([[7], [8]])},
    )

    assert torch.equal(executor.args[0], torch.tensor([[2]]))
    assert torch.equal(executor.kwargs["control"]["input"], torch.tensor([[4]]))
    assert torch.equal(executor.kwargs["new_conditioning"], torch.tensor([[6]]))
    assert torch.equal(executor.kwargs["transformer_options"]["static"], torch.tensor([[7], [8]]))
