# Modified for Director Web; see DIRECTOR_MODIFICATIONS.md.
import ast
from pathlib import Path


ROOT = Path(__file__).parents[3]
CORE = ROOT / "comfy/ldm/minimax/model.py"
RAYLIGHT = ROOT / "custom_nodes/raylight/src/directordeck_raylight/diffusion_models/minimax/xdit_context_parallel.py"


def _module_symbols(path):
    symbols = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            symbols.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            symbols.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return symbols


def _function(path, function_name, class_name=None):
    nodes = ast.parse(path.read_text()).body
    if class_name is not None:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == function_name)


def _signature(function):
    args = function.args
    parameters = []

    def add(kind, argument, default=None):
        parameters.append((kind, argument.arg, ast.dump(default, include_attributes=False) if default is not None else None))

    for argument in args.posonlyargs:
        add("positional-only", argument)
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)
    for argument, default in zip(args.args, defaults):
        add("positional", argument, default)
    if args.vararg is not None:
        add("var-positional", args.vararg)
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        add("keyword-only", argument, default)
    if args.kwarg is not None:
        add("var-keyword", args.kwarg)
    return parameters


def _assigned_dict_keys(function, variable_name):
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != variable_name:
            continue
        assert isinstance(node.value, ast.Dict)
        return {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise AssertionError(f"missing dict assignment for {variable_name}")


def _packed_layout_segment_kinds():
    constructor = _function(CORE, "__init__", "PackedLayout")
    kinds = set()
    for node in ast.walk(constructor):
        tuples = []
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "segments" for target in node.targets)
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            tuples.extend(node.value.elts)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "segments"
        ):
            tuples.extend(node.args)
        for item in tuples:
            if (
                isinstance(item, ast.Tuple)
                and item.elts
                and isinstance(item.elts[0], ast.Constant)
                and isinstance(item.elts[0].value, str)
            ):
                kinds.add(item.elts[0].value)
    return kinds


def test_minimax_core_imports_exist():
    imports = next(
        node.names
        for node in ast.parse(RAYLIGHT.read_text()).body
        if isinstance(node, ast.ImportFrom) and node.module == "comfy.ldm.minimax.model"
    )
    missing = {alias.name for alias in imports} - _module_symbols(CORE)
    assert not missing


def test_minimax_usp_forward_signature_matches_core():
    assert _signature(_function(RAYLIGHT, "usp_dit_forward")) == _signature(
        _function(CORE, "_forward", "MiniMaxH3Model")
    )


def test_minimax_packed_layout_frame_count_is_version_gated():
    forward = _function(RAYLIGHT, "usp_dit_forward")
    source = ast.unparse(forward)

    assert "'frame_count' in inspect.signature(PackedLayout).parameters" in source
    packed_layout_calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PackedLayout"
    ]
    assert len(packed_layout_calls) == 1
    call = packed_layout_calls[0]
    assert not any(keyword.arg == "frame_count" for keyword in call.keywords)
    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "layout_kwargs"
        for keyword in call.keywords
    )


def test_minimax_segment_maps_cover_core_packed_layout_kinds():
    forward = _function(RAYLIGHT, "usp_dit_forward")
    core_kinds = _packed_layout_segment_kinds()

    assert core_kinds <= _assigned_dict_keys(forward, "seg_t")
    assert core_kinds <= _assigned_dict_keys(forward, "seg_tag")
    assert {"cond_audio", "ref_audio"} <= core_kinds
