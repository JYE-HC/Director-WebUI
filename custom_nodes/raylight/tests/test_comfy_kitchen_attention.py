# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
import ast
from pathlib import Path

import pytest
import torch
from yunchang.kernels import AttnType

import raylight.distributed_modules.attention as attention
from raylight.distributed_modules.attention_backends import (
    COMFY_KITCHEN_INT8,
    attention_backend_choices,
    validate_attention_backend_config,
)


ROOT = Path(__file__).parents[1]
ATTENTION = ROOT / "src/raylight/distributed_modules/attention.py"
NODES = ROOT / "src/raylight/nodes.py"
NODES_DEBUG = ROOT / "src/raylight/nodes_debug.py"
PARALLEL_GROUP_MANAGER = ROOT / "src/raylight/distributed_worker/parallel_group_manager.py"


def _xfuser_dropdown_entries(path):
    entries = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value == "XFuser_attention"):
                continue
            assert isinstance(value, ast.Tuple)
            choices, config = value.elts
            assert isinstance(choices, ast.Call)
            assert isinstance(choices.func, ast.Name)
            assert choices.func.id == "attention_backend_choices"
            assert isinstance(config, ast.Dict)
            default = next(
                config_value.value
                for config_key, config_value in zip(config.keys, config.values)
                if isinstance(config_key, ast.Constant) and config_key.value == "default"
            )
            entries.append(default)
    return entries


def _call_lines(path, function_name, class_name=None):
    body = ast.parse(path.read_text()).body
    if class_name is not None:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    calls = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        calls.setdefault(name, []).append(node.lineno)
    return calls


def _install_fake_ck(monkeypatch, kernel, available=True):
    availability_calls = []

    def is_available(device):
        availability_calls.append(device)
        return available

    monkeypatch.setattr(attention, "_load_comfy_kitchen_int8_attention", lambda: (kernel, is_available))
    monkeypatch.setattr(attention, "_current_worker_cuda_device", lambda: (torch.device("cuda", 0), (8, 6)))
    monkeypatch.setattr(attention.PROCESS_GROUP, "ULYSSES_PG", object())
    return availability_calls


def test_choices_append_ck_without_changing_existing_enum_order():
    choices = attention_backend_choices()
    assert choices[:-1] == [member.name for member in AttnType]
    assert choices[-1] == COMFY_KITCHEN_INT8
    assert len(choices) == len(set(choices))


def test_all_initializer_dropdowns_share_choices_and_keep_legacy_default():
    assert _xfuser_dropdown_entries(NODES) == ["TORCH_FLASH", "TORCH_FLASH"]
    assert _xfuser_dropdown_entries(NODES_DEBUG) == ["TORCH_FLASH"]


def test_comfy_kitchen_import_is_lazy():
    tree = ast.parse(ATTENTION.read_text())
    module_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "comfy_kitchen" not in module_imports
    lazy_loader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_comfy_kitchen_int8_attention"
    )
    assert any(
        isinstance(node, ast.ImportFrom) and node.module == "comfy_kitchen"
        for node in ast.walk(lazy_loader)
    )


def test_legacy_backends_never_load_ck_and_keep_their_enum_mapping(monkeypatch):
    created = []

    class FakeLongContextAttention:
        def __init__(self, **kwargs):
            created.append(kwargs)

    def unexpected_ck_load():
        raise AssertionError("legacy backend attempted to import comfy-kitchen")

    monkeypatch.setattr(attention, "xFuserLongContextAttention", FakeLongContextAttention)
    monkeypatch.setattr(attention, "ensure_hf_fp8_cuda_kernel", lambda: None)
    monkeypatch.setattr(attention, "ensure_hf_sm90_kernel", lambda: None)
    monkeypatch.setattr(attention, "_load_comfy_kitchen_int8_attention", unexpected_ck_load)

    for member in AttnType:
        wrapper = attention.make_xfuser_attention(member.name, False, ring_degree=1)
        assert callable(wrapper)

    assert [item["attn_type"] for item in created] == list(AttnType)


def test_ck_rejects_ring_before_loading_kernel(monkeypatch):
    loaded = []
    monkeypatch.setattr(attention, "_load_comfy_kitchen_int8_attention", lambda: loaded.append(True))

    with pytest.raises(ValueError, match="ring_degree must be exactly 1"):
        attention.make_xfuser_attention(COMFY_KITCHEN_INT8, False, ring_degree=2)

    assert not loaded


def test_ck_unavailable_on_worker_fails_without_fallback(monkeypatch):
    availability_calls = _install_fake_ck(monkeypatch, lambda q, k, v, scale=None: q, available=False)

    with pytest.raises(RuntimeError, match=r"unavailable on Ray worker device cuda:0 \(SM86\).*no fallback"):
        attention.make_xfuser_attention(COMFY_KITCHEN_INT8, False, ring_degree=1)

    assert availability_calls == [torch.device("cuda", 0)]


def test_ck_adapter_converts_nhd_bhsd_passes_scale_and_uses_ulysses(monkeypatch):
    kernel_calls = []
    all_to_all_calls = []

    def kernel(q, k, v, *, scale=None):
        kernel_calls.append((q.clone(), k.clone(), v.clone(), scale, q.is_contiguous(), k.is_contiguous(), v.is_contiguous()))
        return q + 1

    availability_calls = _install_fake_ck(monkeypatch, kernel)

    def fake_all_to_all(group, tensor, scatter_idx, gather_idx, use_sync):
        all_to_all_calls.append((group, tuple(tensor.shape), scatter_idx, gather_idx, use_sync))
        return tensor

    monkeypatch.setattr(attention.SeqAllToAll4D, "apply", staticmethod(fake_all_to_all))
    wrapper = attention.make_xfuser_attention(COMFY_KITCHEN_INT8, True, ring_degree=1)

    q = torch.arange(24, dtype=torch.float32).reshape(1, 4, 3, 2)
    output = wrapper(
        q,
        q + 100,
        q + 200,
        heads=4,
        skip_reshape=True,
        skip_output_reshape=True,
        scale=0.125,
        transformer_options={"block_index": 3},
    )

    assert availability_calls == [torch.device("cuda", 0)]
    assert len(kernel_calls) == 1
    kernel_q, kernel_k, kernel_v, scale, *contiguous = kernel_calls[0]
    assert kernel_q.shape == kernel_k.shape == kernel_v.shape == (1, 4, 3, 2)
    assert torch.equal(kernel_q, q)
    assert scale == 0.125
    assert all(contiguous)
    assert torch.equal(output, q + 1)
    assert [(call[2], call[3]) for call in all_to_all_calls] == [(2, 1), (2, 1), (2, 1), (1, 2)]
    assert all(call[4] is True for call in all_to_all_calls)


@pytest.mark.parametrize(
    "invalid_output",
    [
        lambda q: q[:, :, :-1, :],
        lambda q: q.to(torch.float64),
        lambda q: q.to("meta"),
    ],
)
def test_ck_rejects_wrong_kernel_output_contract_before_reverse_ulysses(monkeypatch, invalid_output):
    all_to_all_calls = []

    def kernel(q, k, v, *, scale=None):
        return invalid_output(q)

    _install_fake_ck(monkeypatch, kernel)

    def fake_all_to_all(group, tensor, scatter_idx, gather_idx, use_sync):
        all_to_all_calls.append((scatter_idx, gather_idx))
        return tensor

    monkeypatch.setattr(attention.SeqAllToAll4D, "apply", staticmethod(fake_all_to_all))
    wrapper = attention.make_xfuser_attention(COMFY_KITCHEN_INT8, False, ring_degree=1)
    q = torch.ones(1, 4, 3, 2)

    with pytest.raises(RuntimeError, match="invalid BHSD output contract.*no fallback"):
        wrapper(q, q, q, heads=4, skip_reshape=True)

    assert all_to_all_calls == [(2, 1), (2, 1), (2, 1)]


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("mask", torch.ones(1)),
        ("causal", True),
        ("dropout_p", 0.1),
        ("window_size", (0, 8)),
        ("alibi_slopes", torch.ones(1)),
        ("join_q", torch.ones(1, 2, 3, 4)),
    ],
)
def test_ck_rejects_unsupported_attention_contracts(monkeypatch, argument, value):
    kernel_calls = []
    _install_fake_ck(monkeypatch, lambda q, k, v, scale=None: kernel_calls.append(True))
    wrapper = attention.make_xfuser_attention(COMFY_KITCHEN_INT8, False, ring_degree=1)
    q = torch.ones(1, 4, 3, 2)

    with pytest.raises(ValueError, match=f"does not support {argument}"):
        wrapper(q, q, q, heads=4, skip_reshape=True, **{argument: value})

    assert not kernel_calls


def test_ck_config_validation_happens_before_ray_or_xfuser_distributed_init():
    validate_attention_backend_config(COMFY_KITCHEN_INT8, 1)
    with pytest.raises(ValueError, match="ring_degree must be exactly 1"):
        validate_attention_backend_config(COMFY_KITCHEN_INT8, 2)

    node_calls = _call_lines(NODES, "spawn_actor", "RayInitializer")
    debug_calls = _call_lines(NODES_DEBUG, "spawn_actor", "RayInitializerDebug")
    worker_calls = _call_lines(PARALLEL_GROUP_MANAGER, "initialize_xfuser_parallel")
    assert node_calls["validate_attention_backend_config"][0] < min(node_calls["init"])
    assert debug_calls["validate_attention_backend_config"][0] < min(debug_calls["init"])
    assert worker_calls["validate_attention_backend_config"][0] < worker_calls["init_distributed_environment"][0]
