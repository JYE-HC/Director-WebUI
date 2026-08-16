# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
from types import SimpleNamespace

from raylight.diffusion_models.minimax.lora import (
    is_minimax_h3_fused_int8_fc2,
    normalize_minimax_h3_lora_keys,
)


class FakeTensor:
    def __init__(self, shape, layout=None, transposed=False):
        self.shape = shape
        if layout is not None:
            self._layout_cls = layout
            self._params = SimpleNamespace(transposed=transposed)


class MiniMaxH3Model:
    def __init__(self, use_adaln_curves=False):
        self.use_adaln_curves = use_adaln_curves
        self.blocks = [
            SimpleNamespace(
                attn=SimpleNamespace(qkv_proj=SimpleNamespace(weight=FakeTensor((12, 8)))),
                mlp=SimpleNamespace(fc2=SimpleNamespace(weight=FakeTensor((8, 16)))),
            )
        ]
        self.final_layer = SimpleNamespace(
            adaln_proj=SimpleNamespace(linear=SimpleNamespace(weight=FakeTensor((24, 8))))
        )
        self.token_refiner = SimpleNamespace(
            blocks=[SimpleNamespace(mlp=SimpleNamespace(fc1=SimpleNamespace(weight=FakeTensor((16, 8)))))]
        )


class OtherModel:
    pass


def _patcher(diffusion_model):
    return SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion_model))


def _assert_raises(exception_type, message, function):
    try:
        function()
    except exception_type as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected {exception_type.__name__}: {message}")


def test_normalizes_all_official_h3_turbo_lora_roots_without_mutating_input():
    state_dict = {
        "blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 8)),
        "blocks.0.attn.qkv_proj.lora_B.weight": FakeTensor((12, 2)),
        "blocks.0.attn.qkv_proj.alpha": FakeTensor(()),
        "blocks.0.attn.qkv_proj.dora_scale": FakeTensor((12,)),
        "final_layer.adaln_proj.linear.lora_A.weight": FakeTensor((2, 8)),
        "final_layer.adaln_proj.linear.lora_B.weight": FakeTensor((24, 2)),
        "token_refiner.blocks.0.mlp.fc1.lora_A.weight": FakeTensor((2, 8)),
        "token_refiner.blocks.0.mlp.fc1.lora_B.weight": FakeTensor((16, 2)),
        "metadata": object(),
    }

    normalized, count = normalize_minimax_h3_lora_keys(_patcher(MiniMaxH3Model()), state_dict)

    assert count == 8
    assert set(normalized) == {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight",
        "diffusion_model.blocks.0.attn.qkv_proj.alpha",
        "diffusion_model.blocks.0.attn.qkv_proj.dora_scale",
        "diffusion_model.final_layer.adaln_proj.linear.lora_A.weight",
        "diffusion_model.final_layer.adaln_proj.linear.lora_B.weight",
        "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight",
        "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_B.weight",
        "metadata",
    }
    assert all(not key.startswith("diffusion_model.") for key in state_dict)

    normalized_again, second_count = normalize_minimax_h3_lora_keys(
        _patcher(MiniMaxH3Model()), normalized
    )
    assert normalized_again is normalized
    assert second_count == 0


def test_duplicate_bare_and_prefixed_key_is_rejected():
    state_dict = {
        "blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 8)),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 8)),
    }

    _assert_raises(
        ValueError,
        "both bare and diffusion_model-prefixed",
        lambda: normalize_minimax_h3_lora_keys(_patcher(MiniMaxH3Model()), state_dict),
    )


def test_leaves_non_h3_models_unchanged():
    state_dict = {"blocks.0.attn.qkv_proj.lora_A.weight": object()}

    normalized, count = normalize_minimax_h3_lora_keys(_patcher(OtherModel()), state_dict)

    assert normalized is state_dict
    assert count == 0


def test_validates_already_normalized_h3_state_dict():
    state_dict = {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 8)),
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": FakeTensor((12, 2)),
    }

    normalized, count = normalize_minimax_h3_lora_keys(_patcher(MiniMaxH3Model()), state_dict)

    assert normalized is state_dict
    assert count == 0


def test_rejects_incomplete_pair():
    state_dict = {"blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 8))}

    _assert_raises(
        ValueError,
        "missing B",
        lambda: normalize_minimax_h3_lora_keys(_patcher(MiniMaxH3Model()), state_dict),
    )


def test_rejects_rank_mismatch():
    state_dict = {
        "blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 8)),
        "blocks.0.attn.qkv_proj.lora_B.weight": FakeTensor((12, 3)),
    }

    _assert_raises(
        ValueError,
        "Invalid MiniMax H3 LoRA rank",
        lambda: normalize_minimax_h3_lora_keys(_patcher(MiniMaxH3Model()), state_dict),
    )


def test_rejects_target_shape_mismatch():
    state_dict = {
        "blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 7)),
        "blocks.0.attn.qkv_proj.lora_B.weight": FakeTensor((12, 2)),
    }

    _assert_raises(
        ValueError,
        "shape mismatch",
        lambda: normalize_minimax_h3_lora_keys(_patcher(MiniMaxH3Model()), state_dict),
    )


def test_rejects_missing_target():
    state_dict = {
        "blocks.0.missing.lora_A.weight": FakeTensor((2, 8)),
        "blocks.0.missing.lora_B.weight": FakeTensor((12, 2)),
    }

    _assert_raises(
        ValueError,
        "target does not exist",
        lambda: normalize_minimax_h3_lora_keys(_patcher(MiniMaxH3Model()), state_dict),
    )


def test_pruned_adaln_lora_fails_before_sampling():
    state_dict = {
        "final_layer.adaln_proj.linear.lora_A.weight": FakeTensor((2, 8)),
        "final_layer.adaln_proj.linear.lora_B.weight": FakeTensor((24, 2)),
    }

    _assert_raises(
        RuntimeError,
        "pruned/curve",
        lambda: normalize_minimax_h3_lora_keys(
            _patcher(MiniMaxH3Model(use_adaln_curves=True)), state_dict
        ),
    )

    prefixed = {f"diffusion_model.{key}": value for key, value in state_dict.items()}
    _assert_raises(
        RuntimeError,
        "pruned/curve",
        lambda: normalize_minimax_h3_lora_keys(
            _patcher(MiniMaxH3Model(use_adaln_curves=True)), prefixed
        ),
    )


def test_detects_only_non_transposed_tensorwise_int8_fc2():
    model = MiniMaxH3Model()
    patcher = _patcher(model)

    model.blocks[0].mlp.fc2.weight = FakeTensor((8, 16), layout="TensorWiseINT8Layout")
    assert is_minimax_h3_fused_int8_fc2(patcher, "diffusion_model.blocks.0.mlp.fc2")

    model.blocks[0].mlp.fc2.weight = FakeTensor(
        (8, 16), layout="TensorWiseINT8Layout", transposed=True
    )
    assert not is_minimax_h3_fused_int8_fc2(patcher, "diffusion_model.blocks.0.mlp.fc2")

    model.blocks[0].mlp.fc2.weight = FakeTensor((8, 16), layout="TensorCoreFP8Layout")
    assert not is_minimax_h3_fused_int8_fc2(patcher, "diffusion_model.blocks.0.mlp.fc2")
    assert not is_minimax_h3_fused_int8_fc2(patcher, "diffusion_model.blocks.0.attn.qkv_proj")
    assert not is_minimax_h3_fused_int8_fc2(
        _patcher(OtherModel()), "diffusion_model.blocks.0.mlp.fc2"
    )
