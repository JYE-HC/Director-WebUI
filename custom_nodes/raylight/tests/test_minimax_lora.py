# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
from types import SimpleNamespace

from directordeck_raylight.diffusion_models.minimax.lora import (
    is_minimax_h3_fused_int8_fc2,
    minimax_h3_lora_targets,
    normalize_minimax_h3_lora_keys,
)


class _FakeReduction:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def any(self):
        return self

    def item(self):
        return self.value


class FakeTensor:
    def __init__(self, shape, layout=None, transposed=False, *, finite=True, nonzero=True, scalar=1.0):
        self.shape = shape
        self.finite = finite
        self.nonzero = nonzero
        self.scalar = scalar
        if layout is not None:
            self._layout_cls = layout
            self._params = SimpleNamespace(transposed=transposed)

    def item(self):
        return self.scalar

    def isfinite(self):
        return _FakeReduction(self.finite)

    def ne(self, value):
        assert value == 0
        return _FakeReduction(self.nonzero)


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
        "final_layer.adaln_proj.linear.lora_A.weight": FakeTensor((2, 8)),
        "final_layer.adaln_proj.linear.lora_B.weight": FakeTensor((24, 2)),
        "token_refiner.blocks.0.mlp.fc1.lora_A.weight": FakeTensor((2, 8)),
        "token_refiner.blocks.0.mlp.fc1.lora_B.weight": FakeTensor((16, 2)),
    }

    normalized, count = normalize_minimax_h3_lora_keys(_patcher(MiniMaxH3Model()), state_dict)

    assert count == 7
    assert set(normalized) == {
        "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
        "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight",
        "diffusion_model.blocks.0.attn.qkv_proj.alpha",
        "diffusion_model.final_layer.adaln_proj.linear.lora_A.weight",
        "diffusion_model.final_layer.adaln_proj.linear.lora_B.weight",
        "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_A.weight",
        "diffusion_model.token_refiner.blocks.0.mlp.fc1.lora_B.weight",
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


def test_rejects_h3_keys_for_non_h3_model():
    state_dict = {"blocks.0.attn.qkv_proj.lora_A.weight": object()}

    _assert_raises(
        TypeError,
        "require a MiniMaxH3Model",
        lambda: normalize_minimax_h3_lora_keys(_patcher(OtherModel()), state_dict),
    )


def test_leaves_non_h3_model_and_non_h3_keys_unchanged():
    state_dict = {"other_model.layer.lora_A.weight": object()}

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

    assert minimax_h3_lora_targets(_patcher(MiniMaxH3Model()), normalized) == (
        "diffusion_model.blocks.0.attn.qkv_proj.weight",
    )


def test_rejects_h3_lora_without_adapter_pairs():
    _assert_raises(
        ValueError,
        "no adapter A/B pairs",
        lambda: normalize_minimax_h3_lora_keys(
            _patcher(MiniMaxH3Model()),
            {"blocks.0.attn.qkv_proj.alpha": FakeTensor(())},
        ),
    )


def test_rejects_every_unsupported_pending_key():
    base = {
        "blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 8)),
        "blocks.0.attn.qkv_proj.lora_B.weight": FakeTensor((12, 2)),
    }

    _assert_raises(
        ValueError,
        "Unsupported MiniMax H3 LoRA adapter key",
        lambda: normalize_minimax_h3_lora_keys(
            _patcher(MiniMaxH3Model()),
            {**base, "blocks.0.attn.qkv_proj.dora_scale": FakeTensor((12,))},
        ),
    )
    _assert_raises(
        ValueError,
        "Unsupported key in MiniMax H3 LoRA state dict",
        lambda: normalize_minimax_h3_lora_keys(
            _patcher(MiniMaxH3Model()), {**base, "metadata": FakeTensor(())}
        ),
    )


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


def test_rejects_non_finite_or_zero_alpha():
    base = {
        "blocks.0.attn.qkv_proj.lora_A.weight": FakeTensor((2, 8)),
        "blocks.0.attn.qkv_proj.lora_B.weight": FakeTensor((12, 2)),
    }

    for alpha in (0.0, -0.0, float("nan"), float("inf"), float("-inf")):
        _assert_raises(
            ValueError,
            "alpha must be a finite non-zero scalar",
            lambda alpha=alpha: normalize_minimax_h3_lora_keys(
                _patcher(MiniMaxH3Model()),
                {**base, "blocks.0.attn.qkv_proj.alpha": FakeTensor((), scalar=alpha)},
            ),
        )


def test_rejects_non_finite_or_all_zero_adapter_tensors():
    for suffix, tensor in (
        ("lora_A.weight", FakeTensor((2, 8), finite=False)),
        ("lora_B.weight", FakeTensor((12, 2), finite=False)),
        ("lora_A.weight", FakeTensor((2, 8), nonzero=False)),
        ("lora_B.weight", FakeTensor((12, 2), nonzero=False)),
    ):
        a = tensor if suffix.startswith("lora_A") else FakeTensor((2, 8))
        b = tensor if suffix.startswith("lora_B") else FakeTensor((12, 2))
        message = "only finite values" if not tensor.finite else "must not be all zero"
        _assert_raises(
            ValueError,
            message,
            lambda a=a, b=b: normalize_minimax_h3_lora_keys(
                _patcher(MiniMaxH3Model()),
                {
                    "blocks.0.attn.qkv_proj.lora_A.weight": a,
                    "blocks.0.attn.qkv_proj.lora_B.weight": b,
                },
            ),
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
