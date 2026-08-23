from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


NODE_PATH = Path(__file__).parents[1] / "__init__.py"
SPEC = importlib.util.spec_from_file_location(
    "directordeck_strict_h3_cpu_contract_node",
    NODE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
NODE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NODE
SPEC.loader.exec_module(NODE)


def test_cpu_gate_wire_contract_is_exact() -> None:
    assert NODE.NODE_CLASS_MAPPINGS == {
        "DirectorStrictH3LowVramSagePatch": (
            NODE.DirectorStrictH3LowVramSagePatch
        ),
    }
    assert NODE.DirectorStrictH3LowVramSagePatch.INPUT_TYPES() == {
        "required": {"model": ("MODEL",)},
    }
    assert NODE.DirectorStrictH3LowVramSagePatch.RETURN_TYPES == ("MODEL",)


def test_cpu_gate_never_reports_unproved_gpu_runtime_available() -> None:
    capability = NODE.director_runtime_capability()
    assert capability["available"] is False
    assert capability["code"] != "available"
    assert capability["architecture"] is None


def test_cpu_gate_unavailable_runtime_fails_before_model_clone() -> None:
    class Model:
        clone_count = 0

        def clone(self):
            self.clone_count += 1
            return self

    model = Model()
    with pytest.raises(RuntimeError, match="is unavailable"):
        NODE.DirectorStrictH3LowVramSagePatch().apply_strict_h3_sage(model)
    assert model.clone_count == 0
