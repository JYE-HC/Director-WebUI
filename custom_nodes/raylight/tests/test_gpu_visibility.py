# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
import ast
from pathlib import Path

from raylight.gpu_visibility import resolve_cuda_visible_devices


def _assert_raises(exception_type, message, function):
    try:
        function()
    except exception_type as exc:
        assert message in str(exc)
    else:
        raise AssertionError(f"Expected {exception_type.__name__}: {message}")


def test_maps_comfy_logical_indices_through_parent_mask():
    assert resolve_cuda_visible_devices((0, 1), "7,6", visible_device_count=2) == "7,6"
    assert resolve_cuda_visible_devices((1,), "7,6") == "6"
    assert resolve_cuda_visible_devices((1, 0), "7,6") == "6,7"


def test_uses_physical_indices_when_parent_has_no_mask():
    assert resolve_cuda_visible_devices((6, 7), None) == "6,7"


def test_preserves_gpu_and_mig_uuid_tokens():
    assert resolve_cuda_visible_devices((1, 0), "GPU-aaaa,GPU-bbbb") == "GPU-bbbb,GPU-aaaa"
    assert resolve_cuda_visible_devices((0,), "MIG-aaaa/MIG-1/0,MIG-bbbb/MIG-2/0") == "MIG-aaaa/MIG-1/0"


def test_no_selection_preserves_parent_visibility():
    assert resolve_cuda_visible_devices(None, "7,6") == "7,6"
    assert resolve_cuda_visible_devices((), "7,6") == "7,6"
    assert resolve_cuda_visible_devices(None, None) is None


def test_rejects_invalid_logical_selection_without_parent_mask():
    _assert_raises(
        ValueError,
        "zero-based GPU indices",
        lambda: resolve_cuda_visible_devices((-1,), None),
    )
    _assert_raises(
        ValueError,
        "duplicate GPU indices",
        lambda: resolve_cuda_visible_devices((0, 0), None),
    )


def test_rejects_selection_when_parent_exposes_no_cuda_devices():
    for hidden in ("", "-1", "NoDevFiles"):
        _assert_raises(
            ValueError,
            "parent CUDA_VISIBLE_DEVICES exposes none",
            lambda hidden=hidden: resolve_cuda_visible_devices((0,), hidden),
        )


def test_rejects_indices_outside_parent_visibility():
    _assert_raises(
        ValueError,
        "outside the parent CUDA_VISIBLE_DEVICES range 0-1",
        lambda: resolve_cuda_visible_devices((2,), "7,6"),
    )


def test_rejects_malformed_parent_visibility():
    _assert_raises(
        ValueError,
        "malformed",
        lambda: resolve_cuda_visible_devices((0,), "7,,6"),
    )


def test_rejects_duplicate_parent_devices():
    for parent in ("7,7", "07,7", "GPU-AAAA,gpu-aaaa"):
        _assert_raises(
            ValueError,
            "duplicate devices",
            lambda parent=parent: resolve_cuda_visible_devices((0,), parent),
        )


def test_rejects_parent_mask_count_that_disagrees_with_torch():
    _assert_raises(
        ValueError,
        "torch reports 2 visible CUDA devices",
        lambda: resolve_cuda_visible_devices((0, 1), "7,6,5", visible_device_count=2),
    )


def test_ray_actors_do_not_override_ray_assigned_cuda_visibility():
    worker_path = Path(__file__).parents[1] / "src/raylight/distributed_worker/ray_worker.py"
    tree = ast.parse(worker_path.read_text())
    assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not (
                isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "os"
                and target.value.attr == "environ"
            ):
                continue
            if isinstance(target.slice, ast.Constant) and target.slice.value == "CUDA_VISIBLE_DEVICES":
                assignments.append(node.lineno)

    assert not assignments, f"Ray actors override CUDA_VISIBLE_DEVICES at lines {assignments}"
