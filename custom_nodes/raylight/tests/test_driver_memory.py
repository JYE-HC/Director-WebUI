# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
import ast
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load_driver_memory():
    fake_comfy = types.ModuleType("comfy")
    fake_comfy.__path__ = []
    fake_model_management = types.ModuleType("comfy.model_management")
    fake_comfy.model_management = fake_model_management
    previous_comfy = sys.modules.get("comfy")
    previous_model_management = sys.modules.get("comfy.model_management")
    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.model_management"] = fake_model_management
    try:
        spec = importlib.util.spec_from_file_location(
            "_raylight_driver_memory_under_test",
            ROOT / "src/directordeck_raylight/driver_memory.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_comfy is None:
            sys.modules.pop("comfy", None)
        else:
            sys.modules["comfy"] = previous_comfy
        if previous_model_management is None:
            sys.modules.pop("comfy.model_management", None)
        else:
            sys.modules["comfy.model_management"] = previous_model_management


DRIVER_MEMORY = _load_driver_memory()


class _FakeModelManagement:
    def __init__(self, calls):
        self.calls = calls

    def unload_all_models(self):
        self.calls.append(("unload_all_models",))

    def free_memory(self, required, device):
        self.calls.append(("free_memory", required, str(device)))

    def soft_empty_cache(self):
        self.calls.append(("soft_empty_cache",))


def _run_cleanup(parallel_dict, device_count=8):
    calls = []
    previous_model_management = DRIVER_MEMORY.comfy.model_management
    previous_collect = DRIVER_MEMORY.gc.collect
    previous_device_count = DRIVER_MEMORY.torch.cuda.device_count
    DRIVER_MEMORY.comfy.model_management = _FakeModelManagement(calls)
    DRIVER_MEMORY.gc.collect = lambda: calls.append(("gc",))
    DRIVER_MEMORY.torch.cuda.device_count = lambda: device_count
    try:
        DRIVER_MEMORY.cleanup_driver_models(parallel_dict)
    finally:
        DRIVER_MEMORY.comfy.model_management = previous_model_management
        DRIVER_MEMORY.gc.collect = previous_collect
        DRIVER_MEMORY.torch.cuda.device_count = previous_device_count
    return calls


def test_missing_or_invalid_policy_preserves_legacy_global_cleanup():
    expected = [("gc",), ("unload_all_models",), ("soft_empty_cache",)]
    assert _run_cleanup({}) == expected
    assert _run_cleanup({"driver_cleanup_policy": "future_value"}) == expected


def test_legacy_policy_preserves_global_cleanup_for_every_cluster_scope():
    expected = [("gc",), ("unload_all_models",), ("soft_empty_cache",)]
    for scope in (True, False, None):
        assert _run_cleanup(
            {
                "driver_cleanup_policy": "legacy_all",
                "ray_cluster_is_local": scope,
                "driver_gpu_indices": (0, 1),
            }
        ) == expected


def test_ray_devices_only_frees_selected_driver_logical_gpus():
    calls = _run_cleanup(
        {
            "driver_cleanup_policy": "ray_devices",
            "ray_cluster_is_local": True,
            "driver_gpu_indices": (1, 3),
        },
        device_count=4,
    )
    assert calls == [
        ("gc",),
        ("free_memory", 1e30, "cuda:1"),
        ("free_memory", 1e30, "cuda:3"),
    ]


def test_ray_devices_remote_cluster_does_not_touch_driver_memory():
    assert _run_cleanup(
        {
            "driver_cleanup_policy": "ray_devices",
            "ray_cluster_is_local": False,
            "driver_gpu_indices": (),
        }
    ) == []


def test_ray_devices_unknown_or_invalid_metadata_falls_back_to_global_cleanup():
    expected = [("gc",), ("unload_all_models",), ("soft_empty_cache",)]
    cases = (
        {"driver_cleanup_policy": "ray_devices"},
        {"driver_cleanup_policy": "ray_devices", "ray_cluster_is_local": True},
        {
            "driver_cleanup_policy": "ray_devices",
            "ray_cluster_is_local": True,
            "driver_gpu_indices": (),
        },
        {
            "driver_cleanup_policy": "ray_devices",
            "ray_cluster_is_local": True,
            "driver_gpu_indices": (8,),
        },
        {
            "driver_cleanup_policy": "ray_devices",
            "ray_cluster_is_local": True,
            "driver_gpu_indices": (1, 1),
        },
    )
    for parallel_dict in cases:
        assert _run_cleanup(parallel_dict, device_count=8) == expected


def test_initializer_metadata_uses_logical_selection_and_safe_empty_selection_scope():
    selected = DRIVER_MEMORY.build_driver_cleanup_metadata("ray_devices", True, (2, 0), 8)
    assert selected == {
        "driver_cleanup_policy": "ray_devices",
        "ray_cluster_is_local": True,
        "driver_gpu_indices": (2, 0),
    }

    # With no GPU_SELECT, Ray may schedule fewer workers than visible cards. All
    # driver-visible cards remain the only safe local cleanup scope.
    inherited = DRIVER_MEMORY.build_driver_cleanup_metadata("ray_devices", True, None, 8)
    assert inherited["driver_gpu_indices"] == tuple(range(8))

    remote = DRIVER_MEMORY.build_driver_cleanup_metadata("ray_devices", False, (2, 0), 8)
    assert remote["driver_gpu_indices"] == ()
    assert remote["ray_cluster_is_local"] is False


class _RemoteMethod:
    def __init__(self, callback):
        self.callback = callback

    def remote(self):
        return self.callback()


class _FakeActor:
    def __init__(self, parallel_dict, calls):
        self.get_parallel_dict = _RemoteMethod(lambda: parallel_dict)
        self.clear_sampling_vram = _RemoteMethod(lambda: calls.append(("worker_clear",)) or True)


def test_post_sampling_worker_clear_reuses_the_same_driver_scope():
    calls = []
    parallel_dict = {
        "clear_vram_after_sampling": True,
        "driver_cleanup_policy": "ray_devices",
        "ray_cluster_is_local": True,
        "driver_gpu_indices": (1,),
    }
    actor = _FakeActor(parallel_dict, calls)
    previous_get = DRIVER_MEMORY.ray.get
    previous_model_management = DRIVER_MEMORY.comfy.model_management
    previous_collect = DRIVER_MEMORY.gc.collect
    previous_device_count = DRIVER_MEMORY.torch.cuda.device_count
    DRIVER_MEMORY.ray.get = lambda value: value
    DRIVER_MEMORY.comfy.model_management = _FakeModelManagement(calls)
    DRIVER_MEMORY.gc.collect = lambda: calls.append(("gc",))
    DRIVER_MEMORY.torch.cuda.device_count = lambda: 2
    try:
        DRIVER_MEMORY.clear_ray_worker_vram_after_sampling({"workers": [actor]})
    finally:
        DRIVER_MEMORY.ray.get = previous_get
        DRIVER_MEMORY.comfy.model_management = previous_model_management
        DRIVER_MEMORY.gc.collect = previous_collect
        DRIVER_MEMORY.torch.cuda.device_count = previous_device_count
    assert calls == [
        ("worker_clear",),
        ("gc",),
        ("free_memory", 1e30, "cuda:1"),
    ]


def test_post_sampling_cleanup_disabled_does_not_touch_workers_or_driver():
    calls = []
    actor = _FakeActor({"clear_vram_after_sampling": False}, calls)
    previous_get = DRIVER_MEMORY.ray.get
    DRIVER_MEMORY.ray.get = lambda value: value
    try:
        DRIVER_MEMORY.clear_ray_worker_vram_after_sampling({"workers": [actor]})
    finally:
        DRIVER_MEMORY.ray.get = previous_get
    assert calls == []


def test_empty_actor_set_fails_before_any_destructive_fallback():
    try:
        DRIVER_MEMORY.cleanup_driver_models_for_ray({"workers": []})
    except RuntimeError as exc:
        assert "without Ray workers" in str(exc)
    else:
        raise AssertionError("Expected empty Ray actor set to fail")


def _method_calls(function):
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.append(node.func.attr)
    return calls


def _classes(path):
    tree = ast.parse(path.read_text())
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _method(class_node, name):
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_all_nine_samplers_use_the_shared_pre_and_post_cleanup_helpers():
    sampler_classes = {
        ROOT / "src/directordeck_raylight/nodes.py": (
            "XFuserKSamplerAdvanced",
            "UnifiedParallelSampler",
            "DPKSamplerAdvanced",
        ),
        ROOT / "src/directordeck_raylight/comfy_extra_dist/nodes_custom_sampler.py": (
            "XFuserSamplerCustomAdvanced",
            "XFuserSamplerCustom",
            "UnifiedParallelSamplerCustomAdvanced",
            "UnifiedParallelSamplerCustom",
            "DPSamplerCustomAdvanced",
            "DPSamplerCustom",
        ),
    }
    direct_cleanup_calls = {"unload_all_models", "free_memory", "soft_empty_cache"}
    for path, class_names in sampler_classes.items():
        classes = _classes(path)
        for class_name in class_names:
            calls = _method_calls(_method(classes[class_name], "ray_sample"))
            assert calls.count("cleanup_driver_models_for_ray") == 1, class_name
            assert calls.count("clear_ray_worker_vram_after_sampling") == 1, class_name
            assert not direct_cleanup_calls.intersection(calls), class_name


def test_initializer_contract_is_optional_legacy_default_and_metadata_is_captured():
    nodes_path = ROOT / "src/directordeck_raylight/nodes.py"
    source = nodes_path.read_text()
    classes = _classes(nodes_path)
    spawn_actor = _method(classes["RayInitializer"], "spawn_actor")
    argument_names = [argument.arg for argument in spawn_actor.args.args]
    defaults = [None] * (len(argument_names) - len(spawn_actor.args.defaults)) + list(spawn_actor.args.defaults)
    default_by_name = dict(zip(argument_names, defaults))

    assert isinstance(default_by_name["driver_cleanup_policy"], ast.Constant)
    assert default_by_name["driver_cleanup_policy"].value == "legacy_all"
    assert source.count('"driver_cleanup_policy": (') == 2
    advanced_source = source[source.index("class RayInitializerAdvanced") : source.index("class RayPipeFusionConfig")]
    assert advanced_source.index('"ray_object_store_gb"') < advanced_source.index('"ray_dashboard_address"')
    assert advanced_source.index('"ray_dashboard_address"') < advanced_source.index('"torch_dist_address"')
    assert advanced_source.index('"torch_dist_address"') < advanced_source.index('"driver_cleanup_policy"')
    calls = _method_calls(spawn_actor)
    assert calls.count("build_driver_cleanup_metadata") == 1
    assert calls.count("make_ray_actor_fn") == 1


def test_explicit_clean_vram_node_keeps_legacy_global_behavior():
    classes = _classes(ROOT / "src/directordeck_raylight/nodes.py")
    calls = _method_calls(_method(classes["RayCleanVRAMUsed"], "clean_vram"))
    assert calls.count("unload_all_models") == 1
    assert calls.count("soft_empty_cache") == 1
