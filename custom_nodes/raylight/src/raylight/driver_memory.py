# Added for Director Web; see DIRECTOR_MODIFICATIONS.md.
import gc

import ray
import torch

import comfy.model_management


DRIVER_CLEANUP_POLICIES = ("legacy_all", "ray_devices")


def normalize_driver_cleanup_policy(value):
    if value in DRIVER_CLEANUP_POLICIES:
        return value
    return "legacy_all"


def build_driver_cleanup_metadata(policy, cluster_is_local, selected_gpus, visible_device_count):
    if cluster_is_local:
        gpu_indices = tuple(selected_gpus) if selected_gpus is not None else tuple(range(visible_device_count))
    else:
        gpu_indices = ()
    return {
        "driver_cleanup_policy": normalize_driver_cleanup_policy(policy),
        "ray_cluster_is_local": bool(cluster_is_local),
        "driver_gpu_indices": gpu_indices,
    }


def _driver_gpu_indices(parallel_dict):
    if parallel_dict.get("ray_cluster_is_local") is False:
        return ()
    if parallel_dict.get("ray_cluster_is_local") is not True:
        return None

    values = parallel_dict.get("driver_gpu_indices")
    if not isinstance(values, (list, tuple)) or not values:
        return None

    device_count = torch.cuda.device_count()
    indices = []
    for value in values:
        if isinstance(value, bool):
            return None
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        if index < 0 or index >= device_count or index in indices:
            return None
        indices.append(index)
    return tuple(indices)


def cleanup_driver_models(parallel_dict):
    policy = normalize_driver_cleanup_policy(parallel_dict.get("driver_cleanup_policy"))
    if policy == "ray_devices":
        indices = _driver_gpu_indices(parallel_dict)
        if indices == ():
            return
        if indices is not None:
            gc.collect()
            for index in indices:
                comfy.model_management.free_memory(1e30, torch.device("cuda", index))
            return

    gc.collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


def cleanup_driver_models_for_ray(ray_actors, parallel_dict=None):
    gpu_actors = ray_actors["workers"]
    if not gpu_actors:
        raise RuntimeError("Raylight cannot prepare driver VRAM without Ray workers")
    if parallel_dict is None:
        parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())
    cleanup_driver_models(parallel_dict)


def clear_ray_worker_vram_after_sampling(ray_actors):
    gpu_actors = ray_actors["workers"]
    if not gpu_actors:
        return
    parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())
    if not parallel_dict.get("clear_vram_after_sampling", False):
        return

    ray.get([actor.clear_sampling_vram.remote() for actor in gpu_actors])
    cleanup_driver_models(parallel_dict)
