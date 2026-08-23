import comfy.context_windows
import nodes

from .ray_patch_decorator import ray_patch


class RayLTXVContextWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "context_length": (
                    "INT",
                    {
                        "default": 145,
                        "min": 1,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 8,
                        "tooltip": "The length of the context window in real frames. Must be 8*n + 1.",
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 40,
                        "min": 0,
                        "step": 8,
                        "tooltip": "The overlap of the context window in real frames.",
                    },
                ),
                "context_schedule": (
                    [
                        comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                        comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                        comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
                        comfy.context_windows.ContextSchedules.BATCHED,
                    ],
                    {
                        "default": comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                        "tooltip": "Step-dependent scheduling algorithm for context windows.",
                    },
                ),
                "fuse_method": (
                    comfy.context_windows.ContextFuseMethods.LIST_STATIC,
                    {
                        "default": comfy.context_windows.ContextFuseMethods.PYRAMID,
                        "tooltip": "The method to use to fuse the context windows.",
                    },
                ),
                "freenoise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to apply FreeNoise noise shuffling, improves window blending.",
                    },
                ),
                "retain_first_frame": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Retain the first latent frame in every context window (may help retain initial reference).",
                    },
                ),
                "split_conds_to_windows": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to split multiple conditionings (created by ConditionCombine) to each window based on region index.",
                    },
                ),
                "context_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "tooltip": "The stride of the context window; only applicable to uniform schedules.",
                    },
                ),
                "closed_loop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to close the context window loop; only applicable to looped schedules.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    @ray_patch
    def patch(
        self,
        model,
        context_length,
        context_overlap,
        context_schedule,
        fuse_method,
        freenoise,
        retain_first_frame=False,
        split_conds_to_windows=False,
        context_stride=1,
        closed_loop=False,
    ):
        m = model.clone()

        context_length = max(((context_length - 1) // 8) + 1, 1)
        context_overlap = max(context_overlap // 8, 0)
        retain_index_list = "0" if retain_first_frame else ""

        m.model_options["context_handler"] = comfy.context_windows.IndexListContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=context_length,
            context_overlap=context_overlap,
            context_stride=context_stride,
            closed_loop=closed_loop,
            dim=2,
            freenoise=freenoise,
            cond_retain_index_list=retain_index_list,
            split_conds_to_windows=split_conds_to_windows,
            latent_retain_index_list=retain_index_list,
        )

        comfy.context_windows.create_prepare_sampling_wrapper(m)
        if freenoise:
            comfy.context_windows.create_sampler_sample_wrapper(m)

        return m


NODE_CLASS_MAPPINGS = {
    "RayLTXVContextWindows": RayLTXVContextWindows,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayLTXVContextWindows": "LTXV Context Windows (Ray)",
}
