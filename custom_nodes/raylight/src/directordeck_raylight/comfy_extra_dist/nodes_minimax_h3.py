from .ray_patch_decorator import ray_patch

try:
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
except ImportError as import_error:
    MiniMaxH3SigmaShift = None
    _MINIMAX_H3_IMPORT_ERROR = import_error


class RayMiniMaxH3SigmaShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ray_actors": ("RAY_ACTORS",),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("RAY_ACTORS",)
    RETURN_NAMES = ("ray_actors",)
    FUNCTION = "patch"
    CATEGORY = "Raylight/extra"

    @ray_patch
    def patch(self, model, shift_video, shift_audio):
        if MiniMaxH3SigmaShift is None:
            raise RuntimeError(
                "MiniMax H3 Sigma Shift is unavailable. Install or update ComfyUI to a version that provides "
                "comfy_extras.nodes_minimax_h3."
            ) from _MINIMAX_H3_IMPORT_ERROR

        return MiniMaxH3SigmaShift.execute(model, shift_video, shift_audio)[0]


NODE_CLASS_MAPPINGS = {
    "RayMiniMaxH3SigmaShift": RayMiniMaxH3SigmaShift,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayMiniMaxH3SigmaShift": "MiniMax H3 Sigma Shift (Ray)",
}
