from __future__ import annotations

"""Bundle-6 feature names for the shared execution-hint builder."""

from ..contracts import FeatureEmission, ScopedGraphBuilderProtocol
from ..execution_hints import build_feature_execution_hints


_PRE_SAMPLING = frozenset(
    {
        "auxiliary_models",
        "diffusion_model",
        "execution_strategy",
        "lora",
        "sigma_schedule",
        "comfy_kitchen_attention",
        "multimodal_conditioning",
        "continuity",
        "sampling_pipeline",
    }
)
_SAMPLING = frozenset({"sampling_pipeline"})
_LABEL_OVERRIDES = {"ModelAttentionBackend": "配置 CK Attention"}


def attach_v6_execution_hints(
    feature_id: str,
    builder: ScopedGraphBuilderProtocol,
    emission: FeatureEmission,
) -> FeatureEmission:
    hints, preview = build_feature_execution_hints(
        feature_id=feature_id,
        outputs=emission.outputs,
        builder=builder,
        pre_sampling_features=_PRE_SAMPLING,
        sampling_features=_SAMPLING,
        label_overrides=_LABEL_OVERRIDES,
    )
    if not hints and not preview:
        return emission
    return emission.model_copy(
        update={"progress_hints": hints, "preview_hints": preview}
    )


__all__ = ["attach_v6_execution_hints"]
