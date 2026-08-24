from __future__ import annotations

"""Shared labels and builders for feature-owned execution hints."""

from collections.abc import Mapping, Set

from .contracts import (
    BoundedJsonValue,
    EdgeRef,
    ScopedGraphBuilderProtocol,
    TerminalRef,
)


# Labels shared by typed Bundle-5/6 hints and historical progress fallback.
COMMON_NODE_STAGE_LABELS: dict[str, str] = {
    "CLIPLoader": "加载文本编码器",
    "SelectCLIPDevice": "分配文本编码器设备",
    "VAELoader": "加载 VAE",
    "SelectVAEDevice": "分配 VAE 设备",
    "UNETLoader": "加载生成模型",
    "SelectModelDevice": "分配生成模型设备",
    "LoraLoaderModelOnly": "加载 LoRA",
    "LoraLoaderBypassModelOnly": "加载 LoRA",
    "MiniMaxH3TurboLoRA": "加载 H3 Turbo LoRA",
    "MiniMaxH3SigmaShift": "配置生成模型",
    "DirectorDeckRayInitializerAdvanced": "初始化 RayLight 多卡",
    "DirectorDeckRayLoraLoader": "加载 RayLight LoRA",
    "DirectorDeckRayUNETLoader": "加载 RayLight 生成模型",
    "DirectorDeckRayMiniMaxH3SigmaShift": "配置 RayLight 生成模型",
    "LoadImage": "读取参考图",
    "LoadVideo": "读取参考视频",
    "Video Slice": "裁剪参考视频",
    "GetVideoComponents": "解析参考视频",
    "LoadAudio": "读取参考音频",
    "ImageFromBatch": "处理参考图",
    "MiniMaxH3ImageToVideo": "构建画面条件",
    "MiniMaxH3ReferenceToVideo": "构建多模态条件",
    "BasicGuider": "准备采样引导",
    "DirectorDeckRayBasicGuider": "准备 RayLight 采样引导",
    "BasicScheduler": "生成采样计划",
    "DirectorDeckRayBasicScheduler": "生成 RayLight 采样计划",
    "KSamplerSelect": "选择采样器",
    "RandomNoise": "生成初始噪声",
}

_FEATURE_NODE_STAGE_LABELS = {
    **COMMON_NODE_STAGE_LABELS,
    "TrimAudioDuration": "裁剪参考音频",
    "MiniMaxH3AddGuide": "构建接续条件",
}


def _phase(
    phase_id: str,
    label: str,
    node_id: str,
    kind: str,
    weight: float,
) -> dict[str, BoundedJsonValue]:
    return {
        "id": phase_id,
        "label": label,
        "node_id": node_id,
        "kind": kind,
        "weight": weight,
    }


def build_feature_execution_hints(
    *,
    feature_id: str,
    outputs: Mapping[str, object],
    builder: ScopedGraphBuilderProtocol,
    pre_sampling_features: Set[str],
    sampling_features: Set[str],
    label_overrides: Mapping[str, str] | None = None,
) -> tuple[tuple[BoundedJsonValue, ...], tuple[BoundedJsonValue, ...]]:
    """Build the existing typed progress and preview hints for one feature."""

    phase: dict[str, BoundedJsonValue] | None = None
    preview: tuple[BoundedJsonValue, ...] = ()
    if feature_id in sampling_features:
        samples = outputs.get("samples")
        if not isinstance(samples, EdgeRef):
            raise AssertionError("sampling feature must publish samples")
        phase = _phase("sampling", "采样中", samples.node_id, "fractional", 0.70)
        preview = (
            {
                "node_id": samples.node_id,
                "phase_id": "sampling",
                "publish": True,
                "priority": 100,
            },
        )
    elif feature_id in {"decode_video", "video_decode"}:
        if not builder.emitted_node_ids:
            raise AssertionError("decode feature must emit a video decode node")
        phase = _phase(
            "decode_video",
            "解码视频画面",
            builder.emitted_node_ids[0],
            "milestone",
            0.15,
        )
    elif feature_id == "audio_output":
        video = outputs.get("video")
        if not isinstance(video, EdgeRef):
            raise AssertionError("audio output feature must publish video")
        phase = _phase(
            "assemble_media", "封装音视频", video.node_id, "milestone", 0.10
        )
    elif feature_id == "save_take":
        take = outputs.get("take_output")
        if not isinstance(take, TerminalRef):
            raise AssertionError("save feature must publish take")
        phase = _phase(
            "persist_take", "写入视频文件", take.node_id, "milestone", 0.05
        )
    elif feature_id not in pre_sampling_features:
        return (), ()

    labels = label_overrides or {}
    primary_node_id = str(phase["node_id"]) if phase is not None else None
    hints: list[BoundedJsonValue] = []
    if feature_id in pre_sampling_features:
        for node_id, node in builder.prompt_fragment.items():
            if node_id == primary_node_id:
                continue
            class_type = node.get("class_type")
            if not isinstance(class_type, str):
                continue
            label = labels.get(class_type, _FEATURE_NODE_STAGE_LABELS.get(class_type))
            if label is not None:
                hints.append(
                    _phase(
                        f"{feature_id}_{node_id}_stage",
                        label,
                        node_id,
                        "stage",
                        0.0,
                    )
                )
    if phase is not None:
        hints.append(phase)
    return tuple(hints), preview


__all__ = [
    "COMMON_NODE_STAGE_LABELS",
    "build_feature_execution_hints",
]
