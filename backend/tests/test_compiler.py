from __future__ import annotations

import pytest

from director.compiler import DraftNotRunnable, compile_prompt
from director.schemas import MODEL_ROUTE, RuntimeSettings, default_settings, validate_mode_draft

from .conftest import runnable_draft


def node_types(prompt: dict) -> set[str]:
    return {str(node["class_type"]) for node in prompt.values()}


def node(prompt: dict, class_type: str) -> dict:
    return next(item for item in prompt.values() if item["class_type"] == class_type)


@pytest.mark.parametrize("mode", ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"])
def test_legacy_compiler_uses_server_owned_native_graph(mode: str) -> None:
    settings = default_settings()
    prompt = compile_prompt(
        validate_mode_draft(mode, runnable_draft(mode)), settings, job_id="legacy-job"
    )

    types = node_types(prompt)
    assert "MiniMaxH3Director" not in types
    assert {
        "CLIPLoader",
        "VAELoader",
        "UNETLoader",
        "CreateVideo",
        "SaveVideo",
    } <= types
    assert node(prompt, "UNETLoader")["inputs"]["unet_name"] == getattr(
        settings.models, MODEL_ROUTE[mode]
    ).filename
    expected_conditioning = (
        "MiniMaxH3ImageToVideo"
        if mode in {"t2v", "i2v", "fl2v"}
        else "MiniMaxH3ReferenceToVideo"
    )
    assert expected_conditioning in types


def test_legacy_compiler_has_no_negative_conditioning_and_resolves_random_seed(
    monkeypatch,
) -> None:
    value = runnable_draft("t2v")
    value["sampling"]["seed"] = -1
    monkeypatch.setattr("director.schemas.secrets.randbelow", lambda limit: 4306)

    prompt = compile_prompt(
        validate_mode_draft("t2v", value), default_settings(), job_id="seed-job"
    )

    assert "negative" not in str(prompt).lower()
    assert node(prompt, "RandomNoise")["inputs"]["noise_seed"] == 4306


@pytest.mark.parametrize(
    ("obsolete_loader", "lora_name", "expected_node"),
    [
        ("auto", "minimax_h3_turbo_v4_step600_ema.safetensors", "MiniMaxH3TurboLoRA"),
        (
            "dedicated",
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_10ErosMax_beta1_pruned_compat_v001_T8.safetensors",
            "LoraLoaderBypassModelOnly",
        ),
        (
            "bypass_model_only",
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "LoraLoaderModelOnly",
        ),
        ("model_only", "minimax_h3_turbo_v4_step600.safetensors", "MiniMaxH3TurboLoRA"),
    ],
)
def test_legacy_compiler_derives_lora_dialect_and_ignores_obsolete_selection(
    obsolete_loader: str, lora_name: str, expected_node: str
) -> None:
    raw = default_settings().model_dump(mode="json")
    raw["models"]["fl2va"].update(
        lora_name=lora_name,
        lora_loader=obsolete_loader,
        lora_strength=0.75,
    )
    prompt = compile_prompt(
        validate_mode_draft("t2v", runnable_draft("t2v")),
        RuntimeSettings.model_validate(raw),
        job_id="lora-job",
    )
    assert expected_node in node_types(prompt)


@pytest.mark.parametrize(
    "lora_name",
    [
        "unknown.safetensors",
        "minimax_h3_surprise_compat_v999_T8.safetensors",
        "minimax_h3_surprise_comfyui_bf16.safetensors",
    ],
)
def test_legacy_compiler_refuses_unknown_lora_even_with_obsolete_override(
    lora_name: str,
) -> None:
    raw = default_settings().model_dump(mode="json")
    raw["models"]["fl2va"].update(
        lora_name=lora_name, lora_loader="model_only"
    )
    with pytest.raises(DraftNotRunnable, match="cannot be inferred safely"):
        compile_prompt(
            validate_mode_draft("t2v", runnable_draft("t2v")),
            RuntimeSettings.model_validate(raw),
            job_id="bad-lora",
        )


def test_legacy_compiler_does_not_reinterpret_incomplete_i2v_as_t2v() -> None:
    value = runnable_draft("i2v")
    value["shots"][0]["first_image"] = None

    with pytest.raises(DraftNotRunnable, match="i2v shots missing first_image"):
        compile_prompt(
            validate_mode_draft("i2v", value),
            default_settings(),
            job_id="incomplete-i2v",
        )


@pytest.mark.parametrize(
    ("mode", "valid_prompt", "invalid_tag"),
    [
        ("i2v", "Begin from <Picture 1>.", "<Picture 2>"),
        ("fl2v", "Move from <Picture 1> to <Picture 2>.", "<Picture 3>"),
    ],
)
def test_legacy_fl_picture_tags_follow_connected_keyframes(
    mode: str,
    valid_prompt: str,
    invalid_tag: str,
) -> None:
    value = runnable_draft(mode)
    value["shots"][0]["prompt"] = valid_prompt
    compile_prompt(
        validate_mode_draft(mode, value),
        default_settings(),
        job_id=f"legacy-{mode}-picture-tags",
    )

    value["shots"][0]["prompt"] = f"Invalid {invalid_tag}."
    with pytest.raises(DraftNotRunnable, match=invalid_tag.replace(" ", r"\s")):
        compile_prompt(
            validate_mode_draft(mode, value),
            default_settings(),
            job_id=f"legacy-{mode}-invalid-picture-tag",
        )


def test_legacy_compiler_rejects_non_native_fps_and_obsolete_memory_policy() -> None:
    value = runnable_draft("t2v")
    value["render"]["fps"] = 30.0
    with pytest.raises(DraftNotRunnable, match="must equal 24"):
        compile_prompt(
            validate_mode_draft("t2v", value), default_settings(), job_id="bad-fps"
        )

    raw = default_settings().model_dump(mode="json")
    raw["memory_policy"] = "clear_between_segments"
    with pytest.raises(ValueError, match="keep_resident"):
        RuntimeSettings.model_validate(raw)
