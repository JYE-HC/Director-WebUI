from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import directordeck.workflow.feature_config as feature_config
from directordeck.schemas import RuntimeSettingsV3, UnifiedTimelineDraftV5
from directordeck.workflow.contracts import GRAPH_PHASE_ORDER
from directordeck.workflow.feature_config import (
    LoraConfigV1,
    V6_CONFIG_RESOLVERS,
    V6FeatureConfigurationError,
)
from directordeck.workflow.feature_config_models import AuxiliaryModelsConfigV1
from directordeck.workflow.feature_definitions import (
    BUNDLE6_FEATURE_DEFINITIONS,
    BUNDLE6_FEATURE_DEFINITIONS_BY_ID,
    FeatureDefinition,
)
from directordeck.workflow.templates_v6 import (
    V6_RAYLIGHT_SEGMENT_TEMPLATE,
    V6_STANDARD_SEGMENT_TEMPLATE,
)

from .test_workflow_v5_compat import _v4_pair, _v5_pair


_STANDARD_ORDER = (
    "auxiliary_models",
    "diffusion_model",
    "execution_strategy",
    "lora",
    "comfy_kitchen_attention",
    "sigma_schedule",
    "multimodal_conditioning",
    "continuity",
    "sampling_pipeline",
    "video_decode",
    "audio_output",
    "save_take",
)
_RAYLIGHT_ORDER = (
    "auxiliary_models",
    "comfy_kitchen_attention",
    "execution_strategy",
    "lora",
    "diffusion_model",
    "sigma_schedule",
    "multimodal_conditioning",
    "continuity",
    "sampling_pipeline",
    "video_decode",
    "audio_output",
    "save_take",
)


def _pair() -> tuple[UnifiedTimelineDraftV5, RuntimeSettingsV3]:
    v4, settings_v1 = _v4_pair()
    draft, settings = _v5_pair(v4, settings_v1)
    document = draft.model_dump(mode="json")
    document["features"]["template_bundle_version"] = 6
    document["features"]["project"]["comfy_kitchen_attention"] = {
        "enabled": False,
        "params": {},
    }
    return UnifiedTimelineDraftV5.model_validate(document), settings


def _context(
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    *,
    backend: str = "standard",
) -> SimpleNamespace:
    return SimpleNamespace(
        draft=draft,
        settings=settings,
        segment=draft.segments[0],
        backend=backend,
        family="fl2va",
        job_id="stage1a-v6",
        visible_frames=120,
        sample_frames=120,
        continuity_prefix_frames=0,
        predecessor_segment_id=None,
        continuity_source=None,
        historical_take_id=None,
        clear_raylight_vram_after_sampling=False,
    )


def _with_lora(
    draft: UnifiedTimelineDraftV5,
    filename: str,
) -> UnifiedTimelineDraftV5:
    document = draft.model_dump(mode="json")
    document["features"]["project"]["lora"] = {
        "enabled": True,
        "params": {
            "by_family": {
                "fl2va": {
                    "enabled": True,
                    "filename": filename,
                    "strength": 0.75,
                },
                "ref2va": {
                    "enabled": False,
                    "filename": None,
                    "strength": 1.0,
                },
            }
        },
    }
    return UnifiedTimelineDraftV5.model_validate(document)


def _with_override(
    settings: RuntimeSettingsV3,
    filename: str,
) -> RuntimeSettingsV3:
    document = settings.model_dump(mode="json")
    document["lora_loader_overrides"] = [
        {
            "lora_filename": filename,
            "adapter_id": "minimax_h3_turbo",
            "options": {"low_vram": True},
        }
    ]
    return RuntimeSettingsV3.model_validate(document)


def test_bundle6_has_exactly_twelve_graph_free_ui_free_definitions() -> None:
    assert len(BUNDLE6_FEATURE_DEFINITIONS) == 12
    assert len(BUNDLE6_FEATURE_DEFINITIONS_BY_ID) == 12
    assert set(FeatureDefinition.model_fields) == {
        "id",
        "version",
        "title",
        "description",
        "activation",
        "authoring",
        "backends",
        "families",
    }
    assert tuple(BUNDLE6_FEATURE_DEFINITIONS_BY_ID) == tuple(
        definition.id for definition in BUNDLE6_FEATURE_DEFINITIONS
    )


@pytest.mark.parametrize(
    ("template", "expected_order", "expected_resource_order"),
    (
        (
            V6_STANDARD_SEGMENT_TEMPLATE,
            _STANDARD_ORDER,
            (
                "clip",
                "video_vae",
                "audio_vae",
                "model",
                "conditioning",
                "latent",
                "source_audio",
                "samples",
                "frames",
                "video",
                "take_output",
            ),
        ),
        (
            V6_RAYLIGHT_SEGMENT_TEMPLATE,
            _RAYLIGHT_ORDER,
            (
                "clip",
                "video_vae",
                "audio_vae",
                "ray_actors_init",
                "ray_lora",
                "model",
                "conditioning",
                "latent",
                "source_audio",
                "samples",
                "frames",
                "video",
                "take_output",
            ),
        ),
    ),
    ids=("standard", "raylight"),
)
def test_bundle6_templates_have_exact_order_and_closed_resource_flow(
    template: Any,
    expected_order: tuple[str, ...],
    expected_resource_order: tuple[str, ...],
) -> None:
    assert len(template.entries) == 12
    assert tuple(entry.feature_id for entry in template.entries) == expected_order

    phase_index = {phase: index for index, phase in enumerate(GRAPH_PHASE_ORDER)}
    resources: dict[str, tuple[str, bool]] = {}
    seen: dict[str, int] = {}
    previous_phase = -1
    for index, entry in enumerate(template.entries):
        assert entry.feature_version == 1
        assert entry.feature_id in BUNDLE6_FEATURE_DEFINITIONS_BY_ID
        current_phase = phase_index[entry.graph_phase]
        assert current_phase >= previous_phase
        previous_phase = current_phase
        for dependency in entry.dependencies:
            assert seen[dependency.feature_id] < index
            assert dependency.feature_version == 1
        for read in entry.reads:
            producer = resources.get(read.name)
            if read.required:
                assert producer is not None
                assert producer[1] is True
            if producer is not None:
                assert producer[0] == read.type
        for write in entry.writes:
            previous = resources.get(write.name)
            if write.operation == "define":
                assert previous is None
                resources[write.name] = (write.type, write.required)
            else:
                assert previous is not None
                assert previous[0] == write.type
                resources[write.name] = previous
        seen[entry.feature_id] = index

    assert tuple(resources) == expected_resource_order
    ray_execution = next(
        entry for entry in template.entries if entry.feature_id == "execution_strategy"
    )
    if template is V6_RAYLIGHT_SEGMENT_TEMPLATE:
        assert tuple(
            (dependency.feature_id, dependency.required)
            for dependency in ray_execution.dependencies
        ) == (("comfy_kitchen_attention", False),)
    else:
        assert ray_execution.dependencies == ()


def test_ck_is_project_only_and_canonical_off_is_inactive() -> None:
    draft, settings = _pair()
    definition = BUNDLE6_FEATURE_DEFINITIONS_BY_ID["comfy_kitchen_attention"]
    assert definition.authoring == "project"

    inactive = V6_CONFIG_RESOLVERS["comfy_kitchen_attention"].resolve(
        _context(draft, settings)
    )
    assert inactive.state == "inactive"
    assert inactive.source == "project"
    assert inactive.reason_code == "disabled"

    project_document = draft.model_dump(mode="json")
    project_document["features"]["project"]["comfy_kitchen_attention"] = {
        "enabled": True,
        "params": {},
    }
    project_draft = UnifiedTimelineDraftV5.model_validate(project_document)
    active = V6_CONFIG_RESOLVERS["comfy_kitchen_attention"].resolve(
        _context(project_draft, settings)
    )
    assert active.state == "applicable"
    assert active.source == "project"

    segment_document = draft.model_dump(mode="json")
    segment_document["features"]["by_segment"] = {
        draft.segments[0].id: {
            "comfy_kitchen_attention": {"enabled": True, "params": {}}
        }
    }
    segment_draft = UnifiedTimelineDraftV5.model_validate(segment_document)
    with pytest.raises(V6FeatureConfigurationError) as caught:
        V6_CONFIG_RESOLVERS["comfy_kitchen_attention"].resolve(
            _context(segment_draft, settings)
        )
    assert caught.value.code == "feature_scope_unsupported"


def test_disabled_lora_does_not_call_any_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft, settings = _pair()

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("disabled LoRA must not resolve a loader")

    monkeypatch.setattr(feature_config, "resolve_standard_lora_adapter", unexpected)
    monkeypatch.setattr(feature_config, "resolve_raylight_lora_adapter", unexpected)
    resolved = V6_CONFIG_RESOLVERS["lora"].resolve(_context(draft, settings))
    assert resolved.state == "inactive"
    assert resolved.reason_code == "disabled"


@pytest.mark.parametrize("audio_mode", ("mute", "source"))
def test_fl2va_non_generated_audio_does_not_configure_an_audio_vae(
    audio_mode: str,
) -> None:
    draft, settings = _pair()
    document = draft.model_dump(mode="json")
    document["segments"][0]["audio_mode"] = audio_mode
    draft = UnifiedTimelineDraftV5.model_validate(document)

    resolved = V6_CONFIG_RESOLVERS["auxiliary_models"].resolve(
        _context(draft, settings)
    )

    assert isinstance(resolved.config, AuxiliaryModelsConfigV1)
    assert resolved.config.audio_vae_filename is None
    assert resolved.config.audio_vae_device is None


@pytest.mark.parametrize(
    (
        "filename",
        "override",
        "adapter_id",
        "class_type",
        "source",
        "options",
    ),
    (
        (
            "nested/style.safetensors",
            False,
            "model_only",
            "LoraLoaderModelOnly",
            "factory_default",
            {},
        ),
        (
            "nested/minimax_h3_turbo_v4_step600_ema.safetensors",
            False,
            "minimax_h3_turbo",
            "MiniMaxH3TurboLoRA",
            "factory_default",
            {"low_vram": False},
        ),
        (
            "nested/minimax_h3_turbo_v4_step600_ema.safetensors",
            True,
            "minimax_h3_turbo",
            "MiniMaxH3TurboLoRA",
            "user_override",
            {"low_vram": True},
        ),
    ),
    ids=("fallback", "regex", "exact_override"),
)
def test_v6_standard_lora_reuses_existing_factory_matrix(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    override: bool,
    adapter_id: str,
    class_type: str,
    source: str,
    options: dict[str, bool],
) -> None:
    draft, settings = _pair()
    draft = _with_lora(draft, filename)
    if override:
        settings = _with_override(settings, filename)
    actual_factory = feature_config.resolve_standard_lora_adapter
    calls: list[tuple[str, tuple[str, ...]]] = []

    def factory_spy(binding: Any, overrides: Any):
        records = tuple(overrides)
        calls.append(
            (binding.lora_filename, tuple(record.adapter_id for record in records))
        )
        return actual_factory(binding, records)

    monkeypatch.setattr(feature_config, "resolve_standard_lora_adapter", factory_spy)
    resolved = V6_CONFIG_RESOLVERS["lora"].resolve(
        _context(draft, settings, backend="standard")
    )

    assert calls == [(filename, ("minimax_h3_turbo",) if override else ())]
    assert resolved.state == "applicable"
    assert isinstance(resolved.config, LoraConfigV1)
    assert resolved.config.adapter_id == adapter_id
    assert resolved.config.class_type == class_type
    assert resolved.config.source == source
    assert resolved.config.options == options


def test_v6_raylight_lora_uses_the_existing_fixed_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft, settings = _pair()
    filename = "nested/style.safetensors"
    draft = _with_lora(draft, filename)
    actual_factory = feature_config.resolve_raylight_lora_adapter
    calls: list[str] = []

    def factory_spy(family: str):
        calls.append(family)
        return actual_factory(family)  # type: ignore[arg-type]

    monkeypatch.setattr(feature_config, "resolve_raylight_lora_adapter", factory_spy)
    resolved = V6_CONFIG_RESOLVERS["lora"].resolve(
        _context(draft, settings, backend="raylight")
    )

    assert calls == ["fl2va"]
    assert isinstance(resolved.config, LoraConfigV1)
    assert resolved.config.adapter_id == "ray_lora"
    assert resolved.config.class_type == "DirectorDeckRayLoraLoader"
    assert resolved.config.input_contract == "ray_lora"
    assert resolved.config.source == "backend_fixed"
    assert resolved.config.options == {}
