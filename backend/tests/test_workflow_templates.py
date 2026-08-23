from __future__ import annotations

from directordeck.workflow.contracts import TemplateBundle
from directordeck.workflow.registry import FeatureInterpreterRegistry
from directordeck.workflow.templates import (
    V4_RAYLIGHT_SEGMENT_TEMPLATE,
    V4_STANDARD_SEGMENT_TEMPLATE,
    V4_TEMPLATE_BUNDLE,
)


class _Interpreter:
    def __init__(self, feature_id: str, version: int) -> None:
        self.id = feature_id
        self.version = version

    def validate_params(self, params, ctx) -> None:
        return None

    def resolve(self, params, ctx):
        return None

    def required_capabilities(self, params, ctx, resolution):
        return None

    def cache_identity(self, params, ctx, resolution):
        return None

    def runtime_pool_identity(self, params, ctx, resolution):
        return None

    def emit(self, builder, inputs, params, ctx, resolution):
        return None


def _entry_ids(template) -> tuple[str, ...]:
    return tuple(entry.id for entry in template.entries)


def test_v4_template_bundle_has_the_frozen_dual_segment_and_control_identities() -> None:
    assert V4_TEMPLATE_BUNDLE.version == 4
    assert V4_TEMPLATE_BUNDLE.segment_templates.standard is V4_STANDARD_SEGMENT_TEMPLATE
    assert V4_TEMPLATE_BUNDLE.segment_templates.raylight is V4_RAYLIGHT_SEGMENT_TEMPLATE
    assert V4_STANDARD_SEGMENT_TEMPLATE.id == "h3_standard_segment"
    assert V4_RAYLIGHT_SEGMENT_TEMPLATE.id == "h3_raylight_segment"
    assert V4_TEMPLATE_BUNDLE.control_templates.ray_kill.id == "raylight_kill_control"
    assert V4_TEMPLATE_BUNDLE.control_templates.ray_kill.revision == 1


def test_v4_standard_entry_order_matches_current_prompt_emission() -> None:
    ids = _entry_ids(V4_STANDARD_SEGMENT_TEMPLATE)
    assert ids == (
        "shared_models",
        "standard_model_load",
        "standard_model_device",
        "lora",
        "standard_sigma_shift",
        "family_conditioning",
        "continuity",
        "standard_sampling",
        "decode_video",
        "audio_output",
        "save_take",
    )
    assert ids.index("lora") < ids.index("standard_sigma_shift")


def test_v4_raylight_entry_order_matches_current_prompt_emission() -> None:
    ids = _entry_ids(V4_RAYLIGHT_SEGMENT_TEMPLATE)
    assert ids == (
        "shared_models",
        "raylight_pool_intent",
        "lora",
        "raylight_model_load",
        "raylight_sigma_shift",
        "family_conditioning",
        "continuity",
        "raylight_sampling",
        "decode_video",
        "audio_output",
        "save_take",
    )
    assert ids.index("lora") < ids.index("raylight_model_load")


def test_v4_bundle_is_strict_json_round_trip_stable() -> None:
    payload = V4_TEMPLATE_BUNDLE.model_dump_json()
    restored = TemplateBundle.model_validate_json(payload)
    assert restored == V4_TEMPLATE_BUNDLE
    assert restored.model_dump(mode="json") == V4_TEMPLATE_BUNDLE.model_dump(
        mode="json"
    )


def test_every_v4_entry_is_internal_graph_only_and_backend_specific() -> None:
    for backend, template in (
        ("standard", V4_STANDARD_SEGMENT_TEMPLATE),
        ("raylight", V4_RAYLIGHT_SEGMENT_TEMPLATE),
    ):
        for entry in template.entries:
            assert entry.layer == "graph"
            assert entry.backends == (backend,)
            assert entry.families == ("fl2va", "ref2va")
            assert entry.scopes == ("segment",)
            assert entry.ui == {"visibility": "internal_v4"}


def test_v4_templates_pass_exact_registry_and_resource_flow_validation() -> None:
    registry = FeatureInterpreterRegistry()
    identities = dict.fromkeys(
        (
            (entry.id, entry.version)
            for template in (
                V4_STANDARD_SEGMENT_TEMPLATE,
                V4_RAYLIGHT_SEGMENT_TEMPLATE,
            )
            for entry in template.entries
        )
    )
    for feature_id, version in identities:
        registry.register(_Interpreter(feature_id, version))
    registry.freeze()

    standard = registry.validate_template(V4_STANDARD_SEGMENT_TEMPLATE)
    raylight = registry.validate_template(V4_RAYLIGHT_SEGMENT_TEMPLATE)
    assert tuple(binding.id for binding in standard.bindings) == _entry_ids(
        V4_STANDARD_SEGMENT_TEMPLATE
    )
    assert tuple(binding.id for binding in raylight.bindings) == _entry_ids(
        V4_RAYLIGHT_SEGMENT_TEMPLATE
    )


def test_v4_switches_are_only_current_conditional_graph_behaviors() -> None:
    standard_switches = tuple(
        entry.id
        for entry in V4_STANDARD_SEGMENT_TEMPLATE.entries
        if entry.mode == "switch"
    )
    raylight_switches = tuple(
        entry.id
        for entry in V4_RAYLIGHT_SEGMENT_TEMPLATE.entries
        if entry.mode == "switch"
    )
    assert standard_switches == ("lora", "continuity")
    assert raylight_switches == ("lora", "continuity")


def test_only_unused_mode_specific_resources_are_conditional_v4_writes() -> None:
    for template in (
        V4_STANDARD_SEGMENT_TEMPLATE,
        V4_RAYLIGHT_SEGMENT_TEMPLATE,
    ):
        conditional = tuple(
            (entry.id, write.name)
            for entry in template.entries
            for write in entry.writes
            if not write.required
        )
        assert conditional == (
            ("shared_models", "audio_vae"),
            ("family_conditioning", "source_audio"),
        )
