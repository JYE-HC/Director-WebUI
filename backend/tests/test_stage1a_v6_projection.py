from __future__ import annotations

from copy import deepcopy

import pytest

from directordeck.schemas import RuntimeSettingsV3, UnifiedTimelineDraftV5
from directordeck.workflow.v6_projection import (
    V5V6ProjectionError,
    project_v5_authority_to_v6,
    project_v6_compile_authority,
)
from directordeck.workflow.feature_config import V6FeatureConfigurationError

from .test_workflow_v5_compat import _v4_pair, _v5_pair


def _bundle5_draft() -> UnifiedTimelineDraftV5:
    v4, settings_v1 = _v4_pair()
    draft, _settings_v3 = _v5_pair(v4, settings_v1)
    document = draft.model_dump(mode="json")
    document["features"]["template_bundle_version"] = 5
    return UnifiedTimelineDraftV5.model_validate(document)


def test_runtime_switch_alone_selects_standard_or_raylight_route() -> None:
    v4, settings_v1 = _v4_pair()
    _draft, settings = _v5_pair(v4, settings_v1)
    draft = project_v5_authority_to_v6(_bundle5_draft()).draft
    settings_document = settings.model_dump(mode="json")
    profile = settings_document["placement"]["fl2va"]["raylight"]
    profile.update(gpu_select=[0, 1], ulysses_degree=2)
    settings_document["multi_gpu_enabled"] = False
    standard = RuntimeSettingsV3.model_validate(settings_document)
    assert project_v6_compile_authority(draft, standard).routes[0].backend == "standard"

    profile.update(gpu_select=[0], ulysses_degree=1)
    settings_document["multi_gpu_enabled"] = True
    raylight = RuntimeSettingsV3.model_validate(settings_document)
    assert project_v6_compile_authority(draft, raylight).routes[0].backend == "raylight"


def test_compile_projection_keeps_only_requested_segments() -> None:
    v4, settings_v1 = _v4_pair()
    _draft, settings = _v5_pair(v4, settings_v1)
    document = project_v5_authority_to_v6(_bundle5_draft()).draft.model_dump(
        mode="json"
    )
    second = deepcopy(document["segments"][0])
    second.update(id="segment-2", title="Segment 2")
    document["segments"].append(second)
    draft = UnifiedTimelineDraftV5.model_validate(document)

    projection = project_v6_compile_authority(
        draft,
        settings,
        segment_ids=["segment-2"],
    )

    assert projection.selected_segment_ids == ("segment-2",)
    assert [route.segment.id for route in projection.routes] == ["segment-2"]


def test_compile_projection_reports_all_reachable_missing_model_bindings() -> None:
    v4, settings_v1 = _v4_pair()
    _draft, settings = _v5_pair(v4, settings_v1)
    document = project_v5_authority_to_v6(_bundle5_draft()).draft.model_dump(
        mode="json"
    )
    document["segments"][0]["audio_mode"] = "mute"
    for role in ("clip", "video_vae", "audio_vae", "fl2va", "ref2va"):
        document["model_stack"][role]["filename"] = None
    draft = UnifiedTimelineDraftV5.model_validate(document)

    with pytest.raises(V6FeatureConfigurationError) as caught:
        project_v6_compile_authority(draft, settings)

    assert caught.value.code == "model_binding_required"
    assert caught.value.feature_id is None
    assert caught.value.safe_details == {
        "bindings": ["clip", "video_vae", "fl2va"]
    }


def _project_selection(
    draft: UnifiedTimelineDraftV5,
    feature_id: str,
    *,
    enabled: bool,
    params: dict[str, object],
) -> UnifiedTimelineDraftV5:
    document = draft.model_dump(mode="json")
    document["features"]["project"][feature_id] = {
        "enabled": enabled,
        "params": params,
    }
    return UnifiedTimelineDraftV5.model_validate(document)


def test_unambiguous_standard_ck_becomes_one_v6_project_switch() -> None:
    source = _project_selection(
        _bundle5_draft(),
        "attention_backend_override",
        enabled=True,
        params={"mode": "ck_int8"},
    )
    source_document = source.model_dump(mode="json")

    projection = project_v5_authority_to_v6(source)

    assert source.model_dump(mode="json") == source_document
    assert source.features.template_bundle_version == 5
    assert projection.draft.features.template_bundle_version == 6
    assert set(projection.draft.features.project) == {
        "lora",
        "comfy_kitchen_attention",
    }
    ck = projection.draft.features.project["comfy_kitchen_attention"]
    assert ck.enabled is True
    assert ck.params == {}
    assert projection.draft.features.by_segment == {}
    assert projection.notices == (
        "Standard CK now uses ComfyUI's official ModelAttentionBackend carrier.",
    )


def test_explicit_standard_and_raylight_ck_agree_on_v6_project_switch() -> None:
    source = _project_selection(
        _bundle5_draft(),
        "attention_backend_override",
        enabled=True,
        params={"mode": "ck_int8"},
    )
    source = _project_selection(
        source,
        "raylight_pool_intent",
        enabled=True,
        params={"attention": "ck_int8"},
    )

    projection = project_v5_authority_to_v6(source)

    assert projection.draft.features.project[
        "comfy_kitchen_attention"
    ].enabled is True


@pytest.mark.parametrize(
    ("feature_id", "params", "expected_feature"),
    (
        (
            "attention_backend_override",
            {"mode": "pytorch"},
            "attention_backend_override",
        ),
        (
            "raylight_pool_intent",
            {"attention": "ck_int8"},
            None,
        ),
        (
            "h3_low_vram_attention",
            {},
            "h3_low_vram_attention",
        ),
    ),
    ids=("explicit_pytorch", "ray_only_ck", "retired_low_vram"),
)
def test_ambiguous_attention_authority_is_left_in_bundle5(
    feature_id: str,
    params: dict[str, object],
    expected_feature: str | None,
) -> None:
    source = _project_selection(
        _bundle5_draft(),
        feature_id,
        enabled=True,
        params=params,
    )
    before = source.model_dump(mode="json")

    with pytest.raises(V5V6ProjectionError) as caught:
        project_v5_authority_to_v6(source)

    assert caught.value.code == "attention_migration_conflict"
    assert caught.value.feature_id == expected_feature
    assert source.features.template_bundle_version == 5
    assert source.model_dump(mode="json") == before


def test_conflicting_standard_ck_and_ray_flash_stays_in_bundle5() -> None:
    source = _project_selection(
        _bundle5_draft(),
        "attention_backend_override",
        enabled=True,
        params={"mode": "ck_int8"},
    )
    source = _project_selection(
        source,
        "raylight_pool_intent",
        enabled=True,
        params={"attention": "torch_flash"},
    )

    with pytest.raises(V5V6ProjectionError) as caught:
        project_v5_authority_to_v6(source)

    assert caught.value.code == "attention_migration_conflict"
    assert source.features.template_bundle_version == 5


def test_segment_attention_disagreement_is_not_folded_into_project_ck() -> None:
    source = _bundle5_draft()
    document = source.model_dump(mode="json")
    second = deepcopy(document["segments"][0])
    second["id"] = "segment-2"
    second["title"] = "Segment 2"
    document["segments"].append(second)
    first_id = document["segments"][0]["id"]
    document["features"]["by_segment"] = {
        first_id: {
            "attention_backend_override": {
                "enabled": True,
                "params": {"mode": "ck_int8"},
            }
        }
    }
    divergent = UnifiedTimelineDraftV5.model_validate(document)

    with pytest.raises(V5V6ProjectionError) as caught:
        project_v5_authority_to_v6(divergent)

    assert caught.value.code == "attention_migration_conflict"
    assert divergent.features.template_bundle_version == 5


def test_canonical_bundle6_authority_is_copied_without_reinterpretation() -> None:
    source = project_v5_authority_to_v6(_bundle5_draft()).draft
    before = source.model_dump(mode="json")

    projected = project_v5_authority_to_v6(source)

    assert projected.draft is not source
    assert projected.draft.model_dump(mode="json") == before
    assert projected.notices == ()
