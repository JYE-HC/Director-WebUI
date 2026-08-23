from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import inspect
from typing import Any

import pytest

from directordeck.native_templates import (
    NativeHistoricalTake,
    raylight_runtime_descriptor,
)
from directordeck.schemas import RuntimeSettings, UnifiedTimelineDraft
from directordeck.workflow.execution import CompiledExecutionPlan
from directordeck.workflow.interpreters.builtin import V4BuiltinInterpreter
from directordeck.workflow.v4_compiler import compile_v4_timeline
from directordeck.workflow.v4_execution_adapter import (
    V4ExecutionAdapterError,
    adapt_v4_compile_result,
    compile_v4_execution_plan,
)

from . import extensible_workflow_v0_fixture_builder as fixture_builder


def _class_type(prompt: Mapping[str, Any], node_id: str) -> str:
    node = prompt[node_id]
    assert isinstance(node, Mapping)
    class_type = node["class_type"]
    assert isinstance(class_type, str)
    return class_type


def test_compile_wrapper_invokes_v4_compiler_once_and_returns_plan_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import directordeck.workflow.v4_execution_adapter as adapter

    calls = 0
    actual_compile = adapter.compile_v4_timeline

    def counted_compile(*args: Any, **kwargs: Any):
        nonlocal calls
        calls += 1
        return actual_compile(*args, **kwargs)

    monkeypatch.setattr(adapter, "compile_v4_timeline", counted_compile)
    result = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "one-compile",
    )

    assert calls == 1
    assert isinstance(result, CompiledExecutionPlan)
    assert not hasattr(result, "native_result")
    assert not hasattr(result, "execution_plan")


def test_compile_report_freezes_actual_resolutions_and_emission_notices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_emit = V4BuiltinInterpreter.emit

    def emit_with_notice(self: V4BuiltinInterpreter, *args: Any, **kwargs: Any):
        emission = original_emit(self, *args, **kwargs)
        return emission.model_copy(
            update={
                "notices": (
                    ("Save output uses the frozen test notice.",)
                    if self.id == "save_take"
                    else emission.notices
                )
            }
        )

    monkeypatch.setattr(V4BuiltinInterpreter, "emit", emit_with_notice)
    plan = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "compile-evidence",
    )
    report = plan.model_dump(mode="json")["compile_report"]

    assert any(
        item["resolution"]["state"] == "noop"
        and item["resolution"]["resolution_details"] == {
            "reason": "disabled_by_v4_context"
        }
        for item in report["feature_resolutions"]
    )
    assert report["notices"] == [
        {
            "segment_id": "baseline-t2v",
            "unit_id": "standard-fl2va-000",
            "feature_id": "save_take",
            "message": "Save output uses the frozen test notice.",
        }
    ]


def test_production_submission_source_has_no_native_result_dependency() -> None:
    from directordeck.app import _create_timeline_job_impl

    source = inspect.getsource(_create_timeline_job_impl)
    assert "native_result" not in source
    assert "NativeWorkflowUnit" not in source
    assert "bind_native_workflow_predecessor_output" not in source
    assert "bind_raylight_runtime_epoch" not in source
    assert "raylight_runtime_descriptor" not in source


def test_standard_v4_result_derives_complete_prepared_segment_contract() -> None:
    draft = fixture_builder._draft("t2v")
    result = compile_v4_execution_plan(
        draft,
        fixture_builder._settings("standard"),
        "prepared-standard",
    )

    plan = result
    native = compile_v4_timeline(
        draft,
        fixture_builder._settings("standard"),
        "prepared-standard",
    )
    assert isinstance(plan, CompiledExecutionPlan)
    assert plan.version == 2
    assert plan.template_bundle_version == 4
    plan_document = plan.model_dump(mode="json")
    assert plan_document["node_policy"] == native.node_policy
    assert plan_document["compile_report"]["manifest"] == native.manifest
    assert plan_document["compile_report"]["source"] == (
        "v4_native_compile_adapter_v2"
    )
    assert plan_document["compile_report"]["feature_resolutions"]
    assert plan.effective_execution_digest.algorithm == "sha256-canonical-json-v1"
    assert plan.plan_execution_digest == plan.effective_execution_digest

    unit = plan.segment_units[0]
    native_unit = native.workflows[0]
    assert native_unit.progress_spec is None
    assert native_unit.preview_spec is None
    assert unit.id == native_unit.id
    assert unit.owner_segment_id == draft.segments[0].id
    assert unit.template_id == "h3_standard_segment"
    assert unit.template_revision == 1
    assert unit.model_dump(mode="json")["prompt_base"] == native_unit.prompt
    assert unit.graph_audit_spec == native_unit.graph_audit_spec
    assert unit.continuity_dependency is None
    assert unit.expected_output_spec.model_dump(mode="json") == {
        "width": draft.render.width,
        "height": draft.render.height,
        "fps": draft.render.fps,
        "visible_frame_count": 124,
        "expected_audio_mode": "generated",
        "segment_id": draft.segments[0].id,
        "node_id": native_unit.output_nodes[draft.segments[0].id],
        "kind": "video",
        "role": "take",
    }
    assert [phase.id for phase in unit.progress_spec.phases] == [
        "sampling",
        "decode_video",
        "assemble_media",
        "persist_take",
    ]
    assert [
        _class_type(unit.prompt_base, phase.node_id)
        for phase in unit.progress_spec.phases
    ] == [
        "SamplerCustomAdvanced",
        "VAEDecode",
        "CreateVideo",
        "SaveVideo",
    ]
    assert sum(phase.weight for phase in unit.progress_spec.phases) == pytest.approx(
        1.0
    )
    assert len(unit.preview_spec.sources) == 1
    assert _class_type(
        unit.prompt_base,
        unit.preview_spec.sources[0].node_id,
    ) == "SamplerCustomAdvanced"
    assert unit.preview_spec.sources[0].phase_id == "sampling"
    assert unit.runtime_requirements.model_dump(mode="json") == {
        "endpoint_key": "embedded",
        "backend": "standard",
        "logical_gpu_indices": [],
        "ray_compatibility_key": None,
        "ray_runtime_key": None,
        "requires_standard_driver_access": True,
        "expected_residency_policy": None,
    }
    assert unit.runtime_pool_identity is None
    assert unit.effective_execution_digest.algorithm == (
        "sha256-canonical-json-v1"
    )


@pytest.mark.parametrize(
    ("recipe", "audio_mode", "expected"),
    (
        ("r2v", "generate", "generated"),
        ("v2v", "source", "source"),
        ("t2v", "mute", "none"),
    ),
)
def test_expected_output_audio_mode_is_a_compile_expectation(
    recipe: str,
    audio_mode: str,
    expected: str,
) -> None:
    result = compile_v4_execution_plan(
        fixture_builder._draft(recipe, audio_mode=audio_mode),
        fixture_builder._settings("standard"),
        f"audio-{audio_mode}",
    )

    assert (
        result.segment_units[0].expected_output_spec.expected_audio_mode
        == expected
    )


def test_raylight_v4_result_reuses_exact_runtime_descriptor_identity() -> None:
    result = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "prepared-raylight",
        endpoint_key="embedded",
    )
    native = compile_v4_timeline(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "prepared-raylight",
    ).workflows[0]
    descriptor = raylight_runtime_descriptor(native)
    assert descriptor is not None

    unit = result.segment_units[0]
    requirements = unit.runtime_requirements
    assert requirements.logical_gpu_indices == (0, 1)
    assert requirements.ray_compatibility_key == descriptor["compatibility_key"]
    assert requirements.ray_runtime_key == descriptor["runtime_key"]
    assert requirements.expected_residency_policy == "keep_until_switch"
    assert requirements.requires_standard_driver_access is False
    assert unit.runtime_pool_identity is not None
    assert unit.runtime_pool_identity.placement == {
        "logical_gpu_indices": (0, 1)
    }
    assert [
        contribution.feature
        for contribution in unit.runtime_pool_identity.active_feature_pool_identities
    ] == [
        "raylight_pool_intent@1",
        "raylight_sigma_shift@1",
    ]
    assert _class_type(
        unit.prompt_base,
        unit.preview_spec.sources[0].node_id,
    ) == "DirectorDeckRayXFuserSamplerCustomAdvanced"


def test_plan_digest_excludes_job_output_prefix_but_tracks_sampling() -> None:
    draft = fixture_builder._draft("t2v")
    settings = fixture_builder._settings("standard")
    first = compile_v4_execution_plan(draft, settings, "alpha-job")
    second = compile_v4_execution_plan(draft, settings, "omega-job")

    first_prompt = first.segment_units[0].prompt_base
    second_prompt = second.segment_units[0].prompt_base
    assert first_prompt != second_prompt
    assert (
        first.effective_execution_digest == second.effective_execution_digest
    )

    changed_document = draft.model_dump(mode="json")
    changed_document["sampling"]["fl2va"]["shift"] = 8.0
    changed = compile_v4_execution_plan(
        UnifiedTimelineDraft.model_validate(changed_document),
        settings,
        "digest-job-c",
    )
    assert (
        changed.effective_execution_digest != first.effective_execution_digest
    )


def test_raylight_gpu_placement_changes_unit_and_plan_execution_digests() -> None:
    draft = fixture_builder._draft("t2v")
    settings = fixture_builder._settings("raylight")
    baseline = compile_v4_execution_plan(draft, settings, "placement-baseline")
    changed_document = settings.model_dump(mode="json")
    changed_document["models"]["fl2va"]["raylight"]["gpu_select"] = [1, 2]
    changed = compile_v4_execution_plan(
        draft,
        RuntimeSettings.model_validate(changed_document),
        "placement-baseline",
    )

    baseline_unit = baseline.segment_units[0]
    changed_unit = changed.segment_units[0]
    assert baseline_unit.runtime_pool_identity != changed_unit.runtime_pool_identity
    assert baseline_unit.effective_execution_digest != (
        changed_unit.effective_execution_digest
    )
    assert baseline.plan_execution_digest != changed.plan_execution_digest


def test_same_run_and_historical_continuity_are_projected_without_recompile() -> None:
    draft = fixture_builder._continuity_draft()
    settings = fixture_builder._settings("standard")
    same_run = compile_v4_execution_plan(
        draft,
        settings,
        "continuity-same-run",
    )
    assert len(same_run.segment_units) == 2
    dependency = same_run.segment_units[1].continuity_dependency
    assert dependency is not None
    assert dependency["source"] == "same_run"
    assert dependency["resolved"] is False
    assert dependency["bound_file"] is None

    take = NativeHistoricalTake(
        id="historical-take-1",
        segment_id="baseline-continuity-first",
        output={
            "filename": "predecessor.mp4",
            "subfolder": "director-test",
            "type": "output",
        },
    )
    historical = compile_v4_execution_plan(
        draft,
        settings,
        "continuity-historical",
        ["baseline-continuity-second"],
        historical_takes={"baseline-continuity-second": take},
    )
    historical_unit = historical.segment_units[0]
    historical_dependency = historical_unit.continuity_dependency
    assert historical_dependency is not None
    assert historical_dependency["source"] == "historical_take"
    assert historical_dependency["resolved"] is True
    assert historical_dependency["bound_file"] == (
        "director-test/predecessor.mp4 [output]"
    )
    pointer = str(historical_dependency["input_pointer"])
    node_id, _, input_name = pointer.strip("/").split("/")
    assert historical_unit.prompt_base[node_id]["inputs"][input_name] == (
        "director-test/predecessor.mp4 [output]"
    )


def test_adapter_fails_closed_when_native_plan_does_not_match_captured_draft() -> None:
    draft = fixture_builder._draft("t2v")
    settings = fixture_builder._settings("standard")
    native = compile_v4_timeline(draft, settings, "mismatched-plan")
    tampered_plan = dict(native.plans[0])
    tampered_plan["visible_frame_count"] = 121
    tampered = replace(native, plans=(tampered_plan,))

    with pytest.raises(V4ExecutionAdapterError, match="visible_frame_count"):
        adapt_v4_compile_result(
            tampered,
            draft=draft,
            captured_settings=settings,
        )


def test_adapter_rejects_partial_explicit_execution_authority() -> None:
    draft = fixture_builder._draft("t2v")
    settings = fixture_builder._settings("standard")
    native = compile_v4_timeline(draft, settings, "partial-execution-spec")
    prepared = compile_v4_execution_plan(
        draft,
        settings,
        "partial-execution-spec",
    ).segment_units[0]

    for update in (
        {"progress_spec": prepared.progress_spec},
        {"preview_spec": prepared.preview_spec},
    ):
        tampered = replace(
            native,
            workflows=(replace(native.workflows[0], **update),),
        )
        with pytest.raises(V4ExecutionAdapterError, match="incomplete explicit"):
            adapt_v4_compile_result(
                tampered,
                draft=draft,
                captured_settings=settings,
            )


def test_adapter_rejects_different_captured_settings_backend() -> None:
    draft = fixture_builder._draft("t2v")
    native = compile_v4_timeline(
        draft,
        fixture_builder._settings("standard"),
        "settings-mismatch",
    )
    ray_settings_document = fixture_builder._settings("raylight").model_dump(
        mode="json"
    )
    ray_settings = RuntimeSettings.model_validate(ray_settings_document)

    with pytest.raises(V4ExecutionAdapterError, match="captured settings"):
        adapt_v4_compile_result(
            native,
            draft=draft,
            captured_settings=ray_settings,
        )
