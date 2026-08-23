from __future__ import annotations

from copy import deepcopy
import inspect

import pytest

from directordeck.execution.submission import (
    LockedSubmissionPlanner,
    SubmissionPlanningError,
)
from directordeck.schemas import UnifiedTimelineDraft
from directordeck.workflow.execution import (
    CompiledPlanDigest,
    ContinuityLateBindingEvidence,
    EndpointIdentity,
    LockedSegmentUnit,
    OutputDescriptor,
    PreparedControlUnit,
    RuntimeEpochLateBindingEvidence,
    compiled_execution_plan_digest,
)
from directordeck.workflow.v4_execution_adapter import compile_v4_execution_plan

from . import extensible_workflow_v0_fixture_builder as fixture_builder


ENDPOINT = EndpointIdentity(
    schema_version=1,
    endpoint_key="embedded",
    runtime_instance_id="comfy-boot-1",
)
EMPTY_LEDGER = {
    "version": 2,
    "epoch": 0,
    "current": None,
    "tail_prompt_id": None,
    "tail_action": None,
    "tainted": False,
}


def test_128_waves_reuse_one_precomputed_compiled_plan_digest(
    monkeypatch,
) -> None:
    draft_document = fixture_builder._draft("t2v").model_dump(mode="json")
    source_segment = draft_document["segments"][0]
    draft_document["segments"] = []
    for ordinal in range(128):
        segment = deepcopy(source_segment)
        segment.update(
            id=f"digest-scale-{ordinal:03d}",
            title=f"Digest scale {ordinal:03d}",
            prompt=f"Digest scale prompt {ordinal:03d}",
        )
        draft_document["segments"].append(segment)
    compiled = compile_v4_execution_plan(
        UnifiedTimelineDraft.model_validate(draft_document),
        fixture_builder._settings("standard"),
        "digest-scale",
    )
    source_digest = compiled_execution_plan_digest(compiled)
    assert isinstance(source_digest, CompiledPlanDigest)

    def unexpected_recanonicalization(*_args, **_kwargs):
        raise AssertionError("compiled plan digest was recomputed")

    monkeypatch.setattr(
        "directordeck.execution.submission.compiled_execution_plan_digest",
        unexpected_recanonicalization,
    )
    planner = _planner()
    waves = [
        planner.build_wave(
            compiled,
            source_unit_ordinal=ordinal,
            segment_child_id=f"digest-child-{ordinal:03d}",
            ray_ledger_before=EMPTY_LEDGER,
            source_compiled_plan_digest=(source_digest if ordinal == 0 else None),
        )
        for ordinal in range(128)
    ]

    assert len(waves) == 128
    assert all(wave.source_compiled_plan_digest == source_digest for wave in waves)


def _planner(*ids: str) -> LockedSubmissionPlanner:
    values = iter(ids)
    return LockedSubmissionPlanner(
        ENDPOINT,
        id_factory=lambda: next(values),
    )


def _resident_ray_ledger() -> dict[str, object]:
    plan = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "resident-ray",
    )
    wave = _planner().build_wave(
        plan,
        source_unit_ordinal=0,
        segment_child_id="resident-prompt",
        ray_ledger_before=EMPTY_LEDGER,
    )
    ledger = wave.model_dump(mode="json")["ray_ledger_after_intent"]
    assert isinstance(ledger, dict)
    return {
        **ledger,
        "tainted": False,
        "tail_terminal_certificate": {
            "prompt_id": "resident-prompt",
            "action": "ray_unit",
            "succeeded": True,
        },
    }


def test_standard_wave_materializes_only_from_prepared_plan() -> None:
    compiled = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "standard-wave",
    )
    wave = _planner().build_wave(
        compiled,
        source_unit_ordinal=0,
        segment_child_id="child-segment",
        ray_ledger_before=EMPTY_LEDGER,
    )

    assert len(wave.units) == 1
    segment = wave.units[0]
    assert isinstance(segment, LockedSegmentUnit)
    assert segment.child_id == "child-segment"
    assert segment.requested_prompt_id == "child-segment"
    assert segment.exact_prompt == segment.prompt_base
    assert segment.late_bound_values == {}
    assert wave.source_unit_id == segment.id

    snapshot = _planner().exact_snapshot(wave, segment)
    assert snapshot.exact_prompt == segment.exact_prompt
    assert snapshot.effective_execution_digest == segment.effective_execution_digest


def test_raylight_epoch_is_derived_from_locked_ledger() -> None:
    compiled = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "ray-wave",
    )
    wave = _planner().build_wave(
        compiled,
        source_unit_ordinal=0,
        segment_child_id="child-ray",
        ray_ledger_before=EMPTY_LEDGER,
    )

    segment = wave.units[0]
    assert isinstance(segment, LockedSegmentUnit)
    runtime_evidence = next(
        item
        for item in segment.late_binding_evidence
        if isinstance(item, RuntimeEpochLateBindingEvidence)
    )
    assert runtime_evidence.epoch == 1
    assert wave.ray_ledger_after_intent is not None
    assert wave.ray_ledger_after_intent["epoch"] == 1
    assert wave.ray_ledger_after_intent["tail_prompt_id"] == "child-ray"


def test_planner_materializes_raylight_kill_and_clean_continuation() -> None:
    compiled = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "standard-after-ray",
    )
    before = _resident_ray_ledger()
    wave = _planner("control-child").build_wave(
        compiled,
        source_unit_ordinal=0,
        segment_child_id="segment-child",
        ray_ledger_before=before,
    )

    assert [type(unit) for unit in wave.units] == [
        PreparedControlUnit,
        LockedSegmentUnit,
    ]
    control, segment = wave.units
    assert control.child_id == "control-child"
    assert control.requested_prompt_id == "control-child"
    assert wave.ray_ledger_after_intent["tail_action"] == "shutdown"

    clean = {
        "version": 2,
        "epoch": before["epoch"],
        "current": None,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": False,
    }
    continuation = _planner().segment_continuation(
        wave,
        ray_ledger_before=clean,
    )
    assert continuation.units == (segment,)
    assert continuation.ray_ledger_after_intent == clean


def test_same_run_continuity_is_materialized_from_typed_output_evidence() -> None:
    compiled = compile_v4_execution_plan(
        fixture_builder._continuity_draft(),
        fixture_builder._settings("standard"),
        "continuity-wave",
    )
    dependency = compiled.segment_units[1].continuity_dependency
    assert dependency is not None
    evidence = ContinuityLateBindingEvidence(
        input_pointer=str(dependency["input_pointer"]),
        predecessor_segment_id=str(dependency["predecessor_segment_id"]),
        dependency_source="same_run",
        output=OutputDescriptor(
            filename="take.mp4",
            subfolder="director/continuity-wave",
        ),
    )
    wave = _planner().build_wave(
        compiled,
        source_unit_ordinal=1,
        segment_child_id="child-successor",
        continuity_evidence=(evidence,),
        ray_ledger_before=EMPTY_LEDGER,
    )

    segment = wave.units[0]
    assert isinstance(segment, LockedSegmentUnit)
    assert segment.late_bound_values[evidence.input_pointer] == (
        "director/continuity-wave/take.mp4 [output]"
    )


def test_missing_or_forged_continuity_evidence_fails_closed() -> None:
    compiled = compile_v4_execution_plan(
        fixture_builder._continuity_draft(),
        fixture_builder._settings("standard"),
        "forged-continuity",
    )
    dependency = compiled.segment_units[1].continuity_dependency
    assert dependency is not None

    with pytest.raises(SubmissionPlanningError, match="exactly cover"):
        _planner().build_wave(
            compiled,
            source_unit_ordinal=1,
            segment_child_id="missing-evidence",
            ray_ledger_before=EMPTY_LEDGER,
        )

    forged = ContinuityLateBindingEvidence(
        input_pointer=str(dependency["input_pointer"]),
        predecessor_segment_id="forged-predecessor",
        dependency_source="same_run",
        output=OutputDescriptor(filename="take.mp4"),
    )
    with pytest.raises(SubmissionPlanningError, match="compiled dependency"):
        _planner().build_wave(
            compiled,
            source_unit_ordinal=1,
            segment_child_id="forged-evidence",
            continuity_evidence=(forged,),
            ray_ledger_before=EMPTY_LEDGER,
        )


def test_callers_cannot_supply_runtime_epoch_or_exact_native_prompt() -> None:
    parameters = inspect.signature(LockedSubmissionPlanner.build_wave).parameters
    assert "exact_segment" not in parameters
    assert "late_binding_evidence" not in parameters
    assert "runtime_descriptor" not in parameters
    assert "ray_ledger_after_intent" not in parameters

    compiled = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "forged-runtime-evidence",
    )
    pointer = next(
        declaration.input_pointer
        for declaration in compiled.segment_units[0]
        .graph_audit_spec.allowed_late_bound_inputs
        if declaration.source_kind == "runtime_epoch"
    )
    with pytest.raises(SubmissionPlanningError, match="only typed continuity"):
        _planner().build_wave(
            compiled,
            source_unit_ordinal=0,
            segment_child_id="ray-child",
            continuity_evidence=(
                RuntimeEpochLateBindingEvidence(
                    input_pointer=pointer,
                    epoch=999,
                ),
            ),  # type: ignore[arg-type]
            ray_ledger_before=EMPTY_LEDGER,
        )


def test_tainted_ledger_without_descriptor_fails_closed() -> None:
    compiled = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "tainted-without-runtime",
    )
    with pytest.raises(SubmissionPlanningError, match="no descriptor"):
        _planner().build_wave(
            compiled,
            source_unit_ordinal=0,
            segment_child_id="standard-child",
            ray_ledger_before={**EMPTY_LEDGER, "tainted": True},
        )


def test_endpoint_mismatch_is_rejected() -> None:
    compiled = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "wrong-endpoint",
    )
    planner = LockedSubmissionPlanner(
        ENDPOINT.model_copy(update={"endpoint_key": "remote"})
    )
    with pytest.raises(SubmissionPlanningError, match="different endpoint"):
        planner.build_wave(
            compiled,
            source_unit_ordinal=0,
            segment_child_id="child",
            ray_ledger_before=EMPTY_LEDGER,
        )
