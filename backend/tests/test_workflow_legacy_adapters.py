from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path
from shutil import copyfileobj
from typing import Any

import pytest
from pydantic import ValidationError

from directordeck.app import _job_read
from directordeck.database import Database
from directordeck.native_templates import (
    NativeCompileResult,
    NativeContinuityDependency,
    NativeWorkflowUnit,
)
from directordeck.workflow.legacy import (
    LegacyExecutionRowsEnvelope,
    LegacyAvailableContract,
    LegacyJobChildRowEnvelope,
    LegacyJobRowEnvelope,
    LegacyNativeCompileSnapshot,
    LegacyNativeWorkflowUnitSnapshot,
    LegacyPublicJobEnvelope,
    LegacyRayLedgerEnvelope,
    LegacySegmentTakeRowEnvelope,
    LegacyStage1AdapterResult,
    LegacyUnavailableContract,
    adapt_legacy_stage1_sources,
    classify_legacy_prompt_snapshot,
    freeze_legacy_execution_rows,
    freeze_legacy_job_child_row,
    freeze_legacy_job_row,
    freeze_legacy_public_job_from_rows,
    freeze_legacy_ray_ledger,
    freeze_legacy_segment_take_row,
    freeze_native_compile_result,
    freeze_native_workflow_unit,
    legacy_compile_public_read,
    legacy_output_locators_from_child_row,
    legacy_ray_ledger_projection,
    legacy_row_projection,
    native_compile_result_projection,
    native_workflow_unit_projection,
    restore_legacy_public_job,
    restore_native_compile_result,
    restore_native_workflow_unit,
)

from .extensible_workflow_v0_fixture_builder import (
    FIXTURE_DIR,
    build_native_prompt_goldens,
    database_projection,
    sha256_file,
)


def _native_unit(document: dict[str, Any]) -> NativeWorkflowUnit:
    continuity_document = document["continuity"]
    continuity = (
        NativeContinuityDependency(**continuity_document)
        if continuity_document is not None
        else None
    )
    return NativeWorkflowUnit(
        id=document["id"],
        family=document["family"],
        backend=document["backend"],
        segment_ids=tuple(document["segment_ids"]),
        prompt=deepcopy(document["prompt"]),
        output_nodes=dict(document["output_nodes"]),
        continuity=continuity,
    )


def _native_result(case: dict[str, Any]) -> NativeCompileResult:
    return NativeCompileResult(
        workflows=tuple(_native_unit(unit) for unit in case["units"]),
        manifest=deepcopy(case["manifest"]),
        plans=tuple(deepcopy(case["plans"])),
        families=tuple(case["families"]),
        node_policy=deepcopy(case["node_policy"]),
    )


def _expected_unit_projection(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": document["id"],
        "family": document["family"],
        "backend": document["backend"],
        "segment_ids": document["segment_ids"],
        "prompt": document["prompt"],
        "output_nodes": document["output_nodes"],
        "continuity": document["continuity"],
    }


def test_all_26_current_cases_freeze_and_restore_without_prompt_differences() -> None:
    # Stage 7 deliberately changes the four LoRA manifests/graphs; the phase-0
    # fixture remains immutable and its bounded deltas are checked by the
    # baseline test. Legacy freeze/restore must preserve the current result.
    current = build_native_prompt_goldens()
    assert len(current["cases"]) == 26

    for case in current["cases"]:
        for unit_document in case["units"]:
            unit = _native_unit(unit_document)
            frozen = freeze_native_workflow_unit(unit)
            round_tripped = LegacyNativeWorkflowUnitSnapshot.model_validate_json(
                frozen.model_dump_json()
            )
            assert native_workflow_unit_projection(round_tripped) == (
                _expected_unit_projection(unit_document)
            )
            restored = restore_native_workflow_unit(round_tripped)
            assert restored == unit
            assert list(restored.prompt) == unit_document["node_ids"]

        if case["kind"] != "segment_compile":
            continue
        result = _native_result(case)
        frozen_result = freeze_native_compile_result(result)
        round_tripped_result = LegacyNativeCompileSnapshot.model_validate_json(
            frozen_result.model_dump_json()
        )
        projection = native_compile_result_projection(round_tripped_result)
        assert projection == {
            "workflows": [
                _expected_unit_projection(unit) for unit in case["units"]
            ],
            "manifest": case["manifest"],
            "plans": case["plans"],
            "families": case["families"],
            "node_policy": case["node_policy"],
        }
        assert restore_native_compile_result(round_tripped_result) == result
        assert legacy_compile_public_read(round_tripped_result).model_dump(
            mode="json"
        ) == {
            "execution_strategy": "native_segment_graph_v1",
            "model_families": case["families"],
            "plans": case["plans"],
            "node_policy": case["node_policy"],
        }


@pytest.mark.parametrize(
    ("document", "expected_kind"),
    (
        (None, "absent"),
        ({"1": {"class_type": "SaveVideo", "inputs": {}}}, "native_prompt"),
        (
            {
                "graph_source": "server",
                "units": [],
                "submission_order": [],
            },
            "compile_manifest",
        ),
        (
            {
                "graph_source": "server",
                "units": [],
                "submission_order": [],
                "runtime_epoch": 4,
            },
            "planned_manifest",
        ),
        ({"unexpected": True}, "unknown"),
        (["not", "an", "object"], "unknown"),
    ),
)
def test_prompt_snapshot_classification_is_shape_based_and_round_trips(
    document: Any,
    expected_kind: str,
) -> None:
    classified = classify_legacy_prompt_snapshot(document)
    assert classified.kind == expected_kind
    assert type(classified).model_validate_json(classified.model_dump_json()) == classified


def test_output_locators_fail_closed_without_inventing_expected_outputs() -> None:
    locators, gaps = legacy_output_locators_from_child_row(
        {
            "segment_ids": ["segment-a", "segment-b"],
            "output_nodes": {"segment-a": "41", "segment-b": "42"},
        }
    )
    assert [locator.model_dump(mode="json") for locator in locators] == [
        {"segment_id": "segment-a", "node_id": "41"},
        {"segment_id": "segment-b", "node_id": "42"},
    ]
    assert gaps == ()

    locators, gaps = legacy_output_locators_from_child_row(
        {
            "segment_ids": ["segment-a", "segment-b"],
            "output_nodes": {"segment-a": "41", "segment-b": "41"},
        }
    )
    assert locators == ()
    assert {gap.reason for gap in gaps} == {"ambiguous_legacy_shape"}

    locators, gaps = legacy_output_locators_from_child_row(
        {
            "segment_ids": ["segment-a", "segment-b"],
            "output_nodes": {"segment-a": "41"},
        }
    )
    assert [locator.segment_id for locator in locators] == ["segment-a"]
    assert any(gap.field_path == "output_locators" for gap in gaps)

    assert legacy_output_locators_from_child_row(
        {"segment_ids": [], "output_nodes": {}}
    ) == ((), ())


def _copy_database_fixture(tmp_path: Path) -> tuple[Path, str]:
    source = FIXTURE_DIR / "current_v4.sqlite3.gz"
    digest = sha256_file(source)
    working = tmp_path / "current_v4.sqlite3"
    with gzip.open(source, "rb") as archived, working.open("wb") as unpacked:
        copyfileobj(archived, unpacked)
    return working, digest


def test_phase0_database_rows_are_wrapped_without_mutation_or_false_evidence(
    tmp_path: Path,
) -> None:
    working, source_digest = _copy_database_fixture(tmp_path)
    expected_projection = json.loads(
        (FIXTURE_DIR / "current_v4_expected.json").read_text(encoding="utf-8")
    )
    # First prove the immutable phase-0 fixture still has its exact checked-in
    # v4 bytes. Opening it through Database now intentionally performs the
    # Stage-6 atomic authority migration, so post-initialize equality to this
    # v4 projection would assert a retired behavior.
    assert database_projection(working) == expected_projection

    database = Database(working)
    database.initialize()
    projection_before = database_projection(working)
    assert projection_before["unified_timeline"][0]["document"]["version"] == 5
    assert projection_before["settings"][0]["document"]["schema_version"] == 3

    job = database.get_job("baseline-parent-v0")
    assert job is not None
    children = database.list_job_children(job["id"])
    assert len(children) == 1
    takes = projection_before["segment_takes"]
    ray_ledger = database.get_raylight_runtime_state()
    assert ray_ledger is not None
    originals = deepcopy((job, children, takes, ray_ledger))

    frozen = freeze_legacy_execution_rows(
        job=job,
        children=children,
        takes=takes,
        ray_ledger=ray_ledger,
    )
    round_tripped = LegacyExecutionRowsEnvelope.model_validate_json(
        frozen.model_dump_json()
    )
    assert legacy_row_projection(round_tripped.job) == job
    assert [
        legacy_row_projection(child) for child in round_tripped.children
    ] == children
    assert [legacy_row_projection(take) for take in round_tripped.takes] == takes
    assert legacy_ray_ledger_projection(round_tripped.ray_ledger) == ray_ledger
    assert (job, children, takes, ray_ledger) == originals

    assert round_tripped.job.prompt_snapshot.kind == "native_prompt"
    child = round_tripped.children[0]
    assert child.prompt_snapshot.kind == "native_prompt"
    assert [locator.model_dump(mode="json") for locator in child.output_locators] == [
        {
            "segment_id": "baseline-db-segment",
            "node_id": children[0]["output_nodes"]["baseline-db-segment"],
        }
    ]
    child_missing = {item.field_path for item in child.unavailable_evidence}
    assert {
        "exact_prompt_snapshot.template_id",
        "exact_prompt_snapshot.progress_spec",
        "prompt_ownership.requested_prompt_id",
        "prompt_ownership.actual_prompt_id",
        "prompt_ownership.cleanup_certificate",
    } <= child_missing

    take = round_tripped.takes[0]
    assert take.observed_artifact is None
    assert take.legacy_has_audio is True
    assert take.legacy_has_audio_provenance == "authored_audio_mode_inference"
    observed_missing = {
        item.field_path for item in take.unavailable_evidence
    }
    assert {
        "observed_artifact.width",
        "observed_artifact.height",
        "observed_artifact.fps",
        "observed_artifact.frame_count",
        "observed_artifact.duration_seconds",
        "observed_artifact.has_audio",
        "observed_artifact.media_probe_version",
    } <= observed_missing
    assert {
        "locked_submission_plan.ray_ledger_before",
        "locked_submission_plan.ray_ledger_after_intent",
    } <= {
        item.field_path for item in round_tripped.ray_ledger.unavailable_evidence
    }

    assert database_projection(working) == projection_before
    assert sha256_file(FIXTURE_DIR / "current_v4.sqlite3.gz") == source_digest


def test_each_database_wrapper_has_independent_json_round_trip() -> None:
    job = freeze_legacy_job_row(
        {"id": "job", "prompt_snapshot": None, "settings_snapshot": {}}
    )
    child = freeze_legacy_job_child_row(
        {
            "id": "child",
            "segment_ids": ["segment"],
            "output_nodes": {"segment": "9"},
            "prompt_snapshot": {"9": {"class_type": "SaveVideo", "inputs": {}}},
        }
    )
    take = freeze_legacy_segment_take_row(
        {
            "segment_id": "segment",
            "output_descriptor": {
                "filename": "take.mp4",
                "subfolder": "video",
                "type": "output",
            },
            "has_audio": 0,
        }
    )
    ledger = freeze_legacy_ray_ledger(None)

    assert LegacyJobRowEnvelope.model_validate_json(job.model_dump_json()) == job
    assert LegacyJobChildRowEnvelope.model_validate_json(
        child.model_dump_json()
    ) == child
    assert LegacySegmentTakeRowEnvelope.model_validate_json(
        take.model_dump_json()
    ) == take
    assert LegacyRayLedgerEnvelope.model_validate_json(
        ledger.model_dump_json()
    ) == ledger
    assert take.observed_artifact is None
    assert take.validated_output_descriptor is not None
    assert take.validated_output_descriptor.model_dump(mode="json") == {
        "filename": "take.mp4",
        "subfolder": "video",
        "type": "output",
    }
    assert take.legacy_has_audio is False
    assert take.legacy_has_audio_provenance == "authored_audio_mode_inference"
    assert ledger.ledger is None
    assert any(
        item.field_path == "ray_runtime_ledger"
        for item in ledger.unavailable_evidence
    )


def test_legacy_json_coercion_rejects_cycles_without_recursion_overflow() -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="reference cycles"):
        freeze_legacy_job_row(
            {"id": "job", "prompt_snapshot": cyclic, "settings_snapshot": {}}
        )


def test_take_descriptor_is_only_exposed_after_current_path_safety_validation() -> None:
    row = {
        "id": "unsafe-take",
        "output_descriptor": {
            "filename": "../outside.mp4",
            "subfolder": "../private",
            "type": "output",
        },
        "has_audio": 1,
    }
    frozen = freeze_legacy_segment_take_row(row)
    assert frozen.validated_output_descriptor is None
    assert frozen.observed_artifact is None
    assert any(
        item.field_path == "validated_output_descriptor"
        and item.reason == "ambiguous_legacy_shape"
        for item in frozen.unavailable_evidence
    )
    assert legacy_row_projection(frozen) == row


def test_unified_stage1_result_covers_every_contract_without_false_evidence(
    tmp_path: Path,
) -> None:
    working, _ = _copy_database_fixture(tmp_path)
    database = Database(working)
    database.initialize()
    job = database.get_job("baseline-parent-v0")
    assert job is not None
    children = database.list_job_children(job["id"])
    takes = database_projection(working)["segment_takes"]
    ray_ledger = database.get_raylight_runtime_state()
    assert ray_ledger is not None
    golden = json.loads(
        (FIXTURE_DIR / "native_prompt_goldens.json").read_text(encoding="utf-8")
    )
    compile_case = next(
        case for case in golden["cases"] if case["id"] == "standard-t2v"
    )

    adapted = adapt_legacy_stage1_sources(
        _native_result(compile_case),
        job=job,
        children=children,
        takes=takes,
        ray_ledger=ray_ledger,
    )
    round_tripped = LegacyStage1AdapterResult.model_validate_json(
        adapted.model_dump_json()
    )
    assert round_tripped == adapted
    assert round_tripped.node_policy_snapshot.allowed_nodes == tuple(
        compile_case["node_policy"]["allowed_nodes"]
    )

    for field in (
        "template_bundle",
        "node_contract_registry",
        "host_capability_snapshot",
        "resource_pool",
        "published_value_refs",
        "graph_audit_specs",
        "runtime_effect_contracts",
        "runtime_requirements",
        "expected_output_specs",
        "observed_artifact_specs",
        "progress_specs",
        "preview_specs",
        "prepared_segment_units",
        "prepared_control_units",
        "compiled_execution_plan",
        "locked_submission_plan",
        "exact_prompt_snapshots",
        "prompt_ownership",
        "segment_execution_identities",
        "historical_take_geometry_identities",
        "ray_runtime_identities",
        "effective_execution_digest",
    ):
        slot = getattr(round_tripped, field)
        assert isinstance(slot, LegacyUnavailableContract), field
        assert slot.evidence, field

    assert isinstance(
        round_tripped.source_document_digest, LegacyAvailableContract
    )
    assert round_tripped.source_document_digest.value.algorithm == (
        "sha256-canonical-json-v1"
    )
    assert isinstance(
        round_tripped.comfy_node_cache_identities, LegacyAvailableContract
    )
    assert len(round_tripped.comfy_node_cache_identities.value) == 1
    assert isinstance(
        round_tripped.legacy_output_locators, LegacyAvailableContract
    )
    assert isinstance(
        round_tripped.validated_output_descriptors, LegacyAvailableContract
    )
    assert (
        round_tripped.validated_output_descriptors.value[0]
        .descriptor.filename
        == "baseline-db-segment.mp4"
    )
    assert round_tripped.database_rows.takes[0].observed_artifact is None


_PUBLIC_JOB_ORACLE = {
    "id": "baseline-parent-v0",
    "mode": "timeline",
    "status": "succeeded",
    "display_name": "Extensible workflow v0 database",
    "project_title": "Extensible workflow v0 database",
    "project_id": "default",
    "current_project": False,
    "progress": 1.0,
    "stage": "completed",
    "prompt_id": "caller-assigned-prompt-v0",
    "outputs": [],
    "output_files": [],
    "error": None,
    "preview_url": None,
    "created_at": "2026-08-21T12:00:00+00:00",
    "updated_at": "2026-08-21T12:00:00+00:00",
    "started_at": "2026-08-21T12:00:00+00:00",
    "completed_at": "2026-08-21T12:00:00+00:00",
    "execution_duration_seconds": 0.0,
    "output_count": 1,
    "error_summary": None,
    "children": [
        {
            "id": "baseline-child-v0",
            "family": "fl2va",
            "backend": "standard",
            "segment_ids": ["baseline-db-segment"],
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": "caller-assigned-prompt-v0",
            "outputs": ["output/director-baseline/baseline-db-segment.mp4"],
            "error": None,
        }
    ],
    "segment_results": [
        {
            "segment_id": "baseline-db-segment",
            "child_id": "baseline-child-v0",
            "output_url": (
                "/api/jobs/baseline-parent-v0/segment-output"
                "?segment_id=baseline-db-segment"
            ),
            "output_file": "output/director-baseline/baseline-db-segment.mp4",
            "current_snapshot": False,
        }
    ],
    "live_preview_url": None,
}


def test_public_job_envelope_projects_db_rows_against_independent_golden_oracle(
    tmp_path: Path,
) -> None:
    working, _ = _copy_database_fixture(tmp_path)
    database = Database(working)
    database.initialize()
    job = database.get_job("baseline-parent-v0")
    assert job is not None
    children = database.list_job_children(job["id"])
    takes = database_projection(working)["segment_takes"]
    control = {
        **deepcopy(children[0]),
        "id": "baseline-ray-control-v0",
        "group_index": 0,
        "segment_ids": [],
        "output_nodes": {},
        "prompt_id": "baseline-ray-control-prompt-v0",
        "outputs": [],
        "prompt_snapshot": {
            "1": {"class_type": "RayKill", "inputs": {"ray_actors": ["0", 0]}}
        },
    }
    source_children = [control, *children]

    def current_projector(
        source_job: dict[str, Any],
        projected_children: tuple[dict[str, Any], ...],
        projected_takes: tuple[dict[str, Any], ...],
    ) -> Any:
        assert projected_takes
        source_job["children"] = list(projected_children)
        return _job_read(
            source_job,
            live_preview_available=False,
            current_snapshot=False,
            current_project=False,
        )

    envelope = freeze_legacy_public_job_from_rows(
        job=job,
        children=source_children,
        takes=takes,
        projector=current_projector,
    )
    round_tripped = LegacyPublicJobEnvelope.model_validate_json(
        envelope.model_dump_json()
    )
    restored = restore_legacy_public_job(round_tripped)
    assert restored.model_dump(mode="json") == _PUBLIC_JOB_ORACLE
    assert len(round_tripped.source_children) == 2
    assert len(restored.children) == 1
    assert restored.children[0].id == "baseline-child-v0"
    assert restored.output_count == 1
    assert len(restored.segment_results) == 1

    with pytest.raises(TypeError):
        round_tripped.public_job["status"] = "failed"  # type: ignore[index]
    public_children = round_tripped.public_job["children"]
    assert isinstance(public_children, tuple)
    with pytest.raises(AttributeError):
        public_children.append({})  # type: ignore[union-attr]
    with pytest.raises(TypeError):
        public_children[0]["status"] = "failed"  # type: ignore[index]

    children[0]["outputs"][0]["filename"] = "mutated-after-freeze.mp4"
    assert restore_legacy_public_job(round_tripped).model_dump(
        mode="json"
    ) == _PUBLIC_JOB_ORACLE

    invalid = freeze_legacy_public_job_from_rows(
        job=job,
        children=source_children,
        takes=takes,
        projector=lambda _job, _children, _takes: {"id": "incomplete"},
    )
    assert invalid.public_job["id"] == "incomplete"
    with pytest.raises(ValidationError):
        restore_legacy_public_job(invalid)
