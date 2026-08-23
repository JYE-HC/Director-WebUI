from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
from functools import lru_cache
import json
import sqlite3

import pytest

from directordeck.database import (
    Database,
    ExecutionEvidenceConflict,
    RayRuntimeIntentConflict,
)
from directordeck.compiler import align_h3_frames, timeline_segment_take_fingerprint
from directordeck.execution.submission import LockedSubmissionPlanner
from directordeck.native_templates import (
    NativeHistoricalTake,
)
from directordeck.schemas import UnifiedTimelineDraft
from directordeck.workflow.execution import (
    AssemblySourceArtifactRef,
    CompiledPlanDigest,
    ContinuityLateBindingEvidence,
    EndpointRestartCertificate,
    ExactCancelConfirmedEvidence,
    HistoryTerminalEvidence,
    LockedSubmissionPlan,
    ObservedArtifactSpec,
    ObservedAssemblyArtifactSpec,
    OutputObservationReceipt,
    OutputDescriptor,
    compiled_execution_plan_digest,
    compiled_execution_plan_digest_from_canonical_json,
    sha256_document_digest,
)
from directordeck.workflow.v4_execution_adapter import compile_v4_execution_plan

from . import extensible_workflow_v0_fixture_builder as fixture_builder
from .test_workflow_execution_contracts import (
    NOW,
    endpoint_identity,
)


def _create_parent(
    database: Database,
    job_id: str = "job-1",
    *,
    draft: UnifiedTimelineDraft | None = None,
    project_id: str = Database.LEGACY_DEFAULT_PROJECT_ID,
) -> None:
    now = NOW.isoformat()
    database.create_job(
        {
            "id": job_id,
            "mode": "timeline",
            "status": "preparing",
            "progress": 0.0,
            "stage": "preflight",
            "prompt_id": None,
            "project_id": project_id,
            "outputs": [],
            "error": None,
            "config_snapshot": (
                {}
                if draft is None
                else {
                    "timeline": draft.model_dump(mode="json"),
                    "segment_ids": None,
                }
            ),
            "settings_snapshot": {},
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )


def _empty_ray_state(
    *,
    epoch: int = 7,
) -> dict[str, object]:
    return {
        "version": 2,
        "epoch": epoch,
        "current": None,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": False,
    }


def _standard_locked_plan(
    *,
    ray_before: dict[str, object] | None = None,
    ray_after: dict[str, object] | None = None,
):
    plan = _standard_plan()
    locked = LockedSubmissionPlanner(endpoint_identity()).build_wave(
        plan,
        source_unit_ordinal=0,
        segment_child_id="child-1",
        ray_ledger_before=ray_before,
    )
    if ray_after is not None:
        assert locked.model_dump(mode="json")["ray_ledger_after_intent"] == ray_after
    return locked


@lru_cache(maxsize=1)
def _standard_plan():
    return compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("standard"),
        "database-standard",
        endpoint_key=endpoint_identity().endpoint_key,
    )


def compiled_plan():
    return _standard_plan()


def exact_snapshot():
    plan = _standard_locked_plan()
    return LockedSubmissionPlanner(endpoint_identity()).exact_snapshot(
        plan, plan.units[0]
    )


def _ray_contracts() -> tuple[object, LockedSubmissionPlan, object]:
    compiled = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "database-ray",
    )
    planner = LockedSubmissionPlanner(endpoint_identity(endpoint_key="embedded"))
    locked_plan = planner.build_wave(
        compiled,
        source_unit_ordinal=0,
        segment_child_id="child-ray",
    )
    snapshot = planner.exact_snapshot(locked_plan, locked_plan.units[0])
    return compiled, locked_plan, snapshot


def _control_pair_contracts():
    compiled = compile_v4_execution_plan(
        fixture_builder._draft("t2v"),
        fixture_builder._settings("raylight"),
        "database-control",
    )
    seed = LockedSubmissionPlanner(
        endpoint_identity(endpoint_key="embedded")
    ).build_wave(
        compiled,
        source_unit_ordinal=0,
        segment_child_id="resident-seed",
        ray_ledger_before=_empty_ray_state(epoch=1),
    )
    descriptor = seed.model_dump(mode="json")["ray_ledger_after_intent"][
        "current"
    ]
    before = {
        "version": 2,
        "epoch": 2,
        "current": descriptor,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": True,
    }
    planner = LockedSubmissionPlanner(
        endpoint_identity(endpoint_key="embedded"),
        id_factory=lambda: "child-control",
    )
    pair = planner.build_wave(
        compiled,
        source_unit_ordinal=0,
        segment_child_id="child-after-control",
        ray_ledger_before=before,
    )
    clean = {
        "version": 2,
        "epoch": 2,
        "current": None,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": False,
    }
    continuation = planner.segment_continuation(
        pair,
        ray_ledger_before=clean,
    )
    return compiled, planner, pair, continuation, before, clean


def _same_run_continuity_contracts():
    draft = fixture_builder._continuity_draft()
    compiled = compile_v4_execution_plan(
        draft,
        fixture_builder._settings("standard"),
        "database-same-run",
        endpoint_key=endpoint_identity().endpoint_key,
    )
    planner = LockedSubmissionPlanner(endpoint_identity())
    predecessor_plan = planner.build_wave(
        compiled,
        source_unit_ordinal=0,
        segment_child_id="child-predecessor",
    )
    output = OutputDescriptor(
        filename="predecessor.mp4",
        subfolder="director/database-same-run",
    )
    dependency = compiled.segment_units[1].continuity_dependency
    assert dependency is not None
    evidence = ContinuityLateBindingEvidence(
        input_pointer=str(dependency["input_pointer"]),
        predecessor_segment_id=str(dependency["predecessor_segment_id"]),
        dependency_source="same_run",
        output=output,
    )
    successor_plan = planner.build_wave(
        compiled,
        source_unit_ordinal=1,
        segment_child_id="child-successor",
        continuity_evidence=(evidence,),
    )
    return (
        draft,
        compiled,
        planner,
        predecessor_plan,
        successor_plan,
        evidence,
        output,
    )


def _persist_same_run_predecessor(
    database: Database,
    *,
    terminal: bool = True,
):
    (
        draft,
        execution_plan,
        planner,
        predecessor_plan,
        successor_plan,
        evidence,
        output,
    ) = _same_run_continuity_contracts()
    _create_parent(database, draft=draft, project_id="project-same-run")
    database.create_job_execution_plan("job-1", execution_plan)
    predecessor_snapshot = planner.exact_snapshot(
        predecessor_plan, predecessor_plan.units[0]
    )
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=predecessor_plan,
        exact_snapshot=predecessor_snapshot,
    )
    predecessor = predecessor_plan.units[0]
    terminal_output = {
        **output.model_dump(mode="json"),
        "node_id": predecessor.expected_output_spec.node_id,
    }
    if terminal:
        _persist_observed_success(
            database,
            "child-predecessor",
            output=output,
            has_audio=True,
        )
    return (
        draft,
        planner,
        successor_plan,
        evidence,
        output,
        terminal_output,
    )


def _historical_continuity_contracts():
    draft = fixture_builder._continuity_draft()
    output = OutputDescriptor(
        filename="historical.mp4",
        subfolder="director/database-historical",
    )
    take_id = "historical-take-exact"
    predecessor_id = draft.segments[0].id
    target_id = draft.segments[1].id
    compiled = compile_v4_execution_plan(
        draft,
        fixture_builder._settings("standard"),
        "database-historical",
        [target_id],
        historical_takes={
            target_id: NativeHistoricalTake(
                id=take_id,
                segment_id=predecessor_id,
                output=output.model_dump(mode="json"),
            )
        },
        endpoint_key=endpoint_identity().endpoint_key,
    )
    dependency = compiled.segment_units[0].continuity_dependency
    assert dependency is not None
    evidence = ContinuityLateBindingEvidence(
        input_pointer=str(dependency["input_pointer"]),
        predecessor_segment_id=predecessor_id,
        dependency_source="historical_take",
        historical_take_id=take_id,
        output=output,
    )
    planner = LockedSubmissionPlanner(endpoint_identity())
    plan = planner.build_wave(
        compiled,
        source_unit_ordinal=0,
        segment_child_id="child-historical-successor",
        continuity_evidence=(evidence,),
    )
    take_row = {
        "id": take_id,
        "segment_id": predecessor_id,
        "content_fingerprint": timeline_segment_take_fingerprint(
            draft, draft.segments[0]
        ),
        "project_id": "project-historical",
        "output_descriptor": json.dumps(
            output.model_dump(mode="json"), sort_keys=True
        ),
        "has_audio": 1,
        "source_job_id": "historical-source-job",
        "source_child_id": "historical-source-child",
        "completed_at": NOW.isoformat(),
        "created_at": NOW.isoformat(),
    }
    return draft, compiled, planner, plan, take_row


def _typed_observation_contracts(
    database: Database,
    child_id: str,
    *,
    output: OutputDescriptor,
    has_audio: bool,
):
    ownership = database.get_prompt_ownership(child_id)
    execution = database.get_job_child_execution_evidence(child_id)
    assert ownership is not None
    assert execution is not None
    snapshot = execution["exact_prompt_snapshot"]
    expected = snapshot.expected_output_spec
    assert expected is not None
    observed_at = ownership.updated_at + timedelta(seconds=1)
    history = {
        "prompt_id": ownership.effective_prompt_id,
        "status": "success",
        "outputs": {
            expected.node_id: {
                "videos": [output.model_dump(mode="json")],
            }
        },
    }
    evidence = HistoryTerminalEvidence(
        prompt_id=ownership.effective_prompt_id,
        terminal_status="succeeded",
        history_digest=sha256_document_digest(history),
        observed_at=observed_at,
    )
    receipt = OutputObservationReceipt(
        child_id=child_id,
        segment_id=expected.segment_id,
        node_id=expected.node_id,
        output_descriptor=output,
        exact_prompt_snapshot_digest=sha256_document_digest(
            snapshot.model_dump(mode="json")
        ),
        expected_output_spec_digest=sha256_document_digest(
            expected.model_dump(mode="json")
        ),
        history_evidence=evidence,
    )
    artifact = ObservedArtifactSpec(
        segment_id=expected.segment_id,
        child_id=child_id,
        output_descriptor=output,
        width=expected.width,
        height=expected.height,
        fps=expected.fps,
        frame_count=expected.visible_frame_count,
        duration_seconds=expected.visible_frame_count / expected.fps,
        has_audio=has_audio,
        media_probe_version="test-ffprobe-v1",
        content_hash=None,
    )
    return ownership, receipt, artifact, observed_at


def _persist_observed_success(
    database: Database,
    child_id: str,
    *,
    output: OutputDescriptor,
    has_audio: bool,
):
    ownership, receipt, artifact, observed_at = _typed_observation_contracts(
        database,
        child_id,
        output=output,
        has_audio=has_audio,
    )
    recorded = database.record_output_observation_receipt(
        child_id,
        expected_revision=ownership.ownership_revision,
        receipt=receipt,
        updated_at=observed_at,
    )
    assert recorded is not None
    finalized = database.finalize_observed_artifact(
        child_id,
        artifact=artifact,
        updated_at=observed_at + timedelta(seconds=1),
    )
    return recorded, finalized


def _persist_assembly_sources(
    database: Database,
) -> tuple[dict[str, object], ObservedAssemblyArtifactSpec]:
    (
        draft,
        execution_plan,
        planner,
        predecessor_plan,
        successor_plan,
        _,
        predecessor_output,
    ) = _same_run_continuity_contracts()
    _create_parent(database, draft=draft, project_id="project-assembly")
    database.create_job_execution_plan("job-1", execution_plan)

    predecessor_snapshot = planner.exact_snapshot(
        predecessor_plan, predecessor_plan.units[0]
    )
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=predecessor_plan,
        exact_snapshot=predecessor_snapshot,
    )
    _persist_observed_success(
        database,
        "child-predecessor",
        output=predecessor_output,
        has_audio=True,
    )

    successor_snapshot = planner.exact_snapshot(
        successor_plan, successor_plan.units[0]
    )
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=successor_plan,
        exact_snapshot=successor_snapshot,
    )
    successor_output = OutputDescriptor(
        filename="successor.mp4",
        subfolder="director/database-same-run",
    )
    _persist_observed_success(
        database,
        "child-successor",
        output=successor_output,
        has_audio=True,
    )

    sources = []
    for unit, child_id in zip(
        execution_plan.segment_units,
        ("child-predecessor", "child-successor"),
        strict=True,
    ):
        observed = database.get_observed_artifact(child_id)
        assert observed is not None
        sources.append(
            AssemblySourceArtifactRef(
                segment_id=unit.owner_segment_id,
                child_id=child_id,
                observed_artifact_digest=sha256_document_digest(
                    observed.model_dump(mode="json")
                ),
            )
        )

    first_expected = execution_plan.segment_units[0].expected_output_spec
    frame_count = sum(
        unit.expected_output_spec.visible_frame_count
        for unit in execution_plan.segment_units
    )
    assembly = ObservedAssemblyArtifactSpec(
        job_id="job-1",
        source_compiled_plan_digest=compiled_execution_plan_digest(execution_plan),
        source_artifacts=tuple(sources),
        output_descriptor=OutputDescriptor(
            filename="timeline.mp4",
            subfolder="directordeck/timelines",
        ),
        width=first_expected.width,
        height=first_expected.height,
        fps=first_expected.fps,
        frame_count=frame_count,
        duration_seconds=frame_count / first_expected.fps,
        has_audio=True,
        media_probe_version="test-ffprobe-assembly-v1",
        content_hash="sha256:" + "a" * 64,
    )
    claimed_at = (NOW + timedelta(minutes=10)).isoformat()
    claimed = database.update_job(
        "job-1",
        status="running",
        progress=0.99,
        stage="assembling",
        updated_at=claimed_at,
    )
    return claimed, assembly


def _insert_take(
    database: Database,
    take: dict[str, object],
    *,
    observed: bool = True,
    geometry: tuple[int, int, float, int] = (736, 416, 24.0, 121),
) -> None:
    columns = tuple(take)
    with sqlite3.connect(database.path) as connection:
        connection.row_factory = sqlite3.Row
        if observed:
            database._ensure_artifact_observation_schema(connection)
        connection.execute(
            f"INSERT INTO segment_takes({','.join(columns)}) "
            f"VALUES({','.join('?' for _ in columns)})",
            tuple(take[column] for column in columns),
        )
        if observed:
            output = OutputDescriptor.model_validate_json(
                str(take["output_descriptor"])
            )
            artifact = ObservedArtifactSpec(
                segment_id=str(take["segment_id"]),
                child_id=str(take["source_child_id"]),
                output_descriptor=output,
                width=geometry[0],
                height=geometry[1],
                fps=geometry[2],
                frame_count=geometry[3],
                duration_seconds=geometry[3] / geometry[2],
                has_audio=bool(take["has_audio"]),
                media_probe_version="historical-test-ffprobe-v1",
                content_hash=None,
            )
            artifact_json = database._contract_json(artifact)
            connection.execute(
                "INSERT INTO segment_take_observed_artifacts("
                "take_id, source_child_id, schema_version, observed_artifact, "
                "observed_artifact_digest, receipt_digest, created_at) "
                "VALUES(?, ?, 1, ?, ?, ?, ?)",
                (
                    take["id"],
                    take["source_child_id"],
                    artifact_json,
                    database._execution_document_digest(artifact),
                    "sha256-test-receipt",
                    take["created_at"],
                ),
            )


def _database(tmp_path) -> Database:
    database = Database(tmp_path / "execution-evidence.sqlite3")
    database.initialize()
    return database


def _stage4_schema_names(database: Database) -> set[str]:
    with sqlite3.connect(database.path) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE "
                "'%execution%' OR name LIKE 'prompt_ownership%'"
            ).fetchall()
        }


def _observation_schema_names(database: Database) -> set[str]:
    with sqlite3.connect(database.path) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE name IN "
                "('job_child_output_receipts', "
                "'segment_take_observed_artifacts')"
            ).fetchall()
        }


def _assembly_schema_names(database: Database) -> set[str]:
    with sqlite3.connect(database.path) as connection:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE name IN "
                "('job_observed_assembly_artifacts', "
                "'job_observed_assembly_artifacts_immutable')"
            ).fetchall()
        }


def test_legacy_evidence_reads_return_none_without_materializing_schema(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    before = _stage4_schema_names(database)

    assert database.get_job_execution_plan("legacy-job") is None
    assert database.get_job_child_execution_evidence("legacy-child") is None
    assert database.get_prompt_ownership("legacy-child") is None
    assert database.get_output_observation_receipt("legacy-child") is None
    assert database.get_observed_artifact("legacy-child") is None
    assert database.get_observed_assembly_artifact("legacy-job") is None
    assert database.list_pending_output_observations() == []

    assert _stage4_schema_names(database) == before == set()
    assert _observation_schema_names(database) == set()
    assert _assembly_schema_names(database) == set()


def test_observed_assembly_contract_rejects_duplicate_sources(tmp_path) -> None:
    database = _database(tmp_path)
    _, artifact = _persist_assembly_sources(database)
    document = artifact.model_dump(mode="json")
    document["source_artifacts"][1]["segment_id"] = document[
        "source_artifacts"
    ][0]["segment_id"]

    with pytest.raises(ValueError, match="source segment ids must be unique"):
        ObservedAssemblyArtifactSpec.model_validate_json(json.dumps(document))


def test_observed_assembly_finalize_is_atomic_immutable_and_idempotent(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    claimed, artifact = _persist_assembly_sources(database)
    assert database.get_observed_assembly_artifact("job-1") is None

    settled = database.finalize_observed_assembly_artifact(
        "job-1",
        expected_updated_at=str(claimed["updated_at"]),
        artifact=artifact,
        updated_at=NOW + timedelta(minutes=11),
    )

    assert settled is not None
    assert settled["status"] == "succeeded"
    assert settled["stage"] == "completed"
    assert settled["progress"] == 1.0
    assert settled["outputs"] == [
        {
            "node_id": "assembly",
            **artifact.output_descriptor.model_dump(mode="json"),
        }
    ]
    assert settled["error"] is None
    assert settled["completed_at"] == (NOW + timedelta(minutes=11)).isoformat()
    assert database.get_observed_assembly_artifact("job-1") == artifact

    repeated = database.finalize_observed_assembly_artifact(
        "job-1",
        expected_updated_at=str(claimed["updated_at"]),
        artifact=artifact,
        updated_at=NOW + timedelta(minutes=12),
    )
    assert repeated == settled

    with sqlite3.connect(database.path) as connection:
        with pytest.raises(
            sqlite3.IntegrityError,
            match="observed assembly artifact is immutable",
        ):
            connection.execute(
                "UPDATE job_observed_assembly_artifacts SET created_at = ? "
                "WHERE job_id = ?",
                (NOW.isoformat(), "job-1"),
            )


@pytest.mark.parametrize(
    "forgery",
    ("source_order", "source_digest", "compiled_plan_digest"),
)
def test_observed_assembly_finalize_rejects_forged_source_authority(
    tmp_path,
    forgery: str,
) -> None:
    database = _database(tmp_path)
    claimed, artifact = _persist_assembly_sources(database)
    if forgery == "source_order":
        forged = artifact.model_copy(
            update={"source_artifacts": tuple(reversed(artifact.source_artifacts))}
        )
    elif forgery == "source_digest":
        first = artifact.source_artifacts[0].model_copy(
            update={
                "observed_artifact_digest": sha256_document_digest(
                    {"forged": True}
                )
            }
        )
        forged = artifact.model_copy(
            update={"source_artifacts": (first, *artifact.source_artifacts[1:])}
        )
    else:
        forged = artifact.model_copy(
            update={
                "source_compiled_plan_digest": CompiledPlanDigest(
                    value="sha256-" + "0" * 64
                )
            }
        )

    with pytest.raises(ExecutionEvidenceConflict):
        database.finalize_observed_assembly_artifact(
            "job-1",
            expected_updated_at=str(claimed["updated_at"]),
            artifact=forged,
            updated_at=NOW + timedelta(minutes=11),
        )

    parent = database.get_job("job-1")
    assert parent is not None
    assert parent["status"] == "running"
    assert parent["stage"] == "assembling"
    assert parent["outputs"] == []
    assert database.get_observed_assembly_artifact("job-1") is None


def test_observed_assembly_finalize_loses_to_parent_cancellation(tmp_path) -> None:
    database = _database(tmp_path)
    claimed, artifact = _persist_assembly_sources(database)
    marked, first_claim = database.mark_job_cancel_requested("job-1")
    assert marked is not None and first_claim

    assert database.finalize_observed_assembly_artifact(
        "job-1",
        expected_updated_at=str(claimed["updated_at"]),
        artifact=artifact,
        updated_at=NOW + timedelta(minutes=11),
    ) is None
    assert database.get_observed_assembly_artifact("job-1") is None
    parent = database.get_job("job-1")
    assert parent is not None
    assert parent["cancel_requested"] == 1
    assert parent["status"] == "running"


def test_observed_assembly_finalize_rolls_back_evidence_when_parent_update_fails(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    claimed, artifact = _persist_assembly_sources(database)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_assembly_parent_success "
            "BEFORE UPDATE OF status ON jobs WHEN OLD.id = 'job-1' "
            "BEGIN SELECT RAISE(ABORT, 'injected assembly failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected assembly failure"):
        database.finalize_observed_assembly_artifact(
            "job-1",
            expected_updated_at=str(claimed["updated_at"]),
            artifact=artifact,
            updated_at=NOW + timedelta(minutes=11),
        )

    assert database.get_observed_assembly_artifact("job-1") is None
    parent = database.get_job("job-1")
    assert parent is not None
    assert parent["status"] == "running"
    assert parent["stage"] == "assembling"
    assert parent["outputs"] == []


@pytest.mark.parametrize(
    "corruption",
    ("row_digest", "coherent_source_order", "source_plan_index", "child_status"),
)
def test_observed_assembly_reader_rejects_corruption(
    tmp_path,
    corruption: str,
) -> None:
    database = _database(tmp_path)
    claimed, artifact = _persist_assembly_sources(database)
    assert database.finalize_observed_assembly_artifact(
        "job-1",
        expected_updated_at=str(claimed["updated_at"]),
        artifact=artifact,
        updated_at=NOW + timedelta(minutes=11),
    ) is not None

    with sqlite3.connect(database.path) as connection:
        if corruption != "child_status":
            connection.execute(
                "DROP TRIGGER job_observed_assembly_artifacts_immutable"
            )
        if corruption == "row_digest":
            connection.execute(
                "UPDATE job_observed_assembly_artifacts "
                "SET observed_assembly_artifact_digest = ? WHERE job_id = ?",
                ("sha256-" + "1" * 64, "job-1"),
            )
        elif corruption == "coherent_source_order":
            forged = artifact.model_copy(
                update={
                    "source_artifacts": tuple(reversed(artifact.source_artifacts))
                }
            )
            connection.execute(
                "UPDATE job_observed_assembly_artifacts "
                "SET observed_assembly_artifact = ?, "
                "observed_assembly_artifact_digest = ? WHERE job_id = ?",
                (
                    database._contract_json(forged),
                    database._execution_document_digest(forged),
                    "job-1",
                ),
            )
        elif corruption == "source_plan_index":
            connection.execute(
                "UPDATE job_observed_assembly_artifacts "
                "SET source_compiled_plan_digest = ? WHERE job_id = ?",
                ("sha256-" + "2" * 64, "job-1"),
            )
        else:
            connection.execute(
                "UPDATE job_children SET status = 'failed' WHERE id = ?",
                ("child-predecessor",),
            )

    with pytest.raises(ExecutionEvidenceConflict):
        database.get_observed_assembly_artifact("job-1")


def test_plan_and_child_exact_evidence_are_insert_only_and_round_trip(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan = compiled_plan()

    assert database.create_job_execution_plan("job-1", plan) == plan
    assert database.get_job_execution_plan("job-1") == plan
    with pytest.raises(sqlite3.IntegrityError):
        database.create_job_execution_plan("job-1", plan)

    child, ownership = database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=_standard_locked_plan(),
        exact_snapshot=exact_snapshot(),
    )
    assert child["id"] == "child-1"
    assert child["prompt_id"] == "child-1"
    assert (
        child["prompt_snapshot"]
        == exact_snapshot().model_dump(mode="json")["exact_prompt"]
    )
    assert ownership.state == "submitting"
    assert ownership.ownership_revision == 0
    evidence = database.get_job_child_execution_evidence("child-1")
    assert evidence is not None
    assert evidence["exact_prompt_snapshot"] == exact_snapshot()

    with sqlite3.connect(database.path) as connection:
        indexed = connection.execute(
            "SELECT unit_index_version FROM job_execution_plans "
            "WHERE job_id = 'job-1'"
        ).fetchone()
        assert indexed == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM job_execution_plan_units "
            "WHERE job_id = 'job-1'"
        ).fetchone() == (len(plan.segment_units),)
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE job_execution_plans SET schema_version = 2 "
                "WHERE job_id = 'job-1'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE job_execution_plan_units SET unit_id = 'changed' "
                "WHERE job_id = 'job-1' AND unit_ordinal = 0"
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE job_child_execution_evidence SET unit_id = 'changed' "
                "WHERE child_id = 'child-1'"
            )


def test_submission_intent_lazily_indexes_pre_index_execution_plan(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan = compiled_plan()
    database.create_job_execution_plan("job-1", plan)

    # Reproduce the additive upgrade state created by the pre-index Stage-4
    # schema. The first submission authenticates the full plan once, builds
    # every unit row, and seals the marker in the same transaction.
    with sqlite3.connect(database.path) as connection:
        connection.execute("DROP TRIGGER job_execution_plans_immutable")
        connection.execute(
            "DELETE FROM job_execution_plan_units WHERE job_id = 'job-1'"
        )
        connection.execute(
            "UPDATE job_execution_plans SET unit_index_version = 0 "
            "WHERE job_id = 'job-1'"
        )

    child, _ownership = database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=_standard_locked_plan(),
        exact_snapshot=exact_snapshot(),
    )

    assert child["id"] == "child-1"
    with sqlite3.connect(database.path) as connection:
        assert connection.execute(
            "SELECT unit_index_version FROM job_execution_plans "
            "WHERE job_id = 'job-1'"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM job_execution_plan_units "
            "WHERE job_id = 'job-1'"
        ).fetchone() == (len(plan.segment_units),)


def test_submission_intent_fails_closed_when_sealed_unit_index_is_missing(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    locked = _standard_locked_plan()
    snapshot = LockedSubmissionPlanner(endpoint_identity()).exact_snapshot(
        locked, locked.units[0]
    )

    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "DELETE FROM job_execution_plan_units "
            "WHERE job_id = 'job-1' AND unit_ordinal = 0"
        )

    with pytest.raises(ExecutionEvidenceConflict):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=locked,
            exact_snapshot=snapshot,
        )

    assert database.get_job_child("child-1") is None


def test_submission_intent_rejects_non_exact_persisted_plan_bytes(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    locked = _standard_locked_plan()
    snapshot = LockedSubmissionPlanner(endpoint_identity()).exact_snapshot(
        locked, locked.units[0]
    )
    with sqlite3.connect(database.path) as connection:
        connection.execute("DROP TRIGGER job_execution_plans_immutable")
        row = connection.execute(
            "SELECT compiled_plan FROM job_execution_plans WHERE job_id = ?",
            ("job-1",),
        ).fetchone()
        assert row is not None
        connection.execute(
            "UPDATE job_execution_plans SET compiled_plan = ? WHERE job_id = ?",
            (str(row[0]) + " ", "job-1"),
        )

    with pytest.raises(ExecutionEvidenceConflict):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=locked,
            exact_snapshot=snapshot,
        )

    assert database.get_job_child("child-1") is None


def test_submission_intent_strictly_cross_links_current_prepared_unit(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    locked = _standard_locked_plan()
    snapshot = LockedSubmissionPlanner(endpoint_identity()).exact_snapshot(
        locked, locked.units[0]
    )
    with sqlite3.connect(database.path) as connection:
        connection.execute("DROP TRIGGER job_execution_plans_immutable")
        row = connection.execute(
            "SELECT compiled_plan FROM job_execution_plans WHERE job_id = ?",
            ("job-1",),
        ).fetchone()
        assert row is not None
        document = json.loads(str(row[0]))
        document["segment_units"][0]["template_revision"] += 1
        forged_json = database._contract_json(document)
        forged_digest = compiled_execution_plan_digest_from_canonical_json(
            forged_json
        )
        connection.execute(
            "UPDATE job_execution_plans SET compiled_plan = ?, "
            "compiled_plan_digest = ? WHERE job_id = ?",
            (forged_json, forged_digest.value, "job-1"),
        )
    forged_locked = locked.model_copy(
        update={"source_compiled_plan_digest": forged_digest}
    )

    with pytest.raises(ExecutionEvidenceConflict):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=forged_locked,
            exact_snapshot=snapshot,
        )

    assert database.get_job_child("child-1") is None


@pytest.mark.parametrize(
    "operation",
    ["output_receipt", "successful_history_failure", "prompt_cleanup"],
)
def test_terminal_transactions_revalidate_exact_snapshot_digest_under_lock(
    tmp_path, operation: str
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=_standard_locked_plan(),
        exact_snapshot=exact_snapshot(),
    )
    ownership, receipt, _artifact, observed_at = _typed_observation_contracts(
        database,
        "child-1",
        output=OutputDescriptor(
            filename="digest-guard.mp4",
            subfolder="director/digest-guard",
        ),
        has_audio=False,
    )
    child_before = database.get_job_child("child-1")
    assert child_before is not None

    # Simulate storage corruption below the normal immutable trigger.  Every
    # terminal/receipt writer must re-read and authenticate the exact snapshot
    # inside its own BEGIN IMMEDIATE transaction, not trust an earlier read.
    with sqlite3.connect(database.path) as connection:
        connection.execute("DROP TRIGGER job_child_execution_evidence_immutable")
        connection.execute(
            "UPDATE job_child_execution_evidence "
            "SET exact_prompt_snapshot_digest = ? WHERE child_id = ?",
            ("sha256-" + "0" * 64, "child-1"),
        )

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="exact prompt evidence digest/index mismatch",
    ):
        if operation == "output_receipt":
            database.record_output_observation_receipt(
                "child-1",
                expected_revision=ownership.ownership_revision,
                receipt=receipt,
                updated_at=observed_at,
            )
        elif operation == "successful_history_failure":
            database.fail_successful_history_artifact(
                "child-1",
                expected_revision=ownership.ownership_revision,
                evidence=receipt.history_evidence,
                error="trusted output descriptor is unusable",
                updated_at=observed_at,
            )
        else:
            database.confirm_prompt_cleanup(
                "child-1",
                expected_revision=ownership.ownership_revision,
                evidence=ExactCancelConfirmedEvidence(
                    prompt_id=ownership.effective_prompt_id,
                    confirmation_id="digest-guard-cancel",
                    confirmed_at=observed_at,
                ),
                stage="cancelled",
                updated_at=observed_at,
            )

    assert database.get_prompt_ownership("child-1") == ownership
    child_after = database.get_job_child("child-1")
    assert child_after is not None
    assert {
        key: child_after[key]
        for key in ("status", "stage", "prompt_id", "outputs", "error")
    } == {
        key: child_before[key]
        for key in ("status", "stage", "prompt_id", "outputs", "error")
    }
    with sqlite3.connect(database.path) as connection:
        receipt_table = connection.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'job_child_output_receipts'"
        ).fetchone()
        observation_count = (
            0
            if receipt_table is None
            else int(
                connection.execute(
                    "SELECT COUNT(*) FROM job_child_output_receipts "
                    "WHERE child_id = ?",
                    ("child-1",),
                ).fetchone()[0]
            )
        )
    assert observation_count == 0


def test_submission_intent_commits_child_exact_ownership_and_ray_together(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    frontier = _empty_ray_state()
    database.put_raylight_runtime_state(frontier)

    child, ownership = database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=_standard_locked_plan(
            ray_before=frontier,
            ray_after=frontier,
        ),
        exact_snapshot=exact_snapshot(),
    )

    assert child["stage"] == "submitting"
    assert ownership.requested_prompt_id == "child-1"
    assert database.get_job_child_execution_evidence("child-1") is not None
    assert database.get_raylight_runtime_state() == frontier


def test_submission_intent_failure_rolls_back_every_ledger(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    before = _empty_ray_state()
    database.put_raylight_runtime_state(before)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_stage4_ray_intent "
            "BEFORE UPDATE ON raylight_runtime_state "
            "BEGIN SELECT RAISE(ABORT, 'injected ray intent failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected ray intent failure"):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=_standard_locked_plan(
                ray_before=before,
                ray_after=before,
            ),
            exact_snapshot=exact_snapshot(),
        )

    assert database.get_job_child("child-1") is None
    assert database.get_job_child_execution_evidence("child-1") is None
    assert database.get_prompt_ownership("child-1") is None
    assert database.get_raylight_runtime_state() == before


def test_submission_intent_rejects_stale_ray_before_and_rolls_back(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    database.put_raylight_runtime_state(_empty_ray_state(epoch=8))

    with pytest.raises(RayRuntimeIntentConflict):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=_standard_locked_plan(
                ray_before=_empty_ray_state(epoch=7),
                ray_after=_empty_ray_state(epoch=7),
            ),
            exact_snapshot=exact_snapshot(),
        )

    assert database.get_job_child("child-1") is None
    assert database.get_job_child_execution_evidence("child-1") is None
    assert database.get_prompt_ownership("child-1") is None


def test_two_unit_plan_cannot_persist_its_segment_intent(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, planner, pair, _, before, _ = _control_pair_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.put_raylight_runtime_state(before)
    segment_snapshot = planner.exact_snapshot(pair, pair.units[1])

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="only its first control",
    ):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=pair,
            exact_snapshot=segment_snapshot,
        )

    assert database.get_job_child("child-control") is None
    assert database.get_job_child("child-after-control") is None
    assert database.get_raylight_runtime_state() == before


def test_database_rejects_ray_segment_without_derived_frontier(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    tampered = LockedSubmissionPlan(
        version=locked.version,
        endpoint_identity=locked.endpoint_identity,
        units=locked.units,
        source_compiled_plan_digest=locked.source_compiled_plan_digest,
        source_unit_id=locked.source_unit_id,
        source_unit_ordinal=locked.source_unit_ordinal,
        ray_ledger_before=locked.ray_ledger_before,
        ray_ledger_after_intent=None,
    )

    with pytest.raises(ExecutionEvidenceConflict, match="derived transition"):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=tampered,
            exact_snapshot=snapshot,
        )

    assert database.get_job_child("child-ray") is None
    assert database.get_prompt_ownership("child-ray") is None
    assert database.get_raylight_runtime_state() is None


def test_segment_continuation_requires_terminal_confirmed_control(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, planner, pair, continuation, before, clean = _control_pair_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.put_raylight_runtime_state(before)
    control_snapshot = planner.exact_snapshot(pair, pair.units[0])
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=pair,
        exact_snapshot=control_snapshot,
    )
    # Even a matching clean ledger cannot replace positive terminal evidence.
    database.put_raylight_runtime_state(clean)
    segment_snapshot = planner.exact_snapshot(
        continuation,
        continuation.units[0],
    )
    with pytest.raises(
        ExecutionEvidenceConflict,
        match="not terminal-confirmed",
    ):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=continuation,
            exact_snapshot=segment_snapshot,
        )
    assert database.get_job_child("child-after-control") is None

    control_ownership = database.get_prompt_ownership("child-control")
    assert control_ownership is not None
    confirmed_at = control_ownership.updated_at + timedelta(seconds=1)
    confirmed = database.confirm_prompt_terminal(
        "child-control",
        expected_revision=0,
        evidence=HistoryTerminalEvidence(
            prompt_id="child-control",
            terminal_status="succeeded",
            history_digest=sha256_document_digest({"status": "success"}),
            observed_at=confirmed_at,
        ),
        outputs=[],
        stage="RayLight released",
        error=None,
        updated_at=confirmed_at,
    )
    assert confirmed is not None
    child, _ = database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=continuation,
        exact_snapshot=segment_snapshot,
    )
    assert child["id"] == "child-after-control"
    assert database.get_raylight_runtime_state()["tail_prompt_id"] == (
        "child-after-control"
    )


def test_same_run_continuity_uses_terminal_exact_predecessor_and_registers_take(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    (
        draft,
        planner,
        successor_plan,
        _,
        output,
        _,
    ) = _persist_same_run_predecessor(database)

    take = database.find_latest_segment_take(
        draft.segments[0].id,
        timeline_segment_take_fingerprint(draft, draft.segments[0]),
        require_audio=True,
        project_id="project-same-run",
    )
    assert take is not None
    assert take["source_child_id"] == "child-predecessor"
    assert take["output"] == output.model_dump(mode="json")

    snapshot = planner.exact_snapshot(
        successor_plan, successor_plan.units[0]
    )
    child, ownership = database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=successor_plan,
        exact_snapshot=snapshot,
    )
    assert child["id"] == "child-successor"
    assert ownership.state == "submitting"


@pytest.mark.parametrize(
    "forgery",
    (
        "artifact_digest",
        "receipt_digest",
        "segment_id",
        "output_descriptor",
        "has_audio",
        "source_child_id",
        "source_job_id",
        "ownership",
    ),
)
def test_observed_take_readers_reject_cross_link_forgery(
    tmp_path,
    forgery: str,
) -> None:
    database = _database(tmp_path)
    (
        draft,
        planner,
        successor_plan,
        _,
        _,
        _,
    ) = _persist_same_run_predecessor(database)
    segment_id = draft.segments[0].id
    lookup_segment_id = segment_id
    fingerprint = timeline_segment_take_fingerprint(draft, draft.segments[0])

    with sqlite3.connect(database.path) as connection:
        if forgery == "artifact_digest":
            connection.execute(
                "DROP TRIGGER segment_take_observed_artifacts_immutable"
            )
            connection.execute(
                "UPDATE segment_take_observed_artifacts "
                "SET observed_artifact_digest = ? WHERE source_child_id = ?",
                ("sha256-" + "0" * 64, "child-predecessor"),
            )
        elif forgery == "receipt_digest":
            connection.execute("DROP TRIGGER job_child_output_receipts_immutable")
            connection.execute(
                "UPDATE job_child_output_receipts SET receipt_digest = ? "
                "WHERE child_id = ?",
                ("sha256-" + "1" * 64, "child-predecessor"),
            )
        elif forgery == "segment_id":
            lookup_segment_id = "forged-segment"
            connection.execute(
                "UPDATE segment_takes SET segment_id = ? "
                "WHERE source_child_id = ?",
                (lookup_segment_id, "child-predecessor"),
            )
        elif forgery == "output_descriptor":
            connection.execute(
                "UPDATE segment_takes SET output_descriptor = ? "
                "WHERE source_child_id = ?",
                (
                    json.dumps(
                        {
                            "filename": "forged.mp4",
                            "subfolder": "attacker",
                            "type": "output",
                        },
                        sort_keys=True,
                    ),
                    "child-predecessor",
                ),
            )
        elif forgery == "has_audio":
            connection.execute(
                "UPDATE segment_takes SET has_audio = 0 "
                "WHERE source_child_id = ?",
                ("child-predecessor",),
            )
        elif forgery == "source_child_id":
            connection.execute(
                "UPDATE segment_takes SET source_child_id = ? "
                "WHERE source_child_id = ?",
                ("forged-child", "child-predecessor"),
            )
        elif forgery == "source_job_id":
            connection.execute(
                "UPDATE segment_takes SET source_job_id = ? "
                "WHERE source_child_id = ?",
                ("forged-job", "child-predecessor"),
            )
        else:
            connection.execute(
                "UPDATE prompt_ownership SET state = 'unconfirmed', "
                "cleanup_certificate = NULL WHERE child_id = ?",
                ("child-predecessor",),
            )

    with pytest.raises(ExecutionEvidenceConflict):
        database.get_observed_artifact("child-predecessor")
    with pytest.raises(ExecutionEvidenceConflict):
        database.find_latest_observed_segment_take(
            lookup_segment_id,
            fingerprint,
            project_id="project-same-run",
        )
    with pytest.raises(ExecutionEvidenceConflict):
        database.has_observed_segment_take(
            lookup_segment_id,
            content_fingerprint=fingerprint,
            project_id="project-same-run",
        )

    snapshot = planner.exact_snapshot(
        successor_plan,
        successor_plan.units[0],
    )
    with pytest.raises(ExecutionEvidenceConflict):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=successor_plan,
            exact_snapshot=snapshot,
        )


def test_observed_take_survives_complete_source_job_deletion(tmp_path) -> None:
    database = _database(tmp_path)
    draft, *_ = _persist_same_run_predecessor(database)
    segment_id = draft.segments[0].id
    fingerprint = timeline_segment_take_fingerprint(draft, draft.segments[0])

    assert database.delete_job_if_status("job-1", "preparing") is True

    take = database.find_latest_observed_segment_take(
        segment_id,
        fingerprint,
        project_id="project-same-run",
    )
    assert take is not None
    assert take["source_child_id"] == "child-predecessor"
    assert take["observed_artifact"].child_id == "child-predecessor"


def test_observed_take_rejects_partial_source_child_deletion(tmp_path) -> None:
    database = _database(tmp_path)
    draft, *_ = _persist_same_run_predecessor(database)
    segment_id = draft.segments[0].id
    fingerprint = timeline_segment_take_fingerprint(draft, draft.segments[0])
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM job_children WHERE id = ?",
            ("child-predecessor",),
        )

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="partial source deletion",
    ):
        database.find_latest_observed_segment_take(
            segment_id,
            fingerprint,
            project_id="project-same-run",
        )


def test_same_run_continuity_rejects_succeeded_child_without_terminal_ownership(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    (
        _,
        planner,
        successor_plan,
        _,
        _,
        terminal_output,
    ) = _persist_same_run_predecessor(database, terminal=False)
    database.update_job_child(
        "child-predecessor",
        status="succeeded",
        progress=1.0,
        outputs=[terminal_output],
        completed_at=NOW.isoformat(),
    )
    snapshot = planner.exact_snapshot(
        successor_plan, successor_plan.units[0]
    )

    with pytest.raises(
        ExecutionEvidenceConflict, match="not terminal-confirmed succeeded"
    ):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=successor_plan,
            exact_snapshot=snapshot,
        )

    assert database.get_job_child("child-successor") is None


def test_same_run_continuity_rejects_coherent_prompt_and_evidence_forge(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    (
        _,
        planner,
        successor_plan,
        evidence,
        _,
        _,
    ) = _persist_same_run_predecessor(database)
    forged_output = OutputDescriptor(
        filename="forged.mp4",
        subfolder="director/database-same-run",
    )
    forged_evidence = evidence.model_copy(update={"output": forged_output})
    document = successor_plan.model_dump(mode="json")
    unit_document = document["units"][0]
    node_id = evidence.input_pointer.split("/")[1]
    unit_document["exact_prompt"][node_id]["inputs"]["file"] = (
        forged_evidence.bound_value
    )
    unit_document["late_binding_evidence"] = [
        forged_evidence.model_dump(mode="json")
    ]
    unit_document["late_bound_values"] = {
        evidence.input_pointer: forged_evidence.bound_value
    }
    forged_plan = LockedSubmissionPlan.model_validate_json(json.dumps(document))
    snapshot = planner.exact_snapshot(forged_plan, forged_plan.units[0])

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="differs from terminal evidence",
    ):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=forged_plan,
            exact_snapshot=snapshot,
        )

    assert database.get_job_child("child-successor") is None


def test_deserialized_locked_unit_cannot_omit_non_resource_evidence(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    (
        _,
        planner,
        successor_plan,
        _,
        _,
        _,
    ) = _persist_same_run_predecessor(database)
    document = successor_plan.model_dump(mode="json")
    unit_document = document["units"][0]
    unit_document["exact_prompt"] = deepcopy(unit_document["prompt_base"])
    unit_document["late_binding_evidence"] = []
    unit_document["late_bound_values"] = {}
    deserialized = LockedSubmissionPlan.model_validate_json(json.dumps(document))
    snapshot = planner.exact_snapshot(deserialized, deserialized.units[0])

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="exactly cover non-resource pointers",
    ):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=deserialized,
            exact_snapshot=snapshot,
        )

    assert database.get_job_child("child-successor") is None


def test_persistence_revalidates_deserialized_materialized_prompt_with_v4_registry(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    document = _standard_locked_plan().model_dump(mode="json")
    prompt = document["units"][0]["exact_prompt"]
    noise_id = next(
        node_id
        for node_id, node in prompt.items()
        if node["class_type"] == "RandomNoise"
    )
    prompt[noise_id]["inputs"]["noise_seed"] += 1
    deserialized = LockedSubmissionPlan.model_validate_json(json.dumps(document))
    planner = LockedSubmissionPlanner(endpoint_identity())

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="materialized graph validation",
    ):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=deserialized,
            exact_snapshot=planner.exact_snapshot(
                deserialized, deserialized.units[0]
            ),
        )

    assert database.get_job_child("child-1") is None


def test_historical_continuity_accepts_exact_persisted_take(tmp_path) -> None:
    database = _database(tmp_path)
    draft, execution_plan, planner, plan, take = (
        _historical_continuity_contracts()
    )
    _create_parent(
        database, draft=draft, project_id="project-historical"
    )
    database.create_job_execution_plan("job-1", execution_plan)
    predecessor = draft.segments[0]
    _insert_take(
        database,
        take,
        geometry=(
            draft.render.width,
            draft.render.height,
            draft.render.fps,
            align_h3_frames(predecessor.duration_seconds, draft.render.fps),
        ),
    )

    child, _ = database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=plan,
        exact_snapshot=planner.exact_snapshot(plan, plan.units[0]),
    )

    assert child["id"] == "child-historical-successor"


def test_historical_continuity_rejects_legacy_take_without_observation(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    draft, execution_plan, planner, plan, take = (
        _historical_continuity_contracts()
    )
    _create_parent(
        database, draft=draft, project_id="project-historical"
    )
    database.create_job_execution_plan("job-1", execution_plan)
    _insert_take(database, take, observed=False)

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="requires observed artifact evidence",
    ):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=plan,
            exact_snapshot=planner.exact_snapshot(plan, plan.units[0]),
        )


@pytest.mark.parametrize(
    "forgery",
    ("take_id", "segment", "project", "fingerprint", "audio", "output"),
)
def test_historical_continuity_rejects_non_exact_take_fields(
    tmp_path,
    forgery: str,
) -> None:
    database = _database(tmp_path)
    draft, execution_plan, planner, plan, take = (
        _historical_continuity_contracts()
    )
    _create_parent(
        database, draft=draft, project_id="project-historical"
    )
    database.create_job_execution_plan("job-1", execution_plan)
    forged = dict(take)
    if forgery == "take_id":
        forged["id"] = "different-take-id"
    elif forgery == "segment":
        forged["segment_id"] = "different-segment"
    elif forgery == "project":
        forged["project_id"] = "different-project"
    elif forgery == "fingerprint":
        forged["content_fingerprint"] = "take-geometry-v1:sha256:forged"
    elif forgery == "audio":
        forged["has_audio"] = 0
    else:
        forged["output_descriptor"] = json.dumps(
            {
                "filename": "different.mp4",
                "subfolder": "director/database-historical",
                "type": "output",
            },
            sort_keys=True,
        )
    _insert_take(database, forged)

    with pytest.raises(ExecutionEvidenceConflict, match="historical continuity"):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=plan,
            exact_snapshot=planner.exact_snapshot(plan, plan.units[0]),
        )

    assert database.get_job_child("child-historical-successor") is None


def test_observed_artifact_finalize_rolls_back_without_losing_receipt(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    (
        _,
        _,
        _,
        _,
        _,
        terminal_output,
    ) = _persist_same_run_predecessor(database, terminal=False)
    ownership_before, receipt, artifact, observed_at = (
        _typed_observation_contracts(
            database,
            "child-predecessor",
            output=OutputDescriptor(
                filename=str(terminal_output["filename"]),
                subfolder=str(terminal_output["subfolder"]),
                type="output",
            ),
            has_audio=True,
        )
    )
    recorded = database.record_output_observation_receipt(
        "child-predecessor",
        expected_revision=ownership_before.ownership_revision,
        receipt=receipt,
        updated_at=observed_at,
    )
    assert recorded is not None
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_typed_take_registration "
            "BEFORE INSERT ON segment_takes "
            "BEGIN SELECT RAISE(ABORT, 'injected take registration failure'); END"
        )
    with pytest.raises(
        sqlite3.IntegrityError,
        match="injected take registration failure",
    ):
        database.finalize_observed_artifact(
            "child-predecessor",
            artifact=artifact,
            updated_at=observed_at + timedelta(seconds=1),
        )

    ownership = database.get_prompt_ownership("child-predecessor")
    child = database.get_job_child("child-predecessor")
    assert ownership is not None
    assert ownership.state == "terminal_confirmed"
    assert child is not None
    assert child["status"] == "running"
    assert child["stage"] == "verifying_output"
    assert database.get_output_observation_receipt("child-predecessor") == receipt
    assert database.get_observed_artifact("child-predecessor") is None
    assert not database.has_segment_take(
        child["segment_ids"][0], project_id="project-same-run"
    )


@pytest.mark.parametrize(
    "operation",
    ("existing_receipt", "finalize", "fail", "get", "recovery_list"),
)
def test_receipt_recovery_revalidates_exact_evidence_after_crash_window(
    tmp_path,
    operation: str,
) -> None:
    database = _database(tmp_path)
    (
        _,
        _,
        _,
        _,
        _,
        terminal_output,
    ) = _persist_same_run_predecessor(database, terminal=False)
    ownership, receipt, artifact, observed_at = _typed_observation_contracts(
        database,
        "child-predecessor",
        output=OutputDescriptor(
            filename=str(terminal_output["filename"]),
            subfolder=str(terminal_output["subfolder"]),
            type="output",
        ),
        has_audio=True,
    )
    recorded = database.record_output_observation_receipt(
        "child-predecessor",
        expected_revision=ownership.ownership_revision,
        receipt=receipt,
        updated_at=observed_at,
    )
    assert recorded is not None
    child_before, ownership_after, _ = recorded

    # Simulate process death after the immutable receipt transaction followed
    # by storage corruption below the exact-evidence UPDATE trigger. Every
    # recovery/finalization entry point must authenticate exact evidence anew.
    with sqlite3.connect(database.path) as connection:
        connection.execute("DROP TRIGGER job_child_execution_evidence_immutable")
        connection.execute(
            "UPDATE job_child_execution_evidence "
            "SET exact_prompt_snapshot_digest = ? WHERE child_id = ?",
            ("sha256-" + "9" * 64, "child-predecessor"),
        )

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="exact prompt evidence digest/index mismatch",
    ):
        if operation == "existing_receipt":
            database.record_output_observation_receipt(
                "child-predecessor",
                expected_revision=ownership_after.ownership_revision,
                receipt=receipt,
                updated_at=observed_at + timedelta(seconds=1),
            )
        elif operation == "finalize":
            database.finalize_observed_artifact(
                "child-predecessor",
                artifact=artifact,
                updated_at=observed_at + timedelta(seconds=1),
            )
        elif operation == "fail":
            database.fail_output_observation(
                "child-predecessor",
                error="probe failed permanently",
                updated_at=observed_at + timedelta(seconds=1),
            )
        elif operation == "get":
            database.get_output_observation_receipt("child-predecessor")
        else:
            database.list_pending_output_observations()

    child_after = database.get_job_child("child-predecessor")
    assert child_after is not None
    assert {
        key: child_after[key]
        for key in ("status", "stage", "prompt_id", "outputs", "error")
    } == {
        key: child_before[key]
        for key in ("status", "stage", "prompt_id", "outputs", "error")
    }
    assert database.get_prompt_ownership("child-predecessor") == ownership_after
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM job_child_output_receipts WHERE child_id = ?",
            ("child-predecessor",),
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM segment_take_observed_artifacts "
            "WHERE source_child_id = ?",
            ("child-predecessor",),
        ).fetchone()[0] == 0


def test_receipt_rebinds_actual_id_child_and_matching_ray_tail_atomically(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=locked,
        exact_snapshot=snapshot,
    )
    exact_before = database.get_job_child_execution_evidence("child-ray")
    before_receipt = database.get_prompt_ownership("child-ray")
    assert before_receipt is not None
    receipt_at = before_receipt.updated_at + timedelta(seconds=1)

    receipt = database.record_prompt_submission_receipt(
        "child-ray",
        expected_revision=0,
        actual_prompt_id="actual-9",
        state="owned_actual_id",
        updated_at=receipt_at,
    )

    assert receipt is not None
    assert receipt.actual_prompt_id == "actual-9"
    assert receipt.ownership_revision == 1
    assert database.get_job_child("child-ray")["prompt_id"] == "actual-9"
    assert database.get_raylight_runtime_state()["tail_prompt_id"] == "actual-9"
    assert database.get_job_child_execution_evidence("child-ray") == exact_before
    assert (
        database.record_prompt_submission_receipt(
            "child-ray",
            expected_revision=0,
            actual_prompt_id="late-other-id",
            state="owned_actual_id",
            updated_at=receipt_at + timedelta(seconds=1),
        )
        is None
    )
    assert database.get_prompt_ownership("child-ray") == receipt


def test_receipt_ray_failure_rolls_back_ownership_and_child_rebind(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=locked,
        exact_snapshot=snapshot,
    )
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_stage4_ray_rebind "
            "BEFORE UPDATE ON raylight_runtime_state "
            "BEGIN SELECT RAISE(ABORT, 'injected ray rebind failure'); END"
        )
    before_receipt = database.get_prompt_ownership("child-ray")
    assert before_receipt is not None

    with pytest.raises(sqlite3.IntegrityError, match="injected ray rebind failure"):
        database.record_prompt_submission_receipt(
            "child-ray",
            expected_revision=0,
            actual_prompt_id="actual-9",
            state="owned_actual_id",
            updated_at=before_receipt.updated_at + timedelta(seconds=1),
        )

    ownership = database.get_prompt_ownership("child-ray")
    assert ownership is not None
    assert ownership.state == "submitting"
    assert ownership.ownership_revision == 0
    assert database.get_job_child("child-ray")["prompt_id"] == "child-ray"
    assert database.get_raylight_runtime_state()["tail_prompt_id"] == "child-ray"


def test_terminal_confirmation_cas_updates_child_and_matching_ray_only(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database, draft=fixture_builder._draft("t2v"))
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1", locked_plan=locked, exact_snapshot=snapshot
    )
    before_receipt = database.get_prompt_ownership("child-ray")
    assert before_receipt is not None
    receipt_at = before_receipt.updated_at + timedelta(seconds=1)
    receipt = database.record_prompt_submission_receipt(
        "child-ray",
        expected_revision=0,
        actual_prompt_id="actual-ray",
        state="owned_actual_id",
        updated_at=receipt_at,
    )
    assert receipt is not None
    exact_before = database.get_job_child_execution_evidence("child-ray")
    output = OutputDescriptor(
        filename="result.mp4",
        subfolder="director/database-ray",
    )
    recorded, finalized = _persist_observed_success(
        database,
        "child-ray",
        output=output,
        has_audio=False,
    )

    child, take, artifact = finalized
    ownership = recorded[1]
    assert child["status"] == "succeeded"
    assert ownership.state == "terminal_confirmed"
    assert artifact.content_hash is None
    assert take["has_audio"] is False
    ray = database.get_raylight_runtime_state()
    assert ray is not None
    assert ray["tail_prompt_id"] == "actual-ray"
    assert ray["tainted"] is False
    assert ray["tail_terminal_certificate"]["prompt_id"] == "actual-ray"
    assert database.get_job_child_execution_evidence("child-ray") == exact_before

    # A stale second confirmation cannot overwrite either lifecycle ledger.
    evidence = recorded[2].history_evidence
    assert (
        database.confirm_prompt_terminal(
            "child-ray",
            expected_revision=1,
            evidence=evidence,
            outputs=[],
            stage="changed",
            error=None,
            updated_at=receipt_at + timedelta(seconds=2),
        )
        is None
    )


def test_terminal_confirmation_does_not_settle_a_newer_ray_tail(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1", locked_plan=locked, exact_snapshot=snapshot
    )
    before_receipt = database.get_prompt_ownership("child-ray")
    assert before_receipt is not None
    receipt_at = before_receipt.updated_at + timedelta(seconds=1)
    database.record_prompt_submission_receipt(
        "child-ray",
        expected_revision=0,
        actual_prompt_id="actual-ray",
        state="owned_actual_id",
        updated_at=receipt_at,
    )
    newer = dict(database.get_raylight_runtime_state() or {})
    newer.update(
        tail_prompt_id="newer-tail",
        tail_action="ray_unit",
        tainted=True,
    )
    newer.pop("tail_terminal_certificate", None)
    database.put_raylight_runtime_state(newer)
    evidence = HistoryTerminalEvidence(
        prompt_id="actual-ray",
        terminal_status="failed",
        history_digest=sha256_document_digest({"status": "failed"}),
        observed_at=receipt_at + timedelta(seconds=1),
    )

    assert (
        database.confirm_prompt_terminal(
            "child-ray",
            expected_revision=1,
            evidence=evidence,
            outputs=[],
            stage="failed",
            error="upstream failed",
            updated_at=receipt_at + timedelta(seconds=1),
        )
        is not None
    )
    assert database.get_raylight_runtime_state() == newer


def test_terminal_ray_failure_rolls_back_ownership_and_child_terminal(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1", locked_plan=locked, exact_snapshot=snapshot
    )
    before_receipt = database.get_prompt_ownership("child-ray")
    assert before_receipt is not None
    receipt_at = before_receipt.updated_at + timedelta(seconds=1)
    receipt = database.record_prompt_submission_receipt(
        "child-ray",
        expected_revision=0,
        actual_prompt_id="actual-ray",
        state="owned_actual_id",
        updated_at=receipt_at,
    )
    assert receipt is not None
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_stage4_ray_settle "
            "BEFORE UPDATE ON raylight_runtime_state "
            "BEGIN SELECT RAISE(ABORT, 'injected ray settle failure'); END"
        )
    ownership, output_receipt, _, observed_at = _typed_observation_contracts(
        database,
        "child-ray",
        output=OutputDescriptor(
            filename="result.mp4",
            subfolder="director/database-ray",
        ),
        has_audio=False,
    )
    assert ownership.ownership_revision == 1

    with pytest.raises(sqlite3.IntegrityError, match="injected ray settle failure"):
        database.record_output_observation_receipt(
            "child-ray",
            expected_revision=1,
            receipt=output_receipt,
            updated_at=observed_at,
        )

    assert database.get_prompt_ownership("child-ray") == receipt
    child = database.get_job_child("child-ray")
    assert child is not None
    assert child["status"] == "preparing"
    assert child["prompt_id"] == "actual-ray"
    ray = database.get_raylight_runtime_state()
    assert ray is not None
    assert ray["tail_prompt_id"] == "actual-ray"
    assert ray["tainted"] is True


def test_exact_cleanup_clears_only_matching_ray_tail_and_taints_pool(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=locked,
        exact_snapshot=snapshot,
    )
    before = database.get_raylight_runtime_state()
    ownership = database.get_prompt_ownership("child-ray")
    assert before is not None and ownership is not None
    confirmed_at = ownership.updated_at + timedelta(seconds=1)

    cleaned = database.confirm_prompt_cleanup(
        "child-ray",
        expected_revision=0,
        evidence=ExactCancelConfirmedEvidence(
            prompt_id="child-ray",
            confirmation_id="cancel-ray",
            confirmed_at=confirmed_at,
        ),
        stage="cancelled",
        updated_at=confirmed_at,
    )

    assert cleaned is not None
    after = database.get_raylight_runtime_state()
    assert after is not None
    assert after["epoch"] == before["epoch"]
    assert after["current"] == before["current"]
    assert after["tail_prompt_id"] is None
    assert after["tail_action"] is None
    assert after["tainted"] is True
    assert "tail_terminal_certificate" not in after


def test_typed_restart_certificate_atomically_releases_ray_tail(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=locked,
        exact_snapshot=snapshot,
    )
    submitting = database.get_prompt_ownership("child-ray")
    assert submitting is not None
    database.record_prompt_submission_receipt(
        "child-ray",
        expected_revision=submitting.ownership_revision,
        actual_prompt_id="actual-ray-restart",
        state="owned_actual_id",
        updated_at=submitting.updated_at + timedelta(seconds=1),
    )
    marked, first_claim = database.mark_job_cancel_requested("job-1")
    assert marked is not None and first_claim
    database.update_job(
        "job-1", status="cancelling", stage="restart_certificate_required"
    )
    database.update_job_child(
        "child-ray", status="cancelling", stage="restart_certificate_required"
    )
    before = database.get_raylight_runtime_state()
    assert before is not None and before["tail_prompt_id"] == "actual-ray-restart"
    replacement = snapshot.endpoint_identity.model_copy(
        update={"runtime_instance_id": "replacement-ray-boot"}
    )

    settled = database.confirm_comfy_restart_recovery(
        "job-1", current_endpoint_identity=replacement
    )

    assert settled["status"] == "cancelled"
    assert settled["stage"] == "cancelled_after_confirmed_comfy_restart"
    child = database.get_job_child("child-ray")
    ownership = database.get_prompt_ownership("child-ray")
    assert child is not None and child["status"] == "cancelled"
    assert ownership is not None and ownership.state == "cleanup_confirmed"
    certificate = ownership.cleanup_certificate
    assert isinstance(certificate, EndpointRestartCertificate)
    assert certificate.prompt_id == "actual-ray-restart"
    assert certificate.endpoint_identity == snapshot.endpoint_identity
    assert certificate.restart_id == "replacement-ray-boot"
    after = database.get_raylight_runtime_state()
    assert after is not None
    assert after["epoch"] == before["epoch"]
    assert after["current"] == before["current"]
    assert after["tail_prompt_id"] is None
    assert after["tail_action"] is None
    assert after["tainted"] is True

    assert (
        database.confirm_comfy_restart_recovery(
            "job-1", current_endpoint_identity=replacement
        )["updated_at"]
        == settled["updated_at"]
    )
    with pytest.raises(
        ExecutionEvidenceConflict, match="different replacement boot"
    ):
        database.confirm_comfy_restart_recovery(
            "job-1",
            current_endpoint_identity=replacement.model_copy(
                update={"runtime_instance_id": "later-ray-boot"}
            ),
        )


def test_typed_restart_certificate_rolls_back_all_ledgers_on_ray_failure(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=locked,
        exact_snapshot=snapshot,
    )
    database.mark_job_cancel_requested("job-1")
    database.update_job(
        "job-1", status="cancelling", stage="restart_certificate_required"
    )
    database.update_job_child(
        "child-ray", status="cancelling", stage="restart_certificate_required"
    )
    parent_before = database.get_job("job-1")
    child_before = database.get_job_child("child-ray")
    ownership_before = database.get_prompt_ownership("child-ray")
    ray_before = database.get_raylight_runtime_state()
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_restart_ray_cleanup "
            "BEFORE UPDATE ON raylight_runtime_state "
            "BEGIN SELECT RAISE(ABORT, 'injected restart ray failure'); END"
        )

    with pytest.raises(
        sqlite3.IntegrityError, match="injected restart ray failure"
    ):
        database.confirm_comfy_restart_recovery(
            "job-1",
            current_endpoint_identity=snapshot.endpoint_identity.model_copy(
                update={"runtime_instance_id": "replacement-ray-boot"}
            ),
        )

    assert database.get_job("job-1") == parent_before
    assert database.get_job_child("child-ray") == child_before
    assert database.get_prompt_ownership("child-ray") == ownership_before
    assert database.get_raylight_runtime_state() == ray_before


def test_restart_confirmation_cannot_relabel_fully_released_typed_job(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    _create_parent(database, draft=fixture_builder._draft("t2v"))
    database.create_job_execution_plan("job-1", compiled_plan())
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=_standard_locked_plan(),
        exact_snapshot=exact_snapshot(),
    )
    _persist_observed_success(
        database,
        "child-1",
        output=OutputDescriptor(
            filename="result.mp4",
            subfolder="director/database-standard",
        ),
        has_audio=False,
    )
    database.mark_job_cancel_requested("job-1")
    database.update_job(
        "job-1", status="cancelling", stage="restart_certificate_required"
    )
    parent_before = database.get_job("job-1")
    child_before = database.get_job_child("child-1")
    replacement = endpoint_identity().model_copy(
        update={"runtime_instance_id": "replacement-standard-boot"}
    )

    with pytest.raises(
        ExecutionEvidenceConflict, match="no unreleased prompt ownership"
    ):
        database.confirm_comfy_restart_recovery(
            "job-1", current_endpoint_identity=replacement
        )

    assert database.get_job("job-1") == parent_before
    assert database.get_job_child("child-1") == child_before

    database.update_job_child("child-1", prompt_id=None)
    with pytest.raises(ExecutionEvidenceConflict, match="mix legacy prompt owners"):
        database.confirm_comfy_restart_recovery(
            "job-1", current_endpoint_identity=replacement
        )


def test_restart_certificate_does_not_clear_a_newer_ray_tail(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    plan, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", plan)
    database.persist_job_child_submission_intent(
        "job-1", locked_plan=locked, exact_snapshot=snapshot
    )
    database.mark_job_cancel_requested("job-1")
    database.update_job(
        "job-1", status="cancelling", stage="restart_certificate_required"
    )
    database.update_job_child(
        "child-ray", status="cancelling", stage="restart_certificate_required"
    )
    newer = dict(database.get_raylight_runtime_state() or {})
    newer.update(
        tail_prompt_id="newer-replacement-tail",
        tail_action="ray_unit",
        tainted=False,
    )
    newer.pop("tail_terminal_certificate", None)
    database.put_raylight_runtime_state(newer)

    database.confirm_comfy_restart_recovery(
        "job-1",
        current_endpoint_identity=snapshot.endpoint_identity.model_copy(
            update={"runtime_instance_id": "replacement-ray-boot"}
        ),
    )

    assert database.get_raylight_runtime_state() == newer


def test_pure_legacy_restart_confirmation_remains_supported(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database, "legacy-job")
    now = NOW.isoformat()
    database.create_job_child(
        {
            "id": "legacy-child",
            "job_id": "legacy-job",
            "group_index": 0,
            "family": "fl2va",
            "backend": "standard",
            "segment_ids": ["legacy-segment"],
            "output_nodes": {"legacy-segment": "30"},
            "status": "cancelling",
            "progress": 0.0,
            "stage": "restart_cancel_unconfirmed",
            "prompt_id": "legacy-prompt",
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )
    database.mark_job_cancel_requested("legacy-job")
    database.update_job(
        "legacy-job", status="cancelling", stage="restart_cancel_unconfirmed"
    )
    before_schema = _stage4_schema_names(database)

    settled = database.confirm_comfy_restart_recovery(
        "legacy-job", current_endpoint_identity=endpoint_identity()
    )

    assert settled["status"] == "cancelled"
    assert database.get_job_child("legacy-child")["status"] == "cancelled"
    assert _stage4_schema_names(database) == before_schema == set()


def test_cleanup_and_restart_certificate_use_same_ownership_cas(tmp_path) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=_standard_locked_plan(),
        exact_snapshot=exact_snapshot(),
    )
    before_pending = database.get_prompt_ownership("child-1")
    assert before_pending is not None
    pending_at = before_pending.updated_at + timedelta(seconds=1)
    pending = database.compare_and_set_prompt_ownership(
        "child-1",
        expected_revision=0,
        state="cancel_pending",
        updated_at=pending_at,
    )
    assert pending is not None
    cancel = ExactCancelConfirmedEvidence(
        prompt_id="child-1",
        confirmation_id="cancel-1",
        confirmed_at=pending_at + timedelta(seconds=1),
    )

    cleaned = database.confirm_prompt_cleanup(
        "child-1",
        expected_revision=1,
        evidence=cancel,
        stage="cancelled",
        updated_at=pending_at + timedelta(seconds=1),
    )

    assert cleaned is not None
    child, ownership = cleaned
    assert child["status"] == "cancelled"
    assert ownership.state == "cleanup_confirmed"
    assert ownership.cleanup_certificate == cancel

    # A restart certificate is additionally bound to the immutable endpoint.
    _create_parent(database, "job-2")
    database.create_job_execution_plan("job-2", compiled_plan())
    second_base_plan = _standard_locked_plan()
    second_locked = second_base_plan.units[0].model_copy(
        update={"child_id": "child-2", "requested_prompt_id": "requested-2"}
    )
    second_plan = second_base_plan.model_copy(
        update={"units": (second_locked,)}
    )
    database.persist_job_child_submission_intent(
        "job-2", locked_plan=second_plan, exact_snapshot=exact_snapshot()
    )
    second_ownership = database.get_prompt_ownership("child-2")
    assert second_ownership is not None
    restart_at = second_ownership.updated_at + timedelta(seconds=1)
    wrong_restart = EndpointRestartCertificate(
        certificate_version=1,
        prompt_id="requested-2",
        endpoint_identity=endpoint_identity(runtime_instance_id="other-runtime"),
        restart_id="restart-1",
        queue_and_history_cleared=True,
        confirmed_at=restart_at,
    )
    with pytest.raises(ExecutionEvidenceConflict, match="endpoint"):
        database.confirm_prompt_cleanup(
            "child-2",
            expected_revision=0,
            evidence=wrong_restart,
            stage="restart-cleanup",
            updated_at=restart_at,
        )
    assert database.get_prompt_ownership("child-2").state == "submitting"

    restart = EndpointRestartCertificate(
        certificate_version=1,
        prompt_id="requested-2",
        endpoint_identity=endpoint_identity(),
        restart_id="restart-2",
        queue_and_history_cleared=True,
        confirmed_at=restart_at + timedelta(seconds=1),
    )
    restarted = database.confirm_prompt_cleanup(
        "child-2",
        expected_revision=0,
        evidence=restart,
        stage="restart-cleanup",
        updated_at=restart_at + timedelta(seconds=1),
    )
    assert restarted is not None
    restarted_child, restarted_ownership = restarted
    assert restarted_child["status"] == "cancelled"
    assert restarted_ownership.cleanup_certificate == restart
