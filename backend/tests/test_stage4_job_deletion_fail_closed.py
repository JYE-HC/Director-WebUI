from __future__ import annotations

from datetime import timedelta
import json

from directordeck.schemas import utc_now

from . import extensible_workflow_v0_fixture_builder as fixture_builder
from .test_execution_evidence_database import (
    NOW,
    _create_parent,
    _persist_assembly_sources,
    _standard_locked_plan,
    compiled_plan,
    exact_snapshot,
)
from .test_stage4_output_fail_closed_regressions import _persist_typed_success


def _persist_unreleased_terminal_typed_job(database) -> tuple[str, str]:
    job_id = "job-1"
    child_id = "child-1"
    _create_parent(
        database,
        job_id=job_id,
        draft=fixture_builder._draft("t2v"),
    )
    database.create_job_execution_plan(job_id, compiled_plan())
    database.persist_job_child_submission_intent(
        job_id,
        locked_plan=_standard_locked_plan(),
        exact_snapshot=exact_snapshot(),
    )
    ownership = database.get_prompt_ownership(child_id)
    assert ownership is not None and ownership.state == "submitting"

    # Model a stale/corrupt lifecycle projection: the local rows look terminal,
    # but no upstream terminal or cleanup certificate ever released the prompt.
    completed_at = (NOW + timedelta(minutes=1)).isoformat()
    database.update_job_child(
        child_id,
        status="failed",
        progress=1.0,
        stage="failed_without_prompt_release",
        error="synthetic terminal projection",
        completed_at=completed_at,
    )
    database.update_job(
        job_id,
        status="failed",
        progress=1.0,
        stage="failed_without_prompt_release",
        error="synthetic terminal projection",
        completed_at=completed_at,
    )
    return job_id, child_id


async def test_typed_marker_survives_total_auxiliary_evidence_loss_and_hides_legacy_outputs(
    client,
) -> None:
    database = client.director_app.state.database
    claimed, assembly = _persist_assembly_sources(database)
    settled = database.finalize_observed_assembly_artifact(
        "job-1",
        expected_updated_at=str(claimed["updated_at"]),
        artifact=assembly,
        updated_at=NOW + timedelta(minutes=11),
    )
    assert settled is not None and settled["status"] == "succeeded"
    children = database.list_job_children("job-1")
    assert len(children) == 2

    forged_parent = {
        "node_id": "forged-assembly",
        "filename": "forged-parent.mp4",
        "subfolder": "attacker",
        "type": "output",
    }
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET outputs = ? WHERE id = ?",
            (json.dumps([forged_parent]), "job-1"),
        )
        for child in children:
            segment_id = str(child["segment_ids"][0])
            forged_child = {
                "node_id": f"forged-{segment_id}",
                "filename": f"forged-{segment_id}.mp4",
                "subfolder": "attacker",
                "type": "output",
            }
            connection.execute(
                "UPDATE job_children SET output_nodes = ?, outputs = ? "
                "WHERE id = ?",
                (
                    json.dumps({segment_id: forged_child["node_id"]}),
                    json.dumps([forged_child]),
                    child["id"],
                ),
            )

        # Simulate total loss of every auxiliary Stage-4 authority. The marker
        # on the parent row is intentionally not auxiliary and must remain the
        # durable classification boundary against mutable legacy fallback.
        connection.execute(
            "DELETE FROM job_observed_assembly_artifacts WHERE job_id = ?",
            ("job-1",),
        )
        connection.execute(
            "DELETE FROM segment_take_observed_artifacts "
            "WHERE source_child_id IN (?, ?)",
            tuple(child["id"] for child in children),
        )
        connection.execute(
            "DELETE FROM segment_takes WHERE source_job_id = ?",
            ("job-1",),
        )
        connection.execute(
            "DELETE FROM job_child_output_receipts WHERE child_id IN (?, ?)",
            tuple(child["id"] for child in children),
        )
        connection.execute(
            "DELETE FROM prompt_ownership WHERE child_id IN (?, ?)",
            tuple(child["id"] for child in children),
        )
        connection.execute(
            "DELETE FROM job_child_execution_evidence WHERE child_id IN (?, ?)",
            tuple(child["id"] for child in children),
        )
        connection.execute(
            "DELETE FROM job_execution_plans WHERE job_id = ?",
            ("job-1",),
        )
        marker = connection.execute(
            "SELECT execution_contract_version FROM jobs WHERE id = ?",
            ("job-1",),
        ).fetchone()
        remaining = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in (
                "job_execution_plans",
                "job_child_execution_evidence",
                "prompt_ownership",
                "job_child_output_receipts",
                "segment_take_observed_artifacts",
                "job_observed_assembly_artifacts",
                "segment_takes",
            )
        }

    assert marker is not None and int(marker["execution_contract_version"]) == 1
    assert remaining == {table: 0 for table in remaining}

    details = await client.get("/api/jobs/job-1")
    first_media = await client.get(
        "/api/jobs/job-1/segment-output",
        params={"segment_id": children[0]["segment_ids"][0]},
    )

    assert details.status_code == 200, details.text
    public = details.json()
    assert public["outputs"] == []
    assert public["output_files"] == []
    assert public["segment_results"] == []
    assert all(child["outputs"] == [] for child in public["children"])
    assert first_media.status_code == 404


async def test_parent_marker_alone_prevents_legacy_output_fallback_after_children_are_lost(
    client,
) -> None:
    database = client.director_app.state.database
    job_id, child_id, _segment_id = _persist_typed_success(database)
    forged_parent = {
        "node_id": "forged-parent-node",
        "filename": "forged-parent-only.mp4",
        "subfolder": "attacker",
        "type": "output",
    }
    with database.connect() as connection:
        connection.execute(
            "UPDATE jobs SET outputs = ? WHERE id = ?",
            (json.dumps([forged_parent]), job_id),
        )
        connection.execute(
            "DELETE FROM segment_take_observed_artifacts "
            "WHERE source_child_id = ?",
            (child_id,),
        )
        connection.execute(
            "DELETE FROM segment_takes WHERE source_job_id = ?",
            (job_id,),
        )
        connection.execute(
            "DELETE FROM job_child_output_receipts WHERE child_id = ?",
            (child_id,),
        )
        connection.execute(
            "DELETE FROM prompt_ownership WHERE child_id = ?",
            (child_id,),
        )
        connection.execute(
            "DELETE FROM job_child_execution_evidence WHERE child_id = ?",
            (child_id,),
        )
        connection.execute(
            "DELETE FROM job_children WHERE id = ?",
            (child_id,),
        )
        connection.execute(
            "DELETE FROM job_execution_plans WHERE job_id = ?",
            (job_id,),
        )
        marker = connection.execute(
            "SELECT execution_contract_version FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()

    assert marker is not None and int(marker["execution_contract_version"]) == 1
    details = await client.get(f"/api/jobs/{job_id}")
    media = await client.get(f"/api/jobs/{job_id}/outputs/0")
    imported = await client.post(
        f"/api/jobs/{job_id}/import-output",
        json={"output_index": 0},
    )
    deleted = await client.delete(f"/api/jobs/{job_id}")

    assert details.status_code == 200, details.text
    assert details.json()["outputs"] == []
    assert details.json()["output_files"] == []
    assert details.json()["children"] == []
    assert media.status_code == 404
    assert imported.status_code == 404
    assert deleted.status_code == 409
    assert database.get_job(job_id) is not None


async def test_single_delete_rejects_terminal_typed_job_with_unreleased_ownership(
    client,
) -> None:
    database = client.director_app.state.database
    job_id, child_id = _persist_unreleased_terminal_typed_job(database)
    ownership_before = database.get_prompt_ownership(child_id)
    evidence_before = database.get_job_child_execution_evidence(child_id)

    deleted = await client.delete(f"/api/jobs/{job_id}")

    assert deleted.status_code == 409, deleted.text
    assert database.get_job(job_id) is not None
    assert database.get_job_child(child_id) is not None
    assert database.get_prompt_ownership(child_id) == ownership_before
    assert database.get_job_child_execution_evidence(child_id) == evidence_before


async def test_bulk_clear_retains_terminal_typed_job_with_unreleased_ownership(
    client,
) -> None:
    database = client.director_app.state.database
    job_id, child_id = _persist_unreleased_terminal_typed_job(database)
    _create_parent(database, job_id="released-legacy-terminal")
    database.update_job(
        "released-legacy-terminal",
        status="failed",
        progress=1.0,
        stage="failed",
        completed_at=(NOW + timedelta(minutes=2)).isoformat(),
    )

    cleared = await client.delete("/api/jobs")

    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["deleted_count"] == 1
    assert cleared.json()["active_count"] == 1
    assert database.get_job("released-legacy-terminal") is None
    assert database.get_job(job_id) is not None
    assert database.get_job_child(child_id) is not None
    ownership = database.get_prompt_ownership(child_id)
    assert ownership is not None and ownership.state == "submitting"


async def test_fully_released_typed_terminal_job_can_be_deleted(client) -> None:
    database = client.director_app.state.database
    job_id, child_id, _segment_id = _persist_typed_success(database)
    ownership = database.get_prompt_ownership(child_id)
    assert ownership is not None and ownership.state == "terminal_confirmed"

    deleted = await client.delete(f"/api/jobs/{job_id}")

    assert deleted.status_code == 200, deleted.text
    assert database.get_job(job_id) is None
    assert database.get_job_child(child_id) is None


async def test_released_typed_job_can_delete_with_pre_stage4_terminal_empty_raykill_control(
    client,
) -> None:
    database = client.director_app.state.database
    job_id, child_id, _segment_id = _persist_typed_success(database)
    next_group_index = (
        max(child["group_index"] for child in database.list_job_children(job_id))
        + 1
    )
    now = utc_now()
    database.create_job_child(
        {
            "id": "legacy-empty-raykill-control",
            "job_id": job_id,
            "group_index": next_group_index,
            "family": "fl2va",
            "backend": "raylight",
            "segment_ids": [],
            "output_nodes": {},
            "status": "succeeded",
            "progress": 1.0,
            "stage": "RayLight safe switch completed",
            "prompt_id": "legacy-empty-raykill-prompt",
            "outputs": [],
            "error": None,
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
        }
    )
    assert database.get_prompt_ownership(child_id) is not None
    assert database.get_prompt_ownership("legacy-empty-raykill-control") is None

    deleted = await client.delete(f"/api/jobs/{job_id}")

    assert deleted.status_code == 200, deleted.text
    assert database.get_job(job_id) is None
    assert database.get_job_child("legacy-empty-raykill-control") is None
