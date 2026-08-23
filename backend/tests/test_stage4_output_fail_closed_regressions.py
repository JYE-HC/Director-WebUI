from __future__ import annotations

import json
import sqlite3
from datetime import timedelta

import pytest

from directordeck.database import ExecutionEvidenceConflict
from directordeck.media import VideoProxy
from directordeck.schemas import VideoMetadata
from directordeck.workflow.execution import OutputDescriptor

from . import extensible_workflow_v0_fixture_builder as fixture_builder
from .conftest import VIDEO_METADATA
from .test_execution_evidence_database import (
    NOW,
    _create_parent,
    _persist_observed_success,
    _standard_locked_plan,
    compiled_plan,
    exact_snapshot,
)


def _persist_typed_success(database) -> tuple[str, str, str]:
    job_id = "job-1"
    child_id = "child-1"
    segment_id = "baseline-t2v"
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
    _persist_observed_success(
        database,
        child_id,
        output=OutputDescriptor(
            filename="trusted-stage4.mp4",
            subfolder="segments",
            type="output",
        ),
        has_audio=True,
    )
    database.update_job(
        job_id,
        status="succeeded",
        progress=1.0,
        stage="completed",
        completed_at=(NOW + timedelta(minutes=1)).isoformat(),
    )
    return job_id, child_id, segment_id


def _stub_import_transcode(monkeypatch) -> None:
    monkeypatch.setattr(
        "directordeck.task_management.create_24fps_proxy_bytes",
        lambda _content, _suffix: VideoProxy(
            content=b"normalized-stage4",
            filename_suffix=".mp4",
            metadata=VideoMetadata.model_validate(VIDEO_METADATA),
        ),
    )


async def test_typed_parent_plan_prevents_legacy_fallback_after_all_child_markers_are_lost(
    client,
    monkeypatch,
) -> None:
    database = client.director_app.state.database
    job_id, child_id, segment_id = _persist_typed_success(database)
    _stub_import_transcode(monkeypatch)

    forged_output = {
        "node_id": "forged-node",
        "filename": "forged-marker-loss.mp4",
        "subfolder": "attacker",
        "type": "output",
    }
    with database.connect() as connection:
        connection.execute(
            "UPDATE job_children SET output_nodes = ?, outputs = ? WHERE id = ?",
            (
                json.dumps({segment_id: "forged-node"}),
                json.dumps([forged_output]),
                child_id,
            ),
        )
        connection.execute(
            "DELETE FROM segment_take_observed_artifacts "
            "WHERE source_child_id = ?",
            (child_id,),
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

    assert database.get_job_execution_plan(job_id) is not None
    details = await client.get(f"/api/jobs/{job_id}")
    media = await client.get(
        f"/api/jobs/{job_id}/segment-output",
        params={"segment_id": segment_id},
    )
    imported = await client.post(
        f"/api/jobs/{job_id}/import-output",
        json={"segment_id": segment_id},
    )

    assert details.status_code == 200, details.text
    payload = details.json()
    assert {
        "child_outputs": payload["children"][0]["outputs"],
        "segment_results": payload["segment_results"],
        "media_status": media.status_code,
        "import_status": imported.status_code,
    } == {
        "child_outputs": [],
        "segment_results": [],
        "media_status": 404,
        "import_status": 409,
    }


@pytest.mark.parametrize(
    ("input_name", "forged_value"),
    (
        ("prompt", "coherently forged prompt"),
        ("width", True),
    ),
)
def test_exact_evidence_reader_rejects_coherent_snapshot_that_differs_from_locked_unit(
    tmp_path,
    input_name: str,
    forged_value: object,
) -> None:
    from directordeck.database import Database

    database = Database(tmp_path / "coherent-exact-forgery.sqlite3")
    database.initialize()
    _create_parent(database)
    database.create_job_execution_plan("job-1", compiled_plan())
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=_standard_locked_plan(),
        exact_snapshot=exact_snapshot(),
    )

    forged_document = json.loads(database._contract_json(exact_snapshot()))
    forged_document["exact_prompt"]["10"]["inputs"][input_name] = forged_value
    forged_json = database._contract_json(forged_document)
    forged_digest = database._execution_document_digest(forged_document)
    with sqlite3.connect(database.path) as connection:
        connection.execute("DROP TRIGGER job_child_execution_evidence_immutable")
        connection.execute(
            "UPDATE job_child_execution_evidence "
            "SET exact_prompt_snapshot = ?, exact_prompt_snapshot_digest = ? "
            "WHERE child_id = ?",
            (forged_json, forged_digest, "child-1"),
        )

    with pytest.raises(ExecutionEvidenceConflict):
        database.get_job_child_execution_evidence("child-1")


@pytest.mark.parametrize("drifted_status", ["failed", "cancelled"])
async def test_live_non_succeeded_child_invalidates_observed_output_everywhere(
    client,
    monkeypatch,
    drifted_status: str,
) -> None:
    database = client.director_app.state.database
    job_id, child_id, segment_id = _persist_typed_success(database)
    _stub_import_transcode(monkeypatch)
    with database.connect() as connection:
        connection.execute(
            "UPDATE job_children SET status = ? WHERE id = ?",
            (drifted_status, child_id),
        )

    artifact_error: Exception | None = None
    try:
        database.get_observed_artifact(child_id)
    except ExecutionEvidenceConflict as exc:
        artifact_error = exc
    media = await client.get(
        f"/api/jobs/{job_id}/segment-output",
        params={"segment_id": segment_id},
    )
    imported = await client.post(
        f"/api/jobs/{job_id}/import-output",
        json={"segment_id": segment_id},
    )

    assert {
        "artifact_rejected": isinstance(
            artifact_error, ExecutionEvidenceConflict
        ),
        "media_status": media.status_code,
        "import_status": imported.status_code,
    } == {
        "artifact_rejected": True,
        "media_status": 404,
        "import_status": 409,
    }
