from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import json

import pytest
from pydantic import ValidationError

from directordeck.database import ExecutionEvidenceConflict
from directordeck.host_artifacts import (
    HostOutputProbeResult,
    PermanentHostOutputProbeError,
)
from directordeck.workflow.execution import (
    ExactCancelConfirmedEvidence,
    HistoryTerminalEvidence,
    sha256_document_digest,
)

from .conftest import adapt_legacy_workflow_requests
from .test_execution_evidence_database import (
    _create_parent,
    _database,
    _ray_contracts,
)
from .test_per_segment_execution import (
    _complete_fake_prompt,
    _reconcile,
    _segment,
    _timeline,
    _wait_for_submission_jobs,
)


@pytest.fixture(autouse=True)
def _stage6_v5_request_adapter(client, monkeypatch) -> None:
    adapt_legacy_workflow_requests(client, monkeypatch)


@pytest.mark.parametrize(
    "operation",
    (
        "submission_receipt",
        "ownership_cas",
        "cleanup_confirmation",
        "terminal_confirmation",
    ),
)
def test_prompt_mutations_reject_partial_exact_evidence_deletion_atomically(
    tmp_path,
    operation: str,
) -> None:
    database = _database(tmp_path)
    _create_parent(database)
    compiled, locked, snapshot = _ray_contracts()
    database.create_job_execution_plan("job-1", compiled)
    database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=locked,
        exact_snapshot=snapshot,
    )

    ownership = database.get_prompt_ownership("child-ray")
    child = database.get_job_child("child-ray")
    ray = database.get_raylight_runtime_state()
    assert ownership is not None and ownership.state == "submitting"
    assert child is not None and child["prompt_id"] == "child-ray"
    assert ray is not None and ray["tail_prompt_id"] == "child-ray"

    with database.connect() as connection:
        connection.execute(
            "DELETE FROM job_child_execution_evidence WHERE child_id = ?",
            ("child-ray",),
        )
    assert database.get_job_child_execution_evidence("child-ray") is None
    assert database.get_prompt_ownership("child-ray") == ownership

    child_before = deepcopy(database.get_job_child("child-ray"))
    ownership_before = database.get_prompt_ownership("child-ray")
    ray_before = deepcopy(database.get_raylight_runtime_state())
    updated_at = ownership.updated_at + timedelta(seconds=1)

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="immutable exact execution evidence",
    ):
        if operation == "submission_receipt":
            database.record_prompt_submission_receipt(
                "child-ray",
                expected_revision=ownership.ownership_revision,
                actual_prompt_id="actual-after-exact-row-deletion",
                state="owned_actual_id",
                updated_at=updated_at,
            )
        elif operation == "ownership_cas":
            database.compare_and_set_prompt_ownership(
                "child-ray",
                expected_revision=ownership.ownership_revision,
                state="unconfirmed",
                updated_at=updated_at,
            )
        elif operation == "cleanup_confirmation":
            database.confirm_prompt_cleanup(
                "child-ray",
                expected_revision=ownership.ownership_revision,
                evidence=ExactCancelConfirmedEvidence(
                    prompt_id="child-ray",
                    confirmation_id="cancel-after-exact-row-deletion",
                    confirmed_at=updated_at,
                ),
                stage="cancelled",
                updated_at=updated_at,
            )
        else:
            database.confirm_prompt_terminal(
                "child-ray",
                expected_revision=ownership.ownership_revision,
                evidence=HistoryTerminalEvidence(
                    prompt_id="child-ray",
                    terminal_status="failed",
                    history_digest=sha256_document_digest(
                        {"status": "failed after exact row deletion"}
                    ),
                    observed_at=updated_at,
                ),
                outputs=[],
                stage="failed",
                error="upstream failed",
                updated_at=updated_at,
            )

    assert database.get_job_child("child-ray") == child_before
    assert database.get_prompt_ownership("child-ray") == ownership_before
    assert database.get_raylight_runtime_state() == ray_before


@pytest.mark.parametrize(
    ("failure_kind", "sensitive_value", "public_error"),
    (
        pytest.param(
            "permanent",
            "/private/stage4/render-output.mp4",
            "host output cannot be observed safely",
            id="permanent-error-with-absolute-path",
        ),
        pytest.param(
            "invalid_metadata",
            "token=stage4-sensitive-metadata",
            "host output probe returned invalid metadata",
            id="pydantic-error-with-token",
        ),
    ),
)
async def test_public_job_error_hides_sensitive_host_probe_diagnostics(
    client,
    fake_comfy,
    monkeypatch,
    failure_kind: str,
    sensitive_value: str,
    public_error: str,
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment(f"redacted-{failure_kind}"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_submission_jobs(client)
    database = client.director_app.state.database
    child = database.list_job_children(created.json()["id"])[0]

    if failure_kind == "permanent":
        private_error = PermanentHostOutputProbeError(
            f"unsafe host output path: {sensitive_value}"
        )
        assert sensitive_value in str(private_error)

        def fail_probe(_descriptor):
            raise private_error

    else:
        invalid_metadata = {
            "width": sensitive_value,
            "height": 1080,
            "fps": 24.0,
            "frame_count": 24,
            "duration_seconds": 1.0,
            "has_audio": True,
            "media_probe_version": "stage4-invalid-metadata-v1",
        }
        with pytest.raises(ValidationError) as validation:
            HostOutputProbeResult.model_validate(invalid_metadata)
        assert sensitive_value in str(validation.value)

        def fail_probe(_descriptor):
            return invalid_metadata

    monkeypatch.setattr(fake_comfy, "probe_output", fail_probe)
    _complete_fake_prompt(fake_comfy, child)
    reconciled = await _reconcile(client, created.json())
    response = await client.get(f"/api/jobs/{created.json()['id']}")

    assert reconciled["status"] == "failed"
    assert response.status_code == 200, response.text
    public = response.json()
    assert public["status"] == "failed"
    assert public["error"] == public_error
    assert public["error_summary"] == public_error
    assert public["children"][0]["error"] == public_error
    assert sensitive_value not in json.dumps(public, ensure_ascii=False)
