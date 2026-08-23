from __future__ import annotations

import sqlite3
from datetime import timedelta
from typing import Literal

import pytest

from directordeck.database import ExecutionEvidenceConflict
from directordeck.workflow.execution import (
    HistoryTerminalEvidence,
    LockedSubmissionPlan,
    sha256_document_digest,
)

from .test_execution_evidence_database import (
    _control_pair_contracts,
    _create_parent,
    _database,
)


ControlCorruption = Literal[
    "missing_child",
    "missing_exact_evidence",
    "stored_locked_digest",
    "dependency_locked_digest",
    "nonterminal_ownership",
]


def _persist_terminal_control(database):
    plan, planner, pair, continuation, before, clean = (
        _control_pair_contracts()
    )
    _create_parent(database)
    database.create_job_execution_plan("job-1", plan)
    database.put_raylight_runtime_state(before)

    control_snapshot = planner.exact_snapshot(pair, pair.units[0])
    control_child, submitting = database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=pair,
        exact_snapshot=control_snapshot,
    )
    assert control_child["id"] == "child-control"
    assert submitting.state == "submitting"
    after_intent = database.get_raylight_runtime_state()
    assert after_intent is not None
    assert after_intent["tail_prompt_id"] == "child-control"
    assert after_intent["tail_action"] == "shutdown"
    assert after_intent["tainted"] is True

    receipt_at = submitting.updated_at + timedelta(seconds=1)
    owned = database.record_prompt_submission_receipt(
        "child-control",
        expected_revision=submitting.ownership_revision,
        actual_prompt_id="child-control",
        state="owned_requested_id",
        updated_at=receipt_at,
    )
    assert owned is not None
    assert owned.state == "owned_requested_id"

    terminal_at = owned.updated_at + timedelta(seconds=1)
    terminal = database.confirm_prompt_terminal(
        "child-control",
        expected_revision=owned.ownership_revision,
        evidence=HistoryTerminalEvidence(
            prompt_id=owned.effective_prompt_id,
            terminal_status="succeeded",
            history_digest=sha256_document_digest(
                {
                    "prompt_id": owned.effective_prompt_id,
                    "status": "success",
                }
            ),
            observed_at=terminal_at,
        ),
        outputs=[],
        stage="RayLight released",
        error=None,
        updated_at=terminal_at,
    )
    assert terminal is not None
    terminal_child, terminal_ownership = terminal
    assert terminal_child["status"] == "succeeded"
    assert terminal_ownership.state == "terminal_confirmed"
    assert database.get_raylight_runtime_state() == clean
    return planner, continuation, clean


def _corrupt_control(
    database,
    continuation: LockedSubmissionPlan,
    corruption: ControlCorruption,
) -> LockedSubmissionPlan:
    if corruption == "dependency_locked_digest":
        dependency = continuation.control_dependency
        assert dependency is not None
        return continuation.model_copy(
            update={
                "control_dependency": dependency.model_copy(
                    update={
                        "original_locked_plan_digest": sha256_document_digest(
                            {"forged": "control dependency"}
                        )
                    }
                )
            }
        )

    with sqlite3.connect(database.path) as connection:
        if corruption == "missing_child":
            # Use a raw connection so the orphaned evidence remains available:
            # the continuation must require the live same-job child itself.
            connection.execute(
                "DELETE FROM job_children WHERE id = ?",
                ("child-control",),
            )
        elif corruption == "missing_exact_evidence":
            connection.execute(
                "DELETE FROM job_child_execution_evidence WHERE child_id = ?",
                ("child-control",),
            )
        elif corruption == "stored_locked_digest":
            connection.execute(
                "DROP TRIGGER job_child_execution_evidence_immutable"
            )
            connection.execute(
                "UPDATE job_child_execution_evidence "
                "SET locked_submission_plan_digest = ? WHERE child_id = ?",
                ("sha256-" + "0" * 64, "child-control"),
            )
        else:
            connection.execute(
                "UPDATE prompt_ownership "
                "SET state = 'unconfirmed', cleanup_certificate = NULL "
                "WHERE child_id = ?",
                ("child-control",),
            )
    return continuation


def _assert_segment_intent_absent(database, expected_ray) -> None:
    assert database.get_job_child("child-after-control") is None
    assert (
        database.get_job_child_execution_evidence("child-after-control")
        is None
    )
    assert database.get_prompt_ownership("child-after-control") is None
    assert database.get_raylight_runtime_state() == expected_ray


def test_segment_continuation_accepts_complete_terminal_control_chain(
    tmp_path,
) -> None:
    database = _database(tmp_path)
    planner, continuation, clean = _persist_terminal_control(database)
    dependency = continuation.control_dependency
    assert dependency is not None
    assert dependency.control_child_id == "child-control"

    segment_snapshot = planner.exact_snapshot(
        continuation,
        continuation.units[0],
    )
    child, ownership = database.persist_job_child_submission_intent(
        "job-1",
        locked_plan=continuation,
        exact_snapshot=segment_snapshot,
    )

    assert child["id"] == "child-after-control"
    assert child["stage"] == "submitting"
    assert ownership.state == "submitting"
    assert (
        database.get_job_child_execution_evidence("child-after-control")
        is not None
    )
    expected_after = continuation.model_dump(mode="json")[
        "ray_ledger_after_intent"
    ]
    assert expected_after != clean
    assert database.get_raylight_runtime_state() == expected_after


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("missing_child", "missing its required control child"),
        ("missing_exact_evidence", "incomplete control evidence"),
        ("stored_locked_digest", "execution evidence digest/index mismatch"),
        ("dependency_locked_digest", "not terminal-confirmed"),
        ("nonterminal_ownership", "not terminal-confirmed"),
    ),
)
def test_segment_continuation_fails_closed_and_rolls_back_all_intent_ledgers(
    tmp_path,
    corruption: ControlCorruption,
    message: str,
) -> None:
    database = _database(tmp_path)
    planner, continuation, clean = _persist_terminal_control(database)
    continuation = _corrupt_control(database, continuation, corruption)
    ray_before_attempt = database.get_raylight_runtime_state()
    assert ray_before_attempt == clean

    with pytest.raises(ExecutionEvidenceConflict, match=message):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=continuation,
            exact_snapshot=planner.exact_snapshot(
                continuation,
                continuation.units[0],
            ),
        )

    _assert_segment_intent_absent(database, ray_before_attempt)
