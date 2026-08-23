from __future__ import annotations

from directordeck.execution.submission import LockedSubmissionPlanner
from directordeck.schemas import UnifiedTimelineDraftV5
from directordeck.workflow.v5_compat import compile_v5_execution_plan

from .test_workflow_execution_contracts import endpoint_identity
from .test_execution_evidence_database import _create_parent
from .test_workflow_v5_compat import _v4_pair, _v5_pair


def test_lock_time_and_database_audit_use_current_registry_for_bundle5_strict_nodes(
    tmp_path,
) -> None:
    v4, legacy_settings = _v4_pair()
    draft, settings = _v5_pair(v4, legacy_settings)
    document = draft.model_dump(mode="json")
    document["features"]["template_bundle_version"] = 5
    document["features"]["project"]["attention_backend_override"] = {
        "enabled": True,
        "params": {"mode": "pytorch"},
    }
    current = UnifiedTimelineDraftV5.model_validate(document)
    plan = compile_v5_execution_plan(
        current,
        settings,
        "stage8-lock-time-current-registry",
        [current.segments[0].id],
    )

    planner = LockedSubmissionPlanner(
        endpoint_identity(endpoint_key="embedded")
    )
    locked = planner.build_wave(
        plan,
        source_unit_ordinal=0,
        segment_child_id="stage8-strict-child",
    )

    strict = [
        node
        for node in locked.units[0].exact_prompt.values()
        if node["class_type"] == "DirectorStrictModelAttentionBackend"
    ]
    assert len(strict) == 1
    assert strict[0]["inputs"]["mode"] == "pytorch"

    from directordeck.database import Database

    database = Database(tmp_path / "stage8-current-registry.sqlite3")
    database.initialize()
    _create_parent(database, job_id="stage8-lock-time-current-registry", draft=current)
    database.create_job_execution_plan(
        "stage8-lock-time-current-registry",
        plan,
    )
    child, ownership = database.persist_job_child_submission_intent(
        "stage8-lock-time-current-registry",
        locked_plan=locked,
        exact_snapshot=planner.exact_snapshot(locked, locked.units[0]),
    )
    assert child["id"] == "stage8-strict-child"
    assert ownership.requested_prompt_id == "stage8-strict-child"
