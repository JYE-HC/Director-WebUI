from __future__ import annotations

from typing import Literal

import pytest

from directordeck.compiler import (
    align_h3_frames,
    timeline_segment_take_fingerprint,
)
from directordeck.database import ExecutionEvidenceConflict
from directordeck.workflow.execution import ObservedArtifactSpec

from .test_execution_evidence_database import (
    _create_parent,
    _database,
    _historical_continuity_contracts,
    _insert_take,
    _persist_same_run_predecessor,
    _typed_observation_contracts,
)


GeometryField = Literal["width", "height", "fps", "frame_count"]


def _mismatched_geometry(
    geometry: tuple[int, int, float, int],
    field: GeometryField,
) -> tuple[int, int, float, int]:
    width, height, fps, frame_count = geometry
    if field == "width":
        width += 8
    elif field == "height":
        height += 8
    elif field == "fps":
        fps += 1.0
    else:
        frame_count += 1
    return width, height, fps, frame_count


@pytest.mark.parametrize("field", ("width", "height", "fps", "frame_count"))
def test_same_run_continuity_rejects_actual_predecessor_geometry_drift(
    tmp_path,
    field: GeometryField,
) -> None:
    database = _database(tmp_path)
    (
        _draft,
        planner,
        successor_plan,
        _evidence,
        output,
        _terminal_output,
    ) = _persist_same_run_predecessor(database, terminal=False)
    ownership, receipt, artifact, observed_at = _typed_observation_contracts(
        database,
        "child-predecessor",
        output=output,
        has_audio=True,
    )
    geometry = _mismatched_geometry(
        (artifact.width, artifact.height, artifact.fps, artifact.frame_count),
        field,
    )
    drifted_artifact = ObservedArtifactSpec.model_validate(
        {
            **artifact.model_dump(mode="json"),
            "width": geometry[0],
            "height": geometry[1],
            "fps": geometry[2],
            "frame_count": geometry[3],
            "duration_seconds": geometry[3] / geometry[2],
        }
    )
    recorded = database.record_output_observation_receipt(
        "child-predecessor",
        expected_revision=ownership.ownership_revision,
        receipt=receipt,
        updated_at=observed_at,
    )
    assert recorded is not None
    database.finalize_observed_artifact(
        "child-predecessor",
        artifact=drifted_artifact,
        updated_at=observed_at,
    )

    with pytest.raises(
        ExecutionEvidenceConflict,
        match="same-run continuity output differs",
    ):
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=successor_plan,
            exact_snapshot=planner.exact_snapshot(
                successor_plan,
                successor_plan.units[0],
            ),
        )

    assert database.get_job_child("child-successor") is None


@pytest.mark.parametrize("field", ("width", "height", "fps", "frame_count"))
def test_historical_continuity_rejects_take_whose_actual_geometry_differs_from_fingerprint(
    tmp_path,
    field: GeometryField,
) -> None:
    database = _database(tmp_path)
    draft, execution_plan, planner, plan, take = _historical_continuity_contracts()
    _create_parent(database, draft=draft, project_id="project-historical")
    database.create_job_execution_plan("job-1", execution_plan)
    predecessor = draft.segments[0]
    authored_geometry = (
        draft.render.width,
        draft.render.height,
        draft.render.fps,
        align_h3_frames(predecessor.duration_seconds, draft.render.fps),
    )
    _insert_take(
        database,
        take,
        geometry=_mismatched_geometry(authored_geometry, field),
    )
    fingerprint = timeline_segment_take_fingerprint(draft, predecessor)

    reader_error: Exception | None = None
    try:
        database.find_latest_observed_segment_take(
            predecessor.id,
            fingerprint,
            project_id="project-historical",
        )
    except ExecutionEvidenceConflict as exc:
        reader_error = exc

    submission_error: Exception | None = None
    try:
        database.persist_job_child_submission_intent(
            "job-1",
            locked_plan=plan,
            exact_snapshot=planner.exact_snapshot(plan, plan.units[0]),
        )
    except ExecutionEvidenceConflict as exc:
        submission_error = exc

    assert {
        "reader_rejected": isinstance(reader_error, ExecutionEvidenceConflict),
        "submission_rejected": isinstance(
            submission_error,
            ExecutionEvidenceConflict,
        ),
        "successor_persisted": database.get_job_child(
            "child-historical-successor"
        )
        is not None,
    } == {
        "reader_rejected": True,
        "submission_rejected": True,
        "successor_persisted": False,
    }
