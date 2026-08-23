from __future__ import annotations

from directordeck.schemas import UnifiedTimelineDraftV5

from .test_stage7_job_runtime_snapshot import (
    _active_mapped_lora_draft,
    _seed_readable_job,
    _settings_with_overrides,
    _user_mapping_record,
)


async def test_current_snapshot_uses_current_active_execution_projection(client) -> None:
    database = client.director_app.state.database
    draft = _active_mapped_lora_draft()
    settings = _settings_with_overrides(
        database.get_settings(),
        [_user_mapping_record("dedicated")],
    )
    _, settings_token = database.get_settings_authority()
    database.put_settings_v3_authority(
        settings,
        expected_authority_token=settings_token,
        schema_version=3,
    )
    _, revision = database.get_timeline_authority()
    database.validate_and_put_timeline_authority(
        draft,
        expected_revision=revision,
    )
    job_id = _seed_readable_job(database, draft, settings)

    baseline = await client.get(f"/api/jobs/{job_id}")
    assert baseline.status_code == 200, baseline.text
    assert baseline.json()["current_project"] is True
    assert baseline.json()["segment_results"][0]["current_snapshot"] is True

    # Neither an authored disabled value nor a segment outside the captured
    # selection participates in this job's execution identity.  The UI may
    # still report that the whole project changed, but the generated take is
    # an exact current execution snapshot.
    inactive_change = draft.model_dump(mode="json")
    inactive_change["features"]["project"]["attention_backend_override"] = {
        "enabled": False,
        "params": {"mode": "ck_int8"},
    }
    extra_segment = dict(inactive_change["segments"][0])
    extra_segment.update({
        "id": "stage8-unselected-segment",
        "title": "Not captured by the job",
        "prompt": "This segment must not affect the captured take.",
    })
    inactive_change["segments"].append(extra_segment)
    current, revision = database.validate_and_put_timeline_authority(
        UnifiedTimelineDraftV5.model_validate(inactive_change),
        expected_revision=database.get_timeline_authority()[1],
    )
    unchanged = await client.get(f"/api/jobs/{job_id}")
    assert unchanged.status_code == 200, unchanged.text
    assert unchanged.json()["current_project"] is False
    assert unchanged.json()["segment_results"][0]["current_snapshot"] is True

    active_change = current.model_dump(mode="json")
    active_change["features"]["project"]["lora"]["params"]["by_family"][
        "fl2va"
    ]["strength"] = 0.5
    database.validate_and_put_timeline_authority(
        UnifiedTimelineDraftV5.model_validate(active_change),
        expected_revision=revision,
    )
    stale = await client.get(f"/api/jobs/{job_id}")
    assert stale.status_code == 200, stale.text
    assert stale.json()["segment_results"][0]["current_snapshot"] is False


async def test_current_snapshot_fails_when_captured_segment_is_removed(client) -> None:
    database = client.director_app.state.database
    draft = _active_mapped_lora_draft()
    settings = _settings_with_overrides(
        database.get_settings(),
        [_user_mapping_record("dedicated")],
    )
    _, settings_token = database.get_settings_authority()
    database.put_settings_v3_authority(
        settings,
        expected_authority_token=settings_token,
        schema_version=3,
    )
    _, revision = database.get_timeline_authority()
    database.validate_and_put_timeline_authority(
        draft,
        expected_revision=revision,
    )
    job_id = _seed_readable_job(database, draft, settings)

    removed = draft.model_dump(mode="json")
    replacement = dict(removed["segments"][0])
    replacement.update({
        "id": "stage8-replacement-segment",
        "title": "Replacement",
        "prompt": "The captured segment no longer exists.",
    })
    removed["segments"] = [replacement]
    database.validate_and_put_timeline_authority(
        UnifiedTimelineDraftV5.model_validate(removed),
        expected_revision=database.get_timeline_authority()[1],
    )
    response = await client.get(f"/api/jobs/{job_id}")
    assert response.status_code == 200, response.text
    assert response.json()["segment_results"][0]["current_snapshot"] is False
