from __future__ import annotations

import pytest

import directordeck.app as director_app_module

from .conftest import adapt_legacy_workflow_requests
from .test_per_segment_execution import (
    _segment,
    _timeline,
    _wait_for_submission_jobs,
)


@pytest.fixture(autouse=True)
def _stage6_v5_request_adapter(client, monkeypatch) -> None:
    adapt_legacy_workflow_requests(client, monkeypatch)


@pytest.mark.parametrize("cancel_confirmed", (True, False))
async def test_http_exception_after_first_segment_submission_preserves_exact_cleanup_evidence(
    client,
    fake_comfy,
    monkeypatch,
    cancel_confirmed: bool,
) -> None:
    """A later local HTTP error cannot reclassify an accepted prompt as preflight.

    The first child has crossed ComfyUI's submission boundary before the
    second child's pre-claim hook fails.  That prompt must therefore go
    through the same exact directed-cancel protocol as a transport failure.
    A negative cancel acknowledgement remains owned and recoverable instead
    of being converted into a terminal local failure.
    """

    claim_count = 0

    async def fail_second_claim(_job_id: str, _child_id: str) -> None:
        nonlocal claim_count
        claim_count += 1
        if claim_count == 2:
            raise director_app_module.HTTPException(
                status_code=409,
                detail="forced HTTP failure after first segment submission",
            )

    client.director_app.state.before_submission_claim = fail_second_claim
    cancel_attempts: list[str] = []
    original_cancel = fake_comfy.cancel

    async def observe_cancel(prompt_id: str) -> bool:
        cancel_attempts.append(prompt_id)
        if not cancel_confirmed:
            return False
        return await original_cancel(prompt_id)

    monkeypatch.setattr(fake_comfy, "cancel", observe_cancel)

    response = await client.post(
        "/api/timeline/jobs",
        json={
            "config": _timeline(
                _segment("http-cleanup-first"),
                _segment("http-cleanup-second"),
            )
        },
    )
    assert response.status_code == 200, response.text
    await _wait_for_submission_jobs(client)

    database = client.director_app.state.database
    parent = database.get_job(response.json()["id"])
    children = database.list_job_children(response.json()["id"])
    assert parent is not None
    assert claim_count == 2
    assert len(fake_comfy.prompts) == 1
    assert len(children) == 2

    first, second = children
    exact_prompt_id = str(fake_comfy.prompts[0]["prompt_id"])
    assert first["segment_ids"] == ["http-cleanup-first"]
    assert second["segment_ids"] == ["http-cleanup-second"]
    assert first["prompt_id"] == exact_prompt_id
    assert second["prompt_id"] is None
    assert cancel_attempts == [exact_prompt_id]

    # Cleanup is a lifecycle transition, not evidence deletion.  The exact
    # prompt and ownership row remain the durable authority in either result.
    evidence = database.get_job_child_execution_evidence(first["id"])
    ownership = database.get_prompt_ownership(first["id"])
    assert evidence is not None
    assert (
        evidence["exact_prompt_snapshot"].model_dump(mode="json")["exact_prompt"]
        == fake_comfy.prompts[0]["prompt"]
    )
    assert ownership is not None
    assert ownership.effective_prompt_id == exact_prompt_id

    assert parent["stage"] != "preflight_failed"
    assert first["stage"] != "preflight_failed"
    assert second["stage"] != "preflight_failed"
    if cancel_confirmed:
        assert fake_comfy.cancelled == [exact_prompt_id]
        assert ownership.state == "cleanup_confirmed"
        assert ownership.cleanup_certificate is not None
        assert ownership.cleanup_certificate.prompt_id == exact_prompt_id
        assert parent["status"] == "failed"
        assert parent["stage"] == "submission_failed"
        assert first["status"] == "cancelled"
        assert first["stage"] == "cancelled_after_submission_failure"
        assert second["status"] == "failed"
        assert second["stage"] == "submission_failed"
    else:
        assert fake_comfy.cancelled == []
        assert ownership.state == "unconfirmed"
        assert ownership.cleanup_certificate is None
        assert parent["status"] == "cancelling"
        assert parent["stage"] == "submission_cancel_pending"
        assert first["status"] == "cancelling"
        assert first["stage"] == "submission_cancel_unconfirmed"
        assert first["completed_at"] is None
        assert second["status"] == "cancelled"
        assert second["stage"] == "not_submitted"
