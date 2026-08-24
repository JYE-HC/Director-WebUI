from __future__ import annotations

import asyncio

import pytest

import directordeck.app as director_app_module
from directordeck.app import _recover_interrupted_submission
from directordeck.workflow.interpreters.builtin import V4BuiltinInterpreter
from directordeck.workflow.registry import ValidatedFeatureTemplate

from .conftest import adapt_legacy_workflow_requests
from .test_per_segment_execution import _background_request, _segment, _timeline


@pytest.fixture(autouse=True)
def _stage6_v5_request_adapter(client, monkeypatch) -> None:
    adapt_legacy_workflow_requests(client, monkeypatch)


class _AfterIntentCrash(BaseException):
    pass


async def _wait_for_submission_owner(client) -> None:
    async def finished() -> None:
        while client.director_app.state.submission_tasks:
            await asyncio.sleep(0)

    await asyncio.wait_for(finished(), timeout=1)


async def test_after_intent_crash_has_no_post_and_recovery_never_recompiles(
    client,
    fake_comfy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent_persisted = asyncio.Event()

    async def crash_after_intent(*_args) -> None:
        intent_persisted.set()
        raise _AfterIntentCrash("simulated process exit before POST")

    client.director_app.state.after_submission_intent = crash_after_intent
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("after-intent-crash"))},
    )
    assert created.status_code == 200, created.text
    await asyncio.wait_for(intent_persisted.wait(), timeout=1)
    await _wait_for_submission_owner(client)

    database = client.director_app.state.database
    job_id = created.json()["id"]
    parent = database.get_job(job_id)
    children = database.list_job_children(job_id)
    assert parent is not None
    assert parent["status"] == "cancelling"
    assert parent["stage"] == "submission_interrupted"
    assert len(children) == 1
    child = children[0]
    assert child["stage"] == "submission_interrupted"
    assert fake_comfy.prompts == []

    exact = database.get_job_child_execution_evidence(child["id"])
    ownership = database.get_prompt_ownership(child["id"])
    assert exact is not None
    assert ownership is not None
    assert ownership.state == "submitting"
    assert ownership.effective_prompt_id == child["prompt_id"]

    def forbidden_compile(*_args, **_kwargs):
        raise AssertionError("recovery must not compile")

    def forbidden_interpreter(*_args, **_kwargs):
        raise AssertionError("recovery must not invoke an interpreter")

    async def forbidden_submit(*_args, **_kwargs):
        raise AssertionError("recovery must not resubmit")

    monkeypatch.setattr(
        director_app_module,
        "compile_project_execution_plan",
        forbidden_compile,
    )
    monkeypatch.setattr(
        "directordeck.native_templates.compile_native_timeline",
        forbidden_compile,
    )
    monkeypatch.setattr(
        ValidatedFeatureTemplate,
        "interpreter_for",
        forbidden_interpreter,
    )
    monkeypatch.setattr(V4BuiltinInterpreter, "emit", forbidden_interpreter)
    monkeypatch.setattr(fake_comfy, "submit", forbidden_submit)

    await _recover_interrupted_submission(
        _background_request(client.director_app),
        parent,
    )

    recovered_parent = database.get_job(job_id)
    recovered_child = database.get_job_child(child["id"])
    recovered_ownership = database.get_prompt_ownership(child["id"])
    assert recovered_parent is not None
    assert recovered_parent["stage"] == "restart_cancel_unconfirmed"
    assert recovered_child is not None
    assert recovered_child["stage"] == "restart_cancel_unconfirmed"
    assert recovered_ownership is not None
    assert recovered_ownership.state == "unconfirmed"
    assert fake_comfy.prompts == []


async def test_restart_recovery_claims_persisted_user_cancel_before_stage_transition(
    client,
    fake_comfy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = await client.post(
        "/api/timeline/jobs",
        json={"config": _timeline(_segment("cancel-intent-crash"))},
    )
    assert created.status_code == 200, created.text
    await _wait_for_submission_owner(client)

    database = client.director_app.state.database
    job_id = created.json()["id"]
    parent_before = database.get_job(job_id)
    children_before = database.list_job_children(job_id)
    assert parent_before is not None
    assert parent_before["status"] in {"queued", "running"}
    assert len(children_before) == 1
    child_before = children_before[0]
    ownership_before = database.get_prompt_ownership(child_before["id"])
    exact_before = database.get_job_child_execution_evidence(child_before["id"])
    assert ownership_before is not None
    assert exact_before is not None
    exact_prompt_id = ownership_before.effective_prompt_id
    assert exact_prompt_id == child_before["prompt_id"]
    assert any(exact_prompt_id in item for item in fake_comfy.pending)

    # Simulate process death after the durable API entrypoint has won the
    # cancel-intent CAS, but before _cancel_timeline_job advances either the
    # parent or child stage. Startup recovery must discover this shape from the
    # monotonic intent bit; otherwise no owner ever sends the directed cancel.
    marked, first_claim = database.mark_job_cancel_requested(job_id)
    assert marked is not None and first_claim
    assert marked["status"] == parent_before["status"]
    assert marked["stage"] == parent_before["stage"]

    prepared_count = database.prepare_interrupted_submissions_for_recovery()
    assert prepared_count == 1

    restart_parent = database.get_job(job_id)
    restart_child = database.get_job_child(child_before["id"])
    assert restart_parent is not None
    assert restart_parent["status"] == "cancelling"
    assert restart_parent["stage"] == "restart_cancel_pending"
    assert restart_child is not None
    assert restart_child["stage"] == "restart_cancel_pending"
    assert restart_child["prompt_id"] == exact_prompt_id
    assert database.get_job_child_execution_evidence(restart_child["id"]) == exact_before

    def forbidden_compile(*_args, **_kwargs):
        raise AssertionError("restart cancel recovery must not compile")

    def forbidden_interpreter(*_args, **_kwargs):
        raise AssertionError("restart cancel recovery must not invoke an interpreter")

    async def forbidden_submit(*_args, **_kwargs):
        raise AssertionError("restart cancel recovery must not resubmit")

    monkeypatch.setattr(
        director_app_module,
        "compile_project_execution_plan",
        forbidden_compile,
    )
    monkeypatch.setattr(
        "directordeck.native_templates.compile_native_timeline",
        forbidden_compile,
    )
    monkeypatch.setattr(
        ValidatedFeatureTemplate,
        "interpreter_for",
        forbidden_interpreter,
    )
    monkeypatch.setattr(V4BuiltinInterpreter, "emit", forbidden_interpreter)
    monkeypatch.setattr(fake_comfy, "submit", forbidden_submit)

    await _recover_interrupted_submission(
        _background_request(client.director_app),
        restart_parent,
    )

    recovered_parent = database.get_job(job_id)
    recovered_child = database.get_job_child(child_before["id"])
    recovered_ownership = database.get_prompt_ownership(child_before["id"])
    assert recovered_parent is not None
    assert recovered_parent["status"] == "cancelled"
    assert recovered_child is not None
    assert recovered_child["status"] == "cancelled"
    assert recovered_ownership is not None
    assert recovered_ownership.state == "cleanup_confirmed"
    assert fake_comfy.cancelled == [exact_prompt_id]
