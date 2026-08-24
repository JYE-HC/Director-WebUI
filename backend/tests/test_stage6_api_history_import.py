from __future__ import annotations

from datetime import datetime, timezone
import json
import uuid

import pytest

import directordeck.app as director_app_module
from directordeck.migration_api import (
    DocumentDigest,
    HistoricalCreativeInputError,
    ProjectImportCommitRequest,
    ProjectImportCoordinator,
    ProjectImportCreativeSelection,
    ProjectImportError,
    ProjectImportPreflightRequest,
    prepare_project_import,
    project_import_input_digest,
    resolve_historical_creative_input,
)
from directordeck.migrations import legacy_creative_binding_context
from directordeck.schemas import (
    RuntimeSettingsV1,
    UnifiedTimelineDraftV4,
    UnifiedTimelineDraftV5,
    default_model_stack,
    default_settings,
    default_timeline_draft,
    default_timeline_draft_v5,
)

from .conftest import runnable_draft, wait_for_submission_tasks


def _test_migrate(
    timeline: UnifiedTimelineDraftV4, settings: RuntimeSettingsV1
) -> UnifiedTimelineDraftV5:
    document = timeline.model_dump(mode="json")
    document.update(
        version=5,
        model_stack={
            family: {"filename": getattr(settings.models, family).filename}
            for family in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae")
        },
        features={
            "template_bundle_version": 4,
            "project": {
                "lora": {
                    "enabled": any(
                        getattr(settings.models, family).lora_name is not None
                        for family in ("fl2va", "ref2va")
                    ),
                    "params": {
                        "by_family": {
                            family: {
                                "enabled": getattr(
                                    settings.models, family
                                ).lora_name
                                is not None,
                                "filename": getattr(
                                    settings.models, family
                                ).lora_name,
                                "strength": getattr(
                                    settings.models, family
                                ).lora_strength,
                            }
                            for family in ("fl2va", "ref2va")
                        }
                    },
                }
            },
            "by_segment": {},
        },
    )
    return UnifiedTimelineDraftV5.model_validate(document)


def _complete_frozen_v5_document(prompt: str) -> dict:
    document = default_timeline_draft_v5(
        default_model_stack()
    ).model_dump(mode="json")
    document["segments"][0]["prompt"] = prompt
    return document


@pytest.mark.parametrize("mode", ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"))
async def test_legacy_mode_writes_are_versioned_tombstones(client, mode: str) -> None:
    draft_write = await client.put(f"/api/drafts/{mode}", json=runnable_draft(mode))

    assert draft_write.status_code == 410, draft_write.text
    assert draft_write.json()["detail"] == {
        "code": "legacy_generation_api_retired",
        "message": (
            "Legacy six-mode generation writes are retired; refresh the client "
            "and submit a v5 timeline snapshot."
        ),
        "required_schema": 5,
    }


@pytest.mark.parametrize("mode", ("t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"))
async def test_legacy_mode_job_submission_never_reads_live_settings(
    client, monkeypatch, mode: str
) -> None:
    database = client.director_app.state.database

    def forbidden_settings_read():
        raise AssertionError("retired generation must not read live settings")

    monkeypatch.setattr(database, "get_settings", forbidden_settings_read)
    response = await client.post(
        "/api/jobs",
        json={"mode": mode, "config": runnable_draft(mode)},
    )

    assert response.status_code == 410, response.text
    assert response.json()["detail"]["code"] == "legacy_generation_api_retired"


def test_v4_import_without_its_own_creative_context_never_uses_live_settings() -> None:
    body = ProjectImportPreflightRequest(
        document=default_timeline_draft().model_dump(mode="json")
    )

    digest, proposed, missing_context, missing_models = prepare_project_import(
        body,
        migrate_v4_to_v5=lambda *_args: pytest.fail(
            "migration must not run without file-supplied context"
        ),
    )

    assert digest.algorithm == "sha256-canonical-json-v1"
    assert proposed is None
    assert missing_context == ["creative_selection"]
    assert missing_models == [
        "fl2va",
        "ref2va",
        "clip",
        "video_vae",
        "audio_vae",
    ]


def test_v4_import_explicit_selection_is_ready_and_digest_binds_context() -> None:
    document = default_timeline_draft().model_dump(mode="json")
    context = legacy_creative_binding_context(default_settings())
    selection = ProjectImportCreativeSelection(
        model_stack=context.model_stack,
        lora=context.lora,
    )
    body = ProjectImportPreflightRequest(
        title="Selected import",
        document=document,
        creative_selection=selection,
    )

    digest, proposed, missing_context, missing_models = prepare_project_import(
        body,
        migrate_v4_to_v5=_test_migrate,
    )

    assert proposed is not None
    assert proposed.version == 5
    assert proposed.model_stack == selection.model_stack
    assert proposed.features.project["lora"] == selection.lora
    assert missing_context == []
    assert missing_models == []
    assert digest == project_import_input_digest(body)
    assert digest != project_import_input_digest(
        body.model_copy(update={"title": "Different title"})
    )
    assert digest != project_import_input_digest(
        body.model_copy(
            update={
                "creative_selection": selection.model_copy(
                    update={
                        "model_stack": selection.model_stack.model_copy(
                            update={
                                "fl2va": selection.model_stack.fl2va.model_copy(
                                    update={"filename": "different.safetensors"}
                                )
                            }
                        )
                    }
                )
            }
        )
    )


def test_v4_import_rejects_ambiguous_creative_authorities() -> None:
    context = legacy_creative_binding_context(default_settings())
    with pytest.raises(ValueError, match="exactly one creative"):
        ProjectImportPreflightRequest(
            document=default_timeline_draft().model_dump(mode="json"),
            legacy_runtime_settings=default_settings(),
            legacy_creative_context=context,
        )


@pytest.mark.parametrize("version", (0, 3, 6, 2**31))
def test_import_rejects_unknown_or_future_schema(version: int) -> None:
    document = default_timeline_draft().model_dump(mode="json")
    document["version"] = version

    with pytest.raises(ProjectImportError) as caught:
        prepare_project_import(
            ProjectImportPreflightRequest(document=document),
            migrate_v4_to_v5=_test_migrate,
        )

    assert caught.value.code == "project_import_schema_unsupported"


def test_import_rejects_unknown_feature_before_issuing_token() -> None:
    proposed = _test_migrate(default_timeline_draft(), default_settings())
    document = proposed.model_dump(mode="json")
    document["features"]["project"]["not-installed"] = {
        "enabled": True,
        "params": {},
    }

    with pytest.raises(ProjectImportError) as caught:
        prepare_project_import(
            ProjectImportPreflightRequest(document=document),
            migrate_v4_to_v5=_test_migrate,
        )

    assert caught.value.code == "project_import_unknown_feature"
    assert caught.value.details == {"feature_ids": ["not-installed"]}


def test_import_token_is_digest_bound_single_use_and_short_lived() -> None:
    now = [10.0]
    coordinator = ProjectImportCoordinator(
        token_ttl_seconds=30,
        monotonic=lambda: now[0],
        utcnow=lambda: datetime(2026, 8, 21, tzinfo=timezone.utc),
    )
    proposed = _test_migrate(default_timeline_draft(), default_settings())
    source = DocumentDigest(
        algorithm="sha256-canonical-json-v1", value="sha256-" + "a" * 64
    )
    preflight = coordinator.issue(
        title="Imported",
        input_digest=source,
        proposed_document=proposed,
        missing_model_bindings=[],
    )
    assert preflight.status == "ready"
    assert preflight.commit_token is not None

    with pytest.raises(ProjectImportError) as mismatch:
        coordinator.consume(
            ProjectImportCommitRequest(
                commit_token=preflight.commit_token,
                input_digest=DocumentDigest(
                    algorithm="sha256-canonical-json-v1",
                    value="sha256-" + "b" * 64,
                ),
            )
        )
    assert mismatch.value.code == "project_import_digest_mismatch"

    title, consumed = coordinator.consume(
        ProjectImportCommitRequest(
            commit_token=preflight.commit_token,
            input_digest=source,
        )
    )
    assert title == "Imported"
    assert consumed == proposed

    with pytest.raises(ProjectImportError) as replay:
        coordinator.consume(
            ProjectImportCommitRequest(
                commit_token=preflight.commit_token,
                input_digest=source,
            )
        )
    assert replay.value.code == "project_import_token_invalid"

    expiring = coordinator.issue(
        title="Expired",
        input_digest=source,
        proposed_document=proposed,
        missing_model_bindings=[],
    )
    now[0] = 41.0
    with pytest.raises(ProjectImportError) as expired:
        coordinator.consume(
            ProjectImportCommitRequest(
                commit_token=expiring.commit_token or "",
                input_digest=source,
            )
        )
    assert expired.value.code == "project_import_token_invalid"


def test_historical_v4_resolver_uses_only_the_jobs_immutable_settings() -> None:
    settings = default_settings().model_copy(update={"client_id": "historical-only"})
    job = {
        "config_snapshot": {
            "timeline": default_timeline_draft().model_dump(mode="json")
        },
        "settings_snapshot": settings.model_dump(mode="json"),
    }
    observed_client_ids: list[str] = []

    def migrate(
        timeline: UnifiedTimelineDraftV4, own_settings: RuntimeSettingsV1
    ) -> UnifiedTimelineDraftV5:
        observed_client_ids.append(own_settings.client_id)
        return _test_migrate(timeline, own_settings)

    resolved = resolve_historical_creative_input(job, migrate_v4_to_v5=migrate)

    assert resolved.version == 5
    assert observed_client_ids == ["historical-only"]


@pytest.mark.parametrize(
    "job",
    (
        {},
        {"config_snapshot": {}},
        {
            "config_snapshot": {
                "timeline": default_timeline_draft().model_dump(mode="json")
            },
            "settings_snapshot": None,
        },
    ),
)
def test_historical_resolver_fails_closed_without_exact_snapshot(job) -> None:
    with pytest.raises(HistoricalCreativeInputError):
        resolve_historical_creative_input(job, migrate_v4_to_v5=_test_migrate)


async def test_stale_runtime_settings_puts_are_rejected_and_v3_writes_require_cas(
    client,
) -> None:
    authority = (await client.get("/api/settings/authority")).json()
    assert authority["settings"]["schema_version"] == 3

    stale_v1 = await client.put(
        "/api/settings",
        json=default_settings().model_dump(mode="json"),
    )
    assert stale_v1.status_code == 409, stale_v1.text
    assert stale_v1.json()["detail"]["code"] == "runtime_settings_schema_migrated"

    stale_v1_authority = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 1,
            "document": default_settings().model_dump(mode="json"),
            "expected_authority_token": authority["authority_token"],
        },
    )
    assert stale_v1_authority.status_code == 409, stale_v1_authority.text
    assert stale_v1_authority.json()["detail"] == {
        "code": "runtime_settings_schema_migrated",
        "message": "Runtime settings migrated to schema 3; refresh before saving.",
        "current_schema": 3,
    }

    malformed_v3 = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": default_settings().model_dump(mode="json"),
            "expected_authority_token": authority["authority_token"],
        },
    )
    assert malformed_v3.status_code == 422, malformed_v3.text

    document = authority["settings"]
    document["client_id"] = "stage6-cas"
    updated = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": document,
            "expected_authority_token": authority["authority_token"],
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["settings"]["client_id"] == "stage6-cas"
    assert updated.json()["authority_token"] != authority["authority_token"]

    stale = await client.put(
        "/api/settings/authority",
        json={
            "schema_version": 3,
            "document": document,
            "expected_authority_token": authority["authority_token"],
        },
    )
    assert stale.status_code == 409, stale.text
    assert stale.json()["detail"]["code"] == "runtime_settings_authority_conflict"


@pytest.mark.parametrize(
    "legacy_fields",
    (
        ("legacy_runtime_settings",),
        ("legacy_creative_context",),
        ("creative_selection",),
        (
            "legacy_runtime_settings",
            "legacy_creative_context",
            "creative_selection",
        ),
    ),
)
async def test_v5_import_rejects_every_legacy_context_field(
    client,
    legacy_fields: tuple[str, ...],
) -> None:
    context = legacy_creative_binding_context(default_settings())
    values = {
        "legacy_runtime_settings": default_settings().model_dump(mode="json"),
        "legacy_creative_context": context.model_dump(mode="json"),
        "creative_selection": {
            "model_stack": context.model_stack.model_dump(mode="json"),
            "lora": context.lora.model_dump(mode="json"),
        },
    }
    document = _test_migrate(
        default_timeline_draft(), default_settings()
    ).model_dump(mode="json")

    response = await client.post(
        "/api/projects/import/preflight",
        json={
            "title": "Already v5",
            "document": document,
            **{field: values[field] for field in legacy_fields},
        },
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == {
        "code": "project_import_legacy_context_forbidden",
        "message": "A schema-5 project must not carry legacy creative context.",
        "details": {
            "schema_version": 5,
            "fields": list(legacy_fields),
        },
    }


async def test_v4_timeline_stale_tab_gets_receipt_aware_409(client) -> None:
    database = client.director_app.state.database
    project_id = database.LEGACY_DEFAULT_PROJECT_ID
    authority = (await client.get("/api/timeline/authority")).json()
    assert authority["document"]["version"] == 5

    response = await client.put(
        "/api/timeline/authority",
        json={
            "document": default_timeline_draft().model_dump(mode="json"),
            "expected_revision": max(0, authority["revision"] - 1),
        },
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "timeline_schema_migrated"
    assert detail["current_schema"] == 5
    # A fresh v5 database has no historical v4 document and therefore no
    # receipt.  Upgraded databases return the exact receipt id; the dedicated
    # migration tests exercise that branch and its digest chain.
    assert "migration_id" in detail


async def test_new_project_can_be_incomplete_but_compile_fails_closed(client) -> None:
    created = await client.post(
        "/api/projects",
        json={"title": "Incomplete", "initial_model_stack": None},
    )
    assert created.status_code == 200, created.text
    project_id = created.json()["id"]
    timeline = (
        await client.get(f"/api/projects/{project_id}/timeline")
    ).json()
    assert timeline["version"] == 5
    assert all(
        binding["filename"] is None
        for binding in timeline["model_stack"].values()
    )
    timeline["segments"][0]["prompt"] = "Incomplete model binding test"

    compile_response = await client.post(
        f"/api/projects/{project_id}/compile",
        json={"config": timeline},
    )
    assert compile_response.status_code == 422, compile_response.text
    detail = compile_response.json()["detail"]
    assert detail["code"] == "model_binding_required"
    assert detail["reasons"][0]["safe_details"] == {
        "bindings": ["clip", "video_vae", "audio_vae", "fl2va"]
    }


async def test_v5_job_request_is_required_and_never_falls_back_to_project(
    client, monkeypatch
) -> None:
    missing = await client.post("/api/timeline/compile", json={})
    assert missing.status_code == 422, missing.text

    document = (await client.get("/api/timeline")).json()
    document["segments"][0]["prompt"] = "A complete v5 request snapshot"
    legacy_models = default_settings().models
    for role in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae"):
        document["model_stack"][role]["filename"] = getattr(
            legacy_models, role
        ).filename

    def forbidden_project_document_read(*_args, **_kwargs):
        raise AssertionError("v5 compile must not reload the mutable project")

    monkeypatch.setattr(
        client.director_app.state.database,
        "get_project_timeline",
        forbidden_project_document_read,
    )
    response = await client.post(
        "/api/timeline/compile",
        json={"config": document},
    )
    assert response.status_code == 200, response.text


async def test_v5_submit_captures_one_runtime_snapshot_and_no_project_document(
    client, monkeypatch
) -> None:
    database = client.director_app.state.database
    document = _complete_frozen_v5_document("A submitted v5 snapshot")

    settings_reads = 0
    original_get_settings = database.get_settings

    def counted_settings_read():
        nonlocal settings_reads
        settings_reads += 1
        return original_get_settings()

    def forbidden_project_document_read(*_args, **_kwargs):
        raise AssertionError("v5 submit must not reload the mutable project")

    monkeypatch.setattr(database, "get_settings", counted_settings_read)
    monkeypatch.setattr(
        database,
        "get_project_timeline",
        forbidden_project_document_read,
    )
    response = await client.post(
        "/api/timeline/jobs",
        json={"config": document},
    )

    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    persisted = database.get_job(response.json()["id"])
    assert persisted is not None
    assert persisted["config_snapshot"]["timeline"] == document
    assert persisted["settings_snapshot"]["snapshot_schema_version"] == 1
    assert "lora_loader_overrides" not in persisted["settings_snapshot"]
    assert "runtime_projection" in persisted["settings_snapshot"]
    assert persisted["settings_snapshot"]["control_evidence"] == {
        "progress_client_id": "directordeck",
    }
    assert settings_reads == 1


async def test_v5_submit_projects_plan_invariants_as_safe_structured_errors(
    client, fake_comfy, monkeypatch
) -> None:
    database = client.director_app.state.database
    document = _complete_frozen_v5_document(
        "A safe invariant response boundary"
    )

    unsafe_internal_detail = "/" + "home/private/director/token-secret"
    original_compile = director_app_module.compile_project_execution_plan

    class InvalidManifestPlan:
        def __init__(self, plan) -> None:
            self._plan = plan

        def __getattr__(self, name):
            return getattr(self._plan, name)

        def model_dump(self, *args, **kwargs):
            document = self._plan.model_dump(*args, **kwargs)
            document["compile_report"]["manifest"] = unsafe_internal_detail
            return document

    def compile_with_invalid_manifest(*args, **kwargs):
        plan = original_compile(*args, **kwargs)
        return InvalidManifestPlan(plan)

    monkeypatch.setattr(
        director_app_module,
        "compile_project_execution_plan",
        compile_with_invalid_manifest,
    )
    jobs_before = database.list_jobs()
    response = await client.post(
        "/api/timeline/jobs",
        json={"config": document},
    )

    assert response.status_code == 200, response.text
    await wait_for_submission_tasks(client)
    persisted = database.get_job(response.json()["id"])
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert json.loads(persisted["error"]) == {
        "code": "execution_plan_invariant_failed",
        "message": "The compiled execution evidence is internally inconsistent.",
    }
    assert unsafe_internal_detail not in persisted["error"]
    assert len(database.list_jobs()) == len(jobs_before) + 1
    assert fake_comfy.prompts == []


async def test_project_import_is_two_phase_digest_bound_and_replay_safe(client) -> None:
    direct = await client.post(
        "/api/projects/import",
        json={"title": "bypass", "document": (await client.get("/api/timeline")).json()},
    )
    assert direct.status_code == 410, direct.text
    assert direct.json()["detail"]["code"] == "project_import_preflight_required"

    legacy = await client.post(
        "/api/projects/import/preflight",
        json={"title": "legacy", "document": default_timeline_draft().model_dump(mode="json")},
    )
    assert legacy.status_code == 200, legacy.text
    assert legacy.json()["status"] == "needs_input"
    assert legacy.json()["commit_token"] is None

    context = legacy_creative_binding_context(default_settings())
    selected = await client.post(
        "/api/projects/import/preflight",
        json={
            "title": "Selected legacy",
            "document": default_timeline_draft().model_dump(mode="json"),
            "creative_selection": {
                "model_stack": context.model_stack.model_dump(mode="json"),
                "lora": context.lora.model_dump(mode="json"),
            },
        },
    )
    assert selected.status_code == 200, selected.text
    selected_proposal = selected.json()
    assert selected_proposal["status"] == "ready"
    selected_commit = await client.post(
        "/api/projects/import/commit",
        json={
            "commit_token": selected_proposal["commit_token"],
            "input_digest": selected_proposal["input_digest"],
        },
    )
    assert selected_commit.status_code == 200, selected_commit.text
    selected_id = selected_commit.json()["id"]
    selected_document = (
        await client.get(f"/api/projects/{selected_id}/timeline")
    ).json()
    assert selected_document["version"] == 5
    assert selected_document["title"] == "Selected legacy"
    assert selected_commit.json()["title"] == selected_document["title"]
    assert selected_document["model_stack"] == context.model_stack.model_dump(
        mode="json"
    )

    document = (await client.get("/api/timeline")).json()
    document["title"] = "Imported v5"
    preflight = await client.post(
        "/api/projects/import/preflight",
        json={"title": "Imported v5", "document": document},
    )
    assert preflight.status_code == 200, preflight.text
    proposal = preflight.json()
    assert proposal["status"] == "ready"
    assert proposal["commit_token"]
    assert proposal["input_digest"]["value"].startswith("sha256-")

    commit_body = {
        "commit_token": proposal["commit_token"],
        "input_digest": proposal["input_digest"],
    }
    committed = await client.post(
        "/api/projects/import/commit", json=commit_body
    )
    assert committed.status_code == 200, committed.text
    imported_id = committed.json()["id"]
    imported = (
        await client.get(f"/api/projects/{imported_id}/timeline")
    ).json()
    assert imported == document

    replay = await client.post("/api/projects/import/commit", json=commit_body)
    assert replay.status_code == 409, replay.text
    assert replay.json()["detail"]["code"] == "project_import_token_invalid"


async def test_historical_v4_api_read_and_save_as_use_own_settings_only(
    client, monkeypatch
) -> None:
    database = client.director_app.state.database
    legacy_settings = default_settings().model_copy(deep=True)
    legacy_models = legacy_settings.models.model_copy(
        update={
            "fl2va": legacy_settings.models.fl2va.model_copy(
                update={"filename": "historical-exclusive.safetensors"}
            )
        }
    )
    legacy_settings = legacy_settings.model_copy(update={"models": legacy_models})
    job_id = str(uuid.uuid4())
    now = "2026-08-21T00:00:00+00:00"
    database.create_job(
        {
            "id": job_id,
            "mode": "timeline",
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": None,
            "project_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {
                "timeline": default_timeline_draft().model_dump(mode="json"),
                "segment_ids": None,
            },
            "settings_snapshot": legacy_settings.model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
        }
    )

    def forbidden_live_settings_read():
        raise AssertionError("historical resolver must not read live settings")

    monkeypatch.setattr(database, "get_settings", forbidden_live_settings_read)
    view = await client.get(f"/api/jobs/{job_id}/project")
    assert view.status_code == 200, view.text
    assert view.json()["project"]["version"] == 5
    assert (
        view.json()["project"]["model_stack"]["fl2va"]["filename"]
        == "historical-exclusive.safetensors"
    )

    saved = await client.post(
        f"/api/jobs/{job_id}/save-as-project",
        json={"title": "Recovered"},
    )
    assert saved.status_code == 200, saved.text
    saved_id = saved.json()["id"]
    assert saved_id != database.LEGACY_DEFAULT_PROJECT_ID
    restored = (
        await client.get(f"/api/projects/{saved_id}/timeline")
    ).json()
    assert restored["title"] == "Recovered"
    assert restored["model_stack"]["fl2va"]["filename"] == (
        "historical-exclusive.safetensors"
    )
    assert database.find_latest_segment_take(
        restored["segments"][0]["id"],
        "not-a-real-fingerprint",
        project_id=saved_id,
    ) is None
