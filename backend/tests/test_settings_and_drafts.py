from __future__ import annotations

import json
import sqlite3

import pytest

from directordeck.app import _raylight_recovery_history_state
from directordeck.database import Database
from directordeck.native_templates import (
    build_raylight_shutdown_unit,
    compile_native_timeline,
    raylight_runtime_descriptor,
)
from directordeck.schemas import (
    RuntimeSettings,
    default_settings,
    default_timeline_draft,
    validate_mode_draft,
)

from .conftest import (
    legacy_settings_document,
    runnable_draft,
    runtime_authority_headers,
    save_database_legacy_settings,
    save_legacy_settings_document,
    save_timeline_document,
)


async def test_runtime_and_creative_authorities_round_trip_separately(client) -> None:
    runtime = (await client.get("/api/settings/authority")).json()
    runtime["settings"]["client_id"] = "director-test"
    runtime["settings"]["placement"]["fl2va"]["device"] = "gpu:1"
    saved_runtime = await client.put(
        "/api/settings/authority",
        json={
            "document": runtime["settings"],
            "expected_authority_token": runtime["authority_token"],
            "schema_version": 3,
        },
    )
    assert saved_runtime.status_code == 200, saved_runtime.text

    timeline = (await client.get("/api/timeline/authority")).json()
    timeline["document"]["model_stack"]["fl2va"]["filename"] = (
        "generic_h3_diffusion.safetensors"
    )
    lora = timeline["document"]["features"]["project"]["lora"]
    lora["enabled"] = True
    lora["params"]["by_family"]["fl2va"].update(
        enabled=True,
        filename="style.safetensors",
        strength=0.65,
    )
    saved_timeline = await save_timeline_document(
        client,
        timeline["document"],
    )
    assert saved_timeline.status_code == 200, saved_timeline.text

    current_runtime = (await client.get("/api/settings")).json()
    current_timeline = (await client.get("/api/timeline")).json()
    assert current_runtime["client_id"] == "director-test"
    assert current_runtime["placement"]["fl2va"]["device"] == "gpu:1"
    assert current_timeline["model_stack"]["fl2va"]["filename"] == (
        "generic_h3_diffusion.safetensors"
    )
    assert current_timeline["features"]["project"]["lora"] == lora


@pytest.mark.parametrize(
    ("endpoint", "upstream_method", "payload"),
    (
        ("/api/capabilities", "capabilities", {"connection": "online"}),
        (
            "/api/models",
            "models",
            {"fl2va": [], "ref2va": [], "clip": [], "video_vae": [], "audio_vae": [], "loras": []},
        ),
        ("/api/gpus", "system_stats", {"devices": []}),
        ("/api/raylight/runtime", "system_stats", {"devices": []}),
    ),
)
async def test_runtime_resource_authority_token_rejects_aba_settings_switch(
    client, fake_comfy, monkeypatch, endpoint: str, upstream_method: str, payload: object
) -> None:
    monkeypatch.setattr(
        "directordeck.database.utc_now",
        lambda: "2026-08-15T20:00:00+00:00",
    )
    initial = (await client.get("/api/settings/authority")).json()
    assert "comfy_url" not in initial["settings"]
    assert len(initial["authority_token"]) == 64
    database = client.director_app.state.database
    original, original_token = database.get_settings_authority()

    async def switch_away_and_back() -> object:
        changed = original.model_copy(
            update={"client_id": "temporary-switch"}
        )
        _changed, changed_token = database.put_settings_v3_authority(
            changed,
            expected_authority_token=original_token,
            schema_version=3,
        )
        database.put_settings_v3_authority(
            original,
            expected_authority_token=changed_token,
            schema_version=3,
        )
        return payload

    monkeypatch.setattr(fake_comfy, upstream_method, switch_away_and_back)
    response = await client.get(
        endpoint,
        headers={
            "X-Director-Runtime-Authority": initial["authority_token"],
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "runtime_authority_changed"
    refreshed = (await client.get("/api/settings/authority")).json()
    assert refreshed["settings"] == initial["settings"]
    assert refreshed["authority_token"] != initial["authority_token"]


@pytest.mark.parametrize(
    "endpoint",
    ("/api/capabilities", "/api/models", "/api/gpus", "/api/raylight/runtime"),
)
async def test_runtime_resource_requires_authority_token(client, endpoint: str) -> None:
    missing = await client.get(endpoint)
    assert missing.status_code == 428, missing.text
    assert missing.json()["detail"]["code"] == "runtime_authority_required"

    malformed = await client.get(
        endpoint,
        headers={"X-Director-Runtime-Authority": "not-a-token"},
    )
    assert malformed.status_code == 428, malformed.text
    assert malformed.json()["detail"]["code"] == "runtime_authority_required"


async def test_v1_settings_write_is_retired_and_v3_rejects_creative_fields(
    client,
) -> None:
    retired = await client.put(
        "/api/settings",
        json=default_settings().model_dump(mode="json"),
    )
    assert retired.status_code == 409, retired.text
    assert retired.json()["detail"]["code"] == "runtime_settings_schema_migrated"

    authority = (await client.get("/api/settings/authority")).json()
    authority["settings"]["models"] = default_settings().model_dump(
        mode="json"
    )["models"]
    rejected = await client.put(
        "/api/settings/authority",
        json={
            "document": authority["settings"],
            "expected_authority_token": authority["authority_token"],
            "schema_version": 3,
        },
    )
    assert rejected.status_code == 422, rejected.text


async def test_settings_persist_only_a_lora_loader_mapping(
    client,
) -> None:
    settings = await legacy_settings_document(client)
    binding = settings["models"]["fl2va"]
    binding["lora_name"] = "style.safetensors"
    binding["lora_loader"] = "dedicated"  # obsolete hidden value
    binding["standard_lora_loader_override"] = {
        "loader": "model_only",
        "lora_name": "style.safetensors",
        "model_filename": binding["filename"],
    }

    response = await save_legacy_settings_document(client, settings)

    assert response.status_code == 200, response.text
    saved = response.json()["models"]["fl2va"]
    assert saved["standard_lora_loader_override"] == {
        "loader": "model_only",
        "lora_name": "style.safetensors",
        "model_filename": binding["filename"],
    }

    timeline = (await client.get("/api/timeline")).json()
    saved_runtime = (await client.get("/api/settings")).json()
    assert timeline["features"]["project"]["lora"]["enabled"] is True
    assert saved_runtime["lora_loader_overrides"] == [
        {
            "lora_filename": "style.safetensors",
            "adapter_id": "model_only",
            "options": {},
        }
    ]
    assert "legacy_lora_resolution_compat" not in saved_runtime

    stale = response.json()
    stale["models"]["fl2va"]["lora_name"] = "renamed_generic.safetensors"
    with pytest.raises(ValueError, match="must match"):
        RuntimeSettings.model_validate(stale)


async def test_settings_rejects_unknown_fields_and_cpu_vae(client) -> None:
    authority = (await client.get("/api/settings/authority")).json()
    settings = authority["settings"]
    settings["raylight"] = {"gpus": 4}
    unknown = await client.put(
        "/api/settings/authority",
        json={
            "document": settings,
            "expected_authority_token": authority["authority_token"],
            "schema_version": 3,
        },
    )
    assert unknown.status_code == 422

    settings.pop("raylight")
    settings["placement"]["video_vae_device"] = "cpu"
    invalid_vae = await client.put(
        "/api/settings/authority",
        json={
            "document": settings,
            "expected_authority_token": authority["authority_token"],
            "schema_version": 3,
        },
    )
    assert invalid_vae.status_code == 422


async def test_settings_rejects_invalid_lora_loader_and_strength(client) -> None:
    authority = (await client.get("/api/timeline/authority")).json()
    document = authority["document"]
    family = document["features"]["project"]["lora"]["params"][
        "by_family"
    ]["fl2va"]
    family["lora_loader"] = "guess"
    invalid_loader = await client.put(
        "/api/timeline/authority",
        json={
            "document": document,
            "expected_revision": authority["revision"],
        },
    )
    assert invalid_loader.status_code == 422

    family.pop("lora_loader")
    family["strength"] = 10.01
    invalid_strength = await client.put(
        "/api/timeline/authority",
        json={
            "document": document,
            "expected_revision": authority["revision"],
        },
    )
    assert invalid_strength.status_code == 422


def test_initialize_migrates_legacy_mode_seed_above_json_safe_range(tmp_path) -> None:
    path = tmp_path / "legacy-mode-seed.sqlite3"
    database = Database(path)
    database.initialize()
    with sqlite3.connect(path) as connection:
        document = json.loads(
            connection.execute(
                "SELECT document FROM mode_drafts WHERE mode = 't2v'"
            ).fetchone()[0]
        )
        document["sampling"]["seed"] = 2**64 - 1
        connection.execute(
            "UPDATE mode_drafts SET document = ? WHERE mode = 't2v'",
            (json.dumps(document),),
        )

    database.initialize()

    migrated = database.get_draft("t2v").sampling
    assert 0 <= migrated.seed <= 2**53 - 1
    assert migrated.random_seed is True


def test_initialize_migrates_legacy_timeline_seed_above_json_safe_range(tmp_path) -> None:
    path = tmp_path / "legacy-timeline-seed.sqlite3"
    database = Database(path)
    database.initialize()
    with sqlite3.connect(path) as connection:
        document = default_timeline_draft().model_dump(mode="json")
        document["version"] = 1
        document["segments"][0]["mode"] = "t2v"
        document["segments"][0].pop("continuity", None)
        document["segments"][0].pop("first_image", None)
        document["segments"][0].pop("last_image", None)
        document["ref_image_size"] = document["segments"][0].pop("ref_image_size")
        document["audio_mode"] = document["segments"][0].pop("audio_mode")
        legacy_sampling = dict(document["sampling"]["fl2va"])
        legacy_sampling.update(seed=2**64 - 1, cfg=7.5)
        document["sampling"] = legacy_sampling
        document["prompt"] = "legacy shared direction"
        document["segments"][0]["prompt"] = ""
        connection.execute(
            "UPDATE unified_timeline SET document = ? WHERE singleton = 1",
            (json.dumps(document),),
        )
        # A legacy timeline and v2 runtime settings cannot be published as one
        # authority state.  Reconstruct the coherent pre-v5 pair so startup
        # first normalizes the v1 timeline and then atomically migrates both.
        connection.execute(
            "UPDATE settings SET document = ? WHERE singleton = 1",
            (default_settings().model_dump_json(),),
        )

    database.initialize()

    migrated = database.get_timeline()
    assert migrated.segments[0].prompt == "legacy shared direction"
    assert migrated.sampling.fl2va.seed == migrated.sampling.ref2va.seed
    for sampling in (migrated.sampling.fl2va, migrated.sampling.ref2va):
        assert 0 <= sampling.seed <= 2**53 - 1
        assert sampling.random_seed is True
        assert "cfg" not in sampling.model_dump(mode="json")

@pytest.mark.parametrize("mode", ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"])
async def test_each_legacy_mode_is_read_only_after_retirement(
    client, mode: str
) -> None:
    original = (await client.get(f"/api/drafts/{mode}")).json()
    draft = runnable_draft(mode)
    draft["prompt"] = f"saved-{mode}"
    response = await client.put(f"/api/drafts/{mode}", json=draft)

    assert response.status_code == 410, response.text
    assert response.json()["detail"]["code"] == "legacy_generation_api_retired"
    assert (await client.get(f"/api/drafts/{mode}")).json() == original


async def test_mode_specific_fields_are_forbidden_in_other_modes(client) -> None:
    del client
    t2v = runnable_draft("t2v")
    t2v["shots"][0]["first_image"] = {
        "name": "leak.png", "subfolder": "", "type": "input", "kind": "image"
    }
    with pytest.raises(ValueError, match="first_image"):
        validate_mode_draft("t2v", t2v)


async def test_legacy_write_returns_tombstone_before_body_validation(client) -> None:
    response = await client.put("/api/drafts/t2v", json=runnable_draft("r2v"))
    assert response.status_code == 410, response.text
    assert response.json()["detail"]["code"] == "legacy_generation_api_retired"


async def test_negative_prompt_fields_are_no_longer_part_of_the_api(client) -> None:
    del client
    draft = runnable_draft("t2v")
    draft["negative_prompt"] = "legacy"
    draft["shots"][0]["negative_prompt"] = "legacy shot"

    with pytest.raises(ValueError, match="negative_prompt"):
        validate_mode_draft("t2v", draft)


def test_source_range_allows_only_machine_epsilon_at_video_end() -> None:
    draft = runnable_draft("rv2v")
    draft["shots"][0]["source_video"]["metadata"]["duration"] = 3.02
    draft["shots"][0]["source_start_seconds"] = 0.26576
    draft["shots"][0]["source_duration_seconds"] = 2.7542400000000002

    validate_mode_draft("rv2v", draft)
    draft["shots"][0]["source_duration_seconds"] += 1e-4
    with pytest.raises(ValueError, match="metadata.duration"):
        validate_mode_draft("rv2v", draft)


def test_initialize_migrates_legacy_negative_prompts_and_new_defaults(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    database = Database(path)
    database.initialize()
    now = "2026-08-12T00:00:00+00:00"
    legacy_settings = default_settings().model_dump(mode="json")
    database.create_job({
        "id": "legacy-ray-settings-job",
        "mode": "timeline",
        "status": "succeeded",
        "progress": 1.0,
        "stage": "completed",
        "prompt_id": None,
        "outputs": [],
        "error": None,
        "config_snapshot": {},
        "settings_snapshot": legacy_settings,
        "prompt_snapshot": {},
        "created_at": now,
        "updated_at": now,
        "started_at": now,
        "completed_at": now,
    })
    with sqlite3.connect(path) as connection:
        settings = legacy_settings
        for role in ("fl2va", "ref2va"):
            settings["models"][role].pop("lora_name", None)
            settings["models"][role].pop("lora_strength", None)
            settings["models"][role].pop("lora_loader", None)
            settings["models"][role].pop("lora_low_vram", None)
        settings["models"]["fl2va"]["raylight"]["fsdp"] = True
        settings["models"]["fl2va"]["raylight"]["cpu_offload"] = True
        settings["models"]["fl2va"].update(
            backend="raylight",
            lora_loader="model_only",
            lora_low_vram=True,
        )
        connection.execute(
            "UPDATE settings SET document = ? WHERE singleton = 1",
            (json.dumps(settings),),
        )
        connection.execute(
            "UPDATE unified_timeline SET document = ? WHERE singleton = 1",
            (default_timeline_draft().model_dump_json(),),
        )
        connection.execute(
            "UPDATE jobs SET settings_snapshot = ? WHERE id = ?",
            (json.dumps(settings), "legacy-ray-settings-job"),
        )
        document = json.loads(
            connection.execute(
                "SELECT document FROM mode_drafts WHERE mode = 't2v'"
            ).fetchone()[0]
        )
        document.pop("ref_image_size", None)
        document["negative_prompt"] = "old global value"
        document["shots"][0]["negative_prompt"] = "old shot value"
        connection.execute(
            "UPDATE mode_drafts SET document = ? WHERE mode = 't2v'",
            (json.dumps(document),),
        )

    database.initialize()
    migrated = database.get_draft("t2v").model_dump(mode="json")
    migrated_settings = database.get_settings()
    migrated_timeline = database.get_timeline().model_dump(mode="json")

    assert migrated["ref_image_size"] == "match"
    assert "negative_prompt" not in migrated
    assert "negative_prompt" not in migrated["shots"][0]
    assert migrated_settings.schema_version == 3
    assert migrated_settings.memory_policy == "keep_resident"
    assert migrated_settings.raylight_residency_policy == "keep_until_switch"
    assert migrated_settings.placement.fl2va.raylight.fsdp is False
    assert migrated_settings.placement.fl2va.raylight.cpu_offload is False
    assert migrated_timeline["version"] == 5
    assert migrated_timeline["features"]["project"]["lora"]["enabled"] is False
    migrated_job = database.get_job("legacy-ray-settings-job")
    assert migrated_job is not None
    assert migrated_job["settings_snapshot"]["raylight_residency_policy"] == (
        "keep_until_switch"
    )
    assert migrated_job["settings_snapshot"]["models"]["fl2va"]["raylight"]["fsdp"] is False
    assert migrated_job["settings_snapshot"]["models"]["fl2va"]["raylight"]["cpu_offload"] is False
    # Historical job snapshots remain an audit of what the old client saved;
    # only the live settings authority is canonicalized.
    assert migrated_job["settings_snapshot"]["models"]["fl2va"]["backend"] == "raylight"
    assert migrated_job["settings_snapshot"]["models"]["fl2va"]["lora_loader"] == "model_only"
    assert migrated_job["settings_snapshot"]["models"]["fl2va"]["lora_low_vram"] is True


def _fl_raylight_settings(*, residency_policy: str) -> RuntimeSettings:
    document = default_settings().model_dump(mode="json")
    document["raylight_residency_policy"] = residency_policy
    document["multi_gpu_enabled"] = True
    document["models"]["fl2va"].update(
        {
            "backend": "raylight",
            "raylight": {
                "gpu_select": [0, 1],
                "ulysses_degree": 2,
                "ring_degree": 1,
                "cfg_degree": 1,
                "dp_degree": 1,
                "fsdp": False,
                "cpu_offload": False,
            },
        }
    )
    return RuntimeSettings.model_validate(document)


def test_initialize_preserves_explicit_release_and_records_keyed_migration(
    tmp_path,
) -> None:
    path = tmp_path / "legacy-raylight-release.sqlite3"
    database = Database(path)
    database.initialize()
    release_settings = _fl_raylight_settings(
        residency_policy="release_after_sampling"
    )
    saved_runtime = save_database_legacy_settings(database, release_settings)
    now = "2026-08-13T00:00:00+00:00"
    database.create_job(
        {
            "id": "pre-residency-migration-job",
            "mode": "timeline",
            "status": "succeeded",
            "progress": 1.0,
            "stage": "completed",
            "prompt_id": None,
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": saved_runtime.model_dump(mode="json"),
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": now,
            "completed_at": now,
        }
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM migration_notices WHERE id = ?",
            (Database._RAYLIGHT_RESIDENCY_MIGRATION_ID,),
        )

    database.initialize()

    assert database.get_settings().raylight_residency_policy == "release_after_sampling"
    historical = database.get_job("pre-residency-migration-job")
    assert historical is not None
    assert (
        historical["settings_snapshot"]["raylight_residency_policy"]
        == "release_after_sampling"
    )
    with sqlite3.connect(path) as connection:
        marker = connection.execute(
            "SELECT message FROM migration_notices WHERE id = ?",
            (Database._RAYLIGHT_RESIDENCY_MIGRATION_ID,),
        ).fetchone()
    assert marker is not None
    assert "完整配置 key" in marker[0]


def test_initialize_migrates_old_dedicated_policy_to_keyed_switch(
    tmp_path,
) -> None:
    path = tmp_path / "explicit-raylight-release.sqlite3"
    database = Database(path)
    database.initialize()
    with sqlite3.connect(path) as connection:
        document = default_settings().model_dump(mode="json")
        document["raylight_residency_policy"] = "dedicated_keep_fl2va"
        connection.execute(
            "UPDATE settings SET document = ? WHERE singleton = 1",
            (json.dumps(document),),
        )
        connection.execute(
            "UPDATE unified_timeline SET document = ? WHERE singleton = 1",
            (default_timeline_draft().model_dump_json(),),
        )
        connection.execute(
            "DELETE FROM migration_notices WHERE id = ?",
            (Database._RAYLIGHT_RESIDENCY_MIGRATION_ID,),
        )

    database.initialize()

    assert (
        database.get_settings().raylight_residency_policy
        == "keep_until_switch"
    )


def test_fresh_single_gpu_standard_settings_default_to_keyed_switch(tmp_path) -> None:
    database = Database(tmp_path / "fresh-standard.sqlite3")

    database.initialize()
    database.initialize()

    settings = database.get_settings()
    assert settings.placement.fl2va.raylight.gpu_select == [0]
    assert settings.placement.ref2va.raylight.gpu_select == [0]
    assert settings.raylight_residency_policy == "keep_until_switch"


def test_legacy_raylight_runtime_descriptor_is_discarded_not_falsely_replayed(
    tmp_path,
) -> None:
    path = tmp_path / "legacy-raylight-runtime.sqlite3"
    database = Database(path)
    database.initialize()
    legacy_descriptor = {
        "version": 1,
        "family": "fl2va",
        "compatibility_key": "legacy-family-only-key",
        "runtime_namespace": "legacy-family-only-key-e7",
    }
    legacy_envelope = {
        "version": 1,
        "epoch": 7,
        "current": legacy_descriptor,
        "tail_prompt_id": "unverified-old-tail",
        "tainted": False,
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO raylight_runtime_state(singleton, descriptor, updated_at) "
            "VALUES(1, ?, ?)",
            (json.dumps(legacy_descriptor), "2026-08-13T00:00:00+00:00"),
        )

    state = database.get_raylight_runtime_state()

    assert state == {
        "version": 2,
        "epoch": 7,
        "current": None,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": True,
        "legacy_unknown": True,
    }

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE raylight_runtime_state SET descriptor = ? WHERE singleton = 1",
            (json.dumps(legacy_envelope),),
        )

    assert database.get_raylight_runtime_state() == {
        "version": 2,
        "epoch": 7,
        "current": None,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": True,
        "legacy_unknown": True,
    }


def test_v2_raylight_loader_ids_migrate_to_directordeck_namespace(tmp_path) -> None:
    path = tmp_path / "legacy-raylight-class-ids.sqlite3"
    database = Database(path)
    database.initialize()
    raw_settings = default_settings().model_dump(mode="json")
    raw_settings["multi_gpu_enabled"] = True
    raw_settings["models"]["fl2va"]["lora_name"] = "style.safetensors"
    raw_settings["models"]["fl2va"]["raylight"].update(
        gpu_select=[0, 1],
        ulysses_degree=2,
    )
    compiled = compile_native_timeline(
        default_timeline_draft(),
        RuntimeSettings.model_validate(raw_settings),
        "legacy-raylight-descriptor",
    )
    descriptor = raylight_runtime_descriptor(compiled.workflows[0])
    assert descriptor is not None
    legacy_aliases = {
        "DirectorDeckRayInitializerAdvanced": "RayInitializerAdvanced",
        "DirectorDeckRayLoraLoader": "RayLoraLoader",
        "DirectorDeckRayUNETLoader": "RayUNETLoader",
    }
    for node in descriptor["loader_subgraph"].values():
        node["class_type"] = legacy_aliases[node["class_type"]]
    legacy_state = {
        "version": 2,
        "epoch": 4,
        "current": descriptor,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": False,
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO raylight_runtime_state(singleton, descriptor, updated_at) "
            "VALUES(1, ?, ?)",
            (json.dumps(legacy_state), "2026-08-22T00:00:00+00:00"),
        )

    migrated = database.get_raylight_runtime_state()

    assert migrated is not None
    current = migrated["current"]
    assert current is not None
    barrier = build_raylight_shutdown_unit(current, unit_id="legacy-alias-switch")
    assert {node["class_type"] for node in barrier.prompt.values()} == {
        "DirectorDeckRayInitializerAdvanced",
        "DirectorDeckRayLoraLoader",
        "DirectorDeckRayUNETLoader",
        "DirectorDeckRayKill",
    }


async def test_model_and_gpu_proxy_shapes(client) -> None:
    headers = await runtime_authority_headers(client)
    capabilities = await client.get("/api/capabilities", headers=headers)
    assert capabilities.status_code == 200
    assert capabilities.json()["connection"] == "online"
    assert capabilities.json()["missing_nodes"] == []

    tested = await client.post("/api/capabilities")
    assert tested.status_code == 200
    assert tested.json()["ok"] is True

    models = (await client.get("/api/models", headers=headers)).json()
    expected_diffusion_models = [
        "generic_h3_diffusion.safetensors",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    ]
    assert models["fl2va"] == expected_diffusion_models
    assert models["ref2va"] == expected_diffusion_models

    gpus = (await client.get("/api/gpus", headers=headers)).json()["gpus"]
    assert [gpu["index"] for gpu in gpus] == [0, 1]
    assert all(gpu["visible"] for gpu in gpus)

    raw_stats = (await client.get("/api/system_stats")).json()
    assert raw_stats["devices"][0]["name"] == "NVIDIA RTX A6000"


async def test_gpu_proxy_preserves_dense_comfy_cuda_identity_when_primary_is_first(
    client, fake_comfy
) -> None:
    fake_comfy.system_devices = [
        {"index": 0, "type": "cpu", "name": "CPU", "vram_total": 0, "vram_free": 0},
        {"index": 1, "type": "cuda", "name": "Logical GPU 1", "vram_total": 48_001, "vram_free": 39_001},
        {"index": 0, "type": "cuda", "name": "Logical GPU 0", "vram_total": 48_000, "vram_free": 40_000},
    ]

    headers = await runtime_authority_headers(client)
    gpus = (await client.get("/api/gpus", headers=headers)).json()["gpus"]
    assert [gpu["index"] for gpu in gpus] == [0, 1]
    assert [gpu["name"] for gpu in gpus] == ["Logical GPU 0", "Logical GPU 1"]
    assert [gpu["vram_free"] for gpu in gpus] == [40_000, 39_001]
    runtime = (await client.get("/api/raylight/runtime", headers=headers)).json()
    assert runtime["available_gpu_indexes"] == [0, 1]


@pytest.mark.parametrize(
    "stats",
    [
        {"devices": "not-a-list"},
        {"system": []},
        {"devices": ["not-a-device"]},
        {"devices": [{"index": 0, "name": "missing type"}]},
        {
            "devices": [
                {"index": 0, "type": "cuda"},
                {"index": 0, "type": "cuda"},
            ]
        },
        {
            "devices": [
                {"index": 0, "type": "cuda"},
                {"index": 2, "type": "cuda"},
            ]
        },
        {"devices": [{"index": "0", "type": "cuda"}]},
        {
            "devices": [
                {"index": 0, "type": "cuda", "vram_total": "unknown"}
            ]
        },
    ],
)
async def test_gpu_resources_reject_malformed_system_stats_with_structured_502(
    client, fake_comfy, monkeypatch, stats: object
) -> None:
    async def malformed_system_stats() -> object:
        return stats

    monkeypatch.setattr(fake_comfy, "system_stats", malformed_system_stats)
    headers = await runtime_authority_headers(client)

    for endpoint in ("/api/gpus", "/api/system_stats", "/api/raylight/runtime"):
        response = await client.get(
            endpoint,
            headers=headers if endpoint != "/api/system_stats" else None,
        )
        assert response.status_code == 502, response.text
        assert response.json()["detail"]["code"] == "comfy_system_stats_invalid"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (
            {"status_str": "success", "completed": True, "messages": []},
            "terminal",
        ),
        (
            {
                "status_str": "error",
                "completed": False,
                "messages": [["execution_error", {"node_id": "ray-kill"}]],
            },
            "terminal",
        ),
        (
            {"status_str": "running", "completed": False, "messages": []},
            "nonterminal",
        ),
        (
            {"status_str": "running", "completed": True, "messages": []},
            "invalid",
        ),
        (
            {"status_str": "mystery", "completed": True, "messages": []},
            "invalid",
        ),
        (
            {
                "status_str": "success",
                "completed": True,
                "messages": [["execution_error", {"node_id": "ray-kill"}]],
            },
            "invalid",
        ),
        ({"status_str": "success", "completed": "yes", "messages": []}, "invalid"),
    ],
)
def test_raylight_recovery_history_rejects_contradictory_terminal_flags(
    status: dict, expected: str
) -> None:
    assert _raylight_recovery_history_state({"status": status}) == expected


@pytest.mark.parametrize(
    "history",
    [
        ["not-an-object"],
        {"wrong-prompt-id": {}},
        {"recovery-tail": "not-an-entry"},
        {
            "recovery-tail": {
                "status": {
                    "status_str": "running",
                    "completed": True,
                    "messages": [],
                }
            }
        },
    ],
)
async def test_raylight_recovery_maps_malformed_exact_history_to_structured_502(
    client, fake_comfy, monkeypatch, history: object
) -> None:
    database = client.director_app.state.database
    state = {
        "version": 2,
        "epoch": 4,
        "current": {
            "version": 2,
            "initializer_node_id": "initializer",
            "loader_subgraph": {
                "initializer": {
                    "class_type": "DirectorDeckRayInitializerAdvanced",
                    "inputs": {"GPU": 3, "GPU_SELECT": "0,1,2"},
                }
            },
        },
        "tail_prompt_id": "recovery-tail",
        "tail_action": "shutdown",
        "tainted": True,
    }
    database.put_raylight_runtime_state(state)
    status = (
        await client.get(
            "/api/raylight/runtime",
            headers=await runtime_authority_headers(client),
        )
    ).json()

    async def malformed_history(
        prompt_id: str | None = None, *, max_items: int | None = None
    ) -> object:
        assert prompt_id == "recovery-tail"
        assert max_items is None
        return history

    monkeypatch.setattr(fake_comfy, "history", malformed_history)
    response = await client.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        json={
            "confirmation": "comfyui_process_restarted",
            "expected_epoch": 4,
            "expected_recovery_token": status["recovery_token"],
        },
    )

    assert response.status_code == 502, response.text
    assert response.json()["detail"]["code"] == "comfy_history_invalid"
    assert database.get_raylight_runtime_state() == state
