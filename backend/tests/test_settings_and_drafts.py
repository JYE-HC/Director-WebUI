from __future__ import annotations

import json
import sqlite3

import pytest

from directordeck.app import _raylight_recovery_history_state
from directordeck.database import Database
from directordeck.schemas import RuntimeSettings, default_settings, default_timeline_draft

from .conftest import runnable_draft, runtime_authority_headers


async def test_settings_round_trip(client) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["client_id"] = "director-test"
    settings["models"]["fl2va"]["device"] = "gpu:1"
    settings["models"]["fl2va"].update(
        lora_name="style.safetensors",
        lora_strength=0.65,
        lora_loader="model_only",
        lora_low_vram=False,
    )

    response = await client.put("/api/settings", json=settings)

    assert response.status_code == 200
    assert response.json()["client_id"] == "director-test"
    assert (await client.get("/api/settings")).json()["models"]["fl2va"]["device"] == "gpu:1"
    assert response.json()["models"]["fl2va"]["lora_name"] == "style.safetensors"
    assert response.json()["models"]["fl2va"]["lora_strength"] == 0.65


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
    original = database.get_settings()

    async def switch_away_and_back() -> object:
        changed = RuntimeSettings.model_validate(
            {
                **original.model_dump(mode="json"),
                "client_id": "temporary-switch",
            }
        )
        database.put_settings(changed)
        database.put_settings(original)
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


async def test_settings_canonicalize_obsolete_backend_and_lora_loader(client) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"]["fl2va"].update(
        backend="standard",
        device="gpu:7",
        lora_name=(
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_"
            "10ErosMax_beta1_pruned_compat_v001_T8.safetensors"
        ),
        lora_loader="model_only",
        lora_low_vram=True,
        raylight={
            "gpu_select": [0, 1],
            "ulysses_degree": 2,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "fsdp": False,
            "cpu_offload": False,
        },
    )

    response = await client.put("/api/settings", json=settings)

    assert response.status_code == 200, response.text
    binding = response.json()["models"]["fl2va"]
    assert binding["backend"] == "auto"
    assert binding["lora_loader"] == "auto"
    assert binding["standard_lora_loader_override"] is None
    assert binding["lora_low_vram"] is False
    assert binding["device"] == "default"
    assert binding["raylight"]["gpu_select"] == [0, 1]
    assert (await client.get("/api/settings")).json()["models"]["fl2va"] == binding


async def test_settings_preserve_only_a_scoped_visible_standard_lora_override(
    client,
) -> None:
    settings = (await client.get("/api/settings")).json()
    binding = settings["models"]["fl2va"]
    binding["lora_name"] = "style.safetensors"
    binding["lora_loader"] = "dedicated"  # obsolete hidden value
    binding["standard_lora_loader_override"] = {
        "loader": "model_only",
        "lora_name": "style.safetensors",
        "model_filename": binding["filename"],
    }

    response = await client.put("/api/settings", json=settings)

    assert response.status_code == 200, response.text
    saved = response.json()["models"]["fl2va"]
    assert saved["lora_loader"] == "auto"
    assert saved["standard_lora_loader_override"] == {
        "loader": "model_only",
        "lora_name": "style.safetensors",
        "model_filename": binding["filename"],
    }

    stale = response.json()
    stale["models"]["fl2va"]["lora_name"] = "renamed_generic.safetensors"
    assert (await client.put("/api/settings", json=stale)).status_code == 422


async def test_settings_rejects_unknown_fields_and_cpu_vae(client) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["raylight"] = {"gpus": 4}
    assert (await client.put("/api/settings", json=settings)).status_code == 422

    settings.pop("raylight")
    settings["models"]["video_vae"]["device"] = "cpu"
    assert (await client.put("/api/settings", json=settings)).status_code == 422


async def test_settings_rejects_invalid_lora_loader_and_strength(client) -> None:
    settings = (await client.get("/api/settings")).json()
    settings["models"]["fl2va"]["lora_loader"] = "guess"
    assert (await client.put("/api/settings", json=settings)).status_code == 422

    settings["models"]["fl2va"]["lora_loader"] = "model_only"
    settings["models"]["fl2va"]["lora_strength"] = 10.01
    assert (await client.put("/api/settings", json=settings)).status_code == 422


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

    database.initialize()

    migrated = database.get_timeline()
    assert migrated.segments[0].prompt == "legacy shared direction"
    assert migrated.sampling.fl2va.seed == migrated.sampling.ref2va.seed
    for sampling in (migrated.sampling.fl2va, migrated.sampling.ref2va):
        assert 0 <= sampling.seed <= 2**53 - 1
        assert sampling.random_seed is True
        assert "cfg" not in sampling.model_dump(mode="json")

@pytest.mark.parametrize("mode", ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"])
async def test_each_mode_has_an_independent_persisted_draft(client, mode: str) -> None:
    draft = runnable_draft(mode)
    draft["prompt"] = f"saved-{mode}"
    response = await client.put(f"/api/drafts/{mode}", json=draft)
    assert response.status_code == 200, response.text
    assert (await client.get(f"/api/drafts/{mode}")).json()["prompt"] == f"saved-{mode}"

    other = "i2v" if mode == "t2v" else "t2v"
    assert (await client.get(f"/api/drafts/{other}")).json()["prompt"] != f"saved-{mode}"


async def test_mode_specific_fields_are_forbidden_in_other_modes(client) -> None:
    t2v = runnable_draft("t2v")
    t2v["shots"][0]["first_image"] = {
        "name": "leak.png", "subfolder": "", "type": "input", "kind": "image"
    }
    response = await client.put("/api/drafts/t2v", json=t2v)
    assert response.status_code == 422
    assert "first_image" in response.text


async def test_path_and_body_mode_must_match(client) -> None:
    response = await client.put("/api/drafts/t2v", json=runnable_draft("r2v"))
    assert response.status_code == 422


async def test_negative_prompt_fields_are_no_longer_part_of_the_api(client) -> None:
    draft = runnable_draft("t2v")
    draft["negative_prompt"] = "legacy"
    draft["shots"][0]["negative_prompt"] = "legacy shot"

    response = await client.put("/api/drafts/t2v", json=draft)

    assert response.status_code == 422
    assert "negative_prompt" in response.text


def test_source_range_allows_only_machine_epsilon_at_video_end() -> None:
    draft = runnable_draft("rv2v")
    draft["shots"][0]["source_video"]["metadata"]["duration"] = 3.02
    draft["shots"][0]["source_start_seconds"] = 0.26576
    draft["shots"][0]["source_duration_seconds"] = 2.7542400000000002

    from directordeck.schemas import validate_mode_draft

    validate_mode_draft("rv2v", draft)
    draft["shots"][0]["source_duration_seconds"] += 1e-4
    with pytest.raises(ValueError, match="metadata.duration"):
        validate_mode_draft("rv2v", draft)


def test_initialize_migrates_legacy_negative_prompts_and_new_defaults(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    database = Database(path)
    database.initialize()
    now = "2026-08-12T00:00:00+00:00"
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
        "settings_snapshot": database.get_settings().model_dump(mode="json"),
        "prompt_snapshot": {},
        "created_at": now,
        "updated_at": now,
        "started_at": now,
        "completed_at": now,
    })
    with sqlite3.connect(path) as connection:
        settings = json.loads(
            connection.execute(
                "SELECT document FROM settings WHERE singleton = 1"
            ).fetchone()[0]
        )
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
    migrated_settings = database.get_settings().model_dump(mode="json")

    assert migrated["ref_image_size"] == "match"
    assert "negative_prompt" not in migrated
    assert "negative_prompt" not in migrated["shots"][0]
    assert migrated_settings["models"]["fl2va"]["lora_name"] is None
    assert migrated_settings["models"]["ref2va"]["lora_loader"] == "auto"
    assert migrated_settings["memory_policy"] == "keep_resident"
    assert migrated_settings["raylight_residency_policy"] == (
        "keep_until_switch"
    )
    assert migrated_settings["models"]["fl2va"]["raylight"]["fsdp"] is False
    assert migrated_settings["models"]["fl2va"]["raylight"]["cpu_offload"] is False
    assert migrated_settings["models"]["fl2va"]["backend"] == "auto"
    assert migrated_settings["models"]["fl2va"]["lora_loader"] == "auto"
    assert migrated_settings["models"]["fl2va"]["lora_low_vram"] is False
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
    database.put_settings(release_settings)
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
            "settings_snapshot": release_settings.model_dump(mode="json"),
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
        document = database.get_settings().model_dump(mode="json")
        document["raylight_residency_policy"] = "dedicated_keep_fl2va"
        connection.execute(
            "UPDATE settings SET document = ? WHERE singleton = 1",
            (json.dumps(document),),
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
    assert settings.models.fl2va.raylight.gpu_select == [0]
    assert settings.models.ref2va.raylight.gpu_select == [0]
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
                    "class_type": "RayInitializerAdvanced",
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
