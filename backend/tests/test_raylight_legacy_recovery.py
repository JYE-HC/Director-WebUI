from __future__ import annotations

import json

import pytest

from directordeck.schemas import utc_now

from .conftest import runtime_authority_headers


@pytest.mark.parametrize("legacy_shape", ["direct", "envelope"])
async def test_legacy_raylight_ledger_requires_and_accepts_restart_certificate(
    client,
    fake_comfy,
    legacy_shape: str,
) -> None:
    database = client.director_app.state.database
    descriptor = {
        "version": 1,
        "family": "fl2va",
        "compatibility_key": "legacy-runtime-key",
        "runtime_namespace": "legacy-runtime-key-e7",
    }
    raw_state = (
        descriptor
        if legacy_shape == "direct"
        else {
            "version": 1,
            "epoch": 7,
            "current": descriptor,
            "tail_prompt_id": "unverifiable-legacy-tail",
            "tainted": False,
        }
    )
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO raylight_runtime_state(singleton, descriptor, updated_at) "
            "VALUES(1, ?, ?)",
            (json.dumps(raw_state), utc_now()),
        )

    authority = await runtime_authority_headers(client)
    blocked = await client.get("/api/raylight/runtime", headers=authority)
    assert blocked.status_code == 200, blocked.text
    status = blocked.json()
    assert status["active"] is False
    assert status["recovery_required"] is True
    assert status["epoch"] == 7
    assert status["runtime_gpu_indexes"] == []
    assert status["invalid_gpu_indexes"] == []
    assert status["tainted"] is True
    assert isinstance(status["recovery_token"], str)

    confirmed = await client.post(
        "/api/raylight/runtime/recovery/confirm-comfy-restart",
        json={
            "confirmation": "comfyui_process_restarted",
            "expected_epoch": 7,
            "expected_recovery_token": status["recovery_token"],
        },
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json() == {
        "active": False,
        "recovery_required": False,
        "epoch": 7,
        "runtime_gpu_indexes": [],
        "available_gpu_indexes": [0, 1],
        "invalid_gpu_indexes": [],
        "tainted": False,
        "recovery_token": None,
    }
    assert database.get_raylight_runtime_state() == {
        "version": 2,
        "epoch": 7,
        "current": None,
        "tail_prompt_id": None,
        "tail_action": None,
        "tainted": False,
    }
    backups = list(
        database.path.parent.glob(
            f"{database.path.stem}.before-raylight-recovery-e7-*.sqlite3"
        )
    )
    assert len(backups) == 1
    assert fake_comfy.history_requests == []
    assert fake_comfy.queue_requests == 1
