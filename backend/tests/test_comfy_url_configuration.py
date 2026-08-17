from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from director.app import create_app
from director.comfy import ComfyError
from director.database import Database
from director.schemas import default_settings

from .conftest import runnable_draft, runtime_authority_headers


def test_default_settings_has_an_explicit_empty_comfy_url() -> None:
    settings = default_settings()

    assert settings.comfy_url == ""
    assert settings.model_dump(mode="json")["comfy_url"] == ""
    with pytest.raises(ValidationError):
        default_settings("ftp://comfy.test:8188")


def test_comfy_url_is_persisted_verbatim(tmp_path: Path) -> None:
    app = create_app(database_path=tmp_path / "director.sqlite3")
    app.state.database.initialize()

    for url in (
        "http://127.0.0.1:28188",
        "http://127.0.0.1:28188/",
        "https://comfy.example.com:8443/base/",
    ):
        app.state.database.put_settings(default_settings(url))
        assert app.state.database.get_settings().comfy_url == url


def test_fresh_database_ignores_env_and_restart_preserves_saved_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "director.sqlite3"
    monkeypatch.setenv("DIRECTOR_COMFYUI_URL", "http://env-must-not-win.test:8188")

    first_app = create_app(database_path=path)
    first_app.state.database.initialize()
    assert first_app.state.database.get_settings().comfy_url == ""

    first_app.state.database.put_settings(default_settings("http://saved-comfy.test:8188"))
    restarted_app = create_app(database_path=path)
    restarted_app.state.database.initialize()

    assert (
        str(restarted_app.state.database.get_settings().comfy_url).rstrip("/")
        == "http://saved-comfy.test:8188"
    )


async def test_unconfigured_runtime_does_not_construct_a_client_or_leave_an_orphan_job(
    tmp_path: Path,
) -> None:
    factory_calls: list[object] = []

    def forbidden_factory(settings):
        factory_calls.append(settings)
        raise AssertionError("an unconfigured runtime must not construct a ComfyUI client")

    app = create_app(
        database_path=tmp_path / "director.sqlite3",
        comfy_factory=forbidden_factory,
    )
    app.state.database.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as client:
        headers = await runtime_authority_headers(client)
        capabilities = await client.get("/api/capabilities", headers=headers)
        assert capabilities.status_code == 200
        assert capabilities.json()["connection"] == "offline"
        assert "尚未配置" in capabilities.json()["message"]

        for endpoint in (
            "/api/models",
            "/api/gpus",
            "/api/system_stats",
        ):
            response = await client.get(
                endpoint,
                headers=headers if endpoint != "/api/system_stats" else None,
            )
            assert response.status_code == 409, (endpoint, response.text)
            assert "尚未配置" in response.text

        # Raw ComfyUI queue/history contain executable prompt graphs and are
        # deliberately not exposed by Director's browser API.
        for endpoint in ("/api/queue", "/api/history", "/api/history/prompt-1"):
            assert (await client.get(endpoint)).status_code == 404

        upload = await client.post(
            "/api/assets",
            data={"kind": "image"},
            files={"file": ("frame.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )
        assert upload.status_code == 409

        submitted = await client.post(
            "/api/jobs",
            json={"mode": "t2v", "config": runnable_draft("t2v")},
        )
        assert submitted.status_code == 409
        job_list = (await client.get("/api/jobs")).json()
        assert job_list["jobs"] == []
        assert job_list["total"] == 0
        assert job_list["summary"]["total"] == 0

    assert factory_calls == []
    assert app.state.database.list_jobs() == []


async def test_connection_probe_uses_temporary_url_without_persisting_it(
    tmp_path: Path, fake_comfy
) -> None:
    seen_urls: list[str] = []

    def recording_factory(settings):
        seen_urls.append(str(settings.comfy_url).rstrip("/"))
        return fake_comfy

    app = create_app(
        database_path=tmp_path / "director.sqlite3",
        comfy_factory=recording_factory,
    )
    app.state.database.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as client:
        tested = await client.post(
            "/api/capabilities", json={"comfy_url": "http://probe-comfy.test:8188"}
        )
        persisted = await client.get("/api/settings")

    assert tested.status_code == 200
    assert tested.json()["ok"] is True
    assert seen_urls == ["http://probe-comfy.test:8188"]
    assert persisted.json()["comfy_url"] == ""


async def test_connection_probe_treats_missing_nodes_as_reachable_without_persisting(
    tmp_path: Path, fake_comfy
) -> None:
    async def capabilities_with_missing_nodes():
        return {
            "connection": "online",
            "missing_nodes": ["MissingNativeNode"],
            "latency_ms": 2.5,
        }

    fake_comfy.capabilities = capabilities_with_missing_nodes
    app = create_app(
        database_path=tmp_path / "director.sqlite3",
        comfy_factory=lambda _settings: fake_comfy,
    )
    app.state.database.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as client:
        tested = await client.post(
            "/api/capabilities",
            json={"comfy_url": "http://reachable-comfy.test:8188"},
        )
        persisted = await client.get("/api/settings")

    assert tested.status_code == 200
    assert tested.json() == {
        "ok": True,
        "latency_ms": 2.5,
        "message": "连接成功，但缺少节点: MissingNativeNode",
    }
    assert persisted.json()["comfy_url"] == ""


async def test_connection_probe_reports_transport_failure_without_persisting(
    tmp_path: Path, fake_comfy
) -> None:
    async def unreachable_capabilities():
        raise ComfyError("连接被拒绝")

    fake_comfy.capabilities = unreachable_capabilities
    app = create_app(
        database_path=tmp_path / "director.sqlite3",
        comfy_factory=lambda _settings: fake_comfy,
    )
    app.state.database.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as client:
        tested = await client.post(
            "/api/capabilities",
            json={"comfy_url": "http://offline-comfy.test:8188"},
        )
        persisted = await client.get("/api/settings")

    assert tested.status_code == 200
    assert tested.json() == {"ok": False, "message": "连接被拒绝"}
    assert persisted.json()["comfy_url"] == ""


def test_empty_settings_do_not_backfill_a_legacy_asset_origin(tmp_path: Path) -> None:
    path = tmp_path / "legacy-empty.sqlite3"
    document = {
        "name": "legacy.png",
        "subfolder": "director-web",
        "type": "input",
        "kind": "image",
        "id": "legacy-asset",
    }
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE settings(singleton INTEGER PRIMARY KEY, document TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE assets(id TEXT PRIMARY KEY, document TEXT NOT NULL, created_at TEXT NOT NULL);
            """
        )
        db.execute(
            "INSERT INTO settings VALUES(1, ?, 'then')",
            (default_settings().model_dump_json(),),
        )
        db.execute(
            "INSERT INTO assets VALUES('legacy-asset', ?, 'then')",
            (json.dumps(document),),
        )

    database = Database(path)
    database.initialize()
    with database.connect() as db:
        row = db.execute(
            "SELECT comfy_origin FROM assets WHERE id = 'legacy-asset'"
        ).fetchone()

    assert row is not None
    assert row["comfy_origin"] is None
