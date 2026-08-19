"""Embedded-mode contract: the ComfyUI address is injected, never a setting.

The plugin passes the host instance's loopback URL to ``create_app``. The
persisted settings document has no URL field, the connection test probes the
injected address, and there is no standalone/unconfigured state.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from directordeck.app import create_app
from directordeck.comfy import ComfyError
from directordeck.schemas import RuntimeSettings, default_settings

from .conftest import TEST_COMFY_URL, runtime_authority_headers


def test_runtime_settings_have_no_comfy_url_field() -> None:
    settings = default_settings()

    assert "comfy_url" not in settings.model_dump(mode="json")
    # Pre-plugin documents carrying a URL are rejected outright; the era's
    # local databases were deleted with the transition.
    with pytest.raises(ValidationError):
        RuntimeSettings.model_validate(
            {
                **settings.model_dump(mode="json"),
                "comfy_url": "http://127.0.0.1:8188",
            }
        )


async def test_injected_comfy_url_reaches_the_client_factory(
    tmp_path: Path, fake_comfy
) -> None:
    seen_urls: list[str] = []

    def recording_factory(comfy_url: str):
        seen_urls.append(comfy_url)
        return fake_comfy

    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url=TEST_COMFY_URL,
        comfy_factory=recording_factory,
    )
    app.state.database.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as client:
        headers = await runtime_authority_headers(client)
        capabilities = await client.get("/api/capabilities", headers=headers)
        assert capabilities.status_code == 200
        assert capabilities.json()["connection"] == "online"
        settings = await client.get("/api/settings")

    assert seen_urls == [TEST_COMFY_URL]
    assert settings.status_code == 200
    assert "comfy_url" not in settings.json()


async def test_connection_test_probes_the_injected_address(
    tmp_path: Path, fake_comfy
) -> None:
    seen_urls: list[str] = []

    def recording_factory(comfy_url: str):
        seen_urls.append(comfy_url)
        return fake_comfy

    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url=TEST_COMFY_URL,
        comfy_factory=recording_factory,
    )
    app.state.database.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as client:
        # The request body no longer carries a URL; the probe always targets
        # the embedded host instance.
        tested = await client.post("/api/capabilities")

    assert tested.status_code == 200
    assert tested.json()["ok"] is True
    assert seen_urls == [TEST_COMFY_URL]


async def test_connection_test_reports_transport_failure(
    tmp_path: Path, fake_comfy
) -> None:
    async def unreachable_capabilities():
        raise ComfyError("连接被拒绝")

    fake_comfy.capabilities = unreachable_capabilities
    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url=TEST_COMFY_URL,
        comfy_factory=lambda _comfy_url: fake_comfy,
    )
    app.state.database.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as client:
        tested = await client.post("/api/capabilities")

    assert tested.status_code == 200
    assert tested.json() == {"ok": False, "message": "连接被拒绝"}


async def test_settings_put_response_matches_authoritative_get(
    tmp_path: Path,
) -> None:
    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url=TEST_COMFY_URL,
    )
    app.state.database.initialize()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://testserver",
    ) as client:
        put = await client.put(
            "/api/settings",
            json=default_settings().model_dump(mode="json"),
        )
        assert put.status_code == 200, put.text
        get = await client.get("/api/settings")
        assert get.status_code == 200

    # The browser compares its PUT draft with the authoritative GET
    # byte-for-byte; any silent server-side rewrite becomes an infinite
    # latest-wins retry loop.
    assert put.content == get.content
