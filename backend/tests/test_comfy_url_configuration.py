"""Embedded-mode contract: the ComfyUI callback is injected, never a setting.

The plugin derives a directly reachable host-instance URL for ``create_app``. The
persisted settings document has no URL field, the connection test probes the
injected address, and there is no standalone/unconfigured state.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

import directordeck.app as app_module
import directordeck.comfy as comfy_module
from directordeck.app import create_app
from directordeck.comfy import ComfyClient, ComfyError
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


def test_embedded_tls_certificate_is_shared_by_http_and_websocket_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate = tmp_path / "comfy-cert.pem"
    certificate.write_text("test certificate", encoding="utf-8")

    class FakeTLSContext:
        def __init__(self) -> None:
            self.loaded: list[str] = []
            self.verify_flags = 0

        def load_verify_locations(self, *, cafile: str) -> None:
            self.loaded.append(cafile)

    context = FakeTLSContext()
    monkeypatch.setattr(
        app_module.ssl,
        "create_default_context",
        lambda: context,
    )

    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url="https://127.0.0.1:8188",
        comfy_tls_certfile=certificate,
    )
    comfy_client = app.state.comfy_factory(app.state.comfy_url)

    assert isinstance(comfy_client, ComfyClient)
    assert comfy_client.verify is context
    assert app.state.comfy_tls_context is context
    assert app.state.progress_manager._ssl_context is context
    assert context.loaded == [str(certificate)]
    assert context.verify_flags & app_module.ssl.VERIFY_X509_PARTIAL_CHAIN


def test_embedded_tls_certificate_requires_an_https_callback(tmp_path: Path) -> None:
    certificate = tmp_path / "comfy-cert.pem"
    certificate.write_text("test certificate", encoding="utf-8")

    with pytest.raises(ValueError, match="requires an https"):
        create_app(
            database_path=tmp_path / "directordeck.sqlite3",
            comfy_url="http://127.0.0.1:8188",
            comfy_tls_certfile=certificate,
        )


async def test_embedded_comfy_http_clients_ignore_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []

    class FakeResponse:
        status_code = 200
        is_error = False
        is_redirect = False

        async def aclose(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def build_request(self, *_args: object, **_kwargs: object) -> object:
            return object()

        async def send(self, *_args: object, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(comfy_module.httpx, "AsyncClient", FakeAsyncClient)
    client = ComfyClient("http://192.0.2.25:8188")

    async with client._http():
        pass
    stream = await client.view_stream({"filename": "preview.mp4"})
    await stream.aclose()

    assert len(created) == 2
    assert [kwargs["trust_env"] for kwargs in created] == [False, False]


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
