from __future__ import annotations

import asyncio
import shutil
import socket
import ssl
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import websockets

from directordeck.app import create_app


pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="the minimal real-socket TLS transport gate runs on Linux CI",
)


@pytest.fixture
def tls_leaf(tmp_path: Path) -> tuple[Path, Path]:
    """Create a private-CA leaf that is trusted directly via partial-chain."""

    openssl = shutil.which("openssl")
    assert openssl is not None, "the Linux TLS integration gate requires openssl"

    ca_key = tmp_path / "ca-key.pem"
    ca_cert = tmp_path / "ca-cert.pem"
    leaf_key = tmp_path / "leaf-key.pem"
    leaf_request = tmp_path / "leaf.csr"
    leaf_cert = tmp_path / "leaf-cert.pem"

    commands = (
        (
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=DirectorDeck Test CA",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
            "-keyout",
            str(ca_key),
            "-out",
            str(ca_cert),
        ),
        (
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-nodes",
            "-subj",
            "/CN=DirectorDeck Test Leaf",
            "-addext",
            "subjectAltName=IP:127.0.0.1,IP:::1",
            "-addext",
            "basicConstraints=critical,CA:FALSE",
            "-addext",
            "keyUsage=critical,digitalSignature,keyEncipherment",
            "-addext",
            "extendedKeyUsage=serverAuth",
            "-keyout",
            str(leaf_key),
            "-out",
            str(leaf_request),
        ),
        (
            "x509",
            "-req",
            "-in",
            str(leaf_request),
            "-CA",
            str(ca_cert),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-days",
            "1",
            "-sha256",
            "-copy_extensions",
            "copy",
            "-out",
            str(leaf_cert),
        ),
    )
    for command in commands:
        subprocess.run(
            (openssl, *command),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )

    return leaf_cert, leaf_key


def _server_tls_context(certificate: Path, private_key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=certificate, keyfile=private_key)
    return context


def _poison_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    unreachable_proxy = "http://127.0.0.1:1"
    for variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "WSS_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "wss_proxy",
    ):
        monkeypatch.setenv(variable, unreachable_proxy)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")


def _json_http_handler(requests: list[bytes]):
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=2
            )
            requests.append(request)
            body = b'{"transport":"tls"}'
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    return handle


async def test_httpx_tls_callback_uses_partial_chain_and_ignores_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tls_leaf: tuple[Path, Path],
) -> None:
    certificate, private_key = tls_leaf
    _poison_proxy_environment(monkeypatch)
    requests: list[bytes] = []
    server = await asyncio.start_server(
        _json_http_handler(requests),
        host="127.0.0.1",
        port=0,
        family=socket.AF_INET,
        ssl=_server_tls_context(certificate, private_key),
    )
    port = server.sockets[0].getsockname()[1]
    callback_url = f"https://127.0.0.1:{port}"
    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url=callback_url,
        comfy_tls_certfile=certificate,
    )

    async with server:
        client = app.state.comfy_factory(callback_url)
        result = await asyncio.wait_for(client.system_stats(), timeout=3)

    assert result == {"transport": "tls"}
    assert requests and requests[0].startswith(b"GET /system_stats HTTP/1.1\r\n")
    assert app.state.comfy_tls_context.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN


async def test_wss_ipv6_callback_uses_shared_tls_context_and_ignores_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tls_leaf: tuple[Path, Path],
) -> None:
    if not socket.has_ipv6:
        pytest.skip("IPv6 is unavailable on this runner")

    certificate, private_key = tls_leaf
    _poison_proxy_environment(monkeypatch)
    request_paths: list[str] = []
    feature_messages: list[str | bytes] = []
    feature_received = asyncio.Event()

    async def handle(connection: Any) -> None:
        request_paths.append(connection.request.path)
        feature_messages.append(await connection.recv())
        feature_received.set()
        await connection.wait_closed()

    try:
        server = await websockets.serve(
            handle,
            host="::1",
            port=0,
            ssl=_server_tls_context(certificate, private_key),
        )
    except OSError as exc:
        pytest.skip(f"IPv6 loopback bind is unavailable: {exc}")

    port = server.sockets[0].getsockname()[1]
    callback_url = f"https://[::1]:{port}"
    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url=callback_url,
        comfy_tls_certfile=certificate,
    )
    manager = app.state.progress_manager
    try:
        assert manager._ssl_context is app.state.comfy_tls_context
        assert await manager.ensure_ready(
            callback_url,
            "d5-client",
            timeout_seconds=2,
        )
        await asyncio.wait_for(feature_received.wait(), timeout=2)
    finally:
        await manager.close()
        server.close()
        await server.wait_closed()

    assert request_paths == ["/ws?clientId=d5-client"]
    assert len(feature_messages) == 1
    assert isinstance(feature_messages[0], str)
    assert '"supports_preview_metadata": true' in feature_messages[0]


async def test_tls_callback_rejects_san_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tls_leaf: tuple[Path, Path],
) -> None:
    certificate, private_key = tls_leaf
    _poison_proxy_environment(monkeypatch)
    server = await asyncio.start_server(
        _json_http_handler([]),
        host="127.0.0.1",
        port=0,
        family=socket.AF_INET,
        ssl=_server_tls_context(certificate, private_key),
    )
    port = server.sockets[0].getsockname()[1]
    callback_url = f"https://localhost:{port}"
    app = create_app(
        database_path=tmp_path / "directordeck.sqlite3",
        comfy_url=callback_url,
        comfy_tls_certfile=certificate,
    )

    async with server:
        client = app.state.comfy_factory(callback_url)
        with pytest.raises(httpx.TransportError) as caught:
            await asyncio.wait_for(client.system_stats(), timeout=3)

    message = str(caught.value).lower()
    assert "certificate verify failed" in message
    assert "hostname mismatch" in message or "not valid for 'localhost'" in message
