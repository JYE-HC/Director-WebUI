from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import socket
import sys
import threading
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ENTRY = Path(
    os.environ.get(
        "DIRECTORDECK_PLUGIN_ENTRY",
        REPOSITORY_ROOT / "plugin" / "__init__.py",
    )
).resolve()
MENU_ENTRY = PLUGIN_ENTRY.parent / "web" / "directordeck-menu.js"


class FakeRoutes:
    def __init__(self) -> None:
        self.handlers: dict[tuple[str, str], Any] = {}

    def get(self, path: str):
        return self.route("GET", path)

    def route(self, method: str, path: str):
        def register(handler):
            self.handlers[(method, path)] = handler
            return handler

        return register


class FakeAiohttpApp:
    def __init__(self) -> None:
        self.on_shutdown: list[Any] = []
        self.static_routes: list[Any] = []

    def add_routes(self, routes: list[Any]) -> None:
        self.static_routes.extend(routes)


class FakeWebResponse:
    def __init__(
        self,
        payload: Any = None,
        *,
        status: int = 200,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.body = (
            json.dumps(payload).encode("utf-8") if payload is not None else b""
        )
        self.status = status
        self.reason = reason
        self.headers = headers or {}
        self.chunks: list[bytes] = []
        self.prepared_with: Any = None
        self.eof_written = False

    async def prepare(self, request: Any) -> None:
        self.prepared_with = request

    async def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)
        self.body += chunk

    async def write_eof(self) -> None:
        self.eof_written = True


class FakeUpstreamContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_any(self):
        for chunk in self._chunks:
            yield chunk


class FakeUpstream:
    def __init__(
        self,
        *,
        status: int,
        reason: str,
        headers: dict[str, str],
        chunks: list[bytes],
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers
        self.content = FakeUpstreamContent(chunks)
        self.released = False

    def release(self) -> None:
        self.released = True


class FakeProxySession:
    def __init__(self, upstream: FakeUpstream) -> None:
        self.upstream = upstream
        self.calls: list[dict[str, Any]] = []

    async def request(
        self,
        method: str,
        target: str,
        **kwargs: Any,
    ) -> FakeUpstream:
        self.calls.append({"method": method, "target": target, **kwargs})
        return self.upstream


@pytest.fixture
def loaded_plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Import the real plugin glue against a side-effect-free fake ComfyUI."""
    fake_routes = FakeRoutes()
    fake_app = FakeAiohttpApp()
    fake_prompt_server = SimpleNamespace(app=fake_app, routes=fake_routes)

    server_module = types.ModuleType("server")
    server_module.PromptServer = SimpleNamespace(instance=fake_prompt_server)

    folder_paths_module = types.ModuleType("folder_paths")
    folder_paths_module.base_path = str(tmp_path / "comfyui")
    folder_paths_module.get_user_directory = lambda: str(tmp_path / "user")

    comfy_package = types.ModuleType("comfy")
    comfy_package.__path__ = []
    cli_args_module = types.ModuleType("comfy.cli_args")
    cli_args_module.args = SimpleNamespace(
        listen="127.0.0.1",
        port=8188,
        tls_certfile=None,
        tls_keyfile=None,
    )
    comfy_package.cli_args = cli_args_module

    # The deliberately old version prevents the import-time bootstrap from
    # creating a thread. Individual tests invoke _run_backend with fake
    # uvicorn/directordeck modules after resetting the state.
    version_module = types.ModuleType("comfyui_version")
    version_module.__version__ = "0.0.0"

    aiohttp_module = types.ModuleType("aiohttp")
    aiohttp_module.web = SimpleNamespace(
        Application=FakeAiohttpApp,
        Request=object,
        Response=FakeWebResponse,
        StreamResponse=FakeWebResponse,
        FileResponse=FakeWebResponse,
        HTTPFound=RuntimeError,
        HTTPNotFound=RuntimeError,
        json_response=lambda payload, status=200: FakeWebResponse(
            payload, status=status
        ),
        static=lambda *args, **kwargs: (args, kwargs),
    )

    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_module)
    monkeypatch.setitem(sys.modules, "server", server_module)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths_module)
    monkeypatch.setitem(sys.modules, "comfy", comfy_package)
    monkeypatch.setitem(sys.modules, "comfy.cli_args", cli_args_module)
    monkeypatch.setitem(sys.modules, "comfyui_version", version_module)

    module_name = f"directordeck_plugin_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    module._state = module._BackendState()
    module._proxy_session = None
    fake_backend_path = tmp_path / "fake-backend"
    module._BACKEND_PATH = fake_backend_path

    yield SimpleNamespace(
        module=module,
        app=fake_app,
        routes=fake_routes,
    )

    try:
        sys.path.remove(str(fake_backend_path))
    except ValueError:
        pass


class FakeBackendControl:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.failure: BaseException | None = None
        self.lock_error: str | None = None
        self.server: Any = None
        self.app_kwargs: dict[str, Any] | None = None


def install_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
    plugin: Any,
) -> FakeBackendControl:
    """Install fake uvicorn and Director modules; no socket is ever opened."""
    control = FakeBackendControl()

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeServer:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config
            self.started = False
            self.should_exit = False
            control.server = self

        async def serve(self) -> None:
            self.started = True
            control.started.set()
            while not self.should_exit and not control.release.is_set():
                await asyncio.sleep(0.001)
            if control.failure is not None:
                raise control.failure

    uvicorn_module = types.ModuleType("uvicorn")
    uvicorn_module.Config = FakeConfig
    uvicorn_module.Server = FakeServer

    class FakeInstanceLockError(RuntimeError):
        pass

    class FakeInstanceLock:
        def __init__(self, _database_path: Path) -> None:
            pass

        def acquire(self) -> None:
            if control.lock_error is not None:
                raise FakeInstanceLockError(control.lock_error)

        def release(self) -> None:
            pass

    directordeck_package = types.ModuleType("directordeck")
    directordeck_package.__path__ = []
    app_module = types.ModuleType("directordeck.app")

    def create_app(**kwargs: Any) -> object:
        control.app_kwargs = kwargs
        return object()

    app_module.create_app = create_app
    lock_module = types.ModuleType("directordeck.instance_lock")
    lock_module.DirectorInstanceLock = FakeInstanceLock
    lock_module.DirectorInstanceLockError = FakeInstanceLockError

    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.setitem(sys.modules, "directordeck", directordeck_package)
    monkeypatch.setitem(sys.modules, "directordeck.app", app_module)
    monkeypatch.setitem(sys.modules, "directordeck.instance_lock", lock_module)
    monkeypatch.setattr(plugin, "_internal_port", lambda: 18788)
    return control


def start_fake_backend(plugin: Any, database_path: Path) -> threading.Thread:
    thread = threading.Thread(
        target=plugin._run_backend,
        args=(database_path,),
        name="fake-directordeck-backend",
        daemon=True,
    )
    plugin._state.thread = thread
    thread.start()
    return thread


async def read_status(loaded_plugin: Any) -> dict[str, Any]:
    handler = loaded_plugin.routes.handlers[("GET", "/directordeck/status")]
    response = await handler(None)
    return json.loads(response.body)


@pytest.mark.parametrize(
    ("listen", "expected_host"),
    (
        ("127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "127.0.0.1"),
        ("0.0.0.0,::", "127.0.0.1"),
        ("192.0.2.25", "192.0.2.25"),
        ("::", "[::1]"),
        ("::1", "[::1]"),
        ("render-host.internal", "render-host.internal"),
    ),
)
def test_comfy_callback_url_follows_the_host_listen_binding(
    loaded_plugin: Any,
    listen: str,
    expected_host: str,
) -> None:
    args = sys.modules["comfy.cli_args"].args
    args.listen = listen
    args.tls_certfile = None

    assert loaded_plugin.module._comfyui_callback_url() == (
        f"http://{expected_host}:8188"
    )


def test_comfy_callback_url_preserves_tls_scheme(loaded_plugin: Any) -> None:
    args = sys.modules["comfy.cli_args"].args
    args.listen = "::1"
    args.port = 9443
    args.tls_certfile = "/tmp/comfy-cert.pem"
    args.tls_keyfile = "/tmp/comfy-key.pem"

    assert loaded_plugin.module._comfyui_callback_url() == "https://[::1]:9443"


def test_comfy_callback_host_is_reachable_for_supported_bindings(
    loaded_plugin: Any,
) -> None:
    plugin = loaded_plugin.module
    cases: list[tuple[int, str, str]] = [
        (socket.AF_INET, "0.0.0.0", "0.0.0.0"),
    ]
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    explicit_ipv4 = next(
        (
            info[4][0]
            for info in infos
            if not info[4][0].startswith("127.")
        ),
        None,
    )
    if explicit_ipv4 is not None:
        cases.append((socket.AF_INET, explicit_ipv4, explicit_ipv4))
    if socket.has_ipv6:
        cases.append((socket.AF_INET6, "::", "::"))

    exercised: list[str] = []
    for family, listen, bind_host in cases:
        listener = socket.socket(family, socket.SOCK_STREAM)
        listener.settimeout(1)
        client = socket.socket(family, socket.SOCK_STREAM)
        client.settimeout(1)
        try:
            listener.bind((bind_host, 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            callback_host = plugin._comfyui_callback_host(listen).strip("[]")
            client.connect((callback_host, port))
            accepted, _address = listener.accept()
            accepted.close()
            exercised.append(listen)
        except OSError:
            if family == socket.AF_INET6:
                continue
            raise
        finally:
            client.close()
            listener.close()

    assert "0.0.0.0" in exercised
    if explicit_ipv4 is not None:
        assert explicit_ipv4 in exercised


def test_backend_receives_the_host_tls_certificate(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = loaded_plugin.module
    args = sys.modules["comfy.cli_args"].args
    certificate = tmp_path / "comfy-cert.pem"
    args.listen = "::1"
    args.port = 9443
    args.tls_certfile = str(certificate)
    args.tls_keyfile = str(tmp_path / "comfy-key.pem")
    control = install_fake_backend(monkeypatch, plugin)
    thread = start_fake_backend(plugin, tmp_path / "director.sqlite3")
    assert control.started.wait(timeout=1)
    try:
        assert control.app_kwargs is not None
        assert control.app_kwargs["comfy_url"] == "https://[::1]:9443"
        assert control.app_kwargs["comfy_tls_certfile"] == str(certificate)
    finally:
        control.release.set()
        thread.join(timeout=1)
        assert not thread.is_alive()


async def test_fake_comfy_lifecycle_reports_ready_then_stopped(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = loaded_plugin.module
    control = install_fake_backend(monkeypatch, plugin)
    thread = start_fake_backend(plugin, tmp_path / "director.sqlite3")
    assert control.started.wait(timeout=1)

    ready = await read_status(loaded_plugin)
    assert ready["backend"] == "ready"
    assert ready["error"] is None
    assert control.app_kwargs is not None
    assert control.app_kwargs["public_api_prefix"] == "/directordeck"
    assert control.app_kwargs["comfy_tls_certfile"] is None

    control.release.set()
    thread.join(timeout=1)
    assert not thread.is_alive()

    stopped = await read_status(loaded_plugin)
    assert stopped["backend"] == "stopped"
    assert stopped["error"] is None


async def test_fake_uvicorn_failure_after_ready_preserves_exception(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = loaded_plugin.module
    control = install_fake_backend(monkeypatch, plugin)
    thread = start_fake_backend(plugin, tmp_path / "director.sqlite3")
    assert control.started.wait(timeout=1)
    assert (await read_status(loaded_plugin))["backend"] == "ready"

    control.failure = RuntimeError("backend crashed after ready")
    control.release.set()
    thread.join(timeout=1)
    assert not thread.is_alive()

    failed = await read_status(loaded_plugin)
    assert failed["backend"] == "failed"
    assert failed["error"] == "RuntimeError: backend crashed after ready"


async def test_fake_comfy_shutdown_stops_backend_and_closes_proxy_session(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = loaded_plugin.module
    control = install_fake_backend(monkeypatch, plugin)
    thread = start_fake_backend(plugin, tmp_path / "director.sqlite3")
    assert control.started.wait(timeout=1)
    assert (await read_status(loaded_plugin))["backend"] == "ready"

    class FakeProxySession:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    proxy_session = FakeProxySession()
    plugin._proxy_session = proxy_session
    assert plugin._on_comfy_shutdown in loaded_plugin.app.on_shutdown

    await plugin._on_comfy_shutdown(loaded_plugin.app)

    assert control.server.should_exit is True
    assert not thread.is_alive()
    assert plugin._state.status == "stopped"
    assert proxy_session.closed is True
    assert plugin._proxy_session is None


def test_instance_lock_failure_keeps_owner_diagnostic(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plugin = loaded_plugin.module
    control = install_fake_backend(monkeypatch, plugin)
    control.lock_error = "database is already owned by pid 4321 on render-host"

    plugin._run_backend(tmp_path / "director.sqlite3")

    assert plugin._state.status == "failed"
    assert plugin._state.error == control.lock_error
    assert control.server is None


def test_shutdown_timeout_is_a_terminal_failure(loaded_plugin: Any) -> None:
    plugin = loaded_plugin.module

    class StuckThread:
        def __init__(self) -> None:
            self.timeout: float | None = None

        def join(self, timeout: float) -> None:
            self.timeout = timeout

        def is_alive(self) -> bool:
            return True

    thread = StuckThread()
    server = SimpleNamespace(should_exit=False)
    plugin._state.status = "ready"
    plugin._state.server = server
    plugin._state.thread = thread

    plugin._shutdown_backend()

    assert server.should_exit is True
    assert thread.timeout == plugin._ATEXIT_JOIN_TIMEOUT_SECONDS
    assert plugin._state.status == "failed"
    assert plugin._state.error == (
        "TimeoutError: Director backend did not stop within 10 seconds"
    )


async def test_stopped_backend_proxy_fails_closed(loaded_plugin: Any) -> None:
    plugin = loaded_plugin.module
    plugin._state.status = "stopped"
    handler = loaded_plugin.routes.handlers[
        ("*", "/directordeck/api/{tail:.*}")
    ]

    response = await handler(None)

    assert response.status == 503
    assert json.loads(response.body) == {
        "error": "directordeck_backend_stopped",
        "detail": None,
    }


@pytest.mark.parametrize(
    (
        "method",
        "tail",
        "query",
        "request_headers",
        "can_read_body",
        "upstream_status",
        "upstream_headers",
        "chunks",
    ),
    (
        (
            "GET",
            "assets/asset-1/content",
            "download=0",
            {
                "Host": "comfy.local",
                "Range": "bytes=10-19",
                "Connection": "keep-alive",
            },
            False,
            206,
            {
                "Content-Type": "video/mp4",
                "Content-Range": "bytes 10-19/100",
                "Content-Length": "10",
            },
            [b"01234", b"56789"],
        ),
        (
            "GET",
            "events",
            "",
            {"Accept": "text/event-stream", "Accept-Encoding": "gzip"},
            False,
            200,
            {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Content-Encoding": "gzip",
            },
            [b"event: task\n", b"data: {}\n\n"],
        ),
        (
            "POST",
            "assets",
            "project_id=project-1",
            {
                "Content-Type": "multipart/form-data; boundary=director-boundary",
                "Content-Length": "1048576",
                "X-Request-ID": "upload-1",
            },
            True,
            200,
            {"Content-Type": "application/json", "Content-Length": "12"},
            [b'{"ok":true}'],
        ),
    ),
)
async def test_proxy_streams_range_sse_and_multipart_without_buffering(
    loaded_plugin: Any,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    tail: str,
    query: str,
    request_headers: dict[str, str],
    can_read_body: bool,
    upstream_status: int,
    upstream_headers: dict[str, str],
    chunks: list[bytes],
) -> None:
    plugin = loaded_plugin.module
    plugin._state.status = "ready"
    plugin._state.port = 18788
    plugin._state.server = SimpleNamespace(started=True)
    request_body = object()
    request = SimpleNamespace(
        method=method,
        match_info={"tail": tail},
        query_string=query,
        headers=request_headers,
        content=request_body,
        can_read_body=can_read_body,
    )
    upstream = FakeUpstream(
        status=upstream_status,
        reason="Partial Content" if upstream_status == 206 else "OK",
        headers=upstream_headers,
        chunks=chunks,
    )
    session = FakeProxySession(upstream)

    async def get_proxy_session() -> FakeProxySession:
        return session

    monkeypatch.setattr(plugin, "_get_proxy_session", get_proxy_session)
    handler = loaded_plugin.routes.handlers[("*", "/directordeck/api/{tail:.*}")]

    response = await handler(request)

    assert len(session.calls) == 1
    call = session.calls[0]
    expected_target = f"http://127.0.0.1:18788/api/{tail}"
    if query:
        expected_target = f"{expected_target}?{query}"
    assert call["method"] == method
    assert call["target"] == expected_target
    assert call["allow_redirects"] is False
    assert call["data"] is (request_body if can_read_body else None)
    forwarded_request_headers = {
        name.lower(): value for name, value in call["headers"].items()
    }
    assert "host" not in forwarded_request_headers
    assert "content-length" not in forwarded_request_headers
    assert "accept-encoding" not in forwarded_request_headers
    assert "connection" not in forwarded_request_headers
    for name in ("range", "accept", "content-type", "x-request-id"):
        if name in {key.lower() for key in request_headers}:
            assert forwarded_request_headers[name] == next(
                value
                for key, value in request_headers.items()
                if key.lower() == name
            )

    assert response.status == upstream_status
    assert response.chunks == chunks
    assert response.eof_written is True
    assert response.prepared_with is request
    assert upstream.released is True
    forwarded_response_headers = {
        name.lower(): value for name, value in response.headers.items()
    }
    assert "content-length" not in forwarded_response_headers
    for name in (
        "content-type",
        "content-range",
        "cache-control",
        "content-encoding",
    ):
        if name in {key.lower() for key in upstream_headers}:
            assert forwarded_response_headers[name] == next(
                value
                for key, value in upstream_headers.items()
                if key.lower() == name
            )


def test_sidebar_status_contract_has_bounded_polling_and_recovery() -> None:
    source = MENU_ENTRY.read_text(encoding="utf-8")

    assert "const STARTING_MAX_ATTEMPTS" in source
    assert "startingAttempts < STARTING_MAX_ATTEMPTS" in source
    assert "scheduleStatusQuery(RUNTIME_POLL_MS)" in source
    assert 'data.backend === "stopped"' in source
    assert 'class="director-refresh"' in source
    assert 'refreshButton.addEventListener("click"' in source
    assert "statusEl.classList.toggle" in source
    assert "el.isConnected" in source
