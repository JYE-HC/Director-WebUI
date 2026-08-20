"""DirectorDeck plugin entry.

Embeds the Director Web backend (FastAPI) into the ComfyUI process:

- runs the Director backend with uvicorn on a daemon thread bound to
  127.0.0.1 (internal loopback only, never user facing);
- serves the built frontend from ``dist/`` under ``/directordeck/``;
- reverse-proxies ``/directordeck/api/*`` to the internal backend with fully
  streamed request/response bodies (SSE, Range media, large uploads);
- exposes ``/directordeck/status`` for the menu extension and the SPA to learn
  the explicit ``starting``/``ready``/``stopped``/``failed`` backend state
  without touching the proxy;
- injects this ComfyUI instance's loopback address into the backend at
  construction; the ComfyUI address is not a setting and is never persisted;
- registers the bundled MiniMax-H3-Turbo nodes unconditionally and the
  bundled Director-fork RayLight nodes behind a platform/dependency/conflict
  gate (multi-GPU is opt-in, see docs).

The Director backend stays a pure HTTP/WS client of ComfyUI; nothing here
reaches into ComfyUI internals beyond the documented plugin surface
(PromptServer routes, folder_paths, cli args).
"""

from __future__ import annotations

import asyncio
import atexit
import importlib.util
import ipaddress
import logging
import os
import socket
import sys
import threading
import tomllib
from pathlib import Path

from aiohttp import web

LOGGER = logging.getLogger("DirectorDeck")

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

_PLUGIN_ROOT = Path(__file__).resolve().parent
_BACKEND_PATH = _PLUGIN_ROOT / "backend"
_DIST_DIR = _PLUGIN_ROOT / "dist"
_NODES_DIR = _PLUGIN_ROOT / "nodes"

_DEFAULT_INTERNAL_PORT = 18788
_PORT_SCAN_LIMIT = 20
_ATEXIT_JOIN_TIMEOUT_SECONDS = 10.0
_COMFY_SHUTDOWN_JOIN_TIMEOUT_SECONDS = 8.0

_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class _BackendState:
    def __init__(self) -> None:
        self.status = "starting"
        self.error: str | None = None
        self.port: int | None = None
        self.version: str | None = None
        self.database_path: str | None = None
        self.server = None  # uvicorn.Server once constructed
        self.thread: threading.Thread | None = None
        # RayLight gate outcome: registered | deps_missing | platform_unsupported
        # | conflict | pack_missing | load_failed
        self.raylight = "unknown"
        self.raylight_detail: str | None = None
        self.nodes_error: str | None = None


_state = _BackendState()
_proxy_session = None
_proxy_session_lock = threading.Lock()


def _set_backend_failure(error: str, *, preserve_existing: bool = False) -> None:
    """Record a terminal failure without losing an earlier diagnostic."""
    if preserve_existing and _state.error:
        if error not in _state.error:
            _state.error = f"{_state.error}\n{error}"
    else:
        _state.error = error
    _state.status = "failed"


def _record_shutdown_timeout(thread: threading.Thread, timeout: float) -> None:
    if thread.is_alive():
        _set_backend_failure(
            "TimeoutError: Director backend did not stop within "
            f"{timeout:g} seconds",
            preserve_existing=True,
        )


def _load_node_pack(pack_dir: Path, unique_name: str) -> int:
    """Exec a bundled node pack's __init__.py and merge its node mappings."""
    spec = importlib.util.spec_from_file_location(unique_name, pack_dir / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    NODE_CLASS_MAPPINGS.update(mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
    return len(mappings)


def _foreign_pack_present(name: str) -> bool:
    """A top-level custom_nodes/<name> install that is not ours."""
    import folder_paths

    comfy_root = Path(folder_paths.base_path)
    return (comfy_root / "custom_nodes" / name).is_dir()


def _foreign_raylight_present() -> bool:
    return _foreign_pack_present("raylight")


def _raylight_gate() -> str:
    if not sys.platform.startswith("linux"):
        return "platform_unsupported"
    if importlib.util.find_spec("ray") is None or importlib.util.find_spec("xfuser") is None:
        return "deps_missing"
    if _foreign_raylight_present():
        return "conflict"
    return "ok"


def _load_bundled_nodes() -> None:
    turbo_dir = _NODES_DIR / "ComfyUI-MiniMax-H3-Turbo"
    if _foreign_pack_present("ComfyUI-MiniMax-H3-Turbo"):
        # The bundled copy is an unmodified upstream snapshot; an existing
        # install provides the same nodes. Never register twice.
        LOGGER.info(
            "Director: MiniMax-H3-Turbo nodes skipped; an existing pack already provides them"
        )
    elif turbo_dir.is_dir():
        try:
            count = _load_node_pack(turbo_dir, "director_deck_minimax_h3_turbo")
            LOGGER.info("Director: registered %d MiniMax-H3-Turbo nodes", count)
        except Exception as exc:  # noqa: BLE001 - recorded for /directordeck/status
            _state.nodes_error = f"MiniMax-H3-Turbo: {type(exc).__name__}: {exc}"
            LOGGER.exception("Director: failed to load bundled MiniMax-H3-Turbo nodes")

    gate = _raylight_gate()
    if gate != "ok":
        _state.raylight = gate
        if gate == "deps_missing":
            LOGGER.info(
                "Director: RayLight nodes skipped; install requirements-raylight.txt "
                "and restart to enable multi-GPU"
            )
        elif gate == "platform_unsupported":
            LOGGER.info("Director: RayLight nodes skipped (multi-GPU requires Linux)")
        elif gate == "conflict":
            LOGGER.warning(
                "Director: RayLight nodes skipped; a different raylight pack already "
                "exists in custom_nodes (conflict)"
            )
        return
    raylight_dir = _NODES_DIR / "raylight"
    if not raylight_dir.is_dir():
        _state.raylight = "pack_missing"
        return
    try:
        count = _load_node_pack(raylight_dir, "director_deck_raylight")
        _state.raylight = "registered"
        LOGGER.info("Director: registered %d RayLight nodes", count)
    except Exception as exc:  # noqa: BLE001 - recorded for /directordeck/status
        _state.raylight = "load_failed"
        _state.raylight_detail = f"{type(exc).__name__}: {exc}"
        LOGGER.exception("Director: failed to load bundled RayLight nodes")


def _internal_port() -> int:
    override = os.environ.get("DIRECTOR_INTERNAL_PORT", "").strip()
    if override:
        return int(override)
    candidate = _DEFAULT_INTERNAL_PORT
    for _ in range(_PORT_SCAN_LIMIT):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
            except OSError:
                candidate += 1
                continue
        return candidate
    raise RuntimeError("DirectorDeck: no free loopback port for the backend")


def _database_location() -> Path:
    """The database lives under ComfyUI's user dir, never in the plugin dir."""
    import folder_paths

    db_dir = Path(folder_paths.get_user_directory()) / "directordeck" / "database"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "directordeck.sqlite3"


def _comfyui_callback_host(listen: object) -> str:
    """Choose an address that this process can reach from ComfyUI's binds."""

    addresses = [
        address.strip()
        for address in str(listen or "").split(",")
        if address.strip()
    ]
    if not addresses:
        addresses = ["0.0.0.0"]
    address = addresses[0].removeprefix("[").removesuffix("]")
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return address
    if parsed.is_unspecified:
        parsed = ipaddress.ip_address("::1" if parsed.version == 6 else "127.0.0.1")
    rendered = str(parsed)
    return f"[{rendered}]" if parsed.version == 6 else rendered


def _comfyui_callback_url() -> str:
    from comfy.cli_args import args as comfy_args

    tls_enabled = bool(
        getattr(comfy_args, "tls_certfile", None)
        and getattr(comfy_args, "tls_keyfile", None)
    )
    scheme = "https" if tls_enabled else "http"
    host = _comfyui_callback_host(getattr(comfy_args, "listen", "127.0.0.1"))
    return f"{scheme}://{host}:{getattr(comfy_args, 'port', 8188)}"


_MIN_COMFYUI_VERSION = (0, 33, 0)


def _comfyui_version_check() -> str | None:
    """Coarse version floor; capability probes remain the real gate.

    Returns an error string when the ComfyUI version is known to be too old,
    None otherwise (including unparseable versions, which only log).
    """
    try:
        import comfyui_version

        raw = str(comfyui_version.__version__)
    except Exception as exc:  # noqa: BLE001 - unknown packaging, warn only
        LOGGER.warning("Director: cannot determine ComfyUI version (%s)", exc)
        return None
    numeric: list[int] = []
    for part in raw.replace("-", ".").split("."):
        if part.isdigit():
            numeric.append(int(part))
        else:
            break
    version = tuple((numeric + [0, 0, 0])[:3])
    if version < _MIN_COMFYUI_VERSION:
        return (
            f"ComfyUI {raw} is too old for Director "
            f"(minimum {'.'.join(str(p) for p in _MIN_COMFYUI_VERSION)}); "
            "please upgrade ComfyUI"
        )
    return None


def _plugin_version() -> str | None:
    """Read the plugin package version from the bundled pyproject.toml."""
    try:
        with (_PLUGIN_ROOT / "pyproject.toml").open("rb") as stream:
            return str(tomllib.load(stream)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


def _run_backend(database_path: Path) -> None:
    try:
        if str(_BACKEND_PATH) not in sys.path:
            sys.path.insert(0, str(_BACKEND_PATH))
        import uvicorn

        from directordeck.app import create_app
        from directordeck.instance_lock import (
            DirectorInstanceLock,
            DirectorInstanceLockError,
        )
        from comfy.cli_args import args as comfy_args

        # Probe the single-instance lock ourselves: uvicorn converts a
        # lifespan startup failure into SystemExit, which would hide the
        # actionable owner diagnostic from /directordeck/status.
        probe = DirectorInstanceLock(database_path)
        try:
            probe.acquire()
        except DirectorInstanceLockError as exc:
            _set_backend_failure(str(exc))
            LOGGER.error("Director backend cannot start: %s", exc)
            return
        else:
            probe.release()

        app = create_app(
            database_path=database_path,
            comfy_url=_comfyui_callback_url(),
            comfy_tls_certfile=(
                getattr(comfy_args, "tls_certfile", None)
                if getattr(comfy_args, "tls_keyfile", None)
                else None
            ),
            public_api_prefix="/directordeck",
            raylight_requirements_path=_PLUGIN_ROOT / "requirements-raylight.txt",
        )
        _state.version = _plugin_version()
        port = _internal_port()
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)
        _state.server = server
        _state.port = port
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()
    except BaseException as exc:  # noqa: BLE001 - surfaced via /directordeck/status
        _set_backend_failure(f"{type(exc).__name__}: {exc}")
        LOGGER.exception("Director backend exited with an error")
        return
    # ``uvicorn.Server.started`` remains true after ``serve()`` returns.  Do
    # not leave a backend that was observed as ready stuck in that stale state.
    _state.status = "stopped"
    LOGGER.info("Director backend stopped")


def _shutdown_backend() -> None:
    server = _state.server
    thread = _state.thread
    if server is not None:
        server.should_exit = True
    if thread is not None:
        thread.join(timeout=_ATEXIT_JOIN_TIMEOUT_SECONDS)
        _record_shutdown_timeout(thread, _ATEXIT_JOIN_TIMEOUT_SECONDS)


def _start_backend() -> None:
    _state.status = "starting"
    _state.error = None
    _state.server = None
    _state.port = None
    version_error = _comfyui_version_check()
    if version_error is not None:
        _set_backend_failure(version_error)
        LOGGER.error("Director backend cannot start: %s", version_error)
        return
    database_path = _database_location()
    _state.database_path = str(database_path)
    thread = threading.Thread(
        target=_run_backend,
        args=(database_path,),
        name="directordeck-backend",
        daemon=True,
    )
    _state.thread = thread
    thread.start()
    atexit.register(_shutdown_backend)


async def _on_comfy_shutdown(_app: web.Application) -> None:
    """Stop the backend before aiohttp drains open connections.

    The backend holds long-lived connections to this ComfyUI instance
    (progress websocket, polling clients). aiohttp fires ``on_shutdown``
    before waiting for connection drain, so closing the backend here keeps
    ComfyUI's shutdown from stalling on those connections.
    """
    global _proxy_session
    server = _state.server
    if server is not None:
        server.should_exit = True
    thread = _state.thread
    if thread is not None:
        await asyncio.get_running_loop().run_in_executor(
            None, thread.join, _COMFY_SHUTDOWN_JOIN_TIMEOUT_SECONDS
        )
        _record_shutdown_timeout(thread, _COMFY_SHUTDOWN_JOIN_TIMEOUT_SECONDS)
    if _proxy_session is not None and not _proxy_session.closed:
        await _proxy_session.close()
    _proxy_session = None


def _register_routes() -> None:
    import server as comfy_server

    prompt_server = comfy_server.PromptServer.instance
    prompt_server.app.on_shutdown.append(_on_comfy_shutdown)
    routes = prompt_server.routes

    @routes.get("/directordeck/status")
    async def _director_status(_request: web.Request) -> web.Response:
        if (
            _state.status == "starting"
            and _state.server is not None
            and getattr(_state.server, "started", False)
        ):
            _state.status = "ready"
        return web.json_response(
            {
                "backend": _state.status,
                "error": _state.error,
                "version": _state.version,
                "database_path": _state.database_path,
                "comfy_url": _comfyui_callback_url(),
                "raylight": {
                    "status": _state.raylight,
                    "detail": _state.raylight_detail,
                },
                "nodes_error": _state.nodes_error,
            }
        )

    @routes.get("/directordeck")
    async def _director_root(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/directordeck/")

    @routes.route("*", "/directordeck/api/{tail:.*}")
    async def _director_api_proxy(request: web.Request) -> web.StreamResponse:
        if _state.status in {"failed", "stopped"}:
            error = (
                "directordeck_backend_failed"
                if _state.status == "failed"
                else "directordeck_backend_stopped"
            )
            return web.json_response(
                {"error": error, "detail": _state.error}, status=503
            )
        server = _state.server
        if server is None or not server.started or _state.port is None:
            return web.json_response(
                {"error": "directordeck_backend_starting"}, status=503
            )
        session = await _get_proxy_session()
        tail = request.match_info["tail"]
        target = f"http://127.0.0.1:{_state.port}/api/{tail}"
        if request.query_string:
            target = f"{target}?{request.query_string}"
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS
            and name.lower() not in {"host", "content-length", "accept-encoding"}
        }
        try:
            upstream = await session.request(
                request.method,
                target,
                headers=headers,
                data=request.content if request.can_read_body else None,
                allow_redirects=False,
            )
        except OSError as exc:
            return web.json_response(
                {"error": "directordeck_backend_unreachable", "detail": str(exc)}, status=502
            )
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in _HOP_BY_HOP_HEADERS
            and name.lower() != "content-length"
        }
        downstream = web.StreamResponse(
            status=upstream.status, reason=upstream.reason, headers=response_headers
        )
        await downstream.prepare(request)
        try:
            async for chunk in upstream.content.iter_any():
                await downstream.write(chunk)
        finally:
            upstream.release()
        await downstream.write_eof()
        return downstream

    if _DIST_DIR.is_dir():
        index_file = _DIST_DIR / "index.html"

        @routes.get("/directordeck/")
        async def _director_index(_request: web.Request) -> web.Response:
            if not index_file.is_file():
                raise web.HTTPNotFound
            return web.FileResponse(index_file)

        # Root-level dist files (favicon, manifest, …). Single-segment only;
        # /directordeck/status, /directordeck/api/* and /directordeck/assets/* are all
        # registered before this fallback and win their matches.
        @routes.get("/directordeck/{filename:[^/]+}")
        async def _director_dist_file(request: web.Request) -> web.Response:
            candidate = (_DIST_DIR / request.match_info["filename"]).resolve()
            if candidate.parent != _DIST_DIR or not candidate.is_file():
                raise web.HTTPNotFound
            return web.FileResponse(candidate)

        assets_dir = _DIST_DIR / "assets"
        if assets_dir.is_dir():
            prompt_server.app.add_routes(
                [web.static("/directordeck/assets/", str(assets_dir), show_index=False)]
            )
    else:
        LOGGER.warning("Director: frontend dist not found at %s", _DIST_DIR)


async def _get_proxy_session():
    global _proxy_session
    if _proxy_session is None or _proxy_session.closed:
        import aiohttp

        with _proxy_session_lock:
            if _proxy_session is None or _proxy_session.closed:
                timeout = aiohttp.ClientTimeout(total=None)
                _proxy_session = aiohttp.ClientSession(
                    timeout=timeout, auto_decompress=False
                )
    return _proxy_session


try:
    _load_bundled_nodes()
    _start_backend()
    _register_routes()
except BaseException:  # noqa: BLE001 - a plugin must not break ComfyUI startup
    _state.status = "failed"
    import traceback

    _state.error = traceback.format_exc(limit=5)
    LOGGER.exception("DirectorDeck plugin initialization failed")
