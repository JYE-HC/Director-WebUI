#!/usr/bin/env python3
"""Small process supervisor for Director Web without systemd.

This is used by WSL2 and other Linux environments where a user systemd bus is
not available.  It intentionally starts the frontend through Vite's node
entrypoint directly (not through npm/cmd wrappers) so that stop can reliably
terminate the whole process group.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT_ROOT / ".director-install" / "run"
DATA_DIR = PROJECT_ROOT / "data"
PID_FILES = {
    "backend": RUN_DIR / "backend.pid",
    "frontend": RUN_DIR / "frontend.pid",
    "comfyui": RUN_DIR / "comfyui.pid",
}
LOG_FILES = {
    "backend": DATA_DIR / "director-backend.log",
    "frontend": DATA_DIR / "director-frontend.log",
    "comfyui": DATA_DIR / "comfyui.log",
}


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def pid_of(name: str) -> int | None:
    try:
        value = int(PID_FILES[name].read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None
    return value or None


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_process_exists(pid: int) -> bool:
    import ctypes

    process_query_limited_information = 0x1000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def write_pid(name: str, pid: int) -> None:
    PID_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    PID_FILES[name].write_text(f"{pid}\n", encoding="utf-8")


def remove_pid(name: str) -> None:
    try:
        PID_FILES[name].unlink()
    except FileNotFoundError:
        pass


def open_log(name: str):
    LOG_FILES[name].parent.mkdir(parents=True, exist_ok=True)
    return open(LOG_FILES[name], "ab", buffering=0)


def _kill_windows_tree(pid: int) -> None:
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _terminate_process_tree(pid: int, *, wait_seconds: float = 8.0) -> bool:
    if os.name == "nt":
        _kill_windows_tree(pid)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline and process_exists(pid):
            time.sleep(0.1)
        return not process_exists(pid)

    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline and process_exists(pid):
        time.sleep(0.1)
    if process_exists(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and process_exists(pid):
            time.sleep(0.1)
    return not process_exists(pid)


def stop_process(name: str) -> bool:
    pid = pid_of(name)
    if pid is None:
        print(f"{name}: not running")
        return True
    if not process_exists(pid):
        remove_pid(name)
        print(f"{name}: stale pid {pid} removed")
        return True
    print(f"stopping {name} (pid {pid})")
    stopped = _terminate_process_tree(pid)
    if stopped:
        remove_pid(name)
        print(f"{name}: stopped")
    else:
        print(f"{name}: failed to stop pid {pid}", file=sys.stderr)
    return stopped


def is_running(name: str) -> bool:
    pid = pid_of(name)
    if pid is None:
        return False
    if process_exists(pid):
        return True
    remove_pid(name)
    return False


def status(name: str) -> int:
    pid = pid_of(name)
    if pid is not None and process_exists(pid):
        print(f"{name}: running (pid {pid})")
        return 0
    if pid is not None:
        remove_pid(name)
    print(f"{name}: stopped")
    return 3


def port_is_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))
        return True
    except OSError:
        return False


def _spawn(command: list[str], *, name: str, cwd: Path, extra_env: dict[str, str] | None = None) -> int:
    log = open_log(name)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        environment.update({key: value for key, value in extra_env.items() if value})
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
        creationflags=creationflags,
        close_fds=True,
    )
    log.close()
    write_pid(name, process.pid)
    return process.pid


def _wait_healthy(name: str, pid: int, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_exists(pid):
            tail = tail_text(name, 20)
            raise RuntimeError(f"{name} exited during startup:\n{tail}")
        time.sleep(0.2)


def tail_text(name: str, lines: int = 20) -> str:
    try:
        text = LOG_FILES[name].read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def start_director(args: argparse.Namespace) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("backend", "frontend"):
        if is_running(name):
            print(f"{name} is already running")
            return 1

    backend_bin = Path(env("DIRECTOR_BACKEND_BIN", str(PROJECT_ROOT / ".venv" / "bin" / "director-web")))
    if not backend_bin.exists():
        print(f"backend entrypoint is missing: {backend_bin}", file=sys.stderr)
        print("run ./bootstrap.sh install first", file=sys.stderr)
        return 1

    node_bin = Path(env("DIRECTOR_NODE_BIN_DIR", "")).resolve() if env("DIRECTOR_NODE_BIN_DIR") else None
    if node_bin is None:
        import shutil

        node_candidate = shutil.which("node")
        node_bin = Path(node_candidate).parent if node_candidate else None
    if node_bin is None:
        print("node is not available; set DIRECTOR_NODE_BIN_DIR", file=sys.stderr)
        return 1
    node = node_bin / ("node.exe" if os.name == "nt" else "node")
    if not node.exists():
        print(f"node is missing: {node}", file=sys.stderr)
        return 1

    frontend_dir = PROJECT_ROOT / "frontend"
    vite_entry = frontend_dir / "node_modules" / "vite" / "bin" / "vite.js"
    if not vite_entry.exists():
        print(f"frontend dependencies are missing: {vite_entry}", file=sys.stderr)
        print("run ./bootstrap.sh install first", file=sys.stderr)
        return 1

    backend_host = args.backend_host or env("DIRECTOR_HOST", "127.0.0.1")
    backend_port = str(args.backend_port or env("DIRECTOR_PORT", "8787"))
    frontend_host = args.frontend_host or env("DIRECTOR_FRONTEND_HOST", "127.0.0.1")
    frontend_port = str(args.frontend_port or env("DIRECTOR_FRONTEND_PORT", "4173"))
    try:
        backend_port_number = int(backend_port)
        frontend_port_number = int(frontend_port)
    except ValueError:
        print(f"invalid Director port: backend={backend_port} frontend={frontend_port}", file=sys.stderr)
        return 1
    if not port_is_available(backend_host, backend_port_number):
        print(f"backend port is already in use: {backend_host}:{backend_port}", file=sys.stderr)
        return 1
    if not port_is_available(frontend_host, frontend_port_number):
        print(f"frontend port is already in use: {frontend_host}:{frontend_port}", file=sys.stderr)
        return 1
    api_origin = f"http://127.0.0.1:{backend_port}"

    print(f"starting backend on {backend_host}:{backend_port}")
    backend_pid = _spawn(
        [str(backend_bin)],
        name="backend",
        cwd=PROJECT_ROOT,
        extra_env={
            "DIRECTOR_HOST": backend_host,
            "DIRECTOR_PORT": backend_port,
            "DIRECTOR_TMPDIR": env("DIRECTOR_TMPDIR"),
        },
    )
    try:
        _wait_healthy("backend", backend_pid)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        stop_process("backend")
        return 1

    print(f"starting frontend on {frontend_host}:{frontend_port}")
    frontend_pid = _spawn(
        [str(node), str(vite_entry)],
        name="frontend",
        cwd=frontend_dir,
        extra_env={
            "DIRECTOR_FRONTEND_HOST": frontend_host,
            "DIRECTOR_FRONTEND_PORT": frontend_port,
            "DIRECTOR_API_ORIGIN": api_origin,
            "DIRECTOR_TMPDIR": env("DIRECTOR_TMPDIR"),
        },
    )
    try:
        _wait_healthy("frontend", frontend_pid)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        stop_process("frontend")
        stop_process("backend")
        return 1

    print_service_url("Director 前端", frontend_host, frontend_port)
    print_service_url("Director 后端", backend_host, backend_port)
    return 0


def stop_director(args: argparse.Namespace) -> int:
    del args
    ok = True
    ok = stop_process("frontend") and ok
    ok = stop_process("backend") and ok
    return 0 if ok else 1


def restart_director(args: argparse.Namespace) -> int:
    code = stop_director(args)
    if code != 0:
        return code
    time.sleep(0.3)
    return start_director(args)


def status_director(args: argparse.Namespace) -> int:
    del args
    code = 0
    for name in ("backend", "frontend"):
        code = max(code, status(name))
    return code


def logs_director(args: argparse.Namespace) -> int:
    if args.target in {"backend", "frontend"}:
        files = [str(LOG_FILES[args.target])]
    else:
        files = [str(LOG_FILES["backend"]), str(LOG_FILES["frontend"])]
    try:
        subprocess.call(["tail", "-F", *files])
    except KeyboardInterrupt:
        return 130
    return 0


def print_service_url(label: str, host: str, port: str) -> None:
    print(f"{label}监听: http://{host}:{port}")
    if host in ("0.0.0.0", "::"):
        print(f"{label}访问: http://127.0.0.1:{port}（0.0.0.0 仅用于监听；跨机器请用服务器实际 IP）")


def start_comfyui(args: argparse.Namespace) -> int:
    del args
    if is_running("comfyui"):
        print("comfyui is already running")
        return 1
    root = env("DIRECTOR_COMFYUI_ROOT")
    if not root:
        print("DIRECTOR_COMFYUI_ROOT is not configured; run ./bootstrap.sh install first", file=sys.stderr)
        return 1
    root_path = Path(root)
    python_bin = Path(env("DIRECTOR_COMFYUI_PYTHON", str(root_path / ".venv" / "bin" / "python")))
    if not python_bin.exists():
        print(f"ComfyUI python is missing: {python_bin}", file=sys.stderr)
        return 1
    listen = env("DIRECTOR_COMFYUI_LISTEN", "127.0.0.1")
    port = env("DIRECTOR_COMFYUI_PORT", "28188")
    try:
        port_number = int(port)
    except ValueError:
        print(f"invalid ComfyUI port: {port}", file=sys.stderr)
        return 1
    if not port_is_available(listen, port_number):
        print(f"ComfyUI port is already in use: {listen}:{port}", file=sys.stderr)
        return 1
    print(f"starting ComfyUI on {listen}:{port}")
    pid = _spawn(
        [str(python_bin), "main.py", "--listen", listen, "--port", port],
        name="comfyui",
        cwd=root_path,
        extra_env={"DIRECTOR_TMPDIR": env("DIRECTOR_TMPDIR")},
    )
    try:
        _wait_healthy("comfyui", pid, timeout=12.0)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        stop_process("comfyui")
        return 1
    print_service_url("ComfyUI", listen, port)
    return 0


def stop_comfyui(args: argparse.Namespace) -> int:
    del args
    return 0 if stop_process("comfyui") else 1


def restart_comfyui(args: argparse.Namespace) -> int:
    code = stop_comfyui(args)
    if code != 0:
        return code
    time.sleep(0.3)
    return start_comfyui(args)


def status_comfyui(args: argparse.Namespace) -> int:
    del args
    return status("comfyui")


def logs_comfyui(args: argparse.Namespace) -> int:
    del args
    try:
        subprocess.call(["tail", "-F", str(LOG_FILES["comfyui"])])
    except KeyboardInterrupt:
        return 130
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="director_supervisor")
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="start Director backend and frontend")
    start.add_argument("--host", dest="host", default="")
    start.add_argument("--backend-host", dest="backend_host", default="")
    start.add_argument("--backend-port", dest="backend_port", type=int, default=0)
    start.add_argument("--frontend-host", dest="frontend_host", default="")
    start.add_argument("--frontend-port", dest="frontend_port", type=int, default=0)
    start.set_defaults(func=start_director)

    restart = sub.add_parser("restart", help="restart Director backend and frontend")
    restart.add_argument("--host", dest="host", default="")
    restart.add_argument("--backend-host", dest="backend_host", default="")
    restart.add_argument("--backend-port", dest="backend_port", type=int, default=0)
    restart.add_argument("--frontend-host", dest="frontend_host", default="")
    restart.add_argument("--frontend-port", dest="frontend_port", type=int, default=0)
    restart.set_defaults(func=restart_director)

    sub.add_parser("stop").set_defaults(func=stop_director)
    sub.add_parser("status").set_defaults(func=status_director)
    logs = sub.add_parser("logs")
    logs.add_argument("target", nargs="?", choices=["backend", "frontend"], default="")
    logs.set_defaults(func=logs_director)

    sub.add_parser("start-comfyui").set_defaults(func=start_comfyui)
    sub.add_parser("stop-comfyui").set_defaults(func=stop_comfyui)
    sub.add_parser("restart-comfyui").set_defaults(func=restart_comfyui)
    sub.add_parser("status-comfyui").set_defaults(func=status_comfyui)
    sub.add_parser("logs-comfyui").set_defaults(func=logs_comfyui)
    return parser


def main(argv: list[str] | None = None) -> NoReturn:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"start", "restart"} and args.host:
        if not args.backend_host:
            args.backend_host = args.host
        if not args.frontend_host:
            args.frontend_host = args.host
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
