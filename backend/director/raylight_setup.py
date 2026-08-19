"""Opt-in multi-GPU (RayLight) setup: capability probe and dependency install.

Multi-GPU inference needs the RayLight fork nodes plus the ``ray``/``xfuser``
python packages inside the ComfyUI venv. Those packages are heavy and only
useful on Linux multi-GPU machines, so they are never installed by default:
the settings toggle runs one explicit, user-confirmed install, and node
registration happens on the next ComfyUI start. Capability is always probed
at runtime and never persisted.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess  # noqa: S404 - argv is fixed, no shell
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_LOG_TAIL_LIMIT = 200
_PIP_TIMEOUT_SECONDS = 30 * 60


def platform_supported() -> bool:
    """Ray/xFuser multi-GPU (NCCL) is Linux-only for this release."""
    return sys.platform.startswith("linux")


def dependencies_installed() -> bool:
    importlib.invalidate_caches()
    return (
        importlib.util.find_spec("ray") is not None
        and importlib.util.find_spec("xfuser") is not None
    )


def pip_available() -> bool:
    return importlib.util.find_spec("pip") is not None


class RayLightInstallUnavailable(RuntimeError):
    """No usable package installer in this python environment."""


def _select_installer() -> list[str]:
    """Base install command for this environment.

    ``uv venv`` environments ship without pip, so fall back to ``uv pip``
    when the uv binary is on PATH; everything else fails closed with a
    manual-command hint.
    """
    if pip_available():
        return [sys.executable, "-m", "pip", "install"]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable]
    raise RayLightInstallUnavailable(
        "neither pip (python -m pip) nor uv is available in this environment; "
        "install the requirements manually: "
        f"{sys.executable} -m pip install -r <plugin>/requirements-raylight.txt"
    )


def default_requirements_path() -> Path:
    """Standalone repo layout; the ComfyUI plugin passes its packaged copy."""
    return (
        Path(__file__).resolve().parents[2]
        / "custom_nodes"
        / "raylight"
        / "requirements.txt"
    )


class RayLightInstallConflict(RuntimeError):
    """An install is already running."""


class RayLightInstallManager:
    """Single-flight asyncio wrapper around a pip subprocess."""

    def __init__(self) -> None:
        self.state = "idle"  # idle | running | needs_restart | failed
        self.log_tail: list[str] = []
        self.returncode: int | None = None
        self.error: str | None = None
        self.started_at: float | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._cancel_requested = False
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "log_tail": list(self.log_tail),
            "returncode": self.returncode,
            "error": self.error,
            "started_at": self.started_at,
        }

    def _build_command(
        self,
        requirements_path: Path,
        constraint: Path | None,
        installer: list[str],
    ) -> list[str]:
        command = list(installer)
        if constraint is not None:
            command += ["--constraint", str(constraint)]
        command += ["-r", str(requirements_path)]
        return command

    async def start(self, requirements_path: Path) -> None:
        async with self._lock:
            if self.state == "running":
                raise RayLightInstallConflict("a RayLight install is already running")
            try:
                installer = _select_installer()
            except RayLightInstallUnavailable as exc:
                self.state = "failed"
                self.error = str(exc)
                raise
            self._begin()
            self._task = asyncio.create_task(self._run(requirements_path, installer))

    async def start_packages(self, packages: list[str]) -> None:
        """Install plain pip packages (no requirements file, no constraint)."""
        async with self._lock:
            if self.state == "running":
                raise RayLightInstallConflict("an install is already running")
            try:
                installer = _select_installer()
            except RayLightInstallUnavailable as exc:
                self.state = "failed"
                self.error = str(exc)
                raise
            self._begin()
            self._task = asyncio.create_task(
                self._execute([*installer, *packages])
            )

    def _begin(self) -> None:
        self.state = "running"
        self.log_tail = []
        self.returncode = None
        self.error = None
        self.started_at = time.time()
        self._cancel_requested = False

    def _success_state(self) -> str:
        return "needs_restart"

    def _after_success(self) -> None:
        """Hook for subclasses (e.g. refresh a capability probe)."""

    async def cancel(self) -> None:
        self._cancel_requested = True
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()

    async def _run(self, requirements_path: Path, installer: list[str]) -> None:
        constraint_path: Path | None = None
        try:
            try:
                torch_version = importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                torch_version = None
            if torch_version:
                # Mirror install.sh: never let the resolver move torch.
                fd, raw_path = tempfile.mkstemp(prefix="director-torch-constraint-")
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(f"torch=={torch_version}\n")
                constraint_path = Path(raw_path)
            command = self._build_command(requirements_path, constraint_path, installer)
        except Exception as exc:  # noqa: BLE001 - surfaced in the status payload
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self._append(self.error)
            return
        finally:
            if constraint_path is not None:
                constraint_path.unlink(missing_ok=True)
        await self._execute(command)

    async def _execute(self, command: list[str]) -> None:
        try:
            env = os.environ.copy()
            env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            self._append(f"$ {' '.join(command)}")
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self._process = process
            assert process.stdout is not None
            try:
                async with asyncio.timeout(_PIP_TIMEOUT_SECONDS):
                    async for raw_line in process.stdout:
                        self._append(raw_line.decode("utf-8", errors="replace").rstrip())
                    returncode = await process.wait()
            except TimeoutError:
                process.kill()
                await process.wait()
                self.state = "failed"
                self.error = f"install timed out after {_PIP_TIMEOUT_SECONDS // 60} minutes"
                self._append(self.error)
                return
        except Exception as exc:  # noqa: BLE001 - surfaced in the status payload
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self._append(self.error)
            return
        finally:
            self._process = None

        self.returncode = returncode
        if self._cancel_requested:
            self.state = "idle"
            self._append("install cancelled by user")
        elif returncode == 0:
            self.state = self._success_state()
            self._after_success()
        else:
            self.state = "failed"
            self.error = f"pip exited with code {returncode}"
            self._append(self.error)

    def _append(self, line: str) -> None:
        self.log_tail.append(line)
        if len(self.log_tail) > _LOG_TAIL_LIMIT:
            del self.log_tail[:-_LOG_TAIL_LIMIT]
