"""ffmpeg/ffprobe availability probe and one-click install.

Director needs ffmpeg + ffprobe (with libx264/aac encoders) for asset
probing, 24fps proxy generation and final assembly. They are external
binaries, so the plugin probes first and offers an explicit, user-confirmed
``pip install static-ffmpeg`` as the one-click path. Installing that Python
package and downloading its platform binaries are separate phases. Director
owns the explicit download, reports its progress, and only activates a fully
downloaded bundle in the current process, so success takes effect without a
restart and startup never performs network I/O.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import re
import shutil
import subprocess  # noqa: S404 - fixed argv, no shell
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

from .raylight_setup import RayLightInstallManager

_STATIC_FFMPEG_PACKAGE = "static-ffmpeg==3.0"
_PROBE_TIMEOUT_SECONDS = 15
_BINARY_DOWNLOAD_TIMEOUT_SECONDS = 15 * 60
_DOWNLOAD_SCRIPT = (
    "from static_ffmpeg import run\n"
    "for progress_type in (run.Bar, run.Spinner):\n"
    "    progress_type.check_tty = False\n"
    "    progress_type.hide_cursor = False\n"
    "ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()\n"
    "print(f'FFMPEG={ffmpeg}', flush=True)\n"
    "print(f'FFPROBE={ffprobe}', flush=True)\n"
)
_DOWNLOAD_PERCENT_RE = re.compile(
    r"ffmpeg:[^\r\n]*?(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)%"
)

logger = logging.getLogger(__name__)


def _unavailable_media_status() -> dict[str, Any]:
    return {
        "ffmpeg_available": False,
        "ffprobe_available": False,
        "ffmpeg_path": None,
        "encoders_ok": False,
        "ready": False,
    }


def _write_console_line(message: str) -> None:
    try:
        print(f"[DirectorDeck] {message}", flush=True)
    except (AttributeError, OSError, ValueError):
        # Some Windows Desktop launch modes have no writable stdout. Console
        # mirroring is optional and must never make the installer fail.
        return


def _binary_on_path(name: str) -> str | None:
    return shutil.which(name)


def _tool_responds(executable: str) -> bool:
    try:
        result = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _downloaded_bundle_directory() -> Path | None:
    """Return a complete static-ffmpeg bundle without initiating a download."""

    try:
        importlib.invalidate_caches()
        from static_ffmpeg import run as static_ffmpeg_run

        bundle_dir = Path(static_ffmpeg_run.get_platform_dir())
    except Exception:  # noqa: BLE001 - offline activation must not break startup
        return None
    suffix = ".exe" if sys.platform == "win32" else ""
    required = (
        bundle_dir / "installed.crumb",
        bundle_dir / f"ffmpeg{suffix}",
        bundle_dir / f"ffprobe{suffix}",
    )
    return bundle_dir if all(path.is_file() for path in required) else None


def ensure_media_tools_on_path() -> bool:
    """Activate an already downloaded bundle without network or subprocesses.

    Startup must never trigger ``static_ffmpeg.add_paths()`` because upstream
    downloads its binaries on first use and may block for many minutes. The
    explicit installer owns that download; startup only prepends a complete,
    crumb-marked bundle to this process' PATH.
    """

    bundle_dir = _downloaded_bundle_directory()
    if bundle_dir is None:
        return False
    rendered = str(bundle_dir)
    current_path = os.environ.get("PATH", "")
    entries = current_path.split(os.pathsep) if current_path else []
    normalized = os.path.normcase(os.path.normpath(rendered))
    if all(
        os.path.normcase(os.path.normpath(entry)) != normalized
        for entry in entries
    ):
        os.environ["PATH"] = os.pathsep.join([rendered, *entries])
    return True


def _encoders_ok(ffmpeg_path: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    listing = result.stdout + result.stderr
    encoder_names = {
        match.group(1)
        for line in listing.splitlines()
        if (
            match := re.match(r"^\s*[A-Z.]{6}\s+(\S+)(?:\s|$)", line)
        ) is not None
    }
    return {"libx264", "aac"} <= encoder_names


def media_tools_status() -> dict[str, Any]:
    ffmpeg_path = _binary_on_path("ffmpeg")
    ffprobe_path = _binary_on_path("ffprobe")
    ffmpeg_available = bool(ffmpeg_path and _tool_responds(ffmpeg_path))
    ffprobe_available = bool(ffprobe_path and _tool_responds(ffprobe_path))
    encoders_ok = _encoders_ok(ffmpeg_path) if ffmpeg_available else False
    return {
        "ffmpeg_available": ffmpeg_available,
        "ffprobe_available": ffprobe_available,
        "ffmpeg_path": ffmpeg_path,
        "encoders_ok": encoders_ok,
        "ready": ffmpeg_available and ffprobe_available and encoders_ok,
    }


class FFmpegInstallManager(RayLightInstallManager):
    """Install, download and verify static-ffmpeg without blocking the API."""

    def __init__(self, *, on_ready: Callable[[], None] | None = None) -> None:
        super().__init__()
        self._on_ready = on_ready
        self.progress_percent: float | None = None
        self._verified_status: dict[str, Any] | None = None
        self._last_console_progress_bucket = -1

    def snapshot(self) -> dict[str, Any]:
        return {
            **super().snapshot(),
            "progress_percent": self.progress_percent,
        }

    def _begin(self) -> None:
        super()._begin()
        self.progress_percent = None
        self._verified_status = None
        self._last_console_progress_bucket = -1

    def media_status_snapshot(self) -> dict[str, Any] | None:
        """Return a probe-free status while installing or after verification."""

        if self.state == "running":
            return _unavailable_media_status()
        if self.state == "ready" and self._verified_status is not None:
            return dict(self._verified_status)
        return None

    def _success_state(self) -> str:
        return "ready"

    def _set_phase(self, phase: str, message: str) -> None:
        self.phase = phase
        self._append(message)
        logger.info("DirectorDeck media setup: %s", message)

    def _observe_download_output(self, output: str) -> float | None:
        matches = _DOWNLOAD_PERCENT_RE.findall(output)
        if not matches:
            return None
        observed = min(100.0, max(0.0, float(matches[-1])))
        previous = self.progress_percent
        self.progress_percent = max(previous or 0.0, observed)
        return self.progress_percent if self.progress_percent != previous else None

    async def _report_console_progress(self, progress: float) -> None:
        bucket = 100 if progress >= 100 else int(progress // 5)
        if bucket <= self._last_console_progress_bucket:
            return
        self._last_console_progress_bucket = bucket
        await asyncio.to_thread(
            _write_console_line,
            f"ffmpeg binary download {progress:.1f}%",
        )

    async def _download_binaries(self) -> None:
        self._set_phase(
            "downloading_binaries",
            "downloading ffmpeg binaries; keep the network proxy connected",
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        command = [sys.executable, "-c", _DOWNLOAD_SCRIPT]
        self._append(f"$ {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        self._process = process
        if self._cancel_requested and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
        assert process.stdout is not None
        buffered = ""
        try:
            await asyncio.to_thread(
                _write_console_line,
                "ffmpeg binary download started",
            )
            async with asyncio.timeout(_BINARY_DOWNLOAD_TIMEOUT_SECONDS):
                while chunk := await process.stdout.read(4096):
                    rendered = chunk.decode("utf-8", errors="replace")
                    buffered += rendered
                    progress = self._observe_download_output(buffered)
                    if progress is not None:
                        await self._report_console_progress(progress)
                    pieces = re.split(r"[\r\n]+", buffered)
                    buffered = pieces.pop()[-8192:]
                    for line in pieces:
                        if line.strip():
                            self._append(line.strip())
                returncode = await process.wait()
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            self.returncode = process.returncode
            raise RuntimeError(
                "ffmpeg binary download timed out; keep the proxy connected and retry"
            ) from None
        except BaseException:
            # Event-loop shutdown can cancel this task without going through
            # the user-facing cancel endpoint. Do not leave the downloader
            # running after Director/ComfyUI has begun to stop.
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
            raise
        finally:
            self._process = None
        if buffered.strip():
            self._append(buffered.strip())
        self.returncode = returncode
        if self._cancel_requested:
            return
        if returncode != 0:
            raise RuntimeError(
                "ffmpeg binary download failed; keep the proxy connected and retry "
                f"(exit code {returncode})"
            )
        self.progress_percent = 100.0
        await self._report_console_progress(100.0)

    @staticmethod
    def _verification_error(status: dict[str, Any]) -> str:
        missing: list[str] = []
        if not status["ffmpeg_available"]:
            missing.append("ffmpeg")
        if not status["ffprobe_available"]:
            missing.append("ffprobe")
        if status["ffmpeg_available"] and not status["encoders_ok"]:
            missing.append("libx264/aac encoders")
        detail = ", ".join(missing) or "unknown media capability"
        return f"ffmpeg verification failed: {detail}"

    async def _after_success(self) -> None:
        await self._download_binaries()
        if self._cancel_requested:
            return
        self._set_phase("verifying", "verifying ffmpeg, ffprobe, libx264 and aac")
        activated = await asyncio.to_thread(ensure_media_tools_on_path)
        if self._cancel_requested:
            return
        if not activated:
            raise RuntimeError("downloaded ffmpeg bundle is incomplete")
        status = await asyncio.to_thread(media_tools_status)
        if self._cancel_requested:
            return
        if not status["ready"]:
            raise RuntimeError(self._verification_error(status))
        self._verified_status = dict(status)
        if self._on_ready is not None:
            await asyncio.to_thread(self._on_ready)
        if self._cancel_requested:
            return
        self._append("install complete; ffmpeg is verified and ready")
        logger.info("DirectorDeck media setup: ffmpeg is verified and ready")

    async def start_install(self) -> None:
        await self.start_packages([_STATIC_FFMPEG_PACKAGE])
