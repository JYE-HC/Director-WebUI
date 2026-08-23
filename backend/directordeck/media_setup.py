"""ffmpeg/ffprobe availability probe and one-click install.

Director needs ffmpeg + ffprobe (with libx264/aac encoders) for asset
probing, 24fps proxy generation and final assembly. They are external
binaries, so the plugin probes first and offers an explicit, user-confirmed
``pip install static-ffmpeg`` as the one-click path. static-ffmpeg ships both
binaries inside the venv and exposes ``add_paths()`` to put them on PATH for
the current process, so the install takes effect without a restart.
"""

from __future__ import annotations

import re
import shutil
import subprocess  # noqa: S404 - fixed argv, no shell
from typing import Any

from .raylight_setup import RayLightInstallManager

_STATIC_FFMPEG_PACKAGE = "static-ffmpeg"
_PROBE_TIMEOUT_SECONDS = 15


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


def ensure_media_tools_on_path() -> None:
    """Make ffmpeg/ffprobe resolvable when a bundled copy exists.

    Cheap no-op when the tools are already on PATH or static-ffmpeg is not
    installed. Called once at backend startup and after a successful install.
    """
    if media_tools_status()["ready"]:
        return
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except Exception:  # noqa: BLE001 - probing must never break startup
        return


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
    """Pip-install static-ffmpeg; usable in-process, no restart needed."""

    def _success_state(self) -> str:
        return "ready"

    def _after_success(self) -> None:
        ensure_media_tools_on_path()
        self._append("install complete; ffmpeg is ready")

    async def start_install(self) -> None:
        await self.start_packages([_STATIC_FFMPEG_PACKAGE])
