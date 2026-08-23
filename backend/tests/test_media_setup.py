from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

import directordeck.media_setup as media_setup
from directordeck.media_setup import (
    FFmpegInstallManager,
    ensure_media_tools_on_path,
    media_tools_status,
)


def test_status_reports_missing_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_setup.shutil, "which", lambda name: None)
    status = media_tools_status()
    assert status["ready"] is False
    assert status["ffmpeg_available"] is False
    assert status["ffprobe_available"] is False


def test_status_reports_ready_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        media_setup.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(media_setup, "_tool_responds", lambda path: True)
    monkeypatch.setattr(media_setup, "_encoders_ok", lambda path: True)
    status = media_tools_status()
    assert status == {
        "ffmpeg_available": True,
        "ffprobe_available": True,
        "ffmpeg_path": "/usr/bin/ffmpeg",
        "encoders_ok": True,
        "ready": True,
    }


def test_status_rejects_unhealthy_ffprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        media_setup.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(
        media_setup,
        "_tool_responds",
        lambda path: not path.endswith("ffprobe"),
    )
    monkeypatch.setattr(media_setup, "_encoders_ok", lambda path: True)

    status = media_tools_status()

    assert status["ffmpeg_available"] is True
    assert status["ffprobe_available"] is False
    assert status["ready"] is False


def test_tool_responds_requires_zero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        media_setup.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 1, stdout="vendor version", stderr="failed"
        ),
    )
    assert media_setup._tool_responds("ffprobe") is False


def test_encoders_ok_requires_libx264_and_aac(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(cmd, **kwargs):
        assert cmd[1:] == ["-hide_banner", "-encoders"]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=(
                " V....D libx264              libx264 H.264 encoder\n"
                " A..... aac                  AAC encoder\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(media_setup.subprocess, "run", fake_run)
    assert media_setup._encoders_ok("ffmpeg") is True

    monkeypatch.setattr(
        media_setup.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 0, stdout=" A..... aac AAC encoder\n", stderr=""
        ),
    )
    assert media_setup._encoders_ok("ffmpeg") is False

    monkeypatch.setattr(
        media_setup.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            0,
            stdout=" V....D not-libx264 misleading libx264 text\n A..... aac\n",
            stderr="",
        ),
    )
    assert media_setup._encoders_ok("ffmpeg") is False


def test_encoders_ok_rejects_failed_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        media_setup.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd,
            1,
            stdout=" V....D libx264\n A..... aac\n",
            stderr="probe failed",
        ),
    )
    assert media_setup._encoders_ok("ffmpeg") is False


def test_encoders_ok_survives_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 15)

    monkeypatch.setattr(media_setup.subprocess, "run", fake_run)
    assert media_setup._encoders_ok("ffmpeg") is False


def _fake_static_ffmpeg_bundle(
    monkeypatch: pytest.MonkeyPatch,
    bundle_dir: Path,
) -> None:
    fake_run = types.SimpleNamespace(get_platform_dir=lambda: str(bundle_dir))
    fake_package = types.ModuleType("static_ffmpeg")
    fake_package.run = fake_run
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_package)


def _complete_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    for name in ("installed.crumb", f"ffmpeg{suffix}", f"ffprobe{suffix}"):
        (bundle_dir / name).write_text("ready", encoding="utf-8")


def test_ensure_on_path_activates_only_a_complete_downloaded_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "static_ffmpeg" / "bin"
    _fake_static_ffmpeg_bundle(monkeypatch, bundle_dir)
    original_path = os.environ.get("PATH", "")

    assert ensure_media_tools_on_path() is False
    assert os.environ.get("PATH", "") == original_path

    _complete_bundle(bundle_dir)
    assert ensure_media_tools_on_path() is True
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bundle_dir)


def test_ensure_on_path_never_calls_the_network_downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = tmp_path / "static_ffmpeg" / "bin"
    called: list[bool] = []
    fake_run = types.SimpleNamespace(
        get_platform_dir=lambda: str(bundle_dir),
        get_or_fetch_platform_executables_else_raise=lambda: called.append(True),
    )
    fake_package = types.ModuleType("static_ffmpeg")
    fake_package.run = fake_run
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_package)

    assert ensure_media_tools_on_path() is False
    assert called == []


def test_ffmpeg_download_progress_uses_only_real_upstream_percentages() -> None:
    manager = FFmpegInstallManager()
    manager._observe_download_output("ffmpeg: 42.5% - 12s")
    assert manager.snapshot()["progress_percent"] == 42.5

    manager._observe_download_output("ffmpeg: 12.0% - 48s")
    assert manager.snapshot()["progress_percent"] == 42.5

    manager._observe_download_output("unrelated download: 84%")
    assert manager.snapshot()["progress_percent"] == 42.5

    manager._begin()
    assert manager.snapshot()["progress_percent"] is None


def test_download_script_enables_upstream_progress_for_captured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProgress:
        check_tty = True
        hide_cursor = True

    fake_run = types.SimpleNamespace(
        Bar=FakeProgress,
        Spinner=FakeProgress,
        get_or_fetch_platform_executables_else_raise=lambda: (
            "/bundle/ffmpeg",
            "/bundle/ffprobe",
        ),
    )
    fake_package = types.ModuleType("static_ffmpeg")
    fake_package.run = fake_run
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_package)

    exec(media_setup._DOWNLOAD_SCRIPT, {})  # noqa: S102 - fixed product script

    assert FakeProgress.check_tty is False
    assert FakeProgress.hide_cursor is False


async def test_ffmpeg_install_success_reports_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )
    async def downloaded(_manager: FFmpegInstallManager) -> None:
        return None

    monkeypatch.setattr(FFmpegInstallManager, "_download_binaries", downloaded)
    monkeypatch.setattr(media_setup, "ensure_media_tools_on_path", lambda: True)
    monkeypatch.setattr(
        media_setup,
        "media_tools_status",
        lambda: {
            "ffmpeg_available": True,
            "ffprobe_available": True,
            "ffmpeg_path": "/bundle/ffmpeg",
            "encoders_ok": True,
            "ready": True,
        },
    )
    invalidated: list[bool] = []
    manager = FFmpegInstallManager(on_ready=lambda: invalidated.append(True))
    await manager.start_install()
    assert manager._task is not None
    await manager._task
    assert manager.state == "ready"
    assert manager.phase is None
    assert manager.returncode == 0
    assert invalidated == [True]


async def test_ffmpeg_install_stays_running_until_binary_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )
    downloading = asyncio.Event()
    release = asyncio.Event()

    async def delayed_download(manager: FFmpegInstallManager) -> None:
        manager.phase = "downloading_binaries"
        downloading.set()
        await release.wait()

    monkeypatch.setattr(FFmpegInstallManager, "_download_binaries", delayed_download)
    monkeypatch.setattr(media_setup, "ensure_media_tools_on_path", lambda: True)
    monkeypatch.setattr(
        media_setup,
        "media_tools_status",
        lambda: {
            "ffmpeg_available": True,
            "ffprobe_available": True,
            "ffmpeg_path": "/bundle/ffmpeg",
            "encoders_ok": True,
            "ready": True,
        },
    )
    manager = FFmpegInstallManager()

    await manager.start_install()
    await asyncio.wait_for(downloading.wait(), timeout=5)
    assert manager.snapshot()["state"] == "running"
    assert manager.snapshot()["phase"] == "downloading_binaries"
    release.set()
    assert manager._task is not None
    await manager._task
    assert manager.state == "ready"


async def test_ffmpeg_download_failure_never_reports_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )

    async def failed_download(_manager: FFmpegInstallManager) -> None:
        raise RuntimeError("download unavailable")

    monkeypatch.setattr(FFmpegInstallManager, "_download_binaries", failed_download)
    invalidated: list[bool] = []
    manager = FFmpegInstallManager(on_ready=lambda: invalidated.append(True))

    await manager.start_install()
    assert manager._task is not None
    await manager._task

    assert manager.state == "failed"
    assert manager.phase is None
    assert "download unavailable" in (manager.error or "")
    assert invalidated == []


async def test_ffmpeg_cancel_during_verification_never_reports_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )

    async def downloaded(_manager: FFmpegInstallManager) -> None:
        return None

    verification_started = threading.Event()
    release_verification = threading.Event()

    def delayed_status() -> dict[str, object]:
        verification_started.set()
        release_verification.wait(timeout=5)
        return {
            "ffmpeg_available": True,
            "ffprobe_available": True,
            "ffmpeg_path": "/bundle/ffmpeg",
            "encoders_ok": True,
            "ready": True,
        }

    monkeypatch.setattr(FFmpegInstallManager, "_download_binaries", downloaded)
    monkeypatch.setattr(media_setup, "ensure_media_tools_on_path", lambda: True)
    monkeypatch.setattr(media_setup, "media_tools_status", delayed_status)
    invalidated: list[bool] = []
    manager = FFmpegInstallManager(on_ready=lambda: invalidated.append(True))
    await manager.start_install()
    assert await asyncio.to_thread(verification_started.wait, 5)

    await manager.cancel()
    release_verification.set()
    assert manager._task is not None
    await manager._task

    assert manager.state == "idle"
    assert manager.phase is None
    assert invalidated == []
    assert not any("verified and ready" in line for line in manager.log_tail)


async def test_ffmpeg_binary_download_can_be_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(
        media_setup,
        "_DOWNLOAD_SCRIPT",
        "import time; print('download started', flush=True); time.sleep(30)",
    )
    manager = FFmpegInstallManager()
    await manager.start_install()
    for _ in range(500):
        if manager.phase == "downloading_binaries" and manager._process is not None:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("binary download subprocess did not start")

    await manager.cancel()
    assert manager._task is not None
    await asyncio.wait_for(manager._task, timeout=10)

    assert manager.state == "idle"
    assert manager.phase is None


async def test_ffmpeg_binary_download_is_reaped_when_install_task_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(
        media_setup,
        "_DOWNLOAD_SCRIPT",
        "import time; print('download started', flush=True); time.sleep(30)",
    )
    manager = FFmpegInstallManager()
    await manager.start_install()
    for _ in range(500):
        if manager.phase == "downloading_binaries" and manager._process is not None:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("binary download subprocess did not start")

    task = manager._task
    process = manager._process
    assert task is not None
    assert process is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.returncode is not None
    assert manager._process is None


async def test_media_setup_endpoint_shape(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_setup.shutil, "which", lambda name: None)
    response = await client.get("/api/media/setup")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ready"] is False
    assert payload["install"]["state"] == "idle"


async def test_media_setup_progress_poll_is_probe_free(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = client.director_app.state.ffmpeg_install_manager
    manager.state = "running"
    manager.phase = "downloading_binaries"
    manager.progress_percent = 42.5

    def unexpected_probe() -> dict[str, object]:
        raise AssertionError("progress polling must not run ffmpeg probes")

    monkeypatch.setattr("directordeck.app.media_tools_status", unexpected_probe)
    response = await client.get("/api/media/setup")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["install"]["state"] == "running"
    assert payload["install"]["progress_percent"] == 42.5


async def test_media_setup_replaces_a_racing_probe_with_verified_status(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = client.director_app.state.ffmpeg_install_manager
    ready_status = {
        "ffmpeg_available": True,
        "ffprobe_available": True,
        "ffmpeg_path": "/bundle/ffmpeg",
        "encoders_ok": True,
        "ready": True,
    }

    def racing_probe() -> dict[str, object]:
        manager._verified_status = dict(ready_status)
        manager.state = "ready"
        return {
            "ffmpeg_available": False,
            "ffprobe_available": False,
            "ffmpeg_path": None,
            "encoders_ok": False,
            "ready": False,
        }

    monkeypatch.setattr("directordeck.app.media_tools_status", racing_probe)
    response = await client.get("/api/media/setup")

    assert response.status_code == 200
    assert response.json() == {
        **ready_status,
        "install": manager.snapshot(),
    }


async def test_media_setup_probe_does_not_block_other_backend_routes(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow_status() -> dict[str, object]:
        started.set()
        release.wait(timeout=5)
        return {
            "ffmpeg_available": False,
            "ffprobe_available": False,
            "ffmpeg_path": None,
            "encoders_ok": False,
            "ready": False,
        }

    monkeypatch.setattr("directordeck.app.media_tools_status", slow_status)
    media_request = asyncio.create_task(client.get("/api/media/setup"))
    assert await asyncio.to_thread(started.wait, 2)
    try:
        health = await asyncio.wait_for(client.get("/api/health"), timeout=1)
        assert health.status_code == 200
    finally:
        release.set()
    response = await media_request
    assert response.status_code == 200


async def test_media_install_endpoint_flow(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )
    async def downloaded(_manager: FFmpegInstallManager) -> None:
        return None

    ready_status = {
        "ffmpeg_available": True,
        "ffprobe_available": True,
        "ffmpeg_path": "/bundle/ffmpeg",
        "encoders_ok": True,
        "ready": True,
    }
    monkeypatch.setattr(FFmpegInstallManager, "_download_binaries", downloaded)
    monkeypatch.setattr(media_setup, "ensure_media_tools_on_path", lambda: True)
    monkeypatch.setattr(media_setup, "media_tools_status", lambda: ready_status)
    monkeypatch.setattr(
        "directordeck.app.media_tools_status",
        lambda: ready_status,
    )
    started = await client.post("/api/media/ffmpeg/install")
    assert started.status_code == 200, started.text
    manager = client.director_app.state.ffmpeg_install_manager
    assert manager._task is not None
    await manager._task
    status = await client.get("/api/media/setup")
    assert status.json()["install"]["state"] == "ready"
