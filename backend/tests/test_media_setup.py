from __future__ import annotations

import subprocess
import sys
import types

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


def test_ensure_on_path_noop_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        media_setup.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(media_setup, "_tool_responds", lambda path: True)
    monkeypatch.setattr(media_setup, "_encoders_ok", lambda path: True)
    added = []
    fake_module = types.SimpleNamespace(add_paths=lambda: added.append(True))
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_module)
    ensure_media_tools_on_path()
    assert added == []


def test_ensure_on_path_uses_static_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_setup.shutil, "which", lambda name: None)
    added = []
    fake_module = types.SimpleNamespace(add_paths=lambda: added.append(True))
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_module)
    ensure_media_tools_on_path()
    assert added == [True]


def test_ensure_on_path_replaces_existing_unusable_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        media_setup.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(media_setup, "_tool_responds", lambda path: False)
    added = []
    fake_module = types.SimpleNamespace(add_paths=lambda: added.append(True))
    monkeypatch.setitem(sys.modules, "static_ffmpeg", fake_module)

    ensure_media_tools_on_path()

    assert added == [True]


async def test_ffmpeg_install_success_reports_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(media_setup, "ensure_media_tools_on_path", lambda: None)
    manager = FFmpegInstallManager()
    await manager.start_install()
    assert manager._task is not None
    await manager._task
    assert manager.state == "ready"
    assert manager.returncode == 0


async def test_media_setup_endpoint_shape(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_setup.shutil, "which", lambda name: None)
    response = await client.get("/api/media/setup")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ready"] is False
    assert payload["install"]["state"] == "idle"


async def test_media_install_endpoint_flow(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "directordeck.raylight_setup._select_installer",
        lambda: [sys.executable, "-c", "pass"],
    )
    monkeypatch.setattr(media_setup, "ensure_media_tools_on_path", lambda: None)
    started = await client.post("/api/media/ffmpeg/install")
    assert started.status_code == 200, started.text
    manager = client.director_app.state.ffmpeg_install_manager
    assert manager._task is not None
    await manager._task
    status = await client.get("/api/media/setup")
    assert status.json()["install"]["state"] == "ready"
