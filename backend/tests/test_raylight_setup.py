from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import director.raylight_setup as raylight_setup
from director.raylight_setup import (
    RayLightInstallConflict,
    RayLightInstallManager,
    RayLightInstallUnavailable,
    default_requirements_path,
    dependencies_installed,
    platform_supported,
)


def _quick_command(lines: list[str], returncode: int = 0) -> list[str]:
    script = "; ".join(f"print({line!r})" for line in lines)
    script += f"; raise SystemExit({returncode})"
    return [sys.executable, "-c", script]


def _patch_command(monkeypatch: pytest.MonkeyPatch, command: list[str]) -> None:
    monkeypatch.setattr(
        RayLightInstallManager,
        "_build_command",
        lambda self, requirements_path, constraint, installer: command,
    )


async def _wait_for_task(manager: RayLightInstallManager) -> None:
    task = manager._task
    assert task is not None
    await task


def test_platform_supported_matches_host() -> None:
    assert platform_supported() == sys.platform.startswith("linux")


def test_dependencies_installed_is_boolean() -> None:
    assert isinstance(dependencies_installed(), bool)


def test_default_requirements_path_points_at_repo_layout() -> None:
    assert default_requirements_path().name == "requirements.txt"
    assert default_requirements_path().parent.name == "raylight"


def test_select_installer_falls_back_to_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(raylight_setup, "pip_available", lambda: False)
    monkeypatch.setattr(
        raylight_setup.shutil,
        "which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )
    command = raylight_setup._select_installer()
    assert command[0] == "/usr/bin/uv"
    assert command[1:3] == ["pip", "install"]


def test_select_installer_fails_closed_without_any_installer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(raylight_setup, "pip_available", lambda: False)
    monkeypatch.setattr(raylight_setup.shutil, "which", lambda name: None)
    with pytest.raises(RayLightInstallUnavailable):
        raylight_setup._select_installer()


async def test_install_success_moves_to_needs_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_command(monkeypatch, _quick_command(["fake pip output"]))
    manager = RayLightInstallManager()
    await manager.start(Path("/tmp/fake-requirements.txt"))
    await _wait_for_task(manager)
    assert manager.state == "needs_restart"
    assert manager.returncode == 0
    assert any("fake pip output" in line for line in manager.log_tail)


async def test_install_failure_keeps_log_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_command(monkeypatch, _quick_command(["boom"], returncode=3))
    manager = RayLightInstallManager()
    await manager.start(Path("/tmp/fake-requirements.txt"))
    await _wait_for_task(manager)
    assert manager.state == "failed"
    assert manager.returncode == 3
    assert manager.error is not None and "3" in manager.error
    assert any("boom" in line for line in manager.log_tail)


async def test_double_start_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_command(
        monkeypatch,
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    manager = RayLightInstallManager()
    await manager.start(Path("/tmp/fake-requirements.txt"))
    with pytest.raises(RayLightInstallConflict):
        await manager.start(Path("/tmp/fake-requirements.txt"))
    await manager.cancel()
    await _wait_for_task(manager)
    assert manager.state == "idle"


async def test_setup_endpoint_reports_capability_shape(client) -> None:
    response = await client.get("/api/raylight/setup")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["platform_supported"] == sys.platform.startswith("linux")
    assert isinstance(payload["dependencies_installed"], bool)
    assert payload["install"]["state"] == "idle"


async def test_setup_install_rejects_unsupported_platform(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "director.app.raylight_platform_supported", lambda: False
    )
    response = await client.post("/api/raylight/setup/install")
    assert response.status_code == 400, response.text
    assert "Linux" in response.text


async def test_setup_install_rejects_missing_requirements(
    client, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.director_app.state.raylight_requirements_path = (
        tmp_path / "absent-requirements.txt"
    )
    response = await client.post("/api/raylight/setup/install")
    assert response.status_code == 400, response.text


async def test_setup_install_and_cancel_flow(
    client, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("# fake\n")
    client.director_app.state.raylight_requirements_path = requirements
    _patch_command(
        monkeypatch,
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    started = await client.post("/api/raylight/setup/install")
    assert started.status_code == 200, started.text
    assert started.json()["state"] == "running"
    conflict = await client.post("/api/raylight/setup/install")
    assert conflict.status_code == 409, conflict.text
    cancelled = await client.post("/api/raylight/setup/cancel")
    assert cancelled.status_code == 200, cancelled.text
    manager = client.director_app.state.raylight_install_manager
    await asyncio.wait_for(_wait_for_task(manager), timeout=10)
    status = await client.get("/api/raylight/setup")
    assert status.json()["install"]["state"] == "idle"
