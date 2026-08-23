from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import directordeck.raylight_setup as raylight_setup
from directordeck.raylight_setup import (
    RayLightInstallConflict,
    RayLightInstallManager,
    RayLightInstallUnavailable,
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


async def test_cancel_before_subprocess_spawn_still_terminates_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel that lands before the spawn assigns ``_process`` must not be lost.

    ``cancel()`` contains no await, so calling it immediately after ``start()``
    deterministically reproduces the pre-spawn window: the install task has not
    run yet and ``_process`` is still None. Without the post-spawn re-check in
    ``_execute`` the subprocess would run orphaned and the task would hang.
    """
    _patch_command(
        monkeypatch,
        [sys.executable, "-c", "import time; time.sleep(30)"],
    )
    manager = RayLightInstallManager()
    await manager.start(Path("/tmp/fake-requirements.txt"))
    await manager.cancel()
    assert manager._process is None
    await asyncio.wait_for(_wait_for_task(manager), timeout=10)
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
        "directordeck.app.raylight_platform_supported", lambda: False
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
    # The endpoint consults the name imported into the app namespace; lift the
    # Linux-only release gate so the install/cancel flow runs on Windows CI.
    monkeypatch.setattr(
        "directordeck.app.raylight_platform_supported", lambda: True
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("# fake\n")
    client.director_app.state.raylight_requirements_path = requirements
    async def cancellable_install(
        manager: RayLightInstallManager,
        _command: list[str],
    ) -> None:
        # The manager tests above cover real subprocess termination, including
        # the pre-spawn race.  Keep this endpoint test focused on API
        # single-flight/cancel delegation; some sandbox child watchers can
        # observe SIGTERM yet never wake Process.wait().
        while not manager._cancel_requested:
            await asyncio.sleep(0)
        manager.returncode = -15
        manager.state = "idle"
        manager._append("install cancelled by user")

    monkeypatch.setattr(RayLightInstallManager, "_execute", cancellable_install)
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


async def test_constraint_file_survives_until_pip_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The torch constraint file must still exist while pip runs.

    Regression: ``_run`` used to unlink the constraint file as soon as
    ``_build_command`` returned, so pip received a ``--constraint`` pointing
    at a deleted file and failed with "Could not open requirements file".
    This test keeps the real ``_build_command`` and stubs only ``_execute``.
    """
    real_version = raylight_setup.importlib.metadata.version

    def fake_version(distribution: str) -> str:
        if distribution == "torch":
            return "9.9.9"
        return real_version(distribution)

    monkeypatch.setattr(raylight_setup.importlib.metadata, "version", fake_version)

    manager = RayLightInstallManager()
    captured: dict[str, object] = {}

    async def fake_execute(self: RayLightInstallManager, command: list[str]) -> None:
        constraint = Path(command[command.index("--constraint") + 1])
        captured["constraint"] = constraint
        captured["exists_during_execute"] = constraint.exists()
        if constraint.exists():
            captured["content"] = constraint.read_text(encoding="utf-8")
        manager.state = "needs_restart"
        manager.returncode = 0

    monkeypatch.setattr(RayLightInstallManager, "_execute", fake_execute)
    await manager._run(Path("/tmp/fake-requirements.txt"), ["fake-pip", "install"])

    assert captured["exists_during_execute"] is True
    assert captured["content"] == "torch==9.9.9\n"
    # The temp constraint file is still cleaned up once the install finishes.
    assert not captured["constraint"].exists()
