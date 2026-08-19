# Changelog

## 0.2.0 - Unreleased

- Rebuilt Director as a pure ComfyUI plugin (registry name DirectorDeck): the
  backend embeds in the ComfyUI process, the SPA is served by ComfyUI under
  `/directordeck/`, and the database lives in ComfyUI's `user/directordeck/` directory.
- Removed the standalone deployment form: bootstrap/install.sh/director.sh,
  the process supervisor, the release installer, and all multi-process
  tooling. No migration path is provided for pre-plugin installs.

## 0.1.0-rc1 - 2026-08-16

- Prepared a privacy-clean release history for Director Web.
- Bundled the exact Director-compatible RayLight fork and MiniMax H3 Turbo node.
- Added a non-destructive environment checker and recoverable custom-node installer.
- Added offline registry/schema verification and optional online ComfyUI API checks.
- Added runtime rejection of older/stock RayLight initializer schemas.
- Removed maintainer-specific paths and validation fixture names.
- Fixed Python 3.10 asyncio timeout handling for background reconciliation and
  progress websocket readiness.

This is a release candidate. Real GPU smoke tests remain a publishing gate for
the final `v0.1.0` tag.
