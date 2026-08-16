# Director Web 0.1.0 release candidate

This branch is a clean release candidate assembled from Director source commit
`6622bae10e6031368860a3dc266776d9a4dff430`. It intentionally has new Git
history so local development notes, runtime databases, media, logs and the
maintainer's previous commit identity are not published.

## Compatibility matrix

| Component | Requirement | Tested baseline |
| --- | --- | --- |
| Platform | Linux; user systemd for `director.sh` | Linux + systemd 249 |
| Python | 3.10+; 3.13+ not yet in CI | 3.12.13 |
| Node.js | `^20.19`, `^22.13`, or `>=24` | 25.2.1 |
| ComfyUI | tested commit or an official Git descendant | `8f37cf8c833a8f2d3c62e2adbccebfd165623481` |
| Ray | 2.48.0+ | 2.56.1 |
| xFuser | 0.4.4+ | 0.4.5 |
| yunchang | 0.6.0+ | 0.6.4 |
| comfy-kitchen | 0.2.31+ with INT8 attention API | 0.2.31 |
| comfy-aimdo | 0.4.13+ | 0.4.13 |

The tested ComfyUI commit is newer than the `v0.33.0` tag while retaining that
version string. The installer accepts that commit and every official descendant,
then checks live node/API contracts instead of requiring one exact SHA. It never
changes the ComfyUI checkout; a dirty worktree is reported but not blocked.

## Bundled custom nodes

- `raylight`: a Director-modified RayLight 1.8.0 snapshot based on upstream
  commit `4085a37b...`. Required only when a model pool selects two or more
  GPUs. See its `DIRECTOR_MODIFICATIONS.md` and retained Apache-2.0 license.
- `ComfyUI-MiniMax-H3-Turbo`: upstream v1.2.3 at `55fee864...`, Apache-2.0.
  Required only for the dedicated legacy H3 Turbo LoRA loader. The stale
  upstream `node.zip` (v1.2.0) is deliberately not bundled.

Standard inference without the legacy Turbo LoRA uses only ComfyUI core and
official extras. The old `MiniMaxH3Director` custom node is neither bundled nor
supported.

## Validation boundary

`install.sh verify` imports nodes in CPU mode, validates provenance and input
schemas, and does not load models, queue prompts or start a Ray cluster. A real
release still needs one Standard GPU generation and one generation for every
supported RayLight topology on the maintainer's hardware.

## License and final release gate

Director Web is licensed under GPL-3.0-only, matching ComfyUI's project license.
Bundled third-party Apache-2.0 licenses remain preserved independently. Real GPU
smoke tests are still required before promoting this candidate to the final
`v0.1.0` release.
