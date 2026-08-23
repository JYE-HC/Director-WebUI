# Director release notes (plugin form)

Director ships as **DirectorDeck**, a ComfyUI plugin published to the ComfyUI
Registry (registry id `director-deck`). The backend embeds in the ComfyUI
process, the SPA is served by ComfyUI under `/directordeck/`, and the database
lives in ComfyUI's `user/directordeck/database/` directory. Model weights, LoRAs
and user assets are not distributed. The former standalone deployment form
(three-process install, bootstrap/installer scripts, systemd units) was
removed; no migration path is provided for pre-plugin installs.

## Compatibility matrix

| Component | Requirement |
| --- | --- |
| ComfyUI | v0.33.0 or newer recommended; older or unparseable versions emit a warning but do not block startup |
| Platform | Linux; Windows portable ComfyUI is supported for single-GPU Standard inference |
| Multi-GPU (RayLight) | Linux only; dependencies install on demand from the settings page |
| Python | the host ComfyUI environment (3.12+ recommended) |

The plugin package bundles the Director-maintained RayLight fork under
`nodes/DirectorDeck-RayLight` and exposes only the eight `DirectorDeckRay*`
node types used by Director workflows. Its Python code and Ray workers use the
private `directordeck_raylight` package namespace. An external
`custom_nodes/raylight` install may coexist, but Director never imports or
selects it and never skips its bundled fork because of it. Standard
LoRA loader nodes are installed separately by the user and selected through an
exact loader mapping; DirectorDeck does not bundle or maintain third-party LoRA
nodes. Director does not reject user-installed loaders by module label,
interface slice or implementation fingerprint; genuine import, prompt-validation
and execution errors are reported when they occur. The old `MiniMaxH3Director`
custom node is neither bundled nor supported.

## Validation boundary

`tools/validate_native_comfy_prompts.py` imports nodes in CPU mode and validates
prompt structure and Director-owned packaged nodes; user-installed Standard
LoRA loaders are not runtime-authorized by provenance or fingerprint. The check
does not load models, queue prompts or start a Ray cluster. A real release still
needs one Standard GPU generation and one generation for every supported
RayLight topology on the maintainer's hardware.

## Release flow

1. Mirror this repository to Director-WebUI (`develop` → PR → `main`) and run
   `python tools/check_release.py` in the mirror.
2. Build the plugin package from the mirror: `python tools/build_plugin.py`.
3. Push the assembled package to the DirectorDeck repository
   (`develop` → PR → `main`).
4. Trigger the manual publish workflow in DirectorDeck to submit the release
   to the ComfyUI Registry (PublisherId `jye-hc`).

## License

Director Web is licensed under GPL-3.0-only, matching ComfyUI's project
license. Bundled third-party Apache-2.0 licenses remain preserved
independently (see `THIRD_PARTY_NOTICES.md`).
