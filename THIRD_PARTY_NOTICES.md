# Third-party notices

## RayLight

- Upstream: <https://github.com/komikndr/raylight>
- Baseline commit: `4085a37b62d7504cf7746f50c9d59e4fd1495b2b`
- Upstream metadata version: 1.8.0
- Bundled fork version: 1.8.0+director.1
- Copyright: Micko Lesmana and contributors
- License: Apache License 2.0; retained in the packaged plugin at
  `nodes/DirectorDeck-RayLight/LICENSE`

The bundled copy is modified for Director Web and is not endorsed by upstream.
Modified/new files are marked, and the change summary is in
`nodes/DirectorDeck-RayLight/DIRECTOR_MODIFICATIONS.md`. In the Director source
checkout, the maintained fork lives under `custom_nodes/raylight/`.
Its packaged Python import namespace is `directordeck_raylight`; it does not
claim or replace the upstream `raylight` import namespace.

RayLight also contains an adapted ComfyUI-GGUF expansion whose City96
Apache-2.0 license remains at
`nodes/DirectorDeck-RayLight/src/directordeck_raylight/expansion/comfyui_gguf/LICENSE`,
and a context-parallel implementation retaining its NVIDIA 2025 Apache-2.0
header.
