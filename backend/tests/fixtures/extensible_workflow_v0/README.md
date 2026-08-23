# Extensible workflow phase-0 backend baseline

This directory is the immutable behavior baseline captured from public
`Director-WebUI main@6b959d6f73435afe2a83acef6c62fe9d812200fa`.

The files have intentionally different roles:

- `native_prompt_goldens.json` freezes the complete prompt object, node IDs,
  node order, output map, plan and Ray runtime descriptor for Standard and
  RayLight six-recipe workflows, all current LoRA adapters, both continuity
  sources, all audio modes, RayKill with and without LoRA, and the maximum H3
  reference layout.
- `current_v4.sqlite3.gz` is a deterministic gzip archive of a current-schema
  database containing one v4 timeline authority, runtime settings, one parent,
  one child, one take, a caller-assigned prompt ID and a non-empty Ray
  epoch/tail/taint ledger. Never open the tracked archive through `Database`;
  decompress it to a temporary directory first.
- `current_v4_expected.json` is the reviewable semantic projection of that
  database.
- `manifest.json` identifies the source baseline and pins every payload hash.

Tests never update these files. To make an intentional baseline replacement,
review the production diff first, then run:

```bash
UV_CACHE_DIR=/tmp/directordeck-ws0-uv \
  uv run python -m backend.tests.extensible_workflow_v0_fixture_builder --write
```

Review the full generated diff and state why behavior changed. Snapshot-update
flags and automatic acceptance in normal test runs are deliberately unsupported.
Running the module without `--write` is read-only: it regenerates into a
temporary directory and fails if the checked-in baseline differs.
