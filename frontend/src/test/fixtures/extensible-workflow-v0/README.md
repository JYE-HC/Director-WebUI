# Extensible workflow phase-0 browser baselines

These files freeze the browser persistence bytes that existed immediately before
the extensible-workflow architecture migration. They are migration inputs, not
snapshots that a test runner may regenerate or update automatically.

The JSON files are intentionally one physical line followed by one LF byte.
`extensibleWorkflowV0Baseline.test.ts` compares those exact bytes with the
current production codecs and also exercises their recovery semantics.

## Frozen identity and clock

- database: `/srv/director/data/directordeck-v0.sqlite3`
- project: `project-v0-baseline`
- owner: `tab-v0-baseline-owner`
- wall clock: `1787356800123`
- journal write token: `00000000-0000-4000-8000-0000000000a0`
- timeline WAL localStorage key:
  `directordeck:v7:timeline-wal:dcc008df5a0664d1927d4e592eb79177:tab-v0-baseline-owner`
- RuntimeSettings WAL localStorage key:
  `directordeck:v2:runtime-settings-pending`
- history entry IDs: the production counter from `timeline-history-1` through
  `timeline-history-k`

## Files

- `timeline-project-v4.json`: strict current v4 project with both model
  families, typed media, sampling branches and project output settings.
- `timeline-wal-v7-version2.json`: revision-aware v7 local-storage branch
  (`version: 2`) whose server base is revision 7 and whose head is pending.
- `timeline-history-envelope-undone.json`: history envelope schema 1 after 20
  deterministic edits and one undo. Cursor 19 retains a future entry and
  checkpoints at positions 0 and 20.
- `timeline-history-journal-v2.json`: exact IndexedDB journal v2 record for the
  same undone history, including canonical project digests and write token.
- `runtime-settings-wal-v2.json`: owned RuntimeSettings pending WAL containing
  the legacy creative model/LoRA fields that the v5 migration must isolate.

The RuntimeSettings envelope codec currently lives privately in `App.tsx`.
The baseline test freezes its exact envelope bytes and validates its settings
body with the production `sanitizeRuntimeSettings` parser. `App.test.tsx` also
feeds the raw v2 bytes through the real App recovery path and verifies that they
remain intact until the authority ACK clears the exact WAL key. This avoids
adding a new production export merely for tests.

Do not update these files as part of an ordinary schema or codec change. A
deliberate baseline replacement requires architecture-review approval, a manual
byte diff, and an explanation of why old persisted data no longer needs to be
accepted. Migration tests should consume these files; they must not rewrite
them.
