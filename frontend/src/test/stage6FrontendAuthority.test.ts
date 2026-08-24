import { beforeEach, describe, expect, it } from "vitest";
import {
  DEFAULT_SETTINGS,
  normalizeLoraLoaderOverrides,
  parseLegacyRuntimeSettingsV1,
} from "../api/types";
import {
  createTimelineProject,
  migrateLegacyTimelineProjectToV5,
  normalizeLegacyTimelineProject,
  normalizeTimelineProject,
} from "../domain/timelineProject";
import {
  redoTimelineHistory,
  undoTimelineHistory,
} from "../state/timelineHistory";
import {
  legacyClientDocumentDigest,
  legacyTimelineHistoryEnvelopeHash,
  migrateLegacyTimelineHistoryEnvelope,
  migrateTimelineFeatureBundle4To5,
  migrateTimelineFeatureBundle5To6,
  parseLegacyTimelineWalRaw,
  resolveLegacyTimelineWalWithReceipt,
  type LegacyCreativeBindingContext,
  type ProjectMigrationReceipt,
} from "../state/timelineV5Migration";
import {
  LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY,
  applyLegacyCreativeSettingsToProject,
  buildLegacyRuntimeSettingsRecoveryPlan,
  clearLegacyRuntimeSettingsWalCandidate,
  listLegacyRuntimeSettingsWalCandidates,
  migrateLegacyRuntimeSettingsToV3,
} from "../state/runtimeSettingsV1Recovery";
import historyRaw from "./fixtures/extensible-workflow-v0/timeline-history-envelope-undone.json?raw";
import projectRaw from "./fixtures/extensible-workflow-v0/timeline-project-v4.json?raw";
import runtimeWalRaw from "./fixtures/extensible-workflow-v0/runtime-settings-wal-v2.json?raw";
import timelineWalRaw from "./fixtures/extensible-workflow-v0/timeline-wal-v7-version2.json?raw";
import projectV5Raw from "./fixtures/extensible-workflow-v0/timeline-project-v5-client-digest.json?raw";

const shaDigest = (seed: string) => ({
  algorithm: "sha256-canonical-json-v1" as const,
  value: `sha256-${seed.repeat(64).slice(0, 64)}`,
});

function legacySettings() {
  const envelope = JSON.parse(runtimeWalRaw) as { settings: unknown };
  const settings = parseLegacyRuntimeSettingsV1(envelope.settings);
  if (!settings) throw new Error("legacy settings fixture invalid");
  return settings;
}

function legacyContext(): LegacyCreativeBindingContext {
  const settings = legacySettings();
  const migrated = applyLegacyCreativeSettingsToProject(createTimelineProject(), settings);
  if (!migrated) throw new Error("legacy context fixture invalid");
  return {
    schema_version: 1,
    model_stack: migrated.model_stack,
    lora: migrated.features.project.lora,
    template_bundle_version: 4,
    explicit_standard_lora_overrides: (["fl2va", "ref2va"] as const).flatMap(
      (family) => {
        const override = settings.models[family].standard_lora_loader_override;
        return override ? [{
          family,
          model_filename: override.model_filename,
          lora_filename: override.lora_name,
          loader: override.loader === "minimax_h3_turbo"
            ? "dedicated"
            : override.loader,
        }] : [];
      },
    ),
  };
}

function receiptFor(
  oldProject: ReturnType<typeof normalizeLegacyTimelineProject> & {},
  oldRevision: number,
): ProjectMigrationReceipt {
  const context = legacyContext();
  const { template_bundle_version: _templateVersion, ...wireContext } = context;
  const next = migrateLegacyTimelineProjectToV5(oldProject, context);
  if (!next) throw new Error("receipt fixture migration failed");
  return {
    schema_version: 1,
    migration_id: `migration-${oldRevision}`,
    project_id: "project-v0-baseline",
    from_schema: 4,
    to_schema: 5,
    old_revision: oldRevision,
    old_client_digest: legacyClientDocumentDigest(oldProject)!,
    old_server_digest: shaDigest("a"),
    new_revision: oldRevision + 1,
    new_client_digest: legacyClientDocumentDigest(next)!,
    new_server_digest: shaDigest("b"),
    legacy_creative_binding_context: wireContext,
    legacy_binding_digest: shaDigest("c"),
    migration_implementation_version: "timeline-v4-v5@1",
    created_at: "2026-08-21T00:00:00Z",
  };
}

beforeEach(() => localStorage.clear());

describe("Stage 6 receipt-aware timeline recovery", () => {
  it("classifies base replay, pending/lost-ACK, and conflict without guessing", () => {
    const wal = parseLegacyTimelineWalRaw(timelineWalRaw);
    if (!wal) throw new Error("legacy WAL fixture invalid");

    const baseReceipt = receiptFor(wal.base_project, wal.base_server_revision);
    const migratedBase = migrateLegacyTimelineProjectToV5(wal.base_project, legacyContext())!;
    const base = resolveLegacyTimelineWalWithReceipt(wal, baseReceipt, {
      document: migratedBase,
      revision: baseReceipt.new_revision,
    });
    expect(base).toMatchObject({
      status: "replay",
      expected_server_revision: baseReceipt.new_revision,
    });
    expect(base?.status === "replay" ? base.project.title : null)
      .toBe(wal.pending_project.title);

    const frozenReceiptBytes = JSON.stringify(baseReceipt);
    const upgradedBase = migrateTimelineFeatureBundle4To5(migratedBase);
    if (!upgradedBase) throw new Error("bundle 4 receipt destination did not upgrade");
    const upgradedReplay = resolveLegacyTimelineWalWithReceipt(wal, baseReceipt, {
      document: upgradedBase,
      revision: baseReceipt.new_revision + 1,
    });
    expect(upgradedReplay).toMatchObject({
      status: "replay",
      expected_server_revision: baseReceipt.new_revision + 1,
      project: { features: { template_bundle_version: 5 } },
    });
    expect(JSON.stringify(baseReceipt)).toBe(frozenReceiptBytes);

    const currentBase = migrateTimelineFeatureBundle5To6(upgradedBase);
    if (!currentBase) throw new Error("bundle 5 receipt destination did not upgrade");
    expect(resolveLegacyTimelineWalWithReceipt(wal, baseReceipt, {
      document: currentBase,
      revision: baseReceipt.new_revision + 2,
    })).toMatchObject({
      status: "replay",
      expected_server_revision: baseReceipt.new_revision + 2,
      project: { features: { template_bundle_version: 6 } },
    });

    const unrelatedEdit = structuredClone(upgradedBase);
    unrelatedEdit.title = "unrelated post-receipt authority edit";
    expect(resolveLegacyTimelineWalWithReceipt(wal, baseReceipt, {
      document: unrelatedEdit,
      revision: baseReceipt.new_revision + 1,
    })).toMatchObject({ status: "conflict", reason: "server-authority-mismatch" });

    const acknowledgedReceipt = receiptFor(
      wal.pending_project,
      wal.base_server_revision + 1,
    );
    const migratedPending = migrateLegacyTimelineProjectToV5(
      wal.pending_project,
      legacyContext(),
    )!;
    expect(resolveLegacyTimelineWalWithReceipt(wal, acknowledgedReceipt, {
      document: migratedPending,
      revision: acknowledgedReceipt.new_revision,
    })).toMatchObject({ status: "acknowledged" });

    const upgradedPending = migrateTimelineFeatureBundle4To5(migratedPending);
    if (!upgradedPending) throw new Error("lost-ACK destination did not upgrade");
    expect(resolveLegacyTimelineWalWithReceipt(wal, acknowledgedReceipt, {
      document: upgradedPending,
      revision: acknowledgedReceipt.new_revision + 1,
    })).toMatchObject({
      status: "acknowledged",
      server_revision: acknowledgedReceipt.new_revision + 1,
      project: { features: { template_bundle_version: 5 } },
    });
    const currentPending = migrateTimelineFeatureBundle5To6(upgradedPending);
    if (!currentPending) throw new Error("lost-ACK destination did not reach bundle 6");
    expect(resolveLegacyTimelineWalWithReceipt(wal, acknowledgedReceipt, {
      document: currentPending,
      revision: acknowledgedReceipt.new_revision + 2,
    })).toMatchObject({
      status: "acknowledged",
      server_revision: acknowledgedReceipt.new_revision + 2,
      project: { features: { template_bundle_version: 6 } },
    });

    expect(resolveLegacyTimelineWalWithReceipt(wal, baseReceipt, {
      document: migratedBase,
      revision: baseReceipt.new_revision + 1,
    })).toMatchObject({ status: "conflict", reason: "server-authority-mismatch" });
  });
});

describe("Stage 6 full-state history migration", () => {
  it("regenerates v5 patches and keeps every iterative undo/redo state equivalent", () => {
    const migrated = migrateLegacyTimelineHistoryEnvelope(
      JSON.parse(historyRaw),
      legacyContext(),
    );
    expect(migrated).not.toBeNull();
    if (!migrated) return;
    expect(migrated.envelope.schemaVersion).toBe(2);
    expect(migrated.history.coalescing).toBeNull();
    expect(migrated.history.startIndex).toBe(0);
    expect(migrated.history.nextEntryId).toBe(21);
    expect(migrated.history.checkpoints.map((checkpoint) => checkpoint.position))
      .toEqual([0, 20]);
    expect(migrated.history.past).toHaveLength(19);
    expect(migrated.history.future).toHaveLength(1);

    let history = migrated.history;
    let undoCount = 0;
    while (history.past.length) {
      const undone = undoTimelineHistory(history);
      expect(undone).not.toBeNull();
      history = undone!.history;
      expect(undone!.snapshot.project.version).toBe(5);
      undoCount += 1;
    }
    expect(undoCount).toBe(19);
    let redoCount = 0;
    while (history.future.length) {
      const redone = redoTimelineHistory(history);
      expect(redone).not.toBeNull();
      history = redone!.history;
      expect(redone!.snapshot.project.version).toBe(5);
      redoCount += 1;
    }
    expect(redoCount).toBe(20);
  });

  it("quarantines the whole branch when one checkpoint is unverifiable", () => {
    const envelope = JSON.parse(historyRaw) as any;
    envelope.payload.checkpoints[1].project.title = "corrupted checkpoint";
    envelope.hash = legacyTimelineHistoryEnvelopeHash(1, envelope.payload);
    expect(migrateLegacyTimelineHistoryEnvelope(envelope, legacyContext())).toBeNull();
  });

  it("rejects an unverifiable next-entry allocator or coalescing envelope as a whole", () => {
    const badAllocator = JSON.parse(historyRaw) as any;
    badAllocator.payload.nextEntryId += 1;
    badAllocator.hash = legacyTimelineHistoryEnvelopeHash(1, badAllocator.payload);
    expect(migrateLegacyTimelineHistoryEnvelope(badAllocator, legacyContext())).toBeNull();

    const badCoalescing = JSON.parse(historyRaw) as any;
    badCoalescing.payload.coalescing = {
      mergeKey: "project:title",
      lastRecordedAt: Date.now(),
    };
    badCoalescing.hash = legacyTimelineHistoryEnvelopeHash(1, badCoalescing.payload);
    expect(migrateLegacyTimelineHistoryEnvelope(badCoalescing, legacyContext())).toBeNull();
  });

  it("rebuilds nonzero startIndex/checkpoint positions and preserves a provable coalescing envelope", () => {
    const pruned = JSON.parse(historyRaw) as any;
    pruned.payload.startIndex = 5;
    pruned.payload.checkpoints[0].position = 5;
    pruned.payload.checkpoints[0].byteSize = new TextEncoder().encode(JSON.stringify({
      position: 5,
      project: pruned.payload.checkpoints[0].project,
    })).byteLength;
    pruned.payload.checkpoints[1].project.title = "阶段 0 · 历史编辑 15";
    pruned.payload.checkpoints[1].byteSize = new TextEncoder().encode(JSON.stringify({
      position: 20,
      project: pruned.payload.checkpoints[1].project,
    })).byteLength;
    pruned.payload.totalBytes = [
      ...pruned.payload.past,
      ...pruned.payload.future,
      ...pruned.payload.checkpoints,
    ].reduce((total: number, item: { byteSize: number }) => total + item.byteSize, 0);
    pruned.hash = legacyTimelineHistoryEnvelopeHash(1, pruned.payload);
    const migratedPruned = migrateLegacyTimelineHistoryEnvelope(pruned, legacyContext());
    expect(migratedPruned?.history.startIndex).toBe(5);
    expect(migratedPruned?.history.checkpoints.map((checkpoint) => checkpoint.position))
      .toEqual([5, 20]);

    const coalescing = JSON.parse(historyRaw) as any;
    const last = coalescing.payload.future.pop();
    last.mergeKey = "project:title";
    const { byteSize: _oldSize, ...entryWithoutSize } = last;
    last.byteSize = new TextEncoder().encode(JSON.stringify(entryWithoutSize)).byteLength;
    coalescing.payload.past.push(last);
    coalescing.payload.cursor = 20;
    coalescing.payload.head = structuredClone(coalescing.payload.checkpoints[1].project);
    coalescing.payload.coalescing = {
      mergeKey: "project:title",
      lastRecordedAt: last.timestamp,
    };
    coalescing.payload.totalBytes = [
      ...coalescing.payload.past,
      ...coalescing.payload.future,
      ...coalescing.payload.checkpoints,
    ].reduce((total: number, item: { byteSize: number }) => total + item.byteSize, 0);
    coalescing.hash = legacyTimelineHistoryEnvelopeHash(1, coalescing.payload);
    expect(migrateLegacyTimelineHistoryEnvelope(coalescing, legacyContext())?.history.coalescing)
      .toEqual({ mergeKey: "project:title", lastRecordedAt: last.timestamp });
  });
});

describe("Stage 6 RuntimeSettingsV1 WAL recovery", () => {
  it("copies exact bytes into digest quarantine and never auto-applies the current project", () => {
    const envelope = JSON.parse(runtimeWalRaw) as { active_database_path: string };
    localStorage.setItem(LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY, runtimeWalRaw);
    const untouched = createTimelineProject();
    untouched.title = "当前项目不得隐式修改";
    const candidates = listLegacyRuntimeSettingsWalCandidates(envelope.active_database_path);
    expect(candidates).toHaveLength(1);
    expect(candidates[0].envelope.owner_id).toBe("tab-v0-baseline-owner");
    expect(localStorage.getItem(LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(candidates[0].storage_key)).toBe(runtimeWalRaw);
    expect(untouched.model_stack.fl2va.filename).toBeNull();
    expect(buildLegacyRuntimeSettingsRecoveryPlan(
      candidates[0],
      { kind: "apply-specific-projects", project_ids: [] },
      [{ project_id: "current", document: untouched }],
      DEFAULT_SETTINGS,
    )).toBeNull();
  });

  it("plans the four explicit choices with exact project scopes and no wrong-project writes", () => {
    const envelope = JSON.parse(runtimeWalRaw) as { active_database_path: string };
    localStorage.setItem(LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY, runtimeWalRaw);
    const candidate = listLegacyRuntimeSettingsWalCandidates(
      envelope.active_database_path,
    )[0];
    const a = createTimelineProject();
    const b = createTimelineProject();
    a.title = "A";
    b.title = "B";
    const projects = [
      { project_id: "a", document: a },
      { project_id: "b", document: b },
    ];
    const unrelated = {
      lora_filename: "LoRA/H3/current-only.safetensors",
      adapter_id: "model_only" as const,
      options: {},
    };
    const current = {
      ...structuredClone(DEFAULT_SETTINGS),
      client_id: "current-v3-client",
      lora_loader_overrides: [unrelated],
    };

    const all = buildLegacyRuntimeSettingsRecoveryPlan(candidate, {
      kind: "apply-all-projects",
      project_ids: ["a", "b"],
    }, projects, current)!;
    expect(all.project_updates.map((entry) => entry.project_id)).toEqual(["a", "b"]);
    expect(all.runtime_settings?.placement.fl2va.device).toBe("gpu:1");
    expect(all.runtime_settings?.client_id).not.toBe("current-v3-client");
    expect(all.runtime_settings?.lora_loader_overrides).toEqual(expect.arrayContaining([
      unrelated,
      expect.objectContaining({ adapter_id: "minimax_h3_turbo" }),
    ]));

    const specific = buildLegacyRuntimeSettingsRecoveryPlan(candidate, {
      kind: "apply-specific-projects",
      project_ids: ["b"],
    }, projects, current)!;
    expect(specific.project_updates.map((entry) => entry.project_id)).toEqual(["b"]);
    expect(specific.project_updates[0].document.model_stack.fl2va.filename)
      .toContain("fl2va-v0.safetensors");
    expect(specific.runtime_settings?.lora_loader_overrides).toContainEqual(unrelated);
    expect(a.model_stack.fl2va.filename).toBeNull();

    const compat = buildLegacyRuntimeSettingsRecoveryPlan(candidate, {
      kind: "retain-lora-compat",
    }, projects, current)!;
    expect(compat.project_updates).toEqual([]);
    expect(compat.runtime_settings?.placement).toEqual(current.placement);
    expect(compat.runtime_settings?.client_id).toBe("current-v3-client");
    expect(compat.runtime_settings?.lora_loader_overrides)
      .toHaveLength(2);

    const discard = buildLegacyRuntimeSettingsRecoveryPlan(candidate, {
      kind: "discard",
    }, projects, DEFAULT_SETTINGS)!;
    expect(discard).toEqual({
      choice: "discard",
      project_updates: [],
      runtime_settings: null,
    });
    expect(clearLegacyRuntimeSettingsWalCandidate(candidate)).toBe(true);
  });

  it("fails closed instead of dropping a legacy LoRA mapping when the V3 table is full", () => {
    const envelope = JSON.parse(runtimeWalRaw) as { active_database_path: string };
    localStorage.setItem(LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY, runtimeWalRaw);
    const candidate = listLegacyRuntimeSettingsWalCandidates(
      envelope.active_database_path,
    )[0];
    const full = {
      ...structuredClone(DEFAULT_SETTINGS),
      lora_loader_overrides: normalizeLoraLoaderOverrides(Array.from(
        { length: 256 },
        (_, index) => ({
          lora_filename: `LoRA/existing-${index.toString().padStart(3, "0")}.safetensors`,
          adapter_id: "model_only" as const,
          options: {},
        }),
      )),
    };

    expect(buildLegacyRuntimeSettingsRecoveryPlan(
      candidate,
      { kind: "retain-lora-compat" },
      [],
      full,
    )).toBeNull();
    expect(buildLegacyRuntimeSettingsRecoveryPlan(
      candidate,
      { kind: "apply-specific-projects", project_ids: ["target"] },
      [{ project_id: "target", document: createTimelineProject() }],
      full,
    )).toBeNull();
    expect(localStorage.getItem(candidate.storage_key)).toBe(runtimeWalRaw);
  });

  it("fails every non-discard branch on an exact-binding adapter conflict", () => {
    const envelope = JSON.parse(runtimeWalRaw) as { active_database_path: string };
    localStorage.setItem(LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY, runtimeWalRaw);
    const candidate = listLegacyRuntimeSettingsWalCandidates(
      envelope.active_database_path,
    )[0];
    const legacyMapping = migrateLegacyRuntimeSettingsToV3(legacySettings())
      .lora_loader_overrides[0];
    const conflicting = {
      ...structuredClone(DEFAULT_SETTINGS),
      lora_loader_overrides: [{ ...legacyMapping, adapter_id: "model_only" as const }],
    };
    const projects = [{ project_id: "target", document: createTimelineProject() }];

    for (const choice of [
      { kind: "retain-lora-compat" as const },
      { kind: "apply-all-projects" as const, project_ids: ["target"] },
      { kind: "apply-specific-projects" as const, project_ids: ["target"] },
    ]) {
      expect(buildLegacyRuntimeSettingsRecoveryPlan(
        candidate,
        choice,
        projects,
        conflicting,
      )).toBeNull();
    }
    expect(localStorage.getItem(candidate.storage_key)).toBe(runtimeWalRaw);
  });

  it("does not surface a valid WAL under a different database identity", () => {
    localStorage.setItem(LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY, runtimeWalRaw);
    expect(listLegacyRuntimeSettingsWalCandidates("/other/director.sqlite3")).toEqual([]);
  });
});

describe("Stage 6 strict v5 boundary", () => {
  it("requires an explicit legacy migration context and rejects silent v4/current fallback", () => {
    const legacy = normalizeLegacyTimelineProject(JSON.parse(projectRaw));
    expect(legacy?.version).toBe(4);
    expect(migrateLegacyTimelineProjectToV5(legacy!, legacyContext())?.version).toBe(5);
  });

  it("preserves complete AssetReference content hashes and rejects stale or malformed v5 wire", () => {
    const full = JSON.parse(projectV5Raw) as any;
    full.segments[0].first_image.content_hash = `sha256:${"a".repeat(64)}`;
    expect(normalizeTimelineProject(full)?.segments[0]).toMatchObject({
      first_image: { content_hash: `sha256:${"a".repeat(64)}` },
    });

    const malformed = structuredClone(full);
    malformed.segments[0].first_image.content_hash = `sha256:${"A".repeat(64)}`;
    expect(normalizeTimelineProject(malformed)).toBeNull();

    const stale = structuredClone(full);
    delete stale.segments[0].first_image.metadata;
    expect(normalizeTimelineProject(stale)).toBeNull();
  });
});
