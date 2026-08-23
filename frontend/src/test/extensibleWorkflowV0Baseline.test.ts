import "fake-indexeddb/auto";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { parseLegacyRuntimeSettingsV1 } from "../api/types";
import {
  createTimelineProject,
  migrateLegacyTimelineProjectToV5,
  normalizeLegacyTimelineProject,
  normalizeTimelineProject,
  type LegacyTimelineProjectV4,
} from "../domain/timelineProject";
import { redoTimelineHistory, undoTimelineHistory } from "../state/timelineHistory";
import {
  deleteTimelineHistoryDatabaseForTests,
  hasLegacyV4TimelineHistoryJournalEvidence,
  loadTimelineHistoryJournal,
  timelineHistoryJournalKey,
  type TimelinePersistenceScope,
} from "../state/timelinePersistence";
import {
  legacyClientDocumentDigest,
  migrateLegacyTimelineHistoryEnvelope,
  migrateTimelineFeatureBundle4To5,
  parseLegacyTimelineWalRaw,
  resolveLegacyTimelineWalWithReceipt,
  type LegacyCreativeBindingContext,
  type ProjectMigrationReceipt,
} from "../state/timelineV5Migration";
import {
  applyLegacyCreativeSettingsToProject,
} from "../state/runtimeSettingsV1Recovery";
import historyRaw from "./fixtures/extensible-workflow-v0/timeline-history-envelope-undone.json?raw";
import journalRaw from "./fixtures/extensible-workflow-v0/timeline-history-journal-v2.json?raw";
import projectRaw from "./fixtures/extensible-workflow-v0/timeline-project-v4.json?raw";
import projectV5BackendRaw from "./fixtures/extensible-workflow-v0/timeline-project-v5-backend-model-dump.json?raw";
import projectV5Raw from "./fixtures/extensible-workflow-v0/timeline-project-v5-client-digest.json?raw";
import runtimeSettingsWalRaw from "./fixtures/extensible-workflow-v0/runtime-settings-wal-v2.json?raw";
import timelineWalRaw from "./fixtures/extensible-workflow-v0/timeline-wal-v7-version2.json?raw";

const DATABASE_PATH = "/srv/director/data/directordeck-v0.sqlite3";
const PROJECT_ID = "project-v0-baseline";
const OWNER_ID = "tab-v0-baseline-owner";

const scope: TimelinePersistenceScope = {
  databasePath: DATABASE_PATH,
  projectId: PROJECT_ID,
  ownerId: OWNER_ID,
};

function parseRawFixture<T>(raw: string): T {
  expect(raw.endsWith("\n")).toBe(true);
  expect(raw.slice(0, -1)).not.toContain("\n");
  return JSON.parse(raw) as T;
}

function frozenProject(): LegacyTimelineProjectV4 {
  const parsed = parseRawFixture<unknown>(projectRaw);
  const normalized = normalizeLegacyTimelineProject(parsed);
  if (!normalized) throw new Error("phase-0 v4 project fixture is invalid");
  expect(normalized).toEqual(parsed);
  return normalized;
}

function legacyContext(): LegacyCreativeBindingContext {
  const runtimeEnvelope = parseRawFixture<{ settings: unknown }>(runtimeSettingsWalRaw);
  const settings = parseLegacyRuntimeSettingsV1(runtimeEnvelope.settings);
  if (!settings) throw new Error("phase-0 RuntimeSettingsV1 fixture is invalid");
  const project = applyLegacyCreativeSettingsToProject(createTimelineProject(), settings);
  if (!project) throw new Error("phase-0 creative binding context is invalid");
  return {
    schema_version: 1,
    model_stack: project.model_stack,
    lora: project.features.project.lora,
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
  oldProject: LegacyTimelineProjectV4,
  oldRevision = 7,
  oldServerDigest = `sha256-${"a".repeat(64)}`,
): ProjectMigrationReceipt {
  const context = legacyContext();
  const migrated = migrateLegacyTimelineProjectToV5(oldProject, context);
  if (!migrated) throw new Error("phase-0 project cannot migrate to v5");
  const { template_bundle_version: _templateVersion, ...wireContext } = context;
  return {
    schema_version: 1,
    migration_id: "phase-0-v4-v5",
    project_id: PROJECT_ID,
    from_schema: 4,
    to_schema: 5,
    old_revision: oldRevision,
    old_client_digest: legacyClientDocumentDigest(oldProject)!,
    old_server_digest: {
      algorithm: "sha256-canonical-json-v1",
      value: oldServerDigest,
    },
    new_revision: oldRevision + 1,
    new_client_digest: legacyClientDocumentDigest(migrated)!,
    new_server_digest: {
      algorithm: "sha256-canonical-json-v1",
      value: `sha256-${"b".repeat(64)}`,
    },
    legacy_creative_binding_context: wireContext,
    legacy_binding_digest: {
      algorithm: "sha256-canonical-json-v1",
      value: `sha256-${"c".repeat(64)}`,
    },
    migration_implementation_version: "timeline-v4-v5@1",
    created_at: "2026-08-21T00:00:00Z",
  };
}

async function putRawJournal(value: unknown): Promise<void> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open("directordeck-timeline-history", 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains("journals")) {
        request.result.createObjectStore("journals", { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction("journals", "readwrite");
      transaction.objectStore("journals").put(value);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  } finally {
    database.close();
  }
}

beforeEach(async () => {
  localStorage.clear();
  await deleteTimelineHistoryDatabaseForTests();
});

afterEach(async () => {
  localStorage.clear();
  await deleteTimelineHistoryDatabaseForTests();
});

describe("extensible workflow phase-0 frozen migration baselines", () => {
  it("keeps v4 bytes on the frozen parser and requires explicit creative context for v5", () => {
    const legacy = frozenProject();
    expect(legacy.version).toBe(4);
    expect(legacy.segments.map((segment) => segment.mode)).toEqual(["fl2va", "ref2va"]);
    const migrated = migrateLegacyTimelineProjectToV5(legacy, legacyContext());
    const goldenV5 = parseRawFixture<unknown>(projectV5Raw);
    const backendV5 = parseRawFixture<unknown>(projectV5BackendRaw);
    expect(JSON.stringify(legacy)).toBe(projectRaw.slice(0, -1));
    expect(legacyClientDocumentDigest(legacy)?.value).toBe("fnv1a-93bba5c8");
    expect(normalizeTimelineProject(goldenV5)).toEqual(goldenV5);
    const normalizedBackendV5 = normalizeTimelineProject(backendV5);
    expect(JSON.stringify(normalizedBackendV5)).toBe(projectV5Raw.slice(0, -1));
    expect(legacyClientDocumentDigest(normalizedBackendV5)?.value).toBe("fnv1a-0189db3f");
    expect(legacyClientDocumentDigest(backendV5)?.value).not.toBe("fnv1a-0189db3f");
    expect(JSON.stringify(migrated)).toBe(projectV5Raw.slice(0, -1));
    expect(legacyClientDocumentDigest(migrated)?.value).toBe("fnv1a-0189db3f");
    expect(migrated).toMatchObject({
      version: 5,
      title: legacy.title,
      features: { template_bundle_version: 4 },
    });
  });

  it("parses the exact v7 WAL as legacy evidence and resolves it only through its receipt", () => {
    const wal = parseLegacyTimelineWalRaw(timelineWalRaw.slice(0, -1));
    if (!wal) throw new Error("phase-0 timeline WAL fixture is invalid");
    const receipt = receiptFor(wal.base_project, wal.base_server_revision);
    const server = migrateLegacyTimelineProjectToV5(wal.base_project, legacyContext())!;
    expect(resolveLegacyTimelineWalWithReceipt(wal, receipt, {
      document: server,
      revision: receipt.new_revision,
    })).toMatchObject({
      status: "replay",
      project: { version: 5, title: wal.pending_project.title },
      expected_server_revision: receipt.new_revision,
    });
  });

  it("rebuilds the full-state history as v5 patches with equivalent iterative replay", () => {
    const migrated = migrateLegacyTimelineHistoryEnvelope(
      parseRawFixture(historyRaw),
      legacyContext(),
    );
    if (!migrated) throw new Error("phase-0 history fixture did not migrate");
    expect(migrated.history.past).toHaveLength(19);
    expect(migrated.history.future).toHaveLength(1);
    const earlier = undoTimelineHistory(migrated.history);
    expect(earlier?.snapshot.project).toMatchObject({
      version: 5,
      title: "阶段 0 · 历史编辑 18",
    });
    expect(redoTimelineHistory(earlier!.history)?.snapshot.project.title)
      .toBe("阶段 0 · 历史编辑 19");
    expect(redoTimelineHistory(migrated.history)?.snapshot.project.title)
      .toBe("阶段 0 · 历史编辑 20");
  });

  it("detects and migrates the exact v4 IndexedDB journal without rewriting source first", async () => {
    const raw = parseRawFixture<Record<string, unknown>>(journalRaw);
    expect(raw.key).toBe(timelineHistoryJournalKey(scope));
    await putRawJournal(raw);
    expect(await hasLegacyV4TimelineHistoryJournalEvidence(scope)).toBe(true);

    const legacy = frozenProject();
    const receipt = receiptFor(
      legacy,
      7,
      String(raw.confirmedDocumentHash),
    );
    const authority = {
      document: migrateLegacyTimelineProjectToV5(legacy, legacyContext())!,
      revision: receipt.new_revision,
    };
    const restored = await loadTimelineHistoryJournal(scope, authority, receipt);
    expect(restored).toMatchObject({
      status: "restored",
      confirmedRevision: receipt.new_revision,
      project: { version: 5, title: "阶段 0 · 历史编辑 19" },
    });
    if (restored.status !== "restored") throw new Error("legacy journal did not restore");
    expect(restored.history.future).toHaveLength(1);
    expect(redoTimelineHistory(restored.history)?.snapshot.project.title)
      .toBe("阶段 0 · 历史编辑 20");
    expect(await hasLegacyV4TimelineHistoryJournalEvidence(scope)).toBe(true);
  });

  it("bridges the immutable receipt destination through bundle 5 for journal replay", async () => {
    const raw = parseRawFixture<Record<string, any>>(journalRaw);
    await putRawJournal(raw);
    const legacy = frozenProject();
    const receipt = receiptFor(
      legacy,
      7,
      String(raw.confirmedDocumentHash),
    );
    const receiptBytes = JSON.stringify(receipt);
    const receiptDestination = migrateLegacyTimelineProjectToV5(
      legacy,
      legacyContext(),
    )!;
    const current = migrateTimelineFeatureBundle4To5(receiptDestination)!;

    const restored = await loadTimelineHistoryJournal(scope, {
      document: current,
      revision: receipt.new_revision + 1,
    }, receipt);
    expect(restored).toMatchObject({
      status: "restored",
      confirmedRevision: receipt.new_revision + 1,
      confirmedDocument: { features: { template_bundle_version: 5 } },
      project: {
        title: "阶段 0 · 历史编辑 19",
        features: { template_bundle_version: 5 },
      },
    });
    if (restored.status !== "restored") throw new Error("bundle 5 journal did not restore");
    expect(redoTimelineHistory(restored.history)?.snapshot.project.features.template_bundle_version)
      .toBe(5);
    expect(JSON.stringify(receipt)).toBe(receiptBytes);
    expect(await hasLegacyV4TimelineHistoryJournalEvidence(scope)).toBe(true);

    const unrelated = structuredClone(current);
    unrelated.title = "unrelated authority edit";
    await expect(loadTimelineHistoryJournal(scope, {
      document: unrelated,
      revision: receipt.new_revision + 1,
    }, receipt)).resolves.toMatchObject({
      status: "conflict",
      confirmedRevision: receipt.new_revision + 1,
      localProject: { features: { template_bundle_version: 5 } },
    });
  });

  it("recognizes a legacy journal head whose PUT ACK was lost before both migrations", async () => {
    const raw = parseRawFixture<Record<string, any>>(journalRaw);
    await putRawJournal(raw);
    const pending = normalizeLegacyTimelineProject(raw.history.payload.head);
    if (!pending) throw new Error("legacy journal head is invalid");
    const receipt = receiptFor(
      pending,
      Number(raw.confirmedRevision) + 1,
      String(raw.headDocumentHash),
    );
    const destination = migrateLegacyTimelineProjectToV5(
      pending,
      legacyContext(),
    )!;
    const current = migrateTimelineFeatureBundle4To5(destination)!;

    await expect(loadTimelineHistoryJournal(scope, {
      document: current,
      revision: receipt.new_revision + 1,
    }, receipt)).resolves.toMatchObject({
      status: "acknowledged",
      confirmedRevision: receipt.new_revision + 1,
      project: {
        title: pending.title,
        features: { template_bundle_version: 5 },
      },
    });
    expect(await hasLegacyV4TimelineHistoryJournalEvidence(scope)).toBe(true);
  });

  it("freezes RuntimeSettingsV1 creative bytes under the strict legacy parser", () => {
    const envelope = parseRawFixture<{ settings: unknown }>(runtimeSettingsWalRaw);
    const settings = parseLegacyRuntimeSettingsV1(envelope.settings);
    expect(settings).toEqual(envelope.settings);
    expect(settings?.models.fl2va).toMatchObject({
      filename: "diffusion_models/minimax-h3/fl2va-v0.safetensors",
      lora_name: "loras/minimax-h3/turbo-v0.safetensors",
      standard_lora_loader_override: { loader: "dedicated" },
    });
    expect(settings?.models.ref2va.raylight.gpu_select).toEqual([0, 1]);
  });
});
