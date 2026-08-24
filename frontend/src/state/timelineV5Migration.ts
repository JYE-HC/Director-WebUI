import {
  migrateLegacyTimelineProjectToV5,
  normalizeFeatureSelection,
  normalizeLegacyTimelineProject,
  normalizeTimelineProject,
  projectTimelineFeatureBundle,
  type FeatureSelection,
  type LegacyTimelineProjectV4,
  type ModelStack,
  type TimelineProject,
  type TimelineV5MigrationContext,
  type TimelineWalAuthority,
  type TimelineWalDatabaseIdentity,
} from "../domain/timelineProject";

export {
  migrateTimelineFeatureBundle4To5,
  migrateTimelineFeatureBundle5To6,
} from "../domain/timelineProject";
import {
  applyTimelinePatches,
  createTimelinePatchPair,
  timelineSerializedBytes,
  timelineValuesEqual,
  type TimelinePatch,
} from "./timelinePatches";
import {
  createTimelineHistory,
  deserializeTimelineHistory,
  recordTimelineHistory,
  redoTimelineHistory,
  sealTimelineHistoryCoalescing,
  serializeTimelineHistory,
  undoTimelineHistory,
  type SerializedTimelineHistoryEnvelope,
  type TimelineHistoryContext,
  type TimelineHistoryState,
} from "./timelineHistory";

export const LEGACY_V7_TIMELINE_WAL_STORAGE_PREFIX = "directordeck:v7:timeline-wal:";
export const LEGACY_TIMELINE_WAL_FORMAT = "director-revision-aware-timeline-wal";
export const LEGACY_V7_TIMELINE_WAL_VERSION = 2;

export interface DocumentDigest {
  algorithm: "fnv1a32-json-stringify-v1" | "sha256-canonical-json-v1";
  value: string;
}

export interface LegacyCreativeBindingContext extends TimelineV5MigrationContext {
  schema_version: 1;
  explicit_standard_lora_overrides: Array<{
    family: "fl2va" | "ref2va";
    model_filename: string;
    lora_filename: string;
    loader: "dedicated" | "bypass_model_only" | "model_only";
  }>;
}

export interface ProjectMigrationReceipt {
  schema_version: 1;
  migration_id: string;
  project_id: string;
  from_schema: 4;
  to_schema: 5;
  old_revision: number;
  old_client_digest: DocumentDigest;
  old_server_digest: DocumentDigest;
  new_revision: number;
  new_client_digest: DocumentDigest;
  new_server_digest: DocumentDigest;
  legacy_creative_binding_context: LegacyCreativeBindingContext;
  legacy_binding_digest: DocumentDigest;
  migration_implementation_version: "timeline-v4-v5@1";
  created_at: string;
}

export interface LegacyTimelineWal {
  format: typeof LEGACY_TIMELINE_WAL_FORMAT;
  version: typeof LEGACY_V7_TIMELINE_WAL_VERSION;
  owner_id: string;
  pending: true;
  project_id: string;
  active_database_path: string;
  written_at_ms: number;
  base_server_revision: number;
  base_document_hash: string;
  head_document_hash: string;
  base_project: LegacyTimelineProjectV4;
  pending_project: LegacyTimelineProjectV4;
}

export type ReceiptAwareLegacyWalResolution =
  | {
      status: "replay";
      project: TimelineProject;
      expected_server_revision: number;
      migration_id: string;
    }
  | {
      status: "acknowledged";
      project: TimelineProject;
      server_revision: number;
      migration_id: string;
    }
  | {
      status: "conflict";
      reason:
        | "scope-mismatch"
        | "receipt-mismatch"
        | "server-authority-mismatch"
        | "legacy-base-mismatch";
      legacy_project: LegacyTimelineProjectV4;
      server_project: TimelineProject;
      migration_id: string;
    };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: unknown, keys: readonly string[]): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

function exactLegacyProject(value: unknown): LegacyTimelineProjectV4 | null {
  const normalized = normalizeLegacyTimelineProject(value);
  return normalized && timelineValuesEqual(normalized, value) ? normalized : null;
}

function parseDigest(value: unknown, algorithm: DocumentDigest["algorithm"]): DocumentDigest | null {
  if (!hasExactKeys(value, ["algorithm", "value"]) || value.algorithm !== algorithm) return null;
  const pattern = algorithm === "fnv1a32-json-stringify-v1"
    ? /^fnv1a-[0-9a-f]{8}$/
    : /^sha256-[0-9a-f]{64}$/;
  return typeof value.value === "string" && pattern.test(value.value)
    ? { algorithm, value: value.value }
    : null;
}

function parseModelStack(value: unknown): ModelStack | null {
  if (!isRecord(value)) return null;
  const roles = ["fl2va", "ref2va", "clip", "video_vae", "audio_vae"] as const;
  if (Object.keys(value).length !== roles.length) return null;
  const result = {} as ModelStack;
  for (const role of roles) {
    const selection = value[role];
    if (
      !hasExactKeys(selection, ["filename"]) ||
      !(
        selection.filename === null ||
        (typeof selection.filename === "string" && selection.filename.length >= 1 && selection.filename.length <= 1_024)
      )
    ) return null;
    result[role] = { filename: selection.filename as string | null };
  }
  return result;
}

function parseLegacyCreativeContext(value: unknown): LegacyCreativeBindingContext | null {
  if (!hasExactKeys(value, [
    "schema_version",
    "model_stack",
    "lora",
    "explicit_standard_lora_overrides",
  ]) || value.schema_version !== 1 || !Array.isArray(value.explicit_standard_lora_overrides)) return null;
  const modelStack = parseModelStack(value.model_stack);
  const lora = normalizeFeatureSelection(value.lora);
  if (!modelStack || !lora || value.explicit_standard_lora_overrides.length > 2) return null;
  const overrides: LegacyCreativeBindingContext["explicit_standard_lora_overrides"] = [];
  for (const candidate of value.explicit_standard_lora_overrides) {
    if (
      !hasExactKeys(candidate, ["family", "model_filename", "lora_filename", "loader"]) ||
      !["fl2va", "ref2va"].includes(String(candidate.family)) ||
      !["dedicated", "bypass_model_only", "model_only"].includes(String(candidate.loader)) ||
      typeof candidate.model_filename !== "string" || candidate.model_filename.length < 1 || candidate.model_filename.length > 1_024 ||
      typeof candidate.lora_filename !== "string" || candidate.lora_filename.length < 1 || candidate.lora_filename.length > 1_024
    ) return null;
    overrides.push(candidate as LegacyCreativeBindingContext["explicit_standard_lora_overrides"][number]);
  }
  if (
    overrides.some((record, index) => index > 0 && overrides[index - 1].family >= record.family) ||
    new Set(overrides.map((record) => record.family)).size !== overrides.length
  ) return null;
  return {
    schema_version: 1,
    model_stack: modelStack,
    lora,
    template_bundle_version: 4,
    explicit_standard_lora_overrides: overrides,
  };
}

export function parseProjectMigrationReceipt(value: unknown): ProjectMigrationReceipt | null {
  if (!hasExactKeys(value, [
    "schema_version", "migration_id", "project_id", "from_schema", "to_schema",
    "old_revision", "old_client_digest", "old_server_digest", "new_revision",
    "new_client_digest", "new_server_digest", "legacy_creative_binding_context",
    "legacy_binding_digest", "migration_implementation_version", "created_at",
  ])) return null;
  const oldClient = parseDigest(value.old_client_digest, "fnv1a32-json-stringify-v1");
  const newClient = parseDigest(value.new_client_digest, "fnv1a32-json-stringify-v1");
  const oldServer = parseDigest(value.old_server_digest, "sha256-canonical-json-v1");
  const newServer = parseDigest(value.new_server_digest, "sha256-canonical-json-v1");
  const bindingDigest = parseDigest(value.legacy_binding_digest, "sha256-canonical-json-v1");
  const context = parseLegacyCreativeContext(value.legacy_creative_binding_context);
  if (
    value.schema_version !== 1 || value.from_schema !== 4 || value.to_schema !== 5 ||
    value.migration_implementation_version !== "timeline-v4-v5@1" ||
    typeof value.migration_id !== "string" || value.migration_id.length < 1 || value.migration_id.length > 128 ||
    typeof value.project_id !== "string" || value.project_id.length < 1 || value.project_id.length > 128 ||
    !Number.isSafeInteger(value.old_revision) || (value.old_revision as number) < 0 ||
    !Number.isSafeInteger(value.new_revision) || value.new_revision !== (value.old_revision as number) + 1 ||
    typeof value.created_at !== "string" || value.created_at.length < 1 || value.created_at.length > 64 ||
    !oldClient || !newClient || !oldServer || !newServer || !bindingDigest || !context
  ) return null;
  return {
    schema_version: 1,
    migration_id: value.migration_id,
    project_id: value.project_id,
    from_schema: 4,
    to_schema: 5,
    old_revision: value.old_revision as number,
    old_client_digest: oldClient,
    old_server_digest: oldServer,
    new_revision: value.new_revision as number,
    new_client_digest: newClient,
    new_server_digest: newServer,
    legacy_creative_binding_context: context,
    legacy_binding_digest: bindingDigest,
    migration_implementation_version: "timeline-v4-v5@1",
    created_at: value.created_at,
  };
}

export function legacyClientDocumentDigest(value: unknown): DocumentDigest | null {
  let serialized: string;
  try {
    serialized = JSON.stringify(value);
  } catch {
    return null;
  }
  if (typeof serialized !== "string") return null;
  let hash = 0x811c9dc5;
  for (let index = 0; index < serialized.length; index += 1) {
    hash ^= serialized.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return {
    algorithm: "fnv1a32-json-stringify-v1",
    value: `fnv1a-${hash.toString(16).padStart(8, "0")}`,
  };
}

function legacyWalDocumentHash(value: unknown): string | null {
  let serialized: string;
  try {
    serialized = canonicalJson(value);
  } catch {
    return null;
  }
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(serialized)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `fnv1a64:${hash.toString(16).padStart(16, "0")}`;
}

export function parseLegacyTimelineWal(value: unknown): LegacyTimelineWal | null {
  if (!hasExactKeys(value, [
    "active_database_path", "base_document_hash", "base_project", "base_server_revision",
    "format", "head_document_hash", "owner_id", "pending", "pending_project",
    "project_id", "version", "written_at_ms",
  ])) return null;
  const base = exactLegacyProject(value.base_project);
  const pending = exactLegacyProject(value.pending_project);
  if (
    value.format !== LEGACY_TIMELINE_WAL_FORMAT || value.version !== LEGACY_V7_TIMELINE_WAL_VERSION ||
    value.pending !== true || typeof value.owner_id !== "string" || !/^[A-Za-z0-9._:-]{1,128}$/.test(value.owner_id) ||
    typeof value.project_id !== "string" || !/^[A-Za-z0-9._:-]{1,128}$/.test(value.project_id) ||
    typeof value.active_database_path !== "string" || value.active_database_path.length < 1 ||
    !Number.isSafeInteger(value.written_at_ms) || (value.written_at_ms as number) <= 0 ||
    !Number.isSafeInteger(value.base_server_revision) || (value.base_server_revision as number) < 0 ||
    typeof value.base_document_hash !== "string" || typeof value.head_document_hash !== "string" ||
    !base || !pending || timelineValuesEqual(base, pending) ||
    legacyWalDocumentHash(base) !== value.base_document_hash || legacyWalDocumentHash(pending) !== value.head_document_hash
  ) return null;
  return {
    format: LEGACY_TIMELINE_WAL_FORMAT,
    version: LEGACY_V7_TIMELINE_WAL_VERSION,
    owner_id: value.owner_id,
    pending: true,
    project_id: value.project_id,
    active_database_path: value.active_database_path,
    written_at_ms: value.written_at_ms as number,
    base_server_revision: value.base_server_revision as number,
    base_document_hash: value.base_document_hash,
    head_document_hash: value.head_document_hash,
    base_project: base,
    pending_project: pending,
  };
}

export function parseLegacyTimelineWalRaw(raw: string): LegacyTimelineWal | null {
  try {
    return parseLegacyTimelineWal(JSON.parse(raw));
  } catch {
    return null;
  }
}

function sameDigest(left: DocumentDigest | null, right: DocumentDigest): boolean {
  return left?.algorithm === right.algorithm && left.value === right.value;
}

type ReceiptDestinationAuthority = {
  revision: number;
  featureBundleVersion: 4 | 5 | 6;
};

function receiptDestinationAuthority(
  receiptDestination: TimelineProject,
  receipt: ProjectMigrationReceipt,
  serverProject: TimelineProject,
  serverRevision: number,
): ReceiptDestinationAuthority | null {
  if (!sameDigest(
    legacyClientDocumentDigest(receiptDestination),
    receipt.new_client_digest,
  )) return null;
  if (
    serverRevision === receipt.new_revision &&
    timelineValuesEqual(serverProject, receiptDestination)
  ) {
    return {
      revision: receipt.new_revision,
      featureBundleVersion: 4,
    };
  }
  for (const targetBundleVersion of [5, 6] as const) {
    const projected = projectTimelineFeatureBundle(
      receiptDestination,
      targetBundleVersion,
    );
    const projectedRevision = receipt.new_revision + (projected?.revision_steps ?? 0);
    if (
      projected &&
      projected.revision_steps > 0 &&
      Number.isSafeInteger(projectedRevision) &&
      serverRevision === projectedRevision &&
      timelineValuesEqual(serverProject, projected.document)
    ) {
      return {
        revision: projectedRevision,
        featureBundleVersion: targetBundleVersion,
      };
    }
  }
  return null;
}

export function resolveLegacyTimelineWalWithReceipt(
  walValue: unknown,
  receiptValue: unknown,
  authority: TimelineWalAuthority,
): ReceiptAwareLegacyWalResolution | null {
  const wal = parseLegacyTimelineWal(walValue);
  const receipt = parseProjectMigrationReceipt(receiptValue);
  const serverProject = normalizeTimelineProject(authority.document);
  if (!wal || !receipt || !serverProject || !Number.isSafeInteger(authority.revision)) return null;
  const conflict = (
    reason: Extract<ReceiptAwareLegacyWalResolution, { status: "conflict" }>["reason"],
  ): ReceiptAwareLegacyWalResolution => ({
    status: "conflict",
    reason,
    legacy_project: structuredClone(wal.pending_project),
    server_project: structuredClone(serverProject),
    migration_id: receipt.migration_id,
  });
  if (wal.project_id !== receipt.project_id) return conflict("scope-mismatch");
  const baseDigest = legacyClientDocumentDigest(wal.base_project);
  const pendingDigest = legacyClientDocumentDigest(wal.pending_project);
  const baseMatchesReceipt =
    wal.base_server_revision === receipt.old_revision &&
    sameDigest(baseDigest, receipt.old_client_digest);
  const pendingMatchesReceipt =
    wal.base_server_revision + 1 === receipt.old_revision &&
    sameDigest(pendingDigest, receipt.old_client_digest);
  if (!baseMatchesReceipt && !pendingMatchesReceipt) {
    return conflict("legacy-base-mismatch");
  }
  const receiptSource = baseMatchesReceipt ? wal.base_project : wal.pending_project;
  const receiptDestination = migrateLegacyTimelineProjectToV5(
    receiptSource,
    receipt.legacy_creative_binding_context,
  );
  if (
    !receiptDestination ||
    !sameDigest(
      legacyClientDocumentDigest(receiptDestination),
      receipt.new_client_digest,
    )
  ) return conflict("receipt-mismatch");
  const destinationAuthority = receiptDestinationAuthority(
    receiptDestination,
    receipt,
    serverProject,
    authority.revision,
  );
  if (!destinationAuthority) return conflict("server-authority-mismatch");
  if (pendingMatchesReceipt) {
    return {
      status: "acknowledged",
      project: structuredClone(serverProject),
      server_revision: authority.revision,
      migration_id: receipt.migration_id,
    };
  }
  const migratedPending = migrateLegacyTimelineProjectToV5(
    wal.pending_project,
    receipt.legacy_creative_binding_context,
  );
  if (!migratedPending) return conflict("receipt-mismatch");
  const project = destinationAuthority.featureBundleVersion === 4
    ? migratedPending
    : projectTimelineFeatureBundle(
        migratedPending,
        destinationAuthority.featureBundleVersion,
      )?.document ?? null;
  return project
    ? {
        status: "replay",
        project,
        expected_server_revision: destinationAuthority.revision,
        migration_id: receipt.migration_id,
      }
    : conflict("receipt-mismatch");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value as Record<string, unknown>).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`
  ).join(",")}}`;
}

export function legacyTimelineHistoryEnvelopeHash(schemaVersion: number, payload: unknown): string {
  const input = canonicalJson({
    format: "director-timeline-history",
    version: 1,
    schemaVersion,
    payload,
  });
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(input)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `fnv1a64:${hash.toString(16).padStart(16, "0")}`;
}

interface LegacyHistoryEntry {
  id: string;
  label: string;
  timestamp: number;
  affectedSegmentIds: string[];
  byteSize: number;
  schemaVersion: 1;
  forward: TimelinePatch[];
  inverse: TimelinePatch[];
  beforeContext?: TimelineHistoryContext;
  afterContext?: TimelineHistoryContext;
  mergeKey?: string;
}

function hasExactKeysWithOptional(
  value: unknown,
  required: readonly string[],
  optional: readonly string[],
): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const allowed = new Set([...required, ...optional]);
  return required.every((key) => Object.prototype.hasOwnProperty.call(value, key)) &&
    Object.keys(value).every((key) => allowed.has(key));
}

function parseLegacyHistoryContext(value: unknown): TimelineHistoryContext | null {
  if (!hasExactKeysWithOptional(
    value,
    ["selected_segment_ids", "active_segment_id", "selection_anchor_id"],
    ["restore_segment_selection", "text_editing"],
  )) return null;
  if (
    !Array.isArray(value.selected_segment_ids) ||
    value.selected_segment_ids.length > 128 ||
    !value.selected_segment_ids.every((id) =>
      typeof id === "string" && id.length >= 1 && id.length <= 128) ||
    new Set(value.selected_segment_ids).size !== value.selected_segment_ids.length ||
    !(value.active_segment_id === null ||
      (typeof value.active_segment_id === "string" && value.active_segment_id.length <= 128)) ||
    !(value.selection_anchor_id === null ||
      (typeof value.selection_anchor_id === "string" && value.selection_anchor_id.length <= 128)) ||
    (Object.prototype.hasOwnProperty.call(value, "restore_segment_selection") &&
      typeof value.restore_segment_selection !== "boolean")
  ) return null;
  let textEditing: TimelineHistoryContext["text_editing"];
  if (Object.prototype.hasOwnProperty.call(value, "text_editing")) {
    const text = value.text_editing;
    if (
      !hasExactKeys(text, ["field_key", "start", "end", "direction"]) ||
      typeof text.field_key !== "string" || text.field_key.length < 1 || text.field_key.length > 512 ||
      !Number.isSafeInteger(text.start) || (text.start as number) < 0 ||
      !Number.isSafeInteger(text.end) || (text.end as number) < 0 ||
      !["forward", "backward", "none"].includes(String(text.direction))
    ) return null;
    textEditing = {
      field_key: text.field_key,
      start: text.start as number,
      end: text.end as number,
      direction: text.direction as "forward" | "backward" | "none",
    };
  }
  return {
    selected_segment_ids: [...value.selected_segment_ids] as string[],
    active_segment_id: value.active_segment_id as string | null,
    selection_anchor_id: value.selection_anchor_id as string | null,
    ...(Object.prototype.hasOwnProperty.call(value, "restore_segment_selection")
      ? { restore_segment_selection: value.restore_segment_selection as boolean }
      : {}),
    ...(textEditing ? { text_editing: textEditing } : {}),
  };
}

function legacyAffectedSegmentIds(
  before: LegacyTimelineProjectV4,
  after: LegacyTimelineProjectV4,
): string[] {
  const beforeById = new Map(before.segments.map((segment, index) =>
    [segment.id, { segment, index }] as const));
  const afterById = new Map(after.segments.map((segment, index) =>
    [segment.id, { segment, index }] as const));
  const order = [
    ...before.segments.map((segment) => segment.id),
    ...after.segments.map((segment) => segment.id)
      .filter((id) => !beforeById.has(id)),
  ];
  return order.filter((id) => {
    const left = beforeById.get(id);
    const right = afterById.get(id);
    return !left || !right || left.index !== right.index ||
      !timelineValuesEqual(left.segment, right.segment);
  });
}

function legacyEntryOrdinal(id: string): number | null {
  const suffix = /^timeline-history-([0-9a-z]+)$/.exec(id)?.[1];
  if (!suffix) return null;
  const ordinal = Number.parseInt(suffix, 36);
  return Number.isSafeInteger(ordinal) && ordinal > 0 ? ordinal : null;
}

/**
 * Replays every legacy full state, migrates each with receipt context, then
 * regenerates current forward/inverse patches and checkpoints. No pointer is
 * mechanically rewritten and any unverifiable branch is rejected whole.
 */
export function migrateLegacyTimelineHistoryEnvelope(
  value: unknown,
  context: TimelineV5MigrationContext,
): { history: TimelineHistoryState; envelope: SerializedTimelineHistoryEnvelope } | null {
  try {
    if (!hasExactKeys(value, ["format", "version", "schemaVersion", "hash", "payload"])) return null;
    if (
      value.format !== "director-timeline-history" || value.version !== 1 || value.schemaVersion !== 1 ||
      typeof value.hash !== "string" || !isRecord(value.payload) ||
      legacyTimelineHistoryEnvelopeHash(1, value.payload) !== value.hash || JSON.stringify(value).length > 16 * 1024 * 1024
    ) return null;
    const payload = value.payload;
    if (
      !hasExactKeys(payload, [
        "capacity", "byteBudget", "totalBytes", "startIndex", "nextEntryId",
        "cursor", "past", "future", "checkpoints", "head", "coalescing",
      ]) ||
      !Array.isArray(payload.past) || !Array.isArray(payload.future) || !Array.isArray(payload.checkpoints) ||
      payload.past.length + payload.future.length > 100 ||
      !Number.isInteger(payload.cursor) || payload.cursor !== payload.past.length ||
      !Number.isInteger(payload.capacity) || (payload.capacity as number) < 50 || (payload.capacity as number) > 100 ||
      !Number.isSafeInteger(payload.byteBudget) || (payload.byteBudget as number) <= 0 || (payload.byteBudget as number) > 16 * 1024 * 1024 ||
      !Number.isSafeInteger(payload.totalBytes) || (payload.totalBytes as number) < 0 ||
      !Number.isSafeInteger(payload.startIndex) || (payload.startIndex as number) < 0 ||
      !Number.isSafeInteger(payload.nextEntryId) || (payload.nextEntryId as number) < 1 ||
      (payload.nextEntryId as number) >= Number.MAX_SAFE_INTEGER
    ) return null;
    const rawEntries = [...payload.past, ...[...payload.future].reverse()];
    const entries: LegacyHistoryEntry[] = rawEntries.map((candidate) => {
      if (
        !hasExactKeysWithOptional(candidate, [
          "id", "label", "timestamp", "affectedSegmentIds", "byteSize",
          "schemaVersion", "forward", "inverse",
        ], ["beforeContext", "afterContext", "mergeKey"]) ||
        typeof candidate.id !== "string" || legacyEntryOrdinal(candidate.id) === null ||
        typeof candidate.label !== "string" || candidate.label.length < 1 || candidate.label.length > 256 ||
        typeof candidate.timestamp !== "number" || !Number.isFinite(candidate.timestamp) ||
        !Array.isArray(candidate.affectedSegmentIds) || candidate.affectedSegmentIds.length > 128 ||
        !candidate.affectedSegmentIds.every((id) =>
          typeof id === "string" && id.length >= 1 && id.length <= 128) ||
        new Set(candidate.affectedSegmentIds).size !== candidate.affectedSegmentIds.length ||
        !Number.isSafeInteger(candidate.byteSize) || (candidate.byteSize as number) <= 0 ||
        candidate.schemaVersion !== 1 ||
        !Array.isArray(candidate.forward) || !Array.isArray(candidate.inverse) ||
        candidate.forward.length > 20_000 || candidate.inverse.length > 20_000 ||
        (Object.prototype.hasOwnProperty.call(candidate, "mergeKey") &&
          (typeof candidate.mergeKey !== "string" || candidate.mergeKey.length < 1 || candidate.mergeKey.length > 512))
      ) throw new TypeError("invalid legacy history entry");
      const beforeContext = Object.prototype.hasOwnProperty.call(candidate, "beforeContext")
        ? parseLegacyHistoryContext(candidate.beforeContext)
        : undefined;
      const afterContext = Object.prototype.hasOwnProperty.call(candidate, "afterContext")
        ? parseLegacyHistoryContext(candidate.afterContext)
        : undefined;
      if (
        (Object.prototype.hasOwnProperty.call(candidate, "beforeContext") && !beforeContext) ||
        (Object.prototype.hasOwnProperty.call(candidate, "afterContext") && !afterContext)
      ) throw new TypeError("invalid legacy history context");
      const rebuiltWithoutSize = {
        id: candidate.id,
        label: candidate.label,
        timestamp: candidate.timestamp,
        affectedSegmentIds: [...candidate.affectedSegmentIds] as string[],
        schemaVersion: 1 as const,
        forward: candidate.forward as TimelinePatch[],
        inverse: candidate.inverse as TimelinePatch[],
        ...(beforeContext ? { beforeContext } : {}),
        ...(afterContext ? { afterContext } : {}),
        ...(typeof candidate.mergeKey === "string" ? { mergeKey: candidate.mergeKey } : {}),
      };
      if (timelineSerializedBytes(rebuiltWithoutSize) !== candidate.byteSize) {
        throw new TypeError("invalid legacy history entry size");
      }
      return { ...rebuiltWithoutSize, byteSize: candidate.byteSize as number };
    });
    if (
      new Set(entries.map((entry) => entry.id)).size !== entries.length ||
      entries.some((entry) => legacyEntryOrdinal(entry.id)! >= (payload.nextEntryId as number))
    ) return null;
    const startIndex = payload.startIndex as number;
    const endIndex = startIndex + entries.length;
    if (!Number.isSafeInteger(endIndex)) return null;
    const expectedCheckpointPositions: number[] = [];
    if (entries.length > 0) {
      expectedCheckpointPositions.push(startIndex);
      let position = Math.ceil((startIndex + 1) / 20) * 20;
      for (; position <= endIndex; position += 20) expectedCheckpointPositions.push(position);
    }
    if (payload.checkpoints.length !== expectedCheckpointPositions.length) return null;
    const checkpoints = payload.checkpoints.map((candidate, index) => {
      if (
        !hasExactKeys(candidate, ["position", "project", "byteSize"]) ||
        candidate.position !== expectedCheckpointPositions[index] ||
        !Number.isSafeInteger(candidate.byteSize) || (candidate.byteSize as number) <= 0
      ) throw new TypeError("invalid legacy checkpoint");
      const project = exactLegacyProject(candidate.project);
      if (!project || timelineSerializedBytes({
        position: candidate.position,
        project,
      }) !== candidate.byteSize) throw new TypeError("invalid legacy checkpoint size");
      return {
        position: candidate.position as number,
        project,
        byteSize: candidate.byteSize as number,
      };
    });
    const totalBytes = entries.reduce((total, entry) => total + entry.byteSize, 0) +
      checkpoints.reduce((total, checkpoint) => total + checkpoint.byteSize, 0);
    if (totalBytes !== payload.totalBytes) return null;
    if (!entries.length) {
      if (payload.head !== null || checkpoints.length || payload.coalescing !== null) return null;
      const empty = createTimelineHistory(payload.capacity as number, payload.byteBudget as number);
      const envelope = serializeTimelineHistory(empty);
      return { history: empty, envelope };
    }
    const initial = checkpoints[0]?.project;
    if (!initial) return null;
    const legacyStates: LegacyTimelineProjectV4[] = [initial];
    let cursorState: LegacyTimelineProjectV4 | null = payload.cursor === 0 ? initial : null;
    const checkpointByPosition = new Map(checkpoints.map((checkpoint) =>
      [checkpoint.position, checkpoint.project] as const));
    for (let index = 0; index < entries.length; index += 1) {
      const before = legacyStates[index];
      const applied = applyTimelinePatches(
        before as unknown as TimelineProject,
        entries[index].forward,
      ) as unknown;
      const after = exactLegacyProject(applied);
      if (!after || timelineValuesEqual(after, before)) return null;
      const canonicalPatches = createTimelinePatchPair(
        before as unknown as TimelineProject,
        after as unknown as TimelineProject,
      );
      if (
        !timelineValuesEqual(entries[index].forward, canonicalPatches.forward) ||
        !timelineValuesEqual(entries[index].inverse, canonicalPatches.inverse) ||
        !timelineValuesEqual(entries[index].affectedSegmentIds, legacyAffectedSegmentIds(before, after))
      ) return null;
      const restored = applyTimelinePatches(
        after as unknown as TimelineProject,
        entries[index].inverse,
      );
      if (!timelineValuesEqual(restored, before)) return null;
      legacyStates.push(after);
      const absolutePosition = startIndex + index + 1;
      const checkpoint = checkpointByPosition.get(absolutePosition);
      if (checkpoint && !timelineValuesEqual(checkpoint, after)) return null;
      if (index + 1 === payload.cursor) cursorState = after;
    }
    const rawHead = exactLegacyProject(payload.head);
    if (!cursorState || !rawHead || !timelineValuesEqual(cursorState, rawHead)) return null;
    let legacyCoalescing: TimelineHistoryState["coalescing"] = null;
    if (payload.coalescing !== null) {
      if (
        !hasExactKeys(payload.coalescing, ["mergeKey", "lastRecordedAt"]) ||
        typeof payload.coalescing.mergeKey !== "string" ||
        payload.coalescing.mergeKey.length < 1 || payload.coalescing.mergeKey.length > 512 ||
        typeof payload.coalescing.lastRecordedAt !== "number" ||
        !Number.isFinite(payload.coalescing.lastRecordedAt) ||
        payload.future.length > 0 ||
        entries[(payload.cursor as number) - 1]?.mergeKey !== payload.coalescing.mergeKey ||
        payload.coalescing.lastRecordedAt < entries[(payload.cursor as number) - 1].timestamp
      ) return null;
      legacyCoalescing = {
        mergeKey: payload.coalescing.mergeKey,
        lastRecordedAt: payload.coalescing.lastRecordedAt,
      };
    }
    const states = legacyStates.map((project) =>
      migrateLegacyTimelineProjectToV5(project, context));
    if (states.some((project) => project === null)) return null;
    const migratedStates = states as TimelineProject[];
    let history: TimelineHistoryState = {
      ...createTimelineHistory(payload.capacity as number, payload.byteBudget as number),
      startIndex,
    };
    for (let index = 0; index < entries.length; index += 1) {
      // These are already-committed transitions. Seal before every record so
      // two old entries are never newly coalesced merely because their clocks
      // and merge keys happen to be adjacent.
      history = sealTimelineHistoryCoalescing(history);
      history = recordTimelineHistory(history, {
        label: entries[index].label,
        before: migratedStates[index],
        after: migratedStates[index + 1],
        beforeContext: entries[index].beforeContext,
        afterContext: entries[index].afterContext,
        mergeKey: entries[index].mergeKey,
        now: entries[index].timestamp,
      });
    }
    if (history.nextEntryId !== payload.nextEntryId) return null;
    for (let count = 0; count < payload.future.length; count += 1) {
      const undone = undoTimelineHistory(history);
      if (!undone) return null;
      history = undone.history;
    }
    history = legacyCoalescing && payload.future.length === 0
      ? { ...history, coalescing: legacyCoalescing }
      : sealTimelineHistoryCoalescing(history);
    if (!history.head || !timelineValuesEqual(history.head, migratedStates[payload.cursor as number])) return null;
    // Audit the entire rebuilt branch in both directions, including the
    // original cursor with future entries. This proves iterative Undo/Redo is
    // semantically equivalent to the migrated legacy full-state sequence.
    let audit = history;
    let auditCursor = payload.cursor as number;
    while (auditCursor > 0) {
      const undone = undoTimelineHistory(audit);
      if (!undone || !timelineValuesEqual(
        undone.snapshot.project,
        migratedStates[auditCursor - 1],
      )) return null;
      audit = undone.history;
      auditCursor -= 1;
    }
    while (auditCursor < migratedStates.length - 1) {
      const redone = redoTimelineHistory(audit);
      if (!redone || !timelineValuesEqual(
        redone.snapshot.project,
        migratedStates[auditCursor + 1],
      )) return null;
      audit = redone.history;
      auditCursor += 1;
    }
    const envelope = serializeTimelineHistory(history);
    const verified = deserializeTimelineHistory(envelope, { expectedHead: history.head });
    return verified ? { history: verified, envelope } : null;
  } catch {
    return null;
  }
}

/** Leaves source bytes untouched until the caller has durably ACKed recovery. */
export function legacyTimelineWalCandidates(
  database: TimelineWalDatabaseIdentity,
  projectId: string,
): Array<{ storageKey: string; raw: string; wal: LegacyTimelineWal }> {
  const result: Array<{ storageKey: string; raw: string; wal: LegacyTimelineWal }> = [];
  try {
    for (let index = 0; index < localStorage.length; index += 1) {
      const storageKey = localStorage.key(index);
      if (!storageKey?.startsWith(LEGACY_V7_TIMELINE_WAL_STORAGE_PREFIX)) continue;
      const raw = localStorage.getItem(storageKey);
      const wal = raw ? parseLegacyTimelineWalRaw(raw) : null;
      if (
        raw && wal && wal.active_database_path === database.active_database_path &&
        wal.project_id === projectId
      ) result.push({ storageKey, raw, wal });
    }
  } catch {
    return [];
  }
  return result.sort((left, right) => right.wal.written_at_ms - left.wal.written_at_ms);
}

export function clearLegacyTimelineWalCandidate(candidate: { storageKey: string; raw: string }): boolean {
  try {
    if (localStorage.getItem(candidate.storageKey) !== candidate.raw) return false;
    localStorage.removeItem(candidate.storageKey);
    return localStorage.getItem(candidate.storageKey) === null;
  } catch {
    return false;
  }
}
