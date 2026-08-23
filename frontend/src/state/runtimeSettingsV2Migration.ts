import {
  migrateRuntimeSettingsV2ToV3,
  parseRuntimeSettingsV2,
  sameLoraLoaderBinding,
  type LoraLoaderOverrideRecord,
  type RuntimeSettings,
  type RuntimeSettingsAuthority,
  type RuntimeSettingsV2,
} from "../api/types";
import { isStoragePath } from "../domain/storagePath";
import {
  mergeExactLoraLoaderMappings,
  runtimeSettingsQuarantineKeyMatchesRaw,
  runtimeSettingsRecoveryRawDigest,
} from "./runtimeSettingsRecoveryEvidence";

export const LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY =
  "directordeck:v3:runtime-settings-pending";
export const RUNTIME_SETTINGS_V2_WAL_QUARANTINE_PREFIX =
  "directordeck:v4:runtime-settings-v2-quarantine:";
export const RUNTIME_SETTINGS_V3_WAL_STORAGE_KEY =
  "directordeck:v4:runtime-settings-pending";

const PENDING_FORMAT = "director-pending-runtime-settings";

export interface RuntimeSettingsV2WalEnvelope {
  format: typeof PENDING_FORMAT;
  version: 3;
  owner_id: string;
  pending: true;
  active_database_path: string;
  written_at_ms: number;
  settings: RuntimeSettingsV2;
}

export interface RuntimeSettingsV2WalCandidate {
  kind: "recoverable";
  storage_key: string;
  raw: string;
  raw_digest: string;
  envelope: RuntimeSettingsV2WalEnvelope;
  ownership: RuntimeSettingsV2WalOwnership;
}

export type RuntimeSettingsV2WalDatabaseScope = "current" | "other" | "unknown";

export interface RuntimeSettingsV2WalOwnership {
  owner_id: string | null;
  database_scope: RuntimeSettingsV2WalDatabaseScope;
  written_at_ms: number | null;
}

export interface CorruptRuntimeSettingsV2WalEvidence {
  kind: "corrupt";
  storage_key: string;
  raw: string;
  raw_digest: string;
  ownership: RuntimeSettingsV2WalOwnership;
}

export type RuntimeSettingsV2WalEvidence =
  | RuntimeSettingsV2WalCandidate
  | CorruptRuntimeSettingsV2WalEvidence;

export type RuntimeSettingsV2RecoveryChoice =
  | "merge-exact-mappings"
  | "apply-runtime-and-merge-mappings";

export interface RuntimeSettingsV2RecoveryPlan {
  choice: RuntimeSettingsV2RecoveryChoice;
  document: RuntimeSettings;
  required_mapping_union: LoraLoaderOverrideRecord[];
  required_runtime_fields: Omit<RuntimeSettings, "schema_version" | "lora_loader_overrides"> | null;
}

export interface RuntimeSettingsV2RecoveryTransport {
  read: () => Promise<RuntimeSettingsAuthority>;
  write: (
    document: RuntimeSettings,
    expectedAuthorityToken: string,
  ) => Promise<RuntimeSettingsAuthority>;
}

export interface RuntimeSettingsV2RecoveryResult {
  authority: RuntimeSettingsAuthority;
  acknowledgement: "cas-ack" | "lost-ack-proven";
}

export class RuntimeSettingsV2RecoveryConflictError extends Error {
  constructor(message = "旧运行设置与当前 RuntimeSettingsV3 权威冲突；原始隔离记录仍保留") {
    super(message);
    this.name = "RuntimeSettingsV2RecoveryConflictError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length &&
    actual.every((key, index) => key === wanted[index]);
}

function validOwner(value: unknown): value is string {
  return typeof value === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(value);
}

function quarantineBaseKey(raw: string): string {
  return `${RUNTIME_SETTINGS_V2_WAL_QUARANTINE_PREFIX}${runtimeSettingsRecoveryRawDigest(raw)}`;
}

function quarantineKeyMatchesRaw(storageKey: string, raw: string): boolean {
  return runtimeSettingsQuarantineKeyMatchesRaw(
    storageKey,
    RUNTIME_SETTINGS_V2_WAL_QUARANTINE_PREFIX,
    raw,
    true,
  );
}

export function parseRuntimeSettingsV2WalRaw(
  raw: string,
): RuntimeSettingsV2WalEnvelope | null {
  try {
    const value: unknown = JSON.parse(raw);
    if (
      !isRecord(value) ||
      !hasExactKeys(value, [
        "format", "version", "owner_id", "pending", "active_database_path",
        "written_at_ms", "settings",
      ]) ||
      value.format !== PENDING_FORMAT ||
      value.version !== 3 ||
      !validOwner(value.owner_id) ||
      value.pending !== true ||
      typeof value.active_database_path !== "string" ||
      !isStoragePath(value.active_database_path) ||
      !Number.isSafeInteger(value.written_at_ms) ||
      (value.written_at_ms as number) <= 0
    ) return null;
    const settings = parseRuntimeSettingsV2(value.settings);
    if (!settings) return null;
    return {
      format: PENDING_FORMAT,
      version: 3,
      owner_id: value.owner_id,
      pending: true,
      active_database_path: value.active_database_path,
      written_at_ms: value.written_at_ms as number,
      settings,
    };
  } catch {
    return null;
  }
}

function inspectRuntimeSettingsV2WalOwnership(
  raw: string,
  activeDatabasePath: string,
): RuntimeSettingsV2WalOwnership {
  let ownerId: string | null = null;
  let databaseScope: RuntimeSettingsV2WalDatabaseScope = "unknown";
  let writtenAtMs: number | null = null;
  try {
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value)) return {
      owner_id: ownerId,
      database_scope: databaseScope,
      written_at_ms: writtenAtMs,
    };
    if (validOwner(value.owner_id)) ownerId = value.owner_id;
    if (Number.isSafeInteger(value.written_at_ms) && (value.written_at_ms as number) > 0) {
      writtenAtMs = value.written_at_ms as number;
    }
    if (typeof value.active_database_path === "string" &&
      isStoragePath(value.active_database_path) && isStoragePath(activeDatabasePath)) {
      databaseScope = value.active_database_path === activeDatabasePath ? "current" : "other";
    }
  } catch {
    // Only the allowlisted ownership fields above may cross the UI boundary.
  }
  return {
    owner_id: ownerId,
    database_scope: databaseScope,
    written_at_ms: writtenAtMs,
  };
}

/** Copies exact bytes before compare-deleting the obsolete singleton key. */
function quarantineRuntimeSettingsV2PendingWal(): void {
  try {
    const raw = localStorage.getItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY);
    if (raw === null) return;
    const baseKey = quarantineBaseKey(raw);
    for (let suffix = 0; suffix < 32; suffix += 1) {
      const storageKey = suffix === 0 ? baseKey : `${baseKey}:${suffix}`;
      const existing = localStorage.getItem(storageKey);
      if (existing !== null && existing !== raw) continue;
      if (existing === null) localStorage.setItem(storageKey, raw);
      if (
        localStorage.getItem(storageKey) === raw &&
        localStorage.getItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY) === raw
      ) localStorage.removeItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY);
      return;
    }
  } catch {
    // If exact quarantine cannot be proven, the singleton remains untouched.
  }
}

/**
 * Hydration is evidence-only: every quarantined record remains visible. Only
 * an exact, strictly parsed, same-database record becomes recoverable; all
 * others are tagged corrupt and can only be exported or compare-deleted.
 */
export function listRuntimeSettingsV2WalEvidence(
  activeDatabasePath: string,
): RuntimeSettingsV2WalEvidence[] {
  quarantineRuntimeSettingsV2PendingWal();
  const evidence: RuntimeSettingsV2WalEvidence[] = [];
  try {
    for (let index = 0; index < localStorage.length; index += 1) {
      const storageKey = localStorage.key(index);
      if (!storageKey?.startsWith(RUNTIME_SETTINGS_V2_WAL_QUARANTINE_PREFIX)) continue;
      const raw = localStorage.getItem(storageKey);
      if (raw === null) continue;
      const rawDigest = runtimeSettingsRecoveryRawDigest(raw);
      const ownership = inspectRuntimeSettingsV2WalOwnership(raw, activeDatabasePath);
      const envelope = parseRuntimeSettingsV2WalRaw(raw);
      if (
        !quarantineKeyMatchesRaw(storageKey, raw) ||
        !envelope ||
        !isStoragePath(activeDatabasePath) ||
        envelope.active_database_path !== activeDatabasePath
      ) {
        evidence.push({
          kind: "corrupt",
          storage_key: storageKey,
          raw,
          raw_digest: rawDigest,
          ownership,
        });
        continue;
      }
      evidence.push({
        kind: "recoverable",
        storage_key: storageKey,
        raw,
        raw_digest: rawDigest,
        envelope,
        ownership,
      });
    }
  } catch {
    return [];
  }
  return evidence.sort((left, right) =>
    (right.ownership.written_at_ms ?? 0) - (left.ownership.written_at_ms ?? 0) ||
    left.storage_key.localeCompare(right.storage_key));
}

/** Compatibility projection for code that intentionally handles only safe CAS candidates. */
export function listRuntimeSettingsV2WalCandidates(
  activeDatabasePath: string,
): RuntimeSettingsV2WalCandidate[] {
  return listRuntimeSettingsV2WalEvidence(activeDatabasePath).filter(
    (entry): entry is RuntimeSettingsV2WalCandidate => entry.kind === "recoverable",
  );
}

/** Compare-and-delete only this exact candidate; no other pending WAL is touched. */
export function clearRuntimeSettingsV2WalCandidate(
  candidate: Pick<RuntimeSettingsV2WalEvidence, "storage_key" | "raw">,
): boolean {
  try {
    if (localStorage.getItem(candidate.storage_key) !== candidate.raw) return false;
    localStorage.removeItem(candidate.storage_key);
    return localStorage.getItem(candidate.storage_key) === null;
  } catch {
    return false;
  }
}

function runtimeSettingsV2WalCandidateIsCurrent(
  candidate: Pick<RuntimeSettingsV2WalCandidate, "storage_key" | "raw">,
): boolean {
  try {
    return localStorage.getItem(candidate.storage_key) === candidate.raw;
  } catch {
    return false;
  }
}

/** Any current-format pending bytes belong to an independent V3 writer. */
export function hasIndependentRuntimeSettingsV3PendingWal(): boolean {
  try {
    return localStorage.getItem(RUNTIME_SETTINGS_V3_WAL_STORAGE_KEY) !== null;
  } catch {
    // An unreadable pending authority is still an unresolved conflict.
    return true;
  }
}

function assertNoIndependentRuntimeSettingsV3PendingWal(): void {
  if (hasIndependentRuntimeSettingsV3PendingWal()) {
    throw new RuntimeSettingsV2RecoveryConflictError(
      "检测到独立 RuntimeSettingsV3 待同步记录；V2 原始隔离记录仍保留",
    );
  }
}

export interface RuntimeSettingsV2WalEvidenceExport {
  filename: string;
  mimeType: "application/octet-stream";
  contents: string;
}

export function buildRuntimeSettingsV2WalEvidenceExport(
  evidence: Pick<RuntimeSettingsV2WalEvidence, "raw" | "raw_digest">,
): RuntimeSettingsV2WalEvidenceExport {
  const safeDigest = /^fnv1a64-[0-9a-f]{16}$/.test(evidence.raw_digest)
    ? evidence.raw_digest
    : runtimeSettingsRecoveryRawDigest(evidence.raw);
  return {
    filename: `directordeck-runtime-settings-v2-evidence-${safeDigest}.bin`,
    mimeType: "application/octet-stream",
    contents: evidence.raw,
  };
}

export function downloadRuntimeSettingsV2WalEvidence(
  evidence: Pick<RuntimeSettingsV2WalEvidence, "raw" | "raw_digest">,
): void {
  const download = buildRuntimeSettingsV2WalEvidenceExport(evidence);
  const blob = new Blob([download.contents], { type: download.mimeType });
  const createObjectUrl = URL.createObjectURL?.bind(URL);
  const objectUrl = createObjectUrl ? createObjectUrl(blob) : null;
  const url = objectUrl ??
    `data:${download.mimeType};charset=utf-8,${encodeURIComponent(download.contents)}`;
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = download.filename;
  anchor.rel = "noreferrer";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  if (objectUrl !== null) window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
}

function sameMapping(
  left: LoraLoaderOverrideRecord,
  right: LoraLoaderOverrideRecord,
): boolean {
  return sameLoraLoaderBinding(left, right) &&
    left.adapter_id === right.adapter_id &&
    JSON.stringify(left.options) === JSON.stringify(right.options);
}

function runtimeOnlyFields(
  settings: RuntimeSettings,
): Omit<RuntimeSettings, "schema_version" | "lora_loader_overrides"> {
  return {
    client_id: settings.client_id,
    memory_policy: settings.memory_policy,
    raylight_residency_policy: settings.raylight_residency_policy,
    multi_gpu_enabled: settings.multi_gpu_enabled,
    placement: structuredClone(settings.placement),
  };
}

export function buildRuntimeSettingsV2RecoveryPlan(
  candidate: RuntimeSettingsV2WalCandidate,
  choice: RuntimeSettingsV2RecoveryChoice,
  current: RuntimeSettings,
): RuntimeSettingsV2RecoveryPlan | null {
  const parsed = parseRuntimeSettingsV2WalRaw(candidate.raw);
  if (
    !parsed ||
    candidate.raw_digest !== runtimeSettingsRecoveryRawDigest(candidate.raw) ||
    !quarantineKeyMatchesRaw(candidate.storage_key, candidate.raw) ||
    parsed.active_database_path !== candidate.envelope.active_database_path
  ) return null;
  let migrated: RuntimeSettings;
  try {
    migrated = migrateRuntimeSettingsV2ToV3(parsed.settings);
  } catch {
    return null;
  }
  const mappingUnion = mergeExactLoraLoaderMappings(
    current.lora_loader_overrides,
    migrated.lora_loader_overrides,
  );
  if (!mappingUnion) return null;
  const applyRuntime = choice === "apply-runtime-and-merge-mappings";
  const document: RuntimeSettings = {
    ...(applyRuntime ? migrated : structuredClone(current)),
    schema_version: 3,
    lora_loader_overrides: mappingUnion,
  };
  return {
    choice,
    document,
    required_mapping_union: mappingUnion,
    required_runtime_fields: applyRuntime ? runtimeOnlyFields(migrated) : null,
  };
}

export function runtimeSettingsV2RecoveryIsProven(
  settings: RuntimeSettings,
  plan: RuntimeSettingsV2RecoveryPlan,
): boolean {
  if (!plan.required_mapping_union.every((required) =>
    settings.lora_loader_overrides.some((actual) => sameMapping(actual, required)))) {
    return false;
  }
  return plan.required_runtime_fields === null ||
    JSON.stringify(runtimeOnlyFields(settings)) === JSON.stringify(plan.required_runtime_fields);
}

function isAuthorityConflict(reason: unknown): boolean {
  return typeof reason === "object" && reason !== null &&
    "status" in reason && (reason as { status?: unknown }).status === 409;
}

/**
 * Reads the latest V3 authority, performs at most one whole-document CAS, and
 * clears evidence only for an explicit ACK or a provable lost ACK.
 */
export async function recoverRuntimeSettingsV2WalCandidate(
  candidate: RuntimeSettingsV2WalCandidate,
  choice: RuntimeSettingsV2RecoveryChoice,
  transport: RuntimeSettingsV2RecoveryTransport,
): Promise<RuntimeSettingsV2RecoveryResult> {
  // Re-read localStorage synchronously at the explicit click boundary. The
  // hydration snapshot is advisory and another tab may have written since.
  assertNoIndependentRuntimeSettingsV3PendingWal();
  if (!runtimeSettingsV2WalCandidateIsCurrent(candidate)) {
    throw new RuntimeSettingsV2RecoveryConflictError(
      "RuntimeSettingsV2 隔离原始字节已变化；未执行任何权威写入",
    );
  }
  const initial = await transport.read();
  const plan = buildRuntimeSettingsV2RecoveryPlan(candidate, choice, initial.settings);
  if (!plan) throw new RuntimeSettingsV2RecoveryConflictError();
  if (!runtimeSettingsV2WalCandidateIsCurrent(candidate)) {
    throw new RuntimeSettingsV2RecoveryConflictError(
      "RuntimeSettingsV2 隔离原始字节已变化；未执行任何权威写入",
    );
  }
  assertNoIndependentRuntimeSettingsV3PendingWal();

  let acknowledged: RuntimeSettingsAuthority;
  try {
    acknowledged = await transport.write(plan.document, initial.authority_token);
  } catch (reason) {
    if (isAuthorityConflict(reason)) throw reason;
    let latest: RuntimeSettingsAuthority;
    try {
      latest = await transport.read();
    } catch {
      throw reason;
    }
    if (!runtimeSettingsV2RecoveryIsProven(latest.settings, plan)) throw reason;
    // A V3 WAL created while the CAS was in flight owns the next decision.
    // The server may already contain our change, but the old evidence must not
    // be cleared until the independent writer is resolved.
    assertNoIndependentRuntimeSettingsV3PendingWal();
    if (!clearRuntimeSettingsV2WalCandidate(candidate)) {
      throw new Error("旧运行设置已落地，但隔离原始字节已变化，未执行清理");
    }
    return { authority: latest, acknowledgement: "lost-ack-proven" };
  }

  if (!runtimeSettingsV2RecoveryIsProven(acknowledged.settings, plan)) {
    throw new Error("RuntimeSettingsV3 CAS 响应未证明完整恢复结果");
  }
  assertNoIndependentRuntimeSettingsV3PendingWal();
  if (!clearRuntimeSettingsV2WalCandidate(candidate)) {
    throw new Error("RuntimeSettingsV3 CAS 已确认，但隔离原始字节已变化，未执行清理");
  }
  return { authority: acknowledged, acknowledgement: "cas-ack" };
}

export function discardRuntimeSettingsV2WalCandidate(
  candidate: Pick<RuntimeSettingsV2WalEvidence, "storage_key" | "raw">,
): boolean {
  return clearRuntimeSettingsV2WalCandidate(candidate);
}
