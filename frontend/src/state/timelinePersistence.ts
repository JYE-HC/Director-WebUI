import {
  migrateLegacyTimelineProjectToV5,
  migrateTimelineFeatureBundle4To5,
  normalizeLegacyTimelineProject,
  normalizeTimelineProject,
  type LegacyTimelineProjectV4,
  type TimelineProject,
} from "../domain/timelineProject";
import { isStoragePath } from "../domain/storagePath";
import {
  deserializeTimelineHistory,
  serializeTimelineHistory,
  timelineProjectsEqual,
  type SerializedTimelineHistoryEnvelope,
  type TimelineHistoryState,
} from "./timelineHistory";
import {
  legacyClientDocumentDigest,
  migrateLegacyTimelineHistoryEnvelope,
  type ProjectMigrationReceipt,
} from "./timelineV5Migration";

const TIMELINE_HISTORY_DATABASE = "directordeck-timeline-history";
const TIMELINE_HISTORY_DATABASE_VERSION = 1;
const TIMELINE_HISTORY_STORE = "journals";
const TIMELINE_HISTORY_JOURNAL_FORMAT = "director-timeline-history-journal";
const LEGACY_TIMELINE_HISTORY_JOURNAL_VERSION = 1;
const TIMELINE_HISTORY_JOURNAL_VERSION = 2;
const TIMELINE_HISTORY_JOURNAL_KEY_PREFIX = "directordeck:v2:timeline-history:";
const OWNER_ID_PATTERN = /^[A-Za-z0-9._:-]{1,128}$/;
const WRITE_TOKEN_PATTERN = /^[A-Za-z0-9._:-]{1,128}$/;

export interface TimelinePersistenceProjectScope {
  databasePath: string;
  projectId: string;
}

/** A page-session branch. Every durable mutation is isolated by ownerId. */
export interface TimelinePersistenceScope extends TimelinePersistenceProjectScope {
  ownerId: string;
}

export interface TimelinePersistenceAuthority {
  document: TimelineProject;
  revision: number;
}

/**
 * An opaque compare-and-delete capability for one exact stored journal version.
 * Callers must not construct or alter this value.
 */
export interface TimelineHistoryJournalVersionToken {
  readonly key: string;
  readonly version: string;
}

interface TimelineHistoryJournalRecordV2 {
  key: string;
  format: typeof TIMELINE_HISTORY_JOURNAL_FORMAT;
  version: typeof TIMELINE_HISTORY_JOURNAL_VERSION;
  scope: TimelinePersistenceScope;
  writeToken: string;
  confirmedRevision: number;
  confirmedDocumentHash: string;
  confirmedDocument: TimelineProject;
  headDocumentHash: string;
  history: SerializedTimelineHistoryEnvelope;
  updatedAtMs: number;
}

interface LegacyV4TimelineHistoryJournalRecordV2 {
  key: string;
  format: typeof TIMELINE_HISTORY_JOURNAL_FORMAT;
  version: typeof TIMELINE_HISTORY_JOURNAL_VERSION;
  scope: TimelinePersistenceScope;
  writeToken: string;
  confirmedRevision: number;
  confirmedDocumentHash: string;
  confirmedDocument: LegacyTimelineProjectV4;
  headDocumentHash: string;
  history: unknown;
  updatedAtMs: number;
}

interface LegacyTimelineHistoryJournalRecordV1 {
  key: string;
  format: typeof TIMELINE_HISTORY_JOURNAL_FORMAT;
  version: typeof LEGACY_TIMELINE_HISTORY_JOURNAL_VERSION;
  scope: TimelinePersistenceProjectScope;
  confirmedRevision: number;
  confirmedDocumentHash: string;
  confirmedDocument: TimelineProject;
  headDocumentHash: string;
  history: SerializedTimelineHistoryEnvelope;
  updatedAtMs: number;
}

interface DecodedTimelineHistoryJournal {
  confirmedRevision: number;
  confirmedDocument: TimelineProject;
  history: TimelineHistoryState;
  project: TimelineProject;
  /** Legacy receipt recovery accepts only its exact destination revision. */
  receiptAuthorityMatched?: boolean;
}

interface TimelineHistoryJournalLoadMetadata {
  ownerId: string;
  token: TimelineHistoryJournalVersionToken;
  updatedAtMs: number;
}

export type TimelineHistoryJournalLoadResult =
  | { status: "none" | "unavailable" }
  | {
      status: "corrupt";
      token: TimelineHistoryJournalVersionToken | null;
      updatedAtMs: number | null;
    }
  | (TimelineHistoryJournalLoadMetadata & {
      status: "restored" | "acknowledged";
      history: TimelineHistoryState;
      project: TimelineProject;
      confirmedRevision: number;
      confirmedDocument: TimelineProject;
    })
  | (TimelineHistoryJournalLoadMetadata & {
      status: "conflict";
      confirmedRevision: number;
      confirmedDocument: TimelineProject;
      localProject: TimelineProject;
      history: TimelineHistoryState;
      project: TimelineProject;
    });

export type TimelineHistoryJournalBranchOwnership = "owned" | "foreign" | "legacy";

interface TimelineHistoryJournalBranchMetadata {
  ownership: TimelineHistoryJournalBranchOwnership;
  ownerId: string | null;
  token: TimelineHistoryJournalVersionToken | null;
  updatedAtMs: number | null;
}

export type TimelineHistoryJournalBranchEvidence =
  | (TimelineHistoryJournalBranchMetadata & { status: "corrupt" })
  | (TimelineHistoryJournalBranchMetadata & {
      status: "restored" | "acknowledged" | "conflict";
      confirmedRevision: number;
      confirmedDocument: TimelineProject;
      history: TimelineHistoryState;
      project: TimelineProject;
    });

export type TimelineHistoryJournalBranchListResult =
  | { status: "unavailable" | "corrupt" }
  | {
      status: "available";
      /** All matching branches, newest first. No entry is deduplicated or deleted. */
      branches: TimelineHistoryJournalBranchEvidence[];
      owned: TimelineHistoryJournalBranchEvidence[];
      foreign: TimelineHistoryJournalBranchEvidence[];
      legacy: TimelineHistoryJournalBranchEvidence[];
      corrupt: TimelineHistoryJournalBranchEvidence[];
    };

function isRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(value: unknown, expected: readonly string[]): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const keys = Object.keys(value);
  return keys.length === expected.length && expected.every((key) =>
    Object.prototype.hasOwnProperty.call(value, key));
}

function sameProjectScope(
  left: TimelinePersistenceProjectScope,
  right: TimelinePersistenceProjectScope,
): boolean {
  return left.databasePath === right.databasePath &&
    left.projectId === right.projectId;
}

export function timelineHistoryJournalKey(scope: TimelinePersistenceScope): string {
  const digest = timelineHistoryProjectScopeDigest(scope);
  return `${TIMELINE_HISTORY_JOURNAL_KEY_PREFIX}${digest}:${encodeURIComponent(scope.ownerId)}`;
}

function timelineHistoryProjectScopeDigest(scope: TimelinePersistenceProjectScope): string {
  return sha256Hex(JSON.stringify([
    scope.databasePath,
    scope.projectId,
  ]));
}

export function legacyTimelineHistoryJournalKey(
  scope: TimelinePersistenceProjectScope,
): string {
  return JSON.stringify([scope.databasePath, scope.projectId]);
}

function validProjectScope(scope: TimelinePersistenceProjectScope): boolean {
  return isStoragePath(scope.databasePath) &&
    /^[A-Za-z0-9._:-]{1,128}$/.test(scope.projectId);
}

function validScope(scope: TimelinePersistenceScope): boolean {
  return validProjectScope(scope) && typeof scope.ownerId === "string" &&
    OWNER_ID_PATTERN.test(scope.ownerId);
}

interface ParsedV2JournalKey {
  scopeDigest: string;
  ownerId: string;
}

function parseV2JournalKey(value: unknown): ParsedV2JournalKey | null {
  if (typeof value !== "string" || !value.startsWith(TIMELINE_HISTORY_JOURNAL_KEY_PREFIX)) {
    return null;
  }
  const suffix = value.slice(TIMELINE_HISTORY_JOURNAL_KEY_PREFIX.length);
  const separator = suffix.indexOf(":");
  if (separator !== 64 || suffix.indexOf(":", separator + 1) !== -1) return null;
  const scopeDigest = suffix.slice(0, separator);
  let ownerId: string;
  try {
    ownerId = decodeURIComponent(suffix.slice(separator + 1));
  } catch {
    return null;
  }
  if (!/^[0-9a-f]{64}$/.test(scopeDigest) || !OWNER_ID_PATTERN.test(ownerId)) return null;
  return { scopeDigest, ownerId };
}

function parseLegacyJournalKey(value: unknown): TimelinePersistenceProjectScope | null {
  if (typeof value !== "string") return null;
  let parts: unknown;
  try {
    parts = JSON.parse(value);
  } catch {
    return null;
  }
  if (!Array.isArray(parts) || parts.length !== 2) return null;
  const [databasePath, projectId] = parts;
  const projectScope = { databasePath, projectId };
  if (
    typeof databasePath !== "string" ||
    typeof projectId !== "string" ||
    !validProjectScope(projectScope)
  ) return null;
  return projectScope;
}

const SHA256_ROUND_CONSTANTS = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
  0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
  0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
  0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function rotateRight(value: number, bits: number): number {
  return (value >>> bits) | (value << (32 - bits));
}

/** Synchronous SHA-256 keeps IndexedDB keys opaque without relying on Web Crypto availability. */
function sha256Hex(text: string): string {
  const source = new TextEncoder().encode(text);
  const paddedLength = Math.ceil((source.length + 9) / 64) * 64;
  const bytes = new Uint8Array(paddedLength);
  bytes.set(source);
  bytes[source.length] = 0x80;
  const bitLength = source.length * 8;
  const view = new DataView(bytes.buffer);
  view.setUint32(paddedLength - 8, Math.floor(bitLength / 0x1_0000_0000), false);
  view.setUint32(paddedLength - 4, bitLength >>> 0, false);

  const state = new Uint32Array([
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19,
  ]);
  const words = new Uint32Array(64);
  for (let offset = 0; offset < bytes.length; offset += 64) {
    for (let index = 0; index < 16; index += 1) {
      words[index] = view.getUint32(offset + index * 4, false);
    }
    for (let index = 16; index < 64; index += 1) {
      const left = words[index - 15];
      const right = words[index - 2];
      const sigma0 = rotateRight(left, 7) ^ rotateRight(left, 18) ^ (left >>> 3);
      const sigma1 = rotateRight(right, 17) ^ rotateRight(right, 19) ^ (right >>> 10);
      words[index] = (words[index - 16] + sigma0 + words[index - 7] + sigma1) >>> 0;
    }

    let [a, b, c, d, e, f, g, h] = state;
    for (let index = 0; index < 64; index += 1) {
      const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
      const choose = (e & f) ^ (~e & g);
      const temporary1 = (h + sum1 + choose + SHA256_ROUND_CONSTANTS[index] + words[index]) >>> 0;
      const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
      const majority = (a & b) ^ (a & c) ^ (b & c);
      const temporary2 = (sum0 + majority) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temporary1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temporary1 + temporary2) >>> 0;
    }
    state[0] = (state[0] + a) >>> 0;
    state[1] = (state[1] + b) >>> 0;
    state[2] = (state[2] + c) >>> 0;
    state[3] = (state[3] + d) >>> 0;
    state[4] = (state[4] + e) >>> 0;
    state[5] = (state[5] + f) >>> 0;
    state[6] = (state[6] + g) >>> 0;
    state[7] = (state[7] + h) >>> 0;
  }
  return [...state].map((value) => value.toString(16).padStart(8, "0")).join("");
}

function canonicalDigestValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalDigestValue);
  if (!isRecord(value)) return value;
  return Object.fromEntries(
    Object.keys(value)
      .sort()
      .map((key) => [key, canonicalDigestValue(value[key])]),
  );
}

/** Hash is an integrity hint; exact normalized document equality remains the authority gate. */
export async function timelineProjectDigest(project: unknown): Promise<string> {
  // Persistence hashes must not change format when WebCrypto is unavailable,
  // temporarily rejects, or becomes available after a reload. The module's
  // synchronous SHA-256 implementation is deterministic in every realm.
  return `sha256-${sha256Hex(JSON.stringify(canonicalDigestValue(project)))}`;
}

function openTimelineHistoryDatabase(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") return Promise.resolve(null);
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(
      TIMELINE_HISTORY_DATABASE,
      TIMELINE_HISTORY_DATABASE_VERSION,
    );
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(TIMELINE_HISTORY_STORE)) {
        database.createObjectStore(TIMELINE_HISTORY_STORE, { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("无法打开时间线历史存储"));
    request.onblocked = () => reject(new Error("时间线历史存储升级被其他页面阻塞"));
  });
}

async function readJournal(key: string): Promise<unknown | undefined> {
  const database = await openTimelineHistoryDatabase();
  if (!database) return undefined;
  try {
    return await new Promise((resolve, reject) => {
      const request = database
        .transaction(TIMELINE_HISTORY_STORE, "readonly")
        .objectStore(TIMELINE_HISTORY_STORE)
        .get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("无法读取时间线历史"));
    });
  } finally {
    database.close();
  }
}

async function readAllJournals(): Promise<unknown[] | null> {
  const database = await openTimelineHistoryDatabase();
  if (!database) return null;
  try {
    return await new Promise((resolve, reject) => {
      const request = database
        .transaction(TIMELINE_HISTORY_STORE, "readonly")
        .objectStore(TIMELINE_HISTORY_STORE)
        .getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("无法枚举时间线历史"));
    });
  } finally {
    database.close();
  }
}

async function writeJournal(record: TimelineHistoryJournalRecordV2): Promise<boolean> {
  const database = await openTimelineHistoryDatabase();
  if (!database) return false;
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(TIMELINE_HISTORY_STORE, "readwrite");
      transaction.objectStore(TIMELINE_HISTORY_STORE).put(record);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(
        transaction.error ?? new Error("无法保存时间线历史"),
      );
      transaction.onabort = () => reject(
        transaction.error ?? new Error("时间线历史保存已中止"),
      );
    });
    return true;
  } finally {
    database.close();
  }
}

function randomWriteToken(): string {
  try {
    if (typeof globalThis.crypto?.randomUUID === "function") {
      return globalThis.crypto.randomUUID();
    }
    if (typeof globalThis.crypto?.getRandomValues === "function") {
      const bytes = new Uint8Array(16);
      globalThis.crypto.getRandomValues(bytes);
      return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
    }
  } catch {
    // The timestamp plus two independent PRNG samples remains page-local unique enough
    // for a version token; owner isolation is the cross-page correctness boundary.
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

function parseExactTimelineProject(value: unknown): TimelineProject | null {
  const normalized = normalizeTimelineProject(value);
  return normalized && timelineProjectsEqual(normalized, value as TimelineProject)
    ? normalized
    : null;
}

export async function saveTimelineHistoryJournal(
  scope: TimelinePersistenceScope,
  confirmed: TimelinePersistenceAuthority,
  history: TimelineHistoryState,
): Promise<TimelineHistoryJournalVersionToken | null> {
  if (
    !validScope(scope) ||
    !Number.isSafeInteger(confirmed.revision) ||
    confirmed.revision < 0
  ) {
    throw new RangeError("Invalid timeline history persistence scope or revision.");
  }
  const confirmedDocument = parseExactTimelineProject(confirmed.document);
  if (!confirmedDocument) throw new TypeError("Invalid confirmed timeline document.");
  // A missing head is not authority to delete whatever version may now occupy
  // this branch. The caller must use an exact token with the delete API.
  if (!history.head) return null;

  const serializedHistory = serializeTimelineHistory(history);
  const [confirmedDocumentHash, headDocumentHash] = await Promise.all([
    timelineProjectDigest(confirmedDocument),
    timelineProjectDigest(history.head),
  ]);
  const writeToken = randomWriteToken();
  const key = timelineHistoryJournalKey(scope);
  const record: TimelineHistoryJournalRecordV2 = {
    key,
    format: TIMELINE_HISTORY_JOURNAL_FORMAT,
    version: TIMELINE_HISTORY_JOURNAL_VERSION,
    scope: structuredClone(scope),
    writeToken,
    confirmedRevision: confirmed.revision,
    confirmedDocumentHash,
    confirmedDocument: structuredClone(confirmedDocument),
    headDocumentHash,
    history: serializedHistory,
    updatedAtMs: Date.now(),
  };
  const token = versionTokenForStoredValue(record, key);
  const stored = await writeJournal(record);
  return stored ? token : null;
}

function parseCommonJournalFields(
  value: Record<string, unknown>,
): {
  confirmedRevision: number;
  confirmedDocumentHash: string;
  confirmedDocument: TimelineProject;
  headDocumentHash: string;
  history: SerializedTimelineHistoryEnvelope;
  updatedAtMs: number;
} | null {
  if (
    !Number.isSafeInteger(value.confirmedRevision) ||
    (value.confirmedRevision as number) < 0 ||
    typeof value.confirmedDocumentHash !== "string" ||
    typeof value.headDocumentHash !== "string" ||
    !Number.isSafeInteger(value.updatedAtMs) ||
    (value.updatedAtMs as number) <= 0
  ) return null;
  const confirmedDocument = parseExactTimelineProject(value.confirmedDocument);
  if (!confirmedDocument) return null;
  return {
    confirmedRevision: value.confirmedRevision as number,
    confirmedDocumentHash: value.confirmedDocumentHash,
    confirmedDocument,
    headDocumentHash: value.headDocumentHash,
    history: value.history as SerializedTimelineHistoryEnvelope,
    updatedAtMs: value.updatedAtMs as number,
  };
}

function parseJournalV2(
  value: unknown,
  expectedScope: TimelinePersistenceScope,
): TimelineHistoryJournalRecordV2 | null {
  if (!hasExactKeys(value, [
    "confirmedDocument",
    "confirmedDocumentHash",
    "confirmedRevision",
    "format",
    "headDocumentHash",
    "history",
    "key",
    "scope",
    "updatedAtMs",
    "version",
    "writeToken",
  ])) return null;
  if (
    value.format !== TIMELINE_HISTORY_JOURNAL_FORMAT ||
    value.version !== TIMELINE_HISTORY_JOURNAL_VERSION ||
    value.key !== timelineHistoryJournalKey(expectedScope) ||
    typeof value.writeToken !== "string" ||
    !WRITE_TOKEN_PATTERN.test(value.writeToken) ||
    !hasExactKeys(value.scope, [
      "databasePath",
      "projectId",
      "ownerId",
    ]) ||
    value.scope.databasePath !== expectedScope.databasePath ||
    value.scope.projectId !== expectedScope.projectId ||
    value.scope.ownerId !== expectedScope.ownerId
  ) return null;
  const common = parseCommonJournalFields(value);
  if (!common) return null;
  return {
    key: value.key,
    format: TIMELINE_HISTORY_JOURNAL_FORMAT,
    version: TIMELINE_HISTORY_JOURNAL_VERSION,
    scope: structuredClone(expectedScope),
    writeToken: value.writeToken,
    ...common,
  };
}

function parseLegacyV4JournalV2(
  value: unknown,
  expectedScope: TimelinePersistenceScope,
): LegacyV4TimelineHistoryJournalRecordV2 | null {
  if (!hasExactKeys(value, [
    "confirmedDocument",
    "confirmedDocumentHash",
    "confirmedRevision",
    "format",
    "headDocumentHash",
    "history",
    "key",
    "scope",
    "updatedAtMs",
    "version",
    "writeToken",
  ])) return null;
  if (
    value.format !== TIMELINE_HISTORY_JOURNAL_FORMAT ||
    value.version !== TIMELINE_HISTORY_JOURNAL_VERSION ||
    value.key !== timelineHistoryJournalKey(expectedScope) ||
    typeof value.writeToken !== "string" ||
    !WRITE_TOKEN_PATTERN.test(value.writeToken) ||
    !hasExactKeys(value.scope, ["databasePath", "projectId", "ownerId"]) ||
    value.scope.databasePath !== expectedScope.databasePath ||
    value.scope.projectId !== expectedScope.projectId ||
    value.scope.ownerId !== expectedScope.ownerId ||
    !Number.isSafeInteger(value.confirmedRevision) ||
    (value.confirmedRevision as number) < 0 ||
    typeof value.confirmedDocumentHash !== "string" ||
    typeof value.headDocumentHash !== "string" ||
    !Number.isSafeInteger(value.updatedAtMs) ||
    (value.updatedAtMs as number) <= 0
  ) return null;
  const confirmedDocument = normalizeLegacyTimelineProject(value.confirmedDocument);
  if (
    !confirmedDocument || confirmedDocument.version !== 4 ||
    canonicalSnapshotJson(confirmedDocument) !== canonicalSnapshotJson(value.confirmedDocument)
  ) return null;
  return {
    key: value.key,
    format: TIMELINE_HISTORY_JOURNAL_FORMAT,
    version: TIMELINE_HISTORY_JOURNAL_VERSION,
    scope: structuredClone(expectedScope),
    writeToken: value.writeToken,
    confirmedRevision: value.confirmedRevision as number,
    confirmedDocumentHash: value.confirmedDocumentHash,
    confirmedDocument,
    headDocumentHash: value.headDocumentHash,
    history: value.history,
    updatedAtMs: value.updatedAtMs as number,
  };
}

function parseLegacyJournalV1(
  value: unknown,
  expectedScope: TimelinePersistenceProjectScope,
): LegacyTimelineHistoryJournalRecordV1 | null {
  if (!hasExactKeys(value, [
    "confirmedDocument",
    "confirmedDocumentHash",
    "confirmedRevision",
    "format",
    "headDocumentHash",
    "history",
    "key",
    "scope",
    "updatedAtMs",
    "version",
  ])) return null;
  if (
    value.format !== TIMELINE_HISTORY_JOURNAL_FORMAT ||
    value.version !== LEGACY_TIMELINE_HISTORY_JOURNAL_VERSION ||
    value.key !== legacyTimelineHistoryJournalKey(expectedScope) ||
    !hasExactKeys(value.scope, ["databasePath", "projectId"]) ||
    value.scope.databasePath !== expectedScope.databasePath ||
    value.scope.projectId !== expectedScope.projectId
  ) return null;
  const common = parseCommonJournalFields(value);
  if (!common) return null;
  return {
    key: value.key,
    format: TIMELINE_HISTORY_JOURNAL_FORMAT,
    version: LEGACY_TIMELINE_HISTORY_JOURNAL_VERSION,
    scope: structuredClone(expectedScope),
    ...common,
  };
}

async function decodeJournal(
  record: TimelineHistoryJournalRecordV2 | LegacyTimelineHistoryJournalRecordV1,
  authority: TimelinePersistenceAuthority,
): Promise<DecodedTimelineHistoryJournal | null> {
  const history = deserializeTimelineHistory(record.history);
  if (!history?.head) return null;
  const [confirmedHash, headHash] = await Promise.all([
    timelineProjectDigest(record.confirmedDocument),
    timelineProjectDigest(history.head),
  ]);
  if (
    confirmedHash !== record.confirmedDocumentHash ||
    headHash !== record.headDocumentHash
  ) return null;
  const decoded: DecodedTimelineHistoryJournal = {
    confirmedRevision: record.confirmedRevision,
    confirmedDocument: structuredClone(record.confirmedDocument),
    history,
    project: structuredClone(history.head),
  };
  const upgradedConfirmed = migrateTimelineFeatureBundle4To5(decoded.confirmedDocument);
  const upgradedHead = migrateTimelineFeatureBundle4To5(decoded.project);
  if (!upgradedConfirmed || !upgradedHead) return decoded;
  const confirmedProjectionMatches =
    authority.revision === decoded.confirmedRevision + 1 &&
    timelineProjectsEqual(authority.document, upgradedConfirmed);
  const acknowledgedProjectionMatches =
    authority.revision === decoded.confirmedRevision + 2 &&
    timelineProjectsEqual(authority.document, upgradedHead);
  if (!confirmedProjectionMatches && !acknowledgedProjectionMatches) return decoded;

  try {
    const upgradedHistory = deserializeTimelineHistory(serializeTimelineHistory({
      ...history,
      checkpoints: history.checkpoints.map((checkpoint) => {
        const project = migrateTimelineFeatureBundle4To5(checkpoint.project);
        if (!project) throw new TypeError("Timeline checkpoint is not bundle 4.");
        return { ...checkpoint, project };
      }),
      head: upgradedHead,
    }));
    if (!upgradedHistory?.head) return null;
    return {
      confirmedRevision: authority.revision,
      confirmedDocument: structuredClone(authority.document),
      history: upgradedHistory,
      project: structuredClone(upgradedHistory.head),
    };
  } catch {
    return null;
  }
}

async function decodeLegacyV4Journal(
  record: LegacyV4TimelineHistoryJournalRecordV2,
  receipt: ProjectMigrationReceipt,
  authority: TimelinePersistenceAuthority,
): Promise<DecodedTimelineHistoryJournal | null> {
  const rawHistory = isRecord(record.history) && isRecord(record.history.payload)
    ? record.history.payload
    : null;
  const rawHead = rawHistory?.head;
  const legacyHead = normalizeLegacyTimelineProject(rawHead);
  if (
    receipt.project_id !== record.scope.projectId ||
    !legacyHead ||
    await timelineProjectDigest(record.confirmedDocument) !== record.confirmedDocumentHash ||
    await timelineProjectDigest(rawHead) !== record.headDocumentHash
  ) return null;
  const baseMatchesReceipt =
    receipt.old_revision === record.confirmedRevision &&
    legacyClientDocumentDigest(record.confirmedDocument)?.value ===
      receipt.old_client_digest.value &&
    record.confirmedDocumentHash === receipt.old_server_digest.value;
  const pendingMatchesReceipt =
    receipt.old_revision === record.confirmedRevision + 1 &&
    legacyClientDocumentDigest(legacyHead)?.value === receipt.old_client_digest.value &&
    record.headDocumentHash === receipt.old_server_digest.value;
  if (!baseMatchesReceipt && !pendingMatchesReceipt) return null;
  const receiptSource = baseMatchesReceipt ? record.confirmedDocument : legacyHead;
  const frozenReceiptContext = {
    ...receipt.legacy_creative_binding_context,
    template_bundle_version: 4,
  };
  const receiptDestination = migrateLegacyTimelineProjectToV5(
    receiptSource,
    frozenReceiptContext,
  );
  if (
    !receiptDestination ||
    legacyClientDocumentDigest(receiptDestination)?.value !==
      receipt.new_client_digest.value
  ) return null;
  const upgradedReceiptDestination = migrateTimelineFeatureBundle4To5(
    receiptDestination,
  );
  const directAuthority =
    authority.revision === receipt.new_revision &&
    timelineProjectsEqual(authority.document, receiptDestination);
  const upgradedRevision = receipt.new_revision + 1;
  const upgradedAuthority =
    upgradedReceiptDestination !== null &&
    Number.isSafeInteger(upgradedRevision) &&
    authority.revision === upgradedRevision &&
    timelineProjectsEqual(authority.document, upgradedReceiptDestination);
  // An authority at the post-receipt revision is represented in current bundle
  // form even when it contains an unrelated edit. Classification below keeps
  // that valid local branch as conflict evidence instead of calling it corrupt.
  const targetBundleVersion = upgradedAuthority || authority.revision >= upgradedRevision
    ? 5
    : 4;
  const migrationContext = {
    ...receipt.legacy_creative_binding_context,
    template_bundle_version: targetBundleVersion,
  };
  const migratedConfirmed = migrateLegacyTimelineProjectToV5(
    record.confirmedDocument,
    migrationContext,
  );
  const migratedHistory = migrateLegacyTimelineHistoryEnvelope(
    record.history,
    migrationContext,
  );
  if (!migratedConfirmed || !migratedHistory?.history.head) return null;
  return {
    confirmedRevision: targetBundleVersion === 5
      ? upgradedRevision
      : receipt.new_revision,
    confirmedDocument: migratedConfirmed,
    history: migratedHistory.history,
    project: structuredClone(migratedHistory.history.head),
    receiptAuthorityMatched: directAuthority || upgradedAuthority,
  };
}

function classifyDecodedJournal(
  decoded: DecodedTimelineHistoryJournal,
  authority: TimelinePersistenceAuthority,
): "restored" | "acknowledged" | "conflict" {
  if (decoded.receiptAuthorityMatched === false) return "conflict";
  // A clean journal commonly records confirmedDocument === history.head. It
  // must be classified as acknowledged before the exact-base replay check, or
  // a fresh owner will mistake ordinary durable undo history for pending work.
  if (
    authority.revision >= decoded.confirmedRevision &&
    timelineProjectsEqual(authority.document, decoded.project)
  ) return "acknowledged";
  if (
    authority.revision === decoded.confirmedRevision &&
    timelineProjectsEqual(authority.document, decoded.confirmedDocument)
  ) return "restored";
  return "conflict";
}

function canonicalSnapshotJson(
  value: unknown,
  ancestors = new Set<object>(),
  depth = 0,
): string | null {
  if (depth > 64) return null;
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return null;
    return Object.is(value, -0) ? "-0" : JSON.stringify(value);
  }
  if (typeof value !== "object" || ancestors.has(value)) return null;
  ancestors.add(value);
  let encoded: string | null = null;
  if (Array.isArray(value)) {
    if (Object.keys(value).every((key, index) => key === String(index)) &&
      Object.keys(value).length === value.length) {
      const parts = value.map((item) => canonicalSnapshotJson(item, ancestors, depth + 1));
      if (parts.every((part): part is string => part !== null)) encoded = `[${parts.join(",")}]`;
    }
  } else if (isRecord(value) && Reflect.ownKeys(value).every((key) => typeof key === "string")) {
    const keys = Object.keys(value).sort();
    const parts: string[] = [];
    let valid = true;
    for (const key of keys) {
      const item = canonicalSnapshotJson(value[key], ancestors, depth + 1);
      if (item === null) {
        valid = false;
        break;
      }
      parts.push(`${JSON.stringify(key)}:${item}`);
    }
    if (valid) encoded = `{${parts.join(",")}}`;
  }
  ancestors.delete(value);
  return encoded;
}

function versionTokenForStoredValue(
  value: unknown,
  key: string,
): TimelineHistoryJournalVersionToken | null {
  const snapshot = canonicalSnapshotJson(value);
  return snapshot === null ? null : { key, version: `sha256:${sha256Hex(snapshot)}` };
}

function journalUpdatedAt(value: unknown): number | null {
  return isRecord(value) && Number.isSafeInteger(value.updatedAtMs) &&
      (value.updatedAtMs as number) > 0
    ? value.updatedAtMs as number
    : null;
}

/**
 * Reads the compare-and-delete token for the exact owner key without adopting
 * or classifying its project. This lets a serialized cleanup recover its token
 * after a preceding put completed but before React published the in-memory map.
 */
export async function readTimelineHistoryJournalVersionToken(
  scope: TimelinePersistenceScope,
): Promise<TimelineHistoryJournalVersionToken | null> {
  if (!validScope(scope)) return null;
  const key = timelineHistoryJournalKey(scope);
  try {
    const value = await readJournal(key);
    return value === undefined ? null : versionTokenForStoredValue(value, key);
  } catch {
    return null;
  }
}

/**
 * Cheap read-only probe used only to decide whether hydration should request a
 * v4→v5 receipt. It never classifies, rewrites, or adopts the legacy branch.
 */
export async function hasLegacyV4TimelineHistoryJournalEvidence(
  scope: TimelinePersistenceProjectScope,
): Promise<boolean> {
  if (!validProjectScope(scope)) return false;
  let values: unknown[] | null;
  try {
    values = await readAllJournals();
  } catch {
    return false;
  }
  if (!values) return false;
  const expectedDigest = timelineHistoryProjectScopeDigest(scope);
  return values.some((value) => {
    if (!isRecord(value) || typeof value.key !== "string") return false;
    const v2Key = parseV2JournalKey(value.key);
    const legacyScope = parseLegacyJournalKey(value.key);
    const matches = v2Key?.scopeDigest === expectedDigest ||
      Boolean(legacyScope && sameProjectScope(legacyScope, scope));
    if (!matches || !isRecord(value.confirmedDocument) ||
      value.confirmedDocument.version !== 4 || !isRecord(value.history)) return false;
    return value.history.schemaVersion === 1;
  });
}

/** Reads only the current owner branch; foreign and legacy evidence is never replayed here. */
export async function loadTimelineHistoryJournal(
  scope: TimelinePersistenceScope,
  authority: TimelinePersistenceAuthority,
  legacyReceipt?: ProjectMigrationReceipt | null,
): Promise<TimelineHistoryJournalLoadResult> {
  const authorityDocument = parseExactTimelineProject(authority.document);
  if (
    !validScope(scope) ||
    !Number.isSafeInteger(authority.revision) ||
    authority.revision < 0 ||
    !authorityDocument
  ) {
    return { status: "corrupt", token: null, updatedAtMs: null };
  }
  const normalizedAuthority = { document: authorityDocument, revision: authority.revision };
  const key = timelineHistoryJournalKey(scope);
  let value: unknown | undefined;
  try {
    value = await readJournal(key);
  } catch {
    return { status: "unavailable" };
  }
  if (value === undefined) {
    return typeof indexedDB === "undefined" ? { status: "unavailable" } : { status: "none" };
  }
  const token = versionTokenForStoredValue(value, key);
  const updatedAtMs = journalUpdatedAt(value);
  const record = parseJournalV2(value, scope);
  const legacyRecord = record ? null : parseLegacyV4JournalV2(value, scope);
  if (!record && !legacyRecord) return { status: "corrupt", token, updatedAtMs };
  const decoded = record
    ? await decodeJournal(record, normalizedAuthority)
    : legacyReceipt && legacyRecord
      ? await decodeLegacyV4Journal(legacyRecord, legacyReceipt, normalizedAuthority)
      : null;
  if (!decoded) return { status: "corrupt", token, updatedAtMs };
  if (!token) return { status: "corrupt", token: null, updatedAtMs };

  const metadata: TimelineHistoryJournalLoadMetadata = {
    ownerId: scope.ownerId,
    token,
    updatedAtMs: (record ?? legacyRecord)!.updatedAtMs,
  };
  const classification = classifyDecodedJournal(decoded, normalizedAuthority);
  if (classification === "restored" || classification === "acknowledged") {
    return {
      ...metadata,
      status: classification,
      history: decoded.history,
      project: decoded.project,
      confirmedRevision: decoded.confirmedRevision,
      confirmedDocument: structuredClone(decoded.confirmedDocument),
    };
  }
  return {
    ...metadata,
    status: "conflict",
    confirmedRevision: decoded.confirmedRevision,
    confirmedDocument: structuredClone(decoded.confirmedDocument),
    localProject: decoded.project,
    history: decoded.history,
    project: structuredClone(decoded.project),
  };
}

function newestBranchFirst(
  left: TimelineHistoryJournalBranchEvidence,
  right: TimelineHistoryJournalBranchEvidence,
): number {
  if (left.updatedAtMs !== right.updatedAtMs) {
    if (left.updatedAtMs === null) return 1;
    if (right.updatedAtMs === null) return -1;
    return left.updatedAtMs < right.updatedAtMs ? 1 : -1;
  }
  const leftKey = left.token?.key ?? "";
  const rightKey = right.token?.key ?? "";
  return leftKey.localeCompare(rightKey);
}

async function classifyBranchEvidence(
  value: unknown,
  key: string,
  ownerId: string | null,
  currentScope: TimelinePersistenceScope,
  authority: TimelinePersistenceAuthority,
  legacyReceipt?: ProjectMigrationReceipt | null,
): Promise<TimelineHistoryJournalBranchEvidence> {
  const ownership: TimelineHistoryJournalBranchOwnership = ownerId === null
    ? "legacy"
    : ownerId === currentScope.ownerId ? "owned" : "foreign";
  const metadata: TimelineHistoryJournalBranchMetadata = {
    ownership,
    ownerId,
    token: versionTokenForStoredValue(value, key),
    updatedAtMs: journalUpdatedAt(value),
  };
  const projectScope: TimelinePersistenceProjectScope = {
    databasePath: currentScope.databasePath,
    projectId: currentScope.projectId,
  };
  const record = ownerId === null
    ? parseLegacyJournalV1(value, projectScope)
    : parseJournalV2(value, { ...projectScope, ownerId });
  const legacyRecord = record || ownerId === null
    ? null
    : parseLegacyV4JournalV2(value, { ...projectScope, ownerId });
  if (!record && !legacyRecord) return { ...metadata, status: "corrupt" };
  const decoded = record
    ? await decodeJournal(record, authority)
    : legacyReceipt && legacyRecord
      ? await decodeLegacyV4Journal(legacyRecord, legacyReceipt, authority)
      : null;
  if (!decoded) return { ...metadata, status: "corrupt" };
  return {
    ...metadata,
    status: classifyDecodedJournal(decoded, authority),
    confirmedRevision: decoded.confirmedRevision,
    confirmedDocument: structuredClone(decoded.confirmedDocument),
    history: decoded.history,
    project: decoded.project,
  };
}

/**
 * Enumerates every branch in one database/project scope. Foreign and legacy
 * branches are evidence only: this function never mutates, adopts, or deletes them.
 */
export async function listTimelineHistoryJournalBranches(
  scope: TimelinePersistenceScope,
  authority: TimelinePersistenceAuthority,
  legacyReceipt?: ProjectMigrationReceipt | null,
): Promise<TimelineHistoryJournalBranchListResult> {
  const authorityDocument = parseExactTimelineProject(authority.document);
  if (
    !validScope(scope) ||
    !Number.isSafeInteger(authority.revision) ||
    authority.revision < 0 ||
    !authorityDocument
  ) {
    return { status: "corrupt" };
  }
  const normalizedAuthority = { document: authorityDocument, revision: authority.revision };
  let values: unknown[] | null;
  try {
    values = await readAllJournals();
  } catch {
    return { status: "unavailable" };
  }
  if (values === null) return { status: "unavailable" };

  const expectedDigest = timelineHistoryProjectScopeDigest(scope);
  const matching: Array<{
    value: Record<string, unknown>;
    key: string;
    ownerId: string | null;
  }> = [];
  for (const value of values) {
    if (!isRecord(value) || typeof value.key !== "string") continue;
    const v2Key = parseV2JournalKey(value.key);
    if (v2Key?.scopeDigest === expectedDigest) {
      matching.push({ value, key: value.key, ownerId: v2Key.ownerId });
      continue;
    }
    const legacyScope = parseLegacyJournalKey(value.key);
    if (legacyScope && sameProjectScope(legacyScope, scope)) {
      matching.push({ value, key: value.key, ownerId: null });
    }
  }
  const branches = (await Promise.all(matching.map(({ value, key, ownerId }) =>
    classifyBranchEvidence(
      value,
      key,
      ownerId,
      scope,
      normalizedAuthority,
      legacyReceipt,
    ))))
    .sort(newestBranchFirst);
  return {
    status: "available",
    branches,
    owned: branches.filter((branch) => branch.ownership === "owned"),
    foreign: branches.filter((branch) => branch.ownership === "foreign"),
    legacy: branches.filter((branch) => branch.ownership === "legacy"),
    corrupt: branches.filter((branch) => branch.status === "corrupt"),
  };
}

function validVersionToken(value: unknown): value is TimelineHistoryJournalVersionToken {
  return hasExactKeys(value, ["key", "version"]) &&
    typeof value.key === "string" &&
    typeof value.version === "string" &&
    /^sha256:[0-9a-f]{64}$/.test(value.version);
}

/**
 * Compare-and-delete one exact branch version. Omitting a token deliberately
 * fails closed, so a late cleanup cannot erase a newer put from the same owner.
 */
export async function deleteTimelineHistoryJournal(
  scope: TimelinePersistenceProjectScope,
  expectedToken?: TimelineHistoryJournalVersionToken | null,
): Promise<boolean> {
  if (!validProjectScope(scope) || !validVersionToken(expectedToken)) return false;
  const v2Key = parseV2JournalKey(expectedToken.key);
  const legacyScope = parseLegacyJournalKey(expectedToken.key);
  if (
    v2Key?.scopeDigest !== timelineHistoryProjectScopeDigest(scope) &&
    (!legacyScope || !sameProjectScope(legacyScope, scope))
  ) return false;
  const database = await openTimelineHistoryDatabase();
  if (!database) return false;
  try {
    return await new Promise<boolean>((resolve, reject) => {
      const transaction = database.transaction(TIMELINE_HISTORY_STORE, "readwrite");
      const store = transaction.objectStore(TIMELINE_HISTORY_STORE);
      const request = store.get(expectedToken.key);
      let deleted = false;
      request.onsuccess = () => {
        if (request.result === undefined) {
          // Idempotent convergence: an earlier partial discard already reached
          // the requested state. A present different version still fails below.
          deleted = true;
          return;
        }
        const currentToken = versionTokenForStoredValue(request.result, expectedToken.key);
        if (currentToken?.version !== expectedToken.version) return;
        store.delete(expectedToken.key);
        deleted = true;
      };
      request.onerror = () => transaction.abort();
      transaction.oncomplete = () => resolve(deleted);
      transaction.onerror = () => reject(
        transaction.error ?? new Error("无法清除时间线历史"),
      );
      transaction.onabort = () => reject(
        transaction.error ?? request.error ?? new Error("时间线历史清除已中止"),
      );
    });
  } finally {
    database.close();
  }
}

export function deleteTimelineHistoryDatabaseForTests(): Promise<void> {
  if (typeof indexedDB === "undefined") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(TIMELINE_HISTORY_DATABASE);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("timeline history test database is blocked"));
  });
}
