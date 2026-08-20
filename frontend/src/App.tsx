import { useCallback, useEffect, useLayoutEffect, useReducer, useRef, useState, type CSSProperties } from "react";
import { ApiError, directorApi, taskEventsUrl } from "./api/client";
import {
  EMPTY_CAPABILITIES,
  EMPTY_MODELS,
  rayLightResidencyPolicyAfterBindingChange,
  resolveExecutionBackend,
  sanitizeRuntimeSettings,
  type CapabilityReport,
  type DiffusionModelBinding,
  type DiffusionModelRole,
  type GenerationTask,
  type GPUResource,
  type ModelInventory,
  type ProjectSummary,
  type RayLightRuntimeStatus,
  type RuntimeSettings,
  type RuntimeSettingsAuthority,
  type StorageConfiguration,
  type AssetTrashBatch,
  type AssetTrashRestoreMode,
  type TimelineAuthority,
  type TimelineCompileReport,
} from "./api/types";
import {
  LongFormTimelineWorkspace,
  TIMELINE_RUN_VALIDATION_ID,
} from "./components/LongFormTimelineWorkspace";
import { SettingsPage, validateRuntimeSettingsForm } from "./components/SettingsPage";
import { TaskDrawer } from "./components/TaskDrawer";
import { AssetTrashPanel } from "./components/AssetTrashPanel";
import { TimelineHistoryPanel } from "./components/TimelineHistoryPanel";
import { TimelineGlobalSettings } from "./components/TimelineGlobalSettings";
import { WorkspaceAssetSidebar } from "./components/WorkspaceAssetSidebar";
import { Spinner } from "./components/ui";
import {
  classifyDroppedFiles,
  describeUploadProgress,
  uploadClassifiedDroppedFiles,
  type DroppedUploadProgress,
  type DroppedUploadResult,
} from "./domain/assetDrag";
import { randomSafeSeed } from "./domain/modes";
import { isStoragePath } from "./domain/storagePath";
import { persistUiTheme, readUiTheme } from "./domain/theme";
import {
  loadTimelineSegmentSelectionPreference,
  saveTimelineSegmentSelectionPreference,
} from "./domain/workspacePreferences";
import {
  autoFitSourceAudioTiming,
  createTimelineEditorState,
  createTimelineBranchOwnerId,
  clearLocalTimelineWal,
  DEFAULT_PROJECT_ID,
  discardLocalTimelineWalBranch,
  getTimelineBranchOwnerId,
  listLocalTimelineWalBranches,
  loadAssetLayoutPreference,
  loadLocalTimelineWal,
  normalizeTimelineProject,
  orderAssetsByPreference,
  resolveLocalTimelineWal,
  runnableTimelineSegmentIds,
  saveAssetLayoutPreference,
  saveLocalTimelineWal,
  segmentAssetReferences,
  timelineSamplingFamily,
  timelineAssetUsages,
  timelineEditorReducer,
  validateTimelineProject,
  type LocalTimelineWal,
  type CorruptLocalTimelineWalBranchEvidence,
  type LocalTimelineWalBranchEvidence,
  type TimelineAction,
  type TimelineEditorState,
  type TimelineProject,
} from "./domain/timelineProject";
import {
  directorReducer,
  loadDirectorState,
  saveDirectorState,
} from "./state/directorState";
import {
  canRedoTimelineHistory,
  canUndoTimelineHistory,
  captureTimelineHistoryContext,
  createTimelineHistory,
  jumpTimelineHistory,
  recordTimelineHistory,
  rebaseTimelineHistoryHead,
  redoTimelineHistory,
  resetTimelineHistory,
  sealTimelineHistoryCoalescing,
  timelineHistoryRedoLabel,
  timelineHistoryUndoLabel,
  timelineProjectsEqual,
  undoTimelineHistory,
  type TimelineHistoryContext,
  type TimelineHistoryReplay,
  type TimelineHistoryState,
  type TimelineTextEditingContext,
} from "./state/timelineHistory";
import { reduceTimelineTransaction } from "./state/timelineTransactions";
import {
  createTimelineRevisionChannel,
  runWithTimelineWriterLock,
  type TimelineRevisionChannel,
} from "./state/timelineCoordination";
import {
  deleteTimelineHistoryJournal,
  listTimelineHistoryJournalBranches,
  loadTimelineHistoryJournal,
  readTimelineHistoryJournalVersionToken,
  saveTimelineHistoryJournal,
  type TimelineHistoryJournalBranchEvidence,
  type TimelineHistoryJournalVersionToken,
  type TimelinePersistenceAuthority,
  type TimelinePersistenceScope,
} from "./state/timelinePersistence";

const SIDEBAR_OPEN_KEY = "directordeck:sidebar-open";
const SIDEBAR_WIDTH_KEY = "directordeck:sidebar-expanded-width";
const SIDEBAR_MOBILE_MAX = 760;
const GLOBAL_SETTINGS_ID = "timeline-global-settings";
const TIMELINE_HISTORY_PANEL_ID = "timeline-history-panel";
const ASSET_TRASH_PANEL_ID = "asset-trash-panel";
export const UNBOUND_RUNTIME_SETTINGS_PENDING_KEY = "directordeck:runtime-settings-pending";
export const QUARANTINED_UNBOUND_RUNTIME_SETTINGS_PENDING_KEY = "directordeck:runtime-settings-pending-quarantine";
export const RUNTIME_SETTINGS_PENDING_KEY = "directordeck:v2:runtime-settings-pending";
export const QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY = "directordeck:v2:runtime-settings-pending-quarantine";
const RUNTIME_SETTINGS_PENDING_FORMAT = "director-pending-runtime-settings";
const LEGACY_RUNTIME_SETTINGS_PENDING_VERSION = 1;
const RUNTIME_SETTINGS_PENDING_VERSION = 2;
const RUNTIME_SETTINGS_AUTOSAVE_MS = 300;
const RUNTIME_SETTINGS_RETRY_MS = 1500;
const STORAGE_AUTHORITY_RETRY_MS = 1000;
const RAYLIGHT_RECOVERY_RETRY_MS = 300;
const RAYLIGHT_RECOVERY_MAX_RETRY_MS = 1200;

type TimelineHistoryMode = "record" | "replay" | "skip" | "reset";

interface TimelineDispatchOptions {
  history?: TimelineHistoryMode;
  historyLabel?: string;
  historyMergeKey?: string;
}

type TimelineRecoveryWalEvidence =
  | LocalTimelineWalBranchEvidence
  | CorruptLocalTimelineWalBranchEvidence;

interface TimelineRecoveryBranch {
  id: string;
  ownerId: string | null;
  ownership: "owned" | "foreign" | "legacy";
  updatedAtMs: number | null;
  status: "replay" | "acknowledged" | "conflict" | "corrupt";
  project: TimelineProject | null;
  history: TimelineHistoryState | null;
  walEvidence: TimelineRecoveryWalEvidence | null;
  journalEvidence: TimelineHistoryJournalBranchEvidence | null;
}

interface TimelineRevisionConflict {
  projectId: string;
  localProject: TimelineProject;
  serverAuthority: TimelineAuthority | null;
  source: "cas" | "legacy-wal" | "recovery-branches";
  resolving: boolean;
  recoveryBranches?: TimelineRecoveryBranch[];
  selectedRecoveryBranchId?: string | null;
}

function timelineDocumentHash(project: TimelineProject): string {
  const value = JSON.stringify(project);
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return `fnv1a-${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

function timelineRecoveryBranchId(
  wal: TimelineRecoveryWalEvidence | null,
  journal: TimelineHistoryJournalBranchEvidence | null,
): string {
  if (wal) return `wal:${wal.storage_key}`;
  if (journal?.token) return `journal:${journal.token.key}:${journal.ownerId ?? "legacy"}`;
  return `journal-corrupt:${journal?.ownerId ?? "legacy"}:${journal?.updatedAtMs ?? "unknown"}`;
}

function journalRecoveryProject(
  journal: TimelineHistoryJournalBranchEvidence,
): TimelineProject | null {
  return journal.status === "corrupt" ? null : journal.project;
}

function collectTimelineRecoveryBranches(
  walBranches: ReturnType<typeof listLocalTimelineWalBranches>,
  journalBranches: TimelineHistoryJournalBranchEvidence[],
  authority: TimelineAuthority,
): {
  pending: TimelineRecoveryBranch[];
  newestAcknowledgedHistory: TimelineHistoryState | null;
} {
  const pending: TimelineRecoveryBranch[] = [];
  const consumedJournals = new Set<TimelineHistoryJournalBranchEvidence>();
  const validWals = [
    ...(walBranches.owned ? [walBranches.owned] : []),
    ...walBranches.foreign,
    ...walBranches.legacy,
  ];
  const acknowledgedHistoryCandidates: Array<{
    history: TimelineHistoryState;
    updatedAtMs: number;
  }> = [];
  const divergentJournals = new Set<TimelineHistoryJournalBranchEvidence>();
  for (const walEvidence of validWals) {
    const resolution = resolveLocalTimelineWal(walEvidence.wal, authority);
    const ownerJournal = journalBranches.find((candidate) =>
      candidate.ownerId === walEvidence.wal.owner_id) ?? null;
    const journalMatchesHead = ownerJournal !== null && ownerJournal.status !== "corrupt" &&
      timelineProjectsEqual(ownerJournal.project, walEvidence.wal.pending_project);
    const journal = journalMatchesHead ? ownerJournal : null;
    if (journal) consumedJournals.add(journal);
    else if (ownerJournal) divergentJournals.add(ownerJournal);
    if (resolution.status === "acknowledged") {
      if (journal && journal.status === "acknowledged") {
        acknowledgedHistoryCandidates.push({
          history: journal.history,
          updatedAtMs: journal.updatedAtMs ?? walEvidence.wal.written_at_ms,
        });
      }
      if (ownerJournal && !journalMatchesHead) {
        // localStorage and IndexedDB can diverge even for one owner (quota,
        // crash, or a delayed transaction). Preserve both as independent
        // evidence instead of hiding the newer journal behind this WAL.
        pending.push({
          id: timelineRecoveryBranchId(walEvidence, null),
          ownerId: walEvidence.wal.owner_id,
          ownership: walEvidence.ownership,
          updatedAtMs: walEvidence.wal.written_at_ms,
          status: "acknowledged",
          project: resolution.project,
          history: null,
          walEvidence,
          journalEvidence: null,
        });
      }
      continue;
    }
    pending.push({
      id: timelineRecoveryBranchId(walEvidence, journal),
      ownerId: walEvidence.wal.owner_id,
      ownership: walEvidence.ownership,
      updatedAtMs: walEvidence.wal.written_at_ms,
      status: resolution.status,
      project: resolution.status === "replay" ? resolution.project : resolution.local_project,
      history: journal && (journal.status === "restored" ||
          journal.status === "acknowledged" || journal.status === "conflict")
        ? journal.history
        : null,
      walEvidence,
      journalEvidence: journal,
    });
  }
  for (const walEvidence of walBranches.corrupt) {
    pending.push({
      id: timelineRecoveryBranchId(walEvidence, null),
      ownerId: null,
      ownership: walEvidence.ownership,
      updatedAtMs: null,
      status: "corrupt",
      project: null,
      history: null,
      walEvidence,
      journalEvidence: null,
    });
  }

  for (const journal of journalBranches) {
    if (consumedJournals.has(journal)) continue;
    if (journal.status === "acknowledged") {
      acknowledgedHistoryCandidates.push({
        history: journal.history,
        updatedAtMs: journal.updatedAtMs ?? -1,
      });
      if (!divergentJournals.has(journal)) continue;
    }
    // A clean journal records head === confirmed base: everything that session
    // knew was already synced at confirm time. When the server has since moved
    // on, the branch classifies as "conflict" yet carries no unsynced content
    // to recover. Listing it would re-surface the recovery gate on every
    // refresh forever, so keep its bytes as silent evidence only.
    if (
      journal.status === "conflict" &&
      timelineProjectsEqual(journal.project, journal.confirmedDocument)
    ) continue;
    const project = journalRecoveryProject(journal);
    pending.push({
      id: timelineRecoveryBranchId(null, journal),
      ownerId: journal.ownerId,
      ownership: journal.ownership,
      updatedAtMs: journal.updatedAtMs,
      status: journal.status === "restored" ? "replay" : journal.status,
      project,
      history: journal.status === "corrupt" ? null : journal.history,
      walEvidence: null,
      journalEvidence: journal,
    });
  }
  pending.sort((left, right) =>
    (right.updatedAtMs ?? -1) - (left.updatedAtMs ?? -1) || left.id.localeCompare(right.id));
  acknowledgedHistoryCandidates.sort((left, right) => right.updatedAtMs - left.updatedAtMs);
  // These candidates are already exact server-head matches. A separate
  // pending branch must not make safe undo history disappear from the server
  // workspace shown behind the explicit recovery gate.
  const newestAcknowledgedHistory = acknowledgedHistoryCandidates[0]?.history ?? null;
  return { pending, newestAcknowledgedHistory };
}

function shouldKeepNativeUndo(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  if (target.closest("[data-timeline-history-ignore]")) return true;
  if (target.closest("[data-timeline-history-field]")) return false;
  if (target.closest("textarea, [contenteditable]:not([contenteditable='false'])")) return true;
  const input = target.closest("input");
  if (!(input instanceof HTMLInputElement)) return false;
  return !["button", "checkbox", "radio", "range", "color", "file", "submit", "reset"].includes(input.type);
}

function isTimelineHistoryTextTarget(target: EventTarget | null): boolean {
  return target instanceof Element && Boolean(target.closest("[data-timeline-history-field]"));
}

function timelineHistoryTextControl(
  target: EventTarget | null,
): HTMLInputElement | HTMLTextAreaElement | null {
  if (!(target instanceof Element)) return null;
  const field = target.closest("[data-timeline-history-field]");
  return field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement
    ? field
    : null;
}

function captureTimelineTextEditingContext(
  target: EventTarget | null,
): TimelineTextEditingContext | null {
  const field = timelineHistoryTextControl(target);
  const fieldKey = field?.dataset.timelineHistoryField;
  if (!field || !fieldKey) return null;
  return {
    field_key: fieldKey,
    start: field.selectionStart ?? 0,
    end: field.selectionEnd ?? field.selectionStart ?? 0,
    direction: field.selectionDirection ?? "none",
  };
}

function timelineTextHistoryContext(
  state: TimelineEditorState,
  textEditing: TimelineTextEditingContext,
): TimelineHistoryContext {
  return {
    ...captureTimelineHistoryContext(state),
    restore_segment_selection: false,
    text_editing: textEditing,
  };
}

function collapsedTimelineTextContextAfter(
  before: TimelineTextEditingContext,
  beforeValue: string,
  afterValue: string,
): TimelineTextEditingContext {
  const unchangedSuffix = Math.max(0, beforeValue.length - before.end);
  const cursor = Math.max(
    0,
    Math.min(afterValue.length, afterValue.length - unchangedSuffix),
  );
  return { ...before, start: cursor, end: cursor, direction: "none" };
}

function restoreTimelineTextEditingSelection(
  context: TimelineTextEditingContext | undefined,
  sourceFieldKey: string | null,
): void {
  if (!context || context.field_key !== sourceFieldKey) return;
  window.requestAnimationFrame(() => {
    const stillActive = captureTimelineTextEditingContext(document.activeElement);
    if (stillActive?.field_key !== sourceFieldKey) return;
    const field = [...document.querySelectorAll<HTMLElement>("[data-timeline-history-field]")]
      .find((candidate) => candidate.dataset.timelineHistoryField === context.field_key);
    if (!(field instanceof HTMLInputElement || field instanceof HTMLTextAreaElement)) return;
    const start = Math.min(field.value.length, Math.max(0, context.start));
    const end = Math.min(field.value.length, Math.max(start, context.end));
    field.focus({ preventScroll: true });
    field.setSelectionRange(start, end, context.direction);
  });
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException("The operation was aborted", "AbortError");
}

function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw abortReason(signal);
}

function waitForRayLightRecoveryWindow(
  delayMs: number,
  signal: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let timer: number | null = null;
    let waitingForOnline = false;
    const cleanup = () => {
      if (timer !== null) window.clearTimeout(timer);
      if (waitingForOnline) window.removeEventListener("online", onOnline);
      signal.removeEventListener("abort", onAbort);
    };
    const settle = () => {
      cleanup();
      resolve();
    };
    const onAbort = () => {
      cleanup();
      reject(abortReason(signal));
    };
    const startTimer = () => {
      if (signal.aborted) {
        onAbort();
        return;
      }
      if (navigator.onLine === false) {
        waitingForOnline = true;
        window.addEventListener("online", onOnline, { once: true });
        return;
      }
      timer = window.setTimeout(settle, delayMs);
    };
    const onOnline = () => {
      waitingForOnline = false;
      startTimer();
    };
    signal.addEventListener("abort", onAbort, { once: true });
    startTimer();
  });
}

type RuntimeSettingsOperationOwner = "settings-page" | DiffusionModelRole | "resync";

let runtimeSettingsWalOwnerCache: string | null = null;
let adoptedRuntimeSettingsWalRaw: string | null = null;
let latestRuntimeSettingsWalRaw: string | null = null;

class RuntimeSettingsSupersededError extends Error {
  constructor() {
    super("运行设置已被更新修改取代");
    this.name = "RuntimeSettingsSupersededError";
  }
}

class DatabaseIdentityChangedDuringHydrationError extends Error {
  constructor() {
    super("Director 后端数据库已在页面加载期间变化，请刷新整个页面");
    this.name = "DatabaseIdentityChangedDuringHydrationError";
  }
}

function initialUiToggle(key: string, fallback: boolean): boolean {
  try {
    const saved = window.localStorage.getItem(key);
    if (saved === "true") return true;
    if (saved === "false") return false;
  } catch {
    // Storage may be unavailable in privacy mode.
  }
  return fallback;
}

function sidebarWidthBounds(viewportWidth: number) {
  const minimum = viewportWidth <= 1050 ? 235 : viewportWidth <= 1350 ? 260 : 292;
  return {
    minimum,
    maximum: Math.max(minimum, Math.floor(viewportWidth / 2)),
  };
}

function clampSidebarWidth(width: number, viewportWidth: number): number {
  const { minimum, maximum } = sidebarWidthBounds(viewportWidth);
  return Math.min(maximum, Math.max(minimum, Math.round(width)));
}

function nextRandomSeed(previous: number): number {
  const next = randomSafeSeed();
  if (next !== previous) return next;
  return previous === Number.MAX_SAFE_INTEGER ? 0 : previous + 1;
}

function initialSidebarWidth(): number {
  const viewportWidth = window.innerWidth;
  const { minimum } = sidebarWidthBounds(viewportWidth);
  try {
    const saved = Number(window.localStorage.getItem(SIDEBAR_WIDTH_KEY));
    return clampSidebarWidth(Number.isFinite(saved) && saved > 0 ? saved : minimum, viewportWidth);
  } catch {
    return minimum;
  }
}

function sameRuntimeSettings(left: RuntimeSettings, right: RuntimeSettings): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

function quarantineRuntimeSettingsEntry(sourceKey: string, quarantineKey: string): void {
  try {
    const raw = window.localStorage.getItem(sourceKey);
    if (raw === null) return;
    let destination = quarantineKey;
    let existing = window.localStorage.getItem(destination);
    for (let index = 1; existing !== null && existing !== raw; index += 1) {
      destination = `${quarantineKey}:${index}`;
      existing = window.localStorage.getItem(destination);
    }
    if (existing === null) window.localStorage.setItem(destination, raw);
    if (
      window.localStorage.getItem(destination) === raw &&
      window.localStorage.getItem(sourceKey) === raw
    ) {
      window.localStorage.removeItem(sourceKey);
    }
  } catch {
    // Keep an entry in place when it cannot be copied. Loaders still ignore it.
  }
}

function quarantineUnboundRuntimeSettings(): void {
  quarantineRuntimeSettingsEntry(
    UNBOUND_RUNTIME_SETTINGS_PENDING_KEY,
    QUARANTINED_UNBOUND_RUNTIME_SETTINGS_PENDING_KEY,
  );
}

function validActiveDatabasePath(value: unknown): value is string {
  return isStoragePath(value);
}

interface ActiveDatabaseIdentity {
  active_database_path: string;
}

const INITIAL_TIMELINE_BRANCH_OWNER_ID = getTimelineBranchOwnerId();

interface AssetAuthorityScope {
  database: ActiveDatabaseIdentity;
}

function timelinePersistenceScope(
  database: ActiveDatabaseIdentity,
  projectId: string,
  ownerId: string = INITIAL_TIMELINE_BRANCH_OWNER_ID,
): TimelinePersistenceScope {
  return {
    databasePath: database.active_database_path,
    projectId,
    ownerId,
  };
}

function timelineJournalTokenMapKey(
  database: ActiveDatabaseIdentity,
  projectId: string,
  ownerId: string = INITIAL_TIMELINE_BRANCH_OWNER_ID,
): string {
  return `${database.active_database_path}:${projectId}:${ownerId}`;
}

function isCurrentTimelineHydration(
  signal: AbortSignal,
  expectedProjectId: string,
  currentProjectId: string,
  expectedGeneration: number,
  currentGeneration: number,
): boolean {
  return !signal.aborted &&
    expectedProjectId === currentProjectId &&
    expectedGeneration === currentGeneration;
}

function validActiveDatabaseIdentity(value: ActiveDatabaseIdentity): boolean {
  return validActiveDatabasePath(value.active_database_path);
}

function validRuntimeSettingsWalOwner(value: unknown): value is string {
  return typeof value === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(value);
}

function runtimeSettingsWalOwner(): string {
  if (runtimeSettingsWalOwnerCache) return runtimeSettingsWalOwnerCache;
  const generated = typeof globalThis.crypto?.randomUUID === "function"
    ? `tab-${globalThis.crypto.randomUUID()}`
    : `tab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
  runtimeSettingsWalOwnerCache = generated;
  return generated;
}

function parseRuntimeSettingsWalEnvelope(raw: string): {
  envelope: Record<string, unknown>;
  settings: RuntimeSettings;
} | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    const envelope = parsed as Record<string, unknown>;
    const keys = Object.keys(envelope).sort().join("|");
    const legacy = envelope.version === LEGACY_RUNTIME_SETTINGS_PENDING_VERSION &&
      keys === "active_database_path|format|pending|settings|version|written_at_ms";
    const owned = envelope.version === RUNTIME_SETTINGS_PENDING_VERSION &&
      keys === "active_database_path|format|owner_id|pending|settings|version|written_at_ms" &&
      validRuntimeSettingsWalOwner(envelope.owner_id);
    if (
      (!legacy && !owned) ||
      envelope.format !== RUNTIME_SETTINGS_PENDING_FORMAT ||
      envelope.pending !== true ||
      !validActiveDatabasePath(envelope.active_database_path) ||
      !Number.isSafeInteger(envelope.written_at_ms) ||
      (envelope.written_at_ms as number) <= 0 ||
      typeof envelope.settings !== "object" || envelope.settings === null || Array.isArray(envelope.settings)
    ) return null;
    const normalized = sanitizeRuntimeSettings(envelope.settings);
    if (!sameRuntimeSettings(envelope.settings as RuntimeSettings, normalized)) return null;
    return { envelope, settings: normalized };
  } catch {
    return null;
  }
}

function runtimeSettingsWalOwnedByCurrentTab(
  raw: string,
  database: ActiveDatabaseIdentity,
  owner: string,
): boolean {
  const parsed = parseRuntimeSettingsWalEnvelope(raw);
  return parsed?.envelope.version === RUNTIME_SETTINGS_PENDING_VERSION &&
    parsed.envelope.owner_id === owner &&
    parsed.envelope.active_database_path === database.active_database_path;
}

function loadPendingRuntimeSettings(database: ActiveDatabaseIdentity): RuntimeSettings | null {
  quarantineUnboundRuntimeSettings();
  if (!validActiveDatabaseIdentity(database)) return null;
  try {
    const raw = window.localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY);
    if (!raw) return null;
    const parsed = parseRuntimeSettingsWalEnvelope(raw);
    if (
      !parsed ||
      parsed.envelope.active_database_path !== database.active_database_path
    ) throw new Error("invalid runtime settings WAL envelope");
    adoptedRuntimeSettingsWalRaw = raw;
    return parsed.settings;
  } catch {
    quarantineRuntimeSettingsEntry(
      RUNTIME_SETTINGS_PENDING_KEY,
      QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY,
    );
    return null;
  }
}

function savePendingRuntimeSettings(settings: RuntimeSettings, database: ActiveDatabaseIdentity): void {
  quarantineUnboundRuntimeSettings();
  if (!validActiveDatabaseIdentity(database)) return;
  try {
    const owner = runtimeSettingsWalOwner();
    const raw = JSON.stringify({
      format: RUNTIME_SETTINGS_PENDING_FORMAT,
      version: RUNTIME_SETTINGS_PENDING_VERSION,
      owner_id: owner,
      pending: true,
      active_database_path: database.active_database_path,
      written_at_ms: Date.now(),
      settings,
    });
    const existing = window.localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY);
    if (existing !== null && !runtimeSettingsWalOwnedByCurrentTab(existing, database, owner)) {
      quarantineRuntimeSettingsEntry(
        RUNTIME_SETTINGS_PENDING_KEY,
        QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY,
      );
      if (window.localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY) !== null) return;
    }
    window.localStorage.setItem(RUNTIME_SETTINGS_PENDING_KEY, raw);
    if (window.localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY) === raw) {
      latestRuntimeSettingsWalRaw = raw;
    }
  } catch {
    // The in-memory desired remains usable for this browser session.
  }
}

function clearPendingRuntimeSettings(): void {
  quarantineUnboundRuntimeSettings();
  try {
    const current = window.localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY);
    if (
      current !== null &&
      (current === latestRuntimeSettingsWalRaw || current === adoptedRuntimeSettingsWalRaw)
    ) window.localStorage.removeItem(RUNTIME_SETTINGS_PENDING_KEY);
    latestRuntimeSettingsWalRaw = null;
    adoptedRuntimeSettingsWalRaw = null;
  }
  catch { /* WAL is best effort. */ }
}

function createInitialTimelineState() {
  const state = createTimelineEditorState();
  const layout = loadAssetLayoutPreference();
  state.selected_segment_ids = state.project.segments.map((segment) => segment.id);
  state.active_segment_id = state.project.segments[0]?.id ?? null;
  state.selection_anchor_id = state.project.segments[0].id;
  state.asset_grid_size = layout.size;
  return state;
}

function runtimeTimelineValidation(
  project: TimelineProject,
  capabilities: CapabilityReport,
  settings: RuntimeSettings,
  segmentIds?: readonly string[],
): string[] {
  const errors: string[] = [];
  const selection = segmentIds ? new Set(segmentIds) : null;
  const selectedFamilies = new Set(
    project.segments
      .filter((segment) => segment.enabled && (!selection || selection.has(segment.id)))
      .map((segment) => segment.mode),
  );
  const hasSelectedContinuity = project.segments.some((segment) =>
    segment.enabled &&
    segment.continuity.enabled &&
    (!selection || selection.has(segment.id)),
  );
  const nativeTimeline = capabilities.native_timeline;
  if (hasSelectedContinuity && (
    nativeTimeline?.supported !== true ||
    nativeTimeline.continuity !== true ||
    [...selectedFamilies].some((family) => !nativeTimeline.modes.includes(family))
  )) {
    errors.push("当前原生分段子图不支持所选片段的段间接续；请关闭这些片段的连续性设置");
  }
  if (capabilities.connection === "online") {
    if (capabilities.execution_backends) {
      for (const family of selectedFamilies) {
        const binding = settings.models[family];
        const backend = resolveExecutionBackend(binding);
        const status = capabilities.execution_backends[backend];
        if (!status || !status.available) {
          errors.push(`${family.toUpperCase()} 配置解析为 ${backend === "raylight" ? "RayLight" : "标准"}执行，但当前 ComfyUI 不可用${status?.missing_nodes.length ? `：缺少 ${status.missing_nodes.join("、")}` : ""}`);
        } else if (
          backend === "raylight" &&
          binding.lora_name &&
          status.conditional_requirements?.lora.available === false
        ) {
          const missing = status.conditional_requirements.lora.missing_nodes;
          errors.push(`${family.toUpperCase()} 的 RayLight LoRA 配置不可用${missing.length ? `：缺少 ${missing.join("、")}` : ""}`);
        }
      }
    } else {
      errors.push("当前 ComfyUI 未报告 Standard / RayLight 执行后端能力，请刷新能力或升级后端");
    }
  }
  return errors;
}

const ACTIVE_PROJECT_ID_STORAGE_KEY = "directordeck:v1:active-project-id";

function loadActiveProjectId(): string {
  try {
    const raw = window.localStorage.getItem(ACTIVE_PROJECT_ID_STORAGE_KEY);
    return typeof raw === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(raw)
      ? raw
      : DEFAULT_PROJECT_ID;
  } catch {
    return DEFAULT_PROJECT_ID;
  }
}

function persistActiveProjectId(projectId: string): void {
  try {
    window.localStorage.setItem(ACTIVE_PROJECT_ID_STORAGE_KEY, projectId);
  } catch {
    // Active-project selection is a UI preference; in-memory value remains usable.
  }
}

// Route timeline reads/writes to the legacy singleton endpoints for the
// default project and to project-scoped endpoints for everything else. This
// keeps the pre-multi-project contract intact for the first project while
// giving created projects their own server-owned identity.
function fetchTimelineForProject(projectId: string, signal?: AbortSignal): Promise<TimelineAuthority> {
  return projectId === DEFAULT_PROJECT_ID
    ? directorApi.getTimelineAuthority(signal)
    : directorApi.getProjectTimelineAuthority(projectId, signal);
}

function saveTimelineForProject(
  projectId: string,
  project: TimelineProject,
  expectedRevision: number,
): Promise<TimelineAuthority> {
  return projectId === DEFAULT_PROJECT_ID
    ? directorApi.updateTimelineAuthority(project, expectedRevision)
    : directorApi.updateProjectTimelineAuthority(projectId, project, expectedRevision);
}

function compileTimelineForProject(
  projectId: string,
  payload: { config: TimelineProject; segment_ids?: string[] },
): Promise<TimelineCompileReport> {
  return projectId === DEFAULT_PROJECT_ID
    ? directorApi.compileTimeline(payload)
    : directorApi.compileProjectTimeline(projectId, payload);
}

function submitTimelineForProject(
  projectId: string,
  payload: { config: TimelineProject; segment_ids?: string[] },
): Promise<GenerationTask> {
  return projectId === DEFAULT_PROJECT_ID
    ? directorApi.createTimelineTask(payload)
    : directorApi.createProjectTask(projectId, payload);
}

export default function App() {
  const [state, dispatch] = useReducer(directorReducer, undefined, loadDirectorState);
  const [timeline, rawTimelineDispatch] = useReducer(
    timelineEditorReducer,
    undefined,
    createInitialTimelineState,
  );
  const [timelineHistory, setTimelineHistoryState] = useState(createTimelineHistory);
  const [timelineHistoryAnnouncement, setTimelineHistoryAnnouncement] = useState({
    sequence: 0,
    message: "",
  });
  const [capabilities, setCapabilities] = useState<CapabilityReport>({
    ...EMPTY_CAPABILITIES,
    connection: "checking",
  });
  const [gpus, setGpus] = useState<GPUResource[]>([]);
  const [models, setModels] = useState<ModelInventory>(EMPTY_MODELS);
  const [rayLightRuntimeStatus, setRayLightRuntimeStatus] = useState<RayLightRuntimeStatus | null>(null);
  const [rayLightRecoveryPending, setRayLightRecoveryPending] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [runtimeResourcesReady, setRuntimeResourcesReady] = useState(false);
  const [runtimeSettingsOperationOwner, setRuntimeSettingsOperationOwner] = useState<RuntimeSettingsOperationOwner | null>(null);
  const [runtimeSettingsSyncRequired, setRuntimeSettingsSyncRequired] = useState(false);
  const [runtimeSettingsDraft, setRuntimeSettingsDraft] = useState<RuntimeSettings>(() => state.settings);
  const [runtimeSettingsDraftValid, setRuntimeSettingsDraftValid] = useState(true);
  const [runtimeSettingsPausedError, setRuntimeSettingsPausedError] = useState<string | null>(null);
  const runtimeSettingsDraftValidRef = useRef(true);
  const [timelineDirty, setTimelineDirty] = useState(false);
  const [timelineHydrationStatus, setTimelineHydrationStatus] = useState<
    "loading" | "retrying" | "stale" | "ready"
  >("loading");
  const [timelineHydrationEpoch, setTimelineHydrationEpoch] = useState(0);
  const [timelinePausedError, setTimelinePausedError] = useState<{
    revision: number;
    message: string;
  } | null>(null);
  const [timelineRetryNonce, setTimelineRetryNonce] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [compiling, setCompiling] = useState(false);
  const [compileReport, setCompileReport] = useState<TimelineCompileReport | null>(null);
  const [tasksLoading, setTasksLoading] = useState(false);
  const [assetsDeleting, setAssetsDeleting] = useState(false);
  const [assetsUploading, setAssetsUploading] = useState(false);
  const [assetUploadProgress, setAssetUploadProgress] = useState<DroppedUploadProgress | null>(null);
  const [timelineSyncRequired, setTimelineSyncRequired] = useState(false);
  const [timelineRevisionConflict, setTimelineRevisionConflictState] = useState<TimelineRevisionConflict | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => initialUiToggle(SIDEBAR_OPEN_KEY, true));
  const [sidebarViewportWidth, setSidebarViewportWidth] = useState(() => window.innerWidth);
  const [sidebarWidth, setSidebarWidth] = useState(initialSidebarWidth);
  const [globalSettingsOpen, setGlobalSettingsOpen] = useState(false);
  const [timelineHistoryPanelOpen, setTimelineHistoryPanelOpen] = useState(false);
  const [assetTrashPanelOpen, setAssetTrashPanelOpen] = useState(false);
  const [assetTrashBatches, setAssetTrashBatches] = useState<AssetTrashBatch[]>([]);
  const [assetTrashLoading, setAssetTrashLoading] = useState(false);
  const [assetTrashBusyBatchId, setAssetTrashBusyBatchId] = useState<string | null>(null);
  const [assetTrashConflictBatchIds, setAssetTrashConflictBatchIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const [projectTitleEditing, setProjectTitleEditing] = useState(false);
  const [projectTitleDraft, setProjectTitleDraft] = useState("");
  const [theme, setTheme] = useState(readUiTheme);
  const [deletingTaskIds, setDeletingTaskIds] = useState<ReadonlySet<string>>(() => new Set());
  const [clearingTasks, setClearingTasks] = useState(false);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [activeProjectId, setActiveProjectIdState] = useState<string>(loadActiveProjectId);
  const [projectSwitchHandoffPending, setProjectSwitchHandoffPending] = useState(false);
  const [projectDeletingId, setProjectDeletingId] = useState<string | null>(null);
  const activeProjectIdRef = useRef(activeProjectId);
  const runtimeRequest = useRef(0);
  const runtimeResourceRequest = useRef(0);
  const runtimeResourceRetryTimer = useRef<number | null>(null);
  const runtimeResourceRefreshRef = useRef<(preserveExisting: boolean) => Promise<boolean>>(
    async () => false,
  );
  const externalRuntimeAuthorityRefreshRef = useRef<() => void>(() => undefined);
  const externalRuntimeAuthorityOperationRef = useRef<Promise<void> | null>(null);
  const externalRuntimeAuthorityControllerRef = useRef<AbortController | null>(null);
  const externalRuntimeAuthorityRetryTimer = useRef<number | null>(null);
  const runtimeResourcesReadyRef = useRef(false);
  const runtimeResourcesAuthorityTokenRef = useRef<string | null>(null);
  const rayLightRuntimeStatusRef = useRef<RayLightRuntimeStatus | null>(null);
  const rayLightRecoveryOperationRef = useRef<Promise<void> | null>(null);
  const rayLightRecoveryControllerRef = useRef<AbortController | null>(null);
  const rayLightRecoveryPendingRef = useRef(false);
  const runtimeSettingsOperation = useRef<Promise<RuntimeSettings> | null>(null);
  const runtimeSettingsDesired = useRef<RuntimeSettings | null>(null);
  const runtimeSettingsDesiredOwner = useRef<RuntimeSettingsOperationOwner>("settings-page");
  const runtimeSettingsAutosaveTimer = useRef<number | null>(null);
  const runtimeSettingsRetryTimer = useRef<number | null>(null);
  const runtimeSettingsDrainRef = useRef<() => void>(() => undefined);
  const runtimeSettingsPausedDesiredRef = useRef<RuntimeSettings | null>(null);
  const runtimeSettingsGeneration = useRef(0);
  const runtimeSettingsSyncRequiredRef = useRef(false);
  const runtimeSettingsAuthorityReadyRef = useRef(false);
  const runtimeExecutionIntent = useRef(0);
  const authoritativeSettingsRef = useRef(state.settings);
  const authoritativeSettingsTokenRef = useRef<string | null>(null);
  const assetAuthorityRequired = useRef(false);
  const assetListRequest = useRef(0);
  const taskDeleteLocks = useRef(new Set<string>());
  const taskClearLock = useRef(false);
  const taskListRequest = useRef(0);
  const taskListInFlight = useRef<Promise<void> | null>(null);
  const taskListRefreshQueued = useRef(false);
  const taskListOwnerActive = useRef(false);
  const globalSettingsToggleRef = useRef<HTMLButtonElement>(null);
  const timelineHistoryToggleRef = useRef<HTMLButtonElement>(null);
  const assetTrashToggleRef = useRef<HTMLButtonElement>(null);
  const projectTitleInputRef = useRef<HTMLInputElement>(null);
  const sidebarBrandToggleRef = useRef<HTMLButtonElement>(null);
  const settingsToggleRef = useRef<HTMLButtonElement>(null);
  const timelineRevision = useRef(0);
  const timelineServerRevision = useRef<number | null>(null);
  const timelineServerProjectRef = useRef<TimelineProject | null>(null);
  const timelineBranchOwnerRef = useRef(INITIAL_TIMELINE_BRANCH_OWNER_ID);
  const activeTimelineWalRef = useRef<LocalTimelineWal | null>(null);
  const timelineJournalGeneration = useRef(0);
  const timelineJournalChain = useRef<Promise<void>>(Promise.resolve());
  const timelineJournalTokens = useRef(new Map<string, TimelineHistoryJournalVersionToken>());
  const timelineHydrationGeneration = useRef(0);
  const timelineHydrationReady = useRef(false);
  const activeDatabaseRef = useRef<ActiveDatabaseIdentity | null>(null);
  const databaseIdentityStaleRef = useRef(false);
  const segmentSelectionGeneration = useRef(0);
  const restoredSegmentSelectionKey = useRef<string | null>(null);
  const projectSwitchGeneration = useRef(0);
  const projectListGeneration = useRef(0);
  const projectDeleteIntent = useRef<string | null>(null);
  const timelinePersistedRevision = useRef(0);
  const timelineWriteGeneration = useRef(0);
  const timelineSaveRequest = useRef<Promise<TimelineProject | null> | null>(null);
  const timelineSaveRequestRevision = useRef<number | null>(null);
  const timelineAutosaveTimer = useRef<number | null>(null);
  const timelineRetryTimer = useRef<number | null>(null);
  const timelineAuthorityRetryTimer = useRef<number | null>(null);
  const timelineSyncRequiredRef = useRef(false);
  const timelineRevisionConflictRef = useRef<TimelineRevisionConflict | null>(null);
  const timelineRevisionChannelRef = useRef<TimelineRevisionChannel | null>(null);
  const timelineRemoteAuthorityRequest = useRef(0);
  const projectSwitchHandoffIntent = useRef<number | null>(null);
  const timelineRenderedRevision = useRef(timelineRevision.current);
  const flushTimelineAutosaveRef = useRef<() => Promise<TimelineProject>>(
    async () => { throw new Error("时间线自动保存尚未初始化"); },
  );
  const assetDeleteLock = useRef(false);
  const assetDeleteIntent = useRef(false);
  const assetUploadLock = useRef(false);
  const assetTrashOperationLock = useRef<string | null>(null);
  const assetTrashListRequest = useRef(0);
  const timelineRef = useRef(timeline);
  const timelineHistoryRef = useRef(timelineHistory);
  const timelineTextCompositionActive = useRef(false);
  const timelineTextBeforeInput = useRef<TimelineTextEditingContext | null>(null);
  const timelineHadLocal = useRef(false);

  useLayoutEffect(() => {
    // Reducer dispatches maintain their own synchronous command shadow. Only a
    // committed render may publish React's copy back into the async owner.
    timelineRef.current = timeline;
    timelineRenderedRevision.current = timelineRevision.current;
  }, [timeline]);

  const setSidebarOpenWithFocus = useCallback((open: boolean) => {
    setSidebarOpen(open);
    window.requestAnimationFrame(() => sidebarBrandToggleRef.current?.focus());
  }, []);

  const commitTimelineHistory = useCallback((history: TimelineHistoryState) => {
    timelineHistoryRef.current = history;
    setTimelineHistoryState(history);
  }, []);

  const enqueueTimelineJournalOperation = useCallback((
    operation: (generation: number) => Promise<unknown>,
  ) => {
    const generation = ++timelineJournalGeneration.current;
    // IndexedDB puts are serialized as well as generation-guarded. A stale put
    // that already started may finish, but the latest queued mutation always
    // runs after it and therefore owns the durable value.
    timelineJournalChain.current = timelineJournalChain.current
      .catch(() => undefined)
      .then(async () => {
        if (generation !== timelineJournalGeneration.current) return;
        try {
          await operation(generation);
        } catch {
          // localStorage WAL remains the synchronous crash-recovery boundary.
        }
      });
  }, []);

  const persistTimelineJournal = useCallback((
    history: TimelineHistoryState,
    authority: TimelinePersistenceAuthority,
    database: ActiveDatabaseIdentity,
    projectId: string,
  ) => {
    const ownerId = timelineBranchOwnerRef.current;
    const scope = timelinePersistenceScope(database, projectId, ownerId);
    const tokenKey = timelineJournalTokenMapKey(database, projectId, ownerId);
    enqueueTimelineJournalOperation(async (generation) => {
      if (history.head) {
        const token = await saveTimelineHistoryJournal(scope, authority, history);
        if (generation === timelineJournalGeneration.current && token) {
          timelineJournalTokens.current.set(tokenKey, token);
        }
        return;
      }
      const token = timelineJournalTokens.current.get(tokenKey) ??
        await readTimelineHistoryJournalVersionToken(scope);
      if (!token) return;
      const deleted = await deleteTimelineHistoryJournal(scope, token);
      if (generation === timelineJournalGeneration.current && deleted) {
        timelineJournalTokens.current.delete(tokenKey);
      }
    });
  }, [enqueueTimelineJournalOperation]);

  const acceptTimelineServerAuthority = useCallback((authority: TimelineAuthority) => {
    // Every local authority owner invalidates Broadcast-triggered GETs that
    // started against an older base. Their late responses may never regress
    // the revision/document pair accepted here.
    timelineRemoteAuthorityRequest.current += 1;
    timelineServerRevision.current = authority.revision;
    timelineServerProjectRef.current = structuredClone(authority.document);
  }, []);

  // Keep the existing call shape inside mutation owners, including the asset
  // compensation paths. The implementation is now fail-closed unless an exact
  // server base has been latched for the active project.
  const saveLocalTimeline = useCallback((
    project: TimelineProject,
    database: ActiveDatabaseIdentity,
    projectId: string = DEFAULT_PROJECT_ID,
  ): LocalTimelineWal | null => {
    const baseRevision = timelineServerRevision.current;
    const baseProject = timelineServerProjectRef.current;
    if (baseRevision === null || !baseProject || projectId !== activeProjectIdRef.current) {
      return null;
    }
    const wal = saveLocalTimelineWal({
      database,
      project_id: projectId,
      base_server_revision: baseRevision,
      base_project: baseProject,
      pending_project: project,
      owner_id: timelineBranchOwnerRef.current,
    });
    if (wal) activeTimelineWalRef.current = wal;
    else if (timelineProjectsEqual(project, baseProject)) activeTimelineWalRef.current = null;
    return wal;
  }, []);

  const clearLocalTimeline = useCallback((wal?: LocalTimelineWal) => {
    const target = wal ?? activeTimelineWalRef.current;
    // No target is not authority to delete the domain module's last observed
    // branch: it may belong to the previous project after a project switch.
    if (!target) return;
    clearLocalTimelineWal(target);
    if (!wal || activeTimelineWalRef.current === wal) activeTimelineWalRef.current = null;
  }, []);

  const clearTimelineHistory = useCallback(() => {
    commitTimelineHistory(resetTimelineHistory(timelineHistoryRef.current));
    setTimelineHistoryAnnouncement((current) => ({
      sequence: current.sequence + 1,
      message: "",
    }));
  }, [commitTimelineHistory]);

  const updateProjectSummaryTitle = useCallback((projectId: string, title: string) => {
    // A list response captured before this local mutation must not restore the
    // old title (or make a later membership decision against that old list).
    projectListGeneration.current += 1;
    setProjects((projects) => {
      let changed = false;
      const next = projects.map((project) => {
        if (project.id !== projectId || project.title === title) return project;
        changed = true;
        return { ...project, title };
      });
      return changed ? next : projects;
    });
  }, []);

  const setRuntimeAuthorityRequired = useCallback((required: boolean) => {
    runtimeSettingsSyncRequiredRef.current = required;
    setRuntimeSettingsSyncRequired(required);
  }, []);

  const setTimelineAuthorityRequired = useCallback((required: boolean) => {
    timelineSyncRequiredRef.current = required;
    setTimelineSyncRequired(required);
  }, []);

  const setTimelineRevisionConflict = useCallback((conflict: TimelineRevisionConflict | null) => {
    timelineRevisionConflictRef.current = conflict;
    setTimelineRevisionConflictState(conflict);
  }, []);

  const restartTimelineHydrationForProject = useCallback((projectId: string) => {
    if (projectId === activeProjectIdRef.current) return;

    const previousProjectId = activeProjectIdRef.current;
    const database = activeDatabaseRef.current;
    if (
      database &&
      timelinePersistedRevision.current < timelineRevision.current
    ) {
      saveLocalTimeline(timelineRef.current.project, database, previousProjectId);
    }

    // Invalidate every owner that captured the previous project before changing
    // the synchronous project id. The next hydration effect establishes a fresh
    // server revision/document pair for the fallback project.
    timelineHydrationGeneration.current += 1;
    timelineWriteGeneration.current += 1;
    timelineJournalGeneration.current += 1;
    projectSwitchGeneration.current += 1;
    projectListGeneration.current += 1;
    timelineRemoteAuthorityRequest.current += 1;
    // The old promise remains generation-guarded and cleans up only when it
    // still owns the slot. Detach it now so the fallback project can start its
    // own autosave without waiting for an unrelated project response.
    timelineSaveRequest.current = null;
    timelineSaveRequestRevision.current = null;
    if (timelineAutosaveTimer.current !== null) {
      window.clearTimeout(timelineAutosaveTimer.current);
      timelineAutosaveTimer.current = null;
    }
    if (timelineRetryTimer.current !== null) {
      window.clearTimeout(timelineRetryTimer.current);
      timelineRetryTimer.current = null;
    }
    if (timelineAuthorityRetryTimer.current !== null) {
      window.clearTimeout(timelineAuthorityRetryTimer.current);
      timelineAuthorityRetryTimer.current = null;
    }

    activeProjectIdRef.current = projectId;
    setActiveProjectIdState(projectId);
    persistActiveProjectId(projectId);
    timelineServerRevision.current = null;
    timelineServerProjectRef.current = null;
    activeTimelineWalRef.current = null;
    timelineRevision.current = 0;
    timelinePersistedRevision.current = 0;
    timelineHadLocal.current = false;
    timelineHydrationReady.current = false;
    restoredSegmentSelectionKey.current = null;
    segmentSelectionGeneration.current += 1;
    setTimelineRevisionConflict(null);
    setTimelineAuthorityRequired(false);
    setTimelineDirty(false);
    setTimelinePausedError(null);
    setCompileReport(null);
    clearTimelineHistory();
    setTimelineHydrationStatus("loading");
    setTimelineHydrationEpoch((current) => current + 1);
  }, [clearTimelineHistory, saveLocalTimeline, setTimelineAuthorityRequired, setTimelineRevisionConflict]);

  const dispatchTimeline = useCallback((
    action: TimelineAction,
    options: TimelineDispatchOptions = {},
  ): boolean => {
    const current = timelineRef.current;
    const historyMode = options.history ?? "record";
    const transaction = reduceTimelineTransaction(
      current,
      action,
      historyMode === "replay" ? "replay" : undefined,
    );
    const next = transaction.next;
    const projectChanged = transaction.documentChanged;
    if (!timelineHydrationReady.current && projectChanged) {
      setToast("正在从服务器恢复时间线；恢复完成前暂不能编辑");
      return false;
    }
    if (projectSwitchHandoffIntent.current !== null && projectChanged) {
      setToast("正在完成项目切换交接；完成前不能编辑当前项目");
      return false;
    }
    if ((assetDeleteLock.current || assetDeleteIntent.current || timelineSyncRequiredRef.current || timelineRevisionConflictRef.current) && projectChanged) {
      setToast(assetDeleteIntent.current
          ? "正在建立素材移出的原子边界；完成后再编辑时间线"
        : timelineRevisionConflictRef.current
          ? "检测到服务器时间线已被其他页面修改；请先选择采用服务器版本或保留本地版本"
        : timelineSyncRequiredRef.current
          ? "服务器时间线尚未完成权威回读；恢复同步前暂不能编辑"
          : "正在原子解除素材引用，完成后再编辑时间线");
      return false;
    }
    if (projectChanged && historyMode === "record") {
      const restoreEditingContext = transaction.policy.context === "structural";
      const captureTextContext = transaction.policy.context === "text";
      const fieldKey = transaction.policy.mergeKey;
      const capturedActiveText = captureTextContext
        ? captureTimelineTextEditingContext(document.activeElement)
        : null;
      const activeText = capturedActiveText &&
        (!fieldKey || capturedActiveText.field_key === fieldKey)
        ? capturedActiveText
        : null;
      const pendingBeforeText = timelineTextBeforeInput.current;
      const beforeText = captureTextContext
        ? (pendingBeforeText && (!fieldKey || pendingBeforeText.field_key === fieldKey)
            ? pendingBeforeText
            : activeText)
        : null;
      let afterText = activeText;
      if (beforeText && fieldKey && beforeText.field_key === fieldKey) {
        const segment = current.project.segments.find((candidate) =>
          "id" in action && action.id === candidate.id);
        const nextSegment = next.project.segments.find((candidate) =>
          candidate.id === segment?.id);
        const changedField = fieldKey.endsWith(":prompt")
          ? "prompt"
          : fieldKey.endsWith(":title")
            ? "title"
            : null;
        if (segment && nextSegment && changedField) {
          const control = timelineHistoryTextControl(document.activeElement);
          const nextValue = nextSegment[changedField];
          if (!control || control.value !== nextValue) {
            afterText = collapsedTimelineTextContextAfter(
              beforeText,
              segment[changedField],
              nextValue,
            );
          }
        }
      }
      commitTimelineHistory(recordTimelineHistory(timelineHistoryRef.current, {
        label: options.historyLabel ?? transaction.policy.label,
        before: current.project,
        after: next.project,
        beforeContext: restoreEditingContext
          ? captureTimelineHistoryContext(current)
          : beforeText
            ? timelineTextHistoryContext(current, beforeText)
          : undefined,
        afterContext: restoreEditingContext
          ? captureTimelineHistoryContext(next)
          : afterText
            ? timelineTextHistoryContext(next, afterText)
          : undefined,
        mergeKey: options.historyMergeKey ?? transaction.policy.mergeKey,
        coalesceWindowMs: timelineTextCompositionActive.current
          ? Number.MAX_SAFE_INTEGER
          : undefined,
      }));
      if (captureTextContext) timelineTextBeforeInput.current = null;
    } else if (historyMode === "reset") {
      clearTimelineHistory();
    } else if (
      historyMode !== "replay" &&
      transaction.policy.coalescing !== "preserve"
    ) {
      commitTimelineHistory(sealTimelineHistoryCoalescing(timelineHistoryRef.current));
    }
    // Keep an optimistic reducer shadow so several actions batched before the
    // next render are compared and persisted in their exact order.
    timelineRef.current = next;
    if (projectChanged) {
      if (next.project.title !== current.project.title) {
        updateProjectSummaryTitle(activeProjectIdRef.current, next.project.title);
      }
      timelineRevision.current += 1;
      // A deterministic server rejection belongs only to the exact revision
      // that was submitted. Any real project edit supersedes it and schedules
      // the newly corrected revision through the normal autosave queue.
      setTimelinePausedError(null);
      setTimelineDirty(true);
      setCompileReport(null);
      taskListRequest.current += 1;
      dispatch({ type: "tasks/invalidate-current-snapshots" });
    }
    if (transaction.runnableSelectionChanged) {
      segmentSelectionGeneration.current += 1;
      setCompileReport(null);
    }
    const activeDatabase = activeDatabaseRef.current;
    const selectionPreferenceScope = activeDatabase
      ? `${activeDatabase.active_database_path}:${activeProjectIdRef.current}`
      : null;
    if (
      activeDatabase &&
      restoredSegmentSelectionKey.current === selectionPreferenceScope &&
      (transaction.topologyChanged || transaction.selectionChanged)
    ) {
      saveTimelineSegmentSelectionPreference(
        activeDatabase,
        activeProjectIdRef.current,
        next.project.segments.map((segment) => segment.id),
        next.selected_segment_ids,
      );
    }
    rawTimelineDispatch(action);
    if (transaction.derivedAdjustments.length) {
      rawTimelineDispatch({ type: "project/replace", project: next.project });
      const first = transaction.derivedAdjustments[0];
      const label = first.segment_title || first.segment_id;
      const seconds = first.source_frames_after / next.project.render.fps;
      const omitted = Math.max(0, first.source_frames_before - first.source_frames_after);
      const detail = first.fallback_to_previous_h3_length
        ? `源素材不足 ${first.output_frames_before} 帧，已自动缩短为 ${first.source_frames_after} 帧（${seconds.toFixed(4)} 秒）${omitted ? `，省略末尾 ${omitted} 帧` : ""}`
        : `已自动将源截取从 ${first.source_frames_before} 帧适配为 ${first.source_frames_after} 帧（${seconds.toFixed(4)} 秒）`;
      setToast(`${label}：${detail}${transaction.derivedAdjustments.length > 1 ? `；另有 ${transaction.derivedAdjustments.length - 1} 个片段已自动适配` : ""}`);
    }
    return true;
  }, [clearTimelineHistory, commitTimelineHistory, updateProjectSummaryTitle]);

  const installTimelineAuthority = useCallback((
    project: TimelineProject,
    options: {
      projectId?: string;
      selectedSegmentIds?: readonly string[];
      clearHistory?: boolean;
    } = {},
  ): TimelineEditorState => {
    const current = timelineRef.current;
    const replaceAction: TimelineAction = { type: "project/replace", project };
    let next = reduceTimelineTransaction(current, replaceAction, "authority").next;
    const selectedSegmentIds = options.selectedSegmentIds;
    let selectionAction: TimelineAction | null = null;
    if (selectedSegmentIds !== undefined) {
      selectionAction = {
        type: "segment/set-selection",
        ids: [...selectedSegmentIds],
      };
      next = reduceTimelineTransaction(next, selectionAction).next;
    }
    // This is the only non-user path allowed to install a server-owned
    // document. Keep the command shadow and React reducer in the same order.
    timelineRef.current = next;
    if (selectionAction) {
      // Install project + explicit project-scoped selection atomically. Two
      // separate reducer actions can expose the reused segment IDs from the
      // previous project to a concurrent committed render.
      rawTimelineDispatch({
        type: "history/restore",
        project: next.project,
        selected_segment_ids: [...next.selected_segment_ids],
        active_segment_id: next.active_segment_id,
        selection_anchor_id: next.selection_anchor_id,
      });
    } else {
      rawTimelineDispatch(replaceAction);
    }
    updateProjectSummaryTitle(
      options.projectId ?? activeProjectIdRef.current,
      project.title,
    );
    if (options.clearHistory ?? true) clearTimelineHistory();
    return next;
  }, [clearTimelineHistory, updateProjectSummaryTitle]);

  const dispatchTimelineUi = useCallback((action: TimelineAction): boolean =>
    dispatchTimeline(action, { history: "skip" }), [dispatchTimeline]);

  // Persist the same mechanical correction for timelines loaded from older
  // saves. Without this hydration pass, a legacy mismatch could disable the
  // run buttons before the user performs any edit that normally triggers the
  // reducer-side fit.
  useEffect(() => {
    if (timelineHydrationStatus !== "ready") return;
    if (!autoFitSourceAudioTiming(timeline.project).adjustments.length) return;
    dispatchTimeline(
      { type: "project/replace", project: timeline.project },
      { history: "skip" },
    );
  }, [dispatchTimeline, timeline.project, timelineHydrationStatus]);

  const applyTimelineHistoryReplay = useCallback((
    replay: TimelineHistoryReplay | null,
    announcement: string,
  ) => {
    if (!replay) return;
    const current = timelineRef.current;
    const storedContext = replay.snapshot.context;
    const currentContext = captureTimelineHistoryContext(current);
    const restoreSegmentSelection = storedContext?.restore_segment_selection !== false;
    const context = restoreSegmentSelection && storedContext
      ? storedContext
      : currentContext;
    const sourceFieldKey = captureTimelineTextEditingContext(document.activeElement)?.field_key ?? null;
    const applied = dispatchTimeline({
      type: "history/restore",
      project: replay.snapshot.project,
      selected_segment_ids: context.selected_segment_ids,
      active_segment_id: context.active_segment_id,
      selection_anchor_id: context.selection_anchor_id,
    }, { history: "replay" });
    if (!applied) return;
    commitTimelineHistory(replay.history);
    setTimelineHistoryAnnouncement((currentAnnouncement) => ({
      sequence: currentAnnouncement.sequence + 1,
      message: announcement,
    }));
    restoreTimelineTextEditingSelection(storedContext?.text_editing, sourceFieldKey);
  }, [commitTimelineHistory, dispatchTimeline]);

  const replayTimelineHistory = useCallback((direction: "undo" | "redo") => {
    const replay = direction === "undo"
      ? undoTimelineHistory(timelineHistoryRef.current)
      : redoTimelineHistory(timelineHistoryRef.current);
    applyTimelineHistoryReplay(
      replay,
      replay
        ? direction === "undo"
          ? `已撤销：${replay.label}`
          : `已重做：${replay.label}`
        : "",
    );
  }, [applyTimelineHistoryReplay]);

  const undoTimeline = useCallback(() => {
    replayTimelineHistory("undo");
  }, [replayTimelineHistory]);

  const redoTimeline = useCallback(() => {
    replayTimelineHistory("redo");
  }, [replayTimelineHistory]);

  const jumpTimelineHistoryCursor = useCallback((cursor: number) => {
    const sourceFieldKey = captureTimelineTextEditingContext(document.activeElement)?.field_key ?? null;
    const replay = jumpTimelineHistory(timelineHistoryRef.current, cursor, sourceFieldKey);
    applyTimelineHistoryReplay(replay, replay ? `已跳转到：${replay.label}` : "");
  }, [applyTimelineHistoryReplay]);

  useEffect(() => {
    const sealHistoryInputSession = () => {
      const sealed = sealTimelineHistoryCoalescing(timelineHistoryRef.current);
      if (sealed !== timelineHistoryRef.current) commitTimelineHistory(sealed);
    };
    const sealTimelineTextBoundary = (event: Event) => {
      if (
        isTimelineHistoryTextTarget(event.target) &&
        !timelineTextCompositionActive.current
      ) {
        timelineTextBeforeInput.current = null;
        sealHistoryInputSession();
      }
    };
    const sealTimelineTextNavigation = (event: KeyboardEvent) => {
      if (
        isTimelineHistoryTextTarget(event.target) &&
        !timelineTextCompositionActive.current &&
        !event.isComposing &&
        event.keyCode !== 229 &&
        ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(event.key)
      ) {
        timelineTextBeforeInput.current = null;
        sealHistoryInputSession();
      }
    };
    const sealNonIncrementalInput = (event: Event) => {
      if (!isTimelineHistoryTextTarget(event.target)) return;
      const inputEvent = event as InputEvent;
      if (timelineTextCompositionActive.current || inputEvent.isComposing) return;
      const inputType = inputEvent.inputType;
      if (inputType === "historyUndo" || inputType === "historyRedo") {
        event.preventDefault();
        if (inputType === "historyUndo") undoTimeline();
        else redoTimeline();
        return;
      }
      timelineTextBeforeInput.current = captureTimelineTextEditingContext(event.target);
      if ([
        "insertText",
        "insertCompositionText",
        "deleteContentBackward",
        "deleteContentForward",
      ].includes(inputType)) return;
      sealHistoryInputSession();
      window.queueMicrotask(sealHistoryInputSession);
    };
    const beginComposition = (event: Event) => {
      if (!isTimelineHistoryTextTarget(event.target)) return;
      timelineTextBeforeInput.current = captureTimelineTextEditingContext(event.target);
      timelineTextCompositionActive.current = true;
      sealHistoryInputSession();
    };
    const endComposition = (event: Event) => {
      if (!isTimelineHistoryTextTarget(event.target)) return;
      window.queueMicrotask(() => {
        timelineTextCompositionActive.current = false;
        sealHistoryInputSession();
      });
    };
    document.addEventListener("focusin", sealHistoryInputSession);
    document.addEventListener("pointerdown", sealTimelineTextBoundary);
    document.addEventListener("select", sealTimelineTextBoundary);
    document.addEventListener("keydown", sealTimelineTextNavigation);
    document.addEventListener("beforeinput", sealNonIncrementalInput);
    document.addEventListener("compositionstart", beginComposition);
    document.addEventListener("compositionend", endComposition);
    return () => {
      document.removeEventListener("focusin", sealHistoryInputSession);
      document.removeEventListener("pointerdown", sealTimelineTextBoundary);
      document.removeEventListener("select", sealTimelineTextBoundary);
      document.removeEventListener("keydown", sealTimelineTextNavigation);
      document.removeEventListener("beforeinput", sealNonIncrementalInput);
      document.removeEventListener("compositionstart", beginComposition);
      document.removeEventListener("compositionend", endComposition);
      timelineTextCompositionActive.current = false;
    };
  }, [commitTimelineHistory, redoTimeline, undoTimeline]);

  useEffect(() => {
    const handleTimelineHistoryShortcut = (event: KeyboardEvent) => {
      if (
        state.view !== "workspace" ||
        event.defaultPrevented ||
        timelineTextCompositionActive.current ||
        event.isComposing ||
        event.keyCode === 229 ||
        event.altKey ||
        (!event.ctrlKey && !event.metaKey)
      ) return;
      const key = event.key.toLowerCase();
      const undo = key === "z" && !event.shiftKey;
      const redo = (key === "z" && event.shiftKey) ||
        (key === "y" && event.ctrlKey && !event.metaKey && !event.shiftKey);
      if ((!undo && !redo) || shouldKeepNativeUndo(event.target)) return;
      const projectTextTarget = isTimelineHistoryTextTarget(event.target);
      if (
        (undo && !canUndoTimelineHistory(timelineHistoryRef.current)) ||
        (redo && !canRedoTimelineHistory(timelineHistoryRef.current))
      ) {
        if (projectTextTarget) event.preventDefault();
        return;
      }
      event.preventDefault();
      if (undo) undoTimeline();
      else redoTimeline();
    };
    window.addEventListener("keydown", handleTimelineHistoryShortcut);
    return () => window.removeEventListener("keydown", handleTimelineHistoryShortcut);
  }, [redoTimeline, state.view, undoTimeline]);

  const loadTasks = useCallback(async (signal?: AbortSignal, queueIfBusy = false) => {
    if (taskListInFlight.current) {
      if (queueIfBusy) taskListRefreshQueued.current = true;
      return taskListInFlight.current;
    }
    const requestId = ++taskListRequest.current;
    setTasksLoading(true);
    const operation = (async () => {
      try {
        const result = await directorApi.listTasks(signal, activeProjectIdRef.current);
        if (signal?.aborted || taskListRequest.current !== requestId) return;
        dispatch({ type: "tasks/replace", tasks: result.jobs });
      } catch {
        // Offline editing remains available.
      } finally {
        if (taskListRequest.current === requestId) setTasksLoading(false);
      }
    })();
    taskListInFlight.current = operation;
    try {
      await operation;
    } finally {
      if (taskListInFlight.current === operation) taskListInFlight.current = null;
      if (taskListRefreshQueued.current && taskListOwnerActive.current) {
        taskListRefreshQueued.current = false;
        await loadTasks(undefined, false);
      }
    }
  }, []);

  const invalidateAndRefreshTaskSnapshots = useCallback(() => {
    // A task-list response started before a timeline/settings write carries
    // current_snapshot flags for the old authorities. Suppress those flags
    // immediately, invalidate the in-flight response, then fetch fresh server
    // comparisons. If refresh fails, fail closed and keep historical takes in
    // the drawer without putting them back into the main monitor.
    taskListRequest.current += 1;
    dispatch({ type: "tasks/invalidate-current-snapshots" });
    void loadTasks(undefined, true);
  }, [loadTasks]);

  useEffect(() => {
    if (timelineHydrationStatus !== "ready") return;
    const database = activeDatabaseRef.current;
    const serverRevision = timelineServerRevision.current;
    if (!database || serverRevision === null) return;
    const projectId = activeProjectId;
    const scope = {
      databasePath: database.active_database_path,
      projectId,
    };
    // A legacy WAL deliberately remains local until the user resolves it. Its
    // document must not be advertised as though it were the server revision
    // fetched during hydration.
    const knownServerProject = timelineRevisionConflictRef.current?.serverAuthority?.document ??
      timelineServerProjectRef.current ?? timelineRef.current.project;
    let channel: TimelineRevisionChannel;
    channel = createTimelineRevisionChannel(
      scope,
      {
        revision: serverRevision,
        documentHash: timelineDocumentHash(knownServerProject),
      },
      () => {
        // Asset mutations own the project/document scope until their exact
        // post-mutation authority read completes. A Broadcast hint received in
        // that interval must not install an unrelated intermediate document.
        if (
          assetDeleteIntent.current || assetDeleteLock.current ||
          assetUploadLock.current
        ) return;
        const requestId = ++timelineRemoteAuthorityRequest.current;
        const writeGeneration = timelineWriteGeneration.current;
        void fetchTimelineForProject(projectId).then((authority) => {
          if (
            requestId !== timelineRemoteAuthorityRequest.current ||
            writeGeneration !== timelineWriteGeneration.current ||
            activeProjectIdRef.current !== projectId ||
            activeDatabaseRef.current?.active_database_path !== scope.databasePath
          ) return;
          const serverProject = normalizeTimelineProject(authority.document);
          if (!serverProject) throw new Error("服务器返回的时间线结构无效");
          const currentServerRevision = timelineServerRevision.current;
          const currentServerProject = timelineServerProjectRef.current;
          if (currentServerRevision === null || !currentServerProject) return;
          if (authority.revision < currentServerRevision) return;
          channel.acceptKnown({
            revision: authority.revision,
            documentHash: timelineDocumentHash(serverProject),
          });
          // A recovery gate intentionally displays the server document while
          // durable owned/foreign branches remain untouched. A same-document
          // broadcast is only a hint here; it is never authority to clear an
          // owned WAL or overwrite its journal before the user's choice.
          if (timelineRevisionConflictRef.current?.source === "recovery-branches") return;
          const currentProject = normalizeTimelineProject(structuredClone(timelineRef.current.project));
          if (!currentProject) return;
          if (
            authority.revision === currentServerRevision &&
            !timelineProjectsEqual(currentServerProject, serverProject)
          ) {
            const existing = timelineRevisionConflictRef.current;
            if (existing?.resolving) return;
            saveLocalTimeline(currentProject, database, projectId);
            setTimelineRevisionConflict({
              projectId,
              localProject: currentProject,
              serverAuthority: { document: serverProject, revision: authority.revision },
              source: "cas",
              resolving: false,
            });
            setTimelineDirty(true);
            setToast("服务器对同一修订号返回了不同时间线；已停止自动同步并保留本地状态");
            return;
          }
          const hasLocalChanges = timelinePersistedRevision.current < timelineRevision.current ||
            timelineSaveRequest.current !== null;
          if (timelineProjectsEqual(currentProject, serverProject)) {
            // The authority GET proves that the current local head is already
            // durable. Invalidate any older PUT still in flight before clearing
            // its WAL/base so a late 409 cannot manufacture a false conflict.
            timelineWriteGeneration.current += 1;
            acceptTimelineServerAuthority({ document: serverProject, revision: authority.revision });
            timelinePersistedRevision.current = timelineRevision.current;
            setTimelineDirty(false);
            clearLocalTimeline();
            persistTimelineJournal(
              timelineHistoryRef.current,
              { document: serverProject, revision: authority.revision },
              database,
              projectId,
            );
            return;
          }
          if (hasLocalChanges && authority.revision !== currentServerRevision) {
            const existing = timelineRevisionConflictRef.current;
            if (existing?.resolving) return;
            saveLocalTimeline(currentProject, database, projectId);
            setTimelineRevisionConflict({
              projectId,
              localProject: currentProject,
              serverAuthority: { document: serverProject, revision: authority.revision },
              source: "cas",
              resolving: false,
            });
            setTimelineDirty(true);
            setToast("其他页面已更新服务器时间线；本地草稿已保留，请选择如何处理冲突");
            return;
          }
          if (hasLocalChanges || timelineRevisionConflictRef.current) return;
          timelineWriteGeneration.current += 1;
          acceptTimelineServerAuthority({ document: serverProject, revision: authority.revision });
          installTimelineAuthority(serverProject);
          timelinePersistedRevision.current = timelineRevision.current;
          timelineHadLocal.current = true;
          setTimelineDirty(false);
          setTimelinePausedError(null);
          clearLocalTimeline();
          invalidateAndRefreshTaskSnapshots();
        }).catch(() => {
          // Broadcast is only an invalidation hint. Failed GETs neither replace
          // the document nor weaken CAS; a later notice/write will retry.
        });
      },
    );
    timelineRevisionChannelRef.current = channel;
    return () => {
      timelineRemoteAuthorityRequest.current += 1;
      channel.close();
      if (timelineRevisionChannelRef.current === channel) {
        timelineRevisionChannelRef.current = null;
      }
    };
  }, [acceptTimelineServerAuthority, activeProjectId, clearLocalTimeline, installTimelineAuthority, invalidateAndRefreshTaskSnapshots, persistTimelineJournal, saveLocalTimeline, setTimelineRevisionConflict, timelineHydrationStatus]);

  const loadAssets = useCallback(async (
    signal?: AbortSignal,
    failClosed = false,
    expectedScope?: AssetAuthorityScope,
  ): Promise<boolean> => {
    const activeDatabase = activeDatabaseRef.current;
    const scope = expectedScope ?? (
      activeDatabase && runtimeSettingsAuthorityReadyRef.current
        ? { database: { ...activeDatabase } }
        : null
    );
    const scopeStillCurrent = () => Boolean(
      scope &&
      runtimeSettingsAuthorityReadyRef.current &&
      activeDatabaseRef.current?.active_database_path === scope.database.active_database_path
    );
    if (!scopeStillCurrent()) {
      if (failClosed) {
        dispatchTimelineUi({ type: "assets/replace", assets: [] });
        throw new Error("素材库尚未完成权威确认");
      }
      return false;
    }
    const requestId = ++assetListRequest.current;
    try {
      const response = await directorApi.listAssets(undefined, signal);
      if (
        signal?.aborted ||
        assetListRequest.current !== requestId ||
        !scopeStillCurrent()
      ) {
        if (failClosed && assetListRequest.current === requestId) {
          dispatchTimelineUi({ type: "assets/replace", assets: [] });
          throw new Error("素材库响应已过期");
        }
        return false;
      }
      const preference = loadAssetLayoutPreference();
      const assets = orderAssetsByPreference(response.assets, preference.order);
      const nextAssetIds = new Set(assets.map((asset) => asset.id));
      if (timelineRef.current.assets.some((asset) => !nextAssetIds.has(asset.id))) {
        clearTimelineHistory();
      }
      dispatchTimelineUi({
        type: "assets/replace",
        assets,
      });
      return true;
    } catch (reason) {
      if (signal?.aborted || assetListRequest.current !== requestId) return false;
      if (failClosed) {
        dispatchTimelineUi({ type: "assets/replace", assets: [] });
        throw reason;
      }
      // An older backend may not expose the library yet; new uploads still appear.
      return false;
    }
  }, [clearTimelineHistory, dispatchTimelineUi]);

  const loadAssetTrash = useCallback(async (): Promise<boolean> => {
    const requestId = ++assetTrashListRequest.current;
    setAssetTrashLoading(true);
    try {
      const response = await directorApi.listAssetTrash();
      if (requestId !== assetTrashListRequest.current) return false;
      setAssetTrashBatches(response.batches);
      const visibleBatchIds = new Set(response.batches.map((batch) => batch.batch_id));
      setAssetTrashConflictBatchIds((current) => new Set(
        [...current].filter((batchId) => visibleBatchIds.has(batchId)),
      ));
      return true;
    } catch (reason) {
      if (requestId !== assetTrashListRequest.current) return false;
      setToast(reason instanceof Error ? reason.message : "读取素材回收站失败");
      return false;
    } finally {
      if (requestId === assetTrashListRequest.current) setAssetTrashLoading(false);
    }
  }, []);

  const reconcileAmbiguousAssetTrash = useCallback(async (
    assetIds: readonly string[],
    expectedDatabase: ActiveDatabaseIdentity,
    expectedProjectId: string,
    expectedProjectSwitchGeneration: number,
  ): Promise<"committed" | "rejected" | "unknown"> => {
    const assetRequestId = ++assetListRequest.current;
    const trashRequestId = ++assetTrashListRequest.current;
    setAssetTrashLoading(true);
    try {
      const [assetResponse, trashResponse] = await Promise.all([
        directorApi.listAssets(),
        directorApi.listAssetTrash(),
      ]);
      const activeDatabase = activeDatabaseRef.current;
      if (
        assetListRequest.current !== assetRequestId ||
        assetTrashListRequest.current !== trashRequestId ||
        !activeDatabase ||
        projectSwitchGeneration.current !== expectedProjectSwitchGeneration ||
        activeProjectIdRef.current !== expectedProjectId ||
        activeDatabase.active_database_path !== expectedDatabase.active_database_path
      ) return "unknown";

      const preference = loadAssetLayoutPreference();
      const assets = orderAssetsByPreference(assetResponse.assets, preference.order);
      const liveAssetIds = new Set(assets.map((asset) => asset.id));
      if (timelineRef.current.assets.some((asset) => !liveAssetIds.has(asset.id))) {
        clearTimelineHistory();
      }
      dispatchTimelineUi({ type: "assets/replace", assets });
      setAssetTrashBatches(trashResponse.batches);
      const visibleBatchIds = new Set(trashResponse.batches.map((batch) => batch.batch_id));
      setAssetTrashConflictBatchIds((current) => new Set(
        [...current].filter((batchId) => visibleBatchIds.has(batchId)),
      ));

      const trashedAssetIds = new Set(
        trashResponse.batches.flatMap((batch) => batch.asset_ids),
      );
      if (assetIds.every((assetId) => !liveAssetIds.has(assetId) && trashedAssetIds.has(assetId))) {
        return "committed";
      }
      if (assetIds.every((assetId) => liveAssetIds.has(assetId) && !trashedAssetIds.has(assetId))) {
        return "rejected";
      }
      return "unknown";
    } catch {
      return "unknown";
    } finally {
      if (trashRequestId === assetTrashListRequest.current) setAssetTrashLoading(false);
    }
  }, [clearTimelineHistory, dispatchTimelineUi]);

  const refreshRuntimeResources = useCallback(async (
    preserveExisting: boolean,
    signal?: AbortSignal,
    scheduleBackgroundRetry = true,
    authoritySnapshot?: RuntimeSettingsAuthority,
  ): Promise<boolean> => {
    if (runtimeResourceRetryTimer.current !== null) {
      window.clearTimeout(runtimeResourceRetryTimer.current);
      runtimeResourceRetryTimer.current = null;
    }
    const requestId = ++runtimeResourceRequest.current;
    setLoadingModels(true);
    let initialAuthority: { settings: RuntimeSettings; token: string } | null = null;
    try {
      const snapshot = authoritySnapshot ?? await directorApi.getSettingsAuthority(signal);
      initialAuthority = {
        settings: sanitizeRuntimeSettings(snapshot.settings),
        token: snapshot.authority_token,
      };
    } catch {
      initialAuthority = null;
    }
    const unavailable = () => Promise.reject(new Error("运行设置权威 token 不可用"));
    const [capabilityResult, gpuResult, modelResult, rayLightRuntimeResult] = await Promise.allSettled(
      initialAuthority
        ? [
            directorApi.getCapabilities(signal, initialAuthority.token),
            directorApi.getGpus(signal, initialAuthority.token),
            directorApi.getModels(signal, initialAuthority.token),
            directorApi.getRayLightRuntimeStatus(signal, initialAuthority.token),
          ]
        : [unavailable(), unavailable(), unavailable(), unavailable()],
    );
    // Read the authority again after all resources. The server also checks the
    // same token before and after every upstream call, so A -> B -> A cannot
    // bless a mixed/B snapshot merely because the URL returned to A.
    const [finalAuthorityResult] = await Promise.allSettled([
      directorApi.getSettingsAuthority(signal),
    ]);
    if (
      signal?.aborted ||
      runtimeResourceRequest.current !== requestId
    ) return false;

    const finalAuthority = finalAuthorityResult.status === "fulfilled"
      ? {
          settings: sanitizeRuntimeSettings(finalAuthorityResult.value.settings),
          token: finalAuthorityResult.value.authority_token,
        }
      : null;
    // A concurrent settings write anywhere (this page, another tab, or the
    // server itself) invalidates this in-flight resource batch: its four
    // endpoints resolved their inputs from the pre-write authority. The
    // App-owned reconciliation adopts the newer authority instead.
    const authorityChanged = Boolean(
      initialAuthority && finalAuthority &&
      initialAuthority.token !== finalAuthority.token
    );
    const browserAuthorityChanged = Boolean(
      initialAuthority && (
        authoritativeSettingsTokenRef.current !== initialAuthority.token ||
        !sameRuntimeSettings(authoritativeSettingsRef.current, initialAuthority.settings)
      )
    );
    if (authorityChanged || browserAuthorityChanged) {
      runtimeResourceRequest.current += 1;
      runtimeSettingsAuthorityReadyRef.current = false;
      authoritativeSettingsTokenRef.current = null;
      runtimeResourcesReadyRef.current = false;
      runtimeResourcesAuthorityTokenRef.current = null;
      setRuntimeResourcesReady(false);
      rayLightRuntimeStatusRef.current = null;
      setRayLightRuntimeStatus(null);
      setCapabilities({ ...EMPTY_CAPABILITIES, connection: "checking", message: "运行设置已在服务器端变化，正在重新核对" });
      setGpus([]);
      setModels(EMPTY_MODELS);
      setLoadingModels(false);
      assetAuthorityRequired.current = true;
      assetListRequest.current += 1;
      dispatchTimelineUi({ type: "assets/replace", assets: [] });
      clearTimelineHistory();
      setRuntimeAuthorityRequired(true);
      setCompileReport(null);
      window.queueMicrotask(() => externalRuntimeAuthorityRefreshRef.current());
      return false;
    }

    const visibleGpuSnapshot = gpuResult.status === "fulfilled"
      ? gpuResult.value.filter((gpu) => gpu.visible)
      : [];
    const gpuSnapshotMatchesRuntime = gpuResult.status === "fulfilled" &&
      rayLightRuntimeResult.status === "fulfilled" &&
      rayLightRuntimeResult.value.available_gpu_indexes.length === visibleGpuSnapshot.length &&
      visibleGpuSnapshot.every((gpu, offset) => gpu.index === offset) &&
      rayLightRuntimeResult.value.available_gpu_indexes.every((index, offset) => index === offset);
    const complete = initialAuthority !== null &&
      finalAuthority !== null &&
      initialAuthority.token === finalAuthority.token &&
      capabilityResult.status === "fulfilled" &&
      capabilityResult.value.connection === "online" &&
      gpuResult.status === "fulfilled" &&
      modelResult.status === "fulfilled" &&
      rayLightRuntimeResult.status === "fulfilled" &&
      gpuSnapshotMatchesRuntime;
    if (complete) {
      setCapabilities(capabilityResult.value);
      setGpus(gpuResult.value);
      setModels(modelResult.value);
      rayLightRuntimeStatusRef.current = rayLightRuntimeResult.value;
      setRayLightRuntimeStatus(rayLightRuntimeResult.value);
      setLoadingModels(false);
      runtimeResourcesReadyRef.current = true;
      runtimeResourcesAuthorityTokenRef.current = initialAuthority!.token;
      setRuntimeResourcesReady(true);
      if (runtimeSettingsDesired.current) {
        window.queueMicrotask(() => runtimeSettingsDrainRef.current());
      } else if (runtimeSettingsSyncRequiredRef.current && !assetAuthorityRequired.current) {
        if (runtimeSettingsDraftValidRef.current) {
          setRuntimeAuthorityRequired(false);
          invalidateAndRefreshTaskSnapshots();
        }
      }
      return true;
    }

    // A partial response is not a resource-authority snapshot. Keep a
    // previously confirmed same-origin inventory intact; for a new origin the
    // controls remain fail-closed until the settings anchor and all four
    // resources agree in one attempt.
    if (!preserveExisting) {
      if (capabilityResult.status === "fulfilled") setCapabilities(capabilityResult.value);
      if (gpuResult.status === "fulfilled") setGpus(gpuResult.value);
      if (modelResult.status === "fulfilled") setModels(modelResult.value);
    }
    setLoadingModels(false);
    if (scheduleBackgroundRetry) {
      runtimeResourceRetryTimer.current = window.setTimeout(() => {
        runtimeResourceRetryTimer.current = null;
        void runtimeResourceRefreshRef.current(runtimeResourcesReadyRef.current);
      }, RUNTIME_SETTINGS_RETRY_MS);
    }
    return false;
  }, [clearTimelineHistory, dispatchTimelineUi, invalidateAndRefreshTaskSnapshots, setRuntimeAuthorityRequired]);
  runtimeResourceRefreshRef.current = (preserveExisting) =>
    refreshRuntimeResources(preserveExisting);

  const refreshAuthoritativeResourcesAfterConnectionTest = useCallback(() => {
    // A successful probe re-confirms the single embedded host. App owns
    // resource authority and refreshes all four inventories in one
    // latest-wins generation; a partial failure preserves the previously
    // confirmed snapshot.
    setRuntimeAuthorityRequired(true);
    setCompileReport(null);
    void refreshRuntimeResources(runtimeResourcesReadyRef.current);
  }, [refreshRuntimeResources, setRuntimeAuthorityRequired]);

  const refreshRuntime = useCallback(async (
    signal?: AbortSignal,
    preserveResources = false,
  ) => {
    const requestId = ++runtimeRequest.current;
    if (!preserveResources) {
      setCapabilities({ ...EMPTY_CAPABILITIES, connection: "checking" });
      setGpus([]);
      setModels(EMPTY_MODELS);
      rayLightRuntimeStatusRef.current = null;
      setRayLightRuntimeStatus(null);
      setLoadingModels(false);
    }
    let authoritySnapshot: RuntimeSettingsAuthority;
    let settings: RuntimeSettings;
    try {
      authoritySnapshot = await directorApi.getSettingsAuthority(signal);
      settings = sanitizeRuntimeSettings(authoritySnapshot.settings);
    } catch {
      if (signal?.aborted || runtimeRequest.current !== requestId) return null;
      // A same-endpoint settings mutation deliberately preserves the already
      // confirmed capability/model/GPU inventory. Its authority lock is enough
      // to fail closed while this GET is retried; marking those resources
      // offline here would make a later successful preserve-only retry unlock
      // the workspace with a stale offline presentation.
      if (!preserveResources) {
        setCapabilities({
          ...EMPTY_CAPABILITIES,
          connection: "offline",
          message: "Director 后端无法确认服务器运行设置",
        });
      }
      return null;
    }
    if (signal?.aborted || runtimeRequest.current !== requestId) return null;
    runtimeSettingsAuthorityReadyRef.current = true;
    authoritativeSettingsRef.current = settings;
    authoritativeSettingsTokenRef.current = authoritySnapshot.authority_token;
    dispatch({ type: "settings/replace", settings });
    if (!runtimeSettingsDesired.current && runtimeSettingsDraftValidRef.current) setRuntimeSettingsDraft(settings);
    if (runtimeSettingsDesired.current) window.queueMicrotask(() => runtimeSettingsDrainRef.current());
    await refreshRuntimeResources(
      preserveResources && runtimeResourcesReadyRef.current,
      signal,
      true,
      authoritySnapshot,
    );
    if (signal?.aborted || runtimeRequest.current !== requestId) return null;
    return settings;
  }, [refreshRuntimeResources]);

  const refreshExternalRuntimeAuthority = useCallback(() => {
    if (externalRuntimeAuthorityOperationRef.current) return;
    if (externalRuntimeAuthorityRetryTimer.current !== null) {
      window.clearTimeout(externalRuntimeAuthorityRetryTimer.current);
      externalRuntimeAuthorityRetryTimer.current = null;
    }
    const controller = new AbortController();
    externalRuntimeAuthorityControllerRef.current = controller;
    const operation = (async () => {
      try {
        const settings = await refreshRuntime(controller.signal, false);
        throwIfAborted(controller.signal);
        if (!settings) throw new Error("无法读取服务器权威运行设置");
        if (!runtimeResourcesReadyRef.current) {
          throw new Error("运行资源尚未完成核对");
        }
        const assetsReady = await loadAssets(controller.signal, true);
        throwIfAborted(controller.signal);
        if (!assetsReady) throw new Error("素材库刷新请求已过期");
        assetAuthorityRequired.current = false;
        if (runtimeSettingsDesired.current) {
          window.queueMicrotask(() => runtimeSettingsDrainRef.current());
        } else {
          if (runtimeSettingsDraftValidRef.current) {
            setRuntimeAuthorityRequired(false);
            invalidateAndRefreshTaskSnapshots();
          }
        }
      } catch (reason) {
        if (controller.signal.aborted) return;
        setToast(`${reason instanceof Error ? reason.message : "服务器运行设置核对失败"}；正在自动重试`);
        if (externalRuntimeAuthorityRetryTimer.current === null) {
          externalRuntimeAuthorityRetryTimer.current = window.setTimeout(() => {
            externalRuntimeAuthorityRetryTimer.current = null;
            externalRuntimeAuthorityRefreshRef.current();
          }, RUNTIME_SETTINGS_RETRY_MS);
        }
      }
    })();
    externalRuntimeAuthorityOperationRef.current = operation;
    void operation.finally(() => {
      if (externalRuntimeAuthorityOperationRef.current === operation) {
        externalRuntimeAuthorityOperationRef.current = null;
      }
      if (externalRuntimeAuthorityControllerRef.current === controller) {
        externalRuntimeAuthorityControllerRef.current = null;
      }
    });
  }, [invalidateAndRefreshTaskSnapshots, loadAssets, refreshRuntime, setRuntimeAuthorityRequired]);
  externalRuntimeAuthorityRefreshRef.current = refreshExternalRuntimeAuthority;

  const reconcileRuntimeSettings = useCallback(async (
    owner: RuntimeSettingsOperationOwner,
    nextSettings?: RuntimeSettings,
  ): Promise<RuntimeSettings> => {
    if (rayLightRecoveryPendingRef.current) {
      throw new Error("RayLight 重启恢复正在核对，不能修改运行设置");
    }
    if (assetDeleteLock.current || assetDeleteIntent.current) {
      throw new Error("正在原子解除素材引用，完成前不能切换运行设置");
    }
    if (assetUploadLock.current) {
      throw new Error("正在上传并绑定本地素材，完成前不能切换运行设置");
    }
    if (timelineRevisionConflict) {
      throw new Error("服务器时间线存在修订冲突，处理完成前不能修改运行设置");
    }
    if (timelineSyncRequired) {
      throw new Error("服务器时间线正在自动恢复权威状态，完成前不能修改运行设置");
    }

    const normalized = nextSettings
      ? sanitizeRuntimeSettings(structuredClone(nextSettings))
      : null;
    const generation = ++runtimeSettingsGeneration.current;

    setRuntimeSettingsOperationOwner(owner);
    setRuntimeAuthorityRequired(true);
    setCompileReport(null);
    const invalidateAssetAuthority = () => {
      assetAuthorityRequired.current = true;
      assetListRequest.current += 1;
      dispatchTimelineUi({ type: "assets/replace", assets: [] });
      clearTimelineHistory();
    };

    const operation = (async () => {
      if (assetAuthorityRequired.current) {
        invalidateAssetAuthority();
      }

      let writeError: unknown = null;
      if (normalized) {
        try {
          // The response is deliberately ignored. Only the GET below may
          // become browser authority after a whole-document PUT.
          await directorApi.updateSettings(normalized);
        } catch (reason) {
          // A lost response is ambiguous: the server may still have committed.
          // Continue into the authoritative GET instead of restoring old state.
          writeError = reason;
        }
      }

      const confirmed = await refreshRuntime(undefined, true);
      if (!confirmed) {
        if (writeError instanceof ApiError && writeError.status >= 400 && writeError.status < 500) {
          throw writeError;
        }
        const writeMessage = writeError instanceof Error ? `${writeError.message}；` : "";
        throw new Error(`${writeMessage}无法从 Director 后端权威回读运行设置；生成与素材操作保持锁定`);
      }
      if (runtimeSettingsGeneration.current !== generation) {
        throw new Error("运行设置回读已被更新操作取代");
      }

      if (assetAuthorityRequired.current) {
        try {
          const refreshed = await loadAssets(undefined, true);
          if (!refreshed) throw new Error("素材库刷新请求已过期");
        } catch {
          throw new Error("运行设置已确认，但素材库无法权威刷新；旧素材已清空，生成保持锁定");
        }
        assetAuthorityRequired.current = false;
      }

      if (normalized && !sameRuntimeSettings(confirmed, normalized)) {
        if (writeError) throw writeError;
        throw new Error("服务器权威运行设置与最新修改不一致；生成与素材操作保持锁定并将自动重试");
      }
      return confirmed;
    })();

    runtimeSettingsOperation.current = operation;
    try {
      return await operation;
    } finally {
      if (runtimeSettingsOperation.current === operation) {
        runtimeSettingsOperation.current = null;
        setRuntimeSettingsOperationOwner(null);
      }
    }
  }, [clearTimelineHistory, dispatchTimelineUi, invalidateAndRefreshTaskSnapshots, loadAssets, refreshRuntime, setRuntimeAuthorityRequired, timelineRevisionConflict, timelineSyncRequired]);

  const drainRuntimeSettings = useCallback(() => {
    if (runtimeSettingsOperation.current || runtimeSettingsAutosaveTimer.current !== null) return;
    if (rayLightRecoveryPendingRef.current) return;
    if (!activeDatabaseRef.current || !runtimeSettingsAuthorityReadyRef.current) return;
    const desired = runtimeSettingsDesired.current;
    if (!desired) return;
    if (
      runtimeSettingsPausedDesiredRef.current &&
      sameRuntimeSettings(runtimeSettingsPausedDesiredRef.current, desired)
    ) return;
    if (
      assetDeleteLock.current || assetDeleteIntent.current || assetUploadLock.current ||
      timelineSyncRequiredRef.current || timelineRevisionConflictRef.current || runtimeExecutionIntent.current > 0
    ) {
      if (runtimeSettingsRetryTimer.current === null) {
        runtimeSettingsRetryTimer.current = window.setTimeout(() => {
          runtimeSettingsRetryTimer.current = null;
          runtimeSettingsDrainRef.current();
        }, RUNTIME_SETTINGS_RETRY_MS);
      }
      return;
    }
    const desiredResourcesReady = runtimeResourcesReadyRef.current &&
      runtimeResourcesAuthorityTokenRef.current === authoritativeSettingsTokenRef.current;
    if (
      sameRuntimeSettings(desired, authoritativeSettingsRef.current) &&
      !assetAuthorityRequired.current &&
      desiredResourcesReady
    ) {
      runtimeSettingsDesired.current = null;
      runtimeSettingsPausedDesiredRef.current = null;
      clearPendingRuntimeSettings();
      setRuntimeSettingsDraft(desired);
      setRuntimeAuthorityRequired(false);
      invalidateAndRefreshTaskSnapshots();
      if (timelinePersistedRevision.current < timelineRevision.current) {
        setTimelineRetryNonce((current) => current + 1);
      }
      return;
    }
    if (
      sameRuntimeSettings(desired, authoritativeSettingsRef.current) &&
      !assetAuthorityRequired.current &&
      !desiredResourcesReady
    ) {
      runtimeSettingsDesired.current = null;
      runtimeSettingsPausedDesiredRef.current = null;
      clearPendingRuntimeSettings();
      setRuntimeSettingsDraft(desired);
      void runtimeResourceRefreshRef.current(false);
      if (timelinePersistedRevision.current < timelineRevision.current) {
        setTimelineRetryNonce((current) => current + 1);
      }
      return;
    }

    const snapshot = structuredClone(desired);
    const owner = runtimeSettingsDesiredOwner.current;
    let drainNewerImmediately = false;
    const operation = reconcileRuntimeSettings(owner, snapshot);
    void operation.then((confirmed) => {
      const confirmedResourcesReady = runtimeResourcesReadyRef.current &&
        runtimeResourcesAuthorityTokenRef.current === authoritativeSettingsTokenRef.current;
      if (
        runtimeSettingsDesired.current &&
        sameRuntimeSettings(runtimeSettingsDesired.current, snapshot) &&
        !assetAuthorityRequired.current
      ) {
        runtimeSettingsDesired.current = null;
        runtimeSettingsPausedDesiredRef.current = null;
        clearPendingRuntimeSettings();
        setRuntimeSettingsDraft(confirmed);
        setRuntimeSettingsPausedError(null);
        if (confirmedResourcesReady) {
          setRuntimeAuthorityRequired(false);
          invalidateAndRefreshTaskSnapshots();
        }
        if (timelinePersistedRevision.current < timelineRevision.current) {
          setTimelineRetryNonce((current) => current + 1);
        }
      } else if (
        runtimeSettingsDesired.current &&
        !sameRuntimeSettings(runtimeSettingsDesired.current, snapshot)
      ) {
        drainNewerImmediately = true;
      }
    }).catch((reason) => {
      // Desired stays in the WAL. A failed or ambiguous PUT/GET must remain
      // fail-closed and retry without requiring the settings overlay to exist.
      const superseded = runtimeSettingsDesired.current !== null &&
        !sameRuntimeSettings(runtimeSettingsDesired.current, snapshot);
      const explicitlySuperseded = reason instanceof RuntimeSettingsSupersededError;
      drainNewerImmediately = superseded;
      const deterministicClientError = reason instanceof ApiError &&
        reason.status >= 400 && reason.status < 500;
      if (!superseded && deterministicClientError) {
        runtimeSettingsPausedDesiredRef.current = structuredClone(snapshot);
        setRuntimeSettingsPausedError(reason instanceof Error ? reason.message : "服务器拒绝当前系统设置");
      }
      setToast(reason instanceof Error ? reason.message : "运行设置自动同步失败");
      if (
        !superseded && !explicitlySuperseded && !deterministicClientError &&
        runtimeSettingsRetryTimer.current === null
      ) {
        runtimeSettingsRetryTimer.current = window.setTimeout(() => {
          runtimeSettingsRetryTimer.current = null;
          runtimeSettingsDrainRef.current();
        }, RUNTIME_SETTINGS_RETRY_MS);
      }
    }).finally(() => {
      if (drainNewerImmediately) {
        window.queueMicrotask(() => runtimeSettingsDrainRef.current());
      }
    });
  }, [invalidateAndRefreshTaskSnapshots, reconcileRuntimeSettings, setRuntimeAuthorityRequired]);
  runtimeSettingsDrainRef.current = drainRuntimeSettings;

  const queueRuntimeSettings = useCallback((
    owner: RuntimeSettingsOperationOwner,
    nextSettings: RuntimeSettings,
  ): Promise<RuntimeSettings> => {
    if (rayLightRecoveryPendingRef.current) {
      return Promise.reject(new Error("RayLight 重启恢复正在核对，不能修改运行设置"));
    }
    const normalized = sanitizeRuntimeSettings(structuredClone(nextSettings));
    runtimeSettingsDesired.current = normalized;
    runtimeSettingsPausedDesiredRef.current = null;
    runtimeSettingsDesiredOwner.current = owner;
    setRuntimeSettingsDraft(normalized);
    setRuntimeSettingsPausedError(null);
    runtimeSettingsDraftValidRef.current = true;
    setRuntimeSettingsDraftValid(true);
    setRuntimeAuthorityRequired(true);
    setCompileReport(null);
    taskListRequest.current += 1;
    dispatch({ type: "tasks/invalidate-current-snapshots" });
    const activeDatabase = activeDatabaseRef.current;
    if (activeDatabase) savePendingRuntimeSettings(normalized, activeDatabase);
    if (runtimeSettingsRetryTimer.current !== null) {
      window.clearTimeout(runtimeSettingsRetryTimer.current);
      runtimeSettingsRetryTimer.current = null;
    }
    if (runtimeSettingsAutosaveTimer.current !== null) {
      window.clearTimeout(runtimeSettingsAutosaveTimer.current);
    }
    runtimeSettingsAutosaveTimer.current = window.setTimeout(() => {
      runtimeSettingsAutosaveTimer.current = null;
      runtimeSettingsDrainRef.current();
    }, RUNTIME_SETTINGS_AUTOSAVE_MS);
    return Promise.resolve(normalized);
  }, [setRuntimeAuthorityRequired]);

  const updateRuntimeSettingsDraft = useCallback((draft: RuntimeSettings) => {
    setRuntimeSettingsDraft(draft);
    setRuntimeSettingsPausedError(null);
    const valid = validateRuntimeSettingsForm(draft).length === 0;
    runtimeSettingsDraftValidRef.current = valid;
    setRuntimeSettingsDraftValid(valid);
    if (valid) return;
    // A newer invalid intermediate draft supersedes an older desired snapshot
    // that has not started yet. Keep execution locked until the draft becomes
    // valid; never silently apply the older value after the user cleared it.
    runtimeSettingsDesired.current = null;
    runtimeSettingsPausedDesiredRef.current = null;
    if (runtimeSettingsAutosaveTimer.current !== null) {
      window.clearTimeout(runtimeSettingsAutosaveTimer.current);
      runtimeSettingsAutosaveTimer.current = null;
    }
    clearPendingRuntimeSettings();
    setRuntimeAuthorityRequired(true);
  }, [setRuntimeAuthorityRequired]);

  useEffect(() => {
    if (runtimeSettingsDesired.current) {
      runtimeSettingsAutosaveTimer.current = window.setTimeout(() => {
        runtimeSettingsAutosaveTimer.current = null;
        runtimeSettingsDrainRef.current();
      }, RUNTIME_SETTINGS_AUTOSAVE_MS);
    }
    return () => {
      if (runtimeSettingsAutosaveTimer.current !== null) {
        window.clearTimeout(runtimeSettingsAutosaveTimer.current);
        runtimeSettingsAutosaveTimer.current = null;
      }
      if (runtimeSettingsRetryTimer.current !== null) {
        window.clearTimeout(runtimeSettingsRetryTimer.current);
        runtimeSettingsRetryTimer.current = null;
      }
      if (runtimeResourceRetryTimer.current !== null) {
        window.clearTimeout(runtimeResourceRetryTimer.current);
        runtimeResourceRetryTimer.current = null;
      }
      if (externalRuntimeAuthorityRetryTimer.current !== null) {
        window.clearTimeout(externalRuntimeAuthorityRetryTimer.current);
        externalRuntimeAuthorityRetryTimer.current = null;
      }
      externalRuntimeAuthorityControllerRef.current?.abort(
        new DOMException("Runtime authority owner unmounted", "AbortError"),
      );
      externalRuntimeAuthorityControllerRef.current = null;
      rayLightRecoveryControllerRef.current?.abort(
        new DOMException("RayLight recovery owner unmounted", "AbortError"),
      );
      rayLightRecoveryControllerRef.current = null;
    };
  }, []);

  const runTimelineAutosave = useCallback((): Promise<TimelineProject | null> => {
    if (timelineSaveRequest.current) return timelineSaveRequest.current;
    if (
      databaseIdentityStaleRef.current ||
      timelineSyncRequiredRef.current ||
      timelineRevisionConflictRef.current ||
      assetDeleteLock.current ||
      assetUploadLock.current ||
      timelinePersistedRevision.current >= timelineRevision.current ||
      timelineRenderedRevision.current < timelineRevision.current
    ) {
      return Promise.resolve(null);
    }

    const generation = timelineWriteGeneration.current;
    const revision = timelineRevision.current;
    const projectId = activeProjectIdRef.current;
    const expectedServerRevision = timelineServerRevision.current;
    const expectedServerProject = timelineServerProjectRef.current;
    const database = activeDatabaseRef.current;
    if (expectedServerRevision === null || !expectedServerProject) {
      return Promise.reject(new Error("服务器时间线修订号尚未确认，暂不能同步"));
    }
    if (!database) {
      return Promise.reject(new Error("数据库身份尚未确认，暂不能同步时间线"));
    }
    const snapshot = normalizeTimelineProject(structuredClone(timelineRef.current.project));
    if (!snapshot) return Promise.reject(new Error("时间线结构无效，请检查项目字段"));

    let mayDrainImmediately = false;
    let failedRevisionWasSuperseded = false;
    let observedConflictAuthority: TimelineAuthority | null = null;
    const operation = (async (): Promise<TimelineProject | null> => {
      let response: TimelineAuthority;
      try {
        response = await runWithTimelineWriterLock({
          databasePath: database.active_database_path,
          projectId,
        }, () => saveTimelineForProject(projectId, snapshot, expectedServerRevision));
      } catch (reason) {
        if (!(
          reason instanceof ApiError &&
          reason.status === 409 &&
          reason.code === "timeline_revision_conflict"
        )) throw reason;
        // A response can be lost after commit and a later retry can then see a
        // 409. Only an authority GET proving that the submitted snapshot is the
        // current document converts that ambiguity into an ACK.
        const authority = await fetchTimelineForProject(projectId);
        const serverDocument = normalizeTimelineProject(authority.document);
        if (!serverDocument) throw new Error("服务器返回的时间线结构无效");
        if (!timelineProjectsEqual(snapshot, serverDocument)) {
          observedConflictAuthority = {
            document: serverDocument,
            revision: authority.revision,
          };
          throw reason;
        }
        response = {
          document: serverDocument,
          revision: authority.revision,
        };
      }
      const confirmed = normalizeTimelineProject(response.document);
      if (!confirmed) throw new Error("服务器返回的时间线结构无效");
      // Exclusive mutations advance the generation. Their authority must never
      // be replaced by a response that started before them.
      if (timelineWriteGeneration.current !== generation) return null;

      // The ACK advances the server base even when a newer local edit already
      // exists. The next latest-wins write must compare against this revision,
      // while the older ACK document itself must not replace that newer edit.
      acceptTimelineServerAuthority({ document: confirmed, revision: response.revision });
      timelineRevisionChannelRef.current?.publish({
        revision: response.revision,
        documentHash: timelineDocumentHash(confirmed),
      });
      timelinePersistedRevision.current = Math.max(
        timelinePersistedRevision.current,
        revision,
      );
      mayDrainImmediately = true;
      timelineHadLocal.current = true;
      if (timelineRevision.current !== revision) {
        // A newer edit already replaced the submitted head. Advance both
        // durable recovery layers to the ACKed base before the next CAS write;
        // never clear the newer WAL with the older request's completion.
        const latest = normalizeTimelineProject(structuredClone(timelineRef.current.project));
        if (latest) {
          saveLocalTimeline(latest, database, projectId);
          if (
            timelineHistoryRef.current.head &&
            timelineProjectsEqual(timelineHistoryRef.current.head, latest)
          ) {
            persistTimelineJournal(
              timelineHistoryRef.current,
              { document: confirmed, revision: response.revision },
              database,
              projectId,
            );
          }
        }
        return null;
      }

      let durableHistory = timelineHistoryRef.current;
      if (!timelineProjectsEqual(snapshot, confirmed)) {
        // An exact ACK normally echoes the submitted normalized document. A
        // materially canonicalized response becomes a new authority boundary;
        // keep history only when its current head can be rebased without
        // changing segment order or asset-slot identity.
        const rebased = rebaseTimelineHistoryHead(
          timelineHistoryRef.current,
          snapshot,
          confirmed,
        );
        if (rebased) {
          durableHistory = rebased;
          commitTimelineHistory(rebased);
        } else {
          durableHistory = resetTimelineHistory(timelineHistoryRef.current);
          clearTimelineHistory();
        }
      }
      installTimelineAuthority(confirmed, { clearHistory: false });
      clearLocalTimeline();
      persistTimelineJournal(
        durableHistory,
        { document: confirmed, revision: response.revision },
        database,
        projectId,
      );
      setTimelineDirty(false);
      setTimelinePausedError(null);
      if (timelineRetryTimer.current !== null) {
        window.clearTimeout(timelineRetryTimer.current);
        timelineRetryTimer.current = null;
      }
      invalidateAndRefreshTaskSnapshots();
      if (runtimeSettingsDesired.current) {
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        runtimeSettingsDrainRef.current();
      }
      return confirmed;
    })();

    timelineSaveRequest.current = operation;
    timelineSaveRequestRevision.current = revision;
    void operation.catch(async (reason) => {
      if (
        timelineWriteGeneration.current !== generation ||
        timelineSyncRequiredRef.current ||
        assetDeleteLock.current
      ) return;
      setTimelineDirty(true);
      if (
        reason instanceof ApiError &&
        reason.status === 409 &&
        reason.code === "timeline_revision_conflict"
      ) {
        const localProject = normalizeTimelineProject(structuredClone(timelineRef.current.project));
        if (!localProject) return;
        const conflict: TimelineRevisionConflict = {
          projectId,
          localProject,
          serverAuthority: observedConflictAuthority,
          source: "cas",
          resolving: false,
        };
        setTimelineRevisionConflict(conflict);
        const activeDatabase = activeDatabaseRef.current;
        if (activeDatabase) saveLocalTimeline(localProject, activeDatabase, projectId);
        setToast("服务器时间线已被其他页面修改；已保留本地草稿，请选择如何处理冲突");
        if (observedConflictAuthority) return;
        try {
          const authority = await fetchTimelineForProject(projectId);
          const serverDocument = normalizeTimelineProject(authority.document);
          if (
            !serverDocument ||
            timelineWriteGeneration.current !== generation ||
            activeProjectIdRef.current !== projectId ||
            timelineRevisionConflictRef.current !== conflict
          ) return;
          setTimelineRevisionConflict({
            ...conflict,
            serverAuthority: { document: serverDocument, revision: authority.revision },
          });
        } catch {
          // Resolution actions perform another authoritative GET. A failed
          // diagnostic read must not unlock or discard the local WAL.
        }
        return;
      }
      const deterministicClientError = reason instanceof ApiError &&
        reason.status >= 400 && reason.status < 500;
      failedRevisionWasSuperseded = deterministicClientError &&
        timelineRevision.current > revision;
      if (deterministicClientError && !failedRevisionWasSuperseded) {
        setTimelinePausedError({
          revision,
          message: reason.message || "当前时间线字段无效",
        });
      }
      setToast(
        deterministicClientError
          ? (reason.message || "当前时间线字段无效；修正后会继续同步")
          : `${reason instanceof Error ? reason.message : "时间线同步失败"}；将在连接恢复后自动重试`,
      );
      // A deterministic 4xx will not heal with time. Keep the WAL, then let the
      // next real project edit schedule a fresh attempt. Transport/5xx failures
      // use a bounded retry delay so an outage cannot create a request storm.
      if (deterministicClientError) return;
      if (timelineRetryTimer.current !== null) window.clearTimeout(timelineRetryTimer.current);
      timelineRetryTimer.current = window.setTimeout(() => {
        timelineRetryTimer.current = null;
        setTimelineRetryNonce((current) => current + 1);
      }, 1200);
    }).finally(() => {
      if (timelineSaveRequest.current === operation) {
        timelineSaveRequest.current = null;
        timelineSaveRequestRevision.current = null;
      }
      if (
        (mayDrainImmediately || failedRevisionWasSuperseded) &&
        timelineWriteGeneration.current === generation &&
        !timelineSyncRequiredRef.current &&
        !timelineRevisionConflictRef.current &&
        !assetDeleteLock.current &&
        timelinePersistedRevision.current < timelineRevision.current
      ) {
        setTimelineRetryNonce((current) => current + 1);
      }
    });
    return operation;
  }, [acceptTimelineServerAuthority, clearLocalTimeline, clearTimelineHistory, commitTimelineHistory, installTimelineAuthority, invalidateAndRefreshTaskSnapshots, persistTimelineJournal, saveLocalTimeline, setTimelineRevisionConflict]);

  const flushTimelineAutosave = useCallback(async (): Promise<TimelineProject> => {
    const generation = timelineWriteGeneration.current;
    if (timelineAutosaveTimer.current !== null) {
      window.clearTimeout(timelineAutosaveTimer.current);
      timelineAutosaveTimer.current = null;
    }
    while (timelinePersistedRevision.current < timelineRevision.current) {
      if (timelineRevisionConflictRef.current) {
        throw new Error("服务器时间线存在修订冲突，请先选择采用服务器版本或保留本地版本");
      }
      if (timelineSyncRequiredRef.current) {
        throw new Error("服务器时间线权威状态尚未恢复，暂不能继续");
      }
      if (timelineRenderedRevision.current < timelineRevision.current) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
        continue;
      }
      const before = timelinePersistedRevision.current;
      const attemptedRevision = timelineSaveRequest.current
        ? timelineSaveRequestRevision.current
        : timelineRevision.current;
      try {
        await runTimelineAutosave();
      } catch (reason) {
        // The failed request may have been an older revision already in flight
        // when generation requested an exact flush. Continue only when a newer
        // project revision has replaced it; a failure of the required current
        // revision must still fail closed and prevent task submission.
        if (
          attemptedRevision === null ||
          timelineWriteGeneration.current !== generation ||
          timelineRevision.current <= attemptedRevision
        ) throw reason;
        continue;
      }
      if (
        timelinePersistedRevision.current === before &&
        timelinePersistedRevision.current < timelineRevision.current
      ) {
        await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      }
    }
    const current = normalizeTimelineProject(structuredClone(timelineRef.current.project));
    if (!current) throw new Error("时间线结构无效，请检查项目字段");
    return current;
  }, [runTimelineAutosave]);
  flushTimelineAutosaveRef.current = flushTimelineAutosave;


  useEffect(() => {
    const controller = new AbortController();
    const hydratingProjectId = activeProjectIdRef.current;
    const hydratingOwnerId = timelineBranchOwnerRef.current;
    const hydrationGeneration = ++timelineHydrationGeneration.current;
    const ownsHydration = () => isCurrentTimelineHydration(
      controller.signal,
      hydratingProjectId,
      activeProjectIdRef.current,
      hydrationGeneration,
      timelineHydrationGeneration.current,
    );
    let timelineHydrationRetryTimer: number | null = null;
    let timelineHydrationAttempt = 0;
    taskListOwnerActive.current = true;
    const timelineGeneration = timelineWriteGeneration.current;
    const runtimeAuthorityOperation = refreshRuntime(controller.signal);
    // StrictMode performs setup -> cleanup -> setup. Queue the second setup
    // behind the first request so its cleanup abort cannot leave the initial
    // task history empty until the next polling interval.
    void loadTasks(controller.signal, true);
    const installHydratedProject = (
      project: TimelineProject,
      database: { active_database_path: string },
      projectId: string,
      history: TimelineHistoryState,
    ) => {
      const segmentIds = project.segments.map((segment) => segment.id);
      const restoredSelection = loadTimelineSegmentSelectionPreference(
        database,
        projectId,
        segmentIds,
      );
      installTimelineAuthority(project, {
        projectId,
        selectedSegmentIds: restoredSelection ?? segmentIds,
        clearHistory: false,
      });
      commitTimelineHistory(history);
      restoredSegmentSelectionKey.current =
        `${database.active_database_path}:${projectId}`;
    };
    const hydrateTimeline = async (): Promise<void> => {
      if (!ownsHydration() || timelineHydrationReady.current) return;
      try {
        const storage = await directorApi.getStorage(controller.signal);
        if (!ownsHydration()) return;
        const candidateDatabase = {
          active_database_path: storage.active_database_path,
        };
        const persistedRuntimeSettings = loadPendingRuntimeSettings(candidateDatabase);
        const walBranches = listLocalTimelineWalBranches(
          candidateDatabase,
          hydratingProjectId,
          hydratingOwnerId,
        );
        const wal = walBranches.owned?.wal ?? null;
        // A WAL is only evidence until the current CAS authority has been read.
        // IndexedDB history is classified against that same exact authority.
        let authority: TimelineAuthority;
        try {
          authority = await fetchTimelineForProject(hydratingProjectId, controller.signal);
        } catch (reason) {
          if (!ownsHydration()) return;
          if (!(
            reason instanceof ApiError &&
            reason.status === 404 &&
            hydratingProjectId !== DEFAULT_PROJECT_ID
          )) throw reason;

          // A missing persisted project is the one bootstrap path that cannot
          // wait for a successful timeline latch. Bind the membership decision
          // to the exact database observed before and after the scoped list.
          const listGeneration = ++projectListGeneration.current;
          const list = await directorApi.listProjects(controller.signal);
          if (!ownsHydration() || projectListGeneration.current !== listGeneration) return;
          const verification = await directorApi.getStorage(controller.signal);
          if (!ownsHydration() || projectListGeneration.current !== listGeneration) return;
          if (
            verification.active_database_path !== candidateDatabase.active_database_path
          ) throw new DatabaseIdentityChangedDuringHydrationError();
          if (list.projects.some((project) => project.id === hydratingProjectId)) {
            // GET and list disagreed inside one stable database. Retry instead
            // of guessing whether creation/deletion won the inter-request race.
            throw reason;
          }
          setProjects(list.projects);
          restartTimelineHydrationForProject(DEFAULT_PROJECT_ID);
          return;
        }
        if (!ownsHydration()) return;
        const serverProject = normalizeTimelineProject(authority.document);
        if (!serverProject) throw new Error("服务器返回的时间线结构无效");
        const persistenceAuthority = {
          document: serverProject,
          revision: authority.revision,
        };
        const persistenceScope = timelinePersistenceScope(
          candidateDatabase,
          hydratingProjectId,
          hydratingOwnerId,
        );
        const [journal, journalBranchList] = await Promise.all([
          loadTimelineHistoryJournal(persistenceScope, persistenceAuthority),
          listTimelineHistoryJournalBranches(persistenceScope, persistenceAuthority),
        ]);
        if (!ownsHydration()) return;
        const verification = await directorApi.getStorage(controller.signal);
        if (!ownsHydration()) return;
        if (
          verification.active_database_path !== candidateDatabase.active_database_path
        ) throw new DatabaseIdentityChangedDuringHydrationError();
        if (!ownsHydration()) return;
        activeDatabaseRef.current ??= candidateDatabase;
        // Assets are database-scoped. Timeline hydration itself does not wait
        // on the library.
        void runtimeAuthorityOperation.then((settings) => {
          if (!settings || !ownsHydration()) return;
          return loadAssets(controller.signal, false, {
            database: candidateDatabase,
          });
        }).catch(() => {
          // Asset discovery remains a soft dependency. A failed response stays
          // invisible and a later explicit refresh can retry.
        });
        if (runtimeSettingsDesired.current) {
          // A same-tab edit made while storage identity was loading is newer
          // than any persisted envelope and becomes the first scoped WAL.
          savePendingRuntimeSettings(runtimeSettingsDesired.current, candidateDatabase);
        } else if (persistedRuntimeSettings) {
          runtimeSettingsDesired.current = persistedRuntimeSettings;
          runtimeSettingsDesiredOwner.current = "settings-page";
          runtimeSettingsPausedDesiredRef.current = null;
          runtimeSettingsDraftValidRef.current = true;
          setRuntimeSettingsDraft(persistedRuntimeSettings);
          setRuntimeSettingsDraftValid(true);
          setRuntimeSettingsPausedError(null);
          setRuntimeAuthorityRequired(true);
        }
        if (runtimeSettingsDesired.current) {
          window.queueMicrotask(() => runtimeSettingsDrainRef.current());
        }
        if (
          !ownsHydration() ||
          timelineRevision.current > 0 || timelineHadLocal.current ||
          timelineWriteGeneration.current !== timelineGeneration
        ) {
          // A pending WAL or a newer exclusive authority owns the project. The
          // default in-memory document must never become writable merely because
          // this stale hydration request completed.
          return;
        }

        if ("token" in journal && journal.token) {
          timelineJournalTokens.current.set(
            timelineJournalTokenMapKey(
              candidateDatabase,
              hydratingProjectId,
              hydratingOwnerId,
            ),
            journal.token,
          );
        }
        const durableJournalBranches = journalBranchList.status === "available"
          ? journalBranchList.branches
          : [];
        const recovery = collectTimelineRecoveryBranches(
          walBranches,
          durableJournalBranches,
          persistenceAuthority,
        );
        const explicitRecoverySelectionRequired = recovery.pending.length > 1 ||
          recovery.pending.some(
          (branch) => branch.ownership !== "owned" || branch.status === "corrupt",
          );
        const walResolution = wal
          ? resolveLocalTimelineWal(wal, persistenceAuthority)
          : null;
        let hydratedProject = serverProject;
        let hydratedHistory = resetTimelineHistory(timelineHistoryRef.current);
        let hasPendingProject = false;
        let hydrationConflict: TimelineRevisionConflict | null = null;

        const journalCarriesHistory = journal.status === "restored" ||
          journal.status === "acknowledged";
        if (walResolution?.status === "conflict") {
          hydratedProject = walResolution.local_project;
          hasPendingProject = true;
          hydrationConflict = {
            projectId: hydratingProjectId,
            localProject: hydratedProject,
            serverAuthority: persistenceAuthority,
            source: "cas",
            resolving: false,
          };
          if (journalCarriesHistory && timelineProjectsEqual(journal.project, hydratedProject)) {
            hydratedHistory = journal.history;
          }
        } else if (journal.status === "conflict") {
          hydratedProject = journal.localProject;
          if (timelineProjectsEqual(journal.project, hydratedProject)) {
            hydratedHistory = journal.history;
          }
          hasPendingProject = true;
          hydrationConflict = {
            projectId: hydratingProjectId,
            localProject: hydratedProject,
            serverAuthority: persistenceAuthority,
            source: "cas",
            resolving: false,
          };
        } else if (walResolution?.status === "replay") {
          hydratedProject = walResolution.project;
          hasPendingProject = true;
          if (journalCarriesHistory && timelineProjectsEqual(journal.project, hydratedProject)) {
            hydratedHistory = journal.history;
          }
        } else if (walResolution?.status === "acknowledged") {
          if (journalCarriesHistory && timelineProjectsEqual(journal.project, serverProject)) {
            hydratedHistory = journal.history;
          }
        } else if (journalCarriesHistory) {
          hydratedHistory = journal.history;
          hydratedProject = journal.project;
          hasPendingProject = !timelineProjectsEqual(hydratedProject, serverProject);
        }

        if (explicitRecoverySelectionRequired) {
          // Foreign branches are never installed merely because they exist.
          // The server remains visible and the editor stays gated until the
          // user selects one branch or explicitly continues without deleting it.
          hydratedProject = serverProject;
          hydratedHistory = recovery.newestAcknowledgedHistory ??
            resetTimelineHistory(timelineHistoryRef.current);
          hasPendingProject = false;
          hydrationConflict = {
            projectId: hydratingProjectId,
            localProject: serverProject,
            serverAuthority: persistenceAuthority,
            source: "recovery-branches",
            resolving: false,
            recoveryBranches: recovery.pending,
            selectedRecoveryBranchId: null,
          };
        } else if (
          !hasPendingProject &&
          !hydrationConflict &&
          recovery.newestAcknowledgedHistory
        ) {
          // An acknowledged history branch cannot cause a PUT: its head is
          // already the exact server document. Clone only the newest history
          // into this owner; every foreign record remains untouched.
          hydratedHistory = recovery.newestAcknowledgedHistory;
        }

        acceptTimelineServerAuthority(persistenceAuthority);
        activeTimelineWalRef.current = wal;
        timelineRevision.current = hasPendingProject ? 1 : 0;
        timelinePersistedRevision.current = 0;
        timelineHadLocal.current = hasPendingProject || hydratedHistory.head !== null;
        installHydratedProject(
          hydratedProject,
          candidateDatabase,
          hydratingProjectId,
          hydratedHistory,
        );
        setTimelineRevisionConflict(hydrationConflict);
        setTimelineDirty(hasPendingProject);
        timelineHydrationReady.current = true;
        setTimelineHydrationStatus("ready");

        if (wal && walResolution?.status === "acknowledged" && !hydrationConflict) {
          clearLocalTimeline(wal);
        } else if (!wal && hasPendingProject && !hydrationConflict) {
          saveLocalTimeline(hydratedProject, candidateDatabase, hydratingProjectId);
        }
        if (!hydrationConflict) {
          persistTimelineJournal(
            hydratedHistory,
            persistenceAuthority,
            candidateDatabase,
            hydratingProjectId,
          );
        }
      } catch (reason) {
        if (!ownsHydration()) return;
        if (reason instanceof DatabaseIdentityChangedDuringHydrationError) {
          clearTimelineHistory();
          databaseIdentityStaleRef.current = true;
          timelineServerRevision.current = null;
          timelineServerProjectRef.current = null;
          timelineHydrationReady.current = false;
          setTimelineHydrationStatus("stale");
          setToast(reason.message);
          return;
        }
        setTimelineHydrationStatus("retrying");
        const delay = Math.min(5_000, 500 * 2 ** Math.min(timelineHydrationAttempt, 3));
        timelineHydrationAttempt += 1;
        timelineHydrationRetryTimer = window.setTimeout(() => {
          timelineHydrationRetryTimer = null;
          if (ownsHydration()) void hydrateTimeline();
        }, delay);
      }
    };
    void hydrateTimeline();
    return () => {
      taskListOwnerActive.current = false;
      if (timelineHydrationRetryTimer !== null) window.clearTimeout(timelineHydrationRetryTimer);
      controller.abort();
    };
  }, [acceptTimelineServerAuthority, clearLocalTimeline, clearTimelineHistory, commitTimelineHistory, installTimelineAuthority, persistTimelineJournal, refreshRuntime, loadTasks, loadAssets, saveLocalTimeline, setTimelineRevisionConflict, timelineHydrationEpoch]);

  useEffect(() => { saveDirectorState(state); }, [state]);
  useEffect(() => {
    // A normal list is accepted only after timeline hydration has latched the
    // database. The 404 bootstrap fallback above owns the pre-latch exception.
    if (timelineHydrationStatus !== "ready") return;
    const database = activeDatabaseRef.current;
    if (!database) return;
    const controller = new AbortController();
    const projectId = activeProjectIdRef.current;
    const hydrationGeneration = timelineHydrationGeneration.current;
    const switchGeneration = projectSwitchGeneration.current;
    const listGeneration = ++projectListGeneration.current;
    const ownsList = () => !controller.signal.aborted &&
      timelineHydrationReady.current &&
      timelineHydrationStatus === "ready" &&
      projectListGeneration.current === listGeneration &&
      timelineHydrationGeneration.current === hydrationGeneration &&
      projectSwitchGeneration.current === switchGeneration &&
      activeProjectIdRef.current === projectId &&
      activeDatabaseRef.current?.active_database_path === database.active_database_path;
    void directorApi.listProjects(controller.signal).then((list) => {
      if (!ownsList()) return;
      setProjects(list.projects);
      if (
        projectId !== DEFAULT_PROJECT_ID &&
        !list.projects.some((project) => project.id === projectId)
      ) {
        restartTimelineHydrationForProject(DEFAULT_PROJECT_ID);
      }
    }).catch(() => {
      // The active timeline remains usable when this soft dependency fails.
      // A later hydration/project transition starts a fresh scoped request.
    });
    return () => controller.abort();
  }, [activeProjectId, restartTimelineHydrationForProject, timelineHydrationStatus]);
  useEffect(() => {
    if (timelineHydrationStatus !== "ready") return;
    // Project hand-off updates the synchronous owner before React publishes
    // its matching state. An intermediate render must not apply the previous
    // project's same-ID selection preference to the new document.
    if (activeProjectId !== activeProjectIdRef.current) return;
    const activeDatabase = activeDatabaseRef.current;
    if (!activeDatabase) return;
    const projectSegmentIds = timeline.project.segments.map((segment) => segment.id);
    const restoreKey = `${activeDatabase.active_database_path}:${activeProjectId}`;
    if (restoredSegmentSelectionKey.current === restoreKey) return;
    restoredSegmentSelectionKey.current = restoreKey;
    const restored = loadTimelineSegmentSelectionPreference(
      activeDatabase,
      activeProjectId,
      projectSegmentIds,
    );
    if (restored === null) return;
    const action: TimelineAction = { type: "segment/set-selection", ids: restored };
    dispatchTimeline(action, { history: "skip" });
  }, [activeProjectId, dispatchTimeline, timeline.project.segments, timelineHydrationStatus]);
  useEffect(() => {
    if (
      timelineHydrationStatus !== "ready" || timelineRevisionConflict ||
      timelineSyncRequired || timelineServerRevision.current === null ||
      timelinePersistedRevision.current < timelineRevision.current
    ) return;
    // Clean authority installs inside exclusive mutation owners also flow
    // through this single base mirror, so later WAL writes never reuse an old
    // document merely because that owner did not participate in hydration.
    timelineServerProjectRef.current = structuredClone(timeline.project);
  }, [timeline.project, timelineHydrationStatus, timelineRevisionConflict, timelineSyncRequired]);
  useEffect(() => {
    if (timelineHydrationStatus !== "ready") return;
    if (timelineRevisionConflict || timelineSyncRequired) {
      // Do not let a queued pre-conflict put replace forensic journal bytes.
      timelineJournalGeneration.current += 1;
      return;
    }
    const database = activeDatabaseRef.current;
    const revision = timelineServerRevision.current;
    const baseProject = timelineServerProjectRef.current;
    if (!database || revision === null || !baseProject) return;
    if (
      timelineHistory.head &&
      !timelineProjectsEqual(timelineHistory.head, timeline.project)
    ) return;
    persistTimelineJournal(
      timelineHistory,
      { document: baseProject, revision },
      database,
      activeProjectId,
    );
  }, [activeProjectId, persistTimelineJournal, timeline.project, timelineHistory, timelineHydrationStatus, timelineRevisionConflict, timelineSyncRequired]);
  useEffect(() => {
    if (
      timelineSyncRequired ||
      timelineRevisionConflict ||
      assetDeleteLock.current ||
      timelinePersistedRevision.current >= timelineRevision.current
    ) return;
    // Browser storage is a crash-recovery WAL, not a second authority. Remove
    // it as soon as this exact revision is confirmed by the server.
    const activeDatabase = activeDatabaseRef.current;
    if (!activeDatabase) return;
    saveLocalTimeline(timeline.project, activeDatabase, activeProjectIdRef.current);
    if (timelineAutosaveTimer.current !== null) {
      window.clearTimeout(timelineAutosaveTimer.current);
    }
    if (timelineRetryTimer.current !== null) {
      window.clearTimeout(timelineRetryTimer.current);
      timelineRetryTimer.current = null;
    }
    const timer = window.setTimeout(() => {
      if (timelineAutosaveTimer.current === timer) timelineAutosaveTimer.current = null;
      void runTimelineAutosave().catch(() => undefined);
    }, 150);
    timelineAutosaveTimer.current = timer;
    return () => {
      window.clearTimeout(timer);
      if (timelineAutosaveTimer.current === timer) timelineAutosaveTimer.current = null;
    };
  }, [timeline.project, timelineRetryNonce, timelineSyncRequired, timelineRevisionConflict, runTimelineAutosave]);
  useEffect(() => () => {
    if (timelineAutosaveTimer.current !== null) window.clearTimeout(timelineAutosaveTimer.current);
    if (timelineRetryTimer.current !== null) window.clearTimeout(timelineRetryTimer.current);
    if (timelineAuthorityRetryTimer.current !== null) window.clearTimeout(timelineAuthorityRetryTimer.current);
  }, []);
  useEffect(() => {
    const persistPendingTimeline = () => {
      if (
        !timelineSyncRequiredRef.current &&
        !timelineRevisionConflictRef.current &&
        timelinePersistedRevision.current < timelineRevision.current
      ) {
        const activeDatabase = activeDatabaseRef.current;
        if (activeDatabase) saveLocalTimeline(timelineRef.current.project, activeDatabase, activeProjectIdRef.current);
      }
    };
    window.addEventListener("pagehide", persistPendingTimeline);
    window.addEventListener("beforeunload", persistPendingTimeline);
    return () => {
      window.removeEventListener("pagehide", persistPendingTimeline);
      window.removeEventListener("beforeunload", persistPendingTimeline);
    };
  }, []);
  useEffect(() => {
    saveAssetLayoutPreference(timeline.asset_grid_size, timeline.assets);
  }, [timeline.asset_grid_size, timeline.assets]);
  useEffect(() => {
    try { window.localStorage.setItem(SIDEBAR_OPEN_KEY, String(sidebarOpen)); }
    catch { /* In-memory layout state remains usable. */ }
  }, [sidebarOpen]);
  useEffect(() => {
    try { window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth)); }
    catch { /* In-memory layout state remains usable. */ }
  }, [sidebarWidth]);
  useEffect(() => {
    if (!projectTitleEditing) return;
    projectTitleInputRef.current?.focus();
    projectTitleInputRef.current?.select();
  }, [projectTitleEditing]);
  useEffect(() => {
    const clampToViewport = () => {
      const viewportWidth = window.innerWidth;
      setSidebarViewportWidth(viewportWidth);
      setSidebarWidth((current) => clampSidebarWidth(current, viewportWidth));
    };
    window.addEventListener("resize", clampToViewport);
    return () => window.removeEventListener("resize", clampToViewport);
  }, []);
  useEffect(() => {
    if (!sidebarOpen) return;
    const closeFocusedSidebarOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      const sidebar = document.getElementById("director-sidebar");
      if (!sidebar?.contains(document.activeElement)) return;
      event.preventDefault();
      event.stopPropagation();
      setSidebarOpenWithFocus(false);
    };
    document.addEventListener("keydown", closeFocusedSidebarOnEscape);
    return () => document.removeEventListener("keydown", closeFocusedSidebarOnEscape);
  }, [sidebarOpen, setSidebarOpenWithFocus]);
  useEffect(() => { persistUiTheme(theme); }, [theme]);

  useEffect(() => {
    if (state.view === "workspace") return;
    setTimelineHistoryPanelOpen(false);
    setAssetTrashPanelOpen(false);
  }, [state.view]);

  useEffect(() => {
    setTimelineHistoryPanelOpen(false);
    setAssetTrashPanelOpen(false);
  }, [activeProjectId]);

  useEffect(() => {
    if (!globalSettingsOpen || state.view !== "workspace") return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      const popover = document.getElementById(GLOBAL_SETTINGS_ID);
      if (popover?.contains(target) || globalSettingsToggleRef.current?.contains(target)) return;
      setGlobalSettingsOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (document.querySelector(".task-lightbox,.task-error-dialog,.task-action-menu")) return;
      event.preventDefault();
      setGlobalSettingsOpen(false);
      window.requestAnimationFrame(() => globalSettingsToggleRef.current?.focus());
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer, true);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer, true);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [globalSettingsOpen, state.view]);

  const hasActiveTasks = state.tasks.some((task) =>
    ["queued", "preparing", "running", "cancelling"].includes(task.status));

  useEffect(() => {
    if (typeof EventSource === "undefined") return;
    const refreshTasks = () => void loadTasks(undefined, true);
    let eventSource: EventSource | null = null;
    // Defer one turn so React StrictMode's development-only setup/cleanup
    // probe cannot open a real SSE socket that an intermediate proxy retains.
    const connectTimer = window.setTimeout(() => {
      eventSource = new EventSource(taskEventsUrl());
      eventSource.addEventListener("refresh", refreshTasks);
    }, 0);
    return () => {
      window.clearTimeout(connectTimer);
      eventSource?.removeEventListener("refresh", refreshTasks);
      eventSource?.close();
    };
  }, [loadTasks]);

  useEffect(() => {
    const supportsSSE = typeof EventSource !== "undefined";
    // With SSE the browser refreshes as soon as the backend observes a change
    // (websocket progress/terminal frames, the reconciler's active-job pass, or
    // a wait-gate hint); the interval only guards against a dropped event or a
    // reconnect gap. Without SSE the interval stays the sole refresh mechanism.
    const timer = window.setInterval(
      () => void loadTasks(undefined, true),
      hasActiveTasks ? (supportsSSE ? 10_000 : 2500) : 30_000,
    );
    return () => window.clearInterval(timer);
  }, [hasActiveTasks, loadTasks]);

  useEffect(() => {
    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") void loadTasks(undefined, true);
    };
    document.addEventListener("visibilitychange", refreshWhenVisible);
    return () => document.removeEventListener("visibilitychange", refreshWhenVisible);
  }, [loadTasks]);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 4000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const timelineAuthorityScopeStillCurrent = (
    database: ActiveDatabaseIdentity,
    projectId: string,
  ): boolean => {
    const activeDatabase = activeDatabaseRef.current;
    return activeProjectIdRef.current === projectId &&
      !databaseIdentityStaleRef.current &&
      activeDatabase?.active_database_path === database.active_database_path;
  };

  const resyncTimeline = async (): Promise<void> => {
    // A CAS conflict is a user-owned branch decision. Automatic authority
    // recovery must never overwrite its preserved local document.
    if (timelineRevisionConflictRef.current) return;
    const scheduleRetry = (delayMs: number) => {
      if (!timelineSyncRequiredRef.current || timelineRevisionConflictRef.current) return;
      if (timelineAuthorityRetryTimer.current !== null) {
        window.clearTimeout(timelineAuthorityRetryTimer.current);
      }
      timelineAuthorityRetryTimer.current = window.setTimeout(() => {
        timelineAuthorityRetryTimer.current = null;
        void resyncTimeline();
      }, delayMs);
    };
    if (assetDeleteLock.current || assetDeleteIntent.current) {
      scheduleRetry(500);
      return;
    }
    const projectId = activeProjectIdRef.current;
    const database = activeDatabaseRef.current;
    if (!database || !timelineAuthorityScopeStillCurrent(database, projectId)) return;
    const generation = ++timelineWriteGeneration.current;
    try {
      const authority = await fetchTimelineForProject(projectId);
      const authoritative = normalizeTimelineProject(authority.document);
      if (!authoritative) throw new Error("服务器返回的时间线结构无效");
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, projectId)
      ) {
        // Another authority path won while this GET was in flight. It may not
        // own this explicit recovery gate (for example a Broadcast hint), so
        // retry unless that winner already completed the boundary.
        scheduleRetry(0);
        return;
      }
      acceptTimelineServerAuthority({ document: authoritative, revision: authority.revision });
      timelineRevisionChannelRef.current?.acceptKnown({
        revision: authority.revision,
        documentHash: timelineDocumentHash(authoritative),
      });
      installTimelineAuthority(authoritative);
      timelinePersistedRevision.current = timelineRevision.current;
      setTimelineAuthorityRequired(false);
      setTimelineDirty(false);
      setTimelinePausedError(null);
      clearLocalTimeline();
      persistTimelineJournal(
        timelineHistoryRef.current,
        { document: authoritative, revision: authority.revision },
        database,
        projectId,
      );
      if (timelineAuthorityRetryTimer.current !== null) {
        window.clearTimeout(timelineAuthorityRetryTimer.current);
        timelineAuthorityRetryTimer.current = null;
      }
      invalidateAndRefreshTaskSnapshots();
      if (runtimeSettingsDesired.current) {
        if (runtimeSettingsAutosaveTimer.current !== null) {
          window.clearTimeout(runtimeSettingsAutosaveTimer.current);
          runtimeSettingsAutosaveTimer.current = null;
        }
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        window.queueMicrotask(() => runtimeSettingsDrainRef.current());
      }
    } catch (reason) {
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, projectId)
      ) {
        return;
      }
      setToast(`${reason instanceof Error ? reason.message : "服务器时间线同步失败"}；正在自动重试`);
      scheduleRetry(1200);
    }
  };

  const adoptServerTimelineAfterConflict = async (): Promise<void> => {
    const conflict = timelineRevisionConflictRef.current;
    if (!conflict || conflict.resolving || conflict.projectId !== activeProjectIdRef.current) return;
    const database = activeDatabaseRef.current;
    if (!database || !timelineAuthorityScopeStillCurrent(database, conflict.projectId)) return;
    const resolving = { ...conflict, resolving: true };
    setTimelineRevisionConflict(resolving);
    const generation = ++timelineWriteGeneration.current;
    try {
      // Read again on the click: the diagnostic authority captured when the
      // conflict appeared may itself already be stale.
      const authority = await fetchTimelineForProject(conflict.projectId);
      const serverProject = normalizeTimelineProject(authority.document);
      if (!serverProject) throw new Error("服务器返回的时间线结构无效");
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      acceptTimelineServerAuthority({ document: serverProject, revision: authority.revision });
      timelineRevisionChannelRef.current?.acceptKnown({
        revision: authority.revision,
        documentHash: timelineDocumentHash(serverProject),
      });
      installTimelineAuthority(serverProject);
      timelinePersistedRevision.current = timelineRevision.current;
      timelineHadLocal.current = true;
      setTimelineRevisionConflict(null);
      setTimelineAuthorityRequired(false);
      setTimelineDirty(false);
      setTimelinePausedError(null);
      clearLocalTimeline();
      persistTimelineJournal(
        timelineHistoryRef.current,
        { document: serverProject, revision: authority.revision },
        database,
        conflict.projectId,
      );
      invalidateAndRefreshTaskSnapshots();
      if (runtimeSettingsDesired.current) {
        if (runtimeSettingsAutosaveTimer.current !== null) {
          window.clearTimeout(runtimeSettingsAutosaveTimer.current);
          runtimeSettingsAutosaveTimer.current = null;
        }
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        window.queueMicrotask(() => runtimeSettingsDrainRef.current());
      }
      setToast("已采用服务器时间线，本地冲突草稿已解除");
    } catch (reason) {
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      setTimelineRevisionConflict({ ...resolving, resolving: false });
      setToast(`${reason instanceof Error ? reason.message : "读取服务器时间线失败"}；本地草稿仍被保留`);
    }
  };

  const selectTimelineRecoveryBranch = (branchId: string): void => {
    const conflict = timelineRevisionConflictRef.current;
    if (!conflict || conflict.source !== "recovery-branches" || conflict.resolving) return;
    if (!conflict.recoveryBranches?.some((branch) => branch.id === branchId)) return;
    setTimelineRevisionConflict({
      ...conflict,
      selectedRecoveryBranchId: branchId,
    });
  };

  const continueServerWithRecoveryEvidence = async (): Promise<void> => {
    const conflict = timelineRevisionConflictRef.current;
    if (
      !conflict || conflict.source !== "recovery-branches" || conflict.resolving ||
      conflict.projectId !== activeProjectIdRef.current
    ) return;
    const database = activeDatabaseRef.current;
    if (!database || !timelineAuthorityScopeStillCurrent(database, conflict.projectId)) return;
    const resolving = { ...conflict, resolving: true };
    setTimelineRevisionConflict(resolving);
    const generation = ++timelineWriteGeneration.current;
    try {
      const authority = await fetchTimelineForProject(conflict.projectId);
      const serverProject = normalizeTimelineProject(authority.document);
      if (!serverProject) throw new Error("服务器返回的时间线结构无效");
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      const retainedHistory = timelineHistoryRef.current.head &&
          timelineProjectsEqual(timelineHistoryRef.current.head, serverProject)
        ? timelineHistoryRef.current
        : resetTimelineHistory(timelineHistoryRef.current);
      timelineJournalGeneration.current += 1;
      timelineBranchOwnerRef.current = createTimelineBranchOwnerId();
      activeTimelineWalRef.current = null;
      acceptTimelineServerAuthority({ document: serverProject, revision: authority.revision });
      installTimelineAuthority(serverProject, { clearHistory: false });
      commitTimelineHistory(retainedHistory);
      timelineRevision.current = 0;
      timelinePersistedRevision.current = 0;
      timelineHadLocal.current = retainedHistory.head !== null;
      setTimelineRevisionConflict(null);
      setTimelineAuthorityRequired(false);
      setTimelineDirty(false);
      setTimelinePausedError(null);
      persistTimelineJournal(
        retainedHistory,
        { document: serverProject, revision: authority.revision },
        database,
        conflict.projectId,
      );
      setToast("已继续使用服务器版本；所有本地恢复记录仍保留，可稍后显式处理");
    } catch (reason) {
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      setTimelineRevisionConflict({ ...resolving, resolving: false });
      setToast(`${reason instanceof Error ? reason.message : "读取服务器时间线失败"}；恢复记录仍被保留`);
    }
  };

  const restoreSelectedTimelineBranch = async (): Promise<void> => {
    const conflict = timelineRevisionConflictRef.current;
    if (
      !conflict || conflict.source !== "recovery-branches" || conflict.resolving ||
      conflict.projectId !== activeProjectIdRef.current
    ) return;
    const selected = conflict.recoveryBranches?.find(
      (branch) => branch.id === conflict.selectedRecoveryBranchId,
    );
    if (!selected?.project || selected.status === "corrupt") return;
    const database = activeDatabaseRef.current;
    if (!database || !timelineAuthorityScopeStillCurrent(database, conflict.projectId)) return;
    const resolving = { ...conflict, resolving: true };
    setTimelineRevisionConflict(resolving);
    const generation = ++timelineWriteGeneration.current;
    try {
      const authority = await fetchTimelineForProject(conflict.projectId);
      const serverProject = normalizeTimelineProject(authority.document);
      if (!serverProject) throw new Error("服务器返回的时间线结构无效");
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;

      let recoveryStatus: "replay" | "acknowledged" | "conflict" = "conflict";
      let baseAuthority: TimelinePersistenceAuthority | null = null;
      if (selected.walEvidence?.kind === "wal") {
        const wal = selected.walEvidence.wal;
        recoveryStatus = resolveLocalTimelineWal(wal, {
          document: serverProject,
          revision: authority.revision,
        }).status;
        baseAuthority = {
          document: wal.base_project,
          revision: wal.base_server_revision,
        };
      } else if (
        selected.journalEvidence &&
        selected.journalEvidence.status !== "corrupt"
      ) {
        const journal = selected.journalEvidence;
        baseAuthority = {
          document: journal.confirmedDocument,
          revision: journal.confirmedRevision,
        };
        recoveryStatus = timelineProjectsEqual(serverProject, journal.project)
          ? "acknowledged"
          : authority.revision === journal.confirmedRevision &&
              timelineProjectsEqual(serverProject, journal.confirmedDocument)
            ? "replay"
            : "conflict";
      }
      if (!baseAuthority) throw new Error("恢复记录缺少可验证的服务器基线");

      const restoredProject = recoveryStatus === "acknowledged"
        ? serverProject
        : selected.project;
      const restoredHistory = selected.history?.head &&
          timelineProjectsEqual(selected.history.head, restoredProject)
        ? selected.history
        : resetTimelineHistory(timelineHistoryRef.current);

      // Clone into a fresh owner so selecting B can never overwrite A, even
      // when A happened to be the branch previously owned by this realm.
      timelineJournalGeneration.current += 1;
      timelineBranchOwnerRef.current = createTimelineBranchOwnerId();
      activeTimelineWalRef.current = null;
      acceptTimelineServerAuthority({ document: serverProject, revision: authority.revision });
      installTimelineAuthority(restoredProject, { clearHistory: false });
      commitTimelineHistory(restoredHistory);
      timelineRevision.current = recoveryStatus === "acknowledged" ? 0 : 1;
      timelinePersistedRevision.current = 0;
      timelineHadLocal.current = recoveryStatus !== "acknowledged" || restoredHistory.head !== null;
      setTimelineDirty(recoveryStatus !== "acknowledged");
      setTimelineAuthorityRequired(false);
      setTimelinePausedError(null);

      if (recoveryStatus !== "acknowledged") {
        const clonedWal = saveLocalTimelineWal({
          database,
          project_id: conflict.projectId,
          owner_id: timelineBranchOwnerRef.current,
          base_server_revision: baseAuthority.revision,
          base_project: baseAuthority.document,
          pending_project: restoredProject,
        });
        if (!clonedWal) throw new Error("无法建立独立的本地恢复分支");
        activeTimelineWalRef.current = clonedWal;
        persistTimelineJournal(
          restoredHistory,
          baseAuthority,
          database,
          conflict.projectId,
        );
      } else {
        persistTimelineJournal(
          restoredHistory,
          { document: serverProject, revision: authority.revision },
          database,
          conflict.projectId,
        );
      }

      if (recoveryStatus === "conflict") {
        setTimelineRevisionConflict({
          projectId: conflict.projectId,
          localProject: restoredProject,
          serverAuthority: { document: serverProject, revision: authority.revision },
          source: "cas",
          resolving: false,
        });
        setToast("所选恢复分支的服务器基线已过期；已保留为本地草稿，请再次确认冲突处理方式");
      } else {
        setTimelineRevisionConflict(null);
        setToast(recoveryStatus === "acknowledged"
          ? "服务器已包含所选分支；已恢复其撤销历史"
          : "已恢复所选本地分支，正在按原服务器基线安全同步");
      }
    } catch (reason) {
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      setTimelineRevisionConflict({ ...resolving, resolving: false });
      setToast(`${reason instanceof Error ? reason.message : "恢复本地分支失败"}；原恢复记录未删除`);
    }
  };

  const discardTimelineRecoveryBranch = async (branchId: string): Promise<void> => {
    const conflict = timelineRevisionConflictRef.current;
    if (
      !conflict || conflict.source !== "recovery-branches" || conflict.resolving ||
      conflict.projectId !== activeProjectIdRef.current
    ) return;
    const branch = conflict.recoveryBranches?.find((candidate) => candidate.id === branchId);
    if (!branch) return;
    const label = branch.project?.title ?? "损坏的恢复记录";
    if (!window.confirm(`确认永久丢弃恢复记录“${label}”？此操作无法撤销。`)) return;
    const database = activeDatabaseRef.current;
    const hydrationGeneration = timelineHydrationGeneration.current;
    const switchGeneration = projectSwitchGeneration.current;
    if (!database || !timelineAuthorityScopeStillCurrent(database, conflict.projectId)) return;
    const operationStillCurrent = () =>
      hydrationGeneration === timelineHydrationGeneration.current &&
      switchGeneration === projectSwitchGeneration.current &&
      timelineAuthorityScopeStillCurrent(database, conflict.projectId);
    const resolving = { ...conflict, resolving: true };
    setTimelineRevisionConflict(resolving);
    const walDeleted = branch.walEvidence
      ? discardLocalTimelineWalBranch(branch.walEvidence)
      : true;
    let journalDeleted = branch.journalEvidence === null;
    if (branch.journalEvidence && branch.journalEvidence.token) {
      journalDeleted = await deleteTimelineHistoryJournal(
        timelinePersistenceScope(
          database,
          conflict.projectId,
          branch.ownerId ?? timelineBranchOwnerRef.current,
        ),
        branch.journalEvidence.token,
      );
    }
    if (!operationStillCurrent()) return;
    if (!walDeleted || !journalDeleted) {
      setTimelineRevisionConflict({ ...resolving, resolving: false });
      setToast("恢复记录仅部分完成精确删除，或其中一侧已被其他页面更新；其余证据仍保留，请重试或重新加载");
      return;
    }
    const remaining = conflict.recoveryBranches?.filter((candidate) => candidate.id !== branchId) ?? [];
    if (remaining.length === 0) {
      setTimelineRevisionConflict(null);
      setToast("恢复记录已丢弃；继续使用服务器版本");
      return;
    }
    setTimelineRevisionConflict({
      ...conflict,
      resolving: false,
      recoveryBranches: remaining,
      selectedRecoveryBranchId: conflict.selectedRecoveryBranchId === branchId
        ? null
        : conflict.selectedRecoveryBranchId,
    });
    setToast("所选恢复记录已精确丢弃，其他分支保持不变");
  };

  const keepLocalTimelineAfterConflict = async (): Promise<void> => {
    const conflict = timelineRevisionConflictRef.current;
    if (!conflict || conflict.resolving || conflict.projectId !== activeProjectIdRef.current) return;
    if (!window.confirm("确认以当前本地时间线覆盖服务器版本？服务器在此期间的新修改可能被替换。")) return;
    const localProject = normalizeTimelineProject(structuredClone(timelineRef.current.project));
    if (!localProject) {
      setToast("本地时间线结构无效，无法保留");
      return;
    }
    const database = activeDatabaseRef.current;
    if (!database || !timelineAuthorityScopeStillCurrent(database, conflict.projectId)) return;
    const resolving: TimelineRevisionConflict = {
      ...conflict,
      localProject,
      resolving: true,
    };
    setTimelineRevisionConflict(resolving);
    const generation = ++timelineWriteGeneration.current;
    try {
      // Rebase the explicit overwrite decision onto the newest observed server
      // revision, then still use CAS to close the GET/PUT race.
      const latest = await fetchTimelineForProject(conflict.projectId);
      const latestDocument = normalizeTimelineProject(latest.document);
      if (!latestDocument) throw new Error("服务器返回的时间线结构无效");
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      const response = await runWithTimelineWriterLock({
        databasePath: database.active_database_path,
        projectId: conflict.projectId,
      }, () => saveTimelineForProject(
        conflict.projectId,
        localProject,
        latest.revision,
      ));
      const confirmed = normalizeTimelineProject(response.document);
      if (!confirmed) throw new Error("服务器返回的时间线结构无效");
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      acceptTimelineServerAuthority({ document: confirmed, revision: response.revision });
      timelineRevisionChannelRef.current?.publish({
        revision: response.revision,
        documentHash: timelineDocumentHash(confirmed),
      });
      timelinePersistedRevision.current = timelineRevision.current;
      timelineHadLocal.current = true;
      let durableHistory = timelineHistoryRef.current;
      if (!timelineProjectsEqual(localProject, confirmed)) {
        const rebased = rebaseTimelineHistoryHead(
          timelineHistoryRef.current,
          localProject,
          confirmed,
        );
        if (rebased) {
          durableHistory = rebased;
          commitTimelineHistory(rebased);
        } else {
          durableHistory = resetTimelineHistory(timelineHistoryRef.current);
          clearTimelineHistory();
        }
      }
      installTimelineAuthority(confirmed, { clearHistory: false });
      setTimelineRevisionConflict(null);
      setTimelineAuthorityRequired(false);
      setTimelineDirty(false);
      setTimelinePausedError(null);
      clearLocalTimeline();
      persistTimelineJournal(
        durableHistory,
        { document: confirmed, revision: response.revision },
        database,
        conflict.projectId,
      );
      invalidateAndRefreshTaskSnapshots();
      if (runtimeSettingsDesired.current) runtimeSettingsDrainRef.current();
      setToast("已确认保留本地时间线并写入服务器");
    } catch (reason) {
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      let serverAuthority = resolving.serverAuthority;
      if (
        reason instanceof ApiError &&
        reason.status === 409 &&
        reason.code === "timeline_revision_conflict"
      ) {
        try {
          const latest = await fetchTimelineForProject(conflict.projectId);
          const latestDocument = normalizeTimelineProject(latest.document);
          if (
            latestDocument &&
            timelineWriteGeneration.current === generation &&
            timelineAuthorityScopeStillCurrent(database, conflict.projectId)
          ) {
            serverAuthority = { document: latestDocument, revision: latest.revision };
          }
        } catch {
          // Keep the previous diagnostic authority and the local WAL.
        }
      }
      if (
        timelineWriteGeneration.current !== generation ||
        !timelineAuthorityScopeStillCurrent(database, conflict.projectId)
      ) return;
      setTimelineRevisionConflict({
        ...resolving,
        serverAuthority,
        resolving: false,
      });
      setToast(`${reason instanceof Error ? reason.message : "保留本地时间线失败"}；本地草稿仍被保留`);
    }
  };

  const submitTimeline = async (segmentIds: string[]) => {
    if (submitting) return;
    if (
      rayLightRecoveryPendingRef.current ||
      !runtimeResourcesReadyRef.current ||
      !rayLightRuntimeStatusRef.current ||
      rayLightRuntimeStatusRef.current.recovery_required
    ) {
      setToast("ComfyUI 与 RayLight 运行状态尚未完成权威核对；暂不能生成");
      return;
    }
    if (!segmentIds.length) {
      setToast("请至少勾选一个要生成的片段");
      return;
    }
    if (runtimeSettingsOperation.current || runtimeSettingsSyncRequiredRef.current) {
      setToast("运行设置尚未完成服务器权威回读；暂不能生成");
      return;
    }
    if (assetUploadLock.current) {
      setToast("正在上传并绑定本地素材，完成前不能生成");
      return;
    }
    if (assetDeleteLock.current || assetDeleteIntent.current) { setToast("正在原子解除素材引用，请稍候"); return; }
    if (timelineRevisionConflict) { setToast("服务器时间线存在修订冲突，请先选择处理方式"); return; }
    if (timelineSyncRequired) { setToast("服务器时间线正在自动恢复同步，暂不能生成"); return; }
    const executionGeneration = runtimeSettingsGeneration.current;
    const clickedSegmentSelectionGeneration = segmentSelectionGeneration.current;
    const clickedProjectId = activeProjectIdRef.current;
    const executionSettings = structuredClone(authoritativeSettingsRef.current);
    let expectedTimelineRevision = timelineRevision.current;
    const clickedSegmentIds = [...segmentIds];
    // Starting a run is also a project-authority boundary. Cancel any target
    // project load that began while this project was still idle so its late
    // response cannot replace the project being submitted.
    projectSwitchGeneration.current += 1;
    runtimeExecutionIntent.current += 1;
    setSubmitting(true);
    try {
      let config = await flushTimelineAutosave();
      if (
        activeProjectIdRef.current !== clickedProjectId ||
        timelineRevision.current !== expectedTimelineRevision ||
        segmentSelectionGeneration.current !== clickedSegmentSelectionGeneration
      ) {
        throw new Error("项目、时间线或分段选择在生成确认期间发生变化，请重新生成");
      }
      let validationErrors = [
        ...validateTimelineProject(config, clickedSegmentIds),
        ...runtimeTimelineValidation(config, capabilities, state.settings, clickedSegmentIds),
      ];
      if (validationErrors.length) throw new Error(validationErrors[0]);

      // Re-roll random seeds exactly like one submission would; the same
      // concrete value goes into the flushed timeline and the task payload.
      config = await rerollRandomSeeds(config, clickedSegmentIds);
      expectedTimelineRevision = timelineRevision.current;
      validationErrors = [
        ...validateTimelineProject(config, clickedSegmentIds),
        ...runtimeTimelineValidation(config, capabilities, state.settings, clickedSegmentIds),
      ];
      if (validationErrors.length) throw new Error(validationErrors[0]);
      if (
        activeProjectIdRef.current !== clickedProjectId ||
        timelineRevision.current !== expectedTimelineRevision ||
        segmentSelectionGeneration.current !== clickedSegmentSelectionGeneration
      ) throw new Error("项目、时间线或分段选择在生成确认期间发生变化，请重新生成");
      if (
        runtimeSettingsGeneration.current !== executionGeneration ||
        !sameRuntimeSettings(authoritativeSettingsRef.current, executionSettings)
      ) throw new Error("运行设置权威状态已变化，请重新生成");
      const task = await submitTimelineForProject(clickedProjectId, {
        config: structuredClone(config),
        segment_ids: clickedSegmentIds,
      });
      // Invalidate a list request that may have captured the queue before this
      // task existed. Its late response must not replace the freshly upserted
      // parent and accidentally stop automatic polling.
      taskListRequest.current += 1;
      dispatch({ type: "tasks/upsert", task });
      setGlobalSettingsOpen(false);
      dispatch({ type: "tasks/panel", open: true });
      void loadTasks(undefined, true);
      setToast(`已提交 ${clickedSegmentIds.length} 个原生分段子图`);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "时间线任务提交失败");
    } finally {
      setSubmitting(false);
      runtimeExecutionIntent.current = Math.max(0, runtimeExecutionIntent.current - 1);
      if (runtimeExecutionIntent.current === 0 && runtimeSettingsDesired.current) {
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        runtimeSettingsDrainRef.current();
      }
    }
  };

  /**
   * Simulate the exact submission step for one run: re-roll every family the
   * selected segments use when its editor contract is `random_seed`, persist
   * the concrete value (updating the greyed-out seed fields), and return the
   * authoritative project. Both the generate and the preflight entries share
   * this so the flushed timeline, compile reports and the ComfyUI prompt
   * always carry the same JSON-safe seed.
   */
  const rerollRandomSeeds = async (
    config: TimelineProject,
    segmentIds: string[],
  ): Promise<TimelineProject> => {
    const selection = new Set(segmentIds);
    const usedFamilies = new Set(
      config.segments
        .filter((segment) => segment.enabled && selection.has(segment.id))
        .map((segment) => timelineSamplingFamily(segment.mode)),
    );
    let rerolled = false;
    for (const family of usedFamilies) {
      const sampling = config.sampling[family];
      if (!sampling.random_seed) continue;
      config.sampling[family] = {
        ...sampling,
        seed: nextRandomSeed(sampling.seed),
      };
      rerolled = true;
    }
    if (!rerolled) return config;
    // Keep the disabled fields equal to the exact value sent to the job.
    dispatchTimeline(
      { type: "project/replace", project: structuredClone(config) },
      { historyLabel: "更新随机 Seed" },
    );
    const expected = timelineRevision.current;
    const flushed = await flushTimelineAutosave();
    if (timelineRevision.current !== expected) {
      throw new Error("时间线在随机种子重掷期间发生变化，请重试");
    }
    return flushed;
  };

  const inspectTimelineExecution = async (segmentIds: string[]) => {
    if (compiling) return;
    if (
      rayLightRecoveryPendingRef.current ||
      !runtimeResourcesReadyRef.current ||
      !rayLightRuntimeStatusRef.current ||
      rayLightRuntimeStatusRef.current.recovery_required
    ) {
      setToast("ComfyUI 与 RayLight 运行状态尚未完成权威核对；暂不能预检");
      return;
    }
    if (!segmentIds.length) {
      setToast("请至少勾选一个要预检的片段");
      return;
    }
    if (runtimeSettingsOperation.current || runtimeSettingsSyncRequiredRef.current) {
      setToast("运行设置尚未完成服务器权威回读；暂不能预检执行计划");
      return;
    }
    if (assetUploadLock.current) { setToast("正在上传并绑定本地素材，完成前不能预检"); return; }
    if (assetDeleteLock.current || assetDeleteIntent.current) { setToast("正在原子解除素材引用，请稍候"); return; }
    if (timelineRevisionConflict) { setToast("服务器时间线存在修订冲突，请先选择处理方式"); return; }
    if (timelineSyncRequired) { setToast("服务器时间线正在自动恢复权威状态"); return; }
    const clickedSegmentSelectionGeneration = segmentSelectionGeneration.current;
    const clickedProjectId = activeProjectIdRef.current;
    const clickedSegmentIds = [...segmentIds];
    let config = normalizeTimelineProject(structuredClone(timelineRef.current.project));
    if (!config) { setToast("时间线结构无效，请检查项目字段"); return; }
    let validationErrors = [
      ...validateTimelineProject(config, clickedSegmentIds),
      ...runtimeTimelineValidation(config, capabilities, state.settings, clickedSegmentIds),
    ];
    if (validationErrors.length) { setToast(validationErrors[0]); return; }
    const executionGeneration = runtimeSettingsGeneration.current;
    const executionSettings = structuredClone(authoritativeSettingsRef.current);
    // A pending project GET must not land while preflight owns this project.
    projectSwitchGeneration.current += 1;
    runtimeExecutionIntent.current += 1;
    setCompiling(true);
    try {
      // 与生成入口一致：预检前也模拟一次提交，重掷随机 seed 并持久化，
      // 让 compile 报告携带这次预检实际使用的值，灰显数字框同步显示。
      config = await rerollRandomSeeds(config, clickedSegmentIds);
      const clickedTimelineRevision = timelineRevision.current;
      validationErrors = [
        ...validateTimelineProject(config, clickedSegmentIds),
        ...runtimeTimelineValidation(config, capabilities, state.settings, clickedSegmentIds),
      ];
      if (validationErrors.length) { setToast(validationErrors[0]); return; }
      const report = await compileTimelineForProject(clickedProjectId, { config: structuredClone(config), segment_ids: clickedSegmentIds });
      if (
        activeProjectIdRef.current !== clickedProjectId ||
        timelineRevision.current !== clickedTimelineRevision ||
        segmentSelectionGeneration.current !== clickedSegmentSelectionGeneration ||
        runtimeSettingsGeneration.current !== executionGeneration ||
        !sameRuntimeSettings(authoritativeSettingsRef.current, executionSettings) ||
        runtimeSettingsDesired.current !== null ||
        runtimeSettingsSyncRequiredRef.current
      ) {
        setToast("项目、时间线、分段选择或运行设置已变化，请重新预检");
        return;
      }
      setCompileReport(report);
      setToast(`已预检 ${report.plans.length} 个服务端原生执行计划`);
    } catch (reason) {
      setCompileReport(null);
      setToast(reason instanceof Error ? reason.message : "执行计划预检失败");
    } finally {
      setCompiling(false);
      runtimeExecutionIntent.current = Math.max(0, runtimeExecutionIntent.current - 1);
      if (runtimeExecutionIntent.current === 0 && runtimeSettingsDesired.current) {
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        runtimeSettingsDrainRef.current();
      }
    }
  };

  const updateRuntimeModel = async (
    role: DiffusionModelRole,
    patch: Partial<DiffusionModelBinding>,
  ) => {
    setCompileReport(null);
    const base = runtimeSettingsDesired.current ?? runtimeSettingsDraft;
    const currentBinding = base.models[role];
    const selectedArtifactChanged =
      ("lora_name" in patch && patch.lora_name !== currentBinding.lora_name) ||
      ("filename" in patch && patch.filename !== currentBinding.filename);
    const nextBinding = {
      ...currentBinding,
      ...patch,
      ...(selectedArtifactChanged ? { standard_lora_loader_override: null } : {}),
    };
      const nextResidencyPolicy = rayLightResidencyPolicyAfterBindingChange(
        base,
        role,
        nextBinding,
      );
      const next = sanitizeRuntimeSettings({
        ...base,
        raylight_residency_policy: nextResidencyPolicy,
        models: {
          ...base.models,
          [role]: nextBinding,
        },
      });
    void queueRuntimeSettings(role, next);
  };

  const cancel = async (id: string) => {
    try {
      const task = await directorApi.cancelTask(id, activeProjectIdRef.current);
      taskListRequest.current += 1;
      dispatch({ type: "tasks/upsert", task });
      void loadTasks(undefined, true);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "取消失败");
    }
  };
  const confirmComfyRestartRecovery = async (id: string) => {
    try {
      const task = await directorApi.confirmComfyRestartRecovery(
        id,
        activeProjectIdRef.current,
      );
      dispatch({ type: "tasks/upsert", task });
      setToast("已确认 ComfyUI 重启，导演台任务已结束");
    } catch (reason) {
      const detail = reason instanceof Error ? reason.message : "恢复确认失败";
      setToast(`恢复确认失败：${detail}`);
    } finally {
      taskListRequest.current += 1;
      void loadTasks(undefined, true);
    }
  };
  const confirmRayLightRuntimeRecovery = (): Promise<void> => {
    if (rayLightRecoveryOperationRef.current) {
      return rayLightRecoveryOperationRef.current;
    }
    const expected = rayLightRuntimeStatusRef.current;
    if (
      !runtimeResourcesReadyRef.current ||
      !expected?.recovery_required ||
      !expected.recovery_token
    ) return Promise.reject(new Error("RayLight 恢复状态已变化，请刷新系统设置后重试"));
    if (
      runtimeSettingsOperation.current ||
      runtimeSettingsDesired.current ||
      runtimeSettingsSyncRequiredRef.current
    ) {
      return Promise.reject(new Error("运行设置仍在同步，请完成后再恢复 RayLight"));
    }
    if (databaseIdentityStaleRef.current) {
      return Promise.reject(new Error("数据库状态尚未稳定，暂不能恢复 RayLight"));
    }
    const expectedRecoveryToken = expected.recovery_token;
    const controller = new AbortController();
    rayLightRecoveryControllerRef.current = controller;

    const operation = (async () => {
      rayLightRecoveryPendingRef.current = true;
      setRayLightRecoveryPending(true);
      setRuntimeAuthorityRequired(true);
      setCompileReport(null);
      let mutationMayHaveCommitted = false;
      let attempt = 0;
      try {
        for (;;) {
          await waitForRayLightRecoveryWindow(0, controller.signal);
          if (databaseIdentityStaleRef.current) {
            throw new Error("恢复核对期间数据库状态发生变化");
          }

          let deterministicFailure = false;
          let postError: unknown = null;
          try {
            await directorApi.confirmRayLightRuntimeRecovery(
              expected.epoch,
              expectedRecoveryToken,
              controller.signal,
            );
            mutationMayHaveCommitted = true;
          } catch (reason) {
            throwIfAborted(controller.signal);
            postError = reason;
            const clientFailure = reason instanceof ApiError &&
              reason.status >= 400 && reason.status < 500;
            const retryableInFlight = clientFailure &&
              mutationMayHaveCommitted &&
              reason.code === "raylight_recovery_in_flight";
            deterministicFailure = clientFailure && !retryableInFlight;
            // A lost response or 5xx does not prove that the mutation failed;
            // only the backend's allow-listed in-flight conflict remains
            // retryable after such an ambiguous attempt. Every other 4xx is a
            // definitive rejection even when ambiguity happened earlier.
            if (!clientFailure) mutationMayHaveCommitted = true;
          }

          // Mutation responses (including a nominal 200) never unlock the
          // workspace. Only a fresh, complete four-resource authority snapshot
          // for the same endpoint can certify that the ledger is clean.
          const refreshed = await refreshRuntimeResources(
            true,
            controller.signal,
            false,
          );
          throwIfAborted(controller.signal);
          const verified = rayLightRuntimeStatusRef.current;
          if (
            refreshed &&
            runtimeResourcesReadyRef.current &&
            verified &&
            !verified.recovery_required
          ) {
            setToast("已确认 ComfyUI 重启并恢复 RayLight；下次任务将建立新 GPU 池");
            return;
          }
          if (deterministicFailure) {
            throw postError instanceof Error
              ? postError
              : new Error("服务器拒绝 RayLight 恢复确认");
          }
          attempt += 1;
          const retryDelay = Math.min(
            RAYLIGHT_RECOVERY_RETRY_MS * (2 ** Math.min(attempt - 1, 2)),
            RAYLIGHT_RECOVERY_MAX_RETRY_MS,
          );
          await waitForRayLightRecoveryWindow(retryDelay, controller.signal);
        }
      } finally {
        rayLightRecoveryPendingRef.current = false;
        if (!controller.signal.aborted) setRayLightRecoveryPending(false);
        rayLightRecoveryOperationRef.current = null;
        if (rayLightRecoveryControllerRef.current === controller) {
          rayLightRecoveryControllerRef.current = null;
        }
        if (!controller.signal.aborted && runtimeSettingsDesired.current) {
          runtimeSettingsDrainRef.current();
        }
      }
    })();
    rayLightRecoveryOperationRef.current = operation;
    return operation;
  };
  const cancelMany = async (ids: string[]) => {
    if (!ids.length) return;
    let cancelledCount = 0;
    try {
      for (let index = 0; index < ids.length; index += 100) {
        const result = await directorApi.cancelTasks(
          ids.slice(index, index + 100),
          activeProjectIdRef.current,
        );
        cancelledCount += result.requested_count;
        result.jobs.forEach((task) => dispatch({ type: "tasks/upsert", task }));
      }
      taskListRequest.current += 1;
      void loadTasks(undefined, true);
      setToast(`已向 ${cancelledCount} 个导演台任务发送定向取消`);
    } catch (reason) {
      taskListRequest.current += 1;
      const detail = reason instanceof Error ? reason.message : "批量取消失败";
      setToast(cancelledCount > 0
        ? `已取消前 ${cancelledCount} 个任务，后续批次失败：${detail}`
        : detail);
      void loadTasks(undefined, true);
    }
  };
  const loadTaskProject = async (id: string) => {
    if (runtimeExecutionIntent.current > 0) {
      setToast("生成或预检正在确认当前项目；完成前不能另存历史项目");
      return;
    }
    if (timelineSyncRequiredRef.current || timelineRevisionConflictRef.current || assetDeleteLock.current || assetDeleteIntent.current || assetUploadLock.current || projectDeleteIntent.current !== null) {
      setToast("当前时间线或素材状态尚未稳定，暂不能另存历史项目");
      return;
    }
    try {
      if (timelineDirty) await flushTimelineAutosave();
      const snapshot = await directorApi.getTaskProject(id);
      // Restore the historical source as a brand-new project instead of
      // overwriting the one currently being edited.
      projectListGeneration.current += 1;
      const created = await directorApi.importProject({
        title: snapshot.project.title,
        document: snapshot.project,
      });
      setProjects((current) => [...current, created]);
      if (!await switchProject(created.id)) return;
      if (snapshot.segment_ids !== null && snapshot.segment_ids.length > 0) {
        dispatchTimeline({ type: "segment/set-selection", ids: snapshot.segment_ids });
      }
      dispatch({ type: "tasks/panel", open: false });
      setToast(`已把任务来源项目另存为新项目“${created.title}”；生成前仍会重新校验素材与当前环境`);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "来源项目另存失败");
    }
  };
  const importTaskOutput = async (
    id: string,
    output: { index: number; segmentId?: string },
  ) => {
    if (assetUploadLock.current || assetDeleteLock.current || assetDeleteIntent.current) {
      setToast("素材库正在处理其他操作，请稍候");
      return;
    }
    if (runtimeSettingsOperation.current || runtimeSettingsSyncRequiredRef.current) {
      setToast("运行设置尚未完成权威回读，暂不能导入任务输出");
      return;
    }
    const operationDatabase = activeDatabaseRef.current;
    if (!operationDatabase) {
      setToast("数据库权威状态尚未稳定，暂不能导入任务输出");
      return;
    }
    const operationProjectId = activeProjectIdRef.current;
    assetUploadLock.current = true;
    const operationSwitchGeneration = ++projectSwitchGeneration.current;
    const operationStillCurrent = () =>
      projectSwitchGeneration.current === operationSwitchGeneration &&
      activeProjectIdRef.current === operationProjectId &&
      activeDatabaseRef.current?.active_database_path === operationDatabase.active_database_path;
    setAssetsUploading(true);
    try {
      const asset = await directorApi.importTaskOutput(id, output);
      if (!operationStillCurrent()) {
        throw new Error("导入完成时项目或素材库权威范围已变化；结果未绑定到当前项目");
      }
      dispatchTimelineUi({ type: "assets/add", assets: [asset] });
      dispatchTimelineUi({ type: "assets/select", id: asset.id });
      await loadAssets(undefined, true, {
        database: operationDatabase,
      });
      if (!operationStillCurrent()) return;
      setToast(`已把 ${asset.name} 转为 24fps 输入并加入当前素材库`);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "任务输出导入失败");
    } finally {
      assetUploadLock.current = false;
      setAssetsUploading(false);
      if (
        !timelineSyncRequiredRef.current &&
        !timelineRevisionConflictRef.current &&
        timelinePersistedRevision.current < timelineRevision.current
      ) {
        setTimelineRetryNonce((current) => current + 1);
      }
      if (runtimeSettingsDesired.current) {
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        runtimeSettingsDrainRef.current();
      }
    }
  };
  const exportTaskDiagnostic = async (id: string) => {
    try {
      return await directorApi.getTaskDiagnostic(id);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "任务诊断导出失败");
      throw reason;
    }
  };
  const deleteTask = async (id: string) => {
    if (taskClearLock.current || taskDeleteLocks.current.has(id)) return;
    taskDeleteLocks.current.add(id);
    setDeletingTaskIds((current) => new Set(current).add(id));
    try {
      await directorApi.deleteTask(id);
      taskListRequest.current += 1;
      dispatch({ type: "tasks/remove", id });
      await loadTasks(undefined, true);
      setToast("任务记录已删除；ComfyUI 输出文件保留");
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "任务记录删除失败");
    } finally {
      taskDeleteLocks.current.delete(id);
      setDeletingTaskIds((current) => {
        const next = new Set(current);
        next.delete(id);
        return next;
      });
    }
  };
  const clearTerminalTasks = async () => {
    if (taskClearLock.current || taskDeleteLocks.current.size > 0) return;
    taskClearLock.current = true;
    setClearingTasks(true);
    try {
      const result = await directorApi.clearTerminalTasks();
      taskListRequest.current += 1;
      dispatch({ type: "tasks/clear-terminal" });
      await loadTasks(undefined, true);
      setToast(`已清理 ${result.deleted_count} 条任务记录；ComfyUI 输出文件保留`);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "任务记录清理失败");
    } finally {
      taskClearLock.current = false;
      setClearingTasks(false);
    }
  };
  const deleteAssets = async (ids: string[]) => {
    if (!ids.length || assetDeleteLock.current || assetDeleteIntent.current) return;
    const operationDatabase = activeDatabaseRef.current;
    if (!operationDatabase) {
      setToast("数据库权威状态尚未稳定，暂不能从素材库移出");
      return;
    }
    if (runtimeExecutionIntent.current > 0) {
      setToast("生成或预检正在使用当前素材与运行设置；完成后再从素材库移出");
      return;
    }
    if (assetUploadLock.current) {
      setToast("正在上传并绑定本地素材，完成前不能从素材库移出");
      return;
    }
    if (runtimeSettingsOperation.current || runtimeSettingsSyncRequiredRef.current) {
      setToast("运行设置尚未完成服务器权威回读；暂不能从素材库移出");
      return;
    }
    if (timelineRevisionConflict) { setToast("服务器时间线存在修订冲突，请先选择处理方式"); return; }
    if (timelineSyncRequired) { setToast("服务器时间线正在自动恢复同步，暂不能从素材库移出"); return; }
    const operationProjectId = activeProjectIdRef.current;
    assetDeleteIntent.current = true;
    const operationSwitchGeneration = ++projectSwitchGeneration.current;
    const timelineOperationScopeStillCurrent = () =>
      activeProjectIdRef.current === operationProjectId &&
      activeDatabaseRef.current?.active_database_path === operationDatabase.active_database_path;
    const assetOperationScopeStillCurrent = timelineOperationScopeStillCurrent;
    const operationStillCurrent = () =>
      projectSwitchGeneration.current === operationSwitchGeneration &&
      assetOperationScopeStillCurrent();
    setAssetsDeleting(true);
    let timelineAuthorityRequired = false;
    const prepareCascade = () => {
      if (timelineAuthorityRequired) return;
      timelineAuthorityRequired = true;
      // Clear the browser draft before the request: a lost response can still
      // mean the server committed. Only an authoritative GET may unlock it.
      setTimelineAuthorityRequired(true);
      timelineHadLocal.current = false;
      clearLocalTimeline();
    };
    try {
      // Finish the latest edit before acquiring exclusive cascade ownership.
      // The continuation runs in the same microtask, so a user edit cannot
      // slip between the exact revision confirmation and this lock.
      const snapshot = await flushTimelineAutosave();
      if (!operationStillCurrent()) {
        throw new Error("项目或素材库权威范围已变化，请在同步完成后重试");
      }
      assetDeleteLock.current = true;
      timelineWriteGeneration.current += 1;
      let cascade = ids.some((id) => timelineAssetUsages(snapshot, id).length > 0);
      if (cascade) prepareCascade();
      let batch: AssetTrashBatch;
      try {
        batch = await directorApi.trashAssets(ids, cascade);
      } catch (reason) {
        if (!(reason instanceof ApiError && reason.code === "assets_in_use" && !cascade)) {
          throw reason;
        }
        const detail = reason.details && typeof reason.details === "object"
          ? (reason.details as { detail?: { usages?: unknown } }).detail
          : undefined;
        const usages = Array.isArray(detail?.usages)
          ? detail.usages.filter((usage): usage is string => typeof usage === "string")
          : [];
        const usageLines = usages.length ? `\n\n${usages.join("\n")}` : "";
        if (!window.confirm(`所选素材仍被其他时间线草稿引用：${usageLines}\n\n是否在一个事务中从所有草稿解除引用并移至回收站？ComfyUI 文件会保留。`)) {
          setToast("已取消移至素材回收站；没有素材或项目被修改");
          return;
        }
        cascade = true;
        prepareCascade();
        batch = await directorApi.trashAssets(ids, true);
      }
      if (!operationStillCurrent()) {
        throw new Error("素材移出已提交，但项目权威范围已变化；正在重新核对结果");
      }

      // Even an unused asset can be referenced by an older history cursor.
      // The recycle-bin transaction is an explicit project-history boundary.
      clearTimelineHistory();
      setAssetTrashBatches((current) => [
        batch,
        ...current.filter((candidate) => candidate.batch_id !== batch.batch_id),
      ]);

      if (cascade) {
        const authority = await fetchTimelineForProject(operationProjectId);
        if (!operationStillCurrent()) {
          throw new Error("素材移出已提交，但项目权威范围已变化；正在重新核对时间线");
        }
        const authoritative = normalizeTimelineProject(authority.document);
        if (!authoritative) throw new Error("服务器时间线响应无效");
        acceptTimelineServerAuthority({ document: authoritative, revision: authority.revision });
        timelineRevisionChannelRef.current?.acceptKnown({
          revision: authority.revision,
          documentHash: timelineDocumentHash(authoritative),
        });
        installTimelineAuthority(authoritative);
        timelinePersistedRevision.current = timelineRevision.current;
        setTimelineAuthorityRequired(false);
        setTimelineDirty(false);
        setTimelinePausedError(null);
        timelineHadLocal.current = true;
        clearLocalTimeline();
        invalidateAndRefreshTaskSnapshots();
      } else {
        dispatchTimelineUi({ type: "assets/remove", ids });
      }
      const assetsRefreshed = await loadAssets(undefined, false, {
        database: operationDatabase,
      });
      if (!operationStillCurrent()) return;
      if (!assetsRefreshed) dispatchTimelineUi({ type: "assets/remove", ids: batch.asset_ids });
      setToast(assetsRefreshed
        ? `已将 ${batch.asset_ids.length} 个素材登记移至回收站；ComfyUI 文件保留`
        : `已将 ${batch.asset_ids.length} 个素材登记移至回收站；列表权威刷新失败，ComfyUI 文件仍保留`);
    } catch (reason) {
      if (!timelineAuthorityRequired) {
        setTimelineDirty(timelinePersistedRevision.current < timelineRevision.current);
        const outcome = await reconcileAmbiguousAssetTrash(
          ids,
          operationDatabase,
          operationProjectId,
          operationSwitchGeneration,
        );
        if (outcome === "committed") {
          clearTimelineHistory();
          setToast(`已确认 ${ids.length} 个素材登记移至回收站；先前响应在返回途中丢失，ComfyUI 文件保留`);
        } else if (outcome === "unknown") {
          // The POST may have committed even when neither response nor the
          // follow-up reads are usable. Old cursors and the visible inventory
          // are no longer safe until a complete same-origin authority refresh.
          clearTimelineHistory();
          assetAuthorityRequired.current = true;
          setRuntimeAuthorityRequired(true);
          setToast("素材移出结果尚无法确认；已暂停相关操作并正在重新核对素材库");
          window.queueMicrotask(() => externalRuntimeAuthorityRefreshRef.current());
          void loadAssetTrash();
        } else {
          setToast(reason instanceof Error ? reason.message : "素材移出失败");
        }
      } else {
        await loadAssets(undefined, false, {
          database: operationDatabase,
        });
        setToast(reason instanceof Error ? reason.message : "素材移出失败");
      }
    } finally {
      assetDeleteLock.current = false;
      assetDeleteIntent.current = false;
      setAssetsDeleting(false);
      if (timelineSyncRequiredRef.current && timelineOperationScopeStillCurrent()) {
        void resyncTimeline();
        if (assetOperationScopeStillCurrent()) void loadAssetTrash();
      }
      else if (runtimeSettingsDesired.current) {
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        runtimeSettingsDrainRef.current();
      }
    }
  };

  const restoreAssetTrashBatch = async (
    batch: AssetTrashBatch,
    mode: AssetTrashRestoreMode,
  ): Promise<void> => {
    const restoresReferences = mode === "with_references";
    if (assetTrashOperationLock.current || assetDeleteLock.current || assetDeleteIntent.current) return;
    if (databaseIdentityStaleRef.current) {
      setToast("数据库权威状态尚未稳定，暂不能恢复素材");
      return;
    }
    if (restoresReferences && runtimeExecutionIntent.current > 0) {
      setToast("生成或预检正在确认当前项目，完成前不能恢复项目引用");
      return;
    }
    if (assetUploadLock.current || runtimeSettingsOperation.current || runtimeSettingsSyncRequiredRef.current) {
      setToast("素材或运行设置操作仍在进行，请稍候再恢复");
      return;
    }
    if (timelineRevisionConflictRef.current || timelineSyncRequiredRef.current) {
      setToast("请先完成时间线权威同步或解决修订冲突");
      return;
    }
    const operationDatabase = activeDatabaseRef.current;
    if (!operationDatabase) {
      setToast("数据库权威状态尚未稳定，暂不能恢复素材");
      return;
    }
    const operationProjectId = activeProjectIdRef.current;
    assetTrashOperationLock.current = batch.batch_id;
    assetDeleteIntent.current = true;
    const operationSwitchGeneration = ++projectSwitchGeneration.current;
    const timelineOperationScopeStillCurrent = () =>
      activeProjectIdRef.current === operationProjectId &&
      activeDatabaseRef.current?.active_database_path === operationDatabase.active_database_path;
    const assetOperationScopeStillCurrent = timelineOperationScopeStillCurrent;
    const operationStillCurrent = () =>
      projectSwitchGeneration.current === operationSwitchGeneration &&
      assetOperationScopeStillCurrent();
    setAssetsDeleting(true);
    setAssetTrashBusyBatchId(batch.batch_id);
    let authorityBoundary = false;
    try {
      if (restoresReferences) {
        await flushTimelineAutosave();
        if (!operationStillCurrent()) {
          throw new Error("项目或素材库权威范围已变化，请在同步完成后重试");
        }
        assetDeleteLock.current = true;
        timelineWriteGeneration.current += 1;
        authorityBoundary = true;
        setTimelineAuthorityRequired(true);
        timelineHadLocal.current = false;
        clearLocalTimeline();
      }
      const restored = await directorApi.restoreAssetTrash(batch.batch_id, mode);
      if (!operationStillCurrent()) {
        throw new Error("素材恢复已提交，但项目权威范围已变化；正在重新核对结果");
      }
      setAssetTrashConflictBatchIds((current) => {
        const next = new Set(current);
        next.delete(batch.batch_id);
        return next;
      });
      if (restoresReferences) {
        const authority = await fetchTimelineForProject(operationProjectId);
        if (!operationStillCurrent()) {
          throw new Error("素材恢复已提交，但项目权威范围已变化；正在重新核对时间线");
        }
        const authoritative = normalizeTimelineProject(authority.document);
        if (!authoritative) throw new Error("服务器时间线响应无效");
        acceptTimelineServerAuthority({ document: authoritative, revision: authority.revision });
        timelineRevisionChannelRef.current?.acceptKnown({
          revision: authority.revision,
          documentHash: timelineDocumentHash(authoritative),
        });
        installTimelineAuthority(authoritative);
        timelinePersistedRevision.current = timelineRevision.current;
        setTimelineAuthorityRequired(false);
        setTimelineDirty(false);
        setTimelinePausedError(null);
        timelineHadLocal.current = true;
        clearLocalTimeline();
        clearTimelineHistory();
        invalidateAndRefreshTaskSnapshots();
        authorityBoundary = false;
      }
      const [assetsRefreshed, trashRefreshed] = await Promise.all([
        loadAssets(undefined, false, {
          database: operationDatabase,
        }),
        loadAssetTrash(),
      ]);
      if (!operationStillCurrent()) return;
      const refreshSuffix = assetsRefreshed && trashRefreshed
        ? ""
        : "；恢复已提交，但列表刷新失败，请重试刷新";
      setToast((restored.restored_references
        ? `已恢复 ${restored.restored_asset_ids.length} 个素材及其原引用`
        : `已恢复 ${restored.restored_asset_ids.length} 个素材登记；项目引用未改动`) + refreshSuffix);
    } catch (reason) {
      if (
        restoresReferences &&
        reason instanceof ApiError &&
        reason.code === "asset_trash_restore_conflict"
      ) {
        // The restore itself is atomic and made no changes, but the conflict
        // can be evidence that this active project changed elsewhere. Keep the
        // authority boundary until a fresh GET has installed that revision.
        setAssetTrashConflictBatchIds((current) => new Set(current).add(batch.batch_id));
        setToast("项目在移出后已变化，不能安全恢复旧引用；正在同步最新项目，之后可仅恢复素材登记");
      } else {
        setToast(reason instanceof Error ? reason.message : "素材恢复失败");
      }
    } finally {
      assetDeleteLock.current = false;
      assetDeleteIntent.current = false;
      assetTrashOperationLock.current = null;
      setAssetsDeleting(false);
      setAssetTrashBusyBatchId(null);
      if (
        (authorityBoundary || timelineSyncRequiredRef.current) &&
        timelineOperationScopeStillCurrent()
      ) {
        void resyncTimeline();
        if (assetOperationScopeStillCurrent()) {
          void loadAssets(undefined, false, {
            database: operationDatabase,
            });
          void loadAssetTrash();
        }
      } else if (runtimeSettingsDesired.current) {
        runtimeSettingsDrainRef.current();
      }
    }
  };

  const purgeAssetTrashBatch = async (batch: AssetTrashBatch): Promise<void> => {
    if (assetTrashOperationLock.current || assetDeleteLock.current || assetDeleteIntent.current) return;
    if (databaseIdentityStaleRef.current) {
      setToast("数据库权威状态尚未稳定，暂不能移除恢复记录");
      return;
    }
    if (!window.confirm("仅永久移除 Director 中的恢复记录？此操作不能撤销，但不会删除 ComfyUI 中的文件。")) return;
    assetTrashOperationLock.current = batch.batch_id;
    assetDeleteIntent.current = true;
    projectSwitchGeneration.current += 1;
    setAssetsDeleting(true);
    setAssetTrashBusyBatchId(batch.batch_id);
    try {
      const result = await directorApi.purgeAssetTrash(batch.batch_id);
      setAssetTrashConflictBatchIds((current) => {
        const next = new Set(current);
        next.delete(batch.batch_id);
        return next;
      });
      await loadAssetTrash();
      setToast(`已永久移除 ${result.purged_asset_ids.length} 个 Director 恢复记录；ComfyUI 文件保留`);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "永久移除恢复记录失败");
    } finally {
      assetDeleteIntent.current = false;
      assetTrashOperationLock.current = null;
      setAssetsDeleting(false);
      setAssetTrashBusyBatchId(null);
      if (runtimeSettingsDesired.current) runtimeSettingsDrainRef.current();
    }
  };

  const uploadWorkspaceFiles = useCallback(async (
    files: File[],
    onProgress?: (progress: DroppedUploadProgress) => void,
  ): Promise<DroppedUploadResult> => {
    if (!files.length) return { assets: [], failures: [], authority_stale: false };
    if (assetUploadLock.current) throw new Error("已有一批本地素材正在上传，请稍候");
    if (assetDeleteLock.current || assetDeleteIntent.current) throw new Error("正在原子解除素材引用，完成前不能上传素材");
    if (runtimeSettingsOperation.current || runtimeSettingsSyncRequiredRef.current) {
      throw new Error("运行设置尚未完成服务器权威回读，暂不能上传素材");
    }
    if (timelineRevisionConflict) throw new Error("服务器时间线存在修订冲突，请先选择处理方式");
    if (timelineSyncRequired) throw new Error("服务器时间线正在自动恢复权威状态，暂不能上传素材");
    if (capabilities.connection !== "online") {
      throw new Error("ComfyUI 尚未连接，暂不能上传素材");
    }

    const { accepted, unsupported } = classifyDroppedFiles(files);
    const database = activeDatabaseRef.current;
    if (!database || databaseIdentityStaleRef.current) {
      throw new Error("数据库权威状态尚未稳定，暂不能上传素材");
    }
    const generation = runtimeSettingsGeneration.current;
    const projectId = activeProjectIdRef.current;
    assetUploadLock.current = true;
    const projectGeneration = ++projectSwitchGeneration.current;
    const authorityCurrent = () =>
      runtimeSettingsGeneration.current === generation &&
      projectSwitchGeneration.current === projectGeneration &&
      activeProjectIdRef.current === projectId &&
      activeDatabaseRef.current?.active_database_path === database.active_database_path &&
      !databaseIdentityStaleRef.current &&
      !runtimeSettingsOperation.current;
    setAssetsUploading(true);
    try {
      const result = await uploadClassifiedDroppedFiles(
        accepted,
        (file, kind, report) => directorApi.uploadAsset(file, kind, report),
        authorityCurrent,
        (progress) => {
          setAssetUploadProgress(progress);
          onProgress?.(progress);
        },
      );
      return {
        ...result,
        failures: [
          ...unsupported.map((file) => ({
            file_name: file.name,
            message: "不支持的素材格式",
          })),
          ...result.failures,
        ],
      };
    } finally {
      assetUploadLock.current = false;
      setAssetsUploading(false);
      setAssetUploadProgress(null);
      if (
        !timelineSyncRequiredRef.current &&
        !timelineRevisionConflictRef.current &&
        timelinePersistedRevision.current < timelineRevision.current
      ) {
        setTimelineRetryNonce((current) => current + 1);
      }
      if (runtimeSettingsDesired.current) {
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        runtimeSettingsDrainRef.current();
      }
    }
  }, [capabilities.connection, timelineRevisionConflict, timelineSyncRequired]);

  const activeTasks = state.tasks.filter((task) => ["queued", "preparing", "running", "cancelling"].includes(task.status));
  const activityRank: Record<string, number> = { running: 0, preparing: 1, queued: 2, cancelling: 3 };
  const activeTask = [...activeTasks].sort((left, right) =>
    (activityRank[left.status] ?? 9) - (activityRank[right.status] ?? 9) ||
    Number(right.mode === "timeline") - Number(left.mode === "timeline") ||
    Date.parse(right.updated_at) - Date.parse(left.updated_at))[0] ?? null;
  const segmentCandidates: Record<string, {
    job_id: string;
    job_updated_at: string;
    result: (typeof state.tasks)[number]["segment_results"][number];
  }> = {};
  for (const task of [...state.tasks]
    .filter((candidate) => candidate.mode === "timeline")
    .sort((left, right) =>
      Date.parse(right.created_at) - Date.parse(left.created_at) ||
      right.id.localeCompare(left.id))) {
    for (const result of task.segment_results) {
      segmentCandidates[result.segment_id] ??= {
        job_id: task.id,
        job_updated_at: task.updated_at,
        result,
      };
    }
  }
  const runtimeReady = capabilities.connection === "online" &&
    runtimeResourcesReady && rayLightRuntimeStatus !== null;
  const rayLightRecoveryRequired = runtimeReady &&
    rayLightRuntimeStatus?.recovery_required === true;
  const runtimeAuthorityPending = runtimeSettingsOperationOwner !== null || runtimeSettingsSyncRequired;
  const timelineHydrated = timelineHydrationStatus === "ready";
  const databaseIdentityStale = timelineHydrationStatus === "stale";
  const workspaceRuntimeReady = runtimeReady && !rayLightRecoveryRequired && !rayLightRecoveryPending && timelineHydrated && !runtimeAuthorityPending && !timelineSyncRequired && !timelineRevisionConflict && !assetsDeleting && !assetsUploading;
  const workspaceCapabilities: CapabilityReport = !runtimeReady || rayLightRecoveryRequired || rayLightRecoveryPending || runtimeAuthorityPending || timelineSyncRequired || timelineRevisionConflict || assetsDeleting || assetsUploading
    ? {
        ...capabilities,
        connection: "checking",
        message: !runtimeReady
          ? "ComfyUI 运行资源等待权威核对"
          : rayLightRecoveryRequired
          ? "旧 RayLight 运行状态等待重启确认"
          : rayLightRecoveryPending
            ? "正在核对 RayLight 重启恢复结果"
          : runtimeAuthorityPending
          ? "运行设置等待服务器权威回读"
          : timelineSyncRequired
            ? "时间线等待服务器权威回读"
            : timelineRevisionConflict
              ? "时间线存在服务器修订冲突"
            : assetsDeleting
              ? "正在原子解除素材引用"
              : "正在上传并绑定本地素材",
      }
    : capabilities;
  const activeAssetIds = new Set(timeline.assets.map((asset) => asset.id));
  const inactiveAssetReferences = (segmentIds?: ReadonlySet<string>) =>
    timeline.project.segments.flatMap((segment, index) =>
      segmentIds && !segmentIds.has(segment.id)
        ? []
        : segmentAssetReferences(segment)
            .filter((asset) => !activeAssetIds.has(asset.id))
            .map(() => `${index + 1} · ${segment.title || segment.id}`),
    );
  const timelinePausedMessage = timelinePausedError?.revision === timelineRevision.current
    ? `服务器拒绝当前时间线：${timelinePausedError.message}；请修改，修改后自动应用`
    : null;
  const selectedEnabledIds = runnableTimelineSegmentIds(timeline);
  const inactiveSelectionAssetReferences = inactiveAssetReferences(new Set(selectedEnabledIds));
  const emptySelectionErrors = selectedEnabledIds.length > 0
    ? []
    : timeline.selected_segment_ids.length > 0
      ? ["所选片段均已停用；请启用至少一个所选片段后再生成"]
      : ["请至少选择一个要生成的片段"];
  const selectionTimelineErrors = [
    ...(!timelineHydrated ? [databaseIdentityStale ? "本页数据库身份已过期，请刷新整个页面" : "正在从服务器恢复时间线"] : []),
    ...(runtimeAuthorityPending ? ["运行设置尚未完成服务器权威回读"] : []),
    ...(!runtimeReady ? ["ComfyUI 与 RayLight 运行资源尚未完成权威核对"] : []),
    ...(rayLightRecoveryRequired ? ["旧 RayLight 运行状态引用当前不可见 GPU；请在系统设置确认 ComfyUI 已重启并恢复"] : []),
    ...(rayLightRecoveryPending ? ["正在核对 RayLight 重启恢复结果"] : []),
    ...(timelineSyncRequired ? ["素材级联已提交，但服务器时间线尚未完成权威回读"] : []),
    ...(timelineRevisionConflict ? ["服务器时间线存在修订冲突，请先选择采用服务器版本或保留本地版本"] : []),
    ...(timelinePausedMessage ? [timelinePausedMessage] : []),
    ...emptySelectionErrors,
    ...(selectedEnabledIds.length ? validateTimelineProject(timeline.project, selectedEnabledIds) : []),
    ...(inactiveSelectionAssetReferences.length
      ? [`当前 ComfyUI 素材库不包含所选片段的引用素材：${[...new Set(inactiveSelectionAssetReferences)].join("、")}`]
      : []),
    ...(selectedEnabledIds.length
      ? runtimeTimelineValidation(timeline.project, capabilities, state.settings, selectedEnabledIds)
      : []),
  ];
  const timelineRunActionsReady = workspaceCapabilities.connection === "online" &&
    selectionTimelineErrors.length === 0 && selectedEnabledIds.length > 0;
  const timelineHistoryBlocked = state.view !== "workspace" ||
    !timelineHydrated || databaseIdentityStale ||
    timelineSyncRequired || Boolean(timelineRevisionConflict) || assetsDeleting ||
    projectSwitchHandoffPending;
  const nextUndoLabel = timelineHistoryUndoLabel(timelineHistory);
  const nextRedoLabel = timelineHistoryRedoLabel(timelineHistory);
  const timelineUndoReady = !timelineHistoryBlocked && canUndoTimelineHistory(timelineHistory);
  const timelineRedoReady = !timelineHistoryBlocked && canRedoTimelineHistory(timelineHistory);
  const assetUsages = Object.fromEntries(timeline.assets.map((asset) => [
    asset.id,
    timelineAssetUsages(timeline.project, asset.id).map((usage) =>
      `${usage.segment_index + 1} · ${usage.segment_title} · ${usage.role}`),
  ]));
  const beginProjectTitleEdit = () => {
    setProjectTitleDraft(timeline.project.title);
    setProjectTitleEditing(true);
  };
  const cancelProjectTitleEdit = () => {
    setProjectTitleDraft(timeline.project.title);
    setProjectTitleEditing(false);
  };
  const commitProjectTitleEdit = () => {
    const title = projectTitleDraft.trim();
    setProjectTitleEditing(false);
    if (!title || title === timeline.project.title) {
      setProjectTitleDraft(timeline.project.title);
      return;
    }
    dispatchTimeline({ type: "project/patch", patch: { title } });
  };

  const switchProject = async (targetId: string): Promise<boolean> => {
    // Even choosing the already-active project is meaningful: it cancels an
    // unresolved switch whose response has not taken authority yet.
    const switchGeneration = ++projectSwitchGeneration.current;
    projectListGeneration.current += 1;
    if (targetId === activeProjectIdRef.current) return true;
    if (projectDeleteIntent.current !== null && targetId !== DEFAULT_PROJECT_ID) {
      setToast("项目删除正在确认服务器结果；完成前不能切换项目");
      return false;
    }
    if (timelineRevisionConflictRef.current) {
      setToast("请先处理当前项目的服务器修订冲突，再切换项目");
      return false;
    }
    if (runtimeExecutionIntent.current > 0) {
      setToast("生成或预检正在确认当前项目；完成前不能切换项目");
      return false;
    }
    if (
      assetDeleteIntent.current || assetDeleteLock.current ||
      assetUploadLock.current || assetTrashOperationLock.current !== null
    ) {
      setToast("素材库操作正在确认当前项目；完成前不能切换项目");
      return false;
    }
    const database = activeDatabaseRef.current;
    if (!database || databaseIdentityStaleRef.current) {
      setToast("数据库权威状态尚未稳定，暂不能切换项目");
      return false;
    }
    const ownsProjectSwitch = () => {
      const activeDatabase = activeDatabaseRef.current;
      return projectSwitchGeneration.current === switchGeneration &&
        !databaseIdentityStaleRef.current &&
        !assetDeleteIntent.current &&
        !assetDeleteLock.current &&
        !assetUploadLock.current &&
        assetTrashOperationLock.current === null &&
        activeDatabase?.active_database_path === database.active_database_path;
    };
    if (timelineHydrationReady.current) {
      try {
        await flushTimelineAutosave();
        if (!ownsProjectSwitch()) return false;
      } catch (reason) {
        if (!ownsProjectSwitch()) return false;
        setToast(`切换前同步当前项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
        return false;
      }
    }
    let targetAuthority: TimelineAuthority;
    let serverProject: TimelineProject | null;
    try {
      // Always establish the target project's CAS base, even when a scoped WAL
      // exists. The WAL is inspected only after the second current-project
      // flush so that its adopted marker cannot be cleared by that ACK.
      targetAuthority = await fetchTimelineForProject(targetId);
      serverProject = normalizeTimelineProject(targetAuthority.document);
      if (!ownsProjectSwitch()) return false;
    } catch (reason) {
      if (!ownsProjectSwitch()) return false;
      setToast(`加载目标项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
      return false;
    }
    if (!ownsProjectSwitch()) return false;
    if (!serverProject) { setToast("目标项目时间线结构无效"); return false; }
    const targetPersistenceAuthority = {
      document: serverProject,
      revision: targetAuthority.revision,
    };
    await timelineJournalChain.current.catch(() => undefined);
    if (!ownsProjectSwitch()) return false;
    const targetPersistenceScope = timelinePersistenceScope(
      database,
      targetId,
      timelineBranchOwnerRef.current,
    );
    const [targetJournal, targetJournalBranchList] = await Promise.all([
      loadTimelineHistoryJournal(targetPersistenceScope, targetPersistenceAuthority),
      listTimelineHistoryJournalBranches(targetPersistenceScope, targetPersistenceAuthority),
    ]);
    if (!ownsProjectSwitch()) return false;
    // The target request may have been pending long enough for another edit
    // to occur in the current project. The async journal read is completed
    // first; then drain again immediately before the synchronous WAL inspect
    // and authority hand-off so no edit can be discarded.
    if (timelineHydrationReady.current) {
      try {
        await flushTimelineAutosave();
        if (
          !ownsProjectSwitch() ||
          runtimeExecutionIntent.current > 0
        ) return false;
      } catch (reason) {
        if (!ownsProjectSwitch()) return false;
        setToast(`切换前同步当前项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
        return false;
      }
    }
    // The second flush can enqueue the old project's ACK history while the
    // target GET/IDB read was pending. Drain it before target hand-off bumps
    // the shared journal generation, otherwise the last undo entry can be
    // skipped even though its document reached the server.
    projectSwitchHandoffIntent.current = switchGeneration;
    setProjectSwitchHandoffPending(true);
    try {
      await timelineJournalChain.current.catch(() => undefined);
      if (!ownsProjectSwitch()) return false;
      const targetWalBranches = listLocalTimelineWalBranches(
      database,
      targetId,
      timelineBranchOwnerRef.current,
    );
    const targetWal = targetWalBranches?.owned?.wal ?? null;
    if ("token" in targetJournal && targetJournal.token) {
      timelineJournalTokens.current.set(
        timelineJournalTokenMapKey(database, targetId, timelineBranchOwnerRef.current),
        targetJournal.token,
      );
    }
    const targetRecovery = collectTimelineRecoveryBranches(
      targetWalBranches,
      targetJournalBranchList.status === "available"
        ? targetJournalBranchList.branches
        : [],
      targetPersistenceAuthority,
    );
    const targetExplicitRecoverySelectionRequired = targetRecovery.pending.length > 1 ||
      targetRecovery.pending.some(
      (branch) => branch.ownership !== "owned" || branch.status === "corrupt",
      );
    const targetWalResolution = targetWal
      ? resolveLocalTimelineWal(targetWal, targetPersistenceAuthority)
      : null;
    let targetProject = serverProject;
    let targetHistory = resetTimelineHistory(timelineHistoryRef.current);
    let targetHasPending = false;
    let targetConflict: TimelineRevisionConflict | null = null;
    const targetJournalCarriesHistory = targetJournal.status === "restored" ||
      targetJournal.status === "acknowledged";
    if (targetWalResolution?.status === "conflict") {
      targetProject = targetWalResolution.local_project;
      targetHasPending = true;
      targetConflict = {
        projectId: targetId,
        localProject: targetProject,
        serverAuthority: targetPersistenceAuthority,
        source: "cas",
        resolving: false,
      };
      if (
        targetJournalCarriesHistory &&
        timelineProjectsEqual(targetJournal.project, targetProject)
      ) targetHistory = targetJournal.history;
    } else if (targetJournal.status === "conflict") {
      targetProject = targetJournal.localProject;
      if (timelineProjectsEqual(targetJournal.project, targetProject)) {
        targetHistory = targetJournal.history;
      }
      targetHasPending = true;
      targetConflict = {
        projectId: targetId,
        localProject: targetProject,
        serverAuthority: targetPersistenceAuthority,
        source: "cas",
        resolving: false,
      };
    } else if (targetWalResolution?.status === "replay") {
      targetProject = targetWalResolution.project;
      targetHasPending = true;
      if (
        targetJournalCarriesHistory &&
        timelineProjectsEqual(targetJournal.project, targetProject)
      ) targetHistory = targetJournal.history;
    } else if (targetWalResolution?.status === "acknowledged") {
      if (
        targetJournalCarriesHistory &&
        timelineProjectsEqual(targetJournal.project, serverProject)
      ) targetHistory = targetJournal.history;
    } else if (targetJournalCarriesHistory) {
      targetProject = targetJournal.project;
      targetHistory = targetJournal.history;
      targetHasPending = !timelineProjectsEqual(targetProject, serverProject);
    }
    if (targetExplicitRecoverySelectionRequired) {
      targetProject = serverProject;
      targetHistory = targetRecovery.newestAcknowledgedHistory ??
        resetTimelineHistory(timelineHistoryRef.current);
      targetHasPending = false;
      targetConflict = {
        projectId: targetId,
        localProject: serverProject,
        serverAuthority: targetPersistenceAuthority,
        source: "recovery-branches",
        resolving: false,
        recoveryBranches: targetRecovery.pending,
        selectedRecoveryBranchId: null,
      };
    } else if (
      !targetHasPending &&
      !targetConflict &&
      targetRecovery.newestAcknowledgedHistory
    ) {
      targetHistory = targetRecovery.newestAcknowledgedHistory;
    }
    if (!ownsProjectSwitch()) return false;
    const targetSegmentIds = targetProject.segments.map((segment) => segment.id);
    const targetSelectionScope = database
      ? `${database.active_database_path}:${targetId}`
      : null;
    const restoredSelection = database
      ? loadTimelineSegmentSelectionPreference(database, targetId, targetSegmentIds)
      : null;
    activeProjectIdRef.current = targetId;
    setActiveProjectIdState(targetId);
    persistActiveProjectId(targetId);
    timelineWriteGeneration.current += 1;
    acceptTimelineServerAuthority(targetPersistenceAuthority);
    activeTimelineWalRef.current = targetWal;
    timelineRevision.current = targetHasPending ? 1 : 0;
    timelinePersistedRevision.current = 0;
    timelineHadLocal.current = targetHasPending || targetHistory.head !== null;
    installTimelineAuthority(targetProject, {
      projectId: targetId,
      selectedSegmentIds: restoredSelection ?? targetSegmentIds,
      clearHistory: false,
    });
    commitTimelineHistory(targetHistory);
    segmentSelectionGeneration.current += 1;
    restoredSegmentSelectionKey.current = targetSelectionScope;
    timelineHydrationReady.current = true;
    setTimelineHydrationStatus("ready");
    setTimelineDirty(targetHasPending);
    setCompileReport(null);
    setTimelinePausedError(null);
    setTimelineRevisionConflict(targetConflict);
    if (targetWal && targetWalResolution?.status === "acknowledged" && !targetConflict) {
      clearLocalTimeline(targetWal);
    } else if (!targetWal && targetHasPending && !targetConflict && database) {
      saveLocalTimeline(targetProject, database, targetId);
    }
    if (database && !targetConflict) {
      persistTimelineJournal(
        targetHistory,
        targetPersistenceAuthority,
        database,
        targetId,
      );
    }
    // Refresh the task drawer's current-project comparison for the new project.
    taskListRequest.current += 1;
    void loadTasks(undefined, true);
    setToast(targetConflict
      ? `已切换到项目“${targetProject.title}”；本地恢复记录与服务器权威不匹配，请选择处理方式`
      : targetHasPending
        ? `已切换到项目“${targetProject.title}”；正在恢复并同步本地修改`
      : `已切换到项目“${targetProject.title}”`);
      return true;
    } finally {
      if (projectSwitchHandoffIntent.current === switchGeneration) {
        projectSwitchHandoffIntent.current = null;
        setProjectSwitchHandoffPending(false);
      }
    }
  };

  const createProject = async (title?: string) => {
    if (projectDeleteIntent.current !== null) {
      setToast("项目删除正在确认服务器结果；完成前不能新建项目");
      return;
    }
    projectListGeneration.current += 1;
    if (timelineHydrationReady.current) {
      try {
        await flushTimelineAutosave();
      } catch (reason) {
        setToast(`新建前同步当前项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
        return;
      }
    }
    try {
      const created = await directorApi.createProject(title);
      setProjects((current) => [...current, created]);
      await switchProject(created.id);
    } catch (reason) {
      setToast(`新建项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
    }
  };

  const deleteProject = async (projectId: string) => {
    if (projectId === DEFAULT_PROJECT_ID) return;
    if (projectDeleteIntent.current !== null) {
      setToast("已有项目正在删除，请等待服务器确认");
      return;
    }
    if (runtimeExecutionIntent.current > 0) {
      setToast("生成或预检正在确认当前项目；完成前不能删除项目");
      return;
    }
    projectDeleteIntent.current = projectId;
    setProjectDeletingId(projectId);
    projectListGeneration.current += 1;
    try {
      if (
        activeProjectIdRef.current === projectId &&
        !await switchProject(DEFAULT_PROJECT_ID)
      ) return;
      const response = await directorApi.deleteProject(projectId);
      projectListGeneration.current += 1;
      setProjects((current) => current.filter((project) => project.id !== projectId));
      if (activeProjectIdRef.current === projectId) {
        restartTimelineHydrationForProject(DEFAULT_PROJECT_ID);
      }
      setToast(`已删除项目；${response.orphaned_jobs} 个历史任务已归档为旧任务`);
    } catch (reason) {
      setToast(`删除项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
    } finally {
      if (projectDeleteIntent.current === projectId) {
        projectDeleteIntent.current = null;
        setProjectDeletingId(null);
      }
    }
  };

  return (
    <div
      className={`app-shell app-shell--timeline ${sidebarOpen ? "is-sidebar-open" : "is-sidebar-collapsed"}`}
      style={{ "--sidebar-resized-width": `${sidebarWidth}px` } as CSSProperties}
    >
      <WorkspaceAssetSidebar
          id="director-sidebar"
          open={sidebarOpen}
          width={sidebarWidth}
          minWidth={sidebarWidthBounds(sidebarViewportWidth).minimum}
          maxWidth={sidebarWidthBounds(sidebarViewportWidth).maximum}
          resizable={sidebarViewportWidth > SIDEBAR_MOBILE_MAX}
          assets={timeline.assets}
          selectedIds={timeline.selected_asset_ids}
          gridSize={timeline.asset_grid_size}
          runtimeEnabled={workspaceRuntimeReady}
          connection={capabilities.connection}
          settingsActive={state.view === "settings"}
          deleting={assetsDeleting}
          assetUsages={assetUsages}
          onUploadFiles={uploadWorkspaceFiles}
          onUploaded={(assets) => {
            // An upload may finish while a settings resync owns the library.
            if (
              runtimeSettingsOperation.current ||
              runtimeSettingsSyncRequiredRef.current
            ) return;
            dispatchTimelineUi({ type: "assets/add", assets });
          }}
          onSelect={(id, additive) => dispatchTimelineUi({ type: "assets/select", id, additive })}
          onSelectRange={(ids, additive) => dispatchTimelineUi({
            type: "assets/set-selection", ids, additive,
          })}
          onMove={(draggedId, targetId) => dispatchTimelineUi({ type: "assets/move", draggedId, targetId })}
          onGridSize={(size) => dispatchTimelineUi({ type: "assets/grid-size", size })}
          onDelete={(ids) => void deleteAssets(ids)}
          settingsNavigationDisabled={false}
          toggleButtonRef={sidebarBrandToggleRef}
          settingsButtonRef={settingsToggleRef}
          onToggle={() => setSidebarOpenWithFocus(!sidebarOpen)}
          onWidthChange={(width) => setSidebarWidth(clampSidebarWidth(width, window.innerWidth))}
          onSettings={() => {
            const opening = state.view !== "settings";
            if (opening) {
              setGlobalSettingsOpen(false);
              setTimelineHistoryPanelOpen(false);
              setAssetTrashPanelOpen(false);
              dispatch({ type: "tasks/panel", open: false });
            }
            dispatch({ type: opening ? "navigate/settings" : "navigate/workspace" });
          }}
        />

      <div className="app-main">
        {projectSwitchHandoffPending && <div className="timeline-hydration-notice" role="status" aria-live="polite">
          <Spinner />
          <span>正在完成项目切换交接；完成前不能编辑当前项目</span>
        </div>}
        {!timelineHydrated && <div className="timeline-hydration-notice" role="status" aria-live="polite">
          {timelineHydrationStatus !== "stale" && <Spinner />}
          <span>{timelineHydrationStatus === "stale"
            ? "Director 后端数据库已变化；本页已停止修改。请刷新整个页面后继续。"
            : timelineHydrationStatus === "retrying"
              ? "暂时无法确认数据库或读取服务器时间线，正在自动重试；恢复前编辑已锁定。"
              : "正在从服务器恢复时间线；恢复前编辑已锁定。"}</span>
        </div>}
        {timelineRevisionConflict && <div className="timeline-hydration-notice timeline-revision-conflict" role="alert" aria-live="assertive">
          {timelineRevisionConflict.resolving && <Spinner />}
          {timelineRevisionConflict.source === "recovery-branches" ? <>
            <span>检测到其他页面或旧会话留下的时间线恢复分支。当前显示服务器版本；选择前不会重放或写入任何分支。</span>
            <fieldset disabled={timelineRevisionConflict.resolving}>
              <legend>可恢复的本地分支（{timelineRevisionConflict.recoveryBranches?.length ?? 0}）</legend>
              <ul aria-label="时间线恢复分支列表">
                {timelineRevisionConflict.recoveryBranches?.map((branch) => {
                  const shortOwner = branch.ownerId
                    ? branch.ownerId.length > 18
                      ? `${branch.ownerId.slice(0, 8)}…${branch.ownerId.slice(-6)}`
                      : branch.ownerId
                    : "未知 owner";
                  const statusLabel = branch.status === "replay"
                    ? "基线匹配，可安全恢复"
                    : branch.status === "acknowledged"
                      ? "服务器已包含此版本"
                      : branch.status === "conflict"
                        ? "基线已变化，恢复后需再次确认"
                        : "记录损坏，仅可丢弃";
                  return <li key={branch.id}>
                    <label>
                      <input
                        type="radio"
                        name="timeline-recovery-branch"
                        value={branch.id}
                        checked={timelineRevisionConflict.selectedRecoveryBranchId === branch.id}
                        disabled={branch.status === "corrupt" || timelineRevisionConflict.resolving}
                        onChange={() => selectTimelineRecoveryBranch(branch.id)}
                      />
                      <span>{branch.project?.title ?? "损坏的恢复记录"}；{branch.ownership === "owned" ? "当前会话" : branch.ownership === "legacy" ? "旧版证据" : "其他会话"}；{shortOwner}；{statusLabel}{branch.updatedAtMs ? `；${new Date(branch.updatedAtMs).toLocaleString()}` : ""}</span>
                    </label>
                    <button
                      type="button"
                      className="button button--ghost"
                      disabled={timelineRevisionConflict.resolving}
                      aria-label={`丢弃恢复记录：${branch.project?.title ?? shortOwner}`}
                      onClick={() => void discardTimelineRecoveryBranch(branch.id)}
                    >丢弃恢复记录</button>
                  </li>;
                })}
              </ul>
            </fieldset>
            <div role="group" aria-label="时间线恢复处理">
              <button
                type="button"
                className="button button--ghost"
                disabled={timelineRevisionConflict.resolving}
                onClick={() => void continueServerWithRecoveryEvidence()}
              >继续服务器并保留记录</button>
              <button
                type="button"
                className="button button--primary"
                disabled={timelineRevisionConflict.resolving ||
                  !timelineRevisionConflict.selectedRecoveryBranchId}
                onClick={() => void restoreSelectedTimelineBranch()}
              >恢复所选分支</button>
            </div>
          </> : <>
            <span>{timelineRevisionConflict.source === "legacy-wal"
              ? "检测到未携带服务器修订号的本地恢复草稿。为避免覆盖其他页面的修改，已停止自动同步。请选择采用服务器版本，或确认保留本地版本。"
              : "服务器时间线已被其他页面修改。本地历史和恢复草稿仍被保留，自动同步已停止。请选择冲突处理方式。"}</span>
            <div role="group" aria-label="时间线冲突处理">
              <button
                type="button"
                className="button button--ghost"
                disabled={timelineRevisionConflict.resolving}
                onClick={() => void adoptServerTimelineAfterConflict()}
              >采用服务器版本</button>
              <button
                type="button"
                className="button button--primary"
                disabled={timelineRevisionConflict.resolving}
                onClick={() => void keepLocalTimelineAfterConflict()}
              >保留本地版本</button>
            </div>
          </>}
        </div>}
        <div className="workspace-surface" {...(state.view === "settings" || !timelineHydrated || timelineRevisionConflict ? { inert: true } : {})}>
        <header className="topbar topbar--timeline">
          <div className="topbar__identity">
            <div className="topbar__mode topbar__mode--timeline">
              {projectTitleEditing ? (
                <input
                  ref={projectTitleInputRef}
                  className="topbar__project-title-input"
                  aria-label="编辑项目名称"
                  maxLength={256}
                  value={projectTitleDraft}
                  onChange={(event) => setProjectTitleDraft(event.target.value)}
                  onBlur={commitProjectTitleEdit}
                  onKeyDown={(event) => {
                    if (event.key === "Escape") {
                      event.preventDefault();
                      cancelProjectTitleEdit();
                    } else if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      commitProjectTitleEdit();
                    }
                  }}
                />
              ) : (
                <button
                  type="button"
                  className="topbar__project-title"
                  aria-label={`重命名项目，当前名称：${timeline.project.title}`}
                  title="点击重命名项目"
                  onClick={beginProjectTitleEdit}
                >
                  <span className="topbar__project-title-icon" aria-hidden="true">
                    <svg viewBox="0 0 16 16" focusable="false">
                      <path d="M3 11.8V14h2.2L12.8 6.4 9.6 3.2 2 10.8Zm11-8.6-1.2-1.2a1 1 0 0 0-1.4 0l-1 1L13.6 6l1-1a1 1 0 0 0 0-1.4Z" />
                    </svg>
                  </span>
                  <span className="topbar__project-title-text">{timeline.project.title}</span>
                </button>
              )}
              <select
                className="topbar__project-switcher"
                aria-label="切换项目"
                value={activeProjectId}
                disabled={!timelineHydrated || databaseIdentityStale ||
                  Boolean(timelineRevisionConflict) || submitting || compiling ||
                  assetsDeleting || assetsUploading || projectSwitchHandoffPending ||
                  projectDeletingId !== null}
                onChange={(event) => {
                  const value = event.target.value;
                  if (value === "__new__") void createProject();
                  else void switchProject(value);
                }}
              >
                {projects.map((project) => (
                  <option key={project.id} value={project.id}>{project.title}</option>
                ))}
                {!projects.some((project) => project.id === activeProjectId) && (
                  <option value={activeProjectId}>{timeline.project.title}</option>
                )}
                <option value="__new__">＋ 新建项目</option>
              </select>
              {activeProjectId !== DEFAULT_PROJECT_ID && (
                <button
                  type="button"
                  className="topbar__project-delete"
                  aria-label={`删除项目 ${timeline.project.title}`}
                  disabled={submitting || compiling || projectDeletingId !== null}
                  onClick={() => {
                    if (window.confirm(`确认删除项目“${timeline.project.title}”？已生成的任务会保留为旧任务。`)) {
                      void deleteProject(activeProjectId);
                    }
                  }}
                >删除项目</button>
              )}
            </div>
          </div>

          <div className="topbar__right">
            <div className="topbar__history-actions" role="group" aria-label="项目编辑历史">
              <button
                type="button"
                aria-label="撤销"
                aria-keyshortcuts="Control+Z Meta+Z"
                title={nextUndoLabel ? `撤销：${nextUndoLabel} · Ctrl/Cmd+Z` : "没有可撤销的项目修改"}
                disabled={!timelineUndoReady}
                onPointerDown={(event) => event.preventDefault()}
                onClick={undoTimeline}
              >
                <span aria-hidden="true">↶</span>
              </button>
              <button
                type="button"
                aria-label="重做"
                aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z Control+Y"
                title={nextRedoLabel ? `重做：${nextRedoLabel} · Ctrl/Cmd+Shift+Z` : "没有可重做的项目修改"}
                disabled={!timelineRedoReady}
                onPointerDown={(event) => event.preventDefault()}
                onClick={redoTimeline}
              >
                <span aria-hidden="true">↷</span>
              </button>
            </div>
            <button
              ref={timelineHistoryToggleRef}
              type="button"
              className="topbar__history-toggle"
              aria-label="编辑历史"
              aria-expanded={timelineHistoryPanelOpen}
              aria-controls={TIMELINE_HISTORY_PANEL_ID}
              disabled={!timelineHydrated || state.view !== "workspace"}
              onClick={() => {
                const opening = !timelineHistoryPanelOpen;
                if (opening) {
                  setGlobalSettingsOpen(false);
                  setAssetTrashPanelOpen(false);
                  dispatch({ type: "tasks/panel", open: false });
                }
                setTimelineHistoryPanelOpen(opening);
              }}
            >编辑历史</button>
            <button ref={globalSettingsToggleRef} type="button" className="topbar__global-toggle" aria-label="全局设置" aria-expanded={globalSettingsOpen} aria-controls={GLOBAL_SETTINGS_ID} onClick={() => {
              const opening = !globalSettingsOpen;
              if (opening) {
                setTimelineHistoryPanelOpen(false);
                setAssetTrashPanelOpen(false);
                dispatch({ type: "tasks/panel", open: false });
              }
              setGlobalSettingsOpen(opening);
            }}><span>全局设置</span><i aria-hidden="true" /></button>
            <button
              ref={assetTrashToggleRef}
              type="button"
              className="topbar__asset-trash-toggle"
              aria-label="素材回收站"
              aria-expanded={assetTrashPanelOpen}
              aria-controls={ASSET_TRASH_PANEL_ID}
              disabled={!timelineHydrated || state.view !== "workspace"}
              onClick={() => {
                const opening = !assetTrashPanelOpen;
                if (opening) {
                  setGlobalSettingsOpen(false);
                  setTimelineHistoryPanelOpen(false);
                  dispatch({ type: "tasks/panel", open: false });
                  void loadAssetTrash();
                }
                setAssetTrashPanelOpen(opening);
              }}
            >素材回收站</button>
            <div className="topbar__run-actions" role="group" aria-label="时间线生成操作">
              <button
                type="button"
                className="button button--ghost"
                aria-label={compiling ? "预检中，正在检查执行计划" : "预检执行计划"}
                aria-live="polite"
                aria-describedby={selectionTimelineErrors.length ? TIMELINE_RUN_VALIDATION_ID : undefined}
                title={selectionTimelineErrors[0]}
                disabled={!timelineRunActionsReady || compiling}
                onClick={() => void inspectTimelineExecution(selectedEnabledIds)}
              >
                <span className="topbar__action-label--long">{compiling ? "预检中…" : "预检执行计划"}</span>
                <span className="topbar__action-label--short" aria-hidden="true">{compiling ? "预检中" : "预检"}</span>
              </button>
              <button
                type="button"
                className="button button--primary"
                aria-label={submitting
                  ? `提交中，${selectedEnabledIds.length} 个生成任务`
                  : `生成任务 ${selectedEnabledIds.length}`}
                aria-live="polite"
                aria-describedby={selectionTimelineErrors.length ? TIMELINE_RUN_VALIDATION_ID : undefined}
                title={selectionTimelineErrors[0]}
                disabled={!timelineRunActionsReady || submitting}
                onClick={() => void submitTimeline(selectedEnabledIds)}
              >
                <span className="topbar__action-label--long">{submitting ? "提交中…" : `生成任务 ${selectedEnabledIds.length}`}</span>
                <span className="topbar__action-label--short" aria-hidden="true">{submitting ? "提交中" : `生成 ${selectedEnabledIds.length}`}</span>
              </button>
            </div>
            <button type="button" className="theme-toggle" aria-label={theme === "dark" ? "切换到浅色主题" : "切换到深色主题"} onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}><span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span></button>
            <button type="button" id="task-panel-toggle" className="queue-button" aria-label={`任务，${activeTasks.length} 个进行中`} aria-controls="task-drawer" aria-expanded={state.taskPanelOpen} onClick={() => {
              const opening = !state.taskPanelOpen;
              if (opening) {
                setGlobalSettingsOpen(false);
                setTimelineHistoryPanelOpen(false);
                setAssetTrashPanelOpen(false);
              }
              dispatch({ type: "tasks/panel", open: opening });
            }}><span>任务</span><strong>{activeTasks.length}</strong></button>
          </div>
        </header>

          <>
            <TimelineGlobalSettings
              id={GLOBAL_SETTINGS_ID}
              open={globalSettingsOpen}
              project={timeline.project}
              settings={runtimeSettingsDraft}
              models={models}
              runtimeReady={runtimeSettingsDraftValid && runtimeResourcesReady && capabilities.connection === "online" && !rayLightRecoveryRequired}
              modelSaving={runtimeSettingsOperationOwner !== null}
              onClose={() => { setGlobalSettingsOpen(false); window.requestAnimationFrame(() => globalSettingsToggleRef.current?.focus()); }}
              onProjectPatch={(patch) => dispatchTimeline({ type: "project/patch", patch })}
              onSamplingChange={(family, patch) => dispatchTimeline({
                type: "project/update-sampling",
                family,
                patch,
              })}
              onRuntimeModelChange={(role, patch) => void updateRuntimeModel(role, patch)}
            />
            <TimelineHistoryPanel
              id={TIMELINE_HISTORY_PANEL_ID}
              open={timelineHistoryPanelOpen}
              history={timelineHistory}
              toggleRef={timelineHistoryToggleRef}
              onJump={jumpTimelineHistoryCursor}
              onClose={(restoreFocus) => {
                setTimelineHistoryPanelOpen(false);
                if (restoreFocus) {
                  window.requestAnimationFrame(() => timelineHistoryToggleRef.current?.focus());
                }
              }}
            />
            <AssetTrashPanel
              id={ASSET_TRASH_PANEL_ID}
              open={assetTrashPanelOpen}
              batches={assetTrashBatches}
              loading={assetTrashLoading}
              busyBatchId={assetTrashBusyBatchId}
              conflictBatchIds={assetTrashConflictBatchIds}
              toggleRef={assetTrashToggleRef}
              onRefresh={() => void loadAssetTrash()}
              onRestore={(batch, mode) => void restoreAssetTrashBatch(batch, mode)}
              onPurge={(batch) => void purgeAssetTrashBatch(batch)}
              onClose={(restoreFocus) => {
                setAssetTrashPanelOpen(false);
                if (restoreFocus) {
                  window.requestAnimationFrame(() => assetTrashToggleRef.current?.focus());
                }
              }}
            />
            {!workspaceRuntimeReady && <div className="timeline-runtime-notice">{!runtimeSettingsDraftValid ? "系统设置有无效输入，请打开并修正；有效后自动应用。" : runtimeSettingsPausedError ? `服务器拒绝当前系统设置：${runtimeSettingsPausedError}。请打开并修改；有效修改后自动应用。` : rayLightRecoveryRequired ? "旧 RayLight 运行状态引用了当前不可见 GPU；请打开系统设置，确认 ComfyUI 已重启后执行恢复。" : runtimeSettingsSyncRequired ? "运行设置或素材库正在后台自动核对；恢复权威状态前，生成与素材操作保持锁定。" : runtimeSettingsOperationOwner !== null ? "运行设置正在同步并从服务器权威回读；完成前不能生成或操作素材。" : timelineRevisionConflict ? "服务器时间线存在修订冲突；本地草稿已保留，请在页面顶部选择处理方式。" : timelineSyncRequired ? "素材操作结果正在自动核对；恢复权威时间线前，编辑与生成保持锁定。" : assetsDeleting ? "正在原子解除素材引用；时间线编辑与生成暂时锁定。" : assetsUploading ? assetUploadProgress ? `${describeUploadProgress(assetUploadProgress)}；完成前暂时锁定同步、预检和生成。` : "正在上传并绑定本地素材；完成前暂时锁定同步、预检和生成。" : capabilities.connection === "offline" ? "ComfyUI 当前离线；编辑内容会在 Director 连接恢复后自动同步，暂时不能生成。" : "正在检查 ComfyUI 能力…"}</div>}
            <LongFormTimelineWorkspace
              state={timeline}
              capabilities={workspaceCapabilities}
              activeTask={activeTask}
              segmentCandidates={segmentCandidates}
              compileReport={compileReport}
              selectionValidationErrors={selectionTimelineErrors}
              onDispatch={dispatchTimeline}
              onCloseCompile={() => setCompileReport(null)}
              onCancelTask={(id) => void cancel(id)}
              onUploadFiles={uploadWorkspaceFiles}
            />
          </>
        </div>
        {state.view === "settings" && (
          <SettingsPage
            overlay
            settings={runtimeSettingsDraft}
            confirmedSettings={state.settings}
            resourcesReady={runtimeResourcesReady}
            capabilities={capabilities}
            gpus={gpus}
            models={models}
            rayLightRuntimeStatus={rayLightRuntimeStatus}
            rayLightRecoveryPending={rayLightRecoveryPending}
            rayLightRecoveryDisabled={runtimeAuthorityPending || databaseIdentityStale || activeTasks.length > 0}
            rayLightRecoveryBlockedReason={runtimeAuthorityPending
              ? "运行设置仍在同步"
              : databaseIdentityStale
                ? "本页数据库身份已过期"
                : activeTasks.length > 0
                  ? "仍有 Director 任务未结束"
                  : null}
            loadingModels={loadingModels}
            syncError={runtimeSettingsPausedError}
            runtimeEditingDisabled={!timelineHydrated || Boolean(timelineRevisionConflict) || databaseIdentityStale || rayLightRecoveryPending}
            theme={theme}
            onThemeChange={setTheme}
            onDraftChange={updateRuntimeSettingsDraft}
            onSaved={(next) => queueRuntimeSettings("settings-page", next)}
            onConnectionTestSucceeded={refreshAuthoritativeResourcesAfterConnectionTest}
            onConfirmRayLightRuntimeRecovery={confirmRayLightRuntimeRecovery}
            onRequestClose={(restoreFocus = true) => {
              dispatch({ type: "navigate/workspace" });
              if (restoreFocus) window.requestAnimationFrame(() => settingsToggleRef.current?.focus());
            }}
          />
        )}
      </div>

      <TaskDrawer id="task-drawer" open={state.taskPanelOpen} tasks={state.tasks} loading={tasksLoading} supportsCancel={capabilities.supports_cancel} deletingTaskIds={deletingTaskIds} clearing={clearingTasks} onClose={() => dispatch({ type: "tasks/panel", open: false })} onRefresh={() => void loadTasks(undefined, true)} onCancel={(id) => cancel(id)} onConfirmComfyRestart={(id) => confirmComfyRestartRecovery(id)} onBulkCancel={cancelMany} onLoadProject={loadTaskProject} onLoadGenerationDetails={(id) => directorApi.getTaskGenerationDetails(id)} onExportDiagnostic={exportTaskDiagnostic} onImportOutput={importTaskOutput} onDelete={(id) => deleteTask(id)} onClearCompleted={() => clearTerminalTasks()} />
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        <span key={timelineHistoryAnnouncement.sequence}>{timelineHistoryAnnouncement.message}</span>
      </div>
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}
