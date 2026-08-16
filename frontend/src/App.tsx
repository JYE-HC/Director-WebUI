import { useCallback, useEffect, useReducer, useRef, useState, type CSSProperties } from "react";
import { ApiError, DATABASE_IDENTITY_STALE_EVENT, directorApi, taskEventsUrl } from "./api/client";
import {
  EMPTY_CAPABILITIES,
  EMPTY_MODELS,
  isConfiguredComfyUrl,
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
  type TimelineCompileReport,
} from "./api/types";
import {
  LongFormTimelineWorkspace,
  TIMELINE_RUN_VALIDATION_ID,
} from "./components/LongFormTimelineWorkspace";
import { SettingsPage, validateRuntimeSettingsForm } from "./components/SettingsPage";
import { TaskDrawer } from "./components/TaskDrawer";
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
import { persistUiTheme, readUiTheme } from "./domain/theme";
import {
  loadTimelineSegmentSelectionPreference,
  saveTimelineSegmentSelectionPreference,
} from "./domain/workspacePreferences";
import {
  autoFitSourceAudioTiming,
  createTimelineEditorState,
  clearLocalTimeline,
  DEFAULT_PROJECT_ID,
  loadAssetLayoutPreference,
  loadLocalTimeline,
  normalizeTimelineProject,
  orderAssetsByPreference,
  runnableTimelineSegmentIds,
  saveAssetLayoutPreference,
  saveLocalTimeline,
  segmentAssetReferences,
  timelineSamplingFamily,
  timelineAssetUsages,
  timelineEditorReducer,
  validateTimelineProject,
  type TimelineAction,
  type TimelineProject,
} from "./domain/timelineProject";
import {
  directorReducer,
  loadDirectorState,
  saveDirectorState,
} from "./state/directorState";

const SIDEBAR_OPEN_KEY = "director-web:sidebar-open";
const SIDEBAR_WIDTH_KEY = "director-web:sidebar-expanded-width";
const SIDEBAR_MOBILE_MAX = 760;
const GLOBAL_SETTINGS_ID = "timeline-global-settings";
export const UNBOUND_RUNTIME_SETTINGS_PENDING_KEY = "director-web:runtime-settings-pending";
export const QUARANTINED_UNBOUND_RUNTIME_SETTINGS_PENDING_KEY = "director-web:runtime-settings-pending-quarantine";
export const RUNTIME_SETTINGS_PENDING_KEY = "director-web:v2:runtime-settings-pending";
export const QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY = "director-web:v2:runtime-settings-pending-quarantine";
const RUNTIME_SETTINGS_PENDING_FORMAT = "director-pending-runtime-settings";
const LEGACY_RUNTIME_SETTINGS_PENDING_VERSION = 1;
const RUNTIME_SETTINGS_PENDING_VERSION = 2;
const RUNTIME_SETTINGS_AUTOSAVE_MS = 300;
const RUNTIME_SETTINGS_RETRY_MS = 1500;
const STORAGE_AUTHORITY_RETRY_MS = 1000;
const RAYLIGHT_RECOVERY_RETRY_MS = 300;
const RAYLIGHT_RECOVERY_MAX_RETRY_MS = 1200;

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
type StorageOperationStatus = "idle" | "submitting" | "reconciling" | "recovering";

let runtimeSettingsWalOwnerCache: string | null = null;
let adoptedRuntimeSettingsWalRaw: string | null = null;
let latestRuntimeSettingsWalRaw: string | null = null;

class RuntimeEndpointTimelineBoundaryError extends Error {
  constructor(reason: unknown) {
    super(reason instanceof Error ? reason.message : "切换地址前无法确认最新时间线");
    this.name = "RuntimeEndpointTimelineBoundaryError";
  }
}

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
  return typeof value === "string" && value.length > 1 && value.length <= 4096 &&
    value.startsWith("/") && !/[\u0000-\u001f\u007f]/.test(value);
}

interface ActiveDatabaseIdentity {
  active_database_path: string;
  active_database_identity: string;
}

function validActiveDatabaseIdentity(value: ActiveDatabaseIdentity): boolean {
  return validActiveDatabasePath(value.active_database_path) &&
    /^[0-9a-f]{64}$/.test(value.active_database_identity);
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
      keys === "active_database_identity|active_database_path|format|pending|settings|version|written_at_ms";
    const owned = envelope.version === RUNTIME_SETTINGS_PENDING_VERSION &&
      keys === "active_database_identity|active_database_path|format|owner_id|pending|settings|version|written_at_ms" &&
      validRuntimeSettingsWalOwner(envelope.owner_id);
    if (
      (!legacy && !owned) ||
      envelope.format !== RUNTIME_SETTINGS_PENDING_FORMAT ||
      envelope.pending !== true ||
      !validActiveDatabasePath(envelope.active_database_path) ||
      typeof envelope.active_database_identity !== "string" ||
      !/^[0-9a-f]{64}$/.test(envelope.active_database_identity) ||
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
    parsed.envelope.active_database_path === database.active_database_path &&
    parsed.envelope.active_database_identity === database.active_database_identity;
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
      parsed.envelope.active_database_path !== database.active_database_path ||
      parsed.envelope.active_database_identity !== database.active_database_identity
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
      active_database_identity: database.active_database_identity,
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

const ACTIVE_PROJECT_ID_STORAGE_KEY = "director-web:v1:active-project-id";

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
function fetchTimelineForProject(projectId: string, signal?: AbortSignal) {
  return projectId === DEFAULT_PROJECT_ID
    ? directorApi.getTimeline(signal)
    : directorApi.getProjectTimeline(projectId, signal);
}

function saveTimelineForProject(projectId: string, project: TimelineProject) {
  return projectId === DEFAULT_PROJECT_ID
    ? directorApi.updateTimeline(project)
    : directorApi.updateProjectTimeline(projectId, project);
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
  const [capabilities, setCapabilities] = useState<CapabilityReport>({
    ...EMPTY_CAPABILITIES,
    connection: "checking",
  });
  const [gpus, setGpus] = useState<GPUResource[]>([]);
  const [models, setModels] = useState<ModelInventory>(EMPTY_MODELS);
  const [rayLightRuntimeStatus, setRayLightRuntimeStatus] = useState<RayLightRuntimeStatus | null>(null);
  const [rayLightRecoveryPending, setRayLightRecoveryPending] = useState(false);
  const [loadingModels, setLoadingModels] = useState(false);
  const [runtimeResourcesOrigin, setRuntimeResourcesOrigin] = useState<string | null>(null);
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
  const [storageRestartRequired, setStorageRestartRequired] = useState(false);
  const [storageOperationStatus, setStorageOperationStatus] = useState<StorageOperationStatus>("idle");
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
  const [toast, setToast] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(() => initialUiToggle(SIDEBAR_OPEN_KEY, true));
  const [sidebarViewportWidth, setSidebarViewportWidth] = useState(() => window.innerWidth);
  const [sidebarWidth, setSidebarWidth] = useState(initialSidebarWidth);
  const [globalSettingsOpen, setGlobalSettingsOpen] = useState(false);
  const [projectTitleEditing, setProjectTitleEditing] = useState(false);
  const [projectTitleDraft, setProjectTitleDraft] = useState("");
  const [theme, setTheme] = useState(readUiTheme);
  const [deletingTaskIds, setDeletingTaskIds] = useState<ReadonlySet<string>>(() => new Set());
  const [clearingTasks, setClearingTasks] = useState(false);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [activeProjectId, setActiveProjectIdState] = useState<string>(loadActiveProjectId);
  const activeProjectIdRef = useRef(activeProjectId);
  const runtimeRequest = useRef(0);
  const runtimeResourceRequest = useRef(0);
  const runtimeResourceRetryTimer = useRef<number | null>(null);
  const runtimeResourceRefreshRef = useRef<(origin: string, preserveExisting: boolean) => Promise<boolean>>(
    async () => false,
  );
  const externalRuntimeAuthorityRefreshRef = useRef<() => void>(() => undefined);
  const externalRuntimeAuthorityOperationRef = useRef<Promise<void> | null>(null);
  const externalRuntimeAuthorityControllerRef = useRef<AbortController | null>(null);
  const externalRuntimeAuthorityRetryTimer = useRef<number | null>(null);
  const runtimeResourcesOriginRef = useRef<string | null>(null);
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
  const runtimeEndpointSwitchRequired = useRef(false);
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
  const projectTitleInputRef = useRef<HTMLInputElement>(null);
  const sidebarBrandToggleRef = useRef<HTMLButtonElement>(null);
  const settingsToggleRef = useRef<HTMLButtonElement>(null);
  const timelineRevision = useRef(0);
  const timelineHydrationReady = useRef(false);
  const activeDatabaseRef = useRef<ActiveDatabaseIdentity | null>(null);
  const storageRestartRequiredRef = useRef(false);
  const storageOperationStatusRef = useRef<StorageOperationStatus>("idle");
  const storageAuthorityControllerRef = useRef<AbortController | null>(null);
  const storageRecoveryInProgress = useRef(false);
  const databaseIdentityStaleRef = useRef(false);
  const segmentSelectionGeneration = useRef(0);
  const restoredSegmentSelectionKey = useRef<string | null>(null);
  const projectSwitchGeneration = useRef(0);
  const timelinePersistedRevision = useRef(0);
  const timelineWriteGeneration = useRef(0);
  const timelineSaveRequest = useRef<Promise<TimelineProject | null> | null>(null);
  const timelineSaveRequestRevision = useRef<number | null>(null);
  const timelineAutosaveTimer = useRef<number | null>(null);
  const timelineRetryTimer = useRef<number | null>(null);
  const timelineAuthorityRetryTimer = useRef<number | null>(null);
  const timelineSyncRequiredRef = useRef(false);
  const timelineRenderedRevision = useRef(timelineRevision.current);
  const flushTimelineAutosaveRef = useRef<() => Promise<TimelineProject>>(
    async () => { throw new Error("时间线自动保存尚未初始化"); },
  );
  const assetDeleteLock = useRef(false);
  const assetDeleteIntent = useRef(false);
  const assetUploadLock = useRef(false);
  const timelineRef = useRef(timeline);
  const timelineHadLocal = useRef(false);
  timelineRef.current = timeline;
  timelineRenderedRevision.current = timelineRevision.current;

  const setSidebarOpenWithFocus = useCallback((open: boolean) => {
    setSidebarOpen(open);
    window.requestAnimationFrame(() => sidebarBrandToggleRef.current?.focus());
  }, []);

  const setRuntimeAuthorityRequired = useCallback((required: boolean) => {
    runtimeSettingsSyncRequiredRef.current = required;
    setRuntimeSettingsSyncRequired(required);
  }, []);

  const setTimelineAuthorityRequired = useCallback((required: boolean) => {
    timelineSyncRequiredRef.current = required;
    setTimelineSyncRequired(required);
  }, []);

  const setStorageOperationLock = useCallback((status: StorageOperationStatus) => {
    storageOperationStatusRef.current = status;
    setStorageOperationStatus(status);
  }, []);

  const acceptStorageConfiguration = useCallback((configuration: StorageConfiguration): boolean => {
    const activeDatabase = activeDatabaseRef.current;
    if (
      !activeDatabase ||
      configuration.active_database_path !== activeDatabase.active_database_path ||
      configuration.active_database_identity !== activeDatabase.active_database_identity
    ) {
      databaseIdentityStaleRef.current = true;
      timelineHydrationReady.current = false;
      setTimelineHydrationStatus("stale");
      setToast("Director 后端数据库已变化；请刷新整个页面后继续");
      return false;
    }
    storageRestartRequiredRef.current = configuration.restart_required;
    setStorageRestartRequired(configuration.restart_required);
    setStorageOperationLock("idle");
    return true;
  }, [setStorageOperationLock]);

  const beginStorageOperation = useCallback(() => {
    storageAuthorityControllerRef.current?.abort();
    storageAuthorityControllerRef.current = null;
    setStorageOperationLock("submitting");
  }, [setStorageOperationLock]);

  const abortStorageOperation = useCallback(() => {
    storageAuthorityControllerRef.current?.abort();
    storageAuthorityControllerRef.current = null;
    setStorageOperationLock("idle");
  }, [setStorageOperationLock]);

  const reconcileUncertainStorageOperation = useCallback(async (): Promise<StorageConfiguration> => {
    storageAuthorityControllerRef.current?.abort();
    const controller = new AbortController();
    storageAuthorityControllerRef.current = controller;
    setStorageOperationLock("reconciling");
    try {
      for (;;) {
        try {
          const configuration = await directorApi.getStorage(controller.signal);
          if (controller.signal.aborted) throw new DOMException("Aborted", "AbortError");
          if (!acceptStorageConfiguration(configuration)) {
            throw new DatabaseIdentityChangedDuringHydrationError();
          }
          return configuration;
        } catch (reason) {
          if (
            controller.signal.aborted ||
            databaseIdentityStaleRef.current ||
            reason instanceof DatabaseIdentityChangedDuringHydrationError
          ) throw reason;
          await new Promise<void>((resolve, reject) => {
            const timer = window.setTimeout(resolve, STORAGE_AUTHORITY_RETRY_MS);
            controller.signal.addEventListener("abort", () => {
              window.clearTimeout(timer);
              reject(new DOMException("Aborted", "AbortError"));
            }, { once: true });
          });
        }
      }
    } finally {
      if (storageAuthorityControllerRef.current === controller) {
        storageAuthorityControllerRef.current = null;
      }
    }
  }, [acceptStorageConfiguration, setStorageOperationLock]);

  useEffect(() => () => storageAuthorityControllerRef.current?.abort(), []);

  const dispatchTimeline = useCallback((action: TimelineAction) => {
    const current = timelineRef.current;
    const reduced = timelineEditorReducer(current, action);
    const sourceAudioFit = autoFitSourceAudioTiming(reduced.project);
    const next = sourceAudioFit.project === reduced.project
      ? reduced
      : { ...reduced, project: sourceAudioFit.project };
    const projectChanged = next.project !== current.project;
    const selectionChanged = next.selected_segment_ids.length !== current.selected_segment_ids.length ||
      next.selected_segment_ids.some((id, index) => id !== current.selected_segment_ids[index]);
    const segmentTopologyChanged = next.project.segments.length !== current.project.segments.length ||
      next.project.segments.some((segment, index) => segment.id !== current.project.segments[index]?.id);
    const currentRunnable = runnableTimelineSegmentIds(current);
    const nextRunnable = runnableTimelineSegmentIds(next);
    const executableSelectionChanged = nextRunnable.length !== currentRunnable.length ||
      nextRunnable.some((id, index) => id !== currentRunnable[index]);
    if (databaseIdentityStaleRef.current && projectChanged) {
      setToast("本页数据库身份已过期；请刷新整个页面后继续");
      return;
    }
    if (storageOperationStatusRef.current !== "idle" && projectChanged) {
      setToast("数据库操作结果尚未确认；确认完成前不能继续修改");
      return;
    }
    if (storageRestartRequiredRef.current && projectChanged) {
      setToast("数据库切换正在等待重启；刷新页面前不能继续修改");
      return;
    }
    if (!timelineHydrationReady.current && projectChanged) {
      setToast("正在从服务器恢复时间线；恢复完成前暂不能编辑");
      return;
    }
    if ((assetDeleteLock.current || assetDeleteIntent.current || timelineSyncRequiredRef.current || runtimeEndpointSwitchRequired.current) && projectChanged) {
      setToast(runtimeEndpointSwitchRequired.current
        ? "正在切换 ComfyUI 地址；最新时间线确认并完成新地址核对前暂不能编辑"
        : assetDeleteIntent.current
          ? "正在建立素材移出的原子边界；完成后再编辑时间线"
        : timelineSyncRequiredRef.current
          ? "服务器时间线尚未完成权威回读；恢复同步前暂不能编辑"
          : "正在原子解除素材引用，完成后再编辑时间线");
      return;
    }
    // Keep an optimistic reducer shadow so several actions batched before the
    // next render are compared and persisted in their exact order.
    timelineRef.current = next;
    if (projectChanged) {
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
    if (executableSelectionChanged) {
      segmentSelectionGeneration.current += 1;
      setCompileReport(null);
    }
    const activeDatabase = activeDatabaseRef.current;
    const selectionPreferenceScope = activeDatabase
      ? `${activeDatabase.active_database_identity}:${activeProjectIdRef.current}`
      : null;
    if (
      activeDatabase &&
      restoredSegmentSelectionKey.current === selectionPreferenceScope &&
      (segmentTopologyChanged || selectionChanged)
    ) {
      saveTimelineSegmentSelectionPreference(
        activeDatabase,
        activeProjectIdRef.current,
        next.project.segments.map((segment) => segment.id),
        next.selected_segment_ids,
      );
    }
    rawTimelineDispatch(action);
    if (sourceAudioFit.project !== reduced.project) {
      rawTimelineDispatch({ type: "project/replace", project: sourceAudioFit.project });
      const first = sourceAudioFit.adjustments[0];
      const label = first.segment_title || first.segment_id;
      const seconds = first.source_frames_after / sourceAudioFit.project.render.fps;
      const omitted = Math.max(0, first.source_frames_before - first.source_frames_after);
      const detail = first.fallback_to_previous_h3_length
        ? `源素材不足 ${first.output_frames_before} 帧，已自动缩短为 ${first.source_frames_after} 帧（${seconds.toFixed(4)} 秒）${omitted ? `，省略末尾 ${omitted} 帧` : ""}`
        : `已自动将源截取从 ${first.source_frames_before} 帧适配为 ${first.source_frames_after} 帧（${seconds.toFixed(4)} 秒）`;
      setToast(`${label}：${detail}${sourceAudioFit.adjustments.length > 1 ? `；另有 ${sourceAudioFit.adjustments.length - 1} 个片段已自动适配` : ""}`);
    }
  }, []);

  // Persist the same mechanical correction for timelines loaded from older
  // saves. Without this hydration pass, a legacy mismatch could disable the
  // run buttons before the user performs any edit that normally triggers the
  // reducer-side fit.
  useEffect(() => {
    if (timelineHydrationStatus !== "ready") return;
    if (!autoFitSourceAudioTiming(timeline.project).adjustments.length) return;
    dispatchTimeline({ type: "project/replace", project: timeline.project });
  }, [dispatchTimeline, timeline.project, timelineHydrationStatus]);

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

  const loadAssets = useCallback(async (
    signal?: AbortSignal,
    failClosed = false,
  ): Promise<boolean> => {
    const requestId = ++assetListRequest.current;
    try {
      const response = await directorApi.listAssets(undefined, signal);
      if (signal?.aborted || assetListRequest.current !== requestId) return false;
      const preference = loadAssetLayoutPreference();
      rawTimelineDispatch({
        type: "assets/replace",
        assets: orderAssetsByPreference(response.assets, preference.order),
      });
      return true;
    } catch (reason) {
      if (signal?.aborted || assetListRequest.current !== requestId) return false;
      if (failClosed) {
        rawTimelineDispatch({ type: "assets/replace", assets: [] });
        throw reason;
      }
      // An older backend may not expose the library yet; new uploads still appear.
      return false;
    }
  }, []);

  const refreshRuntimeResources = useCallback(async (
    origin: string,
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
      runtimeResourceRequest.current !== requestId ||
      authoritativeSettingsRef.current.comfy_url !== origin
    ) return false;

    const finalAuthority = finalAuthorityResult.status === "fulfilled"
      ? {
          settings: sanitizeRuntimeSettings(finalAuthorityResult.value.settings),
          token: finalAuthorityResult.value.authority_token,
        }
      : null;
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
    if (
      authorityChanged ||
      browserAuthorityChanged ||
      (initialAuthority && initialAuthority.settings.comfy_url !== origin) ||
      (finalAuthority && finalAuthority.settings.comfy_url !== origin)
    ) {
      // Every resource endpoint resolves its ComfyUI client from the current
      // server settings. A concurrent tab may therefore switch A -> B while
      // this page still names A. Never label B's responses as A: invalidate all
      // endpoint-scoped authorities immediately and let the App-owned settings
      // reconciliation adopt B together with its asset library.
      runtimeResourceRequest.current += 1;
      runtimeSettingsAuthorityReadyRef.current = false;
      authoritativeSettingsTokenRef.current = null;
      runtimeEndpointSwitchRequired.current = true;
      runtimeResourcesOriginRef.current = null;
      runtimeResourcesAuthorityTokenRef.current = null;
      setRuntimeResourcesOrigin(null);
      rayLightRuntimeStatusRef.current = null;
      setRayLightRuntimeStatus(null);
      setCapabilities({ ...EMPTY_CAPABILITIES, connection: "checking", message: "ComfyUI endpoint 已在服务器端变化，正在重新核对" });
      setGpus([]);
      setModels(EMPTY_MODELS);
      setLoadingModels(false);
      assetAuthorityRequired.current = true;
      assetListRequest.current += 1;
      rawTimelineDispatch({ type: "assets/replace", assets: [] });
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
      initialAuthority.settings.comfy_url === origin &&
      finalAuthority.settings.comfy_url === origin &&
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
      runtimeResourcesOriginRef.current = origin;
      runtimeResourcesAuthorityTokenRef.current = initialAuthority!.token;
      setRuntimeResourcesOrigin(origin);
      if (runtimeSettingsDesired.current) {
        window.queueMicrotask(() => runtimeSettingsDrainRef.current());
      } else if (runtimeSettingsSyncRequiredRef.current && !assetAuthorityRequired.current) {
        runtimeEndpointSwitchRequired.current = false;
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
        if (authoritativeSettingsRef.current.comfy_url !== origin) return;
        void runtimeResourceRefreshRef.current(origin, runtimeResourcesOriginRef.current === origin);
      }, RUNTIME_SETTINGS_RETRY_MS);
    }
    return false;
  }, [invalidateAndRefreshTaskSnapshots, setRuntimeAuthorityRequired]);
  runtimeResourceRefreshRef.current = (origin, preserveExisting) =>
    refreshRuntimeResources(origin, preserveExisting);

  const refreshAuthoritativeResourcesAfterConnectionTest = useCallback((testedOrigin: string) => {
    const authoritativeOrigin = authoritativeSettingsRef.current.comfy_url;
    if (
      testedOrigin !== authoritativeOrigin ||
      !isConfiguredComfyUrl(authoritativeOrigin) ||
      runtimeEndpointSwitchRequired.current
    ) return;
    // The probe only establishes reachability for its URL snapshot. App owns
    // resource authority and refreshes all four inventories in one
    // latest-wins generation; a partial same-origin failure preserves the
    // previously confirmed snapshot.
    setRuntimeAuthorityRequired(true);
    setCompileReport(null);
    void refreshRuntimeResources(
      authoritativeOrigin,
      runtimeResourcesOriginRef.current === authoritativeOrigin,
    );
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
    if (!isConfiguredComfyUrl(settings.comfy_url)) {
      runtimeResourceRequest.current += 1;
      if (runtimeResourceRetryTimer.current !== null) {
        window.clearTimeout(runtimeResourceRetryTimer.current);
        runtimeResourceRetryTimer.current = null;
      }
      runtimeResourcesOriginRef.current = null;
      runtimeResourcesAuthorityTokenRef.current = null;
      setRuntimeResourcesOrigin(null);
      setCapabilities({
        ...EMPTY_CAPABILITIES,
        connection: "unknown",
        message: "尚未配置 ComfyUI 地址",
      });
      rayLightRuntimeStatusRef.current = null;
      setRayLightRuntimeStatus(null);
      return settings;
    }
    await refreshRuntimeResources(
      settings.comfy_url,
      preserveResources && runtimeResourcesOriginRef.current === settings.comfy_url,
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
        if (isConfiguredComfyUrl(settings.comfy_url)) {
          if (runtimeResourcesOriginRef.current !== settings.comfy_url) {
            throw new Error("新 endpoint 的运行资源尚未完成同源核对");
          }
          const assetsReady = await loadAssets(controller.signal, true);
          throwIfAborted(controller.signal);
          if (!assetsReady) throw new Error("新 endpoint 的素材库刷新请求已过期");
        } else {
          assetListRequest.current += 1;
          rawTimelineDispatch({ type: "assets/replace", assets: [] });
        }
        assetAuthorityRequired.current = false;
        if (runtimeSettingsDesired.current) {
          window.queueMicrotask(() => runtimeSettingsDrainRef.current());
        } else {
          runtimeEndpointSwitchRequired.current = false;
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
    if (databaseIdentityStaleRef.current) {
      throw new Error("本页数据库身份已过期，请刷新整个页面");
    }
    if (rayLightRecoveryPendingRef.current) {
      throw new Error("RayLight 重启恢复正在核对，不能修改运行设置");
    }
    if (storageRestartRequiredRef.current) {
      throw new Error("数据库切换正在等待重启，不能继续修改运行设置");
    }
    if (assetDeleteLock.current || assetDeleteIntent.current) {
      throw new Error("正在原子解除素材引用，完成前不能切换运行设置");
    }
    if (assetUploadLock.current) {
      throw new Error("正在上传并绑定本地素材，完成前不能切换运行设置");
    }
    if (timelineSyncRequired) {
      throw new Error("服务器时间线正在自动恢复权威状态，完成前不能修改运行设置");
    }

    const normalized = nextSettings
      ? sanitizeRuntimeSettings(structuredClone(nextSettings))
      : null;
    const previousAuthority = authoritativeSettingsRef.current;
    const requestedEndpointChange = normalized !== null &&
      normalized.comfy_url !== previousAuthority.comfy_url;
    const generation = ++runtimeSettingsGeneration.current;
    if (requestedEndpointChange) runtimeEndpointSwitchRequired.current = true;

    setRuntimeSettingsOperationOwner(owner);
    setRuntimeAuthorityRequired(true);
    setCompileReport(null);
    const invalidateAssetAuthority = () => {
      assetAuthorityRequired.current = true;
      assetListRequest.current += 1;
      rawTimelineDispatch({ type: "assets/replace", assets: [] });
    };

    const operation = (async () => {
      if (requestedEndpointChange) {
        // Endpoint authority is an exclusive boundary with the timeline. An A
        // timeline response must be fully confirmed before any B settings PUT
        // can clear A's assets or change compilation authority.
        try {
          await flushTimelineAutosaveRef.current();
        } catch (reason) {
          // The timeline itself must remain editable so a deterministic 4xx
          // can be corrected. Generation stays fail-closed through the runtime
          // sync flag, while the endpoint edit remains queued for the next
          // successful timeline autosave.
          runtimeEndpointSwitchRequired.current = false;
          throw new RuntimeEndpointTimelineBoundaryError(reason);
        }
        if (
          !runtimeSettingsDesired.current || !normalized ||
          !sameRuntimeSettings(runtimeSettingsDesired.current, normalized)
        ) {
          runtimeEndpointSwitchRequired.current = false;
          throw new RuntimeSettingsSupersededError();
        }
        runtimeRequest.current += 1;
        runtimeResourceRequest.current += 1;
        if (runtimeResourceRetryTimer.current !== null) {
          window.clearTimeout(runtimeResourceRetryTimer.current);
          runtimeResourceRetryTimer.current = null;
        }
        runtimeResourcesOriginRef.current = null;
        runtimeResourcesAuthorityTokenRef.current = null;
        authoritativeSettingsTokenRef.current = null;
        setRuntimeResourcesOrigin(null);
        setCapabilities({ ...EMPTY_CAPABILITIES, connection: "checking", message: "等待服务器权威运行设置" });
        setGpus([]);
        setModels(EMPTY_MODELS);
        rayLightRuntimeStatusRef.current = null;
        setRayLightRuntimeStatus(null);
        setLoadingModels(false);
        invalidateAssetAuthority();
      } else if (assetAuthorityRequired.current) {
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

      const confirmed = await refreshRuntime(undefined, !requestedEndpointChange);
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

      if (confirmed.comfy_url !== previousAuthority.comfy_url && !assetAuthorityRequired.current) {
        invalidateAssetAuthority();
      }
      if (assetAuthorityRequired.current) {
        if (isConfiguredComfyUrl(confirmed.comfy_url)) {
          try {
            const refreshed = await loadAssets(undefined, true);
            if (!refreshed) throw new Error("素材库刷新请求已过期");
          } catch {
            throw new Error("运行设置已确认，但新 ComfyUI 对应的素材库无法权威刷新；旧素材已清空，生成保持锁定");
          }
        } else {
          assetListRequest.current += 1;
          rawTimelineDispatch({ type: "assets/replace", assets: [] });
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
  }, [invalidateAndRefreshTaskSnapshots, loadAssets, refreshRuntime, setRuntimeAuthorityRequired, timelineSyncRequired]);

  const drainRuntimeSettings = useCallback(() => {
    if (runtimeSettingsOperation.current || runtimeSettingsAutosaveTimer.current !== null) return;
    if (databaseIdentityStaleRef.current) return;
    if (storageRestartRequiredRef.current) return;
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
      timelineSyncRequiredRef.current || runtimeExecutionIntent.current > 0
    ) {
      if (runtimeSettingsRetryTimer.current === null) {
        runtimeSettingsRetryTimer.current = window.setTimeout(() => {
          runtimeSettingsRetryTimer.current = null;
          runtimeSettingsDrainRef.current();
        }, RUNTIME_SETTINGS_RETRY_MS);
      }
      return;
    }
    const desiredResourcesReady = !isConfiguredComfyUrl(desired.comfy_url) || (
      runtimeResourcesOriginRef.current === desired.comfy_url &&
      runtimeResourcesAuthorityTokenRef.current === authoritativeSettingsTokenRef.current
    );
    if (
      sameRuntimeSettings(desired, authoritativeSettingsRef.current) &&
      !assetAuthorityRequired.current &&
      desiredResourcesReady
    ) {
      runtimeSettingsDesired.current = null;
      runtimeSettingsPausedDesiredRef.current = null;
      clearPendingRuntimeSettings();
      setRuntimeSettingsDraft(desired);
      runtimeEndpointSwitchRequired.current = false;
      setRuntimeAuthorityRequired(false);
      invalidateAndRefreshTaskSnapshots();
      if (!storageRecoveryInProgress.current && timelinePersistedRevision.current < timelineRevision.current) {
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
      void runtimeResourceRefreshRef.current(desired.comfy_url, false);
      if (!storageRecoveryInProgress.current && timelinePersistedRevision.current < timelineRevision.current) {
        setTimelineRetryNonce((current) => current + 1);
      }
      return;
    }

    const snapshot = structuredClone(desired);
    const owner = runtimeSettingsDesiredOwner.current;
    let drainNewerImmediately = false;
    const operation = reconcileRuntimeSettings(owner, snapshot);
    void operation.then((confirmed) => {
      const confirmedResourcesReady = !isConfiguredComfyUrl(confirmed.comfy_url) || (
        runtimeResourcesOriginRef.current === confirmed.comfy_url &&
        runtimeResourcesAuthorityTokenRef.current === authoritativeSettingsTokenRef.current
      );
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
          runtimeEndpointSwitchRequired.current = false;
          setRuntimeAuthorityRequired(false);
          invalidateAndRefreshTaskSnapshots();
        }
        if (!storageRecoveryInProgress.current && timelinePersistedRevision.current < timelineRevision.current) {
          setTimelineRetryNonce((current) => current + 1);
        }
      } else if (
        runtimeSettingsDesired.current &&
        !sameRuntimeSettings(runtimeSettingsDesired.current, snapshot)
      ) {
        drainNewerImmediately = true;
      } else if (
        !runtimeSettingsDesired.current &&
        !runtimeSettingsDraftValidRef.current &&
        !assetAuthorityRequired.current &&
        confirmedResourcesReady
      ) {
        runtimeEndpointSwitchRequired.current = false;
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
      const waitingForTimelineBoundary = reason instanceof RuntimeEndpointTimelineBoundaryError;
      if (!superseded && deterministicClientError && !waitingForTimelineBoundary) {
        runtimeSettingsPausedDesiredRef.current = structuredClone(snapshot);
        setRuntimeSettingsPausedError(reason instanceof Error ? reason.message : "服务器拒绝当前系统设置");
      }
      if (
        !superseded && deterministicClientError &&
        authoritativeSettingsRef.current.comfy_url !== snapshot.comfy_url &&
        !assetAuthorityRequired.current
      ) {
        // The endpoint PUT was rejected and the authoritative readback still
        // names the old endpoint. No cross-endpoint boundary remains in
        // progress, so timeline correction must not stay frozen; runtime sync
        // and the persistent 4xx notice continue to block generation.
        runtimeEndpointSwitchRequired.current = false;
      }
      setToast(reason instanceof Error ? reason.message : "运行设置自动同步失败");
      if (
        !superseded && !explicitlySuperseded && !deterministicClientError && !waitingForTimelineBoundary &&
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
    if (databaseIdentityStaleRef.current) {
      return Promise.reject(new Error("本页数据库身份已过期，请刷新整个页面"));
    }
    if (rayLightRecoveryPendingRef.current) {
      return Promise.reject(new Error("RayLight 重启恢复正在核对，不能修改运行设置"));
    }
    if (storageRestartRequiredRef.current) {
      return Promise.reject(new Error("数据库切换正在等待重启，不能继续修改运行设置"));
    }
    if (storageOperationStatusRef.current !== "idle") {
      return Promise.reject(new Error("数据库操作结果尚未确认，不能继续修改运行设置"));
    }
    const normalized = sanitizeRuntimeSettings(structuredClone(nextSettings));
    runtimeSettingsDesired.current = normalized;
    runtimeSettingsPausedDesiredRef.current = null;
    runtimeSettingsDesiredOwner.current = owner;
    setRuntimeSettingsDraft(normalized);
    setRuntimeSettingsPausedError(null);
    if (
      runtimeExecutionIntent.current === 0 &&
      (normalized.comfy_url !== authoritativeSettingsRef.current.comfy_url ||
        runtimeEndpointSwitchRequired.current)
    ) runtimeEndpointSwitchRequired.current = true;
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
    // A merely invalid endpoint draft has not crossed an authority boundary:
    // keep generation fail-closed, but do not freeze timeline correction. Only
    // an endpoint operation that has already begun retains the exclusive gate.
    runtimeEndpointSwitchRequired.current = runtimeEndpointSwitchRequired.current &&
      runtimeSettingsOperation.current !== null;
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
      storageRestartRequiredRef.current ||
      timelineSyncRequiredRef.current ||
      assetDeleteLock.current ||
      assetUploadLock.current ||
      timelinePersistedRevision.current >= timelineRevision.current ||
      timelineRenderedRevision.current < timelineRevision.current
    ) {
      return Promise.resolve(null);
    }

    const generation = timelineWriteGeneration.current;
    const revision = timelineRevision.current;
    const snapshot = normalizeTimelineProject(structuredClone(timelineRef.current.project));
    if (!snapshot) return Promise.reject(new Error("时间线结构无效，请检查项目字段"));

    let mayDrainImmediately = false;
    let failedRevisionWasSuperseded = false;
    const operation = (async (): Promise<TimelineProject | null> => {
      const response = await saveTimelineForProject(activeProjectIdRef.current, snapshot);
      const confirmed = normalizeTimelineProject(response);
      if (!confirmed) throw new Error("服务器返回的时间线结构无效");
      // Exclusive mutations advance the generation. Their authority must never
      // be replaced by a response that started before them.
      if (timelineWriteGeneration.current !== generation) return null;

      timelinePersistedRevision.current = Math.max(
        timelinePersistedRevision.current,
        revision,
      );
      mayDrainImmediately = true;
      timelineHadLocal.current = true;
      if (timelineRevision.current !== revision) return null;

      const replaceAction: TimelineAction = { type: "project/replace", project: confirmed };
      timelineRef.current = timelineEditorReducer(timelineRef.current, replaceAction);
      rawTimelineDispatch(replaceAction);
      clearLocalTimeline();
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
    void operation.catch((reason) => {
      if (
        timelineWriteGeneration.current !== generation ||
        timelineSyncRequiredRef.current ||
        assetDeleteLock.current
      ) return;
      setTimelineDirty(true);
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
        !assetDeleteLock.current &&
        timelinePersistedRevision.current < timelineRevision.current
      ) {
        setTimelineRetryNonce((current) => current + 1);
      }
    });
    return operation;
  }, [invalidateAndRefreshTaskSnapshots]);

  const flushTimelineAutosave = useCallback(async (): Promise<TimelineProject> => {
    const generation = timelineWriteGeneration.current;
    if (timelineAutosaveTimer.current !== null) {
      window.clearTimeout(timelineAutosaveTimer.current);
      timelineAutosaveTimer.current = null;
    }
    while (timelinePersistedRevision.current < timelineRevision.current) {
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
    const markDatabaseIdentityStale = () => {
      databaseIdentityStaleRef.current = true;
      timelineHydrationReady.current = false;
      setTimelineHydrationStatus("stale");
      setToast("本页数据库身份已过期；请刷新整个页面后继续");
    };
    window.addEventListener(DATABASE_IDENTITY_STALE_EVENT, markDatabaseIdentityStale);
    return () => window.removeEventListener(DATABASE_IDENTITY_STALE_EVENT, markDatabaseIdentityStale);
  }, []);

  const flushRuntimeSettingsForStorageChange = useCallback(async (): Promise<void> => {
    if (!timelineHydrationReady.current || !activeDatabaseRef.current) {
      throw new Error("数据库身份与时间线仍在恢复，暂不能修改存储位置");
    }
    if (!runtimeSettingsAuthorityReadyRef.current) {
      throw new Error("运行设置尚未完成服务器权威读取，暂不能修改存储位置");
    }
    if (!runtimeSettingsDraftValidRef.current) {
      throw new Error("运行设置仍有无效输入，修正并同步后才能修改存储位置");
    }

    // Drain the exact latest desired document. The existing drain owns all
    // PUT + authoritative-GET reconciliation; this loop merely removes its
    // debounce and awaits the operation instead of allowing a storage switch
    // to race it.
    for (;;) {
      if (runtimeSettingsAutosaveTimer.current !== null) {
        window.clearTimeout(runtimeSettingsAutosaveTimer.current);
        runtimeSettingsAutosaveTimer.current = null;
      }
      if (runtimeSettingsRetryTimer.current !== null) {
        window.clearTimeout(runtimeSettingsRetryTimer.current);
        runtimeSettingsRetryTimer.current = null;
      }
      const paused = runtimeSettingsPausedDesiredRef.current;
      if (
        paused && runtimeSettingsDesired.current &&
        sameRuntimeSettings(paused, runtimeSettingsDesired.current)
      ) throw new Error("服务器拒绝当前运行设置，请先修正后再修改存储位置");

      const inFlight = runtimeSettingsOperation.current;
      if (inFlight) {
        await inFlight;
        await Promise.resolve();
        continue;
      }
      if (runtimeSettingsDesired.current) {
        runtimeSettingsDrainRef.current();
        const started = runtimeSettingsOperation.current;
        if (started) {
          await started;
          await Promise.resolve();
          continue;
        }
        if (!runtimeSettingsDesired.current) continue;
        throw new Error("运行设置当前无法同步，未执行数据库操作");
      }

      // Even without a pending write, perform one final authoritative GET so
      // a stale browser settings mirror can never cross the storage boundary.
      const confirmed = await refreshRuntime(undefined, true);
      if (!confirmed || !runtimeSettingsAuthorityReadyRef.current) {
        throw new Error("无法从服务器确认当前运行设置，未执行数据库操作");
      }
      if (!runtimeSettingsDesired.current && !runtimeSettingsOperation.current) return;
    }
  }, [refreshRuntime]);

  const prepareStorageChange = useCallback(async (): Promise<void> => {
    // Check hydration before draining anything. SettingsPage performs its own
    // GET /storage and can otherwise become clickable before App has inspected
    // a database-scoped timeline WAL.
    if (!timelineHydrationReady.current || !activeDatabaseRef.current) {
      throw new Error("数据库身份与时间线仍在恢复，暂不能修改存储位置");
    }
    if (rayLightRecoveryPendingRef.current) {
      throw new Error("RayLight 重启恢复正在核对，暂不能修改数据库存储");
    }
    if (
      storageRestartRequiredRef.current &&
      (runtimeSettingsDesired.current || runtimeSettingsOperation.current ||
        timelinePersistedRevision.current < timelineRevision.current)
    ) {
      throw new Error("当前页已有待同步修改且后端正在等待重启；请刷新页面后处理");
    }
    await flushRuntimeSettingsForStorageChange();
    if (!timelineHydrationReady.current) {
      throw new Error("时间线仍在恢复，未执行数据库操作");
    }
    await flushTimelineAutosave();
    clearPendingRuntimeSettings();
    clearLocalTimeline();
  }, [flushRuntimeSettingsForStorageChange, flushTimelineAutosave]);

  useEffect(() => {
    const controller = new AbortController();
    let timelineHydrationRetryTimer: number | null = null;
    let timelineHydrationAttempt = 0;
    taskListOwnerActive.current = true;
    const timelineGeneration = timelineWriteGeneration.current;
    void refreshRuntime(controller.signal);
    // StrictMode performs setup -> cleanup -> setup. Queue the second setup
    // behind the first request so its cleanup abort cannot leave the initial
    // task history empty until the next polling interval.
    void loadTasks(controller.signal, true);
    void loadAssets(controller.signal);
    const installHydratedProject = (
      project: TimelineProject,
      database: { active_database_path: string; active_database_identity: string },
      projectId: string,
    ) => {
      const segmentIds = project.segments.map((segment) => segment.id);
      const restoredSelection = loadTimelineSegmentSelectionPreference(
        database,
        projectId,
        segmentIds,
      );
      const replaceAction: TimelineAction = { type: "project/replace", project };
      const restoreSelectionAction: TimelineAction = {
        type: "segment/set-selection",
        ids: restoredSelection ?? segmentIds,
      };
      timelineRef.current = timelineEditorReducer(
        timelineEditorReducer(timelineRef.current, replaceAction),
        restoreSelectionAction,
      );
      rawTimelineDispatch(replaceAction);
      rawTimelineDispatch(restoreSelectionAction);
      restoredSegmentSelectionKey.current =
        `${database.active_database_identity}:${projectId}`;
    };
    const hydrateTimeline = async (): Promise<void> => {
      if (controller.signal.aborted || timelineHydrationReady.current) return;
      try {
        const hydratingProjectId = activeProjectIdRef.current;
        const storage = await directorApi.getStorage(controller.signal);
        if (controller.signal.aborted) return;
        const candidateDatabase = {
          active_database_path: storage.active_database_path,
          active_database_identity: storage.active_database_identity,
        };
        const persistedRuntimeSettings = loadPendingRuntimeSettings(candidateDatabase);
        const pending = loadLocalTimeline(candidateDatabase, hydratingProjectId);
        let serverProject: TimelineProject | null = null;
        if (!pending) {
          const value = await fetchTimelineForProject(hydratingProjectId, controller.signal);
          if (controller.signal.aborted) return;
          serverProject = normalizeTimelineProject(value);
          if (!serverProject) throw new Error("服务器返回的时间线结构无效");
        }
        const verification = await directorApi.getStorage(controller.signal);
        if (controller.signal.aborted) return;
        if (
          verification.active_database_identity !== candidateDatabase.active_database_identity ||
          verification.active_database_path !== candidateDatabase.active_database_path
        ) throw new DatabaseIdentityChangedDuringHydrationError();
        const latchedIdentity = directorApi.latchDatabaseIdentity(candidateDatabase.active_database_identity);
        if (latchedIdentity !== candidateDatabase.active_database_identity) {
          throw new DatabaseIdentityChangedDuringHydrationError();
        }
        activeDatabaseRef.current ??= candidateDatabase;
        storageRestartRequiredRef.current = verification.restart_required;
        setStorageRestartRequired(verification.restart_required);
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
        if (pending) {
          timelineRevision.current = 1;
          timelineHadLocal.current = true;
          installHydratedProject(pending, candidateDatabase, hydratingProjectId);
          setTimelineDirty(true);
          timelineHydrationReady.current = true;
          setTimelineHydrationStatus("ready");
          return;
        }
        if (
          timelineRevision.current > 0 || timelineHadLocal.current ||
          timelineWriteGeneration.current !== timelineGeneration
        ) {
          // A pending WAL or a newer exclusive authority owns the project. The
          // default in-memory document must never become writable merely because
          // this stale hydration request completed.
          return;
        }
        const project = serverProject as TimelineProject;
        installHydratedProject(project, candidateDatabase, hydratingProjectId);
        setTimelineDirty(false);
        timelineHydrationReady.current = true;
        setTimelineHydrationStatus("ready");
      } catch (reason) {
        if (controller.signal.aborted) return;
        if (reason instanceof DatabaseIdentityChangedDuringHydrationError) {
          databaseIdentityStaleRef.current = true;
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
          void hydrateTimeline();
        }, delay);
      }
    };
    void hydrateTimeline();
    return () => {
      taskListOwnerActive.current = false;
      if (timelineHydrationRetryTimer !== null) window.clearTimeout(timelineHydrationRetryTimer);
      controller.abort();
    };
  }, [refreshRuntime, loadTasks, loadAssets]);

  useEffect(() => { saveDirectorState(state); }, [state]);
  useEffect(() => {
    // Load the project list once and reconcile the persisted active-project
    // preference against it. The default project always exists server-side.
    let cancelled = false;
    void directorApi.listProjects().then((list) => {
      if (cancelled) return;
      setProjects(list.projects);
      const current = activeProjectIdRef.current;
      if (current !== DEFAULT_PROJECT_ID && !list.projects.some((project) => project.id === current)) {
        activeProjectIdRef.current = DEFAULT_PROJECT_ID;
        setActiveProjectIdState(DEFAULT_PROJECT_ID);
        persistActiveProjectId(DEFAULT_PROJECT_ID);
      }
    }).catch(() => {
      // Project list is a soft dependency; the default project remains usable.
    });
    return () => { cancelled = true; };
  }, []);
  useEffect(() => {
    if (timelineHydrationStatus !== "ready") return;
    const activeDatabase = activeDatabaseRef.current;
    if (!activeDatabase) return;
    const projectSegmentIds = timeline.project.segments.map((segment) => segment.id);
    const restoreKey = `${activeDatabase.active_database_identity}:${activeProjectId}`;
    if (restoredSegmentSelectionKey.current === restoreKey) return;
    restoredSegmentSelectionKey.current = restoreKey;
    const restored = loadTimelineSegmentSelectionPreference(
      activeDatabase,
      activeProjectId,
      projectSegmentIds,
    );
    if (restored === null) return;
    const action: TimelineAction = { type: "segment/set-selection", ids: restored };
    const currentRunnable = runnableTimelineSegmentIds(timelineRef.current);
    const next = timelineEditorReducer(timelineRef.current, action);
    const nextRunnable = runnableTimelineSegmentIds(next);
    if (
      currentRunnable.length !== nextRunnable.length ||
      currentRunnable.some((id, index) => id !== nextRunnable[index])
    ) {
      segmentSelectionGeneration.current += 1;
      setCompileReport(null);
    }
    timelineRef.current = next;
    rawTimelineDispatch(action);
  }, [activeProjectId, timeline.project.segments, timelineHydrationStatus]);
  useEffect(() => {
    if (
      timelineSyncRequired ||
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
  }, [timeline.project, timelineRetryNonce, timelineSyncRequired, runTimelineAutosave]);
  useEffect(() => () => {
    if (timelineAutosaveTimer.current !== null) window.clearTimeout(timelineAutosaveTimer.current);
    if (timelineRetryTimer.current !== null) window.clearTimeout(timelineRetryTimer.current);
    if (timelineAuthorityRetryTimer.current !== null) window.clearTimeout(timelineAuthorityRetryTimer.current);
  }, []);
  useEffect(() => {
    const persistPendingTimeline = () => {
      if (
        !timelineSyncRequiredRef.current &&
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

  const resyncTimeline = async (): Promise<void> => {
    if (assetDeleteLock.current || assetDeleteIntent.current) {
      if (timelineAuthorityRetryTimer.current !== null) window.clearTimeout(timelineAuthorityRetryTimer.current);
      timelineAuthorityRetryTimer.current = window.setTimeout(() => {
        timelineAuthorityRetryTimer.current = null;
        void resyncTimeline();
      }, 500);
      return;
    }
    const generation = ++timelineWriteGeneration.current;
    try {
      const authoritative = normalizeTimelineProject(await fetchTimelineForProject(activeProjectIdRef.current));
      if (!authoritative) throw new Error("服务器返回的时间线结构无效");
      if (timelineWriteGeneration.current !== generation) return;
      const replaceAction: TimelineAction = { type: "project/replace", project: authoritative };
      timelineRef.current = timelineEditorReducer(timelineRef.current, replaceAction);
      rawTimelineDispatch(replaceAction);
      timelinePersistedRevision.current = timelineRevision.current;
      setTimelineAuthorityRequired(false);
      setTimelineDirty(false);
      setTimelinePausedError(null);
      clearLocalTimeline();
      if (timelineAuthorityRetryTimer.current !== null) {
        window.clearTimeout(timelineAuthorityRetryTimer.current);
        timelineAuthorityRetryTimer.current = null;
      }
      invalidateAndRefreshTaskSnapshots();
      if (runtimeSettingsDesired.current) runtimeSettingsDrainRef.current();
    } catch (reason) {
      if (timelineWriteGeneration.current !== generation) return;
      setToast(`${reason instanceof Error ? reason.message : "服务器时间线同步失败"}；正在自动重试`);
      if (timelineAuthorityRetryTimer.current !== null) window.clearTimeout(timelineAuthorityRetryTimer.current);
      timelineAuthorityRetryTimer.current = window.setTimeout(() => {
        timelineAuthorityRetryTimer.current = null;
        void resyncTimeline();
      }, 1200);
    }
  };

  const submitTimeline = async (segmentIds: string[]) => {
    if (submitting) return;
    if (
      rayLightRecoveryPendingRef.current ||
      runtimeResourcesOriginRef.current !== authoritativeSettingsRef.current.comfy_url ||
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
    dispatchTimeline({ type: "project/replace", project: structuredClone(config) });
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
      runtimeResourcesOriginRef.current !== authoritativeSettingsRef.current.comfy_url ||
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
      const task = await directorApi.cancelTask(id);
      taskListRequest.current += 1;
      dispatch({ type: "tasks/upsert", task });
      void loadTasks(undefined, true);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "取消失败");
    }
  };
  const confirmComfyRestartRecovery = async (id: string) => {
    try {
      const task = await directorApi.confirmComfyRestartRecovery(id);
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
    const origin = authoritativeSettingsRef.current.comfy_url;
    const expected = rayLightRuntimeStatusRef.current;
    if (
      !isConfiguredComfyUrl(origin) ||
      runtimeResourcesOriginRef.current !== origin ||
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
    if (
      databaseIdentityStaleRef.current ||
      storageRestartRequiredRef.current ||
      storageOperationStatusRef.current !== "idle"
    ) return Promise.reject(new Error("数据库状态尚未稳定，暂不能恢复 RayLight"));
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
          if (
            authoritativeSettingsRef.current.comfy_url !== origin ||
            databaseIdentityStaleRef.current ||
            storageOperationStatusRef.current !== "idle"
          ) throw new Error("恢复核对期间 endpoint 或数据库状态发生变化");

          let deterministicFailure = false;
          let postError: unknown = null;
          try {
            await directorApi.confirmRayLightRuntimeRecovery(
              origin,
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
            origin,
            true,
            controller.signal,
            false,
          );
          throwIfAborted(controller.signal);
          const verified = rayLightRuntimeStatusRef.current;
          if (
            refreshed &&
            authoritativeSettingsRef.current.comfy_url === origin &&
            runtimeResourcesOriginRef.current === origin &&
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
        const result = await directorApi.cancelTasks(ids.slice(index, index + 100));
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
    if (timelineSyncRequiredRef.current || assetDeleteLock.current || assetDeleteIntent.current || assetUploadLock.current) {
      setToast("当前时间线或素材状态尚未稳定，暂不能另存历史项目");
      return;
    }
    try {
      if (timelineDirty) await flushTimelineAutosave();
      const snapshot = await directorApi.getTaskProject(id);
      // Restore the historical source as a brand-new project instead of
      // overwriting the one currently being edited.
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
    assetUploadLock.current = true;
    setAssetsUploading(true);
    try {
      const asset = await directorApi.importTaskOutput(id, output);
      rawTimelineDispatch({ type: "assets/add", assets: [asset] });
      rawTimelineDispatch({ type: "assets/select", id: asset.id });
      await loadAssets(undefined, true);
      setToast(`已把 ${asset.name} 转为 24fps 输入并加入当前素材库`);
    } catch (reason) {
      setToast(reason instanceof Error ? reason.message : "任务输出导入失败");
    } finally {
      assetUploadLock.current = false;
      setAssetsUploading(false);
      if (
        !timelineSyncRequiredRef.current &&
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
    if (timelineSyncRequired) { setToast("服务器时间线正在自动恢复同步，暂不能从素材库移出"); return; }
    assetDeleteIntent.current = true;
    setAssetsDeleting(true);
    const deleted: string[] = [];
    const failures: string[] = [];
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
      assetDeleteLock.current = true;
      timelineWriteGeneration.current += 1;
      for (const id of ids) {
        const hasLocalUsage = timelineAssetUsages(snapshot, id).length > 0;
        try {
          if (hasLocalUsage) {
            prepareCascade();
            await directorApi.deleteAssetCascade(id);
          } else {
            try {
              await directorApi.deleteAsset(id);
            } catch (reason) {
              const details = reason instanceof ApiError && reason.status === 409 && reason.details && typeof reason.details === "object"
                ? (reason.details as { detail?: { usages?: unknown } }).detail
                : undefined;
              const usages = Array.isArray(details?.usages)
                ? details.usages.filter((usage): usage is string => typeof usage === "string")
                : [];
              if (!usages.length) throw reason;
              if (!window.confirm(`素材还被其他时间线草稿引用：\n\n${usages.join("\n")}\n\n是否从所有草稿原子解除引用并从素材库移出？ComfyUI 文件会保留。`)) {
                failures.push(`已跳过素材 ${id}`);
                continue;
              }
              prepareCascade();
              await directorApi.deleteAssetCascade(id);
            }
          }
          deleted.push(id);
        } catch (reason) {
          failures.push(reason instanceof Error ? reason.message : `素材 ${id} 移出失败`);
        }
      }
      if (deleted.length || timelineAuthorityRequired) {
        try {
          const authoritative = normalizeTimelineProject(await fetchTimelineForProject(activeProjectIdRef.current));
          if (!authoritative) throw new Error("服务器时间线响应无效");
          const replaceAction: TimelineAction = { type: "project/replace", project: authoritative };
          timelineRef.current = timelineEditorReducer(timelineRef.current, replaceAction);
          rawTimelineDispatch(replaceAction);
          timelinePersistedRevision.current = timelineRevision.current;
          setTimelineAuthorityRequired(false);
          setTimelineDirty(false);
          setTimelinePausedError(null);
          timelineHadLocal.current = true;
          clearLocalTimeline();
          invalidateAndRefreshTaskSnapshots();
        } catch {
          rawTimelineDispatch({ type: "assets/remove", ids: deleted });
          await loadAssets();
          if (timelineAuthorityRequired) {
            // Cascade may have committed. Keep the known-good pre-cascade
            // snapshot visible, but never persist/edit it as authoritative.
            setTimelineAuthorityRequired(true);
            setTimelineDirty(false);
            timelineHadLocal.current = false;
            clearLocalTimeline();
            setToast("素材级联结果无法确认；编辑与生成已锁定，正在自动恢复同步");
          } else {
            setToast("素材已从素材库移出，但列表刷新失败；时间线未受影响");
          }
          return;
        }
      }
      await loadAssets();
      setToast(failures.length
        ? `已从素材库移出 ${deleted.length} 个；${failures[0]}`
        : `已从素材库移出 ${deleted.length} 个登记；ComfyUI 文件保留`);
    } catch (reason) {
      if (!timelineAuthorityRequired) setTimelineDirty(
        timelinePersistedRevision.current < timelineRevision.current,
      );
      setToast(reason instanceof Error ? reason.message : "素材移出失败");
    } finally {
      assetDeleteLock.current = false;
      assetDeleteIntent.current = false;
      setAssetsDeleting(false);
      if (timelineSyncRequiredRef.current) void resyncTimeline();
      else if (runtimeSettingsDesired.current) {
        if (runtimeSettingsRetryTimer.current !== null) {
          window.clearTimeout(runtimeSettingsRetryTimer.current);
          runtimeSettingsRetryTimer.current = null;
        }
        runtimeSettingsDrainRef.current();
      }
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
    if (timelineSyncRequired) throw new Error("服务器时间线正在自动恢复权威状态，暂不能上传素材");
    if (capabilities.connection !== "online" || !isConfiguredComfyUrl(authoritativeSettingsRef.current.comfy_url)) {
      throw new Error("ComfyUI 尚未连接，暂不能上传素材");
    }

    const { accepted, unsupported } = classifyDroppedFiles(files);
    const generation = runtimeSettingsGeneration.current;
    const origin = authoritativeSettingsRef.current.comfy_url;
    const authorityCurrent = () =>
      runtimeSettingsGeneration.current === generation &&
      authoritativeSettingsRef.current.comfy_url === origin &&
      (runtimeSettingsDesired.current?.comfy_url ?? runtimeSettingsDraft.comfy_url) === origin &&
      !runtimeSettingsOperation.current;
    assetUploadLock.current = true;
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
  }, [capabilities.connection, runtimeSettingsDraft.comfy_url, timelineSyncRequired]);

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
  const runtimeConfigured = isConfiguredComfyUrl(state.settings.comfy_url);
  const runtimeReady = runtimeConfigured && capabilities.connection === "online" &&
    runtimeResourcesOrigin === state.settings.comfy_url && rayLightRuntimeStatus !== null;
  const rayLightRecoveryRequired = runtimeReady &&
    rayLightRuntimeStatus?.recovery_required === true;
  const runtimeAuthorityPending = runtimeSettingsOperationOwner !== null || runtimeSettingsSyncRequired;
  const timelineHydrated = timelineHydrationStatus === "ready";
  const databaseIdentityStale = timelineHydrationStatus === "stale";
  const storageOperationPending = storageOperationStatus !== "idle";
  const workspaceRuntimeReady = runtimeReady && !rayLightRecoveryRequired && !rayLightRecoveryPending && timelineHydrated && !storageRestartRequired && !storageOperationPending && !runtimeAuthorityPending && !timelineSyncRequired && !assetsDeleting && !assetsUploading;
  const workspaceCapabilities: CapabilityReport = !runtimeReady || rayLightRecoveryRequired || rayLightRecoveryPending || storageRestartRequired || storageOperationPending || runtimeAuthorityPending || timelineSyncRequired || assetsDeleting || assetsUploading
    ? {
        ...capabilities,
        connection: "checking",
        message: !runtimeReady
          ? "ComfyUI 运行资源等待同源权威核对"
          : rayLightRecoveryRequired
          ? "旧 RayLight 运行状态等待重启确认"
          : rayLightRecoveryPending
            ? "正在核对 RayLight 重启恢复结果"
          : storageRestartRequired
          ? "数据库切换正在等待重启"
          : storageOperationPending
            ? "数据库操作结果等待权威确认"
          : runtimeAuthorityPending
          ? "运行设置等待服务器权威回读"
          : timelineSyncRequired
            ? "时间线等待服务器权威回读"
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
    ...(storageRestartRequired ? ["数据库切换正在等待重启"] : []),
    ...(storageOperationPending ? ["数据库操作结果尚未完成权威确认"] : []),
    ...(runtimeAuthorityPending ? ["运行设置尚未完成服务器权威回读"] : []),
    ...(!runtimeReady ? ["ComfyUI 与 RayLight 运行资源尚未完成同源权威核对"] : []),
    ...(rayLightRecoveryRequired ? ["旧 RayLight 运行状态引用当前不可见 GPU；请在系统设置确认 ComfyUI 已重启并恢复"] : []),
    ...(rayLightRecoveryPending ? ["正在核对 RayLight 重启恢复结果"] : []),
    ...(timelineSyncRequired ? ["素材级联已提交，但服务器时间线尚未完成权威回读"] : []),
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
  const assetUsages = Object.fromEntries(timeline.assets.map((asset) => [
    asset.id,
    timelineAssetUsages(timeline.project, asset.id).map((usage) =>
      `${usage.segment_index + 1} · ${usage.segment_title} · ${usage.role}`),
  ]));
  const renderedAssetOrigin = state.settings.comfy_url;
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
    // The project list mirrors the timeline title; update it optimistically.
    setProjects((current) => current.map((project) =>
      project.id === activeProjectIdRef.current ? { ...project, title } : project,
    ));
  };

  const switchProject = async (targetId: string): Promise<boolean> => {
    // Even choosing the already-active project is meaningful: it cancels an
    // unresolved switch whose response has not taken authority yet.
    const switchGeneration = ++projectSwitchGeneration.current;
    if (targetId === activeProjectIdRef.current) return true;
    if (runtimeExecutionIntent.current > 0) {
      setToast("生成或预检正在确认当前项目；完成前不能切换项目");
      return false;
    }
    if (timelineHydrationReady.current) {
      try {
        await flushTimelineAutosave();
        if (projectSwitchGeneration.current !== switchGeneration) return false;
      } catch (reason) {
        if (projectSwitchGeneration.current !== switchGeneration) return false;
        setToast(`切换前同步当前项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
        return false;
      }
    }
    const database = activeDatabaseRef.current;
    let targetProject: TimelineProject | null = database
      ? loadLocalTimeline(database, targetId)
      : null;
    if (!targetProject) {
      try {
        targetProject = normalizeTimelineProject(await fetchTimelineForProject(targetId));
        if (projectSwitchGeneration.current !== switchGeneration) return false;
      } catch (reason) {
        if (projectSwitchGeneration.current !== switchGeneration) return false;
        setToast(`加载目标项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
        return false;
      }
    }
    if (projectSwitchGeneration.current !== switchGeneration) return false;
    if (!targetProject) { setToast("目标项目时间线结构无效"); return false; }
    // The target request may have been pending long enough for another edit
    // to occur in the current project. Drain again immediately before the
    // synchronous authority hand-off so that edit cannot be discarded.
    if (timelineHydrationReady.current) {
      try {
        await flushTimelineAutosave();
        if (
          projectSwitchGeneration.current !== switchGeneration ||
          runtimeExecutionIntent.current > 0
        ) return false;
      } catch (reason) {
        if (projectSwitchGeneration.current !== switchGeneration) return false;
        setToast(`切换前同步当前项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
        return false;
      }
    }
    const targetSegmentIds = targetProject.segments.map((segment) => segment.id);
    const targetSelectionScope = database
      ? `${database.active_database_identity}:${targetId}`
      : null;
    const restoredSelection = database
      ? loadTimelineSegmentSelectionPreference(database, targetId, targetSegmentIds)
      : null;
    const replaceAction: TimelineAction = {
      type: "project/replace",
      project: targetProject,
    };
    const restoreSelectionAction: TimelineAction = {
      type: "segment/set-selection",
      ids: restoredSelection ?? targetSegmentIds,
    };
    const nextState = timelineEditorReducer(
      timelineEditorReducer(timelineRef.current, replaceAction),
      restoreSelectionAction,
    );
    timelineRef.current = nextState;
    segmentSelectionGeneration.current += 1;
    rawTimelineDispatch(replaceAction);
    // A different project starts with its own default selection even when an
    // imported/cloned timeline happens to reuse segment IDs. Its project-
    // scoped browser preference, if any, is restored in the same transition.
    rawTimelineDispatch(restoreSelectionAction);
    timelineRevision.current = 0;
    timelinePersistedRevision.current = 0;
    timelineHadLocal.current = false;
    timelineWriteGeneration.current += 1;
    restoredSegmentSelectionKey.current = targetSelectionScope;
    timelineHydrationReady.current = true;
    setTimelineHydrationStatus("ready");
    setTimelineDirty(false);
    setCompileReport(null);
    setTimelinePausedError(null);
    clearLocalTimeline();
    activeProjectIdRef.current = targetId;
    setActiveProjectIdState(targetId);
    persistActiveProjectId(targetId);
    // Refresh the task drawer's current-project comparison for the new project.
    taskListRequest.current += 1;
    void loadTasks(undefined, true);
    setToast(`已切换到项目“${targetProject.title}”`);
    return true;
  };

  const createProject = async (title?: string) => {
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
    if (runtimeExecutionIntent.current > 0) {
      setToast("生成或预检正在确认当前项目；完成前不能删除项目");
      return;
    }
    try {
      if (
        activeProjectIdRef.current === projectId &&
        !await switchProject(DEFAULT_PROJECT_ID)
      ) return;
      const response = await directorApi.deleteProject(projectId);
      setProjects((current) => current.filter((project) => project.id !== projectId));
      setToast(`已删除项目；${response.orphaned_jobs} 个历史任务已归档为旧任务`);
    } catch (reason) {
      setToast(`删除项目失败：${reason instanceof Error ? reason.message : "未知错误"}`);
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
          runtimeConfigured={runtimeConfigured}
          settingsActive={state.view === "settings"}
          deleting={assetsDeleting}
          assetUsages={assetUsages}
          onUploadFiles={uploadWorkspaceFiles}
          onUploaded={(assets) => {
            // An upload started before an endpoint switch may finish after the
            // new asset list is authoritative. Never merge that old-origin
            // response into the new workspace library.
            if (
              renderedAssetOrigin !== authoritativeSettingsRef.current.comfy_url ||
              runtimeSettingsOperation.current ||
              runtimeSettingsSyncRequiredRef.current
            ) return;
            rawTimelineDispatch({ type: "assets/add", assets });
          }}
          onSelect={(id, additive) => rawTimelineDispatch({ type: "assets/select", id, additive })}
          onSelectRange={(ids, additive) => rawTimelineDispatch({
            type: "assets/set-selection", ids, additive,
          })}
          onMove={(draggedId, targetId) => rawTimelineDispatch({ type: "assets/move", draggedId, targetId })}
          onGridSize={(size) => rawTimelineDispatch({ type: "assets/grid-size", size })}
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
              dispatch({ type: "tasks/panel", open: false });
            }
            dispatch({ type: opening ? "navigate/settings" : "navigate/workspace" });
          }}
        />

      <div className="app-main">
        {storageRestartRequired && <div className="timeline-hydration-notice" role="status" aria-live="polite">
          <span>数据库路径已变更；当前页面停止修改。请重启 Director 并刷新页面后继续。</span>
        </div>}
        {storageOperationPending && <div className="timeline-hydration-notice" role="status" aria-live="polite">
          {storageOperationStatus === "reconciling" && <Spinner />}
          <span>{storageOperationStatus === "reconciling"
            ? "数据库操作响应尚未确认，正在自动核对服务器状态；确认前当前页面停止修改。"
            : storageOperationStatus === "recovering"
              ? "正在恢复当前数据库中保留的修改；完成前当前页面停止修改。"
              : "正在建立数据库存储变更边界；完成前当前页面停止修改。"}</span>
        </div>}
        {!timelineHydrated && <div className="timeline-hydration-notice" role="status" aria-live="polite">
          {timelineHydrationStatus !== "stale" && <Spinner />}
          <span>{timelineHydrationStatus === "stale"
            ? "Director 后端数据库已变化；本页已停止修改。请刷新整个页面后继续。"
            : timelineHydrationStatus === "retrying"
              ? "暂时无法确认数据库或读取服务器时间线，正在自动重试；恢复前编辑已锁定。"
              : "正在从服务器恢复时间线；恢复前编辑已锁定。"}</span>
        </div>}
        <div className="workspace-surface" {...(state.view === "settings" || !timelineHydrated || storageRestartRequired || storageOperationPending ? { inert: true } : {})}>
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
                disabled={!timelineHydrated || submitting || compiling}
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
                  disabled={submitting || compiling}
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
            <button ref={globalSettingsToggleRef} type="button" className="topbar__global-toggle" aria-label="全局设置" aria-expanded={globalSettingsOpen} aria-controls={GLOBAL_SETTINGS_ID} onClick={() => {
              const opening = !globalSettingsOpen;
              if (opening) dispatch({ type: "tasks/panel", open: false });
              setGlobalSettingsOpen(opening);
            }}><span>全局设置</span><i aria-hidden="true" /></button>
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
              if (opening) setGlobalSettingsOpen(false);
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
              runtimeReady={runtimeSettingsDraftValid && runtimeResourcesOrigin === runtimeSettingsDraft.comfy_url && capabilities.connection === "online" && !rayLightRecoveryRequired}
              modelSaving={runtimeSettingsOperationOwner !== null}
              onClose={() => { setGlobalSettingsOpen(false); window.requestAnimationFrame(() => globalSettingsToggleRef.current?.focus()); }}
              onChange={(project) => dispatchTimeline({ type: "project/replace", project })}
              onRuntimeModelChange={(role, patch) => void updateRuntimeModel(role, patch)}
            />
            {!workspaceRuntimeReady && <div className="timeline-runtime-notice">{!runtimeSettingsDraftValid ? "系统设置有无效输入，请打开并修正；有效后自动应用。" : runtimeSettingsPausedError ? `服务器拒绝当前系统设置：${runtimeSettingsPausedError}。请打开并修改；有效修改后自动应用。` : rayLightRecoveryRequired ? "旧 RayLight 运行状态引用了当前不可见 GPU；请打开系统设置，确认 ComfyUI 已重启后执行恢复。" : runtimeSettingsSyncRequired ? "运行设置或素材库正在后台自动核对；恢复权威状态前，生成与素材操作保持锁定。" : runtimeSettingsOperationOwner !== null ? "运行设置正在同步并从服务器权威回读；完成前不能生成或操作素材。" : timelineSyncRequired ? "素材操作结果正在自动核对；恢复权威时间线前，编辑与生成保持锁定。" : assetsDeleting ? "正在原子解除素材引用；时间线编辑与生成暂时锁定。" : assetsUploading ? assetUploadProgress ? `${describeUploadProgress(assetUploadProgress)}；完成前暂时锁定同步、预检和生成。` : "正在上传并绑定本地素材；完成前暂时锁定同步、预检和生成。" : capabilities.connection === "offline" ? "ComfyUI 当前离线；编辑内容会在 Director 连接恢复后自动同步，暂时不能生成。" : !runtimeConfigured ? "尚未配置 ComfyUI；请在系统设置填写服务器地址。" : "正在检查 ComfyUI 能力…"}</div>}
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
            resourcesOrigin={runtimeResourcesOrigin}
            capabilities={capabilities}
            gpus={gpus}
            models={models}
            rayLightRuntimeStatus={rayLightRuntimeStatus}
            rayLightRecoveryPending={rayLightRecoveryPending}
            rayLightRecoveryDisabled={runtimeAuthorityPending || storageOperationPending || databaseIdentityStale || activeTasks.length > 0}
            rayLightRecoveryBlockedReason={runtimeAuthorityPending
              ? "运行设置仍在同步"
              : storageOperationPending
                ? "数据库操作结果仍在核对"
                : databaseIdentityStale
                  ? "本页数据库身份已过期"
                  : activeTasks.length > 0
                    ? "仍有 Director 任务未结束"
                    : null}
            loadingModels={loadingModels}
            syncError={runtimeSettingsPausedError}
            runtimeEditingDisabled={!timelineHydrated || storageRestartRequired || storageOperationPending || databaseIdentityStale || rayLightRecoveryPending}
            storageOperationsDisabled={!timelineHydrated || storageOperationPending || databaseIdentityStale || rayLightRecoveryPending}
            theme={theme}
            onThemeChange={setTheme}
            onDraftChange={updateRuntimeSettingsDraft}
            onSaved={(next) => queueRuntimeSettings("settings-page", next)}
            onBeforeStorageChange={prepareStorageChange}
            onStorageOperationStarted={beginStorageOperation}
            onStorageOperationAborted={abortStorageOperation}
            onStorageOperationUncertain={reconcileUncertainStorageOperation}
            onStorageConfigurationChanged={acceptStorageConfiguration}
            onStorageSwitchCancelled={async () => {
              storageRecoveryInProgress.current = true;
              setStorageOperationLock("recovering");
              try {
                await flushRuntimeSettingsForStorageChange();
                await flushTimelineAutosave();
              } finally {
                storageRecoveryInProgress.current = false;
                setStorageOperationLock("idle");
              }
            }}
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
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}
