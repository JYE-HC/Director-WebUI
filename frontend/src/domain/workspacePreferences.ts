import type { RV2VShotDetectionRequest } from "../api/types";
import {
  DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS,
  type TimelineSegmentCopyOptions,
} from "./timelineProject";
import { isStoragePath } from "./storagePath";

export const TIMELINE_WORKSPACE_PREFERENCES_KEY =
  "directordeck:v1:timeline-workspace-preferences";
const TIMELINE_SEGMENT_SELECTION_KEY_PREFIX =
  "directordeck:v2:timeline-segment-selection";
export const TIMELINE_SEGMENT_COPY_OPTIONS_KEY =
  "directordeck:v1:timeline-segment-copy-options";

export interface TimelineWorkspacePreferences {
  version: 1;
  showLiveMonitor: boolean;
  volume: number;
  loop: boolean;
  compareOriginal: boolean;
  timelineZoom: number;
  evenSplitPieces: number;
  detectionSensitivity: RV2VShotDetectionRequest["sensitivity"];
  minimumShotFrames: number;
  assetFilter: "all" | "image" | "video" | "audio";
  taskTab: "all" | "completed" | "failed";
  taskCurrentProjectOnly: boolean;
  taskSort: "recent" | "duration";
}

export interface TimelinePreferenceDatabase {
  active_database_path: string;
}

const DEFAULT_TIMELINE_WORKSPACE_PREFERENCES: TimelineWorkspacePreferences = {
  version: 1,
  showLiveMonitor: false,
  volume: 0.8,
  loop: false,
  compareOriginal: false,
  timelineZoom: 48,
  evenSplitPieces: 2,
  detectionSensitivity: "medium",
  minimumShotFrames: 12,
  assetFilter: "all",
  taskTab: "all",
  taskCurrentProjectOnly: false,
  taskSort: "recent",
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function boundedNumber(value: unknown, fallback: number, minimum: number, maximum: number): number {
  return finiteNumber(value) ? Math.min(maximum, Math.max(minimum, value)) : fallback;
}

function boundedInteger(value: unknown, fallback: number, minimum: number, maximum: number): number {
  return Math.trunc(boundedNumber(value, fallback, minimum, maximum));
}

function detectionSensitivity(
  value: unknown,
): RV2VShotDetectionRequest["sensitivity"] {
  return value === "low" || value === "high" || value === "medium" ? value : "medium";
}

export function loadTimelineWorkspacePreferences(): TimelineWorkspacePreferences {
  const fallback = { ...DEFAULT_TIMELINE_WORKSPACE_PREFERENCES };
  try {
    const raw = window.localStorage.getItem(TIMELINE_WORKSPACE_PREFERENCES_KEY);
    if (!raw) return fallback;
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value) || value.version !== 1) return fallback;
    return {
      version: 1,
      showLiveMonitor: value.showLiveMonitor === true,
      volume: boundedNumber(value.volume, fallback.volume, 0, 1),
      loop: value.loop === true,
      compareOriginal: value.compareOriginal === true,
      // “适合窗口” intentionally permits a scale below the manual 12 px/s limit.
      timelineZoom: boundedNumber(value.timelineZoom, fallback.timelineZoom, 0.05, 240),
      evenSplitPieces: boundedInteger(value.evenSplitPieces, fallback.evenSplitPieces, 2, 128),
      detectionSensitivity: detectionSensitivity(value.detectionSensitivity),
      minimumShotFrames: boundedInteger(
        value.minimumShotFrames,
        fallback.minimumShotFrames,
        4,
        100_000,
      ),
      assetFilter: value.assetFilter === "image" || value.assetFilter === "video" ||
        value.assetFilter === "audio" ? value.assetFilter : "all",
      taskTab: value.taskTab === "completed" || value.taskTab === "failed"
        ? value.taskTab
        : "all",
      taskCurrentProjectOnly: value.taskCurrentProjectOnly === true,
      taskSort: value.taskSort === "duration" ? "duration" : "recent",
    };
  } catch {
    return fallback;
  }
}

export function saveTimelineWorkspacePreferences(
  preferences: TimelineWorkspacePreferences,
): void {
  try {
    window.localStorage.setItem(
      TIMELINE_WORKSPACE_PREFERENCES_KEY,
      JSON.stringify(preferences),
    );
  } catch {
    // Browser preferences remain best effort when storage is unavailable.
  }
}

export function updateTimelineWorkspacePreferences(
  patch: Partial<Omit<TimelineWorkspacePreferences, "version">>,
): void {
  saveTimelineWorkspacePreferences({
    ...loadTimelineWorkspacePreferences(),
    ...patch,
    version: 1,
  });
}

function copyOption(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

export function normalizeTimelineSegmentCopyOptions(
  value: Partial<TimelineSegmentCopyOptions>,
): TimelineSegmentCopyOptions {
  const fallback = DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS;
  const prompt = copyOption(value.prompt, fallback.prompt);
  const mode = copyOption(value.mode, fallback.mode);
  return {
    mode,
    duration: copyOption(value.duration, fallback.duration),
    continuity: copyOption(value.continuity, fallback.continuity),
    audioMode: copyOption(value.audioMode, fallback.audioMode),
    refImageSize: copyOption(value.refImageSize, fallback.refImageSize),
    prompt,
    // Referenced material has no independent meaning in this workflow: its
    // slots are copied only alongside Prompt and the source generation mode.
    promptReferences: prompt && mode && copyOption(
      value.promptReferences,
      fallback.promptReferences,
    ),
  };
}

export function loadTimelineSegmentCopyOptions(): TimelineSegmentCopyOptions {
  try {
    const raw = window.localStorage.getItem(TIMELINE_SEGMENT_COPY_OPTIONS_KEY);
    if (!raw) return { ...DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS };
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value) || value.version !== 1) {
      return { ...DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS };
    }
    return normalizeTimelineSegmentCopyOptions(value);
  } catch {
    return { ...DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS };
  }
}

export function saveTimelineSegmentCopyOptions(
  options: TimelineSegmentCopyOptions,
): void {
  try {
    window.localStorage.setItem(TIMELINE_SEGMENT_COPY_OPTIONS_KEY, JSON.stringify({
      version: 1,
      ...normalizeTimelineSegmentCopyOptions(options),
    }));
  } catch {
    // The current in-memory options remain usable without browser storage.
  }
}

function validDatabase(database: TimelinePreferenceDatabase): boolean {
  return isStoragePath(database.active_database_path);
}

function segmentSelectionKey(
  database: TimelinePreferenceDatabase,
  projectId: string,
): string {
  return `${TIMELINE_SEGMENT_SELECTION_KEY_PREFIX}:${database.active_database_path}:${projectId}`;
}

export function saveTimelineSegmentSelectionPreference(
  database: TimelinePreferenceDatabase,
  projectId: string,
  projectSegmentIds: readonly string[],
  selectedSegmentIds: readonly string[],
): void {
  if (!validDatabase(database) || !projectId) return;
  const projectIds = [...projectSegmentIds];
  const selectedIds = projectIds.filter((id) => selectedSegmentIds.includes(id));
  const mode = projectIds.length === selectedIds.length ? "all" : "explicit";
  try {
    window.localStorage.setItem(segmentSelectionKey(database, projectId), JSON.stringify({
      version: 2,
      active_database_path: database.active_database_path,
      active_project_id: projectId,
      mode,
      project_segment_ids: projectIds,
      selected_segment_ids: selectedIds,
    }));
  } catch {
    // The current in-memory selection remains usable without browser storage.
  }
}

export function loadTimelineSegmentSelectionPreference(
  database: TimelinePreferenceDatabase,
  projectId: string,
  projectSegmentIds: readonly string[],
): string[] | null {
  if (!validDatabase(database) || !projectId) return null;
  try {
    const raw = window.localStorage.getItem(segmentSelectionKey(database, projectId));
    if (!raw) return null;
    const value: unknown = JSON.parse(raw);
    if (
      !isRecord(value) ||
      value.version !== 2 ||
      value.active_database_path !== database.active_database_path ||
      value.active_project_id !== projectId ||
      (value.mode !== "all" && value.mode !== "explicit") ||
      !Array.isArray(value.project_segment_ids) ||
      !Array.isArray(value.selected_segment_ids) ||
      !value.project_segment_ids.every((id) => typeof id === "string") ||
      !value.selected_segment_ids.every((id) => typeof id === "string")
    ) return null;
    const currentProjectIds = new Set(projectSegmentIds);
    if (!(value.project_segment_ids as string[]).some((id) => currentProjectIds.has(id))) {
      return null;
    }
    if (value.mode === "all") return [...projectSegmentIds];
    const selected = new Set(value.selected_segment_ids as string[]);
    return projectSegmentIds.filter((id) => selected.has(id));
  } catch {
    return null;
  }
}
