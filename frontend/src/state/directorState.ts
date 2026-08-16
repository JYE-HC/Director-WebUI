import {
  createInitialDrafts,
  MODE_ORDER,
  type AssetReference,
  type GenerationMode,
  type ModeDraft,
  type ModeDraftMap,
  type SlottedAssetReference,
} from "../domain/modes";
import {
  DEFAULT_SETTINGS,
  sanitizeRuntimeSettings,
  type GenerationTask,
  type RuntimeSettings,
} from "../api/types";
import {
  appendToLowestFreeSlots,
  fitSourceRangeToVideo,
  normalizeAssetReference,
} from "../domain/assets";
import { normalizeModeDraft } from "../domain/drafts";
import { MINIMAX_H3_REFERENCE_LIMITS } from "../domain/h3Capabilities";

export type AppView = "workspace" | "settings";

export interface DirectorState {
  view: AppView;
  activeMode: GenerationMode;
  drafts: ModeDraftMap;
  settings: RuntimeSettings;
  tasks: GenerationTask[];
  taskPanelOpen: boolean;
  draftSync: Record<GenerationMode, DraftSyncState>;
}

export type DraftSaveStatus = "idle" | "saving" | "saved" | "error";

export interface DraftSyncState {
  revision: number;
  dirty: boolean;
  savingRevision: number | null;
  status: DraftSaveStatus;
}

export type DraftAssetField =
  | "first_image"
  | "last_image"
  | "source_video"
  | "reference_images"
  | "reference_audios"
  | "reference_videos";

export type AssetMutation =
  | { type: "add"; assets: AssetReference[] }
  | { type: "remove"; assetId: string };

export type DirectorAction =
  | { type: "navigate/mode"; mode: GenerationMode }
  | { type: "navigate/workspace" }
  | { type: "navigate/settings" }
  | { type: "draft/replace"; draft: ModeDraft }
  | { type: "draft/hydrate"; mode: GenerationMode; draft: unknown }
  | {
      type: "draft/assets";
      mode: GenerationMode;
      shotId: string;
      field: DraftAssetField;
      mutation: AssetMutation;
    }
  | { type: "draft/save-start"; mode: GenerationMode; revision: number }
  | { type: "draft/save-success"; mode: GenerationMode; revision: number; draft: unknown }
  | { type: "draft/save-error"; mode: GenerationMode; revision: number }
  | { type: "settings/replace"; settings: RuntimeSettings }
  | { type: "tasks/replace"; tasks: GenerationTask[] }
  | { type: "tasks/upsert"; task: GenerationTask }
  | { type: "tasks/invalidate-current-snapshots" }
  | { type: "tasks/remove"; id: string }
  | { type: "tasks/clear-terminal" }
  | { type: "tasks/panel"; open: boolean };

export function createInitialDirectorState(): DirectorState {
  return {
    view: "workspace",
    activeMode: "t2v",
    drafts: createInitialDrafts(),
    settings: structuredClone(DEFAULT_SETTINGS),
    tasks: [],
    taskPanelOpen: false,
    draftSync: createDraftSync(false),
  };
}

function createDraftSync(dirty: boolean): Record<GenerationMode, DraftSyncState> {
  return Object.fromEntries(
    MODE_ORDER.map((mode) => [
      mode,
      { revision: 0, dirty, savingRevision: null, status: "idle" },
    ]),
  ) as Record<GenerationMode, DraftSyncState>;
}

function replaceDraft(drafts: ModeDraftMap, draft: ModeDraft): ModeDraftMap {
  switch (draft.mode) {
    case "t2v":
      return { ...drafts, t2v: draft };
    case "i2v":
      return { ...drafts, i2v: draft };
    case "fl2v":
      return { ...drafts, fl2v: draft };
    case "r2v":
      return { ...drafts, r2v: draft };
    case "v2v":
      return { ...drafts, v2v: draft };
    case "rv2v":
      return { ...drafts, rv2v: draft };
  }
}

function normalizeDraft(value: unknown, mode: GenerationMode): ModeDraft | null {
  switch (mode) {
    case "t2v": return normalizeModeDraft(value, "t2v");
    case "i2v": return normalizeModeDraft(value, "i2v");
    case "fl2v": return normalizeModeDraft(value, "fl2v");
    case "r2v": return normalizeModeDraft(value, "r2v");
    case "v2v": return normalizeModeDraft(value, "v2v");
    case "rv2v": return normalizeModeDraft(value, "rv2v");
  }
}

function withChangedDraft(state: DirectorState, draft: ModeDraft): DirectorState {
  const normalized = normalizeDraft(draft, draft.mode);
  if (!normalized) return state;
  const current = state.draftSync[draft.mode];
  return {
    ...state,
    drafts: replaceDraft(state.drafts, normalized),
    draftSync: {
      ...state.draftSync,
      [draft.mode]: {
        ...current,
        revision: current.revision + 1,
        dirty: true,
        status: current.savingRevision === null ? "idle" : "saving",
      },
    },
  };
}

function mutatePlainAssets(
  current: AssetReference[],
  mutation: AssetMutation,
  kind: AssetReference["kind"],
): AssetReference[] {
  if (mutation.type === "remove") {
    return current.filter((asset) => asset.id !== mutation.assetId);
  }
  const added = mutation.assets
    .map((asset) => normalizeAssetReference(asset, kind))
    .filter((asset): asset is AssetReference => asset !== null);
  return added.length ? [added[0]] : current;
}

function mutateSlottedAssets(
  current: SlottedAssetReference[],
  mutation: AssetMutation,
  kind: AssetReference["kind"],
  maxItems: number,
): SlottedAssetReference[] {
  if (mutation.type === "remove") {
    return current.filter((asset) => asset.id !== mutation.assetId);
  }
  const added = mutation.assets
    .map((asset) => normalizeAssetReference(asset, kind))
    .filter((asset): asset is AssetReference => asset !== null);
  return appendToLowestFreeSlots(current, added, maxItems);
}

function mutateDraftAssets(
  draft: ModeDraft,
  shotId: string,
  field: DraftAssetField,
  mutation: AssetMutation,
): ModeDraft | null {
  switch (draft.mode) {
    case "t2v":
      return null;
    case "i2v":
      if (field !== "first_image") return null;
      return {
        ...draft,
        shots: draft.shots.map((shot) => {
          if (shot.id !== shotId) return shot;
          const next = mutatePlainAssets(
            shot.first_image ? [shot.first_image] : [],
            mutation,
            "image",
          );
          return { ...shot, first_image: next[0] ?? null };
        }),
      };
    case "fl2v":
      if (field !== "first_image" && field !== "last_image") return null;
      return {
        ...draft,
        shots: draft.shots.map((shot) => {
          if (shot.id !== shotId) return shot;
          const current = field === "first_image" ? shot.first_image : shot.last_image;
          const next = mutatePlainAssets(current ? [current] : [], mutation, "image");
          return { ...shot, [field]: next[0] ?? null };
        }),
      };
    case "r2v":
      if (!["reference_images", "reference_audios", "reference_videos"].includes(field))
        return null;
      return {
        ...draft,
        shots: draft.shots.map((shot) => {
          if (shot.id !== shotId) return shot;
          if (field === "reference_images")
            return {
              ...shot,
              reference_images: mutateSlottedAssets(
                shot.reference_images,
                mutation,
                "image",
                MINIMAX_H3_REFERENCE_LIMITS.referenceImages,
              ),
            };
          if (field === "reference_audios")
            return {
              ...shot,
              reference_audios: mutateSlottedAssets(
                shot.reference_audios,
                mutation,
                "audio",
                MINIMAX_H3_REFERENCE_LIMITS.referenceAudios,
              ),
            };
          return {
            ...shot,
            reference_videos: mutateSlottedAssets(
              shot.reference_videos,
              mutation,
              "video",
              MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos,
            ),
          };
        }),
      };
    case "v2v":
      if (field !== "source_video") return null;
      return {
        ...draft,
        shots: draft.shots.map((shot) => {
          if (shot.id !== shotId) return shot;
          const next = mutatePlainAssets(
            shot.source_video ? [shot.source_video] : [],
            mutation,
            "video",
          );
          const sourceVideo = next[0] ?? null;
          const range = sourceVideo
            ? fitSourceRangeToVideo(
                sourceVideo,
                shot.source_start_seconds,
                shot.source_duration_seconds,
              )
            : null;
          return { ...shot, source_video: sourceVideo, ...(range ?? {}) };
        }),
      };
    case "rv2v":
      if (!["source_video", "reference_images", "reference_audios"].includes(field))
        return null;
      return {
        ...draft,
        shots: draft.shots.map((shot) => {
          if (shot.id !== shotId) return shot;
          if (field === "source_video") {
            const next = mutatePlainAssets(
              shot.source_video ? [shot.source_video] : [],
              mutation,
              "video",
            );
            const sourceVideo = next[0] ?? null;
            const range = sourceVideo
              ? fitSourceRangeToVideo(
                  sourceVideo,
                  shot.source_start_seconds,
                  shot.source_duration_seconds,
                )
              : null;
            return { ...shot, source_video: sourceVideo, ...(range ?? {}) };
          }
          if (field === "reference_images")
            return {
              ...shot,
              reference_images: mutateSlottedAssets(
                shot.reference_images,
                mutation,
                "image",
                MINIMAX_H3_REFERENCE_LIMITS.referenceImages,
              ),
            };
          return {
            ...shot,
            reference_audios: mutateSlottedAssets(
              shot.reference_audios,
              mutation,
              "audio",
              MINIMAX_H3_REFERENCE_LIMITS.referenceAudios,
            ),
          };
        }),
      };
  }
}

export function directorReducer(state: DirectorState, action: DirectorAction): DirectorState {
  switch (action.type) {
    case "navigate/workspace":
      return { ...state, view: "workspace" };
    case "navigate/mode":
      return { ...state, view: "workspace", activeMode: action.mode };
    case "navigate/settings":
      return { ...state, view: "settings" };
    case "draft/replace":
      return withChangedDraft(state, action.draft);
    case "draft/hydrate": {
      const sync = state.draftSync[action.mode];
      if (sync.dirty || sync.revision > 0 || sync.savingRevision !== null) return state;
      const draft = normalizeDraft(action.draft, action.mode);
      if (!draft) return state;
      return {
        ...state,
        drafts: replaceDraft(state.drafts, draft),
        draftSync: {
          ...state.draftSync,
          [action.mode]: { ...sync, status: "saved" },
        },
      };
    }
    case "draft/assets": {
      const current = state.drafts[action.mode];
      if (!current.shots.some((shot) => shot.id === action.shotId)) return state;
      const draft = mutateDraftAssets(
        current,
        action.shotId,
        action.field,
        action.mutation,
      );
      return draft ? withChangedDraft(state, draft) : state;
    }
    case "draft/save-start": {
      const sync = state.draftSync[action.mode];
      if (sync.revision !== action.revision || sync.savingRevision !== null) return state;
      return {
        ...state,
        draftSync: {
          ...state.draftSync,
          [action.mode]: {
            ...sync,
            savingRevision: action.revision,
            status: "saving",
          },
        },
      };
    }
    case "draft/save-success": {
      const sync = state.draftSync[action.mode];
      if (sync.savingRevision !== action.revision) return state;
      const unchanged = sync.revision === action.revision;
      const serverDraft = unchanged ? normalizeDraft(action.draft, action.mode) : null;
      const confirmed = unchanged && serverDraft !== null;
      return {
        ...state,
        drafts: serverDraft ? replaceDraft(state.drafts, serverDraft) : state.drafts,
        draftSync: {
          ...state.draftSync,
          [action.mode]: {
            ...sync,
            savingRevision: null,
            dirty: !confirmed,
            status: confirmed ? "saved" : unchanged ? "error" : "idle",
          },
        },
      };
    }
    case "draft/save-error": {
      const sync = state.draftSync[action.mode];
      if (sync.savingRevision !== action.revision) return state;
      return {
        ...state,
        draftSync: {
          ...state.draftSync,
          [action.mode]: { ...sync, savingRevision: null, dirty: true, status: "error" },
        },
      };
    }
    case "settings/replace":
      return { ...state, settings: sanitizeRuntimeSettings(action.settings) };
    case "tasks/replace":
      return { ...state, tasks: action.tasks };
    case "tasks/upsert": {
      const exists = state.tasks.some((task) => task.id === action.task.id);
      return {
        ...state,
        tasks: exists
          ? state.tasks.map((task) => (task.id === action.task.id ? action.task : task))
          : [action.task, ...state.tasks],
      };
    }
    case "tasks/invalidate-current-snapshots":
      return {
        ...state,
        tasks: state.tasks.map((task) => ({
          ...task,
          current_project: false,
          segment_results: task.segment_results.map((result) => ({
            ...result,
            current_snapshot: false,
          })),
        })),
      };
    case "tasks/remove":
      return {
        ...state,
        tasks: state.tasks.filter((task) => task.id !== action.id),
      };
    case "tasks/clear-terminal":
      return {
        ...state,
        tasks: state.tasks.filter(
          (task) => !["succeeded", "failed", "cancelled"].includes(task.status),
        ),
      };
    case "tasks/panel":
      return { ...state, taskPanelOpen: action.open };
  }
}

const STORAGE_KEY = "director-web:v1:workspace";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function loadDirectorState(): DirectorState {
  const fallback = createInitialDirectorState();
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return fallback;
    const parsed: unknown = JSON.parse(saved);
    if (!isRecord(parsed)) return fallback;
    const storedDrafts = isRecord(parsed.drafts) ? parsed.drafts : {};
    const storedDirty = isRecord(parsed.draftDirty) ? parsed.draftDirty : {};
    const normalized = {
      t2v: normalizeModeDraft(storedDrafts.t2v, "t2v"),
      i2v: normalizeModeDraft(storedDrafts.i2v, "i2v"),
      fl2v: normalizeModeDraft(storedDrafts.fl2v, "fl2v"),
      r2v: normalizeModeDraft(storedDrafts.r2v, "r2v"),
      v2v: normalizeModeDraft(storedDrafts.v2v, "v2v"),
      rv2v: normalizeModeDraft(storedDrafts.rv2v, "rv2v"),
    };
    const draftSync = createDraftSync(false);
    for (const mode of MODE_ORDER) {
      // Legacy storage did not persist sync state. Treat valid legacy drafts as
      // unsaved so a later backend hydration cannot destroy offline work.
      draftSync[mode].dirty =
        typeof storedDirty[mode] === "boolean"
          ? storedDirty[mode]
          : normalized[mode] !== null;
    }
    const activeMode =
      typeof parsed.activeMode === "string" &&
      MODE_ORDER.includes(parsed.activeMode as GenerationMode)
        ? (parsed.activeMode as GenerationMode)
        : fallback.activeMode;
    return {
      ...fallback,
      // System settings is a transient overlay. Never resurrect it after a
      // reload, especially with a half-edited form that only lived in memory.
      view: "workspace",
      activeMode,
      tasks: [],
      // Runtime settings are server-authoritative. Local storage only protects
      // draft work and must never resurrect an endpoint that was not confirmed
      // by the Director API during this browser session.
      settings: fallback.settings,
      drafts: {
        t2v: normalized.t2v ?? fallback.drafts.t2v,
        i2v: normalized.i2v ?? fallback.drafts.i2v,
        fl2v: normalized.fl2v ?? fallback.drafts.fl2v,
        r2v: normalized.r2v ?? fallback.drafts.r2v,
        v2v: normalized.v2v ?? fallback.drafts.v2v,
        rv2v: normalized.rv2v ?? fallback.drafts.rv2v,
      },
      taskPanelOpen:
        parsed.layoutVersion === 2 && typeof parsed.taskPanelOpen === "boolean"
          ? parsed.taskPanelOpen
          : fallback.taskPanelOpen,
      draftSync,
    };
  } catch {
    return fallback;
  }
}

export function saveDirectorState(state: DirectorState): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        activeMode: state.activeMode,
        drafts: state.drafts,
        draftDirty: Object.fromEntries(
          MODE_ORDER.map((mode) => [mode, state.draftSync[mode].dirty]),
        ),
        layoutVersion: 2,
        taskPanelOpen: state.taskPanelOpen,
      }),
    );
  } catch {
    // Storage may be unavailable in privacy mode; in-memory state remains usable.
  }
}
