import {
  normalizeTimelineProject,
  type TimelineEditorState,
  type TimelineProject,
  type TimelineSegment,
} from "../domain/timelineProject";
import {
  TIMELINE_PATCH_SCHEMA_VERSION,
  applyTimelinePatches,
  createTimelinePatchPair,
  timelineSerializedBytes,
  timelineValuesEqual,
  type TimelinePatch,
} from "./timelinePatches";

export const MIN_TIMELINE_HISTORY_CAPACITY = 50;
export const MAX_TIMELINE_HISTORY_CAPACITY = 100;
export const DEFAULT_TIMELINE_HISTORY_CAPACITY = MAX_TIMELINE_HISTORY_CAPACITY;
export const DEFAULT_TIMELINE_HISTORY_COALESCE_WINDOW_MS = 800;
export const MAX_TIMELINE_HISTORY_BYTE_BUDGET = 16 * 1024 * 1024;
export const DEFAULT_TIMELINE_HISTORY_BYTE_BUDGET = MAX_TIMELINE_HISTORY_BYTE_BUDGET;
export const TIMELINE_HISTORY_CHECKPOINT_INTERVAL = 20;
export const TIMELINE_HISTORY_SCHEMA_VERSION = TIMELINE_PATCH_SCHEMA_VERSION;
export const TIMELINE_HISTORY_ENVELOPE_FORMAT = "director-timeline-history";
export const TIMELINE_HISTORY_ENVELOPE_VERSION = 1;

export interface TimelineTextEditingContext {
  field_key: string;
  start: number;
  end: number;
  direction: "forward" | "backward" | "none";
}

/**
 * Selection is editor context rather than persisted project data. Structural
 * edits opt into restoring it; text edits can carry caret state while keeping
 * a selection the user changed after the edit.
 */
export interface TimelineHistoryContext {
  selected_segment_ids: string[];
  active_segment_id: string | null;
  selection_anchor_id: string | null;
  /** Undefined retains the phase-one structural restore behaviour. */
  restore_segment_selection?: boolean;
  text_editing?: TimelineTextEditingContext;
}

export interface TimelineHistorySnapshot {
  project: TimelineProject;
  context?: TimelineHistoryContext;
}

export interface TimelineHistoryEntry {
  id: string;
  label: string;
  timestamp: number;
  affectedSegmentIds: readonly string[];
  byteSize: number;
  schemaVersion: number;
  forward: readonly TimelinePatch[];
  inverse: readonly TimelinePatch[];
  beforeContext?: TimelineHistoryContext;
  afterContext?: TimelineHistoryContext;
  mergeKey?: string;
}

export interface TimelineHistoryCheckpoint {
  /** Absolute cursor position; retained positions do not shift when old entries are pruned. */
  position: number;
  project: TimelineProject;
  byteSize: number;
}

export interface TimelineHistoryState {
  capacity: number;
  byteBudget: number;
  totalBytes: number;
  /** Number of entries pruned before the first retained transition. */
  startIndex: number;
  nextEntryId: number;
  past: readonly TimelineHistoryEntry[];
  /** Stack order: the next redo entry is the last item. */
  future: readonly TimelineHistoryEntry[];
  checkpoints: readonly TimelineHistoryCheckpoint[];
  /** One current snapshot is retained so the legacy undo/redo API needs no project argument. */
  head: TimelineProject | null;
  coalescing: TimelineHistoryCoalescing | null;
}

export interface SerializedTimelineHistoryPayload {
  capacity: number;
  byteBudget: number;
  totalBytes: number;
  startIndex: number;
  nextEntryId: number;
  cursor: number;
  past: readonly TimelineHistoryEntry[];
  future: readonly TimelineHistoryEntry[];
  checkpoints: readonly TimelineHistoryCheckpoint[];
  head: TimelineProject | null;
  coalescing: TimelineHistoryCoalescing | null;
}

export interface SerializedTimelineHistoryEnvelope {
  format: typeof TIMELINE_HISTORY_ENVELOPE_FORMAT;
  version: typeof TIMELINE_HISTORY_ENVELOPE_VERSION;
  schemaVersion: number;
  hash: string;
  payload: SerializedTimelineHistoryPayload;
}

export interface DeserializeTimelineHistoryOptions {
  expectedHead?: TimelineProject;
  expectedSchema?: number;
}

export interface TimelineHistoryCoalescing {
  mergeKey: string;
  lastRecordedAt: number;
}

export interface TimelineHistoryChange {
  label: string;
  before: TimelineProject;
  after: TimelineProject;
  beforeContext?: TimelineHistoryContext;
  afterContext?: TimelineHistoryContext;
  /** Consecutive changes with the same stable field key are one user intent. */
  mergeKey?: string;
  /** Injectable monotonic-ish clock value for deterministic tests. */
  now?: number;
  coalesceWindowMs?: number;
}

export interface TimelineHistoryReplay {
  history: TimelineHistoryState;
  snapshot: TimelineHistorySnapshot;
  label: string;
  entryId: string;
  cursor: number;
}

function assertTimelineHistoryCapacity(capacity: number): void {
  if (
    !Number.isInteger(capacity) ||
    capacity < MIN_TIMELINE_HISTORY_CAPACITY ||
    capacity > MAX_TIMELINE_HISTORY_CAPACITY
  ) {
    throw new RangeError(
      `Timeline history capacity must be an integer from ${MIN_TIMELINE_HISTORY_CAPACITY} to ${MAX_TIMELINE_HISTORY_CAPACITY}.`,
    );
  }
}

function assertTimelineHistoryByteBudget(byteBudget: number): void {
  if (
    !Number.isSafeInteger(byteBudget) ||
    byteBudget <= 0 ||
    byteBudget > MAX_TIMELINE_HISTORY_BYTE_BUDGET
  ) {
    throw new RangeError(
      `Timeline history byte budget must be a positive safe integer no greater than ${MAX_TIMELINE_HISTORY_BYTE_BUDGET}.`,
    );
  }
}

function allocateTimelineHistoryEntryId(
  history: TimelineHistoryState,
): { id: string; nextEntryId: number } {
  // Keep the resulting nextEntryId strictly below MAX_SAFE_INTEGER so the
  // state remains serializable and a later allocation can never round/reuse an ID.
  if (
    !Number.isSafeInteger(history.nextEntryId) ||
    history.nextEntryId < 1 ||
    history.nextEntryId + 1 >= Number.MAX_SAFE_INTEGER
  ) {
    throw new RangeError("Timeline history entry ID space is exhausted.");
  }
  return {
    id: `timeline-history-${history.nextEntryId.toString(36)}`,
    nextEntryId: history.nextEntryId + 1,
  };
}

export function createTimelineHistory(
  capacity = DEFAULT_TIMELINE_HISTORY_CAPACITY,
  byteBudget = DEFAULT_TIMELINE_HISTORY_BYTE_BUDGET,
): TimelineHistoryState {
  assertTimelineHistoryCapacity(capacity);
  assertTimelineHistoryByteBudget(byteBudget);
  return {
    capacity,
    byteBudget,
    totalBytes: 0,
    startIndex: 0,
    nextEntryId: 1,
    past: [],
    future: [],
    checkpoints: [],
    head: null,
    coalescing: null,
  };
}

/** Captures structural editor context; DOM caret state is attached by the text transaction owner. */
export function captureTimelineHistoryContext(
  state: Pick<
    TimelineEditorState,
    "selected_segment_ids" | "active_segment_id" | "selection_anchor_id"
  >,
): TimelineHistoryContext {
  return {
    selected_segment_ids: [...state.selected_segment_ids],
    active_segment_id: state.active_segment_id,
    selection_anchor_id: state.selection_anchor_id,
    restore_segment_selection: true,
  };
}

function cloneTimelineTextEditingContext(
  context: TimelineTextEditingContext | undefined,
): TimelineTextEditingContext | undefined {
  return context ? { ...context } : undefined;
}

function cloneTimelineHistoryContext(
  context: TimelineHistoryContext | undefined,
): TimelineHistoryContext | undefined {
  if (!context) return undefined;
  const textEditing = cloneTimelineTextEditingContext(context.text_editing);
  return {
    selected_segment_ids: [...context.selected_segment_ids],
    active_segment_id: context.active_segment_id,
    selection_anchor_id: context.selection_anchor_id,
    // A standalone text replay restores its caret only. Only a synthesized
    // multi-entry jump may explicitly combine it with structural selection.
    ...(textEditing
      ? {
          // Ordinary text entries explicitly opt out of restoring structural
          // selection. A multi-entry jump may intentionally combine the last
          // effective structural context with the last text caret context.
          restore_segment_selection: context.restore_segment_selection === true,
          text_editing: textEditing,
        }
      : context.restore_segment_selection === undefined
        ? {}
        : { restore_segment_selection: context.restore_segment_selection }),
  };
}

function cloneTimelineHistorySnapshot(
  project: TimelineProject,
  context: TimelineHistoryContext | undefined,
): TimelineHistorySnapshot {
  return {
    project: structuredClone(project),
    context: cloneTimelineHistoryContext(context),
  };
}

export function timelineProjectsEqual(left: TimelineProject, right: TimelineProject): boolean {
  return timelineValuesEqual(left, right);
}

function segmentIdentityOrder(before: TimelineProject, after: TimelineProject): string[] {
  return [
    ...before.segments.map((segment) => segment.id),
    ...after.segments
      .map((segment) => segment.id)
      .filter((id) => !before.segments.some((segment) => segment.id === id)),
  ];
}

function affectedSegmentIds(before: TimelineProject, after: TimelineProject): string[] {
  const beforeById = new Map(before.segments.map((segment, index) => [segment.id, { segment, index }]));
  const afterById = new Map(after.segments.map((segment, index) => [segment.id, { segment, index }]));
  return segmentIdentityOrder(before, after).filter((id) => {
    const left = beforeById.get(id);
    const right = afterById.get(id);
    return !left || !right || left.index !== right.index ||
      !timelineValuesEqual(left.segment, right.segment);
  });
}

type TimelineHistoryEntryWithoutSize = Omit<TimelineHistoryEntry, "byteSize">;

function entryWithSize(entry: TimelineHistoryEntryWithoutSize): TimelineHistoryEntry {
  return { ...entry, byteSize: timelineSerializedBytes(entry) };
}

function createTimelineHistoryEntry(
  id: string,
  label: string,
  timestamp: number,
  before: TimelineProject,
  after: TimelineProject,
  beforeContext: TimelineHistoryContext | undefined,
  afterContext: TimelineHistoryContext | undefined,
  mergeKey: string | undefined,
): TimelineHistoryEntry {
  const patches = createTimelinePatchPair(before, after);
  return entryWithSize({
    id,
    label,
    timestamp,
    affectedSegmentIds: affectedSegmentIds(before, after),
    schemaVersion: TIMELINE_HISTORY_SCHEMA_VERSION,
    forward: patches.forward,
    inverse: patches.inverse,
    ...(beforeContext === undefined
      ? {}
      : { beforeContext: cloneTimelineHistoryContext(beforeContext) }),
    ...(afterContext === undefined
      ? {}
      : { afterContext: cloneTimelineHistoryContext(afterContext) }),
    ...(mergeKey === undefined ? {} : { mergeKey }),
  });
}

function replaceTimelineHistoryEntryPatches(
  entry: TimelineHistoryEntry,
  before: TimelineProject,
  after: TimelineProject,
  updates: {
    label?: string;
    beforeContext?: TimelineHistoryContext;
    afterContext?: TimelineHistoryContext;
  } = {},
): TimelineHistoryEntry {
  const patches = createTimelinePatchPair(before, after);
  const hasBeforeContext = Object.prototype.hasOwnProperty.call(updates, "beforeContext");
  const hasAfterContext = Object.prototype.hasOwnProperty.call(updates, "afterContext");
  return entryWithSize({
    id: entry.id,
    label: updates.label ?? entry.label,
    timestamp: entry.timestamp,
    affectedSegmentIds: affectedSegmentIds(before, after),
    schemaVersion: TIMELINE_HISTORY_SCHEMA_VERSION,
    forward: patches.forward,
    inverse: patches.inverse,
    ...((hasBeforeContext ? updates.beforeContext : entry.beforeContext) === undefined
      ? {}
      : {
          beforeContext: cloneTimelineHistoryContext(
            hasBeforeContext ? updates.beforeContext : entry.beforeContext,
          ),
        }),
    ...((hasAfterContext ? updates.afterContext : entry.afterContext) === undefined
      ? {}
      : {
          afterContext: cloneTimelineHistoryContext(
            hasAfterContext ? updates.afterContext : entry.afterContext,
          ),
        }),
    ...(entry.mergeKey === undefined ? {} : { mergeKey: entry.mergeKey }),
  });
}

function createTimelineHistoryCheckpoint(
  position: number,
  project: TimelineProject,
): TimelineHistoryCheckpoint {
  const captured = structuredClone(project);
  return {
    position,
    project: captured,
    byteSize: timelineSerializedBytes({ position, project: captured }),
  };
}

function orderedEntries(history: TimelineHistoryState): TimelineHistoryEntry[] {
  return [...history.past, ...[...history.future].reverse()];
}

function historyStorageBytes(
  entries: readonly TimelineHistoryEntry[],
  checkpoints: readonly TimelineHistoryCheckpoint[],
): number {
  return entries.reduce((total, entry) => total + entry.byteSize, 0) +
    checkpoints.reduce((total, checkpoint) => total + checkpoint.byteSize, 0);
}

function replaceCheckpoint(
  checkpoints: readonly TimelineHistoryCheckpoint[],
  position: number,
  project: TimelineProject,
): TimelineHistoryCheckpoint[] {
  const replacement = createTimelineHistoryCheckpoint(position, project);
  const retained = checkpoints.filter((checkpoint) => checkpoint.position !== position);
  return [...retained, replacement].sort((left, right) => left.position - right.position);
}

function projectAtRelativeCursor(
  history: TimelineHistoryState,
  targetCursor: number,
): TimelineProject {
  const entries = orderedEntries(history);
  if (!Number.isInteger(targetCursor) || targetCursor < 0 || targetCursor > entries.length) {
    throw new RangeError("Timeline history cursor is out of bounds.");
  }
  const targetPosition = history.startIndex + targetCursor;
  const currentPosition = history.startIndex + history.past.length;
  const candidates: Array<{ position: number; project: TimelineProject }> = history.checkpoints
    .map((checkpoint) => ({ position: checkpoint.position, project: checkpoint.project }));
  if (history.head) candidates.push({ position: currentPosition, project: history.head });
  const origin = candidates.reduce<{ position: number; project: TimelineProject } | null>(
    (closest, candidate) => !closest ||
      Math.abs(candidate.position - targetPosition) < Math.abs(closest.position - targetPosition)
      ? candidate
      : closest,
    null,
  );
  if (!origin) throw new Error("Timeline history has no reconstruction checkpoint.");

  let project = structuredClone(origin.project);
  if (origin.position < targetPosition) {
    for (let position = origin.position; position < targetPosition; position += 1) {
      const entry = entries[position - history.startIndex];
      if (!entry) throw new Error("Timeline history forward chain is incomplete.");
      project = applyTimelinePatches(project, entry.forward);
    }
  } else if (origin.position > targetPosition) {
    for (let position = origin.position - 1; position >= targetPosition; position -= 1) {
      const entry = entries[position - history.startIndex];
      if (!entry) throw new Error("Timeline history inverse chain is incomplete.");
      project = applyTimelinePatches(project, entry.inverse);
    }
  }
  return project;
}

function emptyTimelineHistoryLike(history: TimelineHistoryState): TimelineHistoryState {
  return {
    ...history,
    totalBytes: 0,
    startIndex: 0,
    past: [],
    future: [],
    checkpoints: [],
    head: null,
    coalescing: null,
  };
}

/** Applies both the entry-count and exact UTF-8 history-storage budgets. */
function pruneTimelineHistory(history: TimelineHistoryState): TimelineHistoryState {
  if (history.future.length) throw new Error("History pruning requires a committed branch.");
  const originalPast = [...history.past];
  const originalLength = originalPast.length;
  let removeCount = Math.max(0, originalLength - history.capacity);

  for (;;) {
    const retainedPast = originalPast.slice(removeCount);
    if (!retainedPast.length) {
      return {
        ...history,
        totalBytes: 0,
        startIndex: history.startIndex + originalLength,
        past: [],
        checkpoints: [],
        head: null,
        coalescing: null,
      };
    }

    const newStartIndex = history.startIndex + removeCount;
    const baseProject = removeCount === 0
      ? history.checkpoints.find((checkpoint) => checkpoint.position === newStartIndex)?.project
      : projectAtRelativeCursor(history, removeCount);
    if (!baseProject) throw new Error("Timeline history base checkpoint is missing.");
    const retainedCheckpoints = history.checkpoints.filter((checkpoint) =>
      checkpoint.position >= newStartIndex &&
      checkpoint.position <= history.startIndex + originalLength);
    const checkpoints = replaceCheckpoint(retainedCheckpoints, newStartIndex, baseProject);
    const totalBytes = historyStorageBytes(retainedPast, checkpoints);
    if (totalBytes <= history.byteBudget) {
      return {
        ...history,
        totalBytes,
        startIndex: newStartIndex,
        past: retainedPast,
        checkpoints,
      };
    }
    removeCount += 1;
  }
}

function recordFreshTimelineHistory(
  history: TimelineHistoryState,
  change: TimelineHistoryChange,
  timestamp: number,
): TimelineHistoryState {
  const allocation = allocateTimelineHistoryEntryId(history);
  const entry = createTimelineHistoryEntry(
    allocation.id,
    change.label,
    timestamp,
    change.before,
    change.after,
    change.beforeContext,
    change.afterContext,
    change.mergeKey,
  );
  const checkpoints = [createTimelineHistoryCheckpoint(0, change.before)];
  return pruneTimelineHistory({
    ...emptyTimelineHistoryLike(history),
    nextEntryId: allocation.nextEntryId,
    past: [entry],
    checkpoints,
    head: structuredClone(change.after),
    coalescing: change.mergeKey === undefined
      ? null
      : { mergeKey: change.mergeKey, lastRecordedAt: timestamp },
    totalBytes: historyStorageBytes([entry], checkpoints),
  });
}

/**
 * Records one project edit. Entries retain patches and sparse checkpoints,
 * never a before/after project snapshot pair. Context-only changes are ignored.
 */
export function recordTimelineHistory(
  history: TimelineHistoryState,
  change: TimelineHistoryChange,
): TimelineHistoryState {
  if (timelineProjectsEqual(change.before, change.after)) return history;

  const timestamp = change.now ?? Date.now();
  const coalesceWindowMs = change.coalesceWindowMs ??
    DEFAULT_TIMELINE_HISTORY_COALESCE_WINDOW_MS;
  if (!Number.isFinite(timestamp)) throw new RangeError("Timeline history time must be finite.");
  if (!Number.isFinite(coalesceWindowMs) || coalesceWindowMs < 0) {
    throw new RangeError("Timeline history coalesce window must be a non-negative number.");
  }

  // A broken caller chain cannot be represented safely by relative patches.
  // Start a fresh base instead of letting a later undo apply to the wrong head.
  if (history.head && !timelineProjectsEqual(history.head, change.before)) {
    return recordFreshTimelineHistory(history, change, timestamp);
  }
  if (!history.head && (history.past.length || history.future.length)) {
    return recordFreshTimelineHistory(history, change, timestamp);
  }

  const mergeKey = change.mergeKey;
  const last = history.past.at(-1);
  const elapsed = history.coalescing
    ? timestamp - history.coalescing.lastRecordedAt
    : Infinity;
  const canCoalesce = mergeKey !== undefined &&
    history.future.length === 0 &&
    history.coalescing?.mergeKey === mergeKey &&
    elapsed >= 0 &&
    elapsed <= coalesceWindowMs &&
    last?.mergeKey === mergeKey;

  if (canCoalesce && last && mergeKey !== undefined) {
    const originalBefore = applyTimelinePatches(change.before, last.inverse);
    const preceding = history.past.slice(0, -1);
    const collapsed = timelineProjectsEqual(originalBefore, change.after);
    if (collapsed) {
      const cursorPosition = history.startIndex + preceding.length;
      const checkpoints = preceding.length
        ? history.checkpoints.filter((checkpoint) => checkpoint.position <= cursorPosition)
        : [];
      const totalBytes = historyStorageBytes(preceding, checkpoints);
      return {
        ...history,
        past: preceding,
        future: [],
        checkpoints,
        head: preceding.length ? structuredClone(change.after) : null,
        coalescing: null,
        totalBytes,
      };
    }

    const replacement = replaceTimelineHistoryEntryPatches(
      last,
      originalBefore,
      change.after,
      {
        label: change.label,
        // Preserve the first caret/selection context and update only the final one.
        beforeContext: last.beforeContext,
        afterContext: change.afterContext,
      },
    );
    const past = [...preceding, replacement];
    const cursorPosition = history.startIndex + past.length;
    let checkpoints = history.checkpoints.filter((checkpoint) =>
      checkpoint.position <= cursorPosition);
    if (checkpoints.some((checkpoint) => checkpoint.position === cursorPosition)) {
      checkpoints = replaceCheckpoint(checkpoints, cursorPosition, change.after);
    }
    return pruneTimelineHistory({
      ...history,
      past,
      future: [],
      checkpoints,
      head: structuredClone(change.after),
      coalescing: { mergeKey, lastRecordedAt: timestamp },
      totalBytes: historyStorageBytes(past, checkpoints),
    });
  }

  const branchCursorPosition = history.startIndex + history.past.length;
  let checkpoints = history.checkpoints.filter((checkpoint) =>
    checkpoint.position <= branchCursorPosition);
  if (!history.past.length && !checkpoints.some((checkpoint) =>
    checkpoint.position === branchCursorPosition)) {
    checkpoints = replaceCheckpoint(checkpoints, branchCursorPosition, change.before);
  }
  const allocation = allocateTimelineHistoryEntryId(history);
  const entry = createTimelineHistoryEntry(
    allocation.id,
    change.label,
    timestamp,
    change.before,
    change.after,
    change.beforeContext,
    change.afterContext,
    mergeKey,
  );
  const past = [...history.past, entry];
  const cursorPosition = history.startIndex + past.length;
  if (cursorPosition % TIMELINE_HISTORY_CHECKPOINT_INTERVAL === 0) {
    checkpoints = replaceCheckpoint(checkpoints, cursorPosition, change.after);
  }
  return pruneTimelineHistory({
    ...history,
    nextEntryId: allocation.nextEntryId,
    past,
    future: [],
    checkpoints,
    head: structuredClone(change.after),
    coalescing: mergeKey === undefined
      ? null
      : { mergeKey, lastRecordedAt: timestamp },
    totalBytes: historyStorageBytes(past, checkpoints),
  });
}

/** Ends a typing/drag coalescing session without touching either history stack. */
export function sealTimelineHistoryCoalescing(
  history: TimelineHistoryState,
): TimelineHistoryState {
  return history.coalescing === null ? history : { ...history, coalescing: null };
}

export function undoTimelineHistory(
  history: TimelineHistoryState,
): TimelineHistoryReplay | null {
  const entry = history.past.at(-1);
  if (!entry || !history.head) return null;
  const project = applyTimelinePatches(history.head, entry.inverse);
  const past = history.past.slice(0, -1);
  return {
    history: {
      ...history,
      past,
      future: [...history.future, entry],
      head: structuredClone(project),
      coalescing: null,
    },
    snapshot: cloneTimelineHistorySnapshot(project, entry.beforeContext),
    label: entry.label,
    entryId: entry.id,
    cursor: past.length,
  };
}

export function redoTimelineHistory(
  history: TimelineHistoryState,
): TimelineHistoryReplay | null {
  const entry = history.future.at(-1);
  if (!entry || !history.head) return null;
  const project = applyTimelinePatches(history.head, entry.forward);
  const past = [...history.past, entry];
  return {
    history: {
      ...history,
      past,
      future: history.future.slice(0, -1),
      head: structuredClone(project),
      coalescing: null,
    },
    snapshot: cloneTimelineHistorySnapshot(project, entry.afterContext),
    label: entry.label,
    entryId: entry.id,
    cursor: past.length,
  };
}

export function timelineHistoryCursor(history: TimelineHistoryState): number {
  return history.past.length;
}

export function timelineHistoryLength(history: TimelineHistoryState): number {
  return history.past.length + history.future.length;
}

export function timelineHistoryEntries(
  history: TimelineHistoryState,
): readonly TimelineHistoryEntry[] {
  return orderedEntries(history);
}

function replayContextAcrossEntries(
  entries: readonly TimelineHistoryEntry[],
  currentCursor: number,
  targetCursor: number,
): TimelineHistoryContext | undefined {
  const movingBackward = targetCursor < currentCursor;
  const crossedEntries = movingBackward
    ? entries.slice(targetCursor, currentCursor).reverse()
    : entries.slice(currentCursor, targetCursor);
  let structuralContext: TimelineHistoryContext | undefined;
  let textContext: TimelineHistoryContext | undefined;
  for (const entry of crossedEntries) {
    const context = movingBackward ? entry.beforeContext : entry.afterContext;
    if (!context) continue;
    if (context.restore_segment_selection !== false) structuralContext = context;
    if (context.text_editing) textContext = context;
  }
  if (!structuralContext) return textContext;
  if (!textContext?.text_editing) return structuralContext;
  return {
    selected_segment_ids: [...structuralContext.selected_segment_ids],
    active_segment_id: structuralContext.active_segment_id,
    selection_anchor_id: structuralContext.selection_anchor_id,
    restore_segment_selection: true,
    text_editing: cloneTimelineTextEditingContext(textContext.text_editing),
  };
}

/** Jumps to any retained cursor with one reconstructed snapshot and no intermediate revisions. */
export function jumpTimelineHistory(
  history: TimelineHistoryState,
  targetCursor: number,
): TimelineHistoryReplay | null {
  const entries = orderedEntries(history);
  if (!Number.isInteger(targetCursor) || targetCursor < 0 || targetCursor > entries.length) {
    throw new RangeError("Timeline history cursor is out of bounds.");
  }
  const currentCursor = history.past.length;
  if (targetCursor === currentCursor) return null;
  const project = projectAtRelativeCursor(history, targetCursor);
  const movingBackward = targetCursor < currentCursor;
  const entry = movingBackward ? entries[targetCursor] : entries[targetCursor - 1];
  const context = replayContextAcrossEntries(entries, currentCursor, targetCursor);
  return {
    history: {
      ...history,
      past: entries.slice(0, targetCursor),
      future: entries.slice(targetCursor).reverse(),
      head: structuredClone(project),
      coalescing: null,
    },
    snapshot: cloneTimelineHistorySnapshot(project, context),
    label: entry.label,
    entryId: entry.id,
    cursor: targetCursor,
  };
}

function segmentAssetIdentity(segment: TimelineSegment): unknown {
  if (segment.mode === "fl2va") {
    return {
      mode: segment.mode,
      first: segment.first_image?.id ?? null,
      last: segment.last_image?.id ?? null,
    };
  }
  const slotted = (assets: typeof segment.reference_images) =>
    assets.map((asset) => [asset.slot, asset.id]);
  return {
    mode: segment.mode,
    source: segment.source_video?.id ?? null,
    sourceAudioAsReference: segment.source_audio_as_reference,
    images: slotted(segment.reference_images),
    audios: slotted(segment.reference_audios),
    videos: slotted(segment.reference_videos),
  };
}

/** Stable topology and bound-material identities are required before an ACK can rebase history. */
export function canSafelyRebaseTimelineHistoryHead(
  expected: TimelineProject,
  confirmed: TimelineProject,
): boolean {
  return expected.version === confirmed.version &&
    expected.segments.length === confirmed.segments.length &&
    expected.segments.every((segment, index) => {
      const next = confirmed.segments[index];
      return segment.id === next?.id &&
        timelineValuesEqual(segmentAssetIdentity(segment), segmentAssetIdentity(next));
    });
}

/**
 * Rewrites only the two transitions adjacent to the current cursor. Returns
 * null when the expected head is stale, identities changed, or rebasing would
 * exceed the configured history budget; callers then retain the safe clear fallback.
 */
export function rebaseTimelineHistoryHead(
  history: TimelineHistoryState,
  expected: TimelineProject,
  confirmed: TimelineProject,
): TimelineHistoryState | null {
  if (timelineProjectsEqual(expected, confirmed)) return history;
  if (
    !history.head ||
    !timelineProjectsEqual(history.head, expected) ||
    !canSafelyRebaseTimelineHistoryHead(expected, confirmed)
  ) return null;

  const past = [...history.past];
  const future = [...history.future];
  const previous = past.at(-1);
  if (previous) {
    const before = applyTimelinePatches(expected, previous.inverse);
    if (timelineProjectsEqual(before, confirmed)) return null;
    past[past.length - 1] = replaceTimelineHistoryEntryPatches(
      previous,
      before,
      confirmed,
    );
  }
  const next = future.at(-1);
  if (next) {
    const after = applyTimelinePatches(expected, next.forward);
    if (timelineProjectsEqual(confirmed, after)) return null;
    future[future.length - 1] = replaceTimelineHistoryEntryPatches(
      next,
      confirmed,
      after,
    );
  }
  if (!previous && !next) return history;

  const cursorPosition = history.startIndex + past.length;
  let checkpoints = [...history.checkpoints];
  if (checkpoints.some((checkpoint) => checkpoint.position === cursorPosition)) {
    checkpoints = replaceCheckpoint(checkpoints, cursorPosition, confirmed);
  }
  const entries = [...past, ...[...future].reverse()];
  const totalBytes = historyStorageBytes(entries, checkpoints);
  if (totalBytes > history.byteBudget) return null;
  return {
    ...history,
    past,
    future,
    checkpoints,
    head: structuredClone(confirmed),
    coalescing: null,
    totalBytes,
  };
}

const TIMELINE_HISTORY_HASH_PREFIX = "fnv1a64:";
const TIMELINE_HISTORY_HASH_PATTERN = /^fnv1a64:[0-9a-f]{16}$/;
const TIMELINE_HISTORY_ENTRY_ID_PATTERN = /^timeline-history-([0-9a-z]+)$/;
const MAX_TIMELINE_PATCH_PATH_DEPTH = 16;
const MAX_TIMELINE_PATCHES_PER_DIRECTION = 20_000;
const MAX_TIMELINE_HISTORY_FIELD_KEY_LENGTH = 512;

const TIMELINE_PATCH_ROOT_FIELDS = new Set([
  "version",
  "title",
  "render",
  "sampling",
  "export_mode",
  "segments",
]);

/** Every string that can occur as a property in the current TimelineProject schema. */
const TIMELINE_PATCH_FIELDS = new Set([
  ...TIMELINE_PATCH_ROOT_FIELDS,
  "width",
  "height",
  "fps",
  "fl2va",
  "ref2va",
  "steps",
  "seed",
  "random_seed",
  "sampler",
  "scheduler",
  "shift",
  "audio_shift",
  "id",
  "mode",
  "prompt",
  "duration_seconds",
  "enabled",
  "continuity",
  "overlap_frames",
  "ref_image_size",
  "audio_mode",
  "first_image",
  "last_image",
  "source_video",
  "source_start_seconds",
  "source_duration_seconds",
  "source_audio_as_reference",
  "reference_images",
  "reference_audios",
  "reference_videos",
  "name",
  "subfolder",
  "type",
  "kind",
  "filename",
  "path",
  "preview_url",
  "metadata",
  "slot",
  "duration",
  "native_fps",
  "frame_count",
  "probe_method",
  "has_audio",
]);

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

function hasExactKeys(
  value: unknown,
  required: readonly string[],
  optional: readonly string[] = [],
): value is Record<string, unknown> {
  if (!isPlainRecord(value)) return false;
  const allowed = new Set([...required, ...optional]);
  const keys = Object.keys(value);
  return required.every((key) => Object.prototype.hasOwnProperty.call(value, key)) &&
    keys.every((key) => allowed.has(key));
}

function isJsonSafeValue(
  value: unknown,
  ancestors = new Set<object>(),
  depth = 0,
): boolean {
  if (depth > 32) return false;
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "boolean"
  ) return true;
  if (typeof value === "number") return Number.isFinite(value) && !Object.is(value, -0);
  if (typeof value !== "object" || ancestors.has(value)) return false;

  ancestors.add(value);
  let valid: boolean;
  if (Array.isArray(value)) {
    valid = Object.keys(value).length === value.length &&
      value.every((item) => isJsonSafeValue(item, ancestors, depth + 1));
  } else if (isPlainRecord(value)) {
    const stringKeys = Object.keys(value);
    valid = Reflect.ownKeys(value).length === stringKeys.length &&
      stringKeys.every((key) => {
        const descriptor = Object.getOwnPropertyDescriptor(value, key);
        return descriptor?.enumerable === true && "value" in descriptor &&
          isJsonSafeValue(descriptor.value, ancestors, depth + 1);
      });
  } else {
    valid = false;
  }
  ancestors.delete(value);
  return valid;
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`
  ).join(",")}}`;
}

function cloneJsonValue<T>(value: T): T {
  if (!isJsonSafeValue(value)) throw new TypeError("Timeline history is not JSON-safe.");
  return JSON.parse(JSON.stringify(value)) as T;
}

function timelineHistoryEnvelopeHash(
  schemaVersion: number,
  payload: SerializedTimelineHistoryPayload | unknown,
): string {
  const input = canonicalJson({
    format: TIMELINE_HISTORY_ENVELOPE_FORMAT,
    version: TIMELINE_HISTORY_ENVELOPE_VERSION,
    schemaVersion,
    payload,
  });
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(input)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `${TIMELINE_HISTORY_HASH_PREFIX}${hash.toString(16).padStart(16, "0")}`;
}

function parseExactTimelineProject(value: unknown): TimelineProject | null {
  if (!isPlainRecord(value) || value.version !== 4 || !isJsonSafeValue(value)) return null;
  const normalized = normalizeTimelineProject(value);
  if (!normalized || !timelineValuesEqual(value, normalized)) return null;
  return cloneJsonValue(normalized);
}

function parseTimelinePatchPath(value: unknown): TimelinePatch["path"] | null {
  if (
    !Array.isArray(value) ||
    value.length < 1 ||
    value.length > MAX_TIMELINE_PATCH_PATH_DEPTH ||
    typeof value[0] !== "string" ||
    !TIMELINE_PATCH_ROOT_FIELDS.has(value[0])
  ) return null;
  const path: Array<string | number> = [];
  for (const part of value) {
    if (typeof part === "string") {
      if (!TIMELINE_PATCH_FIELDS.has(part)) return null;
      path.push(part);
    } else if (Number.isSafeInteger(part) && (part as number) >= 0 && (part as number) <= 128) {
      path.push(part as number);
    } else {
      return null;
    }
  }
  return path;
}

function parseTimelinePatch(value: unknown): TimelinePatch | null {
  if (!isPlainRecord(value)) return null;
  if (value.op === "remove") {
    if (!hasExactKeys(value, ["op", "path"])) return null;
    const path = parseTimelinePatchPath(value.path);
    return path ? { op: "remove", path } : null;
  }
  if (value.op === "set") {
    if (!hasExactKeys(value, ["op", "path", "value"]) || !isJsonSafeValue(value.value))
      return null;
    const path = parseTimelinePatchPath(value.path);
    return path ? { op: "set", path, value: cloneJsonValue(value.value) } : null;
  }
  return null;
}

function parseTimelinePatches(value: unknown): TimelinePatch[] | null {
  if (
    !Array.isArray(value) ||
    value.length < 1 ||
    value.length > MAX_TIMELINE_PATCHES_PER_DIRECTION
  ) return null;
  const patches: TimelinePatch[] = [];
  for (const candidate of value) {
    const patch = parseTimelinePatch(candidate);
    if (!patch) return null;
    patches.push(patch);
  }
  return patches;
}

function isNullableBoundedString(value: unknown, maxLength = 128): value is string | null {
  return value === null ||
    (typeof value === "string" && value.length > 0 && value.length <= maxLength);
}

function parseTimelineHistoryContext(value: unknown): TimelineHistoryContext | null {
  if (!hasExactKeys(
    value,
    ["selected_segment_ids", "active_segment_id", "selection_anchor_id"],
    ["restore_segment_selection", "text_editing"],
  )) return null;
  if (
    !Array.isArray(value.selected_segment_ids) ||
    value.selected_segment_ids.length > 128 ||
    !value.selected_segment_ids.every((id) =>
      typeof id === "string" && id.length > 0 && id.length <= 128) ||
    new Set(value.selected_segment_ids).size !== value.selected_segment_ids.length ||
    !isNullableBoundedString(value.active_segment_id) ||
    !isNullableBoundedString(value.selection_anchor_id) ||
    (Object.prototype.hasOwnProperty.call(value, "restore_segment_selection") &&
      typeof value.restore_segment_selection !== "boolean")
  ) return null;

  let textEditing: TimelineTextEditingContext | undefined;
  if (Object.prototype.hasOwnProperty.call(value, "text_editing")) {
    const text = value.text_editing;
    if (
      !hasExactKeys(text, ["field_key", "start", "end", "direction"]) ||
      typeof text.field_key !== "string" ||
      text.field_key.length < 1 ||
      text.field_key.length > MAX_TIMELINE_HISTORY_FIELD_KEY_LENGTH ||
      !Number.isSafeInteger(text.start) ||
      (text.start as number) < 0 ||
      !Number.isSafeInteger(text.end) ||
      (text.end as number) < (text.start as number) ||
      !["forward", "backward", "none"].includes(String(text.direction)) ||
      value.restore_segment_selection !== false
    ) return null;
    textEditing = {
      field_key: text.field_key,
      start: text.start as number,
      end: text.end as number,
      direction: text.direction as TimelineTextEditingContext["direction"],
    };
  }

  return {
    selected_segment_ids: [...value.selected_segment_ids] as string[],
    active_segment_id: value.active_segment_id,
    selection_anchor_id: value.selection_anchor_id,
    ...(Object.prototype.hasOwnProperty.call(value, "restore_segment_selection")
      ? { restore_segment_selection: value.restore_segment_selection as boolean }
      : {}),
    ...(textEditing ? { text_editing: textEditing } : {}),
  };
}

function parseTimelineHistoryEntry(
  value: unknown,
  expectedSchema: number,
): TimelineHistoryEntry | null {
  if (!hasExactKeys(
    value,
    [
      "id",
      "label",
      "timestamp",
      "affectedSegmentIds",
      "byteSize",
      "schemaVersion",
      "forward",
      "inverse",
    ],
    ["beforeContext", "afterContext", "mergeKey"],
  )) return null;
  if (
    typeof value.id !== "string" ||
    !TIMELINE_HISTORY_ENTRY_ID_PATTERN.test(value.id) ||
    typeof value.label !== "string" ||
    value.label.length < 1 ||
    value.label.length > 256 ||
    typeof value.timestamp !== "number" ||
    !Number.isFinite(value.timestamp) ||
    !Number.isSafeInteger(value.byteSize) ||
    (value.byteSize as number) <= 0 ||
    value.schemaVersion !== expectedSchema ||
    !Array.isArray(value.affectedSegmentIds) ||
    value.affectedSegmentIds.length > 128 ||
    !value.affectedSegmentIds.every((id) =>
      typeof id === "string" && id.length > 0 && id.length <= 128) ||
    new Set(value.affectedSegmentIds).size !== value.affectedSegmentIds.length ||
    (Object.prototype.hasOwnProperty.call(value, "mergeKey") &&
      (typeof value.mergeKey !== "string" ||
        value.mergeKey.length < 1 ||
        value.mergeKey.length > MAX_TIMELINE_HISTORY_FIELD_KEY_LENGTH))
  ) return null;

  const forward = parseTimelinePatches(value.forward);
  const inverse = parseTimelinePatches(value.inverse);
  if (!forward || !inverse) return null;
  const hasBeforeContext = Object.prototype.hasOwnProperty.call(value, "beforeContext");
  const hasAfterContext = Object.prototype.hasOwnProperty.call(value, "afterContext");
  const beforeContext = hasBeforeContext
    ? parseTimelineHistoryContext(value.beforeContext)
    : undefined;
  const afterContext = hasAfterContext
    ? parseTimelineHistoryContext(value.afterContext)
    : undefined;
  if ((hasBeforeContext && !beforeContext) || (hasAfterContext && !afterContext)) return null;

  const rebuilt = entryWithSize({
    id: value.id,
    label: value.label,
    timestamp: value.timestamp,
    affectedSegmentIds: [...value.affectedSegmentIds] as string[],
    schemaVersion: expectedSchema,
    forward,
    inverse,
    ...(beforeContext ? { beforeContext } : {}),
    ...(afterContext ? { afterContext } : {}),
    ...(typeof value.mergeKey === "string" ? { mergeKey: value.mergeKey } : {}),
  });
  return rebuilt.byteSize === value.byteSize ? rebuilt : null;
}

function parseTimelineHistoryCheckpoint(value: unknown): TimelineHistoryCheckpoint | null {
  if (!hasExactKeys(value, ["position", "project", "byteSize"])) return null;
  if (
    !Number.isSafeInteger(value.position) ||
    (value.position as number) < 0 ||
    !Number.isSafeInteger(value.byteSize) ||
    (value.byteSize as number) <= 0
  ) return null;
  const project = parseExactTimelineProject(value.project);
  if (!project) return null;
  const rebuilt = createTimelineHistoryCheckpoint(value.position as number, project);
  return rebuilt.byteSize === value.byteSize ? rebuilt : null;
}

function parseTimelineHistoryCoalescing(value: unknown): TimelineHistoryCoalescing | undefined {
  if (!hasExactKeys(value, ["mergeKey", "lastRecordedAt"])) return undefined;
  if (
    typeof value.mergeKey !== "string" ||
    value.mergeKey.length < 1 ||
    value.mergeKey.length > MAX_TIMELINE_HISTORY_FIELD_KEY_LENGTH ||
    typeof value.lastRecordedAt !== "number" ||
    !Number.isFinite(value.lastRecordedAt)
  ) return undefined;
  return { mergeKey: value.mergeKey, lastRecordedAt: value.lastRecordedAt };
}

function historyEntryOrdinal(id: string): number | null {
  const suffix = TIMELINE_HISTORY_ENTRY_ID_PATTERN.exec(id)?.[1];
  if (!suffix) return null;
  const ordinal = Number.parseInt(suffix, 36);
  return Number.isSafeInteger(ordinal) && ordinal > 0 ? ordinal : null;
}

function serializableTimelineHistoryEntry(entry: TimelineHistoryEntry): TimelineHistoryEntry {
  const beforeContext = cloneTimelineHistoryContext(entry.beforeContext);
  const afterContext = cloneTimelineHistoryContext(entry.afterContext);
  return entryWithSize({
    id: entry.id,
    label: entry.label,
    timestamp: entry.timestamp,
    affectedSegmentIds: [...entry.affectedSegmentIds],
    schemaVersion: entry.schemaVersion,
    forward: cloneJsonValue(entry.forward),
    inverse: cloneJsonValue(entry.inverse),
    ...(beforeContext ? { beforeContext } : {}),
    ...(afterContext ? { afterContext } : {}),
    ...(entry.mergeKey === undefined ? {} : { mergeKey: entry.mergeKey }),
  });
}

/**
 * Produces a JSON-safe, self-validating persistence envelope. In-memory byte
 * counters are deliberately ignored and rebuilt from the serialized values.
 */
export function serializeTimelineHistory(
  history: TimelineHistoryState,
): SerializedTimelineHistoryEnvelope {
  const past = history.past.map(serializableTimelineHistoryEntry);
  const future = history.future.map(serializableTimelineHistoryEntry);
  const checkpoints = history.checkpoints.map((checkpoint) =>
    createTimelineHistoryCheckpoint(checkpoint.position, cloneJsonValue(checkpoint.project)));
  const payload: SerializedTimelineHistoryPayload = {
    capacity: history.capacity,
    byteBudget: history.byteBudget,
    totalBytes: historyStorageBytes([...past, ...future], checkpoints),
    startIndex: history.startIndex,
    nextEntryId: history.nextEntryId,
    cursor: past.length,
    past,
    future,
    checkpoints,
    head: history.head ? cloneJsonValue(history.head) : null,
    coalescing: history.coalescing ? { ...history.coalescing } : null,
  };
  if (!isJsonSafeValue(payload)) throw new TypeError("Timeline history is not JSON-safe.");
  const envelope: SerializedTimelineHistoryEnvelope = {
    format: TIMELINE_HISTORY_ENVELOPE_FORMAT,
    version: TIMELINE_HISTORY_ENVELOPE_VERSION,
    schemaVersion: TIMELINE_HISTORY_SCHEMA_VERSION,
    hash: timelineHistoryEnvelopeHash(TIMELINE_HISTORY_SCHEMA_VERSION, payload),
    payload,
  };
  const cloned = cloneJsonValue(envelope);
  if (!deserializeTimelineHistory(cloned)) {
    throw new TypeError("Timeline history violates persistence invariants.");
  }
  return cloned;
}

function decodeTimelineHistory(
  value: unknown,
  options: DeserializeTimelineHistoryOptions,
): TimelineHistoryState | null {
  if (!hasExactKeys(value, ["format", "version", "schemaVersion", "hash", "payload"]))
    return null;
  const expectedSchema = options.expectedSchema ?? TIMELINE_HISTORY_SCHEMA_VERSION;
  if (
    expectedSchema !== TIMELINE_HISTORY_SCHEMA_VERSION ||
    value.format !== TIMELINE_HISTORY_ENVELOPE_FORMAT ||
    value.version !== TIMELINE_HISTORY_ENVELOPE_VERSION ||
    value.schemaVersion !== expectedSchema ||
    typeof value.hash !== "string" ||
    !TIMELINE_HISTORY_HASH_PATTERN.test(value.hash) ||
    !isJsonSafeValue(value)
  ) return null;
  if (timelineHistoryEnvelopeHash(expectedSchema, value.payload) !== value.hash) return null;

  const payload = value.payload;
  if (!hasExactKeys(payload, [
    "capacity",
    "byteBudget",
    "totalBytes",
    "startIndex",
    "nextEntryId",
    "cursor",
    "past",
    "future",
    "checkpoints",
    "head",
    "coalescing",
  ])) return null;
  if (
    !Number.isInteger(payload.capacity) ||
    (payload.capacity as number) < MIN_TIMELINE_HISTORY_CAPACITY ||
    (payload.capacity as number) > MAX_TIMELINE_HISTORY_CAPACITY ||
    !Number.isSafeInteger(payload.byteBudget) ||
    (payload.byteBudget as number) <= 0 ||
    (payload.byteBudget as number) > MAX_TIMELINE_HISTORY_BYTE_BUDGET ||
    !Number.isSafeInteger(payload.totalBytes) ||
    (payload.totalBytes as number) < 0 ||
    !Number.isSafeInteger(payload.startIndex) ||
    (payload.startIndex as number) < 0 ||
    !Number.isSafeInteger(payload.nextEntryId) ||
    (payload.nextEntryId as number) < 1 ||
    (payload.nextEntryId as number) >= Number.MAX_SAFE_INTEGER ||
    !Number.isInteger(payload.cursor) ||
    !Array.isArray(payload.past) ||
    !Array.isArray(payload.future) ||
    !Array.isArray(payload.checkpoints)
  ) return null;

  const entryCount = payload.past.length + payload.future.length;
  if (
    entryCount > (payload.capacity as number) ||
    payload.cursor !== payload.past.length ||
    !Number.isSafeInteger((payload.startIndex as number) + entryCount)
  ) return null;

  const past: TimelineHistoryEntry[] = [];
  const future: TimelineHistoryEntry[] = [];
  for (const candidate of payload.past) {
    const entry = parseTimelineHistoryEntry(candidate, expectedSchema);
    if (!entry) return null;
    past.push(entry);
  }
  for (const candidate of payload.future) {
    const entry = parseTimelineHistoryEntry(candidate, expectedSchema);
    if (!entry) return null;
    future.push(entry);
  }
  const entries = [...past, ...[...future].reverse()];
  const entryIds = new Set(entries.map((entry) => entry.id));
  if (entryIds.size !== entries.length || entries.some((entry) => {
    const ordinal = historyEntryOrdinal(entry.id);
    return ordinal === null || ordinal >= (payload.nextEntryId as number);
  })) return null;

  const checkpoints: TimelineHistoryCheckpoint[] = [];
  for (const candidate of payload.checkpoints) {
    const checkpoint = parseTimelineHistoryCheckpoint(candidate);
    if (!checkpoint) return null;
    checkpoints.push(checkpoint);
  }
  const startIndex = payload.startIndex as number;
  const endIndex = startIndex + entryCount;
  const expectedCheckpointPositions: number[] = [];
  if (entryCount > 0) {
    expectedCheckpointPositions.push(startIndex);
    let position = Math.ceil((startIndex + 1) / TIMELINE_HISTORY_CHECKPOINT_INTERVAL) *
      TIMELINE_HISTORY_CHECKPOINT_INTERVAL;
    for (; position <= endIndex; position += TIMELINE_HISTORY_CHECKPOINT_INTERVAL) {
      expectedCheckpointPositions.push(position);
    }
  }
  if (
    checkpoints.length !== expectedCheckpointPositions.length ||
    checkpoints.some((checkpoint, index) =>
      checkpoint.position !== expectedCheckpointPositions[index])
  ) return null;

  const head = payload.head === null ? null : parseExactTimelineProject(payload.head);
  if (payload.head !== null && !head) return null;
  if ((entryCount === 0) !== (head === null)) return null;
  let coalescing: TimelineHistoryCoalescing | null = null;
  if (payload.coalescing !== null) {
    coalescing = parseTimelineHistoryCoalescing(payload.coalescing) ?? null;
    const last = past.at(-1);
    if (
      !coalescing ||
      future.length > 0 ||
      !last ||
      last.mergeKey !== coalescing.mergeKey ||
      coalescing.lastRecordedAt < last.timestamp
    ) return null;
  }

  const totalBytes = historyStorageBytes(entries, checkpoints);
  if (totalBytes !== payload.totalBytes || totalBytes > (payload.byteBudget as number))
    return null;

  if (entries.length) {
    let project = structuredClone(checkpoints[0].project);
    let cursorProject = payload.cursor === 0 ? structuredClone(project) : null;
    const checkpointByPosition = new Map(
      checkpoints.map((checkpoint) => [checkpoint.position, checkpoint.project]),
    );
    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      const before = project;
      const applied = applyTimelinePatches(before, entry.forward);
      const after = parseExactTimelineProject(applied);
      if (!after || timelineProjectsEqual(before, after)) return null;
      const canonicalPatches = createTimelinePatchPair(before, after);
      if (
        !timelineValuesEqual(entry.forward, canonicalPatches.forward) ||
        !timelineValuesEqual(entry.inverse, canonicalPatches.inverse) ||
        !timelineValuesEqual(entry.affectedSegmentIds, affectedSegmentIds(before, after))
      ) return null;
      const restored = applyTimelinePatches(after, entry.inverse);
      if (!timelineProjectsEqual(restored, before)) return null;
      project = after;
      const absolutePosition = startIndex + index + 1;
      const checkpointProject = checkpointByPosition.get(absolutePosition);
      if (checkpointProject && !timelineProjectsEqual(checkpointProject, project)) return null;
      if (index + 1 === payload.cursor) cursorProject = structuredClone(project);
    }
    if (!cursorProject || !head || !timelineProjectsEqual(cursorProject, head)) return null;
  }

  if (options.expectedHead !== undefined) {
    const expectedHead = parseExactTimelineProject(options.expectedHead);
    if (!expectedHead || !head || !timelineProjectsEqual(expectedHead, head)) return null;
  }

  return {
    capacity: payload.capacity as number,
    byteBudget: payload.byteBudget as number,
    totalBytes,
    startIndex,
    nextEntryId: payload.nextEntryId as number,
    past,
    future,
    checkpoints,
    head,
    coalescing,
  };
}

/** Strict fail-closed import boundary for IndexedDB or any other untrusted storage. */
export function deserializeTimelineHistory(
  value: unknown,
  options: DeserializeTimelineHistoryOptions = {},
): TimelineHistoryState | null {
  try {
    return decodeTimelineHistory(value, options);
  } catch {
    return null;
  }
}

export function resetTimelineHistory(
  history: TimelineHistoryState,
): TimelineHistoryState {
  if (
    !history.past.length &&
    !history.future.length &&
    !history.checkpoints.length &&
    history.head === null &&
    history.coalescing === null
  ) return history;
  return emptyTimelineHistoryLike(history);
}

export function canUndoTimelineHistory(history: TimelineHistoryState): boolean {
  return history.past.length > 0;
}

export function canRedoTimelineHistory(history: TimelineHistoryState): boolean {
  return history.future.length > 0;
}

export function timelineHistoryUndoLabel(
  history: TimelineHistoryState,
): string | null {
  return history.past.at(-1)?.label ?? null;
}

export function timelineHistoryRedoLabel(
  history: TimelineHistoryState,
): string | null {
  return history.future.at(-1)?.label ?? null;
}
