import { beforeEach, describe, expect, it, vi } from "vitest";
import { directorApi } from "../api/client";
import type { AssetReference } from "../domain/modes";
import {
  createTimelineEditorState,
  createTimelineProject,
  createTimelineSegment,
  DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS,
  loadLocalTimelineWal,
  retainTimelineFeatureOverrides,
  saveLocalTimelineWal,
  timelineEditorReducer,
  type FeatureSelection,
  type TimelineAction,
  type TimelineEditorState,
  type TimelineProject,
  type TimelineSegment,
} from "../domain/timelineProject";
import {
  createTimelineHistory,
  recordTimelineHistory,
  redoTimelineHistory,
  timelineProjectsEqual,
  undoTimelineHistory,
} from "../state/timelineHistory";
import { reduceTimelineTransaction } from "../state/timelineTransactions";

const ACTIVE_DATABASE = {
  active_database_path: "/srv/director/stage6.sqlite3",
};

const SOURCE_VIDEO: AssetReference = {
  id: "source-video",
  name: "source.mp4",
  subfolder: "directordeck",
  type: "input",
  kind: "video",
  metadata: {
    duration: 12,
    native_fps: 24,
    frame_count: 288,
    width: 1920,
    height: 1080,
    probe_method: "ffprobe",
    has_audio: true,
  },
};

function selection(name: string): FeatureSelection {
  return { enabled: true, params: { preset: name } };
}

function editor(
  segments: TimelineSegment[],
  bySegment: TimelineProject["features"]["by_segment"] = {},
  selectedIds: string[] = [segments[0].id],
): TimelineEditorState {
  const state = createTimelineEditorState();
  return {
    ...state,
    project: {
      ...state.project,
      segments,
      features: {
        ...state.project.features,
        by_segment: structuredClone(bySegment),
      },
    },
    selected_segment_ids: [...selectedIds],
    active_segment_id: selectedIds[0] ?? null,
    selection_anchor_id: selectedIds[0] ?? null,
  };
}

function fl2va(id: string, duration = 4): TimelineSegment {
  return {
    ...createTimelineSegment("fl2va", 1, id),
    duration_seconds: duration,
  };
}

function sourceSegment(id = "source"): TimelineSegment {
  return {
    ...createTimelineSegment("ref2va", 1, id),
    duration_seconds: 12,
    source_video: SOURCE_VIDEO,
    source_start_seconds: 0,
    source_duration_seconds: 12,
  };
}

function featureScope(state: TimelineEditorState, id: string) {
  return state.project.features.by_segment[id];
}

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

describe("Stage 6 segment override structural reducer", () => {
  it("delete prunes only the deleted segment override and reorder keeps ID-bound scopes", () => {
    const initial = editor(
      [fl2va("a"), fl2va("b"), fl2va("c")],
      {
        a: { cascade: selection("a") },
        b: { cascade: selection("b") },
        c: { cascade: selection("c") },
      },
      ["b"],
    );
    const deleted = timelineEditorReducer(initial, {
      type: "segment/delete-selected",
      fallbackId: "fallback-not-used",
    });
    expect(deleted.project.segments.map((segment) => segment.id)).toEqual(["a", "c"]);
    expect(deleted.project.features.by_segment).toEqual({
      a: { cascade: selection("a") },
      c: { cascade: selection("c") },
    });

    const reordered = timelineEditorReducer(deleted, {
      type: "segment/move",
      draggedId: "c",
      targetId: "a",
    });
    expect(reordered.project.segments.map((segment) => segment.id)).toEqual(["c", "a"]);
    expect(featureScope(reordered, "a")).toEqual({ cascade: selection("a") });
    expect(featureScope(reordered, "c")).toEqual({ cascade: selection("c") });
  });

  it.each([
    {
      name: "playhead split",
      initial: () => ({ ...editor([fl2va("source")], {
        source: { cascade: selection("split") },
      }), playhead_seconds: 2 }),
      action: {
        type: "segment/split-selected",
        newId: "split-right",
      } satisfies TimelineAction,
      childIds: ["source", "split-right"],
    },
    {
      name: "smart source cuts",
      initial: () => editor([sourceSegment()], {
        source: { cascade: selection("smart") },
      }),
      action: {
        type: "segment/apply-source-cuts",
        id: "source",
        cutFrames: [96, 192],
        frameRate: 24,
        expected: {
          asset_id: SOURCE_VIDEO.id,
          source_start_seconds: 0,
          source_duration_seconds: 12,
          project_fps: 24,
        },
        newIds: ["smart-2", "smart-3"],
      } satisfies TimelineAction,
      childIds: ["source", "smart-2", "smart-3"],
    },
    {
      name: "even source split",
      initial: () => editor([sourceSegment()], {
        source: { cascade: selection("even") },
      }),
      action: {
        type: "segment/split-evenly",
        id: "source",
        pieces: 3,
        newIds: ["even-2", "even-3"],
      } satisfies TimelineAction,
      childIds: ["source", "even-2", "even-3"],
    },
    {
      name: "duplicate",
      initial: () => editor([fl2va("source")], {
        source: { cascade: selection("duplicate") },
      }),
      action: {
        type: "segment/duplicate-selected",
        newIds: ["duplicate"],
      } satisfies TimelineAction,
      childIds: ["source", "duplicate"],
    },
  ])("$name clones the complete source override to every child", ({ initial, action, childIds }) => {
    const before = initial();
    const after = timelineEditorReducer(before, action);
    expect(after).not.toBe(before);
    expect(after.project.segments.map((segment) => segment.id)).toEqual(childIds);
    for (const id of childIds) {
      expect(featureScope(after, id)).toEqual(featureScope(before, "source"));
    }
    if (childIds.length > 1) {
      expect(featureScope(after, childIds[1])).not.toBe(featureScope(after, "source"));
    }
  });

  it("merge preserves equal overrides and blocks the entire edit when overrides differ", () => {
    const equal = editor(
      [fl2va("a", 4), fl2va("b", 4)],
      {
        a: { cascade: selection("same") },
        b: { cascade: selection("same") },
      },
      ["a", "b"],
    );
    const merged = timelineEditorReducer(equal, { type: "segment/merge-selected" });
    expect(merged.project.segments.map((segment) => segment.id)).toEqual(["a"]);
    expect(merged.project.features.by_segment).toEqual({
      a: { cascade: selection("same") },
    });

    const different = editor(
      [fl2va("a", 4), fl2va("b", 4)],
      {
        a: { cascade: selection("one") },
        b: { cascade: selection("two") },
      },
      ["a", "b"],
    );
    expect(timelineEditorReducer(different, { type: "segment/merge-selected" }))
      .toBe(different);
  });

  it("mode switch fails closed until an explicit incompatible-feature cleanup is in the same action", () => {
    const initial = editor([fl2va("a")], {
      a: {
        "fl2va-only": selection("remove-me"),
        shared: selection("keep-me"),
      },
    });
    const blocked = timelineEditorReducer(initial, {
      type: "segment/set-mode",
      ids: ["a"],
      mode: "ref2va",
      compatibleFeatureIds: ["shared"],
    });
    expect(blocked).toBe(initial);

    const approved = timelineEditorReducer(initial, {
      type: "segment/set-mode",
      ids: ["a"],
      mode: "ref2va",
      compatibleFeatureIds: ["shared"],
      clearIncompatibleFeatureIds: ["fl2va-only"],
    });
    expect(approved.project.segments[0].mode).toBe("ref2va");
    expect(featureScope(approved, "a")).toEqual({ shared: selection("keep-me") });
  });

  it("copy config leaves features off by default and clones them only after explicit opt-in", () => {
    const initial = editor(
      [fl2va("source", 6), fl2va("target", 3)],
      {
        source: { cascade: selection("source") },
        target: { cascade: selection("target") },
      },
      ["source", "target"],
    );
    const withoutFeatures = timelineEditorReducer(initial, {
      type: "segment/apply-config",
      sourceId: "source",
      scope: "selected",
      options: DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS,
    });
    expect(withoutFeatures.project.segments[1].duration_seconds).toBe(6);
    expect(featureScope(withoutFeatures, "target"))
      .toEqual({ cascade: selection("target") });

    const withFeatures = timelineEditorReducer(initial, {
      type: "segment/apply-config",
      sourceId: "source",
      scope: "selected",
      options: { ...DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS, features: true },
    });
    expect(featureScope(withFeatures, "target"))
      .toEqual({ cascade: selection("source") });
    expect(featureScope(withFeatures, "target")).not.toBe(featureScope(withFeatures, "source"));
  });

  it("import/save-as projection prunes orphan overrides without changing valid scopes", () => {
    const project = createTimelineProject();
    project.segments = [fl2va("kept")];
    project.features.by_segment = {
      kept: { cascade: selection("kept") },
      orphan: { cascade: selection("orphan") },
    };
    const retained = retainTimelineFeatureOverrides(project);
    expect(retained.features.by_segment).toEqual({
      kept: { cascade: selection("kept") },
    });
    expect(project.features.by_segment.orphan).toBeDefined();
  });
});

describe("Stage 6 structural transaction authority roundtrip", () => {
  it("keeps one structural edit sequence exact across Undo/Redo, WAL, and CAS body", async () => {
    const initial = editor([fl2va("a", 4), fl2va("b", 4)], {
      a: { cascade: selection("a") },
      b: { cascade: selection("b") },
    });
    let state = initial;
    let history = createTimelineHistory();
    const actions: TimelineAction[] = [
      { type: "segment/set-selection", ids: ["a"] },
      { type: "segment/duplicate-selected", newIds: ["a-copy"] },
      { type: "segment/move", draggedId: "a-copy", targetId: "b" },
    ];
    for (const action of actions) {
      const before = state;
      const transaction = reduceTimelineTransaction(state, action);
      state = transaction.next;
      if (transaction.documentChanged) {
        history = recordTimelineHistory(history, {
          label: transaction.policy.label,
          before: before.project,
          after: state.project,
        });
      }
    }
    const finalProject = structuredClone(state.project);
    expect(finalProject.segments.map((segment) => segment.id)).toEqual(["a", "b", "a-copy"]);
    expect(finalProject.features.by_segment["a-copy"])
      .toEqual(finalProject.features.by_segment.a);

    let replay = history;
    while (replay.past.length) replay = undoTimelineHistory(replay)!.history;
    expect(timelineProjectsEqual(replay.head!, initial.project)).toBe(true);
    while (replay.future.length) replay = redoTimelineHistory(replay)!.history;
    expect(timelineProjectsEqual(replay.head!, finalProject)).toBe(true);

    const wal = saveLocalTimelineWal({
      database: ACTIVE_DATABASE,
      project_id: "stage6-project",
      owner_id: "stage6-owner",
      base_server_revision: 7,
      base_project: initial.project,
      pending_project: finalProject,
    });
    expect(wal).not.toBeNull();
    expect(loadLocalTimelineWal(
      ACTIVE_DATABASE,
      "stage6-project",
      "stage6-owner",
    )?.pending_project).toEqual(finalProject);

    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ document: finalProject, revision: 8 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await expect(directorApi.updateProjectTimelineAuthority(
      "stage6-project",
      finalProject,
      7,
    )).resolves.toEqual({ document: finalProject, revision: 8 });
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      document: finalProject,
      expected_revision: 7,
    });
  });
});
