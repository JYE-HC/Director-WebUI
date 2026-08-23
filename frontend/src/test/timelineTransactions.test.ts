import type { AssetReference } from "../domain/modes";
import {
  createTimelineEditorState,
  createTimelineSegment,
} from "../domain/timelineProject";
import {
  reduceTimelineTransaction,
  timelineTransactionPolicy,
} from "../state/timelineTransactions";

const sourceVideo: AssetReference = {
  id: "transaction-source-video",
  name: "source.mp4",
  subfolder: "director",
  type: "input",
  kind: "video",
  preview_url: "/api/assets/transaction-source-video/preview",
  metadata: {
    duration: 20,
    width: 1920,
    height: 1080,
    native_fps: 24,
    frame_count: 480,
    probe_method: "ffprobe",
    has_audio: true,
  },
};

describe("timeline transaction policy", () => {
  it("assigns explicit document, structural, text and passive UI policies", () => {
    const state = createTimelineEditorState();
    const segment = state.project.segments[0];

    expect(timelineTransactionPolicy(state, {
      type: "project/patch",
      patch: { title: "新名称" },
    })).toMatchObject({ scope: "document", label: "重命名项目", context: "none" });
    expect(timelineTransactionPolicy(state, {
      type: "segment/replace",
      segment: { ...segment, prompt: "新提示词" },
    })).toMatchObject({
      scope: "document",
      label: "编辑提示词",
      context: "text",
      coalescing: "merge",
      mergeKey: `segment:${segment.id}:prompt`,
    });
    expect(timelineTransactionPolicy(state, {
      type: "segment/delete-selected",
      fallbackId: "test-transaction-fallback",
    })).toMatchObject({ scope: "document", context: "structural" });
    expect(timelineTransactionPolicy(state, {
      type: "feature/set-project",
      featureId: "lora",
      selection: state.project.features.project.lora,
    })).toMatchObject({ scope: "document", label: "修改 LoRA" });
    expect(timelineTransactionPolicy(state, {
      type: "feature/set-segment",
      segmentId: segment.id,
      featureId: "opaque_extension",
      selection: { enabled: true, params: {} },
    })).toMatchObject({ scope: "document", label: "修改片段扩展配置" });
    expect(timelineTransactionPolicy(state, {
      type: "playhead/set",
      seconds: 1,
    })).toMatchObject({
      scope: "ui",
      coalescing: "preserve",
      applyDerivedNormalization: false,
    });
  });

  it("never runs document normalization for passive UI actions", () => {
    const state = createTimelineEditorState();
    state.project = {
      ...state.project,
      segments: [{
        ...createTimelineSegment("ref2va", 1),
        audio_mode: "source",
        source_video: sourceVideo,
        source_start_seconds: 0,
        source_duration_seconds: 269 / 24,
        duration_seconds: 269 / 24,
      }],
    };
    state.selected_segment_ids = [state.project.segments[0].id];
    state.active_segment_id = state.project.segments[0].id;
    state.selection_anchor_id = state.project.segments[0].id;

    const passive = reduceTimelineTransaction(state, {
      type: "playhead/set",
      seconds: 1,
    });
    expect(passive.next.project).toBe(state.project);
    expect(passive.derivedAdjustments).toEqual([]);

    const segment = state.project.segments[0];
    const edited = reduceTimelineTransaction(state, {
      type: "segment/replace",
      segment: { ...segment, prompt: "触发同事务自动适配" },
    });
    expect(edited.derivedAdjustments).toHaveLength(1);
    expect(edited.next.project.segments[0]).toMatchObject({
      source_duration_seconds: 277 / 24,
      duration_seconds: 269 / 24,
    });
  });

  it("authority and replay overrides cannot accidentally run derived document edits", () => {
    const state = createTimelineEditorState();
    const replacement = structuredClone(state.project);
    replacement.title = "服务器权威";
    const authority = reduceTimelineTransaction(
      state,
      { type: "project/replace", project: replacement },
      "authority",
    );
    expect(authority.policy).toMatchObject({
      scope: "authority",
      coalescing: "seal",
      applyDerivedNormalization: false,
    });

    const replay = reduceTimelineTransaction(
      state,
      {
        type: "history/restore",
        project: replacement,
        selected_segment_ids: [],
        active_segment_id: null,
        selection_anchor_id: null,
      },
      "replay",
    );
    expect(replay.policy.scope).toBe("replay");
    expect(replay.next.project.title).toBe("服务器权威");
  });
});
