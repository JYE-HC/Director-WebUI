import { SAMPLING_SCHEDULERS, type AssetReference } from "../domain/modes";
import {
  assignAssetToSegment,
  assignAssetsToSegment,
  alignedTimelineSegmentDuration,
  autoFitSourceAudioTiming,
  applyTimelineSegmentConfiguration,
  canMergeSelectedSegments,
  changeSegmentMode,
  clearLocalTimeline,
  clearLocalTimelineWal,
  createTimelineEditorState,
  createTimelineProject,
  createTimelineSegment,
  copyTimelineSegmentConfiguration,
  DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS,
  EMPTY_SIX_SECTION_PROMPT,
  promptSkeleton,
  closestTimelineOutputResolution,
  deriveSegmentRecipe,
  H3_OUTPUT_RESOLUTIONS_16_9,
  inferTimelineOutputAspect,
  insertPromptReferenceToken,
  insertPromptSubjectToken,
  isTimelineOutputResolution,
  LEGACY_TIMELINE_STORAGE_KEY,
  LEGACY_V5_TIMELINE_WAL_STORAGE_KEY,
  LEGACY_V6_TIMELINE_WAL_STORAGE_KEY,
  listLocalTimelineWalBranches,
  loadLocalTimeline,
  loadLocalTimelineWal,
  localTimelineWalStorageKey,
  normalizeTimelineProject,
  orderAssetsByPreference,
  promptSubjectReferences,
  QUARANTINED_LEGACY_V5_TIMELINE_WAL_STORAGE_KEY,
  QUARANTINED_TIMELINE_STORAGE_KEY,
  QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY,
  removeAssetFromSegment,
  reorderSegmentReference,
  resolveLocalTimelineWal,
  discardLocalTimelineWalBranch,
  saveLocalTimeline,
  saveLocalTimelineWal,
  segmentReferenceTag,
  setSourceAudioAsReference,
  sourcePreviewTime,
  sourceTimelineThumbnailTimes,
  splitTimelineSourceSegmentAtCuts,
  splitTimelineSourceSegmentEvenly,
  timelineAssetUsages,
  timelineContinuityBoundaries,
  timelineContinuityRunIssues,
  timelineDuration,
  timelineSegmentAt,
  timelineEditorReducer,
  timelineOutputResolutions,
  timelineProjectDocumentHash,
  TIMELINE_WAL_FORMAT,
  TIMELINE_WAL_VERSION,
  type TimelineProject,
  UNBOUND_TIMELINE_WAL_STORAGE_KEY,
  updateRef2VASourceRange,
  runnableTimelineSegmentIds,
  validateTimelineProject,
} from "../domain/timelineProject";

const image: AssetReference = {
  id: "image-a",
  name: "image.png",
  subfolder: "directordeck",
  type: "input",
  kind: "image",
  preview_url: "/api/assets/image-a/preview",
};

const audio: AssetReference = {
  id: "audio-a",
  name: "voice.wav",
  subfolder: "directordeck",
  type: "input",
  kind: "audio",
};

const video: AssetReference = {
  id: "video-a",
  name: "source.mp4",
  subfolder: "directordeck",
  type: "input",
  kind: "video",
  metadata: {
    duration: 20,
    native_fps: 30,
    frame_count: 600,
    width: 1920,
    height: 1080,
    probe_method: "ffprobe",
    has_audio: true,
  },
};

const ACTIVE_DATABASE = {
  active_database_path: "/srv/director/data/director.sqlite3",
};

function writeTimelineWal(
  baseProject: TimelineProject,
  pendingProject: TimelineProject,
  baseServerRevision = 7,
  projectId = "default",
  database = ACTIVE_DATABASE,
  ownerId?: string,
) {
  return saveLocalTimelineWal({
    database,
    project_id: projectId,
    base_server_revision: baseServerRevision,
    base_project: baseProject,
    pending_project: pendingProject,
    owner_id: ownerId,
  });
}

function readTimelineWalRaw(
  projectId = "default",
  database = ACTIVE_DATABASE,
  ownerId?: string,
): string | null {
  const key = localTimelineWalStorageKey(database, projectId, ownerId);
  return key ? localStorage.getItem(key) : null;
}

describe("统一 timeline domain", () => {
  it("新建片段默认空提示词，按模式生成对应骨架", () => {
    const fl2va = createTimelineSegment("fl2va", 1);
    expect(fl2va.prompt).toBe("");
    expect(createTimelineSegment("ref2va", 1).prompt).toBe("");

    // ref2va → 六段式
    expect(promptSkeleton(createTimelineSegment("ref2va", 1))).toBe(EMPTY_SIX_SECTION_PROMPT);

    // fl2va t2v：三段主体，无图片对齐指令
    const t2v = promptSkeleton(fl2va);
    expect(t2v).toContain("integrated_multimodal_description:");
    expect(t2v).toContain("overall_soundscape:");
    expect(t2v).toContain("non_diegetic_music:");
    expect(t2v).not.toContain("subject_definitions:");
    expect(t2v).not.toContain("fully referenced");

    // fl2va i2v：仅首图 → 首帧对齐指令 + <Picture 1>
    const i2v = promptSkeleton({ ...fl2va, first_image: image });
    expect(i2v).toContain("<Picture 1>");
    expect(i2v).toContain("fully referenced");
    expect(i2v).toContain("integrated_multimodal_description:");

    // fl2va fl2v：首尾图 → 首尾对齐指令 + Picture 2
    const fl2v = promptSkeleton({
      ...fl2va,
      first_image: image,
      last_image: { ...image, id: "image-b" },
    });
    expect(fl2v).toContain("Picture 1");
    expect(fl2v).toContain("Picture 2");
    expect(fl2v).toContain("integrated_multimodal_description:");

    // fl2va 仅尾图（L2VA）→ 收敛到尾帧指令
    const l2v = promptSkeleton({ ...fl2va, last_image: { ...image, id: "image-b" } });
    expect(l2v).toContain("<Picture 1>");
    expect(l2v).toContain("[Shot N]");

    for (const anchored of [
      { ...fl2va, first_image: image },
      {
        ...fl2va,
        first_image: image,
        last_image: { ...image, id: "image-b" },
      },
      { ...fl2va, last_image: { ...image, id: "image-b" } },
    ]) {
      const withSkeleton = { ...anchored, prompt: promptSkeleton(anchored) };
      expect(validateTimelineProject({
        ...createTimelineProject(),
        segments: [withSkeleton],
      }).join(" ")).not.toContain("未绑定素材");
    }
  });

  it("FL2VA 首尾帧按实际图文编码顺序暴露 Picture 标签", () => {
    const first = { ...image, id: "fl-first", name: "first.png" };
    const last = { ...image, id: "fl-last", name: "last.png" };
    const onlyFirst = {
      ...createTimelineSegment("fl2va", 1),
      prompt: "从 <Picture 1> 开始",
      first_image: first,
    };
    const onlyLast = {
      ...createTimelineSegment("fl2va", 1),
      prompt: "落到 <Picture 1>",
      last_image: last,
    };
    const both = {
      ...createTimelineSegment("fl2va", 1),
      prompt: "从 <Picture 1> 过渡到 <Picture 2>",
      first_image: first,
      last_image: last,
    };

    expect(segmentReferenceTag(onlyFirst, first)).toBe("<Picture 1>");
    expect(segmentReferenceTag(onlyLast, last)).toBe("<Picture 1>");
    expect(segmentReferenceTag(both, first)).toBe("<Picture 1>");
    expect(segmentReferenceTag(both, last)).toBe("<Picture 2>");
    for (const segment of [onlyFirst, onlyLast, both]) {
      expect(validateTimelineProject({
        ...createTimelineProject(),
        segments: [segment],
      }).join(" ")).not.toContain("未绑定素材");
    }
    expect(validateTimelineProject({
      ...createTimelineProject(),
      segments: [{ ...both, prompt: "不存在的 <Picture 3>" }],
    }).join(" ")).toContain("<Picture 3>");
  });

  it("首次添加源视频会把完整探测时长放入时间线，而不是截成默认五秒", () => {
    const emptyRef = createTimelineSegment("ref2va", 1);
    const bound = assignAssetToSegment(emptyRef, video, "source_video");
    expect(bound).toMatchObject({
      mode: "ref2va",
      title: "source",
      duration_seconds: 20,
      source_start_seconds: 0,
      source_duration_seconds: 20,
      source_video: expect.objectContaining({ id: video.id }),
    });

    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, {
      type: "segment/insert-video",
      asset: video,
      anchorId: state.project.segments[0].id,
      position: "after",
      id: "test-inserted-video-segment",
    });
    expect(state.project.segments[1]).toMatchObject({
      mode: "ref2va",
      title: "source",
      duration_seconds: 20,
      source_start_seconds: 0,
      source_duration_seconds: 20,
    });

    const named = assignAssetToSegment(
      { ...emptyRef, title: "我的精剪段", prompt: "保留创作内容" },
      video,
      "source_video",
    );
    expect(named.title).toBe("我的精剪段");

    const fullSourceProject = {
      ...createTimelineProject(),
      segments: [bound],
    };
    // 20 seconds is 480 source frames. H3 would pad the generated take to
    // 481 frames, but the editable full-source geometry and media mapping must
    // still end at the exact 20-second source boundary.
    expect(alignedTimelineSegmentDuration(bound, 24)).toBeCloseTo(481 / 24);
    expect(timelineDuration(fullSourceProject)).toBe(20);
    if (bound.mode !== "ref2va") throw new Error("fixture must remain Ref2VA");
    expect(sourcePreviewTime(bound, 10, 24)).toBe(10);
    expect(sourcePreviewTime(bound, 20, 24)).toBe(20);

    const cropped = {
      ...bound,
      duration_seconds: 10,
      source_start_seconds: 2,
      source_duration_seconds: 10,
    };
    expect(timelineDuration({ ...fullSourceProject, segments: [cropped] })).toBeCloseTo(
      alignedTimelineSegmentDuration(cropped, 24),
    );
  });

  it("批量插入视频一次保持输入顺序，并原子维护稳定锚点、选择与焦点", () => {
    const state = createTimelineEditorState();
    const first = state.project.segments[0];
    const tail = createTimelineSegment("fl2va", 2);
    state.project.segments = [first, tail];
    state.selected_segment_ids = [tail.id];
    state.active_segment_id = tail.id;
    state.selection_anchor_id = tail.id;
    const videos = [
      { ...video, id: "batch-video-1", name: "第一段.mp4" },
      { ...video, id: "batch-video-2", name: "第二段.mp4" },
      { ...video, id: "batch-video-3", name: "第三段.mp4" },
    ];

    const next = timelineEditorReducer(state, {
      type: "segment/insert-videos",
      assets: videos,
      anchorId: first.id,
      position: "after",
      ids: ["test-batch-segment-1", "test-batch-segment-2", "test-batch-segment-3"],
    });
    const inserted = next.project.segments.slice(1, 4);

    expect(next.project.segments.map((segment) =>
      segment.mode === "ref2va" ? segment.source_video?.id : segment.id,
    )).toEqual([first.id, ...videos.map((asset) => asset.id), tail.id]);
    expect(inserted.map((segment) => segment.title)).toEqual(["第一段", "第二段", "第三段"]);
    expect(next.selected_segment_ids).toEqual([
      ...inserted.map((segment) => segment.id),
      tail.id,
    ]);
    expect(next.active_segment_id).toBe(inserted[0].id);
    expect(next.selection_anchor_id).toBe(inserted[0].id);
    expect(state.project.segments).toEqual([first, tail]);
  });

  it("批量插入只消费视频并在 reducer 内守住 128 段上限", () => {
    const state = createTimelineEditorState();
    state.project.segments = Array.from({ length: 127 }, (_, index) => ({
      ...createTimelineSegment("fl2va", index + 1),
      id: `existing-segment-${index}`,
    }));
    const anchor = state.project.segments.at(-1)!;
    state.selected_segment_ids = [anchor.id];
    state.active_segment_id = anchor.id;
    state.selection_anchor_id = anchor.id;
    const firstVideo = { ...video, id: "capacity-video-1", name: "可插入.mp4" };
    const overflowVideo = { ...video, id: "capacity-video-2", name: "超额.mp4" };

    const next = timelineEditorReducer(state, {
      type: "segment/insert-videos",
      assets: [image, firstVideo, overflowVideo],
      anchorId: anchor.id,
      position: "after",
      ids: ["test-capacity-segment-1", "test-capacity-segment-2"],
    });

    expect(next.project.segments).toHaveLength(128);
    expect(next.project.segments.at(-1)).toMatchObject({
      mode: "ref2va",
      source_video: { id: firstVideo.id },
    });
    expect(next.project.segments.some((segment) =>
      segment.mode === "ref2va" && segment.source_video?.id === overflowVideo.id,
    )).toBe(false);
    expect(timelineEditorReducer(next, {
      type: "segment/insert-videos",
      assets: [overflowVideo],
      anchorId: next.project.segments.at(-1)!.id,
      ids: ["test-overflow-segment"],
    })).toBe(next);
  });

  it("源范围编辑与更短替换视频保持既有源到输出时长比例", () => {
    const source = {
      ...createTimelineSegment("ref2va", 1),
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 12,
      duration_seconds: 8,
    };
    const resized = updateRef2VASourceRange(source, {
      source_start_seconds: 3,
      source_duration_seconds: 6,
    });
    expect(resized).toMatchObject({
      source_start_seconds: 3,
      source_duration_seconds: 6,
      duration_seconds: 4,
    });

    const shorterVideo = {
      ...video,
      id: "video-shorter",
      metadata: { ...video.metadata!, duration: 8, frame_count: 240 },
    };
    const replaced = assignAssetToSegment(source, shorterVideo, "source_video");
    expect(replaced).toMatchObject({
      source_video: expect.objectContaining({ id: shorterVideo.id }),
      source_start_seconds: 2,
      source_duration_seconds: 6,
      duration_seconds: 4,
    });

    const startOnly = updateRef2VASourceRange(source, {
      source_start_seconds: 4,
      source_duration_seconds: 12,
    });
    expect(startOnly.source_start_seconds).toBe(4);
    expect(startOnly.duration_seconds).toBe(source.duration_seconds);

    for (const invalidRange of [
      { source_start_seconds: 2, source_duration_seconds: 0 },
      { source_start_seconds: -1, source_duration_seconds: 6 },
      { source_start_seconds: Number.NaN, source_duration_seconds: 6 },
      { source_start_seconds: 2, source_duration_seconds: Number.POSITIVE_INFINITY },
    ]) {
      expect(updateRef2VASourceRange(source, invalidRange)).toBe(source);
    }
  });

  it("粒度化命令总是基于 gateway 当前状态合并，不会用旧快照覆盖同批 sibling 字段", () => {
    const initial = createTimelineEditorState();
    const segmentId = initial.project.segments[0].id;

    const withPrompt = timelineEditorReducer(initial, {
      type: "segment/patch-base",
      id: segmentId,
      patch: { prompt: "先完成的提示词" },
    });
    const withContinuity = timelineEditorReducer(withPrompt, {
      type: "segment/set-continuity",
      id: segmentId,
      patch: { enabled: true },
    });
    const withSamplingSteps = timelineEditorReducer(withContinuity, {
      type: "project/update-sampling",
      family: "fl2va",
      patch: { steps: 31 },
    });
    const completed = timelineEditorReducer(withSamplingSteps, {
      type: "project/update-sampling",
      family: "fl2va",
      patch: { shift: 8 },
    });

    expect(completed.project.segments[0]).toMatchObject({
      prompt: "先完成的提示词",
      continuity: { enabled: true },
    });
    expect(completed.project.sampling.fl2va).toMatchObject({
      steps: 31,
      shift: 8,
    });
    expect(initial.project.segments[0].prompt).toBe("");
    expect(initial.project.sampling.fl2va).toMatchObject({ steps: 25, shift: 12 });
  });

  it("保留源音频时重复输入同一源时长不会继承上次自动适配比例", () => {
    const project = createTimelineProject();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      audio_mode: "source" as const,
      source_video: video,
      source_start_seconds: 0,
      source_duration_seconds: 39 / 24,
      duration_seconds: 1,
    };

    const editAndFit = (segment: typeof source) => {
      const edited = updateRef2VASourceRange(segment, {
        source_start_seconds: 0,
        source_duration_seconds: 1,
      });
      return autoFitSourceAudioTiming({ ...project, segments: [edited] })
        .project.segments[0] as typeof source;
    };

    const first = editAndFit(source);
    const second = editAndFit(first);
    expect(first).toMatchObject({
      duration_seconds: 1,
      source_duration_seconds: 39 / 24,
    });
    expect(second).toMatchObject({
      duration_seconds: 1,
      source_duration_seconds: 39 / 24,
    });
    expect(alignedTimelineSegmentDuration(first, 24) * 24).toBe(39);
    expect(alignedTimelineSegmentDuration(second, 24) * 24).toBe(39);
  });

  it("源片按自身区间等距显示关键帧，分割出的长片段也显示多帧", () => {
    const full = {
      ...createTimelineSegment("ref2va", 1),
      source_video: { ...video, preview_url: "/api/assets/video-a/preview" },
      duration_seconds: 20,
      source_duration_seconds: 20,
    };
    expect(sourceTimelineThumbnailTimes(full)).toEqual([2.5, 7.5, 12.5, 17.5]);
    // 分割/裁剪出的 10s 片段也按自身区间等距显示 2 帧
    expect(sourceTimelineThumbnailTimes({
      ...full,
      source_start_seconds: 5,
      source_duration_seconds: 10,
      duration_seconds: 10,
    })).toEqual([7.5, 12.5]);
    // 4 秒的短片段只显示一帧
    expect(sourceTimelineThumbnailTimes({
      ...full,
      source_start_seconds: 5,
      source_duration_seconds: 4,
      duration_seconds: 4,
    })).toEqual([7]);
  });

  it("超长完整源片是可编辑草稿，但生成校验明确要求先分割", () => {
    const longVideo = {
      ...video,
      metadata: { ...video.metadata!, duration: 180, frame_count: 4_320 },
    };
    const project = createTimelineProject();
    project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      prompt: "编辑完整源视频",
      source_video: longVideo,
      duration_seconds: 180,
      source_duration_seconds: 180,
    }];
    const errors = validateTimelineProject(project).join("；");
    expect(errors).toContain("已完整载入源视频");
    expect(errors).toContain("请先用播放头、均分或智能分割");
    expect(errors).not.toContain("不超过 120 秒");
  });

  it("权威素材库替换会清除上一 endpoint 遗留的选择", () => {
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, { type: "assets/replace", assets: [image] });
    state = timelineEditorReducer(state, { type: "assets/select", id: image.id, additive: false });
    expect(state.selected_asset_ids).toEqual([image.id]);

    state = timelineEditorReducer(state, { type: "assets/replace", assets: [] });
    expect(state.assets).toEqual([]);
    expect(state.selected_asset_ids).toEqual([]);
  });

  it("素材区间选择一次性替换或追加稳定 ID", () => {
    const second = { ...image, id: "image-second", name: "second.png" };
    const third = { ...image, id: "image-third", name: "third.png" };
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, {
      type: "assets/replace",
      assets: [image, second, third],
    });
    state = timelineEditorReducer(state, {
      type: "assets/set-selection",
      ids: [second.id, third.id],
    });
    expect(state.selected_asset_ids).toEqual([second.id, third.id]);

    state = timelineEditorReducer(state, {
      type: "assets/set-selection",
      ids: [image.id, "unknown"],
      additive: true,
    });
    expect(state.selected_asset_ids).toEqual([image.id, second.id, third.id]);
  });

  it("模式切换只保留公共字段并清除所有旧模式专属配置", () => {
    const rv2v = {
      ...createTimelineSegment("ref2va", 1),
      source_video: video,
      reference_images: [{ ...image, slot: 5 }],
      reference_audios: [{ ...audio, slot: 2 }],
      prompt: "保留提示词",
    };
    const fl2va = changeSegmentMode(rv2v, "fl2va");
    expect(fl2va).toMatchObject({ mode: "fl2va", prompt: "保留提示词", first_image: null, last_image: null });
    expect(fl2va).not.toHaveProperty("source_video");
    expect(fl2va).not.toHaveProperty("reference_images");
    expect(fl2va).not.toHaveProperty("reference_audios");
  });

  it("无损迁移 version 1 的六种配方为两族，并仍能推导原配方", () => {
    const common = (id: string, mode: string) => ({
      id,
      mode,
      title: id,
      prompt: `${id} prompt`,
      duration_seconds: 5,
      enabled: true,
    });
    const legacy = {
      ...createTimelineProject(),
      version: 1,
      segments: [
        common("legacy-t2v", "t2v"),
        { ...common("legacy-i2v", "i2v"), first_image: image },
        { ...common("legacy-fl2v", "fl2v"), first_image: null, last_image: image },
        {
          ...common("legacy-r2v", "r2v"),
          reference_images: [{ ...image, slot: 0 }],
          reference_audios: [{ ...audio, slot: 0 }],
          reference_videos: [{ ...video, slot: 0 }],
        },
        {
          ...common("legacy-v2v", "v2v"),
          source_video: video,
          source_start_seconds: 1,
          source_duration_seconds: 4,
          source_audio_as_reference: true,
        },
        {
          ...common("legacy-rv2v", "rv2v"),
          source_video: video,
          source_start_seconds: 2,
          source_duration_seconds: 3,
          source_audio_as_reference: false,
          reference_images: [{ ...image, slot: 0 }],
          reference_audios: [{ ...audio, slot: 0 }],
        },
      ],
    };

    const normalized = normalizeTimelineProject(legacy);
    expect(normalized?.version).toBe(4);
    expect(normalized?.segments.map((segment) => segment.mode)).toEqual([
      "fl2va", "fl2va", "fl2va", "ref2va", "ref2va", "ref2va",
    ]);
    expect(normalized?.segments.map(deriveSegmentRecipe)).toEqual([
      "t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v",
    ]);
    expect(normalized?.segments[2]).toMatchObject({ last_image: image });
    expect(normalized?.segments[3]).toMatchObject({
      reference_images: [expect.objectContaining({ id: image.id, slot: 0 })],
      reference_audios: [expect.objectContaining({ id: audio.id, slot: 0 })],
      reference_videos: [expect.objectContaining({ id: video.id, slot: 0 })],
    });
    expect(normalized?.segments[4]).toMatchObject({
      source_video: expect.objectContaining({ id: video.id }),
      source_start_seconds: 1,
      source_duration_seconds: 4,
      source_audio_as_reference: true,
    });
  });

  it("与后端一致拒绝 version 1 跨配方隐藏字段", () => {
    const legacy = {
      ...createTimelineProject(),
      version: 1,
      segments: [{
        id: "invalid-legacy-t2v",
        mode: "t2v",
        title: "损坏的旧片段",
        prompt: "prompt",
        duration_seconds: 5,
        enabled: true,
        first_image: image,
      }],
    };

    expect(normalizeTimelineProject(legacy)).toBeNull();
  });

  it("Ctrl/Shift 选择使用稳定 ID，重排不改 ID", () => {
    let state = createTimelineEditorState();
    const first = state.project.segments[0].id;
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after", id: "test-select-second" });
    const second = state.project.segments[1].id;
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after", id: "test-select-third" });
    const third = state.project.segments[2].id;
    state = timelineEditorReducer(state, { type: "segment/select", id: first });
    expect(runnableTimelineSegmentIds(state)).toEqual([first]);
    expect(state.active_segment_id).toBe(first);
    state = timelineEditorReducer(state, { type: "segment/select", id: second, additive: true });
    expect(state.selected_segment_ids).toEqual([first, second]);
    expect(state.active_segment_id).toBe(first);
    expect(state.selection_anchor_id).toBe(first);
    state = timelineEditorReducer(state, { type: "segment/select", id: second, additive: true });
    expect(state.selected_segment_ids).toEqual([first]);
    expect(state.active_segment_id).toBe(first);
    state = timelineEditorReducer(state, { type: "segment/select", id: second, additive: true });
    state = timelineEditorReducer(state, { type: "segment/select", id: third, range: true });
    expect(state.selected_segment_ids).toEqual([first, second, third]);
    expect(state.active_segment_id).toBe(first);
    expect(runnableTimelineSegmentIds(state)).toEqual([first, second, third]);
    state = timelineEditorReducer(state, { type: "segment/move", draggedId: third, targetId: first });
    expect(state.project.segments.map((segment) => segment.id)).toEqual([third, first, second]);
    expect(state.selected_segment_ids).toEqual([first, second, third]);
    expect(runnableTimelineSegmentIds(state)).toEqual([third, first, second]);
  });

  it("Shift 范围只覆盖目标所在轨道，不会隐式选中另一轨的片段", () => {
    const first = createTimelineSegment("fl2va", 1);
    const disabledSecond = { ...createTimelineSegment("fl2va", 2), enabled: false };
    const third = createTimelineSegment("ref2va", 3);
    const disabledFourth = { ...createTimelineSegment("ref2va", 4), enabled: false };
    let state = createTimelineEditorState();
    state.project.segments = [first, disabledSecond, third, disabledFourth];
    state.selected_segment_ids = [first.id];
    state.active_segment_id = first.id;
    state.selection_anchor_id = first.id;

    state = timelineEditorReducer(state, { type: "segment/select", id: third.id, range: true });
    expect(state.selected_segment_ids).toEqual([first.id, third.id]);
    expect(state.selected_segment_ids).not.toContain(disabledSecond.id);
    expect(state.active_segment_id).toBe(first.id);
    expect(state.selection_anchor_id).toBe(first.id);
    expect(runnableTimelineSegmentIds(state)).toEqual([first.id, third.id]);

    state = timelineEditorReducer(state, {
      type: "segment/toggle-selection",
      id: disabledSecond.id,
    });
    state = timelineEditorReducer(state, {
      type: "segment/select",
      id: third.id,
      additive: true,
      range: true,
    });
    expect(state.selected_segment_ids).toEqual([first.id, third.id, disabledSecond.id]);
    expect(runnableTimelineSegmentIds(state)).toEqual([first.id, third.id]);

    state = timelineEditorReducer(state, { type: "segment/select", id: disabledSecond.id });
    state = timelineEditorReducer(state, {
      type: "segment/select",
      id: disabledFourth.id,
      range: true,
    });
    expect(state.selected_segment_ids).toEqual([disabledSecond.id, disabledFourth.id]);
    expect(state.selected_segment_ids).not.toContain(third.id);
    expect(state.active_segment_id).toBe(disabledSecond.id);
    expect(runnableTimelineSegmentIds(state)).toEqual([]);

    state = timelineEditorReducer(state, { type: "segment/select", id: third.id, range: true });
    expect(state.selected_segment_ids).toEqual([disabledSecond.id, third.id]);
    expect(state.active_segment_id).toBe(disabledSecond.id);
    expect(state.selection_anchor_id).toBe(third.id);

    state = timelineEditorReducer(state, {
      type: "segment/select",
      id: disabledSecond.id,
      additive: true,
      range: true,
    });
    expect(state.selected_segment_ids).toEqual([disabledSecond.id, third.id]);
    expect(state.active_segment_id).toBe(disabledSecond.id);
    expect(state.selection_anchor_id).toBe(disabledSecond.id);
    expect(runnableTimelineSegmentIds(state)).toEqual([third.id]);
  });

  it("在片段 02 前插空段时分配新的默认编号且保留已有名称", () => {
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after", id: "test-numbering-second" });
    expect(state.project.segments.map((segment) => segment.title)).toEqual(["片段 01", "片段 02"]);

    state = timelineEditorReducer(state, { type: "segment/insert", position: "before", id: "test-numbering-third" });

    expect(state.project.segments.map((segment) => segment.title)).toEqual([
      "片段 01",
      "片段 03",
      "片段 02",
    ]);
  });

  it("乱序默认名称的前后插均按现有最大编号递增", () => {
    const state = createTimelineEditorState();
    const segments = [1, 7, 3, 4].map((number) => createTimelineSegment("fl2va", number));
    const initial = {
      ...state,
      project: { ...state.project, segments },
      selected_segment_ids: [segments[3].id],
      active_segment_id: segments[3].id,
      selection_anchor_id: segments[3].id,
    };

    const before = timelineEditorReducer(initial, { type: "segment/insert", position: "before", id: "test-unordered-before" });
    const after = timelineEditorReducer(initial, { type: "segment/insert", position: "after", id: "test-unordered-after" });

    expect(before.project.segments.map((segment) => segment.title)).toEqual([
      "片段 01",
      "片段 07",
      "片段 03",
      "片段 08",
      "片段 04",
    ]);
    expect(after.project.segments.map((segment) => segment.title)).toEqual([
      "片段 01",
      "片段 07",
      "片段 03",
      "片段 04",
      "片段 08",
    ]);
  });

  it("参考素材槽删除后不重排，新素材填最低空槽", () => {
    let segment = createTimelineSegment("ref2va", 1);
    segment = assignAssetToSegment(segment, image) as typeof segment;
    segment = assignAssetToSegment(segment, { ...image, id: "image-b", name: "b.png" }) as typeof segment;
    expect(segment.reference_images.map((asset) => asset.slot)).toEqual([0, 1]);
    segment.reference_images = segment.reference_images.filter((asset) => asset.slot !== 0);
    segment = assignAssetToSegment(segment, { ...image, id: "image-c", name: "c.png" }) as typeof segment;
    expect(segment.reference_images.map((asset) => [asset.id, asset.slot])).toEqual([
      ["image-b", 1],
      ["image-c", 0],
    ]);
    expect(segmentReferenceTag(segment, segment.reference_images[0])).toBe("<Picture 2>");
  });

  it("片段内部重排参考图片时保持数量并按资产语义重写标签", () => {
    const first = { ...image, id: "picture-first", name: "first.png", slot: 0 };
    const second = { ...image, id: "picture-second", name: "second.png", slot: 1 };
    const third = { ...image, id: "picture-third", name: "third.png", slot: 2 };
    const segment = {
      ...createTimelineSegment("ref2va", 1),
      prompt: "主角 <Picture 1>，服装 <Picture 2>，场景 <Picture 3>",
      reference_images: [first, second, third],
    };

    const reordered = reorderSegmentReference(segment, "image", third.id, first.id);
    expect(reordered.mode).toBe("ref2va");
    if (reordered.mode !== "ref2va") throw new Error("fixture must remain ref2va");
    expect(reordered.reference_images).toHaveLength(3);
    expect(reordered.reference_images.map(({ id, slot }) => ({ id, slot }))).toEqual([
      { id: third.id, slot: 0 },
      { id: first.id, slot: 1 },
      { id: second.id, slot: 2 },
    ]);
    expect(reordered.prompt).toBe("主角 <Picture 2>，服装 <Picture 3>，场景 <Picture 1>");
    expect(validateTimelineProject({ ...createTimelineProject(), segments: [reordered] })).toEqual([]);
  });

  it("源音轨参考占 Audio 1，并整体偏移 RV2V 独立音频标签", () => {
    const rv2v = {
      ...createTimelineSegment("ref2va", 1),
      source_video: video,
      source_audio_as_reference: true,
      reference_audios: [{ ...audio, slot: 0 }],
      prompt: "源 <Audio 1>，对白 <Audio 2>",
    };
    expect(segmentReferenceTag(rv2v, rv2v.reference_audios[0])).toBe("<Audio 2>");
    expect(validateTimelineProject({
      ...createTimelineProject(),
      segments: [rv2v],
    }).join(" ")).not.toContain("未绑定素材");

    const disabled = setSourceAudioAsReference(rv2v, false);
    if (disabled.mode !== "ref2va") throw new Error("fixture must remain rv2v");
    expect(disabled.source_audio_as_reference).toBe(false);
    expect(disabled.prompt).toBe("源 ，对白 <Audio 1>");
    expect(segmentReferenceTag(disabled, disabled.reference_audios[0])).toBe("<Audio 1>");
  });

  it("静音或历史源视频不能启用源音轨参考", () => {
    const historical = structuredClone(video);
    if (!historical.metadata) throw new Error("fixture requires metadata");
    delete (historical.metadata as Partial<typeof historical.metadata>).has_audio;
    const normalized = normalizeTimelineProject({
      ...createTimelineProject(),
      segments: [{
        ...createTimelineSegment("ref2va", 1),
        source_video: historical,
        source_audio_as_reference: true,
      }],
    })!;
    const source = normalized.segments[0];
    expect(source.mode).toBe("ref2va");
    if (source.mode !== "ref2va") throw new Error("fixture must remain v2v");
    expect(source.source_video?.metadata?.has_audio).toBe(false);
    expect(validateTimelineProject(normalized).join(" ")).toContain("没有可用音轨");
    expect(setSourceAudioAsReference({ ...source, source_audio_as_reference: false }, true).source_audio_as_reference).toBe(false);
  });

  it("片段内删除首个参考会压密槽位并同步重写提示词", () => {
    const first = { ...image, id: "picture-first", slot: 0 };
    const second = { ...image, id: "picture-second", slot: 1 };
    const segment = {
      ...createTimelineSegment("ref2va", 1),
      prompt: "先看 <Picture 1>，再看 <Picture 2>",
      reference_images: [first, second],
      reference_audios: [],
      reference_videos: [],
    };

    const removed = removeAssetFromSegment(segment, first.id);
    expect(removed).toMatchObject({
      mode: "ref2va",
      prompt: "先看 ，再看 <Picture 1>",
      reference_images: [{ id: second.id, slot: 0 }],
    });
    const rebound = assignAssetToSegment(removed, { ...image, id: "picture-third" });
    expect(rebound.mode === "ref2va" && rebound.reference_images.map((asset) => asset.slot)).toEqual([0, 1]);
    expect(validateTimelineProject({ ...createTimelineProject(), segments: [rebound] })).toEqual([]);
  });

  it("改写参考标签时完整保留六段式提示词的换行、空行和缩进", () => {
    const first = { ...image, id: "picture-first", slot: 0 };
    const second = { ...image, id: "picture-second", slot: 1 };
    const prompt = [
      "subject_definitions:",
      "<Picture 1> 主体  ",
      "",
      "visual_style:",
      "  保留 <Picture 2>",
      "",
      "camera_and_motion:",
      "固定镜头",
    ].join("\n");
    const segment = {
      ...createTimelineSegment("ref2va", 1),
      prompt,
      reference_images: [first, second],
      reference_audios: [],
      reference_videos: [],
    };

    const removed = removeAssetFromSegment(segment, first.id);
    expect(removed.prompt).toBe([
      "subject_definitions:",
      " 主体  ",
      "",
      "visual_style:",
      "  保留 <Picture 1>",
      "",
      "camera_and_motion:",
      "固定镜头",
    ].join("\n"));

    const withSourceAudio = {
      ...segment,
      source_video: video,
      source_audio_as_reference: true,
      reference_audios: [{ ...audio, slot: 0 }],
      prompt: "subject:\n<Audio 1> 环境声  \n\nsoundscape:\n  <Audio 2> 对白",
    };
    const disabled = setSourceAudioAsReference(withSourceAudio, false);
    expect(disabled.prompt).toBe("subject:\n 环境声  \n\nsoundscape:\n  <Audio 1> 对白");
  });

  it("在提交前拒绝少于五帧的参考视频与源视频裁剪", () => {
    const referenceProject = createTimelineProject();
    referenceProject.segments = [{
      ...createTimelineSegment("ref2va", 1),
      prompt: "使用 <Video 1>",
      reference_images: [],
      reference_audios: [],
      reference_videos: [{
        ...video,
        slot: 0,
        metadata: { ...video.metadata!, native_fps: 24, frame_count: 4, duration: 4 / 24 },
      }],
    }];
    expect(validateTimelineProject(referenceProject).join(" ")).toContain("参考视频至少需要 5 帧");

    const sourceProject = createTimelineProject();
    sourceProject.segments = [{
      ...createTimelineSegment("ref2va", 1),
      prompt: "编辑 <Video 1>",
      source_video: { ...video, metadata: { ...video.metadata!, native_fps: 24 } },
      source_start_seconds: 0,
      source_duration_seconds: 4 / 24,
    }];
    expect(validateTimelineProject(sourceProject).join(" ")).toContain("源视频范围至少需要 5 帧");
  });

  it("Ref2VA 的源视频会占一个视频槽，只允许再绑定两个独立参考视频", () => {
    const project = createTimelineProject();
    const referenceVideos = [0, 1, 2].map((slot) => ({
      ...video,
      id: `reference-video-${slot}`,
      name: `reference-${slot}.mp4`,
      slot,
    }));
    project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      prompt: "用 <Video 1> 重绘，并参考 <Video 2> 与 <Video 3>",
      source_video: video,
      reference_videos: referenceVideos.slice(0, 2),
    }];

    expect(validateTimelineProject(project)).toEqual([]);

    const refSegment = project.segments[0];
    if (refSegment.mode !== "ref2va") throw new Error("fixture must remain Ref2VA");
    project.segments[0] = {
      ...refSegment,
      reference_videos: referenceVideos,
    };
    const errors = validateTimelineProject(project).join(" ");
    expect(errors).toContain("源视频与独立参考视频合计不能超过 3 个");
    expect(errors).toContain("参考素材槽位无效或重复");
  });

  it("播放头拆分 V2V 时按输出比例拆开源区间且无重叠缺口", () => {
    let state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      duration_seconds: 8,
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 12,
      prompt: "重绘",
    };
    state = { ...state, project: { ...state.project, segments: [source] }, playhead_seconds: 2 };
    state = timelineEditorReducer(state, { type: "segment/split-selected", newId: "test-split-right" });
    const [left, right] = state.project.segments;
    expect(left).toMatchObject({ mode: "ref2va", duration_seconds: 2, source_start_seconds: 2, source_duration_seconds: 3 });
    expect(right).toMatchObject({ mode: "ref2va", duration_seconds: 6, source_start_seconds: 5, source_duration_seconds: 9 });
    expect(right.id).not.toBe(left.id);
    expect(state.selected_segment_ids).toEqual([left.id, right.id]);
    expect(state.active_segment_id).toBe(right.id);
  });

  it("智能分镜按全源 cut frame 原子拆分 V2V/RV2V，并拒绝过期结果", () => {
    let state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      duration_seconds: 10,
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 12,
      reference_images: [{ ...image, slot: 0 }],
    };
    state = { ...state, project: { ...state.project, segments: [source] } };
    const expected = { asset_id: video.id, source_start_seconds: 2, source_duration_seconds: 12, project_fps: 24 };
    const split = splitTimelineSourceSegmentAtCuts(
      state,
      source.id,
      [0, 5 * 24, 9 * 24, 20 * 24],
      24,
      ["test-cut-piece-1", "test-cut-piece-2", "test-cut-piece-3", "test-cut-piece-4"],
      expected,
    );
    expect(split.project.segments).toHaveLength(3);
    expect(split.project.segments).toEqual([
      expect.objectContaining({ id: source.id, mode: "ref2va", source_start_seconds: 2, source_duration_seconds: 3, duration_seconds: 2.5, reference_images: source.reference_images }),
      expect.objectContaining({ mode: "ref2va", source_start_seconds: 5, source_duration_seconds: 4, duration_seconds: 10 / 3, reference_images: source.reference_images }),
      expect.objectContaining({ mode: "ref2va", source_start_seconds: 9, source_duration_seconds: 5, duration_seconds: 25 / 6, reference_images: source.reference_images }),
    ]);
    expect(split.project.segments[1].id).not.toBe(source.id);
    expect(split.selected_segment_ids).toEqual(split.project.segments.map((segment) => segment.id));
    expect(split.active_segment_id).toBe(split.project.segments[0].id);

    const stale = splitTimelineSourceSegmentAtCuts(state, source.id, [5 * 24], 24, ["test-stale-piece-1"], {
      ...expected,
      source_duration_seconds: 11,
    });
    expect(stale).toBe(state);
  });

  it("在精确代理帧边界均分源视频并保持连续区间", () => {
    let state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      duration_seconds: 10,
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 10,
      reference_images: [{ ...image, slot: 0 }],
    };
    state = { ...state, project: { ...state.project, segments: [source] } };

    const split = splitTimelineSourceSegmentEvenly(
      state,
      source.id,
      4,
      ["test-even-piece-1", "test-even-piece-2", "test-even-piece-3"],
    );
    expect(split.project.segments).toHaveLength(4);
    expect(split.project.segments.map((segment) => segment.mode === "ref2va" ? [
      segment.source_start_seconds,
      segment.source_duration_seconds,
    ] : null)).toEqual([[2, 2.5], [4.5, 2.5], [7, 2.5], [9.5, 2.5]]);
    expect(split.project.segments.every((segment) =>
      segment.mode === "ref2va" && segment.reference_images[0]?.id === image.id,
    )).toBe(true);
    expect(split.selected_segment_ids).toEqual(split.project.segments.map((segment) => segment.id));
    expect(split.active_segment_id).toBe(split.project.segments[0].id);

    const oddFrameSource = {
      ...source,
      source_duration_seconds: 241 / 24,
    };
    const oddFrameState = {
      ...state,
      project: { ...state.project, segments: [oddFrameSource] },
    };
    const oddFrameSplit = splitTimelineSourceSegmentEvenly(
      oddFrameState,
      source.id,
      4,
      ["test-odd-piece-1", "test-odd-piece-2", "test-odd-piece-3"],
    );
    expect(oddFrameSplit.project.segments.map((segment) =>
      segment.mode === "ref2va" ? Math.round(segment.source_duration_seconds * 24) : 0,
    )).toEqual([61, 60, 60, 60]);

    expect(splitTimelineSourceSegmentEvenly(
      state,
      source.id,
      49,
      Array.from({ length: 48 }, (_, index) => `test-even-overflow-${index}`),
    )).toBe(state);
    expect(splitTimelineSourceSegmentEvenly(state, source.id, 1, [])).toBe(state);
  });

  it("允许显式全选或清空片段复选状态", () => {
    const project = {
      ...createTimelineProject(),
      segments: [createTimelineSegment("fl2va", 1), createTimelineSegment("ref2va", 2)],
    };
    let state = { ...createTimelineEditorState(), project };
    state = timelineEditorReducer(state, {
      type: "segment/set-selection",
      ids: project.segments.map((segment) => segment.id),
    });
    expect(state.selected_segment_ids).toEqual(project.segments.map((segment) => segment.id));
    expect(state.active_segment_id).toBe(project.segments[0].id);
    state = timelineEditorReducer(state, { type: "segment/set-selection", ids: [] });
    expect(state.selected_segment_ids).toEqual([]);
    expect(state.active_segment_id).toBeNull();
    expect(state.selection_anchor_id).toBeNull();

    state = timelineEditorReducer(state, { type: "project/replace", project: structuredClone(project) });
    expect(state.selected_segment_ids).toEqual([]);
    expect(state.active_segment_id).toBeNull();
    expect(state.selection_anchor_id).toBeNull();
  });

  it("历史回放原子恢复项目与选择上下文，并保留素材、网格和合法播放头", () => {
    const first = createTimelineSegment("fl2va", 1);
    const second = createTimelineSegment("ref2va", 2);
    const third = createTimelineSegment("fl2va", 3);
    const restoredProject = {
      ...createTimelineProject(),
      title: "撤销后的项目",
      segments: [first, second, third],
    };
    const assets = [image, audio, video];
    const state = {
      ...createTimelineEditorState(),
      assets,
      selected_asset_ids: [image.id],
      asset_grid_size: "large" as const,
      playhead_seconds: 2,
    };

    const restored = timelineEditorReducer(state, {
      type: "history/restore",
      project: restoredProject,
      selected_segment_ids: [first.id, third.id],
      active_segment_id: third.id,
      selection_anchor_id: first.id,
    });

    expect(restored.project).toBe(restoredProject);
    expect(restored.selected_segment_ids).toEqual([first.id, third.id]);
    expect(restored.active_segment_id).toBe(third.id);
    expect(restored.selection_anchor_id).toBe(first.id);
    expect(restored.assets).toBe(assets);
    expect(restored.selected_asset_ids).toEqual([image.id]);
    expect(restored.asset_grid_size).toBe("large");
    expect(restored.playhead_seconds).toBe(2);
  });

  it("历史回放过滤已失效稳定 ID，并把失效焦点和锚点收敛到首个有效选择", () => {
    const first = createTimelineSegment("fl2va", 1);
    const second = createTimelineSegment("ref2va", 2);
    const project = { ...createTimelineProject(), segments: [first, second] };

    const restored = timelineEditorReducer(createTimelineEditorState(), {
      type: "history/restore",
      project,
      selected_segment_ids: ["deleted-segment", second.id],
      active_segment_id: "deleted-active",
      selection_anchor_id: "deleted-anchor",
    });

    expect(restored.selected_segment_ids).toEqual([second.id]);
    expect(restored.active_segment_id).toBe(second.id);
    expect(restored.selection_anchor_id).toBe(second.id);

    const empty = timelineEditorReducer(restored, {
      type: "history/restore",
      project,
      selected_segment_ids: ["deleted-segment"],
      active_segment_id: second.id,
      selection_anchor_id: second.id,
    });
    expect(empty.selected_segment_ids).toEqual([]);
    expect(empty.active_segment_id).toBeNull();
    expect(empty.selection_anchor_id).toBeNull();
  });

  it("用单一选择派生启用运行集合，禁用和重新启用不丢失选择", () => {
    let state = createTimelineEditorState();
    const first = state.project.segments[0].id;
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after", mode: "ref2va", id: "test-enabled-second" });
    const second = state.project.segments[1].id;
    expect(state.selected_segment_ids).toEqual([first, second]);
    expect(runnableTimelineSegmentIds(state)).toEqual([first, second]);

    state = timelineEditorReducer(state, { type: "segment/select", id: first });
    expect(state.selected_segment_ids).toEqual([first]);
    state = timelineEditorReducer(state, { type: "segment/select", id: second, additive: true });
    expect(state.selected_segment_ids).toEqual([first, second]);
    expect(state.active_segment_id).toBe(first);

    state = timelineEditorReducer(state, {
      type: "segment/set-enabled",
      ids: [second],
      enabled: false,
    });
    expect(runnableTimelineSegmentIds(state)).toEqual([first]);
    expect(state.selected_segment_ids).toEqual([first, second]);
    expect(state.active_segment_id).toBe(first);
    expect(state.project.segments[1].enabled).toBe(false);

    state = timelineEditorReducer(state, {
      type: "segment/set-enabled",
      ids: [second],
      enabled: true,
    });
    expect(state.selected_segment_ids).toEqual([first, second]);
    expect(runnableTimelineSegmentIds(state)).toEqual([first, second]);

    state = timelineEditorReducer(state, { type: "segment/toggle-selection", id: first });
    expect(state.selected_segment_ids).toEqual([second]);
    expect(state.active_segment_id).toBe(second);
    state = timelineEditorReducer(state, { type: "segment/toggle-selection", id: second });
    expect(state.selected_segment_ids).toEqual([]);
    expect(state.active_segment_id).toBeNull();

    let fallback = createTimelineEditorState();
    fallback = timelineEditorReducer(fallback, { type: "segment/delete-selected", fallbackId: "test-delete-fallback" });
    expect(runnableTimelineSegmentIds(fallback)).toEqual([fallback.project.segments[0].id]);
  });

  it("无共享 ID 的项目替换默认选择全部片段，并只运行其中启用段", () => {
    const first = createTimelineSegment("fl2va", 1);
    const second = { ...createTimelineSegment("ref2va", 2), enabled: false };
    const replacement = { ...createTimelineProject(), segments: [first, second] };

    const state = timelineEditorReducer(createTimelineEditorState(), {
      type: "project/replace",
      project: replacement,
    });

    expect(state.selected_segment_ids).toEqual([first.id, second.id]);
    expect(state.active_segment_id).toBe(first.id);
    expect(runnableTimelineSegmentIds(state)).toEqual([first.id]);
  });

  it("复制片段后只选择副本并把第一个副本设为活动片段", () => {
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after", id: "test-duplicate-source-second" });
    state = timelineEditorReducer(state, {
      type: "segment/duplicate-selected",
      ids: ["test-duplicate-copy-1", "test-duplicate-copy-2"],
    });

    const copiedIds = state.selected_segment_ids;
    expect(copiedIds).toHaveLength(2);
    expect(state.active_segment_id).toBe(copiedIds[0]);
    expect(runnableTimelineSegmentIds(state)).toEqual(copiedIds);
  });

  it("将绝对播放头定位到稳定片段，并按比例映射 V2V 源时间", () => {
    const first = { ...createTimelineSegment("fl2va", 1), duration_seconds: 2 };
    const second = {
      ...createTimelineSegment("ref2va", 2),
      duration_seconds: 4,
      source_video: video,
      source_start_seconds: 3,
      source_duration_seconds: 8,
    };
    const project = { ...createTimelineProject(), segments: [first, second] };
    const position = timelineSegmentAt(project, 3);
    expect(position).toMatchObject({ index: 1 });
    expect(position?.start_seconds).toBeCloseTo(56 / 24);
    expect(position?.local_seconds).toBeCloseTo(3 - 56 / 24);
    expect(position?.segment.id).toBe(second.id);
    expect(sourcePreviewTime(second, position?.local_seconds ?? 0, 24)).toBeCloseTo(
      3 + 8 * ((3 - 56 / 24) / (107 / 24)),
    );
    expect(timelineSegmentAt(project, 56 / 24)?.segment.id).toBe(second.id);
  });

  it("统一用 17k+5 实际时基，停用段可见但不占节目时间", () => {
    const first = { ...createTimelineSegment("fl2va", 1), duration_seconds: 5 };
    const disabled = { ...createTimelineSegment("fl2va", 2), duration_seconds: 5, enabled: false };
    const third = { ...createTimelineSegment("fl2va", 3), duration_seconds: 5 };
    const project = { ...createTimelineProject(), segments: [first, disabled, third] };
    expect(alignedTimelineSegmentDuration(first, 24)).toBeCloseTo(124 / 24);
    expect(timelineDuration(project)).toBeCloseTo(248 / 24);
    expect(timelineSegmentAt(project, 124 / 24)?.segment.id).toBe(third.id);
  });

  it("# 只插入当前片段已引入的 Ref2VA 或 FL2VA 稳定标签", () => {
    const r2v = {
      ...createTimelineSegment("ref2va", 1),
      prompt: "中文，#role 走入",
      reference_images: [{ ...image, slot: 0 }],
    };
    const tagged = insertPromptReferenceToken(r2v, "<Picture 1>", 3, 8, "#role");
    expect(tagged.mode).toBe("ref2va");
    expect(tagged.prompt).toBe("中文， <Picture 1> 走入");
    if (tagged.mode !== "ref2va") throw new Error("fixture changed mode");
    expect(tagged.reference_images).toEqual([expect.objectContaining({ id: image.id, slot: 0 })]);

    const i2v = {
      ...createTimelineSegment("fl2va", 1),
      prompt: "镜头，#face推进",
      first_image: image,
    };
    const anchored = insertPromptReferenceToken(i2v, "<Picture 1>", 3, 8, "#face");
    expect(anchored).toMatchObject({
      mode: "fl2va",
      prompt: "镜头， <Picture 1> 推进",
      first_image: image,
    });

    const unbound = { ...createTimelineSegment("ref2va", 1), prompt: "#role" };
    expect(insertPromptReferenceToken(unbound, "<Picture 1>", 0, 5, "#role")).toBe(unbound);
    expect(insertPromptReferenceToken(r2v, "<Picture 1>", 3, 8, "#stale")).toBe(r2v);
    const legacyAtMention = { ...r2v, prompt: "中文，@role 走入" };
    expect(insertPromptReferenceToken(legacyAtMention, "<Picture 1>", 3, 8, "@role")).toBe(legacyAtMention);
  });

  it("@ 识别 subject 每个字母的任意大小写组合，且只读取定义区标签", () => {
    const prompt = [
      "SUBJECT_DEFINITIONS:",
      "<SuBjEcT 1> is the lead dancer.",
      "<sUBJect 2> is the rehearsal room.",
      "<SUBject 2> is a duplicate definition.",
      "summary:",
      "<subject 99> is only a reference here.",
      "镜头 @lead推进",
    ].join("\n");
    expect(promptSubjectReferences(prompt)).toEqual([
      { number: 1, token: "<SuBjEcT 1>", definition: "is the lead dancer." },
      { number: 2, token: "<sUBJect 2>", definition: "is the rehearsal room." },
    ]);

    const segment = { ...createTimelineSegment("ref2va", 1), prompt };
    const start = prompt.indexOf("@lead");
    const tagged = insertPromptSubjectToken(
      segment,
      "<SuBjEcT 1>",
      start,
      start + "@lead".length,
      "@lead",
    );
    expect(tagged.prompt).toContain("镜头 <SuBjEcT 1> 推进");
    expect(insertPromptSubjectToken(segment, "<subject 99>", start, start + 5, "@lead")).toBe(segment);
    const wrongTrigger = { ...segment, prompt: prompt.replace("@lead", "#lead") };
    expect(insertPromptSubjectToken(wrongTrigger, "<SuBjEcT 1>", start, start + 5, "#lead")).toBe(wrongTrigger);
  });

  it("FL2VA 锚点增删时按素材身份迁移 Picture 编号", () => {
    const first = { ...image, id: "fl-first", name: "first.png" };
    const last = { ...image, id: "fl-last", name: "last.png" };
    const lastOnly = {
      ...createTimelineSegment("fl2va", 1),
      prompt: "尾帧保持 <Picture 1>",
      last_image: last,
    };

    const both = assignAssetToSegment(lastOnly, first, "first_image");
    expect(both).toMatchObject({
      first_image: first,
      last_image: last,
      prompt: "尾帧保持 <Picture 2>",
    });

    const authored = {
      ...both,
      prompt: "首帧 <Picture 1>  \n\n  尾帧 <Picture 2>",
    };
    const removedFirst = removeAssetFromSegment(authored, first.id);
    expect(removedFirst).toMatchObject({
      first_image: null,
      last_image: last,
      prompt: "首帧   \n\n  尾帧 <Picture 1>",
    });
    expect(removeAssetFromSegment(removedFirst, last.id)).toMatchObject({
      last_image: null,
      prompt: "首帧   \n\n  尾帧 ",
    });
  });

  it("应用片段配置复制 strict union，保留目标身份和启用状态", () => {
    let state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      title: "源配置",
      prompt: "重绘",
      source_video: video,
      reference_images: [{ ...image, slot: 0 }],
    };
    const target = { ...createTimelineSegment("fl2va", 2), title: "保留名称", enabled: false, first_image: image };
    state = { ...state, project: { ...state.project, segments: [source, target] } };
    state = applyTimelineSegmentConfiguration(state, source.id, "following", {
      ...DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS,
      prompt: true,
      promptReferences: true,
    });
    expect(state.project.segments[1]).toMatchObject({
      id: target.id, title: "保留名称", enabled: false, mode: "ref2va", prompt: "重绘",
      source_video: video,
      reference_images: [{ ...image, slot: 0 }],
    });
    expect(state.project.segments[1]).not.toHaveProperty("first_image");
  });

  it("应用到所选只更新所选目标并排除来源，选择与焦点保持不变", () => {
    const state = createTimelineEditorState();
    const source = { ...state.project.segments[0], duration_seconds: 9 };
    const selectedTarget = { ...createTimelineSegment("fl2va", 2), duration_seconds: 3, enabled: false };
    const unselectedTarget = { ...createTimelineSegment("fl2va", 3), duration_seconds: 4 };
    const initial = {
      ...state,
      project: { ...state.project, segments: [source, selectedTarget, unselectedTarget] },
      selected_segment_ids: [source.id, selectedTarget.id],
      active_segment_id: source.id,
      selection_anchor_id: source.id,
    };
    const next = applyTimelineSegmentConfiguration(initial, source.id, "selected", {
      mode: false,
      duration: true,
      continuity: false,
      audioMode: false,
      refImageSize: false,
      prompt: false,
      promptReferences: false,
    });

    expect(next.project.segments.map((segment) => segment.duration_seconds)).toEqual([9, 9, 4]);
    expect(next.project.segments[1]).toMatchObject({
      id: selectedTarget.id,
      title: selectedTarget.title,
      enabled: false,
    });
    expect(next).toMatchObject({
      selected_segment_ids: [source.id, selectedTarget.id],
      active_segment_id: source.id,
      selection_anchor_id: source.id,
    });
  });

  it("提示词可独立逐字节复制并保留目标自身素材绑定", () => {
    const source = {
      ...createTimelineSegment("ref2va", 1),
      prompt: "主体  <Picture 1>   \n\n  镜头 <Video 1> ",
      source_video: video,
      reference_images: [{ ...image, slot: 0 }],
    };
    const targetImage = { ...image, id: "target-image", name: "目标图片.png" };
    const target = {
      ...createTimelineSegment("ref2va", 2),
      prompt: "旧提示词",
      reference_images: [{ ...targetImage, slot: 0 }],
    };
    const copied = copyTimelineSegmentConfiguration(source, target, {
      mode: false,
      duration: false,
      continuity: false,
      audioMode: false,
      refImageSize: false,
      prompt: true,
      promptReferences: false,
    });

    expect(copied.prompt).toBe(source.prompt);
    expect(copied).toMatchObject({ reference_images: [{ id: targetImage.id, slot: 0 }] });
    expect(copied).toMatchObject({ source_video: null });
  });

  it("连同引用素材复制完整 Ref 布局且不与来源共享数组", () => {
    const source = {
      ...createTimelineSegment("ref2va", 1),
      prompt: "<Video 1> <Audio 1> <Picture 1>",
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 6,
      source_audio_as_reference: true,
      reference_images: [{ ...image, slot: 0 }],
      reference_audios: [{ ...audio, slot: 0 }],
    };
    const target = { ...createTimelineSegment("fl2va", 2), first_image: image };
    const copied = copyTimelineSegmentConfiguration(source, target, {
      mode: true,
      duration: false,
      continuity: false,
      audioMode: false,
      refImageSize: false,
      prompt: true,
      promptReferences: true,
    });

    expect(copied).toMatchObject({
      mode: "ref2va",
      prompt: source.prompt,
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 6,
      source_audio_as_reference: true,
      reference_images: [{ ...image, slot: 0 }],
      reference_audios: [{ ...audio, slot: 0 }],
    });
    expect(copied).not.toHaveProperty("first_image");
    if (copied.mode === "ref2va") {
      expect(copied.reference_images).not.toBe(source.reference_images);
      expect(copied.reference_audios).not.toBe(source.reference_audios);
    }
  });

  it("复制 FL 首帧引用时同步关闭目标连续性", () => {
    const source = {
      ...createTimelineSegment("fl2va", 1),
      prompt: "首帧 <Picture 1>",
      first_image: image,
    };
    const target = {
      ...createTimelineSegment("fl2va", 2),
      continuity: { enabled: true as const, overlap_frames: 39 as const },
    };
    const copied = copyTimelineSegmentConfiguration(source, target, {
      mode: true,
      duration: false,
      continuity: false,
      audioMode: false,
      refImageSize: false,
      prompt: true,
      promptReferences: true,
    });
    expect(copied).toMatchObject({
      first_image: image,
      continuity: { enabled: false, overlap_frames: 39 },
    });
  });

  it("没有复制项或引用素材依赖非法时不制造项目写入", () => {
    const state = createTimelineEditorState();
    const source = state.project.segments[0];
    const target = createTimelineSegment("fl2va", 2);
    const initial = {
      ...state,
      project: { ...state.project, segments: [source, target] },
    };
    const emptyOptions = {
      mode: false,
      duration: false,
      continuity: false,
      audioMode: false,
      refImageSize: false,
      prompt: false,
      promptReferences: false,
    };
    expect(applyTimelineSegmentConfiguration(initial, source.id, "following", emptyOptions))
      .toBe(initial);
    expect(applyTimelineSegmentConfiguration(initial, source.id, "following", {
      ...emptyOptions,
      prompt: true,
      promptReferences: true,
    })).toBe(initial);
  });

  it("引用计数返回稳定片段位置与角色", () => {
    const segment = { ...createTimelineSegment("ref2va", 1), reference_images: [{ ...image, slot: 0 }] };
    const project = { ...createTimelineProject(), segments: [segment] };
    expect(timelineAssetUsages(project, image.id)).toEqual([
      expect.objectContaining({ segment_id: segment.id, segment_index: 0, role: "<Picture 1>" }),
    ]);
  });

  it("选择生成只校验选中片段，不被未选启用段的专属素材阻断", () => {
    const valid = { ...createTimelineSegment("fl2va", 1), prompt: "有效段" };
    const invalid = { ...createTimelineSegment("ref2va", 2), prompt: "未选中" };
    const project = { ...createTimelineProject(), segments: [valid, invalid] };
    expect(validateTimelineProject(project).join(" ")).toContain("至少需要源视频或一个独立参考素材");
    expect(validateTimelineProject(project, [valid.id]).join(" ")).not.toContain("至少需要源视频或一个独立参考素材");
  });

  it("合并只允许兼容素材且相邻源区间，并遵守 512 帧上限", () => {
    let state = createTimelineEditorState();
    const first = { ...createTimelineSegment("ref2va", 1), prompt: "重绘", source_video: video, source_start_seconds: 0, source_duration_seconds: 4, duration_seconds: 4 };
    const second = { ...createTimelineSegment("ref2va", 2), prompt: "重绘", source_video: video, source_start_seconds: 4, source_duration_seconds: 4, duration_seconds: 4 };
    state = {
      ...state,
      project: { ...state.project, segments: [first, second] },
      selected_segment_ids: [first.id, second.id],
      active_segment_id: first.id,
      selection_anchor_id: first.id,
    };
    expect(canMergeSelectedSegments(state)).toBe(true);
    state = timelineEditorReducer(state, { type: "segment/merge-selected" });
    expect(state.project.segments).toHaveLength(1);
    expect(state.project.segments[0]).toMatchObject({ duration_seconds: 8, source_duration_seconds: 8 });
    expect(state.selected_segment_ids).toEqual([first.id]);
    expect(state.active_segment_id).toBe(first.id);
    expect(runnableTimelineSegmentIds(state)).toEqual([first.id]);

    const incompatible = createTimelineEditorState();
    const a = { ...createTimelineSegment("fl2va", 1), first_image: image };
    const b = { ...createTimelineSegment("fl2va", 2), first_image: { ...image, id: "different-image" } };
    const flState = { ...incompatible, project: { ...incompatible.project, segments: [a, b] }, selected_segment_ids: [a.id, b.id] };
    expect(canMergeSelectedSegments(flState)).toBe(false);

    const mixedEnabledState = {
      ...incompatible,
      project: {
        ...incompatible.project,
        segments: [first, { ...second, enabled: false }],
      },
      selected_segment_ids: [first.id, second.id],
      active_segment_id: first.id,
      selection_anchor_id: first.id,
    };
    expect(canMergeSelectedSegments(mixedEnabledState)).toBe(false);
  });

  it("normalizer 按模式白名单剥离 UI 和其他模式字段", () => {
    const project = createTimelineProject() as unknown as Record<string, unknown>;
    project.playhead_seconds = 12;
    const segments = project.segments as Record<string, unknown>[];
    segments[0].source_video = video;
    segments[0].reference_images = [{ ...image, slot: 0 }];
    segments[0].candidate_take = "/output.mp4";
    const normalized = normalizeTimelineProject(project)!;
    expect(normalized).not.toHaveProperty("playhead_seconds");
    expect(normalized.segments[0]).not.toHaveProperty("source_video");
    expect(normalized.segments[0]).not.toHaveProperty("reference_images");
    expect(normalized.segments[0]).not.toHaveProperty("candidate_take");
  });

  it("normalizer 将旧 FPS/单套采样迁移为 24fps 与两套无 CFG 配置", () => {
    const legacy = structuredClone(createTimelineProject()) as unknown as Record<string, unknown>;
    legacy.version = 1;
    legacy.segments = [{
      id: "legacy-flat-sampling",
      mode: "t2v",
      title: "旧片段",
      prompt: "旧提示词",
      duration_seconds: 5,
      enabled: true,
    }];
    (legacy.render as Record<string, unknown>).fps = 30;
    legacy.sampling = {
      steps: 31,
      seed: 17,
      sampler: "euler",
      scheduler: "normal",
      shift: 3,
      audio_shift: 4,
      cfg: 7.5,
    };
    const normalized = normalizeTimelineProject(legacy)!;
    expect(normalized.render.fps).toBe(24);
    expect(normalized.sampling.fl2va).toEqual({
      steps: 31,
      seed: 17,
      random_seed: false,
      sampler: "euler",
      scheduler: "normal",
      shift: 3,
      audio_shift: 4,
    });
    expect(normalized.sampling.ref2va).toEqual(normalized.sampling.fl2va);
    expect(normalized.sampling.fl2va).not.toHaveProperty("cfg");
    expect(normalized.sampling.ref2va).not.toHaveProperty("cfg");
  });

  it("把 v2 全片连续性逐段迁移到 v4，并拒绝混入逐段字段的伪 v2", () => {
    const legacy = structuredClone(createTimelineProject()) as unknown as Record<string, unknown>;
    legacy.version = 2;
    legacy.continuity = { enabled: true, overlap_frames: 39 };
    legacy.ref_image_size = "match";
    legacy.audio_mode = "generate";
    legacy.segments = (legacy.segments as Record<string, unknown>[]).map((segment) => {
      const migrated = { ...segment };
      delete migrated.continuity;
      delete migrated.ref_image_size;
      delete migrated.audio_mode;
      return migrated;
    });
    const normalized = normalizeTimelineProject(legacy)!;
    expect(normalized.version).toBe(4);
    expect(normalized).not.toHaveProperty("continuity");
    expect(normalized.segments.every((segment) =>
      segment.continuity.enabled && segment.continuity.overlap_frames === 39)).toBe(true);

    const smuggled = structuredClone(legacy) as Record<string, unknown>;
    (smuggled.segments as Record<string, unknown>[])[0].continuity = {
      enabled: false,
      overlap_frames: 5,
    };
    expect(normalizeTimelineProject(smuggled)).toBeNull();
  });

  it("把 v3 的全片音频与参考图策略复制到每个 v4 片段", () => {
    const legacy = structuredClone(createTimelineProject()) as unknown as Record<string, unknown>;
    legacy.version = 3;
    legacy.ref_image_size = "max";
    legacy.audio_mode = "mute";
    legacy.segments = (legacy.segments as Record<string, unknown>[]).map((segment) => {
      const migrated = { ...segment };
      delete migrated.ref_image_size;
      delete migrated.audio_mode;
      return migrated;
    });

    const normalized = normalizeTimelineProject(legacy)!;

    expect(normalized.version).toBe(4);
    expect(normalized).not.toHaveProperty("ref_image_size");
    expect(normalized).not.toHaveProperty("audio_mode");
    expect(normalized.segments[0]).toMatchObject({
      ref_image_size: "max",
      audio_mode: "mute",
    });
  });

  it("beta 属于统一调度器契约，并在 v2 与旧单套采样迁移中原样保留", () => {
    expect(SAMPLING_SCHEDULERS).toEqual(["simple", "normal", "karras", "beta"]);

    const current = createTimelineProject();
    current.sampling.fl2va.scheduler = "beta";
    current.sampling.ref2va.scheduler = "beta";
    expect(normalizeTimelineProject(current)?.sampling).toMatchObject({
      fl2va: { scheduler: "beta" },
      ref2va: { scheduler: "beta" },
    });

    const legacy = structuredClone(current) as unknown as Record<string, unknown>;
    legacy.version = 1;
    legacy.segments = [{
      id: "legacy-beta",
      mode: "t2v",
      title: "旧 beta 项目",
      prompt: "测试 beta 调度器",
      duration_seconds: 5,
      enabled: true,
    }];
    legacy.sampling = { ...current.sampling.fl2va, scheduler: "beta" };
    expect(normalizeTimelineProject(legacy)?.sampling).toMatchObject({
      fl2va: { scheduler: "beta" },
      ref2va: { scheduler: "beta" },
    });
  });

  it("输出规格严格使用原生 multiple=32 档位并为 9:16 转置", () => {
    expect(H3_OUTPUT_RESOLUTIONS_16_9).toEqual([
      [608, 352], [736, 416], [864, 480], [960, 544], [1056, 608],
      [1152, 640], [1216, 672], [1280, 736], [1344, 768], [1376, 768],
      [1504, 832], [1664, 928], [1824, 1024], [1920, 1088],
    ]);
    expect(timelineOutputResolutions("9:16")).toEqual(
      H3_OUTPUT_RESOLUTIONS_16_9.map(([width, height]) => ({
        width: height,
        height: width,
      })),
    );
    expect(isTimelineOutputResolution(864, 480)).toBe(true);
    expect(isTimelineOutputResolution(480, 864)).toBe(true);
    expect(isTimelineOutputResolution(1024, 576)).toBe(false);
    expect(inferTimelineOutputAspect(1024, 576)).toBe("16:9");
    expect(inferTimelineOutputAspect(800, 600)).toBeNull();
    expect(closestTimelineOutputResolution(1920, 1088, "9:16")).toEqual({
      width: 1088,
      height: 1920,
    });
  });

  it("normalizer 保留旧项目的自定义 32 倍数尺寸，只继续固定 24fps", () => {
    const legacy = structuredClone(createTimelineProject()) as unknown as Record<string, unknown>;
    legacy.version = 1;
    legacy.render = { width: 1024, height: 576, fps: 30 };
    legacy.segments = [{
      id: "legacy-custom-resolution",
      mode: "t2v",
      title: "自定义规格",
      prompt: "",
      duration_seconds: 5,
      enabled: true,
    }];
    const normalized = normalizeTimelineProject(legacy)!;
    expect(normalized.render).toEqual({ width: 1024, height: 576, fps: 24 });
    expect(inferTimelineOutputAspect(normalized.render.width, normalized.render.height)).toBe("16:9");
    expect(isTimelineOutputResolution(normalized.render.width, normalized.render.height)).toBe(false);
  });

  it("旧全片提示词只填充空片段，-1 Seed 迁移为可见安全整数", () => {
    const legacy = structuredClone(createTimelineProject()) as unknown as Record<string, unknown>;
    legacy.version = 1;
    const first = { id: "legacy-first", mode: "t2v", title: "旧片段 1", prompt: "", duration_seconds: 5, enabled: true };
    const second = { id: "legacy-second", mode: "t2v", title: "旧片段 2", prompt: "片段自己的提示词", duration_seconds: 5, enabled: true };
    legacy.prompt = "旧全片默认提示词";
    legacy.segments = [first, second];
    legacy.sampling = {
      steps: 25,
      seed: -1,
      sampler: "res_multistep",
      scheduler: "simple",
      shift: 5,
      audio_shift: 3,
      cfg: 1,
    };

    const normalized = normalizeTimelineProject(legacy)!;
    expect(normalized).not.toHaveProperty("prompt");
    expect(normalized.segments.map((segment) => segment.prompt)).toEqual([
      "旧全片默认提示词",
      "片段自己的提示词",
    ]);
    expect(normalized.sampling.fl2va.random_seed).toBe(true);
    expect(normalized.sampling.fl2va.seed).toBe(normalized.sampling.ref2va.seed);
    expect(Number.isSafeInteger(normalized.sampling.fl2va.seed)).toBe(true);
    expect(normalized.sampling.fl2va.seed).toBeGreaterThanOrEqual(0);
  });

  it("explicit v2 不再迁移已删除的单套采样或全片提示词", () => {
    const flatSampling = structuredClone(createTimelineProject()) as unknown as Record<string, unknown>;
    flatSampling.sampling = {
      steps: 25,
      seed: 42,
      sampler: "res_multistep",
      scheduler: "simple",
      shift: 12,
      audio_shift: 3,
    };
    expect(normalizeTimelineProject(flatSampling)).toBeNull();

    const sharedPrompt = structuredClone(createTimelineProject()) as unknown as Record<string, unknown>;
    sharedPrompt.prompt = "v2 不允许的全片提示词";
    expect(normalizeTimelineProject(sharedPrompt)).toBeNull();
  });

  it("提交前阻止每种参考 modality 的稀疏 slot", () => {
    const project = createTimelineProject();
    project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      prompt: "参考素材",
      reference_images: [{ ...image, slot: 1 }],
      reference_audios: [{ ...audio, slot: 0 }],
      reference_videos: [{ ...video, slot: 2 }],
    }];
    const errors = validateTimelineProject(project);
    expect(errors).toEqual(expect.arrayContaining([
      expect.stringContaining("参考图片槽位必须连续为 0..N-1"),
      expect.stringContaining("参考视频槽位必须连续为 0..N-1"),
    ]));
    expect(errors.join(" ")).not.toContain("参考音频槽位必须连续");
  });

  it("校验模式素材、标签和源范围，但允许跨模型族段间接续", () => {
    const project = createTimelineProject();
    project.segments = [
      { ...createTimelineSegment("fl2va", 1), prompt: "人物走进画面" },
      { ...createTimelineSegment("ref2va", 2), prompt: "重绘", source_video: video, source_start_seconds: 18, source_duration_seconds: 4, continuity: { enabled: true, overlap_frames: 22 } },
    ];
    const errors = validateTimelineProject(project);
    expect(errors).toEqual(expect.arrayContaining([
      expect.stringContaining("源视频范围超过"),
    ]));
    expect(errors.join(" ")).not.toContain("模型族不同");
    expect(errors.join(" ")).not.toContain("未绑定素材");
  });

  it("段间接续按完整启用时间线解析直接前驱，并把显式首帧作为硬断点", () => {
    const first = { ...createTimelineSegment("fl2va", 1), id: "first", prompt: "第一段" };
    const disabled = { ...createTimelineSegment("ref2va", 2), id: "disabled", enabled: false };
    const second = { ...createTimelineSegment("fl2va", 3), id: "second", prompt: "第二段" };
    const anchored = { ...createTimelineSegment("fl2va", 4), id: "anchored", prompt: "锚点段", first_image: image };
    const ref = { ...createTimelineSegment("ref2va", 5), id: "ref", prompt: "参考段", source_video: video };
    const project = { ...createTimelineProject(), segments: [first, disabled, second, anchored, ref] };

    expect(timelineContinuityBoundaries(project).map((boundary) => ({
      predecessor: boundary.predecessor.id,
      target: boundary.segment.id,
      kind: boundary.kind,
    }))).toEqual([
      { predecessor: "first", target: "second", kind: "eligible" },
      { predecessor: "second", target: "anchored", kind: "explicit-first-image" },
      { predecessor: "anchored", target: "ref", kind: "eligible" },
    ]);

    const refFirst = { ...createTimelineSegment("ref2va", 1), id: "ref-first" };
    const refSecond = { ...createTimelineSegment("ref2va", 2), id: "ref-second" };
    expect(timelineContinuityBoundaries({
      ...createTimelineProject(),
      segments: [refFirst, refSecond],
    })[0]).toMatchObject({
      predecessor: { id: refFirst.id },
      segment: { id: refSecond.id },
      kind: "eligible",
    });
  });

  it("所选后段未包含直接前驱时表达历史成片意图，且不会把非连续所选段错当相邻", () => {
    const first = { ...createTimelineSegment("fl2va", 1), id: "first", prompt: "第一段" };
    const second = { ...createTimelineSegment("fl2va", 2), id: "second", prompt: "第二段", continuity: { enabled: true, overlap_frames: 39 as const } };
    const third = { ...createTimelineSegment("fl2va", 3), id: "third", prompt: "第三段", continuity: { enabled: true, overlap_frames: 39 as const } };
    const project = { ...createTimelineProject(), segments: [first, second, third] };

    const onlyThird = timelineContinuityRunIssues(project, [third.id]);
    expect(onlyThird).toHaveLength(1);
    expect(onlyThird[0]).toMatchObject({
      code: "historical-take-required",
      boundary: { predecessor: expect.objectContaining({ id: second.id }), segment: expect.objectContaining({ id: third.id }) },
    });
    expect(onlyThird[0].message).toContain("最后 39 帧");
    expect(onlyThird[0].message).toContain("复用直接前驱");
    expect(onlyThird[0].message).not.toContain("第一段");

    expect(timelineContinuityRunIssues(project, [second.id, third.id])[0].boundary.predecessor.id).toBe(first.id);
    expect(timelineContinuityRunIssues(project, [first.id, second.id, third.id])).toEqual([]);
    expect(validateTimelineProject(project, [third.id]).join(" ")).not.toContain("直接前驱");
  });

  it("显式首帧切断其入边，但其输出可跨族接续下一段", () => {
    const first = { ...createTimelineSegment("fl2va", 1), id: "first", prompt: "第一段" };
    const anchored = { ...createTimelineSegment("fl2va", 2), id: "anchored", prompt: "锚点段", first_image: image, continuity: { enabled: true, overlap_frames: 22 as const } };
    const ref = { ...createTimelineSegment("ref2va", 3), id: "ref", prompt: "参考段", source_video: video, continuity: { enabled: true, overlap_frames: 22 as const } };
    const project = { ...createTimelineProject(), segments: [first, anchored, ref] };

    expect(timelineContinuityRunIssues(project, [anchored.id])).toEqual([]);
    expect(timelineContinuityRunIssues(project, [first.id])).toEqual([]);
    expect(timelineContinuityRunIssues(project, [ref.id])).toEqual([
      expect.objectContaining({
        code: "historical-take-required",
        boundary: expect.objectContaining({
          predecessor: expect.objectContaining({ id: anchored.id }),
        }),
      }),
    ]);
  });

  it("段间接续阻止前段尾帧不足和后段内部采样超过 512 帧", () => {
    const shortPredecessor = {
      ...createTimelineSegment("fl2va", 1),
      id: "short-predecessor",
      prompt: "短前段",
      duration_seconds: 0.1,
    };
    const longTarget = {
      ...createTimelineSegment("fl2va", 2),
      id: "long-target",
      prompt: "长后段",
      duration_seconds: 498 / 24,
      continuity: { enabled: true, overlap_frames: 22 as const },
    };
    const project = {
      ...createTimelineProject(),
      segments: [shortPredecessor, longTarget],
    };

    const issues = timelineContinuityRunIssues(project, [shortPredecessor.id, longTarget.id]);
    expect(issues).toEqual(expect.arrayContaining([
      expect.objectContaining({
        code: "predecessor-too-short",
        message: expect.stringContaining("只有 5 个可见帧，少于段间接续需要的 22 帧"),
      }),
      expect.objectContaining({
        code: "sample-too-long",
        message: expect.stringContaining("内部采样 532 帧（可见 498 帧）"),
      }),
    ]));
    expect(validateTimelineProject(project, [shortPredecessor.id, longTarget.id]).join(" ")).toContain(
      "超过 MiniMax H3 的 512 帧上限",
    );
  });

  it.each([
    [5, 481],
    [22, 464],
    [39, 447],
    [56, 430],
  ] as const)("接续尾帧 %i 在 498/515 内部采样临界点严格阻断", (overlapFrames, maximumVisibleFrames) => {
    const predecessor = {
      ...createTimelineSegment("fl2va", 1),
      id: `predecessor-${overlapFrames}`,
      prompt: "足够长的前段",
      duration_seconds: 498 / 24,
    };
    const target = {
      ...createTimelineSegment("fl2va", 2),
      id: `target-${overlapFrames}`,
      prompt: "临界后段",
      duration_seconds: maximumVisibleFrames / 24,
      continuity: { enabled: true, overlap_frames: overlapFrames },
    };
    const project = {
      ...createTimelineProject(),
      segments: [predecessor, target],
    };
    const selected = [predecessor.id, target.id];

    expect(timelineContinuityRunIssues(project, selected)).toEqual([]);
    target.duration_seconds = (maximumVisibleFrames + 17) / 24;
    expect(timelineContinuityRunIssues(project, selected)).toEqual([
      expect.objectContaining({
        code: "sample-too-long",
        message: expect.stringContaining("内部采样 515 帧"),
      }),
    ]);
  });

  it("T2V、尾帧 FL2V 与 Ref2V 可混排接续，显式首帧只重置自身入边", () => {
    const firstRef = {
      ...createTimelineSegment("ref2va", 1),
      id: "ref-first",
      prompt: "参考第一段",
      source_video: video,
    };
    const secondRef = {
      ...createTimelineSegment("ref2va", 2),
      id: "ref-second",
      prompt: "参考第二段",
      source_video: video,
      continuity: { enabled: true, overlap_frames: 22 as const },
    };
    const refProject = {
      ...createTimelineProject(),
      segments: [firstRef, secondRef],
    };
    expect(timelineContinuityBoundaries(refProject)).toEqual([
      expect.objectContaining({ kind: "eligible", predecessor: firstRef, segment: secondRef }),
    ]);
    expect(timelineContinuityRunIssues(refProject, [firstRef.id, secondRef.id])).toEqual([]);

    const tailOnly = {
      ...createTimelineSegment("fl2va", 2),
      id: "tail-only",
      prompt: "只有尾帧",
      last_image: image,
    };
    const flFirst = { ...createTimelineSegment("fl2va", 1), id: "fl-first", prompt: "FL 第一段" };
    expect(timelineContinuityBoundaries({
      ...createTimelineProject(),
      segments: [flFirst, tailOnly],
    })[0]).toMatchObject({ kind: "eligible", segment: { id: tailOnly.id } });

    const anchoredFl = {
      ...createTimelineSegment("fl2va", 2),
      id: "anchored-after-ref",
      prompt: "跨族显式锚点",
      first_image: image,
      continuity: { enabled: true, overlap_frames: 22 as const },
    };
    const resetProject = {
      ...createTimelineProject(),
      segments: [firstRef, anchoredFl],
    };
    expect(timelineContinuityBoundaries(resetProject)[0]).toMatchObject({
      kind: "explicit-first-image",
      predecessor: { id: firstRef.id },
      segment: { id: anchoredFl.id },
    });
    expect(timelineContinuityRunIssues(resetProject, [anchoredFl.id])).toEqual([]);

    const mixedProject = {
      ...createTimelineProject(),
      segments: [
        flFirst,
        { ...firstRef, continuity: { enabled: true, overlap_frames: 22 as const } },
        { ...tailOnly, continuity: { enabled: true, overlap_frames: 22 as const } },
      ],
    };
    expect(timelineContinuityBoundaries(mixedProject).map((boundary) => boundary.kind)).toEqual([
      "eligible",
      "eligible",
    ]);
    expect(timelineContinuityRunIssues(
      mixedProject,
      mixedProject.segments.map((segment) => segment.id),
    )).toEqual([]);
  });

  it("保留源音频要求源裁剪帧数与 H3 对齐输出相同", () => {
    const project = createTimelineProject();
    project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      audio_mode: "source",
      prompt: "重绘",
      source_video: video,
      duration_seconds: 2,
      source_start_seconds: 0,
      source_duration_seconds: 2,
    }];
    expect(validateTimelineProject(project).join(" ")).toContain("保留源音频时不能变速");
    const segment = project.segments[0];
    if (segment.mode !== "ref2va") throw new Error("test fixture must remain v2v");
    segment.source_duration_seconds = 56 / 24;
    expect(validateTimelineProject(project).join(" ")).not.toContain("保留源音频时不能变速");

    if (!segment.source_video?.metadata) throw new Error("fixture requires metadata");
    segment.source_video.metadata.has_audio = false;
    expect(validateTimelineProject(project).join(" ")).toContain("无法保留源音频");
  });

  it("保留源音频时自动延长源范围，素材不足则回退到前一个 H3 合法长度", () => {
    const project = createTimelineProject();
    const shortVideo: AssetReference = {
      ...video,
      metadata: {
        ...video.metadata!,
        duration: 269 / 24,
        native_fps: 24,
        frame_count: 269,
      },
    };
    project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      audio_mode: "source",
      title: "片段 01",
      prompt: "重绘",
      source_video: shortVideo,
      source_start_seconds: 0,
      source_duration_seconds: 269 / 24,
      duration_seconds: 269 / 24,
    }];

    const shortened = autoFitSourceAudioTiming(project);
    expect(shortened.adjustments).toEqual([expect.objectContaining({
      source_frames_before: 269,
      source_frames_after: 260,
      output_frames_before: 277,
      fallback_to_previous_h3_length: true,
    })]);
    expect(shortened.project.segments[0]).toMatchObject({
      source_duration_seconds: 260 / 24,
      duration_seconds: 260 / 24,
    });
    expect(validateTimelineProject(shortened.project).join(" ")).not.toContain("保留源音频时不能变速");

    const extendable = structuredClone(project);
    const segment = extendable.segments[0];
    if (segment.mode !== "ref2va" || !segment.source_video?.metadata)
      throw new Error("test fixture must remain source-backed Ref2VA");
    segment.source_video.metadata.duration = 20;
    segment.source_video.metadata.frame_count = 480;
    const extended = autoFitSourceAudioTiming(extendable);
    expect(extended.adjustments[0]).toMatchObject({
      source_frames_before: 269,
      source_frames_after: 277,
      fallback_to_previous_h3_length: false,
    });
    expect(extended.project.segments[0]).toMatchObject({
      source_duration_seconds: 277 / 24,
      duration_seconds: 269 / 24,
    });
    expect(validateTimelineProject(extended.project).join(" ")).not.toContain("保留源音频时不能变速");
  });

  it("保留源音频的校验只使用两族编辑概念", () => {
    const project = createTimelineProject();
    project.segments[0].audio_mode = "source";
    project.segments[0].prompt = "人物走入画面";
    const errors = validateTimelineProject(project).join(" ");
    expect(errors).toContain("Ref2VA 且已绑定源视频");
    expect(errors).not.toMatch(/V2V|RV2V/);
  });

  it("素材顺序偏好只按 ID 排序，未知新素材保持在尾部", () => {
    expect(orderAssetsByPreference([image, audio, video], [video.id, image.id]).map((asset) => asset.id)).toEqual([
      video.id,
      image.id,
      audio.id,
    ]);
  });

  it("FL2VA 引入首帧时原子关闭连续性，尾帧不影响入边", () => {
    const segment = {
      ...createTimelineSegment("fl2va", 2),
      continuity: { enabled: true, overlap_frames: 39 as const },
    };

    const anchored = assignAssetToSegment(segment, image, "first_image");
    expect(anchored).toMatchObject({
      mode: "fl2va",
      first_image: { id: image.id },
      continuity: { enabled: false, overlap_frames: 39 },
    });

    const tailOnly = assignAssetToSegment(segment, image, "last_image");
    expect(tailOnly).toMatchObject({
      mode: "fl2va",
      last_image: { id: image.id },
      continuity: { enabled: true, overlap_frames: 39 },
    });
  });

  it("Ref2VA 批量自动绑定以首个视频补源，并把独立参考视频限制在剩余两个槽", () => {
    const source = { ...video, id: "video-source", name: "source.mp4" };
    const referenceA = { ...video, id: "video-reference-a", name: "reference-a.mp4" };
    const referenceB = { ...video, id: "video-reference-b", name: "reference-b.mp4" };
    const overflow = { ...video, id: "video-overflow", name: "overflow.mp4" };
    const segment = createTimelineSegment("ref2va", 1);

    const result = assignAssetsToSegment(
      segment,
      [source, image, audio, referenceA, referenceB, overflow],
    );

    expect(result.segment).toMatchObject({
      mode: "ref2va",
      source_video: expect.objectContaining({ id: source.id }),
      reference_images: [expect.objectContaining({ id: image.id, slot: 0 })],
      reference_audios: [expect.objectContaining({ id: audio.id, slot: 0 })],
      reference_videos: [
        expect.objectContaining({ id: referenceA.id, slot: 0 }),
        expect.objectContaining({ id: referenceB.id, slot: 1 }),
      ],
    });
    expect(result.accepted.map((asset) => asset.id)).toEqual([
      source.id,
      image.id,
      audio.id,
      referenceA.id,
      referenceB.id,
    ]);
    expect(result.rejected).toEqual([{ asset: overflow, reason: "capacity" }]);

    const singleOverflow = assignAssetToSegment(result.segment, overflow, "reference_video");
    expect(singleOverflow).toBe(result.segment);
  });

  it("旧 v2/v3 浏览器数据逐字节隔离，clear 不删除人工恢复副本", () => {
    localStorage.clear();
    const legacy = createTimelineProject();
    legacy.title = "待人工判断的旧镜像";
    const legacyRaw = JSON.stringify(legacy);
    const unboundRaw = JSON.stringify({
      format: "director-pending-timeline",
      version: 1,
      pending: true,
      written_at_ms: Date.now(),
      project: legacy,
    });
    localStorage.setItem(LEGACY_TIMELINE_STORAGE_KEY, legacyRaw);
    localStorage.setItem(UNBOUND_TIMELINE_WAL_STORAGE_KEY, unboundRaw);

    expect(loadLocalTimelineWal(ACTIVE_DATABASE)).toBeNull();
    expect(localStorage.getItem(LEGACY_TIMELINE_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(UNBOUND_TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_TIMELINE_STORAGE_KEY)).toBe(legacyRaw);
    expect(localStorage.getItem(QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY)).toBe(unboundRaw);

    clearLocalTimelineWal();
    expect(localStorage.getItem(QUARANTINED_TIMELINE_STORAGE_KEY)).toBe(legacyRaw);
    expect(localStorage.getItem(QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY)).toBe(unboundRaw);
  });

  it("旧 v5 WAL 原样隔离，绝不自动迁移到 revision-aware key", () => {
    localStorage.clear();
    const project = createTimelineProject();
    project.title = "v5 待人工判断";
    const raw = JSON.stringify({
      format: "director-pending-timeline",
      version: 4,
      owner_id: "old-tab",
      pending: true,
      project_id: "default",
      active_database_path: ACTIVE_DATABASE.active_database_path,
      written_at_ms: Date.now(),
      project,
    });
    localStorage.setItem(LEGACY_V5_TIMELINE_WAL_STORAGE_KEY, raw);

    expect(loadLocalTimelineWal(ACTIVE_DATABASE)).toBeNull();
    expect(localStorage.getItem(LEGACY_V5_TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_LEGACY_V5_TIMELINE_WAL_STORAGE_KEY)).toBe(raw);
    expect(readTimelineWalRaw()).toBeNull();
  });

  it("同步保存完整 base/head 权威并使用稳定 canonical document hash", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "尚未确认的新标题" };
    const wal = writeTimelineWal(base, pending);
    const raw = readTimelineWalRaw();

    expect(wal).not.toBeNull();
    expect(raw).not.toBeNull();
    expect(JSON.parse(raw!)).toMatchObject({
      format: TIMELINE_WAL_FORMAT,
      version: TIMELINE_WAL_VERSION,
      owner_id: expect.any(String),
      pending: true,
      project_id: "default",
      active_database_path: ACTIVE_DATABASE.active_database_path,
      base_server_revision: 7,
      base_document_hash: timelineProjectDocumentHash(base),
      head_document_hash: timelineProjectDocumentHash(pending),
      base_project: base,
      pending_project: pending,
    });
    const reordered: TimelineProject = {
      segments: structuredClone(base.segments),
      export_mode: base.export_mode,
      sampling: structuredClone(base.sampling),
      render: structuredClone(base.render),
      title: base.title,
      version: 4,
    };
    expect(timelineProjectDocumentHash(reordered)).toBe(timelineProjectDocumentHash(base));
    expect(loadLocalTimelineWal(ACTIVE_DATABASE)).toEqual(wal);
  });

  it("authority revision 与 exact base 同时匹配时才允许 replay", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "可安全重放" };
    const wal = writeTimelineWal(base, pending)!;
    const raw = readTimelineWalRaw();

    expect(resolveLocalTimelineWal(wal, { revision: 7, document: structuredClone(base) })).toEqual({
      status: "replay",
      project: pending,
      expected_server_revision: 7,
    });
    // Resolution is pure: caller decides when to replay or clear evidence.
    expect(readTimelineWalRaw()).toBe(raw);
  });

  it("revision 或 exact base 不匹配时返回冲突并原地保留取证 WAL", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "本地待同步" };
    const wal = writeTimelineWal(base, pending)!;
    const raw = readTimelineWalRaw();

    const revisionConflict = resolveLocalTimelineWal(wal, {
      revision: 8,
      document: structuredClone(base),
    });
    expect(revisionConflict).toMatchObject({
      status: "conflict",
      reason: "revision-mismatch",
      local_project: pending,
      server_revision: 8,
      base_server_revision: 7,
    });

    const differentBase = { ...structuredClone(base), title: "同 revision 的不同服务端文档" };
    expect(resolveLocalTimelineWal(wal, { revision: 7, document: differentBase })).toMatchObject({
      status: "conflict",
      reason: "base-document-mismatch",
      local_project: pending,
      server_project: differentBase,
    });
    expect(readTimelineWalRaw()).toBe(raw);
  });

  it("server exact 等于 pending head 且 revision 已推进时识别 lost ACK", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "服务端其实已经提交" };
    writeTimelineWal(base, pending);
    const wal = loadLocalTimelineWal(ACTIVE_DATABASE)!;

    expect(resolveLocalTimelineWal(wal, {
      revision: 8,
      document: structuredClone(pending),
    })).toEqual({
      status: "acknowledged",
      project: pending,
      server_revision: 8,
    });
    expect(readTimelineWalRaw()).not.toBeNull();
    clearLocalTimelineWal(wal);
    expect(readTimelineWalRaw()).toBeNull();
  });

  it("lost ACK 只匹配被核对的具体 branch head", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pendingA = { ...structuredClone(base), title: "A 已提交但响应丢失" };
    const pendingB = { ...structuredClone(base), title: "B 仍待处理" };
    const walA = writeTimelineWal(
      base,
      pendingA,
      7,
      "default",
      ACTIVE_DATABASE,
      "tab-lost-ack-a",
    )!;
    const walB = writeTimelineWal(
      base,
      pendingB,
      7,
      "default",
      ACTIVE_DATABASE,
      "tab-lost-ack-b",
    )!;

    expect(resolveLocalTimelineWal(walA, { revision: 8, document: pendingA }).status)
      .toBe("acknowledged");
    expect(resolveLocalTimelineWal(walB, { revision: 8, document: pendingA })).toMatchObject({
      status: "conflict",
      local_project: pendingB,
    });
    expect(listLocalTimelineWalBranches(ACTIVE_DATABASE, "default", "tab-current").foreign)
      .toHaveLength(2);
  });

  it("项目和数据库 scope 使用不透明物理 key，读取其他 scope 不移动任何 bytes", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "仅属于项目 A" };
    writeTimelineWal(base, pending, 3, "project-a", ACTIVE_DATABASE, "tab-owner-a");
    const key = localTimelineWalStorageKey(ACTIVE_DATABASE, "project-a", "tab-owner-a")!;
    const raw = localStorage.getItem(key);

    expect(key).not.toContain(ACTIVE_DATABASE.active_database_path);
    expect(key).not.toContain("project-a");
    expect(loadLocalTimelineWal(ACTIVE_DATABASE, "project-b", "tab-owner-a")).toBeNull();
    expect(loadLocalTimelineWal({
      active_database_path: "/srv/director/data/other.sqlite3".repeat(64),
    }, "project-a", "tab-owner-a")).toBeNull();
    expect(localStorage.getItem(key)).toBe(raw);
  });

  it("Windows 盘符数据库身份的 WAL 正常读写", () => {
    localStorage.clear();
    const windowsDatabase = {
      active_database_path: "D:\\ComfyUI\\user\\directordeck\\database\\directordeck.sqlite3",
    };
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "Windows 待同步" };
    const wal = writeTimelineWal(base, pending, 7, "default", windowsDatabase, "tab-windows");

    expect(wal).not.toBeNull();
    expect(loadLocalTimelineWal(windowsDatabase, "default", "tab-windows")).toEqual(wal);
    clearLocalTimelineWal(wal!);
    expect(loadLocalTimelineWal(windowsDatabase, "default", "tab-windows")).toBeNull();
  });

  it("损坏 hash 或 widened 项目失败封闭且原 bytes 保留为可见证据", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "待同步" };
    writeTimelineWal(base, pending, 7, "default", ACTIVE_DATABASE, "tab-corrupt");
    const key = localTimelineWalStorageKey(ACTIVE_DATABASE, "default", "tab-corrupt")!;
    const corrupt = JSON.parse(localStorage.getItem(key)!) as Record<string, unknown>;
    corrupt.head_document_hash = "fnv1a64:0000000000000000";
    const corruptRaw = JSON.stringify(corrupt);
    localStorage.setItem(key, corruptRaw);

    expect(loadLocalTimelineWal(ACTIVE_DATABASE, "default", "tab-corrupt")).toBeNull();
    expect(localStorage.getItem(key)).toBe(corruptRaw);
    expect(listLocalTimelineWalBranches(ACTIVE_DATABASE, "default", "tab-current").corrupt)
      .toEqual([expect.objectContaining({ storage_key: key, storage_token: corruptRaw })]);

    localStorage.clear();
    writeTimelineWal(base, pending, 7, "default", ACTIVE_DATABASE, "tab-widened");
    const widenedKey = localTimelineWalStorageKey(ACTIVE_DATABASE, "default", "tab-widened")!;
    const widened = JSON.parse(localStorage.getItem(widenedKey)!) as Record<string, unknown>;
    const widenedBase = {
      ...(widened.base_project as TimelineProject),
      unexpected_second_authority: true,
    } as TimelineProject;
    widened.base_project = widenedBase;
    widened.base_document_hash = timelineProjectDocumentHash(widenedBase);
    const widenedRaw = JSON.stringify(widened);
    localStorage.setItem(widenedKey, widenedRaw);

    expect(loadLocalTimelineWal(ACTIVE_DATABASE, "default", "tab-widened")).toBeNull();
    expect(localStorage.getItem(widenedKey)).toBe(widenedRaw);
  });

  it("两个 owner 同 scope 使用独立 branch，逐分支精确 clear 且 stale token 不删新写", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pendingA = { ...structuredClone(base), title: "tab A" };
    const pendingB = { ...structuredClone(base), title: "tab B" };
    const walA = writeTimelineWal(base, pendingA, 7, "default", ACTIVE_DATABASE, "tab-a")!;
    writeTimelineWal(base, pendingB, 7, "default", ACTIVE_DATABASE, "tab-b");
    const keyA = localTimelineWalStorageKey(ACTIVE_DATABASE, "default", "tab-a")!;
    const keyB = localTimelineWalStorageKey(ACTIVE_DATABASE, "default", "tab-b")!;

    const branches = listLocalTimelineWalBranches(ACTIVE_DATABASE, "default", "tab-a");
    expect(branches.owned?.wal.pending_project.title).toBe("tab A");
    expect(branches.foreign.map((branch) => branch.wal.pending_project.title)).toEqual(["tab B"]);
    expect(localStorage.getItem(keyA)).not.toBeNull();
    expect(localStorage.getItem(keyB)).not.toBeNull();

    clearLocalTimelineWal(walA);
    expect(localStorage.getItem(keyA)).toBeNull();
    expect(localStorage.getItem(keyB)).not.toBeNull();

    const staleEvidence = branches.foreign[0]!;
    const newerB = { ...structuredClone(base), title: "tab B newer" };
    writeTimelineWal(base, newerB, 7, "default", ACTIVE_DATABASE, "tab-b");
    expect(discardLocalTimelineWalBranch(staleEvidence)).toBe(false);
    expect(loadLocalTimelineWal(ACTIVE_DATABASE, "default", "tab-b")?.pending_project.title)
      .toBe("tab B newer");
  });

  it("旧 v6 单例只作为 foreign evidence，绝不自动 load 或迁移", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "旧 v6 待恢复" };
    writeTimelineWal(base, pending, 7, "default", ACTIVE_DATABASE, "old-tab");
    const v7Key = localTimelineWalStorageKey(ACTIVE_DATABASE, "default", "old-tab")!;
    const legacyEnvelope = JSON.parse(localStorage.getItem(v7Key)!) as Record<string, unknown>;
    legacyEnvelope.version = 1;
    const legacyRaw = JSON.stringify(legacyEnvelope);
    localStorage.removeItem(v7Key);
    localStorage.setItem(LEGACY_V6_TIMELINE_WAL_STORAGE_KEY, legacyRaw);

    expect(loadLocalTimelineWal(ACTIVE_DATABASE, "default", "old-tab")).toBeNull();
    const branches = listLocalTimelineWalBranches(ACTIVE_DATABASE, "default", "new-tab");
    expect(branches.legacy).toHaveLength(1);
    expect(branches.legacy[0]?.wal.pending_project).toEqual(pending);
    expect(localStorage.getItem(LEGACY_V6_TIMELINE_WAL_STORAGE_KEY)).toBe(legacyRaw);

    clearLocalTimelineWal();
    expect(localStorage.getItem(LEGACY_V6_TIMELINE_WAL_STORAGE_KEY)).toBe(legacyRaw);
    expect(discardLocalTimelineWalBranch(branches.legacy[0]!)).toBe(true);
    expect(localStorage.getItem(LEGACY_V6_TIMELINE_WAL_STORAGE_KEY)).toBeNull();
  });

  it("旧 API 无 base authority 时失败封闭，提供 exact base 时安全委托新 API", () => {
    localStorage.clear();
    const base = createTimelineProject();
    const pending = { ...structuredClone(base), title: "兼容入口编辑" };

    saveLocalTimeline(pending, ACTIVE_DATABASE);
    expect(readTimelineWalRaw()).toBeNull();
    expect(loadLocalTimeline(ACTIVE_DATABASE)).toBeNull();

    saveLocalTimeline(pending, ACTIVE_DATABASE, "default", 11, base);
    expect(loadLocalTimeline(ACTIVE_DATABASE, "default", {
      revision: 11,
      document: base,
    })).toEqual(pending);
    clearLocalTimeline();
    expect(readTimelineWalRaw()).toBeNull();
  });
});
