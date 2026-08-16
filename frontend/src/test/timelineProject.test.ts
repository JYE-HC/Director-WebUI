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
  createTimelineEditorState,
  createTimelineProject,
  createTimelineSegment,
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
  loadLocalTimeline,
  normalizeTimelineProject,
  orderAssetsByPreference,
  promptSubjectReferences,
  QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY,
  QUARANTINED_TIMELINE_STORAGE_KEY,
  QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY,
  removeAssetFromSegment,
  reorderSegmentReference,
  saveLocalTimeline,
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
  TIMELINE_WAL_STORAGE_KEY,
  UNBOUND_TIMELINE_WAL_STORAGE_KEY,
  updateRef2VASourceRange,
  runnableTimelineSegmentIds,
  validateTimelineProject,
} from "../domain/timelineProject";

const image: AssetReference = {
  id: "image-a",
  name: "image.png",
  subfolder: "director-web",
  type: "input",
  kind: "image",
  preview_url: "/api/assets/image-a/preview",
};

const audio: AssetReference = {
  id: "audio-a",
  name: "voice.wav",
  subfolder: "director-web",
  type: "input",
  kind: "audio",
};

const video: AssetReference = {
  id: "video-a",
  name: "source.mp4",
  subfolder: "director-web",
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
  active_database_identity: "a".repeat(64),
};

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
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after" });
    const second = state.project.segments[1].id;
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after" });
    const third = state.project.segments[2].id;
    state = timelineEditorReducer(state, { type: "segment/select", id: first });
    expect(state.run_selected_segment_ids).toEqual([first]);
    state = timelineEditorReducer(state, { type: "segment/select", id: third, range: true });
    expect(state.selected_segment_ids).toEqual([first, second, third]);
    expect(state.run_selected_segment_ids).toEqual([first, second, third]);
    state = timelineEditorReducer(state, { type: "segment/move", draggedId: third, targetId: first });
    expect(state.project.segments.map((segment) => segment.id)).toEqual([third, first, second]);
    expect(state.selected_segment_ids).toEqual([first, second, third]);
    expect(state.run_selected_segment_ids).toEqual([first, second, third]);
  });

  it("在片段 02 前插空段时分配新的默认编号且保留已有名称", () => {
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after" });
    expect(state.project.segments.map((segment) => segment.title)).toEqual(["片段 01", "片段 02"]);

    state = timelineEditorReducer(state, { type: "segment/insert", position: "before" });

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
      selection_anchor_id: segments[3].id,
      run_selected_segment_ids: segments.map((segment) => segment.id),
    };

    const before = timelineEditorReducer(initial, { type: "segment/insert", position: "before" });
    const after = timelineEditorReducer(initial, { type: "segment/insert", position: "after" });

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
    state = timelineEditorReducer(state, { type: "segment/split-selected" });
    const [left, right] = state.project.segments;
    expect(left).toMatchObject({ mode: "ref2va", duration_seconds: 2, source_start_seconds: 2, source_duration_seconds: 3 });
    expect(right).toMatchObject({ mode: "ref2va", duration_seconds: 6, source_start_seconds: 5, source_duration_seconds: 9 });
    expect(right.id).not.toBe(left.id);
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
    const split = splitTimelineSourceSegmentAtCuts(state, source.id, [0, 5 * 24, 9 * 24, 20 * 24], 24, expected);
    expect(split.project.segments).toHaveLength(3);
    expect(split.project.segments).toEqual([
      expect.objectContaining({ id: source.id, mode: "ref2va", source_start_seconds: 2, source_duration_seconds: 3, duration_seconds: 2.5, reference_images: source.reference_images }),
      expect.objectContaining({ mode: "ref2va", source_start_seconds: 5, source_duration_seconds: 4, duration_seconds: 10 / 3, reference_images: source.reference_images }),
      expect.objectContaining({ mode: "ref2va", source_start_seconds: 9, source_duration_seconds: 5, duration_seconds: 25 / 6, reference_images: source.reference_images }),
    ]);
    expect(split.project.segments[1].id).not.toBe(source.id);
    expect(split.selected_segment_ids).toEqual(split.project.segments.map((segment) => segment.id));

    const stale = splitTimelineSourceSegmentAtCuts(state, source.id, [5 * 24], 24, {
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

    const split = splitTimelineSourceSegmentEvenly(state, source.id, 4);
    expect(split.project.segments).toHaveLength(4);
    expect(split.project.segments.map((segment) => segment.mode === "ref2va" ? [
      segment.source_start_seconds,
      segment.source_duration_seconds,
    ] : null)).toEqual([[2, 2.5], [4.5, 2.5], [7, 2.5], [9.5, 2.5]]);
    expect(split.project.segments.every((segment) =>
      segment.mode === "ref2va" && segment.reference_images[0]?.id === image.id,
    )).toBe(true);
    expect(split.selected_segment_ids).toEqual(split.project.segments.map((segment) => segment.id));

    const oddFrameSource = {
      ...source,
      source_duration_seconds: 241 / 24,
    };
    const oddFrameState = {
      ...state,
      project: { ...state.project, segments: [oddFrameSource] },
    };
    const oddFrameSplit = splitTimelineSourceSegmentEvenly(oddFrameState, source.id, 4);
    expect(oddFrameSplit.project.segments.map((segment) =>
      segment.mode === "ref2va" ? Math.round(segment.source_duration_seconds * 24) : 0,
    )).toEqual([61, 60, 60, 60]);

    expect(splitTimelineSourceSegmentEvenly(state, source.id, 49)).toBe(state);
    expect(splitTimelineSourceSegmentEvenly(state, source.id, 1)).toBe(state);
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
    state = timelineEditorReducer(state, { type: "segment/set-selection", ids: [] });
    expect(state.selected_segment_ids).toEqual([]);
    expect(state.selection_anchor_id).toBeNull();
  });

  it("片段点击原子同步编辑选择和运行范围，启停仍按运行意图裁剪", () => {
    let state = createTimelineEditorState();
    const first = state.project.segments[0].id;
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after", mode: "ref2va" });
    const second = state.project.segments[1].id;
    expect(runnableTimelineSegmentIds(state)).toEqual([first, second]);

    state = timelineEditorReducer(state, { type: "segment/select", id: first });
    expect(state.selected_segment_ids).toEqual([first]);
    expect(state.run_selected_segment_ids).toEqual([first]);
    state = timelineEditorReducer(state, { type: "segment/select", id: second, additive: true });
    expect(state.selected_segment_ids).toEqual([first, second]);
    expect(state.run_selected_segment_ids).toEqual([first, second]);

    state = timelineEditorReducer(state, {
      type: "segment/set-enabled",
      ids: [second],
      enabled: false,
    });
    expect(runnableTimelineSegmentIds(state)).toEqual([first]);
    expect(state.run_selected_segment_ids).toEqual([first]);
    expect(state.project.segments[1].enabled).toBe(false);

    state = timelineEditorReducer(state, { type: "segment/select", id: second });
    expect(state.selected_segment_ids).toEqual([second]);
    expect(state.run_selected_segment_ids).toEqual([first]);

    state = timelineEditorReducer(state, { type: "segment/set-selection", ids: [first, second] });
    state = timelineEditorReducer(state, { type: "segment/set-run-selection", ids: [] });
    expect(state.selected_segment_ids).toEqual([first, second]);
    expect(state.run_selected_segment_ids).toEqual([]);

    let fallback = createTimelineEditorState();
    fallback = timelineEditorReducer(fallback, { type: "segment/delete-selected" });
    expect(runnableTimelineSegmentIds(fallback)).toEqual([fallback.project.segments[0].id]);
  });

  it("无共享 ID 的项目替换保持单一编辑焦点，同时默认运行全部启用段", () => {
    const first = createTimelineSegment("fl2va", 1);
    const second = createTimelineSegment("ref2va", 2);
    const replacement = { ...createTimelineProject(), segments: [first, second] };

    const state = timelineEditorReducer(createTimelineEditorState(), {
      type: "project/replace",
      project: replacement,
    });

    expect(state.selected_segment_ids).toEqual([first.id]);
    expect(state.run_selected_segment_ids).toEqual([first.id, second.id]);
  });

  it("复制片段逐一继承各自源片段的运行勾选", () => {
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after" });
    const [first, second] = state.project.segments.map((segment) => segment.id);
    state = timelineEditorReducer(state, { type: "segment/set-run-selection", ids: [second] });
    state = timelineEditorReducer(state, { type: "segment/set-selection", ids: [first, second] });
    state = timelineEditorReducer(state, { type: "segment/duplicate-selected" });

    const copiedIds = state.selected_segment_ids;
    expect(copiedIds).toHaveLength(2);
    expect(runnableTimelineSegmentIds(state)).toEqual([second, copiedIds[1]]);
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
    state = applyTimelineSegmentConfiguration(state, source.id, "following");
    expect(state.project.segments[1]).toMatchObject({
      id: target.id, title: "保留名称", enabled: false, mode: "ref2va", prompt: "重绘",
    });
    expect(state.project.segments[1]).not.toHaveProperty("first_image");
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
    state = { ...state, project: { ...state.project, segments: [first, second] }, selected_segment_ids: [first.id, second.id], selection_anchor_id: first.id, run_selected_segment_ids: [second.id] };
    expect(canMergeSelectedSegments(state)).toBe(true);
    state = timelineEditorReducer(state, { type: "segment/merge-selected" });
    expect(state.project.segments).toHaveLength(1);
    expect(state.project.segments[0]).toMatchObject({ duration_seconds: 8, source_duration_seconds: 8 });
    expect(runnableTimelineSegmentIds(state)).toEqual([first.id]);

    const incompatible = createTimelineEditorState();
    const a = { ...createTimelineSegment("fl2va", 1), first_image: image };
    const b = { ...createTimelineSegment("fl2va", 2), first_image: { ...image, id: "different-image" } };
    const flState = { ...incompatible, project: { ...incompatible.project, segments: [a, b] }, selected_segment_ids: [a.id, b.id] };
    expect(canMergeSelectedSegments(flState)).toBe(false);
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

  it("旧 v2 长期镜像隔离后不作为 pending WAL 回放，且 clear 不删除人工恢复副本", () => {
    localStorage.clear();
    const legacy = createTimelineProject();
    legacy.title = "待人工判断的旧镜像";
    const raw = JSON.stringify(legacy);
    localStorage.setItem(LEGACY_TIMELINE_STORAGE_KEY, raw);

    expect(loadLocalTimeline(ACTIVE_DATABASE)).toBeNull();
    expect(localStorage.getItem(LEGACY_TIMELINE_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_TIMELINE_STORAGE_KEY)).toBe(raw);

    const pending = createTimelineProject();
    pending.title = "明确待同步的新 WAL";
    saveLocalTimeline(pending, ACTIVE_DATABASE);
    expect(loadLocalTimeline(ACTIVE_DATABASE)).toMatchObject({ title: "明确待同步的新 WAL" });
    clearLocalTimeline();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_TIMELINE_STORAGE_KEY)).toBe(raw);
  });

  it("无数据库身份的 v3 WAL 原样隔离且永不回放", () => {
    localStorage.clear();
    const project = createTimelineProject();
    const raw = JSON.stringify({
      format: "director-pending-timeline",
      version: 1,
      pending: true,
      written_at_ms: Date.now(),
      project,
    });
    localStorage.setItem(UNBOUND_TIMELINE_WAL_STORAGE_KEY, raw);

    expect(loadLocalTimeline(ACTIVE_DATABASE)).toBeNull();
    expect(localStorage.getItem(UNBOUND_TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY)).toBe(raw);
    clearLocalTimeline();
    expect(localStorage.getItem(QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY)).toBe(raw);
  });

  it("新 WAL 只恢复数据库路径与身份都精确匹配的 pending envelope", () => {
    localStorage.clear();
    const project = createTimelineProject();
    project.title = "数据库 A 的待同步时间线";
    saveLocalTimeline(project, ACTIVE_DATABASE);
    const raw = localStorage.getItem(TIMELINE_WAL_STORAGE_KEY);
    expect(raw).not.toBeNull();
    expect(JSON.parse(raw!)).toMatchObject({
      version: 4,
      owner_id: expect.any(String),
      pending: true,
      project_id: "default",
      active_database_path: ACTIVE_DATABASE.active_database_path,
      active_database_identity: ACTIVE_DATABASE.active_database_identity,
    });
    expect(loadLocalTimeline(ACTIVE_DATABASE)).toMatchObject({ title: project.title });

    const otherDatabase = {
      active_database_path: "/srv/director/data/other.sqlite3",
      active_database_identity: "b".repeat(64),
    };
    expect(loadLocalTimeline(otherDatabase)).toBeNull();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY)).toBe(raw);
  });

  it("损坏的新 WAL 只隔离，不参与恢复", () => {
    localStorage.clear();
    const project = createTimelineProject();
    localStorage.setItem(TIMELINE_WAL_STORAGE_KEY, JSON.stringify({
      format: "director-pending-timeline",
      version: 2,
      pending: false,
      active_database_path: ACTIVE_DATABASE.active_database_path,
      active_database_identity: ACTIVE_DATABASE.active_database_identity,
      written_at_ms: Date.now(),
      project,
    }));
    const raw = localStorage.getItem(TIMELINE_WAL_STORAGE_KEY);
    expect(loadLocalTimeline(ACTIVE_DATABASE)).toBeNull();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY)).toBe(raw);
  });

  it("隔离槽已有另一份 WAL 时仍为新 WAL 创建副本后再删除源键", () => {
    localStorage.clear();
    const firstQuarantine = "first-database-wal";
    localStorage.setItem(QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY, firstQuarantine);
    const project = createTimelineProject();
    project.title = "第二个数据库的 WAL";
    saveLocalTimeline(project, ACTIVE_DATABASE);
    const raw = localStorage.getItem(TIMELINE_WAL_STORAGE_KEY);

    expect(loadLocalTimeline({
      active_database_path: "/srv/director/data/third.sqlite3",
      active_database_identity: "c".repeat(64),
    })).toBeNull();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY)).toBe(firstQuarantine);
    expect(localStorage.getItem(`${QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY}:1`)).toBe(raw);
  });

  it("旧数据库页面保存前先无损隔离另一数据库页面的 pending WAL", () => {
    localStorage.clear();
    const databaseB = {
      active_database_path: "/srv/director/data/database-b.sqlite3",
      active_database_identity: "b".repeat(64),
    };
    const projectB = createTimelineProject();
    projectB.title = "数据库 B 未同步项目";
    const rawB = JSON.stringify({
      format: "director-pending-timeline",
      version: 3,
      owner_id: "tab-b",
      pending: true,
      active_database_path: databaseB.active_database_path,
      active_database_identity: databaseB.active_database_identity,
      written_at_ms: Date.now(),
      project: projectB,
    });
    localStorage.setItem(TIMELINE_WAL_STORAGE_KEY, rawB);
    const projectA = createTimelineProject();
    projectA.title = "旧数据库 A 页面修改";

    saveLocalTimeline(projectA, ACTIVE_DATABASE);

    expect(localStorage.getItem(QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY)).toBe(rawB);
    expect(JSON.parse(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)!)).toMatchObject({
      active_database_identity: ACTIVE_DATABASE.active_database_identity,
      project: { title: projectA.title },
    });
  });

  it("同数据库另一 tab 的 WAL 会先隔离，旧请求清理也不会删除后来写入的 WAL", () => {
    localStorage.clear();
    const otherTabProject = createTimelineProject();
    otherTabProject.title = "同库另一个标签页";
    const otherTabRaw = JSON.stringify({
      format: "director-pending-timeline",
      version: 3,
      owner_id: "other-tab",
      pending: true,
      active_database_path: ACTIVE_DATABASE.active_database_path,
      active_database_identity: ACTIVE_DATABASE.active_database_identity,
      written_at_ms: Date.now(),
      project: otherTabProject,
    });
    localStorage.setItem(TIMELINE_WAL_STORAGE_KEY, otherTabRaw);
    const currentProject = createTimelineProject();
    currentProject.title = "当前标签页";
    saveLocalTimeline(currentProject, ACTIVE_DATABASE);
    expect(localStorage.getItem(QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY)).toBe(otherTabRaw);

    const laterOtherTabRaw = JSON.stringify({
      ...JSON.parse(otherTabRaw),
      written_at_ms: Date.now() + 1,
      project: { ...otherTabProject, title: "稍后写入的另一标签页" },
    });
    localStorage.setItem(TIMELINE_WAL_STORAGE_KEY, laterOtherTabRaw);
    clearLocalTimeline();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBe(laterOtherTabRaw);
  });

  it("v5 WAL 按项目隔离：只恢复匹配项目的 pending 编辑", () => {
    localStorage.clear();
    const project = createTimelineProject();
    project.title = "项目 A 的待同步编辑";
    saveLocalTimeline(project, ACTIVE_DATABASE, "project-a");
    expect(loadLocalTimeline(ACTIVE_DATABASE, "project-a")).toMatchObject({
      title: "项目 A 的待同步编辑",
    });
    // 另一个项目绝不能拿到 A 的 pending 编辑，且会被隔离而不是静默回放。
    expect(loadLocalTimeline(ACTIVE_DATABASE, "project-b")).toBeNull();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(
      localStorage.getItem(QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY),
    ).not.toBeNull();
  });
});
