import { act, createEvent, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useReducer, type ComponentProps } from "react";
import { directorApi } from "../api/client";
import {
  EMPTY_CAPABILITIES,
  type CapabilityReport,
  type GenerationTask,
  type TimelineCompileReport,
} from "../api/types";
import { LongFormTimelineWorkspace } from "../components/LongFormTimelineWorkspace";
import type { AssetReference } from "../domain/modes";
import {
  alignedTimelineSegmentDuration,
  autoFitSourceAudioTiming,
  createTimelineEditorState,
  createTimelineSegment,
  EMPTY_SIX_SECTION_PROMPT,
  sourcePreviewTime,
  timelineEditorReducer,
  type TimelineAction,
  type TimelineEditorState,
} from "../domain/timelineProject";
import { loadTimelineWorkspacePreferences } from "../domain/workspacePreferences";

const image: AssetReference = {
  id: "longform-image",
  name: "角色参考.png",
  subfolder: "director",
  type: "input",
  kind: "image",
  preview_url: "/api/assets/longform-image/preview",
};

const video: AssetReference = {
  id: "longform-video",
  name: "源视频.mp4",
  subfolder: "director",
  type: "input",
  kind: "video",
  preview_url: "/api/assets/longform-video/preview",
  metadata: {
    duration: 30,
    native_fps: 24,
    frame_count: 720,
    width: 1920,
    height: 1080,
    probe_method: "ffprobe",
    has_audio: true,
  },
};

const replacementVideo: AssetReference = {
  ...video,
  id: "longform-video-replacement",
  name: "替换机位.mp4",
};

const capabilities = {
  ...EMPTY_CAPABILITIES,
  connection: "online" as const,
  supports_cancel: true,
};

function commonProps(state: TimelineEditorState) {
  return {
    state,
    capabilities,
    activeTask: null,
    segmentCandidates: {},
    compileReport: null,
    validationErrors: [],
    selectionValidationErrors: [],
    onCloseCompile: () => undefined,
    onCancelTask: () => undefined,
  };
}

function timelineEditorReducerWithSourceAudioFit(
  state: TimelineEditorState,
  action: TimelineAction,
): TimelineEditorState {
  const reduced = timelineEditorReducer(state, action);
  const fitted = autoFitSourceAudioTiming(reduced.project);
  return fitted.project === reduced.project
    ? reduced
    : { ...reduced, project: fitted.project };
}

function Harness({
  initial,
  workspaceCapabilities = capabilities,
  onUploadFiles,
  segmentCandidates = {},
  fitSourceAudio = false,
}: {
  initial: TimelineEditorState;
  workspaceCapabilities?: CapabilityReport;
  onUploadFiles?: ComponentProps<typeof LongFormTimelineWorkspace>["onUploadFiles"];
  segmentCandidates?: ComponentProps<typeof LongFormTimelineWorkspace>["segmentCandidates"];
  fitSourceAudio?: boolean;
}) {
  const [state, dispatch] = useReducer(
    fitSourceAudio ? timelineEditorReducerWithSourceAudioFit : timelineEditorReducer,
    initial,
  );
  return <>
    <LongFormTimelineWorkspace {...commonProps(state)} capabilities={workspaceCapabilities} segmentCandidates={segmentCandidates} onDispatch={dispatch} onUploadFiles={onUploadFiles} />
    <pre data-testid="timeline-state">{JSON.stringify(state)}</pre>
  </>;
}

function readState(): TimelineEditorState {
  return JSON.parse(screen.getByTestId("timeline-state").textContent ?? "null") as TimelineEditorState;
}

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe("统一时间线关键交互", () => {
  it("预检计划区分最终可见帧与内部采样帧，并显示接续前驱和裁剪开销", () => {
    const state = createTimelineEditorState();
    const first = { ...state.project.segments[0], prompt: "第一段" };
    const second = { ...createTimelineSegment("fl2va", 2), prompt: "第二段" };
    state.project.segments = [first, second];
    const report: TimelineCompileReport = {
      execution_strategy: "native_segment_graph_v1",
      model_families: ["fl2va"],
      plans: [{
        segment_id: second.id,
        mode: "fl2va",
        recipe: "t2v",
        model_family: "fl2va",
        backend: "raylight",
        frame_count: 124,
        visible_frame_count: 124,
        sample_frame_count: 158,
        continuity_context_frames: 22,
        alignment_tail_frame_count: 12,
        predecessor_segment_id: first.id,
        continuity_source: "historical_take",
        historical_take_id: "take-first-v1",
        anchor_reset: false,
        seed_mode: "fixed",
        seed: 7,
        conditioning_node: "MiniMaxH3ImageToVideo",
        node_classes: ["MiniMaxH3AddGuide", "SamplerCustomAdvanced"],
      }],
      node_policy: {
        graph_source: "server",
        accepts_client_workflow: false,
        allowed_nodes: ["MiniMaxH3AddGuide", "SamplerCustomAdvanced"],
        custom_nodes: [],
        provenance: {
          MiniMaxH3AddGuide: "comfy-core-official-minimax-h3",
          SamplerCustomAdvanced: "comfy-extras",
        },
      },
    };

    render(<LongFormTimelineWorkspace
      {...commonProps(state)}
      compileReport={report}
      onDispatch={() => undefined}
    />);

    const plan = screen.getByRole("region", { name: "服务端执行计划" });
    expect(screen.getByRole("article", { name: "执行计划 2 · 片段 02" })).toBeInTheDocument();
    expect(plan).toHaveTextContent("2 · 片段 02 · FL2VA");
    expect(plan).toHaveTextContent("124f 可见 / 158f 采样");
    expect(plan).toHaveTextContent("前驱 1 · 片段 01");
    expect(plan).toHaveTextContent("来源：复用前驱成片");
    expect(plan).toHaveTextContent("take take-fir");
    expect(plan).toHaveTextContent("接续上下文 22f");
    expect(plan).toHaveTextContent("对齐尾帧 12f");
    expect(plan).toHaveTextContent("RayLight");
  });

  it("完整源视频在主轨按完整时长显示，并绘制等距关键帧缩略图", () => {
    const state = createTimelineEditorState();
    const shortSource = {
      ...video,
      metadata: { ...video.metadata!, duration: 20, frame_count: 480 },
    };
    state.project = {
      ...state.project,
      segments: [{
        ...createTimelineSegment("ref2va", 1),
        id: state.project.segments[0].id,
        prompt: "编辑源片",
        duration_seconds: 20,
        source_video: shortSource,
        source_start_seconds: 0,
        source_duration_seconds: 20,
      }],
    };
    render(<LongFormTimelineWorkspace {...commonProps(state)} onDispatch={() => undefined} />);

    const clip = document.querySelector<HTMLElement>(".timeline-clip");
    expect(clip).not.toBeNull();
    expect(clip).toHaveAttribute(
      "data-duration-seconds",
      "20",
    );
    const filmstrip = screen.getByLabelText(`${shortSource.name} 关键帧缩略图`);
    expect(filmstrip).toBeInTheDocument();
    expect(screen.getAllByRole("img", { name: /源视频关键帧/ })).toHaveLength(4);
    expect(filmstrip.querySelectorAll(".timeline-source-filmstrip__frame")).toHaveLength(4);
    expect(filmstrip.querySelectorAll(".timeline-source-filmstrip__backdrop")).toHaveLength(4);
    expect(filmstrip.querySelectorAll(".timeline-source-filmstrip__image")).toHaveLength(4);
    expect(screen.getByText(/源 0\.00–20\.00s/)).toBeInTheDocument();
  });

  it("按 MiniMax H3 能力显示各参考区上限，源视频动态占用一条视频容量", () => {
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      source_video: video,
      reference_images: [{ ...image, slot: 0 }],
      reference_videos: [{ ...replacementVideo, slot: 0 }],
    }];
    render(<Harness initial={state} />);

    const sourceRegion = screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" });
    expect(sourceRegion).toHaveTextContent("1/1");
    expect(sourceRegion).not.toHaveTextContent("源视频占用 MiniMax H3");
    expect(screen.getByRole("region", { name: "参考图片" })).toHaveTextContent("1/9");
    expect(screen.getByRole("region", { name: "参考视频" })).toHaveTextContent("1/2");
    expect(screen.getByRole("region", { name: "参考音频" })).toHaveTextContent("0/3");
    expect(screen.getByRole("region", { name: "参考视频" })).toHaveTextContent(
      "与源视频共享 MiniMax H3 的 3 路视频容量；当前视频总计 2/3",
    );
  });

  it("参考素材区为视频素材渲染视频缩略图而非仅占位图标", () => {
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      source_video: video,
      reference_videos: [{ ...replacementVideo, slot: 0 }],
    }];
    render(<Harness initial={state} />);

    const sourceZone = screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" });
    expect(sourceZone).toHaveClass("segment-reference-grid--source");
    const sourceCard = sourceZone.querySelector("article");
    expect(sourceCard).toHaveTextContent("<Video 1>");
    expect(sourceCard).toHaveTextContent(video.name);
    const sourceThumb = sourceZone.querySelector("video");
    expect(sourceThumb).not.toBeNull();
    expect(sourceThumb).toHaveAttribute("src", video.preview_url);

    const referenceZone = screen.getByRole("region", { name: "参考视频" });
    const referenceThumb = referenceZone.querySelector("video");
    expect(referenceThumb).not.toBeNull();
    expect(referenceThumb).toHaveAttribute("src", replacementVideo.preview_url);
  });

  it("FL2VA 首尾帧显示并可插入对应 Picture 标签", () => {
    const first = { ...image, id: "fl-first", name: "首帧.png" };
    const last = { ...image, id: "fl-last", name: "尾帧.png" };
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("fl2va", 1),
      id: state.project.segments[0].id,
      prompt: "从 <Picture 1> 过渡到 <Picture 2>",
      first_image: first,
      last_image: last,
    }];
    render(<Harness initial={state} />);

    expect(screen.getByRole("region", { name: "首帧（可选）" })).toHaveTextContent(
      "<Picture 1>",
    );
    expect(screen.getByRole("region", { name: "尾帧（设置后使用首尾帧配方）" })).toHaveTextContent(
      "<Picture 2>",
    );
    expect(screen.getByRole("button", { name: "<Picture 1>" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "<Picture 2>" })).toBeInTheDocument();
    expect(screen.queryByText(/未绑定的素材标签/)).not.toBeInTheDocument();
  });

  it("参考图片区内部拖动只重排标签，不增加引用数量", () => {
    const first = { ...image, id: "reorder-first", name: "第一张.png", slot: 0 };
    const second = { ...image, id: "reorder-second", name: "第二张.png", slot: 1 };
    const third = { ...image, id: "reorder-third", name: "第三张.png", slot: 2 };
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: "人物 <Picture 1>，衣着 <Picture 2>，场景 <Picture 3>",
      reference_images: [first, second, third],
    }];
    render(<Harness initial={state} workspaceCapabilities={{ ...capabilities, connection: "offline" }} />);

    const zone = screen.getByRole("region", { name: "参考图片" });
    expect(zone).toHaveClass("segment-reference-grid--content-sized");
    const firstCard = zone.querySelector<HTMLElement>('[data-reference-asset-id="reorder-first"]')!;
    const thirdCard = zone.querySelector<HTMLElement>('[data-reference-asset-id="reorder-third"]')!;
    const values = new Map<string, string>();
    const dataTransfer = {
      types: ["application/x-director-segment-reference"],
      getData: vi.fn((type: string) => values.get(type) ?? ""),
      setData: vi.fn((type: string, value: string) => values.set(type, value)),
      dropEffect: "none",
      effectAllowed: "copyMove",
      files: [],
    };
    const bindSpy = vi.spyOn(dataTransfer, "getData");

    fireEvent.dragStart(thirdCard, { dataTransfer });
    expect(dataTransfer.setData).toHaveBeenCalledWith(
      "application/x-director-segment-reference",
      expect.any(String),
    );
    fireEvent.dragOver(firstCard, { dataTransfer });
    expect(dataTransfer.dropEffect).toBe("move");
    fireEvent.drop(firstCard, { dataTransfer });

    const movedThirdCard = zone.querySelector<HTMLElement>('[data-reference-asset-id="reorder-third"]')!;
    expect(movedThirdCard).not.toHaveClass("is-dragging");

    const segment = readState().project.segments[0];
    expect(segment).toMatchObject({
      mode: "ref2va",
      prompt: "人物 <Picture 2>，衣着 <Picture 3>，场景 <Picture 1>",
    });
    if (segment.mode !== "ref2va") throw new Error("fixture must remain ref2va");
    expect(segment.reference_images).toHaveLength(3);
    expect(segment.reference_images.map(({ id, slot }) => ({ id, slot }))).toEqual([
      { id: third.id, slot: 0 },
      { id: first.id, slot: 1 },
      { id: second.id, slot: 2 },
    ]);
    expect(bindSpy).not.toHaveBeenCalledWith("application/x-director-asset");

    dataTransfer.setData.mockClear();
    const removeButton = screen.getByRole("button", { name: "从片段移除 第三张.png" });
    fireEvent.dragStart(removeButton, { dataTransfer });
    expect(dataTransfer.setData).not.toHaveBeenCalled();
  });

  it("九张参考图片完整展开，不启用图片区内部滚动", () => {
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      reference_images: Array.from({ length: 9 }, (_, slot) => ({
        ...image,
        id: `expanded-picture-${slot}`,
        name: `参考图片 ${slot + 1}.png`,
        slot,
      })),
    }];
    render(<Harness initial={state} />);

    const zone = screen.getByRole("region", { name: "参考图片" });
    expect(zone).toHaveTextContent("9/9");
    expect(zone).toHaveClass("segment-reference-grid--content-sized");
    expect(zone.querySelectorAll("article")).toHaveLength(9);
  });

  it("没有源视频时参考视频显示三路容量", () => {
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
    }];
    render(<Harness initial={state} />);

    expect(screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" })).toHaveTextContent("0/1");
    expect(screen.getByRole("region", { name: "参考视频" })).toHaveTextContent("0/3");
  });

  it("三路独立参考视频已满时站内源视频拖入不可复制并给出具体反馈", async () => {
    const state = createTimelineEditorState();
    const references = [video, replacementVideo, {
      ...video,
      id: "third-reference-video",
      name: "第三机位.mp4",
    }];
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      reference_videos: references.map((asset, slot) => ({ ...asset, slot })),
    }];
    const incoming = { ...video, id: "new-source", name: "新源视频.mp4" };
    state.assets = [incoming];
    render(<Harness initial={state} />);

    const sourceZone = screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" });
    const dataTransfer = {
      types: ["application/x-director-asset", "application/x-director-asset-video"],
      getData: vi.fn((type: string) => type === "application/x-director-asset" ? incoming.id : ""),
      setData: vi.fn(),
      dropEffect: "copy",
      effectAllowed: "copyMove",
      files: [],
    };

    expect(fireEvent.dragOver(sourceZone, { dataTransfer })).toBe(false);
    expect(dataTransfer.dropEffect).toBe("none");
    fireEvent.drop(sourceZone, { dataTransfer });

    expect(readState().project.segments[0]).toMatchObject({
      source_video: null,
      reference_videos: references.map((asset, slot) => ({ id: asset.id, slot })),
    });
    expect(await screen.findByText(/当前独立参考视频已占满，需先移除 1 个/)).toBeInTheDocument();
  });

  it("参考图片区满额时外部文件仍上传入素材库但不绑定", async () => {
    const state = createTimelineEditorState();
    const references = Array.from({ length: 9 }, (_, slot) => ({
      ...image,
      id: `filled-image-${slot}`,
      name: `参考图-${slot}.png`,
      slot,
    }));
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      reference_images: references,
    }];
    const overflow = { ...image, id: "external-overflow", name: "额外参考.png" };
    const onUploadFiles = vi.fn().mockResolvedValue({
      assets: [overflow],
      failures: [],
      authority_stale: false,
    });
    render(<Harness initial={state} onUploadFiles={onUploadFiles} />);
    const zone = screen.getByRole("region", { name: "参考图片" });
    const file = new File(["image"], "额外参考.png", { type: "image/png" });
    const dataTransfer = {
      types: ["Files"],
      getData: vi.fn(() => ""),
      setData: vi.fn(),
      dropEffect: "none",
      effectAllowed: "copy",
      files: [file],
    };

    expect(fireEvent.dragOver(zone, { dataTransfer })).toBe(false);
    expect(dataTransfer.dropEffect).toBe("copy");
    fireEvent.drop(zone, { dataTransfer });

    await waitFor(() => expect(onUploadFiles).toHaveBeenCalledWith([file]));
    await waitFor(() => expect(readState().assets).toContainEqual(overflow));
    expect(readState().project.segments[0]).toMatchObject({ reference_images: references });
    expect(screen.getByText(
      "MiniMax H3 最多支持 9 张参考图片；1 项仅加入素材库",
    )).toBeInTheDocument();
  });

  it("把侧栏完整多选集拖入 Ref2VA 后按视频、图片和音频分类绑定", () => {
    const state = createTimelineEditorState();
    const audio: AssetReference = {
      id: "longform-audio",
      name: "对白.wav",
      subfolder: "director",
      type: "input",
      kind: "audio",
    };
    state.project.segments = [{ ...createTimelineSegment("ref2va", 1), id: state.project.segments[0].id }];
    state.assets = [video, image, audio, replacementVideo];
    state.selected_asset_ids = state.assets.map((asset) => asset.id);
    render(<Harness initial={state} />);

    const inspector = screen.getByRole("region", { name: "当前片段编辑器" });
    const dataTransfer = {
      types: ["application/x-director-assets", "application/x-director-asset"],
      getData: vi.fn((type: string) => type === "application/x-director-assets"
        ? JSON.stringify(state.selected_asset_ids)
        : type === "application/x-director-asset" ? video.id : ""),
      setData: vi.fn(),
      dropEffect: "none",
      effectAllowed: "copyMove",
      files: [],
    };
    fireEvent.dragOver(inspector, { dataTransfer });
    fireEvent.drop(inspector, { dataTransfer });

    expect(readState().project.segments[0]).toMatchObject({
      mode: "ref2va",
      title: "源视频",
      source_video: { id: video.id },
      duration_seconds: 30,
      source_start_seconds: 0,
      source_duration_seconds: 30,
      reference_images: [{ id: image.id, slot: 0 }],
      reference_audios: [{ id: audio.id, slot: 0 }],
      reference_videos: [{ id: replacementVideo.id, slot: 0 }],
    });
  });

  it("纯绑定成功提示三秒后消失，重复或失败提示持续保留", () => {
    vi.useFakeTimers();
    try {
      const state = createTimelineEditorState();
      state.assets = [image];
      render(<Harness initial={state} />);
      const inspector = screen.getByRole("region", { name: "当前片段编辑器" });
      const dataTransfer = {
        types: ["application/x-director-asset"],
        getData: vi.fn((type: string) => type === "application/x-director-asset" ? image.id : ""),
        setData: vi.fn(),
        dropEffect: "none",
        effectAllowed: "copyMove",
        files: [],
      };

      fireEvent.dragOver(inspector, { dataTransfer });
      fireEvent.drop(inspector, { dataTransfer });
      expect(screen.getByText("已绑定 1 项")).toBeInTheDocument();
      act(() => vi.advanceTimersByTime(2_999));
      expect(screen.getByText("已绑定 1 项")).toBeInTheDocument();
      act(() => vi.advanceTimersByTime(1));
      expect(screen.queryByText("已绑定 1 项")).not.toBeInTheDocument();

      fireEvent.drop(inspector, { dataTransfer });
      expect(screen.getByText("1 项已存在")).toBeInTheDocument();
      act(() => vi.advanceTimersByTime(10_000));
      expect(screen.getByText("1 项已存在")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("外部混合文件拖入片段后一步上传、入库并分类绑定，失败项可见", async () => {
    const state = createTimelineEditorState();
    state.project.segments = [{ ...createTimelineSegment("ref2va", 1), id: state.project.segments[0].id }];
    const uploadedAudio: AssetReference = {
      id: "external-audio",
      name: "外部对白.wav",
      subfolder: "director",
      type: "input",
      kind: "audio",
    };
    const onUploadFiles = vi.fn().mockResolvedValue({
      assets: [video, image, uploadedAudio],
      failures: [{ file_name: "损坏.txt", message: "不支持的素材格式" }],
      authority_stale: false,
    });
    render(<Harness initial={state} onUploadFiles={onUploadFiles} />);
    const inspector = screen.getByRole("region", { name: "当前片段编辑器" });
    const files = [
      new File(["v"], "源.mp4", { type: "video/mp4" }),
      new File(["i"], "角色.png", { type: "image/png" }),
      new File(["a"], "对白.wav", { type: "audio/wav" }),
      new File(["x"], "损坏.txt", { type: "text/plain" }),
    ];
    const dataTransfer = {
      types: ["Files"],
      getData: vi.fn(() => ""),
      setData: vi.fn(),
      dropEffect: "none",
      effectAllowed: "copy",
      files,
    };
    fireEvent.drop(inspector, { dataTransfer });

    await waitFor(() => expect(onUploadFiles).toHaveBeenCalledWith(files));
    await waitFor(() => expect(readState().assets).toEqual([video, image, uploadedAudio]));
    expect(readState().project.segments[0]).toMatchObject({
      title: "源视频",
      source_video: { id: video.id },
      duration_seconds: 30,
      source_start_seconds: 0,
      source_duration_seconds: 30,
      reference_images: [{ id: image.id, slot: 0 }],
      reference_audios: [{ id: uploadedAudio.id, slot: 0 }],
    });
    expect(screen.getByText(/1 项上传失败：损坏.txt/)).toBeInTheDocument();
  });

  it("站内视频拖入空源视频区会用素材全长重建当前时间线段", () => {
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
    }];
    state.assets = [video];
    render(<Harness initial={state} />);
    const sourceZone = screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" });
    const dataTransfer = {
      types: ["application/x-director-asset", "application/x-director-asset-video"],
      getData: vi.fn((type: string) => type === "application/x-director-asset" ? video.id : ""),
      setData: vi.fn(),
      dropEffect: "copy",
      effectAllowed: "copyMove",
      files: [],
    };

    fireEvent.drop(sourceZone, { dataTransfer });

    expect(readState().project.segments[0]).toMatchObject({
      title: "源视频",
      duration_seconds: 30,
      source_video: { id: video.id },
      source_start_seconds: 0,
      source_duration_seconds: 30,
    });
  });

  it("外部视频直接拖入空源视频区后也显示完整探测时长", async () => {
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
    }];
    const onUploadFiles = vi.fn().mockResolvedValue({
      assets: [video],
      failures: [],
      authority_stale: false,
    });
    render(<Harness initial={state} onUploadFiles={onUploadFiles} />);
    const file = new File(["video"], "源视频.mp4", { type: "video/mp4" });
    const dataTransfer = {
      types: ["Files"],
      getData: vi.fn(() => ""),
      setData: vi.fn(),
      dropEffect: "copy",
      effectAllowed: "copy",
      files: [file],
    };

    fireEvent.drop(screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" }), { dataTransfer });

    await waitFor(() => expect(onUploadFiles).toHaveBeenCalledWith([file]));
    await waitFor(() => expect(readState().project.segments[0]).toMatchObject({
      title: "源视频",
      duration_seconds: 30,
      source_video: { id: video.id },
      source_start_seconds: 0,
      source_duration_seconds: 30,
    }));
  });

  it("离线时仍拦截操作系统文件默认导航，但不开始上传", () => {
    const state = createTimelineEditorState();
    const onUploadFiles = vi.fn();
    const offline = { ...capabilities, connection: "offline" as const };
    render(<Harness initial={state} workspaceCapabilities={offline} onUploadFiles={onUploadFiles} />);
    const dataTransfer = {
      types: ["Files"],
      getData: vi.fn(() => ""),
      dropEffect: "copy",
      files: [new File(["image"], "offline.png", { type: "image/png" })],
    };

    const reference = screen.getByRole("region", { name: "首帧（可选）" });
    expect(fireEvent.dragOver(reference, { dataTransfer })).toBe(false);
    expect(dataTransfer.dropEffect).toBe("none");
    expect(fireEvent.drop(reference, { dataTransfer })).toBe(false);

    const inspector = screen.getByRole("region", { name: "当前片段编辑器" });
    dataTransfer.dropEffect = "copy";
    expect(fireEvent.dragOver(inspector, { dataTransfer })).toBe(false);
    expect(dataTransfer.dropEffect).toBe("none");
    expect(fireEvent.drop(inspector, { dataTransfer })).toBe(false);
    expect(onUploadFiles).not.toHaveBeenCalled();
  });

  it("音频与参考图策略只修改当前片段，输出规格不再占用时间线主界面", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    state.project.segments.push(createTimelineSegment("fl2va", 2));
    const supportedCapabilities: CapabilityReport = {
      ...capabilities,
      native_timeline: {
        supported: true,
        modes: ["fl2va", "ref2va"],
        continuity: true,
      },
    };
    render(<Harness initial={state} workspaceCapabilities={supportedCapabilities} />);

    const workspace = screen.getByRole("main", { name: "长视频时间线工作区" });
    expect(within(workspace).queryByRole("region", { name: "输出规格" })).not.toBeInTheDocument();
    expect(within(workspace).queryByLabelText("画幅")).not.toBeInTheDocument();
    expect(within(workspace).queryByLabelText("分辨率")).not.toBeInTheDocument();
    expect(within(workspace).queryByLabelText("导出方式")).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("音频策略"), "source");
    await user.selectOptions(screen.getByLabelText("参考图采样尺寸"), "max");

    expect(readState().project.segments[0]).toMatchObject({
      audio_mode: "source",
      ref_image_size: "max",
    });
    expect(readState().project.segments[1]).toMatchObject({
      audio_mode: "generate",
      ref_image_size: "match",
    });
  });

  it("时间线挂载不会改写旧项目的自定义画布", () => {
    const state = createTimelineEditorState();
    state.project.render = { width: 640, height: 640, fps: 24 };
    render(<Harness initial={state} />);

    expect(screen.getByLabelText("项目预览画布 640×640")).toBeInTheDocument();
    expect(readState().project.render).toEqual({ width: 640, height: 640, fps: 24 });
  });

  it("不支持连续性时仍可在片段编辑中清除迁移来的旧值", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const second = createTimelineSegment("fl2va", 2);
    second.continuity.enabled = true;
    state.project.segments.push(second);
    state.selected_segment_ids = [second.id];
    state.active_segment_id = second.id;
    state.selection_anchor_id = second.id;
    render(<Harness initial={state} />);

    const toggle = screen.getByRole("checkbox", { name: "启用当前片段连续性" });
    expect(toggle).toBeEnabled();
    expect(toggle).toBeChecked();
    expect(screen.getByRole("group", { name: "当前片段连续性" })).toHaveAccessibleDescription(
      /不支持这个片段的连续性/,
    );
    await user.click(toggle);
    expect(readState().project.segments[1].continuity.enabled).toBe(false);
  });

  it("每个片段独立保存连续性开关和接续帧数", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const second = createTimelineSegment("fl2va", 2);
    second.continuity = { enabled: true, overlap_frames: 39 };
    const third = createTimelineSegment("fl2va", 3);
    third.continuity = { enabled: false, overlap_frames: 5 };
    state.project.segments.push(second, third);
    state.selected_segment_ids = [second.id];
    state.active_segment_id = second.id;
    state.selection_anchor_id = second.id;
    const supportedCapabilities: CapabilityReport = {
      ...capabilities,
      native_timeline: { supported: true, modes: ["fl2va", "ref2va"], continuity: true },
    };
    render(<Harness initial={state} workspaceCapabilities={supportedCapabilities} />);

    expect(screen.getByRole("checkbox", { name: "启用当前片段连续性" })).toBeChecked();
    expect(screen.getByLabelText("当前片段接续尾帧数")).toHaveValue("39");
    await user.click(screen.getByRole("button", { name: /^聚焦并选择片段 3：/ }));
    expect(screen.getByRole("checkbox", { name: "启用当前片段连续性" })).not.toBeChecked();
    expect(screen.getByLabelText("当前片段接续尾帧数")).toHaveValue("5");
    await user.click(screen.getByRole("checkbox", { name: "启用当前片段连续性" }));
    await user.selectOptions(screen.getByLabelText("当前片段接续尾帧数"), "56");
    await user.click(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ }));
    expect(screen.getByRole("checkbox", { name: "启用当前片段连续性" })).toBeChecked();
    expect(screen.getByLabelText("当前片段接续尾帧数")).toHaveValue("39");
    expect(readState().project.segments.map((segment) => segment.continuity)).toEqual([
      { enabled: false, overlap_frames: 22 },
      { enabled: true, overlap_frames: 39 },
      { enabled: true, overlap_frames: 56 },
    ]);
  });

  it("FL2VA 开启连续性后引入首帧会自动取消并灰选连续性", () => {
    const state = createTimelineEditorState();
    const second = {
      ...createTimelineSegment("fl2va", 2),
      continuity: { enabled: true, overlap_frames: 39 as const },
    };
    state.project.segments.push(second);
    state.selected_segment_ids = [second.id];
    state.active_segment_id = second.id;
    state.selection_anchor_id = second.id;
    state.assets = [image];
    const supportedCapabilities: CapabilityReport = {
      ...capabilities,
      native_timeline: { supported: true, modes: ["fl2va", "ref2va"], continuity: true },
    };
    render(<Harness initial={state} workspaceCapabilities={supportedCapabilities} />);

    const toggle = screen.getByRole("checkbox", { name: "启用当前片段连续性" });
    expect(toggle).toBeChecked();
    expect(toggle).toBeEnabled();
    const dataTransfer = {
      types: ["application/x-director-asset", "application/x-director-asset-image"],
      getData: vi.fn((type: string) => type === "application/x-director-asset" ? image.id : ""),
      setData: vi.fn(),
      dropEffect: "none",
      effectAllowed: "copyMove",
      files: [],
    };
    const firstFrameZone = screen.getByRole("region", { name: "首帧（可选）" });
    fireEvent.dragOver(firstFrameZone, { dataTransfer });
    fireEvent.drop(firstFrameZone, { dataTransfer });

    expect(readState().project.segments[1]).toMatchObject({
      mode: "fl2va",
      first_image: { id: image.id },
      continuity: { enabled: false, overlap_frames: 39 },
    });
    expect(toggle).not.toBeChecked();
    expect(toggle).toBeDisabled();
    expect(screen.getByRole("group", { name: "当前片段连续性" })).toHaveAccessibleDescription(
      /显式首帧会形成画面锚点/,
    );
  });

  it("逐片段连续性解释裁剪语义，并在所选后段缺少直接前驱时显示可访问状态", () => {
    const state = createTimelineEditorState();
    state.project.segments[0].prompt = "第一段";
    const second = { ...createTimelineSegment("fl2va", 2), prompt: "第二段" };
    second.continuity = { enabled: true, overlap_frames: 22 };
    state.project.segments.push(second);
    state.selected_segment_ids = [second.id];
    state.active_segment_id = second.id;
    state.selection_anchor_id = second.id;
    const supportedCapabilities: CapabilityReport = {
      ...capabilities,
      native_timeline: {
        supported: true,
        modes: ["fl2va", "ref2va"],
        continuity: true,
      },
    };
    render(<Harness initial={state} workspaceCapabilities={supportedCapabilities} />);

    expect(screen.getByRole("group", { name: "当前片段连续性" })).toHaveAccessibleDescription(
      /最后 22 帧；导出时会裁掉引导帧/,
    );
    expect(screen.getByText("复用前驱成片 1 段")).toBeInTheDocument();
    const missing = screen.getByText("复用前驱成片");
    expect(missing).toHaveClass("is-history");
    expect(missing).toHaveAttribute("title", expect.stringContaining("复用直接前驱第 1 段"));
    const successorClip = screen.getByRole("button", { name: /^聚焦并选择片段 2：/ });
    expect(successorClip).toHaveAccessibleDescription(/复用直接前驱第 1 段/);

    fireEvent.click(screen.getByRole("button", { name: /^聚焦并选择片段 1：/ }), { ctrlKey: true });
    expect(screen.getByText("接续 1 段")).toBeInTheDocument();
    expect(screen.getByText("接续")).toHaveAttribute("title", expect.stringContaining("最后 22 帧"));
    expect(successorClip).toHaveAccessibleDescription(/最后 22 帧接续生成/);
  });

  it("未选择的边界只显示可接续，不冒充本次已激活接续", () => {
    const state = createTimelineEditorState();
    state.project.segments[0].prompt = "第一段";
    const second = { ...createTimelineSegment("fl2va", 2), prompt: "第二段" };
    second.continuity.enabled = true;
    state.project.segments.push(second);
    render(<Harness initial={state} workspaceCapabilities={{
      ...capabilities,
      native_timeline: { supported: true, modes: ["fl2va", "ref2va"], continuity: true },
    }} />);

    expect(screen.getByText("当前选择没有已启用的接续边界")).toBeInTheDocument();
    const eligible = screen.getByText("可接续");
    expect(eligible).toHaveClass("is-eligible");
    expect(eligible).not.toHaveClass("is-active");
    expect(screen.queryByText("接续")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ })).toHaveAccessibleDescription(
      /同时选择第 1 段与第 2 段后，可使用前段最后 22 帧接续生成/,
    );
  });

  it("段间接续直接标出尾帧不足与内部采样上限，不把参数问题显示成可运行", () => {
    const state = createTimelineEditorState();
    const first = {
      ...state.project.segments[0],
      prompt: "短前段",
      duration_seconds: 0.1,
    };
    const second = {
      ...createTimelineSegment("fl2va", 2),
      prompt: "长后段",
      duration_seconds: 498 / 24,
      continuity: { enabled: true, overlap_frames: 22 as const },
    };
    state.project.segments = [first, second];
    state.selected_segment_ids = [first.id, second.id];
    state.active_segment_id = second.id;
    state.selection_anchor_id = second.id;
    render(<Harness initial={state} workspaceCapabilities={{
      ...capabilities,
      native_timeline: { supported: true, modes: ["fl2va", "ref2va"], continuity: true },
    }} />);

    expect(screen.getByLabelText("当前片段接续尾帧数")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText("2 个接续参数问题")).toBeInTheDocument();
    expect(screen.getByText("前段过短")).toHaveAttribute("title", expect.stringContaining("只有 5 个可见帧"));
    expect(screen.queryByText("接续")).not.toBeInTheDocument();
  });

  it("时间线明确标出显式首帧断点，并允许其输出跨模型族接续", () => {
    const state = createTimelineEditorState();
    const first = { ...state.project.segments[0], prompt: "第一段" };
    const anchored = { ...createTimelineSegment("fl2va", 2), prompt: "锚点段", first_image: image, continuity: { enabled: true, overlap_frames: 22 as const } };
    const ref = { ...createTimelineSegment("ref2va", 3), prompt: "参考段", source_video: video, continuity: { enabled: true, overlap_frames: 22 as const } };
    state.project.segments = [first, anchored, ref];
    state.selected_segment_ids = [first.id, anchored.id, ref.id];
    state.active_segment_id = first.id;
    state.selection_anchor_id = first.id;
    render(<Harness initial={state} workspaceCapabilities={{
      ...capabilities,
      native_timeline: { supported: true, modes: ["fl2va", "ref2va"], continuity: true },
    }} />);

    expect(screen.getByText("首帧硬断点")).toHaveAttribute("title", expect.stringContaining("不读取前段尾帧"));
    expect(screen.queryByText("跨族硬断点")).not.toBeInTheDocument();
    expect(screen.getByText("接续")).toHaveAttribute("title", expect.stringContaining("最后 22 帧"));
  });

  it("视频落在 clip 左/右边缘只发出一个带稳定 anchorId 的原子插入动作", async () => {
    const state = createTimelineEditorState();
    state.assets = [video];
    const onDispatch = vi.fn();
    render(<LongFormTimelineWorkspace {...commonProps(state)} onDispatch={onDispatch} />);
    const clip = document.querySelector<HTMLElement>(".timeline-clip:not(.timeline-clip--add)");
    expect(clip).not.toBeNull();
    vi.spyOn(clip!, "getBoundingClientRect").mockReturnValue({
      x: 100, y: 0, left: 100, right: 300, top: 0, bottom: 100, width: 200, height: 100,
      toJSON: () => ({}),
    });
    const dataTransfer = {
      types: ["application/x-director-asset", "application/x-director-asset-video"],
      getData: vi.fn((type: string) => type === "application/x-director-asset" ? video.id : ""),
      setData: vi.fn(),
      dropEffect: "none",
      effectAllowed: "copyMove",
      files: [],
    };
    const dragOver = createEvent.dragOver(clip!, { dataTransfer });
    Object.defineProperty(dragOver, "clientX", { value: 110 });
    fireEvent(clip!, dragOver);
    expect(dataTransfer.dropEffect).toBe("copy");
    const dropClip = document.querySelector<HTMLElement>(".timeline-clip:not(.timeline-clip--add)");
    expect(dropClip).not.toBeNull();
    vi.spyOn(dropClip!, "getBoundingClientRect").mockReturnValue({
      x: 100, y: 0, left: 100, right: 300, top: 0, bottom: 100, width: 200, height: 100,
      toJSON: () => ({}),
    });
    const drop = createEvent.drop(dropClip!, { dataTransfer });
    Object.defineProperty(drop, "clientX", { value: 110 });
    fireEvent(dropClip!, drop);
    expect(dataTransfer.getData).toHaveBeenCalledWith("application/x-director-asset");
    expect(onDispatch).toHaveBeenCalledTimes(1);
    expect(onDispatch).toHaveBeenCalledWith({
      type: "segment/insert-video",
      asset: video,
      anchorId: state.project.segments[0].id,
      position: "before",
    });
  });

  it("视频落在轨道空白处时以当前检查器焦点作为插入锚点", () => {
    const state = createTimelineEditorState();
    const first = state.project.segments[0];
    const second = createTimelineSegment("fl2va", 2);
    const third = createTimelineSegment("fl2va", 3);
    state.project.segments = [first, second, third];
    state.selected_segment_ids = [first.id, second.id, third.id];
    state.active_segment_id = second.id;
    state.selection_anchor_id = second.id;
    state.assets = [video];
    const onDispatch = vi.fn();
    render(<LongFormTimelineWorkspace {...commonProps(state)} onDispatch={onDispatch} />);
    const track = document.querySelector<HTMLElement>(".director-timeline__track");
    expect(track).not.toBeNull();
    const dataTransfer = {
      types: ["application/x-director-asset", "application/x-director-asset-video"],
      getData: vi.fn((type: string) => type === "application/x-director-asset" ? video.id : ""),
      setData: vi.fn(),
      dropEffect: "copy",
      effectAllowed: "copyMove",
      files: [],
    };

    fireEvent.drop(track!, { dataTransfer });

    expect(onDispatch).toHaveBeenCalledWith({
      type: "segment/insert-video",
      asset: video,
      anchorId: second.id,
      position: "after",
    });
  });

  it("操作系统 Files 落在时间线片段时不会继承素材插片或重排语义", () => {
    const state = createTimelineEditorState();
    const onDispatch = vi.fn();
    render(<LongFormTimelineWorkspace {...commonProps(state)} onDispatch={onDispatch} />);
    const clip = document.querySelector<HTMLElement>(".timeline-clip:not(.timeline-clip--add)");
    expect(clip).not.toBeNull();
    const dataTransfer = {
      types: ["Files"],
      getData: vi.fn(() => ""),
      setData: vi.fn(),
      dropEffect: "copy",
      effectAllowed: "copy",
      files: [new File(["v"], "外部视频.mp4", { type: "video/mp4" })],
    };

    fireEvent.dragOver(clip!, { dataTransfer });
    fireEvent.drop(clip!, { dataTransfer });

    expect(dataTransfer.dropEffect).toBe("none");
    expect(onDispatch).not.toHaveBeenCalled();
  });

  it("中文标点后输入 # 会在触发位置列出当前片段素材、显示图片缩略图并插入稳定标签", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const libraryOnlyImage = {
      ...image,
      id: "library-only-image",
      name: "素材库角色.png",
    };
    const r2v = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: "",
      reference_images: [{ ...image, slot: 0 }],
    };
    state.project = { ...state.project, segments: [r2v] };
    // The introduced reference deliberately is not in the latest library
    // snapshot; the compatible library-only image must not leak into #.
    state.assets = [libraryOnlyImage];
    render(<Harness initial={state} />);

    const prompt = screen.getByLabelText("片段提示词");
    await user.type(prompt, "中文，#角");
    const picker = screen.getByRole("listbox", { name: "当前片段已引入的参考素材" });
    const option = within(picker).getByRole("option", { name: /角色参考\.png.*<Picture 1>/ });
    expect(option.querySelector("img")).toHaveAttribute("src", image.preview_url);
    expect(picker.parentElement).toBe(document.body);
    expect(picker.style.left).toMatch(/px$/);
    expect(picker.style.top).toMatch(/px$/);
    expect(screen.queryByText(libraryOnlyImage.name)).not.toBeInTheDocument();
    await user.keyboard("{Enter}");
    expect(prompt).toHaveValue("中文， <Picture 1>");
    const next = readState().project.segments[0];
    expect(next).toMatchObject({
      mode: "ref2va",
      reference_images: [expect.objectContaining({ id: image.id, slot: 0 })],
    });
    if (next.mode !== "ref2va") throw new Error("fixture changed mode");
    expect(next.reference_images).toHaveLength(1);
  });

  it("Ref2VA 已有源视频时 # 显示视频缩略图并插入已引入独立参考的 Video 2", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const libraryOnlyVideo = {
      ...replacementVideo,
      id: "library-only-video",
      name: "替换素材库机位.mp4",
    };
    const v2v = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: "",
      source_video: video,
      reference_videos: [{ ...replacementVideo, slot: 0 }],
    };
    state.project = { ...state.project, segments: [v2v] };
    state.assets = [video, libraryOnlyVideo];
    render(<Harness initial={state} />);

    const prompt = screen.getByLabelText("片段提示词");
    await user.type(prompt, "换机位 #替换");
    const picker = screen.getByRole("listbox", { name: "当前片段已引入的参考素材" });
    const option = within(picker).getByRole("option", { name: /替换机位\.mp4.*<Video 2>/ });
    expect(option.querySelector("video")).toHaveAttribute("src", replacementVideo.preview_url);
    expect(screen.queryByText(libraryOnlyVideo.name)).not.toBeInTheDocument();
    await user.keyboard("{Enter}");
    expect(prompt).toHaveValue("换机位 <Video 2>");
    const next = readState().project.segments[0];
    expect(next).toMatchObject({
      mode: "ref2va",
      source_video: expect.objectContaining({ id: video.id }),
      reference_videos: [expect.objectContaining({ id: replacementVideo.id, slot: 0 })],
    });
    if (next.mode !== "ref2va") throw new Error("fixture changed mode");
    expect(next.reference_videos).toHaveLength(1);
  });

  it("片段尚未引入参考素材时 # 不会推荐素材库内容", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const r2v = { ...createTimelineSegment("ref2va", 1), id: state.project.segments[0].id, prompt: "" };
    state.project = { ...state.project, segments: [r2v] };
    state.assets = [image, replacementVideo];
    render(<Harness initial={state} />);

    await user.type(screen.getByLabelText("片段提示词"), "#角");
    const picker = screen.getByRole("listbox", { name: "当前片段已引入的参考素材" });
    expect(picker).toHaveTextContent("请先在当前片段的参考素材区引入素材");
    expect(within(picker).queryByRole("option")).not.toBeInTheDocument();
    expect(readState().project.segments[0]).toMatchObject({
      mode: "ref2va",
      reference_images: [],
      reference_videos: [],
    });
  });

  it("@ 从 subject_definitions 识别大小写主体并插入定义中的原始标签", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const initialPrompt = [
      "subject_definitions:",
      "<Subject 1> is the woman in a red coat.",
      "<subject 2> is the blue rehearsal room.",
      "<SUBJECT 3> is a small dog.",
      "<SuBjEcT 4> is a dancer with a silver hat.",
      "summary:",
      "<Subject 99> appears only outside the definitions block.",
      "",
    ].join("\n");
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: initialPrompt,
    }];
    render(<Harness initial={state} />);

    const prompt = screen.getByLabelText("片段提示词");
    await user.type(prompt, "@");
    const picker = screen.getByRole("listbox", { name: "提示词中已定义的主体" });
    expect(within(picker).getByRole("option", { name: /<Subject 1>.*red coat/ })).toBeInTheDocument();
    expect(within(picker).getByRole("option", { name: /<subject 2>.*blue rehearsal room/ })).toBeInTheDocument();
    expect(within(picker).getByRole("option", { name: /<SUBJECT 3>.*small dog/ })).toBeInTheDocument();
    expect(within(picker).getByRole("option", { name: /<SuBjEcT 4>.*silver hat/ })).toBeInTheDocument();
    expect(within(picker).queryByRole("option", { name: /Subject 99/ })).not.toBeInTheDocument();

    await user.type(prompt, "blue");
    expect(within(picker).getAllByRole("option")).toHaveLength(1);
    await user.keyboard("{Enter}");
    expect(prompt).toHaveValue(`${initialPrompt}<subject 2>`);
  });

  it("# 弹窗打开期间参考集合变化会关闭旧选择范围", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const secondImage = {
      ...image,
      id: "second-introduced-image",
      name: "第二角色.png",
      slot: 1,
    };
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: "<Picture 1> ",
      reference_images: [{ ...image, slot: 0 }, secondImage],
    }];
    render(<Harness initial={state} />);

    const prompt = screen.getByLabelText("片段提示词");
    await user.type(prompt, "#第二");
    expect(screen.getByRole("listbox", { name: "当前片段已引入的参考素材" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: `从片段移除 ${image.name}` }));
    await waitFor(() => expect(screen.queryByRole("listbox", { name: "当前片段已引入的参考素材" })).not.toBeInTheDocument());
    expect(prompt).toHaveValue(" #第二");
    expect(readState().project.segments[0]).toMatchObject({
      mode: "ref2va",
      reference_images: [{ id: secondImage.id, slot: 0 }],
    });
  });

  it("V2V 仅对服务器确认有音轨的源视频开放 Audio 1 条件开关", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    state.project = {
      ...state.project,
      segments: [{
        ...createTimelineSegment("ref2va", 1),
        id: state.project.segments[0].id,
        prompt: "",
        source_video: video,
      }],
    };
    const view = render(<Harness initial={state} />);

    const toggle = screen.getByLabelText("参考源视频音轨");
    expect(toggle).toBeEnabled();
    await user.click(toggle);
    expect(readState().project.segments[0]).toMatchObject({
      mode: "ref2va",
      source_audio_as_reference: true,
    });
    const prompt = screen.getByLabelText("片段提示词");
    await user.type(prompt, "#");
    const picker = screen.getByRole("listbox", { name: "当前片段已引入的参考素材" });
    const options = within(picker).getAllByRole("option");
    expect(options).toHaveLength(2);
    expect(options[1].querySelector("img, video")).toBeNull();
    expect(prompt).toHaveAttribute("aria-activedescendant", options[0].id);
    await user.keyboard("{ArrowDown}");
    expect(options[1]).toHaveAttribute("aria-selected", "true");
    expect(prompt).toHaveAttribute("aria-activedescendant", options[1].id);
    await user.keyboard("{Enter}");
    expect(prompt).toHaveValue("<Audio 1>");

    await user.clear(prompt);
    await user.click(screen.getByRole("button", { name: "插入源音轨引用 Audio 1" }));
    expect(prompt).toHaveValue("<Audio 1> ");

    const silent = structuredClone(state);
    const silentSegment = silent.project.segments[0];
    if (silentSegment.mode !== "ref2va" || !silentSegment.source_video?.metadata)
      throw new Error("fixture must remain v2v");
    silentSegment.source_video.metadata.has_audio = false;
    view.unmount();
    render(<Harness initial={silent} />);
    expect(screen.getByLabelText("参考源视频音轨")).toBeDisabled();
    expect(screen.queryByText(/静音或历史素材缺少音轨信息/)).not.toBeInTheDocument();
  });

  it("切换片段后关闭上一片段遗留的 # 选择范围", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const first = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      reference_images: [{ ...image, slot: 0 }],
    };
    const second = createTimelineSegment("ref2va", 2);
    state.project = { ...state.project, segments: [first, second] };
    state.assets = [image];
    render(<Harness initial={state} />);

    await user.type(screen.getByLabelText("片段提示词"), "#角");
    expect(screen.getByRole("listbox", { name: "当前片段已引入的参考素材" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ }));
    await waitFor(() => expect(screen.queryByRole("listbox", { name: "当前片段已引入的参考素材" })).not.toBeInTheDocument());
    expect(screen.getByLabelText("片段提示词")).toHaveValue("");
  });

  it("片段提示词为空时点按钮填入 ref2va 六段式框架，非空时按钮禁用", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const segment = { ...createTimelineSegment("ref2va", 1), id: state.project.segments[0].id };
    state.project = { ...state.project, segments: [segment] };
    render(<Harness initial={state} />);

    const button = screen.getByRole("button", { name: "填入框架" });
    expect(button).toBeEnabled();
    await user.click(button);
    expect(screen.getByLabelText("片段提示词")).toHaveValue(EMPTY_SIX_SECTION_PROMPT);
    expect(button).toBeDisabled();
  });

  it("fl2va 片段按当前锚点填入三段式框架并带首帧对齐指令", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const segment = {
      ...createTimelineSegment("fl2va", 1),
      id: state.project.segments[0].id,
      first_image: image,
    };
    state.project = { ...state.project, segments: [segment] };
    render(<Harness initial={state} />);

    await user.click(screen.getByRole("button", { name: "填入框架" }));
    const value = (screen.getByLabelText("片段提示词") as HTMLTextAreaElement).value;
    expect(value).toContain("fully referenced");
    expect(value).toContain("<Picture 1>");
    expect(value).toContain("integrated_multimodal_description:");
    expect(value).not.toContain("subject_definitions:");
  });

  it("应用当前配置到后续要求确认，并保留目标 ID/名称/启用状态", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: "运镜",
      source_video: video,
      reference_images: [{ ...image, slot: 0 }],
    };
    const target = { ...createTimelineSegment("fl2va", 2), title: "目标保留", enabled: false };
    state.project = { ...state.project, segments: [source, target] };
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<Harness initial={state} />);
    await user.click(screen.getByRole("button", { name: "应用到后续" }));
    const copied = readState().project.segments[1];
    expect(copied).toMatchObject({ id: target.id, title: "目标保留", enabled: false, mode: "ref2va", prompt: "运镜" });
    expect(copied).not.toHaveProperty("first_image");
  });

  it("统一检查器调用智能分镜，按当前源区间连续拆分", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      duration_seconds: 10,
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 10,
    };
    state.project = { ...state.project, segments: [source] };
    vi.spyOn(directorApi, "detectRV2VShots").mockResolvedValue({
      cut_frames: [0, 5 * 24, 8 * 24, 30 * 24],
      shot_count: 3,
      warnings: [],
    });
    render(<Harness initial={state} />);
    await user.click(screen.getByRole("button", { name: "智能分割" }));
    await waitFor(() => expect(readState().project.segments).toHaveLength(3));
    expect(directorApi.detectRV2VShots).toHaveBeenCalledWith({
      asset_id: video.id,
      frame_rate: 24,
      sensitivity: "medium",
      min_shot_frames: 12,
    });
    expect(readState().project.segments).toEqual([
      expect.objectContaining({ source_start_seconds: 2, source_duration_seconds: 3 }),
      expect.objectContaining({ source_start_seconds: 5, source_duration_seconds: 3 }),
      expect.objectContaining({ source_start_seconds: 8, source_duration_seconds: 4 }),
    ]);
  });

  it("智能分镜请求未完成时卸载 Inspector，迟到结果不再修改时间线", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      duration_seconds: 10,
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 10,
    };
    const second = createTimelineSegment("fl2va", 2);
    state.project = { ...state.project, segments: [source, second] };
    let resolveDetection!: (value: { cut_frames: number[]; shot_count: number; warnings: string[] }) => void;
    vi.spyOn(directorApi, "detectRV2VShots").mockImplementation(() => new Promise((resolve) => { resolveDetection = resolve; }));
    render(<Harness initial={state} />);
    await user.click(screen.getByRole("button", { name: "智能分割" }));
    fireEvent.click(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ }));
    expect(screen.getByRole("button", { name: "智能分割" })).toBeDisabled();
    resolveDetection({ cut_frames: [5 * 24], shot_count: 2, warnings: [] });
    await Promise.resolve();
    expect(readState().project.segments).toHaveLength(2);
    expect(readState().project.segments[0]).toMatchObject({ id: source.id, source_start_seconds: 2, source_duration_seconds: 10 });
  });

  it("生成候选在暂停时按时间线片段局部时间 seek，实时预览带 updated_at 防缓存", async () => {
    const state = createTimelineEditorState();
    state.project.segments[0].prompt = "测试";
    state.playhead_seconds = 2.5;
    const segmentId = state.project.segments[0].id;
    const task: GenerationTask = {
      id: "running-live-preview",
      mode: "timeline",
      status: "running",
      progress: 0.5,
      stage: "sampling 12/25",
      prompt_id: null,
      error: null,
      preview_url: null,
      outputs: [],
      created_at: "2026-08-12T00:00:00Z",
      updated_at: "2026-08-12T00:01:02Z",
      started_at: "2026-08-12T00:00:01Z",
      completed_at: null,
      children: [],
      segment_results: [],
      live_preview_url: "/api/jobs/running-live-preview/live-preview",
    };
    render(<LongFormTimelineWorkspace
      {...commonProps(state)}
      activeTask={task}
      segmentCandidates={{
        [segmentId]: {
          job_id: "candidate-job",
          job_updated_at: "2026-08-12T00:02:00Z",
          result: {
            segment_id: segmentId,
            child_id: "candidate-child",
            output_url: "/api/jobs/candidate-job/segment-output?segment_id=current",
            output_file: "output/director/current.mp4",
            current_snapshot: true,
          },
        },
      }}
      onDispatch={() => undefined}
    />);
    const candidate = screen.getByLabelText(`片段 ${segmentId} 的最新生成候选`) as HTMLVideoElement;
    await waitFor(() => expect(candidate.currentTime).toBeCloseTo(2.5));
    expect(candidate.playbackRate).toBe(1);
    expect(candidate).not.toHaveAttribute("controls");
    expect(candidate.parentElement).toBe(screen.getByLabelText("项目预览画布 864×480"));
    expect(screen.getByLabelText("项目预览画布 864×480")).toHaveStyle({ aspectRatio: "864 / 480" });
    const liveToggle = screen.getByRole("button", { name: "实时执行" });
    expect(liveToggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(liveToggle);
    expect(liveToggle).toHaveAttribute("aria-expanded", "true");
    expect(liveToggle).toHaveTextContent("实时执行");
    expect(liveToggle).not.toHaveTextContent(/展开|收起|显示|隐藏/);
    expect(screen.getByLabelText("实时预览画布 864×480")).toHaveStyle({ aspectRatio: "864 / 480" });
    expect(screen.getByAltText("任务 running- 最新采样预览帧")).toHaveAttribute(
      "src",
      expect.stringContaining("?v=2026-08-12T00%3A01%3A02Z"),
    );
  });

  it("同一候选对象刷新不会把正在播放的视频误报为载入中，缓冲恢复后清除提示", () => {
    const state = createTimelineEditorState();
    const segmentId = state.project.segments[0].id;
    const candidate = {
      job_id: "stable-candidate-job",
      job_updated_at: "2026-08-12T00:02:00Z",
      result: {
        segment_id: segmentId,
        child_id: "stable-candidate-child",
        output_url: "/api/jobs/stable-candidate-job/segment-output",
        output_file: "output/director/stable.mp4",
        current_snapshot: false,
      },
    };
    const view = render(<LongFormTimelineWorkspace
      {...commonProps(state)}
      segmentCandidates={{ [segmentId]: candidate }}
      onDispatch={() => undefined}
    />);
    const videoElement = screen.getByLabelText(`片段 ${segmentId} 的最新生成候选`);
    const canvas = screen.getByLabelText("项目预览画布 864×480");

    fireEvent.loadedMetadata(videoElement);
    expect(screen.queryByText("载入预览…")).not.toBeInTheDocument();
    expect(canvas).toHaveAttribute("aria-busy", "false");

    view.rerender(<LongFormTimelineWorkspace
      {...commonProps(state)}
      segmentCandidates={{ [segmentId]: structuredClone(candidate) }}
      onDispatch={() => undefined}
    />);
    expect(screen.queryByText("载入预览…")).not.toBeInTheDocument();
    expect(canvas).toHaveAttribute("aria-busy", "false");

    fireEvent.waiting(videoElement);
    expect(screen.getByText("载入预览…")).toBeInTheDocument();
    expect(canvas).toHaveAttribute("aria-busy", "true");
    fireEvent.playing(videoElement);
    expect(screen.queryByText("载入预览…")).not.toBeInTheDocument();
    expect(canvas).toHaveAttribute("aria-busy", "false");
  });

  it("原视频对比同时同步生成候选与 Ref2VA 裁剪源，并只播放一条音轨", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: "对比镜头",
      source_video: video,
      source_start_seconds: 6,
      source_duration_seconds: 8,
      duration_seconds: 4,
    };
    state.project = { ...state.project, segments: [source] };
    state.playhead_seconds = 2;
    render(<LongFormTimelineWorkspace
      {...commonProps(state)}
      segmentCandidates={{
        [source.id]: {
          job_id: "compare-candidate-job",
          job_updated_at: "2026-08-12T00:02:00Z",
          result: {
            segment_id: source.id,
            child_id: "compare-candidate-child",
            output_url: "/api/jobs/compare-candidate-job/segment-output",
            output_file: "output/director/compare.mp4",
            current_snapshot: true,
          },
        },
      }}
      onDispatch={() => undefined}
    />);

    const toggle = screen.getByRole("button", { name: "原视频对比" });
    expect(toggle).toBeEnabled();
    expect(toggle).toHaveAttribute("aria-pressed", "false");
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("group", { name: "生成视频与原视频对比" })).toBeInTheDocument();

    const candidate = screen.getByLabelText(`片段 ${source.id} 的最新生成候选`) as HTMLVideoElement;
    const original = screen.getByLabelText(`原视频 ${video.name}`) as HTMLVideoElement;
    fireEvent.loadedMetadata(candidate);
    fireEvent.loadedMetadata(original);
    expect(candidate.currentTime).toBeCloseTo(2);
    expect(candidate.playbackRate).toBe(1);
    expect(original.currentTime).toBeCloseTo(sourcePreviewTime(source, 2, 24));
    expect(original.playbackRate).toBeCloseTo(
      source.source_duration_seconds / alignedTimelineSegmentDuration(source, 24),
    );
    expect(original.muted).toBe(true);
    expect(candidate.parentElement).toBe(screen.getByLabelText("项目预览画布 864×480"));
    expect(original.parentElement).toBe(screen.getByLabelText("原视频对比画布 864×480"));
  });

  it("原视频对比循环回到开头后会重新播放候选与裁剪源", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: "循环对比镜头",
      source_video: video,
      source_start_seconds: 6,
      source_duration_seconds: 8,
      duration_seconds: 4,
    };
    state.project = { ...state.project, segments: [source] };
    state.playhead_seconds = 2;
    render(<Harness
      initial={state}
      segmentCandidates={{
        [source.id]: {
          job_id: "loop-candidate-job",
          job_updated_at: "2026-08-16T00:00:00Z",
          result: {
            segment_id: source.id,
            child_id: "loop-candidate-child",
            output_url: "/api/jobs/loop-candidate-job/segment-output",
            output_file: "output/director/loop.mp4",
            current_snapshot: true,
          },
        },
      }}
    />);

    await user.click(screen.getByRole("button", { name: "原视频对比" }));
    await user.click(screen.getByRole("checkbox", { name: "循环" }));
    const candidate = screen.getByLabelText(`片段 ${source.id} 的最新生成候选`) as HTMLVideoElement;
    const original = screen.getByLabelText(`原视频 ${video.name}`) as HTMLVideoElement;
    const candidatePlay = vi.fn().mockResolvedValue(undefined);
    const originalPlay = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(candidate, "play", { configurable: true, value: candidatePlay });
    Object.defineProperty(original, "play", { configurable: true, value: originalPlay });
    fireEvent.loadedMetadata(candidate);
    fireEvent.loadedMetadata(original);

    await user.click(screen.getByRole("button", { name: "播放" }));
    await waitFor(() => {
      expect(candidatePlay).toHaveBeenCalledTimes(1);
      expect(originalPlay).toHaveBeenCalledTimes(1);
    });

    candidate.currentTime = source.duration_seconds;
    original.currentTime = source.source_start_seconds + source.source_duration_seconds;
    fireEvent.ended(candidate);

    await waitFor(() => {
      expect(readState().playhead_seconds).toBe(0);
      expect(candidate.currentTime).toBeCloseTo(0);
      expect(original.currentTime).toBeCloseTo(sourcePreviewTime(source, 0, 24));
      expect(candidatePlay).toHaveBeenCalledTimes(2);
      expect(originalPlay).toHaveBeenCalledTimes(2);
    });
    expect(screen.getByRole("button", { name: "暂停" })).toBeEnabled();
  });

  it("混合时间线对比经过 FL2VA 时保持双栏占位，进入 Ref2VA 后恢复原视频", async () => {
    const user = userEvent.setup();
    let state = createTimelineEditorState();
    const first = { ...state.project.segments[0], prompt: "生成镜头" };
    const second = {
      ...createTimelineSegment("ref2va", 2),
      prompt: "源片镜头",
      source_video: video,
      source_start_seconds: 3,
      source_duration_seconds: 5,
      duration_seconds: 5,
    };
    state = {
      ...state,
      project: { ...state.project, segments: [first, second] },
      selected_segment_ids: [first.id, second.id],
      active_segment_id: first.id,
      selection_anchor_id: first.id,
    };
    render(<Harness initial={state} />);

    const toggle = screen.getByRole("button", { name: "原视频对比" });
    await user.click(toggle);
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    const originalPlaceholder = screen.getByText("FL2VA 片段没有原视频").closest(".monitor__empty");
    const generatedPlaceholder = screen.getByText("当前片段尚无生成候选").closest(".monitor__empty");
    expect(originalPlaceholder).toHaveClass("monitor__empty--canvas");
    expect(generatedPlaceholder).toHaveClass("monitor__empty--canvas");
    expect(originalPlaceholder?.parentElement).toHaveClass("monitor__canvas");
    expect(generatedPlaceholder?.parentElement).toHaveClass("monitor__canvas");

    fireEvent.change(screen.getByLabelText("主预览播放头"), {
      target: { value: String(alignedTimelineSegmentDuration(first, 24)) },
    });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByText("FL2VA 片段没有原视频")).not.toBeInTheDocument();
    expect(screen.getByLabelText(`原视频 ${video.name}`)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "原视频" })).toBeInTheDocument();
  });

  it("主预览提供逐帧、首尾、播放头、音量和循环控制，并与时间线帧率同步", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    state.playhead_seconds = 1;
    render(<Harness initial={state} />);

    const total = Number((screen.getByLabelText("主预览播放头") as HTMLInputElement).max);
    expect(screen.getByLabelText("当前播放时间")).toHaveTextContent("00:01.00");
    expect(document.querySelector(".monitor__timecode")).not.toBeInTheDocument();
    expect(screen.getByLabelText("总播放时间")).toBeInTheDocument();
    const frameInput = screen.getByLabelText("当前预览帧");
    expect(frameInput).toHaveValue(25);
    fireEvent.change(frameInput, { target: { value: "1" } });
    fireEvent.blur(frameInput);
    expect(readState().playhead_seconds).toBe(0);
    fireEvent.focus(frameInput);
    fireEvent.change(frameInput, { target: { value: "25" } });
    fireEvent.blur(frameInput);
    expect(readState().playhead_seconds).toBeCloseTo(1);
    expect(screen.getByRole("button", { name: "上一帧" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "下一帧" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "跳到开头" }).querySelector(".transport-boundary-icon")).not.toBeNull();
    expect(screen.getByRole("button", { name: "跳到结尾" }).querySelector(".transport-boundary-icon")).not.toBeNull();
    expect(screen.getByRole("button", { name: "跳到开头" })).not.toHaveTextContent("|◀");
    const compare = screen.getByRole("button", { name: "原视频对比" });
    expect(compare).toBeDisabled();
    expect(compare).toHaveAttribute("title", "当前启用时间线没有可对比的 Ref2VA 源视频");

    await user.click(screen.getByRole("button", { name: "下一帧" }));
    expect(readState().playhead_seconds).toBeCloseTo(1 + 1 / 24);
    await user.click(screen.getByRole("button", { name: "上一帧" }));
    expect(readState().playhead_seconds).toBeCloseTo(1);

    fireEvent.change(screen.getByLabelText("主预览播放头"), { target: { value: "2" } });
    expect(readState().playhead_seconds).toBeCloseTo(2);
    expect(screen.getByLabelText("时间线播放头")).toHaveAttribute("data-seconds", "2");

    await user.click(screen.getByRole("button", { name: "跳到结尾" }));
    expect(readState().playhead_seconds).toBeCloseTo(total);
    await user.click(screen.getByRole("button", { name: "跳到开头" }));
    expect(readState().playhead_seconds).toBe(0);

    await user.click(screen.getByRole("button", { name: "静音" }));
    expect(screen.getByLabelText("预览音量")).toHaveValue("0");
    await user.click(screen.getByRole("button", { name: "取消静音" }));
    expect(screen.getByLabelText("预览音量")).toHaveValue("0.8");
    await user.click(screen.getByRole("checkbox", { name: "循环" }));
    expect(screen.getByRole("checkbox", { name: "循环" })).toBeChecked();
  });

  it("刷新后恢复实时执行、播放控制、对比、缩放和智能分割工具偏好", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 8,
      duration_seconds: 4,
    }];

    const firstView = render(<Harness initial={state} />);
    await user.click(screen.getByRole("button", { name: "实时执行" }));
    await user.click(screen.getByRole("checkbox", { name: "循环" }));
    await user.click(screen.getByRole("button", { name: "原视频对比" }));
    fireEvent.change(screen.getByLabelText("预览音量"), { target: { value: "0.35" } });
    fireEvent.change(screen.getByLabelText("时间线缩放比例"), { target: { value: "84" } });
    const evenSplitInput = screen.getByLabelText("均分片段数量");
    fireEvent.change(evenSplitInput, { target: { value: "3" } });
    fireEvent.blur(evenSplitInput);
    await user.selectOptions(screen.getByLabelText("智能分割灵敏度"), "high");
    const minimumFramesInput = screen.getByLabelText("智能分割最短镜头帧数");
    await user.clear(minimumFramesInput);
    expect((minimumFramesInput as HTMLInputElement).value).toBe("");
    await user.type(minimumFramesInput, "1");
    expect(minimumFramesInput).toHaveValue(1);
    await user.type(minimumFramesInput, "2");
    expect(minimumFramesInput).toHaveValue(12);
    await user.tab();

    await waitFor(() => expect(loadTimelineWorkspacePreferences()).toMatchObject({
      showLiveMonitor: true,
      volume: 0.35,
      loop: true,
      compareOriginal: true,
      timelineZoom: 84,
      evenSplitPieces: 3,
      detectionSensitivity: "high",
      minimumShotFrames: 12,
    }));

    firstView.unmount();
    render(<Harness initial={structuredClone(state)} />);
    expect(screen.getByRole("button", { name: "实时执行" })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("checkbox", { name: "循环" })).toBeChecked();
    expect(screen.getByRole("button", { name: "原视频对比" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText("预览音量")).toHaveValue("0.35");
    expect(screen.getByLabelText("时间线缩放比例")).toHaveValue("84");
    expect(screen.getByLabelText("均分片段数量")).toHaveValue(3);
    expect(screen.getByLabelText("智能分割灵敏度")).toHaveValue("high");
    expect(screen.getByLabelText("智能分割最短镜头帧数")).toHaveValue(12);
  });

  it("源视频在 metadata、seek 和 ended 生命周期中保持裁剪映射与片段边界", async () => {
    const state = createTimelineEditorState();
    const source = {
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      source_video: video,
      source_start_seconds: 6,
      source_duration_seconds: 8,
      duration_seconds: 4,
    };
    const next = createTimelineSegment("fl2va", 2);
    state.project = { ...state.project, segments: [source, next] };
    state.playhead_seconds = 2;
    render(<Harness initial={state} />);

    const sourcePreview = screen.getByLabelText(`源视频 ${video.name}`) as HTMLVideoElement;
    Object.defineProperty(sourcePreview, "duration", { configurable: true, value: 30 });
    fireEvent.loadedMetadata(sourcePreview);
    const outputDuration = alignedTimelineSegmentDuration(source, 24);
    expect(sourcePreview.currentTime).toBeCloseTo(sourcePreviewTime(source, 2, 24));
    expect(sourcePreview.playbackRate).toBeCloseTo(8 / outputDuration);

    fireEvent.seeking(sourcePreview);
    expect(screen.getByText("定位画面…")).toBeInTheDocument();
    fireEvent.seeked(sourcePreview);
    expect(screen.queryByText("定位画面…")).not.toBeInTheDocument();

    fireEvent.ended(sourcePreview);
    expect(readState().playhead_seconds).toBeCloseTo(outputDuration);
  });

  it("非采样节点执行时显示完整生成阶段，不暗示应当存在预览帧", () => {
    const state = createTimelineEditorState();
    const task: GenerationTask = {
      id: "raylight-loading",
      mode: "timeline",
      status: "running",
      progress: 0.1,
      stage: "片段 1/1 · 加载 RayLight 生成模型",
      prompt_id: null,
      error: null,
      preview_url: null,
      outputs: [],
      created_at: "2026-08-13T00:00:00Z",
      updated_at: "2026-08-13T00:00:01Z",
      started_at: "2026-08-13T00:00:01Z",
      completed_at: null,
      children: [],
      segment_results: [],
      live_preview_url: null,
    };

    render(<LongFormTimelineWorkspace
      {...commonProps(state)}
      activeTask={task}
      onDispatch={() => undefined}
    />);

    fireEvent.click(screen.getByRole("button", { name: "实时执行" }));
    expect(screen.getByText("实时执行进度")).toBeInTheDocument();
    expect(screen.getByText(/加载 RayLight 生成模型 · 当前阶段没有预览帧/)).toBeInTheDocument();
  });

  it("片段编辑控制带按视觉顺序组织，并始终呈现项目入点", () => {
    render(<Harness initial={createTimelineEditorState()} />);

    const inspector = screen.getByRole("region", { name: "当前片段编辑器" });
    const controlbar = inspector.querySelector(".segment-inspector__controlbar");
    expect(controlbar).not.toBeNull();
    expect(Array.from(controlbar?.children ?? []).map((element) => element.className)).toEqual([
      "segment-inspector__title",
      "field field--inline segment-inspector__name",
      "segment-inspector__divider",
      "segment-inspector__continuity",
      "field field--inline segment-inspector__mode",
      "field field--inline segment-inspector__duration",
      "field field--inline segment-inspector__audio-mode",
      "field field--inline segment-inspector__ref-image-size",
      "segment-inspector__head-actions",
      "segment-inspector__divider",
      "segment-inspector__timing",
    ]);
    expect(controlbar).toHaveTextContent("片段编辑");
    expect(controlbar).toContainElement(screen.getByRole("button", { name: "应用到后续" }));
    expect(inspector).not.toHaveTextContent("选择模型族并绑定素材；实际生成配方由当前素材自动确定");
    expect(inspector).not.toHaveTextContent("当前配方");
    expect(controlbar).toContainElement(screen.getByText("片段名称"));
    expect(controlbar).toContainElement(screen.getByText("生成模式"));
    expect(controlbar).toContainElement(screen.getByText("生成时长（秒）"));
    expect(controlbar).toContainElement(screen.getByLabelText("音频策略"));
    expect(controlbar).toContainElement(screen.getByLabelText("参考图采样尺寸"));
    expect(controlbar?.querySelectorAll(".field--inline")).toHaveLength(5);
    const timing = controlbar?.querySelector(".segment-inspector__timing");
    expect(timing).toHaveTextContent("项目入点");
    expect(timing).toHaveTextContent(/请求 5\.00s → 实际 .* · \d+f/);
  });

  it("有源 Ref2VA 灰显生成时长，并由源裁剪按既有变速比例更新", () => {
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      prompt: "参考源视频",
      duration_seconds: 4,
      source_video: video,
      source_start_seconds: 2,
      source_duration_seconds: 8,
    }];
    render(<Harness initial={state} />);

    const duration = screen.getByLabelText("生成时长（秒）");
    expect(duration).toBeDisabled();
    expect(duration).toHaveValue(4);
    expect(duration).toHaveAccessibleDescription(/通过下方源视频裁剪范围调整片段时长/);
    const sourceRegion = screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" });
    const sourceLayout = sourceRegion.querySelector(".segment-reference-grid__source-layout");
    const sourceControls = sourceRegion.querySelector(".segment-source-range");
    expect(Array.from(sourceLayout?.children ?? []).map((element) => element.className)).toEqual([
      "segment-reference-grid__assets",
      "segment-reference-grid__settings",
    ]);
    expect(sourceControls).toContainElement(screen.getByLabelText("源视频入点（秒）"));
    expect(sourceControls).toContainElement(screen.getByLabelText("源截取时长（秒）"));
    expect(sourceControls).toContainElement(screen.getByLabelText("参考源视频音轨"));
    expect(sourceRegion).not.toHaveTextContent("主视频固定引用为");
    expect(sourceRegion).not.toHaveTextContent("将裁剪后的主视频音轨送入");

    const sourceDuration = screen.getByLabelText("源截取时长（秒）");
    fireEvent.change(sourceDuration, { target: { value: "4" } });
    expect(readState().project.segments[0]).toMatchObject({
      duration_seconds: 4,
      source_duration_seconds: 8,
    });
    fireEvent.blur(sourceDuration);
    expect(readState().project.segments[0]).toMatchObject({
      duration_seconds: 2,
      source_duration_seconds: 4,
    });
  });

  it("保留源音频时反复回车提交 1 秒都稳定计算为 39 帧", async () => {
    const user = userEvent.setup();
    const state = createTimelineEditorState();
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      audio_mode: "source",
      duration_seconds: 1,
      source_video: video,
      source_start_seconds: 0,
      source_duration_seconds: 39 / 24,
    }];
    render(<Harness initial={state} fitSourceAudio />);

    const sourceDuration = screen.getByLabelText("源截取时长（秒）");
    for (let attempt = 0; attempt < 2; attempt += 1) {
      await user.click(sourceDuration);
      await user.clear(sourceDuration);
      await user.type(sourceDuration, "1{Enter}");
      await waitFor(() => {
        expect(readState().project.segments[0]).toMatchObject({
          duration_seconds: 1,
          source_duration_seconds: 39 / 24,
        });
      });
      expect(sourceDuration).toHaveValue(39 / 24);
      expect(screen.getByRole("region", { name: "当前片段编辑器" })).toHaveTextContent(
        "请求 1.00s → 实际 1.6250s · 39f",
      );
    }
  });

  it("保留源音频时显示素材总量与 H3 自动裁剪结果", () => {
    const state = createTimelineEditorState();
    const fittedFrames = 260;
    state.project.segments = [{
      ...createTimelineSegment("ref2va", 1),
      id: state.project.segments[0].id,
      duration_seconds: fittedFrames / 24,
      audio_mode: "source",
      source_video: {
        ...video,
        metadata: { ...video.metadata!, duration: 11.21, frame_count: 269 },
      },
      source_start_seconds: 0,
      source_duration_seconds: fittedFrames / 24,
    }];
    render(<Harness initial={state} />);

    expect(screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" })).toHaveTextContent(
      "素材总长11.21秒，共269帧，为满足H3约束，Director自动裁剪到10.8333秒，260帧",
    );
  });

  it("FL2VA 与无源 Ref2VA 仍可直接编辑生成时长", async () => {
    const user = userEvent.setup();
    render(<Harness initial={createTimelineEditorState()} />);
    expect(screen.getByLabelText("生成时长（秒）")).toBeEnabled();

    await user.selectOptions(screen.getByLabelText("片段生成模式"), "ref2va");
    expect(screen.getByLabelText("生成时长（秒）")).toBeEnabled();
  });

  it("把全选和禁用所选放在标题旁，并移除标题说明文字", () => {
    render(<Harness initial={createTimelineEditorState()} />);

    const title = screen.getByText("长视频编排");
    const titleRow = title.closest(".director-timeline__title");
    expect(titleRow).not.toBeNull();
    expect(titleRow).toContainElement(screen.getByRole("checkbox", { name: "全选" }));
    expect(titleRow).toContainElement(screen.getByRole("button", { name: /^禁用所选/ }));
    expect(titleRow).not.toHaveTextContent(/点击片段选择运行|Ctrl\/⌘ 多选/);
    const timelineHeader = title.closest(".director-timeline")?.querySelector(":scope > header");
    const summary = screen.getByLabelText("项目摘要");
    expect(timelineHeader).toContainElement(summary);
    expect(summary.querySelectorAll(".director-timeline__metric")).toHaveLength(3);
    expect(summary).toHaveTextContent("1段");
    expect(summary).toHaveTextContent("864×480");
    expect(timelineHeader).toContainElement(screen.getByRole("checkbox", { name: "全选" }));
    expect(timelineHeader).toContainElement(screen.getByLabelText("时间线缩放"));
    const actions = timelineHeader?.querySelector(".director-timeline__actions");
    expect(actions).not.toContainElement(screen.getByRole("checkbox", { name: "全选" }));
    expect(actions).not.toContainElement(screen.getByRole("button", { name: /^禁用所选/ }));
    expect(actions).toContainElement(screen.getByLabelText("时间线缩放"));
    expect(actions).not.toContainElement(summary);
    expect(document.querySelector(".timeline-commandbar .director-timeline__summary")).toBeNull();
  });

  it("全选覆盖停用片段，禁用所选后仍保持单一选择", async () => {
    const user = userEvent.setup();
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after" });
    const [first, second] = state.project.segments;
    state = timelineEditorReducer(state, {
      type: "segment/set-enabled",
      ids: [second.id],
      enabled: false,
    });
    render(<Harness initial={state} />);

    expect(screen.getByRole("checkbox", { name: "全选" })).toBeChecked();
    const firstClip = screen.getByRole("button", { name: /^聚焦并选择片段 1：/ });
    expect(firstClip).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("checkbox", { name: `多选片段 1：${first.title}` })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: `多选停用片段 2：${second.title}` })).toBeChecked();
    expect(document.querySelectorAll(".timeline-clip.is-selected")).toHaveLength(1);
    expect(screen.getByRole("button", { name: /^禁用所选/ })).toHaveAccessibleName("禁用所选，1 个已启用片段");

    await user.click(screen.getByRole("button", { name: /^禁用所选/ }));
    expect(readState().selected_segment_ids).toEqual([first.id, second.id]);
    expect(readState().project.segments.every((segment) => !segment.enabled)).toBe(true);
    expect(screen.getByRole("checkbox", { name: "全选" })).toBeChecked();
    expect(screen.getByRole("button", { name: /^禁用所选/ })).toBeDisabled();
    expect(document.querySelectorAll(".director-timeline__disabled article.is-selected")).toHaveLength(2);

    await user.click(screen.getByRole("checkbox", { name: "全选" }));
    expect(screen.getByRole("checkbox", { name: "全选" })).not.toBeChecked();
    expect(readState()).toMatchObject({
      selected_segment_ids: [],
      active_segment_id: null,
      selection_anchor_id: null,
    });
    expect(screen.getByRole("region", { name: "当前片段编辑器" })).toHaveTextContent("未选择片段");

    await user.click(screen.getByRole("checkbox", { name: "全选" }));
    expect(readState().selected_segment_ids).toEqual([first.id, second.id]);
    expect(screen.getByRole("checkbox", { name: "全选" })).toBeChecked();
  });

  it("右上角复选框累加多选时保留当前检查器焦点", () => {
    const state = createTimelineEditorState();
    const first = state.project.segments[0];
    const second = createTimelineSegment("fl2va", 2);
    const third = createTimelineSegment("ref2va", 3);
    state.project.segments = [first, second, third];
    render(<Harness initial={state} />);

    fireEvent.click(screen.getByRole("checkbox", { name: `多选片段 2：${second.title}` }));
    fireEvent.click(screen.getByRole("checkbox", { name: `多选片段 3：${third.title}` }));

    expect(readState()).toMatchObject({
      selected_segment_ids: [first.id, second.id, third.id],
      active_segment_id: first.id,
    });
    expect(within(screen.getByRole("region", { name: "当前片段编辑器" }))
      .getByLabelText("片段名称")).toHaveValue(first.title);
    const batchInspector = screen.getByRole("region", { name: "批量片段编辑" });
    expect(batchInspector).toHaveTextContent("已选择 3 个片段");
    expect(batchInspector).toHaveTextContent("专属素材字段会重新初始化");
  });

  it("片段整块复用普通、Ctrl 和 Shift 手势更新同一选择", () => {
    const state = createTimelineEditorState();
    const first = state.project.segments[0];
    const second = createTimelineSegment("fl2va", 2);
    const third = createTimelineSegment("ref2va", 3);
    state.project.segments = [first, second, third];
    render(<Harness initial={state} />);

    fireEvent.click(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ }));
    expect(readState().selected_segment_ids).toEqual([second.id]);
    expect(readState().active_segment_id).toBe(second.id);

    fireEvent.click(screen.getByRole("button", { name: /^聚焦并选择片段 1：/ }), { ctrlKey: true });
    expect(readState().selected_segment_ids).toEqual([second.id, first.id]);
    expect(readState().active_segment_id).toBe(first.id);

    fireEvent.click(screen.getByRole("button", { name: /^聚焦并选择片段 3：/ }), { shiftKey: true });
    expect(readState().selected_segment_ids).toEqual([first.id, second.id, third.id]);
    expect(readState().active_segment_id).toBe(third.id);
  });

  it("Shift 范围停留在目标可见轨，不会顺带选中另一轨片段", () => {
    const state = createTimelineEditorState();
    const first = state.project.segments[0];
    const disabledSecond = { ...createTimelineSegment("fl2va", 2), enabled: false };
    const third = createTimelineSegment("ref2va", 3);
    const disabledFourth = { ...createTimelineSegment("ref2va", 4), enabled: false };
    state.project.segments = [first, disabledSecond, third, disabledFourth];
    state.selected_segment_ids = [first.id];
    state.active_segment_id = first.id;
    state.selection_anchor_id = first.id;
    render(<Harness initial={state} />);

    fireEvent.click(screen.getByRole("button", { name: /^聚焦并选择片段 3：/ }), {
      shiftKey: true,
    });

    expect(readState()).toMatchObject({
      selected_segment_ids: [first.id, third.id],
      active_segment_id: third.id,
      selection_anchor_id: first.id,
    });
    expect(screen.getByRole("checkbox", {
      name: `多选停用片段 2：${disabledSecond.title}`,
    })).not.toBeChecked();

    fireEvent.click(screen.getByRole("button", { name: /^选择停用片段 2：/ }));
    fireEvent.click(screen.getByRole("button", { name: /^选择停用片段 4：/ }), {
      shiftKey: true,
    });
    expect(readState()).toMatchObject({
      selected_segment_ids: [disabledSecond.id, disabledFourth.id],
      active_segment_id: disabledFourth.id,
      selection_anchor_id: disabledSecond.id,
    });
    expect(screen.getByRole("button", { name: /^聚焦并选择片段 3：/ }))
      .toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: /^禁用所选/ })).toBeDisabled();
  });

  it("拖动片段结束后的浏览器 click 不会误改单一选择", () => {
    const state = createTimelineEditorState();
    const first = state.project.segments[0];
    const second = createTimelineSegment("fl2va", 2);
    state.project.segments = [first, second];
    state.selected_segment_ids = [first.id, second.id];
    render(<Harness initial={state} />);

    const firstClip = screen.getByRole("button", { name: /^聚焦并选择片段 1：/ });
    const dataTransfer = {
      effectAllowed: "none",
      dropEffect: "none",
      getData: vi.fn(() => ""),
      setData: vi.fn(),
      types: [] as string[],
    };
    fireEvent.dragStart(firstClip, { dataTransfer });
    fireEvent.dragEnd(firstClip, { dataTransfer });
    fireEvent.click(firstClip);

    expect(readState().selected_segment_ids).toEqual([first.id, second.id]);
  });

  it("停用轨可选择和多选，重新启用后仍保持选择", () => {
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after" });
    const second = state.project.segments[1];
    state = timelineEditorReducer(state, {
      type: "segment/set-enabled",
      ids: [second.id],
      enabled: false,
    });
    render(<Harness initial={state} />);

    const selectDisabled = screen.getByRole("button", {
      name: new RegExp(`^选择停用片段 2：${second.title}`),
    });
    const selectCheckbox = screen.getByRole("checkbox", { name: `多选停用片段 2：${second.title}` });
    const enable = screen.getByRole("button", { name: `启用片段 2：${second.title}` });
    expect(selectDisabled).not.toContainElement(enable);
    expect(selectDisabled.closest("article")).toContainElement(enable);
    expect(selectCheckbox).toBeChecked();

    fireEvent.click(selectDisabled);
    expect(readState().selected_segment_ids).toEqual([second.id]);
    expect(readState().active_segment_id).toBe(second.id);

    fireEvent.click(enable);
    expect(readState().project.segments.find((segment) => segment.id === second.id)?.enabled).toBe(true);
    expect(readState().selected_segment_ids).toEqual([second.id]);
  });

  it("按钮和键盘删除多个片段时共用带数量的确认", async () => {
    const user = userEvent.setup();
    let state = createTimelineEditorState();
    state = timelineEditorReducer(state, { type: "segment/insert", position: "after" });
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    render(<Harness initial={state} />);

    await user.click(screen.getByRole("button", { name: "删除所选" }));
    expect(confirm).toHaveBeenNthCalledWith(1, expect.stringContaining("2 个片段"));
    expect(readState().project.segments).toHaveLength(2);

    const timeline = screen.getByRole("region", { name: "主时间线" });
    timeline.focus();
    fireEvent.keyDown(timeline, { key: "Delete" });
    expect(confirm).toHaveBeenNthCalledWith(2, expect.stringContaining("2 个片段"));
    expect(readState().project.segments).toHaveLength(1);
  });
});
