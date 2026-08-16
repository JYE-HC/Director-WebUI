import {
  createInitialDrafts,
  type AssetReference,
  type SlottedAssetReference,
} from "../domain/modes";
import type { GenerationTask } from "../api/types";
import {
  createInitialDirectorState,
  directorReducer,
  loadDirectorState,
  saveDirectorState,
} from "../state/directorState";

const validImage: AssetReference = {
  id: "asset-image-1",
  name: "valid.png",
  subfolder: "director-web",
  type: "input",
  kind: "image",
};

const validVideo: AssetReference = {
  id: "asset-video-1",
  name: "valid.mp4",
  subfolder: "director-web",
  type: "input",
  kind: "video",
  metadata: {
    duration: 12,
    native_fps: 30,
    frame_count: 360,
    width: 1920,
    height: 1080,
    probe_method: "ffprobe_nb_frames",
    has_audio: true,
  },
};

beforeEach(() => localStorage.clear());

describe("六种模式草稿隔离", () => {
  it("为每个模式创建独立且正确的 discriminant", () => {
    const drafts = createInitialDrafts();
    expect(Object.keys(drafts)).toEqual(["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"]);
    for (const [mode, draft] of Object.entries(drafts)) expect(draft.mode).toBe(mode);
    expect(drafts.i2v.shots[0]).toHaveProperty("first_image");
    expect(drafts.fl2v.shots[0]).toHaveProperty("last_image");
    expect(drafts.r2v.shots[0]).toHaveProperty("reference_videos");
    expect(drafts.v2v.shots[0]).toHaveProperty("source_video");
    expect(drafts.rv2v.shots[0]).not.toHaveProperty("reference_videos");
    expect(drafts.t2v.shots[0]).not.toHaveProperty("source_video");
  });

  it("修改一个模式不会改变其余五个模式", () => {
    const initial = createInitialDirectorState();
    const untouched = {
      i2v: structuredClone(initial.drafts.i2v), fl2v: structuredClone(initial.drafts.fl2v),
      r2v: structuredClone(initial.drafts.r2v), v2v: structuredClone(initial.drafts.v2v), rv2v: structuredClone(initial.drafts.rv2v),
    };
    const next = directorReducer(initial, { type: "draft/replace", draft: { ...initial.drafts.t2v, prompt: "只属于 T2V" } });
    expect(next.drafts.t2v.prompt).toBe("只属于 T2V");
    expect(next.drafts.i2v).toEqual(untouched.i2v); expect(next.drafts.fl2v).toEqual(untouched.fl2v);
    expect(next.drafts.r2v).toEqual(untouched.r2v); expect(next.drafts.v2v).toEqual(untouched.v2v); expect(next.drafts.rv2v).toEqual(untouched.rv2v);
  });

  it("hydration 剔除所有缺少稳定 ID 的旧素材并保留有效素材", () => {
    const state = createInitialDirectorState();
    const legacyImage = {
      name: "legacy.png",
      subfolder: "director-web",
      type: "input",
      kind: "image",
    } as unknown as AssetReference;
    const legacyVideo = {
      name: "legacy.mp4",
      subfolder: "director-web",
      type: "input",
      kind: "video",
    } as unknown as AssetReference;

    state.drafts.i2v.shots[0].first_image = legacyImage;
    state.drafts.fl2v.shots[0].first_image = legacyImage;
    state.drafts.fl2v.shots[0].last_image = validImage;
    state.drafts.r2v.shots[0].reference_images = [
      legacyImage as unknown as SlottedAssetReference,
      { ...validImage, slot: 4 },
    ];
    state.drafts.v2v.shots[0].source_video = legacyVideo;
    state.drafts.rv2v.shots[0].source_video = legacyVideo;
    state.drafts.rv2v.shots[0].reference_images = [
      legacyImage as unknown as SlottedAssetReference,
    ];
    saveDirectorState(state);

    const hydrated = loadDirectorState();
    expect(hydrated.drafts.i2v.shots[0].first_image).toBeNull();
    expect(hydrated.drafts.fl2v.shots[0].first_image).toBeNull();
    expect(hydrated.drafts.fl2v.shots[0].last_image).toEqual(validImage);
    expect(hydrated.drafts.r2v.shots[0].reference_images).toEqual([{ ...validImage, slot: 4 }]);
    expect(hydrated.drafts.v2v.shots[0].source_video).toBeNull();
    expect(hydrated.drafts.rv2v.shots[0].source_video).toBeNull();
    expect(hydrated.drafts.rv2v.shots[0].reference_images).toEqual([]);
  });

  it("local hydration 保留视频 metadata、引用 slot，并剥离普通源视频的 slot", () => {
    const state = createInitialDirectorState();
    state.drafts.r2v.prompt = "参考生成";
    state.drafts.r2v.shots[0].reference_videos = [{ ...validVideo, slot: 2 }];
    state.drafts.v2v.prompt = "视频重绘";
    state.drafts.v2v.shots[0].source_video = {
      ...validVideo,
      slot: 1,
    } as AssetReference;
    saveDirectorState(state);

    const hydrated = loadDirectorState();
    expect(hydrated.drafts.r2v.shots[0].reference_videos).toEqual([
      { ...validVideo, slot: 2 },
    ]);
    expect(hydrated.drafts.v2v.shots[0].source_video).toEqual(validVideo);
    expect(hydrated.drafts.v2v.shots[0].source_video).not.toHaveProperty("slot");
  });

  it("浏览器不持久化或恢复运行设置，服务器 GET 是唯一权威", () => {
    const initial = createInitialDirectorState();
    const state = directorReducer(initial, {
      type: "settings/replace",
      settings: { ...initial.settings, comfy_url: "http://stale-comfy.test:8188" },
    });
    saveDirectorState(state);
    const key = localStorage.key(0)!;
    const stored = JSON.parse(localStorage.getItem(key)!) as Record<string, unknown>;
    expect(stored).not.toHaveProperty("settings");

    stored.settings = {
      ...state.settings,
      comfy_url: "http://legacy-local.test:8188",
      raylight: { gpu_ids: [0, 1] },
    };
    localStorage.setItem(key, JSON.stringify(stored));

    const hydrated = loadDirectorState();
    expect(hydrated.settings.comfy_url).toBe("");
    expect(hydrated.settings).not.toHaveProperty("raylight");
  });

  it("新布局默认收起任务栏，且不恢复旧版默认展开状态", () => {
    const initial = createInitialDirectorState();
    expect(initial.taskPanelOpen).toBe(false);
    saveDirectorState({ ...initial, taskPanelOpen: true });
    expect(loadDirectorState().taskPanelOpen).toBe(true);

    const key = localStorage.key(0)!;
    const legacy = JSON.parse(localStorage.getItem(key)!) as Record<string, unknown>;
    delete legacy.layoutVersion;
    legacy.taskPanelOpen = true;
    localStorage.setItem(key, JSON.stringify(legacy));
    expect(loadDirectorState().taskPanelOpen).toBe(false);
  });

  it("系统设置浮层不写入本地布局，也不从旧版 view 恢复", () => {
    const state = directorReducer(createInitialDirectorState(), { type: "navigate/settings" });
    saveDirectorState(state);
    const key = localStorage.key(0)!;
    const stored = JSON.parse(localStorage.getItem(key)!) as Record<string, unknown>;
    expect(stored).not.toHaveProperty("view");

    stored.view = "settings";
    localStorage.setItem(key, JSON.stringify(stored));
    expect(loadDirectorState().view).toBe("workspace");
  });

  it("按模式白名单重建本地草稿，单个损坏模式不会清空其他模式", () => {
    const state = createInitialDirectorState();
    state.drafts.t2v.prompt = "保留 T2V";
    state.drafts.i2v.prompt = "损坏前 I2V";
    saveDirectorState(state);

    const key = localStorage.key(0)!;
    const stored = JSON.parse(localStorage.getItem(key)!) as {
      drafts: Record<string, Record<string, unknown>>;
    };
    const t2v = stored.drafts.t2v;
    const shots = t2v.shots as Record<string, unknown>[];
    const sampling = t2v.sampling as Record<string, unknown>;
    const render = t2v.render as Record<string, unknown>;
    t2v.raylight = { gpu_ids: [0, 1] };
    t2v.foreign_mode_config = { model: "other-mode.safetensors" };
    shots[0].first_image = validImage;
    shots[0].source_video = validVideo;
    shots[0].reference_images = [{ ...validImage, slot: 0 }];
    sampling.sampler = "foreign_sampler";
    sampling.scheduler = "foreign_scheduler";
    sampling.private_parameter = 123;
    render.private_parameter = 456;
    stored.drafts.i2v = { mode: "t2v" };
    localStorage.setItem(key, JSON.stringify(stored));

    const hydrated = loadDirectorState();
    expect(hydrated.drafts.t2v.prompt).toBe("保留 T2V");
    expect(hydrated.drafts.t2v).not.toHaveProperty("raylight");
    expect(hydrated.drafts.t2v).not.toHaveProperty("foreign_mode_config");
    expect(hydrated.drafts.t2v.shots[0]).not.toHaveProperty("first_image");
    expect(hydrated.drafts.t2v.shots[0]).not.toHaveProperty("source_video");
    expect(hydrated.drafts.t2v.shots[0]).not.toHaveProperty("reference_images");
    expect(hydrated.drafts.t2v.sampling.sampler).toBe("res_multistep");
    expect(hydrated.drafts.t2v.sampling.scheduler).toBe("simple");
    expect(hydrated.drafts.t2v.sampling).not.toHaveProperty("private_parameter");
    expect(hydrated.drafts.t2v.render).not.toHaveProperty("private_parameter");
    expect(hydrated.drafts.i2v.prompt).toBe("");
    expect(hydrated.drafts.i2v.mode).toBe("i2v");
  });

  it("旧六模式本地草稿保留 beta 调度器", () => {
    const state = createInitialDirectorState();
    state.drafts.t2v.sampling.scheduler = "beta";
    saveDirectorState(state);

    expect(loadDirectorState().drafts.t2v.sampling.scheduler).toBe("beta");
  });

  it("素材上传 delta 合并进最新草稿，并忽略已删除镜头的迟到结果", () => {
    let state = createInitialDirectorState();
    const draft = structuredClone(state.drafts.fl2v);
    const removedShotId = draft.shots[0].id;
    draft.prompt = "上传前";
    draft.shots.push({
      ...structuredClone(draft.shots[0]),
      id: "fl2v-surviving-shot",
      title: "保留镜头",
    });
    state = directorReducer(state, { type: "draft/replace", draft });

    state = directorReducer(state, {
      type: "draft/replace",
      draft: { ...state.drafts.fl2v, prompt: "上传期间的新提示词" },
    });
    state = directorReducer(state, {
      type: "draft/assets",
      mode: "fl2v",
      shotId: removedShotId,
      field: "last_image",
      mutation: { type: "add", assets: [{ ...validImage, id: "last-image" }] },
    });
    state = directorReducer(state, {
      type: "draft/assets",
      mode: "fl2v",
      shotId: removedShotId,
      field: "first_image",
      mutation: { type: "add", assets: [validImage] },
    });

    expect(state.drafts.fl2v.prompt).toBe("上传期间的新提示词");
    expect(state.drafts.fl2v.shots[0].first_image).toEqual(validImage);
    expect(state.drafts.fl2v.shots[0].last_image?.id).toBe("last-image");

    state = directorReducer(state, {
      type: "draft/replace",
      draft: {
        ...state.drafts.fl2v,
        shots: state.drafts.fl2v.shots.filter((shot) => shot.id !== removedShotId),
      },
    });
    const afterDelete = state;
    state = directorReducer(state, {
      type: "draft/assets",
      mode: "fl2v",
      shotId: removedShotId,
      field: "first_image",
      mutation: { type: "add", assets: [{ ...validImage, id: "late-image" }] },
    });
    expect(state).toBe(afterDelete);
    expect(state.drafts.fl2v.shots.map((shot) => shot.id)).toEqual([
      "fl2v-surviving-shot",
    ]);
  });

  it("源视频上传在 reducer 最新状态中按探测时长收紧截取范围", () => {
    let state = createInitialDirectorState();
    const shotId = state.drafts.v2v.shots[0].id;
    const shortVideo = {
      ...validVideo,
      id: "short-video",
      metadata: { ...validVideo.metadata!, duration: 3, frame_count: 90 },
    };
    state = directorReducer(state, {
      type: "draft/assets",
      mode: "v2v",
      shotId,
      field: "source_video",
      mutation: { type: "add", assets: [shortVideo] },
    });
    expect(state.drafts.v2v.shots[0].source_video).toEqual(shortVideo);
    expect(state.drafts.v2v.shots[0].source_start_seconds).toBe(0);
    expect(state.drafts.v2v.shots[0].source_duration_seconds).toBe(3);
  });

  it("迟到的后端 hydration 不覆盖已开始编辑的本地草稿", () => {
    let state = createInitialDirectorState();
    const firstServerDraft = { ...state.drafts.t2v, prompt: "首次服务端草稿" };
    state = directorReducer(state, {
      type: "draft/hydrate",
      mode: "t2v",
      draft: firstServerDraft,
    });
    expect(state.drafts.t2v.prompt).toBe("首次服务端草稿");

    state = directorReducer(state, {
      type: "draft/replace",
      draft: { ...state.drafts.t2v, prompt: "用户正在编辑" },
    });
    state = directorReducer(state, {
      type: "draft/hydrate",
      mode: "t2v",
      draft: { ...firstServerDraft, prompt: "迟到的旧响应" },
    });
    expect(state.drafts.t2v.prompt).toBe("用户正在编辑");
    expect(state.draftSync.t2v.dirty).toBe(true);
  });

  it("按模式 revision 处理保存响应，不让旧响应覆盖保存期间的新编辑", () => {
    let state = createInitialDirectorState();
    state = directorReducer(state, {
      type: "draft/replace",
      draft: { ...state.drafts.t2v, prompt: "第一版" },
    });
    const firstRevision = state.draftSync.t2v.revision;
    const firstSnapshot = structuredClone(state.drafts.t2v);
    state = directorReducer(state, {
      type: "draft/save-start",
      mode: "t2v",
      revision: firstRevision,
    });
    state = directorReducer(state, {
      type: "draft/replace",
      draft: { ...state.drafts.t2v, prompt: "保存期间第二版" },
    });
    state = directorReducer(state, {
      type: "draft/save-success",
      mode: "t2v",
      revision: firstRevision,
      draft: firstSnapshot,
    });
    expect(state.drafts.t2v.prompt).toBe("保存期间第二版");
    expect(state.draftSync.t2v).toMatchObject({
      dirty: true,
      savingRevision: null,
      status: "idle",
    });
    expect(state.draftSync.i2v).toMatchObject({ revision: 0, dirty: false });

    const secondRevision = state.draftSync.t2v.revision;
    const secondSnapshot = structuredClone(state.drafts.t2v);
    state = directorReducer(state, {
      type: "draft/save-start",
      mode: "t2v",
      revision: secondRevision,
    });
    state = directorReducer(state, {
      type: "draft/save-success",
      mode: "t2v",
      revision: secondRevision,
      draft: secondSnapshot,
    });
    expect(state.draftSync.t2v).toMatchObject({
      dirty: false,
      savingRevision: null,
      status: "saved",
    });
  });

  it("后端返回错误模式草稿时不误报保存成功", () => {
    let state = createInitialDirectorState();
    state = directorReducer(state, {
      type: "draft/replace",
      draft: { ...state.drafts.t2v, prompt: "待保存" },
    });
    const revision = state.draftSync.t2v.revision;
    state = directorReducer(state, {
      type: "draft/save-start",
      mode: "t2v",
      revision,
    });
    state = directorReducer(state, {
      type: "draft/save-success",
      mode: "t2v",
      revision,
      draft: state.drafts.i2v,
    });
    expect(state.draftSync.t2v).toMatchObject({
      dirty: true,
      savingRevision: null,
      status: "error",
    });
    expect(state.drafts.t2v.prompt).toBe("待保存");
  });

  it("任务删除只移除目标，批清仅移除三个终态并保留活动任务", () => {
    const makeTask = (id: string, status: GenerationTask["status"]): GenerationTask => ({
      id,
      mode: "t2v",
      status,
      progress: status === "succeeded" ? 1 : 0,
      stage: null,
      prompt_id: null,
      outputs: [],
      error: null,
      preview_url: null,
      created_at: "2026-08-12T00:00:00Z",
      updated_at: "2026-08-12T00:00:00Z",
      started_at: null,
      completed_at: null,
      children: [],
      current_project: true,
      segment_results: [],
      live_preview_url: null,
    });
    let state = {
      ...createInitialDirectorState(),
      tasks: [
        makeTask("running", "running"),
        makeTask("succeeded", "succeeded"),
        makeTask("failed", "failed"),
        makeTask("cancelled", "cancelled"),
        makeTask("cancelling", "cancelling"),
      ],
    };

    state = directorReducer(state, { type: "tasks/remove", id: "failed" });
    expect(state.tasks.map((task) => task.id)).not.toContain("failed");
    state = directorReducer(state, { type: "tasks/clear-terminal" });
    expect(state.tasks.map((task) => task.id)).toEqual(["running", "cancelling"]);
  });

  it("服务器权威时间线或设置改变后保守失效已缓存的当前候选标记", () => {
    const task: GenerationTask = {
      id: "candidate",
      mode: "timeline",
      status: "succeeded",
      progress: 1,
      stage: "completed",
      prompt_id: null,
      outputs: [],
      output_files: [],
      error: null,
      preview_url: null,
      created_at: "2026-08-12T00:00:00Z",
      updated_at: "2026-08-12T00:00:00Z",
      started_at: null,
      completed_at: "2026-08-12T00:00:00Z",
      children: [],
      current_project: true,
      segment_results: [{
        segment_id: "segment-a",
        child_id: "child-a",
        output_url: "/api/jobs/candidate/segment-output?segment_id=segment-a",
        output_file: "output/segment-a.mp4",
        current_snapshot: true,
      }],
      live_preview_url: null,
    };

    const state = directorReducer(
      { ...createInitialDirectorState(), tasks: [task] },
      { type: "tasks/invalidate-current-snapshots" },
    );

    expect(state.tasks[0].segment_results[0].current_snapshot).toBe(false);
    expect(state.tasks[0].current_project).toBe(false);
    expect(state.tasks[0].segment_results[0].output_url).toBe(
      task.segment_results[0].output_url,
    );
  });
});
