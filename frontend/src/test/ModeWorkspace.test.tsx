import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { directorApi } from "../api/client";
import { EMPTY_CAPABILITIES, EMPTY_MODELS, DEFAULT_SETTINGS as UNCONFIGURED_SETTINGS, type ModelInventory } from "../api/types";
import { ModeWorkspace, validateModeDraft } from "../components/ModeWorkspace";
import { deleteRV2VShot, moveRV2VShot, splitRV2VShot, splitRV2VShotAtSourceCuts } from "../components/TimelineEditor";
import {
  createInitialDirectorState,
  directorReducer,
  type AssetMutation,
  type DraftAssetField,
} from "../state/directorState";
import {
  createInitialDrafts,
  type AssetReference,
  type ModeDraft,
} from "../domain/modes";

const capabilities = { ...EMPTY_CAPABILITIES, connection: "online" as const, supported_modes: ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"] as const };
const DEFAULT_SETTINGS = {
  ...UNCONFIGURED_SETTINGS,
  comfy_url: "http://comfy.test:8188",
};
const MODELS: ModelInventory = {
  ...EMPTY_MODELS,
  fl2va: [DEFAULT_SETTINGS.models.fl2va.filename, "fl2va-alt.safetensors"],
  ref2va: [DEFAULT_SETTINGS.models.ref2va.filename, "ref2va-alt.safetensors"],
  loras: ["minimax-h3-turbo.safetensors", "cinema-style.safetensors"],
};

const sourceVideo: AssetReference = {
  id: "source-video",
  name: "source.mp4",
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

const image = (id: string, slot: number) => ({
  id,
  name: `${id}.png`,
  subfolder: "director-web",
  type: "input" as const,
  kind: "image" as const,
  slot,
});

beforeEach(() => localStorage.removeItem("director-web:global-settings-open"));

describe("模式表单", () => {
  it("未保存 ComfyUI 地址时，即使能力数据过期为在线也禁用提交和素材上传", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const upload = vi.spyOn(directorApi, "uploadAsset");
    const draft = createInitialDrafts().i2v;
    draft.prompt = "镜头";
    render(<ModeWorkspace draft={draft} settings={UNCONFIGURED_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={onSubmit} />);

    expect(screen.getByText(/尚未配置 ComfyUI；请先到系统设置填写地址/)).toBeInTheDocument();
    fireEvent.submit(screen.getByRole("form", { name: "I2V 模式工作区" }));
    expect(screen.getByLabelText("起始帧")).toBeDisabled();
    expect(screen.getByText("配置 ComfyUI 地址并等待连接就绪后可上传素材")).toBeInTheDocument();
    await user.upload(
      screen.getByLabelText("起始帧"),
      new File(["image"], "first.png", { type: "image/png" }),
    );
    expect(upload).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("已配置但连接离线时仍禁用素材上传", () => {
    const draft = createInitialDrafts().i2v;
    render(<ModeWorkspace draft={draft} settings={DEFAULT_SETTINGS} capabilities={{ ...EMPTY_CAPABILITIES, connection: "offline" }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);

    expect(screen.getByLabelText("起始帧")).toBeDisabled();
    expect(screen.getByText("配置 ComfyUI 地址并等待连接就绪后可上传素材")).toBeInTheDocument();
  });

  it.each([
    ["t2v", "此镜头由提示词直接生成，无需素材。"],
    ["i2v", "起始帧"],
    ["fl2v", "首帧"],
    ["r2v", "参考视频"],
    ["v2v", "源视频"],
    ["rv2v", "参考音频"],
  ] as const)("%s 渲染自己的专属镜头字段", (mode, expected) => {
    render(<ModeWorkspace draft={createInitialDrafts()[mode]} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);
    if (mode === "t2v") expect(screen.getByText(expected)).toBeInTheDocument();
    else expect(screen.getByLabelText(expected)).toBeInTheDocument();
    if (mode === "rv2v") expect(screen.queryByLabelText("参考视频")).not.toBeInTheDocument();
  });

  it("输入正向提示词时只返回当前模式草稿", async () => {
    const user = userEvent.setup(); let changed: ModeDraft | undefined;
    render(<ModeWorkspace draft={createInitialDrafts().t2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={(draft) => { changed = draft; }} onSubmit={() => {}} />);
    await user.type(screen.getByLabelText("默认提示词"), "雨夜追车");
    expect(changed?.mode).toBe("t2v");
    expect(changed && "source_video" in changed).toBe(false);
  });

  it("六种模式校验各自专属素材，不接受错误模式字段替代", () => {
    const drafts = createInitialDrafts();
    for (const mode of Object.keys(drafts) as (keyof typeof drafts)[]) drafts[mode].prompt = "镜头描述";
    expect(validateModeDraft(drafts.t2v)).toEqual([]);
    expect(validateModeDraft(drafts.i2v)).toContain("每个启用的 I2V 镜头都需要有效起始帧");
    expect(validateModeDraft(drafts.fl2v)).toContain("每个启用的 FL2V 镜头至少需要有效首帧或尾帧");
    expect(validateModeDraft(drafts.r2v)).toContain("每个启用的 R2V 参考组至少需要一个参考素材");
    expect(validateModeDraft(drafts.v2v)).toContain("每个启用的 V2V 片段都需要有效源视频");
    expect(validateModeDraft(drafts.rv2v)).toEqual(["每个启用的 RV2V 片段都需要有效源视频"]);
  });

  it("全局提示词为空时要求每个启用镜头分别填写提示词", () => {
    const draft = createInitialDrafts().t2v;
    draft.shots.push({ ...draft.shots[0], id: "t2v-shot-2", title: "镜头 02", prompt: "第二镜头" });
    expect(validateModeDraft(draft)).toContain("默认提示词为空时，每个启用镜头都必须填写提示词");
    draft.shots[0].prompt = "第一镜头";
    expect(validateModeDraft(draft)).toEqual([]);
  });

  it("准确说明默认提示词和镜头非空覆盖关系", () => {
    render(<ModeWorkspace draft={createInitialDrafts().t2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);
    expect(screen.getByText("默认提示词，镜头非空时覆盖。")).toBeInTheDocument();
    expect(screen.getByText("作为所有镜头的默认值；镜头提示词非空时覆盖")).toBeInTheDocument();
    expect(screen.getByText("非空时覆盖默认提示词；留空则继承默认提示词")).toBeInTheDocument();
  });

  it("MiniMax H3 表单不提供负面提示词", () => {
    render(<ModeWorkspace draft={createInitialDrafts().t2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);
    expect(screen.queryByText(/负面提示词/)).not.toBeInTheDocument();
    expect(screen.queryByLabelText("负面提示词")).not.toBeInTheDocument();
  });

  it("将四类通用字段归入受顶栏控制的当前模式全局设置", () => {
    const onChange = vi.fn();
    const onRuntimeModelChange = vi.fn();
    const view = render(<ModeWorkspace globalSettingsOpen draft={createInitialDrafts().t2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={onChange} onRuntimeModelChange={onRuntimeModelChange} onSubmit={() => {}} />);

    const globalSettings = screen.getByRole("region", { name: "当前模式全局设置" });
    expect(globalSettings).not.toHaveAttribute("hidden");
    expect(within(globalSettings).getByLabelText("Diffusion 模型快捷选择")).toBeInTheDocument();
    expect(within(globalSettings).getByText("输出规格")).toBeInTheDocument();
    expect(within(globalSettings).getByText("推理参数")).toBeInTheDocument();
    expect(within(globalSettings).getByLabelText("默认提示词")).toBeInTheDocument();
    expect(globalSettings).toHaveTextContent("FL2VA（T2V / I2V / FL2V）");
    expect(screen.getByLabelText("参考图采样尺寸").closest("section")).toHaveTextContent("推理参数");
    expect(globalSettings.contains(screen.getByText("镜头时间线").closest("section"))).toBe(false);

    view.rerender(<ModeWorkspace globalSettingsOpen={false} draft={createInitialDrafts().t2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={onChange} onRuntimeModelChange={onRuntimeModelChange} onSubmit={() => {}} />);
    expect(document.getElementById("workspace-global-settings")).toHaveAttribute("hidden");
    expect(screen.getByText("镜头时间线")).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
    expect(onRuntimeModelChange).not.toHaveBeenCalled();
  });

  it("全局设置可以修改共享模型族，但参考图尺寸只写入当前模式草稿", async () => {
    const user = userEvent.setup();
    const onRuntimeModelChange = vi.fn();
    const onChange = vi.fn();
    const quickSettings = structuredClone(DEFAULT_SETTINGS);
    quickSettings.models.fl2va.lora_name = "minimax-h3-turbo.safetensors";
    render(<ModeWorkspace draft={createInitialDrafts().t2v} settings={quickSettings} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} models={MODELS} submitting={false} onChange={onChange} onRuntimeModelChange={onRuntimeModelChange} onSubmit={() => {}} />);

    expect(screen.getByText("FL2VA（T2V / I2V / FL2V）")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Diffusion 模型快捷选择"), "fl2va-alt.safetensors");
    expect(onRuntimeModelChange).toHaveBeenCalledWith("fl2va", { filename: "fl2va-alt.safetensors" });
    await user.selectOptions(screen.getByLabelText("LoRA 模型快捷选择"), "cinema-style.safetensors");
    expect(onRuntimeModelChange).toHaveBeenCalledWith("fl2va", { lora_name: "cinema-style.safetensors" });
    expect(screen.queryByLabelText("LoRA 加载器快捷选择")).not.toBeInTheDocument();
    expect(screen.getByLabelText("LoRA 加载状态")).toHaveTextContent("自动探测");
    await user.selectOptions(screen.getByLabelText("参考图采样尺寸"), "max");
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ mode: "t2v", ref_image_size: "max" }));
  });

  it("随机种子将实际安全整数保留在灰选输入框，并可原值切回固定", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const draft = createInitialDrafts().t2v;
    draft.sampling.seed = 7;
    draft.sampling.random_seed = false;
    const { rerender } = render(<ModeWorkspace draft={draft} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={onChange} onSubmit={() => {}} />);

    expect(screen.getByLabelText("Seed")).toBeEnabled();
    expect(screen.getByLabelText("Seed")).toHaveValue(7);
    await user.click(screen.getByLabelText("随机种子"));
    const randomDraft = onChange.mock.calls.at(-1)?.[0] as ModeDraft;
    expect(randomDraft.sampling.random_seed).toBe(true);
    expect(Number.isSafeInteger(randomDraft.sampling.seed)).toBe(true);
    expect(randomDraft.sampling.seed).toBeGreaterThanOrEqual(0);
    rerender(<ModeWorkspace draft={randomDraft} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={onChange} onSubmit={() => {}} />);
    expect(screen.getByLabelText("Seed")).toBeDisabled();
    expect(screen.getByLabelText("Seed")).toHaveValue(randomDraft.sampling.seed);
    await user.click(screen.getByLabelText("随机种子"));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
      sampling: expect.objectContaining({
        random_seed: false,
        seed: randomDraft.sampling.seed,
      }),
    }));
  });

  it("画面和采样表单按原生 H3 契约固定帧率并不提供 CFG", () => {
    render(<ModeWorkspace draft={{ ...createInitialDrafts().t2v, prompt: "镜头" }} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);
    expect(screen.getByLabelText("宽度")).toHaveAttribute("step", "32");
    expect(screen.getByLabelText("宽度")).toHaveAttribute("max", "8192");
    expect(screen.getByLabelText("帧率")).toHaveTextContent("24 fps");
    expect(screen.queryByLabelText("CFG")).not.toBeInTheDocument();
    expect(screen.queryByText(/BasicGuider 固定值/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("步数")).toHaveAttribute("max", "200");
    expect(screen.getByLabelText("Shift")).toHaveAttribute("min", "0.01");
  });

  it("旧六模式表单也从统一契约提供 beta 调度器", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ModeWorkspace draft={createInitialDrafts().t2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={onChange} onSubmit={() => {}} />);

    const scheduler = screen.getByRole("combobox", { name: "调度器" });
    expect(within(scheduler).getAllByRole("option").map((option) => option.getAttribute("value"))).toEqual([
      "simple", "normal", "karras", "beta",
    ]);
    await user.selectOptions(scheduler, "beta");
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
      sampling: expect.objectContaining({ scheduler: "beta" }),
    }));
  });

  it("在提交前拦截后端的 32 倍数与 512 帧限制", () => {
    const draft = { ...createInitialDrafts().t2v, prompt: "镜头" };
    draft.render.width = 850;
    expect(validateModeDraft(draft)).toContain("画面宽高必须是 32–8192 范围内的 32 倍数");
    draft.render.width = 864;
    draft.render.fps = 240;
    expect(validateModeDraft(draft)).toContain("镜头时长与帧率组合超过 MiniMax H3 的 512 帧上限");
  });

  it("FL2V 独立要求至少 0.1 秒，其他模式仍只要求大于 0", () => {
    const t2v = { ...createInitialDrafts().t2v, prompt: "镜头" };
    t2v.shots[0].duration_seconds = 0.05;
    expect(validateModeDraft(t2v)).toEqual([]);

    const fl2v = { ...createInitialDrafts().fl2v, prompt: "镜头" };
    fl2v.shots[0].duration_seconds = 0.05;
    expect(validateModeDraft(fl2v)).toContain("FL2V 镜头时长必须在 0.1–120 秒之间");

    const { unmount } = render(<ModeWorkspace draft={fl2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);
    expect(screen.getByLabelText("生成时长（秒）")).toHaveAttribute("min", "0.1");
    unmount();
    render(<ModeWorkspace draft={t2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);
    expect(screen.getByLabelText("生成时长（秒）")).toHaveAttribute("min", "0.01");
  });

  it("显示对齐后的总有效帧、有效时长和单镜头帧数", () => {
    render(<ModeWorkspace draft={createInitialDrafts().t2v} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);
    expect(screen.getAllByText("124f").length).toBeGreaterThanOrEqual(1);
    expect(document.querySelector(".timeline-total")).toHaveTextContent("5.17s");
    expect(screen.getByText("输入 5s → 124 帧 · 实际 5.17s")).toBeInTheDocument();
  });

  it.each(["v2v", "rv2v"] as const)("%s 校验截取终点不超过探测视频时长", (mode) => {
    const draft = createInitialDrafts()[mode];
    draft.prompt = "视频重绘";
    draft.shots[0].source_video = sourceVideo;
    draft.shots[0].source_start_seconds = 8;
    draft.shots[0].source_duration_seconds = 4;
    expect(validateModeDraft(draft)).not.toContain("源视频截取范围不能超过素材实际时长");

    draft.shots[0].source_duration_seconds = 4.01;
    expect(validateModeDraft(draft)).toContain("源视频截取范围不能超过素材实际时长");
    draft.shots[0].source_start_seconds = Number.NaN;
    expect(validateModeDraft(draft)).toContain("源视频起点或时长超出 0–86400 秒范围");
  });

  it("源视频卡显示探测数据，并按起点动态限制可用时长", () => {
    const draft = createInitialDrafts().v2v;
    draft.prompt = "视频重绘";
    draft.shots[0].source_video = sourceVideo;
    draft.shots[0].source_start_seconds = 8;
    draft.shots[0].source_duration_seconds = 4;
    render(<ModeWorkspace draft={draft} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);

    expect(screen.getByText("12.00s · 30.00fps · 360帧 · 1920×1080")).toBeInTheDocument();
    expect(screen.getByLabelText("源起点（秒）")).toHaveAttribute("max", "12");
    expect(screen.getByLabelText("源时长（秒）")).toHaveAttribute("max", "4");
    expect(screen.getByLabelText("源时长（秒）")).toHaveAttribute("min", "0.01");
  });

  it("选择较短源视频时自动把默认时长调整为剩余可用时长", async () => {
    const user = userEvent.setup();
    const uploaded = {
      ...sourceVideo,
      metadata: { ...sourceVideo.metadata!, duration: 3, frame_count: 90 },
    };
    const uploadSpy = vi.spyOn(directorApi, "uploadAsset").mockResolvedValueOnce(uploaded);
    const draft = createInitialDrafts().v2v;
    draft.prompt = "视频重绘";
    let state = createInitialDirectorState();
    state = directorReducer(state, { type: "draft/replace", draft });
    const mutate = (
      mode: ModeDraft["mode"],
      shotId: string,
      field: DraftAssetField,
      mutation: AssetMutation,
    ) => {
      state = directorReducer(state, { type: "draft/assets", mode, shotId, field, mutation });
    };
    render(<ModeWorkspace draft={draft} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onAssetsChange={mutate} onSubmit={() => {}} />);

    await user.upload(
      screen.getByLabelText("源视频"),
      new File(["video"], "source.mp4", { type: "video/mp4" }),
    );
    expect(state.drafts.v2v.shots[0].source_video).toEqual(uploaded);
    expect(state.drafts.v2v.shots[0].source_duration_seconds).toBe(3);
    uploadSpy.mockRestore();
  });

  it("参考素材显示官方标签，删除素材不重排剩余 slot", async () => {
    const user = userEvent.setup();
    const draft = createInitialDrafts().r2v;
    draft.prompt = "参考 <Picture 1> 与 <Picture 3>";
    draft.shots[0].reference_images = [image("hero", 0), image("wardrobe", 2)];
    let state = createInitialDirectorState();
    state = directorReducer(state, { type: "draft/replace", draft });
    const mutate = (
      mode: ModeDraft["mode"],
      shotId: string,
      field: DraftAssetField,
      mutation: AssetMutation,
    ) => {
      state = directorReducer(state, { type: "draft/assets", mode, shotId, field, mutation });
    };
    render(<ModeWorkspace draft={draft} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onAssetsChange={mutate} onSubmit={() => {}} />);

    expect(screen.getByText("<Picture 1>")).toBeInTheDocument();
    expect(screen.getByText("<Picture 3>")).toBeInTheDocument();
    await user.click(screen.getByLabelText("移除 hero.png"));
    expect(state.drafts.r2v.shots[0].reference_images).toEqual([image("wardrobe", 2)]);
  });

  it("阻止提示词引用当前镜头不存在的稳定素材槽", () => {
    const r2v = createInitialDrafts().r2v;
    r2v.prompt = "让 <Picture 2> 出场";
    r2v.shots[0].reference_images = [image("hero", 0), image("wardrobe", 2)];
    expect(validateModeDraft(r2v)).toContain(
      "R2V 提示词引用了不存在的素材标签：<Picture 2>",
    );
    r2v.prompt = "让 <Picture 1> 和 <Picture 3> 出场";
    expect(validateModeDraft(r2v)).toContain(
      "R2V 各类参考素材槽位必须连续为 0..N-1",
    );
    r2v.shots[0].reference_images = [image("hero", 0), image("wardrobe", 1)];
    r2v.prompt = "让 <Picture 1> 和 <Picture 2> 出场";
    expect(validateModeDraft(r2v)).toEqual([]);

    const rv2v = createInitialDrafts().rv2v;
    rv2v.prompt = "编辑 <Video 2> 并参考 <Picture 1>";
    rv2v.shots[0].source_video = sourceVideo;
    rv2v.shots[0].source_duration_seconds = 5;
    rv2v.shots[0].reference_images = [image("hero", 0)];
    expect(validateModeDraft(rv2v)).toContain(
      "RV2V 提示词引用了不存在的素材标签：<Video 2>",
    );
  });

  it("素材标签解析与后端一样接受尖括号内空白", () => {
    const draft = createInitialDrafts().r2v;
    draft.prompt = "让 < Picture 2 > 出场";
    draft.shots[0].reference_images = [image("hero", 0)];
    expect(validateModeDraft(draft)).toContain(
      "R2V 提示词引用了不存在的素材标签：< Picture 2 >",
    );
  });

  it("RV2V 时间线按播放头等比分割生成区间与源区间", () => {
    const draft = createInitialDrafts().rv2v;
    draft.shots[0].source_video = sourceVideo;
    draft.shots[0].duration_seconds = 8;
    draft.shots[0].source_start_seconds = 2;
    draft.shots[0].source_duration_seconds = 4;
    draft.shots[0].reference_images = [image("hero", 0)];

    const split = splitRV2VShot(draft, draft.shots[0].id, 3);
    expect(split.shots).toHaveLength(2);
    expect(split.shots[0]).toMatchObject({
      duration_seconds: 3,
      source_start_seconds: 2,
      source_duration_seconds: 1.5,
    });
    expect(split.shots[1]).toMatchObject({
      duration_seconds: 5,
      source_start_seconds: 3.5,
      source_duration_seconds: 2.5,
    });
    expect(split.shots[1].id).not.toBe(split.shots[0].id);
    expect(split.shots[1].reference_images).toEqual(split.shots[0].reference_images);
    expect(split.shots[1].reference_images).not.toBe(split.shots[0].reference_images);

    const decimal = createInitialDrafts().rv2v;
    decimal.shots[0].source_video = {
      ...sourceVideo,
      metadata: { ...sourceVideo.metadata!, duration: 3.02 },
    };
    decimal.shots[0].source_duration_seconds = 3.02;
    const decimalSplit = splitRV2VShot(decimal, decimal.shots[0].id, 0.44);
    expect(validateModeDraft(decimalSplit)).not.toContain(
      "源视频截取范围不能超过素材实际时长",
    );

    const moved = moveRV2VShot(split, split.shots[1].id, -1);
    expect(moved.shots[0].id).toBe(split.shots[1].id);
    const deleted = deleteRV2VShot(moved, moved.shots[0].id);
    expect(deleted.shots).toHaveLength(1);
    expect(deleteRV2VShot(deleted, deleted.shots[0].id)).toBe(deleted);

    const full = {
      ...draft,
      shots: Array.from({ length: 128 }, (_, index) => ({
        ...structuredClone(draft.shots[0]),
        id: `full-${index}`,
      })),
    };
    expect(splitRV2VShot(full, full.shots[0].id, 2)).toBe(full);
  });

  it("RV2V 页面显示专属视觉轨道并可从默认播放头分割", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const draft = createInitialDrafts().rv2v;
    draft.prompt = "重绘源视频";
    draft.shots[0].source_video = sourceVideo;
    draft.shots[0].source_duration_seconds = 5;
    render(<ModeWorkspace draft={draft} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={onChange} onSubmit={() => {}} />);

    expect(screen.getByLabelText("RV2V 可视时间线")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "RV2V 片段轨道" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "在播放头分割" }));
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({
      mode: "rv2v",
      shots: expect.arrayContaining([expect.any(Object), expect.any(Object)]),
    }));
    expect(onChange.mock.calls.at(-1)?.[0].shots).toHaveLength(2);
  });

  it("RV2V 智能分镜只采用当前源范围内的全素材切点并映射变速输出", () => {
    const draft = createInitialDrafts().rv2v;
    draft.shots[0].source_video = sourceVideo;
    draft.shots[0].duration_seconds = 8;
    draft.shots[0].source_start_seconds = 2;
    draft.shots[0].source_duration_seconds = 4;
    draft.shots[0].reference_images = [image("hero", 0)];

    const split = splitRV2VShotAtSourceCuts(
      draft,
      draft.shots[0].id,
      [0, 20, 30, 45, 60, 120],
      10,
    );
    expect(split.shots.map((shot) => shot.duration_seconds)).toEqual([2, 3, 3]);
    expect(split.shots.map((shot) => shot.source_start_seconds)).toEqual([2, 3, 4.5]);
    expect(split.shots.map((shot) => shot.source_duration_seconds)).toEqual([1, 1.5, 1.5]);
    expect(split.shots.every((shot) => shot.reference_images[0]?.id === "hero")).toBe(true);
  });

  it("RV2V 智能分镜提交检测参数、展示等待态并应用返回切点", async () => {
    const user = userEvent.setup();
    let resolveDetection!: (value: { cut_frames: number[]; shot_count: number; warnings: string[] }) => void;
    const detect = vi.spyOn(directorApi, "detectRV2VShots").mockImplementation(
      () => new Promise((resolve) => { resolveDetection = resolve; }),
    );
    const onChange = vi.fn();
    const draft = createInitialDrafts().rv2v;
    draft.prompt = "重绘源视频";
    draft.shots[0].source_video = sourceVideo;
    draft.shots[0].source_duration_seconds = 5;
    const view = render(<ModeWorkspace draft={draft} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={onChange} onSubmit={() => {}} />);

    await user.selectOptions(screen.getByLabelText("智能分镜灵敏度"), "high");
    const minFrames = screen.getByLabelText("智能分镜最短镜头帧数");
    expect(minFrames).toHaveAttribute("min", "4");
    fireEvent.change(minFrames, { target: { value: "1" } });
    expect(minFrames).toHaveValue(1);
    expect(detect).not.toHaveBeenCalled();
    fireEvent.blur(minFrames);
    expect(minFrames).toHaveValue(4);
    await user.clear(minFrames);
    await user.type(minFrames, "100001");
    expect(minFrames).toHaveValue(100001);
    await user.keyboard("{Enter}");
    expect(minFrames).toHaveValue(100000);

    await user.clear(minFrames);
    expect((minFrames as HTMLInputElement).value).toBe("");
    await user.type(minFrames, "1");
    expect(minFrames).toHaveValue(1);
    await user.type(minFrames, "2");
    expect(minFrames).toHaveValue(12);
    await user.click(screen.getByRole("button", { name: "智能分镜" }));
    expect(detect).toHaveBeenCalledWith({
      asset_id: "source-video",
      frame_rate: 24,
      sensitivity: "high",
      min_shot_frames: 12,
    });
    expect(screen.getByRole("button", { name: "检测中…" })).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText("动作、构图、运镜…"), {
      target: { value: "检测期间新增的镜头说明" },
    });
    const editedDraft = onChange.mock.calls.at(-1)?.[0];
    view.rerender(<ModeWorkspace draft={editedDraft} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={onChange} onSubmit={() => {}} />);

    resolveDetection({ cut_frames: [0, 48, 96, 120, 288], shot_count: 4, warnings: ["低置信度切点已忽略"] });
    expect(await screen.findByText(/智能分镜完成：当前片段已拆为 3 段/)).toBeInTheDocument();
    expect(screen.getByText(/低置信度切点已忽略/)).toBeInTheDocument();
    expect(onChange.mock.calls.at(-1)?.[0].shots).toHaveLength(3);
    expect(onChange.mock.calls.at(-1)?.[0].shots[0].prompt).toBe("检测期间新增的镜头说明");
    detect.mockRestore();
  });

  it("RV2V 总播放头与所选片段的源视频区间双向联动", () => {
    const draft = createInitialDrafts().rv2v;
    draft.prompt = "重绘源视频";
    draft.shots[0].source_video = {
      ...sourceVideo,
      preview_url: "/api/assets/source-video/preview",
    };
    draft.shots[0].duration_seconds = 8;
    draft.shots[0].source_start_seconds = 2;
    draft.shots[0].source_duration_seconds = 4;
    const split = splitRV2VShot(draft, draft.shots[0].id, 3);
    render(<ModeWorkspace draft={split} settings={DEFAULT_SETTINGS} capabilities={{ ...capabilities, supported_modes: [...capabilities.supported_modes] }} submitting={false} onChange={() => {}} onSubmit={() => {}} />);

    let preview = document.querySelector(".rv2v-editor__preview video") as HTMLVideoElement;
    fireEvent.loadedMetadata(preview);
    expect(preview.currentTime).toBeCloseTo(2.75);

    const playhead = screen.getByLabelText("RV2V 播放头") as HTMLInputElement;
    fireEvent.change(playhead, { target: { value: "4" } });
    expect(within(screen.getByRole("group", { name: "RV2V 片段轨道" })).getAllByRole("button")[1]).toHaveAttribute("aria-pressed", "true");
    preview = document.querySelector(".rv2v-editor__preview video") as HTMLVideoElement;
    fireEvent.loadedMetadata(preview);
    expect(preview.currentTime).toBeCloseTo(4);

    preview.currentTime = 4.5;
    fireEvent.timeUpdate(preview);
    expect(playhead).toHaveValue("5");
  });
});
