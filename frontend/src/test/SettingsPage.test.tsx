import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { directorApi } from "../api/client";
import { DEFAULT_SETTINGS, EMPTY_CAPABILITIES, EMPTY_MODELS, resolveExecutionBackend, sanitizeRuntimeSettings, type ModelInventory, type RayLightRuntimeStatus, type RuntimeSettings } from "../api/types";
import { SettingsPage, validateRuntimeSettingsForm } from "../components/SettingsPage";

const CONFIGURED_SETTINGS: RuntimeSettings = {
  ...DEFAULT_SETTINGS,
};
const ONLINE_CAPABILITIES = { ...EMPTY_CAPABILITIES, connection: "online" as const };
const confirmConfiguredSettings = async () => CONFIGURED_SETTINGS;
const STORAGE_CONFIGURATION = {
  active_database_path: "/srv/directordeck/data/directordeck.sqlite3",
};
const MODEL_INVENTORY: ModelInventory = {
  fl2va: [DEFAULT_SETTINGS.models.fl2va.filename],
  ref2va: [DEFAULT_SETTINGS.models.ref2va.filename],
  clip: [DEFAULT_SETTINGS.models.clip.filename],
  video_vae: [DEFAULT_SETTINGS.models.video_vae.filename],
  audio_vae: [DEFAULT_SETTINGS.models.audio_vae.filename],
  loras: ["minimax-h3-turbo.safetensors", "style.safetensors"],
};

beforeEach(() => {
  vi.spyOn(directorApi, "getStorage").mockImplementation(() => new Promise(() => undefined));
});
afterEach(() => vi.restoreAllMocks());

describe("共享设置表单", () => {
  it("只在显式确认旧 ComfyUI 已重启后请求恢复不可见 GPU 的 RayLight 状态", async () => {
    const user = userEvent.setup();
    const onRecover = vi.fn(async () => undefined);
    const confirm = vi.spyOn(window, "confirm")
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
    const blockedStatus: RayLightRuntimeStatus = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3, 4, 5, 6, 7],
      available_gpu_indexes: [0, 1, 2, 3],
      invalid_gpu_indexes: [4, 5, 6, 7],
      tainted: false,
      recovery_token: "a".repeat(64),
    };
    const view = render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        rayLightRuntimeStatus={blockedStatus}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
        onConfirmRayLightRuntimeRecovery={onRecover}
      />,
    );

    const alert = screen.getByRole("alert", { name: "旧 RayLight 运行状态引用了当前不可见 GPU" });
    expect(alert).toHaveTextContent("旧 runtime 逻辑 GPU：0, 1, 2, 3, 4, 5, 6, 7");
    expect(alert).toHaveTextContent("当前可见逻辑 GPU：0, 1, 2, 3");
    expect(alert).toHaveTextContent("当前不可见逻辑 GPU：4, 5, 6, 7");
    expect(alert).toHaveTextContent("不是物理显卡编号");
    const recover = within(alert).getByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    });
    expect(recover).toHaveAccessibleDescription(
      /测试连接成功或当前队列为空都不能证明旧进程已退出/,
    );

    await user.click(recover);
    expect(onRecover).not.toHaveBeenCalled();
    expect(confirm).toHaveBeenLastCalledWith(expect.stringContaining("测试连接成功或当前队列为空，都不能证明旧进程已经退出"));
    expect(confirm).toHaveBeenLastCalledWith(expect.stringContaining("当前不可见逻辑 GPU：4, 5, 6, 7"));

    await user.click(recover);
    await waitFor(() => expect(onRecover).toHaveBeenCalledTimes(1));

    view.rerender(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        rayLightRuntimeStatus={blockedStatus}
        rayLightRecoveryPending
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
        onConfirmRayLightRuntimeRecovery={onRecover}
      />,
    );
    const pending = screen.getByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    });
    expect(pending).toBeDisabled();
    expect(pending).toHaveAttribute("aria-busy", "true");

    view.rerender(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        rayLightRuntimeStatus={{
          ...blockedStatus,
          active: false,
          recovery_required: false,
          runtime_gpu_indexes: [],
          invalid_gpu_indexes: [],
          recovery_token: null,
        }}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );
    expect(screen.queryByRole("alert", { name: "旧 RayLight 运行状态引用了当前不可见 GPU" })).not.toBeInTheDocument();
  });

  it("连接卡为只读状态并保留测试连接按钮", () => {
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    expect(screen.queryByLabelText("ComfyUI 地址")).not.toBeInTheDocument();
    expect(screen.getByText("ComfyUI 在线")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "测试连接" })).toBeEnabled();
  });

  it("数据库路径默认隐藏，显示切换不改动只读值", async () => {
    const user = userEvent.setup();
    vi.mocked(directorApi.getStorage).mockResolvedValue(STORAGE_CONFIGURATION);
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    const currentStorage = await screen.findByLabelText("当前数据库存储配置");
    const currentVisibility = screen.getByRole("button", { name: "当前数据库路径显示状态" });
    expect(currentStorage).not.toHaveTextContent(STORAGE_CONFIGURATION.active_database_path);
    expect(currentVisibility).toHaveAttribute("aria-pressed", "false");
    await user.click(currentVisibility);
    expect(currentStorage).toHaveTextContent(STORAGE_CONFIGURATION.active_database_path);
    expect(currentVisibility).toHaveAttribute("aria-pressed", "true");
  });

  it("模型族就绪只读取两族原生时间线能力，不把旧六配方当成可选模式", () => {
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        resourcesReady
        capabilities={{
          ...ONLINE_CAPABILITIES,
          supported_modes: ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"],
          native_timeline: {
            supported: true,
            modes: ["fl2va"],
            continuity: false,
          },
        }}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    const panel = screen.getByRole("heading", { name: "模型族就绪情况" }).closest("section");
    if (!panel) throw new Error("family readiness panel missing");
    const rows = within(panel).getAllByRole("listitem");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("FL2VA");
    expect(rows[0]).toHaveTextContent("可用");
    expect(rows[1]).toHaveTextContent("Ref2VA");
    expect(rows[1]).toHaveTextContent("未就绪");
    for (const legacyRecipe of ["T2V", "I2V", "FL2V", "R2V", "V2V", "RV2V"]) {
      expect(within(panel).queryByText(legacyRecipe, { exact: true })).not.toBeInTheDocument();
    }
  });

  it("数据存储面板只读展示当前数据库路径", async () => {
    vi.mocked(directorApi.getStorage).mockResolvedValue(STORAGE_CONFIGURATION);
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    expect(await screen.findByLabelText("当前数据库存储配置")).toBeInTheDocument();
    expect(screen.queryByLabelText("数据库目标路径")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /迁移/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /保存路径/ })).not.toBeInTheDocument();
  });

  it("测试连接探测当前宿主实例并触发资源重核对", async () => {
    const user = userEvent.setup();
    let resolveProbe!: (result: { ok: boolean; latency_ms?: number; message: string }) => void;
    const probe = vi.spyOn(directorApi, "testConnection").mockImplementation(
      () => new Promise((resolve) => { resolveProbe = resolve; }),
    );
    const onConnectionTestSucceeded = vi.fn();
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
        onConnectionTestSucceeded={onConnectionTestSucceeded}
      />,
    );

    await user.click(screen.getByRole("button", { name: "测试连接" }));

    expect(probe).toHaveBeenCalledWith();
    const connectionCard = screen.getByText("正在测试连接").closest(".connection-card");
    expect(connectionCard).not.toBeNull();
    expect(connectionCard).toHaveTextContent("正在等待 ComfyUI 响应");
    expect(screen.getByRole("button", { name: /测试中/ })).toBeDisabled();
    expect(onConnectionTestSucceeded).not.toHaveBeenCalled();

    await act(async () => resolveProbe({ ok: true, latency_ms: 2.5, message: "连接成功" }));

    expect(await screen.findByText("当前实例可连接")).toBeInTheDocument();
    expect(screen.getByText("连接成功 · 响应 2.5 ms")).toBeInTheDocument();
    expect(onConnectionTestSucceeded).toHaveBeenCalledOnce();
    expect(onConnectionTestSucceeded).toHaveBeenCalledWith();
  });

  it("测试连接失败直接显示在连接卡且不触发资源重核对", async () => {
    const user = userEvent.setup();
    vi.spyOn(directorApi, "testConnection").mockResolvedValue({
      ok: false,
      message: "连接被拒绝",
    });
    const onConnectionTestSucceeded = vi.fn();
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
        onConnectionTestSucceeded={onConnectionTestSucceeded}
      />,
    );

    await user.click(screen.getByRole("button", { name: "测试连接" }));

    expect(await screen.findByText("当前实例连接失败")).toBeInTheDocument();
    expect(screen.getByText("连接被拒绝")).toBeInTheDocument();
    expect(onConnectionTestSucceeded).not.toHaveBeenCalled();
  });

  it("父级权威模型清单到达后启用模型控件", () => {
    const { rerender } = render(
      <SettingsPage
        settings={DEFAULT_SETTINGS}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={EMPTY_MODELS}
        loadingModels={true}
        onSaved={confirmConfiguredSettings}
      />,
    );

    rerender(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );
    expect(screen.getByLabelText("FL2VA 扩散模型模型")).toBeEnabled();
  });

  it("两个扩散模型槽都展示完整且去重的 diffusion 清单", () => {
    const models: ModelInventory = {
      fl2va: ["model-a.safetensors", "shared-model.safetensors"],
      ref2va: ["model-b.safetensors", "shared-model.safetensors"],
      clip: [DEFAULT_SETTINGS.models.clip.filename],
      video_vae: [DEFAULT_SETTINGS.models.video_vae.filename],
      audio_vae: [DEFAULT_SETTINGS.models.audio_vae.filename],
      loras: [],
    };
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={models}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    const fl2vaOptions = Array.from(
      (screen.getByLabelText("FL2VA 扩散模型模型") as HTMLSelectElement).options,
    ).map((option) => option.value);
    const ref2vaOptions = Array.from(
      (screen.getByLabelText("REF2VA 扩散模型模型") as HTMLSelectElement).options,
    ).map((option) => option.value);
    for (const model of ["model-a.safetensors", "model-b.safetensors", "shared-model.safetensors"]) {
      expect(fl2vaOptions.filter((value) => value === model)).toHaveLength(1);
      expect(ref2vaOptions.filter((value) => value === model)).toHaveLength(1);
    }
  });

  it("权威清单不含已配置模型/LoRA 文件时显示占位并明确提示，不出现幻影项", () => {
    const models: ModelInventory = {
      fl2va: ["model-a.safetensors"],
      ref2va: ["model-a.safetensors"],
      clip: [DEFAULT_SETTINGS.models.clip.filename],
      video_vae: [DEFAULT_SETTINGS.models.video_vae.filename],
      audio_vae: [DEFAULT_SETTINGS.models.audio_vae.filename],
      loras: ["style.safetensors"],
    };
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.models.fl2va.lora_name = "missing-lora.safetensors";
    render(
      <SettingsPage
        settings={settings}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={models}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    const fl2vaSelect = screen.getByLabelText("FL2VA 扩散模型模型") as HTMLSelectElement;
    expect(Array.from(fl2vaSelect.options).map((option) => option.value))
      .toEqual(["", "model-a.safetensors"]);
    expect(fl2vaSelect.value).toBe("");
    const loraSelect = screen.getByLabelText("FL2VA 扩散模型 LoRA") as HTMLSelectElement;
    expect(Array.from(loraSelect.options).map((option) => option.value))
      .toEqual(["", "style.safetensors"]);
    expect(loraSelect.value).toBe("");
    expect(screen.getAllByText(/不在当前 ComfyUI 模型清单中/)).toHaveLength(2);
    expect(screen.getByText(/不在当前 ComfyUI 清单中/)).toHaveTextContent("missing-lora.safetensors");
  });

  it("清单尚未就绪时保留已配置名称且不报缺失", () => {
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={EMPTY_MODELS}
        loadingModels={true}
        onSaved={confirmConfiguredSettings}
      />,
    );

    const select = screen.getByLabelText("FL2VA 扩散模型模型") as HTMLSelectElement;
    expect(select.value).toBe(CONFIGURED_SETTINGS.models.fl2va.filename);
    expect(screen.queryByText(/不在当前 ComfyUI 模型清单中/)).not.toBeInTheDocument();
  });

  it("两个扩散模型槽从同一清单独立自动同步各自选择", async () => {
    const user = userEvent.setup();
    const models: ModelInventory = {
      fl2va: ["model-a.safetensors", "model-b.safetensors"],
      ref2va: ["model-a.safetensors", "model-b.safetensors"],
      clip: [DEFAULT_SETTINGS.models.clip.filename],
      video_vae: [DEFAULT_SETTINGS.models.video_vae.filename],
      audio_vae: [DEFAULT_SETTINGS.models.audio_vae.filename],
      loras: ["style.safetensors"],
    };
    const confirmed = structuredClone(CONFIGURED_SETTINGS);
    confirmed.models.fl2va.filename = "model-a.safetensors";
    confirmed.models.ref2va.filename = "model-b.safetensors";
    const saveSettings = vi.fn(async () => confirmed);
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={models}
        loadingModels={false}
        onSaved={saveSettings}
      />,
    );

    await user.selectOptions(screen.getByLabelText("FL2VA 扩散模型模型"), "model-a.safetensors");
    await user.selectOptions(screen.getByLabelText("REF2VA 扩散模型模型"), "model-b.safetensors");
    await waitFor(() => expect(saveSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        models: expect.objectContaining({
          fl2va: expect.objectContaining({ filename: "model-a.safetensors" }),
          ref2va: expect.objectContaining({ filename: "model-b.safetensors" }),
        }),
      }),
    ));
  });

  it("切换运行环境进入检测态时使用父级清空后的模型清单", async () => {
    const oldModels: ModelInventory = {
      fl2va: [DEFAULT_SETTINGS.models.fl2va.filename, "old-endpoint-fl2va.safetensors"],
      ref2va: [DEFAULT_SETTINGS.models.ref2va.filename],
      clip: [DEFAULT_SETTINGS.models.clip.filename],
      video_vae: [DEFAULT_SETTINGS.models.video_vae.filename],
      audio_vae: [DEFAULT_SETTINGS.models.audio_vae.filename],
      loras: [],
    };
    const { rerender } = render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={oldModels}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );
    expect(
      await screen.findAllByRole("option", { name: "old-endpoint-fl2va.safetensors" }),
    ).toHaveLength(2);

    rerender(
      <SettingsPage
        settings={{ ...CONFIGURED_SETTINGS, client_id: "changed-client" }}
        capabilities={{ ...EMPTY_CAPABILITIES, connection: "checking" }}
        gpus={[]}
        models={EMPTY_MODELS}
        loadingModels={true}
        onSaved={confirmConfiguredSettings}
      />,
    );
    await waitFor(() =>
      expect(
        screen.queryAllByRole("option", { name: "old-endpoint-fl2va.safetensors" }),
      ).toHaveLength(0),
    );
    expect(screen.getByLabelText("FL2VA 扩散模型模型")).toBeDisabled();
  });

  it("普通模型允许 CPU，但两个 VAE 设备列表不提供 CPU", () => {
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[{ index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true }]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );
    const diffusionValues = Array.from(
      (screen.getByLabelText("FL2VA 扩散模型设备") as HTMLSelectElement).options,
    ).map((option) => option.value);
    const videoVaeValues = Array.from(
      (screen.getByLabelText("视频 VAE设备") as HTMLSelectElement).options,
    ).map((option) => option.value);
    const audioVaeValues = Array.from(
      (screen.getByLabelText("音频 VAE设备") as HTMLSelectElement).options,
    ).map((option) => option.value);
    expect(diffusionValues).toEqual(["default", "cpu", "gpu:0"]);
    expect(videoVaeValues).toEqual(["default", "gpu:0"]);
    expect(audioVaeValues).toEqual(["default", "gpu:0"]);
  });

  it("在调用设置 API 前校验 client_id、模型文件和 VAE 设备", () => {
    const invalid = structuredClone(DEFAULT_SETTINGS) as RuntimeSettings;
    invalid.client_id = "包含 空格";
    invalid.models.fl2va.filename = "";
    (invalid.models.video_vae as { device: string }).device = "cpu";

    const errors = validateRuntimeSettingsForm(invalid);
    expect(errors).toContain("客户端 ID 只能包含字母、数字、点、下划线、冒号和连字符");
    expect(errors).toContain("FL2VA 扩散模型必须选择有效模型文件");
    expect(errors).toContain("视频 VAE不允许放在 CPU");

    (invalid as { memory_policy: string }).memory_policy = "unknown";
    expect(validateRuntimeSettingsForm(invalid)).toContain("片段间显存策略无效");
  });

  it("原生分段子任务将旧逐段清理策略归一为稳定 loader 复用", () => {
    const legacy = structuredClone(CONFIGURED_SETTINGS) as unknown as Record<string, unknown>;
    legacy.memory_policy = "clear_between_segments";
    legacy.raylight_residency_policy = "dedicated_keep_fl2va";
    const normalized = sanitizeRuntimeSettings(legacy);
    expect(normalized.memory_policy).toBe("keep_resident");
    expect(normalized.raylight_residency_policy).toBe("keep_until_switch");
    render(
      <SettingsPage
        settings={normalized}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    const policy = screen.getByLabelText("片段间显存策略");
    expect(policy).toHaveTextContent("Standard 稳定 loader 复用");
    expect(screen.getByLabelText("RayLight 显存驻留策略")).toHaveValue(
      "keep_until_switch",
    );
    expect(screen.getByText(/family、模型、LoRA、GPU 池和拓扑完全一致时复用/)).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /片段间清理/ })).not.toBeInTheDocument();
  });

  it("RayLight 按完整配置 key 常驻并明确安全切换边界", () => {
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.raylight_residency_policy = "keep_until_switch";
    render(
      <SettingsPage
        settings={settings}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    expect(screen.getByLabelText("RayLight 显存驻留策略")).toHaveValue(
      "keep_until_switch",
    );
    expect(screen.getByText(/family、模型、LoRA、GPU 池和拓扑完全一致时复用/)).toBeInTheDocument();
    expect(screen.getByText(/RayKill 安全屏障/)).toBeInTheDocument();
  });

  it("RayLight 驻留策略作为权威设置自动写回后端", async () => {
    const user = userEvent.setup();
    const saveSettings = vi.fn(async (value: RuntimeSettings) => value);
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.models.ref2va.backend = "auto";
    settings.models.ref2va.raylight = {
      ...settings.models.ref2va.raylight,
      gpu_select: [0, 1],
      ulysses_degree: 2,
    };
    settings.raylight_residency_policy = "release_after_sampling";
    render(
      <SettingsPage
        settings={settings}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={saveSettings}
      />,
    );

    await user.selectOptions(
      screen.getByLabelText("RayLight 显存驻留策略"),
      "keep_until_switch",
    );
    await waitFor(() => expect(saveSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        raylight_residency_policy: "keep_until_switch",
      }),
    ));
  });

  it("GPU 池自动解析执行后端，并保留用户明确选择的释放策略", async () => {
    const user = userEvent.setup();
    const saveSettings = vi.fn(async (value: RuntimeSettings) => value);
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.multi_gpu_enabled = true;
    settings.raylight_residency_policy = "release_after_sampling";
    settings.models.fl2va.backend = "auto";
    settings.models.fl2va.raylight = {
      ...settings.models.fl2va.raylight,
      gpu_select: [0, 1],
      ulysses_degree: 2,
    };
    render(
      <SettingsPage
        settings={settings}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[
          { index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 1, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 2, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
        ]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={saveSettings}
      />,
    );

    const topology = screen.getByRole("region", { name: "FL2VA 扩散模型执行拓扑" });
    expect(within(topology).getByText(/RAYLIGHT · 2 卡/)).toBeInTheDocument();
    expect(screen.queryByLabelText("FL2VA 扩散模型执行后端")).not.toBeInTheDocument();

    await user.selectOptions(
      screen.getByLabelText("RayLight 显存驻留策略"),
      "release_after_sampling",
    );
    expect(screen.getByText(/已明确选择任务后释放.*下次任务将重新加载/)).toBeInTheDocument();
    await user.selectOptions(
      screen.getByLabelText("FL2VA 扩散模型 LoRA"),
      "style.safetensors",
    );
    await user.click(
      screen.getByLabelText("FL2VA 扩散模型 RayLight 逻辑 GPU 2"),
    );
    expect(screen.getByLabelText("RayLight 显存驻留策略")).toHaveValue(
      "release_after_sampling",
    );
    await waitFor(() => expect(saveSettings).toHaveBeenCalledWith(
      expect.objectContaining({
        raylight_residency_policy: "release_after_sampling",
        models: expect.objectContaining({
          fl2va: expect.objectContaining({
            backend: "auto",
            lora_name: "style.safetensors",
          }),
        }),
      }),
    ));
  });

  it("Ref2VA GPU 池从单卡改为多卡时自动切入 RayLight", async () => {
    const user = userEvent.setup();
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.multi_gpu_enabled = true;
    settings.models.ref2va.backend = "auto";
    render(
      <SettingsPage
        settings={settings}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[
          { index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 1, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
        ]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    const topology = screen.getByRole("region", { name: "REF2VA 扩散模型执行拓扑" });
    expect(within(topology).getByText(/STANDARD · default/)).toBeInTheDocument();
    await user.click(screen.getByLabelText("REF2VA 扩散模型 RayLight 逻辑 GPU 1"));
    expect(within(topology).getByText(/RAYLIGHT · 2 卡/)).toBeInTheDocument();
  });

  it("Auto 后端的 GPU 池扩为多卡并解析到 RayLight 时自动常驻", async () => {
    const user = userEvent.setup();
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.multi_gpu_enabled = true;
    settings.raylight_residency_policy = "release_after_sampling";
    render(
      <SettingsPage
        settings={settings}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[
          { index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 1, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
        ]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    expect(screen.queryByLabelText("FL2VA 扩散模型执行后端")).not.toBeInTheDocument();
    await user.click(
      screen.getByLabelText("FL2VA 扩散模型 RayLight 逻辑 GPU 1"),
    );

    expect(screen.getByLabelText("RayLight 显存驻留策略")).toHaveValue(
      "keep_until_switch",
    );
    expect(screen.getByText(/FL2VA 已切换为 RayLight，已自动选择“按运行配置常驻”/)).toBeInTheDocument();
  });

  it("按 key 常驻不再永久绑定某个模型族", () => {
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.raylight_residency_policy = "keep_until_switch";
    expect(validateRuntimeSettingsForm(settings)).toEqual([]);
  });

  it("LoRA 强度遵循 -10–10 边界并拒绝旧隐藏加载契约", () => {
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.models.fl2va.lora_name = "style.safetensors";
    settings.models.fl2va.lora_strength = -10;
    settings.models.ref2va.lora_strength = 10;
    expect(validateRuntimeSettingsForm(settings)).toEqual([]);

    settings.models.fl2va.lora_strength = -10.01;
    settings.models.ref2va.lora_strength = 10.01;
    (settings.models.ref2va as { lora_loader: string }).lora_loader = "unknown";
    const errors = validateRuntimeSettingsForm(settings);
    expect(errors).toContain("FL2VA 扩散模型的 LoRA 强度必须在 -10–10 之间");
    expect(errors).toContain("REF2VA 扩散模型的 LoRA 强度必须在 -10–10 之间");
    expect(errors).toContain("REF2VA 扩散模型的 LoRA 加载方式必须由系统自动选择");
  });

  it("Standard LoRA 默认自动探测，并允许可见的作用域化显式选择", async () => {
    const user = userEvent.setup();
    const confirmed = structuredClone(CONFIGURED_SETTINGS);
    confirmed.models.fl2va.lora_name = "style.safetensors";
    confirmed.models.fl2va.lora_strength = 1.25;
    confirmed.models.fl2va.lora_loader = "auto";
    confirmed.models.fl2va.lora_low_vram = false;
    const saveSettings = vi.fn(async () => confirmed);
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={saveSettings}
      />,
    );

    const lora = screen.getByLabelText("FL2VA 扩散模型 LoRA");
    const strength = screen.getByLabelText("FL2VA 扩散模型 LoRA 强度");
    const loader = screen.getByLabelText("FL2VA 扩散模型 Standard LoRA 加载器");
    expect(screen.queryByLabelText("FL2VA 扩散模型 LoRA 低显存模式")).not.toBeInTheDocument();
    expect(strength).toHaveAttribute("min", "-10");
    expect(strength).toHaveAttribute("max", "10");
    expect(loader).toBeDisabled();
    await user.selectOptions(lora, "style.safetensors");
    expect(strength).toBeEnabled();
    expect(loader).toBeEnabled();
    expect(loader).toHaveValue("");
    await user.selectOptions(loader, "model_only");
    await user.clear(strength);
    await user.type(strength, "1.25");
    await user.tab();
    await waitFor(() => expect(saveSettings).toHaveBeenCalledWith(expect.objectContaining({
      models: expect.objectContaining({
        fl2va: expect.objectContaining({
          lora_name: "style.safetensors",
          lora_strength: 1.25,
          lora_loader: "auto",
          standard_lora_loader_override: {
            loader: "model_only",
            lora_name: "style.safetensors",
            model_filename: CONFIGURED_SETTINGS.models.fl2va.filename,
          },
          lora_low_vram: false,
        }),
      }),
    })));
  });

  it("旧设置安全迁移为标准单卡，RayLight 拓扑始终严格匹配 GPU 池", () => {
    const legacy = structuredClone(CONFIGURED_SETTINGS) as unknown as Record<string, unknown>;
    const legacyModels = legacy.models as Record<string, Record<string, unknown>>;
    delete legacyModels.fl2va.backend;
    delete legacyModels.fl2va.raylight;
    legacyModels.ref2va.backend = "raylight";
    legacyModels.ref2va.raylight = {
      gpu_select: [0, 1, 1, 999],
      ulysses_degree: 1,
      ring_degree: 1,
      cfg_degree: 9,
      dp_degree: 4,
      fsdp: false,
      cpu_offload: true,
    };

    const normalized = sanitizeRuntimeSettings(legacy);
    expect(normalized.models.fl2va).toMatchObject({
      backend: "auto",
      raylight: { gpu_select: [0], ulysses_degree: 1, ring_degree: 1 },
    });
    expect(resolveExecutionBackend(normalized.models.fl2va)).toBe("standard");
    expect(normalized.models.ref2va.raylight).toMatchObject({
      gpu_select: [0, 1],
      ulysses_degree: 2,
      ring_degree: 1,
      cfg_degree: 1,
      dp_degree: 1,
      fsdp: false,
      cpu_offload: false,
    });
    expect(normalized.models.ref2va.device).toBe("default");
    expect(resolveExecutionBackend(normalized.models.ref2va)).toBe("raylight");
  });

  it("只保留与底模和 LoRA 完全匹配的 Standard loader 覆盖", () => {
    const settings = structuredClone(CONFIGURED_SETTINGS);
    settings.models.fl2va.lora_name = "style.safetensors";
    settings.models.fl2va.standard_lora_loader_override = {
      loader: "model_only",
      lora_name: "style.safetensors",
      model_filename: settings.models.fl2va.filename,
    };
    expect(sanitizeRuntimeSettings(settings).models.fl2va.standard_lora_loader_override).toEqual(
      settings.models.fl2va.standard_lora_loader_override,
    );

    settings.models.fl2va.standard_lora_loader_override!.model_filename = "stale.safetensors";
    expect(sanitizeRuntimeSettings(settings).models.fl2va.standard_lora_loader_override).toBeNull();
  });

  it("原生时间线 v1 对 FSDP 与 CPU offload 失败封闭", () => {
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    const fsdp = screen.getByLabelText("FL2VA 扩散模型 RayLight FSDP");
    const offload = screen.getByLabelText("FL2VA 扩散模型 RayLight CPU offload");
    expect(offload).toBeDisabled();
    expect(offload).not.toBeChecked();
    expect(fsdp).toBeDisabled();
    expect(fsdp).not.toBeChecked();
  });

  it("多卡开关只认显式 true，其余一律视为关闭", () => {
    const enabled = structuredClone(CONFIGURED_SETTINGS);
    enabled.multi_gpu_enabled = true;
    expect(sanitizeRuntimeSettings(enabled).multi_gpu_enabled).toBe(true);
    const missing = structuredClone(CONFIGURED_SETTINGS) as unknown as Record<string, unknown>;
    delete missing.multi_gpu_enabled;
    expect(sanitizeRuntimeSettings(missing).multi_gpu_enabled).toBe(false);
    const bogus = structuredClone(CONFIGURED_SETTINGS) as unknown as Record<string, unknown>;
    bogus.multi_gpu_enabled = "yes";
    expect(sanitizeRuntimeSettings(bogus).multi_gpu_enabled).toBe(false);
  });

  it("多卡推理开关随设置渲染并提交切换", async () => {
    const user = userEvent.setup();
    const saved: RuntimeSettings[] = [];
    vi.spyOn(directorApi, "getRayLightSetup").mockResolvedValue({
      enabled: false,
      platform_supported: true,
      dependencies_installed: false,
      requirements_available: true,
      install: { state: "idle", log_tail: [], returncode: null, error: null, started_at: null },
    });
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={async (next) => { saved.push(next); return next; }}
      />,
    );
    const toggle = await screen.findByLabelText("启用多卡推理");
    expect(toggle).not.toBeChecked();
    await user.click(toggle);
    await waitFor(() => expect(saved.length).toBeGreaterThan(0));
    expect(saved.at(-1)?.multi_gpu_enabled).toBe(true);
  });

  it("未启用多卡时 GPU 池勾选整体禁用并提示", () => {
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[
          { index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 1, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
        ]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );

    expect(screen.getByLabelText("FL2VA 扩散模型 RayLight 逻辑 GPU 1")).toBeDisabled();
    expect(screen.getByLabelText("REF2VA 扩散模型 RayLight 逻辑 GPU 1")).toBeDisabled();
    expect(screen.getAllByText("启用多卡推理后可配置多卡 GPU 池。")).toHaveLength(2);
  });

  it("池为多卡时关闭多卡开关会把两族收敛为单卡并重置拓扑", async () => {
    const user = userEvent.setup();
    const saved: RuntimeSettings[] = [];
    const enabled = structuredClone(CONFIGURED_SETTINGS);
    enabled.multi_gpu_enabled = true;
    for (const family of ["fl2va", "ref2va"] as const) {
      enabled.models[family].raylight = {
        ...enabled.models[family].raylight,
        gpu_select: [0, 1],
        ulysses_degree: 2,
        ring_degree: 1,
      };
    }
    vi.spyOn(directorApi, "getRayLightSetup").mockResolvedValue({
      enabled: true,
      platform_supported: true,
      dependencies_installed: true,
      requirements_available: true,
      install: { state: "idle", log_tail: [], returncode: null, error: null, started_at: null },
    });
    render(
      <SettingsPage
        settings={enabled}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[
          { index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 1, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
        ]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={async (next) => { saved.push(next); return next; }}
      />,
    );

    await user.click(await screen.findByLabelText("启用多卡推理"));

    await waitFor(() => expect(saved.length).toBeGreaterThan(0));
    const last = saved.at(-1)!;
    expect(last.multi_gpu_enabled).toBe(false);
    for (const family of ["fl2va", "ref2va"] as const) {
      expect(last.models[family].raylight.gpu_select).toEqual([0]);
      expect(last.models[family].raylight.ulysses_degree).toBe(1);
      expect(last.models[family].raylight.ring_degree).toBe(1);
    }
  });

  it("启用多卡后 GPU 池可正常多选", async () => {
    const user = userEvent.setup();
    const saved: RuntimeSettings[] = [];
    const enabled = structuredClone(CONFIGURED_SETTINGS);
    enabled.multi_gpu_enabled = true;
    vi.spyOn(directorApi, "getRayLightSetup").mockResolvedValue({
      enabled: true,
      platform_supported: true,
      dependencies_installed: true,
      requirements_available: true,
      install: { state: "idle", log_tail: [], returncode: null, error: null, started_at: null },
    });
    render(
      <SettingsPage
        settings={enabled}
        resourcesReady
        capabilities={ONLINE_CAPABILITIES}
        gpus={[
          { index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 1, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
        ]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={async (next) => { saved.push(next); return next; }}
      />,
    );

    const checkbox = await screen.findByLabelText("FL2VA 扩散模型 RayLight 逻辑 GPU 1");
    expect(checkbox).toBeEnabled();
    await user.click(checkbox);
    await waitFor(() => expect(saved.length).toBeGreaterThan(0));
    expect(saved.at(-1)?.models.fl2va.raylight.gpu_select).toEqual([0, 1]);
  });

  it("开启多卡后可从设置页发起组件安装", async () => {
    const user = userEvent.setup();
    const enabled = structuredClone(CONFIGURED_SETTINGS);
    enabled.multi_gpu_enabled = true;
    vi.spyOn(directorApi, "getRayLightSetup").mockResolvedValue({
      enabled: true,
      platform_supported: true,
      dependencies_installed: false,
      requirements_available: true,
      install: { state: "idle", log_tail: [], returncode: null, error: null, started_at: null },
    });
    const install = vi.spyOn(directorApi, "installRayLight").mockResolvedValue({
      state: "running",
      log_tail: [],
      returncode: null,
      error: null,
      started_at: 1,
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(
      <SettingsPage
        settings={enabled}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
      />,
    );
    const installButton = await screen.findByRole("button", { name: "安装多卡组件" });
    await user.click(installButton);
    expect(install).toHaveBeenCalledTimes(1);
  });

  it("多卡 GPU 池自动使用 RayLight 并清除隐藏旧设置", async () => {
    const user = userEvent.setup();
    const configured = structuredClone(CONFIGURED_SETTINGS);
    configured.multi_gpu_enabled = true;
    configured.models.fl2va.device = "gpu:1";
    configured.models.fl2va.lora_name = "style.safetensors";
    configured.models.fl2va.lora_loader = "dedicated";
    configured.models.fl2va.lora_low_vram = true;
    configured.models.fl2va.raylight = {
      ...configured.models.fl2va.raylight,
      gpu_select: [0, 1],
      ulysses_degree: 2,
    };
    const normalized = sanitizeRuntimeSettings(configured);
    const confirmed = structuredClone(normalized);
    const saveSettings = vi.fn(async () => confirmed);
    render(
      <SettingsPage
        settings={normalized}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[
          { index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 1, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
        ]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={saveSettings}
      />,
    );

    expect(screen.queryByLabelText("FL2VA 扩散模型执行后端")).not.toBeInTheDocument();
    expect(screen.getByLabelText("FL2VA 扩散模型设备")).toHaveValue("default");
    expect(screen.queryByLabelText("FL2VA 扩散模型 LoRA 低显存模式")).not.toBeInTheDocument();
    expect(normalized.models.fl2va).toMatchObject({ backend: "auto", device: "default", lora_low_vram: false, lora_loader: "auto", standard_lora_loader_override: null });
    expect(screen.queryByRole("button", { name: "保存设置" })).not.toBeInTheDocument();
    expect(saveSettings).not.toHaveBeenCalled();
  });

  it("增加第二张逻辑 GPU 后自动切入 RayLight 并原子更新 Ulysses 拓扑", async () => {
    const user = userEvent.setup();
    const confirmed = structuredClone(CONFIGURED_SETTINGS);
    confirmed.multi_gpu_enabled = true;
    confirmed.models.fl2va.backend = "auto";
    confirmed.models.fl2va.raylight = {
      ...confirmed.models.fl2va.raylight,
      gpu_select: [0, 1],
      ulysses_degree: 2,
      ring_degree: 1,
    };
    const saveSettings = vi.fn(async () => confirmed);
    const initial = structuredClone(CONFIGURED_SETTINGS);
    initial.multi_gpu_enabled = true;
    render(
      <SettingsPage
        settings={initial}
        resourcesReady
        capabilities={{
          ...ONLINE_CAPABILITIES,
          execution_backends: {
            standard: { available: true, missing_nodes: [] },
            raylight: { available: true, missing_nodes: [] },
          },
        }}
        gpus={[
          { index: 0, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
          { index: 1, name: "A6000", vram_total: 48, vram_free: 40, visible: true },
        ]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={saveSettings}
      />,
    );

    const topology = screen.getByRole("region", { name: "FL2VA 扩散模型执行拓扑" });
    expect(within(topology).getByText(/STANDARD · default/)).toBeInTheDocument();
    expect(screen.queryByLabelText("FL2VA 扩散模型执行后端")).not.toBeInTheDocument();
    await user.click(screen.getByLabelText("FL2VA 扩散模型 RayLight 逻辑 GPU 1"));
    expect(within(topology).getByText(/RAYLIGHT · 2 卡/)).toBeInTheDocument();
    expect(screen.getByLabelText("FL2VA 扩散模型 RayLight Ulysses degree")).toHaveValue("2");
    await waitFor(() => expect(saveSettings).toHaveBeenCalledWith(expect.objectContaining({
      raylight_residency_policy: "keep_until_switch",
      models: expect.objectContaining({
        fl2va: expect.objectContaining({
          backend: "auto",
          raylight: expect.objectContaining({
            gpu_select: [0, 1],
            ulysses_degree: 2,
            ring_degree: 1,
            cfg_degree: 1,
            dp_degree: 1,
          }),
        }),
      }),
    })));
  });

  it("慢到的父级设置刷新不会覆盖等待自动同步的本地表单", async () => {
    const user = userEvent.setup();
    const onSaved = vi.fn(confirmConfiguredSettings);
    const { rerender } = render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        confirmedSettings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={onSaved}
      />,
    );
    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, "browser-edit");

    rerender(
      <SettingsPage
        settings={{ ...CONFIGURED_SETTINGS, client_id: "browser-edit" }}
        confirmedSettings={{ ...CONFIGURED_SETTINGS, client_id: "late-server-value" }}
        capabilities={EMPTY_CAPABILITIES}
        gpus={[]}
        models={EMPTY_MODELS}
        loadingModels={true}
        onSaved={onSaved}
      />,
    );
    expect(clientId).toHaveValue("browser-edit");
    await waitFor(() => expect(onSaved).toHaveBeenCalledWith(
      expect.objectContaining({ client_id: "browser-edit" }),
    ));
  });

  it("把每份有效编辑交给 App 队列且旧响应不会覆盖新的本地输入", async () => {
    const user = userEvent.setup();
    let resolveFirst!: (settings: RuntimeSettings) => void;
    const onSaved = vi.fn()
      .mockImplementationOnce(() => new Promise<RuntimeSettings>((resolve) => { resolveFirst = resolve; }))
      .mockImplementationOnce(async (value: RuntimeSettings) => value);
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={EMPTY_CAPABILITIES}
        gpus={[]}
        models={EMPTY_MODELS}
        loadingModels={false}
        onSaved={onSaved}
      />,
    );
    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, "first-version");
    await waitFor(() => expect(onSaved).toHaveBeenCalledWith(
      expect.objectContaining({ client_id: "first-version" }),
    ));

    await user.clear(clientId);
    await user.type(clientId, "newer-local-version");
    expect(onSaved).toHaveBeenLastCalledWith(
      expect.objectContaining({ client_id: "newer-local-version" }),
    );
    await act(async () => {
      resolveFirst({ ...CONFIGURED_SETTINGS, client_id: "first-version" });
    });

    expect(clientId).toHaveValue("newer-local-version");
    expect(screen.queryByText(/运行设置(?:已保存|未保存)/)).not.toBeInTheDocument();
  });

  it("自动同步完成前仍保留新客户端 ID，直到权威 GET 返回", async () => {
    const user = userEvent.setup();
    const nextSettings = {
      ...CONFIGURED_SETTINGS,
      client_id: "new-client",
    };
    vi.spyOn(directorApi, "updateSettings").mockResolvedValue(nextSettings);
    let resolveRefresh!: (settings: RuntimeSettings) => void;
    const onSaved = vi.fn(
      () => new Promise<RuntimeSettings>((resolve) => { resolveRefresh = resolve; }),
    );
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={onSaved}
      />,
    );

    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, nextSettings.client_id);
    await waitFor(() => expect(onSaved).toHaveBeenLastCalledWith(
      expect.objectContaining({ client_id: nextSettings.client_id }),
    ));
    expect(clientId).toHaveValue(nextSettings.client_id);
    expect(screen.queryByRole("button", { name: "保存设置" })).not.toBeInTheDocument();

    await act(async () => resolveRefresh(nextSettings));
    await waitFor(() => expect(clientId).toHaveValue(nextSettings.client_id));
  });

  it("权威 GET 失败时保留表单值并明确提示未确认", async () => {
    const user = userEvent.setup();
    const nextClientId = "unconfirmed-client";
    vi.spyOn(directorApi, "updateSettings").mockResolvedValue({
      ...CONFIGURED_SETTINGS,
      client_id: nextClientId,
    });
    const onSaved = vi.fn().mockRejectedValue(
      new Error("修改可能已写入，但无法从 Director 后端重新确认；请检查服务后重试"),
    );
    render(
      <SettingsPage
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={onSaved}
      />,
    );

    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, nextClientId);

    expect(await screen.findByText(/修改可能已写入，但无法从 Director 后端重新确认/)).toBeInTheDocument();
    expect(clientId).toHaveValue(nextClientId);
    expect(screen.queryByRole("button", { name: "保存设置" })).not.toBeInTheDocument();
  });

  it("浮层锁定背景滚动、约束焦点并由 Escape 请求关闭", async () => {
    const user = userEvent.setup();
    const onRequestClose = vi.fn();
    const { unmount } = render(
      <SettingsPage
        overlay
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
        onRequestClose={onRequestClose}
      />,
    );

    const dialog = screen.getByRole("dialog", { name: "系统设置" });
    const close = screen.getByRole("button", { name: "关闭系统设置" });
    expect(close).toHaveFocus();
    expect(document.body.style.overflow).toBe("hidden");
    await user.tab({ shift: true });
    expect(dialog).toContainElement(document.activeElement as HTMLElement);
    expect(document.activeElement).not.toHaveClass("settings-overlay__backdrop");
    await user.tab();
    expect(close).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(onRequestClose).toHaveBeenCalledTimes(1);
    unmount();
    expect(document.body.style.overflow).toBe("");
  });

  it("浮层可随时关闭，关闭动作会立即交接尚在防抖中的有效修改", async () => {
    const user = userEvent.setup();
    const onRequestClose = vi.fn();
    const onSaved = vi.fn(async (value: RuntimeSettings) => value);
    const { container } = render(
      <SettingsPage
        overlay
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={onSaved}
        onRequestClose={onRequestClose}
      />,
    );

    await user.type(screen.getByLabelText("客户端 ID"), "-dirty");
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    expect(onRequestClose).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(onSaved).toHaveBeenCalledWith(
      expect.objectContaining({ client_id: "directordeck-dirty" }),
    ));

    await user.click(container.querySelector(".settings-overlay__backdrop")!);
    expect(onRequestClose).toHaveBeenCalledTimes(2);
    expect(onSaved).toHaveBeenLastCalledWith(
      expect.objectContaining({ client_id: "directordeck-dirty" }),
    );
  });

  it("捕获阶段点外关闭，但设置入口和浮层内容不会触发反弹", () => {
    const onRequestClose = vi.fn();
    const toggle = document.createElement("button");
    toggle.id = "system-settings-toggle";
    const outside = document.createElement("button");
    outside.textContent = "侧栏其他操作";
    document.body.append(toggle, outside);
    const view = render(
      <SettingsPage
        overlay
        settings={CONFIGURED_SETTINGS}
        capabilities={ONLINE_CAPABILITIES}
        gpus={[]}
        models={MODEL_INVENTORY}
        loadingModels={false}
        onSaved={confirmConfiguredSettings}
        onRequestClose={onRequestClose}
      />,
    );

    fireEvent.pointerDown(screen.getByLabelText("客户端 ID"));
    fireEvent.pointerDown(toggle);
    expect(onRequestClose).not.toHaveBeenCalled();
    fireEvent.pointerDown(outside);
    expect(onRequestClose).toHaveBeenCalledOnce();
    expect(onRequestClose).toHaveBeenCalledWith(false);

    view.unmount();
    toggle.remove();
    outside.remove();
  });
});
