import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { directorApi } from "../api/client";
import {
  DEFAULT_SETTINGS,
  EMPTY_CAPABILITIES,
  EMPTY_MODELS,
  type DirectorDeckConfig,
  type FeatureCatalog,
  type FeatureCapabilityEvaluation,
  type GPUResource,
  type LoraLoaderOverrideRecord,
  type ModelInventory,
  type RayLightRuntimeStatus,
  type RuntimeSettings,
} from "../api/types";
import { SettingsPage, validateRuntimeSettingsForm } from "../components/SettingsPage";
import {
  applyLoraLoaderOverrideEdit,
  type LoraLoaderOverrideEdit,
} from "../state/loraLoaderOverrides";

const ONLINE_CAPABILITIES = { ...EMPTY_CAPABILITIES, connection: "online" as const };
const GPUS: GPUResource[] = [
  { index: 0, name: "GPU 0", vram_total: 24_000, vram_free: 20_000, visible: true },
  { index: 1, name: "GPU 1", vram_total: 24_000, vram_free: 18_000, visible: true },
];
const AVAILABLE_ADAPTER_CAPABILITY: FeatureCapabilityEvaluation = {
  available: true,
  reasons: [],
  verified_contracts: ["directordeck.node.test-lora"],
  runtime_fingerprints: [`sha256:${"c".repeat(64)}`],
};
const LORA_FEATURE_CATALOG: FeatureCatalog = {
  template_bundle_version: 5,
  host_capability_revision: `sha256:${"a".repeat(64)}`,
  entries: [{
    id: "lora",
    version: 1,
    title: "LoRA",
    description: "Apply the selected fail-closed LoRA adapter.",
    mode: "switch",
    layer: "graph",
    scopes: ["project"],
    params_schema: {},
    defaults: {},
    backends: ["standard", "raylight"],
    availability: { state: "conditional", reasons: [] },
    adapter_options: [
      {
        adapter_id: "model_only",
        display_name: "LoRA加载器（仅模型）",
        class_type: "LoraLoaderModelOnly",
        is_default: true,
        backend: "standard",
        supported_families: ["fl2va", "ref2va"],
        configuration_options: [],
        adapter_fingerprint: `sha256:${"1".repeat(64)}`,
        capability: AVAILABLE_ADAPTER_CAPABILITY,
      },
      {
        adapter_id: "minimax_h3_turbo",
        display_name: "MiniMax-H3 Turbo LoRA",
        class_type: "MiniMaxH3TurboLoRA",
        is_default: false,
        backend: "standard",
        supported_families: ["fl2va", "ref2va"],
        configuration_options: [{
          id: "low_vram",
          type: "boolean",
          label: "low_vram",
          description: "启用低显存模式",
          default: false,
        }],
        adapter_fingerprint: `sha256:${"2".repeat(64)}`,
        capability: AVAILABLE_ADAPTER_CAPABILITY,
      },
    ],
    ui: { visibility: "internal_v4" },
  }],
};
const DIRECTORDECK_CONFIG: DirectorDeckConfig = {
  schema_version: 1,
  lora: {
    loaders: LORA_FEATURE_CATALOG.entries[0].adapter_options.map((option) => ({
      id: option.adapter_id,
      display_name: option.display_name,
      class_type: option.class_type,
      input_contract: option.adapter_id === "minimax_h3_turbo"
        ? "dedicated_model"
        : "model_only",
      supported_families: option.supported_families,
      options: option.configuration_options,
    })),
    fallback_policy: {
      loader_ids: ["model_only"],
      default_loader_id: "model_only",
    },
    loader_policies: [{
      lora_filename: "minimax_h3_turbo_.*\\.safetensors$",
      loader_ids: ["minimax_h3_turbo"],
      default_loader_id: "minimax_h3_turbo",
    }],
  },
};

function runtimeSettings(): RuntimeSettings {
  return structuredClone(DEFAULT_SETTINGS);
}

function renderSettings(options: {
  settings?: RuntimeSettings;
  resourcesReady?: boolean;
  featureCatalog?: FeatureCatalog | null;
  gpus?: GPUResource[];
  models?: ModelInventory;
  runtimeEditingDisabled?: boolean;
  overlay?: boolean;
  onSaved?: (settings: RuntimeSettings) => Promise<RuntimeSettings>;
  onSaveLoraLoaderOverride?: (edit: LoraLoaderOverrideEdit) => Promise<RuntimeSettings>;
  onConnectionTestSucceeded?: () => void;
  onConfirmRayLightRuntimeRecovery?: () => Promise<void>;
  onRequestClose?: (restoreFocus?: boolean) => void;
  rayLightRuntimeStatus?: RayLightRuntimeStatus | null;
} = {}) {
  const settings = options.settings ?? runtimeSettings();
  return render(
    <SettingsPage
      settings={settings}
      confirmedSettings={settings}
      resourcesReady={options.resourcesReady ?? true}
      capabilities={ONLINE_CAPABILITIES}
      featureCatalog={options.featureCatalog === undefined
        ? LORA_FEATURE_CATALOG
        : options.featureCatalog}
      gpus={options.gpus ?? GPUS}
      models={options.models ?? EMPTY_MODELS}
      loadingModels={false}
      runtimeEditingDisabled={options.runtimeEditingDisabled}
      overlay={options.overlay}
      rayLightRuntimeStatus={options.rayLightRuntimeStatus}
      onSaved={options.onSaved ?? (async (next) => next)}
      onSaveLoraLoaderOverride={options.onSaveLoraLoaderOverride ?? (async (edit) =>
        applyLoraLoaderOverrideEdit(edit.base_settings, edit))}
      onConnectionTestSucceeded={options.onConnectionTestSucceeded}
      onConfirmRayLightRuntimeRecovery={options.onConfirmRayLightRuntimeRecovery}
      onRequestClose={options.onRequestClose}
    />,
  );
}

beforeEach(() => {
  vi.spyOn(directorApi, "getStorage").mockImplementation(() => new Promise(() => undefined));
  vi.spyOn(directorApi, "getDirectorDeckConfig").mockResolvedValue(DIRECTORDECK_CONFIG);
  vi.spyOn(directorApi, "getRayLightSetup").mockImplementation(() => new Promise(() => undefined));
  vi.spyOn(directorApi, "getMediaSetup").mockImplementation(() => new Promise(() => undefined));
});

afterEach(() => vi.restoreAllMocks());

describe("RuntimeSettingsV3 system settings", () => {
  it("validates runtime placement and stable LoRA-path mappings", () => {
    expect(validateRuntimeSettingsForm(runtimeSettings())).toEqual([]);

    const invalidDevice = runtimeSettings();
    invalidDevice.placement.video_vae_device = "cpu" as unknown as "default";
    expect(validateRuntimeSettingsForm(invalidDevice)).toContain("视频 VAE不允许放在 CPU");

    const invalidTopology = runtimeSettings();
    invalidTopology.placement.fl2va.raylight.gpu_select = [0, 1];
    invalidTopology.placement.fl2va.raylight.ulysses_degree = 1;
    invalidTopology.placement.fl2va.raylight.ring_degree = 1;
    expect(validateRuntimeSettingsForm(invalidTopology).join(" ")).toContain("Ulysses × Ring");

    const invalidMappings = runtimeSettings();
    invalidMappings.lora_loader_overrides = [
      {
        lora_filename: "z-ref-lora.safetensors",
        adapter_id: "model_only",
        options: {},
      },
      {
        lora_filename: "a-fl-lora.safetensors",
        adapter_id: "model_only",
        options: {},
      },
    ];
    expect(validateRuntimeSettingsForm(invalidMappings))
      .toContain("Standard LoRA 精确加载器映射无效");
  });

  it("does not expose model filenames or creative LoRA controls in system settings", async () => {
    renderSettings();
    expect(screen.getByText(/这里只保存 runtime placement/)).toBeInTheDocument();
    expect(screen.queryByLabelText(/LoRA 强度/)).not.toBeInTheDocument();
    expect(await screen.findByText(/为 LoRA 选择当前版本允许的加载器/))
      .toBeInTheDocument();
  });

  it("loads the fallback loader list independently of the host feature catalog", async () => {
    renderSettings({
      featureCatalog: null,
      models: { ...EMPTY_MODELS, loras: ["LoRA/H3/Style.Exact"] },
    });

    const loaderList = await screen.findByRole("listbox", { name: "加载器列表" });
    expect(await within(loaderList).findByRole("option", {
      name: /LoRA加载器（仅模型）/,
    })).toBeInTheDocument();
    expect(within(loaderList).queryByRole("option", {
      name: /MiniMax-H3 Turbo LoRA/,
    })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("LoRA 加载器映射迁移提示")).not.toBeInTheDocument();
  });

  it("applies the first matching filename regex policy and exposes the full path tooltip", async () => {
    const filename = "nested/minimax_h3_turbo_v4_step600_ema.safetensors";
    const styleFilename = "nested/style_detail.safetensors";
    renderSettings({
      featureCatalog: null,
      models: { ...EMPTY_MODELS, loras: [filename, styleFilename] },
    });

    const loaderList = await screen.findByRole("listbox", { name: "加载器列表" });
    expect(await within(loaderList).findByRole("option", {
      name: /MiniMax-H3 Turbo LoRA/,
    })).toHaveTextContent("默认");
    expect(within(loaderList).queryByRole("option", {
      name: /LoRA加载器（仅模型）/,
    })).not.toBeInTheDocument();
    const loraList = within(screen.getByRole("listbox", {
      name: "LoRA 列表",
    }));
    const loraOption = loraList.getByRole("option", { name: new RegExp(filename) });
    expect(loraOption).toHaveAttribute("title", filename);
    expect(within(loraOption).getByText(filename)).toHaveAttribute("title", filename);
    expect(loraOption).toHaveTextContent("MiniMax-H3 Turbo LoRA（默认）");
    const styleOption = loraList.getByRole("option", {
      name: new RegExp(styleFilename),
    });
    expect(styleOption).toHaveAttribute("title", styleFilename);
    expect(styleOption).toHaveTextContent("LoRA加载器（仅模型）（默认）");
  });

  it("keeps an incompatible historical mapping visible and recoverable", async () => {
    const filename = "nested/style_detail.safetensors";
    const settings = runtimeSettings();
    settings.lora_loader_overrides = [{
      lora_filename: filename,
      adapter_id: "minimax_h3_turbo",
      options: { low_vram: false },
    }];
    renderSettings({
      settings,
      featureCatalog: null,
      models: { ...EMPTY_MODELS, loras: [filename] },
    });

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "这条历史映射不在当前 LoRA 的允许清单中",
    );
    const loraOption = within(screen.getByRole("listbox", {
      name: "LoRA 列表",
    })).getByRole("option");
    expect(loraOption).toHaveTextContent("与当前规则不兼容");
    const loaderList = screen.getByRole("listbox", { name: "加载器列表" });
    expect(within(loaderList).getByRole("option", {
      name: /LoRA加载器（仅模型）/,
    })).toBeInTheDocument();
    expect(within(loaderList).queryByRole("option", {
      name: /MiniMax-H3 Turbo LoRA/,
    })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "保存映射" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "恢复系统默认" })).toBeEnabled();
  });

  it("selects LoRA and loader in separate lists, edits low_vram, and restores default", async () => {
    const user = userEvent.setup();
    const onMapping = vi.fn(async (edit: LoraLoaderOverrideEdit) =>
      applyLoraLoaderOverrideEdit(edit.base_settings, edit));
    const models: ModelInventory = {
      ...EMPTY_MODELS,
      loras: ["LoRA/H3/minimax_h3_turbo_Turbo.Exact.safetensors"],
    };
    const view = renderSettings({ models, onSaveLoraLoaderOverride: onMapping });

    await user.click(within(screen.getByRole("listbox", {
      name: "加载器列表",
    })).getByRole("option", { name: /MiniMax-H3 Turbo LoRA/ }));
    await user.click(screen.getByLabelText("LoRA 加载器配置 low_vram"));
    await user.click(screen.getByRole("button", { name: "保存映射" }));

    await waitFor(() => expect(onMapping).toHaveBeenCalledTimes(1));
    const created = onMapping.mock.calls[0][0].next as LoraLoaderOverrideRecord;
    expect(created).toEqual({
      lora_filename: "LoRA/H3/minimax_h3_turbo_Turbo.Exact.safetensors",
      adapter_id: "minimax_h3_turbo",
      options: { low_vram: true },
    });
    const saved = applyLoraLoaderOverrideEdit(runtimeSettings(), onMapping.mock.calls[0][0]);
    view.rerender(<SettingsPage
      settings={saved}
      confirmedSettings={saved}
      resourcesReady
      capabilities={ONLINE_CAPABILITIES}
      featureCatalog={LORA_FEATURE_CATALOG}
      gpus={GPUS}
      models={models}
      loadingModels={false}
      onSaved={async (next) => next}
      onSaveLoraLoaderOverride={onMapping}
    />);
    await user.click(screen.getByRole("button", { name: "恢复系统默认" }));
    await waitFor(() => expect(onMapping).toHaveBeenCalledTimes(2));
    expect(onMapping.mock.calls[1][0]).toMatchObject({
      original: created,
      next: null,
    });
  });

  it("keeps the loader and option draft visible when the same LoRA conflicts", async () => {
    const user = userEvent.setup();
    const onMapping = vi.fn().mockRejectedValue(
      new Error("同一条 LoRA 加载器映射已被其他页面修改；请重新检查后再保存"),
    );
    renderSettings({
      models: {
        ...EMPTY_MODELS,
        loras: ["LoRA/minimax_h3_turbo_Same.safetensors"],
      },
      onSaveLoraLoaderOverride: onMapping,
    });
    const loaderList = screen.getByRole("listbox", { name: "加载器列表" });
    await user.click(within(loaderList).getByRole("option", {
      name: /MiniMax-H3 Turbo LoRA/,
    }));
    await user.click(screen.getByLabelText("LoRA 加载器配置 low_vram"));
    await user.click(screen.getByRole("button", { name: "保存映射" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("同一条 LoRA 加载器映射已被其他页面修改");
    expect(within(loaderList).getByRole("option", {
      name: /MiniMax-H3 Turbo LoRA/,
    })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("LoRA 加载器配置 low_vram")).toBeChecked();
  });

  it("shows node observation without treating fingerprint as execution authority", async () => {
    const user = userEvent.setup();
    const featureCatalog = structuredClone(LORA_FEATURE_CATALOG);
    const option = featureCatalog.entries[0].adapter_options.find((item) =>
      item.adapter_id === "minimax_h3_turbo");
    expect(option).toBeDefined();
    if (!option) return;
    option.capability = {
      available: false,
      reasons: [{
        code: "node_unavailable",
        feature_id: "lora",
        segment_id: null,
        unit_id: null,
        backend: "standard",
        rule: "host_node_registry",
        message: "A required ComfyUI node is unavailable.",
        remediation: "Install the supported node implementation and restart ComfyUI.",
        safe_details: { class_type: "LoraLoaderModelOnly" },
      }],
      verified_contracts: [],
      runtime_fingerprints: [],
    };
    const onMapping = vi.fn(async (edit: LoraLoaderOverrideEdit) =>
      applyLoraLoaderOverrideEdit(edit.base_settings, edit));
    renderSettings({
      featureCatalog,
      models: {
        ...EMPTY_MODELS,
        loras: ["LoRA/H3/minimax_h3_turbo_Style.Exact.safetensors"],
      },
      onSaveLoraLoaderOverride: onMapping,
    });

    await user.click(within(screen.getByRole("listbox", {
      name: "加载器列表",
    })).getByRole("option", { name: /MiniMax-H3 Turbo LoRA/ }));

    const evidence = screen.getByRole("status", { name: "LoRA 映射节点检测" });
    expect(evidence).toHaveTextContent("当前未检测到节点");
    expect(evidence).toHaveTextContent("仍可保存配置");
    expect(evidence).not.toHaveTextContent(option.adapter_fingerprint);
    expect(evidence).not.toHaveTextContent("node_unavailable");
    expect(screen.getByRole("button", { name: "保存映射" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "保存映射" }));
    await waitFor(() => expect(onMapping).toHaveBeenCalledTimes(1));
    expect(onMapping.mock.calls[0][0].next).toMatchObject({
      adapter_id: "minimax_h3_turbo",
    });
  });

  it("writes a complete v3 runtime document when placement changes", async () => {
    const user = userEvent.setup();
    const onSaved = vi.fn(async (settings: RuntimeSettings) => settings);
    renderSettings({ onSaved });

    await user.selectOptions(screen.getByLabelText("FL2VA 扩散模型设备"), "gpu:1");
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1));
    const saved = onSaved.mock.calls[0][0];
    expect(saved).toEqual({
      ...DEFAULT_SETTINGS,
      placement: {
        ...DEFAULT_SETTINGS.placement,
        fl2va: {
          ...DEFAULT_SETTINGS.placement.fl2va,
          device: "gpu:1",
        },
      },
    });
    expect(saved).not.toHaveProperty("models");
  });

  it("enables a two-card RayLight pool atomically and keeps a valid topology", async () => {
    const user = userEvent.setup();
    const onSaved = vi.fn(async (settings: RuntimeSettings) => settings);
    renderSettings({ onSaved });

    await user.click(screen.getByRole("checkbox", { name: "启用多卡推理" }));
    await user.click(screen.getByRole("checkbox", {
      name: "FL2VA 扩散模型 RayLight 逻辑 GPU 1",
    }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(2));
    const saved = onSaved.mock.calls[1][0];
    expect(saved.multi_gpu_enabled).toBe(true);
    expect(saved.placement.fl2va).toMatchObject({
      device: "default",
      raylight: {
        gpu_select: [0, 1],
        ulysses_degree: 2,
        ring_degree: 1,
        cfg_degree: 1,
        dp_degree: 1,
      },
    });
  });

  it("disabling multi-GPU collapses both families to one-card Standard topology", async () => {
    const user = userEvent.setup();
    const settings = runtimeSettings();
    settings.multi_gpu_enabled = true;
    for (const family of ["fl2va", "ref2va"] as const) {
      settings.placement[family].raylight.gpu_select = [0, 1];
      settings.placement[family].raylight.ulysses_degree = 2;
      settings.placement[family].raylight.ring_degree = 1;
    }
    const onSaved = vi.fn(async (next: RuntimeSettings) => next);
    renderSettings({ settings, onSaved });

    await user.click(screen.getByRole("checkbox", { name: "启用多卡推理" }));
    await waitFor(() => expect(onSaved).toHaveBeenCalledTimes(1));
    const saved = onSaved.mock.calls[0][0];
    expect(saved.multi_gpu_enabled).toBe(false);
    expect(saved.placement.fl2va.raylight).toMatchObject({
      gpu_select: [0], ulysses_degree: 1, ring_degree: 1,
    });
    expect(saved.placement.ref2va.raylight).toMatchObject({
      gpu_select: [0], ulysses_degree: 1, ring_degree: 1,
    });
  });

  it("locks every runtime mutation while stale-schema recovery owns authority", () => {
    const onSaved = vi.fn(async (settings: RuntimeSettings) => settings);
    renderSettings({ runtimeEditingDisabled: true, onSaved });
    expect(screen.getByText("运行设置暂时锁定。")).toBeInTheDocument();
    expect(screen.getByLabelText("客户端 ID")).toBeDisabled();
    expect(screen.getByLabelText("FL2VA 扩散模型设备")).toBeDisabled();
    fireEvent.change(screen.getByLabelText("客户端 ID"), { target: { value: "blocked" } });
    expect(onSaved).not.toHaveBeenCalled();
  });

  it("keeps the connection card read-only and refreshes host resources only after a successful probe", async () => {
    const user = userEvent.setup();
    const onSucceeded = vi.fn();
    vi.spyOn(directorApi, "testConnection").mockResolvedValue({
      ok: true,
      message: "connected",
      latency_ms: 12,
    });
    renderSettings({ onConnectionTestSucceeded: onSucceeded });

    expect(screen.queryByLabelText("ComfyUI 地址")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "测试连接" }));
    await waitFor(() => expect(onSucceeded).toHaveBeenCalledTimes(1));
    expect(screen.getByText("当前实例可连接")).toBeInTheDocument();
    expect(screen.getByText(/响应 12 ms/)).toBeInTheDocument();
  });

  it("keeps the database path hidden until the user explicitly reveals it", async () => {
    const user = userEvent.setup();
    vi.mocked(directorApi.getStorage).mockResolvedValue({
      active_database_path: "/srv/private/directordeck.sqlite3",
    });
    renderSettings();

    const storage = await screen.findByLabelText("当前数据库存储配置");
    expect(storage).not.toHaveTextContent("/srv/private/directordeck.sqlite3");
    const toggle = screen.getByRole("button", { name: "当前数据库路径显示状态" });
    await user.click(toggle);
    expect(storage).toHaveTextContent("/srv/private/directordeck.sqlite3");
  });

  it("requires an explicit restart confirmation before clearing stale RayLight evidence", async () => {
    const user = userEvent.setup();
    const onRecover = vi.fn(async () => undefined);
    const confirm = vi.spyOn(window, "confirm")
      .mockReturnValueOnce(false)
      .mockReturnValueOnce(true);
    renderSettings({
      onConfirmRayLightRuntimeRecovery: onRecover,
      rayLightRuntimeStatus: {
        active: true,
        recovery_required: true,
        epoch: 36,
        runtime_gpu_indexes: [0, 1],
        available_gpu_indexes: [0],
        invalid_gpu_indexes: [1],
        tainted: false,
        recovery_token: "a".repeat(64),
      },
    });
    const alert = screen.getByRole("alert", {
      name: "旧 RayLight 运行状态引用了当前不可见 GPU",
    });
    expect(alert).toHaveTextContent("当前不可见逻辑 GPU：1");
    const recover = within(alert).getByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    });
    await user.click(recover);
    expect(onRecover).not.toHaveBeenCalled();
    await user.click(recover);
    await waitFor(() => expect(onRecover).toHaveBeenCalledTimes(1));
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining("测试连接成功或当前队列为空"));
  });

  it("overlay delegates Escape close with focus restoration", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderSettings({ overlay: true, onRequestClose: onClose });
    expect(screen.getByRole("dialog", { name: "系统设置" })).toHaveAttribute("aria-modal", "true");
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledWith(true);
  });
});
