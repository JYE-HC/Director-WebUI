import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import App, {
  QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY,
  QUARANTINED_UNBOUND_RUNTIME_SETTINGS_PENDING_KEY,
  RUNTIME_SETTINGS_PENDING_KEY,
  UNBOUND_RUNTIME_SETTINGS_PENDING_KEY,
} from "../App";
import { ApiError, DATABASE_IDENTITY_STALE_EVENT, directorApi } from "../api/client";
import {
  DEFAULT_SETTINGS,
  EMPTY_CAPABILITIES,
  EMPTY_RAYLIGHT_RUNTIME_STATUS,
  type CapabilityReport,
  type GenerationTask,
  type TimelineCompileReport,
} from "../api/types";
import { MODE_ORDER, type AssetReference } from "../domain/modes";
import {
  createTimelineProject,
  createTimelineSegment,
  DEFAULT_PROJECT_ID,
  LEGACY_TIMELINE_STORAGE_KEY,
  QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY,
  QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY,
  QUARANTINED_TIMELINE_STORAGE_KEY,
  saveLocalTimeline as saveLocalTimelineForDatabase,
  TIMELINE_WAL_STORAGE_KEY,
  UNBOUND_TIMELINE_WAL_STORAGE_KEY,
} from "../domain/timelineProject";
import {
  loadTimelineSegmentSelectionPreference,
  saveTimelineSegmentSelectionPreference,
} from "../domain/workspacePreferences";

const ACTIVE_DATABASE_PATH = "/srv/director/data/director.sqlite3";
const ACTIVE_DATABASE_IDENTITY = "a".repeat(64);
const ACTIVE_DATABASE = {
  active_database_path: ACTIVE_DATABASE_PATH,
  active_database_identity: ACTIVE_DATABASE_IDENTITY,
};
const ACTIVE_STORAGE_CONFIGURATION = {
  ...ACTIVE_DATABASE,
  configured_database_path: ACTIVE_DATABASE_PATH,
  recommended_database_path: "/srv/director/.data/database/director.sqlite3",
  source: "default" as const,
  restart_required: false,
};
const saveLocalTimeline = (project: Parameters<typeof saveLocalTimelineForDatabase>[0]) =>
  saveLocalTimelineForDatabase(project, ACTIVE_DATABASE);
const saveRuntimeSettingsWal = (
  settings: typeof CONFIGURED_SETTINGS,
  database = ACTIVE_DATABASE,
) => localStorage.setItem(RUNTIME_SETTINGS_PENDING_KEY, JSON.stringify({
  format: "director-pending-runtime-settings",
  version: 1,
  pending: true,
  active_database_path: database.active_database_path,
  active_database_identity: database.active_database_identity,
  written_at_ms: Date.now(),
  settings,
}));

const imageAsset: AssetReference = {
  id: "asset-image-app",
  name: "角色参考.png",
  subfolder: "director",
  type: "input",
  kind: "image",
  preview_url: "/api/assets/asset-image-app/preview",
};

const endpointBImageAsset: AssetReference = {
  ...imageAsset,
  id: "asset-image-endpoint-b",
  name: "B 服务器素材.png",
  preview_url: "/api/assets/asset-image-endpoint-b/preview",
};

const CONFIGURED_SETTINGS = {
  ...DEFAULT_SETTINGS,
  comfy_url: "http://comfy.test:8188",
};

function runtimeAuthority(settings: typeof CONFIGURED_SETTINGS) {
  const document = JSON.stringify(settings);
  let hash = 0x811c9dc5;
  for (let index = 0; index < document.length; index += 1) {
    hash ^= document.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  const block = (hash >>> 0).toString(16).padStart(8, "0");
  return {
    settings,
    authority_token: block.repeat(8),
  };
}

const ONLINE_CAPABILITIES: CapabilityReport = {
  ...EMPTY_CAPABILITIES,
  connection: "online",
  supported_modes: [...MODE_ORDER],
  execution_backends: {
    standard: { available: true, missing_nodes: [] },
    raylight: { available: true, missing_nodes: [] },
  },
};

const ORIGINAL_VIEWPORT_WIDTH = window.innerWidth;

const MODEL_INVENTORY = {
  fl2va: [DEFAULT_SETTINGS.models.fl2va.filename, "alternate-diffusion.safetensors"],
  ref2va: [DEFAULT_SETTINGS.models.ref2va.filename, "alternate-diffusion.safetensors"],
  clip: [DEFAULT_SETTINGS.models.clip.filename],
  video_vae: [DEFAULT_SETTINGS.models.video_vae.filename],
  audio_vae: [DEFAULT_SETTINGS.models.audio_vae.filename],
  loras: ["minimax-h3-turbo.safetensors"],
};

const GPU_ZERO = {
  index: 0,
  name: "A6000-0",
  vram_total: 48 * 1024 ** 3,
  vram_free: 40 * 1024 ** 3,
  visible: true,
};

const GPU_ONE = {
  index: 1,
  name: "A6000-1",
  vram_total: 48 * 1024 ** 3,
  vram_free: 39 * 1024 ** 3,
  visible: true,
};

const queuedTimelineTask: GenerationTask = {
  id: "job-timeline-submit",
  mode: "timeline",
  status: "queued",
  progress: 0,
  stage: "queued",
  prompt_id: "prompt-timeline-submit",
  outputs: [],
  error: null,
  preview_url: null,
  created_at: "2026-08-12T00:00:00Z",
  updated_at: "2026-08-12T00:00:00Z",
  started_at: null,
  completed_at: null,
  children: [],
  segment_results: [],
  live_preview_url: null,
};

const succeededTask: GenerationTask = {
  ...queuedTimelineTask,
  id: "job-terminal-1234",
  status: "succeeded",
  progress: 1,
  stage: "completed",
  outputs: ["/api/jobs/job-terminal-1234/outputs/0"],
  output_files: ["output/video/final.mp4"],
  completed_at: "2026-08-12T00:01:00Z",
};

function mockCommonRequests(settings = CONFIGURED_SETTINGS) {
  vi.spyOn(directorApi, "getSettings").mockResolvedValue(settings);
  vi.spyOn(directorApi, "getSettingsAuthority").mockResolvedValue(runtimeAuthority(settings));
  vi.spyOn(directorApi, "getStorage").mockResolvedValue(ACTIVE_STORAGE_CONFIGURATION);
  vi.spyOn(directorApi, "getCapabilities").mockResolvedValue(ONLINE_CAPABILITIES);
  vi.spyOn(directorApi, "getGpus").mockResolvedValue([]);
  vi.spyOn(directorApi, "getRayLightRuntimeStatus").mockResolvedValue(
    EMPTY_RAYLIGHT_RUNTIME_STATUS,
  );
  vi.spyOn(directorApi, "getModels").mockResolvedValue(MODEL_INVENTORY);
  vi.spyOn(directorApi, "listTasks").mockResolvedValue({ jobs: [] });
  vi.spyOn(directorApi, "listAssets").mockResolvedValue({ assets: [], outputs_preserved: true });
  vi.spyOn(directorApi, "getTimeline").mockResolvedValue(createTimelineProject());
  vi.spyOn(directorApi, "updateTimeline").mockImplementation(async (project) => project);
  vi.spyOn(directorApi, "listProjects").mockResolvedValue({
    projects: [{
      id: "default",
      title: "未命名长视频",
      created_at: "2026-08-12T00:00:00Z",
      updated_at: "2026-08-12T00:00:00Z",
      segment_count: 1,
    }],
  });
}

async function waitUntilReady() {
  expect(await screen.findByText("ComfyUI 已连接")).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByText(/正在从服务器恢复时间线|暂时无法确认数据库或读取服务器时间线/)).not.toBeInTheDocument());
}

async function waitUntilTimelineReady() {
  await waitFor(() => expect(directorApi.getTimeline).toHaveBeenCalled());
  await waitFor(() => expect(screen.queryByText("正在从服务器恢复时间线")).not.toBeInTheDocument());
}

async function openGlobalSettings(user: ReturnType<typeof userEvent.setup>) {
  const toggle = screen.getByRole("button", { name: "全局设置" });
  await user.click(toggle);
  expect(toggle).toHaveAttribute("aria-expanded", "true");
}

beforeEach(() => {
  localStorage.clear();
  directorApi.resetDatabaseIdentityForTests();
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  Object.defineProperty(window, "innerWidth", { configurable: true, value: ORIGINAL_VIEWPORT_WIDTH });
  delete document.documentElement.dataset.theme;
});

describe("统一长视频时间线应用", () => {
  it("以素材库和统一 timeline 替代六模式导航", async () => {
    mockCommonRequests();
    render(<App />);
    await waitUntilReady();

    expect(screen.getByRole("complementary", { name: "当前工作区素材库" })).toBeInTheDocument();
    expect(screen.getByRole("main", { name: "长视频时间线工作区" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "主时间线" })).toBeInTheDocument();
    expect(screen.getByText("时间线主预览")).toBeInTheDocument();
    const liveToggle = screen.getByRole("button", { name: "实时执行" });
    expect(liveToggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(liveToggle);
    expect(liveToggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("实时执行进度")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "片段生成模式" })).toHaveValue("fl2va");
    expect(screen.queryByRole("navigation", { name: "生成模式" })).not.toBeInTheDocument();
    expect(document.getElementById("timeline-global-settings")).toHaveAttribute("hidden");
    expect(screen.queryByLabelText("时间线设置预览")).not.toBeInTheDocument();
    expect(screen.queryByText("LONG-FORM WORKSPACE")).not.toBeInTheDocument();
    expect(screen.queryByText("LONG-FORM STUDIO")).not.toBeInTheDocument();
    expect(screen.queryByText("WORKSPACE ASSETS")).not.toBeInTheDocument();
    expect(screen.queryByText("PROGRAM MONITOR")).not.toBeInTheDocument();
    expect(screen.queryByText("LIVE PROGRESS")).not.toBeInTheDocument();
    expect(screen.queryByText("MASTER TIMELINE")).not.toBeInTheDocument();
    expect(screen.queryByText("TL")).not.toBeInTheDocument();
    const projectSwitcher = screen.getByRole("combobox", { name: "切换项目" });
    expect(projectSwitcher).toHaveClass("topbar__project-switcher");
    expect(projectSwitcher.closest(".topbar__identity")).not.toBeNull();
    projectSwitcher.focus();
    expect(projectSwitcher).toHaveFocus();
    expect(screen.getByRole("button", { name: /重命名项目，当前名称：/ }).querySelector(".topbar__project-title-icon svg")).not.toBeNull();
    expect(screen.getByRole("button", { name: "全局设置" })).toHaveTextContent("全局设置");
    expect(screen.getByRole("button", { name: "全局设置" })).not.toHaveTextContent("展开");
    expect(screen.getByRole("button", { name: "任务，0 个进行中" })).toHaveTextContent("任务0");
    for (const button of screen.getAllByRole("button").filter((item) => item.hasAttribute("aria-expanded"))) {
      expect(`${button.getAttribute("aria-label") ?? ""} ${button.textContent ?? ""}`).not.toMatch(/展开|收起|显示|隐藏/);
    }
  });

  it("首次时间线 GET 失败时锁定默认项目并自动重试，成功后才加载服务器项目", async () => {
    mockCommonRequests();
    const serverProject = createTimelineProject();
    serverProject.title = "服务器恢复项目";
    serverProject.segments[0].prompt = "服务器权威提示词";
    vi.mocked(directorApi.getTimeline)
      .mockRejectedValueOnce(new Error("后端尚未就绪"))
      .mockResolvedValue(serverProject);
    const update = vi.mocked(directorApi.updateTimeline);

    render(<App />);

    expect(await screen.findByText(/暂时无法确认数据库或读取服务器时间线/)).toBeInTheDocument();
    const prompt = screen.getByLabelText("片段提示词");
    fireEvent.change(prompt, { target: { value: "绝不能反写的默认项目" } });
    expect(prompt).not.toHaveValue("绝不能反写的默认项目");
    expect(update).not.toHaveBeenCalled();

    expect((await screen.findAllByText("服务器恢复项目", {}, { timeout: 2_000 })).length).toBeGreaterThan(0);
    expect(screen.getByLabelText("片段提示词")).toHaveValue("服务器权威提示词");
    expect(directorApi.getTimeline).toHaveBeenCalledTimes(2);
    expect(update).not.toHaveBeenCalled();
  });

  it("旧 v2 长期镜像只隔离保留，不参与启动恢复或反写服务器", async () => {
    const staleMirror = createTimelineProject();
    staleMirror.title = "旧浏览器镜像";
    staleMirror.segments[0].prompt = "不能回放";
    const rawMirror = JSON.stringify(staleMirror);
    localStorage.setItem(LEGACY_TIMELINE_STORAGE_KEY, rawMirror);
    mockCommonRequests();
    const serverProject = createTimelineProject();
    serverProject.title = "当前服务器项目";
    serverProject.segments[0].prompt = "服务器内容";
    vi.mocked(directorApi.getTimeline).mockResolvedValue(serverProject);
    const update = vi.mocked(directorApi.updateTimeline);

    render(<App />);

    expect((await screen.findAllByText("当前服务器项目")).length).toBeGreaterThan(0);
    expect(screen.queryByText("旧浏览器镜像")).not.toBeInTheDocument();
    expect(screen.getByLabelText("片段提示词")).toHaveValue("服务器内容");
    expect(localStorage.getItem(LEGACY_TIMELINE_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_TIMELINE_STORAGE_KEY)).toBe(rawMirror);
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(update).not.toHaveBeenCalled();
  });

  it("先确认数据库身份再恢复精确匹配的时间线 WAL，确认前默认项目保持只读", async () => {
    const pending = createTimelineProject();
    pending.title = "数据库 A 的待同步项目";
    pending.segments[0].prompt = "只属于数据库 A";
    saveLocalTimeline(pending);
    mockCommonRequests();
    let resolveStorage!: (value: typeof ACTIVE_STORAGE_CONFIGURATION) => void;
    vi.mocked(directorApi.getStorage)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveStorage = resolve; }))
      .mockResolvedValue(ACTIVE_STORAGE_CONFIGURATION);
    const update = vi.mocked(directorApi.updateTimeline);

    render(<App />);
    expect((await screen.findAllByText(/正在从服务器恢复时间线/)).length).toBeGreaterThan(0);
    expect(screen.queryByText(pending.title)).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "身份未确认时的编辑" } });
    expect(update).not.toHaveBeenCalled();
    expect(directorApi.getTimeline).not.toHaveBeenCalled();

    await act(async () => resolveStorage(ACTIVE_STORAGE_CONFIGURATION));
    expect((await screen.findAllByText(pending.title)).length).toBeGreaterThan(0);
    expect(screen.getByLabelText("片段提示词")).toHaveValue("只属于数据库 A");
    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({ title: pending.title })));
    expect(directorApi.getTimeline).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
  });

  it("storage A 后读取到 timeline B 时二次身份核对失败并保持整页只读", async () => {
    const databaseBPath = "/srv/director/data/database-b.sqlite3";
    const databaseB = {
      active_database_path: databaseBPath,
      active_database_identity: "b".repeat(64),
      configured_database_path: databaseBPath,
      recommended_database_path: "/srv/director/.data/database/director.sqlite3",
      source: "explicit" as const,
      restart_required: false,
    };
    const databaseBProject = createTimelineProject();
    databaseBProject.title = "不应被 A 页面接受的 B 项目";
    databaseBProject.segments[0].prompt = "来自数据库 B";
    mockCommonRequests();
    vi.mocked(directorApi.getStorage)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValue(databaseB);
    vi.mocked(directorApi.getTimeline).mockResolvedValue(databaseBProject);
    const update = vi.mocked(directorApi.updateTimeline);
    const user = userEvent.setup();

    render(<App />);

    expect((await screen.findAllByText(/Director 后端数据库已变化/)).length).toBeGreaterThan(0);
    expect(screen.queryByText(databaseBProject.title)).not.toBeInTheDocument();
    const prompt = screen.getByLabelText("片段提示词");
    const before = (prompt as HTMLTextAreaElement).value;
    fireEvent.change(prompt, { target: { value: "竞态后不得编辑" } });
    expect(prompt).toHaveValue(before);
    expect(update).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    expect(await screen.findByLabelText("数据库目标路径")).toBeDisabled();
    expect(screen.getByRole("button", { name: "保存路径（重启后切换）" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "迁移当前数据库并切换" })).toBeDisabled();
  });

  it("精确 stale database identity 事件会锁定已加载页面且不再生成 WAL", async () => {
    mockCommonRequests();
    const update = vi.mocked(directorApi.updateTimeline);
    render(<App />);
    await waitUntilReady();
    update.mockClear();

    act(() => window.dispatchEvent(new Event(DATABASE_IDENTITY_STALE_EVENT)));

    expect((await screen.findAllByText(/本页已停止修改.*刷新整个页面/)).length).toBeGreaterThan(0);
    const prompt = screen.getByLabelText("片段提示词");
    const before = (prompt as HTMLTextAreaElement).value;
    fireEvent.change(prompt, { target: { value: "stale 页面不得继续修改" } });
    expect(prompt).toHaveValue(before);
    expect(update).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
  });

  it("跨库 v4 WAL 与无身份 v3 WAL 都只隔离，服务器项目保持权威", async () => {
    const stale = createTimelineProject();
    stale.title = "其他数据库的待同步项目";
    saveLocalTimelineForDatabase(stale, {
      active_database_path: "/srv/director/data/other.sqlite3",
      active_database_identity: "b".repeat(64),
    });
    const mismatchedRaw = localStorage.getItem(TIMELINE_WAL_STORAGE_KEY);
    const unboundRaw = JSON.stringify({
      format: "director-pending-timeline",
      version: 1,
      pending: true,
      written_at_ms: Date.now(),
      project: stale,
    });
    localStorage.setItem(UNBOUND_TIMELINE_WAL_STORAGE_KEY, unboundRaw);
    mockCommonRequests();
    const server = createTimelineProject();
    server.title = "当前数据库服务器项目";
    vi.mocked(directorApi.getTimeline).mockResolvedValue(server);
    const update = vi.mocked(directorApi.updateTimeline);

    render(<App />);
    expect((await screen.findAllByText(server.title)).length).toBeGreaterThan(0);
    expect(screen.queryByText(stale.title)).not.toBeInTheDocument();
    expect(update).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY)).toBe(mismatchedRaw);
    expect(localStorage.getItem(UNBOUND_TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY)).toBe(unboundRaw);
  });

  it("跨库与旧无身份 RuntimeSettings WAL 均隔离且不触发 PUT", async () => {
    const staleSettings = { ...CONFIGURED_SETTINGS, client_id: "stale-other-database" };
    saveRuntimeSettingsWal(staleSettings, {
      active_database_path: "/srv/director/data/other.sqlite3",
      active_database_identity: "b".repeat(64),
    });
    const mismatchedRaw = localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY);
    const unboundRaw = JSON.stringify({ ...staleSettings, client_id: "old-unbound" });
    localStorage.setItem(UNBOUND_RUNTIME_SETTINGS_PENDING_KEY, unboundRaw);
    mockCommonRequests();
    const updateSettings = vi.spyOn(directorApi, "updateSettings");

    render(<App />);
    await waitUntilReady();

    expect(updateSettings).not.toHaveBeenCalled();
    expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY)).toBe(mismatchedRaw);
    expect(localStorage.getItem(UNBOUND_RUNTIME_SETTINGS_PENDING_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_UNBOUND_RUNTIME_SETTINGS_PENDING_KEY)).toBe(unboundRaw);
  });

  it("RuntimeSettings 隔离槽已有旧 WAL 时仍保留第二份跨库 WAL", async () => {
    const firstQuarantine = "first-runtime-settings-wal";
    localStorage.setItem(QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY, firstQuarantine);
    const staleSettings = { ...CONFIGURED_SETTINGS, client_id: "second-database-wal" };
    saveRuntimeSettingsWal(staleSettings, {
      active_database_path: "/srv/director/data/other.sqlite3",
      active_database_identity: "b".repeat(64),
    });
    const raw = localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY);
    mockCommonRequests();
    const updateSettings = vi.spyOn(directorApi, "updateSettings");

    render(<App />);
    await waitUntilReady();

    expect(updateSettings).not.toHaveBeenCalled();
    expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).toBeNull();
    expect(localStorage.getItem(QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY)).toBe(firstQuarantine);
    expect(localStorage.getItem(`${QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY}:1`)).toBe(raw);
  });

  it("旧数据库页面写 RuntimeSettings 前保全 B 页 WAL，迟到清理也不删除 B 的后续 WAL", async () => {
    const user = userEvent.setup();
    const desired = { ...CONFIGURED_SETTINGS, client_id: "database-a-current-tab" };
    let resolveUpdate!: (settings: typeof CONFIGURED_SETTINGS) => void;
    mockCommonRequests();
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(desired));
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockImplementation(
      () => new Promise((resolve) => { resolveUpdate = resolve; }),
    );

    render(<App />);
    await waitUntilReady();
    const databaseBSettings = { ...CONFIGURED_SETTINGS, client_id: "database-b-pending" };
    const databaseBRaw = JSON.stringify({
      format: "director-pending-runtime-settings",
      version: 2,
      owner_id: "database-b-tab",
      pending: true,
      active_database_path: "/srv/director/data/database-b.sqlite3",
      active_database_identity: "b".repeat(64),
      written_at_ms: Date.now(),
      settings: databaseBSettings,
    });
    localStorage.setItem(RUNTIME_SETTINGS_PENDING_KEY, databaseBRaw);

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    fireEvent.change(screen.getByLabelText("客户端 ID"), {
      target: { value: desired.client_id },
    });

    expect(localStorage.getItem(QUARANTINED_MISMATCHED_RUNTIME_SETTINGS_PENDING_KEY)).toBe(databaseBRaw);
    expect(JSON.parse(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)!)).toMatchObject({
      version: 2,
      owner_id: expect.any(String),
      active_database_identity: ACTIVE_DATABASE_IDENTITY,
      settings: { client_id: desired.client_id },
    });
    await waitFor(() => expect(updateSettings).toHaveBeenCalledWith(desired));
    const laterDatabaseBRaw = JSON.stringify({
      ...JSON.parse(databaseBRaw),
      written_at_ms: Date.now() + 1,
      settings: { ...databaseBSettings, client_id: "database-b-later" },
    });
    localStorage.setItem(RUNTIME_SETTINGS_PENDING_KEY, laterDatabaseBRaw);
    await act(async () => resolveUpdate(desired));

    await waitFor(() => expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).toBe(laterDatabaseBRaw));
  });

  it("同库 RuntimeSettings WAL 在 storage identity 确认后才自动同步并清除", async () => {
    const pendingSettings = { ...CONFIGURED_SETTINGS, client_id: "same-database-pending" };
    saveRuntimeSettingsWal(pendingSettings);
    mockCommonRequests();
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(pendingSettings));
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(pendingSettings);

    render(<App />);
    await waitFor(() => expect(updateSettings).toHaveBeenCalledWith(pendingSettings));
    await waitFor(() => expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).toBeNull());
  });

  it("Settings 自己先读到 storage 时，App hydration 未完成仍禁用切库且不清 WAL", async () => {
    const pending = createTimelineProject();
    pending.title = "尚未检查身份的 WAL";
    saveLocalTimeline(pending);
    mockCommonRequests();
    let resolveAppStorage!: (value: typeof ACTIVE_STORAGE_CONFIGURATION) => void;
    vi.mocked(directorApi.getStorage)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveAppStorage = resolve; }))
      .mockResolvedValue(ACTIVE_STORAGE_CONFIGURATION);
    const migrate = vi.spyOn(directorApi, "migrateStorage");
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const user = userEvent.setup();

    const view = render(<App />);
    expect((await screen.findAllByText(/正在从服务器恢复时间线/)).length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const path = await screen.findByLabelText("数据库目标路径");
    expect(path).toBeDisabled();
    expect(screen.getByRole("button", { name: "迁移当前数据库并切换" })).toBeDisabled();
    expect(migrate).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).not.toBeNull();
    view.unmount();
    resolveAppStorage(ACTIVE_STORAGE_CONFIGURATION);
  });

  it("切库成功后关闭设置仍全局锁写且不产生新 WAL，PUT active 后解除", async () => {
    const user = userEvent.setup();
    const target = "/srv/director/data/next.sqlite3";
    const pendingStorage = {
      ...ACTIVE_STORAGE_CONFIGURATION,
      configured_database_path: target,
      restart_required: true,
    };
    mockCommonRequests();
    vi.mocked(directorApi.getStorage)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValue(pendingStorage);
    const migrate = vi.spyOn(directorApi, "migrateStorage").mockResolvedValue({
      ...pendingStorage,
      migrated_from: ACTIVE_DATABASE_PATH,
      migrated_to: target,
    });
    const cancelSwitch = vi.spyOn(directorApi, "updateStorage").mockResolvedValue(ACTIVE_STORAGE_CONFIGURATION);
    const updateTimeline = vi.mocked(directorApi.updateTimeline);
    const updateSettings = vi.spyOn(directorApi, "updateSettings");
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const path = await screen.findByLabelText("数据库目标路径");
    await user.clear(path);
    await user.type(path, target);
    await user.click(screen.getByRole("button", { name: "迁移当前数据库并切换" }));

    await waitFor(() => expect(migrate).toHaveBeenCalledWith(target));
    expect(await screen.findByText(/当前页面停止修改/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    const prompt = screen.getByLabelText("片段提示词");
    const before = (prompt as HTMLTextAreaElement).value;
    fireEvent.change(prompt, { target: { value: "切库锁定后不应写入" } });
    expect(prompt).toHaveValue(before);
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateTimeline).not.toHaveBeenCalled();
    expect(updateSettings).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).toBeNull();

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    await user.click(await screen.findByRole("button", { name: "取消切换并继续使用当前库" }));
    await waitFor(() => expect(cancelSwitch).toHaveBeenCalledWith(ACTIVE_DATABASE_PATH));
    await waitFor(() => expect(screen.queryByText(/当前页面停止修改/)).not.toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "取消后恢复写入" } });
    await waitFor(() => expect(updateTimeline).toHaveBeenCalledWith(
      expect.objectContaining({ segments: [expect.objectContaining({ prompt: "取消后恢复写入" })] }),
    ));
  });

  it("lost-response 后带同库 WAL 的 pending switch 可先取消冻结，再按顺序恢复两类 WAL", async () => {
    const user = userEvent.setup();
    const pendingTimeline = createTimelineProject();
    pendingTimeline.title = "冻结前尚未确认的时间线";
    pendingTimeline.segments[0].prompt = "待恢复时间线";
    saveLocalTimeline(pendingTimeline);
    const pendingSettings = { ...CONFIGURED_SETTINGS, client_id: "pending-after-lost-response" };
    saveRuntimeSettingsWal(pendingSettings);
    const pendingStorage = {
      ...ACTIVE_STORAGE_CONFIGURATION,
      configured_database_path: "/srv/director/data/next.sqlite3",
      restart_required: true,
    };
    mockCommonRequests();
    vi.mocked(directorApi.getStorage).mockResolvedValue(pendingStorage);
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(pendingSettings));
    const cancelSwitch = vi.spyOn(directorApi, "updateStorage").mockResolvedValue(ACTIVE_STORAGE_CONFIGURATION);
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(pendingSettings);
    const updateTimeline = vi.mocked(directorApi.updateTimeline);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    expect((await screen.findAllByText(pendingTimeline.title)).length).toBeGreaterThan(0);
    expect(await screen.findByText(/当前页面停止修改/)).toBeInTheDocument();
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateSettings).not.toHaveBeenCalled();
    expect(updateTimeline).not.toHaveBeenCalled();
    expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).not.toBeNull();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    await user.click(await screen.findByRole("button", { name: "取消切换并继续使用当前库" }));

    await waitFor(() => expect(cancelSwitch).toHaveBeenCalledWith(ACTIVE_DATABASE_PATH));
    await waitFor(() => expect(updateSettings).toHaveBeenCalledWith(pendingSettings));
    await waitFor(() => expect(updateTimeline).toHaveBeenCalledWith(
      expect.objectContaining({ title: pendingTimeline.title }),
    ));
    expect(cancelSwitch.mock.invocationCallOrder[0]).toBeLessThan(updateSettings.mock.invocationCallOrder[0]);
    expect(cancelSwitch.mock.invocationCallOrder[0]).toBeLessThan(updateTimeline.mock.invocationCallOrder[0]);
    await waitFor(() => expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).toBeNull());
    await waitFor(() => expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull());
  });

  it("迁移响应丢失但服务器已提交时通过 storage GET 转为重启锁，关闭设置也不能继续写", async () => {
    const user = userEvent.setup();
    const target = "/srv/director/data/response-lost.sqlite3";
    const pendingStorage = {
      ...ACTIVE_STORAGE_CONFIGURATION,
      configured_database_path: target,
      restart_required: true,
    };
    mockCommonRequests();
    vi.mocked(directorApi.getStorage)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValue(pendingStorage);
    const migrate = vi.spyOn(directorApi, "migrateStorage")
      .mockRejectedValue(new TypeError("response connection lost"));
    const updateTimeline = vi.mocked(directorApi.updateTimeline);
    const updateSettings = vi.spyOn(directorApi, "updateSettings");
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    updateTimeline.mockClear();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const path = await screen.findByLabelText("数据库目标路径");
    await user.clear(path);
    await user.type(path, target);
    await user.click(screen.getByRole("button", { name: "迁移当前数据库并切换" }));

    await waitFor(() => expect(migrate).toHaveBeenCalledWith(target));
    expect(await screen.findByRole("button", { name: "取消切换并继续使用当前库" })).toBeInTheDocument();
    expect(screen.getByText(/服务器确认数据库迁移或路径切换正在等待重启/)).toBeInTheDocument();
    expect(screen.getByText(/当前页面停止修改/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    const prompt = screen.getByLabelText("片段提示词");
    const before = (prompt as HTMLTextAreaElement).value;
    fireEvent.change(prompt, { target: { value: "响应丢失后不得写入" } });
    expect(prompt).toHaveValue(before);
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateTimeline).not.toHaveBeenCalled();
    expect(updateSettings).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).toBeNull();
  });

  it("迁移响应与后续 storage GET 都不可达时持续全局锁定并自动重试核对", async () => {
    const user = userEvent.setup();
    const target = "/srv/director/data/unknown-result.sqlite3";
    mockCommonRequests();
    vi.mocked(directorApi.getStorage)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockResolvedValueOnce(ACTIVE_STORAGE_CONFIGURATION)
      .mockRejectedValue(new TypeError("storage authority unavailable"));
    vi.spyOn(directorApi, "migrateStorage")
      .mockRejectedValue(new TypeError("response connection lost"));
    const updateTimeline = vi.mocked(directorApi.updateTimeline);
    const updateSettings = vi.spyOn(directorApi, "updateSettings");
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    updateTimeline.mockClear();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const path = await screen.findByLabelText("数据库目标路径");
    await user.clear(path);
    await user.type(path, target);
    await user.click(screen.getByRole("button", { name: "迁移当前数据库并切换" }));

    expect((await screen.findAllByText(/数据库操作响应尚未确认，正在自动核对服务器状态/)).length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    expect(screen.getByText(/数据库操作响应尚未确认，正在自动核对服务器状态/)).toBeInTheDocument();
    const prompt = screen.getByLabelText("片段提示词");
    const before = (prompt as HTMLTextAreaElement).value;
    fireEvent.change(prompt, { target: { value: "未知结果时不得写入" } });
    expect(prompt).toHaveValue(before);
    await waitFor(() => expect(directorApi.getStorage).toHaveBeenCalledTimes(5), { timeout: 2_500 });
    expect(updateTimeline).not.toHaveBeenCalled();
    expect(updateSettings).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).toBeNull();
  });

  it("存储请求收到明确 4xx 拒绝时安全解除操作锁并恢复编辑", async () => {
    const user = userEvent.setup();
    const target = "/srv/director/data/rejected.sqlite3";
    mockCommonRequests();
    vi.spyOn(directorApi, "migrateStorage")
      .mockRejectedValue(new ApiError("目标数据库校验失败", 422));
    const updateTimeline = vi.mocked(directorApi.updateTimeline);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    updateTimeline.mockClear();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const path = await screen.findByLabelText("数据库目标路径");
    await user.clear(path);
    await user.type(path, target);
    await user.click(screen.getByRole("button", { name: "迁移当前数据库并切换" }));

    expect(await screen.findByText("目标数据库校验失败")).toBeInTheDocument();
    expect(screen.queryByText(/数据库操作响应尚未确认/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    fireEvent.change(screen.getByLabelText("片段提示词"), {
      target: { value: "明确拒绝后恢复编辑" },
    });
    await waitFor(() => expect(updateTimeline).toHaveBeenCalledWith(expect.objectContaining({
      segments: [expect.objectContaining({ prompt: "明确拒绝后恢复编辑" })],
    })));
  });

  it("顶栏项目名可用键盘进入编辑，并实时同步且不显示任何保存控件", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const update = vi.mocked(directorApi.updateTimeline);

    render(<App />);
    await waitUntilReady();

    const rename = screen.getByRole("button", { name: /重命名项目，当前名称：未命名长视频/ });
    rename.focus();
    await user.keyboard("{Enter}");

    const input = screen.getByRole("textbox", { name: "编辑项目名称" });
    expect(input).toHaveFocus();
    expect((input as HTMLInputElement).selectionStart).toBe(0);
    expect((input as HTMLInputElement).selectionEnd).toBe("未命名长视频".length);
    await user.clear(input);
    await user.type(input, "键盘重命名项目{Enter}");

    expect(screen.getByRole("button", { name: /重命名项目，当前名称：键盘重命名项目/ })).toBeInTheDocument();
    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({ title: "键盘重命名项目" })));
    expect(screen.queryByRole("button", { name: /保存时间线|重新同步/ })).not.toBeInTheDocument();
    expect(screen.queryByText(/本地未保存|保存中|保存失败|时间线已同步/)).not.toBeInTheDocument();
  });

  it("顶栏点击重命名支持 Escape 取消、空白恢复和失焦提交", async () => {
    const user = userEvent.setup();
    mockCommonRequests();

    render(<App />);
    await waitUntilReady();

    await user.click(screen.getByRole("button", { name: /重命名项目，当前名称：未命名长视频/ }));
    let input = screen.getByRole("textbox", { name: "编辑项目名称" });
    await user.clear(input);
    await user.type(input, "应被取消{Escape}");
    expect(screen.queryByRole("textbox", { name: "编辑项目名称" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /重命名项目，当前名称：未命名长视频/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /重命名项目，当前名称：未命名长视频/ }));
    input = screen.getByRole("textbox", { name: "编辑项目名称" });
    await user.clear(input);
    fireEvent.blur(input);
    expect(screen.getByRole("button", { name: /重命名项目，当前名称：未命名长视频/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /重命名项目，当前名称：未命名长视频/ }));
    input = screen.getByRole("textbox", { name: "编辑项目名称" });
    await user.clear(input);
    await user.type(input, "  失焦重命名项目  ");
    fireEvent.blur(input);
    expect(screen.getByRole("button", { name: /重命名项目，当前名称：失焦重命名项目/ })).toBeInTheDocument();
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalledWith(
      expect.objectContaining({ title: "失焦重命名项目" }),
    ));
  });

  it("快速连续编辑只同步最终快照，纯选择不触发时间线写入", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments.push(createTimelineSegment("fl2va", 2));
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(project);
    const update = vi.mocked(directorApi.updateTimeline);

    render(<App />);
    await waitUntilReady();
    const prompt = screen.getByLabelText("片段提示词");
    await user.clear(prompt);
    await user.type(prompt, "快速连续输入");

    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    expect(update).toHaveBeenLastCalledWith(expect.objectContaining({
      segments: expect.arrayContaining([expect.objectContaining({ prompt: "快速连续输入" })]),
    }));
    await waitFor(() => expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull());

    await user.click(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ }));
    expect(screen.getByRole("button", { name: /^聚焦并选择片段 1：/ }))
      .toHaveAttribute("aria-pressed", "false");
    await new Promise((resolve) => window.setTimeout(resolve, 250));
    expect(update).toHaveBeenCalledTimes(1);
  });

  it("在途旧响应不覆盖后续编辑，并在旧请求结束后严格串行同步最新版", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    let resolveFirst!: (value: ReturnType<typeof createTimelineProject>) => void;
    const update = vi.mocked(directorApi.updateTimeline)
      .mockImplementationOnce((value) => new Promise((resolve) => { resolveFirst = resolve; }))
      .mockImplementation(async (value) => value);

    render(<App />);
    await waitUntilReady();
    const prompt = screen.getByLabelText("片段提示词");
    await user.clear(prompt);
    await user.type(prompt, "第一版");
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await user.type(prompt, "，第二版");
    await new Promise((resolve) => window.setTimeout(resolve, 250));
    expect(update).toHaveBeenCalledTimes(1);

    const stale = structuredClone(update.mock.calls[0][0]);
    stale.title = "迟到响应不应覆盖";
    await act(async () => resolveFirst(stale));
    await waitFor(() => expect(update).toHaveBeenCalledTimes(2));
    expect(update).toHaveBeenLastCalledWith(expect.objectContaining({
      segments: [expect.objectContaining({ prompt: "第一版，第二版" })],
    }));
    expect(screen.queryByText("迟到响应不应覆盖")).not.toBeInTheDocument();
  });

  it("确定性 422 不定时重试；失败期间已有的新编辑会立即同步最新 revision", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    let rejectFirst!: (reason: unknown) => void;
    const update = vi.mocked(directorApi.updateTimeline)
      .mockImplementationOnce(() => new Promise((_, reject) => { rejectFirst = reject; }))
      .mockImplementation(async (value) => value);

    render(<App />);
    await waitUntilReady();
    const prompt = screen.getByLabelText("片段提示词");
    await user.clear(prompt);
    await user.type(prompt, "无效中间态");
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await user.type(prompt, "，已修正");
    await act(async () => rejectFirst(new ApiError("字段无效", 422)));

    await waitFor(() => expect(update).toHaveBeenCalledTimes(2));
    expect(update).toHaveBeenLastCalledWith(expect.objectContaining({
      segments: [expect.objectContaining({ prompt: "无效中间态，已修正" })],
    }));
    await new Promise((resolve) => window.setTimeout(resolve, 1450));
    expect(update).toHaveBeenCalledTimes(2);
  });

  it("确定性 422 持久阻断当前 revision，下一次真实编辑成功后自动恢复", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      mockCommonRequests();
      const update = vi.mocked(directorApi.updateTimeline)
        .mockRejectedValueOnce(new ApiError("字段无效", 422))
        .mockImplementation(async (value) => value);

      render(<App />);
      await waitUntilReady();
      const prompt = screen.getByLabelText("片段提示词");
      fireEvent.change(prompt, { target: { value: "错误 revision" } });
      await act(async () => { await vi.advanceTimersByTimeAsync(151); });
      await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
      await waitFor(() => expect(screen.getByText(
        "服务器拒绝当前时间线：字段无效；请修改，修改后自动应用",
      )).toBeInTheDocument());

      await act(async () => { await vi.advanceTimersByTimeAsync(4_100); });
      expect(update).toHaveBeenCalledTimes(1);
      expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).not.toBeNull();
      expect(screen.queryByText("字段无效")).not.toBeInTheDocument();
      expect(screen.getByText(
        "服务器拒绝当前时间线：字段无效；请修改，修改后自动应用",
      )).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
      expect(screen.getByRole("button", { name: /生成任务 1/ })).toBeDisabled();

      fireEvent.change(prompt, { target: { value: "错误 revision 已修正" } });
      expect(screen.queryByText(/服务器拒绝当前时间线/)).not.toBeInTheDocument();
      await act(async () => { await vi.advanceTimersByTimeAsync(151); });
      await waitFor(() => expect(update).toHaveBeenCalledTimes(2));
      expect(update).toHaveBeenLastCalledWith(expect.objectContaining({
        segments: [expect.objectContaining({ prompt: "错误 revision 已修正" })],
      }));
      await waitFor(() => expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull());
      expect(screen.queryByText(/服务器拒绝当前时间线/)).not.toBeInTheDocument();
      expect(screen.getByRole("button", { name: "预检执行计划" })).toBeEnabled();
      expect(screen.getByRole("button", { name: /生成任务 1/ })).toBeEnabled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("从 Director 品牌切换素材库宽栏与常驻窄轨", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    render(<App />);
    await waitUntilReady();

    const toggle = screen.getByRole("button", { name: "素材库" });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    await user.click(toggle);
    const rail = screen.getByRole("complementary", { name: "当前工作区素材库" });
    expect(rail).toHaveClass("is-collapsed");
    expect(document.getElementById("director-sidebar-content")).toHaveAttribute("hidden");
    expect(screen.getByRole("button", { name: "系统设置" })).toBeInTheDocument();

    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle.querySelector(".brand__mark")).toBeInTheDocument();
    await waitFor(() => expect(toggle).toHaveFocus());
    await user.click(toggle);
    expect(rail).toHaveClass("is-open");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    await waitFor(() => expect(toggle).toHaveFocus());

    await user.keyboard("{Escape}");
    await waitFor(() => expect(toggle).toHaveFocus());
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    expect(await screen.findByRole("dialog", { name: "系统设置" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
  });

  it("持久化素材栏宽度，并在视口变化时限制在半屏以内", async () => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 1400 });
    localStorage.setItem("director-web:sidebar-expanded-width", "损坏值");
    mockCommonRequests();
    render(<App />);
    await waitUntilReady();

    const handle = screen.getByRole("separator", { name: "调整素材库宽度" });
    expect(handle).toHaveAttribute("aria-valuemin", "292");
    expect(handle).toHaveAttribute("aria-valuemax", "700");
    expect(handle).toHaveAttribute("aria-valuenow", "292");
    fireEvent.keyDown(handle, { key: "End" });
    await waitFor(() => expect(handle).toHaveAttribute("aria-valuenow", "700"));
    await waitFor(() => expect(localStorage.getItem("director-web:sidebar-expanded-width")).toBe("700"));

    Object.defineProperty(window, "innerWidth", { configurable: true, value: 1000 });
    fireEvent(window, new Event("resize"));
    await waitFor(() => {
      expect(handle).toHaveAttribute("aria-valuemin", "235");
      expect(handle).toHaveAttribute("aria-valuemax", "500");
      expect(handle).toHaveAttribute("aria-valuenow", "500");
      expect(localStorage.getItem("director-web:sidebar-expanded-width")).toBe("500");
    });
  });

  it("后端离线时仍可编排，但所有生成入口均禁用", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    vi.mocked(directorApi.getSettingsAuthority).mockRejectedValue(new Error("离线"));

    render(<App />);
    expect(await screen.findByText("ComfyUI 离线")).toBeInTheDocument();
    await user.type(screen.getByLabelText("片段提示词"), "雨夜追车");
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeDisabled();
    expect(screen.getByText(/编辑内容会在 Director 连接恢复后自动同步/)).toBeInTheDocument();
  });

  it("无 LoRA 的 RayLight 不会被条件 RayLoraLoader 能力误拦", async () => {
    mockCommonRequests({
      ...CONFIGURED_SETTINGS,
      models: {
        ...CONFIGURED_SETTINGS.models,
        fl2va: {
          ...CONFIGURED_SETTINGS.models.fl2va,
          backend: "raylight",
          lora_name: null,
          raylight: {
            ...CONFIGURED_SETTINGS.models.fl2va.raylight,
            gpu_select: [0, 1],
            ulysses_degree: 2,
          },
        },
      },
    });
    vi.mocked(directorApi.getCapabilities).mockResolvedValue({
      ...ONLINE_CAPABILITIES,
      execution_backends: {
        standard: { available: true, missing_nodes: [] },
        raylight: {
          available: true,
          missing_nodes: [],
          conditional_requirements: {
            lora: { available: false, missing_nodes: ["RayLoraLoader"] },
          },
        },
      },
    });

    render(<App />);
    await waitUntilReady();

    expect(screen.queryByText(/RayLight LoRA 配置不可用/)).not.toBeInTheDocument();
    expect(screen.queryByText(/配置解析为 RayLight 执行，但当前 ComfyUI 不可用/)).not.toBeInTheDocument();
  });

  it("启用 RayLight LoRA 时按条件节点能力失败封闭", async () => {
    mockCommonRequests({
      ...CONFIGURED_SETTINGS,
      models: {
        ...CONFIGURED_SETTINGS.models,
        fl2va: {
          ...CONFIGURED_SETTINGS.models.fl2va,
          backend: "raylight",
          lora_name: "minimax-h3-turbo.safetensors",
          raylight: {
            ...CONFIGURED_SETTINGS.models.fl2va.raylight,
            gpu_select: [0, 1],
            ulysses_degree: 2,
          },
        },
      },
    });
    vi.mocked(directorApi.getCapabilities).mockResolvedValue({
      ...ONLINE_CAPABILITIES,
      execution_backends: {
        standard: { available: true, missing_nodes: [] },
        raylight: {
          available: true,
          missing_nodes: [],
          conditional_requirements: {
            lora: { available: false, missing_nodes: ["RayLoraLoader"] },
          },
        },
      },
    });

    render(<App />);
    await waitUntilReady();

    expect(screen.getByRole("button", { name: /生成任务/ })).toBeDisabled();
    expect(screen.getByText(/RayLight LoRA 配置不可用.*RayLoraLoader/)).toBeInTheDocument();
  });

  it("全局设置是默认关闭的悬浮层，并同时编辑两个共享模型族", async () => {
    const user = userEvent.setup();
    const initial = structuredClone(CONFIGURED_SETTINGS);
    initial.models.ref2va.lora_name = "style.safetensors";
    initial.models.ref2va.standard_lora_loader_override = {
      loader: "model_only",
      lora_name: "style.safetensors",
      model_filename: initial.models.ref2va.filename,
      comfy_origin: initial.comfy_url,
    };
    mockCommonRequests(initial);
    const confirmed = structuredClone(initial);
    confirmed.models.ref2va.filename = "alternate-diffusion.safetensors";
    confirmed.models.ref2va.standard_lora_loader_override = null;
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(initial))
      .mockResolvedValueOnce(runtimeAuthority(initial))
      .mockResolvedValue(runtimeAuthority(confirmed));
    const update = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(confirmed);

    render(<App />);
    await waitUntilReady();
    expect(screen.queryByRole("region", { name: "输出规格" })).not.toBeInTheDocument();
    await openGlobalSettings(user);
    const globalSettings = screen.getByRole("region", { name: "时间线全局设置" });
    expect(globalSettings).toBeVisible();
    const specs = within(globalSettings).getByRole("region", { name: "输出规格" });
    expect(specs).toBeVisible();
    expect(within(specs).getByLabelText("画幅")).toHaveValue("16:9");
    expect(within(specs).getByLabelText("分辨率")).toHaveValue("864x480");
    expect(within(specs).getByLabelText("导出方式")).toHaveValue("all");
    expect(within(globalSettings).queryByLabelText("音频策略")).not.toBeInTheDocument();
    expect(within(globalSettings).queryByLabelText("参考图采样尺寸")).not.toBeInTheDocument();
    await user.selectOptions(within(specs).getByLabelText("分辨率"), "1920x1088");
    await user.selectOptions(within(specs).getByLabelText("画幅"), "9:16");
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenLastCalledWith(
      expect.objectContaining({ render: { width: 1088, height: 1920, fps: 24 } }),
    ));
    expect(screen.getByLabelText("FL2VA Diffusion 模型快捷选择")).toBeInTheDocument();
    expect(screen.getByLabelText("REF2VA Diffusion 模型快捷选择")).toBeInTheDocument();
    const flFamily = within(globalSettings).getByRole("region", { name: "FL2VA" });
    const refFamily = within(globalSettings).getByRole("region", { name: "Ref2VA" });
    expect(within(flFamily).getByLabelText("FL2VA Diffusion 模型快捷选择")).toBeInTheDocument();
    expect(within(flFamily).getByLabelText("FL2VA 步数")).toBeInTheDocument();
    expect(within(refFamily).getByLabelText("REF2VA LoRA 模型快捷选择")).toBeInTheDocument();
    expect(within(refFamily).getByLabelText("Ref2VA Seed")).toBeInTheDocument();
    expect(document.getElementById("timeline-global-settings-fl2va-title")?.parentElement).toHaveClass("timeline-family-settings__title");
    expect(within(flFamily).getByLabelText("FL2VA Diffusion 模型快捷选择").closest(".field")).toHaveClass("field--inline");
    expect(within(flFamily).getByLabelText("FL2VA 步数").closest(".field")).toHaveClass("field--inline");
    expect(within(flFamily).getByLabelText("FL2VA Seed").closest(".field")).toHaveClass("field--inline");
    expect(within(flFamily).queryByText("每次生成前重掷，输入框显示本次实际数值")).not.toBeInTheDocument();
    expect(within(flFamily).queryByText("FL2VA 推理参数")).not.toBeInTheDocument();
    expect(within(refFamily).queryByText("Ref2VA 推理参数")).not.toBeInTheDocument();
    expect(within(flFamily).getByLabelText("FL2VA Diffusion 模型快捷选择").closest("header")).toBe(flFamily.querySelector(":scope > header"));
    expect(within(refFamily).getByLabelText("REF2VA LoRA 模型快捷选择").closest("header")).toBe(refFamily.querySelector(":scope > header"));
    const flScheduler = within(flFamily).getByLabelText("FL2VA 调度器");
    const refScheduler = within(refFamily).getByLabelText("Ref2VA 调度器");
    expect(within(flScheduler).getAllByRole("option").map((option) => option.getAttribute("value"))).toEqual([
      "simple", "normal", "karras", "beta",
    ]);
    await user.selectOptions(flScheduler, "beta");
    expect(flScheduler).toHaveValue("beta");
    expect(refScheduler).toHaveValue("simple");
    expect(within(globalSettings).queryByText("LoRA 加载器")).not.toBeInTheDocument();
    expect(within(globalSettings).queryByText("执行策略")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("全片默认提示词")).not.toBeInTheDocument();
    expect(screen.queryByText(/^CFG$/)).not.toBeInTheDocument();
    const refSteps = screen.getByLabelText("Ref2VA 步数");
    expect(refSteps).toHaveValue(25);
    await user.clear(screen.getByLabelText("FL2VA 步数"));
    await user.type(screen.getByLabelText("FL2VA 步数"), "31");
    expect(screen.getByLabelText("FL2VA 步数")).toHaveValue(31);
    expect(refSteps).toHaveValue(25);
    await user.selectOptions(
      screen.getByLabelText("REF2VA Diffusion 模型快捷选择"),
      "alternate-diffusion.safetensors",
    );
    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({
      models: expect.objectContaining({
        ref2va: expect.objectContaining({
          filename: "alternate-diffusion.safetensors",
          standard_lora_loader_override: null,
        }),
      }),
    })));
    await user.keyboard("{Escape}");
    expect(document.getElementById("timeline-global-settings")).toHaveAttribute("hidden");
    await waitFor(() => expect(screen.getByRole("button", { name: "全局设置" })).toHaveFocus());
  });

  it("预检与生成主操作只渲染在顶栏右侧，并跟随可执行选择数量", async () => {
    const user = userEvent.setup();
    mockCommonRequests();

    render(<App />);
    await waitUntilReady();

    const topbar = document.querySelector(".topbar--timeline");
    const commandbar = document.querySelector(".timeline-commandbar");
    expect(topbar).not.toBeNull();
    expect(commandbar).not.toBeNull();
    const runActions = within(topbar as HTMLElement).getByRole("group", { name: "时间线生成操作" });
    const globalSettings = within(topbar as HTMLElement).getByRole("button", { name: "全局设置" });
    expect(runActions).toBeInTheDocument();
    expect(globalSettings.closest(".topbar__right")).not.toBeNull();
    expect(globalSettings.nextElementSibling).toBe(runActions);
    expect(within(topbar as HTMLElement).getByRole("button", { name: "预检执行计划" })).toBeInTheDocument();
    expect(within(topbar as HTMLElement).getByRole("button", { name: "生成任务 1" })).toBeInTheDocument();
    expect(within(commandbar as HTMLElement).queryByRole("button", { name: "预检执行计划" })).not.toBeInTheDocument();
    expect(within(commandbar as HTMLElement).queryByRole("button", { name: /生成任务/ })).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "预检执行计划" })).toHaveLength(1);
    expect(screen.getAllByRole("button", { name: "生成任务 1" })).toHaveLength(1);
    const preflight = screen.getByRole("button", { name: "预检执行计划" });
    expect(within(preflight).getByText("预检执行计划")).toHaveClass("topbar__action-label--long");
    expect(within(preflight).getByText("预检")).toHaveClass("topbar__action-label--short");
    expect(within(preflight).getByText("预检")).toHaveAttribute("aria-hidden", "true");
    const generate = screen.getByRole("button", { name: "生成任务 1" });
    expect(within(generate).getByText("生成任务 1")).toHaveClass("topbar__action-label--long");
    expect(within(generate).getByText("生成 1")).toHaveClass("topbar__action-label--short");
    expect(within(generate).getByText("生成 1")).toHaveAttribute("aria-hidden", "true");

    await user.click(screen.getByRole("checkbox", { name: "全选" }));
    expect(screen.getByRole("button", { name: "生成任务 0" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
  });

  it("按当前数据库和项目恢复统一选择，并在刷新后保留明确取消全选", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    const second = createTimelineSegment("fl2va", 2);
    project.segments = [project.segments[0], second];
    const segmentIds = project.segments.map((segment) => segment.id);
    saveTimelineSegmentSelectionPreference(
      ACTIVE_DATABASE,
      DEFAULT_PROJECT_ID,
      segmentIds,
      [second.id],
    );
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(project);

    const firstView = render(<App />);
    await waitUntilReady();
    expect(screen.getByRole("button", { name: `聚焦并选择片段 1：${project.segments[0].title}` }))
      .toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: `聚焦并选择片段 2：${second.title}` }))
      .toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "生成任务 1" })).toBeInTheDocument();
    expect(directorApi.updateTimeline).not.toHaveBeenCalled();

    const selectAll = screen.getByRole("checkbox", { name: "全选" });
    await user.click(selectAll);
    await waitFor(() => expect(loadTimelineSegmentSelectionPreference(
      ACTIVE_DATABASE,
      DEFAULT_PROJECT_ID,
      segmentIds,
    )).toEqual(segmentIds));
    expect(screen.getByRole("button", { name: "生成任务 2" })).toBeInTheDocument();

    await user.click(selectAll);
    await waitFor(() => expect(loadTimelineSegmentSelectionPreference(
      ACTIVE_DATABASE,
      DEFAULT_PROJECT_ID,
      segmentIds,
    )).toEqual([]));
    expect(screen.getByRole("button", { name: "生成任务 0" })).toBeDisabled();
    expect(directorApi.updateTimeline).not.toHaveBeenCalled();

    firstView.unmount();
    render(<App />);
    await waitUntilReady();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: `聚焦并选择片段 1：${project.segments[0].title}` }))
        .toHaveAttribute("aria-pressed", "false");
      expect(screen.getByRole("button", { name: `聚焦并选择片段 2：${second.title}` }))
        .toHaveAttribute("aria-pressed", "false");
    });
    expect(screen.getByRole("button", { name: "生成任务 0" })).toBeDisabled();
  });

  it("切换到复用相同 segment ID 的项目时不继承上一项目的子集选择", async () => {
    const user = userEvent.setup();
    const projectA = createTimelineProject();
    const second = createTimelineSegment("fl2va", 2);
    projectA.title = "项目 A";
    projectA.segments = [projectA.segments[0], second];
    const projectB = structuredClone(projectA);
    projectB.title = "项目 B";
    projectB.segments = projectB.segments.map((segment, index) => ({
      ...segment,
      title: `B 片段 ${index + 1}`,
    }));
    const segmentIds = projectA.segments.map((segment) => segment.id);
    saveTimelineSegmentSelectionPreference(
      ACTIVE_DATABASE,
      DEFAULT_PROJECT_ID,
      segmentIds,
      [second.id],
    );
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(projectA);
    vi.mocked(directorApi.listProjects).mockResolvedValue({
      projects: [
        {
          id: DEFAULT_PROJECT_ID,
          title: projectA.title,
          created_at: "2026-08-12T00:00:00Z",
          updated_at: "2026-08-12T00:00:00Z",
          segment_count: 2,
        },
        {
          id: "project-b",
          title: projectB.title,
          created_at: "2026-08-12T00:00:00Z",
          updated_at: "2026-08-12T00:00:00Z",
          segment_count: 2,
        },
      ],
    });
    vi.spyOn(directorApi, "getProjectTimeline").mockResolvedValue(projectB);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(screen.getByRole("button", {
      name: /^聚焦并选择片段 1：/,
    })).toHaveAttribute("aria-pressed", "false"));

    await user.selectOptions(
      await screen.findByRole("combobox", { name: "切换项目" }),
      "project-b",
    );
    await waitFor(() => expect(screen.getByRole("checkbox", { name: "全选" })).toBeChecked());
    expect(screen.getByRole("button", { name: /^聚焦并选择片段 1：B 片段 1/ }))
      .toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /^聚焦并选择片段 2：B 片段 2/ }))
      .toHaveAttribute("aria-pressed", "true");
    expect(loadTimelineSegmentSelectionPreference(
      ACTIVE_DATABASE,
      "project-b",
      segmentIds,
    )).toBeNull();
  });

  it("快速切换项目时迟到的旧项目响应不能覆盖最新目标", async () => {
    const projectA = createTimelineProject();
    projectA.title = "项目 A";
    const projectB = structuredClone(projectA);
    projectB.title = "项目 B";
    const projectC = structuredClone(projectA);
    projectC.title = "项目 C";
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(projectA);
    vi.mocked(directorApi.listProjects).mockResolvedValue({
      projects: [
        { id: DEFAULT_PROJECT_ID, title: projectA.title, created_at: "2026-08-12T00:00:00Z", updated_at: "2026-08-12T00:00:00Z", segment_count: 1 },
        { id: "project-b", title: projectB.title, created_at: "2026-08-12T00:00:00Z", updated_at: "2026-08-12T00:00:00Z", segment_count: 1 },
        { id: "project-c", title: projectC.title, created_at: "2026-08-12T00:00:00Z", updated_at: "2026-08-12T00:00:00Z", segment_count: 1 },
      ],
    });
    let resolveProjectB!: (project: typeof projectB) => void;
    const getProjectTimeline = vi.spyOn(directorApi, "getProjectTimeline")
      .mockImplementation((projectId) => projectId === "project-b"
        ? new Promise((resolve) => { resolveProjectB = resolve; })
        : Promise.resolve(projectC));

    render(<App />);
    await waitUntilReady();
    const switcher = await screen.findByRole("combobox", { name: "切换项目" });
    fireEvent.change(switcher, { target: { value: "project-b" } });
    await waitFor(() => expect(getProjectTimeline).toHaveBeenCalledWith("project-b", undefined));
    fireEvent.change(switcher, { target: { value: "project-c" } });
    await waitFor(() => expect(screen.getByRole("button", {
      name: "重命名项目，当前名称：项目 C",
    })).toBeInTheDocument());

    await act(async () => resolveProjectB(projectB));
    await new Promise((resolve) => window.setTimeout(resolve, 20));
    expect(screen.getByRole("button", { name: "重命名项目，当前名称：项目 C" }))
      .toBeInTheDocument();
  });

  it("目标项目仍在加载时改回当前项目会取消迟到的切换", async () => {
    const projectA = createTimelineProject();
    projectA.title = "项目 A";
    const projectB = structuredClone(projectA);
    projectB.title = "项目 B";
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(projectA);
    vi.mocked(directorApi.listProjects).mockResolvedValue({
      projects: [
        { id: DEFAULT_PROJECT_ID, title: projectA.title, created_at: "2026-08-12T00:00:00Z", updated_at: "2026-08-12T00:00:00Z", segment_count: 1 },
        { id: "project-b", title: projectB.title, created_at: "2026-08-12T00:00:00Z", updated_at: "2026-08-12T00:00:00Z", segment_count: 1 },
      ],
    });
    let resolveProjectB!: (project: typeof projectB) => void;
    const getProjectTimeline = vi.spyOn(directorApi, "getProjectTimeline")
      .mockImplementation(() => new Promise((resolve) => { resolveProjectB = resolve; }));

    render(<App />);
    await waitUntilReady();
    const switcher = await screen.findByRole("combobox", { name: "切换项目" });
    fireEvent.change(switcher, { target: { value: "project-b" } });
    await waitFor(() => expect(getProjectTimeline).toHaveBeenCalledWith("project-b", undefined));
    fireEvent.change(switcher, { target: { value: DEFAULT_PROJECT_ID } });

    await act(async () => resolveProjectB(projectB));
    await new Promise((resolve) => window.setTimeout(resolve, 20));
    expect(switcher).toHaveValue(DEFAULT_PROJECT_ID);
    expect(screen.getByRole("button", { name: "重命名项目，当前名称：项目 A" }))
      .toBeInTheDocument();
  });

  it("目标项目加载期间的新编辑会在切换交接前同步", async () => {
    const projectA = createTimelineProject();
    projectA.title = "项目 A";
    projectA.segments[0].prompt = "原提示词";
    const projectB = structuredClone(projectA);
    projectB.title = "项目 B";
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(projectA);
    vi.mocked(directorApi.listProjects).mockResolvedValue({
      projects: [
        { id: DEFAULT_PROJECT_ID, title: projectA.title, created_at: "2026-08-12T00:00:00Z", updated_at: "2026-08-12T00:00:00Z", segment_count: 1 },
        { id: "project-b", title: projectB.title, created_at: "2026-08-12T00:00:00Z", updated_at: "2026-08-12T00:00:00Z", segment_count: 1 },
      ],
    });
    let resolveProjectB!: (project: typeof projectB) => void;
    vi.spyOn(directorApi, "getProjectTimeline")
      .mockImplementation(() => new Promise((resolve) => { resolveProjectB = resolve; }));
    const updateTimeline = vi.mocked(directorApi.updateTimeline);

    render(<App />);
    await waitUntilReady();
    fireEvent.change(await screen.findByRole("combobox", { name: "切换项目" }), {
      target: { value: "project-b" },
    });
    await waitFor(() => expect(directorApi.getProjectTimeline)
      .toHaveBeenCalledWith("project-b", undefined));
    fireEvent.change(screen.getByLabelText("片段提示词"), {
      target: { value: "加载等待期间的新提示词" },
    });

    await act(async () => resolveProjectB(projectB));
    await waitFor(() => expect(updateTimeline).toHaveBeenCalledWith(expect.objectContaining({
      title: "项目 A",
      segments: expect.arrayContaining([
        expect.objectContaining({ prompt: "加载等待期间的新提示词" }),
      ]),
    })));
    expect(screen.getByRole("button", { name: "重命名项目，当前名称：项目 B" }))
      .toBeInTheDocument();
  });

  it("只选择停用片段时明确阻止预检和生成，重新启用后自动恢复", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "启用片段";
    const disabled = {
      ...createTimelineSegment("fl2va", 2),
      prompt: "停用片段",
      enabled: false,
    };
    project.segments.push(disabled);
    const segmentIds = project.segments.map((segment) => segment.id);
    saveTimelineSegmentSelectionPreference(
      ACTIVE_DATABASE,
      DEFAULT_PROJECT_ID,
      segmentIds,
      [disabled.id],
    );
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(project);
    const compile = vi.spyOn(directorApi, "compileTimeline");
    const submit = vi.spyOn(directorApi, "createTimelineTask");

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(screen.getByRole("checkbox", {
      name: `多选停用片段 2：${disabled.title}`,
    })).toBeChecked());

    expect(screen.getByRole("checkbox", { name: "全选" })).toBePartiallyChecked();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "所选片段均已停用；请启用至少一个所选片段后再生成",
    );
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "生成任务 0" })).toBeDisabled();
    expect(compile).not.toHaveBeenCalled();
    expect(submit).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: `启用片段 2：${disabled.title}` }));
    await waitFor(() => expect(screen.getByRole("button", { name: "生成任务 1" })).toBeEnabled());
    expect(screen.getByRole("checkbox", {
      name: `多选片段 2：${disabled.title}`,
    })).toBeChecked();
  });

  it("全选包含停用片段，但预检和生成 payload 只提交已启用交集", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    const first = { ...project.segments[0], prompt: "运行片段" };
    const disabled = {
      ...createTimelineSegment("ref2va", 2),
      enabled: false,
    };
    project.segments = [first, disabled];
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(project);
    const compile = vi.spyOn(directorApi, "compileTimeline").mockRejectedValue(
      new Error("仅检查请求边界"),
    );
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);

    render(<App />);
    await waitUntilReady();
    expect(screen.getByRole("checkbox", { name: "全选" })).toBeChecked();
    expect(screen.getByRole("checkbox", {
      name: `多选停用片段 2：${disabled.title}`,
    })).toBeChecked();
    expect(screen.getByRole("button", { name: "生成任务 1" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "预检执行计划" }));
    await waitFor(() => expect(compile).toHaveBeenCalledWith({
      config: expect.objectContaining({ segments: expect.any(Array) }),
      segment_ids: [first.id],
    }));
    await user.click(screen.getByRole("button", { name: "生成任务 1" }));
    await waitFor(() => expect(submit).toHaveBeenCalledWith({
      config: expect.objectContaining({ segments: expect.any(Array) }),
      segment_ids: [first.id],
    }));
  });

  it.each([320, 560])("%dpx 顶栏使用长短双标签且保持完整可访问名", async (width) => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: width });
    mockCommonRequests();

    render(<App />);
    await waitUntilReady();

    const actions = screen.getByRole("group", { name: "时间线生成操作" });
    const preflight = within(actions).getByRole("button", { name: "预检执行计划" });
    const generate = within(actions).getByRole("button", { name: "生成任务 1" });
    expect(actions.closest(".topbar__right")).not.toBeNull();
    expect(within(preflight).getByText("预检执行计划")).toHaveClass("topbar__action-label--long");
    expect(within(preflight).getByText("预检")).toHaveAttribute("aria-hidden", "true");
    expect(within(generate).getByText("生成任务 1")).toHaveClass("topbar__action-label--long");
    expect(within(generate).getByText("生成 1")).toHaveAttribute("aria-hidden", "true");
  });

  it("任务抽屉与全局设置互斥，避免两个悬浮层互相遮挡", async () => {
    const user = userEvent.setup();
    mockCommonRequests();

    render(<App />);
    await waitUntilReady();
    const settingsToggle = screen.getByRole("button", { name: "全局设置" });
    const taskToggle = screen.getByRole("button", { name: "任务，0 个进行中" });
    expect(taskToggle.querySelector("strong")).toHaveTextContent("0");

    await user.click(settingsToggle);
    expect(settingsToggle).toHaveAttribute("aria-expanded", "true");
    await user.click(taskToggle);
    expect(settingsToggle).toHaveAttribute("aria-expanded", "false");
    expect(taskToggle).toHaveAttribute("aria-expanded", "true");
    expect(taskToggle.querySelector("strong")).toHaveTextContent("0");

    await user.click(settingsToggle);
    expect(settingsToggle).toHaveAttribute("aria-expanded", "true");
    expect(taskToggle).toHaveAttribute("aria-expanded", "false");
    expect(taskToggle.querySelector("strong")).toHaveTextContent("0");
  });

  it("点击工作区收起任务面板且不吞掉原操作，面板内部和任务按钮不误触发", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "从顶栏预检";
    saveLocalTimeline(project);
    mockCommonRequests();
    const compile = vi.spyOn(directorApi, "compileTimeline").mockResolvedValue({
      execution_strategy: "native_segment_graph_v1",
      model_families: ["fl2va"],
      plans: [],
      node_policy: { graph_source: "server", accepts_client_workflow: false, allowed_nodes: [], custom_nodes: [], provenance: {} },
    });

    render(<App />);
    await waitUntilReady();
    const taskToggle = screen.getByRole("button", { name: "任务，0 个进行中" });
    await user.click(taskToggle);
    expect(taskToggle).toHaveAttribute("aria-expanded", "true");

    await user.click(screen.getByRole("searchbox", { name: "搜索任务" }));
    expect(taskToggle).toHaveAttribute("aria-expanded", "true");

    const preflight = screen.getByRole("button", { name: "预检执行计划" });
    expect(preflight).toBeVisible();
    await user.click(preflight);
    expect(taskToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("complementary", { name: "任务列表" })).not.toBeInTheDocument();
    await waitFor(() => expect(compile).toHaveBeenCalledWith({
      config: expect.objectContaining({ version: 4 }),
      segment_ids: [project.segments[0].id],
    }));

    await user.click(taskToggle);
    const liveToggle = screen.getByRole("button", { name: "实时执行" });
    await user.click(liveToggle);
    expect(taskToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("complementary", { name: "任务列表" })).not.toBeInTheDocument();
    expect(liveToggle).toHaveAttribute("aria-expanded", "true");
    expect(liveToggle).toHaveFocus();

    await user.click(taskToggle);
    expect(taskToggle).toHaveAttribute("aria-expanded", "true");
    await user.click(taskToggle);
    expect(taskToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("complementary", { name: "任务列表" })).not.toBeInTheDocument();
  });

  it("全局设置删除无操作的自动执行摘要，快捷模型选择仍保留驻留策略", async () => {
    const user = userEvent.setup();
    const initial = structuredClone(CONFIGURED_SETTINGS);
    initial.raylight_residency_policy = "release_after_sampling";
    initial.models.fl2va.backend = "auto";
    initial.models.fl2va.raylight = {
      ...initial.models.fl2va.raylight,
      gpu_select: [0, 1],
      ulysses_degree: 2,
    };
    const confirmed = structuredClone(initial);
    confirmed.models.fl2va.filename = "alternate-diffusion.safetensors";
    mockCommonRequests(initial);
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(initial))
      .mockResolvedValueOnce(runtimeAuthority(initial))
      .mockResolvedValue(runtimeAuthority(confirmed));
    const update = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(confirmed);

    render(<App />);
    await waitUntilReady();
    await openGlobalSettings(user);
    const flFamily = screen.getByRole("region", { name: "FL2VA" });
    const refFamily = screen.getByRole("region", { name: "Ref2VA" });
    expect(within(flFamily).queryByLabelText("FL2VA 自动执行状态")).not.toBeInTheDocument();
    expect(within(refFamily).queryByLabelText("Ref2VA 自动执行状态")).not.toBeInTheDocument();
    expect(within(flFamily).queryByText("自动执行")).not.toBeInTheDocument();
    expect(within(flFamily).queryByText("RayLight · 2 卡")).not.toBeInTheDocument();
    expect(within(flFamily).queryByText("执行策略")).not.toBeInTheDocument();
    expect(within(flFamily).queryByText("LoRA 加载器")).not.toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("FL2VA Diffusion 模型快捷选择"), "alternate-diffusion.safetensors");

    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({
      raylight_residency_policy: "release_after_sampling",
      models: expect.objectContaining({
        fl2va: expect.objectContaining({ filename: "alternate-diffusion.safetensors" }),
      }),
    })));
  });

  it("原生子图连续性能力缺省为关闭，旧草稿必须显式清除才能提交", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "延续镜头";
    project.segments[0].continuity.enabled = true;
    saveLocalTimeline(project);
    mockCommonRequests();

    render(<App />);
    await waitUntilReady();
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeDisabled();
    expect(screen.getByText(/当前原生分段子图不支持所选片段的段间接续/)).toBeInTheDocument();

    const continuityToggle = screen.getByRole("checkbox", { name: "启用当前片段连续性" });
    expect(continuityToggle).toBeChecked();
    await user.click(continuityToggle);
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeEnabled();
  });

  it("段间接续只选后段时保留显式勾选，并把历史前驱解析交给服务端", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "第一段";
    const second = { ...createTimelineSegment("fl2va", 2), prompt: "第二段" };
    second.continuity.enabled = true;
    project.segments.push(second);
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.getCapabilities).mockResolvedValue({
      ...ONLINE_CAPABILITIES,
      native_timeline: {
        supported: true,
        modes: ["fl2va", "ref2va"],
        continuity: true,
      },
    });
    const compile = vi.spyOn(directorApi, "compileTimeline").mockRejectedValue(
      new ApiError("没有可复用的前驱成片", 409),
    );
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockRejectedValue(
      new ApiError("没有可复用的前驱成片", 409),
    );

    render(<App />);
    await waitUntilReady();
    const preflight = screen.getByRole("button", { name: "预检执行计划" });
    const generateAll = screen.getByRole("button", { name: "生成任务 2" });
    expect(preflight).toBeEnabled();
    expect(generateAll).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ }));

    expect(screen.getByRole("button", { name: /^聚焦并选择片段 1：/ })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ })).toHaveAttribute("aria-pressed", "true");
    expect(preflight).toBeEnabled();
    const generateSelected = screen.getByRole("button", { name: "生成任务 1" });
    expect(generateSelected).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("复用前驱成片 1 段")).toBeInTheDocument();

    await user.click(preflight);
    await waitFor(() => expect(compile).toHaveBeenCalledWith({
      config: expect.objectContaining({ segments: expect.any(Array) }),
      segment_ids: [second.id],
    }));
    await user.click(generateSelected);
    await waitFor(() => expect(submit).toHaveBeenCalledWith({
      config: expect.objectContaining({ segments: expect.any(Array) }),
      segment_ids: [second.id],
    }));
    expect(screen.getByRole("button", { name: /^聚焦并选择片段 1：/ })).toHaveAttribute("aria-pressed", "false");
  });

  it("段间接续尾帧不足或内部采样超过 512 帧时在顶栏前直接阻断", async () => {
    const project = createTimelineProject();
    const first = {
      ...project.segments[0],
      prompt: "短前段",
      duration_seconds: 0.1,
    };
    const second = {
      ...createTimelineSegment("fl2va", 2),
      prompt: "长后段",
      duration_seconds: 498 / 24,
      continuity: { enabled: true, overlap_frames: 22 as const },
    };
    project.segments = [first, second];
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.getCapabilities).mockResolvedValue({
      ...ONLINE_CAPABILITIES,
      native_timeline: {
        supported: true,
        modes: ["fl2va", "ref2va"],
        continuity: true,
      },
    });

    render(<App />);
    await waitUntilReady();

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("只有 5 个可见帧，少于段间接续需要的 22 帧");
    expect(alert).toHaveTextContent("内部采样 532 帧（可见 498 帧）");
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
    const generate = screen.getByRole("button", { name: "生成任务 2" });
    expect(generate).toBeDisabled();
    expect(generate).toHaveAttribute("title", expect.stringContaining("请延长前段或降低接续尾帧数"));
  });

  it("所选执行后端不可用时禁止提交原生子图", async () => {
    const project = createTimelineProject();
    project.segments[0].prompt = "雾中森林";
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.getCapabilities).mockResolvedValue({
      ...ONLINE_CAPABILITIES,
      execution_backends: {
        ...ONLINE_CAPABILITIES.execution_backends!,
        standard: { available: false, missing_nodes: ["SamplerCustomAdvanced"] },
      },
    });

    render(<App />);
    await waitUntilReady();
    expect(screen.getByRole("alert")).toHaveTextContent("SamplerCustomAdvanced");
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeDisabled();
  });

  it("实时同步和生成使用统一 timeline API，而不是旧六模式接口", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const update = vi.spyOn(directorApi, "updateTimeline").mockImplementation(async (project) => project);
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);
    vi.mocked(directorApi.listTasks)
      .mockResolvedValueOnce({ jobs: [] })
      .mockResolvedValue({ jobs: [queuedTimelineTask] });

    render(<App />);
    await waitUntilReady();
    fireEvent.change(screen.getByLabelText("片段提示词"), {
      target: { value: "雨夜街头，低机位推进" },
    });
    await waitFor(() => expect(update).toHaveBeenCalledWith(expect.objectContaining({
      version: 4,
      segments: [expect.objectContaining({ mode: "fl2va", prompt: "雨夜街头，低机位推进" })],
    })));
    expect(screen.queryByRole("button", { name: /保存时间线/ })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
    await waitFor(() => expect(submit).toHaveBeenCalledWith({
      config: expect.objectContaining({
        version: 4,
        segments: [expect.objectContaining({ mode: "fl2va" })],
      }),
      segment_ids: [expect.any(String)],
    }));
    const taskDrawer = await screen.findByRole("complementary", { name: "任务列表" });
    expect(taskDrawer).toBeInTheDocument();
    expect(taskDrawer.closest(".app-shell")).not.toHaveClass("is-task-panel-open");
    expect(screen.getAllByText(/时间线/).length).toBeGreaterThan(0);
  });

  it("生成会等待当前 exact revision 的实时同步确认后再提交", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "等待权威确认后生成";
    saveLocalTimeline(project);
    mockCommonRequests();
    let resolveUpdate!: (value: typeof project) => void;
    const update = vi.mocked(directorApi.updateTimeline).mockImplementationOnce(
      () => new Promise((resolve) => { resolveUpdate = resolve; }),
    );
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
    expect(submit).not.toHaveBeenCalled();

    await act(async () => resolveUpdate(project));
    await waitFor(() => expect(submit).toHaveBeenCalledWith({
      config: expect.objectContaining({
        segments: [expect.objectContaining({ prompt: "等待权威确认后生成" })],
      }),
      segment_ids: [project.segments[0].id],
    }));
  });

  it("生成等待 exact revision 时若项目被编辑或删除所选片段则不提交旧点击", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "点击时版本";
    saveLocalTimeline(project);
    mockCommonRequests();
    let resolveUpdate!: (value: typeof project) => void;
    vi.mocked(directorApi.updateTimeline).mockImplementationOnce(
      () => new Promise((resolve) => { resolveUpdate = resolve; }),
    );
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "点击后的新版本" } });
    await user.click(screen.getByRole("button", { name: "删除所选" }));
    await act(async () => resolveUpdate(project));
    await waitFor(() => expect(screen.getByText(
      /时间线或分段选择在生成确认期间发生变化/,
    )).toBeInTheDocument());
    expect(submit).not.toHaveBeenCalled();
  });

  it("生成等待 exact revision 时若只改变分段选择也不提交旧集合", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    const first = { ...project.segments[0], prompt: "第一段" };
    const second = { ...createTimelineSegment("fl2va", 2), prompt: "第二段" };
    project.segments = [first, second];
    saveLocalTimeline(project);
    mockCommonRequests();
    let resolveUpdate!: (value: typeof project) => void;
    vi.mocked(directorApi.updateTimeline).mockImplementationOnce(
      () => new Promise((resolve) => { resolveUpdate = resolve; }),
    );
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "生成任务 2" }));
    expect(screen.getByRole("combobox", { name: "切换项目" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ }));
    await act(async () => resolveUpdate(project));

    await waitFor(() => expect(screen.getByText(
      /时间线或分段选择在生成确认期间发生变化/,
    )).toBeInTheDocument());
    expect(submit).not.toHaveBeenCalled();
  });

  it("生成意图持有点击时运行设置，任务响应前的新设置只排队不 PUT", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "执行设置边界";
    saveLocalTimeline(project);
    mockCommonRequests();
    const confirmed = structuredClone(CONFIGURED_SETTINGS);
    confirmed.models.fl2va.filename = "alternate-diffusion.safetensors";
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(confirmed));
    let resolveTimeline!: (value: typeof project) => void;
    vi.mocked(directorApi.updateTimeline).mockImplementationOnce(
      () => new Promise((resolve) => { resolveTimeline = resolve; }),
    );
    let resolveTask!: (value: GenerationTask) => void;
    const createTask = vi.spyOn(directorApi, "createTimelineTask").mockImplementation(
      () => new Promise((resolve) => { resolveTask = resolve; }),
    );
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(confirmed);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
    await openGlobalSettings(user);
    await user.selectOptions(
      screen.getByLabelText("FL2VA Diffusion 模型快捷选择"),
      "alternate-diffusion.safetensors",
    );
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateSettings).not.toHaveBeenCalled();

    await act(async () => resolveTimeline(project));
    await waitFor(() => expect(createTask).toHaveBeenCalledTimes(1));
    expect(updateSettings).not.toHaveBeenCalled();
    await act(async () => resolveTask(queuedTimelineTask));
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
  });

  it("生成内部随机 Seed 更新不被排队 endpoint 冻结，仍先按旧权威提交任务", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "随机种子执行边界";
    project.sampling.fl2va = { ...project.sampling.fl2va, seed: 101, random_seed: true };
    saveLocalTimeline(project);
    vi.spyOn(globalThis.crypto, "getRandomValues").mockImplementation(
      <T extends ArrayBufferView | null>(array: T): T => {
        if (array instanceof Uint32Array) { array[0] = 0; array[1] = 909; }
        return array;
      },
    );
    mockCommonRequests();
    const endpointB = { ...CONFIGURED_SETTINGS, comfy_url: "http://queued-during-seed.test:8188" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointB));
    let resolveTimeline!: (value: typeof project) => void;
    vi.mocked(directorApi.updateTimeline)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveTimeline = resolve; }))
      .mockImplementation(async (value) => value);
    let resolveTask!: (value: GenerationTask) => void;
    const createTask = vi.spyOn(directorApi, "createTimelineTask").mockImplementation(
      () => new Promise((resolve) => { resolveTask = resolve; }),
    );
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(endpointB);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointB.comfy_url);
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateSettings).not.toHaveBeenCalled();

    await act(async () => resolveTimeline(project));
    await waitFor(() => expect(createTask).toHaveBeenCalledTimes(1));
    expect(createTask).toHaveBeenCalledWith({
      config: expect.objectContaining({
        sampling: expect.objectContaining({ fl2va: expect.objectContaining({ seed: 909 }) }),
      }),
      segment_ids: [project.segments[0].id],
    });
    expect(updateSettings).not.toHaveBeenCalled();
    await act(async () => resolveTask(queuedTimelineTask));
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
  });

  it.each(["生成", "预检"] as const)("%s执行在途时素材移出不调用删除接口", async (kind) => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments = [{ ...createTimelineSegment("fl2va", 1), prompt: "执行素材边界", first_image: imageAsset }];
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.listAssets).mockResolvedValue({ assets: [imageAsset], outputs_preserved: true });
    let resolveTask!: (value: GenerationTask) => void;
    vi.spyOn(directorApi, "createTimelineTask").mockImplementation(
      () => new Promise((resolve) => { resolveTask = resolve; }),
    );
    let resolveCompile!: (value: TimelineCompileReport) => void;
    vi.spyOn(directorApi, "compileTimeline").mockImplementation(
      () => new Promise((resolve) => { resolveCompile = resolve; }),
    );
    const remove = vi.spyOn(directorApi, "deleteAsset");
    const cascade = vi.spyOn(directorApi, "deleteAssetCascade");
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(await screen.findByRole("listitem"));
    if (kind === "生成") {
      await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
      await waitFor(() => expect(directorApi.createTimelineTask).toHaveBeenCalledTimes(1));
    } else {
      await user.click(screen.getByRole("button", { name: "预检执行计划" }));
      await waitFor(() => expect(directorApi.compileTimeline).toHaveBeenCalledTimes(1));
    }
    expect(screen.getByRole("combobox", { name: "切换项目" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "移出素材库" }));
    expect(remove).not.toHaveBeenCalled();
    expect(cascade).not.toHaveBeenCalled();
    expect(screen.getByText(/生成或预检正在使用当前素材/)).toBeInTheDocument();

    if (kind === "生成") await act(async () => resolveTask(queuedTimelineTask));
    else await act(async () => resolveCompile({
      execution_strategy: "native_segment_graph_v1",
      model_families: ["fl2va"],
      plans: [],
      node_policy: { graph_source: "server", accepts_client_workflow: false, allowed_nodes: [], custom_nodes: [], provenance: {} },
    }));
  });

  it("预检响应前时间线与 endpoint 变化时丢弃旧报告，再自动同步设置", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "预检设置边界";
    saveLocalTimeline(project);
    mockCommonRequests();
    const endpointB = { ...CONFIGURED_SETTINGS, comfy_url: "http://comfy-after-compile.test:8188" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointB));
    const report: TimelineCompileReport = {
      execution_strategy: "native_segment_graph_v1",
      model_families: ["fl2va"],
      plans: [],
      node_policy: {
        graph_source: "server",
        accepts_client_workflow: false,
        allowed_nodes: [],
        custom_nodes: [],
        provenance: {},
      },
    };
    let resolveCompile!: (value: TimelineCompileReport) => void;
    const compile = vi.spyOn(directorApi, "compileTimeline").mockImplementation(
      () => new Promise((resolve) => { resolveCompile = resolve; }),
    );
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(endpointB);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "预检执行计划" }));
    await waitFor(() => expect(compile).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointB.comfy_url);
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "预检后的新版本" } });
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateSettings).not.toHaveBeenCalled();

    await act(async () => resolveCompile(report));
    expect(screen.queryByRole("region", { name: "服务端执行计划" })).not.toBeInTheDocument();
    expect(screen.getByText(/时间线、分段选择或运行设置已变化，请重新预检/)).toBeInTheDocument();
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
  });

  it("预检与生成使用同一显式分段集合，勾选变化会丢弃在途旧报告", async () => {
    const user = userEvent.setup();
    const first = { ...createTimelineSegment("fl2va", 1), prompt: "第一段" };
    const second = { ...createTimelineSegment("fl2va", 2), prompt: "第二段" };
    saveLocalTimeline({ ...createTimelineProject(), segments: [first, second] });
    mockCommonRequests();
    let resolveCompile!: (value: TimelineCompileReport) => void;
    const compile = vi.spyOn(directorApi, "compileTimeline").mockImplementation(
      () => new Promise((resolve) => { resolveCompile = resolve; }),
    );

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "预检执行计划" }));
    await waitFor(() => expect(compile).toHaveBeenCalledWith({
      config: expect.objectContaining({ segments: expect.any(Array) }),
      segment_ids: [first.id, second.id],
    }));

    await user.click(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ }));
    await act(async () => resolveCompile({
      execution_strategy: "native_segment_graph_v1",
      model_families: ["fl2va"],
      plans: [],
      node_policy: { graph_source: "server", accepts_client_workflow: false, allowed_nodes: [], custom_nodes: [], provenance: {} },
    }));
    expect(screen.queryByRole("region", { name: "服务端执行计划" })).not.toBeInTheDocument();
    expect(screen.getByText(/时间线、分段选择或运行设置已变化，请重新预检/)).toBeInTheDocument();
  });

  it("生成等待的旧 revision 失败但已被新编辑取代时，继续确认最新版再提交", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    let rejectFirst!: (reason: unknown) => void;
    const update = vi.mocked(directorApi.updateTimeline)
      .mockImplementationOnce(() => new Promise((_, reject) => { rejectFirst = reject; }))
      .mockImplementation(async (value) => value);
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);

    render(<App />);
    await waitUntilReady();
    const prompt = screen.getByLabelText("片段提示词");
    await user.clear(prompt);
    await user.type(prompt, "旧 revision");
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await user.type(prompt, "，最新版");
    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));

    await act(async () => rejectFirst(new ApiError("旧字段无效", 422)));
    await waitFor(() => expect(update.mock.calls.some(([value]) =>
      value.segments[0]?.prompt === "旧 revision，最新版",
    )).toBe(true));
    await waitFor(() => expect(submit).toHaveBeenCalledWith({
      config: expect.objectContaining({
        segments: [expect.objectContaining({ prompt: "旧 revision，最新版" })],
      }),
      segment_ids: [expect.any(String)],
    }));
  });

  it("随机 Seed 在提交前重掷为可见实际数值，且只更新所选片段的模型族", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "雨夜长镜头";
    project.sampling.fl2va = { ...project.sampling.fl2va, seed: 101, random_seed: true };
    project.sampling.ref2va = { ...project.sampling.ref2va, seed: 202, random_seed: true };
    saveLocalTimeline(project);
    vi.spyOn(globalThis.crypto, "getRandomValues").mockImplementation(
      <T extends ArrayBufferView | null>(array: T): T => {
        if (array instanceof Uint32Array) {
          array[0] = 0;
          array[1] = 303;
        }
        return array;
      },
    );
    mockCommonRequests();
    const update = vi.spyOn(directorApi, "updateTimeline").mockImplementation(async (value) => value);
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    update.mockClear();
    await openGlobalSettings(user);
    expect(screen.getByLabelText("FL2VA Seed")).toBeDisabled();
    expect(screen.getByLabelText("FL2VA Seed")).toHaveValue(101);
    await user.keyboard("{Escape}");

    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    expect(update.mock.invocationCallOrder[0]).toBeLessThan(submit.mock.invocationCallOrder[0]);
    expect(update.mock.calls[0][0].sampling.fl2va.seed).toBe(303);
    const submitted = submit.mock.calls[0][0].config!;
    expect(submitted.sampling.fl2va).toMatchObject({ seed: 303, random_seed: true });
    expect(submitted.sampling.ref2va).toMatchObject({ seed: 202, random_seed: true });

    await openGlobalSettings(user);
    expect(screen.getByLabelText("FL2VA Seed")).toBeDisabled();
    expect(screen.getByLabelText("FL2VA Seed")).toHaveValue(303);
    expect(screen.getByLabelText("Ref2VA Seed")).toHaveValue(202);
  });

  it("预检同样在提交前重掷随机 Seed，compile 报告携带实际值并同步灰显框", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "预检重掷";
    project.sampling.fl2va = { ...project.sampling.fl2va, seed: 101, random_seed: true };
    project.sampling.ref2va = { ...project.sampling.ref2va, seed: 202, random_seed: false };
    saveLocalTimeline(project);
    vi.spyOn(globalThis.crypto, "getRandomValues").mockImplementation(
      <T extends ArrayBufferView | null>(array: T): T => {
        if (array instanceof Uint32Array) {
          array[0] = 0;
          array[1] = 303;
        }
        return array;
      },
    );
    mockCommonRequests();
    const update = vi.spyOn(directorApi, "updateTimeline").mockImplementation(async (value) => value);
    const compile = vi.spyOn(directorApi, "compileTimeline").mockResolvedValue({
      execution_strategy: "native_segment_graph_v1",
      model_families: ["fl2va"],
      plans: [],
      node_policy: {
        graph_source: "server",
        accepts_client_workflow: false,
        allowed_nodes: [],
        custom_nodes: [],
        provenance: {},
      },
    });

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    update.mockClear();

    await user.click(screen.getByRole("button", { name: "预检执行计划" }));
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(compile).toHaveBeenCalledTimes(1));
    expect(update.mock.invocationCallOrder[0]).toBeLessThan(compile.mock.invocationCallOrder[0]);
    expect(update.mock.calls[0][0].sampling.fl2va.seed).toBe(303);
    const compiled = compile.mock.calls[0][0].config!;
    expect(compiled.sampling.fl2va).toMatchObject({ seed: 303, random_seed: true });
    expect(compiled.sampling.ref2va).toMatchObject({ seed: 202, random_seed: false });

    await openGlobalSettings(user);
    expect(screen.getByLabelText("FL2VA Seed")).toBeDisabled();
    expect(screen.getByLabelText("FL2VA Seed")).toHaveValue(303);
    expect(screen.getByLabelText("Ref2VA Seed")).toHaveValue(202);
  });

  it("唯一生成入口按项目导出规格提交稳定片段 ID", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const submit = vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);

    render(<App />);
    await waitUntilReady();
    await user.type(screen.getByLabelText("片段提示词"), "雾中森林");
    await openGlobalSettings(user);
    await user.selectOptions(screen.getByLabelText("导出方式"), "segments");
    await user.click(screen.getByRole("button", { name: "关闭全局设置" }));
    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    const firstCall = submit.mock.calls[0][0];
    expect(firstCall.config?.export_mode).toBe("segments");
    expect(firstCall.segment_ids).toEqual([expect.any(String)]);

    await openGlobalSettings(user);
    await user.selectOptions(screen.getByLabelText("导出方式"), "all");
    await user.click(screen.getByRole("button", { name: "关闭全局设置" }));
    await user.click(screen.getByRole("button", { name: /生成任务 1/ }));
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(2));
    expect(submit.mock.calls[1][0]).toEqual({
      config: expect.objectContaining({ export_mode: "all" }),
      segment_ids: firstCall.segment_ids,
    });
  });

  it("提交成功会使操作前的迟到任务列表失效并保留新父任务", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    let resolveStale!: (value: { jobs: GenerationTask[] }) => void;
    vi.mocked(directorApi.listTasks)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveStale = resolve; }))
      .mockResolvedValue({ jobs: [queuedTimelineTask] });
    vi.spyOn(directorApi, "createTimelineTask").mockResolvedValue(queuedTimelineTask);

    render(<App />);
    await waitUntilReady();
    await user.type(screen.getByLabelText("片段提示词"), "雾中森林");
    await user.click(screen.getByRole("button", { name: /生成任务/ }));
    await waitFor(() => expect(directorApi.createTimelineTask).toHaveBeenCalledTimes(1));
    expect(screen.getByRole("button", { name: "任务，1 个进行中" })).toBeInTheDocument();

    await act(async () => resolveStale({ jobs: [] }));
    await waitFor(() => expect(directorApi.listTasks).toHaveBeenCalledTimes(2));
    expect(screen.getByRole("button", { name: "任务，1 个进行中" })).toBeInTheDocument();
  });

  it("主题偏好独立保存，系统设置以浮层打开并关闭后回到时间线", async () => {
    const user = userEvent.setup();
    localStorage.setItem("director-web:theme", "light");
    mockCommonRequests();

    render(<App />);
    await waitUntilReady();
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    await user.click(screen.getByRole("button", { name: "切换到深色主题" }));
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    await user.click(screen.getByRole("button", { name: "全局设置" }));
    await user.click(screen.getByRole("button", { name: "任务，0 个进行中" }));
    const settingsToggle = screen.getByRole("button", { name: "系统设置" });
    await user.click(settingsToggle);
    expect(await screen.findByRole("dialog", { name: "系统设置" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "全局设置" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: "任务，0 个进行中" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("main", { name: "长视频时间线工作区" })).toBeInTheDocument();
    expect(document.querySelector(".workspace-surface")).toHaveAttribute("inert");
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    expect(screen.queryByRole("dialog", { name: "系统设置" })).not.toBeInTheDocument();
    expect(screen.getByRole("main", { name: "长视频时间线工作区" })).toBeInTheDocument();
    await waitFor(() => expect(settingsToggle).toHaveFocus());
  });

  it("系统设置入口可再次点击关闭，点侧栏其他位置也关闭且不抢焦点", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    render(<App />);
    await waitUntilReady();

    const settingsToggle = screen.getByRole("button", { name: "系统设置" });
    await user.click(settingsToggle);
    expect(await screen.findByRole("dialog", { name: "系统设置" })).toBeInTheDocument();
    expect(settingsToggle).toHaveAttribute("aria-expanded", "true");
    expect(settingsToggle.closest("[inert]")).toBeNull();
    expect(document.getElementById("director-sidebar-content")).toHaveAttribute("inert");

    await user.click(settingsToggle);
    expect(screen.queryByRole("dialog", { name: "系统设置" })).not.toBeInTheDocument();
    expect(settingsToggle).toHaveAttribute("aria-expanded", "false");

    await user.click(settingsToggle);
    expect(await screen.findByRole("dialog", { name: "系统设置" })).toBeInTheDocument();
    const sidebarToggle = screen.getByRole("button", { name: "素材库" });
    await user.click(sidebarToggle);
    expect(screen.queryByRole("dialog", { name: "系统设置" })).not.toBeInTheDocument();
    expect(sidebarToggle).toHaveFocus();
  });

  it("当前权威 ComfyUI 重启后测试连接会原子刷新 GPU、能力与模型资源", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    vi.mocked(directorApi.getGpus)
      .mockResolvedValueOnce([GPU_ZERO])
      .mockResolvedValue([GPU_ZERO, GPU_ONE]);
    vi.mocked(directorApi.getRayLightRuntimeStatus)
      .mockResolvedValueOnce({
        ...EMPTY_RAYLIGHT_RUNTIME_STATUS,
        available_gpu_indexes: [0],
      })
      .mockResolvedValue({
        ...EMPTY_RAYLIGHT_RUNTIME_STATUS,
        available_gpu_indexes: [0, 1],
      });
    vi.spyOn(directorApi, "testConnection").mockResolvedValue({
      ok: true,
      latency_ms: 3,
      message: "连接成功",
    });

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const gpuPanel = screen.getByRole("heading", { name: "GPU 状态" }).closest("section");
    if (!gpuPanel) throw new Error("GPU status panel missing");
    expect(within(gpuPanel).getByText("GPU 0")).toBeInTheDocument();
    expect(within(gpuPanel).queryByText("GPU 1")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "测试连接" }));

    expect(await screen.findByText("当前填写地址可连接")).toBeInTheDocument();
    await waitFor(() => expect(within(gpuPanel).getByText("GPU 1")).toBeInTheDocument());
    expect(directorApi.getCapabilities).toHaveBeenCalledTimes(2);
    expect(directorApi.getGpus).toHaveBeenCalledTimes(2);
    expect(directorApi.getModels).toHaveBeenCalledTimes(2);
    expect(directorApi.getRayLightRuntimeStatus).toHaveBeenCalledTimes(2);
  });

  it("直接资源刷新的 head 已是同 URL 新设置时先重建设置权威", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const externallyChanged = {
      ...CONFIGURED_SETTINGS,
      client_id: "changed-in-another-page",
    };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      // The external write completed before the connection-test refresh head.
      .mockResolvedValue(runtimeAuthority(externallyChanged));
    vi.spyOn(directorApi, "testConnection").mockResolvedValue({
      ok: true,
      message: "连接成功",
    });
    const updateSettings = vi.spyOn(directorApi, "updateSettings");

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    expect(screen.getByLabelText("客户端 ID")).toHaveValue(CONFIGURED_SETTINGS.client_id);
    await user.click(screen.getByRole("button", { name: "测试连接" }));

    await waitFor(() => expect(screen.getByLabelText("客户端 ID")).toHaveValue(
      externallyChanged.client_id,
    ));
    expect(updateSettings).not.toHaveBeenCalled();
    expect(directorApi.getCapabilities).toHaveBeenCalledTimes(3);
  });

  it("旧 RayLight GPU 状态阻断执行，且 POST 回包后仍等待权威状态 GET 才解锁", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "重启确认后建立新 RayLight 池";
    saveLocalTimeline(project);
    mockCommonRequests();
    const blockedStatus = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3, 4, 5, 6, 7],
      available_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [0, 1, 2, 3, 4, 5, 6, 7],
      tainted: false,
      recovery_token: "a".repeat(64),
    };
    const recoveredStatus = {
      ...blockedStatus,
      active: false,
      recovery_required: false,
      runtime_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [] as number[],
      recovery_token: null,
    };
    let resolveAuthoritativeStatus!: (status: typeof recoveredStatus) => void;
    vi.mocked(directorApi.getRayLightRuntimeStatus)
      .mockResolvedValueOnce(blockedStatus)
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveAuthoritativeStatus = resolve;
      }))
      .mockResolvedValue(recoveredStatus);
    const confirmRecovery = vi.spyOn(directorApi, "confirmRayLightRuntimeRecovery")
      .mockResolvedValue(recoveredStatus);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    expect(await screen.findByText(/旧 RayLight 运行状态引用了当前不可见 GPU；请打开系统设置/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /生成任务 1/ })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const recover = await screen.findByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    });
    fireEvent.click(recover);
    fireEvent.click(recover);
    await waitFor(() => expect(confirmRecovery).toHaveBeenCalledWith(
      "http://comfy.test:8188",
      36,
      "a".repeat(64),
      expect.any(AbortSignal),
    ));
    expect(confirmRecovery).toHaveBeenCalledTimes(1);

    // The mutation response is intentionally ignored. Until a fresh same-origin
    // status GET completes, the warning and execution lock must remain.
    expect(screen.getByRole("alert", { name: "旧 RayLight 运行状态引用了当前不可见 GPU" })).toBeInTheDocument();
    expect(recover).toBeDisabled();
    expect(recover).toHaveAttribute("aria-busy", "true");
    expect(await screen.findByRole("button", { name: "保存路径（重启后切换）" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "迁移当前数据库并切换" })).toBeDisabled();

    await act(async () => resolveAuthoritativeStatus(recoveredStatus));
    await waitFor(() => expect(screen.queryByRole("alert", {
      name: "旧 RayLight 运行状态引用了当前不可见 GPU",
    })).not.toBeInTheDocument());
    expect(await screen.findByText(/已确认 ComfyUI 重启并恢复 RayLight/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "预检执行计划" })).toBeEnabled());
    expect(screen.getByRole("button", { name: /生成任务 1/ })).toBeEnabled();
  });

  it("首次 RayLight 状态读取失败时保持预检与生成关闭", async () => {
    const project = createTimelineProject();
    project.segments[0].prompt = "状态读取失败不得生成";
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.getRayLightRuntimeStatus).mockRejectedValue(
      new Error("runtime status unavailable"),
    );

    render(<App />);
    await waitFor(() => expect(directorApi.getRayLightRuntimeStatus).toHaveBeenCalled());
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /生成任务 1/ })).toBeDisabled();
  });

  it("四资源部分失败时不展示未形成同源权威的 RayLight 告警", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    vi.mocked(directorApi.getModels).mockRejectedValue(new Error("models unavailable"));
    vi.mocked(directorApi.getRayLightRuntimeStatus).mockResolvedValue({
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3],
      available_gpu_indexes: [],
      invalid_gpu_indexes: [0, 1, 2, 3],
      tainted: true,
      recovery_token: "b".repeat(64),
    });

    render(<App />);
    await waitUntilTimelineReady();
    await waitFor(() => expect(directorApi.getModels).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    expect(await screen.findByRole("dialog", { name: "系统设置" })).toBeInTheDocument();
    expect(screen.queryByRole("alert", {
      name: "旧 RayLight 运行状态引用了当前不可见 GPU",
    })).not.toBeInTheDocument();
  });

  it("恢复 POST 丢失响应且首次 GET 仍为旧状态时会用同一证书重试到权威 clean", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "恢复重试后生成";
    saveLocalTimeline(project);
    mockCommonRequests();
    const blockedStatus = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3],
      available_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [0, 1, 2, 3],
      tainted: true,
      recovery_token: "c".repeat(64),
    };
    const recoveredStatus = {
      ...blockedStatus,
      active: false,
      recovery_required: false,
      runtime_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [] as number[],
      tainted: false,
      recovery_token: null,
    };
    vi.mocked(directorApi.getRayLightRuntimeStatus)
      .mockResolvedValueOnce(blockedStatus)
      .mockResolvedValueOnce(blockedStatus)
      .mockResolvedValue(recoveredStatus);
    const confirmRecovery = vi.spyOn(directorApi, "confirmRayLightRuntimeRecovery")
      .mockRejectedValueOnce(new TypeError("response lost"))
      .mockResolvedValue(recoveredStatus);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    expect(await screen.findByText(/旧 RayLight 运行状态引用了当前不可见 GPU；请打开系统设置/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    await user.click(await screen.findByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    }));

    await waitFor(() => expect(confirmRecovery).toHaveBeenCalledTimes(2), { timeout: 2_000 });
    expect(confirmRecovery.mock.calls).toEqual([
      ["http://comfy.test:8188", 36, "c".repeat(64), expect.any(AbortSignal)],
      ["http://comfy.test:8188", 36, "c".repeat(64), expect.any(AbortSignal)],
    ]);
    await waitFor(() => expect(screen.queryByRole("alert", {
      name: "旧 RayLight 运行状态引用了当前不可见 GPU",
    })).not.toBeInTheDocument());
    expect(screen.queryByText(/RayLight 恢复失败/)).not.toBeInTheDocument();
    expect(await screen.findByText(/已确认 ComfyUI 重启并恢复 RayLight/)).toBeInTheDocument();
  });

  it("恢复 POST 丢响应后不被瞬时 409 与旧 blocked GET 提前解除锁", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "恢复竞态收敛后生成";
    saveLocalTimeline(project);
    mockCommonRequests();
    const blockedStatus = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3],
      available_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [0, 1, 2, 3],
      tainted: true,
      recovery_token: "d".repeat(64),
    };
    const recoveredStatus = {
      ...blockedStatus,
      active: false,
      recovery_required: false,
      runtime_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [] as number[],
      tainted: false,
      recovery_token: null,
    };
    vi.mocked(directorApi.getRayLightRuntimeStatus)
      .mockResolvedValueOnce(blockedStatus)
      .mockResolvedValueOnce(blockedStatus)
      .mockResolvedValueOnce(blockedStatus)
      .mockResolvedValue(recoveredStatus);
    let resolveConvergedRetry!: (status: typeof recoveredStatus) => void;
    const confirmRecovery = vi.spyOn(directorApi, "confirmRayLightRuntimeRecovery")
      .mockRejectedValueOnce(new TypeError("response lost"))
      .mockRejectedValueOnce(new ApiError(
        "endpoint submission lock is busy",
        409,
        { detail: { code: "raylight_recovery_in_flight", message: "endpoint submission lock is busy" } },
        "raylight_recovery_in_flight",
      ))
      .mockImplementationOnce(() => new Promise((resolve) => {
        resolveConvergedRetry = resolve;
      }))
      .mockResolvedValue(recoveredStatus);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const recover = await screen.findByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    });
    await user.click(recover);

    // The third serial POST proves that the preceding 409 + still-blocked
    // authority read did not terminate the ambiguous operation.
    await waitFor(() => expect(confirmRecovery).toHaveBeenCalledTimes(3), { timeout: 2_500 });
    expect(confirmRecovery.mock.calls.slice(0, 3)).toEqual([
      ["http://comfy.test:8188", 36, "d".repeat(64), expect.any(AbortSignal)],
      ["http://comfy.test:8188", 36, "d".repeat(64), expect.any(AbortSignal)],
      ["http://comfy.test:8188", 36, "d".repeat(64), expect.any(AbortSignal)],
    ]);
    expect(recover).toBeDisabled();
    expect(recover).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("button", { name: "保存路径（重启后切换）" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "迁移当前数据库并切换" })).toBeDisabled();
    expect(screen.queryByText(/RayLight 恢复失败/)).not.toBeInTheDocument();

    await act(async () => resolveConvergedRetry(recoveredStatus));
    await waitFor(() => expect(screen.queryByRole("alert", {
      name: "旧 RayLight 运行状态引用了当前不可见 GPU",
    })).not.toBeInTheDocument());
    expect(await screen.findByText(/已确认 ComfyUI 重启并恢复 RayLight/)).toBeInTheDocument();
  });

  it("ambiguous 恢复后的非 in-flight 409 仍作为确定失败退出", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const blockedStatus = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3],
      available_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [0, 1, 2, 3],
      tainted: true,
      recovery_token: "e".repeat(64),
    };
    vi.mocked(directorApi.getRayLightRuntimeStatus).mockResolvedValue(blockedStatus);
    const confirmRecovery = vi.spyOn(directorApi, "confirmRayLightRuntimeRecovery")
      .mockRejectedValueOnce(new TypeError("response lost"))
      .mockRejectedValue(new ApiError("recorded runtime tail changed", 409));
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const recover = await screen.findByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    });
    await user.click(recover);

    expect(await screen.findByText(/RayLight 恢复失败：recorded runtime tail changed/)).toBeInTheDocument();
    expect(confirmRecovery).toHaveBeenCalledTimes(2);
    expect(recover).toBeEnabled();
    expect(recover).not.toHaveAttribute("aria-busy");
    expect(screen.getByRole("button", { name: "保存路径（重启后切换）" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "迁移当前数据库并切换" })).toBeEnabled();
  });

  it("恢复期间服务器 endpoint 外部切换时不把新 endpoint 的 clean 状态标成旧源", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "外部切换核对";
    saveLocalTimeline(project);
    mockCommonRequests();
    const endpointB = { ...CONFIGURED_SETTINGS, comfy_url: "http://external-b.test:8188" };
    const blockedStatus = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3],
      available_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [0, 1, 2, 3],
      tainted: true,
      recovery_token: "f".repeat(64),
    };
    const endpointBClean = {
      ...EMPTY_RAYLIGHT_RUNTIME_STATUS,
      available_gpu_indexes: [] as number[],
    };
    const authorityA = {
      settings: CONFIGURED_SETTINGS,
      authority_token: "a".repeat(64),
    };
    const authorityB = {
      settings: endpointB,
      authority_token: "b".repeat(64),
    };
    vi.mocked(directorApi.getSettingsAuthority)
      // Initial App resource snapshot (head + tail), then the recovery head.
      .mockResolvedValueOnce(authorityA)
      .mockResolvedValueOnce(authorityA)
      .mockResolvedValueOnce(authorityA)
      // Recovery tail observes B. The App must discard the whole batch and
      // let external-authority reconciliation adopt B as a fresh snapshot.
      .mockResolvedValue(authorityB);
    vi.mocked(directorApi.getRayLightRuntimeStatus)
      .mockResolvedValueOnce(blockedStatus)
      .mockResolvedValue(endpointBClean);
    const confirmRecovery = vi.spyOn(directorApi, "confirmRayLightRuntimeRecovery")
      .mockRejectedValue(new ApiError("ComfyUI endpoint changed", 409));
    const updateSettings = vi.spyOn(directorApi, "updateSettings")
      .mockImplementation(async (settings) => settings);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    await user.click(await screen.findByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    }));

    await waitFor(() => expect(confirmRecovery).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByLabelText("ComfyUI 地址")).toHaveValue(endpointB.comfy_url));
    expect(screen.queryByText(/已确认 ComfyUI 重启并恢复 RayLight/)).not.toBeInTheDocument();
    expect(confirmRecovery).toHaveBeenCalledTimes(1);
    expect(updateSettings).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "预检执行计划" })).toBeEnabled());
  });

  it("恢复四资源批次遇到同 URL 的 authority ABA 时丢弃 clean 快照", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const blockedStatus = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3],
      available_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [0, 1, 2, 3],
      tainted: true,
      recovery_token: "9".repeat(64),
    };
    const cleanStatus = {
      ...EMPTY_RAYLIGHT_RUNTIME_STATUS,
      available_gpu_indexes: [] as number[],
    };
    const oldAuthority = {
      settings: CONFIGURED_SETTINGS,
      authority_token: "a".repeat(64),
    };
    const newAuthority = {
      settings: CONFIGURED_SETTINGS,
      authority_token: "b".repeat(64),
    };
    vi.mocked(directorApi.getSettingsAuthority)
      // Initial App snapshot, then its tail.
      .mockResolvedValueOnce(oldAuthority)
      .mockResolvedValueOnce(oldAuthority)
      // Recovery head still sees A; the tail observes A -> B -> A's new
      // monotonic revision even though the canonical URL is unchanged.
      .mockResolvedValueOnce(oldAuthority)
      .mockResolvedValue(newAuthority);
    vi.mocked(directorApi.getRayLightRuntimeStatus)
      .mockResolvedValueOnce(blockedStatus)
      .mockResolvedValue(cleanStatus);
    const confirmRecovery = vi.spyOn(directorApi, "confirmRayLightRuntimeRecovery")
      .mockRejectedValue(new ApiError("runtime authority changed", 409));
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    await user.click(await screen.findByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    }));

    await waitFor(() => expect(confirmRecovery).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(directorApi.getCapabilities).toHaveBeenCalledTimes(3));
    expect(screen.queryByText(/已确认 ComfyUI 重启并恢复 RayLight/)).not.toBeInTheDocument();
    expect(directorApi.getSettingsAuthority).toHaveBeenCalledTimes(6);
  });

  it("App unmount 会中止外部运行权威刷新且不再调度 GET", async () => {
    mockCommonRequests();
    const oldAuthority = {
      settings: CONFIGURED_SETTINGS,
      authority_token: "c".repeat(64),
    };
    const newAuthority = {
      settings: CONFIGURED_SETTINGS,
      authority_token: "d".repeat(64),
    };
    let externalSignal: AbortSignal | undefined;
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(oldAuthority)
      // Initial resource tail changes token and schedules the external owner.
      .mockResolvedValueOnce(newAuthority)
      .mockImplementationOnce((signal) => new Promise((_resolve, reject) => {
        externalSignal = signal;
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      }))
      .mockResolvedValue(newAuthority);

    const view = render(<App />);
    await waitFor(() => expect(externalSignal).toBeInstanceOf(AbortSignal));
    expect(externalSignal?.aborted).toBe(false);

    view.unmount();
    expect(externalSignal?.aborted).toBe(true);
    await new Promise((resolve) => window.setTimeout(resolve, 50));
    expect(directorApi.getSettingsAuthority).toHaveBeenCalledTimes(3);
  });

  it("App unmount 会中止恢复 owner 的在途请求与退避重试", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const blockedStatus = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3],
      available_gpu_indexes: [] as number[],
      invalid_gpu_indexes: [0, 1, 2, 3],
      tainted: true,
      recovery_token: "1".repeat(64),
    };
    vi.mocked(directorApi.getRayLightRuntimeStatus).mockResolvedValue(blockedStatus);
    const confirmRecovery = vi.spyOn(directorApi, "confirmRayLightRuntimeRecovery")
      .mockRejectedValue(new TypeError("response lost"));
    vi.spyOn(window, "confirm").mockReturnValue(true);

    const view = render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    await user.click(await screen.findByRole("button", {
      name: "确认 ComfyUI 已重启并恢复 RayLight",
    }));
    await waitFor(() => expect(directorApi.getRayLightRuntimeStatus).toHaveBeenCalledTimes(2));
    expect(confirmRecovery).toHaveBeenCalledTimes(1);
    const signal = confirmRecovery.mock.calls[0][3];
    expect(signal).toBeInstanceOf(AbortSignal);
    expect(signal?.aborted).toBe(false);

    view.unmount();
    expect(signal?.aborted).toBe(true);
    await new Promise((resolve) => window.setTimeout(resolve, 400));
    expect(confirmRecovery).toHaveBeenCalledTimes(1);
    expect(directorApi.getRayLightRuntimeStatus).toHaveBeenCalledTimes(2);
  });

  it("当前地址资源刷新部分失败时保留同源旧快照并锁定执行，重试完整成功后再更新", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "重启后重新核对资源";
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.getGpus)
      .mockResolvedValueOnce([GPU_ZERO])
      .mockResolvedValue([GPU_ZERO, GPU_ONE]);
    vi.mocked(directorApi.getRayLightRuntimeStatus)
      .mockResolvedValueOnce({
        ...EMPTY_RAYLIGHT_RUNTIME_STATUS,
        available_gpu_indexes: [0],
      })
      .mockResolvedValue({
        ...EMPTY_RAYLIGHT_RUNTIME_STATUS,
        available_gpu_indexes: [0, 1],
      });
    vi.mocked(directorApi.getModels)
      .mockResolvedValueOnce(MODEL_INVENTORY)
      .mockRejectedValueOnce(new Error("重启后的模型清单暂不可用"))
      .mockResolvedValue(MODEL_INVENTORY);
    vi.spyOn(directorApi, "testConnection").mockResolvedValue({
      ok: true,
      message: "连接成功",
    });
    const updateSettings = vi.spyOn(directorApi, "updateSettings")
      .mockImplementation(async (settings) => settings);

    render(<App />);
    await waitUntilReady();
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeEnabled();
    expect(screen.getByRole("button", { name: /生成任务 1/ })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const gpuPanel = screen.getByRole("heading", { name: "GPU 状态" }).closest("section");
    if (!gpuPanel) throw new Error("GPU status panel missing");
    await user.click(screen.getByRole("button", { name: "测试连接" }));

    await waitFor(() => expect(directorApi.getModels).toHaveBeenCalledTimes(2));
    expect(within(gpuPanel).getByText("A6000-0")).toBeInTheDocument();
    expect(within(gpuPanel).queryByText("A6000-1")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /生成任务 1/ })).toBeDisabled();
    expect(updateSettings).not.toHaveBeenCalled();

    await waitFor(() => expect(directorApi.getModels).toHaveBeenCalledTimes(3), { timeout: 3_000 });
    await waitFor(() => expect(within(gpuPanel).getByText("A6000-1")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeEnabled();
    expect(screen.getByRole("button", { name: /生成任务 1/ })).toBeEnabled();
    expect(updateSettings).not.toHaveBeenCalled();
  }, 5_000);

  it("临时不同 URL 的成功测试不刷新或污染当前权威资源", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    vi.mocked(directorApi.getGpus).mockResolvedValue([GPU_ZERO]);
    vi.mocked(directorApi.getRayLightRuntimeStatus).mockResolvedValue({
      ...EMPTY_RAYLIGHT_RUNTIME_STATUS,
      available_gpu_indexes: [0],
    });
    vi.spyOn(directorApi, "testConnection").mockResolvedValue({
      ok: true,
      message: "临时地址连接成功",
    });

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    await user.clear(screen.getByLabelText("客户端 ID"));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, "http://temporary-comfy.test:8188");
    await user.click(screen.getByRole("button", { name: "测试连接" }));

    expect(await screen.findByText("当前填写地址可连接")).toBeInTheDocument();
    expect(directorApi.testConnection).toHaveBeenCalledWith("http://temporary-comfy.test:8188");
    expect(directorApi.getCapabilities).toHaveBeenCalledTimes(1);
    expect(directorApi.getGpus).toHaveBeenCalledTimes(1);
    expect(directorApi.getModels).toHaveBeenCalledTimes(1);
    const gpuPanel = screen.getByRole("heading", { name: "GPU 状态" }).closest("section");
    if (!gpuPanel) throw new Error("GPU status panel missing");
    expect(within(gpuPanel).getByText("A6000-0")).toBeInTheDocument();
    expect(within(gpuPanel).queryByText("A6000-1")).not.toBeInTheDocument();
  });

  it("系统设置自动同步由 App 持有，关闭浮层后仍完成 endpoint 权威切换", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "雨夜长镜头";
    saveLocalTimeline(project);
    mockCommonRequests();
    const endpointBSettings = {
      ...CONFIGURED_SETTINGS,
      comfy_url: "http://comfy-b.test:8188",
    };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointBSettings));
    vi.mocked(directorApi.listAssets)
      .mockResolvedValueOnce({ assets: [imageAsset], outputs_preserved: true })
      .mockResolvedValue({ assets: [endpointBImageAsset], outputs_preserved: true });
    let resolvePut!: (settings: typeof endpointBSettings) => void;
    const update = vi.spyOn(directorApi, "updateSettings").mockImplementation(
      () => new Promise((resolve) => { resolvePut = resolve; }),
    );

    render(<App />);
    await waitUntilReady();
    expect(await screen.findByText(imageAsset.name)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointBSettings.comfy_url);
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));

    const navigation = screen.getByRole("button", { name: "系统设置" });
    expect(navigation).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    expect(screen.queryByRole("dialog", { name: "系统设置" })).not.toBeInTheDocument();

    await act(async () => resolvePut(endpointBSettings));
    await waitFor(() => expect(directorApi.listAssets).toHaveBeenCalledTimes(2));
    expect(navigation).toBeEnabled();
    expect(await screen.findByText(endpointBImageAsset.name)).toBeInTheDocument();
    expect(screen.queryByText(imageAsset.name)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeEnabled();
  });

  it("系统设置连续输入只提交最终快照且 PUT 严格单在途", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const firstConfirmed = { ...CONFIGURED_SETTINGS, client_id: "first-version" };
    const finalConfirmed = { ...CONFIGURED_SETTINGS, client_id: "final-version" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(firstConfirmed))
      .mockResolvedValueOnce(runtimeAuthority(firstConfirmed))
      .mockResolvedValue(runtimeAuthority(finalConfirmed));
    let resolveFirst!: (value: typeof firstConfirmed) => void;
    const update = vi.spyOn(directorApi, "updateSettings")
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = resolve; }))
      .mockImplementation(async (value) => value);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, firstConfirmed.client_id);
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await user.clear(clientId);
    await user.type(clientId, finalConfirmed.client_id);
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(update).toHaveBeenCalledTimes(1);

    await act(async () => resolveFirst(firstConfirmed));
    await waitFor(() => expect(update).toHaveBeenCalledTimes(2));
    expect(update).toHaveBeenLastCalledWith(expect.objectContaining({ client_id: "final-version" }));
  });

  it("系统设置确定性 422 不受其他 autosave kick 重试，下一次有效编辑才恢复", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const update = vi.spyOn(directorApi, "updateSettings")
      .mockRejectedValueOnce(new ApiError("客户端 ID 无效", 422))
      .mockImplementation(async (value) => value);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, "first-valid");
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => window.setTimeout(resolve, 1650));
    expect(update).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "触发时间线自动同步" } });
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalledTimes(1));
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(update).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const reopenedClientId = screen.getByLabelText("客户端 ID");
    await user.clear(reopenedClientId);
    await user.type(reopenedClientId, "fixed-valid");
    await waitFor(() => expect(update).toHaveBeenCalledTimes(2));
  });

  it("本地素材上传期间排队 endpoint 切换时丢弃旧源结果并继续最新设置", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments = [{ ...createTimelineSegment("ref2va", 1), id: project.segments[0].id }];
    saveLocalTimeline(project);
    mockCommonRequests();
    const uploadedVideo: AssetReference = {
      id: "external-video-app",
      name: "外部源视频.mp4",
      subfolder: "director",
      type: "input",
      kind: "video",
      preview_url: "/api/assets/external-video-app/preview",
      metadata: {
        duration: 5,
        native_fps: 24,
        frame_count: 120,
        width: 864,
        height: 480,
        probe_method: "ffprobe",
        has_audio: true,
      },
    };
    let resolveUpload!: (asset: AssetReference) => void;
    const upload = vi.spyOn(directorApi, "uploadAsset").mockImplementation(
      () => new Promise((resolve) => { resolveUpload = resolve; }),
    );
    const update = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(CONFIGURED_SETTINGS);

    render(<App />);
    await waitUntilReady();
    const file = new File(["video"], "外部源视频.mp4", { type: "video/mp4" });
    fireEvent.drop(screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" }), {
      dataTransfer: {
        types: ["Files"],
        files: [file],
        getData: vi.fn(() => ""),
        setData: vi.fn(),
        dropEffect: "copy",
        effectAllowed: "copy",
      },
    });
    await waitFor(() => expect(upload).toHaveBeenCalledWith(file, "video", expect.any(Function)));
    expect(screen.getByText(/完成前暂时锁定同步、预检和生成/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "保存时间线" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, "http://comfy-other.test:8188");
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(update).not.toHaveBeenCalled();

    await act(async () => resolveUpload(uploadedVideo));
    await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
    expect(within(screen.getByRole("region", { name: "源视频（可选），占用 <Video 1>" })).queryByText(/外部源视频\.mp4/)).not.toBeInTheDocument();
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
  });

  it("endpoint 切换先等待在途时间线确认，并在边界内拒绝新的项目编辑", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const endpointB = { ...CONFIGURED_SETTINGS, comfy_url: "http://comfy-boundary.test:8188" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointB));
    let resolveTimeline!: (project: ReturnType<typeof createTimelineProject>) => void;
    const updateTimeline = vi.mocked(directorApi.updateTimeline)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveTimeline = resolve; }))
      .mockImplementation(async (project) => project);
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(endpointB);

    render(<App />);
    await waitUntilReady();
    const prompt = screen.getByLabelText("片段提示词");
    fireEvent.change(prompt, { target: { value: "切换前的精确版本" } });
    await waitFor(() => expect(updateTimeline).toHaveBeenCalledTimes(1));
    const exactProject = updateTimeline.mock.calls[0][0];

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointB.comfy_url);
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateSettings).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "不应越过边界" } });
    expect(screen.getByLabelText("片段提示词")).toHaveValue("切换前的精确版本");
    expect(screen.getByText(/正在切换 ComfyUI 地址/)).toBeInTheDocument();

    await act(async () => resolveTimeline(exactProject));
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
  });

  it("endpoint 边界等待期间的无效新草稿取消旧目标并恢复时间线编辑", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    let resolveTimeline!: (project: ReturnType<typeof createTimelineProject>) => void;
    const updateTimeline = vi.mocked(directorApi.updateTimeline)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveTimeline = resolve; }))
      .mockImplementation(async (project) => project);
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(CONFIGURED_SETTINGS);

    render(<App />);
    await waitUntilReady();
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "边界基线" } });
    await waitFor(() => expect(updateTimeline).toHaveBeenCalledTimes(1));
    const exactProject = updateTimeline.mock.calls[0][0];
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, "http://queued-b.test:8188");
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateSettings).not.toHaveBeenCalled();
    await user.clear(endpoint);
    await user.type(endpoint, "invalid-c");

    await act(async () => resolveTimeline(exactProject));
    await new Promise((resolve) => window.setTimeout(resolve, 50));
    expect(updateSettings).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "无效设置下仍可修时间线" } });
    expect(screen.getByLabelText("片段提示词")).toHaveValue("无效设置下仍可修时间线");
  });

  it("仅输入无效 endpoint 不冻结时间线编辑", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const updateSettings = vi.spyOn(directorApi, "updateSettings");

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, "invalid-endpoint");
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "时间线仍可编辑" } });
    expect(screen.getByLabelText("片段提示词")).toHaveValue("时间线仍可编辑");
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("endpoint 切换遇到时间线 422 时不提交设置，修正时间线后自动继续", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const endpointB = { ...CONFIGURED_SETTINGS, comfy_url: "http://comfy-after-fix.test:8188" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointB));
    let rejectTimeline!: (reason: unknown) => void;
    const updateTimeline = vi.mocked(directorApi.updateTimeline)
      .mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectTimeline = reject; }))
      .mockImplementation(async (project) => project);
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(endpointB);

    render(<App />);
    await waitUntilReady();
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "服务端拒绝的版本" } });
    await waitFor(() => expect(updateTimeline).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointB.comfy_url);
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateSettings).not.toHaveBeenCalled();

    await act(async () => rejectTimeline(new ApiError("提示词字段无效", 422)));
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    expect(await screen.findByText(/服务器拒绝当前时间线.*提示词字段无效/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "修正后的版本" } });
    expect(screen.getByLabelText("片段提示词")).toHaveValue("修正后的版本");
    await waitFor(() => expect(updateTimeline).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
  });

  it("endpoint PUT 422 且回读仍为旧地址时解除时间线 gate 但保持生成锁", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValue(runtimeAuthority(CONFIGURED_SETTINGS));
    const updateSettings = vi.spyOn(directorApi, "updateSettings")
      .mockRejectedValue(new ApiError("新地址被服务器拒绝", 422));

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, "http://rejected-endpoint.test:8188");
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    expect(await screen.findByText(/服务器拒绝当前系统设置.*新地址被服务器拒绝/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "拒绝后仍可编辑时间线" } });
    expect(screen.getByLabelText("片段提示词")).toHaveValue("拒绝后仍可编辑时间线");
  });

  it("endpoint PUT 后出现无效新草稿，资源重试成功会解除时间线 gate", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const endpointB = { ...CONFIGURED_SETTINGS, comfy_url: "http://confirmed-b.test:8188" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointB));
    vi.mocked(directorApi.getCapabilities)
      .mockResolvedValueOnce(ONLINE_CAPABILITIES)
      .mockResolvedValueOnce({ ...ONLINE_CAPABILITIES, connection: "offline" })
      .mockResolvedValue(ONLINE_CAPABILITIES);
    vi.mocked(directorApi.getModels)
      .mockResolvedValueOnce(MODEL_INVENTORY)
      .mockRejectedValueOnce(new Error("B model inventory pending"))
      .mockResolvedValue(MODEL_INVENTORY);
    let resolvePut!: (value: typeof endpointB) => void;
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockImplementation(
      () => new Promise((resolve) => { resolvePut = resolve; }),
    );

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointB.comfy_url);
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    await user.clear(endpoint);
    await user.type(endpoint, "invalid-c");
    await act(async () => resolvePut(endpointB));
    await waitFor(() => expect(directorApi.getCapabilities).toHaveBeenCalledTimes(3), { timeout: 3_500 });

    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    expect(screen.getByText(/系统设置有无效输入/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("片段提示词"), { target: { value: "资源恢复后仍可编辑" } });
    expect(screen.getByLabelText("片段提示词")).toHaveValue("资源恢复后仍可编辑");
  });

  it.each(["完成", "失败"] as const)("侧栏素材上传%s后自动续写上传期间的时间线编辑", async (outcome) => {
    mockCommonRequests();
    let resolveUpload!: (asset: AssetReference) => void;
    let rejectUpload!: (reason: unknown) => void;
    const upload = vi.spyOn(directorApi, "uploadAsset").mockImplementation(
      () => new Promise((resolve, reject) => {
        resolveUpload = resolve;
        rejectUpload = reject;
      }),
    );
    const update = vi.mocked(directorApi.updateTimeline);

    render(<App />);
    await waitUntilReady();
    const file = new File(["image"], "上传锁测试.png", { type: "image/png" });
    const input = document.querySelector<HTMLInputElement>(
      '.asset-sidebar__upload input[type="file"]',
    );
    expect(input).not.toBeNull();
    fireEvent.change(input!, { target: { files: [file] } });
    await waitFor(() => expect(upload).toHaveBeenCalledWith(file, "image", expect.any(Function)));

    fireEvent.change(screen.getByLabelText("片段提示词"), {
      target: { value: `上传${outcome}期间的编辑` },
    });
    await new Promise((resolve) => window.setTimeout(resolve, 250));
    expect(update).not.toHaveBeenCalled();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).not.toBeNull();

    if (outcome === "完成") {
      await act(async () => resolveUpload({ ...imageAsset, id: "upload-lock-image", name: file.name }));
    } else {
      await act(async () => rejectUpload(new Error("模拟上传失败")));
    }
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    expect(update).toHaveBeenCalledWith(expect.objectContaining({
      segments: [expect.objectContaining({ prompt: `上传${outcome}期间的编辑` })],
    }));
    await waitFor(() => expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull());
  });

  it("工作区快捷模型自动同步期间仍可打开设置，但阻断素材删除、compile 和生成", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "雾中推进";
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.listAssets).mockResolvedValue({ assets: [imageAsset], outputs_preserved: true });
    const confirmed = structuredClone(CONFIGURED_SETTINGS);
    confirmed.models.ref2va.filename = "alternate-diffusion.safetensors";
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(confirmed));
    let resolvePut!: (settings: typeof confirmed) => void;
    const update = vi.spyOn(directorApi, "updateSettings").mockImplementation(
      () => new Promise((resolve) => { resolvePut = resolve; }),
    );
    const compile = vi.spyOn(directorApi, "compileTimeline");
    const submit = vi.spyOn(directorApi, "createTimelineTask");
    const remove = vi.spyOn(directorApi, "deleteAsset");

    render(<App />);
    await waitUntilReady();
    await user.click(await screen.findByRole("listitem"));
    await openGlobalSettings(user);
    await user.selectOptions(
      screen.getByLabelText("REF2VA Diffusion 模型快捷选择"),
      "alternate-diffusion.safetensors",
    );
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));

    expect(screen.getByRole("button", { name: "系统设置" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "移出素材库" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "预检执行计划" }));
    fireEvent.click(screen.getByRole("button", { name: /生成任务/ }));
    fireEvent.click(screen.getByRole("button", { name: "移出素材库" }));
    expect(compile).not.toHaveBeenCalled();
    expect(submit).not.toHaveBeenCalled();
    expect(remove).not.toHaveBeenCalled();

    await act(async () => resolvePut(confirmed));
    await waitFor(
      () => expect(screen.getByRole("button", { name: /生成任务/ })).toBeEnabled(),
      { timeout: 10_000 },
    );
    expect(update).toHaveBeenCalledWith(expect.objectContaining({
      models: expect.objectContaining({
        ref2va: expect.objectContaining({ filename: "alternate-diffusion.safetensors" }),
      }),
    }));
  });

  it("endpoint 权威 GET 成功但素材刷新失败时清空旧源并保持生成锁", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "海边远景";
    saveLocalTimeline(project);
    mockCommonRequests();
    const endpointBSettings = {
      ...CONFIGURED_SETTINGS,
      comfy_url: "http://comfy-b.test:8188",
    };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointBSettings));
    vi.mocked(directorApi.listAssets)
      .mockResolvedValueOnce({ assets: [imageAsset], outputs_preserved: true })
      .mockRejectedValue(new Error("B 素材接口离线"));
    vi.spyOn(directorApi, "updateSettings").mockResolvedValue(endpointBSettings);

    render(<App />);
    await waitUntilReady();
    expect(await screen.findByText(imageAsset.name)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointBSettings.comfy_url);
    expect(await screen.findByText(/新 ComfyUI 对应的素材库无法权威刷新/)).toBeInTheDocument();

    const navigation = screen.getByRole("button", { name: "系统设置" });
    expect(navigation).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    expect(screen.queryByText(imageAsset.name)).not.toBeInTheDocument();
    expect(screen.getByText(/运行设置或素材库正在后台自动核对/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /生成任务/ })).toBeDisabled();
    expect(screen.getByRole("button", { name: "预检执行计划" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /同步运行设置/ })).not.toBeInTheDocument();
  });

  it("PUT 丢失响应但权威 GET 已确认目标设置时按成功收敛", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    const confirmed = {
      ...CONFIGURED_SETTINGS,
      client_id: "committed-despite-lost-response",
    };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(confirmed));
    vi.spyOn(directorApi, "updateSettings").mockRejectedValue(new Error("PUT response lost"));

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, confirmed.client_id);
    await waitFor(() => expect(directorApi.updateSettings).toHaveBeenCalled());
    await waitFor(() => expect(directorApi.getSettingsAuthority).toHaveBeenCalledTimes(4));
    expect(clientId).toHaveValue(confirmed.client_id);
    expect(screen.queryByText("PUT response lost")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "系统设置" })).toBeEnabled();
  });

  it("同 endpoint 权威 GET 短暂失败时保留资源，重试成功后恢复生成", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "资源保持在线";
    saveLocalTimeline(project);
    mockCommonRequests();
    const confirmed = { ...CONFIGURED_SETTINGS, client_id: "retry-client" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockRejectedValueOnce(new Error("temporary GET failure"))
      .mockResolvedValue(runtimeAuthority(confirmed));
    vi.spyOn(directorApi, "updateSettings").mockResolvedValue(confirmed);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    expect(screen.getAllByRole("option", { name: "alternate-diffusion.safetensors" }).length).toBeGreaterThan(0);
    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, confirmed.client_id);
    await waitFor(() => expect(directorApi.updateSettings).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(directorApi.getSettingsAuthority).toHaveBeenCalledTimes(3), { timeout: 2000 });
    expect(screen.getByText("ComfyUI 在线")).toBeInTheDocument();
    expect(screen.getAllByRole("option", { name: "alternate-diffusion.safetensors" }).length).toBeGreaterThan(0);

    await waitFor(() => expect(directorApi.getSettingsAuthority).toHaveBeenCalledTimes(5), { timeout: 3000 });
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /生成任务/ })).toBeEnabled());
  });

  it("初始资源读取在途时同 endpoint 编辑建立最新资源权威且不重复 PUT", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "资源重试";
    saveLocalTimeline(project);
    mockCommonRequests();
    const confirmed = { ...CONFIGURED_SETTINGS, client_id: "resource-retry-client" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(confirmed));
    let resolveInitialCapabilities!: (value: CapabilityReport) => void;
    vi.mocked(directorApi.getCapabilities)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveInitialCapabilities = resolve; }))
      .mockResolvedValue(ONLINE_CAPABILITIES);
    const update = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(confirmed);

    render(<App />);
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, confirmed.client_id);
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));

    await waitFor(() => expect(directorApi.getCapabilities).toHaveBeenCalledTimes(2));
    expect(update).toHaveBeenCalledTimes(1);
    await act(async () => resolveInitialCapabilities({
      ...ONLINE_CAPABILITIES,
      connection: "offline",
      message: "迟到的旧读取",
    }));
    expect(screen.getByText("ComfyUI 在线")).toBeInTheDocument();
    expect(screen.getAllByRole("option", { name: "alternate-diffusion.safetensors" }).length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /生成任务/ })).toBeEnabled());
  }, 8_000);

  it("endpoint 资源部分失败时只重试资源接口，设置 PUT 保持一次", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "新端点资源重试";
    saveLocalTimeline(project);
    mockCommonRequests();
    const endpointB = { ...CONFIGURED_SETTINGS, comfy_url: "http://comfy-resource-b.test:8188" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointB));
    vi.mocked(directorApi.getCapabilities)
      .mockResolvedValueOnce(ONLINE_CAPABILITIES)
      .mockResolvedValueOnce({ ...ONLINE_CAPABILITIES, connection: "offline" })
      .mockResolvedValue(ONLINE_CAPABILITIES);
    vi.mocked(directorApi.getModels)
      .mockResolvedValueOnce(MODEL_INVENTORY)
      .mockRejectedValueOnce(new Error("model inventory unavailable"))
      .mockResolvedValue(MODEL_INVENTORY);
    const update = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(endpointB);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointB.comfy_url);
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(directorApi.getCapabilities).toHaveBeenCalledTimes(3), { timeout: 3_500 });
    expect(directorApi.getModels).toHaveBeenCalledTimes(3);
    expect(update).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("button", { name: "关闭系统设置" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /生成任务/ })).toBeEnabled());
  });

  it("设置 PUT 422 且权威 GET 失败时停止重试并保留可行动提示", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockRejectedValue(new Error("readback unavailable"));
    const update = vi.spyOn(directorApi, "updateSettings")
      .mockRejectedValue(new ApiError("客户端 ID 被服务器拒绝", 422));

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const clientId = screen.getByLabelText("客户端 ID");
    await user.clear(clientId);
    await user.type(clientId, "rejected-client");
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 4_200)); });
    expect(update).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/服务器拒绝当前系统设置.*客户端 ID 被服务器拒绝.*请修改/)).toHaveAttribute("role", "alert");
    await user.clear(clientId);
    await user.type(clientId, "corrected-client");
    expect(screen.queryByText(/客户端 ID 被服务器拒绝/)).not.toBeInTheDocument();
    expect(localStorage.getItem(RUNTIME_SETTINGS_PENDING_KEY)).not.toBeNull();
  }, 8_000);

  it("任务删除后自动刷新列表并保持记录移除", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    vi.mocked(directorApi.listTasks)
      .mockResolvedValueOnce({ jobs: [succeededTask] })
      .mockResolvedValue({ jobs: [] });
    vi.spyOn(directorApi, "deleteTask").mockResolvedValue({
      deleted_job_id: succeededTask.id,
      outputs_preserved: true,
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(screen.getByRole("button", { name: /任务，0 个进行中/ }));
    const drawer = screen.getByRole("complementary", { name: "任务列表" });
    const more = await within(drawer).findByRole("button", { name: "任务 job-term 的更多操作" });
    await user.click(more);
    await user.click(screen.getByRole("menuitem", { name: "移除任务记录" }));
    await waitFor(() => expect(directorApi.listTasks).toHaveBeenCalledTimes(2));
    expect(screen.queryByRole("button", { name: "任务 job-term 的更多操作" })).not.toBeInTheDocument();
  });

  it("确认 ComfyUI 已重启后立即更新任务、刷新列表并提示成功", async () => {
    const user = userEvent.setup();
    const recoveryTask: GenerationTask = {
      ...queuedTimelineTask,
      id: "job-restart-recovery",
      status: "cancelling",
      stage: "restart_cancel_unconfirmed",
    };
    const recoveredTask: GenerationTask = {
      ...recoveryTask,
      status: "cancelled",
      progress: 1,
      stage: "restart_cancel_confirmed",
      completed_at: "2026-08-12T00:03:00Z",
    };
    mockCommonRequests();
    vi.mocked(directorApi.listTasks)
      .mockResolvedValueOnce({ jobs: [recoveryTask] })
      .mockResolvedValue({ jobs: [recoveredTask] });
    const confirmRecovery = vi.spyOn(
      directorApi,
      "confirmComfyRestartRecovery",
    ).mockResolvedValue(recoveredTask);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(await screen.findByRole("button", { name: "任务，1 个进行中" }));
    await user.click(screen.getByRole("button", { name: "任务 job-rest 的更多操作" }));
    await user.click(screen.getByRole("menuitem", {
      name: "确认 ComfyUI 已重启并结束任务",
    }));

    await waitFor(() => expect(confirmRecovery).toHaveBeenCalledWith(recoveryTask.id));
    expect(await screen.findByText(
      "已确认 ComfyUI 重启，导演台任务已结束",
    )).toHaveAttribute("role", "status");
    await waitFor(() => expect(directorApi.listTasks).toHaveBeenCalledTimes(2));
    expect(screen.getByText("已取消")).toBeInTheDocument();
  });

  it("人工重启确认失败时显示错误并刷新权威任务", async () => {
    const user = userEvent.setup();
    const recoveryTask: GenerationTask = {
      ...queuedTimelineTask,
      id: "job-restart-failed",
      status: "cancelling",
      stage: "restart_cancel_failed",
    };
    mockCommonRequests();
    vi.mocked(directorApi.listTasks)
      .mockResolvedValueOnce({ jobs: [recoveryTask] })
      .mockResolvedValue({ jobs: [recoveryTask] });
    const confirmRecovery = vi.spyOn(
      directorApi,
      "confirmComfyRestartRecovery",
    ).mockRejectedValue(new ApiError("服务器拒绝该确认", 409));
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(await screen.findByRole("button", { name: "任务，1 个进行中" }));
    await user.click(screen.getByRole("button", { name: "任务 job-rest 的更多操作" }));
    await user.click(screen.getByRole("menuitem", {
      name: "确认 ComfyUI 已重启并结束任务",
    }));

    await waitFor(() => expect(confirmRecovery).toHaveBeenCalledWith(recoveryTask.id));
    expect(await screen.findByText(
      "恢复确认失败：服务器拒绝该确认",
    )).toHaveAttribute("role", "status");
    await waitFor(() => expect(directorApi.listTasks).toHaveBeenCalledTimes(2));
  });

  it("StrictMode 首次请求被清理中止后仍立即重新加载任务历史", async () => {
    mockCommonRequests();
    vi.mocked(directorApi.listTasks)
      .mockImplementationOnce((signal) => new Promise((_, reject) => {
        signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      }))
      .mockResolvedValue({ jobs: [queuedTimelineTask] });

    render(<StrictMode><App /></StrictMode>);

    expect(await screen.findByRole("button", { name: "任务，1 个进行中" })).toBeInTheDocument();
    expect(directorApi.listTasks).toHaveBeenCalledTimes(2);
  });

  it("任务刷新和活跃状态变化不会重建 SSE 连接", async () => {
    class FakeEventSource extends EventTarget {
      static instances: FakeEventSource[] = [];

      readonly url: string;
      readonly close = vi.fn();

      constructor(url: string | URL) {
        super();
        this.url = String(url);
        FakeEventSource.instances.push(this);
      }
    }

    vi.stubGlobal(
      "EventSource",
      FakeEventSource as unknown as typeof EventSource,
    );
    mockCommonRequests();
    const view = render(<StrictMode><App /></StrictMode>);
    await waitUntilReady();
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));

    const source = FakeEventSource.instances[0];
    expect(source.url).toBe("/api/tasks/events");
    const initialTaskReads = vi.mocked(directorApi.listTasks).mock.calls.length;
    vi.mocked(directorApi.listTasks).mockResolvedValueOnce({
      jobs: [queuedTimelineTask],
    });
    act(() => source.dispatchEvent(new Event("refresh")));

    await waitFor(() => expect(directorApi.listTasks).toHaveBeenCalledTimes(
      initialTaskReads + 1,
    ));
    expect(await screen.findByRole("button", { name: "任务，1 个进行中" })).toBeInTheDocument();
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(source.close).not.toHaveBeenCalled();

    const readsBeforeCompletion = vi.mocked(directorApi.listTasks).mock.calls.length;
    vi.mocked(directorApi.listTasks).mockResolvedValueOnce({
      jobs: [succeededTask],
    });
    act(() => source.dispatchEvent(new Event("refresh")));
    await waitFor(() => expect(directorApi.listTasks).toHaveBeenCalledTimes(
      readsBeforeCompletion + 1,
    ));
    expect(await screen.findByRole("button", { name: "任务，0 个进行中" })).toBeInTheDocument();
    expect(FakeEventSource.instances).toHaveLength(1);
    expect(source.close).not.toHaveBeenCalled();

    const readsBeforeUnmount = vi.mocked(directorApi.listTasks).mock.calls.length;
    view.unmount();
    expect(source.close).toHaveBeenCalledTimes(1);
    act(() => source.dispatchEvent(new Event("refresh")));
    expect(directorApi.listTasks).toHaveBeenCalledTimes(readsBeforeUnmount);
  });

  it("把历史来源项目另存为新项目并持久化显式统一选择", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    const first = { ...project.segments[0], prompt: "历史第一段" };
    const second = { ...createTimelineSegment("fl2va", 2), prompt: "历史第二段" };
    project.segments = [first, second];
    mockCommonRequests();
    vi.mocked(directorApi.listTasks).mockResolvedValue({ jobs: [succeededTask] });
    vi.spyOn(directorApi, "getTaskProject").mockResolvedValue({
      job_id: succeededTask.id,
      project,
      segment_ids: [second.id],
    });
    const imported = {
      id: "imported-project",
      title: project.title,
      created_at: "2026-08-12T00:00:00Z",
      updated_at: "2026-08-12T00:00:00Z",
      segment_count: 2,
    };
    vi.spyOn(directorApi, "importProject").mockResolvedValue(imported);
    vi.spyOn(directorApi, "getProjectTimeline").mockResolvedValue(project);
    vi.spyOn(directorApi, "updateProjectTimeline").mockImplementation(async (_id, value) => value);
    const compile = vi.spyOn(directorApi, "compileProjectTimeline").mockRejectedValue(
      new Error("仅检查请求边界"),
    );

    render(<App />);
    await waitUntilReady();
    await waitUntilTimelineReady();
    await user.click(screen.getByRole("button", { name: "任务，0 个进行中" }));
    await user.click(await screen.findByRole("button", { name: "任务 job-term 的更多操作" }));
    await user.click(screen.getByRole("menuitem", { name: "另存为新项目" }));

    await waitFor(() => expect(screen.getByRole("button", { name: /^聚焦并选择片段 1：/ })).toHaveAttribute("aria-pressed", "false"));
    expect(screen.getByRole("button", { name: /^聚焦并选择片段 2：/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "生成任务 1" })).toBeEnabled();
    await waitFor(() => expect(loadTimelineSegmentSelectionPreference(
      ACTIVE_DATABASE,
      imported.id,
      project.segments.map((segment) => segment.id),
    )).toEqual([second.id]));

    await user.click(screen.getByRole("button", { name: "预检执行计划" }));
    await waitFor(() => expect(compile).toHaveBeenCalledWith(
      "imported-project",
      expect.objectContaining({
        config: expect.objectContaining({ segments: expect.any(Array) }),
        segment_ids: [second.id],
      }),
    ));
  });

  it("批量取消超过服务端单批上限时按 100 个父任务安全分批", async () => {
    const user = userEvent.setup();
    mockCommonRequests();
    vi.mocked(directorApi.getCapabilities).mockResolvedValue({
      ...ONLINE_CAPABILITIES,
      supports_cancel: true,
    });
    const queued = Array.from({ length: 101 }, (_, index) => ({
      ...queuedTimelineTask,
      id: `queued-bulk-${String(index).padStart(3, "0")}`,
      prompt_id: `prompt-bulk-${index}`,
    }));
    vi.mocked(directorApi.listTasks)
      .mockResolvedValueOnce({ jobs: queued })
      .mockResolvedValue({ jobs: [] });
    const cancelTasks = vi.spyOn(directorApi, "cancelTasks")
      .mockImplementation(async (ids) => ({
        jobs: ids.map((id) => ({
          ...queuedTimelineTask,
          id,
          status: "cancelled" as const,
          completed_at: "2026-08-12T00:02:00Z",
        })),
        requested_count: ids.length,
        terminal_count: ids.length,
      }));
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(await screen.findByRole("button", { name: "任务，101 个进行中" }));
    await user.click(screen.getByRole("button", { name: "清空等待" }));

    await waitFor(() => expect(cancelTasks).toHaveBeenCalledTimes(2));
    expect(cancelTasks.mock.calls[0][0]).toHaveLength(100);
    expect(cancelTasks.mock.calls[1][0]).toHaveLength(1);
    expect(new Set(cancelTasks.mock.calls.flatMap(([ids]) => ids))).toHaveProperty("size", 101);
  }, 10_000);

  it("任务自动轮询慢于间隔时保持单请求且最终更新", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      mockCommonRequests();
      let resolveInitial!: (value: { jobs: GenerationTask[] }) => void;
      let resolveSlowPoll!: (value: { jobs: GenerationTask[] }) => void;
      vi.mocked(directorApi.listTasks)
        .mockImplementationOnce(() => new Promise((resolve) => { resolveInitial = resolve; }))
        .mockImplementationOnce(() => new Promise((resolve) => { resolveSlowPoll = resolve; }))
        .mockResolvedValue({ jobs: [{ ...queuedTimelineTask, id: "queued-after-slow-refresh" }] });

      render(<App />);
      await waitUntilReady();
      await act(async () => { await vi.advanceTimersByTimeAsync(5_100); });
      expect(directorApi.listTasks).toHaveBeenCalledTimes(1);

      await act(async () => resolveInitial({ jobs: [queuedTimelineTask] }));
      await waitFor(() => expect(screen.getByRole("button", { name: "任务，1 个进行中" })).toBeInTheDocument());
      await act(async () => { await vi.advanceTimersByTimeAsync(7_600); });
      await waitFor(() => expect(directorApi.listTasks).toHaveBeenCalledTimes(2));
      expect(directorApi.listTasks).toHaveBeenCalledTimes(2);

      await act(async () => resolveSlowPoll({ jobs: [{ ...queuedTimelineTask, id: "queued-after-slow-refresh" }] }));
      await waitFor(() => expect(screen.getByRole("button", { name: "任务，1 个进行中" })).toBeInTheDocument());
    } finally {
      vi.useRealTimers();
    }
  });

  it("服务器时间线只在没有本地编辑时 hydration", async () => {
    mockCommonRequests();
    const serverProject = createTimelineProject();
    serverProject.title = "服务端长片";
    serverProject.segments[0].prompt = "服务端片段提示";
    vi.mocked(directorApi.getTimeline).mockResolvedValue(serverProject);

    render(<App />);
    await waitUntilReady();
    expect((await screen.findAllByText("服务端长片")).length).toBeGreaterThan(0);
    expect(screen.getByLabelText("片段提示词")).toHaveValue("服务端片段提示");
  });

  it("移出被引用素材先保存完整时间线，再级联删除并回读服务器降级结果", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments = [{ ...createTimelineSegment("fl2va", 1), prompt: "人物走近", first_image: imageAsset }];
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.listAssets)
      .mockResolvedValueOnce({ assets: [imageAsset], outputs_preserved: true })
      .mockResolvedValue({ assets: [], outputs_preserved: true });
    const authoritative = {
      ...project,
      segments: [{
        id: project.segments[0].id,
        mode: "fl2va" as const,
        title: project.segments[0].title,
        prompt: project.segments[0].prompt,
        duration_seconds: project.segments[0].duration_seconds,
        enabled: true,
        continuity: project.segments[0].continuity,
        ref_image_size: project.segments[0].ref_image_size,
        audio_mode: project.segments[0].audio_mode,
        first_image: null,
        last_image: null,
      }],
    };
    vi.mocked(directorApi.getTimeline).mockResolvedValue(authoritative);
    const save = vi.spyOn(directorApi, "updateTimeline").mockImplementation(async (value) => value);
    const cascade = vi.spyOn(directorApi, "deleteAssetCascade").mockResolvedValue({
      deleted_asset_id: imageAsset.id,
      outputs_preserved: true,
      unbound_usages: [`timeline:${project.segments[0].id}:first_image`],
    });
    const plainDelete = vi.spyOn(directorApi, "deleteAsset");
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(await screen.findByRole("listitem"));
    await user.click(screen.getByRole("button", { name: "移出素材库" }));

    await waitFor(() => expect(cascade).toHaveBeenCalledWith(imageAsset.id));
    expect(save).toHaveBeenCalledWith(expect.objectContaining({ segments: [expect.objectContaining({ mode: "fl2va", first_image: imageAsset })] }));
    expect(save.mock.invocationCallOrder[0]).toBeLessThan(cascade.mock.invocationCallOrder[0]);
    expect(plainDelete).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.getByLabelText("片段生成模式")).toHaveValue("fl2va"));
    expect(directorApi.getTimeline).toHaveBeenCalledTimes(1);
  });

  it("级联成功但权威回读失败时不套用不完整本地降级，并自动恢复权威时间线", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments = [{ ...createTimelineSegment("fl2va", 1), prompt: "人物走近", first_image: imageAsset }];
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.listAssets)
      .mockResolvedValueOnce({ assets: [imageAsset], outputs_preserved: true })
      .mockResolvedValue({ assets: [], outputs_preserved: true });
    const authoritative = {
      ...project,
      title: "服务器权威降级",
      segments: [{
        id: project.segments[0].id,
        mode: "fl2va" as const,
        title: project.segments[0].title,
        prompt: project.segments[0].prompt,
        duration_seconds: project.segments[0].duration_seconds,
        enabled: true,
        continuity: project.segments[0].continuity,
        ref_image_size: project.segments[0].ref_image_size,
        audio_mode: project.segments[0].audio_mode,
        first_image: null,
        last_image: null,
      }],
    };
    vi.mocked(directorApi.getTimeline)
      .mockRejectedValueOnce(new Error("临时回读失败"))
      .mockResolvedValue(authoritative);
    vi.spyOn(directorApi, "updateTimeline").mockImplementation(async (value) => value);
    vi.spyOn(directorApi, "deleteAssetCascade").mockResolvedValue({
      deleted_asset_id: imageAsset.id,
      outputs_preserved: true,
      unbound_usages: [`timeline:${project.segments[0].id}:first_image`],
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(await screen.findByRole("listitem"));
    await user.click(screen.getByRole("button", { name: "移出素材库" }));

    expect(screen.queryByRole("button", { name: "重新同步" })).not.toBeInTheDocument();
    expect(await screen.findByText("服务器权威降级")).toBeInTheDocument();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).toBeNull();
    expect(screen.queryByRole("button", { name: /保存时间线|重新同步/ })).not.toBeInTheDocument();
  });

  it("资产级联与在途自动同步串行化，旧响应不能覆盖权威时间线", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.title = "编辑中项目";
    project.segments = [{ ...createTimelineSegment("fl2va", 1), prompt: "人物走近", first_image: imageAsset }];
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.listAssets)
      .mockResolvedValueOnce({ assets: [imageAsset], outputs_preserved: true })
      .mockResolvedValue({ assets: [], outputs_preserved: true });
    const authoritative = {
      ...project,
      title: "级联后的权威项目",
      segments: [{
        id: project.segments[0].id,
        mode: "fl2va" as const,
        title: project.segments[0].title,
        prompt: project.segments[0].prompt,
        duration_seconds: project.segments[0].duration_seconds,
        enabled: true,
        continuity: project.segments[0].continuity,
        ref_image_size: project.segments[0].ref_image_size,
        audio_mode: project.segments[0].audio_mode,
        first_image: null,
        last_image: null,
      }],
    };
    vi.mocked(directorApi.getTimeline).mockResolvedValue(authoritative);
    let resolveManualSave!: (value: typeof project) => void;
    const update = vi.spyOn(directorApi, "updateTimeline")
      .mockImplementationOnce(() => new Promise((resolve) => { resolveManualSave = resolve; }))
      .mockImplementation(async (value) => value);
    const cascade = vi.spyOn(directorApi, "deleteAssetCascade").mockResolvedValue({
      deleted_asset_id: imageAsset.id,
      outputs_preserved: true,
      unbound_usages: [`timeline:${project.segments[0].id}:first_image`],
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(update).toHaveBeenCalledTimes(1));
    await user.click(await screen.findByRole("listitem"));
    await user.click(screen.getByRole("button", { name: "移出素材库" }));
    expect(cascade).not.toHaveBeenCalled();

    await act(async () => resolveManualSave({ ...project, title: "迟到的旧保存响应" }));
    await waitFor(() => expect(cascade).toHaveBeenCalledTimes(1));
    expect(update).toHaveBeenCalledTimes(1);
    expect(update.mock.invocationCallOrder[0]).toBeLessThan(cascade.mock.invocationCallOrder[0]);
    expect(await screen.findByText("级联后的权威项目")).toBeInTheDocument();
    expect(screen.queryByText("迟到的旧保存响应")).not.toBeInTheDocument();
  });

  it("素材移出意图从时间线 flush 前阻断 endpoint PUT，级联结束后自动继续", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments = [{ ...createTimelineSegment("fl2va", 1), prompt: "删除边界", first_image: imageAsset }];
    saveLocalTimeline(project);
    mockCommonRequests();
    const endpointB = { ...CONFIGURED_SETTINGS, comfy_url: "http://comfy-after-delete.test:8188" };
    vi.mocked(directorApi.getSettingsAuthority)
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(runtimeAuthority(CONFIGURED_SETTINGS))
      .mockResolvedValue(runtimeAuthority(endpointB));
    vi.mocked(directorApi.listAssets)
      .mockResolvedValueOnce({ assets: [imageAsset], outputs_preserved: true })
      .mockResolvedValue({ assets: [], outputs_preserved: true });
    const authoritative = {
      ...project,
      segments: [{ ...project.segments[0], first_image: null, last_image: null }],
    };
    vi.mocked(directorApi.getTimeline).mockResolvedValue(authoritative);
    let resolveTimeline!: (value: typeof project) => void;
    vi.mocked(directorApi.updateTimeline).mockImplementationOnce(
      () => new Promise((resolve) => { resolveTimeline = resolve; }),
    );
    let resolveCascade!: (value: {
      deleted_asset_id: string;
      outputs_preserved: true;
      unbound_usages: string[];
    }) => void;
    const cascade = vi.spyOn(directorApi, "deleteAssetCascade").mockImplementation(
      () => new Promise((resolve) => { resolveCascade = resolve; }),
    );
    const updateSettings = vi.spyOn(directorApi, "updateSettings").mockResolvedValue(endpointB);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalledTimes(1));
    await user.click(await screen.findByRole("listitem"));
    await user.click(screen.getByRole("button", { name: "移出素材库" }));
    expect(screen.getByText("正在原子解除素材引用；时间线编辑与生成暂时锁定。")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "系统设置" }));
    const endpoint = screen.getByLabelText("ComfyUI 地址");
    await user.clear(endpoint);
    await user.type(endpoint, endpointB.comfy_url);
    await new Promise((resolve) => window.setTimeout(resolve, 350));
    expect(updateSettings).not.toHaveBeenCalled();

    await act(async () => resolveTimeline(project));
    await waitFor(() => expect(cascade).toHaveBeenCalledTimes(1));
    expect(updateSettings).not.toHaveBeenCalled();
    await act(async () => resolveCascade({
      deleted_asset_id: imageAsset.id,
      outputs_preserved: true,
      unbound_usages: [`timeline:${project.segments[0].id}:first_image`],
    }));
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
  });

  it("级联等待的在途同步失败时保留 WAL 且不显示保存功能", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments = [{ ...createTimelineSegment("fl2va", 1), first_image: imageAsset }];
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.listAssets).mockResolvedValue({ assets: [imageAsset], outputs_preserved: true });
    let rejectAutosave!: (reason: unknown) => void;
    vi.spyOn(directorApi, "updateTimeline")
      .mockImplementationOnce(() => new Promise((_, reject) => { rejectAutosave = reject; }))
      .mockImplementation(async (value) => value);
    const cascade = vi.spyOn(directorApi, "deleteAssetCascade");
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await waitFor(() => expect(directorApi.updateTimeline).toHaveBeenCalledTimes(1));
    await user.click(await screen.findByRole("listitem"));
    await user.click(screen.getByRole("button", { name: "移出素材库" }));
    await act(async () => rejectAutosave(new Error("前置同步失败")));

    await waitFor(() => expect(screen.getAllByText(/前置同步失败/).length).toBeGreaterThan(0));
    expect(screen.queryByText("保存失败")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "保存时间线" })).not.toBeInTheDocument();
    expect(localStorage.getItem(TIMELINE_WAL_STORAGE_KEY)).not.toBeNull();
    expect(cascade).not.toHaveBeenCalled();
  });

  it("本地无引用但隐藏草稿 409 时要求二次确认，然后走原子级联", async () => {
    const user = userEvent.setup();
    const project = createTimelineProject();
    project.segments[0].prompt = "雨夜";
    saveLocalTimeline(project);
    mockCommonRequests();
    vi.mocked(directorApi.listAssets)
      .mockResolvedValueOnce({ assets: [imageAsset], outputs_preserved: true })
      .mockResolvedValue({ assets: [], outputs_preserved: true });
    vi.mocked(directorApi.getTimeline).mockResolvedValue(project);
    const save = vi.spyOn(directorApi, "updateTimeline").mockImplementation(async (value) => value);
    const plainDelete = vi.spyOn(directorApi, "deleteAsset").mockRejectedValue(new ApiError(
      "素材仍被引用",
      409,
      { detail: { message: "素材仍被引用", usages: ["legacy:i2v:shot-1:first_image"] } },
    ));
    const cascade = vi.spyOn(directorApi, "deleteAssetCascade").mockResolvedValue({
      deleted_asset_id: imageAsset.id,
      outputs_preserved: true,
      unbound_usages: ["legacy:i2v:shot-1:first_image"],
    });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<App />);
    await waitUntilReady();
    await user.click(await screen.findByRole("listitem"));
    await user.click(screen.getByRole("button", { name: "移出素材库" }));
    await waitFor(() => expect(cascade).toHaveBeenCalledWith(imageAsset.id));
    expect(save.mock.invocationCallOrder[0]).toBeLessThan(plainDelete.mock.invocationCallOrder[0]);
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining("legacy:i2v:shot-1:first_image"));
  });

  it("主监视器按稳定片段 ID 和任务创建时间显示最新历史候选", async () => {
    const project = createTimelineProject();
    project.segments[0].prompt = "月光";
    mockCommonRequests();
    vi.mocked(directorApi.getTimeline).mockResolvedValue(project);
    const candidateTask: GenerationTask = {
      ...succeededTask,
      id: "candidate-job-1234",
      created_at: "2026-08-12T00:03:00Z",
      updated_at: "2026-08-12T00:03:00Z",
      segment_results: [{
        segment_id: project.segments[0].id,
        child_id: "candidate-child",
        output_url: "/api/jobs/candidate-job-1234/segment-output?segment_id=first",
        output_file: "output/director/candidate.mp4",
        current_snapshot: false,
      }],
    };
    const olderCandidate: GenerationTask = {
      ...candidateTask,
      id: "older-candidate-with-later-update",
      created_at: "2026-08-12T00:01:00Z",
      updated_at: "2026-08-12T00:10:00Z",
      segment_results: candidateTask.segment_results.map((result) => ({
        ...result,
        output_url: "/api/jobs/older-candidate-with-later-update/segment-output?segment_id=first",
      })),
    };
    const queued: GenerationTask = { ...queuedTimelineTask, id: "queued-newer", updated_at: "2026-08-12T00:05:00Z" };
    const running: GenerationTask = {
      ...queuedTimelineTask,
      id: "running-preview",
      status: "running",
      progress: 0.4,
      stage: "sampling 10/25",
      updated_at: "2026-08-12T00:04:00Z",
      live_preview_url: "/api/jobs/running-preview/live-preview",
    };
    const historicalCandidate = {
      ...candidateTask,
      segment_results: candidateTask.segment_results.map((result) => ({
        ...result,
        current_snapshot: false,
      })),
    };
    vi.mocked(directorApi.listTasks)
      .mockResolvedValueOnce({ jobs: [queued, running, olderCandidate, candidateTask] })
      .mockResolvedValue({ jobs: [queued, running, historicalCandidate] });
    vi.spyOn(directorApi, "updateTimeline").mockImplementation(async (value) => value);

    render(<App />);
    await waitUntilReady();
    expect(await screen.findByLabelText(new RegExp(`片段 ${project.segments[0].id} 的最新生成候选`))).toHaveAttribute("src", candidateTask.segment_results[0].output_url);
    fireEvent.click(screen.getByRole("button", { name: "实时执行" }));
    const live = screen.getByAltText("任务 running- 最新采样预览帧");
    expect(live).toHaveAttribute("src", expect.stringContaining("/api/jobs/running-preview/live-preview?v="));
    expect(screen.getByText(/running- · sampling 10\/25/)).toBeInTheDocument();

    await userEvent.setup().type(screen.getByLabelText("片段提示词"), "修改后");
    expect(screen.getByLabelText(new RegExp(`片段 ${project.segments[0].id} 的最新生成候选`))).toHaveAttribute("src", candidateTask.segment_results[0].output_url);
    await waitFor(() => expect(directorApi.listTasks).toHaveBeenCalledTimes(2));
    expect(screen.getByLabelText(new RegExp(`片段 ${project.segments[0].id} 的最新生成候选`))).toHaveAttribute("src", candidateTask.segment_results[0].output_url);
  });
});
