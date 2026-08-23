import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps } from "react";
import type {
  GenerationTask,
  TaskGenerationDetails,
  TaskProjectSnapshotResponse,
} from "../api/types";
import { TaskDrawer } from "../components/TaskDrawer";
import { buildHistoricalProjectConfigDownload } from "../domain/historicalProjectExport";
import { createTimelineProject } from "../domain/timelineProject";
import { loadTimelineWorkspacePreferences } from "../domain/workspacePreferences";

const baseTask: GenerationTask = {
  id: "job-active-1234",
  mode: "timeline",
  status: "running",
  progress: 0.42,
  stage: "采样 · 片段 2/3 · 第 7/25 步",
  prompt_id: "prompt-1",
  outputs: [],
  error: null,
  preview_url: null,
  created_at: "2026-08-12T15:00:00Z",
  updated_at: "2026-08-12T15:01:00Z",
  started_at: "2026-08-12T15:00:10Z",
  completed_at: null,
  children: [],
  segment_results: [],
  live_preview_url: null,
};

const completedTask: GenerationTask = {
  ...baseTask,
  id: "job-done-5678",
  status: "succeeded",
  progress: 1,
  stage: "completed",
  outputs: ["/api/jobs/job-done-5678/outputs/0"],
  output_files: ["output/video/final.mp4"],
  preview_url: "/api/jobs/job-done-5678/outputs/0",
  completed_at: "2026-08-12T15:02:00Z",
  segment_results: [],
};

const defaultProps: ComponentProps<typeof TaskDrawer> = {
  open: true,
  tasks: [],
  loading: false,
  onClose: () => undefined,
  onRefresh: () => undefined,
  onCancel: () => undefined,
  onDelete: () => undefined,
  onClearCompleted: () => undefined,
  onExportDiagnostic: vi.fn().mockResolvedValue({
    schema_version: 1,
    id: "job-diagnostic",
    display_name: "脱敏诊断",
    project_title: null,
    mode: "timeline",
    status: "failed",
    progress: 1,
    stage: "failed",
    created_at: "2026-08-12T15:00:00Z",
    updated_at: "2026-08-12T15:03:00Z",
    started_at: "2026-08-12T15:00:10Z",
    completed_at: "2026-08-12T15:03:00Z",
    execution_duration_seconds: 170,
    output_files: [],
    error_summary: "已脱敏错误",
    children: [],
    settings_included: false,
    workflow_included: false,
  }),
};

const generationDetails: TaskGenerationDetails = {
  schema_version: 2,
  job_id: baseTask.id,
  project_title: "暴风雪故事",
  render: {
    width: 1280,
    height: 704,
    fps: 24,
    export_mode: "segments",
    total_duration_seconds: 10,
  },
  sampling: [{
    family: "fl2va",
    steps: 4,
    seed: 123456789,
    random_seed: true,
    sampler: "euler",
    scheduler: "karras",
    shift: 9.5,
    audio_shift: 2.5,
  }],
  models: [{
    family: "fl2va",
    filename: "minimax-h3.safetensors",
    device: "default",
    lora_name: "snow-style.safetensors",
    lora_strength: 0.75,
    backends: ["raylight"],
    logical_gpu_indices: [0, 1],
    ulysses_degree: 2,
    ring_degree: 1,
  }],
  shared_models: [{
    role: "clip",
    filename: "qwen.safetensors",
    device: "default",
  }],
  runtime_snapshot_available: true,
  segments: [{
    id: "segment-1",
    title: "雪山环境",
    family: "fl2va",
    recipe: "t2v",
    duration_seconds: 10,
    prompt: "POV walks through a blizzard with a red apple.",
    continuity_enabled: false,
    continuity_overlap_frames: 22,
    ref_image_size: "max",
    audio_mode: "generate",
    has_first_image: false,
    has_last_image: false,
    has_source_video: false,
    source_audio_as_reference: false,
    reference_image_count: 0,
    reference_audio_count: 0,
    reference_video_count: 0,
  }],
};

function renderDrawer(overrides: Partial<ComponentProps<typeof TaskDrawer>> = {}) {
  return render(<TaskDrawer {...defaultProps} {...overrides} />);
}

beforeEach(() => localStorage.clear());
afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("ComfyUI 风格的任务历史抽屉", () => {
  it("从顶栏下缘开始并占满剩余视口高度", () => {
    renderDrawer();

    const drawer = screen.getByRole("complementary", { name: "任务列表" });
    expect(drawer.style.top).toBe("var(--topbar-height, 52px)");
    expect(drawer.style.bottom).toBe("0px");
  });

  it("使用紧凑的单行标题、活动摘要和搜索筛选", () => {
    const { container } = renderDrawer();

    expect(screen.getByRole("heading", { name: "任务历史" })).toBeInTheDocument();
    expect(screen.queryByText("导演台任务")).not.toBeInTheDocument();
    expect(screen.getByText("没有活动任务")).toBeInTheDocument();
    expect(screen.queryByText("状态自动同步")).not.toBeInTheDocument();
    const header = container.querySelector(".task-drawer__header");
    expect(header).toContainElement(screen.getByRole("heading", { name: "任务历史" }));
    expect(header).toContainElement(screen.getByText("没有活动任务"));
    expect(header).toContainElement(screen.getByRole("button", { name: "清空等待" }));
    expect(header).toContainElement(screen.getByRole("button", { name: "取消运行中" }));
    expect(header).toContainElement(screen.getByRole("button", { name: "关闭任务列表" }));
    expect(container.querySelector(".task-drawer__active-summary")).not.toBeInTheDocument();
    expect(container.querySelector(".task-drawer__filters")?.children).toHaveLength(2);
    expect(container.querySelector(".task-drawer__filters > .task-search")).toBeInTheDocument();
    expect(container.querySelector(".task-drawer__filters > .task-filter-actions")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "全部 0" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "已完成 0" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /失败/ })).not.toBeInTheDocument();
    expect(screen.queryByText("QUEUE")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /刷新/ })).not.toBeInTheDocument();
  });

  it("紧凑行只给对应状态显示取消、移除或查看操作", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    const onDelete = vi.fn();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const failedTask: GenerationTask = {
      ...baseTask,
      id: "job-failed-9999",
      status: "failed",
      error: "采样节点失败",
      completed_at: "2026-08-12T15:03:00Z",
    };
    renderDrawer({
      tasks: [baseTask, completedTask, failedTask],
      supportsCancel: true,
      onCancel,
      onDelete,
    });

    expect(screen.getByText("1 个运行中 · 0 个排队中")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "取消" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "查看" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "移除任务 job-fail" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "移除任务 job-acti" })).not.toBeInTheDocument();
    expect(screen.getByLabelText("完成 42%").querySelector("i")).toHaveStyle({ width: "42%" });

    await user.click(screen.getByRole("button", { name: "取消" }));
    expect(onCancel).toHaveBeenCalledWith(baseTask.id);
    await user.click(screen.getByRole("button", { name: "移除任务 job-fail" }));
    expect(onDelete).toHaveBeenCalledWith(failedTask.id);
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining("ComfyUI 输出文件会保留"));

    await user.click(screen.getByRole("button", { name: "查看" }));
    expect(screen.getByRole("dialog", { name: "任务输出查看器" })).toBeInTheDocument();
    expect(screen.getByText("ComfyUI output/video/final.mp4")).toBeInTheDocument();
  });

  it("失败页包含 failed 和 cancelled，且搜索采用短延迟过滤", async () => {
    const user = userEvent.setup();
    const failedTask: GenerationTask = {
      ...baseTask,
      id: "job-failed-searchable",
      status: "failed",
      error: "显存不足",
      completed_at: "2026-08-12T15:03:00Z",
    };
    const cancelledTask: GenerationTask = {
      ...baseTask,
      id: "job-cancelled-0001",
      status: "cancelled",
      stage: "用户取消",
      completed_at: "2026-08-12T15:04:00Z",
    };
    const { container } = renderDrawer({ tasks: [completedTask, failedTask, cancelledTask] });

    await user.click(screen.getByRole("tab", { name: "失败 2" }));
    expect(screen.getByText("已取消")).toBeInTheDocument();
    expect(container.querySelectorAll(".task-row")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: "查看" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "全部 3" }));
    await user.type(screen.getByRole("searchbox", { name: "搜索任务" }), "cancelled-0001");
    await waitFor(() => {
      expect(screen.getByText("已取消")).toBeInTheDocument();
      expect(container.querySelectorAll(".task-row")).toHaveLength(1);
    });
  });

  it("刷新后恢复任务标签、当前项目筛选和排序，但不恢复搜索词", async () => {
    const user = userEvent.setup();
    const tasks: GenerationTask[] = [
      { ...baseTask, current_project: true },
      { ...completedTask, current_project: true },
    ];
    const firstView = renderDrawer({ tasks });

    await user.click(screen.getByRole("tab", { name: "已完成 1" }));
    await user.click(screen.getByRole("button", { name: "当前项目" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "任务排序" }), "duration");
    await user.type(screen.getByRole("searchbox", { name: "搜索任务" }), "不会恢复");
    await waitFor(() => expect(loadTimelineWorkspacePreferences()).toMatchObject({
      taskTab: "completed",
      taskCurrentProjectOnly: true,
      taskSort: "duration",
    }));

    firstView.unmount();
    renderDrawer({ tasks });
    expect(screen.getByRole("tab", { name: "已完成 1" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("button", { name: "当前项目" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("combobox", { name: "任务排序" })).toHaveValue("duration");
    expect(screen.getByRole("searchbox", { name: "搜索任务" })).toHaveValue("");
  });

  it("按服务端任务归属过滤当前项目", async () => {
    const user = userEvent.setup();
    const current = { ...completedTask, current_project: true };
    const historical = {
      ...completedTask,
      id: "job-old-project",
      current_project: false,
      output_files: ["output/video/old.mp4"],
    };
    renderDrawer({ tasks: [current, historical] as GenerationTask[] });

    await user.click(screen.getByRole("button", { name: "当前项目" }));
    expect(screen.getByRole("button", { name: "查看" })).toBeInTheDocument();
    expect(screen.queryByText("old.mp4", { exact: false })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "当前项目" })).toHaveAttribute("aria-pressed", "true");
  });

  it("批量清等待和取消运行只传递 Director 父任务 ID", async () => {
    const user = userEvent.setup();
    const onBulkCancel = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    const queuedOne = { ...baseTask, id: "queued-one", status: "queued" as const };
    const queuedTwo = { ...baseTask, id: "queued-two", status: "queued" as const };
    renderDrawer({
      tasks: [queuedOne, queuedTwo, baseTask],
      supportsCancel: true,
      onBulkCancel,
    });

    await user.click(screen.getByRole("button", { name: "清空等待" }));
    expect(onBulkCancel).toHaveBeenNthCalledWith(1, ["queued-one", "queued-two"]);
    await user.click(screen.getByRole("button", { name: "取消运行中" }));
    expect(onBulkCancel).toHaveBeenNthCalledWith(2, [baseTask.id]);
  });

  it("悬停 200ms 后显示分段执行详情，并在离开后延迟收起", () => {
    vi.useFakeTimers();
    const parent: GenerationTask = {
      ...baseTask,
      id: "timeline-parent-1",
      children: [
        {
          id: "child-fl",
          family: "fl2va",
          backend: "standard",
          segment_ids: ["segment-a"],
          status: "succeeded",
          progress: 1,
          stage: "completed",
          prompt_id: "prompt-fl",
          outputs: ["output/director/child-fl.mp4"],
          error: null,
        },
        {
          id: "child-ref",
          family: "ref2va",
          backend: "raylight",
          segment_ids: ["segment-b", "segment-c"],
          status: "running",
          progress: 0.4,
          stage: "采样 10/25",
          prompt_id: "prompt-ref",
          outputs: [],
          error: null,
        },
      ],
    };
    const { container } = renderDrawer({ tasks: [parent] });
    const row = container.querySelector(".task-row");
    expect(row).not.toBeNull();
    fireEvent.mouseEnter(row!);
    act(() => vi.advanceTimersByTime(199));
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    expect(screen.getByText("FL2VA · 标准执行")).toBeInTheDocument();
    expect(screen.getByText("Ref2VA · RayLight")).toBeInTheDocument();
    expect(screen.getByText("2 段 · 采样 10/25")).toBeInTheDocument();
    fireEvent.mouseLeave(row!);
    act(() => vi.advanceTimersByTime(149));
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("排队详情把当前运行工作计入前方数量和预计等待", () => {
    vi.useFakeTimers();
    const running = { ...baseTask, progress: 0.5 };
    const queued = {
      ...baseTask,
      id: "queued-after-running",
      status: "queued" as const,
      progress: 0,
      started_at: null,
    };
    const completed = {
      ...completedTask,
      started_at: "2026-08-12T15:00:00Z",
      completed_at: "2026-08-12T15:02:00Z",
    };
    const { container } = renderDrawer({ tasks: [running, queued, completed] });
    const queuedRow = [...container.querySelectorAll<HTMLElement>(".task-row")]
      .find((row) => row.querySelector(
        '[aria-label="任务 queued-a 的更多操作"]',
      ));
    expect(queuedRow).toBeDefined();
    fireEvent.mouseEnter(queuedRow!);
    act(() => vi.advanceTimersByTime(200));
    expect(screen.getByText(/前方 1 个任务 · 预计等待约 01:00/)).toBeInTheDocument();
  });

  it("完成任务的查看器合并父输出和分段候选，并传回稳定 segment ID", async () => {
    const user = userEvent.setup();
    const onImportOutput = vi.fn();
    const withSegmentResult: GenerationTask = {
      ...completedTask,
      segment_results: [{
        segment_id: "segment-a",
        child_id: "child-a",
        output_url: "/api/jobs/job-done-5678/segment-output?segment_id=segment-a",
        output_file: "output/director/segment-a.mp4",
        current_snapshot: true,
      }],
    };
    const view = renderDrawer({ tasks: [withSegmentResult], onImportOutput });

    await user.click(screen.getByRole("button", { name: "查看" }));
    await waitFor(() => expect(
      screen.getByRole("button", { name: "关闭查看器" }),
    ).toHaveFocus());
    const outputVideo = screen.getByRole("dialog", { name: "任务输出查看器" })
      .querySelector<HTMLVideoElement>("video");
    expect(outputVideo).not.toBeNull();
    expect(outputVideo).toHaveClass("task-lightbox__media");
    expect(outputVideo).toHaveStyle({
      objectFit: "contain",
      objectPosition: "center",
    });
    expect(outputVideo?.parentElement).toHaveClass("task-lightbox__stage");
    expect(document.body.style.overflow).toBe("hidden");
    expect((view.container as HTMLElement & { inert: boolean }).inert).toBe(true);
    expect(screen.getByText("1 / 2")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "下一个输出" }));
    expect(screen.getByText("2 / 2")).toBeInTheDocument();
    expect(screen.getByText("ComfyUI output/director/segment-a.mp4")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "加入素材库" }));
    expect(onImportOutput).toHaveBeenCalledWith(withSegmentResult.id, {
      index: 1,
      segmentId: "segment-a",
    });
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "任务输出查看器" })).not.toBeInTheDocument();
    expect(document.body.style.overflow).toBe("");
    expect((view.container as HTMLElement & { inert?: boolean }).inert).not.toBe(true);
    await waitFor(() => expect(screen.getByRole("button", { name: "查看" })).toHaveFocus());
  });

  it("更多菜单支持项目恢复、ID/错误处理和脱敏诊断入口", async () => {
    const user = userEvent.setup();
    const onLoadProject = vi.fn();
    const historicalProject = createTimelineProject();
    historicalProject.title = "不可变历史创作配置";
    const historicalSnapshot: TaskProjectSnapshotResponse = {
      job_id: "job-menu-failed",
      project: historicalProject,
      segment_ids: [historicalProject.segments[0].id],
    };
    const onExportProjectConfig = vi.fn().mockResolvedValue(historicalSnapshot);
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    const onExportDiagnostic = vi.fn(
      () => new Promise<never>(() => undefined),
    );
    const failedTask: GenerationTask = {
      ...baseTask,
      id: "job-menu-failed",
      status: "failed",
      error: "服务器错误详情",
      completed_at: "2026-08-12T15:03:00Z",
    };
    renderDrawer({
      tasks: [failedTask],
      onLoadProject,
      onExportProjectConfig,
      onExportDiagnostic,
    });

    const menuTrigger = screen.getByRole("button", { name: "任务 job-menu 的更多操作" });
    expect(menuTrigger).toHaveAttribute("aria-expanded", "false");
    await user.click(menuTrigger);
    const menu = screen.getByRole("menu");
    expect(menuTrigger).toHaveAttribute("aria-expanded", "true");
    expect(menuTrigger).toHaveAttribute("aria-controls", menu.id);
    expect(menu).toHaveTextContent("另存为新项目");
    expect(menu).toHaveTextContent("导出配置");
    expect(menu).toHaveTextContent("导出脱敏诊断");
    expect(menu).toHaveTextContent("复制任务 ID");
    expect(menu).toHaveTextContent("查看错误详情");
    expect(menu).not.toHaveTextContent("工作流提示图");
    await user.click(screen.getByRole("menuitem", { name: "另存为新项目" }));
    expect(onLoadProject).toHaveBeenCalledWith(failedTask.id);

    await user.click(screen.getByRole("button", { name: "任务 job-menu 的更多操作" }));
    await user.click(screen.getByRole("menuitem", { name: "导出配置" }));
    await waitFor(() => expect(onExportProjectConfig).toHaveBeenCalledWith(failedTask.id));
    await waitFor(() => expect(anchorClick).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "任务 job-menu 的更多操作" }));
    await user.click(screen.getByRole("menuitem", { name: "导出脱敏诊断" }));
    expect(onExportDiagnostic).toHaveBeenCalledWith(failedTask.id);

    await user.click(screen.getByRole("button", { name: "任务 job-menu 的更多操作" }));
    await user.click(screen.getByRole("menuitem", { name: "查看错误详情" }));
    expect(screen.getByRole("dialog", { name: "任务错误详情" })).toHaveTextContent("服务器错误详情");
  });

  it("历史配置下载只序列化只读 v5 creative view，可原样再次导入", () => {
    const project = createTimelineProject();
    project.title = "历史可移植配置";
    project.model_stack.fl2va.filename = "historical-fl2va.safetensors";
    const snapshot = {
      job_id: "job/history unsafe",
      project,
      segment_ids: [project.segments[0].id],
      runtime_settings: { comfy_base_url: "http://private.example" },
    } as TaskProjectSnapshotResponse & { runtime_settings: unknown };

    const download = buildHistoricalProjectConfigDownload(snapshot.job_id, snapshot);

    expect(download.filename).toBe("director-project-job-history-unsafe.json");
    expect(download.mimeType).toBe("application/json");
    expect(JSON.parse(download.contents)).toEqual(project);
    expect(download.contents).not.toContain("runtime_settings");
    expect(download.contents).not.toContain("private.example");
    expect(() => buildHistoricalProjectConfigDownload("different-job", snapshot))
      .toThrow("任务来源项目与请求任务不匹配");
  });

  it("生成参数按需加载到可关闭、可回焦的结构化弹窗", async () => {
    const user = userEvent.setup();
    const onLoadGenerationDetails = vi.fn().mockResolvedValue(generationDetails);
    renderDrawer({ tasks: [baseTask], onLoadGenerationDetails });

    const menuTrigger = screen.getByRole("button", { name: "任务 job-acti 的更多操作" });
    await user.click(menuTrigger);
    await user.click(screen.getByRole("menuitem", { name: "查看生成参数" }));

    expect(onLoadGenerationDetails).toHaveBeenCalledWith(baseTask.id);
    const dialog = await screen.findByRole("dialog", { name: "任务生成参数" });
    expect(dialog).toHaveTextContent("暴风雪故事");
    expect(dialog).toHaveTextContent("1280 × 704");
    expect(dialog).toHaveTextContent("123456789");
    expect(dialog).toHaveTextContent("提交时随机（上方为实际值）");
    expect(dialog).toHaveTextContent("snow-style.safetensors（强度 0.75）");
    expect(dialog).toHaveTextContent("逻辑 GPU");
    expect(dialog).toHaveTextContent("雪山环境");
    const promptDisclosure = dialog.querySelector(".task-parameter-segments details");
    expect(promptDisclosure).not.toHaveAttribute("open");
    await user.click(screen.getByText("提示词"));
    expect(promptDisclosure).toHaveAttribute("open");
    expect(dialog).toHaveTextContent("POV walks through a blizzard with a red apple.");

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "任务生成参数" })).not.toBeInTheDocument();
    await waitFor(() => expect(menuTrigger).toHaveFocus());
  });

  it.each([
    "restart_cancel_pending",
    "restart_cancel_failed",
    "restart_cancel_unconfirmed",
    "restart_certificate_required",
  ])("取消恢复阶段 %s 同时保留重试取消和人工重启确认", async (stage) => {
    const user = userEvent.setup();
    const onConfirmComfyRestart = vi.fn();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const recoveryTask: GenerationTask = {
      ...baseTask,
      id: `job-${stage}`,
      status: "cancelling",
      stage,
    };
    renderDrawer({
      tasks: [recoveryTask],
      supportsCancel: false,
      onConfirmComfyRestart,
    });

    expect(screen.getByRole("button", { name: "重试取消" })).toBeDisabled();
    await user.click(screen.getByRole("button", {
      name: `任务 ${recoveryTask.id.slice(0, 8)} 的更多操作`,
    }));
    const recoveryAction = screen.getByRole("menuitem", {
      name: "确认 ComfyUI 已重启并结束任务",
    });
    expect(recoveryAction).toBeEnabled();
    expect(screen.getByRole("menuitem", { name: "重试取消" })).toBeDisabled();

    await user.click(recoveryAction);

    expect(confirm).toHaveBeenCalledWith(expect.stringMatching(
      /若实际未重启.*仍可能继续运行并占用 GPU/s,
    ));
    expect(onConfirmComfyRestart).toHaveBeenCalledWith(recoveryTask.id);
  });

  it("非取消中任务或非指定恢复阶段不显示人工重启确认", async () => {
    const user = userEvent.setup();
    const tasks: GenerationTask[] = [
      { ...baseTask, id: "running-recovery-stage", stage: "restart_cancel_pending" },
      { ...baseTask, id: "cancel-wrong-stage", status: "cancelling", stage: "submission_cancel_unconfirmed" },
      { ...baseTask, id: "failed-recovery-stage", status: "failed", stage: "restart_cancel_failed" },
    ];
    renderDrawer({ tasks, onConfirmComfyRestart: vi.fn() });

    for (const task of tasks) {
      await user.click(screen.getByRole("button", {
        name: `任务 ${task.id.slice(0, 8)} 的更多操作`,
      }));
      expect(screen.queryByRole("menuitem", {
        name: "确认 ComfyUI 已重启并结束任务",
      })).not.toBeInTheDocument();
      await user.keyboard("{Escape}");
    }
  });

  it("只为当前运行任务轮询实时缩略图，并用查询参数避免首帧缓存", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-12T16:00:00Z"));
    const current = {
      ...baseTask,
      id: "running-current",
      live_preview_url: "/api/jobs/running-current/live-preview",
      started_at: "2026-08-12T15:00:00Z",
    };
    const concurrent = {
      ...baseTask,
      id: "running-concurrent",
      live_preview_url: "/api/jobs/running-concurrent/live-preview",
      started_at: "2026-08-12T15:01:00Z",
    };
    const { container } = renderDrawer({ tasks: [current, concurrent] });
    const first = container.querySelector<HTMLImageElement>(".task-row__thumb img");
    expect(first).not.toBeNull();
    expect(container.querySelectorAll(".task-row__thumb img")).toHaveLength(1);
    expect(first!.src).toContain("running-current/live-preview?frame=");
    const initialSource = first!.src;
    act(() => vi.advanceTimersByTime(500));
    expect(first!.src).not.toBe(initialSource);
  });

  it("旧的重启确认任务不会遮住真正运行任务的阶段、进度和实时缩略图", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-12T16:00:00Z"));
    const staleCancellation: GenerationTask = {
      ...baseTask,
      id: "old-restart-certificate",
      status: "cancelling",
      stage: "restart_certificate_required",
      progress: 0.01,
      created_at: "2026-08-12T14:00:00Z",
      started_at: "2026-08-12T14:00:10Z",
      live_preview_url: "/api/jobs/old-restart-certificate/live-preview",
    };
    const currentRun: GenerationTask = {
      ...baseTask,
      id: "current-running-task",
      status: "running",
      stage: "构建多模态条件",
      progress: 0.45,
      created_at: "2026-08-12T15:00:00Z",
      started_at: "2026-08-12T15:00:10Z",
      live_preview_url: "/api/jobs/current-running-task/live-preview",
      children: [{
        id: "current-running-child",
        family: "fl2va",
        backend: "standard",
        segment_ids: ["segment-1"],
        status: "running",
        progress: 0.7,
        stage: "采样 3/4",
        prompt_id: "current-prompt",
        outputs: [],
        error: null,
      }],
    };

    const { container } = renderDrawer({
      tasks: [staleCancellation, currentRun],
    });

    expect(screen.getByText("采样 3/4")).toBeInTheDocument();
    expect(screen.queryByText("构建多模态条件")).not.toBeInTheDocument();
    expect(screen.getByText("取消中")).toBeInTheDocument();
    expect(screen.getByLabelText("完成 70%")).toBeInTheDocument();
    expect(screen.queryByLabelText("完成 1%")).not.toBeInTheDocument();
    const preview = container.querySelector<HTMLImageElement>(".task-row__thumb img");
    expect(preview).not.toBeNull();
    expect(preview!.src).toContain("current-running-task/live-preview?frame=");
    expect(preview!.src).not.toContain("old-restart-certificate");
  });

  it("清空历史先说明仅删记录，关闭抽屉时完全卸载", async () => {
    const user = userEvent.setup();
    const onClearCompleted = vi.fn();
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    const view = renderDrawer({ tasks: [baseTask], onClearCompleted });

    await user.click(screen.getByRole("button", { name: "清空历史记录" }));
    expect(onClearCompleted).toHaveBeenCalledTimes(1);
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining("活动任务会保留"));
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining("ComfyUI 输出文件会保留"));

    view.rerender(<TaskDrawer {...defaultProps} open={false} tasks={[baseTask]} />);
    expect(screen.queryByRole("complementary", { name: "任务列表" })).not.toBeInTheDocument();
  });

  it("Escape 关闭抽屉并回焦任务按钮，但先交给菜单和模态层处理", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const toggle = document.createElement("button");
    toggle.id = "task-panel-toggle";
    document.body.append(toggle);
    const view = renderDrawer({ tasks: [completedTask], onClose });

    await user.click(screen.getByRole("button", { name: "任务 job-done 的更多操作" }));
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "查看" }));
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "任务输出查看器" })).not.toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(toggle).toHaveFocus());

    view.unmount();
    toggle.remove();
  });

  it("主按钮、抽屉内容和任务模态层不触发外点关闭，工作区外点才关闭", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const toggle = document.createElement("button");
    toggle.id = "task-panel-toggle";
    const workspace = document.createElement("button");
    workspace.textContent = "工作区操作";
    document.body.append(toggle, workspace);
    const failedTask: GenerationTask = {
      ...baseTask,
      id: "job-failed-outside",
      status: "failed",
      error: "测试错误详情",
      completed_at: "2026-08-12T15:03:00Z",
    };
    const view = renderDrawer({ tasks: [completedTask, failedTask], onClose });

    fireEvent.pointerDown(screen.getByRole("searchbox", { name: "搜索任务" }));
    fireEvent.pointerDown(toggle);
    expect(onClose).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "查看" }));
    fireEvent.pointerDown(screen.getByRole("dialog", { name: "任务输出查看器" }));
    expect(onClose).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "关闭查看器" }));
    expect(screen.getByRole("complementary", { name: "任务列表" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "任务 job-fail 的更多操作" }));
    await user.click(screen.getByRole("menuitem", { name: "查看错误详情" }));
    fireEvent.pointerDown(screen.getByRole("dialog", { name: "任务错误详情" }));
    expect(onClose).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "关闭错误详情" }));

    fireEvent.pointerDown(workspace);
    expect(onClose).toHaveBeenCalledTimes(1);

    view.unmount();
    toggle.remove();
    workspace.remove();
  });

  it("未声明原子取消能力时禁用所有取消入口", () => {
    renderDrawer({ tasks: [baseTask] });

    expect(screen.getByRole("button", { name: "取消" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "取消运行中" })).toBeDisabled();
  });
});
