import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createPortal } from "react-dom";
import type {
  GenerationTask,
  TaskDiagnostic,
  TaskGenerationDetails,
  TaskProjectSnapshotResponse,
  TaskStatus,
} from "../api/types";
import { downloadHistoricalProjectConfig } from "../domain/historicalProjectExport";
import { MODE_META } from "../domain/modes";
import {
  loadTimelineWorkspacePreferences,
  updateTimelineWorkspacePreferences,
} from "../domain/workspacePreferences";
import { formatTime, Spinner } from "./ui";

const STATUS_LABEL: Record<TaskStatus, string> = {
  queued: "排队中",
  preparing: "准备中",
  running: "生成中",
  succeeded: "已完成",
  failed: "失败",
  cancelling: "取消中",
  cancelled: "已取消",
};

const STATUS_MARK: Record<TaskStatus, string> = {
  queued: "…",
  preparing: "◌",
  running: "▶",
  succeeded: "✓",
  failed: "!",
  cancelling: "◌",
  cancelled: "×",
};

const TERMINAL_STATUSES = new Set<TaskStatus>([
  "succeeded",
  "failed",
  "cancelled",
]);
const ACTIVE_STATUSES = new Set<TaskStatus>([
  "queued",
  "preparing",
  "running",
  "cancelling",
]);
const RUNNING_STATUSES = new Set<TaskStatus>([
  "preparing",
  "running",
  "cancelling",
]);
const COMFY_RESTART_CONFIRMATION_STAGES = new Set([
  "restart_cancel_pending",
  "restart_cancel_failed",
  "restart_cancel_unconfirmed",
]);

type TaskTab = "all" | "completed" | "failed";
type TaskSort = "recent" | "duration";

/**
 * These fields are deliberately optional while the lightweight task-list API
 * rolls out. The drawer remains compatible with historical full JobRead rows.
 */
export type TaskDrawerTask = GenerationTask & {
  display_name?: string | null;
  project_title?: string | null;
  project_id?: string | null;
  current_project?: boolean;
  execution_duration_seconds?: number | null;
};

export interface TaskDrawerProps {
  id?: string;
  open: boolean;
  tasks: GenerationTask[];
  loading: boolean;
  supportsCancel?: boolean;
  deletingTaskIds?: ReadonlySet<string>;
  clearing?: boolean;
  /** Optional until list rows expose a stable project identity. */
  currentProjectId?: string | null;
  onClose: () => void;
  /** Kept for call-site compatibility; synchronization is automatic. */
  onRefresh: () => void;
  onCancel: (id: string) => void | Promise<void>;
  onConfirmComfyRestart?: (id: string) => void | Promise<void>;
  onBulkCancel?: (ids: string[]) => void | Promise<void>;
  onDelete: (id: string) => void | Promise<void>;
  onClearCompleted: () => void | Promise<void>;
  onLoadProject?: (id: string) => void | Promise<void>;
  onExportProjectConfig?: (id: string) => Promise<TaskProjectSnapshotResponse>;
  onExportDiagnostic?: (id: string) => Promise<TaskDiagnostic>;
  onLoadGenerationDetails?: (id: string) => Promise<TaskGenerationDetails>;
  onImportOutput?: (
    taskId: string,
    output: { index: number; segmentId?: string },
  ) => void | Promise<void>;
}

interface MediaOutput {
  url: string;
  file: string | null;
  index: number;
  segmentId?: string;
}

interface MenuState {
  taskId: string;
  x: number;
  y: number;
}

interface GenerationDetailsState {
  taskId: string;
  loading: boolean;
  details: TaskGenerationDetails | null;
  error: string | null;
}

function confirmRecordDeletion(message: string): boolean {
  return window.confirm(
    `${message}\n\n只删除导演台任务记录，ComfyUI 输出文件会保留。`,
  );
}

function canConfirmComfyRestart(task: GenerationTask): boolean {
  return task.status === "cancelling" &&
    task.stage !== null &&
    COMFY_RESTART_CONFIRMATION_STAGES.has(task.stage);
}

function confirmComfyRestartRecovery(task: GenerationTask): boolean {
  return window.confirm(
    "仅当 ComfyUI 进程确实已经重启时才能执行此操作。\n" +
    "若实际未重启，之前迟到提交的任务仍可能继续运行并占用 GPU。\n\n" +
    `确认 ComfyUI 已重启，并结束导演台任务 ${task.id.slice(0, 8)} 吗？`,
  );
}

function displayOutputPath(path: string): string {
  const relative = path.replaceAll("\\", "/").replace(/^\/+/, "");
  return /^(input|output|temp)\//.test(relative)
    ? `ComfyUI ${relative}`
    : `ComfyUI output/${relative}`;
}

function taskTitle(task: TaskDrawerTask): string {
  const explicit = task.display_name?.trim() || task.project_title?.trim();
  if (explicit) return explicit;
  if (task.mode === "timeline") return "长视频生成";
  return MODE_META[task.mode].label;
}

function taskModeLabel(task: GenerationTask): string {
  if (task.mode === "timeline") return "长视频";
  return MODE_META[task.mode].label;
}

function timestamp(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = new Date(value).valueOf();
  return Number.isFinite(parsed) ? parsed : null;
}

function taskDurationSeconds(task: TaskDrawerTask, now: number): number | null {
  if (
    typeof task.execution_duration_seconds === "number" &&
    Number.isFinite(task.execution_duration_seconds) &&
    task.execution_duration_seconds >= 0
  ) {
    return task.execution_duration_seconds;
  }
  const started = timestamp(task.started_at);
  if (started === null) return null;
  const completed = timestamp(task.completed_at);
  const updated = timestamp(task.updated_at);
  const ended = completed ?? (ACTIVE_STATUSES.has(task.status) ? now : updated);
  return ended === null ? null : Math.max(0, (ended - started) / 1000);
}

function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds)) return "—";
  const rounded = Math.max(0, Math.round(seconds));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const remaining = rounded % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function localDateKey(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "undated";
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function dateGroupLabel(key: string, now: Date): string {
  if (key === "undated") return "日期未知";
  const today = localDateKey(now.toISOString());
  const yesterdayDate = new Date(
    now.getFullYear(),
    now.getMonth(),
    now.getDate() - 1,
  );
  const yesterday = localDateKey(yesterdayDate.toISOString());
  if (key === today) return "今天";
  if (key === yesterday) return "昨天";
  const [year, month, day] = key.split("-").map(Number);
  return year === now.getFullYear()
    ? `${month}月${day}日`
    : `${year}年${month}月${day}日`;
}

function belongsToCurrentProject(
  task: TaskDrawerTask,
  currentProjectId: string | null | undefined,
): boolean {
  if (typeof task.current_project === "boolean") return task.current_project;
  if (currentProjectId && task.project_id) {
    return currentProjectId === task.project_id;
  }
  // Older responses expose this authority only on completed segment results.
  return task.segment_results.some((result) => result.current_snapshot);
}

function getMediaOutputs(task: GenerationTask): MediaOutput[] {
  const outputFiles = task.output_files ?? [];
  const urls = task.outputs.length > 0
    ? task.outputs
    : task.preview_url
      ? [task.preview_url]
      : [];
  const outputs: MediaOutput[] = urls.map((url, index) => ({
    url,
    file: outputFiles[index] ?? null,
    index,
  }));
  const outputIdentity = (output: MediaOutput) =>
    output.file?.replaceAll("\\", "/") || output.url;
  const seen = new Set(outputs.map(outputIdentity));
  task.segment_results.forEach((result) => {
    const candidate = {
      url: result.output_url,
      file: result.output_file,
      index: outputs.length,
      segmentId: result.segment_id,
    };
    const identity = outputIdentity(candidate);
    if (seen.has(identity)) return;
    seen.add(identity);
    outputs.push(candidate);
  });
  return outputs;
}

function mediaKind(output: MediaOutput): "video" | "image" | "audio" | "text" | "other" {
  const candidate = (output.file ?? output.url).split("?")[0].toLowerCase();
  if (/\.(png|jpe?g|webp|gif|avif)$/.test(candidate)) return "image";
  if (/\.(mp3|wav|flac|ogg|m4a|aac)$/.test(candidate)) return "audio";
  if (/\.(txt|md|json|csv|log)$/.test(candidate)) return "text";
  if (/\.(mp4|webm|mov|mkv|m4v)$/.test(candidate)) return "video";
  // Director generation currently emits video; extension-less proxy URLs are
  // therefore treated as video unless the server supplies a typed filename.
  return output.file === null ? "video" : "other";
}

function triggerDownload(url: string, filename: string | null): void {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename?.split(/[\\/]/).at(-1) ?? "";
  anchor.rel = "noreferrer";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
}

function openInNewWindow(url: string): void {
  window.open(url, "_blank", "noopener,noreferrer");
}

async function copyText(value: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(value);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.append(textarea);
    textarea.select();
    document.execCommand?.("copy");
    textarea.remove();
  }
}

function downloadDiagnostic(taskId: string, diagnostic: TaskDiagnostic): void {
  const contents = JSON.stringify(diagnostic, null, 2);
  const blob = new Blob([contents], { type: "application/json" });
  const createObjectUrl = URL.createObjectURL?.bind(URL);
  if (!createObjectUrl) {
    triggerDownload(
      `data:application/json;charset=utf-8,${encodeURIComponent(contents)}`,
      `director-task-${taskId}.json`,
    );
    return;
  }
  const url = createObjectUrl(blob);
  triggerDownload(url, `director-task-${taskId}.json`);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function groupTasks(
  tasks: TaskDrawerTask[],
  sort: TaskSort,
  now: number,
): Array<{ key: string; label: string; tasks: TaskDrawerTask[] }> {
  const chronological = [...tasks].sort(
    (left, right) =>
      (timestamp(right.created_at) ?? 0) - (timestamp(left.created_at) ?? 0),
  );
  const groups = new Map<string, TaskDrawerTask[]>();
  chronological.forEach((task) => {
    const key = localDateKey(task.created_at);
    const group = groups.get(key) ?? [];
    group.push(task);
    groups.set(key, group);
  });
  if (sort === "duration") {
    groups.forEach((group) => {
      group.sort(
        (left, right) =>
          (taskDurationSeconds(right, now) ?? -1) -
          (taskDurationSeconds(left, now) ?? -1),
      );
    });
  }
  const date = new Date(now);
  return [...groups].map(([key, group]) => ({
    key,
    label: dateGroupLabel(key, date),
    tasks: group,
  }));
}

function useModalLifecycle(onClose: () => void, restoreFocus?: HTMLElement | null) {
  const closeRef = useRef(onClose);
  const dialogRef = useRef<HTMLElement | null>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  closeRef.current = onClose;

  useEffect(() => {
    restoreFocusRef.current = restoreFocus ?? (
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    );
    const overlay = dialogRef.current?.parentElement ?? null;
    const background = [...document.body.children]
      .filter((element): element is HTMLElement =>
        element instanceof HTMLElement && element !== overlay,
      )
      .map((element) => ({
        element: element as HTMLElement & { inert: boolean },
        inert: (element as HTMLElement & { inert: boolean }).inert,
      }));
    const previousOverflow = document.body.style.overflow;
    background.forEach(({ element }) => { element.inert = true; });
    document.body.style.overflow = "hidden";
    const focusDialog = window.requestAnimationFrame(() => {
      dialogRef.current?.querySelector<HTMLElement>(
        "button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex='-1'])",
      )?.focus();
    });
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = [...dialogRef.current.querySelectorAll<HTMLElement>(
        "button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex='-1'])",
      )].filter((element) => element.tabIndex >= 0);
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable.at(-1)!;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(focusDialog);
      document.removeEventListener("keydown", handleKeyDown);
      background.forEach(({ element, inert }) => { element.inert = inert; });
      document.body.style.overflow = previousOverflow;
      window.requestAnimationFrame(() => restoreFocusRef.current?.focus());
    };
  }, []);

  return dialogRef;
}

function taskMatchesSearch(task: TaskDrawerTask, query: string): boolean {
  if (!query) return true;
  const searchable = [
    taskTitle(task),
    taskModeLabel(task),
    task.id,
    task.stage ?? "",
    task.error ?? "",
    ...(task.output_files ?? []),
    ...task.segment_results.map((result) => result.output_file),
  ].join("\n").toLocaleLowerCase("zh-CN");
  return searchable.includes(query.toLocaleLowerCase("zh-CN"));
}

function TaskThumbnail({
  task,
  currentActive,
}: {
  task: GenerationTask;
  currentActive: boolean;
}) {
  const [previewToken, setPreviewToken] = useState(0);
  const showLivePreview = Boolean(
    currentActive &&
    task.live_preview_url &&
    RUNNING_STATUSES.has(task.status),
  );
  useEffect(() => {
    if (!showLivePreview) return;
    setPreviewToken(Date.now());
    const timer = window.setInterval(() => setPreviewToken(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [showLivePreview, task.live_preview_url]);
  if (showLivePreview && task.live_preview_url) {
    const separator = task.live_preview_url.includes("?") ? "&" : "?";
    return (
      <span className="task-row__thumb">
        <img src={`${task.live_preview_url}${separator}frame=${previewToken}`} alt="" />
      </span>
    );
  }
  if (task.status === "succeeded" && task.preview_url) {
    return (
      <span className="task-row__thumb">
        <video src={task.preview_url} muted playsInline preload="metadata" />
      </span>
    );
  }
  return (
    <span
      className={`task-row__thumb task-row__thumb--${task.status}`}
      aria-hidden="true"
    >
      {STATUS_MARK[task.status]}
    </span>
  );
}

function TaskDetails({
  task,
  now,
  queuedAhead,
  estimatedWaitSeconds,
}: {
  task: TaskDrawerTask;
  now: number;
  queuedAhead: number | null;
  estimatedWaitSeconds: number | null;
}) {
  return (
    <section
      className="task-row__details"
      role="tooltip"
      aria-label={`任务 ${task.id.slice(0, 8)} 的详情`}
    >
      <header>
        <strong>{taskTitle(task)}</strong>
        <span>{STATUS_LABEL[task.status]}</span>
      </header>
      <dl>
        <div><dt>创建时间</dt><dd>{formatTime(task.created_at)}</dd></div>
        <div><dt>执行耗时</dt><dd>{formatDuration(taskDurationSeconds(task, now))}</dd></div>
        <div><dt>完成进度</dt><dd>{Math.round(Math.max(0, Math.min(1, task.progress)) * 100)}%</dd></div>
        <div><dt>任务编号</dt><dd title={task.id}>{task.id}</dd></div>
      </dl>
      {queuedAhead !== null && (
        <p className="task-row__queue-estimate">
          前方 {queuedAhead} 个任务
          {estimatedWaitSeconds === null
            ? ""
            : ` · 预计等待约 ${formatDuration(estimatedWaitSeconds)}`}
        </p>
      )}
      {task.stage && <p className="task-row__stage-detail">{task.stage}</p>}
      {task.children.length > 0 && (
        <div
          className="task-row__children"
          role="region"
          aria-label={`任务 ${task.id.slice(0, 8)} 的分段子任务`}
        >
          <strong>{task.children.length} 个分段执行单元</strong>
          {task.children.map((child) => (
            <div key={child.id}>
              <span>
                {child.family === "fl2va" ? "FL2VA" : "Ref2VA"}
                {` · ${child.backend === "raylight" ? "RayLight" : "标准执行"}`}
              </span>
              <em>{STATUS_LABEL[child.status]} · {Math.round(child.progress * 100)}%</em>
              <small>{child.segment_ids.length} 段{child.stage ? ` · ${child.stage}` : ""}</small>
            </div>
          ))}
        </div>
      )}
      {task.output_files && task.output_files.length > 0 && (
        <div className="task-row__file-list">
          {task.output_files.map((path, index) => (
            <code key={`${path}-${index}`}>{displayOutputPath(path)}</code>
          ))}
        </div>
      )}
      {task.error && <pre>{task.error}</pre>}
    </section>
  );
}

function MediaLightbox({
  task,
  initialIndex,
  onClose,
  onImportOutput,
  restoreFocus,
}: {
  task: GenerationTask;
  initialIndex: number;
  onClose: () => void;
  onImportOutput?: (
    taskId: string,
    output: { index: number; segmentId?: string },
  ) => void | Promise<void>;
  restoreFocus?: HTMLElement | null;
}) {
  const outputs = useMemo(() => getMediaOutputs(task), [task]);
  const [index, setIndex] = useState(() =>
    Math.max(0, Math.min(initialIndex, Math.max(0, outputs.length - 1))),
  );
  const output = outputs[index] ?? null;
  const dialogRef = useModalLifecycle(onClose, restoreFocus);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft" && index > 0) setIndex(index - 1);
      if (event.key === "ArrowRight" && index < outputs.length - 1) setIndex(index + 1);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [index, outputs.length]);

  if (!output) return null;
  const kind = mediaKind(output);
  return createPortal(
    <div
      className="task-lightbox"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        ref={dialogRef}
        tabIndex={-1}
        className="task-lightbox__dialog"
        role="dialog"
        aria-modal="true"
        aria-label="任务输出查看器"
      >
        <header>
          <div>
            <strong>{taskTitle(task as TaskDrawerTask)}</strong>
            <small>{output.file ? displayOutputPath(output.file) : `输出 ${index + 1}`}</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="关闭查看器">×</button>
        </header>
        <div className="task-lightbox__stage">
          {kind === "video" && (
            <video
              key={output.url}
              className="task-lightbox__media"
              src={output.url}
              controls
              autoPlay
              playsInline
              preload="metadata"
              style={{ objectFit: "contain", objectPosition: "center" }}
            />
          )}
          {kind === "image" && (
            <img
              className="task-lightbox__media"
              src={output.url}
              alt={`输出 ${index + 1}`}
              style={{ objectFit: "contain", objectPosition: "center" }}
            />
          )}
          {kind === "audio" && (
            <div className="task-lightbox__audio"><span>♫</span><audio src={output.url} controls autoPlay /></div>
          )}
          {kind === "text" && (
            <iframe src={output.url} title={`文本输出 ${index + 1}`} sandbox="" />
          )}
          {kind === "other" && (
            <div className="task-lightbox__unsupported">
              <strong>此格式请在新窗口中查看</strong>
              <button type="button" onClick={() => openInNewWindow(output.url)}>打开输出</button>
            </div>
          )}
        </div>
        <footer>
          <div className="task-lightbox__paging">
            <button type="button" onClick={() => setIndex(index - 1)} disabled={index === 0} aria-label="上一个输出">‹</button>
            <span aria-live="polite">{index + 1} / {outputs.length}</span>
            <button type="button" onClick={() => setIndex(index + 1)} disabled={index === outputs.length - 1} aria-label="下一个输出">›</button>
          </div>
          <div className="task-lightbox__actions">
            {onImportOutput && (
              <button type="button" onClick={() => void onImportOutput(task.id, {
                index: output.index,
                segmentId: output.segmentId,
              })}>加入素材库</button>
            )}
            <button type="button" onClick={() => triggerDownload(output.url, output.file)}>下载</button>
            <button type="button" onClick={() => openInNewWindow(output.url)}>新窗口打开</button>
          </div>
        </footer>
      </section>
    </div>,
    document.body,
  );
}

function GenerationDetailsDialog({
  state,
  onClose,
  restoreFocus,
}: {
  state: GenerationDetailsState;
  onClose: () => void;
  restoreFocus?: HTMLElement | null;
}) {
  const dialogRef = useModalLifecycle(onClose, restoreFocus);
  const details = state.details;
  return createPortal(
    <div className="task-parameter-dialog" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label="任务生成参数"
      >
        <header>
          <div>
            <strong>任务生成参数</strong>
            <small>{details?.project_title ?? `任务 ${state.taskId.slice(0, 8)}`}</small>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="关闭生成参数">×</button>
        </header>
        <div className="task-parameter-dialog__body">
          {state.loading && <div className="task-parameter-dialog__state" role="status"><Spinner /> 正在读取任务快照…</div>}
          {state.error && <div className="task-parameter-dialog__state is-error" role="alert">{state.error}</div>}
          {details && (
            <>
              <section className="task-parameter-section">
                <h3>输出规格</h3>
                <dl className="task-parameter-grid">
                  <div><dt>画面</dt><dd>{details.render.width} × {details.render.height}</dd></div>
                  <div><dt>帧率</dt><dd>{details.render.fps} fps</dd></div>
                  <div><dt>本次总时长</dt><dd>{formatDuration(details.render.total_duration_seconds)}</dd></div>
                  <div><dt>导出</dt><dd>{details.render.export_mode === "all" ? "整片与分段" : "仅分段"}</dd></div>
                </dl>
              </section>

              <section className="task-parameter-section">
                <h3>采样</h3>
                <div className="task-parameter-cards">
                  {details.sampling.map((sampling) => (
                    <article key={sampling.family}>
                      <h4>{sampling.family === "fl2va" ? "FL2VA" : "Ref2VA"}</h4>
                      <dl className="task-parameter-grid">
                        <div><dt>步数</dt><dd>{sampling.steps}</dd></div>
                        <div><dt>Seed</dt><dd title={String(sampling.seed)}>{sampling.seed}</dd></div>
                        <div><dt>Seed 意图</dt><dd>{sampling.random_seed ? "提交时随机（上方为实际值）" : "固定"}</dd></div>
                        <div><dt>采样器</dt><dd>{sampling.sampler}</dd></div>
                        <div><dt>调度器</dt><dd>{sampling.scheduler}</dd></div>
                        <div><dt>视频 / 音频 shift</dt><dd>{sampling.shift} / {sampling.audio_shift}</dd></div>
                      </dl>
                    </article>
                  ))}
                </div>
              </section>

              <section className="task-parameter-section">
                <h3>模型与执行</h3>
                {!details.runtime_snapshot_available && (
                  <p className="task-parameter-note">这条旧任务没有可兼容的运行参数快照；仍可查看输出、采样和分段参数。</p>
                )}
                <div className="task-parameter-cards">
                  {details.models.map((model) => (
                    <article key={model.family}>
                      <h4>{model.family === "fl2va" ? "FL2VA" : "Ref2VA"}</h4>
                      <dl className="task-parameter-grid">
                        <div className="is-wide"><dt>扩散模型</dt><dd title={model.filename}>{model.filename}</dd></div>
                        <div><dt>执行后端</dt><dd>{model.backends.map((backend) => backend === "raylight" ? "RayLight" : "标准执行").join("、")}</dd></div>
                        <div><dt>加载设备</dt><dd>{model.device}</dd></div>
                        <div className="is-wide"><dt>LoRA</dt><dd>{model.lora_name ? `${model.lora_name}（强度 ${model.lora_strength}）` : "未使用"}</dd></div>
                        {model.logical_gpu_indices.length > 0 && (
                          <div className="is-wide"><dt>逻辑 GPU</dt><dd>{model.logical_gpu_indices.join(", ")} · Ulysses {model.ulysses_degree} · Ring {model.ring_degree}</dd></div>
                        )}
                      </dl>
                    </article>
                  ))}
                </div>
                {details.shared_models.length > 0 && (
                  <dl className="task-parameter-shared">
                    {details.shared_models.map((model) => (
                      <div key={model.role}>
                        <dt>{{ clip: "CLIP", video_vae: "视频 VAE", audio_vae: "音频 VAE" }[model.role]}</dt>
                        <dd title={model.filename}>{model.filename} · {model.device}</dd>
                      </div>
                    ))}
                  </dl>
                )}
              </section>

              <section className="task-parameter-section">
                <h3>本次生成分段（{details.segments.length}）</h3>
                <div className="task-parameter-segments">
                  {details.segments.map((segment, index) => {
                    const references = [
                      segment.has_first_image ? "首帧图" : "",
                      segment.has_last_image ? "尾帧图" : "",
                      segment.has_source_video ? "源视频" : "",
                      segment.reference_image_count ? `${segment.reference_image_count} 张参考图` : "",
                      segment.reference_audio_count ? `${segment.reference_audio_count} 条参考音频` : "",
                      segment.reference_video_count ? `${segment.reference_video_count} 路参考视频` : "",
                      segment.source_audio_as_reference ? "源视频音频作参考" : "",
                    ].filter(Boolean);
                    return (
                      <article key={segment.id}>
                        <header>
                          <div><strong>{segment.title || `分段 ${index + 1}`}</strong><small>{segment.family === "fl2va" ? "FL2VA" : "Ref2VA"} · {segment.recipe.toUpperCase()}</small></div>
                          <span>{formatDuration(segment.duration_seconds)}</span>
                        </header>
                        <p>{references.length > 0 ? references.join(" · ") : "无参考素材"}</p>
                        <p>音频：{{ generate: "生成音频", source: "使用源音频", mute: "静音" }[segment.audio_mode]} · 参考图：{segment.ref_image_size === "max" ? "最高保真" : "匹配画布"}</p>
                        <p>{segment.continuity_enabled ? `连续生成 · 重叠 ${segment.continuity_overlap_frames} 帧` : "独立生成"}</p>
                        <details>
                          <summary>提示词</summary>
                          <pre>{segment.prompt || "（空）"}</pre>
                          <button type="button" onClick={() => void copyText(segment.prompt)}>复制提示词</button>
                        </details>
                      </article>
                    );
                  })}
                </div>
              </section>
            </>
          )}
        </div>
      </section>
    </div>,
    document.body,
  );
}

function ErrorDialog({
  task,
  onClose,
  restoreFocus,
}: {
  task: GenerationTask;
  onClose: () => void;
  restoreFocus?: HTMLElement | null;
}) {
  const dialogRef = useModalLifecycle(onClose, restoreFocus);
  return createPortal(
    <div className="task-error-dialog" role="presentation" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section ref={dialogRef} tabIndex={-1} role="dialog" aria-modal="true" aria-label="任务错误详情">
        <header>
          <strong>任务错误详情</strong>
          <button type="button" className="icon-button" onClick={onClose} aria-label="关闭错误详情">×</button>
        </header>
        <pre>{task.error || "没有可显示的错误详情。"}</pre>
        <footer>
          <button type="button" onClick={() => void copyText(task.error || "")}>复制错误详情</button>
        </footer>
      </section>
    </div>,
    document.body,
  );
}

export function TaskDrawer(props: TaskDrawerProps) {
  const {
    id,
    open,
    tasks: rawTasks,
    loading,
    supportsCancel = false,
    deletingTaskIds = new Set(),
    clearing = false,
    currentProjectId,
    onClose,
    onCancel,
    onConfirmComfyRestart,
    onBulkCancel,
    onDelete,
    onClearCompleted,
    onLoadProject,
    onExportProjectConfig,
    onExportDiagnostic,
    onLoadGenerationDetails,
    onImportOutput,
  } = props;
  const tasks = rawTasks as TaskDrawerTask[];
  const [initialPreferences] = useState(loadTimelineWorkspacePreferences);
  const [tab, setTab] = useState<TaskTab>(initialPreferences.taskTab);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [currentProjectOnly, setCurrentProjectOnly] = useState(
    initialPreferences.taskCurrentProjectOnly,
  );
  const [sort, setSort] = useState<TaskSort>(initialPreferences.taskSort);
  const [hoveredTaskId, setHoveredTaskId] = useState<string | null>(null);
  const [menu, setMenu] = useState<MenuState | null>(null);
  const [viewer, setViewer] = useState<{ taskId: string; index: number } | null>(null);
  const [errorTaskId, setErrorTaskId] = useState<string | null>(null);
  const [generationDetails, setGenerationDetails] = useState<GenerationDetailsState | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const hoverShowTimer = useRef<number | null>(null);
  const hoverHideTimer = useRef<number | null>(null);
  const drawerRef = useRef<HTMLElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const menuTriggerRef = useRef<HTMLElement | null>(null);
  const modalReturnFocusRef = useRef<HTMLElement | null>(null);
  const generationDetailsRequestRef = useRef(0);

  const queuedTasks = tasks.filter((task) => task.status === "queued");
  const runningTasks = tasks.filter((task) => RUNNING_STATUSES.has(task.status));
  const failedTasks = tasks.filter((task) => ["failed", "cancelled"].includes(task.status));
  const completedTasks = tasks.filter((task) => task.status === "succeeded");
  const terminalCount = tasks.filter((task) => TERMINAL_STATUSES.has(task.status)).length;
  const projectFilterAvailable = tasks.some(
    (task) =>
      typeof task.current_project === "boolean" ||
      Boolean(currentProjectId && task.project_id) ||
      task.segment_results.length > 0,
  );

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedSearch(search.trim()), 150);
    return () => window.clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    updateTimelineWorkspacePreferences({
      taskTab: tab,
      taskCurrentProjectOnly: currentProjectOnly,
      taskSort: sort,
    });
  }, [currentProjectOnly, sort, tab]);

  useEffect(() => {
    if (!open || !tasks.some((task) => ACTIVE_STATUSES.has(task.status))) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [open, tasks]);

  useEffect(() => {
    if (failedTasks.length === 0 && tab === "failed") setTab("all");
  }, [failedTasks.length, tab]);

  useEffect(() => {
    if (!open) return;
    const closeDrawerOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      if (document.querySelector(".task-lightbox,.task-error-dialog,.task-parameter-dialog,.task-action-menu")) return;
      event.preventDefault();
      onClose();
      window.requestAnimationFrame(() => document.getElementById("task-panel-toggle")?.focus());
    };
    document.addEventListener("keydown", closeDrawerOnEscape);
    return () => document.removeEventListener("keydown", closeDrawerOnEscape);
  }, [onClose, open]);

  useEffect(() => {
    if (!open) return;
    const closeDrawerOnOutsidePointer = (event: PointerEvent) => {
      if (event.button !== 0 || !(event.target instanceof Element)) return;
      if (drawerRef.current?.contains(event.target)) return;
      if (event.target.closest(
        "#task-panel-toggle,.task-lightbox,.task-error-dialog,.task-parameter-dialog,.task-action-menu",
      )) return;
      onClose();
    };
    document.addEventListener("pointerdown", closeDrawerOnOutsidePointer, true);
    return () => document.removeEventListener("pointerdown", closeDrawerOnOutsidePointer, true);
  }, [onClose, open]);

  useEffect(() => () => {
    if (hoverShowTimer.current !== null) window.clearTimeout(hoverShowTimer.current);
    if (hoverHideTimer.current !== null) window.clearTimeout(hoverHideTimer.current);
  }, []);

  useEffect(() => {
    if (!menu) return;
    const focusMenu = window.requestAnimationFrame(() => {
      menuRef.current?.querySelector<HTMLElement>("button:not(:disabled)")?.focus();
    });
    const closeMenu = () => setMenu(null);
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeMenu();
        window.requestAnimationFrame(() => menuTriggerRef.current?.focus());
        return;
      }
      if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
      const items = [...(menuRef.current?.querySelectorAll<HTMLElement>(
        "button:not(:disabled)",
      ) ?? [])];
      if (items.length === 0) return;
      event.preventDefault();
      const current = Math.max(0, items.indexOf(document.activeElement as HTMLElement));
      const next = event.key === "Home"
        ? 0
        : event.key === "End"
          ? items.length - 1
          : event.key === "ArrowDown"
            ? (current + 1) % items.length
            : (current - 1 + items.length) % items.length;
      items[next].focus();
    };
    window.addEventListener("pointerdown", closeMenu);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(focusMenu);
      window.removeEventListener("pointerdown", closeMenu);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [menu]);

  const visibleTasks = useMemo(() => {
    return tasks.filter((task) => {
      if (tab === "completed" && task.status !== "succeeded") return false;
      if (tab === "failed" && !["failed", "cancelled"].includes(task.status)) return false;
      if (currentProjectOnly && !belongsToCurrentProject(task, currentProjectId)) return false;
      return taskMatchesSearch(task, debouncedSearch);
    });
  }, [currentProjectId, currentProjectOnly, debouncedSearch, tab, tasks]);
  const groups = useMemo(
    () => groupTasks(visibleTasks, sort, now),
    [now, sort, visibleTasks],
  );
  const currentActiveTaskId = useMemo(
    () => [...runningTasks].sort(
      (left, right) =>
        (timestamp(left.started_at ?? left.created_at) ?? 0) -
        (timestamp(right.started_at ?? right.created_at) ?? 0),
    )[0]?.id ?? null,
    [runningTasks],
  );
  const averageCompletedDuration = useMemo(() => {
    const durations = completedTasks
      .map((task) => taskDurationSeconds(task, now))
      .filter((value): value is number => value !== null && value > 0);
    return durations.length > 0
      ? durations.reduce((total, value) => total + value, 0) / durations.length
      : null;
  }, [completedTasks, now]);
  const queueOrder = useMemo(
    () => [...queuedTasks].sort(
      (left, right) =>
        (timestamp(left.created_at) ?? 0) - (timestamp(right.created_at) ?? 0),
    ),
    [queuedTasks],
  );
  const viewerTask = viewer
    ? tasks.find((task) => task.id === viewer.taskId) ?? null
    : null;
  const errorTask = errorTaskId
    ? tasks.find((task) => task.id === errorTaskId) ?? null
    : null;
  const menuTask = menu
    ? tasks.find((task) => task.id === menu.taskId) ?? null
    : null;

  if (!open) return null;

  const runBatchCancel = async (batch: TaskDrawerTask[]) => {
    if (batch.length === 0 || bulkBusy) return;
    setBulkBusy(true);
    try {
      if (onBulkCancel) {
        await onBulkCancel(batch.map((task) => task.id));
      } else {
        await Promise.all(batch.map((task) => onCancel(task.id)));
      }
    } finally {
      setBulkBusy(false);
    }
  };

  const closeGenerationDetails = () => {
    generationDetailsRequestRef.current += 1;
    setGenerationDetails(null);
  };
  const openGenerationDetails = (taskId: string) => {
    if (!onLoadGenerationDetails) return;
    const requestId = generationDetailsRequestRef.current + 1;
    generationDetailsRequestRef.current = requestId;
    setGenerationDetails({ taskId, loading: true, details: null, error: null });
    setMenu(null);
    void Promise.resolve()
      .then(() => onLoadGenerationDetails(taskId))
      .then((details) => {
        if (generationDetailsRequestRef.current !== requestId) return;
        setGenerationDetails({ taskId, loading: false, details, error: null });
      })
      .catch((reason: unknown) => {
        if (generationDetailsRequestRef.current !== requestId) return;
        setGenerationDetails({
          taskId,
          loading: false,
          details: null,
          error: reason instanceof Error ? reason.message : "任务生成参数读取失败",
        });
      });
  };

  const showHoverDetails = (taskId: string) => {
    if (hoverHideTimer.current !== null) window.clearTimeout(hoverHideTimer.current);
    if (hoverShowTimer.current !== null) window.clearTimeout(hoverShowTimer.current);
    hoverShowTimer.current = window.setTimeout(() => setHoveredTaskId(taskId), 200);
  };
  const hideHoverDetails = () => {
    if (hoverShowTimer.current !== null) window.clearTimeout(hoverShowTimer.current);
    if (hoverHideTimer.current !== null) window.clearTimeout(hoverHideTimer.current);
    hoverHideTimer.current = window.setTimeout(() => setHoveredTaskId(null), 150);
  };
  const openMenu = (
    task: GenerationTask,
    event: ReactMouseEvent<HTMLElement>,
    fromContextMenu = false,
  ) => {
    event.preventDefault();
    event.stopPropagation();
    const currentTarget = event.currentTarget;
    menuTriggerRef.current = currentTarget instanceof HTMLButtonElement
      ? currentTarget
      : currentTarget.querySelector<HTMLElement>(".task-row__more");
    setHoveredTaskId(null);
    const rect = event.currentTarget.getBoundingClientRect();
    const desiredX = fromContextMenu ? event.clientX : rect.right - 218;
    const desiredY = fromContextMenu ? event.clientY : rect.bottom + 4;
    const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
    const viewportHeight = document.documentElement.clientHeight || window.innerHeight;
    setMenu({
      taskId: task.id,
      x: Math.max(8, Math.min(desiredX, viewportWidth - 226)),
      y: Math.max(8, Math.min(desiredY, viewportHeight - 330)),
    });
  };

  return (
    <aside
      ref={drawerRef}
      id={id}
      className="task-drawer is-open"
      style={{ top: "var(--topbar-height, 52px)", bottom: 0 }}
      aria-label="任务列表"
    >
      <header className="task-drawer__header">
        <div className="task-drawer__heading">
          <h2>任务历史</h2>
          {runningTasks.length > 0 || queuedTasks.length > 0 ? (
            <span aria-live="polite">{runningTasks.length} 个运行中 · {queuedTasks.length} 个排队中</span>
          ) : (
            <span aria-live="polite">没有活动任务</span>
          )}
        </div>
        <div className="task-drawer__header-actions" aria-label="活动任务操作">
          <button
            type="button"
            disabled={!supportsCancel || queuedTasks.length === 0 || bulkBusy}
            onClick={() => {
              if (window.confirm(`确定取消 ${queuedTasks.length} 个导演台排队任务吗？不会影响 ComfyUI 中其他来源的任务。`)) {
                void runBatchCancel(queuedTasks);
              }
            }}
          >
            清空等待
          </button>
          <button
            type="button"
            disabled={!supportsCancel || runningTasks.length === 0 || bulkBusy}
            onClick={() => {
              if (window.confirm(`确定取消 ${runningTasks.length} 个导演台运行任务吗？不会影响 ComfyUI 中其他来源的任务。`)) {
                void runBatchCancel(runningTasks);
              }
            }}
          >
            取消运行中
          </button>
          <button type="button" className="icon-button" onClick={() => {
            onClose();
            window.requestAnimationFrame(() => document.getElementById("task-panel-toggle")?.focus());
          }} aria-label="关闭任务列表">×</button>
        </div>
      </header>

      <div className="task-drawer__filters">
        <label className="task-search">
          <span aria-hidden="true">⌕</span>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="搜索任务"
            aria-label="搜索任务"
          />
          {search && <button type="button" onClick={() => setSearch("")} aria-label="清空搜索">×</button>}
        </label>
        <div className="task-filter-actions">
          <button
            type="button"
            className={currentProjectOnly ? "is-active" : ""}
            aria-pressed={currentProjectOnly}
            disabled={!projectFilterAvailable}
            title={projectFilterAvailable ? undefined : "等待任务列表提供项目归属"}
            onClick={() => setCurrentProjectOnly((value) => !value)}
          >
            当前项目
          </button>
          <label>
            <span className="sr-only">任务排序</span>
            <select value={sort} onChange={(event) => setSort(event.target.value as TaskSort)} aria-label="任务排序">
              <option value="recent">最新任务</option>
              <option value="duration">总耗时</option>
            </select>
          </label>
        </div>
      </div>

      <nav className="task-tabs" role="tablist" aria-label="任务状态筛选">
        <button type="button" role="tab" aria-selected={tab === "all"} onClick={() => setTab("all")}>全部 <span>{tasks.length}</span></button>
        <button type="button" role="tab" aria-selected={tab === "completed"} onClick={() => setTab("completed")}>已完成 <span>{completedTasks.length}</span></button>
        {failedTasks.length > 0 && (
          <button type="button" role="tab" aria-selected={tab === "failed"} onClick={() => setTab("failed")}>失败 <span>{failedTasks.length}</span></button>
        )}
      </nav>

      <div className="task-list" aria-busy={loading}>
        {loading && tasks.length === 0 && (
          <div className="task-list__loading"><Spinner label="加载任务历史" /><span>正在加载任务历史</span></div>
        )}
        {!loading && tasks.length === 0 && (
          <div className="empty-slate">
            <span className="empty-slate__mark">✓</span>
            <strong>还没有任务记录</strong>
            <p>提交生成后，任务会显示在这里。</p>
          </div>
        )}
        {tasks.length > 0 && groups.length === 0 && (
          <div className="empty-slate">
            <span className="empty-slate__mark">⌕</span>
            <strong>没有符合条件的任务</strong>
            <p>调整搜索词或筛选条件后再试。</p>
          </div>
        )}
        {groups.map((group) => (
          <section className="task-date-group" key={group.key} aria-labelledby={`task-date-${group.key}`}>
            <header id={`task-date-${group.key}`}><span>{group.label}</span><small>{group.tasks.length}</small></header>
            <div>
              {group.tasks.map((task) => {
                const cancellable = ACTIVE_STATUSES.has(task.status);
                const deleting = deletingTaskIds.has(task.id);
                const progressPercent = Math.round(Math.max(0, Math.min(1, task.progress)) * 100);
                const isCurrentActive = task.id === currentActiveTaskId;
                const created = timestamp(task.created_at);
                const recentlyQueued = task.status === "queued" && created !== null && now - created < 3000;
                const stateText = task.status === "queued"
                  ? recentlyQueued ? "已加入队列" : "排队中"
                  : RUNNING_STATUSES.has(task.status)
                    ? isCurrentActive ? task.stage || STATUS_LABEL[task.status] : STATUS_LABEL[task.status]
                    : task.status === "succeeded"
                      ? `${task.output_files?.[0]?.split(/[\\/]/).at(-1) ?? "生成完成"} · ${formatDuration(taskDurationSeconds(task, now))}`
                      : STATUS_LABEL[task.status];
                const queueIndex = queueOrder.findIndex((candidate) => candidate.id === task.id);
                const runningWorkAhead = runningTasks.reduce(
                  (total, candidate) => total + Math.max(
                    0,
                    1 - Math.max(0, Math.min(1, candidate.progress)),
                  ),
                  0,
                );
                const queuedAhead = queueIndex < 0
                  ? null
                  : runningTasks.length + queueIndex;
                const estimatedWaitSeconds = queueIndex < 0 || averageCompletedDuration === null
                  ? null
                  : (runningWorkAhead + queueIndex) * averageCompletedDuration;
                const outputs = getMediaOutputs(task);
                return (
                  <article
                    className={`task-row task-row--${task.status}`}
                    key={task.id}
                    onMouseEnter={() => showHoverDetails(task.id)}
                    onMouseLeave={hideHoverDetails}
                    onFocus={() => showHoverDetails(task.id)}
                    onBlur={(event) => {
                      if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                        hideHoverDetails();
                      }
                    }}
                    onContextMenu={(event) => openMenu(task, event, true)}
                    onClick={(event) => {
                      if (task.status === "succeeded" && outputs.length > 0) {
                        modalReturnFocusRef.current = event.currentTarget.querySelector<HTMLElement>(
                          ".task-row__more",
                        );
                        setViewer({ taskId: task.id, index: 0 });
                      }
                    }}
                  >
                    <TaskThumbnail task={task} currentActive={isCurrentActive} />
                    <div className="task-row__copy">
                      <div><strong>{taskTitle(task)}</strong><span>{taskModeLabel(task)}</span></div>
                      <small title={task.stage ?? undefined}>{stateText}</small>
                    </div>
                    <div className="task-row__actions">
                      {cancellable && (
                        <button
                          type="button"
                          disabled={!supportsCancel}
                          title={supportsCancel ? undefined : "当前 ComfyUI 版本不支持安全的定向取消"}
                          onClick={(event) => {
                            event.stopPropagation();
                            void onCancel(task.id);
                          }}
                        >
                          {task.status === "cancelling" ? "重试取消" : "取消"}
                        </button>
                      )}
                      {["failed", "cancelled"].includes(task.status) && (
                        <button
                          type="button"
                          className="is-danger"
                          disabled={deleting || clearing}
                          aria-label={`移除任务 ${task.id.slice(0, 8)}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            if (confirmRecordDeletion(`确定移除任务 ${task.id.slice(0, 8)} 的记录吗？`)) {
                              void onDelete(task.id);
                            }
                          }}
                        >
                          {deleting ? "移除中" : "移除"}
                        </button>
                      )}
                      {task.status === "succeeded" && outputs.length > 0 && (
                        <button type="button" onClick={(event) => {
                          event.stopPropagation();
                          modalReturnFocusRef.current = event.currentTarget;
                          setViewer({ taskId: task.id, index: 0 });
                        }}>查看</button>
                      )}
                      <button
                        type="button"
                        className="task-row__more"
                        aria-label={`任务 ${task.id.slice(0, 8)} 的更多操作`}
                        aria-haspopup="menu"
                        aria-expanded={menu?.taskId === task.id}
                        aria-controls={menu?.taskId === task.id ? `task-actions-${task.id}` : undefined}
                        onClick={(event) => openMenu(task, event)}
                      >
                        •••
                      </button>
                    </div>
                    {isCurrentActive && RUNNING_STATUSES.has(task.status) && (
                      <span className="task-row__progress" aria-label={`完成 ${progressPercent}%`}><i style={{ width: `${progressPercent}%` }} /></span>
                    )}
                    {hoveredTaskId === task.id && !menu && (
                      <TaskDetails
                        task={task}
                        now={now}
                        queuedAhead={queuedAhead}
                        estimatedWaitSeconds={estimatedWaitSeconds}
                      />
                    )}
                  </article>
                );
              })}
            </div>
          </section>
        ))}
      </div>

      <footer className="task-drawer__footer">
        <small>移除记录不会删除 ComfyUI 输出文件。</small>
        <button
          type="button"
          disabled={loading || clearing || deletingTaskIds.size > 0}
          onClick={() => {
            if (confirmRecordDeletion(`确定清空所有已结束任务吗？活动任务会保留。当前列表中有 ${terminalCount} 个已结束任务。`)) {
              void onClearCompleted();
            }
          }}
        >
          {clearing ? <Spinner label="清空任务历史" /> : null}
          清空历史记录
        </button>
      </footer>

      {menu && menuTask && (
        <div
          id={`task-actions-${menuTask.id}`}
          ref={menuRef}
          className="task-action-menu"
          role="menu"
          aria-label={`任务 ${menuTask.id.slice(0, 8)} 的操作`}
          style={{ left: menu.x, top: menu.y }}
          onPointerDown={(event) => event.stopPropagation()}
        >
          {menuTask.status === "succeeded" && getMediaOutputs(menuTask).length > 0 && (
            <>
              <button type="button" role="menuitem" onClick={() => {
                modalReturnFocusRef.current = menuTriggerRef.current;
                setViewer({ taskId: menuTask.id, index: 0 });
                setMenu(null);
              }}>查看结果</button>
              <button type="button" role="menuitem" disabled={!onImportOutput} onClick={() => {
                const output = getMediaOutputs(menuTask)[0];
                if (output) void onImportOutput?.(menuTask.id, {
                  index: output.index,
                  segmentId: output.segmentId,
                });
                setMenu(null);
              }}>加入素材库</button>
              <button type="button" role="menuitem" onClick={() => {
                const output = getMediaOutputs(menuTask)[0];
                if (output) triggerDownload(output.url, output.file);
                setMenu(null);
              }}>下载输出</button>
              <button type="button" role="menuitem" onClick={() => {
                const output = getMediaOutputs(menuTask)[0];
                if (output) openInNewWindow(output.url);
                setMenu(null);
              }}>在新窗口打开</button>
              <hr />
            </>
          )}
          <button type="button" role="menuitem" disabled={!onLoadProject} onClick={() => {
            void onLoadProject?.(menuTask.id);
            setMenu(null);
          }}>另存为新项目</button>
          <button type="button" role="menuitem" disabled={!onExportProjectConfig} onClick={() => {
            const taskId = menuTask.id;
            setMenu(null);
            void onExportProjectConfig?.(taskId).then((snapshot) => {
              downloadHistoricalProjectConfig(taskId, snapshot);
            }).catch(() => undefined);
          }}>导出配置</button>
          <button type="button" role="menuitem" disabled={!onLoadGenerationDetails} onClick={() => {
            modalReturnFocusRef.current = menuTriggerRef.current;
            openGenerationDetails(menuTask.id);
          }}>查看生成参数</button>
          <button type="button" role="menuitem" disabled={!onExportDiagnostic} onClick={() => {
            const taskId = menuTask.id;
            setMenu(null);
            void onExportDiagnostic?.(taskId).then((diagnostic) => {
              downloadDiagnostic(taskId, diagnostic);
            }).catch(() => undefined);
          }}>导出脱敏诊断</button>
          <button type="button" role="menuitem" onClick={() => {
            void copyText(menuTask.id);
            setMenu(null);
          }}>复制任务 ID</button>
          {menuTask.error && (
            <>
              <button type="button" role="menuitem" onClick={() => {
                modalReturnFocusRef.current = menuTriggerRef.current;
                setErrorTaskId(menuTask.id);
                setMenu(null);
              }}>查看错误详情</button>
              <button type="button" role="menuitem" onClick={() => {
                void copyText(menuTask.error ?? "");
                setMenu(null);
              }}>复制错误详情</button>
            </>
          )}
          {canConfirmComfyRestart(menuTask) && (
            <button
              type="button"
              role="menuitem"
              className="is-danger"
              disabled={!onConfirmComfyRestart}
              onClick={() => {
                if (confirmComfyRestartRecovery(menuTask)) {
                  try {
                    void Promise.resolve(
                      onConfirmComfyRestart?.(menuTask.id),
                    ).catch(() => undefined);
                  } catch {
                    // The App callback owns user-visible error reporting. Keep
                    // a defensive boundary for alternate drawer consumers.
                  }
                  setMenu(null);
                }
              }}
            >确认 ComfyUI 已重启并结束任务</button>
          )}
          {ACTIVE_STATUSES.has(menuTask.status) ? (
            <button type="button" role="menuitem" className="is-danger" disabled={!supportsCancel} onClick={() => {
              void onCancel(menuTask.id);
              setMenu(null);
            }}>{menuTask.status === "cancelling" ? "重试取消" : "取消任务"}</button>
          ) : (
            <button type="button" role="menuitem" className="is-danger" disabled={deletingTaskIds.has(menuTask.id) || clearing} onClick={() => {
              if (confirmRecordDeletion(`确定移除任务 ${menuTask.id.slice(0, 8)} 的记录吗？`)) {
                void onDelete(menuTask.id);
              }
              setMenu(null);
            }}>移除任务记录</button>
          )}
        </div>
      )}
      {viewerTask && viewer && (
        <MediaLightbox
          task={viewerTask}
          initialIndex={viewer.index}
          onClose={() => setViewer(null)}
          onImportOutput={onImportOutput}
          restoreFocus={modalReturnFocusRef.current}
        />
      )}
      {errorTask && <ErrorDialog task={errorTask} onClose={() => setErrorTaskId(null)} restoreFocus={modalReturnFocusRef.current} />}
      {generationDetails && (
        <GenerationDetailsDialog
          state={generationDetails}
          onClose={closeGenerationDetails}
          restoreFocus={modalReturnFocusRef.current}
        />
      )}
    </aside>
  );
}
