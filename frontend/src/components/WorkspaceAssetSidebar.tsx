import {
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type Ref,
} from "react";
import type { CapabilityReport } from "../api/types";
import {
  describeUploadProgress,
  type DroppedUploadProgress,
  type DroppedUploadResult,
} from "../domain/assetDrag";
import type { AssetGridSize } from "../domain/timelineProject";
import type { AssetKind, AssetReference } from "../domain/modes";
import {
  loadTimelineWorkspacePreferences,
  updateTimelineWorkspacePreferences,
} from "../domain/workspacePreferences";
import { Spinner, StatusDot } from "./ui";

export interface WorkspaceAssetSidebarProps {
  id?: string;
  open: boolean;
  width: number;
  minWidth: number;
  maxWidth: number;
  resizable: boolean;
  assets: AssetReference[];
  selectedIds: string[];
  gridSize: AssetGridSize;
  runtimeEnabled: boolean;
  connection: CapabilityReport["connection"];
  settingsActive: boolean;
  settingsNavigationDisabled?: boolean;
  deleting?: boolean;
  /** Human-readable segment locations keyed by stable workspace asset id. */
  assetUsages?: Record<string, string[]>;
  onUploadFiles: (
    files: File[],
    onProgress?: (progress: DroppedUploadProgress) => void,
  ) => Promise<DroppedUploadResult>;
  onUploaded: (assets: AssetReference[]) => void;
  onSelect: (id: string, additive: boolean) => void;
  onSelectRange: (ids: string[], additive: boolean) => void;
  onMove: (draggedId: string, targetId: string) => void;
  onGridSize: (size: AssetGridSize) => void;
  onDelete: (ids: string[]) => void;
  onToggle: () => void;
  onWidthChange: (width: number) => void;
  toggleButtonRef?: Ref<HTMLButtonElement>;
  onSettings: () => void;
  settingsButtonRef?: Ref<HTMLButtonElement>;
}

type AssetFilter = "all" | AssetKind;

const ACCEPTED_ASSETS = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "audio/*",
  "video/*",
  ".mkv",
  ".mov",
  ".m4v",
  ".flac",
  ".ogg",
].join(",");

function assetSummary(asset: AssetReference): string {
  if (asset.kind === "video" && asset.metadata) {
    return `${asset.metadata.duration.toFixed(1)}s · ${asset.metadata.width}×${asset.metadata.height}`;
  }
  return asset.kind === "image" ? "参考图片" : asset.kind === "audio" ? "参考音频" : "视频素材";
}

export function WorkspaceAssetSidebar({
  id,
  open,
  width,
  minWidth,
  maxWidth,
  resizable,
  assets,
  selectedIds,
  gridSize,
  runtimeEnabled,
  connection,
  settingsActive,
  settingsNavigationDisabled = false,
  deleting = false,
  assetUsages = {},
  onUploadFiles,
  onUploaded,
  onSelect,
  onSelectRange,
  onMove,
  onGridSize,
  onDelete,
  onToggle,
  onWidthChange,
  toggleButtonRef,
  onSettings,
  settingsButtonRef,
}: WorkspaceAssetSidebarProps) {
  const [filter, setFilter] = useState<AssetFilter>(
    () => loadTimelineWorkspacePreferences().assetFilter,
  );
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<DroppedUploadProgress | null>(null);
  const [fileDropActive, setFileDropActive] = useState(false);
  const [resizing, setResizing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileDragDepth = useRef(0);
  const selectionAnchorId = useRef<string | null>(null);
  const resizeDrag = useRef<{ pointerId: number; startX: number; startWidth: number } | null>(null);
  const filtered = filter === "all" ? assets : assets.filter((asset) => asset.kind === filter);
  const selected = new Set(selectedIds);
  const uploadEnabled = runtimeEnabled && !settingsActive && !uploading && !deleting;
  const contentId = `${id ?? "workspace-assets"}-content`;
  const runtimeStatusId = `${id ?? "workspace-assets"}-runtime-status`;

  useEffect(() => {
    updateTimelineWorkspacePreferences({ assetFilter: filter });
  }, [filter]);

  useEffect(() => {
    if (open && uploadEnabled) return;
    fileDragDepth.current = 0;
    setFileDropActive(false);
  }, [open, uploadEnabled]);

  useEffect(() => {
    if (open && resizable) return;
    resizeDrag.current = null;
    setResizing(false);
    document.body.classList.remove("is-resizing-asset-sidebar");
  }, [open, resizable]);

  useEffect(() => {
    if (!resizing) return;
    const finish = () => {
      resizeDrag.current = null;
      setResizing(false);
      document.body.classList.remove("is-resizing-asset-sidebar");
    };
    window.addEventListener("pointerup", finish);
    window.addEventListener("pointercancel", finish);
    return () => {
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      document.body.classList.remove("is-resizing-asset-sidebar");
    };
  }, [resizing]);

  const clampWidth = (next: number) => Math.min(maxWidth, Math.max(minWidth, Math.round(next)));

  const finishResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = resizeDrag.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    resizeDrag.current = null;
    setResizing(false);
    document.body.classList.remove("is-resizing-asset-sidebar");
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture?.(event.pointerId);
    }
  };

  const resizeWithKeyboard = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 40 : 10;
    let next: number | null = null;
    if (event.key === "ArrowLeft") next = width - step;
    if (event.key === "ArrowRight") next = width + step;
    if (event.key === "Home") next = minWidth;
    if (event.key === "End") next = maxWidth;
    if (next === null) return;
    event.preventDefault();
    onWidthChange(clampWidth(next));
  };

  const isSystemFileDrag = (event: DragEvent<HTMLElement>) => {
    const types = Array.from(event.dataTransfer.types ?? []);
    return !types.includes("application/x-director-asset") &&
      (types.includes("Files") || event.dataTransfer.files.length > 0);
  };

  const upload = async (files: FileList | null) => {
    if (!runtimeEnabled || settingsActive || !files?.length || uploading) return;
    setUploading(true);
    setError(null);
    try {
      const result = await onUploadFiles(Array.from(files), setUploadProgress);
      if (result.authority_stale) {
        setError("上传期间 ComfyUI 设置已变化，旧地址返回的素材未加入当前素材库");
        return;
      }
      if (result.assets.length) onUploaded(result.assets);
      if (result.failures.length) {
        setError(result.failures.map((failure) => `${failure.file_name}：${failure.message}`).join("；"));
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "素材上传失败");
    } finally {
      setUploading(false);
      setUploadProgress(null);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const runtimeLabel = connection === "offline"
    ? "ComfyUI 离线"
    : connection === "online"
      ? "ComfyUI 已连接"
      : connection === "checking"
        ? "正在检测 ComfyUI"
        : "等待 ComfyUI 状态";
  const runtimeDotState = connection === "offline" ? "offline" : connection;

  const deleteSelected = () => {
    if (!runtimeEnabled || !selectedIds.length || deleting) return;
    const referenced = selectedIds.flatMap((assetId) => {
      const asset = assets.find((item) => item.id === assetId);
      const usages = assetUsages[assetId] ?? [];
      return usages.length ? [{ name: asset?.name ?? assetId, usages }] : [];
    });
    if (referenced.length) {
      const details = referenced
        .map(({ name, usages }) => `• ${name}（${usages.length} 处）：${usages.join("、")}`)
        .join("\n");
      const confirmed = window.confirm(
        `以下素材仍被时间线片段引用：\n\n${details}\n\n从素材库移出后将解除这些引用，但会保留 ComfyUI 文件。是否继续？`,
      );
      if (!confirmed) return;
    }
    onDelete([...selectedIds]);
  };

  return (
    <aside
      id={id}
      className={`asset-sidebar ${open ? "is-open" : "is-collapsed"} ${open && fileDropActive ? "asset-sidebar--file-drop" : ""} ${resizing ? "is-resizing" : ""}`}
      aria-label="当前工作区素材库"
      aria-describedby={`${id ?? "workspace-assets"}-drop-help`}
      onClick={(event) => {
        if (open) return;
        const target = event.target;
        if (target instanceof Element && target.closest("button, a, input, select, textarea, [role='button']")) return;
        onToggle();
      }}
      onDragEnter={(event) => {
        if (!isSystemFileDrag(event)) return;
        event.preventDefault();
        if (!open || !uploadEnabled) return;
        fileDragDepth.current += 1;
        setFileDropActive(true);
      }}
      onDragOver={(event) => {
        if (!isSystemFileDrag(event)) return;
        event.preventDefault();
        if (!open || !uploadEnabled) {
          event.dataTransfer.dropEffect = "none";
          return;
        }
        event.dataTransfer.dropEffect = "copy";
        setFileDropActive(true);
      }}
      onDragLeave={(event) => {
        if (!isSystemFileDrag(event)) return;
        fileDragDepth.current = Math.max(0, fileDragDepth.current - 1);
        if (fileDragDepth.current === 0) setFileDropActive(false);
      }}
      onDrop={(event) => {
        if (!isSystemFileDrag(event)) return;
        event.preventDefault();
        fileDragDepth.current = 0;
        setFileDropActive(false);
        if (!open || !uploadEnabled) return;
        void upload(event.dataTransfer.files);
      }}
    >
      <span id={`${id ?? "workspace-assets"}-drop-help`} className="sr-only">
        {!open ? "素材库内容区当前未启用文件拖放" : runtimeEnabled ? "可将电脑中的图片、视频或音频文件拖放到素材库批量导入" : "连接 ComfyUI 后可拖放文件导入"}
      </span>
      {open && fileDropActive && (
        <div className="asset-sidebar__drop-overlay" role="status" aria-live="polite">
          <span aria-hidden="true">＋</span><strong>释放以导入素材</strong><small>支持批量图片、视频和音频</small>
        </div>
      )}
      {open && resizable && !settingsActive && (
        <div
          className="asset-sidebar__resize-handle"
          role="separator"
          aria-label="调整素材库宽度"
          aria-orientation="vertical"
          aria-valuemin={minWidth}
          aria-valuemax={maxWidth}
          aria-valuenow={width}
          aria-valuetext={`${width} 像素`}
          tabIndex={0}
          onKeyDown={resizeWithKeyboard}
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            event.preventDefault();
            resizeDrag.current = {
              pointerId: event.pointerId,
              startX: event.clientX,
              startWidth: width,
            };
            setResizing(true);
            document.body.classList.add("is-resizing-asset-sidebar");
            event.currentTarget.setPointerCapture?.(event.pointerId);
            event.currentTarget.focus();
          }}
          onPointerMove={(event) => {
            const drag = resizeDrag.current;
            if (!drag || drag.pointerId !== event.pointerId) return;
            onWidthChange(clampWidth(drag.startWidth + event.clientX - drag.startX));
          }}
          onPointerUp={finishResize}
          onPointerCancel={finishResize}
          onLostPointerCapture={(event) => {
            if (resizeDrag.current?.pointerId !== event.pointerId) return;
            resizeDrag.current = null;
            setResizing(false);
            document.body.classList.remove("is-resizing-asset-sidebar");
          }}
        />
      )}
      <header className="asset-sidebar__brand">
        <button ref={toggleButtonRef} type="button" className="asset-sidebar__brand-toggle" aria-label="素材库" aria-expanded={open} aria-controls={contentId} onClick={onToggle}>
          <span className="brand__mark" aria-hidden="true"><span /><span /><span /></span>
          {open && <span className="asset-sidebar__brand-copy" aria-hidden="true"><strong>DIRECTOR</strong></span>}
        </button>
      </header>

      <div id={contentId} className="asset-sidebar__content" hidden={!open} {...(settingsActive ? { inert: true } : {})}>
      <section className="asset-sidebar__head">
        <div><h2>当前工作区素材</h2></div>
        <label className={`asset-sidebar__upload ${!uploadEnabled ? "is-disabled" : ""}`}>
          <input className="asset-sidebar__file-input" ref={inputRef} type="file" multiple accept={ACCEPTED_ASSETS} disabled={!uploadEnabled} onChange={(event) => void upload(event.target.files)} />
          {uploading ? <Spinner label="上传素材" /> : <span aria-hidden="true">＋</span>}
          <span title={uploadProgress ? describeUploadProgress(uploadProgress) : undefined}>
            {uploading
              ? uploadProgress?.stage === "uploading" && uploadProgress.percent !== undefined
                ? `${uploadProgress.percent}%`
                : "处理中"
              : "导入"}
          </span>
        </label>
      </section>

      {uploadProgress && (
        <div className="asset-sidebar__upload-progress" role="status">
          {describeUploadProgress(uploadProgress)}
        </div>
      )}

      <div className="asset-sidebar__controls">
        <div className="asset-filter" role="tablist" aria-label="素材类型">
          {(["all", "image", "video", "audio"] as const).map((kind) => (
            <button key={kind} type="button" role="tab" aria-selected={filter === kind} onClick={() => setFilter(kind)}>
              {kind === "all" ? "全部" : kind === "image" ? "图片" : kind === "video" ? "视频" : "音频"}
            </button>
          ))}
        </div>
        <div className="asset-size" aria-label="素材网格大小">
          {(["small", "medium", "large"] as const).map((size, index) => (
            <button key={size} type="button" className={gridSize === size ? "is-active" : ""} aria-label={`${["小", "中", "大"][index]}网格`} aria-pressed={gridSize === size} onClick={() => onGridSize(size)}>
              <span style={{ width: 7 + index * 3, height: 7 + index * 3 }} />
            </button>
          ))}
        </div>
      </div>

      {error && <p className="asset-sidebar__error" role="alert">{error}</p>}
      <div className={`workspace-assets workspace-assets--${gridSize}`} role="list" aria-label="素材列表">
        {filtered.length ? filtered.map((asset) => {
          const usageCount = assetUsages[asset.id]?.length ?? 0;
          return (
          <article
            key={asset.id}
            role="listitem"
            className={`workspace-asset ${selected.has(asset.id) ? "is-selected" : ""}`}
            draggable
            onDragStart={(event) => {
              const draggedIds = selected.has(asset.id) && selectedIds.length
                ? assets.filter((candidate) => selected.has(candidate.id)).map((candidate) => candidate.id)
                : [asset.id];
              event.dataTransfer.effectAllowed = "copyMove";
              event.dataTransfer.setData("application/x-director-asset", asset.id);
              event.dataTransfer.setData("application/x-director-assets", JSON.stringify(draggedIds));
              event.dataTransfer.setData(`application/x-director-asset-${asset.kind}`, asset.id);
              event.dataTransfer.setData("text/plain", asset.id);
            }}
            onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }}
            onDrop={(event) => {
              event.preventDefault();
              const dragged = event.dataTransfer.getData("application/x-director-asset");
              if (dragged) onMove(dragged, asset.id);
            }}
            onClick={(event) => {
              const additive = event.ctrlKey || event.metaKey;
              if (event.shiftKey && selectionAnchorId.current) {
                const anchor = filtered.findIndex(
                  (candidate) => candidate.id === selectionAnchorId.current,
                );
                const target = filtered.findIndex((candidate) => candidate.id === asset.id);
                if (anchor >= 0 && target >= 0) {
                  const [start, end] = anchor < target
                    ? [anchor, target]
                    : [target, anchor];
                  onSelectRange(
                    filtered.slice(start, end + 1).map((candidate) => candidate.id),
                    additive,
                  );
                  return;
                }
              }
              selectionAnchorId.current = asset.id;
              onSelect(asset.id, additive);
            }}
          >
            <div className="workspace-asset__preview">
              {asset.kind === "image" && asset.preview_url ? <img src={asset.preview_url} alt="" draggable={false} /> : asset.kind === "video" && asset.preview_url ? <video src={asset.preview_url} muted preload="metadata" /> : <span aria-hidden="true">{asset.kind === "audio" ? "♫" : asset.kind === "video" ? "▶" : "▧"}</span>}
              <i>{asset.kind === "image" ? "IMG" : asset.kind === "video" ? "VID" : "AUD"}</i>
              {usageCount > 0 && (
                <b
                  className="workspace-asset__usage"
                  aria-label={`${asset.name} 被 ${usageCount} 个片段位置引用`}
                  title={(assetUsages[asset.id] ?? []).join("、")}
                >
                  引用 {usageCount}
                </b>
              )}
              <input aria-label={`选择素材 ${asset.name}`} type="checkbox" checked={selected.has(asset.id)} readOnly />
            </div>
            <strong title={asset.name}>{asset.name}</strong>
            <small>{assetSummary(asset)}</small>
          </article>
          );
        }) : (
          <div className="workspace-assets__empty">
            <span aria-hidden="true">◇</span>
            <strong>还没有素材</strong>
            <small>{runtimeEnabled ? "导入图片、视频或音频；可直接拖到时间线" : "连接 ComfyUI 后可导入素材"}</small>
          </div>
        )}
      </div>

      <div className="asset-sidebar__batch">
        <span>已选 {selectedIds.length} / {assets.length}</span>
        <button className="asset-sidebar__batch-remove" type="button" disabled={!runtimeEnabled || !selectedIds.length || deleting} onClick={deleteSelected}>{deleting ? "移出中…" : "移出素材库"}</button>
      </div>
      </div>

      <footer className="asset-sidebar__footer">
        <button id="system-settings-toggle" ref={settingsButtonRef} type="button" aria-label="系统设置" aria-describedby={open ? runtimeStatusId : undefined} aria-haspopup="dialog" aria-expanded={settingsActive} aria-controls="system-settings-dialog" disabled={settingsNavigationDisabled} className={`utility-nav ${settingsActive ? "is-active" : ""}`} onClick={onSettings}>
          <span className="utility-nav__icon" aria-hidden="true">⌘</span>
          {open && <span className="asset-sidebar__settings-label"><strong>系统设置</strong></span>}
        </button>
        {open && (
          <div id={runtimeStatusId} className="asset-sidebar__settings-runtime" role="status" aria-live="polite" aria-atomic="true" title={runtimeLabel}>
            <StatusDot state={runtimeDotState} />
            <span>{runtimeLabel}</span>
          </div>
        )}
      </footer>
    </aside>
  );
}
