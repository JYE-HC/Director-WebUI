import { useEffect, type RefObject } from "react";
import {
  timelineHistoryCursor,
  timelineHistoryEntries,
  type TimelineHistoryState,
} from "../state/timelineHistory";

export interface TimelineHistoryPanelProps {
  id: string;
  open: boolean;
  history: TimelineHistoryState;
  toggleRef: RefObject<HTMLButtonElement | null>;
  onJump: (cursor: number) => void;
  onClose: (restoreFocus: boolean) => void;
}

function formatHistoryBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

/** Non-modal, inspectable project history. Each jump produces one final replay. */
export function TimelineHistoryPanel({
  id,
  open,
  history,
  toggleRef,
  onJump,
  onClose,
}: TimelineHistoryPanelProps) {
  const entries = timelineHistoryEntries(history);
  const cursor = timelineHistoryCursor(history);

  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      event.preventDefault();
      onClose(true);
    };
    const closeOutside = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      const panel = document.getElementById(id);
      if (panel?.contains(target) || toggleRef.current?.contains(target)) return;
      onClose(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    document.addEventListener("pointerdown", closeOutside);
    return () => {
      document.removeEventListener("keydown", closeOnEscape);
      document.removeEventListener("pointerdown", closeOutside);
    };
  }, [id, onClose, open, toggleRef]);

  return (
    <aside
      id={id}
      className="timeline-history-panel"
      aria-label="编辑历史"
      hidden={!open}
    >
      <header className="timeline-history-panel__head">
        <div>
          <small>项目级事务</small>
          <strong>编辑历史</strong>
        </div>
        <button type="button" className="icon-button" aria-label="关闭编辑历史" onClick={() => onClose(true)}>×</button>
      </header>
      <div className="timeline-history-panel__summary">
        <span>当前位置 {cursor} / {entries.length}</span>
        <span>{formatHistoryBytes(history.totalBytes)}</span>
      </div>
      <ol className="timeline-history-panel__list" aria-label="项目历史状态">
        <li data-history-status={cursor === 0 ? "current" : "applied"}>
          <button
            type="button"
            aria-current={cursor === 0 ? "step" : undefined}
            disabled={cursor === 0}
            onClick={() => onJump(0)}
          >
            <span className="timeline-history-panel__marker" aria-hidden="true" />
            <span className="timeline-history-panel__entry">
              <strong>初始状态</strong>
              <small>{cursor === 0 ? "当前位置" : "编辑起点"}</small>
            </span>
          </button>
        </li>
        {entries.map((entry, index) => {
          const targetCursor = index + 1;
          const status = targetCursor === cursor
            ? "current"
            : targetCursor < cursor
              ? "applied"
              : "undone";
          const statusLabel = status === "current"
            ? "当前位置"
            : status === "applied"
              ? "已应用"
              : "已撤销";
          return (
            <li key={entry.id} data-history-status={status}>
              <button
                type="button"
                aria-current={status === "current" ? "step" : undefined}
                aria-label={`${entry.label}，${statusLabel}`}
                disabled={status === "current"}
                onClick={() => onJump(targetCursor)}
              >
                <span className="timeline-history-panel__marker" aria-hidden="true" />
                <span className="timeline-history-panel__entry">
                  <strong>{entry.label}</strong>
                  <small>
                    {statusLabel}
                    {entry.affectedSegmentIds.length
                      ? ` · ${entry.affectedSegmentIds.length} 个片段`
                      : " · 全局"}
                    {` · ${formatHistoryBytes(entry.byteSize)}`}
                  </small>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      {!entries.length && (
        <p className="timeline-history-panel__empty">项目修改会显示在这里。素材上传、任务与回收站操作使用各自的记录。</p>
      )}
    </aside>
  );
}
