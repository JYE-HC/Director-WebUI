import { useEffect, type RefObject } from "react";
import type { AssetTrashBatch, AssetTrashRestoreMode } from "../api/types";

export interface AssetTrashPanelProps {
  id: string;
  open: boolean;
  batches: readonly AssetTrashBatch[];
  loading: boolean;
  busyBatchId: string | null;
  conflictBatchIds: ReadonlySet<string>;
  toggleRef: RefObject<HTMLButtonElement | null>;
  onRefresh: () => void;
  onRestore: (batch: AssetTrashBatch, mode: AssetTrashRestoreMode) => void;
  onPurge: (batch: AssetTrashBatch) => void;
  onClose: (restoreFocus: boolean) => void;
}

function batchDescription(batch: AssetTrashBatch): string {
  const referenceCount = batch.unbound_usages.length;
  return referenceCount
    ? `${batch.asset_ids.length} 项素材 · 已解除 ${referenceCount} 处引用`
    : `${batch.asset_ids.length} 项素材 · 未解除项目引用`;
}

/** Separate compensation domain: these actions never participate in project Ctrl/Cmd+Z. */
export function AssetTrashPanel({
  id,
  open,
  batches,
  loading,
  busyBatchId,
  conflictBatchIds,
  toggleRef,
  onRefresh,
  onRestore,
  onPurge,
  onClose,
}: AssetTrashPanelProps) {
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      event.preventDefault();
      onClose(true);
    };
    const handlePointerDown = (event: PointerEvent) => {
      if (!(event.target instanceof Node)) return;
      const panel = document.getElementById(id);
      if (panel?.contains(event.target) || toggleRef.current?.contains(event.target)) return;
      onClose(false);
    };
    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("pointerdown", handlePointerDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [id, onClose, open, toggleRef]);

  return (
    <aside id={id} className="asset-trash-panel" aria-label="素材回收站" hidden={!open}>
      <header className="asset-trash-panel__head">
        <div><small>独立于项目编辑历史</small><strong>素材回收站</strong></div>
        <div>
          <button type="button" disabled={loading || busyBatchId !== null} onClick={onRefresh}>刷新</button>
          <button type="button" className="icon-button" aria-label="关闭素材回收站" onClick={() => onClose(true)}>×</button>
        </div>
      </header>
      <p className="asset-trash-panel__notice">这里只恢复 Director 素材登记与可验证的项目引用；ComfyUI 中的输入文件始终保留。</p>
      {loading && <p className="asset-trash-panel__empty" role="status">正在读取回收站…</p>}
      {!loading && !batches.length && (
        <p className="asset-trash-panel__empty">当前 ComfyUI 地址下没有可恢复的素材。</p>
      )}
      <div className="asset-trash-panel__batches">
        {batches.map((batch) => {
          const busy = busyBatchId === batch.batch_id;
          const conflict = conflictBatchIds.has(batch.batch_id);
          const actionsDisabled = loading || busyBatchId !== null;
          return (
            <article key={batch.batch_id} aria-busy={busy || undefined}>
              <header>
                <div>
                  <strong>{batch.assets.map((asset) => asset.name).join("、")}</strong>
                  <small>{batchDescription(batch)}</small>
                </div>
                <time dateTime={batch.created_at}>{new Date(batch.created_at).toLocaleString("zh-CN")}</time>
              </header>
              {conflict && (
                <p className="asset-trash-panel__conflict" role="alert">项目在移出后又发生了修改，无法安全恢复旧引用。仍可仅恢复素材登记。</p>
              )}
              <div className="asset-trash-panel__actions">
                {batch.unbound_usages.length > 0 && !conflict && (
                  <button
                    type="button"
                    disabled={actionsDisabled}
                    onClick={() => onRestore(batch, "with_references")}
                  >恢复素材与原引用</button>
                )}
                <button
                  type="button"
                  disabled={actionsDisabled}
                  onClick={() => onRestore(batch, "registration_only")}
                >仅恢复素材</button>
                <button
                  type="button"
                  className="is-danger"
                  disabled={actionsDisabled}
                  onClick={() => onPurge(batch)}
                >永久移除恢复记录</button>
              </div>
            </article>
          );
        })}
      </div>
    </aside>
  );
}
