import { createRef } from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { AssetTrashPanel } from "../components/AssetTrashPanel";
import type { AssetTrashBatch } from "../api/types";

const batch: AssetTrashBatch = {
  batch_id: "trash-batch-1",
  comfy_origin: "http://127.0.0.1:8188",
  asset_ids: ["asset-a"],
  assets: [{
    id: "asset-a",
    name: "参考图.png",
    subfolder: "director-web",
    type: "input",
    kind: "image",
  }],
  cascade: true,
  unbound_usages: ["timeline.segments[0].first_image"],
  unbound_usages_by_asset: {
    "asset-a": ["timeline.segments[0].first_image"],
  },
  created_at: "2026-08-16T12:00:00Z",
  remote_files_preserved: true,
};

describe("AssetTrashPanel", () => {
  it("明确区分两档恢复与只清 Director 恢复记录", () => {
    const onRestore = vi.fn();
    const onPurge = vi.fn();
    render(<AssetTrashPanel
      id="asset-trash"
      open
      batches={[batch]}
      loading={false}
      busyBatchId={null}
      conflictBatchIds={new Set()}
      toggleRef={createRef<HTMLButtonElement>()}
      onRefresh={vi.fn()}
      onRestore={onRestore}
      onPurge={onPurge}
      onClose={vi.fn()}
    />);

    expect(screen.getByText(/ComfyUI 中的输入文件始终保留/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "恢复素材与原引用" }));
    expect(onRestore).toHaveBeenCalledWith(batch, "with_references");
    fireEvent.click(screen.getByRole("button", { name: "仅恢复素材" }));
    expect(onRestore).toHaveBeenCalledWith(batch, "registration_only");
    fireEvent.click(screen.getByRole("button", { name: "永久移除恢复记录" }));
    expect(onPurge).toHaveBeenCalledWith(batch);
  });

  it("引用恢复冲突时只提供安全的 registration-only 降级", () => {
    render(<AssetTrashPanel
      id="asset-trash"
      open
      batches={[batch]}
      loading={false}
      busyBatchId={null}
      conflictBatchIds={new Set([batch.batch_id])}
      toggleRef={createRef<HTMLButtonElement>()}
      onRefresh={vi.fn()}
      onRestore={vi.fn()}
      onPurge={vi.fn()}
      onClose={vi.fn()}
    />);

    expect(screen.getByRole("alert")).toHaveTextContent("无法安全恢复旧引用");
    expect(screen.queryByRole("button", { name: "恢复素材与原引用" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "仅恢复素材" })).toBeEnabled();
  });

  it("分别保留多个批次的引用恢复冲突", () => {
    const secondBatch: AssetTrashBatch = {
      ...batch,
      batch_id: "trash-batch-2",
      asset_ids: ["asset-b"],
      assets: [{
        ...batch.assets[0],
        id: "asset-b",
        name: "第二张参考图.png",
      }],
      unbound_usages: ["projects.project-2.segments[0].last_image"],
      unbound_usages_by_asset: {
        "asset-b": ["projects.project-2.segments[0].last_image"],
      },
    };
    const props = {
      id: "asset-trash",
      open: true,
      batches: [batch, secondBatch],
      loading: false,
      busyBatchId: null,
      toggleRef: createRef<HTMLButtonElement>(),
      onRefresh: vi.fn(),
      onRestore: vi.fn(),
      onPurge: vi.fn(),
      onClose: vi.fn(),
    } as const;
    const { rerender } = render(
      <AssetTrashPanel {...props} conflictBatchIds={new Set([batch.batch_id])} />,
    );

    let articles = screen.getAllByRole("article");
    expect(within(articles[0]).getByRole("alert")).toBeInTheDocument();
    expect(within(articles[0]).queryByRole("button", { name: "恢复素材与原引用" }))
      .not.toBeInTheDocument();
    expect(within(articles[1]).getByRole("button", { name: "恢复素材与原引用" }))
      .toBeEnabled();

    rerender(
      <AssetTrashPanel
        {...props}
        conflictBatchIds={new Set([batch.batch_id, secondBatch.batch_id])}
      />,
    );
    articles = screen.getAllByRole("article");
    expect(within(articles[0]).getByRole("alert")).toBeInTheDocument();
    expect(within(articles[1]).getByRole("alert")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "恢复素材与原引用" }))
      .not.toBeInTheDocument();
  });

  it("读取回收站期间禁用刷新与所有批次动作但允许关闭", () => {
    const onRestore = vi.fn();
    const onPurge = vi.fn();
    render(<AssetTrashPanel
      id="asset-trash"
      open
      batches={[batch]}
      loading
      busyBatchId={null}
      conflictBatchIds={new Set()}
      toggleRef={createRef<HTMLButtonElement>()}
      onRefresh={vi.fn()}
      onRestore={onRestore}
      onPurge={onPurge}
      onClose={vi.fn()}
    />);

    expect(screen.getByRole("button", { name: "刷新" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "恢复素材与原引用" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "仅恢复素材" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "永久移除恢复记录" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "关闭素材回收站" })).toBeEnabled();
  });
});
