import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { WorkspaceAssetSidebar, type WorkspaceAssetSidebarProps } from "../components/WorkspaceAssetSidebar";
import type { AssetReference } from "../domain/modes";
import type { DroppedUploadResult } from "../domain/assetDrag";
import {
  loadTimelineWorkspacePreferences,
  updateTimelineWorkspacePreferences,
} from "../domain/workspacePreferences";

const imageAsset: AssetReference = {
  id: "asset-image",
  name: "角色参考.png",
  subfolder: "director",
  type: "input",
  kind: "image",
  preview_url: "/api/assets/asset-image/preview",
};

const videoAsset: AssetReference = {
  id: "asset-video",
  name: "表演参考.mp4",
  subfolder: "director",
  type: "input",
  kind: "video",
  preview_url: "/api/assets/asset-video/preview",
};

function renderSidebar(overrides: Partial<WorkspaceAssetSidebarProps> = {}) {
  const props: WorkspaceAssetSidebarProps = {
    id: "asset-library",
    open: true,
    width: 292,
    minWidth: 292,
    maxWidth: 720,
    resizable: true,
    assets: [imageAsset, videoAsset],
    selectedIds: [],
    gridSize: "medium",
    runtimeEnabled: true,
    connection: "online",
    settingsActive: false,
    onUploadFiles: vi.fn().mockResolvedValue({ assets: [], failures: [], authority_stale: false }),
    onUploaded: vi.fn(),
    onSelect: vi.fn(),
    onSelectRange: vi.fn(),
    onMove: vi.fn(),
    onGridSize: vi.fn(),
    onDelete: vi.fn(),
    onToggle: vi.fn(),
    onWidthChange: vi.fn(),
    onSettings: vi.fn(),
    ...overrides,
  };
  render(<WorkspaceAssetSidebar {...props} />);
  return props;
}

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe("WorkspaceAssetSidebar", () => {
  it("刷新后恢复素材类型筛选", async () => {
    const user = userEvent.setup();
    updateTimelineWorkspacePreferences({ assetFilter: "video" });
    renderSidebar();

    expect(screen.getByRole("tab", { name: "视频" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(videoAsset.name)).toBeInTheDocument();
    expect(screen.queryByText(imageAsset.name)).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "图片" }));
    expect(loadTimelineWorkspacePreferences().assetFilter).toBe("image");
  });

  it("Director 品牌本身就是素材库收起入口", async () => {
    const user = userEvent.setup();
    const props = renderSidebar();

    const toggle = screen.getByRole("button", { name: "素材库" });
    expect(toggle).toHaveTextContent("DIRECTOR");
    expect(toggle).not.toHaveTextContent("LONG-FORM STUDIO");
    expect(screen.queryByText("WORKSPACE ASSETS")).not.toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-controls", "asset-library-content");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    await user.click(toggle);
    expect(props.onToggle).toHaveBeenCalledTimes(1);
  });

  it("收起后保留窄轨中的 Logo 和系统设置，并可点击空白窄轨展开", async () => {
    const user = userEvent.setup();
    const props = renderSidebar({ open: false });

    const rail = screen.getByRole("complementary", { name: "当前工作区素材库" });
    expect(rail).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "素材库" })).toHaveAttribute("aria-expanded", "false");
    const settings = screen.getByRole("button", { name: "系统设置" });
    expect(settings).toBeInTheDocument();
    expect(settings).not.toHaveAttribute("aria-describedby");
    expect(settings.querySelector(".asset-sidebar__settings-label")).not.toBeInTheDocument();
    expect(settings.querySelector(".asset-sidebar__settings-runtime")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "当前工作区素材" })).not.toBeInTheDocument();
    expect(document.getElementById("asset-library-content")).toHaveAttribute("hidden");
    expect(screen.queryByRole("separator", { name: "调整素材库宽度" })).not.toBeInTheDocument();
    expect(props.onToggle).not.toHaveBeenCalled();

    await user.click(rail);
    expect(props.onToggle).toHaveBeenCalledTimes(1);
  });

  it("收起态的品牌和系统设置保留各自操作，不被整轨点击重复触发", async () => {
    const user = userEvent.setup();
    const props = renderSidebar({ open: false });

    await user.click(screen.getByRole("button", { name: "素材库" }));
    expect(props.onToggle).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: "系统设置" }));
    expect(props.onSettings).toHaveBeenCalledTimes(1);
    expect(props.onToggle).toHaveBeenCalledTimes(1);
  });

  it("把系统设置标题和 ComfyUI 连接状态放在同一个按钮行内", () => {
    const props = renderSidebar();

    const settings = screen.getByRole("button", { name: "系统设置" });
    const status = screen.getByRole("status");
    expect(settings).toHaveAttribute("aria-expanded", "false");
    expect(settings).toHaveAttribute("aria-controls", "system-settings-dialog");
    expect(settings).toHaveAttribute("aria-describedby", "asset-library-runtime-status");
    expect(settings).toHaveTextContent("系统设置");
    expect(settings).not.toContainElement(status);
    expect(status).toHaveTextContent("ComfyUI 已连接");
    expect(status.parentElement).toBe(settings.parentElement);
    expect(settings).not.toHaveTextContent("ComfyUI · 模型 · GPU");
    expect(status.querySelector(".status-dot")).toHaveClass("status-dot--online");
    expect(document.querySelector(".asset-sidebar__footer > .sidebar__runtime")).not.toBeInTheDocument();
    status.click();
    expect(props.onSettings).not.toHaveBeenCalled();
  });

  it.each([
    ["离线", { connection: "offline" as const }, "ComfyUI 离线", "status-dot--offline"],
    ["检测中", { connection: "checking" as const }, "正在检测 ComfyUI", "status-dot--checking"],
    ["未知", { connection: "unknown" as const }, "等待 ComfyUI 状态", "status-dot--unknown"],
  ])("在系统设置按钮内保留%s状态", (_name, overrides, label, dotClass) => {
    renderSidebar(overrides);

    expect(screen.getByRole("button", { name: "系统设置" })).toHaveAccessibleName("系统设置");
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(label);
    expect(status.querySelector(".status-dot")).toHaveClass(dotClass);
  });

  it("将素材导入控件限制为裁剪输入，不用透明全屏元素撑宽页面", () => {
    renderSidebar();

    const fileInput = document.querySelector<HTMLInputElement>(
      '.asset-sidebar__upload input[type="file"]',
    );
    expect(fileInput).toHaveClass("asset-sidebar__file-input");
  });

  it("素材移出处理中使用一致文案并禁用重复操作", () => {
    renderSidebar({ selectedIds: [imageAsset.id], deleting: true });
    expect(screen.getByRole("button", { name: "移出中…" })).toBeDisabled();
  });

  it("通过拖动或键盘在最小值和半屏上限之间调整宽度", () => {
    const onWidthChange = vi.fn();
    renderSidebar({ onWidthChange });
    const handle = screen.getByRole("separator", { name: "调整素材库宽度" });

    expect(handle).toHaveAttribute("aria-orientation", "vertical");
    expect(handle).toHaveAttribute("aria-valuemin", "292");
    expect(handle).toHaveAttribute("aria-valuemax", "720");
    expect(handle).toHaveAttribute("aria-valuenow", "292");

    fireEvent.keyDown(handle, { key: "ArrowRight" });
    fireEvent.keyDown(handle, { key: "ArrowLeft", shiftKey: true });
    fireEvent.keyDown(handle, { key: "End" });
    expect(onWidthChange).toHaveBeenNthCalledWith(1, 302);
    expect(onWidthChange).toHaveBeenNthCalledWith(2, 292);
    expect(onWidthChange).toHaveBeenNthCalledWith(3, 720);

    fireEvent.pointerDown(handle, { pointerId: 7, button: 0, clientX: 292 });
    expect(document.body).toHaveClass("is-resizing-asset-sidebar");
    fireEvent.pointerMove(handle, { pointerId: 7, clientX: 900 });
    expect(onWidthChange).toHaveBeenLastCalledWith(720);
    fireEvent.pointerUp(handle, { pointerId: 7, clientX: 900 });
    expect(document.body).not.toHaveClass("is-resizing-asset-sidebar");
  });

  it("在移动端策略禁用时不显示宽度拖动柄", () => {
    renderSidebar({ resizable: false });
    expect(screen.queryByRole("separator", { name: "调整素材库宽度" })).not.toBeInTheDocument();
  });

  it("接收操作系统多文件拖放并批量上传", async () => {
    const first = new File(["image"], "first.png", { type: "image/png" });
    const second = new File(["video"], "second.mp4", { type: "video/mp4" });
    const uploadedImage = { ...imageAsset, id: "uploaded-image", name: first.name };
    const uploadedVideo = { ...videoAsset, id: "uploaded-video", name: second.name };
    const onUploadFiles = vi.fn().mockResolvedValue({
      assets: [uploadedImage, uploadedVideo],
      failures: [],
      authority_stale: false,
    });
    const props = renderSidebar({ onUploadFiles });
    const sidebar = screen.getByRole("complementary", { name: "当前工作区素材库" });
    const dataTransfer = {
      types: ["Files"],
      files: [first, second],
      dropEffect: "none",
    };

    fireEvent.dragEnter(sidebar, { dataTransfer });
    expect(document.querySelector(".asset-sidebar__drop-overlay")).toHaveTextContent("释放以导入素材");
    fireEvent.dragOver(sidebar, { dataTransfer });
    expect(dataTransfer.dropEffect).toBe("copy");
    fireEvent.drop(sidebar, { dataTransfer });

    await waitFor(() => expect(onUploadFiles).toHaveBeenCalledWith(
      [first, second], expect.any(Function),
    ));
    expect(props.onUploaded).toHaveBeenCalledWith([uploadedImage, uploadedVideo]);
    expect(document.querySelector(".asset-sidebar__drop-overlay")).not.toBeInTheDocument();
  });

  it("显示当前文件的上传百分比和服务端处理阶段", async () => {
    const file = new File(["video"], "slow.mp4", { type: "video/mp4" });
    let finish!: (value: DroppedUploadResult) => void;
    const onUploadFiles = vi.fn((files, onProgress) => {
      onProgress?.({
        file_name: files[0].name,
        file_index: 0,
        total_files: 1,
        completed_files: 0,
        stage: "uploading",
        percent: 42,
      });
      return new Promise<DroppedUploadResult>((resolve) => { finish = resolve; });
    });
    renderSidebar({ onUploadFiles });
    const input = document.querySelector<HTMLInputElement>(".asset-sidebar__file-input")!;

    fireEvent.change(input, { target: { files: [file] } });
    expect(await screen.findByText("slow.mp4：正在上传 42%"))
      .toHaveClass("asset-sidebar__upload-progress");
    finish({ assets: [], failures: [], authority_stale: false });
    await waitFor(() => expect(screen.queryByText(/slow.mp4：/)).not.toBeInTheDocument());
  });

  it("内部素材拖动不会被误判为文件上传", () => {
    const onUploadFiles = vi.fn();
    renderSidebar({ onUploadFiles });
    const sidebar = screen.getByRole("complementary", { name: "当前工作区素材库" });
    const dataTransfer = {
      types: ["application/x-director-asset", "text/plain"],
      files: [],
      getData: (type: string) => type === "application/x-director-asset" ? imageAsset.id : "",
      dropEffect: "none",
    };

    fireEvent.dragEnter(sidebar, { dataTransfer });
    fireEvent.dragOver(sidebar, { dataTransfer });
    fireEvent.drop(sidebar, { dataTransfer });

    expect(onUploadFiles).not.toHaveBeenCalled();
    expect(document.querySelector(".asset-sidebar__drop-overlay")).not.toBeInTheDocument();
  });

  it("拖动已选素材时携带完整且按素材库排序的多选载荷", () => {
    renderSidebar({ selectedIds: [videoAsset.id, imageAsset.id] });
    const setData = vi.fn();
    const dragged = screen.getByText("角色参考.png").closest("article");

    fireEvent.dragStart(dragged!, {
      dataTransfer: { effectAllowed: "none", setData },
    });

    expect(setData).toHaveBeenCalledWith("application/x-director-asset", imageAsset.id);
    expect(setData).toHaveBeenCalledWith(
      "application/x-director-assets",
      JSON.stringify([imageAsset.id, videoAsset.id]),
    );
  });

  it("Shift 点击按当前可见顺序选择连续素材区间", () => {
    const third: AssetReference = {
      ...imageAsset,
      id: "asset-third",
      name: "场景参考.png",
    };
    const onSelect = vi.fn();
    const onSelectRange = vi.fn();
    renderSidebar({
      assets: [imageAsset, videoAsset, third],
      onSelect,
      onSelectRange,
    });

    fireEvent.click(screen.getByText(imageAsset.name).closest("article")!);
    fireEvent.click(screen.getByText(third.name).closest("article")!, { shiftKey: true });

    expect(onSelect).toHaveBeenCalledWith(imageAsset.id, false);
    expect(onSelectRange).toHaveBeenCalledWith(
      [imageAsset.id, videoAsset.id, third.id],
      false,
    );
  });

  it("Ctrl+Shift 点击把可见区间追加到现有选择", () => {
    const third: AssetReference = {
      ...imageAsset,
      id: "asset-third",
      name: "场景参考.png",
    };
    const onSelectRange = vi.fn();
    renderSidebar({ assets: [imageAsset, videoAsset, third], onSelectRange });

    fireEvent.click(screen.getByText(videoAsset.name).closest("article")!);
    fireEvent.click(screen.getByText(third.name).closest("article")!, {
      shiftKey: true,
      ctrlKey: true,
    });

    expect(onSelectRange).toHaveBeenCalledWith([videoAsset.id, third.id], true);
  });

  it("运行时禁用时不接收系统文件拖放", () => {
    const onUploadFiles = vi.fn();
    renderSidebar({ runtimeEnabled: false, connection: "offline", onUploadFiles });
    const sidebar = screen.getByRole("complementary", { name: "当前工作区素材库" });
    const dataTransfer = {
      types: ["Files"],
      files: [new File(["image"], "offline.png", { type: "image/png" })],
      dropEffect: "none",
    };

    expect(fireEvent.dragEnter(sidebar, { dataTransfer })).toBe(false);
    expect(fireEvent.dragOver(sidebar, { dataTransfer })).toBe(false);
    expect(dataTransfer.dropEffect).toBe("none");
    expect(fireEvent.drop(sidebar, { dataTransfer })).toBe(false);

    expect(onUploadFiles).not.toHaveBeenCalled();
    expect(document.querySelector(".asset-sidebar__drop-overlay")).not.toBeInTheDocument();
  });

  it("显示引用次数，删除被引用素材前列出片段并允许取消", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    const confirm = vi.spyOn(window, "confirm").mockReturnValueOnce(false).mockReturnValueOnce(true);
    renderSidebar({
      selectedIds: [imageAsset.id],
      assetUsages: { [imageAsset.id]: ["片段 01 · 首帧", "片段 03 · 参考图片 1"] },
      onDelete,
    });

    expect(screen.getByLabelText("角色参考.png 被 2 个片段位置引用")).toHaveTextContent("引用 2");
    await user.click(screen.getByRole("button", { name: "移出素材库" }));
    expect(confirm).toHaveBeenLastCalledWith(expect.stringContaining("片段 01 · 首帧"));
    expect(confirm).toHaveBeenLastCalledWith(expect.stringContaining("片段 03 · 参考图片 1"));
    expect(onDelete).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "移出素材库" }));
    expect(onDelete).toHaveBeenCalledWith([imageAsset.id]);
  });
});
