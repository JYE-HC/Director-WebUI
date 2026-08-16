import {
  classifyDroppedFiles,
  DIRECTOR_ASSET_MIME,
  DIRECTOR_ASSETS_MIME,
  DIRECTOR_SEGMENT_REFERENCE_MIME,
  directorAssetIdsFromTransfer,
  directorSegmentReferenceFromTransfer,
  uploadClassifiedDroppedFiles,
} from "../domain/assetDrag";
import type { AssetReference } from "../domain/modes";

function transfer(values: Record<string, string>) {
  return {
    getData: (type: string) => values[type] ?? "",
  };
}

describe("素材拖放载荷", () => {
  it("优先读取完整选择集，保持顺序并去重", () => {
    expect(directorAssetIdsFromTransfer(transfer({
      [DIRECTOR_ASSETS_MIME]: JSON.stringify(["video", "image", "video", "", 3]),
      [DIRECTOR_ASSET_MIME]: "legacy",
    }))).toEqual(["video", "image"]);
  });

  it("批量载荷损坏时兼容单素材载荷", () => {
    expect(directorAssetIdsFromTransfer(transfer({
      [DIRECTOR_ASSETS_MIME]: "{bad-json",
      [DIRECTOR_ASSET_MIME]: " legacy ",
    }))).toEqual(["legacy"]);
  });

  it("片段内部引用拖动使用独立载荷，不会退化为新增绑定", () => {
    const payload = {
      segmentId: "segment-a",
      assetId: "picture-a",
      target: "reference_image",
    };
    expect(directorSegmentReferenceFromTransfer(transfer({
      [DIRECTOR_SEGMENT_REFERENCE_MIME]: JSON.stringify(payload),
      [DIRECTOR_ASSET_MIME]: "library-copy",
    }))).toEqual(payload);
    expect(directorAssetIdsFromTransfer(transfer({
      [DIRECTOR_SEGMENT_REFERENCE_MIME]: JSON.stringify(payload),
    }))).toEqual([]);
  });

  it("按 MIME 或扩展名识别外部文件，并保留不支持项供界面反馈", () => {
    const image = new File(["image"], "frame.png", { type: "" });
    const video = new File(["video"], "shot.bin", { type: "video/mp4" });
    const audio = new File(["audio"], "voice.flac", { type: "application/octet-stream" });
    const unsupported = new File(["text"], "notes.txt", { type: "text/plain" });
    const result = classifyDroppedFiles([image, video, audio, unsupported]);

    expect(result.accepted.map(({ file, kind }) => [file.name, kind])).toEqual([
      ["frame.png", "image"],
      ["shot.bin", "video"],
      ["voice.flac", "audio"],
    ]);
    expect(result.unsupported).toEqual([unsupported]);
  });

  it("单个上传失败不丢前后成功项", async () => {
    const first = new File(["a"], "a.png", { type: "image/png" });
    const failed = new File(["b"], "b.wav", { type: "audio/wav" });
    const last = new File(["c"], "c.mp4", { type: "video/mp4" });
    const entries = classifyDroppedFiles([first, failed, last]).accepted;
    const upload = vi.fn(async (file: File, kind: AssetReference["kind"]) => {
      if (file === failed) throw new Error("音频损坏");
      return {
        id: `asset-${file.name}`,
        name: file.name,
        subfolder: "director-web",
        type: "input" as const,
        kind,
        ...(kind === "video" ? {
          metadata: {
            duration: 1,
            native_fps: 24,
            frame_count: 24,
            width: 640,
            height: 480,
            probe_method: "test",
            has_audio: false,
          },
        } : {}),
      } as AssetReference;
    });

    const result = await uploadClassifiedDroppedFiles(entries, upload, () => true);
    expect(result.assets.map((asset) => asset.name)).toEqual(["a.png", "c.mp4"]);
    expect(result.failures).toEqual([{ file_name: "b.wav", message: "音频损坏" }]);
    expect(result.authority_stale).toBe(false);
    expect(upload).toHaveBeenCalledTimes(3);
  });

  it("设置世代变化后丢弃整批响应，防止旧 endpoint 素材混入", async () => {
    const file = new File(["a"], "a.png", { type: "image/png" });
    let current = true;
    const upload = vi.fn(async () => {
      current = false;
      return {
        id: "old-endpoint-asset",
        name: file.name,
        subfolder: "director-web",
        type: "input" as const,
        kind: "image" as const,
      };
    });

    const result = await uploadClassifiedDroppedFiles(
      classifyDroppedFiles([file]).accepted,
      upload,
      () => current,
    );
    expect(result).toEqual({ assets: [], failures: [], authority_stale: true });
  });

  it("批量上传最多并发两个文件并保持结果顺序", async () => {
    const files = ["a.png", "b.png", "c.png", "d.png"].map(
      (name) => new File([name], name, { type: "image/png" }),
    );
    const entries = classifyDroppedFiles(files).accepted;
    let active = 0;
    let maxActive = 0;
    const releases: Array<() => void> = [];
    const upload = vi.fn(async (file: File) => {
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise<void>((resolve) => releases.push(resolve));
      active -= 1;
      return {
        id: `asset-${file.name}`,
        name: file.name,
        subfolder: "director-web",
        type: "input" as const,
        kind: "image" as const,
      };
    });
    const pending = uploadClassifiedDroppedFiles(entries, upload, () => true);
    await vi.waitFor(() => expect(upload).toHaveBeenCalledTimes(2));
    releases.splice(0).forEach((release) => release());
    await vi.waitFor(() => expect(upload).toHaveBeenCalledTimes(4));
    releases.splice(0).forEach((release) => release());

    const result = await pending;
    expect(maxActive).toBe(2);
    expect(result.assets.map((asset) => asset.name)).toEqual(files.map((file) => file.name));
  });
});
