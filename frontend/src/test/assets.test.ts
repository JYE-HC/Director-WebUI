import {
  appendToLowestFreeSlots,
  fitSourceRangeToVideo,
  normalizeAssetReference,
  sanitizeDraftAssetReferences,
} from "../domain/assets";
import {
  createInitialDrafts,
  type AssetReference,
  type SlottedAssetReference,
  type VideoMetadata,
} from "../domain/modes";

const metadata: VideoMetadata = {
  duration: 12,
  native_fps: 30,
  frame_count: 360,
  width: 1920,
  height: 1080,
  probe_method: "ffprobe_nb_frames",
  has_audio: true,
};

function asset(id: string, kind: "image" | "audio" | "video"): AssetReference {
  return {
    id,
    name: `${id}.${kind === "image" ? "png" : kind === "audio" ? "wav" : "mp4"}`,
    subfolder: "directordeck",
    type: "input",
    kind,
    ...(kind === "video" ? { metadata } : {}),
  };
}

describe("媒体素材运行时契约", () => {
  it("完整保留合法视频 metadata，并拒绝任一非法探测字段", () => {
    const video = {
      ...asset("video-1", "video"),
      content_hash: `sha256:${"a".repeat(64)}`,
    };
    expect(normalizeAssetReference(video, "video")).toEqual(video);

    const invalidValues = [
      { duration: 0 },
      { native_fps: Number.NaN },
      { frame_count: 1.5 },
      { width: 0 },
      { height: -1 },
      { probe_method: " " },
    ];
    for (const patch of invalidValues) {
      expect(
        normalizeAssetReference(
          { ...video, metadata: { ...metadata, ...patch } },
          "video",
        ),
      ).toBeNull();
    }
    expect(normalizeAssetReference({ ...video, metadata: undefined }, "video")).toBeNull();
  });

  it("保留 null 或小写 sha256 内容摘要，并拒绝非规范摘要", () => {
    const image = asset("image-hash", "image");
    expect(normalizeAssetReference({ ...image, content_hash: null }, "image"))
      .toEqual({ ...image, content_hash: null });
    expect(normalizeAssetReference({ ...image, content_hash: `sha256:${"b".repeat(64)}` }, "image"))
      .toEqual({ ...image, content_hash: `sha256:${"b".repeat(64)}` });
    expect(normalizeAssetReference({ ...image, content_hash: `sha256:${"B".repeat(64)}` }, "image"))
      .toBeNull();
    expect(normalizeAssetReference({ ...image, content_hash: "sha256:short" }, "image"))
      .toBeNull();
  });

  it("图片和音频会丢弃 metadata，普通素材也不会保留 slot", () => {
    const image = { ...asset("image-1", "image"), metadata, slot: 7 };
    expect(normalizeAssetReference(image, "image")).toEqual(asset("image-1", "image"));
  });

  it("保留显式槽位，为旧草稿缺槽引用分配最低空位，并丢弃重复/越界槽", () => {
    const draft = createInitialDrafts().r2v;
    draft.shots[0].reference_images = [
      { ...asset("explicit-3", "image"), slot: 3 },
      asset("legacy", "image") as SlottedAssetReference,
      { ...asset("duplicate-3", "image"), slot: 3 },
      { ...asset("out-of-range", "image"), slot: 9 },
      { ...asset("explicit-0", "image"), slot: 0 },
    ];

    const sanitized = sanitizeDraftAssetReferences(draft);
    expect(sanitized.shots[0].reference_images.map(({ id, slot }) => ({ id, slot }))).toEqual([
      { id: "explicit-3", slot: 3 },
      { id: "legacy", slot: 1 },
      { id: "explicit-0", slot: 0 },
    ]);
  });

  it("删除不重排已有槽，新上传占用最低空闲槽", () => {
    const current: SlottedAssetReference[] = [
      { ...asset("picture-1", "image"), slot: 0 },
      { ...asset("picture-3", "image"), slot: 2 },
    ];
    const next = appendToLowestFreeSlots(current, [asset("picture-2", "image")], 9);
    expect(next.map(({ id, slot }) => ({ id, slot }))).toEqual([
      { id: "picture-1", slot: 0 },
      { id: "picture-3", slot: 2 },
      { id: "picture-2", slot: 1 },
    ]);

    const afterDelete = next.filter((item) => item.slot !== 0);
    const afterUpload = appendToLowestFreeSlots(afterDelete, [asset("new-picture-1", "image")], 9);
    expect(afterUpload.map(({ id, slot }) => ({ id, slot }))).toEqual([
      { id: "picture-3", slot: 2 },
      { id: "picture-2", slot: 1 },
      { id: "new-picture-1", slot: 0 },
    ]);
  });

  it("新源视频只在当前范围越界时把时长收敛到剩余可用时长", () => {
    const video = asset("video-1", "video");
    expect(fitSourceRangeToVideo(video, 2, 5)).toEqual({
      source_start_seconds: 2,
      source_duration_seconds: 5,
    });
    expect(fitSourceRangeToVideo(video, 10, 5)).toEqual({
      source_start_seconds: 10,
      source_duration_seconds: 2,
    });
    expect(fitSourceRangeToVideo(video, 13, 5)).toEqual({
      source_start_seconds: 0,
      source_duration_seconds: 5,
    });
    expect(fitSourceRangeToVideo({ ...video, metadata: undefined }, 0, 5)).toBeNull();
  });
});
