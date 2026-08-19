import type { AssetReference } from "../domain/modes";
import {
  assignAssetsToSegment,
  assignAssetToSegment,
  createTimelineEditorState,
  createTimelineSegment,
  timelineEditorReducer,
  validateTimelineProject,
} from "../domain/timelineProject";

const image = (id: string): AssetReference => ({
  id,
  name: `${id}.png`,
  subfolder: "directordeck",
  type: "input",
  kind: "image",
});
const audio = (id: string): AssetReference => ({
  id,
  name: `${id}.wav`,
  subfolder: "directordeck",
  type: "input",
  kind: "audio",
});
const video = (id: string): AssetReference => ({
  id,
  name: `${id}.mp4`,
  subfolder: "directordeck",
  type: "input",
  kind: "video",
  metadata: {
    duration: 10,
    native_fps: 24,
    frame_count: 240,
    width: 864,
    height: 480,
    probe_method: "test",
    has_audio: true,
  },
});

describe("多素材原子分类绑定", () => {
  it("Ref2VA 将首个视频设为源，其余视频/图片/音频放入对应参考", () => {
    const assets = [image("picture"), video("source"), audio("voice"), video("motion")];
    const result = assignAssetsToSegment(createTimelineSegment("ref2va", 1), assets);

    expect(result.accepted).toEqual(assets);
    expect(result.rejected).toEqual([]);
    expect(result.segment).toMatchObject({
      mode: "ref2va",
      source_video: { id: "source" },
      reference_images: [{ id: "picture", slot: 0 }],
      reference_audios: [{ id: "voice", slot: 0 }],
      reference_videos: [{ id: "motion", slot: 0 }],
    });
  });

  it("FL2VA 自动只填空的首尾锚点，并明确报告不兼容与容量", () => {
    const result = assignAssetsToSegment(
      createTimelineSegment("fl2va", 1),
      [image("first"), audio("wrong"), image("last"), image("overflow")],
    );
    expect(result.segment).toMatchObject({
      first_image: { id: "first" },
      last_image: { id: "last" },
    });
    expect(result.rejected.map(({ asset, reason }) => [asset.id, reason])).toEqual([
      ["wrong", "incompatible"],
      ["overflow", "capacity"],
    ]);
  });

  it("显式参考图片区只绑定图片，全部成功上传仍原子进入素材库", () => {
    const state = createTimelineEditorState();
    const segment = createTimelineSegment("ref2va", 1);
    state.project.segments = [segment];
    const uploads = [image("picture"), audio("voice")];
    const next = timelineEditorReducer(state, {
      type: "segment/bind-assets",
      id: segment.id,
      assets: uploads,
      target: "reference_image",
      select: true,
    });

    expect(next.assets).toEqual(uploads);
    expect(next.project.segments[0]).toMatchObject({
      reference_images: [{ id: "picture", slot: 0 }],
      reference_audios: [],
    });
    expect(next.selected_asset_ids).toEqual([]);
  });

  it.each(["auto", "source_video"] as const)(
    "首次通过 %s 绑定源视频时保持独立视频标签的素材身份",
    (target) => {
      const segment = createTimelineSegment("ref2va", 1);
      segment.reference_videos = [
        { ...video("reference-a"), slot: 0 },
        { ...video("reference-b"), slot: 1 },
      ];
      segment.prompt = "先看 <Video 1>，再看 <Video 2>。";

      const result = assignAssetsToSegment(segment, [video("source")], target);

      expect(result.segment).toMatchObject({
        source_video: { id: "source" },
        prompt: "先看 <Video 2>，再看 <Video 3>。",
      });
      const replaced = assignAssetToSegment(
        result.segment,
        video("replacement-source"),
        "source_video",
      );
      expect(replaced.prompt).toBe("先看 <Video 2>，再看 <Video 3>。");
    },
  );

  it("拒绝同一稳定素材 ID 占用多个参考槽或同时充当源与参考", () => {
    const state = createTimelineEditorState();
    const segment = createTimelineSegment("ref2va", 1);
    const duplicate = video("duplicate");
    segment.prompt = "<Video 1> 与 <Video 2>";
    segment.source_video = duplicate;
    segment.reference_videos = [
      { ...duplicate, slot: 0 },
      { ...duplicate, slot: 1 },
    ];
    state.project.segments = [segment];

    expect(validateTimelineProject(state.project)).toEqual(
      expect.arrayContaining([
        expect.stringContaining("同一素材 ID 只能占用一个槽位"),
        expect.stringContaining("源视频不能同时占用独立参考视频槽位"),
      ]),
    );
  });

  it("按 MiniMax H3 的 9 图、3 音频、3 路视频上限稳定接纳并拒绝溢出", () => {
    const source = video("source");
    const pictures = Array.from({ length: 10 }, (_, index) => image(`picture-${index}`));
    const audios = Array.from({ length: 4 }, (_, index) => audio(`audio-${index}`));
    const references = Array.from({ length: 3 }, (_, index) => video(`reference-${index}`));

    const result = assignAssetsToSegment(
      createTimelineSegment("ref2va", 1),
      [source, ...pictures, ...audios, ...references],
    );

    expect(result.segment).toMatchObject({
      source_video: { id: source.id },
      reference_images: pictures.slice(0, 9).map((asset, slot) => ({ ...asset, slot })),
      reference_audios: audios.slice(0, 3).map((asset, slot) => ({ ...asset, slot })),
      reference_videos: references.slice(0, 2).map((asset, slot) => ({ ...asset, slot })),
    });
    expect(result.rejected.map(({ asset, reason }) => [asset.id, reason])).toEqual([
      [pictures[9].id, "capacity"],
      [audios[3].id, "capacity"],
      [references[2].id, "capacity"],
    ]);
  });

  it("三路独立参考视频已满时拒绝首次加入源视频，且不改写提示词", () => {
    const segment = createTimelineSegment("ref2va", 1);
    segment.reference_videos = [0, 1, 2].map((slot) => ({
      ...video(`reference-${slot}`),
      slot,
    }));
    segment.prompt = "依次参考 <Video 1>、<Video 2>、<Video 3>。";

    const result = assignAssetsToSegment(segment, [video("source")], "source_video");

    expect(result.segment).toBe(segment);
    expect(result.segment.prompt).toBe("依次参考 <Video 1>、<Video 2>、<Video 3>。");
    expect(result.rejected).toEqual([
      { asset: video("source"), reason: "capacity" },
    ]);
  });
});
