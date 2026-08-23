export const MODE_ORDER = ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"] as const;

export type GenerationMode = (typeof MODE_ORDER)[number];

export interface ModeMeta {
  label: string;
  shortLabel: string;
  description: string;
  accent: string;
}

export const MODE_META: Record<GenerationMode, ModeMeta> = {
  t2v: { label: "文生视频", shortLabel: "T2V", description: "从文字描述直接生成完整镜头", accent: "amber" },
  i2v: { label: "图生视频", shortLabel: "I2V", description: "为每个镜头指定独立起始画面", accent: "coral" },
  fl2v: { label: "首尾帧视频", shortLabel: "FL2V", description: "逐镜头定义首尾视觉锚点", accent: "violet" },
  r2v: { label: "参考生视频", shortLabel: "R2V", description: "为每组镜头绑定图像、音频与视频参考", accent: "cyan" },
  v2v: { label: "视频重绘", shortLabel: "V2V", description: "按源视频时间范围重新演绎画面", accent: "blue" },
  rv2v: { label: "参考视频重绘", shortLabel: "RV2V", description: "逐段结合源视频与图像、音频参考", accent: "green" },
};

export type AssetKind = "image" | "audio" | "video";

export interface VideoMetadata {
  duration: number;
  native_fps: number;
  frame_count: number;
  width: number;
  height: number;
  probe_method: string;
  /** Server-probed stream fact; historical assets without it are silent. */
  has_audio: boolean;
}

/** ComfyUI input identity returned by POST /api/assets. */
export interface AssetReference {
  id: string;
  name: string;
  subfolder: string;
  type: "input";
  kind: AssetKind;
  filename?: string | null;
  path?: string | null;
  preview_url?: string | null;
  /** Immutable upload identity when known; legacy assets may predate hashing. */
  content_hash?: string | null;
  /** Video probe data; the complete v5 wire uses null for image/audio assets. */
  metadata?: VideoMetadata | null;
}

/** A reference asset whose zero-based slot remains stable after deletions. */
export interface SlottedAssetReference extends AssetReference {
  slot: number;
}

export interface RenderConfig {
  width: number;
  height: number;
  fps: number;
}

export const SAMPLING_SCHEDULERS = ["simple", "normal", "karras", "beta"] as const;
export type SamplingScheduler = (typeof SAMPLING_SCHEDULERS)[number];

export function isSamplingScheduler(value: unknown): value is SamplingScheduler {
  return typeof value === "string" &&
    (SAMPLING_SCHEDULERS as readonly string[]).includes(value);
}

export interface SamplingConfig {
  steps: number;
  seed: number;
  random_seed: boolean;
  sampler: "res_multistep" | "euler" | "dpmpp_2m";
  scheduler: SamplingScheduler;
  shift: number;
  audio_shift: number;
}

/** Browser-safe 53-bit seed that can round-trip through the JSON API exactly. */
export function randomSafeSeed(): number {
  const words = new Uint32Array(2);
  globalThis.crypto.getRandomValues(words);
  return (words[0] & 0x1fffff) * 0x1_0000_0000 + words[1];
}

export interface ShotBase {
  id: string;
  title: string;
  prompt: string;
  duration_seconds: number;
  enabled: boolean;
}

export type T2VShot = ShotBase;

export interface I2VShot extends ShotBase {
  first_image: AssetReference | null;
}

export interface FL2VShot extends ShotBase {
  first_image: AssetReference | null;
  last_image: AssetReference | null;
}

export interface R2VShot extends ShotBase {
  reference_images: SlottedAssetReference[];
  reference_audios: SlottedAssetReference[];
  reference_videos: SlottedAssetReference[];
}

export interface V2VShot extends ShotBase {
  source_video: AssetReference | null;
  source_start_seconds: number;
  source_duration_seconds: number;
}

export interface RV2VShot extends ShotBase {
  source_video: AssetReference | null;
  source_start_seconds: number;
  source_duration_seconds: number;
  reference_images: SlottedAssetReference[];
  reference_audios: SlottedAssetReference[];
}

interface DraftBase<M extends GenerationMode, T extends ShotBase> {
  mode: M;
  prompt: string;
  ref_image_size: "match" | "max";
  render: RenderConfig;
  sampling: SamplingConfig;
  shots: T[];
}

export type T2VDraft = DraftBase<"t2v", T2VShot>;
export type I2VDraft = DraftBase<"i2v", I2VShot>;
export type FL2VDraft = DraftBase<"fl2v", FL2VShot>;
export type R2VDraft = DraftBase<"r2v", R2VShot>;
export type V2VDraft = DraftBase<"v2v", V2VShot>;
export type RV2VDraft = DraftBase<"rv2v", RV2VShot>;

export interface ModeDraftMap {
  t2v: T2VDraft;
  i2v: I2VDraft;
  fl2v: FL2VDraft;
  r2v: R2VDraft;
  v2v: V2VDraft;
  rv2v: RV2VDraft;
}

export type ModeDraft = ModeDraftMap[GenerationMode];
export type ModeShot = ModeDraft["shots"][number];

function commonDraft<M extends GenerationMode>(mode: M) {
  return {
    mode,
    prompt: "",
    ref_image_size: "match" as const,
    render: { width: 864, height: 480, fps: 24 },
    sampling: {
      steps: 25,
      seed: randomSafeSeed(),
      random_seed: true,
      sampler: "res_multistep" as const,
      scheduler: "simple" as const,
      shift: 12,
      audio_shift: 3,
    },
  };
}

function commonShot(mode: GenerationMode, index: number): ShotBase {
  return {
    id: createLocalId(`${mode}-shot`),
    title: `镜头 ${String(index).padStart(2, "0")}`,
    prompt: "",
    duration_seconds: 5,
    enabled: true,
  };
}

export function createTimelineItem(mode: "t2v", index: number): T2VShot;
export function createTimelineItem(mode: "i2v", index: number): I2VShot;
export function createTimelineItem(mode: "fl2v", index: number): FL2VShot;
export function createTimelineItem(mode: "r2v", index: number): R2VShot;
export function createTimelineItem(mode: "v2v", index: number): V2VShot;
export function createTimelineItem(mode: "rv2v", index: number): RV2VShot;
export function createTimelineItem(mode: GenerationMode, index: number): ModeShot {
  const base = commonShot(mode, index);
  switch (mode) {
    case "t2v":
      return base;
    case "i2v":
      return { ...base, first_image: null };
    case "fl2v":
      return { ...base, first_image: null, last_image: null };
    case "r2v":
      return { ...base, reference_images: [], reference_audios: [], reference_videos: [] };
    case "v2v":
      return { ...base, source_video: null, source_start_seconds: 0, source_duration_seconds: 5 };
    case "rv2v":
      return {
        ...base,
        source_video: null,
        source_start_seconds: 0,
        source_duration_seconds: 5,
        reference_images: [],
        reference_audios: [],
      };
  }
}

export function createInitialDrafts(): ModeDraftMap {
  return {
    t2v: { ...commonDraft("t2v"), shots: [createTimelineItem("t2v", 1)] },
    i2v: { ...commonDraft("i2v"), shots: [createTimelineItem("i2v", 1)] },
    fl2v: { ...commonDraft("fl2v"), shots: [createTimelineItem("fl2v", 1)] },
    r2v: { ...commonDraft("r2v"), shots: [createTimelineItem("r2v", 1)] },
    v2v: { ...commonDraft("v2v"), shots: [createTimelineItem("v2v", 1)] },
    rv2v: { ...commonDraft("rv2v"), shots: [createTimelineItem("rv2v", 1)] },
  };
}

export function createLocalId(prefix: string): string {
  const random = Math.random().toString(36).slice(2, 9);
  return `${prefix}-${Date.now().toString(36)}-${random}`;
}
