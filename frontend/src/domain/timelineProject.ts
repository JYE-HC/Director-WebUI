import {
  createLocalId,
  isSamplingScheduler,
  randomSafeSeed,
  type AssetReference,
  type GenerationMode as LegacyGenerationMode,
  type RenderConfig,
  type SamplingConfig,
  type SlottedAssetReference,
} from "./modes";
import {
  appendToLowestFreeSlots,
  fitSourceRangeToVideo,
  isStableAssetReference,
  isStableSlottedAssetReference,
  normalizeAssetReference,
  normalizeSlottedAssetList,
} from "./assets";
import {
  alignH3FrameCount,
  alignH3Frames,
  H3_MAX_SHOT_FRAMES,
  roundPositiveHalfEven,
} from "./timing";
import {
  MINIMAX_H3_REFERENCE_LIMITS,
  maxSlotForCapacity,
  minimaxH3ReferenceCapacities,
} from "./h3Capabilities";
import {
  limitPromptCharacters,
  MINIMAX_H3_PROMPT_MAX_CHARACTERS,
  promptCharacterCount,
} from "./promptLimits";

export type AssetGridSize = "small" | "medium" | "large";

export const TIMELINE_MODE_ORDER = ["fl2va", "ref2va"] as const;

/** Backend id of the pre-multi-project singleton timeline (the first project). */
export const DEFAULT_PROJECT_ID = "default";

export type TimelineGenerationMode = (typeof TIMELINE_MODE_ORDER)[number];
export type DerivedGenerationRecipe = LegacyGenerationMode;

export type TimelineOutputAspect = "16:9" | "9:16";

/** Exact multiple-of-32 output tiers exposed by the native Director node. */
export const H3_OUTPUT_RESOLUTIONS_16_9 = [
  [608, 352],
  [736, 416],
  [864, 480],
  [960, 544],
  [1056, 608],
  [1152, 640],
  [1216, 672],
  [1280, 736],
  [1344, 768],
  [1376, 768],
  [1504, 832],
  [1664, 928],
  [1824, 1024],
  [1920, 1088],
] as const;

export interface TimelineOutputResolution {
  width: number;
  height: number;
}

export function timelineOutputResolutions(
  aspect: TimelineOutputAspect,
): TimelineOutputResolution[] {
  return H3_OUTPUT_RESOLUTIONS_16_9.map(([landscapeWidth, landscapeHeight]) =>
    aspect === "16:9"
      ? { width: landscapeWidth, height: landscapeHeight }
      : { width: landscapeHeight, height: landscapeWidth },
  );
}

/**
 * Keeps imported custom dimensions visible without claiming they are a native
 * preset. Orientation is still useful for deciding which preset list to show.
 */
export function inferTimelineOutputAspect(
  width: number,
  height: number,
): TimelineOutputAspect | null {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0)
    return null;
  if (H3_OUTPUT_RESOLUTIONS_16_9.some(
    ([landscapeWidth, landscapeHeight]) =>
      landscapeWidth === width && landscapeHeight === height,
  ) || width * 9 === height * 16) return "16:9";
  if (H3_OUTPUT_RESOLUTIONS_16_9.some(
    ([landscapeWidth, landscapeHeight]) =>
      landscapeHeight === width && landscapeWidth === height,
  ) || width * 16 === height * 9) return "9:16";
  return null;
}

export function isTimelineOutputResolution(
  width: number,
  height: number,
  aspect = inferTimelineOutputAspect(width, height),
): boolean {
  return aspect !== null && timelineOutputResolutions(aspect).some(
    (resolution) => resolution.width === width && resolution.height === height,
  );
}

/** Chooses the native tier with the nearest pixel area after an explicit aspect change. */
export function closestTimelineOutputResolution(
  width: number,
  height: number,
  aspect: TimelineOutputAspect,
): TimelineOutputResolution {
  const targetArea = Number.isFinite(width * height) && width > 0 && height > 0
    ? width * height
    : 864 * 480;
  return timelineOutputResolutions(aspect).reduce((closest, candidate) =>
    Math.abs(candidate.width * candidate.height - targetArea) <
      Math.abs(closest.width * closest.height - targetArea)
      ? candidate
      : closest,
  );
}

export interface TimelineModeMeta {
  label: string;
  shortLabel: string;
  description: string;
  accent: string;
}

export const TIMELINE_MODE_META: Record<TimelineGenerationMode, TimelineModeMeta> = {
  fl2va: {
    label: "文 / 图生视频",
    shortLabel: "FL2VA",
    description: "按首帧与尾帧素材自动选择文生、图生或首尾帧生成",
    accent: "violet",
  },
  ref2va: {
    label: "参考生视频",
    shortLabel: "Ref2VA",
    description: "按源视频与独立参考素材自动选择参考生成或视频重绘",
    accent: "cyan",
  },
};

interface TimelineSegmentBase<M extends TimelineGenerationMode> {
  id: string;
  mode: M;
  title: string;
  prompt: string;
  duration_seconds: number;
  enabled: boolean;
  continuity: TimelineSegmentContinuity;
  ref_image_size: "match" | "max";
  audio_mode: "generate" | "source" | "mute";
}

export interface TimelineSegmentContinuity {
  enabled: boolean;
  overlap_frames: 5 | 22 | 39 | 56;
}

export interface FL2VASegment extends TimelineSegmentBase<"fl2va"> {
  first_image: AssetReference | null;
  last_image: AssetReference | null;
}

export interface Ref2VASegment extends TimelineSegmentBase<"ref2va"> {
  source_video: AssetReference | null;
  source_start_seconds: number;
  source_duration_seconds: number;
  source_audio_as_reference: boolean;
  reference_images: SlottedAssetReference[];
  reference_audios: SlottedAssetReference[];
  reference_videos: SlottedAssetReference[];
}

export type TimelineSegment = FL2VASegment | Ref2VASegment;

export type SourceVideoSegment = Ref2VASegment;

export interface TimelineProject {
  version: 4;
  title: string;
  render: RenderConfig;
  sampling: {
    fl2va: SamplingConfig;
    ref2va: SamplingConfig;
  };
  export_mode: "all" | "segments";
  segments: TimelineSegment[];
}

export type TimelineContinuityBoundaryKind =
  | "eligible"
  | "explicit-first-image";

/**
 * One physical boundary on the active program track. Disabled segments do not
 * occupy program time, so the immediately preceding enabled segment is the
 * only possible AddGuide source for the target.
 */
export interface TimelineContinuityBoundary {
  predecessor: TimelineSegment;
  predecessor_index: number;
  segment: TimelineSegment;
  segment_index: number;
  kind: TimelineContinuityBoundaryKind;
}

export interface TimelineContinuityRunIssue {
  code:
    | "historical-take-required"
    | "predecessor-too-short"
    | "sample-too-long";
  boundary: TimelineContinuityBoundary;
  message: string;
}

export interface TimelineEditorState {
  project: TimelineProject;
  assets: AssetReference[];
  /** Editing focus/selection. Direct clip gestures mirror enabled IDs into the run set. */
  selected_segment_ids: string[];
  selection_anchor_id: string | null;
  /** Browser-persisted run set, intentionally kept outside server project data. */
  run_selected_segment_ids: string[];
  selected_asset_ids: string[];
  asset_grid_size: AssetGridSize;
  playhead_seconds: number;
}

export const DEFAULT_TIMELINE_SAMPLING: SamplingConfig = {
  steps: 25,
  seed: 0,
  random_seed: true,
  sampler: "res_multistep",
  scheduler: "simple",
  shift: 12,
  audio_shift: 3,
};

export function timelineSamplingFamily(
  mode: TimelineGenerationMode,
): keyof TimelineProject["sampling"] {
  return mode;
}

/**
 * Derives the concrete native H3 recipe from the stable two-family editor
 * contract. The browser never stores the six recipes as mutable segment mode.
 */
export function deriveSegmentRecipe(
  segment: TimelineSegment,
): DerivedGenerationRecipe {
  if (segment.mode === "fl2va") {
    if (segment.last_image) return "fl2v";
    return segment.first_image ? "i2v" : "t2v";
  }
  const hasReferences = Boolean(
    segment.reference_images.length ||
    segment.reference_audios.length ||
    segment.reference_videos.length,
  );
  if (!segment.source_video) return "r2v";
  return hasReferences ? "rv2v" : "v2v";
}

/**
 * Segment identities outlive array positions and may be matched against
 * server-side generated takes. Prefer a collision-resistant UUID while
 * retaining the legacy generator for older browsers and test environments.
 */
function createSegmentId(): string {
  try {
    const uuid = globalThis.crypto?.randomUUID?.();
    if (uuid) return `segment-${uuid}`;
  } catch {
    // Some embedded browser shells expose crypto but deny randomUUID().
  }
  return createLocalId("segment");
}

/** Ref2VA 全参考模式的六段式空提示词骨架。 */
export const EMPTY_SIX_SECTION_PROMPT = [
  "subject_definitions:",
  "",
  "summary:",
  "",
  "retention_analysis:",
  "",
  "detailed_description:",
  "",
  "overall_soundscape:",
  "",
  "non_diegetic_music:",
].join("\n");

export interface PromptSubjectReference {
  number: number;
  token: string;
  definition: string;
}

const H3_PROMPT_SECTION = /^\s*([a-z][a-z0-9_]*)\s*:\s*$/i;
const H3_SUBJECT_DEFINITION = /^\s*<\s*(subject)\s+(\d+)\s*>\s*(.*)$/i;

/** Reads canonical subject tags only from the prompt's subject_definitions block. */
export function promptSubjectReferences(prompt: string): PromptSubjectReference[] {
  const references: PromptSubjectReference[] = [];
  const seen = new Set<number>();
  let inSubjectDefinitions = false;

  for (const line of prompt.split(/\r?\n/)) {
    const section = line.match(H3_PROMPT_SECTION);
    if (section) {
      inSubjectDefinitions = section[1].toLocaleLowerCase() === "subject_definitions";
      continue;
    }
    if (!inSubjectDefinitions) continue;
    const definition = line.match(H3_SUBJECT_DEFINITION);
    if (!definition) continue;
    const number = Number(definition[2]);
    if (!Number.isSafeInteger(number) || number < 1 || seen.has(number)) continue;
    seen.add(number);
    references.push({
      number,
      token: `<${definition[1]} ${number}>`,
      definition: definition[3].trim(),
    });
  }
  return references;
}

/** fl2va 三段主体（T2V / I2V / FL2V 共享）。 */
const FL2VA_CORE_SECTIONS = [
  "integrated_multimodal_description:",
  "",
  "overall_soundscape:",
  "",
  "non_diegetic_music:",
].join("\n");

/** I2V 首帧对齐指令（H3 base 规范原文）。 */
const I2V_ALIGNMENT_INSTRUCTION =
  "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.";
/** FL2V 首尾帧对齐指令。 */
const FL2V_ALIGNMENT_INSTRUCTION =
  "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.";
/** 仅尾图（L2VA）收敛到尾帧对齐指令。 */
const L2V_ALIGNMENT_INSTRUCTION =
  "How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.";

/**
 * 按片段模式与 recipe 生成空提示词骨架：ref2va 六段式；fl2va 三段式并
 * 附带对应首行图片对齐指令（t2v 无 / i2v 首帧 / fl2v 首尾或仅尾）。
 */
export function promptSkeleton(segment: TimelineSegment): string {
  if (segment.mode === "ref2va") return EMPTY_SIX_SECTION_PROMPT;
  const recipe = deriveSegmentRecipe(segment);
  if (recipe === "t2v") return FL2VA_CORE_SECTIONS;
  if (recipe === "i2v") {
    return `${I2V_ALIGNMENT_INSTRUCTION}\n\n${FL2VA_CORE_SECTIONS}`;
  }
  return segment.first_image
    ? `${FL2V_ALIGNMENT_INSTRUCTION}\n\n${FL2VA_CORE_SECTIONS}`
    : `${L2V_ALIGNMENT_INSTRUCTION}\n\n${FL2VA_CORE_SECTIONS}`;
}

function segmentBase<M extends TimelineGenerationMode>(
  mode: M,
  index: number,
): TimelineSegmentBase<M> {
  return {
    id: createSegmentId(),
    mode,
    title: defaultTimelineSegmentTitle(index),
    prompt: "",
    duration_seconds: 5,
    enabled: true,
    continuity: { enabled: false, overlap_frames: 22 },
    ref_image_size: "match",
    audio_mode: "generate",
  };
}

function defaultTimelineSegmentTitle(index: number): string {
  return `片段 ${String(index).padStart(2, "0")}`;
}

function nextDefaultTimelineSegmentNumber(project: TimelineProject): number {
  let highest = 0;
  project.segments.forEach((segment) => {
    const match = /^片段 (\d+)$/.exec(segment.title);
    if (!match) return;
    const value = Number(match[1]);
    if (Number.isSafeInteger(value) && value > highest) highest = value;
  });
  if (highest < Number.MAX_SAFE_INTEGER) return highest + 1;

  // A manually entered MAX_SAFE_INTEGER title must not make the allocator
  // produce an imprecise number. At 128 segments this fallback is bounded.
  const titles = new Set(project.segments.map((segment) => segment.title));
  let fallback = 1;
  while (titles.has(defaultTimelineSegmentTitle(fallback))) fallback += 1;
  return fallback;
}

export function createTimelineSegment(
  mode: "fl2va",
  index: number,
): FL2VASegment;
export function createTimelineSegment(
  mode: "ref2va",
  index: number,
): Ref2VASegment;
export function createTimelineSegment(
  mode: TimelineGenerationMode,
  index: number,
): TimelineSegment;
export function createTimelineSegment(
  mode: TimelineGenerationMode,
  index: number,
): TimelineSegment {
  const base = segmentBase(mode, index);
  switch (mode) {
    case "fl2va":
      return { ...base, mode: "fl2va", first_image: null, last_image: null };
    case "ref2va":
      return {
        ...base,
        mode: "ref2va",
        source_video: null,
        source_start_seconds: 0,
        source_duration_seconds: 5,
        source_audio_as_reference: false,
        reference_images: [],
        reference_audios: [],
        reference_videos: [],
      };
  }
}

export function createTimelineProject(): TimelineProject {
  return {
    version: 4,
    title: "未命名长视频",
    render: { width: 864, height: 480, fps: 24 },
    sampling: {
      fl2va: { ...DEFAULT_TIMELINE_SAMPLING, seed: randomSafeSeed() },
      ref2va: { ...DEFAULT_TIMELINE_SAMPLING, seed: randomSafeSeed() },
    },
    export_mode: "all",
    segments: [createTimelineSegment("fl2va", 1)],
  };
}

export function createTimelineEditorState(): TimelineEditorState {
  const project = createTimelineProject();
  return {
    project,
    assets: [],
    selected_segment_ids: [project.segments[0].id],
    selection_anchor_id: project.segments[0].id,
    run_selected_segment_ids: [project.segments[0].id],
    selected_asset_ids: [],
    asset_grid_size: "medium",
    playhead_seconds: 0,
  };
}

/** Duration of the native H3 take produced for one segment. */
export function alignedTimelineSegmentDuration(
  segment: TimelineSegment,
  fps: number,
): number {
  if (!Number.isFinite(fps) || fps <= 0) return 0;
  return alignH3Frames(segment.duration_seconds, fps) / fps;
}

/**
 * Duration used by the editing/program timeline. Native-sized generation
 * segments show their aligned output time. A complete, unchanged source range
 * always occupies its exact source/edit duration, including when it is short
 * enough to generate: H3's 17k+5 padding is an execution detail and cannot
 * invent extra material frames on the editor timeline. Cropped/split segments
 * continue to show their aligned generated duration.
 */
export function timelineSegmentPlaybackDuration(
  segment: TimelineSegment,
  fps: number,
): number {
  if (!Number.isFinite(fps) || fps <= 0) return 0;
  if (segment.mode === "ref2va" && segment.source_video?.metadata) {
    const metadataDuration = segment.source_video.metadata.duration;
    const tolerance = Math.max(1e-6, metadataDuration * 1e-9);
    const completeSourceRange =
      Math.abs(segment.source_start_seconds) <= tolerance &&
      Math.abs(segment.source_duration_seconds - metadataDuration) <= tolerance &&
      Math.abs(segment.duration_seconds - segment.source_duration_seconds) <= tolerance;
    if (completeSourceRange) return Math.max(0, segment.duration_seconds);
  }
  const alignedFrames = alignH3Frames(segment.duration_seconds, fps);
  return alignedFrames / fps;
}

/** Disabled segments remain editable but do not occupy program/export time. */
export function effectiveTimelineSegmentDuration(
  segment: TimelineSegment,
  fps: number,
): number {
  return segment.enabled ? timelineSegmentPlaybackDuration(segment, fps) : 0;
}

export function segmentAssetReferences(segment: TimelineSegment): AssetReference[] {
  const bySlot = (assets: SlottedAssetReference[]) =>
    [...assets].sort((left, right) => left.slot - right.slot);
  switch (segment.mode) {
    case "fl2va":
      return [segment.first_image, segment.last_image].filter(
        (asset): asset is AssetReference => asset !== null,
      );
    case "ref2va":
      return [
        ...(segment.source_video ? [segment.source_video] : []),
        ...bySlot(segment.reference_images),
        ...bySlot(segment.reference_audios),
        ...bySlot(segment.reference_videos),
      ];
  }
}

function copyBase<M extends TimelineGenerationMode>(
  source: TimelineSegment,
  target: TimelineSegmentBase<M>,
): TimelineSegmentBase<M> {
  return {
    ...target,
    id: source.id,
    title: source.title,
    prompt: source.prompt,
    duration_seconds: source.duration_seconds,
    enabled: source.enabled,
    continuity: { ...source.continuity },
    ref_image_size: source.ref_image_size,
    audio_mode: source.audio_mode,
  };
}

/**
 * Switches mode through a fresh factory and copies only common fields.
 * This is the boundary that prevents FL2VA anchors or Ref2VA source/reference
 * fields from leaking into the other family's private configuration.
 */
export function changeSegmentMode(
  segment: TimelineSegment,
  mode: TimelineGenerationMode,
): TimelineSegment {
  if (segment.mode === mode) return segment;
  const fresh = createTimelineSegment(mode, 1);
  const base = copyBase(segment, fresh);
  switch (mode) {
    case "fl2va":
      return { ...base, mode: "fl2va", first_image: null, last_image: null };
    case "ref2va":
      return {
        ...base,
        mode: "ref2va",
        source_video: null,
        source_start_seconds: 0,
        source_duration_seconds: segment.duration_seconds,
        source_audio_as_reference: false,
        reference_images: [],
        reference_audios: [],
        reference_videos: [],
      };
  }
}

function timelineIndexById(project: TimelineProject, id: string): number {
  return project.segments.findIndex((segment) => segment.id === id);
}

export function selectTimelineSegment(
  state: TimelineEditorState,
  id: string,
  options: { additive?: boolean; range?: boolean } = {},
): TimelineEditorState {
  const index = timelineIndexById(state.project, id);
  if (index < 0) return state;
  if (options.range && state.selection_anchor_id) {
    const anchor = timelineIndexById(state.project, state.selection_anchor_id);
    if (anchor >= 0) {
      const [start, end] = anchor < index ? [anchor, index] : [index, anchor];
      const range = state.project.segments
        .slice(start, end + 1)
        .map((segment) => segment.id);
      return {
        ...state,
        selected_segment_ids: options.additive
          ? [...new Set([...state.selected_segment_ids, ...range])]
          : range,
      };
    }
  }
  if (options.additive) {
    const exists = state.selected_segment_ids.includes(id);
    const selected = exists
      ? state.selected_segment_ids.filter((candidate) => candidate !== id)
      : [...state.selected_segment_ids, id];
    return {
      ...state,
      selected_segment_ids: selected.length ? selected : [id],
      selection_anchor_id: id,
    };
  }
  return { ...state, selected_segment_ids: [id], selection_anchor_id: id };
}

function touchProject(project: TimelineProject, segments: TimelineSegment[]): TimelineProject {
  return { ...project, segments };
}

function enabledTimelineSegmentIds(project: TimelineProject): string[] {
  return project.segments
    .filter((segment) => segment.enabled)
    .map((segment) => segment.id);
}

/** Whether the current explicit run set represents every enabled segment. */
function runSelectionCoversAllEnabled(state: TimelineEditorState): boolean {
  const selected = new Set(state.run_selected_segment_ids);
  return enabledTimelineSegmentIds(state.project).every((id) => selected.has(id));
}

export function insertTimelineSegment(
  state: TimelineEditorState,
  position: "before" | "after",
  mode: TimelineGenerationMode = "fl2va",
): TimelineEditorState {
  if (state.project.segments.length >= 128) return state;
  const selectedAllForRun = runSelectionCoversAllEnabled(state);
  const selectedId = state.selected_segment_ids.at(-1);
  const selectedIndex = selectedId
    ? timelineIndexById(state.project, selectedId)
    : state.project.segments.length - 1;
  const insertionIndex = selectedIndex < 0
    ? state.project.segments.length
    : selectedIndex + (position === "after" ? 1 : 0);
  const segment = createTimelineSegment(mode, nextDefaultTimelineSegmentNumber(state.project));
  const segments = [...state.project.segments];
  segments.splice(insertionIndex, 0, segment);
  return {
    ...state,
    project: touchProject(state.project, segments),
    selected_segment_ids: [segment.id],
    selection_anchor_id: segment.id,
    run_selected_segment_ids: selectedAllForRun
      ? [...state.run_selected_segment_ids, segment.id]
      : state.run_selected_segment_ids,
  };
}

export function insertTimelineVideoAsset(
  state: TimelineEditorState,
  asset: AssetReference,
  position: "before" | "after" = "after",
): TimelineEditorState {
  return insertTimelineVideoAssetAtAnchor(
    state,
    asset,
    state.selected_segment_ids.at(-1) ?? null,
    position,
  );
}

/** Inserts and binds a source-backed Ref2VA segment in one reducer transition. */
export function insertTimelineVideoAssetAtAnchor(
  state: TimelineEditorState,
  asset: AssetReference,
  anchorId: string | null,
  position: "before" | "after" = "after",
): TimelineEditorState {
  if (asset.kind !== "video" || state.project.segments.length >= 128) return state;
  const selectedAllForRun = runSelectionCoversAllEnabled(state);
  const anchorIndex = anchorId ? timelineIndexById(state.project, anchorId) : -1;
  const fallbackIndex = state.selected_segment_ids.at(-1)
    ? timelineIndexById(state.project, state.selected_segment_ids.at(-1) as string)
    : state.project.segments.length - 1;
  const resolvedAnchor = anchorIndex >= 0 ? anchorIndex : fallbackIndex;
  const insertionIndex = resolvedAnchor < 0
    ? state.project.segments.length
    : resolvedAnchor + (position === "after" ? 1 : 0);
  // A source video is editing material, not a five-second reference sample.
  // Materialize its complete server-probed range on the timeline first;
  // compile validation remains the boundary that requires the user to split a
  // long source into native H3-sized generation segments.
  const duration = Math.max(0.01, asset.metadata?.duration ?? 5);
  const nextDefaultIndex = nextDefaultTimelineSegmentNumber(state.project);
  const segment: Ref2VASegment = {
    ...createTimelineSegment("ref2va", nextDefaultIndex),
    title: asset.name.replace(/\.[^.]+$/, "") || defaultTimelineSegmentTitle(
      nextDefaultIndex,
    ),
    duration_seconds: duration,
    source_video: asset,
    source_start_seconds: 0,
    source_duration_seconds: duration,
  };
  const segments = [...state.project.segments];
  segments.splice(insertionIndex, 0, segment);
  return {
    ...state,
    project: touchProject(state.project, segments),
    selected_segment_ids: [segment.id],
    selection_anchor_id: segment.id,
    run_selected_segment_ids: selectedAllForRun
      ? [...state.run_selected_segment_ids, segment.id]
      : state.run_selected_segment_ids,
  };
}

export function moveTimelineSegment(
  state: TimelineEditorState,
  draggedId: string,
  targetId: string,
): TimelineEditorState {
  const from = timelineIndexById(state.project, draggedId);
  const to = timelineIndexById(state.project, targetId);
  if (from < 0 || to < 0 || from === to) return state;
  const segments = [...state.project.segments];
  const [dragged] = segments.splice(from, 1);
  segments.splice(to, 0, dragged);
  return { ...state, project: touchProject(state.project, segments) };
}

export function deleteSelectedSegments(state: TimelineEditorState): TimelineEditorState {
  const selected = new Set(state.selected_segment_ids);
  if (!selected.size) return state;
  const selectedAllForRun = runSelectionCoversAllEnabled(state);
  let segments = state.project.segments.filter((segment) => !selected.has(segment.id));
  const createdFallback = segments.length === 0;
  if (createdFallback) segments = [createTimelineSegment("fl2va", 1)];
  const first = segments[Math.min(
    Math.max(0, timelineIndexById(state.project, state.selected_segment_ids[0])),
    segments.length - 1,
  )];
  return clampTimelinePlayhead({
    ...state,
    project: touchProject(state.project, segments),
    selected_segment_ids: [first.id],
    selection_anchor_id: first.id,
    run_selected_segment_ids: createdFallback && selectedAllForRun
      ? [first.id]
      : state.run_selected_segment_ids.filter((id) => !selected.has(id)),
  });
}

function assetIdentity(asset: AssetReference | null): string {
  return asset?.id ?? "";
}

function slottedIdentity(assets: SlottedAssetReference[]): string {
  return assets
    .map((asset) => `${asset.slot}:${asset.id}`)
    .sort()
    .join("|");
}

function mergeCompatible(
  selected: TimelineSegment[],
  fps: number,
): boolean {
  if (!selected.length) return false;
  const mode = selected[0].mode;
  const duration = selected.reduce((total, segment) => total + segment.duration_seconds, 0);
  if (
    selected.some((segment) => segment.mode !== mode) ||
    !Number.isFinite(duration) ||
    duration <= 0 ||
    duration > 120 ||
    alignH3Frames(duration, fps) > H3_MAX_SHOT_FRAMES
  ) return false;
  const first = selected[0];
  if (mode === "fl2va") {
    if (first.mode !== "fl2va" || first.last_image) return false;
    return selected.every((segment) =>
      segment.mode === "fl2va" &&
      !segment.last_image &&
      assetIdentity(segment.first_image) === assetIdentity(first.first_image));
  }
  if (first.mode !== "ref2va") return false;
  const pictures = slottedIdentity(first.reference_images);
  const audios = slottedIdentity(first.reference_audios);
  const videos = slottedIdentity(first.reference_videos);
  for (let index = 0; index < selected.length; index += 1) {
    const segment = selected[index];
    if (
      segment.mode !== "ref2va" ||
      assetIdentity(segment.source_video) !== assetIdentity(first.source_video) ||
      segment.source_audio_as_reference !== first.source_audio_as_reference ||
      slottedIdentity(segment.reference_images) !== pictures ||
      slottedIdentity(segment.reference_audios) !== audios ||
      slottedIdentity(segment.reference_videos) !== videos
    ) return false;
    if (index > 0 && first.source_video) {
      const previous = selected[index - 1];
      if (
        previous.mode !== "ref2va" ||
        Math.abs(previous.source_start_seconds + previous.source_duration_seconds - segment.source_start_seconds) > 1e-6
      ) return false;
    }
  }
  return true;
}

function selectedContiguousSegments(state: TimelineEditorState): TimelineSegment[] | null {
  const indexes = state.selected_segment_ids
    .map((id) => timelineIndexById(state.project, id))
    .filter((index) => index >= 0)
    .sort((a, b) => a - b);
  if (indexes.length < 2) return null;
  if (indexes.some((index, offset) => offset > 0 && index !== indexes[offset - 1] + 1))
    return null;
  const segments = indexes.map((index) => state.project.segments[index]);
  return mergeCompatible(segments, state.project.render.fps) ? segments : null;
}

export function canMergeSelectedSegments(state: TimelineEditorState): boolean {
  return selectedContiguousSegments(state) !== null;
}

export function mergeSelectedSegments(state: TimelineEditorState): TimelineEditorState {
  const selected = selectedContiguousSegments(state);
  if (!selected) return state;
  const selectedIds = new Set(selected.map((segment) => segment.id));
  const selectedForRun = selected.some((segment) => state.run_selected_segment_ids.includes(segment.id));
  const first = selected[0];
  const merged = {
    ...first,
    title: `${first.title} · 合并`,
    prompt: selected.map((segment) => segment.prompt.trim()).filter(Boolean).join("\n\n"),
    duration_seconds: selected.reduce(
      (total, segment) => total + Math.max(0, segment.duration_seconds),
      0,
    ),
  } as TimelineSegment;
  if (merged.mode === "ref2va" && merged.source_video) {
    merged.source_duration_seconds = selected.reduce(
      (total, segment) =>
        total + (segment.mode === "ref2va"
          ? Math.max(0, segment.source_duration_seconds)
          : 0),
      0,
    );
  }
  const segments = state.project.segments.flatMap((segment) => {
    if (segment.id === first.id) return [merged];
    return selectedIds.has(segment.id) ? [] : [segment];
  });
  return clampTimelinePlayhead({
    ...state,
    project: touchProject(state.project, segments),
    selected_segment_ids: [merged.id],
    selection_anchor_id: merged.id,
    run_selected_segment_ids: segments
      .map((segment) => segment.id)
      .filter((id) => id === merged.id
        ? selectedForRun
        : state.run_selected_segment_ids.includes(id)),
  });
}

export function canSplitSelectedSegment(state: TimelineEditorState): boolean {
  if (state.project.segments.length >= 128 || state.selected_segment_ids.length !== 1) return false;
  const id = state.selected_segment_ids[0];
  const segment = state.project.segments.find((candidate) => candidate.id === id);
  if (!segment?.enabled) return false;
  const position = timelineSegmentAt(state.project, state.playhead_seconds);
  if (!position || position.segment.id !== id) return false;
  const effectiveDuration = timelineSegmentPlaybackDuration(segment, state.project.render.fps);
  if (effectiveDuration <= 0) return false;
  const ratio = position.local_seconds / effectiveDuration;
  const leftRequest = segment.duration_seconds * ratio;
  const rightRequest = segment.duration_seconds - leftRequest;
  const minimumRequest = deriveSegmentRecipe(segment) === "fl2v" ? 0.1 : Number.EPSILON;
  return leftRequest >= minimumRequest && rightRequest >= minimumRequest;
}

export function splitSelectedSegment(state: TimelineEditorState): TimelineEditorState {
  if (!canSplitSelectedSegment(state)) return state;
  const id = state.selected_segment_ids[0];
  const index = timelineIndexById(state.project, id);
  const source = state.project.segments[index];
  const position = timelineSegmentAt(state.project, state.playhead_seconds);
  if (!position || position.segment.id !== id) return state;
  const effectiveDuration = timelineSegmentPlaybackDuration(source, state.project.render.fps);
  const ratio = effectiveDuration > 0
    ? Math.min(1, Math.max(0, position.local_seconds / effectiveDuration))
    : 0;
  const requestOffset = source.duration_seconds * ratio;
  const right = structuredClone(source) as TimelineSegment;
  right.id = createSegmentId();
  right.title = `${source.title} · 后段`;
  right.duration_seconds = source.duration_seconds - requestOffset;
  const left = { ...source, duration_seconds: requestOffset } as TimelineSegment;
  if (source.mode === "ref2va" && source.source_video && left.mode === "ref2va" && right.mode === "ref2va") {
    const sourceOffset = source.source_duration_seconds * ratio;
    left.source_duration_seconds = sourceOffset;
    right.source_start_seconds = source.source_start_seconds + sourceOffset;
    right.source_duration_seconds = source.source_duration_seconds - sourceOffset;
  }
  const segments = [...state.project.segments];
  segments.splice(index, 1, left, right);
  return clampTimelinePlayhead({
    ...state,
    project: touchProject(state.project, segments),
    selected_segment_ids: [right.id],
    selection_anchor_id: right.id,
    run_selected_segment_ids: state.run_selected_segment_ids.includes(source.id)
      ? state.run_selected_segment_ids.flatMap((id) => id === source.id ? [left.id, right.id] : [id])
      : state.run_selected_segment_ids,
  });
}

export interface SourceCutExpectation {
  asset_id: string;
  source_start_seconds: number;
  source_duration_seconds: number;
  project_fps: number;
}

/**
 * Applies full-source cut frames to one Ref2VA source-video segment in a single reducer
 * transition. Every child inherits the same strict mode union and maps a
 * contiguous source subrange onto the corresponding requested output ratio.
 */
export function splitTimelineSourceSegmentAtCuts(
  state: TimelineEditorState,
  segmentId: string,
  cutFrames: readonly number[],
  frameRate: number,
  expected?: SourceCutExpectation,
): TimelineEditorState {
  const index = timelineIndexById(state.project, segmentId);
  const source = state.project.segments[index];
  if (
    index < 0 ||
    source.mode !== "ref2va" ||
    !source.source_video ||
    !Number.isFinite(frameRate) ||
    frameRate <= 0 ||
    source.source_duration_seconds <= 0 ||
    source.duration_seconds <= 0
  ) return state;
  if (expected && (
    source.source_video.id !== expected.asset_id ||
    source.source_start_seconds !== expected.source_start_seconds ||
    source.source_duration_seconds !== expected.source_duration_seconds ||
    state.project.render.fps !== expected.project_fps
  )) return state;
  const sourceStart = source.source_start_seconds;
  const sourceEnd = sourceStart + source.source_duration_seconds;
  const cuts = [...new Set(cutFrames)]
    .filter((frame) => Number.isInteger(frame) && frame >= 0)
    .map((frame) => frame / frameRate)
    .filter((seconds) => seconds > sourceStart + 0.001 && seconds < sourceEnd - 0.001)
    .sort((left, right) => left - right)
    .slice(0, Math.max(0, 128 - state.project.segments.length));
  if (!cuts.length) return state;
  const boundaries = [sourceStart, ...cuts, sourceEnd];
  const replacements = boundaries.slice(0, -1).map((start, pieceIndex) => {
    const end = boundaries[pieceIndex + 1];
    const sourceDuration = end - start;
    const piece = structuredClone(source) as SourceVideoSegment;
    piece.id = pieceIndex === 0 ? source.id : createSegmentId();
    piece.title = pieceIndex === 0 ? source.title : `${source.title} · 分镜 ${pieceIndex + 1}`;
    piece.source_start_seconds = start;
    piece.source_duration_seconds = sourceDuration;
    piece.duration_seconds = source.duration_seconds * sourceDuration / source.source_duration_seconds;
    return piece;
  });
  const segments = [...state.project.segments];
  segments.splice(index, 1, ...replacements);
  return clampTimelinePlayhead({
    ...state,
    project: touchProject(state.project, segments),
    selected_segment_ids: replacements.map((segment) => segment.id),
    selection_anchor_id: replacements[0].id,
    run_selected_segment_ids: state.run_selected_segment_ids.includes(source.id)
      ? state.run_selected_segment_ids.flatMap((id) =>
          id === source.id ? replacements.map((segment) => segment.id) : [id])
      : state.run_selected_segment_ids,
  });
}

/**
 * Evenly divides one source-backed Ref2VA segment on exact proxy-frame
 * boundaries. This deliberately does not duplicate FL2VA anchors: the tool is
 * the deterministic companion to source-video shot detection.
 */
export function splitTimelineSourceSegmentEvenly(
  state: TimelineEditorState,
  segmentId: string,
  pieces: number,
): TimelineEditorState {
  const source = state.project.segments.find((segment) => segment.id === segmentId);
  const count = Math.trunc(pieces);
  const fps = state.project.render.fps;
  if (
    source?.mode !== "ref2va" ||
    !source.source_video ||
    !Number.isInteger(count) ||
    count < 2 ||
    count - 1 > 128 - state.project.segments.length ||
    !Number.isFinite(fps) ||
    fps <= 0
  ) return state;
  const startFrame = roundPositiveHalfEven(source.source_start_seconds * fps);
  const endFrame = roundPositiveHalfEven(
    (source.source_start_seconds + source.source_duration_seconds) * fps,
  );
  const totalFrames = endFrame - startFrame;
  // The stock H3 reference node rejects reference videos shorter than 5
  // frames, so every generated source range must retain that minimum.
  if (totalFrames < count * 5) return state;
  // Keep every boundary integral and distribute any remainder from the first
  // piece onward. This mirrors the visual order instead of hiding all spare
  // frames in the last piece.
  const baseFrames = Math.floor(totalFrames / count);
  const remainderFrames = totalFrames % count;
  let cursorFrame = startFrame;
  const cuts = Array.from({ length: count - 1 }, (_, index) => {
    cursorFrame += baseFrames + (index < remainderFrames ? 1 : 0);
    return cursorFrame;
  });
  return splitTimelineSourceSegmentAtCuts(state, segmentId, cuts, fps, {
    asset_id: source.source_video.id,
    source_start_seconds: source.source_start_seconds,
    source_duration_seconds: source.source_duration_seconds,
    project_fps: fps,
  });
}

export function duplicateSelectedSegments(state: TimelineEditorState): TimelineEditorState {
  if (!state.selected_segment_ids.length || state.project.segments.length >= 128) return state;
  const selected = new Set(state.selected_segment_ids);
  const capacity = 128 - state.project.segments.length;
  const sources = state.project.segments
    .filter((segment) => selected.has(segment.id))
    .slice(0, capacity);
  const copies = sources.map((segment) => ({
    ...structuredClone(segment),
    id: createSegmentId(),
    title: `${segment.title} · 副本`,
  } as TimelineSegment));
  if (!copies.length) return state;
  const lastSelectedIndex = Math.max(...state.selected_segment_ids.map((id) => timelineIndexById(state.project, id)));
  const segments = [...state.project.segments];
  segments.splice(lastSelectedIndex + 1, 0, ...copies);
  return {
    ...state,
    project: touchProject(state.project, segments),
    selected_segment_ids: copies.map((segment) => segment.id),
    selection_anchor_id: copies[0].id,
    // A duplicate inherits the source segment's execution checkbox. This is
    // independent from whether the rest of the timeline happens to be fully
    // selected for execution.
    run_selected_segment_ids: [
      ...state.run_selected_segment_ids,
      ...copies
        .filter((_, index) => state.run_selected_segment_ids.includes(sources[index].id))
        .map((segment) => segment.id),
    ],
  };
}

/**
 * Copies exactly one strict union variant while preserving target identity and
 * participation state. This deliberately cannot retain fields from the
 * target's previous mode.
 */
export function copyTimelineSegmentConfiguration(
  source: TimelineSegment,
  target: TimelineSegment,
): TimelineSegment {
  return {
    ...structuredClone(source),
    id: target.id,
    title: target.title,
    enabled: target.enabled,
  } as TimelineSegment;
}

export function applyTimelineSegmentConfiguration(
  state: TimelineEditorState,
  sourceId: string,
  scope: "following" | "all-other",
): TimelineEditorState {
  const sourceIndex = timelineIndexById(state.project, sourceId);
  if (sourceIndex < 0) return state;
  const source = state.project.segments[sourceIndex];
  const segments = state.project.segments.map((target, index) => {
    const included = scope === "following"
      ? index > sourceIndex
      : target.id !== sourceId;
    return included ? copyTimelineSegmentConfiguration(source, target) : target;
  });
  return clampTimelinePlayhead({ ...state, project: touchProject(state.project, segments) });
}

export function updateTimelineSegment(
  state: TimelineEditorState,
  id: string,
  updater: (segment: TimelineSegment) => TimelineSegment,
): TimelineEditorState {
  let changed = false;
  const segments = state.project.segments.map((segment) => {
    if (segment.id !== id) return segment;
    const next = updater(segment);
    changed ||= next !== segment;
    return next;
  });
  return changed
    ? clampTimelinePlayhead({ ...state, project: touchProject(state.project, segments) })
    : state;
}

export function addWorkspaceAssets(
  state: TimelineEditorState,
  assets: AssetReference[],
): TimelineEditorState {
  const known = new Set(state.assets.map((asset) => asset.id));
  const additions = assets.filter((asset) => !known.has(asset.id));
  return additions.length ? { ...state, assets: [...state.assets, ...additions] } : state;
}

export function moveWorkspaceAsset(
  state: TimelineEditorState,
  draggedId: string,
  targetId: string,
): TimelineEditorState {
  const from = state.assets.findIndex((asset) => asset.id === draggedId);
  const to = state.assets.findIndex((asset) => asset.id === targetId);
  if (from < 0 || to < 0 || from === to) return state;
  const assets = [...state.assets];
  const [dragged] = assets.splice(from, 1);
  assets.splice(to, 0, dragged);
  return { ...state, assets };
}

export function removeWorkspaceAssets(
  state: TimelineEditorState,
  assetIds: Iterable<string>,
): TimelineEditorState {
  const removed = new Set(assetIds);
  if (!removed.size) return state;
  const assets = state.assets.filter((asset) => !removed.has(asset.id));
  return {
    ...state,
    assets,
    selected_asset_ids: state.selected_asset_ids.filter((id) => !removed.has(id)),
  };
}

export interface TimelineAssetUsage {
  segment_id: string;
  segment_title: string;
  segment_index: number;
  role: string;
}

export function timelineAssetUsages(
  project: TimelineProject,
  assetId: string,
): TimelineAssetUsage[] {
  const usages: TimelineAssetUsage[] = [];
  const add = (segment: TimelineSegment, segmentIndex: number, role: string) => {
    usages.push({
      segment_id: segment.id,
      segment_title: segment.title,
      segment_index: segmentIndex,
      role,
    });
  };
  project.segments.forEach((segment, index) => {
    if (segment.mode === "fl2va") {
      if (segment.first_image?.id === assetId) add(segment, index, "首帧");
      if (segment.last_image?.id === assetId) add(segment, index, "尾帧");
    }
    if (segment.mode === "ref2va") {
      segment.reference_images.filter((asset) => asset.id === assetId).forEach((asset) => add(segment, index, `<Picture ${asset.slot + 1}>`));
      const audioOffset = segment.source_audio_as_reference ? 1 : 0;
      const videoOffset = segment.source_video ? 1 : 0;
      segment.reference_audios.filter((asset) => asset.id === assetId).forEach((asset) => add(segment, index, `<Audio ${asset.slot + 1 + audioOffset}>`));
      segment.reference_videos.filter((asset) => asset.id === assetId).forEach((asset) => add(segment, index, `<Video ${asset.slot + 1 + videoOffset}>`));
      if (segment.source_video?.id === assetId)
        add(
          segment,
          index,
          segment.source_audio_as_reference
            ? "<Video 1> 主视频 / <Audio 1> 源音轨"
            : "<Video 1> 主视频",
        );
    }
  });
  return usages;
}

export type TimelineAction =
  | { type: "project/replace"; project: TimelineProject }
  | { type: "project/patch"; patch: Partial<Pick<TimelineProject, "title" | "render" | "sampling" | "export_mode">> }
  | { type: "segment/select"; id: string; additive?: boolean; range?: boolean }
  | { type: "segment/set-selection"; ids: string[] }
  | { type: "segment/set-run-selection"; ids: string[] }
  | { type: "segment/set-enabled"; ids: string[]; enabled: boolean }
  | { type: "segment/insert"; position: "before" | "after"; mode?: TimelineGenerationMode }
  | { type: "segment/insert-video"; position?: "before" | "after"; anchorId?: string | null; asset: AssetReference }
  | { type: "segment/move"; draggedId: string; targetId: string }
  | { type: "segment/delete-selected" }
  | { type: "segment/merge-selected" }
  | { type: "segment/split-selected" }
  | { type: "segment/apply-source-cuts"; id: string; cutFrames: number[]; frameRate: number; expected: SourceCutExpectation }
  | { type: "segment/split-evenly"; id: string; pieces: number }
  | { type: "segment/duplicate-selected" }
  | { type: "segment/replace"; segment: TimelineSegment }
  | { type: "segment/bind-asset"; id: string; asset: AssetReference; target?: SegmentAssetTarget; select?: boolean }
  | { type: "segment/bind-assets"; id: string; assets: AssetReference[]; target?: SegmentAssetTarget; select?: boolean }
  | { type: "segment/reorder-reference"; id: string; kind: SegmentReferenceKind; draggedAssetId: string; targetAssetId: string }
  | { type: "segment/set-mode"; ids: string[]; mode: TimelineGenerationMode }
  | { type: "segment/insert-reference-token"; id: string; token: string; selectionStart: number; selectionEnd: number; expectedMention: string }
  | { type: "segment/insert-subject-token"; id: string; token: string; selectionStart: number; selectionEnd: number; expectedMention: string }
  | { type: "segment/apply-config"; sourceId: string; scope: "following" | "all-other" }
  | { type: "assets/add"; assets: AssetReference[] }
  | { type: "assets/replace"; assets: AssetReference[] }
  | { type: "assets/move"; draggedId: string; targetId: string }
  | { type: "assets/select"; id: string; additive?: boolean }
  | { type: "assets/set-selection"; ids: string[]; additive?: boolean }
  | { type: "assets/clear-selection" }
  | { type: "assets/remove"; ids: string[] }
  | { type: "assets/grid-size"; size: AssetGridSize }
  | { type: "playhead/set"; seconds: number };

export function timelineEditorReducer(
  state: TimelineEditorState,
  action: TimelineAction,
): TimelineEditorState {
  switch (action.type) {
    case "project/replace": {
      const first = action.project.segments[0];
      const ids = new Set(action.project.segments.map((segment) => segment.id));
      const selected = state.selected_segment_ids.filter((id) => ids.has(id));
      const selectedAllForRun = runSelectionCoversAllEnabled(state);
      const sharesSegmentIdentity = state.project.segments.some((segment) => ids.has(segment.id));
      const runSelected = selectedAllForRun || !sharesSegmentIdentity
        ? enabledTimelineSegmentIds(action.project)
        : action.project.segments
            .filter((segment) => segment.enabled && state.run_selected_segment_ids.includes(segment.id))
            .map((segment) => segment.id);
      return clampTimelinePlayhead({
        ...state,
        project: action.project,
        selected_segment_ids: selected.length ? selected : first ? [first.id] : [],
        selection_anchor_id: state.selection_anchor_id && ids.has(state.selection_anchor_id)
          ? state.selection_anchor_id
          : selected[0] ?? first?.id ?? null,
        run_selected_segment_ids: runSelected,
      });
    }
    case "project/patch":
      return clampTimelinePlayhead({
        ...state,
        project: {
          ...state.project,
          ...action.patch,
        },
      });
    case "segment/select": {
      const target = state.project.segments.find((segment) => segment.id === action.id);
      if (!target) return state;
      // Disabled clips remain editor-selectable, but cannot participate in a
      // run. Selecting one from the disabled rail must not erase the current
      // runnable set.
      if (!target.enabled) {
        const selected = selectTimelineSegment(state, action.id, action);
        return selected.selected_asset_ids.length
          ? { ...selected, selected_asset_ids: [] }
          : selected;
      }
      // Direct gestures operate on the prominently displayed run set. This
      // also reconciles a temporary edit/run divergence caused by header or
      // structural commands (for example: edit focus A, run selection B,
      // Ctrl-click A should add A to B rather than accidentally clear B).
      const selected = selectTimelineSegment({
        ...state,
        selected_segment_ids: state.project.segments
          .filter((segment) => segment.enabled && state.run_selected_segment_ids.includes(segment.id))
          .map((segment) => segment.id),
      }, action.id, action);
      const next = {
        ...selected,
        run_selected_segment_ids: selected.project.segments
          .filter((segment) =>
            segment.enabled && selected.selected_segment_ids.includes(segment.id))
          .map((segment) => segment.id),
      };
      return next.selected_asset_ids.length
        ? { ...next, selected_asset_ids: [] }
        : next;
    }
    case "segment/set-selection": {
      const requested = new Set(action.ids);
      const selected = state.project.segments
        .map((segment) => segment.id)
        .filter((id) => requested.has(id));
      return {
        ...state,
        selected_segment_ids: selected,
        selection_anchor_id: selected.at(-1) ?? null,
        selected_asset_ids: selected.length ? [] : state.selected_asset_ids,
      };
    }
    case "segment/set-run-selection": {
      const requested = new Set(action.ids);
      return {
        ...state,
        run_selected_segment_ids: state.project.segments
          .filter((segment) => segment.enabled && requested.has(segment.id))
          .map((segment) => segment.id),
      };
    }
    case "segment/set-enabled": {
      const ids = new Set(action.ids);
      const selectedAllForRun = runSelectionCoversAllEnabled(state);
      const project = touchProject(
        state.project,
        state.project.segments.map((segment) =>
          ids.has(segment.id) ? { ...segment, enabled: action.enabled } : segment),
      );
      return clampTimelinePlayhead({
        ...state,
        project,
        run_selected_segment_ids: action.enabled && selectedAllForRun
          ? enabledTimelineSegmentIds(project)
          : state.run_selected_segment_ids.filter((id) =>
              !ids.has(id) && project.segments.some((segment) => segment.enabled && segment.id === id)),
      });
    }
    case "segment/insert":
      return insertTimelineSegment(state, action.position, action.mode);
    case "segment/insert-video":
      return insertTimelineVideoAssetAtAnchor(
        state,
        action.asset,
        action.anchorId ?? state.selected_segment_ids.at(-1) ?? null,
        action.position,
      );
    case "segment/move":
      return moveTimelineSegment(state, action.draggedId, action.targetId);
    case "segment/delete-selected":
      return deleteSelectedSegments(state);
    case "segment/merge-selected":
      return mergeSelectedSegments(state);
    case "segment/split-selected":
      return splitSelectedSegment(state);
    case "segment/apply-source-cuts":
      return splitTimelineSourceSegmentAtCuts(
        state,
        action.id,
        action.cutFrames,
        action.frameRate,
        action.expected,
      );
    case "segment/split-evenly":
      return splitTimelineSourceSegmentEvenly(state, action.id, action.pieces);
    case "segment/duplicate-selected":
      return duplicateSelectedSegments(state);
    case "segment/replace":
      return updateTimelineSegment(state, action.segment.id, () => action.segment);
    case "segment/bind-asset": {
      const updated = updateTimelineSegment(state, action.id, (segment) =>
        assignAssetToSegment(segment, action.asset, action.target));
      if (!action.select || updated === state) return updated;
      return {
        ...updated,
        selected_segment_ids: [action.id],
        selection_anchor_id: action.id,
        selected_asset_ids: [],
      };
    }
    case "segment/bind-assets": {
      // Upload and binding become one reducer transition: every successful
      // upload enters the library even when a zone rejects it for type/capacity.
      let updated = addWorkspaceAssets(state, action.assets);
      let bound = false;
      updated = updateTimelineSegment(updated, action.id, (segment) => {
        const result = assignAssetsToSegment(segment, action.assets, action.target);
        bound = result.accepted.length > 0;
        return result.segment;
      });
      if (!action.select || !bound) return updated;
      return {
        ...updated,
        selected_segment_ids: [action.id],
        selection_anchor_id: action.id,
        selected_asset_ids: [],
      };
    }
    case "segment/reorder-reference":
      return updateTimelineSegment(state, action.id, (segment) =>
        reorderSegmentReference(
          segment,
          action.kind,
          action.draggedAssetId,
          action.targetAssetId,
        ));
    case "segment/set-mode": {
      const ids = new Set(action.ids);
      return {
        ...state,
        project: touchProject(
          state.project,
          state.project.segments.map((segment) =>
            ids.has(segment.id) ? changeSegmentMode(segment, action.mode) : segment,
          ),
        ),
      };
    }
    case "segment/insert-reference-token":
      return updateTimelineSegment(state, action.id, (segment) =>
        insertPromptReferenceToken(
          segment,
          action.token,
          action.selectionStart,
          action.selectionEnd,
          action.expectedMention,
        ));
    case "segment/insert-subject-token":
      return updateTimelineSegment(state, action.id, (segment) =>
        insertPromptSubjectToken(
          segment,
          action.token,
          action.selectionStart,
          action.selectionEnd,
          action.expectedMention,
        ));
    case "segment/apply-config":
      return applyTimelineSegmentConfiguration(state, action.sourceId, action.scope);
    case "assets/add":
      return addWorkspaceAssets(state, action.assets);
    case "assets/replace":
      return {
        ...state,
        assets: action.assets,
        selected_asset_ids: state.selected_asset_ids.filter((id) =>
          action.assets.some((asset) => asset.id === id)),
      };
    case "assets/move":
      return moveWorkspaceAsset(state, action.draggedId, action.targetId);
    case "assets/select": {
      const selected = action.additive
        ? state.selected_asset_ids.includes(action.id)
          ? state.selected_asset_ids.filter((id) => id !== action.id)
          : [...state.selected_asset_ids, action.id]
        : [action.id];
      return { ...state, selected_asset_ids: selected };
    }
    case "assets/set-selection": {
      const requested = new Set(action.ids);
      if (action.additive) {
        for (const id of state.selected_asset_ids) requested.add(id);
      }
      const selected = state.assets
        .filter((asset) => requested.has(asset.id))
        .map((asset) => asset.id);
      return { ...state, selected_asset_ids: selected };
    }
    case "assets/clear-selection":
      return state.selected_asset_ids.length
        ? { ...state, selected_asset_ids: [] }
        : state;
    case "assets/remove":
      return removeWorkspaceAssets(state, action.ids);
    case "assets/grid-size":
      return { ...state, asset_grid_size: action.size };
    case "playhead/set":
      return {
        ...state,
        playhead_seconds: Number.isFinite(action.seconds)
          ? Math.min(timelineDuration(state.project), Math.max(0, action.seconds))
          : state.playhead_seconds,
      };
  }
}

export function timelineDuration(project: TimelineProject): number {
  return project.segments.reduce(
    (total, segment) => total + effectiveTimelineSegmentDuration(segment, project.render.fps),
    0,
  );
}

/** Resolves the exact ephemeral run set without conflating it with edit focus. */
export function runnableTimelineSegmentIds(state: TimelineEditorState): string[] {
  const selected = new Set(state.run_selected_segment_ids);
  return state.project.segments
    .filter((segment) => segment.enabled && selected.has(segment.id))
    .map((segment) => segment.id);
}

/** Resolve AddGuide seams from the complete enabled timeline, never from the run subset. */
export function timelineContinuityBoundaries(
  project: TimelineProject,
): TimelineContinuityBoundary[] {
  const enabled = project.segments.flatMap((segment, index) =>
    segment.enabled ? [{ segment, index }] : [],
  );
  return enabled.slice(1).map(({ segment, index }, activeIndex) => {
    const predecessor = enabled[activeIndex];
    let kind: TimelineContinuityBoundaryKind = "eligible";
    // An authored FL first frame deliberately starts a fresh shot. It wins over
    // the automatic AddGuide seam because both would occupy output frame zero.
    if (segment.mode === "fl2va" && segment.first_image !== null) {
      kind = "explicit-first-image";
    }
    return {
      predecessor: predecessor.segment,
      predecessor_index: predecessor.index,
      segment,
      segment_index: index,
      kind,
    };
  });
}

/**
 * Validate the exact explicit run set for each target segment's incoming seam.
 * Eligible targets whose direct predecessor is outside the explicit run set
 * carry a historical-take intent. The browser never expands the user's stable
 * ID selection or chooses a take; the server resolves that dependency.
 */
export function timelineContinuityRunIssues(
  project: TimelineProject,
  segmentIds?: readonly string[],
): TimelineContinuityRunIssue[] {
  const selected = segmentIds === undefined
    ? new Set(project.segments.filter((segment) => segment.enabled).map((segment) => segment.id))
    : new Set(segmentIds);
  const quote = (segment: TimelineSegment, index: number) =>
    `第 ${index + 1} 段“${segment.title || segment.id}”`;
  const issues: TimelineContinuityRunIssue[] = [];
  for (const boundary of timelineContinuityBoundaries(project)) {
    if (!boundary.segment.continuity.enabled || !selected.has(boundary.segment.id)) continue;
    const target = quote(boundary.segment, boundary.segment_index);
    const predecessor = quote(boundary.predecessor, boundary.predecessor_index);
    const overlapFrames = boundary.segment.continuity.overlap_frames;
    if (boundary.kind === "eligible" && !selected.has(boundary.predecessor.id)) {
      issues.push({
        code: "historical-take-required",
        boundary,
        message: `${target}将请求服务端按稳定分段 ID、分辨率、帧率和可见帧数，复用直接前驱${predecessor}在当前 ComfyUI 上最新生成的成功成片，并读取最后 ${overlapFrames} 帧接续；提示词、生成方式、参考素材与推理参数不参与匹配`,
      });
    }
    if (boundary.kind === "eligible") {
      const predecessorFrames = alignH3Frames(
        boundary.predecessor.duration_seconds,
        project.render.fps,
      );
      if (predecessorFrames < overlapFrames) {
        issues.push({
          code: "predecessor-too-short",
          boundary,
          message: `${predecessor}只有 ${predecessorFrames} 个可见帧，少于段间接续需要的 ${overlapFrames} 帧；请延长前段或降低接续尾帧数`,
        });
      }
      const visibleFrames = alignH3Frames(
        boundary.segment.duration_seconds,
        project.render.fps,
      );
      const sampleFrames = alignH3FrameCount(visibleFrames + overlapFrames);
      if (sampleFrames > H3_MAX_SHOT_FRAMES) {
        issues.push({
          code: "sample-too-long",
          boundary,
          message: `${target}加入前段 ${overlapFrames} 帧接续上下文后需内部采样 ${sampleFrames} 帧（可见 ${visibleFrames} 帧），超过 MiniMax H3 的 ${H3_MAX_SHOT_FRAMES} 帧上限；请缩短后段或降低接续尾帧数`,
        });
      }
    }
  }
  return issues;
}

export function clampTimelinePlayhead(state: TimelineEditorState): TimelineEditorState {
  const clamped = Math.min(
    timelineDuration(state.project),
    Math.max(0, Number.isFinite(state.playhead_seconds) ? state.playhead_seconds : 0),
  );
  return clamped === state.playhead_seconds ? state : { ...state, playhead_seconds: clamped };
}

export interface TimelinePosition {
  segment: TimelineSegment;
  index: number;
  start_seconds: number;
  local_seconds: number;
}

/** Resolves an absolute project playhead to a stable segment identity. */
export function timelineSegmentAt(
  project: TimelineProject,
  playheadSeconds: number,
): TimelinePosition | null {
  const enabled = project.segments
    .map((segment, index) => ({ segment, index }))
    .filter(({ segment }) => segment.enabled);
  if (!enabled.length) return null;
  const total = timelineDuration(project);
  const playhead = Math.min(
    Math.max(0, Number.isFinite(playheadSeconds) ? playheadSeconds : 0),
    total,
  );
  let start = 0;
  for (const [enabledIndex, { segment, index }] of enabled.entries()) {
    const duration = effectiveTimelineSegmentDuration(segment, project.render.fps);
    const end = start + duration;
    if (playhead < end || enabledIndex === enabled.length - 1) {
      return {
        segment,
        index,
        start_seconds: start,
        local_seconds: Math.min(duration, Math.max(0, playhead - start)),
      };
    }
    start = end;
  }
  return null;
}

/** Maps project-local output time onto the selected Ref2VA source range. */
export function sourcePreviewTime(
  segment: SourceVideoSegment,
  localSeconds: number,
  fps: number,
): number {
  const outputDuration = timelineSegmentPlaybackDuration(segment, fps);
  const ratio = outputDuration > 0
    ? Math.min(1, Math.max(0, localSeconds / outputDuration))
    : 0;
  return segment.source_start_seconds + segment.source_duration_seconds * ratio;
}

/**
 * Representative source-frame positions for the visual timeline filmstrip.
 * A complete source receives several evenly spaced frames; after it is split,
 * one midpoint per generated segment is enough for the adjacent clips to form
 * the strip without multiplying media decoders across a 128-segment project.
 */
export function sourceTimelineThumbnailTimes(
  segment: SourceVideoSegment,
): number[] {
  const metadataDuration = segment.source_video?.metadata?.duration;
  if (
    !segment.source_video?.preview_url ||
    !Number.isFinite(metadataDuration) ||
    !Number.isFinite(segment.source_start_seconds) ||
    !Number.isFinite(segment.source_duration_seconds) ||
    segment.source_duration_seconds <= 0
  ) return [];
  const sourceStart = Math.max(0, segment.source_start_seconds);
  const sourceDuration = Math.min(
    segment.source_duration_seconds,
    Math.max(0, (metadataDuration as number) - sourceStart),
  );
  if (sourceDuration <= 0) return [];
  // Every source-backed clip shows evenly spaced keyframes across its own
  // source range, whether it spans the whole source or a split/cropped slice.
  // ~5s per keyframe keeps long clips scannable without a canvas per second.
  const count = Math.min(8, Math.max(1, Math.ceil(sourceDuration / 5)));
  return Array.from(
    { length: count },
    (_, index) => sourceStart + sourceDuration * (index + 0.5) / count,
  );
}

export function segmentStartTime(project: TimelineProject, segmentId: string): number {
  let cursor = 0;
  for (const segment of project.segments) {
    if (segment.id === segmentId) return cursor;
    cursor += effectiveTimelineSegmentDuration(segment, project.render.fps);
  }
  return cursor;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeTimelineSampling(value: unknown): SamplingConfig | null {
  if (!isRecord(value)) return null;
  if (![value.steps, value.seed, value.shift, value.audio_shift].every(finiteNumber)) return null;
  const sampler = ["res_multistep", "euler", "dpmpp_2m"].includes(String(value.sampler))
    ? value.sampler as SamplingConfig["sampler"]
    : "res_multistep";
  const scheduler = isSamplingScheduler(value.scheduler)
    ? value.scheduler as SamplingConfig["scheduler"]
    : "simple";
  const legacyRandom = value.seed === -1;
  return {
    steps: value.steps as number,
    seed: legacyRandom ? randomSafeSeed() : value.seed as number,
    random_seed: value.random_seed === true || legacyRandom,
    sampler,
    scheduler,
    shift: value.shift as number,
    audio_shift: value.audio_shift as number,
  };
}

function normalizeSegmentContinuity(
  value: unknown,
  fallback: TimelineSegmentContinuity = { enabled: false, overlap_frames: 22 },
): TimelineSegmentContinuity {
  if (!isRecord(value)) return { ...fallback };
  const overlapFrames = [5, 22, 39, 56].includes(Number(value.overlap_frames))
    ? value.overlap_frames as TimelineSegmentContinuity["overlap_frames"]
    : fallback.overlap_frames;
  return {
    enabled: value.enabled === true,
    overlap_frames: overlapFrames,
  };
}

function normalizeSegmentBase(
  value: unknown,
  legacyContinuity?: TimelineSegmentContinuity,
  legacySettings?: Pick<TimelineSegmentBase<TimelineGenerationMode>, "ref_image_size" | "audio_mode">,
): Omit<TimelineSegmentBase<TimelineGenerationMode>, "mode"> | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.id !== "string" ||
    !value.id ||
    value.id.length > 128 ||
    typeof value.title !== "string" ||
    value.title.length > 256 ||
    typeof value.prompt !== "string" ||
    promptCharacterCount(value.prompt) > MINIMAX_H3_PROMPT_MAX_CHARACTERS ||
    !finiteNumber(value.duration_seconds) ||
    typeof value.enabled !== "boolean"
  ) return null;
  const refImageSize = value.ref_image_size === "match" || value.ref_image_size === "max"
    ? value.ref_image_size
    : legacySettings?.ref_image_size;
  const audioMode = ["generate", "source", "mute"].includes(String(value.audio_mode))
    ? value.audio_mode as TimelineSegmentBase<TimelineGenerationMode>["audio_mode"]
    : legacySettings?.audio_mode;
  if (!refImageSize || !audioMode) return null;
  return {
    id: value.id,
    title: value.title,
    duration_seconds: value.duration_seconds,
    prompt: value.prompt,
    enabled: value.enabled,
    continuity: normalizeSegmentContinuity(value.continuity, legacyContinuity),
    ref_image_size: refImageSize,
    audio_mode: audioMode,
  };
}

function normalizeFamilySegment(
  value: unknown,
  legacyContinuity?: TimelineSegmentContinuity,
  legacySettings?: Pick<TimelineSegmentBase<TimelineGenerationMode>, "ref_image_size" | "audio_mode">,
): TimelineSegment | null {
  if (!isRecord(value)) return null;
  const base = normalizeSegmentBase(value, legacyContinuity, legacySettings);
  if (!base) return null;
  switch (value.mode) {
    case "fl2va":
      return {
        ...base,
        mode: "fl2va",
        first_image: normalizeAssetReference(value.first_image, "image"),
        last_image: normalizeAssetReference(value.last_image, "image"),
      };
    case "ref2va":
      if (!finiteNumber(value.source_start_seconds) || !finiteNumber(value.source_duration_seconds))
        return null;
      return {
        ...base,
        mode: "ref2va",
        source_video: normalizeAssetReference(value.source_video, "video"),
        source_start_seconds: value.source_start_seconds,
        source_duration_seconds: value.source_duration_seconds,
        source_audio_as_reference: value.source_audio_as_reference === true,
        reference_images: normalizeSlottedAssetList(
          value.reference_images,
          "image",
          maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages),
        ),
        reference_audios: normalizeSlottedAssetList(
          value.reference_audios,
          "audio",
          maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios),
        ),
        reference_videos: normalizeSlottedAssetList(
          value.reference_videos,
          "video",
          maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos),
        ),
      };
    default:
      return null;
  }
}

/** Converts a persisted/API v1 six-recipe segment into the current family union. */
function migrateV1Segment(
  value: unknown,
  legacyContinuity?: TimelineSegmentContinuity,
  legacySettings?: Pick<TimelineSegmentBase<TimelineGenerationMode>, "ref_image_size" | "audio_mode">,
): TimelineSegment | null {
  if (!isRecord(value)) return null;
  const base = normalizeSegmentBase(value, legacyContinuity, legacySettings);
  if (!base) return null;
  const commonKeys = ["id", "mode", "title", "prompt", "duration_seconds", "enabled"];
  const allowedKeys: Record<LegacyGenerationMode, readonly string[]> = {
    t2v: commonKeys,
    i2v: [...commonKeys, "first_image"],
    fl2v: [...commonKeys, "first_image", "last_image"],
    r2v: [...commonKeys, "reference_images", "reference_audios", "reference_videos"],
    v2v: [
      ...commonKeys,
      "source_video",
      "source_start_seconds",
      "source_duration_seconds",
      "source_audio_as_reference",
    ],
    rv2v: [
      ...commonKeys,
      "source_video",
      "source_start_seconds",
      "source_duration_seconds",
      "source_audio_as_reference",
      "reference_images",
      "reference_audios",
    ],
  };
  if (
    typeof value.mode !== "string" ||
    !(value.mode in allowedKeys) ||
    Object.keys(value).some((key) => !allowedKeys[value.mode as LegacyGenerationMode].includes(key))
  ) return null;
  switch (value.mode) {
    case "t2v":
      return { ...base, mode: "fl2va", first_image: null, last_image: null };
    case "i2v":
      return {
        ...base,
        mode: "fl2va",
        first_image: normalizeAssetReference(value.first_image, "image"),
        last_image: null,
      };
    case "fl2v":
      return {
        ...base,
        mode: "fl2va",
        first_image: normalizeAssetReference(value.first_image, "image"),
        last_image: normalizeAssetReference(value.last_image, "image"),
      };
    case "r2v":
      return {
        ...base,
        mode: "ref2va",
        source_video: null,
        source_start_seconds: 0,
        source_duration_seconds: base.duration_seconds,
        source_audio_as_reference: false,
        reference_images: normalizeSlottedAssetList(
          value.reference_images,
          "image",
          maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages),
        ),
        reference_audios: normalizeSlottedAssetList(
          value.reference_audios,
          "audio",
          maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios),
        ),
        reference_videos: normalizeSlottedAssetList(
          value.reference_videos,
          "video",
          maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos),
        ),
      };
    case "v2v":
    case "rv2v": {
      if (
        !finiteNumber(value.source_start_seconds) ||
        !finiteNumber(value.source_duration_seconds)
      ) return null;
      const source = normalizeAssetReference(value.source_video, "video");
      const sourceFields = {
        source_video: source,
        source_start_seconds: value.source_start_seconds,
        source_duration_seconds: value.source_duration_seconds,
        source_audio_as_reference: value.source_audio_as_reference === true,
      };
      return {
        ...base,
        mode: "ref2va",
        ...sourceFields,
        reference_images: value.mode === "rv2v"
          ? normalizeSlottedAssetList(
              value.reference_images,
              "image",
              maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages),
            )
          : [],
        reference_audios: value.mode === "rv2v"
          ? normalizeSlottedAssetList(
              value.reference_audios,
              "audio",
              maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios),
            )
          : [],
        reference_videos: [],
      };
    }
    default:
      return null;
  }
}

/** Rebuilds API/local data into v4, migrating shared v1-v3 segment settings. */
export function normalizeTimelineProject(value: unknown): TimelineProject | null {
  if (
    !isRecord(value) ||
    (value.version !== 1 && value.version !== 2 && value.version !== 3 && value.version !== 4) ||
    typeof value.title !== "string" ||
    value.title.length < 1 ||
    value.title.length > 256 ||
    !Array.isArray(value.segments) ||
    value.segments.length < 1 ||
    value.segments.length > 128 ||
    !isRecord(value.render) ||
    !isRecord(value.sampling)
  ) return null;
  const legacyDocument = value.version === 1;
  // Match the backend migration boundary exactly. A current document may not use
  // removed v1 fields and rely on the browser to silently reinterpret them.
  if (!legacyDocument && Object.prototype.hasOwnProperty.call(value, "prompt"))
    return null;
  if ((value.version === 3 || value.version === 4) && Object.prototype.hasOwnProperty.call(value, "continuity"))
    return null;
  if (value.version === 4 && (
    Object.prototype.hasOwnProperty.call(value, "ref_image_size") ||
    Object.prototype.hasOwnProperty.call(value, "audio_mode")
  )) return null;
  if (value.version === 2 && value.segments.some((segment) =>
    isRecord(segment) && Object.prototype.hasOwnProperty.call(segment, "continuity")))
    return null;
  if ((value.version === 2 || value.version === 3) && value.segments.some((segment) =>
    isRecord(segment) && (
      Object.prototype.hasOwnProperty.call(segment, "ref_image_size") ||
      Object.prototype.hasOwnProperty.call(segment, "audio_mode")
    ))) return null;
  const render = value.render;
  const sampling = value.sampling;
  if (![render.width, render.height, render.fps].every(finiteNumber)) return null;
  const familySampling = isRecord(sampling.fl2va) && isRecord(sampling.ref2va)
    ? {
        fl2va: normalizeTimelineSampling(sampling.fl2va),
        ref2va: normalizeTimelineSampling(sampling.ref2va),
      }
    : legacyDocument ? (() => {
        const legacy = normalizeTimelineSampling(sampling);
        return { fl2va: legacy, ref2va: legacy ? { ...legacy } : null };
      })() : { fl2va: null, ref2va: null };
  if (!familySampling.fl2va || !familySampling.ref2va) return null;
  const legacyPrompt = legacyDocument && typeof value.prompt === "string" && promptCharacterCount(value.prompt) <= MINIMAX_H3_PROMPT_MAX_CHARACTERS
    ? value.prompt
    : "";
  const legacyContinuity = value.version === 3 || value.version === 4
    ? undefined
    : normalizeSegmentContinuity(value.continuity);
  const legacySettings = value.version === 4
    ? undefined
    : {
        ref_image_size: value.ref_image_size === "max" ? "max" as const : "match" as const,
        audio_mode: ["generate", "source", "mute"].includes(String(value.audio_mode))
          ? value.audio_mode as TimelineSegmentBase<TimelineGenerationMode>["audio_mode"]
          : "generate" as const,
      };
  const normalizeSegment = value.version === 1
    ? (segment: unknown) => migrateV1Segment(segment, legacyContinuity, legacySettings)
    : (segment: unknown) => normalizeFamilySegment(segment, legacyContinuity, legacySettings);
  const segments = value.segments.map(normalizeSegment).map((segment) =>
    segment && !segment.prompt.trim() && legacyPrompt.trim()
      ? { ...segment, prompt: legacyPrompt }
      : segment);
  if (
    segments.some((segment) => segment === null) ||
    new Set(segments.map((segment) => segment?.id)).size !== segments.length
  ) return null;
  return {
    version: 4,
    title: value.title,
    render: {
      width: render.width as number,
      height: render.height as number,
      // Native H3 latent and reference-video semantics are fixed at 24fps.
      fps: 24,
    },
    sampling: familySampling as TimelineProject["sampling"],
    export_mode: value.export_mode === "segments" ? "segments" : "all",
    segments: segments as TimelineSegment[],
  };
}

export type SegmentAssetTarget =
  | "auto"
  | "first_image"
  | "last_image"
  | "source_video"
  | "reference"
  | "reference_image"
  | "reference_video"
  | "reference_audio";

export type SegmentReferenceKind = "image" | "video" | "audio";

/**
 * Moves a slotted Ref2VA reference and rewrites prompt labels by asset
 * identity. Slot numbers are part of the H3 prompt contract, so changing the
 * visual order without this old-label -> new-label rewrite would silently make
 * existing prompt text refer to different assets.
 */
export function reorderSegmentReference(
  segment: TimelineSegment,
  kind: SegmentReferenceKind,
  draggedAssetId: string,
  targetAssetId: string,
): TimelineSegment {
  if (segment.mode !== "ref2va" || draggedAssetId === targetAssetId) return segment;
  const values = kind === "image"
    ? segment.reference_images
    : kind === "video"
      ? segment.reference_videos
      : segment.reference_audios;
  const ordered = [...values].sort((left, right) => left.slot - right.slot);
  const from = ordered.findIndex((asset) => asset.id === draggedAssetId);
  const to = ordered.findIndex((asset) => asset.id === targetAssetId);
  if (from < 0 || to < 0) return segment;

  const [dragged] = ordered.splice(from, 1);
  ordered.splice(to, 0, dragged);
  const offset = kind === "video"
    ? (segment.source_video ? 1 : 0)
    : kind === "audio"
      ? (segment.source_audio_as_reference ? 1 : 0)
      : 0;
  const oldToNewLabel = new Map<number, number>();
  const reordered = ordered.map((asset, slot) => {
    oldToNewLabel.set(asset.slot + 1 + offset, slot + 1 + offset);
    return { ...asset, slot };
  });
  const labelKind = kind === "image" ? "Picture" : kind === "video" ? "Video" : "Audio";
  const prompt = segment.prompt.replace(
    /<\s*(Picture|Audio|Video)\s+(\d+)\s*>/gi,
    (raw, rawKind: string, rawLabel: string) => {
      if (rawKind.toLocaleLowerCase() !== labelKind.toLocaleLowerCase()) return raw;
      const mapped = oldToNewLabel.get(Number(rawLabel));
      return mapped === undefined ? raw : `<${labelKind} ${mapped}>`;
    },
  );

  if (kind === "image") return { ...segment, prompt, reference_images: reordered };
  if (kind === "video") return { ...segment, prompt, reference_videos: reordered };
  return { ...segment, prompt, reference_audios: reordered };
}

/**
 * Updates a source-backed Ref2VA range. Generated-audio segments preserve the
 * existing source-to-output playback ratio. With source audio, an explicit
 * source-duration edit instead establishes the same requested output duration;
 * the project-level H3 fitter can then make the source/output frame counts
 * equal without feeding its previous automatic adjustment back into the next
 * edit as a new playback ratio.
 */
export function updateRef2VASourceRange(
  segment: Ref2VASegment,
  range: {
    source_start_seconds: number;
    source_duration_seconds: number;
  },
): Ref2VASegment {
  const previousSourceDuration = segment.source_duration_seconds;
  const nextSourceStart = range.source_start_seconds;
  const nextSourceDuration = range.source_duration_seconds;
  if (
    !Number.isFinite(nextSourceStart) || nextSourceStart < 0 ||
    !Number.isFinite(nextSourceDuration) || nextSourceDuration <= 0
  ) {
    return segment;
  }
  if (nextSourceDuration === previousSourceDuration) {
    return { ...segment, ...range };
  }
  const ratio = Number.isFinite(previousSourceDuration) && previousSourceDuration > 0 &&
      Number.isFinite(segment.duration_seconds) && segment.duration_seconds > 0
    ? segment.duration_seconds / previousSourceDuration
    : 1;
  const nextDuration = segment.audio_mode === "source"
    ? nextSourceDuration
    : nextSourceDuration * ratio;
  if (!Number.isFinite(nextDuration) || nextDuration <= 0) return segment;
  return {
    ...segment,
    ...range,
    duration_seconds: nextDuration,
  };
}

function assignSourceVideo(segment: Ref2VASegment, asset: AssetReference): Ref2VASegment {
  if (asset.kind !== "video") return segment;
  const prepared = asset.metadata?.has_audio === true
    ? segment
    : setSourceAudioAsReference(segment, false);
  const addingFirstSource = prepared.source_video === null;
  const independentVideoLabels = new Map(
    prepared.reference_videos.map((reference) => [
      reference.slot + 1,
      reference.slot + 2,
    ]),
  );
  // The official Ref2VA node assigns the source to <Video 1>. Independent
  // reference videos that previously occupied Video 1..N must move to
  // Video 2..N+1 on the first source bind. Replacing an existing source does
  // not change the label layout.
  const prompt = addingFirstSource
    ? prepared.prompt.replace(
        /<\s*Video\s+(\d+)\s*>/gi,
        (raw, label: string) => {
          const shifted = independentVideoLabels.get(Number(label));
          return shifted === undefined ? raw : `<Video ${shifted}>`;
        },
      )
    : prepared.prompt;
  const fittedRange = fitSourceRangeToVideo(
    asset,
    prepared.source_start_seconds,
    prepared.source_duration_seconds,
  );
  const completeSourceDuration = asset.metadata?.duration;
  const untouchedFactorySegment = addingFirstSource &&
    /^片段\s+\d+$/.test(prepared.title.trim()) &&
    (!prepared.prompt.trim() || prepared.prompt === EMPTY_SIX_SECTION_PROMPT) &&
    prepared.reference_images.length === 0 &&
    prepared.reference_audios.length === 0 &&
    prepared.reference_videos.length === 0;
  const fitted = fittedRange
    ? updateRef2VASourceRange(prepared, fittedRange)
    : prepared;
  return {
    ...fitted,
    prompt,
    ...(untouchedFactorySegment
      ? { title: asset.name.replace(/\.[^.]+$/, "") || prepared.title }
      : {}),
    source_video: asset,
    // The first source bind establishes the editable source timeline and must
    // therefore expose the whole video. Replacing an existing source keeps the
    // user's current trim whenever it still fits the replacement.
    ...(addingFirstSource && completeSourceDuration
      ? {
          duration_seconds: completeSourceDuration,
          source_start_seconds: 0,
          source_duration_seconds: completeSourceDuration,
        }
      : {}),
  };
}

function rewriteFl2vaPictureLabels(
  prompt: string,
  labels: ReadonlyMap<number, number | null>,
): string {
  return prompt.replace(/<\s*Picture\s+(\d+)\s*>/gi, (raw, rawLabel: string) => {
    const mapped = labels.get(Number(rawLabel));
    if (mapped === undefined) return raw;
    return mapped === null ? "" : `<Picture ${mapped}>`;
  });
}

function assignFl2vaFirstImage(
  segment: FL2VASegment,
  asset: AssetReference,
): FL2VASegment {
  // The stock FL2VA tokenizer labels connected keyframes by presentation
  // order. A last-only segment therefore exposes that image as <Picture 1>;
  // adding a first frame moves the retained last frame to <Picture 2>.
  const prompt = segment.first_image === null && segment.last_image !== null
    ? rewriteFl2vaPictureLabels(segment.prompt, new Map([[1, 2]]))
    : segment.prompt;
  // A first-frame anchor and an incoming continuity guide both target output
  // frame zero. Binding the explicit anchor wins and atomically disables the
  // now-inapplicable seam for every asset entry path.
  return {
    ...segment,
    prompt,
    first_image: asset,
    continuity: segment.continuity.enabled
      ? { ...segment.continuity, enabled: false }
      : segment.continuity,
  };
}

/** Assigns one asset to an explicit v2 editor zone, or to the next natural slot. */
export function assignAssetToSegment(
  segment: TimelineSegment,
  asset: AssetReference,
  target: SegmentAssetTarget = "auto",
): TimelineSegment {
  switch (segment.mode) {
    case "fl2va":
      if (asset.kind !== "image") return segment;
      if (
        target === "auto" &&
        segmentAssetReferences(segment).some((bound) => bound.id === asset.id)
      ) return segment;
      if (target === "last_image") return { ...segment, last_image: asset };
      if (target === "first_image") return assignFl2vaFirstImage(segment, asset);
      if (target !== "auto") return segment;
      if (!segment.first_image) return assignFl2vaFirstImage(segment, asset);
      return { ...segment, last_image: asset };
    case "ref2va":
      if (target === "source_video") {
        if (asset.kind !== "video") return segment;
        // The stock Ref2VA node exposes three video inputs in total. Adding a
        // source consumes the first one, so a source cannot be introduced when
        // all three independent reference slots are already occupied.
        if (
          !segment.source_video &&
          segment.reference_videos.length >=
            MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos
        ) return segment;
        return assignSourceVideo(segment, asset);
      }
      if (target === "reference_image" && asset.kind !== "image") return segment;
      if (target === "reference_video" && asset.kind !== "video") return segment;
      if (target === "reference_audio" && asset.kind !== "audio") return segment;
      if (!["auto", "reference", "reference_image", "reference_video", "reference_audio"].includes(target)) return segment;
      if (asset.kind === "video" && target === "auto" && !segment.source_video)
        return assignAssetToSegment(segment, asset, "source_video");
      if (asset.kind === "image") {
        const { referenceImages } = minimaxH3ReferenceCapacities(
          segment.source_video !== null,
        );
        if (segment.reference_images.length >= referenceImages || segment.reference_images.some((item) => item.id === asset.id))
          return segment;
        return {
          ...segment,
          reference_images: appendToLowestFreeSlots(
            segment.reference_images,
            [asset],
            referenceImages,
          ),
        };
      }
      if (asset.kind === "audio") {
        const { referenceAudios } = minimaxH3ReferenceCapacities(
          segment.source_video !== null,
        );
        if (segment.reference_audios.length >= referenceAudios || segment.reference_audios.some((item) => item.id === asset.id))
          return segment;
        return {
          ...segment,
          reference_audios: appendToLowestFreeSlots(
            segment.reference_audios,
            [asset],
            referenceAudios,
          ),
        };
      }
      if (segment.source_video?.id === asset.id) return segment;
      if (asset.kind !== "video") return segment;
      const videoCapacity = minimaxH3ReferenceCapacities(
        segment.source_video !== null,
      ).referenceVideos;
      if (segment.reference_videos.length >= videoCapacity || segment.reference_videos.some((item) => item.id === asset.id))
        return segment;
      return {
        ...segment,
        reference_videos: appendToLowestFreeSlots(segment.reference_videos, [asset], videoCapacity),
      };
  }
}

export function segmentAcceptsAsset(
  segment: TimelineSegment,
  asset: AssetReference,
  target: SegmentAssetTarget = "auto",
): boolean {
  switch (segment.mode) {
    case "fl2va":
      return asset.kind === "image" && ["auto", "first_image", "last_image"].includes(target);
    case "ref2va":
      if (target === "source_video") return asset.kind === "video";
      if (target === "reference_image") return asset.kind === "image";
      if (target === "reference_video") return asset.kind === "video";
      if (target === "reference_audio") return asset.kind === "audio";
      return ["auto", "reference"].includes(target);
  }
}

export type AssetBindingRejectionReason = "incompatible" | "capacity" | "duplicate";

export interface AssetBindingRejection {
  asset: AssetReference;
  reason: AssetBindingRejectionReason;
}

export interface AssetBindingResult {
  segment: TimelineSegment;
  accepted: AssetReference[];
  rejected: AssetBindingRejection[];
}

function segmentHasAsset(segment: TimelineSegment, assetId: string): boolean {
  return segmentAssetReferences(segment).some((asset) => asset.id === assetId);
}

/**
 * Atomically classifies a multi-asset drop. Automatic Ref2VA assignment sends
 * the first video to the empty source slot and subsequent videos, images and
 * audio to their independent reference grids. FL2VA fills only free anchors;
 * replacing an occupied anchor requires dropping on that explicit zone.
 */
export function assignAssetsToSegment(
  segment: TimelineSegment,
  assets: readonly AssetReference[],
  target: SegmentAssetTarget = "auto",
): AssetBindingResult {
  let next = segment;
  const accepted: AssetReference[] = [];
  const rejected: AssetBindingRejection[] = [];
  const seen = new Set<string>();
  let explicitSlotConsumed = false;

  for (const asset of assets) {
    if (seen.has(asset.id)) {
      rejected.push({ asset, reason: "duplicate" });
      continue;
    }
    seen.add(asset.id);
    if (!segmentAcceptsAsset(next, asset, target)) {
      rejected.push({ asset, reason: "incompatible" });
      continue;
    }

    if (next.mode === "fl2va") {
      if (segmentHasAsset(next, asset.id)) {
        rejected.push({ asset, reason: "duplicate" });
        continue;
      }
      if (target === "first_image" || target === "last_image") {
        if (explicitSlotConsumed) {
          rejected.push({ asset, reason: "capacity" });
          continue;
        }
        next = assignAssetToSegment(next, asset, target);
        explicitSlotConsumed = true;
        accepted.push(asset);
        continue;
      }
      if (!next.first_image) next = assignAssetToSegment(next, asset, "first_image");
      else if (!next.last_image) next = assignAssetToSegment(next, asset, "last_image");
      else {
        rejected.push({ asset, reason: "capacity" });
        continue;
      }
      accepted.push(asset);
      continue;
    }

    if (segmentHasAsset(next, asset.id)) {
      rejected.push({ asset, reason: "duplicate" });
      continue;
    }
    if (target === "source_video") {
      if (explicitSlotConsumed) {
        rejected.push({ asset, reason: "capacity" });
        continue;
      }
      const assigned = assignAssetToSegment(next, asset, target);
      if (assigned === next) {
        rejected.push({ asset, reason: "capacity" });
        continue;
      }
      next = assigned;
      explicitSlotConsumed = true;
      accepted.push(asset);
      continue;
    }
    if (target === "auto" && asset.kind === "video" && !next.source_video) {
      const assigned = assignAssetToSegment(next, asset, "source_video");
      if (assigned === next) {
        rejected.push({ asset, reason: "capacity" });
        continue;
      }
      next = assigned;
      accepted.push(asset);
      continue;
    }
    const list = asset.kind === "image"
      ? next.reference_images
      : asset.kind === "audio"
        ? next.reference_audios
        : next.reference_videos;
    const h3Capacity = minimaxH3ReferenceCapacities(next.source_video !== null);
    const capacity = asset.kind === "image"
      ? h3Capacity.referenceImages
      : asset.kind === "video"
        ? h3Capacity.referenceVideos
        : h3Capacity.referenceAudios;
    if (list.length >= capacity) {
      rejected.push({ asset, reason: "capacity" });
      continue;
    }
    next = assignAssetToSegment(next, asset, "reference");
    accepted.push(asset);
  }
  return { segment: next, accepted, rejected };
}

/** Inserts a canonical token only when it is already exposed by this segment. */
export function insertPromptReferenceToken(
  segment: TimelineSegment,
  token: string,
  selectionStart: number,
  selectionEnd: number,
  expectedMention: string,
): TimelineSegment {
  if (!expectedMention.startsWith("#") || !segmentPromptReferenceTags(segment).includes(token)) return segment;
  return replacePromptMention(segment, token, selectionStart, selectionEnd, expectedMention);
}

/** Inserts a subject tag only when it is defined in this prompt's subject block. */
export function insertPromptSubjectToken(
  segment: TimelineSegment,
  token: string,
  selectionStart: number,
  selectionEnd: number,
  expectedMention: string,
): TimelineSegment {
  if (!expectedMention.startsWith("@") ||
    !promptSubjectReferences(segment.prompt).some((reference) => reference.token === token)) return segment;
  return replacePromptMention(segment, token, selectionStart, selectionEnd, expectedMention);
}

function replacePromptMention(
  segment: TimelineSegment,
  token: string,
  selectionStart: number,
  selectionEnd: number,
  expectedMention: string,
): TimelineSegment {
  const start = Math.max(0, Math.min(segment.prompt.length, Math.trunc(selectionStart)));
  const end = Math.max(start, Math.min(segment.prompt.length, Math.trunc(selectionEnd)));
  if (segment.prompt.slice(start, end) !== expectedMention) return segment;
  const prefix = segment.prompt.slice(0, start);
  const suffix = segment.prompt.slice(end);
  const before = prefix && !/\s$/.test(prefix) ? " " : "";
  const after = suffix && !/^\s/.test(suffix) ? " " : "";
  return {
    ...segment,
    prompt: limitPromptCharacters(`${prefix}${before}${token}${after}${suffix}`),
  };
}

export function removeAssetFromSegment(
  segment: TimelineSegment,
  assetId: string,
): TimelineSegment {
  const removeSlotted = (
    values: SlottedAssetReference[],
    oldLabelOffset = 0,
    newLabelOffset = oldLabelOffset,
  ) => {
    const ordered = [...values].sort((left, right) => left.slot - right.slot);
    const removed = ordered.some((asset) => asset.id === assetId);
    const labels = new Map<number, number | null>();
    const retained: SlottedAssetReference[] = [];
    for (const asset of ordered) {
      const oldLabel = asset.slot + 1 + oldLabelOffset;
      if (asset.id === assetId) {
        labels.set(oldLabel, null);
        continue;
      }
      const slot = retained.length;
      labels.set(oldLabel, slot + 1 + newLabelOffset);
      retained.push({ ...asset, slot });
    }
    return { removed, labels, retained };
  };
  const rewrite = (
    maps: Partial<Record<"Picture" | "Audio" | "Video", Map<number, number | null>>>,
    clearKinds = new Set<string>(),
  ) => {
    return segment.prompt.replace(/<\s*(Picture|Audio|Video)\s+(\d+)\s*>/gi, (raw, rawKind: string, rawLabel: string) => {
      const kind = `${rawKind[0].toUpperCase()}${rawKind.slice(1).toLowerCase()}` as "Picture" | "Audio" | "Video";
      if (clearKinds.has(kind)) return "";
      const mapping = maps[kind];
      if (!mapping?.has(Number(rawLabel))) return raw;
      const next = mapping.get(Number(rawLabel));
      return next == null ? "" : `<${kind} ${next}>`;
    });
  };
  switch (segment.mode) {
    case "fl2va": {
      const firstRemoved = segment.first_image?.id === assetId;
      const lastRemoved = segment.last_image?.id === assetId;
      if (!firstRemoved && !lastRemoved) return segment;
      const labels = new Map<number, number | null>();
      if (segment.first_image) labels.set(1, firstRemoved ? null : 1);
      if (segment.last_image) {
        const oldLastLabel = segment.first_image ? 2 : 1;
        const newLastLabel = segment.first_image && !firstRemoved ? 2 : 1;
        labels.set(oldLastLabel, lastRemoved ? null : newLastLabel);
      }
      const next: FL2VASegment = {
        ...segment,
        prompt: rewrite({ Picture: labels }),
        first_image: firstRemoved ? null : segment.first_image,
        last_image: lastRemoved ? null : segment.last_image,
      };
      return next;
    }
    case "ref2va": {
      const sourceRemoved = segment.source_video?.id === assetId;
      const pictures = removeSlotted(segment.reference_images);
      const sourceAudioOffset = segment.source_audio_as_reference ? 1 : 0;
      const sourceVideoOffset = segment.source_video ? 1 : 0;
      const audios = removeSlotted(
        segment.reference_audios,
        sourceAudioOffset,
        sourceRemoved ? 0 : sourceAudioOffset,
      );
      const videos = removeSlotted(
        segment.reference_videos,
        sourceVideoOffset,
        sourceRemoved ? 0 : sourceVideoOffset,
      );
      if (sourceRemoved && segment.source_audio_as_reference) audios.labels.set(1, null);
      if (sourceRemoved) videos.labels.set(1, null);
      if (!sourceRemoved && !pictures.removed && !audios.removed && !videos.removed)
        return segment;
      const prompt = rewrite({ Picture: pictures.labels, Audio: audios.labels, Video: videos.labels });
      const next: Ref2VASegment = {
        ...segment,
        prompt,
        source_video: sourceRemoved ? null : segment.source_video,
        source_audio_as_reference: sourceRemoved ? false : segment.source_audio_as_reference,
        reference_images: pictures.retained,
        reference_audios: audios.retained,
        reference_videos: videos.retained,
      };
      return next;
    }
  }
}

export function segmentReferenceTag(
  segment: TimelineSegment,
  asset: AssetReference,
): string | null {
  if (segment.mode === "fl2va") {
    const pictures = [segment.first_image, segment.last_image].filter(
      (candidate): candidate is AssetReference => candidate !== null,
    );
    let index = pictures.findIndex((candidate) => candidate === asset);
    if (index < 0) index = pictures.findIndex((candidate) => candidate.id === asset.id);
    return index < 0 ? null : `<Picture ${index + 1}>`;
  }
  if (segment.mode === "ref2va" && segment.source_video?.id === asset.id)
    return "<Video 1>";
  const label = asset.kind === "image" ? "Picture" : asset.kind === "audio" ? "Audio" : "Video";
  const references = segmentAssetReferences(segment).filter((candidate) => "slot" in candidate);
  const reference = references.find((candidate) => candidate.id === asset.id) as SlottedAssetReference | undefined;
  const offset = label === "Audio"
    ? (segment.source_audio_as_reference ? 1 : 0)
    : label === "Video"
      ? (segment.source_video ? 1 : 0)
      : 0;
  return reference ? `<${label} ${reference.slot + 1 + offset}>` : null;
}

/** Every prompt label exposed by the official H3 reference node for a segment. */
export function segmentPromptReferenceTags(segment: TimelineSegment): string[] {
  const tags = segmentAssetReferences(segment)
    .map((asset) => segmentReferenceTag(segment, asset))
    .filter((tag): tag is string => tag !== null);
  if (
    segment.mode === "ref2va" &&
    segment.source_audio_as_reference
  ) tags.push("<Audio 1>");
  return [...new Set(tags)];
}

function rewriteAudioReferenceLabels(
  prompt: string,
  labelMap: ReadonlyMap<number, number | null>,
): string {
  return prompt
    .replace(/<\s*Audio\s+(\d+)\s*>/gi, (raw, label: string) => {
      const mapped = labelMap.get(Number(label));
      if (mapped === undefined) return raw;
      return mapped === null ? "" : `<Audio ${mapped}>`;
    });
}

/**
 * Toggles paired source soundtrack conditioning without changing prompt
 * identity. Ref2VA's independent audio slots stay dense internally, while their
 * official prompt labels move by one because the paired soundtrack is first.
 */
export function setSourceAudioAsReference(
  segment: SourceVideoSegment,
  enabled: boolean,
): SourceVideoSegment {
  if (enabled && segment.source_video?.metadata?.has_audio !== true) return segment;
  if (segment.source_audio_as_reference === enabled) return segment;
  const labels = new Map<number, number | null>();
  segment.reference_audios.forEach((asset) => {
    labels.set(
      asset.slot + 1 + (segment.source_audio_as_reference ? 1 : 0),
      asset.slot + 1 + (enabled ? 1 : 0),
    );
  });
  if (segment.source_audio_as_reference && !enabled) labels.set(1, null);
  const effectivePrompt = segment.prompt;
  const rewritten = rewriteAudioReferenceLabels(effectivePrompt, labels);
  return {
    ...segment,
    source_audio_as_reference: enabled,
    ...(rewritten !== effectivePrompt ? { prompt: rewritten } : {}),
  };
}

function promptTokens(prompt: string): { raw: string; kind: "Picture" | "Audio" | "Video"; slot: number }[] {
  return Array.from(prompt.matchAll(/<\s*(Picture|Audio|Video)\s+(\d+)\s*>/gi), (match) => ({
    raw: match[0],
    kind: `${match[1][0].toUpperCase()}${match[1].slice(1).toLowerCase()}` as "Picture" | "Audio" | "Video",
    slot: Number(match[2]),
  }));
}

function invalidPromptTokens(segment: TimelineSegment, prompt: string): string[] {
  const available = {
    Picture: new Set<number>(),
    Audio: new Set<number>(),
    Video: new Set<number>(),
  };
  if (segment.mode === "fl2va") {
    if (segment.first_image || segment.last_image) available.Picture.add(1);
    if (segment.first_image && segment.last_image) available.Picture.add(2);
  } else {
    segment.reference_images.forEach((asset) => available.Picture.add(asset.slot + 1));
    const audioOffset = segment.source_audio_as_reference ? 1 : 0;
    segment.reference_audios.forEach((asset) =>
      available.Audio.add(asset.slot + 1 + audioOffset));
    const videoOffset = segment.source_video ? 1 : 0;
    segment.reference_videos.forEach((asset) =>
      available.Video.add(asset.slot + 1 + videoOffset));
    if (segment.source_video) available.Video.add(1);
    if (segment.source_audio_as_reference) available.Audio.add(1);
  }
  return promptTokens(prompt)
    .filter((token) => !available[token.kind].has(token.slot))
    .map((token) => token.raw);
}

function duplicateSlots(assets: SlottedAssetReference[]): boolean {
  return new Set(assets.map((asset) => asset.slot)).size !== assets.length;
}

function duplicateAssetIds(assets: SlottedAssetReference[]): boolean {
  return new Set(assets.map((asset) => asset.id)).size !== assets.length;
}

function denseSlots(assets: SlottedAssetReference[]): boolean {
  return [...assets]
    .sort((left, right) => left.slot - right.slot)
    .every((asset, index) => asset.slot === index);
}

/** Mirrors the backend's frame-accurate source trim calculation. */
export function sourceTrimFrameCount(
  segment: SourceVideoSegment,
  fps: number,
): number | null {
  const duration = segment.source_video?.metadata?.duration;
  if (
    !Number.isFinite(duration) ||
    !Number.isFinite(fps) ||
    fps <= 0 ||
    !Number.isFinite(segment.source_start_seconds) ||
    !Number.isFinite(segment.source_duration_seconds)
  ) return null;
  const fullFrames = Math.max(
    1,
    roundPositiveHalfEven((duration as number) * fps),
  );
  const sourceStart = Math.min(
    fullFrames - 1,
    Math.max(0, roundPositiveHalfEven(segment.source_start_seconds * fps)),
  );
  const sourceEnd = Math.min(
    fullFrames,
    Math.max(
      sourceStart + 1,
      roundPositiveHalfEven(
        (segment.source_start_seconds + segment.source_duration_seconds) * fps,
      ),
    ),
  );
  return sourceEnd - sourceStart;
}

export interface SourceAudioTimingAdjustment {
  segment_id: string;
  segment_title: string;
  source_frames_before: number;
  source_frames_after: number;
  output_frames_before: number;
  fallback_to_previous_h3_length: boolean;
}

export interface SourceAudioTimingFit {
  project: TimelineProject;
  adjustments: SourceAudioTimingAdjustment[];
}

function previousH3FrameCount(availableFrames: number): number {
  for (let frames = Math.trunc(availableFrames); frames >= 5; frames -= 1) {
    if (alignH3FrameCount(frames) === frames) return frames;
  }
  return 0;
}

/**
 * Makes source soundtrack timing frame-exact without asking the user to solve
 * MiniMax H3's 17k+5 lattice. Prefer extending the current source range to the
 * already-aligned output. If the source ends first, fall back to the largest
 * legal H3 length that still fits and shorten both source and output.
 *
 * Long source clips that exceed H3's per-shot limit remain untouched: those
 * require a creative split point rather than an automatic tail crop.
 */
export function autoFitSourceAudioTiming(
  project: TimelineProject,
): SourceAudioTimingFit {
  const fps = project.render.fps;
  if (!Number.isFinite(fps) || fps <= 0) return { project, adjustments: [] };

  const adjustments: SourceAudioTimingAdjustment[] = [];
  const segments = project.segments.map((segment) => {
    if (
      !segment.enabled ||
      segment.audio_mode !== "source" ||
      segment.mode !== "ref2va" ||
      !segment.source_video?.metadata
    ) return segment;
    const metadataDuration = segment.source_video.metadata.duration;
    if (!Number.isFinite(metadataDuration) || metadataDuration <= 0) return segment;
    const sourceFrames = sourceTrimFrameCount(segment, fps);
    const outputFrames = alignH3Frames(segment.duration_seconds, fps);
    if (
      sourceFrames === null ||
      sourceFrames === outputFrames ||
      outputFrames < 5 ||
      outputFrames > H3_MAX_SHOT_FRAMES
    ) return segment;

    const fullFrames = Math.max(1, roundPositiveHalfEven(metadataDuration * fps));
    const sourceStartFrame = Math.min(
      fullFrames - 1,
      Math.max(0, roundPositiveHalfEven(segment.source_start_seconds * fps)),
    );
    const availableFrames = fullFrames - sourceStartFrame;
    const targetFrames = outputFrames <= availableFrames
      ? outputFrames
      : previousH3FrameCount(availableFrames);
    if (targetFrames < 5) return segment;

    const remainingSeconds = metadataDuration - segment.source_start_seconds;
    const durationCandidates = [
      targetFrames / fps,
      (sourceStartFrame + targetFrames) / fps - segment.source_start_seconds,
      remainingSeconds,
    ];
    let fitted: Ref2VASegment | null = null;
    for (const sourceDuration of durationCandidates) {
      if (
        !Number.isFinite(sourceDuration) ||
        sourceDuration <= 0 ||
        sourceDuration > remainingSeconds + 1e-6
      ) continue;
      const candidate: Ref2VASegment = {
        ...segment,
        source_duration_seconds: Math.min(sourceDuration, remainingSeconds),
        duration_seconds: targetFrames < outputFrames
          ? targetFrames / fps
          : segment.duration_seconds,
      };
      if (
        sourceTrimFrameCount(candidate, fps) === targetFrames &&
        alignH3Frames(candidate.duration_seconds, fps) === targetFrames
      ) {
        fitted = candidate;
        break;
      }
    }
    if (!fitted) return segment;
    adjustments.push({
      segment_id: segment.id,
      segment_title: segment.title,
      source_frames_before: sourceFrames,
      source_frames_after: targetFrames,
      output_frames_before: outputFrames,
      fallback_to_previous_h3_length: targetFrames < outputFrames,
    });
    return fitted;
  });

  return adjustments.length
    ? { project: touchProject(project, segments), adjustments }
    : { project, adjustments };
}

/** Submission validation shared by the top bar and timeline command bar. */
export function validateTimelineProject(
  project: TimelineProject,
  segmentIds?: readonly string[],
): string[] {
  const errors: string[] = [];
  const selection = segmentIds ? new Set(segmentIds) : null;
  if (!project.title.trim() || project.title.length > 256) errors.push("项目名称必须为 1–256 个字符");
  if (
    ![project.render.width, project.render.height].every((value) =>
      Number.isInteger(value) && value >= 32 && value <= 8192 && value % 32 === 0,
    )
  ) errors.push("项目宽高必须是 32–8192 范围内的 32 倍数");
  if (project.render.fps !== 24)
    errors.push("原生 MiniMax H3 时间线帧率固定为 24fps");
  for (const [familyLabel, sampling] of [
    ["FL2VA", project.sampling.fl2va],
    ["Ref2VA", project.sampling.ref2va],
  ] as const) {
    if (!Number.isInteger(sampling.steps) || sampling.steps < 1 || sampling.steps > 200)
      errors.push(`${familyLabel} 采样步数必须是 1–200 的整数`);
    if (!Number.isSafeInteger(sampling.seed) || sampling.seed < 0)
      errors.push(`${familyLabel} Seed 必须是非负 JavaScript 安全整数`);
    if (typeof sampling.random_seed !== "boolean")
      errors.push(`${familyLabel} 随机 Seed 状态无效`);
    if (![sampling.shift, sampling.audio_shift].every((value) => Number.isFinite(value) && value >= 0.01 && value <= 100))
      errors.push(`${familyLabel} Video / Audio Shift 必须在 0.01–100 之间`);
  }
  if (project.segments.length < 1 || project.segments.length > 128)
    errors.push("时间线片段数量必须在 1–128 之间");
  if (new Set(project.segments.map((segment) => segment.id)).size !== project.segments.length)
    errors.push("时间线片段 ID 必须唯一");
  if (selection) {
    const known = new Set(project.segments.map((segment) => segment.id));
    if (!selection.size || [...selection].some((id) => !known.has(id)))
      errors.push("选中的时间线片段不存在");
  }
  const scopedSegments = selection
    ? project.segments.filter((segment) => selection.has(segment.id))
    : project.segments;
  const enabled = scopedSegments.filter((segment) => segment.enabled);
  if (!enabled.length) errors.push("至少启用一个时间线片段");
  const totalFrames = enabled.reduce((total, segment) => total + alignH3Frames(segment.duration_seconds, project.render.fps), 0);
  if (totalFrames > 100_000) errors.push("启用片段的总有效帧数不能超过 100,000");
  for (const segment of scopedSegments) {
    const index = project.segments.findIndex((candidate) => candidate.id === segment.id);
    const label = `${index + 1} · ${segment.title || segment.id}`;
    const recipe = deriveSegmentRecipe(segment);
    const minimum = recipe === "fl2v" ? 0.1 : Number.EPSILON;
    const editableDurationLimit = segment.mode === "ref2va" ? 86_400 : 120;
    if (!Number.isFinite(segment.duration_seconds) || segment.duration_seconds < minimum || segment.duration_seconds > editableDurationLimit)
      errors.push(`${label} 的时长必须${recipe === "fl2v" ? "不小于 0.1" : "大于 0"}且不超过 ${editableDurationLimit.toLocaleString()} 秒`);
    if (alignH3Frames(segment.duration_seconds, project.render.fps) > H3_MAX_SHOT_FRAMES) {
      errors.push(segment.mode === "ref2va" && segment.source_video
        ? `${label} 已完整载入源视频，但超过 MiniMax H3 的 512 帧生成上限；请先用播放头、均分或智能分割切成较短片段`
        : `${label} 超过 MiniMax H3 的 512 帧上限`);
    }
    if (promptCharacterCount(segment.prompt) > MINIMAX_H3_PROMPT_MAX_CHARACTERS)
      errors.push(`${label} 的提示词超过 ${MINIMAX_H3_PROMPT_MAX_CHARACTERS.toLocaleString()} 字符`);
    if (segment.enabled && !segment.prompt.trim())
      errors.push(`${label} 没有片段提示词`);
    if (!segment.enabled) continue;
    if (segment.audio_mode === "source" && (segment.mode !== "ref2va" || !segment.source_video)) {
      errors.push(`${label} 使用“保留源音频”时必须是 Ref2VA 且已绑定源视频`);
    }
    if (segment.mode === "fl2va") {
      if (segment.first_image && !isStableAssetReference(segment.first_image, "image"))
        errors.push(`${label} 的首帧素材无效`);
      if (segment.last_image && !isStableAssetReference(segment.last_image, "image"))
        errors.push(`${label} 的尾帧素材无效`);
    }
    if (segment.mode === "ref2va") {
      const hasReferences = Boolean(segment.reference_images.length || segment.reference_audios.length || segment.reference_videos.length);
      if (!segment.source_video && !hasReferences)
        errors.push(`${label} 至少需要源视频或一个独立参考素材`);
      const h3Capacity = minimaxH3ReferenceCapacities(segment.source_video !== null);
      const maxIndependentVideoSlot = maxSlotForCapacity(h3Capacity.referenceVideos);
      if (segment.reference_images.some((asset) => !isStableSlottedAssetReference(asset, "image", maxSlotForCapacity(h3Capacity.referenceImages))) || segment.reference_audios.some((asset) => !isStableSlottedAssetReference(asset, "audio", maxSlotForCapacity(h3Capacity.referenceAudios))) || segment.reference_videos.some((asset) => !isStableSlottedAssetReference(asset, "video", maxIndependentVideoSlot)) || duplicateSlots(segment.reference_images) || duplicateSlots(segment.reference_audios) || duplicateSlots(segment.reference_videos) || duplicateAssetIds(segment.reference_images) || duplicateAssetIds(segment.reference_audios) || duplicateAssetIds(segment.reference_videos))
        errors.push(`${label} 的参考素材槽位无效或重复，且同一素材 ID 只能占用一个槽位`);
      if (segment.reference_videos.length + (segment.source_video ? h3Capacity.sourceVideo : 0) > h3Capacity.totalReferenceVideos)
        errors.push(`${label} 的源视频与独立参考视频合计不能超过 ${h3Capacity.totalReferenceVideos} 个`);
      if (!denseSlots(segment.reference_images))
        errors.push(`${label} 的参考图片槽位必须连续为 0..N-1`);
      if (!denseSlots(segment.reference_audios))
        errors.push(`${label} 的参考音频槽位必须连续为 0..N-1`);
      if (!denseSlots(segment.reference_videos))
        errors.push(`${label} 的参考视频槽位必须连续为 0..N-1`);
      if (segment.reference_videos.some((asset) => (asset.metadata?.frame_count ?? 0) < 5))
        errors.push(`${label} 的参考视频至少需要 5 帧`);
      if (segment.source_video && !isStableAssetReference(segment.source_video, "video"))
        errors.push(`${label} 的源视频无效`);
      if (segment.source_video && (!Number.isFinite(segment.source_start_seconds) || segment.source_start_seconds < 0 || !Number.isFinite(segment.source_duration_seconds) || segment.source_duration_seconds <= 0))
        errors.push(`${label} 的源视频范围无效`);
      if (segment.source_video?.metadata && segment.source_start_seconds + segment.source_duration_seconds > segment.source_video.metadata.duration + 1e-6)
        errors.push(`${label} 的源视频范围超过素材实际时长`);
      if (segment.source_audio_as_reference && segment.source_video?.metadata?.has_audio !== true)
        errors.push(`${label} 已启用源音轨参考，但源视频没有可用音轨`);
      const trimFrames = sourceTrimFrameCount(segment, project.render.fps);
      if (trimFrames !== null && trimFrames < 5)
        errors.push(`${label} 的源视频范围至少需要 5 帧`);
      if (segment.source_video && segment.reference_videos.some((asset) => asset.id === segment.source_video?.id))
        errors.push(`${label} 的源视频不能同时占用独立参考视频槽位`);
      if (segment.audio_mode === "source" && segment.source_video) {
        if (segment.source_video.metadata?.has_audio !== true) {
          errors.push(`${label} 无法保留源音频：源视频没有可用音轨`);
        } else {
          const sourceFrames = sourceTrimFrameCount(segment, project.render.fps);
          const outputFrames = alignH3Frames(segment.duration_seconds, project.render.fps);
          if (sourceFrames !== outputFrames) {
            errors.push(`${label} 保留源音频时不能变速：源截取为 ${sourceFrames ?? "—"} 帧，输出为 ${outputFrames} 帧（建议源时长 ${(outputFrames / project.render.fps).toFixed(4)} 秒）`);
          }
        }
      }
    }
    const invalidTokens = invalidPromptTokens(segment, segment.prompt);
    if (invalidTokens.length) errors.push(`${label} 引用了未绑定素材：${[...new Set(invalidTokens)].join("、")}`);
  }
  errors.push(...timelineContinuityRunIssues(project, segmentIds)
    .filter((issue) => issue.code !== "historical-take-required")
    .map((issue) => issue.message));
  return [...new Set(errors)];
}

/**
 * v2 was historically a durable mirror, so its mere presence cannot prove
 * that the browser owns a write newer than SQLite. Never replay it as a WAL.
 */
export const LEGACY_TIMELINE_STORAGE_KEY = "director-web:v2:timeline";
export const QUARANTINED_TIMELINE_STORAGE_KEY = "director-web:v2:timeline-quarantine";
export const UNBOUND_TIMELINE_WAL_STORAGE_KEY = "director-web:v3:timeline-wal";
export const QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY = "director-web:v3:timeline-wal-quarantine";
// v4 carried a single unscoped pending timeline (pre-multi-project). Its bytes
// are quarantined rather than replayed because they cannot name a project.
export const LEGACY_V4_TIMELINE_WAL_STORAGE_KEY = "director-web:v4:timeline-wal";
export const TIMELINE_WAL_STORAGE_KEY = "director-web:v5:timeline-wal";
export const QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY = "director-web:v5:timeline-wal-quarantine";
const TIMELINE_WAL_FORMAT = "director-pending-timeline";
const LEGACY_BOUND_TIMELINE_WAL_VERSION = 2;
const TIMELINE_WAL_VERSION = 4;
let timelineWalOwnerCache: string | null = null;
let adoptedTimelineWalRaw: string | null = null;
let latestTimelineWalRaw: string | null = null;
const ASSET_LAYOUT_STORAGE_KEY = "director-web:v2:asset-layout";

export interface AssetLayoutPreference {
  size: AssetGridSize;
  order: string[];
}

export function loadAssetLayoutPreference(): AssetLayoutPreference {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(ASSET_LAYOUT_STORAGE_KEY) ?? "null");
    if (!isRecord(value)) throw new Error("invalid layout");
    const size = ["small", "medium", "large"].includes(String(value.size))
      ? value.size as AssetGridSize
      : "medium";
    const order = Array.isArray(value.order)
      ? value.order.filter((id): id is string => typeof id === "string" && id.length > 0)
      : [];
    return { size, order: [...new Set(order)] };
  } catch {
    return { size: "medium", order: [] };
  }
}

export function saveAssetLayoutPreference(
  size: AssetGridSize,
  assets: AssetReference[],
): void {
  try {
    localStorage.setItem(ASSET_LAYOUT_STORAGE_KEY, JSON.stringify({
      size,
      // IDs are opaque Director identities. Never persist endpoint, paths or
      // other server-authoritative asset metadata as a UI layout preference.
      order: assets.map((asset) => asset.id),
    }));
  } catch {
    // In-memory order remains usable.
  }
}

export function orderAssetsByPreference(
  assets: AssetReference[],
  order: string[],
): AssetReference[] {
  const priority = new Map(order.map((id, index) => [id, index]));
  return [...assets].sort((left, right) => {
    const leftIndex = priority.get(left.id);
    const rightIndex = priority.get(right.id);
    if (leftIndex === undefined && rightIndex === undefined) return 0;
    if (leftIndex === undefined) return 1;
    if (rightIndex === undefined) return -1;
    return leftIndex - rightIndex;
  });
}

function quarantineStorageEntry(sourceKey: string, quarantineKey: string): void {
  try {
    const raw = localStorage.getItem(sourceKey);
    if (raw === null) return;
    // Keep every distinct WAL byte-for-byte. A fixed first key is convenient
    // for manual recovery, while numbered siblings prevent a later database's
    // WAL from being discarded merely because that first slot is occupied.
    let destination = quarantineKey;
    let existing = localStorage.getItem(destination);
    for (let index = 1; existing !== null && existing !== raw; index += 1) {
      destination = `${quarantineKey}:${index}`;
      existing = localStorage.getItem(destination);
    }
    if (existing === null) localStorage.setItem(destination, raw);
    // Only delete the replay-capable source after its exact bytes are known to
    // exist in quarantine. Quota/security failures therefore preserve it.
    if (
      localStorage.getItem(destination) === raw &&
      localStorage.getItem(sourceKey) === raw
    ) localStorage.removeItem(sourceKey);
  } catch {
    // If quarantine cannot be written, leave the old key untouched. It remains
    // ignored by hydration and therefore cannot overwrite server authority.
  }
}

function quarantineObsoleteTimelineStorage(): void {
  quarantineStorageEntry(LEGACY_TIMELINE_STORAGE_KEY, QUARANTINED_TIMELINE_STORAGE_KEY);
  // v3 proved that a write was pending, but it did not identify which SQLite
  // database owned that write. It must remain recoverable without being replayed.
  quarantineStorageEntry(
    UNBOUND_TIMELINE_WAL_STORAGE_KEY,
    QUARANTINED_UNBOUND_TIMELINE_WAL_STORAGE_KEY,
  );
  // v4 predates project scoping and cannot name a project; quarantine rather
  // than replay so a stale pending edit cannot cross project boundaries.
  quarantineStorageEntry(
    LEGACY_V4_TIMELINE_WAL_STORAGE_KEY,
    QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY,
  );
}

function isActiveDatabaseIdentity(value: unknown): value is string {
  return typeof value === "string" &&
    value.length > 1 &&
    value.length <= 4096 &&
    value.startsWith("/") &&
    !/[\u0000-\u001f\u007f]/.test(value);
}

export interface TimelineWalDatabaseIdentity {
  active_database_path: string;
  active_database_identity: string;
}

function isTimelineWalDatabaseIdentity(value: TimelineWalDatabaseIdentity): boolean {
  return isActiveDatabaseIdentity(value.active_database_path) &&
    /^[0-9a-f]{64}$/.test(value.active_database_identity);
}

function validTimelineWalOwner(value: unknown): value is string {
  return typeof value === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(value);
}

function validTimelineWalProjectId(value: unknown): value is string {
  // "default" and server-generated UUIDs both satisfy this shape.
  return typeof value === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(value);
}

function timelineWalOwner(): string {
  if (timelineWalOwnerCache) return timelineWalOwnerCache;
  // Keep ownership document-scoped. sessionStorage can be cloned when a tab is
  // duplicated, which would let two live pages mistake each other's WAL for
  // their own and clear it after a late response.
  const generated = typeof globalThis.crypto?.randomUUID === "function"
    ? `tab-${globalThis.crypto.randomUUID()}`
    : `tab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
  timelineWalOwnerCache = generated;
  return generated;
}

function parseTimelineWalEnvelope(raw: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value)) return null;
    const keys = Object.keys(value).sort().join("|");
    const legacy = value.version === LEGACY_BOUND_TIMELINE_WAL_VERSION &&
      keys === "active_database_identity|active_database_path|format|pending|project|version|written_at_ms";
    const owned = value.version === TIMELINE_WAL_VERSION &&
      keys === "active_database_identity|active_database_path|format|owner_id|pending|project|project_id|version|written_at_ms" &&
      validTimelineWalOwner(value.owner_id) &&
      validTimelineWalProjectId(value.project_id);
    if (
      (!legacy && !owned) ||
      value.format !== TIMELINE_WAL_FORMAT ||
      value.pending !== true ||
      !isActiveDatabaseIdentity(value.active_database_path) ||
      typeof value.active_database_identity !== "string" ||
      !/^[0-9a-f]{64}$/.test(value.active_database_identity) ||
      !Number.isSafeInteger(value.written_at_ms) ||
      (value.written_at_ms as number) <= 0
    ) return null;
    return value;
  } catch {
    return null;
  }
}

function timelineWalOwnedByCurrentTab(
  raw: string,
  database: TimelineWalDatabaseIdentity,
  owner: string,
): boolean {
  const value = parseTimelineWalEnvelope(raw);
  return value?.version === TIMELINE_WAL_VERSION &&
    value.owner_id === owner &&
    value.active_database_path === database.active_database_path &&
    value.active_database_identity === database.active_database_identity;
}

export function loadLocalTimeline(
  database: TimelineWalDatabaseIdentity,
  projectId: string = DEFAULT_PROJECT_ID,
): TimelineProject | null {
  quarantineObsoleteTimelineStorage();
  if (!isTimelineWalDatabaseIdentity(database)) return null;
  try {
    // Only an explicit pending-write envelope owned by the currently active
    // SQLite database AND the requested project may win over the server GET.
    const raw = localStorage.getItem(TIMELINE_WAL_STORAGE_KEY);
    if (!raw) return null;
    const value = parseTimelineWalEnvelope(raw);
    if (
      !value ||
      value.active_database_path !== database.active_database_path ||
      value.active_database_identity !== database.active_database_identity ||
      value.project_id !== projectId
    ) throw new Error("invalid timeline WAL envelope");
    const project = normalizeTimelineProject(value.project);
    if (!project) throw new Error("invalid timeline WAL project");
    adoptedTimelineWalRaw = raw;
    return project;
  } catch {
    quarantineStorageEntry(
      TIMELINE_WAL_STORAGE_KEY,
      QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY,
    );
    return null;
  }
}

export function saveLocalTimeline(
  project: TimelineProject,
  database: TimelineWalDatabaseIdentity,
  projectId: string = DEFAULT_PROJECT_ID,
): void {
  quarantineObsoleteTimelineStorage();
  if (!isTimelineWalDatabaseIdentity(database)) return;
  try {
    const owner = timelineWalOwner();
    const raw = JSON.stringify({
      format: TIMELINE_WAL_FORMAT,
      version: TIMELINE_WAL_VERSION,
      owner_id: owner,
      pending: true,
      project_id: projectId,
      active_database_path: database.active_database_path,
      active_database_identity: database.active_database_identity,
      written_at_ms: Date.now(),
      project,
    });
    const existing = localStorage.getItem(TIMELINE_WAL_STORAGE_KEY);
    if (existing !== null && !timelineWalOwnedByCurrentTab(existing, database, owner)) {
      quarantineStorageEntry(
        TIMELINE_WAL_STORAGE_KEY,
        QUARANTINED_MISMATCHED_TIMELINE_WAL_STORAGE_KEY,
      );
      // Never overwrite bytes that could not be copied, or that another tab
      // raced into the shared source key while quarantine was in progress.
      if (localStorage.getItem(TIMELINE_WAL_STORAGE_KEY) !== null) return;
    }
    localStorage.setItem(TIMELINE_WAL_STORAGE_KEY, raw);
    if (localStorage.getItem(TIMELINE_WAL_STORAGE_KEY) === raw) {
      latestTimelineWalRaw = raw;
    }
  } catch {
    // In-memory editing remains available when browser storage is unavailable.
  }
}

/**
 * Discards a browser draft that can no longer be trusted to match the server.
 * Used after an atomic server mutation succeeds but its authoritative document
 * cannot be read back; a reload must then prefer GET /api/timeline.
 */
export function clearLocalTimeline(): void {
  quarantineObsoleteTimelineStorage();
  try {
    const current = localStorage.getItem(TIMELINE_WAL_STORAGE_KEY);
    if (current !== null && (current === latestTimelineWalRaw || current === adoptedTimelineWalRaw)) {
      localStorage.removeItem(TIMELINE_WAL_STORAGE_KEY);
    }
    latestTimelineWalRaw = null;
    adoptedTimelineWalRaw = null;
  } catch {
    // The current tab remains fail-closed even if storage is unavailable.
  }
}
