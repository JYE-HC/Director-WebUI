import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type DragEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import type {
  CapabilityReport,
  GenerationTask,
  RV2VShotDetectionRequest,
  TimelineCompileReport,
} from "../api/types";
import { directorApi } from "../api/client";
import {
  DIRECTOR_ASSET_MIME,
  DIRECTOR_ASSETS_MIME,
  DIRECTOR_SEGMENT_REFERENCE_MIME,
  directorAssetIdsFromTransfer,
  directorSegmentReferenceFromTransfer,
  type DroppedUploadProgress,
  type DroppedUploadResult,
} from "../domain/assetDrag";
import { MODE_META as RECIPE_META, type AssetReference } from "../domain/modes";
import {
  assignAssetsToSegment,
  alignedTimelineSegmentDuration,
  canMergeSelectedSegments,
  canSplitSelectedSegment,
  createSegmentId,
  deriveSegmentRecipe,
  promptSubjectReferences,
  promptSkeleton,
  effectiveTimelineSegmentDuration,
  segmentAcceptsAsset,
  segmentAssetReferences,
  segmentPromptReferenceTags,
  segmentReferenceTag,
  segmentStartTime,
  sourceTrimFrameCount,
  sourcePreviewTime,
  sourceTimelineThumbnailTimes,
  timelineSegmentPlaybackDuration,
  timelineDuration,
  timelineContinuityBoundaries,
  timelineContinuityRunIssues,
  timelineSegmentAt,
  runnableTimelineSegmentIds,
  DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS,
  TIMELINE_MODE_META,
  TIMELINE_MODE_ORDER,
  type SegmentAssetTarget,
  type AssetBindingResult,
  type TimelineEditorState,
  type TimelineGenerationMode,
  type TimelineSegment,
  type TimelineSegmentCopyOptions,
  type TimelineUserAction,
} from "../domain/timelineProject";
import { alignH3Frames } from "../domain/timing";
import { minimaxH3ReferenceCapacities } from "../domain/h3Capabilities";
import {
  limitPromptCharacters,
  MINIMAX_H3_PROMPT_MAX_CHARACTERS,
  promptCharacterCount,
} from "../domain/promptLimits";
import { DeferredNumberInput, Field } from "./ui";
import {
  loadTimelineSegmentCopyOptions,
  loadTimelineWorkspacePreferences,
  normalizeTimelineSegmentCopyOptions,
  saveTimelineSegmentCopyOptions,
  updateTimelineWorkspacePreferences,
} from "../domain/workspacePreferences";

interface LongFormTimelineWorkspaceProps {
  state: TimelineEditorState;
  capabilities: CapabilityReport;
  activeTask: GenerationTask | null;
  segmentCandidates: Record<string, {
    job_id: string;
    job_updated_at: string;
    result: GenerationTask["segment_results"][number];
  }>;
  compileReport: TimelineCompileReport | null;
  selectionValidationErrors: string[];
  onDispatch: (action: TimelineUserAction) => void;
  onCloseCompile: () => void;
  onCancelTask: (taskId: string) => void;
  onUploadFiles?: (
    files: File[],
    onProgress?: (progress: DroppedUploadProgress) => void,
  ) => Promise<DroppedUploadResult>;
}

export const TIMELINE_RUN_VALIDATION_ID = "timeline-run-validation";
export const TIMELINE_ZOOM_MIN = 12;
export const TIMELINE_ZOOM_MAX = 240;
const TIMELINE_MAX_TICKS = 2_000;

function clampTimelineZoom(value: number): number {
  return Math.min(TIMELINE_ZOOM_MAX, Math.max(TIMELINE_ZOOM_MIN, value));
}

/** Snaps every visual seek path to the same project-frame boundary. */
export function snapTimelineSeconds(seconds: number, fps: number, total: number): number {
  if (!Number.isFinite(seconds) || !Number.isFinite(fps) || fps <= 0) return 0;
  const bounded = Math.min(Math.max(0, total), Math.max(0, seconds));
  return Math.min(Math.max(0, total), Math.round(bounded * fps) / fps);
}

/** Chooses a readable major interval while keeping very long timelines bounded. */
export function timelineMajorTickSeconds(pixelsPerSecond: number, total: number): number {
  const candidates = [1 / 24, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  const readable = candidates.find((seconds) => seconds * pixelsPerSecond >= 72) ?? 600;
  const bounded = Math.max(readable, total > 0 ? total / TIMELINE_MAX_TICKS : readable);
  return candidates.find((seconds) => seconds >= bounded) ?? bounded;
}

function formatRulerClock(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  const precision = remainder % 1 === 0 ? 0 : remainder < 10 ? 2 : 1;
  return `${minutes}:${remainder.toFixed(precision).padStart(precision ? 5 : 2, "0")}`;
}

function TransportBoundaryIcon({ edge }: { edge: "start" | "end" }) {
  return edge === "start" ? <svg className="transport-boundary-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
    <path d="M3 2.5v11M13 3.25 5.25 8 13 12.75Z" />
  </svg> : <svg className="transport-boundary-icon" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
    <path d="M13 2.5v11M3 3.25 10.75 8 3 12.75Z" />
  </svg>;
}

function interactiveTimelineTarget(target: EventTarget | null): boolean {
  return target instanceof Element && Boolean(target.closest(
    ".timeline-clip, input, textarea, select, button, a, audio, video, [contenteditable='true'], [role='button'], [role='slider']",
  ));
}

type ReferenceKind = "image" | "video" | "audio";
type SegmentDropZone = "before" | "bind" | "after";

type PromptPickerKind = "subject" | "asset";

interface PromptPickerState {
  kind: PromptPickerKind;
  trigger: "@" | "#";
  start: number;
  end: number;
  query: string;
  activeIndex: number;
  position: { left: number; top: number };
}

interface PromptAssetSuggestion {
  type: "asset";
  key: string;
  name: string;
  kind: AssetReference["kind"];
  token: string;
  role: string;
  previewUrl: string | null;
}

interface PromptSubjectSuggestion {
  type: "subject";
  key: string;
  token: string;
  definition: string;
}

type PromptSuggestion = PromptAssetSuggestion | PromptSubjectSuggestion;

function promptPickerPosition(textarea: HTMLTextAreaElement, index: number): { left: number; top: number } {
  const rect = textarea.getBoundingClientRect();
  const computed = window.getComputedStyle(textarea);
  const mirror = document.createElement("div");
  mirror.setAttribute("aria-hidden", "true");
  Object.assign(mirror.style, {
    position: "fixed",
    visibility: "hidden",
    pointerEvents: "none",
    boxSizing: computed.boxSizing,
    left: `${rect.left}px`,
    top: `${rect.top}px`,
    width: `${rect.width}px`,
    height: `${rect.height}px`,
    padding: computed.padding,
    borderStyle: computed.borderStyle,
    borderWidth: computed.borderWidth,
    fontFamily: computed.fontFamily,
    fontSize: computed.fontSize,
    fontWeight: computed.fontWeight,
    fontStyle: computed.fontStyle,
    letterSpacing: computed.letterSpacing,
    lineHeight: computed.lineHeight,
    textAlign: computed.textAlign,
    textIndent: computed.textIndent,
    textTransform: computed.textTransform,
    whiteSpace: "pre-wrap",
    overflowWrap: "break-word",
    overflow: "hidden",
  });
  mirror.append(document.createTextNode(textarea.value.slice(0, index)));
  const marker = document.createElement("span");
  marker.textContent = "\u200b";
  mirror.append(marker);
  document.body.append(mirror);
  const markerRect = marker.getBoundingClientRect();
  mirror.remove();

  const viewportWidth = Math.max(320, window.innerWidth);
  const viewportHeight = Math.max(240, window.innerHeight);
  const popupWidth = Math.min(360, viewportWidth - 16);
  const popupHeight = 240;
  const lineHeight = Number.parseFloat(computed.lineHeight) || markerRect.height || 20;
  const anchorLeft = markerRect.left - textarea.scrollLeft;
  const anchorTop = markerRect.top - textarea.scrollTop;
  const left = Math.min(Math.max(8, anchorLeft), Math.max(8, viewportWidth - popupWidth - 8));
  const below = anchorTop + lineHeight + 6;
  const top = below + popupHeight <= viewportHeight - 8
    ? below
    : Math.max(8, anchorTop - popupHeight - 6);
  return { left, top };
}

function formatClock(seconds: number): string {
  const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const minutes = Math.floor(safe / 60);
  return `${String(minutes).padStart(2, "0")}:${(safe - minutes * 60).toFixed(2).padStart(5, "0")}`;
}

function seekVideo(video: HTMLVideoElement, seconds: number, tolerance = 0): void {
  if (!Number.isFinite(seconds)) return;
  const duration = Number.isFinite(video.duration) && video.duration >= 0
    ? video.duration
    : Number.POSITIVE_INFINITY;
  const next = Math.min(duration, Math.max(0, seconds));
  if (Number.isFinite(video.currentTime) && Math.abs(video.currentTime - next) <= tolerance) return;
  try {
    video.currentTime = next;
  } catch {
    // Some browsers reject seeks until metadata has arrived. loadedmetadata
    // invokes the same synchronization again with the authoritative playhead.
  }
}

/**
 * Builds a small source filmstrip with one decoder and several canvas frames.
 * Using one <video> per thumbnail would multiply range requests and decoder
 * state across every timeline segment; sequential seeks keep the same visual
 * result bounded to one media element per source-backed clip.
 */
function SourceVideoFilmstrip({
  segment,
}: {
  segment: Extract<TimelineSegment, { mode: "ref2va" }>;
}) {
  const source = segment.source_video;
  const times = useMemo(
    () => sourceTimelineThumbnailTimes(segment),
    [
      segment.source_start_seconds,
      segment.source_duration_seconds,
      source?.id,
      source?.preview_url,
      source?.metadata?.duration,
    ],
  );
  const canvasesRef = useRef<Array<HTMLCanvasElement | null>>([]);
  const backdropCanvasesRef = useRef<Array<HTMLCanvasElement | null>>([]);
  const frameIndexRef = useRef(0);
  const canvasWidth = 192;
  const canvasHeight = Math.max(1, Math.round(
    canvasWidth * (source?.metadata?.height ?? 9) / (source?.metadata?.width ?? 16),
  ));

  useEffect(() => {
    frameIndexRef.current = 0;
  }, [times]);

  if (!source?.preview_url || !times.length) return null;
  const seekCurrentFrame = (video: HTMLVideoElement) => {
    seekVideo(video, times[frameIndexRef.current] ?? times[0]);
  };
  const captureAndAdvance = (video: HTMLVideoElement) => {
    const index = frameIndexRef.current;
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      const decodedHeight = Math.max(1, Math.round(
        canvasWidth * video.videoHeight / video.videoWidth,
      ));
      [backdropCanvasesRef.current[index], canvasesRef.current[index]].forEach((canvas) => {
        if (!canvas) return;
        if (canvas.width !== canvasWidth) canvas.width = canvasWidth;
        if (canvas.height !== decodedHeight) canvas.height = decodedHeight;
        canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
      });
    }
    if (index + 1 >= times.length) return;
    frameIndexRef.current = index + 1;
    seekCurrentFrame(video);
  };

  return <div
    className="timeline-source-filmstrip"
    style={{ "--source-thumb-count": times.length } as CSSProperties}
    aria-label={`${source.name} 关键帧缩略图`}
  >
    {times.map((seconds, index) => <span className="timeline-source-filmstrip__frame" key={`${seconds}-${index}`}>
      <canvas
        className="timeline-source-filmstrip__backdrop"
        ref={(canvas) => { backdropCanvasesRef.current[index] = canvas; }}
        width={canvasWidth}
        height={canvasHeight}
        aria-hidden="true"
      />
      <canvas
        className="timeline-source-filmstrip__image"
        ref={(canvas) => { canvasesRef.current[index] = canvas; }}
        width={canvasWidth}
        height={canvasHeight}
        role="img"
        aria-label={`源视频关键帧 ${index + 1}，${formatClock(seconds)}`}
      />
    </span>)}
    <video
      className="timeline-source-filmstrip__decoder"
      src={source.preview_url}
      muted
      playsInline
      preload="metadata"
      tabIndex={-1}
      aria-hidden="true"
      onLoadedMetadata={(event) => seekCurrentFrame(event.currentTarget)}
      onSeeked={(event) => captureAndAdvance(event.currentTarget)}
    />
  </div>;
}

function segmentModeLabel(mode: TimelineGenerationMode): string {
  return `${TIMELINE_MODE_META[mode].shortLabel} · ${TIMELINE_MODE_META[mode].label}`;
}

function assetGlyph(kind: AssetReference["kind"]): string {
  return kind === "image" ? "▧" : kind === "audio" ? "♫" : "▶";
}

function dropAssetId(event: DragEvent): string {
  return dropAssetIds(event)[0] ?? "";
}

function dropAssetIds(event: DragEvent): string[] {
  return directorAssetIdsFromTransfer(event.dataTransfer);
}

function directorAssetDragKind(event: DragEvent): AssetReference["kind"] | null {
  const types = Array.from(event.dataTransfer.types ?? []);
  if (types.includes("application/x-director-asset-video")) return "video";
  if (types.includes("application/x-director-asset-image")) return "image";
  if (types.includes("application/x-director-asset-audio")) return "audio";
  return null;
}

function hasDirectorAssetDrag(event: DragEvent): boolean {
  const types = Array.from(event.dataTransfer.types ?? []);
  return types.includes(DIRECTOR_ASSET_MIME) || types.includes(DIRECTOR_ASSETS_MIME) || Boolean(dropAssetId(event));
}

function hasDirectorSegmentReferenceDrag(event: DragEvent): boolean {
  return Array.from(event.dataTransfer.types ?? []).includes(
    DIRECTOR_SEGMENT_REFERENCE_MIME,
  );
}

function hasSystemFileDrag(event: DragEvent): boolean {
  return Array.from(event.dataTransfer.types ?? []).includes("Files") || event.dataTransfer.files.length > 0;
}

function segmentAcceptsAssetKind(segment: TimelineSegment, kind: AssetReference["kind"]): boolean {
  return segment.mode === "fl2va" ? kind === "image" : true;
}

interface AssetDropNotice {
  message: string;
  transient: boolean;
}

function persistentAssetDropNotice(message: string): AssetDropNotice {
  return { message, transient: false };
}

function assetBindingNotice(
  result: AssetBindingResult,
  target: SegmentAssetTarget,
  uploadResult?: DroppedUploadResult,
): AssetDropNotice {
  const parts: string[] = [];
  if (result.accepted.length) parts.push(`已绑定 ${result.accepted.length} 项`);
  const incompatible = result.rejected.filter((item) => item.reason === "incompatible").length;
  const capacity = result.rejected.filter((item) => item.reason === "capacity");
  const duplicate = result.rejected.filter((item) => item.reason === "duplicate").length;
  if (incompatible) parts.push(`${incompatible} 项类型不兼容`);
  if (capacity.length) {
    const messages = new Set(capacity.map(({ asset }) => {
      if (result.segment.mode === "fl2va") {
        return target === "first_image" || target === "last_image"
          ? "首帧或尾帧区域一次只能绑定 1 张图片"
          : "FL2VA 最多绑定首帧和尾帧各 1 张图片";
      }
      const limits = minimaxH3ReferenceCapacities(
        result.segment.source_video !== null,
      );
      if (target === "source_video") {
        if (!result.segment.source_video && result.segment.reference_videos.length >= limits.totalReferenceVideos) {
          return `MiniMax H3 最多支持 ${limits.totalReferenceVideos} 路视频；当前独立参考视频已占满，需先移除 1 个`;
        }
        return "源视频区域一次只能绑定或替换 1 个视频";
      }
      if (asset.kind === "image") {
        return `MiniMax H3 最多支持 ${limits.referenceImages} 张参考图片`;
      }
      if (asset.kind === "audio") {
        return `MiniMax H3 最多支持 ${limits.referenceAudios} 条独立参考音频`;
      }
      return result.segment.source_video
        ? `MiniMax H3 最多支持 ${limits.totalReferenceVideos} 路视频；源视频已占 1 路，最多再添加 ${limits.referenceVideos} 路参考视频`
        : `MiniMax H3 最多支持 ${limits.totalReferenceVideos} 路参考视频`;
    }));
    parts.push([...messages].join("；"));
  }
  if (duplicate) parts.push(`${duplicate} 项已存在`);
  if (uploadResult?.failures.length) {
    parts.push(`${uploadResult.failures.length} 项上传失败：${uploadResult.failures
      .map((item) => `${item.file_name}（${item.message}）`)
      .join("、")}`);
  }
  const onlyAdded = uploadResult
    ? Math.max(0, uploadResult.assets.length - result.accepted.length)
    : 0;
  if (onlyAdded) {
    parts.push(`${onlyAdded} 项仅加入素材库`);
  }
  return {
    message: parts.join("；") || "没有可绑定的素材",
    transient: result.accepted.length > 0 && !incompatible && !capacity.length &&
      !duplicate && !uploadResult?.failures.length && !onlyAdded,
  };
}

interface SegmentGridCapacity {
  limit: number;
  full: boolean;
  bindingBlocked: boolean;
  note: string | null;
}

function segmentGridCapacity(
  segment: TimelineSegment,
  target: SegmentAssetTarget,
  currentCount: number,
): SegmentGridCapacity {
  if (segment.mode === "fl2va") {
    const full = currentCount >= 1;
    return {
      limit: 1,
      full,
      // A filled anchor remains a valid replacement target.
      bindingBlocked: false,
      note: full ? "当前锚点已占用；拖入新图片将替换现有图片。" : null,
    };
  }

  const limits = minimaxH3ReferenceCapacities(segment.source_video !== null);
  const totalVideos = segment.reference_videos.length + (segment.source_video ? 1 : 0);
  if (target === "source_video") {
    const sharedVideoCapacityFull = !segment.source_video &&
      segment.reference_videos.length >= limits.totalReferenceVideos;
    return {
      limit: limits.sourceVideo,
      full: currentCount >= limits.sourceVideo || sharedVideoCapacityFull,
      // An existing source can always be replaced; three independent videos
      // leave no shared H3 video input in which to introduce the first source.
      bindingBlocked: sharedVideoCapacityFull,
      note: sharedVideoCapacityFull
        ? `MiniMax H3 最多支持 ${limits.totalReferenceVideos} 路视频，当前参考视频已占满；移除 1 个后才能绑定源视频。本地文件仍可导入素材库。`
        : null,
    };
  }

  const limit = target === "reference_image"
    ? limits.referenceImages
    : target === "reference_audio"
      ? limits.referenceAudios
      : limits.referenceVideos;
  const full = currentCount >= limit;
  const fullSuffix = "已达上限；本地文件仍可导入素材库，但不会绑定到当前片段。";
  if (target === "reference_video") {
    return {
      limit,
      full,
      bindingBlocked: full,
      note: `与源视频共享 MiniMax H3 的 ${limits.totalReferenceVideos} 路视频容量；当前视频总计 ${totalVideos}/${limits.totalReferenceVideos}${full ? `，${fullSuffix}` : "。"}`,
    };
  }
  return {
    limit,
    full,
    bindingBlocked: full,
    note: full
      ? `MiniMax H3 最多支持 ${limit} ${target === "reference_image" ? "张参考图片" : "条独立参考音频"}，${fullSuffix}`
      : null,
  };
}

function refsForZone(
  segment: TimelineSegment,
  kind: ReferenceKind,
  target: SegmentAssetTarget,
): AssetReference[] {
  if (segment.mode === "fl2va") {
    if (target === "first_image") return segment.first_image ? [segment.first_image] : [];
    if (target === "last_image") return segment.last_image ? [segment.last_image] : [];
    return [];
  }
  if (target === "source_video") return segment.source_video ? [segment.source_video] : [];
  const references = kind === "image"
    ? segment.reference_images
    : kind === "audio"
      ? segment.reference_audios
      : segment.reference_videos;
  return references
    .filter((asset) => asset.kind === kind)
    .sort((left, right) => {
      const leftSlot = "slot" in left ? Number(left.slot) : -1;
      const rightSlot = "slot" in right ? Number(right.slot) : -1;
      return leftSlot - rightSlot;
    });
}

function clipDropZone(clientX: number, left: number, width: number): SegmentDropZone {
  if (!Number.isFinite(width) || width <= 0) return "bind";
  const ratio = (clientX - left) / width;
  if (ratio <= 0.25) return "before";
  if (ratio >= 0.75) return "after";
  return "bind";
}

function segmentAssetRole(segment: TimelineSegment, asset: AssetReference, occurrence: number): string | null {
  const token = segmentReferenceTag(segment, asset);
  if (token) return token;
  if (segment.mode === "fl2va") {
    if (segment.first_image?.id === asset.id && occurrence === 0) return "首帧";
    return "尾帧";
  }
  return null;
}

function SegmentReferenceGrid({
  title,
  kind,
  segment,
  emptyText,
  target,
  onBind,
  onDropFiles,
  runtimeEnabled,
  onRemove,
  onReorder,
  children,
  footer,
}: {
  title: string;
  kind: ReferenceKind;
  segment: TimelineSegment;
  emptyText: string;
  target: SegmentAssetTarget;
  onBind: (assetIds: string[], target: SegmentAssetTarget) => void;
  onDropFiles: (files: File[], target: SegmentAssetTarget) => void;
  runtimeEnabled: boolean;
  onRemove: (assetId: string) => void;
  onReorder?: (draggedAssetId: string, targetAssetId: string) => void;
  children?: ReactNode;
  footer?: ReactNode;
}) {
  const assets = refsForZone(segment, kind, target);
  const capacity = segmentGridCapacity(segment, target, assets.length);
  const capacityDescriptionId = `segment-reference-capacity-${segment.id}-${target}`;
  const [dropActive, setDropActive] = useState(false);
  const [draggedReferenceId, setDraggedReferenceId] = useState<string | null>(null);
  const [reorderTargetId, setReorderTargetId] = useState<string | null>(null);
  const reorderable = Boolean(onReorder && target === "reference_image");
  const assetGrid = (
    <div className="segment-reference-grid__assets">
      {assets.map((asset, index) => {
        const token = segmentAssetRole(segment, asset, index);
        return (
          <article
            key={asset.id}
            data-reference-asset-id={asset.id}
            draggable={reorderable && assets.length > 1}
            className={`${draggedReferenceId === asset.id ? "is-dragging" : ""} ${reorderTargetId === asset.id ? "is-reorder-target" : ""}`}
            onDragStart={(event) => {
              if (!reorderable) {
                event.preventDefault();
                return;
              }
              event.dataTransfer.effectAllowed = "move";
              event.dataTransfer.setData(DIRECTOR_SEGMENT_REFERENCE_MIME, JSON.stringify({
                segmentId: segment.id,
                assetId: asset.id,
                target,
              }));
              setDraggedReferenceId(asset.id);
            }}
            onDragOver={(event) => {
              if (!reorderable || !hasDirectorSegmentReferenceDrag(event)) return;
              event.preventDefault();
              event.stopPropagation();
              event.dataTransfer.dropEffect = "move";
              if (asset.id !== draggedReferenceId) setReorderTargetId(asset.id);
            }}
            onDragLeave={() => {
              if (reorderTargetId === asset.id) setReorderTargetId(null);
            }}
            onDrop={(event) => {
              if (!reorderable) return;
              const payload = directorSegmentReferenceFromTransfer(event.dataTransfer);
              if (!payload) return;
              event.preventDefault();
              event.stopPropagation();
              setDropActive(false);
              setDraggedReferenceId(null);
              setReorderTargetId(null);
              if (
                payload.segmentId === segment.id &&
                payload.target === target &&
                payload.assetId !== asset.id
              ) onReorder?.(payload.assetId, asset.id);
            }}
            onDragEnd={() => {
              setDraggedReferenceId(null);
              setReorderTargetId(null);
              setDropActive(false);
            }}
          >
            <div>{asset.kind === "image" && asset.preview_url ? <img src={asset.preview_url} alt="" draggable={false} /> : asset.kind === "video" && asset.preview_url ? <video src={asset.preview_url} muted preload="metadata" draggable={false} /> : <span aria-hidden="true">{assetGlyph(asset.kind)}</span>}</div>
            <strong title={asset.name}>{token && <em>{token}</em>}{asset.name}</strong>
            <button
              type="button"
              draggable={false}
              aria-label={`从片段移除 ${asset.name}`}
              onDragStart={(event) => {
                event.preventDefault();
                event.stopPropagation();
              }}
              onClick={() => onRemove(asset.id)}
            >×</button>
          </article>
        );
      })}
      {!assets.length && <div className="segment-reference-grid__empty"><span>＋</span><small>{emptyText}</small></div>}
    </div>
  );
  return (
    <section
      className={`segment-reference-grid ${target === "reference_image" ? "segment-reference-grid--content-sized" : ""} ${target === "source_video" ? "segment-reference-grid--source" : ""} ${capacity.full ? "is-full" : ""} ${capacity.bindingBlocked ? "is-bind-blocked" : ""} ${dropActive ? "is-drop-active" : ""}`}
      aria-label={title}
      aria-describedby={capacity.note ? capacityDescriptionId : undefined}
      onDragEnter={(event) => {
        if (hasDirectorSegmentReferenceDrag(event)) {
          event.preventDefault();
          event.stopPropagation();
          return;
        }
        if (hasSystemFileDrag(event)) {
          event.preventDefault();
          event.stopPropagation();
          if (!runtimeEnabled) return;
          setDropActive(true);
          return;
        }
        if (hasDirectorAssetDrag(event)) {
          event.preventDefault();
          setDropActive(!capacity.bindingBlocked);
        }
      }}
      onDragOver={(event) => {
        if (hasDirectorSegmentReferenceDrag(event)) {
          event.preventDefault();
          event.stopPropagation();
          event.dataTransfer.dropEffect = "move";
          return;
        }
        if (hasSystemFileDrag(event)) {
          event.preventDefault();
          event.stopPropagation();
          if (!runtimeEnabled) {
            event.dataTransfer.dropEffect = "none";
            return;
          }
          event.dataTransfer.dropEffect = "copy";
          setDropActive(true);
          return;
        }
        if (!hasDirectorAssetDrag(event)) return;
        event.preventDefault();
        event.stopPropagation();
        event.dataTransfer.dropEffect = capacity.bindingBlocked ? "none" : "copy";
        setDropActive(!capacity.bindingBlocked);
      }}
      onDragLeave={(event) => {
        if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
        setDropActive(false);
      }}
      onDrop={(event) => {
        setDropActive(false);
        if (hasDirectorSegmentReferenceDrag(event)) {
          event.preventDefault();
          event.stopPropagation();
          return;
        }
        if (hasSystemFileDrag(event)) {
          event.preventDefault();
          event.stopPropagation();
          if (!runtimeEnabled) return;
          onDropFiles(Array.from(event.dataTransfer.files), target);
          return;
        }
        const ids = dropAssetIds(event);
        if (!ids.length) return;
        event.preventDefault();
        event.stopPropagation();
        onBind(ids, target);
      }}
    >
      <header><strong>{title}</strong><span>{assets.length}/{capacity.limit}</span></header>
      {children ? (
        <div className="segment-reference-grid__source-layout">
          {assetGrid}
          <div className="segment-reference-grid__settings">{children}</div>
        </div>
      ) : assetGrid}
      {footer && <div className="segment-reference-grid__footer">{footer}</div>}
      {capacity.note && <p id={capacityDescriptionId} className="segment-reference-grid__capacity">{capacity.note}</p>}
    </section>
  );
}

function PromptReferenceButton({ asset, segment, onInsert }: { asset: AssetReference; segment: TimelineSegment; onInsert: (token: string) => void }) {
  const token = segmentReferenceTag(segment, asset);
  if (!token) return null;
  return <button type="button" onClick={() => onInsert(token)}>{token}</button>;
}

function validateTokenReferences(segment: TimelineSegment): string[] {
  const valid = new Set(segmentPromptReferenceTags(segment));
  return Array.from(segment.prompt.matchAll(/<\s*(Picture|Audio|Video)\s+(\d+)\s*>/gi), (match) => match[0]).filter((token) => {
    const match = token.match(/<\s*(Picture|Audio|Video)\s+(\d+)\s*>/i);
    if (!match) return false;
    const kind = `${match[1][0].toUpperCase()}${match[1].slice(1).toLowerCase()}`;
    const normalized = `<${kind} ${Number(match[2])}>`;
    return !valid.has(normalized);
  });
}

function segmentAudioModeLabel(mode: TimelineSegment["audio_mode"]): string {
  switch (mode) {
    case "generate": return "生成音频";
    case "source": return "保留源音频";
    case "mute": return "静音";
  }
}

function segmentCopyOptionCount(options: TimelineSegmentCopyOptions): number {
  return [
    options.mode,
    options.duration,
    options.continuity,
    options.audioMode,
    options.refImageSize,
    options.prompt,
    options.promptReferences,
  ].filter(Boolean).length;
}

function SegmentInspector({
  state,
  segment,
  capabilities,
  runtimeEnabled,
  onDispatch,
  onBindAssets,
  onDropFiles,
  dropNotice,
  uploadingFiles,
}: {
  state: TimelineEditorState;
  segment: TimelineSegment;
  capabilities: CapabilityReport;
  runtimeEnabled: boolean;
  onDispatch: (action: TimelineUserAction) => void;
  onBindAssets: (segmentId: string, assetIds: string[], target?: SegmentAssetTarget) => void;
  onDropFiles: (segmentId: string, files: File[], target?: SegmentAssetTarget) => void;
  dropNotice: AssetDropNotice | null;
  uploadingFiles: boolean;
}) {
  const promptRef = useRef<HTMLTextAreaElement>(null);
  const [promptPicker, setPromptPicker] = useState<PromptPickerState | null>(null);
  const [copyOptions, setCopyOptions] = useState(loadTimelineSegmentCopyOptions);
  const [copyOptionsOpen, setCopyOptionsOpen] = useState(false);
  const [copyFeedback, setCopyFeedback] = useState<string | null>(null);
  const copyOptionsRootRef = useRef<HTMLDivElement>(null);
  const copyOptionsButtonRef = useRef<HTMLButtonElement>(null);
  const updateCopyOptions = (patch: Partial<TimelineSegmentCopyOptions>) => {
    const next = normalizeTimelineSegmentCopyOptions({ ...copyOptions, ...patch });
    setCopyOptions(next);
    saveTimelineSegmentCopyOptions(next);
  };
  const restoreDefaultCopyOptions = () => {
    const next = { ...DEFAULT_TIMELINE_SEGMENT_COPY_OPTIONS };
    setCopyOptions(next);
    saveTimelineSegmentCopyOptions(next);
  };
  useEffect(() => {
    if (!copyOptionsOpen) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (event.target instanceof Node && !copyOptionsRootRef.current?.contains(event.target)) {
        setCopyOptionsOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      setCopyOptionsOpen(false);
      copyOptionsButtonRef.current?.focus();
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer, true);
    document.addEventListener("keydown", closeOnEscape, true);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePointer, true);
      document.removeEventListener("keydown", closeOnEscape, true);
    };
  }, [copyOptionsOpen]);
  useEffect(() => {
    setCopyOptionsOpen(false);
    setCopyFeedback(null);
  }, [segment.id]);
  useEffect(() => {
    if (!copyFeedback) return;
    const timer = window.setTimeout(() => setCopyFeedback(null), 3_000);
    return () => window.clearTimeout(timer);
  }, [copyFeedback]);
  const updateBase = (patch: Partial<Pick<TimelineSegment, "title" | "prompt" | "duration_seconds" | "enabled" | "audio_mode" | "ref_image_size">>) => onDispatch({
    type: "segment/patch-base",
    id: segment.id,
    patch,
  });
  const insertToken = (token: string) => {
    setPromptPicker(null);
    const textarea = promptRef.current;
    const start = textarea?.selectionStart ?? segment.prompt.length;
    const end = textarea?.selectionEnd ?? start;
    const spacer = start > 0 && !/\s$/.test(segment.prompt.slice(0, start)) ? " " : "";
    const next = limitPromptCharacters(
      `${segment.prompt.slice(0, start)}${spacer}${token} ${segment.prompt.slice(end)}`,
    );
    updateBase({ prompt: next });
    window.requestAnimationFrame(() => {
      const cursor = start + spacer.length + token.length + 1;
      textarea?.focus();
      textarea?.setSelectionRange(cursor, cursor);
    });
  };
  const remove = (assetId: string) => onDispatch({
    type: "segment/remove-asset",
    id: segment.id,
    assetId,
  });
  const reorderReferenceImage = (draggedAssetId: string, targetAssetId: string) => onDispatch({
    type: "segment/reorder-reference",
    id: segment.id,
    kind: "image",
    draggedAssetId,
    targetAssetId,
  });
  const unresolved = validateTokenReferences(segment);
  const recipe = deriveSegmentRecipe(segment);
  const sourceIndex = state.project.segments.findIndex((candidate) => candidate.id === segment.id);
  const followingCount = Math.max(0, state.project.segments.length - sourceIndex - 1);
  const selectedTargetCount = state.selected_segment_ids.filter((id) =>
    id !== segment.id && state.project.segments.some((candidate) => candidate.id === id),
  ).length;
  const copyOptionCount = segmentCopyOptionCount(copyOptions);
  const referenceCount = segmentAssetReferences(segment).length;
  const applyConfiguration = (scope: "following" | "selected", targetCount: number) => {
    if (!targetCount || !copyOptionCount) return;
    onDispatch({
      type: "segment/apply-config",
      sourceId: segment.id,
      scope,
      options: copyOptions,
    });
    setCopyFeedback(`已向 ${targetCount} 个片段应用 ${copyOptionCount} 项设置`);
  };
  const continuityBoundary = timelineContinuityBoundaries(state.project)
    .find((boundary) => boundary.segment.id === segment.id) ?? null;
  const nativeTimeline = capabilities.native_timeline;
  const nativeContinuitySupported = capabilities.connection === "online" &&
    nativeTimeline?.supported === true &&
    nativeTimeline.continuity === true &&
    nativeTimeline.modes.includes(segment.mode);
  const continuityCanEnable = segment.enabled &&
    nativeContinuitySupported &&
    continuityBoundary?.kind === "eligible";
  const continuityParameterIssue = timelineContinuityRunIssues(state.project)
    .find((issue) => issue.boundary.segment.id === segment.id &&
      (issue.code === "predecessor-too-short" || issue.code === "sample-too-long"));
  const continuityHelpId = `segment-continuity-help-${segment.id}`;
  const durationHelpId = `segment-duration-help-${segment.id}`;
  const durationLockedBySource = segment.mode === "ref2va" && Boolean(segment.source_video);
  const sourceCropFrames = segment.mode === "ref2va" && segment.source_video && segment.audio_mode === "source"
    ? sourceTrimFrameCount(segment, state.project.render.fps)
    : null;
  const sourceCropSummary = segment.mode === "ref2va" && segment.source_video?.metadata && sourceCropFrames !== null
    ? `素材总长${segment.source_video.metadata.duration.toFixed(2)}秒，共${segment.source_video.metadata.frame_count}帧，为满足H3约束，Director自动裁剪到${segment.source_duration_seconds.toFixed(4)}秒，${sourceCropFrames}帧`
    : null;
  const continuityHelp = !nativeContinuitySupported
    ? "当前原生分段子图不支持这个片段的连续性"
    : !segment.enabled
      ? "片段重新启用后才可读取前段尾帧"
      : !continuityBoundary
        ? "当前是第一个启用片段，没有可读取的前段"
        : continuityBoundary.kind === "explicit-first-image"
          ? "显式首帧会形成画面锚点，不能同时自动读取前段尾帧"
          : continuityParameterIssue?.message ?? `生成时读取前一启用片段“${continuityBoundary.predecessor.title}”的最后 ${segment.continuity.overlap_frames} 帧；导出时会裁掉引导帧`;
  const introducedReferences: PromptAssetSuggestion[] = segmentAssetReferences(segment)
    .flatMap((asset, index) => {
      const token = segmentReferenceTag(segment, asset);
      if (!token) return [];
      const sourceVideo = segment.mode === "ref2va" && segment.source_video?.id === asset.id;
      return [{
        type: "asset" as const,
        key: `asset-${asset.id}-${token}-${index}`,
        name: asset.name,
        kind: asset.kind,
        token,
        role: sourceVideo ? `${token} · 源视频` : token,
        previewUrl: asset.preview_url ?? null,
      }];
    });
  if (segment.mode === "ref2va" && segment.source_audio_as_reference) {
    introducedReferences.push({
      type: "asset",
      key: "source-audio",
      name: segment.source_video ? `${segment.source_video.name} · 源音轨` : "源视频音轨",
      kind: "audio",
      token: "<Audio 1>",
      role: "<Audio 1> · 源音轨",
      previewUrl: null,
    });
  }
  const definedSubjects: PromptSubjectSuggestion[] = promptSubjectReferences(segment.prompt)
    .map((subject) => ({
      type: "subject",
      key: `subject-${subject.number}-${subject.token}`,
      token: subject.token,
      definition: subject.definition,
    }));
  const pickerSuggestions: PromptSuggestion[] = promptPicker
    ? (promptPicker.kind === "asset" ? introducedReferences : definedSubjects).filter((suggestion) => {
        const query = promptPicker.query.toLocaleLowerCase();
        if (!query) return true;
        if (suggestion.type === "asset") {
          return suggestion.name.toLocaleLowerCase().includes(query) ||
            suggestion.token.toLocaleLowerCase().includes(query);
        }
        return suggestion.token.toLocaleLowerCase().includes(query) ||
          suggestion.definition.toLocaleLowerCase().includes(query);
      })
    : [];
  const introducedReferenceSignature = introducedReferences
    .map((reference) => `${reference.key}:${reference.token}`)
    .join("|");
  const definedSubjectSignature = definedSubjects
    .map((subject) => subject.key)
    .join("|");
  const activePickerIndex = promptPicker && pickerSuggestions.length
    ? Math.min(promptPicker.activeIndex, pickerSuggestions.length - 1)
    : -1;
  const activePickerOptionId = activePickerIndex >= 0
    ? `prompt-mentions-${segment.id}-option-${activePickerIndex}`
    : undefined;
  const choosePickerSuggestion = (suggestion: PromptSuggestion) => {
    if (!promptPicker) return;
    const expectedMention = `${promptPicker.trigger}${promptPicker.query}`;
    if (segment.prompt.slice(promptPicker.start, promptPicker.end) !== expectedMention) {
      setPromptPicker(null);
      return;
    }
    onDispatch({
      type: suggestion.type === "asset"
        ? "segment/insert-reference-token"
        : "segment/insert-subject-token",
      id: segment.id,
      token: suggestion.token,
      selectionStart: promptPicker.start,
      selectionEnd: promptPicker.end,
      expectedMention,
    });
    setPromptPicker(null);
    window.requestAnimationFrame(() => {
      const needsLeadingSpace = promptPicker.start > 0 && !/\s$/.test(segment.prompt.slice(0, promptPicker.start));
      const cursor = promptPicker.start + (needsLeadingSpace ? 1 : 0) + suggestion.token.length;
      promptRef.current?.focus();
      promptRef.current?.setSelectionRange(cursor, cursor);
    });
  };
  const updatePromptPicker = (textarea: HTMLTextAreaElement, prompt: string, caret: number) => {
    const match = prompt.slice(0, caret).match(/([@#])([^\s@#<>]*)$/);
    if (!match) {
      setPromptPicker(null);
      return;
    }
    const trigger = match[1] as "@" | "#";
    const start = caret - match[0].length;
    setPromptPicker({
      kind: trigger === "@" ? "subject" : "asset",
      trigger,
      start,
      end: caret,
      query: match[2],
      activeIndex: 0,
      position: promptPickerPosition(textarea, start),
    });
  };
  useEffect(() => {
    setPromptPicker(null);
  }, [segment.id, segment.mode, introducedReferenceSignature, definedSubjectSignature]);
  useEffect(() => {
    if (!promptPicker) return;
    const reposition = () => {
      const textarea = promptRef.current;
      if (!textarea) return;
      setPromptPicker((current) => current
        ? { ...current, position: promptPickerPosition(textarea, current.start) }
        : current);
    };
    window.addEventListener("resize", reposition);
    window.addEventListener("scroll", reposition, true);
    return () => {
      window.removeEventListener("resize", reposition);
      window.removeEventListener("scroll", reposition, true);
    };
  }, [promptPicker?.start]);

  return (
    <section
      className={`segment-inspector segment-inspector--${segment.mode}`}
      aria-label="当前片段编辑器"
      aria-busy={uploadingFiles}
      onDragOver={(event) => {
        if (hasSystemFileDrag(event)) {
          event.preventDefault();
          event.dataTransfer.dropEffect = runtimeEnabled ? "copy" : "none";
          return;
        }
        if (hasDirectorAssetDrag(event)) {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }
      }}
      onDrop={(event) => {
        if (hasSystemFileDrag(event)) {
          event.preventDefault();
          if (!runtimeEnabled) return;
          onDropFiles(segment.id, Array.from(event.dataTransfer.files));
          return;
        }
        const ids = dropAssetIds(event);
        if (!ids.length) return;
        event.preventDefault();
        onBindAssets(segment.id, ids);
      }}
    >
      <header className="segment-inspector__controlbar">
        <h2 className="segment-inspector__title">片段编辑</h2>
        <Field label="片段名称" className="field--inline segment-inspector__name"><input data-timeline-history-field={`segment:${segment.id}:title`} maxLength={256} value={segment.title} onChange={(event) => updateBase({ title: event.target.value })} /></Field>
        <span className="segment-inspector__divider" aria-hidden="true" />
        <div className="segment-inspector__continuity" role="group" aria-label="当前片段连续性" aria-describedby={continuityHelpId} title={continuityHelp}>
          <label className="toggle"><input aria-label="启用当前片段连续性" type="checkbox" disabled={!segment.continuity.enabled && !continuityCanEnable} checked={segment.continuity.enabled} onChange={(event) => onDispatch({ type: "segment/set-continuity", id: segment.id, patch: { enabled: event.target.checked } })} /><span />连续性</label>
          <label className="segment-inspector__continuity-frames"><span>接续帧</span><select aria-label="当前片段接续尾帧数" aria-invalid={Boolean(continuityParameterIssue) || undefined} disabled={!segment.continuity.enabled} value={segment.continuity.overlap_frames} onChange={(event) => onDispatch({ type: "segment/set-continuity", id: segment.id, patch: { overlap_frames: Number(event.target.value) as TimelineSegment["continuity"]["overlap_frames"] } })}>{[5, 22, 39, 56].map((frames) => <option key={frames} value={frames}>{frames}</option>)}</select></label>
          <span id={continuityHelpId} className="sr-only">{continuityHelp}</span>
        </div>
        <Field label="生成模式" className="field--inline segment-inspector__mode"><select aria-label="片段生成模式" value={segment.mode} onChange={(event) => onDispatch({ type: "segment/set-mode", ids: [segment.id], mode: event.target.value as TimelineGenerationMode })}>{TIMELINE_MODE_ORDER.map((mode) => <option key={mode} value={mode}>{segmentModeLabel(mode)}</option>)}</select></Field>
        <Field label="生成时长（秒）" className="field--inline segment-inspector__duration"><DeferredNumberInput aria-label="生成时长（秒）" min={recipe === "fl2v" ? 0.1 : 0.01} max={durationLockedBySource ? 86_400 : 120} step="0.01" value={segment.duration_seconds} disabled={durationLockedBySource} aria-describedby={durationLockedBySource ? durationHelpId : undefined} title={durationLockedBySource ? "已绑定源视频；生成时长由源视频片段确定" : undefined} onValueCommit={(value) => updateBase({ duration_seconds: value })} />{durationLockedBySource && <small id={durationHelpId} className="sr-only">已绑定源视频；请通过下方源视频裁剪范围调整片段时长。</small>}</Field>
        <Field label="音频策略" className="field--inline segment-inspector__audio-mode"><select aria-label="音频策略" value={segment.audio_mode} onChange={(event) => updateBase({ audio_mode: event.target.value as TimelineSegment["audio_mode"] })}><option value="generate">生成音频</option><option value="source">保留源音频</option><option value="mute">静音</option></select></Field>
        <Field label="参考图采样尺寸" className="field--inline segment-inspector__ref-image-size"><select aria-label="参考图采样尺寸" value={segment.ref_image_size} onChange={(event) => updateBase({ ref_image_size: event.target.value as TimelineSegment["ref_image_size"] })}><option value="match">match（匹配画布）</option><option value="max">max（最高保真）</option></select></Field>
        <div className="segment-inspector__head-actions">
          <span className="segment-copy-source" title={`当前从“${segment.title}”复制设置`}>复制来源：<strong>{segment.title}</strong></span>
          <button type="button" disabled={followingCount === 0 || copyOptionCount === 0} onClick={() => applyConfiguration("following", followingCount)}>应用到后续</button>
          <button type="button" disabled={selectedTargetCount === 0 || copyOptionCount === 0} onClick={() => applyConfiguration("selected", selectedTargetCount)}>应用到所选（{selectedTargetCount}）</button>
          <div
            ref={copyOptionsRootRef}
            className="segment-copy-options"
            onBlur={(event) => {
              if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) {
                setCopyOptionsOpen(false);
              }
            }}
          >
            <button
              ref={copyOptionsButtonRef}
              className="segment-copy-options__trigger"
              type="button"
              aria-label="复制设置"
              aria-expanded={copyOptionsOpen}
              aria-controls="segment-copy-options-popover"
              onClick={() => setCopyOptionsOpen((open) => !open)}
            ><span>复制 {copyOptionCount} 项</span><span aria-hidden="true">▴</span></button>
            {copyOptionsOpen && <section id="segment-copy-options-popover" className="segment-copy-options__popover" aria-label="复制设置">
              <header><strong>从“{segment.title}”复制设置</strong><small>当前编辑片段是复制来源；同时用于两个应用按钮</small></header>
              <div className="segment-copy-options__fields">
                <label><input type="checkbox" checked={copyOptions.mode} onChange={(event) => updateCopyOptions({ mode: event.target.checked })} /><span><strong>生成模式</strong><small>{segmentModeLabel(segment.mode)}；跨模式会重置目标素材</small></span></label>
                <label><input type="checkbox" checked={copyOptions.continuity} onChange={(event) => updateCopyOptions({ continuity: event.target.checked })} /><span><strong>连续性</strong><small>{segment.continuity.enabled ? `开启 · ${segment.continuity.overlap_frames} 帧` : "关闭"}</small></span></label>
                <label><input type="checkbox" checked={copyOptions.duration} onChange={(event) => updateCopyOptions({ duration: event.target.checked })} /><span><strong>生成时长</strong><small>{segment.duration_seconds.toFixed(2)} 秒</small></span></label>
                <label><input type="checkbox" checked={copyOptions.audioMode} onChange={(event) => updateCopyOptions({ audioMode: event.target.checked })} /><span><strong>音频策略</strong><small>{segmentAudioModeLabel(segment.audio_mode)}</small></span></label>
                <label><input type="checkbox" checked={copyOptions.refImageSize} onChange={(event) => updateCopyOptions({ refImageSize: event.target.checked })} /><span><strong>参考图采样尺寸</strong><small>{segment.ref_image_size === "match" ? "match（匹配画布）" : "max（最高保真）"}</small></span></label>
                <label><input type="checkbox" checked={copyOptions.prompt} onChange={(event) => updateCopyOptions({ prompt: event.target.checked })} /><span><strong>提示词</strong><small>{promptCharacterCount(segment.prompt).toLocaleString()} 个字符</small></span></label>
                <label className="segment-copy-options__nested"><input type="checkbox" disabled={!copyOptions.prompt || !copyOptions.mode} checked={copyOptions.promptReferences} onChange={(event) => updateCopyOptions({ promptReferences: event.target.checked })} /><span><strong>连同引用素材</strong><small>{referenceCount} 项；需同时复制提示词和生成模式</small></span></label>
              </div>
              <footer><button type="button" onClick={restoreDefaultCopyOptions}>恢复默认</button></footer>
            </section>}
          </div>
        </div>
        {copyFeedback && <span className="segment-copy-feedback" role="status" aria-live="polite">{copyFeedback}</span>}
        <span className="segment-inspector__divider" aria-hidden="true" />
        <div className="segment-inspector__timing"><small>项目入点</small><div><strong>{formatClock(segmentStartTime(state.project, segment.id))}</strong><span>{segment.mode === "ref2va" && segment.source_video && alignH3Frames(segment.duration_seconds, state.project.render.fps) > 512 ? `源片时间线 ${segment.duration_seconds.toFixed(2)}s · 待分割` : `请求 ${segment.duration_seconds.toFixed(2)}s → 实际 ${alignedTimelineSegmentDuration(segment, state.project.render.fps).toFixed(4)}s · ${alignH3Frames(segment.duration_seconds, state.project.render.fps)}f`}</span></div></div>
      </header>

      <div className="segment-inspector__refs">
        {segment.mode === "fl2va" ? <>
          <SegmentReferenceGrid title="首帧（可选）" kind="image" target="first_image" segment={segment} emptyText="从素材库拖入，或直接拖入本地图片" onBind={(ids, target) => onBindAssets(segment.id, ids, target)} onDropFiles={(files, target) => onDropFiles(segment.id, files, target)} runtimeEnabled={runtimeEnabled} onRemove={remove} />
          <SegmentReferenceGrid title="尾帧（设置后使用首尾帧配方）" kind="image" target="last_image" segment={segment} emptyText="从素材库拖入，或直接拖入本地图片" onBind={(ids, target) => onBindAssets(segment.id, ids, target)} onDropFiles={(files, target) => onDropFiles(segment.id, files, target)} runtimeEnabled={runtimeEnabled} onRemove={remove} />
        </> : <>
          <SegmentReferenceGrid
            title="源视频（可选），占用 <Video 1>"
            kind="video"
            target="source_video"
            segment={segment}
            emptyText="从素材库拖入，或直接拖入本地源视频"
            onBind={(ids, target) => onBindAssets(segment.id, ids, target)}
            onDropFiles={(files, target) => onDropFiles(segment.id, files, target)}
            runtimeEnabled={runtimeEnabled}
            onRemove={remove}
            footer={sourceCropSummary ? (
              <small className="segment-source-crop-summary">{sourceCropSummary}</small>
            ) : null}
          >
            {segment.source_video && (
              <div className="segment-source-range">
                <Field label="源视频入点（秒）" className="segment-source-range__field"><DeferredNumberInput min="0" max={segment.source_video.metadata?.duration ?? 86_400} step="0.01" value={segment.source_start_seconds} onValueCommit={(value) => onDispatch({ type: "segment/set-source-range", id: segment.id, patch: { source_start_seconds: value } })} /></Field>
                <Field label="源截取时长（秒）" className="segment-source-range__field"><DeferredNumberInput min="0.01" max={segment.source_video.metadata ? Math.max(0.01, segment.source_video.metadata.duration - segment.source_start_seconds) : 86_400} step="0.01" value={segment.source_duration_seconds} onValueCommit={(value) => onDispatch({ type: "segment/set-source-range", id: segment.id, patch: { source_duration_seconds: value } })} /></Field>
                <label className="source-audio-reference-toggle">
                  <input
                    aria-label="参考源视频音轨"
                    type="checkbox"
                    checked={segment.source_audio_as_reference}
                    disabled={segment.source_video.metadata?.has_audio !== true && !segment.source_audio_as_reference}
                    onChange={(event) => onDispatch({
                      type: "segment/set-source-audio-reference",
                      id: segment.id,
                      enabled: event.target.checked,
                    })}
                  />
                  <strong>参考源视频音轨</strong>
                </label>
              </div>
            )}
          </SegmentReferenceGrid>
          <SegmentReferenceGrid title="参考图片" kind="image" target="reference_image" segment={segment} emptyText="从素材库拖入，或直接拖入本地图片" onBind={(ids, target) => onBindAssets(segment.id, ids, target)} onDropFiles={(files, target) => onDropFiles(segment.id, files, target)} runtimeEnabled={runtimeEnabled} onRemove={remove} onReorder={reorderReferenceImage} />
          <SegmentReferenceGrid title="参考视频" kind="video" target="reference_video" segment={segment} emptyText="从素材库拖入，或直接拖入本地视频" onBind={(ids, target) => onBindAssets(segment.id, ids, target)} onDropFiles={(files, target) => onDropFiles(segment.id, files, target)} runtimeEnabled={runtimeEnabled} onRemove={remove} />
          <SegmentReferenceGrid title="参考音频" kind="audio" target="reference_audio" segment={segment} emptyText="从素材库拖入，或直接拖入本地音频" onBind={(ids, target) => onBindAssets(segment.id, ids, target)} onDropFiles={(files, target) => onDropFiles(segment.id, files, target)} runtimeEnabled={runtimeEnabled} onRemove={remove} />
        </>}
        {uploadingFiles && <p className="timeline-drop-notice" role="status">正在上传并分类绑定素材…</p>}
        {!uploadingFiles && dropNotice && <p className="timeline-drop-notice" role="status">{dropNotice.message}</p>}
      </div>

      <div className="segment-prompt-editor">
          <header>
            <div className="segment-prompt-title">
              <strong>片段提示词</strong>
              <button
                type="button"
                className="segment-prompt-skeleton"
                onClick={() => updateBase({ prompt: promptSkeleton(segment) })}
                disabled={segment.prompt.trim() !== ""}
                title={segment.prompt.trim() ? "提示词非空时不可填入框架" : "按当前模式填入提示词框架"}
              >填入框架</button>
            </div>
            <div>{segmentAssetReferences(segment).map((asset, index) => <PromptReferenceButton key={`${asset.id}-${segmentReferenceTag(segment, asset)}-${index}`} asset={asset} segment={segment} onInsert={insertToken} />)}{segment.mode === "ref2va" && segment.source_audio_as_reference && <button type="button" aria-label="插入源音轨引用 Audio 1" onClick={() => insertToken("<Audio 1>")}>&lt;Audio 1&gt; 源音轨</button>}</div>
          </header>
        <textarea
          ref={promptRef}
          data-timeline-history-field={`segment:${segment.id}:prompt`}
          aria-label="片段提示词"
          aria-autocomplete="list"
          aria-expanded={promptPicker !== null}
          aria-controls={promptPicker ? `prompt-mentions-${segment.id}` : undefined}
          aria-activedescendant={activePickerOptionId}
          value={segment.prompt}
          placeholder="描述动作、构图、运镜与声音；输入 @ 选择主体，输入 # 选择当前片段已引入的素材…"
          onChange={(event) => {
            const prompt = limitPromptCharacters(event.target.value);
            const caret = event.target.selectionStart;
            updateBase({ prompt });
            updatePromptPicker(event.target, prompt, caret);
          }}
          onSelect={(event) => updatePromptPicker(
            event.currentTarget,
            event.currentTarget.value,
            event.currentTarget.selectionStart,
          )}
          onBlur={() => setPromptPicker(null)}
          onKeyDown={(event) => {
            if (!promptPicker) return;
            if (event.key === "Escape") { event.preventDefault(); setPromptPicker(null); return; }
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
              event.preventDefault();
              if (!pickerSuggestions.length) return;
              const delta = event.key === "ArrowDown" ? 1 : -1;
              setPromptPicker({ ...promptPicker, activeIndex: (activePickerIndex + delta + pickerSuggestions.length) % pickerSuggestions.length });
              return;
            }
            if (event.key === "Enter" && pickerSuggestions.length) {
              event.preventDefault();
              choosePickerSuggestion(pickerSuggestions[activePickerIndex]);
            }
          }}
        />
        {promptPicker && createPortal(<div
          id={`prompt-mentions-${segment.id}`}
          className={`prompt-mention-picker prompt-mention-picker--${promptPicker.kind}`}
          role="listbox"
          aria-label={promptPicker.kind === "asset" ? "当前片段已引入的参考素材" : "提示词中已定义的主体"}
          style={{ left: promptPicker.position.left, top: promptPicker.position.top }}
        >
          {pickerSuggestions.map((suggestion, index) => <button
            className={`prompt-mention-picker__option prompt-mention-picker__option--${suggestion.type}${suggestion.type === "asset" ? ` is-${suggestion.kind}` : ""}`}
            key={suggestion.key}
            id={`prompt-mentions-${segment.id}-option-${index}`}
            type="button"
            role="option"
            aria-selected={index === activePickerIndex}
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => choosePickerSuggestion(suggestion)}
          >
            {suggestion.type === "asset" && suggestion.kind !== "audio" && <span className="prompt-mention-picker__preview" aria-hidden="true">
              {suggestion.previewUrl
                ? suggestion.kind === "image"
                  ? <img src={suggestion.previewUrl} alt="" draggable={false} />
                  : <video src={suggestion.previewUrl} muted playsInline preload="metadata" tabIndex={-1} />
                : <span>{assetGlyph(suggestion.kind)}</span>}
            </span>}
            <span className="prompt-mention-picker__copy">
              <strong>{suggestion.type === "asset" ? suggestion.name : suggestion.token}</strong>
              <small>{suggestion.type === "asset"
                ? suggestion.role
                : suggestion.definition || "未填写主体定义"}</small>
            </span>
          </button>)}
          {!pickerSuggestions.length && <p>{promptPicker.kind === "asset"
            ? introducedReferences.length
              ? "没有匹配的已引入参考素材"
              : "请先在当前片段的参考素材区引入素材"
            : definedSubjects.length
              ? "没有匹配的已定义主体"
              : "请先在 subject_definitions 中定义主体"}</p>}
        </div>, document.body)}
        <footer><span>{promptCharacterCount(segment.prompt).toLocaleString()} / {MINIMAX_H3_PROMPT_MAX_CHARACTERS.toLocaleString()}</span><small>{segment.prompt.trim() ? "使用片段提示词" : "启用片段必须填写提示词"}</small></footer>
        {unresolved.length > 0 && <p className="inline-error" role="alert">未绑定的素材标签：{[...new Set(unresolved)].join("、")}</p>}
      </div>
    </section>
  );
}

export function LongFormTimelineWorkspace({
  state,
  capabilities,
  activeTask,
  segmentCandidates,
  compileReport,
  selectionValidationErrors,
  onDispatch,
  onCloseCompile,
  onCancelTask,
  onUploadFiles,
}: LongFormTimelineWorkspaceProps) {
  const [initialPreferences] = useState(loadTimelineWorkspacePreferences);
  const [loop, setLoop] = useState(initialPreferences.loop);
  const [volume, setVolume] = useState(initialPreferences.volume);
  const [playing, setPlaying] = useState(false);
  const [playbackRestartToken, setPlaybackRestartToken] = useState(0);
  const [showLiveMonitor, setShowLiveMonitor] = useState(initialPreferences.showLiveMonitor);
  const [compareOriginal, setCompareOriginal] = useState(initialPreferences.compareOriginal);
  const [programMediaStatus, setProgramMediaStatus] = useState<"idle" | "loading" | "ready" | "seeking" | "error">("idle");
  const [originalMediaStatus, setOriginalMediaStatus] = useState<"idle" | "loading" | "ready" | "seeking" | "error">("idle");
  const [draggedSegmentId, setDraggedSegmentId] = useState<string | null>(null);
  const [assetDropTarget, setAssetDropTarget] = useState<{ id: string; zone: SegmentDropZone } | null>(null);
  const [assetDropNotice, setAssetDropNotice] = useState<AssetDropNotice | null>(null);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [timelineZoom, setTimelineZoom] = useState(initialPreferences.timelineZoom);
  const [evenSplitPieces, setEvenSplitPieces] = useState(initialPreferences.evenSplitPieces);
  const [detectionSensitivity, setDetectionSensitivity] = useState<RV2VShotDetectionRequest["sensitivity"]>(initialPreferences.detectionSensitivity);
  const [minimumShotFrames, setMinimumShotFrames] = useState(initialPreferences.minimumShotFrames);
  const [detectingShots, setDetectingShots] = useState(false);
  const [timelineToolNotice, setTimelineToolNotice] = useState<string | null>(null);
  const [livePreviewToken, setLivePreviewToken] = useState(0);
  const assetDropTargetRef = useRef<{ id: string; zone: SegmentDropZone } | null>(null);
  const detectionRequestRef = useRef(0);
  const timelineViewportRef = useRef<HTMLDivElement>(null);
  const timelineCanvasRef = useRef<HTMLDivElement>(null);
  const timelineScrubbingRef = useRef(false);
  const suppressSegmentClickRef = useRef<string | null>(null);
  const appendVideoInputRef = useRef<HTMLInputElement>(null);
  const selectAllCheckboxRef = useRef<HTMLInputElement>(null);
  const latestStateRef = useRef(state);
  latestStateRef.current = state;
  useEffect(() => {
    if (!assetDropNotice?.transient) return;
    const notice = assetDropNotice;
    const timer = window.setTimeout(() => {
      setAssetDropNotice((current) => current === notice ? null : current);
    }, 3_000);
    return () => window.clearTimeout(timer);
  }, [assetDropNotice]);
  const programVideoRef = useRef<HTMLVideoElement>(null);
  const originalVideoRef = useRef<HTMLVideoElement>(null);
  const programClockRef = useRef<{
    hasMappedMedia: boolean;
    candidate: boolean;
    segmentStart: number;
    segmentDuration: number;
    sourceStart: number;
    sourceRate: number;
  } | null>(null);

  useEffect(() => {
    updateTimelineWorkspacePreferences({
      showLiveMonitor,
      volume,
      loop,
      compareOriginal,
      timelineZoom,
      evenSplitPieces,
      detectionSensitivity,
      minimumShotFrames,
    });
  }, [
    showLiveMonitor,
    volume,
    loop,
    compareOriginal,
    timelineZoom,
    evenSplitPieces,
    detectionSensitivity,
    minimumShotFrames,
  ]);
  const playheadRef = useRef(state.playhead_seconds);
  playheadRef.current = state.playhead_seconds;
  const explicitPreview = state.assets.find((asset) => state.selected_asset_ids.includes(asset.id)) ?? null;
  const timelinePosition = useMemo(
    () => timelineSegmentAt(state.project, state.playhead_seconds),
    [state.project, state.playhead_seconds],
  );
  const timelineSource = timelinePosition?.segment.mode === "ref2va"
    ? timelinePosition.segment.source_video
    : null;
  const showingTimeline = playing || !explicitPreview;
  const comparisonAvailable = state.project.segments.some((segment) =>
    segment.enabled && segment.mode === "ref2va" && Boolean(segment.source_video?.preview_url));
  const comparisonActive = compareOriginal && showingTimeline && comparisonAvailable;
  const segmentCandidate = showingTimeline && timelinePosition
    ? segmentCandidates[timelinePosition.segment.id] ?? null
    : null;
  const preview = showingTimeline && !segmentCandidate ? timelineSource : !showingTimeline ? explicitPreview : null;
  const previewIdentity = segmentCandidate
    ? `${segmentCandidate.job_id}:${segmentCandidate.result.segment_id}:${segmentCandidate.result.output_url}`
    : preview?.id;
  const programMediaIdentity = segmentCandidate
    ? previewIdentity
    : !comparisonActive && preview?.kind === "video" && preview.preview_url
      ? `source:${preview.id}:${preview.preview_url}`
      : null;
  const sourceSegment = timelinePosition?.segment.mode === "ref2va" && timelinePosition.segment.source_video
    ? timelinePosition.segment
    : null;
  const mappedProgramTime = segmentCandidate && timelinePosition
    ? timelinePosition.local_seconds
    : showingTimeline && timelinePosition && timelineSource && sourceSegment
      ? sourcePreviewTime(sourceSegment, timelinePosition.local_seconds, state.project.render.fps)
      : null;
  const mappedOriginalTime = showingTimeline && timelinePosition && sourceSegment
    ? sourcePreviewTime(sourceSegment, timelinePosition.local_seconds, state.project.render.fps)
    : null;
  const sourceOutputDuration = sourceSegment
    ? timelineSegmentPlaybackDuration(sourceSegment, state.project.render.fps)
    : 0;
  const sourcePlaybackRate = segmentCandidate
    ? 1
    : showingTimeline && sourceSegment && sourceOutputDuration > 0
      ? sourceSegment.source_duration_seconds / sourceOutputDuration
      : 1;
  const originalPlaybackRate = showingTimeline && sourceSegment && sourceOutputDuration > 0
    ? sourceSegment.source_duration_seconds / sourceOutputDuration
    : 1;
  const originalPreviewIdentity = sourceSegment?.source_video
    ? `${sourceSegment.id}:${sourceSegment.source_video.id}:${sourceSegment.source_start_seconds}:${sourceSegment.source_duration_seconds}`
    : null;
  const selectedSegments = state.project.segments.filter((segment) => state.selected_segment_ids.includes(segment.id));
  const runnableSelection = runnableTimelineSegmentIds(state);
  const runnableSelected = new Set(runnableSelection);
  const continuityBoundaries = timelineContinuityBoundaries(state.project);
  const continuityBoundaryByTarget = new Map(
    continuityBoundaries.map((boundary) => [boundary.segment.id, boundary]),
  );
  const continuityRunIssues = timelineContinuityRunIssues(state.project, runnableSelection);
  const continuityIssueByTarget = new Map<string, (typeof continuityRunIssues)[number]>();
  continuityRunIssues.forEach((issue) => {
    const previous = continuityIssueByTarget.get(issue.boundary.segment.id);
    if (!previous || previous.code === "historical-take-required") {
      continuityIssueByTarget.set(issue.boundary.segment.id, issue);
    }
  });
  const historicalContinuityRequests = continuityRunIssues.filter((issue) => issue.code === "historical-take-required");
  const parameterContinuityIssues = continuityRunIssues.filter((issue) =>
    issue.code === "predecessor-too-short" || issue.code === "sample-too-long",
  );
  const blockingContinuityIssueCount = parameterContinuityIssues.length;
  const configuredContinuityCount = state.project.segments.filter((segment) =>
    segment.enabled && segment.continuity.enabled,
  ).length;
  const continuityBlockedStatus = [
    historicalContinuityRequests.length
      ? `复用前驱成片 ${historicalContinuityRequests.length} 段`
      : "",
    parameterContinuityIssues.length
      ? `${parameterContinuityIssues.length} 个接续参数问题`
      : "",
  ].filter(Boolean).join(" · ");
  const activeContinuityBoundaryCount = continuityBoundaries.filter((boundary) =>
    boundary.kind === "eligible" &&
    boundary.segment.continuity.enabled &&
    runnableSelected.has(boundary.predecessor.id) &&
    runnableSelected.has(boundary.segment.id),
  ).length;
  const activeSegment = state.active_segment_id
    ? state.project.segments.find((segment) => segment.id === state.active_segment_id) ?? null
    : null;
  const total = timelineDuration(state.project);
  const projectFps = Number.isFinite(state.project.render.fps) && state.project.render.fps > 0
    ? state.project.render.fps
    : 24;
  const frameStep = 1 / projectFps;
  const totalFrames = Math.max(0, Math.round(total * projectFps));
  const currentFrame = totalFrames > 0
    ? Math.min(totalFrames, Math.max(1, Math.floor(state.playhead_seconds * projectFps + 1e-7) + 1))
    : 0;
  const projectWidth = Number.isFinite(state.project.render.width) && state.project.render.width > 0
    ? state.project.render.width
    : 16;
  const projectHeight = Number.isFinite(state.project.render.height) && state.project.render.height > 0
    ? state.project.render.height
    : 9;
  const projectAspectStyle = {
    "--monitor-project-aspect": projectWidth / projectHeight,
    aspectRatio: `${projectWidth} / ${projectHeight}`,
  } as CSSProperties;
  programClockRef.current = timelinePosition ? {
    hasMappedMedia: mappedProgramTime !== null,
    candidate: Boolean(segmentCandidate),
    segmentStart: timelinePosition.start_seconds,
    segmentDuration: effectiveTimelineSegmentDuration(timelinePosition.segment, projectFps),
    sourceStart: sourceSegment?.source_start_seconds ?? 0,
    sourceRate: Math.min(16, Math.max(0.0625, sourcePlaybackRate)),
  } : null;
  const progress = activeTask ? Math.round(Math.min(1, Math.max(0, activeTask.progress)) * 100) : 0;
  const ranges = useMemo(() => {
    let cursor = 0;
    return state.project.segments.map((segment, index) => {
      const start = cursor;
      cursor += effectiveTimelineSegmentDuration(segment, state.project.render.fps);
      return { segment, index, start, end: cursor };
    });
  }, [state.project]);
  const enabledRanges = ranges.filter(({ segment }) => segment.enabled);
  const disabledRanges = ranges.filter(({ segment }) => !segment.enabled);
  const majorTick = timelineMajorTickSeconds(timelineZoom, total);
  const rulerTicks = useMemo(() => {
    if (total <= 0) return [0];
    const count = Math.min(TIMELINE_MAX_TICKS, Math.ceil(total / majorTick));
    const ticks = Array.from({ length: count + 1 }, (_, index) => Math.min(total, index * majorTick));
    if (ticks.at(-1) !== total) ticks.push(total);
    return ticks;
  }, [majorTick, total]);
  const canvasWidth = Math.max(1, total * timelineZoom);
  const sourceFrameCount = activeSegment?.mode === "ref2va" && activeSegment.source_video
    ? Math.max(0, Math.round(activeSegment.source_duration_seconds * projectFps))
    : 0;
  const maxEvenSplitPieces = Math.min(
    Math.max(1, 129 - state.project.segments.length),
    Math.max(1, Math.floor(sourceFrameCount / 5)),
  );

  const setProgramPlayhead = (seconds: number) => {
    const next = snapTimelineSeconds(
      Number.isFinite(seconds) ? seconds : playheadRef.current,
      projectFps,
      total,
    );
    playheadRef.current = next;
    if (latestStateRef.current.selected_asset_ids.length) {
      onDispatch({ type: "assets/clear-selection" });
    }
    onDispatch({ type: "playhead/set", seconds: next });
  };

  const updateTimelineZoom = (requested: number) => {
    const next = timelineZoom < TIMELINE_ZOOM_MIN
      ? requested <= timelineZoom ? timelineZoom : TIMELINE_ZOOM_MIN
      : clampTimelineZoom(requested);
    if (next === timelineZoom) return;
    const viewport = timelineViewportRef.current;
    const playheadViewportX = viewport
      ? state.playhead_seconds * timelineZoom - viewport.scrollLeft
      : 0;
    setTimelineZoom(next);
    if (viewport) {
      window.requestAnimationFrame(() => {
        viewport.scrollLeft = Math.max(0, state.playhead_seconds * next - playheadViewportX);
      });
    }
  };

  const fitTimeline = () => {
    const viewportWidth = timelineViewportRef.current?.clientWidth ?? 0;
    if (viewportWidth <= 0 || total <= 0) {
      updateTimelineZoom(TIMELINE_ZOOM_MIN);
      return;
    }
    // Fit is allowed below the manual 12 px/s range so a long programme
    // genuinely fits the viewport instead of merely jumping to minimum zoom.
    setTimelineZoom(Math.min(TIMELINE_ZOOM_MAX, Math.max(0.05, viewportWidth / total)));
    window.requestAnimationFrame(() => {
      if (timelineViewportRef.current) timelineViewportRef.current.scrollLeft = 0;
    });
  };

  const seekTimelinePointer = (clientX: number) => {
    const canvas = timelineCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    setPlaying(false);
    setProgramPlayhead((clientX - rect.left) / timelineZoom);
  };

  const stepProgramFrame = (direction: -1 | 1) => {
    setPlaying(false);
    const frame = Math.min(
      Math.max(0, totalFrames - 1),
      Math.max(0, Math.round(playheadRef.current * projectFps) + direction),
    );
    setProgramPlayhead(frame / projectFps);
  };

  const toggleProgramPlayback = () => {
    if (total <= 0) return;
    if (!playing && playheadRef.current >= total - frameStep / 2) {
      setProgramPlayhead(0);
    } else if (latestStateRef.current.selected_asset_ids.length) {
      onDispatch({ type: "assets/clear-selection" });
    }
    setPlaying((value) => !value);
  };

  const synchronizeProgramVideo = (video: HTMLVideoElement, tolerance = 0) => {
    video.volume = volume;
    video.muted = volume === 0;
    video.playbackRate = Math.min(16, Math.max(0.0625, sourcePlaybackRate));
    if (mappedProgramTime !== null) seekVideo(video, mappedProgramTime, tolerance);
  };

  const synchronizeOriginalVideo = (video: HTMLVideoElement, tolerance = 0) => {
    // The generated take is the audible programme track. Keeping the source
    // pane muted prevents doubled or slightly offset audio during comparison.
    video.volume = 0;
    video.muted = true;
    video.playbackRate = Math.min(16, Math.max(0.0625, originalPlaybackRate));
    if (mappedOriginalTime !== null) seekVideo(video, mappedOriginalTime, tolerance);
  };

  const handleProgramVideoEnded = () => {
    if (!showingTimeline || !timelinePosition) {
      setPlaying(false);
      return;
    }
    const segmentEnd = timelinePosition.start_seconds
      + effectiveTimelineSegmentDuration(timelinePosition.segment, projectFps);
    if (segmentEnd < total - frameStep / 2) {
      setProgramPlayhead(segmentEnd);
      return;
    }
    if (loop && total > 0) {
      setProgramPlayhead(0);
      setPlaybackRestartToken((token) => token + 1);
      return;
    }
    setProgramPlayhead(total);
    setPlaying(false);
  };

  const bindAssetsById = (
    segmentId: string,
    assetIds: string[],
    target: SegmentAssetTarget = "auto",
  ) => {
    const current = latestStateRef.current;
    const segment = current.project.segments.find((candidate) => candidate.id === segmentId);
    if (!segment) {
      setAssetDropNotice(persistentAssetDropNotice("目标片段已不存在，本次拖放已忽略"));
      return;
    }
    const byId = new Map(current.assets.map((asset) => [asset.id, asset]));
    const assets = assetIds.flatMap((id) => {
      const asset = byId.get(id);
      return asset ? [asset] : [];
    });
    const result = assignAssetsToSegment(segment, assets, target);
    if (assets.length) {
      onDispatch({ type: "segment/bind-assets", id: segmentId, assets, target, select: true });
    }
    setAssetDropNotice(assetBindingNotice(result, target));
  };

  const uploadAndBindFiles = async (
    segmentId: string,
    files: File[],
    target: SegmentAssetTarget = "auto",
  ) => {
    if (!files.length || uploadingFiles) return;
    if (!onUploadFiles) {
      setAssetDropNotice(persistentAssetDropNotice("当前环境未提供直接上传入口"));
      return;
    }
    setUploadingFiles(true);
    setAssetDropNotice(null);
    try {
      const uploadResult = await onUploadFiles(files);
      if (uploadResult.authority_stale) {
        setAssetDropNotice(persistentAssetDropNotice("上传期间 ComfyUI 设置已变化，旧地址返回的素材未加入当前工作区"));
        return;
      }
      const current = latestStateRef.current;
      const segment = current.project.segments.find((candidate) => candidate.id === segmentId);
      if (!segment) {
        if (uploadResult.assets.length) {
          onDispatch({ type: "assets/add", assets: uploadResult.assets });
        }
        setAssetDropNotice(persistentAssetDropNotice("目标片段已删除；上传成功的素材已加入素材库，但未绑定"));
        return;
      }
      const result = assignAssetsToSegment(segment, uploadResult.assets, target);
      if (uploadResult.assets.length) {
        onDispatch({
          type: "segment/bind-assets",
          id: segmentId,
          assets: uploadResult.assets,
          target,
          select: true,
        });
      }
      setAssetDropNotice(assetBindingNotice(result, target, uploadResult));
    } catch (reason) {
      setAssetDropNotice(persistentAssetDropNotice(reason instanceof Error ? reason.message : "素材上传失败"));
    } finally {
      setUploadingFiles(false);
    }
  };

  const appendUploadedVideos = async (files: File[]) => {
    if (!files.length || uploadingFiles) return;
    if (!onUploadFiles) {
      setTimelineToolNotice("当前环境未提供视频上传入口");
      return;
    }
    setUploadingFiles(true);
    setTimelineToolNotice(null);
    try {
      const result = await onUploadFiles(files);
      if (result.authority_stale) {
        setTimelineToolNotice("上传期间 ComfyUI 设置已变化，旧地址返回的视频未加入当前工作区");
        return;
      }
      if (result.assets.length) onDispatch({ type: "assets/add", assets: result.assets });
      const capacity = Math.max(0, 128 - latestStateRef.current.project.segments.length);
      const videos = result.assets.filter((asset) => asset.kind === "video");
      const inserted = videos.slice(0, capacity);
      const anchorId = latestStateRef.current.project.segments.at(-1)?.id ?? null;
      if (inserted.length) onDispatch({
        type: "segment/insert-videos",
        assets: inserted,
        anchorId,
        position: "after",
        ids: inserted.map(() => createSegmentId()),
      });
      const parts = inserted.length ? [`已按顺序追加 ${inserted.length} 个视频片段`] : [];
      if (videos.length > inserted.length) parts.push(`${videos.length - inserted.length} 个视频仅入库（已达 128 段上限）`);
      const nonVideos = result.assets.length - videos.length;
      if (nonVideos) parts.push(`${nonVideos} 个非视频素材仅加入素材库`);
      if (result.failures.length) parts.push(`${result.failures.length} 个文件上传失败`);
      setTimelineToolNotice(parts.join("；") || "没有可追加的视频");
    } catch (reason) {
      setTimelineToolNotice(reason instanceof Error ? reason.message : "视频上传失败");
    } finally {
      setUploadingFiles(false);
    }
  };

  const detectActiveSegmentShots = async () => {
    if (
      activeSegment?.mode !== "ref2va" ||
      !activeSegment.source_video ||
      detectingShots ||
      state.project.segments.length >= 128
    ) return;
    const requestId = ++detectionRequestRef.current;
    const segmentId = activeSegment.id;
    const expected = {
      asset_id: activeSegment.source_video.id,
      source_start_seconds: activeSegment.source_start_seconds,
      source_duration_seconds: activeSegment.source_duration_seconds,
      project_fps: state.project.render.fps,
    };
    setDetectingShots(true);
    setTimelineToolNotice(null);
    try {
      const result = await directorApi.detectRV2VShots({
        asset_id: expected.asset_id,
        frame_rate: expected.project_fps,
        sensitivity: detectionSensitivity,
        min_shot_frames: minimumShotFrames,
      });
      if (requestId !== detectionRequestRef.current) return;
      const latest = latestStateRef.current;
      const current = latest.project.segments.find((segment) => segment.id === segmentId);
      if (
        current?.mode !== "ref2va" ||
        current.source_video?.id !== expected.asset_id ||
        current.source_start_seconds !== expected.source_start_seconds ||
        current.source_duration_seconds !== expected.source_duration_seconds ||
        latest.project.render.fps !== expected.project_fps
      ) {
        setTimelineToolNotice("源视频、截取范围或帧率已变化，本次检测结果已忽略");
        return;
      }
      const startFrame = expected.source_start_seconds * expected.project_fps;
      const endFrame = (expected.source_start_seconds + expected.source_duration_seconds) * expected.project_fps;
      const inRangeCuts = [...new Set(result.cut_frames)].filter((frame) =>
        Number.isInteger(frame) && frame > startFrame + 0.001 && frame < endFrame - 0.001);
      const appliedCuts = inRangeCuts.slice(0, Math.max(0, 128 - latest.project.segments.length));
      if (!appliedCuts.length) {
        setTimelineToolNotice(latest.project.segments.length >= 128
          ? "时间线已达 128 段上限"
          : "当前源截取范围内未检测到可用切点");
        return;
      }
      onDispatch({
        type: "segment/apply-source-cuts",
        id: segmentId,
        cutFrames: appliedCuts,
        frameRate: expected.project_fps,
        expected,
        pieceIds: appliedCuts.map(() => createSegmentId()),
      });
      const truncated = appliedCuts.length < inRangeCuts.length ? "；超出 128 段上限的切点已忽略" : "";
      const warnings = result.warnings.length ? `；${result.warnings.join("；")}` : "";
      setTimelineToolNotice(`智能分割已将当前段拆为 ${appliedCuts.length + 1} 段${truncated}${warnings}`);
    } catch (reason) {
      if (requestId === detectionRequestRef.current)
        setTimelineToolNotice(reason instanceof Error ? reason.message : "智能分割失败");
    } finally {
      if (requestId === detectionRequestRef.current) setDetectingShots(false);
    }
  };

  useEffect(() => {
    detectionRequestRef.current += 1;
    setDetectingShots(false);
    return () => { detectionRequestRef.current += 1; };
  }, [activeSegment?.id,
    activeSegment?.mode === "ref2va" ? activeSegment.source_video?.id : null,
    activeSegment?.mode === "ref2va" ? activeSegment.source_start_seconds : null,
    activeSegment?.mode === "ref2va" ? activeSegment.source_duration_seconds : null,
    state.project.render.fps]);

  useEffect(() => {
    if (!selectAllCheckboxRef.current) return;
    selectAllCheckboxRef.current.indeterminate =
      selectedSegments.length > 0 && selectedSegments.length < state.project.segments.length;
  }, [selectedSegments.length, state.project.segments.length]);

  useEffect(() => {
    setLivePreviewToken(0);
    if (
      !showLiveMonitor ||
      !activeTask?.live_preview_url ||
      !["queued", "preparing", "running", "cancelling"].includes(activeTask.status)
    ) return;
    const interval = window.setInterval(() => {
      setLivePreviewToken((token) => (token + 1) % Number.MAX_SAFE_INTEGER);
    }, 500);
    return () => window.clearInterval(interval);
  }, [activeTask?.id, activeTask?.live_preview_url, activeTask?.status, showLiveMonitor]);

  useEffect(() => {
    if (!playing || total <= 0) return;
    let frame = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const delta = (now - previous) / 1000;
      previous = now;
      const video = programVideoRef.current ?? (comparisonActive ? originalVideoRef.current : null);
      const mapping = programClockRef.current;
      let next = playheadRef.current + delta;
      if (video && mapping?.hasMappedMedia && Number.isFinite(video.currentTime)) {
        const mediaLocal = mapping.candidate
          ? video.currentTime
          : (video.currentTime - mapping.sourceStart) / mapping.sourceRate;
        next = mapping.segmentStart + Math.min(mapping.segmentDuration, Math.max(0, mediaLocal));
      }
      if (next >= total) {
        if (loop) {
          playheadRef.current = 0;
          onDispatch({ type: "playhead/set", seconds: 0 });
          setPlaybackRestartToken((token) => token + 1);
        }
        else { playheadRef.current = total; onDispatch({ type: "playhead/set", seconds: total }); setPlaying(false); return; }
      } else if (Math.abs(next - playheadRef.current) > 1e-4) {
        playheadRef.current = next;
        onDispatch({ type: "playhead/set", seconds: next });
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, loop, total, onDispatch, comparisonActive]);

  useEffect(() => {
    setProgramMediaStatus(programMediaIdentity ? "loading" : "idle");
  }, [programMediaIdentity]);

  useEffect(() => {
    setOriginalMediaStatus(
      comparisonActive && sourceSegment?.source_video?.preview_url ? "loading" : "idle",
    );
  }, [comparisonActive, originalPreviewIdentity, sourceSegment?.source_video?.preview_url]);

  useEffect(() => {
    const video = programVideoRef.current;
    if (!video) return;
    synchronizeProgramVideo(video, playing ? 0.35 : 0.02);
  }, [previewIdentity, volume]);

  useEffect(() => {
    const video = programVideoRef.current;
    if (!video || mappedProgramTime === null) return;
    synchronizeProgramVideo(video, playing ? 0.35 : 0.02);
  }, [mappedProgramTime, playing, previewIdentity, sourcePlaybackRate]);

  useEffect(() => {
    const video = originalVideoRef.current;
    if (!comparisonActive || !video || mappedOriginalTime === null) return;
    synchronizeOriginalVideo(video, playing ? 0.35 : 0.02);
  }, [comparisonActive, mappedOriginalTime, originalPlaybackRate, originalPreviewIdentity, playing]);

  useEffect(() => {
    const programVideo = programVideoRef.current;
    const originalVideo = comparisonActive ? originalVideoRef.current : null;
    if (!playing) {
      programVideo?.pause();
      originalVideo?.pause();
      return;
    }
    if (programVideo) {
      void programVideo.play().catch(() => {
        // Keep UI and the audible programme truth aligned if autoplay fails.
        setPlaying(false);
      });
    }
    if (originalVideo) {
      void originalVideo.play().catch(() => {
        // A muted comparison follower must not stop a healthy generated take.
        if (!programVideo) setPlaying(false);
      });
    }
  }, [comparisonActive, originalPreviewIdentity, playbackRestartToken, previewIdentity, playing]);

  const deleteSelectedSegments = () => {
    const count = state.selected_segment_ids.length;
    if (!count) return;
    if (
      count > 1 &&
      !window.confirm(`确定删除所选的 ${count} 个片段吗？此操作会同时移除它们的时间线配置。`)
    ) return;
    onDispatch({ type: "segment/delete-selected", fallbackId: createSegmentId() });
  };

  const handleTimelineKey = (event: ReactKeyboardEvent) => {
    if (event.nativeEvent.isComposing || interactiveTimelineTarget(event.target)) return;
    const command = event.ctrlKey || event.metaKey;
    if (command && event.key.toLocaleLowerCase() === "a") {
      event.preventDefault();
      onDispatch({ type: "segment/set-selection", ids: state.project.segments.map((segment) => segment.id) });
      return;
    }
    if (command && event.key.toLocaleLowerCase() === "d") {
      event.preventDefault();
      onDispatch({
        type: "segment/duplicate-selected",
        ids: state.selected_segment_ids.map(() => createSegmentId()),
      });
      return;
    }
    if (command || event.altKey) return;
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      deleteSelectedSegments();
      return;
    }
    if (event.key === " ") {
      event.preventDefault();
      toggleProgramPlayback();
      return;
    }
    if (event.key.toLocaleLowerCase() === "s") {
      event.preventDefault();
      onDispatch({ type: "segment/split-selected", newId: createSegmentId() });
      return;
    }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const direction = event.key === "ArrowLeft" ? -1 : 1;
      setPlaying(false);
      setProgramPlayhead(playheadRef.current + direction * (event.shiftKey ? 1 : frameStep));
      return;
    }
    if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      setPlaying(false);
      setProgramPlayhead(event.key === "Home" ? 0 : total);
      return;
    }
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      updateTimelineZoom(timelineZoom + 12);
      return;
    }
    if (event.key === "-") {
      event.preventDefault();
      updateTimelineZoom(timelineZoom - 12);
      return;
    }
    if (event.key === "0") {
      event.preventDefault();
      fitTimeline();
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      onDispatch({ type: "segment/set-selection", ids: [] });
    }
  };
  const handleTimelineAssetDrop = (
    event: DragEvent,
    targetId?: string,
    zone: SegmentDropZone = "bind",
  ) => {
    // System files are only accepted by the Ref2VA inspector zones. Keeping
    // this guard ahead of the Director payload parser prevents a browser file
    // drop from inheriting the single-video before/after insertion semantics.
    if (hasSystemFileDrag(event)) {
      event.preventDefault();
      assetDropTargetRef.current = null;
      setAssetDropTarget(null);
      return;
    }
    const ids = dropAssetIds(event);
    if (!ids.length) return;
    event.preventDefault();
    const assets = ids.flatMap((id) => {
      const asset = state.assets.find((candidate) => candidate.id === id);
      return asset ? [asset] : [];
    });
    if (!assets.length) return;
    if (targetId) {
      const target = state.project.segments.find((segment) => segment.id === targetId);
      if (!target) return;
      if (assets.length === 1 && assets[0].kind === "video" && zone !== "bind") {
        onDispatch({ type: "segment/insert-video", asset: assets[0], anchorId: target.id, position: zone, id: createSegmentId() });
      } else {
        bindAssetsById(target.id, ids);
      }
      assetDropTargetRef.current = null;
      setAssetDropTarget(null);
      return;
    }
    const firstVideo = assets.find((asset) => asset.kind === "video");
    if (firstVideo) onDispatch({ type: "segment/insert-video", asset: firstVideo, anchorId: state.active_segment_id ?? state.selected_segment_ids.at(-1) ?? null, position: "after", id: createSegmentId() });
    assetDropTargetRef.current = null;
    setAssetDropTarget(null);
  };

  const mediaStatusOverlay = (
    status: "idle" | "loading" | "ready" | "seeking" | "error",
  ) => status !== "idle" && status !== "ready" ? (
    <div className={`monitor__media-status is-${status}`} role="status">
      {status === "seeking" ? "定位画面…" : status === "error" ? "预览媒体加载失败" : "载入预览…"}
    </div>
  ) : null;

  const candidateVideo = segmentCandidate ? <video
    ref={programVideoRef}
    key={previewIdentity}
    src={segmentCandidate.result.output_url}
    muted={volume === 0}
    preload="metadata"
    playsInline
    aria-label={`片段 ${segmentCandidate.result.segment_id} 的最新生成候选`}
    onLoadStart={() => setProgramMediaStatus("loading")}
    onLoadedMetadata={(event) => { synchronizeProgramVideo(event.currentTarget); setProgramMediaStatus("ready"); }}
    onCanPlay={() => setProgramMediaStatus("ready")}
    onWaiting={() => setProgramMediaStatus("loading")}
    onPlaying={() => setProgramMediaStatus("ready")}
    onSeeking={() => setProgramMediaStatus("seeking")}
    onSeeked={() => setProgramMediaStatus("ready")}
    onEnded={handleProgramVideoEnded}
    onError={() => { setProgramMediaStatus("error"); setPlaying(false); }}
  /> : null;

  const sourcePreviewVideo = preview?.kind === "video" && preview.preview_url ? <video
    ref={programVideoRef}
    key={preview.id}
    src={preview.preview_url}
    controls={!showingTimeline}
    muted={volume === 0}
    preload="metadata"
    playsInline
    aria-label={`源视频 ${preview.name}`}
    onLoadStart={() => setProgramMediaStatus("loading")}
    onLoadedMetadata={(event) => { synchronizeProgramVideo(event.currentTarget); setProgramMediaStatus("ready"); }}
    onCanPlay={() => setProgramMediaStatus("ready")}
    onWaiting={() => setProgramMediaStatus("loading")}
    onPlaying={() => setProgramMediaStatus("ready")}
    onSeeking={() => setProgramMediaStatus("seeking")}
    onSeeked={() => setProgramMediaStatus("ready")}
    onEnded={handleProgramVideoEnded}
    onError={() => { setProgramMediaStatus("error"); setPlaying(false); }}
  /> : null;

  const originalComparisonVideo = comparisonActive && sourceSegment?.source_video?.preview_url ? <video
    ref={originalVideoRef}
    key={`original:${originalPreviewIdentity}`}
    src={sourceSegment.source_video.preview_url}
    muted
    preload="metadata"
    playsInline
    aria-label={`原视频 ${sourceSegment.source_video.name}`}
    onLoadStart={() => setOriginalMediaStatus("loading")}
    onLoadedMetadata={(event) => { synchronizeOriginalVideo(event.currentTarget); setOriginalMediaStatus("ready"); }}
    onCanPlay={() => setOriginalMediaStatus("ready")}
    onWaiting={() => setOriginalMediaStatus("loading")}
    onPlaying={() => setOriginalMediaStatus("ready")}
    onSeeking={() => setOriginalMediaStatus("seeking")}
    onSeeked={() => setOriginalMediaStatus("ready")}
    onEnded={() => { if (!segmentCandidate) handleProgramVideoEnded(); }}
    onError={() => {
      setOriginalMediaStatus("error");
      if (!segmentCandidate) setPlaying(false);
    }}
  /> : null;

  const programFallback = preview?.kind === "image" && preview.preview_url
    ? <img src={preview.preview_url} alt={preview.name} />
    : sourcePreviewVideo
      ? sourcePreviewVideo
      : preview?.kind === "audio" && preview.preview_url
        ? <div className="monitor__audio"><span>♫</span><strong>{preview.name}</strong><audio src={preview.preview_url} controls /></div>
        : <div className="monitor__empty monitor__empty--canvas"><span>{timelinePosition ? TIMELINE_MODE_META[timelinePosition.segment.mode].shortLabel : "DIRECTOR"}</span><strong>{timelinePosition?.segment.title ?? state.project.title}</strong><small>{showingTimeline && timelinePosition ? "该片段尚无生成候选；当前仅预览占位画面或源素材" : "从左侧选择素材进行显式预览"}</small></div>;
  const comparisonDisabledReason = !showingTimeline
    ? "显式素材预览期间不可对比时间线原视频"
    : !comparisonAvailable
      ? "当前启用时间线没有可对比的 Ref2VA 源视频"
      : undefined;

  return (
    <main className="longform-workspace" aria-label="长视频时间线工作区">
      {selectionValidationErrors.length > 0 && <div id={TIMELINE_RUN_VALIDATION_ID} className="timeline-validation" role="alert"><strong>所选片段生成前还需完成 {selectionValidationErrors.length} 项</strong><span>{selectionValidationErrors.slice(0, 4).join("；")}{selectionValidationErrors.length > 4 ? `；另有 ${selectionValidationErrors.length - 4} 项` : ""}</span></div>}
      {compileReport && <section className="execution-plan-report" aria-label="服务端执行计划">
        <header><div><strong>原生分段执行计划</strong><small>{compileReport.execution_strategy} · 浏览器不接收可执行工作流或提示词</small></div><button type="button" className="icon-button" aria-label="关闭执行计划" onClick={onCloseCompile}>×</button></header>
        <div className="execution-plan-report__summary"><span>{compileReport.plans.length} 个分段计划</span><span>{compileReport.model_families.map((family) => family.toUpperCase()).join(" + ")}</span><span>图来源：{compileReport.node_policy.graph_source === "server" ? "服务端" : "未知"}</span><span>客户端工作流：{compileReport.node_policy.accepts_client_workflow ? "允许" : "禁止"}</span></div>
        <div className="execution-plan-report__plans">{compileReport.plans.map((plan) => {
          const targetIndex = state.project.segments.findIndex((segment) => segment.id === plan.segment_id);
          const targetLabel = targetIndex >= 0
            ? `${targetIndex + 1} · ${state.project.segments[targetIndex].title}`
            : plan.segment_id;
          const predecessorIndex = plan.predecessor_segment_id === null
            ? -1
            : state.project.segments.findIndex((segment) => segment.id === plan.predecessor_segment_id);
          const predecessorLabel = predecessorIndex >= 0
            ? `${predecessorIndex + 1} · ${state.project.segments[predecessorIndex].title}`
            : plan.predecessor_segment_id;
          return <article key={plan.segment_id} aria-label={`执行计划 ${targetLabel}`}>
            <header><strong title={targetLabel}>{targetLabel} · {TIMELINE_MODE_META[plan.mode].shortLabel} · 派生配方：{RECIPE_META[plan.recipe].label}</strong><em>{plan.backend === "raylight" ? "RayLight" : "标准"}</em></header>
            <span>{plan.visible_frame_count}f 可见 / {plan.sample_frame_count}f 采样 · seed {plan.seed}{plan.seed_mode === "random" ? "（随机）" : "（固定）"}</span>
            <small>{predecessorLabel
              ? `${plan.continuity_source === "historical_take" ? "来源：复用前驱成片" : "来源：同次运行"} · 前驱 ${predecessorLabel} · 接续上下文 ${plan.continuity_context_frames}f · 对齐尾帧 ${plan.alignment_tail_frame_count}f${plan.historical_take_id ? ` · take ${plan.historical_take_id.slice(0, 8)}` : ""}`
              : `来源：无接续 · ${plan.anchor_reset ? "锚点重置" : "无接续前驱"} · 对齐尾帧 ${plan.alignment_tail_frame_count}f`}</small>
            <small>{plan.conditioning_node}</small>
            <details><summary>{plan.node_classes.length} 类节点</summary><code>{plan.node_classes.join(" → ")}</code></details>
          </article>;
        })}</div>
        <footer><span>允许节点 {compileReport.node_policy.allowed_nodes.length}</span><span>自定义节点 {compileReport.node_policy.custom_nodes.length}</span>{compileReport.node_policy.custom_nodes.length > 0 && <code>{compileReport.node_policy.custom_nodes.join("、")}</code>}</footer>
      </section>}
      <section className={`director-monitors ${showLiveMonitor ? "is-live-open" : ""}`}>
        <article className="monitor monitor--program">
          <header><strong>时间线主预览</strong><em>{comparisonActive ? `原视频对比 · ${timelinePosition ? TIMELINE_MODE_META[timelinePosition.segment.mode].shortLabel : "无片段"}` : segmentCandidate ? `最新生成候选 · ${segmentCandidate.job_id.slice(0, 8)}` : showingTimeline ? timelineSource ? "源视频 · 无生成候选" : "片段占位画面 · 无生成候选" : "显式素材预览"} · {projectWidth}×{projectHeight}</em><button type="button" className="monitor__aux-toggle" aria-label="实时执行" aria-controls="live-execution-monitor" aria-expanded={showLiveMonitor} onClick={() => setShowLiveMonitor((open) => !open)}><span>实时执行</span>{activeTask && <i className="monitor__aux-toggle-status" aria-hidden="true" />}</button></header>
          <div className="monitor__screen">
            {comparisonActive ? <div className="monitor__comparison" role="group" aria-label="生成视频与原视频对比">
              <section className="monitor__comparison-pane" aria-label="生成视频">
                <header><strong>生成视频</strong><small>{segmentCandidate ? `候选 ${segmentCandidate.job_id.slice(0, 8)}` : "尚无生成候选"}</small></header>
                <div className="monitor__comparison-stage">
                  <div
                    className="monitor__canvas monitor__canvas--program"
                    style={projectAspectStyle}
                    aria-label={`项目预览画布 ${projectWidth}×${projectHeight}`}
                    aria-busy={programMediaStatus === "loading" || programMediaStatus === "seeking"}
                  >
                    {candidateVideo ?? <div className="monitor__empty monitor__empty--canvas"><span>GENERATED</span><strong>{timelinePosition?.segment.title ?? state.project.title}</strong><small>当前片段尚无生成候选</small></div>}
                    {mediaStatusOverlay(programMediaStatus)}
                  </div>
                </div>
              </section>
              <section className="monitor__comparison-pane" aria-label="原视频">
                <header><strong>原视频</strong><small>{sourceSegment?.source_video ? `${sourceSegment.source_start_seconds.toFixed(2)}–${(sourceSegment.source_start_seconds + sourceSegment.source_duration_seconds).toFixed(2)}s` : "当前片段无原视频"}</small></header>
                <div className="monitor__comparison-stage">
                  <div
                    className="monitor__canvas monitor__canvas--original"
                    style={projectAspectStyle}
                    aria-label={`原视频对比画布 ${projectWidth}×${projectHeight}`}
                    aria-busy={originalMediaStatus === "loading" || originalMediaStatus === "seeking"}
                  >
                    {originalComparisonVideo ?? <div className="monitor__empty monitor__empty--canvas"><span>ORIGINAL</span><strong>{timelinePosition?.segment.title ?? state.project.title}</strong><small>{timelinePosition?.segment.mode === "fl2va" ? "FL2VA 片段没有原视频" : timelinePosition?.segment.mode === "ref2va" ? "当前 Ref2VA 片段未绑定可预览的原视频" : "播放头未落在片段内"}</small></div>}
                    {mediaStatusOverlay(originalMediaStatus)}
                  </div>
                </div>
              </section>
            </div> : <div
                className="monitor__canvas monitor__canvas--program"
                style={projectAspectStyle}
                aria-label={`项目预览画布 ${projectWidth}×${projectHeight}`}
                aria-busy={programMediaStatus === "loading" || programMediaStatus === "seeking"}
              >
                {candidateVideo ?? programFallback}
                {mediaStatusOverlay(programMediaStatus)}
              </div>}
          </div>
          <footer className="transport-controls">
            <div className="transport-controls__seek">
              <output aria-label="当前播放时间">{formatClock(state.playhead_seconds)}</output>
              <input
                aria-label="主预览播放头"
                type="range"
                min="0"
                max={Math.max(total, 0.01)}
                step={frameStep}
                value={Math.min(state.playhead_seconds, total)}
                disabled={total <= 0}
                onPointerDown={() => setPlaying(false)}
                onChange={(event) => { setPlaying(false); setProgramPlayhead(Number(event.target.value)); }}
              />
              <output aria-label="总播放时间">{formatClock(total)}</output>
              <label className="transport-controls__frames"><span>帧</span><DeferredNumberInput
                aria-label="当前预览帧"
                min="1"
                max={Math.max(1, totalFrames)}
                step="1"
                value={Math.min(Math.max(1, currentFrame), Math.max(1, totalFrames))}
                disabled={totalFrames <= 0}
                onFocus={() => setPlaying(false)}
                normalizeValue={Math.trunc}
                onValueCommit={(frame) => {
                  setPlaying(false);
                  setProgramPlayhead((frame - 1) / projectFps);
                }}
              /><span>/ {totalFrames}</span></label>
            </div>
            <div className="transport-controls__buttons">
              <button type="button" aria-label="跳到开头" title="跳到开头" disabled={total <= 0} onClick={() => { setPlaying(false); setProgramPlayhead(0); }}><TransportBoundaryIcon edge="start" /></button>
              <button type="button" aria-label="上一帧" title="上一帧" disabled={total <= 0 || state.playhead_seconds <= 0} onClick={() => stepProgramFrame(-1)}>‹</button>
              <button type="button" className="transport-controls__play" aria-label={playing ? "暂停" : "播放"} disabled={total <= 0} onClick={toggleProgramPlayback}>{playing ? "Ⅱ" : "▶"}</button>
              <button type="button" aria-label="下一帧" title="下一帧" disabled={totalFrames <= 0 || currentFrame >= totalFrames} onClick={() => stepProgramFrame(1)}>›</button>
              <button type="button" aria-label="跳到结尾" title="跳到结尾" disabled={total <= 0} onClick={() => { setPlaying(false); setProgramPlayhead(total); }}><TransportBoundaryIcon edge="end" /></button>
              <button type="button" aria-label={volume === 0 ? "取消静音" : "静音"} title={volume === 0 ? "取消静音" : "静音"} onClick={() => setVolume((value) => value === 0 ? 0.8 : 0)}>{volume === 0 ? "🔇" : "🔊"}</button>
              <label><span>音量</span><input aria-label="预览音量" type="range" min="0" max="1" step="0.05" value={volume} onChange={(event) => setVolume(Number(event.target.value))} /></label>
              <button type="button" className="transport-controls__compare" aria-label="原视频对比" aria-pressed={comparisonActive} disabled={Boolean(comparisonDisabledReason)} title={comparisonDisabledReason} onClick={() => setCompareOriginal((current) => !current)}>原视频对比</button>
              <label className="toggle"><input type="checkbox" checked={loop} onChange={(event) => setLoop(event.target.checked)} /><span />循环</label>
            </div>
          </footer>
        </article>

        {showLiveMonitor && <article id="live-execution-monitor" className="monitor monitor--latent">
          <header><strong>实时执行进度</strong><em>{activeTask ? `${activeTask.id.slice(0, 8)} · ${activeTask.stage ?? activeTask.status}` : "空闲"}</em></header>
          <div className="monitor__screen">
            <div className="monitor__canvas monitor__canvas--live" style={projectAspectStyle} aria-label={`实时预览画布 ${projectWidth}×${projectHeight}`}>
              {activeTask?.live_preview_url ? <img
                className="monitor__live-preview"
                key={`${activeTask.id}:${activeTask.live_preview_url}`}
                src={`${activeTask.live_preview_url}${activeTask.live_preview_url.includes("?") ? "&" : "?"}v=${encodeURIComponent(activeTask.updated_at)}&frame=${livePreviewToken}`}
                alt={`任务 ${activeTask.id.slice(0, 8)} 最新采样预览帧`}
              /> : <div className="latent-placeholder"><span className={activeTask ? "is-active" : ""} /><strong>{activeTask ? `${progress}%` : "就绪"}</strong><small>{activeTask ? `${activeTask.stage ?? "ComfyUI 正在执行"} · 当前阶段没有预览帧` : "采样时显示预览帧；其他阶段显示当前执行状态"}</small></div>}
            </div>
          </div>
          <footer className="latent-status">
            <div><span><i style={{ width: `${progress}%` }} /></span><small>{activeTask ? `${progress}% · ${activeTask.stage ?? activeTask.status}` : "尚无活动任务"}</small></div>
            <button type="button" className="button button--danger" title={capabilities.supports_cancel ? undefined : "当前 ComfyUI 版本不支持安全的原子取消"} disabled={!capabilities.supports_cancel || !activeTask || !["queued", "preparing", "running", "cancelling"].includes(activeTask.status)} onClick={() => activeTask && onCancelTask(activeTask.id)}>{activeTask?.status === "cancelling" ? "重试终止" : "终止任务"}</button>
          </footer>
        </article>}
      </section>

      <section className="timeline-commandbar" aria-label="时间线功能按钮区">
        <input
          ref={appendVideoInputRef}
          className="timeline-commandbar__file-input"
          aria-label="选择要导入并追加的视频"
          type="file"
          accept="video/*,.mp4,.m4v,.mov,.webm,.mkv,.avi,.mpeg,.mpg"
          multiple
          onChange={(event) => {
            const files = Array.from(event.currentTarget.files ?? []);
            event.currentTarget.value = "";
            void appendUploadedVideos(files);
          }}
        />
        <div className="timeline-commandbar__group timeline-commandbar__group--edit">
          <button type="button" disabled={!onUploadFiles || uploadingFiles || capabilities.connection !== "online" || state.project.segments.length >= 128} onClick={() => appendVideoInputRef.current?.click()}>{uploadingFiles ? "导入中…" : "导入并追加视频"}</button>
          <button type="button" disabled={state.project.segments.length >= 128} onClick={() => onDispatch({ type: "segment/insert", position: "before", mode: activeSegment?.mode ?? "fl2va", id: createSegmentId() })}>＋ 前插空段</button>
          <button type="button" disabled={state.project.segments.length >= 128} onClick={() => onDispatch({ type: "segment/insert", position: "after", mode: activeSegment?.mode ?? "fl2va", id: createSegmentId() })}>＋ 后插空段</button>
          <button type="button" disabled={!canSplitSelectedSegment(state)} onClick={() => onDispatch({ type: "segment/split-selected", newId: createSegmentId() })}>播放头拆分</button>
          <label className="timeline-commandbar__inline-field"><span>均分</span><DeferredNumberInput aria-label="均分片段数量" min="2" max={Math.max(2, maxEvenSplitPieces)} step="1" value={evenSplitPieces} normalizeValue={Math.trunc} onValueCommit={setEvenSplitPieces} /></label>
          <button type="button" disabled={activeSegment?.mode !== "ref2va" || !activeSegment.source_video || evenSplitPieces > maxEvenSplitPieces || state.project.segments.length >= 128} onClick={() => activeSegment && onDispatch({ type: "segment/split-evenly", id: activeSegment.id, pieces: evenSplitPieces, pieceIds: Array.from({ length: Math.max(0, evenSplitPieces - 1) }, () => createSegmentId()) })}>均分当前段</button>
          <label className="timeline-commandbar__inline-field"><span>灵敏度</span><select aria-label="智能分割灵敏度" value={detectionSensitivity} disabled={detectingShots} onChange={(event) => setDetectionSensitivity(event.target.value as RV2VShotDetectionRequest["sensitivity"])}><option value="low">低</option><option value="medium">中</option><option value="high">高</option></select></label>
          <label className="timeline-commandbar__inline-field"><span>最短帧</span><DeferredNumberInput aria-label="智能分割最短镜头帧数" min="4" max="100000" step="1" value={minimumShotFrames} disabled={detectingShots} normalizeValue={Math.trunc} onValueCommit={setMinimumShotFrames} /></label>
          <button type="button" disabled={capabilities.connection !== "online" || detectingShots || activeSegment?.mode !== "ref2va" || !activeSegment.source_video || state.project.segments.length >= 128} onClick={() => void detectActiveSegmentShots()}>{detectingShots ? "检测中…" : "智能分割"}</button>
          <button type="button" disabled={!canMergeSelectedSegments(state)} onClick={() => onDispatch({ type: "segment/merge-selected" })}>合并所选</button>
          <button type="button" disabled={!state.selected_segment_ids.length || state.project.segments.length >= 128} onClick={() => onDispatch({ type: "segment/duplicate-selected", ids: state.selected_segment_ids.map(() => createSegmentId()) })}>复制片段</button>
          <button type="button" className="is-danger" disabled={!state.selected_segment_ids.length} onClick={deleteSelectedSegments}>删除所选</button>
        </div>
      </section>

      <section className="director-timeline" tabIndex={0} onKeyDown={handleTimelineKey} aria-label="主时间线">
        <header>
          <div className="director-timeline__title">
            <strong>长视频编排</strong>
            <div className="director-timeline__selection-controls" role="group" aria-label="分段选择与启用状态">
              <label className="timeline-selection-filter"><input
                ref={selectAllCheckboxRef}
                type="checkbox"
                checked={state.project.segments.length > 0 && selectedSegments.length === state.project.segments.length}
                onChange={(event) => onDispatch({
                  type: "segment/set-selection",
                  ids: event.target.checked ? state.project.segments.map((segment) => segment.id) : [],
                })}
              /><span>全选</span></label>
              <button
                type="button"
                className="timeline-disable-selected"
                aria-label={`禁用所选，${runnableSelection.length} 个已启用片段`}
                disabled={!runnableSelection.length}
                onClick={() => onDispatch({ type: "segment/set-enabled", ids: runnableSelection, enabled: false })}
              >禁用所选</button>
              <output className="timeline-selection-count" aria-live="polite">已选 {selectedSegments.length} 个片段</output>
            </div>
          </div>
          <div className="director-timeline__summary" role="group" aria-label="项目摘要">
            <span className="director-timeline__metric"><b>{state.project.segments.length}</b><small>段</small></span>
            <span className="director-timeline__metric is-time"><b>{formatClock(total)}</b><small>总时长</small></span>
            <span className="director-timeline__metric"><b>{state.project.render.width}×{state.project.render.height}</b><small>{state.project.render.fps}fps</small></span>
            <em>{state.project.export_mode === "all" ? "组装完整视频" : "输出独立片段"}</em>
            {configuredContinuityCount > 0 && <em className={`timeline-run-continuity ${blockingContinuityIssueCount ? "is-blocked" : ""}`} role="status" aria-live="polite">
              {continuityRunIssues.length
                ? continuityBlockedStatus
                : activeContinuityBoundaryCount > 0
                  ? `接续 ${activeContinuityBoundaryCount} 段`
                  : "当前选择没有已启用的接续边界"}
            </em>}
            {timelineToolNotice && <em role="status">{timelineToolNotice}</em>}
          </div>
          <div className="director-timeline__actions">
            <div className="timeline-zoom-controls" role="group" aria-label="时间线缩放">
              <button type="button" aria-label="缩小时间线" disabled={timelineZoom <= TIMELINE_ZOOM_MIN} onClick={() => updateTimelineZoom(timelineZoom - 12)}>−</button>
              <input aria-label="时间线缩放比例" type="range" min={TIMELINE_ZOOM_MIN} max={TIMELINE_ZOOM_MAX} step="1" value={clampTimelineZoom(timelineZoom)} onChange={(event) => updateTimelineZoom(Number(event.target.value))} />
              <button type="button" aria-label="放大时间线" disabled={timelineZoom >= TIMELINE_ZOOM_MAX} onClick={() => updateTimelineZoom(timelineZoom + 12)}>＋</button>
              <button type="button" onClick={fitTimeline}>适合窗口</button>
              <output>{timelineZoom.toFixed(timelineZoom < 10 ? 2 : 0)} px/s</output>
            </div>
          </div>
        </header>
        <div ref={timelineViewportRef} className="director-timeline__viewport" data-testid="timeline-viewport">
          <div
            ref={timelineCanvasRef}
            className="director-timeline__canvas"
            data-testid="timeline-canvas"
            style={{ width: `${canvasWidth}px`, "--timeline-pps": timelineZoom, "--timeline-grid": `${timelineZoom}px` } as CSSProperties}
            onPointerDown={(event) => {
              if (interactiveTimelineTarget(event.target)) return;
              timelineScrubbingRef.current = true;
              event.currentTarget.setPointerCapture?.(event.pointerId);
              seekTimelinePointer(event.clientX);
            }}
            onPointerMove={(event) => {
              if (timelineScrubbingRef.current) seekTimelinePointer(event.clientX);
            }}
            onPointerUp={(event) => {
              timelineScrubbingRef.current = false;
              event.currentTarget.releasePointerCapture?.(event.pointerId);
            }}
            onPointerCancel={() => { timelineScrubbingRef.current = false; }}
          >
            <div className="director-timeline__ruler" aria-label="时间刻度">
              {rulerTicks.map((seconds, index) => <span
                key={`${seconds}-${index}`}
                className="director-timeline__tick"
                style={{ left: `${seconds * timelineZoom}px` }}
                data-seconds={seconds}
              ><i />{formatRulerClock(seconds)}</span>)}
            </div>
            <div
              className="director-timeline__playhead"
              style={{ left: `${state.playhead_seconds * timelineZoom}px` }}
              aria-label="时间线播放头"
              data-seconds={state.playhead_seconds}
            ><span /></div>
            <div className="director-timeline__track" onDragOver={(event) => { if (hasDirectorAssetDrag(event)) { event.preventDefault(); event.dataTransfer.dropEffect = state.project.segments.length >= 128 ? "none" : "copy"; } }} onDrop={(event) => handleTimelineAssetDrop(event)}>
          {enabledRanges.map(({ segment, index, start, end }) => {
            const selected = state.selected_segment_ids.includes(segment.id);
            const meta = TIMELINE_MODE_META[segment.mode];
            const recipe = deriveSegmentRecipe(segment);
            const refs = segmentAssetReferences(segment);
            const continuityBoundary = segment.continuity.enabled
              ? continuityBoundaryByTarget.get(segment.id) ?? null
              : null;
            const continuityIssue = continuityIssueByTarget.get(segment.id) ?? null;
            const continuitySelected = runnableSelected.has(segment.id);
            const continuityStatus = continuityBoundary
              ? continuityBoundary.kind === "explicit-first-image"
                ? {
                    tone: "hard",
                    short: "首帧硬断点",
                    description: `第 ${index + 1} 段使用显式首帧，不读取前段尾帧`,
                  }
                : continuityIssue?.code === "historical-take-required"
                    ? {
                        tone: "history",
                        short: "复用前驱成片",
                        description: continuityIssue.message,
                      }
                    : continuityIssue?.code === "predecessor-too-short"
                      ? {
                          tone: "missing",
                          short: "前段过短",
                          description: continuityIssue.message,
                        }
                      : continuityIssue?.code === "sample-too-long"
                        ? {
                            tone: "missing",
                            short: "采样超限",
                            description: continuityIssue.message,
                          }
                        : continuitySelected && runnableSelected.has(continuityBoundary.predecessor.id)
                          ? {
                              tone: "active",
                              short: "接续",
                              description: `第 ${index + 1} 段将使用第 ${continuityBoundary.predecessor_index + 1} 段最后 ${segment.continuity.overlap_frames} 帧接续生成`,
                            }
                          : {
                              tone: "eligible",
                              short: "可接续",
                              description: `同时选择第 ${continuityBoundary.predecessor_index + 1} 段与第 ${index + 1} 段后，可使用前段最后 ${segment.continuity.overlap_frames} 帧接续生成`,
                            }
              : null;
            const continuityDescriptionId = continuityStatus
              ? `timeline-continuity-boundary-${index}`
              : undefined;
            const editing = state.active_segment_id === segment.id;
            const editingDescriptionId = editing
              ? `timeline-editing-source-${segment.id}`
              : undefined;
            const clipDescriptionIds = [continuityDescriptionId, editingDescriptionId]
              .filter(Boolean)
              .join(" ") || undefined;
            return (
              <article
                key={segment.id}
                className={`timeline-clip timeline-clip--${meta.accent} ${selected ? "is-selected" : ""} ${assetDropTarget?.id === segment.id ? `is-drop-${assetDropTarget.zone}` : ""}`}
                style={{ left: `${start * timelineZoom}px`, width: `${Math.max(0, end - start) * timelineZoom}px` }}
                data-start-seconds={start}
                data-duration-seconds={Math.max(0, end - start)}
                onDragLeave={(event) => {
                  if (event.currentTarget.contains(event.relatedTarget as Node | null)) return;
                  if (assetDropTargetRef.current?.id === segment.id) {
                    assetDropTargetRef.current = null;
                    setAssetDropTarget(null);
                  }
                }}
                onDragOver={(event) => {
                  if (hasSystemFileDrag(event)) {
                    event.preventDefault();
                    event.dataTransfer.dropEffect = "none";
                    assetDropTargetRef.current = null;
                    setAssetDropTarget(null);
                    return;
                  }
                  if (!hasDirectorAssetDrag(event)) { event.preventDefault(); event.dataTransfer.dropEffect = "move"; return; }
                  const assetIds = dropAssetIds(event);
                  const assets = assetIds.flatMap((id) => {
                    const asset = state.assets.find((candidate) => candidate.id === id);
                    return asset ? [asset] : [];
                  });
                  const asset = assets[0];
                  const kind = asset?.kind ?? directorAssetDragKind(event);
                  if (!kind) return;
                  const rect = event.currentTarget.getBoundingClientRect();
                  const zone = assets.length === 1 && kind === "video" ? clipDropZone(event.clientX, rect.left, rect.width) : "bind";
                  const accepted = zone === "bind"
                    ? assets.length
                      ? assets.some((candidate) => segmentAcceptsAsset(segment, candidate))
                      : segmentAcceptsAssetKind(segment, kind)
                    : state.project.segments.length < 128;
                  if (!accepted) { event.dataTransfer.dropEffect = "none"; return; }
                  event.preventDefault();
                  event.dataTransfer.dropEffect = "copy";
                  assetDropTargetRef.current = { id: segment.id, zone };
                  if (assetDropTarget?.id !== segment.id || assetDropTarget.zone !== zone)
                    setAssetDropTarget({ id: segment.id, zone });
                }}
                onDrop={(event) => {
                  event.stopPropagation();
                  if (hasSystemFileDrag(event)) {
                    event.preventDefault();
                    assetDropTargetRef.current = null;
                    setAssetDropTarget(null);
                    return;
                  }
                  const assetIds = dropAssetIds(event);
                  if (assetIds.length) {
                    const assets = assetIds.flatMap((id) => {
                      const asset = state.assets.find((candidate) => candidate.id === id);
                      return asset ? [asset] : [];
                    });
                    const rect = event.currentTarget.getBoundingClientRect();
                    const remembered = assetDropTargetRef.current?.id === segment.id
                      ? assetDropTargetRef.current.zone
                      : null;
                    const zone = remembered ?? (assets.length === 1 && assets[0]?.kind === "video" ? clipDropZone(event.clientX, rect.left, rect.width) : "bind");
                    handleTimelineAssetDrop(event, segment.id, zone);
                    return;
                  }
                  event.preventDefault();
                  const dragged = event.dataTransfer.getData("application/x-director-segment") || draggedSegmentId;
                  if (dragged) onDispatch({ type: "segment/move", draggedId: dragged, targetId: segment.id });
                }}
              >
                <button
                  type="button"
                  className="timeline-clip__select-surface"
                  draggable
                  aria-label={`聚焦并选择片段 ${index + 1}：${segment.title}`}
                  aria-describedby={clipDescriptionIds}
                  aria-pressed={selected}
                  title="单击设为编辑与复制来源；Ctrl/Cmd 或 Shift 仅调整多选"
                  onClick={(event) => {
                    if (suppressSegmentClickRef.current === segment.id) {
                      suppressSegmentClickRef.current = null;
                      return;
                    }
                    onDispatch({ type: "segment/select", id: segment.id, additive: event.ctrlKey || event.metaKey, range: event.shiftKey });
                  }}
                  onDragStart={(event) => {
                    suppressSegmentClickRef.current = segment.id;
                    setDraggedSegmentId(segment.id);
                    event.dataTransfer.effectAllowed = "move";
                    event.dataTransfer.setData("application/x-director-segment", segment.id);
                  }}
                  onDragEnd={() => {
                    setDraggedSegmentId(null);
                    assetDropTargetRef.current = null;
                    setAssetDropTarget(null);
                    window.setTimeout(() => {
                      if (suppressSegmentClickRef.current === segment.id)
                        suppressSegmentClickRef.current = null;
                    }, 0);
                  }}
                />
                <header>
                  <span>{String(index + 1).padStart(2, "0")}</span><strong>{meta.shortLabel}</strong>
                  {editing && <>
                    <span className="timeline-editing-badge" aria-hidden="true">编辑中</span>
                    <span id={editingDescriptionId} className="sr-only">当前编辑片段，也是复制来源</span>
                  </>}
                  {continuityStatus && <>
                    <span className={`timeline-clip__continuity is-${continuityStatus.tone}`} title={continuityStatus.description}>{continuityStatus.short}</span>
                    <span id={continuityDescriptionId} className="sr-only">{continuityStatus.description}</span>
                  </>}
                  <label className="timeline-clip__selection" title={selected ? "移出多选" : "加入多选"}><input
                    type="checkbox"
                    aria-label={`多选片段 ${index + 1}：${segment.title}`}
                    checked={selected}
                    onClick={(event) => event.stopPropagation()}
                    onChange={() => onDispatch({ type: "segment/toggle-selection", id: segment.id })}
                  /></label>
                </header>
                {assetDropTarget?.id === segment.id && <div className="timeline-clip__drop-hint" aria-live="polite">{assetDropTarget.zone === "before" ? "在此片段前新建 Ref2VA" : assetDropTarget.zone === "after" ? "在此片段后新建 Ref2VA" : "绑定当前片段素材"}</div>}
                <div className="timeline-clip__thumb">
                  {segment.mode === "ref2va" && segment.source_video?.preview_url
                    ? <SourceVideoFilmstrip key={`${segment.source_video.id}:${segment.source_start_seconds}:${segment.source_duration_seconds}`} segment={segment} />
                    : refs[0]?.kind === "image" && refs[0].preview_url
                      ? <img src={refs[0].preview_url} alt="" />
                      : refs[0]?.kind === "video" && refs[0].preview_url
                        ? <video src={refs[0].preview_url} muted preload="metadata" />
                        : <span>{meta.shortLabel}</span>}
                </div>
                <strong title={segment.title}>{segment.title}</strong>
                <small title={segment.mode === "ref2va" && segment.source_video ? `源视频 ${segment.source_start_seconds.toFixed(2)}–${(segment.source_start_seconds + segment.source_duration_seconds).toFixed(2)} 秒 / 素材总长 ${segment.source_video.metadata?.duration.toFixed(2) ?? "—"} 秒` : undefined}>{RECIPE_META[recipe].label} · {segment.enabled ? `${start.toFixed(2)}–${end.toFixed(2)}s` : "已停用 · 不计入时基"}{segment.mode === "ref2va" && segment.source_video ? ` · 源 ${segment.source_start_seconds.toFixed(2)}–${(segment.source_start_seconds + segment.source_duration_seconds).toFixed(2)}s` : ""}{alignH3Frames(segment.duration_seconds, state.project.render.fps) > 512 && segment.mode === "ref2va" && segment.source_video ? " · 待分割" : ` · 请求 ${segment.duration_seconds.toFixed(2)}s / 实际 ${alignedTimelineSegmentDuration(segment, state.project.render.fps).toFixed(2)}s`} · {refs.length} 素材</small>
              </article>
            );
          })}
            </div>
          </div>
        </div>
        {disabledRanges.length > 0 && <section className="director-timeline__disabled" aria-label="停用片段">
          <header><strong>停用片段</strong><small>不占节目时间；启用后回到主轨道</small></header>
          <div>{disabledRanges.map(({ segment, index }) => <article key={segment.id} className={state.selected_segment_ids.includes(segment.id) ? "is-selected" : ""}>
            <span aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
            <button
              type="button"
              className="director-timeline__disabled-select"
              aria-label={`选择停用片段 ${index + 1}：${segment.title}`}
              aria-describedby={state.active_segment_id === segment.id ? `timeline-editing-source-${segment.id}` : undefined}
              aria-pressed={state.selected_segment_ids.includes(segment.id)}
              title="单击设为编辑与复制来源；Ctrl/Cmd 或 Shift 仅调整多选"
              onClick={(event) => onDispatch({ type: "segment/select", id: segment.id, additive: event.ctrlKey || event.metaKey, range: event.shiftKey })}
            ><strong>{segment.title}</strong><small>{TIMELINE_MODE_META[segment.mode].shortLabel}</small>{state.active_segment_id === segment.id && <>
              <span className="timeline-editing-badge" aria-hidden="true">编辑中</span>
              <span id={`timeline-editing-source-${segment.id}`} className="sr-only">当前编辑片段，也是复制来源</span>
            </>}</button>
            <label className="director-timeline__disabled-checkbox" title={state.selected_segment_ids.includes(segment.id) ? "移出多选" : "加入多选"}><input
              type="checkbox"
              aria-label={`多选停用片段 ${index + 1}：${segment.title}`}
              checked={state.selected_segment_ids.includes(segment.id)}
              onClick={(event) => event.stopPropagation()}
              onChange={() => onDispatch({ type: "segment/toggle-selection", id: segment.id })}
            /></label>
            <button
              type="button"
              aria-label={`启用片段 ${index + 1}：${segment.title}`}
              onClick={() => onDispatch({ type: "segment/set-enabled", ids: [segment.id], enabled: true })}
            >启用</button>
          </article>)}</div>
        </section>}
      </section>

      {activeSegment ? <SegmentInspector
        state={state}
        segment={activeSegment}
        capabilities={capabilities}
        runtimeEnabled={capabilities.connection === "online" && !uploadingFiles}
        onDispatch={onDispatch}
        onBindAssets={bindAssetsById}
        onDropFiles={(segmentId, files, target) => void uploadAndBindFiles(segmentId, files, target)}
        dropNotice={assetDropNotice}
        uploadingFiles={uploadingFiles}
      /> : (
        <section className="segment-inspector segment-inspector--empty" aria-label="当前片段编辑器">
          <div><h2>未选择片段</h2><p>选择一个片段后可在这里编辑其生成配置。</p></div>
        </section>
      )}
    </main>
  );
}
