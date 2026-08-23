import type {
  AssetKind,
  AssetReference,
  FL2VDraft,
  I2VDraft,
  ModeDraft,
  R2VDraft,
  RV2VDraft,
  SlottedAssetReference,
  T2VDraft,
  VideoMetadata,
  V2VDraft,
} from "./modes";
import {
  MINIMAX_H3_REFERENCE_LIMITS,
  maxSlotForCapacity,
} from "./h3Capabilities";

/** Classifies an operating-system file before it is uploaded to ComfyUI. */
export function inferAssetKindFromFile(file: File): AssetKind | null {
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("audio/")) return "audio";
  if (file.type.startsWith("video/")) return "video";
  const extension = file.name.split(".").at(-1)?.toLowerCase();
  if (["png", "jpg", "jpeg", "webp"].includes(extension ?? "")) return "image";
  if (["wav", "mp3", "flac", "ogg", "oga", "m4a", "aac"].includes(extension ?? "")) return "audio";
  if (["mp4", "m4v", "mov", "webm", "mkv", "avi", "mpeg", "mpg"].includes(extension ?? "")) return "video";
  return null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafePathComponent(value: string): boolean {
  const normalized = value.replaceAll("\\", "/");
  return !normalized.startsWith("/") && !normalized.split("/").includes("..");
}

function normalizeVideoMetadata(value: unknown): VideoMetadata | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.duration !== "number" ||
    !Number.isFinite(value.duration) ||
    value.duration <= 0 ||
    typeof value.native_fps !== "number" ||
    !Number.isFinite(value.native_fps) ||
    value.native_fps <= 0 ||
    ![value.frame_count, value.width, value.height].every(
      (field) => typeof field === "number" && Number.isInteger(field) && field > 0,
    ) ||
    typeof value.probe_method !== "string" ||
    !value.probe_method.trim() ||
    value.probe_method.length > 128
  ) {
    return null;
  }
  return {
    duration: value.duration,
    native_fps: value.native_fps,
    frame_count: value.frame_count as number,
    width: value.width as number,
    height: value.height as number,
    probe_method: value.probe_method,
    // Historical saved metadata predates stream probing. Fail closed instead
    // of presenting a conditioning option that would pass an empty audio edge.
    has_audio: value.has_audio === true,
  };
}

/**
 * Normalizes an API/local-storage asset into the exact backend payload shape.
 * Missing stable IDs, wrong media kinds and unsafe paths are rejected.
 */
export function normalizeAssetReference(
  value: unknown,
  expectedKind: AssetKind,
  options: {
    includeContentHash?: boolean;
    requireContentHash?: boolean;
    completeWireShape?: boolean;
  } = {},
): AssetReference | null {
  if (!isRecord(value)) return null;
  if (typeof value.id !== "string" || !value.id.trim()) return null;
  if (typeof value.name !== "string" || !value.name || value.name.length > 1024) return null;
  if (value.kind !== expectedKind || value.type !== "input") return null;

  const subfolder = value.subfolder === undefined ? "" : value.subfolder;
  if (typeof subfolder !== "string" || subfolder.length > 1024) return null;
  if (!isSafePathComponent(value.name) || !isSafePathComponent(subfolder)) return null;

  const asset: AssetReference = {
    id: value.id.trim(),
    name: value.name,
    subfolder,
    type: "input",
    kind: expectedKind,
  };
  for (const key of ["filename", "path", "preview_url"] as const) {
    const optional = value[key];
    if (options.completeWireShape) {
      if (optional !== undefined && typeof optional !== "string" && optional !== null) return null;
      asset[key] = typeof optional === "string" ? optional : null;
    } else if (typeof optional === "string" || optional === null) {
      asset[key] = optional;
    }
  }
  if (
    options.includeContentHash !== false &&
    options.requireContentHash === true &&
    !Object.prototype.hasOwnProperty.call(value, "content_hash")
  ) return null;
  if (options.includeContentHash !== false && value.content_hash !== undefined) {
    if (
      value.content_hash !== null &&
      (typeof value.content_hash !== "string" ||
        !/^sha256:[0-9a-f]{64}$/.test(value.content_hash))
    ) return null;
    asset.content_hash = value.content_hash;
  } else if (options.includeContentHash !== false && options.completeWireShape) {
    asset.content_hash = null;
  }
  if (expectedKind === "video") {
    const metadata = normalizeVideoMetadata(value.metadata);
    if (!metadata) return null;
    asset.metadata = metadata;
  } else if (options.completeWireShape) {
    if (value.metadata !== undefined && value.metadata !== null) return null;
    asset.metadata = null;
  }
  // Non-video metadata and slot fields are deliberately omitted. The backend
  // accepts metadata only for videos, while plain first/last/source assets are
  // never allowed to acquire a reference slot.
  return asset;
}

export function isStableAssetReference(
  value: unknown,
  expectedKind: AssetKind,
): value is AssetReference {
  return normalizeAssetReference(value, expectedKind) !== null;
}

export function isStableSlottedAssetReference(
  value: unknown,
  expectedKind: AssetKind,
  maxSlot: number,
): value is SlottedAssetReference {
  return normalizeSlottedAssetReference(value, expectedKind, maxSlot) !== null;
}

export function normalizeSlottedAssetReference(
  value: unknown,
  kind: AssetKind,
  maxSlot: number,
  options: {
    includeContentHash?: boolean;
    requireContentHash?: boolean;
    completeWireShape?: boolean;
  } = {},
): SlottedAssetReference | null {
  if (!isRecord(value) || !Number.isInteger(value.slot)) return null;
  const slot = value.slot as number;
  if (slot < 0 || slot > maxSlot) return null;
  const asset = normalizeAssetReference(value, kind, options);
  return asset ? { ...asset, slot } : null;
}

export function normalizeSlottedAssetList(
  values: unknown,
  kind: AssetKind,
  maxSlot: number,
  options: {
    includeContentHash?: boolean;
    requireContentHash?: boolean;
    completeWireShape?: boolean;
  } = {},
): SlottedAssetReference[] {
  if (!Array.isArray(values)) return [];
  const acceptedExplicit = new Map<number, SlottedAssetReference>();
  const used = new Set<number>();
  values.forEach((value, index) => {
    const asset = normalizeSlottedAssetReference(value, kind, maxSlot, options);
    if (!asset || used.has(asset.slot)) return;
    used.add(asset.slot);
    acceptedExplicit.set(index, asset);
  });

  const result: SlottedAssetReference[] = [];
  values.forEach((value, index) => {
    const explicit = acceptedExplicit.get(index);
    if (explicit) {
      result.push(explicit);
      return;
    }
    // Migrate pre-slot local drafts while preserving all explicit server slots.
    if (!isRecord(value) || value.slot !== undefined) return;
    const asset = normalizeAssetReference(value, kind, options);
    if (!asset) return;
    let slot = 0;
    while (slot <= maxSlot && used.has(slot)) slot += 1;
    if (slot > maxSlot) return;
    used.add(slot);
    result.push({ ...asset, slot });
  });
  return result;
}

/** Assigns new uploads to the lowest free slots without renumbering old refs. */
export function appendToLowestFreeSlots(
  assets: SlottedAssetReference[],
  uploaded: AssetReference[],
  maxItems: number,
): SlottedAssetReference[] {
  const used = new Set(assets.map((asset) => asset.slot));
  const knownIds = new Set(assets.map((asset) => asset.id));
  const next = [...assets];
  for (const asset of uploaded) {
    // A stable asset identity may occupy only one slot per modality. Without
    // this guard the prompt token lookup by asset ID becomes ambiguous.
    if (knownIds.has(asset.id)) continue;
    let slot = 0;
    while (slot < maxItems && used.has(slot)) slot += 1;
    if (slot >= maxItems) break;
    used.add(slot);
    knownIds.add(asset.id);
    next.push({ ...asset, slot });
  }
  return next;
}

export interface SourceVideoRange {
  source_start_seconds: number;
  source_duration_seconds: number;
}

/** Keeps an existing source range when possible and fits it to a new video. */
export function fitSourceRangeToVideo(
  asset: AssetReference,
  sourceStartSeconds: number,
  sourceDurationSeconds: number,
): SourceVideoRange | null {
  const normalized = normalizeAssetReference(asset, "video");
  if (!normalized?.metadata) return null;
  const videoDuration = normalized.metadata.duration;
  const sourceStart =
    Number.isFinite(sourceStartSeconds) &&
    sourceStartSeconds >= 0 &&
    sourceStartSeconds < videoDuration
      ? sourceStartSeconds
      : 0;
  const remaining = videoDuration - sourceStart;
  const sourceDuration =
    Number.isFinite(sourceDurationSeconds) &&
    sourceDurationSeconds > 0 &&
    sourceDurationSeconds <= remaining
      ? sourceDurationSeconds
      : remaining;
  return {
    source_start_seconds: sourceStart,
    source_duration_seconds: sourceDuration,
  };
}

export function sanitizeDraftAssetReferences(draft: T2VDraft): T2VDraft;
export function sanitizeDraftAssetReferences(draft: I2VDraft): I2VDraft;
export function sanitizeDraftAssetReferences(draft: FL2VDraft): FL2VDraft;
export function sanitizeDraftAssetReferences(draft: R2VDraft): R2VDraft;
export function sanitizeDraftAssetReferences(draft: V2VDraft): V2VDraft;
export function sanitizeDraftAssetReferences(draft: RV2VDraft): RV2VDraft;
export function sanitizeDraftAssetReferences(draft: ModeDraft): ModeDraft;
export function sanitizeDraftAssetReferences(draft: ModeDraft): ModeDraft {
  switch (draft.mode) {
    case "t2v":
      return draft;
    case "i2v":
      return {
        ...draft,
        shots: draft.shots.map((shot) => ({
          ...shot,
          first_image: normalizeAssetReference(shot.first_image, "image"),
        })),
      };
    case "fl2v":
      return {
        ...draft,
        shots: draft.shots.map((shot) => ({
          ...shot,
          first_image: normalizeAssetReference(shot.first_image, "image"),
          last_image: normalizeAssetReference(shot.last_image, "image"),
        })),
      };
    case "r2v":
      return {
        ...draft,
        shots: draft.shots.map((shot) => ({
          ...shot,
          reference_images: normalizeSlottedAssetList(
            shot.reference_images,
            "image",
            maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages),
          ),
          reference_audios: normalizeSlottedAssetList(
            shot.reference_audios,
            "audio",
            maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios),
          ),
          reference_videos: normalizeSlottedAssetList(
            shot.reference_videos,
            "video",
            maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos),
          ),
        })),
      };
    case "v2v":
      return {
        ...draft,
        shots: draft.shots.map((shot) => ({
          ...shot,
          source_video: normalizeAssetReference(shot.source_video, "video"),
        })),
      };
    case "rv2v":
      return {
        ...draft,
        shots: draft.shots.map((shot) => ({
          ...shot,
          source_video: normalizeAssetReference(shot.source_video, "video"),
          reference_images: normalizeSlottedAssetList(
            shot.reference_images,
            "image",
            maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages),
          ),
          reference_audios: normalizeSlottedAssetList(
            shot.reference_audios,
            "audio",
            maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios),
          ),
        })),
      };
  }
}
