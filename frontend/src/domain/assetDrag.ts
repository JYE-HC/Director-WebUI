import type { AssetKind, AssetReference } from "./modes";
import type { AssetUploadProgress } from "../api/types";
import { inferAssetKindFromFile } from "./assets";

export const DIRECTOR_ASSET_MIME = "application/x-director-asset";
export const DIRECTOR_ASSETS_MIME = "application/x-director-assets";
export const DIRECTOR_SEGMENT_REFERENCE_MIME = "application/x-director-segment-reference";

export type ReorderableSegmentReferenceTarget =
  | "reference_image"
  | "reference_video"
  | "reference_audio";

export interface SegmentReferenceDragPayload {
  segmentId: string;
  assetId: string;
  target: ReorderableSegmentReferenceTarget;
}

/**
 * Decodes the private payload used only for reordering references inside one
 * segment. It deliberately does not fall back to the asset-library MIME type:
 * an internal move must never be mistaken for a new binding.
 */
export function directorSegmentReferenceFromTransfer(
  transfer: Pick<DataTransfer, "getData">,
): SegmentReferenceDragPayload | null {
  const raw = transfer.getData(DIRECTOR_SEGMENT_REFERENCE_MIME);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const candidate = parsed as Partial<SegmentReferenceDragPayload>;
    if (
      typeof candidate.segmentId !== "string" || !candidate.segmentId.trim() ||
      typeof candidate.assetId !== "string" || !candidate.assetId.trim() ||
      !["reference_image", "reference_video", "reference_audio"].includes(
        String(candidate.target),
      )
    ) return null;
    return {
      segmentId: candidate.segmentId,
      assetId: candidate.assetId,
      target: candidate.target as ReorderableSegmentReferenceTarget,
    };
  } catch {
    return null;
  }
}

/**
 * Reads a same-page asset drag without trusting a single, possibly stale ID.
 * The legacy singular payload remains supported for old saved browser tabs.
 */
export function directorAssetIdsFromTransfer(
  transfer: Pick<DataTransfer, "getData">,
): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  const add = (value: unknown) => {
    if (typeof value !== "string") return;
    const id = value.trim();
    if (!id || seen.has(id)) return;
    seen.add(id);
    ids.push(id);
  };

  const batch = transfer.getData(DIRECTOR_ASSETS_MIME);
  if (batch) {
    try {
      const parsed: unknown = JSON.parse(batch);
      if (Array.isArray(parsed)) parsed.forEach(add);
    } catch {
      // Fall through to the singular compatibility payload.
    }
  }
  if (!ids.length) add(transfer.getData(DIRECTOR_ASSET_MIME));
  return ids;
}

export interface ClassifiedDroppedFile {
  file: File;
  kind: AssetKind;
}

export function classifyDroppedFiles(files: Iterable<File>): {
  accepted: ClassifiedDroppedFile[];
  unsupported: File[];
} {
  const accepted: ClassifiedDroppedFile[] = [];
  const unsupported: File[] = [];
  for (const file of files) {
    const kind = inferAssetKindFromFile(file);
    if (kind) accepted.push({ file, kind });
    else unsupported.push(file);
  }
  return { accepted, unsupported };
}

export interface DroppedUploadFailure {
  file_name: string;
  message: string;
}

export interface DroppedUploadResult {
  assets: AssetReference[];
  failures: DroppedUploadFailure[];
  authority_stale: boolean;
}

export interface DroppedUploadProgress extends AssetUploadProgress {
  file_name: string;
  file_index: number;
  total_files: number;
  completed_files: number;
}

export function describeUploadProgress(progress: DroppedUploadProgress): string {
  const prefix = progress.total_files > 1
    ? `${progress.completed_files}/${progress.total_files} · ${progress.file_name}`
    : progress.file_name;
  const detail = progress.stage === "queued"
    ? "等待上传"
    : progress.stage === "uploading"
      ? `正在上传${progress.percent === undefined ? "" : ` ${progress.percent}%`}`
      : progress.stage === "processing"
        ? "上传完成，服务器处理中"
        : progress.stage === "analyzing"
          ? "正在分析并标准化视频"
          : progress.stage === "forwarding"
            ? progress.strategy === "remux"
              ? "快速整理完成，正在发送到 ComfyUI"
              : progress.strategy === "transcode"
                ? "转码完成，正在发送到 ComfyUI"
                : "正在发送到 ComfyUI"
            : progress.stage === "complete"
              ? "导入完成"
              : "导入失败";
  return `${prefix}：${detail}`;
}

/**
 * Uploads every supported file so one bad item cannot discard earlier or later
 * successes. Responses from an obsolete endpoint/settings generation are never
 * allowed to enter the current asset library.
 */
export async function uploadClassifiedDroppedFiles(
  entries: readonly ClassifiedDroppedFile[],
  upload: (
    file: File,
    kind: AssetKind,
    onProgress: (progress: AssetUploadProgress) => void,
  ) => Promise<AssetReference>,
  authorityCurrent: () => boolean,
  onProgress?: (progress: DroppedUploadProgress) => void,
): Promise<DroppedUploadResult> {
  const assets: Array<AssetReference | undefined> = new Array(entries.length);
  const failures: Array<DroppedUploadFailure | undefined> = new Array(entries.length);
  let nextIndex = 0;
  let completedFiles = 0;
  let authorityStale = false;
  const worker = async () => {
    while (true) {
      if (!authorityCurrent()) {
        authorityStale = true;
        return;
      }
      const index = nextIndex++;
      const entry = entries[index];
      if (!entry) return;
      const report = (progress: AssetUploadProgress) => onProgress?.({
        ...progress,
        file_name: entry.file.name,
        file_index: index,
        total_files: entries.length,
        completed_files: completedFiles,
      });
      report({ stage: "queued" });
      try {
        const asset = await upload(entry.file, entry.kind, report);
        if (!authorityCurrent()) {
          authorityStale = true;
          return;
        }
        assets[index] = asset;
      } catch (reason) {
        if (!authorityCurrent()) {
          authorityStale = true;
          return;
        }
        failures[index] = {
          file_name: entry.file.name,
          message: reason instanceof Error ? reason.message : "素材上传失败",
        };
      } finally {
        completedFiles += 1;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(2, entries.length) }, () => worker()));
  if (authorityStale) return { assets: [], failures: failures.filter(Boolean) as DroppedUploadFailure[], authority_stale: true };
  return {
    assets: assets.filter(Boolean) as AssetReference[],
    failures: failures.filter(Boolean) as DroppedUploadFailure[],
    authority_stale: false,
  };
}
