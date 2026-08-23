import type { TaskProjectSnapshotResponse } from "../api/types";

export interface HistoricalProjectConfigDownload {
  filename: string;
  mimeType: "application/json";
  contents: string;
}

/**
 * Build the portable file from the immutable job creative snapshot only.
 * Runtime settings are deliberately not an input to this projection.
 */
export function buildHistoricalProjectConfigDownload(
  taskId: string,
  snapshot: TaskProjectSnapshotResponse,
): HistoricalProjectConfigDownload {
  if (snapshot.job_id !== taskId) {
    throw new Error("任务来源项目与请求任务不匹配");
  }
  const safeTaskId = taskId.replace(/[^a-zA-Z0-9._-]+/g, "-") || "historical";
  return {
    filename: `director-project-${safeTaskId}.json`,
    mimeType: "application/json",
    contents: `${JSON.stringify(snapshot.project, null, 2)}\n`,
  };
}

export function downloadHistoricalProjectConfig(
  taskId: string,
  snapshot: TaskProjectSnapshotResponse,
): void {
  const download = buildHistoricalProjectConfigDownload(taskId, snapshot);
  const blob = new Blob([download.contents], { type: download.mimeType });
  const createObjectUrl = URL.createObjectURL?.bind(URL);
  const objectUrl = createObjectUrl ? createObjectUrl(blob) : null;
  const url = objectUrl ??
    `data:${download.mimeType};charset=utf-8,${encodeURIComponent(download.contents)}`;
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = download.filename;
  anchor.rel = "noreferrer";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  if (objectUrl !== null) {
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  }
}
