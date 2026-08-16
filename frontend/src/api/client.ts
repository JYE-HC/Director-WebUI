import {
  isSamplingScheduler,
  type AssetKind,
  type AssetReference,
  type GenerationMode,
  type ModeDraftMap,
} from "../domain/modes";
import {
  normalizeTimelineProject,
  type TimelineGenerationMode,
  type TimelineProject,
} from "../domain/timelineProject";
import { normalizeAssetReference } from "../domain/assets";
import { alignH3FrameCount } from "../domain/timing";
import type {
  CapabilityReport,
  AssetCascadeDeleteResponse,
  AssetDeleteResponse,
  AssetListResponse,
  AssetUploadProgress,
  ConnectionTestResult,
  CreateTaskRequest,
  GenerationTask,
  GPUResource,
  JobClearResponse,
  JobDeleteResponse,
  ModelInventory,
  ProjectDeleteResponse,
  ProjectListResponse,
  ProjectSummary,
  RayLightRuntimeStatus,
  RV2VShotDetectionRequest,
  RV2VShotDetectionResponse,
  RuntimeSettings,
  RuntimeSettingsAuthority,
  StorageConfiguration,
  StorageMigrationResult,
  TaskListResponse,
  TaskBulkCancelResponse,
  TaskDiagnostic,
  TaskGenerationDetails,
  TaskProjectSnapshotResponse,
  TimelineCompileReport,
  TimelineTaskRequest,
} from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");
export const DATABASE_IDENTITY_STALE_EVENT = "director:stale-database-identity";
let latchedDatabaseIdentity: string | null = null;

export function taskEventsUrl(): string {
  return `${API_BASE}/tasks/events`;
}

export type ApiErrorCode = "raylight_recovery_in_flight";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly details?: unknown,
    public readonly code?: ApiErrorCode,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  headers.set("Accept", "application/json");
  const method = (init?.method ?? "GET").toUpperCase();
  if (latchedDatabaseIdentity && method !== "GET" && method !== "HEAD") {
    headers.set("X-Director-Database-Identity", latchedDatabaseIdentity);
  }
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    let responseText = "";
    try {
      responseText = await response.text();
    } catch {
      // The HTTP status is already authoritative even if a broken proxy/body
      // stream prevents reading its diagnostic payload.
      throw new ApiError(`HTTP ${response.status}`, response.status);
    }
    let details: unknown = responseText;
    if (responseText) {
      try {
        details = JSON.parse(responseText) as unknown;
      } catch {
        // Plain-text and HTML errors still retain their HTTP status below.
      }
    }
    if (
      response.status === 409 && isRecord(details) &&
      details.code === "stale_database_identity"
    ) window.dispatchEvent(new Event(DATABASE_IDENTITY_STALE_EVENT));
    const parsed = parseHttpError(details, response.status);
    throw new ApiError(parsed.message, response.status, parsed.details, parsed.code);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function uploadId(): string {
  if (typeof globalThis.crypto?.randomUUID === "function") return globalThis.crypto.randomUUID();
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function parseUploadProgress(value: unknown): AssetUploadProgress {
  if (!isRecord(value) || ![
    "processing", "analyzing", "forwarding", "complete", "failed",
  ].includes(String(value.stage))) {
    throw new ApiError("素材上传进度响应结构无效", 502, value);
  }
  return {
    stage: value.stage as AssetUploadProgress["stage"],
    ...(typeof value.strategy === "string" && ["passthrough", "remux", "transcode"].includes(value.strategy)
      ? { strategy: value.strategy as AssetUploadProgress["strategy"] }
      : {}),
    ...(typeof value.input_bytes === "number" ? { input_bytes: value.input_bytes } : {}),
    ...(typeof value.output_bytes === "number" ? { output_bytes: value.output_bytes } : {}),
    ...(typeof value.elapsed_seconds === "number" ? { elapsed_seconds: value.elapsed_seconds } : {}),
  };
}

function uploadAssetWithProgress(
  file: File,
  kind: AssetKind,
  onProgress: (progress: AssetUploadProgress) => void,
): Promise<AssetReference> {
  const id = uploadId();
  const body = new FormData();
  body.append("file", file);
  body.append("kind", kind);
  body.append("upload_id", id);
  let stopped = false;
  let polling = false;
  let pollTimer: number | null = null;
  const poll = async () => {
    if (stopped || polling) return;
    polling = true;
    try {
      const progress = await request<unknown>(`/uploads/${encodeURIComponent(id)}`);
      if (!stopped) onProgress(parseUploadProgress(progress));
    } catch (reason) {
      if (!(reason instanceof ApiError && reason.status === 404)) return;
    } finally {
      polling = false;
    }
  };
  const startPolling = () => {
    if (stopped || pollTimer !== null) return;
    void poll();
    pollTimer = window.setInterval(() => void poll(), 400);
  };
  const stopPolling = () => {
    stopped = true;
    if (pollTimer !== null) window.clearInterval(pollTimer);
  };
  onProgress({ stage: "uploading", percent: 0 });

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/assets`);
    xhr.setRequestHeader("Accept", "application/json");
    if (latchedDatabaseIdentity) {
      xhr.setRequestHeader("X-Director-Database-Identity", latchedDatabaseIdentity);
    }
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      onProgress({
        stage: "uploading",
        ...(event.lengthComputable
          ? { percent: Math.min(100, Math.round(event.loaded / event.total * 100)) }
          : {}),
      });
    };
    xhr.upload.onload = () => {
      // Upload 100% only means the browser finished sending the request body.
      // The server still has to normalize, forward and register the asset.
      onProgress({ stage: "processing" });
      startPolling();
    };
    xhr.onload = () => {
      stopPolling();
      if (xhr.status < 200 || xhr.status >= 300) {
        const parsed = parseHttpError(xhr.response, xhr.status);
        reject(new ApiError(parsed.message, xhr.status, parsed.details));
        return;
      }
      const payload = xhr.response;
      const assetValue = isRecord(payload) ? payload.asset : undefined;
      const normalized = normalizeAssetReference(assetValue, kind);
      if (!normalized) {
        reject(new ApiError("素材上传响应缺少有效的稳定 ID", 502, assetValue));
        return;
      }
      onProgress({ stage: "complete" });
      resolve(normalized);
    };
    xhr.onerror = () => {
      stopPolling();
      reject(new ApiError("素材上传网络连接失败", 0));
    };
    xhr.onabort = () => {
      stopPolling();
      reject(new ApiError("素材上传已取消", 0));
    };
    xhr.send(body);
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length &&
    actual.every((key, index) => key === expected[index]);
}

function parseRuntimeSettingsAuthority(value: unknown): RuntimeSettingsAuthority {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["settings", "authority_token"]) ||
    !isRecord(value.settings) ||
    typeof value.authority_token !== "string" ||
    !/^[0-9a-f]{64}$/.test(value.authority_token)
  ) throw new ApiError("运行设置权威响应结构无效", 502, value);
  return {
    settings: value.settings as unknown as RuntimeSettings,
    authority_token: value.authority_token,
  };
}

function runtimeAuthorityHeaders(authorityToken: string): HeadersInit {
  if (!/^[0-9a-f]{64}$/.test(authorityToken)) {
    throw new Error("运行设置权威 token 无效");
  }
  return { "X-Director-Runtime-Authority": authorityToken };
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === "string");
}

function parseLogicalGpuIndexes(value: unknown): number[] | null {
  if (
    !Array.isArray(value) ||
    !value.every((entry) => isNonNegativeInteger(entry)) ||
    new Set(value).size !== value.length
  ) return null;
  return [...value];
}

function parseRayLightRuntimeStatus(value: unknown): RayLightRuntimeStatus {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "active",
      "recovery_required",
      "epoch",
      "runtime_gpu_indexes",
      "available_gpu_indexes",
      "invalid_gpu_indexes",
      "tainted",
      "recovery_token",
    ]) ||
    typeof value.active !== "boolean" ||
    typeof value.recovery_required !== "boolean" ||
    typeof value.tainted !== "boolean" ||
    !(value.recovery_token === null || (
      typeof value.recovery_token === "string" &&
      /^[0-9a-f]{64}$/.test(value.recovery_token)
    )) ||
    !isNonNegativeInteger(value.epoch)
  ) {
    throw new ApiError("RayLight 运行状态响应结构无效", 502, value);
  }
  const runtime = parseLogicalGpuIndexes(value.runtime_gpu_indexes);
  const available = parseLogicalGpuIndexes(value.available_gpu_indexes);
  const invalid = parseLogicalGpuIndexes(value.invalid_gpu_indexes);
  const expectedInvalid = runtime === null || available === null
    ? null
    : runtime.filter((index) => !available.includes(index));
  if (
    runtime === null ||
    available === null ||
    invalid === null ||
    value.active !== (runtime.length > 0) ||
    value.recovery_required !== (invalid.length > 0) ||
    value.recovery_required !== (value.recovery_token !== null) ||
    expectedInvalid === null ||
    invalid.length !== expectedInvalid.length ||
    invalid.some((index, offset) => index !== expectedInvalid[offset])
  ) {
    throw new ApiError("RayLight 运行状态响应结构无效", 502, value);
  }
  return {
    active: value.active,
    recovery_required: value.recovery_required,
    epoch: value.epoch,
    runtime_gpu_indexes: runtime,
    available_gpu_indexes: available,
    invalid_gpu_indexes: invalid,
    tainted: value.tainted,
    recovery_token: value.recovery_token,
  };
}

/**
 * Exposes only the backend's user-facing error allow-list. In particular, an
 * object detail is never stringified wholesale because it may also contain
 * internal diagnostics that do not belong in a toast or ApiError payload.
 */
function parseHttpError(
  value: unknown,
  status: number,
): { message: string; details?: unknown; code?: ApiErrorCode } {
  const fallback = `请求失败（${status}）`;
  if (!isRecord(value) || !("detail" in value)) return { message: fallback };
  const detail = value.detail;
  if (typeof detail === "string" && detail.trim()) {
    return { message: detail, details: { detail } };
  }
  if (!isRecord(detail)) {
    return { message: fallback };
  }
  const code: ApiErrorCode | undefined = detail.code === "raylight_recovery_in_flight"
    ? detail.code
    : undefined;
  if (typeof detail.message !== "string" || !detail.message.trim()) {
    return code
      ? { message: fallback, code, details: { detail: { code } } }
      : { message: fallback };
  }
  const usages = isStringArray(detail.usages) ? [...detail.usages] : [];
  const safeDetail = {
    ...(code ? { code } : {}),
    message: detail.message,
    ...(usages.length ? { usages } : {}),
  };
  return {
    message: usages.length
      ? `${detail.message}（引用位置：${usages.join("、")}）`
      : detail.message,
    details: { detail: safeDetail },
    ...(code ? { code } : {}),
  };
}

function parseCapabilities(value: unknown): CapabilityReport {
  if (!isRecord(value)) {
    throw new ApiError("ComfyUI 能力响应结构无效", 502, value);
  }
  const connection = ["unknown", "checking", "online", "offline"].includes(
    String(value.connection),
  ) ? value.connection as CapabilityReport["connection"] : "offline";
  const supportedModes = Array.isArray(value.supported_modes)
    ? value.supported_modes.filter(
        (mode): mode is GenerationMode =>
          typeof mode === "string" && ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"].includes(mode),
      )
    : [];
  const native = isRecord(value.native_timeline) ? value.native_timeline : null;
  const backends = isRecord(value.execution_backends) ? value.execution_backends : null;
  const parseBackend = (name: "standard" | "raylight") => {
    const backend = backends && isRecord(backends[name]) ? backends[name] : null;
    const conditional = backend && isRecord(backend.conditional_requirements)
      ? backend.conditional_requirements
      : null;
    const lora = conditional && isRecord(conditional.lora) ? conditional.lora : null;
    return {
      available: backend?.available === true,
      missing_nodes: isStringArray(backend?.missing_nodes) ? [...backend.missing_nodes] : [],
      ...(lora
        ? {
            conditional_requirements: {
              lora: {
                available: lora.available === true,
                missing_nodes: isStringArray(lora.missing_nodes) ? [...lora.missing_nodes] : [],
              },
            },
          }
        : {}),
    };
  };
  return {
    connection,
    supported_modes: supportedModes,
    // Missing or malformed cancel capability is deliberately fail-closed.
    supports_cancel: value.supports_cancel === true,
    available_nodes: isStringArray(value.available_nodes) ? [...value.available_nodes] : [],
    missing_nodes: isStringArray(value.missing_nodes) ? [...value.missing_nodes] : [],
    ...(typeof value.message === "string" ? { message: value.message } : {}),
    ...(typeof value.latency_ms === "number" && Number.isFinite(value.latency_ms)
      ? { latency_ms: value.latency_ms }
      : {}),
    ...(isRecord(value.features) ? { features: { ...value.features } } : {}),
    ...(native
      ? {
          native_timeline: {
            supported: native.supported === true,
            modes: Array.isArray(native.modes)
              ? native.modes.filter(
                  (mode): mode is TimelineGenerationMode =>
                    typeof mode === "string" && ["fl2va", "ref2va"].includes(mode),
                )
              : [],
            // A stale or malformed endpoint must not unlock an editor control
            // for a native timeline implementation it simultaneously marks
            // unavailable.
            continuity: native.supported === true && native.continuity === true,
          },
        }
      : {}),
    execution_backends: {
      standard: parseBackend("standard"),
      raylight: parseBackend("raylight"),
    },
  };
}

const COMPILE_PROVENANCE = new Set([
  "comfy-core",
  "comfy-core-official-minimax-h3",
  "comfy-extras",
  "raylight",
  "lora-custom",
]);

/**
 * The browser may inspect a server-redacted plan, never an executable prompt.
 * Rebuilding an exact allow-listed response also prevents a compromised or
 * outdated endpoint from smuggling raw workflow data into UI state/devtools.
 */
function parseTimelineCompileReport(value: unknown): TimelineCompileReport {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["execution_strategy", "model_families", "plans", "node_policy"]) ||
    value.execution_strategy !== "native_segment_graph_v1" ||
    !Array.isArray(value.model_families) ||
    !value.model_families.every((family) => family === "fl2va" || family === "ref2va") ||
    !Array.isArray(value.plans) ||
    value.plans.length === 0 ||
    !isRecord(value.node_policy)
  ) {
    throw new ApiError("执行计划响应结构无效", 502, value);
  }

  const plans = value.plans.map((plan) => {
    if (
      !isRecord(plan) ||
      !hasExactKeys(plan, [
        "segment_id",
        "mode",
        "recipe",
        "model_family",
        "backend",
        "frame_count",
        "visible_frame_count",
        "sample_frame_count",
        "continuity_context_frames",
        "alignment_tail_frame_count",
        "predecessor_segment_id",
        "continuity_source",
        "historical_take_id",
        "anchor_reset",
        "seed_mode",
        "seed",
        "conditioning_node",
        "node_classes",
      ]) ||
      typeof plan.segment_id !== "string" ||
      !["fl2va", "ref2va"].includes(String(plan.mode)) ||
      !["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"].includes(String(plan.recipe)) ||
      (plan.model_family !== "fl2va" && plan.model_family !== "ref2va") ||
      plan.mode !== plan.model_family ||
      (plan.mode === "fl2va" && !["t2v", "i2v", "fl2v"].includes(String(plan.recipe))) ||
      (plan.mode === "ref2va" && !["r2v", "v2v", "rv2v"].includes(String(plan.recipe))) ||
      (plan.backend !== "standard" && plan.backend !== "raylight") ||
      typeof plan.frame_count !== "number" ||
      !Number.isInteger(plan.frame_count) ||
      plan.frame_count < 5 ||
      typeof plan.visible_frame_count !== "number" ||
      !Number.isInteger(plan.visible_frame_count) ||
      plan.visible_frame_count < 5 ||
      plan.frame_count !== plan.visible_frame_count ||
      typeof plan.sample_frame_count !== "number" ||
      !Number.isInteger(plan.sample_frame_count) ||
      plan.sample_frame_count < 5 ||
      plan.sample_frame_count > 512 ||
      alignH3FrameCount(plan.visible_frame_count) !== plan.visible_frame_count ||
      alignH3FrameCount(plan.sample_frame_count) !== plan.sample_frame_count ||
      typeof plan.continuity_context_frames !== "number" ||
      ![0, 5, 22, 39, 56].includes(plan.continuity_context_frames) ||
      typeof plan.alignment_tail_frame_count !== "number" ||
      !Number.isInteger(plan.alignment_tail_frame_count) ||
      plan.alignment_tail_frame_count < 0 ||
      plan.alignment_tail_frame_count > 16 ||
      plan.sample_frame_count !== plan.visible_frame_count + plan.continuity_context_frames + plan.alignment_tail_frame_count ||
      plan.alignment_tail_frame_count !== alignH3FrameCount(
        plan.visible_frame_count + plan.continuity_context_frames,
      ) - plan.visible_frame_count - plan.continuity_context_frames ||
      (plan.predecessor_segment_id !== null && typeof plan.predecessor_segment_id !== "string") ||
      (typeof plan.predecessor_segment_id === "string" && plan.predecessor_segment_id.length === 0) ||
      ![null, "same_run", "historical_take"].includes(plan.continuity_source as null | string) ||
      (plan.historical_take_id !== null &&
        (typeof plan.historical_take_id !== "string" || plan.historical_take_id.trim().length === 0)) ||
      plan.segment_id.length === 0 ||
      typeof plan.anchor_reset !== "boolean" ||
      (plan.predecessor_segment_id === null && plan.continuity_context_frames !== 0) ||
      (typeof plan.predecessor_segment_id === "string" && (plan.continuity_context_frames === 0 || plan.anchor_reset)) ||
      (plan.anchor_reset && plan.predecessor_segment_id !== null) ||
      (plan.continuity_source === null &&
        (plan.predecessor_segment_id !== null || plan.historical_take_id !== null)) ||
      (plan.continuity_source === "same_run" &&
        (typeof plan.predecessor_segment_id !== "string" || plan.historical_take_id !== null)) ||
      (plan.continuity_source === "historical_take" &&
        (typeof plan.predecessor_segment_id !== "string" || typeof plan.historical_take_id !== "string")) ||
      (plan.continuity_context_frames === 0 && plan.continuity_source !== null) ||
      (plan.continuity_context_frames !== 0 && plan.continuity_source === null) ||
      (plan.seed_mode !== "fixed" && plan.seed_mode !== "random") ||
      typeof plan.seed !== "number" ||
      !Number.isSafeInteger(plan.seed) ||
      plan.seed < 0 ||
      (plan.conditioning_node !== "MiniMaxH3ImageToVideo" &&
        plan.conditioning_node !== "MiniMaxH3ReferenceToVideo") ||
      (plan.mode === "fl2va" && plan.conditioning_node !== "MiniMaxH3ImageToVideo") ||
      (plan.mode === "ref2va" && plan.conditioning_node !== "MiniMaxH3ReferenceToVideo") ||
      !isStringArray(plan.node_classes)
    ) {
      throw new ApiError("执行计划分段结构无效", 502, plan);
    }
    const parsed: TimelineCompileReport["plans"][number] = {
      segment_id: plan.segment_id,
      mode: plan.mode as TimelineCompileReport["plans"][number]["mode"],
      recipe: plan.recipe as TimelineCompileReport["plans"][number]["recipe"],
      model_family: plan.model_family as TimelineCompileReport["plans"][number]["model_family"],
      backend: plan.backend as TimelineCompileReport["plans"][number]["backend"],
      frame_count: plan.frame_count,
      visible_frame_count: plan.visible_frame_count,
      sample_frame_count: plan.sample_frame_count,
      continuity_context_frames: plan.continuity_context_frames as TimelineCompileReport["plans"][number]["continuity_context_frames"],
      alignment_tail_frame_count: plan.alignment_tail_frame_count,
      predecessor_segment_id: plan.predecessor_segment_id,
      continuity_source: plan.continuity_source as TimelineCompileReport["plans"][number]["continuity_source"],
      historical_take_id: plan.historical_take_id,
      anchor_reset: plan.anchor_reset,
      seed_mode: plan.seed_mode,
      seed: plan.seed,
      conditioning_node: plan.conditioning_node as TimelineCompileReport["plans"][number]["conditioning_node"],
      node_classes: [...plan.node_classes],
    };
    return parsed;
  });

  const plansById = new Map<string, TimelineCompileReport["plans"][number]>();
  for (const plan of plans) {
    if (plansById.has(plan.segment_id)) {
      throw new ApiError("执行计划包含重复分段", 502, plan);
    }
    if (plan.continuity_source === "same_run") {
      const predecessor = plansById.get(plan.predecessor_segment_id as string);
      if (!predecessor) {
        throw new ApiError("执行计划接续依赖无效", 502, plan);
      }
    }
    plansById.set(plan.segment_id, plan);
  }
  const expectedFamilies = (["fl2va", "ref2va"] as const).filter((family) =>
    plans.some((plan) => plan.model_family === family),
  );
  if (
    value.model_families.length !== expectedFamilies.length ||
    value.model_families.some((family, index) => family !== expectedFamilies[index])
  ) {
    throw new ApiError("执行计划模型族摘要无效", 502, value.model_families);
  }

  const policy = value.node_policy;
  if (
    !hasExactKeys(policy, [
      "graph_source",
      "accepts_client_workflow",
      "allowed_nodes",
      "custom_nodes",
      "provenance",
    ]) ||
    policy.graph_source !== "server" ||
    policy.accepts_client_workflow !== false ||
    !isStringArray(policy.allowed_nodes) ||
    !isStringArray(policy.custom_nodes) ||
    !isRecord(policy.provenance) ||
    !Object.values(policy.provenance).every(
      (entry) => typeof entry === "string" && COMPILE_PROVENANCE.has(entry),
    )
  ) {
    throw new ApiError("执行计划节点策略无效", 502, policy);
  }

  return {
    execution_strategy: "native_segment_graph_v1",
    model_families: [...value.model_families] as TimelineCompileReport["model_families"],
    plans,
    node_policy: {
      graph_source: "server",
      accepts_client_workflow: false,
      allowed_nodes: [...policy.allowed_nodes],
      custom_nodes: [...policy.custom_nodes],
      provenance: { ...policy.provenance } as TimelineCompileReport["node_policy"]["provenance"],
    },
  };
}

function parseJobDeleteResponse(value: unknown, expectedId: string): JobDeleteResponse {
  if (
    !isRecord(value) ||
    value.deleted_job_id !== expectedId ||
    value.outputs_preserved !== true
  ) {
    throw new ApiError("任务删除响应结构无效", 502, value);
  }
  return {
    deleted_job_id: expectedId,
    outputs_preserved: true,
  };
}

const TASK_STATUSES = new Set([
  "queued", "preparing", "running", "succeeded", "failed", "cancelling", "cancelled",
]);
const TASK_MODES = new Set(["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v", "timeline"]);

/**
 * Rebuilds the exact public JobRead projection. Unknown fields are rejected,
 * so an outdated or compromised endpoint cannot send executable workflow or
 * prompt graphs into browser state through a job response.
 */
function parseGenerationTask(value: unknown): GenerationTask {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "id", "mode", "status", "display_name", "project_title", "project_id",
      "current_project", "progress", "stage", "prompt_id", "outputs",
      "output_files", "error", "preview_url", "created_at", "updated_at",
      "started_at", "completed_at", "execution_duration_seconds", "output_count",
      "error_summary", "children", "segment_results", "live_preview_url",
    ]) ||
    typeof value.id !== "string" || !value.id ||
    !TASK_MODES.has(String(value.mode)) ||
    !TASK_STATUSES.has(String(value.status)) ||
    typeof value.display_name !== "string" || !value.display_name ||
    !(value.project_title === null || typeof value.project_title === "string") ||
    !(value.project_id === null || typeof value.project_id === "string") ||
    typeof value.current_project !== "boolean" ||
    typeof value.progress !== "number" || !Number.isFinite(value.progress) ||
    value.progress < 0 || value.progress > 1 ||
    !(value.stage === null || typeof value.stage === "string") ||
    !(value.prompt_id === null || typeof value.prompt_id === "string") ||
    !isStringArray(value.outputs) || !isStringArray(value.output_files) ||
    !(value.error === null || typeof value.error === "string") ||
    !(value.preview_url === null || typeof value.preview_url === "string") ||
    typeof value.created_at !== "string" || typeof value.updated_at !== "string" ||
    !(value.started_at === null || typeof value.started_at === "string") ||
    !(value.completed_at === null || typeof value.completed_at === "string") ||
    !(value.execution_duration_seconds === null ||
      (typeof value.execution_duration_seconds === "number" &&
        Number.isFinite(value.execution_duration_seconds) &&
        value.execution_duration_seconds >= 0)) ||
    !isNonNegativeInteger(value.output_count) ||
    !(value.error_summary === null || typeof value.error_summary === "string") ||
    !Array.isArray(value.children) || !Array.isArray(value.segment_results) ||
    !(value.live_preview_url === null || typeof value.live_preview_url === "string")
  ) throw new ApiError("任务响应结构无效", 502, value);

  const children: GenerationTask["children"] = value.children.map((child) => {
    if (
      !isRecord(child) ||
      !hasExactKeys(child, [
        "id", "family", "backend", "segment_ids", "status", "progress",
        "stage", "prompt_id", "outputs", "error",
      ]) ||
      typeof child.id !== "string" || !child.id ||
      (child.family !== "fl2va" && child.family !== "ref2va") ||
      (child.backend !== "standard" && child.backend !== "raylight") ||
      !isStringArray(child.segment_ids) ||
      !TASK_STATUSES.has(String(child.status)) ||
      typeof child.progress !== "number" || !Number.isFinite(child.progress) ||
      child.progress < 0 || child.progress > 1 ||
      !(child.stage === null || typeof child.stage === "string") ||
      !(child.prompt_id === null || typeof child.prompt_id === "string") ||
      !isStringArray(child.outputs) ||
      !(child.error === null || typeof child.error === "string")
    ) throw new ApiError("任务子单元响应结构无效", 502, child);
    return {
      id: child.id,
      family: child.family as GenerationTask["children"][number]["family"],
      backend: child.backend as GenerationTask["children"][number]["backend"],
      segment_ids: [...child.segment_ids],
      status: child.status as GenerationTask["children"][number]["status"],
      progress: child.progress,
      stage: child.stage,
      prompt_id: child.prompt_id,
      outputs: [...child.outputs],
      error: child.error,
    };
  });
  const segmentResults = value.segment_results.map((result) => {
    if (
      !isRecord(result) ||
      !hasExactKeys(result, [
        "segment_id", "child_id", "output_url", "output_file", "current_snapshot",
      ]) ||
      ![result.segment_id, result.child_id, result.output_url, result.output_file].every(
        (entry) => typeof entry === "string" && entry.length > 0,
      ) ||
      typeof result.current_snapshot !== "boolean"
    ) throw new ApiError("任务分段候选响应结构无效", 502, result);
    return {
      segment_id: result.segment_id as string,
      child_id: result.child_id as string,
      output_url: result.output_url as string,
      output_file: result.output_file as string,
      current_snapshot: result.current_snapshot as boolean,
    };
  });
  return {
    id: value.id,
    mode: value.mode as GenerationTask["mode"],
    status: value.status as GenerationTask["status"],
    display_name: value.display_name,
    project_title: value.project_title,
    project_id: value.project_id,
    current_project: value.current_project,
    progress: value.progress,
    stage: value.stage,
    prompt_id: value.prompt_id,
    outputs: [...value.outputs],
    output_files: [...value.output_files],
    error: value.error,
    preview_url: value.preview_url,
    created_at: value.created_at,
    updated_at: value.updated_at,
    started_at: value.started_at,
    completed_at: value.completed_at,
    execution_duration_seconds: value.execution_duration_seconds,
    output_count: value.output_count,
    error_summary: value.error_summary,
    children,
    segment_results: segmentResults,
    live_preview_url: value.live_preview_url,
  };
}

function parseTaskList(value: unknown): TaskListResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["jobs", "total", "limit", "offset", "has_more", "summary"]) ||
    !Array.isArray(value.jobs) ||
    !isNonNegativeInteger(value.total) ||
    !isNonNegativeInteger(value.limit) || value.limit < 1 ||
    !isNonNegativeInteger(value.offset) ||
    typeof value.has_more !== "boolean" ||
    !isRecord(value.summary) ||
    !hasExactKeys(value.summary, [
      "total", "active", "queued", "preparing", "running", "cancelling",
      "succeeded", "failed", "cancelled",
    ]) ||
    !Object.values(value.summary).every(isNonNegativeInteger)
  )
    throw new ApiError("任务列表响应结构无效", 502, value);
  return {
    jobs: value.jobs.map(parseGenerationTask),
    total: value.total,
    limit: value.limit,
    offset: value.offset,
    has_more: value.has_more,
    summary: value.summary as unknown as TaskListResponse["summary"],
  };
}

function parseTaskBulkCancel(value: unknown): TaskBulkCancelResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["jobs", "requested_count", "terminal_count"]) ||
    !Array.isArray(value.jobs) ||
    !isNonNegativeInteger(value.requested_count) || value.requested_count < 1 ||
    !isNonNegativeInteger(value.terminal_count) ||
    value.terminal_count > value.requested_count ||
    value.jobs.length !== value.requested_count
  ) throw new ApiError("批量取消响应结构无效", 502, value);
  return {
    jobs: value.jobs.map(parseGenerationTask),
    requested_count: value.requested_count,
    terminal_count: value.terminal_count,
  };
}

function parseTaskProjectSnapshot(value: unknown): TaskProjectSnapshotResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["job_id", "project", "segment_ids"]) ||
    typeof value.job_id !== "string" || !value.job_id ||
    !(value.segment_ids === null || isStringArray(value.segment_ids))
  ) throw new ApiError("任务来源项目响应结构无效", 502, value);
  const project = normalizeTimelineProject(value.project);
  if (!project) throw new ApiError("任务来源项目结构无效", 502, value);
  return {
    job_id: value.job_id,
    project,
    segment_ids: value.segment_ids === null ? null : [...value.segment_ids],
  };
}

function parseTimelineProjectResponse(value: unknown): TimelineProject {
  const sampling = isRecord(value) &&
    (value.version === 2 || value.version === 3 || value.version === 4) &&
    isRecord(value.sampling)
    ? value.sampling
    : null;
  const fl2va = sampling && isRecord(sampling.fl2va) ? sampling.fl2va : null;
  const ref2va = sampling && isRecord(sampling.ref2va) ? sampling.ref2va : null;
  if (
    !fl2va ||
    !ref2va ||
    !isSamplingScheduler(fl2va.scheduler) ||
    !isSamplingScheduler(ref2va.scheduler)
  ) throw new ApiError("时间线响应结构无效", 502, value);
  const project = normalizeTimelineProject(value);
  if (!project) throw new ApiError("时间线响应结构无效", 502, value);
  return project;
}

function parseProjectSummary(value: unknown): ProjectSummary {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["id", "title", "created_at", "updated_at", "segment_count"]) ||
    typeof value.id !== "string" || !value.id ||
    typeof value.title !== "string" ||
    typeof value.created_at !== "string" ||
    typeof value.updated_at !== "string" ||
    !isNonNegativeInteger(value.segment_count) || value.segment_count < 1
  ) throw new ApiError("项目摘要响应结构无效", 502, value);
  return {
    id: value.id,
    title: value.title,
    created_at: value.created_at,
    updated_at: value.updated_at,
    segment_count: value.segment_count,
  };
}

function parseProjectList(value: unknown): ProjectListResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["projects"]) ||
    !Array.isArray(value.projects)
  ) throw new ApiError("项目列表响应结构无效", 502, value);
  return { projects: value.projects.map(parseProjectSummary) };
}

function parseProjectDelete(value: unknown, expectedId: string): ProjectDeleteResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["deleted_project_id", "outputs_preserved", "orphaned_jobs"]) ||
    value.deleted_project_id !== expectedId ||
    value.outputs_preserved !== true ||
    !isNonNegativeInteger(value.orphaned_jobs)
  ) throw new ApiError("项目删除响应结构无效", 502, value);
  return {
    deleted_project_id: value.deleted_project_id as string,
    outputs_preserved: true,
    orphaned_jobs: value.orphaned_jobs as number,
  };
}

function isStoragePath(value: unknown, allowHomeRelative = false): value is string {
  return typeof value === "string" &&
    value.length > 1 &&
    value.length <= 4096 &&
    !/[\u0000-\u001f\u007f]/.test(value) &&
    (value.startsWith("/") || (allowHomeRelative && value.startsWith("~/")));
}

const STORAGE_CONFIGURATION_SOURCES = new Set([
  "explicit",
  "environment",
  "bootstrap",
  "legacy",
  "default",
]);

function hasStorageConfigurationFields(value: Record<string, unknown>): boolean {
  return isStoragePath(value.active_database_path) &&
    typeof value.active_database_identity === "string" &&
    /^[0-9a-f]{64}$/.test(value.active_database_identity) &&
    isStoragePath(value.configured_database_path, true) &&
    isStoragePath(value.recommended_database_path) &&
    typeof value.source === "string" &&
    STORAGE_CONFIGURATION_SOURCES.has(value.source) &&
    typeof value.restart_required === "boolean";
}

function parseStorageConfiguration(value: unknown): StorageConfiguration {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "active_database_path",
      "active_database_identity",
      "configured_database_path",
      "recommended_database_path",
      "source",
      "restart_required",
    ]) ||
    !hasStorageConfigurationFields(value)
  ) throw new ApiError("数据存储响应结构无效", 502, value);
  return {
    active_database_path: value.active_database_path as string,
    active_database_identity: value.active_database_identity as string,
    configured_database_path: value.configured_database_path as string,
    recommended_database_path: value.recommended_database_path as string,
    source: value.source as StorageConfiguration["source"],
    restart_required: value.restart_required as boolean,
  };
}

function parseStorageMigrationResult(value: unknown): StorageMigrationResult {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "active_database_path",
      "active_database_identity",
      "configured_database_path",
      "recommended_database_path",
      "source",
      "restart_required",
      "migrated_from",
      "migrated_to",
    ]) ||
    !hasStorageConfigurationFields(value) ||
    !isStoragePath(value.migrated_from) ||
    !isStoragePath(value.migrated_to)
  ) throw new ApiError("数据库迁移响应结构无效", 502, value);
  return {
    active_database_path: value.active_database_path as string,
    active_database_identity: value.active_database_identity as string,
    configured_database_path: value.configured_database_path as string,
    recommended_database_path: value.recommended_database_path as string,
    source: value.source as StorageConfiguration["source"],
    restart_required: value.restart_required as boolean,
    migrated_from: value.migrated_from as string,
    migrated_to: value.migrated_to as string,
  };
}

function parseTaskDiagnostic(value: unknown): TaskDiagnostic {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "schema_version", "id", "display_name", "project_title", "mode", "status",
      "progress", "stage", "created_at", "updated_at", "started_at", "completed_at",
      "execution_duration_seconds", "output_files", "error_summary", "children",
      "settings_included", "workflow_included",
    ]) ||
    value.schema_version !== 1 ||
    typeof value.id !== "string" || !value.id ||
    typeof value.display_name !== "string" || !value.display_name ||
    !(value.project_title === null || typeof value.project_title === "string") ||
    !TASK_MODES.has(String(value.mode)) ||
    !TASK_STATUSES.has(String(value.status)) ||
    typeof value.progress !== "number" || !Number.isFinite(value.progress) ||
    value.progress < 0 || value.progress > 1 ||
    !(value.stage === null || typeof value.stage === "string") ||
    typeof value.created_at !== "string" || typeof value.updated_at !== "string" ||
    !(value.started_at === null || typeof value.started_at === "string") ||
    !(value.completed_at === null || typeof value.completed_at === "string") ||
    !(value.execution_duration_seconds === null ||
      (typeof value.execution_duration_seconds === "number" &&
        Number.isFinite(value.execution_duration_seconds) &&
        value.execution_duration_seconds >= 0)) ||
    !isStringArray(value.output_files) ||
    !(value.error_summary === null || typeof value.error_summary === "string") ||
    !Array.isArray(value.children) ||
    value.settings_included !== false || value.workflow_included !== false
  ) throw new ApiError("任务诊断响应结构无效", 502, value);

  const children = value.children.map((child) => {
    if (
      !isRecord(child) ||
      !hasExactKeys(child, [
        "id", "family", "backend", "segment_ids", "status", "progress", "stage",
        "output_files", "error_summary",
      ]) ||
      typeof child.id !== "string" || !child.id ||
      (child.family !== "fl2va" && child.family !== "ref2va") ||
      (child.backend !== "standard" && child.backend !== "raylight") ||
      !isStringArray(child.segment_ids) ||
      !TASK_STATUSES.has(String(child.status)) ||
      typeof child.progress !== "number" || !Number.isFinite(child.progress) ||
      child.progress < 0 || child.progress > 1 ||
      !(child.stage === null || typeof child.stage === "string") ||
      !isStringArray(child.output_files) ||
      !(child.error_summary === null || typeof child.error_summary === "string")
    ) throw new ApiError("任务诊断子单元结构无效", 502, child);
    return {
      id: child.id,
      family: child.family,
      backend: child.backend,
      segment_ids: [...child.segment_ids],
      status: child.status,
      progress: child.progress,
      stage: child.stage,
      output_files: [...child.output_files],
      error_summary: child.error_summary,
    } as TaskDiagnostic["children"][number];
  });
  return {
    schema_version: 1,
    id: value.id,
    display_name: value.display_name,
    project_title: value.project_title,
    mode: value.mode as TaskDiagnostic["mode"],
    status: value.status as TaskDiagnostic["status"],
    progress: value.progress,
    stage: value.stage,
    created_at: value.created_at,
    updated_at: value.updated_at,
    started_at: value.started_at,
    completed_at: value.completed_at,
    execution_duration_seconds: value.execution_duration_seconds,
    output_files: [...value.output_files],
    error_summary: value.error_summary,
    children,
    settings_included: false,
    workflow_included: false,
  };
}

function parseTaskGenerationDetails(value: unknown): TaskGenerationDetails {
  const families = new Set(["fl2va", "ref2va"]);
  const backends = new Set(["standard", "raylight"]);
  const recipes = new Set(["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"]);
  const finite = (candidate: unknown): candidate is number =>
    typeof candidate === "number" && Number.isFinite(candidate);
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "schema_version", "job_id", "project_title", "render", "sampling",
      "models", "shared_models", "runtime_snapshot_available", "segments",
    ]) ||
    value.schema_version !== 2 ||
    typeof value.job_id !== "string" || !value.job_id ||
    typeof value.project_title !== "string" || !value.project_title ||
    !isRecord(value.render) ||
    !hasExactKeys(value.render, [
      "width", "height", "fps", "export_mode",
      "total_duration_seconds",
    ]) ||
    !isNonNegativeInteger(value.render.width) || value.render.width <= 0 ||
    !isNonNegativeInteger(value.render.height) || value.render.height <= 0 ||
    !finite(value.render.fps) || value.render.fps <= 0 ||
    !["all", "segments"].includes(String(value.render.export_mode)) ||
    !finite(value.render.total_duration_seconds) || value.render.total_duration_seconds <= 0 ||
    !Array.isArray(value.sampling) || !Array.isArray(value.models) ||
    !Array.isArray(value.shared_models) || !Array.isArray(value.segments) ||
    typeof value.runtime_snapshot_available !== "boolean"
  ) throw new ApiError("任务生成参数响应结构无效", 502, value);

  const sampling = value.sampling.map((item) => {
    if (
      !isRecord(item) ||
      !hasExactKeys(item, [
        "family", "steps", "seed", "random_seed", "sampler", "scheduler",
        "shift", "audio_shift",
      ]) ||
      !families.has(String(item.family)) ||
      !isNonNegativeInteger(item.steps) || item.steps <= 0 ||
      !isNonNegativeInteger(item.seed) ||
      typeof item.random_seed !== "boolean" ||
      !["res_multistep", "euler", "dpmpp_2m"].includes(String(item.sampler)) ||
      !isSamplingScheduler(item.scheduler) ||
      !finite(item.shift) || item.shift <= 0 ||
      !finite(item.audio_shift) || item.audio_shift <= 0
    ) throw new ApiError("任务采样参数结构无效", 502, item);
    return { ...item } as TaskGenerationDetails["sampling"][number];
  });
  const models = value.models.map((item) => {
    if (
      !isRecord(item) ||
      !hasExactKeys(item, [
        "family", "filename", "device", "lora_name", "lora_strength", "backends",
        "logical_gpu_indices", "ulysses_degree", "ring_degree",
      ]) ||
      !families.has(String(item.family)) ||
      typeof item.filename !== "string" || !item.filename ||
      typeof item.device !== "string" || !item.device ||
      !(item.lora_name === null || (typeof item.lora_name === "string" && item.lora_name)) ||
      !finite(item.lora_strength) ||
      !Array.isArray(item.backends) || item.backends.length === 0 ||
      !item.backends.every((backend) => backends.has(String(backend))) ||
      !Array.isArray(item.logical_gpu_indices) ||
      !item.logical_gpu_indices.every(isNonNegativeInteger) ||
      !(item.ulysses_degree === null || isNonNegativeInteger(item.ulysses_degree)) ||
      !(item.ring_degree === null || isNonNegativeInteger(item.ring_degree))
    ) throw new ApiError("任务模型参数结构无效", 502, item);
    return {
      ...item,
      backends: [...item.backends],
      logical_gpu_indices: [...item.logical_gpu_indices],
    } as TaskGenerationDetails["models"][number];
  });
  const sharedModels = value.shared_models.map((item) => {
    if (
      !isRecord(item) ||
      !hasExactKeys(item, ["role", "filename", "device"]) ||
      !["clip", "video_vae", "audio_vae"].includes(String(item.role)) ||
      typeof item.filename !== "string" || !item.filename ||
      typeof item.device !== "string" || !item.device
    ) throw new ApiError("任务共享模型参数结构无效", 502, item);
    return { ...item } as TaskGenerationDetails["shared_models"][number];
  });
  const segments = value.segments.map((item) => {
    if (
      !isRecord(item) ||
      !hasExactKeys(item, [
        "id", "title", "family", "recipe", "duration_seconds", "prompt",
        "continuity_enabled", "continuity_overlap_frames", "ref_image_size", "audio_mode", "has_first_image",
        "has_last_image", "has_source_video", "source_audio_as_reference",
        "reference_image_count", "reference_audio_count", "reference_video_count",
      ]) ||
      typeof item.id !== "string" || !item.id ||
      typeof item.title !== "string" ||
      !families.has(String(item.family)) || !recipes.has(String(item.recipe)) ||
      !finite(item.duration_seconds) || item.duration_seconds <= 0 ||
      typeof item.prompt !== "string" ||
      typeof item.continuity_enabled !== "boolean" ||
      ![5, 22, 39, 56].includes(Number(item.continuity_overlap_frames)) ||
      !["match", "max"].includes(String(item.ref_image_size)) ||
      !["generate", "source", "mute"].includes(String(item.audio_mode)) ||
      typeof item.has_first_image !== "boolean" ||
      typeof item.has_last_image !== "boolean" ||
      typeof item.has_source_video !== "boolean" ||
      typeof item.source_audio_as_reference !== "boolean" ||
      !isNonNegativeInteger(item.reference_image_count) ||
      !isNonNegativeInteger(item.reference_audio_count) ||
      !isNonNegativeInteger(item.reference_video_count)
    ) throw new ApiError("任务分段参数结构无效", 502, item);
    return { ...item } as TaskGenerationDetails["segments"][number];
  });
  return {
    schema_version: 2,
    job_id: value.job_id,
    project_title: value.project_title,
    render: { ...value.render } as TaskGenerationDetails["render"],
    sampling,
    models,
    shared_models: sharedModels,
    runtime_snapshot_available: value.runtime_snapshot_available,
    segments,
  };
}

function parseAssetCascadeDeleteResponse(
  value: unknown,
  expectedId: string,
): AssetCascadeDeleteResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["deleted_asset_id", "outputs_preserved", "unbound_usages"]) ||
    value.deleted_asset_id !== expectedId ||
    value.outputs_preserved !== true ||
    !isStringArray(value.unbound_usages)
  ) {
    throw new ApiError("素材移出并解除引用响应结构无效", 502, value);
  }
  return {
    deleted_asset_id: expectedId,
    outputs_preserved: true,
    unbound_usages: [...value.unbound_usages],
  };
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function parseJobClearResponse(value: unknown): JobClearResponse {
  if (
    !isRecord(value) ||
    !isNonNegativeInteger(value.deleted_count) ||
    !isNonNegativeInteger(value.active_count) ||
    value.outputs_preserved !== true
  ) {
    throw new ApiError("任务清理响应结构无效", 502, value);
  }
  return {
    deleted_count: value.deleted_count,
    active_count: value.active_count,
    outputs_preserved: true,
  };
}

export const directorApi = {
  latchDatabaseIdentity(identity: string): string {
    if (!/^[0-9a-f]{64}$/.test(identity)) throw new Error("数据库身份无效");
    latchedDatabaseIdentity ??= identity;
    return latchedDatabaseIdentity;
  },
  /** Test isolation only; production code must never clear a live page latch. */
  resetDatabaseIdentityForTests(): void {
    latchedDatabaseIdentity = null;
  },
  getCapabilities: (signal: AbortSignal | undefined, authorityToken: string) =>
    request<unknown>("/capabilities", {
      signal,
      headers: runtimeAuthorityHeaders(authorityToken),
    }).then(parseCapabilities),
  getGpus: (signal: AbortSignal | undefined, authorityToken: string) =>
    request<{ gpus: GPUResource[] }>("/gpus", {
      signal,
      headers: runtimeAuthorityHeaders(authorityToken),
    }).then(({ gpus }) => gpus),
  getRayLightRuntimeStatus: (signal: AbortSignal | undefined, authorityToken: string) =>
    request<unknown>("/raylight/runtime", {
      signal,
      headers: runtimeAuthorityHeaders(authorityToken),
    }).then(parseRayLightRuntimeStatus),
  getModels: (signal: AbortSignal | undefined, authorityToken: string) =>
    request<ModelInventory>("/models", {
      signal,
      headers: runtimeAuthorityHeaders(authorityToken),
    }),
  getSettings: (signal?: AbortSignal) => request<RuntimeSettings>("/settings", { signal }),
  getSettingsAuthority: (signal?: AbortSignal) =>
    request<unknown>("/settings/authority", { signal }).then(parseRuntimeSettingsAuthority),
  updateSettings: (settings: RuntimeSettings) =>
    request<RuntimeSettings>("/settings", { method: "PUT", body: JSON.stringify(settings) }),
  getStorage: (signal?: AbortSignal) =>
    request<unknown>("/storage", { signal }).then(parseStorageConfiguration),
  updateStorage: (databasePath: string) =>
    request<unknown>("/storage", {
      method: "PUT",
      body: JSON.stringify({ database_path: databasePath }),
    }).then(parseStorageConfiguration),
  migrateStorage: (targetPath: string) =>
    request<unknown>("/storage/migrate", {
      method: "POST",
      body: JSON.stringify({ target_path: targetPath }),
    }).then(parseStorageMigrationResult),
  testConnection: (comfyUrl: string) =>
    request<ConnectionTestResult>("/capabilities", {
      method: "POST",
      body: JSON.stringify({ comfy_url: comfyUrl }),
    }),
  confirmRayLightRuntimeRecovery: (
    expectedComfyOrigin: string,
    expectedEpoch: number,
    expectedRecoveryToken: string,
    signal?: AbortSignal,
  ) => request<unknown>(
    "/raylight/runtime/recovery/confirm-comfy-restart",
    {
      method: "POST",
      signal,
      body: JSON.stringify({
        confirmation: "comfyui_process_restarted",
        expected_comfy_origin: expectedComfyOrigin,
        expected_epoch: expectedEpoch,
        expected_recovery_token: expectedRecoveryToken,
      }),
    },
  ).then(parseRayLightRuntimeStatus),

  getDraft<M extends GenerationMode>(mode: M, signal?: AbortSignal) {
    return request<ModeDraftMap[M]>(`/drafts/${mode}`, { signal });
  },
  updateDraft<M extends GenerationMode>(mode: M, draft: ModeDraftMap[M]) {
    return request<ModeDraftMap[M]>(`/drafts/${mode}`, { method: "PUT", body: JSON.stringify(draft) });
  },

  getTimeline: (signal?: AbortSignal) =>
    request<unknown>("/timeline", { signal }).then(parseTimelineProjectResponse),
  updateTimeline: (timeline: TimelineProject) =>
    request<unknown>("/timeline", {
      method: "PUT",
      body: JSON.stringify(timeline),
    }).then(parseTimelineProjectResponse),
  createTimelineTask: (payload: TimelineTaskRequest) =>
    request<unknown>("/timeline/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    }).then(parseGenerationTask),
  compileTimeline: (payload: TimelineTaskRequest) =>
    request<unknown>("/timeline/compile", {
      method: "POST",
      body: JSON.stringify(payload),
    }).then(parseTimelineCompileReport),

  // --- Project management (multi-project) ---

  listProjects: (signal?: AbortSignal) =>
    request<unknown>("/projects", { signal }).then(parseProjectList),
  createProject: (title?: string) =>
    request<unknown>("/projects", {
      method: "POST",
      body: JSON.stringify({ title: title ?? "" }),
    }).then(parseProjectSummary),
  importProject: (payload: { title: string; document: TimelineProject }) =>
    request<unknown>("/projects/import", {
      method: "POST",
      body: JSON.stringify(payload),
    }).then(parseProjectSummary),
  renameProject: (projectId: string, title: string) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }).then(parseProjectSummary),
  deleteProject: (projectId: string) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}`, {
      method: "DELETE",
    }).then((value) => parseProjectDelete(value, projectId)),
  getProjectTimeline: (projectId: string, signal?: AbortSignal) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}/timeline`, { signal })
      .then(parseTimelineProjectResponse),
  updateProjectTimeline: (projectId: string, timeline: TimelineProject) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}/timeline`, {
      method: "PUT",
      body: JSON.stringify(timeline),
    }).then(parseTimelineProjectResponse),
  compileProjectTimeline: (projectId: string, payload: TimelineTaskRequest) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}/compile`, {
      method: "POST",
      body: JSON.stringify(payload),
    }).then(parseTimelineCompileReport),
  createProjectTask: (projectId: string, payload: TimelineTaskRequest) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}/jobs`, {
      method: "POST",
      body: JSON.stringify(payload),
    }).then(parseGenerationTask),

  uploadAsset(
    file: File,
    kind: AssetKind,
    onProgress?: (progress: AssetUploadProgress) => void,
  ): Promise<AssetReference> {
    if (onProgress) return uploadAssetWithProgress(file, kind, onProgress);
    const body = new FormData();
    body.append("file", file);
    body.append("kind", kind);
    return request<{ asset: unknown }>("/assets", { method: "POST", body }).then(
      ({ asset }) => {
        const normalized = normalizeAssetReference(asset, kind);
        if (!normalized) {
          throw new ApiError("素材上传响应缺少有效的稳定 ID", 502, asset);
        }
        return normalized;
      },
    );
  },

  listAssets(kind?: AssetKind, signal?: AbortSignal): Promise<AssetListResponse> {
    const query = kind ? `?kind=${encodeURIComponent(kind)}` : "";
    return request<AssetListResponse>(`/assets${query}`, { signal });
  },
  deleteAsset(assetId: string): Promise<AssetDeleteResponse> {
    return request<AssetDeleteResponse>(`/assets/${encodeURIComponent(assetId)}`, {
      method: "DELETE",
    });
  },
  deleteAssetCascade(assetId: string): Promise<AssetCascadeDeleteResponse> {
    return request<unknown>(`/assets/${encodeURIComponent(assetId)}?cascade=true`, {
      method: "DELETE",
    }).then((value) => parseAssetCascadeDeleteResponse(value, assetId));
  },

  detectRV2VShots: (payload: RV2VShotDetectionRequest) =>
    request<RV2VShotDetectionResponse>("/rv2v/detect-shots", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  createTask: (payload: CreateTaskRequest) =>
    request<unknown>("/jobs", { method: "POST", body: JSON.stringify(payload) }).then(parseGenerationTask),
  async listTasks(signal?: AbortSignal, projectId?: string): Promise<TaskListResponse> {
    const jobs: GenerationTask[] = [];
    const seen = new Set<string>();
    let offset = 0;
    let summary: TaskListResponse["summary"];
    let total = 0;
    const projectQuery = projectId ? `&project_id=${encodeURIComponent(projectId)}` : "";
    for (;;) {
      const page = await request<unknown>(
        `/jobs?limit=256&offset=${offset}&sort_by=created_at&sort_order=asc${projectQuery}`,
        { signal },
      ).then(parseTaskList);
      if (page.offset !== offset || page.limit !== 256) {
        throw new ApiError("任务列表分页响应与请求不一致", 502, page);
      }
      summary ??= page.summary;
      total = Math.max(total, page.total ?? 0);
      for (const job of page.jobs) {
        if (!seen.has(job.id)) {
          seen.add(job.id);
          jobs.push(job);
        }
      }
      if (!page.has_more) break;
      if (page.jobs.length === 0) {
        throw new ApiError("任务列表分页没有取得进展", 502, page);
      }
      offset += page.jobs.length;
    }
    return {
      jobs,
      total,
      limit: jobs.length || 256,
      offset: 0,
      has_more: false,
      summary,
    };
  },
  getTask: (taskId: string, signal?: AbortSignal) =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}`, { signal }).then(parseGenerationTask),
  cancelTask: (taskId: string) =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}/cancel`, { method: "POST" }).then(parseGenerationTask),
  confirmComfyRestartRecovery: (taskId: string) =>
    request<unknown>(
      `/jobs/${encodeURIComponent(taskId)}/recovery/confirm-comfy-restart`,
      {
        method: "POST",
        body: JSON.stringify({ confirmation: "comfyui_process_restarted" }),
      },
    ).then(parseGenerationTask),
  cancelTasks: (taskIds: string[]) =>
    request<unknown>("/jobs/cancel", {
      method: "POST",
      body: JSON.stringify({ job_ids: taskIds }),
    }).then(parseTaskBulkCancel),
  getTaskProject: (taskId: string) =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}/project`).then(parseTaskProjectSnapshot),
  getTaskDiagnostic: (taskId: string) =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}/diagnostic`).then(parseTaskDiagnostic),
  getTaskGenerationDetails: (taskId: string) =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}/generation-details`)
      .then(parseTaskGenerationDetails),
  importTaskOutput: (
    taskId: string,
    output: { index: number; segmentId?: string },
  ): Promise<AssetReference> =>
    request<{ asset: unknown }>(`/jobs/${encodeURIComponent(taskId)}/import-output`, {
      method: "POST",
      body: JSON.stringify(output.segmentId
        ? { segment_id: output.segmentId }
        : { output_index: output.index }),
    }).then(({ asset }) => {
      const normalized = normalizeAssetReference(asset, "video");
      if (!normalized) throw new ApiError("任务输出导入响应结构无效", 502, asset);
      return normalized;
    }),
  deleteTask: (taskId: string) =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}`, { method: "DELETE" })
      .then((value) => parseJobDeleteResponse(value, taskId)),
  clearTerminalTasks: () =>
    request<unknown>("/jobs", { method: "DELETE" }).then(parseJobClearResponse),
};
