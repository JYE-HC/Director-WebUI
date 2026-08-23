import {
  isSamplingScheduler,
  type AssetKind,
  type AssetReference,
  type GenerationMode,
  type ModeDraftMap,
} from "../domain/modes";
import {
  normalizeFeatureConfiguration,
  normalizeFeatureSelection,
  normalizeTimelineProject,
  type TimelineGenerationMode,
  type TimelineProject,
} from "../domain/timelineProject";
import { normalizeAssetReference } from "../domain/assets";
import { alignH3FrameCount } from "../domain/timing";
import { isStoragePath } from "../domain/storagePath";
import { parseProjectMigrationReceipt, type ProjectMigrationReceipt } from "../state/timelineV5Migration";
import { parseLegacyRuntimeSettingsV1, parseRuntimeSettingsV3 } from "./types";
import type {
  CapabilityReport,
  AssetCascadeDeleteResponse,
  AssetDeleteResponse,
  AssetListResponse,
  AssetTrashBatch,
  AssetTrashConflictOwner,
  AssetTrashListResponse,
  AssetTrashPurgeResponse,
  AssetTrashRestoreMode,
  AssetTrashRestoreResponse,
  AssetUploadProgress,
  ConnectionTestResult,
  CreateTaskRequest,
  DirectorDeckConfig,
  CapabilityJsonValue,
  CapabilityReason,
  EffectiveFeatureResolution,
  EffectiveSegmentResolution,
  FeatureCatalog,
  FeatureCatalogAdapterOption,
  FeatureCatalogEntry,
  FeatureCatalogFetchResult,
  FeatureCapabilityEvaluation,
  FeaturePreflightRequest,
  FeaturePreflightResponse,
  FeatureResolutionEvidence,
  GenerationTask,
  GPUResource,
  JobClearResponse,
  JobDeleteResponse,
  MediaToolInstallSnapshot,
  MediaToolsStatus,
  ModelInventory,
  OperationalReadiness,
  ProjectDeleteResponse,
  ProjectListResponse,
  ProjectImportCommitRequest,
  ProjectImportCreativeSelection,
  ProjectImportLegacyCreativeContext,
  ProjectImportPreflightRequest,
  ProjectImportPreflightResponse,
  ProjectSummary,
  RayLightInstallSnapshot,
  RayLightRuntimeStatus,
  RayLightSetupStatus,
  RV2VShotDetectionRequest,
  RV2VShotDetectionResponse,
  RuntimeSettings,
  RuntimeSettingsAuthority,
  RuntimeSettingsMigrationNoticeList,
  ResolvedImplementationIdentity,
  StorageConfiguration,
  TaskListResponse,
  TaskStatus,
  TaskBulkCancelResponse,
  TaskDiagnostic,
  TaskGenerationDetails,
  TaskProjectSnapshotResponse,
  TimelineAuthority,
  TimelineCompileReport,
  TimelineTaskRequest,
} from "./types";

// The SPA is served by the ComfyUI plugin under /directordeck/, and the API is
// proxied at /directordeck/api on the same origin.
const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "/directordeck/api"
).replace(/\/$/, "");

export function taskEventsUrl(): string {
  return `${API_BASE}/tasks/events`;
}

export type ApiErrorCode =
  | "raylight_recovery_in_flight"
  | "node_unavailable"
  | "timeline_revision_conflict"
  | "timeline_schema_migrated"
  | "runtime_settings_schema_migrated"
  | "runtime_settings_authority_conflict"
  | "project_import_preflight_required"
  | "assets_in_use"
  | "asset_trash_restore_conflict"
  | "asset_trash_purge_conflict";

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

async function apiErrorFromResponse(response: Response): Promise<ApiError> {
  let responseText = "";
  try {
    responseText = await response.text();
  } catch {
    // The HTTP status is already authoritative even if a broken proxy/body
    // stream prevents reading its diagnostic payload.
    return new ApiError(`HTTP ${response.status}`, response.status);
  }
  let details: unknown = responseText;
  if (responseText) {
    try {
      details = JSON.parse(responseText) as unknown;
    } catch {
      // Plain-text and HTML errors still retain their HTTP status below.
    }
  }
  const parsed = parseHttpError(details, response.status);
  return new ApiError(parsed.message, response.status, parsed.details, parsed.code);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  headers.set("Accept", "application/json");
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok) throw await apiErrorFromResponse(response);
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
        reject(new ApiError(parsed.message, xhr.status, parsed.details, parsed.code));
        return;
      }
      const payload = xhr.response;
      const assetValue = isRecord(payload) ? payload.asset : undefined;
      const normalized = normalizeAssetReference(assetValue, kind, { completeWireShape: true });
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
    typeof value.authority_token !== "string" ||
    !/^[0-9a-f]{64}$/.test(value.authority_token)
  ) throw new ApiError("运行设置权威响应结构无效", 502, value);
  const settings = parseRuntimeSettingsV3(value.settings);
  if (!settings) throw new ApiError("运行设置权威响应包含过期或无效 schema", 502, value);
  return {
    settings,
    authority_token: value.authority_token,
  };
}

function parseRuntimeSettingsMigrationNotices(
  value: unknown,
): RuntimeSettingsMigrationNoticeList {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["notices"]) ||
    !Array.isArray(value.notices) ||
    value.notices.length > 256
  ) throw new ApiError("运行设置迁移提示响应结构无效", 502, value);
  const ids = new Set<string>();
  const notices = value.notices.map((notice) => {
    if (
      !isRecord(notice) ||
      !hasExactKeys(notice, [
        "schema_version", "id", "code", "severity", "action",
        "legacy_strategy_version", "message", "created_at",
      ]) ||
      notice.schema_version !== 1 ||
      typeof notice.id !== "string" ||
      !/^[a-z0-9._-]{1,128}$/.test(notice.id) ||
      ids.has(notice.id) ||
      notice.code !== "legacy_lora_resolution_review_required" ||
      notice.severity !== "warning" ||
      notice.action !== "review_lora_loader_mappings" ||
      notice.legacy_strategy_version !==
        "v4-known-filename-or-safetensors-metadata-v1" ||
      typeof notice.message !== "string" ||
      [...notice.message].length < 1 || [...notice.message].length > 1_024 ||
      typeof notice.created_at !== "string" ||
      !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(
        notice.created_at,
      ) ||
      !Number.isFinite(Date.parse(notice.created_at))
    ) throw new ApiError("运行设置迁移提示响应结构无效", 502, value);
    ids.add(notice.id);
    return {
      schema_version: 1 as const,
      id: notice.id,
      code: "legacy_lora_resolution_review_required" as const,
      severity: "warning" as const,
      action: "review_lora_loader_mappings" as const,
      legacy_strategy_version:
        "v4-known-filename-or-safetensors-metadata-v1" as const,
      message: notice.message,
      created_at: notice.created_at,
    };
  });
  return { notices };
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

const CAPABILITY_IDENTIFIER = /^[A-Za-z][A-Za-z0-9_.:-]{0,127}$/;
const CAPABILITY_DIGEST = /^sha256:[0-9a-f]{64}$/;
const CATALOG_STRONG_ETAG = /^"sha256:[0-9a-f]{64}"$/;

function isCapabilityIdentifier(value: unknown): value is string {
  return typeof value === "string" && CAPABILITY_IDENTIFIER.test(value);
}

function isBoundedNonemptyString(value: unknown, maxLength: number): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maxLength;
}

function isBoundedOpaqueId(value: unknown, maxLength: number): value is string {
  return isBoundedNonemptyString(value, maxLength) &&
    !/[\u0000-\u001f\u007f]/.test(value);
}

function isCapabilityDigest(value: unknown): value is string {
  return typeof value === "string" && CAPABILITY_DIGEST.test(value);
}

function isCapabilityJsonValue(
  value: unknown,
  depth = 0,
): value is CapabilityJsonValue {
  if (value === null || typeof value === "boolean") return true;
  if (typeof value === "string") {
    return value.length <= 65_536 &&
      !/[\uD800-\uDFFF]/.test(value);
  }
  if (typeof value === "number") {
    return Number.isFinite(value) &&
      !Object.is(value, -0) &&
      (!Number.isInteger(value) || Number.isSafeInteger(value));
  }
  if (depth >= 12) return false;
  if (Array.isArray(value)) {
    return value.length <= 256 && value.every((item) =>
      isCapabilityJsonValue(item, depth + 1));
  }
  if (!isRecord(value) || Object.keys(value).length > 256) return false;
  return Object.entries(value).every(([key, item]) =>
    key.length > 0 &&
    key.length <= 256 &&
    !/[\u0000-\u001f\uD800-\uDFFF]/.test(key) &&
    isCapabilityJsonValue(item, depth + 1));
}

function parseCapabilityJsonObject(
  value: unknown,
  errorMessage: string,
): { [key: string]: CapabilityJsonValue } {
  if (!isRecord(value) || !isCapabilityJsonValue(value)) {
    throw new ApiError(errorMessage, 502, value);
  }
  return structuredClone(value) as { [key: string]: CapabilityJsonValue };
}

function parseUniqueIdentifiers(value: unknown): string[] | null {
  if (!Array.isArray(value) || !value.every(isCapabilityIdentifier)) return null;
  if (new Set(value).size !== value.length) return null;
  return [...value];
}

function parseCapabilityReason(value: unknown): CapabilityReason {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "code",
      "feature_id",
      "segment_id",
      "unit_id",
      "backend",
      "rule",
      "message",
      "remediation",
      "safe_details",
    ]) ||
    !isCapabilityIdentifier(value.code) ||
    !(value.feature_id === null || isCapabilityIdentifier(value.feature_id)) ||
    !(value.segment_id === null || isBoundedOpaqueId(value.segment_id, 128)) ||
    !(value.unit_id === null || isBoundedOpaqueId(value.unit_id, 256)) ||
    !(value.backend === null || value.backend === "standard" || value.backend === "raylight") ||
    !isCapabilityIdentifier(value.rule) ||
    !isBoundedNonemptyString(value.message, 4_096) || !value.message.trim() ||
    !isBoundedNonemptyString(value.remediation, 4_096) || !value.remediation.trim()
  ) throw new ApiError("功能能力原因结构无效", 502, value);
  return {
    code: value.code,
    feature_id: value.feature_id,
    segment_id: value.segment_id,
    unit_id: value.unit_id,
    backend: value.backend,
    rule: value.rule,
    message: value.message,
    remediation: value.remediation,
    safe_details: parseCapabilityJsonObject(
      value.safe_details,
      "功能能力原因安全详情结构无效",
    ),
  };
}

function parseCapabilityReasons(value: unknown): CapabilityReason[] {
  if (!Array.isArray(value)) {
    throw new ApiError("功能能力原因列表结构无效", 502, value);
  }
  return value.map(parseCapabilityReason);
}

function parseFeatureCapabilityEvaluation(
  value: unknown,
): FeatureCapabilityEvaluation {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "available",
      "reasons",
      "verified_contracts",
      "runtime_fingerprints",
    ]) ||
    typeof value.available !== "boolean"
  ) throw new ApiError("功能能力证据结构无效", 502, value);
  const verifiedContracts = parseUniqueIdentifiers(value.verified_contracts);
  const runtimeFingerprints = Array.isArray(value.runtime_fingerprints) &&
    value.runtime_fingerprints.every(isCapabilityDigest) &&
    new Set(value.runtime_fingerprints).size === value.runtime_fingerprints.length
    ? [...value.runtime_fingerprints]
    : null;
  if (verifiedContracts === null || runtimeFingerprints === null) {
    throw new ApiError("功能能力证据结构无效", 502, value);
  }
  const reasons = parseCapabilityReasons(value.reasons);
  if (value.available === (reasons.length > 0)) {
    throw new ApiError("功能能力可用性摘要无效", 502, value);
  }
  return {
    available: value.available,
    reasons,
    verified_contracts: verifiedContracts,
    runtime_fingerprints: runtimeFingerprints,
  };
}

const SEMANTIC_VERSION = /^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$/;

function parseResolvedImplementationIdentity(
  value: unknown,
): ResolvedImplementationIdentity {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "role",
      "class_type",
      "implementation_id",
      "semantic_version",
      "runtime_fingerprint",
      "binding_key",
    ]) ||
    !isCapabilityIdentifier(value.role) ||
    !isBoundedNonemptyString(value.class_type, 128) ||
    !isCapabilityIdentifier(value.implementation_id) ||
    typeof value.semantic_version !== "string" ||
    value.semantic_version.length > 128 ||
    !SEMANTIC_VERSION.test(value.semantic_version) ||
    !isCapabilityDigest(value.runtime_fingerprint) ||
    !isCapabilityIdentifier(value.binding_key)
  ) throw new ApiError("功能实现身份结构无效", 502, value);
  return {
    role: value.role,
    class_type: value.class_type,
    implementation_id: value.implementation_id,
    semantic_version: value.semantic_version,
    runtime_fingerprint: value.runtime_fingerprint,
    binding_key: value.binding_key,
  };
}

function parseFeatureResolutionEvidence(
  value: unknown,
): FeatureResolutionEvidence {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["state", "implementations", "resolution_details"]) ||
    (value.state !== "active" && value.state !== "noop") ||
    !Array.isArray(value.implementations)
  ) throw new ApiError("功能解析详情结构无效", 502, value);
  const implementations = value.implementations.map(
    parseResolvedImplementationIdentity,
  );
  if (
    new Set(implementations.map((item) => item.binding_key)).size !==
    implementations.length
  ) throw new ApiError("功能解析详情包含重复绑定键", 502, value);
  const resolutionDetails = parseCapabilityJsonObject(
    value.resolution_details,
    "功能解析详情的安全数据结构无效",
  );
  if (
    (value.state === "active" && implementations.length === 0) ||
    (value.state === "noop" && (
      implementations.length > 0 ||
      typeof resolutionDetails.reason !== "string" ||
      resolutionDetails.reason.trim().length === 0
    ))
  ) throw new ApiError("功能解析状态与实现证据不一致", 502, value);
  return {
    state: value.state,
    implementations,
    resolution_details: resolutionDetails,
  };
}

function parseOperationalReadiness(value: unknown): OperationalReadiness {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "endpoint_online",
      "submission_allowed",
      "ray_recovery_required",
      "ray_tainted",
      "invalid_runtime_gpu_indices",
      "blocking_reason_codes",
    ]) ||
    typeof value.endpoint_online !== "boolean" ||
    typeof value.submission_allowed !== "boolean" ||
    typeof value.ray_recovery_required !== "boolean" ||
    typeof value.ray_tainted !== "boolean" ||
    !Array.isArray(value.invalid_runtime_gpu_indices) ||
    !value.invalid_runtime_gpu_indices.every((index) =>
      Number.isSafeInteger(index) && Number(index) >= 0 && Number(index) <= 255) ||
    new Set(value.invalid_runtime_gpu_indices).size !== value.invalid_runtime_gpu_indices.length
  ) throw new ApiError("运行就绪状态结构无效", 502, value);
  const blockingReasons = parseUniqueIdentifiers(value.blocking_reason_codes);
  const intrinsicallyBlocked =
    !value.endpoint_online ||
    value.ray_recovery_required ||
    value.invalid_runtime_gpu_indices.length > 0;
  if (
    blockingReasons === null ||
    (value.submission_allowed && (intrinsicallyBlocked || blockingReasons.length > 0)) ||
    (!value.submission_allowed && blockingReasons.length === 0)
  ) throw new ApiError("运行就绪状态语义无效", 502, value);
  return {
    endpoint_online: value.endpoint_online,
    submission_allowed: value.submission_allowed,
    ray_recovery_required: value.ray_recovery_required,
    ray_tainted: value.ray_tainted,
    invalid_runtime_gpu_indices: [...value.invalid_runtime_gpu_indices] as number[],
    blocking_reason_codes: blockingReasons,
  };
}

function parseStringArrayRecord(value: unknown): Record<string, string[]> | null {
  if (!isRecord(value)) return null;
  const parsed: Array<[string, string[]]> = [];
  for (const [key, entries] of Object.entries(value)) {
    if (!key || !isStringArray(entries)) return null;
    parsed.push([key, [...entries]]);
  }
  return Object.fromEntries(parsed);
}

function parseAssetTrashConflictOwners(
  value: unknown,
): AssetTrashConflictOwner[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const ownerKinds = new Set(["timeline", "project", "draft", "asset", "batch"]);
  const allowedReasons = new Set([
    "owner_missing",
    "document_changed",
    "revision_changed",
    "revision_exhausted",
    "registration_changed",
    "inverse_document_unavailable",
  ]);
  const allowedKeys = new Set([
    "owner_kind",
    "owner_id",
    "reason",
    "expected_revision",
    "actual_revision",
    "message",
  ]);
  const parsed: AssetTrashConflictOwner[] = [];
  for (const item of value) {
    const reasons: string[] = isRecord(item) && typeof item.reason === "string"
      ? item.reason.split(",")
      : [];
    if (
      !isRecord(item) ||
      Object.keys(item).some((key) => !allowedKeys.has(key)) ||
      !ownerKinds.has(String(item.owner_kind)) ||
      typeof item.owner_id !== "string" || !item.owner_id ||
      typeof item.reason !== "string" || !item.reason ||
      reasons.length === 0 ||
      new Set(reasons).size !== reasons.length ||
      reasons.some((reason) => !allowedReasons.has(reason)) ||
      ("expected_revision" in item &&
        item.expected_revision !== null &&
        !isNonNegativeInteger(item.expected_revision)) ||
      ("actual_revision" in item &&
        item.actual_revision !== null &&
        !isNonNegativeInteger(item.actual_revision)) ||
      ("message" in item &&
        (typeof item.message !== "string" || !item.message))
    ) return null;
    parsed.push({
      owner_kind: item.owner_kind as AssetTrashConflictOwner["owner_kind"],
      owner_id: item.owner_id,
      reason: item.reason,
      ...(item.expected_revision === null || isNonNegativeInteger(item.expected_revision)
        ? { expected_revision: item.expected_revision }
        : {}),
      ...(item.actual_revision === null || isNonNegativeInteger(item.actual_revision)
        ? { actual_revision: item.actual_revision }
        : {}),
      ...(typeof item.message === "string" ? { message: item.message } : {}),
    });
  }
  return parsed;
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
  const candidateCode: ApiErrorCode | undefined =
    detail.code === "raylight_recovery_in_flight" ||
    detail.code === "node_unavailable" ||
    detail.code === "timeline_revision_conflict" ||
    detail.code === "timeline_schema_migrated" ||
    detail.code === "runtime_settings_schema_migrated" ||
    detail.code === "runtime_settings_authority_conflict" ||
    detail.code === "project_import_preflight_required" ||
    detail.code === "assets_in_use" ||
    detail.code === "asset_trash_restore_conflict" ||
    detail.code === "asset_trash_purge_conflict"
      ? detail.code
      : undefined;
  const assetCode = candidateCode === "assets_in_use" ||
    candidateCode === "asset_trash_restore_conflict" ||
    candidateCode === "asset_trash_purge_conflict";
  const conflicts = candidateCode === "asset_trash_restore_conflict" ||
      candidateCode === "asset_trash_purge_conflict"
    ? parseAssetTrashConflictOwners(detail.conflicts)
    : null;
  const code: ApiErrorCode | undefined =
    assetCode && detail.remote_files_preserved !== true
      ? undefined
      : (candidateCode === "asset_trash_restore_conflict" ||
          candidateCode === "asset_trash_purge_conflict") && conflicts === null
        ? undefined
        : candidateCode;
  if (typeof detail.message !== "string" || !detail.message.trim()) {
    return code
      ? { message: fallback, code, details: { detail: { code } } }
      : { message: fallback };
  }
  const missingNodeClassTypes: string[] = [];
  if (code === "node_unavailable" && Array.isArray(detail.reasons)) {
    const seen = new Set<string>();
    for (const reason of detail.reasons.slice(0, 256)) {
      if (
        !isRecord(reason) ||
        reason.code !== "node_unavailable" ||
        !isRecord(reason.safe_details)
      ) continue;
      const classType = reason.safe_details.class_type;
      if (
        typeof classType !== "string" ||
        classType.trim() !== classType ||
        !isBoundedOpaqueId(classType, 256) ||
        seen.has(classType)
      ) continue;
      seen.add(classType);
      missingNodeClassTypes.push(classType);
    }
  }
  const usages = code === "node_unavailable"
    ? []
    : isStringArray(detail.usages) ? [...detail.usages] : [];
  const usagesByAsset = code === "assets_in_use"
    ? parseStringArrayRecord(detail.usages_by_asset)
    : null;
  const timelineConflict = code === "timeline_revision_conflict";
  const safeDetail = {
    ...(code ? { code } : {}),
    message: detail.message,
    ...(missingNodeClassTypes.length ? { missing_node_class_types: missingNodeClassTypes } : {}),
    ...(usages.length ? { usages } : {}),
    ...(timelineConflict && typeof detail.project_id === "string" && detail.project_id
      ? { project_id: detail.project_id }
      : {}),
    ...(timelineConflict && isNonNegativeInteger(detail.expected_revision)
      ? { expected_revision: detail.expected_revision }
      : {}),
    ...(timelineConflict && isNonNegativeInteger(detail.actual_revision)
      ? { actual_revision: detail.actual_revision }
      : {}),
    ...(assetCode && detail.remote_files_preserved === true
      ? { remote_files_preserved: true as const }
      : {}),
    ...(usagesByAsset ? { usages_by_asset: usagesByAsset } : {}),
    ...(conflicts ? { conflicts } : {}),
  };
  const message = missingNodeClassTypes.length
    ? `缺少 ComfyUI 节点：${missingNodeClassTypes.join("、")}。请更新 ComfyUI，或安装/启用对应节点后重启。`
    : usages.length
      ? `${detail.message}（引用位置：${usages.join("、")}）`
      : detail.message;
  return {
    message,
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

function parseFeatureCatalogEntry(value: unknown): FeatureCatalogEntry {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "id",
      "version",
      "title",
      "description",
      "mode",
      "layer",
      "scopes",
      "params_schema",
      "defaults",
      "backends",
      "availability",
      "adapter_options",
      "ui",
    ]) ||
    !isCapabilityIdentifier(value.id) ||
    !Number.isSafeInteger(value.version) || Number(value.version) < 1 ||
    !isBoundedNonemptyString(value.title, 256) || !value.title.trim() ||
    !isBoundedNonemptyString(value.description, 4_096) || !value.description.trim() ||
    (value.mode !== "switch" && value.mode !== "needed") ||
    value.layer !== "graph" ||
    !isRecord(value.availability) ||
    !hasExactKeys(value.availability, ["state", "reasons"]) ||
    !["available", "unavailable", "conditional"].includes(
      String(value.availability.state),
    )
  ) throw new ApiError("功能目录条目结构无效", 502, value);
  const scopes = parseUniqueIdentifiers(value.scopes);
  const backends = Array.isArray(value.backends) &&
    value.backends.every((backend) => backend === "standard" || backend === "raylight") &&
    new Set(value.backends).size === value.backends.length
    ? [...value.backends]
    : null;
  if (!scopes?.length || !backends?.length) {
    throw new ApiError("功能目录条目作用域或后端结构无效", 502, value);
  }
  const availabilityReasons = parseCapabilityReasons(value.availability.reasons);
  if (
    (value.availability.state === "available" && availabilityReasons.length > 0) ||
    (value.availability.state === "unavailable" && availabilityReasons.length === 0)
  ) throw new ApiError("功能目录可用性摘要无效", 502, value.availability);
  // Loader choices come from /config.  Catalog adapter options are only
  // advisory host observations and must never invalidate the execution
  // catalog or lock generation when an older/newer backend returns a
  // different observation shape.
  const seenAdapterIds = new Set<string>();
  const adapterOptions: FeatureCatalogAdapterOption[] = (
    Array.isArray(value.adapter_options) ? value.adapter_options : []
  ).flatMap((option): FeatureCatalogAdapterOption[] => {
      if (
        !isRecord(option) ||
        !hasExactKeys(option, [
          "adapter_id",
          "display_name",
          "class_type",
          "is_default",
          "backend",
          "supported_families",
          "configuration_options",
          "adapter_fingerprint",
          "capability",
        ]) ||
        typeof option.adapter_id !== "string" ||
        !/^[a-z][a-z0-9_]{0,63}$/.test(option.adapter_id) ||
        !isBoundedNonemptyString(option.display_name, 128) ||
        !isBoundedNonemptyString(option.class_type, 256) ||
        typeof option.is_default !== "boolean" ||
        option.backend !== "standard" ||
        !Array.isArray(option.supported_families) ||
        option.supported_families.length === 0 ||
        option.supported_families.some((family) =>
          family !== "fl2va" && family !== "ref2va") ||
        new Set(option.supported_families).size !== option.supported_families.length ||
        !Array.isArray(option.configuration_options) ||
        option.configuration_options.some((definition) =>
          !isRecord(definition) ||
          !hasExactKeys(definition, [
            "id", "type", "label", "description", "default",
          ]) ||
          typeof definition.id !== "string" ||
          !/^[a-z][a-z0-9_]{0,63}$/.test(definition.id) ||
          definition.type !== "boolean" ||
          !isBoundedNonemptyString(definition.label, 128) ||
          !isBoundedNonemptyString(definition.description, 512) ||
          typeof definition.default !== "boolean") ||
        !isCapabilityDigest(option.adapter_fingerprint) ||
        seenAdapterIds.has(option.adapter_id)
      ) return [];
      try {
        const parsed: FeatureCatalogAdapterOption = {
          adapter_id: option.adapter_id as FeatureCatalogAdapterOption["adapter_id"],
          display_name: option.display_name,
          class_type: option.class_type,
          is_default: option.is_default,
          backend: "standard",
          supported_families: [...option.supported_families],
          configuration_options: option.configuration_options.map((definition) => ({
            id: definition.id as string,
            type: "boolean" as const,
            label: definition.label as string,
            description: definition.description as string,
            default: definition.default as boolean,
          })),
          adapter_fingerprint: option.adapter_fingerprint,
          capability: parseFeatureCapabilityEvaluation(option.capability),
        };
        seenAdapterIds.add(option.adapter_id);
        return [parsed];
      } catch {
        return [];
      }
    },
  );
  return {
    id: value.id,
    version: Number(value.version),
    title: value.title,
    description: value.description,
    mode: value.mode,
    layer: "graph",
    scopes,
    params_schema: parseCapabilityJsonObject(
      value.params_schema,
      "功能目录参数结构无效",
    ),
    defaults: parseCapabilityJsonObject(value.defaults, "功能目录默认值结构无效"),
    backends,
    availability: {
      state: value.availability.state as FeatureCatalogEntry["availability"]["state"],
      reasons: availabilityReasons,
    },
    adapter_options: adapterOptions,
    ui: parseCapabilityJsonObject(value.ui, "功能目录 UI 元数据结构无效"),
  };
}

function parseFeatureCatalog(value: unknown): FeatureCatalog {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "template_bundle_version",
      "host_capability_revision",
      "entries",
    ]) ||
    !Number.isSafeInteger(value.template_bundle_version) ||
    Number(value.template_bundle_version) < 1 ||
    !isCapabilityDigest(value.host_capability_revision) ||
    !Array.isArray(value.entries)
  ) throw new ApiError("功能目录响应结构无效", 502, value);
  const entries = value.entries.map(parseFeatureCatalogEntry);
  if (
    new Set(entries.map((entry) => `${entry.id}\u0000${entry.version}`)).size !==
    entries.length
  ) {
    throw new ApiError("功能目录包含重复功能版本", 502, value.entries);
  }
  return {
    template_bundle_version: Number(value.template_bundle_version),
    host_capability_revision: value.host_capability_revision,
    entries,
  };
}

function parseDirectorDeckConfig(value: unknown): DirectorDeckConfig {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["schema_version", "lora"]) ||
    value.schema_version !== 1 ||
    !isRecord(value.lora) ||
    !hasExactKeys(value.lora, [
      "loaders", "fallback_policy", "loader_policies",
    ]) ||
    !Array.isArray(value.lora.loaders) ||
    value.lora.loaders.length === 0 ||
    !isRecord(value.lora.fallback_policy) ||
    !Array.isArray(value.lora.loader_policies)
  ) throw new ApiError("DirectorDeck 配置响应结构无效", 502, value);
  const loaders = value.lora.loaders.map((loader) => {
    if (
      !isRecord(loader) ||
      !hasExactKeys(loader, [
        "id", "display_name", "class_type", "input_contract",
        "supported_families", "options",
      ]) ||
      typeof loader.id !== "string" ||
      !/^[a-z][a-z0-9_]{0,63}$/.test(loader.id) ||
      !isBoundedNonemptyString(loader.display_name, 128) ||
      !isBoundedNonemptyString(loader.class_type, 256) ||
      (loader.input_contract !== "model_only" &&
        loader.input_contract !== "dedicated_model") ||
      !Array.isArray(loader.supported_families) ||
      loader.supported_families.length === 0 ||
      loader.supported_families.some((family) =>
        family !== "fl2va" && family !== "ref2va") ||
      new Set(loader.supported_families).size !== loader.supported_families.length ||
      !Array.isArray(loader.options)
    ) throw new ApiError("DirectorDeck LoRA 加载器配置无效", 502, loader);
    const options = loader.options.map((option) => {
      if (
        !isRecord(option) ||
        !hasExactKeys(option, ["id", "type", "label", "description", "default"]) ||
        typeof option.id !== "string" ||
        !/^[a-z][a-z0-9_]{0,63}$/.test(option.id) ||
        option.type !== "boolean" ||
        !isBoundedNonemptyString(option.label, 128) ||
        !isBoundedNonemptyString(option.description, 512) ||
        typeof option.default !== "boolean"
      ) throw new ApiError("DirectorDeck LoRA 加载器配置项无效", 502, option);
      return {
        id: option.id as string,
        type: "boolean" as const,
        label: option.label as string,
        description: option.description as string,
        default: option.default,
      };
    });
    if (new Set(options.map((option) => option.id)).size !== options.length) {
      throw new ApiError("DirectorDeck LoRA 加载器配置项重复", 502, loader);
    }
    return {
      id: loader.id as DirectorDeckConfig["lora"]["loaders"][number]["id"],
      display_name: loader.display_name as string,
      class_type: loader.class_type as string,
      input_contract: loader.input_contract as DirectorDeckConfig["lora"]["loaders"][number]["input_contract"],
      supported_families: [...loader.supported_families] as DirectorDeckConfig["lora"]["loaders"][number]["supported_families"],
      options,
    };
  });
  const loaderIds = new Set(loaders.map((loader) => loader.id));
  if (
    loaderIds.size !== loaders.length
  ) throw new ApiError("DirectorDeck LoRA 加载器清单无效", 502, value);
  const fallbackPolicy = value.lora.fallback_policy;
  if (
    !hasExactKeys(fallbackPolicy, ["loader_ids", "default_loader_id"]) ||
    !Array.isArray(fallbackPolicy.loader_ids) ||
    fallbackPolicy.loader_ids.length === 0 ||
    fallbackPolicy.loader_ids.some((id) =>
      typeof id !== "string" || !/^[a-z][a-z0-9_]{0,63}$/.test(id) ||
      !loaderIds.has(id)) ||
    new Set(fallbackPolicy.loader_ids).size !== fallbackPolicy.loader_ids.length ||
    typeof fallbackPolicy.default_loader_id !== "string" ||
    !fallbackPolicy.loader_ids.includes(fallbackPolicy.default_loader_id)
  ) throw new ApiError("DirectorDeck LoRA 回退策略无效", 502, fallbackPolicy);
  const policyPatterns = new Set<string>();
  const loaderPolicies = value.lora.loader_policies.map((policy) => {
    if (
      !isRecord(policy) ||
      !hasExactKeys(policy, [
        "lora_filename", "loader_ids", "default_loader_id",
      ]) ||
      !isBoundedNonemptyString(policy.lora_filename, 1_024) ||
      policyPatterns.has(policy.lora_filename) ||
      !Array.isArray(policy.loader_ids) ||
      policy.loader_ids.length === 0 ||
      policy.loader_ids.some((id) =>
        typeof id !== "string" || !/^[a-z][a-z0-9_]{0,63}$/.test(id) ||
        !loaderIds.has(id)) ||
      new Set(policy.loader_ids).size !== policy.loader_ids.length ||
      typeof policy.default_loader_id !== "string" ||
      !policy.loader_ids.includes(policy.default_loader_id)
    ) throw new ApiError("DirectorDeck LoRA 加载器策略无效", 502, policy);
    try {
      new RegExp(policy.lora_filename);
    } catch {
      throw new ApiError("DirectorDeck LoRA 文件名正则表达式无效", 502, policy);
    }
    policyPatterns.add(policy.lora_filename);
    return {
      lora_filename: policy.lora_filename,
      loader_ids: [...policy.loader_ids] as DirectorDeckConfig["lora"]["loader_policies"][number]["loader_ids"],
      default_loader_id: policy.default_loader_id as DirectorDeckConfig["lora"]["loader_policies"][number]["default_loader_id"],
    };
  });
  return {
    schema_version: 1,
    lora: {
      loaders,
      fallback_policy: {
        loader_ids: [...fallbackPolicy.loader_ids] as DirectorDeckConfig["lora"]["fallback_policy"]["loader_ids"],
        default_loader_id: fallbackPolicy.default_loader_id as DirectorDeckConfig["lora"]["fallback_policy"]["default_loader_id"],
      },
      loader_policies: loaderPolicies,
    },
  };
}

function parseEffectiveFeatureResolution(
  value: unknown,
): EffectiveFeatureResolution {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "id",
      "version",
      "state",
      "adapter_fingerprint",
      "capability",
    ]) ||
    !isBoundedNonemptyString(value.id, 128) ||
    !Number.isSafeInteger(value.version) || Number(value.version) < 1 ||
    (value.state !== "active" && value.state !== "noop") ||
    !isCapabilityDigest(value.adapter_fingerprint)
  ) throw new ApiError("功能预检解析项结构无效", 502, value);
  return {
    id: value.id,
    version: Number(value.version),
    state: value.state,
    adapter_fingerprint: value.adapter_fingerprint,
    capability: parseFeatureCapabilityEvaluation(value.capability),
  };
}

function parseEffectiveSegmentResolution(
  value: unknown,
): EffectiveSegmentResolution {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "unit_id",
      "backend",
      "family",
      "template_id",
      "features",
    ]) ||
    !isBoundedOpaqueId(value.unit_id, 256) ||
    (value.backend !== "standard" && value.backend !== "raylight") ||
    (value.family !== "fl2va" && value.family !== "ref2va") ||
    (value.template_id !== "h3_standard_segment" &&
      value.template_id !== "h3_raylight_segment") ||
    (value.backend === "standard" && value.template_id !== "h3_standard_segment") ||
    (value.backend === "raylight" && value.template_id !== "h3_raylight_segment") ||
    !Array.isArray(value.features) ||
    value.features.length === 0 ||
    value.features.length > 64
  ) throw new ApiError("功能预检分段结构无效", 502, value);
  const features = value.features.map(parseEffectiveFeatureResolution);
  if (new Set(features.map((feature) => feature.id)).size !== features.length) {
    throw new ApiError("功能预检分段包含重复功能 ID", 502, value.features);
  }
  return {
    unit_id: value.unit_id,
    backend: value.backend,
    family: value.family,
    template_id: value.template_id,
    features,
  };
}

function parseFeaturePreflight(value: unknown): FeaturePreflightResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "template_bundle_version",
      "host_capability_revision",
      "operational_readiness",
      "valid",
      "errors",
      "effective_by_segment",
    ]) ||
    !Number.isSafeInteger(value.template_bundle_version) ||
    Number(value.template_bundle_version) < 1 ||
    !isCapabilityDigest(value.host_capability_revision) ||
    typeof value.valid !== "boolean" ||
    !isRecord(value.effective_by_segment)
  ) throw new ApiError("功能预检响应结构无效", 502, value);
  const operationalReadiness = parseOperationalReadiness(value.operational_readiness);
  const errors = parseCapabilityReasons(value.errors);
  // Feature validity describes the selected graph and its required
  // capabilities. Transient Ray ledger readiness is a separate top-level
  // diagnostic and is re-evaluated by the locked submission planner; coupling
  // the two here would turn an unrelated Standard preflight into a UI gate.
  if (value.valid !== (errors.length === 0)) {
    throw new ApiError("功能预检有效性摘要无效", 502, value);
  }
  const effectiveBySegment = Object.fromEntries(
    Object.entries(value.effective_by_segment).map(([segmentId, effective]) => {
      if (!isBoundedOpaqueId(segmentId, 128)) {
        throw new ApiError("功能预检包含无效分段 ID", 502, segmentId);
      }
      return [segmentId, parseEffectiveSegmentResolution(effective)];
    }),
  );
  return {
    template_bundle_version: Number(value.template_bundle_version),
    host_capability_revision: value.host_capability_revision,
    operational_readiness: operationalReadiness,
    valid: value.valid,
    errors,
    effective_by_segment: effectiveBySegment,
  };
}

async function requestFeatureCatalog(
  etag?: string,
  signal?: AbortSignal,
): Promise<FeatureCatalogFetchResult> {
  if (etag !== undefined && !CATALOG_STRONG_ETAG.test(etag)) {
    throw new TypeError("功能目录 ETag 无效");
  }
  const headers = new Headers({ Accept: "application/json" });
  if (etag) headers.set("If-None-Match", etag);
  const response = await fetch(`${API_BASE}/features/catalog`, { signal, headers });
  if (response.status === 304) {
    const responseBody = await response.text();
    const returnedEtag = response.headers.get("ETag");
    if (
      !etag || responseBody.length > 0 ||
      returnedEtag !== etag
    ) throw new ApiError("功能目录 304 响应结构无效", 502);
    return { status: "not_modified", etag };
  }
  if (!response.ok) throw await apiErrorFromResponse(response);
  if (response.status !== 200) {
    throw new ApiError("功能目录 HTTP 状态无效", 502, response.status);
  }
  const responseEtag = response.headers.get("ETag");
  if (responseEtag === null || !CATALOG_STRONG_ETAG.test(responseEtag)) {
    throw new ApiError("功能目录缺少有效的强 ETag", 502, responseEtag);
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw new ApiError("功能目录响应不是有效 JSON", 502);
  }
  return {
    status: "fresh",
    etag: responseEtag,
    catalog: parseFeatureCatalog(payload),
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
    !hasExactKeys(value, [
      "template_bundle_version",
      "host_capability_revision",
      "execution_strategy",
      "model_families",
      "plans",
      "node_policy",
      "features",
      "effective_execution_digest",
    ]) ||
    !Number.isSafeInteger(value.template_bundle_version) ||
    Number(value.template_bundle_version) < 1 ||
    Number(value.template_bundle_version) > 2_147_483_647 ||
    !isCapabilityDigest(value.host_capability_revision) ||
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

  const features = value.features;
  if (
    !isRecord(features) ||
    !hasExactKeys(features, [
      "requested",
      "effective_by_segment",
      "resolutions",
      "notices",
    ]) ||
    !isRecord(features.requested) ||
    !isRecord(features.requested.by_segment) ||
    !isRecord(features.effective_by_segment) ||
    !Array.isArray(features.resolutions) ||
    features.resolutions.length > 8_192 ||
    !Array.isArray(features.notices) ||
    features.notices.length > 256 ||
    !isRecord(value.effective_execution_digest) ||
    !hasExactKeys(value.effective_execution_digest, ["algorithm", "value"]) ||
    value.effective_execution_digest.algorithm !== "sha256-canonical-json-v1" ||
    typeof value.effective_execution_digest.value !== "string" ||
    !/^sha256-[0-9a-f]{64}$/.test(value.effective_execution_digest.value)
  ) throw new ApiError("执行计划功能证据结构无效", 502, value);

  const requested = normalizeFeatureConfiguration(
    features.requested,
    new Set(Object.keys(features.requested.by_segment)),
  );
  if (
    !requested ||
    requested.template_bundle_version !== value.template_bundle_version ||
    Object.keys(requested.by_segment).length !==
      Object.keys(features.requested.by_segment).length
  ) throw new ApiError("执行计划请求功能结构无效", 502, features.requested);

  const effectiveBySegment: TimelineCompileReport["features"]["effective_by_segment"] = {};
  for (const [segmentId, rawResolution] of Object.entries(features.effective_by_segment)) {
    if (!isBoundedOpaqueId(segmentId, 128)) {
      throw new ApiError("执行计划有效功能分段 ID 无效", 502, segmentId);
    }
    effectiveBySegment[segmentId] = parseEffectiveSegmentResolution(rawResolution);
  }
  const planIds = plans.map((plan) => plan.segment_id);
  if (
    Object.keys(effectiveBySegment).length !== planIds.length ||
    planIds.some((segmentId) => !(segmentId in effectiveBySegment)) ||
    plans.some((plan) => {
      const effective = effectiveBySegment[plan.segment_id];
      return effective.backend !== plan.backend || effective.family !== plan.model_family;
    })
  ) throw new ApiError("执行计划有效功能范围与分段不一致", 502, features.effective_by_segment);

  const resolutions: TimelineCompileReport["features"]["resolutions"] = [];
  for (const rawResolution of features.resolutions) {
    if (
      !isRecord(rawResolution) ||
      !hasExactKeys(rawResolution, [
        "segment_id",
        "unit_id",
        "feature_id",
        "version",
        "backend",
        "family",
        "template_id",
        "resolution",
        "adapter_fingerprint",
        "capability",
      ]) ||
      !isBoundedOpaqueId(rawResolution.segment_id, 128) ||
      !isBoundedOpaqueId(rawResolution.unit_id, 256) ||
      !isCapabilityIdentifier(rawResolution.feature_id) ||
      !Number.isSafeInteger(rawResolution.version) ||
      Number(rawResolution.version) < 1 ||
      Number(rawResolution.version) > 2_147_483_647 ||
      (rawResolution.backend !== "standard" && rawResolution.backend !== "raylight") ||
      (rawResolution.family !== "fl2va" && rawResolution.family !== "ref2va") ||
      (rawResolution.template_id !== "h3_standard_segment" &&
        rawResolution.template_id !== "h3_raylight_segment") ||
      !isCapabilityDigest(rawResolution.adapter_fingerprint)
    ) throw new ApiError("执行计划功能解析证据无效", 502, rawResolution);
    resolutions.push({
      segment_id: rawResolution.segment_id,
      unit_id: rawResolution.unit_id,
      feature_id: rawResolution.feature_id,
      version: Number(rawResolution.version),
      backend: rawResolution.backend,
      family: rawResolution.family,
      template_id: rawResolution.template_id,
      resolution: parseFeatureResolutionEvidence(rawResolution.resolution),
      adapter_fingerprint: rawResolution.adapter_fingerprint,
      capability: parseFeatureCapabilityEvaluation(rawResolution.capability),
    });
  }
  const actualResolutionKeys = resolutions.map((resolution) =>
    `${resolution.segment_id}\u0000${resolution.unit_id}\u0000${resolution.feature_id}\u0000${resolution.version}`);
  if (new Set(actualResolutionKeys).size !== actualResolutionKeys.length) {
    throw new ApiError("执行计划功能解析证据包含重复身份", 502, features.resolutions);
  }
  for (const plan of plans) {
    const effective = effectiveBySegment[plan.segment_id];
    const actual = resolutions.filter((item) => item.segment_id === plan.segment_id);
    if (
      actual.length === 0 ||
      actual.length !== effective.features.length ||
      actual.some((item, index) => {
        const feature = effective.features[index];
        return item.unit_id !== effective.unit_id ||
          item.backend !== effective.backend ||
          item.family !== effective.family ||
          item.template_id !== effective.template_id ||
          item.feature_id !== feature.id ||
          item.version !== feature.version ||
          item.resolution.state !== feature.state ||
          item.adapter_fingerprint !== feature.adapter_fingerprint ||
          JSON.stringify(item.capability) !== JSON.stringify(feature.capability);
      })
    ) throw new ApiError(
      "执行计划功能解析证据与有效功能不一致",
      502,
      features.resolutions,
    );
  }
  if (resolutions.some((item) => !plansById.has(item.segment_id))) {
    throw new ApiError("执行计划功能解析证据引用未编译分段", 502, features.resolutions);
  }

  const notices: TimelineCompileReport["features"]["notices"] = [];
  const resolutionNoticeKeys = new Set(resolutions.map((item) =>
    `${item.segment_id}\u0000${item.unit_id}\u0000${item.feature_id}`));
  for (const rawNotice of features.notices) {
    if (
      !isRecord(rawNotice) ||
      !hasExactKeys(rawNotice, ["segment_id", "unit_id", "feature_id", "message"]) ||
      !isBoundedOpaqueId(rawNotice.segment_id, 128) ||
      !isBoundedOpaqueId(rawNotice.unit_id, 256) ||
      !isCapabilityIdentifier(rawNotice.feature_id) ||
      !isBoundedNonemptyString(rawNotice.message, 4_096) ||
      !rawNotice.message.trim() ||
      !resolutionNoticeKeys.has(
        `${rawNotice.segment_id}\u0000${rawNotice.unit_id}\u0000${rawNotice.feature_id}`,
      )
    ) throw new ApiError("执行计划功能提示结构无效", 502, rawNotice);
    notices.push({
      segment_id: rawNotice.segment_id,
      unit_id: rawNotice.unit_id,
      feature_id: rawNotice.feature_id,
      message: rawNotice.message,
    });
  }

  return {
    template_bundle_version: Number(value.template_bundle_version),
    host_capability_revision: value.host_capability_revision,
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
    features: {
      requested,
      effective_by_segment: effectiveBySegment,
      resolutions,
      notices,
    },
    effective_execution_digest: {
      algorithm: "sha256-canonical-json-v1",
      value: value.effective_execution_digest.value,
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
    value.version === 5 &&
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

function parseTimelineAuthority(value: unknown): TimelineAuthority {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["document", "revision"]) ||
    !isNonNegativeInteger(value.revision)
  ) throw new ApiError("时间线权威响应结构无效", 502, value);
  return {
    document: parseTimelineProjectResponse(value.document),
    revision: value.revision,
  };
}

function timelineAuthorityWriteBody(
  document: TimelineProject,
  expectedRevision: number,
): string {
  if (!isNonNegativeInteger(expectedRevision)) {
    throw new RangeError("时间线 expected revision 必须是非负安全整数");
  }
  return JSON.stringify({ document, expected_revision: expectedRevision });
}

function timelineTaskBody(payload: TimelineTaskRequest): string {
  const raw: unknown = payload;
  if (
    !isRecord(raw) ||
    !("config" in raw) ||
    Object.keys(raw).some((key) => key !== "config" && key !== "segment_ids")
  ) throw new TypeError("时间线任务请求必须携带显式 v5 config");
  const config = normalizeTimelineProject(raw.config);
  if (!config) throw new TypeError("时间线任务 config 不是严格 v5 文档");
  if (raw.segment_ids === undefined) return JSON.stringify({ config });
  if (
    !Array.isArray(raw.segment_ids) ||
    raw.segment_ids.length < 1 ||
    raw.segment_ids.length > 128 ||
    !raw.segment_ids.every((segmentId) => isBoundedOpaqueId(segmentId, 128)) ||
    new Set(raw.segment_ids).size !== raw.segment_ids.length ||
    raw.segment_ids.some((segmentId) =>
      !config.segments.some((segment) => segment.id === segmentId))
  ) throw new TypeError("时间线任务 segment_ids 结构无效");
  return JSON.stringify({ config, segment_ids: [...raw.segment_ids] });
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
  return {
    projects: value.projects.map(parseProjectSummary),
  };
}

function parseDocumentDigest(value: unknown) {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["algorithm", "value"]) ||
    !["fnv1a32-json-stringify-v1", "sha256-canonical-json-v1"].includes(String(value.algorithm)) ||
    typeof value.value !== "string" ||
    !(value.algorithm === "fnv1a32-json-stringify-v1"
      ? /^fnv1a-[0-9a-f]{8}$/.test(value.value)
      : /^sha256-[0-9a-f]{64}$/.test(value.value))
  ) throw new ApiError("项目导入摘要结构无效", 502, value);
  return {
    algorithm: value.algorithm as ProjectImportPreflightResponse["input_digest"]["algorithm"],
    value: value.value,
  };
}

function parseImportModelStack(
  value: unknown,
): ProjectImportCreativeSelection["model_stack"] | null {
  if (!isRecord(value)) return null;
  const roles = ["fl2va", "ref2va", "clip", "video_vae", "audio_vae"] as const;
  if (!hasExactKeys(value, roles)) return null;
  const result = {} as ProjectImportCreativeSelection["model_stack"];
  for (const role of roles) {
    const selection = value[role];
    if (
      !isRecord(selection) ||
      !hasExactKeys(selection, ["filename"]) ||
      !(selection.filename === null || (
        typeof selection.filename === "string" &&
        selection.filename.length >= 1 &&
        selection.filename.length <= 1_024
      ))
    ) return null;
    result[role] = { filename: selection.filename };
  }
  return result;
}

function parseImportLoraSelection(value: unknown) {
  const selection = normalizeFeatureSelection(value);
  if (!selection || !isRecord(selection.params.by_family)) return null;
  const byFamily = selection.params.by_family;
  if (!hasExactKeys(byFamily, ["fl2va", "ref2va"])) return null;
  for (const family of ["fl2va", "ref2va"] as const) {
    const candidate = byFamily[family];
    if (
      !isRecord(candidate) ||
      !hasExactKeys(candidate, ["enabled", "filename", "strength"]) ||
      typeof candidate.enabled !== "boolean" ||
      !(candidate.filename === null || (
        typeof candidate.filename === "string" &&
        candidate.filename.length >= 1 &&
        candidate.filename.length <= 1_024
      )) ||
      typeof candidate.strength !== "number" ||
      !Number.isFinite(candidate.strength) ||
      candidate.strength < -10 ||
      candidate.strength > 10
    ) return null;
  }
  return selection;
}

function parseProjectImportCreativeSelection(
  value: unknown,
): ProjectImportCreativeSelection | null {
  if (!isRecord(value) || !hasExactKeys(value, ["model_stack", "lora"])) return null;
  const modelStack = parseImportModelStack(value.model_stack);
  const lora = parseImportLoraSelection(value.lora);
  return modelStack && lora ? { model_stack: modelStack, lora } : null;
}

function parseProjectImportLegacyCreativeContext(
  value: unknown,
): ProjectImportLegacyCreativeContext | null {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "schema_version",
      "model_stack",
      "lora",
      "explicit_standard_lora_overrides",
    ]) ||
    value.schema_version !== 1 ||
    !Array.isArray(value.explicit_standard_lora_overrides) ||
    value.explicit_standard_lora_overrides.length > 2
  ) return null;
  const modelStack = parseImportModelStack(value.model_stack);
  const lora = parseImportLoraSelection(value.lora);
  if (!modelStack || !lora) return null;
  const overrides: ProjectImportLegacyCreativeContext["explicit_standard_lora_overrides"] = [];
  for (const candidate of value.explicit_standard_lora_overrides) {
    if (
      !isRecord(candidate) ||
      !hasExactKeys(candidate, ["family", "model_filename", "lora_filename", "loader"]) ||
      (candidate.family !== "fl2va" && candidate.family !== "ref2va") ||
      !["dedicated", "bypass_model_only", "model_only"].includes(String(candidate.loader)) ||
      typeof candidate.model_filename !== "string" ||
      candidate.model_filename.length < 1 || candidate.model_filename.length > 1_024 ||
      typeof candidate.lora_filename !== "string" ||
      candidate.lora_filename.length < 1 || candidate.lora_filename.length > 1_024
    ) return null;
    overrides.push({
      family: candidate.family,
      model_filename: candidate.model_filename,
      lora_filename: candidate.lora_filename,
      loader: candidate.loader as ProjectImportLegacyCreativeContext["explicit_standard_lora_overrides"][number]["loader"],
    });
  }
  if (
    overrides.some((entry, index) => index > 0 && overrides[index - 1].family >= entry.family) ||
    new Set(overrides.map((entry) => entry.family)).size !== overrides.length
  ) return null;
  return {
    schema_version: 1,
    model_stack: modelStack,
    lora,
    explicit_standard_lora_overrides: overrides,
  };
}

function projectImportPreflightBody(payload: ProjectImportPreflightRequest): string {
  if (
    typeof payload.title !== "string" ||
    payload.title.length > 256 ||
    !isRecord(payload.document)
  ) throw new TypeError("项目导入预检请求结构无效");
  const runtime = payload.legacy_runtime_settings === undefined
    ? null
    : parseLegacyRuntimeSettingsV1(payload.legacy_runtime_settings);
  const context = payload.legacy_creative_context === undefined
    ? null
    : parseProjectImportLegacyCreativeContext(payload.legacy_creative_context);
  const selection = payload.creative_selection === undefined
    ? null
    : parseProjectImportCreativeSelection(payload.creative_selection);
  const supplied = [
    payload.legacy_runtime_settings,
    payload.legacy_creative_context,
    payload.creative_selection,
  ].filter((value) => value !== undefined).length;
  if (
    supplied > 1 ||
    (payload.legacy_runtime_settings !== undefined && !runtime) ||
    (payload.legacy_creative_context !== undefined && !context) ||
    (payload.creative_selection !== undefined && !selection)
  ) throw new TypeError("项目导入预检创作上下文无效");
  return JSON.stringify({
    title: payload.title,
    document: payload.document,
    ...(runtime ? { legacy_runtime_settings: runtime } : {}),
    ...(context ? { legacy_creative_context: context } : {}),
    ...(selection ? { creative_selection: selection } : {}),
  });
}

function parseProjectImportPreflight(value: unknown): ProjectImportPreflightResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "schema_version", "status", "input_digest", "proposed_document",
      "missing_context", "missing_model_bindings", "capability_issues",
      "commit_token", "expires_at",
    ]) ||
    value.schema_version !== 1 ||
    !["ready", "needs_input"].includes(String(value.status)) ||
    !isStringArray(value.missing_context) ||
    !isStringArray(value.missing_model_bindings) ||
    !Array.isArray(value.capability_issues) ||
    value.capability_issues.some((issue) => !isRecord(issue)) ||
    !(value.commit_token === null ||
      (typeof value.commit_token === "string" && value.commit_token.length >= 32 && value.commit_token.length <= 256)) ||
    !(value.expires_at === null || typeof value.expires_at === "string")
  ) throw new ApiError("项目导入预检响应结构无效", 502, value);
  const proposed = value.proposed_document === null
    ? null
    : normalizeTimelineProject(value.proposed_document);
  if (
    value.proposed_document !== null && !proposed ||
    (value.status === "ready" && (!proposed || value.commit_token === null || value.expires_at === null)) ||
    (value.status === "needs_input" && value.commit_token !== null)
  ) throw new ApiError("项目导入预检响应状态无效", 502, value);
  return {
    schema_version: 1,
    status: value.status as ProjectImportPreflightResponse["status"],
    input_digest: parseDocumentDigest(value.input_digest),
    proposed_document: proposed,
    missing_context: [...value.missing_context],
    missing_model_bindings: [...value.missing_model_bindings],
    capability_issues: value.capability_issues.map((issue) => ({ ...issue })),
    commit_token: value.commit_token as string | null,
    expires_at: value.expires_at as string | null,
  };
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

function parseStorageConfiguration(value: unknown): StorageConfiguration {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, ["active_database_path"]) ||
    !isStoragePath(value.active_database_path)
  ) throw new ApiError("数据存储响应结构无效", 502, value);
  return {
    active_database_path: value.active_database_path,
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

function parseUniqueOpaqueIds(value: unknown): string[] | null {
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.length > 128 ||
    !value.every((item) => typeof item === "string" && item.trim() === item && item.length > 0) ||
    new Set(value).size !== value.length
  ) return null;
  return [...value];
}

function sameStrings(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length &&
    left.every((item, index) => item === right[index]);
}

function parseAssetList(value: unknown): AssetListResponse {
  if (!isRecord(value)) {
    throw new ApiError("素材列表响应结构无效", 502, value);
  }
  if (
    !hasExactKeys(value, ["assets", "outputs_preserved"]) ||
    value.outputs_preserved !== true ||
    !Array.isArray(value.assets)
  ) throw new ApiError("素材列表响应结构无效", 502, value);
  const assets: AssetReference[] = [];
  for (const item of value.assets) {
    if (!isRecord(item) || !["image", "audio", "video"].includes(String(item.kind))) {
      throw new ApiError("素材列表响应结构无效", 502, value);
    }
    const normalized = normalizeAssetReference(item, item.kind as AssetKind, { completeWireShape: true });
    if (!normalized) throw new ApiError("素材列表响应结构无效", 502, value);
    assets.push(normalized);
  }
  if (new Set(assets.map((asset) => asset.id)).size !== assets.length) {
    throw new ApiError("素材列表响应结构无效", 502, value);
  }
  return {
    assets,
    outputs_preserved: true,
  };
}

function parseAssetTrashBatch(
  value: unknown,
  expected?: { assetIds: readonly string[]; cascade: boolean },
): AssetTrashBatch {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "batch_id",
      "asset_ids",
      "assets",
      "cascade",
      "unbound_usages",
      "unbound_usages_by_asset",
      "created_at",
      "remote_files_preserved",
    ]) ||
    typeof value.batch_id !== "string" || !value.batch_id ||
    typeof value.cascade !== "boolean" ||
    typeof value.created_at !== "string" || !value.created_at ||
    value.remote_files_preserved !== true ||
    !Array.isArray(value.assets) ||
    !isStringArray(value.unbound_usages)
  ) throw new ApiError("素材回收批次响应结构无效", 502, value);
  const assetIds = parseUniqueOpaqueIds(value.asset_ids);
  const usagesByAsset = parseStringArrayRecord(value.unbound_usages_by_asset);
  if (
    assetIds === null ||
    usagesByAsset === null ||
    !sameStrings(Object.keys(usagesByAsset).sort(), [...assetIds].sort()) ||
    !sameStrings(
      value.unbound_usages,
      assetIds.flatMap((assetId) => usagesByAsset[assetId]),
    ) ||
    (expected && (
      !sameStrings(assetIds, expected.assetIds) ||
      value.cascade !== expected.cascade
    ))
  ) throw new ApiError("素材回收批次响应结构无效", 502, value);
  const assets: AssetReference[] = [];
  for (const [index, item] of value.assets.entries()) {
    if (
      !isRecord(item) ||
      !["image", "audio", "video"].includes(String(item.kind))
    ) throw new ApiError("素材回收批次响应结构无效", 502, value);
    const normalized = normalizeAssetReference(item, item.kind as AssetKind, { completeWireShape: true });
    if (!normalized || normalized.id !== assetIds[index]) {
      throw new ApiError("素材回收批次响应结构无效", 502, value);
    }
    assets.push(normalized);
  }
  if (assets.length !== assetIds.length) {
    throw new ApiError("素材回收批次响应结构无效", 502, value);
  }
  return {
    batch_id: value.batch_id,
    asset_ids: assetIds,
    assets,
    cascade: value.cascade,
    unbound_usages: [...value.unbound_usages],
    unbound_usages_by_asset: Object.fromEntries(
      assetIds.map((assetId) => [assetId, usagesByAsset[assetId]]),
    ),
    created_at: value.created_at,
    remote_files_preserved: true,
  };
}

function parseAssetTrashList(value: unknown): AssetTrashListResponse {
  if (!isRecord(value)) {
    throw new ApiError("素材回收站响应结构无效", 502, value);
  }
  if (
    !hasExactKeys(value, ["batches", "remote_files_preserved"]) ||
    !Array.isArray(value.batches) ||
    value.remote_files_preserved !== true
  ) throw new ApiError("素材回收站响应结构无效", 502, value);
  const batches = value.batches.map((item) => parseAssetTrashBatch(item));
  const batchIds = batches.map((item) => item.batch_id);
  const assetIds = batches.flatMap((item) => item.asset_ids);
  if (
    new Set(batchIds).size !== batchIds.length ||
    new Set(assetIds).size !== assetIds.length
  ) throw new ApiError("素材回收站响应结构无效", 502, value);
  return { batches, remote_files_preserved: true };
}

function parseAssetTrashRestoreResponse(
  value: unknown,
  expectedBatchId: string,
  expectedMode: AssetTrashRestoreMode,
): AssetTrashRestoreResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "batch_id",
      "restored_asset_ids",
      "restored_references",
      "mode",
      "remote_files_preserved",
    ]) ||
    value.batch_id !== expectedBatchId ||
    value.mode !== expectedMode ||
    typeof value.restored_references !== "boolean" ||
    (expectedMode === "registration_only" && value.restored_references) ||
    value.remote_files_preserved !== true
  ) throw new ApiError("素材恢复响应结构无效", 502, value);
  const restoredAssetIds = parseUniqueOpaqueIds(value.restored_asset_ids);
  if (restoredAssetIds === null) {
    throw new ApiError("素材恢复响应结构无效", 502, value);
  }
  return {
    batch_id: expectedBatchId,
    restored_asset_ids: restoredAssetIds,
    restored_references: value.restored_references,
    mode: expectedMode,
    remote_files_preserved: true,
  };
}

function parseAssetTrashPurgeResponse(
  value: unknown,
  expectedBatchId: string,
): AssetTrashPurgeResponse {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      "batch_id",
      "purged_asset_ids",
      "remote_files_preserved",
    ]) ||
    value.batch_id !== expectedBatchId ||
    value.remote_files_preserved !== true
  ) throw new ApiError("素材回收批次清理响应结构无效", 502, value);
  const purgedAssetIds = parseUniqueOpaqueIds(value.purged_asset_ids);
  if (purgedAssetIds === null) {
    throw new ApiError("素材回收批次清理响应结构无效", 502, value);
  }
  return {
    batch_id: expectedBatchId,
    purged_asset_ids: purgedAssetIds,
    remote_files_preserved: true,
  };
}

function assetTrashIds(assetIds: readonly string[]): string[] {
  const parsed = parseUniqueOpaqueIds(assetIds);
  if (parsed === null || parsed.length > 128) {
    throw new RangeError("素材回收批次必须包含 1 至 128 个不重复的稳定 ID");
  }
  return parsed;
}

function assetTrashBatchId(batchId: string): string {
  if (!batchId || batchId.trim() !== batchId) {
    throw new Error("素材回收批次 ID 无效");
  }
  return batchId;
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

function activeProjectQuery(projectId?: string): string {
  return projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
}

export const directorApi = {
  getDirectorDeckConfig: (signal?: AbortSignal) =>
    request<unknown>("/config", { signal }).then(parseDirectorDeckConfig),
  getFeatureCatalog: (etag?: string, signal?: AbortSignal) =>
    requestFeatureCatalog(etag, signal),
  preflightFeatures: (payload: FeaturePreflightRequest, signal?: AbortSignal) =>
    request<unknown>("/features/preflight", {
      method: "POST",
      signal,
      body: JSON.stringify(payload),
    }).then(parseFeaturePreflight),
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
  getSettings: (signal?: AbortSignal) =>
    request<unknown>("/settings", { signal }).then((value) => {
      const settings = parseRuntimeSettingsV3(value);
      if (!settings) throw new ApiError("运行设置响应包含过期或无效 schema", 502, value);
      return settings;
    }),
  getSettingsAuthority: (signal?: AbortSignal) =>
    request<unknown>("/settings/authority", { signal }).then(parseRuntimeSettingsAuthority),
  getSettingsMigrationNotices: (signal?: AbortSignal) =>
    request<unknown>("/settings/migration-notices", { signal })
      .then(parseRuntimeSettingsMigrationNotices),
  updateSettingsAuthority: (settings: RuntimeSettings, expectedAuthorityToken: string) => {
    if (!/^[0-9a-f]{64}$/.test(expectedAuthorityToken)) {
      throw new Error("运行设置 expected authority token 无效");
    }
    return request<unknown>("/settings/authority", {
      method: "PUT",
      body: JSON.stringify({
        document: settings,
        expected_authority_token: expectedAuthorityToken,
        schema_version: 3,
      }),
    }).then(parseRuntimeSettingsAuthority);
  },
  getRayLightSetup: (signal?: AbortSignal) =>
    request<RayLightSetupStatus>("/raylight/setup", { signal }),
  installRayLight: () =>
    request<RayLightInstallSnapshot>("/raylight/setup/install", { method: "POST" }),
  cancelRayLightInstall: () =>
    request<RayLightInstallSnapshot>("/raylight/setup/cancel", { method: "POST" }),
  getMediaSetup: (signal?: AbortSignal) =>
    request<MediaToolsStatus>("/media/setup", { signal }),
  installFfmpeg: () =>
    request<MediaToolInstallSnapshot>("/media/ffmpeg/install", { method: "POST" }),
  cancelFfmpegInstall: () =>
    request<MediaToolInstallSnapshot>("/media/ffmpeg/cancel", { method: "POST" }),
  getStorage: (signal?: AbortSignal) =>
    request<unknown>("/storage", { signal }).then(parseStorageConfiguration),
  // The only endpoint is the embedded host instance; the probe takes no URL.
  testConnection: () =>
    request<ConnectionTestResult>("/capabilities", { method: "POST" }),
  confirmRayLightRuntimeRecovery: (
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
  getTimelineAuthority: (signal?: AbortSignal) =>
    request<unknown>("/timeline/authority", { signal }).then(parseTimelineAuthority),
  updateTimelineAuthority: (
    timeline: TimelineProject,
    expectedRevision: number,
  ) => request<unknown>("/timeline/authority", {
    method: "PUT",
    body: timelineAuthorityWriteBody(timeline, expectedRevision),
  }).then(parseTimelineAuthority),
  createTimelineTask: (payload: TimelineTaskRequest) =>
    request<unknown>("/timeline/jobs", {
      method: "POST",
      body: timelineTaskBody(payload),
    }).then(parseGenerationTask),
  compileTimeline: (payload: TimelineTaskRequest) =>
    request<unknown>("/timeline/compile", {
      method: "POST",
      body: timelineTaskBody(payload),
    }).then(parseTimelineCompileReport),

  // --- Project management (multi-project) ---

  listProjects: (signal?: AbortSignal) =>
    request<unknown>("/projects", { signal }).then(parseProjectList),
  createProject: (title?: string) =>
    request<unknown>("/projects", {
      method: "POST",
      body: JSON.stringify({ title: title ?? "" }),
    }).then(parseProjectSummary),
  preflightProjectImport: (payload: ProjectImportPreflightRequest) =>
    request<unknown>("/projects/import/preflight", {
      method: "POST",
      body: projectImportPreflightBody(payload),
    }).then(parseProjectImportPreflight),
  commitProjectImport: (payload: ProjectImportCommitRequest) =>
    request<unknown>("/projects/import/commit", {
      method: "POST",
      body: JSON.stringify(payload),
    }).then(parseProjectSummary),
  getLatestProjectMigrationReceipt: (projectId: string, signal?: AbortSignal) =>
    request<unknown>(
      `/projects/${encodeURIComponent(projectId)}/migration-receipts/latest?from=4&to=5`,
      { signal },
    ).then((value): ProjectMigrationReceipt => {
      const receipt = parseProjectMigrationReceipt(value);
      if (!receipt) throw new ApiError("项目迁移回执结构无效", 502, value);
      return receipt;
    }),
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
  getProjectTimelineAuthority: (projectId: string, signal?: AbortSignal) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}/timeline/authority`, { signal })
      .then(parseTimelineAuthority),
  updateProjectTimelineAuthority: (
    projectId: string,
    timeline: TimelineProject,
    expectedRevision: number,
  ) => request<unknown>(`/projects/${encodeURIComponent(projectId)}/timeline/authority`, {
    method: "PUT",
    body: timelineAuthorityWriteBody(timeline, expectedRevision),
  }).then(parseTimelineAuthority),
  compileProjectTimeline: (projectId: string, payload: TimelineTaskRequest) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}/compile`, {
      method: "POST",
      body: timelineTaskBody(payload),
    }).then(parseTimelineCompileReport),
  createProjectTask: (projectId: string, payload: TimelineTaskRequest) =>
    request<unknown>(`/projects/${encodeURIComponent(projectId)}/jobs`, {
      method: "POST",
      body: timelineTaskBody(payload),
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
        const normalized = normalizeAssetReference(asset, kind, { completeWireShape: true });
        if (!normalized) {
          throw new ApiError("素材上传响应缺少有效的稳定 ID", 502, asset);
        }
        return normalized;
      },
    );
  },

  listAssets(kind?: AssetKind, signal?: AbortSignal): Promise<AssetListResponse> {
    const query = kind ? `?kind=${encodeURIComponent(kind)}` : "";
    return request<unknown>(`/assets${query}`, { signal }).then(parseAssetList);
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
  trashAssets(
    assetIds: readonly string[],
    cascade = false,
  ): Promise<AssetTrashBatch> {
    const normalizedIds = assetTrashIds(assetIds);
    if (typeof cascade !== "boolean") throw new TypeError("素材回收 cascade 标志无效");
    return request<unknown>("/asset-trash", {
      method: "POST",
      body: JSON.stringify({ asset_ids: normalizedIds, cascade }),
    }).then((value) => parseAssetTrashBatch(value, {
      assetIds: normalizedIds,
      cascade,
    }));
  },
  listAssetTrash(signal?: AbortSignal): Promise<AssetTrashListResponse> {
    return request<unknown>("/asset-trash", { signal }).then(parseAssetTrashList);
  },
  restoreAssetTrash(
    batchId: string,
    mode: AssetTrashRestoreMode,
  ): Promise<AssetTrashRestoreResponse> {
    const normalizedBatchId = assetTrashBatchId(batchId);
    if (mode !== "registration_only" && mode !== "with_references") {
      throw new Error("素材恢复模式无效");
    }
    return request<unknown>(
      `/asset-trash/${encodeURIComponent(normalizedBatchId)}/restore`,
      { method: "POST", body: JSON.stringify({ mode }) },
    ).then((value) => parseAssetTrashRestoreResponse(
      value,
      normalizedBatchId,
      mode,
    ));
  },
  purgeAssetTrash(batchId: string): Promise<AssetTrashPurgeResponse> {
    const normalizedBatchId = assetTrashBatchId(batchId);
    return request<unknown>(`/asset-trash/${encodeURIComponent(normalizedBatchId)}`, {
      method: "DELETE",
    }).then((value) => parseAssetTrashPurgeResponse(value, normalizedBatchId));
  },

  detectRV2VShots: (payload: RV2VShotDetectionRequest) =>
    request<RV2VShotDetectionResponse>("/rv2v/detect-shots", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  createTask: (payload: CreateTaskRequest) =>
    request<unknown>("/jobs", { method: "POST", body: JSON.stringify(payload) }).then(parseGenerationTask),
  async listTasks(
    signal?: AbortSignal,
    projectId?: string,
    statuses: readonly TaskStatus[] = [],
  ): Promise<TaskListResponse> {
    const jobs: GenerationTask[] = [];
    const seen = new Set<string>();
    let offset = 0;
    let summary: TaskListResponse["summary"];
    let total = 0;
    const projectQuery = projectId ? `&project_id=${encodeURIComponent(projectId)}` : "";
    const statusQuery = statuses.map(
      (status) => `&status=${encodeURIComponent(status)}`,
    ).join("");
    for (;;) {
      const page = await request<unknown>(
        `/jobs?limit=256&offset=${offset}&sort_by=created_at&sort_order=asc${projectQuery}${statusQuery}`,
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
  getTask: (taskId: string, signal?: AbortSignal, activeProjectId?: string) =>
    request<unknown>(
      `/jobs/${encodeURIComponent(taskId)}${activeProjectQuery(activeProjectId)}`,
      { signal },
    ).then(parseGenerationTask),
  cancelTask: (taskId: string, activeProjectId?: string) =>
    request<unknown>(
      `/jobs/${encodeURIComponent(taskId)}/cancel${activeProjectQuery(activeProjectId)}`,
      { method: "POST" },
    ).then(parseGenerationTask),
  confirmComfyRestartRecovery: (taskId: string, activeProjectId?: string) =>
    request<unknown>(
      `/jobs/${encodeURIComponent(taskId)}/recovery/confirm-comfy-restart${activeProjectQuery(activeProjectId)}`,
      {
        method: "POST",
        body: JSON.stringify({ confirmation: "comfyui_process_restarted" }),
      },
    ).then(parseGenerationTask),
  cancelTasks: (taskIds: string[], activeProjectId?: string) =>
    request<unknown>(`/jobs/cancel${activeProjectQuery(activeProjectId)}`, {
      method: "POST",
      body: JSON.stringify({ job_ids: taskIds }),
    }).then(parseTaskBulkCancel),
  getTaskProject: (taskId: string) =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}/project`).then(parseTaskProjectSnapshot),
  saveHistoricalJobAsProject: (taskId: string, title = "") =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}/save-as-project`, {
      method: "POST",
      body: JSON.stringify({ title }),
    }).then(parseProjectSummary),
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
      const normalized = normalizeAssetReference(asset, "video", { completeWireShape: true });
      if (!normalized) throw new ApiError("任务输出导入响应结构无效", 502, asset);
      return normalized;
    }),
  deleteTask: (taskId: string) =>
    request<unknown>(`/jobs/${encodeURIComponent(taskId)}`, { method: "DELETE" })
      .then((value) => parseJobDeleteResponse(value, taskId)),
  clearTerminalTasks: () =>
    request<unknown>("/jobs", { method: "DELETE" }).then(parseJobClearResponse),
};
