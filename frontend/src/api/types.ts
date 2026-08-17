import type { AssetReference, GenerationMode, ModeDraft } from "../domain/modes";
import type {
  DerivedGenerationRecipe,
  TimelineGenerationMode,
  TimelineProject,
} from "../domain/timelineProject";

export type ConnectionState = "online" | "offline" | "checking" | "unknown";
export type DeviceTarget = "default" | "cpu" | `gpu:${number}`;
export type VaeDeviceTarget = Exclude<DeviceTarget, "cpu">;
export type ModelRole = "fl2va" | "ref2va" | "clip" | "video_vae" | "audio_vae";
export type DiffusionModelRole = "fl2va" | "ref2va";
export type ExecutionBackendPreference = "auto" | "standard" | "raylight";
export type ResolvedExecutionBackend = Exclude<ExecutionBackendPreference, "auto">;
export type LoraLoader =
  | "auto"
  | "dedicated"
  | "bypass_model_only"
  | "model_only";
export type StandardLoraLoader = Exclude<LoraLoader, "auto">;

export interface StandardLoraLoaderOverride {
  loader: StandardLoraLoader;
  lora_name: string;
  model_filename: string;
  comfy_origin: string;
}
export type MemoryPolicy = "keep_resident";
export type RayLightResidencyPolicy =
  | "release_after_sampling"
  | "keep_until_switch";

export interface ModelBinding<D extends DeviceTarget = DeviceTarget> {
  filename: string;
  device: D;
}

export interface RayLightProfile {
  /** ComfyUI-process logical GPU indexes; never physical PCI device indexes. */
  gpu_select: number[];
  ulysses_degree: number;
  ring_degree: number;
  /** RayLight conditioning/data-parallel topology degrees are fixed in v1. */
  cfg_degree: 1;
  dp_degree: 1;
  fsdp: boolean;
  cpu_offload: boolean;
}

export interface DiffusionModelBinding extends ModelBinding {
  lora_name: string | null;
  lora_strength: number;
  lora_loader: LoraLoader;
  standard_lora_loader_override: StandardLoraLoaderOverride | null;
  lora_low_vram: boolean;
  backend: ExecutionBackendPreference;
  raylight: RayLightProfile;
}

export function resolveExecutionBackend(
  binding: DiffusionModelBinding,
): ResolvedExecutionBackend {
  return binding.raylight.gpu_select.length >= 2 ? "raylight" : "standard";
}

export function describeLoraLoader(binding: DiffusionModelBinding): string {
  if (resolveExecutionBackend(binding) === "raylight") return "RayLoraLoader";
  switch (binding.standard_lora_loader_override?.loader) {
    case "dedicated": return "显式：MiniMax H3 专用";
    case "bypass_model_only": return "显式：量化旁路 Model Only";
    case "model_only": return "显式：ComfyUI 通用 Model Only";
    default: return "自动探测";
  }
}

export interface SettingsModels {
  fl2va: DiffusionModelBinding;
  ref2va: DiffusionModelBinding;
  clip: ModelBinding;
  video_vae: ModelBinding<VaeDeviceTarget>;
  audio_vae: ModelBinding<VaeDeviceTarget>;
}

/** Exact GET/PUT /api/settings payload. */
export interface RuntimeSettings {
  comfy_url: string;
  client_id: string;
  memory_policy: MemoryPolicy;
  raylight_residency_policy: RayLightResidencyPolicy;
  models: SettingsModels;
}

export interface RuntimeSettingsAuthority {
  settings: RuntimeSettings;
  authority_token: string;
}

/** Exact GET/PUT /api/storage payload. Storage changes never belong to RuntimeSettings. */
export type StorageConfigurationSource =
  | "explicit"
  | "environment"
  | "bootstrap"
  | "legacy"
  | "default";

export interface StorageConfiguration {
  active_database_path: string;
  active_database_identity: string;
  configured_database_path: string;
  recommended_database_path: string;
  source: StorageConfigurationSource;
  restart_required: boolean;
}

/** Exact POST /api/storage/migrate response. */
export interface StorageMigrationResult extends StorageConfiguration {
  migrated_from: string;
  migrated_to: string;
}

/**
 * A newly selected RayLight backend defaults to keyed residency. Merely
 * editing an already-RayLight model/LoRA must not override an intentional
 * "release after sampling" choice.
 */
export function rayLightResidencyPolicyAfterBindingChange(
  settings: Pick<RuntimeSettings, "models" | "raylight_residency_policy">,
  role: DiffusionModelRole,
  nextBinding: DiffusionModelBinding,
): RayLightResidencyPolicy {
  const currentBinding = settings.models[role];
  const currentBackend = resolveExecutionBackend(currentBinding);
  const nextBackend = resolveExecutionBackend(nextBinding);
  // Default to keeping weights only when the GPU pool actually changes the
  // resolved runtime from Standard to RayLight. Hidden legacy preferences are
  // not a second routing authority.
  if (
    settings.raylight_residency_policy === "release_after_sampling" &&
    nextBackend === "raylight" &&
    currentBackend !== "raylight"
  ) return "keep_until_switch";

  return settings.raylight_residency_policy;
}

export interface CapabilityReport {
  connection: ConnectionState;
  supported_modes: GenerationMode[];
  supports_cancel: boolean;
  available_nodes: string[];
  missing_nodes: string[];
  message?: string;
  latency_ms?: number;
  features?: Record<string, unknown>;
  native_timeline?: {
    supported: boolean;
    modes: TimelineGenerationMode[];
    continuity: boolean;
  };
  execution_backends?: Record<ResolvedExecutionBackend, {
    available: boolean;
    missing_nodes: string[];
    conditional_requirements?: {
      lora: {
        available: boolean;
        missing_nodes: string[];
      };
    };
  }>;
}

export interface GPUResource {
  index: number;
  name: string;
  vram_total: number;
  vram_free: number;
  visible: boolean;
}

export interface RayLightRuntimeStatus {
  active: boolean;
  recovery_required: boolean;
  epoch: number;
  runtime_gpu_indexes: number[];
  available_gpu_indexes: number[];
  invalid_gpu_indexes: number[];
  tainted: boolean;
  recovery_token: string | null;
}

export interface ModelInventory {
  fl2va: string[];
  ref2va: string[];
  clip: string[];
  video_vae: string[];
  audio_vae: string[];
  loras: string[];
}

export type TaskStatus = "queued" | "preparing" | "running" | "succeeded" | "failed" | "cancelling" | "cancelled";

export interface GenerationTaskChild {
  id: string;
  family: DiffusionModelRole;
  backend: ResolvedExecutionBackend;
  segment_ids: string[];
  status: TaskStatus;
  progress: number;
  stage: string | null;
  prompt_id: string | null;
  outputs: string[];
  error: string | null;
}

export interface GenerationSegmentResult {
  segment_id: string;
  child_id: string;
  output_url: string;
  /** ComfyUI-owned, server-redacted relative location label. */
  output_file: string;
  /** Exact job timeline/settings snapshots still match server authorities; historical outputs remain usable. */
  current_snapshot: boolean;
}

export interface GenerationTask {
  id: string;
  mode: GenerationMode | "timeline";
  status: TaskStatus;
  /** Server-derived display metadata; optional only for old local test data. */
  display_name?: string;
  project_title?: string | null;
  /** Stable owning project; null for legacy six-mode tasks and pre-project tasks. */
  project_id?: string | null;
  current_project?: boolean;
  /** Normalized progress reported by the backend, from 0 to 1. */
  progress: number;
  stage: string | null;
  prompt_id: string | null;
  error: string | null;
  preview_url: string | null;
  outputs: string[];
  /** ComfyUI-owned relative output paths. Download URLs remain in `outputs`. */
  output_files?: string[];
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  execution_duration_seconds?: number | null;
  output_count?: number;
  error_summary?: string | null;
  children: GenerationTaskChild[];
  /** Latest unambiguous generated candidate for each stable segment ID. */
  segment_results: GenerationSegmentResult[];
  /** Ephemeral authenticated sampler image; null when no current frame exists. */
  live_preview_url: string | null;
}

export interface CreateTaskRequest {
  mode: GenerationMode;
  config?: ModeDraft;
}

export interface TimelineTaskRequest {
  config?: TimelineProject;
  /** Stable segment identities. The workspace always sends its checkbox set. */
  segment_ids?: string[];
}

export interface TimelineCompileReport {
  execution_strategy: "native_segment_graph_v1";
  model_families: DiffusionModelRole[];
  plans: Array<{
    segment_id: string;
    mode: TimelineGenerationMode;
    recipe: DerivedGenerationRecipe;
    model_family: DiffusionModelRole;
    backend: ResolvedExecutionBackend;
    /** Backward-compatible alias of the saved, visible take length. */
    frame_count: number;
    /** Frames retained in the saved segment and therefore in final duration. */
    visible_frame_count: number;
    /** Internal H3 sample length before guide-prefix and alignment-tail trimming. */
    sample_frame_count: number;
    /** Tail frames read from the direct predecessor as AddGuide context. */
    continuity_context_frames: 0 | 5 | 22 | 39 | 56;
    /** H3 alignment padding trimmed together with the guide prefix. */
    alignment_tail_frame_count: number;
    predecessor_segment_id: string | null;
    /** Whether the predecessor is produced in this run or resolved from a persisted take. */
    continuity_source: null | "same_run" | "historical_take";
    /** Server-selected immutable take identity; present only for historical_take. */
    historical_take_id: string | null;
    /** True when this segment deliberately starts without predecessor context. */
    anchor_reset: boolean;
    seed_mode: "fixed" | "random";
    /** Exact browser-safe value carried by this server-owned workflow. */
    seed: number;
    conditioning_node: "MiniMaxH3ImageToVideo" | "MiniMaxH3ReferenceToVideo";
    node_classes: string[];
  }>;
  node_policy: {
    graph_source: "server";
    accepts_client_workflow: false;
    allowed_nodes: string[];
    custom_nodes: string[];
    provenance: Record<
      string,
      "comfy-core" | "comfy-core-official-minimax-h3" | "comfy-extras" | "raylight" | "lora-custom"
    >;
  };
}

export interface AssetListResponse {
  assets: import("../domain/modes").AssetReference[];
  outputs_preserved: true;
  /** Missing only when talking to a pre-scope-metadata Director backend. */
  active_database_identity?: string;
  /** Canonical active ComfyUI origin; paired with active_database_identity. */
  comfy_origin?: string;
}

export type AssetUploadStage =
  | "queued"
  | "uploading"
  | "processing"
  | "analyzing"
  | "forwarding"
  | "complete"
  | "failed";

export interface AssetUploadProgress {
  stage: AssetUploadStage;
  percent?: number;
  strategy?: "passthrough" | "remux" | "transcode";
  input_bytes?: number;
  output_bytes?: number;
  elapsed_seconds?: number;
}

export interface AssetDeleteResponse {
  deleted_asset_id: string;
  outputs_preserved: true;
}

export interface AssetCascadeDeleteResponse extends AssetDeleteResponse {
  unbound_usages: string[];
}

export type AssetTrashRestoreMode = "registration_only" | "with_references";

export interface AssetTrashBatch {
  batch_id: string;
  comfy_origin: string;
  asset_ids: string[];
  assets: AssetReference[];
  cascade: boolean;
  unbound_usages: string[];
  unbound_usages_by_asset: Record<string, string[]>;
  created_at: string;
  remote_files_preserved: true;
}

export interface AssetTrashListResponse {
  batches: AssetTrashBatch[];
  remote_files_preserved: true;
  /** Missing only when talking to a pre-scope-metadata Director backend. */
  active_database_identity?: string;
  /** Canonical active ComfyUI origin; paired with active_database_identity. */
  comfy_origin?: string;
}

export interface AssetTrashRestoreResponse {
  batch_id: string;
  restored_asset_ids: string[];
  restored_references: boolean;
  mode: AssetTrashRestoreMode;
  remote_files_preserved: true;
}

export interface AssetTrashPurgeResponse {
  batch_id: string;
  purged_asset_ids: string[];
  remote_files_preserved: true;
}

export type AssetTrashConflictOwnerKind =
  | "timeline"
  | "project"
  | "draft"
  | "asset"
  | "batch";

export interface AssetTrashConflictOwner {
  owner_kind: AssetTrashConflictOwnerKind;
  owner_id: string;
  reason: string;
  expected_revision?: number | null;
  actual_revision?: number | null;
  message?: string;
}

export interface TaskListResponse {
  jobs: GenerationTask[];
  total?: number;
  limit?: number;
  offset?: number;
  has_more?: boolean;
  summary?: TaskStatusSummary;
}

export interface TaskStatusSummary {
  total: number;
  active: number;
  queued: number;
  preparing: number;
  running: number;
  cancelling: number;
  succeeded: number;
  failed: number;
  cancelled: number;
}

export interface TaskBulkCancelResponse {
  jobs: GenerationTask[];
  requested_count: number;
  terminal_count: number;
}

export interface TaskProjectSnapshotResponse {
  job_id: string;
  project: TimelineProject;
  segment_ids: string[] | null;
}

export interface ProjectSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  segment_count: number;
}

export interface ProjectListResponse {
  projects: ProjectSummary[];
  active_database_identity: string;
}

export interface ProjectDeleteResponse {
  deleted_project_id: string;
  outputs_preserved: true;
  orphaned_jobs: number;
}

/** Server-owned timeline document paired with its exact CAS revision. */
export interface TimelineAuthority {
  document: TimelineProject;
  revision: number;
}

export interface TaskDiagnosticChild {
  id: string;
  family: DiffusionModelRole;
  backend: ResolvedExecutionBackend;
  segment_ids: string[];
  status: TaskStatus;
  progress: number;
  stage: string | null;
  output_files: string[];
  error_summary: string | null;
}

export interface TaskDiagnostic {
  schema_version: 1;
  id: string;
  display_name: string;
  project_title: string | null;
  mode: GenerationMode | "timeline";
  status: TaskStatus;
  progress: number;
  stage: string | null;
  created_at: string;
  updated_at: string;
  started_at: string | null;
  completed_at: string | null;
  execution_duration_seconds: number | null;
  output_files: string[];
  error_summary: string | null;
  children: TaskDiagnosticChild[];
  settings_included: false;
  workflow_included: false;
}

export interface TaskGenerationDetails {
  schema_version: 2;
  job_id: string;
  project_title: string;
  render: {
    width: number;
    height: number;
    fps: number;
    export_mode: "all" | "segments";
    total_duration_seconds: number;
  };
  sampling: Array<{
    family: DiffusionModelRole;
    steps: number;
    seed: number;
    random_seed: boolean;
    sampler: "res_multistep" | "euler" | "dpmpp_2m";
    scheduler: "simple" | "normal" | "karras" | "beta";
    shift: number;
    audio_shift: number;
  }>;
  models: Array<{
    family: DiffusionModelRole;
    filename: string;
    device: string;
    lora_name: string | null;
    lora_strength: number;
    backends: ResolvedExecutionBackend[];
    /** ComfyUI/RayLight logical indexes, not physical PCI device numbers. */
    logical_gpu_indices: number[];
    ulysses_degree: number | null;
    ring_degree: number | null;
  }>;
  shared_models: Array<{
    role: "clip" | "video_vae" | "audio_vae";
    filename: string;
    device: string;
  }>;
  runtime_snapshot_available: boolean;
  segments: Array<{
    id: string;
    title: string;
    family: DiffusionModelRole;
    recipe: DerivedGenerationRecipe;
    duration_seconds: number;
    prompt: string;
    continuity_enabled: boolean;
    continuity_overlap_frames: 5 | 22 | 39 | 56;
    ref_image_size: "match" | "max";
    audio_mode: "generate" | "source" | "mute";
    has_first_image: boolean;
    has_last_image: boolean;
    has_source_video: boolean;
    source_audio_as_reference: boolean;
    reference_image_count: number;
    reference_audio_count: number;
    reference_video_count: number;
  }>;
}

export interface JobDeleteResponse {
  deleted_job_id: string;
  outputs_preserved: true;
}

export interface JobClearResponse {
  deleted_count: number;
  active_count: number;
  outputs_preserved: true;
}

export interface ConnectionTestResult {
  ok: boolean;
  latency_ms?: number;
  message: string;
}

export interface RV2VShotDetectionRequest {
  asset_id: string;
  frame_rate: number;
  sensitivity: "low" | "medium" | "high";
  min_shot_frames: number;
}

export interface RV2VShotDetectionResponse {
  /** Full-source frame positions, including frame zero and the source end. */
  cut_frames: number[];
  shot_count: number;
  warnings: string[];
}

export const DEFAULT_SETTINGS: RuntimeSettings = {
  comfy_url: "",
  client_id: "director-web",
  memory_policy: "keep_resident",
  raylight_residency_policy: "keep_until_switch",
  models: {
    fl2va: {
      filename: "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
      device: "default",
      lora_name: null,
      lora_strength: 1,
      lora_loader: "auto",
      standard_lora_loader_override: null,
      lora_low_vram: false,
      backend: "auto",
      raylight: {
        gpu_select: [0],
        ulysses_degree: 1,
        ring_degree: 1,
        cfg_degree: 1,
        dp_degree: 1,
        fsdp: false,
        cpu_offload: false,
      },
    },
    ref2va: {
      filename: "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
      device: "default",
      lora_name: null,
      lora_strength: 1,
      lora_loader: "auto",
      standard_lora_loader_override: null,
      lora_low_vram: false,
      backend: "auto",
      raylight: {
        gpu_select: [0],
        ulysses_degree: 1,
        ring_degree: 1,
        cfg_degree: 1,
        dp_degree: 1,
        fsdp: false,
        cpu_offload: false,
      },
    },
    clip: { filename: "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", device: "default" },
    video_vae: { filename: "minimax_h3_video_vae_fp16.safetensors", device: "default" },
    audio_vae: { filename: "minimax_h3_audio_vae_fp32.safetensors", device: "default" },
  },
};

export function isConfiguredComfyUrl(value: string): boolean {
  if (!value.trim()) return false;
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeDevice(value: unknown, allowCpu: boolean, fallback: DeviceTarget): DeviceTarget {
  if (value === "default") return value;
  if (allowCpu && value === "cpu") return value;
  if (typeof value === "string" && /^gpu:(0|[1-9][0-9]*)$/.test(value)) {
    return value as DeviceTarget;
  }
  return fallback;
}

function normalizeRayLightProfile(
  value: unknown,
  fallback: RayLightProfile,
): RayLightProfile {
  if (!isRecord(value)) return structuredClone(fallback);
  const selected = Array.isArray(value.gpu_select)
    ? value.gpu_select.filter(
        (index): index is number =>
          typeof index === "number" &&
          Number.isInteger(index) &&
          index >= 0 &&
          index <= 255,
      )
    : [];
  const gpuSelect = [...new Set(selected)].slice(0, 8);
  if (!gpuSelect.length) gpuSelect.push(...fallback.gpu_select);
  const validDegree = (candidate: unknown): candidate is number =>
    typeof candidate === "number" &&
    Number.isInteger(candidate) &&
    candidate >= 1 &&
    candidate <= 8;
  let ulyssesDegree = validDegree(value.ulysses_degree)
    ? value.ulysses_degree
    : fallback.ulysses_degree;
  let ringDegree = validDegree(value.ring_degree)
    ? value.ring_degree
    : fallback.ring_degree;
  if (ulyssesDegree * ringDegree !== gpuSelect.length) {
    // A stale/partial topology must never be sent to strict backend settings.
    // U=world-size/R=1 is the deterministic migration preset.
    ulyssesDegree = gpuSelect.length;
    ringDegree = 1;
  }
  return {
    gpu_select: gpuSelect,
    ulysses_degree: ulyssesDegree,
    ring_degree: ringDegree,
    cfg_degree: 1,
    dp_degree: 1,
    // Native timeline v1 migrates these fail-closed until the RayLight FSDP
    // CUDA cleanup path has passed real multi-GPU execution validation.
    fsdp: false,
    cpu_offload: false,
  };
}

/** Removes stale/unknown local-storage settings before they can reach PUT /api/settings. */
export function sanitizeRuntimeSettings(value: unknown): RuntimeSettings {
  const fallback = structuredClone(DEFAULT_SETTINGS);
  if (!isRecord(value)) return fallback;

  const comfyUrl =
    typeof value.comfy_url === "string" && isConfiguredComfyUrl(value.comfy_url)
      ? value.comfy_url
      : fallback.comfy_url;
  const clientId =
    typeof value.client_id === "string" &&
    /^[A-Za-z0-9._:-]{1,128}$/.test(value.client_id)
      ? value.client_id
      : fallback.client_id;
  // Native per-segment child jobs reuse stable loader instances where the
  // execution backend permits it; there is no portable per-segment unload.
  const memoryPolicy: MemoryPolicy = "keep_resident";
  const raylightResidencyPolicy: RayLightResidencyPolicy =
    [
      "release_after_sampling",
      "keep_until_switch",
    ].includes(String(value.raylight_residency_policy))
      ? value.raylight_residency_policy as RayLightResidencyPolicy
      : "keep_until_switch";
  const models = isRecord(value.models) ? value.models : {};

  const normalizeModel = <K extends ModelRole>(role: K, allowCpu: boolean) => {
    const candidate = isRecord(models[role]) ? models[role] : {};
    const known = fallback.models[role];
    const binding = {
      filename:
        typeof candidate.filename === "string" &&
        candidate.filename.length >= 1 &&
        candidate.filename.length <= 1024
          ? candidate.filename
          : known.filename,
      device: normalizeDevice(candidate.device, allowCpu, known.device),
    };
    if (role !== "fl2va" && role !== "ref2va") return binding;
    const knownDiffusion = known as DiffusionModelBinding;
    const loraName =
      candidate.lora_name === null || candidate.lora_name === ""
        ? null
        : typeof candidate.lora_name === "string" &&
            candidate.lora_name.length <= 1024
          ? candidate.lora_name
          : knownDiffusion.lora_name;
    const overrideCandidate = isRecord(candidate.standard_lora_loader_override)
      ? candidate.standard_lora_loader_override
      : null;
    const overrideLoader = overrideCandidate?.loader;
    const standardLoraLoaderOverride: StandardLoraLoaderOverride | null =
      loraName !== null &&
      ["dedicated", "bypass_model_only", "model_only"].includes(String(overrideLoader)) &&
      overrideCandidate?.lora_name === loraName &&
      overrideCandidate?.model_filename === binding.filename &&
      typeof overrideCandidate?.comfy_origin === "string" &&
      overrideCandidate.comfy_origin.replace(/\/+$/, "") === comfyUrl.replace(/\/+$/, "")
        ? {
            loader: overrideLoader as StandardLoraLoader,
            lora_name: loraName,
            model_filename: binding.filename,
            comfy_origin: comfyUrl,
          }
        : null;
    const normalized: DiffusionModelBinding = {
      ...binding,
      lora_name: loraName,
      lora_strength:
        typeof candidate.lora_strength === "number" &&
        Number.isFinite(candidate.lora_strength) &&
        candidate.lora_strength >= -10 &&
        candidate.lora_strength <= 10
          ? candidate.lora_strength
          : knownDiffusion.lora_strength,
      lora_loader: "auto",
      standard_lora_loader_override: standardLoraLoaderOverride,
      lora_low_vram: false,
      backend: "auto",
      raylight: normalizeRayLightProfile(candidate.raylight, knownDiffusion.raylight),
    };
    // RayLight owns placement through its logical GPU pool. Carrying a stale
    // standard gpu:N/cpu target into a RayLight compile is rejected by the
    // server and, more importantly, makes the UI imply two authorities.
    if (resolveExecutionBackend(normalized) === "raylight") {
      normalized.device = "default";
      normalized.standard_lora_loader_override = null;
    }
    return normalized;
  };

  return {
    comfy_url: comfyUrl,
    client_id: clientId,
    memory_policy: memoryPolicy,
    raylight_residency_policy: raylightResidencyPolicy,
    models: {
      fl2va: normalizeModel("fl2va", true) as DiffusionModelBinding,
      ref2va: normalizeModel("ref2va", true) as DiffusionModelBinding,
      clip: normalizeModel("clip", true),
      video_vae: normalizeModel("video_vae", false) as ModelBinding<VaeDeviceTarget>,
      audio_vae: normalizeModel("audio_vae", false) as ModelBinding<VaeDeviceTarget>,
    },
  };
}

export const EMPTY_CAPABILITIES: CapabilityReport = {
  connection: "unknown",
  supported_modes: [],
  supports_cancel: false,
  available_nodes: [],
  missing_nodes: [],
};

export const EMPTY_MODELS: ModelInventory = {
  fl2va: [],
  ref2va: [],
  clip: [],
  video_vae: [],
  audio_vae: [],
  loras: [],
};

export const EMPTY_RAYLIGHT_RUNTIME_STATUS: RayLightRuntimeStatus = {
  active: false,
  recovery_required: false,
  epoch: 0,
  runtime_gpu_indexes: [],
  available_gpu_indexes: [],
  invalid_gpu_indexes: [],
  tainted: false,
  recovery_token: null,
};
