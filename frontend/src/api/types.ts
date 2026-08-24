import type { AssetReference, GenerationMode, ModeDraft } from "../domain/modes";
import type {
  DerivedGenerationRecipe,
  FeatureConfiguration,
  FeatureSelection,
  ModelStack,
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
  | "model_only"
  | "minimax_h3_turbo";
export type StandardLoraLoader = Exclude<LoraLoader, "auto">;

export interface StandardLoraLoaderOverride {
  loader: StandardLoraLoader;
  lora_name: string;
  model_filename: string;
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
  binding: Pick<DiffusionModelBinding, "raylight"> | RuntimeDiffusionPlacement,
): ResolvedExecutionBackend {
  return binding.raylight.gpu_select.length >= 2 ? "raylight" : "standard";
}

export function describeLoraLoader(binding: DiffusionModelBinding): string {
  if (resolveExecutionBackend(binding) === "raylight") return "DirectorDeckRayLoraLoader";
  switch (binding.standard_lora_loader_override?.loader) {
    case "dedicated": return "显式：MiniMax-H3 Turbo LoRA";
    case "bypass_model_only":
    case "model_only": return "显式：LoRA加载器（仅模型）";
    default: return "默认：LoRA加载器（仅模型）";
  }
}

export interface SettingsModels {
  fl2va: DiffusionModelBinding;
  ref2va: DiffusionModelBinding;
  clip: ModelBinding;
  video_vae: ModelBinding<VaeDeviceTarget>;
  audio_vae: ModelBinding<VaeDeviceTarget>;
}

/** Frozen pre-v5 payload used only by historical snapshots and WAL recovery. */
export interface LegacyRuntimeSettingsV1 {
  client_id: string;
  memory_policy: MemoryPolicy;
  raylight_residency_policy: RayLightResidencyPolicy;
  multi_gpu_enabled: boolean;
  models: SettingsModels;
}

export interface RuntimeDiffusionPlacement {
  device: DeviceTarget;
  raylight: RayLightProfile;
}

export interface RuntimePlacementV2 {
  fl2va: RuntimeDiffusionPlacement;
  ref2va: RuntimeDiffusionPlacement;
  clip_device: DeviceTarget;
  video_vae_device: VaeDeviceTarget;
  audio_vae_device: VaeDeviceTarget;
}

export interface LegacyStandardLoraOverrideEvidence {
  family: DiffusionModelRole;
  model_filename: string;
  lora_filename: string;
  loader: StandardLoraLoader;
}

export interface LegacyLoraResolutionCompat {
  schema_version: 1;
  auto_resolution_strategy_version: "v4-known-filename-or-safetensors-metadata-v1";
  explicit_overrides: LegacyStandardLoraOverrideEvidence[];
}

/** Frozen Stage-6 authority used only for pending-WAL migration. */
export interface RuntimeSettingsV2 {
  schema_version: 2;
  client_id: string;
  memory_policy: MemoryPolicy;
  raylight_residency_policy: RayLightResidencyPolicy;
  multi_gpu_enabled: boolean;
  placement: RuntimePlacementV2;
  legacy_lora_resolution_compat: LegacyLoraResolutionCompat;
}

export type LoraLoaderAdapterId = string;

export interface LoraLoaderBindingKey {
  lora_filename: string;
}

export interface LoraLoaderOverrideRecord extends LoraLoaderBindingKey {
  adapter_id: LoraLoaderAdapterId;
  options: Record<string, boolean>;
}

/** Exact current GET/PUT /api/settings authority document. */
export interface RuntimeSettings {
  schema_version: 3;
  client_id: string;
  memory_policy: MemoryPolicy;
  raylight_residency_policy: RayLightResidencyPolicy;
  multi_gpu_enabled: boolean;
  placement: RuntimePlacementV2;
  lora_loader_overrides: LoraLoaderOverrideRecord[];
}

export interface RuntimeSettingsAuthority {
  settings: RuntimeSettings;
  authority_token: string;
}

export interface RuntimeSettingsAuthorityWriteRequest {
  document: RuntimeSettings;
  expected_authority_token: string;
  schema_version: 3;
}

export interface RuntimeSettingsMigrationNotice {
  schema_version: 1;
  id: string;
  code: "legacy_lora_resolution_review_required";
  severity: "warning";
  action: "review_lora_loader_mappings";
  legacy_strategy_version: "v4-known-filename-or-safetensors-metadata-v1";
  message: string;
  created_at: string;
}

export interface RuntimeSettingsMigrationNoticeList {
  notices: RuntimeSettingsMigrationNotice[];
}

/** Exact GET /api/storage payload: the fixed database location of the host. */
export interface StorageConfiguration {
  active_database_path: string;
}

/**
 * A newly selected RayLight backend defaults to keyed residency. Merely
 * editing an already-RayLight model/LoRA must not override an intentional
 * "release after sampling" choice.
 */
export function rayLightResidencyPolicyAfterBindingChange(
  settings: Pick<LegacyRuntimeSettingsV1, "models" | "raylight_residency_policy">,
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

export interface ComfyKitchenAttentionCapability {
  context_revision: string;
  backend: ResolvedExecutionBackend | null;
  state: "available" | "unavailable" | "unknown";
  reasons: Array<{ code: string; message: string }>;
}

/** Bounded, server-redacted JSON carried by capability diagnostics and UI hints. */
export type CapabilityJsonValue =
  | null
  | boolean
  | number
  | string
  | CapabilityJsonValue[]
  | { [key: string]: CapabilityJsonValue };

export interface CapabilityReason {
  code: string;
  feature_id: string | null;
  segment_id: string | null;
  unit_id: string | null;
  backend: ResolvedExecutionBackend | null;
  rule: string;
  message: string;
  remediation: string;
  safe_details: { [key: string]: CapabilityJsonValue };
}

export type FeatureAvailabilityState = "available" | "unavailable" | "conditional";

export interface FeatureCatalogAdapterOption {
  adapter_id: LoraLoaderAdapterId;
  display_name: string;
  class_type: string;
  is_default: boolean;
  backend: "standard";
  supported_families: DiffusionModelRole[];
  configuration_options: Array<{
    id: string;
    type: "boolean";
    label: string;
    description: string;
    default: boolean;
  }>;
  adapter_fingerprint: string;
  capability: FeatureCapabilityEvaluation;
}

export interface FeatureCatalogEntry {
  id: string;
  version: number;
  title: string;
  description: string;
  mode: "switch" | "needed";
  layer: "graph";
  scopes: string[];
  params_schema: { [key: string]: CapabilityJsonValue };
  defaults: { [key: string]: CapabilityJsonValue };
  backends: ResolvedExecutionBackend[];
  availability: {
    state: FeatureAvailabilityState;
    reasons: CapabilityReason[];
  };
  adapter_options: FeatureCatalogAdapterOption[];
  ui: { [key: string]: CapabilityJsonValue };
}

export interface FeatureCatalog {
  template_bundle_version: number;
  host_capability_revision: string;
  entries: FeatureCatalogEntry[];
}

export interface DirectorDeckLoraLoaderDefinition {
  id: LoraLoaderAdapterId;
  display_name: string;
  class_type: string;
  input_contract: "model_only" | "dedicated_model";
  supported_families: DiffusionModelRole[];
  options: Array<{
    id: string;
    type: "boolean";
    label: string;
    description: string;
    default: boolean;
  }>;
}

/** Product-owned configuration; it does not claim ownership of host nodes. */
export interface DirectorDeckConfig {
  schema_version: 1;
  lora: {
    loaders: DirectorDeckLoraLoaderDefinition[];
    fallback_policy: {
      loader_ids: LoraLoaderAdapterId[];
      default_loader_id: LoraLoaderAdapterId;
    } | null;
    loader_policies: Array<{
      /** Regular expression searched against the complete relative LoRA path. */
      lora_filename: string;
      loader_ids: LoraLoaderAdapterId[];
      default_loader_id: LoraLoaderAdapterId;
    }>;
  };
  diagnostics: Array<{
    code: string;
    message: string;
  }>;
}

export type FeatureCatalogFetchResult =
  | { status: "fresh"; etag: string; catalog: FeatureCatalog }
  | { status: "not_modified"; etag: string };

export interface OperationalReadiness {
  endpoint_online: boolean;
  submission_allowed: boolean;
  ray_recovery_required: boolean;
  ray_tainted: boolean;
  invalid_runtime_gpu_indices: number[];
  blocking_reason_codes: string[];
}

export interface FeatureCapabilityEvaluation {
  available: boolean;
  reasons: CapabilityReason[];
  verified_contracts: string[];
  runtime_fingerprints: string[];
}

export interface ResolvedImplementationIdentity {
  role: string;
  class_type: string;
  implementation_id: string;
  semantic_version: string;
  runtime_fingerprint: string;
  binding_key: string;
}

export interface FeatureResolutionEvidence {
  state: "active" | "noop";
  implementations: ResolvedImplementationIdentity[];
  resolution_details: { [key: string]: CapabilityJsonValue };
}

export interface EffectiveFeatureResolution {
  id: string;
  version: number;
  state: "active" | "noop";
  adapter_fingerprint: string;
  capability: FeatureCapabilityEvaluation;
}

export interface EffectiveSegmentResolution {
  unit_id: string;
  backend: ResolvedExecutionBackend;
  family: DiffusionModelRole;
  template_id: "h3_standard_segment" | "h3_raylight_segment";
  features: EffectiveFeatureResolution[];
}

/** Stage 6 preflight consumes the same explicit v5 creative authority as compile/submit. */
export interface FeaturePreflightRequest {
  config: TimelineProject;
  segment_ids?: string[] | null;
  /** Stable database-owned scope used for historical continuity resolution. */
  project_id?: string;
}

export interface FeaturePreflightResponse {
  template_bundle_version: number;
  host_capability_revision: string;
  operational_readiness: OperationalReadiness;
  valid: boolean;
  errors: CapabilityReason[];
  effective_by_segment: Record<string, EffectiveSegmentResolution>;
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
  config: TimelineProject;
  /** Stable segment identities. The workspace always sends its checkbox set. */
  segment_ids?: string[];
}

export interface TimelineCompileReport {
  template_bundle_version: number;
  host_capability_revision: string;
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
  features: {
    requested: FeatureConfiguration;
    effective_by_segment: Record<string, EffectiveSegmentResolution>;
    resolutions: Array<{
      segment_id: string;
      unit_id: string;
      feature_id: string;
      version: number;
      backend: ResolvedExecutionBackend;
      family: DiffusionModelRole;
      template_id: EffectiveSegmentResolution["template_id"];
      resolution: FeatureResolutionEvidence;
      adapter_fingerprint: string;
      capability: FeatureCapabilityEvaluation;
    }>;
    /** Bundle-6 compiler evidence is preserved as bounded opaque JSON. */
    uses?: CapabilityJsonValue[];
    notices: Array<{
      segment_id: string;
      unit_id: string;
      feature_id: string;
      message: string;
    }>;
    /** Non-authorizing Bundle-6 host compatibility diagnostics. */
    advisories?: CapabilityReason[];
  };
  effective_execution_digest: {
    algorithm: "sha256-canonical-json-v1";
    value: string;
  };
}

export interface AssetListResponse {
  assets: import("../domain/modes").AssetReference[];
  outputs_preserved: true;
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
}

export interface ProjectDeleteResponse {
  deleted_project_id: string;
  outputs_preserved: true;
  orphaned_jobs: number;
}

export interface DocumentDigest {
  algorithm: "fnv1a32-json-stringify-v1" | "sha256-canonical-json-v1";
  value: string;
}

export interface ProjectImportLegacyCreativeContext {
  schema_version: 1;
  model_stack: ModelStack;
  lora: FeatureSelection;
  explicit_standard_lora_overrides: LegacyStandardLoraOverrideEvidence[];
}

export interface ProjectImportCreativeSelection {
  model_stack: ModelStack;
  lora: FeatureSelection;
}

interface ProjectImportPreflightRequestBase {
  title: string;
  document: unknown;
}

/** A v4 import may carry exactly one immutable creative authority. */
export type ProjectImportPreflightRequest = ProjectImportPreflightRequestBase & (
  | {
      legacy_runtime_settings: LegacyRuntimeSettingsV1;
      legacy_creative_context?: never;
      creative_selection?: never;
    }
  | {
      legacy_runtime_settings?: never;
      legacy_creative_context: ProjectImportLegacyCreativeContext;
      creative_selection?: never;
    }
  | {
      legacy_runtime_settings?: never;
      legacy_creative_context?: never;
      creative_selection: ProjectImportCreativeSelection;
    }
  | {
      legacy_runtime_settings?: never;
      legacy_creative_context?: never;
      creative_selection?: never;
    }
);

export interface ProjectImportPreflightResponse {
  schema_version: 1;
  status: "ready" | "needs_input";
  input_digest: DocumentDigest;
  proposed_document: TimelineProject | null;
  missing_context: string[];
  missing_model_bindings: string[];
  capability_issues: Array<Record<string, unknown>>;
  commit_token: string | null;
  expires_at: string | null;
}

export interface ProjectImportCommitRequest {
  commit_token: string;
  input_digest: DocumentDigest;
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

export const DEFAULT_LEGACY_SETTINGS: LegacyRuntimeSettingsV1 = {
  client_id: "directordeck",
  memory_policy: "keep_resident",
  raylight_residency_policy: "keep_until_switch",
  multi_gpu_enabled: false,
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: unknown, expected: readonly string[]): value is Record<string, unknown> {
  if (!isRecord(value)) return false;
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length &&
    actual.every((key, index) => key === wanted[index]);
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value as Record<string, unknown>).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`
  ).join(",")}}`;
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
export function sanitizeLegacyRuntimeSettings(value: unknown): LegacyRuntimeSettingsV1 {
  const fallback = structuredClone(DEFAULT_LEGACY_SETTINGS);
  if (!isRecord(value)) return fallback;

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
  // Boolean intent flag; anything but an explicit true means disabled.
  const multiGpuEnabled = value.multi_gpu_enabled === true;
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
      overrideCandidate?.model_filename === binding.filename
        ? {
            loader: overrideLoader as StandardLoraLoader,
            lora_name: loraName,
            model_filename: binding.filename,
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
    client_id: clientId,
    memory_policy: memoryPolicy,
    raylight_residency_policy: raylightResidencyPolicy,
    multi_gpu_enabled: multiGpuEnabled,
    models: {
      fl2va: normalizeModel("fl2va", true) as DiffusionModelBinding,
      ref2va: normalizeModel("ref2va", true) as DiffusionModelBinding,
      clip: normalizeModel("clip", true),
      video_vae: normalizeModel("video_vae", false) as ModelBinding<VaeDeviceTarget>,
      audio_vae: normalizeModel("audio_vae", false) as ModelBinding<VaeDeviceTarget>,
    },
  };
}

/**
 * Frozen RuntimeSettingsV1 boundary for historical snapshots and recovery
 * evidence. Unlike the sanitizer this parser rejects defaults, unknown keys,
 * and lossy normalization; callers may therefore prove that the exact bytes
 * they quarantine are the bytes they later offer for recovery.
 */
export function parseLegacyRuntimeSettingsV1(value: unknown): LegacyRuntimeSettingsV1 | null {
  if (!hasExactKeys(value, [
    "client_id",
    "memory_policy",
    "raylight_residency_policy",
    "multi_gpu_enabled",
    "models",
  ])) return null;
  const models = value.models;
  if (!hasExactKeys(models, ["fl2va", "ref2va", "clip", "video_vae", "audio_vae"])) return null;
  for (const role of ["clip", "video_vae", "audio_vae"] as const) {
    if (!hasExactKeys(models[role], ["filename", "device"])) return null;
  }
  const diffusionKeys = [
    "filename", "device", "lora_name", "lora_strength", "lora_loader",
    "standard_lora_loader_override", "lora_low_vram", "backend", "raylight",
  ];
  for (const role of ["fl2va", "ref2va"] as const) {
    const binding = models[role];
    if (!hasExactKeys(binding, diffusionKeys) || !hasExactKeys(binding.raylight, [
      "gpu_select", "ulysses_degree", "ring_degree", "cfg_degree", "dp_degree",
      "fsdp", "cpu_offload",
    ])) return null;
    if (
      binding.standard_lora_loader_override !== null &&
      !hasExactKeys(binding.standard_lora_loader_override, [
        "loader", "lora_name", "model_filename",
      ])
    ) return null;
  }
  const normalized = sanitizeLegacyRuntimeSettings(value);
  try {
    return canonicalJson(value) === canonicalJson(normalized) ? normalized : null;
  } catch {
    return null;
  }
}

const DEFAULT_RUNTIME_PLACEMENT: RuntimePlacementV2 = {
  fl2va: {
    device: "default",
    raylight: structuredClone(DEFAULT_LEGACY_SETTINGS.models.fl2va.raylight),
  },
  ref2va: {
    device: "default",
    raylight: structuredClone(DEFAULT_LEGACY_SETTINGS.models.ref2va.raylight),
  },
  clip_device: "default",
  video_vae_device: "default",
  audio_vae_device: "default",
};

export const DEFAULT_SETTINGS: RuntimeSettings = {
  schema_version: 3,
  client_id: "directordeck",
  memory_policy: "keep_resident",
  raylight_residency_policy: "keep_until_switch",
  multi_gpu_enabled: false,
  placement: structuredClone(DEFAULT_RUNTIME_PLACEMENT),
  lora_loader_overrides: [],
};

/** ECMAScript relational string order: lexicographic UTF-16 code units. */
export function compareUtf16Strings(left: string, right: string): number {
  if (left === right) return 0;
  return left < right ? -1 : 1;
}

export function compareLoraLoaderBinding(
  left: LoraLoaderBindingKey,
  right: LoraLoaderBindingKey,
): number {
  return compareUtf16Strings(left.lora_filename, right.lora_filename);
}

export function sameLoraLoaderBinding(
  left: LoraLoaderBindingKey,
  right: LoraLoaderBindingKey,
): boolean {
  return compareLoraLoaderBinding(left, right) === 0;
}

function isWellFormedUtf16(value: string): boolean {
  for (let index = 0; index < value.length; index += 1) {
    const unit = value.charCodeAt(index);
    if (unit >= 0xD800 && unit <= 0xDBFF) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xDC00 && next <= 0xDFFF)) return false;
      index += 1;
    } else if (unit >= 0xDC00 && unit <= 0xDFFF) {
      return false;
    }
  }
  return true;
}

function validLoraLoaderOverride(value: unknown): value is LoraLoaderOverrideRecord {
  return isRecord(value) &&
    hasExactKeys(value, ["lora_filename", "adapter_id", "options"]) &&
    typeof value.lora_filename === "string" && isWellFormedUtf16(value.lora_filename) &&
    [...value.lora_filename].length >= 1 && [...value.lora_filename].length <= 1_024 &&
    typeof value.adapter_id === "string" && /^[a-z][a-z0-9_]{0,63}$/.test(value.adapter_id) &&
    isRecord(value.options) && Object.keys(value.options).length <= 16 &&
    Object.entries(value.options).every(([key, option]) =>
      /^[a-z][a-z0-9_]{0,63}$/.test(key) && typeof option === "boolean");
}

function migrateLoraLoaderOverride(value: unknown): unknown {
  if (!isRecord(value)) return value;
  const keys = Object.keys(value).sort();
  if (keys.join("\u0000") === ["adapter_id", "lora_filename", "options"].join("\u0000")) {
    return value;
  }
  if (keys.join("\u0000") !== [
    "adapter_id", "family", "lora_filename", "model_filename",
  ].join("\u0000")) return value;
  const adapterId = value.adapter_id === "dedicated"
    ? "minimax_h3_turbo"
    : value.adapter_id === "bypass_model_only"
      ? "model_only"
      : value.adapter_id;
  return {
    lora_filename: value.lora_filename,
    adapter_id: adapterId,
    options: adapterId === "minimax_h3_turbo" ? { low_vram: false } : {},
  };
}

export function normalizeLoraLoaderOverrides(
  values: readonly unknown[],
): LoraLoaderOverrideRecord[] {
  const sorted = values.slice(0, 256)
    .map(migrateLoraLoaderOverride)
    .filter(validLoraLoaderOverride)
    .map((record) => ({ ...record, options: { ...record.options } }))
    .sort(compareLoraLoaderBinding);
  const collapsed = new Map<string, LoraLoaderOverrideRecord>();
  for (const record of sorted) {
    const previous = collapsed.get(record.lora_filename);
    if (!previous || (
      record.adapter_id === "minimax_h3_turbo" &&
      previous.adapter_id !== "minimax_h3_turbo"
    )) collapsed.set(record.lora_filename, record);
  }
  return [...collapsed.values()].sort(compareLoraLoaderBinding);
}

function sanitizeRuntimeSettingsBase(value: Record<string, unknown>): Omit<
  RuntimeSettings,
  "schema_version" | "lora_loader_overrides"
> {
  const fallback = DEFAULT_SETTINGS;
  const clientId =
    typeof value.client_id === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(value.client_id)
      ? value.client_id
      : fallback.client_id;
  const placement = isRecord(value.placement) ? value.placement : {};
  const normalizeDiffusionPlacement = (
    role: DiffusionModelRole,
  ): RuntimeDiffusionPlacement => {
    const candidate = isRecord(placement[role]) ? placement[role] : {};
    return {
      device: normalizeDevice(candidate.device, true, fallback.placement[role].device),
      raylight: normalizeRayLightProfile(candidate.raylight, fallback.placement[role].raylight),
    };
  };
  return {
    client_id: clientId,
    memory_policy: "keep_resident",
    raylight_residency_policy: ["release_after_sampling", "keep_until_switch"].includes(
      String(value.raylight_residency_policy),
    )
      ? value.raylight_residency_policy as RayLightResidencyPolicy
      : fallback.raylight_residency_policy,
    multi_gpu_enabled: value.multi_gpu_enabled === true,
    placement: {
      fl2va: normalizeDiffusionPlacement("fl2va"),
      ref2va: normalizeDiffusionPlacement("ref2va"),
      clip_device: normalizeDevice(placement.clip_device, true, fallback.placement.clip_device),
      video_vae_device: normalizeDevice(
        placement.video_vae_device,
        false,
        fallback.placement.video_vae_device,
      ) as VaeDeviceTarget,
      audio_vae_device: normalizeDevice(
        placement.audio_vae_device,
        false,
        fallback.placement.audio_vae_device,
      ) as VaeDeviceTarget,
    },
  };
}

function strictRuntimePlacement(value: Record<string, unknown>): boolean {
  if (!hasExactKeys(value.placement, [
    "fl2va", "ref2va", "clip_device", "video_vae_device", "audio_vae_device",
  ])) return false;
  for (const role of ["fl2va", "ref2va"] as const) {
    const placement = value.placement[role];
    if (!hasExactKeys(placement, ["device", "raylight"]) || !hasExactKeys(placement.raylight, [
      "gpu_select", "ulysses_degree", "ring_degree", "cfg_degree", "dp_degree",
      "fsdp", "cpu_offload",
    ])) return false;
  }
  return true;
}

/** Frozen parser for Stage-6 pending settings WALs. */
export function parseRuntimeSettingsV2(value: unknown): RuntimeSettingsV2 | null {
  if (!hasExactKeys(value, [
    "schema_version",
    "client_id",
    "memory_policy",
    "raylight_residency_policy",
    "multi_gpu_enabled",
    "placement",
    "legacy_lora_resolution_compat",
  ]) || value.schema_version !== 2) return null;
  if (!strictRuntimePlacement(value)) return null;
  const compat = value.legacy_lora_resolution_compat;
  if (!hasExactKeys(compat, [
    "schema_version", "auto_resolution_strategy_version", "explicit_overrides",
  ]) || !Array.isArray(compat.explicit_overrides)) return null;
  if (compat.explicit_overrides.some((entry) => !hasExactKeys(entry, [
    "family", "model_filename", "lora_filename", "loader",
  ]))) return null;
  const explicitOverrides = compat.explicit_overrides
    .filter((entry): entry is LegacyStandardLoraOverrideEvidence =>
      isRecord(entry) &&
      (entry.family === "fl2va" || entry.family === "ref2va") &&
      typeof entry.model_filename === "string" && entry.model_filename.length >= 1 &&
      entry.model_filename.length <= 1_024 &&
      typeof entry.lora_filename === "string" && entry.lora_filename.length >= 1 &&
      entry.lora_filename.length <= 1_024 &&
      (entry.loader === "dedicated" ||
        entry.loader === "bypass_model_only" ||
        entry.loader === "model_only"))
    .map((entry) => ({ ...entry }))
    .sort((left, right) => compareUtf16Strings(left.family, right.family))
    .filter((entry, index, records) => index === 0 || records[index - 1].family !== entry.family);
  const normalized: RuntimeSettingsV2 = {
    schema_version: 2,
    ...sanitizeRuntimeSettingsBase(value),
    legacy_lora_resolution_compat: {
      schema_version: 1,
      auto_resolution_strategy_version: "v4-known-filename-or-safetensors-metadata-v1",
      explicit_overrides: explicitOverrides,
    },
  };
  try {
    return canonicalJson(value) === canonicalJson(normalized) ? normalized : null;
  } catch {
    return null;
  }
}

export function migrateRuntimeSettingsV2ToV3(value: RuntimeSettingsV2): RuntimeSettings {
  const legacyRecords = value.legacy_lora_resolution_compat.explicit_overrides.map((record) => ({
    lora_filename: record.lora_filename,
    adapter_id: record.loader === "dedicated" ? "minimax_h3_turbo" : "model_only",
    options: record.loader === "dedicated" ? { low_vram: false } : {},
  })).sort(compareLoraLoaderBinding);
  const migratedRecords = normalizeLoraLoaderOverrides(legacyRecords);
  return {
    schema_version: 3,
    client_id: value.client_id,
    memory_policy: value.memory_policy,
    raylight_residency_policy: value.raylight_residency_policy,
    multi_gpu_enabled: value.multi_gpu_enabled,
    placement: structuredClone(value.placement),
    lora_loader_overrides: migratedRecords,
  };
}

/** Normalizes only RuntimeSettingsV3. Older documents require explicit migration. */
export function sanitizeRuntimeSettings(value: unknown): RuntimeSettings {
  const fallback = structuredClone(DEFAULT_SETTINGS);
  if (!isRecord(value) || value.schema_version !== 3) return fallback;
  return {
    schema_version: 3,
    ...sanitizeRuntimeSettingsBase(value),
    lora_loader_overrides: Array.isArray(value.lora_loader_overrides)
      ? normalizeLoraLoaderOverrides(value.lora_loader_overrides)
      : [],
  };
}

export function parseRuntimeSettingsV3(value: unknown): RuntimeSettings | null {
  if (!hasExactKeys(value, [
    "schema_version",
    "client_id",
    "memory_policy",
    "raylight_residency_policy",
    "multi_gpu_enabled",
    "placement",
    "lora_loader_overrides",
  ]) || value.schema_version !== 3 || !strictRuntimePlacement(value)) return null;
  if (
    !Array.isArray(value.lora_loader_overrides) ||
    value.lora_loader_overrides.length > 256 ||
    value.lora_loader_overrides.some((entry) => !validLoraLoaderOverride(entry))
  ) return null;
  const normalized = sanitizeRuntimeSettings(value);
  try {
    return canonicalJson(value) === canonicalJson(normalized) ? normalized : null;
  } catch {
    return null;
  }
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

/** Exact GET /api/raylight/setup payload (multi-GPU capability and install). */
export type RayLightInstallState = "idle" | "running" | "needs_restart" | "ready" | "failed";

export interface RayLightInstallSnapshot {
  state: RayLightInstallState;
  log_tail: string[];
  returncode: number | null;
  error: string | null;
  started_at: number | null;
}

export interface RayLightSetupStatus {
  enabled: boolean;
  platform_supported: boolean;
  dependencies_installed: boolean;
  requirements_available: boolean;
  install: RayLightInstallSnapshot;
}

/** Exact GET /api/media/setup payload (ffmpeg/ffprobe capability and install). */
export type MediaToolInstallPhase =
  | "installing_package"
  | "downloading_binaries"
  | "verifying";

export interface MediaToolInstallSnapshot extends RayLightInstallSnapshot {
  /** Optional while older DirectorDeck backends remain supported. */
  phase?: MediaToolInstallPhase | null;
  /** Real download progress; absent/null means the backend cannot determine it yet. */
  progress_percent?: number | null;
}

export interface MediaToolsStatus {
  ffmpeg_available: boolean;
  ffprobe_available: boolean;
  ffmpeg_path: string | null;
  encoders_ok: boolean;
  ready: boolean;
  install: MediaToolInstallSnapshot;
}
