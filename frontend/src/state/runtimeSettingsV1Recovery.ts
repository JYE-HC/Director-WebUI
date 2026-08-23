import {
  parseLegacyRuntimeSettingsV1,
  sanitizeRuntimeSettings,
  type LegacyRuntimeSettingsV1,
  type LegacyStandardLoraOverrideEvidence,
  type RuntimeSettings,
} from "../api/types";
import {
  normalizeTimelineProject,
  type FeatureSelection,
  type TimelineProject,
} from "../domain/timelineProject";
import { isStoragePath } from "../domain/storagePath";
import {
  mergeExactLoraLoaderMappings,
  runtimeSettingsQuarantineKeyMatchesRaw,
  runtimeSettingsRecoveryRawDigest,
} from "./runtimeSettingsRecoveryEvidence";

/** Last key written by the RuntimeSettingsV1 browser. It is evidence only. */
export const LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY =
  "directordeck:v2:runtime-settings-pending";
export const LEGACY_UNBOUND_RUNTIME_SETTINGS_WAL_STORAGE_KEY =
  "directordeck:runtime-settings-pending";
export const LEGACY_RUNTIME_SETTINGS_WAL_QUARANTINE_PREFIX =
  "directordeck:v3:runtime-settings-v1-quarantine:";

const LEGACY_RUNTIME_SETTINGS_WAL_FORMAT = "director-pending-runtime-settings";

export interface LegacyRuntimeSettingsWalEnvelope {
  format: typeof LEGACY_RUNTIME_SETTINGS_WAL_FORMAT;
  version: 1 | 2;
  owner_id: string | null;
  pending: true;
  active_database_path: string;
  written_at_ms: number;
  settings: LegacyRuntimeSettingsV1;
}

export interface LegacyRuntimeSettingsWalCandidate {
  storage_key: string;
  raw: string;
  raw_digest: string;
  envelope: LegacyRuntimeSettingsWalEnvelope;
}

export type LegacyRuntimeSettingsRecoveryChoice =
  | { kind: "apply-all-projects"; project_ids: string[] }
  | { kind: "apply-specific-projects"; project_ids: string[] }
  | { kind: "retain-lora-compat" }
  | { kind: "discard" };

export interface LegacyRuntimeSettingsRecoveryPlan {
  choice: LegacyRuntimeSettingsRecoveryChoice["kind"];
  project_updates: Array<{ project_id: string; document: TimelineProject }>;
  runtime_settings: RuntimeSettings | null;
}

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

function validOwner(value: unknown): value is string {
  return typeof value === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(value);
}

export function parseLegacyRuntimeSettingsWalRaw(
  raw: string,
): LegacyRuntimeSettingsWalEnvelope | null {
  try {
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value)) return null;
    const isV1 = value.version === 1 && hasExactKeys(value, [
      "format", "version", "pending", "active_database_path",
      "written_at_ms", "settings",
    ]);
    const isV2 = value.version === 2 && hasExactKeys(value, [
      "format", "version", "owner_id", "pending", "active_database_path",
      "written_at_ms", "settings",
    ]) && validOwner(value.owner_id);
    if (
      (!isV1 && !isV2) ||
      value.format !== LEGACY_RUNTIME_SETTINGS_WAL_FORMAT ||
      value.pending !== true ||
      typeof value.active_database_path !== "string" ||
      !isStoragePath(value.active_database_path) ||
      !Number.isSafeInteger(value.written_at_ms) ||
      (value.written_at_ms as number) <= 0
    ) return null;
    const settings = parseLegacyRuntimeSettingsV1(value.settings);
    if (!settings) return null;
    return {
      format: LEGACY_RUNTIME_SETTINGS_WAL_FORMAT,
      version: value.version as 1 | 2,
      owner_id: isV2 ? value.owner_id as string : null,
      pending: true,
      active_database_path: value.active_database_path,
      written_at_ms: value.written_at_ms as number,
      settings,
    };
  } catch {
    return null;
  }
}

function quarantineKeyForRaw(raw: string): string {
  return `${LEGACY_RUNTIME_SETTINGS_WAL_QUARANTINE_PREFIX}${runtimeSettingsRecoveryRawDigest(raw)}`;
}

/**
 * Copies old bytes before removing their singleton source key. No candidate is
 * interpreted as the current project: callers must later choose an explicit
 * project id set and obtain CAS authorities for every member.
 */
export function quarantineLegacyRuntimeSettingsWals(): void {
  try {
    for (const sourceKey of [
      LEGACY_RUNTIME_SETTINGS_WAL_STORAGE_KEY,
      LEGACY_UNBOUND_RUNTIME_SETTINGS_WAL_STORAGE_KEY,
    ]) {
      const raw = localStorage.getItem(sourceKey);
      if (raw === null) continue;
      const destination = quarantineKeyForRaw(raw);
      const existing = localStorage.getItem(destination);
      if (existing !== null && existing !== raw) continue;
      if (existing === null) localStorage.setItem(destination, raw);
      if (
        localStorage.getItem(destination) === raw &&
        localStorage.getItem(sourceKey) === raw
      ) localStorage.removeItem(sourceKey);
    }
  } catch {
    // The singleton remains untouched when durable quarantine cannot be proven.
  }
}

export function listLegacyRuntimeSettingsWalCandidates(
  activeDatabasePath: string,
): LegacyRuntimeSettingsWalCandidate[] {
  quarantineLegacyRuntimeSettingsWals();
  if (!isStoragePath(activeDatabasePath)) return [];
  const candidates: LegacyRuntimeSettingsWalCandidate[] = [];
  try {
    for (let index = 0; index < localStorage.length; index += 1) {
      const storageKey = localStorage.key(index);
      if (!storageKey?.startsWith(LEGACY_RUNTIME_SETTINGS_WAL_QUARANTINE_PREFIX)) continue;
      const raw = localStorage.getItem(storageKey);
      if (raw === null || !runtimeSettingsQuarantineKeyMatchesRaw(
        storageKey,
        LEGACY_RUNTIME_SETTINGS_WAL_QUARANTINE_PREFIX,
        raw,
      )) continue;
      const envelope = parseLegacyRuntimeSettingsWalRaw(raw);
      if (!envelope || envelope.active_database_path !== activeDatabasePath) continue;
      candidates.push({
        storage_key: storageKey,
        raw,
        raw_digest: runtimeSettingsRecoveryRawDigest(raw),
        envelope,
      });
    }
  } catch {
    return [];
  }
  return candidates.sort((left, right) =>
    right.envelope.written_at_ms - left.envelope.written_at_ms ||
    left.storage_key.localeCompare(right.storage_key));
}

export function clearLegacyRuntimeSettingsWalCandidate(
  candidate: Pick<LegacyRuntimeSettingsWalCandidate, "storage_key" | "raw">,
): boolean {
  try {
    if (localStorage.getItem(candidate.storage_key) !== candidate.raw) return false;
    localStorage.removeItem(candidate.storage_key);
    return localStorage.getItem(candidate.storage_key) === null;
  } catch {
    return false;
  }
}

function legacyLoraSelection(settings: LegacyRuntimeSettingsV1): FeatureSelection {
  const byFamily = Object.fromEntries(
    (["fl2va", "ref2va"] as const).map((family) => {
      const binding = settings.models[family];
      return [family, {
        enabled: binding.lora_name !== null,
        filename: binding.lora_name,
        strength: binding.lora_strength,
      }];
    }),
  );
  return {
    enabled: byFamily.fl2va.enabled || byFamily.ref2va.enabled,
    params: { by_family: byFamily },
  };
}

export function applyLegacyCreativeSettingsToProject(
  project: TimelineProject,
  settings: LegacyRuntimeSettingsV1,
): TimelineProject | null {
  const next: TimelineProject = {
    ...structuredClone(project),
    model_stack: {
      fl2va: { filename: settings.models.fl2va.filename },
      ref2va: { filename: settings.models.ref2va.filename },
      clip: { filename: settings.models.clip.filename },
      video_vae: { filename: settings.models.video_vae.filename },
      audio_vae: { filename: settings.models.audio_vae.filename },
    },
    features: {
      ...structuredClone(project.features),
      project: {
        ...structuredClone(project.features.project),
        lora: legacyLoraSelection(settings),
      },
    },
  };
  return normalizeTimelineProject(next);
}

function legacyOverrideEvidence(
  settings: LegacyRuntimeSettingsV1,
): LegacyStandardLoraOverrideEvidence[] {
  const overrides: LegacyStandardLoraOverrideEvidence[] = [];
  for (const family of ["fl2va", "ref2va"] as const) {
    const override = settings.models[family].standard_lora_loader_override;
    if (!override) continue;
    overrides.push({
      family,
      model_filename: override.model_filename,
      lora_filename: override.lora_name,
      loader: override.loader,
    });
  }
  return overrides;
}

export function migrateLegacyRuntimeSettingsToV3(
  settings: LegacyRuntimeSettingsV1,
): RuntimeSettings {
  return sanitizeRuntimeSettings({
    schema_version: 3,
    client_id: settings.client_id,
    memory_policy: settings.memory_policy,
    raylight_residency_policy: settings.raylight_residency_policy,
    multi_gpu_enabled: settings.multi_gpu_enabled,
    placement: {
      fl2va: {
        device: settings.models.fl2va.device,
        raylight: settings.models.fl2va.raylight,
      },
      ref2va: {
        device: settings.models.ref2va.device,
        raylight: settings.models.ref2va.raylight,
      },
      clip_device: settings.models.clip.device,
      video_vae_device: settings.models.video_vae.device,
      audio_vae_device: settings.models.audio_vae.device,
    },
    lora_loader_overrides: legacyOverrideEvidence(settings).map((record) => ({
      lora_filename: record.lora_filename,
      adapter_id: record.loader === "dedicated" ? "minimax_h3_turbo" : "model_only",
      options: record.loader === "dedicated" ? { low_vram: false } : {},
    })),
  });
}

function retainLegacyCompat(
  current: RuntimeSettings,
  legacy: LegacyRuntimeSettingsV1,
  applyLegacyRuntimeFields: boolean,
): RuntimeSettings | null {
  const migrated = migrateLegacyRuntimeSettingsToV3(legacy);
  const mappingUnion = mergeExactLoraLoaderMappings(
    current.lora_loader_overrides,
    migrated.lora_loader_overrides,
  );
  if (!mappingUnion) return null;
  return {
    ...structuredClone(applyLegacyRuntimeFields ? migrated : current),
    schema_version: 3,
    lora_loader_overrides: mappingUnion,
  };
}

/** Pure explicit-scope planner used by the four recovery UI choices. */
export function buildLegacyRuntimeSettingsRecoveryPlan(
  candidate: LegacyRuntimeSettingsWalCandidate,
  choice: LegacyRuntimeSettingsRecoveryChoice,
  projects: ReadonlyArray<{ project_id: string; document: TimelineProject }>,
  currentRuntimeSettings: RuntimeSettings,
): LegacyRuntimeSettingsRecoveryPlan | null {
  const parsed = parseLegacyRuntimeSettingsWalRaw(candidate.raw);
  if (
    !parsed ||
    candidate.raw_digest !== runtimeSettingsRecoveryRawDigest(candidate.raw) ||
    !runtimeSettingsQuarantineKeyMatchesRaw(
      candidate.storage_key,
      LEGACY_RUNTIME_SETTINGS_WAL_QUARANTINE_PREFIX,
      candidate.raw,
    ) ||
    parsed.active_database_path !== candidate.envelope.active_database_path
  ) return null;
  const available = new Map(projects.map((project) => [project.project_id, project.document]));
  const requestedIds = "project_ids" in choice ? choice.project_ids : [];
  if (
    requestedIds.some((id) => typeof id !== "string" || !id || !available.has(id)) ||
    new Set(requestedIds).size !== requestedIds.length ||
    ((choice.kind === "apply-all-projects" || choice.kind === "apply-specific-projects") &&
      requestedIds.length === 0)
  ) return null;
  if (
    choice.kind === "apply-all-projects" &&
    (requestedIds.length !== available.size || requestedIds.some((id) => !available.has(id)))
  ) return null;

  const projectUpdates: LegacyRuntimeSettingsRecoveryPlan["project_updates"] = [];
  if (choice.kind === "apply-all-projects" || choice.kind === "apply-specific-projects") {
    for (const projectId of requestedIds) {
      const document = available.get(projectId)!;
      const migrated = applyLegacyCreativeSettingsToProject(
        document,
        parsed.settings,
      );
      if (!migrated) return null;
      projectUpdates.push({ project_id: projectId, document: migrated });
    }
  }
  const runtimeSettings = choice.kind === "discard"
    ? null
    : retainLegacyCompat(
      currentRuntimeSettings,
      parsed.settings,
      choice.kind !== "retain-lora-compat",
    );
  if (choice.kind !== "discard" && !runtimeSettings) return null;
  return {
    choice: choice.kind,
    project_updates: projectUpdates,
    runtime_settings: runtimeSettings,
  };
}
