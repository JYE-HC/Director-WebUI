import {
  normalizeLoraLoaderOverrides,
  sameLoraLoaderBinding,
  type LoraLoaderOverrideRecord,
} from "../api/types";

export const MAX_LORA_LOADER_OVERRIDES = 256;

export function runtimeSettingsRecoveryRawDigest(raw: string): string {
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(raw)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return `fnv1a64-${hash.toString(16).padStart(16, "0")}`;
}

export function runtimeSettingsQuarantineKeyMatchesRaw(
  storageKey: string,
  prefix: string,
  raw: string,
  allowCollisionSuffix = false,
): boolean {
  const base = `${prefix}${runtimeSettingsRecoveryRawDigest(raw)}`;
  if (storageKey === base) return true;
  if (!allowCollisionSuffix || !storageKey.startsWith(`${base}:`)) return false;
  const suffix = storageKey.slice(base.length + 1);
  return /^[1-9][0-9]*$/.test(suffix);
}

/**
 * Lossless exact-binding union shared by every historical settings recovery
 * path. The tolerant UI normalizer is only accepted when it preserves every
 * input record; recovery must never truncate or silently pick an adapter.
 */
export function mergeExactLoraLoaderMappings(
  current: readonly LoraLoaderOverrideRecord[],
  recovered: readonly LoraLoaderOverrideRecord[],
): LoraLoaderOverrideRecord[] | null {
  const records = current.map((record) => ({
    ...record,
    options: { ...record.options },
  }));
  for (const record of recovered) {
    const existing = records.find((candidate) =>
      sameLoraLoaderBinding(candidate, record));
    if (existing && (
      existing.adapter_id !== record.adapter_id ||
      JSON.stringify(existing.options) !== JSON.stringify(record.options)
    )) return null;
    if (!existing) records.push({ ...record, options: { ...record.options } });
  }
  if (records.length > MAX_LORA_LOADER_OVERRIDES) return null;
  const normalized = normalizeLoraLoaderOverrides(records);
  return normalized.length === records.length ? normalized : null;
}
