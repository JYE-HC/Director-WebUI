import {
  normalizeLoraLoaderOverrides,
  sameLoraLoaderBinding,
  type LoraLoaderBindingKey,
  type LoraLoaderOverrideRecord,
  type RuntimeSettings,
  type RuntimeSettingsAuthority,
} from "../api/types";

export interface LoraLoaderOverrideEdit {
  base_settings: RuntimeSettings;
  original: LoraLoaderOverrideRecord | null;
  next: LoraLoaderOverrideRecord | null;
}

export interface RuntimeSettingsAuthorityTransport {
  read: () => Promise<RuntimeSettingsAuthority>;
  write: (
    settings: RuntimeSettings,
    expectedAuthorityToken: string,
  ) => Promise<RuntimeSettingsAuthority>;
}

export class LoraLoaderOverrideConflictError extends Error {
  constructor(message = "同一条 LoRA 加载器映射已被其他页面修改；请重新检查后再保存") {
    super(message);
    this.name = "LoraLoaderOverrideConflictError";
  }
}

function sameRecord(
  left: LoraLoaderOverrideRecord | null,
  right: LoraLoaderOverrideRecord | null,
): boolean {
  return left === null
    ? right === null
    : right !== null &&
      sameLoraLoaderBinding(left, right) &&
      left.adapter_id === right.adapter_id &&
      JSON.stringify(left.options) === JSON.stringify(right.options);
}

function findRecord(
  settings: RuntimeSettings,
  binding: LoraLoaderBindingKey,
): LoraLoaderOverrideRecord | null {
  return settings.lora_loader_overrides.find((record) =>
    sameLoraLoaderBinding(record, binding)) ?? null;
}

function affectedBindings(edit: LoraLoaderOverrideEdit): LoraLoaderBindingKey[] {
  const bindings = [edit.original, edit.next]
    .filter((record): record is LoraLoaderOverrideRecord => record !== null);
  return bindings.filter((binding, index) =>
    bindings.findIndex((candidate) => sameLoraLoaderBinding(candidate, binding)) === index);
}

function editBaseStillCurrent(
  settings: RuntimeSettings,
  edit: LoraLoaderOverrideEdit,
): boolean {
  return affectedBindings(edit).every((binding) =>
    sameRecord(
      findRecord(settings, binding),
      findRecord(edit.base_settings, binding),
    ));
}

function isValidNextRecord(record: LoraLoaderOverrideRecord): boolean {
  const normalized = normalizeLoraLoaderOverrides([record]);
  return normalized.length === 1 && JSON.stringify(normalized[0]) === JSON.stringify(record);
}

export function applyLoraLoaderOverrideEdit(
  settings: RuntimeSettings,
  edit: LoraLoaderOverrideEdit,
): RuntimeSettings {
  if (!edit.original && !edit.next) {
    throw new Error("LoRA 加载器映射修改为空");
  }
  if (edit.next && !isValidNextRecord(edit.next)) {
    throw new Error("LoRA 加载器映射字段无效");
  }
  let records = settings.lora_loader_overrides.filter((record) =>
    !edit.original || !sameLoraLoaderBinding(record, edit.original));
  const next = edit.next;
  if (next) {
    records = records.filter((record) => !sameLoraLoaderBinding(record, next));
    records.push({ ...next, options: { ...next.options } });
  }
  const normalized = normalizeLoraLoaderOverrides(records);
  if (normalized.length !== records.length) {
    throw new Error("LoRA 加载器映射超过 256 条或包含重复绑定");
  }
  return {
    ...structuredClone(settings),
    lora_loader_overrides: normalized,
  };
}

function editIsApplied(
  settings: RuntimeSettings,
  edit: LoraLoaderOverrideEdit,
): boolean {
  if (
    edit.original &&
    (!edit.next || !sameLoraLoaderBinding(edit.original, edit.next)) &&
    findRecord(settings, edit.original) !== null
  ) return false;
  return edit.next
    ? sameRecord(findRecord(settings, edit.next), edit.next)
    : edit.original !== null && findRecord(settings, edit.original) === null;
}

function isAuthorityConflict(reason: unknown): boolean {
  return typeof reason === "object" && reason !== null &&
    "status" in reason && (reason as { status?: unknown }).status === 409;
}

async function recoverAmbiguousWrite(
  edit: LoraLoaderOverrideEdit,
  reason: unknown,
  transport: RuntimeSettingsAuthorityTransport,
): Promise<RuntimeSettingsAuthority> {
  let latest: RuntimeSettingsAuthority;
  try {
    latest = await transport.read();
  } catch {
    throw reason;
  }
  if (editIsApplied(latest.settings, edit)) return latest;
  throw reason;
}

/**
 * Whole-document CAS with one narrowly-scoped rebase. A retry is permitted
 * only when every binding touched by this edit is unchanged from its visible
 * edit base; unrelated server records are always preserved from the fresh GET.
 */
export async function saveLoraLoaderOverrideWithCas(
  edit: LoraLoaderOverrideEdit,
  initialAuthority: RuntimeSettingsAuthority,
  transport: RuntimeSettingsAuthorityTransport,
): Promise<RuntimeSettingsAuthority> {
  if (!editBaseStillCurrent(initialAuthority.settings, edit)) {
    throw new LoraLoaderOverrideConflictError();
  }
  const firstDocument = applyLoraLoaderOverrideEdit(initialAuthority.settings, edit);
  try {
    return await transport.write(firstDocument, initialAuthority.authority_token);
  } catch (firstReason) {
    if (!isAuthorityConflict(firstReason)) {
      return recoverAmbiguousWrite(edit, firstReason, transport);
    }
    const latest = await transport.read();
    if (editIsApplied(latest.settings, edit)) return latest;
    if (!editBaseStillCurrent(latest.settings, edit)) {
      throw new LoraLoaderOverrideConflictError();
    }
    const rebasedDocument = applyLoraLoaderOverrideEdit(latest.settings, edit);
    try {
      return await transport.write(rebasedDocument, latest.authority_token);
    } catch (retryReason) {
      if (isAuthorityConflict(retryReason)) {
        try {
          const afterRetry = await transport.read();
          if (editIsApplied(afterRetry.settings, edit)) return afterRetry;
        } catch {
          // The stable retry-exhausted conflict below is the public result.
        }
        throw new LoraLoaderOverrideConflictError(
          "LoRA 加载器映射在重试时再次变化；请重新检查后再保存",
        );
      }
      return recoverAmbiguousWrite(edit, retryReason, transport);
    }
  }
}
