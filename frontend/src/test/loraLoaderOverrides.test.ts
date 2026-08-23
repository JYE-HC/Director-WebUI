import { describe, expect, it, vi } from "vitest";
import {
  compareLoraLoaderBinding,
  compareUtf16Strings,
  DEFAULT_SETTINGS,
  normalizeLoraLoaderOverrides,
  type LoraLoaderOverrideRecord,
  type RuntimeSettings,
} from "../api/types";
import {
  applyLoraLoaderOverrideEdit,
  LoraLoaderOverrideConflictError,
  saveLoraLoaderOverrideWithCas,
  type LoraLoaderOverrideEdit,
} from "../state/loraLoaderOverrides";

const record = (
  lora: string,
  adapter: LoraLoaderOverrideRecord["adapter_id"] = "model_only",
  options: Record<string, boolean> = {},
): LoraLoaderOverrideRecord => ({
  lora_filename: lora,
  adapter_id: adapter,
  options,
});

const settingsWith = (...records: LoraLoaderOverrideRecord[]): RuntimeSettings => ({
  ...structuredClone(DEFAULT_SETTINGS),
  lora_loader_overrides: normalizeLoraLoaderOverrides(records),
});

const editFor = (
  base: RuntimeSettings,
  next: LoraLoaderOverrideRecord,
): LoraLoaderOverrideEdit => ({ base_settings: base, original: null, next });

describe("RuntimeSettingsV3 exact LoRA loader mappings", () => {
  it("uses ECMAScript UTF-16 LoRA-path order without locale collation", () => {
    expect(compareUtf16Strings("😀", "\uE000")).toBe(-1);
    expect(compareUtf16Strings("Model/A", "Model/a")).toBe(-1);
    const values = [
      record("LoRA/Z.safetensors"),
      record("LoRA/a.safetensors"),
      record("LoRA/A.safetensors", "minimax_h3_turbo", { low_vram: false }),
    ];
    const normalized = normalizeLoraLoaderOverrides(values);
    expect(normalized.map((entry) => entry.lora_filename)).toEqual([
      "LoRA/A.safetensors",
      "LoRA/Z.safetensors",
      "LoRA/a.safetensors",
    ]);
    expect(compareLoraLoaderBinding(normalized[0], normalized[1])).toBeLessThan(0);
  });

  it("restore default deletes only the exact case-sensitive LoRA path", () => {
    const upper = record("LoRA/Turbo.safetensors");
    const lower = record("lora/Turbo.safetensors");
    const base = settingsWith(upper, lower);
    const restored = applyLoraLoaderOverrideEdit(base, {
      base_settings: base,
      original: upper,
      next: null,
    });
    expect(restored.lora_loader_overrides).toEqual([lower]);
  });

  it("enforces the 256-record bound without losing an existing record", () => {
    const full = settingsWith(...Array.from({ length: 256 }, (_, index) =>
      record(`LoRA/${index.toString().padStart(3, "0")}`)));
    expect(() => applyLoraLoaderOverrideEdit(full, editFor(
      full,
      record("LoRA/new"),
    ))).toThrow("超过 256 条");
    expect(full.lora_loader_overrides).toHaveLength(256);
  });

  it("rebases once after 409 when only a different binding changed", async () => {
    const target = record("LoRA/Target", "minimax_h3_turbo", { low_vram: true });
    const concurrent = record("LoRA/Other", "model_only");
    const base = settingsWith();
    const fresh = settingsWith(concurrent);
    const write = vi.fn()
      .mockRejectedValueOnce({ status: 409 })
      .mockImplementationOnce(async (settings: RuntimeSettings) => ({
        settings,
        authority_token: "c".repeat(64),
      }));
    const read = vi.fn().mockResolvedValue({
      settings: fresh,
      authority_token: "b".repeat(64),
    });

    const committed = await saveLoraLoaderOverrideWithCas(
      editFor(base, target),
      { settings: base, authority_token: "a".repeat(64) },
      { read, write },
    );

    expect(write).toHaveBeenCalledTimes(2);
    expect(write.mock.calls[1][1]).toBe("b".repeat(64));
    expect(write.mock.calls[1][0].lora_loader_overrides).toEqual(
      normalizeLoraLoaderOverrides([target, concurrent]),
    );
    expect(committed.settings.lora_loader_overrides).toContainEqual(concurrent);
  });

  it("surfaces a visible same-binding conflict and never retries", async () => {
    const target = record("LoRA/Target", "minimax_h3_turbo", { low_vram: false });
    const base = settingsWith();
    const changed = settingsWith({ ...target, adapter_id: "model_only", options: {} });
    const write = vi.fn().mockRejectedValue({ status: 409 });
    const read = vi.fn().mockResolvedValue({
      settings: changed,
      authority_token: "b".repeat(64),
    });

    await expect(saveLoraLoaderOverrideWithCas(
      editFor(base, target),
      { settings: base, authority_token: "a".repeat(64) },
      { read, write },
    )).rejects.toBeInstanceOf(LoraLoaderOverrideConflictError);
    expect(write).toHaveBeenCalledTimes(1);
  });

  it("recognizes a lost write response by authoritative read and does not duplicate it", async () => {
    const target = record("LoRA/Target", "minimax_h3_turbo", { low_vram: false });
    const base = settingsWith();
    const committed = applyLoraLoaderOverrideEdit(base, editFor(base, target));
    const write = vi.fn().mockRejectedValue(new Error("response lost"));
    const read = vi.fn().mockResolvedValue({
      settings: committed,
      authority_token: "b".repeat(64),
    });

    await expect(saveLoraLoaderOverrideWithCas(
      editFor(base, target),
      { settings: base, authority_token: "a".repeat(64) },
      { read, write },
    )).resolves.toEqual({ settings: committed, authority_token: "b".repeat(64) });
    expect(write).toHaveBeenCalledTimes(1);
  });
});
