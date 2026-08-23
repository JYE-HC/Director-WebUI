import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SETTINGS,
  type RuntimeSettings,
  type RuntimeSettingsAuthority,
  type RuntimeSettingsV2,
} from "../api/types";
import {
  buildRuntimeSettingsV2WalEvidenceExport,
  buildRuntimeSettingsV2RecoveryPlan,
  discardRuntimeSettingsV2WalCandidate,
  LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY,
  listRuntimeSettingsV2WalCandidates,
  listRuntimeSettingsV2WalEvidence,
  recoverRuntimeSettingsV2WalCandidate,
  RUNTIME_SETTINGS_V2_WAL_QUARANTINE_PREFIX,
} from "../state/runtimeSettingsV2Migration";

const DATABASE_PATH = "/srv/director/data/directordeck.sqlite3";
const CURRENT_V3_WAL_KEY = "directordeck:v4:runtime-settings-pending";

function runtimeV2(): RuntimeSettingsV2 {
  return {
    schema_version: 2,
    client_id: "v2-pending-client",
    memory_policy: "keep_resident",
    raylight_residency_policy: "keep_until_switch",
    multi_gpu_enabled: false,
    placement: structuredClone(DEFAULT_SETTINGS.placement),
    legacy_lora_resolution_compat: {
      schema_version: 1,
      auto_resolution_strategy_version: "v4-known-filename-or-safetensors-metadata-v1",
      explicit_overrides: [{
        family: "fl2va",
        model_filename: "Models/H3/Base.Exact.safetensors",
        lora_filename: "LoRA/H3/Turbo.Exact.safetensors",
        loader: "dedicated",
      }],
    },
  };
}

function envelopeRaw(settings: unknown = runtimeV2(), databasePath = DATABASE_PATH): string {
  return `${JSON.stringify({
    format: "director-pending-runtime-settings",
    version: 3,
    owner_id: "tab-stage6-pending",
    pending: true,
    active_database_path: databasePath,
    written_at_ms: 123456789,
    settings,
  })}\n`;
}

function settingsWith(
  overrides: RuntimeSettings["lora_loader_overrides"] = [],
): RuntimeSettings {
  return {
    ...structuredClone(DEFAULT_SETTINGS),
    client_id: "current-v3-client",
    lora_loader_overrides: overrides,
  };
}

function authority(settings: RuntimeSettings, token = "a".repeat(64)): RuntimeSettingsAuthority {
  return { settings: structuredClone(settings), authority_token: token };
}

function installCandidate(raw = envelopeRaw()) {
  localStorage.setItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY, raw);
  const [candidate] = listRuntimeSettingsV2WalCandidates(DATABASE_PATH);
  expect(candidate).toBeDefined();
  return candidate;
}

beforeEach(() => localStorage.clear());

describe("RuntimeSettingsV2 pending WAL explicit recovery", () => {
  it("hydrates by exact-byte quarantine only and preserves an independent V3 WAL", () => {
    const raw = envelopeRaw();
    const currentV3Raw = JSON.stringify({ version: 4, evidence: "independent" });
    localStorage.setItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY, raw);
    localStorage.setItem(CURRENT_V3_WAL_KEY, currentV3Raw);

    const [candidate] = listRuntimeSettingsV2WalCandidates(DATABASE_PATH);

    expect(candidate.raw).toBe(raw);
    expect(candidate.storage_key).toMatch(
      new RegExp(`^${RUNTIME_SETTINGS_V2_WAL_QUARANTINE_PREFIX}`),
    );
    expect(localStorage.getItem(candidate.storage_key)).toBe(raw);
    expect(localStorage.getItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(CURRENT_V3_WAL_KEY)).toBe(currentV3Raw);
  });

  it("surfaces malformed evidence as corrupt without ever exposing a CAS candidate", () => {
    const malformedRaw = envelopeRaw({ ...runtimeV2(), unexpected: true });
    localStorage.setItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY, malformedRaw);

    const evidence = listRuntimeSettingsV2WalEvidence(DATABASE_PATH);
    expect(evidence).toHaveLength(1);
    expect(evidence[0]).toMatchObject({
      kind: "corrupt",
      raw: malformedRaw,
      ownership: {
        owner_id: "tab-stage6-pending",
        database_scope: "current",
        written_at_ms: 123456789,
      },
    });
    expect(listRuntimeSettingsV2WalCandidates(DATABASE_PATH)).toEqual([]);
    expect(localStorage.getItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(evidence[0].storage_key)).toBe(malformedRaw);
  });

  it("surfaces non-exact and cross-database quarantines as corrupt tagged evidence", () => {
    const nonExactRaw = envelopeRaw();
    const foreignRaw = envelopeRaw(runtimeV2(), "/srv/director/data/other.sqlite3");
    localStorage.setItem(`${RUNTIME_SETTINGS_V2_WAL_QUARANTINE_PREFIX}wrong-digest`, nonExactRaw);
    localStorage.setItem(LEGACY_RUNTIME_SETTINGS_V2_WAL_STORAGE_KEY, foreignRaw);

    const evidence = listRuntimeSettingsV2WalEvidence(DATABASE_PATH);

    expect(evidence).toHaveLength(2);
    expect(evidence.every((entry) => entry.kind === "corrupt")).toBe(true);
    expect(evidence.map((entry) => entry.ownership.database_scope).sort())
      .toEqual(["current", "other"]);
    expect(evidence.map((entry) => buildRuntimeSettingsV2WalEvidenceExport(entry).contents))
      .toEqual(expect.arrayContaining([nonExactRaw, foreignRaw]));
    expect(evidence.every((entry) =>
      buildRuntimeSettingsV2WalEvidenceExport(entry).filename.endsWith(".bin"))).toBe(true);
  });

  it("merges exact mappings on top of the newest authority and preserves unrelated records", () => {
    const candidate = installCandidate();
    const unrelated = {
      lora_filename: "LoRA/H3/Other.safetensors",
      adapter_id: "model_only" as const,
      options: {},
    };
    const current = settingsWith([unrelated]);

    const plan = buildRuntimeSettingsV2RecoveryPlan(
      candidate,
      "merge-exact-mappings",
      current,
    );

    expect(plan?.document.client_id).toBe("current-v3-client");
    expect(plan?.document.lora_loader_overrides).toEqual(expect.arrayContaining([
      unrelated,
      expect.objectContaining({
        lora_filename: "LoRA/H3/Turbo.Exact.safetensors",
        adapter_id: "minimax_h3_turbo",
        options: { low_vram: false },
      }),
    ]));
  });

  it("fails closed when the same exact binding already selects another adapter", () => {
    const candidate = installCandidate();
    const current = settingsWith([{
      lora_filename: "LoRA/H3/Turbo.Exact.safetensors",
      adapter_id: "model_only",
      options: {},
    }]);

    expect(buildRuntimeSettingsV2RecoveryPlan(
      candidate,
      "merge-exact-mappings",
      current,
    )).toBeNull();
  });

  it("treats the same exact binding and adapter as an idempotent union member", () => {
    const candidate = installCandidate();
    const existing = {
      lora_filename: "LoRA/H3/Turbo.Exact.safetensors",
      adapter_id: "minimax_h3_turbo" as const,
      options: { low_vram: false },
    };

    const plan = buildRuntimeSettingsV2RecoveryPlan(
      candidate,
      "merge-exact-mappings",
      settingsWith([existing]),
    );

    expect(plan?.document.lora_loader_overrides).toEqual([existing]);
  });

  it("fails closed when a distinct recovered binding would exceed 256 records", () => {
    const candidate = installCandidate();
    const current = settingsWith(Array.from({ length: 256 }, (_, index) => ({
      lora_filename: `LoRA/H3/existing-${index.toString().padStart(3, "0")}.safetensors`,
      adapter_id: "model_only" as const,
      options: {},
    })));

    expect(buildRuntimeSettingsV2RecoveryPlan(
      candidate,
      "merge-exact-mappings",
      current,
    )).toBeNull();
    expect(localStorage.getItem(candidate.storage_key)).toBe(candidate.raw);
  });

  it("uses one whole-document CAS and clears only the acknowledged candidate", async () => {
    const candidate = installCandidate();
    const unrelatedRaw = JSON.stringify({ evidence: "unrelated" });
    localStorage.setItem("directordeck:test:unrelated", unrelatedRaw);
    const current = settingsWith();
    const read = vi.fn().mockResolvedValue(authority(current));
    const write = vi.fn().mockImplementation(async (document: RuntimeSettings) =>
      authority(document, "b".repeat(64)));

    const result = await recoverRuntimeSettingsV2WalCandidate(
      candidate,
      "merge-exact-mappings",
      { read, write },
    );

    expect(result.acknowledgement).toBe("cas-ack");
    expect(read).toHaveBeenCalledTimes(1);
    expect(write).toHaveBeenCalledTimes(1);
    expect(write).toHaveBeenCalledWith(
      expect.objectContaining({ lora_loader_overrides: [expect.any(Object)] }),
      "a".repeat(64),
    );
    expect(localStorage.getItem(candidate.storage_key)).toBeNull();
    expect(localStorage.getItem("directordeck:test:unrelated")).toBe(unrelatedRaw);
  });

  it("rechecks an independent V3 WAL at click time and performs zero transport calls", async () => {
    const candidate = installCandidate();
    localStorage.setItem(CURRENT_V3_WAL_KEY, JSON.stringify({ evidence: "new-tab" }));
    const read = vi.fn();
    const write = vi.fn();

    await expect(recoverRuntimeSettingsV2WalCandidate(
      candidate,
      "merge-exact-mappings",
      { read, write },
    )).rejects.toThrow("独立 RuntimeSettingsV3");

    expect(read).not.toHaveBeenCalled();
    expect(write).not.toHaveBeenCalled();
    expect(localStorage.getItem(candidate.storage_key)).toBe(candidate.raw);
  });

  it("preserves V2 evidence when another tab creates V3 WAL before CAS acknowledgement", async () => {
    const candidate = installCandidate();
    const current = settingsWith();
    const read = vi.fn().mockResolvedValue(authority(current));
    const write = vi.fn().mockImplementation(async (document: RuntimeSettings) => {
      localStorage.setItem(CURRENT_V3_WAL_KEY, JSON.stringify({ evidence: "during-cas" }));
      return authority(document, "b".repeat(64));
    });

    await expect(recoverRuntimeSettingsV2WalCandidate(
      candidate,
      "merge-exact-mappings",
      { read, write },
    )).rejects.toThrow("独立 RuntimeSettingsV3");

    expect(read).toHaveBeenCalledTimes(1);
    expect(write).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem(candidate.storage_key)).toBe(candidate.raw);
  });

  it("preserves quarantine on CAS 409 without rebasing or retrying", async () => {
    const candidate = installCandidate();
    const read = vi.fn().mockResolvedValue(authority(settingsWith()));
    const conflict = Object.assign(new Error("conflict"), { status: 409 });
    const write = vi.fn().mockRejectedValue(conflict);

    await expect(recoverRuntimeSettingsV2WalCandidate(
      candidate,
      "merge-exact-mappings",
      { read, write },
    )).rejects.toBe(conflict);

    expect(read).toHaveBeenCalledTimes(1);
    expect(write).toHaveBeenCalledTimes(1);
    expect(localStorage.getItem(candidate.storage_key)).toBe(candidate.raw);
  });

  it("clears a lost ACK only after a GET proves intended fields and mapping union", async () => {
    const old = runtimeV2();
    old.client_id = "recovered-runtime-client";
    const replacementRaw = envelopeRaw(old);
    const replacementCandidate = installCandidate(replacementRaw);
    const initial = settingsWith([{
      lora_filename: "LoRA/H3/Unrelated.safetensors",
      adapter_id: "model_only",
      options: {},
    }]);
    let attempted: RuntimeSettings | null = null;
    const read = vi.fn()
      .mockResolvedValueOnce(authority(initial))
      .mockImplementation(async () => authority({
        ...structuredClone(attempted!),
        lora_loader_overrides: [
          ...attempted!.lora_loader_overrides,
          {
            lora_filename: "LoRA/H3/Later.safetensors",
            adapter_id: "model_only",
            options: {},
          },
        ],
      }, "c".repeat(64)));
    const lostAck = new TypeError("network response lost");
    const write = vi.fn().mockImplementation(async (document: RuntimeSettings) => {
      attempted = structuredClone(document);
      throw lostAck;
    });

    const result = await recoverRuntimeSettingsV2WalCandidate(
      replacementCandidate,
      "apply-runtime-and-merge-mappings",
      { read, write },
    );

    expect(result.acknowledgement).toBe("lost-ack-proven");
    expect(read).toHaveBeenCalledTimes(2);
    expect(write).toHaveBeenCalledTimes(1);
    expect(result.authority.settings.client_id).toBe("recovered-runtime-client");
    expect(localStorage.getItem(replacementCandidate.storage_key)).toBeNull();
  });

  it("preserves quarantine when the lost-ACK GET cannot prove the mapping union", async () => {
    const candidate = installCandidate();
    const current = settingsWith();
    const read = vi.fn()
      .mockResolvedValueOnce(authority(current))
      .mockResolvedValueOnce(authority(current, "c".repeat(64)));
    const write = vi.fn().mockRejectedValue(new TypeError("network response lost"));

    await expect(recoverRuntimeSettingsV2WalCandidate(
      candidate,
      "merge-exact-mappings",
      { read, write },
    )).rejects.toThrow("network response lost");
    expect(localStorage.getItem(candidate.storage_key)).toBe(candidate.raw);
  });

  it("explicit discard compare-deletes only the candidate and performs no transport write", () => {
    const candidate = installCandidate();
    const currentV3Raw = JSON.stringify({ version: 4, evidence: "independent" });
    localStorage.setItem(CURRENT_V3_WAL_KEY, currentV3Raw);
    const read = vi.fn();
    const write = vi.fn();

    expect(discardRuntimeSettingsV2WalCandidate(candidate)).toBe(true);

    expect(read).not.toHaveBeenCalled();
    expect(write).not.toHaveBeenCalled();
    expect(localStorage.getItem(candidate.storage_key)).toBeNull();
    expect(localStorage.getItem(CURRENT_V3_WAL_KEY)).toBe(currentV3Raw);
  });
});
