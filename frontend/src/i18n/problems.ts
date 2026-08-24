import { ApiError } from "../api/client";
import type { Translator } from "./translator";

const MODEL_BINDINGS = new Set([
  "fl2va",
  "ref2va",
  "clip",
  "video_vae",
  "audio_vae",
  "loras:fl2va",
  "loras:ref2va",
]);
const ERROR_CODE = /^[A-Za-z][A-Za-z0-9_.:-]{0,127}$/;

export interface LocalizedProblem {
  code: string;
  message: string;
  remediation: string;
  technicalMessage: string | null;
  action: "open_global_settings" | null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stableCode(value: unknown): string | null {
  return typeof value === "string" && ERROR_CODE.test(value) ? value : null;
}

function problemDetail(value: unknown): Record<string, unknown> | null {
  if (!isRecord(value)) return null;
  if (isRecord(value.detail)) return value.detail;
  return value;
}

function detailBindings(detail: Record<string, unknown> | null): string[] {
  if (!detail) return [];
  const direct = Array.isArray(detail.bindings) ? detail.bindings : [];
  const reasons = Array.isArray(detail.reasons) ? detail.reasons : [];
  const candidates: unknown[] = [...direct];
  for (const reason of reasons.slice(0, 32)) {
    if (!isRecord(reason) || !isRecord(reason.safe_details)) continue;
    if (Array.isArray(reason.safe_details.bindings)) {
      candidates.push(...reason.safe_details.bindings.slice(0, 5));
    }
  }
  const seen = new Set<string>();
  return candidates.filter((binding): binding is string => {
    if (typeof binding !== "string" || !MODEL_BINDINGS.has(binding) || seen.has(binding)) return false;
    seen.add(binding);
    return true;
  });
}

function detailNodes(detail: Record<string, unknown> | null): string[] {
  if (!detail) return [];
  const direct = Array.isArray(detail.missing_node_class_types)
    ? detail.missing_node_class_types
    : [];
  const reasons = Array.isArray(detail.reasons) ? detail.reasons : [];
  const candidates: unknown[] = [...direct];
  for (const reason of reasons.slice(0, 32)) {
    if (!isRecord(reason) || !isRecord(reason.safe_details)) continue;
    candidates.push(reason.safe_details.class_type);
  }
  const seen = new Set<string>();
  return candidates.filter((node): node is string => {
    if (
      typeof node !== "string" ||
      !node ||
      node.trim() !== node ||
      node.length > 256 ||
      /[\u0000-\u001f\u007f]/.test(node) ||
      seen.has(node)
    ) return false;
    seen.add(node);
    return true;
  });
}

function detailMaxFrames(detail: Record<string, unknown> | null): number {
  if (!detail) return 512;
  const candidates: unknown[] = [detail.max_frames];
  if (Array.isArray(detail.reasons)) {
    for (const reason of detail.reasons.slice(0, 32)) {
      if (isRecord(reason) && isRecord(reason.safe_details)) {
        candidates.push(reason.safe_details.max_frames);
      }
    }
  }
  const value = candidates.find((candidate) =>
    typeof candidate === "number" &&
    Number.isSafeInteger(candidate) &&
    candidate >= 1 &&
    candidate <= 1_000_000,
  );
  return typeof value === "number" ? value : 512;
}

function parseStoredError(raw: string): Record<string, unknown> | null {
  try {
    return problemDetail(JSON.parse(raw) as unknown);
  } catch {
    return null;
  }
}

function sourceProblem(source: unknown): {
  code: string;
  detail: Record<string, unknown> | null;
  technicalMessage: string | null;
} {
  if (source instanceof ApiError) {
    const detail = problemDetail(source.details);
    return {
      code: stableCode(source.code) ?? stableCode(detail?.code) ?? "unknown",
      detail,
      technicalMessage: source.message,
    };
  }
  if (typeof source === "string") {
    const detail = parseStoredError(source);
    return {
      code: stableCode(detail?.code) ?? "unknown",
      detail,
      technicalMessage: typeof detail?.message === "string" ? detail.message : source,
    };
  }
  if (source instanceof Error) {
    return { code: "unknown", detail: null, technicalMessage: source.message };
  }
  const detail = problemDetail(source);
  return {
    code: stableCode(detail?.code) ?? "unknown",
    detail,
    technicalMessage: typeof detail?.message === "string" ? detail.message : null,
  };
}

export function localizeProblem(source: unknown, translator: Translator): LocalizedProblem {
  const problem = sourceProblem(source);
  const bindings = detailBindings(problem.detail);
  const nodes = detailNodes(problem.detail);
  const params = {
    code: problem.code,
    bindings: bindings.length
      ? bindings.map((binding) => translator.t(`modelRoles.${binding}`)).join("、")
      : translator.t("modelRoles.required"),
    nodes: nodes.length
      ? nodes.join("、")
      : translator.t("errors.node_unavailable.unknownNodes"),
    maxFrames: detailMaxFrames(problem.detail),
  };
  const messageKey = `errors.${problem.code}.message`;
  const remediationKey = `errors.${problem.code}.remediation`;
  const translatedMessage = translator.t(messageKey, params);
  const translatedRemediation = translator.t(remediationKey, params);
  return {
    code: problem.code,
    message: translatedMessage === messageKey
      ? translator.t("errors.generic.message", params)
      : translatedMessage,
    remediation: translatedRemediation === remediationKey
      ? translator.t("errors.generic.remediation", params)
      : translatedRemediation,
    technicalMessage: problem.technicalMessage,
    action: problem.code === "model_binding_required" ? "open_global_settings" : null,
  };
}
