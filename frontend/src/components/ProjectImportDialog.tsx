import { useRef, useState, type ChangeEvent } from "react";
import { directorApi } from "../api/client";
import type {
  ProjectImportCreativeSelection,
  ProjectImportLegacyCreativeContext,
  ProjectImportPreflightRequest,
  ProjectImportPreflightResponse,
  ProjectSummary,
  LegacyRuntimeSettingsV1,
} from "../api/types";
import {
  loraFeatureSelection,
  type FeatureSelection,
  type ModelStack,
  type TimelineGenerationMode,
  type TimelineProject,
} from "../domain/timelineProject";

interface ProjectImportDialogProps {
  currentProject: TimelineProject;
  disabled?: boolean;
  onImported: (project: ProjectSummary) => void | Promise<void>;
}

interface ImportSource {
  title: string;
  document: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function creativeSelectionFromProject(
  project: TimelineProject,
): ProjectImportCreativeSelection {
  return {
    model_stack: structuredClone(project.model_stack),
    lora: structuredClone(loraFeatureSelection(project.features)),
  };
}

function importTitle(value: unknown, filename: string): string {
  if (isRecord(value) && typeof value.title === "string") return value.title;
  return filename.replace(/\.json$/i, "").slice(0, 256);
}

function preflightPayloadFromFile(
  value: unknown,
  filename: string,
): ProjectImportPreflightRequest {
  if (!isRecord(value) || !("document" in value)) {
    return {
      title: importTitle(value, filename),
      document: value,
    };
  }
  const base = {
    title: importTitle(value, filename),
    document: value.document,
  };
  const authorities = [
    "legacy_runtime_settings",
    "legacy_creative_context",
    "creative_selection",
  ].filter((key) => value[key] !== undefined);
  if (authorities.length > 1) {
    throw new Error("导入文件包含多个创作配置权威，无法确定使用哪一个");
  }
  if (value.legacy_runtime_settings !== undefined) return {
    ...base,
    legacy_runtime_settings: value.legacy_runtime_settings as LegacyRuntimeSettingsV1,
  };
  if (value.legacy_creative_context !== undefined) return {
    ...base,
    legacy_creative_context:
      value.legacy_creative_context as ProjectImportLegacyCreativeContext,
  };
  if (value.creative_selection !== undefined) return {
    ...base,
    creative_selection: value.creative_selection as ProjectImportCreativeSelection,
  };
  return base;
}

function loraFamilies(selection: FeatureSelection) {
  return selection.params.by_family as {
    fl2va: { enabled: boolean; filename: string | null; strength: number };
    ref2va: { enabled: boolean; filename: string | null; strength: number };
  };
}

function requiresCreativeSelection(
  document: unknown,
  response: ProjectImportPreflightResponse,
): boolean {
  return isRecord(document) &&
    document.version === 4 &&
    response.status === "needs_input" &&
    response.missing_context.includes("creative_selection");
}

export function ProjectImportDialog({
  currentProject,
  disabled = false,
  onImported,
}: ProjectImportDialogProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState<ImportSource | null>(null);
  const [selection, setSelection] = useState<ProjectImportCreativeSelection | null>(null);
  const [preflight, setPreflight] = useState<ProjectImportPreflightResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runPreflight = async (payload: ProjectImportPreflightRequest) => {
    setBusy(true);
    setError(null);
    try {
      const response = await directorApi.preflightProjectImport(payload);
      setPreflight(response);
      if (requiresCreativeSelection(payload.document, response)) {
        // This is a visible draft copied from the current project authority,
        // never from mutable RuntimeSettings. It is submitted only after the
        // user explicitly confirms the selection below.
        setSelection(creativeSelectionFromProject(currentProject));
      } else {
        setSelection(null);
      }
    } catch (reason) {
      setPreflight(null);
      setError(reason instanceof Error ? reason.message : "项目导入预检失败");
    } finally {
      setBusy(false);
    }
  };

  const selectFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setOpen(true);
    setPreflight(null);
    setSelection(null);
    setError(null);
    try {
      const value: unknown = JSON.parse(await file.text());
      const payload = preflightPayloadFromFile(value, file.name);
      setSource({ title: payload.title, document: payload.document });
      await runPreflight(payload);
    } catch (reason) {
      setSource(null);
      setError(reason instanceof Error ? reason.message : "无法读取项目 JSON 文件");
    }
  };

  const updateModel = (role: keyof ModelStack, filename: string) => {
    setSelection((current) => current ? {
      ...current,
      model_stack: {
        ...current.model_stack,
        [role]: { filename: filename.trim() || null },
      },
    } : current);
  };

  const updateLora = (
    family: TimelineGenerationMode,
    patch: Partial<{ enabled: boolean; filename: string | null; strength: number }>,
  ) => {
    setSelection((current) => {
      if (!current) return current;
      const families = loraFamilies(current.lora);
      const nextFamilies = {
        ...families,
        [family]: { ...families[family], ...patch },
      };
      return {
        ...current,
        lora: {
          enabled: nextFamilies.fl2va.enabled || nextFamilies.ref2va.enabled,
          params: { by_family: nextFamilies },
        },
      };
    });
  };

  const confirmSelection = async () => {
    if (!source || !selection) return;
    await runPreflight({
      ...source,
      creative_selection: structuredClone(selection),
    });
  };

  const commit = async () => {
    if (
      preflight?.status !== "ready" ||
      !preflight.commit_token
    ) return;
    setBusy(true);
    setError(null);
    try {
      const imported = await directorApi.commitProjectImport({
        commit_token: preflight.commit_token,
        input_digest: preflight.input_digest,
      });
      await onImported(imported);
      setOpen(false);
      setSource(null);
      setSelection(null);
      setPreflight(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "提交导入项目失败");
    } finally {
      setBusy(false);
    }
  };

  const lora = selection ? loraFamilies(selection.lora) : null;
  return <>
    <input
      ref={fileInputRef}
      type="file"
      accept="application/json,.json"
      aria-label="选择项目 JSON 文件"
      hidden
      disabled={disabled || busy}
      onChange={(event) => void selectFile(event)}
    />
    <button
      type="button"
      className="topbar__project-import"
      disabled={disabled || busy}
      onClick={() => fileInputRef.current?.click()}
    >导入项目</button>
    {open && <div className="settings-overlay" role="dialog" aria-modal="true" aria-labelledby="project-import-title">
      <header>
        <h2 id="project-import-title">导入项目</h2>
        <button type="button" aria-label="关闭项目导入" disabled={busy} onClick={() => setOpen(false)}>×</button>
      </header>
      {busy && <p role="status">正在核对导入文件…</p>}
      {error && <p role="alert">{error}</p>}
      {source && <p>文件将创建为项目“{source.title || "未命名项目"}”。</p>}
      {preflight?.status === "needs_input" && selection && lora && <section aria-label="导入创作配置">
        <p>旧项目没有可信创作上下文。请明确确认 model stack 与 LoRA；这里的初值来自当前项目顶部配置，不会读取系统运行设置。</p>
        {(["fl2va", "ref2va", "clip", "video_vae", "audio_vae"] as const).map((role) =>
          <label key={role}>{role}
            <input
              aria-label={`导入 ${role} 模型`}
              value={selection.model_stack[role].filename ?? ""}
              onChange={(event) => updateModel(role, event.target.value)}
            />
          </label>)}
        {(["fl2va", "ref2va"] as const).map((family) => <fieldset key={family}>
          <legend>{family} LoRA</legend>
          <label><input
            type="checkbox"
            aria-label={`导入 ${family} LoRA 启用`}
            checked={lora[family].enabled}
            onChange={(event) => updateLora(family, { enabled: event.target.checked })}
          />启用</label>
          <label>文件<input
            aria-label={`导入 ${family} LoRA 文件`}
            value={lora[family].filename ?? ""}
            onChange={(event) => updateLora(family, { filename: event.target.value.trim() || null })}
          /></label>
          <label>强度<input
            type="number"
            min="-10"
            max="10"
            step="0.01"
            aria-label={`导入 ${family} LoRA 强度`}
            value={lora[family].strength}
            onChange={(event) => {
              const strength = Number(event.target.value);
              if (Number.isFinite(strength)) updateLora(family, { strength });
            }}
          /></label>
        </fieldset>)}
        <button type="button" disabled={busy} onClick={() => void confirmSelection()}>
          确认创作配置并重新预检
        </button>
      </section>}
      {preflight?.status === "needs_input" && !selection && <section aria-label="导入预检问题">
        <p>这个文件还不能导入。请先解决以下缺失项或宿主能力问题，再重新选择文件预检。</p>
        <dl>
          <dt>缺失上下文</dt><dd>{preflight.missing_context.join("、") || "无"}</dd>
          <dt>缺失模型绑定</dt><dd>{preflight.missing_model_bindings.join("、") || "无"}</dd>
          <dt>能力问题</dt><dd>{preflight.capability_issues.length
            ? preflight.capability_issues.map((issue) => JSON.stringify(issue)).join("；")
            : "无"}</dd>
        </dl>
      </section>}
      {preflight?.status === "ready" && <section aria-label="导入预检结果">
        <p>迁移文档已就绪；只有点击提交才会创建项目。</p>
        <dl>
          <dt>缺失上下文</dt><dd>{preflight.missing_context.join("、") || "无"}</dd>
          <dt>缺失模型绑定</dt><dd>{preflight.missing_model_bindings.join("、") || "无"}</dd>
          <dt>能力问题</dt><dd>{preflight.capability_issues.length
            ? preflight.capability_issues.map((issue) => JSON.stringify(issue)).join("；")
            : "无"}</dd>
        </dl>
        <button type="button" disabled={busy} onClick={() => void commit()}>提交导入项目</button>
      </section>}
    </div>}
  </>;
}
