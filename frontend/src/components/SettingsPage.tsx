import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, directorApi } from "../api/client";
import {
  isConfiguredComfyUrl,
  rayLightResidencyPolicyAfterBindingChange,
  resolveExecutionBackend,
  type CapabilityReport,
  type ConnectionTestResult,
  type DeviceTarget,
  type DiffusionModelBinding,
  type GPUResource,
  type ModelInventory,
  type ModelRole,
  type RayLightProfile,
  type RayLightRuntimeStatus,
  type RuntimeSettings,
  type StandardLoraLoader,
  type StorageConfiguration,
} from "../api/types";
import {
  TIMELINE_MODE_META,
  TIMELINE_MODE_ORDER,
} from "../domain/timelineProject";
import type { UiTheme } from "../domain/theme";
import { DeferredNumberInput, Field, formatBytes, Panel, Spinner, StatusDot } from "./ui";

function sameSettings(left: RuntimeSettings, right: RuntimeSettings): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

export function validateDatabasePathInput(value: string): string | null {
  const path = value.trim();
  if (!path) return "请填写数据库目标路径";
  if (path.length > 4096) return "数据库路径不能超过 4096 个字符";
  if (/[\u0000-\u001f\u007f]/.test(path)) return "数据库路径不能包含控制字符";
  if (!path.startsWith("/") && !path.startsWith("~/"))
    return "数据库路径必须是绝对路径或以 ~/ 开头";
  return null;
}

const STORAGE_SOURCE_LABEL: Record<StorageConfiguration["source"], string> = {
  explicit: "启动参数",
  environment: "环境变量",
  bootstrap: "启动引导配置",
  legacy: "旧版路径兼容",
  default: "默认路径",
};

const HIDDEN_PATH_VALUE = "••••••••••••••••";

function VisibilityToggle({
  label,
  visible,
  onToggle,
}: {
  label: string;
  visible: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      className="visibility-toggle"
      aria-label={label}
      aria-pressed={visible}
      title={visible ? "隐藏内容" : "显示内容"}
      onClick={onToggle}
    >
      {visible ? (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
          <circle cx="12" cy="12" r="2.75" />
        </svg>
      ) : (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M3 3l18 18M10.6 6.1A10.6 10.6 0 0 1 12 6c6 0 9.5 6 9.5 6a16.6 16.6 0 0 1-2.7 3.4M14.1 14.1A3 3 0 0 1 9.9 9.9M6.2 7.2C3.8 9 2.5 12 2.5 12s3.5 6 9.5 6c1.2 0 2.3-.2 3.3-.6" />
        </svg>
      )}
    </button>
  );
}

function safeStorageError(reason: unknown, fallback: string): string {
  return reason instanceof ApiError ? reason.message : fallback;
}

function storageRequestWasDefinitivelyRejected(reason: unknown): boolean {
  return reason instanceof ApiError && reason.status >= 400 && reason.status < 500;
}

const MODEL_META: Record<ModelRole, { label: string; description: string; allowCpu: boolean }> = {
  fl2va: { label: "FL2VA 扩散模型", description: "按首尾帧素材自动选择生成配方", allowCpu: true },
  ref2va: { label: "REF2VA 扩散模型", description: "按源视频与参考素材自动选择生成配方", allowCpu: true },
  clip: { label: "文本编码器", description: "共享 CLIP / Qwen 编码器", allowCpu: true },
  video_vae: { label: "视频 VAE", description: "共享画面编码与解码", allowCpu: false },
  audio_vae: { label: "音频 VAE", description: "共享音频编码与解码", allowCpu: false },
};

const STANDARD_LORA_LOADER_LABELS: Record<StandardLoraLoader, string> = {
  dedicated: "MiniMax H3 专用",
  bypass_model_only: "量化旁路 Model Only",
  model_only: "ComfyUI 通用 Model Only",
};

export function validateRuntimeSettingsForm(settings: RuntimeSettings): string[] {
  const errors: string[] = [];
  try {
    const url = new URL(settings.comfy_url);
    if (url.protocol !== "http:" && url.protocol !== "https:") errors.push("ComfyUI 地址必须使用 HTTP 或 HTTPS");
  } catch {
    errors.push("ComfyUI 地址不是有效 URL");
  }
  if (!/^[A-Za-z0-9._:-]{1,128}$/.test(settings.client_id))
    errors.push("客户端 ID 只能包含字母、数字、点、下划线、冒号和连字符");
  if (settings.memory_policy !== "keep_resident")
    errors.push("片段间显存策略无效");
  if (![
    "release_after_sampling",
    "keep_until_switch",
  ].includes(settings.raylight_residency_policy))
    errors.push("RayLight 显存驻留策略无效");
  for (const [role, binding] of Object.entries(settings.models) as [ModelRole, RuntimeSettings["models"][ModelRole]][]) {
    if (!binding.filename || binding.filename.length > 1024)
      errors.push(`${MODEL_META[role].label}必须选择有效模型文件`);
    if (!/^(default|cpu|gpu:(0|[1-9][0-9]*))$/.test(binding.device))
      errors.push(`${MODEL_META[role].label}设备值无效`);
    if (!MODEL_META[role].allowCpu && binding.device === "cpu")
      errors.push(`${MODEL_META[role].label}不允许放在 CPU`);
    if (role === "fl2va" || role === "ref2va") {
      const diffusion = binding as DiffusionModelBinding;
      if (diffusion.lora_name !== null && (!diffusion.lora_name || diffusion.lora_name.length > 1024))
        errors.push(`${MODEL_META[role].label}的 LoRA 文件无效`);
      if (!Number.isFinite(diffusion.lora_strength) || diffusion.lora_strength < -10 || diffusion.lora_strength > 10)
        errors.push(`${MODEL_META[role].label}的 LoRA 强度必须在 -10–10 之间`);
      if (diffusion.lora_loader !== "auto" || diffusion.lora_low_vram)
        errors.push(`${MODEL_META[role].label}的 LoRA 加载方式必须由系统自动选择`);
      const override = diffusion.standard_lora_loader_override;
      if (override !== null && (
        !["dedicated", "bypass_model_only", "model_only"].includes(String(override.loader)) ||
        diffusion.lora_name === null ||
        override.lora_name !== diffusion.lora_name ||
        override.model_filename !== diffusion.filename ||
        typeof override.comfy_origin !== "string" ||
        override.comfy_origin.replace(/\/+$/, "") !== settings.comfy_url.replace(/\/+$/, "") ||
        resolveExecutionBackend(diffusion) !== "standard"
      )) errors.push(`${MODEL_META[role].label}的 Standard LoRA 加载器覆盖与当前地址、底模或 LoRA 不匹配`);
      if (diffusion.backend !== "auto")
        errors.push(`${MODEL_META[role].label}的执行后端必须由逻辑 GPU 池自动选择`);
      const raylight = diffusion.raylight;
      if (
        !Array.isArray(raylight.gpu_select) ||
        raylight.gpu_select.length < 1 ||
        raylight.gpu_select.length > 8 ||
        raylight.gpu_select.some((index) => !Number.isInteger(index) || index < 0 || index > 255) ||
        new Set(raylight.gpu_select).size !== raylight.gpu_select.length
      ) errors.push(`${MODEL_META[role].label}的 RayLight 逻辑 GPU 池必须包含 1–8 个唯一编号`);
      if (
        !Number.isInteger(raylight.ulysses_degree) ||
        raylight.ulysses_degree < 1 ||
        raylight.ulysses_degree > 8 ||
        !Number.isInteger(raylight.ring_degree) ||
        raylight.ring_degree < 1 ||
        raylight.ring_degree > 8 ||
        raylight.ulysses_degree * raylight.ring_degree !== raylight.gpu_select.length
      ) errors.push(`${MODEL_META[role].label}的 RayLight Ulysses × Ring 必须等于 GPU 池大小`);
      if (raylight.cfg_degree !== 1 || raylight.dp_degree !== 1)
        errors.push(`${MODEL_META[role].label}的 RayLight 条件 / 数据并行度必须固定为 1`);
      if (raylight.cpu_offload && !raylight.fsdp)
        errors.push(`${MODEL_META[role].label}只有启用 FSDP 后才能启用 CPU offload`);
      if (raylight.fsdp || raylight.cpu_offload)
        errors.push(`${MODEL_META[role].label}的 FSDP / CPU offload 在原生时间线 v1 暂未开放`);
      if (resolveExecutionBackend(diffusion) === "raylight" && diffusion.device !== "default")
        errors.push(`${MODEL_META[role].label}使用 RayLight 时标准执行设备必须为 default`);
    }
  }
  return errors;
}

export function SettingsPage({ settings, confirmedSettings = settings, resourcesOrigin = confirmedSettings.comfy_url, capabilities, gpus, models, rayLightRuntimeStatus = null, rayLightRecoveryPending = false, rayLightRecoveryDisabled = false, rayLightRecoveryBlockedReason = null, loadingModels, syncError = null, runtimeEditingDisabled = false, storageOperationsDisabled = false, overlay = false, theme = "dark", onThemeChange = () => undefined, onDraftChange = () => undefined, onSaved, onBeforeStorageChange = async () => undefined, onStorageOperationStarted = () => undefined, onStorageOperationAborted = () => undefined, onStorageOperationUncertain = async () => { throw new Error("无法核对数据库存储状态"); }, onStorageConfigurationChanged = () => undefined, onStorageSwitchCancelled = async () => undefined, onConnectionTestSucceeded = () => undefined, onConfirmRayLightRuntimeRecovery = async () => undefined, onRequestClose }: {
  settings: RuntimeSettings; confirmedSettings?: RuntimeSettings; capabilities: CapabilityReport; gpus: GPUResource[];
  resourcesOrigin?: string | null;
  models: ModelInventory;
  rayLightRuntimeStatus?: RayLightRuntimeStatus | null;
  rayLightRecoveryPending?: boolean;
  rayLightRecoveryDisabled?: boolean;
  rayLightRecoveryBlockedReason?: string | null;
  loadingModels: boolean;
  syncError?: string | null;
  runtimeEditingDisabled?: boolean;
  /** A stale live-page database identity makes every storage action unsafe until reload. */
  storageOperationsDisabled?: boolean;
  overlay?: boolean;
  theme?: UiTheme;
  onThemeChange?: (theme: UiTheme) => void;
  onDraftChange?: (settings: RuntimeSettings) => void;
  onRequestClose?: (restoreFocus?: boolean) => void;
  /** App-owned whole-document write followed by its authoritative runtime GET. */
  onSaved: (settings: RuntimeSettings) => Promise<RuntimeSettings>;
  /** App-owned timeline flush. Storage mutation must not start unless it resolves. */
  onBeforeStorageChange?: () => Promise<void>;
  /** Establishes a global fail-closed boundary before a storage request can begin. */
  onStorageOperationStarted?: () => void;
  /** Releases that boundary only when no storage request was sent or a 4xx rejected it. */
  onStorageOperationAborted?: () => void;
  /** Reconciles an ambiguous network/5xx result until GET /storage is authoritative. */
  onStorageOperationUncertain?: () => Promise<StorageConfiguration>;
  /** Notifies App so a pending database switch can freeze all ordinary writes. */
  onStorageConfigurationChanged?: (configuration: StorageConfiguration) => void;
  /** Drains database-scoped WALs after PUT active has unfrozen the backend. */
  onStorageSwitchCancelled?: () => Promise<void>;
  /** App decides whether this exact probed URL is the current authoritative endpoint. */
  onConnectionTestSucceeded?: (testedUrl: string) => void;
  /** App-owned explicit restart certificate followed by authoritative refresh. */
  onConfirmRayLightRuntimeRecovery?: () => Promise<void>;
}) {
  const [working, setWorking] = useState<RuntimeSettings>(() => structuredClone(settings));
  const [testing, setTesting] = useState(false);
  const [connectionProbe, setConnectionProbe] = useState<{
    url: string;
    result: ConnectionTestResult | null;
  } | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [residencyNotice, setResidencyNotice] = useState<string | null>(null);
  const [storageConfiguration, setStorageConfiguration] = useState<StorageConfiguration | null>(null);
  const [storageTarget, setStorageTarget] = useState("");
  const [comfyUrlVisible, setComfyUrlVisible] = useState(false);
  const [currentDatabaseVisible, setCurrentDatabaseVisible] = useState(false);
  const [storageTargetVisible, setStorageTargetVisible] = useState(false);
  const [storageLoading, setStorageLoading] = useState(true);
  const [storageOperation, setStorageOperation] = useState<"save" | "migrate" | null>(null);
  const [storageMessage, setStorageMessage] = useState<{
    kind: "success" | "error";
    text: string;
  } | null>(null);
  const editRevision = useRef(0);
  const workingRef = useRef(working);
  const confirmedRef = useRef(structuredClone(confirmedSettings));
  const hasLocalChanges = useRef(false);
  const onSavedRef = useRef(onSaved);
  const mountedRef = useRef(true);
  const connectionProbeRevision = useRef(0);
  const storageRequestRevision = useRef(0);
  const storageTargetEdited = useRef(false);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const connectionStatusRef = useRef<HTMLDivElement>(null);
  const rayLightRecoveryRequestRef = useRef(false);
  const overlayRef = useRef<HTMLDivElement>(null);
  const pageRef = useRef<HTMLElement>(null);
  const validationErrors = validateRuntimeSettingsForm(working);
  const runtimeConfigured = isConfiguredComfyUrl(working.comfy_url);
  const effectiveRuntimeEditingDisabled = runtimeEditingDisabled || storageOperation !== null ||
    storageConfiguration?.restart_required === true;
  const runtimeResourcesReady = !effectiveRuntimeEditingDisabled && runtimeConfigured && capabilities.connection === "online" &&
    resourcesOrigin === working.comfy_url;
  const awaitingCurrentEndpoint = runtimeConfigured && resourcesOrigin !== working.comfy_url;
  const currentRayLightRuntimeStatus = resourcesOrigin === working.comfy_url
    ? rayLightRuntimeStatus
    : null;
  const diffusionModels = [...new Set([...models.fl2va, ...models.ref2va])].sort();
  const rayLightFamilies = (["fl2va", "ref2va"] as const)
    .filter((role) => resolveExecutionBackend(working.models[role]) === "raylight")
    .map((role) => role === "fl2va" ? "FL2VA" : "Ref2VA");
  const storagePathError = validateDatabasePathInput(storageTarget);
  const cancellingPendingStorageSwitch = storageConfiguration?.restart_required === true &&
    storageTarget.trim() === storageConfiguration.active_database_path;

  onSavedRef.current = onSaved;
  workingRef.current = working;

  const requestClose = useCallback((restoreFocus = true) => {
    if (!onRequestClose) return;
    onRequestClose(restoreFocus);
  }, [onRequestClose]);

  const loadStorageConfiguration = useCallback(async (signal?: AbortSignal) => {
    const revision = ++storageRequestRevision.current;
    setStorageLoading(true);
    try {
      const configuration = await directorApi.getStorage(signal);
      if (signal?.aborted || storageRequestRevision.current !== revision) return;
      setStorageConfiguration(configuration);
      if (!storageTargetEdited.current) {
        setStorageTarget(
          configuration.restart_required
            ? configuration.active_database_path
            : configuration.source === "legacy"
              ? configuration.recommended_database_path
              : configuration.configured_database_path,
        );
      }
      setStorageMessage(null);
    } catch (reason) {
      if (signal?.aborted || storageRequestRevision.current !== revision) return;
      setStorageMessage({
        kind: "error",
        text: safeStorageError(reason, "无法读取数据库存储配置"),
      });
    } finally {
      if (!signal?.aborted && storageRequestRevision.current === revision) setStorageLoading(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void loadStorageConfiguration(controller.signal);
    return () => controller.abort();
  }, [loadStorageConfiguration]);

  useEffect(() => {
    if (
      !currentRayLightRuntimeStatus?.recovery_required &&
      message?.startsWith("RayLight 恢复失败：")
    ) setMessage(null);
  }, [currentRayLightRuntimeStatus?.recovery_required]);

  useEffect(() => {
    if (!overlay) return;
    closeButtonRef.current?.focus();
    const previousBodyOverflow = document.body.style.overflow;
    const previousOverscroll = document.documentElement.style.overscrollBehavior;
    document.body.style.overflow = "hidden";
    document.documentElement.style.overscrollBehavior = "none";
    return () => {
      document.body.style.overflow = previousBodyOverflow;
      document.documentElement.style.overscrollBehavior = previousOverscroll;
    };
  }, [overlay]);

  useEffect(() => {
    if (!overlay) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        requestClose(true);
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = Array.from(overlayRef.current?.querySelectorAll<HTMLElement>(
        'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [href], [tabindex]:not([tabindex="-1"])',
      ) ?? []).filter((element) =>
        element.tabIndex >= 0 && !element.matches(":disabled") &&
        !element.hidden && element.getAttribute("aria-hidden") !== "true");
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable.at(-1)!;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      } else if (!overlayRef.current?.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [overlay, requestClose]);

  useEffect(() => {
    if (!overlay) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (event.button !== 0 || !(event.target instanceof Element)) return;
      if (pageRef.current?.contains(event.target)) return;
      // The entry owns its toggle click. Excluding it avoids closing on
      // pointerdown and immediately reopening on the following click.
      if (event.target.closest("#system-settings-toggle")) return;
      requestClose(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer, true);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer, true);
  }, [overlay, requestClose]);

  useEffect(() => {
    confirmedRef.current = structuredClone(confirmedSettings);
    // Parent refreshes may arrive while an automatic write is still in flight.
    // An older confirmation must not overwrite a newer local draft; the exact
    // final confirmation clears the draft ownership.
    if (!hasLocalChanges.current || sameSettings(confirmedSettings, workingRef.current)) {
      connectionProbeRevision.current += 1;
      setConnectionProbe(null);
      setTesting(false);
      const confirmed = structuredClone(confirmedSettings);
      confirmedRef.current = confirmed;
      workingRef.current = structuredClone(confirmed);
      hasLocalChanges.current = false;
      setWorking(structuredClone(confirmed));
      setResidencyNotice(null);
    }
  }, [confirmedSettings]);

  useEffect(() => {
    // The App-owned draft can advance while this overlay is mounted (for
    // example through the global model quick controls). Adopt that desired
    // document without mistaking it for server authority.
    if (sameSettings(settings, workingRef.current)) return;
    workingRef.current = structuredClone(settings);
    hasLocalChanges.current = !sameSettings(settings, confirmedRef.current);
    setWorking(structuredClone(settings));
  }, [settings]);

  const change = (next: RuntimeSettings) => {
    if (effectiveRuntimeEditingDisabled) return;
    const revision = ++editRevision.current;
    const endpointChanged = next.comfy_url !== working.comfy_url;
    const nextSettings = endpointChanged
      ? {
          ...next,
          models: {
            ...next.models,
            fl2va: { ...next.models.fl2va, standard_lora_loader_override: null },
            ref2va: { ...next.models.ref2va, standard_lora_loader_override: null },
          },
        }
      : next;
    if (endpointChanged) {
      // A result belongs only to the exact URL snapshot that was probed.
      // Invalidating the request token prevents a slow response from making a
      // newly edited, untested address look online.
      connectionProbeRevision.current += 1;
      setConnectionProbe(null);
      setTesting(false);
    }
    workingRef.current = nextSettings;
    hasLocalChanges.current = !sameSettings(nextSettings, confirmedRef.current);
    setWorking(nextSettings);
    onDraftChange(structuredClone(nextSettings));
    setMessage(null);
    if (validateRuntimeSettingsForm(nextSettings).length > 0) return;
    const snapshot = structuredClone(nextSettings);
    void Promise.resolve(onSavedRef.current(snapshot)).catch((reason) => {
      if (editRevision.current !== revision || !mountedRef.current) return;
      setMessage(reason instanceof Error ? reason.message : "系统设置自动同步失败");
    });
  };
  const test = async () => {
    if (effectiveRuntimeEditingDisabled) return;
    const snapshot = working.comfy_url.trim();
    try {
      const url = new URL(snapshot);
      if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error();
    } catch {
      setMessage("请先填写有效的 ComfyUI HTTP/HTTPS 地址");
      return;
    }
    const requestRevision = ++connectionProbeRevision.current;
    setTesting(true); setMessage(null);
    setConnectionProbe({ url: snapshot, result: null });
    try {
      const result = await directorApi.testConnection(snapshot);
      if (connectionProbeRevision.current !== requestRevision) return;
      setConnectionProbe({ url: snapshot, result });
      if (result.ok) onConnectionTestSucceeded(snapshot);
    }
    catch (reason) {
      if (connectionProbeRevision.current !== requestRevision) return;
      const failure = reason instanceof Error ? reason.message : "连接失败";
      setConnectionProbe({ url: snapshot, result: { ok: false, message: failure } });
    }
    finally {
      if (connectionProbeRevision.current === requestRevision) setTesting(false);
    }
  };
  const recoverRayLightRuntime = async () => {
    if (
      !currentRayLightRuntimeStatus?.recovery_required ||
      rayLightRecoveryPending ||
      rayLightRecoveryRequestRef.current
    ) return;
    const recorded = currentRayLightRuntimeStatus.runtime_gpu_indexes.join(", ");
    const invalid = currentRayLightRuntimeStatus.invalid_gpu_indexes.join(", ");
    const confirmed = window.confirm(
      "仅当旧 ComfyUI 进程确实已经重启时才能清理此记录。测试连接成功或当前队列为空，都不能证明旧进程已经退出；若实际未重启，旧 RayLight actor 仍可能运行并占用 GPU。\n\n" +
      `旧 runtime 逻辑 GPU：${recorded}\n当前不可见逻辑 GPU：${invalid}\n\n` +
      "确认旧 ComfyUI 进程已重启，并清除这个 endpoint 的旧 RayLight 运行状态吗？",
    );
    if (!confirmed) return;
    rayLightRecoveryRequestRef.current = true;
    setMessage(null);
    try {
      await onConfirmRayLightRuntimeRecovery();
      window.requestAnimationFrame(() => connectionStatusRef.current?.focus());
    } catch (reason) {
      setMessage(reason instanceof Error ? `RayLight 恢复失败：${reason.message}` : "RayLight 恢复失败");
    } finally {
      rayLightRecoveryRequestRef.current = false;
    }
  };
  const flushTimelineBeforeStorageChange = async (): Promise<boolean> => {
    try {
      await onBeforeStorageChange();
      return true;
    } catch (reason) {
      const detail = reason instanceof ApiError && reason.message ? `：${reason.message}` : "";
      setStorageMessage({
        kind: "error",
        text: `当前时间线未能同步，未执行数据库操作${detail}`,
      });
      return false;
    }
  };
  const saveStoragePath = async () => {
    if (storageOperationsDisabled) return;
    const target = storageTarget.trim();
    const validation = validateDatabasePathInput(target);
    if (validation) {
      setStorageMessage({ kind: "error", text: validation });
      return;
    }
    const confirmation = cancellingPendingStorageSwitch
      ? `将取消待重启的数据库切换，并继续使用当前数据库。后端解除写入冻结后，会继续同步本页保留的修改。\n\n当前数据库：${target}\n\n确认取消切换吗？`
      : `仅保存数据库路径不会复制当前数据。目标必须已经是有效的 Director 数据库，否则后端会拒绝。保存成功后，当前进程将继续读取旧库并停止修改，必须重启 Director 才会切换。\n\n目标：${target}\n\n确认仅保存此路径吗？`;
    if (!window.confirm(confirmation)) return;
    setStorageOperation("save");
    setStorageMessage(null);
    onStorageOperationStarted();
    try {
      // PUT active is the recovery escape hatch for an ambiguous/lost storage
      // response. It must run before ordinary WAL flushes because the backend
      // intentionally rejects those writes while a switch is pending.
      if (!cancellingPendingStorageSwitch && !await flushTimelineBeforeStorageChange()) {
        onStorageOperationAborted();
        return;
      }
      const configuration = await directorApi.updateStorage(target);
      setStorageConfiguration(configuration);
      onStorageConfigurationChanged(configuration);
      storageTargetEdited.current = false;
      setStorageTarget(configuration.restart_required
        ? configuration.active_database_path
        : configuration.configured_database_path);
      if (cancellingPendingStorageSwitch && !configuration.restart_required) {
        try {
          await onStorageSwitchCancelled();
          setStorageMessage({
            kind: "success",
            text: "已取消数据库切换，继续使用当前数据库；本页待同步修改已恢复。",
          });
        } catch {
          setStorageMessage({
            kind: "error",
            text: "已取消数据库切换，但本页待同步修改暂未恢复；数据仍保留在本地并会自动重试。",
          });
        }
        return;
      }
      setStorageMessage({
        kind: "success",
        text: configuration.restart_required
          ? "数据库路径已保存。当前进程保持读取原数据库并停止修改；请重启 Director 后切换。"
          : "数据库路径已确认，当前进程已经使用该数据库。",
      });
    } catch (reason) {
      if (!storageRequestWasDefinitivelyRejected(reason)) {
        if (mountedRef.current) {
          setStorageMessage({
            kind: "error",
            text: "数据库操作响应尚未确认，正在自动核对服务器状态；确认前当前页面保持锁定。",
          });
        }
        try {
          const configuration = await onStorageOperationUncertain();
          if (mountedRef.current) {
            setStorageConfiguration(configuration);
            storageTargetEdited.current = false;
            setStorageTarget(configuration.restart_required
              ? configuration.active_database_path
              : configuration.configured_database_path);
          }
          if (cancellingPendingStorageSwitch && !configuration.restart_required) {
            try {
              await onStorageSwitchCancelled();
              if (mountedRef.current) setStorageMessage({
                kind: "success",
                text: "已确认数据库切换取消成功；本页待同步修改已恢复。",
              });
            } catch {
              if (mountedRef.current) setStorageMessage({
                kind: "error",
                text: "已确认数据库切换取消成功，但本页待同步修改暂未恢复；数据仍保留在本地并会自动重试。",
              });
            }
          } else if (mountedRef.current) {
            setStorageMessage({
              kind: configuration.restart_required ? "error" : "success",
              text: configuration.restart_required
                ? "服务器确认数据库切换正在等待重启。当前页面保持锁定；如需继续使用当前库，请先取消切换并恢复待同步修改。"
                : "服务器已确认没有待重启的数据库切换，当前页面已恢复修改。",
            });
          }
        } catch {
          if (mountedRef.current) setStorageMessage({
            kind: "error",
            text: "暂时无法确认数据库操作结果；当前页面保持锁定并会继续自动核对，请勿重启或继续修改。",
          });
        }
        return;
      }
      onStorageOperationAborted();
      setStorageMessage({
        kind: "error",
        text: safeStorageError(reason, "保存数据库路径失败"),
      });
    } finally {
      if (mountedRef.current) setStorageOperation(null);
    }
  };
  const migrateStorage = async () => {
    if (storageOperationsDisabled) return;
    const target = storageTarget.trim();
    const validation = validateDatabasePathInput(target);
    if (validation) {
      setStorageMessage({ kind: "error", text: validation });
      return;
    }
    if (!window.confirm(
      `将复制并校验当前 Director 数据库，然后保存新路径。迁移成功后，当前进程将继续读取旧库并停止修改，必须重启 Director 才会切换。\n\n目标：${target}\n\n确认迁移吗？`,
    )) return;
    setStorageOperation("migrate");
    setStorageMessage(null);
    onStorageOperationStarted();
    try {
      if (!await flushTimelineBeforeStorageChange()) {
        onStorageOperationAborted();
        return;
      }
      const result = await directorApi.migrateStorage(target);
      setStorageConfiguration(result);
      onStorageConfigurationChanged(result);
      storageTargetEdited.current = false;
      setStorageTarget(result.restart_required
        ? result.active_database_path
        : result.configured_database_path);
      setStorageMessage({
        kind: "success",
        text: `数据库已从 ${result.migrated_from} 迁移到 ${result.migrated_to}。当前进程保持读取原数据库并停止修改；请重启 Director 后切换。`,
      });
    } catch (reason) {
      if (!storageRequestWasDefinitivelyRejected(reason)) {
        if (mountedRef.current) setStorageMessage({
          kind: "error",
          text: "数据库迁移响应尚未确认，正在自动核对服务器状态；确认前当前页面保持锁定。",
        });
        try {
          const configuration = await onStorageOperationUncertain();
          if (mountedRef.current) {
            setStorageConfiguration(configuration);
            storageTargetEdited.current = false;
            setStorageTarget(configuration.restart_required
              ? configuration.active_database_path
              : configuration.configured_database_path);
            setStorageMessage({
              kind: configuration.restart_required ? "error" : "success",
              text: configuration.restart_required
                ? "服务器确认数据库迁移或路径切换正在等待重启。当前页面保持锁定；如需继续使用当前库，请先取消切换并恢复待同步修改。"
                : "服务器已确认没有待重启的数据库切换，当前页面已恢复修改。",
            });
          }
        } catch {
          if (mountedRef.current) setStorageMessage({
            kind: "error",
            text: "暂时无法确认数据库迁移结果；当前页面保持锁定并会继续自动核对，请勿重启或继续修改。",
          });
        }
        return;
      }
      onStorageOperationAborted();
      setStorageMessage({
        kind: "error",
        text: safeStorageError(reason, "数据库迁移失败，当前数据库未切换"),
      });
    } finally {
      if (mountedRef.current) setStorageOperation(null);
    }
  };
  const setModel = (role: ModelRole, key: "filename" | "device", value: string) => {
    if (key === "device" && !MODEL_META[role].allowCpu && value === "cpu") return;
    const current = working.models[role];
    const next = {
      ...current,
      [key]: value,
      ...(key === "filename" && (role === "fl2va" || role === "ref2va")
        ? { standard_lora_loader_override: null }
        : {}),
    };
    change({ ...working, models: { ...working.models, [role]: next } });
  };
  const setDiffusion = (
    role: "fl2va" | "ref2va",
    patch: Partial<DiffusionModelBinding>,
  ) => {
    const currentBinding = working.models[role];
    const selectedArtifactChanged =
      ("lora_name" in patch && patch.lora_name !== currentBinding.lora_name) ||
      ("filename" in patch && patch.filename !== currentBinding.filename);
    const nextBinding = {
      ...currentBinding,
      ...patch,
      ...(selectedArtifactChanged ? { standard_lora_loader_override: null } : {}),
    };
    const nextResidencyPolicy = rayLightResidencyPolicyAfterBindingChange(
      working,
      role,
      nextBinding,
    );
    if (nextResidencyPolicy !== working.raylight_residency_policy) {
      const label = role === "fl2va" ? "FL2VA" : "Ref2VA";
      setResidencyNotice(
        nextResidencyPolicy === "release_after_sampling"
          ? `${label} 使用 RayLight，但会在每次采样后释放。`
          : `${label} 已切换为 RayLight，已自动选择“按运行配置常驻”。同一配置直接复用；切换模型族、模型、LoRA、GPU 池、拓扑或 Standard 时由 Director 先安全释放旧池。`,
      );
    }
    change({
      ...working,
      raylight_residency_policy: nextResidencyPolicy,
      models: {
        ...working.models,
        [role]: nextBinding,
      },
    });
  };
  const setRayLight = (
    role: "fl2va" | "ref2va",
    raylight: RayLightProfile,
  ) => {
    const normalizedRayLight = {
      ...raylight,
      cpu_offload: raylight.fsdp ? raylight.cpu_offload : false,
    };
    const resolvesToRayLight = normalizedRayLight.gpu_select.length >= 2;
    setDiffusion(role, {
      backend: "auto",
      raylight: normalizedRayLight,
      ...(resolvesToRayLight
        ? {
            device: "default" as const,
            lora_low_vram: false,
            lora_loader: "auto" as const,
            standard_lora_loader_override: null,
          }
        : {}),
    });
  };
  const toggleRayLightGpu = (
    role: "fl2va" | "ref2va",
    index: number,
    selected: boolean,
  ) => {
    const current = working.models[role].raylight;
    const nextPool = selected
      ? [...new Set([...current.gpu_select, index])].sort((left, right) => left - right)
      : current.gpu_select.filter((candidate) => candidate !== index);
    if (!nextPool.length || nextPool.length > 8) return;
    setRayLight(role, {
      ...current,
      gpu_select: nextPool,
      // Pool changes are atomic with a known-valid topology preset.
      ulysses_degree: nextPool.length,
      ring_degree: 1,
      cfg_degree: 1,
      dp_degree: 1,
    });
  };
  const setRayLightAxis = (
    role: "fl2va" | "ref2va",
    axis: "ulysses_degree" | "ring_degree",
    value: number,
  ) => {
    const current = working.models[role].raylight;
    const other = current.gpu_select.length / value;
    if (!Number.isInteger(other)) return;
    setRayLight(role, {
      ...current,
      [axis]: value,
      [axis === "ulysses_degree" ? "ring_degree" : "ulysses_degree"]: other,
      cfg_degree: 1,
      dp_degree: 1,
    });
  };

  const content = (
    <section ref={pageRef} className={`settings-page ${overlay ? "settings-page--overlay" : ""}`}>
      <header className="settings-hero"><div><h1 id={overlay ? "system-settings-title" : undefined}>系统设置</h1><p>共享连接和模型资源。有效修改会自动应用，并刷新该实例的模型与 GPU。</p></div><div className="settings-hero__actions">{overlay && <button ref={closeButtonRef} type="button" className="icon-button settings-overlay__close" aria-label="关闭系统设置" onClick={() => requestClose(true)}>×</button>}</div></header>
      {hasLocalChanges.current && validationErrors.length > 0 && <div className="notice" role="alert">{validationErrors.join("；")}</div>}
      {syncError && <div className="notice" role="alert">服务器拒绝当前系统设置：{syncError}。请修改；有效修改后自动应用。</div>}
      {effectiveRuntimeEditingDisabled && <div className="notice" role="status">{storageOperationsDisabled
          ? "数据库身份与时间线尚未确认，或本页身份已过期；恢复或刷新整个页面前设置保持锁定。"
        : storageConfiguration?.restart_required
          ? "数据库切换正在等待重启；运行设置已锁定。可在“数据存储”中把路径改回当前数据库以取消切换。"
          : "运行设置暂时锁定。"}</div>}
      {message && <div className="notice" role="alert">{message}</div>}
      <div className="settings-layout">
        <div className="settings-main">
          <Panel eyebrow="连接" title="ComfyUI" description="浏览器仅连接导演台后端，由后端代理 ComfyUI。">
            <div ref={connectionStatusRef} className="connection-card" tabIndex={-1}><StatusDot state={connectionProbe ? connectionProbe.result === null ? "checking" : connectionProbe.result.ok ? "online" : "offline" : awaitingCurrentEndpoint ? "checking" : capabilities.connection} /><div><strong>{connectionProbe ? connectionProbe.result === null ? "正在测试连接" : connectionProbe.result.ok ? "当前填写地址可连接" : "当前填写地址连接失败" : awaitingCurrentEndpoint ? "等待当前地址确认" : capabilities.connection === "offline" ? "ComfyUI 离线" : !runtimeConfigured ? "ComfyUI 尚未配置" : capabilities.connection === "online" ? "ComfyUI 在线" : "等待检测"}</strong><small>{connectionProbe ? connectionProbe.result === null ? "正在等待 ComfyUI 响应" : connectionProbe.result.latency_ms === undefined ? connectionProbe.result.message : `${connectionProbe.result.message} · 响应 ${connectionProbe.result.latency_ms} ms` : awaitingCurrentEndpoint ? working.comfy_url : capabilities.connection === "offline" ? capabilities.message || "无法读取运行环境" : !runtimeConfigured ? "填写有效地址后自动启用运行资源" : capabilities.latency_ms === undefined ? "尚未读取延迟" : `响应 ${capabilities.latency_ms} ms`}</small></div><button type="button" className="button button--ghost" onClick={() => void test()} disabled={testing || effectiveRuntimeEditingDisabled}>{testing ? <><Spinner />测试中…</> : "测试连接"}</button></div>
            {currentRayLightRuntimeStatus?.recovery_required && <div className="raylight-recovery-alert" role="alert" aria-labelledby="raylight-recovery-title" aria-describedby="raylight-recovery-description">
              <strong id="raylight-recovery-title">旧 RayLight 运行状态引用了当前不可见 GPU</strong>
              <div id="raylight-recovery-description" className="raylight-recovery-alert__details">
                <p>旧 runtime 逻辑 GPU：<code>{currentRayLightRuntimeStatus.runtime_gpu_indexes.join(", ")}</code></p>
                <p>当前可见逻辑 GPU：<code>{currentRayLightRuntimeStatus.available_gpu_indexes.join(", ") || "无"}</code></p>
                <p>当前不可见逻辑 GPU：<code>{currentRayLightRuntimeStatus.invalid_gpu_indexes.join(", ")}</code></p>
                <small>GPU_SELECT 使用 ComfyUI 进程内的逻辑编号，不是物理显卡编号。只有确认旧 ComfyUI 进程确实已经重启后才能恢复；测试连接成功或当前队列为空都不能证明旧进程已退出。</small>
                {rayLightRecoveryDisabled && !rayLightRecoveryPending && rayLightRecoveryBlockedReason && <small>暂不可恢复：{rayLightRecoveryBlockedReason}</small>}
              </div>
              <button
                type="button"
                className="button button--danger"
                aria-label="确认 ComfyUI 已重启并恢复 RayLight"
                aria-describedby="raylight-recovery-description"
                aria-busy={rayLightRecoveryPending || undefined}
                disabled={rayLightRecoveryPending || rayLightRecoveryDisabled}
                onClick={() => void recoverRayLightRuntime()}
              >
                {rayLightRecoveryPending ? <><Spinner />恢复中…</> : "确认 ComfyUI 已重启并恢复 RayLight"}
              </button>
            </div>}
            <div className="field-grid field-grid--two">
              <div className="field">
                <label className="field__label" htmlFor="comfy-url">ComfyUI 地址</label>
                <div className="sensitive-value">
                  <input id="comfy-url" type={comfyUrlVisible ? "url" : "password"} required disabled={effectiveRuntimeEditingDisabled} autoComplete="off" value={working.comfy_url} placeholder="http://comfyui-host:8188" onChange={(event) => change({ ...working, comfy_url: event.target.value })} />
                  <VisibilityToggle label="ComfyUI 地址显示状态" visible={comfyUrlVisible} onToggle={() => setComfyUrlVisible((current) => !current)} />
                </div>
              </div>
              <Field label="客户端 ID"><input required disabled={effectiveRuntimeEditingDisabled} maxLength={128} pattern="[A-Za-z0-9._:-]+" value={working.client_id} onChange={(event) => change({ ...working, client_id: event.target.value })} /></Field>
            </div>
          </Panel>
          <Panel eyebrow="数据" title="数据存储" description="数据库路径独立于 ComfyUI 运行设置；路径保存或迁移后都不会热切换当前进程。" action={storageLoading ? <Spinner label="读取数据存储配置" /> : undefined}>
            {storageConfiguration && <div className="storage-current" aria-label="当前数据库存储配置">
              <div className="storage-current__path">
                <span>当前数据库</span>
                <div className="sensitive-value sensitive-value--code">
                  <code title={currentDatabaseVisible ? storageConfiguration.active_database_path : undefined}>{currentDatabaseVisible ? storageConfiguration.active_database_path : HIDDEN_PATH_VALUE}</code>
                  <VisibilityToggle label="当前数据库路径显示状态" visible={currentDatabaseVisible} onToggle={() => setCurrentDatabaseVisible((current) => !current)} />
                </div>
              </div>
              <div><span>配置来源</span><strong>{STORAGE_SOURCE_LABEL[storageConfiguration.source]}</strong></div>
              {storageConfiguration.configured_database_path !== storageConfiguration.active_database_path && <div className="storage-current__path"><span>重启后路径</span><code title={currentDatabaseVisible ? storageConfiguration.configured_database_path : undefined}>{currentDatabaseVisible ? storageConfiguration.configured_database_path : HIDDEN_PATH_VALUE}</code></div>}
              <div><span>切换状态</span><strong>{storageConfiguration.restart_required ? "等待重启" : "当前已生效"}</strong></div>
            </div>}
            {!storageConfiguration && !storageLoading && <div className="notice notice--error" role="alert">无法确认当前数据库，存储操作保持禁用。</div>}
            <div className="field">
              <label className="field__label" htmlFor="database-target-path">数据库目标路径</label>
              <div className="sensitive-value">
                <input
                  id="database-target-path"
                  type={storageTargetVisible ? "text" : "password"}
                  disabled={storageOperationsDisabled || storageOperation !== null}
                  spellCheck={false}
                  autoComplete="off"
                  value={storageTarget}
                  placeholder="/path/to/director.sqlite3 或 ~/director.sqlite3"
                  aria-invalid={storageTargetEdited.current && Boolean(storagePathError) || undefined}
                  aria-describedby={storageTargetEdited.current && storagePathError ? "database-path-error database-path-hint" : "database-path-hint"}
                  onChange={(event) => {
                    storageTargetEdited.current = true;
                    setStorageTarget(event.target.value);
                    setStorageMessage(null);
                  }}
                />
                <VisibilityToggle label="数据库目标路径显示状态" visible={storageTargetVisible} onToggle={() => setStorageTargetVisible((current) => !current)} />
              </div>
              {storageTargetEdited.current && storagePathError && <small id="database-path-error" className="inline-error">{storagePathError}</small>}
              <small id="database-path-hint">只接受绝对路径或以 ~/ 开头的路径。推荐迁移当前数据库，避免重启后进入空库。</small>
            </div>
            <div className="storage-actions" role="group" aria-label="数据库存储操作">
              <button type="button" className="button button--ghost" disabled={storageOperationsDisabled || storageLoading || storageOperation !== null || !storageConfiguration || Boolean(storagePathError)} onClick={() => void saveStoragePath()}>{storageOperation === "save" ? <><Spinner />保存中…</> : cancellingPendingStorageSwitch ? "取消切换并继续使用当前库" : "保存路径（重启后切换）"}</button>
              <button type="button" className="button button--primary" disabled={storageOperationsDisabled || storageLoading || storageOperation !== null || !storageConfiguration || Boolean(storagePathError)} onClick={() => void migrateStorage()}>{storageOperation === "migrate" ? <><Spinner />迁移中…</> : "迁移当前数据库并切换"}</button>
            </div>
            <small className="storage-warning">路径保存或迁移成功后，当前进程保持读取旧库并停止修改；请重启 Director 后继续。</small>
            {storageMessage && <div className={`notice ${storageMessage.kind === "error" ? "notice--error" : ""}`} role={storageMessage.kind === "error" ? "alert" : "status"}>{storageMessage.text}</div>}
          </Panel>
          <Panel eyebrow="界面" title="界面主题" description="主题仅作用于当前浏览器，不写入 Director 或 ComfyUI 设置。">
            <fieldset className="theme-options" aria-label="界面主题">
              <label className={`theme-option ${theme === "light" ? "is-active" : ""}`}><input type="radio" name="ui-theme" value="light" checked={theme === "light"} onChange={() => onThemeChange("light")} /><span className="theme-option__swatch theme-option__swatch--light" aria-hidden="true" /><strong>暖色浅色</strong><small>纸张暖白与橙棕强调</small></label>
              <label className={`theme-option ${theme === "dark" ? "is-active" : ""}`}><input type="radio" name="ui-theme" value="dark" checked={theme === "dark"} onChange={() => onThemeChange("dark")} /><span className="theme-option__swatch theme-option__swatch--dark" aria-hidden="true" /><strong>深色</strong><small>沿用导演台暗色工作区</small></label>
            </fieldset>
          </Panel>
          <Panel eyebrow="运行" title="模型复用策略" description="RayLight 默认按完整配置键常驻；兼容任务直接复用，不兼容任务由 Director 显式安全切换。">
            <Field label="共享策略" hint="Standard 原生子图不主动卸载模型，由 ComfyUI 缓存按显存压力管理。">
              <div className="fixed-runtime-value" aria-label="片段间显存策略"><strong>Standard 稳定 loader 复用</strong><small>原生分段子任务固定策略</small></div>
            </Field>
            <Field label="RayLight 驻留" hint={working.raylight_residency_policy === "keep_until_switch"
              ? "family、模型、LoRA、GPU 池和拓扑完全一致时复用 CUDA 权重；任一项变化会先运行 RayKill 安全屏障并重建 Ray 池，切到 Standard 也一样。无需重启 ComfyUI。"
              : "每次 Ray 采样结束后卸载 worker CUDA 权重，下一次任务必须重新加载。"}>
              <select
                aria-label="RayLight 显存驻留策略"
                disabled={effectiveRuntimeEditingDisabled}
                value={working.raylight_residency_policy}
                onChange={(event) => {
                  const policy = event.target.value as RuntimeSettings["raylight_residency_policy"];
                  setResidencyNotice(
                    policy === "release_after_sampling" && rayLightFamilies.length
                      ? `已明确选择任务后释放：${rayLightFamilies.join("、")} 每次任务结束都会卸载 RayLight 模型，下次任务将重新加载。`
                      : null,
                  );
                  change({ ...working, raylight_residency_policy: policy });
                }}
              >
                <option value="release_after_sampling">任务后释放（后续需重载）</option>
                <option value="keep_until_switch">按运行配置常驻，切换时安全释放（推荐）</option>
              </select>
            </Field>
            {residencyNotice && <div className="notice" role="status">{residencyNotice}</div>}
            {!residencyNotice && working.raylight_residency_policy === "release_after_sampling" && rayLightFamilies.length > 0 && <div className="notice" role="status">当前 {rayLightFamilies.join("、")} 已使用 RayLight，但设置为任务后释放：每次任务完成都会卸载模型，下次任务需要重新加载。</div>}
          </Panel>
          <Panel eyebrow="模型" title="模型与运行设备" description="填写有效 ComfyUI 地址后自动读取资源；模型和设备修改也会自动应用。VAE 只允许 default 或 GPU。" action={loadingModels ? <Spinner label="读取模型" /> : undefined}>
            {!runtimeConfigured && <div className="notice">填写有效 ComfyUI 地址后，模型和运行设备选项会自动启用。</div>}
            <div className="model-bindings">
              {(Object.keys(MODEL_META) as ModelRole[]).map((role) => {
                const meta = MODEL_META[role]; const binding = working.models[role];
                const discoveredDevices: DeviceTarget[] = gpus.filter((gpu) => gpu.visible).map((gpu) => `gpu:${gpu.index}` as const);
                const devices: DeviceTarget[] = [...new Set(["default" as const, ...(meta.allowCpu ? ["cpu" as const] : []), ...discoveredDevices, binding.device])];
                const availableModels = role === "fl2va" || role === "ref2va" ? diffusionModels : models[role];
                // An authoritative inventory lists exactly what the endpoint
                // holds. A stale configured name stays visible only while the
                // inventory is not authoritative yet (offline/loading), never
                // as a phantom entry next to the real list.
                const inventoryAuthoritative = runtimeResourcesReady && !loadingModels;
                const filenameMissing = Boolean(binding.filename) && !availableModels.includes(binding.filename);
                const filenames = filenameMissing && !inventoryAuthoritative ? [binding.filename, ...availableModels] : availableModels;
                const diffusionRole: "fl2va" | "ref2va" | null = role === "fl2va" || role === "ref2va" ? role : null;
                const diffusion = diffusionRole ? binding as DiffusionModelBinding : null;
                const loraMissing = Boolean(diffusion?.lora_name) && !models.loras.includes(diffusion!.lora_name as string);
                const loras = loraMissing && !inventoryAuthoritative && diffusion?.lora_name
                  ? [diffusion.lora_name, ...models.loras]
                  : models.loras;
                const resolvedBackend = diffusion ? resolveExecutionBackend(diffusion) : null;
                const backendCapability = resolvedBackend
                  ? capabilities.execution_backends?.[resolvedBackend]
                  : undefined;
                const rayGpuIndexes = diffusion
                  ? [...new Set([
                      ...gpus.filter((gpu) => gpu.visible).map((gpu) => gpu.index),
                      ...diffusion.raylight.gpu_select,
                    ])].sort((left, right) => left - right)
                  : [];
                const degreeOptions = diffusion
                  ? Array.from(
                      { length: Math.min(8, diffusion.raylight.gpu_select.length) },
                      (_, index) => index + 1,
                    ).filter((degree) => diffusion.raylight.gpu_select.length % degree === 0)
                  : [];
                return <div className="model-binding-group" key={role}>
                  <div className={`model-row ${diffusion ? "model-row--diffusion" : ""}`}>
                    <div className="model-row__name"><strong>{meta.label}</strong><small>{meta.description}</small></div>
                    <Field label="模型文件"><select required disabled={!runtimeResourcesReady} aria-label={`${meta.label}模型`} value={inventoryAuthoritative && filenameMissing ? "" : binding.filename} onChange={(event) => setModel(role, "filename", event.target.value)}><option value="">请选择模型</option>{filenames.map((filename) => <option key={filename} value={filename}>{filename}</option>)}</select>{inventoryAuthoritative && filenameMissing && <small className="inline-error">已配置的文件不在当前 ComfyUI 模型清单中：{binding.filename}</small>}</Field>
                    <Field label={diffusion ? "标准执行设备" : "运行设备"} hint={diffusion && resolvedBackend === "raylight" ? "RayLight 固定为 default；切回 Standard 后可重新选择逻辑设备。" : diffusion ? "标准原生节点使用此 ComfyUI 逻辑设备。" : undefined}><select required disabled={!runtimeResourcesReady || resolvedBackend === "raylight"} aria-label={`${meta.label}设备`} value={binding.device} onChange={(event) => setModel(role, "device", event.target.value)}>{devices.map((device) => <option key={device} value={device}>{device === "default" ? "default（ComfyUI 自动）" : device}</option>)}</select></Field>
                  </div>
                  {diffusion && diffusionRole && <section className="model-execution-panel" aria-label={`${meta.label}执行拓扑`}>
                    <header>
                      <div><strong>自动执行</strong></div>
                      <span className={`execution-badge ${backendCapability?.available === true ? "is-ready" : "is-unavailable"}`}>{resolvedBackend === "raylight" ? `RAYLIGHT · ${diffusion.raylight.gpu_select.length} 卡` : `STANDARD · ${binding.device}`}</span>
                    </header>
                    <p>{resolvedBackend === "standard" ? `GPU 池只有 1 张卡，自动使用标准原生子图；gpu:N 是 ComfyUI 进程内逻辑编号。` : `GPU 池有 ${diffusion.raylight.gpu_select.length} 张卡，自动使用 RayLight；逻辑 GPU ${diffusion.raylight.gpu_select.join(", ")}，U${diffusion.raylight.ulysses_degree} × R${diffusion.raylight.ring_degree} × 条件1 × 数据1。`}</p>
                    {backendCapability?.available !== true && <p className="inline-error">当前 ComfyUI 的 {resolvedBackend === "raylight" ? "RayLight" : "标准"}后端不可用：{backendCapability?.missing_nodes.join("、") || "尚未报告执行能力"}</p>}
                    <div className="raylight-topology">
                      <fieldset disabled={!runtimeResourcesReady}>
                        <legend>RayLight 逻辑 GPU 池</legend>
                        <div className="raylight-gpu-pool">
                          {rayGpuIndexes.map((index) => <label key={index} className={diffusion.raylight.gpu_select.includes(index) ? "is-selected" : ""}><input type="checkbox" aria-label={`${meta.label} RayLight 逻辑 GPU ${index}`} checked={diffusion.raylight.gpu_select.includes(index)} onChange={(event) => toggleRayLightGpu(diffusionRole, index, event.target.checked)} /><span>GPU {index}</span><small>逻辑编号</small></label>)}
                        </div>
                      </fieldset>
                      <Field label="Ulysses degree"><select aria-label={`${meta.label} RayLight Ulysses degree`} disabled={!runtimeResourcesReady || resolvedBackend !== "raylight"} value={diffusion.raylight.ulysses_degree} onChange={(event) => setRayLightAxis(diffusionRole, "ulysses_degree", Number(event.target.value))}>{degreeOptions.map((degree) => <option key={degree} value={degree}>{degree}</option>)}</select></Field>
                      <Field label="Ring degree"><select aria-label={`${meta.label} RayLight Ring degree`} disabled={!runtimeResourcesReady || resolvedBackend !== "raylight"} value={diffusion.raylight.ring_degree} onChange={(event) => setRayLightAxis(diffusionRole, "ring_degree", Number(event.target.value))}>{degreeOptions.map((degree) => <option key={degree} value={degree}>{degree}</option>)}</select></Field>
                      <div className="fixed-runtime-value fixed-runtime-value--topology"><span>条件 / 数据并行</span><strong>1 × 1</strong><small>当前拓扑固定</small></div>
                      <label className="check-field"><input type="checkbox" aria-label={`${meta.label} RayLight FSDP`} disabled checked={false} /><span><strong>FSDP</strong><small>v1 暂停开放，等待显存清理实机验证</small></span></label>
                      <label className="check-field"><input type="checkbox" aria-label={`${meta.label} RayLight CPU offload`} disabled checked={false} /><span><strong>CPU offload</strong><small>随 FSDP 暂停开放</small></span></label>
                    </div>
                    <small className="raylight-contract-note">1 卡自动走 Standard；2–8 卡自动走 RayLight。拓扑必须满足 Ulysses × Ring × 条件并行 × 数据并行 = 逻辑 GPU 池大小；修改 GPU 池会原子重置为 U=池大小、Ring=1。</small>
                  </section>}
                  {diffusion && diffusionRole && <div className="lora-row">
                    <div className="model-row__name"><strong>共享 LoRA</strong><small>{diffusion.lora_name ? "应用到该模型族的所有任务模式" : "不加载 LoRA"}</small></div>
                    <Field label="LoRA 文件"><select disabled={!runtimeResourcesReady} aria-label={`${meta.label} LoRA`} value={inventoryAuthoritative && loraMissing ? "" : (diffusion.lora_name ?? "")} onChange={(event) => setDiffusion(diffusionRole!, { lora_name: event.target.value || null })}><option value="">不使用 LoRA</option>{loras.map((filename) => <option key={filename} value={filename}>{filename}</option>)}</select>{inventoryAuthoritative && loraMissing && <small className="inline-error">已配置的 LoRA 不在当前 ComfyUI 清单中：{diffusion.lora_name}</small>}</Field>
                    <Field label="模型强度"><DeferredNumberInput aria-label={`${meta.label} LoRA 强度`} min="-10" max="10" step="0.01" disabled={!runtimeResourcesReady || !diffusion.lora_name} value={diffusion.lora_strength} onValueCommit={(value) => setDiffusion(diffusionRole!, { lora_strength: value })} /></Field>
                    {resolvedBackend === "raylight" ? <div className="fixed-runtime-value" aria-label={`${meta.label} LoRA 加载状态`}><span>加载方式</span><strong>RayLoraLoader</strong><small>RayLight 固定使用；切回 Standard 后重新自动探测</small></div> : <Field label="Standard 加载器" hint="默认由 ComfyUI safetensors metadata 自动识别；只有识别失败时才需要显式选择。"><select aria-label={`${meta.label} Standard LoRA 加载器`} disabled={!runtimeResourcesReady || !diffusion.lora_name} value={diffusion.standard_lora_loader_override?.loader ?? ""} onChange={(event) => {
                      const loader = event.target.value as StandardLoraLoader | "";
                      setDiffusion(diffusionRole, {
                        standard_lora_loader_override: loader && diffusion.lora_name
                          ? {
                              loader,
                              lora_name: diffusion.lora_name,
                              model_filename: diffusion.filename,
                              comfy_origin: working.comfy_url,
                            }
                          : null,
                      });
                    }}><option value="">自动探测（推荐）</option>{(Object.entries(STANDARD_LORA_LOADER_LABELS) as [StandardLoraLoader, string][]).map(([loader, label]) => <option value={loader} key={loader}>{label}</option>)}</select></Field>}
                  </div>}
                </div>;
              })}
            </div>
          </Panel>
        </div>
        <aside className="settings-aside">
          <Panel eyebrow="资源" title="GPU 状态"><div className="gpu-list">{gpus.length === 0 && <p className="muted">{runtimeConfigured ? "尚未读取 GPU 数据" : "填写有效 ComfyUI 地址后读取 GPU 状态"}</p>}{gpus.map((gpu) => { const used = gpu.vram_total ? Math.round((1 - gpu.vram_free / gpu.vram_total) * 100) : 0; return <div className="gpu-card" key={gpu.index}><header><span>GPU {gpu.index}</span><small>ComfyUI 逻辑编号</small></header><strong>{gpu.name}</strong><div className="progress"><span style={{ width: `${used}%` }} /></div><footer><span>显存 {used}%</span><span>{formatBytes(gpu.vram_free)} 可用</span></footer></div>;})}</div></Panel>
          <Panel eyebrow="能力" title="模型族就绪情况"><ul className="capability-list">{TIMELINE_MODE_ORDER.map((mode) => { const familyAvailable = capabilities.native_timeline?.supported === true && capabilities.native_timeline.modes.includes(mode); const ready = runtimeResourcesReady && familyAvailable; return <li key={mode}><span className={ready ? "capability-ok" : "capability-missing"}>{ready ? "✓" : "!"}</span><span><strong>{TIMELINE_MODE_META[mode].shortLabel}</strong><small>{TIMELINE_MODE_META[mode].label}</small></span><em>{ready ? "可用" : "未就绪"}</em></li>; })}</ul>{capabilities.missing_nodes.length > 0 && <div className="missing-nodes"><strong>缺少节点</strong><code>{capabilities.missing_nodes.join(", ")}</code></div>}</Panel>
        </aside>
      </div>
    </section>
  );
  if (!overlay) return content;
  return (
    <div ref={overlayRef} id="system-settings-dialog" className="settings-overlay" role="dialog" aria-modal="true" aria-labelledby="system-settings-title">
      <div className="settings-overlay__backdrop" aria-hidden="true" />
      {content}
    </div>
  );
}
