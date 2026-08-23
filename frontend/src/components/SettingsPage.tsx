import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { directorApi } from "../api/client";
import {
  normalizeLoraLoaderOverrides,
  resolveExecutionBackend,
  type CapabilityReport,
  type ConnectionTestResult,
  type DeviceTarget,
  type DirectorDeckConfig,
  type FeatureCatalog,
  type FeatureCapabilityEvaluation,
  type GPUResource,
  type LoraLoaderAdapterId,
  type LoraLoaderOverrideRecord,
  type MediaToolsStatus,
  type ModelInventory,
  type ModelRole,
  type RayLightProfile,
  type RayLightRuntimeStatus,
  type RayLightSetupStatus,
  type RuntimeSettings,
  type StorageConfiguration,
} from "../api/types";
import type { LoraLoaderOverrideEdit } from "../state/loraLoaderOverrides";
import type { UiTheme } from "../domain/theme";
import { Field, formatBytes, Panel, Spinner, StatusDot } from "./ui";

function sameSettings(left: RuntimeSettings, right: RuntimeSettings): boolean {
  return JSON.stringify(left) === JSON.stringify(right);
}

const HIDDEN_PATH_VALUE = "••••••••••••••••";

interface LoraLoaderMappingDraft {
  lora_filename: string;
  adapter_id: LoraLoaderAdapterId | "";
  options: Record<string, boolean>;
}

interface LoraLoaderChoice {
  adapter_id: LoraLoaderAdapterId;
  display_name: string;
  class_type: string;
  is_default: boolean;
  configuration_options: DirectorDeckConfig["lora"]["loaders"][number]["options"];
  capability: FeatureCapabilityEvaluation | null;
}

const EMPTY_LORA_LOADER_MAPPING_DRAFT: LoraLoaderMappingDraft = {
  lora_filename: "",
  adapter_id: "",
  options: {},
};

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

const MODEL_META: Record<ModelRole, { label: string; description: string; allowCpu: boolean }> = {
  fl2va: { label: "FL2VA 扩散模型", description: "按首尾帧素材自动选择生成配方", allowCpu: true },
  ref2va: { label: "REF2VA 扩散模型", description: "按源视频与参考素材自动选择生成配方", allowCpu: true },
  clip: { label: "文本编码器", description: "共享 CLIP / Qwen 编码器", allowCpu: true },
  video_vae: { label: "视频 VAE", description: "共享画面编码与解码", allowCpu: false },
  audio_vae: { label: "音频 VAE", description: "共享音频编码与解码", allowCpu: false },
};

export function validateRuntimeSettingsForm(settings: RuntimeSettings): string[] {
  const errors: string[] = [];
  if (settings.schema_version !== 3) errors.push("运行设置 schema 必须为 v3");
  if (!/^[A-Za-z0-9._:-]{1,128}$/.test(settings.client_id))
    errors.push("客户端 ID 只能包含字母、数字、点、下划线、冒号和连字符");
  if (settings.memory_policy !== "keep_resident")
    errors.push("片段间显存策略无效");
  if (![
    "release_after_sampling",
    "keep_until_switch",
  ].includes(settings.raylight_residency_policy))
    errors.push("RayLight 显存驻留策略无效");
  const placements = {
    fl2va: settings.placement.fl2va,
    ref2va: settings.placement.ref2va,
    clip: { device: settings.placement.clip_device },
    video_vae: { device: settings.placement.video_vae_device },
    audio_vae: { device: settings.placement.audio_vae_device },
  };
  for (const [role, binding] of Object.entries(placements) as [ModelRole, (typeof placements)[ModelRole]][]) {
    if (!/^(default|cpu|gpu:(0|[1-9][0-9]*))$/.test(binding.device))
      errors.push(`${MODEL_META[role].label}设备值无效`);
    if (!MODEL_META[role].allowCpu && binding.device === "cpu")
      errors.push(`${MODEL_META[role].label}不允许放在 CPU`);
    if (role === "fl2va" || role === "ref2va") {
      const diffusion = binding as RuntimeSettings["placement"]["fl2va"];
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
  const overrides = settings.lora_loader_overrides;
  const normalizedOverrides = normalizeLoraLoaderOverrides(overrides);
  if (
    overrides.length > 256 ||
    normalizedOverrides.length !== overrides.length ||
    normalizedOverrides.some((record, index) =>
      JSON.stringify(record) !== JSON.stringify(overrides[index]))
  ) errors.push("Standard LoRA 精确加载器映射无效");
  return errors;
}

export function SettingsPage({ settings, confirmedSettings = settings, resourcesReady = false, capabilities, featureCatalog = null, gpus, models, rayLightRuntimeStatus = null, rayLightRecoveryPending = false, rayLightRecoveryDisabled = false, rayLightRecoveryBlockedReason = null, loadingModels, syncError = null, runtimeEditingDisabled = false, overlay = false, theme = "dark", onThemeChange = () => undefined, onDraftChange = () => undefined, onSaved, onSaveLoraLoaderOverride, onConnectionTestSucceeded = () => undefined, onHostCapabilitiesChanged = () => undefined, onConfirmRayLightRuntimeRecovery = async () => undefined, onRequestClose }: {
  settings: RuntimeSettings; confirmedSettings?: RuntimeSettings; capabilities: CapabilityReport; gpus: GPUResource[];
  /** True once App has confirmed the four runtime resources against the host. */
  resourcesReady?: boolean;
  /** Last host-revision-verified feature catalog; null means adapter editing is unavailable. */
  featureCatalog?: FeatureCatalog | null;
  models: ModelInventory;
  rayLightRuntimeStatus?: RayLightRuntimeStatus | null;
  rayLightRecoveryPending?: boolean;
  rayLightRecoveryDisabled?: boolean;
  rayLightRecoveryBlockedReason?: string | null;
  loadingModels: boolean;
  syncError?: string | null;
  runtimeEditingDisabled?: boolean;
  overlay?: boolean;
  theme?: UiTheme;
  onThemeChange?: (theme: UiTheme) => void;
  onDraftChange?: (settings: RuntimeSettings) => void;
  onRequestClose?: (restoreFocus?: boolean) => void;
  /** App-owned whole-document write followed by its authoritative runtime GET. */
  onSaved: (settings: RuntimeSettings) => Promise<RuntimeSettings>;
  /** App-owned exact mapping CAS with a single binding-safe rebase. */
  onSaveLoraLoaderOverride: (edit: LoraLoaderOverrideEdit) => Promise<RuntimeSettings>;
  /** App re-reads capabilities/GPU/models after a successful host probe. */
  onConnectionTestSucceeded?: () => void;
  /** A completed install changed the live host snapshot and catalog identity. */
  onHostCapabilitiesChanged?: () => void;
  /** App-owned explicit restart certificate followed by authoritative refresh. */
  onConfirmRayLightRuntimeRecovery?: () => Promise<void>;
}) {
  const [working, setWorking] = useState<RuntimeSettings>(() => structuredClone(settings));
  const [testing, setTesting] = useState(false);
  const [probeResult, setProbeResult] = useState<ConnectionTestResult | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [rayLightSetup, setRayLightSetup] = useState<RayLightSetupStatus | null>(null);
  const [rayLightInstallBusy, setRayLightInstallBusy] = useState(false);
  const [mediaSetup, setMediaSetup] = useState<MediaToolsStatus | null>(null);
  const [ffmpegInstallBusy, setFfmpegInstallBusy] = useState(false);
  const [residencyNotice, setResidencyNotice] = useState<string | null>(null);
  const [storageConfiguration, setStorageConfiguration] = useState<StorageConfiguration | null>(null);
  const [currentDatabaseVisible, setCurrentDatabaseVisible] = useState(false);
  const [storageLoading, setStorageLoading] = useState(true);
  const [mappingDraft, setMappingDraft] = useState<LoraLoaderMappingDraft>(
    EMPTY_LORA_LOADER_MAPPING_DRAFT,
  );
  const [mappingOriginal, setMappingOriginal] = useState<LoraLoaderOverrideRecord | null>(null);
  const [mappingEditBase, setMappingEditBase] = useState<RuntimeSettings>(
    () => structuredClone(confirmedSettings),
  );
  const [mappingEditing, setMappingEditing] = useState(false);
  const [mappingBusy, setMappingBusy] = useState(false);
  const [productConfig, setProductConfig] = useState<DirectorDeckConfig | null>(null);
  const [productConfigError, setProductConfigError] = useState<string | null>(null);
  const editRevision = useRef(0);
  const workingRef = useRef(working);
  const confirmedRef = useRef(structuredClone(confirmedSettings));
  const hasLocalChanges = useRef(false);
  const onSavedRef = useRef(onSaved);
  const onHostCapabilitiesChangedRef = useRef(onHostCapabilitiesChanged);
  const rayLightInstallWasRunningRef = useRef(false);
  const ffmpegInstallWasRunningRef = useRef(false);
  const mountedRef = useRef(true);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const connectionStatusRef = useRef<HTMLDivElement>(null);
  const rayLightRecoveryRequestRef = useRef(false);
  const overlayRef = useRef<HTMLDivElement>(null);
  const pageRef = useRef<HTMLElement>(null);
  const validationErrors = validateRuntimeSettingsForm(working);
  const effectiveRuntimeEditingDisabled = runtimeEditingDisabled;
  const runtimeResourcesReady = !effectiveRuntimeEditingDisabled && capabilities.connection === "online" &&
    resourcesReady;
  const currentRayLightRuntimeStatus = resourcesReady ? rayLightRuntimeStatus : null;
  const rayLightFamilies = (["fl2va", "ref2va"] as const)
    .filter((role) => resolveExecutionBackend(working.placement[role]) === "raylight")
    .map((role) => role === "fl2va" ? "FL2VA" : "Ref2VA");
  const mappingAuthorityBlocked = effectiveRuntimeEditingDisabled || mappingBusy ||
    !sameSettings(working, confirmedRef.current);
  const observedLoraAdapters = featureCatalog?.entries.find((entry) =>
    entry.id === "lora")?.adapter_options ?? [];
  const allStandardLoraAdapterOptions = useMemo<LoraLoaderChoice[]>(() => {
    if (productConfig) {
      return productConfig.lora.loaders.map((loader) => ({
        adapter_id: loader.id,
        display_name: loader.display_name,
        class_type: loader.class_type,
        is_default: loader.id === productConfig.lora.fallback_policy.default_loader_id,
        configuration_options: loader.options,
        capability: observedLoraAdapters.find((option) =>
          option.adapter_id === loader.id)?.capability ?? null,
      }));
    }
    return observedLoraAdapters
      .filter((option) => option.backend === "standard")
      .map((option) => ({
        adapter_id: option.adapter_id,
        display_name: option.display_name,
        class_type: option.class_type,
        is_default: option.is_default,
        configuration_options: option.configuration_options,
        capability: option.capability,
      }));
  }, [observedLoraAdapters, productConfig]);
  const compiledLoraLoaderPolicies = useMemo(() => (
    productConfig?.lora.loader_policies.map((policy) => ({
      policy,
      expression: new RegExp(policy.lora_filename),
    })) ?? []
  ), [productConfig]);
  const loraLoaderPolicyFor = useCallback((loraFilename: string) => (
    compiledLoraLoaderPolicies.find(({ expression }) =>
      expression.test(loraFilename))?.policy ??
      productConfig?.lora.fallback_policy ?? null
  ), [compiledLoraLoaderPolicies, productConfig]);
  const loraLoaderOptionsFor = useCallback((loraFilename: string) => {
    const policy = loraLoaderPolicyFor(loraFilename);
    if (!policy) return allStandardLoraAdapterOptions;
    const allowed = new Set(policy.loader_ids);
    return allStandardLoraAdapterOptions.filter((option) =>
      allowed.has(option.adapter_id));
  }, [allStandardLoraAdapterOptions, loraLoaderPolicyFor]);
  const defaultLoraAdapterFor = useCallback((loraFilename: string) => {
    const policy = loraLoaderPolicyFor(loraFilename);
    const defaultId = policy?.default_loader_id ??
      productConfig?.lora.fallback_policy.default_loader_id;
    return allStandardLoraAdapterOptions.find((option) =>
      option.adapter_id === defaultId) ??
      allStandardLoraAdapterOptions.find((option) => option.is_default);
  }, [allStandardLoraAdapterOptions, loraLoaderPolicyFor, productConfig]);
  const standardLoraAdapterOptions = loraLoaderOptionsFor(
    mappingDraft.lora_filename,
  );
  const defaultLoraAdapterOption = standardLoraAdapterOptions.find((option) =>
    option.adapter_id === defaultLoraAdapterFor(
      mappingDraft.lora_filename,
    )?.adapter_id);
  const selectedLoraAdapterOption = allStandardLoraAdapterOptions.find((option) =>
    option.adapter_id === mappingDraft.adapter_id);
  const selectedLoraPolicy = loraLoaderPolicyFor(mappingDraft.lora_filename);
  const selectedLoraAdapterAllowed = !mappingDraft.adapter_id ||
    selectedLoraPolicy === null ||
    selectedLoraPolicy.loader_ids.includes(mappingDraft.adapter_id);

  onSavedRef.current = onSaved;
  onHostCapabilitiesChangedRef.current = onHostCapabilitiesChanged;
  workingRef.current = working;

  useEffect(() => {
    if (mappingBusy || mappingEditing) return;
    const currentFilename = mappingDraft.lora_filename;
    const loraFilename = currentFilename && models.loras.includes(currentFilename)
      ? currentFilename
      : models.loras[0] ?? "";
    if (!loraFilename) return;
    const original = confirmedSettings.lora_loader_overrides.find((record) =>
      record.lora_filename === loraFilename) ?? null;
    const adapterId = original?.adapter_id ??
      defaultLoraAdapterFor(loraFilename)?.adapter_id ?? "";
    const adapter = allStandardLoraAdapterOptions.find((option) =>
      option.adapter_id === adapterId);
    const options = Object.fromEntries((adapter?.configuration_options ?? []).map(
      (definition) => [
        definition.id,
        original?.options[definition.id] ?? definition.default,
      ],
    ));
    if (
      mappingDraft.lora_filename === loraFilename &&
      mappingDraft.adapter_id === adapterId &&
      JSON.stringify(mappingDraft.options) === JSON.stringify(options) &&
      JSON.stringify(mappingOriginal) === JSON.stringify(original)
    ) return;
    setMappingEditBase(structuredClone(confirmedSettings));
    setMappingOriginal(original ? { ...original, options: { ...original.options } } : null);
    setMappingDraft({
      lora_filename: loraFilename,
      adapter_id: adapterId,
      options,
    });
    setMappingEditing(false);
  }, [
    allStandardLoraAdapterOptions,
    confirmedSettings,
    defaultLoraAdapterFor,
    mappingBusy,
    mappingDraft,
    mappingEditing,
    mappingOriginal,
    models.loras,
  ]);

  const acceptRayLightSetup = useCallback((status: RayLightSetupStatus) => {
    const wasRunning = rayLightInstallWasRunningRef.current;
    rayLightInstallWasRunningRef.current = status.install.state === "running";
    setRayLightSetup(status);
    if (wasRunning && (
      status.install.state === "needs_restart" || status.dependencies_installed
    )) onHostCapabilitiesChangedRef.current();
  }, []);

  const acceptMediaSetup = useCallback((status: MediaToolsStatus) => {
    const wasRunning = ffmpegInstallWasRunningRef.current;
    ffmpegInstallWasRunningRef.current = status.install.state === "running";
    setMediaSetup(status);
    if (wasRunning && (
      status.install.state === "ready" || status.ready
    )) onHostCapabilitiesChangedRef.current();
  }, []);

  const requestClose = useCallback((restoreFocus = true) => {
    if (!onRequestClose) return;
    onRequestClose(restoreFocus);
  }, [onRequestClose]);

  const loadStorageConfiguration = useCallback(async (signal?: AbortSignal) => {
    setStorageLoading(true);
    try {
      const configuration = await directorApi.getStorage(signal);
      if (signal?.aborted) return;
      setStorageConfiguration(configuration);
    } catch {
      if (signal?.aborted) return;
      setStorageConfiguration(null);
    } finally {
      if (!signal?.aborted) setStorageLoading(false);
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
    const controller = new AbortController();
    void directorApi.getDirectorDeckConfig(controller.signal)
      .then((config) => {
        if (!controller.signal.aborted) {
          setProductConfig(config);
          setProductConfigError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setProductConfigError(
            reason instanceof Error ? reason.message : "无法读取加载器清单",
          );
        }
      });
    return () => controller.abort();
  }, []);

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
      setProbeResult(null);
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
    workingRef.current = next;
    hasLocalChanges.current = !sameSettings(next, confirmedRef.current);
    setWorking(next);
    onDraftChange(structuredClone(next));
    setMessage(null);
    if (validateRuntimeSettingsForm(next).length > 0) return;
    const snapshot = structuredClone(next);
    void Promise.resolve(onSavedRef.current(snapshot)).catch((reason) => {
      if (editRevision.current !== revision || !mountedRef.current) return;
      setMessage(reason instanceof Error ? reason.message : "系统设置自动同步失败");
    });
  };

  const mappingOptionsFor = (
    adapterId: string,
    values: Record<string, boolean> = {},
  ): Record<string, boolean> => {
    const adapter = allStandardLoraAdapterOptions.find((option) =>
      option.adapter_id === adapterId);
    if (!adapter) return {};
    return Object.fromEntries(adapter.configuration_options.map((definition) => [
      definition.id,
      values[definition.id] ?? definition.default,
    ]));
  };

  const beginMappingEdit = (loraFilename: string) => {
    if (effectiveRuntimeEditingDisabled || mappingBusy) return;
    const original = confirmedRef.current.lora_loader_overrides.find((record) =>
      record.lora_filename === loraFilename) ?? null;
    const adapterId = original?.adapter_id ??
      defaultLoraAdapterFor(loraFilename)?.adapter_id ?? "";
    setMappingEditBase(structuredClone(confirmedRef.current));
    setMappingOriginal(original ? { ...original, options: { ...original.options } } : null);
    setMappingDraft({
      lora_filename: loraFilename,
      adapter_id: adapterId,
      options: mappingOptionsFor(adapterId, original?.options),
    });
    setMappingEditing(false);
    setMessage(null);
  };

  const updateMappingDraft = (patch: Partial<LoraLoaderMappingDraft>) => {
    if (effectiveRuntimeEditingDisabled || mappingBusy) return;
    if (!mappingEditing) {
      setMappingEditBase(structuredClone(confirmedRef.current));
      setMappingEditing(true);
    }
    setMappingDraft((current) => ({ ...current, ...patch }));
    setMessage(null);
  };

  const acceptMappingAuthority = (confirmed: RuntimeSettings) => {
    const selectedFilename = mappingDraft.lora_filename;
    const next = structuredClone(confirmed);
    confirmedRef.current = next;
    workingRef.current = structuredClone(next);
    hasLocalChanges.current = false;
    setWorking(structuredClone(next));
    onDraftChange(structuredClone(next));
    const original = next.lora_loader_overrides.find((record) =>
      record.lora_filename === selectedFilename) ?? null;
    const adapterId = original?.adapter_id ??
      defaultLoraAdapterFor(selectedFilename)?.adapter_id ?? "";
    setMappingOriginal(original ? { ...original, options: { ...original.options } } : null);
    setMappingDraft({
      lora_filename: selectedFilename,
      adapter_id: adapterId,
      options: mappingOptionsFor(adapterId, original?.options),
    });
    setMappingEditing(false);
  };

  const saveMapping = async () => {
    if (
      !mappingEditing || mappingBusy || effectiveRuntimeEditingDisabled ||
      !mappingDraft.lora_filename ||
      !mappingDraft.adapter_id || !selectedLoraAdapterOption
    ) return;
    const next: LoraLoaderOverrideRecord = {
      lora_filename: mappingDraft.lora_filename,
      adapter_id: mappingDraft.adapter_id,
      options: mappingOptionsFor(mappingDraft.adapter_id, mappingDraft.options),
    };
    setMappingBusy(true);
    setMessage(null);
    try {
      const confirmed = await onSaveLoraLoaderOverride({
        base_settings: structuredClone(mappingEditBase),
        original: mappingOriginal
          ? { ...mappingOriginal, options: { ...mappingOriginal.options } }
          : null,
        next,
      });
      if (!mountedRef.current) return;
      acceptMappingAuthority(confirmed);
    } catch (reason) {
      if (mountedRef.current) {
        setMessage(reason instanceof Error ? reason.message : "LoRA 加载器映射保存失败");
      }
    } finally {
      if (mountedRef.current) setMappingBusy(false);
    }
  };

  const restoreMappingDefault = async (record: LoraLoaderOverrideRecord) => {
    if (mappingBusy || effectiveRuntimeEditingDisabled) return;
    setMappingBusy(true);
    setMessage(null);
    try {
      const confirmed = await onSaveLoraLoaderOverride({
        base_settings: structuredClone(confirmedRef.current),
        original: { ...record, options: { ...record.options } },
        next: null,
      });
      if (!mountedRef.current) return;
      acceptMappingAuthority(confirmed);
    } catch (reason) {
      if (mountedRef.current) {
        setMessage(reason instanceof Error ? reason.message : "恢复系统默认映射失败");
      }
    } finally {
      if (mountedRef.current) setMappingBusy(false);
    }
  };

  useEffect(() => {
    let cancelled = false;
    void directorApi.getRayLightSetup().then((status) => {
      if (!cancelled) acceptRayLightSetup(status);
    }).catch(() => {
      if (!cancelled) setRayLightSetup(null);
    });
    return () => {
      cancelled = true;
    };
  }, [acceptRayLightSetup]);

  useEffect(() => {
    if (rayLightSetup?.install.state !== "running") return;
    const timer = window.setTimeout(() => {
      void directorApi.getRayLightSetup().then((status) => {
        if (mountedRef.current) acceptRayLightSetup(status);
      }).catch(() => undefined);
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [acceptRayLightSetup, rayLightSetup]);

  const startRayLightInstall = async () => {
    const confirmed = window.confirm(
      "将安装多卡组件：ray、xfuser 等 Python 包。\n" +
      "执行方式：在 ComfyUI 的 Python 环境中运行 pip install -r requirements-raylight.txt（torch 版本固定不变）。\n" +
      "安装完成后需要重启 ComfyUI 才能生效。继续？",
    );
    if (!confirmed) return;
    setRayLightInstallBusy(true);
    try {
      const started = await directorApi.installRayLight();
      if (started.state === "running" || started.state === "needs_restart") {
        rayLightInstallWasRunningRef.current = true;
      }
      acceptRayLightSetup(await directorApi.getRayLightSetup());
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "多卡组件安装启动失败");
    } finally {
      setRayLightInstallBusy(false);
    }
  };

  const cancelRayLightInstall = async () => {
    try {
      await directorApi.cancelRayLightInstall();
      acceptRayLightSetup(await directorApi.getRayLightSetup());
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "取消安装失败");
    }
  };

  useEffect(() => {
    let cancelled = false;
    void directorApi.getMediaSetup().then((status) => {
      if (!cancelled) acceptMediaSetup(status);
    }).catch(() => {
      if (!cancelled) setMediaSetup(null);
    });
    return () => {
      cancelled = true;
    };
  }, [acceptMediaSetup]);

  useEffect(() => {
    if (mediaSetup?.install.state !== "running") return;
    const timer = window.setTimeout(() => {
      void directorApi.getMediaSetup().then((status) => {
        if (mountedRef.current) acceptMediaSetup(status);
      }).catch(() => undefined);
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [acceptMediaSetup, mediaSetup]);

  const startFfmpegInstall = async () => {
    const confirmed = window.confirm(
      "将安装媒体组件：static-ffmpeg（含 ffmpeg 与 ffprobe，全编码器）。\n" +
      "执行方式：在 ComfyUI 的 Python 环境中运行 pip install static-ffmpeg。\n" +
      "安装完成立即生效，无需重启。继续？",
    );
    if (!confirmed) return;
    setFfmpegInstallBusy(true);
    try {
      const started = await directorApi.installFfmpeg();
      if (started.state === "running" || started.state === "ready") {
        ffmpegInstallWasRunningRef.current = true;
      }
      acceptMediaSetup(await directorApi.getMediaSetup());
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "ffmpeg 安装启动失败");
    } finally {
      setFfmpegInstallBusy(false);
    }
  };

  const cancelFfmpegInstall = async () => {
    try {
      await directorApi.cancelFfmpegInstall();
      acceptMediaSetup(await directorApi.getMediaSetup());
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : "取消安装失败");
    }
  };
  const test = async () => {
    if (effectiveRuntimeEditingDisabled) return;
    setTesting(true); setMessage(null);
    setProbeResult(null);
    try {
      const result = await directorApi.testConnection();
      if (!mountedRef.current) return;
      setProbeResult(result);
      if (result.ok) onConnectionTestSucceeded();
    }
    catch (reason) {
      if (!mountedRef.current) return;
      const failure = reason instanceof Error ? reason.message : "连接失败";
      setProbeResult({ ok: false, message: failure });
    }
    finally {
      if (mountedRef.current) setTesting(false);
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
  const setPlacementDevice = (role: ModelRole, value: DeviceTarget) => {
    if (!MODEL_META[role].allowCpu && value === "cpu") return;
    if (role === "fl2va" || role === "ref2va") {
      const current = working.placement[role];
      if (resolveExecutionBackend(current) === "raylight" && value !== "default") return;
      change({
        ...working,
        placement: {
          ...working.placement,
          [role]: { ...current, device: value },
        },
      });
      return;
    }
    const key = `${role}_device` as "clip_device" | "video_vae_device" | "audio_vae_device";
    change({ ...working, placement: { ...working.placement, [key]: value } });
  };
  const setDiffusionPlacement = (
    role: "fl2va" | "ref2va",
    patch: Partial<RuntimeSettings["placement"]["fl2va"]>,
  ) => {
    const currentPlacement = working.placement[role];
    const nextPlacement = { ...currentPlacement, ...patch };
    const switchedToRayLight =
      resolveExecutionBackend(currentPlacement) === "standard" &&
      resolveExecutionBackend(nextPlacement) === "raylight";
    const nextResidencyPolicy = switchedToRayLight && working.raylight_residency_policy === "release_after_sampling"
      ? "keep_until_switch"
      : working.raylight_residency_policy;
    if (switchedToRayLight) {
      const label = role === "fl2va" ? "FL2VA" : "Ref2VA";
      setResidencyNotice(
        `${label} 已切换为 RayLight，已自动选择“按运行配置常驻”。切换运行拓扑时由 Director 先安全释放旧池。`,
      );
    }
    change({
      ...working,
      raylight_residency_policy: nextResidencyPolicy,
      placement: {
        ...working.placement,
        [role]: nextPlacement,
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
    setDiffusionPlacement(role, {
      raylight: normalizedRayLight,
      ...(resolvesToRayLight ? { device: "default" as const } : {}),
    });
  };
  const setMultiGpuEnabled = (enabled: boolean) => {
    if (enabled) {
      change({ ...working, multi_gpu_enabled: true });
      return;
    }
    // A disabled multi-GPU switch must never leave a multi-card RayLight pool
    // behind: the backend fails closed when it compiles raylight units with
    // multi_gpu_enabled=false. Converge both diffusion families to their first
    // selected card atomically, with the single-card topology preset.
    change({
      ...working,
      multi_gpu_enabled: false,
      placement: {
        ...working.placement,
        fl2va: {
          ...working.placement.fl2va,
          raylight: {
            ...working.placement.fl2va.raylight,
            gpu_select: [working.placement.fl2va.raylight.gpu_select[0] ?? 0],
            ulysses_degree: 1,
            ring_degree: 1,
          },
        },
        ref2va: {
          ...working.placement.ref2va,
          raylight: {
            ...working.placement.ref2va.raylight,
            gpu_select: [working.placement.ref2va.raylight.gpu_select[0] ?? 0],
            ulysses_degree: 1,
            ring_degree: 1,
          },
        },
      },
    });
  };
  const toggleRayLightGpu = (
    role: "fl2va" | "ref2va",
    index: number,
    selected: boolean,
  ) => {
    const current = working.placement[role].raylight;
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
    const current = working.placement[role].raylight;
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
      {effectiveRuntimeEditingDisabled && <div className="notice" role="status">运行设置暂时锁定。</div>}
      {message && <div className="notice" role="alert">{message}</div>}
      <div className="settings-layout">
        <div className="settings-main">
          <Panel eyebrow="连接" title="ComfyUI" description="浏览器仅连接导演台后端，由后端代理 ComfyUI。">
            <div ref={connectionStatusRef} className="connection-card" tabIndex={-1}><StatusDot state={testing || probeResult === null && capabilities.connection === "checking" ? "checking" : probeResult ? probeResult.ok ? "online" : "offline" : capabilities.connection} /><div><strong>{testing ? "正在测试连接" : probeResult ? probeResult.ok ? "当前实例可连接" : "当前实例连接失败" : capabilities.connection === "offline" ? "ComfyUI 离线" : capabilities.connection === "online" ? "ComfyUI 在线" : "等待检测"}</strong><small>{testing ? "正在等待 ComfyUI 响应" : probeResult ? probeResult.latency_ms === undefined ? probeResult.message : `${probeResult.message} · 响应 ${probeResult.latency_ms} ms` : capabilities.connection === "offline" ? capabilities.message || "无法读取运行环境" : capabilities.latency_ms === undefined ? "尚未读取延迟" : `响应 ${capabilities.latency_ms} ms`}</small></div><button type="button" className="button button--ghost" onClick={() => void test()} disabled={testing || effectiveRuntimeEditingDisabled}>{testing ? <><Spinner />测试中…</> : "测试连接"}</button></div>
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
            <Field label="客户端 ID"><input required disabled={effectiveRuntimeEditingDisabled} maxLength={128} pattern="[A-Za-z0-9._:-]+" value={working.client_id} onChange={(event) => change({ ...working, client_id: event.target.value })} /></Field>
          </Panel>
          <Panel eyebrow="数据" title="数据存储" description="数据库固定在宿主 ComfyUI 的用户目录下，随安装走；运行时不可切换。" action={storageLoading ? <Spinner label="读取数据存储配置" /> : undefined}>
            {storageConfiguration && <div className="storage-current" aria-label="当前数据库存储配置">
              <div className="storage-current__path">
                <span>当前数据库</span>
                <div className="sensitive-value sensitive-value--code">
                  <code title={currentDatabaseVisible ? storageConfiguration.active_database_path : undefined}>{currentDatabaseVisible ? storageConfiguration.active_database_path : HIDDEN_PATH_VALUE}</code>
                  <VisibilityToggle label="当前数据库路径显示状态" visible={currentDatabaseVisible} onToggle={() => setCurrentDatabaseVisible((current) => !current)} />
                </div>
              </div>
            </div>}
            {!storageConfiguration && !storageLoading && <div className="notice notice--error" role="alert">无法读取当前数据库路径。</div>}
          </Panel>
          <Panel eyebrow="媒体" title="媒体工具 (ffmpeg)" description="素材探测、代理转码与长片拼接依赖 ffmpeg/ffprobe；缺失时可在此一键安装，立即生效无需重启。">
            {mediaSetup === null ? <p className="muted">正在检测媒体工具…</p> : (
              <>
                <div className="connection-card">
                  <StatusDot state={mediaSetup.ready ? "online" : "offline"} />
                  <div>
                    <strong>{mediaSetup.ready ? "ffmpeg 可用" : "ffmpeg 未就绪"}</strong>
                    <small>{mediaSetup.ready
                      ? mediaSetup.ffmpeg_path ?? "已就绪"
                      : mediaSetup.ffmpeg_available
                        ? "已找到 ffmpeg，但缺少 libx264/aac 编码器或 ffprobe"
                        : "未检测到 ffmpeg/ffprobe"}</small>
                  </div>
                  {!mediaSetup.ready && mediaSetup.install.state !== "running" && (
                    <button type="button" className="button button--ghost" disabled={ffmpegInstallBusy} onClick={() => void startFfmpegInstall()}>{ffmpegInstallBusy ? "正在启动…" : "安装 ffmpeg"}</button>
                  )}
                </div>
                {mediaSetup.install.state === "running" && (
                  <div className="raylight-setup">
                    <p><Spinner /> 正在安装 ffmpeg，请稍候…</p>
                    <button type="button" className="button button--ghost" onClick={() => void cancelFfmpegInstall()}>取消安装</button>
                  </div>
                )}
                {mediaSetup.install.state === "failed" && <div className="notice notice--error" role="alert">ffmpeg 安装失败：{mediaSetup.install.error ?? "未知错误"}。可重试，或按平台指引手动安装。</div>}
                {mediaSetup.install.log_tail.length > 0 && (
                  <details className="raylight-setup-log"><summary>安装日志</summary><pre>{mediaSetup.install.log_tail.join("\n")}</pre></details>
                )}
              </>
            )}
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
              ? "family、模型、LoRA、GPU 池和拓扑完全一致时复用 CUDA 权重；任一项变化会先运行 RayLight 清理屏障并重建 Ray 池，切到 Standard 也一样。无需重启 ComfyUI。"
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
          <Panel eyebrow="运行" title="多卡推理" description="RayLight 多卡组件默认不安装；开启后按提示安装一次并重启 ComfyUI 生效。关闭不会卸载组件，下次开启可直接使用。">
            <label className="check-field">
              <input
                type="checkbox"
                aria-label="启用多卡推理"
                disabled={effectiveRuntimeEditingDisabled || rayLightSetup?.platform_supported === false}
                checked={working.multi_gpu_enabled}
                onChange={(event) => setMultiGpuEnabled(event.target.checked)}
              />
              <span><strong>启用多卡推理（RayLight）</strong><small>GPU 池配置 2 张及以上逻辑卡时自动使用 RayLight 执行</small></span>
            </label>
            {rayLightSetup?.platform_supported === false && <div className="notice" role="status">多卡推理目前仅支持 Linux。</div>}
            {working.multi_gpu_enabled && rayLightSetup && !rayLightSetup.dependencies_installed && rayLightSetup.install.state !== "running" && rayLightSetup.install.state !== "needs_restart" && (
              <div className="raylight-setup">
                <p>多卡组件（ray、xfuser）尚未安装：将在 ComfyUI 的 Python 环境中执行 pip install -r requirements-raylight.txt，完成后需重启 ComfyUI。</p>
                <button type="button" className="button" disabled={!rayLightSetup.requirements_available || rayLightInstallBusy} onClick={() => void startRayLightInstall()}>{rayLightInstallBusy ? "正在启动…" : "安装多卡组件"}</button>
                {!rayLightSetup.requirements_available && <small className="inline-error">当前安装缺少 requirements-raylight.txt；请手动安装后重启。</small>}
              </div>
            )}
            {rayLightSetup?.install.state === "running" && (
              <div className="raylight-setup">
                <p><Spinner /> 正在安装多卡组件，请稍候…</p>
                <button type="button" className="button button--ghost" onClick={() => void cancelRayLightInstall()}>取消安装</button>
              </div>
            )}
            {rayLightSetup?.install.state === "needs_restart" && <div className="notice" role="status">多卡组件已安装完成。重启 ComfyUI 后多卡生效。</div>}
            {rayLightSetup?.install.state === "failed" && <div className="notice notice--error" role="alert">多卡组件安装失败：{rayLightSetup.install.error ?? "未知错误"}。可重试，或按日志中的命令手动安装后重启。</div>}
            {working.multi_gpu_enabled && rayLightSetup?.dependencies_installed && <div className="notice" role="status">多卡组件已安装；节点注册状态见各模型族的执行能力标记。</div>}
            {rayLightSetup && rayLightSetup.install.log_tail.length > 0 && (
              <details className="raylight-setup-log"><summary>安装日志</summary><pre>{rayLightSetup.install.log_tail.join("\n")}</pre></details>
            )}
          </Panel>
          <Panel eyebrow="宿主知识" title="Standard LoRA 加载器映射" description="为 LoRA 选择当前版本允许的加载器。">
            {models.loras.length === 0
              ? <div className="notice" role="status">当前 ComfyUI 没有可选 LoRA 文件。</div>
              : <div className="lora-loader-mapping-workbench">
                <section className="lora-loader-choice-panel" aria-label="LoRA 列表区域">
                  <header><strong>LoRA 列表</strong><small>{models.loras.length} 个</small></header>
                  <div className="lora-loader-choice-list" role="listbox" aria-label="LoRA 列表">
                    {models.loras.map((filename) => {
                      const override = working.lora_loader_overrides.find((record) =>
                        record.lora_filename === filename);
                      const loader = allStandardLoraAdapterOptions.find((option) =>
                        option.adapter_id === override?.adapter_id) ??
                        defaultLoraAdapterFor(filename);
                      const overrideAllowed = !override ||
                        (loraLoaderPolicyFor(filename)?.loader_ids.includes(
                          override.adapter_id,
                        ) ?? true);
                      return <button
                        type="button"
                        role="option"
                        title={filename}
                        aria-selected={mappingDraft.lora_filename === filename}
                        className={mappingDraft.lora_filename === filename ? "is-selected" : ""}
                        key={filename}
                        disabled={mappingAuthorityBlocked}
                        onClick={() => beginMappingEdit(filename)}
                      >
                        <code title={filename}>{filename}</code>
                        <small>{override
                          ? `${loader?.display_name ?? override.adapter_id}${overrideAllowed ? "" : " · 与当前规则不兼容"}`
                          : `${loader?.display_name ?? "默认加载器"}（默认）`}</small>
                      </button>;
                    })}
                  </div>
                </section>
                <section className="lora-loader-choice-panel" aria-label="加载器列表区域">
                  <header><strong>加载器列表</strong><small>版本内置</small></header>
                  <div className="lora-loader-choice-list" role="listbox" aria-label="加载器列表">
                    {standardLoraAdapterOptions.length === 0 && <p className="muted">
                      {productConfigError ?? "正在读取加载器清单…"}
                    </p>}
                    {standardLoraAdapterOptions.map((option) => <button
                      type="button"
                      role="option"
                      aria-selected={mappingDraft.adapter_id === option.adapter_id}
                      className={mappingDraft.adapter_id === option.adapter_id ? "is-selected" : ""}
                      key={option.adapter_id}
                      disabled={mappingAuthorityBlocked || !mappingDraft.lora_filename}
                      onClick={() => updateMappingDraft({
                        adapter_id: option.adapter_id,
                        options: mappingOptionsFor(option.adapter_id),
                      })}
                    >
                      <strong>{option.display_name}</strong>
                      <small>{option.class_type}{
                        option.adapter_id === defaultLoraAdapterOption?.adapter_id
                          ? " · 默认"
                          : ""
                      }</small>
                    </button>)}
                  </div>
                </section>
                <section className="lora-loader-config-panel" aria-label="加载器配置项区域">
                  <header><strong>配置项</strong><small>{selectedLoraAdapterOption?.display_name ?? "未选择"}</small></header>
                  {!selectedLoraAdapterAllowed && <div className="notice notice--warning" role="alert">
                    这条历史映射不在当前 LoRA 的允许清单中。它会保留供复核，但不能再次保存或用于新任务；请选择允许的加载器，或恢复系统默认。
                  </div>}
                  {selectedLoraAdapterOption?.configuration_options.length
                    ? <div className="lora-loader-config-fields">
                      {selectedLoraAdapterOption.configuration_options.map((definition) => <label
                        className="check-field"
                        key={definition.id}
                      >
                        <input
                          type="checkbox"
                          aria-label={`LoRA 加载器配置 ${definition.label}`}
                          disabled={mappingAuthorityBlocked}
                          checked={mappingDraft.options[definition.id] ?? definition.default}
                          onChange={(event) => updateMappingDraft({
                            options: {
                              ...mappingDraft.options,
                              [definition.id]: event.target.checked,
                            },
                          })}
                        />
                        <span><strong>{definition.label}</strong><small>{definition.description}</small></span>
                      </label>)}
                    </div>
                    : <p className="muted">此加载器没有额外配置项。</p>}
                  {selectedLoraAdapterOption && <div
                    className={`lora-loader-adapter-evidence${selectedLoraAdapterOption.capability?.available === false ? " lora-loader-adapter-evidence--unavailable" : ""}`}
                    role="status"
                    aria-label="LoRA 映射节点检测"
                  >
                    <strong>{selectedLoraAdapterOption.capability === null
                      ? "节点状态尚未读取"
                      : selectedLoraAdapterOption.capability.available
                        ? "当前已检测到节点"
                        : "当前未检测到节点"}</strong>
                    <p>{selectedLoraAdapterOption.capability === null
                      ? "加载器清单仍可选择和保存；节点状态由宿主能力检查更新。"
                      : selectedLoraAdapterOption.capability.available
                      ? "这里只表示宿主当前存在同名节点，不认证其来源或实现。"
                      : "仍可保存配置；实际使用时若节点缺失，任务会返回明确错误。"}</p>
                  </div>}
                  <div className="lora-loader-mapping-actions">
                    <button
                      type="button"
                      className="button button--primary"
                      disabled={
                        mappingAuthorityBlocked || !mappingEditing ||
                        !mappingDraft.lora_filename || !mappingDraft.adapter_id ||
                        !selectedLoraAdapterOption || !selectedLoraAdapterAllowed
                      }
                      onClick={() => void saveMapping()}
                    >{mappingBusy ? "保存中…" : "保存映射"}</button>
                    <button
                      type="button"
                      className="button button--ghost"
                      disabled={mappingAuthorityBlocked || !mappingOriginal}
                      onClick={() => mappingOriginal && void restoreMappingDefault(mappingOriginal)}
                    >恢复系统默认</button>
                  </div>
                </section>
              </div>}
          </Panel>
          <Panel eyebrow="运行" title="运行设备与 RayLight 拓扑" description="这里只保存 runtime placement；模型文件与 LoRA 请在当前项目的全局设置中编辑。">
            <div className="model-bindings">
              {(Object.keys(MODEL_META) as ModelRole[]).map((role) => {
                const meta = MODEL_META[role];
                const diffusionRole: "fl2va" | "ref2va" | null = role === "fl2va" || role === "ref2va" ? role : null;
                const diffusion = diffusionRole ? working.placement[diffusionRole] : null;
                const binding = diffusion ?? {
                  device: working.placement[`${role}_device` as "clip_device" | "video_vae_device" | "audio_vae_device"],
                };
                const discoveredDevices: DeviceTarget[] = gpus.filter((gpu) => gpu.visible).map((gpu) => `gpu:${gpu.index}` as const);
                const devices: DeviceTarget[] = [...new Set(["default" as const, ...(meta.allowCpu ? ["cpu" as const] : []), ...discoveredDevices, binding.device])];
                const resolvedBackend = diffusion ? resolveExecutionBackend(diffusion) : null;
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
                    <Field label={diffusion ? "Standard 执行设备" : "运行设备"} hint={diffusion && resolvedBackend === "raylight" ? "RayLight 固定为 default；切回 Standard 后可重新选择逻辑设备。" : diffusion ? "标准原生节点使用此 ComfyUI 逻辑设备。" : undefined}><select required disabled={!runtimeResourcesReady || resolvedBackend === "raylight"} aria-label={`${meta.label}设备`} value={binding.device} onChange={(event) => setPlacementDevice(role, event.target.value as DeviceTarget)}>{devices.map((device) => <option key={device} value={device}>{device === "default" ? "default（ComfyUI 自动）" : device}</option>)}</select></Field>
                  </div>
                  {diffusion && diffusionRole && <section className="model-execution-panel" aria-label={`${meta.label}执行拓扑`}>
                    <header>
                      <div><strong>自动执行</strong></div>
                      <span className="execution-badge">{resolvedBackend === "raylight" ? `RAYLIGHT · ${diffusion.raylight.gpu_select.length} 卡` : `STANDARD · ${binding.device}`}</span>
                    </header>
                    <p>{resolvedBackend === "standard" ? `GPU 池只有 1 张卡，自动使用标准原生子图；gpu:N 是 ComfyUI 进程内逻辑编号。` : `GPU 池有 ${diffusion.raylight.gpu_select.length} 张卡，自动使用 RayLight；逻辑 GPU ${diffusion.raylight.gpu_select.join(", ")}，U${diffusion.raylight.ulysses_degree} × R${diffusion.raylight.ring_degree} × 条件1 × 数据1。`}</p>
                    <div className="raylight-topology">
                      <fieldset disabled={!runtimeResourcesReady || !working.multi_gpu_enabled}>
                        <legend>RayLight 逻辑 GPU 池</legend>
                        <div className="raylight-gpu-pool">
                          {rayGpuIndexes.map((index) => <label key={index} className={diffusion.raylight.gpu_select.includes(index) ? "is-selected" : ""}><input type="checkbox" aria-label={`${meta.label} RayLight 逻辑 GPU ${index}`} checked={diffusion.raylight.gpu_select.includes(index)} onChange={(event) => toggleRayLightGpu(diffusionRole, index, event.target.checked)} /><span>GPU {index}</span><small>逻辑编号</small></label>)}
                        </div>
                      </fieldset>
                      {!working.multi_gpu_enabled && <small>启用多卡推理后可配置多卡 GPU 池。</small>}
                      <Field label="Ulysses degree"><select aria-label={`${meta.label} RayLight Ulysses degree`} disabled={!runtimeResourcesReady || resolvedBackend !== "raylight"} value={diffusion.raylight.ulysses_degree} onChange={(event) => setRayLightAxis(diffusionRole, "ulysses_degree", Number(event.target.value))}>{degreeOptions.map((degree) => <option key={degree} value={degree}>{degree}</option>)}</select></Field>
                      <Field label="Ring degree"><select aria-label={`${meta.label} RayLight Ring degree`} disabled={!runtimeResourcesReady || resolvedBackend !== "raylight"} value={diffusion.raylight.ring_degree} onChange={(event) => setRayLightAxis(diffusionRole, "ring_degree", Number(event.target.value))}>{degreeOptions.map((degree) => <option key={degree} value={degree}>{degree}</option>)}</select></Field>
                      <div className="fixed-runtime-value fixed-runtime-value--topology"><span>条件 / 数据并行</span><strong>1 × 1</strong><small>当前拓扑固定</small></div>
                      <label className="check-field"><input type="checkbox" aria-label={`${meta.label} RayLight FSDP`} disabled checked={false} /><span><strong>FSDP</strong><small>v1 暂停开放，等待显存清理实机验证</small></span></label>
                      <label className="check-field"><input type="checkbox" aria-label={`${meta.label} RayLight CPU offload`} disabled checked={false} /><span><strong>CPU offload</strong><small>随 FSDP 暂停开放</small></span></label>
                    </div>
                    <small className="raylight-contract-note">1 卡自动走 Standard；2–8 卡自动走 RayLight。拓扑必须满足 Ulysses × Ring × 条件并行 × 数据并行 = 逻辑 GPU 池大小；修改 GPU 池会原子重置为 U=池大小、Ring=1。</small>
                  </section>}
                </div>;
              })}
            </div>
          </Panel>
        </div>
        <aside className="settings-aside">
          <Panel eyebrow="资源" title="GPU 状态"><div className="gpu-list">{gpus.length === 0 && <p className="muted">尚未读取 GPU 数据</p>}{gpus.map((gpu) => { const used = gpu.vram_total ? Math.round((1 - gpu.vram_free / gpu.vram_total) * 100) : 0; return <div className="gpu-card" key={gpu.index}><header><span>GPU {gpu.index}</span><small>ComfyUI 逻辑编号</small></header><strong>{gpu.name}</strong><div className="progress"><span style={{ width: `${used}%` }} /></div><footer><span>显存 {used}%</span><span>{formatBytes(gpu.vram_free)} 可用</span></footer></div>;})}</div></Panel>
          <Panel eyebrow="能力" title="服务端统一预检"><p className="muted">功能、模型、LoRA 与执行后端是否可用，由生成和执行计划预检使用当前宿主快照统一判定。</p></Panel>
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
