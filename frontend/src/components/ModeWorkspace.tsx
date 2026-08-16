import type { FormEvent } from "react";
import { describeLoraLoader, EMPTY_MODELS, isConfiguredComfyUrl, type CapabilityReport, type DiffusionModelBinding, type DiffusionModelRole, type ModelInventory, type RuntimeSettings } from "../api/types";
import { isStableAssetReference, isStableSlottedAssetReference } from "../domain/assets";
import {
  MODE_META,
  randomSafeSeed,
  SAMPLING_SCHEDULERS,
  type ModeDraft,
} from "../domain/modes";
import {
  alignH3Frames,
  H3_MAX_SHOT_FRAMES,
} from "../domain/timing";
import {
  MINIMAX_H3_REFERENCE_LIMITS,
  maxSlotForCapacity,
} from "../domain/h3Capabilities";
import { TimelineEditor } from "./TimelineEditor";
import {
  limitPromptCharacters,
  MINIMAX_H3_PROMPT_MAX_CHARACTERS,
  promptCharacterCount,
} from "../domain/promptLimits";
import type {
  AssetMutation,
  DraftAssetField,
} from "../state/directorState";
import { DeferredNumberInput, Field, Panel, Spinner } from "./ui";

interface ModeWorkspaceProps {
  formId?: string;
  globalSettingsId?: string;
  globalSettingsOpen?: boolean;
  onGlobalSettingsClose?: () => void;
  draft: ModeDraft;
  settings: RuntimeSettings;
  capabilities: CapabilityReport;
  models?: ModelInventory;
  runtimeModelSaving?: boolean;
  submitting: boolean;
  dirty?: boolean;
  onChange: (draft: ModeDraft) => void;
  onRuntimeModelChange?: (
    role: DiffusionModelRole,
    patch: Partial<DiffusionModelBinding>,
  ) => void;
  onAssetsChange?: (
    mode: ModeDraft["mode"],
    shotId: string,
    field: DraftAssetField,
    mutation: AssetMutation,
  ) => void;
  onSubmit: () => void;
}

type ReferenceTagKind = "Picture" | "Audio" | "Video";

function referenceTags(prompt: string): { kind: ReferenceTagKind; number: number; raw: string }[] {
  return Array.from(
    prompt.matchAll(/<\s*(Picture|Audio|Video)\s+(\d+)\s*>/gi),
    (match) => ({
      kind: `${match[1][0].toUpperCase()}${match[1].slice(1).toLowerCase()}` as ReferenceTagKind,
      number: Number(match[2]),
      raw: match[0],
    }),
  );
}

function invalidTags(
  prompt: string,
  available: Record<ReferenceTagKind, Set<number>>,
): string[] {
  return referenceTags(prompt)
    .filter((tag) => !available[tag.kind].has(tag.number))
    .map((tag) => tag.raw);
}

function hasDenseSlots(assets: { slot: number }[]): boolean {
  return [...assets]
    .sort((left, right) => left.slot - right.slot)
    .every((asset, index) => asset.slot === index);
}

export function validateModeDraft(draft: ModeDraft): string[] {
  const errors: string[] = [];
  const { render, sampling } = draft;
  if (
    ![render.width, render.height].every(
      (value) => Number.isInteger(value) && value >= 32 && value <= 8192 && value % 32 === 0,
    )
  )
    errors.push("画面宽高必须是 32–8192 范围内的 32 倍数");
  if (render.fps !== 24)
    errors.push("原生 MiniMax H3 帧率固定为 24fps");
  if (!Number.isInteger(sampling.steps) || sampling.steps < 1 || sampling.steps > 200)
    errors.push("采样步数必须是 1–200 的整数");
  if (
    !Number.isSafeInteger(sampling.seed) ||
    sampling.seed < 0 ||
    sampling.seed > Number.MAX_SAFE_INTEGER
  )
    errors.push("Seed 必须是非负 JavaScript 安全整数");
  if (
    ![sampling.shift, sampling.audio_shift].every(
      (value) => Number.isFinite(value) && value >= 0.01 && value <= 100,
    )
  )
    errors.push("视频与音频 Shift 必须在 0.01–100 之间");
  if (promptCharacterCount(draft.prompt) > MINIMAX_H3_PROMPT_MAX_CHARACTERS)
    errors.push("全局提示词超过后端长度限制");
  if (draft.ref_image_size !== "match" && draft.ref_image_size !== "max")
    errors.push("参考图采样尺寸必须选择 match 或 max");
  if (draft.shots.length < 1 || draft.shots.length > 128)
    errors.push("镜头数量必须在 1–128 之间");
  if (new Set(draft.shots.map((shot) => shot.id)).size !== draft.shots.length)
    errors.push("镜头 ID 必须唯一");
  if (!draft.prompt.trim() && draft.shots.some((shot) => shot.enabled && !shot.prompt.trim())) errors.push("默认提示词为空时，每个启用镜头都必须填写提示词");
  if (!draft.shots.some((shot) => shot.enabled)) errors.push("至少启用一个镜头");
  if (
    draft.shots.some(
      (shot) =>
        !shot.id ||
        shot.id.length > 128 ||
        shot.title.length > 256 ||
        promptCharacterCount(shot.prompt) > MINIMAX_H3_PROMPT_MAX_CHARACTERS,
    )
  )
    errors.push("镜头文字字段超过后端长度限制");
  if (draft.mode === "fl2v") {
    if (
      draft.shots.some(
        (shot) =>
          !Number.isFinite(shot.duration_seconds) ||
          shot.duration_seconds < 0.1 ||
          shot.duration_seconds > 120,
      )
    )
      errors.push("FL2V 镜头时长必须在 0.1–120 秒之间");
  } else if (
    draft.shots.some(
      (shot) =>
        !Number.isFinite(shot.duration_seconds) ||
        shot.duration_seconds <= 0 ||
        shot.duration_seconds > 120,
    )
  ) {
    errors.push("镜头时长必须大于 0 且不超过 120 秒");
  }
  if (
    draft.shots.some(
      (shot) =>
        shot.enabled &&
        alignH3Frames(shot.duration_seconds, render.fps) > H3_MAX_SHOT_FRAMES,
    )
  )
    errors.push("镜头时长与帧率组合超过 MiniMax H3 的 512 帧上限");
  switch (draft.mode) {
    case "i2v":
      if (draft.shots.some((shot) => shot.enabled && !isStableAssetReference(shot.first_image, "image"))) errors.push("每个启用的 I2V 镜头都需要有效起始帧");
      if (draft.shots.some((shot) => shot.first_image && !isStableAssetReference(shot.first_image, "image"))) errors.push("I2V 中存在失效素材，请重新上传");
      break;
    case "fl2v":
      if (draft.shots.some((shot) => shot.enabled && !isStableAssetReference(shot.first_image, "image") && !isStableAssetReference(shot.last_image, "image"))) errors.push("每个启用的 FL2V 镜头至少需要有效首帧或尾帧");
      if (draft.shots.some((shot) => (shot.first_image && !isStableAssetReference(shot.first_image, "image")) || (shot.last_image && !isStableAssetReference(shot.last_image, "image")))) errors.push("FL2V 中存在失效素材，请重新上传");
      break;
    case "r2v":
      if (draft.shots.some((shot) => shot.enabled && !shot.reference_images.length && !shot.reference_audios.length && !shot.reference_videos.length)) errors.push("每个启用的 R2V 参考组至少需要一个参考素材");
      if (draft.shots.some((shot) => shot.reference_images.some((asset) => !isStableSlottedAssetReference(asset, "image", maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages))) || shot.reference_audios.some((asset) => !isStableSlottedAssetReference(asset, "audio", maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios))) || shot.reference_videos.some((asset) => !isStableSlottedAssetReference(asset, "video", maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos))))) errors.push("R2V 中存在失效素材，请重新上传");
      if (draft.shots.some((shot) => new Set(shot.reference_images.map((asset) => asset.slot)).size !== shot.reference_images.length || new Set(shot.reference_audios.map((asset) => asset.slot)).size !== shot.reference_audios.length || new Set(shot.reference_videos.map((asset) => asset.slot)).size !== shot.reference_videos.length)) errors.push("R2V 同类参考素材的槽位必须唯一");
      if (draft.shots.some((shot) => shot.enabled && (!hasDenseSlots(shot.reference_images) || !hasDenseSlots(shot.reference_audios) || !hasDenseSlots(shot.reference_videos)))) errors.push("R2V 各类参考素材槽位必须连续为 0..N-1");
      {
        const stale = draft.shots.flatMap((shot) =>
          !shot.enabled
            ? []
            : invalidTags(shot.prompt || draft.prompt, {
                Picture: new Set(shot.reference_images.map((asset) => asset.slot + 1)),
                Audio: new Set(shot.reference_audios.map((asset) => asset.slot + 1)),
                Video: new Set(shot.reference_videos.map((asset) => asset.slot + 1)),
              }),
        );
        if (stale.length)
          errors.push(`R2V 提示词引用了不存在的素材标签：${[...new Set(stale)].join("、")}`);
      }
      break;
    case "v2v":
      if (draft.shots.some((shot) => shot.enabled && !isStableAssetReference(shot.source_video, "video"))) errors.push("每个启用的 V2V 片段都需要有效源视频");
      if (draft.shots.some((shot) => shot.source_video && !isStableAssetReference(shot.source_video, "video"))) errors.push("V2V 中存在失效素材，请重新上传");
      if (draft.shots.some((shot) => !Number.isFinite(shot.source_start_seconds) || shot.source_start_seconds < 0 || shot.source_start_seconds > 86_400 || !Number.isFinite(shot.source_duration_seconds) || shot.source_duration_seconds <= 0 || shot.source_duration_seconds > 86_400)) errors.push("源视频起点或时长超出 0–86400 秒范围");
      if (draft.shots.some((shot) => shot.source_video?.metadata && shot.source_start_seconds + shot.source_duration_seconds > shot.source_video.metadata.duration + 1e-6)) errors.push("源视频截取范围不能超过素材实际时长");
      {
        const stale = draft.shots.flatMap((shot) =>
          !shot.enabled
            ? []
            : invalidTags(shot.prompt || draft.prompt, {
                Picture: new Set(),
                Audio: new Set(),
                Video: new Set([1]),
              }),
        );
        if (stale.length)
          errors.push(`V2V 提示词只能引用源视频 <Video 1>：${[...new Set(stale)].join("、")}`);
      }
      break;
    case "rv2v":
      if (draft.shots.some((shot) => shot.enabled && !isStableAssetReference(shot.source_video, "video"))) errors.push("每个启用的 RV2V 片段都需要有效源视频");
      if (draft.shots.some((shot) => shot.source_video && !isStableAssetReference(shot.source_video, "video"))) errors.push("RV2V 中存在失效源视频，请重新上传");
      if (draft.shots.some((shot) => !Number.isFinite(shot.source_start_seconds) || shot.source_start_seconds < 0 || shot.source_start_seconds > 86_400 || !Number.isFinite(shot.source_duration_seconds) || shot.source_duration_seconds <= 0 || shot.source_duration_seconds > 86_400)) errors.push("源视频起点或时长超出 0–86400 秒范围");
      if (draft.shots.some((shot) => shot.source_video?.metadata && shot.source_start_seconds + shot.source_duration_seconds > shot.source_video.metadata.duration + 1e-6)) errors.push("源视频截取范围不能超过素材实际时长");
      if (draft.shots.some((shot) => shot.reference_images.some((asset) => !isStableSlottedAssetReference(asset, "image", maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages))) || shot.reference_audios.some((asset) => !isStableSlottedAssetReference(asset, "audio", maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios))))) errors.push("RV2V 中存在失效素材，请重新上传");
      if (draft.shots.some((shot) => new Set(shot.reference_images.map((asset) => asset.slot)).size !== shot.reference_images.length || new Set(shot.reference_audios.map((asset) => asset.slot)).size !== shot.reference_audios.length)) errors.push("RV2V 同类参考素材的槽位必须唯一");
      if (draft.shots.some((shot) => shot.enabled && (!hasDenseSlots(shot.reference_images) || !hasDenseSlots(shot.reference_audios)))) errors.push("RV2V 各类参考素材槽位必须连续为 0..N-1");
      {
        const stale = draft.shots.flatMap((shot) =>
          !shot.enabled
            ? []
            : invalidTags(shot.prompt || draft.prompt, {
                Picture: new Set(shot.reference_images.map((asset) => asset.slot + 1)),
                Audio: new Set(shot.reference_audios.map((asset) => asset.slot + 1)),
                Video: new Set([1]),
              }),
        );
        if (stale.length)
          errors.push(`RV2V 提示词引用了不存在的素材标签：${[...new Set(stale)].join("、")}`);
      }
      break;
  }
  return errors;
}

function patchCommon(draft: ModeDraft, patch: Partial<Pick<ModeDraft, "prompt" | "ref_image_size" | "render" | "sampling">>): ModeDraft {
  return { ...draft, ...patch } as ModeDraft;
}

export function ModeWorkspace({ formId = "mode-workspace-form", globalSettingsId = "workspace-global-settings", globalSettingsOpen = true, onGlobalSettingsClose, draft, settings, capabilities, models = EMPTY_MODELS, runtimeModelSaving = false, onChange, onRuntimeModelChange, onAssetsChange = () => undefined, onSubmit }: ModeWorkspaceProps) {
  const meta = MODE_META[draft.mode];
  const errors = validateModeDraft(draft);
  const runtimeConfigured = isConfiguredComfyUrl(settings.comfy_url);
  const runtimeReady = runtimeConfigured && capabilities.connection === "online";
  const supported = runtimeReady && capabilities.supported_modes.includes(draft.mode);
  const family: DiffusionModelRole = ["t2v", "i2v", "fl2v"].includes(draft.mode) ? "fl2va" : "ref2va";
  const diffusion = settings.models[family];
  const familyLabel = family === "fl2va" ? "FL2VA（T2V / I2V / FL2V）" : "Ref2VA（R2V / V2V / RV2V）";
  const diffusionOptions = diffusion.filename && !models[family].includes(diffusion.filename)
    ? [diffusion.filename, ...models[family]]
    : models[family];
  const loraOptions = diffusion.lora_name && !models.loras.includes(diffusion.lora_name)
    ? [diffusion.lora_name, ...models.loras]
    : models.loras;
  const submit = (event: FormEvent) => { event.preventDefault(); if (!errors.length && supported) onSubmit(); };

  return (
    <form id={formId} aria-label={`${meta.shortLabel} 模式工作区`} className={`workspace mode-accent--${meta.accent}`} onSubmit={submit}>
      {!supported && (
        <div className="notice notice--error">
          {capabilities.connection === "offline"
            ? "ComfyUI 当前离线；可以继续编辑草稿，但暂时不能提交任务。"
            : !runtimeConfigured
              ? "尚未配置 ComfyUI；请先到系统设置填写地址。草稿编辑仍可使用。"
            : capabilities.connection === "checking" || capabilities.connection === "unknown"
              ? "正在检查 ComfyUI 能力，完成后才能提交任务。"
              : `当前 ComfyUI 缺少 ${meta.shortLabel} 所需节点，请先检查能力页。`}
        </div>
      )}
      {errors.length > 0 && <div className="notice"><strong>提交前还需完成：</strong>{errors.join("；")}</div>}

      <section
        id={globalSettingsId}
        className="global-settings"
        aria-label="当前模式全局设置"
        hidden={!globalSettingsOpen}
      >
        <header className="global-settings__popover-head">
          <div><strong>当前模式全局设置</strong></div>
          {onGlobalSettingsClose && <button type="button" className="icon-button" aria-label="关闭全局设置" onClick={onGlobalSettingsClose}>×</button>}
        </header>
        <div className="global-settings__body">
          <section className="runtime-quickbar" aria-label="共享模型设置">
            <div className="runtime-quickbar__scope"><span className="eyebrow">共享模型设置</span><strong>{familyLabel}</strong><small>同模型族页面共享；参考图尺寸仍只属于当前模式</small>{runtimeModelSaving && <Spinner label="同步共享模型设置" />}</div>
            <Field label="Diffusion 模型"><select aria-label="Diffusion 模型快捷选择" disabled={!runtimeReady || runtimeModelSaving || !onRuntimeModelChange} value={diffusion.filename} onChange={(event) => onRuntimeModelChange?.(family, { filename: event.target.value })}>{diffusionOptions.map((filename) => <option key={filename} value={filename}>{filename}</option>)}</select></Field>
            <Field label="LoRA 模型"><select aria-label="LoRA 模型快捷选择" disabled={!runtimeReady || runtimeModelSaving || !onRuntimeModelChange} value={diffusion.lora_name ?? ""} onChange={(event) => onRuntimeModelChange?.(family, { lora_name: event.target.value || null })}><option value="">不使用 LoRA</option>{loraOptions.map((filename) => <option key={filename} value={filename}>{filename}</option>)}</select></Field>
            <div className="fixed-runtime-value" aria-label="LoRA 加载状态"><span>加载方式</span><strong>{describeLoraLoader(diffusion)}</strong><small>显式选择仅用于 Standard 自动探测无法确认时</small></div>
          </section>

          <div className="workspace-grid">
            <Panel eyebrow="创意 01" title="默认提示词" description="默认提示词，镜头非空时覆盖。" className="prompt-panel">
              <Field label="默认提示词" hint="作为所有镜头的默认值；镜头提示词非空时覆盖">
                <textarea aria-label="默认提示词" rows={8} value={draft.prompt} placeholder="例如：雨夜的霓虹街口，镜头低机位缓慢推进…"
                  onChange={(event) => onChange(patchCommon(draft, { prompt: limitPromptCharacters(event.target.value) }))} />
              </Field>
            </Panel>

            <div className="settings-stack">
              <Panel eyebrow="画面" title="输出规格">
                <div className="aspect-presets">
                  {[[864,480,"16:9"],[480,864,"9:16"],[640,640,"1:1"]].map(([width,height,label]) => <button type="button" key={label} className={draft.render.width === width && draft.render.height === height ? "is-active" : ""}
                    onClick={() => onChange(patchCommon(draft, { render: { ...draft.render, width: Number(width), height: Number(height) } }))}>{label}</button>)}
                </div>
                <div className="field-grid field-grid--three">
                  <Field label="宽度"><DeferredNumberInput min="32" max="8192" step="32" value={draft.render.width} normalizeValue={Math.trunc} onValueCommit={(value) => onChange(patchCommon(draft, { render: { ...draft.render, width: value } }))} /></Field>
                  <Field label="高度"><DeferredNumberInput min="32" max="8192" step="32" value={draft.render.height} normalizeValue={Math.trunc} onValueCommit={(value) => onChange(patchCommon(draft, { render: { ...draft.render, height: value } }))} /></Field>
                  <Field label="帧率"><div className="fixed-runtime-value" aria-label="帧率"><strong>24 fps</strong><small>H3 原生固定值</small></div></Field>
                </div>
              </Panel>
              <Panel eyebrow="采样" title="推理参数">
                <div className="field-grid field-grid--three">
                  <Field label="步数"><DeferredNumberInput min="1" max="200" step="1" value={draft.sampling.steps} normalizeValue={Math.trunc} onValueCommit={(value) => onChange(patchCommon(draft, { sampling: { ...draft.sampling, steps: value } }))} /></Field>
                  <div className="field"><span className="field__label">Seed</span><div className="seed-input"><DeferredNumberInput aria-label="Seed" min="0" max={Number.MAX_SAFE_INTEGER} step="1" disabled={draft.sampling.random_seed} value={draft.sampling.seed} normalizeValue={Math.trunc} onValueCommit={(value) => onChange(patchCommon(draft, { sampling: { ...draft.sampling, seed: value } }))} /><label><input aria-label="随机种子" type="checkbox" checked={draft.sampling.random_seed} onChange={(event) => onChange(patchCommon(draft, { sampling: { ...draft.sampling, random_seed: event.target.checked, ...(event.target.checked ? { seed: randomSafeSeed() } : {}) } }))} /><span />随机</label></div><small>{draft.sampling.random_seed ? "提交时重掷；输入框显示下一次使用的实际数值" : "固定种子可复现同一采样起点"}</small></div>
                  <Field label="采样器"><select value={draft.sampling.sampler} onChange={(event) => onChange(patchCommon(draft, { sampling: { ...draft.sampling, sampler: event.target.value as typeof draft.sampling.sampler } }))}><option value="res_multistep">res_multistep</option><option value="euler">euler</option><option value="dpmpp_2m">dpmpp_2m</option></select></Field>
                  <Field label="调度器"><select value={draft.sampling.scheduler} onChange={(event) => onChange(patchCommon(draft, { sampling: { ...draft.sampling, scheduler: event.target.value as typeof draft.sampling.scheduler } }))}>{SAMPLING_SCHEDULERS.map((scheduler) => <option value={scheduler} key={scheduler}>{scheduler}</option>)}</select></Field>
                  <Field label="Shift / Audio"><div className="inline-inputs"><DeferredNumberInput aria-label="Shift" min="0.01" max="100" step="0.01" value={draft.sampling.shift} onValueCommit={(value) => onChange(patchCommon(draft, { sampling: { ...draft.sampling, shift: value } }))} /><DeferredNumberInput aria-label="Audio Shift" min="0.01" max="100" step="0.01" value={draft.sampling.audio_shift} onValueCommit={(value) => onChange(patchCommon(draft, { sampling: { ...draft.sampling, audio_shift: value } }))} /></div></Field>
                  <Field label="参考图采样尺寸" hint={draft.ref_image_size === "max" ? "max：最高身份保真，速度与显存开销更高" : "match：匹配生成像素面积，默认且更快"}><select aria-label="参考图采样尺寸" value={draft.ref_image_size} onChange={(event) => onChange(patchCommon(draft, { ref_image_size: event.target.value as ModeDraft["ref_image_size"] }))}><option value="match">match（匹配输出面积）</option><option value="max">max（最高参考质量）</option></select></Field>
                </div>
              </Panel>
            </div>
          </div>
        </div>
      </section>
      <TimelineEditor draft={draft} runtimeEnabled={runtimeReady} onChange={onChange} onAssetsChange={onAssetsChange} />
    </form>
  );
}
