import type {
  DiffusionModelBinding,
  DiffusionModelRole,
  ModelInventory,
  RuntimeSettings,
} from "../api/types";
import { describeLoraLoader } from "../api/types";
import {
  closestTimelineOutputResolution,
  inferTimelineOutputAspect,
  isTimelineOutputResolution,
  timelineOutputResolutions,
  type TimelineOutputAspect,
  type TimelineProject,
} from "../domain/timelineProject";
import { randomSafeSeed, SAMPLING_SCHEDULERS, type SamplingConfig } from "../domain/modes";
import { DeferredNumberInput, Field, Spinner } from "./ui";

interface TimelineGlobalSettingsProps {
  id: string;
  open: boolean;
  project: TimelineProject;
  settings: RuntimeSettings;
  models: ModelInventory;
  runtimeReady: boolean;
  modelSaving: boolean;
  onClose: () => void;
  onChange: (project: TimelineProject) => void;
  onRuntimeModelChange: (
    role: DiffusionModelRole,
    patch: Partial<DiffusionModelBinding>,
  ) => void;
}

function modelOptions(
  selected: string | null,
  inventory: string[],
): string[] {
  return selected && !inventory.includes(selected) ? [selected, ...inventory] : inventory;
}

function GlobalOutputSpecs({
  project,
  onChange,
}: Pick<TimelineGlobalSettingsProps, "project" | "onChange">) {
  const aspect = inferTimelineOutputAspect(project.render.width, project.render.height);
  const resolutions = aspect ? timelineOutputResolutions(aspect) : [];
  const nativeResolution = isTimelineOutputResolution(
    project.render.width,
    project.render.height,
    aspect,
  );
  const resolutionValue = nativeResolution
    ? `${project.render.width}x${project.render.height}`
    : `custom:${project.render.width}x${project.render.height}`;
  const changeAspect = (nextAspect: TimelineOutputAspect) => {
    const resolution = closestTimelineOutputResolution(
      project.render.width,
      project.render.height,
      nextAspect,
    );
    onChange({ ...project, render: { ...project.render, ...resolution, fps: 24 } });
  };
  const changeResolution = (value: string) => {
    const resolution = resolutions.find(
      (candidate) => `${candidate.width}x${candidate.height}` === value,
    );
    if (resolution) onChange({
      ...project,
      render: { ...project.render, ...resolution, fps: 24 },
    });
  };
  return (
    <section className="timeline-project-specs" aria-label="输出规格">
      <header className="timeline-project-specs__head">
        <strong>输出规格</strong><small>全片共用</small>
      </header>
      <div className="timeline-project-specs__fields timeline-project-specs__fields--global">
        <Field label="画幅" className="field--inline">
          <select aria-label="画幅" value={aspect ?? "custom"} onChange={(event) => changeAspect(event.target.value as TimelineOutputAspect)}>
            {!aspect && <option value="custom" disabled>自定义（旧项目）</option>}
            <option value="16:9">16:9</option><option value="9:16">9:16</option>
          </select>
        </Field>
        <Field label="分辨率" className="field--inline">
          <select aria-label="分辨率" value={resolutionValue} onChange={(event) => changeResolution(event.target.value)}>
            {!nativeResolution && <option value={resolutionValue}>{project.render.width} × {project.render.height}（自定义）</option>}
            {resolutions.map(({ width, height }) => <option key={`${width}x${height}`} value={`${width}x${height}`}>{width} × {height}</option>)}
          </select>
        </Field>
        <Field label="导出方式" className="field--inline">
          <select aria-label="导出方式" value={project.export_mode} onChange={(event) => onChange({ ...project, export_mode: event.target.value as TimelineProject["export_mode"] })}><option value="all">组装完整视频</option><option value="segments">输出独立片段</option></select>
        </Field>
      </div>
    </section>
  );
}

export function TimelineGlobalSettings({
  id,
  open,
  project,
  settings,
  models,
  runtimeReady,
  modelSaving,
  onClose,
  onChange,
  onRuntimeModelChange,
}: TimelineGlobalSettingsProps) {
  const updateSampling = (
    role: keyof TimelineProject["sampling"],
    patch: Partial<SamplingConfig>,
  ) => onChange({
    ...project,
    sampling: {
      ...project.sampling,
      [role]: { ...project.sampling[role], ...patch },
    },
  });

  return (
    <section id={id} className="timeline-global-settings" aria-label="时间线全局设置" hidden={!open}>
      <header className="timeline-global-settings__head">
        <div><strong>长视频全局设置</strong><small>集中设置输出规格、模型与推理参数</small></div>
        <button type="button" className="icon-button" aria-label="关闭全局设置" onClick={onClose}>×</button>
      </header>

      <div className="timeline-global-settings__body">
        <GlobalOutputSpecs project={project} onChange={onChange} />
        {(["fl2va", "ref2va"] as const).map((role) => {
          const binding = settings.models[role];
          const sampling = project.sampling[role];
          const label = role === "fl2va" ? "FL2VA" : "Ref2VA";
          return (
            <section className="timeline-family-settings" aria-labelledby={`${id}-${role}-title`} key={role}>
              <header>
                <div className="timeline-family-settings__title"><strong id={`${id}-${role}-title`}>{label}</strong><small>{role === "fl2va" ? "文 / 图 / 首尾帧生成" : "参考 / 源视频生成"}</small><small>LoRA 加载：{describeLoraLoader(binding)}</small>{modelSaving && <Spinner label={`同步 ${label} 模型`} />}</div>
                <div className="timeline-family-settings__models">
                  <Field label="Diffusion 模型" className="field--inline"><select aria-label={`${role.toUpperCase()} Diffusion 模型快捷选择`} disabled={!runtimeReady} value={binding.filename} onChange={(event) => onRuntimeModelChange(role, { filename: event.target.value })}>{modelOptions(binding.filename, models[role]).map((filename) => <option value={filename} key={filename}>{filename}</option>)}</select></Field>
                  <Field label="LoRA" className="field--inline"><select aria-label={`${role.toUpperCase()} LoRA 模型快捷选择`} disabled={!runtimeReady} value={binding.lora_name ?? ""} onChange={(event) => onRuntimeModelChange(role, { lora_name: event.target.value || null })}><option value="">不使用 LoRA</option>{modelOptions(binding.lora_name, models.loras).map((filename) => <option value={filename} key={filename}>{filename}</option>)}</select></Field>
                </div>
              </header>
              <div className="timeline-family-settings__sampling">
                <div className="field-grid field-grid--two timeline-sampling-fields">
                <Field label="步数" className="field--inline"><DeferredNumberInput aria-label={`${label} 步数`} min="1" max="200" step="1" value={sampling.steps} normalizeValue={Math.trunc} onValueCommit={(value) => updateSampling(role, { steps: value })} /></Field>
                <Field label="Seed" className="field--inline"><div className="seed-input"><DeferredNumberInput aria-label={`${label} Seed`} min="0" max={Number.MAX_SAFE_INTEGER} step="1" disabled={sampling.random_seed} value={sampling.seed} normalizeValue={Math.trunc} onValueCommit={(value) => updateSampling(role, { seed: value })} /><label><input aria-label={`${label} 随机 Seed`} type="checkbox" checked={sampling.random_seed} onChange={(event) => updateSampling(role, { random_seed: event.target.checked, ...(event.target.checked ? { seed: randomSafeSeed() } : {}) })} /><span />随机</label></div></Field>
                <Field label="采样器" className="field--inline"><select aria-label={`${label} 采样器`} value={sampling.sampler} onChange={(event) => updateSampling(role, { sampler: event.target.value as SamplingConfig["sampler"] })}><option value="res_multistep">res_multistep</option><option value="euler">euler</option><option value="dpmpp_2m">dpmpp_2m</option></select></Field>
                <Field label="调度器" className="field--inline"><select aria-label={`${label} 调度器`} value={sampling.scheduler} onChange={(event) => updateSampling(role, { scheduler: event.target.value as SamplingConfig["scheduler"] })}>{SAMPLING_SCHEDULERS.map((scheduler) => <option value={scheduler} key={scheduler}>{scheduler}</option>)}</select></Field>
                <Field label="Video Shift" className="field--inline"><DeferredNumberInput aria-label={`${label} Video Shift`} min="0.01" max="100" step="0.01" value={sampling.shift} onValueCommit={(value) => updateSampling(role, { shift: value })} /></Field>
                <Field label="Audio Shift" className="field--inline"><DeferredNumberInput aria-label={`${label} Audio Shift`} min="0.01" max="100" step="0.01" value={sampling.audio_shift} onValueCommit={(value) => updateSampling(role, { audio_shift: value })} /></Field>
                </div>
              </div>
            </section>
          );
        })}
      </div>
    </section>
  );
}
