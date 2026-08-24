import type { ModelInventory } from "../api/types";
import {
  closestTimelineOutputResolution,
  inferTimelineOutputAspect,
  isTimelineOutputResolution,
  timelineOutputResolutions,
  loraFeatureSelection,
  TIMELINE_OUTPUT_ASPECTS,
  type LoraFamilySelection,
  type ModelStack,
  type TimelineOutputAspect,
  type TimelineGenerationMode,
  type TimelineProject,
} from "../domain/timelineProject";
import { randomSafeSeed, SAMPLING_SCHEDULERS, type SamplingConfig } from "../domain/modes";
import { useTranslator } from "../i18n";
import { DeferredNumberInput, Field } from "./ui";
import {
  ComfyKitchenAttentionField,
  type ComfyKitchenAttentionUiCapability,
} from "./ComfyKitchenAttentionField";

interface TimelineGlobalSettingsProps {
  id: string;
  open: boolean;
  project: TimelineProject;
  models: ModelInventory;
  runtimeReady: boolean;
  comfyKitchenAttentionCapability: ComfyKitchenAttentionUiCapability;
  onClose: () => void;
  onProjectPatch: (
    patch: Partial<Pick<TimelineProject, "render" | "export_mode">>,
  ) => void;
  onSamplingChange: (
    family: keyof TimelineProject["sampling"],
    patch: Partial<SamplingConfig>,
  ) => void;
  onModelChange: (role: keyof ModelStack, filename: string | null) => void;
  onLoraChange: (
    family: TimelineGenerationMode,
    patch: Partial<LoraFamilySelection>,
  ) => void;
  onComfyKitchenAttentionChange: (enabled: boolean) => void;
}

function modelOptions(
  selected: string | null,
  inventory: string[],
): string[] {
  return selected && !inventory.includes(selected) ? [selected, ...inventory] : inventory;
}

function GlobalOutputSpecs({
  project,
  onProjectPatch,
}: Pick<TimelineGlobalSettingsProps, "project" | "onProjectPatch">) {
  const { t } = useTranslator();
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
    onProjectPatch({ render: { ...project.render, ...resolution, fps: 24 } });
  };
  const changeResolution = (value: string) => {
    const resolution = resolutions.find(
      (candidate) => `${candidate.width}x${candidate.height}` === value,
    );
    if (resolution) onProjectPatch({
      render: { ...project.render, ...resolution, fps: 24 },
    });
  };
  return (
    <section className="timeline-project-specs" aria-label={t("globalSettings.output.ariaLabel")}>
      <header className="timeline-project-specs__head">
        <strong>{t("globalSettings.output.title")}</strong><small>{t("globalSettings.shared")}</small>
      </header>
      <div className="timeline-project-specs__fields timeline-project-specs__fields--global">
        <Field label={t("globalSettings.output.aspect")} className="field--inline">
          <select aria-label={t("globalSettings.output.aspect")} value={aspect ?? "custom"} onChange={(event) => changeAspect(event.target.value as TimelineOutputAspect)}>
            {!aspect && <option value="custom" disabled>{t("globalSettings.output.customLegacy")}</option>}
            {TIMELINE_OUTPUT_ASPECTS.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </Field>
        <Field label={t("globalSettings.output.resolution")} className="field--inline">
          <select aria-label={t("globalSettings.output.resolution")} value={resolutionValue} onChange={(event) => changeResolution(event.target.value)}>
            {!nativeResolution && <option value={resolutionValue}>{t("globalSettings.output.customResolution", { width: project.render.width, height: project.render.height })}</option>}
            {resolutions.map(({ width, height }) => <option key={`${width}x${height}`} value={`${width}x${height}`}>{width} × {height}</option>)}
          </select>
        </Field>
        <Field label={t("globalSettings.output.exportMode")} className="field--inline">
          <select aria-label={t("globalSettings.output.exportMode")} value={project.export_mode} onChange={(event) => onProjectPatch({ export_mode: event.target.value as TimelineProject["export_mode"] })}><option value="all">{t("globalSettings.output.exportAll")}</option><option value="segments">{t("globalSettings.output.exportSegments")}</option></select>
        </Field>
      </div>
    </section>
  );
}

export function TimelineGlobalSettings({
  id,
  open,
  project,
  models,
  runtimeReady,
  comfyKitchenAttentionCapability,
  onClose,
  onProjectPatch,
  onSamplingChange,
  onModelChange,
  onLoraChange,
  onComfyKitchenAttentionChange,
}: TimelineGlobalSettingsProps) {
  const { t } = useTranslator();
  const updateSampling = (
    role: keyof TimelineProject["sampling"],
    patch: Partial<SamplingConfig>,
  ) => onSamplingChange(role, patch);

  return (
    <section id={id} className="timeline-global-settings" aria-label={t("globalSettings.ariaLabel")} hidden={!open}>
      <header className="timeline-global-settings__head">
        <strong>{t("globalSettings.title")}</strong>
        <button type="button" className="icon-button" aria-label={t("globalSettings.closeLabel")} onClick={onClose}>×</button>
      </header>

      <div className="timeline-global-settings__body">
        <GlobalOutputSpecs project={project} onProjectPatch={onProjectPatch} />
        <ComfyKitchenAttentionField
          id={`${id}-ck-attention`}
          project={project}
          capability={comfyKitchenAttentionCapability}
          onChange={onComfyKitchenAttentionChange}
        />
        {(["fl2va", "ref2va"] as const).map((role) => {
          const modelSelection = project.model_stack[role];
          const lora = loraFeatureSelection(project.features).params.by_family[role];
          const sampling = project.sampling[role];
          const label = role === "fl2va" ? "FL2VA" : "Ref2VA";
          const familyDescription = t(`globalSettings.family.${role}.description`);
          return (
            <section className="timeline-family-settings" aria-labelledby={`${id}-${role}-title`} key={role}>
              <header>
                <div className="timeline-family-settings__title"><strong id={`${id}-${role}-title`}>{label}</strong><small>{familyDescription}</small></div>
                <div className="timeline-family-settings__models" data-timeline-history-ignore>
                  <Field label={t("globalSettings.family.diffusionModel")} className="field--inline"><select aria-label={t("globalSettings.family.diffusionSelectLabel", { family: role.toUpperCase() })} disabled={!runtimeReady} value={modelSelection.filename ?? ""} onChange={(event) => onModelChange(role, event.target.value || null)}><option value="">{t("globalSettings.model.unbound")}</option>{modelOptions(modelSelection.filename, models[role]).map((filename) => <option value={filename} key={filename}>{filename}</option>)}</select></Field>
                  <Field label={t("globalSettings.family.lora")} className="field--inline"><select aria-label={t("globalSettings.family.loraSelectLabel", { family: role.toUpperCase() })} disabled={!runtimeReady} value={lora.enabled ? lora.filename ?? "" : ""} onChange={(event) => onLoraChange(role, { enabled: Boolean(event.target.value), filename: event.target.value || null })}><option value="">{t("globalSettings.family.noLora")}</option>{modelOptions(lora.filename, models.loras).map((filename) => <option value={filename} key={filename}>{filename}</option>)}</select></Field>
                  <Field label={t("globalSettings.family.loraStrength")} className="field--inline"><DeferredNumberInput aria-label={t("globalSettings.family.loraStrengthLabel", { family: label })} min="-10" max="10" step="0.01" disabled={!runtimeReady || !lora.enabled || !lora.filename} value={lora.strength} onValueCommit={(strength) => onLoraChange(role, { strength })} /></Field>
                </div>
              </header>
              <div className="timeline-family-settings__sampling">
                <div className="field-grid timeline-sampling-fields">
                <Field label={t("globalSettings.family.steps")} className="field--inline"><DeferredNumberInput aria-label={t("globalSettings.family.stepsLabel", { family: label })} min="1" max="200" step="1" value={sampling.steps} normalizeValue={Math.trunc} onValueCommit={(value) => updateSampling(role, { steps: value })} /></Field>
                <Field label={t("globalSettings.family.seed")} className="field--inline"><div className="seed-input"><DeferredNumberInput aria-label={t("globalSettings.family.seedLabel", { family: label })} min="0" max={Number.MAX_SAFE_INTEGER} step="1" disabled={sampling.random_seed} value={sampling.seed} normalizeValue={Math.trunc} onValueCommit={(value) => updateSampling(role, { seed: value })} /><label><input aria-label={t("globalSettings.family.randomSeedLabel", { family: label })} type="checkbox" checked={sampling.random_seed} onChange={(event) => updateSampling(role, { random_seed: event.target.checked, ...(event.target.checked ? { seed: randomSafeSeed() } : {}) })} /><span />{t("globalSettings.family.random")}</label></div></Field>
                <Field label={t("globalSettings.family.sampler")} className="field--inline"><select aria-label={t("globalSettings.family.samplerLabel", { family: label })} value={sampling.sampler} onChange={(event) => updateSampling(role, { sampler: event.target.value as SamplingConfig["sampler"] })}><option value="res_multistep">res_multistep</option><option value="euler">euler</option><option value="dpmpp_2m">dpmpp_2m</option></select></Field>
                <Field label={t("globalSettings.family.scheduler")} className="field--inline"><select aria-label={t("globalSettings.family.schedulerLabel", { family: label })} value={sampling.scheduler} onChange={(event) => updateSampling(role, { scheduler: event.target.value as SamplingConfig["scheduler"] })}>{SAMPLING_SCHEDULERS.map((scheduler) => <option value={scheduler} key={scheduler}>{scheduler}</option>)}</select></Field>
                <Field label={t("globalSettings.family.videoShift")} className="field--inline"><DeferredNumberInput aria-label={t("globalSettings.family.videoShiftLabel", { family: label })} min="0.01" max="100" step="0.01" value={sampling.shift} onValueCommit={(value) => updateSampling(role, { shift: value })} /></Field>
                <Field label={t("globalSettings.family.audioShift")} className="field--inline"><DeferredNumberInput aria-label={t("globalSettings.family.audioShiftLabel", { family: label })} min="0.01" max="100" step="0.01" value={sampling.audio_shift} onValueCommit={(value) => updateSampling(role, { audio_shift: value })} /></Field>
                </div>
              </div>
            </section>
          );
        })}
        <section id={`${id}-codec-models`} tabIndex={-1} className="timeline-family-settings timeline-family-settings--codec" aria-labelledby={`${id}-codec-models-title`}>
          <header>
            <div className="timeline-family-settings__title"><strong id={`${id}-codec-models-title`}>{t("globalSettings.codec.title")}</strong></div>
            <div className="timeline-family-settings__models" data-timeline-history-ignore>
              {(["clip", "video_vae", "audio_vae"] as const).map((role) => {
                const labels = { clip: "CLIP", video_vae: "Video VAE", audio_vae: "Audio VAE" } as const;
                const selected = project.model_stack[role].filename;
                return <Field label={labels[role]} className="field--inline" key={role}><select aria-label={t("globalSettings.codec.selectLabel", { model: labels[role] })} disabled={!runtimeReady} value={selected ?? ""} onChange={(event) => onModelChange(role, event.target.value || null)}><option value="">{t("globalSettings.model.unbound")}</option>{modelOptions(selected, models[role]).map((filename) => <option value={filename} key={filename}>{filename}</option>)}</select></Field>;
              })}
            </div>
          </header>
        </section>
      </div>
    </section>
  );
}
