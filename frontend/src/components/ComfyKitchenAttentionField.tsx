import type { ComfyKitchenAttentionCapability } from "../api/types";
import {
  comfyKitchenAttentionFeatureSelection,
  type TimelineProject,
} from "../domain/timelineProject";
import { useTranslator, type Translator } from "../i18n";

export type ComfyKitchenAttentionUiCapability =
  | ComfyKitchenAttentionCapability
  | { state: "checking"; backend: null; reasons: [] };

interface ComfyKitchenAttentionFieldProps {
  id: string;
  project: TimelineProject;
  capability: ComfyKitchenAttentionUiCapability;
  onChange: (enabled: boolean) => void;
}

function capabilityReason(
  capability: ComfyKitchenAttentionUiCapability,
  legacyBundle: boolean,
  translator: Translator,
): string {
  if (legacyBundle) {
    return translator.t("globalSettings.ck.reason.legacyBundle");
  }
  if (capability.state === "checking") {
    return translator.t("globalSettings.ck.reason.checking");
  }
  if (capability.state === "available") {
    return translator.t("globalSettings.ck.reason.available", {
      backend: capability.backend === "raylight" ? "RayLight" : "Standard",
    });
  }
  const reason = capability.reasons[0];
  if (reason) {
    return translator.t(
      `globalSettings.ck.reasonCode.${reason.code}.${capability.state}`,
      undefined,
      translator.t(`globalSettings.ck.reason.${capability.state}`),
    );
  }
  return translator.t(`globalSettings.ck.reason.${capability.state}`);
}

export function ComfyKitchenAttentionField({
  id,
  project,
  capability,
  onChange,
}: ComfyKitchenAttentionFieldProps) {
  const translator = useTranslator();
  const checked = comfyKitchenAttentionFeatureSelection(project.features).enabled;
  const legacyBundle = project.features.template_bundle_version < 6;
  const unavailable = capability.state === "unavailable";
  const disabled = legacyBundle || (unavailable && !checked);
  const reasonId = `${id}-reason`;
  const reason = capabilityReason(capability, legacyBundle, translator);
  const state = legacyBundle ? "unavailable" : capability.state;

  return (
    <section className="timeline-project-specs timeline-ck-attention" aria-label={translator.t("globalSettings.ck.ariaLabel")}>
      <header className="timeline-project-specs__head">
        <strong>{translator.t("globalSettings.ck.title")}</strong><small>{translator.t("globalSettings.shared")}</small>
      </header>
      <div
        className="timeline-ck-attention__control"
        data-state={state}
        tabIndex={disabled ? 0 : undefined}
        aria-describedby={disabled ? reasonId : undefined}
        title={reason}
      >
        <label>
          <input
            type="checkbox"
            aria-label={translator.t("globalSettings.ck.enableLabel")}
            aria-describedby={reasonId}
            checked={checked}
            disabled={disabled}
            onChange={(event) => onChange(event.target.checked)}
          />
          <span>{translator.t("globalSettings.ck.productName")}</span>
        </label>
        <small>{translator.t(legacyBundle
          ? "globalSettings.ck.state.legacy"
          : `globalSettings.ck.state.${capability.state}`)}</small>
        <span id={reasonId} role="tooltip" className="timeline-ck-attention__tooltip">
          {reason}
        </span>
      </div>
    </section>
  );
}
