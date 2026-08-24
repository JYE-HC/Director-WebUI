import type { TimelineCompileReport } from "../api/types";
import type { TimelineSegment } from "../domain/timelineProject";
import { useTranslator, type LocalizedProblem } from "../i18n";

interface PreflightResultPanelProps {
  report: TimelineCompileReport | null;
  failure: LocalizedProblem | null;
  segments: TimelineSegment[];
  onClose: () => void;
  onOpenGlobalSettings: () => void;
}

function joined(parts: Array<string | null>): string {
  return parts.filter((part): part is string => Boolean(part)).join(" · ");
}

export function PreflightResultPanel({
  report,
  failure,
  segments,
  onClose,
  onOpenGlobalSettings,
}: PreflightResultPanelProps) {
  const translator = useTranslator();
  const t = translator.t;

  if (failure) {
    const technical = failure.technicalMessage && failure.technicalMessage !== failure.message
      ? failure.technicalMessage
      : null;
    return (
      <section
        className="execution-plan-report execution-plan-report--failure"
        role="alert"
        aria-label={t("preflight.failure.title")}
      >
        <header>
          <strong>{t("preflight.failure.title")}</strong>
          <button type="button" className="icon-button" aria-label={t("preflight.failure.closeLabel")} onClick={onClose}>×</button>
        </header>
        <div className="execution-plan-report__failure-body">
          <strong>{failure.message}</strong>
          <p>{failure.remediation}</p>
          <div className="execution-plan-report__failure-actions">
            {failure.action === "open_global_settings" && (
              <button type="button" className="button button--primary" onClick={onOpenGlobalSettings}>
                {t("preflight.failure.openGlobalSettings")}
              </button>
            )}
            {(technical || failure.code !== "unknown") && (
              <details>
                <summary>{t("preflight.failure.technicalDetails")}</summary>
                <code>{t("common.errorCode", { code: failure.code })}</code>
                {technical && <pre>{technical}</pre>}
              </details>
            )}
          </div>
        </div>
      </section>
    );
  }

  if (!report) return null;
  return (
    <section className="execution-plan-report" aria-label={t("preflight.success.ariaLabel")}>
      <header>
        <div>
          <strong>{t("preflight.success.title")}</strong>
          <small>{t("preflight.success.subtitle", { strategy: report.execution_strategy })}</small>
        </div>
        <button type="button" className="icon-button" aria-label={t("preflight.success.closeLabel")} onClick={onClose}>×</button>
      </header>
      <div className="execution-plan-report__summary">
        <span>{t("preflight.success.planCount", { count: report.plans.length })}</span>
        <span>{report.model_families.map((family) => family.toUpperCase()).join(" + ")}</span>
        <span>{t("preflight.success.graphSource", { source: t(report.node_policy.graph_source === "server" ? "preflight.success.serverSource" : "preflight.success.unknownSource") })}</span>
        <span>{t("preflight.success.clientWorkflow", { state: t(report.node_policy.accepts_client_workflow ? "preflight.success.allowed" : "preflight.success.denied") })}</span>
      </div>
      <div className="execution-plan-report__plans">
        {report.plans.map((plan) => {
          const targetIndex = segments.findIndex((segment) => segment.id === plan.segment_id);
          const targetLabel = targetIndex >= 0 ? `${targetIndex + 1} · ${segments[targetIndex].title}` : plan.segment_id;
          const predecessorIndex = plan.predecessor_segment_id === null
            ? -1
            : segments.findIndex((segment) => segment.id === plan.predecessor_segment_id);
          const predecessorLabel = predecessorIndex >= 0
            ? `${predecessorIndex + 1} · ${segments[predecessorIndex].title}`
            : plan.predecessor_segment_id;
          const continuity = predecessorLabel
            ? joined([
              t(plan.continuity_source === "historical_take" ? "preflight.success.historicalSource" : "preflight.success.sameRunSource"),
              t("preflight.success.predecessor", { predecessor: predecessorLabel }),
              t("preflight.success.contextFrames", { frames: plan.continuity_context_frames }),
              t("preflight.success.alignmentFrames", { frames: plan.alignment_tail_frame_count }),
              plan.historical_take_id ? t("preflight.success.take", { take: plan.historical_take_id.slice(0, 8) }) : null,
            ])
            : joined([
              t("preflight.success.noContinuity"),
              t(plan.anchor_reset ? "preflight.success.anchorReset" : "preflight.success.noPredecessor"),
              t("preflight.success.alignmentFrames", { frames: plan.alignment_tail_frame_count }),
            ]);
          return (
            <article key={plan.segment_id} aria-label={t("preflight.success.segmentPlanLabel", { target: targetLabel })}>
              <header>
                <strong title={targetLabel}>{joined([
                  targetLabel,
                  plan.mode === "fl2va" ? "FL2VA" : "Ref2VA",
                  t("preflight.success.derivedRecipe", { recipe: t(`generationModes.${plan.recipe}.label`) }),
                ])}</strong>
                <em>{plan.backend === "raylight" ? "RayLight" : t("preflight.success.standardBackend")}</em>
              </header>
              <span>{t("preflight.success.frameSummary", {
                visible: plan.visible_frame_count,
                sample: plan.sample_frame_count,
                seed: plan.seed,
                seedMode: t(plan.seed_mode === "random" ? "preflight.success.randomSeed" : "preflight.success.fixedSeed"),
              })}</span>
              <small>{continuity}</small>
              <small>{plan.conditioning_node}</small>
              <details><summary>{t("preflight.success.nodeCount", { count: plan.node_classes.length })}</summary><code>{plan.node_classes.join(" → ")}</code></details>
            </article>
          );
        })}
      </div>
      <footer>
        <span>{t("preflight.success.allowedNodeCount", { count: report.node_policy.allowed_nodes.length })}</span>
        <span>{t("preflight.success.customNodeCount", { count: report.node_policy.custom_nodes.length })}</span>
        {report.node_policy.custom_nodes.length > 0 && <code>{report.node_policy.custom_nodes.join("、")}</code>}
      </footer>
    </section>
  );
}
