import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { TimelineCompileReport } from "../api/types";
import { PreflightResultPanel } from "../components/PreflightResultPanel";
import { createTimelineProject } from "../domain/timelineProject";
import type { LocalizedProblem } from "../i18n";

function emptyReport(): TimelineCompileReport {
  const project = createTimelineProject();
  return {
    template_bundle_version: project.features.template_bundle_version,
    host_capability_revision: `sha256:${"a".repeat(64)}`,
    execution_strategy: "native_segment_graph_v1",
    model_families: [],
    plans: [],
    node_policy: {
      graph_source: "server",
      accepts_client_workflow: false,
      allowed_nodes: [],
      custom_nodes: [],
      provenance: {},
    },
    features: {
      requested: project.features,
      effective_by_segment: {},
      resolutions: [],
      notices: [],
    },
    effective_execution_digest: {
      algorithm: "sha256-canonical-json-v1",
      value: `sha256-${"b".repeat(64)}`,
    },
  };
}

const failure: LocalizedProblem = {
  code: "model_binding_required",
  message: "当前项目缺少编码模型（CLIP）。",
  remediation: "请打开全局设置完成模型选择。",
  technicalMessage: "safe technical fallback",
  action: "open_global_settings",
};

describe("PreflightResultPanel", () => {
  it("失败结果使用持久 alert、可读操作和折叠技术详情", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onOpenGlobalSettings = vi.fn();
    render(
      <PreflightResultPanel
        report={null}
        failure={failure}
        segments={[]}
        onClose={onClose}
        onOpenGlobalSettings={onOpenGlobalSettings}
      />,
    );

    const alert = screen.getByRole("alert", { name: "执行计划预检未通过" });
    expect(alert).toHaveTextContent(failure.message);
    expect(alert).toHaveTextContent(failure.remediation);
    const details = screen.getByText("技术详情").closest("details");
    expect(details).not.toHaveAttribute("open");

    await user.click(screen.getByRole("button", { name: "打开全局设置" }));
    expect(onOpenGlobalSettings).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("button", { name: "关闭预检结果" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("成功结果保留具名 region，空闲时不渲染面板", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const view = render(
      <PreflightResultPanel
        report={emptyReport()}
        failure={null}
        segments={[]}
        onClose={onClose}
        onOpenGlobalSettings={() => undefined}
      />,
    );

    expect(screen.getByRole("region", { name: "服务端执行计划" }))
      .toHaveTextContent("0 个分段计划");
    await user.click(screen.getByRole("button", { name: "关闭执行计划" }));
    expect(onClose).toHaveBeenCalledTimes(1);

    view.rerender(
      <PreflightResultPanel
        report={null}
        failure={null}
        segments={[]}
        onClose={onClose}
        onOpenGlobalSettings={() => undefined}
      />,
    );
    expect(screen.queryByRole("region", { name: "服务端执行计划" })).not.toBeInTheDocument();
  });
});
