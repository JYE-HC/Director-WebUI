import { fireEvent, render, screen } from "@testing-library/react";
import { ComfyKitchenAttentionField } from "../components/ComfyKitchenAttentionField";
import { createTimelineProject } from "../domain/timelineProject";
import { I18nProvider } from "../i18n";

const available = {
  context_revision: "ctx:test-standard",
  backend: "standard" as const,
  state: "available" as const,
  reasons: [],
};

describe("ComfyKitchenAttentionField", () => {
  it("reads Bundle 6 project authority and only writes on a user toggle", () => {
    const project = createTimelineProject();
    const onChange = vi.fn();
    const view = render(
      <ComfyKitchenAttentionField
        id="ck"
        project={project}
        capability={{ state: "checking", backend: null, reasons: [] }}
        onChange={onChange}
      />,
    );
    const checkbox = screen.getByRole("checkbox", {
      name: "启用 Comfy Kitchen Attention",
    });
    expect(checkbox).toBeEnabled();
    expect(checkbox).not.toBeChecked();
    expect(checkbox).toHaveAttribute("aria-describedby", "ck-reason");
    expect(checkbox.closest(".timeline-ck-attention__control"))
      .not.toHaveAttribute("tabindex");

    view.rerender(
      <ComfyKitchenAttentionField
        id="ck"
        project={project}
        capability={available}
        onChange={onChange}
      />,
    );
    expect(onChange).not.toHaveBeenCalled();
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledOnce();
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("blocks only unavailable off-to-on and exposes its reason to hover and focus", () => {
    const reason = "raw backend reason";
    const project = createTimelineProject();
    render(
      <ComfyKitchenAttentionField
        id="ck-unavailable"
        project={project}
        capability={{
          context_revision: "ctx:cpu",
          backend: "standard",
          state: "unavailable",
          reasons: [{ code: "target_device_not_cuda", message: reason }],
        }}
        onChange={vi.fn()}
      />,
    );
    const checkbox = screen.getByRole("checkbox", {
      name: "启用 Comfy Kitchen Attention",
    });
    const control = checkbox.closest(".timeline-ck-attention__control");
    const tooltip = screen.getByRole("tooltip");
    expect(checkbox).toBeDisabled();
    expect(control).toHaveAttribute("tabindex", "0");
    expect(control).toHaveAttribute("aria-describedby", tooltip.id);
    expect(control).toHaveAttribute("title", "当前推理目标不是 CUDA。");
    (control as HTMLElement).focus();
    expect(control).toHaveFocus();
    expect(tooltip).toHaveTextContent("当前推理目标不是 CUDA。");
  });

  it("keeps an enabled selection operable when capability later becomes unavailable", () => {
    const project = createTimelineProject();
    project.features.project.comfy_kitchen_attention = { enabled: true, params: {} };
    const onChange = vi.fn();
    render(
      <ComfyKitchenAttentionField
        id="ck-enabled"
        project={project}
        capability={{
          context_revision: "ctx:lost",
          backend: "raylight",
          state: "unavailable",
          reasons: [{ code: "raylight_ring_degree_incompatible", message: "需要 ring degree 1。" }],
        }}
        onChange={onChange}
      />,
    );
    const checkbox = screen.getByRole("checkbox", {
      name: "启用 Comfy Kitchen Attention",
    });
    expect(checkbox).toBeChecked();
    expect(checkbox).toBeEnabled();
    fireEvent.click(checkbox);
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it("keeps only the CK control unavailable for an ambiguous Bundle 5 project", () => {
    const project = createTimelineProject();
    project.features.template_bundle_version = 5;
    delete project.features.project.comfy_kitchen_attention;
    render(
      <ComfyKitchenAttentionField
        id="ck-legacy"
        project={project}
        capability={available}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("checkbox", {
      name: "启用 Comfy Kitchen Attention",
    })).toBeDisabled();
    expect(screen.getByRole("tooltip")).toHaveTextContent(
      "其他编辑和生成不受影响",
    );
  });

  it("reads visible labels and reasons from the active Translator", () => {
    const project = createTimelineProject();
    render(
      <I18nProvider locale="test" catalogs={{
        test: {
          "globalSettings.ck.ariaLabel": "Acceleration test",
          "globalSettings.ck.title": "Acceleration title test",
          "globalSettings.ck.enableLabel": "Enable CK test",
          "globalSettings.ck.state.available": "Ready test",
          "globalSettings.ck.reason.available": "Backend {backend} ready test",
          "globalSettings.shared": "Shared test",
        },
      }}>
        <ComfyKitchenAttentionField
          id="ck-translated"
          project={project}
          capability={available}
          onChange={vi.fn()}
        />
      </I18nProvider>,
    );

    const region = screen.getByRole("region", { name: "Acceleration test" });
    expect(region).toHaveTextContent("Acceleration title test");
    expect(region).toHaveTextContent("Shared test");
    expect(region).toHaveTextContent("Ready test");
    expect(screen.getByRole("checkbox", { name: "Enable CK test" }))
      .toHaveAttribute("aria-describedby", "ck-translated-reason");
    expect(screen.getByRole("tooltip"))
      .toHaveTextContent("Backend Standard ready test");
  });
});
