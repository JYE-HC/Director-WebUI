import { fireEvent, render, screen, waitFor } from "@testing-library/react";

import { directorApi } from "../api/client";
import {
  DEFAULT_SETTINGS,
  type ComfyKitchenAttentionCapability,
  type RuntimeSettings,
} from "../api/types";
import type { TimelineGenerationMode } from "../domain/timelineProject";
import {
  comfyKitchenAttentionSettingsContextKey,
  useComfyKitchenAttentionCapability,
} from "../hooks/useComfyKitchenAttentionCapability";

const AVAILABLE: ComfyKitchenAttentionCapability = {
  context_revision: "ctx:test",
  backend: "standard",
  state: "available",
  reasons: [],
};

function settings(): RuntimeSettings {
  return structuredClone(DEFAULT_SETTINGS);
}

function Harness({
  active,
  familyModes,
  confirmedSettings,
  draftSettings,
}: {
  active: boolean;
  familyModes: readonly TimelineGenerationMode[];
  confirmedSettings: RuntimeSettings;
  draftSettings: RuntimeSettings;
}) {
  const { capability, refreshHostCapability } =
    useComfyKitchenAttentionCapability({
      active,
      familyModes,
      connection: "online",
      confirmedSettings,
      draftSettings,
    });
  return <>
    <output aria-label="CK 状态">{capability.state}</output>
    <button type="button" onClick={refreshHostCapability}>刷新宿主</button>
  </>;
}

describe("useComfyKitchenAttentionCapability", () => {
  afterEach(() => vi.restoreAllMocks());

  it("仅在全局设置需要展示时请求，并按 family 内容复用结果", async () => {
    const request = vi.spyOn(directorApi, "getComfyKitchenAttentionCapability")
      .mockResolvedValue(AVAILABLE);
    const current = settings();
    const view = render(<Harness
      active={false}
      familyModes={["ref2va", "fl2va", "fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);

    expect(request).not.toHaveBeenCalled();
    view.rerender(<Harness
      active
      familyModes={["ref2va", "fl2va", "fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    await waitFor(() => expect(screen.getByLabelText("CK 状态")).toHaveTextContent("available"));
    expect(request).toHaveBeenCalledTimes(1);
    expect(request.mock.calls[0][0]).toEqual(["fl2va", "ref2va"]);

    view.rerender(<Harness
      active={false}
      familyModes={["fl2va", "ref2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    view.rerender(<Harness
      active
      familyModes={["ref2va", "fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    expect(screen.getByLabelText("CK 状态")).toHaveTextContent("available");
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("忽略无关项目与运行设置变化，只在 CK 上下文确认后重查", async () => {
    const request = vi.spyOn(directorApi, "getComfyKitchenAttentionCapability")
      .mockResolvedValue(AVAILABLE);
    const current = settings();
    const view = render(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1));

    const unrelated = settings();
    unrelated.client_id = "another-client";
    unrelated.lora_loader_overrides = [{
      lora_filename: "style.safetensors",
      adapter_id: "model_only",
      options: {},
    }];
    view.rerender(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={unrelated}
      draftSettings={unrelated}
    />);
    expect(request).toHaveBeenCalledTimes(1);

    const relevantDraft = structuredClone(unrelated);
    relevantDraft.placement.fl2va.device = "cpu";
    view.rerender(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={unrelated}
      draftSettings={relevantDraft}
    />);
    expect(screen.getByLabelText("CK 状态")).toHaveTextContent("checking");
    expect(request).toHaveBeenCalledTimes(1);

    view.rerender(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={relevantDraft}
      draftSettings={relevantDraft}
    />);
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2));
  });

  it("只纳入后端 CK 评估实际读取的设置，并支持显式宿主刷新", async () => {
    const standard = settings();
    const unrelated = settings();
    unrelated.placement.clip_device = "cpu";
    unrelated.placement.ref2va.device = "cpu";
    expect(comfyKitchenAttentionSettingsContextKey(standard, ["fl2va"]))
      .toBe(comfyKitchenAttentionSettingsContextKey(unrelated, ["fl2va"]));

    const ray = settings();
    ray.multi_gpu_enabled = true;
    ray.placement.fl2va.raylight.gpu_select = [1, 0];
    ray.placement.fl2va.raylight.ring_degree = 1;
    const reordered = structuredClone(ray);
    reordered.placement.fl2va.raylight.gpu_select = [0, 1];
    expect(comfyKitchenAttentionSettingsContextKey(ray, ["fl2va"]))
      .toBe(comfyKitchenAttentionSettingsContextKey(reordered, ["fl2va"]));
    reordered.placement.fl2va.raylight.ring_degree = 2;
    expect(comfyKitchenAttentionSettingsContextKey(ray, ["fl2va"]))
      .not.toBe(comfyKitchenAttentionSettingsContextKey(reordered, ["fl2va"]));

    const request = vi.spyOn(directorApi, "getComfyKitchenAttentionCapability")
      .mockResolvedValue(AVAILABLE);
    render(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={standard}
      draftSettings={standard}
    />);
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "刷新宿主" }));
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2));
  });

  it("不缓存 unknown 或请求失败，重新打开时可再次探测", async () => {
    const request = vi.spyOn(directorApi, "getComfyKitchenAttentionCapability")
      .mockResolvedValueOnce({
        context_revision: "ctx:unknown",
        backend: "standard",
        state: "unknown",
        reasons: [{ code: "host_not_connected", message: "not connected" }],
      })
      .mockResolvedValueOnce(AVAILABLE);
    const current = settings();
    const view = render(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    await waitFor(() => expect(screen.getByLabelText("CK 状态")).toHaveTextContent("unknown"));

    view.rerender(<Harness
      active={false}
      familyModes={["fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    view.rerender(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByLabelText("CK 状态")).toHaveTextContent("available"));
  });

  it("请求失败也只保留为本次 unknown 展示", async () => {
    const request = vi.spyOn(directorApi, "getComfyKitchenAttentionCapability")
      .mockRejectedValueOnce(new Error("temporary failure"))
      .mockResolvedValueOnce(AVAILABLE);
    const current = settings();
    const view = render(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    await waitFor(() => expect(screen.getByLabelText("CK 状态")).toHaveTextContent("unknown"));

    view.rerender(<Harness
      active={false}
      familyModes={["fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    view.rerender(<Harness
      active
      familyModes={["fl2va"]}
      confirmedSettings={current}
      draftSettings={current}
    />);
    await waitFor(() => expect(request).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.getByLabelText("CK 状态")).toHaveTextContent("available"));
  });
});
