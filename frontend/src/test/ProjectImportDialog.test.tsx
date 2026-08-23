import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { directorApi } from "../api/client";
import type { ProjectImportPreflightResponse, ProjectSummary } from "../api/types";
import { ProjectImportDialog } from "../components/ProjectImportDialog";
import {
  createTimelineProject,
  updateLoraFeatureFamily,
} from "../domain/timelineProject";

const importedSummary: ProjectSummary = {
  id: "imported-project",
  title: "旧片导入",
  created_at: "2026-08-22T00:00:00Z",
  updated_at: "2026-08-22T00:00:00Z",
  segment_count: 1,
};

function preflight(
  status: "ready" | "needs_input",
  proposed = createTimelineProject(),
): ProjectImportPreflightResponse {
  return {
    schema_version: 1,
    status,
    input_digest: {
      algorithm: "sha256-canonical-json-v1",
      value: `sha256-${"a".repeat(64)}`,
    },
    proposed_document: status === "ready" ? proposed : null,
    missing_context: status === "needs_input" ? ["creative_selection"] : [],
    missing_model_bindings: status === "ready" ? ["audio_vae"] : [],
    capability_issues: status === "ready" ? [{ code: "node_missing" }] : [],
    commit_token: status === "ready" ? "c".repeat(64) : null,
    expires_at: status === "ready" ? "2026-08-22T01:00:00Z" : null,
  };
}

describe("ProjectImportDialog", () => {
  afterEach(() => vi.restoreAllMocks());

  it("context-free v4 必须显式编辑并确认当前项目创作配置后才能 commit", async () => {
    const user = userEvent.setup();
    const current = createTimelineProject();
    current.model_stack.fl2va.filename = "current-fl2va.safetensors";
    current.model_stack.clip.filename = "current-clip.safetensors";
    current.features.project.lora = updateLoraFeatureFamily(
      current.features,
      "fl2va",
      { enabled: true, filename: "current-lora.safetensors", strength: 0.75 },
    );
    const legacy = { version: 4, title: "旧片导入", segments: [{ id: "legacy-1" }] };
    const preflightSpy = vi.spyOn(directorApi, "preflightProjectImport")
      .mockResolvedValueOnce(preflight("needs_input"))
      .mockResolvedValueOnce(preflight("ready"));
    vi.spyOn(directorApi, "commitProjectImport").mockResolvedValue(importedSummary);
    const onImported = vi.fn();

    render(<ProjectImportDialog currentProject={current} onImported={onImported} />);
    await user.upload(
      screen.getByLabelText("选择项目 JSON 文件"),
      new File([JSON.stringify(legacy)], "legacy.json", { type: "application/json" }),
    );

    await waitFor(() => expect(preflightSpy).toHaveBeenCalledTimes(1));
    expect(preflightSpy).toHaveBeenNthCalledWith(1, {
      title: "旧片导入",
      document: legacy,
    });
    expect(await screen.findByLabelText("导入 fl2va 模型"))
      .toHaveValue("current-fl2va.safetensors");
    expect(screen.getByLabelText("导入 fl2va LoRA 文件"))
      .toHaveValue("current-lora.safetensors");
    await user.clear(screen.getByLabelText("导入 fl2va 模型"));
    await user.type(screen.getByLabelText("导入 fl2va 模型"), "chosen-fl2va.safetensors");
    await user.click(screen.getByRole("button", { name: "确认创作配置并重新预检" }));

    await waitFor(() => expect(preflightSpy).toHaveBeenCalledTimes(2));
    expect(preflightSpy.mock.calls[1][0]).toEqual({
      title: "旧片导入",
      document: legacy,
      creative_selection: expect.objectContaining({
        model_stack: expect.objectContaining({
          fl2va: { filename: "chosen-fl2va.safetensors" },
          clip: { filename: "current-clip.safetensors" },
        }),
        lora: expect.objectContaining({ enabled: true }),
      }),
    });
    expect(preflightSpy.mock.calls[1][0]).not.toHaveProperty("legacy_runtime_settings");
    expect(await screen.findByText("audio_vae")).toBeInTheDocument();
    expect(screen.getByText(/node_missing/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "提交导入项目" }));
    await waitFor(() => expect(directorApi.commitProjectImport).toHaveBeenCalledWith({
      commit_token: "c".repeat(64),
      input_digest: preflight("ready").input_digest,
    }));
    expect(onImported).toHaveBeenCalledWith(importedSummary);
  });

  it("v5 文件直接 preflight 为 ready，仍需显式 commit", async () => {
    const user = userEvent.setup();
    const current = createTimelineProject();
    current.title = "V5 可移植项目";
    const preflightSpy = vi.spyOn(directorApi, "preflightProjectImport")
      .mockResolvedValue(preflight("ready", current));
    const commitSpy = vi.spyOn(directorApi, "commitProjectImport")
      .mockResolvedValue(importedSummary);

    render(<ProjectImportDialog currentProject={current} onImported={vi.fn()} />);
    await user.upload(
      screen.getByLabelText("选择项目 JSON 文件"),
      new File([JSON.stringify(current)], "portable.json", { type: "application/json" }),
    );
    await waitFor(() => expect(preflightSpy).toHaveBeenCalledWith({
      title: "V5 可移植项目",
      document: current,
    }));
    expect(screen.queryByRole("region", { name: "导入创作配置" }))
      .not.toBeInTheDocument();
    expect(commitSpy).not.toHaveBeenCalled();
    await user.click(await screen.findByRole("button", { name: "提交导入项目" }));
    await waitFor(() => expect(commitSpy).toHaveBeenCalledTimes(1));
  });

  it("v5 needs_input 只显示缺失与能力问题，不进入 v4 创作配置选择循环", async () => {
    const user = userEvent.setup();
    const current = createTimelineProject();
    current.title = "素材尚未就绪的 V5 项目";
    const response: ProjectImportPreflightResponse = {
      ...preflight("needs_input"),
      missing_context: ["source_assets"],
      missing_model_bindings: ["video_vae"],
      capability_issues: [{ code: "asset_missing" }],
    };
    const preflightSpy = vi.spyOn(directorApi, "preflightProjectImport")
      .mockResolvedValue(response);
    const commitSpy = vi.spyOn(directorApi, "commitProjectImport");

    render(<ProjectImportDialog currentProject={current} onImported={vi.fn()} />);
    await user.upload(
      screen.getByLabelText("选择项目 JSON 文件"),
      new File([JSON.stringify(current)], "missing-assets.json", {
        type: "application/json",
      }),
    );

    const issues = await screen.findByRole("region", { name: "导入预检问题" });
    expect(issues).toHaveTextContent("source_assets");
    expect(issues).toHaveTextContent("video_vae");
    expect(issues).toHaveTextContent("asset_missing");
    expect(screen.queryByRole("region", { name: "导入创作配置" }))
      .not.toBeInTheDocument();
    expect(screen.queryByLabelText("导入 fl2va 模型")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "提交导入项目" }))
      .not.toBeInTheDocument();
    expect(preflightSpy).toHaveBeenCalledTimes(1);
    expect(commitSpy).not.toHaveBeenCalled();
  });
});
