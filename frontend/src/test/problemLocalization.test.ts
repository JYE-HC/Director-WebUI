import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { createTranslator, localizeProblem } from "../i18n";

const translator = createTranslator();

describe("Director problem localization", () => {
  it("用稳定错误码和逻辑模型角色生成中文说明", () => {
    const error = new ApiError(
      "The selected project workflow configuration is invalid.",
      422,
      {
        detail: {
          code: "model_binding_required",
          bindings: ["clip", "video_vae", "audio_vae"],
        },
      },
      "model_binding_required",
    );

    expect(localizeProblem(error, translator)).toEqual({
      code: "model_binding_required",
      message: "当前项目缺少编码模型（CLIP）、视频编解码模型（Video VAE）、音频编解码模型（Audio VAE）。",
      remediation: "请打开“全局设置”，完成对应模型选择后重新预检。",
      technicalMessage: "The selected project workflow configuration is invalid.",
      action: "open_global_settings",
    });
  });

  it("任务历史中的结构化错误复用同一翻译链路", () => {
    const stored = JSON.stringify({
      code: "model_binding_required",
      message: "safe fallback",
      reasons: [{
        code: "model_binding_required",
        safe_details: { bindings: ["fl2va", "clip"] },
      }],
    });

    expect(localizeProblem(stored, translator)).toMatchObject({
      code: "model_binding_required",
      message: "当前项目缺少FL2VA Diffusion 模型、编码模型（CLIP）。",
      technicalMessage: "safe fallback",
    });
  });

  it("未知错误使用中文通用说明并保留技术文本", () => {
    expect(localizeProblem(new Error("third-party detail"), translator)).toEqual({
      code: "unknown",
      message: "请求未能完成（错误代码：unknown）。",
      remediation: "请检查当前配置后重试。",
      technicalMessage: "third-party detail",
      action: null,
    });
  });

  it("旧响应缺少可选详情时仍生成完整中文句子", () => {
    const error = new ApiError(
      "legacy response",
      422,
      { detail: { code: "model_binding_required" } },
    );

    expect(localizeProblem(error, translator)).toMatchObject({
      code: "model_binding_required",
      message: "当前项目缺少必需的模型配置。",
      action: "open_global_settings",
    });
  });

  it("使用后端返回的安全参数而不是前端写死帧数", () => {
    const error = new ApiError(
      "frame limit",
      422,
      { detail: { code: "segment_frame_limit_exceeded", max_frames: 768 } },
      "segment_frame_limit_exceeded",
    );

    expect(localizeProblem(error, translator)).toMatchObject({
      code: "segment_frame_limit_exceeded",
      message: "所选片段超过 MiniMax H3 的 768 帧生成上限。",
    });
  });

  it("拒绝不受控错误码和节点名并回退安全中文说明", () => {
    const malformed = {
      code: "bad code\nprivate",
      message: "technical",
      reasons: [{ safe_details: { class_type: "Bad\nNode" } }],
    };

    expect(localizeProblem(malformed, translator)).toMatchObject({
      code: "unknown",
      message: "请求未能完成（错误代码：unknown）。",
    });
  });
});
