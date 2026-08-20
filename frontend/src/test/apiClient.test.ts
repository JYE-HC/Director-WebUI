import { ApiError, directorApi } from "../api/client";
import { DEFAULT_SETTINGS, type GenerationTask } from "../api/types";
import { createInitialDrafts, MODE_ORDER } from "../domain/modes";
import { createTimelineProject } from "../domain/timelineProject";

const fetchMock = vi.fn<typeof fetch>();
const CONFIGURED_SETTINGS = {
  ...DEFAULT_SETTINGS,
};
const RUNTIME_AUTHORITY_TOKEN = "f".repeat(64);

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const job: GenerationTask = {
  id: "job-1",
  mode: "t2v",
  status: "queued",
  display_name: "旧版 t2v 任务",
  project_title: "旧版 t2v 任务",
  project_id: null,
  current_project: false,
  progress: 0,
  stage: "queued",
  prompt_id: "prompt-1",
  outputs: [],
  output_files: [],
  error: null,
  preview_url: null,
  created_at: "2026-08-12T00:00:00Z",
  updated_at: "2026-08-12T00:00:00Z",
  started_at: null,
  completed_at: null,
  execution_duration_seconds: null,
  output_count: 0,
  error_summary: null,
  children: [],
  segment_results: [],
  live_preview_url: null,
};

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => vi.unstubAllGlobals());

describe("Director REST 契约", () => {
  const taskSummary = {
    total: 1,
    active: 1,
    queued: 1,
    preparing: 0,
    running: 0,
    cancelling: 0,
    succeeded: 0,
    failed: 0,
    cancelled: 0,
  };
  it("能力、GPU、模型和设置使用固定 /api 路由", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ connection: "online", supported_modes: [], supports_cancel: true, available_nodes: [], missing_nodes: [] }))
      .mockResolvedValueOnce(jsonResponse({ gpus: [{ index: 0, name: "A6000", vram_total: 1, vram_free: 1, visible: true }] }))
      .mockResolvedValueOnce(jsonResponse({ fl2va: [], ref2va: [], clip: [], video_vae: [], audio_vae: [], loras: [] }))
      .mockResolvedValueOnce(jsonResponse(CONFIGURED_SETTINGS));

    await directorApi.getCapabilities(undefined, RUNTIME_AUTHORITY_TOKEN);
    await directorApi.getGpus(undefined, RUNTIME_AUTHORITY_TOKEN);
    await directorApi.getModels(undefined, RUNTIME_AUTHORITY_TOKEN);
    await directorApi.getSettings();

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/directordeck/api/capabilities",
      "/directordeck/api/gpus",
      "/directordeck/api/models",
      "/directordeck/api/settings",
    ]);
    expect(fetchMock.mock.calls.every(([, init]) => !init?.method || init.method === "GET")).toBe(true);
  });

  it("运行资源携带同一 settings authority token", async () => {
    const token = "f".repeat(64);
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        settings: CONFIGURED_SETTINGS,
        authority_token: token,
      }))
      .mockResolvedValueOnce(jsonResponse({
        connection: "online",
        supported_modes: [],
        supports_cancel: true,
        available_nodes: [],
        missing_nodes: [],
      }));

    await expect(directorApi.getSettingsAuthority()).resolves.toEqual({
      settings: CONFIGURED_SETTINGS,
      authority_token: token,
    });
    await directorApi.getCapabilities(undefined, token);

    expect(fetchMock.mock.calls[0][0]).toBe("/directordeck/api/settings/authority");
    expect(new Headers(fetchMock.mock.calls[1][1]?.headers).get(
      "X-Director-Runtime-Authority",
    )).toBe(token);
  });

  it("RayLight 运行状态 GET 与重启确认 POST 使用严格 endpoint/epoch 证书", async () => {
    const blocked = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 2, 3, 4, 5, 6, 7],
      available_gpu_indexes: [0, 1, 2, 3],
      invalid_gpu_indexes: [4, 5, 6, 7],
      tainted: false,
      recovery_token: "a".repeat(64),
    };
    const recovered = {
      ...blocked,
      active: false,
      recovery_required: false,
      runtime_gpu_indexes: [],
      invalid_gpu_indexes: [],
      recovery_token: null,
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse(blocked))
      .mockResolvedValueOnce(jsonResponse(recovered));
    const controller = new AbortController();

    await expect(directorApi.getRayLightRuntimeStatus(
      undefined,
      RUNTIME_AUTHORITY_TOKEN,
    )).resolves.toEqual(blocked);
    await expect(directorApi.confirmRayLightRuntimeRecovery(
      36,
      "a".repeat(64),
      controller.signal,
    )).resolves.toEqual(recovered);

    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method ?? "GET"])).toEqual([
      ["/directordeck/api/raylight/runtime", "GET"],
      ["/directordeck/api/raylight/runtime/recovery/confirm-comfy-restart", "POST"],
    ]);
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual({
      confirmation: "comfyui_process_restarted",
      expected_epoch: 36,
      expected_recovery_token: "a".repeat(64),
    });
    expect(fetchMock.mock.calls[1][1]?.signal).toBe(controller.signal);
  });

  it("只保留可重试 RayLight 恢复冲突的安全错误码", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        detail: {
          code: "raylight_recovery_in_flight",
          message: "another recovery owns the endpoint lock",
          internal_owner: "secret-owner",
        },
      }, 409))
      .mockResolvedValueOnce(jsonResponse({
        detail: {
          code: "unknown_internal_conflict",
          message: "another conflict",
          internal_owner: "secret-owner",
        },
      }, 409));

    const retryable = await directorApi.confirmRayLightRuntimeRecovery(
      36,
      "a".repeat(64),
    ).then(
      () => { throw new Error("预期恢复占用冲突请求失败"); },
      (reason): ApiError => reason as ApiError,
    );
    expect(retryable).toMatchObject({
      status: 409,
      code: "raylight_recovery_in_flight",
      details: {
        detail: {
          code: "raylight_recovery_in_flight",
          message: "another recovery owns the endpoint lock",
        },
      },
    });
    expect(JSON.stringify(retryable.details)).not.toContain("secret-owner");

    const definitive = await directorApi.confirmRayLightRuntimeRecovery(
      36,
      "a".repeat(64),
    ).then(
      () => { throw new Error("预期普通恢复冲突请求失败"); },
      (reason): ApiError => reason as ApiError,
    );
    expect(definitive.code).toBeUndefined();
    expect(JSON.stringify(definitive.details)).not.toContain("unknown_internal_conflict");
    expect(JSON.stringify(definitive.details)).not.toContain("secret-owner");
  });

  it("纯文本 4xx 仍保留 ApiError HTTP 状态并作为确定失败", async () => {
    fetchMock.mockResolvedValueOnce(new Response("proxy conflict", { status: 409 }));

    const error = await directorApi.confirmRayLightRuntimeRecovery(
      36,
      "a".repeat(64),
    ).then(
      () => { throw new Error("预期纯文本冲突请求失败"); },
      (reason): ApiError => reason as ApiError,
    );

    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(409);
    expect(error.code).toBeUndefined();
  });

  it("错误体流读取失败也不丢失已收到的 HTTP 状态", async () => {
    fetchMock.mockResolvedValueOnce({
      ok: false,
      status: 409,
      text: vi.fn().mockRejectedValue(new TypeError("body stream reset")),
    } as unknown as Response);

    const error = await directorApi.confirmRayLightRuntimeRecovery(
      36,
      "a".repeat(64),
    ).then(
      () => { throw new Error("预期错误体流请求失败"); },
      (reason): ApiError => reason as ApiError,
    );

    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(409);
    expect(error.message).toBe("HTTP 409");
  });

  it("RayLight 运行状态拒绝额外字段、非法逻辑 GPU 与不一致派生字段", async () => {
    const valid = {
      active: true,
      recovery_required: true,
      epoch: 36,
      runtime_gpu_indexes: [0, 1, 4],
      available_gpu_indexes: [0, 1],
      invalid_gpu_indexes: [4],
      tainted: false,
      recovery_token: "b".repeat(64),
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ ...valid, unexpected: true }))
      .mockResolvedValueOnce(jsonResponse({ ...valid, runtime_gpu_indexes: [0, 0, 4] }))
      .mockResolvedValueOnce(jsonResponse({ ...valid, available_gpu_indexes: [0, -1] }))
      .mockResolvedValueOnce(jsonResponse({ ...valid, invalid_gpu_indexes: [5] }))
      .mockResolvedValueOnce(jsonResponse({ ...valid, invalid_gpu_indexes: [1] }))
      .mockResolvedValueOnce(jsonResponse({ ...valid, recovery_required: false }))
      .mockResolvedValueOnce(jsonResponse({
        ...valid,
        runtime_gpu_indexes: [0, 1, 4, 5],
        invalid_gpu_indexes: [4],
      }));

    for (let index = 0; index < 7; index += 1) {
      await expect(directorApi.getRayLightRuntimeStatus(
        undefined,
        RUNTIME_AUTHORITY_TOKEN,
      )).rejects.toThrow(
        "RayLight 运行状态响应结构无效",
      );
    }
  });

  it("数据存储 GET 只返回固定数据库路径", async () => {
    const active = "/srv/directordeck/data/directordeck.sqlite3";
    fetchMock.mockResolvedValueOnce(jsonResponse({ active_database_path: active }));

    await expect(directorApi.getStorage()).resolves.toEqual({
      active_database_path: active,
    });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(["/directordeck/api/storage"]);
  });

  it("数据存储 GET 接受 Windows 绝对路径", async () => {
    const active = "D:\\Programs\\ComfyUI\\user\\directordeck\\database\\directordeck.sqlite3";
    fetchMock.mockResolvedValueOnce(jsonResponse({ active_database_path: active }));

    await expect(directorApi.getStorage()).resolves.toEqual({
      active_database_path: active,
    });
  });

  it("数据存储响应拒绝额外字段与相对路径", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        active_database_path: "/srv/directordeck/data/directordeck.sqlite3",
        unexpected: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        active_database_path: "data/directordeck.sqlite3",
      }));

    await expect(directorApi.getStorage()).rejects.toThrow("数据存储响应结构无效");
    await expect(directorApi.getStorage()).rejects.toThrow("数据存储响应结构无效");
  });

  it("时间线 API 严格解析并往返 beta 调度器", async () => {
    const project = createTimelineProject();
    project.sampling.fl2va.scheduler = "beta";
    project.sampling.ref2va.scheduler = "beta";
    fetchMock
      .mockResolvedValueOnce(jsonResponse(project))
      .mockResolvedValueOnce(jsonResponse(project));

    await expect(directorApi.getTimeline()).resolves.toMatchObject({
      sampling: {
        fl2va: { scheduler: "beta" },
        ref2va: { scheduler: "beta" },
      },
    });
    await expect(directorApi.updateTimeline(project)).resolves.toMatchObject({
      sampling: {
        fl2va: { scheduler: "beta" },
        ref2va: { scheduler: "beta" },
      },
    });
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({
      sampling: {
        fl2va: { scheduler: "beta" },
        ref2va: { scheduler: "beta" },
      },
    });
  });

  it("时间线 API 对未知调度器失败封闭而不是静默降级", async () => {
    const invalid = createTimelineProject() as unknown as Record<string, unknown>;
    const sampling = invalid.sampling as Record<string, Record<string, unknown>>;
    sampling.fl2va.scheduler = "foreign_scheduler";
    fetchMock.mockResolvedValueOnce(jsonResponse(invalid));

    await expect(directorApi.getTimeline()).rejects.toMatchObject({
      name: "ApiError",
      status: 502,
      message: "时间线响应结构无效",
    });
  });

  it("时间线 authority API 严格往返 revision 并发送 expected_revision", async () => {
    const project = createTimelineProject();
    const firstAuthority = { document: project, revision: 7 };
    const secondAuthority = { document: project, revision: 8 };
    fetchMock
      .mockResolvedValueOnce(jsonResponse(firstAuthority))
      .mockResolvedValueOnce(jsonResponse(secondAuthority))
      .mockResolvedValueOnce(jsonResponse(firstAuthority))
      .mockResolvedValueOnce(jsonResponse(secondAuthority));

    await expect(directorApi.getTimelineAuthority()).resolves.toEqual(firstAuthority);
    await expect(directorApi.updateTimelineAuthority(project, 7))
      .resolves.toEqual(secondAuthority);
    await expect(directorApi.getProjectTimelineAuthority("project/one"))
      .resolves.toEqual(firstAuthority);
    await expect(directorApi.updateProjectTimelineAuthority("project/one", project, 7))
      .resolves.toEqual(secondAuthority);

    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method ?? "GET"])).toEqual([
      ["/directordeck/api/timeline/authority", "GET"],
      ["/directordeck/api/timeline/authority", "PUT"],
      ["/directordeck/api/projects/project%2Fone/timeline/authority", "GET"],
      ["/directordeck/api/projects/project%2Fone/timeline/authority", "PUT"],
    ]);
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual({
      document: project,
      expected_revision: 7,
    });
    expect(JSON.parse(String(fetchMock.mock.calls[3][1]?.body))).toEqual({
      document: project,
      expected_revision: 7,
    });
  });

  it("时间线 authority envelope 拒绝额外字段与非安全 revision", async () => {
    const project = createTimelineProject();
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ document: project, revision: 0, extra: true }))
      .mockResolvedValueOnce(jsonResponse({ document: project, revision: 1.5 }))
      .mockResolvedValueOnce(jsonResponse({ document: project, revision: 2 ** 53 }));

    await expect(directorApi.getTimelineAuthority()).rejects.toMatchObject({
      name: "ApiError",
      status: 502,
      message: "时间线权威响应结构无效",
    });
    await expect(directorApi.getTimelineAuthority()).rejects.toThrow(
      "时间线权威响应结构无效",
    );
    await expect(directorApi.getTimelineAuthority()).rejects.toThrow(
      "时间线权威响应结构无效",
    );
    expect(() => directorApi.updateTimelineAuthority(project, -1)).toThrow(
      "时间线 expected revision 必须是非负安全整数",
    );
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("时间线 CAS 409 只暴露可恢复冲突码与安全 revision", async () => {
    const project = createTimelineProject();
    fetchMock.mockResolvedValueOnce(jsonResponse({
      detail: {
        code: "timeline_revision_conflict",
        message: "timeline changed on the server",
        project_id: "project-1",
        expected_revision: 4,
        actual_revision: 5,
        internal_owner: "secret-owner",
      },
    }, 409));

    const conflict = await directorApi.updateProjectTimelineAuthority(
      "project-1",
      project,
      4,
    ).then(
      () => { throw new Error("预期时间线 revision 冲突"); },
      (reason): ApiError => reason as ApiError,
    );

    expect(conflict).toMatchObject({
      status: 409,
      code: "timeline_revision_conflict",
      details: {
        detail: {
          code: "timeline_revision_conflict",
          message: "timeline changed on the server",
          project_id: "project-1",
          expected_revision: 4,
          actual_revision: 5,
        },
      },
    });
    expect(JSON.stringify(conflict.details)).not.toContain("secret-owner");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      document: project,
      expected_revision: 4,
    });
  });

  it("能力响应缺少安全取消字段时按不支持处理", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      connection: "online",
      supported_modes: ["t2v", "unknown"],
      available_nodes: [],
      missing_nodes: [],
    }));

    await expect(directorApi.getCapabilities(
      undefined,
      RUNTIME_AUTHORITY_TOKEN,
    )).resolves.toMatchObject({
      connection: "online",
      supported_modes: ["t2v"],
      supports_cancel: false,
    });
  });

  it("能力响应区分 RayLight 基础节点与条件 LoRA 节点", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      connection: "online",
      supported_modes: MODE_ORDER,
      supports_cancel: true,
      available_nodes: [],
      missing_nodes: [],
      execution_backends: {
        standard: { available: true, missing_nodes: [] },
        raylight: {
          available: true,
          missing_nodes: [],
          conditional_requirements: {
            lora: { available: false, missing_nodes: ["RayLoraLoader"] },
          },
        },
      },
    }));

    await expect(directorApi.getCapabilities(
      undefined,
      RUNTIME_AUTHORITY_TOKEN,
    )).resolves.toMatchObject({
      execution_backends: {
        raylight: {
          available: true,
          missing_nodes: [],
          conditional_requirements: {
            lora: { available: false, missing_nodes: ["RayLoraLoader"] },
          },
        },
      },
    });
  });

  it("原生时间线能力只接收 FL2VA / Ref2VA 两族", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      connection: "online",
      supported_modes: MODE_ORDER,
      supports_cancel: true,
      available_nodes: [],
      missing_nodes: [],
      native_timeline: {
        supported: true,
        modes: ["fl2va", "t2v", "ref2va", "rv2v", "unknown"],
        continuity: false,
      },
    }));

    await expect(directorApi.getCapabilities(
      undefined,
      RUNTIME_AUTHORITY_TOKEN,
    )).resolves.toMatchObject({
      native_timeline: {
        supported: true,
        modes: ["fl2va", "ref2va"],
        continuity: false,
      },
    });
  });

  it("段间接续能力依赖原生时间线本身可用，缺失或矛盾字段均按关闭处理", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        connection: "online",
        supported_modes: MODE_ORDER,
        native_timeline: {
          supported: false,
          modes: ["fl2va", "ref2va"],
          continuity: true,
        },
      }))
      .mockResolvedValueOnce(jsonResponse({
        connection: "online",
        supported_modes: MODE_ORDER,
        native_timeline: {
          supported: true,
          modes: ["fl2va", "ref2va"],
        },
      }))
      .mockResolvedValueOnce(jsonResponse({
        connection: "online",
        supported_modes: MODE_ORDER,
        native_timeline: {
          supported: true,
          modes: ["fl2va", "ref2va"],
          continuity: true,
        },
      }));

    await expect(directorApi.getCapabilities(
      undefined,
      RUNTIME_AUTHORITY_TOKEN,
    )).resolves.toMatchObject({
      native_timeline: { supported: false, continuity: false },
    });
    await expect(directorApi.getCapabilities(
      undefined,
      RUNTIME_AUTHORITY_TOKEN,
    )).resolves.toMatchObject({
      native_timeline: { supported: true, continuity: false },
    });
    await expect(directorApi.getCapabilities(
      undefined,
      RUNTIME_AUTHORITY_TOKEN,
    )).resolves.toMatchObject({
      native_timeline: { supported: true, continuity: true },
    });
  });

  it.each(MODE_ORDER)("%s 草稿只访问对应固定模式路由", async (mode) => {
    const draft = createInitialDrafts()[mode];
    fetchMock
      .mockResolvedValueOnce(jsonResponse(draft))
      .mockResolvedValueOnce(jsonResponse(draft));

    await directorApi.getDraft(mode);
    await directorApi.updateDraft(mode, draft);

    expect(fetchMock.mock.calls[0][0]).toBe(`/directordeck/api/drafts/${mode}`);
    expect(fetchMock.mock.calls[1][0]).toBe(`/directordeck/api/drafts/${mode}`);
    expect(fetchMock.mock.calls[1][1]?.method).toBe("PUT");
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toEqual(draft);
  });

  it("设置和连接测试发送后端要求的 snake_case payload", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(CONFIGURED_SETTINGS))
      .mockResolvedValueOnce(jsonResponse({ ok: true, message: "连接成功" }));

    await directorApi.updateSettings(CONFIGURED_SETTINGS);
    await directorApi.testConnection();

    expect(fetchMock.mock.calls[0][0]).toBe("/directordeck/api/settings");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("PUT");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual(CONFIGURED_SETTINGS);
    expect(fetchMock.mock.calls[1][0]).toBe("/directordeck/api/capabilities");
    expect(fetchMock.mock.calls[1][1]?.method).toBe("POST");
    expect(fetchMock.mock.calls[1][1]?.body).toBeUndefined();
  });

  it("RV2V 智能分镜发送完整素材 ID 与显式检测参数", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      cut_frames: [0, 48, 120],
      shot_count: 2,
      warnings: [],
    }));
    const payload = {
      asset_id: "asset-video-1",
      frame_rate: 24,
      sensitivity: "medium" as const,
      min_shot_frames: 12,
    };

    await expect(directorApi.detectRV2VShots(payload)).resolves.toEqual({
      cut_frames: [0, 48, 120],
      shot_count: 2,
      warnings: [],
    });
    expect(fetchMock.mock.calls[0][0]).toBe("/directordeck/api/rv2v/detect-shots");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual(payload);
  });

  it("素材上传发送 file+kind，并拒绝没有稳定 ID 的响应", async () => {
    const file = new File(["image"], "frame.png", { type: "image/png" });
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        asset: {
          id: "asset-1",
          name: "frame.png",
          subfolder: "directordeck",
          type: "input",
          kind: "image",
        },
      }),
    );
    await expect(directorApi.uploadAsset(file, "image")).resolves.toMatchObject({ id: "asset-1" });
    const body = fetchMock.mock.calls[0][1]?.body as FormData;
    expect(fetchMock.mock.calls[0][0]).toBe("/directordeck/api/assets");
    expect(body.get("kind")).toBe("image");
    expect(body.get("file")).toBeInstanceOf(File);

    fetchMock.mockResolvedValueOnce(
      jsonResponse({ asset: { name: "legacy.png", subfolder: "", type: "input", kind: "image" } }),
    );
    await expect(directorApi.uploadAsset(file, "image")).rejects.toThrow(
      "素材上传响应缺少有效的稳定 ID",
    );
  });

  it("视频上传保留服务端探测 metadata，缺失 metadata 时拒绝响应", async () => {
    const file = new File(["video"], "source.mp4", { type: "video/mp4" });
    const metadata = {
      duration: 12,
      native_fps: 30,
      frame_count: 360,
      width: 1920,
      height: 1080,
      probe_method: "ffprobe_nb_frames",
    };
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        asset: {
          id: "asset-video-1",
          name: "source.mp4",
          subfolder: "directordeck",
          type: "input",
          kind: "video",
          metadata,
        },
      }),
    );

    await expect(directorApi.uploadAsset(file, "video")).resolves.toMatchObject({
      id: "asset-video-1",
      metadata,
    });

    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        asset: {
          id: "asset-video-2",
          name: "source.mp4",
          subfolder: "directordeck",
          type: "input",
          kind: "video",
        },
      }),
    );
    await expect(directorApi.uploadAsset(file, "video")).rejects.toThrow(
      "素材上传响应缺少有效的稳定 ID",
    );
  });

  it("带进度回调时报告浏览器上行和服务端处理阶段", async () => {
    const file = new File(["image"], "frame.png", { type: "image/png" });
    const progress = vi.fn();
    let xhr: FakeXMLHttpRequest | null = null;
    let sentBody: FormData | null = null;

    class FakeXMLHttpRequest {
      upload: {
        onprogress: ((event: ProgressEvent) => void) | null;
        onload: (() => void) | null;
      } = { onprogress: null, onload: null };
      status = 200;
      response: unknown = {
        asset: {
          id: "asset-progress",
          name: "frame.png",
          subfolder: "directordeck",
          type: "input",
          kind: "image",
        },
      };
      responseType = "";
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      onabort: (() => void) | null = null;
      constructor() { xhr = this; }
      open() {}
      setRequestHeader() {}
      send(body: FormData) {
        sentBody = body;
        this.upload.onprogress?.({ lengthComputable: true, loaded: 5, total: 10 } as ProgressEvent);
        this.upload.onload?.();
      }
    }
    vi.stubGlobal("XMLHttpRequest", FakeXMLHttpRequest);
    fetchMock.mockResolvedValueOnce(jsonResponse({
      stage: "analyzing",
      input_bytes: 10,
      output_bytes: 0,
    }));

    const pending = directorApi.uploadAsset(file, "image", progress);
    await vi.waitFor(() => expect(progress).toHaveBeenCalledWith({
      stage: "analyzing",
      input_bytes: 10,
      output_bytes: 0,
    }));
    xhr!.onload?.();

    await expect(pending).resolves.toMatchObject({ id: "asset-progress" });
    expect(progress).toHaveBeenCalledWith({ stage: "uploading", percent: 50 });
    expect(progress).toHaveBeenCalledWith({ stage: "processing" });
    expect(progress).toHaveBeenLastCalledWith({ stage: "complete" });
    expect(sentBody!.get("upload_id")).toMatch(/^[0-9a-f-]{36}$/);
  });

  it("素材级联删除使用 exact query、不发 body，并严格重建响应", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      deleted_asset_id: "asset/1",
      outputs_preserved: true,
      unbound_usages: ["timeline.segments[0].first_image"],
    }));

    await expect(directorApi.deleteAssetCascade("asset/1")).resolves.toEqual({
      deleted_asset_id: "asset/1",
      outputs_preserved: true,
      unbound_usages: ["timeline.segments[0].first_image"],
    });
    expect(fetchMock.mock.calls[0][0]).toBe("/directordeck/api/assets/asset%2F1?cascade=true");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("DELETE");
    expect(fetchMock.mock.calls[0][1]?.body).toBeUndefined();
    expect(new Headers(fetchMock.mock.calls[0][1]?.headers).has("Content-Type")).toBe(false);
  });

  it("素材与回收站列表使用固定路由并严格解析", async () => {
    const assetValue = {
      id: "asset-scoped",
      name: "scoped.png",
      subfolder: "directordeck",
      type: "input",
      kind: "image",
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        assets: [assetValue],
        outputs_preserved: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        batches: [],
        remote_files_preserved: true,
      }));

    await expect(directorApi.listAssets("image")).resolves.toEqual({
      assets: [assetValue],
      outputs_preserved: true,
    });
    await expect(directorApi.listAssetTrash()).resolves.toEqual({
      batches: [],
      remote_files_preserved: true,
    });
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/directordeck/api/assets?kind=image",
      "/directordeck/api/asset-trash",
    ]);
  });

  it("素材与回收站列表拒绝额外 envelope 字段", async () => {
    const assetValue = {
      id: "asset-1",
      name: "one.png",
      subfolder: "directordeck",
      type: "input",
      kind: "image",
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        assets: [assetValue],
        outputs_preserved: true,
        active_database_identity: "a".repeat(64),
      }))
      .mockResolvedValueOnce(jsonResponse({
        batches: [],
        remote_files_preserved: true,
        comfy_origin: "http://comfy.test:8188",
      }));

    await expect(directorApi.listAssets()).rejects.toThrow("素材列表响应结构无效");
    await expect(directorApi.listAssetTrash()).rejects.toThrow("素材回收站响应结构无效");
  });

  it("保留非级联素材删除方法的旧路由契约", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      deleted_asset_id: "asset/1",
      outputs_preserved: true,
    }));

    await expect(directorApi.deleteAsset("asset/1")).resolves.toEqual({
      deleted_asset_id: "asset/1",
      outputs_preserved: true,
    });
    expect(fetchMock.mock.calls[0][0]).toBe("/directordeck/api/assets/asset%2F1");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("DELETE");
    expect(fetchMock.mock.calls[0][1]?.body).toBeUndefined();
  });

  it("素材级联删除拒绝错误 ID、非保留输出、非字符串 usages 和额外字段", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        deleted_asset_id: "another-asset",
        outputs_preserved: true,
        unbound_usages: [],
      }))
      .mockResolvedValueOnce(jsonResponse({
        deleted_asset_id: "asset-1",
        outputs_preserved: false,
        unbound_usages: [],
      }))
      .mockResolvedValueOnce(jsonResponse({
        deleted_asset_id: "asset-1",
        outputs_preserved: true,
        unbound_usages: [1],
      }))
      .mockResolvedValueOnce(jsonResponse({
        deleted_asset_id: "asset-1",
        outputs_preserved: true,
        unbound_usages: [],
        debug: "must-not-be-accepted",
      }));

    await expect(directorApi.deleteAssetCascade("asset-1")).rejects.toThrow("素材移出并解除引用响应结构无效");
    await expect(directorApi.deleteAssetCascade("asset-1")).rejects.toThrow("素材移出并解除引用响应结构无效");
    await expect(directorApi.deleteAssetCascade("asset-1")).rejects.toThrow("素材移出并解除引用响应结构无效");
    await expect(directorApi.deleteAssetCascade("asset-1")).rejects.toThrow("素材移出并解除引用响应结构无效");
  });

  it("素材回收站用一次批量请求完成 trash、list、restore 和 purge", async () => {
    const first = {
      id: "asset/one",
      name: "one.png",
      subfolder: "directordeck",
      type: "input",
      kind: "image",
    };
    const second = {
      id: "asset-two",
      name: "two.wav",
      subfolder: "directordeck",
      type: "input",
      kind: "audio",
    };
    const batch = {
      batch_id: "batch/one",
      asset_ids: [first.id, second.id],
      assets: [first, second],
      cascade: true,
      unbound_usages: ["timeline.segments[0].first_image"],
      unbound_usages_by_asset: {
        [first.id]: ["timeline.segments[0].first_image"],
        [second.id]: [],
      },
      created_at: "2026-08-16T12:00:00Z",
      remote_files_preserved: true,
    };
    const restored = {
      batch_id: batch.batch_id,
      restored_asset_ids: batch.asset_ids,
      restored_references: true,
      mode: "with_references",
      remote_files_preserved: true,
    };
    const purged = {
      batch_id: batch.batch_id,
      purged_asset_ids: batch.asset_ids,
      remote_files_preserved: true,
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse(batch))
      .mockResolvedValueOnce(jsonResponse({
        batches: [batch],
        remote_files_preserved: true,
      }))
      .mockResolvedValueOnce(jsonResponse(restored))
      .mockResolvedValueOnce(jsonResponse(purged));

    await expect(
      directorApi.trashAssets([first.id, second.id], true),
    ).resolves.toEqual(batch);
    await expect(directorApi.listAssetTrash()).resolves.toEqual({
      batches: [batch],
      remote_files_preserved: true,
    });
    await expect(
      directorApi.restoreAssetTrash(batch.batch_id, "with_references"),
    ).resolves.toEqual(restored);
    await expect(directorApi.purgeAssetTrash(batch.batch_id)).resolves.toEqual(purged);

    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method ?? "GET"])).toEqual([
      ["/directordeck/api/asset-trash", "POST"],
      ["/directordeck/api/asset-trash", "GET"],
      ["/directordeck/api/asset-trash/batch%2Fone/restore", "POST"],
      ["/directordeck/api/asset-trash/batch%2Fone", "DELETE"],
    ]);
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      asset_ids: [first.id, second.id],
      cascade: true,
    });
    expect(JSON.parse(String(fetchMock.mock.calls[2][1]?.body))).toEqual({
      mode: "with_references",
    });
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it("素材回收站严格拒绝不自洽 envelope、错误 mode 和非保留远端文件", async () => {
    const assetValue = {
      id: "asset-1",
      name: "one.png",
      subfolder: "directordeck",
      type: "input",
      kind: "image",
    };
    const batch = {
      batch_id: "batch-1",
      asset_ids: [assetValue.id],
      assets: [assetValue],
      cascade: true,
      unbound_usages: [],
      unbound_usages_by_asset: { [assetValue.id]: [] },
      created_at: "2026-08-16T12:00:00Z",
      remote_files_preserved: true,
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ ...batch, debug: true }))
      .mockResolvedValueOnce(jsonResponse({
        ...batch,
        asset_ids: ["different-asset"],
      }))
      .mockResolvedValueOnce(jsonResponse({
        batches: [batch, batch],
        remote_files_preserved: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        batch_id: "batch-1",
        restored_asset_ids: ["asset-1"],
        restored_references: true,
        mode: "registration_only",
        remote_files_preserved: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        batch_id: "batch-1",
        purged_asset_ids: ["asset-1"],
        remote_files_preserved: false,
      }));

    await expect(directorApi.trashAssets(["asset-1"], true))
      .rejects.toThrow("素材回收批次响应结构无效");
    await expect(directorApi.trashAssets(["asset-1"], true))
      .rejects.toThrow("素材回收批次响应结构无效");
    await expect(directorApi.listAssetTrash())
      .rejects.toThrow("素材回收站响应结构无效");
    await expect(directorApi.restoreAssetTrash("batch-1", "registration_only"))
      .rejects.toThrow("素材恢复响应结构无效");
    await expect(directorApi.purgeAssetTrash("batch-1"))
      .rejects.toThrow("素材回收批次清理响应结构无效");
  });

  it("素材回收批量请求在发送前拒绝空、重复和超量 ID", () => {
    expect(() => directorApi.trashAssets([])).toThrow(
      "素材回收批次必须包含 1 至 128 个不重复的稳定 ID",
    );
    expect(() => directorApi.trashAssets(["same", "same"])).toThrow(
      "素材回收批次必须包含 1 至 128 个不重复的稳定 ID",
    );
    expect(() => directorApi.trashAssets(
      Array.from({ length: 129 }, (_, index) => `asset-${index}`),
    )).toThrow("素材回收批次必须包含 1 至 128 个不重复的稳定 ID");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("素材恢复 409 只暴露白名单 owner 冲突和远端文件保留事实", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      detail: {
        code: "asset_trash_restore_conflict",
        message: "asset references changed after the trash operation",
        conflicts: [{
          owner_kind: "project",
          owner_id: "project-1",
          reason: "document_changed,revision_changed",
          expected_revision: 4,
          actual_revision: 5,
        }],
        remote_files_preserved: true,
        internal_trace: "secret-detail",
      },
      debug_secret: "secret-root",
    }, 409));

    const error = await directorApi.restoreAssetTrash(
      "batch-1",
      "with_references",
    ).catch((reason: unknown) => reason);

    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 409,
      code: "asset_trash_restore_conflict",
      details: {
        detail: {
          code: "asset_trash_restore_conflict",
          message: "asset references changed after the trash operation",
          conflicts: [{
            owner_kind: "project",
            owner_id: "project-1",
            reason: "document_changed,revision_changed",
            expected_revision: 4,
            actual_revision: 5,
          }],
          remote_files_preserved: true,
        },
      },
    });
    expect(JSON.stringify((error as ApiError).details)).not.toContain("secret");
  });

  it("批量素材仍被引用时保留可判定错误码和安全 usage 映射", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      detail: {
        code: "assets_in_use",
        message: "one or more assets are still referenced by saved drafts",
        usages: ["timeline.segments[0].first_image"],
        usages_by_asset: {
          "asset-1": ["timeline.segments[0].first_image"],
          "asset-2": [],
        },
        remote_files_preserved: true,
        database_query: "secret sql",
      },
    }, 409));

    const error = await directorApi.trashAssets(["asset-1", "asset-2"])
      .catch((reason: unknown) => reason);

    expect(error).toMatchObject({
      status: 409,
      code: "assets_in_use",
      details: {
        detail: {
          code: "assets_in_use",
          usages: ["timeline.segments[0].first_image"],
          usages_by_asset: {
            "asset-1": ["timeline.segments[0].first_image"],
            "asset-2": [],
          },
          remote_files_preserved: true,
        },
      },
    });
    expect(JSON.stringify((error as ApiError).details)).not.toContain("secret sql");
  });

  it("HTTP detail 对象只暴露可读 message/usages，不泄漏其他字段", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({
      detail: {
        message: "asset is still referenced by saved drafts",
        usages: ["timeline.segments[0].first_image", "drafts.i2v.shots[0]"],
        debug_secret: "private stack detail",
      },
      internal_trace: "private trace",
    }, 409));

    const error = await directorApi.deleteAsset("asset-1").catch((reason: unknown) => reason);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      status: 409,
      message: "asset is still referenced by saved drafts（引用位置：timeline.segments[0].first_image、drafts.i2v.shots[0]）",
      details: {
        detail: {
          message: "asset is still referenced by saved drafts",
          usages: ["timeline.segments[0].first_image", "drafts.i2v.shots[0]"],
        },
      },
    });
    expect(JSON.stringify((error as ApiError).details)).not.toContain("private");
  });

  it("任务提交、列表、详情和取消与 /api/jobs 契约一致", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(job))
      .mockResolvedValueOnce(jsonResponse({
        jobs: [job], total: 1, limit: 256, offset: 0, has_more: false,
        summary: taskSummary,
      }))
      .mockResolvedValueOnce(jsonResponse(job))
      .mockResolvedValueOnce(jsonResponse({ ...job, status: "cancelled", progress: 1 }));
    const config = { ...createInitialDrafts().t2v, prompt: "镜头" };

    await directorApi.createTask({ mode: "t2v", config });
    await directorApi.listTasks();
    await directorApi.getTask("job-1");
    await directorApi.cancelTask("job-1");

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/directordeck/api/jobs",
      "/directordeck/api/jobs?limit=256&offset=0&sort_by=created_at&sort_order=asc",
      "/directordeck/api/jobs/job-1",
      "/directordeck/api/jobs/job-1/cancel",
    ]);
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({ mode: "t2v", config });
    expect(fetchMock.mock.calls[3][1]?.method).toBe("POST");
  });

  it("任务详情和 mutation 对非默认活动项目使用 URL 编码的 project_id", async () => {
    const activeProjectId = "project /?&当前";
    const encodedProjectId = encodeURIComponent(activeProjectId);
    const cancelled = { ...job, status: "cancelled" as const, progress: 1 };
    fetchMock
      .mockResolvedValueOnce(jsonResponse(job))
      .mockResolvedValueOnce(jsonResponse(cancelled))
      .mockResolvedValueOnce(jsonResponse(cancelled))
      .mockResolvedValueOnce(jsonResponse({
        jobs: [cancelled], requested_count: 1, terminal_count: 1,
      }));
    const controller = new AbortController();

    await directorApi.getTask(job.id, controller.signal, activeProjectId);
    await directorApi.cancelTask(job.id, activeProjectId);
    await directorApi.confirmComfyRestartRecovery(job.id, activeProjectId);
    await directorApi.cancelTasks([job.id], activeProjectId);

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      `/directordeck/api/jobs/job-1?project_id=${encodedProjectId}`,
      `/directordeck/api/jobs/job-1/cancel?project_id=${encodedProjectId}`,
      `/directordeck/api/jobs/job-1/recovery/confirm-comfy-restart?project_id=${encodedProjectId}`,
      `/directordeck/api/jobs/cancel?project_id=${encodedProjectId}`,
    ]);
    expect(fetchMock.mock.calls[0][1]?.signal).toBe(controller.signal);
    expect(JSON.parse(String(fetchMock.mock.calls[2][1]?.body))).toEqual({
      confirmation: "comfyui_process_restarted",
    });
    expect(JSON.parse(String(fetchMock.mock.calls[3][1]?.body))).toEqual({
      job_ids: [job.id],
    });
  });

  it("任务响应按公开字段精确重建，拒绝 workflow/prompt 图泄漏", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ ...job, workflow: { nodes: [] } }))
      .mockResolvedValueOnce(jsonResponse({
        jobs: [{ ...job, prompt: { "1": { class_type: "Sampler" } } }],
        total: 1, limit: 256, offset: 0, has_more: false, summary: taskSummary,
      }));
    await expect(directorApi.getTask(job.id)).rejects.toThrow("任务响应结构无效");
    await expect(directorApi.listTasks()).rejects.toThrow("任务响应结构无效");
  });

  it("人工确认 ComfyUI 重启只发送固定令牌并严格解析任务", async () => {
    const recovered = {
      ...job,
      status: "cancelled",
      progress: 1,
      stage: "restart_cancel_confirmed",
      completed_at: "2026-08-12T00:03:00Z",
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse(recovered))
      .mockResolvedValueOnce(jsonResponse({ id: job.id, status: "cancelled" }));

    await expect(
      directorApi.confirmComfyRestartRecovery(job.id),
    ).resolves.toMatchObject({ id: job.id, status: "cancelled" });
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "/directordeck/api/jobs/job-1/recovery/confirm-comfy-restart",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ confirmation: "comfyui_process_restarted" }),
      }),
    );
    await expect(
      directorApi.confirmComfyRestartRecovery(job.id),
    ).rejects.toThrow("任务响应结构无效");
  });

  it("任务列表按升序稳定拉取所有分页而不是只显示前 100 条", async () => {
    const second = { ...job, id: "job-2" };
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        jobs: [job], total: 2, limit: 256, offset: 0, has_more: true,
        summary: { ...taskSummary, total: 2 },
      }))
      .mockResolvedValueOnce(jsonResponse({
        jobs: [second], total: 2, limit: 256, offset: 1, has_more: false,
        summary: { ...taskSummary, total: 2 },
      }));

    const result = await directorApi.listTasks();

    expect(result.jobs.map((task) => task.id)).toEqual([job.id, second.id]);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/directordeck/api/jobs?limit=256&offset=0&sort_by=created_at&sort_order=asc",
      "/directordeck/api/jobs?limit=256&offset=1&sort_by=created_at&sort_order=asc",
    ]);
  });

  it("批量取消、来源项目和生成结果导入只传稳定任务内身份", async () => {
    const project = createTimelineProject();
    const cancelled = { ...job, status: "cancelled", progress: 1 };
    const asset = {
      id: "imported-video",
      name: "take_24fps.mp4",
      filename: "take_24fps.mp4",
      path: "director-web/take_24fps.mp4",
      subfolder: "directordeck",
      type: "input",
      kind: "video",
      preview_url: "/directordeck/api/assets/imported-video/preview",
      content_hash: `sha256:${"a".repeat(64)}`,
      metadata: {
        duration: 5,
        native_fps: 24,
        frame_count: 120,
        width: 864,
        height: 480,
        probe_method: "ffprobe",
        has_audio: true,
      },
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        jobs: [cancelled], requested_count: 1, terminal_count: 1,
      }))
      .mockResolvedValueOnce(jsonResponse({
        job_id: job.id, project, segment_ids: [project.segments[0].id],
      }))
      .mockResolvedValueOnce(jsonResponse({
        schema_version: 1,
        id: job.id,
        display_name: "测试任务",
        project_title: "测试项目",
        mode: "timeline",
        status: "succeeded",
        progress: 1,
        stage: "completed",
        created_at: job.created_at,
        updated_at: job.updated_at,
        started_at: job.started_at,
        completed_at: job.completed_at,
        execution_duration_seconds: 60,
        output_files: ["output/final.mp4"],
        error_summary: null,
        children: [],
        settings_included: false,
        workflow_included: false,
      }))
      .mockResolvedValueOnce(jsonResponse({ asset }));

    await expect(directorApi.cancelTasks([job.id])).resolves.toMatchObject({
      requested_count: 1,
      jobs: [{ id: job.id, status: "cancelled" }],
    });
    await expect(directorApi.getTaskProject(job.id)).resolves.toMatchObject({
      job_id: job.id,
      project: { title: project.title },
    });
    await expect(directorApi.getTaskDiagnostic(job.id)).resolves.toMatchObject({
      id: job.id,
      settings_included: false,
      workflow_included: false,
    });
    await expect(directorApi.importTaskOutput(job.id, {
      index: 3,
      segmentId: project.segments[0].id,
    })).resolves.toMatchObject({ id: "imported-video", kind: "video" });

    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/directordeck/api/jobs/cancel",
      "/directordeck/api/jobs/job-1/project",
      "/directordeck/api/jobs/job-1/diagnostic",
      "/directordeck/api/jobs/job-1/import-output",
    ]);
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      job_ids: [job.id],
    });
    expect(JSON.parse(String(fetchMock.mock.calls[3][1]?.body))).toEqual({
      segment_id: project.segments[0].id,
    });
  });

  it("生成参数使用独立懒加载路由并严格拒绝工作流等额外字段", async () => {
    const details = {
      schema_version: 2,
      job_id: job.id,
      project_title: "历史项目",
      render: {
        width: 1280,
        height: 704,
        fps: 24,
        export_mode: "segments",
        total_duration_seconds: 10,
      },
      sampling: [{
        family: "fl2va",
        steps: 4,
        seed: 123456789,
        random_seed: false,
        sampler: "euler",
        scheduler: "karras",
        shift: 9.5,
        audio_shift: 2.5,
      }],
      models: [{
        family: "fl2va",
        filename: "minimax-h3.safetensors",
        device: "default",
        lora_name: null,
        lora_strength: 1,
        backends: ["standard"],
        logical_gpu_indices: [],
        ulysses_degree: null,
        ring_degree: null,
      }],
      shared_models: [{
        role: "clip",
        filename: "qwen.safetensors",
        device: "default",
      }],
      runtime_snapshot_available: true,
      segments: [{
        id: "segment-1",
        title: "镜头一",
        family: "fl2va",
        recipe: "t2v",
        duration_seconds: 10,
        prompt: "A safe prompt",
        continuity_enabled: false,
        continuity_overlap_frames: 22,
        ref_image_size: "max",
        audio_mode: "generate",
        has_first_image: false,
        has_last_image: false,
        has_source_video: false,
        source_audio_as_reference: false,
        reference_image_count: 0,
        reference_audio_count: 0,
        reference_video_count: 0,
      }],
    };
    fetchMock
      .mockResolvedValueOnce(jsonResponse(details))
      .mockResolvedValueOnce(jsonResponse({ ...details, workflow: { secret: true } }));

    await expect(directorApi.getTaskGenerationDetails(job.id)).resolves.toMatchObject({
      job_id: job.id,
      render: { total_duration_seconds: 10 },
      segments: [{ prompt: "A safe prompt" }],
    });
    await expect(directorApi.getTaskGenerationDetails(job.id)).rejects.toBeInstanceOf(ApiError);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/directordeck/api/jobs/job-1/generation-details",
      "/directordeck/api/jobs/job-1/generation-details",
    ]);
  });

  it("编译端点只接收服务端脱敏计划并拒绝 workflow/prompt 泄漏", async () => {
    const project = createTimelineProject();
    const report = {
      execution_strategy: "native_segment_graph_v1",
      model_families: ["fl2va"],
      plans: [{
        segment_id: project.segments[0].id,
        mode: "fl2va",
        recipe: "t2v",
        model_family: "fl2va",
        backend: "standard",
        frame_count: 124,
        visible_frame_count: 124,
        sample_frame_count: 124,
        continuity_context_frames: 0,
        alignment_tail_frame_count: 0,
        predecessor_segment_id: null,
        continuity_source: null,
        historical_take_id: null,
        anchor_reset: true,
        seed_mode: "fixed",
        seed: 7,
        conditioning_node: "MiniMaxH3ImageToVideo",
        node_classes: ["UNETLoader", "BasicGuider", "SamplerCustomAdvanced"],
      }],
      node_policy: {
        graph_source: "server",
        accepts_client_workflow: false,
        allowed_nodes: ["UNETLoader", "BasicGuider", "SamplerCustomAdvanced"],
        custom_nodes: [],
        provenance: {
          UNETLoader: "comfy-core",
          BasicGuider: "comfy-extras",
          SamplerCustomAdvanced: "comfy-extras",
        },
      },
    };
    fetchMock.mockResolvedValueOnce(jsonResponse(report));

    await expect(directorApi.compileTimeline({ config: project })).resolves.toEqual(report);
    expect(fetchMock.mock.calls[0][0]).toBe("/directordeck/api/timeline/compile");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({ config: project });

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      prompt: { "1": { class_type: "MiniMaxH3Director", inputs: {} } },
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划响应结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      workflow: { nodes: [] },
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划响应结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{ ...report.plans[0], seed_mode: "random", seed: 19 }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).resolves.toMatchObject({
      plans: [{ seed_mode: "random", seed: 19 }],
    });

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      model_families: ["fl2va", "ref2va"],
      plans: [
        report.plans[0],
        {
          ...report.plans[0],
          segment_id: "successor-segment",
          mode: "ref2va",
          model_family: "ref2va",
          recipe: "r2v",
          conditioning_node: "MiniMaxH3ReferenceToVideo",
          sample_frame_count: 158,
          continuity_context_frames: 22,
          alignment_tail_frame_count: 12,
          predecessor_segment_id: project.segments[0].id,
          continuity_source: "same_run",
          historical_take_id: null,
          anchor_reset: false,
        },
      ],
    }));
    await expect(directorApi.compileTimeline({ config: project })).resolves.toMatchObject({
      plans: [
        expect.objectContaining({ segment_id: project.segments[0].id }),
        expect.objectContaining({
          model_family: "ref2va",
          visible_frame_count: 124,
          sample_frame_count: 158,
          continuity_context_frames: 22,
          alignment_tail_frame_count: 12,
          predecessor_segment_id: project.segments[0].id,
          continuity_source: "same_run",
          historical_take_id: null,
          anchor_reset: false,
        }),
      ],
    });

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{
        ...report.plans[0],
        sample_frame_count: 158,
        continuity_context_frames: 22,
        alignment_tail_frame_count: 12,
        predecessor_segment_id: "historical-predecessor",
        continuity_source: "historical_take",
        historical_take_id: "take-2026-08-13",
        anchor_reset: false,
      }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).resolves.toMatchObject({
      plans: [expect.objectContaining({
        predecessor_segment_id: "historical-predecessor",
        continuity_source: "historical_take",
        historical_take_id: "take-2026-08-13",
      })],
    });

    const malformedSemanticReports = [
      {
        ...report,
        model_families: [],
      },
      {
        ...report,
        plans: [],
      },
      {
        ...report,
        plans: [report.plans[0], { ...report.plans[0] }],
      },
      {
        ...report,
        plans: [{
          ...report.plans[0],
          sample_frame_count: 141,
          continuity_context_frames: 5,
          alignment_tail_frame_count: 12,
          predecessor_segment_id: "missing-predecessor",
          anchor_reset: false,
        }],
      },
      {
        ...report,
        plans: [{
          ...report.plans[0],
          sample_frame_count: 141,
          continuity_context_frames: 5,
          alignment_tail_frame_count: 12,
          predecessor_segment_id: report.plans[0].segment_id,
          anchor_reset: false,
        }],
      },
      {
        ...report,
        plans: [{
          ...report.plans[0],
          sample_frame_count: 147,
          continuity_context_frames: 22,
          alignment_tail_frame_count: 1,
          predecessor_segment_id: "missing-predecessor",
          anchor_reset: false,
        }],
      },
      {
        ...report,
        plans: [{
          ...report.plans[0],
          sample_frame_count: 125,
          alignment_tail_frame_count: 1,
        }],
      },
    ];
    for (const malformed of malformedSemanticReports) {
      fetchMock.mockResolvedValueOnce(jsonResponse(malformed));
      await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(/执行计划/);
    }

    const { sample_frame_count: _omittedSampleFrames, ...legacyPlan } = report.plans[0];
    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [legacyPlan],
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划分段结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{ ...report.plans[0], sample_frame_count: 157 }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划分段结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{ ...report.plans[0], seed_mode: "random", seed: null }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划分段结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{ ...report.plans[0], seed: Number.MAX_SAFE_INTEGER + 1 }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划分段结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{ ...report.plans[0], seed_mode: "random_at_submit", seed: 7 }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划分段结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{ ...report.plans[0], model_family: "ref2va" }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划分段结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{ ...report.plans[0], recipe: "rv2v" }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划分段结构无效",
    );

    fetchMock.mockResolvedValueOnce(jsonResponse({
      ...report,
      plans: [{
        ...report.plans[0],
        conditioning_node: "MiniMaxH3ReferenceToVideo",
      }],
    }));
    await expect(directorApi.compileTimeline({ config: project })).rejects.toThrow(
      "执行计划分段结构无效",
    );
  });

  it("单删和批清使用 DELETE，并严格校验输出保留响应", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        deleted_job_id: "job/1",
        outputs_preserved: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        deleted_count: 3,
        active_count: 2,
        outputs_preserved: true,
      }));

    await expect(directorApi.deleteTask("job/1")).resolves.toEqual({
      deleted_job_id: "job/1",
      outputs_preserved: true,
    });
    await expect(directorApi.clearTerminalTasks()).resolves.toEqual({
      deleted_count: 3,
      active_count: 2,
      outputs_preserved: true,
    });
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "/directordeck/api/jobs/job%2F1",
      "/directordeck/api/jobs",
    ]);
    expect(fetchMock.mock.calls.every(([, init]) => init?.method === "DELETE")).toBe(true);
  });

  it("删除接口拒绝错误任务 ID、未保留输出和非整数计数", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        deleted_job_id: "another-job",
        outputs_preserved: true,
      }))
      .mockResolvedValueOnce(jsonResponse({
        deleted_count: 1,
        active_count: 0,
        outputs_preserved: false,
      }))
      .mockResolvedValueOnce(jsonResponse({
        deleted_count: 1.5,
        active_count: 0,
        outputs_preserved: true,
      }));

    await expect(directorApi.deleteTask("job-1")).rejects.toThrow("任务删除响应结构无效");
    await expect(directorApi.clearTerminalTasks()).rejects.toThrow("任务清理响应结构无效");
    await expect(directorApi.clearTerminalTasks()).rejects.toThrow("任务清理响应结构无效");
  });

  it("项目 CRUD 与项目级时间线端点使用稳定路由并严格解析", async () => {
    const summary = {
      id: "project-1",
      title: "第二部影片",
      created_at: "2026-08-12T00:00:00Z",
      updated_at: "2026-08-12T00:00:00Z",
      segment_count: 1,
    };
    const project = createTimelineProject();

    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        projects: [summary],
      }))
      .mockResolvedValueOnce(jsonResponse(summary))
      .mockResolvedValueOnce(jsonResponse(summary))
      .mockResolvedValueOnce(jsonResponse({
        deleted_project_id: "project-1",
        outputs_preserved: true,
        orphaned_jobs: 2,
      }))
      .mockResolvedValueOnce(jsonResponse(project))
      .mockResolvedValueOnce(jsonResponse(project));

    await expect(directorApi.listProjects()).resolves.toEqual({
      projects: [summary],
    });
    await expect(directorApi.createProject("第二部影片")).resolves.toEqual(summary);
    await expect(directorApi.renameProject("project-1", "改名")).resolves.toEqual(summary);
    await expect(directorApi.deleteProject("project-1")).resolves.toEqual({
      deleted_project_id: "project-1",
      outputs_preserved: true,
      orphaned_jobs: 2,
    });
    await expect(directorApi.getProjectTimeline("project-1")).resolves.toEqual(project);
    await expect(directorApi.updateProjectTimeline("project-1", project)).resolves.toEqual(project);

    expect(fetchMock.mock.calls.map(([url, init]) => [url, init?.method ?? "GET"])).toEqual([
      ["/directordeck/api/projects", "GET"],
      ["/directordeck/api/projects", "POST"],
      ["/directordeck/api/projects/project-1", "PATCH"],
      ["/directordeck/api/projects/project-1", "DELETE"],
      ["/directordeck/api/projects/project-1/timeline", "GET"],
      ["/directordeck/api/projects/project-1/timeline", "PUT"],
    ]);
  });

  it("项目列表拒绝多余的 envelope 字段", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({
        projects: [],
        extra: true,
      }));

    await expect(directorApi.listProjects()).rejects.toThrow("项目列表响应结构无效");
  });

  it("导入项目 POST 到 /api/projects/import 并严格解析摘要", async () => {
    const summary = {
      id: "project-2",
      title: "历史来源",
      created_at: "2026-08-12T00:00:00Z",
      updated_at: "2026-08-12T00:00:00Z",
      segment_count: 1,
    };
    const project = createTimelineProject();
    fetchMock.mockResolvedValueOnce(jsonResponse(summary));

    await expect(directorApi.importProject({ title: "历史来源", document: project }))
      .resolves.toEqual(summary);
    expect(fetchMock.mock.calls[0][0]).toBe("/directordeck/api/projects/import");
    expect(fetchMock.mock.calls[0][1]?.method).toBe("POST");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      title: "历史来源",
      document: project,
    });
  });
});
