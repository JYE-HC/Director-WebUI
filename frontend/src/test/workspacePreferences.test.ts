import {
  loadTimelineSegmentCopyOptions,
  loadTimelineSegmentSelectionPreference,
  loadTimelineWorkspacePreferences,
  saveTimelineSegmentCopyOptions,
  saveTimelineSegmentSelectionPreference,
  saveTimelineWorkspacePreferences,
  TIMELINE_SEGMENT_COPY_OPTIONS_KEY,
  TIMELINE_WORKSPACE_PREFERENCES_KEY,
} from "../domain/workspacePreferences";

const DATABASE = {
  active_database_path: "/srv/director/data/director.sqlite3",
};
const PROJECT_ID = "project-a";

beforeEach(() => localStorage.clear());

describe("时间线工作区浏览器偏好", () => {
  it("严格恢复版本化偏好，并逐项限制损坏或越界值", () => {
    expect(loadTimelineWorkspacePreferences()).toMatchObject({
      version: 1,
      showLiveMonitor: false,
      volume: 0.8,
      loop: false,
      compareOriginal: false,
      timelineZoom: 48,
      evenSplitPieces: 2,
      detectionSensitivity: "medium",
      minimumShotFrames: 12,
    });

    localStorage.setItem(TIMELINE_WORKSPACE_PREFERENCES_KEY, "{坏 JSON");
    expect(loadTimelineWorkspacePreferences().volume).toBe(0.8);

    localStorage.setItem(TIMELINE_WORKSPACE_PREFERENCES_KEY, JSON.stringify({
      version: 1,
      showLiveMonitor: true,
      volume: 4,
      loop: true,
      compareOriginal: true,
      timelineZoom: 0.01,
      evenSplitPieces: 999,
      detectionSensitivity: "invalid",
      minimumShotFrames: 1,
    }));
    expect(loadTimelineWorkspacePreferences()).toEqual({
      version: 1,
      showLiveMonitor: true,
      volume: 1,
      loop: true,
      compareOriginal: true,
      timelineZoom: 0.05,
      evenSplitPieces: 128,
      detectionSensitivity: "medium",
      minimumShotFrames: 4,
      assetFilter: "all",
      taskTab: "all",
      taskCurrentProjectOnly: false,
      taskSort: "recent",
    });

    localStorage.setItem(TIMELINE_WORKSPACE_PREFERENCES_KEY, JSON.stringify({ version: 2 }));
    expect(loadTimelineWorkspacePreferences().timelineZoom).toBe(48);
  });

  it("保存播放、预览和时间线工具偏好", () => {
    saveTimelineWorkspacePreferences({
      version: 1,
      showLiveMonitor: true,
      volume: 0.35,
      loop: true,
      compareOriginal: true,
      timelineZoom: 7.5,
      evenSplitPieces: 4,
      detectionSensitivity: "high",
      minimumShotFrames: 24,
      assetFilter: "video",
      taskTab: "completed",
      taskCurrentProjectOnly: true,
      taskSort: "duration",
    });
    expect(loadTimelineWorkspacePreferences()).toMatchObject({
      showLiveMonitor: true,
      volume: 0.35,
      loop: true,
      compareOriginal: true,
      timelineZoom: 7.5,
      evenSplitPieces: 4,
      detectionSensitivity: "high",
      minimumShotFrames: 24,
      assetFilter: "video",
      taskTab: "completed",
      taskCurrentProjectOnly: true,
      taskSort: "duration",
    });
  });

  it("版本化保存片段复制选项，并约束提示词引用素材依赖", () => {
    expect(loadTimelineSegmentCopyOptions()).toEqual({
      mode: true,
      duration: true,
      continuity: true,
      audioMode: true,
      refImageSize: true,
      prompt: false,
      promptReferences: false,
    });

    saveTimelineSegmentCopyOptions({
      mode: true,
      duration: false,
      continuity: false,
      audioMode: true,
      refImageSize: false,
      prompt: true,
      promptReferences: true,
    });
    expect(loadTimelineSegmentCopyOptions()).toEqual({
      mode: true,
      duration: false,
      continuity: false,
      audioMode: true,
      refImageSize: false,
      prompt: true,
      promptReferences: true,
    });

    localStorage.setItem(TIMELINE_SEGMENT_COPY_OPTIONS_KEY, JSON.stringify({
      version: 1,
      mode: false,
      prompt: true,
      promptReferences: true,
      duration: "损坏",
    }));
    expect(loadTimelineSegmentCopyOptions()).toMatchObject({
      mode: false,
      duration: true,
      prompt: true,
      promptReferences: false,
    });

    localStorage.setItem(TIMELINE_SEGMENT_COPY_OPTIONS_KEY, JSON.stringify({ version: 2 }));
    expect(loadTimelineSegmentCopyOptions().prompt).toBe(false);
  });

  it("按数据库与项目恢复统一选择，并保留停用片段、全选意图和明确空集合", () => {
    const project = ["first", "second", "disabled"];

    saveTimelineSegmentSelectionPreference(
      DATABASE,
      PROJECT_ID,
      project,
      ["second", "disabled"],
    );
    expect(loadTimelineSegmentSelectionPreference(DATABASE, PROJECT_ID, project))
      .toEqual(["second", "disabled"]);
    expect(loadTimelineSegmentSelectionPreference(
      DATABASE,
      PROJECT_ID,
      [...project, "new"],
    )).toEqual(["second", "disabled"]);

    saveTimelineSegmentSelectionPreference(DATABASE, PROJECT_ID, project, project);
    expect(loadTimelineSegmentSelectionPreference(
      DATABASE,
      PROJECT_ID,
      [...project, "new"],
    )).toEqual(["first", "second", "disabled", "new"]);

    saveTimelineSegmentSelectionPreference(DATABASE, PROJECT_ID, project, []);
    expect(loadTimelineSegmentSelectionPreference(DATABASE, PROJECT_ID, project)).toEqual([]);
    expect(loadTimelineSegmentSelectionPreference(DATABASE, PROJECT_ID, ["replacement"]))
      .toBeNull();
    expect(loadTimelineSegmentSelectionPreference({
      active_database_path: "/srv/director/other.sqlite3",
    }, PROJECT_ID, project)).toBeNull();
    expect(loadTimelineSegmentSelectionPreference({
      active_database_path: "",
    }, PROJECT_ID, project)).toBeNull();
    expect(loadTimelineSegmentSelectionPreference(DATABASE, "project-b", project)).toBeNull();
  });

  it("隔离旧运行选择 key 与损坏的 v2 envelope", () => {
    const project = ["first", "disabled"];
    localStorage.setItem(
      `directordeck:v1:timeline-run-selection:${DATABASE.active_database_path}`,
      JSON.stringify({ version: 1, selected_segment_ids: ["first"] }),
    );
    expect(loadTimelineSegmentSelectionPreference(DATABASE, PROJECT_ID, project)).toBeNull();

    localStorage.setItem(
      `directordeck:v2:timeline-segment-selection:${DATABASE.active_database_path}:${PROJECT_ID}`,
      JSON.stringify({ version: 1 }),
    );
    expect(loadTimelineSegmentSelectionPreference(DATABASE, PROJECT_ID, project)).toBeNull();
  });
});
