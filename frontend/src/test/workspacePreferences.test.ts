import {
  loadTimelineRunSelectionPreference,
  loadTimelineWorkspacePreferences,
  saveTimelineRunSelectionPreference,
  saveTimelineWorkspacePreferences,
  TIMELINE_WORKSPACE_PREFERENCES_KEY,
} from "../domain/workspacePreferences";

const DATABASE = {
  active_database_path: "/srv/director/data/director.sqlite3",
  active_database_identity: "a".repeat(64),
};

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

  it("按数据库身份恢复运行集合，并区分全选、子集和明确空集合", () => {
    const project = ["first", "second", "disabled"];
    const enabled = ["first", "second"];

    saveTimelineRunSelectionPreference(DATABASE, project, enabled, ["second"]);
    expect(loadTimelineRunSelectionPreference(DATABASE, project, enabled)).toEqual(["second"]);
    expect(loadTimelineRunSelectionPreference(
      DATABASE,
      [...project, "new"],
      ["first", "second", "new"],
    )).toEqual(["second"]);

    saveTimelineRunSelectionPreference(DATABASE, project, enabled, enabled);
    expect(loadTimelineRunSelectionPreference(
      DATABASE,
      [...project, "new"],
      ["first", "second", "new"],
    )).toEqual(["first", "second", "new"]);

    saveTimelineRunSelectionPreference(DATABASE, project, enabled, []);
    expect(loadTimelineRunSelectionPreference(DATABASE, project, enabled)).toEqual([]);
    expect(loadTimelineRunSelectionPreference(DATABASE, ["replacement"], ["replacement"]))
      .toBeNull();
    expect(loadTimelineRunSelectionPreference({
      ...DATABASE,
      active_database_path: "/srv/director/other.sqlite3",
    }, project, enabled)).toBeNull();
    expect(loadTimelineRunSelectionPreference({
      ...DATABASE,
      active_database_identity: "b".repeat(64),
    }, project, enabled)).toBeNull();
  });
});
