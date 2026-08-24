import {
  autoFitSourceAudioTiming,
  runnableTimelineSegmentIds,
  timelineEditorReducer,
  type SourceAudioTimingAdjustment,
  type TimelineAction,
  type TimelineEditorState,
} from "../domain/timelineProject";

export type TimelineTransactionScope = "document" | "ui" | "authority" | "replay";
export type TimelineTransactionContextPolicy = "none" | "structural" | "text";
export type TimelineTransactionCoalescingPolicy = "preserve" | "seal" | "merge";

export interface TimelineTransactionPolicy {
  scope: TimelineTransactionScope;
  label: string;
  mergeKey?: string;
  context: TimelineTransactionContextPolicy;
  coalescing: TimelineTransactionCoalescingPolicy;
  applyDerivedNormalization: boolean;
}

export interface TimelineTransactionResult {
  policy: TimelineTransactionPolicy;
  next: TimelineEditorState;
  documentChanged: boolean;
  selectionChanged: boolean;
  topologyChanged: boolean;
  runnableSelectionChanged: boolean;
  derivedAdjustments: readonly SourceAudioTimingAdjustment[];
}

function assertNever(action: never): never {
  throw new Error(`Unhandled timeline action policy: ${JSON.stringify(action)}`);
}

function documentPolicy(
  label: string,
  options: {
    mergeKey?: string;
    context?: TimelineTransactionContextPolicy;
  } = {},
): TimelineTransactionPolicy {
  return {
    scope: "document",
    label,
    ...(options.mergeKey === undefined ? {} : { mergeKey: options.mergeKey }),
    context: options.context ?? "none",
    coalescing: options.mergeKey === undefined ? "seal" : "merge",
    applyDerivedNormalization: true,
  };
}

function uiPolicy(
  label: string,
  coalescing: TimelineTransactionCoalescingPolicy = "seal",
): TimelineTransactionPolicy {
  return {
    scope: "ui",
    label,
    context: "none",
    coalescing,
    applyDerivedNormalization: false,
  };
}

/**
 * Exhaustive policy for every reducer action. New actions must make an
 * explicit history decision here or TypeScript will reject the build.
 */
export function timelineTransactionPolicy(
  state: TimelineEditorState,
  action: TimelineAction,
): TimelineTransactionPolicy {
  switch (action.type) {
    case "project/patch":
      return Object.keys(action.patch).length === 1 && action.patch.title !== undefined
        ? documentPolicy("重命名项目")
        : documentPolicy("修改全局设置");
    case "project/replace":
      return documentPolicy("修改全局设置");
    case "project/update-sampling":
      return documentPolicy(
        `修改 ${action.family === "fl2va" ? "FL2VA" : "Ref2VA"} 推理参数`,
      );
    case "project/update-model":
      return documentPolicy("修改创作模型");
    case "feature/set-project":
      return documentPolicy(
        action.featureId === "lora"
          ? "修改 LoRA"
          : action.featureId === "comfy_kitchen_attention"
            ? "修改 CK Attention"
            : "修改项目扩展配置",
      );
    case "feature/set-segment":
      return documentPolicy("修改片段扩展配置");
    case "feature/clear-segment":
      return documentPolicy("清理不兼容片段配置", { context: "structural" });
    case "history/restore":
      return {
        scope: "replay",
        label: "回放项目历史",
        context: "structural",
        coalescing: "seal",
        applyDerivedNormalization: false,
      };
    case "segment/set-enabled":
      return documentPolicy(action.enabled ? "启用片段" : "停用片段");
    case "segment/insert":
      return documentPolicy("插入片段", { context: "structural" });
    case "segment/insert-video":
      return documentPolicy("导入视频片段", { context: "structural" });
    case "segment/insert-videos":
      return documentPolicy(
        action.assets.length > 1
          ? `导入 ${action.assets.length} 个视频片段`
          : "导入视频片段",
        { context: "structural" },
      );
    case "segment/move":
      return documentPolicy("移动片段", { context: "structural" });
    case "segment/delete-selected":
      return documentPolicy(
        state.selected_segment_ids.length > 1
          ? `删除 ${state.selected_segment_ids.length} 个片段`
          : "删除片段",
        { context: "structural" },
      );
    case "segment/merge-selected":
      return documentPolicy("合并片段", { context: "structural" });
    case "segment/split-selected":
    case "segment/apply-source-cuts":
      return documentPolicy("拆分片段", { context: "structural" });
    case "segment/split-evenly":
      return documentPolicy("均分片段", { context: "structural" });
    case "segment/duplicate-selected":
      return documentPolicy(
        state.selected_segment_ids.length > 1
          ? `复制 ${state.selected_segment_ids.length} 个片段`
          : "复制片段",
        { context: "structural" },
      );
    case "segment/replace": {
      const previous = state.project.segments.find(
        (segment) => segment.id === action.segment.id,
      );
      const changedKeys = previous
        ? Object.keys(action.segment).filter((key) =>
            !Object.is(
              previous[key as keyof typeof previous],
              action.segment[key as keyof typeof action.segment],
            ))
        : [];
      if (changedKeys.length === 1 && changedKeys[0] === "prompt") {
        return documentPolicy("编辑提示词", {
          mergeKey: `segment:${action.segment.id}:prompt`,
          context: "text",
        });
      }
      if (changedKeys.length === 1 && changedKeys[0] === "title") {
        return documentPolicy("重命名片段", {
          mergeKey: `segment:${action.segment.id}:title`,
          context: "text",
        });
      }
      return documentPolicy("修改片段");
    }
    case "segment/patch-base": {
      const changedKeys = Object.keys(action.patch);
      if (changedKeys.length === 1 && changedKeys[0] === "prompt") {
        return documentPolicy("编辑提示词", {
          mergeKey: `segment:${action.id}:prompt`,
          context: "text",
        });
      }
      if (changedKeys.length === 1 && changedKeys[0] === "title") {
        return documentPolicy("重命名片段", {
          mergeKey: `segment:${action.id}:title`,
          context: "text",
        });
      }
      return documentPolicy("修改片段");
    }
    case "segment/set-continuity":
      return documentPolicy("修改片段连续性");
    case "segment/remove-asset":
      return documentPolicy("移除参考素材");
    case "segment/set-source-range":
      return documentPolicy("调整源视频范围");
    case "segment/set-source-audio-reference":
      return documentPolicy("设置源视频音轨参考");
    case "segment/bind-asset":
    case "segment/bind-assets":
      return documentPolicy("绑定参考素材");
    case "segment/reorder-reference":
      return documentPolicy("调整参考素材顺序");
    case "segment/set-mode":
      return documentPolicy("切换生成模式");
    case "segment/insert-reference-token":
    case "segment/insert-subject-token":
      return documentPolicy("插入引用标签", { context: "text" });
    case "segment/apply-config":
      return documentPolicy("批量应用片段设置");
    case "segment/select":
    case "segment/toggle-selection":
    case "segment/set-selection":
      return uiPolicy("选择片段");
    case "assets/add":
      return uiPolicy("加入素材库");
    case "assets/replace":
      return uiPolicy("刷新素材库");
    case "assets/move":
      return uiPolicy("调整素材顺序");
    case "assets/select":
    case "assets/set-selection":
    case "assets/clear-selection":
      return uiPolicy("选择素材", "preserve");
    case "assets/remove":
      return uiPolicy("移出素材库");
    case "assets/grid-size":
      return uiPolicy("调整素材网格", "preserve");
    case "playhead/set":
      return uiPolicy("移动播放头", "preserve");
    default:
      return assertNever(action);
  }
}

function sameOrderedIds(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((id, index) => id === right[index]);
}

/**
 * Pure reducer gateway. Derived source-audio fitting is part of the same
 * document transaction and can never be triggered by a passive UI action.
 */
export function reduceTimelineTransaction(
  state: TimelineEditorState,
  action: TimelineAction,
  scopeOverride?: Extract<TimelineTransactionScope, "authority" | "replay">,
): TimelineTransactionResult {
  const declared = timelineTransactionPolicy(state, action);
  const policy = scopeOverride === undefined
    ? declared
    : {
        ...declared,
        scope: scopeOverride,
        context: scopeOverride === "replay" ? declared.context : "none" as const,
        coalescing: "seal" as const,
        applyDerivedNormalization: false,
      };
  const reduced = timelineEditorReducer(state, action);
  const fit = policy.applyDerivedNormalization
    ? autoFitSourceAudioTiming(reduced.project)
    : { project: reduced.project, adjustments: [] };
  const next = fit.project === reduced.project
    ? reduced
    : { ...reduced, project: fit.project };
  const currentRunnable = runnableTimelineSegmentIds(state);
  const nextRunnable = runnableTimelineSegmentIds(next);
  return {
    policy,
    next,
    documentChanged: next.project !== state.project,
    selectionChanged: !sameOrderedIds(
      state.selected_segment_ids,
      next.selected_segment_ids,
    ),
    topologyChanged: !sameOrderedIds(
      state.project.segments.map((segment) => segment.id),
      next.project.segments.map((segment) => segment.id),
    ),
    runnableSelectionChanged: !sameOrderedIds(currentRunnable, nextRunnable),
    derivedAdjustments: fit.adjustments,
  };
}
