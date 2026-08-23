import type { GenerationTask } from "../api/types";

/**
 * Project the newest durable child progress onto the task shown to the user.
 *
 * Native websocket events are persisted on the exact child immediately,
 * while the process reconciler intentionally updates the parent less often.
 * The child rows in JobRead are therefore the freshest display evidence. This
 * projection changes no server authority and never overrides cancellation or
 * terminal states.
 */
export function taskLivePresentation(task: GenerationTask): GenerationTask {
  if (!["queued", "preparing", "running"].includes(task.status)) return task;
  const segmentChildren = task.children.filter(
    (child) => child.segment_ids.length > 0,
  );
  const runningChild = segmentChildren.find(
    (child) => child.status === "running",
  );
  if (!runningChild) return task;

  const totalSegments = segmentChildren.reduce(
    (total, child) => total + child.segment_ids.length,
    0,
  );
  const childProgress = totalSegments > 0
    ? segmentChildren.reduce(
      (total, child) => total + child.progress * child.segment_ids.length,
      0,
    ) / totalSegments
    : task.progress;

  return {
    ...task,
    status: "running",
    progress: Math.max(task.progress, Math.min(1, Math.max(0, childProgress))),
    stage: runningChild.stage ?? task.stage,
  };
}
