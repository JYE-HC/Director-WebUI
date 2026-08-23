import type { TimelineProject } from "../domain/timelineProject";

/** Schema 2 adds model_stack/features and never replays v4 pointers in place. */
export const TIMELINE_PATCH_SCHEMA_VERSION = 2;

export type TimelinePatchPathPart = string | number;
export type TimelinePatchPath = readonly TimelinePatchPathPart[];

export type TimelinePatch =
  | {
      op: "set";
      path: TimelinePatchPath;
      value: unknown;
    }
  | {
      op: "remove";
      path: TimelinePatchPath;
    };

export interface TimelinePatchPair {
  forward: readonly TimelinePatch[];
  inverse: readonly TimelinePatch[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Timeline projects are JSON-shaped, so a deterministic structural comparison is sufficient. */
export function timelineValuesEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (typeof left !== typeof right || left === null || right === null) return false;
  if (typeof left !== "object") return false;

  if (Array.isArray(left) || Array.isArray(right)) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length)
      return false;
    return left.every((value, index) => timelineValuesEqual(value, right[index]));
  }

  const leftRecord = left as Record<string, unknown>;
  const rightRecord = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftRecord).sort();
  const rightKeys = Object.keys(rightRecord).sort();
  if (leftKeys.length !== rightKeys.length) return false;
  return leftKeys.every(
    (key, index) => key === rightKeys[index] &&
      timelineValuesEqual(leftRecord[key], rightRecord[key]),
  );
}

function clonePatchValue<T>(value: T): T {
  return structuredClone(value);
}

function diffTimelineValue(
  before: unknown,
  after: unknown,
  path: TimelinePatchPathPart[],
  patches: TimelinePatch[],
): void {
  if (timelineValuesEqual(before, after)) return;

  if (Array.isArray(before) && Array.isArray(after)) {
    const commonLength = Math.min(before.length, after.length);
    for (let index = 0; index < commonLength; index += 1) {
      diffTimelineValue(before[index], after[index], [...path, index], patches);
    }
    // Tail removals run backwards so every recorded index remains valid.
    for (let index = before.length - 1; index >= after.length; index -= 1) {
      patches.push({ op: "remove", path: [...path, index] });
    }
    // Array additions are always appended after the common prefix has settled.
    for (let index = before.length; index < after.length; index += 1) {
      patches.push({
        op: "set",
        path: [...path, index],
        value: clonePatchValue(after[index]),
      });
    }
    return;
  }

  if (isRecord(before) && isRecord(after)) {
    const beforeKeys = Object.keys(before).sort();
    const afterKeys = Object.keys(after).sort();
    const beforeSet = new Set(beforeKeys);
    const afterSet = new Set(afterKeys);

    for (const key of beforeKeys) {
      if (!afterSet.has(key)) patches.push({ op: "remove", path: [...path, key] });
    }
    for (const key of beforeKeys) {
      if (afterSet.has(key)) {
        diffTimelineValue(before[key], after[key], [...path, key], patches);
      }
    }
    for (const key of afterKeys) {
      if (!beforeSet.has(key)) {
        patches.push({
          op: "set",
          path: [...path, key],
          value: clonePatchValue(after[key]),
        });
      }
    }
    return;
  }

  patches.push({ op: "set", path: [...path], value: clonePatchValue(after) });
}

/** Creates deterministic forward and inverse patches without retaining either input snapshot. */
export function createTimelinePatchPair(
  before: TimelineProject,
  after: TimelineProject,
): TimelinePatchPair {
  const forward: TimelinePatch[] = [];
  const inverse: TimelinePatch[] = [];
  diffTimelineValue(before, after, [], forward);
  diffTimelineValue(after, before, [], inverse);
  return { forward, inverse };
}

const UNSAFE_PATH_PARTS = new Set(["__proto__", "prototype", "constructor"]);

function assertSafePath(path: TimelinePatchPath): void {
  for (const part of path) {
    if (
      (typeof part === "string" && UNSAFE_PATH_PARTS.has(part)) ||
      (typeof part === "number" && (!Number.isInteger(part) || part < 0))
    ) throw new Error("Invalid timeline patch path");
  }
}

function resolvePatchParent(
  root: unknown,
  path: TimelinePatchPath,
): { parent: Record<string, unknown> | unknown[]; key: TimelinePatchPathPart } {
  if (!path.length) throw new Error("A root patch has no parent");
  let current = root;
  for (const part of path.slice(0, -1)) {
    if (typeof part === "number") {
      if (!Array.isArray(current) || part >= current.length) {
        throw new Error("Timeline patch array path is out of bounds");
      }
      current = current[part];
    } else {
      if (!isRecord(current) || !Object.prototype.hasOwnProperty.call(current, part)) {
        throw new Error("Timeline patch object path does not exist");
      }
      current = current[part];
    }
  }
  if (!Array.isArray(current) && !isRecord(current)) {
    throw new Error("Timeline patch parent is not a container");
  }
  return { parent: current, key: path[path.length - 1] };
}

/** Applies patches to an isolated clone and never mutates the supplied project or patch values. */
export function applyTimelinePatches(
  project: TimelineProject,
  patches: readonly TimelinePatch[],
): TimelineProject {
  let root: unknown = structuredClone(project);
  for (const patch of patches) {
    assertSafePath(patch.path);
    if (!patch.path.length) {
      if (patch.op === "remove") throw new Error("Cannot remove the timeline project root");
      root = clonePatchValue(patch.value);
      continue;
    }

    const { parent, key } = resolvePatchParent(root, patch.path);
    if (Array.isArray(parent)) {
      if (typeof key !== "number") throw new Error("Timeline array patches require numeric indexes");
      if (patch.op === "remove") {
        if (key >= parent.length) throw new Error("Timeline patch removal is out of bounds");
        parent.splice(key, 1);
      } else {
        if (key > parent.length) throw new Error("Timeline patch insertion is out of bounds");
        if (key === parent.length) parent.push(clonePatchValue(patch.value));
        else parent[key] = clonePatchValue(patch.value);
      }
      continue;
    }

    if (typeof key !== "string") throw new Error("Timeline object patches require string keys");
    if (patch.op === "remove") {
      if (!Object.prototype.hasOwnProperty.call(parent, key)) {
        throw new Error("Timeline patch removal target does not exist");
      }
      delete parent[key];
    } else {
      parent[key] = clonePatchValue(patch.value);
    }
  }
  return root as TimelineProject;
}

/** Exact UTF-8 size used by the history memory budget. */
export function timelineSerializedBytes(value: unknown): number {
  return new TextEncoder().encode(JSON.stringify(value)).byteLength;
}
