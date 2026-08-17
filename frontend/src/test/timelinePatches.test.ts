import {
  createTimelineProject,
  createTimelineSegment,
  type TimelineProject,
} from "../domain/timelineProject";
import {
  applyTimelinePatches,
  createTimelinePatchPair,
  timelineValuesEqual,
  type TimelinePatch,
} from "../state/timelinePatches";

function complexProject(): TimelineProject {
  const project = createTimelineProject();
  project.segments.push(createTimelineSegment("ref2va", 2));
  project.segments[0].prompt = "first";
  project.segments[1].prompt = "second";
  return project;
}

describe("timeline patches", () => {
  it("produces deterministic patches and exact forward/inverse round trips", () => {
    const before = complexProject();
    const after = structuredClone(before);
    after.title = "patched";
    after.segments.reverse();
    after.segments[0].prompt = "changed second";
    after.segments.push(createTimelineSegment("fl2va", 3));

    const first = createTimelinePatchPair(before, after);
    const second = createTimelinePatchPair(
      structuredClone(before),
      structuredClone(after),
    );

    expect(first).toEqual(second);
    expect(applyTimelinePatches(before, first.forward)).toEqual(after);
    expect(applyTimelinePatches(after, first.inverse)).toEqual(before);
    expect(before.segments.map((segment) => segment.prompt)).toEqual(["first", "second"]);
  });

  it("isolates patch values and every applied result from caller mutation", () => {
    const before = complexProject();
    const after = structuredClone(before);
    after.render = { ...after.render, width: 1280, height: 720 };
    const pair = createTimelinePatchPair(before, after);

    after.render.width = 32;
    const applied = applyTimelinePatches(before, pair.forward);
    expect(applied.render.width).toBe(1280);
    applied.render.width = 64;
    expect(applyTimelinePatches(before, pair.forward).render.width).toBe(1280);
  });

  it("handles object removal, array shrink and inverse array growth in valid order", () => {
    const before = complexProject();
    const after = structuredClone(before);
    after.segments = [after.segments[1]];
    const pair = createTimelinePatchPair(before, after);

    expect(applyTimelinePatches(before, pair.forward)).toEqual(after);
    expect(applyTimelinePatches(after, pair.inverse)).toEqual(before);
    expect(pair.forward.some((patch) => patch.op === "remove")).toBe(true);
  });

  it("rejects unsafe or invalid external patch paths", () => {
    const project = createTimelineProject();
    const unsafe: TimelinePatch[] = [{
      op: "set",
      path: ["__proto__", "polluted"],
      value: true,
    }];
    expect(() => applyTimelinePatches(project, unsafe)).toThrow(/Invalid timeline patch path/);
    expect(() => applyTimelinePatches(project, [{
      op: "remove",
      path: ["segments", 99],
    }])).toThrow(/out of bounds/);
  });

  it("compares object keys semantically rather than by insertion order", () => {
    expect(timelineValuesEqual(
      { title: "same", nested: { one: 1, two: 2 } },
      { nested: { two: 2, one: 1 }, title: "same" },
    )).toBe(true);
  });
});
