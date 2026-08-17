import {
  createTimelineProject,
  createTimelineSegment,
  type TimelineProject,
} from "../domain/timelineProject";
import {
  DEFAULT_TIMELINE_HISTORY_BYTE_BUDGET,
  DEFAULT_TIMELINE_HISTORY_CAPACITY,
  DEFAULT_TIMELINE_HISTORY_COALESCE_WINDOW_MS,
  MAX_TIMELINE_HISTORY_CAPACITY,
  MAX_TIMELINE_HISTORY_BYTE_BUDGET,
  MIN_TIMELINE_HISTORY_CAPACITY,
  TIMELINE_HISTORY_CHECKPOINT_INTERVAL,
  TIMELINE_HISTORY_SCHEMA_VERSION,
  canRedoTimelineHistory,
  canSafelyRebaseTimelineHistoryHead,
  canUndoTimelineHistory,
  captureTimelineHistoryContext,
  createTimelineHistory,
  deserializeTimelineHistory,
  jumpTimelineHistory,
  recordTimelineHistory,
  rebaseTimelineHistoryHead,
  redoTimelineHistory,
  resetTimelineHistory,
  sealTimelineHistoryCoalescing,
  serializeTimelineHistory,
  timelineHistoryCursor,
  timelineHistoryEntries,
  timelineHistoryLength,
  timelineHistoryRedoLabel,
  timelineHistoryUndoLabel,
  undoTimelineHistory,
  type TimelineHistoryContext,
  type SerializedTimelineHistoryEnvelope,
} from "../state/timelineHistory";

function titled(project: TimelineProject, title: string): TimelineProject {
  return { ...project, title };
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value).sort().map((key) =>
    `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`
  ).join(",")}}`;
}

function rehashTimelineHistoryEnvelope(
  envelope: SerializedTimelineHistoryEnvelope,
): SerializedTimelineHistoryEnvelope {
  const input = canonicalJson({
    format: envelope.format,
    version: envelope.version,
    schemaVersion: envelope.schemaVersion,
    payload: envelope.payload,
  });
  let hash = 0xcbf29ce484222325n;
  for (const byte of new TextEncoder().encode(input)) {
    hash ^= BigInt(byte);
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  envelope.hash = `fnv1a64:${hash.toString(16).padStart(16, "0")}`;
  return envelope;
}

function context(
  selected: string[],
  active: string | null = selected[0] ?? null,
  anchor: string | null = active,
  options: Pick<TimelineHistoryContext, "restore_segment_selection" | "text_editing"> = {},
): TimelineHistoryContext {
  return {
    selected_segment_ids: selected,
    active_segment_id: active,
    selection_anchor_id: anchor,
    ...options,
  };
}

describe("timeline history", () => {
  it("uses 100 entries and 16 MiB by default with validated configurable limits", () => {
    const history = createTimelineHistory();
    expect(history.capacity).toBe(DEFAULT_TIMELINE_HISTORY_CAPACITY);
    expect(history.byteBudget).toBe(DEFAULT_TIMELINE_HISTORY_BYTE_BUDGET);
    expect(DEFAULT_TIMELINE_HISTORY_BYTE_BUDGET).toBe(16 * 1024 * 1024);
    expect(MAX_TIMELINE_HISTORY_BYTE_BUDGET).toBe(DEFAULT_TIMELINE_HISTORY_BYTE_BUDGET);
    expect(DEFAULT_TIMELINE_HISTORY_COALESCE_WINDOW_MS).toBe(800);
    expect(createTimelineHistory(MIN_TIMELINE_HISTORY_CAPACITY).capacity).toBe(50);
    expect(createTimelineHistory(75, 4_096).byteBudget).toBe(4_096);
    expect(createTimelineHistory(MAX_TIMELINE_HISTORY_CAPACITY).capacity).toBe(100);
    expect(() => createTimelineHistory(49)).toThrow(RangeError);
    expect(() => createTimelineHistory(101)).toThrow(RangeError);
    expect(() => createTimelineHistory(75.5)).toThrow(RangeError);
    expect(() => createTimelineHistory(75, 0)).toThrow(RangeError);
    expect(() => createTimelineHistory(
      75,
      MAX_TIMELINE_HISTORY_BYTE_BUDGET + 1,
    )).toThrow(RangeError);
  });

  it("stores isolated patches and metadata rather than before/after snapshot pairs", () => {
    const initial = createTimelineProject();
    const edited = titled(initial, "第一次编辑");
    const segmentId = initial.segments[0].id;
    const beforeContext = context([segmentId]);
    const afterContext = context([], null, null);

    const history = recordTimelineHistory(createTimelineHistory(), {
      label: "重命名项目",
      before: initial,
      after: edited,
      beforeContext,
      afterContext,
      now: 1_234,
    });

    edited.title = "调用方后续突变";
    beforeContext.selected_segment_ids.length = 0;
    const entry = history.past[0];
    expect(entry).not.toHaveProperty("before");
    expect(entry).not.toHaveProperty("after");
    expect(entry).toMatchObject({
      id: "timeline-history-1",
      label: "重命名项目",
      timestamp: 1_234,
      schemaVersion: TIMELINE_HISTORY_SCHEMA_VERSION,
      affectedSegmentIds: [],
    });
    expect(entry.forward.length).toBeGreaterThan(0);
    expect(entry.inverse.length).toBeGreaterThan(0);
    expect(entry.byteSize).toBeGreaterThan(0);
    expect(history.totalBytes).toBeGreaterThan(entry.byteSize);
    expect(canUndoTimelineHistory(history)).toBe(true);
    expect(timelineHistoryUndoLabel(history)).toBe("重命名项目");

    const undone = undoTimelineHistory(history)!;
    expect(undone.snapshot.project.title).toBe(initial.title);
    expect(undone.snapshot.context).toEqual(context([segmentId]));
    expect(canUndoTimelineHistory(undone.history)).toBe(false);
    expect(canRedoTimelineHistory(undone.history)).toBe(true);
    expect(timelineHistoryRedoLabel(undone.history)).toBe("重命名项目");

    const redone = redoTimelineHistory(undone.history)!;
    expect(redone.snapshot.project.title).toBe("第一次编辑");
    expect(redone.snapshot.context).toEqual(afterContext);
    redone.snapshot.project.title = "不能污染历史";
    expect(undoTimelineHistory(redone.history)?.snapshot.project.title).toBe(initial.title);
  });

  it("captures structural context explicitly without unrelated editor state", () => {
    const selected = ["segment-a", "segment-b"];
    const captured = captureTimelineHistoryContext({
      selected_segment_ids: selected,
      active_segment_id: "segment-b",
      selection_anchor_id: "segment-a",
    });
    selected.length = 0;
    expect(captured).toEqual({
      selected_segment_ids: ["segment-a", "segment-b"],
      active_segment_id: "segment-b",
      selection_anchor_id: "segment-a",
      restore_segment_selection: true,
    });
  });

  it("does not record semantic no-ops or context-only changes", () => {
    const project = createTimelineProject();
    const history = createTimelineHistory();
    const result = recordTimelineHistory(history, {
      label: "只改选择",
      before: project,
      after: structuredClone(project),
      beforeContext: context([project.segments[0].id]),
      afterContext: context([], null, null),
    });

    expect(result).toBe(history);
    expect(undoTimelineHistory(result)).toBeNull();
    expect(redoTimelineHistory(result)).toBeNull();
  });

  it("coalesces matching text keys, preserving first before caret and final after caret", () => {
    const initial = createTimelineProject();
    const first = titled(initial, "导");
    const second = titled(first, "导演");
    const beforeText = context([initial.segments[0].id], undefined, undefined, {
      restore_segment_selection: false,
      text_editing: { field_key: "segment:1:prompt", start: 0, end: 0, direction: "none" },
    });
    const middleText = context([initial.segments[0].id], undefined, undefined, {
      restore_segment_selection: false,
      text_editing: { field_key: "segment:1:prompt", start: 1, end: 1, direction: "none" },
    });
    const finalText = context([initial.segments[0].id], undefined, undefined, {
      restore_segment_selection: false,
      text_editing: { field_key: "segment:1:prompt", start: 2, end: 2, direction: "none" },
    });
    let history = recordTimelineHistory(createTimelineHistory(), {
      label: "编辑项目名称",
      before: initial,
      after: first,
      beforeContext: beforeText,
      afterContext: middleText,
      mergeKey: "project:title",
      now: 1_000,
    });
    history = recordTimelineHistory(history, {
      label: "编辑项目名称",
      before: first,
      after: second,
      beforeContext: middleText,
      afterContext: finalText,
      mergeKey: "project:title",
      now: 1_500,
    });

    expect(history.past).toHaveLength(1);
    expect(history.past[0].beforeContext).toEqual(beforeText);
    expect(history.past[0].afterContext).toEqual(finalText);
    const undone = undoTimelineHistory(history)!;
    expect(undone.snapshot.project.title).toBe(initial.title);
    expect(undone.snapshot.context).toEqual(beforeText);
    const redone = redoTimelineHistory(undone.history)!;
    expect(redone.snapshot.project.title).toBe("导演");
    expect(redone.snapshot.context).toEqual(finalText);
  });

  it("starts a new entry after the coalesce window expires or a session is sealed", () => {
    const initial = createTimelineProject();
    const first = titled(initial, "一");
    const second = titled(first, "二");
    const third = titled(second, "三");
    let history = recordTimelineHistory(createTimelineHistory(), {
      label: "编辑项目名称",
      before: initial,
      after: first,
      mergeKey: "project:title",
      now: 1_000,
    });
    history = recordTimelineHistory(history, {
      label: "编辑项目名称",
      before: first,
      after: second,
      mergeKey: "project:title",
      now: 1_801,
    });
    expect(history.past).toHaveLength(2);

    const past = history.past;
    const future = history.future;
    history = sealTimelineHistoryCoalescing(history);
    expect(history.past).toBe(past);
    expect(history.future).toBe(future);
    expect(history.coalescing).toBeNull();
    history = recordTimelineHistory(history, {
      label: "编辑项目名称",
      before: second,
      after: third,
      mergeKey: "project:title",
      now: 1_900,
    });
    expect(history.past).toHaveLength(3);
  });

  it("does not coalesce different keys and resets a discontinuous caller chain", () => {
    const initial = createTimelineProject();
    const first = titled(initial, "一");
    const second = titled(first, "二");
    const third = titled(first, "三");
    let history = recordTimelineHistory(createTimelineHistory(), {
      label: "第一项",
      before: initial,
      after: first,
      mergeKey: "project:title",
    });
    history = recordTimelineHistory(history, {
      label: "第二项",
      before: first,
      after: second,
      mergeKey: "segment:title",
    });
    expect(history.past).toHaveLength(2);

    history = recordTimelineHistory(history, {
      label: "断链项",
      before: first,
      after: third,
      mergeKey: "segment:title",
    });
    expect(history.past).toHaveLength(1);
    expect(undoTimelineHistory(history)?.snapshot.project.title).toBe("一");
  });

  it("removes a coalesced entry when typing returns to its original project", () => {
    const initial = createTimelineProject();
    const edited = titled(initial, "临时名称");
    let history = recordTimelineHistory(createTimelineHistory(), {
      label: "编辑项目名称",
      before: initial,
      after: edited,
      mergeKey: "project:title",
    });
    history = recordTimelineHistory(history, {
      label: "编辑项目名称",
      before: edited,
      after: structuredClone(initial),
      mergeKey: "project:title",
    });

    expect(history.past).toEqual([]);
    expect(history.checkpoints).toEqual([]);
    expect(history.totalBytes).toBe(0);
    expect(canUndoTimelineHistory(history)).toBe(false);
  });

  it("clears redo on a real branch while a no-op preserves it", () => {
    const initial = createTimelineProject();
    const first = titled(initial, "一");
    const second = titled(first, "二");
    let history = recordTimelineHistory(createTimelineHistory(), {
      label: "第一项",
      before: initial,
      after: first,
    });
    history = recordTimelineHistory(history, {
      label: "第二项",
      before: first,
      after: second,
    });
    history = undoTimelineHistory(history)!.history;
    expect(history.future).toHaveLength(1);

    const afterNoOp = recordTimelineHistory(history, {
      label: "无变化",
      before: first,
      after: structuredClone(first),
    });
    expect(afterNoOp).toBe(history);
    expect(afterNoOp.future).toHaveLength(1);

    const branch = titled(first, "分支");
    history = recordTimelineHistory(afterNoOp, {
      label: "分支编辑",
      before: first,
      after: branch,
    });
    expect(history.future).toEqual([]);
    expect(timelineHistoryUndoLabel(history)).toBe("分支编辑");
  });

  it("trims oldest records at capacity while retaining a reconstructable base", () => {
    const capacity = MIN_TIMELINE_HISTORY_CAPACITY;
    let history = createTimelineHistory(capacity);
    let current = titled(createTimelineProject(), "0");
    for (let index = 1; index <= capacity + 5; index += 1) {
      const next = titled(current, String(index));
      history = recordTimelineHistory(history, {
        label: `编辑 ${index}`,
        before: current,
        after: next,
      });
      current = next;
    }

    expect(history.past).toHaveLength(capacity);
    expect(history.startIndex).toBe(5);
    for (let index = 0; index < capacity; index += 1) {
      const replay = undoTimelineHistory(history)!;
      history = replay.history;
      current = replay.snapshot.project;
    }
    expect(current.title).toBe("5");
    expect(undoTimelineHistory(history)).toBeNull();
  });

  it("enforces the byte budget in addition to capacity", () => {
    const byteBudget = 4_096;
    let history = createTimelineHistory(MAX_TIMELINE_HISTORY_CAPACITY, byteBudget);
    let current = createTimelineProject();
    for (let index = 0; index < 20; index += 1) {
      const next = titled(current, `${index}:${"长".repeat(300)}`);
      history = recordTimelineHistory(history, {
        label: `大编辑 ${index}`,
        before: current,
        after: next,
      });
      current = next;
    }
    expect(history.totalBytes).toBeLessThanOrEqual(byteBudget);
    expect(history.past.length).toBeLessThan(20);
  });

  it("stores sparse checkpoints every twenty committed transactions", () => {
    let history = createTimelineHistory();
    let current = createTimelineProject();
    for (let index = 1; index <= 45; index += 1) {
      const next = titled(current, String(index));
      history = recordTimelineHistory(history, {
        label: `编辑 ${index}`,
        before: current,
        after: next,
      });
      current = next;
    }
    expect(TIMELINE_HISTORY_CHECKPOINT_INTERVAL).toBe(20);
    expect(history.checkpoints.map((checkpoint) => checkpoint.position)).toEqual([0, 20, 40]);
  });

  it("jumps to arbitrary retained cursors from checkpoints in one replay", () => {
    let history = createTimelineHistory();
    const projects = [createTimelineProject()];
    for (let index = 1; index <= 45; index += 1) {
      const next = titled(projects.at(-1)!, String(index));
      history = recordTimelineHistory(history, {
        label: `编辑 ${index}`,
        before: projects.at(-1)!,
        after: next,
      });
      projects.push(next);
    }
    expect(timelineHistoryCursor(history)).toBe(45);
    expect(timelineHistoryLength(history)).toBe(45);
    expect(timelineHistoryEntries(history)).toHaveLength(45);

    const seventh = jumpTimelineHistory(history, 7)!;
    expect(seventh.snapshot.project).toEqual(projects[7]);
    expect(seventh.cursor).toBe(7);
    expect(seventh.history.past).toHaveLength(7);
    expect(seventh.history.future).toHaveLength(38);
    const thirtyThird = jumpTimelineHistory(seventh.history, 33)!;
    expect(thirtyThird.snapshot.project).toEqual(projects[33]);
    expect(jumpTimelineHistory(thirtyThird.history, 33)).toBeNull();
    expect(() => jumpTimelineHistory(history, 46)).toThrow(RangeError);
  });

  it("safely rebases canonical ACKs at the current head and preserves undo/redo", () => {
    const initial = createTimelineProject();
    const client = titled(initial, "客户端标题");
    const confirmed = titled(client, "客户端标题（规范）");
    const history = recordTimelineHistory(createTimelineHistory(), {
      label: "重命名",
      before: initial,
      after: client,
    });

    expect(canSafelyRebaseTimelineHistoryHead(client, confirmed)).toBe(true);
    const rebased = rebaseTimelineHistoryHead(history, client, confirmed)!;
    expect(undoTimelineHistory(rebased)?.snapshot.project).toEqual(initial);
    const undone = undoTimelineHistory(rebased)!;
    expect(redoTimelineHistory(undone.history)?.snapshot.project).toEqual(confirmed);
  });

  it("rebases both transitions around an undone cursor", () => {
    const initial = createTimelineProject();
    const first = titled(initial, "一");
    const second = titled(first, "二");
    let history = recordTimelineHistory(createTimelineHistory(), {
      label: "第一项", before: initial, after: first,
    });
    history = recordTimelineHistory(history, {
      label: "第二项", before: first, after: second,
    });
    history = undoTimelineHistory(history)!.history;
    const canonicalFirst = titled(first, "一（规范）");
    const rebased = rebaseTimelineHistoryHead(history, first, canonicalFirst)!;

    expect(undoTimelineHistory(rebased)?.snapshot.project).toEqual(initial);
    expect(redoTimelineHistory(rebased)?.snapshot.project).toEqual(second);
  });

  it("rejects rebase when the expected head, segment topology, or material identity is unsafe", () => {
    const initial = createTimelineProject();
    const client = titled(initial, "客户端");
    const history = recordTimelineHistory(createTimelineHistory(), {
      label: "编辑", before: initial, after: client,
    });
    const staleExpected = titled(client, "别的 head");
    expect(rebaseTimelineHistoryHead(history, staleExpected, client)).toBeNull();

    const changedIdentity = structuredClone(client);
    changedIdentity.segments[0].id = "server-replaced-id";
    expect(canSafelyRebaseTimelineHistoryHead(client, changedIdentity)).toBe(false);
    expect(rebaseTimelineHistoryHead(history, client, changedIdentity)).toBeNull();

    const refExpected = structuredClone(client);
    refExpected.segments = [createTimelineSegment("ref2va", 1)];
    const refConfirmed = structuredClone(refExpected);
    if (refConfirmed.segments[0]?.mode !== "ref2va") throw new Error("invalid fixture");
    refConfirmed.segments[0].source_audio_as_reference = true;
    expect(canSafelyRebaseTimelineHistoryHead(refExpected, refConfirmed)).toBe(false);
  });

  it("round-trips deterministic randomized edit sequences and random jumps", () => {
    let seed = 0x5eed1234;
    const random = () => {
      seed = (Math.imul(seed, 1_664_525) + 1_013_904_223) >>> 0;
      return seed;
    };
    const initial = createTimelineProject();
    initial.segments.push(createTimelineSegment("ref2va", 2));
    const projects: TimelineProject[] = [initial];
    let history = createTimelineHistory();
    for (let index = 1; index <= 80; index += 1) {
      const before = projects.at(-1)!;
      const after = structuredClone(before);
      const choice = random() % 5;
      if (choice === 0) after.title = `project-${index}-${random()}`;
      else if (choice === 1) after.segments[0].prompt = `prompt-${index}-${random()}`;
      else if (choice === 2) after.segments[1].enabled = !after.segments[1].enabled;
      else if (choice === 3) after.export_mode = after.export_mode === "all" ? "segments" : "all";
      else after.segments.reverse();
      history = recordTimelineHistory(history, {
        label: `随机编辑 ${index}`,
        before,
        after,
        now: index * 1_000,
      });
      projects.push(after);
    }

    const complete = history;
    for (let cursor = 79; cursor >= 0; cursor -= 1) {
      const replay = undoTimelineHistory(history)!;
      expect(replay.snapshot.project).toEqual(projects[cursor]);
      history = replay.history;
    }
    for (let cursor = 1; cursor <= 80; cursor += 1) {
      const replay = redoTimelineHistory(history)!;
      expect(replay.snapshot.project).toEqual(projects[cursor]);
      history = replay.history;
    }
    for (let iteration = 0; iteration < 25; iteration += 1) {
      const cursor = random() % 81;
      const jumped = jumpTimelineHistory(complete, cursor);
      if (cursor === 80) expect(jumped).toBeNull();
      else expect(jumped?.snapshot.project).toEqual(projects[cursor]);
    }
  });

  it("round-trips a checkpointed history at an undone cursor through JSON", () => {
    let history = createTimelineHistory();
    let current = createTimelineProject();
    const segmentId = current.segments[0].id;
    for (let index = 1; index <= 25; index += 1) {
      const next = titled(current, `持久化-${index}`);
      history = recordTimelineHistory(history, {
        label: `编辑 ${index}`,
        before: current,
        after: next,
        beforeContext: index === 1
          ? context([segmentId], segmentId, segmentId, {
              restore_segment_selection: false,
              text_editing: {
                field_key: `segment:${segmentId}:prompt`,
                start: 0,
                end: 0,
                direction: "none",
              },
            })
          : undefined,
        now: index * 1_000,
      });
      current = next;
    }
    for (let index = 0; index < 7; index += 1) {
      history = undoTimelineHistory(history)!.history;
    }

    const envelope = serializeTimelineHistory(history);
    const jsonRoundTrip = JSON.parse(JSON.stringify(envelope)) as unknown;
    const restored = deserializeTimelineHistory(jsonRoundTrip, {
      expectedHead: history.head!,
      expectedSchema: TIMELINE_HISTORY_SCHEMA_VERSION,
    });

    expect(envelope.hash).toMatch(/^fnv1a64:[0-9a-f]{16}$/);
    expect(envelope.payload.cursor).toBe(18);
    expect(envelope.payload.checkpoints.map((checkpoint) => checkpoint.position)).toEqual([0, 20]);
    expect(restored).toEqual(history);
    expect(undoTimelineHistory(restored!)?.snapshot.project).toEqual(
      undoTimelineHistory(history)!.snapshot.project,
    );
    expect(jumpTimelineHistory(restored!, 25)?.snapshot.project.title).toBe("持久化-25");
  });

  it("fails closed for corrupt envelopes, schema drift, and a mismatched current head", () => {
    const initial = createTimelineProject();
    const edited = titled(initial, "已编辑");
    const history = recordTimelineHistory(createTimelineHistory(), {
      label: "编辑",
      before: initial,
      after: edited,
    });
    const envelope = serializeTimelineHistory(history);

    const badHash = structuredClone(envelope);
    badHash.hash = "fnv1a64:0000000000000000";
    expect(deserializeTimelineHistory(badHash)).toBeNull();

    const badByteSize = structuredClone(envelope);
    badByteSize.payload.past[0].byteSize += 1;
    expect(deserializeTimelineHistory(badByteSize)).toBeNull();

    const extraKey = structuredClone(envelope) as unknown as Record<string, unknown>;
    extraKey.unexpected = true;
    expect(deserializeTimelineHistory(extraKey)).toBeNull();

    expect(deserializeTimelineHistory(envelope, { expectedSchema: 999 })).toBeNull();
    expect(deserializeTimelineHistory(envelope, {
      expectedHead: titled(edited, "另一个 head"),
    })).toBeNull();
    expect(deserializeTimelineHistory(null)).toBeNull();
  });

  it("rejects a validly rehashed envelope that exceeds hard resource and ID limits", () => {
    const initial = createTimelineProject();
    const edited = titled(initial, "已编辑");
    const history = recordTimelineHistory(createTimelineHistory(), {
      label: "编辑",
      before: initial,
      after: edited,
    });

    const oversizedBudget = structuredClone(serializeTimelineHistory(history));
    oversizedBudget.payload.byteBudget = MAX_TIMELINE_HISTORY_BYTE_BUDGET + 1;
    expect(deserializeTimelineHistory(
      rehashTimelineHistoryEnvelope(oversizedBudget),
    )).toBeNull();

    const exhaustedId = structuredClone(serializeTimelineHistory(history));
    exhaustedId.payload.nextEntryId = Number.MAX_SAFE_INTEGER;
    expect(deserializeTimelineHistory(
      rehashTimelineHistoryEnvelope(exhaustedId),
    )).toBeNull();
  });

  it("refuses to allocate an entry ID that would leave an unusable counter", () => {
    const initial = createTimelineProject();
    const exhausted = {
      ...createTimelineHistory(),
      nextEntryId: Number.MAX_SAFE_INTEGER - 1,
    };
    expect(() => recordTimelineHistory(exhausted, {
      label: "不可分配",
      before: initial,
      after: titled(initial, "已编辑"),
    })).toThrow(/entry ID space is exhausted/);
  });

  it("recomputes byte counters and refuses non-allow-listed patch paths on export", () => {
    const initial = createTimelineProject();
    const edited = titled(initial, "已编辑");
    const history = recordTimelineHistory(createTimelineHistory(), {
      label: "编辑",
      before: initial,
      after: edited,
    });
    const staleCounters = {
      ...history,
      totalBytes: 1,
      past: [{ ...history.past[0], byteSize: 1 }],
    };
    const corrected = serializeTimelineHistory(staleCounters);
    expect(corrected.payload.totalBytes).not.toBe(1);
    expect(corrected.payload.past[0].byteSize).not.toBe(1);
    expect(deserializeTimelineHistory(corrected)).not.toBeNull();

    const unsafe = {
      ...history,
      past: [{
        ...history.past[0],
        forward: [{ op: "set" as const, path: ["__proto__", "polluted"], value: true }],
      }],
    };
    expect(() => serializeTimelineHistory(unsafe)).toThrow(TypeError);
    expect(({} as { polluted?: boolean }).polluted).toBeUndefined();
  });

  it("reset clears both directions, patches and checkpoints while retaining configuration", () => {
    const initial = createTimelineProject();
    const edited = titled(initial, "编辑后");
    let history = recordTimelineHistory(createTimelineHistory(75, 9_999), {
      label: "编辑",
      before: initial,
      after: edited,
    });
    history = undoTimelineHistory(history)!.history;
    const reset = resetTimelineHistory(history);

    expect(reset).toMatchObject({
      capacity: 75,
      byteBudget: 9_999,
      totalBytes: 0,
      past: [],
      future: [],
      checkpoints: [],
      head: null,
      coalescing: null,
    });
    expect(resetTimelineHistory(reset)).toBe(reset);
  });
});
