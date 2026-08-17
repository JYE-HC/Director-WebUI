import {
  createTimelineRevisionChannel,
  runWithTimelineWriterLock,
  timelineCoordinationScopeKey,
} from "../state/timelineCoordination";

const scope = {
  databaseIdentity: "a".repeat(64),
  projectId: "project-private-name",
};

describe("timeline coordination", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("creates a stable opaque scope key", () => {
    const key = timelineCoordinationScopeKey(scope);
    expect(key).toMatch(/^[0-9a-f]{16}$/);
    expect(key).not.toContain(scope.projectId);
    expect(timelineCoordinationScopeKey(scope)).toBe(key);
    expect(timelineCoordinationScopeKey({ ...scope, projectId: "other" })).not.toBe(key);
  });

  it("uses an exclusive Web Lock when available and safely falls back without it", async () => {
    const request = vi.fn(async (
      _name: string,
      _options: unknown,
      callback: () => Promise<string> | string,
    ) => callback());
    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: { request },
    });

    await expect(runWithTimelineWriterLock(scope, () => "locked")).resolves.toBe("locked");
    expect(request).toHaveBeenCalledWith(
      expect.stringMatching(/^director-timeline-writer:[0-9a-f]{16}$/),
      { mode: "exclusive" },
      expect.any(Function),
    );

    Object.defineProperty(navigator, "locks", {
      configurable: true,
      value: undefined,
    });
    await expect(runWithTimelineWriterLock(scope, () => "fallback")).resolves.toBe("fallback");
  });

  it("broadcasts only newer or same-revision-different-hash authority hints", () => {
    class FakeBroadcastChannel {
      static channels = new Map<string, Set<FakeBroadcastChannel>>();
      onmessage: ((event: MessageEvent<unknown>) => void) | null = null;

      constructor(readonly name: string) {
        const peers = FakeBroadcastChannel.channels.get(name) ?? new Set();
        peers.add(this);
        FakeBroadcastChannel.channels.set(name, peers);
      }

      postMessage(value: unknown) {
        for (const peer of FakeBroadcastChannel.channels.get(this.name) ?? []) {
          if (peer !== this) peer.onmessage?.(new MessageEvent("message", { data: value }));
        }
      }

      close() {
        FakeBroadcastChannel.channels.get(this.name)?.delete(this);
      }
    }
    vi.stubGlobal("BroadcastChannel", FakeBroadcastChannel);
    const firstReceived = vi.fn();
    const secondReceived = vi.fn();
    const first = createTimelineRevisionChannel(
      scope,
      { revision: 4, documentHash: "hash-four" },
      firstReceived,
    );
    const second = createTimelineRevisionChannel(
      scope,
      { revision: 3, documentHash: "hash-three" },
      secondReceived,
    );

    first.publish({ revision: 5, documentHash: "hash-five" });
    expect(secondReceived).toHaveBeenLastCalledWith({
      revision: 5,
      documentHash: "hash-five",
    });
    second.publish({ revision: 2, documentHash: "old" });
    expect(firstReceived).not.toHaveBeenCalled();
    second.publish({ revision: 5, documentHash: "conflicting-five" });
    expect(firstReceived).toHaveBeenLastCalledWith({
      revision: 5,
      documentHash: "conflicting-five",
    });

    first.close();
    second.close();
  });
});
