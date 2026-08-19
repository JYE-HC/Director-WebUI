export interface TimelineCoordinationScope {
  databasePath: string;
  projectId: string;
}

export interface TimelineRevisionNotice {
  revision: number;
  documentHash: string;
}

interface TimelineRevisionWireNotice {
  format: "director-timeline-revision";
  version: 1;
  sender_id: string;
  revision: number;
  document_hash: string;
}

export interface TimelineRevisionChannel {
  publish: (notice: TimelineRevisionNotice) => void;
  acceptKnown: (notice: TimelineRevisionNotice) => void;
  close: () => void;
}

function fnv1a(value: string, seed: number): string {
  let hash = seed >>> 0;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/** Opaque scope name: database paths and project ids never enter browser lock names. */
export function timelineCoordinationScopeKey(scope: TimelineCoordinationScope): string {
  const source = `${scope.databasePath}\u0000${scope.projectId}`;
  return `${fnv1a(source, 0x811c9dc5)}${fnv1a(source, 0x9e3779b9)}`;
}

interface LockManagerLike {
  request<T>(
    name: string,
    options: { mode: "exclusive"; signal?: AbortSignal },
    callback: () => T | PromiseLike<T>,
  ): Promise<T>;
}

/** Serializes writers when Web Locks exists; server CAS remains authoritative without it. */
export function runWithTimelineWriterLock<T>(
  scope: TimelineCoordinationScope,
  operation: () => T | PromiseLike<T>,
  signal?: AbortSignal,
): Promise<T> {
  const locks = (navigator as Navigator & { locks?: LockManagerLike }).locks;
  if (!locks) return Promise.resolve(operation());
  return locks.request(
    `director-timeline-writer:${timelineCoordinationScopeKey(scope)}`,
    { mode: "exclusive", ...(signal ? { signal } : {}) },
    operation,
  );
}

function validNotice(value: unknown): value is TimelineRevisionWireNotice {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const notice = value as Record<string, unknown>;
  return Object.keys(notice).sort().join("|") ===
      "document_hash|format|revision|sender_id|version" &&
    notice.format === "director-timeline-revision" &&
    notice.version === 1 &&
    typeof notice.sender_id === "string" &&
    /^[A-Za-z0-9._:-]{1,128}$/.test(notice.sender_id) &&
    Number.isSafeInteger(notice.revision) &&
    (notice.revision as number) >= 0 &&
    typeof notice.document_hash === "string" &&
    /^[A-Za-z0-9._:-]{1,128}$/.test(notice.document_hash);
}

function createSenderId(): string {
  return typeof globalThis.crypto?.randomUUID === "function"
    ? `tab-${globalThis.crypto.randomUUID()}`
    : `tab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
}

/**
 * BroadcastChannel is only an invalidation hint. Lower/duplicate revisions are
 * ignored; consumers must still perform an authority GET before adopting data.
 */
export function createTimelineRevisionChannel(
  scope: TimelineCoordinationScope,
  initial: TimelineRevisionNotice | null,
  onRemoteRevision: (notice: TimelineRevisionNotice) => void,
): TimelineRevisionChannel {
  let known = initial;
  let closed = false;
  const senderId = createSenderId();
  let channel: BroadcastChannel | null = null;
  try {
    if (typeof BroadcastChannel !== "undefined") {
      channel = new BroadcastChannel(
        `director-timeline-revision:${timelineCoordinationScopeKey(scope)}`,
      );
    }
  } catch {
    channel = null;
  }

  if (channel) channel.onmessage = (event: MessageEvent<unknown>) => {
    if (closed || !validNotice(event.data) || event.data.sender_id === senderId) return;
    const incoming = {
      revision: event.data.revision,
      documentHash: event.data.document_hash,
    };
    if (
      known && (
        incoming.revision < known.revision ||
        (incoming.revision === known.revision && incoming.documentHash === known.documentHash)
      )
    ) return;
    if (!known || incoming.revision > known.revision) known = incoming;
    onRemoteRevision(incoming);
  };

  return {
    publish(notice) {
      if (
        closed ||
        !Number.isSafeInteger(notice.revision) ||
        notice.revision < 0 ||
        (known !== null && notice.revision < known.revision)
      ) return;
      known = notice;
      const wire: TimelineRevisionWireNotice = {
        format: "director-timeline-revision",
        version: 1,
        sender_id: senderId,
        revision: notice.revision,
        document_hash: notice.documentHash,
      };
      channel?.postMessage(wire);
    },
    acceptKnown(notice) {
      if (!known || notice.revision >= known.revision) known = notice;
    },
    close() {
      if (closed) return;
      closed = true;
      channel?.close();
      channel = null;
    },
  };
}
