import "fake-indexeddb/auto";
import { createTimelineProject } from "../domain/timelineProject";
import { createTimelineHistory, recordTimelineHistory } from "../state/timelineHistory";
import {
  deleteTimelineHistoryJournal,
  deleteTimelineHistoryDatabaseForTests,
  legacyTimelineHistoryJournalKey,
  listTimelineHistoryJournalBranches,
  loadTimelineHistoryJournal,
  readTimelineHistoryJournalVersionToken,
  saveTimelineHistoryJournal,
  timelineHistoryJournalKey,
  timelineProjectDigest,
  type TimelinePersistenceScope,
} from "../state/timelinePersistence";

const scope: TimelinePersistenceScope = {
  databasePath: "/srv/director/data/director.sqlite3",
  projectId: "project-history",
  ownerId: "session-owner-a",
};

const DATABASE_NAME = "directordeck-timeline-history";
const STORE_NAME = "journals";

async function openTestDatabase(): Promise<IDBDatabase> {
  return await new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readRawJournal(key: string): Promise<Record<string, unknown>> {
  const database = await openTestDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const request = database.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(key);
      request.onsuccess = () => resolve(request.result as Record<string, unknown>);
      request.onerror = () => reject(request.error);
    });
  } finally {
    database.close();
  }
}

async function putRawJournal(value: Record<string, unknown>): Promise<void> {
  const database = await openTestDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put(value);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  } finally {
    database.close();
  }
}

async function beginRawJournalPut(
  value: Record<string, unknown>,
): Promise<{ committed: Promise<void> }> {
  const database = await openTestDatabase();
  const transaction = database.transaction(STORE_NAME, "readwrite");
  transaction.objectStore(STORE_NAME).put(value);
  const committed = new Promise<void>((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  }).finally(() => database.close());
  return { committed };
}

describe("timeline IndexedDB persistence", () => {
  beforeEach(async () => deleteTimelineHistoryDatabaseForTests());
  afterEach(async () => deleteTimelineHistoryDatabaseForTests());

  function fixture() {
    const base = createTimelineProject();
    const edited = { ...base, title: "跨刷新编辑" };
    const history = recordTimelineHistory(createTimelineHistory(), {
      label: "重命名项目",
      before: base,
      after: edited,
    });
    return { base, edited, history };
  }

  it("只在数据库、项目、revision 与权威基线全部精确匹配时恢复", async () => {
    const { base, edited, history } = fixture();
    const token = await saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 7 },
      history,
    );
    expect(token).toMatchObject({ key: timelineHistoryJournalKey(scope) });

    const restored = await loadTimelineHistoryJournal(scope, {
      document: base,
      revision: 7,
    });
    expect(restored).toMatchObject({
      status: "restored",
      project: edited,
      confirmedRevision: 7,
    });
    if (restored.status !== "restored") throw new Error("expected restored history");
    expect(restored.history.past).toHaveLength(1);

    await expect(loadTimelineHistoryJournal(
      { ...scope, projectId: "another-project" },
      { document: base, revision: 7 },
    )).resolves.toEqual({ status: "none" });
  });

  it("把服务器已经等于 pending head 识别为丢失 ACK", async () => {
    const { base, edited, history } = fixture();
    await saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 3 },
      history,
    );

    await expect(loadTimelineHistoryJournal(scope, {
      document: edited,
      revision: 4,
    })).resolves.toMatchObject({
      status: "acknowledged",
      project: edited,
      confirmedRevision: 3,
      confirmedDocument: base,
    });
  });

  it("confirmed document 已等于 history head 时归类为 clean acknowledged 而非 replay", async () => {
    const { edited, history } = fixture();
    await saveTimelineHistoryJournal(
      scope,
      { document: edited, revision: 4 },
      history,
    );

    await expect(loadTimelineHistoryJournal(scope, {
      document: edited,
      revision: 4,
    })).resolves.toMatchObject({
      status: "acknowledged",
      project: edited,
      confirmedRevision: 4,
      confirmedDocument: edited,
    });
    const listed = await listTimelineHistoryJournalBranches({
      ...scope,
      ownerId: "fresh-owner",
    }, {
      document: edited,
      revision: 4,
    });
    expect(listed.status).toBe("available");
    if (listed.status !== "available") throw new Error("expected available branch list");
    expect(listed.foreign).toHaveLength(1);
    expect(listed.foreign[0]).toMatchObject({
      ownerId: scope.ownerId,
      status: "acknowledged",
      project: edited,
    });
  });

  it("WebCrypto 保存时失败、读取时恢复也不会改变 journal digest 算法", async () => {
    const { base, edited, history } = fixture();
    const digest = vi.spyOn(globalThis.crypto.subtle, "digest")
      .mockRejectedValueOnce(new DOMException("temporary crypto failure", "OperationError"));
    await saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 12 },
      history,
    );
    expect(digest).not.toHaveBeenCalled();
    digest.mockRestore();

    await expect(loadTimelineHistoryJournal(scope, {
      document: base,
      revision: 12,
    })).resolves.toMatchObject({
      status: "restored",
      project: edited,
      confirmedRevision: 12,
    });
  });

  it("authority 分叉时保持本地取证记录但绝不自动返回可写历史", async () => {
    const { base, edited, history } = fixture();
    await saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 10 },
      history,
    );
    const remote = { ...base, title: "另一标签页提交" };

    await expect(loadTimelineHistoryJournal(scope, {
      document: remote,
      revision: 11,
    })).resolves.toMatchObject({
      status: "conflict",
      confirmedRevision: 10,
      confirmedDocument: base,
      localProject: edited,
    });
    // Conflict reads are inert and do not delete evidence.
    await expect(loadTimelineHistoryJournal(scope, {
      document: base,
      revision: 10,
    })).resolves.toMatchObject({ status: "restored", project: edited });
  });

  it("用 opaque scope digest + owner 隔离物理 key", () => {
    const key = timelineHistoryJournalKey(scope);
    expect(key).toBe(
      "directordeck:v2:timeline-history:" +
        "0124c138950f13c515ddf3414748689064af9b775ed120eb06a32bcedcb04b6d:" +
        "session-owner-a",
    );
    expect(key).not.toContain(scope.databasePath);
    expect(key).not.toContain(scope.projectId);
    expect(timelineHistoryJournalKey({ ...scope, ownerId: "session-owner-b" })).not.toBe(key);
    expect(timelineHistoryJournalKey({
      ...scope,
      databasePath: "/srv/director/data/another.sqlite3",
    })).not.toBe(key);
  });

  it("枚举 owned/foreign 全部分支且 lost ACK 只匹配具体 branch head", async () => {
    const base = createTimelineProject();
    const editedA = { ...base, title: "标签页 A" };
    const editedB = { ...base, title: "标签页 B" };
    const middle = { ...base, title: "标签页 C 中间态" };
    const historyA = recordTimelineHistory(createTimelineHistory(), {
      label: "A 编辑",
      before: base,
      after: editedA,
    });
    const historyB = recordTimelineHistory(createTimelineHistory(), {
      label: "B 编辑",
      before: base,
      after: editedB,
    });
    const historyC = recordTimelineHistory(recordTimelineHistory(createTimelineHistory(), {
      label: "C 第一步",
      before: base,
      after: middle,
    }), {
      label: "C 第二步",
      before: middle,
      after: editedB,
    });
    const scopeB = { ...scope, ownerId: "session-owner-b" };
    const scopeC = { ...scope, ownerId: "session-owner-c" };
    await saveTimelineHistoryJournal(scope, { document: base, revision: 7 }, historyA);
    await saveTimelineHistoryJournal(scopeB, { document: base, revision: 7 }, historyB);
    await saveTimelineHistoryJournal(scopeC, { document: base, revision: 7 }, historyC);

    // Make ordering deterministic without changing any authority/hash field.
    const rawA = await readRawJournal(timelineHistoryJournalKey(scope));
    const rawB = await readRawJournal(timelineHistoryJournalKey(scopeB));
    const rawC = await readRawJournal(timelineHistoryJournalKey(scopeC));
    await putRawJournal({ ...rawA, updatedAtMs: 1_000 });
    await putRawJournal({ ...rawB, updatedAtMs: 2_000 });
    await putRawJournal({ ...rawC, updatedAtMs: 3_000 });

    const listed = await listTimelineHistoryJournalBranches(scope, {
      document: editedB,
      revision: 8,
    });
    expect(listed.status).toBe("available");
    if (listed.status !== "available") throw new Error("expected available branch list");
    expect(listed.branches.map((branch) => branch.ownerId)).toEqual([
      "session-owner-c",
      "session-owner-b",
      "session-owner-a",
    ]);
    expect(listed.owned).toHaveLength(1);
    expect(listed.owned[0]).toMatchObject({
      ownerId: "session-owner-a",
      status: "conflict",
      confirmedRevision: 7,
      confirmedDocument: base,
      project: editedA,
    });
    expect(listed.foreign).toHaveLength(2);
    expect(listed.foreign.map((branch) => branch.status)).toEqual([
      "acknowledged",
      "acknowledged",
    ]);
    expect(listed.foreign[0]).toMatchObject({ ownerId: "session-owner-c", project: editedB });
    if (listed.foreign[0].status === "corrupt") throw new Error("expected decoded branch");
    expect(listed.foreign[0].history.past).toHaveLength(2);

    await expect(loadTimelineHistoryJournal(scope, {
      document: editedB,
      revision: 8,
    })).resolves.toMatchObject({ status: "conflict", localProject: editedA });
    await expect(loadTimelineHistoryJournal(scopeB, {
      document: editedB,
      revision: 8,
    })).resolves.toMatchObject({ status: "acknowledged", project: editedB });

    // Enumeration/classification is inert: every branch remains available.
    const listedAgain = await listTimelineHistoryJournalBranches(scope, {
      document: editedB,
      revision: 8,
    });
    expect(listedAgain).toMatchObject({
      status: "available",
      branches: [{ ownerId: "session-owner-c" }, { ownerId: "session-owner-b" }, {
        ownerId: "session-owner-a",
      }],
    });
  });

  it("compare-and-delete 必须匹配整条 record，旧 token/无 token/复用 writeToken 均不能误删", async () => {
    const { base, history } = fixture();
    const firstToken = await saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 5 },
      history,
    );
    expect(firstToken).not.toBeNull();

    const secondProject = { ...base, title: "同 owner 的更新版本" };
    const secondHistory = recordTimelineHistory(createTimelineHistory(), {
      label: "第二次编辑",
      before: base,
      after: secondProject,
    });
    const secondToken = await saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 5 },
      secondHistory,
    );
    expect(secondToken).not.toBeNull();
    await expect(deleteTimelineHistoryJournal(scope)).resolves.toBe(false);
    await expect(deleteTimelineHistoryJournal(scope, firstToken)).resolves.toBe(false);
    await expect(deleteTimelineHistoryJournal(
      { ...scope, projectId: "different-project" },
      secondToken,
    )).resolves.toBe(false);

    // An empty history is not an implicit delete capability.
    await expect(saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 5 },
      createTimelineHistory(),
    )).resolves.toBeNull();

    // Simulate another writer replacing bytes while incorrectly retaining its
    // writeToken. Full-record comparison must still reject the earlier token.
    const raw = await readRawJournal(timelineHistoryJournalKey(scope));
    await putRawJournal({ ...raw, updatedAtMs: (raw.updatedAtMs as number) + 1 });
    await expect(deleteTimelineHistoryJournal(scope, secondToken)).resolves.toBe(false);

    const current = await loadTimelineHistoryJournal(scope, {
      document: base,
      revision: 5,
    });
    expect(current).toMatchObject({ status: "restored", project: secondProject });
    if (current.status !== "restored") throw new Error("expected current journal");
    await expect(deleteTimelineHistoryJournal(scope, current.token)).resolves.toBe(true);
    // A partial multi-store discard may already have removed this exact branch;
    // retrying the same capability must converge instead of getting stuck.
    await expect(deleteTimelineHistoryJournal(scope, current.token)).resolves.toBe(true);
    await expect(loadTimelineHistoryJournal(scope, {
      document: base,
      revision: 5,
    })).resolves.toEqual({ status: "none" });
  });

  it("deferred put 完成后可重读 exact-owner token 并安全清理", async () => {
    const { base, history } = fixture();
    const initialToken = await saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 6 },
      history,
    );
    expect(initialToken).not.toBeNull();
    const raw = await readRawJournal(timelineHistoryJournalKey(scope));
    await expect(deleteTimelineHistoryJournal(scope, initialToken)).resolves.toBe(true);

    const deferred = await beginRawJournalPut(raw);
    const tokenRead = readTimelineHistoryJournalVersionToken(scope);
    await deferred.committed;
    const token = await tokenRead;
    expect(token).toMatchObject({
      key: timelineHistoryJournalKey(scope),
      version: expect.stringMatching(/^sha256:[0-9a-f]{64}$/),
    });
    await expect(deleteTimelineHistoryJournal(scope, token)).resolves.toBe(true);
    await expect(readTimelineHistoryJournalVersionToken(scope)).resolves.toBeNull();
  });

  it("把旧 v1 ownerless journal 保留为 legacy evidence，且只允许 token 精确删除", async () => {
    const { base, edited, history } = fixture();
    await saveTimelineHistoryJournal(scope, { document: base, revision: 9 }, history);
    const raw = structuredClone(await readRawJournal(timelineHistoryJournalKey(scope)));
    delete raw.writeToken;
    raw.version = 1;
    raw.key = legacyTimelineHistoryJournalKey(scope);
    raw.scope = {
      databasePath: scope.databasePath,
      projectId: scope.projectId,
    };
    await putRawJournal(raw);

    const freshOwner = { ...scope, ownerId: "fresh-page-owner" };
    await expect(loadTimelineHistoryJournal(freshOwner, {
      document: base,
      revision: 9,
    })).resolves.toEqual({ status: "none" });
    const listed = await listTimelineHistoryJournalBranches(freshOwner, {
      document: base,
      revision: 9,
    });
    expect(listed.status).toBe("available");
    if (listed.status !== "available") throw new Error("expected available branch list");
    expect(listed.legacy).toHaveLength(1);
    expect(listed.legacy[0]).toMatchObject({
      ownership: "legacy",
      ownerId: null,
      status: "restored",
      project: edited,
    });
    expect(listed.legacy[0].token?.version).toMatch(/^sha256:[0-9a-f]{64}$/);

    const stillPresent = await listTimelineHistoryJournalBranches(freshOwner, {
      document: base,
      revision: 9,
    });
    expect(stillPresent.status === "available" ? stillPresent.legacy : []).toHaveLength(1);
    const legacyToken = listed.legacy[0].token;
    expect(legacyToken).not.toBeNull();
    await expect(deleteTimelineHistoryJournal(freshOwner, legacyToken)).resolves.toBe(true);
    const afterDelete = await listTimelineHistoryJournalBranches(freshOwner, {
      document: base,
      revision: 9,
    });
    expect(afterDelete.status === "available" ? afterDelete.legacy : []).toHaveLength(0);
    expect(afterDelete.status === "available" ? afterDelete.foreign : []).toHaveLength(1);
  });

  it("严格拒绝损坏 codec/hash 但仍枚举 evidence 与精确清理 token", async () => {
    const { base, history } = fixture();
    const originalToken = await saveTimelineHistoryJournal(
      scope,
      { document: base, revision: 12 },
      history,
    );
    const raw = await readRawJournal(timelineHistoryJournalKey(scope));
    await putRawJournal({ ...raw, headDocumentHash: "tampered" });

    const loaded = await loadTimelineHistoryJournal(scope, {
      document: base,
      revision: 12,
    });
    expect(loaded).toMatchObject({ status: "corrupt" });
    if (loaded.status !== "corrupt") throw new Error("expected corrupt journal");
    expect(loaded.token).not.toBeNull();
    const listed = await listTimelineHistoryJournalBranches(scope, {
      document: base,
      revision: 12,
    });
    expect(listed.status === "available" ? listed.corrupt : []).toHaveLength(1);
    await expect(deleteTimelineHistoryJournal(scope, originalToken)).resolves.toBe(false);
    await expect(deleteTimelineHistoryJournal(scope, loaded.token)).resolves.toBe(true);
  });

  it("authority document 本身不满足严格 schema 时失败封闭", async () => {
    const { base, history } = fixture();
    await saveTimelineHistoryJournal(scope, { document: base, revision: 2 }, history);
    const malformed = { ...base, unrecognized: true } as typeof base;
    await expect(loadTimelineHistoryJournal(scope, {
      document: malformed,
      revision: 2,
    })).resolves.toMatchObject({ status: "corrupt" });
    await expect(listTimelineHistoryJournalBranches(scope, {
      document: malformed,
      revision: 2,
    })).resolves.toEqual({ status: "corrupt" });
  });

  it("produces a stable content digest while exact equality remains the restore gate", async () => {
    const project = createTimelineProject();
    expect(await timelineProjectDigest(project)).toBe(await timelineProjectDigest(
      structuredClone(project),
    ));
    expect(await timelineProjectDigest({ ...project, title: "different" })).not.toBe(
      await timelineProjectDigest(project),
    );
  });
});
