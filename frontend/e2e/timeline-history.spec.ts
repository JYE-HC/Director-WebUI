import { expect, test, type Locator, type Page } from "@playwright/test";
import {
  createDirectorApiMockAuthority,
  INITIAL_PROMPT,
  installDirectorApiMock,
} from "./mockDirectorApi";
import { TIMELINE_WAL_STORAGE_PREFIX } from "../src/domain/timelineProject";

interface BrowserTimelineWalEvidence {
  key: string;
  raw: string;
  ownerId: string;
  pendingPrompt: string;
}

async function findTimelineWalByPrompt(
  page: Page,
  expectedPrompt: string,
): Promise<BrowserTimelineWalEvidence | null> {
  return page.evaluate(({ prefix, prompt }) => {
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);
      if (!key?.startsWith(prefix)) continue;
      const raw = localStorage.getItem(key);
      if (raw === null) continue;
      try {
        const value = JSON.parse(raw) as {
          owner_id?: unknown;
          pending_project?: { segments?: Array<{ prompt?: unknown }> };
        };
        const pendingPrompt = value.pending_project?.segments?.[0]?.prompt;
        if (typeof value.owner_id === "string" && pendingPrompt === prompt) {
          return { key, raw, ownerId: value.owner_id, pendingPrompt };
        }
      } catch {
        // A corrupt branch is not the valid offline edit this helper seeks.
      }
    }
    return null;
  }, { prefix: TIMELINE_WAL_STORAGE_PREFIX, prompt: expectedPrompt });
}

async function openPrompt(page: Page): Promise<Locator> {
  await installDirectorApiMock(page);
  await page.goto("/");
  const prompt = page.getByRole("textbox", { name: "片段提示词", exact: true });
  await expect(prompt).toBeVisible();
  await expect(prompt).toHaveValue(INITIAL_PROMPT);
  await expect(page.getByText("正在从服务器恢复时间线")).toHaveCount(0);
  return prompt;
}

async function setNativeTextareaValue(
  prompt: Locator,
  value: string,
  data: string,
  isComposing: boolean,
): Promise<void> {
  await prompt.evaluate((element, input) => {
    const setter = Object.getOwnPropertyDescriptor(
      HTMLTextAreaElement.prototype,
      "value",
    )?.set;
    if (!setter) throw new Error("textarea value setter is unavailable");
    setter.call(element, input.value);
    element.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      cancelable: false,
      data: input.data,
      inputType: input.isComposing ? "insertCompositionText" : "insertText",
      isComposing: input.isComposing,
    }));
  }, { value, data, isComposing });
}

test("focused project prompt owns Ctrl+Z and Ctrl+Shift+Z", async ({ page }) => {
  const prompt = await openPrompt(page);

  await prompt.focus();
  await prompt.press("End");
  await page.keyboard.insertText(" slowly");
  await expect(prompt).toHaveValue(`${INITIAL_PROMPT} slowly`);

  await prompt.press("Control+z");
  await expect(prompt).toHaveValue(INITIAL_PROMPT);

  await prompt.press("Control+Shift+z");
  await expect(prompt).toHaveValue(`${INITIAL_PROMPT} slowly`);
});

test("caret navigation seals one typing transaction", async ({ page }) => {
  const prompt = await openPrompt(page);

  await prompt.focus();
  await prompt.press("End");
  await page.keyboard.insertText("AB");
  await prompt.press("ArrowLeft");
  await page.keyboard.insertText("X");
  await expect(prompt).toHaveValue(`${INITIAL_PROMPT}AXB`);

  await prompt.press("Control+z");
  await expect(prompt).toHaveValue(`${INITIAL_PROMPT}AB`);
  await prompt.press("Control+z");
  await expect(prompt).toHaveValue(INITIAL_PROMPT);
});

test("a selected range replacement round-trips through project history", async ({ page }) => {
  const prompt = await openPrompt(page);

  await prompt.focus();
  await prompt.evaluate((element) => {
    const textarea = element as HTMLTextAreaElement;
    textarea.setSelectionRange(0, 6);
    textarea.dispatchEvent(new Event("select", { bubbles: true }));
  });
  await page.keyboard.insertText("SHOT");
  await expect(prompt).toHaveValue("SHOT move");

  await prompt.press("Control+z");
  await expect(prompt).toHaveValue(INITIAL_PROMPT);
  await expect.poll(() => prompt.evaluate((element) => ({
    start: (element as HTMLTextAreaElement).selectionStart,
    end: (element as HTMLTextAreaElement).selectionEnd,
    direction: (element as HTMLTextAreaElement).selectionDirection,
  }))).toEqual({ start: 0, end: 6, direction: "forward" });
  await prompt.press("Control+Shift+z");
  await expect(prompt).toHaveValue("SHOT move");
  await expect.poll(() => prompt.evaluate((element) => ({
    start: (element as HTMLTextAreaElement).selectionStart,
    end: (element as HTMLTextAreaElement).selectionEnd,
  }))).toEqual({ start: 4, end: 4 });
});

test("beforeinput historyUndo and historyRedo use the application stack", async ({ page }) => {
  const prompt = await openPrompt(page);

  await prompt.focus();
  await prompt.press("End");
  await page.keyboard.insertText(" close-up");
  await expect(prompt).toHaveValue(`${INITIAL_PROMPT} close-up`);

  const undoWasNotCancelled = await prompt.evaluate((element) => element.dispatchEvent(
    new InputEvent("beforeinput", {
      bubbles: true,
      cancelable: true,
      inputType: "historyUndo",
    }),
  ));
  expect(undoWasNotCancelled).toBe(false);
  await expect(prompt).toHaveValue(INITIAL_PROMPT);

  const redoWasNotCancelled = await prompt.evaluate((element) => element.dispatchEvent(
    new InputEvent("beforeinput", {
      bubbles: true,
      cancelable: true,
      inputType: "historyRedo",
    }),
  ));
  expect(redoWasNotCancelled).toBe(false);
  await expect(prompt).toHaveValue(`${INITIAL_PROMPT} close-up`);
});

test("an IME session stays atomic across ArrowDown and the coalescing timeout", async ({ page }) => {
  const prompt = await openPrompt(page);
  const undo = page.getByRole("button", { name: "撤销", exact: true });

  await prompt.focus();
  await prompt.evaluate((element) => {
    const textarea = element as HTMLTextAreaElement;
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);
    textarea.dispatchEvent(new CompositionEvent("compositionstart", {
      bubbles: true,
      data: "",
    }));
  });
  await setNativeTextareaValue(prompt, `${INITIAL_PROMPT}你`, "你", true);
  // The normal text merge window is 800 ms. A real composition session owns
  // its own boundary and must not split merely because the IME remains open.
  await page.waitForTimeout(900);
  await prompt.press("ArrowDown");
  await setNativeTextareaValue(prompt, `${INITIAL_PROMPT}你好`, "好", true);
  await prompt.evaluate((element) => element.dispatchEvent(
    new CompositionEvent("compositionend", {
      bubbles: true,
      data: "你好",
    }),
  ));
  await expect(prompt).toHaveValue(`${INITIAL_PROMPT}你好`);

  await undo.click();
  await expect(prompt).toHaveValue(INITIAL_PROMPT);
  await expect(undo).toBeDisabled();
});

test("history buttons expose stable names and replay both directions", async ({ page }) => {
  const prompt = await openPrompt(page);
  const undo = page.getByRole("button", { name: "撤销", exact: true });
  const redo = page.getByRole("button", { name: "重做", exact: true });

  await expect(undo).toBeDisabled();
  await expect(redo).toBeDisabled();

  await prompt.focus();
  await prompt.press("End");
  await page.keyboard.insertText("!");
  await expect(undo).toBeEnabled();
  await expect(undo).toHaveAttribute("title", /撤销：编辑提示词/);

  await undo.click();
  await expect(prompt).toHaveValue(INITIAL_PROMPT);
  await expect(prompt).toBeFocused();
  await expect(redo).toBeEnabled();
  await expect(redo).toHaveAttribute("title", /重做：编辑提示词/);

  await redo.click();
  await expect(prompt).toHaveValue(`${INITIAL_PROMPT}!`);
  await expect(prompt).toBeFocused();
});

test("two tabs sharing one project have one CAS winner and an explicit loser conflict", async ({
  context,
  page,
}) => {
  const sharedAuthority = createDirectorApiMockAuthority();
  const peer = await context.newPage();
  await Promise.all([
    installDirectorApiMock(page, sharedAuthority),
    installDirectorApiMock(peer, sharedAuthority),
  ]);
  await Promise.all([page.goto("/"), peer.goto("/")]);

  const firstPrompt = page.getByRole("textbox", { name: "片段提示词", exact: true });
  const secondPrompt = peer.getByRole("textbox", { name: "片段提示词", exact: true });
  await Promise.all([
    expect(firstPrompt).toHaveValue(INITIAL_PROMPT),
    expect(secondPrompt).toHaveValue(INITIAL_PROMPT),
  ]);

  await Promise.all([
    firstPrompt.fill("first tab branch"),
    secondPrompt.fill("second tab branch"),
  ]);
  await expect.poll(() => sharedAuthority.revision).toBe(1);
  await expect.poll(async () => (
    await page.getByRole("button", { name: "采用服务器版本" }).count()
  ) + (
    await peer.getByRole("button", { name: "采用服务器版本" }).count()
  )).toBe(1);

  const firstLost = await page.getByRole("button", { name: "采用服务器版本" }).count() === 1;
  const loser = firstLost ? page : peer;
  const winner = firstLost ? peer : page;
  const serverPrompt = sharedAuthority.project.segments[0].prompt;
  await expect(winner.getByRole("textbox", { name: "片段提示词", exact: true }))
    .toHaveValue(serverPrompt);
  await loser.getByRole("button", { name: "采用服务器版本" }).click();
  await expect(loser.getByRole("textbox", { name: "片段提示词", exact: true }))
    .toHaveValue(serverPrompt);
  await expect(loser.getByRole("button", { name: "采用服务器版本" })).toHaveCount(0);
  await peer.close();
});

test("a foreign offline branch waits for explicit recovery and preserves its evidence", async ({
  context,
  page,
}) => {
  const authority = createDirectorApiMockAuthority();
  const offlinePrompt = "offline page A recovery branch";
  authority.rejectTimelineAuthorityPuts = true;
  await installDirectorApiMock(page, authority);
  await page.goto("/");

  const firstPrompt = page.getByRole("textbox", { name: "片段提示词", exact: true });
  await expect(firstPrompt).toHaveValue(INITIAL_PROMPT);
  await firstPrompt.fill(offlinePrompt);
  await expect(firstPrompt).toHaveValue(offlinePrompt);
  await expect.poll(() => authority.timelineAuthorityPutAttempts).toBeGreaterThan(0);
  await expect.poll(async () => Boolean(await findTimelineWalByPrompt(page, offlinePrompt)))
    .toBe(true);
  const pageABranch = await findTimelineWalByPrompt(page, offlinePrompt);
  expect(pageABranch).not.toBeNull();
  expect(authority.revision).toBe(0);
  expect(authority.project.segments[0].prompt).toBe(INITIAL_PROMPT);

  await page.close();
  authority.rejectTimelineAuthorityPuts = false;
  const attemptsBeforeRecoveryPage = authority.timelineAuthorityPutAttempts;
  const recoveryPage = await context.newPage();
  try {
    await installDirectorApiMock(recoveryPage, authority);
    await recoveryPage.goto("/");

    const recoveryAlert = recoveryPage.getByRole("alert").filter({
      hasText: "检测到其他页面或旧会话留下的时间线恢复分支",
    });
    await expect(recoveryAlert).toBeVisible();
    const serverPrompt = recoveryPage.locator('textarea[aria-label="片段提示词"]');
    await expect(serverPrompt).toHaveValue(INITIAL_PROMPT);
    await expect(recoveryPage.getByRole("button", {
      name: "恢复所选分支",
      exact: true,
    })).toBeDisabled();

    // Hydration and merely displaying a foreign branch must not enqueue an
    // authority write. Wait beyond the normal autosave debounce to prove it.
    await recoveryPage.waitForTimeout(900);
    expect(authority.timelineAuthorityPutAttempts).toBe(attemptsBeforeRecoveryPage);
    expect(authority.revision).toBe(0);

    const foreignBeforeRestore = await recoveryPage.evaluate(({ prefix, ownerId }) => {
      for (let index = 0; index < localStorage.length; index += 1) {
        const key = localStorage.key(index);
        if (!key?.startsWith(prefix)) continue;
        const raw = localStorage.getItem(key);
        if (raw === null) continue;
        try {
          const value = JSON.parse(raw) as { owner_id?: unknown };
          if (value.owner_id === ownerId) return { key, raw };
        } catch {
          // Ignore unrelated corrupt evidence.
        }
      }
      return null;
    }, {
      prefix: TIMELINE_WAL_STORAGE_PREFIX,
      ownerId: pageABranch!.ownerId,
    });
    expect(foreignBeforeRestore).not.toBeNull();

    const branchChoice = recoveryPage.getByRole("radio", {
      name: /E2E Undo 项目；其他会话；.*基线匹配，可安全恢复/,
    });
    const restore = recoveryPage.getByRole("button", {
      name: "恢复所选分支",
      exact: true,
    });
    await branchChoice.check();
    await expect(restore).toBeEnabled();
    await recoveryPage.waitForTimeout(200);
    expect(authority.timelineAuthorityPutAttempts).toBe(attemptsBeforeRecoveryPage);

    await restore.click();
    await expect.poll(() => authority.revision).toBe(1);
    await expect(serverPrompt).toHaveValue(offlinePrompt);
    expect(authority.project.segments[0].prompt).toBe(offlinePrompt);
    expect(authority.timelineAuthorityPutAttempts).toBe(attemptsBeforeRecoveryPage + 1);

    // Recovery clones into page B's owner. The exact page-A bytes remain as
    // durable foreign evidence even after page B receives its successful ACK.
    await expect.poll(async () => recoveryPage.evaluate(({ key, raw }) => (
      localStorage.getItem(key) === raw
    ), foreignBeforeRestore!)).toBe(true);
  } finally {
    await recoveryPage.close();
  }
});
