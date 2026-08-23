import { afterEach, describe, expect, it, vi } from "vitest";

import menuSource from "../../../plugin/web/directordeck-menu.js?raw";

type SidebarTabDefinition = {
  render: (element: HTMLElement) => void;
};

type DirectorExtension = {
  setup: () => void;
};

const loadSidebarTab = (): SidebarTabDefinition => {
  let sidebarTab: SidebarTabDefinition | undefined;
  const app = {
    extensionManager: {
      registerSidebarTab(definition: SidebarTabDefinition) {
        sidebarTab = definition;
      },
    },
    registerExtension(extension: DirectorExtension) {
      extension.setup();
    },
  };
  const executableSource = menuSource.replace(
    'import { app } from "../../../scripts/app.js";',
    "",
  );

  new Function("app", executableSource)(app);
  if (!sidebarTab) {
    throw new Error("DirectorDeck sidebar tab was not registered");
  }
  return sidebarTab;
};

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("DirectorDeck ComfyUI sidebar status", () => {
  it("queries the backend even when ComfyUI has not attached the render target", async () => {
    const fetchStatus = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ backend: "ready", version: "0.2.8" }),
    });
    vi.stubGlobal("fetch", fetchStatus);
    const element = document.createElement("div");

    loadSidebarTab().render(element);

    expect(element.isConnected).toBe(false);
    expect(fetchStatus).toHaveBeenCalledWith(
      "/directordeck/status",
      expect.objectContaining({ cache: "no-store" }),
    );
    await vi.waitFor(() => {
      expect(element.querySelector(".director-status")).toHaveTextContent(
        "后端状态：运行中（v0.2.8）",
      );
    });
  });

  it("turns a pending status request into a retryable timeout", async () => {
    vi.useFakeTimers();
    const fetchStatus = vi.fn(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("Status request timed out", "AbortError"));
          });
        }),
    );
    vi.stubGlobal("fetch", fetchStatus);
    const element = document.createElement("div");

    loadSidebarTab().render(element);
    await vi.advanceTimersByTimeAsync(5000);

    expect(element.querySelector(".director-status")).toHaveTextContent(
      "后端状态：查询超时",
    );
    expect(element.querySelector(".director-refresh")).not.toHaveAttribute(
      "hidden",
    );
  });
});
