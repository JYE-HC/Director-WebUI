// DirectorDeck sidebar entry: open the Director workspace in a new tab,
// show embedded-backend status, and link to documentation. The workspace
// itself is a full-page SPA served at /directordeck/.
import { app } from "../../../scripts/app.js";

const STARTING_POLL_MS = 1000;
const STARTING_MAX_ATTEMPTS = 30;
const RUNTIME_POLL_MS = 10000;

app.registerExtension({
  name: "DirectorDeck",
  setup() {
    app.extensionManager.registerSidebarTab({
      id: "directordeck",
      icon: "pi pi-video",
      title: "Director",
      tooltip: "Director 长视频导演台",
      type: "custom",
      render(el) {
        el.className = "director-tab";
        el.innerHTML = `
          <style>
            .director-tab { padding: 12px; display: flex; flex-direction: column; gap: 10px; font-size: 13px; }
            .director-tab button, .director-tab a.director-link {
              padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border-color, #444);
              background: var(--comfy-input-bg, #222); color: inherit; cursor: pointer;
              text-align: center; text-decoration: none; font-size: 13px;
            }
            .director-tab button:hover, .director-tab a.director-link:hover { filter: brightness(1.2); }
            .director-tab .director-status { white-space: pre-wrap; word-break: break-all; opacity: 0.85; }
            .director-tab .director-status.director-error { color: #e5534b; opacity: 1; }
          </style>
          <button class="director-open">打开 Director</button>
          <div class="director-status" role="status" aria-live="polite">后端状态：查询中…</div>
          <button class="director-refresh" type="button" hidden>重新查询状态</button>
          <a class="director-link" href="https://github.com/JYE-HC/Director-WebUI" target="_blank" rel="noreferrer">文档</a>
        `;
        el.querySelector(".director-open").addEventListener("click", () => {
          window.open("/directordeck/", "_blank", "noopener");
        });
        const statusEl = el.querySelector(".director-status");
        const refreshButton = el.querySelector(".director-refresh");
        let timerId = null;
        let requestSequence = 0;
        let startingAttempts = 0;
        let hasBeenReady = false;

        const renderStatus = (
          message,
          { error = false, refresh = false } = {},
        ) => {
          statusEl.textContent = message;
          statusEl.classList.toggle("director-error", error);
          refreshButton.hidden = !refresh;
        };

        const scheduleStatusQuery = (delay) => {
          if (timerId !== null) {
            window.clearTimeout(timerId);
          }
          timerId = window.setTimeout(() => {
            timerId = null;
            if (el.isConnected) {
              void queryStatus();
            }
          }, delay);
        };

        const queryStatus = async () => {
          if (!el.isConnected) return;
          const requestId = ++requestSequence;
          try {
            const response = await fetch("/directordeck/status", {
              cache: "no-store",
            });
            if (!response.ok) {
              throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();
            if (requestId !== requestSequence || !el.isConnected) return;

            if (data.backend === "ready") {
              hasBeenReady = true;
              startingAttempts = 0;
              renderStatus(`后端状态：运行中（v${data.version || "?"}）`);
              scheduleStatusQuery(RUNTIME_POLL_MS);
              return;
            }

            if (data.backend === "starting") {
              startingAttempts += 1;
              if (startingAttempts < STARTING_MAX_ATTEMPTS) {
                renderStatus("后端状态：启动中…");
                scheduleStatusQuery(STARTING_POLL_MS);
              } else {
                renderStatus("后端状态：启动状态查询超时", {
                  error: true,
                  refresh: true,
                });
              }
              return;
            }

            if (data.backend === "failed") {
              renderStatus(
                `后端状态：${hasBeenReady ? "运行失败" : "启动失败"}\n${data.error || ""}`,
                { error: true, refresh: true },
              );
              return;
            }

            if (data.backend === "stopped") {
              renderStatus(
                `后端状态：${hasBeenReady ? "已停止" : "启动已停止"}${
                  data.error ? `\n${data.error}` : ""
                }`,
                { error: true, refresh: true },
              );
              return;
            }

            renderStatus("后端状态：未知", { error: true, refresh: true });
          } catch (_error) {
            if (requestId !== requestSequence || !el.isConnected) return;
            renderStatus("后端状态：无法查询", { error: true, refresh: true });
          }
        };

        refreshButton.addEventListener("click", () => {
          if (timerId !== null) {
            window.clearTimeout(timerId);
            timerId = null;
          }
          startingAttempts = 0;
          renderStatus("后端状态：查询中…");
          void queryStatus();
        });

        void queryStatus();
      },
    });
  },
});
