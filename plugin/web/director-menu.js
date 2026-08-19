// ComfyUI-Director sidebar entry: open the Director workspace in a new tab,
// show embedded-backend status, and link to documentation. The workspace
// itself is a full-page SPA served at /director/.
import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "ComfyUI.Director",
  setup() {
    app.extensionManager.registerSidebarTab({
      id: "director",
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
          <div class="director-status">后端状态：查询中…</div>
          <a class="director-link" href="https://github.com/JYE-HC/Director-WebUI" target="_blank" rel="noreferrer">文档</a>
        `;
        el.querySelector(".director-open").addEventListener("click", () => {
          window.open("/director/", "_blank", "noopener");
        });
        const statusEl = el.querySelector(".director-status");
        fetch("/director/status")
          .then((resp) => resp.json())
          .then((data) => {
            if (data.backend === "ready") {
              statusEl.textContent = `后端状态：运行中（v${data.version || "?"}）`;
            } else if (data.backend === "failed") {
              statusEl.textContent = `后端状态：启动失败\n${data.error || ""}`;
              statusEl.classList.add("director-error");
            } else {
              statusEl.textContent = "后端状态：启动中…";
            }
          })
          .catch(() => {
            statusEl.textContent = "后端状态：无法查询";
            statusEl.classList.add("director-error");
          });
      },
    });
  },
});
