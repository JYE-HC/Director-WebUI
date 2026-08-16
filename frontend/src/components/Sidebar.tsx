import { MODE_META, MODE_ORDER, type GenerationMode } from "../domain/modes";
import type { CapabilityReport } from "../api/types";
import type { AppView } from "../state/directorState";
import { StatusDot } from "./ui";

interface SidebarProps {
  id?: string;
  activeMode: GenerationMode;
  view: AppView;
  connection: CapabilityReport["connection"];
  runtimeConfigured: boolean;
  onModeSelect: (mode: GenerationMode) => void;
  onSettingsSelect: () => void;
}

export function Sidebar({
  id,
  activeMode,
  view,
  connection,
  runtimeConfigured,
  onModeSelect,
  onSettingsSelect,
}: SidebarProps) {
  const runtimeLabel = connection === "offline"
    ? "ComfyUI 离线"
    : !runtimeConfigured
      ? "ComfyUI 尚未配置"
      : connection === "online"
        ? "ComfyUI 已连接"
        : connection === "checking"
          ? "正在检测 ComfyUI"
          : "等待 ComfyUI 状态";
  const statusState = connection === "offline"
    ? "offline"
    : runtimeConfigured
      ? connection
      : "unknown";
  return (
    <aside id={id} className="sidebar">
      <div className="brand">
        <div className="brand__mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div>
          <strong>DIRECTOR</strong>
          <small>AI 电影导演台</small>
        </div>
      </div>

      <div className="sidebar__section-label">
        <span>生成模式</span>
        <span>06</span>
      </div>
      <nav className="mode-nav" aria-label="生成模式">
        {MODE_ORDER.map((mode, index) => {
          const meta = MODE_META[mode];
          const active = view === "workspace" && activeMode === mode;
          return (
            <button
              type="button"
              key={mode}
              className={`mode-nav__item mode-nav__item--${meta.accent} ${active ? "is-active" : ""}`}
              aria-current={active ? "page" : undefined}
              onClick={() => onModeSelect(mode)}
            >
              <span className="mode-nav__index">{String(index + 1).padStart(2, "0")}</span>
              <span className="mode-nav__copy">
                <strong>{meta.shortLabel}</strong>
                <small>{meta.label}</small>
              </span>
              <span className="mode-nav__indicator" aria-hidden="true" />
            </button>
          );
        })}
      </nav>

      <div className="sidebar__footer">
        <button
          type="button"
          className={`utility-nav ${view === "settings" ? "is-active" : ""}`}
          onClick={onSettingsSelect}
        >
          <span className="utility-nav__icon" aria-hidden="true">⌘</span>
          <span>
            <strong>系统设置</strong>
            <small>ComfyUI · 模型 · GPU</small>
          </span>
        </button>
        <div className="sidebar__runtime">
          <StatusDot state={statusState} />
          <span>{runtimeLabel}</span>
        </div>
      </div>
    </aside>
  );
}
