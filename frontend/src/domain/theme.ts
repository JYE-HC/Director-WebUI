export type UiTheme = "dark" | "light";

export const THEME_STORAGE_KEY = "director-web:theme";

export function readUiTheme(): UiTheme {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // Browser UI preferences remain usable when storage is unavailable.
  }
  return "dark";
}

export function persistUiTheme(theme: UiTheme): void {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // The active document can still use the selected theme for this session.
  }
}
