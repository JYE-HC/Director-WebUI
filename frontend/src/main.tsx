import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { RootErrorBoundary } from "./components/RootErrorBoundary";
import { readUiTheme } from "./domain/theme";
import { I18nProvider } from "./i18n";
import "./styles.css";

document.documentElement.dataset.theme = readUiTheme();
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RootErrorBoundary>
      <I18nProvider>
        <App />
      </I18nProvider>
    </RootErrorBoundary>
  </StrictMode>,
);
