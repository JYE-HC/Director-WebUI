import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { RootErrorBoundary } from "./components/RootErrorBoundary";
import { readUiTheme } from "./domain/theme";
import "./styles.css";

document.documentElement.dataset.theme = readUiTheme();
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </StrictMode>,
);
