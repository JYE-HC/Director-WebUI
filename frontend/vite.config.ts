import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

declare const process: { env: Record<string, string | undefined> };

const frontendHost = process.env.DIRECTOR_FRONTEND_HOST || "127.0.0.1";
const frontendPort = Number(process.env.DIRECTOR_FRONTEND_PORT || "4173");
const apiOrigin = process.env.DIRECTOR_API_ORIGIN || "http://127.0.0.1:8787";

export default defineConfig({
  plugins: [react()],
  server: {
    host: frontendHost,
    port: frontendPort,
    strictPort: true,
    proxy: {
      "/api": apiOrigin,
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true,
    globals: true,
  },
});
