import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

declare const process: { env: Record<string, string | undefined> };

const frontendHost = process.env.DIRECTOR_FRONTEND_HOST || "127.0.0.1";
const frontendPort = Number(process.env.DIRECTOR_FRONTEND_PORT || "4173");
const apiOrigin = process.env.DIRECTOR_API_ORIGIN || "http://127.0.0.1:8787";

export default defineConfig(({ command }) => ({
  // Built assets are served by the ComfyUI plugin under /director/; a
  // relative base keeps the same dist usable from any mount point.
  base: command === "build" ? "./" : "/",
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
    // Playwright specs use a different runner. Keep Vitest's discovery rooted
    // in the jsdom unit/integration suite so `npm test` never imports `test()`
    // from @playwright/test as a zero-test Vitest suite.
    include: ["src/test/**/*.{test,spec}.{ts,tsx}"],
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
    css: true,
    globals: true,
    // Must stay above asyncUtilTimeout (15s) so slow-runner waits report the
    // Testing Library error instead of a bare Vitest timeout.
    testTimeout: 20000,
  },
}));
