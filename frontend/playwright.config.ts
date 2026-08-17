import { defineConfig, devices } from "@playwright/test";

declare const process: { env: Record<string, string | undefined> };

const host = "127.0.0.1";
// Keep E2E isolated from the project's normal developer server (4173). Reusing
// a long-lived Vite process after package changes can serve obsolete optimize
// hashes even though its root URL still looks healthy.
const port = 4174;

export default defineConfig({
  testDir: "./e2e",
  // Vite may invalidate its first optimized-dependency hash while several
  // fresh browser pages race the initial prebundle. One worker keeps the real
  // browser suite deterministic and also prevents its stateful API mocks from
  // creating unnecessary server pressure.
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  outputDir: "/tmp/director-web-playwright-results",
  use: {
    baseURL: `http://${host}:${port}`,
    headless: true,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    // Rebuild Vite's optimized dependency graph before accepting a browser.
    // This matters after Playwright itself is first added to node_modules: an
    // old dev-cache hash otherwise produces a blank page with HTTP 504.
    command: `npm run dev -- --force --host ${host} --port ${port}`,
    url: `http://${host}:${port}`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          executablePath: "/usr/bin/google-chrome",
          args: ["--no-sandbox"],
        },
      },
    },
    {
      name: "webkit",
      use: {
        ...devices["Desktop Safari"],
      },
    },
  ],
});
