import type { Page, Route } from "@playwright/test";
import {
  DEFAULT_SETTINGS,
  EMPTY_RAYLIGHT_RUNTIME_STATUS,
  type RuntimeSettings,
} from "../src/api/types";
import {
  createTimelineProject,
  DEFAULT_PROJECT_ID,
  type TimelineProject,
} from "../src/domain/timelineProject";

export const INITIAL_PROMPT = "camera move";

const DATABASE_IDENTITY = "e".repeat(64);
const RUNTIME_AUTHORITY = "a".repeat(64);
const NOW = "2026-08-16T00:00:00Z";

const settings: RuntimeSettings = {
  ...structuredClone(DEFAULT_SETTINGS),
  comfy_url: "http://comfy.test:8188",
};

function initialProject(): TimelineProject {
  const project = createTimelineProject();
  project.title = "E2E Undo 项目";
  project.sampling.fl2va.seed = 101;
  project.sampling.ref2va.seed = 202;
  project.segments[0] = {
    ...project.segments[0],
    id: "segment-e2e-prompt",
    title: "E2E 片段",
    prompt: INITIAL_PROMPT,
  };
  return project;
}

export interface DirectorApiMockAuthority {
  project: TimelineProject;
  revision: number;
  timelineAuthorityPutAttempts: number;
  rejectTimelineAuthorityPuts: boolean;
}

export function createDirectorApiMockAuthority(): DirectorApiMockAuthority {
  return {
    project: initialProject(),
    revision: 0,
    timelineAuthorityPutAttempts: 0,
    rejectTimelineAuthorityPuts: false,
  };
}

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

/**
 * Installs one stateful, page-local Director API authority.
 *
 * Tests exercise the production client, hydration and autosave code. Only the
 * network edge is replaced: a timeline PUT becomes the next GET authority,
 * matching the backend's exact-echo acknowledgement contract.
 */
export async function installDirectorApiMock(
  page: Page,
  authority: DirectorApiMockAuthority = createDirectorApiMockAuthority(),
): Promise<void> {

  await page.route("**/directordeck/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    // The glob also sees Vite source modules such as /src/api/client.ts.
    // Intercept only the server namespace and let application modules load.
    if (path !== "/directordeck/api" && !path.startsWith("/directordeck/api/")) {
      await route.continue();
      return;
    }

    if (path === "/directordeck/api/tasks/events") {
      // HTTP 204 tells EventSource not to reconnect. Task events are unrelated
      // to document history and a closed stream keeps the test deterministic.
      await route.fulfill({ status: 204 });
      return;
    }
    if (path === "/directordeck/api/storage" && method === "GET") {
      await json(route, {
        active_database_path: "/srv/directordeck/e2e.sqlite3",
      });
      return;
    }
    if (path === "/directordeck/api/settings/authority" && method === "GET") {
      await json(route, { settings, authority_token: RUNTIME_AUTHORITY });
      return;
    }
    if (path === "/directordeck/api/settings" && method === "GET") {
      await json(route, settings);
      return;
    }
    if (path === "/directordeck/api/capabilities" && method === "GET") {
      await json(route, {
        connection: "online",
        supported_modes: ["t2v", "i2v", "fl2v", "r2v", "v2v", "rv2v"],
        supports_cancel: true,
        available_nodes: [],
        missing_nodes: [],
        execution_backends: {
          standard: { available: true, missing_nodes: [] },
          raylight: { available: true, missing_nodes: [] },
        },
      });
      return;
    }
    if (path === "/directordeck/api/gpus" && method === "GET") {
      await json(route, { gpus: [] });
      return;
    }
    if (path === "/directordeck/api/models" && method === "GET") {
      await json(route, {
        fl2va: [settings.models.fl2va.filename],
        ref2va: [settings.models.ref2va.filename],
        clip: [settings.models.clip.filename],
        video_vae: [settings.models.video_vae.filename],
        audio_vae: [settings.models.audio_vae.filename],
        loras: [],
      });
      return;
    }
    if (path === "/directordeck/api/raylight/runtime" && method === "GET") {
      await json(route, EMPTY_RAYLIGHT_RUNTIME_STATUS);
      return;
    }
    if (path === "/directordeck/api/projects" && method === "GET") {
      await json(route, {
        active_database_identity: DATABASE_IDENTITY,
        projects: [{
          id: DEFAULT_PROJECT_ID,
          title: authority.project.title,
          created_at: NOW,
          updated_at: NOW,
          segment_count: authority.project.segments.length,
        }],
      });
      return;
    }
    if (path === "/directordeck/api/timeline" && method === "GET") {
      await json(route, authority.project);
      return;
    }
    if (path === "/directordeck/api/timeline/authority" && method === "GET") {
      await json(route, { document: authority.project, revision: authority.revision });
      return;
    }
    if (path === "/directordeck/api/timeline/authority" && method === "PUT") {
      authority.timelineAuthorityPutAttempts += 1;
      if (authority.rejectTimelineAuthorityPuts) {
        await route.abort("failed");
        return;
      }
      const body = request.postDataJSON() as {
        document: TimelineProject;
        expected_revision: number;
      };
      if (body.expected_revision !== authority.revision) {
        await json(route, {
          detail: {
            code: "timeline_revision_conflict",
            message: "timeline revision conflict",
            project_id: DEFAULT_PROJECT_ID,
            expected_revision: body.expected_revision,
            actual_revision: authority.revision,
          },
        }, 409);
        return;
      }
      authority.project = structuredClone(body.document);
      authority.revision += 1;
      await json(route, { document: authority.project, revision: authority.revision });
      return;
    }
    if (path === "/directordeck/api/timeline" && method === "PUT") {
      const body = request.postDataJSON() as TimelineProject;
      authority.project = structuredClone(body);
      authority.revision += 1;
      await json(route, authority.project);
      return;
    }
    if (path === "/directordeck/api/assets" && method === "GET") {
      await json(route, {
        assets: [],
        outputs_preserved: true,
        active_database_identity: DATABASE_IDENTITY,
        comfy_origin: settings.comfy_url,
      });
      return;
    }
    if (path === "/directordeck/api/jobs" && method === "GET") {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      await json(route, {
        jobs: [],
        total: 0,
        limit: 256,
        offset,
        has_more: false,
        summary: {
          total: 0,
          active: 0,
          queued: 0,
          preparing: 0,
          running: 0,
          cancelling: 0,
          succeeded: 0,
          failed: 0,
          cancelled: 0,
        },
      });
      return;
    }

    await json(route, { detail: `unhandled E2E API route: ${method} ${path}` }, 404);
  });
}
