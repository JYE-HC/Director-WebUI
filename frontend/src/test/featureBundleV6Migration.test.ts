import { describe, expect, it } from "vitest";
import {
  createTimelineProject,
  migrateTimelineFeatureBundle5To6,
  projectTimelineFeatureBundle,
  resolveLocalTimelineWal,
  timelineProjectDocumentHash,
  TIMELINE_WAL_FORMAT,
  TIMELINE_WAL_VERSION,
  type FeatureSelection,
  type LocalTimelineWal,
  type TimelineProject,
} from "../domain/timelineProject";

function bundle5(): TimelineProject {
  const project = createTimelineProject();
  project.features = {
    template_bundle_version: 5,
    project: { lora: structuredClone(project.features.project.lora) },
    by_segment: {},
  };
  return project;
}

function select(
  project: TimelineProject,
  featureId: string,
  selection: FeatureSelection,
): TimelineProject {
  const result = structuredClone(project);
  result.features.project[featureId] = selection;
  return result;
}

function wal(base: TimelineProject, pending: TimelineProject, revision = 7): LocalTimelineWal {
  return {
    format: TIMELINE_WAL_FORMAT,
    version: TIMELINE_WAL_VERSION,
    owner_id: "bundle6-test-owner",
    pending: true,
    project_id: "default",
    active_database_path: "/tmp/directordeck.sqlite3",
    written_at_ms: 1,
    base_server_revision: revision,
    base_document_hash: timelineProjectDocumentHash(base),
    head_document_hash: timelineProjectDocumentHash(pending),
    base_project: base,
    pending_project: pending,
  };
}

describe("Bundle 5 to 6 pure feature projection", () => {
  it("keeps LoRA and maps the implicit Ray default to CK off", () => {
    const source = bundle5();
    const result = migrateTimelineFeatureBundle5To6(source);

    expect(result?.features).toEqual({
      template_bundle_version: 6,
      project: {
        lora: source.features.project.lora,
        comfy_kitchen_attention: { enabled: false, params: {} },
      },
      by_segment: {},
    });
    expect(source.features.template_bundle_version).toBe(5);
  });

  it("maps an explicit Standard CK request and stays idempotent", () => {
    const source = select(bundle5(), "attention_backend_override", {
      enabled: true,
      params: { mode: "ck_int8" },
    });
    const migrated = migrateTimelineFeatureBundle5To6(source);

    expect(migrated?.features.project.comfy_kitchen_attention).toEqual({
      enabled: true,
      params: {},
    });
    expect(migrateTimelineFeatureBundle5To6(migrated)).toEqual(migrated);
  });

  it.each([
    ["explicit PyTorch", "attention_backend_override", { mode: "pytorch" }],
    ["Ray-only CK", "raylight_pool_intent", { attention: "ck_int8" }],
    ["retired low-VRAM", "h3_low_vram_attention", {}],
  ])("leaves %s authority in Bundle 5", (_label, featureId, params) => {
    const source = select(bundle5(), featureId, { enabled: true, params });
    expect(migrateTimelineFeatureBundle5To6(source)).toBeNull();
    expect(source.features.template_bundle_version).toBe(5);
  });

  it("does not expand disagreeing segment intent into one hidden project value", () => {
    const source = bundle5();
    const second = structuredClone(source.segments[0]);
    second.id = "segment-2";
    source.segments.push(second);
    source.features.by_segment[source.segments[0].id] = {
      attention_backend_override: { enabled: true, params: { mode: "ck_int8" } },
    };

    expect(migrateTimelineFeatureBundle5To6(source)).toBeNull();
  });

  it("projects a legacy marker chain without runtime or capability input", () => {
    const source = bundle5();
    source.features.template_bundle_version = 4;

    const result = projectTimelineFeatureBundle(source, 6);

    expect(result?.revision_steps).toBe(2);
    expect(result?.document.features.template_bundle_version).toBe(6);
    expect(result?.document.features.project.comfy_kitchen_attention.enabled).toBe(false);
  });
});

describe("Bundle 6 WAL recovery projection", () => {
  it("replays a mapped pending branch after the server migration revision", () => {
    const base = bundle5();
    const pending = { ...structuredClone(base), title: "pending Bundle 5 edit" };
    const migratedBase = migrateTimelineFeatureBundle5To6(base)!;
    const migratedPending = migrateTimelineFeatureBundle5To6(pending)!;

    expect(resolveLocalTimelineWal(wal(base, pending), {
      revision: 8,
      document: migratedBase,
    })).toEqual({
      status: "replay",
      project: migratedPending,
      expected_server_revision: 8,
    });
  });

  it("recognizes a mapped pending branch whose PUT was acknowledged before migration", () => {
    const base = bundle5();
    const pending = { ...structuredClone(base), title: "acknowledged Bundle 5 edit" };
    const migratedPending = migrateTimelineFeatureBundle5To6(pending)!;

    expect(resolveLocalTimelineWal(wal(base, pending), {
      revision: 9,
      document: migratedPending,
    })).toEqual({
      status: "acknowledged",
      project: migratedPending,
      server_revision: 9,
    });
  });

  it("keeps an unmappable branch as a local conflict", () => {
    const base = bundle5();
    const pending = select(base, "attention_backend_override", {
      enabled: true,
      params: { mode: "pytorch" },
    });
    const migratedBase = migrateTimelineFeatureBundle5To6(base)!;

    expect(resolveLocalTimelineWal(wal(base, pending), {
      revision: 8,
      document: migratedBase,
    })).toMatchObject({
      status: "conflict",
      local_project: pending,
      server_project: migratedBase,
    });
  });
});
