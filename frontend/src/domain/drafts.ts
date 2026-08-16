import {
  createInitialDrafts,
  isSamplingScheduler,
  randomSafeSeed,
  type GenerationMode,
  type ModeDraftMap,
  type RenderConfig,
  type SamplingConfig,
  type ShotBase,
} from "./modes";
import {
  normalizeAssetReference,
  normalizeSlottedAssetList,
} from "./assets";
import {
  MINIMAX_H3_REFERENCE_LIMITS,
  maxSlotForCapacity,
} from "./h3Capabilities";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function normalizeRender(value: unknown): RenderConfig | null {
  if (!isRecord(value)) return null;
  if (![value.width, value.height, value.fps].every(finiteNumber)) return null;
  return {
    width: value.width as number,
    height: value.height as number,
    // MiniMax H3 native conditioning/reference video paths are fixed at 24fps.
    fps: 24,
  };
}

function normalizeSampling(value: unknown): SamplingConfig | null {
  if (!isRecord(value)) return null;
  if (
    ![value.steps, value.seed, value.shift, value.audio_shift].every(
      finiteNumber,
    )
  )
    return null;
  const sampler = ["res_multistep", "euler", "dpmpp_2m"].includes(
    String(value.sampler),
  )
    ? (value.sampler as SamplingConfig["sampler"])
    : "res_multistep";
  const scheduler = isSamplingScheduler(value.scheduler)
    ? (value.scheduler as SamplingConfig["scheduler"])
    : "simple";
  return {
    steps: value.steps as number,
    seed: value.seed === -1 ? randomSafeSeed() : value.seed as number,
    random_seed: value.random_seed === true || value.seed === -1,
    sampler,
    scheduler,
    shift: value.shift as number,
    audio_shift: value.audio_shift as number,
  };
}

function normalizeShotBase(value: unknown): (ShotBase & Record<string, unknown>) | null {
  if (!isRecord(value)) return null;
  if (
    typeof value.id !== "string" ||
    typeof value.title !== "string" ||
    typeof value.prompt !== "string" ||
    !finiteNumber(value.duration_seconds) ||
    typeof value.enabled !== "boolean"
  )
    return null;
  return {
    id: value.id,
    title: value.title,
    prompt: value.prompt,
    duration_seconds: value.duration_seconds,
    enabled: value.enabled,
  };
}

/**
 * Rebuild one draft from an explicit per-mode allow-list.
 *
 * API responses and localStorage are both untrusted runtime input. Rebuilding
 * prevents stale fields from another task mode (or a removed feature) from
 * surviving in memory and later causing FastAPI's extra=forbid contract to
 * reject save/submission.
 */
export function normalizeModeDraft<M extends GenerationMode>(
  value: unknown,
  expectedMode: M,
): ModeDraftMap[M] | null {
  if (!isRecord(value) || value.mode !== expectedMode) return null;
  if (
    typeof value.prompt !== "string" ||
    !Array.isArray(value.shots) ||
    value.shots.length < 1 ||
    value.shots.length > 128
  )
    return null;
  const render = normalizeRender(value.render);
  const sampling = normalizeSampling(value.sampling);
  if (!render || !sampling) return null;

  const bases = value.shots.map(normalizeShotBase);
  if (bases.some((shot) => shot === null)) return null;
  const shots = bases as (ShotBase & Record<string, unknown>)[];
  const common = {
    mode: expectedMode,
    prompt: value.prompt,
    // Drafts saved before this field existed migrate to the official default.
    // Removed compatibility fields such as negative_prompt are intentionally
    // not copied into the rebuilt allow-list.
    ref_image_size: value.ref_image_size === "max" ? "max" as const : "match" as const,
    render,
    sampling,
  };

  switch (expectedMode) {
    case "t2v":
      return { ...common, mode: "t2v", shots } as unknown as ModeDraftMap[M];
    case "i2v":
      return {
        ...common,
        mode: "i2v",
        shots: shots.map((shot, index) => ({
          ...shot,
          first_image: normalizeAssetReference(
            (value.shots as Record<string, unknown>[])[index]?.first_image,
            "image",
          ),
        })),
      } as unknown as ModeDraftMap[M];
    case "fl2v":
      return {
        ...common,
        mode: "fl2v",
        shots: shots.map((shot, index) => {
          const source = (value.shots as Record<string, unknown>[])[index];
          return {
            ...shot,
            first_image: normalizeAssetReference(source?.first_image, "image"),
            last_image: normalizeAssetReference(source?.last_image, "image"),
          };
        }),
      } as unknown as ModeDraftMap[M];
    case "r2v":
      return {
        ...common,
        mode: "r2v",
        shots: shots.map((shot, index) => {
          const source = (value.shots as Record<string, unknown>[])[index];
          return {
            ...shot,
            reference_images: normalizeSlottedAssetList(
              source?.reference_images,
              "image",
              maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages),
            ),
            reference_audios: normalizeSlottedAssetList(
              source?.reference_audios,
              "audio",
              maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios),
            ),
            reference_videos: normalizeSlottedAssetList(
              source?.reference_videos,
              "video",
              maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos),
            ),
          };
        }),
      } as unknown as ModeDraftMap[M];
    case "v2v": {
      const normalizedShots = shots.map((shot, index) => {
        const source = (value.shots as Record<string, unknown>[])[index];
        if (
          !finiteNumber(source?.source_start_seconds) ||
          !finiteNumber(source?.source_duration_seconds)
        )
          return null;
        return {
          ...shot,
          source_video: normalizeAssetReference(source.source_video, "video"),
          source_start_seconds: source.source_start_seconds,
          source_duration_seconds: source.source_duration_seconds,
        };
      });
      if (normalizedShots.some((shot) => shot === null)) return null;
      return {
        ...common,
        mode: "v2v",
        shots: normalizedShots,
      } as unknown as ModeDraftMap[M];
    }
    case "rv2v": {
      const normalizedShots = shots.map((shot, index) => {
        const source = (value.shots as Record<string, unknown>[])[index];
        if (
          !finiteNumber(source?.source_start_seconds) ||
          !finiteNumber(source?.source_duration_seconds)
        )
          return null;
        return {
          ...shot,
          source_video: normalizeAssetReference(source.source_video, "video"),
          source_start_seconds: source.source_start_seconds,
          source_duration_seconds: source.source_duration_seconds,
          reference_images: normalizeSlottedAssetList(
            source.reference_images,
            "image",
            maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceImages),
          ),
          reference_audios: normalizeSlottedAssetList(
            source.reference_audios,
            "audio",
            maxSlotForCapacity(MINIMAX_H3_REFERENCE_LIMITS.referenceAudios),
          ),
        };
      });
      if (normalizedShots.some((shot) => shot === null)) return null;
      return {
        ...common,
        mode: "rv2v",
        shots: normalizedShots,
      } as unknown as ModeDraftMap[M];
    }
  }
}

export function normalizedDraftOrDefault<M extends GenerationMode>(
  value: unknown,
  mode: M,
): ModeDraftMap[M] {
  return normalizeModeDraft(value, mode) ?? createInitialDrafts()[mode];
}
