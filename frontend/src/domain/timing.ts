import type { ModeDraft } from "./modes";

export const H3_FRAME_STRIDE = 17;
export const H3_FRAME_OFFSET = 5;
export const H3_MAX_SHOT_FRAMES = 512;

/**
 * Python's round() for a non-negative finite number: exact .5 ties go to the
 * nearest even integer. JavaScript Math.round() cannot be used for this.
 */
export function roundPositiveHalfEven(value: number): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError("half-even rounding requires a finite non-negative value");
  }
  const lower = Math.floor(value);
  const fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower % 2 === 0 ? lower : lower + 1;
}

/** Align an already-counted frame request to MiniMax H3's 17k+5 lattice. */
export function alignH3FrameCount(rawFrames: number): number {
  if (!Number.isFinite(rawFrames) || rawFrames <= 0) return 0;
  const frames = Math.max(5, Math.trunc(rawFrames));
  return frames +
    ((H3_FRAME_OFFSET - (frames % H3_FRAME_STRIDE)) % H3_FRAME_STRIDE +
      H3_FRAME_STRIDE) %
      H3_FRAME_STRIDE;
}

/** Matches backend align_h3_frames for values accepted by the draft schema. */
export function alignH3Frames(durationSeconds: number, fps: number): number {
  if (
    !Number.isFinite(durationSeconds) ||
    !Number.isFinite(fps) ||
    durationSeconds <= 0 ||
    fps <= 0
  )
    return 0;
  return alignH3FrameCount(roundPositiveHalfEven(durationSeconds * fps));
}

export interface EffectiveTiming {
  frames: number;
  durationSeconds: number;
}

export function effectiveShotTiming(
  durationSeconds: number,
  fps: number,
): EffectiveTiming {
  const frames = alignH3Frames(durationSeconds, fps);
  return {
    frames,
    durationSeconds: frames > 0 && fps > 0 ? frames / fps : 0,
  };
}

export function effectiveDraftTiming(draft: ModeDraft): EffectiveTiming {
  const frames = draft.shots.reduce(
    (total, shot) =>
      total +
      (shot.enabled ? alignH3Frames(shot.duration_seconds, draft.render.fps) : 0),
    0,
  );
  return {
    frames,
    durationSeconds:
      frames > 0 && Number.isFinite(draft.render.fps) && draft.render.fps > 0
        ? frames / draft.render.fps
        : 0,
  };
}
