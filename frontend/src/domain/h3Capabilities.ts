/**
 * Stock MiniMaxH3ReferenceToVideo Autogrow capacities.
 *
 * Keep every editor, migration and binding path on this contract. A Ref2VA
 * source video is compiled into the first reference-video input, so it shares
 * the three video inputs with independent reference videos. Its paired audio
 * uses ref_video_audios and does not consume a standalone reference-audio slot.
 */
export const MINIMAX_H3_REFERENCE_LIMITS = Object.freeze({
  sourceVideos: 1,
  referenceImages: 9,
  totalReferenceVideos: 3,
  referenceAudios: 3,
});

export interface MiniMaxH3ReferenceCapacities {
  sourceVideo: number;
  referenceImages: number;
  referenceVideos: number;
  referenceAudios: number;
  totalReferenceVideos: number;
}

export function minimaxH3ReferenceCapacities(
  hasSourceVideo: boolean,
): MiniMaxH3ReferenceCapacities {
  return {
    sourceVideo: MINIMAX_H3_REFERENCE_LIMITS.sourceVideos,
    referenceImages: MINIMAX_H3_REFERENCE_LIMITS.referenceImages,
    referenceVideos:
      MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos -
      (hasSourceVideo ? MINIMAX_H3_REFERENCE_LIMITS.sourceVideos : 0),
    referenceAudios: MINIMAX_H3_REFERENCE_LIMITS.referenceAudios,
    totalReferenceVideos: MINIMAX_H3_REFERENCE_LIMITS.totalReferenceVideos,
  };
}

export function maxSlotForCapacity(capacity: number): number {
  return capacity - 1;
}
