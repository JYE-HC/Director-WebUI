"""Audited MiniMax H3 reference-input capacities.

These values mirror the stock ``MiniMaxH3ReferenceToVideo`` Autogrow
contract.  Keep the limits in one object so schema validation and native
workflow construction cannot quietly drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class H3ReferenceLimits:
    reference_images: int
    reference_video_channels: int
    standalone_reference_audios: int
    source_videos: int
    paired_reference_video_audios: int

    def independent_reference_video_capacity(
        self, *, has_source_video: bool
    ) -> int:
        """Return video slots left after the optional source uses channel 0."""

        occupied = self.source_videos if has_source_video else 0
        return self.reference_video_channels - occupied


H3_REFERENCE_LIMITS = H3ReferenceLimits(
    reference_images=9,
    reference_video_channels=3,
    standalone_reference_audios=3,
    source_videos=1,
    paired_reference_video_audios=3,
)
