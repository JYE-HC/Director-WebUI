from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .workflow.execution import OutputDescriptor


class HostOutputProbeError(RuntimeError):
    """The host could not safely produce facts for one ComfyUI output."""


class PermanentHostOutputProbeError(HostOutputProbeError):
    """The trusted descriptor can never become a valid observable video."""


class HostOutputProbeResult(BaseModel):
    """Path-free media facts returned by the in-process plugin provider."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    fps: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    frame_count: Annotated[int, Field(gt=0)]
    duration_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    has_audio: bool
    media_probe_version: Annotated[str, Field(min_length=1, max_length=128)]


class HostOutputProbeProvider(Protocol):
    """Narrow plugin-owned bridge to ComfyUI's output filesystem.

    Implementations run in a worker thread.  They must validate the descriptor
    against the current host output root and return metadata only: absolute
    paths and media bytes never cross this boundary.
    """

    def probe_output(
        self,
        descriptor: OutputDescriptor,
    ) -> HostOutputProbeResult | Mapping[str, Any]: ...


__all__ = [
    "HostOutputProbeError",
    "HostOutputProbeProvider",
    "HostOutputProbeResult",
    "PermanentHostOutputProbeError",
]
