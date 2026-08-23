from __future__ import annotations

"""Safe host-capability capture and transient readiness construction.

The provider lives in the ComfyUI plugin process.  This module remains a pure
backend consumer: it never imports ComfyUI internals and never augments the
provider snapshot with queue, task, or Ray-ledger state.
"""

from collections.abc import Iterable

from pydantic import model_validator

from ..workflow.contracts import (
    ContractModel,
    HostCapabilityProvider,
    HostCapabilitySnapshot,
    OperationalReadiness,
    Sha256Digest,
)


class CapturedHostCapabilities(ContractModel):
    """A revalidated immutable snapshot paired with its canonical revision."""

    snapshot: HostCapabilitySnapshot
    host_capability_revision: Sha256Digest

    @model_validator(mode="after")
    def _bind_revision(self) -> "CapturedHostCapabilities":
        if self.host_capability_revision != self.snapshot.host_capability_revision():
            raise ValueError("host capability revision does not match snapshot")
        return self


def host_capability_revision(snapshot: HostCapabilitySnapshot) -> str:
    """Return the static revision; observation time is deliberately excluded."""

    if not isinstance(snapshot, HostCapabilitySnapshot):
        raise TypeError("host capability snapshot must be validated")
    return snapshot.host_capability_revision()


def capture_host_capabilities(
    provider: HostCapabilityProvider,
) -> CapturedHostCapabilities:
    """Capture and JSON-round-trip provider output at the trust boundary.

    A plugin provider cannot smuggle subclass attributes or mutable containers
    through this boundary.  ``HostCapabilitySnapshot`` performs the privacy,
    path, credential, registry, and GPU-inventory validation.
    """

    snapshot = provider.snapshot()
    if not isinstance(snapshot, HostCapabilitySnapshot):
        raise TypeError("HostCapabilityProvider.snapshot() returned an invalid type")
    safe_snapshot = HostCapabilitySnapshot.model_validate_json(
        snapshot.model_dump_json()
    )
    return CapturedHostCapabilities(
        snapshot=safe_snapshot,
        host_capability_revision=safe_snapshot.host_capability_revision(),
    )


def build_operational_readiness(
    *,
    endpoint_online: bool,
    ray_recovery_required: bool = False,
    ray_tainted: bool = False,
    ray_cleanup_available: bool = False,
    runtime_gpu_indices: Iterable[int] = (),
    available_logical_gpu_count: int = 0,
    blocking_reason_codes: Iterable[str] = (),
) -> OperationalReadiness:
    """Project transient transport/ledger facts into one strict readiness value.

    Callers read endpoint status and the durable Ray ledger, then pass only the
    normalized facts here.  This function does not read either authority and
    its result must never participate in the static catalog ETag.
    """

    if type(endpoint_online) is not bool:
        raise TypeError("endpoint_online must be boolean")
    if (
        type(ray_recovery_required) is not bool
        or type(ray_tainted) is not bool
        or type(ray_cleanup_available) is not bool
    ):
        raise TypeError("Ray readiness flags must be boolean")
    if (
        type(available_logical_gpu_count) is not int
        or available_logical_gpu_count < 0
        or available_logical_gpu_count > 256
    ):
        raise ValueError("available logical GPU count must be between 0 and 256")

    requested_indices = tuple(dict.fromkeys(runtime_gpu_indices))
    if any(
        type(index) is not int or index < 0 or index > 255
        for index in requested_indices
    ):
        raise ValueError("runtime GPU indices must be integers between 0 and 255")
    invalid_indices = tuple(
        index
        for index in requested_indices
        if index >= available_logical_gpu_count
    )

    reasons = list(dict.fromkeys(blocking_reason_codes))
    if not endpoint_online and "endpoint_offline" not in reasons:
        reasons.append("endpoint_offline")
    if ray_recovery_required and "ray_recovery_required" not in reasons:
        reasons.append("ray_recovery_required")
    if (
        ray_tainted
        and not ray_cleanup_available
        and "ray_cleanup_unavailable" not in reasons
    ):
        reasons.append("ray_cleanup_unavailable")
    if invalid_indices and "invalid_runtime_gpu_indices" not in reasons:
        reasons.append("invalid_runtime_gpu_indices")

    submission_allowed = not reasons
    return OperationalReadiness(
        endpoint_online=endpoint_online,
        submission_allowed=submission_allowed,
        ray_recovery_required=ray_recovery_required,
        ray_tainted=ray_tainted,
        invalid_runtime_gpu_indices=invalid_indices,
        blocking_reason_codes=tuple(reasons),
    )


__all__ = [
    "CapturedHostCapabilities",
    "build_operational_readiness",
    "capture_host_capabilities",
    "host_capability_revision",
]
