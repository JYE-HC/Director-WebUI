from __future__ import annotations

"""Exact project-bundle compile-authority dispatcher."""

from collections.abc import Mapping

from ..native_templates import NativeHistoricalTake
from ..schemas import RuntimeSettingsV3, UnifiedTimelineDraftV5
from .execution import CompiledExecutionPlan
from .v5_compat import compile_v5_execution_plan
from .v6_execution_adapter import compile_v6_execution_plan


class ProjectCompilerBundleError(ValueError):
    def __init__(self, bundle_version: int) -> None:
        super().__init__(
            f"project template bundle {bundle_version} has no installed compiler"
        )
        self.bundle_version = bundle_version


def compile_project_execution_plan(
    draft: UnifiedTimelineDraftV5,
    settings: RuntimeSettingsV3,
    job_id: str,
    segment_ids: list[str] | None = None,
    *,
    historical_takes: Mapping[str, NativeHistoricalTake] | None = None,
) -> CompiledExecutionPlan:
    """Compile with the exact compiler named by the captured authority."""

    bundle = draft.features.template_bundle_version
    if bundle == 5:
        return compile_v5_execution_plan(
            draft,
            settings,
            job_id,
            segment_ids,
            historical_takes=historical_takes,
        )
    if bundle == 6:
        return compile_v6_execution_plan(
            draft,
            settings,
            job_id,
            segment_ids,
            historical_takes=historical_takes,
        )
    raise ProjectCompilerBundleError(bundle)


__all__ = [
    "ProjectCompilerBundleError",
    "compile_project_execution_plan",
]
