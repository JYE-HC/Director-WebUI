from __future__ import annotations

"""Pure current-bundle feature migration and effective-selection resolution.

The resolver is deliberately graph-free.  Catalog, preflight and compile share
the same immutable v5 template definitions, while callers may resolve the same
captured draft more than once without database, host, or runtime side effects.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError

from ..schemas import (
    FeatureConfiguration,
    FeatureSelection,
    LoraFeatureParams,
    StrictModel,
    UnifiedTimelineDraftV5,
)
from .contracts import (
    Backend,
    ContractModel,
    FrozenMap,
    Identifier,
    JsonObject,
    ModelFamily,
    PositiveVersion,
)
from .templates import V5_TEMPLATE_BUNDLE


class EmptyFeatureParams(StrictModel):
    """Strict empty object used by parameterless feature contracts."""


class AttentionBackendOverrideParams(StrictModel):
    mode: Literal["pytorch", "ck_int8"]


class RayLightPoolIntentParams(StrictModel):
    attention: Literal["ck_int8", "torch_flash"]


FeatureParamModel = type[StrictModel]


class V5FeatureConfigurationError(ValueError):
    """One captured feature document cannot resolve against bundle 5."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        feature_id: str | None = None,
        segment_id: str | None = None,
        backend: Backend | None = None,
        safe_details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.feature_id = feature_id
        self.segment_id = segment_id
        self.backend = backend
        self.safe_details = dict(safe_details or {})


@dataclass(frozen=True, slots=True)
class _FeatureDefinition:
    id: str
    version: int
    mode: Literal["switch", "needed"]
    scopes: tuple[Literal["project", "segment"], ...]
    visibility: Literal["user", "internal"]
    params_model: FeatureParamModel
    backends: tuple[Backend, ...]
    default: FeatureSelection


class EffectiveFeatureSelection(ContractModel):
    id: Identifier
    version: PositiveVersion
    mode: Literal["switch", "needed"]
    selection_enabled: bool
    active: bool
    params: JsonObject
    source: Literal["template_default", "project", "segment", "context"]

    def active_cache_projection(self) -> dict[str, Any] | None:
        """Return the only selection data allowed into execution identity."""

        if not self.active:
            return None
        return {
            "feature": f"{self.id}@{self.version}",
            "params": _plain_json(self.params),
        }


class EffectiveSegmentFeatures(ContractModel):
    segment_id: str
    backend: Backend
    family: ModelFamily
    template_id: Literal["h3_standard_segment", "h3_raylight_segment"]
    features: tuple[EffectiveFeatureSelection, ...]


class EffectiveFeatureConfiguration(ContractModel):
    template_bundle_version: Literal[5] = 5
    source_template_bundle_version: Literal[4, 5]
    migrated_from_bundle_4: bool
    effective_by_segment: FrozenMap[str, EffectiveSegmentFeatures]


@dataclass(frozen=True, slots=True)
class MigratedFeatureConfiguration:
    """Detached, deterministic bundle-5 compatibility view."""

    source_template_bundle_version: Literal[4, 5]
    configuration: FeatureConfiguration


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _selection_from_default(
    value: Mapping[str, Any],
    feature_id: str,
    mode: Literal["switch", "needed"],
) -> FeatureSelection:
    try:
        return FeatureSelection(
            enabled=mode == "needed",
            params=_plain_json(value),
        )
    except ValidationError as exc:  # pragma: no cover - import-time invariant.
        raise AssertionError(
            f"bundle 5 feature {feature_id!r} has an invalid default selection"
        ) from exc


def _build_definitions() -> OrderedDict[str, _FeatureDefinition]:
    params_models: dict[str, FeatureParamModel] = {
        "attention_backend_override": AttentionBackendOverrideParams,
        "h3_low_vram_attention": EmptyFeatureParams,
        "raylight_pool_intent": RayLightPoolIntentParams,
        "lora": LoraFeatureParams,
    }
    merged: OrderedDict[str, _FeatureDefinition] = OrderedDict()
    for template in (
        V5_TEMPLATE_BUNDLE.segment_templates.standard,
        V5_TEMPLATE_BUNDLE.segment_templates.raylight,
    ):
        for entry in template.entries:
            visibility = entry.ui.get("visibility")
            if visibility not in {"user", "internal"}:
                raise AssertionError(
                    f"bundle 5 feature {entry.id!r} has unstable visibility"
                )
            scopes = tuple(entry.scopes)
            if not scopes or any(scope not in {"project", "segment"} for scope in scopes):
                raise AssertionError(
                    f"bundle 5 feature {entry.id!r} has an invalid scope"
                )
            definition = _FeatureDefinition(
                id=entry.id,
                version=entry.version,
                mode=entry.mode,
                scopes=scopes,  # type: ignore[arg-type]
                visibility=visibility,  # type: ignore[arg-type]
                params_model=params_models.get(entry.id, EmptyFeatureParams),
                backends=entry.backends,
                default=_selection_from_default(entry.defaults, entry.id, entry.mode),
            )
            existing = merged.get(entry.id)
            if existing is None:
                merged[entry.id] = definition
                continue
            if (
                existing.version != definition.version
                or existing.mode != definition.mode
                or existing.scopes != definition.scopes
                or existing.visibility != definition.visibility
                or existing.params_model is not definition.params_model
                or existing.default.model_dump(mode="json")
                != definition.default.model_dump(mode="json")
            ):
                raise AssertionError(
                    f"bundle 5 feature {entry.id!r} has conflicting definitions"
                )
            merged[entry.id] = _FeatureDefinition(
                id=existing.id,
                version=existing.version,
                mode=existing.mode,
                scopes=existing.scopes,
                visibility=existing.visibility,
                params_model=existing.params_model,
                backends=tuple(
                    dict.fromkeys((*existing.backends, *entry.backends))
                ),
                default=existing.default,
            )
    return merged


_DEFINITIONS = _build_definitions()
_V4_AUTHORABLE_FEATURES = frozenset({"lora"})


def _validate_params(
    definition: _FeatureDefinition,
    selection: FeatureSelection,
    *,
    segment_id: str | None,
) -> FeatureSelection:
    try:
        definition.params_model.model_validate(selection.params)
    except ValidationError as exc:
        raise V5FeatureConfigurationError(
            "The feature parameters do not match the installed feature version.",
            code="feature_params_invalid",
            feature_id=definition.id,
            segment_id=segment_id,
        ) from exc
    if definition.mode == "needed" and not selection.enabled:
        raise V5FeatureConfigurationError(
            "A required feature cannot be disabled.",
            code="needed_feature_disabled",
            feature_id=definition.id,
            segment_id=segment_id,
        )
    return FeatureSelection.model_validate(selection.model_dump(mode="json"))


def _validate_explicit_selection(
    feature_id: str,
    selection: FeatureSelection,
    *,
    scope: Literal["project", "segment"],
    source_bundle_version: Literal[4, 5],
    segment_id: str | None,
) -> FeatureSelection:
    definition = _DEFINITIONS.get(feature_id)
    if definition is None or (
        source_bundle_version == 4 and feature_id not in _V4_AUTHORABLE_FEATURES
    ):
        raise V5FeatureConfigurationError(
            "The project contains a feature unknown to its template bundle.",
            code="unknown_feature",
            feature_id=feature_id,
            segment_id=segment_id,
        )
    if scope not in definition.scopes:
        raise V5FeatureConfigurationError(
            "The feature selection is stored at an unsupported scope.",
            code="feature_scope_unsupported",
            feature_id=feature_id,
            segment_id=segment_id,
            safe_details={"scope": scope},
        )
    validated = _validate_params(definition, selection, segment_id=segment_id)
    if definition.visibility != "user":
        raise V5FeatureConfigurationError(
            "The feature is controlled internally and cannot be authored.",
            code="feature_not_authorable",
            feature_id=feature_id,
            segment_id=segment_id,
        )
    return validated


def migrate_feature_configuration_to_v5(
    features: FeatureConfiguration,
) -> MigratedFeatureConfiguration:
    """Return a detached bundle-5 view without mutating the timeline draft.

    Bundle 4 has one explicit compatibility migration: its project-scoped LoRA
    selection retains exact bytes/semantics and bundle-5-only switches inherit
    their disabled template defaults.  No unknown field or wrong scope is
    silently dropped.
    """

    source = features.template_bundle_version
    if source not in {4, 5}:
        raise V5FeatureConfigurationError(
            "The project feature bundle is not supported by this DirectorDeck build.",
            code="template_bundle_version_unsupported",
            safe_details={"actual": source, "supported": [4, 5]},
        )
    source_version: Literal[4, 5] = source  # type: ignore[assignment]
    project: dict[str, FeatureSelection] = {}
    for feature_id, selection in features.project.items():
        project[feature_id] = _validate_explicit_selection(
            feature_id,
            selection,
            scope="project",
            source_bundle_version=source_version,
            segment_id=None,
        )
    by_segment: dict[str, dict[str, FeatureSelection]] = {}
    for segment_id, selections in features.by_segment.items():
        by_segment[segment_id] = {
            feature_id: _validate_explicit_selection(
                feature_id,
                selection,
                scope="segment",
                source_bundle_version=source_version,
                segment_id=segment_id,
            )
            for feature_id, selection in selections.items()
        }
    migrated = FeatureConfiguration(
        template_bundle_version=V5_TEMPLATE_BUNDLE.version,
        project=project,
        by_segment=by_segment,
    )
    return MigratedFeatureConfiguration(
        source_template_bundle_version=source_version,
        configuration=migrated,
    )


def migrate_timeline_feature_authority_to_v5(
    draft: UnifiedTimelineDraftV5,
) -> UnifiedTimelineDraftV5:
    """Validate and detach one schema-v5 authority at current bundle 5.

    This is the single pure upgrade used by startup persistence, CAS writes,
    imports and compile projection.  A bundle-4 authority keeps every authored
    selection exactly and acquires only the bundle-version marker; new bundle-5
    defaults remain implicit in the template and are never injected as authored
    project records.
    """

    migrated = migrate_feature_configuration_to_v5(draft.features)
    return draft.model_copy(
        update={"features": migrated.configuration},
        deep=True,
    )


def _route_selection(
    definition: _FeatureDefinition,
    *,
    project: Mapping[str, FeatureSelection],
    segment: Mapping[str, FeatureSelection],
    segment_id: str,
) -> tuple[FeatureSelection, Literal["template_default", "project", "segment"]]:
    selection = definition.default
    source: Literal["template_default", "project", "segment"] = "template_default"
    if definition.id in project:
        selection = project[definition.id]
        source = "project"
    if definition.id in segment:
        # Whole-object replacement is intentional: params are never deep-merged
        # with project/default values.
        selection = segment[definition.id]
        source = "segment"
    return _validate_params(definition, selection, segment_id=segment_id), source


def resolve_v5_effective_features(
    draft: UnifiedTimelineDraftV5,
    *,
    selected_segment_ids: tuple[str, ...],
    backend_by_family: Mapping[ModelFamily, Backend],
    contextual_switches: Mapping[str, frozenset[str]] | None = None,
) -> EffectiveFeatureConfiguration:
    """Resolve defaults -> project -> segment by whole-selection replacement."""

    migrated = migrate_feature_configuration_to_v5(draft.features)
    configuration = migrated.configuration
    segments = {segment.id: segment for segment in draft.segments}
    if (
        not selected_segment_ids
        or len(selected_segment_ids) != len(set(selected_segment_ids))
        or any(segment_id not in segments for segment_id in selected_segment_ids)
    ):
        raise V5FeatureConfigurationError(
            "The captured feature selection has invalid segment identities.",
            code="segment_selection_invalid",
        )
    contextual = contextual_switches or {}
    effective: dict[str, EffectiveSegmentFeatures] = {}
    for segment_id in selected_segment_ids:
        segment = segments[segment_id]
        backend = backend_by_family[segment.mode]
        template = (
            V5_TEMPLATE_BUNDLE.segment_templates.standard
            if backend == "standard"
            else V5_TEMPLATE_BUNDLE.segment_templates.raylight
        )
        entries = {entry.id: entry for entry in template.entries}
        segment_config = configuration.by_segment.get(segment_id, {})

        for feature_id, selection in (
            *configuration.project.items(),
            *segment_config.items(),
        ):
            if feature_id not in entries and selection.enabled:
                raise V5FeatureConfigurationError(
                    "An active feature does not support this execution backend.",
                    code="feature_backend_unsupported",
                    feature_id=feature_id,
                    segment_id=segment_id,
                    backend=backend,
                )

        resolved: list[EffectiveFeatureSelection] = []
        for entry in template.entries:
            definition = _DEFINITIONS[entry.id]
            selection, source = _route_selection(
                definition,
                project=configuration.project,
                segment=segment_config,
                segment_id=segment_id,
            )
            active = selection.enabled
            effective_source: Literal[
                "template_default", "project", "segment", "context"
            ] = source
            if entry.id == "lora" and active:
                lora = LoraFeatureParams.model_validate(selection.params)
                active = lora.by_family[segment.mode].enabled
            if entry.id in {"continuity"}:
                active = entry.id in contextual.get(segment_id, frozenset())
                effective_source = "context"
            if entry.mode == "needed":
                active = True
            resolved.append(
                EffectiveFeatureSelection(
                    id=entry.id,
                    version=entry.version,
                    mode=entry.mode,
                    selection_enabled=selection.enabled,
                    active=active,
                    params=selection.params,
                    source=effective_source,
                )
            )
        active_by_id = {item.id: item.active for item in resolved}
        for entry in template.entries:
            if not active_by_id[entry.id]:
                continue
            for required_id in entry.requires:
                if not active_by_id.get(required_id, False):
                    raise V5FeatureConfigurationError(
                        "An active feature requires another inactive feature.",
                        code="feature_requirement_unsatisfied",
                        feature_id=entry.id,
                        segment_id=segment_id,
                        backend=backend,
                        safe_details={"required_feature_id": required_id},
                    )
            for conflicting_id in entry.conflicts:
                if active_by_id.get(conflicting_id, False):
                    raise V5FeatureConfigurationError(
                        "Two active features conflict in the selected template.",
                        code="feature_conflict",
                        feature_id=entry.id,
                        segment_id=segment_id,
                        backend=backend,
                        safe_details={"conflicting_feature_id": conflicting_id},
                    )
        effective[segment_id] = EffectiveSegmentFeatures(
            segment_id=segment_id,
            backend=backend,
            family=segment.mode,
            template_id=template.id,
            features=tuple(resolved),
        )
    return EffectiveFeatureConfiguration(
        source_template_bundle_version=(
            migrated.source_template_bundle_version
        ),
        migrated_from_bundle_4=(
            migrated.source_template_bundle_version == 4
        ),
        effective_by_segment=effective,
    )


def feature_parameter_model(feature_id: str, version: int) -> FeatureParamModel:
    """Exact registry hook used when a concrete interpreter is installed."""

    definition = _DEFINITIONS.get(feature_id)
    if definition is None or definition.version != version:
        raise KeyError(f"unknown bundle-5 feature: {feature_id}@{version}")
    return definition.params_model


__all__ = [
    "AttentionBackendOverrideParams",
    "EffectiveFeatureConfiguration",
    "EffectiveFeatureSelection",
    "EffectiveSegmentFeatures",
    "EmptyFeatureParams",
    "MigratedFeatureConfiguration",
    "RayLightPoolIntentParams",
    "V5FeatureConfigurationError",
    "feature_parameter_model",
    "migrate_feature_configuration_to_v5",
    "migrate_timeline_feature_authority_to_v5",
    "resolve_v5_effective_features",
]
