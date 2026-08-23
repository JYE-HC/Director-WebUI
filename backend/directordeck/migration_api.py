from __future__ import annotations

"""Fail-closed API helpers for the Stage-6 creative-authority cut-over.

This module deliberately has no database or ComfyUI dependency.  Historical
resolution consumes only immutable job bytes supplied by its caller, while the
short-lived import coordinator owns only an opaque, process-local capability
token.  The actual project insert remains one database transaction.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets
import threading
import time
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator

from .migrations.timeline_v4_v5 import (
    LegacyCreativeBindingContext,
    migrate_timeline_v4_with_context,
)

from .schemas import (
    FeatureSelection,
    LoraFeatureParams,
    ModelStack,
    RuntimeSettingsV1,
    StrictModel,
    UnifiedTimelineDraftV4,
    UnifiedTimelineDraftV5,
)
from .workflow.canonical import canonical_json_bytes
from .workflow.effective_features import (
    V5FeatureConfigurationError,
    migrate_timeline_feature_authority_to_v5,
)
from .workflow.execution import DocumentDigest, sha256_document_digest


_MAX_PENDING_IMPORTS = 256


class ProjectImportCreativeSelection(StrictModel):
    """Explicit creative authority supplied after a context-free v4 preflight."""

    model_stack: ModelStack
    lora: FeatureSelection

    @model_validator(mode="after")
    def validate_lora_contract(self) -> "ProjectImportCreativeSelection":
        LoraFeatureParams.model_validate(self.lora.params)
        return self


class ProjectImportPreflightRequest(StrictModel):
    title: Annotated[str, Field(min_length=0, max_length=256)] = ""
    document: dict[str, Any]
    # A standalone v4 file has no creative model authority of its own.  A
    # validated, file-supplied v1 snapshot is the only deterministic migration
    # context accepted here; live settings are never substituted.
    legacy_runtime_settings: RuntimeSettingsV1 | None = None
    # New exports may carry the smaller exact context directly.  A standalone
    # legacy file instead comes back through preflight with an explicit model
    # stack/LoRA selection supplied by the user.  These paths are intentionally
    # mutually exclusive so the server never has to choose an authority.
    legacy_creative_context: LegacyCreativeBindingContext | None = None
    creative_selection: ProjectImportCreativeSelection | None = None

    @model_validator(mode="after")
    def one_legacy_creative_authority(self) -> "ProjectImportPreflightRequest":
        supplied = sum(
            value is not None
            for value in (
                self.legacy_runtime_settings,
                self.legacy_creative_context,
                self.creative_selection,
            )
        )
        # A declared v5 document is rejected with the stable import API error
        # in ``prepare_project_import`` below.  Let every legacy field reach
        # that boundary, including a payload carrying more than one, rather
        # than leaking a generic request-validation error first.  The frozen
        # v4 ambiguity rule remains unchanged.
        if self.document.get("version") != 5 and supplied > 1:
            raise ValueError(
                "legacy import accepts exactly one creative context or selection"
            )
        return self


class ProjectImportPreflightRead(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["ready", "needs_input"]
    input_digest: DocumentDigest
    proposed_document: UnifiedTimelineDraftV5 | None = None
    missing_context: list[str] = Field(default_factory=list)
    missing_model_bindings: list[str] = Field(default_factory=list)
    capability_issues: list[dict[str, Any]] = Field(default_factory=list)
    commit_token: str | None = None
    expires_at: str | None = None


class ProjectImportCommitRequest(StrictModel):
    commit_token: Annotated[str, Field(min_length=32, max_length=256)]
    input_digest: DocumentDigest


class HistoricalSaveAsProjectRequest(StrictModel):
    title: Annotated[str, Field(min_length=0, max_length=256)] = ""


class ProjectImportError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 422,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = dict(details or {})


class HistoricalCreativeInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


V4ToV5Migration = Callable[
    [UnifiedTimelineDraftV4, RuntimeSettingsV1], UnifiedTimelineDraftV5
]


def document_digest(value: Any) -> DocumentDigest:
    return sha256_document_digest(value)


def project_import_input_digest(
    body: ProjectImportPreflightRequest,
) -> DocumentDigest:
    """Bind the token to every preflight input, not merely the legacy file."""

    return document_digest(
        {
            "schema_version": 1,
            "title": body.title,
            "document": body.document,
            "legacy_runtime_settings": (
                body.legacy_runtime_settings.model_dump(mode="json")
                if body.legacy_runtime_settings is not None
                else None
            ),
            "legacy_creative_context": (
                body.legacy_creative_context.model_dump(mode="json")
                if body.legacy_creative_context is not None
                else None
            ),
            "creative_selection": (
                body.creative_selection.model_dump(mode="json")
                if body.creative_selection is not None
                else None
            ),
        }
    )


def _document_version(document: Mapping[str, Any]) -> int:
    version = document.get("version")
    if type(version) is not int:
        raise ProjectImportError(
            "project_import_schema_required",
            "The imported project must declare an integer schema version.",
        )
    if version not in {4, 5}:
        raise ProjectImportError(
            "project_import_schema_unsupported",
            "The imported project schema has no supported migration path.",
            details={"schema_version": version},
        )
    return version


def _upgrade_current_features(
    document: UnifiedTimelineDraftV5,
) -> UnifiedTimelineDraftV5:
    """Validate bundle 5 or deterministically upgrade a bundle-4 v5 file."""

    try:
        return migrate_timeline_feature_authority_to_v5(document)
    except V5FeatureConfigurationError as exc:
        if exc.code == "template_bundle_version_unsupported":
            raise ProjectImportError(
                "project_import_template_bundle_unsupported",
                "The imported project targets an unsupported workflow template bundle.",
                details={
                    "template_bundle_version": (
                        document.features.template_bundle_version
                    )
                },
            ) from exc
        if exc.code == "unknown_feature":
            raise ProjectImportError(
                "project_import_unknown_feature",
                "The imported project contains a feature with no installed migration path.",
                details={"feature_ids": [exc.feature_id]},
            ) from exc
        details: dict[str, Any] = {"reason_code": exc.code}
        if exc.feature_id is not None:
            details["feature_id"] = exc.feature_id
        if exc.segment_id is not None:
            details["segment_id"] = exc.segment_id
        details.update(exc.safe_details)
        raise ProjectImportError(
            "project_import_feature_configuration_invalid",
            "The imported project feature configuration is invalid.",
            details=details,
        ) from exc


def _missing_model_bindings(document: UnifiedTimelineDraftV5) -> list[str]:
    return [
        name
        for name in ("fl2va", "ref2va", "clip", "video_vae", "audio_vae")
        if getattr(document.model_stack, name).filename is None
    ]


def prepare_project_import(
    body: ProjectImportPreflightRequest,
    *,
    migrate_v4_to_v5: V4ToV5Migration,
) -> tuple[DocumentDigest, UnifiedTimelineDraftV5 | None, list[str], list[str]]:
    """Validate/migrate one file without consulting live project or settings."""

    digest = project_import_input_digest(body)
    version = _document_version(body.document)
    legacy_context_fields = [
        name
        for name, value in (
            ("legacy_runtime_settings", body.legacy_runtime_settings),
            ("legacy_creative_context", body.legacy_creative_context),
            ("creative_selection", body.creative_selection),
        )
        if value is not None
    ]
    if version == 5 and legacy_context_fields:
        raise ProjectImportError(
            "project_import_legacy_context_forbidden",
            "A schema-5 project must not carry legacy creative context.",
            details={
                "schema_version": 5,
                "fields": legacy_context_fields,
            },
        )
    try:
        if version == 5:
            proposed = UnifiedTimelineDraftV5.model_validate(body.document)
        else:
            legacy = UnifiedTimelineDraftV4.model_validate(body.document)
            if (
                body.legacy_runtime_settings is None
                and body.legacy_creative_context is None
                and body.creative_selection is None
            ):
                return (
                    digest,
                    None,
                    ["creative_selection"],
                    ["fl2va", "ref2va", "clip", "video_vae", "audio_vae"],
                )
            if body.legacy_runtime_settings is not None:
                proposed = migrate_v4_to_v5(legacy, body.legacy_runtime_settings)
            else:
                context = body.legacy_creative_context
                if context is None:
                    assert body.creative_selection is not None
                    context = LegacyCreativeBindingContext(
                        model_stack=body.creative_selection.model_stack,
                        lora=body.creative_selection.lora,
                    )
                proposed = migrate_timeline_v4_with_context(legacy, context)
    except ValidationError as exc:
        raise ProjectImportError(
            "project_import_document_invalid",
            "The imported project does not satisfy its declared schema.",
        ) from exc
    proposed = _upgrade_current_features(proposed)
    return digest, proposed, [], _missing_model_bindings(proposed)


@dataclass(frozen=True, slots=True)
class _PendingImport:
    title: str
    input_digest: DocumentDigest
    proposed_document_json: bytes
    proposed_document_digest: DocumentDigest
    expires_at_monotonic: float
    expires_at_utc: str


class ProjectImportCoordinator:
    """Issue bounded, single-use capabilities for a checked import proposal."""

    def __init__(
        self,
        *,
        token_ttl_seconds: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if token_ttl_seconds <= 0 or token_ttl_seconds > 3_600:
            raise ValueError("import token TTL must be in (0, 3600]")
        self._token_ttl_seconds = float(token_ttl_seconds)
        self._monotonic = monotonic
        self._utcnow = utcnow
        self._pending: dict[str, _PendingImport] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _token_key(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _prune_locked(self, now: float) -> None:
        expired = [
            key
            for key, pending in self._pending.items()
            if pending.expires_at_monotonic <= now
        ]
        for key in expired:
            self._pending.pop(key, None)

    def issue(
        self,
        *,
        title: str,
        input_digest: DocumentDigest,
        proposed_document: UnifiedTimelineDraftV5,
        missing_model_bindings: Sequence[str],
        capability_issues: Sequence[Mapping[str, Any]] = (),
    ) -> ProjectImportPreflightRead:
        now = self._monotonic()
        expires_at = self._utcnow() + timedelta(seconds=self._token_ttl_seconds)
        token = secrets.token_urlsafe(32)
        key = self._token_key(token)
        encoded = canonical_json_bytes(proposed_document)
        proposed_digest = document_digest(proposed_document)
        with self._lock:
            self._prune_locked(now)
            if len(self._pending) >= _MAX_PENDING_IMPORTS:
                raise ProjectImportError(
                    "project_import_capacity_exceeded",
                    "Too many project imports are awaiting commit; retry shortly.",
                    status_code=503,
                )
            self._pending[key] = _PendingImport(
                title=title,
                input_digest=input_digest,
                proposed_document_json=encoded,
                proposed_document_digest=proposed_digest,
                expires_at_monotonic=now + self._token_ttl_seconds,
                expires_at_utc=expires_at.isoformat(),
            )
        return ProjectImportPreflightRead(
            status="ready",
            input_digest=input_digest,
            proposed_document=proposed_document,
            missing_model_bindings=list(missing_model_bindings),
            capability_issues=[dict(issue) for issue in capability_issues],
            commit_token=token,
            expires_at=expires_at.isoformat(),
        )

    def consume(
        self, body: ProjectImportCommitRequest
    ) -> tuple[str, UnifiedTimelineDraftV5]:
        now = self._monotonic()
        key = self._token_key(body.commit_token)
        with self._lock:
            self._prune_locked(now)
            pending = self._pending.get(key)
            if pending is None:
                raise ProjectImportError(
                    "project_import_token_invalid",
                    "The project import token is invalid or expired.",
                    status_code=409,
                )
            if (
                pending.input_digest.algorithm != body.input_digest.algorithm
                or not secrets.compare_digest(
                    pending.input_digest.value, body.input_digest.value
                )
            ):
                raise ProjectImportError(
                    "project_import_digest_mismatch",
                    "The project import token does not match this input digest.",
                    status_code=409,
                )
            # Pop while holding the lock.  Concurrent commits can therefore
            # never create two projects from one preflight capability.
            self._pending.pop(key)
        try:
            raw_document = json.loads(pending.proposed_document_json)
            proposed = UnifiedTimelineDraftV5.model_validate(raw_document)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ProjectImportError(
                "project_import_proposal_invalid",
                "The checked project import proposal is no longer valid.",
                status_code=409,
            ) from exc
        if document_digest(proposed) != pending.proposed_document_digest:
            raise ProjectImportError(
                "project_import_proposal_changed",
                "The checked project import proposal changed before commit.",
                status_code=409,
            )
        return pending.title, proposed


def resolve_historical_creative_input(
    job: Mapping[str, Any],
    *,
    migrate_v4_to_v5: V4ToV5Migration,
) -> UnifiedTimelineDraftV5:
    """Resolve a read-only v5 view from one job's immutable snapshots only."""

    snapshot = job.get("config_snapshot")
    timeline = snapshot.get("timeline") if isinstance(snapshot, Mapping) else None
    if not isinstance(timeline, Mapping):
        raise HistoricalCreativeInputError(
            "historical_creative_snapshot_unavailable",
            "This historical task has no compatible creative snapshot.",
        )
    version = timeline.get("version")
    try:
        if version == 5:
            resolved = UnifiedTimelineDraftV5.model_validate(timeline)
        elif version == 4:
            legacy_timeline = UnifiedTimelineDraftV4.model_validate(timeline)
            legacy_settings = RuntimeSettingsV1.model_validate(
                job.get("settings_snapshot")
            )
            resolved = migrate_v4_to_v5(legacy_timeline, legacy_settings)
        else:
            raise HistoricalCreativeInputError(
                "historical_creative_schema_unsupported",
                "This historical task uses an unsupported creative schema.",
            )
    except ValidationError as exc:
        raise HistoricalCreativeInputError(
            "historical_creative_snapshot_invalid",
            "This historical task has invalid immutable creative evidence.",
        ) from exc
    resolved = _upgrade_current_features(resolved)
    # Never return the coordinator's mutable instance to callers that may
    # retitle it for save-as.
    return resolved.model_copy(deep=True)


__all__ = [
    "DocumentDigest",
    "HistoricalCreativeInputError",
    "HistoricalSaveAsProjectRequest",
    "ProjectImportCommitRequest",
    "ProjectImportCoordinator",
    "ProjectImportCreativeSelection",
    "ProjectImportError",
    "ProjectImportPreflightRead",
    "ProjectImportPreflightRequest",
    "document_digest",
    "project_import_input_digest",
    "prepare_project_import",
    "resolve_historical_creative_input",
]
