from __future__ import annotations

"""Atomic v4 creative-authority to v5 project-authority migration.

The pure conversion functions in this module are shared by database startup,
import preflight, and browser-WAL recovery.  The SQLite entry point validates
every source and destination document before changing any authority, then
publishes projects, the default timeline, receipts, and runtime settings in one
``BEGIN IMMEDIATE`` transaction.
"""

import json
import sqlite3
import uuid
from typing import Annotated, Any, Literal

from pydantic import Field, ValidationError, model_validator

from ..schemas import (
    AssetReference,
    FeatureSelection,
    LegacyLoraResolutionCompat,
    LegacyStandardLoraOverrideEvidence,
    LoraFamilyFeatureSelection,
    LoraFeatureParams,
    MAX_TIMELINE_REVISION,
    ModelStack,
    RuntimeDiffusionPlacement,
    RuntimePlacementV2,
    RuntimeSettingsV1,
    RuntimeSettingsV2,
    RuntimeSettingsV3,
    SlottedAssetReference,
    StrictModel,
    UnifiedTimelineDraftV4,
    UnifiedTimelineDraftV5,
)
from ..workflow.execution import (
    DocumentDigest,
    legacy_fnv1a32_document_digest,
    sha256_document_digest,
)


MIGRATION_IMPLEMENTATION_VERSION = "timeline-v4-v5@1"
MIGRATION_RECEIPT_SCHEMA_VERSION = 1
_MIGRATION_NAMESPACE = uuid.UUID("ac6bb70a-19dc-57f8-bdc6-03d510903d12")


def _client_video_metadata_projection(asset: AssetReference) -> dict[str, Any] | None:
    metadata = asset.metadata
    if metadata is None:
        return None
    return {
        "duration": metadata.duration,
        "native_fps": metadata.native_fps,
        "frame_count": metadata.frame_count,
        "width": metadata.width,
        "height": metadata.height,
        "probe_method": metadata.probe_method,
        "has_audio": metadata.has_audio,
    }


def _legacy_client_asset_projection(
    asset: AssetReference,
    *,
    v5_complete_wire: bool,
    legacy_raw: Any = None,
) -> dict[str, Any]:
    """Reproduce the frozen browser normalizer's insertion order exactly.

    The legacy FNV receipt digest is intentionally JSON.stringify/order
    sensitive.  Pydantic's inheritance/field order is a different wire order,
    so hashing ``model_dump`` directly makes an otherwise identical WAL look
    conflicted.  V4 retains only optional keys present in the frozen source;
    V5 uses the complete FastAPI AssetReference response shape.
    """

    result: dict[str, Any] = {
        "id": asset.id,
        "name": asset.name,
        "subfolder": asset.subfolder,
        "type": asset.type,
        "kind": asset.kind,
    }
    if v5_complete_wire:
        result.update(
            filename=asset.filename,
            path=asset.path,
            preview_url=asset.preview_url,
            content_hash=asset.content_hash,
            metadata=_client_video_metadata_projection(asset),
        )
    else:
        raw = legacy_raw if isinstance(legacy_raw, dict) else {}
        for key in ("filename", "path", "preview_url"):
            if key in raw and (raw[key] is None or isinstance(raw[key], str)):
                result[key] = getattr(asset, key)
        if asset.kind == "video":
            result["metadata"] = _client_video_metadata_projection(asset)
    if isinstance(asset, SlottedAssetReference):
        result["slot"] = asset.slot
    return result


def _legacy_client_sampling_projection(sampling: Any) -> dict[str, Any]:
    return {
        "steps": sampling.steps,
        "seed": sampling.seed,
        "random_seed": sampling.random_seed,
        "sampler": sampling.sampler,
        "scheduler": sampling.scheduler,
        "shift": sampling.shift,
        "audio_shift": sampling.audio_shift,
    }


def _legacy_client_segment_projection(
    segment: Any,
    *,
    v5_complete_wire: bool,
    legacy_raw: Any = None,
) -> dict[str, Any]:
    raw = legacy_raw if isinstance(legacy_raw, dict) else {}
    result: dict[str, Any] = {
        "id": segment.id,
        "title": segment.title,
        "duration_seconds": segment.duration_seconds,
        "prompt": segment.prompt,
        "enabled": segment.enabled,
        "continuity": {
            "enabled": segment.continuity.enabled,
            "overlap_frames": segment.continuity.overlap_frames,
        },
        "ref_image_size": segment.ref_image_size,
        "audio_mode": segment.audio_mode,
        "mode": segment.mode,
    }
    if segment.mode == "fl2va":
        result["first_image"] = (
            _legacy_client_asset_projection(
                segment.first_image,
                v5_complete_wire=v5_complete_wire,
                legacy_raw=raw.get("first_image"),
            )
            if segment.first_image is not None
            else None
        )
        result["last_image"] = (
            _legacy_client_asset_projection(
                segment.last_image,
                v5_complete_wire=v5_complete_wire,
                legacy_raw=raw.get("last_image"),
            )
            if segment.last_image is not None
            else None
        )
        return result
    result["source_video"] = (
        _legacy_client_asset_projection(
            segment.source_video,
            v5_complete_wire=v5_complete_wire,
            legacy_raw=raw.get("source_video"),
        )
        if segment.source_video is not None
        else None
    )
    result["source_start_seconds"] = segment.source_start_seconds
    result["source_duration_seconds"] = segment.source_duration_seconds
    result["source_audio_as_reference"] = segment.source_audio_as_reference
    for key in ("reference_images", "reference_audios", "reference_videos"):
        raw_assets = raw.get(key) if isinstance(raw.get(key), list) else []
        result[key] = [
            _legacy_client_asset_projection(
                asset,
                v5_complete_wire=v5_complete_wire,
                legacy_raw=(raw_assets[index] if index < len(raw_assets) else None),
            )
            for index, asset in enumerate(getattr(segment, key))
        ]
    return result


def legacy_client_timeline_v4_projection(
    document: UnifiedTimelineDraftV4,
    *,
    source_raw: Any = None,
) -> dict[str, Any]:
    """Frozen TS v4 normalization used only by migration receipt digests."""

    raw_segments = (
        source_raw.get("segments")
        if isinstance(source_raw, dict) and isinstance(source_raw.get("segments"), list)
        else []
    )
    return {
        "version": 4,
        "title": document.title,
        "render": {
            "width": document.render.width,
            "height": document.render.height,
            "fps": document.render.fps,
        },
        "sampling": {
            "fl2va": _legacy_client_sampling_projection(document.sampling.fl2va),
            "ref2va": _legacy_client_sampling_projection(document.sampling.ref2va),
        },
        "export_mode": document.export_mode,
        "segments": [
            _legacy_client_segment_projection(
                segment,
                v5_complete_wire=False,
                legacy_raw=(raw_segments[index] if index < len(raw_segments) else None),
            )
            for index, segment in enumerate(document.segments)
        ],
    }


def legacy_client_timeline_v5_projection(
    document: UnifiedTimelineDraftV5,
) -> dict[str, Any]:
    """Current TS strict normalization used only by receipt FNV compatibility."""

    return {
        "version": 5,
        "title": document.title,
        "render": {
            "width": document.render.width,
            "height": document.render.height,
            "fps": document.render.fps,
        },
        "sampling": {
            "fl2va": _legacy_client_sampling_projection(document.sampling.fl2va),
            "ref2va": _legacy_client_sampling_projection(document.sampling.ref2va),
        },
        "export_mode": document.export_mode,
        "model_stack": document.model_stack.model_dump(mode="json"),
        "features": document.features.model_dump(mode="json"),
        "segments": [
            _legacy_client_segment_projection(
                segment,
                v5_complete_wire=True,
            )
            for segment in document.segments
        ],
    }


class WorkflowMigrationConflict(RuntimeError):
    """Persisted authorities cannot be migrated without guessing or overwrite."""

    code = "workflow_migration_conflict"

    def __init__(self, reason: str, *, project_id: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.project_id = project_id


class TimelineSchemaMigrated(RuntimeError):
    """A stale v4 writer attempted to replace a migrated project authority."""

    code = "timeline_schema_migrated"

    def __init__(self, project_id: str, migration_id: str | None) -> None:
        super().__init__(f"timeline schema for {project_id} has migrated to v5")
        self.project_id = project_id
        self.migration_id = migration_id
        self.current_schema = 5


class RuntimeSettingsSchemaMigrated(RuntimeError):
    """A stale settings writer attempted to replace current v3 authority."""

    code = "runtime_settings_schema_migrated"

    def __init__(self) -> None:
        super().__init__("runtime settings schema has migrated to v3")
        self.current_schema = 3


class LegacyCreativeBindingContext(StrictModel):
    """Exact normalized settings-owned creative input consumed by v4→v5."""

    schema_version: Literal[1] = 1
    model_stack: ModelStack
    lora: FeatureSelection
    explicit_standard_lora_overrides: Annotated[
        list[LegacyStandardLoraOverrideEvidence], Field(max_length=2)
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def stable_override_order(self) -> "LegacyCreativeBindingContext":
        families = [record.family for record in self.explicit_standard_lora_overrides]
        if families != sorted(families) or len(families) != len(set(families)):
            raise ValueError("legacy creative overrides must be unique and sorted")
        return self


class ProjectMigrationReceipt(StrictModel):
    schema_version: Literal[1] = 1
    migration_id: Annotated[str, Field(min_length=1, max_length=128)]
    project_id: Annotated[str, Field(min_length=1, max_length=128)]
    from_schema: Literal[4] = 4
    to_schema: Literal[5] = 5
    old_revision: Annotated[int, Field(ge=0, le=MAX_TIMELINE_REVISION)]
    old_client_digest: DocumentDigest
    old_server_digest: DocumentDigest
    new_revision: Annotated[int, Field(ge=1, le=MAX_TIMELINE_REVISION)]
    new_client_digest: DocumentDigest
    new_server_digest: DocumentDigest
    legacy_creative_binding_context: LegacyCreativeBindingContext
    legacy_binding_digest: DocumentDigest
    migration_implementation_version: Literal[
        "timeline-v4-v5@1"
    ] = MIGRATION_IMPLEMENTATION_VERSION
    created_at: Annotated[str, Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def validate_receipt_contract(self) -> "ProjectMigrationReceipt":
        if self.new_revision != self.old_revision + 1:
            raise ValueError("migration receipt must advance revision exactly once")
        if self.old_client_digest.algorithm != "fnv1a32-json-stringify-v1":
            raise ValueError("old client digest must use the legacy FNV algorithm")
        if self.new_client_digest.algorithm != "fnv1a32-json-stringify-v1":
            raise ValueError("new client digest must use the legacy FNV algorithm")
        for digest in (
            self.old_server_digest,
            self.new_server_digest,
            self.legacy_binding_digest,
        ):
            if digest.algorithm != "sha256-canonical-json-v1":
                raise ValueError("server and binding digests must use canonical SHA-256")
        expected_binding = sha256_document_digest(
            self.legacy_creative_binding_context.model_dump(mode="json")
        )
        if expected_binding != self.legacy_binding_digest:
            raise ValueError("legacy creative binding digest is inconsistent")
        return self


def legacy_creative_binding_context(
    settings: RuntimeSettingsV1,
) -> LegacyCreativeBindingContext:
    model_stack = ModelStack.model_validate(
        {
            "fl2va": {"filename": settings.models.fl2va.filename},
            "ref2va": {"filename": settings.models.ref2va.filename},
            "clip": {"filename": settings.models.clip.filename},
            "video_vae": {"filename": settings.models.video_vae.filename},
            "audio_vae": {"filename": settings.models.audio_vae.filename},
        }
    )
    family_values = {
        family: LoraFamilyFeatureSelection(
            enabled=binding.lora_name is not None,
            filename=binding.lora_name,
            strength=binding.lora_strength,
        )
        for family in ("fl2va", "ref2va")
        for binding in (getattr(settings.models, family),)
    }
    lora_params = LoraFeatureParams(by_family=family_values)
    lora = FeatureSelection(
        enabled=any(value.enabled for value in family_values.values()),
        params=lora_params.model_dump(mode="json"),
    )
    overrides: list[LegacyStandardLoraOverrideEvidence] = []
    for family in ("fl2va", "ref2va"):
        binding = getattr(settings.models, family)
        override = binding.standard_lora_loader_override
        if override is None:
            continue
        overrides.append(
            LegacyStandardLoraOverrideEvidence(
                family=family,
                model_filename=override.model_filename,
                lora_filename=override.lora_name,
                loader=override.loader,
            )
        )
    return LegacyCreativeBindingContext(
        model_stack=model_stack,
        lora=lora,
        explicit_standard_lora_overrides=overrides,
    )


def migrate_timeline_v4_with_context(
    timeline: UnifiedTimelineDraftV4,
    context: LegacyCreativeBindingContext,
) -> UnifiedTimelineDraftV5:
    source = timeline.model_dump(mode="json", exclude={"version"})
    return UnifiedTimelineDraftV5.model_validate(
        {
            **source,
            "version": 5,
            "model_stack": context.model_stack.model_dump(mode="json"),
            "features": {
                "template_bundle_version": 4,
                "project": {
                    "lora": context.lora.model_dump(mode="json"),
                },
                "by_segment": {},
            },
        }
    )


def migrate_timeline_v4_to_v5(
    timeline: UnifiedTimelineDraftV4,
    settings: RuntimeSettingsV1,
) -> UnifiedTimelineDraftV5:
    return migrate_timeline_v4_with_context(
        timeline,
        legacy_creative_binding_context(settings),
    )


def migrate_runtime_settings_v1_to_v2(
    settings: RuntimeSettingsV1,
) -> RuntimeSettingsV2:
    context = legacy_creative_binding_context(settings)
    compat = LegacyLoraResolutionCompat(
        explicit_overrides=context.explicit_standard_lora_overrides
    )
    return RuntimeSettingsV2(
        schema_version=2,
        client_id=settings.client_id,
        memory_policy=settings.memory_policy,
        raylight_residency_policy=settings.raylight_residency_policy,
        multi_gpu_enabled=settings.multi_gpu_enabled,
        placement=RuntimePlacementV2(
            fl2va=RuntimeDiffusionPlacement(
                device=settings.models.fl2va.device,
                raylight=settings.models.fl2va.raylight,
            ),
            ref2va=RuntimeDiffusionPlacement(
                device=settings.models.ref2va.device,
                raylight=settings.models.ref2va.raylight,
            ),
            clip_device=settings.models.clip.device,
            video_vae_device=settings.models.video_vae.device,
            audio_vae_device=settings.models.audio_vae.device,
        ),
        legacy_lora_resolution_compat=compat,
    )


def _receipt_json(receipt: ProjectMigrationReceipt) -> str:
    return json.dumps(
        receipt.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _ensure_receipt_schema(db: sqlite3.Connection) -> None:
    # Deliberately avoid ``executescript``: sqlite3 commits any pending
    # transaction before that helper, which would split the cross-document
    # migration after ``BEGIN IMMEDIATE``.
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS project_migration_receipts (
            migration_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            from_schema INTEGER NOT NULL CHECK (from_schema >= 1),
            to_schema INTEGER NOT NULL CHECK (to_schema > from_schema),
            receipt TEXT NOT NULL,
            receipt_digest TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(project_id, from_schema, to_schema)
        )
        """
    )
    db.execute(
        """
        CREATE INDEX IF NOT EXISTS project_migration_receipts_project_idx
            ON project_migration_receipts(
                project_id, from_schema, to_schema, created_at DESC
            )
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS project_migration_receipts_immutable_update
        BEFORE UPDATE ON project_migration_receipts
        BEGIN
            SELECT RAISE(ABORT, 'project migration receipts are immutable');
        END
        """
    )
    db.execute(
        """
        CREATE TRIGGER IF NOT EXISTS project_migration_receipts_immutable_delete
        BEFORE DELETE ON project_migration_receipts
        BEGIN
            SELECT RAISE(ABORT, 'project migration receipts are immutable');
        END
        """
    )


def decode_project_migration_receipt(
    row: sqlite3.Row,
) -> ProjectMigrationReceipt:
    try:
        raw = json.loads(str(row["receipt"]))
        receipt = ProjectMigrationReceipt.model_validate(raw)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise WorkflowMigrationConflict("migration receipt is malformed") from exc
    actual_digest = sha256_document_digest(receipt.model_dump(mode="json")).value
    if actual_digest != str(row["receipt_digest"]):
        raise WorkflowMigrationConflict("migration receipt digest is inconsistent")
    if (
        receipt.migration_id != str(row["migration_id"])
        or receipt.project_id != str(row["project_id"])
        or receipt.from_schema != int(row["from_schema"])
        or receipt.to_schema != int(row["to_schema"])
    ):
        raise WorkflowMigrationConflict("migration receipt index is inconsistent")
    return receipt


def _project_authorities(
    db: sqlite3.Connection,
) -> list[tuple[str, str, int, str]]:
    rows: list[tuple[str, str, int, str]] = []
    default = db.execute(
        "SELECT document, revision FROM unified_timeline WHERE singleton = 1"
    ).fetchone()
    if default is None:
        raise WorkflowMigrationConflict("default timeline authority is missing")
    rows.append(("timeline", "default", int(default["revision"]), str(default["document"])))
    rows.extend(
        (
            "project",
            str(row["id"]),
            int(row["revision"]),
            str(row["document"]),
        )
        for row in db.execute(
            "SELECT id, document, revision FROM projects ORDER BY id"
        ).fetchall()
    )
    return rows


def _receipt_by_project(
    db: sqlite3.Connection,
) -> dict[str, ProjectMigrationReceipt]:
    receipts: dict[str, ProjectMigrationReceipt] = {}
    for row in db.execute(
        "SELECT migration_id, project_id, from_schema, to_schema, receipt, "
        "receipt_digest, created_at FROM project_migration_receipts "
        "WHERE from_schema = 4 AND to_schema = 5 ORDER BY project_id"
    ).fetchall():
        receipt = decode_project_migration_receipt(row)
        if receipt.project_id in receipts:
            raise WorkflowMigrationConflict(
                "multiple v4 to v5 receipts exist for one project",
                project_id=receipt.project_id,
            )
        receipts[receipt.project_id] = receipt
    return receipts


def _verify_migrated_authority(
    *,
    project_id: str,
    revision: int,
    document: dict[str, Any],
    receipt: ProjectMigrationReceipt,
) -> None:
    if revision < receipt.new_revision:
        raise WorkflowMigrationConflict(
            "migrated project revision moved backwards",
            project_id=project_id,
        )
    if revision == receipt.new_revision:
        current_digest = sha256_document_digest(document)
        if current_digest != receipt.new_server_digest:
            raise WorkflowMigrationConflict(
                "migrated project document does not match its receipt",
                project_id=project_id,
            )


def migrate_v4_authorities_to_v5(
    db: sqlite3.Connection,
    *,
    created_at: str,
) -> list[ProjectMigrationReceipt]:
    """Migrate every live authority and settings document in one transaction."""

    if db.in_transaction:
        raise RuntimeError("v4 to v5 migration requires an idle connection")
    db.execute("BEGIN IMMEDIATE")
    try:
        _ensure_receipt_schema(db)
        settings_row = db.execute(
            "SELECT document, revision FROM settings WHERE singleton = 1"
        ).fetchone()
        if settings_row is None:
            raise WorkflowMigrationConflict("runtime settings authority is missing")
        try:
            raw_settings = json.loads(str(settings_row["document"]))
        except (json.JSONDecodeError, TypeError) as exc:
            raise WorkflowMigrationConflict("runtime settings document is malformed") from exc
        authorities = _project_authorities(db)
        receipts = _receipt_by_project(db)

        if isinstance(raw_settings, dict) and raw_settings.get(
            "schema_version"
        ) in {2, 3}:
            try:
                if raw_settings.get("schema_version") == 2:
                    RuntimeSettingsV2.model_validate(raw_settings)
                else:
                    RuntimeSettingsV3.model_validate(raw_settings)
            except ValidationError as exc:
                raise WorkflowMigrationConflict(
                    "migrated runtime settings document is invalid"
                ) from exc
            for _kind, project_id, revision, raw_document in authorities:
                try:
                    document = json.loads(raw_document)
                except (json.JSONDecodeError, TypeError) as exc:
                    raise WorkflowMigrationConflict(
                        "project document is malformed", project_id=project_id
                    ) from exc
                if not isinstance(document, dict) or document.get("version") != 5:
                    raise WorkflowMigrationConflict(
                        "runtime settings migrated before every project",
                        project_id=project_id,
                    )
                try:
                    validated = UnifiedTimelineDraftV5.model_validate(document)
                except ValidationError as exc:
                    raise WorkflowMigrationConflict(
                        "project v5 document is invalid", project_id=project_id
                    ) from exc
                receipt = receipts.get(project_id)
                if receipt is not None:
                    _verify_migrated_authority(
                        project_id=project_id,
                        revision=revision,
                        document=validated.model_dump(mode="json"),
                        receipt=receipt,
                    )
            db.commit()
            return []

        if isinstance(raw_settings, dict) and "schema_version" in raw_settings:
            if raw_settings.get("schema_version") == 1:
                raw_settings = dict(raw_settings)
                raw_settings.pop("schema_version", None)
            else:
                raise WorkflowMigrationConflict("unknown runtime settings schema")
        try:
            legacy_settings = RuntimeSettingsV1.model_validate(raw_settings)
        except ValidationError as exc:
            raise WorkflowMigrationConflict("runtime settings v1 document is invalid") from exc
        context = legacy_creative_binding_context(legacy_settings)
        context_digest = sha256_document_digest(context.model_dump(mode="json"))
        created: list[ProjectMigrationReceipt] = []
        for kind, project_id, old_revision, raw_document in authorities:
            if project_id in receipts:
                raise WorkflowMigrationConflict(
                    "receipt exists while its project is still v4",
                    project_id=project_id,
                )
            if old_revision >= MAX_TIMELINE_REVISION:
                raise WorkflowMigrationConflict(
                    "project revision is exhausted", project_id=project_id
                )
            try:
                source_raw = json.loads(raw_document)
                source = UnifiedTimelineDraftV4.model_validate(source_raw)
            except (json.JSONDecodeError, TypeError, ValidationError) as exc:
                raise WorkflowMigrationConflict(
                    "project is not a valid v4 migration source",
                    project_id=project_id,
                ) from exc
            destination = migrate_timeline_v4_with_context(source, context)
            destination_document = destination.model_dump(mode="json")
            legacy_client_source = legacy_client_timeline_v4_projection(
                source,
                source_raw=source_raw,
            )
            legacy_client_destination = legacy_client_timeline_v5_projection(
                destination
            )
            new_revision = old_revision + 1
            # The old authority was authored by the frozen v4 browser shape.
            # Canonical SHA still preserves the semantic distinction between
            # an absent optional key and an explicit null, so use that exact
            # normalized source for both legacy receipt domains.
            old_server_digest = sha256_document_digest(legacy_client_source)
            migration_id = str(
                uuid.uuid5(
                    _MIGRATION_NAMESPACE,
                    f"{project_id}\0{old_revision}\0{old_server_digest.value}",
                )
            )
            receipt = ProjectMigrationReceipt(
                migration_id=migration_id,
                project_id=project_id,
                old_revision=old_revision,
                old_client_digest=legacy_fnv1a32_document_digest(
                    legacy_client_source
                ),
                old_server_digest=old_server_digest,
                new_revision=new_revision,
                new_client_digest=legacy_fnv1a32_document_digest(
                    legacy_client_destination
                ),
                new_server_digest=sha256_document_digest(destination_document),
                legacy_creative_binding_context=context,
                legacy_binding_digest=context_digest,
                created_at=created_at,
            )
            serialized = json.dumps(
                destination_document,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if kind == "timeline":
                cursor = db.execute(
                    "UPDATE unified_timeline SET document = ?, updated_at = ?, "
                    "revision = revision + 1 WHERE singleton = 1 AND revision = ?",
                    (serialized, created_at, old_revision),
                )
            else:
                cursor = db.execute(
                    "UPDATE projects SET document = ?, title = ?, updated_at = ?, "
                    "revision = revision + 1 WHERE id = ? AND revision = ?",
                    (
                        serialized,
                        destination.title,
                        created_at,
                        project_id,
                        old_revision,
                    ),
                )
            if cursor.rowcount != 1:
                raise WorkflowMigrationConflict(
                    "project migration CAS failed", project_id=project_id
                )
            receipt_json = _receipt_json(receipt)
            db.execute(
                "INSERT INTO project_migration_receipts("
                "migration_id, project_id, from_schema, to_schema, receipt, "
                "receipt_digest, created_at) VALUES(?, ?, 4, 5, ?, ?, ?)",
                (
                    receipt.migration_id,
                    project_id,
                    receipt_json,
                    sha256_document_digest(
                        receipt.model_dump(mode="json")
                    ).value,
                    created_at,
                ),
            )
            created.append(receipt)

        migrated_settings = migrate_runtime_settings_v1_to_v2(legacy_settings)
        cursor = db.execute(
            "UPDATE settings SET document = ?, updated_at = ?, "
            "revision = revision + 1 WHERE singleton = 1 AND revision = ?",
            (
                migrated_settings.model_dump_json(),
                created_at,
                int(settings_row["revision"]),
            ),
        )
        if cursor.rowcount != 1:
            raise WorkflowMigrationConflict("runtime settings migration CAS failed")
        db.commit()
        return created
    except BaseException:
        db.rollback()
        raise


__all__ = [
    "MIGRATION_IMPLEMENTATION_VERSION",
    "LegacyCreativeBindingContext",
    "ProjectMigrationReceipt",
    "RuntimeSettingsSchemaMigrated",
    "TimelineSchemaMigrated",
    "WorkflowMigrationConflict",
    "decode_project_migration_receipt",
    "legacy_client_timeline_v4_projection",
    "legacy_client_timeline_v5_projection",
    "legacy_creative_binding_context",
    "migrate_runtime_settings_v1_to_v2",
    "migrate_timeline_v4_to_v5",
    "migrate_timeline_v4_with_context",
    "migrate_v4_authorities_to_v5",
]
