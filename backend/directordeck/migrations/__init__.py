"""Versioned, fail-closed migrations for Director's persisted authorities."""

from .feature_bundle_v4_v5 import (
    FEATURE_BUNDLE_V4_V5_MIGRATION_VERSION,
    FeatureBundleMigrationConflict,
    migrate_feature_bundle_v4_authorities_to_v5,
)
from .feature_bundle_v5_v6 import (
    FEATURE_BUNDLE_V5_V6_MIGRATION_VERSION,
    FeatureBundleV5V6MigrationOutcome,
    feature_bundle_migration_notice_prefix,
    migrate_feature_bundle_v5_authorities_to_v6,
)

from .runtime_settings_v2_v3 import (
    RUNTIME_SETTINGS_V2_V3_MIGRATION_VERSION,
    ensure_runtime_settings_migration_notice_schema,
    migrate_runtime_settings_v2_to_v3,
)

from .timeline_v4_v5 import (
    MIGRATION_IMPLEMENTATION_VERSION,
    LegacyCreativeBindingContext,
    ProjectMigrationReceipt,
    RuntimeSettingsSchemaMigrated,
    TimelineSchemaMigrated,
    WorkflowMigrationConflict,
    legacy_client_timeline_v4_projection,
    legacy_client_timeline_v5_projection,
    legacy_creative_binding_context,
    migrate_runtime_settings_v1_to_v2,
    migrate_timeline_v4_to_v5,
    migrate_timeline_v4_with_context,
    migrate_v4_authorities_to_v5,
)

__all__ = [
    "FEATURE_BUNDLE_V4_V5_MIGRATION_VERSION",
    "FEATURE_BUNDLE_V5_V6_MIGRATION_VERSION",
    "MIGRATION_IMPLEMENTATION_VERSION",
    "RUNTIME_SETTINGS_V2_V3_MIGRATION_VERSION",
    "FeatureBundleMigrationConflict",
    "FeatureBundleV5V6MigrationOutcome",
    "LegacyCreativeBindingContext",
    "ProjectMigrationReceipt",
    "RuntimeSettingsSchemaMigrated",
    "TimelineSchemaMigrated",
    "WorkflowMigrationConflict",
    "ensure_runtime_settings_migration_notice_schema",
    "legacy_client_timeline_v4_projection",
    "legacy_client_timeline_v5_projection",
    "legacy_creative_binding_context",
    "feature_bundle_migration_notice_prefix",
    "migrate_runtime_settings_v1_to_v2",
    "migrate_runtime_settings_v2_to_v3",
    "migrate_feature_bundle_v4_authorities_to_v5",
    "migrate_feature_bundle_v5_authorities_to_v6",
    "migrate_timeline_v4_to_v5",
    "migrate_timeline_v4_with_context",
    "migrate_v4_authorities_to_v5",
]
