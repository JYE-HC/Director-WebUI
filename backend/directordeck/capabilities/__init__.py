"""Public Stage-5 host capability, catalog, and preflight interfaces."""

from .catalog import (
    CatalogAdapterOption,
    CatalogAvailability,
    FeatureCatalog,
    FeatureCatalogEntry,
    build_feature_catalog,
    feature_catalog_etag,
    quote_feature_catalog_etag,
)
from .evaluator import (
    CapabilityEvaluation,
    CapabilityEvaluator,
    CapabilityReason,
    STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE,
    STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE,
    STRICT_H3_SAGE_RUNTIME_PROBE,
    contextual_runtime_capability_id,
    resolution_adapter_fingerprint,
    runtime_probe_key,
)
from .preflight import (
    EffectiveFeaturePreflight,
    EffectiveSegmentPreflight,
    FeaturePreflightReport,
    preflight_projected_v5_timeline,
    preflight_v4_timeline,
)
from .snapshot import (
    CapturedHostCapabilities,
    build_operational_readiness,
    capture_host_capabilities,
    host_capability_revision,
)

__all__ = [
    "CapabilityEvaluation",
    "CapabilityEvaluator",
    "CapabilityReason",
    "CapturedHostCapabilities",
    "CatalogAdapterOption",
    "CatalogAvailability",
    "EffectiveFeaturePreflight",
    "EffectiveSegmentPreflight",
    "FeatureCatalog",
    "FeatureCatalogEntry",
    "FeaturePreflightReport",
    "STRICT_ATTENTION_CK_INT8_RUNTIME_PROBE",
    "STRICT_ATTENTION_PYTORCH_RUNTIME_PROBE",
    "STRICT_H3_SAGE_RUNTIME_PROBE",
    "build_feature_catalog",
    "build_operational_readiness",
    "capture_host_capabilities",
    "contextual_runtime_capability_id",
    "feature_catalog_etag",
    "host_capability_revision",
    "preflight_v4_timeline",
    "preflight_projected_v5_timeline",
    "quote_feature_catalog_etag",
    "resolution_adapter_fingerprint",
    "runtime_probe_key",
]
