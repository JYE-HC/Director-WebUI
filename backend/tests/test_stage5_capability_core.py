from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from directordeck.capabilities import (
    CapabilityEvaluator,
    build_feature_catalog,
    build_operational_readiness,
    capture_host_capabilities,
    feature_catalog_etag,
    preflight_v4_timeline,
    quote_feature_catalog_etag,
)
from directordeck.workflow.contracts import (
    CapabilitySet,
    HostCapabilitySnapshot,
    LogicalGpuCapability,
    MediaToolCapability,
    OperationalReadiness,
    PackageCapability,
    RayLightInstallation,
)
from directordeck.workflow.interpreters import (
    V4BuiltinInterpreter,
    V4BuiltinParams,
)
from directordeck.workflow.node_contracts import V4_NODE_CONTRACT_REGISTRY
from directordeck.workflow.v4_compiler import (
    V4CapabilityEvaluationError,
    V4_VALIDATED_TEMPLATES,
    build_v4_route_context,
    compile_v4_timeline,
    resolve_v4_active_feature,
)
from directordeck.workflow.v4_resolver import CreativeCompileInputResolver

from . import extensible_workflow_v0_fixture_builder as fixture_builder


def _fingerprint(seed: str) -> str:
    return "sha256:" + seed * 64


def _snapshot(
    *,
    generated_at: datetime | None = None,
    raylight_installed: bool = True,
    missing_class_type: str | None = None,
    drift_class_type: str | None = None,
    drift_module: str | None = None,
    unavailable_media_tools: tuple[str, ...] = (),
    unavailable_packages: tuple[str, ...] = (),
    gpu_backends: tuple[str, ...] = ("cuda", "cuda", "cuda", "cuda"),
) -> HostCapabilitySnapshot:
    node_registry: dict[str, str] = {}
    object_info: dict[str, object] = {}
    module_fingerprints: dict[str, str] = {}
    for contract in V4_NODE_CONTRACT_REGISTRY.contracts.values():
        if contract.class_type == missing_class_type:
            continue
        module = contract.allowed_python_modules[0]
        node_registry[contract.class_type] = module
        observed = contract.object_info_contract
        if contract.class_type == drift_class_type:
            observed = observed.model_copy(
                update={"output_node": not observed.output_node}
            )
        object_info[contract.class_type] = observed
    if drift_module is not None:
        module_fingerprints[drift_module] = _fingerprint("f")
    raylight_contracts_available = raylight_installed and all(
        contract.class_type != missing_class_type
        for contract in V4_NODE_CONTRACT_REGISTRY.contracts.values()
        if "custom_nodes.DirectorDeck-RayLight" in contract.allowed_python_modules
    )
    return HostCapabilitySnapshot(
        schema_version=1,
        generated_at=generated_at
        or datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc),
        node_registry=node_registry,
        object_info_slices=object_info,
        module_fingerprints=module_fingerprints,
        importable_packages={
            name: PackageCapability(
                importable=name not in unavailable_packages,
                version=None if name in unavailable_packages else "1.0.0",
            )
            for name in ("ray", "xfuser")
        },
        gpu_inventory=tuple(
            LogicalGpuCapability(logical_index=index, backend=backend)
            for index, backend in enumerate(gpu_backends)
        ),
        raylight_installation=RayLightInstallation(
            installed=raylight_installed,
            node_contracts_available=raylight_contracts_available,
            reason_codes=() if raylight_installed else ("not_installed",),
        ),
        media_tool_status={
            name: MediaToolCapability(
                available=name not in unavailable_media_tools,
                version=(
                    None if name in unavailable_media_tools else "7.1"
                ),
            )
            for name in ("ffmpeg", "ffprobe")
        },
    )


def _ready() -> OperationalReadiness:
    return build_operational_readiness(
        endpoint_online=True,
        available_logical_gpu_count=4,
    )


def _with_export_mode(draft, export_mode: str):
    document = draft.model_dump(mode="json")
    document["export_mode"] = export_mode
    return type(draft).model_validate(document)


def _reason_semantics(reason) -> tuple[object, ...]:
    return (
        reason.code,
        reason.rule,
        reason.message,
        reason.remediation,
        dict(reason.safe_details),
    )


def test_capture_revalidates_snapshot_and_binds_static_revision() -> None:
    snapshot = _snapshot()

    class Provider:
        def snapshot(self) -> HostCapabilitySnapshot:
            return snapshot

    captured = capture_host_capabilities(Provider())

    assert captured.snapshot == snapshot
    assert captured.snapshot is not snapshot
    assert captured.host_capability_revision == snapshot.host_capability_revision()


def test_readiness_allows_taint_only_when_locked_cleanup_is_available() -> None:
    repairable = build_operational_readiness(
        endpoint_online=True,
        ray_tainted=True,
        ray_cleanup_available=True,
        available_logical_gpu_count=4,
    )
    blocked = build_operational_readiness(
        endpoint_online=True,
        ray_tainted=True,
        ray_cleanup_available=False,
        available_logical_gpu_count=4,
    )

    assert repairable.submission_allowed is True
    assert repairable.ray_tainted is True
    assert repairable.blocking_reason_codes == ()
    assert blocked.submission_allowed is False
    assert blocked.blocking_reason_codes == ("ray_cleanup_unavailable",)
    assert OperationalReadiness.model_validate_json(
        repairable.model_dump_json()
    ) == repairable


def test_readiness_normalizes_endpoint_recovery_and_gpu_blocks() -> None:
    readiness = build_operational_readiness(
        endpoint_online=False,
        ray_recovery_required=True,
        runtime_gpu_indices=(0, 3, 3),
        available_logical_gpu_count=2,
    )

    assert readiness.submission_allowed is False
    assert readiness.invalid_runtime_gpu_indices == (3,)
    assert readiness.blocking_reason_codes == (
        "endpoint_offline",
        "ray_recovery_required",
        "invalid_runtime_gpu_indices",
    )


def test_catalog_revision_and_etag_ignore_observation_time_and_readiness() -> None:
    first = _snapshot()
    later = _snapshot(generated_at=first.generated_at + timedelta(hours=1))

    first_catalog = build_feature_catalog(first)
    later_catalog = build_feature_catalog(later)
    first_etag = feature_catalog_etag(
        template_bundle_version=first_catalog.template_bundle_version,
        host_capability_revision=first_catalog.host_capability_revision,
    )
    later_etag = feature_catalog_etag(
        template_bundle_version=later_catalog.template_bundle_version,
        host_capability_revision=later_catalog.host_capability_revision,
    )

    assert first_catalog == later_catalog
    assert first_etag == later_etag
    assert quote_feature_catalog_etag(first_etag) == f'"{first_etag}"'
    assert all(entry.availability.state == "conditional" for entry in first_catalog.entries)
    assert "ray_tainted" not in first_catalog.model_dump_json()


def test_catalog_keeps_raylight_package_observation_advisory() -> None:
    catalog = build_feature_catalog(_snapshot(raylight_installed=False))
    by_id = {entry.id: entry for entry in catalog.entries}

    assert by_id["lora"].backends == ("standard", "raylight")
    assert by_id["lora"].title == "LoRA"
    assert by_id["lora"].availability.state == "conditional"
    assert by_id["raylight_pool_intent"].availability.state == "conditional"
    assert {
        reason.code
        for reason in by_id["raylight_pool_intent"].availability.reasons
    } == {"context_required"}


def test_catalog_uses_evaluator_for_known_static_host_failure() -> None:
    catalog = build_feature_catalog(_snapshot(missing_class_type="SaveVideo"))
    save_take = next(entry for entry in catalog.entries if entry.id == "save_take")

    assert save_take.availability.state == "unavailable"
    assert {reason.code for reason in save_take.availability.reasons} == {
        "node_unavailable"
    }


def test_missing_ffprobe_has_one_evaluator_reason_across_catalog_preflight_and_compile() -> None:
    snapshot = _snapshot(unavailable_media_tools=("ffprobe",))
    draft = fixture_builder._draft("t2v")
    settings = fixture_builder._settings("standard")

    catalog = build_feature_catalog(snapshot)
    save_take = next(entry for entry in catalog.entries if entry.id == "save_take")
    assert save_take.availability.state == "unavailable"
    catalog_reason = next(
        reason
        for reason in save_take.availability.reasons
        if reason.backend == "standard"
    )

    preflight = preflight_v4_timeline(
        draft=draft,
        settings=settings,
        snapshot=snapshot,
        readiness=_ready(),
    )
    assert preflight.valid is False
    preflight_reason = next(
        reason
        for reason in preflight.errors
        if reason.feature_id == "save_take"
        and reason.code == "media_tool_unavailable"
    )

    with pytest.raises(V4CapabilityEvaluationError) as failure:
        compile_v4_timeline(
            draft,
            settings,
            "missing-ffprobe",
            host_capability_snapshot=snapshot,
            operational_readiness=_ready(),
            capability_evaluator=CapabilityEvaluator(
                V4_NODE_CONTRACT_REGISTRY
            ),
        )
    compile_reason = failure.value.evaluation.reasons[0]

    assert _reason_semantics(catalog_reason) == _reason_semantics(
        preflight_reason
    ) == _reason_semantics(compile_reason)
    assert preflight_reason.safe_details == {"tool": "ffprobe"}


def test_ffmpeg_is_required_only_for_full_multisegment_all_export() -> None:
    snapshot = _snapshot(unavailable_media_tools=("ffmpeg",))
    settings = fixture_builder._settings("standard")
    single_all = _with_export_mode(fixture_builder._draft("t2v"), "all")
    multi_segments = fixture_builder._continuity_draft()
    multi_all = _with_export_mode(multi_segments, "all")

    catalog = build_feature_catalog(snapshot)
    save_take = next(entry for entry in catalog.entries if entry.id == "save_take")
    assert save_take.availability.state == "conditional"

    for draft in (single_all, multi_segments):
        report = preflight_v4_timeline(
            draft=draft,
            settings=settings,
            snapshot=snapshot,
            readiness=_ready(),
        )
        assert report.valid is True
        compiled = compile_v4_timeline(
            draft,
            settings,
            f"ffmpeg-not-needed-{draft.export_mode}",
            host_capability_snapshot=snapshot,
            operational_readiness=_ready(),
            capability_evaluator=CapabilityEvaluator(
                V4_NODE_CONTRACT_REGISTRY
            ),
        )
        assert compiled.workflows

    rejected = preflight_v4_timeline(
        draft=multi_all,
        settings=settings,
        snapshot=snapshot,
        readiness=_ready(),
    )
    assert rejected.valid is False
    preflight_reasons = tuple(
        reason
        for reason in rejected.errors
        if reason.code == "media_tool_unavailable"
    )
    assert preflight_reasons
    assert {
        reason.safe_details["tool"] for reason in preflight_reasons
    } == {"ffmpeg"}

    with pytest.raises(V4CapabilityEvaluationError) as failure:
        compile_v4_timeline(
            multi_all,
            settings,
            "ffmpeg-required",
            host_capability_snapshot=snapshot,
            operational_readiness=_ready(),
            capability_evaluator=CapabilityEvaluator(
                V4_NODE_CONTRACT_REGISTRY
            ),
        )
    assert _reason_semantics(failure.value.evaluation.reasons[0]) == (
        _reason_semantics(preflight_reasons[0])
    )


def test_registered_package_and_unknown_capabilities_fail_closed() -> None:
    draft = fixture_builder._draft("t2v")
    settings = fixture_builder._settings("standard")
    compile_input = CreativeCompileInputResolver.resolve_v4(
        draft,
        settings,
        None,
        None,
        None,
    )
    route = compile_input.routes[0]
    context = build_v4_route_context(
        route,
        draft=compile_input.materialize_draft(),
        settings=compile_input.materialize_settings(),
        job_id="unknown-capability",
    )
    template = V4_VALIDATED_TEMPLATES[route.template_id]
    entry = template.template.entries[0]
    binding = resolve_v4_active_feature(
        entry=entry,
        template=template,
        params=V4BuiltinParams(),
        context=context,
    )
    evaluator = CapabilityEvaluator(V4_NODE_CONTRACT_REGISTRY)

    package_evaluation = evaluator.evaluate(
        feature_id=entry.id,
        ctx=context,
        resolution=binding.resolution,
        required_capabilities=CapabilitySet(
            ids=(*binding.required_capabilities.ids, "package.ray")
        ),
        snapshot=_snapshot(unavailable_packages=("ray",)),
        readiness=_ready(),
    )
    assert package_evaluation.available is False
    assert {
        reason.code for reason in package_evaluation.reasons
    } == {"package_unavailable"}

    unknown_evaluation = evaluator.evaluate(
        feature_id=entry.id,
        ctx=context,
        resolution=binding.resolution,
        required_capabilities=CapabilitySet(
            ids=(*binding.required_capabilities.ids, "future.unregistered")
        ),
        snapshot=_snapshot(),
        readiness=_ready(),
    )
    assert unknown_evaluation.available is False
    assert {
        reason.code for reason in unknown_evaluation.reasons
    } == {"unknown_capability"}


def test_v4_preflight_stops_before_emit_and_projects_disabled_switches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_emit(*args, **kwargs):
        raise AssertionError("preflight must stop before emit")

    monkeypatch.setattr(V4BuiltinInterpreter, "emit", forbidden_emit)

    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("standard"),
        snapshot=_snapshot(),
        readiness=_ready(),
    )

    assert report.valid is True
    assert report.errors == ()
    segment = next(iter(report.effective_by_segment.values()))
    feature_by_id = {feature.id: feature for feature in segment.features}
    assert feature_by_id["lora"].state == "noop"
    assert feature_by_id["continuity"].state == "noop"
    assert feature_by_id["lora"].capability.available is True
    assert all(
        feature.adapter_fingerprint.startswith("sha256:")
        for feature in segment.features
    )


def test_v4_raylight_preflight_accepts_taint_with_durable_cleanup_path() -> None:
    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("raylight"),
        snapshot=_snapshot(),
        readiness=build_operational_readiness(
            endpoint_online=True,
            ray_tainted=True,
            ray_cleanup_available=True,
            available_logical_gpu_count=4,
        ),
    )

    assert report.valid is True
    segment = next(iter(report.effective_by_segment.values()))
    assert segment.backend == "raylight"
    assert all(feature.capability.available for feature in segment.features)


def test_v4_raylight_preflight_keeps_package_observation_advisory() -> None:
    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("raylight"),
        snapshot=_snapshot(raylight_installed=False),
        readiness=_ready(),
    )

    assert report.valid is True
    assert report.errors == ()


def test_raylight_conditional_lora_contract_is_checked_only_when_active() -> None:
    snapshot = _snapshot(missing_class_type="DirectorDeckRayLoraLoader")
    assert snapshot.raylight_installation.installed is True
    assert snapshot.raylight_installation.node_contracts_available is False

    no_lora = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("raylight"),
        snapshot=snapshot,
        readiness=_ready(),
    )
    assert no_lora.valid is True

    with_lora = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings(
            "raylight",
            lora_family="fl2va",
            lora_name="conditional-ray-lora.safetensors",
        ),
        snapshot=snapshot,
        readiness=_ready(),
    )
    assert with_lora.valid is False
    assert {
        (error.code, error.feature_id)
        for error in with_lora.errors
    } == {("node_unavailable", "lora")}


def test_raylight_requires_exact_cleanup_contract_before_first_runtime() -> None:
    snapshot = _snapshot(missing_class_type="DirectorDeckRayKill")
    ray_draft = fixture_builder._draft("t2v")
    ray_settings = fixture_builder._settings("raylight")

    catalog = build_feature_catalog(snapshot)
    ray_pool = next(
        entry for entry in catalog.entries if entry.id == "raylight_pool_intent"
    )
    assert ray_pool.availability.state == "unavailable"
    catalog_reason = next(
        reason
        for reason in ray_pool.availability.reasons
        if reason.code == "raylight_cleanup_unavailable"
    )

    preflight = preflight_v4_timeline(
        draft=ray_draft,
        settings=ray_settings,
        snapshot=snapshot,
        readiness=_ready(),
    )
    assert preflight.valid is False
    preflight_reason = next(
        reason
        for reason in preflight.errors
        if reason.code == "raylight_cleanup_unavailable"
    )
    assert _reason_semantics(catalog_reason) == _reason_semantics(
        preflight_reason
    )

    with pytest.raises(V4CapabilityEvaluationError) as failure:
        compile_v4_timeline(
            ray_draft,
            ray_settings,
            "missing-raykill",
            host_capability_snapshot=snapshot,
            operational_readiness=_ready(),
            capability_evaluator=CapabilityEvaluator(
                V4_NODE_CONTRACT_REGISTRY
            ),
        )
    assert _reason_semantics(failure.value.evaluation.reasons[0]) == (
        _reason_semantics(preflight_reason)
    )

    standard_draft = fixture_builder._draft("t2v")
    standard_settings = fixture_builder._settings("standard")
    standard_preflight = preflight_v4_timeline(
        draft=standard_draft,
        settings=standard_settings,
        snapshot=snapshot,
        readiness=_ready(),
    )
    assert standard_preflight.valid is True
    assert compile_v4_timeline(
        standard_draft,
        standard_settings,
        "missing-raykill-standard",
        host_capability_snapshot=snapshot,
        operational_readiness=_ready(),
        capability_evaluator=CapabilityEvaluator(V4_NODE_CONTRACT_REGISTRY),
    ).workflows


@pytest.mark.parametrize(
    ("package", "expected_feature"),
    (("ray", "raylight_pool_intent"), ("xfuser", "raylight_sampling")),
)
def test_v4_raylight_features_declare_required_packages(
    package: str,
    expected_feature: str,
) -> None:
    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("raylight"),
        snapshot=_snapshot(unavailable_packages=(package,)),
        readiness=_ready(),
    )

    assert report.valid is False
    failures = [
        error
        for error in report.errors
        if error.code == "package_unavailable"
        and error.feature_id == expected_feature
    ]
    assert failures
    assert failures[0].safe_details == {"package": package}


@pytest.mark.parametrize(
    "gpu_backends",
    ((), ("xpu", "xpu"), ("mps",)),
    ids=("empty", "xpu-only", "mps-only"),
)
def test_raylight_requires_two_cuda_logical_gpus(
    gpu_backends: tuple[str, ...],
) -> None:
    snapshot = _snapshot(gpu_backends=gpu_backends)
    catalog = build_feature_catalog(snapshot)
    ray_pool = next(
        entry for entry in catalog.entries if entry.id == "raylight_pool_intent"
    )
    assert ray_pool.availability.state == "unavailable"
    assert "raylight_cuda_unavailable" in {
        reason.code for reason in ray_pool.availability.reasons
    }

    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("raylight"),
        snapshot=snapshot,
        readiness=_ready(),
    )
    assert report.valid is False
    failures = [
        error
        for error in report.errors
        if error.code == "raylight_cuda_unavailable"
    ]
    assert failures
    assert {
        error.safe_details["cuda_gpu_count"] for error in failures
    } == {0}


def test_v4_preflight_fails_closed_when_required_class_type_is_missing() -> None:
    snapshot = _snapshot(missing_class_type="SaveVideo")
    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("standard"),
        snapshot=snapshot,
        readiness=_ready(),
    )

    assert report.valid is False
    failures = [error for error in report.errors if error.code == "node_unavailable"]
    assert failures
    assert failures[0].feature_id == "save_take"
    assert set(failures[0].model_dump(mode="json")) == {
        "code",
        "feature_id",
        "segment_id",
        "unit_id",
        "backend",
        "rule",
        "message",
        "remediation",
        "safe_details",
    }
    rendered = f"{failures[0].message} {failures[0].remediation}".lower()
    assert "supported node" not in rendered
    assert "supported runtime" not in rendered
    assert "mapped" in rendered
    assert "install or enable" in rendered


def test_standard_preflight_ignores_interface_module_and_fingerprint_drift() -> None:
    snapshot = _snapshot(drift_class_type="SaveVideo")
    drifted = snapshot.model_copy(
        update={
            "node_registry": {
                **snapshot.node_registry,
                "SaveVideo": "custom_nodes.user_video_nodes",
            },
            "module_fingerprints": {
                **snapshot.module_fingerprints,
                "custom_nodes.user_video_nodes": _fingerprint("f"),
            },
        }
    )

    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("standard"),
        snapshot=drifted,
        readiness=_ready(),
    )

    assert report.valid is True


def test_raylight_preflight_ignores_live_fingerprint_observations() -> None:
    ray_module = V4_NODE_CONTRACT_REGISTRY.require(
        "DirectorDeckRayInitializerAdvanced"
    ).allowed_python_modules[0]
    snapshot = _snapshot(drift_module=ray_module)
    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("raylight"),
        snapshot=snapshot,
        readiness=_ready(),
    )

    assert report.valid is True


def test_v4_preflight_returns_stable_safe_error_for_invalid_creative_input() -> None:
    report = preflight_v4_timeline(
        draft=fixture_builder._draft("t2v"),
        settings=fixture_builder._settings("standard"),
        snapshot=_snapshot(),
        readiness=_ready(),
        segment_ids=["missing-secret-segment"],
    )

    assert report.valid is False
    assert report.effective_by_segment == {}
    assert tuple(error.code for error in report.errors) == (
        "segment_selection_invalid",
    )
    assert "missing-secret-segment" not in report.model_dump_json()
