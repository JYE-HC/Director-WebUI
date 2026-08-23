from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from directordeck.workflow.contracts import (
    FeatureEmission,
    MAX_JSON_CONTAINER_ITEMS,
    MAX_JSON_DEPTH,
    MAX_JSON_STRING_LENGTH,
    FrozenMap,
    GraphAuditSpec,
    NodeContractEvidence,
    ResolvedImplementationIdentity,
    canonical_json_bytes as contracts_canonical_json_bytes,
)
from directordeck.workflow.execution import (
    CompiledPlanDigest,
    CompiledExecutionPlan,
    ComfyNodeCacheIdentity,
    ComfyNodeIdentity,
    ContinuityLateBindingEvidence,
    DocumentDigest,
    EndpointIdentity,
    EndpointRestartCertificate,
    ExactCancelConfirmedEvidence,
    ExactPromptSnapshot,
    ExpectedOutputGeometry,
    ExpectedOutputSpec,
    FeatureExecutionIdentity,
    HistoricalTakeGeometryIdentity,
    HistoryTerminalEvidence,
    InvalidOwnershipTransition,
    LegacyOutputLocator,
    LockedSegmentUnit,
    LockedSubmissionPlan,
    ObservedArtifactSpec,
    OutputDescriptor,
    OwnershipRevisionConflict,
    PreparedControlUnit,
    PreparedSegmentUnit,
    PreviewSource,
    PreviewSpec,
    ProgressPhase,
    ProgressSpec,
    PromptOwnership,
    RayRuntimeIdentity,
    RuntimePoolIdentityContribution,
    RuntimeRequirements,
    RuntimeEpochLateBindingEvidence,
    SegmentExecutionIdentity,
    canonical_json,
    canonical_json_bytes,
    canonical_values_equal,
    comfy_node_cache_identity,
    compiled_execution_plan_digest,
    digest_document,
    derive_feature_execution_specs,
    effective_execution_digest,
    legacy_fnv1a32_document_digest,
    locked_submission_plan_from_compiled,
    sha256_document_digest,
    transition_prompt_ownership,
)


FINGERPRINT_A = "sha256:" + "a" * 64
FINGERPRINT_B = "sha256:" + "b" * 64
NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def prompt() -> dict[str, object]:
    return {
        "10": {
            "class_type": "KSampler",
            "inputs": {"seed": 42, "conditioning": ["5", 0]},
        },
        "20": {
            "class_type": "PreviewImage",
            "inputs": {"images": ["10", 0]},
        },
        "30": {
            "class_type": "SaveVideo",
            "inputs": {"images": ["10", 0], "filename_prefix": "director/test"},
        },
    }


def control_prompt() -> dict[str, object]:
    return {
        "900": {
            "class_type": "DirectorDeckRayKill",
            "inputs": {"runtime_epoch": "epoch-7"},
        }
    }


def evidence(
    node_id: str,
    class_type: str,
    *,
    terminal: str | None = None,
    artifact: str | None = None,
    fingerprint: str = FINGERPRINT_A,
) -> NodeContractEvidence:
    return NodeContractEvidence(
        contract_id=f"contract.{node_id}",
        semantic_version="1.0.0",
        class_type=class_type,
        python_module="test_nodes.runtime",
        runtime_fingerprint=fingerprint,
        execution_terminal_role=terminal,
        persistent_artifact_role=artifact,
    )


def segment_audit() -> GraphAuditSpec:
    return GraphAuditSpec(
        version=1,
        unit_kind="segment",
        take_node_id="30",
        node_contract_snapshot={
            "10": evidence("10", "KSampler"),
            "20": evidence("20", "PreviewImage"),
            "30": evidence(
                "30", "SaveVideo", terminal="take", artifact="take"
            ),
        },
    )


def control_audit() -> GraphAuditSpec:
    return GraphAuditSpec(
        version=1,
        unit_kind="control",
        control_kind="ray_kill",
        node_contract_snapshot={
            "900": evidence("900", "DirectorDeckRayKill", terminal="ray_kill")
        },
    )


def expected_output(*, segment_id: str = "segment-1", node_id: str = "30") -> ExpectedOutputSpec:
    return ExpectedOutputSpec(
        segment_id=segment_id,
        node_id=node_id,
        width=736,
        height=416,
        fps=24.0,
        visible_frame_count=121,
        expected_audio_mode="generated",
    )


def progress() -> ProgressSpec:
    return ProgressSpec(
        version=1,
        phases=(
            ProgressPhase(
                id="sampling",
                label="Sampling",
                node_id="10",
                kind="fractional",
                weight=0.8,
            ),
            ProgressPhase(
                id="persist",
                label="Persisting",
                node_id="30",
                kind="milestone",
                weight=0.2,
            ),
        ),
    )


def previews() -> PreviewSpec:
    return PreviewSpec(
        version=1,
        sources=(
            PreviewSource(
                node_id="10",
                phase_id="sampling",
                publish=True,
                priority=10,
            ),
            PreviewSource(
                node_id="20",
                phase_id="persist",
                publish=True,
                priority=20,
                supersedes=("10",),
            ),
        ),
    )


def standard_requirements() -> RuntimeRequirements:
    return RuntimeRequirements(
        endpoint_key="local-comfy",
        backend="standard",
        logical_gpu_indices=(0,),
        requires_standard_driver_access=True,
    )


def ray_requirements() -> RuntimeRequirements:
    return RuntimeRequirements(
        endpoint_key="local-comfy",
        backend="raylight",
        logical_gpu_indices=(0, 1),
        ray_compatibility_key="compat-v1",
        ray_runtime_key="runtime-v1",
        requires_standard_driver_access=False,
        expected_residency_policy="keep_until_switch",
    )


def endpoint_identity(
    *, endpoint_key: str = "local-comfy", runtime_instance_id: str = "runtime-7"
) -> EndpointIdentity:
    return EndpointIdentity(
        endpoint_key=endpoint_key,
        runtime_instance_id=runtime_instance_id,
    )


def runtime_pool() -> RayRuntimeIdentity:
    return RayRuntimeIdentity(
        schema_version=1,
        placement={"logical_gpu_indices": [0, 1]},
        fixed_parameters={"ulysses_degree": 2, "ring_degree": 1},
        active_feature_pool_identities=(
            RuntimePoolIdentityContribution(
                feature="base_sampling@1.0.0",
                identity={"attention_backend": "flash"},
            ),
        ),
    )


def standard_segment() -> PreparedSegmentUnit:
    return PreparedSegmentUnit(
        id="unit-1",
        owner_segment_id="segment-1",
        family="fl2va",
        backend="standard",
        template_id="h3_standard_segment",
        template_revision=1,
        prompt_base=prompt(),
        graph_audit_spec=segment_audit(),
        expected_output_spec=expected_output(),
        progress_spec=progress(),
        preview_spec=previews(),
        continuity_dependency=None,
        runtime_requirements=standard_requirements(),
        runtime_pool_identity=None,
        effective_execution_digest=execution_digest(),
    )


def ray_segment() -> PreparedSegmentUnit:
    return PreparedSegmentUnit(
        id="unit-ray",
        owner_segment_id="segment-ray",
        family="ref2va",
        backend="raylight",
        template_id="h3_raylight_segment",
        template_revision=2,
        prompt_base=prompt(),
        graph_audit_spec=segment_audit(),
        expected_output_spec=expected_output(segment_id="segment-ray"),
        progress_spec=progress(),
        preview_spec=previews(),
        continuity_dependency={"source": "same_run", "segment_id": "segment-1"},
        runtime_requirements=ray_requirements(),
        runtime_pool_identity=runtime_pool(),
        effective_execution_digest=sha256_document_digest(
            {
                "unit": "unit-ray",
                "runtime_pool": runtime_pool().model_dump(mode="json"),
            }
        ),
    )


def locked_segment(*, group_index: int = 1) -> LockedSegmentUnit:
    unit = standard_segment()
    return LockedSegmentUnit(
        **unit.model_dump(),
        child_id="child-1",
        requested_prompt_id="requested-1",
        group_index=group_index,
        exact_prompt=unit.prompt_base,
        late_bound_values={},
    )


def control_unit(*, group_index: int = 0, preceding: str = "unit-1") -> PreparedControlUnit:
    return PreparedControlUnit(
        id="control-1",
        family="fl2va",
        template_revision=1,
        child_id="child-control-1",
        requested_prompt_id="requested-control-1",
        group_index=group_index,
        prompt_base=control_prompt(),
        graph_audit_spec=control_audit(),
        runtime_descriptor_digest=sha256_document_digest({"runtime": "old"}),
        effective_execution_digest=sha256_document_digest(
            {"control": "control-1", "runtime": "old"}
        ),
        preceding_unit_id=preceding,
    )


def implementation() -> ResolvedImplementationIdentity:
    return ResolvedImplementationIdentity(
        role="sampler",
        class_type="KSampler",
        implementation_id="builtin.sampler",
        semantic_version="1.0.0",
        runtime_fingerprint=FINGERPRINT_A,
        binding_key="sampling.primary",
    )


def feature_identity() -> FeatureExecutionIdentity:
    return FeatureExecutionIdentity(
        feature="base_sampling@1.0.0",
        effective_cache_params={"steps": 6, "scheduler": "beta"},
        resolved_implementations=(implementation(),),
    )


def segment_identity() -> SegmentExecutionIdentity:
    return SegmentExecutionIdentity(
        schema_version=1,
        segment_creative_input={"prompt": "镜头推进 😀"},
        render={"width": 736, "height": 416, "fps": 24},
        family_sampling={"steps": 6, "seed": 42},
        model_stack_projection={"diffusion": "model.safetensors"},
        runtime_placement_projection=None,
        feature_execution_identities=(feature_identity(),),
        continuity_input_identity=None,
        expected_output_geometry=expected_output().geometry,
    )


def execution_digest() -> DocumentDigest:
    return effective_execution_digest(
        segment_identity(),
        template_id="h3_standard_segment",
        template_revision=1,
        resolved_node_contract_identities=tuple(segment_audit().node_contract_snapshot.values()),
    )


def compiled_plan() -> CompiledExecutionPlan:
    return CompiledExecutionPlan(
        version=1,
        template_bundle_version=1,
        segment_units=(standard_segment(),),
        compile_report={"notices": []},
        node_policy={"loader_ids": ["1", "2"]},
        effective_execution_digest=execution_digest(),
    )


def locked_plan() -> LockedSubmissionPlan:
    source = compiled_plan()
    return locked_submission_plan_from_compiled(
        source,
        endpoint_identity=endpoint_identity(),
        units=(control_unit(), locked_segment()),
        source_unit_id="unit-1",
        source_unit_ordinal=0,
        ray_ledger_before={"tail_prompt_id": "old-prompt"},
        ray_ledger_after_intent={"tainted": False},
    )


def exact_snapshot() -> ExactPromptSnapshot:
    return ExactPromptSnapshot(
        schema_version=1,
        unit_id="unit-1",
        unit_kind="segment",
        owner_segment_id="segment-1",
        control_kind=None,
        family="fl2va",
        backend="standard",
        template_id="h3_standard_segment",
        template_revision=1,
        endpoint_identity=endpoint_identity(),
        exact_prompt=prompt(),
        graph_audit_spec=segment_audit(),
        expected_output_spec=expected_output(),
        progress_spec=progress(),
        preview_spec=previews(),
        effective_execution_digest=execution_digest(),
    )


def ownership() -> PromptOwnership:
    return PromptOwnership(
        requested_prompt_id="requested-1",
        state="prepared",
        ownership_revision=0,
        updated_at=NOW,
    )


def cancel_evidence(prompt_id: str = "actual-9") -> ExactCancelConfirmedEvidence:
    return ExactCancelConfirmedEvidence(
        prompt_id=prompt_id,
        confirmation_id="cancel-confirmation-1",
        confirmed_at=NOW + timedelta(seconds=3),
    )


def history_evidence(prompt_id: str = "actual-9") -> HistoryTerminalEvidence:
    return HistoryTerminalEvidence(
        prompt_id=prompt_id,
        terminal_status="succeeded",
        history_digest=sha256_document_digest(
            {"prompt_id": prompt_id, "status": "success"}
        ),
        observed_at=NOW + timedelta(seconds=3),
    )


def restart_evidence(prompt_id: str = "actual-9") -> EndpointRestartCertificate:
    return EndpointRestartCertificate(
        certificate_version=1,
        prompt_id=prompt_id,
        endpoint_identity=endpoint_identity(runtime_instance_id="runtime-8"),
        restart_id="restart-8",
        queue_and_history_cleared=True,
        confirmed_at=NOW + timedelta(seconds=3),
    )


def rebuild(model_type: type, value: object, **changes: object) -> object:
    payload = value.model_dump()  # type: ignore[attr-defined]
    payload.update(changes)
    return model_type(**payload)


def test_all_execution_contracts_are_strictly_immutable_and_json_round_trip() -> None:
    output = expected_output()
    phase = progress().phases[0]
    source = previews().sources[0]
    node_cache = comfy_node_cache_identity(prompt())
    feature = feature_identity()
    segment = segment_identity()
    historical = HistoricalTakeGeometryIdentity(
        project_id="project-1",
        segment_id="segment-1",
        width=736,
        height=416,
        fps=24.0,
        visible_frame_count=121,
        required_audio_capability=True,
    )
    examples = (
        sha256_document_digest({"hello": "世界"}),
        OutputDescriptor(filename="take.mp4", subfolder="director/project-1"),
        output.geometry,
        output,
        ObservedArtifactSpec(
            segment_id="segment-1",
            child_id="child-1",
            output_descriptor=OutputDescriptor(
                filename="take.mp4", subfolder="director/project-1"
            ),
            width=736,
            height=416,
            fps=24.0,
            frame_count=121,
            duration_seconds=5.0416666667,
            has_audio=True,
            media_probe_version="ffprobe-v1",
            content_hash=None,
        ),
        LegacyOutputLocator(segment_id="segment-1", node_id="30"),
        phase,
        progress(),
        source,
        previews(),
        standard_requirements(),
        ray_requirements(),
        node_cache.nodes[0],
        node_cache,
        implementation(),
        feature,
        segment,
        historical,
        runtime_pool().active_feature_pool_identities[0],
        runtime_pool(),
        endpoint_identity(),
        standard_segment(),
        ray_segment(),
        locked_segment(),
        control_unit(),
        compiled_plan(),
        compiled_execution_plan_digest(compiled_plan()),
        locked_plan(),
        exact_snapshot(),
        ownership(),
        cancel_evidence(),
        history_evidence(),
        restart_evidence(),
        ContinuityLateBindingEvidence(
            input_pointer="/10/inputs/source",
            predecessor_segment_id="segment-0",
            dependency_source="same_run",
            output=OutputDescriptor(
                filename="take.mp4", subfolder="director/project-1"
            ),
        ),
        RuntimeEpochLateBindingEvidence(
            input_pointer="/10/inputs/runtime_epoch",
            epoch=7,
        ),
    )

    for value in examples:
        restored = type(value).model_validate_json(value.model_dump_json())
        assert restored.model_dump(mode="json") == value.model_dump(mode="json")

    unit = standard_segment()
    assert isinstance(unit.graph_audit_spec, GraphAuditSpec)
    assert isinstance(feature.resolved_implementations[0], ResolvedImplementationIdentity)
    assert isinstance(unit.prompt_base, FrozenMap)
    assert isinstance(unit.prompt_base["10"]["inputs"], FrozenMap)
    with pytest.raises(ValidationError, match="frozen"):
        unit.owner_segment_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        unit.prompt_base["10"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        unit.prompt_base["10"]["inputs"]["seed"] = 99  # type: ignore[index]


def test_document_digest_matches_javascript_utf16_and_utf8_fixed_vectors() -> None:
    # Verified against App.tsx timelineDocumentHash and Node crypto.  The
    # astral character is two charCodeAt iterations for FNV, but four UTF-8
    # bytes for canonical SHA-256.
    value = {"z": "中文😀", "a": [1, True, None, 1.5]}

    assert contracts_canonical_json_bytes is canonical_json_bytes
    assert canonical_json(value) == '{"a":[1,true,null,1.5],"z":"中文😀"}'
    assert sha256_document_digest(value) == DocumentDigest(
        algorithm="sha256-canonical-json-v1",
        value="sha256-cd0730b1a3c8815918199092ef3fae2f2f01242e2132ecbb70a1bd24970fa15a",
    )
    assert legacy_fnv1a32_document_digest(value) == DocumentDigest(
        algorithm="fnv1a32-json-stringify-v1",
        value="fnv1a-2d9838b3",
    )
    assert digest_document(value) == sha256_document_digest(value)
    assert digest_document(
        value, algorithm="fnv1a32-json-stringify-v1"
    ) == legacy_fnv1a32_document_digest(value)

    reversed_properties = {"a": [1, True, None, 1.5], "z": "中文😀"}
    assert sha256_document_digest(reversed_properties) == sha256_document_digest(value)
    assert legacy_fnv1a32_document_digest(reversed_properties) != (
        legacy_fnv1a32_document_digest(value)
    )
    assert canonical_json([1e-7, 1e-6, 1e20, 1e21]) == (
        "[1e-7,0.000001,100000000000000000000,1e+21]"
    )
    assert canonical_json({"\ue000": 1, "😀": 2}) == '{"😀":2,"\ue000":1}'


@pytest.mark.parametrize(
    ("left", "right"),
    (
        (
            {"b": [1, True, None], "a": "中文😀"},
            {"a": "中文😀", "b": (1.0, True, None)},
        ),
        ({"value": 1}, {"value": 1.0}),
        ({"value": True}, {"value": 1}),
        ({"value": False}, {"value": 0.0}),
        ({"nested": [{"x": "y"}]}, {"nested": ({"x": "z"},)}),
        ({"a": 1}, {"b": 1}),
        ({"\ue000": 1, "😀": 2}, {"😀": 2.0, "\ue000": 1.0}),
    ),
)
def test_streaming_canonical_equality_matches_serialized_equality(
    left: object,
    right: object,
) -> None:
    assert canonical_values_equal(left, right) is (
        canonical_json(left) == canonical_json(right)
    )


def test_streaming_canonical_equality_rejects_invalid_values() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycles"):
        canonical_values_equal(cyclic, cyclic)
    with pytest.raises(ValueError, match="negative zero"):
        canonical_values_equal({"value": -0.0}, {"value": 0})
    with pytest.raises(ValueError, match="NaN or Infinity"):
        canonical_values_equal({"value": float("nan")}, {"value": 0})
    with pytest.raises(ValueError, match="surrogate"):
        canonical_values_equal({"value": "\ud800"}, {"value": ""})
    with pytest.raises(ValueError, match="keys must be strings"):
        canonical_values_equal({1: "value"}, {1: "value"})
    with pytest.raises(ValueError, match="unsupported"):
        canonical_values_equal({1}, {1})


def test_legacy_fnv_matches_json_stringify_for_lone_and_paired_surrogates() -> None:
    assert legacy_fnv1a32_document_digest({"bad": "\ud800"}) == DocumentDigest(
        algorithm="fnv1a32-json-stringify-v1",
        value="fnv1a-cf73593d",
    )
    assert legacy_fnv1a32_document_digest({"paired": "\ud83d\ude00"}) == (
        legacy_fnv1a32_document_digest({"paired": "😀"})
    )
    with pytest.raises(ValueError, match="surrogate"):
        sha256_document_digest({"bad": "\ud800"})


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf"), -0.0])
def test_canonical_sha256_rejects_ambiguous_numbers(number: float) -> None:
    with pytest.raises(ValueError):
        sha256_document_digest({"number": number})


def test_document_digest_rejects_mislabeled_values_and_unsafe_json() -> None:
    with pytest.raises(ValidationError):
        DocumentDigest(
            algorithm="sha256-canonical-json-v1", value="fnv1a-00000000"
        )
    with pytest.raises(ValidationError):
        DocumentDigest(
            algorithm="fnv1a32-json-stringify-v1", value="sha256-" + "0" * 64
        )
    with pytest.raises(ValueError, match="safe range"):
        sha256_document_digest({"too_large": 2**53})
    with pytest.raises(ValueError, match="surrogate"):
        sha256_document_digest({"bad": "\ud800"})


@pytest.mark.parametrize(
    ("filename", "subfolder"),
    [
        ("../take.mp4", ""),
        ("take.mp4", "../outside"),
        ("take.mp4", "."),
        ("take.mp4", "/absolute"),
        ("folder/take.mp4", ""),
        ("take.mp4", "bad\\folder"),
        ("take[1].mp4", ""),
    ],
)
def test_observed_output_descriptor_fails_closed(
    filename: str, subfolder: str
) -> None:
    with pytest.raises(ValidationError, match="unsafe"):
        OutputDescriptor(filename=filename, subfolder=subfolder)


def test_expected_observed_and_legacy_output_contracts_are_distinct() -> None:
    assert expected_output().kind == "video"
    assert expected_output().role == "take"
    with pytest.raises(ValidationError):
        LegacyOutputLocator(
            segment_id="segment-1",
            node_id="30",
            width=736,  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        ObservedArtifactSpec(
            segment_id="segment-1",
            child_id="child-1",
            output_descriptor=OutputDescriptor(filename="take.mp4"),
            width=736,
            height=416,
            fps=24.0,
            frame_count=0,
            duration_seconds=5.0,
            has_audio=False,
            media_probe_version="ffprobe-v1",
        )


def test_progress_and_preview_specs_enforce_normalization_uniqueness_and_dag() -> None:
    with pytest.raises(ValidationError, match="sum to 1"):
        ProgressSpec(
            version=1,
            phases=(
                ProgressPhase(
                    id="a", label="A", node_id="10", kind="fractional", weight=0.4
                ),
                ProgressPhase(
                    id="b", label="B", node_id="30", kind="milestone", weight=0.5
                ),
            ),
        )
    with pytest.raises(ValidationError, match="node ids must be unique"):
        ProgressSpec(
            version=1,
            phases=(
                ProgressPhase(
                    id="a", label="A", node_id="10", kind="fractional", weight=0.5
                ),
                ProgressPhase(
                    id="b", label="B", node_id="10", kind="milestone", weight=0.5
                ),
            ),
        )
    with pytest.raises(ValidationError, match="unknown nodes"):
        PreviewSpec(
            version=1,
            sources=(
                PreviewSource(
                    node_id="10",
                    phase_id="sampling",
                    publish=True,
                    priority=1,
                    supersedes=("missing",),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="acyclic"):
        PreviewSpec(
            version=1,
            sources=(
                PreviewSource(
                    node_id="10",
                    phase_id="sampling",
                    publish=True,
                    priority=1,
                    supersedes=("20",),
                ),
                PreviewSource(
                    node_id="20",
                    phase_id="persist",
                    publish=True,
                    priority=2,
                    supersedes=("10",),
                ),
            ),
        )


def test_feature_execution_hints_derive_ordered_scoped_authority() -> None:
    progress_spec, preview_spec = derive_feature_execution_specs(
        (
            (
                ("10", "11"),
                FeatureEmission(
                    progress_hints=(
                        {
                            "id": "coarse_sampling",
                            "label": "Coarse sampling",
                            "node_id": "11",
                            "kind": "fractional",
                            "weight": 0.6,
                        },
                    ),
                    preview_hints=(
                        {
                            "node_id": "11",
                            "phase_id": "coarse_sampling",
                            "publish": True,
                            "priority": 10,
                        },
                    ),
                ),
            ),
            (
                ("20",),
                FeatureEmission(
                    progress_hints=(
                        {
                            "id": "detail_sampling",
                            "label": "Detail sampling",
                            "node_id": "20",
                            "kind": "fractional",
                            "weight": 0.4,
                        },
                    ),
                    preview_hints=(
                        {
                            "node_id": "20",
                            "phase_id": "detail_sampling",
                            "publish": True,
                            "priority": 20,
                            "supersedes": ("11",),
                        },
                    ),
                ),
            ),
        )
    )

    assert progress_spec is not None and preview_spec is not None
    assert [phase.id for phase in progress_spec.phases] == [
        "coarse_sampling",
        "detail_sampling",
    ]
    assert [source.node_id for source in preview_spec.sources] == ["11", "20"]
    assert preview_spec.sources[1].supersedes == ("11",)


def test_feature_execution_hints_define_all_or_no_explicit_authority() -> None:
    assert derive_feature_execution_specs(()) == (None, None)

    progress_only = FeatureEmission(
        progress_hints=(
            {
                "id": "persist",
                "label": "Persist",
                "node_id": "30",
                "kind": "milestone",
                "weight": 1.0,
            },
        )
    )
    progress_spec, preview_spec = derive_feature_execution_specs(
        ((("30",), progress_only),)
    )
    assert progress_spec is not None and preview_spec is not None
    assert preview_spec.sources == ()

    stage_and_progress = FeatureEmission(
        progress_hints=(
            {
                "id": "load",
                "label": "Load model",
                "node_id": "10",
                "kind": "stage",
                "weight": 0.0,
            },
            {
                "id": "persist",
                "label": "Persist",
                "node_id": "30",
                "kind": "milestone",
                "weight": 1.0,
            },
        )
    )
    progress_spec, _ = derive_feature_execution_specs(
        ((("10", "30"), stage_and_progress),)
    )
    assert progress_spec is not None
    assert [phase.kind for phase in progress_spec.phases] == ["stage", "milestone"]
    assert [phase.weight for phase in progress_spec.phases] == [0.0, 1.0]

    preview_only = FeatureEmission(
        preview_hints=(
            {
                "node_id": "10",
                "phase_id": "sampling",
                "publish": True,
                "priority": 1,
            },
        )
    )
    with pytest.raises(ValueError, match="require explicit progress"):
        derive_feature_execution_specs(((('10',), preview_only),))


@pytest.mark.parametrize(
    "hint",
    (
        {
            "id": "bad-stage",
            "label": "Bad stage",
            "node_id": "10",
            "kind": "stage",
            "weight": 0.1,
        },
        {
            "id": "bad-milestone",
            "label": "Bad milestone",
            "node_id": "10",
            "kind": "milestone",
            "weight": 0.0,
        },
    ),
)
def test_feature_execution_hints_reject_invalid_stage_weight(hint: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="progress hint is invalid"):
        derive_feature_execution_specs(
            ((("10",), FeatureEmission(progress_hints=(hint,))),)
        )


@pytest.mark.parametrize(
    ("scoped_emissions", "message"),
    (
        (
            (
                (
                    ("10",),
                    FeatureEmission(
                        progress_hints=(
                            {
                                "id": "sampling",
                                "label": "Sampling",
                                "node_id": "outside",
                                "kind": "fractional",
                                "weight": 1.0,
                            },
                        )
                    ),
                ),
            ),
            "must belong",
        ),
        (
            (
                (
                    ("10",),
                    FeatureEmission(
                        progress_hints=(
                            {
                                "id": "sampling",
                                "label": "Sampling",
                                "node_id": "10",
                                "kind": "fractional",
                                "weight": 0.9,
                            },
                        ),
                        preview_hints=(
                            {
                                "node_id": "10",
                                "phase_id": "sampling",
                                "publish": True,
                                "priority": 1,
                            },
                        ),
                    ),
                ),
            ),
            "sum to 1",
        ),
        (
            (
                (
                    ("10",),
                    FeatureEmission(
                        progress_hints=(
                            {
                                "id": "sampling",
                                "label": "Sampling",
                                "node_id": "10",
                                "kind": "fractional",
                                "weight": 1.0,
                            },
                        ),
                        preview_hints=(
                            {
                                "node_id": "10",
                                "phase_id": "unknown",
                                "publish": True,
                                "priority": 1,
                            },
                        ),
                    ),
                ),
            ),
            "unknown progress phase",
        ),
    ),
)
def test_feature_execution_hints_fail_closed(
    scoped_emissions: tuple[tuple[tuple[str, ...], FeatureEmission], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        derive_feature_execution_specs(scoped_emissions)


def test_feature_execution_hints_reject_duplicate_phase_and_node_claims() -> None:
    duplicate_phases = FeatureEmission(
        progress_hints=(
            {
                "id": "sampling",
                "label": "First",
                "node_id": "10",
                "kind": "fractional",
                "weight": 0.5,
            },
            {
                "id": "sampling",
                "label": "Second",
                "node_id": "20",
                "kind": "fractional",
                "weight": 0.5,
            },
        )
    )
    with pytest.raises(ValidationError, match="phase ids must be unique"):
        derive_feature_execution_specs(((('10', '20'), duplicate_phases),))

    duplicate_preview_nodes = FeatureEmission(
        progress_hints=(
            {
                "id": "sampling",
                "label": "Sampling",
                "node_id": "10",
                "kind": "fractional",
                "weight": 1.0,
            },
        ),
        preview_hints=(
            {
                "node_id": "10",
                "phase_id": "sampling",
                "publish": True,
                "priority": 1,
            },
            {
                "node_id": "10",
                "phase_id": "sampling",
                "publish": False,
                "priority": 2,
            },
        ),
    )
    with pytest.raises(ValidationError, match="source node ids must be unique"):
        derive_feature_execution_specs(((('10',), duplicate_preview_nodes),))


def test_prompt_references_and_graph_audit_must_match_exact_nodes() -> None:
    with pytest.raises(ValidationError, match="expected output node"):
        rebuild(
            PreparedSegmentUnit,
            standard_segment(),
            expected_output_spec=expected_output(node_id="missing"),
        )
    bad_preview = PreviewSpec(
        version=1,
        sources=(
            PreviewSource(
                node_id="missing",
                phase_id="sampling",
                publish=True,
                priority=1,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="preview spec references nodes"):
        rebuild(PreparedSegmentUnit, standard_segment(), preview_spec=bad_preview)
    bad_phase_preview = PreviewSpec(
        version=1,
        sources=(
            PreviewSource(
                node_id="10",
                phase_id="unknown",
                publish=True,
                priority=1,
            ),
        ),
    )
    with pytest.raises(ValidationError, match="unknown progress phase"):
        rebuild(PreparedSegmentUnit, standard_segment(), preview_spec=bad_phase_preview)

    incomplete_audit = GraphAuditSpec(
        version=1,
        unit_kind="segment",
        take_node_id="30",
        node_contract_snapshot={
            "10": evidence("10", "KSampler"),
            "30": evidence(
                "30", "SaveVideo", terminal="take", artifact="take"
            ),
        },
    )
    with pytest.raises(ValidationError, match="exactly cover"):
        rebuild(
            PreparedSegmentUnit,
            standard_segment(),
            graph_audit_spec=incomplete_audit,
        )
    class_drifted_prompt = prompt()
    class_drifted_prompt["10"]["class_type"] = "DifferentSampler"  # type: ignore[index]
    with pytest.raises(ValidationError, match="class_type does not match"):
        rebuild(
            PreparedSegmentUnit,
            standard_segment(),
            prompt_base=class_drifted_prompt,
        )


def test_runtime_requirements_separate_target_from_resident_state() -> None:
    assert ray_requirements().expected_residency_policy == "keep_until_switch"
    with pytest.raises(ValidationError, match="at least two"):
        rebuild(
            RuntimeRequirements,
            ray_requirements(),
            logical_gpu_indices=(0,),
        )
    with pytest.raises(ValidationError, match="compatibility and runtime"):
        rebuild(
            RuntimeRequirements,
            ray_requirements(),
            ray_runtime_key=None,
        )
    with pytest.raises(ValidationError, match="cannot carry Ray"):
        rebuild(
            RuntimeRequirements,
            standard_requirements(),
            ray_runtime_key="stale-runtime",
        )
    with pytest.raises(ValidationError, match="unique"):
        rebuild(
            RuntimeRequirements,
            ray_requirements(),
            logical_gpu_indices=(0, 0),
        )


def test_prepared_units_and_compiled_plan_reject_cross_contract_drift() -> None:
    with pytest.raises(ValidationError, match="match unit owner"):
        rebuild(
            PreparedSegmentUnit,
            standard_segment(),
            expected_output_spec=expected_output(segment_id="other-segment"),
        )
    with pytest.raises(ValidationError, match="backend must match"):
        rebuild(
            PreparedSegmentUnit,
            standard_segment(),
            runtime_requirements=ray_requirements(),
        )
    with pytest.raises(ValidationError, match="requires a runtime pool"):
        rebuild(
            PreparedSegmentUnit,
            ray_segment(),
            runtime_pool_identity=None,
        )
    with pytest.raises(ValidationError, match="cannot carry a Ray runtime"):
        rebuild(
            PreparedSegmentUnit,
            standard_segment(),
            runtime_pool_identity=runtime_pool(),
        )
    with pytest.raises(ValidationError):
        rebuild(PreparedSegmentUnit, standard_segment(), family="wan")
    with pytest.raises(ValidationError):
        rebuild(PreparedSegmentUnit, standard_segment(), template_revision="1")

    duplicate = standard_segment()
    with pytest.raises(ValidationError, match="unit ids must be unique"):
        CompiledExecutionPlan(
            version=1,
            template_bundle_version=1,
            segment_units=(duplicate, duplicate),
            compile_report={},
            node_policy={},
            effective_execution_digest=execution_digest(),
        )
    with pytest.raises(ValidationError):
        rebuild(CompiledExecutionPlan, compiled_plan(), template_bundle_version="1")
    with pytest.raises(ValidationError):
        CompiledExecutionPlan(
            version=1,
            template_bundle_version=1,
            segment_units=(control_unit(),),  # type: ignore[arg-type]
            compile_report={},
            node_policy={},
            effective_execution_digest=execution_digest(),
        )
    with pytest.raises(ValidationError, match="prepared segment units only"):
        CompiledExecutionPlan(
            version=1,
            template_bundle_version=1,
            segment_units=(locked_segment(),),
            compile_report={},
            node_policy={},
            effective_execution_digest=execution_digest(),
        )


def test_locked_plan_has_only_ordered_segments_and_registered_controls() -> None:
    source = compiled_plan()
    plan = locked_plan()
    assert [unit.kind for unit in plan.units] == ["control", "segment"]
    assert plan.units[0].preceding_unit_id == plan.units[1].id  # type: ignore[union-attr]
    assert isinstance(plan.source_compiled_plan_digest, CompiledPlanDigest)
    assert plan.validate_source_compiled_plan(source) is plan

    with pytest.raises(ValidationError, match="globally adjacent"):
        LockedSubmissionPlan(
            version=1,
            endpoint_identity=endpoint_identity(),
            units=(control_unit(), locked_segment(group_index=2)),
            source_compiled_plan_digest=compiled_execution_plan_digest(source),
            source_unit_id="unit-1",
            source_unit_ordinal=0,
            ray_ledger_before=None,
            ray_ledger_after_intent=None,
        )
    with pytest.raises(ValidationError, match="immediately precede"):
        LockedSubmissionPlan(
            version=1,
            endpoint_identity=endpoint_identity(),
            units=(control_unit(preceding="missing"), locked_segment()),
            source_compiled_plan_digest=compiled_execution_plan_digest(source),
            source_unit_id="unit-1",
            source_unit_ordinal=0,
            ray_ledger_before=None,
            ray_ledger_after_intent=None,
        )
    with pytest.raises(ValidationError, match="must end with one segment"):
        LockedSubmissionPlan(
            version=1,
            endpoint_identity=endpoint_identity(),
            units=(locked_segment(group_index=1), control_unit(group_index=2)),
            source_compiled_plan_digest=compiled_execution_plan_digest(source),
            source_unit_id="unit-1",
            source_unit_ordinal=0,
            ray_ledger_before=None,
            ray_ledger_after_intent=None,
        )

    other_endpoint_requirements = rebuild(
        RuntimeRequirements,
        standard_requirements(),
        endpoint_key="other-comfy",
    )
    wrong_endpoint_segment = rebuild(
        LockedSegmentUnit,
        locked_segment(),
        runtime_requirements=other_endpoint_requirements,
    )
    with pytest.raises(ValidationError, match="endpoint key must match"):
        LockedSubmissionPlan(
            version=1,
            endpoint_identity=endpoint_identity(),
            units=(control_unit(), wrong_endpoint_segment),
            source_compiled_plan_digest=compiled_execution_plan_digest(source),
            source_unit_id="unit-1",
            source_unit_ordinal=0,
            ray_ledger_before=None,
            ray_ledger_after_intent=None,
        )

    with pytest.raises(ValidationError):
        LockedSubmissionPlan(
            version=1,
            endpoint_identity=endpoint_identity(),
            units=(control_unit(), locked_segment()),
            source_compiled_plan_digest=execution_digest(),
            source_unit_id="unit-1",
            source_unit_ordinal=0,
            ray_ledger_before=None,
            ray_ledger_after_intent=None,
        )

    changed_source = rebuild(
        CompiledExecutionPlan,
        source,
        compile_report={"changed": True},
    )
    with pytest.raises(ValueError, match="source digest does not match"):
        plan.validate_source_compiled_plan(changed_source)
    drifted_locked = rebuild(
        LockedSegmentUnit,
        locked_segment(),
        template_revision=2,
    )
    with pytest.raises(ValueError, match="identity drifted"):
        locked_submission_plan_from_compiled(
            source,
            endpoint_identity=endpoint_identity(),
            units=(control_unit(), drifted_locked),
            source_unit_id="unit-1",
            source_unit_ordinal=0,
            ray_ledger_before=None,
            ray_ledger_after_intent=None,
        )

    drifted_prompt = prompt()
    drifted_prompt["10"]["inputs"]["seed"] = 99  # type: ignore[index]
    with pytest.raises(ValueError, match="identity drifted"):
        locked_submission_plan_from_compiled(
            source,
            endpoint_identity=endpoint_identity(),
            units=(
                control_unit(),
                rebuild(
                    LockedSegmentUnit,
                    locked_segment(),
                    prompt_base=drifted_prompt,
                ),
            ),
            source_unit_id="unit-1",
            source_unit_ordinal=0,
            ray_ledger_before=None,
            ray_ledger_after_intent=None,
        )

    ray_source = rebuild(
        CompiledExecutionPlan,
        source,
        segment_units=(ray_segment(),),
    )
    locked_ray = LockedSegmentUnit(
        **ray_segment().model_dump(),
        child_id="child-ray",
        requested_prompt_id="requested-ray",
        group_index=1,
        exact_prompt=ray_segment().prompt_base,
        late_bound_values={},
    )
    drifted_continuity = rebuild(
        LockedSegmentUnit,
        locked_ray,
        continuity_dependency={
            "source": "historical_take",
            "segment_id": "wrong-segment",
        },
    )
    with pytest.raises(ValueError, match="identity drifted"):
        locked_submission_plan_from_compiled(
            ray_source,
            endpoint_identity=endpoint_identity(),
            units=(
                control_unit(preceding="unit-ray"),
                drifted_continuity,
            ),
            source_unit_id="unit-ray",
            source_unit_ordinal=0,
            ray_ledger_before=None,
            ray_ledger_after_intent=None,
        )

    with pytest.raises(ValidationError, match="canonical SHA-256"):
        rebuild(
            PreparedControlUnit,
            control_unit(),
            runtime_descriptor_digest=legacy_fnv1a32_document_digest({"old": True}),
        )


def test_exact_snapshot_distinguishes_segment_and_control_units() -> None:
    snapshot = exact_snapshot()
    assert isinstance(snapshot.endpoint_identity, EndpointIdentity)
    assert snapshot.exact_prompt["10"]["inputs"]["seed"] == 42
    with pytest.raises(ValidationError, match="requires an owner"):
        rebuild(ExactPromptSnapshot, snapshot, owner_segment_id=None)
    with pytest.raises(ValidationError):
        rebuild(
            ExactPromptSnapshot,
            snapshot,
            endpoint_identity={"endpoint_key": "local-comfy", "ticket": 7},
        )

    control_snapshot = ExactPromptSnapshot(
        schema_version=1,
        unit_id="control-1",
        unit_kind="control",
        owner_segment_id=None,
        control_kind="ray_kill",
        family="fl2va",
        backend="raylight",
        template_id="raylight_kill_control",
        template_revision=1,
        endpoint_identity=endpoint_identity(),
        exact_prompt=control_prompt(),
        graph_audit_spec=control_audit(),
        expected_output_spec=None,
        progress_spec=None,
        preview_spec=None,
        effective_execution_digest=execution_digest(),
    )
    assert control_snapshot.progress_spec is None
    with pytest.raises(ValidationError, match="cannot carry user-facing"):
        rebuild(
            ExactPromptSnapshot,
            control_snapshot,
            expected_output_spec=expected_output(),
        )


def test_prompt_ownership_effective_id_and_cas_transitions_are_monotonic() -> None:
    prepared = ownership()
    assert prepared.effective_prompt_id == "requested-1"

    submitting = transition_prompt_ownership(
        prepared,
        expected_revision=0,
        state="submitting",
        updated_at=NOW + timedelta(seconds=1),
    )
    actual = transition_prompt_ownership(
        submitting,
        expected_revision=1,
        state="owned_actual_id",
        actual_prompt_id="actual-9",
        updated_at=NOW + timedelta(seconds=2),
    )
    assert actual.effective_prompt_id == "actual-9"
    assert actual.ownership_revision == 2

    with pytest.raises(OwnershipRevisionConflict):
        transition_prompt_ownership(
            actual,
            expected_revision=1,
            state="cancel_pending",
            updated_at=NOW + timedelta(seconds=3),
        )
    with pytest.raises(InvalidOwnershipTransition, match="cannot transition"):
        transition_prompt_ownership(
            actual,
            expected_revision=2,
            state="submitting",
            updated_at=NOW + timedelta(seconds=3),
        )
    with pytest.raises(InvalidOwnershipTransition, match="cannot be cleared"):
        transition_prompt_ownership(
            actual,
            expected_revision=2,
            state="cancel_pending",
            actual_prompt_id=None,
            updated_at=NOW + timedelta(seconds=3),
        )
    with pytest.raises(InvalidOwnershipTransition, match="backward"):
        transition_prompt_ownership(
            actual,
            expected_revision=2,
            state="cancel_pending",
            updated_at=NOW,
        )

    cleaned = transition_prompt_ownership(
        actual,
        expected_revision=2,
        state="cleanup_confirmed",
        cleanup_certificate=cancel_evidence(),
        updated_at=NOW + timedelta(seconds=3),
    )
    assert isinstance(cleaned.cleanup_certificate, ExactCancelConfirmedEvidence)
    assert cleaned.cleanup_certificate.prompt_id == "actual-9"
    with pytest.raises(InvalidOwnershipTransition):
        transition_prompt_ownership(
            cleaned,
            expected_revision=3,
            state="owned_actual_id",
            updated_at=NOW + timedelta(seconds=4),
        )

    terminal = transition_prompt_ownership(
        actual,
        expected_revision=2,
        state="terminal_confirmed",
        cleanup_certificate=history_evidence(),
        updated_at=NOW + timedelta(seconds=3),
    )
    assert isinstance(terminal.cleanup_certificate, HistoryTerminalEvidence)
    restarted = transition_prompt_ownership(
        actual,
        expected_revision=2,
        state="cleanup_confirmed",
        cleanup_certificate=restart_evidence(),
        updated_at=NOW + timedelta(seconds=3),
    )
    assert isinstance(restarted.cleanup_certificate, EndpointRestartCertificate)

    pending = transition_prompt_ownership(
        submitting,
        expected_revision=1,
        state="cancel_pending",
        updated_at=NOW + timedelta(seconds=2),
    )
    pending_with_actual = transition_prompt_ownership(
        pending,
        expected_revision=2,
        state="cancel_pending",
        actual_prompt_id="late-actual",
        updated_at=NOW + timedelta(seconds=3),
    )
    assert pending_with_actual.effective_prompt_id == "late-actual"
    with pytest.raises(InvalidOwnershipTransition, match="cannot transition"):
        transition_prompt_ownership(
            pending_with_actual,
            expected_revision=3,
            state="owned_actual_id",
            updated_at=NOW + timedelta(seconds=4),
        )


def test_prompt_ownership_state_requires_matching_evidence() -> None:
    with pytest.raises(ValidationError, match="requires the actual"):
        PromptOwnership(
            requested_prompt_id="requested-1",
            state="owned_actual_id",
            ownership_revision=1,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError, match="cannot carry actual"):
        PromptOwnership(
            requested_prompt_id="requested-1",
            actual_prompt_id="actual-1",
            state="owned_requested_id",
            ownership_revision=1,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError, match="structured release evidence"):
        PromptOwnership(
            requested_prompt_id="requested-1",
            state="cleanup_confirmed",
            ownership_revision=1,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError, match="structured release evidence"):
        PromptOwnership(
            requested_prompt_id="requested-1",
            state="terminal_confirmed",
            ownership_revision=1,
            updated_at=NOW,
        )
    with pytest.raises(ValidationError, match="effective prompt id"):
        PromptOwnership(
            requested_prompt_id="requested-1",
            actual_prompt_id="actual-9",
            state="cleanup_confirmed",
            ownership_revision=1,
            cleanup_certificate=cancel_evidence("requested-1"),
            updated_at=NOW,
        )
    with pytest.raises(ValidationError, match="history terminal evidence"):
        PromptOwnership(
            requested_prompt_id="requested-1",
            state="terminal_confirmed",
            ownership_revision=1,
            cleanup_certificate=cancel_evidence("requested-1"),
            updated_at=NOW,
        )
    with pytest.raises(ValidationError, match="cancel or endpoint restart"):
        PromptOwnership(
            requested_prompt_id="requested-1",
            state="cleanup_confirmed",
            ownership_revision=1,
            cleanup_certificate=history_evidence("requested-1"),
            updated_at=NOW,
        )
    with pytest.raises(ValidationError):
        ExactCancelConfirmedEvidence(
            prompt_id="requested-1",
            confirmation_id="",
            confirmed_at=NOW,
        )
    with pytest.raises(ValidationError):
        EndpointRestartCertificate(
            certificate_version=2,
            prompt_id="requested-1",
            endpoint_identity=endpoint_identity(),
            restart_id="restart-1",
            queue_and_history_cleared=True,
            confirmed_at=NOW,
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        PromptOwnership(
            requested_prompt_id="requested-1",
            state="prepared",
            ownership_revision=0,
            updated_at=datetime(2026, 8, 21, 12, 0),
        )


def test_five_identity_models_have_narrow_non_overlapping_semantics() -> None:
    cache_identity = comfy_node_cache_identity(prompt())
    assert isinstance(cache_identity, ComfyNodeCacheIdentity)
    assert tuple(node.node_id for node in cache_identity.nodes) == ("10", "20", "30")
    assert cache_identity == comfy_node_cache_identity(dict(reversed(tuple(prompt().items()))))

    with pytest.raises(ValidationError):
        FeatureExecutionIdentity(
            feature="base_sampling@1.0.0",
            effective_cache_params={"steps": 6},
            resolved_implementations=(implementation(),),
            mapping_source="user_override",  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError, match="bindings must be unique"):
        FeatureExecutionIdentity(
            feature="base_sampling@1.0.0",
            effective_cache_params={},
            resolved_implementations=(implementation(), implementation()),
        )
    with pytest.raises(ValidationError, match="must be unique"):
        rebuild(
            SegmentExecutionIdentity,
            segment_identity(),
            feature_execution_identities=(feature_identity(), feature_identity()),
        )
    with pytest.raises(ValidationError, match="must be unique"):
        rebuild(
            RayRuntimeIdentity,
            runtime_pool(),
            active_feature_pool_identities=(
                runtime_pool().active_feature_pool_identities[0],
                runtime_pool().active_feature_pool_identities[0],
            ),
        )


def test_identity_json_is_recursively_typed_bounded_and_always_digestible() -> None:
    historical = HistoricalTakeGeometryIdentity(
        project_id="project-1",
        segment_id="segment-1",
        width=736,
        height=416,
        fps=24.0,
        visible_frame_count=121,
        required_audio_capability=True,
    )
    identities = (
        comfy_node_cache_identity(prompt()),
        feature_identity(),
        segment_identity(),
        historical,
        runtime_pool(),
    )
    for identity in identities:
        assert canonical_json_bytes(identity)

    inputs_schema = ComfyNodeIdentity.model_json_schema()["properties"]["inputs"]
    assert inputs_schema

    deep: dict[str, object] = {}
    cursor = deep
    for _ in range(MAX_JSON_DEPTH):
        child: dict[str, object] = {}
        cursor["nested"] = child
        cursor = child
    with pytest.raises(ValidationError, match="maximum depth"):
        ComfyNodeIdentity(node_id="1", class_type="Node", inputs=deep)

    with pytest.raises(ValidationError, match="maximum length"):
        ComfyNodeIdentity(
            node_id="1",
            class_type="Node",
            inputs={"items": list(range(MAX_JSON_CONTAINER_ITEMS + 1))},
        )
    with pytest.raises(ValidationError, match="string exceeds"):
        ComfyNodeIdentity(
            node_id="1",
            class_type="Node",
            inputs={"text": "x" * (MAX_JSON_STRING_LENGTH + 1)},
        )
    with pytest.raises(ValidationError, match="negative zero"):
        FeatureExecutionIdentity(
            feature="base_sampling@1.0.0",
            effective_cache_params={"shift": -0.0},
            resolved_implementations=(implementation(),),
        )
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValidationError, match="reference cycles"):
        ComfyNodeIdentity(node_id="1", class_type="Node", inputs=cyclic)


def test_effective_execution_digest_changes_only_with_effective_identity() -> None:
    identity = segment_identity()
    contracts = tuple(segment_audit().node_contract_snapshot.values())
    baseline = effective_execution_digest(
        identity,
        template_id="h3_standard_segment",
        template_revision=1,
        resolved_node_contract_identities=contracts,
    )
    equivalent = effective_execution_digest(
        rebuild(
            SegmentExecutionIdentity,
            identity,
            render={"fps": 24, "height": 416, "width": 736},
        ),
        template_id="h3_standard_segment",
        template_revision=1,
        resolved_node_contract_identities=contracts,
    )
    changed_template = effective_execution_digest(
        identity,
        template_id="h3_standard_segment",
        template_revision=2,
        resolved_node_contract_identities=contracts,
    )
    changed_contracts = tuple(
        evidence(
            node_id,
            item.class_type,
            terminal=item.execution_terminal_role,
            artifact=item.persistent_artifact_role,
            fingerprint=FINGERPRINT_B,
        )
        for node_id, item in segment_audit().node_contract_snapshot.items()
    )
    changed_runtime = effective_execution_digest(
        identity,
        template_id="h3_standard_segment",
        template_revision=1,
        resolved_node_contract_identities=changed_contracts,
    )

    assert equivalent == baseline
    assert changed_template != baseline
    assert changed_runtime != baseline
    assert baseline.algorithm == "sha256-canonical-json-v1"


def test_comfy_node_cache_identity_rejects_invalid_prompt_shape() -> None:
    with pytest.raises(ValueError, match="inputs"):
        comfy_node_cache_identity({"1": {"class_type": "KSampler"}})
    with pytest.raises(ValidationError, match="unique"):
        ComfyNodeCacheIdentity(
            nodes=(
                ComfyNodeIdentity(node_id="1", class_type="A", inputs={}),
                ComfyNodeIdentity(node_id="1", class_type="B", inputs={}),
            )
        )
