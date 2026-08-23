from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from directordeck.workflow.contracts import (
    MAX_PUBLISHED_LIST_ITEMS,
    MAX_PUBLISHED_RECORD_FIELDS,
    AllowedLateBoundInput,
    CapabilitySet,
    ControlTemplate,
    ControlTemplateSet,
    EdgeRef,
    FeatureResolution,
    FeatureTemplateEntry,
    FrozenMap,
    GraphAuditSpec,
    HostCapabilityProvider,
    HostCapabilitySnapshot,
    ListRef,
    LogicalGpuCapability,
    MediaToolCapability,
    NodeContract,
    NodeContractEvidence,
    NodeContractRegistry,
    NodeOutputContract,
    ObjectInfoContract,
    ObjectInfoInputContract,
    ObjectInfoOutputContract,
    OperationalReadiness,
    PackageCapability,
    PublicResourceRead,
    PublicResourceWrite,
    PublishedValueRef,
    RayLightInstallation,
    RecordRef,
    ResolvedImplementationIdentity,
    Resource,
    ResourcePool,
    ResourcePoolTransaction,
    ResourceReadDeclaration,
    ResourceWriteDeclaration,
    RuntimeEffectContract,
    SegmentTemplate,
    SegmentTemplateSet,
    TemplateBundle,
    TerminalRef,
    canonical_json_bytes,
    canonical_sha256,
)


def _fingerprint(character: str = "a") -> str:
    return f"sha256:{character * 64}"


def _feature(
    feature_id: str,
    phase: str,
    *,
    backends: tuple[str, ...] = ("standard",),
    reads: tuple[ResourceReadDeclaration, ...] = (),
    writes: tuple[ResourceWriteDeclaration, ...] = (),
    conflicts: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
) -> FeatureTemplateEntry:
    return FeatureTemplateEntry(
        id=feature_id,
        version=1,
        title=feature_id,
        description=f"Contract for {feature_id}",
        mode="needed",
        graph_phase=phase,
        reads=reads,
        writes=writes,
        params_schema={
            "type": "object",
            "properties": {"strength": {"type": "number"}},
        },
        defaults={"strength": 1.0},
        cache_policy={"identity": "params"},
        backends=backends,
        families=("fl2va", "ref2va"),
        conflicts=conflicts,
        requires=requires,
        scopes=("project",),
        ui={"group": "models"},
    )


def _bundle() -> TemplateBundle:
    standard = SegmentTemplate(
        id="h3_standard_segment",
        revision=2,
        entries=(
            _feature("shared_models", "bootstrap"),
            _feature("standard_model", "model_load"),
            _feature("sampling", "sampling"),
        ),
    )
    raylight = SegmentTemplate(
        id="h3_raylight_segment",
        revision=3,
        entries=(
            _feature("ray_pool", "bootstrap", backends=("raylight",)),
            _feature("ray_model", "model_load", backends=("raylight",)),
            _feature("ray_sampling", "sampling", backends=("raylight",)),
        ),
    )
    return TemplateBundle(
        version=4,
        segment_templates=SegmentTemplateSet(
            standard=standard,
            raylight=raylight,
        ),
        control_templates=ControlTemplateSet(
            ray_kill=ControlTemplate(id="raylight_kill_control", revision=1)
        ),
    )


def _outputs() -> tuple[ObjectInfoOutputContract, ...]:
    return (ObjectInfoOutputContract(index=0, port_type="VIDEO", name="video"),)


def _object_info(*, output_node: bool = False) -> ObjectInfoContract:
    return ObjectInfoContract(
        normalization_version=1,
        required_inputs={
            "model": ObjectInfoInputContract(port_type="MODEL"),
            "mode": ObjectInfoInputContract(
                port_type="COMBO",
                enum_values=("fast", "quality"),
                has_director_default=True,
                director_default="quality",
            ),
        },
        optional_inputs={
            "seed": ObjectInfoInputContract(
                port_type="INT",
                has_director_default=True,
                director_default=0,
            )
        },
        director_supplied_inputs=("model", "mode", "seed"),
        outputs=_outputs(),
        output_node=output_node,
    )


def _runtime_effect(
    policy: str = "strict_transform",
    unsupported_behavior: str = "raise",
) -> RuntimeEffectContract:
    families = ("fl2va", "ref2va") if policy == "strict_transform" else ()
    backends = ("standard", "raylight") if policy == "strict_transform" else ()
    return RuntimeEffectContract(
        policy=policy,
        unsupported_behavior=unsupported_behavior,
        validation_method="node_contract",
        verified_model_families=families,
        verified_backends=backends,
        notes=("Frozen stage-1 contract",),
    )


def _node_contract(
    class_type: str,
    module: str,
    character: str,
    *,
    effect: RuntimeEffectContract | None = None,
    terminal: str | None = None,
    persistent: str | None = None,
) -> NodeContract:
    object_info = _object_info(output_node=terminal is not None)
    return NodeContract(
        contract_id=f"contract.{character}",
        semantic_version="1.0.0",
        class_type=class_type,
        allowed_python_modules=(module,),
        object_info_contract=object_info,
        output_contract=NodeOutputContract(slots=object_info.outputs),
        execution_terminal_role=terminal,
        persistent_artifact_role=persistent,
        runtime_effect_contract=effect or _runtime_effect(),
        supported_runtime_fingerprints=(_fingerprint(character),),
    )


def _implementation(
    class_type: str = "SaveVideo",
    character: str = "a",
) -> ResolvedImplementationIdentity:
    return ResolvedImplementationIdentity(
        role="save_take",
        class_type=class_type,
        implementation_id=f"contract.{character}",
        semantic_version="1.0.0",
        runtime_fingerprint=_fingerprint(character),
        binding_key=f"binding.{character}",
    )


def _evidence(
    class_type: str,
    character: str,
    *,
    terminal: str | None = None,
    persistent: str | None = None,
) -> NodeContractEvidence:
    return NodeContractEvidence(
        contract_id=f"contract.{character}",
        semantic_version="1.0.0",
        class_type=class_type,
        python_module=(
            "custom_nodes.DirectorDeck-RayLight" if class_type == "DirectorDeckRayKill" else "nodes"
        ),
        runtime_fingerprint=_fingerprint(character),
        execution_terminal_role=terminal,
        persistent_artifact_role=persistent,
    )


def _resource(
    *,
    node_id: str = "10",
    revision: int = 1,
    feature_id: str = "decode_video",
) -> Resource:
    return Resource(
        name="video.decoded",
        type="VIDEO",
        value=EdgeRef(node_id=node_id, output_slot=0),
        source_feature_id=feature_id,
        revision=revision,
        producer_node_ids=(node_id,),
    )


def _segment_audit() -> GraphAuditSpec:
    resource = _resource()
    return GraphAuditSpec(
        version=1,
        unit_kind="segment",
        take_node_id="30",
        node_contract_snapshot={
            "10": _evidence("VAEDecode", "a"),
            "20": _evidence("CreateVideo", "b"),
            "30": _evidence("SaveVideo", "c", terminal="take", persistent="take"),
        },
        public_writes=(PublicResourceWrite(operation="define", resource=resource),),
        public_reads=(
            PublicResourceRead(
                resource_name=resource.name,
                type=resource.type,
                revision=resource.revision,
                consumer_node_id="20",
                input_pointer="/20/inputs/video",
                value=resource.value,
            ),
        ),
        allowed_late_bound_inputs=(
            AllowedLateBoundInput(
                input_pointer="/20/inputs/video",
                value_kind="edge",
                source_kind="resource",
                resource_name=resource.name,
                revision=resource.revision,
            ),
        ),
        structural_influence_features=(resource.source_feature_id,),
    )


def _host_snapshot(
    *,
    generated_at: datetime | None = None,
    module_fingerprint: str | None = None,
    object_info: ObjectInfoContract | None = None,
) -> HostCapabilitySnapshot:
    return HostCapabilitySnapshot(
        schema_version=1,
        generated_at=generated_at or datetime(2026, 8, 21, tzinfo=timezone.utc),
        node_registry={"SaveVideo": "nodes"},
        object_info_slices={"SaveVideo": object_info or _object_info(output_node=True)},
        module_fingerprints={"nodes": module_fingerprint or _fingerprint("a")},
        importable_packages={
            "torch": PackageCapability(importable=True, version="2.13.0")
        },
        gpu_inventory=(
            LogicalGpuCapability(
                logical_index=0,
                backend="cuda",
                total_memory_mb=24_576,
            ),
        ),
        raylight_installation=RayLightInstallation(
            installed=True,
            package_version="1.8.0",
            node_contracts_available=True,
        ),
        media_tool_status={
            "ffmpeg": MediaToolCapability(available=True, version="7.1")
        },
    )


def test_template_bundle_is_strict_frozen_and_json_round_trips() -> None:
    bundle = _bundle()

    assert TemplateBundle.model_validate_json(bundle.model_dump_json()) == bundle
    with pytest.raises(ValidationError, match="frozen"):
        bundle.version = 5  # type: ignore[misc]
    with pytest.raises(TypeError):
        bundle.segment_templates.standard.entries[0].params_schema["x"] = 1  # type: ignore[index]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        TemplateBundle.model_validate({**bundle.model_dump(), "unknown": True})


def test_template_rejects_non_monotonic_duplicate_and_wrong_backend_entries() -> None:
    with pytest.raises(ValidationError, match="monotonic"):
        SegmentTemplate(
            id="h3_standard_segment",
            revision=1,
            entries=(
                _feature("sample", "sampling"),
                _feature("load", "model_load"),
            ),
        )
    duplicate = _feature("same", "bootstrap")
    with pytest.raises(ValidationError, match="entry ids"):
        SegmentTemplate(
            id="h3_standard_segment",
            revision=1,
            entries=(duplicate, duplicate),
        )
    with pytest.raises(ValidationError, match="does not support"):
        SegmentTemplate(
            id="h3_standard_segment",
            revision=1,
            entries=(_feature("ray", "bootstrap", backends=("raylight",)),),
        )


def test_feature_entry_rejects_invalid_dependencies_and_unbounded_json() -> None:
    with pytest.raises(ValidationError, match="itself"):
        _feature("lora", "model_patch", conflicts=("lora",))
    with pytest.raises(ValidationError, match="disjoint"):
        _feature(
            "lora",
            "model_patch",
            conflicts=("attention",),
            requires=("attention",),
        )
    with pytest.raises(ValidationError, match="unique"):
        FeatureTemplateEntry(
            **{
                **_feature("lora", "model_patch").model_dump(),
                "backends": ("standard", "standard"),
            }
        )
    with pytest.raises(ValidationError):
        FeatureTemplateEntry(
            **{
                **_feature("lora", "model_patch").model_dump(),
                "defaults": {"bad": object()},
            }
        )
    with pytest.raises(ValidationError, match="cannot own asset"):
        FeatureTemplateEntry(
            **{
                **_feature("lora", "model_patch").model_dump(),
                "params_schema": {
                    "type": "object",
                    "properties": {"asset_id": {"type": "string"}},
                },
            }
        )
    with pytest.raises(ValidationError, match="AssetReference"):
        FeatureTemplateEntry(
            **{
                **_feature("lora", "model_patch").model_dump(),
                "defaults": {
                    "reference": {
                        "name": "private.png",
                        "subfolder": "input",
                        "type": "input",
                        "kind": "image",
                    }
                },
            }
        )


def test_published_value_ref_is_bounded_json_safe_and_round_trips() -> None:
    value = RecordRef(
        fields={
            "primary": EdgeRef(node_id="10", output_slot=0),
            "references": ListRef(
                items=(
                    EdgeRef(node_id="11", output_slot=1),
                    EdgeRef(node_id="12", output_slot=2),
                )
            ),
        }
    )
    adapter = TypeAdapter(PublishedValueRef)
    restored = adapter.validate_json(adapter.dump_json(value))

    assert restored == value
    with pytest.raises(TypeError):
        value.fields["extra"] = EdgeRef(node_id="13", output_slot=0)  # type: ignore[index]
    with pytest.raises(ValidationError, match="maximum length"):
        ListRef(
            items=tuple(
                EdgeRef(node_id=str(index), output_slot=0)
                for index in range(MAX_PUBLISHED_LIST_ITEMS + 1)
            )
        )
    with pytest.raises(ValidationError, match="maximum length"):
        RecordRef(
            fields={
                f"field_{index}": EdgeRef(node_id=str(index), output_slot=0)
                for index in range(MAX_PUBLISHED_RECORD_FIELDS + 1)
            }
        )
    with pytest.raises(ValidationError, match="JSON-safe"):
        RecordRef(fields={"bad/key": EdgeRef(node_id="10", output_slot=0)})

    too_deep: PublishedValueRef = EdgeRef(node_id="10", output_slot=0)
    for _ in range(7):
        too_deep = ListRef(items=(too_deep,))
    with pytest.raises(ValidationError, match="recursion depth"):
        ListRef(items=(too_deep,))


def test_resource_requires_exact_producer_nodes() -> None:
    with pytest.raises(ValidationError, match="exactly match"):
        Resource(
            name="video.decoded",
            type="VIDEO",
            value=EdgeRef(node_id="10", output_slot=0),
            source_feature_id="decode",
            revision=1,
            producer_node_ids=("11",),
        )


def test_resource_pool_is_immutable_transactional_and_json_round_trips() -> None:
    empty = ResourcePool()
    first = empty.define(
        name="video.decoded",
        type="VIDEO",
        value=EdgeRef(node_id="10", output_slot=0),
        source_feature_id="decode",
        producer_node_ids=("10",),
    )
    second = first.replace(
        name="video.decoded",
        expected_type="VIDEO",
        expected_revision=1,
        value=EdgeRef(node_id="20", output_slot=0),
        source_feature_id="upscale",
        producer_node_ids=("20",),
    )

    assert len(empty.resources) == 0
    assert first.read_required("video.decoded", expected_type="VIDEO").revision == 1
    assert second.read_required(
        "video.decoded", expected_type="VIDEO", expected_revision=2
    ).revision == 2
    assert second.read_optional("audio.decoded", expected_type="AUDIO") is None
    assert ResourcePool.from_snapshot_json(second.snapshot_json()) == second
    with pytest.raises(TypeError):
        second.resources["x"] = _resource()  # type: ignore[index]

    transaction = first.begin()
    staged = transaction.define(
        name="audio.decoded",
        type="AUDIO",
        value=EdgeRef(node_id="40", output_slot=0),
        source_feature_id="decode_audio",
        producer_node_ids=("40",),
    )
    assert staged.commit().read_required("audio.decoded", expected_type="AUDIO")
    assert transaction.rollback() == first
    assert "audio.decoded" not in transaction.staged.resources
    with pytest.raises(ValueError, match="already defined"):
        staged.define(
            name="audio.decoded",
            type="AUDIO",
            value=EdgeRef(node_id="41", output_slot=0),
            source_feature_id="decode_audio",
            producer_node_ids=("41",),
        )
    assert staged.commit().resources["audio.decoded"].revision == 1


def test_resource_pool_rejects_define_replace_read_and_terminal_violations() -> None:
    pool = ResourcePool().define(
        name="video.decoded",
        type="VIDEO",
        value=EdgeRef(node_id="10", output_slot=0),
        source_feature_id="decode",
        producer_node_ids=("10",),
    )
    with pytest.raises(ValueError, match="already defined"):
        pool.define(
            name="video.decoded",
            type="VIDEO",
            value=EdgeRef(node_id="11", output_slot=0),
            source_feature_id="decode",
            producer_node_ids=("11",),
        )
    with pytest.raises(KeyError, match="missing"):
        pool.read_required("audio.decoded", expected_type="AUDIO")
    with pytest.raises(TypeError, match="expected"):
        pool.read_required("video.decoded", expected_type="AUDIO")
    with pytest.raises(ValueError, match="revision"):
        pool.read_required(
            "video.decoded",
            expected_type="VIDEO",
            expected_revision=2,
        )
    with pytest.raises(KeyError):
        pool.replace(
            name="missing",
            expected_type="VIDEO",
            value=EdgeRef(node_id="12", output_slot=0),
            source_feature_id="upscale",
            producer_node_ids=("12",),
        )

    terminal_pool = ResourcePool().define(
        name="take.saved",
        type="TAKE",
        value=TerminalRef(node_id="30"),
        source_feature_id="save_take",
        producer_node_ids=("30",),
    )
    with pytest.raises(TypeError, match="terminal"):
        terminal_pool.read_required("take.saved", expected_type="TAKE")
    assert terminal_pool.read_required(
        "take.saved", expected_type="TAKE", allow_terminal=True
    )
    with pytest.raises(TypeError, match="terminal"):
        terminal_pool.replace(
            name="take.saved",
            expected_type="TAKE",
            value=EdgeRef(node_id="31", output_slot=0),
            source_feature_id="invalid_replace",
            producer_node_ids=("31",),
        )


def test_resource_pool_transaction_rejects_content_rewrite_and_revision_jumps() -> None:
    base = ResourcePool().define(
        name="video.decoded",
        type="VIDEO",
        value=EdgeRef(node_id="10", output_slot=0),
        source_feature_id="decode",
        producer_node_ids=("10",),
    )
    rewritten_same_revision = ResourcePool(
        resources={
            "video.decoded": _resource(node_id="20", revision=1, feature_id="rewrite")
        }
    )
    skipped_revision = ResourcePool(
        resources={
            "video.decoded": _resource(node_id="30", revision=3, feature_id="skip")
        }
    )

    with pytest.raises(ValidationError, match="same resource revision"):
        ResourcePoolTransaction(base=base, staged=rewritten_same_revision)
    with pytest.raises(ValidationError, match="skip a resource revision"):
        ResourcePoolTransaction(base=base, staged=skipped_revision)
    changed_type = ResourcePool(
        resources={
            "video.decoded": Resource(
                name="video.decoded",
                type="AUDIO",
                value=EdgeRef(node_id="40", output_slot=0),
                source_feature_id="change_type",
                revision=2,
                producer_node_ids=("40",),
            )
        }
    )
    with pytest.raises(ValidationError, match="cannot change resource"):
        ResourcePoolTransaction(base=base, staged=changed_type)


def test_feature_resolution_is_explicit_and_json_round_trips() -> None:
    resolution = FeatureResolution(
        state="active",
        implementations=(_implementation(),),
        resolution_details={"selected": "strict"},
    )
    noop = FeatureResolution(
        state="noop",
        implementations=(),
        resolution_details={"reason": "host behavior is already exact"},
    )

    assert FeatureResolution.model_validate_json(resolution.model_dump_json()) == resolution
    assert FeatureResolution.model_validate_json(noop.model_dump_json()) == noop
    with pytest.raises(ValidationError, match="requires an implementation"):
        FeatureResolution(state="active", implementations=(), resolution_details={})
    with pytest.raises(ValidationError, match="cannot claim"):
        FeatureResolution(
            state="noop",
            implementations=(_implementation(),),
            resolution_details={"reason": "not used"},
        )
    with pytest.raises(ValidationError, match="non-empty reason"):
        FeatureResolution(state="noop", implementations=(), resolution_details={})
    duplicate = _implementation()
    with pytest.raises(ValidationError, match="binding keys"):
        FeatureResolution(
            state="active",
            implementations=(duplicate, duplicate),
            resolution_details={},
        )


def test_runtime_effect_and_object_info_negative_contracts() -> None:
    with pytest.raises(ValidationError, match="must raise"):
        _runtime_effect("strict_transform", "identity")
    with pytest.raises(ValidationError, match="verified"):
        RuntimeEffectContract(
            policy="strict_transform",
            unsupported_behavior="raise",
            validation_method="node_contract",
            verified_model_families=(),
            verified_backends=(),
        )
    with pytest.raises(ValidationError, match="cannot report identity"):
        _runtime_effect("side_effect_only", "identity")
    with pytest.raises(ValidationError, match="enum_values"):
        ObjectInfoInputContract(
            port_type="COMBO",
            enum_values=("a", "b"),
            has_director_default=True,
            director_default="c",
        )
    shared = ObjectInfoInputContract(port_type="MODEL")
    with pytest.raises(ValidationError, match="disjoint"):
        ObjectInfoContract(
            normalization_version=1,
            required_inputs={"model": shared},
            optional_inputs={"model": shared},
        )
    with pytest.raises(ValidationError, match="absent"):
        ObjectInfoContract(
            normalization_version=1,
            required_inputs={"model": shared},
            director_supplied_inputs=("unknown",),
        )
    with pytest.raises(ValidationError, match="contiguous"):
        ObjectInfoContract(
            normalization_version=1,
            outputs=(ObjectInfoOutputContract(index=1, port_type="VIDEO"),),
        )


def test_official_and_third_party_nodes_share_one_registry_and_round_trip() -> None:
    official = _node_contract("SaveVideo", "nodes", "a")
    third_party = _node_contract(
        "DirectorDeckRayKill",
        "custom_nodes.DirectorDeck-RayLight",
        "b",
        effect=_runtime_effect("side_effect_only", "raise"),
        terminal="ray_kill",
    )
    registry = NodeContractRegistry().register(official).register(third_party)

    assert registry.require("SaveVideo") == official
    assert registry.require("DirectorDeckRayKill") == third_party
    assert NodeContractRegistry.model_validate_json(registry.model_dump_json()) == registry
    with pytest.raises(ValueError, match="already registered"):
        registry.register(official)
    with pytest.raises(KeyError, match="unknown"):
        registry.require("MissingNode")


def test_node_contract_rejects_role_output_module_and_fingerprint_drift() -> None:
    object_info = _object_info()
    with pytest.raises(ValidationError, match="outputs"):
        NodeContract(
            contract_id="bad.output",
            semantic_version="1.0.0",
            class_type="BadOutput",
            allowed_python_modules=("nodes",),
            object_info_contract=object_info,
            output_contract=NodeOutputContract(slots=()),
            runtime_effect_contract=_runtime_effect(),
            supported_runtime_fingerprints=(_fingerprint("a"),),
        )
    with pytest.raises(ValidationError):
        _node_contract("BadModule", "/" + "home/user/custom.py", "a")
    with pytest.raises(ValidationError):
        NodeContract(
            **{
                **_node_contract("BadFingerprint", "nodes", "a").model_dump(),
                "supported_runtime_fingerprints": ("unknown",),
            }
        )
    with pytest.raises(ValidationError, match="persistent artifact"):
        _node_contract("Take", "nodes", "a", terminal="take")
    with pytest.raises(ValidationError, match="RayKill must use"):
        _node_contract("DirectorDeckRayKill", "custom_nodes.DirectorDeck-RayLight", "a", terminal="ray_kill")
    with pytest.raises(ValidationError, match="registry key"):
        NodeContractRegistry(contracts={"Wrong": _node_contract("Right", "nodes", "a")})


def test_registry_enforces_runtime_fingerprint_and_strict_output_effect() -> None:
    strict = _node_contract("StrictPatch", "nodes", "a")
    identity = _node_contract(
        "OptionalPatch",
        "custom_nodes.optional",
        "b",
        effect=_runtime_effect("identity_allowed", "identity"),
    )
    registry = NodeContractRegistry().register(strict).register(identity)
    strict_resolution = FeatureResolution(
        state="active",
        implementations=(_implementation("StrictPatch", "a"),),
        resolution_details={},
    )
    identity_resolution = FeatureResolution(
        state="active",
        implementations=(_implementation("OptionalPatch", "b"),),
        resolution_details={},
    )

    assert registry.validate_resolution(
        strict_resolution,
        output_affecting=True,
        model_family="fl2va",
        backend="standard",
    ) == (
        strict,
    )
    assert registry.validate_resolution(
        identity_resolution,
        output_affecting=False,
        model_family="fl2va",
        backend="standard",
    ) == (
        identity,
    )
    with pytest.raises(ValueError, match="strict_transform"):
        registry.validate_resolution(
            identity_resolution,
            output_affecting=True,
            model_family="fl2va",
            backend="standard",
        )
    unknown_fingerprint = FeatureResolution(
        state="active",
        implementations=(
            _implementation("StrictPatch", "a").model_copy(
                update={
                    "runtime_fingerprint": _fingerprint("f"),
                    "binding_key": "binding.f",
                }
            ),
        ),
        resolution_details={},
    )
    with pytest.raises(ValueError, match="adapter identity does not match"):
        registry.validate_resolution(
            unknown_fingerprint,
            output_affecting=True,
            model_family="fl2va",
            backend="standard",
        )


def test_graph_audit_segment_and_control_round_trip() -> None:
    segment = _segment_audit()
    control = GraphAuditSpec(
        version=1,
        unit_kind="control",
        control_kind="ray_kill",
        node_contract_snapshot={
            "90": _evidence("DirectorDeckRayKill", "d", terminal="ray_kill")
        },
    )

    assert GraphAuditSpec.model_validate_json(segment.model_dump_json()) == segment
    assert GraphAuditSpec.model_validate_json(control.model_dump_json()) == control


def test_graph_audit_rejects_terminal_resource_and_shape_violations() -> None:
    with pytest.raises(ValidationError, match="data read"):
        PublicResourceRead(
            resource_name="take.saved",
            type="TAKE",
            revision=1,
            consumer_node_id="20",
            input_pointer="/20/inputs/take",
            value=TerminalRef(node_id="30"),
        )
    with pytest.raises(ValidationError, match="define"):
        PublicResourceWrite(
            operation="define",
            resource=_resource(revision=2),
        )
    with pytest.raises(ValidationError, match="increment"):
        PublicResourceWrite(
            operation="replace",
            resource=_resource(revision=3),
            previous_revision=1,
        )
    with pytest.raises(ValidationError, match="name and revision"):
        AllowedLateBoundInput(
            input_pointer="/20/inputs/video",
            value_kind="edge",
            source_kind="resource",
            resource_name="video.decoded",
        )
    with pytest.raises(ValidationError, match="cannot claim"):
        AllowedLateBoundInput(
            input_pointer="/20/inputs/video",
            value_kind="edge",
            source_kind="continuity",
            resource_name="video.decoded",
            revision=1,
        )
    unmatched_late_bound = AllowedLateBoundInput(
        input_pointer="/20/inputs/audio",
        value_kind="edge",
        source_kind="resource",
        resource_name="audio.decoded",
        revision=1,
    )

    segment = _segment_audit().model_dump()
    with pytest.raises(ValidationError, match="take_node_id"):
        GraphAuditSpec(**{**segment, "take_node_id": None})
    with pytest.raises(ValidationError, match="consumer"):
        bad_read = dict(segment["public_reads"][0])
        bad_read["consumer_node_id"] = "missing"
        GraphAuditSpec(**{**segment, "public_reads": (bad_read,)})
    with pytest.raises(ValidationError, match="structural influence"):
        GraphAuditSpec(
            **{**segment, "structural_influence_features": ("unknown_feature",)}
        )
    with pytest.raises(ValidationError, match="late-bound"):
        late = segment["allowed_late_bound_inputs"][0]
        GraphAuditSpec(
            **{**segment, "allowed_late_bound_inputs": (late, late)}
        )
    with pytest.raises(ValidationError, match="public write revision"):
        GraphAuditSpec(
            **{
                **segment,
                "allowed_late_bound_inputs": (unmatched_late_bound,),
            }
        )


def test_host_capability_snapshot_is_sanitized_static_and_round_trips() -> None:
    snapshot = _host_snapshot()
    later = _host_snapshot(
        generated_at=snapshot.generated_at + timedelta(minutes=5)
    )
    changed = _host_snapshot(module_fingerprint=_fingerprint("b"))

    assert HostCapabilitySnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    assert snapshot.host_capability_revision() == later.host_capability_revision()
    assert snapshot.host_capability_revision() != changed.host_capability_revision()
    assert canonical_json_bytes(snapshot).startswith(b'{"generated_at"')
    assert canonical_sha256(snapshot).startswith("sha256:")
    with pytest.raises(TypeError):
        snapshot.node_registry["Other"] = "nodes"  # type: ignore[index]

    class Provider:
        def snapshot(self) -> HostCapabilitySnapshot:
            return snapshot

    assert isinstance(Provider(), HostCapabilityProvider)


def test_host_capability_snapshot_rejects_private_and_inconsistent_state() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _host_snapshot(generated_at=datetime(2026, 8, 21))
    with pytest.raises(ValidationError, match="absent node"):
        HostCapabilitySnapshot(
            **{
                **_host_snapshot().model_dump(),
                "object_info_slices": {"Unknown": _object_info()},
            }
        )
    sparse = HostCapabilitySnapshot(
        **{**_host_snapshot().model_dump(), "module_fingerprints": {}}
    )
    assert sparse.module_fingerprints == {}
    advisory = HostCapabilitySnapshot(
        **{
            **_host_snapshot().model_dump(),
            "module_fingerprints": {
                "custom_nodes.absent": _fingerprint("c")
            },
        }
    )
    assert advisory.module_fingerprints == {
        "custom_nodes.absent": _fingerprint("c")
    }
    with pytest.raises(ValidationError, match="contiguous"):
        HostCapabilitySnapshot(
            **{
                **_host_snapshot().model_dump(),
                "gpu_inventory": (
                    LogicalGpuCapability(logical_index=1, backend="cuda"),
                ),
            }
        )

    private_default = ObjectInfoContract(
        normalization_version=1,
        required_inputs={
            "model": ObjectInfoInputContract(
                port_type="STRING",
                has_director_default=True,
                director_default="/" + "home/alice/private/model.safetensors",
            )
        },
        director_supplied_inputs=("model",),
    )
    with pytest.raises(ValidationError, match="absolute paths"):
        _host_snapshot(object_info=private_default)
    for private_location in (
        r"\\server\share\private",
        r"\\?\C:\private",
        "file:///" + "home/alice/private",
        "FILE://server/share/private",
    ):
        private_location_default = ObjectInfoContract(
            normalization_version=1,
            required_inputs={
                "model": ObjectInfoInputContract(
                    port_type="STRING",
                    has_director_default=True,
                    director_default=private_location,
                )
            },
            director_supplied_inputs=("model",),
        )
        with pytest.raises(ValidationError, match="absolute paths"):
            _host_snapshot(object_info=private_location_default)
    credential_default = ObjectInfoContract(
        normalization_version=1,
        required_inputs={
            "endpoint": ObjectInfoInputContract(
                port_type="STRING",
                has_director_default=True,
                director_default="https://alice:" + "secret@example.invalid/api",
            )
        },
        director_supplied_inputs=("endpoint",),
    )
    with pytest.raises(ValidationError, match="URL credentials"):
        _host_snapshot(object_info=credential_default)
    token_default = ObjectInfoContract(
        normalization_version=1,
        required_inputs={
            "access_token": ObjectInfoInputContract(
                port_type="STRING",
                has_director_default=True,
                director_default="plain-private-value",
            )
        },
        director_supplied_inputs=("access_token",),
    )
    with pytest.raises(ValidationError, match="sensitive defaults"):
        _host_snapshot(object_info=token_default)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        HostCapabilitySnapshot.model_validate(
            {**_host_snapshot().model_dump(), "queue_length": 3}
        )


def test_operational_readiness_is_separate_strict_and_round_trips() -> None:
    ready = OperationalReadiness(
        endpoint_online=True,
        submission_allowed=True,
        ray_recovery_required=False,
        ray_tainted=False,
    )
    blocked = OperationalReadiness(
        endpoint_online=True,
        submission_allowed=False,
        ray_recovery_required=True,
        ray_tainted=True,
        invalid_runtime_gpu_indices=(4,),
        blocking_reason_codes=("ray_recovery_required", "invalid_gpu_index"),
    )

    assert OperationalReadiness.model_validate_json(ready.model_dump_json()) == ready
    assert OperationalReadiness.model_validate_json(blocked.model_dump_json()) == blocked
    with pytest.raises(ValidationError, match="cannot be allowed"):
        OperationalReadiness(
            endpoint_online=False,
            submission_allowed=True,
            ray_recovery_required=False,
            ray_tainted=False,
        )
    with pytest.raises(ValidationError, match="reason code"):
        OperationalReadiness(
            endpoint_online=True,
            submission_allowed=False,
            ray_recovery_required=False,
            ray_tainted=False,
        )
    with pytest.raises(ValidationError, match="unique"):
        OperationalReadiness(
            endpoint_online=True,
            submission_allowed=False,
            ray_recovery_required=True,
            ray_tainted=False,
            invalid_runtime_gpu_indices=(1, 1),
            blocking_reason_codes=("invalid_gpu",),
        )


def test_capability_set_is_frozen_unique_and_json_round_trips() -> None:
    capabilities = CapabilitySet(ids=("node.save_video", "media.ffmpeg"))

    assert CapabilitySet.model_validate_json(capabilities.model_dump_json()) == capabilities
    with pytest.raises(ValidationError, match="unique"):
        CapabilitySet(ids=("node.save_video", "node.save_video"))
    with pytest.raises(ValidationError, match="namespaced"):
        CapabilitySet(ids=("unqualified",))


def test_uninstalled_raylight_cannot_claim_runtime_contracts() -> None:
    with pytest.raises(ValidationError, match="uninstalled"):
        RayLightInstallation(
            installed=False,
            package_version="1.8.0",
            node_contracts_available=True,
        )
