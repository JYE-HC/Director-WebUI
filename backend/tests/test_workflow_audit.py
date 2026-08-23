from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from directordeck.workflow.audit import (
    FeatureAuditTrace,
    GraphAuditError,
    ResolvedNodeEmission,
    build_graph_audit_spec,
    validate_bound_graph,
)
from directordeck.workflow.contracts import (
    AllowedLateBoundInput,
    EdgeRef,
    FeatureResolution,
    ListRef,
    NodeContract,
    NodeContractEvidence,
    NodeContractRegistry,
    NodeOutputContract,
    ObjectInfoContract,
    ObjectInfoInputContract,
    ObjectInfoOutputContract,
    PublicResourceRead,
    PublicResourceWrite,
    RecordRef,
    Resource,
    ResolvedImplementationIdentity,
    RuntimeEffectContract,
    TerminalRef,
)


_FINGERPRINT = "sha256:" + "a" * 64


def _contract(
    class_type: str,
    *,
    outputs: int = 1,
    output_type: str = "VALUE",
    terminal: str | None = None,
    persistent: str | None = None,
    effect_policy: str | None = None,
) -> NodeContract:
    slots = tuple(
        ObjectInfoOutputContract(index=index, port_type=output_type)
        for index in range(outputs)
    )
    policy = effect_policy or (
        "side_effect_only" if terminal is not None else "strict_transform"
    )
    effect = RuntimeEffectContract(
        policy=policy,
        unsupported_behavior="identity" if policy == "identity_allowed" else "raise",
        validation_method="director_owned_implementation",
        verified_model_families=("fl2va",) if policy == "strict_transform" else (),
        verified_backends=("standard",) if policy == "strict_transform" else (),
    )
    if class_type == "Transform":
        required_inputs = {
            "model": ObjectInfoInputContract(port_type="VALUE"),
        }
        optional_inputs = {
            "continuity_file": ObjectInfoInputContract(port_type="STRING"),
            "continuity": ObjectInfoInputContract(port_type="VALUE"),
            "runtime_epoch": ObjectInfoInputContract(port_type="INT"),
            "continuity_slot": ObjectInfoInputContract(port_type="VALUE"),
            "quality": ObjectInfoInputContract(
                port_type="COMBO", enum_values=("strict", "balanced")
            ),
        }
    elif class_type == "BundleTransform":
        required_inputs = {
            "bundle": ObjectInfoInputContract(port_type="REFERENCE_BUNDLE"),
        }
        optional_inputs = {}
    elif class_type == "SaveVideo":
        required_inputs = {}
        optional_inputs = {
            "video": ObjectInfoInputContract(port_type="VALUE"),
        }
    else:
        required_inputs = {}
        optional_inputs = {}
    return NodeContract(
        contract_id=f"contract.{class_type}",
        semantic_version="1.0.0",
        class_type=class_type,
        allowed_python_modules=("directordeck.test_nodes",),
        object_info_contract=ObjectInfoContract(
            normalization_version=1,
            required_inputs=required_inputs,
            optional_inputs=optional_inputs,
            director_supplied_inputs=tuple((*required_inputs, *optional_inputs)),
            outputs=slots,
            output_node=terminal is not None,
        ),
        output_contract=NodeOutputContract(slots=slots),
        execution_terminal_role=terminal,
        persistent_artifact_role=persistent,
        runtime_effect_contract=effect,
        supported_runtime_fingerprints=(_FINGERPRINT,),
    )


def _registry(
    *,
    transform_policy: str = "strict_transform",
    source_outputs: int = 1,
) -> NodeContractRegistry:
    registry = NodeContractRegistry()
    for contract in (
        _contract("Source", outputs=source_outputs),
        _contract("Transform", effect_policy=transform_policy),
        _contract("SaveVideo", outputs=0, terminal="take", persistent="take"),
        _contract("DirectorDeckRayKill", outputs=0, terminal="ray_kill"),
    ):
        registry = registry.register(contract)
    return registry


def _evidence(registry: NodeContractRegistry, class_type: str) -> NodeContractEvidence:
    contract = registry.require(class_type)
    return NodeContractEvidence(
        contract_id=contract.contract_id,
        semantic_version=contract.semantic_version,
        class_type=class_type,
        python_module="directordeck.test_nodes",
        runtime_fingerprint=_FINGERPRINT,
        execution_terminal_role=contract.execution_terminal_role,
        persistent_artifact_role=contract.persistent_artifact_role,
    )


def _trace(
    feature_id: str,
    node_id: str,
    class_type: str,
    *,
    structural: bool = True,
    output_affecting: bool | None = None,
) -> FeatureAuditTrace:
    binding_key = f"{feature_id}.primary"
    return FeatureAuditTrace(
        feature_id=feature_id,
        resolution=FeatureResolution(
            state="active",
            implementations=(
                ResolvedImplementationIdentity(
                    role="primary",
                    class_type=class_type,
                    implementation_id=f"contract.{class_type}",
                    semantic_version="1.0.0",
                    runtime_fingerprint=_FINGERPRINT,
                    binding_key=binding_key,
                ),
            ),
        ),
        emitted_nodes=(
            ResolvedNodeEmission(
                node_id=node_id,
                implementation_binding_key=binding_key,
                output_affecting=(
                    class_type not in {"SaveVideo", "DirectorDeckRayKill"}
                    if output_affecting is None
                    else output_affecting
                ),
            ),
        ),
        structural_influence=structural,
    )


def _resource(
    name: str,
    node_id: str,
    feature_id: str,
    revision: int,
    *,
    terminal: bool = False,
    output_slot: int = 0,
) -> Resource:
    value = (
        TerminalRef(node_id=node_id)
        if terminal
        else EdgeRef(node_id=node_id, output_slot=output_slot)
    )
    return Resource(
        name=name,
        type="TAKE" if terminal else "VALUE",
        value=value,
        source_feature_id=feature_id,
        revision=revision,
        producer_node_ids=(node_id,),
    )


def _valid_ingredients() -> dict[str, Any]:
    registry = _registry()
    model_1 = _resource("model", "1", "source", 1)
    model_2 = _resource("model", "2", "coarse", 2)
    model_3 = _resource("model", "3", "detail", 3)
    take = _resource("take_output", "4", "save", 1, terminal=True)
    return {
        "prompt": {
            "1": {"class_type": "Source", "inputs": {}},
            "2": {"class_type": "Transform", "inputs": {"model": ["1", 0]}},
            "3": {"class_type": "Transform", "inputs": {"model": ["2", 0]}},
            "4": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0]}},
        },
        "node_contract_registry": registry,
        "node_contract_snapshot": {
            "1": _evidence(registry, "Source"),
            "2": _evidence(registry, "Transform"),
            "3": _evidence(registry, "Transform"),
            "4": _evidence(registry, "SaveVideo"),
        },
        "public_writes": [
            PublicResourceWrite(operation="define", resource=model_1),
            PublicResourceWrite(
                operation="replace", resource=model_2, previous_revision=1
            ),
            PublicResourceWrite(
                operation="replace", resource=model_3, previous_revision=2
            ),
            PublicResourceWrite(operation="define", resource=take),
        ],
        "public_reads": [
            PublicResourceRead(
                resource_name="model",
                type="VALUE",
                revision=1,
                consumer_node_id="2",
                input_pointer="/2/inputs/model",
                value=model_1.value,
            ),
            PublicResourceRead(
                resource_name="model",
                type="VALUE",
                revision=2,
                consumer_node_id="3",
                input_pointer="/3/inputs/model",
                value=model_2.value,
            ),
            PublicResourceRead(
                resource_name="model",
                type="VALUE",
                revision=3,
                consumer_node_id="4",
                input_pointer="/4/inputs/video",
                value=model_3.value,
            ),
        ],
        "feature_traces": [
            _trace("source", "1", "Source"),
            _trace("coarse", "2", "Transform"),
            _trace("detail", "3", "Transform"),
            _trace("save", "4", "SaveVideo"),
        ],
        "unit_kind": "segment",
        "take_node_id": "4",
        "model_family": "fl2va",
        "backend": "standard",
    }


def test_build_graph_audit_proves_latest_revisions_and_take_cone() -> None:
    ingredients = _valid_ingredients()

    spec = build_graph_audit_spec(**ingredients)

    assert spec.take_node_id == "4"
    assert spec.structural_influence_features == (
        "source",
        "coarse",
        "detail",
        "save",
    )
    assert [write.resource.revision for write in spec.public_writes[:3]] == [1, 2, 3]


def test_type_correct_stale_edge_is_rejected() -> None:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["3"]["inputs"]["model"] = ["1", 0]
    first_resource = ingredients["public_writes"][0].resource
    ingredients["public_reads"][1] = PublicResourceRead(
        resource_name="model",
        type="VALUE",
        revision=1,
        consumer_node_id="3",
        input_pointer="/3/inputs/model",
        value=first_resource.value,
    )

    with pytest.raises(GraphAuditError, match="latest exact revision"):
        build_graph_audit_spec(**ingredients)


def test_replace_requires_the_recorded_previous_read_to_influence_new_producer() -> None:
    ingredients = _valid_ingredients()
    registry = ingredients["node_contract_registry"]
    ingredients["prompt"]["2"]["inputs"]["model"] = ["5", 0]
    ingredients["prompt"]["5"] = {"class_type": "Source", "inputs": {}}
    ingredients["prompt"]["6"] = {
        "class_type": "Transform",
        "inputs": {"model": ["1", 0]},
    }
    ingredients["node_contract_snapshot"]["5"] = _evidence(registry, "Source")
    ingredients["node_contract_snapshot"]["6"] = _evidence(registry, "Transform")
    ingredients["public_reads"][0] = ingredients["public_reads"][0].model_copy(
        update={
            "consumer_node_id": "6",
            "input_pointer": "/6/inputs/model",
        }
    )
    transform_contract = registry.require("Transform")
    source_contract = registry.require("Source")
    ingredients["feature_traces"][1] = FeatureAuditTrace(
        feature_id="coarse",
        resolution=FeatureResolution(
            state="active",
            implementations=(
                ResolvedImplementationIdentity(
                    role="source",
                    class_type="Source",
                    implementation_id=source_contract.contract_id,
                    semantic_version=source_contract.semantic_version,
                    runtime_fingerprint=_FINGERPRINT,
                    binding_key="coarse.source",
                ),
                ResolvedImplementationIdentity(
                    role="transform",
                    class_type="Transform",
                    implementation_id=transform_contract.contract_id,
                    semantic_version=transform_contract.semantic_version,
                    runtime_fingerprint=_FINGERPRINT,
                    binding_key="coarse.transform",
                ),
            ),
        ),
        emitted_nodes=(
            ResolvedNodeEmission(
                node_id="5",
                implementation_binding_key="coarse.source",
                output_affecting=True,
            ),
            ResolvedNodeEmission(
                node_id="2",
                implementation_binding_key="coarse.transform",
                output_affecting=True,
            ),
            ResolvedNodeEmission(
                node_id="6",
                implementation_binding_key="coarse.transform",
                output_affecting=True,
            ),
        ),
        structural_influence=True,
    )

    with pytest.raises(GraphAuditError, match="do not depend on the previous revision"):
        build_graph_audit_spec(**ingredients)


def test_dead_public_write_outside_take_cone_is_rejected() -> None:
    ingredients = _valid_ingredients()
    registry = ingredients["node_contract_registry"]
    ingredients["prompt"]["5"] = {"class_type": "Source", "inputs": {}}
    ingredients["node_contract_snapshot"]["5"] = _evidence(registry, "Source")
    unused = _resource("unused", "5", "unused_feature", 1)
    ingredients["public_writes"].insert(
        -1, PublicResourceWrite(operation="define", resource=unused)
    )
    ingredients["feature_traces"].insert(
        -1, _trace("unused_feature", "5", "Source")
    )

    with pytest.raises(GraphAuditError, match="dead public write"):
        build_graph_audit_spec(**ingredients)


def test_hidden_second_persistent_save_is_rejected() -> None:
    ingredients = _valid_ingredients()
    registry = ingredients["node_contract_registry"]
    ingredients["prompt"]["5"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ["3", 0]},
    }
    ingredients["node_contract_snapshot"]["5"] = _evidence(registry, "SaveVideo")
    ingredients["feature_traces"].insert(
        -1, _trace("hidden_save", "5", "SaveVideo", structural=False)
    )

    with pytest.raises(GraphAuditError, match="matching take terminal/artifact"):
        build_graph_audit_spec(**ingredients)


def test_undeclared_output_slot_is_rejected_for_prompt_and_public_value() -> None:
    ingredients = _valid_ingredients()
    bad_source = _resource("model", "1", "source", 1, output_slot=1)
    ingredients["public_writes"][0] = PublicResourceWrite(
        operation="define", resource=bad_source
    )
    ingredients["public_reads"][0] = PublicResourceRead(
        resource_name="model",
        type="VALUE",
        revision=1,
        consumer_node_id="2",
        input_pointer="/2/inputs/model",
        value=bad_source.value,
    )
    ingredients["prompt"]["2"]["inputs"]["model"] = ["1", 1]

    with pytest.raises(GraphAuditError, match="undeclared output slot"):
        build_graph_audit_spec(**ingredients)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "missing required inputs"),
        ("unknown", "undeclared inputs"),
        ("wrong_literal_type", "does not match port type"),
        ("invalid_enum", "outside its enum"),
    ),
)
def test_object_info_input_contract_is_enforced(
    mutation: str,
    message: str,
) -> None:
    ingredients = _valid_ingredients()
    inputs = ingredients["prompt"]["2"]["inputs"]
    if mutation == "missing":
        del inputs["model"]
    elif mutation == "unknown":
        inputs["surprise"] = 1
    elif mutation == "wrong_literal_type":
        inputs["model"] = "not-a-model-edge"
    else:
        inputs["quality"] = "unsafe_fallback"

    with pytest.raises(GraphAuditError, match=message):
        build_graph_audit_spec(**ingredients)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("implementation_id", "contract.Source", "implementation id"),
        ("semantic_version", "9.9.9", "semantic version"),
    ),
)
def test_resolution_identity_and_version_must_match_node_contract(
    field: str,
    value: str,
    message: str,
) -> None:
    ingredients = _valid_ingredients()
    trace = ingredients["feature_traces"][1]
    implementation = trace.resolution.implementations[0].model_copy(
        update={field: value}
    )
    ingredients["feature_traces"][1] = trace.model_copy(
        update={
            "resolution": trace.resolution.model_copy(
                update={"implementations": (implementation,)}
            )
        }
    )

    with pytest.raises(GraphAuditError, match=message):
        build_graph_audit_spec(**ingredients)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model_family", "ref2va", "not verified for model family"),
        ("backend", "raylight", "not verified for backend"),
    ),
)
def test_runtime_effect_contract_is_bound_to_unit_context(
    field: str,
    value: str,
    message: str,
) -> None:
    ingredients = _valid_ingredients()
    ingredients[field] = value

    with pytest.raises(GraphAuditError, match=message):
        build_graph_audit_spec(**ingredients)


def test_valid_output_slot_with_wrong_port_type_is_rejected() -> None:
    ingredients = _valid_ingredients()
    registry = NodeContractRegistry()
    for contract in (
        _contract("Source", output_type="OTHER_VALUE"),
        _contract("Transform"),
        _contract("SaveVideo", outputs=0, terminal="take", persistent="take"),
        _contract("DirectorDeckRayKill", outputs=0, terminal="ray_kill"),
    ):
        registry = registry.register(contract)
    ingredients["node_contract_registry"] = registry
    ingredients["node_contract_snapshot"] = {
        node_id: _evidence(registry, ingredients["prompt"][node_id]["class_type"])
        for node_id in ingredients["prompt"]
    }

    with pytest.raises(GraphAuditError, match="has port type.*expected"):
        build_graph_audit_spec(**ingredients)


def test_public_resource_type_must_match_its_scalar_producer_slot() -> None:
    ingredients = _valid_ingredients()
    wrong = ingredients["public_writes"][0].resource.model_copy(
        update={"type": "OTHER_VALUE"}
    )
    ingredients["public_writes"][0] = PublicResourceWrite(
        operation="define", resource=wrong
    )
    ingredients["public_reads"][0] = ingredients["public_reads"][0].model_copy(
        update={"type": "OTHER_VALUE"}
    )

    with pytest.raises(GraphAuditError, match="declares type.*producer slot"):
        build_graph_audit_spec(**ingredients)


def test_composite_public_resource_is_a_single_exact_typed_read() -> None:
    registry = NodeContractRegistry()
    for contract in (
        _contract("Source"),
        _contract("BundleTransform"),
        _contract("SaveVideo", outputs=0, terminal="take", persistent="take"),
    ):
        registry = registry.register(contract)
    bundle_value = RecordRef(
        fields={
            "first": EdgeRef(node_id="1", output_slot=0),
            "others": ListRef(
                items=(EdgeRef(node_id="2", output_slot=0),)
            ),
        }
    )
    bundle = Resource(
        name="reference_bundle",
        type="REFERENCE_BUNDLE",
        value=bundle_value,
        source_feature_id="sources",
        revision=1,
        producer_node_ids=("1", "2"),
    )
    model = _resource("model", "3", "bundle_transform", 1)
    take = _resource("take_output", "4", "save", 1, terminal=True)
    source_contract = registry.require("Source")
    source_resolution = FeatureResolution(
        state="active",
        implementations=(
            ResolvedImplementationIdentity(
                role="primary",
                class_type="Source",
                implementation_id=source_contract.contract_id,
                semantic_version=source_contract.semantic_version,
                runtime_fingerprint=_FINGERPRINT,
                binding_key="sources.primary",
            ),
        ),
    )
    source_trace = FeatureAuditTrace(
        feature_id="sources",
        resolution=source_resolution,
        emitted_nodes=(
            ResolvedNodeEmission(
                node_id="1",
                implementation_binding_key="sources.primary",
                output_affecting=True,
            ),
            ResolvedNodeEmission(
                node_id="2",
                implementation_binding_key="sources.primary",
                output_affecting=True,
            ),
        ),
        structural_influence=True,
    )
    prompt = {
        "1": {"class_type": "Source", "inputs": {}},
        "2": {"class_type": "Source", "inputs": {}},
        "3": {
            "class_type": "BundleTransform",
            "inputs": {
                "bundle": {
                    "first": ["1", 0],
                    "others": [["2", 0]],
                }
            },
        },
        "4": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0]}},
    }

    spec = build_graph_audit_spec(
        prompt=prompt,
        node_contract_registry=registry,
        node_contract_snapshot={
            node_id: _evidence(registry, node["class_type"])
            for node_id, node in prompt.items()
        },
        public_writes=(
            PublicResourceWrite(operation="define", resource=bundle),
            PublicResourceWrite(operation="define", resource=model),
            PublicResourceWrite(operation="define", resource=take),
        ),
        public_reads=(
            PublicResourceRead(
                resource_name=bundle.name,
                type=bundle.type,
                revision=1,
                consumer_node_id="3",
                input_pointer="/3/inputs/bundle",
                value=bundle.value,
            ),
            PublicResourceRead(
                resource_name=model.name,
                type=model.type,
                revision=1,
                consumer_node_id="4",
                input_pointer="/4/inputs/video",
                value=model.value,
            ),
        ),
        feature_traces=(
            source_trace,
            _trace("bundle_transform", "3", "BundleTransform"),
            _trace("save", "4", "SaveVideo"),
        ),
        model_family="fl2va",
        backend="standard",
        unit_kind="segment",
        take_node_id="4",
    )

    assert spec.public_reads[0].value == bundle_value


def test_resolution_and_emitted_class_must_align_bidirectionally() -> None:
    ingredients = _valid_ingredients()
    ingredients["feature_traces"][1] = _trace("coarse", "2", "Source")

    with pytest.raises(GraphAuditError, match="emitted class"):
        build_graph_audit_spec(**ingredients)


def test_runtime_effect_is_checked_per_emitted_binding_not_whole_feature() -> None:
    ingredients = _valid_ingredients()
    identity_registry = _registry(transform_policy="identity_allowed")
    ingredients["node_contract_registry"] = identity_registry
    ingredients["node_contract_snapshot"]["2"] = _evidence(
        identity_registry, "Transform"
    )
    ingredients["node_contract_snapshot"]["3"] = _evidence(
        identity_registry, "Transform"
    )

    with pytest.raises(GraphAuditError, match="fail-closed strict_transform"):
        build_graph_audit_spec(**ingredients)

    # Compile-only compatibility skips only the effect policy.  All other
    # evidence is unchanged and remains validated.
    build_graph_audit_spec(**ingredients, enforce_runtime_effects=False)

    ingredients["feature_traces"][1] = _trace(
        "coarse", "2", "Transform", output_affecting=False
    )
    ingredients["feature_traces"][2] = _trace(
        "detail", "3", "Transform", output_affecting=False
    )
    build_graph_audit_spec(**ingredients)


def test_raylight_control_has_one_terminal_and_no_artifact() -> None:
    registry = _registry()
    prompt = {"1": {"class_type": "DirectorDeckRayKill", "inputs": {}}}

    spec = build_graph_audit_spec(
        prompt=prompt,
        node_contract_registry=registry,
        node_contract_snapshot={"1": _evidence(registry, "DirectorDeckRayKill")},
        public_writes=(),
        public_reads=(),
        feature_traces=(
            _trace("ray_kill", "1", "DirectorDeckRayKill", structural=False),
        ),
        model_family="fl2va",
        backend="raylight",
        unit_kind="control",
        control_kind="ray_kill",
    )

    assert spec.control_kind == "ray_kill"
    assert spec.take_node_id is None
    assert spec.public_writes == ()


def test_control_rejects_hidden_media_artifact() -> None:
    registry = _registry()
    prompt = {
        "1": {"class_type": "DirectorDeckRayKill", "inputs": {}},
        "2": {"class_type": "SaveVideo", "inputs": {}},
    }

    with pytest.raises(GraphAuditError, match="RayKill terminal and no artifact"):
        build_graph_audit_spec(
            prompt=prompt,
            node_contract_registry=registry,
            node_contract_snapshot={
                "1": _evidence(registry, "DirectorDeckRayKill"),
                "2": _evidence(registry, "SaveVideo"),
            },
            public_writes=(),
            public_reads=(),
            feature_traces=(
                _trace("ray_kill", "1", "DirectorDeckRayKill", structural=False),
                _trace("hidden_save", "2", "SaveVideo", structural=False),
            ),
            model_family="fl2va",
            backend="raylight",
            unit_kind="control",
            control_kind="ray_kill",
        )


def _continuity_audit() -> tuple[dict[str, Any], Any, dict[str, Any]]:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["3"]["inputs"]["continuity_file"] = "__unbound__"
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/3/inputs/continuity_file",
            value_kind="string",
            source_kind="continuity",
        ),
    )
    spec = build_graph_audit_spec(**ingredients)
    return ingredients["prompt"], spec, ingredients


def test_late_bound_continuity_changes_only_whitelisted_pointer() -> None:
    prompt_base, spec, ingredients = _continuity_audit()
    bound = deepcopy(prompt_base)
    bound["3"]["inputs"]["continuity_file"] = "take/previous.mp4"

    validate_bound_graph(
        prompt_base=prompt_base,
        bound_prompt=bound,
        spec=spec,
        node_contract_registry=ingredients["node_contract_registry"],
        model_family="fl2va",
        backend="standard",
        feature_traces=ingredients["feature_traces"],
        expected_late_bound_values={
            "/3/inputs/continuity_file": "take/previous.mp4"
        },
    )


def test_late_bound_kind_must_match_the_node_input_contract() -> None:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["3"]["inputs"]["continuity_file"] = "__unbound__"
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/3/inputs/continuity_file",
            value_kind="integer",
            source_kind="continuity",
        ),
    )

    with pytest.raises(GraphAuditError, match="incompatible with input port type 'STRING'"):
        build_graph_audit_spec(**ingredients)


def test_resource_late_binding_uses_typed_ref_for_base_cone_and_exact_bound_value() -> None:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["2"]["inputs"]["model"] = "__unbound_resource__"
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/2/inputs/model",
            value_kind="edge",
            source_kind="resource",
            resource_name="model",
            revision=1,
        ),
    )
    spec = build_graph_audit_spec(**ingredients)
    bound = deepcopy(ingredients["prompt"])
    bound["2"]["inputs"]["model"] = ["1", 0]

    with pytest.raises(GraphAuditError, match="requires feature audit traces"):
        validate_bound_graph(
            prompt_base=ingredients["prompt"],
            bound_prompt=bound,
            spec=spec,
            node_contract_registry=ingredients["node_contract_registry"],
            model_family="fl2va",
            backend="standard",
        )
    validate_bound_graph(
        prompt_base=ingredients["prompt"],
        bound_prompt=bound,
        spec=spec,
        node_contract_registry=ingredients["node_contract_registry"],
        model_family="fl2va",
        backend="standard",
        enforce_runtime_effects=False,
    )


def test_typed_continuity_edge_does_not_masquerade_as_public_resource_read() -> None:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["3"]["inputs"]["continuity"] = "__unbound__"
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/3/inputs/continuity",
            value_kind="edge",
            source_kind="continuity",
        ),
    )
    spec = build_graph_audit_spec(**ingredients)
    bound = deepcopy(ingredients["prompt"])
    bound["3"]["inputs"]["continuity"] = ["1", 0]

    validate_bound_graph(
        prompt_base=ingredients["prompt"],
        bound_prompt=bound,
        spec=spec,
        node_contract_registry=ingredients["node_contract_registry"],
        model_family="fl2va",
        backend="standard",
        feature_traces=ingredients["feature_traces"],
        expected_late_bound_values={"/3/inputs/continuity": ["1", 0]},
    )


def test_runtime_epoch_and_unbound_continuity_can_materialize_incrementally() -> None:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["3"]["inputs"].update(
        continuity_file="__unbound_continuity__",
        runtime_epoch=0,
    )
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/3/inputs/continuity_file",
            value_kind="string",
            source_kind="continuity",
        ),
        AllowedLateBoundInput(
            input_pointer="/3/inputs/runtime_epoch",
            value_kind="integer",
            source_kind="runtime_epoch",
        ),
    )
    spec = build_graph_audit_spec(**ingredients)

    epoch_bound = deepcopy(ingredients["prompt"])
    epoch_bound["3"]["inputs"]["runtime_epoch"] = 11
    with pytest.raises(GraphAuditError, match="missing typed expected value"):
        validate_bound_graph(
            prompt_base=ingredients["prompt"],
            bound_prompt=epoch_bound,
            spec=spec,
            node_contract_registry=ingredients["node_contract_registry"],
            model_family="fl2va",
            backend="standard",
            feature_traces=ingredients["feature_traces"],
        )
    validate_bound_graph(
        prompt_base=ingredients["prompt"],
        bound_prompt=epoch_bound,
        spec=spec,
        node_contract_registry=ingredients["node_contract_registry"],
        model_family="fl2va",
        backend="standard",
        feature_traces=ingredients["feature_traces"],
        expected_late_bound_values={"/3/inputs/runtime_epoch": 11},
    )
    assert epoch_bound["3"]["inputs"]["continuity_file"] == (
        "__unbound_continuity__"
    )

    continuity_bound = deepcopy(epoch_bound)
    continuity_bound["3"]["inputs"]["continuity_file"] = "take/previous.mp4"
    validate_bound_graph(
        prompt_base=epoch_bound,
        bound_prompt=continuity_bound,
        spec=spec,
        node_contract_registry=ingredients["node_contract_registry"],
        model_family="fl2va",
        backend="standard",
        feature_traces=ingredients["feature_traces"],
        expected_late_bound_values={
            "/3/inputs/continuity_file": "take/previous.mp4",
            "/3/inputs/runtime_epoch": 11,
        },
    )


def test_incremental_binding_rejects_unproven_prior_pointer_mutation() -> None:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["3"]["inputs"].update(
        continuity_file="__unbound_continuity__",
        runtime_epoch=0,
    )
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/3/inputs/continuity_file",
            value_kind="string",
            source_kind="continuity",
        ),
        AllowedLateBoundInput(
            input_pointer="/3/inputs/runtime_epoch",
            value_kind="integer",
            source_kind="runtime_epoch",
        ),
    )
    spec = build_graph_audit_spec(**ingredients)
    poisoned_base = deepcopy(ingredients["prompt"])
    poisoned_base["3"]["inputs"]["continuity_file"] = "evil.mp4"
    epoch_bound = deepcopy(poisoned_base)
    epoch_bound["3"]["inputs"]["runtime_epoch"] = 11

    with pytest.raises(GraphAuditError, match="missing typed expected value"):
        validate_bound_graph(
            prompt_base=poisoned_base,
            bound_prompt=epoch_bound,
            spec=spec,
            node_contract_registry=ingredients["node_contract_registry"],
            model_family="fl2va",
            backend="standard",
            feature_traces=ingredients["feature_traces"],
            expected_late_bound_values={"/3/inputs/runtime_epoch": 11},
        )


def test_materialized_combo_late_binding_rejects_out_of_enum_value() -> None:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["3"]["inputs"]["quality"] = "__unbound__"
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/3/inputs/quality",
            value_kind="string",
            source_kind="continuity",
        ),
    )
    spec = build_graph_audit_spec(**ingredients)
    bound = deepcopy(ingredients["prompt"])
    bound["3"]["inputs"]["quality"] = "totally-unsupported"

    with pytest.raises(GraphAuditError, match="outside its enum"):
        validate_bound_graph(
            prompt_base=ingredients["prompt"],
            bound_prompt=bound,
            spec=spec,
            node_contract_registry=ingredients["node_contract_registry"],
            model_family="fl2va",
            backend="standard",
            feature_traces=ingredients["feature_traces"],
            expected_late_bound_values={
                "/3/inputs/quality": "totally-unsupported"
            },
        )


@pytest.mark.parametrize("mutation", ["node_set", "class_type", "non_whitelist"])
def test_late_binding_rejects_structural_or_non_whitelisted_changes(
    mutation: str,
) -> None:
    prompt_base, spec, ingredients = _continuity_audit()
    bound = deepcopy(prompt_base)
    bound["3"]["inputs"]["continuity_file"] = "take/previous.mp4"
    if mutation == "node_set":
        bound["9"] = {"class_type": "Source", "inputs": {}}
    elif mutation == "class_type":
        bound["2"]["class_type"] = "Source"
    else:
        bound["3"]["inputs"]["model"] = ["1", 0]

    with pytest.raises(GraphAuditError, match="outside whitelist|non-whitelisted"):
        validate_bound_graph(
            prompt_base=prompt_base,
            bound_prompt=bound,
            spec=spec,
            node_contract_registry=ingredients["node_contract_registry"],
            model_family="fl2va",
            backend="standard",
            feature_traces=ingredients["feature_traces"],
            expected_late_bound_values={
                "/3/inputs/continuity_file": "take/previous.mp4"
            },
        )


def test_late_binding_output_slot_still_uses_node_contract() -> None:
    ingredients = _valid_ingredients()
    ingredients["prompt"]["3"]["inputs"]["continuity_slot"] = "__unbound__"
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/3/inputs/continuity_slot",
            value_kind="edge",
            source_kind="continuity",
        ),
    )
    spec = build_graph_audit_spec(**ingredients)
    bound = deepcopy(ingredients["prompt"])
    bound["3"]["inputs"]["continuity_slot"] = ["2", 1]

    with pytest.raises(GraphAuditError, match="undeclared output slot"):
        validate_bound_graph(
            prompt_base=ingredients["prompt"],
            bound_prompt=bound,
            spec=spec,
            node_contract_registry=ingredients["node_contract_registry"],
            model_family="fl2va",
            backend="standard",
            feature_traces=ingredients["feature_traces"],
            expected_late_bound_values={"/3/inputs/continuity_slot": ["2", 1]},
        )


def test_late_binding_cannot_switch_between_two_declared_output_slots() -> None:
    ingredients = _valid_ingredients()
    registry = _registry(source_outputs=2)
    ingredients["node_contract_registry"] = registry
    ingredients["node_contract_snapshot"]["1"] = _evidence(registry, "Source")
    ingredients["prompt"]["3"]["inputs"]["continuity"] = ["1", 0]
    ingredients["allowed_late_bound_inputs"] = (
        AllowedLateBoundInput(
            input_pointer="/3/inputs/continuity",
            value_kind="edge",
            source_kind="continuity",
        ),
    )
    spec = build_graph_audit_spec(**ingredients)
    bound = deepcopy(ingredients["prompt"])
    bound["3"]["inputs"]["continuity"] = ["1", 1]

    with pytest.raises(GraphAuditError, match="changed an existing edge output slot"):
        validate_bound_graph(
            prompt_base=ingredients["prompt"],
            bound_prompt=bound,
            spec=spec,
            node_contract_registry=registry,
            model_family="fl2va",
            backend="standard",
            feature_traces=ingredients["feature_traces"],
            expected_late_bound_values={"/3/inputs/continuity": ["1", 1]},
        )
