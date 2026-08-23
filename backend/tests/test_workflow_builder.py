from __future__ import annotations

from copy import deepcopy

import pytest

from directordeck.workflow.builder import GraphBuilderError, PromptGraphBuilder
from directordeck.workflow.contracts import (
    EdgeRef,
    FeatureEmission,
    ListRef,
    RecordRef,
    ResourcePool,
    TerminalRef,
)


def _define(
    pool: ResourcePool,
    *,
    name: str,
    type: str,
    value: EdgeRef | TerminalRef | ListRef | RecordRef,
    feature_id: str,
):
    producer_ids: list[str] = []

    def collect(item: EdgeRef | TerminalRef | ListRef | RecordRef) -> None:
        if isinstance(item, (EdgeRef, TerminalRef)):
            if item.node_id not in producer_ids:
                producer_ids.append(item.node_id)
        elif isinstance(item, ListRef):
            for child in item.items:
                collect(child)
        else:
            for child in item.fields.values():
                collect(child)

    collect(value)
    return pool.begin().define(
        name=name,
        type=type,
        value=value,
        source_feature_id=feature_id,
        producer_node_ids=producer_ids,
    )


def test_scopes_commit_plain_prompt_and_continuous_global_node_ids() -> None:
    graph = PromptGraphBuilder()
    pool = ResourcePool()

    load = graph.begin_scope("model_load")
    node_1 = load.add_node(
        "UNETLoader",
        {"unet_name": "model.safetensors", "options": {"dtype": "default"}},
    )
    model_v1 = load.edge(node_1)
    pool = load.commit(
        public_outputs={"model": model_v1},
        resource_transaction=_define(
            pool,
            name="model",
            type="MODEL",
            value=model_v1,
            feature_id="model_load",
        ),
    )
    assert pool is not None

    patch = graph.begin_scope("model_patch")
    node_2 = patch.add_node(
        "PatchModel",
        model=pool.read_required("model", expected_type="MODEL").value,
        options=["strict", {"scale": 1.0}],
    )
    model_v2 = patch.edge(node_2)
    pool = patch.commit(
        public_outputs={"model": model_v2},
        resource_transaction=pool.begin().replace(
            name="model",
            expected_type="MODEL",
            expected_revision=1,
            value=model_v2,
            source_feature_id="model_patch",
            producer_node_ids=(node_2,),
        ),
    )
    assert pool is not None
    assert pool.resources["model"].revision == 2
    assert graph.prompt == {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": "model.safetensors",
                "options": {"dtype": "default"},
            },
        },
        "2": {
            "class_type": "PatchModel",
            "inputs": {
                "model": ["1", 0],
                "options": ["strict", {"scale": 1.0}],
            },
        },
    }
    assert all(type(value) is dict for value in graph.prompt.values())
    assert type(graph.prompt["2"]["inputs"]["model"]) is list


def test_scope_rollback_and_context_exit_restore_the_node_counter() -> None:
    graph = PromptGraphBuilder()

    abandoned = graph.begin_scope("abandoned")
    assert abandoned.add_node("First", {}) == "1"
    assert abandoned.add_node("Second", {}) == "2"
    abandoned.rollback()
    assert graph.prompt == {}

    with graph.begin_scope("context") as context:
        assert context.add_node("Reused", {}) == "1"
    assert graph.prompt == {}

    committed = graph.begin_scope("committed")
    assert committed.add_node("Stable", {}) == "1"
    committed.commit()
    assert tuple(graph.prompt) == ("1",)


def test_failed_atomic_commit_discards_graph_and_does_not_consume_id() -> None:
    graph = PromptGraphBuilder()
    base = ResourcePool()
    scope = graph.begin_scope("wrong_feature")
    node_id = scope.add_node("Producer", {})
    output = scope.edge(node_id)
    wrong_transaction = _define(
        base,
        name="model",
        type="MODEL",
        value=output,
        feature_id="some_other_feature",
    )

    with pytest.raises(GraphBuilderError, match="wrong source feature"):
        scope.commit(
            public_outputs={"model": output},
            resource_transaction=wrong_transaction,
        )

    assert graph.prompt == {}
    assert base.resources == {}
    retry = graph.begin_scope("retry")
    assert retry.add_node("Producer", {}) == "1"
    retry.rollback()


def test_resource_delta_and_public_outputs_must_match_exactly() -> None:
    graph = PromptGraphBuilder()
    base = ResourcePool()
    scope = graph.begin_scope("producer")
    node_id = scope.add_node("Producer", {})
    edge = scope.edge(node_id)
    transaction = _define(
        base,
        name="model",
        type="MODEL",
        value=edge,
        feature_id="producer",
    )

    with pytest.raises(GraphBuilderError, match="delta must exactly match"):
        scope.commit(public_outputs={}, resource_transaction=transaction)

    assert graph.prompt == {}

    scope = graph.begin_scope("producer")
    node_id = scope.add_node("Producer", {})
    edge = scope.edge(node_id)
    with pytest.raises(GraphBuilderError, match="require a resource-pool"):
        scope.commit(public_outputs={"model": edge})
    assert graph.prompt == {}


def test_public_outputs_must_recursively_originate_in_current_scope() -> None:
    graph = PromptGraphBuilder()
    first = graph.begin_scope("first")
    node_1 = first.add_node("Existing", {})
    first.commit()

    second = graph.begin_scope("second")
    node_2 = second.add_node("New", {})
    mixed = RecordRef(
        fields={
            "current": second.edge(node_2),
            "old": ListRef(items=(second.edge(node_1),)),
        }
    )
    with pytest.raises(GraphBuilderError, match="does not originate"):
        second.validate_public_outputs({"mixed": mixed})
    second.rollback()

    valid = graph.begin_scope("valid")
    node_2 = valid.add_node("A", {})
    node_3 = valid.add_node("B", source=valid.edge(node_2))
    output = RecordRef(
        fields={
            "a": valid.edge(node_2),
            "nested": ListRef(items=(valid.edge(node_3, 1),)),
        }
    )
    pool = valid.commit(
        public_outputs={"bundle": output},
        resource_transaction=_define(
            ResourcePool(),
            name="bundle",
            type="BUNDLE",
            value=output,
            feature_id="valid",
        ),
    )
    assert pool is not None
    assert pool.resources["bundle"].producer_node_ids == ("2", "3")


def test_composite_resource_is_lowered_with_exact_leaf_edge_evidence() -> None:
    graph = PromptGraphBuilder()
    producer = graph.begin_scope("producer")
    image = producer.add_node("ImageSource", {})
    audio = producer.add_node("AudioSource", {})
    bundle = RecordRef(
        fields={
            "image": producer.edge(image),
            "tracks": ListRef(items=(producer.edge(audio, 1),)),
        }
    )
    pool = producer.commit(
        public_outputs={"reference_bundle": bundle},
        resource_transaction=_define(
            ResourcePool(),
            name="reference_bundle",
            type="REFERENCE_BUNDLE",
            value=bundle,
            feature_id="producer",
        ),
    )
    assert pool is not None

    consumer = graph.begin_scope("consumer")
    consumer_id = consumer.add_node(
        "BundleConsumer",
        {"bundle": pool.read_required(
            "reference_bundle", expected_type="REFERENCE_BUNDLE"
        ).value},
    )

    assert consumer.prompt_fragment[consumer_id]["inputs"]["bundle"] == {
        "image": [image, 0],
        "tracks": [[audio, 1]],
    }
    assert [item.input_pointer for item in consumer.input_edge_evidence] == [
        f"/{consumer_id}/inputs/bundle/image",
        f"/{consumer_id}/inputs/bundle/tracks/0",
    ]
    consumer.rollback()


def test_terminals_are_owned_by_scope_and_can_never_feed_downstream() -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("save_take")
    save_id = scope.add_node("SaveVideo", {"filename_prefix": "take"})
    terminal = scope.terminal(save_id)

    with pytest.raises(GraphBuilderError, match="terminal node cannot publish"):
        scope.edge(save_id)
    with pytest.raises(GraphBuilderError, match="cannot be consumed downstream"):
        scope.add_node("Consumer", source=terminal)

    pool = scope.commit(
        public_outputs={"take_output": terminal},
        resource_transaction=_define(
            ResourcePool(),
            name="take_output",
            type="TAKE",
            value=terminal,
            feature_id="save_take",
        ),
    )
    assert pool is not None
    assert graph.terminal_node_ids == ("1",)

    next_scope = graph.begin_scope("after_save")
    with pytest.raises(GraphBuilderError, match="terminal node cannot publish"):
        next_scope.edge(save_id)
    next_scope.rollback()


def test_consumed_node_cannot_later_be_reclassified_as_terminal() -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("feature")
    producer = scope.add_node("Producer", {})
    scope.add_node("Consumer", value=scope.edge(producer))
    with pytest.raises(GraphBuilderError, match="already consumed"):
        scope.terminal(producer)
    scope.rollback()


def test_unknown_and_raw_edges_are_rejected_fail_closed() -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("feature")

    with pytest.raises(GraphBuilderError, match="unknown node"):
        scope.add_node("Consumer", value=EdgeRef(node_id="999", output_slot=0))
    with pytest.raises(GraphBuilderError, match="raw edge-shaped"):
        scope.add_node("Consumer", value=["999", 0])
    with pytest.raises(GraphBuilderError, match="raw edge-shaped"):
        scope.add_node("Consumer", value=("literal", 3))
    scope.rollback()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (-0.0, "negative zero"),
        (9_007_199_254_740_992, "safe range"),
        ("\ud800", "lone surrogate"),
        ({"\ud800": 1}, "key is not JSON-safe"),
    ],
)
def test_node_inputs_share_the_strict_canonical_json_domain(
    value: object,
    message: str,
) -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("feature")
    with pytest.raises(GraphBuilderError, match=message):
        scope.add_node("Invalid", value=value)
    scope.rollback()


def test_node_input_cycles_are_rejected() -> None:
    graph = PromptGraphBuilder()
    cycle: list[object] = []
    cycle.append(cycle)
    scope = graph.begin_scope("feature")
    with pytest.raises(GraphBuilderError, match="reference cycles"):
        scope.add_node("Invalid", value=cycle)
    scope.rollback()


def test_only_one_scope_can_be_active_and_closed_scopes_cannot_mutate() -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("one")
    with pytest.raises(GraphBuilderError, match="already active"):
        graph.begin_scope("two")
    scope.rollback()
    with pytest.raises(GraphBuilderError, match="already closed"):
        scope.add_node("Late", {})


def test_prompt_and_fragment_snapshots_are_detached() -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("feature")
    scope.add_node("Node", nested={"items": [1, 2, 3]})
    fragment = scope.prompt_fragment
    fragment["1"]["inputs"]["nested"]["items"].append(4)
    scope.commit()
    prompt = graph.prompt
    prompt["1"]["inputs"]["nested"]["items"].append(5)
    assert graph.prompt["1"]["inputs"]["nested"]["items"] == [1, 2, 3]


def test_commit_emission_uses_the_same_atomic_provenance_gate() -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("save_take")
    save = scope.add_node("SaveVideo", video="value")
    output = scope.terminal(save)
    emission = FeatureEmission(outputs={"take_output": output})
    transaction = _define(
        ResourcePool(),
        name="take_output",
        type="TAKE",
        value=output,
        feature_id="save_take",
    )
    committed = scope.commit_emission(emission, transaction)
    assert committed.resources["take_output"].value == output


def test_failed_add_node_does_not_change_prompt_fragment() -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("feature")
    before = deepcopy(scope.prompt_fragment)
    with pytest.raises(GraphBuilderError):
        scope.add_node("Invalid", value=["raw", 0])
    assert scope.prompt_fragment == before
    assert scope.add_node("Valid", {}) == "1"
    scope.rollback()


def test_failed_add_node_rolls_back_input_edge_evidence() -> None:
    graph = PromptGraphBuilder()
    scope = graph.begin_scope("feature")
    producer = scope.add_node("Producer", {})
    edge = scope.edge(producer)
    with pytest.raises(GraphBuilderError, match="negative zero"):
        scope.add_node("Invalid", first=edge, later=-0.0)
    assert scope.input_edges == ()
    # The failed consumer never existed, so the producer can still be terminal.
    assert scope.terminal(producer) == TerminalRef(node_id=producer)
    scope.rollback()


def test_input_edge_evidence_records_consumer_and_escaped_json_pointer() -> None:
    graph = PromptGraphBuilder()
    with graph.begin_scope("producer") as scope:
        producer = scope.add_node("Producer", {})
        scope.commit()

    with graph.begin_scope("consumer") as scope:
        scope.add_node(
            "Consumer",
            {
                "nested/key~": {
                    "items": [scope.edge(producer, 2)],
                },
            },
        )
        evidence = scope.input_edge_evidence
        assert len(evidence) == 1
        assert evidence[0].feature_id == "consumer"
        assert evidence[0].consumer_node_id == "2"
        assert evidence[0].input_pointer == (
            "/2/inputs/nested~1key~0/items/0"
        )
        assert evidence[0].value == EdgeRef(node_id="1", output_slot=2)
        scope.commit()

    assert graph.input_edge_evidence == evidence
    assert graph.node_feature_ids == {"1": "producer", "2": "consumer"}


def test_rolled_back_scope_does_not_publish_audit_evidence() -> None:
    graph = PromptGraphBuilder()
    with graph.begin_scope("producer") as scope:
        producer = scope.add_node("Producer", {})
        scope.commit()

    with graph.begin_scope("rolled_back") as scope:
        scope.add_node("Consumer", model=scope.edge(producer))

    assert graph.prompt == {
        "1": {"class_type": "Producer", "inputs": {}},
    }
    assert graph.input_edge_evidence == ()
    assert graph.node_feature_ids == {"1": "producer"}
