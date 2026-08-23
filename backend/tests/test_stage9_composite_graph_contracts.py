from __future__ import annotations

"""Stage-9 acceptance tests for composite graphs, without a business feature.

Every node contract, template entry and interpreter in this module is synthetic
and test-local.  The fixture proves that the workflow contracts can represent a
private, multi-sampler subgraph while still exposing exactly one segment take.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from directordeck.database import Database
from directordeck.execution.submission import LockedSubmissionPlanner
from directordeck.native_templates import NativeWorkflowUnit
from directordeck.progress import (
    ComfyExecutionEvent,
    ComfyPreviewEvent,
    ComfyProgressEvent,
    LivePreviewCache,
    ResolvedPreviewSource,
    child_execution_snapshot,
    child_progress_snapshot,
    preview_phase_index_for_event,
    preview_source_for_node,
)
from directordeck.workflow.audit import (
    FeatureAuditTrace,
    GraphAuditError,
    build_graph_audit_spec,
    collect_prompt_input_edges,
)
from directordeck.workflow.builder import PromptGraphBuilder
from directordeck.workflow.contracts import (
    CapabilitySet,
    EdgeRef,
    FeatureEmission,
    FeatureResolution,
    FeatureTemplateEntry,
    NodeContract,
    NodeContractEvidence,
    NodeContractRegistry,
    NodeOutputContract,
    ObjectInfoContract,
    ObjectInfoInputContract,
    ObjectInfoOutputContract,
    PublicResourceRead,
    PublicResourceWrite,
    Resource,
    ResourcePool,
    ResourceReadDeclaration,
    ResourceWriteDeclaration,
    ResolvedImplementationIdentity,
    RuntimeEffectContract,
    SegmentTemplate,
    TerminalRef,
)
from directordeck.workflow.execution import (
    CompiledExecutionPlan,
    EndpointIdentity,
    ExpectedOutputSpec,
    PreparedSegmentUnit,
    RuntimeRequirements,
    canonical_json,
    derive_feature_execution_specs,
    sha256_document_digest,
)
from directordeck.workflow.registry import FeatureInterpreterRegistry
from directordeck.workflow.v4_compiler import (
    _commit_emission,
    _node_contract_snapshot,
    _read_resources,
    _scope_public_reads,
    _scope_public_writes,
    _scope_trace_parts,
)


_FINGERPRINT = "sha256:" + "9" * 64
_MODULE = "directordeck.tests.synthetic_stage9_nodes"
_SEGMENT_ID = "stage9-segment"


class _NoParams(BaseModel):
    pass


def _phase(
    phase_id: str,
    label: str,
    node_id: str,
    kind: str,
    weight: float,
) -> dict[str, Any]:
    return {
        "id": phase_id,
        "label": label,
        "node_id": node_id,
        "kind": kind,
        "weight": weight,
    }


def _preview(
    node_id: str,
    phase_id: str,
    *,
    publish: bool,
    priority: int,
    supersedes: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "phase_id": phase_id,
        "publish": publish,
        "priority": priority,
        "supersedes": list(supersedes),
    }


def _output_slots(*types: str) -> tuple[ObjectInfoOutputContract, ...]:
    return tuple(
        ObjectInfoOutputContract(index=index, port_type=port_type)
        for index, port_type in enumerate(types)
    )


def _node_contract(
    class_type: str,
    *,
    required: Mapping[str, str] | None = None,
    optional: Mapping[str, str] | None = None,
    outputs: tuple[str, ...] = (),
    take: bool = False,
) -> NodeContract:
    output_slots = _output_slots(*outputs)
    required_inputs = {
        name: ObjectInfoInputContract(port_type=port_type)
        for name, port_type in (required or {}).items()
    }
    optional_inputs = {
        name: ObjectInfoInputContract(port_type=port_type)
        for name, port_type in (optional or {}).items()
    }
    effect = RuntimeEffectContract(
        policy="side_effect_only" if take else "strict_transform",
        unsupported_behavior="raise",
        validation_method="director_owned_implementation",
        verified_model_families=() if take else ("fl2va",),
        verified_backends=() if take else ("standard",),
    )
    return NodeContract(
        contract_id=f"test.stage9.{class_type}",
        semantic_version="1.0.0",
        class_type=class_type,
        allowed_python_modules=(_MODULE,),
        object_info_contract=ObjectInfoContract(
            normalization_version=1,
            required_inputs=required_inputs,
            optional_inputs=optional_inputs,
            director_supplied_inputs=tuple((*required_inputs, *optional_inputs)),
            outputs=output_slots,
            output_node=take,
        ),
        output_contract=NodeOutputContract(slots=output_slots),
        execution_terminal_role="take" if take else None,
        persistent_artifact_role="take" if take else None,
        runtime_effect_contract=effect,
        supported_runtime_fingerprints=(_FINGERPRINT,),
    )


def _node_registry() -> NodeContractRegistry:
    registry = NodeContractRegistry()
    for contract in (
        _node_contract("SyntheticLatentSource", outputs=("LATENT",)),
        _node_contract(
            "SyntheticSampler",
            required={"latent": "LATENT"},
            outputs=("LATENT",),
        ),
        _node_contract(
            "SyntheticDecode",
            required={"samples": "LATENT"},
            optional={"private_leak": "IMAGE"},
            outputs=("IMAGE",),
        ),
        _node_contract(
            "SyntheticTransform",
            required={"images": "IMAGE"},
            outputs=("IMAGE",),
        ),
        _node_contract(
            "SyntheticEncode",
            required={"images": "IMAGE"},
            outputs=("LATENT",),
        ),
        _node_contract(
            "SyntheticAssemble",
            required={"images": "IMAGE"},
            outputs=("VIDEO",),
        ),
        _node_contract("SaveVideo", required={"video": "VIDEO"}, take=True),
    ):
        registry = registry.register(contract)
    return registry


def _implementation(
    feature_id: str,
    class_type: str,
) -> ResolvedImplementationIdentity:
    return ResolvedImplementationIdentity(
        role=class_type,
        class_type=class_type,
        implementation_id=f"test.stage9.{class_type}",
        semantic_version="1.0.0",
        runtime_fingerprint=_FINGERPRINT,
        binding_key=f"{feature_id}.{class_type}",
    )


class _SyntheticInterpreter:
    version = 1

    def __init__(self, feature_id: str, *, hidden_second_save: bool = False) -> None:
        self.id = feature_id
        self.hidden_second_save = hidden_second_save

    def validate_params(self, params: BaseModel, ctx: Any) -> None:
        assert isinstance(params, _NoParams)
        assert ctx.backend == "standard" and ctx.family == "fl2va"

    def resolve(self, params: BaseModel, ctx: Any) -> FeatureResolution:
        self.validate_params(params, ctx)
        classes = {
            "synthetic_source": ("SyntheticLatentSource",),
            "synthetic_composite": (
                "SyntheticSampler",
                "SyntheticDecode",
                "SyntheticTransform",
                "SyntheticEncode",
            ),
            "synthetic_final_decode": ("SyntheticDecode",),
            "synthetic_assemble": ("SyntheticAssemble",),
            "synthetic_save": ("SaveVideo",),
        }[self.id]
        return FeatureResolution(
            state="active",
            implementations=tuple(
                _implementation(self.id, class_type) for class_type in classes
            ),
        )

    def required_capabilities(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> CapabilitySet:
        return CapabilitySet()

    def cache_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> dict[str, Any]:
        return {"synthetic": self.id, "version": self.version}

    def runtime_pool_identity(
        self,
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> None:
        return None

    def emit(
        self,
        builder: Any,
        inputs: Mapping[str, Resource],
        params: BaseModel,
        ctx: Any,
        resolution: FeatureResolution,
    ) -> FeatureEmission:
        if self.id == "synthetic_source":
            source = builder.add_node("SyntheticLatentSource", {})
            return FeatureEmission(outputs={"latent": builder.edge(source, 0)})

        if self.id == "synthetic_composite":
            source = inputs["latent"].value
            assert isinstance(source, EdgeRef)
            sampler_a = builder.add_node(
                "SyntheticSampler", {"latent": source}
            )
            intermediate_decode = builder.add_node(
                "SyntheticDecode",
                {"samples": builder.edge(sampler_a, 0)},
            )
            transform = builder.add_node(
                "SyntheticTransform",
                {"images": builder.edge(intermediate_decode, 0)},
            )
            encode = builder.add_node(
                "SyntheticEncode",
                {"images": builder.edge(transform, 0)},
            )
            sampler_b = builder.add_node(
                "SyntheticSampler",
                {"latent": builder.edge(encode, 0)},
            )
            return FeatureEmission(
                outputs={"samples": builder.edge(sampler_b, 0)},
                progress_hints=(
                    _phase("sample_a", "第一次采样", sampler_a, "fractional", 0.20),
                    _phase(
                        "intermediate_decode",
                        "中间解码",
                        intermediate_decode,
                        "milestone",
                        0.05,
                    ),
                    _phase("transform", "中间变换", transform, "milestone", 0.05),
                    _phase("sample_b", "第二次采样", sampler_b, "fractional", 0.40),
                ),
                preview_hints=(
                    _preview(
                        sampler_a,
                        "sample_a",
                        publish=True,
                        priority=10,
                    ),
                    _preview(
                        transform,
                        "transform",
                        publish=False,
                        priority=15,
                    ),
                    _preview(
                        sampler_b,
                        "sample_b",
                        publish=True,
                        priority=20,
                        supersedes=(sampler_a,),
                    ),
                ),
            )

        if self.id == "synthetic_final_decode":
            samples = inputs["samples"].value
            assert isinstance(samples, EdgeRef)
            decode = builder.add_node("SyntheticDecode", {"samples": samples})
            return FeatureEmission(
                outputs={"frames": builder.edge(decode, 0)},
                progress_hints=(
                    _phase(
                        "final_decode",
                        "最终解码",
                        decode,
                        "milestone",
                        0.15,
                    ),
                ),
            )

        if self.id == "synthetic_assemble":
            frames = inputs["frames"].value
            assert isinstance(frames, EdgeRef)
            assemble = builder.add_node(
                "SyntheticAssemble", {"images": frames}
            )
            return FeatureEmission(
                outputs={"video": builder.edge(assemble, 0)},
                progress_hints=(
                    _phase(
                        "assemble", "封装视频", assemble, "milestone", 0.10
                    ),
                ),
            )

        assert self.id == "synthetic_save"
        video = inputs["video"].value
        assert isinstance(video, EdgeRef)
        save = builder.add_node("SaveVideo", {"video": video})
        take = builder.terminal(save)
        if self.hidden_second_save:
            builder.add_node("SaveVideo", {"video": video})
        return FeatureEmission(
            outputs={"take_output": take},
            progress_hints=(
                _phase("persist", "写入视频", save, "milestone", 0.05),
            ),
        )


def _entry(
    feature_id: str,
    phase: str,
    *,
    reads: tuple[ResourceReadDeclaration, ...] = (),
    writes: tuple[ResourceWriteDeclaration, ...] = (),
    requires: tuple[str, ...] = (),
) -> FeatureTemplateEntry:
    return FeatureTemplateEntry(
        id=feature_id,
        version=1,
        title=feature_id,
        description=f"Synthetic Stage-9 feature {feature_id}",
        mode="needed",
        graph_phase=phase,
        reads=reads,
        writes=writes,
        backends=("standard",),
        families=("fl2va",),
        requires=requires,
        scopes=("segment",),
        ui={"visibility": "test_only"},
    )


def _synthetic_template() -> SegmentTemplate:
    return SegmentTemplate(
        id="h3_standard_segment",
        revision=9001,
        entries=(
            _entry(
                "synthetic_source",
                "bootstrap",
                writes=(
                    ResourceWriteDeclaration(
                        name="latent", type="LATENT", operation="define"
                    ),
                ),
            ),
            _entry(
                "synthetic_composite",
                "sampling",
                reads=(ResourceReadDeclaration(name="latent", type="LATENT"),),
                writes=(
                    ResourceWriteDeclaration(
                        name="samples", type="LATENT", operation="define"
                    ),
                ),
                requires=("synthetic_source",),
            ),
            _entry(
                "synthetic_final_decode",
                "decode",
                reads=(ResourceReadDeclaration(name="samples", type="LATENT"),),
                writes=(
                    ResourceWriteDeclaration(
                        name="frames", type="IMAGE", operation="define"
                    ),
                ),
                requires=("synthetic_composite",),
            ),
            _entry(
                "synthetic_assemble",
                "postprocess",
                reads=(ResourceReadDeclaration(name="frames", type="IMAGE"),),
                writes=(
                    ResourceWriteDeclaration(
                        name="video", type="VIDEO", operation="define"
                    ),
                ),
                requires=("synthetic_final_decode",),
            ),
            _entry(
                "synthetic_save",
                "persist",
                reads=(ResourceReadDeclaration(name="video", type="VIDEO"),),
                writes=(
                    ResourceWriteDeclaration(
                        name="take_output", type="TAKE", operation="define"
                    ),
                ),
                requires=("synthetic_assemble",),
            ),
        ),
    )


@dataclass(frozen=True)
class _CompiledSyntheticGraph:
    prompt: dict[str, Any]
    registry: NodeContractRegistry
    node_contract_snapshot: dict[str, NodeContractEvidence]
    public_writes: tuple[PublicResourceWrite, ...]
    public_reads: tuple[PublicResourceRead, ...]
    feature_traces: tuple[FeatureAuditTrace, ...]
    scoped_emissions: tuple[tuple[tuple[str, ...], FeatureEmission], ...]
    take_node_id: str

    def audit_ingredients(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "node_contract_registry": self.registry,
            "node_contract_snapshot": self.node_contract_snapshot,
            "public_writes": self.public_writes,
            "public_reads": self.public_reads,
            "feature_traces": self.feature_traces,
            "model_family": "fl2va",
            "backend": "standard",
            "unit_kind": "segment",
            "take_node_id": self.take_node_id,
        }


def _compile_synthetic_graph(
    *, hidden_second_save: bool = False
) -> _CompiledSyntheticGraph:
    template = _synthetic_template()
    feature_registry = FeatureInterpreterRegistry()
    for entry in template.entries:
        feature_registry.register(
            _SyntheticInterpreter(
                entry.id,
                hidden_second_save=(
                    hidden_second_save and entry.id == "synthetic_save"
                ),
            )
        )
    validated = feature_registry.freeze().validate_template(template)
    context = SimpleNamespace(
        backend="standard", family="fl2va", template_bundle_version=9001
    )
    params = _NoParams()
    graph = PromptGraphBuilder()
    pool = ResourcePool()
    public_writes: list[PublicResourceWrite] = []
    public_reads: list[PublicResourceRead] = []
    trace_parts: list[tuple[str, FeatureResolution, Any]] = []
    scoped_emissions: list[tuple[tuple[str, ...], FeatureEmission]] = []

    for entry in template.entries:
        interpreter = validated.interpreter_for(entry)
        resolution = interpreter.resolve(params, context)
        inputs = _read_resources(pool, entry)
        before = pool
        with graph.begin_scope(entry.id) as scope:
            emission = interpreter.emit(
                scope, inputs, params, context, resolution
            )
            pool = _commit_emission(
                pool=pool,
                entry=entry,
                emission=emission,
                scope=scope,
            )
        public_reads.extend(_scope_public_reads(inputs=inputs, scope=scope))
        public_writes.extend(
            _scope_public_writes(
                before=before,
                after=pool,
                entry=entry,
                emission=emission,
            )
        )
        trace_parts.append(
            _scope_trace_parts(
                entry=entry,
                resolution=resolution,
                scope=scope,
            )
        )
        scoped_emissions.append((scope.emitted_node_ids, emission))

    take = pool.read_required(
        "take_output", expected_type="TAKE", allow_terminal=True
    ).value
    assert isinstance(take, TerminalRef)
    prompt = graph.prompt
    structural_features = {
        write.resource.source_feature_id for write in public_writes
    }
    traces = tuple(
        FeatureAuditTrace(
            feature_id=feature_id,
            resolution=resolution,
            emitted_nodes=emitted_nodes,
            structural_influence=feature_id in structural_features,
        )
        for feature_id, resolution, emitted_nodes in trace_parts
    )
    node_registry = _node_registry()
    return _CompiledSyntheticGraph(
        prompt=prompt,
        registry=node_registry,
        node_contract_snapshot=_node_contract_snapshot(prompt, node_registry),
        public_writes=tuple(public_writes),
        public_reads=tuple(public_reads),
        feature_traces=traces,
        scoped_emissions=tuple(scoped_emissions),
        take_node_id=take.node_id,
    )


def _unit_and_prepared() -> tuple[NativeWorkflowUnit, PreparedSegmentUnit, _CompiledSyntheticGraph]:
    compiled = _compile_synthetic_graph()
    progress, preview = derive_feature_execution_specs(
        compiled.scoped_emissions
    )
    assert progress is not None and preview is not None
    audit = build_graph_audit_spec(**compiled.audit_ingredients())
    unit = NativeWorkflowUnit(
        id="stage9-synthetic-unit",
        family="fl2va",
        backend="standard",
        segment_ids=(_SEGMENT_ID,),
        prompt=compiled.prompt,
        output_nodes={_SEGMENT_ID: compiled.take_node_id},
        graph_audit_spec=audit,
        graph_audit_traces=compiled.feature_traces,
        progress_spec=progress,
        preview_spec=preview,
    )
    prepared = PreparedSegmentUnit(
        id=unit.id,
        owner_segment_id=_SEGMENT_ID,
        family="fl2va",
        backend="standard",
        template_id="h3_standard_segment",
        template_revision=9001,
        prompt_base=unit.prompt,
        graph_audit_spec=audit,
        expected_output_spec=ExpectedOutputSpec(
            segment_id=_SEGMENT_ID,
            node_id=compiled.take_node_id,
            width=736,
            height=416,
            fps=24.0,
            visible_frame_count=121,
            expected_audio_mode="none",
        ),
        progress_spec=progress,
        preview_spec=preview,
        continuity_dependency=None,
        runtime_requirements=RuntimeRequirements(
            endpoint_key="synthetic-host",
            backend="standard",
            logical_gpu_indices=(),
            requires_standard_driver_access=True,
        ),
        runtime_pool_identity=None,
        effective_execution_digest=sha256_document_digest(
            {"stage": 9, "unit": unit.id, "prompt": unit.prompt}
        ),
    )
    return unit, prepared, compiled


def _compiled_plan(prepared: PreparedSegmentUnit) -> CompiledExecutionPlan:
    return CompiledExecutionPlan(
        version=1,
        template_bundle_version=5,
        segment_units=(prepared,),
        compile_report={"source": "stage9_synthetic_test_only"},
        node_policy={"source": "stage9_synthetic_test_only"},
        effective_execution_digest=sha256_document_digest(
            {
                "stage": 9,
                "unit_id": prepared.id,
                "unit_digest": prepared.effective_execution_digest.model_dump(
                    mode="json"
                ),
            }
        ),
    )


def _create_parent(database: Database, job_id: str) -> None:
    now = datetime(2026, 8, 22, tzinfo=timezone.utc).isoformat()
    database.create_job(
        {
            "id": job_id,
            "mode": "timeline",
            "status": "preparing",
            "progress": 0.0,
            "stage": "preflight",
            "prompt_id": None,
            "project_id": Database.LEGACY_DEFAULT_PROJECT_ID,
            "outputs": [],
            "error": None,
            "config_snapshot": {},
            "settings_snapshot": {},
            "prompt_snapshot": {},
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "completed_at": None,
        }
    )


def _node_ids(prompt: Mapping[str, Any], class_type: str) -> tuple[str, ...]:
    return tuple(
        node_id
        for node_id, node in prompt.items()
        if node["class_type"] == class_type
    )


def _take_cone(prompt: Mapping[str, Any], take_node_id: str) -> frozenset[str]:
    reverse: dict[str, set[str]] = {node_id: set() for node_id in prompt}
    for edge in collect_prompt_input_edges(prompt):
        reverse[edge.consumer_node_id].add(edge.source_node_id)
    cone: set[str] = set()
    pending = [take_node_id]
    while pending:
        node_id = pending.pop()
        if node_id in cone:
            continue
        cone.add(node_id)
        pending.extend(reverse[node_id])
    return frozenset(cone)


def _persisted_child(unit: NativeWorkflowUnit) -> dict[str, Any]:
    assert unit.progress_spec is not None and unit.preview_spec is not None
    return {
        "segment_ids": [_SEGMENT_ID],
        "group_index": 0,
        "progress": 0.0,
        "prompt_snapshot": unit.prompt,
        "exact_prompt_snapshot": {
            "owner_segment_id": _SEGMENT_ID,
            "expected_output_spec": {"segment_id": _SEGMENT_ID},
            "exact_prompt": unit.prompt,
            "progress_spec": unit.progress_spec.model_dump(mode="json"),
            "preview_spec": unit.preview_spec.model_dump(mode="json"),
        },
    }


def test_composite_interpreter_is_one_segment_take_with_private_nodes_in_its_cone() -> None:
    unit, prepared, compiled = _unit_and_prepared()
    prompt = unit.prompt
    sampler_ids = _node_ids(prompt, "SyntheticSampler")
    decode_ids = _node_ids(prompt, "SyntheticDecode")
    transform_id = _node_ids(prompt, "SyntheticTransform")[0]
    encode_id = _node_ids(prompt, "SyntheticEncode")[0]
    cone = _take_cone(prompt, compiled.take_node_id)

    assert len(sampler_ids) == 2
    assert len(decode_ids) == 2
    assert set((*sampler_ids, *decode_ids, transform_id, encode_id)) <= cone
    assert cone == frozenset(prompt)
    assert prepared.owner_segment_id == _SEGMENT_ID
    assert prepared.expected_output_spec.node_id == compiled.take_node_id
    assert prepared.graph_audit_spec.take_node_id == compiled.take_node_id
    assert unit.output_nodes == {_SEGMENT_ID: compiled.take_node_id}
    assert [
        node_id
        for node_id, evidence in compiled.node_contract_snapshot.items()
        if evidence.persistent_artifact_role is not None
    ] == [compiled.take_node_id]

    composite_write = next(
        write
        for write in compiled.public_writes
        if write.resource.source_feature_id == "synthetic_composite"
    )
    assert composite_write.resource.name == "samples"
    assert composite_write.resource.producer_node_ids == (sampler_ids[1],)
    assert set((sampler_ids[0], decode_ids[0], transform_id, encode_id)).isdisjoint(
        composite_write.resource.producer_node_ids
    )
    assert tuple(write.resource.name for write in compiled.public_writes) == (
        "latent",
        "samples",
        "frames",
        "video",
        "take_output",
    )

    assert unit.progress_spec is prepared.progress_spec
    assert unit.preview_spec is prepared.preview_spec
    assert [phase.weight for phase in unit.progress_spec.phases] == [
        0.20,
        0.05,
        0.05,
        0.40,
        0.15,
        0.10,
        0.05,
    ]


def test_composite_specs_survive_lock_exact_snapshot_and_sqlite_round_trip(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import directordeck.database as database_module
    import directordeck.execution.submission as submission_module

    unit, prepared, compiled = _unit_and_prepared()
    plan = _compiled_plan(prepared)
    # Synthetic contracts stay test-local.  Patch only the two validation
    # boundaries which ordinarily consume the production bundle-5 registry.
    monkeypatch.setattr(
        submission_module, "CURRENT_NODE_CONTRACT_REGISTRY", compiled.registry
    )
    monkeypatch.setattr(
        database_module, "CURRENT_NODE_CONTRACT_REGISTRY", compiled.registry
    )
    endpoint = EndpointIdentity(
        endpoint_key="synthetic-host",
        runtime_instance_id="stage9-runtime",
    )
    planner = LockedSubmissionPlanner(endpoint)
    locked = planner.build_wave(
        plan,
        source_unit_ordinal=0,
        segment_child_id="stage9-child",
    )
    locked_unit = locked.units[0]
    exact = planner.exact_snapshot(locked, locked_unit)

    expected_progress = canonical_json(
        prepared.progress_spec.model_dump(mode="json")
    )
    expected_preview = canonical_json(
        prepared.preview_spec.model_dump(mode="json")
    )
    assert canonical_json(locked_unit.progress_spec.model_dump(mode="json")) == (
        expected_progress
    )
    assert canonical_json(locked_unit.preview_spec.model_dump(mode="json")) == (
        expected_preview
    )
    assert exact.owner_segment_id == _SEGMENT_ID
    assert exact.graph_audit_spec.take_node_id == unit.output_nodes[_SEGMENT_ID]
    assert canonical_json(exact.progress_spec.model_dump(mode="json")) == (
        expected_progress
    )
    assert canonical_json(exact.preview_spec.model_dump(mode="json")) == (
        expected_preview
    )

    database = Database(tmp_path / "stage9-composite.sqlite3")
    database.initialize()
    _create_parent(database, "stage9-job")
    database.create_job_execution_plan("stage9-job", plan)
    restored_plan = database.get_job_execution_plan("stage9-job")
    assert restored_plan is not None
    assert canonical_json(
        restored_plan.segment_units[0].progress_spec.model_dump(mode="json")
    ) == expected_progress
    assert canonical_json(
        restored_plan.segment_units[0].preview_spec.model_dump(mode="json")
    ) == expected_preview

    with sqlite3.connect(database.path) as connection:
        indexed_json = connection.execute(
            "SELECT prepared_unit FROM job_execution_plan_units "
            "WHERE job_id = ? AND unit_ordinal = 0",
            ("stage9-job",),
        ).fetchone()[0]
    indexed_unit = PreparedSegmentUnit.model_validate_json(indexed_json)
    assert canonical_json(indexed_unit.progress_spec.model_dump(mode="json")) == (
        expected_progress
    )
    assert canonical_json(indexed_unit.preview_spec.model_dump(mode="json")) == (
        expected_preview
    )

    database.persist_job_child_submission_intent(
        "stage9-job",
        locked_plan=locked,
        exact_snapshot=exact,
    )
    evidence = database.get_job_child_execution_evidence("stage9-child")
    assert evidence is not None
    persisted_locked = evidence["locked_submission_plan"].units[0]
    persisted_exact = evidence["exact_prompt_snapshot"]
    assert canonical_json(
        persisted_locked.progress_spec.model_dump(mode="json")
    ) == expected_progress
    assert canonical_json(
        persisted_locked.preview_spec.model_dump(mode="json")
    ) == expected_preview
    assert canonical_json(
        persisted_exact.progress_spec.model_dump(mode="json")
    ) == expected_progress
    assert canonical_json(
        persisted_exact.preview_spec.model_dump(mode="json")
    ) == expected_preview


def test_composite_progress_is_monotonic_and_intermediate_decode_does_not_jump() -> None:
    unit, _, _ = _unit_and_prepared()
    assert unit.progress_spec is not None
    phases = {phase.id: phase for phase in unit.progress_spec.phases}
    child = _persisted_child(unit)
    observations: list[float] = []

    def record(snapshot: Any) -> None:
        assert snapshot is not None
        observations.append(snapshot.progress)
        child["progress"] = snapshot.progress

    record(
        child_progress_snapshot(
            child,
            ComfyProgressEvent(
                prompt_id="stage9-prompt",
                node_id=phases["sample_a"].node_id,
                value=100.0,
                maximum=100.0,
            ),
        )
    )
    record(
        child_execution_snapshot(
            child,
            ComfyExecutionEvent(
                prompt_id="stage9-prompt",
                node_id=phases["intermediate_decode"].node_id,
            ),
        )
    )
    assert observations[-1] == pytest.approx(0.25)
    assert observations[-1] < 0.9
    record(
        child_execution_snapshot(
            child,
            ComfyExecutionEvent(
                prompt_id="stage9-prompt",
                node_id=phases["transform"].node_id,
            ),
        )
    )
    record(
        child_progress_snapshot(
            child,
            ComfyProgressEvent(
                prompt_id="stage9-prompt",
                node_id=phases["sample_b"].node_id,
                value=50.0,
                maximum=100.0,
            ),
        )
    )
    record(
        child_progress_snapshot(
            child,
            ComfyProgressEvent(
                prompt_id="stage9-prompt",
                node_id=phases["sample_b"].node_id,
                value=100.0,
                maximum=100.0,
            ),
        )
    )
    for phase_id in ("final_decode", "assemble", "persist"):
        record(
            child_execution_snapshot(
                child,
                ComfyExecutionEvent(
                    prompt_id="stage9-prompt",
                    node_id=phases[phase_id].node_id,
                ),
            )
        )

    assert observations == sorted(observations)
    assert observations == pytest.approx(
        [0.20, 0.25, 0.30, 0.50, 0.70, 0.85, 0.95, 1.0]
    )


def test_composite_preview_policy_hides_transform_and_rejects_late_sampler_a() -> None:
    unit, _, _ = _unit_and_prepared()
    assert unit.preview_spec is not None
    child = _persisted_child(unit)
    sources = {source.phase_id: source for source in unit.preview_spec.sources}
    early = preview_source_for_node(child, sources["sample_a"].node_id)
    late = preview_source_for_node(child, sources["sample_b"].node_id)

    assert isinstance(early, ResolvedPreviewSource)
    assert early.priority == 10
    assert preview_source_for_node(child, sources["transform"].node_id) is None
    assert isinstance(late, ResolvedPreviewSource)
    assert late.priority == 20
    assert late.supersedes == (sources["sample_a"].node_id,)

    cache = LivePreviewCache()

    def event(node_id: str, content: bytes) -> ComfyPreviewEvent:
        return ComfyPreviewEvent(
            prompt_id="stage9-prompt",
            node_id=node_id,
            mime_type="image/png",
            content=content,
        )

    assert cache.put(
        job_id="stage9-job",
        child_id="stage9-child",
        segment_id=_SEGMENT_ID,
        event=event(early.node_id, b"sampler-a"),
        source=early,
    )
    transform_started = ComfyExecutionEvent(
        prompt_id="stage9-prompt",
        node_id=sources["transform"].node_id,
    )
    transform_phase_index = preview_phase_index_for_event(
        child, transform_started
    )
    assert transform_phase_index is not None
    assert cache.advance_phase(
        job_id="stage9-job",
        child_id="stage9-child",
        prompt_id="stage9-prompt",
        phase_index=transform_phase_index,
    )
    # The private, non-publishing transform has started. Sampler A is now
    # stale even though sampler B has not produced its first preview yet.
    assert not cache.put(
        job_id="stage9-job",
        child_id="stage9-child",
        segment_id=_SEGMENT_ID,
        event=event(early.node_id, b"late-before-sampler-b"),
        source=early,
    )
    stored = cache.get("stage9-job")
    assert stored is not None and stored.content == b"sampler-a"
    assert cache.put(
        job_id="stage9-job",
        child_id="stage9-child",
        segment_id=_SEGMENT_ID,
        event=event(late.node_id, b"sampler-b"),
        source=late,
    )
    assert not cache.put(
        job_id="stage9-job",
        child_id="stage9-child",
        segment_id=_SEGMENT_ID,
        event=event(early.node_id, b"late-sampler-a"),
        source=early,
    )
    stored = cache.get("stage9-job")
    assert stored is not None and stored.content == b"sampler-b"


def test_graph_audit_rejects_a_hidden_second_persistent_artifact() -> None:
    compiled = _compile_synthetic_graph(hidden_second_save=True)

    with pytest.raises(GraphAuditError, match="matching take terminal/artifact"):
        build_graph_audit_spec(**compiled.audit_ingredients())


def test_graph_audit_rejects_a_private_edge_leaking_across_feature_scopes() -> None:
    compiled = _compile_synthetic_graph()
    prompt = deepcopy(compiled.prompt)
    private_transform = _node_ids(prompt, "SyntheticTransform")[0]
    final_decode = _node_ids(prompt, "SyntheticDecode")[1]
    prompt[final_decode]["inputs"]["private_leak"] = [private_transform, 0]
    ingredients = compiled.audit_ingredients()
    ingredients["prompt"] = prompt

    with pytest.raises(GraphAuditError, match="cross-feature edge"):
        build_graph_audit_spec(**ingredients)


def _replace_hint_node(
    emission: FeatureEmission,
    *,
    collection: str,
    index: int,
    node_id: str,
) -> FeatureEmission:
    progress_hints = [dict(item) for item in emission.progress_hints]
    preview_hints = [dict(item) for item in emission.preview_hints]
    hints = progress_hints if collection == "progress" else preview_hints
    hints[index]["node_id"] = node_id
    return FeatureEmission(
        outputs=dict(emission.outputs),
        progress_hints=tuple(progress_hints),
        preview_hints=tuple(preview_hints),
        notices=emission.notices,
        emission_details=dict(emission.emission_details),
    )


@pytest.mark.parametrize(
    ("collection", "replacement"),
    (
        ("progress", "cross_scope"),
        ("preview", "missing"),
    ),
)
def test_execution_spec_hints_cannot_name_cross_scope_or_missing_nodes(
    collection: str,
    replacement: str,
) -> None:
    compiled = _compile_synthetic_graph()
    scoped = list(compiled.scoped_emissions)
    composite_index = 1
    composite_scope, composite_emission = scoped[composite_index]
    node_id = (
        compiled.scoped_emissions[0][0][0]
        if replacement == "cross_scope"
        else "missing-node"
    )
    scoped[composite_index] = (
        composite_scope,
        _replace_hint_node(
            composite_emission,
            collection=collection,
            index=0,
            node_id=node_id,
        ),
    )

    with pytest.raises(ValueError, match="scope|emitted|node"):
        derive_feature_execution_specs(scoped)


def test_partial_progress_hints_fail_closed_instead_of_using_class_fallback() -> None:
    compiled = _compile_synthetic_graph()
    composite_only = (compiled.scoped_emissions[1],)

    with pytest.raises(ValueError, match="sum to 1|complete|weight"):
        derive_feature_execution_specs(composite_only)
