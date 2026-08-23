"""Pure lock-time materialization for one externally submitted wave.

The caller owns the endpoint ticket and lock.  This module never reads the
database, invokes a feature interpreter, or performs network I/O; it only
binds already-compiled identity to the exact native prompt selected under that
lock and validates the result with the Stage-3 graph audit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from typing import Any
import uuid

from ..native_templates import (
    NativeTemplateError,
    NativeWorkflowUnit,
    build_raylight_shutdown_unit,
    raylight_runtime_descriptor,
    raylight_runtime_logical_gpu_indices,
)
from ..workflow.execution import (
    CompiledPlanDigest,
    CompiledExecutionPlan,
    ControlContinuationDependency,
    ContinuityLateBindingEvidence,
    EndpointIdentity,
    ExactPromptSnapshot,
    LateBindingEvidence,
    LockedSegmentUnit,
    LockedSubmissionPlan,
    LockedSubmissionUnit,
    PreparedControlUnit,
    PreparedSegmentUnit,
    RuntimeEpochLateBindingEvidence,
    canonical_json,
    compiled_execution_plan_digest,
    sha256_document_digest,
)
from ..workflow.node_contracts import (
    CURRENT_NODE_CONTRACT_REGISTRY,
    V4_NODE_CONTRACT_REGISTRY,
)
from ..workflow.templates import V4_TEMPLATE_BUNDLE


class SubmissionPlanningError(ValueError):
    """Lock-time evidence cannot be derived without changing compiled intent."""


_RAY_LEDGER_KEYS = frozenset(
    {
        "version",
        "epoch",
        "current",
        "tail_prompt_id",
        "tail_action",
        "tainted",
        "tail_terminal_certificate",
        "legacy_unknown",
    }
)


def _normalize_ray_ledger(
    value: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SubmissionPlanningError(f"{label} Ray ledger must be an object")
    document = json.loads(canonical_json(value))
    extra = set(document) - _RAY_LEDGER_KEYS
    if extra:
        raise SubmissionPlanningError(
            f"{label} Ray ledger has unknown fields: {', '.join(sorted(extra))}"
        )
    epoch = document.get("epoch")
    current = document.get("current")
    tail_prompt_id = document.get("tail_prompt_id")
    tail_action = document.get("tail_action")
    tainted = document.get("tainted")
    if (
        document.get("version") != 2
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or epoch < 0
        or (current is not None and not isinstance(current, dict))
        or not isinstance(tainted, bool)
    ):
        raise SubmissionPlanningError(f"{label} Ray ledger is invalid")
    if tail_prompt_id is not None and (
        not isinstance(tail_prompt_id, str) or not tail_prompt_id
    ):
        raise SubmissionPlanningError(f"{label} Ray ledger tail prompt is invalid")
    if tail_action not in {None, "ray_unit", "shutdown"} or (
        (tail_prompt_id is None) != (tail_action is None)
    ):
        raise SubmissionPlanningError(f"{label} Ray ledger tail action is invalid")
    certificate = document.get("tail_terminal_certificate")
    if certificate is not None:
        if (
            not isinstance(certificate, dict)
            or set(certificate) != {"prompt_id", "action", "succeeded"}
            or certificate.get("prompt_id") != tail_prompt_id
            or certificate.get("action") != tail_action
            or not isinstance(certificate.get("succeeded"), bool)
        ):
            raise SubmissionPlanningError(
                f"{label} Ray ledger terminal certificate is invalid"
            )
    legacy_unknown = document.get("legacy_unknown", False)
    if not isinstance(legacy_unknown, bool):
        raise SubmissionPlanningError(f"{label} Ray ledger legacy flag is invalid")
    normalized = {
        "version": 2,
        "epoch": epoch,
        "current": current,
        "tail_prompt_id": tail_prompt_id,
        "tail_action": tail_action,
        "tainted": tainted,
    }
    if certificate is not None:
        normalized["tail_terminal_certificate"] = certificate
    if legacy_unknown:
        normalized["legacy_unknown"] = True
    return normalized


def _thaw_json(value: Any) -> Any:
    """Restore mutable JSON containers without changing scalar identity."""

    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _assert_settled_ray_frontier(
    ledger: Mapping[str, Any] | None,
) -> None:
    if ledger is None or ledger.get("tail_prompt_id") is None:
        return
    certificate = ledger.get("tail_terminal_certificate")
    if not isinstance(certificate, Mapping):
        raise SubmissionPlanningError(
            "Ray ledger frontier has no exact terminal certificate"
        )
    succeeded = bool(certificate["succeeded"])
    action = certificate["action"]
    tainted = bool(ledger["tainted"])
    if action == "shutdown" and succeeded:
        raise SubmissionPlanningError(
            "successful RayKill frontier must already be cleared"
        )
    if action == "ray_unit" and tainted == succeeded:
        raise SubmissionPlanningError(
            "Ray generation frontier taint contradicts its terminal certificate"
        )
    if action == "shutdown" and not tainted:
        raise SubmissionPlanningError("failed RayKill frontier must remain tainted")


def _validate_runtime_descriptor(
    descriptor: Mapping[str, Any],
    *,
    epoch: int,
    prepared: PreparedSegmentUnit | LockedSegmentUnit | None = None,
) -> dict[str, Any]:
    normalized = json.loads(canonical_json(descriptor))
    compatibility_key = normalized.get("compatibility_key")
    runtime_key = normalized.get("runtime_key")
    namespace = normalized.get("runtime_namespace")
    if (
        normalized.get("version") != 2
        or not isinstance(compatibility_key, str)
        or not compatibility_key
        or not isinstance(runtime_key, str)
        or not runtime_key
        or namespace != f"{compatibility_key}-e{epoch}"
    ):
        raise SubmissionPlanningError(
            "Ray runtime descriptor does not match the ledger epoch"
        )
    try:
        logical_gpu_indices = raylight_runtime_logical_gpu_indices(normalized)
    except NativeTemplateError as exc:
        raise SubmissionPlanningError(
            f"Ray runtime descriptor is invalid: {exc}"
        ) from exc
    if prepared is not None:
        requirements = prepared.runtime_requirements
        if (
            prepared.backend != "raylight"
            or compatibility_key != requirements.ray_compatibility_key
            or runtime_key != requirements.ray_runtime_key
            or logical_gpu_indices != requirements.logical_gpu_indices
        ):
            raise SubmissionPlanningError(
                "exact Ray descriptor does not satisfy runtime requirements"
            )
    return normalized


def _locked_segment_runtime_descriptor(
    segment: LockedSegmentUnit,
) -> dict[str, Any] | None:
    if segment.backend == "standard":
        return None
    native = NativeWorkflowUnit(
        id=segment.id,
        family=segment.family,
        backend=segment.backend,
        segment_ids=(segment.owner_segment_id,),
        prompt=_thaw_json(segment.exact_prompt),
        output_nodes={segment.owner_segment_id: segment.expected_output_spec.node_id},
        graph_audit_spec=segment.graph_audit_spec,
    )
    try:
        descriptor = raylight_runtime_descriptor(native)
    except NativeTemplateError as exc:
        raise SubmissionPlanningError(
            f"exact Ray segment has no valid runtime descriptor: {exc}"
        ) from exc
    if descriptor is None:
        raise SubmissionPlanningError("exact Ray segment has no runtime descriptor")
    return descriptor


def _requires_raylight_kill(
    prepared: PreparedSegmentUnit | LockedSegmentUnit,
    ledger: Mapping[str, Any] | None,
) -> bool:
    if ledger is None:
        return False
    if bool(ledger.get("legacy_unknown")):
        raise SubmissionPlanningError(
            "legacy-unknown Ray ledger requires explicit restart recovery"
        )
    current = ledger.get("current")
    if current is None:
        if bool(ledger.get("tainted")):
            raise SubmissionPlanningError(
                "tainted Ray ledger has no descriptor for a safe RayKill"
            )
        return False
    if not isinstance(current, Mapping):  # Normalization already proves this.
        raise SubmissionPlanningError("Ray ledger current descriptor is invalid")
    normalized_current = _validate_runtime_descriptor(
        current,
        epoch=int(ledger["epoch"]),
    )
    if bool(ledger.get("tainted")) or prepared.backend == "standard":
        return True
    requirements = prepared.runtime_requirements
    try:
        current_gpu_indices = raylight_runtime_logical_gpu_indices(normalized_current)
    except NativeTemplateError as exc:  # Covered above; retain local context.
        raise SubmissionPlanningError(str(exc)) from exc
    return (
        normalized_current.get("compatibility_key")
        != requirements.ray_compatibility_key
        or normalized_current.get("runtime_key") != requirements.ray_runtime_key
        or current_gpu_indices != requirements.logical_gpu_indices
    )


def _ray_segment_after_intent(
    segment: LockedSegmentUnit,
    before: dict[str, Any] | None,
) -> dict[str, Any] | None:
    _assert_settled_ray_frontier(before)
    if _requires_raylight_kill(segment, before):
        raise SubmissionPlanningError(
            "segment submission requires a preceding RayKill control"
        )
    if segment.backend == "standard":
        return before
    current = before.get("current") if before is not None else None
    prior_epoch = int(before["epoch"]) if before is not None else 0
    target_epoch = prior_epoch + 1 if current is None else prior_epoch
    if target_epoch < 1:
        raise SubmissionPlanningError("Ray runtime epoch must be positive")
    descriptor = _locked_segment_runtime_descriptor(segment)
    assert descriptor is not None
    normalized_descriptor = _validate_runtime_descriptor(
        descriptor,
        epoch=target_epoch,
        prepared=segment,
    )
    return {
        "version": 2,
        "epoch": target_epoch,
        "current": normalized_descriptor,
        "tail_prompt_id": segment.requested_prompt_id,
        "tail_action": "ray_unit",
        "tainted": True,
    }


def _validate_control_graph(
    control: PreparedControlUnit,
    current: Mapping[str, Any],
) -> None:
    descriptor_digest = sha256_document_digest(current)
    if descriptor_digest != control.runtime_descriptor_digest:
        raise SubmissionPlanningError(
            "RayKill control descriptor digest differs from the ledger"
        )
    try:
        expected = build_raylight_shutdown_unit(current, unit_id=control.id)
    except NativeTemplateError as exc:
        raise SubmissionPlanningError(
            f"RayKill control cannot replay the current descriptor: {exc}"
        ) from exc
    if (
        control.family != expected.family
        or canonical_json(control.prompt_base) != canonical_json(expected.prompt)
        or control.graph_audit_spec != expected.graph_audit_spec
    ):
        raise SubmissionPlanningError(
            "RayKill control is not the registered graph for the current descriptor"
        )
    expected_execution_digest = sha256_document_digest(
        {
            "schema_version": 1,
            "unit_kind": "control",
            "control_kind": "ray_kill",
            "family": control.family,
            "template": {
                "id": control.template_id,
                "revision": control.template_revision,
            },
            "runtime_descriptor_digest": descriptor_digest.model_dump(mode="json"),
            "exact_prompt": control.prompt_base,
            "graph_audit_spec": control.graph_audit_spec.model_dump(mode="json"),
        }
    )
    if expected_execution_digest != control.effective_execution_digest:
        raise SubmissionPlanningError(
            "RayKill control execution digest does not match its exact graph"
        )


def _control_after_intent(
    control: PreparedControlUnit,
    segment: LockedSegmentUnit,
    before: dict[str, Any] | None,
) -> dict[str, Any]:
    _assert_settled_ray_frontier(before)
    if before is None or not isinstance(before.get("current"), Mapping):
        raise SubmissionPlanningError(
            "RayKill control requires a resident runtime descriptor"
        )
    if not _requires_raylight_kill(segment, before):
        raise SubmissionPlanningError(
            "RayKill control is not required by the current runtime frontier"
        )
    current = _validate_runtime_descriptor(
        before["current"], epoch=int(before["epoch"])
    )
    _validate_control_graph(control, current)
    if segment.backend == "raylight":
        descriptor = _locked_segment_runtime_descriptor(segment)
        assert descriptor is not None
        _validate_runtime_descriptor(
            descriptor,
            epoch=int(before["epoch"]) + 1,
            prepared=segment,
        )
    after = dict(before)
    after.update(
        tail_prompt_id=control.requested_prompt_id,
        tail_action="shutdown",
        tainted=True,
    )
    after.pop("tail_terminal_certificate", None)
    after.pop("legacy_unknown", None)
    return after


def validate_locked_submission_transition(
    plan: LockedSubmissionPlan,
    selected_unit: LockedSubmissionUnit,
) -> LockedSubmissionPlan:
    """Prove one persisted POST intent against the locked Ray frontier."""

    if selected_unit not in plan.units:
        raise SubmissionPlanningError("selected unit is not part of locked plan")
    before = _normalize_ray_ledger(
        plan.ray_ledger_before,
        label="before-intent",
    )
    after = _normalize_ray_ledger(
        plan.ray_ledger_after_intent,
        label="after-intent",
    )
    if len(plan.units) == 2:
        control, segment = plan.units
        if not isinstance(control, PreparedControlUnit) or not isinstance(
            segment, LockedSegmentUnit
        ):
            raise SubmissionPlanningError("two-unit locked plan is malformed")
        if selected_unit != control:
            raise SubmissionPlanningError(
                "two-unit plan may persist only its first control intent"
            )
        expected_after = _control_after_intent(control, segment, before)
    else:
        segment = plan.units[0]
        if not isinstance(segment, LockedSegmentUnit) or selected_unit != segment:
            raise SubmissionPlanningError(
                "single-unit locked plan must persist its segment"
            )
        expected_after = _ray_segment_after_intent(segment, before)
    if canonical_json(after) != canonical_json(expected_after):
        raise SubmissionPlanningError(
            "Ray ledger after-intent state is not the derived transition"
        )
    return plan


class LockedSubmissionPlanner:
    """Build immutable single-segment submission waves under an endpoint lock."""

    def __init__(
        self,
        endpoint_identity: EndpointIdentity,
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.endpoint_identity = endpoint_identity
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._digest_source_plan: CompiledExecutionPlan | None = None
        self._digest_source_value: CompiledPlanDigest | None = None

    def _source_digest(
        self,
        compiled_plan: CompiledExecutionPlan,
        supplied: CompiledPlanDigest | None,
    ) -> CompiledPlanDigest:
        """Bind one plan identity to one digest for this planner instance."""

        if self._digest_source_plan is compiled_plan:
            cached = self._digest_source_value
            assert cached is not None
            if supplied is not None and supplied != cached:
                raise SubmissionPlanningError(
                    "precomputed compiled plan digest changed during planning"
                )
            return cached
        digest = supplied or compiled_execution_plan_digest(compiled_plan)
        self._digest_source_plan = compiled_plan
        self._digest_source_value = digest
        return digest

    @staticmethod
    def requires_raylight_kill(
        prepared: PreparedSegmentUnit,
        ray_ledger: Mapping[str, Any] | None,
    ) -> bool:
        """Return the conservative lock-time compatibility decision."""

        normalized = _normalize_ray_ledger(
            ray_ledger,
            label="current",
        )
        _assert_settled_ray_frontier(normalized)
        return _requires_raylight_kill(prepared, normalized)

    @staticmethod
    def _set_exact_input(
        prompt: dict[str, Any],
        pointer: str,
        value: Any,
    ) -> None:
        """Apply one registered JSON pointer without accepting structural edits."""

        def decode(token: str) -> str:
            return token.replace("~1", "/").replace("~0", "~")

        tokens = [decode(token) for token in pointer.removeprefix("/").split("/")]
        if len(tokens) != 3 or tokens[1] != "inputs":
            raise SubmissionPlanningError(
                f"late-bound pointer {pointer!r} is not one node input"
            )
        node = prompt.get(tokens[0])
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise SubmissionPlanningError(
                f"late-bound pointer {pointer!r} has no exact prompt target"
            )
        if tokens[2] not in node["inputs"]:
            raise SubmissionPlanningError(
                f"late-bound pointer {pointer!r} targets an absent input"
            )
        node["inputs"][tokens[2]] = value

    @classmethod
    def _materialize_segment(
        cls,
        prepared: PreparedSegmentUnit,
        evidence: Sequence[ContinuityLateBindingEvidence],
        *,
        template_bundle_version: int,
        runtime_epoch: int | None,
        segment_child_id: str,
        source_unit_ordinal: int,
    ) -> LockedSegmentUnit:
        """Bind exact continuity and Ray values from independent authorities.

        The prepared prompt is the sole graph source.  Callers supply only
        typed continuity output evidence; the planner derives Ray epoch from
        the locked ledger and never accepts a caller-authored exact graph.
        """

        continuity_evidence = tuple(evidence)
        if any(
            not isinstance(item, ContinuityLateBindingEvidence)
            for item in continuity_evidence
        ):
            raise SubmissionPlanningError(
                "callers may supply only typed continuity evidence"
            )
        declarations = {
            declaration.input_pointer: declaration
            for declaration in prepared.graph_audit_spec.allowed_late_bound_inputs
            if declaration.source_kind != "resource"
        }
        continuity_declarations = {
            pointer: declaration
            for pointer, declaration in declarations.items()
            if declaration.source_kind == "continuity"
        }
        runtime_declarations = {
            pointer: declaration
            for pointer, declaration in declarations.items()
            if declaration.source_kind == "runtime_epoch"
        }
        evidence_by_pointer = {
            item.input_pointer: item for item in continuity_evidence
        }
        if len(evidence_by_pointer) != len(continuity_evidence):
            raise SubmissionPlanningError(
                "late-binding evidence pointers must be unique"
            )
        if set(evidence_by_pointer) != set(continuity_declarations):
            raise SubmissionPlanningError(
                "continuity evidence must exactly cover declared continuity pointers"
            )

        typed_evidence: list[LateBindingEvidence] = []
        expected_values: dict[str, Any] = {}
        dependency = prepared.continuity_dependency
        for pointer, declaration in continuity_declarations.items():
            item = evidence_by_pointer[pointer]
            if item.source_kind != declaration.source_kind:
                raise SubmissionPlanningError(
                    f"late-binding evidence source differs at {pointer!r}"
                )
            if declaration.value_kind != "string":
                raise SubmissionPlanningError(
                    "continuity evidence must target a string input"
                )
            if (
                not isinstance(dependency, Mapping)
                or dependency.get("input_pointer") != pointer
                or dependency.get("predecessor_segment_id")
                != item.predecessor_segment_id
                or dependency.get("source") != item.dependency_source
                or dependency.get("historical_take_id") != item.historical_take_id
            ):
                raise SubmissionPlanningError(
                    "continuity evidence differs from the compiled dependency"
                )
            if (
                item.dependency_source == "historical_take"
                and (
                    dependency.get("resolved") is not True
                    or dependency.get("bound_file") != item.bound_value
                )
            ):
                raise SubmissionPlanningError(
                    "historical continuity evidence differs from the compiled take"
                )
            typed_evidence.append(item)
            expected_values[pointer] = item.bound_value

        if prepared.backend == "raylight":
            if runtime_epoch is None or len(runtime_declarations) != 1:
                raise SubmissionPlanningError(
                    "RayLight materialization requires one locked runtime epoch"
                )
            pointer, declaration = next(iter(runtime_declarations.items()))
            if declaration.value_kind != "string":
                raise SubmissionPlanningError(
                    "runtime epoch evidence must target a string input"
                )
            compatibility_key = prepared.runtime_requirements.ray_compatibility_key
            if compatibility_key is None:
                raise SubmissionPlanningError(
                    "RayLight materialization has no compatibility key"
                )
            runtime_evidence = RuntimeEpochLateBindingEvidence(
                input_pointer=pointer,
                epoch=runtime_epoch,
            )
            typed_evidence.append(runtime_evidence)
            expected_namespace = f"{compatibility_key}-e{runtime_epoch}"
            expected_values[pointer] = expected_namespace
        elif runtime_epoch is not None or runtime_declarations:
            raise SubmissionPlanningError(
                "Standard materialization cannot bind a Ray runtime epoch"
            )

        exact_prompt = _thaw_json(prepared.prompt_base)
        if not isinstance(exact_prompt, dict):  # Frozen contract invariant.
            raise SubmissionPlanningError("prepared prompt is not an object")
        for pointer, value in expected_values.items():
            cls._set_exact_input(exact_prompt, pointer, value)

        segment = LockedSegmentUnit(
            **prepared.model_dump(),
            child_id=segment_child_id,
            requested_prompt_id=segment_child_id,
            group_index=source_unit_ordinal * 2 + 1,
            exact_prompt=exact_prompt,
            late_binding_evidence=tuple(typed_evidence),
            late_bound_values=expected_values,
        )
        if template_bundle_version == V4_TEMPLATE_BUNDLE.version:
            node_contract_registry = V4_NODE_CONTRACT_REGISTRY
        elif template_bundle_version == 5:
            node_contract_registry = CURRENT_NODE_CONTRACT_REGISTRY
        else:
            raise SubmissionPlanningError(
                "prepared segment uses an unsupported template bundle"
            )
        segment.validate_materialized_prompt(
            node_contract_registry=node_contract_registry
        )
        return segment

    @staticmethod
    def _target_runtime_epoch(
        prepared: PreparedSegmentUnit,
        before: Mapping[str, Any] | None,
        *,
        requires_kill: bool,
    ) -> int | None:
        if prepared.backend == "standard":
            return None
        prior_epoch = int(before["epoch"]) if before is not None else 0
        current = before.get("current") if before is not None else None
        return prior_epoch + 1 if requires_kill or current is None else prior_epoch

    def build_wave(
        self,
        compiled_plan: CompiledExecutionPlan,
        *,
        source_unit_ordinal: int,
        segment_child_id: str,
        continuity_evidence: Sequence[ContinuityLateBindingEvidence] = (),
        ray_ledger_before: Mapping[str, Any] | None = None,
        source_compiled_plan_digest: CompiledPlanDigest | None = None,
    ) -> LockedSubmissionPlan:
        if not 0 <= source_unit_ordinal < len(compiled_plan.segment_units):
            raise SubmissionPlanningError("source unit ordinal is out of range")
        prepared = compiled_plan.segment_units[source_unit_ordinal]
        source_digest = self._source_digest(
            compiled_plan,
            source_compiled_plan_digest,
        )
        if prepared.runtime_requirements.endpoint_key != self.endpoint_identity.endpoint_key:
            raise SubmissionPlanningError(
                "prepared segment targets a different endpoint"
            )
        before = _normalize_ray_ledger(
            ray_ledger_before,
            label="before-intent",
        )
        _assert_settled_ray_frontier(before)
        requires_kill = _requires_raylight_kill(prepared, before)
        runtime_epoch = self._target_runtime_epoch(
            prepared,
            before,
            requires_kill=requires_kill,
        )
        segment = self._materialize_segment(
            prepared,
            continuity_evidence,
            template_bundle_version=compiled_plan.template_bundle_version,
            runtime_epoch=runtime_epoch,
            segment_child_id=segment_child_id,
            source_unit_ordinal=source_unit_ordinal,
        )

        units: tuple[LockedSubmissionUnit, ...]
        if not requires_kill:
            units = (segment,)
        else:
            if before is None or not isinstance(before.get("current"), dict):
                raise SubmissionPlanningError(
                    "RayKill materialization requires the locked resident descriptor"
                )
            runtime_descriptor = before["current"]
            control_child_id = self._id_factory()
            control_requested_prompt_id = control_child_id
            control_unit = build_raylight_shutdown_unit(
                runtime_descriptor,
                unit_id=(
                    f"switch-{source_unit_ordinal:03d}-{control_child_id}"
                ),
            )
            control_template = V4_TEMPLATE_BUNDLE.control_templates.ray_kill
            runtime_descriptor_digest = sha256_document_digest(runtime_descriptor)
            control_execution_digest = sha256_document_digest(
                {
                    "schema_version": 1,
                    "unit_kind": "control",
                    "control_kind": "ray_kill",
                    "family": control_unit.family,
                    "template": {
                        "id": control_template.id,
                        "revision": control_template.revision,
                    },
                    "runtime_descriptor_digest": (
                        runtime_descriptor_digest.model_dump(mode="json")
                    ),
                    "exact_prompt": control_unit.prompt,
                    "graph_audit_spec": control_unit.graph_audit_spec.model_dump(
                        mode="json"
                    ),
                }
            )
            control = PreparedControlUnit(
                id=control_unit.id,
                family=control_unit.family,
                template_revision=control_template.revision,
                child_id=control_child_id,
                requested_prompt_id=control_requested_prompt_id,
                group_index=source_unit_ordinal * 2,
                prompt_base=control_unit.prompt,
                graph_audit_spec=control_unit.graph_audit_spec,
                runtime_descriptor_digest=runtime_descriptor_digest,
                effective_execution_digest=control_execution_digest,
                preceding_unit_id=segment.id,
            )
            units = (control, segment)

        plan = LockedSubmissionPlan(
            version=1,
            endpoint_identity=self.endpoint_identity,
            units=units,
            source_compiled_plan_digest=source_digest,
            source_unit_id=prepared.id,
            source_unit_ordinal=source_unit_ordinal,
            ray_ledger_before=before,
            ray_ledger_after_intent=None,
        )
        selected_unit = units[0]
        before = plan.ray_ledger_before
        if isinstance(selected_unit, PreparedControlUnit):
            derived_after = _control_after_intent(
                selected_unit,
                segment,
                before,
            )
        else:
            derived_after = _ray_segment_after_intent(segment, before)
        plan = LockedSubmissionPlan(
            version=plan.version,
            endpoint_identity=plan.endpoint_identity,
            units=plan.units,
            source_compiled_plan_digest=plan.source_compiled_plan_digest,
            source_unit_id=plan.source_unit_id,
            source_unit_ordinal=plan.source_unit_ordinal,
            ray_ledger_before=plan.ray_ledger_before,
            ray_ledger_after_intent=derived_after,
        )
        plan = plan.validate_source_prepared_unit(
            prepared,
            verified_source_compiled_plan_digest=source_digest,
        )
        return validate_locked_submission_transition(plan, selected_unit)

    @staticmethod
    def segment_continuation(
        plan: LockedSubmissionPlan,
        *,
        ray_ledger_before: Mapping[str, Any] | None,
        ray_ledger_after_intent: Mapping[str, Any] | None = None,
    ) -> LockedSubmissionPlan:
        """Create the post-control, segment-only transaction authority."""

        if len(plan.units) != 2 or not isinstance(plan.units[0], PreparedControlUnit):
            raise SubmissionPlanningError(
                "segment continuation requires a two-unit control plan"
            )
        segment = plan.units[-1]
        if not isinstance(segment, LockedSegmentUnit):  # Contract invariant.
            raise SubmissionPlanningError("locked wave has no target segment")
        original_before = _normalize_ray_ledger(
            plan.ray_ledger_before,
            label="control before-intent",
        )
        continuation_before = _normalize_ray_ledger(
            ray_ledger_before,
            label="continuation before-intent",
        )
        if original_before is None or continuation_before is None:
            raise SubmissionPlanningError(
                "control continuation requires a durable Ray ledger"
            )
        if (
            continuation_before["epoch"] != original_before["epoch"]
            or continuation_before["current"] is not None
            or continuation_before["tail_prompt_id"] is not None
            or continuation_before["tail_action"] is not None
            or continuation_before["tainted"]
            or continuation_before.get("tail_terminal_certificate") is not None
            or continuation_before.get("legacy_unknown")
        ):
            raise SubmissionPlanningError(
                "RayKill control has not produced the exact clean continuation frontier"
            )
        derived_after = _ray_segment_after_intent(
            segment,
            continuation_before,
        )
        if ray_ledger_after_intent is not None:
            hinted_after = _normalize_ray_ledger(
                ray_ledger_after_intent,
                label="caller continuation after-intent",
            )
            if canonical_json(hinted_after) != canonical_json(derived_after):
                raise SubmissionPlanningError(
                    "caller continuation transition differs from planner authority"
                )
        continuation = LockedSubmissionPlan(
            version=plan.version,
            endpoint_identity=plan.endpoint_identity,
            units=(segment,),
            source_compiled_plan_digest=plan.source_compiled_plan_digest,
            source_unit_id=plan.source_unit_id,
            source_unit_ordinal=plan.source_unit_ordinal,
            ray_ledger_before=continuation_before,
            ray_ledger_after_intent=derived_after,
            control_dependency=ControlContinuationDependency(
                control_child_id=plan.units[0].child_id,
                control_unit_id=plan.units[0].id,
                control_requested_prompt_id=plan.units[0].requested_prompt_id,
                control_group_index=plan.units[0].group_index,
                original_locked_plan_digest=sha256_document_digest(
                    plan.model_dump(mode="json")
                ),
                control_exact_prompt_snapshot_digest=sha256_document_digest(
                    LockedSubmissionPlanner.exact_snapshot(
                        plan,
                        plan.units[0],
                    ).model_dump(mode="json")
                ),
            ),
        )
        return validate_locked_submission_transition(continuation, segment)

    @staticmethod
    def exact_snapshot(
        plan: LockedSubmissionPlan,
        unit: LockedSubmissionUnit,
    ) -> ExactPromptSnapshot:
        if unit not in plan.units:
            raise SubmissionPlanningError("unit is not part of locked plan")
        if isinstance(unit, LockedSegmentUnit):
            prompt = unit.exact_prompt
            owner_segment_id = unit.owner_segment_id
            control_kind = None
            expected_output = unit.expected_output_spec
            progress = unit.progress_spec
            preview = unit.preview_spec
        else:
            prompt = unit.prompt_base
            owner_segment_id = None
            control_kind = unit.control_kind
            expected_output = None
            progress = None
            preview = None
        return ExactPromptSnapshot(
            schema_version=1,
            unit_id=unit.id,
            unit_kind=unit.kind,
            owner_segment_id=owner_segment_id,
            control_kind=control_kind,
            family=unit.family,
            backend=unit.backend,
            template_id=unit.template_id,
            template_revision=unit.template_revision,
            endpoint_identity=plan.endpoint_identity,
            exact_prompt=prompt,
            graph_audit_spec=unit.graph_audit_spec,
            expected_output_spec=expected_output,
            progress_spec=progress,
            preview_spec=preview,
            effective_execution_digest=unit.effective_execution_digest,
        )


__all__ = [
    "LockedSubmissionPlanner",
    "SubmissionPlanningError",
    "validate_locked_submission_transition",
]
