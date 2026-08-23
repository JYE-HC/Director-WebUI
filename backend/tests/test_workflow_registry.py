from __future__ import annotations

from types import SimpleNamespace

import pytest

from directordeck.workflow.contracts import (
    FeatureTemplateEntry,
    ResourceReadDeclaration,
    ResourceWriteDeclaration,
    SegmentTemplate,
)
from directordeck.workflow.registry import (
    FeatureInterpreterNotFoundError,
    FeatureInterpreterRegistry,
    FeatureInterpreterRegistryError,
    FeatureInterpreterRegistryFrozenError,
    FeatureTemplateValidationError,
    RegisteredFeatureInterpreter,
    ValidatedFeatureTemplate,
)


class _Interpreter:
    def __init__(self, feature_id: str, version: int) -> None:
        self.id = feature_id
        self.version = version

    def validate_params(self, params, ctx) -> None:
        return None

    def resolve(self, params, ctx):
        return None

    def required_capabilities(self, params, ctx, resolution):
        return None

    def cache_identity(self, params, ctx, resolution):
        return None

    def runtime_pool_identity(self, params, ctx, resolution):
        return None

    def emit(self, builder, inputs, params, ctx, resolution):
        return None


def _entry(
    feature_id: str,
    phase: str,
    *,
    version: int = 1,
    reads: tuple[ResourceReadDeclaration, ...] = (),
    writes: tuple[ResourceWriteDeclaration, ...] = (),
    requires: tuple[str, ...] = (),
    conflicts: tuple[str, ...] = (),
) -> FeatureTemplateEntry:
    return FeatureTemplateEntry(
        id=feature_id,
        version=version,
        title=feature_id,
        description=f"Description for {feature_id}",
        mode="needed",
        graph_phase=phase,
        reads=reads,
        writes=writes,
        backends=("standard",),
        families=("fl2va", "ref2va"),
        requires=requires,
        conflicts=conflicts,
        scopes=("segment",),
    )


def _template(*entries: FeatureTemplateEntry) -> SegmentTemplate:
    return SegmentTemplate(
        id="h3_standard_segment",
        revision=1,
        entries=entries,
    )


def test_registry_resolves_only_the_exact_id_and_version() -> None:
    v1 = _Interpreter("lora", 1)
    v2 = _Interpreter("lora", 2)
    registry = FeatureInterpreterRegistry().register(v1).register(v2)

    assert registry.identities == (("lora", 1), ("lora", 2))
    assert registry.require("lora", 1) is v1
    assert registry.require("lora", 2) is v2
    with pytest.raises(FeatureInterpreterNotFoundError, match="lora@3"):
        registry.require("lora", 3)
    with pytest.raises(FeatureInterpreterNotFoundError, match="missing@1"):
        registry.require("missing", 1)


def test_duplicate_registration_and_post_freeze_mutation_fail() -> None:
    registry = FeatureInterpreterRegistry().register(_Interpreter("lora", 1))
    with pytest.raises(FeatureInterpreterRegistryError, match="already registered"):
        registry.register(_Interpreter("lora", 1))

    assert registry.freeze() is registry
    assert registry.freeze() is registry
    assert registry.frozen is True
    with pytest.raises(FeatureInterpreterRegistryFrozenError, match="frozen"):
        registry.register(_Interpreter("sampling", 1))


@pytest.mark.parametrize(
    "interpreter",
    [
        _Interpreter("invalid id", 1),
        _Interpreter("valid", 0),
        _Interpreter("valid", True),
        SimpleNamespace(id="valid", version=1),
        _Interpreter,
    ],
)
def test_invalid_interpreter_contracts_are_rejected(interpreter: object) -> None:
    with pytest.raises(FeatureInterpreterRegistryError):
        FeatureInterpreterRegistry().register(interpreter)  # type: ignore[arg-type]


def test_template_validation_requires_frozen_registry_and_exact_bindings() -> None:
    first = _Interpreter("load", 1)
    second = _Interpreter("sample", 2)
    registry = FeatureInterpreterRegistry().register(first).register(second)
    template = _template(
        _entry(
            "load",
            "model_load",
            writes=(ResourceWriteDeclaration(name="model", type="MODEL", operation="define"),),
        ),
        _entry(
            "sample",
            "sampling",
            version=2,
            reads=(ResourceReadDeclaration(name="model", type="MODEL"),),
            writes=(ResourceWriteDeclaration(name="samples", type="LATENT", operation="define"),),
            requires=("load",),
        ),
    )

    with pytest.raises(FeatureTemplateValidationError, match="freeze"):
        registry.validate_template(template)
    validated = registry.freeze().validate_template(template)
    assert isinstance(validated, ValidatedFeatureTemplate)
    assert validated.id == "h3_standard_segment"
    assert validated.revision == 1
    assert validated.interpreter_for(template.entries[0]) is first
    assert validated.interpreter_for(template.entries[1]) is second


def test_template_validation_never_falls_back_to_registered_version() -> None:
    registry = FeatureInterpreterRegistry().register(_Interpreter("load", 2)).freeze()
    template = _template(_entry("load", "model_load", version=1))
    with pytest.raises(FeatureInterpreterNotFoundError, match="load@1"):
        registry.validate_template(template)


def test_validated_template_rejects_reordered_or_partial_bindings() -> None:
    interpreter = _Interpreter("load", 1)
    template = _template(_entry("load", "model_load"))
    with pytest.raises(FeatureTemplateValidationError, match="exactly follow"):
        ValidatedFeatureTemplate(template=template, bindings=())
    with pytest.raises(FeatureTemplateValidationError, match="exactly follow"):
        ValidatedFeatureTemplate(
            template=template,
            bindings=(
                RegisteredFeatureInterpreter(
                    id="other",
                    version=1,
                    interpreter=interpreter,
                ),
            ),
        )


def test_required_dependency_must_exist_and_precede_consumer() -> None:
    registry = (
        FeatureInterpreterRegistry()
        .register(_Interpreter("first", 1))
        .register(_Interpreter("second", 1))
        .freeze()
    )
    unknown = _template(
        _entry("first", "model_load", requires=("missing",)),
        _entry("second", "sampling"),
    )
    with pytest.raises(FeatureTemplateValidationError, match="requires unknown"):
        registry.validate_template(unknown)

    future = _template(
        _entry("first", "model_load", requires=("second",)),
        _entry("second", "sampling"),
    )
    with pytest.raises(FeatureTemplateValidationError, match="before it executes"):
        registry.validate_template(future)


def test_conflict_must_name_an_entry_in_the_same_template() -> None:
    registry = FeatureInterpreterRegistry().register(_Interpreter("first", 1)).freeze()
    template = _template(
        _entry("first", "model_load", conflicts=("not_in_template",))
    )
    with pytest.raises(FeatureTemplateValidationError, match="conflicts with unknown"):
        registry.validate_template(template)


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (
            (
                _entry(
                    "first",
                    "model_load",
                    reads=(ResourceReadDeclaration(name="model", type="MODEL"),),
                ),
            ),
            "requires undefined resource",
        ),
        (
            (
                _entry(
                    "first",
                    "model_load",
                    writes=(ResourceWriteDeclaration(name="model", type="MODEL", operation="replace"),),
                ),
            ),
            "replaces undefined resource",
        ),
        (
            (
                _entry(
                    "first",
                    "model_load",
                    writes=(ResourceWriteDeclaration(name="model", type="MODEL", operation="define"),),
                ),
                _entry(
                    "second",
                    "sampling",
                    writes=(ResourceWriteDeclaration(name="model", type="MODEL", operation="define"),),
                ),
            ),
            "redefines resource",
        ),
        (
            (
                _entry(
                    "first",
                    "model_load",
                    writes=(ResourceWriteDeclaration(name="model", type="MODEL", operation="define"),),
                ),
                _entry(
                    "second",
                    "sampling",
                    reads=(ResourceReadDeclaration(name="model", type="LATENT"),),
                ),
            ),
            "already declared 'MODEL'",
        ),
        (
            (
                _entry(
                    "first",
                    "model_load",
                    writes=(ResourceWriteDeclaration(name="model", type="MODEL", operation="define"),),
                ),
                _entry(
                    "second",
                    "sampling",
                    writes=(ResourceWriteDeclaration(name="model", type="LATENT", operation="replace"),),
                ),
            ),
            "already declared 'MODEL'",
        ),
    ],
)
def test_invalid_resource_declaration_flows_fail_closed(
    entries: tuple[FeatureTemplateEntry, ...],
    message: str,
) -> None:
    registry = FeatureInterpreterRegistry()
    for entry in entries:
        registry.register(_Interpreter(entry.id, entry.version))
    with pytest.raises(FeatureTemplateValidationError, match=message):
        registry.freeze().validate_template(_template(*entries))


def test_optional_undefined_read_is_a_declared_deterministic_branch() -> None:
    entry = _entry(
        "feature",
        "sampling",
        reads=(
            ResourceReadDeclaration(
                name="optional_audio",
                type="AUDIO",
                required=False,
            ),
        ),
    )
    registry = FeatureInterpreterRegistry().register(_Interpreter("feature", 1)).freeze()
    assert registry.validate_template(_template(entry)).bindings[0].interpreter.id == "feature"


def test_conditional_write_requires_an_optional_downstream_read() -> None:
    producer = _entry(
        "producer",
        "conditioning",
        writes=(
            ResourceWriteDeclaration(
                name="source_audio",
                type="AUDIO",
                operation="define",
                required=False,
            ),
        ),
    )
    optional_consumer = _entry(
        "consumer",
        "decode",
        reads=(
            ResourceReadDeclaration(
                name="source_audio",
                type="AUDIO",
                required=False,
            ),
        ),
    )
    registry = (
        FeatureInterpreterRegistry()
        .register(_Interpreter("producer", 1))
        .register(_Interpreter("consumer", 1))
        .freeze()
    )
    assert registry.validate_template(
        _template(producer, optional_consumer)
    ).bindings

    required_consumer = _entry(
        "consumer",
        "decode",
        reads=(
            ResourceReadDeclaration(
                name="source_audio",
                type="AUDIO",
                required=True,
            ),
        ),
    )
    with pytest.raises(
        FeatureTemplateValidationError,
        match="requires conditionally defined resource 'source_audio'",
    ):
        registry.validate_template(_template(producer, required_consumer))


def test_switch_defined_resource_cannot_be_required_downstream() -> None:
    switch = _entry(
        "switch",
        "model_prepare",
        writes=(
            ResourceWriteDeclaration(
                name="optional_adapter",
                type="MODEL",
                operation="define",
            ),
        ),
    ).model_copy(update={"mode": "switch"})
    consumer = _entry(
        "consumer",
        "sampling",
        reads=(
            ResourceReadDeclaration(
                name="optional_adapter",
                type="MODEL",
                required=True,
            ),
        ),
    )
    registry = (
        FeatureInterpreterRegistry()
        .register(_Interpreter("switch", 1))
        .register(_Interpreter("consumer", 1))
        .freeze()
    )

    with pytest.raises(
        FeatureTemplateValidationError,
        match="requires conditionally defined resource 'optional_adapter'",
    ):
        registry.validate_template(_template(switch, consumer))
