from __future__ import annotations

"""Fail-closed runtime registry for feature interpreter implementations."""

import re
from dataclasses import dataclass
from typing import Any

from .contracts import FeatureInterpreter, FeatureTemplateEntry, SegmentTemplate


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_INTERPRETER_METHODS = (
    "validate_params",
    "resolve",
    "required_capabilities",
    "cache_identity",
    "runtime_pool_identity",
    "emit",
)


class FeatureInterpreterRegistryError(ValueError):
    """Base error for invalid registration or exact resolution."""


class FeatureInterpreterNotFoundError(FeatureInterpreterRegistryError, KeyError):
    """No interpreter exists for the requested exact id and version."""


class FeatureInterpreterRegistryFrozenError(FeatureInterpreterRegistryError):
    """The immutable compilation registry can no longer be changed."""


class FeatureTemplateValidationError(FeatureInterpreterRegistryError):
    """A template and its exact interpreters/resource declarations disagree."""


@dataclass(frozen=True, slots=True)
class RegisteredFeatureInterpreter:
    id: str
    version: int
    interpreter: FeatureInterpreter


@dataclass(frozen=True, slots=True)
class ValidatedFeatureTemplate:
    """A segment template bound to one exact interpreter per ordered entry."""

    template: SegmentTemplate
    bindings: tuple[RegisteredFeatureInterpreter, ...]

    def __post_init__(self) -> None:
        expected = tuple((entry.id, entry.version) for entry in self.template.entries)
        actual = tuple((binding.id, binding.version) for binding in self.bindings)
        if actual != expected:
            raise FeatureTemplateValidationError(
                "validated bindings must exactly follow template entry order"
            )

    @property
    def id(self) -> str:
        return self.template.id

    @property
    def revision(self) -> int:
        return self.template.revision

    def interpreter_for(self, entry: FeatureTemplateEntry) -> FeatureInterpreter:
        for binding in self.bindings:
            if binding.id == entry.id and binding.version == entry.version:
                return binding.interpreter
        raise FeatureInterpreterNotFoundError(
            f"template has no exact binding for {entry.id}@{entry.version}"
        )


class FeatureInterpreterRegistry:
    """Register explicitly versioned interpreters; never guess or fall back."""

    __slots__ = ("_frozen", "_registrations")

    def __init__(self) -> None:
        self._registrations: dict[
            tuple[str, int], RegisteredFeatureInterpreter
        ] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def identities(self) -> tuple[tuple[str, int], ...]:
        return tuple(self._registrations)

    def register(
        self,
        interpreter: FeatureInterpreter,
    ) -> FeatureInterpreterRegistry:
        if self._frozen:
            raise FeatureInterpreterRegistryFrozenError(
                "feature interpreter registry is frozen"
            )
        if isinstance(interpreter, type):
            raise FeatureInterpreterRegistryError(
                "register an interpreter instance, not its class"
            )
        feature_id = getattr(interpreter, "id", None)
        version = getattr(interpreter, "version", None)
        if not isinstance(feature_id, str) or _IDENTIFIER.fullmatch(feature_id) is None:
            raise FeatureInterpreterRegistryError(
                "interpreter id is not a valid bounded feature identifier"
            )
        if type(version) is not int or version < 1 or version > 2_147_483_647:
            raise FeatureInterpreterRegistryError(
                "interpreter version must be a positive 32-bit integer"
            )
        missing_methods = tuple(
            name
            for name in _INTERPRETER_METHODS
            if not callable(getattr(interpreter, name, None))
        )
        if missing_methods:
            raise FeatureInterpreterRegistryError(
                "interpreter does not implement required methods: "
                + ", ".join(missing_methods)
            )
        identity = (feature_id, version)
        if identity in self._registrations:
            raise FeatureInterpreterRegistryError(
                f"feature interpreter already registered: {feature_id}@{version}"
            )
        self._registrations[identity] = RegisteredFeatureInterpreter(
            id=feature_id,
            version=version,
            interpreter=interpreter,
        )
        return self

    def freeze(self) -> FeatureInterpreterRegistry:
        self._frozen = True
        return self

    def require(self, feature_id: str, version: int) -> FeatureInterpreter:
        """Resolve exactly; another version of the same id is never a fallback."""

        identity = (feature_id, version)
        try:
            return self._registrations[identity].interpreter
        except KeyError as exc:
            raise FeatureInterpreterNotFoundError(
                f"unknown exact feature interpreter: {feature_id}@{version}"
            ) from exc

    def validate_template(self, template: SegmentTemplate) -> ValidatedFeatureTemplate:
        if not self._frozen:
            raise FeatureTemplateValidationError(
                "freeze the feature interpreter registry before validating a template"
            )
        _validate_template_dependencies(template)
        _validate_template_resource_flow(template)
        bindings = tuple(
            self._registrations[(entry.id, entry.version)]
            if (entry.id, entry.version) in self._registrations
            else _raise_missing(entry)
            for entry in template.entries
        )
        return ValidatedFeatureTemplate(template=template, bindings=bindings)


def _raise_missing(entry: FeatureTemplateEntry) -> Any:
    raise FeatureInterpreterNotFoundError(
        f"unknown exact feature interpreter: {entry.id}@{entry.version}"
    )


def _validate_template_dependencies(template: SegmentTemplate) -> None:
    positions = {entry.id: index for index, entry in enumerate(template.entries)}
    for index, entry in enumerate(template.entries):
        for dependency in entry.requires:
            position = positions.get(dependency)
            if position is None:
                raise FeatureTemplateValidationError(
                    f"feature {entry.id!r} requires unknown feature {dependency!r}"
                )
            if position >= index:
                raise FeatureTemplateValidationError(
                    f"feature {entry.id!r} requires {dependency!r} before it executes"
                )
        for conflict in entry.conflicts:
            if conflict not in positions:
                raise FeatureTemplateValidationError(
                    f"feature {entry.id!r} conflicts with unknown feature {conflict!r}"
                )


def _validate_template_resource_flow(template: SegmentTemplate) -> None:
    # Value is (port type, definitely defined on every active route).
    resources: dict[str, tuple[str, bool]] = {}
    for entry in template.entries:
        for declaration in entry.reads:
            current = resources.get(declaration.name)
            if current is None:
                if declaration.required:
                    raise FeatureTemplateValidationError(
                        f"feature {entry.id!r} requires undefined resource "
                        f"{declaration.name!r}"
                    )
                continue
            current_type, definitely_defined = current
            if current_type != declaration.type:
                raise FeatureTemplateValidationError(
                    f"feature {entry.id!r} reads resource {declaration.name!r} "
                    f"as {declaration.type!r}, already declared {current_type!r}"
                )
            if declaration.required and not definitely_defined:
                raise FeatureTemplateValidationError(
                    f"feature {entry.id!r} requires conditionally defined resource "
                    f"{declaration.name!r}"
                )
        for declaration in entry.writes:
            current = resources.get(declaration.name)
            if declaration.operation == "define":
                if current is not None:
                    raise FeatureTemplateValidationError(
                        f"feature {entry.id!r} redefines resource "
                        f"{declaration.name!r}"
                    )
                resources[declaration.name] = (
                    declaration.type,
                    entry.mode == "needed" and declaration.required,
                )
            else:
                if current is None:
                    raise FeatureTemplateValidationError(
                        f"feature {entry.id!r} replaces undefined resource "
                        f"{declaration.name!r}"
                    )
                current_type, definitely_defined = current
                if current_type != declaration.type:
                    raise FeatureTemplateValidationError(
                        f"feature {entry.id!r} replaces resource "
                        f"{declaration.name!r} with type {declaration.type!r}, "
                        f"already declared {current_type!r}"
                    )
                resources[declaration.name] = (
                    current_type,
                    definitely_defined,
                )


__all__ = [
    "FeatureInterpreterNotFoundError",
    "FeatureInterpreterRegistry",
    "FeatureInterpreterRegistryError",
    "FeatureInterpreterRegistryFrozenError",
    "FeatureTemplateValidationError",
    "RegisteredFeatureInterpreter",
    "ValidatedFeatureTemplate",
]
