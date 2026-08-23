from __future__ import annotations

"""Process-wide, immutable DirectorDeck product configuration.

The configuration is loaded and validated once during application startup.
Runtime consumers only read the in-memory snapshot and never reopen the JSON
file.  This file describes DirectorDeck-maintained support; it is not a scan or
certification of third-party nodes installed in the user's ComfyUI instance.
"""

import json
import re
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal, Pattern

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class LoraLoaderOptionDefinition(_ConfigModel):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    type: Literal["boolean"]
    label: Annotated[str, Field(min_length=1, max_length=128)]
    description: Annotated[str, Field(min_length=1, max_length=512)]
    default: bool


class SupportedLoraLoaderDefinition(_ConfigModel):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    display_name: Annotated[str, Field(min_length=1, max_length=128)]
    class_type: Annotated[str, Field(min_length=1, max_length=256)]
    input_contract: Literal["dedicated_model", "model_only"]
    supported_families: tuple[Literal["fl2va", "ref2va"], ...]
    options: tuple[LoraLoaderOptionDefinition, ...] = ()

    @field_validator("supported_families", "options", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_definition(self) -> "SupportedLoraLoaderDefinition":
        if not self.supported_families:
            raise ValueError("LoRA loader must support at least one model family")
        if len(self.supported_families) != len(set(self.supported_families)):
            raise ValueError("LoRA loader families must be unique")
        option_ids = tuple(option.id for option in self.options)
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("LoRA loader option ids must be unique")
        return self


class LoraLoaderSelectionPolicy(_ConfigModel):
    loader_ids: tuple[
        Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")], ...
    ]
    default_loader_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]

    @field_validator("loader_ids", mode="before")
    @classmethod
    def freeze_loader_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_policy(self) -> "LoraLoaderSelectionPolicy":
        if not self.loader_ids:
            raise ValueError("LoRA loader policy must allow at least one loader")
        if len(self.loader_ids) != len(set(self.loader_ids)):
            raise ValueError("LoRA loader policy ids must be unique")
        if self.default_loader_id not in self.loader_ids:
            raise ValueError("LoRA loader policy default must be allowed")
        return self


class LoraLoaderPolicy(LoraLoaderSelectionPolicy):
    lora_filename: Annotated[str, Field(min_length=1, max_length=1_024)]

    @model_validator(mode="after")
    def validate_regular_expression(self) -> "LoraLoaderPolicy":
        try:
            re.compile(self.lora_filename)
        except re.error as exc:
            raise ValueError(f"invalid LoRA filename regular expression: {exc}") from exc
        return self


class LoraConfiguration(_ConfigModel):
    loaders: tuple[SupportedLoraLoaderDefinition, ...]
    fallback_policy: LoraLoaderSelectionPolicy
    loader_policies: tuple[LoraLoaderPolicy, ...] = ()

    @field_validator("loaders", "loader_policies", mode="before")
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_loaders(self) -> "LoraConfiguration":
        if not self.loaders:
            raise ValueError("DirectorDeck must configure at least one LoRA loader")
        loader_ids = tuple(loader.id for loader in self.loaders)
        if len(loader_ids) != len(set(loader_ids)):
            raise ValueError("supported LoRA loader ids must be unique")
        patterns = tuple(policy.lora_filename for policy in self.loader_policies)
        if len(patterns) != len(set(patterns)):
            raise ValueError("LoRA loader policy regular expressions must be unique")
        known_ids = set(loader_ids)
        for policy in (self.fallback_policy, *self.loader_policies):
            unknown = set(policy.loader_ids) - known_ids
            if unknown:
                raise ValueError(
                    "LoRA loader policy references unknown loaders: "
                    + ", ".join(sorted(unknown))
                )
        return self

    def require_lora_loader(self, loader_id: str) -> SupportedLoraLoaderDefinition:
        for loader in self.loaders:
            if loader.id == loader_id:
                return loader
        raise KeyError(loader_id)

    def normalize_lora_loader_options(
        self,
        loader_id: str,
        values: object,
    ) -> dict[str, bool]:
        loader = self.require_lora_loader(loader_id)
        if not isinstance(values, dict) or any(
            not isinstance(key, str) or not isinstance(value, bool)
            for key, value in values.items()
        ):
            raise ValueError("LoRA loader options must be a boolean object")
        definitions = {option.id: option for option in loader.options}
        unknown = set(values) - set(definitions)
        if unknown:
            raise ValueError(
                f"unsupported options for LoRA loader {loader_id!r}: "
                + ", ".join(sorted(unknown))
            )
        return {
            option.id: values.get(option.id, option.default)
            for option in loader.options
        }


class DirectorDeckConfig(_ConfigModel):
    schema_version: Literal[1]
    lora: LoraConfiguration

    @property
    def default_loader_id(self) -> str:
        return self.lora.fallback_policy.default_loader_id

    @property
    def loaders(self) -> tuple[SupportedLoraLoaderDefinition, ...]:
        return self.lora.loaders

    def require_lora_loader(self, loader_id: str) -> SupportedLoraLoaderDefinition:
        return self.lora.require_lora_loader(loader_id)

    def normalize_lora_loader_options(
        self,
        loader_id: str,
        values: object,
    ) -> dict[str, bool]:
        return self.lora.normalize_lora_loader_options(loader_id, values)


class DirectorDeckConfigManager:
    """Load one validated config file once and publish an immutable snapshot."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot: DirectorDeckConfig | None = None
        self._source: Path | None = None
        self._compiled_lora_policies: tuple[
            tuple[Pattern[str], LoraLoaderPolicy], ...
        ] = ()

    @staticmethod
    def default_path() -> Path:
        return Path(__file__).with_name("config") / "directordeck.json"

    def initialize(self, path: str | Path | None = None) -> DirectorDeckConfig:
        source = Path(path) if path is not None else self.default_path()
        source = source.resolve()
        with self._lock:
            if self._snapshot is not None:
                if source != self._source:
                    raise RuntimeError(
                        "DirectorDeck configuration is already initialized from "
                        f"{self._source}"
                    )
                return self._snapshot
            try:
                raw = json.loads(source.read_text(encoding="utf-8"))
                snapshot = DirectorDeckConfig.model_validate(raw)
            except Exception as exc:
                raise RuntimeError(
                    f"DirectorDeck configuration is invalid: {source}: {exc}"
                ) from exc
            self._source = source
            self._snapshot = snapshot
            self._compiled_lora_policies = tuple(
                (re.compile(policy.lora_filename), policy)
                for policy in snapshot.lora.loader_policies
            )
            return snapshot

    def get(self) -> DirectorDeckConfig:
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError(
                "DirectorDeck configuration has not been initialized during startup"
            )
        return snapshot

    def lora_loader_policy(
        self,
        lora_filename: str,
    ) -> LoraLoaderSelectionPolicy:
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError(
                "DirectorDeck configuration has not been initialized during startup"
            )
        for pattern, policy in self._compiled_lora_policies:
            if pattern.search(lora_filename) is not None:
                return policy
        return snapshot.lora.fallback_policy


DIRECTORDECK_CONFIG = DirectorDeckConfigManager()


def initialize_directordeck_config(
    path: str | Path | None = None,
) -> DirectorDeckConfig:
    return DIRECTORDECK_CONFIG.initialize(path)


def get_directordeck_config() -> DirectorDeckConfig:
    return DIRECTORDECK_CONFIG.get()


def get_lora_loader_policy(lora_filename: str) -> LoraLoaderSelectionPolicy:
    return DIRECTORDECK_CONFIG.lora_loader_policy(lora_filename)


def is_lora_loader_allowed(lora_filename: str, loader_id: str) -> bool:
    return loader_id in get_lora_loader_policy(lora_filename).loader_ids


__all__ = [
    "DIRECTORDECK_CONFIG",
    "DirectorDeckConfig",
    "DirectorDeckConfigManager",
    "LoraConfiguration",
    "LoraLoaderOptionDefinition",
    "LoraLoaderPolicy",
    "LoraLoaderSelectionPolicy",
    "SupportedLoraLoaderDefinition",
    "get_directordeck_config",
    "get_lora_loader_policy",
    "initialize_directordeck_config",
    "is_lora_loader_allowed",
]
