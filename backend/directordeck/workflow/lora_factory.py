from __future__ import annotations

"""Versioned, configuration-backed LoRA adapter resolution.

This module owns host-knowledge resolution only. Creative selection remains
in the v5 project. DirectorDeck maintains a small explicit loader list in its
startup configuration; an optional user record maps one exact LoRA path to a
loader and its non-creative options. Unmapped files use the configured default
from the first matching filename policy, or the fallback policy. This module
never scans arbitrary nodes, guesses from code-owned filename rules, or
certifies a third-party implementation.
"""

from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import Field, model_validator

from ..config_manager import get_directordeck_config, get_lora_loader_policy
from .contracts import ContractModel, FrozenMap, JsonValue, ModelFamily


LoraAdapterId: TypeAlias = str
LoraAdapterResolutionSource: TypeAlias = Literal[
    "user_override",
    "factory_default",
    "backend_fixed",
]
LoraAdapterInputContract: TypeAlias = Literal[
    "dedicated_model",
    "model_only",
    "bypass_model_only",
    "ray_lora",
]


class LoraLoaderOverrideLike(Protocol):
    lora_filename: str
    adapter_id: str
    options: Mapping[str, bool]


class LoraLoaderBindingKey(ContractModel):
    family: ModelFamily
    model_filename: Annotated[str, Field(min_length=1, max_length=4_096)]
    lora_filename: Annotated[str, Field(min_length=1, max_length=4_096)]


class LoraAdapterOptionDefinition(ContractModel):
    id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    type: Literal["boolean"]
    label: Annotated[str, Field(min_length=1, max_length=128)]
    description: Annotated[str, Field(min_length=1, max_length=512)]
    default: bool


class LoraAdapterContract(ContractModel):
    adapter_id: LoraAdapterId
    class_type: Annotated[str, Field(min_length=1, max_length=256)]
    input_contract: LoraAdapterInputContract
    supported_families: tuple[ModelFamily, ...]
    backend: Literal["standard", "raylight"]
    display_name: Annotated[str, Field(min_length=1, max_length=128)]
    option_definitions: tuple[LoraAdapterOptionDefinition, ...] = ()

    @model_validator(mode="after")
    def validate_adapter(self) -> "LoraAdapterContract":
        if not self.supported_families:
            raise ValueError("LoRA adapter must support at least one model family")
        if len(self.supported_families) != len(set(self.supported_families)):
            raise ValueError("LoRA adapter supported families must be unique")
        if self.backend == "raylight" and self.adapter_id != "ray_lora":
            raise ValueError("RayLight LoRA must use the fixed ray_lora adapter")
        if self.backend == "standard" and self.adapter_id == "ray_lora":
            raise ValueError("Standard LoRA cannot use the ray_lora adapter")
        return self


class ResolvedLoraAdapter(ContractModel):
    adapter: LoraAdapterContract
    binding: LoraLoaderBindingKey | None
    source: LoraAdapterResolutionSource
    options: FrozenMap[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_resolution(self) -> "ResolvedLoraAdapter":
        if self.source == "backend_fixed":
            if (
                self.adapter.backend != "raylight"
                or self.binding is not None
                or self.options
            ):
                raise ValueError("backend-fixed LoRA resolution must be RayLight")
        elif self.adapter.backend != "standard" or self.binding is None:
            raise ValueError("mapped LoRA resolution must be Standard")
        return self


class LoraAdapterResolutionError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        binding: LoraLoaderBindingKey | None = None,
        adapter_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.binding = binding
        self.adapter_id = adapter_id


_RAY_ADAPTER = LoraAdapterContract(
    adapter_id="ray_lora",
    class_type="DirectorDeckRayLoraLoader",
    input_contract="ray_lora",
    supported_families=("fl2va", "ref2va"),
    backend="raylight",
    display_name="DirectorDeck RayLight LoRA",
)


@lru_cache(maxsize=1)
def standard_lora_adapters() -> tuple[LoraAdapterContract, ...]:
    """Project the startup-loaded JSON list into immutable workflow contracts."""

    config = get_directordeck_config()
    return tuple(
        LoraAdapterContract(
            adapter_id=definition.id,
            class_type=definition.class_type,
            input_contract=definition.input_contract,
            supported_families=definition.supported_families,
            backend="standard",
            display_name=definition.display_name,
            option_definitions=tuple(
                LoraAdapterOptionDefinition.model_validate(
                    option.model_dump(mode="json")
                )
                for option in definition.options
            ),
        )
        for definition in config.loaders
    )


@lru_cache(maxsize=1)
def _lora_adapter_registry() -> dict[str, LoraAdapterContract]:
    return {
        adapter.adapter_id: adapter
        for adapter in (*standard_lora_adapters(), _RAY_ADAPTER)
    }


# Frozen historical v4 payloads can still name these retired ids. They are
# compatibility readers only and never appear in the current supported list.
_LEGACY_ADAPTERS: dict[str, LoraAdapterContract] = {
    "dedicated": LoraAdapterContract(
        adapter_id="dedicated",
        class_type="MiniMaxH3TurboLoRA",
        input_contract="dedicated_model",
        supported_families=("fl2va", "ref2va"),
        backend="standard",
        display_name="MiniMax-H3 Turbo LoRA（历史兼容）",
        option_definitions=(
            LoraAdapterOptionDefinition(
                id="low_vram",
                type="boolean",
                label="low_vram",
                description="历史兼容选项。",
                default=False,
            ),
        ),
    ),
    "bypass_model_only": LoraAdapterContract(
        adapter_id="bypass_model_only",
        class_type="LoraLoaderBypassModelOnly",
        input_contract="bypass_model_only",
        supported_families=("fl2va", "ref2va"),
        backend="standard",
        display_name="量化旁路 Model Only（历史兼容）",
    ),
}


def require_lora_adapter(adapter_id: str) -> LoraAdapterContract:
    try:
        return _lora_adapter_registry().get(adapter_id) or _LEGACY_ADAPTERS[adapter_id]
    except KeyError as exc:
        raise LoraAdapterResolutionError(
            "The selected LoRA adapter is unknown to this DirectorDeck build.",
            code="lora_adapter_unknown",
            adapter_id=adapter_id,
        ) from exc


def _matching_user_override(
    binding: LoraLoaderBindingKey,
    overrides: Iterable[LoraLoaderOverrideLike],
) -> LoraLoaderOverrideLike | None:
    matches = tuple(
        record
        for record in overrides
        if record.lora_filename == binding.lora_filename
    )
    if len(matches) > 1:
        # RuntimeSettingsV3 rejects this at its authority boundary.  Retain a
        # fail-closed invariant here for historical/corrupt call sites.
        raise LoraAdapterResolutionError(
            "The exact LoRA path has multiple user mappings.",
            code="lora_loader_mapping_conflict",
            binding=binding,
        )
    return matches[0] if matches else None


def resolve_standard_lora_adapter(
    binding: LoraLoaderBindingKey,
    overrides: Iterable[LoraLoaderOverrideLike],
) -> ResolvedLoraAdapter:
    """Resolve one LoRA within its configured loader allowlist."""

    config = get_directordeck_config()
    policy = get_lora_loader_policy(binding.lora_filename)
    override = _matching_user_override(binding, overrides)
    if override is not None:
        adapter = require_lora_adapter(override.adapter_id)
        if override.adapter_id not in policy.loader_ids:
            raise LoraAdapterResolutionError(
                "The selected LoRA loader is not allowed for this LoRA file.",
                code="lora_loader_not_allowed_for_file",
                binding=binding,
                adapter_id=override.adapter_id,
            )
        if adapter.backend != "standard" or binding.family not in adapter.supported_families:
            raise LoraAdapterResolutionError(
                "The selected LoRA adapter does not support this Standard family.",
                code="lora_adapter_incompatible",
                binding=binding,
                adapter_id=override.adapter_id,
            )
        try:
            options = config.normalize_lora_loader_options(
                override.adapter_id,
                dict(override.options),
            )
        except (KeyError, ValueError) as exc:
            raise LoraAdapterResolutionError(
                "The selected LoRA loader configuration is invalid.",
                code="lora_loader_options_invalid",
                binding=binding,
                adapter_id=override.adapter_id,
            ) from exc
        return ResolvedLoraAdapter(
            adapter=adapter,
            binding=binding,
            source="user_override",
            options=options,
        )

    adapter = require_lora_adapter(policy.default_loader_id)
    return ResolvedLoraAdapter(
        adapter=adapter,
        binding=binding,
        source="factory_default",
        options=config.normalize_lora_loader_options(adapter.adapter_id, {}),
    )


def resolve_raylight_lora_adapter(family: ModelFamily) -> ResolvedLoraAdapter:
    """Return the fixed Ray adapter without constructing a Standard key."""

    if family not in _RAY_ADAPTER.supported_families:
        raise LoraAdapterResolutionError(
            "The fixed RayLight LoRA adapter does not support this family.",
            code="lora_adapter_incompatible",
            adapter_id=_RAY_ADAPTER.adapter_id,
        )
    return ResolvedLoraAdapter(
        adapter=_RAY_ADAPTER,
        binding=None,
        source="backend_fixed",
        options={},
    )


__all__ = [
    "LoraAdapterContract",
    "LoraAdapterOptionDefinition",
    "LoraAdapterId",
    "LoraAdapterInputContract",
    "LoraAdapterResolutionError",
    "LoraAdapterResolutionSource",
    "LoraLoaderBindingKey",
    "ResolvedLoraAdapter",
    "require_lora_adapter",
    "resolve_raylight_lora_adapter",
    "resolve_standard_lora_adapter",
    "standard_lora_adapters",
]
