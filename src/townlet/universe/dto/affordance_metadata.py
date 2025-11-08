"""Affordance metadata DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class AffordanceInfo:
    """Metadata for a single affordance."""

    id: str
    name: str
    enabled: bool
    effects: Mapping[str, float]
    cost: float
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "effects", MappingProxyType(dict(self.effects)))


@dataclass(frozen=True)
class AffordanceMetadata:
    """Collection of affordance metadata."""

    affordances: tuple[AffordanceInfo, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "affordances", tuple(self.affordances))

    def get_affordance_by_name(self, name: str) -> AffordanceInfo:
        """Lookup affordance by display name."""
        for affordance in self.affordances:
            if affordance.name == name:
                return affordance
        raise KeyError(f"Affordance '{name}' not found")
