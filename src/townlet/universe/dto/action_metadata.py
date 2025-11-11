"""Action metadata DTOs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import torch


@dataclass(frozen=True)
class ActionMetadata:
    """Metadata for a single action in the compiled universe."""

    id: int
    name: str
    type: Literal["movement", "interaction", "passive", "transaction"]
    enabled: bool
    source: Literal["substrate", "custom", "affordance"]
    costs: Mapping[str, float] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "costs", MappingProxyType(dict(self.costs)))


@dataclass(frozen=True)
class ActionSpaceMetadata:
    """Rich metadata describing the action space."""

    total_actions: int
    actions: tuple[ActionMetadata, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        if self.total_actions < 0:
            raise ValueError("total_actions must be >= 0")

    def get_enabled_actions(self) -> list[ActionMetadata]:
        """Return the subset of enabled actions."""
        return [action for action in self.actions if action.enabled]

    def get_action_mask(self, num_agents: int, device: torch.device) -> torch.Tensor:
        """Return a [num_agents, total_actions] bool mask with disabled actions masked out."""
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")

        mask = torch.ones(num_agents, self.total_actions, dtype=torch.bool, device=device)
        for action in self.actions:
            if action.id >= self.total_actions:
                raise ValueError(f"Action '{action.name}' has id {action.id} which exceeds total_actions={self.total_actions}")
            if not action.enabled:
                mask[:, action.id] = False
        return mask
