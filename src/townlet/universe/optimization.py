"""Optimization data structures for compiled universes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class OptimizationData:
    """Pre-computed tensors/lookup tables used at runtime.

    Stage 6 will progressively populate these fields; for now we provide
    placeholders so downstream plumbing can be validated.
    """

    base_depletions: torch.Tensor
    cascade_data: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    modulation_data: list[dict[str, Any]] = field(default_factory=list)
    action_mask_table: torch.Tensor | None = None
    affordance_position_map: dict[str, torch.Tensor | None] = field(default_factory=dict)
