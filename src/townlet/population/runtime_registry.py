"""
Agent runtime registry.

Maintains per-agent tensors for training-facing state while providing
JSON-safe snapshots for telemetry and inference pipelines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AgentTelemetrySnapshot:
    """Serialisable view of agent runtime metrics."""

    agent_id: str
    survival_time: int
    curriculum_stage: int
    epsilon: float
    intrinsic_weight: float
    timestamp_unix: float

    def to_dict(self) -> dict[str, Any]:
        """Convert snapshot to JSON-safe dictionary."""
        return {
            "agent_id": self.agent_id,
            "survival_time": int(self.survival_time),
            "curriculum_stage": int(self.curriculum_stage),
            "epsilon": float(self.epsilon),
            "intrinsic_weight": float(self.intrinsic_weight),
            "timestamp_unix": float(self.timestamp_unix),
        }


class AgentRuntimeRegistry:
    """Owns per-agent runtime tensors and exposes serialisable snapshots."""

    def __init__(self, agent_ids: list[str], device: torch.device):
        if not agent_ids:
            raise ValueError("agent_ids must contain at least one entry")

        self._agent_ids = agent_ids
        self.device = device
        num_agents = len(agent_ids)

        self._survival_time = torch.zeros(num_agents, dtype=torch.long, device=device)
        self._curriculum_stage = torch.ones(num_agents, dtype=torch.long, device=device)
        self._epsilon = torch.zeros(num_agents, dtype=torch.float32, device=device)
        self._intrinsic_weight = torch.zeros(num_agents, dtype=torch.float32, device=device)

    # --------------------------------------------------------------------- #
    # Tensor accessors
    # --------------------------------------------------------------------- #
    def get_curriculum_stage_tensor(self) -> torch.Tensor:
        """Return tensor of curriculum stages."""
        return self._curriculum_stage

    def get_curriculum_stage(self, agent_idx: int) -> int:
        """Return curriculum stage for an agent."""
        return int(self._curriculum_stage[agent_idx].item())

    def get_survival_time_tensor(self) -> torch.Tensor:
        """Return tensor of survival times."""
        return self._survival_time

    def get_survival_time(self, agent_idx: int) -> int:
        """Return survival time for an agent."""
        return int(self._survival_time[agent_idx].item())

    def get_epsilon(self, agent_idx: int) -> float:
        """Return epsilon for an agent."""
        return float(self._epsilon[agent_idx].item())

    def get_intrinsic_weight(self, agent_idx: int) -> float:
        """Return intrinsic weight for an agent."""
        return float(self._intrinsic_weight[agent_idx].item())

    # --------------------------------------------------------------------- #
    # Mutation helpers
    # --------------------------------------------------------------------- #
    def record_survival_time(self, agent_idx: int, steps: int | torch.Tensor) -> None:
        """Record survival time for an agent."""
        self._survival_time[agent_idx] = self._ensure_long_tensor(steps)

    def set_curriculum_stage(self, agent_idx: int, stage: int | torch.Tensor) -> None:
        """Set curriculum stage for an agent."""
        self._curriculum_stage[agent_idx] = self._ensure_long_tensor(stage)

    def set_epsilon(self, agent_idx: int, epsilon: float | torch.Tensor) -> None:
        """Set exploration epsilon for an agent."""
        self._epsilon[agent_idx] = self._ensure_float_tensor(epsilon)

    def set_intrinsic_weight(self, agent_idx: int, weight: float | torch.Tensor) -> None:
        """Set intrinsic reward weight for an agent."""
        self._intrinsic_weight[agent_idx] = self._ensure_float_tensor(weight)

    # --------------------------------------------------------------------- #
    # Snapshot API
    # --------------------------------------------------------------------- #
    def get_snapshot_for_agent(self, agent_idx: int) -> AgentTelemetrySnapshot:
        """Return JSON-safe snapshot for a single agent."""
        return AgentTelemetrySnapshot(
            agent_id=self._agent_ids[agent_idx],
            survival_time=int(self._survival_time[agent_idx].item()),
            curriculum_stage=int(self._curriculum_stage[agent_idx].item()),
            epsilon=self._epsilon[agent_idx].item(),
            intrinsic_weight=self._intrinsic_weight[agent_idx].item(),
            timestamp_unix=time.time(),
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _ensure_float_tensor(self, value: float | torch.Tensor) -> torch.Tensor:
        tensor = self._ensure_tensor(value, dtype=torch.float32)
        return tensor.to(self.device, dtype=torch.float32)

    def _ensure_long_tensor(self, value: int | torch.Tensor) -> torch.Tensor:
        tensor = self._ensure_tensor(value, dtype=torch.long)
        return tensor.to(self.device, dtype=torch.long)

    def _ensure_tensor(self, value: float | int | torch.Tensor | list[float], dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device, dtype=dtype)
        if isinstance(value, float | int):
            return torch.tensor(value, device=self.device, dtype=dtype)
        if isinstance(value, list):
            return torch.tensor(value, device=self.device, dtype=dtype)
        raise TypeError(f"Unsupported value type: {type(value)!r}")
