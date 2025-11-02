"""
State representations for Townlet training.

Contains DTOs for cold path (config, checkpoints, telemetry) using Pydantic
for validation, and hot path (training loop) using PyTorch tensors.
"""

from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field


class CurriculumDecision(BaseModel):
    """
    Cold path: Curriculum decision for environment configuration.

    Returned by CurriculumManager to specify environment settings.
    Validated at construction, immutable, serializable.
    """

    model_config = ConfigDict(frozen=True)

    difficulty_level: float = Field(..., ge=0.0, le=1.0, description="Difficulty level from 0.0 (easiest) to 1.0 (hardest)")
    active_meters: list[str] = Field(..., min_length=1, max_length=6, description="Which meters are active (e.g., ['energy', 'hygiene'])")
    depletion_multiplier: float = Field(..., gt=0.0, le=10.0, description="Depletion rate multiplier (0.1 = 10x slower, 1.0 = normal)")
    reward_mode: str = Field(..., pattern=r"^(shaped|sparse)$", description="Reward mode: 'shaped' (dense) or 'sparse'")
    reason: str = Field(..., min_length=1, description="Human-readable explanation for this decision")


class ExplorationConfig(BaseModel):
    """
    Cold path: Configuration for exploration strategy.

    Defines parameters for epsilon-greedy, RND, or adaptive intrinsic exploration.
    """

    model_config = ConfigDict(frozen=True)

    strategy_type: str = Field(
        ...,
        pattern=r"^(epsilon_greedy|rnd|adaptive_intrinsic)$",
        description="Exploration strategy: epsilon_greedy, rnd, or adaptive_intrinsic",
    )
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0, description="Epsilon for epsilon-greedy (0.0 = greedy, 1.0 = random)")
    epsilon_decay: float = Field(default=0.995, gt=0.0, le=1.0, description="Epsilon decay per episode (0.995 = ~1% decay)")
    intrinsic_weight: float = Field(default=0.0, ge=0.0, description="Weight for intrinsic motivation rewards")
    rnd_hidden_dim: int = Field(default=256, gt=0, description="Hidden dimension for RND networks")
    rnd_learning_rate: float = Field(default=0.0001, gt=0.0, description="Learning rate for RND predictor network")


class PopulationCheckpoint(BaseModel):
    """
    Cold path: Serializable population state for checkpointing.

    Contains all state needed to restore a population training run:
    per-agent curriculum state, exploration state, Pareto frontier, etc.
    """

    model_config = ConfigDict(frozen=True)

    generation: int = Field(..., ge=0, description="Generation number (for genetic algorithms)")
    num_agents: int = Field(..., ge=1, le=1000, description="Number of agents in population (1-1000)")
    agent_ids: list[str] = Field(..., description="List of agent identifiers")
    curriculum_states: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-agent curriculum manager state")
    exploration_states: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-agent exploration strategy state")
    pareto_frontier: list[str] = Field(default_factory=list, description="Agent IDs on Pareto frontier (non-dominated solutions)")
    metrics_summary: dict[str, float] = Field(default_factory=dict, description="Summary metrics (avg_survival, avg_reward, etc.)")


class BatchedAgentState:
    """
    Hot path: Vectorized agent state for GPU training loops.

    All data is batched tensors (batch_size = num_agents).
    Optimized for GPU operations, minimal validation overhead.
    Use slots for memory efficiency.
    """

    __slots__ = [
        "observations",
        "actions",
        "rewards",
        "dones",
        "epsilons",
        "intrinsic_rewards",
        "survival_times",
        "curriculum_difficulties",
        "device",
        "info",
    ]

    def __init__(
        self,
        observations: torch.Tensor,  # [batch, obs_dim]
        actions: torch.Tensor,  # [batch]
        rewards: torch.Tensor,  # [batch]
        dones: torch.Tensor,  # [batch] bool
        epsilons: torch.Tensor,  # [batch]
        intrinsic_rewards: torch.Tensor,  # [batch]
        survival_times: torch.Tensor,  # [batch]
        curriculum_difficulties: torch.Tensor,  # [batch]
        device: torch.device,
        info: dict | None = None,  # Environment info dict
    ):
        """
        Construct batched agent state.

        All tensors must be on the same device.
        No validation in __init__ for performance (hot path).
        """
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.epsilons = epsilons
        self.intrinsic_rewards = intrinsic_rewards
        self.survival_times = survival_times
        self.curriculum_difficulties = curriculum_difficulties
        self.device = device
        self.info = info if info is not None else {}

    @property
    def batch_size(self) -> int:
        """Get batch size from observations shape."""
        return self.observations.shape[0]

    def to(self, device: torch.device) -> "BatchedAgentState":
        """
        Move all tensors to specified device.

        Returns new BatchedAgentState (tensors are immutable after .to()).
        """
        return BatchedAgentState(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            epsilons=self.epsilons.to(device),
            intrinsic_rewards=self.intrinsic_rewards.to(device),
            survival_times=self.survival_times.to(device),
            curriculum_difficulties=self.curriculum_difficulties.to(device),
            device=device,
        )

    def detach_cpu_summary(self) -> dict[str, np.ndarray]:
        """
        Extract summary for telemetry (cold path).

        Returns dict of numpy arrays (CPU). Used for logging, checkpoints.
        """
        return {
            "rewards": self.rewards.detach().cpu().numpy(),
            "survival_times": self.survival_times.detach().cpu().numpy(),
            "epsilons": self.epsilons.detach().cpu().numpy(),
            "curriculum_difficulties": self.curriculum_difficulties.detach().cpu().numpy(),
        }
