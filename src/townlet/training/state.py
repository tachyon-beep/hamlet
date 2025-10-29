"""
State representations for Townlet training.

Contains DTOs for cold path (config, checkpoints, telemetry) using Pydantic
for validation, and hot path (training loop) using PyTorch tensors.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any


class CurriculumDecision(BaseModel):
    """
    Cold path: Curriculum decision for environment configuration.

    Returned by CurriculumManager to specify environment settings.
    Validated at construction, immutable, serializable.
    """
    model_config = ConfigDict(frozen=True)

    difficulty_level: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Difficulty level from 0.0 (easiest) to 1.0 (hardest)"
    )
    active_meters: List[str] = Field(
        ...,
        min_length=1,
        max_length=6,
        description="Which meters are active (e.g., ['energy', 'hygiene'])"
    )
    depletion_multiplier: float = Field(
        ...,
        gt=0.0,
        le=10.0,
        description="Depletion rate multiplier (0.1 = 10x slower, 1.0 = normal)"
    )
    reward_mode: str = Field(
        ...,
        pattern=r'^(shaped|sparse)$',
        description="Reward mode: 'shaped' (dense) or 'sparse'"
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation for this decision"
    )


class ExplorationConfig(BaseModel):
    """
    Cold path: Configuration for exploration strategy.

    Defines parameters for epsilon-greedy, RND, or adaptive intrinsic exploration.
    """
    model_config = ConfigDict(frozen=True)

    strategy_type: str = Field(
        ...,
        pattern=r'^(epsilon_greedy|rnd|adaptive_intrinsic)$',
        description="Exploration strategy: epsilon_greedy, rnd, or adaptive_intrinsic"
    )
    epsilon: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Epsilon for epsilon-greedy (0.0 = greedy, 1.0 = random)"
    )
    epsilon_decay: float = Field(
        default=0.995,
        gt=0.0,
        le=1.0,
        description="Epsilon decay per episode (0.995 = ~1% decay)"
    )
    intrinsic_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight for intrinsic motivation rewards"
    )
    rnd_hidden_dim: int = Field(
        default=256,
        gt=0,
        description="Hidden dimension for RND networks"
    )
    rnd_learning_rate: float = Field(
        default=0.0001,
        gt=0.0,
        description="Learning rate for RND predictor network"
    )


class PopulationCheckpoint(BaseModel):
    """
    Cold path: Serializable population state for checkpointing.

    Contains all state needed to restore a population training run:
    per-agent curriculum state, exploration state, Pareto frontier, etc.
    """
    model_config = ConfigDict(frozen=True)

    generation: int = Field(
        ...,
        ge=0,
        description="Generation number (for genetic algorithms)"
    )
    num_agents: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Number of agents in population (1-1000)"
    )
    agent_ids: List[str] = Field(
        ...,
        description="List of agent identifiers"
    )
    curriculum_states: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-agent curriculum manager state"
    )
    exploration_states: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-agent exploration strategy state"
    )
    pareto_frontier: List[str] = Field(
        default_factory=list,
        description="Agent IDs on Pareto frontier (non-dominated solutions)"
    )
    metrics_summary: Dict[str, float] = Field(
        default_factory=dict,
        description="Summary metrics (avg_survival, avg_reward, etc.)"
    )
