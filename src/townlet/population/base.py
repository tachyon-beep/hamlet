"""
Abstract base class for population managers.

Population managers coordinate multiple agents, handle Pareto frontier tracking,
and (in future) manage genetic reproduction.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from townlet.training.state import BatchedAgentState, PopulationCheckpoint

# Avoid circular import
if TYPE_CHECKING:
    from townlet.environment.vectorized_env import (
        VectorizedHamletEnv,  # type: ignore[import-untyped]
    )


class PopulationManager(ABC):
    """
    Abstract interface for population management.

    Implementations coordinate multiple agents (n=1 to 100), track performance,
    and manage reproduction (future feature).
    """

    @abstractmethod
    def step_population(
        self,
        envs: "VectorizedHamletEnv",  # Forward reference
    ) -> BatchedAgentState:
        """
        Execute one training step for entire population (GPU).

        Coordinates:
        - Action selection via exploration strategy
        - Environment stepping (vectorized)
        - Reward calculation (extrinsic + intrinsic)
        - Replay buffer updates
        - Q-network training

        Args:
            envs: Vectorized environment [num_agents parallel]

        Returns:
            BatchedAgentState with all agent data after step

        Note:
            Hot path - called every step. Must be GPU-optimized.
        """
        pass

    @abstractmethod
    def get_checkpoint(self) -> PopulationCheckpoint:
        """
        Return Pydantic checkpoint (cold path).

        Aggregates:
        - Agent network weights
        - Curriculum states (per agent)
        - Exploration states (per agent)
        - Pareto frontier
        - Metrics summary

        Returns:
            PopulationCheckpoint (Pydantic DTO)
        """
        pass
