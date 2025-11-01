"""
Abstract base class for curriculum managers.

Curriculum managers control environment difficulty progression based on
agent performance metrics (survival time, learning progress, policy entropy).
"""

from abc import ABC, abstractmethod
from typing import Any

from townlet.training.state import BatchedAgentState, CurriculumDecision


class CurriculumManager(ABC):
    """
    Abstract interface for curriculum management.

    Implementations control environment difficulty by returning CurriculumDecisions
    that specify depletion rates, active meters, and reward mode.
    """

    @abstractmethod
    def get_batch_decisions(
        self,
        agent_states: BatchedAgentState,
        agent_ids: list[str],
    ) -> list[CurriculumDecision]:
        """
        Get curriculum decisions for batch of agents.

        Called once per episode (not per step) to determine environment
        configuration for each agent.

        Args:
            agent_states: Current state for all agents [num_agents, ...]
            agent_ids: List of agent identifiers (for per-agent tracking)

        Returns:
            List of CurriculumDecisions (one per agent)

        Note:
            Input is GPU tensors, output is CPU DTOs. Overhead acceptable
            since this runs once per episode, not per step.
        """
        pass

    @abstractmethod
    def checkpoint_state(self) -> dict[str, Any]:
        """
        Return serializable state for checkpoint saving.

        Should include:
        - Per-agent curriculum stage
        - Performance history (survival times, rewards)
        - Any internal state needed to resume

        Returns:
            Dict compatible with JSON/YAML serialization
        """
        pass

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """
        Restore curriculum manager from checkpoint.

        Args:
            state: Dict from checkpoint_state()

        Raises:
            ValueError: If state is invalid or incompatible
        """
        pass
