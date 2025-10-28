"""
Base algorithm interface for Hamlet agents.

Provides abstract interface that all RL algorithms must implement,
enabling pluggable algorithm architectures.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAlgorithm(ABC):
    """
    Abstract base class for reinforcement learning algorithms.

    All RL algorithms (DQN, PPO, A2C, etc.) must inherit from this
    and implement the required methods. This enables algorithm-agnostic
    training infrastructure.
    """

    def __init__(self, agent_id: str):
        """
        Initialize algorithm.

        Args:
            agent_id: Unique identifier for this agent
        """
        self.agent_id = agent_id

    @abstractmethod
    def select_action(self, observation: Any, explore: bool = True) -> int:
        """
        Select action given observation.

        Args:
            observation: Environment observation (dict or array)
            explore: Whether to use exploration (e.g., epsilon-greedy)

        Returns:
            Action index (integer)
        """
        pass

    @abstractmethod
    def learn(self, batch: tuple) -> float:
        """
        Update policy from experience batch.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)
                Format depends on algorithm requirements

        Returns:
            Training loss value
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """
        Save algorithm state to file.

        Args:
            filepath: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, filepath: str):
        """
        Load algorithm state from file.

        Args:
            filepath: Path to load checkpoint from
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get algorithm configuration as dictionary.

        Returns:
            Dictionary of hyperparameters and settings
        """
        return {
            "agent_id": self.agent_id,
            "algorithm": self.__class__.__name__,
        }
