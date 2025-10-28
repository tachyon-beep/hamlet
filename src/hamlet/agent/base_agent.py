"""
Base agent interface for Hamlet.

Defines the abstract interface that all agent types must implement.
Allows swapping between different agent implementations (random, DRL, scripted).
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for agents.

    All agent types (DRL, random, scripted) implement this interface.
    """

    def __init__(self, agent_id: str):
        """
        Initialize agent.

        Args:
            agent_id: Unique identifier for this agent
        """
        self.agent_id = agent_id

    @abstractmethod
    def select_action(self, observation):
        """
        Select an action given the current observation.

        Args:
            observation: Environment observation

        Returns:
            Action index (integer)
        """
        pass

    @abstractmethod
    def learn(self, experience):
        """
        Learn from experience.

        Args:
            experience: Tuple of (state, action, reward, next_state, done)
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save agent state to file."""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load agent state from file."""
        pass
