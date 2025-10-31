"""
Epsilon-greedy exploration strategy (vectorized).

Simple baseline: epsilon probability of random action, 1-epsilon probability
of greedy action. No intrinsic motivation.
"""

from typing import Dict, Any
import torch

from townlet.exploration.base import ExplorationStrategy
from townlet.exploration.action_selection import epsilon_greedy_action_selection
from townlet.training.state import BatchedAgentState


class EpsilonGreedyExploration(ExplorationStrategy):
    """
    Vectorized epsilon-greedy exploration.

    No intrinsic motivation - just simple epsilon-greedy action selection.
    Epsilon decays over time with exponential schedule.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize epsilon-greedy exploration.

        Args:
            epsilon: Initial exploration rate (1.0 = full random)
            epsilon_decay: Decay per episode (0.995 = ~1% decay)
            epsilon_min: Minimum epsilon (prevents pure greedy)
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_actions(
        self,
        q_values: torch.Tensor,  # [batch, num_actions]
        agent_states: BatchedAgentState,
        action_masks: torch.Tensor | None = None,  # [batch, num_actions] bool
    ) -> torch.Tensor:
        """
        Select actions with epsilon-greedy strategy and action masking.

        Args:
            q_values: Q-values for each action [batch, num_actions]
            agent_states: Current state (contains per-agent epsilons)
            action_masks: Optional action validity masks [batch, num_actions] bool

        Returns:
            actions: [batch] selected actions
        """
        return epsilon_greedy_action_selection(
            q_values=q_values,
            epsilons=agent_states.epsilons,
            action_masks=action_masks,
        )

    def compute_intrinsic_rewards(
        self,
        observations: torch.Tensor,  # [batch, obs_dim]
    ) -> torch.Tensor:
        """
        Compute intrinsic rewards (none for epsilon-greedy).

        Args:
            observations: Current observations

        Returns:
            intrinsic_rewards: [batch] all zeros
        """
        batch_size = observations.shape[0]
        return torch.zeros(batch_size, device=observations.device)

    def update(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Update exploration networks (no-op for epsilon-greedy).

        Args:
            batch: Experience batch (ignored)
        """
        pass  # No networks to update

    def decay_epsilon(self) -> None:
        """Decay epsilon (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def checkpoint_state(self) -> Dict[str, Any]:
        """
        Return serializable state.

        Returns:
            Dict with epsilon state
        """
        return {
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore from checkpoint.

        Args:
            state: Dict from checkpoint_state()
        """
        self.epsilon = state["epsilon"]
        self.epsilon_decay = state["epsilon_decay"]
        self.epsilon_min = state["epsilon_min"]
