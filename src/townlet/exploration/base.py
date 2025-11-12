"""
Abstract base class for exploration strategies.

Exploration strategies control action selection (exploration vs exploitation)
and optionally provide intrinsic motivation rewards (RND, ICM, etc.).
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from townlet.training.state import BatchedAgentState


class ExplorationStrategy(ABC):
    """
    Abstract interface for exploration strategies.

    Implementations manage epsilon-greedy, RND, adaptive intrinsic motivation, etc.
    All methods operate on batched tensors for GPU efficiency.
    """

    @abstractmethod
    def select_actions(
        self,
        q_values: torch.Tensor,  # [batch, num_actions]
        agent_states: BatchedAgentState,
        action_masks: torch.Tensor | None = None,  # [batch, num_actions] bool
    ) -> torch.Tensor:
        """
        Select actions for batch of agents (GPU).

        This runs EVERY STEP for all agents. Must be GPU-optimized.

        Args:
            q_values: Q-values for each action [batch, num_actions]
            agent_states: Current state (contains epsilons, curriculum stage, etc.)
            action_masks: Optional action validity masks [batch, num_actions] bool
                True = valid action, False = invalid (boundary constraint)

        Returns:
            actions: [batch] tensor of selected actions (int)

        Note:
            Hot path - minimize overhead. No validation, no CPU transfers.
        """
        pass

    @abstractmethod
    def compute_intrinsic_rewards(
        self,
        observations: torch.Tensor,  # [batch, obs_dim]
        update_stats: bool = False,
    ) -> torch.Tensor:
        """
        Compute intrinsic motivation rewards (GPU).

        For RND: prediction error as novelty signal.
        For ICM: forward model prediction error.
        For epsilon-greedy: returns zeros.

        Args:
            observations: Current observations [batch, obs_dim]

        Returns:
            intrinsic_rewards: [batch] tensor

        Note:
            Hot path - runs every step. Return zeros if no intrinsic motivation.
        """
        pass

    @abstractmethod
    def update(self, batch: dict[str, torch.Tensor]) -> None:
        """
        Update exploration networks (RND, ICM, etc.) from experience batch.

        For epsilon-greedy: no-op (nothing to update).
        For RND: train predictor network.
        For ICM: train forward/inverse models.

        Args:
            batch: Dict of tensors (states, actions, rewards, next_states, dones)

        Note:
            Called after replay buffer sampling. Can be slow (not hot path).
        """
        pass

    @abstractmethod
    def checkpoint_state(self) -> dict[str, Any]:
        """
        Return serializable state for checkpoint saving.

        Should include:
        - Network weights (RND predictor, ICM models)
        - Optimizer state
        - Current epsilon (if applicable)
        - Intrinsic weight (if adaptive)

        Returns:
            Dict compatible with torch.save()
        """
        pass

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """
        Restore exploration strategy from checkpoint.

        Args:
            state: Dict from checkpoint_state()

        Raises:
            ValueError: If state is invalid or incompatible
        """
        pass
