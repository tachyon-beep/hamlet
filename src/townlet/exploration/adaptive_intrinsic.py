"""Adaptive intrinsic exploration with variance-based annealing."""

from typing import List, Dict, Any
import torch

from townlet.exploration.base import ExplorationStrategy
from townlet.exploration.rnd import RNDExploration
from townlet.training.state import BatchedAgentState


class AdaptiveIntrinsicExploration(ExplorationStrategy):
    """RND with adaptive annealing based on survival variance.

    Automatically reduces intrinsic weight when agent demonstrates
    consistent performance (low survival time variance).

    Composition: Contains RNDExploration instance for novelty computation.
    """

    def __init__(
        self,
        obs_dim: int = 70,
        embed_dim: int = 128,
        rnd_learning_rate: float = 1e-4,
        rnd_training_batch_size: int = 128,
        initial_intrinsic_weight: float = 1.0,
        min_intrinsic_weight: float = 0.0,
        variance_threshold: float = 100.0,  # Increased from 10.0 to prevent premature annealing
        survival_window: int = 100,
        decay_rate: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        device: torch.device = torch.device('cpu'),
    ):
        """Initialize adaptive intrinsic exploration.

        Args:
            obs_dim: Observation dimension
            embed_dim: RND embedding dimension
            rnd_learning_rate: Learning rate for RND predictor
            rnd_training_batch_size: Batch size for RND training
            initial_intrinsic_weight: Starting intrinsic weight
            min_intrinsic_weight: Minimum intrinsic weight (floor)
            variance_threshold: Variance threshold for annealing trigger
            survival_window: Number of episodes to track for variance
            decay_rate: Exponential decay rate (weight *= decay_rate)
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_min: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            device: Device for tensors
        """
        # RND instance (composition)
        self.rnd = RNDExploration(
            obs_dim=obs_dim,
            embed_dim=embed_dim,
            learning_rate=rnd_learning_rate,
            training_batch_size=rnd_training_batch_size,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            device=device,
        )

        # Annealing parameters
        self.current_intrinsic_weight = initial_intrinsic_weight
        self.min_intrinsic_weight = min_intrinsic_weight
        self.variance_threshold = variance_threshold
        self.survival_window = survival_window
        self.decay_rate = decay_rate
        self.device = device

        # Survival tracking
        self.survival_history: List[float] = []

    def select_actions(
        self,
        q_values: torch.Tensor,
        agent_states: BatchedAgentState,
        action_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select actions using epsilon-greedy with action masking (delegates to RND).

        Args:
            q_values: Q-values for each action [batch, num_actions]
            agent_states: Current state (contains per-agent epsilons)
            action_masks: Optional action validity masks [batch, num_actions] bool

        Returns:
            actions: [batch] selected actions
        """
        return self.rnd.select_actions(q_values, agent_states, action_masks)

    def compute_intrinsic_rewards(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted intrinsic rewards.

        Args:
            observations: [batch, obs_dim] state observations

        Returns:
            [batch] intrinsic rewards scaled by current weight
        """
        # Get RND novelty
        rnd_novelty = self.rnd.compute_intrinsic_rewards(observations)

        # Scale by current weight
        return rnd_novelty * self.current_intrinsic_weight

    def update(self, batch: Dict[str, torch.Tensor]) -> None:
        """Update RND predictor network from experience batch.

        Args:
            batch: Experience batch with 'observations' key
        """
        self.rnd.update(batch)

    def update_on_episode_end(self, survival_time: float) -> None:
        """Update survival history and check for annealing trigger.

        Call after each episode completes.

        Args:
            survival_time: Number of steps agent survived this episode
        """
        self.survival_history.append(survival_time)

        # Keep only recent window
        if len(self.survival_history) > self.survival_window:
            self.survival_history = self.survival_history[-self.survival_window:]

        # Check for annealing
        if self.should_anneal():
            self.anneal_weight()

    def should_anneal(self) -> bool:
        """Check if variance is below threshold AND performance is good.

        Returns:
            True if agent performance is consistent enough to reduce exploration
        """
        if len(self.survival_history) < self.survival_window:
            return False  # Not enough data

        recent_survivals = torch.tensor(
            self.survival_history[-self.survival_window:],
            dtype=torch.float32,
        )
        variance = torch.var(recent_survivals).item()
        mean_survival = torch.mean(recent_survivals).item()

        # DEFENSIVE: Only anneal if BOTH low variance AND good performance
        # Low variance + low survival = "consistently failing" (NOT ready to anneal)
        # Low variance + high survival = "consistently succeeding" (ready to anneal)
        MIN_SURVIVAL_FOR_ANNEALING = 50.0  # Don't anneal until surviving at least 50 steps

        return (variance < self.variance_threshold and
                mean_survival > MIN_SURVIVAL_FOR_ANNEALING)

    def anneal_weight(self) -> None:
        """Reduce intrinsic weight via exponential decay."""
        new_weight = self.current_intrinsic_weight * self.decay_rate
        self.current_intrinsic_weight = max(new_weight, self.min_intrinsic_weight)

    def get_intrinsic_weight(self) -> float:
        """Get current intrinsic weight (for logging/visualization)."""
        return self.current_intrinsic_weight

    def decay_epsilon(self) -> None:
        """Decay epsilon (delegates to RND)."""
        self.rnd.decay_epsilon()

    def checkpoint_state(self) -> Dict[str, Any]:
        """Return serializable state for checkpoint saving.

        Returns:
            Dict with RND state, intrinsic weight, and survival history
        """
        return {
            'rnd_state': self.rnd.checkpoint_state(),
            'current_intrinsic_weight': self.current_intrinsic_weight,
            'min_intrinsic_weight': self.min_intrinsic_weight,
            'variance_threshold': self.variance_threshold,
            'survival_window': self.survival_window,
            'decay_rate': self.decay_rate,
            'survival_history': self.survival_history,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore from checkpoint.

        Args:
            state: Dict from checkpoint_state()
        """
        self.rnd.load_state(state['rnd_state'])
        self.current_intrinsic_weight = state['current_intrinsic_weight']
        self.min_intrinsic_weight = state['min_intrinsic_weight']
        self.variance_threshold = state['variance_threshold']
        self.survival_window = state['survival_window']
        self.decay_rate = state['decay_rate']
        self.survival_history = state['survival_history']
