"""Random Network Distillation (RND) for intrinsic motivation."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from townlet.exploration.action_selection import epsilon_greedy_action_selection
from townlet.exploration.base import ExplorationStrategy
from townlet.training.state import BatchedAgentState


class RNDNetwork(nn.Module):
    """3-layer MLP for RND embeddings.

    Architecture: [obs_dim → 256 → 128 → embed_dim]
    Matches SimpleQNetwork architecture for consistency.
    """

    def __init__(self, obs_dim: int = 70, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform observations to embeddings.

        Args:
            x: [batch, obs_dim] observations

        Returns:
            [batch, embed_dim] embeddings
        """
        return self.net(x)


class RNDExploration(ExplorationStrategy):
    """Random Network Distillation for novelty-based intrinsic rewards.

    Uses prediction error as intrinsic reward signal:
    - High error = novel state = high intrinsic reward
    - Low error = familiar state = low intrinsic reward
    """

    def __init__(
        self,
        obs_dim: int = 70,
        embed_dim: int = 128,
        learning_rate: float = 1e-4,
        training_batch_size: int = 128,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize RND with fixed and predictor networks.

        Args:
            obs_dim: Observation dimension
            embed_dim: Embedding dimension
            learning_rate: Learning rate for predictor network
            training_batch_size: Batch size for predictor training
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_min: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            device: Device for tensors
        """
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim
        self.training_batch_size = training_batch_size
        self.device = device

        # Epsilon parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Fixed network (random, frozen)
        self.fixed_network = RNDNetwork(obs_dim, embed_dim).to(device)
        for param in self.fixed_network.parameters():
            param.requires_grad = False

        # Predictor network (trained to match fixed)
        self.predictor_network = RNDNetwork(obs_dim, embed_dim).to(device)

        # Optimizer for predictor
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=learning_rate)

        # Observation buffer for mini-batch training
        self.obs_buffer = []
        self.step_counter = 0

    def select_actions(
        self,
        q_values: torch.Tensor,
        agent_states: BatchedAgentState,
        action_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select actions using epsilon-greedy with action masking.

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
        """Compute RND novelty signal (prediction error).

        Args:
            observations: [batch, obs_dim] state observations

        Returns:
            [batch] intrinsic rewards (MSE between fixed and predictor)
        """
        with torch.no_grad():
            target_features = self.fixed_network(observations)
            predicted_features = self.predictor_network(observations)
            # MSE per sample (high error = novel = high reward)
            mse_per_sample = ((target_features - predicted_features) ** 2).mean(dim=1)

        return mse_per_sample

    def update(self, batch: dict[str, torch.Tensor]) -> None:
        """Update predictor network from experience batch.

        Args:
            batch: Experience batch with 'observations' key
        """
        if "observations" not in batch:
            return

        observations = batch["observations"]

        # Add to buffer
        for i in range(observations.shape[0]):
            self.obs_buffer.append(observations[i].detach())

        # Train if buffer is full
        if len(self.obs_buffer) >= self.training_batch_size:
            self.update_predictor()

    def update_predictor(self) -> float:
        """Train predictor network on accumulated observations.

        Called every training_batch_size steps.

        Returns:
            Prediction loss (for logging)
        """
        if len(self.obs_buffer) < self.training_batch_size:
            return 0.0

        # Stack observations into batch
        obs_batch = torch.stack(self.obs_buffer[: self.training_batch_size]).to(self.device)

        # Clear buffer
        self.obs_buffer = self.obs_buffer[self.training_batch_size :]

        # Compute loss
        target = self.fixed_network(obs_batch).detach()
        predicted = self.predictor_network(obs_batch)
        loss = F.mse_loss(predicted, target)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_novelty_map(self, grid_size: int = 8) -> torch.Tensor:
        """Get novelty values for all grid positions (for visualization).

        Args:
            grid_size: Size of environment grid

        Returns:
            [grid_size, grid_size] tensor of novelty values
        """
        # Generate observations for all grid positions
        # (Simplified: just grid encoding, meters set to 0.5)
        novelty_map = torch.zeros(grid_size, grid_size, device=self.device)

        for row in range(grid_size):
            for col in range(grid_size):
                # Create observation with agent at (row, col)
                obs = torch.zeros(1, self.obs_dim, device=self.device)

                # Grid encoding (one-hot for position)
                flat_idx = row * grid_size + col
                obs[0, flat_idx] = 1.0

                # Meters (placeholder: all 0.5)
                obs[0, 64:70] = 0.5

                # Compute novelty
                novelty = self.compute_intrinsic_rewards(obs)
                novelty_map[row, col] = novelty.item()

        return novelty_map

    def decay_epsilon(self) -> None:
        """Decay epsilon (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def checkpoint_state(self) -> dict[str, Any]:
        """Return serializable state for checkpoint saving.

        Returns:
            Dict with network weights, optimizer state, and epsilon
        """
        return {
            "fixed_network": self.fixed_network.state_dict(),
            "predictor_network": self.predictor_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "obs_dim": self.obs_dim,
            "embed_dim": self.embed_dim,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore from checkpoint.

        Args:
            state: Dict from checkpoint_state()
        """
        self.fixed_network.load_state_dict(state["fixed_network"])
        self.predictor_network.load_state_dict(state["predictor_network"])

        # Gracefully handle missing optimizer (backwards compatibility)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

        self.epsilon = state["epsilon"]
        self.epsilon_min = state["epsilon_min"]
        self.epsilon_decay = state["epsilon_decay"]
