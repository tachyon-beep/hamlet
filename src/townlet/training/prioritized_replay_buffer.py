"""Prioritized Experience Replay buffer (Schaul et al. 2016)."""

from __future__ import annotations

import numpy as np
import torch


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with TD-error-based sampling.

    Samples transitions proportional to their TD error (priority).
    High TD-error transitions are sampled more frequently.

    Reference: Schaul et al. 2016 - "Prioritized Experience Replay"
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: bool = True,
        device: torch.device | None = None,
    ):
        """Initialize prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0=uniform, 1=full prioritization)
            beta: Importance sampling exponent (anneals to 1.0)
            beta_annealing: Whether to anneal beta to 1.0 over training
            device: PyTorch device
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.device = device if device else torch.device("cpu")

        # Storage tensors (initialized on first push)
        self.observations: torch.Tensor | None = None
        self.actions: torch.Tensor | None = None
        self.rewards: torch.Tensor | None = None
        self.next_observations: torch.Tensor | None = None
        self.dones: torch.Tensor | None = None

        # Priorities (TD errors)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0  # Initial priority for new transitions
        self.position = 0
        self.size_current = 0

    def push(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards_extrinsic: torch.Tensor,
        rewards_intrinsic: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Add batch of transitions to buffer with max priority.

        Args:
            observations: [batch, obs_dim] observations
            actions: [batch] actions
            rewards_extrinsic: [batch] extrinsic rewards
            rewards_intrinsic: [batch] intrinsic rewards
            next_observations: [batch, obs_dim] next observations
            dones: [batch] done flags
        """
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]

        # Initialize storage on first push
        if self.observations is None:
            self.observations = torch.zeros(self.capacity, obs_dim, device=self.device)
            self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
            self.rewards = torch.zeros(self.capacity, device=self.device)
            self.next_observations = torch.zeros(self.capacity, obs_dim, device=self.device)
            self.dones = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)

        # Mypy: guard attributes after allocation
        assert self.observations is not None
        assert self.actions is not None
        assert self.rewards is not None
        assert self.next_observations is not None
        assert self.dones is not None

        # Move tensors to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        next_observations = next_observations.to(self.device)
        dones = dones.to(self.device)

        # Combine extrinsic + intrinsic rewards (PER needs combined rewards for TD error)
        rewards = (rewards_extrinsic + rewards_intrinsic).to(self.device)

        # Loop over batch and store each transition
        for i in range(batch_size):
            idx = self.position % self.capacity

            # Direct tensor indexing (no list operations, no device churn)
            self.observations[idx] = observations[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_observations[idx] = next_observations[i]
            self.dones[idx] = dones[i]

            # Assign max priority to new transition
            self.priorities[idx] = self.max_priority

            self.position = (self.position + 1) % self.capacity
            self.size_current = min(self.size_current + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """Sample batch with priority-based sampling.

        Returns:
            Batch dict with keys: observations, actions, rewards,
            next_observations, dones, weights, indices
        """
        # Guard: buffer must have enough transitions
        if self.size_current < batch_size:
            raise ValueError(f"Buffer size ({self.size_current}) < batch_size ({batch_size})")

        # Compute sampling probabilities from priorities
        priorities = self.priorities[: self.size_current]
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size_current, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (self.size_current * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize by max weight

        # Mypy: guard attributes
        assert self.observations is not None
        assert self.actions is not None
        assert self.rewards is not None
        assert self.next_observations is not None
        assert self.dones is not None

        # Convert indices to tensor for vectorized gathering
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)

        # Gather batch (direct tensor indexing, no stacking or device transfer)
        batch = {
            "observations": self.observations[indices_tensor],
            "actions": self.actions[indices_tensor],
            "rewards": self.rewards[indices_tensor],
            "next_observations": self.next_observations[indices_tensor],
            "dones": self.dones[indices_tensor],
            "weights": torch.tensor(weights, dtype=torch.float32, device=self.device),
            "indices": indices,
        }

        return batch

    def update_priorities(self, indices: np.ndarray, td_errors: torch.Tensor) -> None:
        """Update priorities for sampled transitions.

        Args:
            indices: Indices of sampled transitions
            td_errors: Absolute TD errors (|Q_target - Q_pred|)
        """
        td_errors_np = td_errors.detach().cpu().numpy()

        for idx, td_error in zip(indices, td_errors_np):
            self.priorities[idx] = abs(td_error) + 1e-6  # Small epsilon to avoid zero priority

        self.max_priority = max(self.max_priority, self.priorities[: self.size_current].max())

    def anneal_beta(self, total_steps: int, current_step: int) -> None:
        """Anneal beta toward 1.0 over training.

        Args:
            total_steps: Total training steps
            current_step: Current training step
        """
        if self.beta_annealing:
            progress = min(current_step / total_steps, 1.0)
            self.beta = 0.4 + (1.0 - 0.4) * progress  # Anneal from 0.4 to 1.0

    def size(self) -> int:
        """Return current buffer size."""
        return self.size_current

    def __len__(self) -> int:
        """Return current buffer size (required by VectorizedPopulation)."""
        return self.size_current

    def serialize(self) -> dict:
        """Serialize buffer contents for checkpointing.

        Returns:
            Dictionary containing buffer state (observations, priorities, metadata)
        """
        if self.observations is None:
            # Empty buffer
            return {
                "capacity": self.capacity,
                "alpha": self.alpha,
                "beta": self.beta,
                "beta_annealing": self.beta_annealing,
                "observations": None,
                "actions": None,
                "rewards": None,
                "next_observations": None,
                "dones": None,
                "priorities": self.priorities.copy(),
                "max_priority": self.max_priority,
                "position": self.position,
                "size_current": self.size_current,
            }

        assert self.observations is not None
        assert self.actions is not None
        assert self.rewards is not None
        assert self.next_observations is not None
        assert self.dones is not None

        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_annealing": self.beta_annealing,
            "observations": self.observations[: self.size_current].cpu(),
            "actions": self.actions[: self.size_current].cpu(),
            "rewards": self.rewards[: self.size_current].cpu(),
            "next_observations": self.next_observations[: self.size_current].cpu(),
            "dones": self.dones[: self.size_current].cpu(),
            "priorities": self.priorities.copy(),
            "max_priority": self.max_priority,
            "position": self.position,
            "size_current": self.size_current,
        }

    def load_from_serialized(self, state: dict) -> None:
        """Restore buffer from serialized state.

        Args:
            state: Dictionary from serialize()
        """
        self.capacity = state["capacity"]
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.beta_annealing = state["beta_annealing"]
        self.priorities = state["priorities"].copy()
        self.max_priority = state["max_priority"]
        self.position = state["position"]
        self.size_current = state["size_current"]

        if state["observations"] is None:
            # Empty buffer
            self.observations = None
            self.actions = None
            self.rewards = None
            self.next_observations = None
            self.dones = None
            return

        # Initialize storage if needed
        obs_dim = state["observations"].shape[1]
        if self.observations is None:
            self.observations = torch.zeros(self.capacity, obs_dim, device=self.device)
            self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
            self.rewards = torch.zeros(self.capacity, device=self.device)
            self.next_observations = torch.zeros(self.capacity, obs_dim, device=self.device)
            self.dones = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)

        assert self.observations is not None
        assert self.actions is not None
        assert self.rewards is not None
        assert self.next_observations is not None
        assert self.dones is not None

        # Restore data
        self.observations[: self.size_current] = state["observations"].to(self.device)
        self.actions[: self.size_current] = state["actions"].to(self.device)
        self.rewards[: self.size_current] = state["rewards"].to(self.device)
        self.next_observations[: self.size_current] = state["next_observations"].to(self.device)
        self.dones[: self.size_current] = state["dones"].to(self.device)
