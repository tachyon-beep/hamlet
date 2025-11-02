"""Replay buffer for off-policy learning with dual rewards."""

from __future__ import annotations

from typing import Any

import torch


class ReplayBuffer:
    """Circular buffer storing transitions with separate extrinsic/intrinsic rewards.

    Stores: (obs, action, reward_extrinsic, reward_intrinsic, next_obs, done)
    Samples: Random mini-batches with combined rewards
    """

    def __init__(self, capacity: int = 10000, device: torch.device = torch.device("cpu")):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Device for tensor storage (CPU or CUDA)
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Storage tensors (initialized on first push)
        self.observations: torch.Tensor | None = None
        self.actions: torch.Tensor | None = None
        self.rewards_extrinsic: torch.Tensor | None = None
        self.rewards_intrinsic: torch.Tensor | None = None
        self.next_observations: torch.Tensor | None = None
        self.dones: torch.Tensor | None = None

    def push(
        self,
        observations: torch.Tensor,  # [batch, obs_dim]
        actions: torch.Tensor,  # [batch]
        rewards_extrinsic: torch.Tensor,  # [batch]
        rewards_intrinsic: torch.Tensor,  # [batch]
        next_observations: torch.Tensor,  # [batch, obs_dim]
        dones: torch.Tensor,  # [batch]
    ) -> None:
        """Add batch of transitions to buffer.

        Uses FIFO eviction when buffer is full.
        """
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]

        # Initialize storage on first push
        if self.observations is None:
            self.observations = torch.zeros(self.capacity, obs_dim, device=self.device)
            self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
            self.rewards_extrinsic = torch.zeros(self.capacity, device=self.device)
            self.rewards_intrinsic = torch.zeros(self.capacity, device=self.device)
            self.next_observations = torch.zeros(self.capacity, obs_dim, device=self.device)
            self.dones = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)

        # Mypy: guard attributes after allocation
        assert self.observations is not None
        assert self.actions is not None
        assert self.rewards_extrinsic is not None
        assert self.rewards_intrinsic is not None
        assert self.next_observations is not None
        assert self.dones is not None

        # Move tensors to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        rewards_extrinsic = rewards_extrinsic.to(self.device)
        rewards_intrinsic = rewards_intrinsic.to(self.device)
        next_observations = next_observations.to(self.device)
        dones = dones.to(self.device)

        # Circular buffer logic
        for i in range(batch_size):
            idx = self.position % self.capacity

            self.observations[idx] = observations[i]
            self.actions[idx] = actions[i]
            self.rewards_extrinsic[idx] = rewards_extrinsic[i]
            self.rewards_intrinsic[idx] = rewards_intrinsic[i]
            self.next_observations[idx] = next_observations[i]
            self.dones[idx] = dones[i]

            self.position += 1
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, intrinsic_weight: float) -> dict[str, torch.Tensor]:
        """Sample random mini-batch with combined rewards.

        Args:
            batch_size: Number of transitions to sample
            intrinsic_weight: Weight for intrinsic rewards (0.0-1.0)

        Returns:
            Dictionary with keys: observations, actions, rewards, next_observations, dones, mask
            'rewards' = rewards_extrinsic + rewards_intrinsic * intrinsic_weight
            'mask' = bool tensor [batch_size] (all True for feed-forward training)
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer size ({self.size}) < batch_size ({batch_size})")

        # Random indices (sample without replacement if batch_size == size)
        if batch_size == self.size:
            indices = torch.randperm(self.size, device=self.device)
        else:
            indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # Combine rewards
        assert self.observations is not None
        assert self.actions is not None
        assert self.rewards_extrinsic is not None
        assert self.rewards_intrinsic is not None
        assert self.next_observations is not None
        assert self.dones is not None

        combined_rewards = self.rewards_extrinsic[indices] + self.rewards_intrinsic[indices] * intrinsic_weight

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": combined_rewards,
            "next_observations": self.next_observations[indices],
            "dones": self.dones[indices],
            "mask": torch.ones(batch_size, dtype=torch.bool, device=self.device),
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def serialize(self) -> dict[str, Any]:
        """
        Serialize buffer contents for checkpointing (P1.1).

        Returns:
            Dictionary with all buffer state on CPU for saving
        """
        if self.observations is None:
            # Empty buffer
            return {
                "size": 0,
                "position": 0,
                "capacity": self.capacity,
                "observations": None,
                "actions": None,
                "rewards_extrinsic": None,
                "rewards_intrinsic": None,
                "next_observations": None,
                "dones": None,
            }

        assert self.observations is not None
        assert self.actions is not None
        assert self.rewards_extrinsic is not None
        assert self.rewards_intrinsic is not None
        assert self.next_observations is not None
        assert self.dones is not None

        return {
            "size": self.size,
            "position": self.position,
            "capacity": self.capacity,
            "observations": self.observations[: self.size].cpu(),
            "actions": self.actions[: self.size].cpu(),
            "rewards_extrinsic": self.rewards_extrinsic[: self.size].cpu(),
            "rewards_intrinsic": self.rewards_intrinsic[: self.size].cpu(),
            "next_observations": self.next_observations[: self.size].cpu(),
            "dones": self.dones[: self.size].cpu(),
        }

    def load_from_serialized(self, state: dict[str, Any]) -> None:
        """
        Restore buffer from serialized state (P1.1).

        Args:
            state: Dictionary from serialize()
        """
        if state["observations"] is None:
            # Empty buffer
            self.size = 0
            self.position = 0
            return

        self.size = state["size"]
        self.position = state["position"]

        # Initialize storage if needed
        obs_dim = state["observations"].shape[1]
        if self.observations is None:
            self.observations = torch.zeros(self.capacity, obs_dim, device=self.device)
            self.actions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)
            self.rewards_extrinsic = torch.zeros(self.capacity, device=self.device)
            self.rewards_intrinsic = torch.zeros(self.capacity, device=self.device)
            self.next_observations = torch.zeros(self.capacity, obs_dim, device=self.device)
            self.dones = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)

        assert self.observations is not None
        assert self.actions is not None
        assert self.rewards_extrinsic is not None
        assert self.rewards_intrinsic is not None
        assert self.next_observations is not None
        assert self.dones is not None

        # Restore data
        self.observations[: self.size] = state["observations"].to(self.device)
        self.actions[: self.size] = state["actions"].to(self.device)
        self.rewards_extrinsic[: self.size] = state["rewards_extrinsic"].to(self.device)
        self.rewards_intrinsic[: self.size] = state["rewards_intrinsic"].to(self.device)
        self.next_observations[: self.size] = state["next_observations"].to(self.device)
        self.dones[: self.size] = state["dones"].to(self.device)
