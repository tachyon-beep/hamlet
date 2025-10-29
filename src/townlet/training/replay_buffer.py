"""Replay buffer for off-policy learning with dual rewards."""

from typing import Dict
import torch


class ReplayBuffer:
    """Circular buffer storing transitions with separate extrinsic/intrinsic rewards.

    Stores: (obs, action, reward_extrinsic, reward_intrinsic, next_obs, done)
    Samples: Random mini-batches with combined rewards
    """

    def __init__(self, capacity: int = 10000, device: torch.device = torch.device('cpu')):
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
        self.observations = None
        self.actions = None
        self.rewards_extrinsic = None
        self.rewards_intrinsic = None
        self.next_observations = None
        self.dones = None

    def push(
        self,
        observations: torch.Tensor,      # [batch, obs_dim]
        actions: torch.Tensor,           # [batch]
        rewards_extrinsic: torch.Tensor, # [batch]
        rewards_intrinsic: torch.Tensor, # [batch]
        next_observations: torch.Tensor, # [batch, obs_dim]
        dones: torch.Tensor,             # [batch]
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

    def sample(self, batch_size: int, intrinsic_weight: float) -> Dict[str, torch.Tensor]:
        """Sample random mini-batch with combined rewards.

        Args:
            batch_size: Number of transitions to sample
            intrinsic_weight: Weight for intrinsic rewards (0.0-1.0)

        Returns:
            Dictionary with keys: observations, actions, rewards, next_observations, dones
            'rewards' = rewards_extrinsic + rewards_intrinsic * intrinsic_weight
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer size ({self.size}) < batch_size ({batch_size})")

        # Random indices (sample without replacement if batch_size == size)
        if batch_size == self.size:
            indices = torch.randperm(self.size, device=self.device)
        else:
            indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # Combine rewards
        combined_rewards = (
            self.rewards_extrinsic[indices] +
            self.rewards_intrinsic[indices] * intrinsic_weight
        )

        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': combined_rewards,
            'next_observations': self.next_observations[indices],
            'dones': self.dones[indices],
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
