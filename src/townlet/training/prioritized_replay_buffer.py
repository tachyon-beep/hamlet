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

        # Storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

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

        # Combine extrinsic + intrinsic rewards
        rewards = rewards_extrinsic + rewards_intrinsic

        # Loop over batch and store each transition
        for i in range(batch_size):
            if self.size_current < self.capacity:
                self.observations.append(observations[i].cpu())
                self.actions.append(actions[i].cpu())
                self.rewards.append(rewards[i].cpu())
                self.next_observations.append(next_observations[i].cpu())
                self.dones.append(dones[i].cpu())
                self.size_current += 1
            else:
                # Overwrite oldest transition
                self.observations[self.position] = observations[i].cpu()
                self.actions[self.position] = actions[i].cpu()
                self.rewards[self.position] = rewards[i].cpu()
                self.next_observations[self.position] = next_observations[i].cpu()
                self.dones[self.position] = dones[i].cpu()

            # Assign max priority to new transition
            self.priorities[self.position] = self.max_priority

            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> dict:
        """Sample batch with priority-based sampling.

        Returns:
            Batch dict with keys: observations, actions, rewards,
            next_observations, dones, weights, indices
        """
        # Compute sampling probabilities from priorities
        priorities = self.priorities[: self.size_current]
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size_current, batch_size, p=probs, replace=False)

        # Compute importance sampling weights
        weights = (self.size_current * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize by max weight

        # Gather batch
        batch = {
            "observations": torch.stack([self.observations[i] for i in indices]).to(self.device),
            "actions": torch.stack([self.actions[i] for i in indices]).to(self.device),
            "rewards": torch.stack([self.rewards[i] for i in indices]).to(self.device),
            "next_observations": torch.stack([self.next_observations[i] for i in indices]).to(self.device),
            "dones": torch.stack([self.dones[i] for i in indices]).to(self.device),
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
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_annealing": self.beta_annealing,
            "observations": [obs.cpu() for obs in self.observations],
            "actions": [act.cpu() for act in self.actions],
            "rewards": [rew.cpu() for rew in self.rewards],
            "next_observations": [next_obs.cpu() for next_obs in self.next_observations],
            "dones": [done.cpu() for done in self.dones],
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
        self.observations = [obs.to(self.device) for obs in state["observations"]]
        self.actions = [act.to(self.device) for act in state["actions"]]
        self.rewards = [rew.to(self.device) for rew in state["rewards"]]
        self.next_observations = [next_obs.to(self.device) for next_obs in state["next_observations"]]
        self.dones = [done.to(self.device) for done in state["dones"]]
        self.priorities = state["priorities"].copy()
        self.max_priority = state["max_priority"]
        self.position = state["position"]
        self.size_current = state["size_current"]
