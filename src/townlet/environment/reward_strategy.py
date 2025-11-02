"""
Reward Strategy Module

Encapsulates reward calculation logic for the Hamlet environment.
Uses baseline-relative reward: reward = steps_lived - R

Where R is the expected survival time of a random-walking agent that never interacts.
"""

import torch


class RewardStrategy:
    """
    Calculates rewards relative to baseline random-walk survival.

    Reward Formula: reward = steps_lived - R

    Where R = baseline survival time (steps until death with no interactions)

    This provides:
    - Negative reward if agent does worse than random walk
    - Zero reward if agent matches random walk baseline
    - Positive reward if agent outperforms baseline (actual learning!)

    R is recalculated when curriculum stage changes (different depletion rates).
    """

    def __init__(self, device: torch.device, num_agents: int = 1):
        """
        Initialize reward strategy.

        Args:
            device: torch device for tensor operations
            num_agents: number of agents for vectorized baseline (P2.1)
        """
        self.device = device
        self.num_agents = num_agents

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
        baseline_steps: torch.Tensor | float | list[float],
    ) -> torch.Tensor:
        """
        Calculate baseline-relative rewards.

        Reward = steps_lived - R (baseline survival)

        Args:
            step_counts: [num_agents] current step count for each agent
            dones: [num_agents] whether each agent is dead
            baseline_steps: Baseline survival expectation(s)

        Returns:
            rewards: [num_agents] calculated rewards (can be positive or negative)
        """
        if step_counts.shape[0] != self.num_agents or dones.shape[0] != self.num_agents:
            raise ValueError(
                f"RewardStrategy expected tensors shaped [{self.num_agents}], got " f"step_counts={step_counts.shape}, dones={dones.shape}"
            )

        baseline_tensor = self._prepare_baseline_tensor(baseline_steps)

        rewards = torch.zeros(step_counts.shape[0], device=self.device)
        rewards = torch.where(
            dones,
            step_counts.float() - baseline_tensor,
            0.0,
        )

        return rewards

    def _prepare_baseline_tensor(self, baseline_steps: torch.Tensor | float | list[float]) -> torch.Tensor:
        """Normalise baseline input to [num_agents] float tensor on device."""
        if isinstance(baseline_steps, torch.Tensor):
            tensor = baseline_steps.to(self.device, dtype=torch.float32)
        elif isinstance(baseline_steps, (float, int)):
            tensor = torch.full((self.num_agents,), float(baseline_steps), dtype=torch.float32, device=self.device)
        elif isinstance(baseline_steps, list):
            tensor = torch.tensor(baseline_steps, dtype=torch.float32, device=self.device)
        else:
            raise TypeError(f"Unsupported baseline type: {type(baseline_steps)!r}")

        if tensor.shape != (self.num_agents,):
            raise ValueError(f"baseline tensor shape {tensor.shape} does not match num_agents={self.num_agents}")

        return tensor
