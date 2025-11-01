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

    def __init__(self, device: torch.device):
        """
        Initialize reward strategy.

        Args:
            device: torch device for tensor operations
        """
        self.device = device
        self.baseline_survival_steps = 100.0  # Default R, updated via set_baseline()

    def set_baseline_survival_steps(self, baseline_steps: float):
        """
        Set the baseline survival steps (R) for current curriculum stage.

        Args:
            baseline_steps: Expected survival time of random-walking agent
        """
        self.baseline_survival_steps = baseline_steps

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate baseline-relative rewards.

        Reward = steps_lived - R (baseline survival)

        Args:
            step_counts: [num_agents] current step count for each agent
            dones: [num_agents] whether each agent is dead

        Returns:
            rewards: [num_agents] calculated rewards (can be positive or negative)
        """
        # Reward = steps survived - baseline
        # Only give reward on death (terminal state)
        rewards = torch.zeros(step_counts.shape[0], device=self.device)

        # On death: reward = steps_survived - R
        rewards = torch.where(
            dones,
            step_counts.float() - self.baseline_survival_steps,
            0.0,  # No reward for ongoing episode
        )

        return rewards
