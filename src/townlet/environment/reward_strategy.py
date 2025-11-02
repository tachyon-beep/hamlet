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
        # P2.1: Vectorized baseline - one per agent for multi-agent curriculum support
        self.baseline_survival_steps = torch.full(
            (num_agents,), 100.0, dtype=torch.float32, device=device
        )

    def set_baseline_survival_steps(self, baseline_steps: torch.Tensor | float):
        """
        Set the baseline survival steps (R) for current curriculum stage.

        P2.1: Now accepts either:
        - torch.Tensor[num_agents]: Per-agent baselines (multi-agent curriculum)
        - float: Shared baseline (backwards compatibility, broadcasts to all agents)

        Args:
            baseline_steps: Expected survival time(s) of random-walking agent(s)
        """
        if isinstance(baseline_steps, torch.Tensor):
            # P2.1: Per-agent baselines
            assert baseline_steps.shape == (self.num_agents,), \
                f"baseline_steps must be [num_agents={self.num_agents}], got {baseline_steps.shape}"
            self.baseline_survival_steps = baseline_steps.to(self.device)
        else:
            # Backwards compatibility: broadcast scalar to all agents
            self.baseline_survival_steps = torch.full(
                (self.num_agents,), float(baseline_steps), dtype=torch.float32, device=self.device
            )

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
