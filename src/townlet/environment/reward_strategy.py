"""
Reward Strategy Module

Encapsulates reward calculation logic for the Hamlet environment.
Supports milestone-based survival rewards to prevent reward hacking.
"""
import torch


class RewardStrategy:
    """
    Calculates rewards based on survival milestones.

    Problem with constant per-step rewards: Rewards aimless wandering equally to strategic play.
    Solution: Milestone bonuses that reward longevity without constant accumulation.

    Reward Structure:
    - Every 10 steps: +0.5 ("you're making progress!")
    - Every 100 steps: +5.0 ("happy birthday!" 🎂)
    - Death: -100.0

    This prevents left-right oscillation from being rewarded while still encouraging survival.

    Note: Step 0 triggers both milestones (0 % 10 == 0 AND 0 % 100 == 0) for 5.5 reward.
    """

    def __init__(self, device: torch.device):
        """
        Initialize reward strategy.

        Args:
            device: torch device for tensor operations
        """
        self.device = device

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate milestone-based survival rewards.

        Args:
            step_counts: [num_agents] current step count for each agent
            dones: [num_agents] whether each agent is dead

        Returns:
            rewards: [num_agents] calculated rewards
        """
        num_agents = step_counts.shape[0]
        rewards = torch.zeros(num_agents, device=self.device)

        # Milestone bonuses (only for alive agents)
        alive_mask = ~dones

        # Every 10 steps: +0.5 bonus
        decade_milestone = (step_counts % 10 == 0) & alive_mask
        rewards += torch.where(decade_milestone, 0.5, 0.0)

        # Every 100 steps: +5.0 bonus ("Happy Birthday!")
        century_milestone = (step_counts % 100 == 0) & alive_mask
        rewards += torch.where(century_milestone, 5.0, 0.0)

        # Death penalty: -100.0
        rewards = torch.where(dones, -100.0, rewards)

        return rewards
