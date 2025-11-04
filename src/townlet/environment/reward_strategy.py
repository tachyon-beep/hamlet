"""
Reward Strategy Module

Encapsulates reward calculation logic for the Hamlet environment.
Uses interoception-aware per-step survival rewards: health × energy.

This models human interoception - we're aware of our internal state
(tiredness, sickness) and use that immediate feedback to guide behavior.

Previous approaches:
- Flat per-step (+1.0 alive): No gradient for optimal resource timing
- Episodic (steps - baseline): Too sparse for minimal environments
"""

import torch


class RewardStrategy:
    """
    Calculates interoception-aware per-step survival rewards.

    Reward Formula:
    - Alive: health × energy (both normalized to [0,1])
    - Dead: 0.0

    This models human interoception - we're constantly aware of:
    - Fatigue (energy depletion)
    - Pain/sickness (health depletion)

    This immediate feedback creates a natural gradient for resource management.
    Agents learn WHEN to use resources (bed, hospital) based on ROI:
    - High energy (95%): reward ≈ 1.0 → ROI of sleep is LOW → wait
    - Low energy (20%): reward ≈ 0.2 → ROI of sleep is HIGH → act now

    Humans don't need to die to learn "low energy is bad" - we feel tired.
    This reward structure gives agents the same immediate awareness.

    Note: baseline_steps parameter is retained for API compatibility but
    is no longer used in reward calculation.
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
        meters: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate interoception-aware per-step survival rewards.

        Reward = health × energy (normalized) when alive, 0.0 when dead

        Args:
            step_counts: [num_agents] current step count for each agent
            dones: [num_agents] whether each agent is dead
            baseline_steps: Baseline survival expectation (retained for API
                           compatibility, not used in interoception rewards)
            meters: [num_agents, 8] meter values already normalized to [0,1]
                   (energy=0, health=6)

        Returns:
            rewards: [num_agents] calculated rewards (health × energy when alive, 0.0 when dead)
        """
        if step_counts.shape[0] != self.num_agents or dones.shape[0] != self.num_agents:
            raise ValueError(
                f"RewardStrategy expected tensors shaped [{self.num_agents}], got " f"step_counts={step_counts.shape}, dones={dones.shape}"
            )

        if meters.shape != (self.num_agents, 8):
            raise ValueError(
                f"RewardStrategy expected meters shaped [{self.num_agents}, 8], got {meters.shape}"
            )

        # Extract health and energy (already normalized to [0, 1] in environment)
        energy = meters[:, 0].clamp(min=0.0, max=1.0)  # Clamp to [0, 1]
        health = meters[:, 6].clamp(min=0.0, max=1.0)  # Clamp to [0, 1]

        # Interoception-aware reward: health × energy
        # (baseline_steps parameter retained for API compatibility but unused)
        rewards = torch.where(
            dones,
            0.0,              # Dead: no reward
            health * energy,  # Alive: modulated by internal state
        )

        return rewards
