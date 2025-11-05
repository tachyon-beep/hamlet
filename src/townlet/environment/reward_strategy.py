"""
Reward Strategy Module

Encapsulates reward calculation logic for the Hamlet environment.
Uses interoception-aware per-step survival rewards: health × energy.

This models human interoception - we're aware of our internal state
(tiredness, sickness) and use that immediate feedback to guide behavior.

Previous approaches:
- Flat per-step (+1.0 alive): No gradient for optimal resource timing
- Episodic rewards: Too sparse for minimal environments
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
    """

    def __init__(self, device: torch.device, num_agents: int, meter_count: int, energy_idx: int = 0, health_idx: int = 6):
        """
        Initialize reward strategy.

        Args:
            device: torch device for tensor operations
            num_agents: number of agents
            meter_count: number of meters in the universe (TASK-001)
            energy_idx: index of energy meter (default 0 - semantic fallback to first meter)
            health_idx: index of health meter (default 6 - semantic fallback to 6th meter)

        Note (PDR-002):
            num_agents and meter_count must be explicitly provided (no UAC defaults).
            energy_idx and health_idx have semantic defaults (fallback meter indices).
        """
        self.device = device
        self.num_agents = num_agents
        self.meter_count = meter_count
        self.energy_idx = energy_idx
        self.health_idx = health_idx

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
        meters: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate interoception-aware per-step survival rewards.

        Reward = health × energy (normalized) when alive, 0.0 when dead

        Args:
            step_counts: [num_agents] current step count for each agent
            dones: [num_agents] whether each agent is dead
            meters: [num_agents, meter_count] meter values already normalized to [0,1]
                   (energy at energy_idx, health at health_idx)

        Returns:
            rewards: [num_agents] calculated rewards (health × energy when alive, 0.0 when dead)
        """
        if step_counts.shape[0] != self.num_agents or dones.shape[0] != self.num_agents:
            raise ValueError(
                f"RewardStrategy expected tensors shaped [{self.num_agents}], got " f"step_counts={step_counts.shape}, dones={dones.shape}"
            )

        if meters.shape != (self.num_agents, self.meter_count):
            raise ValueError(f"RewardStrategy expected meters shaped [{self.num_agents}, {self.meter_count}], got {meters.shape}")

        # Extract health and energy (already normalized to [0, 1] in environment)
        energy = meters[:, self.energy_idx].clamp(min=0.0, max=1.0)  # Clamp to [0, 1]
        health = meters[:, self.health_idx].clamp(min=0.0, max=1.0)  # Clamp to [0, 1]

        # Interoception-aware reward: health × energy
        rewards = torch.where(
            dones,
            0.0,  # Dead: no reward
            health * energy,  # Alive: modulated by internal state
        )

        return rewards
