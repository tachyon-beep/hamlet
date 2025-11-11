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
                f"RewardStrategy expected tensors shaped [{self.num_agents}], got step_counts={step_counts.shape}, dones={dones.shape}"
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


class AdaptiveRewardStrategy:
    """
    Interoception-aware survival rewards with adaptive intrinsic motivation suppression.

    Fixes the "Low Energy Delirium" bug where multiplicative rewards create perverse
    incentives during crisis states (see docs/teachable_moments/low_energy_delerium.md).

    Reward Formula:
    - Base: 1.0 (constant survival value)
    - Health bonus: +0.5 * (health - 0.5)
    - Energy bonus: +0.5 * (energy - 0.5)
    - Total: base + health_bonus + energy_bonus

    Intrinsic Weight Adaptation (Crisis Suppression):
    - When healthy (energy=1.0, health=1.0): intrinsic_weight = 1.0 (explore freely!)
    - When critical (energy=0.2, health=1.0): intrinsic_weight = 0.2 (focus on survival!)
    - Formula: effective_weight = max(health, energy)

    Example Comparison (energy=0.2, health=1.0, intrinsic_base=0.1):

    OLD (RewardStrategy - BROKEN):
        Heading to bed (familiar, novelty=3.0):
            extrinsic: 0.2, intrinsic: 0.3, total: 0.5
        Novel detour (novelty=8.0):
            extrinsic: 0.2, intrinsic: 0.8, total: 1.0
        → Agent explores instead of sleeping! → DIES

    NEW (AdaptiveRewardStrategy - FIXED):
        Heading to bed (familiar, novelty=3.0):
            extrinsic: 0.95, intrinsic: 0.06 (3.0 * 0.1 * 0.2), total: 1.01
        Novel detour (novelty=8.0):
            extrinsic: 0.95, intrinsic: 0.16 (8.0 * 0.1 * 0.2), total: 1.11
        → Agent slightly prefers novelty (marginal) but still values survival → SURVIVES

    Pedagogical Value:
    This bug demonstrates:
    1. Reward shaping pitfalls (multiplicative vs additive)
    2. Exploration-exploitation tradeoffs under resource constraints
    3. How seemingly reasonable design choices create emergent failure modes
    4. The importance of testing reward structures under extreme states

    See Also:
    - docs/teachable_moments/low_energy_delerium.md (full analysis)
    - RewardStrategy (original implementation with bug)
    """

    def __init__(
        self,
        device: torch.device,
        num_agents: int,
        meter_count: int,
        energy_idx: int = 0,
        health_idx: int = 6,
        base_reward: float = 1.0,
        bonus_scale: float = 0.5,
    ):
        """
        Initialize adaptive reward strategy.

        Args:
            device: torch device for tensor operations
            num_agents: number of agents
            meter_count: number of meters in the universe
            energy_idx: index of energy meter (default 0)
            health_idx: index of health meter (default 6)
            base_reward: constant base survival reward (default 1.0)
            bonus_scale: multiplier for health/energy bonuses (default 0.5)

        Note (PDR-002):
            All parameters must be explicitly provided (no UAC defaults).
            energy_idx and health_idx have semantic defaults (fallback indices).
        """
        self.device = device
        self.num_agents = num_agents
        self.meter_count = meter_count
        self.energy_idx = energy_idx
        self.health_idx = health_idx
        self.base_reward = base_reward
        self.bonus_scale = bonus_scale

    def calculate_rewards(
        self,
        step_counts: torch.Tensor,
        dones: torch.Tensor,
        meters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate adaptive interoception-aware survival rewards.

        Returns both extrinsic rewards and per-agent intrinsic weight multipliers
        for crisis suppression of exploration.

        Args:
            step_counts: [num_agents] current step count for each agent
            dones: [num_agents] whether each agent is dead
            meters: [num_agents, meter_count] meter values normalized to [0,1]

        Returns:
            rewards: [num_agents] extrinsic rewards (base + bonuses when alive, 0 when dead)
            intrinsic_weights: [num_agents] adaptive intrinsic weight multipliers
                              (1.0 when healthy, <1.0 when critical for crisis suppression)
        """
        if step_counts.shape[0] != self.num_agents or dones.shape[0] != self.num_agents:
            raise ValueError(
                f"AdaptiveRewardStrategy expected tensors shaped [{self.num_agents}], "
                f"got step_counts={step_counts.shape}, dones={dones.shape}"
            )

        if meters.shape != (self.num_agents, self.meter_count):
            raise ValueError(f"AdaptiveRewardStrategy expected meters shaped [{self.num_agents}, {self.meter_count}], got {meters.shape}")

        # Extract health and energy (already normalized to [0, 1])
        energy = meters[:, self.energy_idx].clamp(min=0.0, max=1.0)
        health = meters[:, self.health_idx].clamp(min=0.0, max=1.0)

        # Additive reward structure (prevents intrinsic dominance in crisis)
        base = torch.full_like(energy, self.base_reward)
        health_bonus = self.bonus_scale * (health - 0.5)
        energy_bonus = self.bonus_scale * (energy - 0.5)

        extrinsic_rewards = torch.where(
            dones,
            torch.zeros_like(energy),  # Dead: no reward
            base + health_bonus + energy_bonus,  # Alive: base + bonuses
        )

        # Adaptive intrinsic weight (crisis suppression)
        # When critical (low health OR low energy), suppress exploration
        # max() means "as safe as your best resource" (more forgiving than min)
        resource_state = torch.maximum(health, energy)
        intrinsic_weights = resource_state  # Range: [0.0, 1.0]

        return extrinsic_rewards, intrinsic_weights
