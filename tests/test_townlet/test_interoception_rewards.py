"""
Tests for interoception-based reward calculation.

Reward Formula:
- Alive: health × energy (both normalized to [0,1])
- Dead: 0.0

This models human interoception - we're aware of our internal state
and use that immediate feedback to guide behavior.
"""

import torch

from townlet.environment.reward_strategy import RewardStrategy


class TestInteroceptionRewards:
    """Test interoception-aware reward calculations."""

    def test_full_health_and_energy_gives_max_reward(self):
        """Both meters at 100% should give reward = 1.0"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=1)

        step_counts = torch.tensor([50], device=device)
        dones = torch.tensor([False], device=device)
        baseline = torch.tensor([100.0], device=device)

        # Meters: [energy, ..., health] (indices 0 and 6)
        # NOTE: Meters are stored normalized to [0, 1] in the environment
        meters = torch.zeros(1, 8, device=device)
        meters[0, 0] = 1.0  # energy = 100% (normalized to 1.0)
        meters[0, 6] = 1.0  # health = 100% (normalized to 1.0)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # health × energy = 1.0 × 1.0 = 1.0
        assert torch.isclose(rewards[0], torch.tensor(1.0))

    def test_half_health_and_energy_gives_quarter_reward(self):
        """Both meters at 50% should give reward = 0.25"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=1)

        step_counts = torch.tensor([50], device=device)
        dones = torch.tensor([False], device=device)
        baseline = torch.tensor([100.0], device=device)

        meters = torch.zeros(1, 8, device=device)
        meters[0, 0] = 0.5  # energy = 50% (normalized to 0.5)
        meters[0, 6] = 0.5  # health = 50% (normalized to 0.5)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # health × energy = 0.5 × 0.5 = 0.25
        assert torch.isclose(rewards[0], torch.tensor(0.25))

    def test_critical_energy_gives_low_reward(self):
        """Energy at 10% should give reward = 0.1"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=1)

        step_counts = torch.tensor([50], device=device)
        dones = torch.tensor([False], device=device)
        baseline = torch.tensor([100.0], device=device)

        meters = torch.zeros(1, 8, device=device)
        meters[0, 0] = 0.1  # energy = 10% (normalized to 0.1, CRITICAL!)
        meters[0, 6] = 1.0  # health = 100% (normalized to 1.0)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # health × energy = 1.0 × 0.1 = 0.1
        assert torch.isclose(rewards[0], torch.tensor(0.1))

    def test_critical_health_gives_low_reward(self):
        """Health at 10% should give reward = 0.1"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=1)

        step_counts = torch.tensor([50], device=device)
        dones = torch.tensor([False], device=device)
        baseline = torch.tensor([100.0], device=device)

        meters = torch.zeros(1, 8, device=device)
        meters[0, 0] = 1.0  # energy = 100% (normalized to 1.0)
        meters[0, 6] = 0.1  # health = 10% (normalized to 0.1, CRITICAL!)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # health × energy = 0.1 × 1.0 = 0.1
        assert torch.isclose(rewards[0], torch.tensor(0.1))

    def test_dead_gives_zero_reward(self):
        """Dead agents get 0.0 regardless of meter values"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=1)

        step_counts = torch.tensor([50], device=device)
        dones = torch.tensor([True], device=device)  # DEAD
        baseline = torch.tensor([100.0], device=device)

        meters = torch.zeros(1, 8, device=device)
        meters[0, 0] = 1.0  # energy = 100% (normalized to 1.0)
        meters[0, 6] = 1.0  # health = 100% (normalized to 1.0)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # Dead: 0.0 regardless of meters
        assert torch.isclose(rewards[0], torch.tensor(0.0))

    def test_reward_gradient_decreases_with_energy_depletion(self):
        """Reward should decrease smoothly as energy depletes"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=10)

        step_counts = torch.full((10,), 50, device=device)
        dones = torch.tensor([False] * 10, device=device)
        baseline = torch.full((10,), 100.0, device=device)

        # Energy levels: 100%, 90%, 80%, ..., 10% (normalized to [0, 1])
        meters = torch.zeros(10, 8, device=device)
        meters[:, 0] = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
        meters[:, 6] = 1.0  # health = 100% (normalized to 1.0)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # Rewards should be monotonically decreasing
        expected = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        assert torch.allclose(rewards, expected, atol=1e-6)

    def test_both_meters_low_gives_very_low_reward(self):
        """Both meters at 20% should give reward = 0.04"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=1)

        step_counts = torch.tensor([50], device=device)
        dones = torch.tensor([False], device=device)
        baseline = torch.tensor([100.0], device=device)

        meters = torch.zeros(1, 8, device=device)
        meters[0, 0] = 0.2  # energy = 20% (normalized to 0.2)
        meters[0, 6] = 0.2  # health = 20% (normalized to 0.2)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # health × energy = 0.2 × 0.2 = 0.04
        assert torch.isclose(rewards[0], torch.tensor(0.04))

    def test_multiple_agents_different_meter_states(self):
        """Each agent gets reward based on their own meter state"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([50, 50, 50], device=device)
        dones = torch.tensor([False, False, True], device=device)
        baseline = torch.full((3,), 100.0, device=device)

        meters = torch.zeros(3, 8, device=device)
        # Agent 0: full health/energy
        meters[0, 0] = 1.0  # energy = 100% (normalized to 1.0)
        meters[0, 6] = 1.0  # health = 100% (normalized to 1.0)
        # Agent 1: half health/energy
        meters[1, 0] = 0.5  # energy = 50% (normalized to 0.5)
        meters[1, 6] = 0.5  # health = 50% (normalized to 0.5)
        # Agent 2: dead (should get 0.0)
        meters[2, 0] = 1.0
        meters[2, 6] = 1.0

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # Agent 0: 1.0 × 1.0 = 1.0
        assert torch.isclose(rewards[0], torch.tensor(1.0))
        # Agent 1: 0.5 × 0.5 = 0.25
        assert torch.isclose(rewards[1], torch.tensor(0.25))
        # Agent 2: dead = 0.0
        assert torch.isclose(rewards[2], torch.tensor(0.0))

    def test_meters_clamped_to_valid_range(self):
        """Meters should be clamped to [0, 1] (already normalized)"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=2)

        step_counts = torch.tensor([50, 50], device=device)
        dones = torch.tensor([False, False], device=device)
        baseline = torch.full((2,), 100.0, device=device)

        meters = torch.zeros(2, 8, device=device)
        # Agent 0: meters > 1.0 (should clamp to 1.0)
        meters[0, 0] = 1.5
        meters[0, 6] = 1.2
        # Agent 1: meters < 0 (should clamp to 0)
        meters[1, 0] = -0.1
        meters[1, 6] = -0.05

        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # Agent 0: clamped to (1.0, 1.0) → 1.0 × 1.0 = 1.0
        assert torch.isclose(rewards[0], torch.tensor(1.0))
        # Agent 1: clamped to (0, 0) → 0.0 × 0.0 = 0.0
        assert torch.isclose(rewards[1], torch.tensor(0.0))

    def test_baseline_parameter_still_accepted(self):
        """Baseline parameter should be accepted for API compatibility"""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=1)

        step_counts = torch.tensor([50], device=device)
        dones = torch.tensor([False], device=device)
        baseline = torch.tensor([100.0], device=device)  # Should be accepted but not used

        meters = torch.zeros(1, 8, device=device)
        meters[0, 0] = 0.8  # energy = 80% (normalized to 0.8)
        meters[0, 6] = 0.8  # health = 80% (normalized to 0.8)

        # Should not raise error
        rewards = strategy.calculate_rewards(step_counts, dones, baseline, meters)

        # Reward based on meters, not baseline
        assert torch.isclose(rewards[0], torch.tensor(0.64))  # 0.8 × 0.8
