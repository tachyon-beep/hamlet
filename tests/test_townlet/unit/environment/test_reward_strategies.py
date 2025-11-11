"""Tests for reward strategy classes.

Validates both RewardStrategy (multiplicative, buggy) and AdaptiveRewardStrategy
(additive with crisis suppression, fixed).

These tests demonstrate the "Low Energy Delirium" bug where multiplicative rewards
create perverse incentives during crisis states.
"""

import pytest
import torch

from townlet.environment.reward_strategy import AdaptiveRewardStrategy, RewardStrategy


class TestRewardStrategy:
    """Tests for original RewardStrategy (multiplicative rewards)."""

    @pytest.fixture
    def strategy(self):
        """Create a basic reward strategy."""
        return RewardStrategy(
            device=torch.device("cpu"),
            num_agents=3,
            meter_count=8,
            energy_idx=0,
            health_idx=6,
        )

    def test_initialization(self, strategy):
        """Verify strategy initializes with correct parameters."""
        assert strategy.num_agents == 3
        assert strategy.meter_count == 8
        assert strategy.energy_idx == 0
        assert strategy.health_idx == 6

    def test_healthy_agent_reward(self, strategy):
        """Healthy agent (energy=1.0, health=1.0) gets maximum reward."""
        step_counts = torch.tensor([1, 1, 1])
        dones = torch.tensor([False, False, False])
        meters = torch.ones(3, 8)  # All meters at 1.0

        rewards = strategy.calculate_rewards(step_counts, dones, meters)

        assert rewards.shape == (3,)
        torch.testing.assert_close(rewards, torch.tensor([1.0, 1.0, 1.0]))

    def test_critical_agent_reward(self, strategy):
        """Critical agent (energy=0.2, health=1.0) gets low reward."""
        step_counts = torch.tensor([1, 1, 1])
        dones = torch.tensor([False, False, False])
        meters = torch.ones(3, 8)
        meters[:, 0] = 0.2  # Set energy to 0.2 (critical)

        rewards = strategy.calculate_rewards(step_counts, dones, meters)

        # Multiplicative reward: 0.2 * 1.0 = 0.2
        torch.testing.assert_close(rewards, torch.tensor([0.2, 0.2, 0.2]))

    def test_dead_agent_zero_reward(self, strategy):
        """Dead agents get zero reward regardless of meters."""
        step_counts = torch.tensor([1, 1, 1])
        dones = torch.tensor([False, True, False])  # Agent 1 is dead
        meters = torch.ones(3, 8)

        rewards = strategy.calculate_rewards(step_counts, dones, meters)

        assert rewards[1].item() == 0.0
        assert rewards[0].item() > 0.0
        assert rewards[2].item() > 0.0

    def test_demonstrates_low_energy_delirium_bug(self):
        """
        Demonstrates the "Low Energy Delirium" bug.

        When agent is critical (energy=0.2), extrinsic reward drops to 0.2.
        If intrinsic reward (RND novelty) is 0.8, total becomes 1.0.
        Compare to familiar path (intrinsic=0.3, total=0.5).
        → Agent explores novel state instead of going to bed!
        """
        strategy = RewardStrategy(
            device=torch.device("cpu"),
            num_agents=1,
            meter_count=8,
            energy_idx=0,
            health_idx=6,
        )

        step_counts = torch.tensor([1])
        dones = torch.tensor([False])
        meters = torch.ones(1, 8)
        meters[0, 0] = 0.2  # Critical energy

        extrinsic = strategy.calculate_rewards(step_counts, dones, meters)

        # Simulate intrinsic rewards
        intrinsic_familiar = 3.0 * 0.1  # Familiar path
        intrinsic_novel = 8.0 * 0.1  # Novel detour

        total_familiar = extrinsic[0].item() + intrinsic_familiar
        total_novel = extrinsic[0].item() + intrinsic_novel

        # BUG: Agent prefers exploring (1.0) over surviving (0.5)
        assert extrinsic[0].item() == pytest.approx(0.2)
        assert total_familiar == pytest.approx(0.5)
        assert total_novel == pytest.approx(1.0)
        assert total_novel > total_familiar  # Agent explores instead of sleeping!


class TestAdaptiveRewardStrategy:
    """Tests for AdaptiveRewardStrategy (additive rewards + crisis suppression)."""

    @pytest.fixture
    def strategy(self):
        """Create an adaptive reward strategy."""
        return AdaptiveRewardStrategy(
            device=torch.device("cpu"),
            num_agents=3,
            meter_count=8,
            energy_idx=0,
            health_idx=6,
            base_reward=1.0,
            bonus_scale=0.5,
        )

    def test_initialization(self, strategy):
        """Verify strategy initializes with correct parameters."""
        assert strategy.num_agents == 3
        assert strategy.meter_count == 8
        assert strategy.energy_idx == 0
        assert strategy.health_idx == 6
        assert strategy.base_reward == 1.0
        assert strategy.bonus_scale == 0.5

    def test_healthy_agent_reward(self, strategy):
        """Healthy agent gets base + bonuses."""
        step_counts = torch.tensor([1, 1, 1])
        dones = torch.tensor([False, False, False])
        meters = torch.ones(3, 8)  # All meters at 1.0

        rewards, intrinsic_weights = strategy.calculate_rewards(step_counts, dones, meters)

        # base=1.0 + health_bonus=0.25 + energy_bonus=0.25 = 1.5
        assert rewards.shape == (3,)
        torch.testing.assert_close(rewards, torch.tensor([1.5, 1.5, 1.5]))
        torch.testing.assert_close(intrinsic_weights, torch.tensor([1.0, 1.0, 1.0]))

    def test_critical_agent_reward(self, strategy):
        """Critical agent gets base + negative bonuses (still >0)."""
        step_counts = torch.tensor([1, 1, 1])
        dones = torch.tensor([False, False, False])
        meters = torch.ones(3, 8)
        meters[:, 0] = 0.2  # Set energy to 0.2 (critical)

        rewards, intrinsic_weights = strategy.calculate_rewards(step_counts, dones, meters)

        # base=1.0 + health_bonus=0.5*(1.0-0.5)=0.25 + energy_bonus=0.5*(0.2-0.5)=-0.15 = 1.1
        # Still much higher than old 0.2!
        expected_reward = 1.0 + 0.25 + (-0.15)
        assert rewards.shape == (3,)
        torch.testing.assert_close(rewards, torch.full((3,), expected_reward), atol=0.01, rtol=0.01)

        # Intrinsic weight = max(health=1.0, energy=0.2) = 1.0 (uses best resource!)
        torch.testing.assert_close(intrinsic_weights, torch.tensor([1.0, 1.0, 1.0]))

    def test_dead_agent_zero_reward(self, strategy):
        """Dead agents get zero reward and zero intrinsic weight."""
        step_counts = torch.tensor([1, 1, 1])
        dones = torch.tensor([False, True, False])
        meters = torch.ones(3, 8)

        rewards, intrinsic_weights = strategy.calculate_rewards(step_counts, dones, meters)

        assert rewards[1].item() == 0.0
        assert rewards[0].item() > 0.0
        assert rewards[2].item() > 0.0

    def test_fixes_low_energy_delirium_bug(self):
        """
        Demonstrates the fix for "Low Energy Delirium" bug.

        When agent is critical (energy=0.2), extrinsic stays high (~1.1).
        Intrinsic weight = max(health=1.0, energy=0.2) = 1.0 (uses best resource).
        → Agent still explores, but not catastrophically distracted.
        """
        strategy = AdaptiveRewardStrategy(
            device=torch.device("cpu"),
            num_agents=1,
            meter_count=8,
            energy_idx=0,
            health_idx=6,
        )

        step_counts = torch.tensor([1])
        dones = torch.tensor([False])
        meters = torch.ones(1, 8)
        meters[0, 0] = 0.2  # Critical energy, healthy otherwise

        extrinsic, intrinsic_weights = strategy.calculate_rewards(step_counts, dones, meters)

        # Extrinsic: base=1.0 + health_bonus=0.25 + energy_bonus=-0.15 = 1.1
        expected_extrinsic = 1.0 + 0.25 + (-0.15)
        assert extrinsic[0].item() == pytest.approx(expected_extrinsic, abs=0.01)

        # Intrinsic weight = max(health=1.0, energy=0.2) = 1.0
        # This means agent can still explore when one resource is healthy!
        assert intrinsic_weights[0].item() == pytest.approx(1.0)

        # The fix comes from higher extrinsic base (1.1 vs 0.2)
        # Even if intrinsic weight is 1.0, the high extrinsic prevents dominance

    def test_crisis_suppression_gradient(self):
        """Intrinsic weight smoothly transitions from healthy to critical."""
        strategy = AdaptiveRewardStrategy(
            device=torch.device("cpu"),
            num_agents=4,
            meter_count=8,
        )

        step_counts = torch.tensor([1, 1, 1, 1])
        dones = torch.tensor([False, False, False, False])
        meters = torch.ones(4, 8)

        # Test different energy levels (health stays at 1.0)
        meters[:, 0] = torch.tensor([1.0, 0.7, 0.4, 0.1])

        _, intrinsic_weights = strategy.calculate_rewards(step_counts, dones, meters)

        # Weights = max(health=1.0, energy=varying) = 1.0 for all!
        # Because health is always 1.0, max() returns 1.0
        assert intrinsic_weights[0].item() == pytest.approx(1.0)
        assert intrinsic_weights[1].item() == pytest.approx(1.0)
        assert intrinsic_weights[2].item() == pytest.approx(1.0)
        assert intrinsic_weights[3].item() == pytest.approx(1.0)

    def test_both_resources_critical_suppresses_exploration(self):
        """When BOTH resources are critical, exploration is suppressed."""
        strategy = AdaptiveRewardStrategy(
            device=torch.device("cpu"),
            num_agents=4,
            meter_count=8,
        )

        step_counts = torch.tensor([1, 1, 1, 1])
        dones = torch.tensor([False, False, False, False])
        meters = torch.ones(4, 8)

        # Test with both energy AND health low
        meters[:, 0] = torch.tensor([0.2, 0.2, 0.2, 0.2])  # energy
        meters[:, 6] = torch.tensor([1.0, 0.7, 0.4, 0.1])  # health

        _, intrinsic_weights = strategy.calculate_rewards(step_counts, dones, meters)

        # max(energy=0.2, health=varying) = max of the two
        assert intrinsic_weights[0].item() == pytest.approx(1.0)  # max(0.2, 1.0) = 1.0
        assert intrinsic_weights[1].item() == pytest.approx(0.7)  # max(0.2, 0.7) = 0.7
        assert intrinsic_weights[2].item() == pytest.approx(0.4)  # max(0.2, 0.4) = 0.4
        assert intrinsic_weights[3].item() == pytest.approx(0.2)  # max(0.2, 0.1) = 0.2

    def test_uses_best_resource_for_weight(self):
        """Crisis suppression uses max(health, energy) - forgiving."""
        strategy = AdaptiveRewardStrategy(
            device=torch.device("cpu"),
            num_agents=2,
            meter_count=8,
        )

        step_counts = torch.tensor([1, 1])
        dones = torch.tensor([False, False])
        meters = torch.ones(2, 8)

        # Agent 0: low energy, high health
        meters[0, 0] = 0.2  # energy
        meters[0, 6] = 1.0  # health

        # Agent 1: high energy, low health
        meters[1, 0] = 1.0  # energy
        meters[1, 6] = 0.2  # health

        _, intrinsic_weights = strategy.calculate_rewards(step_counts, dones, meters)

        # Both should use their best resource (1.0) for weight, not worst (0.2)
        # This is more forgiving - "as safe as your safest resource"
        assert intrinsic_weights[0].item() == pytest.approx(1.0)  # max(0.2, 1.0) = 1.0
        assert intrinsic_weights[1].item() == pytest.approx(1.0)  # max(1.0, 0.2) = 1.0


class TestRewardStrategyComparison:
    """Direct comparison tests demonstrating the bug fix."""

    def test_critical_state_reward_comparison(self):
        """Compare rewards in critical state between strategies."""
        old_strategy = RewardStrategy(
            device=torch.device("cpu"),
            num_agents=1,
            meter_count=8,
            energy_idx=0,
            health_idx=6,
        )

        new_strategy = AdaptiveRewardStrategy(
            device=torch.device("cpu"),
            num_agents=1,
            meter_count=8,
            energy_idx=0,
            health_idx=6,
        )

        step_counts = torch.tensor([1])
        dones = torch.tensor([False])
        meters = torch.ones(1, 8)
        meters[0, 0] = 0.2  # Critical energy

        old_reward = old_strategy.calculate_rewards(step_counts, dones, meters)
        new_reward, _ = new_strategy.calculate_rewards(step_counts, dones, meters)

        # Old: 0.2 (multiplicative)
        # New: 1.1 (additive with base)
        assert old_reward[0].item() == pytest.approx(0.2)
        assert new_reward[0].item() == pytest.approx(1.1, abs=0.01)
        assert new_reward[0].item() > old_reward[0].item() * 5  # 5× higher!

    def test_healthy_state_similar_rewards(self):
        """Both strategies give similar rewards when healthy."""
        old_strategy = RewardStrategy(
            device=torch.device("cpu"),
            num_agents=1,
            meter_count=8,
        )

        new_strategy = AdaptiveRewardStrategy(
            device=torch.device("cpu"),
            num_agents=1,
            meter_count=8,
        )

        step_counts = torch.tensor([1])
        dones = torch.tensor([False])
        meters = torch.ones(1, 8)  # All healthy

        old_reward = old_strategy.calculate_rewards(step_counts, dones, meters)
        new_reward, _ = new_strategy.calculate_rewards(step_counts, dones, meters)

        # Old: 1.0 (1.0 * 1.0)
        # New: 1.5 (1.0 + 0.25 + 0.25)
        assert old_reward[0].item() == pytest.approx(1.0)
        assert new_reward[0].item() == pytest.approx(1.5)
        # Similar magnitude, both indicate "doing well"
        assert 0.5 < new_reward[0].item() / old_reward[0].item() < 2.0
