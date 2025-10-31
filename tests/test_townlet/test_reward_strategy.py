"""
Characterization tests for reward calculation (RED phase baseline).

These tests document the actual behavior of the reward system BEFORE extraction,
to ensure zero behavioral changes during refactoring.
"""
import torch
import pytest

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.environment.reward_strategy import RewardStrategy


class TestRewardCalculationViaEnv:
    """Test reward calculation through environment (legacy interface)."""

    def test_milestone_every_10_steps(self):
        """Every 10 steps: +0.5 bonus for alive agents."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()
        
        # Test at steps 10 and 20
        env.step_counts = torch.tensor([10, 20])
        env.dones = torch.tensor([False, False])
        
        rewards = env._calculate_shaped_rewards()
        
        # Both get decade milestone bonus
        assert torch.isclose(rewards[0], torch.tensor(0.5))
        assert torch.isclose(rewards[1], torch.tensor(0.5))

    def test_milestone_every_100_steps(self):
        """Every 100 steps: +5.5 bonus (decade +0.5 + century +5.0)."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()
        
        # Test at steps 100 and 200
        env.step_counts = torch.tensor([100, 200])
        env.dones = torch.tensor([False, False])
        
        rewards = env._calculate_shaped_rewards()
        
        # Both get decade + century bonuses: 0.5 + 5.0 = 5.5
        assert torch.isclose(rewards[0], torch.tensor(5.5))
        assert torch.isclose(rewards[1], torch.tensor(5.5))

    def test_death_penalty(self):
        """Dead agents receive -100.0 penalty, overriding milestone bonuses."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()
        
        # One alive, one dead - both at milestone step
        env.step_counts = torch.tensor([10, 10])
        env.dones = torch.tensor([False, True])
        
        rewards = env._calculate_shaped_rewards()
        
        # Alive agent gets milestone, dead agent gets penalty
        assert torch.isclose(rewards[0], torch.tensor(0.5))
        assert torch.isclose(rewards[1], torch.tensor(-100.0))

    def test_dead_agents_no_milestone_bonus(self):
        """Dead agents only receive death penalty, no milestone bonuses."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()
        
        # Dead agent at century milestone
        env.step_counts = torch.tensor([100])
        env.dones = torch.tensor([True])
        
        rewards = env._calculate_shaped_rewards()
        
        # Should only get death penalty, no milestone bonuses
        assert torch.isclose(rewards[0], torch.tensor(-100.0))

    def test_zero_steps_both_milestones(self):
        """At step 0, gets both decade and century bonuses (0 % 10 == 0, 0 % 100 == 0)."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()
        
        env.step_counts = torch.tensor([0, 0])
        env.dones = torch.tensor([False, False])
        
        rewards = env._calculate_shaped_rewards()
        
        # Step 0 is both decade and century milestone: +0.5 + 5.0 = 5.5
        assert torch.isclose(rewards[0], torch.tensor(5.5))
        assert torch.isclose(rewards[1], torch.tensor(5.5))

    def test_combined_milestones(self):
        """Step 100 triggers both decade and century milestones."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()
        
        env.step_counts = torch.tensor([100])
        env.dones = torch.tensor([False])
        
        rewards = env._calculate_shaped_rewards()
        
        # 100 % 10 == 0 AND 100 % 100 == 0: +0.5 + 5.0 = 5.5
        assert torch.isclose(rewards[0], torch.tensor(5.5))


class TestRewardStrategyDirect:
    """Test RewardStrategy class directly (post-extraction interface)."""

    def test_decade_milestone_direct(self):
        """Test decade milestone bonus through RewardStrategy."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        
        step_counts = torch.tensor([10, 20, 30])
        dones = torch.tensor([False, False, False])
        
        rewards = strategy.calculate_rewards(step_counts, dones)
        
        # All at decade milestones
        assert torch.all(torch.isclose(rewards, torch.tensor(0.5)))

    def test_century_milestone_direct(self):
        """Test century milestone bonus through RewardStrategy."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        
        step_counts = torch.tensor([100, 200])
        dones = torch.tensor([False, False])
        
        rewards = strategy.calculate_rewards(step_counts, dones)
        
        # Both get decade + century: 5.5
        assert torch.all(torch.isclose(rewards, torch.tensor(5.5)))

    def test_death_penalty_direct(self):
        """Test death penalty through RewardStrategy."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        
        step_counts = torch.tensor([10, 20])
        dones = torch.tensor([True, True])
        
        rewards = strategy.calculate_rewards(step_counts, dones)
        
        # Both dead: -100.0
        assert torch.all(torch.isclose(rewards, torch.tensor(-100.0)))

    def test_mixed_alive_dead_direct(self):
        """Test mixed alive/dead agents through RewardStrategy."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        
        step_counts = torch.tensor([10, 10, 15])
        dones = torch.tensor([False, True, False])
        
        rewards = strategy.calculate_rewards(step_counts, dones)
        
        # Agent 0: alive at milestone = 0.5
        # Agent 1: dead at milestone = -100.0
        # Agent 2: alive at non-milestone = 0.0
        assert torch.isclose(rewards[0], torch.tensor(0.5))
        assert torch.isclose(rewards[1], torch.tensor(-100.0))
        assert torch.isclose(rewards[2], torch.tensor(0.0))

    def test_step_zero_edge_case_direct(self):
        """Test step 0 edge case through RewardStrategy."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        
        step_counts = torch.tensor([0, 0])
        dones = torch.tensor([False, False])
        
        rewards = strategy.calculate_rewards(step_counts, dones)
        
        # Step 0 triggers both milestones: 5.5
        assert torch.all(torch.isclose(rewards, torch.tensor(5.5)))

    def test_non_milestone_steps_direct(self):
        """Test non-milestone steps through RewardStrategy."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        
        step_counts = torch.tensor([5, 15, 25, 99])
        dones = torch.tensor([False, False, False, False])
        
        rewards = strategy.calculate_rewards(step_counts, dones)
        
        # Steps 5, 15, 25 at decade milestones (10, 20, 30 nearby but not exact)
        # Actually: 15, 25 ARE decade milestones (15 % 10 == 5, 25 % 10 == 5)
        # Let me reconsider: 15 % 10 == 5 (not 0), so NOT a milestone
        # 5, 15, 25, 99 are all NOT milestones
        assert torch.isclose(rewards[0], torch.tensor(0.0))  # 5 % 10 = 5
        assert torch.isclose(rewards[1], torch.tensor(0.0))  # 15 % 10 = 5
        assert torch.isclose(rewards[2], torch.tensor(0.0))  # 25 % 10 = 5
        assert torch.isclose(rewards[3], torch.tensor(0.0))  # 99 % 10 = 9
