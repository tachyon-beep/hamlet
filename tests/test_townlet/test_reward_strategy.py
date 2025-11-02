"""
Tests for baseline-relative reward calculation system.

Reward Formula: reward = steps_survived - R
Where R = baseline survival steps (expected survival of random-walking agent)
"""

import torch

from townlet.environment.reward_strategy import RewardStrategy
from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestRewardCalculationViaEnv:
    """Test reward calculation through environment interface."""

    def test_milestone_every_10_steps(self):
        """Alive agents get no reward (only on death)."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Both alive at steps 10 and 20
        env.step_counts = torch.tensor([10, 20])
        env.dones = torch.tensor([False, False])

        rewards = env._calculate_shaped_rewards()

        # No reward until death
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(0.0))

    def test_milestone_every_100_steps(self):
        """Alive agents get no reward, even at high step counts."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Both alive at steps 100 and 200
        env.step_counts = torch.tensor([100, 200])
        env.dones = torch.tensor([False, False])

        rewards = env._calculate_shaped_rewards()

        # No reward until death
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(0.0))

    def test_death_penalty(self):
        """Dead agents receive baseline-relative reward."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set baseline to 100 steps
        env.reward_strategy.set_baseline_survival_steps(100.0)

        # One alive, one dead at step 50
        env.step_counts = torch.tensor([50, 50])
        env.dones = torch.tensor([False, True])

        rewards = env._calculate_shaped_rewards()

        # Alive agent: no reward yet
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        # Dead agent: 50 - 100 = -50
        assert torch.isclose(rewards[1], torch.tensor(-50.0))

    def test_dead_agents_no_milestone_bonus(self):
        """Dead agents get baseline-relative reward, not milestone bonuses."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set baseline to 100 steps
        env.reward_strategy.set_baseline_survival_steps(100.0)

        # Dead agent at step 150
        env.step_counts = torch.tensor([150])
        env.dones = torch.tensor([True])

        rewards = env._calculate_shaped_rewards()

        # Reward = 150 - 100 = +50
        assert torch.isclose(rewards[0], torch.tensor(50.0))

    def test_zero_steps_both_milestones(self):
        """Agent dying at step 0 gets baseline-relative reward."""
        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set baseline to 100 steps
        env.reward_strategy.set_baseline_survival_steps(100.0)

        # Both dead at step 0 (immediate death)
        env.step_counts = torch.tensor([0, 0])
        env.dones = torch.tensor([True, True])

        rewards = env._calculate_shaped_rewards()

        # Reward = 0 - 100 = -100
        assert torch.isclose(rewards[0], torch.tensor(-100.0))
        assert torch.isclose(rewards[1], torch.tensor(-100.0))

    def test_combined_milestones(self):
        """Agent surviving longer than baseline gets positive reward on death."""
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            device=torch.device("cpu"),
            enable_temporal_mechanics=False,
        )
        env.reset()

        # Set baseline to 50 steps
        env.reward_strategy.set_baseline_survival_steps(50.0)

        # Agent dies at step 100
        env.step_counts = torch.tensor([100])
        env.dones = torch.tensor([True])

        rewards = env._calculate_shaped_rewards()

        # Reward = 100 - 50 = +50
        assert torch.isclose(rewards[0], torch.tensor(50.0))


class TestRewardStrategyDirect:
    """Test RewardStrategy class directly."""

    def test_decade_milestone_direct(self):
        """Alive agents get zero reward."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        strategy.set_baseline_survival_steps(100.0)

        step_counts = torch.tensor([10, 20, 30])
        dones = torch.tensor([False, False, False])

        rewards = strategy.calculate_rewards(step_counts, dones)

        # All alive: no reward
        assert torch.all(torch.isclose(rewards, torch.tensor(0.0)))

    def test_century_milestone_direct(self):
        """Dead agents get baseline-relative reward."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        strategy.set_baseline_survival_steps(100.0)

        step_counts = torch.tensor([100, 200])
        dones = torch.tensor([True, True])

        rewards = strategy.calculate_rewards(step_counts, dones)

        # 100 - 100 = 0, 200 - 100 = +100
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(100.0))

    def test_death_penalty_direct(self):
        """Dead agents get baseline-relative reward (not fixed penalty)."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        strategy.set_baseline_survival_steps(100.0)

        step_counts = torch.tensor([50, 80])
        dones = torch.tensor([True, True])

        rewards = strategy.calculate_rewards(step_counts, dones)

        # 50 - 100 = -50, 80 - 100 = -20
        assert torch.isclose(rewards[0], torch.tensor(-50.0))
        assert torch.isclose(rewards[1], torch.tensor(-20.0))

    def test_mixed_alive_dead_direct(self):
        """Mixed alive/dead: alive get 0, dead get baseline-relative."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        strategy.set_baseline_survival_steps(100.0)

        step_counts = torch.tensor([50, 150, 100])
        dones = torch.tensor([False, True, True])

        rewards = strategy.calculate_rewards(step_counts, dones)

        # Agent 0: alive = 0
        # Agent 1: dead, 150 - 100 = +50
        # Agent 2: dead, 100 - 100 = 0
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(50.0))
        assert torch.isclose(rewards[2], torch.tensor(0.0))

    def test_step_zero_edge_case_direct(self):
        """Immediate death (step 0) gets large negative reward."""
        strategy = RewardStrategy(device=torch.device("cpu"))
        strategy.set_baseline_survival_steps(100.0)

        step_counts = torch.tensor([0, 0])
        dones = torch.tensor([True, True])

        rewards = strategy.calculate_rewards(step_counts, dones)

        # 0 - 100 = -100
        assert torch.all(torch.isclose(rewards, torch.tensor(-100.0)))

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
