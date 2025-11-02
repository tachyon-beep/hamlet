"""
Tests for baseline-relative reward calculation system.

Reward Formula: reward = steps_survived - R
Where R = baseline survival steps (expected survival of random-walking agent)
"""

import torch

from townlet.environment.reward_strategy import RewardStrategy
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.population.runtime_registry import AgentRuntimeRegistry


def _attach_registry_with_baseline(env: VectorizedHamletEnv, baseline: float | list[float]) -> AgentRuntimeRegistry:
    """Helper to attach runtime registry to environment with preset baseline."""
    registry = AgentRuntimeRegistry(
        agent_ids=[f"agent-{idx}" for idx in range(env.num_agents)],
        device=env.device,
    )
    env.attach_runtime_registry(registry)

    if isinstance(baseline, (float, int)):
        values = torch.full((env.num_agents,), float(baseline), dtype=torch.float32, device=env.device)
    else:
        values = torch.tensor(baseline, dtype=torch.float32, device=env.device)

    registry.set_baselines(values)
    return registry


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

        _attach_registry_with_baseline(env, 100.0)

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

        _attach_registry_with_baseline(env, 100.0)

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

        _attach_registry_with_baseline(env, 100.0)

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

        _attach_registry_with_baseline(env, 100.0)

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

        _attach_registry_with_baseline(env, 100.0)

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

        _attach_registry_with_baseline(env, 50.0)

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
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([10, 20, 30], device=device)
        dones = torch.tensor([False, False, False], device=device)
        baseline = torch.full((3,), 100.0, dtype=torch.float32, device=device)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline)
        assert torch.allclose(rewards, torch.zeros_like(rewards))

    def test_century_milestone_direct(self):
        """Dead agents get baseline-relative reward."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=2)

        step_counts = torch.tensor([100, 200], device=device)
        dones = torch.tensor([True, True], device=device)
        baseline = torch.full((2,), 100.0, dtype=torch.float32, device=device)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline)
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(100.0))

    def test_death_penalty_direct(self):
        """Dead agents get baseline-relative reward (not fixed penalty)."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=2)

        step_counts = torch.tensor([50, 80], device=device)
        dones = torch.tensor([True, True], device=device)
        baseline = torch.full((2,), 100.0, dtype=torch.float32, device=device)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline)
        assert torch.isclose(rewards[0], torch.tensor(-50.0))
        assert torch.isclose(rewards[1], torch.tensor(-20.0))

    def test_mixed_alive_dead_direct(self):
        """Mixed alive/dead: alive get 0, dead get baseline-relative."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([50, 150, 100], device=device)
        dones = torch.tensor([False, True, True], device=device)
        baseline = torch.full((3,), 100.0, dtype=torch.float32, device=device)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline)
        assert torch.isclose(rewards[0], torch.tensor(0.0))
        assert torch.isclose(rewards[1], torch.tensor(50.0))
        assert torch.isclose(rewards[2], torch.tensor(0.0))

    def test_step_zero_edge_case_direct(self):
        """Immediate death (step 0) gets large negative reward."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=2)

        step_counts = torch.tensor([0, 0], device=device)
        dones = torch.tensor([True, True], device=device)
        baseline = torch.full((2,), 100.0, dtype=torch.float32, device=device)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline)
        assert torch.allclose(rewards, torch.tensor([-100.0, -100.0]))

    def test_non_milestone_steps_direct(self):
        """Test non-milestone steps through RewardStrategy."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=4)

        step_counts = torch.tensor([5, 15, 25, 99], device=device)
        dones = torch.tensor([False, False, False, False], device=device)
        baseline = torch.full((4,), 100.0, dtype=torch.float32, device=device)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline)
        assert torch.allclose(rewards, torch.zeros_like(rewards))

    def test_vectorized_baseline_support(self):
        """Different per-agent baselines produce different rewards."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([100, 100, 100], device=device)
        dones = torch.tensor([True, True, True], device=device)
        baseline = torch.tensor([80.0, 100.0, 120.0], dtype=torch.float32, device=device)

        rewards = strategy.calculate_rewards(step_counts, dones, baseline)
        assert torch.isclose(rewards[0], torch.tensor(20.0))
        assert torch.isclose(rewards[1], torch.tensor(0.0))
        assert torch.isclose(rewards[2], torch.tensor(-20.0))
