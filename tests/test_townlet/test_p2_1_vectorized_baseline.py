"""
Test P2.1: Per-agent vectorized reward baseline support.

Focus:
1. RewardStrategy works with explicit per-agent baseline tensors.
2. VectorizedHamletEnv updates the runtime registry when curriculum multipliers change.
"""

import torch

from townlet.environment.reward_strategy import RewardStrategy
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.population.runtime_registry import AgentRuntimeRegistry


def _attach_registry(env: VectorizedHamletEnv) -> AgentRuntimeRegistry:
    registry = AgentRuntimeRegistry(
        agent_ids=[f"agent-{idx}" for idx in range(env.num_agents)],
        device=env.device,
    )
    env.attach_runtime_registry(registry)
    return registry


class TestVectorizedRewardBaseline:
    """Test per-agent reward baselines (P2.1)."""

    def test_reward_strategy_vectorized_baseline(self):
        """Per-agent baselines work with interoception rewards (dead agents get 0.0)."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        baselines = torch.tensor([100.0, 150.0, 200.0], dtype=torch.float32, device=device)
        step_counts = torch.tensor([120, 120, 120], device=device)
        dones = torch.tensor([True, True, True], device=device)

        meters = torch.zeros(3, 8, device=device)
        meters[:, 0] = 0.0  # depleted energy
        meters[:, 6] = 0.0  # depleted health

        rewards = strategy.calculate_rewards(step_counts, dones, baselines, meters)
        # All dead: 0.0 regardless of baseline
        expected = torch.tensor([0.0, 0.0, 0.0], device=device)
        assert torch.equal(rewards, expected)

    def test_reward_strategy_validates_shape(self):
        """Meters tensor must match number of agents."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([100, 100, 100], device=device)
        dones = torch.tensor([True, True, True], device=device)
        baseline = torch.tensor([100.0, 100.0, 100.0], device=device)
        wrong_meters = torch.zeros(2, 8, device=device)  # Wrong: 2 agents instead of 3

        try:
            strategy.calculate_rewards(step_counts, dones, baseline, wrong_meters)
            assert False, "Expected ValueError for mismatched meters shape"
        except ValueError as exc:
            assert "expected meters shaped" in str(exc).lower()

    def test_env_updates_registry_with_per_agent_baselines(self):
        """Environment pushes per-agent baselines into runtime registry."""
        device = torch.device("cpu")
        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            device=device,
            partial_observability=False,
        )
        registry = _attach_registry(env)

        multipliers = torch.tensor([0.2, 0.5, 1.0], device=device)
        baselines = env.update_baseline_for_curriculum(multipliers)

        registry_baselines = registry.get_baseline_tensor()
        assert torch.equal(registry_baselines, baselines)
        assert baselines[0] > baselines[1] > baselines[2]

    def test_env_scalar_multiplier_broadcasts(self):
        """Scalar curriculum multiplier yields identical baselines."""
        device = torch.device("cpu")
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            device=device,
            partial_observability=False,
        )
        registry = _attach_registry(env)

        baselines = env.update_baseline_for_curriculum(0.5)

        registry_baselines = registry.get_baseline_tensor()
        assert torch.equal(registry_baselines, baselines)
        assert torch.allclose(baselines, torch.full((4,), baselines[0], device=device))

    def test_env_rewards_use_registry_baseline(self):
        """With interoception rewards, alive agents get health × energy."""
        device = torch.device("cpu")
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            device=device,
            partial_observability=False,
        )
        _attach_registry(env)

        multipliers = torch.tensor([0.2, 0.4, 0.6, 1.0], device=device)
        env.update_baseline_for_curriculum(multipliers)

        # Agents with different health/energy levels
        env.step_counts = torch.full((4,), 150, device=device)
        env.dones = torch.zeros(4, dtype=torch.bool, device=device)  # All alive
        env.meters[0, 0] = 20.0  # energy = 20%
        env.meters[0, 6] = 100.0  # health = 100%
        env.meters[1, 0] = 50.0  # energy = 50%
        env.meters[1, 6] = 100.0  # health = 100%
        env.meters[2, 0] = 75.0  # energy = 75%
        env.meters[2, 6] = 100.0  # health = 100%
        env.meters[3, 0] = 100.0  # energy = 100%
        env.meters[3, 6] = 100.0  # health = 100%

        rewards = env._calculate_shaped_rewards()

        # Rewards are health × energy: 0.2, 0.5, 0.75, 1.0
        assert rewards[0] < rewards[1] < rewards[2] < rewards[3]
