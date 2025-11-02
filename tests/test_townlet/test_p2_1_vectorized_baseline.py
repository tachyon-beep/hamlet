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
        """Per-agent baselines produce per-agent rewards."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        baselines = torch.tensor([100.0, 150.0, 200.0], dtype=torch.float32, device=device)
        step_counts = torch.tensor([120, 120, 120], device=device)
        dones = torch.tensor([True, True, True], device=device)

        rewards = strategy.calculate_rewards(step_counts, dones, baselines)
        expected = torch.tensor([20.0, -30.0, -80.0], device=device)
        assert torch.equal(rewards, expected)

    def test_reward_strategy_validates_shape(self):
        """Baseline tensor must match number of agents."""
        device = torch.device("cpu")
        strategy = RewardStrategy(device=device, num_agents=3)

        step_counts = torch.tensor([100, 100, 100], device=device)
        dones = torch.tensor([True, True, True], device=device)
        wrong_baseline = torch.tensor([100.0, 120.0], device=device)

        try:
            strategy.calculate_rewards(step_counts, dones, wrong_baseline)
            assert False, "Expected ValueError for mismatched baseline shape"
        except ValueError as exc:
            assert "baseline tensor shape" in str(exc)

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
        """Agents with different baselines receive different rewards."""
        device = torch.device("cpu")
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            device=device,
            partial_observability=False,
        )
        registry = _attach_registry(env)

        multipliers = torch.tensor([0.2, 0.4, 0.6, 1.0], device=device)
        env.update_baseline_for_curriculum(multipliers)

        env.step_counts = torch.full((4,), 150, device=device)
        env.dones = torch.ones(4, dtype=torch.bool, device=device)
        rewards = env._calculate_shaped_rewards()

        assert rewards[0] < rewards[1] < rewards[2] < rewards[3]
