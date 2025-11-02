"""
Test P2.1: Per-agent vectorized reward baseline support.

Verifies that the reward system can handle:
1. Per-agent baselines (torch.Tensor[num_agents])
2. Scalar baselines (float, backwards compatibility)
3. Per-agent curriculum multipliers affecting baselines
"""

import torch

from townlet.environment.reward_strategy import RewardStrategy
from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestVectorizedRewardBaseline:
    """Test per-agent reward baselines (P2.1)."""

    def test_reward_strategy_accepts_vectorized_baseline(self):
        """Test that RewardStrategy accepts per-agent baseline tensor."""
        device = torch.device("cpu")
        num_agents = 3

        strategy = RewardStrategy(device=device, num_agents=num_agents)

        # Per-agent baselines
        baselines = torch.tensor([100.0, 150.0, 200.0], device=device)
        strategy.set_baseline_survival_steps(baselines)

        assert torch.equal(strategy.baseline_survival_steps, baselines)

    def test_reward_strategy_accepts_scalar_baseline(self):
        """Test backwards compatibility: scalar baseline broadcasts to all agents."""
        device = torch.device("cpu")
        num_agents = 3

        strategy = RewardStrategy(device=device, num_agents=num_agents)

        # Scalar baseline (backwards compatible)
        strategy.set_baseline_survival_steps(120.0)

        expected = torch.tensor([120.0, 120.0, 120.0], device=device)
        assert torch.equal(strategy.baseline_survival_steps, expected)

    def test_vectorized_baseline_affects_rewards(self):
        """Test that per-agent baselines produce different rewards."""
        device = torch.device("cpu")
        num_agents = 3

        strategy = RewardStrategy(device=device, num_agents=num_agents)

        # Different baselines for each agent
        baselines = torch.tensor([100.0, 150.0, 200.0], device=device)
        strategy.set_baseline_survival_steps(baselines)

        # All agents die at step 120
        step_counts = torch.tensor([120, 120, 120], device=device)
        dones = torch.tensor([True, True, True], device=device)

        rewards = strategy.calculate_rewards(step_counts, dones)

        # Agent 0: 120 - 100 = +20
        # Agent 1: 120 - 150 = -30
        # Agent 2: 120 - 200 = -80
        expected = torch.tensor([20.0, -30.0, -80.0], device=device)
        assert torch.equal(rewards, expected)

    def test_env_update_baseline_accepts_tensor(self):
        """Test that environment accepts per-agent multipliers."""
        device = torch.device("cpu")
        num_agents = 3

        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            device=device,
            partial_observability=False,
        )

        # Per-agent multipliers (different curriculum stages)
        multipliers = torch.tensor([0.2, 0.5, 1.0], device=device)
        env.update_baseline_for_curriculum(multipliers)

        # Verify baselines were set (should be different for each agent)
        baselines = env.reward_strategy.baseline_survival_steps
        assert baselines.shape == (num_agents,)

        # Higher multiplier = faster depletion = lower baseline
        # So baselines[0] > baselines[1] > baselines[2]
        assert baselines[0] > baselines[1]
        assert baselines[1] > baselines[2]

    def test_env_update_baseline_accepts_scalar(self):
        """Test backwards compatibility: environment accepts scalar multiplier."""
        device = torch.device("cpu")
        num_agents = 3

        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            device=device,
            partial_observability=False,
        )

        # Scalar multiplier (backwards compatible)
        env.update_baseline_for_curriculum(0.5)

        # Verify baselines were set (should be same for all agents)
        baselines = env.reward_strategy.baseline_survival_steps
        assert baselines.shape == (num_agents,)
        assert torch.all(baselines == baselines[0])

    def test_different_agents_different_baselines_different_rewards(self):
        """Integration test: different agents with different baselines get different rewards."""
        device = torch.device("cpu")
        num_agents = 4

        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            device=device,
            partial_observability=False,
        )

        # Set per-agent baselines via multipliers
        # Lower multiplier = easier = higher baseline
        multipliers = torch.tensor([0.2, 0.4, 0.6, 1.0], device=device)
        env.update_baseline_for_curriculum(multipliers)

        # All agents survive 150 steps
        step_counts = torch.full((num_agents,), 150, device=device)
        dones = torch.ones(num_agents, dtype=torch.bool, device=device)

        rewards = env.reward_strategy.calculate_rewards(step_counts, dones)

        # All agents survived same duration but baselines differ
        # Agent 0 (multiplier 0.2, highest baseline) should get lowest reward
        # Agent 3 (multiplier 1.0, lowest baseline) should get highest reward
        assert rewards[0] < rewards[1] < rewards[2] < rewards[3]

    def test_baseline_shape_validation(self):
        """Test that setting wrong-shaped baseline raises error."""
        device = torch.device("cpu")
        num_agents = 3

        strategy = RewardStrategy(device=device, num_agents=num_agents)

        # Wrong shape: 2 elements instead of 3
        wrong_shape = torch.tensor([100.0, 150.0], device=device)

        try:
            strategy.set_baseline_survival_steps(wrong_shape)
            assert False, "Should have raised assertion error"
        except AssertionError as e:
            assert "baseline_steps must be [num_agents=3]" in str(e)

    def test_per_agent_baseline_survives_reset(self):
        """Test that per-agent baselines persist across environment resets."""
        device = torch.device("cpu")
        num_agents = 2

        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            device=device,
            partial_observability=False,
        )

        # Set per-agent baselines
        multipliers = torch.tensor([0.5, 1.0], device=device)
        env.update_baseline_for_curriculum(multipliers)

        initial_baselines = env.reward_strategy.baseline_survival_steps.clone()

        # Reset environment
        env.reset()

        # Baselines should not change
        assert torch.equal(env.reward_strategy.baseline_survival_steps, initial_baselines)
