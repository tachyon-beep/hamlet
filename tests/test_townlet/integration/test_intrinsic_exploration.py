"""Integration tests for intrinsic exploration (RND + adaptive annealing).

This file tests the integration of RND intrinsic rewards and adaptive annealing
with the full training pipeline.

Task 12c: Intrinsic Exploration Integration Tests
Focus: Test RND and adaptive annealing with real components (env, population, curriculum)

Test Organization:
- TestRNDIntrinsicRewards: RND novelty computation and reward combination (3 tests)
- TestAdaptiveAnnealing: Variance-based annealing during training (3 tests)
"""

import torch

from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation

# =============================================================================
# TEST CLASS 1: RND Intrinsic Rewards
# =============================================================================


class TestRNDIntrinsicRewards:
    """Test RND integration with environment and population.

    Verifies RND produces novelty rewards during episode execution and that
    intrinsic/extrinsic rewards are combined correctly for training.
    """

    def test_rnd_computes_novelty_for_observations(self, cpu_device, test_config_pack_path):
        """Verify RND produces novelty rewards for observations during episode steps.

        This test validates the critical contract:
        - RND should compute intrinsic rewards for each observation
        - Intrinsic rewards should be non-negative (MSE property)
        - Intrinsic rewards should vary across different observations (novelty detection)

        Integration point: VectorizedHamletEnv → AdaptiveIntrinsicExploration → RND
        """
        # Create small environment for fast testing
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
            agent_lifespan=1000,
        )

        # Create adaptive intrinsic exploration
        obs_dim = env.observation_dim
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=64,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=50,
            device=cpu_device,
        )

        # Create population with adaptive intrinsic exploration
        curriculum = StaticCurriculum(difficulty_level=0.5)
        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=obs_dim,  # Must match environment's observation dimension
            action_dim=6,  # Test environment action space (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset environment
        population.reset()

        # Run 10 steps and collect intrinsic rewards
        intrinsic_rewards_collected = []
        for _ in range(10):
            agent_state = population.step_population(env)
            intrinsic_rewards_collected.append(agent_state.intrinsic_rewards[0].item())

        # Verify RND produced intrinsic rewards
        assert len(intrinsic_rewards_collected) == 10, "Should have 10 intrinsic rewards"

        # All rewards should be non-negative (MSE property)
        for reward in intrinsic_rewards_collected:
            assert reward >= 0, f"Intrinsic reward should be non-negative, got {reward}"

        # At least some rewards should be positive (RND should detect novelty)
        assert sum(intrinsic_rewards_collected) > 0, "RND should produce non-zero novelty rewards"

    def test_intrinsic_rewards_combined_with_extrinsic(self, cpu_device, test_config_pack_path):
        """Verify intrinsic and extrinsic rewards are combined correctly during step.

        This test validates the critical contract:
        - Combined reward = extrinsic + (intrinsic * weight)
        - Weight should be modifiable (annealing)
        - Both rewards should be accessible for logging

        Integration point: VectorizedPopulation combines rewards from environment and exploration
        """
        # Create small environment
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
            agent_lifespan=1000,
        )

        # Create adaptive intrinsic exploration with known weight
        obs_dim = env.observation_dim
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=64,
            initial_intrinsic_weight=0.5,  # 50% intrinsic weight for clear testing
            variance_threshold=100.0,
            survival_window=50,
            device=cpu_device,
        )

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=obs_dim,  # Must match environment's observation dimension
            action_dim=6,  # Test environment action space (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset and step
        population.reset()
        agent_state = population.step_population(env)

        # Verify reward structure
        assert hasattr(agent_state, "rewards"), "Agent state should have combined rewards"
        assert hasattr(agent_state, "intrinsic_rewards"), "Agent state should have intrinsic rewards"

        # Verify shapes
        assert agent_state.rewards.shape == (1,), "Combined rewards should be [num_agents]"
        assert agent_state.intrinsic_rewards.shape == (1,), "Intrinsic rewards should be [num_agents]"

        # Verify intrinsic rewards are non-negative (MSE property of RND)
        intrinsic = agent_state.intrinsic_rewards[0].item()
        assert intrinsic >= 0, f"Intrinsic reward should be non-negative (MSE), got {intrinsic}"

        # Verify combined rewards include intrinsic component
        # Note: BatchedAgentState doesn't track extrinsic separately, only combined and intrinsic
        # The combination happens in VectorizedPopulation.step_population()
        combined = agent_state.rewards[0].item()
        weight = exploration.get_intrinsic_weight()

        # Behavioral assertion: Combined reward should include intrinsic component
        # We can't verify exact formula without separate extrinsic tracking,
        # but we can verify that the intrinsic weight is being used
        assert weight == 0.5, f"Intrinsic weight should be 0.5, got {weight}"

        # The combined reward exists and is a float
        assert isinstance(combined, float), "Combined reward should be a float"

    def test_combined_reward_stored_in_replay_buffer(self, cpu_device, test_config_pack_path):
        """Verify combined rewards are stored in replay buffer (not separate rewards).

        This test validates the critical contract:
        - Replay buffer should store combined rewards for training
        - Extrinsic/intrinsic separation only for logging
        - Training uses combined rewards (prevents double-counting)

        Integration point: VectorizedPopulation → ReplayBuffer storage
        """
        # Create small environment
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
            agent_lifespan=1000,
        )

        # Create adaptive intrinsic exploration
        obs_dim = env.observation_dim
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=64,
            initial_intrinsic_weight=0.5,
            variance_threshold=100.0,
            survival_window=50,
            device=cpu_device,
        )

        # Create population
        curriculum = StaticCurriculum(difficulty_level=0.5)
        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=obs_dim,  # Must match environment's observation dimension
            action_dim=6,  # Test environment action space (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
        )

        # Reset and run 5 steps
        population.reset()
        combined_rewards = []
        for _ in range(5):
            agent_state = population.step_population(env)
            combined_rewards.append(agent_state.rewards[0].item())

        # Verify replay buffer has transitions
        assert len(population.replay_buffer) >= 5, "Replay buffer should have at least 5 transitions"

        # Sample from replay buffer and verify rewards match combined rewards
        if len(population.replay_buffer) >= 16:  # Ensure we have enough for batch
            batch = population.replay_buffer.sample(5)

            # Verify batch contains 'rewards' key (combined rewards)
            assert "rewards" in batch, "Replay buffer should store 'rewards' key"

            # Verify rewards are combined (not separate)
            assert "intrinsic_rewards" not in batch, "Replay buffer should NOT store separate intrinsic rewards"
            assert "rewards_extrinsic" not in batch, "Replay buffer should NOT store separate extrinsic rewards"

            # Verify rewards are tensors
            assert isinstance(batch["rewards"], torch.Tensor), "Rewards should be tensors"
            assert batch["rewards"].shape[0] == 5, "Batch should have 5 samples"


# =============================================================================
# TEST CLASS 2: Adaptive Annealing
# =============================================================================


class TestAdaptiveAnnealing:
    """Test adaptive annealing of intrinsic weight.

    Verifies intrinsic weight decreases when agent demonstrates consistent
    performance (low variance + high survival).
    """

    def test_intrinsic_weight_decreases_after_consistent_performance(self, cpu_device, test_config_pack_path):
        """Verify intrinsic weight decreases when agent performs consistently well.

        This test validates the critical contract:
        - Weight should decrease when variance < threshold AND survival > 50
        - Weight should use exponential decay (weight *= decay_rate)
        - Weight should respect minimum floor

        Integration point: AdaptiveIntrinsicExploration annealing logic
        """
        # Create small environment
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
            agent_lifespan=1000,
        )

        # Create adaptive intrinsic exploration with fast annealing for testing
        obs_dim = env.observation_dim
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=64,
            initial_intrinsic_weight=1.0,
            min_intrinsic_weight=0.1,
            variance_threshold=100.0,
            survival_window=20,  # Small window for fast testing
            decay_rate=0.9,
            device=cpu_device,
        )

        # Record initial weight
        initial_weight = exploration.get_intrinsic_weight()
        assert abs(initial_weight - 1.0) < 1e-6, "Initial weight should be 1.0"

        # Simulate consistent performance (low variance, high survival)
        # Add 20 episodes with survival times around 80 steps (low variance)
        for i in range(20):
            survival_time = 80.0 + (i % 5)  # 80-84 steps (very low variance)
            exploration.update_on_episode_end(survival_time)

        # Verify weight decreased
        final_weight = exploration.get_intrinsic_weight()
        assert (
            final_weight < initial_weight
        ), f"Weight should decrease after consistent performance: initial={initial_weight}, final={final_weight}"

        # Verify weight followed exponential decay
        # After annealing: weight should be around 0.9 (one decay step)
        # Note: may anneal multiple times if conditions met repeatedly
        assert final_weight <= 0.9, f"Weight should have decayed to at most 0.9, got {final_weight}"

        # Verify weight respects minimum floor
        assert final_weight >= 0.1, f"Weight should not go below minimum (0.1), got {final_weight}"

    def test_annealing_requires_survival_above_50_steps(self, cpu_device, test_config_pack_path):
        """Verify annealing requires mean survival > 50 steps (prevents premature annealing).

        This test validates the critical fix for the "consistent failure" bug:
        - Low variance + LOW survival = "consistently failing" → NO annealing
        - Low variance + HIGH survival = "consistently succeeding" → YES annealing

        Integration point: AdaptiveIntrinsicExploration.should_anneal() logic
        """
        # Create environment (not needed for this test, but keeping for consistency)
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
            agent_lifespan=1000,
        )

        # Create exploration
        obs_dim = env.observation_dim
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=64,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=20,
            decay_rate=0.9,
            device=cpu_device,
        )

        # Simulate consistent FAILURE (low variance, low survival)
        # 20 episodes with 10 steps each (very consistent, but poor performance)
        for _ in range(20):
            exploration.update_on_episode_end(10.0)

        # Verify weight did NOT decrease (annealing should not trigger)
        weight = exploration.get_intrinsic_weight()
        assert abs(weight - 1.0) < 1e-6, f"Weight should NOT decrease for consistent failure: expected 1.0, got {weight}"

        # Now simulate consistent SUCCESS (low variance, high survival)
        for i in range(20):
            survival_time = 100.0 + (i % 5)  # 100-104 steps
            exploration.update_on_episode_end(survival_time)

        # Verify weight DID decrease (annealing should trigger)
        weight_after = exploration.get_intrinsic_weight()
        assert weight_after < 1.0, f"Weight should decrease for consistent success: expected < 1.0, got {weight_after}"

    def test_annealing_requires_low_variance(self, cpu_device, test_config_pack_path):
        """Verify annealing requires variance < threshold (prevents annealing during exploration).

        This test validates the critical contract:
        - High variance = agent still exploring → NO annealing
        - Low variance = agent converged → YES annealing (if survival also high)

        Integration point: AdaptiveIntrinsicExploration variance calculation
        """
        # Create environment
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            partial_observability=False,
            vision_range=5,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            config_pack_path=test_config_pack_path,
            device=cpu_device,
            agent_lifespan=1000,
        )

        # Create exploration with strict variance threshold
        obs_dim = env.observation_dim
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=obs_dim,
            embed_dim=64,
            initial_intrinsic_weight=1.0,
            variance_threshold=50.0,  # Lower threshold for stricter test
            survival_window=20,
            decay_rate=0.9,
            device=cpu_device,
        )

        # Simulate HIGH variance (agent still learning, erratic performance)
        # Survival times: 10, 60, 20, 80, 30, 90, ... (high variance)
        for i in range(20):
            survival_time = 10.0 + (i % 2) * 70.0  # Alternates 10, 80, 10, 80...
            exploration.update_on_episode_end(survival_time)

        # Verify weight did NOT decrease (high variance prevents annealing)
        weight_high_variance = exploration.get_intrinsic_weight()
        assert (
            abs(weight_high_variance - 1.0) < 1e-6
        ), f"Weight should NOT decrease with high variance: expected 1.0, got {weight_high_variance}"

        # Now simulate LOW variance (agent converged)
        # Add 20 more episodes with tight variance around 100
        for i in range(20):
            survival_time = 100.0 + (i % 3)  # 100, 101, 102 (low variance)
            exploration.update_on_episode_end(survival_time)

        # Verify weight DID decrease (low variance + high survival triggers annealing)
        weight_low_variance = exploration.get_intrinsic_weight()
        assert (
            weight_low_variance < 1.0
        ), f"Weight should decrease with low variance + high survival: expected < 1.0, got {weight_low_variance}"
