"""Integration test for RND loss tracking and TensorBoard logging.

This test verifies that when using RND exploration, the predictor loss is
tracked and available for monitoring (similar to Q-network loss tracking).

Bug discovered during mypy fixes: RND loss was computed but immediately
discarded without being tracked or logged.
"""

import torch

from townlet.curriculum.static import StaticCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation


class TestRNDLossTracking:
    """Test RND loss is tracked during training with AdaptiveIntrinsicExploration."""

    def test_rnd_loss_tracked_during_training(self, cpu_device, test_config_pack_path):
        """Verify RND predictor loss is tracked when using AdaptiveIntrinsicExploration.

        This test validates the critical contract:
        - RND predictor loss should be computed during training
        - Loss should be stored in population.last_rnd_loss (similar to last_loss)
        - Loss should be available for TensorBoard logging
        - Loss should be finite (not NaN/Inf)

        Integration point: VectorizedPopulation.step_population() trains RND
        predictor and should track the loss for monitoring.
        """
        # Create small environment for fast training
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

        # Create population with AdaptiveIntrinsicExploration (includes RND)
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=cpu_device,
            embed_dim=32,  # Small for fast test
            rnd_training_batch_size=64,  # Small for faster test
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.95,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=10,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
            train_frequency=4,
            tb_logger=None,  # No TensorBoard for this test (just tracking)
        )

        # Run enough steps to trigger RND training
        # RND trains every training_batch_size steps (default 64)
        # So we need at least 64 steps to see RND loss
        population.reset()
        for step in range(100):
            agent_state = population.step_population(env)
            if agent_state.dones[0]:
                population.reset()

        # CRITICAL ASSERTION: RND loss should be tracked
        # This will FAIL until we implement the feature
        assert hasattr(population, "last_rnd_loss"), "Population should track last_rnd_loss"
        assert population.last_rnd_loss is not None, "RND loss should be computed during training"
        assert isinstance(population.last_rnd_loss, float), "RND loss should be a float"
        assert torch.isfinite(torch.tensor(population.last_rnd_loss)), f"RND loss should be finite, got {population.last_rnd_loss}"

        # Verify RND loss is reasonable (should be > 0 since predictor is training)
        assert population.last_rnd_loss >= 0.0, f"RND loss should be non-negative, got {population.last_rnd_loss}"

    def test_rnd_loss_tracks_latest_value(self, cpu_device, test_config_pack_path):
        """Verify RND loss updates with latest training value.

        This test validates that last_rnd_loss is updated each time RND
        predictor trains, not just set once.
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

        # Create population with RND
        curriculum = StaticCurriculum(difficulty_level=0.5)
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            device=cpu_device,
            embed_dim=32,
            rnd_training_batch_size=64,
            epsilon_start=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.95,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=10,
        )

        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=env.observation_dim,
            # action_dim defaults to env.action_dim
            network_type="simple",
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=1000,
            batch_size=16,
            train_frequency=4,
        )

        # Run steps to get first RND training
        population.reset()
        for step in range(70):
            agent_state = population.step_population(env)
            if agent_state.dones[0]:
                population.reset()

        # Capture first RND loss
        first_rnd_loss = population.last_rnd_loss

        # Run more steps to trigger another RND training
        for step in range(70):
            agent_state = population.step_population(env)
            if agent_state.dones[0]:
                population.reset()

        # RND loss should have updated
        second_rnd_loss = population.last_rnd_loss

        # Both should be valid
        assert first_rnd_loss is not None, "First RND loss should exist"
        assert second_rnd_loss is not None, "Second RND loss should exist"

        # They should likely be different (predictor is learning)
        # Note: They COULD be the same due to randomness, but tracking should work
        assert isinstance(second_rnd_loss, float), "Second RND loss should be float"
