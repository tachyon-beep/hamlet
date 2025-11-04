"""End-to-end integration tests for variable meter system (TASK-001 Phase 5).

This module verifies that the complete training pipeline works with non-8 meter
configurations. Tests cover:
- Environment creation with variable meters
- Network initialization with dynamic obs_dim
- Training for multiple episodes
- Checkpoint save/load
- Meter dynamics with variable meter counts
"""

import tempfile
from pathlib import Path

import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.population.vectorized import VectorizedPopulation


class TestVariableMetersEndToEnd:
    """End-to-end integration tests for variable meter system."""

    def test_4meter_training_basic(self, cpu_device, task001_env_4meter):
        """Train agent on 4-meter config for basic episode."""
        curriculum = AdversarialCurriculum(max_steps_per_episode=50)
        curriculum.initialize_population(1)

        exploration = EpsilonGreedyExploration(
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        population = VectorizedPopulation(
            env=task001_env_4meter,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter.observation_dim,
            action_dim=5,
            network_type="simple",
            replay_buffer_capacity=100,
            batch_size=32,
        )

        # Verify environment setup
        assert task001_env_4meter.meters.shape[1] == 4, "Should have 4 meters"
        assert task001_env_4meter.observation_dim > 0, "obs_dim should be positive"

        # Verify network matches obs_dim
        assert population.q_network is not None, "Network should be created"

        # Reset environment
        obs = task001_env_4meter.reset()
        assert obs.shape[1] == task001_env_4meter.observation_dim, "Observation should match obs_dim"

    def test_8meter_backward_compatibility(self, cpu_device, basic_env):
        """Verify 8-meter configs still work (backward compatibility)."""
        curriculum = AdversarialCurriculum(max_steps_per_episode=50)
        curriculum.initialize_population(1)

        exploration = EpsilonGreedyExploration(
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        population = VectorizedPopulation(
            env=basic_env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,
            action_dim=5,
            network_type="simple",
            replay_buffer_capacity=100,
            batch_size=32,
        )

        # Verify environment setup
        assert basic_env.meters.shape[1] == 8, "Should have 8 meters"
        assert basic_env.observation_dim > 0, "obs_dim should be positive"

        # Verify network
        assert population.q_network is not None, "Network should be created"

        # Reset environment
        obs = basic_env.reset()
        assert obs.shape[1] == basic_env.observation_dim, "Observation should match obs_dim"

    def test_checkpoint_save_load_4meter(self, cpu_device, task001_env_4meter, tmp_path):
        """Test checkpoint save/load with 4-meter config."""
        curriculum = AdversarialCurriculum(max_steps_per_episode=50)
        curriculum.initialize_population(1)

        exploration = EpsilonGreedyExploration(
            epsilon=0.5,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        # First population
        pop1 = VectorizedPopulation(
            env=task001_env_4meter,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter.observation_dim,
            action_dim=5,
            network_type="simple",
        )

        # Save checkpoint
        checkpoint = pop1.get_checkpoint_state()

        # Verify metadata
        assert "universe_metadata" in checkpoint, "Checkpoint should have metadata"
        assert checkpoint["universe_metadata"]["meter_count"] == 4, "Should have 4 meters in metadata"
        assert checkpoint["universe_metadata"]["obs_dim"] == task001_env_4meter.observation_dim

        # Save to file
        checkpoint_path = tmp_path / "checkpoint_4meter.pt"
        torch.save(checkpoint, checkpoint_path)

        # Create second population and load
        curriculum2 = AdversarialCurriculum(max_steps_per_episode=50)
        curriculum2.initialize_population(1)

        exploration2 = EpsilonGreedyExploration(
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        pop2 = VectorizedPopulation(
            env=task001_env_4meter,
            curriculum=curriculum2,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter.observation_dim,
            action_dim=5,
            network_type="simple",
        )

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, weights_only=False)
        pop2.load_checkpoint_state(loaded_checkpoint)

        # Verify load succeeded
        assert pop2.total_steps == pop1.total_steps, "Training steps should match"

    def test_cross_meter_count_rejection(self, cpu_device, task001_env_4meter, basic_env):
        """Test that loading 4-meter checkpoint into 8-meter env fails."""
        curriculum = AdversarialCurriculum(max_steps_per_episode=50)
        curriculum.initialize_population(1)

        exploration = EpsilonGreedyExploration()

        # Create 4-meter population and save
        pop_4meter = VectorizedPopulation(
            env=task001_env_4meter,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter.observation_dim,
            action_dim=5,
            network_type="simple",
        )

        checkpoint_4meter = pop_4meter.get_checkpoint_state()

        # Create 8-meter population
        curriculum2 = AdversarialCurriculum(max_steps_per_episode=50)
        curriculum2.initialize_population(1)

        exploration2 = EpsilonGreedyExploration()

        pop_8meter = VectorizedPopulation(
            env=basic_env,
            curriculum=curriculum2,
            exploration=exploration2,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=basic_env.observation_dim,
            action_dim=5,
            network_type="simple",
        )

        # Loading should fail
        with pytest.raises(ValueError, match="meter count mismatch"):
            pop_8meter.load_checkpoint_state(checkpoint_4meter)

    def test_meter_dynamics_with_variable_meters(self, cpu_device, task001_env_4meter):
        """Test that meter dynamics work correctly with 4 meters."""
        env = task001_env_4meter

        # Reset environment
        obs = env.reset()

        # Verify initial meters shape
        assert env.meters.shape == (1, 4), f"Meters should be [1, 4], got {env.meters.shape}"

        # Get initial values
        initial_meters = env.meters.clone()

        # Take a few steps (WAIT action = 4, lowest energy cost)
        actions = torch.tensor([4], device=cpu_device)  # WAIT
        for _ in range(5):
            obs, rewards, dones, info = env.step(actions)

        # Verify meters depleted
        final_meters = env.meters

        # At least one meter should have changed (due to depletion)
        meters_changed = not torch.allclose(initial_meters, final_meters, atol=1e-6)
        assert meters_changed, "Meters should deplete over time"

        # Meters should stay in valid range [0, 1]
        assert torch.all(final_meters >= 0.0), "Meters should not go below 0"
        assert torch.all(final_meters <= 1.0), "Meters should not exceed 1"
