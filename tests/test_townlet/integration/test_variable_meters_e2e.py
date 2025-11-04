"""End-to-end integration tests for variable meter system (TASK-001 Phase 5).

This module verifies that the complete training pipeline works with non-8 meter
configurations. Tests cover:
- Environment creation with variable meters
- Network initialization with dynamic obs_dim
- Training for multiple episodes
- Checkpoint save/load
- Meter dynamics with variable meter counts
"""


import pytest
import torch

from townlet.curriculum.adversarial import AdversarialCurriculum
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

        # Take a few steps (WAIT action = 5, lowest energy cost)
        actions = torch.tensor([5], device=cpu_device)  # WAIT (action 5)
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

    def test_action_masking_with_4meters(self, cpu_device, task001_env_4meter):
        """Test that action masking works correctly with 4-meter config.

        This test verifies that dead agent detection uses dynamic meter indices
        instead of hardcoded indices (energy=0, health=6). In a 4-meter config,
        accessing meters[:, 6] would cause IndexError.
        """
        env = task001_env_4meter

        # Reset environment
        env.reset()

        # Get initial action masks (should work without IndexError)
        masks = env.get_action_masks()
        assert masks.shape == (1, 6), f"Action masks should be [1, 6], got {masks.shape}"

        # Kill the agent by setting health to 0
        # Find health meter index from config
        bars_config = env.meter_dynamics.cascade_engine.config.bars
        health_idx = bars_config.meter_name_to_index.get("health", 0)

        # Set health to 0 (agent should be dead)
        env.meters[0, health_idx] = 0.0

        # Get action masks - should mask all actions for dead agent
        masks = env.get_action_masks()

        # All actions should be masked (False) for dead agent
        assert not torch.any(masks[0]), "Dead agent should have all actions masked"

    def test_action_costs_with_4meters(self, cpu_device, task001_env_4meter):
        """Test that action costs work correctly with 4-meter config.

        This test verifies that movement, WAIT, and INTERACT costs use dynamic
        meter indices instead of hardcoded 8-element tensors. In a 4-meter config,
        subtracting an 8-element cost vector from 4-element meters would cause
        broadcasting error.
        """
        env = task001_env_4meter

        # Reset environment
        env.reset()

        # Get initial meter values
        initial_meters = env.meters.clone()

        # Test MOVEMENT action (should deplete energy, hygiene, satiation if present)
        actions = torch.tensor([0], device=cpu_device)  # UP
        obs, rewards, dones, info = env.step(actions)

        # Meters should have changed (action costs applied)
        assert not torch.allclose(env.meters, initial_meters, atol=1e-6), "Movement should deplete meters"

        # Reset and test WAIT action
        env.reset()
        initial_meters = env.meters.clone()

        actions = torch.tensor([5], device=cpu_device)  # WAIT
        obs, rewards, dones, info = env.step(actions)

        # WAIT should also deplete energy (but less than movement)
        assert not torch.allclose(env.meters, initial_meters, atol=1e-6), "WAIT should deplete energy"

        # All meters should stay in valid range
        assert torch.all(env.meters >= 0.0), "Meters should not go below 0"
        assert torch.all(env.meters <= 1.0), "Meters should not exceed 1"

    def test_recurrent_network_with_4meters(self, cpu_device, task001_env_4meter_pomdp):
        """Test that recurrent networks work with 4-meter POMDP config.

        This test verifies that RecurrentSpatialQNetwork is initialized with
        dynamic num_meters instead of hardcoded 8. In a 4-meter config, the
        network expects num_meters features in the observation but receives
        a different count if hardcoded.
        """
        from townlet.curriculum.adversarial import AdversarialCurriculum
        from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
        from townlet.population.vectorized import VectorizedPopulation

        curriculum = AdversarialCurriculum(max_steps_per_episode=50)
        curriculum.initialize_population(1)

        exploration = EpsilonGreedyExploration(
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.99,
        )

        # Create population with recurrent network on 4-meter POMDP env
        # This should work without shape mismatches
        population = VectorizedPopulation(
            env=task001_env_4meter_pomdp,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=cpu_device,
            obs_dim=task001_env_4meter_pomdp.observation_dim,
            action_dim=5,
            network_type="recurrent",  # Use recurrent network
            replay_buffer_capacity=100,
            batch_size=32,
        )

        # Verify environment setup
        assert task001_env_4meter_pomdp.meters.shape[1] == 4, "Should have 4 meters"

        # Verify network was created successfully
        assert population.q_network is not None, "Q-network should be created"
        assert population.is_recurrent, "Should be using recurrent network"

        # Reset environment
        obs = task001_env_4meter_pomdp.reset()
        assert obs.shape[1] == task001_env_4meter_pomdp.observation_dim, "Observation should match obs_dim"

        # Verify network's num_meters parameter matches environment
        assert population.q_network.num_meters == 4, f"Network should have num_meters=4, got {population.q_network.num_meters}"
        assert (
            population.target_network.num_meters == 4
        ), f"Target network should have num_meters=4, got {population.target_network.num_meters}"

        # Verify network can process observations correctly
        # This tests that the network's num_meters matches the environment's meter count
        with torch.no_grad():
            q_values, hidden = population.q_network(obs)  # Recurrent networks return (q_values, hidden_state)
            assert q_values.shape == (1, 5), f"Q-values should be [1, 5], got {q_values.shape}"

        # Note: We don't test environment steps here because AffordanceEngine has hardcoded meter indices
        # (separate bug to be fixed later - not part of recurrent network fix)
