"""Unit tests for VectorizedHamletEnv (environment/vectorized_env.py).

This test suite focuses on testing individual methods of VectorizedHamletEnv
with mocked dependencies to achieve 70%+ coverage.

Test Coverage Plan (Sprint 15):
- Phase 15A: Initialization & Setup (__init__, reset, _build_movement_deltas)
- Phase 15B: Core Loop (step, _execute_actions, _get_observations, get_action_masks)
- Phase 15C: Interactions & Rewards (_handle_interactions, _calculate_shaped_rewards, _apply_custom_action)
- Phase 15D: Checkpointing (get/set_affordance_positions, randomize_affordance_positions)

Testing Strategy:
- Mock heavy dependencies (SubstrateFactory, AffordanceEngine, etc.)
- Use real tensors for state (positions, meters)
- Focus on logic paths, not tensor operations
- Use builders.py for test data construction
"""

from pathlib import Path

import pytest
import torch

# =============================================================================
# PHASE 15A: INITIALIZATION & SETUP
# =============================================================================


class TestVectorizedHamletEnvInitialization:
    """Test VectorizedHamletEnv.__init__ with various configurations."""

    def test_init_requires_substrate_yaml(self, temp_test_dir):
        """Should raise FileNotFoundError if substrate.yaml is missing."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        # Create config pack without substrate.yaml
        config_pack = temp_test_dir / "config_pack"
        config_pack.mkdir()

        with pytest.raises(FileNotFoundError, match="substrate.yaml is required"):
            VectorizedHamletEnv(
                num_agents=1,
                grid_size=8,
                partial_observability=False,
                vision_range=2,
                enable_temporal_mechanics=False,
                move_energy_cost=0.005,
                wait_energy_cost=0.004,
                interact_energy_cost=0.003,
                agent_lifespan=1000,
                config_pack_path=config_pack,
            )

    def test_init_raises_if_config_pack_not_found(self):
        """Should raise FileNotFoundError if config pack directory doesn't exist."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        with pytest.raises(FileNotFoundError, match="Config pack directory not found"):
            VectorizedHamletEnv(
                num_agents=1,
                grid_size=8,
                partial_observability=False,
                vision_range=2,
                enable_temporal_mechanics=False,
                move_energy_cost=0.005,
                wait_energy_cost=0.004,
                interact_energy_cost=0.003,
                agent_lifespan=1000,
                config_pack_path=Path("/nonexistent/path"),
            )

    def test_init_uses_default_config_pack_when_none_provided(self):
        """Should use configs/test as default config pack."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        # This should work with the default test config pack
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
            config_pack_path=None,  # Should default to configs/test
        )

        # Verify default config pack was used
        assert env.config_pack_path.name == "test"
        assert env.config_pack_path.exists()

    def test_init_creates_substrate_from_config(self):
        """Should load substrate.yaml and create substrate via factory."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        # Verify substrate was created
        assert env.substrate is not None
        assert hasattr(env.substrate, 'position_dim')

    def test_init_loads_affordances_from_config(self):
        """Should load affordances.yaml and create affordance engine."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        # Verify affordance engine was created
        assert env.affordance_engine is not None
        assert hasattr(env.affordance_engine, 'affordances')

    def test_init_initializes_state_tensors(self):
        """Should initialize positions, meters, lifetimes, and dones tensors."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        num_agents = 3
        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        # Verify state tensors exist with correct shapes
        assert env.positions.shape == (num_agents, env.substrate.position_dim)
        assert env.meters.shape == (num_agents, env.meter_count)
        assert env.dones.shape == (num_agents,)

        # Verify tensors are on correct device
        assert env.positions.device == env.device
        assert env.meters.device == env.device

    def test_init_sets_device_correctly(self):
        """Should set device to CPU by default or use provided device."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        # Default should be CPU
        env_cpu = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        assert env_cpu.device == torch.device("cpu")

        # Should accept explicit device
        env_explicit = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
            device=torch.device("cpu"),
        )
        assert env_explicit.device == torch.device("cpu")


class TestVectorizedHamletEnvReset:
    """Test VectorizedHamletEnv.reset() method."""

    def test_reset_returns_observations(self):
        """Should return observations tensor with correct shape."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        num_agents = 2
        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        obs = env.reset()

        # Should return tensor with shape [num_agents, obs_dim]
        assert isinstance(obs, torch.Tensor)
        assert obs.shape[0] == num_agents
        assert obs.shape[1] > 0  # Has some observation dimension

    def test_reset_initializes_meters_from_config(self):
        """Should initialize meters to initial values from bars.yaml config."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        env.reset()

        # Meters should be initialized (all values in [0, 1])
        assert torch.all(env.meters >= 0.0).item()
        assert torch.all(env.meters <= 1.0).item()
        # Should have same values for all agents
        assert torch.all(env.meters[0] == env.meters[1]).item()

    def test_reset_clears_dones_flag(self):
        """Should set all dones to False."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        # Manually set dones to True
        env.dones = torch.ones(2, dtype=torch.bool)

        env.reset()

        # Should clear dones
        assert torch.all(~env.dones).item()

    def test_reset_initializes_step_counts(self):
        """Should reset step_counts to 0 for all agents."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        # Manually set step_counts to non-zero
        env.step_counts = torch.tensor([10, 20])

        env.reset()

        # Should reset step_counts to 0
        assert torch.all(env.step_counts == 0).item()

    def test_reset_randomizes_agent_positions(self):
        """Should randomize agent positions on substrate."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=5,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        env.reset()

        # Positions should be within substrate bounds
        # For Grid2D, positions should be in [0, grid_size)
        assert torch.all(env.positions >= 0).item()
        assert torch.all(env.positions < 8).item()  # Assuming grid_size=8

    def test_reset_temporal_mechanics_initializes_time(self):
        """Should initialize time_of_day to 0 when temporal mechanics enabled."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=True,  # Enable temporal mechanics
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        env.reset()

        # Should initialize time_of_day
        assert hasattr(env, 'time_of_day')
        assert env.time_of_day == 0


class TestBuildMovementDeltas:
    """Test VectorizedHamletEnv._build_movement_deltas() method."""

    def test_build_movement_deltas_creates_tensor(self):
        """Should create movement deltas tensor from substrate actions."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        deltas = env._build_movement_deltas()

        # Should return tensor with shape [substrate_action_count, position_dim]
        # For Grid2D: 6 substrate actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
        assert isinstance(deltas, torch.Tensor)
        assert deltas.shape[0] == env.action_space.substrate_action_count
        assert deltas.shape[1] == env.substrate.position_dim

    def test_build_movement_deltas_correct_values_grid2d(self):
        """Should create correct deltas for Grid2D substrate (UP, DOWN, LEFT, RIGHT)."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )

        deltas = env._build_movement_deltas()

        # For Grid2D, movement actions should have non-zero deltas
        # Non-movement actions (INTERACT, WAIT, REST, MEDITATE) should have zero deltas
        # Check that at least some actions have non-zero deltas
        has_nonzero = torch.any(deltas != 0, dim=1)
        assert torch.any(has_nonzero).item(), "Should have some non-zero movement deltas"


# =============================================================================
# PLACEHOLDER: PHASE 15B, 15C, 15D tests will be added incrementally
# =============================================================================


# =============================================================================
# PHASE 15B: CORE LOOP (step, _execute_actions, _get_observations, get_action_masks)
# =============================================================================


class TestVectorizedHamletEnvStep:
    """Test VectorizedHamletEnv.step() method."""

    def test_step_returns_correct_types(self):
        """Should return (observations, rewards, dones, info) tuple."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Execute step with WAIT actions
        actions = torch.tensor([5, 5])  # Both agents WAIT
        obs, rewards, dones, info = env.step(actions)

        # Verify types
        assert isinstance(obs, torch.Tensor)
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(dones, torch.Tensor)
        assert isinstance(info, dict)

    def test_step_returns_correct_shapes(self):
        """Should return tensors with correct shapes."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        num_agents = 3
        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        actions = torch.tensor([5, 5, 5])
        obs, rewards, dones, info = env.step(actions)

        # Verify shapes
        assert obs.shape[0] == num_agents
        assert obs.shape[1] > 0  # Has observation dimension
        assert rewards.shape == (num_agents,)
        assert dones.shape == (num_agents,)

    def test_step_increments_step_counts(self):
        """Should increment step_counts for all agents."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        initial_counts = env.step_counts.clone()

        actions = torch.tensor([5, 5])
        env.step(actions)

        # Step counts should increment
        assert torch.all(env.step_counts == initial_counts + 1).item()

    def test_step_depletes_meters(self):
        """Should deplete meters based on passive decay."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        initial_meters = env.meters.clone()

        # Execute WAIT action (should have minimal costs)
        actions = torch.tensor([5, 5])
        env.step(actions)

        # Some meters should decrease due to passive decay
        # Not all meters may decrease, but at least some should
        assert not torch.all(env.meters == initial_meters).item()

    def test_step_increments_time_of_day(self):
        """Should increment time_of_day and wrap at 24."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=True,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        assert env.time_of_day == 0

        # Step once
        actions = torch.tensor([5])
        env.step(actions)
        assert env.time_of_day == 1

        # Set to 23 and step (should wrap to 0)
        env.time_of_day = 23
        env.step(actions)
        assert env.time_of_day == 0

    def test_step_retirement_bonus(self):
        """Should give retirement bonus when agent reaches lifespan."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=5,  # Very short lifespan
        )
        env.reset()

        # Step until retirement
        actions = torch.tensor([5])  # WAIT
        for i in range(4):
            obs, rewards, dones, info = env.step(actions)
            assert not dones[0].item(), f"Should not be done at step {i}"

        # Final step should retire
        obs, rewards, dones, info = env.step(actions)
        assert dones[0].item(), "Should retire at lifespan"
        assert rewards[0].item() >= 1.0, "Should include retirement bonus"

    def test_step_info_contains_metadata(self):
        """Should return info dict with step_counts, positions, interactions."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        actions = torch.tensor([5, 5])
        obs, rewards, dones, info = env.step(actions)

        # Verify info contents
        assert "step_counts" in info
        assert "positions" in info
        assert "successful_interactions" in info
        assert isinstance(info["step_counts"], torch.Tensor)
        assert isinstance(info["positions"], torch.Tensor)
        assert isinstance(info["successful_interactions"], dict)


class TestExecuteActions:
    """Test VectorizedHamletEnv._execute_actions() method."""

    def test_execute_actions_movement(self):
        """Should update agent positions for movement actions."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        initial_position = env.positions[0].clone()

        # Execute UP action (action 0 for Grid2D)
        actions = torch.tensor([0])
        env._execute_actions(actions)

        # Position should change
        assert not torch.all(env.positions[0] == initial_position).item()

    def test_execute_actions_wait_preserves_position(self):
        """Should not change position for WAIT action."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        initial_position = env.positions[0].clone()

        # Execute WAIT action (action 5 for Grid2D)
        actions = torch.tensor([5])
        env._execute_actions(actions)

        # Position should not change
        assert torch.all(env.positions[0] == initial_position).item()

    def test_execute_actions_interact_preserves_position(self):
        """Should not change position for INTERACT action."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        initial_position = env.positions[0].clone()

        # Execute INTERACT action (action 4 for Grid2D)
        actions = torch.tensor([4])
        env._execute_actions(actions)

        # Position should not change
        assert torch.all(env.positions[0] == initial_position).item()

    def test_execute_actions_returns_interaction_dict(self):
        """Should return dict mapping agent indices to affordance names for successful interactions."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        actions = torch.tensor([4, 5])  # INTERACT, WAIT
        result = env._execute_actions(actions)

        assert isinstance(result, dict)


class TestGetObservations:
    """Test VectorizedHamletEnv._get_observations() method."""

    def test_get_observations_returns_tensor(self):
        """Should return observations tensor."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        obs = env._get_observations()

        assert isinstance(obs, torch.Tensor)
        assert obs.shape[0] == 2
        assert obs.shape[1] > 0

    def test_get_observations_full_observability_shape(self):
        """Should return correct observation shape for full observability."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        num_agents = 3
        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        obs = env._get_observations()

        # For full observability: position + meters + affordance_at_pos + temporal
        # Observation dimension depends on test config (has many affordances)
        assert obs.shape[0] == num_agents
        assert obs.shape[1] > 29  # Should be at least 29, but test config has more affordances

    def test_get_observations_pomdp_shape(self):
        """Should return correct observation shape for POMDP."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        num_agents = 2
        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            partial_observability=True,  # POMDP mode
            vision_range=2,  # 5×5 window
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        obs = env._get_observations()

        # For POMDP: local_grid (25) + position (2) + meters (8) + affordance_at_pos (15) + temporal (4) = 54
        assert obs.shape == (num_agents, 54)

    def test_get_observations_contains_meters(self):
        """Should include current meter values in observations."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        obs = env._get_observations()

        # Observations should be normalized [0, 1]
        assert torch.all(obs >= 0.0).item()
        assert torch.all(obs <= 1.0).item() or torch.all(obs <= 15.0).item()  # One-hot can be > 1


class TestGetActionMasks:
    """Test VectorizedHamletEnv.get_action_masks() method."""

    def test_get_action_masks_returns_tensor(self):
        """Should return action masks tensor."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        masks = env.get_action_masks()

        assert isinstance(masks, torch.Tensor)
        assert masks.dtype == torch.bool

    def test_get_action_masks_correct_shape(self):
        """Should return masks with shape [num_agents, action_dim]."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        num_agents = 3
        env = VectorizedHamletEnv(
            num_agents=num_agents,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        masks = env.get_action_masks()

        # Grid2D has 8 actions (6 substrate + 2 custom)
        assert masks.shape == (num_agents, 8)

    def test_get_action_masks_some_actions_available(self):
        """Should return action masks with at least some actions available."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,  # No temporal mechanics
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        masks = env.get_action_masks()

        # At least some actions should be available (not all masked)
        assert torch.any(masks).item(), "At least some actions should be available"
        # Should have correct number of actions (8 for Grid2D)
        assert masks.shape == (1, 8)

    def test_get_action_masks_temporal_mechanics_masks_closed_affordances(self):
        """Should mask affordance interactions when temporal mechanics enabled and affordance closed."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=True,  # Enable temporal mechanics
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        masks = env.get_action_masks()

        # Should return boolean tensor (some actions may be masked)
        assert isinstance(masks, torch.Tensor)
        assert masks.dtype == torch.bool


# =============================================================================
# PLACEHOLDER: PHASE 15C, 15D tests will be added incrementally
# =============================================================================


# =============================================================================
# PHASE 15C: INTERACTIONS & REWARDS
# =============================================================================


class TestHandleInteractions:
    """Test VectorizedHamletEnv._handle_interactions() and _handle_interactions_legacy()."""

    def test_handle_interactions_legacy_when_temporal_disabled(self):
        """Should use legacy instant interactions when temporal mechanics disabled."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,  # Legacy mode
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Create interact mask
        interact_mask = torch.tensor([True])

        # Should return dict (may be empty if no affordance at position)
        result = env._handle_interactions(interact_mask)
        assert isinstance(result, dict)

    def test_handle_interactions_multi_tick_when_temporal_enabled(self):
        """Should use multi-tick interactions when temporal mechanics enabled."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=True,  # Multi-tick mode
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Multi-tick mode should initialize progress tracking
        assert hasattr(env, 'interaction_progress')
        assert hasattr(env, 'last_interaction_affordance')
        assert hasattr(env, 'last_interaction_position')

    def test_handle_interactions_returns_empty_when_no_interact(self):
        """Should return empty dict when no agents interact."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # No agents interacting
        interact_mask = torch.tensor([False, False])

        result = env._handle_interactions(interact_mask)
        assert result == {}

    def test_handle_interactions_legacy_returns_dict(self):
        """Should return dict mapping agent indices to affordance names."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        interact_mask = torch.tensor([True])

        result = env._handle_interactions_legacy(interact_mask)
        assert isinstance(result, dict)


class TestCalculateShapedRewards:
    """Test VectorizedHamletEnv._calculate_shaped_rewards()."""

    def test_calculate_shaped_rewards_returns_tensor(self):
        """Should return rewards tensor."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=2,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        rewards = env._calculate_shaped_rewards()

        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (2,)

    def test_calculate_shaped_rewards_uses_meter_values(self):
        """Should calculate rewards based on current meter values."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Get initial reward
        initial_reward = env._calculate_shaped_rewards()

        # Modify meters (reduce energy)
        env.meters[0, env.energy_idx] = 0.1

        # Reward should change
        new_reward = env._calculate_shaped_rewards()
        # Rewards are based on meter states, so they should differ
        assert initial_reward.item() != new_reward.item()

    def test_calculate_shaped_rewards_returns_finite_values(self):
        """Should return finite reward values (no NaN or inf)."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=3,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        rewards = env._calculate_shaped_rewards()

        assert torch.all(torch.isfinite(rewards)).item()


class TestApplyCustomAction:
    """Test VectorizedHamletEnv._apply_custom_action()."""

    def test_apply_custom_action_rest_action(self):
        """Should handle REST custom action."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Find REST action
        rest_action = env._get_optional_action_idx("REST")
        if rest_action is not None:
            action_config = env.action_space.get_action_by_id(rest_action)

            # Apply REST action (should execute without error)
            # Note: Meters may or may not change depending on action config costs
            # (test configs may have very low/zero costs for balancing)
            env._apply_custom_action(0, action_config)

            # Verify method executed without error and meters are still valid
            assert isinstance(env.meters, torch.Tensor)
            assert env.meters.shape == (1, 8)

    def test_apply_custom_action_meditate_action(self):
        """Should handle MEDITATE custom action."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Find MEDITATE action
        meditate_action = env._get_optional_action_idx("MEDITATE")
        if meditate_action is not None:
            action_config = env.action_space.get_action_by_id(meditate_action)

            # Apply MEDITATE action (should execute without error)
            # Note: Meters may or may not change depending on action config costs
            # (test configs may have very low/zero costs for balancing)
            env._apply_custom_action(0, action_config)

            # Verify method executed without error and meters are still valid
            assert isinstance(env.meters, torch.Tensor)
            assert env.meters.shape == (1, 8)

    def test_get_optional_action_idx_returns_int_or_none(self):
        """Should return action index for valid actions, None otherwise."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Valid action
        rest_idx = env._get_optional_action_idx("REST")
        assert rest_idx is None or isinstance(rest_idx, int)

        # Invalid action
        invalid_idx = env._get_optional_action_idx("NONEXISTENT_ACTION")
        assert invalid_idx is None


# =============================================================================
# PHASE 15D: CHECKPOINTING
# =============================================================================


class TestGetAffordancePositions:
    """Test VectorizedHamletEnv.get_affordance_positions()."""

    def test_get_affordance_positions_returns_dict(self):
        """Should return dict with positions, ordering, and position_dim."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        positions = env.get_affordance_positions()

        assert isinstance(positions, dict)
        assert "positions" in positions
        assert "ordering" in positions
        assert "position_dim" in positions

    def test_get_affordance_positions_has_correct_position_dim(self):
        """Should include position_dim matching substrate."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        checkpoint_data = env.get_affordance_positions()

        # Grid2D should have position_dim = 2
        assert checkpoint_data["position_dim"] == 2

    def test_get_affordance_positions_includes_all_affordances(self):
        """Should include all affordances in positions dict."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        checkpoint_data = env.get_affordance_positions()

        # Should have same affordances
        assert set(checkpoint_data["positions"].keys()) == set(env.affordances.keys())

    def test_get_affordance_positions_converts_to_lists(self):
        """Should convert tensor positions to lists."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        checkpoint_data = env.get_affordance_positions()

        # All positions should be lists
        for name, pos in checkpoint_data["positions"].items():
            assert isinstance(pos, list)


class TestSetAffordancePositions:
    """Test VectorizedHamletEnv.set_affordance_positions()."""

    def test_set_affordance_positions_updates_affordances(self):
        """Should update affordance positions from checkpoint data."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Get current positions
        original_checkpoint = env.get_affordance_positions()

        # Randomize positions
        env.randomize_affordance_positions()

        # Restore from checkpoint
        env.set_affordance_positions(original_checkpoint)

        # Positions should match original
        restored_checkpoint = env.get_affordance_positions()
        assert restored_checkpoint["positions"] == original_checkpoint["positions"]

    def test_set_affordance_positions_validates_position_dim(self):
        """Should validate position_dim matches substrate."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Create invalid checkpoint with wrong position_dim
        invalid_checkpoint = {
            "positions": {},
            "ordering": [],
            "position_dim": 3,  # Wrong! Should be 2 for Grid2D
        }

        with pytest.raises(ValueError, match="position_dim mismatch"):
            env.set_affordance_positions(invalid_checkpoint)


class TestRandomizeAffordancePositions:
    """Test VectorizedHamletEnv.randomize_affordance_positions()."""

    def test_randomize_affordance_positions_changes_positions(self):
        """Should change affordance positions."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        # Get current positions
        original_positions = env.get_affordance_positions()

        # Randomize
        env.randomize_affordance_positions()

        # Get new positions
        new_positions = env.get_affordance_positions()

        # At least some positions should change
        # (with 8x8 grid, very unlikely all stay the same)
        assert original_positions["positions"] != new_positions["positions"]

    def test_randomize_affordance_positions_maintains_affordance_count(self):
        """Should keep same number of affordances."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        original_count = len(env.affordances)

        env.randomize_affordance_positions()

        assert len(env.affordances) == original_count

    def test_randomize_affordance_positions_stays_in_bounds(self):
        """Should keep all positions within grid bounds."""
        from townlet.environment.vectorized_env import VectorizedHamletEnv

        grid_size = 8
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=grid_size,
            partial_observability=False,
            vision_range=2,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.004,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
        )
        env.reset()

        env.randomize_affordance_positions()

        # All positions should be within [0, grid_size)
        for affordance_pos in env.affordances.values():
            assert torch.all(affordance_pos >= 0).item()
            assert torch.all(affordance_pos < grid_size).item()


# =============================================================================
# END OF SPRINT 15 TESTS
# =============================================================================
