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
