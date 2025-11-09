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

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
import torch
import yaml

from townlet.universe.errors import CompilationError


@pytest.fixture
def cpu_env_factory(env_factory, cpu_device):
    """Helper to build CPU-bound environments for these tests."""

    def _build(**kwargs):
        return env_factory(device_override=cpu_device, **kwargs)

    return _build


@pytest.fixture
def custom_env_builder(tmp_path, test_config_pack_path, env_factory, cpu_device):
    """Return a builder that clones config packs with training overrides."""

    def _build(*, num_agents: int = 1, overrides: dict | None = None, source_pack: Path | None = None):
        source = Path(source_pack) if source_pack is not None else test_config_pack_path
        target = tmp_path / f"env_{uuid.uuid4().hex}"
        shutil.copytree(source, target)

        if overrides:
            training_path = target / "training.yaml"
            with open(training_path) as f:
                training_data = yaml.safe_load(f)

            for section, updates in overrides.items():
                section_data = training_data.get(section, {}) or {}
                section_data.update(updates)
                training_data[section] = section_data

            with open(training_path, "w") as f:
                yaml.safe_dump(training_data, f, sort_keys=False)

        return env_factory(config_dir=target, num_agents=num_agents, device_override=cpu_device)

    return _build


# =============================================================================
# PHASE 15A: INITIALIZATION & SETUP
# =============================================================================


class TestVectorizedHamletEnvInitialization:
    """Test VectorizedHamletEnv.__init__ with various configurations."""

    def test_init_requires_substrate_yaml(self, temp_test_dir, compile_universe):
        """Should raise FileNotFoundError if substrate.yaml is missing."""
        config_pack = temp_test_dir / "config_pack"
        shutil.copytree(Path("configs/test"), config_pack)
        (config_pack / "substrate.yaml").unlink()

        with pytest.raises(CompilationError, match="Substrate config not found"):
            compile_universe(config_pack)

    def test_init_raises_if_config_pack_not_found(self, compile_universe):
        """Should raise FileNotFoundError if config pack directory doesn't exist."""
        with pytest.raises(CompilationError, match="Config file not found"):
            compile_universe(Path("/nonexistent/path"))

    def test_env_init_does_not_construct_cascade_engine(self, monkeypatch, cpu_env_factory):
        """VectorizedHamletEnv should rely on optimization tensors, not CascadeEngine."""

        def _boom(self, *args, **kwargs):  # pragma: no cover - executed on regression
            raise AssertionError("CascadeEngine should not be instantiated at runtime")

        monkeypatch.setattr("townlet.environment.cascade_engine.CascadeEngine.__init__", _boom)

        env = cpu_env_factory()
        env.reset()

    def test_env_defaults_to_test_config_pack(self, cpu_env_factory, test_config_pack_path: Path):
        """Env factory defaults to configs/test pack (compiled path)."""

        env = cpu_env_factory()
        assert Path(env.config_pack_path).resolve() == test_config_pack_path.resolve()

    def test_init_creates_substrate_from_config(self, cpu_env_factory):
        """Should load substrate.yaml and create substrate via factory."""
        env = cpu_env_factory(num_agents=2)

        # Verify substrate was created
        assert env.substrate is not None
        assert hasattr(env.substrate, "position_dim")

    def test_init_loads_affordances_from_config(self, cpu_env_factory):
        """Should load affordances.yaml and create affordance engine."""
        env = cpu_env_factory()

        # Verify affordance engine was created
        assert env.affordance_engine is not None
        assert hasattr(env.affordance_engine, "affordances")

    def test_init_initializes_state_tensors(self, cpu_env_factory):
        """Should initialize positions, meters, lifetimes, and dones tensors."""
        num_agents = 3
        env = cpu_env_factory(num_agents=num_agents)
        env.reset()

        # Verify state tensors exist with correct shapes
        assert env.positions.shape == (num_agents, env.substrate.position_dim)
        assert env.meters.shape == (num_agents, env.meter_count)
        assert env.dones.shape == (num_agents,)

        # Verify tensors are on correct device
        assert env.positions.device == env.device
        assert env.meters.device == env.device

    def test_init_sets_device_correctly(self, env_factory):
        """Should set device to CPU by default or use provided device."""
        env_cpu = env_factory(device_override="cpu")
        assert env_cpu.device == torch.device("cpu")

        env_explicit = env_factory(device_override=torch.device("cpu"))
        assert env_explicit.device == torch.device("cpu")


class TestVectorizedHamletEnvReset:
    """Test VectorizedHamletEnv.reset() method."""

    def test_reset_returns_observations(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        obs = env.reset()
        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (env.num_agents, env.observation_dim)

    def test_reset_initializes_meters_from_config(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.reset()
        assert torch.all(env.meters >= 0.0)
        assert torch.all(env.meters <= 1.0)
        assert torch.all(env.meters[0] == env.meters[1])

    def test_reset_clears_dones_flag(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.dones = torch.ones(2, dtype=torch.bool)
        env.reset()
        assert torch.all(~env.dones)

    def test_reset_initializes_step_counts(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.step_counts = torch.tensor([10, 20])
        env.reset()
        assert torch.all(env.step_counts == 0)

    def test_reset_randomizes_agent_positions(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=5)
        env.reset()
        assert torch.all(env.positions >= 0)
        assert torch.all(env.positions < env.grid_size)

    def test_reset_temporal_mechanics_initializes_time(self, env_factory, cpu_device):
        env = env_factory(
            config_dir=Path("configs/L3_temporal_mechanics"),
            num_agents=1,
            device_override=cpu_device,
        )
        env.reset()
        assert env.enable_temporal_mechanics is True
        assert env.time_of_day == 0


class TestBuildMovementDeltas:
    """Test VectorizedHamletEnv._build_movement_deltas() method."""

    def test_build_movement_deltas_creates_tensor(self, cpu_env_factory):
        """Should create movement deltas tensor from substrate actions."""

        env = cpu_env_factory()

        deltas = env._build_movement_deltas()

        # Should return tensor with shape [substrate_action_count, position_dim]
        # For Grid2D: 6 substrate actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT)
        assert isinstance(deltas, torch.Tensor)
        assert deltas.shape[0] == env.action_space.substrate_action_count
        assert deltas.shape[1] == env.substrate.position_dim

    def test_build_movement_deltas_correct_values_grid2d(self, cpu_env_factory):
        """Should create correct deltas for Grid2D substrate (UP, DOWN, LEFT, RIGHT)."""

        env = cpu_env_factory()

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

    def test_step_returns_correct_types(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.reset()

        actions = torch.full((2,), 5, device=env.device, dtype=torch.long)
        obs, rewards, dones, info = env.step(actions)

        assert isinstance(obs, torch.Tensor)
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(dones, torch.Tensor)
        assert isinstance(info, dict)

    def test_step_returns_correct_shapes(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=3)
        env.reset()

        actions = torch.full((3,), 5, device=env.device, dtype=torch.long)
        obs, rewards, dones, _ = env.step(actions)

        assert obs.shape == (3, env.observation_dim)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)

    def test_step_increments_step_counts(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.reset()

        initial_counts = env.step_counts.clone()
        actions = torch.full((2,), 5, device=env.device, dtype=torch.long)
        env.step(actions)

        assert torch.all(env.step_counts == initial_counts + 1)

    def test_step_depletes_meters(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.reset()

        initial_meters = env.meters.clone()
        actions = torch.full((2,), 5, device=env.device, dtype=torch.long)
        env.step(actions)

        assert not torch.allclose(env.meters, initial_meters)

    def test_step_increments_time_of_day(self, custom_env_builder):
        env = custom_env_builder(
            overrides={"environment": {"enable_temporal_mechanics": True}},
        )
        env.reset()

        actions = torch.tensor([5], device=env.device)
        env.step(actions)
        assert env.time_of_day == 1

        env.time_of_day = 23
        env.step(actions)
        assert env.time_of_day == 0

    def test_step_retirement_bonus(self, custom_env_builder):
        env = custom_env_builder(
            overrides={"curriculum": {"max_steps_per_episode": 5}},
        )
        env.reset()

        actions = torch.tensor([5], device=env.device)
        for _ in range(4):
            _, _, dones, _ = env.step(actions)
            assert not dones[0]

        _, rewards, dones, _ = env.step(actions)
        assert dones[0]
        assert rewards[0].item() >= 1.0

    def test_step_info_contains_metadata(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.reset()

        actions = torch.full((2,), 5, device=env.device, dtype=torch.long)
        _, _, _, info = env.step(actions)

        assert set(info.keys()) >= {"step_counts", "positions", "successful_interactions"}
        assert isinstance(info["step_counts"], torch.Tensor)
        assert isinstance(info["positions"], torch.Tensor)
        assert isinstance(info["successful_interactions"], dict)


class TestExecuteActions:
    """Test VectorizedHamletEnv._execute_actions() method."""

    def test_execute_actions_movement(self, cpu_env_factory):
        """Should update agent positions for movement actions."""

        env = cpu_env_factory()
        env.reset()

        # Place agent away from borders so the UP action can't clamp back into the
        # same cell, which previously made this test randomly fail when the
        # sampled spawn started on y=0.
        env.positions[0] = torch.tensor([3, 3], device=env.device, dtype=env.positions.dtype)

        initial_position = env.positions[0].clone()

        # Execute UP action (action 0 for Grid2D)
        actions = torch.tensor([0], device=env.device)
        env._execute_actions(actions)

        # Position should change
        assert not torch.all(env.positions[0] == initial_position).item()

    def test_execute_actions_wait_preserves_position(self, cpu_env_factory):
        """Should not change position for WAIT action."""
        env = cpu_env_factory()
        env.reset()

        initial_position = env.positions[0].clone()

        # Execute WAIT action (action 5 for Grid2D)
        actions = torch.tensor([5], device=env.device)
        env._execute_actions(actions)

        # Position should not change
        assert torch.all(env.positions[0] == initial_position).item()

    def test_execute_actions_interact_preserves_position(self, cpu_env_factory):
        """Should not change position for INTERACT action."""
        env = cpu_env_factory()
        env.reset()

        initial_position = env.positions[0].clone()

        # Execute INTERACT action (action 4 for Grid2D)
        actions = torch.tensor([4], device=env.device)
        env._execute_actions(actions)

        # Position should not change
        assert torch.all(env.positions[0] == initial_position).item()

    def test_execute_actions_returns_interaction_dict(self, cpu_env_factory):
        """Should return dict mapping agent indices to affordance names for successful interactions."""
        env = cpu_env_factory(num_agents=2)
        env.reset()

        actions = torch.tensor([4, 5], device=env.device)  # INTERACT, WAIT
        result = env._execute_actions(actions)

        assert isinstance(result, dict)


class TestGetObservations:
    """Test VectorizedHamletEnv._get_observations() method."""

    def test_get_observations_returns_tensor(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.reset()

        obs = env._get_observations()

        assert obs.shape == (2, env.observation_dim)

    def test_get_observations_full_observability_shape(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=3)
        env.reset()
        obs = env._get_observations()
        assert obs.shape == (3, env.observation_dim)
        assert env.partial_observability is False

    def test_get_observations_pomdp_shape(self, custom_env_builder):
        env = custom_env_builder(
            num_agents=2,
            overrides={"environment": {"partial_observability": True, "vision_range": 2}},
        )
        env.reset()
        obs = env._get_observations()
        assert env.partial_observability is True
        assert obs.shape == (2, env.observation_dim)

    def test_get_observations_contains_meters(self, cpu_env_factory):
        env = cpu_env_factory()
        env.reset()
        obs = env._get_observations()
        assert torch.all(obs[:, -4:] >= -1.0)

    def test_get_observations_uses_substrate_position_encoder(self, cpu_env_factory, monkeypatch):
        env = cpu_env_factory(num_agents=2)
        env.reset()

        expected = torch.full((env.num_agents, env.substrate.position_dim), 0.42, device=env.device)

        def fake_encoder(positions, affordances):
            return expected

        def fail_normalize(_):
            raise AssertionError("normalize_positions should not be called when encoder exists")

        monkeypatch.setattr(env.substrate, "_encode_position_features", fake_encoder)
        monkeypatch.setattr(env.substrate, "normalize_positions", fail_normalize, raising=False)

        env._get_observations()
        stored = env.vfs_registry.get("position", reader="agent")
        assert torch.allclose(stored, expected)

    def test_get_observations_falls_back_to_encode_observation(self, cpu_env_factory, monkeypatch):
        env = cpu_env_factory(config_dir=Path("configs/L1_continuous_2D"))
        env.reset()

        expected = torch.full((env.num_agents, env.substrate.position_dim), 0.25, device=env.device)

        def fake_encode_observation(positions, affordances):
            return expected

        def fail_normalize(_):
            raise AssertionError("normalize_positions fallback should not trigger when encode_observation exists")

        monkeypatch.setattr(env.substrate, "encode_observation", fake_encode_observation)
        monkeypatch.setattr(env.substrate, "normalize_positions", fail_normalize, raising=False)

        env._get_observations()
        stored = env.vfs_registry.get("position", reader="agent")
        assert torch.allclose(stored, expected)

    def test_get_observations_handles_agent_private_scope(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.reset()

        env.vfs_registry.variables["energy"].scope = "agent_private"
        obs = env._get_observations()
        assert obs.shape[0] == env.num_agents


class TestGetActionMasks:
    """Test VectorizedHamletEnv.get_action_masks() method."""

    def test_get_action_masks_returns_tensor(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=2)
        env.reset()
        masks = env.get_action_masks()
        assert masks.dtype == torch.bool

    def test_get_action_masks_correct_shape(self, cpu_env_factory):
        env = cpu_env_factory(num_agents=3)
        env.reset()
        masks = env.get_action_masks()
        assert masks.shape == (3, env.action_dim)

    def test_get_action_masks_some_actions_available(self, cpu_env_factory):
        env = cpu_env_factory()
        env.reset()
        masks = env.get_action_masks()
        assert torch.any(masks)
        assert masks.shape == (1, env.action_dim)

    def test_get_action_masks_temporal_mechanics_masks_closed_affordances(self, custom_env_builder):
        env = custom_env_builder(
            overrides={"environment": {"enable_temporal_mechanics": True}},
        )
        env.reset()

        bar_pos = env.affordances.get("Bar")
        if bar_pos is None:
            pytest.skip("Test config missing 'Bar' affordance")

        env.positions[0] = bar_pos.clone()

        env.time_of_day = 10  # Bar closed mid-morning
        closed_masks = env.get_action_masks()
        assert not closed_masks[0, env.interact_action_idx]

        env.time_of_day = 20  # Bar open in evening
        open_masks = env.get_action_masks()
        assert open_masks[0, env.interact_action_idx]


# =============================================================================
# PLACEHOLDER: PHASE 15C, 15D tests will be added incrementally
# =============================================================================


# =============================================================================
# PHASE 15C: INTERACTIONS & REWARDS
# =============================================================================


class TestHandleInteractions:
    """Test VectorizedHamletEnv._handle_interactions() and _handle_interactions_legacy()."""

    def test_handle_interactions_legacy_when_temporal_disabled(self, cpu_env_factory):
        """Should use legacy instant interactions when temporal mechanics disabled."""
        env = cpu_env_factory()
        env.reset()

        # Create interact mask
        interact_mask = torch.tensor([True])

        # Should return dict (may be empty if no affordance at position)
        result = env._handle_interactions(interact_mask)
        assert isinstance(result, dict)

    def test_handle_interactions_multi_tick_when_temporal_enabled(self, custom_env_builder):
        """Should use multi-tick interactions when temporal mechanics enabled."""
        env = custom_env_builder(overrides={"environment": {"enable_temporal_mechanics": True}})
        env.reset()

        # Multi-tick mode should initialize progress tracking
        assert hasattr(env, "interaction_progress")
        assert hasattr(env, "last_interaction_affordance")
        assert hasattr(env, "last_interaction_position")

    def test_handle_interactions_returns_empty_when_no_interact(self, cpu_env_factory):
        """Should return empty dict when no agents interact."""
        env = cpu_env_factory(num_agents=2)
        env.reset()

        # No agents interacting
        interact_mask = torch.tensor([False, False])

        result = env._handle_interactions(interact_mask)
        assert result == {}

    def test_handle_interactions_legacy_returns_dict(self, cpu_env_factory):
        """Should return dict mapping agent indices to affordance names."""
        env = cpu_env_factory()
        env.reset()

        interact_mask = torch.tensor([True])

        result = env._handle_interactions_legacy(interact_mask)
        assert isinstance(result, dict)


class TestCalculateShapedRewards:
    """Test VectorizedHamletEnv._calculate_shaped_rewards()."""

    def test_calculate_shaped_rewards_returns_tensor(self, cpu_env_factory):
        """Should return rewards tensor."""
        env = cpu_env_factory(num_agents=2)
        env.reset()

        rewards = env._calculate_shaped_rewards()

        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (2,)

    def test_calculate_shaped_rewards_uses_meter_values(self, cpu_env_factory):
        """Should calculate rewards based on current meter values."""
        env = cpu_env_factory()
        env.reset()

        # Get initial reward
        initial_reward = env._calculate_shaped_rewards()

        # Modify meters (reduce energy)
        env.meters[0, env.energy_idx] = 0.1

        # Reward should change
        new_reward = env._calculate_shaped_rewards()
        # Rewards are based on meter states, so they should differ
        assert initial_reward.item() != new_reward.item()

    def test_calculate_shaped_rewards_returns_finite_values(self, cpu_env_factory):
        """Should return finite reward values (no NaN or inf)."""
        env = cpu_env_factory(num_agents=3)
        env.reset()

        rewards = env._calculate_shaped_rewards()

        assert torch.all(torch.isfinite(rewards)).item()


class TestApplyCustomAction:
    """Test VectorizedHamletEnv._apply_custom_action()."""

    def test_apply_custom_action_rest_action(self, cpu_env_factory):
        """Should handle REST custom action."""
        env = cpu_env_factory()
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
            assert env.meters.shape == (1, env.meter_count)

    def test_apply_custom_action_meditate_action(self, cpu_env_factory):
        """Should handle MEDITATE custom action."""
        env = cpu_env_factory()
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

    def test_static_affordance_positions_respected_when_randomization_disabled(
        self,
        tmp_path,
        test_config_pack_path: Path,
        env_factory,
        cpu_device,
    ):
        target = tmp_path / "static_positions_pack"
        shutil.copytree(test_config_pack_path, target)

        training_path = target / "training.yaml"
        training_data = yaml.safe_load(training_path.read_text())
        training_data.setdefault("environment", {})["randomize_affordances"] = False
        training_path.write_text(yaml.safe_dump(training_data, sort_keys=False))

        affordance_path = target / "affordances.yaml"
        affordance_data = yaml.safe_load(affordance_path.read_text())
        fixed_positions = {
            "Bed": [0, 0],
            "LuxuryBed": [1, 0],
        }
        for entry in affordance_data.get("affordances", []):
            if entry["name"] in fixed_positions:
                entry["position"] = fixed_positions[entry["name"]]
        affordance_path.write_text(yaml.safe_dump(affordance_data, sort_keys=False))

        env = env_factory(config_dir=target, num_agents=1, device_override=cpu_device)
        env.reset()

        assert env.randomize_affordances is False
        bed_pos = env.affordances.get("Bed")
        luxury_bed_pos = env.affordances.get("LuxuryBed")
        assert torch.allclose(bed_pos, torch.tensor([0, 0], dtype=env.substrate.position_dtype, device=env.device))
        assert torch.allclose(luxury_bed_pos, torch.tensor([1, 0], dtype=env.substrate.position_dtype, device=env.device))

        # Subsequent calls to randomize should no-op when disabled
        env.randomize_affordance_positions()
        assert torch.allclose(bed_pos, env.affordances.get("Bed"))


# =============================================================================
# END OF SPRINT 15 TESTS
# =============================================================================
