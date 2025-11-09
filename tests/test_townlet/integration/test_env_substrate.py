"""Test environment substrate integration (core functionality)."""

import shutil
from pathlib import Path

import pytest
import torch
import yaml

from townlet.substrate.grid2d import Grid2DSubstrate

CONFIG_L1 = Path("configs/L1_full_observability")
ASPARTIAL_CONFIG = Path("configs/aspatial_test")


def test_env_loads_substrate_config(cpu_env_factory):
    """Environment should load substrate.yaml and create substrate instance."""
    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=1)

    # Verify substrate loaded
    assert hasattr(env, "grid_size")
    assert env.grid_size == 8


def test_env_substrate_accessible(cpu_env_factory):
    """Environment should expose substrate for inspection."""
    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=1)

    # Verify substrate is accessible and correct type
    assert hasattr(env, "substrate")
    assert isinstance(env.substrate, Grid2DSubstrate)
    assert env.substrate.width == 8


def test_env_initializes_positions_via_substrate(cpu_env_factory):
    """Environment should use substrate.initialize_positions() in reset()."""
    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=5)

    # Reset environment
    env.reset()

    # Positions should be initialized via substrate
    assert env.positions.shape == (5, 2)  # [num_agents, position_dim]
    assert env.positions.dtype == torch.long
    assert env.positions.device == torch.device("cpu")

    # Positions should be within grid bounds
    assert (env.positions >= 0).all()
    assert (env.positions < 8).all()


def test_env_applies_movement_via_substrate(cpu_env_factory):
    """Environment should use substrate.apply_movement() for boundary handling."""
    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=1)

    env.reset()

    # Place agent at top-left corner
    env.positions = torch.tensor([[0, 0]], dtype=torch.long, device=torch.device("cpu"))

    # Try to move up (action 0 = UP)
    action = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))

    env.step(action)

    # Position should be clamped by substrate (boundary="clamp")
    assert (env.positions[:, 0] >= 0).all()  # X within bounds
    assert (env.positions[:, 1] >= 0).all()  # Y within bounds


def test_env_randomizes_affordances_via_substrate(cpu_env_factory):
    """Environment should use substrate.get_all_positions() for affordance placement."""
    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=1)

    # Randomize affordances
    env.randomize_affordance_positions()

    # All affordances should have valid positions within grid
    for affordance_name, position in env.affordances.items():
        assert position.shape == (2,)  # [x, y]
        assert (position >= 0).all()
        assert (position < 8).all()

    # Affordances should not overlap (each at unique position)
    positions_list = [tuple(pos.tolist()) for pos in env.affordances.values()]
    assert len(positions_list) == len(set(positions_list))  # All unique


# =============================================================================
# INTEGRATION TESTS (from TASK-002A)
# =============================================================================
# Phase 4 complete - integration tests enabled
# pytestmark = pytest.mark.skip(reason="Phase 4 (Environment Integration) not yet complete")


@pytest.mark.parametrize(
    "config_name",
    [
        "L0_0_minimal",
        "L0_5_dual_resource",
        "L1_full_observability",
        "L2_partial_observability",
        "L3_temporal_mechanics",
    ],
)
def test_env_observation_dim_unchanged(cpu_env_factory, config_name):
    """Environment with substrate.yaml using coordinate encoding (all same dims)."""
    config_path = Path("configs") / config_name
    env = cpu_env_factory(config_dir=config_path, num_agents=1)

    # Calculate expected observation dimension based on observability mode
    if env.partial_observability:
        # POMDP: local_window + position + meters + affordances + temporal
        window_size = 2 * env.vision_range + 1
        local_window_dim = window_size**env.substrate.position_dim
        expected_obs_dim = local_window_dim + env.substrate.position_dim + env.meter_count + (env.num_affordance_types + 1) + 4
    else:
        # Full observability: grid_encoding + position + meters + affordances + temporal
        expected_obs_dim = env.substrate.get_observation_dim() + env.meter_count + (env.num_affordance_types + 1) + 4

    # Observation dimension should match expected breakdown
    assert env.observation_dim == expected_obs_dim

    # Verify substrate loaded correctly
    assert env.substrate is not None
    assert env.substrate.position_dim == 2  # 2D grid


@pytest.mark.parametrize(
    "config_name,expected_grid_size",
    [
        ("L0_0_minimal", 3),
        ("L0_5_dual_resource", 7),
        ("L1_full_observability", 8),
        ("L2_partial_observability", 8),
        ("L3_temporal_mechanics", 8),
    ],
)
def test_env_substrate_dimensions(cpu_env_factory, config_name, expected_grid_size):
    """Environment substrate should have correct grid dimensions."""
    config_path = Path("configs") / config_name
    env = cpu_env_factory(config_dir=config_path, num_agents=1)

    # Substrate should be Grid2D with correct dimensions
    assert env.substrate.width == expected_grid_size
    assert env.substrate.height == expected_grid_size
    assert env.substrate.width == env.substrate.height  # Square grid


def test_env_substrate_boundary_behavior(cpu_env_factory):
    """Environment substrate should use clamp boundary (legacy behavior)."""
    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=1)

    # Test boundary clamping (agent at edge trying to move out of bounds)
    positions = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")  # Top-left corner
    action_deltas = torch.tensor([[-1, -1]], dtype=torch.long, device="cpu")  # Try to move up-left

    new_positions = env.substrate.apply_movement(positions, action_deltas)

    # Should clamp to [0, 0] (not wrap or bounce)
    assert (new_positions == torch.tensor([[0, 0]], dtype=torch.long)).all()


def test_env_substrate_distance_metric(cpu_env_factory):
    """Environment substrate should use manhattan distance (legacy behavior)."""
    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=1)

    # Test manhattan distance calculation
    pos1 = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")
    pos2 = torch.tensor([[3, 4]], dtype=torch.long, device="cpu")

    distance = env.substrate.compute_distance(pos1, pos2)

    # Manhattan distance: |3-0| + |4-0| = 7
    assert distance.item() == 7.0


# =============================================================================
# BACKWARD COMPATIBILITY TESTS (from TASK-002A)
# =============================================================================


def test_grid_size_overridden_by_substrate(cpu_env_factory, tmp_path):
    """grid_size parameter should be overridden by substrate dimensions.

    Backward compatibility test: ensures substrate.yaml takes precedence.
    """
    import shutil

    config_copy = tmp_path / "l1_override"
    shutil.copytree(CONFIG_L1, config_copy)

    training_yaml = config_copy / "training.yaml"
    config = training_yaml.read_text()
    data = yaml.safe_load(config)
    data["environment"]["grid_size"] = 999
    training_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    env = cpu_env_factory(config_dir=config_copy, num_agents=1)

    # Verify grid_size comes from substrate (8), not parameter (999)
    assert env.grid_size == 8  # From substrate.yaml, NOT parameter
    assert env.substrate.width == 8
    assert env.substrate.height == 8


def test_aspatial_preserves_grid_size_parameter(cpu_env_factory, tmp_path):
    """Aspatial substrate should keep grid_size parameter value.

    Backward compatibility test: aspatial has no width/height, so parameter preserved.
    """
    config_copy = tmp_path / "aspatial_override"
    shutil.copytree(ASPARTIAL_CONFIG, config_copy)

    training_yaml = config_copy / "training.yaml"
    data = yaml.safe_load(training_yaml.read_text())
    data["environment"]["grid_size"] = 12
    training_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    env = cpu_env_factory(config_dir=config_copy, num_agents=1)

    # Verify grid_size parameter is preserved (not overridden)
    assert env.grid_size == 12  # From parameter (aspatial has no width/height)
    assert env.substrate.position_dim == 0  # Aspatial has no position
    assert not hasattr(env.substrate, "width")


# =============================================================================
# ACTION SPACE BUILDER INTEGRATION (TASK-002B Phase 4)
# =============================================================================


def test_vectorized_env_loads_composed_action_space(cpu_env_factory):
    """VectorizedHamletEnv should use ActionSpaceBuilder.

    TDD Test (RED phase): Verifies env has action_space attribute from builder.
    """
    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=1)

    # Should have composed action space (substrate + custom)
    assert hasattr(env, "action_space"), "Environment should have action_space attribute"
    assert env.action_space.action_dim >= 6  # At least substrate actions

    # action_dim should match action_space
    assert env.action_dim == env.action_space.action_dim

    # Should have cached action indices (no more hardcoded formulas)
    assert hasattr(env, "interact_action_idx"), "Should cache INTERACT action index"
    assert hasattr(env, "wait_action_idx"), "Should cache WAIT action index"

    # Indices should be valid (within action space bounds)
    assert 0 <= env.interact_action_idx < env.action_dim
    assert 0 <= env.wait_action_idx < env.action_dim


def test_action_masks_include_base_masking(cpu_env_factory):
    """Action masks should start with base mask from ActionSpace.

    TDD Test (RED phase): Verifies get_action_masks() uses
    action_space.get_base_action_mask() for disabled action masking.

    This test verifies integration between get_action_masks() and ActionSpace,
    even though all actions are currently enabled (enabled_actions loading is future task).
    """
    from unittest.mock import patch

    env = cpu_env_factory(config_dir=CONFIG_L1, num_agents=2)

    env.reset()

    # Mock the get_base_action_mask to verify it's being called
    with patch.object(env.action_space, "get_base_action_mask", wraps=env.action_space.get_base_action_mask) as mock_base_mask:
        # Call get_action_masks
        masks = env.get_action_masks()

        # Verify get_base_action_mask was called
        mock_base_mask.assert_called_once()

        # Verify it was called with correct arguments
        call_args = mock_base_mask.call_args
        assert call_args.kwargs["num_agents"] == 2
        assert call_args.kwargs["device"] == torch.device("cpu")

    # Verify shape and dtype
    assert masks.shape == (2, env.action_dim), f"Mask shape should be (2, {env.action_dim}), got {masks.shape}"
    assert masks.dtype == torch.bool, "Masks should be bool tensor"
