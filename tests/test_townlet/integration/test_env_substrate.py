"""Test environment substrate integration (core functionality)."""

from pathlib import Path

import pytest
import torch
import yaml

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.grid2d import Grid2DSubstrate


def test_env_loads_substrate_config():
    """Environment should load substrate.yaml and create substrate instance."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Verify substrate loaded
    assert hasattr(env, "grid_size")
    assert env.grid_size == 8


def test_env_substrate_accessible():
    """Environment should expose substrate for inspection."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Verify substrate is accessible and correct type
    assert hasattr(env, "substrate")
    assert isinstance(env.substrate, Grid2DSubstrate)
    assert env.substrate.width == 8


def test_env_initializes_positions_via_substrate():
    """Environment should use substrate.initialize_positions() in reset()."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=5,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Reset environment
    env.reset()

    # Positions should be initialized via substrate
    assert env.positions.shape == (5, 2)  # [num_agents, position_dim]
    assert env.positions.dtype == torch.long
    assert env.positions.device == torch.device("cpu")

    # Positions should be within grid bounds
    assert (env.positions >= 0).all()
    assert (env.positions < 8).all()


def test_env_applies_movement_via_substrate():
    """Environment should use substrate.apply_movement() for boundary handling."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    env.reset()

    # Place agent at top-left corner
    env.positions = torch.tensor([[0, 0]], dtype=torch.long, device=torch.device("cpu"))

    # Try to move up (action 0 = UP)
    action = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))

    env.step(action)

    # Position should be clamped by substrate (boundary="clamp")
    assert (env.positions[:, 0] >= 0).all()  # X within bounds
    assert (env.positions[:, 1] >= 0).all()  # Y within bounds


def test_env_randomizes_affordances_via_substrate():
    """Environment should use substrate.get_all_positions() for affordance placement."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

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


def load_enabled_affordances(config_pack_path: Path) -> list[str]:
    """Load enabled_affordances from training.yaml."""
    training_yaml = config_pack_path / "training.yaml"
    with open(training_yaml) as f:
        config = yaml.safe_load(f)
    return config["environment"]["enabled_affordances"]


def load_partial_observability(config_pack_path: Path) -> bool:
    """Load partial_observability setting from training.yaml."""
    training_yaml = config_pack_path / "training.yaml"
    with open(training_yaml) as f:
        config = yaml.safe_load(f)
    return config["environment"]["partial_observability"]


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
def test_env_observation_dim_unchanged(config_name):
    """Environment with substrate.yaml using coordinate encoding (all same dims)."""
    config_path = Path("configs") / config_name
    enabled_affordances = load_enabled_affordances(config_path)
    partial_observability = load_partial_observability(config_path)

    env = VectorizedHamletEnv(
        config_pack_path=config_path,
        num_agents=1,
        grid_size=8,
        partial_observability=partial_observability,  # Use config's POMDP setting
        vision_range=2,
        enable_temporal_mechanics=False,
        enabled_affordances=enabled_affordances,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Calculate expected observation dimension based on observability mode
    if partial_observability:
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
def test_env_substrate_dimensions(config_name, expected_grid_size):
    """Environment substrate should have correct grid dimensions."""
    config_path = Path("configs") / config_name
    enabled_affordances = load_enabled_affordances(config_path)

    env = VectorizedHamletEnv(
        config_pack_path=config_path,
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        enabled_affordances=enabled_affordances,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Substrate should be Grid2D with correct dimensions
    assert env.substrate.width == expected_grid_size
    assert env.substrate.height == expected_grid_size
    assert env.substrate.width == env.substrate.height  # Square grid


def test_env_substrate_boundary_behavior():
    """Environment substrate should use clamp boundary (legacy behavior)."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs") / "L1_full_observability",
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Test boundary clamping (agent at edge trying to move out of bounds)
    positions = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")  # Top-left corner
    action_deltas = torch.tensor([[-1, -1]], dtype=torch.long, device="cpu")  # Try to move up-left

    new_positions = env.substrate.apply_movement(positions, action_deltas)

    # Should clamp to [0, 0] (not wrap or bounce)
    assert (new_positions == torch.tensor([[0, 0]], dtype=torch.long)).all()


def test_env_substrate_distance_metric():
    """Environment substrate should use manhattan distance (legacy behavior)."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs") / "L1_full_observability",
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Test manhattan distance calculation
    pos1 = torch.tensor([[0, 0]], dtype=torch.long, device="cpu")
    pos2 = torch.tensor([[3, 4]], dtype=torch.long, device="cpu")

    distance = env.substrate.compute_distance(pos1, pos2)

    # Manhattan distance: |3-0| + |4-0| = 7
    assert distance.item() == 7.0


# =============================================================================
# BACKWARD COMPATIBILITY TESTS (from TASK-002A)
# =============================================================================


def test_grid_size_overridden_by_substrate():
    """grid_size parameter should be overridden by substrate dimensions.

    Backward compatibility test: ensures substrate.yaml takes precedence.
    """
    from pathlib import Path

    import torch

    from townlet.environment.vectorized_env import VectorizedHamletEnv

    # Use L1_full_observability which has 8×8 grid in substrate.yaml
    # But pass grid_size=999 as parameter

    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=999,  # Parameter value (should be overridden)
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Verify grid_size comes from substrate (8), not parameter (999)
    assert env.grid_size == 8  # From substrate.yaml, NOT parameter
    assert env.substrate.width == 8
    assert env.substrate.height == 8


def test_aspatial_preserves_grid_size_parameter(tmp_path):
    """Aspatial substrate should keep grid_size parameter value.

    Backward compatibility test: aspatial has no width/height, so parameter preserved.
    """
    import shutil
    from pathlib import Path

    import torch

    from townlet.environment.vectorized_env import VectorizedHamletEnv

    # Create config pack with aspatial substrate
    config_pack = tmp_path / "test_config"
    config_pack.mkdir()

    # Create aspatial substrate.yaml
    substrate_yaml = config_pack / "substrate.yaml"
    substrate_yaml.write_text(
        """
version: "1.0"
description: "Aspatial substrate test"
type: "aspatial"
aspatial: {}
"""
    )

    # Copy complete config files from test config
    test_config = Path("configs/test")
    shutil.copy(test_config / "bars.yaml", config_pack / "bars.yaml")
    shutil.copy(test_config / "affordances.yaml", config_pack / "affordances.yaml")
    shutil.copy(test_config / "cascades.yaml", config_pack / "cascades.yaml")
    shutil.copy(test_config / "variables_reference.yaml", config_pack / "variables_reference.yaml")

    # Create environment with aspatial substrate and grid_size parameter
    env = VectorizedHamletEnv(
        config_pack_path=config_pack,
        num_agents=1,
        grid_size=12,  # Should be preserved for aspatial
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        enabled_affordances=[],  # Aspatial can't have positioned affordances
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Verify grid_size parameter is preserved (not overridden)
    assert env.grid_size == 12  # From parameter (aspatial has no width/height)
    assert env.substrate.position_dim == 0  # Aspatial has no position
    assert not hasattr(env.substrate, "width")


# =============================================================================
# ACTION SPACE BUILDER INTEGRATION (TASK-002B Phase 4)
# =============================================================================


def test_vectorized_env_loads_composed_action_space():
    """VectorizedHamletEnv should use ActionSpaceBuilder.

    TDD Test (RED phase): Verifies env has action_space attribute from builder.
    """
    from pathlib import Path

    import torch

    from townlet.environment.vectorized_env import VectorizedHamletEnv

    config_path = Path("configs/L1_full_observability")

    env = VectorizedHamletEnv(
        config_pack_path=config_path,
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

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


def test_action_masks_include_base_masking():
    """Action masks should start with base mask from ActionSpace.

    TDD Test (RED phase): Verifies get_action_masks() uses
    action_space.get_base_action_mask() for disabled action masking.

    This test verifies integration between get_action_masks() and ActionSpace,
    even though all actions are currently enabled (enabled_actions loading is future task).
    """
    from pathlib import Path
    from unittest.mock import patch

    import torch

    from townlet.environment.vectorized_env import VectorizedHamletEnv

    config_path = Path("configs/L1_full_observability")

    env = VectorizedHamletEnv(
        config_pack_path=config_path,
        num_agents=2,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

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
