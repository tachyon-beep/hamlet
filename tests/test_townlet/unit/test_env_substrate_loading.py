"""Test environment loads and uses substrate configuration."""

from pathlib import Path

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.grid2d import Grid2DSubstrate


def test_env_loads_substrate_config():
    """Environment should load substrate.yaml and create substrate instance."""
    # Note: This test will initially PASS with legacy behavior
    # After Phase 4 integration, it will load from substrate.yaml

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

    # After integration, should have substrate attribute
    # For now, check legacy grid_size exists
    assert hasattr(env, "grid_size")
    assert env.grid_size == 8


def test_env_substrate_accessible():
    """Environment should expose substrate for inspection."""
    # This test will FAIL initially, becomes valid after integration

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

    # After integration:
    assert hasattr(env, "substrate")
    assert isinstance(env.substrate, Grid2DSubstrate)
    assert env.substrate.width == 8


def test_missing_substrate_yaml_raises_helpful_error(tmp_path):
    """Should fail fast with migration instructions when substrate.yaml missing."""
    import shutil

    # Create config pack directory without substrate.yaml
    config_pack = tmp_path / "test_config"
    config_pack.mkdir()

    # Copy complete config files from test config (but NO substrate.yaml)
    test_config = Path("configs/test")
    shutil.copy(test_config / "bars.yaml", config_pack / "bars.yaml")
    shutil.copy(test_config / "affordances.yaml", config_pack / "affordances.yaml")
    shutil.copy(test_config / "cascades.yaml", config_pack / "cascades.yaml")

    # Attempt to create environment without substrate.yaml
    with pytest.raises(FileNotFoundError) as exc_info:
        VectorizedHamletEnv(
            config_pack_path=config_pack,
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

    # Verify error message contains migration instructions
    error_msg = str(exc_info.value)
    assert "substrate.yaml is required" in error_msg
    assert "Quick fix:" in error_msg
    assert "configs/templates/substrate.yaml" in error_msg
    assert "TASK-002A" in error_msg


def test_non_square_grid_rejected(tmp_path):
    """Should reject non-square grids with clear error message."""
    import shutil

    # Create config pack with non-square substrate
    config_pack = tmp_path / "test_config"
    config_pack.mkdir()

    # Create substrate.yaml with non-square grid (8×10)
    substrate_yaml = config_pack / "substrate.yaml"
    substrate_yaml.write_text(
        """
version: "1.0"
description: "Non-square grid test"
type: "grid"

grid:
  topology: "square"
  width: 8
  height: 10
  boundary: "clamp"
  distance_metric: "manhattan"
"""
    )

    # Copy complete config files from test config
    test_config = Path("configs/test")
    shutil.copy(test_config / "bars.yaml", config_pack / "bars.yaml")
    shutil.copy(test_config / "affordances.yaml", config_pack / "affordances.yaml")
    shutil.copy(test_config / "cascades.yaml", config_pack / "cascades.yaml")

    # Attempt to create environment with non-square grid
    with pytest.raises(ValueError) as exc_info:
        VectorizedHamletEnv(
            config_pack_path=config_pack,
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

    # Verify error message is clear
    error_msg = str(exc_info.value)
    assert "Non-square grids not yet supported" in error_msg
    assert "8×10" in error_msg or "8x10" in error_msg


def test_grid_size_overridden_by_substrate():
    """grid_size parameter should be overridden by substrate dimensions."""
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
    """Aspatial substrate should keep grid_size parameter value."""
    import shutil

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

    # Create environment with aspatial substrate and grid_size parameter
    env = VectorizedHamletEnv(
        config_pack_path=config_pack,
        num_agents=1,
        grid_size=12,  # Should be preserved for aspatial
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
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


def test_substrate_initialize_positions_correctness():
    """Grid2D.initialize_positions() should return valid grid positions."""
    from townlet.substrate.grid2d import Grid2DSubstrate

    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    # Correct shape and type
    assert positions.shape == (10, 2)
    assert positions.dtype == torch.long

    # Within bounds
    assert (positions >= 0).all()
    assert (positions < 8).all()


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

    # Try to move up (UP action decreases Y, but we're at boundary)
    # UP action should be action 0 based on action_dim
    action = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))

    env.step(action)

    # Position should be clamped by substrate (boundary="clamp" in configs)
    # Since we're at [0, 0] and trying to move up, should stay at [0, 0]
    assert (env.positions[:, 0] >= 0).all()  # X within bounds
    assert (env.positions[:, 1] >= 0).all()  # Y within bounds


def test_substrate_movement_matches_legacy():
    """Substrate movement should produce identical results to legacy torch.clamp."""
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

    # Test substrate.apply_movement directly
    substrate = env.substrate
    positions = torch.tensor([[3, 3]], dtype=torch.long, device=torch.device("cpu"))

    # Move up (delta [0, -1])
    deltas = torch.tensor([[0, -1]], dtype=torch.long, device=torch.device("cpu"))
    new_positions = substrate.apply_movement(positions, deltas)

    # Should move to [3, 2]
    assert (new_positions == torch.tensor([[3, 2]], dtype=torch.long)).all()

    # Test boundary clamping at edge
    edge_positions = torch.tensor([[0, 0]], dtype=torch.long, device=torch.device("cpu"))
    up_left_delta = torch.tensor([[-1, -1]], dtype=torch.long, device=torch.device("cpu"))
    clamped = substrate.apply_movement(edge_positions, up_left_delta)

    # Should clamp to [0, 0] (not go negative)
    assert (clamped == torch.tensor([[0, 0]], dtype=torch.long)).all()


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
