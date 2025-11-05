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
