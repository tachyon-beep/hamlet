"""Integration tests for substrate.yaml migration (environment loading).

NOTE: These tests will FAIL until Phase 4 (Environment Integration) is complete.
They document expected behavior after VectorizedEnv is updated to load substrate.yaml.
"""

from pathlib import Path

import pytest
import torch
import yaml

from townlet.environment.vectorized_env import VectorizedHamletEnv

# Phase 4 complete - integration tests enabled
# pytestmark = pytest.mark.skip(reason="Phase 4 (Environment Integration) not yet complete")


def load_enabled_affordances(config_pack_path: Path) -> list[str]:
    """Load enabled_affordances from training.yaml."""
    training_yaml = config_pack_path / "training.yaml"
    with open(training_yaml) as f:
        config = yaml.safe_load(f)
    return config["environment"]["enabled_affordances"]


@pytest.mark.parametrize(
    "config_name,expected_obs_dim",
    [
        ("L0_0_minimal", 29),  # 2 coords + 8 meters + 15 affordances + 4 temporal
        ("L0_5_dual_resource", 29),  # 2 coords + 8 meters + 15 affordances + 4 temporal
        ("L1_full_observability", 29),  # 2 coords + 8 + 15 + 4
        ("L2_partial_observability", 29),  # Same as L1 (full obs_dim, not local window)
        ("L3_temporal_mechanics", 29),  # Same as L1
    ],
)
def test_env_observation_dim_unchanged(config_name, expected_obs_dim):
    """Environment with substrate.yaml using coordinate encoding (all same dims)."""
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

    # Observation dimension should match legacy hardcoded behavior
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
