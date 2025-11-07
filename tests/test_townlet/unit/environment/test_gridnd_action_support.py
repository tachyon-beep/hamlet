"""Tests for N-dimensional substrate support in VectorizedHamletEnv.

These tests guard against regressions when enabling GridND/ContinuousND
substrates with position_dim >= 4. They focus on action execution and masking,
which are critical for the new high-dimensional environments.
"""

from __future__ import annotations

import itertools
import shutil
from pathlib import Path

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def gridnd_4d_config_pack(tmp_path: Path) -> Path:
    """Create a temporary config pack backed by a 4D GridND substrate."""
    project_root = Path(__file__).parent.parent.parent.parent.parent
    source_config = project_root / "configs" / "L1_3D_house"
    dest_config = tmp_path / "gridnd_4d_support"

    shutil.copytree(source_config, dest_config)

    substrate_yaml = dest_config / "substrate.yaml"
    substrate_yaml.write_text(
        """version: "1.0"
description: "4D hypercube grid for action support tests"
type: "gridnd"

gridnd:
  dimension_sizes: [5, 5, 5, 5]
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
  topology: "hypercube"
"""
    )

    # Update VFS config to match 4D substrate (L1_3D_house has 3D position)
    import yaml

    vfs_yaml = dest_config / "variables_reference.yaml"
    with open(vfs_yaml) as f:
        vfs_config = yaml.safe_load(f)

    # Find and update position variable from 3D to 4D
    for var in vfs_config["variables"]:
        if var["id"] == "position":
            var["dims"] = 4
            var["default"] = [0.0, 0.0, 0.0, 0.0]
            var["description"] = "Normalized agent position (4D) in [0, 1]^4 range"
            break

    # Update normalization spec for position observation (must match dims)
    for obs in vfs_config.get("exposed_observations", []):
        if obs["id"] == "obs_position":
            # Update shape to match 4D
            obs["shape"] = [4]
            # Update normalization min/max to 4D arrays (were 3D)
            if obs.get("normalization"):
                obs["normalization"]["min"] = [0.0, 0.0, 0.0, 0.0]
                obs["normalization"]["max"] = [1.0, 1.0, 1.0, 1.0]
            break

    with open(vfs_yaml, "w") as f:
        yaml.safe_dump(vfs_config, f, sort_keys=False)

    return dest_config


@pytest.fixture
def gridnd_env(gridnd_4d_config_pack: Path, cpu_device: torch.device) -> VectorizedHamletEnv:
    """Instantiate a 4D GridND environment for action support tests."""
    return VectorizedHamletEnv(
        num_agents=1,
        grid_size=5,  # Ignored by GridND substrates
        partial_observability=False,
        vision_range=1,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.003,
        agent_lifespan=100,
        device=cpu_device,
        config_pack_path=gridnd_4d_config_pack,
    )


class TestGridNDActions:
    """Validate VectorizedHamletEnv behavior for N-dimensional substrates."""

    def test_interact_mask_uses_nd_index(self, gridnd_env: VectorizedHamletEnv) -> None:
        """INTERACT mask should live at index 2 * position_dim for GridND."""
        env = gridnd_env
        env.reset()

        interact_idx = 2 * env.substrate.position_dim

        # Find a position that is guaranteed to be off all affordances.
        occupied_positions = {tuple(pos.tolist()) for pos in env.affordances.values()}
        candidate_position_tensor = None
        for candidate in itertools.product(*(range(size) for size in env.substrate.dimension_sizes)):
            if candidate not in occupied_positions:
                candidate_position_tensor = torch.tensor(candidate, dtype=env.positions.dtype, device=env.device)
                break

        assert candidate_position_tensor is not None, "Expected at least one free cell in GridND test config"

        env.positions[0] = candidate_position_tensor
        action_masks = env.get_action_masks()

        assert not action_masks[0, interact_idx], "INTERACT should be masked off-affordance at index 2 * position_dim"

    def test_execute_actions_supports_high_dim_movements(self, gridnd_env: VectorizedHamletEnv) -> None:
        """_execute_actions should handle ± movements for every GridND axis."""
        env = gridnd_env
        env.reset()

        dim_sizes = env.substrate.dimension_sizes
        last_dim = env.substrate.position_dim - 1

        start_position = torch.tensor(
            [size // 2 for size in dim_sizes],
            dtype=env.positions.dtype,
            device=env.device,
        )
        env.positions[0] = start_position.clone()

        positive_action_index = env.substrate.position_dim + last_dim  # D{last_dim}_POS
        actions = torch.tensor([positive_action_index], device=env.device)

        env.step(actions)

        expected_position = start_position.clone()
        expected_position[last_dim] = min(dim_sizes[last_dim] - 1, start_position[last_dim].item() + 1)

        assert torch.equal(env.positions[0], expected_position), "High-dimensional movement should update the targeted axis"
