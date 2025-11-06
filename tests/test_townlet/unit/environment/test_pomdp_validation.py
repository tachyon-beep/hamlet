"""Tests for POMDP validation in VectorizedHamletEnv.

Tests environment validation logic for partial observability (POMDP) configurations
across different substrate types and dimensions.
"""

import shutil
from pathlib import Path

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestGridNDPOMDPValidation:
    """Test POMDP rejection for N≥4 dimensional grids."""

    @pytest.fixture
    def gridnd_4d_config_pack(self, tmp_path):
        """Create minimal 4D GridND config pack."""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        source_config = project_root / "configs" / "L1_3D_house"
        dest_config = tmp_path / "gridnd_4d_test"

        # Copy the L1_3D_house config as base
        shutil.copytree(source_config, dest_config)

        # Replace substrate.yaml with 4D GridND config
        substrate_yaml = dest_config / "substrate.yaml"
        substrate_yaml.write_text(
            """version: "1.0"
description: "4D hypercube grid for POMDP rejection test"
type: "gridnd"

gridnd:
  dimension_sizes: [5, 5, 5, 5]  # 4D hypercube
  boundary: "clamp"
  distance_metric: "manhattan"
  observation_encoding: "relative"
  topology: "hypercube"
"""
        )

        return str(dest_config)

    def test_gridnd_4d_pomdp_rejected(self, gridnd_4d_config_pack):
        """4D GridND should reject partial_observability=True."""
        with pytest.raises(ValueError, match=r"Partial observability.*is not supported for 4D substrates"):
            VectorizedHamletEnv(
                num_agents=1,
                grid_size=5,  # Ignored for GridND
                partial_observability=True,  # Should be rejected
                vision_range=2,
                enable_temporal_mechanics=False,
                move_energy_cost=0.005,
                wait_energy_cost=0.001,
                interact_energy_cost=0.003,
                agent_lifespan=1000,
                device=torch.device("cpu"),
                config_pack_path=gridnd_4d_config_pack,
            )


class TestGrid3DPOMDPValidation:
    """Test vision_range validation for Grid3D POMDP."""

    @pytest.fixture
    def grid3d_config_pack(self, tmp_path):
        """Create Grid3D config pack by copying L1_3D_house."""
        project_root = Path(__file__).parent.parent.parent.parent.parent
        source_config = project_root / "configs" / "L1_3D_house"
        dest_config = tmp_path / "grid3d_test"

        # Copy the entire config directory
        shutil.copytree(source_config, dest_config)

        return str(dest_config)

    def test_grid3d_pomdp_accepts_vision_range_2(self, grid3d_config_pack):
        """Grid3D POMDP should accept vision_range=2 (5×5×5 = 125 cells)."""
        # This should NOT raise an error
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=8,  # Ignored for Grid3D
            partial_observability=True,
            vision_range=2,  # Maximum allowed
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.003,
            agent_lifespan=1000,
            device=torch.device("cpu"),
            config_pack_path=grid3d_config_pack,
        )
        assert env.partial_observability is True
        assert env.vision_range == 2
        assert env.substrate.position_dim == 3

    def test_grid3d_pomdp_rejects_vision_range_3(self, grid3d_config_pack):
        """Grid3D POMDP should reject vision_range=3 (7×7×7 = 343 cells, too large)."""
        with pytest.raises(ValueError, match="Grid3D POMDP with vision_range=3 requires 343 cells"):
            VectorizedHamletEnv(
                num_agents=1,
                grid_size=8,
                partial_observability=True,
                vision_range=3,  # Too large
                enable_temporal_mechanics=False,
                move_energy_cost=0.005,
                wait_energy_cost=0.001,
                interact_energy_cost=0.003,
                agent_lifespan=1000,
                device=torch.device("cpu"),
                config_pack_path=grid3d_config_pack,
            )

    def test_grid3d_pomdp_rejects_vision_range_4(self, grid3d_config_pack):
        """Grid3D POMDP should reject vision_range=4 (9×9×9 = 729 cells, way too large)."""
        with pytest.raises(ValueError, match="Grid3D POMDP with vision_range=4 requires 729 cells"):
            VectorizedHamletEnv(
                num_agents=1,
                grid_size=8,
                partial_observability=True,
                vision_range=4,  # Way too large
                enable_temporal_mechanics=False,
                move_energy_cost=0.005,
                wait_energy_cost=0.001,
                interact_energy_cost=0.003,
                agent_lifespan=1000,
                device=torch.device("cpu"),
                config_pack_path=grid3d_config_pack,
            )
