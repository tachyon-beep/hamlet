"""Tests for POMDP validation in VectorizedHamletEnv.

Tests environment validation logic for partial observability (POMDP) configurations
across different substrate types and dimensions.
"""

import shutil
from pathlib import Path

import pytest
import torch
import yaml

from tests.test_townlet.utils.builders import make_vectorized_env_from_pack


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

        return dest_config

    def test_gridnd_4d_pomdp_rejected(self, gridnd_4d_config_pack):
        """4D GridND should reject partial_observability=True."""
        training_path = gridnd_4d_config_pack / "training.yaml"
        training_cfg = yaml.safe_load(training_path.read_text())
        training_cfg["environment"]["partial_observability"] = True
        training_cfg["environment"]["vision_range"] = 2
        training_path.write_text(yaml.safe_dump(training_cfg, sort_keys=False))

        with pytest.raises(ValueError, match=r"Partial observability.*is not supported for 4D substrates"):
            make_vectorized_env_from_pack(
                gridnd_4d_config_pack,
                num_agents=1,
                device=torch.device("cpu"),
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

        return dest_config

    def test_grid3d_pomdp_accepts_vision_range_2(self, grid3d_config_pack):
        """Grid3D POMDP should accept vision_range=2 (5×5×5 = 125 cells)."""
        training_path = grid3d_config_pack / "training.yaml"
        training_cfg = yaml.safe_load(training_path.read_text())
        training_cfg["environment"]["partial_observability"] = True
        training_cfg["environment"]["vision_range"] = 2
        training_path.write_text(yaml.safe_dump(training_cfg, sort_keys=False))

        env = make_vectorized_env_from_pack(
            grid3d_config_pack,
            num_agents=1,
            device=torch.device("cpu"),
        )
        assert env.partial_observability is True
        assert env.vision_range == 2
        assert env.substrate.position_dim == 3

    def test_grid3d_pomdp_rejects_vision_range_3(self, grid3d_config_pack):
        """Grid3D POMDP should reject vision_range=3 (7×7×7 = 343 cells, too large)."""
        training_path = grid3d_config_pack / "training.yaml"
        training_cfg = yaml.safe_load(training_path.read_text())
        training_cfg["environment"]["partial_observability"] = True
        training_cfg["environment"]["vision_range"] = 3
        training_path.write_text(yaml.safe_dump(training_cfg, sort_keys=False))

        with pytest.raises(ValueError, match="Grid3D POMDP with vision_range=3 requires 343 cells"):
            make_vectorized_env_from_pack(
                grid3d_config_pack,
                num_agents=1,
                device=torch.device("cpu"),
            )

    def test_grid3d_pomdp_rejects_vision_range_4(self, grid3d_config_pack):
        """Grid3D POMDP should reject vision_range=4 (9×9×9 = 729 cells, way too large)."""
        training_path = grid3d_config_pack / "training.yaml"
        training_cfg = yaml.safe_load(training_path.read_text())
        training_cfg["environment"]["partial_observability"] = True
        training_cfg["environment"]["vision_range"] = 4
        training_path.write_text(yaml.safe_dump(training_cfg, sort_keys=False))

        with pytest.raises(ValueError, match="Grid3D POMDP with vision_range=4 requires 729 cells"):
            make_vectorized_env_from_pack(
                grid3d_config_pack,
                num_agents=1,
                device=torch.device("cpu"),
            )
