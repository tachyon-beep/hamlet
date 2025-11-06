"""Tests for SubstrateFactory support of GridND and ContinuousND substrates."""

import tempfile
from pathlib import Path

import pytest
import torch

from townlet.substrate.config import SubstrateConfig, load_substrate_config
from townlet.substrate.continuousnd import ContinuousNDSubstrate
from townlet.substrate.factory import SubstrateFactory
from townlet.substrate.gridnd import GridNDSubstrate


class TestGridNDFactoryFromConfig:
    """Test factory creation of GridND substrates from config."""

    def test_gridnd_4d_basic(self):
        """Test creating 4D GridND substrate from config."""
        config_dict = {
            "version": "1.0",
            "description": "4D grid for testing",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [8, 8, 8, 8],
                "boundary": "clamp",
                "distance_metric": "manhattan",
                "observation_encoding": "relative",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, GridNDSubstrate)
        assert substrate.dimension_sizes == [8, 8, 8, 8]
        assert substrate.boundary == "clamp"
        assert substrate.distance_metric == "manhattan"
        assert substrate.observation_encoding == "relative"
        assert substrate.position_dim == 4
        assert substrate.action_space_size == 2 * 4 + 2  # 10 actions

    def test_gridnd_5d_with_different_sizes(self):
        """Test creating 5D GridND with varying dimension sizes."""
        config_dict = {
            "version": "1.0",
            "description": "5D grid with different sizes",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [3, 4, 5, 6, 7],
                "boundary": "wrap",
                "distance_metric": "euclidean",
                "observation_encoding": "scaled",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, GridNDSubstrate)
        assert substrate.dimension_sizes == [3, 4, 5, 6, 7]
        assert substrate.boundary == "wrap"
        assert substrate.distance_metric == "euclidean"
        assert substrate.observation_encoding == "scaled"
        assert substrate.position_dim == 5
        assert substrate.action_space_size == 12  # 2*5 + 2

    def test_gridnd_10d_absolute_encoding(self):
        """Test 10D GridND with absolute observation encoding."""
        config_dict = {
            "version": "1.0",
            "description": "10D grid",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [4] * 10,
                "boundary": "bounce",
                "distance_metric": "chebyshev",
                "observation_encoding": "absolute",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, GridNDSubstrate)
        assert substrate.position_dim == 10
        assert len(substrate.dimension_sizes) == 10
        assert substrate.observation_encoding == "absolute"

    def test_gridnd_position_initialization(self):
        """Test that initialized GridND positions are valid."""
        config_dict = {
            "version": "1.0",
            "description": "4D grid test",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [5, 6, 7, 8],
                "boundary": "clamp",
                "distance_metric": "manhattan",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        positions = substrate.initialize_positions(num_agents=100, device=torch.device("cpu"))

        assert positions.shape == (100, 4)
        assert positions.dtype == torch.long
        # Check all positions are within bounds
        for dim_idx, size in enumerate(substrate.dimension_sizes):
            assert (positions[:, dim_idx] >= 0).all()
            assert (positions[:, dim_idx] < size).all()

    def test_gridnd_observation_encoding_relative(self):
        """Test relative observation encoding for GridND."""
        config_dict = {
            "version": "1.0",
            "description": "4D grid",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [10, 10, 10, 10],
                "boundary": "clamp",
                "observation_encoding": "relative",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        # Create test positions
        positions = torch.tensor([[0, 0, 0, 0], [9, 9, 9, 9]], dtype=torch.long)
        observations = substrate.encode_observation(positions, {})

        assert observations.shape == (2, 4)  # relative encoding: N dims
        # First position (0,0,0,0) should encode as all zeros
        assert (observations[0] == 0).all()
        # Last position (9,9,9,9) should encode as all ones (normalized)
        assert (observations[1] == 1).all()

    def test_gridnd_observation_encoding_scaled(self):
        """Test scaled observation encoding for GridND."""
        config_dict = {
            "version": "1.0",
            "description": "4D grid",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [5, 6, 7, 8],
                "boundary": "clamp",
                "observation_encoding": "scaled",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        positions = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        observations = substrate.encode_observation(positions, {})

        assert observations.shape == (1, 8)  # scaled encoding: 2N dims
        # First 4 dims should be coordinates (all zero)
        assert (observations[0, :4] == 0).all()
        # Next 4 dims should be dimension sizes
        assert observations[0, 4] == 5.0
        assert observations[0, 5] == 6.0
        assert observations[0, 6] == 7.0
        assert observations[0, 7] == 8.0


class TestContinuousNDFactoryFromConfig:
    """Test factory creation of ContinuousND substrates from config."""

    def test_continuousnd_4d_basic(self):
        """Test creating 4D ContinuousND substrate from config."""
        config_dict = {
            "version": "1.0",
            "description": "4D continuous for testing",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 4,
                "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
                "boundary": "clamp",
                "movement_delta": 0.5,
                "interaction_radius": 1.0,
                "distance_metric": "euclidean",
                "observation_encoding": "relative",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, ContinuousNDSubstrate)
        assert substrate.position_dim == 4
        assert substrate.bounds == [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
        assert substrate.boundary == "clamp"
        assert substrate.movement_delta == 0.5
        assert substrate.interaction_radius == 1.0
        assert substrate.distance_metric == "euclidean"
        assert substrate.observation_encoding == "relative"
        assert substrate.action_space_size == 10  # 2*4 + 2

    def test_continuousnd_6d_different_bounds(self):
        """Test 6D ContinuousND with different bounds per dimension."""
        config_dict = {
            "version": "1.0",
            "description": "6D continuous with varied bounds",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 6,
                "bounds": [
                    (0.0, 5.0),
                    (0.0, 10.0),
                    (-5.0, 5.0),
                    (0.0, 1.0),
                    (0.0, 100.0),
                    (-10.0, 10.0),
                ],
                "boundary": "wrap",
                "movement_delta": 0.1,
                "interaction_radius": 0.5,
                "distance_metric": "manhattan",
                "observation_encoding": "scaled",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, ContinuousNDSubstrate)
        assert substrate.position_dim == 6
        assert len(substrate.bounds) == 6
        assert substrate.bounds[0] == (0.0, 5.0)
        assert substrate.bounds[2] == (-5.0, 5.0)
        assert substrate.movement_delta == 0.1
        assert substrate.distance_metric == "manhattan"

    def test_continuousnd_position_initialization(self):
        """Test that initialized ContinuousND positions are valid."""
        config_dict = {
            "version": "1.0",
            "description": "4D continuous test",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 4,
                "bounds": [(0.0, 10.0), (0.0, 20.0), (-5.0, 5.0), (0.0, 1.0)],
                "boundary": "clamp",
                "movement_delta": 0.5,
                "interaction_radius": 1.0,
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        positions = substrate.initialize_positions(num_agents=50, device=torch.device("cpu"))

        assert positions.shape == (50, 4)
        assert positions.dtype == torch.float32
        # Check all positions are within bounds
        assert (positions[:, 0] >= 0.0).all() and (positions[:, 0] <= 10.0).all()
        assert (positions[:, 1] >= 0.0).all() and (positions[:, 1] <= 20.0).all()
        assert (positions[:, 2] >= -5.0).all() and (positions[:, 2] <= 5.0).all()
        assert (positions[:, 3] >= 0.0).all() and (positions[:, 3] <= 1.0).all()

    def test_continuousnd_observation_encoding_relative(self):
        """Test relative observation encoding for ContinuousND."""
        config_dict = {
            "version": "1.0",
            "description": "4D continuous",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 4,
                "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
                "boundary": "clamp",
                "movement_delta": 0.5,
                "interaction_radius": 1.0,
                "observation_encoding": "relative",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        # Test boundary positions
        positions = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]],
            dtype=torch.float32,
        )
        observations = substrate.encode_observation(positions, {})

        assert observations.shape == (2, 4)  # relative: N dims
        # First position (0,0,0,0) should be all zeros
        assert (observations[0] == 0).all()
        # Last position (10,10,10,10) should be all ones
        assert (observations[1] == 1).all()

    def test_continuousnd_observation_encoding_absolute(self):
        """Test absolute observation encoding for ContinuousND."""
        config_dict = {
            "version": "1.0",
            "description": "4D continuous",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 4,
                "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
                "boundary": "clamp",
                "movement_delta": 0.5,
                "interaction_radius": 1.0,
                "observation_encoding": "absolute",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        positions = torch.tensor(
            [[1.5, 2.5, 3.5, 4.5]],
            dtype=torch.float32,
        )
        observations = substrate.encode_observation(positions, {})

        assert observations.shape == (1, 4)  # absolute: N dims
        # Should match input positions exactly
        assert torch.allclose(observations[0], positions[0])


class TestFactoryConfigValidation:
    """Test that factory properly validates config structures."""

    def test_gridnd_missing_config_raises_error(self):
        """Test that missing gridnd config raises appropriate error."""
        config_dict = {
            "version": "1.0",
            "description": "Invalid - missing gridnd",
            "type": "gridnd",
        }
        with pytest.raises(ValueError, match="type='gridnd' requires gridnd configuration"):
            SubstrateConfig(**config_dict)

    def test_continuousnd_missing_config_raises_error(self):
        """Test that missing continuous config for continuousnd raises error."""
        config_dict = {
            "version": "1.0",
            "description": "Invalid - missing continuous",
            "type": "continuousnd",
        }
        with pytest.raises(ValueError, match="type='continuousnd' requires continuous configuration"):
            SubstrateConfig(**config_dict)

    def test_gridnd_dimension_validation(self):
        """Test that GridND config validates dimension count."""
        # Less than 4 dimensions should fail
        config_dict = {
            "version": "1.0",
            "description": "Invalid - too few dimensions",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [8, 8, 8],  # Only 3D
                "boundary": "clamp",
            },
        }
        with pytest.raises(ValueError, match="GridND requires at least 4 dimensions"):
            SubstrateConfig(**config_dict)

    def test_continuousnd_dimension_validation(self):
        """Test that ContinuousND config validates dimension count."""
        # Less than 4 dimensions should fail
        config_dict = {
            "version": "1.0",
            "description": "Invalid - too few dimensions",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 3,
                "bounds": [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
                "boundary": "clamp",
                "movement_delta": 0.5,
                "interaction_radius": 1.0,
            },
        }
        with pytest.raises(ValueError, match="type='continuousnd' expects 4\\+ dimensions"):
            SubstrateConfig(**config_dict)


class TestFactoryYAMLLoading:
    """Test factory loading substrates from YAML files."""

    def test_gridnd_from_yaml(self):
        """Test loading GridND substrate from YAML file."""
        yaml_content = """
version: "1.0"
description: "4D GridND substrate from YAML"
type: gridnd
gridnd:
  dimension_sizes: [8, 8, 8, 8]
  boundary: clamp
  distance_metric: manhattan
  observation_encoding: relative
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_substrate_config(config_path)
            substrate = SubstrateFactory.build(config, torch.device("cpu"))

            assert isinstance(substrate, GridNDSubstrate)
            assert substrate.dimension_sizes == [8, 8, 8, 8]
            assert substrate.boundary == "clamp"
        finally:
            config_path.unlink()

    def test_continuousnd_from_yaml(self):
        """Test loading ContinuousND substrate from YAML file."""
        yaml_content = """
version: "1.0"
description: "4D ContinuousND substrate from YAML"
type: continuousnd
continuous:
  dimensions: 4
  bounds: [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]
  boundary: wrap
  movement_delta: 0.5
  interaction_radius: 1.0
  distance_metric: euclidean
  observation_encoding: scaled
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            config = load_substrate_config(config_path)
            substrate = SubstrateFactory.build(config, torch.device("cpu"))

            assert isinstance(substrate, ContinuousNDSubstrate)
            assert substrate.position_dim == 4
            assert substrate.boundary == "wrap"
            assert substrate.observation_encoding == "scaled"
        finally:
            config_path.unlink()


class TestFactoryEdgeCases:
    """Test factory behavior with edge cases and boundary conditions."""

    def test_gridnd_max_dimensions(self):
        """Test GridND with maximum allowed dimensions (100D)."""
        config_dict = {
            "version": "1.0",
            "description": "100D grid",
            "type": "gridnd",
            "gridnd": {
                "dimension_sizes": [2] * 100,  # 100D grid
                "boundary": "clamp",
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert substrate.position_dim == 100
        assert substrate.action_space_size == 202  # 2*100 + 2

    def test_continuousnd_max_dimensions(self):
        """Test ContinuousND with maximum allowed dimensions (100D)."""
        bounds = [(0.0, 1.0)] * 100
        config_dict = {
            "version": "1.0",
            "description": "100D continuous",
            "type": "continuousnd",
            "continuous": {
                "dimensions": 100,
                "bounds": bounds,
                "boundary": "clamp",
                "movement_delta": 0.01,
                "interaction_radius": 0.1,
            },
        }
        config = SubstrateConfig(**config_dict)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert substrate.position_dim == 100

    def test_gridnd_boundary_modes(self):
        """Test GridND with all boundary modes."""
        for boundary_mode in ["clamp", "wrap", "bounce", "sticky"]:
            config_dict = {
                "version": "1.0",
                "description": f"GridND with {boundary_mode}",
                "type": "gridnd",
                "gridnd": {
                    "dimension_sizes": [8, 8, 8, 8],
                    "boundary": boundary_mode,
                },
            }
            config = SubstrateConfig(**config_dict)
            substrate = SubstrateFactory.build(config, torch.device("cpu"))
            assert substrate.boundary == boundary_mode

    def test_continuousnd_boundary_modes(self):
        """Test ContinuousND with all boundary modes."""
        for boundary_mode in ["clamp", "wrap", "bounce", "sticky"]:
            config_dict = {
                "version": "1.0",
                "description": f"ContinuousND with {boundary_mode}",
                "type": "continuousnd",
                "continuous": {
                    "dimensions": 4,
                    "bounds": [(0.0, 10.0)] * 4,
                    "boundary": boundary_mode,
                    "movement_delta": 0.5,
                    "interaction_radius": 1.0,
                },
            }
            config = SubstrateConfig(**config_dict)
            substrate = SubstrateFactory.build(config, torch.device("cpu"))
            assert substrate.boundary == boundary_mode

    def test_gridnd_distance_metrics(self):
        """Test GridND with all distance metrics."""
        for metric in ["manhattan", "euclidean", "chebyshev"]:
            config_dict = {
                "version": "1.0",
                "description": f"GridND with {metric} metric",
                "type": "gridnd",
                "gridnd": {
                    "dimension_sizes": [8, 8, 8, 8],
                    "boundary": "clamp",
                    "distance_metric": metric,
                },
            }
            config = SubstrateConfig(**config_dict)
            substrate = SubstrateFactory.build(config, torch.device("cpu"))
            assert substrate.distance_metric == metric

    def test_continuousnd_distance_metrics(self):
        """Test ContinuousND with all distance metrics."""
        for metric in ["euclidean", "manhattan", "chebyshev"]:
            config_dict = {
                "version": "1.0",
                "description": f"ContinuousND with {metric} metric",
                "type": "continuousnd",
                "continuous": {
                    "dimensions": 4,
                    "bounds": [(0.0, 10.0)] * 4,
                    "boundary": "clamp",
                    "movement_delta": 0.5,
                    "interaction_radius": 1.0,
                    "distance_metric": metric,
                },
            }
            config = SubstrateConfig(**config_dict)
            substrate = SubstrateFactory.build(config, torch.device("cpu"))
            assert substrate.distance_metric == metric


class TestFactoryIntegration:
    """Integration tests with multiple substrate types."""

    def test_factory_creates_correct_types(self):
        """Test that factory creates correct substrate types."""
        configs = [
            {
                "version": "1.0",
                "description": "GridND",
                "type": "gridnd",
                "gridnd": {
                    "dimension_sizes": [8, 8, 8, 8],
                    "boundary": "clamp",
                },
                "expected_type": GridNDSubstrate,
            },
            {
                "version": "1.0",
                "description": "ContinuousND",
                "type": "continuousnd",
                "continuous": {
                    "dimensions": 4,
                    "bounds": [(0.0, 10.0)] * 4,
                    "boundary": "clamp",
                    "movement_delta": 0.5,
                    "interaction_radius": 1.0,
                },
                "expected_type": ContinuousNDSubstrate,
            },
        ]

        for config_dict in configs:
            expected_type = config_dict.pop("expected_type")
            config = SubstrateConfig(**config_dict)
            substrate = SubstrateFactory.build(config, torch.device("cpu"))
            assert isinstance(substrate, expected_type)
