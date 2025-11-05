"""Test substrate configuration schema."""

import pytest
import torch

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.config import (
    AspatialSubstrateConfig,
    Grid2DSubstrateConfig,
    SubstrateConfig,
)
from townlet.substrate.factory import SubstrateFactory
from townlet.substrate.grid2d import Grid2DSubstrate


def test_grid2d_config_valid():
    """Valid Grid2D config should parse successfully."""
    config_data = {
        "topology": "square",
        "width": 8,
        "height": 8,
        "boundary": "clamp",
        "distance_metric": "manhattan",
    }

    config = Grid2DSubstrateConfig(**config_data)

    assert config.width == 8
    assert config.height == 8
    assert config.boundary == "clamp"


def test_grid2d_config_invalid_dimensions():
    """Grid2D config with invalid dimensions should fail."""
    config_data = {
        "topology": "square",
        "width": 0,  # Invalid!
        "height": 8,
        "boundary": "clamp",
        "distance_metric": "manhattan",
    }

    with pytest.raises(ValueError, match="greater than 0"):
        Grid2DSubstrateConfig(**config_data)


def test_aspatial_config_valid():
    """Valid aspatial config should parse successfully."""
    config_data = {}  # No fields needed for aspatial

    config = AspatialSubstrateConfig(**config_data)

    # Just verify it parses successfully (no fields to check)
    assert config is not None


def test_substrate_config_grid2d():
    """SubstrateConfig with type='grid' should require grid config."""
    config_data = {
        "version": "1.0",
        "description": "Test grid substrate",
        "type": "grid",
        "grid": {
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
        },
    }

    config = SubstrateConfig(**config_data)

    assert config.type == "grid"
    assert config.grid is not None
    assert config.grid.width == 8


def test_substrate_config_missing_grid():
    """SubstrateConfig with type='grid' but missing grid config should fail."""
    config_data = {
        "version": "1.0",
        "description": "Test",
        "type": "grid",
        # Missing grid config!
    }

    with pytest.raises(ValueError, match="grid configuration"):
        SubstrateConfig(**config_data)


def test_factory_build_grid2d():
    """Factory should build Grid2DSubstrate from config."""
    config_data = {
        "version": "1.0",
        "description": "Test grid",
        "type": "grid",
        "grid": {
            "topology": "square",
            "width": 8,
            "height": 8,
            "boundary": "clamp",
            "distance_metric": "manhattan",
        },
    }

    config = SubstrateConfig(**config_data)
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))

    assert isinstance(substrate, Grid2DSubstrate)
    assert substrate.width == 8
    assert substrate.height == 8


def test_factory_build_aspatial():
    """Factory should build AspatialSubstrate from config."""
    config_data = {
        "version": "1.0",
        "description": "Test aspatial",
        "type": "aspatial",
        "aspatial": {},  # Empty dict (no fields needed)
    }

    config = SubstrateConfig(**config_data)
    substrate = SubstrateFactory.build(config, device=torch.device("cpu"))

    assert isinstance(substrate, AspatialSubstrate)
    assert substrate.position_dim == 0


# Phase 5C: observation_encoding tests
def test_grid_config_observation_encoding_valid():
    """Test Grid config accepts valid observation_encoding values."""
    from townlet.substrate.config import GridConfig

    for encoding in ["relative", "scaled", "absolute"]:
        config = GridConfig(
            topology="square",
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding=encoding,
        )
        assert config.observation_encoding == encoding


def test_grid_config_observation_encoding_default():
    """Test Grid config defaults to relative for backward compatibility."""
    from townlet.substrate.config import GridConfig

    config = GridConfig(
        topology="square",
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        # observation_encoding NOT provided
    )
    assert config.observation_encoding == "relative"


def test_grid_config_observation_encoding_invalid():
    """Test Grid config rejects invalid observation_encoding."""
    from townlet.substrate.config import GridConfig

    with pytest.raises(ValueError, match="observation_encoding"):
        GridConfig(
            topology="square",
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="invalid",  # Not in Literal
        )


def test_continuous_config_observation_encoding_valid():
    """Test Continuous config accepts valid observation_encoding values."""
    from townlet.substrate.config import ContinuousConfig

    for encoding in ["relative", "scaled", "absolute"]:
        config = ContinuousConfig(
            dimensions=2,
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding=encoding,
        )
        assert config.observation_encoding == encoding


def test_continuous_config_observation_encoding_default():
    """Test Continuous config defaults to relative for backward compatibility."""
    from townlet.substrate.config import ContinuousConfig

    config = ContinuousConfig(
        dimensions=2,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        # observation_encoding NOT provided
    )
    assert config.observation_encoding == "relative"
