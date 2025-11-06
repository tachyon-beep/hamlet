"""Tests for SubstrateFactory."""

import torch

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.config import AspatialSubstrateConfig, ContinuousConfig, GridConfig, GridNDConfig, SubstrateConfig
from townlet.substrate.continuous import Continuous2DSubstrate
from townlet.substrate.factory import SubstrateFactory
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate
from townlet.substrate.gridnd import GridNDSubstrate


def test_factory_propagates_grid2d_topology():
    """Factory should pass topology from config to Grid2D substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test Grid2D config",
        type="grid",
        grid=GridConfig(
            topology="square",
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, Grid2DSubstrate)
    assert substrate.topology == "square"


def test_factory_propagates_grid3d_topology():
    """Factory should pass topology from config to Grid3D substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test Grid3D config",
        type="grid",
        grid=GridConfig(
            topology="cubic",
            width=8,
            height=8,
            depth=3,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, Grid3DSubstrate)
    assert substrate.topology == "cubic"


def test_factory_propagates_gridnd_topology():
    """Factory should pass topology from config to GridND substrate."""
    config = SubstrateConfig(
        version="1.0",
        description="Test GridND config",
        type="gridnd",
        gridnd=GridNDConfig(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
            topology="hypercube",
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, GridNDSubstrate)
    assert substrate.topology == "hypercube"


def test_factory_sets_gridnd_topology_when_config_uses_default():
    """Factory should use GridND topology default when not specified in config."""
    config = SubstrateConfig(
        version="1.0",
        description="Test GridND config with default topology",
        type="gridnd",
        gridnd=GridNDConfig(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
            # topology not specified, uses default
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, GridNDSubstrate)
    assert substrate.topology == "hypercube"


def test_factory_continuous_substrates_have_no_topology():
    """Factory should create continuous substrates without topology attribute."""
    config = SubstrateConfig(
        version="1.0",
        description="Test Continuous2D config",
        type="continuous",
        continuous=ContinuousConfig(
            dimensions=2,
            bounds=[(0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
            distance_metric="euclidean",
            observation_encoding="relative",
        ),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, Continuous2DSubstrate)
    assert not hasattr(substrate, "topology")


def test_factory_aspatial_substrate_has_no_topology():
    """Factory should create aspatial substrate without topology attribute."""
    config = SubstrateConfig(
        version="1.0",
        description="Test Aspatial config",
        type="aspatial",
        aspatial=AspatialSubstrateConfig(),
    )
    substrate = SubstrateFactory.build(config, torch.device("cpu"))
    assert isinstance(substrate, AspatialSubstrate)
    assert not hasattr(substrate, "topology")
