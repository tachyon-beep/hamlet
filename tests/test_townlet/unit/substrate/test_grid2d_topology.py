"""Tests for Grid2D substrate topology attribute."""

from townlet.substrate.grid2d import Grid2DSubstrate


def test_grid2d_stores_topology_when_provided():
    """Grid2D should store topology attribute when explicitly provided."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="square",
    )
    assert substrate.topology == "square"


def test_grid2d_topology_defaults_to_square():
    """Grid2D topology should default to 'square' if not provided."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert substrate.topology == "square"


def test_grid2d_topology_attribute_exists():
    """Grid2D should have topology attribute (not inherited from base)."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(substrate, "topology")
