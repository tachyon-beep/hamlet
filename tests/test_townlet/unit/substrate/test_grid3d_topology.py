"""Tests for Grid3D substrate topology attribute."""

from townlet.substrate.grid3d import Grid3DSubstrate


def test_grid3d_stores_topology_when_provided():
    """Grid3D should store topology attribute when explicitly provided."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="cubic",
    )
    assert substrate.topology == "cubic"


def test_grid3d_topology_defaults_to_cubic():
    """Grid3D topology should default to 'cubic' if not provided."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert substrate.topology == "cubic"


def test_grid3d_topology_attribute_exists():
    """Grid3D should have topology attribute."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(substrate, "topology")
