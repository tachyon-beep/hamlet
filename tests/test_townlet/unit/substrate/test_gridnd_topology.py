"""Tests for GridND substrate topology attribute."""

from townlet.substrate.gridnd import GridNDSubstrate


def test_gridnd_stores_topology_when_provided():
    """GridND should store topology attribute when explicitly provided."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="hypercube",
    )
    assert substrate.topology == "hypercube"


def test_gridnd_topology_defaults_to_hypercube():
    """GridND topology should default to 'hypercube' if not provided."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert substrate.topology == "hypercube"


def test_gridnd_topology_attribute_exists():
    """GridND should have topology attribute."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(substrate, "topology")


def test_gridnd_topology_is_hypercube_for_all_dimensions():
    """GridND topology should be 'hypercube' regardless of dimensionality."""
    for num_dims in [4, 5, 7, 10]:
        substrate = GridNDSubstrate(
            dimension_sizes=[5] * num_dims,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )
        assert substrate.topology == "hypercube"
