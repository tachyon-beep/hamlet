"""Test GridND substrate for N-dimensional grids (N≥4)."""

import pytest
import torch

from townlet.substrate.gridnd import GridNDSubstrate


def test_gridnd_4d_initialization():
    """GridND should initialize correctly for 4D hypercube."""
    substrate = GridNDSubstrate(
        dimension_sizes=[8, 8, 8, 8],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    assert substrate.position_dim == 4
    assert substrate.position_dtype == torch.long
    assert substrate.action_space_size == 9  # 2*4 + 1
    assert substrate.get_observation_dim() == 4  # Normalized coordinates


def test_gridnd_requires_minimum_4_dimensions():
    """GridND should reject dimensions < 4."""
    with pytest.raises(ValueError, match="GridND requires at least 4 dimensions"):
        GridNDSubstrate(
            dimension_sizes=[8, 8, 8],  # Only 3D
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )


def test_gridnd_validates_positive_dimensions():
    """GridND should reject non-positive dimension sizes."""
    with pytest.raises(ValueError, match="Dimension sizes must be positive"):
        GridNDSubstrate(
            dimension_sizes=[8, 8, 0, 8],  # Zero size
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )


def test_gridnd_asymmetric_dimensions():
    """GridND should support different sizes per dimension."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 5, 3, 7],  # Different sizes
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    assert substrate.position_dim == 4
    assert substrate.dimension_sizes == [10, 5, 3, 7]


def test_gridnd_warns_at_10_dimensions():
    """GridND should emit warning at N≥10."""
    with pytest.warns(UserWarning, match="10 dimensions"):
        substrate = GridNDSubstrate(
            dimension_sizes=[3] * 10,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

    assert substrate.action_space_size == 21  # 2*10 + 1


def test_gridnd_movement_with_clamp_boundary():
    """Test movement with clamp boundary (hard walls)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Agent at corner [0, 0, 0, 0]
    positions = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)

    # Try to move negative (should be clamped)
    deltas = torch.tensor([[-1, -1, -1, -1]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Should stay at [0, 0, 0, 0]
    assert torch.equal(new_positions, torch.tensor([[0, 0, 0, 0]], dtype=torch.long))


def test_gridnd_movement_with_wrap_boundary():
    """Test movement with wrap boundary (toroidal)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="wrap",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Agent at [0, 0, 0, 0]
    positions = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)

    # Move negative (should wrap to [4, 4, 4, 4])
    deltas = torch.tensor([[-1, -1, -1, -1]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    assert torch.equal(new_positions, torch.tensor([[4, 4, 4, 4]], dtype=torch.long))


def test_gridnd_distance_manhattan():
    """Test Manhattan distance in 4D."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 10, 10, 10],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    pos1 = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    pos2 = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)

    distance = substrate.compute_distance(pos1, pos2)

    # Manhattan: |3-0| + |4-0| + |5-0| + |6-0| = 18
    assert distance[0] == 18


def test_gridnd_distance_euclidean():
    """Test Euclidean distance in 4D."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 10, 10, 10],
        boundary="clamp",
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    pos1 = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
    pos2 = torch.tensor([[3, 4, 0, 0]], dtype=torch.long)

    distance = substrate.compute_distance(pos1, pos2)

    # Euclidean: sqrt(3^2 + 4^2 + 0^2 + 0^2) = sqrt(25) = 5.0
    assert torch.allclose(distance, torch.tensor([5.0]))


def test_gridnd_observation_encoding_relative():
    """Test relative encoding (normalized [0,1])."""
    substrate = GridNDSubstrate(
        dimension_sizes=[8, 8, 8, 8],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Corner: [0, 0, 0, 0] → [0.0, 0.0, 0.0, 0.0]
    # Opposite corner: [7, 7, 7, 7] → [1.0, 1.0, 1.0, 1.0]
    positions = torch.tensor([[0, 0, 0, 0], [7, 7, 7, 7]], dtype=torch.long)

    encoded = substrate.encode_observation(positions, {})

    assert encoded.shape == (2, 4)
    assert torch.allclose(encoded[0], torch.zeros(4))
    assert torch.allclose(encoded[1], torch.ones(4))
    assert substrate.get_observation_dim() == 4


def test_gridnd_observation_encoding_scaled():
    """Test scaled encoding (normalized + sizes)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 5, 3, 7],  # Different sizes
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="scaled",
    )

    positions = torch.tensor([[5, 2, 1, 3]], dtype=torch.long)

    encoded = substrate.encode_observation(positions, {})

    # First N dims: normalized positions
    # Last N dims: dimension sizes
    assert encoded.shape == (1, 8)  # 2 * 4 dimensions

    # Verify sizes in last N dims
    assert torch.allclose(encoded[0, 4:], torch.tensor([10.0, 5.0, 3.0, 7.0]))
    assert substrate.get_observation_dim() == 8


def test_gridnd_neighbors_4d_interior():
    """Test interior position has 8 neighbors (2*4) in 4D."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 10, 10, 10],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Interior position
    position = torch.tensor([5, 5, 5, 5], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)

    # Should have 8 neighbors (2*4 dimensions)
    assert len(neighbors) == 8

    # Verify each neighbor is ±1 in exactly one dimension
    for neighbor in neighbors:
        diff = torch.abs(neighbor - position).sum()
        assert diff == 1, "Neighbor should differ by 1 in exactly one dimension"


def test_gridnd_neighbors_4d_corner():
    """Test corner position has fewer neighbors (clamp boundary)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Corner position [0, 0, 0, 0]
    position = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)

    # Should have 4 neighbors (only positive directions)
    assert len(neighbors) == 4

    # All neighbors should be [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]
    for neighbor in neighbors:
        assert (neighbor >= 0).all()
        assert neighbor.sum() == 1  # Exactly one dimension = 1


def test_gridnd_is_on_position():
    """Test exact position matching in N dimensions."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10, 10, 10, 10],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    agent_positions = torch.tensor(
        [
            [5, 5, 5, 5],  # On target
            [5, 5, 5, 6],  # Off by 1 in one dimension
            [0, 0, 0, 0],  # Far from target
        ],
        dtype=torch.long,
    )

    target_position = torch.tensor([5, 5, 5, 5], dtype=torch.long)

    on_position = substrate.is_on_position(agent_positions, target_position)

    assert on_position[0]
    assert not on_position[1]
    assert not on_position[2]
