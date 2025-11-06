"""Test edge cases and warnings for GridND."""

import pytest
import torch

from townlet.substrate.gridnd import GridNDSubstrate


def test_gridnd_get_all_positions_small_grid():
    """Test get_all_positions() on small 4D grid."""
    substrate = GridNDSubstrate(
        dimension_sizes=[2, 2, 2, 2],
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    positions = substrate.get_all_positions()

    # 2^4 = 16 positions
    assert len(positions) == 16

    # Verify all positions are unique
    assert len(set(tuple(p) for p in positions)) == 16


def test_gridnd_get_all_positions_moderate_grid():
    """Test get_all_positions() on moderate 4D grid."""
    substrate = GridNDSubstrate(
        dimension_sizes=[10] * 4,  # 10^4 = 10,000 positions
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Should NOT warn (10K < 100K threshold)
    positions = substrate.get_all_positions()
    assert len(positions) == 10_000


def test_gridnd_get_all_positions_raises_on_huge_grid():
    """Test get_all_positions() raises on absurdly large grids."""
    substrate = GridNDSubstrate(
        dimension_sizes=[100] * 4,  # 100^4 = 100M positions
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    with pytest.raises(MemoryError, match="too large for memory"):
        substrate.get_all_positions()


def test_gridnd_rejects_dimension_count_above_100():
    """Test GridND rejects dimension count > 100."""
    with pytest.raises(ValueError, match="exceeds limit"):
        GridNDSubstrate(
            dimension_sizes=[3] * 101,  # 101 dimensions
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )


def test_gridnd_asymmetric_bounds():
    """Test GridND with very different dimension sizes."""
    substrate = GridNDSubstrate(
        dimension_sizes=[100, 3, 50, 7],  # Very asymmetric
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Test positions initialize within bounds
    positions = substrate.initialize_positions(10, torch.device("cpu"))
    assert (positions[:, 0] >= 0).all() and (positions[:, 0] < 100).all()
    assert (positions[:, 1] >= 0).all() and (positions[:, 1] < 3).all()
    assert (positions[:, 2] >= 0).all() and (positions[:, 2] < 50).all()
    assert (positions[:, 3] >= 0).all() and (positions[:, 3] < 7).all()


def test_gridnd_bounce_boundary_large_delta():
    """Test bounce boundary with large movement delta."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="bounce",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Agent at center [2, 2, 2, 2]
    positions = torch.tensor([[2, 2, 2, 2]], dtype=torch.long)

    # Huge delta (should still be bounded after bounce)
    deltas = torch.tensor([[10, -10, 8, -8]], dtype=torch.float32)

    new_positions = substrate.apply_movement(positions, deltas)

    # Should be within bounds [0, 4]
    assert (new_positions >= 0).all()
    assert (new_positions < 5).all()


def test_gridnd_sticky_boundary():
    """Test sticky boundary (stay in place if out of bounds)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[5, 5, 5, 5],
        boundary="sticky",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Agent at [2, 0, 2, 4]
    positions = torch.tensor([[2, 0, 2, 4]], dtype=torch.long)

    # Try to move out of bounds in dims 1 and 3
    deltas = torch.tensor([[0, -1, 0, 1]], dtype=torch.float32)

    new_positions = substrate.apply_movement(positions, deltas)

    # Dims 1 and 3 should stay (sticky), dims 0 and 2 unchanged
    assert torch.equal(new_positions, torch.tensor([[2, 0, 2, 4]], dtype=torch.long))
