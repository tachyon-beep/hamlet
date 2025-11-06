"""Property-based tests for N-dimensional substrates.

These tests verify universal invariants that must hold for ALL N.
"""

import pytest
import torch

from townlet.substrate.continuousnd import ContinuousNDSubstrate
from townlet.substrate.gridnd import GridNDSubstrate


@pytest.mark.parametrize(
    "dimensions,size,metric",
    [
        (4, 3, "manhattan"),
        (4, 3, "euclidean"),
        (4, 3, "chebyshev"),
        (7, 3, "manhattan"),
    ],
)
def test_distance_symmetric_gridnd(dimensions, size, metric):
    """PROPERTY: Distance is symmetric d(a,b) == d(b,a) for all N."""
    substrate = GridNDSubstrate(
        dimension_sizes=[size] * dimensions,
        boundary="clamp",
        distance_metric=metric,
        observation_encoding="relative",
    )

    # Two random positions
    pos_a = torch.randint(0, size, (1, dimensions), dtype=torch.long)
    pos_b = torch.randint(0, size, (1, dimensions), dtype=torch.long)

    # PROPERTY: d(a,b) == d(b,a)
    dist_ab = substrate.compute_distance(pos_a, pos_b)
    dist_ba = substrate.compute_distance(pos_b, pos_a)

    assert torch.allclose(dist_ab, dist_ba, atol=1e-6), f"Distance not symmetric in {dimensions}D {metric}"


@pytest.mark.parametrize(
    "dimensions,size,boundary_mode",
    [
        (4, 5, "clamp"),
        (7, 3, "clamp"),
        (4, 5, "wrap"),
        (7, 3, "sticky"),
    ],
)
def test_boundary_idempotence_gridnd(dimensions, size, boundary_mode):
    """PROPERTY: boundary(boundary(x)) == boundary(x)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[size] * dimensions,
        boundary=boundary_mode,
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Out-of-bounds position
    positions = torch.full((1, dimensions), size * 2, dtype=torch.long)
    deltas = torch.zeros((1, dimensions), dtype=torch.float32)

    # Apply boundary once
    bounded_once = substrate.apply_movement(positions, deltas)

    # Apply boundary again
    bounded_twice = substrate.apply_movement(bounded_once, deltas)

    # PROPERTY: boundary(boundary(x)) == boundary(x)
    assert torch.equal(bounded_once, bounded_twice), f"Boundary not idempotent in {dimensions}D {boundary_mode}"


@pytest.mark.parametrize(
    "dimensions,size",
    [
        (4, 5),
        (7, 3),
    ],
)
def test_movement_reversible_gridnd(dimensions, size):
    """PROPERTY: move + reverse = identity (for wrap boundary)."""
    substrate = GridNDSubstrate(
        dimension_sizes=[size] * dimensions,
        boundary="wrap",
        distance_metric="manhattan",
        observation_encoding="relative",
    )

    # Interior position
    center = size // 2
    positions = torch.full((1, dimensions), center, dtype=torch.long)

    # Random movement delta
    deltas = torch.randint(-1, 2, (1, dimensions), dtype=torch.float32)

    # Move forward
    moved = substrate.apply_movement(positions, deltas)

    # Move backward
    reversed_pos = substrate.apply_movement(moved, -deltas)

    # PROPERTY: move + reverse = identity
    assert torch.equal(reversed_pos, positions), f"Movement not reversible in {dimensions}D"


# ========================================
# ContinuousND Property Tests
# ========================================


@pytest.mark.parametrize(
    "dimensions,metric",
    [
        (4, "euclidean"),
        (4, "manhattan"),
        (4, "chebyshev"),
        (7, "euclidean"),
        (7, "manhattan"),
        (7, "chebyshev"),
    ],
)
def test_distance_symmetric_continuousnd(dimensions, metric):
    """PROPERTY: Distance is symmetric d(a,b) == d(b,a) for all N."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0)] * dimensions,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric=metric,
        observation_encoding="relative",
    )

    # Two random positions
    pos_a = torch.rand(1, dimensions, dtype=torch.float32) * 10.0
    pos_b = torch.rand(1, dimensions, dtype=torch.float32) * 10.0

    # PROPERTY: d(a,b) == d(b,a)
    dist_ab = substrate.compute_distance(pos_a, pos_b)
    dist_ba = substrate.compute_distance(pos_b, pos_a)

    assert torch.allclose(dist_ab, dist_ba, atol=1e-6), f"Distance not symmetric in {dimensions}D {metric}"


@pytest.mark.parametrize(
    "dimensions,boundary_mode",
    [
        (4, "clamp"),
        (7, "clamp"),
        (4, "wrap"),
        (7, "wrap"),
        (4, "sticky"),
        (7, "sticky"),
    ],
)
def test_boundary_idempotence_continuousnd(dimensions, boundary_mode):
    """PROPERTY: boundary(boundary(x)) == boundary(x)."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0)] * dimensions,
        boundary=boundary_mode,
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Out-of-bounds position
    positions = torch.full((1, dimensions), 20.0, dtype=torch.float32)
    deltas = torch.zeros((1, dimensions), dtype=torch.float32)

    # Apply boundary once
    bounded_once = substrate.apply_movement(positions, deltas)

    # Apply boundary again
    bounded_twice = substrate.apply_movement(bounded_once, deltas)

    # PROPERTY: boundary(boundary(x)) == boundary(x)
    assert torch.allclose(bounded_once, bounded_twice, atol=1e-6), f"Boundary not idempotent in {dimensions}D {boundary_mode}"


@pytest.mark.parametrize(
    "dimensions",
    [4, 7],
)
def test_movement_reversible_continuousnd(dimensions):
    """PROPERTY: move + reverse = identity (for wrap boundary)."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0)] * dimensions,
        boundary="wrap",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Interior position
    positions = torch.full((1, dimensions), 5.0, dtype=torch.float32)

    # Random movement delta
    deltas = torch.rand(1, dimensions, dtype=torch.float32) * 2.0 - 1.0  # [-1, 1] range

    # Move forward
    moved = substrate.apply_movement(positions, deltas)

    # Move backward
    reversed_pos = substrate.apply_movement(moved, -deltas)

    # PROPERTY: move + reverse = identity (with tolerance for float precision)
    assert torch.allclose(reversed_pos, positions, atol=1e-6), f"Movement not reversible in {dimensions}D"
