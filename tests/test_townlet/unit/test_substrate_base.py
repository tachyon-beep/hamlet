"""Test spatial substrate abstract interface."""

import pytest
import torch

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.base import SpatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate


def test_substrate_module_exists():
    """Substrate module should be importable."""
    assert SpatialSubstrate is not None


def test_substrate_is_abstract():
    """SpatialSubstrate should not be instantiable directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        SpatialSubstrate()


def test_grid2d_substrate_creation():
    """Grid2DSubstrate should instantiate with width/height."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    assert substrate.position_dim == 2
    assert substrate.get_observation_dim() == 64  # 8×8


def test_grid2d_initialize_positions():
    """Grid2D should initialize random positions in valid range."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    assert positions.shape == (10, 2)
    assert torch.all(positions >= 0)
    assert torch.all(positions[:, 0] < 8)  # x < width
    assert torch.all(positions[:, 1] < 8)  # y < height


def test_grid2d_apply_movement_clamp():
    """Grid2D with clamp boundary should keep agents in bounds."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    # Agent at top-left corner tries to move up-left
    positions = torch.tensor([[0, 0]], dtype=torch.long)
    deltas = torch.tensor([[-1, -1]], dtype=torch.long)

    new_positions = substrate.apply_movement(positions, deltas)

    # Should clamp to [0, 0] (can't go negative)
    assert torch.equal(new_positions, torch.tensor([[0, 0]], dtype=torch.long))


def test_grid2d_compute_distance_manhattan():
    """Grid2D should compute Manhattan distance correctly."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    pos1 = torch.tensor([[0, 0], [3, 4]], dtype=torch.long)
    pos2 = torch.tensor([5, 7], dtype=torch.long)  # Single position

    distances = substrate.compute_distance(pos1, pos2)

    # Manhattan: |0-5| + |0-7| = 12, |3-5| + |4-7| = 5
    assert torch.equal(distances, torch.tensor([12, 5], dtype=torch.long))


def test_grid2d_is_on_position():
    """Grid2D should check exact position match."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    agent_positions = torch.tensor([[3, 4], [5, 7], [3, 4]], dtype=torch.long)
    target_position = torch.tensor([3, 4], dtype=torch.long)

    on_target = substrate.is_on_position(agent_positions, target_position)

    assert torch.equal(on_target, torch.tensor([True, False, True]))


def test_grid2d_apply_movement_wrap():
    """Grid2D with wrap boundary should wrap positions (toroidal)."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="wrap", distance_metric="manhattan")

    # Agent at bottom-right tries to move right-down (should wrap to top-left)
    positions = torch.tensor([[7, 7]], dtype=torch.long)
    deltas = torch.tensor([[2, 2]], dtype=torch.long)

    new_positions = substrate.apply_movement(positions, deltas)

    # Should wrap: (7+2) % 8 = 1, (7+2) % 8 = 1
    assert torch.equal(new_positions, torch.tensor([[1, 1]], dtype=torch.long))


def test_grid2d_apply_movement_sticky():
    """Grid2D with sticky boundary should keep agent in place when hitting wall."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="sticky", distance_metric="manhattan")

    # Agent at edge tries to move out of bounds
    positions = torch.tensor([[0, 7]], dtype=torch.long)
    deltas = torch.tensor([[-1, 2]], dtype=torch.long)

    new_positions = substrate.apply_movement(positions, deltas)

    # Should stay in place for out-of-bounds dimensions
    assert torch.equal(new_positions, torch.tensor([[0, 7]], dtype=torch.long))


def test_grid2d_apply_movement_bounce():
    """Grid2D with bounce boundary should reflect agent back from wall."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="bounce", distance_metric="manhattan")

    # Agent at left edge tries to move 2 steps left (out of bounds by 2)
    positions = torch.tensor([[0, 3]], dtype=torch.long)
    deltas = torch.tensor([[-2, 0]], dtype=torch.long)

    new_positions = substrate.apply_movement(positions, deltas)

    # Should bounce back: would be at x=-2, reflects to x=2
    assert torch.equal(new_positions, torch.tensor([[2, 3]], dtype=torch.long))

    # Agent at top-right corner tries to move out on both axes
    positions = torch.tensor([[7, 0]], dtype=torch.long)
    deltas = torch.tensor([[3, -1]], dtype=torch.long)

    new_positions = substrate.apply_movement(positions, deltas)

    # Should bounce: x would be 10 (past 7), reflects to 4; y would be -1, reflects to 1
    assert torch.equal(new_positions, torch.tensor([[4, 1]], dtype=torch.long))


def test_grid2d_compute_distance_euclidean():
    """Grid2D should compute Euclidean distance correctly."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="euclidean")

    pos1 = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
    pos2 = torch.tensor([3, 4], dtype=torch.long)  # Single position

    distances = substrate.compute_distance(pos1, pos2)

    # Euclidean: sqrt(3² + 4²) = 5.0 for both agents
    expected = torch.tensor([5.0, 5.0])
    assert torch.allclose(distances, expected)


def test_grid2d_compute_distance_chebyshev():
    """Grid2D should compute Chebyshev distance correctly."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="chebyshev")

    pos1 = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    pos2 = torch.tensor([5, 3], dtype=torch.long)  # Single position

    distances = substrate.compute_distance(pos1, pos2)

    # Chebyshev: max(|0-5|, |0-3|) = 5, max(|1-5|, |1-3|) = 4
    assert torch.equal(distances, torch.tensor([5, 4], dtype=torch.long))


def test_grid2d_encode_observation():
    """Grid2D should encode grid with affordances and agent positions."""
    substrate = Grid2DSubstrate(width=3, height=3, boundary="clamp", distance_metric="manhattan")

    # Agent at [1, 1]
    positions = torch.tensor([[1, 1]], dtype=torch.long)

    # Affordance at [2, 2]
    affordances = {"Bed": torch.tensor([2, 2], dtype=torch.long)}

    encoding = substrate.encode_observation(positions, affordances)

    # Should be [1, 9] shape (1 agent, 3×3=9 cells)
    assert encoding.shape == (1, 9)

    # Agent position (1,1) -> flat index 1*3+1=4 should be 1.0
    assert encoding[0, 4] == 1.0

    # Affordance position (2,2) -> flat index 2*3+2=8 should be 1.0
    assert encoding[0, 8] == 1.0

    # All other cells should be 0.0
    mask = torch.ones(9, dtype=torch.bool)
    mask[4] = False
    mask[8] = False
    assert torch.all(encoding[0, mask] == 0.0)


def test_grid2d_encode_observation_overlap():
    """Grid2D should handle agent on affordance (overlap = 2.0)."""
    substrate = Grid2DSubstrate(width=3, height=3, boundary="clamp", distance_metric="manhattan")

    # Agent at [1, 1]
    positions = torch.tensor([[1, 1]], dtype=torch.long)

    # Affordance also at [1, 1] (same position)
    affordances = {"Bed": torch.tensor([1, 1], dtype=torch.long)}

    encoding = substrate.encode_observation(positions, affordances)

    # Overlapping position should be 2.0 (affordance 1.0 + agent 1.0)
    assert encoding[0, 4] == 2.0


def test_grid2d_get_valid_neighbors():
    """Grid2D should return valid 4-connected neighbors."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    # Test center position (has all 4 neighbors)
    position = torch.tensor([4, 4], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)

    assert len(neighbors) == 4
    expected_neighbors = [
        torch.tensor([4, 3]),  # UP
        torch.tensor([4, 5]),  # DOWN
        torch.tensor([3, 4]),  # LEFT
        torch.tensor([5, 4]),  # RIGHT
    ]

    for expected in expected_neighbors:
        found = any(torch.equal(n, expected) for n in neighbors)
        assert found, f"Expected neighbor {expected} not found"


def test_grid2d_get_valid_neighbors_corner():
    """Grid2D should filter out-of-bounds neighbors at corners."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    # Test top-left corner (only 2 valid neighbors)
    position = torch.tensor([0, 0], dtype=torch.long)
    neighbors = substrate.get_valid_neighbors(position)

    assert len(neighbors) == 2  # Only DOWN and RIGHT are valid

    # Verify no negative positions
    for neighbor in neighbors:
        assert torch.all(neighbor >= 0)
        assert neighbor[0] < 8 and neighbor[1] < 8


def test_aspatial_substrate_creation():
    """AspatialSubstrate should represent no positioning."""
    substrate = AspatialSubstrate()

    assert substrate.position_dim == 0  # No position!
    assert substrate.get_observation_dim() == 0  # No position encoding


def test_aspatial_initialize_positions():
    """Aspatial should return empty position tensors."""
    substrate = AspatialSubstrate()

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    assert positions.shape == (10, 0)  # Empty position vectors


def test_aspatial_compute_distance():
    """Aspatial should return zero distance (no spatial meaning)."""
    substrate = AspatialSubstrate()

    pos1 = torch.zeros((5, 0))  # 5 agents with no position
    pos2 = torch.zeros((0,))  # Target with no position

    distances = substrate.compute_distance(pos1, pos2)

    assert distances.shape == (5,)
    assert torch.all(distances == 0)  # All distances are zero


def test_aspatial_is_on_position():
    """Aspatial should always return True (no positioning concept)."""
    substrate = AspatialSubstrate()

    agent_positions = torch.zeros((10, 0))
    target_position = torch.zeros((0,))

    on_target = substrate.is_on_position(agent_positions, target_position)

    assert torch.all(on_target)  # All agents are "everywhere"
