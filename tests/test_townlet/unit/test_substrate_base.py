"""Test spatial substrate abstract interface."""

import pytest
import torch

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
    substrate = Grid2DSubstrate(width=8, height=8)

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    assert positions.shape == (10, 2)
    assert torch.all(positions >= 0)
    assert torch.all(positions[:, 0] < 8)  # x < width
    assert torch.all(positions[:, 1] < 8)  # y < height


def test_grid2d_apply_movement_clamp():
    """Grid2D with clamp boundary should keep agents in bounds."""
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp")

    # Agent at top-left corner tries to move up-left
    positions = torch.tensor([[0, 0]], dtype=torch.long)
    deltas = torch.tensor([[-1, -1]], dtype=torch.long)

    new_positions = substrate.apply_movement(positions, deltas)

    # Should clamp to [0, 0] (can't go negative)
    assert torch.equal(new_positions, torch.tensor([[0, 0]], dtype=torch.long))


def test_grid2d_compute_distance_manhattan():
    """Grid2D should compute Manhattan distance correctly."""
    substrate = Grid2DSubstrate(width=8, height=8, distance_metric="manhattan")

    pos1 = torch.tensor([[0, 0], [3, 4]], dtype=torch.long)
    pos2 = torch.tensor([5, 7], dtype=torch.long)  # Single position

    distances = substrate.compute_distance(pos1, pos2)

    # Manhattan: |0-5| + |0-7| = 12, |3-5| + |4-7| = 5
    assert torch.equal(distances, torch.tensor([12, 5], dtype=torch.long))


def test_grid2d_is_on_position():
    """Grid2D should check exact position match."""
    substrate = Grid2DSubstrate(width=8, height=8)

    agent_positions = torch.tensor([[3, 4], [5, 7], [3, 4]], dtype=torch.long)
    target_position = torch.tensor([3, 4], dtype=torch.long)

    on_target = substrate.is_on_position(agent_positions, target_position)

    assert torch.equal(on_target, torch.tensor([True, False, True]))
