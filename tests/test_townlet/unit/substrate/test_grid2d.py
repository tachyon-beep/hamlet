"""Test configurable observation encoding for Grid2D substrate."""

import pytest
import torch

from townlet.substrate.grid2d import Grid2DSubstrate


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def grid2d_relative(device):
    """Grid2D with relative encoding (normalized coordinates)."""
    return Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )


@pytest.fixture
def grid2d_scaled(device):
    """Grid2D with scaled encoding (normalized + ranges)."""
    return Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="scaled",
    )


@pytest.fixture
def grid2d_absolute(device):
    """Grid2D with absolute encoding (raw coordinates)."""
    return Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="absolute",
    )


def test_grid2d_relative_encoding_dimensions(grid2d_relative):
    """Relative encoding should return [num_agents, 2] normalized positions."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_relative.encode_observation(positions, affordances)

    assert encoded.shape == (3, 2), "Should return [num_agents, 2]"
    assert encoded.dtype == torch.float32


def test_grid2d_relative_encoding_values(grid2d_relative):
    """Relative encoding should normalize to [0, 1] range."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_relative.encode_observation(positions, affordances)

    # 0 / 7 = 0.0, 7 / 7 = 1.0, 3 / 7 = 0.428..., 4 / 7 = 0.571...
    assert torch.allclose(encoded[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(encoded[1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(encoded[2], torch.tensor([3 / 7, 4 / 7]))


def test_grid2d_scaled_encoding_dimensions(grid2d_scaled):
    """Scaled encoding should return [num_agents, 4] (normalized + ranges)."""
    positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_scaled.encode_observation(positions, affordances)

    assert encoded.shape == (2, 4), "Should return [num_agents, 4] (2 pos + 2 ranges)"
    assert encoded.dtype == torch.float32


def test_grid2d_scaled_encoding_values(grid2d_scaled):
    """Scaled encoding should have normalized positions + range metadata."""
    positions = torch.tensor([[3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_scaled.encode_observation(positions, affordances)

    # First 2 dims: normalized positions
    assert torch.allclose(encoded[0, :2], torch.tensor([3 / 7, 4 / 7]))
    # Last 2 dims: range sizes (width=8, height=8)
    assert torch.allclose(encoded[0, 2:], torch.tensor([8.0, 8.0]))


def test_grid2d_absolute_encoding_dimensions(grid2d_absolute):
    """Absolute encoding should return [num_agents, 2] raw coordinates."""
    positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_absolute.encode_observation(positions, affordances)

    assert encoded.shape == (2, 2), "Should return [num_agents, 2]"
    assert encoded.dtype == torch.float32


def test_grid2d_absolute_encoding_values(grid2d_absolute):
    """Absolute encoding should return raw unnormalized coordinates."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_absolute.encode_observation(positions, affordances)

    # Should be raw float coordinates
    assert torch.allclose(encoded[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(encoded[1], torch.tensor([7.0, 7.0]))
    assert torch.allclose(encoded[2], torch.tensor([3.0, 4.0]))


def test_grid2d_get_observation_dim_relative(grid2d_relative):
    """get_observation_dim() should return 2 for relative encoding."""
    assert grid2d_relative.get_observation_dim() == 2


def test_grid2d_get_observation_dim_scaled(grid2d_scaled):
    """get_observation_dim() should return 4 for scaled encoding."""
    assert grid2d_scaled.get_observation_dim() == 4


def test_grid2d_get_observation_dim_absolute(grid2d_absolute):
    """get_observation_dim() should return 2 for absolute encoding."""
    assert grid2d_absolute.get_observation_dim() == 2


def test_grid2d_default_encoding_is_relative():
    """Grid2D should default to relative encoding for backward compatibility."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        # observation_encoding NOT provided
    )
    assert substrate.observation_encoding == "relative"
