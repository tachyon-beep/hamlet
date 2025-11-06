"""Test configurable observation encoding for Grid3D substrate."""

import pytest
import torch

from townlet.substrate.grid3d import Grid3DSubstrate


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def grid3d_relative(device):
    """Grid3D with relative encoding (normalized coordinates)."""
    return Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )


@pytest.fixture
def grid3d_scaled(device):
    """Grid3D with scaled encoding (normalized + ranges)."""
    return Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="scaled",
    )


@pytest.fixture
def grid3d_absolute(device):
    """Grid3D with absolute encoding (raw coordinates)."""
    return Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="absolute",
    )


def test_grid3d_relative_encoding_dimensions(grid3d_relative):
    """Relative encoding should return [num_agents, 195] (192 grid + 3 position)."""
    positions = torch.tensor([[0, 0, 0], [7, 7, 2], [3, 4, 1]], dtype=torch.long)
    affordances = {}

    encoded = grid3d_relative.encode_observation(positions, affordances)

    assert encoded.shape == (3, 195), "Should return [num_agents, 195] (192 grid + 3 position)"
    assert encoded.dtype == torch.float32


def test_grid3d_relative_encoding_values(grid3d_relative):
    """Relative encoding position features (last 3 dims) should normalize to [0, 1] range."""
    positions = torch.tensor([[0, 0, 0], [7, 7, 2], [3, 4, 1]], dtype=torch.long)
    affordances = {}

    encoded = grid3d_relative.encode_observation(positions, affordances)

    # Position features are last 3 dimensions (after 192-dim grid)
    # 0 / 7 = 0.0, 7 / 7 = 1.0, 3 / 7 = 0.428..., 4 / 7 = 0.571...
    # For depth: 0 / 2 = 0.0, 2 / 2 = 1.0, 1 / 2 = 0.5
    assert torch.allclose(encoded[0, -3:], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(encoded[1, -3:], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(encoded[2, -3:], torch.tensor([3 / 7, 4 / 7, 0.5]))


def test_grid3d_scaled_encoding_dimensions(grid3d_scaled):
    """Scaled encoding should return [num_agents, 198] (192 grid + 6 position+metadata)."""
    positions = torch.tensor([[0, 0, 0], [7, 7, 2]], dtype=torch.long)
    affordances = {}

    encoded = grid3d_scaled.encode_observation(positions, affordances)

    assert encoded.shape == (2, 198), "Should return [num_agents, 198] (192 grid + 6 position+metadata)"
    assert encoded.dtype == torch.float32


def test_grid3d_scaled_encoding_values(grid3d_scaled):
    """Scaled encoding position features (last 6 dims) should have normalized positions + range metadata."""
    positions = torch.tensor([[3, 4, 1]], dtype=torch.long)
    affordances = {}

    encoded = grid3d_scaled.encode_observation(positions, affordances)

    # Last 6 dims: normalized positions (3) + range sizes (3)
    assert torch.allclose(encoded[0, -6:-3], torch.tensor([3 / 7, 4 / 7, 0.5]))
    # Last 3 dims: range sizes (width=8, height=8, depth=3)
    assert torch.allclose(encoded[0, -3:], torch.tensor([8.0, 8.0, 3.0]))


def test_grid3d_absolute_encoding_dimensions(grid3d_absolute):
    """Absolute encoding should return [num_agents, 195] (192 grid + 3 raw position)."""
    positions = torch.tensor([[0, 0, 0], [7, 7, 2]], dtype=torch.long)
    affordances = {}

    encoded = grid3d_absolute.encode_observation(positions, affordances)

    assert encoded.shape == (2, 195), "Should return [num_agents, 195] (192 grid + 3 position)"
    assert encoded.dtype == torch.float32


def test_grid3d_absolute_encoding_values(grid3d_absolute):
    """Absolute encoding position features (last 3 dims) should return raw unnormalized coordinates."""
    positions = torch.tensor([[0, 0, 0], [7, 7, 2], [3, 4, 1]], dtype=torch.long)
    affordances = {}

    encoded = grid3d_absolute.encode_observation(positions, affordances)

    # Position features are last 3 dims - should be raw float coordinates
    assert torch.allclose(encoded[0, -3:], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(encoded[1, -3:], torch.tensor([7.0, 7.0, 2.0]))
    assert torch.allclose(encoded[2, -3:], torch.tensor([3.0, 4.0, 1.0]))


def test_grid3d_get_observation_dim_relative(grid3d_relative):
    """get_observation_dim() should return 195 for relative encoding (192 grid + 3 position)."""
    assert grid3d_relative.get_observation_dim() == 195


def test_grid3d_get_observation_dim_scaled(grid3d_scaled):
    """get_observation_dim() should return 198 for scaled encoding (192 grid + 6 position+metadata)."""
    assert grid3d_scaled.get_observation_dim() == 198


def test_grid3d_get_observation_dim_absolute(grid3d_absolute):
    """get_observation_dim() should return 195 for absolute encoding (192 grid + 3 position)."""
    assert grid3d_absolute.get_observation_dim() == 195


def test_grid3d_default_encoding_is_relative():
    """Grid3D should default to relative encoding for backward compatibility."""
    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        # observation_encoding NOT provided
    )
    assert substrate.observation_encoding == "relative"
