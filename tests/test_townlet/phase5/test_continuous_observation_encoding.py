"""Test configurable observation encoding for Continuous substrates.

This tests the retrofit of Continuous1D/2D/3D substrates to support
three observation encoding modes:
- relative: Normalized [0, 1] positions
- scaled: Normalized + range metadata
- absolute: Raw unnormalized positions
"""

import pytest
import torch

from townlet.substrate.continuous import (
    Continuous1DSubstrate,
    Continuous2DSubstrate,
    Continuous3DSubstrate,
)


@pytest.fixture
def device():
    """Test device (CPU for unit tests)."""
    return torch.device("cpu")


# ==================== Continuous1D Tests ====================


@pytest.fixture
def continuous_1d_relative():
    """1D continuous substrate with relative encoding."""
    return Continuous1DSubstrate(
        min_x=0.0,
        max_x=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="relative",
    )


@pytest.fixture
def continuous_1d_scaled():
    """1D continuous substrate with scaled encoding."""
    return Continuous1DSubstrate(
        min_x=0.0,
        max_x=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="scaled",
    )


@pytest.fixture
def continuous_1d_absolute():
    """1D continuous substrate with absolute encoding."""
    return Continuous1DSubstrate(
        min_x=0.0,
        max_x=10.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="absolute",
    )


def test_continuous_1d_relative_dimensions(continuous_1d_relative):
    """Test Continuous1D relative encoding dimensions."""
    assert continuous_1d_relative.get_observation_dim() == 1


def test_continuous_1d_scaled_dimensions(continuous_1d_scaled):
    """Test Continuous1D scaled encoding dimensions."""
    assert continuous_1d_scaled.get_observation_dim() == 2


def test_continuous_1d_absolute_dimensions(continuous_1d_absolute):
    """Test Continuous1D absolute encoding dimensions."""
    assert continuous_1d_absolute.get_observation_dim() == 1


def test_continuous_1d_relative_encoding(continuous_1d_relative, device):
    """Test Continuous1D relative encoding values."""
    # Positions at bounds: [0.0], [5.0], [10.0]
    positions = torch.tensor([[0.0], [5.0], [10.0]], device=device)
    obs = continuous_1d_relative.encode_observation(positions, {})

    assert obs.shape == (3, 1)
    assert torch.allclose(obs[0], torch.tensor([0.0], device=device))  # min → 0.0
    assert torch.allclose(obs[1], torch.tensor([0.5], device=device))  # mid → 0.5
    assert torch.allclose(obs[2], torch.tensor([1.0], device=device))  # max → 1.0


def test_continuous_1d_scaled_encoding(continuous_1d_scaled, device):
    """Test Continuous1D scaled encoding values."""
    # Positions at bounds: [0.0], [5.0], [10.0]
    positions = torch.tensor([[0.0], [5.0], [10.0]], device=device)
    obs = continuous_1d_scaled.encode_observation(positions, {})

    assert obs.shape == (3, 2)
    # First column: normalized position
    assert torch.allclose(obs[0, 0], torch.tensor(0.0, device=device))
    assert torch.allclose(obs[1, 0], torch.tensor(0.5, device=device))
    assert torch.allclose(obs[2, 0], torch.tensor(1.0, device=device))
    # Second column: range size (max - min = 10.0)
    assert torch.allclose(obs[:, 1], torch.tensor([10.0, 10.0, 10.0], device=device))


def test_continuous_1d_absolute_encoding(continuous_1d_absolute, device):
    """Test Continuous1D absolute encoding values."""
    # Positions at bounds: [0.0], [5.0], [10.0]
    positions = torch.tensor([[0.0], [5.0], [10.0]], device=device)
    obs = continuous_1d_absolute.encode_observation(positions, {})

    assert obs.shape == (3, 1)
    assert torch.allclose(obs[0], torch.tensor([0.0], device=device))
    assert torch.allclose(obs[1], torch.tensor([5.0], device=device))
    assert torch.allclose(obs[2], torch.tensor([10.0], device=device))


# ==================== Continuous2D Tests ====================


@pytest.fixture
def continuous_2d_relative():
    """2D continuous substrate with relative encoding."""
    return Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=20.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="relative",
    )


@pytest.fixture
def continuous_2d_scaled():
    """2D continuous substrate with scaled encoding."""
    return Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=20.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="scaled",
    )


@pytest.fixture
def continuous_2d_absolute():
    """2D continuous substrate with absolute encoding."""
    return Continuous2DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=20.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="absolute",
    )


def test_continuous_2d_relative_dimensions(continuous_2d_relative):
    """Test Continuous2D relative encoding dimensions."""
    assert continuous_2d_relative.get_observation_dim() == 2


def test_continuous_2d_scaled_dimensions(continuous_2d_scaled):
    """Test Continuous2D scaled encoding dimensions."""
    assert continuous_2d_scaled.get_observation_dim() == 4


def test_continuous_2d_absolute_dimensions(continuous_2d_absolute):
    """Test Continuous2D absolute encoding dimensions."""
    assert continuous_2d_absolute.get_observation_dim() == 2


def test_continuous_2d_relative_encoding(continuous_2d_relative, device):
    """Test Continuous2D relative encoding values."""
    # Positions: (0, 0), (5, 10), (10, 20)
    positions = torch.tensor([[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]], device=device)
    obs = continuous_2d_relative.encode_observation(positions, {})

    assert obs.shape == (3, 2)
    assert torch.allclose(obs[0], torch.tensor([0.0, 0.0], device=device))
    assert torch.allclose(obs[1], torch.tensor([0.5, 0.5], device=device))
    assert torch.allclose(obs[2], torch.tensor([1.0, 1.0], device=device))


def test_continuous_2d_scaled_encoding(continuous_2d_scaled, device):
    """Test Continuous2D scaled encoding values."""
    # Positions: (0, 0), (5, 10), (10, 20)
    positions = torch.tensor([[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]], device=device)
    obs = continuous_2d_scaled.encode_observation(positions, {})

    assert obs.shape == (3, 4)
    # First 2 columns: normalized positions
    assert torch.allclose(obs[0, :2], torch.tensor([0.0, 0.0], device=device))
    assert torch.allclose(obs[1, :2], torch.tensor([0.5, 0.5], device=device))
    assert torch.allclose(obs[2, :2], torch.tensor([1.0, 1.0], device=device))
    # Last 2 columns: range sizes (x_range=10.0, y_range=20.0)
    assert torch.allclose(obs[:, 2], torch.tensor([10.0, 10.0, 10.0], device=device))
    assert torch.allclose(obs[:, 3], torch.tensor([20.0, 20.0, 20.0], device=device))


def test_continuous_2d_absolute_encoding(continuous_2d_absolute, device):
    """Test Continuous2D absolute encoding values."""
    # Positions: (0, 0), (5, 10), (10, 20)
    positions = torch.tensor([[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]], device=device)
    obs = continuous_2d_absolute.encode_observation(positions, {})

    assert obs.shape == (3, 2)
    assert torch.allclose(obs[0], torch.tensor([0.0, 0.0], device=device))
    assert torch.allclose(obs[1], torch.tensor([5.0, 10.0], device=device))
    assert torch.allclose(obs[2], torch.tensor([10.0, 20.0], device=device))


# ==================== Continuous3D Tests ====================


@pytest.fixture
def continuous_3d_relative():
    """3D continuous substrate with relative encoding."""
    return Continuous3DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=20.0,
        min_z=0.0,
        max_z=30.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="relative",
    )


@pytest.fixture
def continuous_3d_scaled():
    """3D continuous substrate with scaled encoding."""
    return Continuous3DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=20.0,
        min_z=0.0,
        max_z=30.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="scaled",
    )


@pytest.fixture
def continuous_3d_absolute():
    """3D continuous substrate with absolute encoding."""
    return Continuous3DSubstrate(
        min_x=0.0,
        max_x=10.0,
        min_y=0.0,
        max_y=20.0,
        min_z=0.0,
        max_z=30.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="absolute",
    )


def test_continuous_3d_relative_dimensions(continuous_3d_relative):
    """Test Continuous3D relative encoding dimensions."""
    assert continuous_3d_relative.get_observation_dim() == 3


def test_continuous_3d_scaled_dimensions(continuous_3d_scaled):
    """Test Continuous3D scaled encoding dimensions."""
    assert continuous_3d_scaled.get_observation_dim() == 6


def test_continuous_3d_absolute_dimensions(continuous_3d_absolute):
    """Test Continuous3D absolute encoding dimensions."""
    assert continuous_3d_absolute.get_observation_dim() == 3


def test_continuous_3d_relative_encoding(continuous_3d_relative, device):
    """Test Continuous3D relative encoding values."""
    # Positions: (0, 0, 0), (5, 10, 15), (10, 20, 30)
    positions = torch.tensor([[0.0, 0.0, 0.0], [5.0, 10.0, 15.0], [10.0, 20.0, 30.0]], device=device)
    obs = continuous_3d_relative.encode_observation(positions, {})

    assert obs.shape == (3, 3)
    assert torch.allclose(obs[0], torch.tensor([0.0, 0.0, 0.0], device=device))
    assert torch.allclose(obs[1], torch.tensor([0.5, 0.5, 0.5], device=device))
    assert torch.allclose(obs[2], torch.tensor([1.0, 1.0, 1.0], device=device))


def test_continuous_3d_scaled_encoding(continuous_3d_scaled, device):
    """Test Continuous3D scaled encoding values."""
    # Positions: (0, 0, 0), (5, 10, 15), (10, 20, 30)
    positions = torch.tensor([[0.0, 0.0, 0.0], [5.0, 10.0, 15.0], [10.0, 20.0, 30.0]], device=device)
    obs = continuous_3d_scaled.encode_observation(positions, {})

    assert obs.shape == (3, 6)
    # First 3 columns: normalized positions
    assert torch.allclose(obs[0, :3], torch.tensor([0.0, 0.0, 0.0], device=device))
    assert torch.allclose(obs[1, :3], torch.tensor([0.5, 0.5, 0.5], device=device))
    assert torch.allclose(obs[2, :3], torch.tensor([1.0, 1.0, 1.0], device=device))
    # Last 3 columns: range sizes (x=10.0, y=20.0, z=30.0)
    assert torch.allclose(obs[:, 3], torch.tensor([10.0, 10.0, 10.0], device=device))
    assert torch.allclose(obs[:, 4], torch.tensor([20.0, 20.0, 20.0], device=device))
    assert torch.allclose(obs[:, 5], torch.tensor([30.0, 30.0, 30.0], device=device))


def test_continuous_3d_absolute_encoding(continuous_3d_absolute, device):
    """Test Continuous3D absolute encoding values."""
    # Positions: (0, 0, 0), (5, 10, 15), (10, 20, 30)
    positions = torch.tensor([[0.0, 0.0, 0.0], [5.0, 10.0, 15.0], [10.0, 20.0, 30.0]], device=device)
    obs = continuous_3d_absolute.encode_observation(positions, {})

    assert obs.shape == (3, 3)
    assert torch.allclose(obs[0], torch.tensor([0.0, 0.0, 0.0], device=device))
    assert torch.allclose(obs[1], torch.tensor([5.0, 10.0, 15.0], device=device))
    assert torch.allclose(obs[2], torch.tensor([10.0, 20.0, 30.0], device=device))


# ==================== Edge Case Tests ====================


def test_continuous_1d_non_zero_min():
    """Test Continuous1D with non-zero minimum."""
    substrate = Continuous1DSubstrate(
        min_x=5.0,
        max_x=15.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="relative",
    )

    positions = torch.tensor([[5.0], [10.0], [15.0]])
    obs = substrate.encode_observation(positions, {})

    # Should still normalize to [0, 1]
    assert torch.allclose(obs[0], torch.tensor([0.0]))
    assert torch.allclose(obs[1], torch.tensor([0.5]))
    assert torch.allclose(obs[2], torch.tensor([1.0]))


def test_continuous_2d_asymmetric_bounds():
    """Test Continuous2D with asymmetric bounds."""
    substrate = Continuous2DSubstrate(
        min_x=-5.0,
        max_x=5.0,
        min_y=0.0,
        max_y=20.0,
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        observation_encoding="scaled",
    )

    positions = torch.tensor([[-5.0, 0.0], [0.0, 10.0], [5.0, 20.0]])
    obs = substrate.encode_observation(positions, {})

    # Normalized positions
    assert torch.allclose(obs[0, :2], torch.tensor([0.0, 0.0]))
    assert torch.allclose(obs[1, :2], torch.tensor([0.5, 0.5]))
    assert torch.allclose(obs[2, :2], torch.tensor([1.0, 1.0]))

    # Range metadata
    assert torch.allclose(obs[:, 2], torch.tensor([10.0, 10.0, 10.0]))  # x_range
    assert torch.allclose(obs[:, 3], torch.tensor([20.0, 20.0, 20.0]))  # y_range


def test_continuous_invalid_encoding_mode():
    """Test that invalid encoding mode raises error."""
    with pytest.raises(TypeError):
        Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            observation_encoding="invalid",  # Should fail type checking
        )
