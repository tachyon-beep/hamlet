"""Test ContinuousND edge cases and boundary conditions."""

import pytest
import torch

from townlet.substrate.continuousnd import ContinuousNDSubstrate


def test_continuousnd_bounce_boundary_single_reflection():
    """Test bounce boundary with single reflection."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        boundary="bounce",
        movement_delta=1.0,
        interaction_radius=0.5,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Agent near boundary
    positions = torch.tensor([[9.5, 9.5, 9.5, 9.5]], dtype=torch.float32)

    # Move past boundary (should bounce back)
    deltas = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Should reflect: 10.5 → 9.5
    expected = torch.tensor([[9.5, 9.5, 9.5, 9.5]], dtype=torch.float32)
    assert torch.allclose(new_positions, expected, atol=1e-5)


def test_continuousnd_bounce_boundary_negative():
    """Test bounce boundary with negative overshoot."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        boundary="bounce",
        movement_delta=1.0,
        interaction_radius=0.5,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Agent near lower boundary
    positions = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    # Move past boundary (should bounce back)
    deltas = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Should reflect: -0.5 → 0.5
    expected = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    assert torch.allclose(new_positions, expected, atol=1e-5)


def test_continuousnd_sticky_boundary():
    """Test sticky boundary (agent stays in place if out of bounds)."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        boundary="sticky",
        movement_delta=1.0,
        interaction_radius=0.5,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Agent near boundary
    positions = torch.tensor([[9.5, 9.5, 9.5, 9.5]], dtype=torch.float32)

    # Try to move out of bounds (should stay in place)
    deltas = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Should stay at original position
    assert torch.allclose(new_positions, positions)


def test_continuousnd_large_movement_delta():
    """Test with very large movement deltas."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=100.0,  # Large delta
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Agent in middle
    positions = torch.tensor([[5.0, 5.0, 5.0, 5.0]], dtype=torch.float32)

    # Move with large delta (should clamp to boundary)
    deltas = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Should clamp to max boundary [10.0, 10.0, 10.0, 10.0]
    expected = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
    assert torch.allclose(new_positions, expected)


def test_continuousnd_negative_bounds():
    """Test with negative coordinate bounds."""
    substrate = ContinuousNDSubstrate(
        bounds=[(-10.0, 10.0), (-20.0, 20.0), (-5.0, 5.0), (-100.0, 0.0)],
        boundary="clamp",
        movement_delta=1.0,
        interaction_radius=0.5,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Agent at origin (within bounds)
    positions = torch.tensor([[0.0, 0.0, 0.0, -50.0]], dtype=torch.float32)

    # Move negative
    deltas = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    expected = torch.tensor([[-1.0, -1.0, -1.0, -51.0]], dtype=torch.float32)
    assert torch.allclose(new_positions, expected)


def test_continuousnd_interaction_radius_larger_than_space():
    """Test that interaction_radius must be <= space range."""
    with pytest.raises(ValueError, match="Space too small for affordance interaction"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],  # Range = 1.0
            boundary="clamp",
            movement_delta=0.1,
            interaction_radius=2.0,  # Larger than range!
            distance_metric="euclidean",
            observation_encoding="relative",
        )


def test_continuousnd_warns_if_interaction_radius_less_than_delta():
    """Test warning when interaction_radius < movement_delta."""
    with pytest.warns(UserWarning, match="interaction_radius.*<.*movement_delta"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=2.0,
            interaction_radius=1.0,  # Smaller than movement_delta
            distance_metric="euclidean",
            observation_encoding="relative",
        )


def test_continuousnd_validates_movement_delta_positive():
    """Test that movement_delta must be positive."""
    with pytest.raises(ValueError, match="movement_delta must be positive"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=0.0,  # Invalid
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )


def test_continuousnd_validates_interaction_radius_positive():
    """Test that interaction_radius must be positive."""
    with pytest.raises(ValueError, match="interaction_radius must be positive"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=-1.0,  # Invalid
            distance_metric="euclidean",
            observation_encoding="relative",
        )


def test_continuousnd_invalid_boundary_mode():
    """Test that invalid boundary mode raises error."""
    with pytest.raises(ValueError, match="Unknown boundary mode"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="invalid",
            movement_delta=1.0,
            interaction_radius=0.5,
            distance_metric="euclidean",
            observation_encoding="relative",
        )


def test_continuousnd_invalid_distance_metric():
    """Test that invalid distance metric raises error."""
    with pytest.raises(ValueError, match="Unknown distance metric"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.5,
            distance_metric="invalid",
            observation_encoding="relative",
        )


def test_continuousnd_invalid_observation_encoding():
    """Test that invalid observation encoding raises error."""
    with pytest.raises(ValueError, match="Unknown observation encoding"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.5,
            distance_metric="euclidean",
            observation_encoding="invalid",
        )


def test_continuousnd_mixed_positive_negative_bounds():
    """Test bounds spanning negative and positive coordinates."""
    substrate = ContinuousNDSubstrate(
        bounds=[(-5.0, 5.0), (-10.0, 10.0), (-1.0, 1.0), (-100.0, 100.0)],
        boundary="wrap",
        movement_delta=1.0,
        interaction_radius=0.5,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Agent at origin
    positions = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    # Move negative (should wrap to positive side)
    deltas = torch.tensor([[-6.0, -11.0, -2.0, -101.0]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Wrap around: -6 in [-5, 5] wraps to 4, etc.
    # Range sizes: 10, 20, 2, 200
    # -6 - (-5) = -1 → -1 % 10 = 9 → 9 + (-5) = 4
    expected = torch.tensor([[4.0, 9.0, 0.0, 99.0]], dtype=torch.float32)
    assert torch.allclose(new_positions, expected, atol=1e-5)


def test_continuousnd_distance_broadcasting():
    """Test distance computation with broadcasting (single target vs multiple agents)."""
    substrate = ContinuousNDSubstrate(
        bounds=[(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)],
        boundary="clamp",
        movement_delta=0.5,
        interaction_radius=1.0,
        distance_metric="euclidean",
        observation_encoding="relative",
    )

    # Multiple agents
    agent_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Single target
    target_position = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    distances = substrate.compute_distance(agent_positions, target_position)

    assert distances.shape == (3,)
    assert torch.allclose(distances[0], torch.tensor(0.0))
    assert torch.allclose(distances[1], torch.tensor(5.0))  # sqrt(3^2 + 4^2)
    assert torch.allclose(distances[2], torch.tensor(2.0))  # sqrt(1^2 + 1^2 + 1^2 + 1^2)


def test_continuousnd_zero_range_dimension():
    """Test that zero-range dimensions (min == max) are rejected."""
    with pytest.raises(ValueError, match="min.*must be < max"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (5.0, 5.0), (0.0, 10.0), (0.0, 10.0)],  # Dimension 1 has zero range
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )


def test_continuousnd_inverted_bounds():
    """Test that inverted bounds (min > max) are rejected."""
    with pytest.raises(ValueError, match="min.*must be < max"):
        ContinuousNDSubstrate(
            bounds=[(0.0, 10.0), (10.0, 0.0), (0.0, 10.0), (0.0, 10.0)],  # Dimension 1 inverted
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
            observation_encoding="relative",
        )
