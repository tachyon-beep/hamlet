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
    """Relative encoding should return [num_agents, 66] (64 grid + 2 position)."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_relative.encode_observation(positions, affordances)

    assert encoded.shape == (3, 66), "Should return [num_agents, 66] (64 grid + 2 position)"
    assert encoded.dtype == torch.float32


def test_grid2d_relative_encoding_values(grid2d_relative):
    """Relative encoding position features (last 2 dims) should normalize to [0, 1] range."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_relative.encode_observation(positions, affordances)

    # Position features are last 2 dimensions (after 64-dim grid)
    # 0 / 7 = 0.0, 7 / 7 = 1.0, 3 / 7 = 0.428..., 4 / 7 = 0.571...
    assert torch.allclose(encoded[0, -2:], torch.tensor([0.0, 0.0]))
    assert torch.allclose(encoded[1, -2:], torch.tensor([1.0, 1.0]))
    assert torch.allclose(encoded[2, -2:], torch.tensor([3 / 7, 4 / 7]))


def test_grid2d_scaled_encoding_dimensions(grid2d_scaled):
    """Scaled encoding should return [num_agents, 68] (64 grid + 4 position+metadata)."""
    positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_scaled.encode_observation(positions, affordances)

    assert encoded.shape == (2, 68), "Should return [num_agents, 68] (64 grid + 4 position+metadata)"
    assert encoded.dtype == torch.float32


def test_grid2d_scaled_encoding_values(grid2d_scaled):
    """Scaled encoding position features (last 4 dims) should have normalized positions + range metadata."""
    positions = torch.tensor([[3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_scaled.encode_observation(positions, affordances)

    # Last 4 dims: normalized positions (2) + range sizes (2)
    assert torch.allclose(encoded[0, -4:-2], torch.tensor([3 / 7, 4 / 7]))
    # Last 2 dims: range sizes (width=8, height=8)
    assert torch.allclose(encoded[0, -2:], torch.tensor([8.0, 8.0]))


def test_grid2d_absolute_encoding_dimensions(grid2d_absolute):
    """Absolute encoding should return [num_agents, 66] (64 grid + 2 raw position)."""
    positions = torch.tensor([[0, 0], [7, 7]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_absolute.encode_observation(positions, affordances)

    assert encoded.shape == (2, 66), "Should return [num_agents, 66] (64 grid + 2 position)"
    assert encoded.dtype == torch.float32


def test_grid2d_absolute_encoding_values(grid2d_absolute):
    """Absolute encoding position features (last 2 dims) should return raw unnormalized coordinates."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    affordances = {}

    encoded = grid2d_absolute.encode_observation(positions, affordances)

    # Position features are last 2 dims - should be raw float coordinates
    assert torch.allclose(encoded[0, -2:], torch.tensor([0.0, 0.0]))
    assert torch.allclose(encoded[1, -2:], torch.tensor([7.0, 7.0]))
    assert torch.allclose(encoded[2, -2:], torch.tensor([3.0, 4.0]))


def test_grid2d_get_observation_dim_relative(grid2d_relative):
    """get_observation_dim() should return 66 for relative encoding (64 grid + 2 position)."""
    assert grid2d_relative.get_observation_dim() == 66


def test_grid2d_get_observation_dim_scaled(grid2d_scaled):
    """get_observation_dim() should return 68 for scaled encoding (64 grid + 4 position+metadata)."""
    assert grid2d_scaled.get_observation_dim() == 68


def test_grid2d_get_observation_dim_absolute(grid2d_absolute):
    """get_observation_dim() should return 66 for absolute encoding (64 grid + 2 position)."""
    assert grid2d_absolute.get_observation_dim() == 66


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


# =============================================================================
# BACKWARD COMPATIBILITY TESTS (from TASK-002A)
# =============================================================================


def test_substrate_initialize_positions_correctness():
    """Grid2D.initialize_positions() should return valid grid positions.

    Legacy validation test: ensures position initialization works correctly.
    """
    substrate = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")

    positions = substrate.initialize_positions(num_agents=10, device=torch.device("cpu"))

    # Correct shape and type
    assert positions.shape == (10, 2)
    assert positions.dtype == torch.long

    # Within bounds
    assert (positions >= 0).all()
    assert (positions < 8).all()


def test_substrate_movement_matches_legacy():
    """Substrate movement should produce identical results to legacy torch.clamp.

    Legacy validation test: ensures substrate.apply_movement matches old hardcoded behavior.
    """
    from pathlib import Path

    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Test substrate.apply_movement directly
    substrate = env.substrate
    positions = torch.tensor([[3, 3]], dtype=torch.long, device=torch.device("cpu"))

    # Move up (delta [0, -1])
    deltas = torch.tensor([[0, -1]], dtype=torch.long, device=torch.device("cpu"))
    new_positions = substrate.apply_movement(positions, deltas)

    # Should move to [3, 2]
    assert (new_positions == torch.tensor([[3, 2]], dtype=torch.long)).all()

    # Test boundary clamping at edge
    edge_positions = torch.tensor([[0, 0]], dtype=torch.long, device=torch.device("cpu"))
    up_left_delta = torch.tensor([[-1, -1]], dtype=torch.long, device=torch.device("cpu"))
    clamped = substrate.apply_movement(edge_positions, up_left_delta)

    # Should clamp to [0, 0] (not go negative)
    assert (clamped == torch.tensor([[0, 0]], dtype=torch.long)).all()


# =============================================================================
# TOPOLOGY ATTRIBUTE TESTS
# =============================================================================


def test_grid2d_stores_topology_when_provided():
    """Grid2D should store topology attribute when explicitly provided."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="square",
    )
    assert substrate.topology == "square"


def test_grid2d_topology_defaults_to_square():
    """Grid2D topology should default to 'square' if not provided."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert substrate.topology == "square"


def test_grid2d_topology_attribute_exists():
    """Grid2D should have topology attribute (not inherited from base)."""
    substrate = Grid2DSubstrate(
        width=8,
        height=8,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(substrate, "topology")
