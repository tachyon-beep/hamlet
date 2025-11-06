"""Unit tests for Grid3DSubstrate."""

import pytest
import torch

from townlet.substrate.grid3d import Grid3DSubstrate


class TestGrid3DInitialization:
    """Tests for Grid3D initialization."""

    def test_initialization_valid(self):
        """Valid grid initializes."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        assert substrate.position_dim == 3
        assert substrate.position_dtype == torch.long
        assert substrate.width == 8
        assert substrate.height == 8
        assert substrate.depth == 3

    def test_initialization_invalid_dimensions(self):
        """Invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            Grid3DSubstrate(width=0, height=8, depth=3, boundary="clamp")

        with pytest.raises(ValueError, match="dimensions must be positive"):
            Grid3DSubstrate(width=8, height=-1, depth=3, boundary="clamp")

    def test_initialization_invalid_boundary(self):
        """Invalid boundary raises ValueError."""
        with pytest.raises(ValueError, match="Unknown boundary mode"):
            Grid3DSubstrate(width=8, height=8, depth=3, boundary="invalid")


class TestGrid3DPositionInitialization:
    """Tests for position initialization."""

    def test_initialize_positions_shape(self):
        """Positions have correct shape."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        positions = substrate.initialize_positions(100, torch.device("cpu"))

        assert positions.shape == (100, 3)
        assert positions.dtype == torch.long

    def test_initialize_positions_in_bounds(self):
        """Positions within grid bounds."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        positions = substrate.initialize_positions(1000, torch.device("cpu"))

        assert (positions[:, 0] >= 0).all()
        assert (positions[:, 0] < 8).all()
        assert (positions[:, 1] >= 0).all()
        assert (positions[:, 1] < 8).all()
        assert (positions[:, 2] >= 0).all()
        assert (positions[:, 2] < 3).all()


class TestGrid3DMovement:
    """Tests for 3D movement."""

    def test_movement_x_axis(self):
        """Movement along X axis."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        positions = torch.tensor([[4, 4, 1]], dtype=torch.long)
        deltas = torch.tensor([[1, 0, 0]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        assert torch.equal(new_positions, torch.tensor([[5, 4, 1]]))

    def test_movement_z_axis(self):
        """Movement along Z axis (vertical)."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        positions = torch.tensor([[4, 4, 1]], dtype=torch.long)
        deltas = torch.tensor([[0, 0, 1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Z from 1 → 2 (going up one floor)
        assert torch.equal(new_positions, torch.tensor([[4, 4, 2]]))

    def test_movement_clamp_boundary(self):
        """Clamp boundary prevents out of bounds."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        positions = torch.tensor([[7, 7, 2]], dtype=torch.long)
        deltas = torch.tensor([[1, 1, 1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # All dimensions clamped to max
        assert torch.equal(new_positions, torch.tensor([[7, 7, 2]]))

    def test_movement_wrap_boundary(self):
        """Wrap boundary uses toroidal wraparound in 3D."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="wrap")
        positions = torch.tensor([[7, 7, 2]], dtype=torch.long)
        deltas = torch.tensor([[1, 1, 1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Wraps: (8 % 8, 8 % 8, 3 % 3) = (0, 0, 0)
        assert torch.equal(new_positions, torch.tensor([[0, 0, 0]]))

    def test_movement_batch(self):
        """Batch movement works."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        positions = torch.tensor([[1, 2, 0], [3, 4, 1], [5, 6, 2]], dtype=torch.long)
        deltas = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        expected = torch.tensor([[2, 2, 0], [3, 5, 1], [5, 6, 2]])
        assert torch.equal(new_positions, expected)


class TestGrid3DDistance:
    """Tests for distance calculations."""

    def test_distance_manhattan(self):
        """Manhattan distance in 3D."""
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=3,
            boundary="clamp",
            distance_metric="manhattan",
        )
        pos1 = torch.tensor([[0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 4, 2]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # |3| + |4| + |2| = 9
        assert distance[0].item() == 9

    def test_distance_euclidean(self):
        """Euclidean distance in 3D."""
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=3,
            boundary="clamp",
            distance_metric="euclidean",
        )
        pos1 = torch.tensor([[0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 4, 0]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # sqrt(9 + 16) = 5
        assert torch.isclose(distance[0], torch.tensor(5.0))

    def test_distance_chebyshev(self):
        """Chebyshev distance in 3D."""
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=3,
            boundary="clamp",
            distance_metric="chebyshev",
        )
        pos1 = torch.tensor([[0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 7, 1]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # max(3, 7, 1) = 7
        assert distance[0].item() == 7


class TestGrid3DObservationEncoding:
    """Tests for observation encoding."""

    def test_encode_observation_shape(self):
        """Observation encoding returns grid+position (195 dims for 8×8×3 grid)."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        positions = torch.tensor([[0, 0, 0], [7, 7, 2], [4, 4, 1]], dtype=torch.long)

        obs = substrate.encode_observation(positions, {})

        # Should be [num_agents, 195] (192 grid + 3 position)
        assert obs.shape == (3, 195)
        assert obs.dtype == torch.float32

    def test_encode_observation_normalization(self):
        """Position features (last 3 dims) normalized to [0, 1]."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        positions = torch.tensor([[0, 0, 0], [7, 7, 2]], dtype=torch.long)

        obs = substrate.encode_observation(positions, {})

        # Position features are last 3 dimensions (after 192-dim grid)
        # Min corner: [0, 0, 0]
        assert torch.allclose(obs[0, -3:], torch.tensor([0.0, 0.0, 0.0]))

        # Max corner: [7, 7, 2]
        # Normalized: [7/7, 7/7, 2/2] = [1, 1, 1]
        assert torch.allclose(obs[1, -3:], torch.tensor([1.0, 1.0, 1.0]))

    def test_encode_observation_scales_with_grid_size(self):
        """Large grids produce grid+position (100K grid + 3 position = 100,003 dims)."""
        substrate = Grid3DSubstrate(width=100, height=100, depth=10, boundary="clamp")
        positions = torch.tensor([[50, 50, 5]], dtype=torch.long)

        obs = substrate.encode_observation(positions, {})

        # 100,000 grid cells + 3 position features = 100,003 dims
        assert obs.shape == (1, 100003)

        # Position features (last 3 dims): Middle of grid ≈ [0.5, 0.5, 0.5]
        expected = torch.tensor([50 / 99, 50 / 99, 5 / 9])
        assert torch.allclose(obs[0, -3:], expected, atol=0.01)


class TestGrid3DPositionChecks:
    """Tests for position checking."""

    def test_is_on_position_exact_match(self):
        """is_on_position returns True for exact match."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        agent_positions = torch.tensor([[3, 4, 1], [5, 6, 2]], dtype=torch.long)
        target_position = torch.tensor([3, 4, 1], dtype=torch.long)

        on_position = substrate.is_on_position(agent_positions, target_position)

        assert on_position[0].item() is True
        assert on_position[1].item() is False

    def test_is_on_position_no_match(self):
        """is_on_position returns False for different positions."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        agent_positions = torch.tensor([[3, 4, 1]], dtype=torch.long)
        target_position = torch.tensor([3, 4, 2], dtype=torch.long)

        on_position = substrate.is_on_position(agent_positions, target_position)

        # Different floor (z=1 vs z=2)
        assert on_position[0].item() is False


class TestGrid3DNeighbors:
    """Tests for neighbor enumeration."""

    def test_get_valid_neighbors_interior(self):
        """Interior position has 6 neighbors."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        neighbors = substrate.get_valid_neighbors(torch.tensor([4, 4, 1]))

        assert len(neighbors) == 6

        expected = {
            (4, 3, 1),  # Y-
            (4, 5, 1),  # Y+
            (3, 4, 1),  # X-
            (5, 4, 1),  # X+
            (4, 4, 0),  # Z-
            (4, 4, 2),  # Z+
        }
        assert {tuple(n.tolist()) for n in neighbors} == expected

    def test_get_valid_neighbors_corner_clamp(self):
        """Corner position with clamp has fewer neighbors."""
        substrate = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp")
        neighbors = substrate.get_valid_neighbors(torch.tensor([0, 0, 0]))

        # Only 3 neighbors (no negatives)
        assert len(neighbors) == 3

        expected = {
            (0, 1, 0),  # Y+
            (1, 0, 0),  # X+
            (0, 0, 1),  # Z+
        }
        assert {tuple(n.tolist()) for n in neighbors} == expected

    def test_get_all_positions(self):
        """get_all_positions returns all grid cells."""
        substrate = Grid3DSubstrate(width=2, height=2, depth=2, boundary="clamp")
        all_positions = substrate.get_all_positions()

        # 2*2*2 = 8 cells
        assert len(all_positions) == 8

        # Should contain all corners
        assert [0, 0, 0] in all_positions
        assert [1, 1, 1] in all_positions


class TestGrid3DConfiguration:
    """Tests for config integration."""

    def test_config_cubic_topology(self):
        """Config with cubic topology creates Grid3D."""
        from townlet.substrate.config import GridConfig, SubstrateConfig
        from townlet.substrate.factory import SubstrateFactory

        config = SubstrateConfig(
            version="1.0",
            description="Test 3D grid",
            type="grid",
            grid=GridConfig(
                topology="cubic",
                width=8,
                height=8,
                depth=3,
                boundary="clamp",
            ),
        )

        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, Grid3DSubstrate)
        assert substrate.position_dim == 3
        assert substrate.position_dtype == torch.long


# =============================================================================
# TOPOLOGY ATTRIBUTE TESTS
# =============================================================================


def test_grid3d_stores_topology_when_provided():
    """Grid3D should store topology attribute when explicitly provided."""
    from townlet.substrate.grid3d import Grid3DSubstrate

    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
        topology="cubic",
    )
    assert substrate.topology == "cubic"


def test_grid3d_topology_defaults_to_cubic():
    """Grid3D topology should default to 'cubic' if not provided."""
    from townlet.substrate.grid3d import Grid3DSubstrate

    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert substrate.topology == "cubic"


def test_grid3d_topology_attribute_exists():
    """Grid3D should have topology attribute."""
    from townlet.substrate.grid3d import Grid3DSubstrate

    substrate = Grid3DSubstrate(
        width=8,
        height=8,
        depth=3,
        boundary="clamp",
        distance_metric="manhattan",
        observation_encoding="relative",
    )
    assert hasattr(substrate, "topology")
