"""Comprehensive boundary mode and edge case tests for Grid3D substrate.

This module tests uncovered paths in Grid3D:
- 3D boundary mode edge cases (bounce with Z-axis, sticky on corners)
- Full grid encoding with 3D positions
- Invalid encoding modes for 3D
- 3D movement with all axes
- Distance metrics in 3D space
- Partial observation encoding for 3D

Coverage targets:
- src/townlet/substrate/grid3d.py:132-137 (boundary handling)
- src/townlet/substrate/grid3d.py:281-289 (full grid encoding)
- src/townlet/substrate/grid3d.py:417-423 (invalid encoding error paths)
- src/townlet/substrate/grid3d.py:457-462 (get_valid_neighbors)
"""

import pytest
import torch

from townlet.substrate.grid3d import Grid3DSubstrate


class TestGrid3DBoundaryEdgeCases:
    """Test 3D boundary mode edge cases including Z-axis."""

    def test_bounce_boundary_z_axis_positive(self):
        """Bounce boundary should handle Z-axis reflections correctly.

        Grid3D adds forward/backward (Z-axis) movement.
        Coverage target: lines 132-137 (z-axis boundary handling logic)
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="bounce",
            distance_metric="manhattan",
        )

        # Agent at center, move forward past boundary
        positions = torch.tensor([[2, 2, 2]], dtype=torch.long)
        # Move far forward (z+4 would take us to z=6, beyond depth=5)
        deltas = torch.tensor([[0, 0, 4]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # z=6 should bounce to z = 2*(5-1) - 6 = 8 - 6 = 2
        assert new_positions[0, 0].item() == 2, "X should not change"
        assert new_positions[0, 1].item() == 2, "Y should not change"
        assert new_positions[0, 2].item() == 2, "Should bounce from far boundary on Z"

    def test_bounce_boundary_z_axis_negative(self):
        """Bounce boundary should handle negative Z positions.

        Coverage target: Z-axis negative position bounce logic
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="bounce",
            distance_metric="manhattan",
        )

        # Agent near front, move backward past boundary
        positions = torch.tensor([[2, 2, 1]], dtype=torch.long)
        deltas = torch.tensor([[0, 0, -3]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # z = 1 - 3 = -2, bounce to abs(-2) = 2
        assert new_positions[0, 2].item() == 2, "Should bounce negative Z to positive"

    def test_sticky_boundary_3d_corner(self):
        """Sticky boundary should handle 3D corner (all axes out of bounds).

        Coverage target: sticky boundary logic for 3D
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="sticky",
            distance_metric="manhattan",
        )

        # Agent at corner [0, 0, 0]
        positions = torch.tensor([[0, 0, 0]], dtype=torch.long)
        # Try to move diagonally out of bounds in all axes
        deltas = torch.tensor([[-1, -1, -1]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # Should stick at [0, 0, 0]
        assert new_positions[0, 0].item() == 0, "X should stick"
        assert new_positions[0, 1].item() == 0, "Y should stick"
        assert new_positions[0, 2].item() == 0, "Z should stick"

    def test_wrap_boundary_3d_all_axes(self):
        """Wrap boundary should handle wraparound in all 3 axes.

        Coverage target: wrap boundary logic for 3D
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="wrap",
            distance_metric="manhattan",
        )

        # Agent at edge, move beyond in all axes
        positions = torch.tensor([[4, 4, 4]], dtype=torch.long)
        deltas = torch.tensor([[2, 2, 2]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # Each axis should wrap: (4+2) % 5 = 1
        assert new_positions[0, 0].item() == 1, "X should wrap"
        assert new_positions[0, 1].item() == 1, "Y should wrap"
        assert new_positions[0, 2].item() == 1, "Z should wrap"


class TestGrid3DMovementEdgeCases:
    """Test 3D-specific movement edge cases."""

    def test_movement_forward_backward_z_axis(self):
        """Should handle forward (Z+) and backward (Z-) movement correctly.

        Coverage target: Z-axis movement validation
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = torch.tensor([[2, 2, 2]], dtype=torch.long)

        # Move forward (Z+)
        forward_delta = torch.tensor([[0, 0, 1]], dtype=torch.long)
        new_pos_forward = substrate.apply_movement(positions, forward_delta)
        assert new_pos_forward[0, 2].item() == 3, "Should move forward on Z"

        # Move backward (Z-)
        backward_delta = torch.tensor([[0, 0, -1]], dtype=torch.long)
        new_pos_backward = substrate.apply_movement(positions, backward_delta)
        assert new_pos_backward[0, 2].item() == 1, "Should move backward on Z"

    def test_movement_clamp_all_boundaries_3d(self):
        """Clamp mode should work on all 6 boundaries (X, Y, Z × 2 sides).

        Coverage target: Clamp boundary for all 3 dimensions
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Test all 6 boundary clamping scenarios
        test_cases = [
            # (position, delta, expected_position, description)
            ([[0, 2, 2]], [[-1, 0, 0]], [[0, 2, 2]], "X min clamp"),
            ([[4, 2, 2]], [[1, 0, 0]], [[4, 2, 2]], "X max clamp"),
            ([[2, 0, 2]], [[0, -1, 0]], [[2, 0, 2]], "Y min clamp"),
            ([[2, 4, 2]], [[0, 1, 0]], [[2, 4, 2]], "Y max clamp"),
            ([[2, 2, 0]], [[0, 0, -1]], [[2, 2, 0]], "Z min clamp"),
            ([[2, 2, 4]], [[0, 0, 1]], [[2, 2, 4]], "Z max clamp"),
        ]

        for pos, delta, expected, desc in test_cases:
            positions = torch.tensor(pos, dtype=torch.long)
            deltas = torch.tensor(delta, dtype=torch.long)
            new_positions = substrate.apply_movement(positions, deltas)
            expected_tensor = torch.tensor(expected, dtype=torch.long)
            assert torch.equal(new_positions, expected_tensor), f"Failed: {desc}"


class TestGrid3DDistanceMetrics:
    """Test distance metrics in 3D space."""

    def test_distance_manhattan_3d(self):
        """Manhattan distance should sum |dx| + |dy| + |dz|.

        Coverage target: Manhattan distance for 3D
        """
        substrate = Grid3DSubstrate(
            width=10,
            height=10,
            depth=10,
            boundary="clamp",
            distance_metric="manhattan",
        )

        pos1 = torch.tensor([[1, 2, 3]], dtype=torch.long)
        pos2 = torch.tensor([[4, 6, 8]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        assert distance.item() == 12, "Manhattan distance in 3D should sum all axes"

    def test_distance_euclidean_3d(self):
        """Euclidean distance should be sqrt(dx² + dy² + dz²).

        Coverage target: Euclidean distance for 3D
        """
        substrate = Grid3DSubstrate(
            width=10,
            height=10,
            depth=10,
            boundary="clamp",
            distance_metric="euclidean",
        )

        pos1 = torch.tensor([[0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 4, 0]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # sqrt(3² + 4² + 0²) = sqrt(9 + 16) = sqrt(25) = 5.0
        assert abs(distance.item() - 5.0) < 0.001, "Euclidean distance should be Pythagorean 3D"

    def test_distance_chebyshev_3d(self):
        """Chebyshev distance should return max(|dx|, |dy|, |dz|).

        Coverage target: Chebyshev distance for 3D
        """
        substrate = Grid3DSubstrate(
            width=10,
            height=10,
            depth=10,
            boundary="clamp",
            distance_metric="chebyshev",
        )

        pos1 = torch.tensor([[1, 2, 3]], dtype=torch.long)
        pos2 = torch.tensor([[4, 6, 5]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # max(|1-4|, |2-6|, |3-5|) = max(3, 4, 2) = 4
        assert distance.item() == 4, "Chebyshev should return max component in 3D"


class TestGrid3DObservationEncoding:
    """Test observation encoding modes for 3D."""

    def test_relative_encoding_3d_dimensions(self):
        """Relative encoding should return normalized [x, y, z].

        Coverage target: 3D relative encoding
        """
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        positions = torch.tensor([[0, 0, 0], [7, 7, 7]], dtype=torch.long)
        affordances = {}

        encoded = substrate.encode_observation(positions, affordances)

        # Grid is 8*8*8 = 512 dims, relative position is 3 dims
        # Total = 512 + 3 = 515
        assert encoded.shape == (2, 515), "Should return grid + 3 position dims"

        # Check normalization: [0, 0, 0] → [0.0, 0.0, 0.0]
        assert torch.allclose(encoded[0, -3:], torch.tensor([0.0, 0.0, 0.0]))

        # [7, 7, 7] → [1.0, 1.0, 1.0]
        assert torch.allclose(encoded[1, -3:], torch.tensor([1.0, 1.0, 1.0]))

    def test_scaled_encoding_3d_dimensions(self):
        """Scaled encoding should return normalized + range metadata.

        Coverage target: 3D scaled encoding (6 dims: 3 normalized + 3 range sizes)
        """
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="scaled",
        )

        positions = torch.tensor([[3, 4, 5]], dtype=torch.long)
        affordances = {}

        encoded = substrate.encode_observation(positions, affordances)

        # Grid is 512 dims, scaled position is 6 dims (3 normalized + 3 sizes)
        # Total = 512 + 6 = 518
        assert encoded.shape == (1, 518), "Should return grid + 6 position dims"

        # Check last 6 dims: [normalized x, y, z, width, height, depth]
        last_6 = encoded[0, -6:]

        # Normalized: [3/7, 4/7, 5/7]
        assert torch.allclose(last_6[:3], torch.tensor([3 / 7, 4 / 7, 5 / 7]))

        # Range sizes: [8.0, 8.0, 8.0]
        assert torch.allclose(last_6[3:], torch.tensor([8.0, 8.0, 8.0]))

    def test_absolute_encoding_3d_dimensions(self):
        """Absolute encoding should return raw [x, y, z] coordinates.

        Coverage target: 3D absolute encoding
        """
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="absolute",
        )

        positions = torch.tensor([[3, 4, 5]], dtype=torch.long)
        affordances = {}

        encoded = substrate.encode_observation(positions, affordances)

        # Grid is 512 dims, absolute position is 3 dims
        # Total = 512 + 3 = 515
        assert encoded.shape == (1, 515), "Should return grid + 3 position dims"

        # Last 3 dims should be raw coordinates
        assert torch.allclose(encoded[0, -3:], torch.tensor([3.0, 4.0, 5.0]))


class TestGrid3DGetValidNeighbors:
    """Test 3D neighbor calculation (6-connected in 3D)."""

    def test_get_valid_neighbors_interior_6_connected(self):
        """Interior position should have 6 neighbors in 3D.

        Grid3D has 6-connected neighbors (±X, ±Y, ±Z).
        Coverage target: 6-connected neighbor generation
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        position = torch.tensor([2, 2, 2], dtype=torch.long)
        neighbors = substrate.get_valid_neighbors(position)

        # Should have 6 neighbors (clamp mode, interior position)
        assert len(neighbors) == 6, "Interior 3D position should have 6 neighbors"

        # Expected neighbors: [1,2,2], [3,2,2], [2,1,2], [2,3,2], [2,2,1], [2,2,3]
        neighbor_coords = [n.tolist() for n in neighbors]
        expected = [
            [1, 2, 2],  # -X
            [3, 2, 2],  # +X
            [2, 1, 2],  # -Y
            [2, 3, 2],  # +Y
            [2, 2, 1],  # -Z
            [2, 2, 3],  # +Z
        ]

        for exp in expected:
            assert exp in neighbor_coords, f"Expected neighbor {exp} not found"

    def test_get_valid_neighbors_corner_3d_clamp(self):
        """Corner position in 3D should have only 3 neighbors with clamp.

        Coverage target: Boundary filtering for 3D corners
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Corner at [0, 0, 0]
        position = torch.tensor([0, 0, 0], dtype=torch.long)
        neighbors = substrate.get_valid_neighbors(position)

        # Only +X, +Y, +Z are valid
        assert len(neighbors) == 3, "Corner should have 3 neighbors (clamp mode)"

        neighbor_coords = [n.tolist() for n in neighbors]
        expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for exp in expected:
            assert exp in neighbor_coords, f"Expected neighbor {exp} not found"


class TestGrid3DPartialObservationEdgeCases:
    """Test partial observation encoding for 3D (POMDP)."""

    def test_partial_observation_3d_window_size(self):
        """3D POMDP should create (2*vision_range+1)³ local cube.

        Coverage target: 3D partial observation window size
        """
        substrate = Grid3DSubstrate(
            width=10,
            height=10,
            depth=10,
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = torch.tensor([[5, 5, 5]], dtype=torch.long)
        affordances = {}

        # vision_range=2 → 5×5×5 cube = 125 cells
        encoded = substrate.encode_partial_observation(positions, affordances, vision_range=2)

        window_size = 2 * 2 + 1  # 5
        expected_cells = window_size**3  # 125

        assert encoded.shape == (1, expected_cells), f"Should return {expected_cells}-dim local cube"

    def test_partial_observation_3d_affordance_in_cube(self):
        """Affordances within 3D vision cube should be marked.

        Coverage target: 3D vision window affordance detection
        """
        substrate = Grid3DSubstrate(
            width=10,
            height=10,
            depth=10,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Agent at center
        positions = torch.tensor([[5, 5, 5]], dtype=torch.long)
        affordances = {
            "NearBed": torch.tensor([5, 5, 5], dtype=torch.long),  # At agent
            "FarBed": torch.tensor([0, 0, 0], dtype=torch.long),  # Outside cube
        }

        encoded = substrate.encode_partial_observation(positions, affordances, vision_range=2)

        # At least NearBed should be visible (FarBed should not contribute)
        assert encoded.sum() >= 1.0, "NearBed should be visible in local cube"


class TestGrid3DErrorPaths:
    """Test error handling and edge cases in Grid3D."""

    def test_invalid_encoding_mode_error(self):
        """Should raise ValueError for invalid observation_encoding.

        Coverage target: Invalid encoding error path
        """
        substrate = Grid3DSubstrate(
            width=5,
            height=5,
            depth=5,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        # Force invalid encoding
        substrate.observation_encoding = "INVALID"

        positions = torch.tensor([[2, 2, 2]], dtype=torch.long)
        affordances = {}

        with pytest.raises(ValueError) as exc_info:
            substrate.encode_observation(positions, affordances)

        assert "Invalid observation_encoding" in str(exc_info.value)

    def test_invalid_dimensions_raises_error(self):
        """Should raise ValueError for non-positive dimensions.

        Coverage target: Initialization validation
        """
        with pytest.raises(ValueError) as exc_info:
            Grid3DSubstrate(
                width=0,  # Invalid
                height=5,
                depth=5,
                boundary="clamp",
                distance_metric="manhattan",
            )

        assert "must be positive" in str(exc_info.value).lower()

    def test_invalid_boundary_mode_raises_error(self):
        """Should raise ValueError for unsupported boundary mode.

        Coverage target: Boundary mode validation
        """
        with pytest.raises(ValueError) as exc_info:
            Grid3DSubstrate(
                width=5,
                height=5,
                depth=5,
                boundary="INVALID_MODE",
                distance_metric="manhattan",
            )

        assert "boundary mode" in str(exc_info.value).lower()

    def test_invalid_distance_metric_raises_error(self):
        """Should raise ValueError for unsupported distance metric.

        Coverage target: Distance metric validation
        """
        with pytest.raises(ValueError) as exc_info:
            Grid3DSubstrate(
                width=5,
                height=5,
                depth=5,
                boundary="clamp",
                distance_metric="INVALID_METRIC",
            )

        assert "distance metric" in str(exc_info.value).lower()
