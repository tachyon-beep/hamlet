"""Comprehensive boundary mode and edge case tests for Grid2D substrate.

This module tests uncovered paths in Grid2D:
- Boundary mode edge cases (bounce with multiple reflections, sticky on corners)
- Full grid encoding with agents and affordances
- Invalid encoding modes
- Edge cases in partial observation encoding
- Distance metric edge cases

Coverage targets:
- src/townlet/substrate/grid2d.py:324-376 (full grid encoding)
- src/townlet/substrate/grid2d.py:395 (invalid encoding error path)
- src/townlet/substrate/grid2d.py:429 (invalid encoding in get_observation_dim)
- src/townlet/substrate/grid2d.py:108-132 (bounce and sticky boundary detailed behavior)
"""

import pytest
import torch

from townlet.substrate.grid2d import Grid2DSubstrate


class TestGrid2DBoundaryEdgeCases:
    """Test boundary mode edge cases and corner behavior."""

    def test_bounce_boundary_multiple_reflections_x_axis(self):
        """Bounce boundary should handle multiple reflections correctly on x-axis.

        When agent moves far beyond boundary, it should reflect properly.
        Coverage target: lines 113-117 (x-axis bouncing logic)
        """
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="bounce",
            distance_metric="manhattan",
        )

        # Agent at center, move far right (beyond boundary)
        positions = torch.tensor([[3, 3]], dtype=torch.long)
        # Delta would take us to x=8 (out of bounds)
        deltas = torch.tensor([[5, 0]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # x=8 should bounce to x = 2*(8-1) - 8 = 14 - 8 = 6
        assert new_positions[0, 0].item() == 6, "Should bounce from right boundary"
        assert new_positions[0, 1].item() == 3, "Y should not change"

    def test_bounce_boundary_multiple_reflections_y_axis(self):
        """Bounce boundary should handle multiple reflections correctly on y-axis.

        Coverage target: lines 119-123 (y-axis bouncing logic)
        """
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="bounce",
            distance_metric="manhattan",
        )

        # Agent at center, move far down (beyond boundary)
        positions = torch.tensor([[3, 3]], dtype=torch.long)
        # Delta would take us to y=9 (well beyond boundary)
        deltas = torch.tensor([[0, 6]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # y=9 should bounce to y = 2*(8-1) - 9 = 14 - 9 = 5
        assert new_positions[0, 0].item() == 3, "X should not change"
        assert new_positions[0, 1].item() == 5, "Should bounce from bottom boundary"

    def test_bounce_boundary_negative_positions(self):
        """Bounce boundary should handle negative positions (absolute value reflection).

        Coverage target: lines 114, 116, 120, 122 (negative position bounce masks)
        """
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="bounce",
            distance_metric="manhattan",
        )

        # Agent near edge, move left past boundary
        positions = torch.tensor([[1, 2]], dtype=torch.long)
        deltas = torch.tensor([[-3, -4]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # x = 1 - 3 = -2, bounce to abs(-2) = 2
        # y = 2 - 4 = -2, bounce to abs(-2) = 2
        assert new_positions[0, 0].item() == 2, "Should bounce negative x to positive"
        assert new_positions[0, 1].item() == 2, "Should bounce negative y to positive"

    def test_sticky_boundary_corner_behavior(self):
        """Sticky boundary should keep agent in place when hitting corner.

        Coverage target: lines 126-131 (sticky boundary logic)
        """
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="sticky",
            distance_metric="manhattan",
        )

        # Agent at top-left corner
        positions = torch.tensor([[0, 0]], dtype=torch.long)
        # Try to move diagonally out of bounds
        deltas = torch.tensor([[-1, -1]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # Should stay at [0, 0] (both axes out of bounds)
        assert new_positions[0, 0].item() == 0, "Should stick at left boundary"
        assert new_positions[0, 1].item() == 0, "Should stick at top boundary"

    def test_sticky_boundary_partial_movement(self):
        """Sticky boundary should only prevent movement on out-of-bounds axis.

        If one axis is valid and one is out of bounds, only the invalid axis should stick.
        Coverage target: lines 127-131 (independent axis checking)
        """
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="sticky",
            distance_metric="manhattan",
        )

        # Agent at left edge, mid-height
        positions = torch.tensor([[0, 4]], dtype=torch.long)
        # Try to move left (invalid) and down (valid)
        deltas = torch.tensor([[-1, 1]], dtype=torch.long)

        new_positions = substrate.apply_movement(positions, deltas)

        # x should stick at 0, y should move to 5
        assert new_positions[0, 0].item() == 0, "X should stick at left boundary"
        assert new_positions[0, 1].item() == 5, "Y should move normally"


class TestGrid2DFullGridEncoding:
    """Test full grid encoding with agents and affordances.

    Coverage target: lines 324-376 (_encode_full_grid method)
    """

    def test_encode_full_grid_with_affordances(self):
        """Should mark affordances in global grid encoding.

        Coverage target: lines 343-353 (affordance grid population)
        """
        substrate = Grid2DSubstrate(
            width=5,
            height=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = torch.tensor([[2, 2]], dtype=torch.long)
        affordances = {
            "Bed": torch.tensor([1, 1], dtype=torch.long),
            "Hospital": torch.tensor([3, 3], dtype=torch.long),
        }

        encoded = substrate.encode_observation(positions, affordances)

        # Global grid is first 25 dims (5x5)
        grid = encoded[0, :25]

        # Check affordance positions are marked
        bed_idx = 1 * 5 + 1  # y=1, x=1
        hospital_idx = 3 * 5 + 3  # y=3, x=3
        agent_idx = 2 * 5 + 2  # y=2, x=2 (no affordance here)

        assert grid[bed_idx] == 1.0, "Bed should be marked in grid"
        assert grid[hospital_idx] == 1.0, "Hospital should be marked in grid"
        assert grid[agent_idx] == 1.0, "Agent position should be marked (empty cell + agent)"

    def test_encode_full_grid_agent_affordance_overlap(self):
        """Should handle agent and affordance at same position (value 2.0).

        Coverage target: lines 370-374 (agent overlay with clamping)
        """
        substrate = Grid2DSubstrate(
            width=5,
            height=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Agent and affordance at same position
        positions = torch.tensor([[2, 2]], dtype=torch.long)
        affordances = {
            "Bed": torch.tensor([2, 2], dtype=torch.long),
        }

        encoded = substrate.encode_observation(positions, affordances)

        grid = encoded[0, :25]
        overlap_idx = 2 * 5 + 2  # y=2, x=2

        # Should be 2.0 (affordance 1.0 + agent 1.0, clamped to max 2.0)
        assert grid[overlap_idx] == 2.0, "Overlap should be clamped to 2.0"

    def test_encode_full_grid_out_of_bounds_affordance_ignored(self):
        """Should ignore affordances with positions outside grid bounds.

        Coverage target: lines 351 (bounds check for affordances)
        """
        substrate = Grid2DSubstrate(
            width=5,
            height=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = torch.tensor([[2, 2]], dtype=torch.long)
        affordances = {
            "Bed": torch.tensor([1, 1], dtype=torch.long),
            "OutOfBounds": torch.tensor([10, 10], dtype=torch.long),  # Invalid position
        }

        # Should not raise error, just ignore out-of-bounds affordance
        encoded = substrate.encode_observation(positions, affordances)

        grid = encoded[0, :25]
        bed_idx = 1 * 5 + 1

        assert grid[bed_idx] == 1.0, "Valid affordance should be marked"
        # Out of bounds affordance should be ignored (no crash)

    def test_encode_full_grid_empty_affordances(self):
        """Should handle empty affordance dict gracefully.

        Coverage target: lines 343 (iteration over empty dict)
        """
        substrate = Grid2DSubstrate(
            width=5,
            height=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = torch.tensor([[2, 2]], dtype=torch.long)
        affordances = {}

        encoded = substrate.encode_observation(positions, affordances)

        grid = encoded[0, :25]
        agent_idx = 2 * 5 + 2

        # Only agent should be marked (0.0 + 1.0 for agent)
        assert grid[agent_idx] == 1.0, "Agent should be marked"
        assert grid.sum() == 1.0, "Only agent should contribute to grid"

    def test_encode_full_grid_zero_agents(self):
        """Should handle zero agents case (early return path).

        Coverage target: lines 358-359 (zero agents early return)
        """
        substrate = Grid2DSubstrate(
            width=5,
            height=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = torch.tensor([], dtype=torch.long).reshape(0, 2)
        affordances = {
            "Bed": torch.tensor([1, 1], dtype=torch.long),
        }

        encoded = substrate.encode_observation(positions, affordances)

        # Should return empty tensor or handle gracefully
        assert encoded.shape[0] == 0, "Should handle zero agents"

    def test_encode_full_grid_agent_out_of_bounds_raises_error(self):
        """Should raise error if agent position is out of bounds.

        Coverage target: lines 366-368 (out-of-bounds agent error)
        """
        substrate = Grid2DSubstrate(
            width=5,
            height=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Create invalid agent position (beyond grid)
        positions = torch.tensor([[10, 10]], dtype=torch.long)
        affordances = {}

        with pytest.raises(ValueError) as exc_info:
            substrate.encode_observation(positions, affordances)

        error_msg = str(exc_info.value)
        assert "out of bounds" in error_msg.lower(), "Should mention out of bounds"
        assert "10" in error_msg, "Should mention invalid position"


class TestGrid2DInvalidEncodingMode:
    """Test error handling for invalid observation encoding modes.

    Coverage targets:
    - line 395 (invalid encoding in _encode_position_features)
    - line 429 (invalid encoding in get_observation_dim)
    """

    def test_invalid_encoding_mode_in_encode_position_features(self):
        """Should raise ValueError for invalid observation_encoding.

        This tests the error path in _encode_position_features.
        Coverage target: line 395
        """
        # Manually set invalid encoding (bypassing __init__ validation)
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )
        substrate.observation_encoding = "INVALID_MODE"  # Force invalid value

        positions = torch.tensor([[3, 4]], dtype=torch.long)
        affordances = {}

        with pytest.raises(ValueError) as exc_info:
            substrate.encode_observation(positions, affordances)

        error_msg = str(exc_info.value)
        assert "Invalid observation_encoding" in error_msg
        assert "INVALID_MODE" in error_msg

    def test_invalid_encoding_mode_in_get_observation_dim(self):
        """Should raise ValueError for invalid encoding in get_observation_dim.

        Coverage target: line 429
        """
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )
        substrate.observation_encoding = "INVALID_MODE"  # Force invalid value

        with pytest.raises(ValueError) as exc_info:
            substrate.get_observation_dim()

        error_msg = str(exc_info.value)
        assert "Invalid observation_encoding" in error_msg
        assert "INVALID_MODE" in error_msg


class TestGrid2DPartialObservationEdgeCases:
    """Test edge cases in partial observation encoding (POMDP).

    Coverage target: lines 483-538 (encode_partial_observation)
    """

    def test_partial_observation_affordance_outside_vision(self):
        """Affordances outside vision range should not be marked in local grid.

        Coverage target: lines 534 (vision window boundary check)
        """
        substrate = Grid2DSubstrate(
            width=10,
            height=10,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Agent at center
        positions = torch.tensor([[5, 5]], dtype=torch.long)
        # Affordance far away (outside 5x5 window with vision_range=2)
        affordances = {
            "FarBed": torch.tensor([0, 0], dtype=torch.long),  # Outside vision
            "NearBed": torch.tensor([5, 5], dtype=torch.long),  # At agent position
        }

        encoded = substrate.encode_partial_observation(positions, affordances, vision_range=2)

        # Vision window is 5x5 = 25 cells
        assert encoded.shape == (1, 25), "Should return 5x5 local grid"

        # Center cell (where agent and NearBed are) should be marked
        center_idx = 2 * 5 + 2  # Center of 5x5 window
        assert encoded[0, center_idx] == 1.0, "Affordance at agent position should be marked"

        # FarBed should not contribute (outside vision)
        # Only one cell should be marked
        assert encoded.sum() == 1.0, "Only NearBed should be visible"

    def test_partial_observation_affordance_at_window_edge(self):
        """Affordances exactly at vision window edge should be included.

        Coverage target: lines 529-535 (relative position calculation and bounds check)
        """
        substrate = Grid2DSubstrate(
            width=10,
            height=10,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Agent at [5, 5], vision_range=2 means window from [3,3] to [7,7]
        positions = torch.tensor([[5, 5]], dtype=torch.long)
        affordances = {
            "EdgeBed": torch.tensor([7, 7], dtype=torch.long),  # At edge of vision
        }

        encoded = substrate.encode_partial_observation(positions, affordances, vision_range=2)

        # Window is 5x5, affordance at [7,7] should be at window position [4,4]
        # rel_x = 7 - 5 + 2 = 4, rel_y = 7 - 5 + 2 = 4
        edge_idx = 4 * 5 + 4
        assert encoded[0, edge_idx] == 1.0, "Affordance at vision edge should be marked"

    def test_partial_observation_multiple_agents(self):
        """Should create separate local grids for each agent.

        Coverage target: lines 518 (agent loop)
        """
        substrate = Grid2DSubstrate(
            width=10,
            height=10,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Two agents at different positions
        positions = torch.tensor([[2, 2], [7, 7]], dtype=torch.long)
        affordances = {
            "Bed1": torch.tensor([2, 2], dtype=torch.long),  # Visible to agent 0
            "Bed2": torch.tensor([7, 7], dtype=torch.long),  # Visible to agent 1
        }

        encoded = substrate.encode_partial_observation(positions, affordances, vision_range=2)

        assert encoded.shape == (2, 25), "Should have 2 separate 5x5 grids"

        # Agent 0 sees Bed1 at center
        center_idx = 2 * 5 + 2
        assert encoded[0, center_idx] == 1.0, "Agent 0 should see Bed1"

        # Agent 1 sees Bed2 at center
        assert encoded[1, center_idx] == 1.0, "Agent 1 should see Bed2"


class TestGrid2DDistanceMetricEdgeCases:
    """Test distance metric edge cases and broadcasting.

    Coverage target: lines 135-155 (compute_distance with broadcasting)
    """

    def test_distance_broadcasting_single_target(self):
        """Should handle broadcasting when pos2 is single position [2].

        Coverage target: lines 142-143 (unsqueeze for broadcasting)
        """
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Multiple agent positions
        pos1 = torch.tensor([[0, 0], [3, 4], [7, 7]], dtype=torch.long)
        # Single target position [2] (will be broadcast)
        pos2 = torch.tensor([5, 5], dtype=torch.long)

        distances = substrate.compute_distance(pos1, pos2)

        assert distances.shape == (3,), "Should return distance for each agent"
        assert distances[0].item() == 10, "Manhattan distance [0,0] to [5,5]"
        assert distances[1].item() == 3, "Manhattan distance [3,4] to [5,5]"
        assert distances[2].item() == 4, "Manhattan distance [7,7] to [5,5]"

    def test_chebyshev_distance_computes_max(self):
        """Chebyshev distance should return max(|x1-x2|, |y1-y2|).

        Coverage target: lines 153-155 (chebyshev metric)
        """
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="chebyshev",
        )

        pos1 = torch.tensor([[0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 7]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # max(|0-3|, |0-7|) = max(3, 7) = 7
        assert distance.item() == 7, "Chebyshev should return max component"


class TestGrid2DGetValidNeighbors:
    """Test get_valid_neighbors edge cases.

    Coverage target: lines 445-467 (get_valid_neighbors)
    """

    def test_get_valid_neighbors_wrap_boundary_returns_all(self):
        """Wrap boundary should return all 4 neighbors without filtering.

        Coverage target: lines 463-465 (clamp boundary filtering vs others)
        """
        substrate = Grid2DSubstrate(
            width=5,
            height=5,
            boundary="wrap",
            distance_metric="manhattan",
        )

        # Corner position
        position = torch.tensor([0, 0], dtype=torch.long)
        neighbors = substrate.get_valid_neighbors(position)

        # Wrap mode returns all 4 neighbors (even if they appear out of bounds)
        assert len(neighbors) == 4, "Wrap mode should return all 4 neighbors"

    def test_get_valid_neighbors_clamp_boundary_filters(self):
        """Clamp boundary should filter out-of-bounds neighbors.

        Coverage target: lines 464-465 (bounds filtering for clamp)
        """
        substrate = Grid2DSubstrate(
            width=5,
            height=5,
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Corner position
        position = torch.tensor([0, 0], dtype=torch.long)
        neighbors = substrate.get_valid_neighbors(position)

        # Clamp mode filters out negative positions
        # Only RIGHT and DOWN are valid from [0,0]
        assert len(neighbors) == 2, "Clamp mode should filter out-of-bounds neighbors"

        # Check that returned neighbors are in bounds
        for neighbor in neighbors:
            assert 0 <= neighbor[0] < 5, "Neighbor x should be in bounds"
            assert 0 <= neighbor[1] < 5, "Neighbor y should be in bounds"
