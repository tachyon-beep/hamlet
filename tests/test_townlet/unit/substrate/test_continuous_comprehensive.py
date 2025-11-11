"""Comprehensive coverage tests for Continuous substrate (1D/2D/3D).

This module tests uncovered paths in Continuous substrates to improve coverage from 17% → target 70%+:

Coverage targets:
- src/townlet/substrate/continuous.py:306-355 (get_default_actions for 1D/2D/3D)
- src/townlet/substrate/continuous.py:146-162 (bounce boundary complex reflection)
- src/townlet/substrate/continuous.py:183-184 (chebyshev distance)
- src/townlet/substrate/continuous.py:356-392 (get_valid_neighbors error)
- src/townlet/substrate/continuous.py:400-423 (normalize_positions)
- src/townlet/substrate/continuous.py:475-487 (encode_partial_observation NotImplementedError)
- src/townlet/substrate/continuous.py:285, 302 (invalid encoding errors)
"""

import pytest
import torch

from townlet.substrate.continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate


class TestContinuousActionGeneration:
    """Test get_default_actions() for 1D/2D/3D continuous substrates.

    Coverage target: lines 306-355 (completely untested)
    """

    def test_get_default_actions_1d_count(self):
        """1D continuous should generate 4 actions (LEFT, RIGHT, INTERACT, WAIT).

        Coverage target: 1D action generation
        """
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # 1D: 2 movement + INTERACT + WAIT = 4 actions
        assert len(actions) == 4, "1D should have 4 actions"

        # Verify action names
        names = [a.name for a in actions]
        assert names == ["LEFT", "RIGHT", "INTERACT", "WAIT"]

    def test_get_default_actions_2d_count(self):
        """2D continuous should generate 6 actions (UP, DOWN, LEFT, RIGHT, INTERACT, WAIT).

        Coverage target: 2D action generation
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # 2D: 4 movement + INTERACT + WAIT = 6 actions
        assert len(actions) == 6, "2D should have 6 actions"

        # Verify action names
        names = [a.name for a in actions]
        assert names == ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "WAIT"]

    def test_get_default_actions_3d_count(self):
        """3D continuous should generate 8 actions (all 6 directions + INTERACT + WAIT).

        Coverage target: 3D action generation
        """
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # 3D: 6 movement + INTERACT + WAIT = 8 actions
        assert len(actions) == 8, "3D should have 8 actions"

        # Verify action names (3D uses UP_Z/DOWN_Z for Z-axis)
        names = [a.name for a in actions]
        assert names == ["UP", "DOWN", "LEFT", "RIGHT", "UP_Z", "DOWN_Z", "INTERACT", "WAIT"]

    def test_get_default_actions_deltas_1d(self):
        """1D actions should have correct deltas.

        Coverage target: 1D delta assignment
        """
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # LEFT should be [-1]
        assert actions[0].delta == [-1], "LEFT delta should be [-1]"
        # RIGHT should be [1]
        assert actions[1].delta == [1], "RIGHT delta should be [1]"

    def test_get_default_actions_deltas_3d(self):
        """3D actions should have correct deltas including Z-axis.

        Coverage target: 3D delta assignment
        """
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # UP_Z should be [0, 0, -1] (upward in Z)
        assert actions[4].delta == [0, 0, -1], "UP_Z delta should be [0, 0, -1]"
        # DOWN_Z should be [0, 0, 1] (downward in Z)
        assert actions[5].delta == [0, 0, 1], "DOWN_Z delta should be [0, 0, 1]"

    def test_get_default_actions_interact_and_wait(self):
        """INTERACT and WAIT should have no deltas and appropriate costs.

        Coverage target: INTERACT and WAIT generation
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # INTERACT (second-to-last)
        interact = actions[-2]
        assert interact.name == "INTERACT"
        assert interact.type == "interaction"
        assert interact.delta is None
        assert "energy" in interact.costs

        # WAIT (last)
        wait = actions[-1]
        assert wait.name == "WAIT"
        assert wait.type == "passive"
        assert wait.delta is None
        assert "energy" in wait.costs


class TestContinuousBounceEdgeCases:
    """Test complex bounce boundary behavior.

    Coverage target: lines 146-162 (bounce reflection with modulo arithmetic)
    """

    def test_bounce_boundary_multiple_reflections_positive(self):
        """Bounce should handle multiple reflections when agent moves far beyond boundary.

        Coverage target: lines 153-162 (complex reflection logic)
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="bounce",
            movement_delta=1.0,
            interaction_radius=0.5,
        )

        # Agent near right edge, move far right (multiple bounces)
        positions = torch.tensor([[9.0, 5.0]], dtype=torch.float32)
        # Move right by 25 units (would go to 34, way beyond 10)
        deltas = torch.tensor([[25.0, 0.0]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Position 34 in [0, 10]: normalized = 34, range_size = 10
        # 34 % 20 = 14, 14 >= 10, so reflect: 20 - 14 = 6
        assert 0.0 <= new_positions[0, 0] <= 10.0, "Should stay in bounds"
        assert torch.allclose(new_positions[0, 0], torch.tensor(6.0)), "Should bounce to position 6"

    def test_bounce_boundary_exceed_half_reflection(self):
        """Bounce should reflect when normalized position exceeds half range.

        Coverage target: lines 158-159 (exceed_half mask and reflection)
        """
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=5.0,
            boundary="bounce",
            movement_delta=1.0,
            interaction_radius=0.5,
        )

        # Agent at 4, move right by 3 (would go to 7, beyond 5)
        positions = torch.tensor([[4.0]], dtype=torch.float32)
        deltas = torch.tensor([[3.0]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Position 7 in [0, 5]: normalized = 7, range_size = 5
        # 7 % 10 = 7, 7 >= 5, so reflect: 10 - 7 = 3
        assert torch.allclose(new_positions[0, 0], torch.tensor(3.0)), "Should bounce to position 3"


class TestContinuousChebyshevDistance:
    """Test Chebyshev distance metric (L∞ norm).

    Coverage target: lines 183-184 (chebyshev metric)
    """

    def test_distance_chebyshev_2d(self):
        """Chebyshev distance should return max of absolute differences.

        Coverage target: lines 183-184
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="chebyshev",
        )

        pos1 = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        pos2 = torch.tensor([[3.0, 7.0]], dtype=torch.float32)

        distance = substrate.compute_distance(pos1, pos2)

        # max(|0-3|, |0-7|) = max(3, 7) = 7
        assert torch.allclose(distance, torch.tensor([7.0])), "Chebyshev should return max component"

    def test_distance_chebyshev_3d(self):
        """Chebyshev distance in 3D should return max across all dimensions.

        Coverage target: Chebyshev for 3D
        """
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="chebyshev",
        )

        pos1 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        pos2 = torch.tensor([[4.0, 6.0, 5.0]], dtype=torch.float32)

        distance = substrate.compute_distance(pos1, pos2)

        # max(|1-4|, |2-6|, |3-5|) = max(3, 4, 2) = 4
        assert torch.allclose(distance, torch.tensor([4.0])), "Chebyshev should return max component in 3D"


class TestContinuousGetValidNeighbors:
    """Test get_valid_neighbors() raises NotImplementedError.

    Coverage target: lines 356-392 (error path)
    """

    def test_get_valid_neighbors_raises_not_implemented_1d(self):
        """1D continuous should raise NotImplementedError for get_valid_neighbors.

        Continuous spaces have infinite neighbors, so enumeration is not meaningful.
        Coverage target: lines 389-392
        """
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        position = torch.tensor([5.0], dtype=torch.float32)

        with pytest.raises(NotImplementedError) as exc_info:
            substrate.get_valid_neighbors(position)

        error_msg = str(exc_info.value).lower()
        assert "continuous" in error_msg
        assert "positions" in error_msg or "neighbors" in error_msg

    def test_get_valid_neighbors_raises_not_implemented_2d(self):
        """2D continuous should raise NotImplementedError for get_valid_neighbors.

        Coverage target: lines 389-392
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        position = torch.tensor([5.0, 5.0], dtype=torch.float32)

        with pytest.raises(NotImplementedError):
            substrate.get_valid_neighbors(position)

    def test_get_valid_neighbors_raises_not_implemented_3d(self):
        """3D continuous should raise NotImplementedError for get_valid_neighbors.

        Coverage target: lines 389-392
        """
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        position = torch.tensor([5.0, 5.0, 5.0], dtype=torch.float32)

        with pytest.raises(NotImplementedError):
            substrate.get_valid_neighbors(position)


class TestContinuousNormalizePositions:
    """Test normalize_positions() method.

    Coverage target: lines 400-423 (normalize_positions)
    """

    def test_normalize_positions_1d(self):
        """Should normalize 1D positions to [0, 1] range.

        Coverage target: lines 419-423 (calls _encode_relative)
        """
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor([[0.0], [5.0], [10.0]], dtype=torch.float32)

        normalized = substrate.normalize_positions(positions)

        # Should be [0.0, 0.5, 1.0]
        expected = torch.tensor([[0.0], [0.5], [1.0]], dtype=torch.float32)
        assert torch.allclose(normalized, expected), "Should normalize to [0, 1] range"

    def test_normalize_positions_2d(self):
        """Should normalize 2D positions to [0, 1] range per dimension.

        Coverage target: normalize_positions for 2D
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=20.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor([[0.0, 0.0], [10.0, 20.0]], dtype=torch.float32)

        normalized = substrate.normalize_positions(positions)

        # Should be [[0.0, 0.0], [1.0, 1.0]]
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        assert torch.allclose(normalized, expected), "Should normalize each dimension independently"

    def test_normalize_positions_negative_bounds(self):
        """Should normalize positions with negative bounds correctly.

        Coverage target: Normalization with negative ranges
        """
        substrate = Continuous2DSubstrate(
            min_x=-5.0,
            max_x=5.0,
            min_y=-10.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor([[-5.0, -10.0], [0.0, 0.0], [5.0, 10.0]], dtype=torch.float32)

        normalized = substrate.normalize_positions(positions)

        # Min should map to 0, center to 0.5, max to 1
        expected = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=torch.float32)
        assert torch.allclose(normalized, expected), "Should handle negative bounds correctly"


class TestContinuousPartialObservation:
    """Test encode_partial_observation() error handling.

    Coverage target: lines 475-487 (NotImplementedError)
    """

    def test_encode_partial_observation_raises_not_implemented_1d(self):
        """Should raise NotImplementedError for 1D continuous POMDP.

        Partial observability doesn't make sense for continuous spaces.
        Coverage target: lines 483-487
        """
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor([[5.0]], dtype=torch.float32)

        with pytest.raises(NotImplementedError) as exc_info:
            substrate.encode_partial_observation(positions, {}, vision_range=2)

        error_msg = str(exc_info.value).lower()
        assert "continuous" in error_msg
        assert "partial" in error_msg or "pomdp" in error_msg

    def test_encode_partial_observation_raises_not_implemented_2d(self):
        """Should raise NotImplementedError for 2D continuous POMDP.

        Coverage target: lines 483-487
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

        with pytest.raises(NotImplementedError):
            substrate.encode_partial_observation(positions, {}, vision_range=2)

    def test_encode_partial_observation_raises_not_implemented_3d(self):
        """Should raise NotImplementedError for 3D continuous POMDP.

        Coverage target: lines 483-487
        """
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32)

        with pytest.raises(NotImplementedError):
            substrate.encode_partial_observation(positions, {}, vision_range=2)


class TestContinuousEncodingErrorPaths:
    """Test error handling for invalid observation encodings.

    Coverage targets:
    - line 285 (invalid encoding in encode_observation)
    - line 302 (invalid encoding in get_observation_dim)
    """

    def test_encode_observation_invalid_encoding_mode(self):
        """Should raise ValueError for invalid observation_encoding.

        Coverage target: line 285
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            observation_encoding="relative",
        )

        # Force invalid encoding
        substrate.observation_encoding = "INVALID"

        positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

        with pytest.raises(ValueError) as exc_info:
            substrate.encode_observation(positions, {})

        assert "Invalid observation_encoding" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)

    def test_get_observation_dim_invalid_encoding_mode(self):
        """Should raise ValueError for invalid encoding in get_observation_dim.

        Coverage target: line 302
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            observation_encoding="relative",
        )

        # Force invalid encoding
        substrate.observation_encoding = "INVALID"

        with pytest.raises(ValueError) as exc_info:
            substrate.get_observation_dim()

        assert "Invalid observation_encoding" in str(exc_info.value)


class TestContinuousCoordinateSemantics:
    """Test coordinate_semantics property.

    Coverage target: lines 114-117 (coordinate_semantics property)
    """

    def test_coordinate_semantics_1d(self):
        """1D should return X: position semantics.

        Coverage target: lines 116 (1D case)
        """
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        semantics = substrate.coordinate_semantics

        assert "X" in semantics
        assert semantics["X"] == "position"

    def test_coordinate_semantics_2d(self):
        """2D should return X: horizontal, Y: vertical semantics.

        Coverage target: lines 116 (2D case)
        """
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        semantics = substrate.coordinate_semantics

        assert "X" in semantics
        assert "Y" in semantics
        assert semantics["X"] == "horizontal"
        assert semantics["Y"] == "vertical"

    def test_coordinate_semantics_3d(self):
        """3D should return X, Y, Z semantics.

        Coverage target: lines 116 (3D case)
        """
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        semantics = substrate.coordinate_semantics

        assert "X" in semantics
        assert "Y" in semantics
        assert "Z" in semantics
        assert semantics["X"] == "horizontal"
        assert semantics["Y"] == "vertical"
        assert semantics["Z"] == "depth"
