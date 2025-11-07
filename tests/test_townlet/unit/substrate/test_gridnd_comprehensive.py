"""Comprehensive edge case and coverage tests for GridND substrate.

This module tests uncovered paths in GridND to improve coverage from 13% → target 70%+:

Coverage targets:
- src/townlet/substrate/gridnd.py:206-300 (get_default_actions for N-D)
- src/townlet/substrate/gridnd.py:158-169 (bounce boundary edge cases)
- src/townlet/substrate/gridnd.py:171-174 (sticky boundary)
- src/townlet/substrate/gridnd.py:441-482 (get_all_positions with memory warnings)
- src/townlet/substrate/gridnd.py:487-521 (encode_partial_observation error)
- src/townlet/substrate/gridnd.py:359, 376 (invalid encoding errors)
- src/townlet/substrate/gridnd.py:125-134 (initialize_positions)
- src/townlet/substrate/gridnd.py:202-204 (chebyshev distance)
"""

import warnings

import pytest
import torch

from townlet.substrate.gridnd import GridNDSubstrate


class TestGridNDActionGeneration:
    """Test get_default_actions() for N-dimensional action spaces.

    Coverage target: lines 206-300 (completely untested)
    """

    def test_get_default_actions_4d_count(self):
        """4D grid should generate 10 actions (8 movement + INTERACT + WAIT).

        Coverage target: Action generation for 4D
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
        )

        actions = substrate.get_default_actions()

        # 2*4 + 2 = 10 actions
        assert len(actions) == 10, "4D should have 10 actions"

        # Verify action IDs are sequential
        for i, action in enumerate(actions):
            assert action.id == i, f"Action {i} should have id={i}"

    def test_get_default_actions_7d_count(self):
        """7D grid should generate 16 actions (14 movement + INTERACT + WAIT).

        Coverage target: Action generation for 7D
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[3, 3, 3, 3, 3, 3, 3],
            boundary="clamp",
            distance_metric="manhattan",
        )

        actions = substrate.get_default_actions()

        # 2*7 + 2 = 16 actions
        assert len(actions) == 16, "7D should have 16 actions"

    def test_get_default_actions_movement_naming(self):
        """Movement actions should be named DIM{N}_NEG and DIM{N}_POS.

        Coverage target: lines 228, 249 (action naming)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
        )

        actions = substrate.get_default_actions()

        # First 8 actions are movement (4 dimensions × 2 directions)
        movement_actions = actions[:8]

        expected_names = [
            "DIM0_NEG",
            "DIM0_POS",
            "DIM1_NEG",
            "DIM1_POS",
            "DIM2_NEG",
            "DIM2_POS",
            "DIM3_NEG",
            "DIM3_POS",
        ]

        for i, (action, expected_name) in enumerate(zip(movement_actions, expected_names)):
            assert action.name == expected_name, f"Action {i} should be named {expected_name}"
            assert action.type == "movement", f"Action {i} should be movement type"
            assert action.source == "substrate", f"Action {i} should be from substrate"

    def test_get_default_actions_deltas_correct(self):
        """Movement actions should have correct deltas for each dimension.

        Coverage target: lines 223-224, 244-245 (delta assignment)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
        )

        actions = substrate.get_default_actions()

        # Check DIM0_NEG (index 0): delta should be [-1, 0, 0, 0]
        assert actions[0].delta == [-1, 0, 0, 0], "DIM0_NEG delta incorrect"

        # Check DIM0_POS (index 1): delta should be [1, 0, 0, 0]
        assert actions[1].delta == [1, 0, 0, 0], "DIM0_POS delta incorrect"

        # Check DIM3_NEG (index 6): delta should be [0, 0, 0, -1]
        assert actions[6].delta == [0, 0, 0, -1], "DIM3_NEG delta incorrect"

        # Check DIM3_POS (index 7): delta should be [0, 0, 0, 1]
        assert actions[7].delta == [0, 0, 0, 1], "DIM3_POS delta incorrect"

    def test_get_default_actions_interact_and_wait(self):
        """Last two actions should be INTERACT and WAIT.

        Coverage target: lines 265-298 (INTERACT and WAIT generation)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
        )

        actions = substrate.get_default_actions()

        # Second-to-last action: INTERACT
        interact = actions[-2]
        assert interact.name == "INTERACT"
        assert interact.type == "interaction"
        assert interact.delta is None
        assert "energy" in interact.costs

        # Last action: WAIT
        wait = actions[-1]
        assert wait.name == "WAIT"
        assert wait.type == "passive"
        assert wait.delta is None
        assert "energy" in wait.costs

    def test_get_default_actions_costs_assigned(self):
        """All actions should have energy costs assigned.

        Coverage target: lines 232, 253, 272, 290 (cost assignment)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
        )

        actions = substrate.get_default_actions()

        # All actions should have energy cost
        for action in actions:
            assert "energy" in action.costs, f"{action.name} missing energy cost"
            assert action.costs["energy"] > 0, f"{action.name} energy cost should be positive"


class TestGridNDBoundaryEdgeCases:
    """Test boundary mode edge cases for N-dimensional grids.

    Coverage targets:
    - lines 158-169 (bounce boundary logic)
    - lines 171-174 (sticky boundary logic)
    """

    def test_bounce_boundary_negative_positions(self):
        """Bounce should reflect negative positions (abs value).

        Coverage target: lines 160-162 (negative mask bounce)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="bounce",
            distance_metric="manhattan",
        )

        # Move beyond all lower boundaries
        positions = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        deltas = torch.tensor([[-3, -2, -4, -1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # -2 → abs(-2) = 2, -1 → abs(-1) = 1, -3 → abs(-3) = 3, 0 → 0
        expected = torch.tensor([[2, 1, 3, 0]], dtype=torch.long)
        assert torch.equal(new_positions, expected), "Bounce should reflect negative positions"

    def test_bounce_boundary_exceed_upper(self):
        """Bounce should reflect positions beyond upper boundary.

        Coverage target: lines 164-166 (exceed mask bounce)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="bounce",
            distance_metric="manhattan",
        )

        # Move beyond upper boundaries (size=5, max index=4)
        positions = torch.tensor([[3, 3, 3, 3]], dtype=torch.long)
        deltas = torch.tensor([[2, 3, 1, 4]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # 3+2=5 → 2*(5-1)-5 = 8-5 = 3
        # 3+3=6 → 2*4-6 = 8-6 = 2
        # 3+1=4 → 4 (valid, no bounce)
        # 3+4=7 → 2*4-7 = 8-7 = 1
        expected = torch.tensor([[3, 2, 4, 1]], dtype=torch.long)
        assert torch.equal(new_positions, expected), "Bounce should reflect from upper boundary"

    def test_bounce_boundary_safety_clamp(self):
        """Bounce with large velocities should clamp after reflection.

        Coverage target: lines 168-169 (safety clamp)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="bounce",
            distance_metric="manhattan",
        )

        # Extreme deltas that could exceed bounds even after bounce
        positions = torch.tensor([[2, 2, 2, 2]], dtype=torch.long)
        deltas = torch.tensor([[100, -100, 50, -50]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Safety clamp ensures all values are [0, 4]
        assert (new_positions >= 0).all(), "Bounce should clamp to min 0"
        assert (new_positions <= 4).all(), "Bounce should clamp to max 4"

    def test_sticky_boundary_all_dimensions(self):
        """Sticky boundary should prevent movement on out-of-bounds axes.

        Coverage target: lines 171-174 (sticky boundary logic)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="sticky",
            distance_metric="manhattan",
        )

        # At corner, try to move out in all dimensions
        positions = torch.tensor([[0, 0, 4, 4]], dtype=torch.long)
        deltas = torch.tensor([[-1, 1, 1, -1]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Dim 0: -1 out of bounds → stick at 0
        # Dim 1: +1 valid → move to 1
        # Dim 2: +1 (4+1=5) out of bounds → stick at 4
        # Dim 3: -1 valid → move to 3
        expected = torch.tensor([[0, 1, 4, 3]], dtype=torch.long)
        assert torch.equal(new_positions, expected), "Sticky should only block invalid axes"


class TestGridNDInitializePositions:
    """Test random position initialization.

    Coverage target: lines 125-134 (initialize_positions)
    """

    def test_initialize_positions_shape(self):
        """Should return [num_agents, N] tensor with correct dtype.

        Coverage target: lines 131-134 (tensor stacking and device)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[10, 8, 6, 5],
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = substrate.initialize_positions(num_agents=50, device=torch.device("cpu"))

        assert positions.shape == (50, 4), "Should return [num_agents, N] shape"
        assert positions.dtype == torch.long, "Should be long tensor"
        assert positions.device.type == "cpu", "Should be on correct device"

    def test_initialize_positions_in_bounds(self):
        """Initialized positions should be within grid bounds.

        Coverage target: Validate output of initialize_positions
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[10, 8, 6, 5],
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = substrate.initialize_positions(num_agents=100, device=torch.device("cpu"))

        # Check each dimension is within bounds
        for dim_idx, dim_size in enumerate([10, 8, 6, 5]):
            assert (positions[:, dim_idx] >= 0).all(), f"Dimension {dim_idx} has negative values"
            assert (positions[:, dim_idx] < dim_size).all(), f"Dimension {dim_idx} exceeds bounds"


class TestGridNDGetAllPositions:
    """Test get_all_positions() with memory warnings and errors.

    Coverage targets:
    - lines 458-482 (enumeration with warnings/errors)
    """

    def test_get_all_positions_small_grid(self):
        """Small grid should enumerate all positions without warnings.

        Coverage target: lines 458, 480-482 (normal enumeration)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[3, 3, 3, 3],  # 3^4 = 81 positions
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Should not warn or error
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Treat warnings as errors
            positions = substrate.get_all_positions()

        assert len(positions) == 81, "Should enumerate all 81 positions"

        # Verify first and last positions
        assert positions[0] == [0, 0, 0, 0]
        assert positions[-1] == [2, 2, 2, 2]

    def test_get_all_positions_warns_large_count(self):
        """Should warn when total positions > 100,000.

        Coverage target: lines 473-478 (warning path)
        """
        # Use a grid that exceeds 100,000 threshold but is below 10 million error threshold
        substrate_large = GridNDSubstrate(
            dimension_sizes=[20, 20, 20, 20],  # 20^4 = 160,000 positions
            boundary="clamp",
            distance_metric="manhattan",
        )

        with pytest.warns(UserWarning, match="160,000 positions"):
            positions = substrate_large.get_all_positions()

        assert len(positions) == 160_000, "Should generate all positions despite warning"

    def test_get_all_positions_raises_memory_error(self):
        """Should raise MemoryError when total positions > 10 million.

        Coverage target: lines 465-471 (memory error path)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[101, 100, 100, 10],  # 101*100^2*10 = 10,100,000 (exceeds threshold)
            boundary="clamp",
            distance_metric="manhattan",
        )

        with pytest.raises(MemoryError, match="10,100,000 positions"):
            substrate.get_all_positions()


class TestGridNDPartialObservation:
    """Test encode_partial_observation() error handling.

    Coverage target: lines 487-521 (NotImplementedError)
    """

    def test_encode_partial_observation_raises_not_implemented(self):
        """Should raise NotImplementedError for N≥4 dimensions.

        POMDP with local windows is impractical for high-D grids.
        Coverage target: lines 516-521 (error message)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
        )

        positions = torch.tensor([[2, 2, 2, 2]], dtype=torch.long)
        affordances = {}

        with pytest.raises(NotImplementedError) as exc_info:
            substrate.encode_partial_observation(positions, affordances, vision_range=2)

        error_msg = str(exc_info.value)
        assert "not supported" in error_msg.lower()
        assert "4D" in error_msg
        assert "625 cells" in error_msg  # (2*2+1)^4 = 5^4 = 625


class TestGridNDEncodingErrorPaths:
    """Test error handling for invalid observation encodings.

    Coverage targets:
    - line 359 (invalid encoding in encode_observation)
    - line 376 (invalid encoding in get_observation_dim)
    """

    def test_encode_observation_invalid_encoding_mode(self):
        """Should raise ValueError for invalid observation_encoding.

        Coverage target: line 359
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        # Force invalid encoding
        substrate.observation_encoding = "INVALID"

        positions = torch.tensor([[2, 2, 2, 2]], dtype=torch.long)

        with pytest.raises(ValueError) as exc_info:
            substrate.encode_observation(positions, {})

        assert "Invalid observation_encoding" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)

    def test_get_observation_dim_invalid_encoding_mode(self):
        """Should raise ValueError for invalid encoding in get_observation_dim.

        Coverage target: line 376
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[5, 5, 5, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="relative",
        )

        # Force invalid encoding
        substrate.observation_encoding = "INVALID"

        with pytest.raises(ValueError) as exc_info:
            substrate.get_observation_dim()

        assert "Invalid observation_encoding" in str(exc_info.value)


class TestGridNDDistanceEdgeCases:
    """Test distance metric edge cases.

    Coverage target: lines 202-204 (chebyshev metric)
    """

    def test_distance_chebyshev_4d(self):
        """Chebyshev distance should return max absolute difference.

        Coverage target: lines 202-204
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[10, 10, 10, 10],
            boundary="clamp",
            distance_metric="chebyshev",
        )

        pos1 = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        pos2 = torch.tensor([[3, 7, 2, 5]], dtype=torch.long)

        distance = substrate.compute_distance(pos1, pos2)

        # max(|3-0|, |7-0|, |2-0|, |5-0|) = max(3, 7, 2, 5) = 7
        assert distance[0] == 7, "Chebyshev should return max component"

    def test_distance_broadcasting_single_target(self):
        """Should handle broadcasting when pos2 is single position [N].

        Coverage target: lines 191-192 (unsqueeze for broadcasting)
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[10, 10, 10, 10],
            boundary="clamp",
            distance_metric="manhattan",
        )

        # Multiple agents
        pos1 = torch.tensor([[0, 0, 0, 0], [5, 5, 5, 5], [3, 3, 3, 3]], dtype=torch.long)

        # Single target [N] (will be broadcast)
        pos2 = torch.tensor([5, 5, 5, 5], dtype=torch.long)

        distances = substrate.compute_distance(pos1, pos2)

        assert distances.shape == (3,), "Should return distance for each agent"
        assert distances[0].item() == 20, "Distance [0,0,0,0] to [5,5,5,5]"
        assert distances[1].item() == 0, "Distance [5,5,5,5] to [5,5,5,5]"
        assert distances[2].item() == 8, "Distance [3,3,3,3] to [5,5,5,5]"


class TestGridNDValidationEdgeCases:
    """Test validation edge cases and error handling.

    Coverage for initialization validation
    """

    def test_exceeds_100_dimensions_raises_error(self):
        """Should raise ValueError when dimensions > 100.

        Coverage target: lines 64-65
        """
        with pytest.raises(ValueError, match="exceeds limit"):
            GridNDSubstrate(
                dimension_sizes=[5] * 101,  # 101 dimensions
                boundary="clamp",
                distance_metric="manhattan",
            )

    def test_invalid_boundary_mode_raises_error(self):
        """Should raise ValueError for unsupported boundary mode.

        Coverage target: lines 82-83
        """
        with pytest.raises(ValueError, match="boundary mode"):
            GridNDSubstrate(
                dimension_sizes=[5, 5, 5, 5],
                boundary="INVALID",
                distance_metric="manhattan",
            )

    def test_invalid_distance_metric_raises_error(self):
        """Should raise ValueError for unsupported distance metric.

        Coverage target: lines 85-86
        """
        with pytest.raises(ValueError, match="distance metric"):
            GridNDSubstrate(
                dimension_sizes=[5, 5, 5, 5],
                boundary="clamp",
                distance_metric="INVALID",
            )

    def test_invalid_observation_encoding_raises_error(self):
        """Should raise ValueError for unsupported observation encoding.

        Coverage target: lines 88-89
        """
        with pytest.raises(ValueError, match="observation encoding"):
            GridNDSubstrate(
                dimension_sizes=[5, 5, 5, 5],
                boundary="clamp",
                distance_metric="manhattan",
                observation_encoding="INVALID",
            )


class TestGridNDAbsoluteEncoding:
    """Test absolute encoding mode.

    Coverage target: lines 335-337 (_encode_absolute)
    """

    def test_absolute_encoding_returns_raw_coordinates(self):
        """Absolute encoding should return raw unnormalized coordinates.

        Coverage target: lines 335-337
        """
        substrate = GridNDSubstrate(
            dimension_sizes=[10, 8, 6, 5],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="absolute",
        )

        positions = torch.tensor([[7, 6, 4, 3]], dtype=torch.long)

        encoded = substrate.encode_observation(positions, {})

        # Should return raw coordinates as float
        expected = torch.tensor([[7.0, 6.0, 4.0, 3.0]], dtype=torch.float32)
        assert torch.allclose(encoded, expected), "Absolute encoding should return raw coords"
        assert substrate.get_observation_dim() == 4, "Absolute has N dimensions"


class TestGridNDNormalizePositions:
    """Test normalize_positions() method.

    Coverage target: lines 378-387 (normalize_positions calls _encode_relative)
    """

    def test_normalize_positions_always_relative(self):
        """normalize_positions() should always use relative encoding.

        Coverage target: lines 387 (calls _encode_relative)
        """
        # Even with "scaled" encoding, normalize should use relative
        substrate = GridNDSubstrate(
            dimension_sizes=[10, 10, 10, 10],
            boundary="clamp",
            distance_metric="manhattan",
            observation_encoding="scaled",  # Note: scaled mode
        )

        positions = torch.tensor([[0, 0, 0, 0], [9, 9, 9, 9]], dtype=torch.long)

        normalized = substrate.normalize_positions(positions)

        # Should be [0,0,0,0] and [1,1,1,1] regardless of encoding mode
        assert torch.allclose(normalized[0], torch.zeros(4))
        assert torch.allclose(normalized[1], torch.ones(4))
        assert normalized.shape == (2, 4), "Should return N dimensions (not 2N)"
