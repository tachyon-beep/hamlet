"""Comprehensive tests for ContinuousND substrate focusing on uncovered code paths.

This test file targets uncovered lines in continuousnd.py to push coverage from 88% → 95%+.
Focus areas:
- Lines 383-467: get_default_actions() method
- Lines 308, 325, 336: Edge case gaps
"""

import pytest
import torch

from townlet.substrate.continuousnd import ContinuousNDSubstrate

# =============================================================================
# ACTION GENERATION TESTS (Lines 383-467)
# =============================================================================


class TestContinuousNDActionGeneration:
    """Test get_default_actions() for N-dimensional continuous substrates."""

    def test_get_default_actions_4d_count(self):
        """4D continuous should generate 10 actions (8 movement + INTERACT + WAIT)."""
        substrate = ContinuousNDSubstrate(
            bounds=[
                (0.0, 10.0),
                (0.0, 10.0),
                (0.0, 10.0),
                (0.0, 10.0),
            ],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # 2*4 + 2 = 10 actions
        assert len(actions) == 10

        # Verify action IDs are sequential
        for i, action in enumerate(actions):
            assert action.id == i

    def test_get_default_actions_7d_count(self):
        """7D continuous should generate 16 actions (14 movement + INTERACT + WAIT)."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 7,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # 2*7 + 2 = 16 actions
        assert len(actions) == 16

    def test_get_default_actions_10d_count(self):
        """10D continuous should generate 22 actions (20 movement + INTERACT + WAIT)."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 10,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # 2*10 + 2 = 22 actions
        assert len(actions) == 22

    def test_get_default_actions_dimension_names_4d(self):
        """Action names should follow DIM{N}_NEG/POS pattern for 4D."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()
        names = [a.name for a in actions]

        # Expected pattern: DIM0_NEG, DIM0_POS, DIM1_NEG, DIM1_POS, ..., INTERACT, WAIT
        expected = [
            "DIM0_NEG",
            "DIM0_POS",
            "DIM1_NEG",
            "DIM1_POS",
            "DIM2_NEG",
            "DIM2_POS",
            "DIM3_NEG",
            "DIM3_POS",
            "INTERACT",
            "WAIT",
        ]
        assert names == expected

    def test_get_default_actions_deltas_4d(self):
        """Deltas should be ±1 scaled by movement_delta in correct dimension."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # DIM0_NEG should be [-1, 0, 0, 0]
        assert actions[0].delta == [-1, 0, 0, 0]
        # DIM0_POS should be [1, 0, 0, 0]
        assert actions[1].delta == [1, 0, 0, 0]
        # DIM1_NEG should be [0, -1, 0, 0]
        assert actions[2].delta == [0, -1, 0, 0]
        # DIM1_POS should be [0, 1, 0, 0]
        assert actions[3].delta == [0, 1, 0, 0]
        # DIM3_NEG should be [0, 0, 0, -1]
        assert actions[6].delta == [0, 0, 0, -1]
        # DIM3_POS should be [0, 0, 0, 1]
        assert actions[7].delta == [0, 0, 0, 1]

    def test_get_default_actions_interact_and_wait(self):
        """INTERACT and WAIT should be last two actions with correct properties."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # Second-to-last should be INTERACT
        interact = actions[-2]
        assert interact.name == "INTERACT"
        assert interact.type == "interaction"
        assert interact.delta is None
        assert interact.costs == {"energy": 0.003}
        assert interact.source == "substrate"

        # Last should be WAIT
        wait = actions[-1]
        assert wait.name == "WAIT"
        assert wait.type == "passive"
        assert wait.delta is None
        assert wait.costs == {"energy": 0.004}
        assert wait.source == "substrate"

    def test_get_default_actions_all_movement_costs(self):
        """All movement actions should have energy/hygiene/satiation costs."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # Check first 8 movement actions
        for action in actions[:8]:
            assert action.type == "movement"
            assert action.costs == {
                "energy": 0.005,
                "hygiene": 0.003,
                "satiation": 0.004,
            }
            assert action.source == "substrate"
            assert action.enabled is True

    def test_get_default_actions_description_includes_movement_delta(self):
        """Action descriptions should include the movement_delta value."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=2.5,  # Non-standard value
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # DIM0_NEG should mention -2.5
        assert "2.5" in actions[0].description
        assert "dimension 0" in actions[0].description.lower()

        # DIM0_POS should mention +2.5
        assert "2.5" in actions[1].description
        assert "dimension 0" in actions[1].description.lower()

    def test_get_default_actions_7d_all_dimensions_covered(self):
        """7D substrate should generate actions for all 7 dimensions."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 7,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        actions = substrate.get_default_actions()

        # Check that we have actions for dimensions 0-6
        names = [a.name for a in actions[:14]]  # Movement actions only
        for dim in range(7):
            assert f"DIM{dim}_NEG" in names
            assert f"DIM{dim}_POS" in names


# =============================================================================
# BOUNDARY EDGE CASES
# =============================================================================


class TestContinuousNDBounceEdgeCases:
    """Test complex bounce reflection logic for N-dimensional spaces."""

    def test_bounce_boundary_multiple_reflections_4d(self):
        """Bounce should handle multiple reflections in 4D space."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="bounce",
            movement_delta=1.0,
            interaction_radius=0.5,
        )

        # Position near boundary
        positions = torch.tensor([[9.0, 5.0, 5.0, 5.0]], dtype=torch.float32)

        # Move beyond boundary by 3 units (should reflect)
        deltas = torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        new_positions = substrate.apply_movement(positions, deltas)

        # x = 12 → reflects to x = 2*10 - 12 = 8
        assert torch.allclose(new_positions[0, 0], torch.tensor(8.0), atol=1e-5), f"Expected 8.0, got {new_positions[0, 0].item()}"

    def test_bounce_boundary_negative_direction_4d(self):
        """Bounce should handle negative reflections in 4D space."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="bounce",
            movement_delta=1.0,
            interaction_radius=0.5,
        )

        # Position near lower boundary
        positions = torch.tensor([[1.0, 5.0, 5.0, 5.0]], dtype=torch.float32)

        # Move beyond lower boundary
        deltas = torch.tensor([[-3.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        new_positions = substrate.apply_movement(positions, deltas)

        # x = -2 → reflects to x = 0 - (-2) = 2
        assert torch.allclose(new_positions[0, 0], torch.tensor(2.0), atol=1e-5), f"Expected 2.0, got {new_positions[0, 0].item()}"


# =============================================================================
# NORMALIZE POSITIONS
# =============================================================================


class TestContinuousNDNormalizePositions:
    """Test normalize_positions() for N-dimensional continuous substrates."""

    def test_normalize_positions_4d(self):
        """normalize_positions should map 4D positions to [0,1] range."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],  # Min corner
                [10.0, 10.0, 10.0, 10.0],  # Max corner
                [5.0, 2.5, 7.5, 1.0],  # Mixed
            ],
            dtype=torch.float32,
        )

        normalized = substrate.normalize_positions(positions)

        # Min corner → [0, 0, 0, 0]
        assert torch.allclose(normalized[0], torch.zeros(4))
        # Max corner → [1, 1, 1, 1]
        assert torch.allclose(normalized[1], torch.ones(4))
        # Mixed → [0.5, 0.25, 0.75, 0.1]
        assert torch.allclose(normalized[2], torch.tensor([0.5, 0.25, 0.75, 0.1]), atol=1e-5)

    def test_normalize_positions_asymmetric_bounds(self):
        """normalize_positions should handle asymmetric bounds correctly."""
        substrate = ContinuousNDSubstrate(
            bounds=[
                (0.0, 10.0),
                (-5.0, 5.0),
                (0.0, 100.0),
                (50.0, 150.0),
            ],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor(
            [
                [0.0, -5.0, 0.0, 50.0],  # Min corner
                [10.0, 5.0, 100.0, 150.0],  # Max corner
            ],
            dtype=torch.float32,
        )

        normalized = substrate.normalize_positions(positions)

        # All should map to [0, 0, 0, 0] and [1, 1, 1, 1]
        assert torch.allclose(normalized[0], torch.zeros(4))
        assert torch.allclose(normalized[1], torch.ones(4))

    def test_normalize_positions_negative_bounds(self):
        """normalize_positions should handle fully negative bounds."""
        substrate = ContinuousNDSubstrate(
            bounds=[
                (-10.0, -5.0),
                (-20.0, -10.0),
                (-100.0, 0.0),
                (-50.0, 50.0),
            ],
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )

        positions = torch.tensor(
            [
                [-10.0, -20.0, -100.0, -50.0],  # Min corner
                [-5.0, -10.0, 0.0, 50.0],  # Max corner
                [-7.5, -15.0, -50.0, 0.0],  # Middle
            ],
            dtype=torch.float32,
        )

        normalized = substrate.normalize_positions(positions)

        assert torch.allclose(normalized[0], torch.zeros(4))
        assert torch.allclose(normalized[1], torch.ones(4))
        assert torch.allclose(normalized[2], torch.tensor([0.5, 0.5, 0.5, 0.5]))


# =============================================================================
# ENCODING ERROR PATHS
# =============================================================================


class TestContinuousNDValidationErrorPaths:
    """Test parameter validation in constructor."""

    def test_constructor_validates_observation_encoding(self):
        """Constructor should raise ValueError for invalid observation encoding."""
        with pytest.raises(ValueError) as exc_info:
            ContinuousNDSubstrate(
                bounds=[(0.0, 10.0)] * 4,
                boundary="clamp",
                movement_delta=0.5,
                interaction_radius=1.0,
                observation_encoding="invalid_mode",  # Invalid
            )

        error_msg = str(exc_info.value).lower()
        assert "invalid_mode" in error_msg or "observation" in error_msg

    def test_constructor_validates_boundary_mode(self):
        """Constructor should raise ValueError for invalid boundary mode."""
        with pytest.raises(ValueError) as exc_info:
            ContinuousNDSubstrate(
                bounds=[(0.0, 10.0)] * 4,
                boundary="invalid_boundary",  # Invalid
                movement_delta=0.5,
                interaction_radius=1.0,
            )

        error_msg = str(exc_info.value).lower()
        assert "invalid_boundary" in error_msg or "boundary" in error_msg


# =============================================================================
# COORDINATE SEMANTICS
# =============================================================================


class TestContinuousNDCoordinateSemantics:
    """Test dimension semantics and coordinate system correctness."""

    def test_dimension_semantics_4d(self):
        """4D substrate should track 4 independent dimensions correctly."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.5,
        )

        # Move in each dimension independently
        base_position = torch.tensor([[5.0, 5.0, 5.0, 5.0]], dtype=torch.float32)

        # Move in dimension 0
        delta_dim0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        new_pos = substrate.apply_movement(base_position, delta_dim0)
        assert torch.allclose(new_pos, torch.tensor([[6.0, 5.0, 5.0, 5.0]]))

        # Move in dimension 3
        delta_dim3 = torch.tensor([[0.0, 0.0, 0.0, -2.0]], dtype=torch.float32)
        new_pos = substrate.apply_movement(base_position, delta_dim3)
        assert torch.allclose(new_pos, torch.tensor([[5.0, 5.0, 5.0, 3.0]]))

    def test_dimension_semantics_7d(self):
        """7D substrate should independently track 7 dimensions."""
        substrate = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 7,
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.5,
        )

        base_position = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]], dtype=torch.float32)

        # Move in dimension 6 (last dimension)
        delta_dim6 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]], dtype=torch.float32)
        new_pos = substrate.apply_movement(base_position, delta_dim6)
        expected = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 8.0]])
        assert torch.allclose(new_pos, expected)

    def test_observation_dim_consistency_across_dimensions(self):
        """get_observation_dim() should scale linearly with dimensionality."""
        # 4D substrate with "relative" encoding → 4 dims
        substrate_4d = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.5,
            observation_encoding="relative",
        )
        assert substrate_4d.get_observation_dim() == 4

        # 7D substrate with "relative" encoding → 7 dims
        substrate_7d = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 7,
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.5,
            observation_encoding="relative",
        )
        assert substrate_7d.get_observation_dim() == 7

        # 4D substrate with "scaled" encoding → 8 dims (4 coords + 4 bounds)
        substrate_4d_scaled = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 4,
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.5,
            observation_encoding="scaled",
        )
        assert substrate_4d_scaled.get_observation_dim() == 8

        # 7D substrate with "scaled" encoding → 14 dims (7 coords + 7 bounds)
        substrate_7d_scaled = ContinuousNDSubstrate(
            bounds=[(0.0, 10.0)] * 7,
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.5,
            observation_encoding="scaled",
        )
        assert substrate_7d_scaled.get_observation_dim() == 14
