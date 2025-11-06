"""Tests for SpatialSubstrate base class contracts."""

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate


class TestActionSpaceSizeProperty:
    """Test that all substrates implement action_space_size property."""

    def test_grid2d_action_space_size(self):
        """Grid2D has 6 actions (UP/DOWN/LEFT/RIGHT/INTERACT/WAIT)."""
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
        )
        assert substrate.action_space_size == 6
        assert substrate.action_space_size == 2 * substrate.position_dim + 2

    def test_grid3d_action_space_size(self):
        """Grid3D has 8 actions (±X/±Y/±Z/INTERACT/WAIT)."""
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=3,
            boundary="clamp",
            distance_metric="manhattan",
        )
        assert substrate.action_space_size == 8
        assert substrate.action_space_size == 2 * substrate.position_dim + 2

    def test_continuous1d_action_space_size(self):
        """Continuous1D has 4 actions (±X/INTERACT/WAIT)."""
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
        )
        assert substrate.action_space_size == 4
        assert substrate.action_space_size == 2 * substrate.position_dim + 2

    def test_continuous2d_action_space_size(self):
        """Continuous2D has 6 actions (±X/±Y/INTERACT/WAIT)."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
        )
        assert substrate.action_space_size == 6
        assert substrate.action_space_size == 2 * substrate.position_dim + 2

    def test_continuous3d_action_space_size(self):
        """Continuous3D has 8 actions (±X/±Y/±Z/INTERACT/WAIT)."""
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=5.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
        )
        assert substrate.action_space_size == 8
        assert substrate.action_space_size == 2 * substrate.position_dim + 2

    def test_aspatial_action_space_size(self):
        """Aspatial has 2 actions (INTERACT + WAIT)."""
        substrate = AspatialSubstrate()
        assert substrate.action_space_size == 2
        # Aspatial is special: position_dim=0, but action_space_size=2 (not 2*0+2=2)
        # This is correct: aspatial has INTERACT + WAIT actions

    def test_action_space_formula_consistency(self):
        """Verify 2N+2 formula holds for spatial substrates."""
        test_cases = [
            (Grid2DSubstrate(8, 8, "clamp", "manhattan"), 2, 6),
            (Grid3DSubstrate(8, 8, 3, "clamp", "manhattan"), 3, 8),
            (Continuous1DSubstrate(0.0, 10.0, "clamp", 0.5, 1.0), 1, 4),
            (Continuous2DSubstrate(0.0, 10.0, 0.0, 10.0, "clamp", 0.5, 1.0), 2, 6),
            (Continuous3DSubstrate(0.0, 10.0, 0.0, 10.0, 0.0, 10.0, "clamp", 0.5, 1.0), 3, 8),
        ]

        for substrate, expected_dim, expected_actions in test_cases:
            assert substrate.position_dim == expected_dim
            assert substrate.action_space_size == expected_actions
            assert substrate.action_space_size == 2 * substrate.position_dim + 2
