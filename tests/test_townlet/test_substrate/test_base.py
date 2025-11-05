"""Tests for SpatialSubstrate base class contracts."""

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate


class TestActionSpaceSizeProperty:
    """Test that all substrates implement action_space_size property."""

    def test_grid2d_action_space_size(self):
        """Grid2D has 5 actions (UP/DOWN/LEFT/RIGHT/INTERACT)."""
        substrate = Grid2DSubstrate(
            width=8,
            height=8,
            boundary="clamp",
            distance_metric="manhattan",
        )
        assert substrate.action_space_size == 5
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_grid3d_action_space_size(self):
        """Grid3D has 7 actions (±X/±Y/±Z/INTERACT)."""
        substrate = Grid3DSubstrate(
            width=8,
            height=8,
            depth=3,
            boundary="clamp",
            distance_metric="manhattan",
        )
        assert substrate.action_space_size == 7
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_continuous1d_action_space_size(self):
        """Continuous1D has 3 actions (±X/INTERACT)."""
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
            distance_metric="euclidean",
        )
        assert substrate.action_space_size == 3
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_continuous2d_action_space_size(self):
        """Continuous2D has 5 actions (±X/±Y/INTERACT)."""
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
        assert substrate.action_space_size == 5
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_continuous3d_action_space_size(self):
        """Continuous3D has 7 actions (±X/±Y/±Z/INTERACT)."""
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
        assert substrate.action_space_size == 7
        assert substrate.action_space_size == 2 * substrate.position_dim + 1

    def test_aspatial_action_space_size(self):
        """Aspatial has 1 action (INTERACT only)."""
        substrate = AspatialSubstrate()
        assert substrate.action_space_size == 1
        # Aspatial is special: position_dim=0, but action_space_size=1 (not 2*0+1=1)
        # This is correct: aspatial only has INTERACT action

    def test_action_space_formula_consistency(self):
        """Verify 2N+1 formula holds for spatial substrates."""
        test_cases = [
            (Grid2DSubstrate(8, 8, "clamp", "manhattan"), 2, 5),
            (Grid3DSubstrate(8, 8, 3, "clamp", "manhattan"), 3, 7),
            (Continuous1DSubstrate(0.0, 10.0, "clamp", 0.5, 1.0), 1, 3),
            (Continuous2DSubstrate(0.0, 10.0, 0.0, 10.0, "clamp", 0.5, 1.0), 2, 5),
            (Continuous3DSubstrate(0.0, 10.0, 0.0, 10.0, 0.0, 10.0, "clamp", 0.5, 1.0), 3, 7),
        ]

        for substrate, expected_dim, expected_actions in test_cases:
            assert substrate.position_dim == expected_dim
            assert substrate.action_space_size == expected_actions
            assert substrate.action_space_size == 2 * substrate.position_dim + 1
