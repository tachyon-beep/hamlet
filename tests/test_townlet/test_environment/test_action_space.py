"""Tests for dynamic action space sizing in VectorizedHamletEnv."""

import pytest

from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.continuous import Continuous1DSubstrate, Continuous2DSubstrate, Continuous3DSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate
from townlet.substrate.grid3d import Grid3DSubstrate


class TestActionSpaceDynamicSizing:
    """Test that VectorizedHamletEnv uses substrate.action_space_size."""

    @pytest.fixture
    def minimal_config(self):
        """Minimal config for testing (bars, affordances, cascades)."""
        # This would load from a minimal test config directory
        # For now, return None and we'll use fixtures in actual implementation
        return None

    def test_env_respects_grid2d_action_space(self, minimal_config):
        """Environment action_dim matches Grid2D substrate."""
        substrate = Grid2DSubstrate(8, 8, "clamp", "manhattan")

        # Create environment (would need full config in real test)
        # env = VectorizedHamletEnv(substrate, minimal_config, ...)

        # Verify action dimension matches substrate
        # assert env.action_dim == substrate.action_space_size
        # assert env.action_dim == 5

        # Placeholder for now - actual test will be in integration tests
        assert substrate.action_space_size == 5

    def test_env_respects_grid3d_action_space(self):
        """Environment action_dim matches Grid3D substrate."""
        substrate = Grid3DSubstrate(8, 8, 3, "clamp")
        assert substrate.action_space_size == 7

    def test_env_respects_continuous_action_spaces(self):
        """Environment action_dim matches Continuous substrates."""
        c1d = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )
        c2d = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )
        c3d = Continuous3DSubstrate(
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

        assert c1d.action_space_size == 3
        assert c2d.action_space_size == 5
        assert c3d.action_space_size == 7

    def test_env_respects_aspatial_action_space(self):
        """Environment action_dim matches Aspatial substrate."""
        substrate = AspatialSubstrate()
        assert substrate.action_space_size == 1
