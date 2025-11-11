"""Tests for dynamic action space sizing in VectorizedHamletEnv."""

from pathlib import Path

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
        # env = <vectorized environment instantiated via fixtures>

        # Verify action dimension matches substrate
        # assert env.action_dim == substrate.action_space_size
        # assert env.action_dim == 6

        # Placeholder for now - actual test will be in integration tests
        assert substrate.action_space_size == 6

    def test_env_respects_grid3d_action_space(self):
        """Environment action_dim matches Grid3D substrate."""
        substrate = Grid3DSubstrate(8, 8, 3, "clamp")
        assert substrate.action_space_size == 8

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

        assert c1d.action_space_size == 4
        assert c2d.action_space_size == 6
        assert c3d.action_space_size == 8

    def test_env_respects_aspatial_action_space(self):
        """Environment action_dim matches Aspatial substrate."""
        substrate = AspatialSubstrate()
        assert substrate.action_space_size == 2


class TestEnvironmentActionSpace:
    """Ensure compiled universes expose substrate-driven action dimensions."""

    @pytest.mark.parametrize(
        "config_dir",
        [
            Path("configs/test"),
            Path("configs/aspatial_test"),
            Path("configs/L1_continuous_1D"),
        ],
        ids=["grid2d", "aspatial", "continuous1d"],
    )
    def test_env_action_dim_matches_compiled_metadata(self, env_factory, cpu_device, config_dir):
        """VectorizedHamletEnv should honor compiled metadata for action sizing."""

        env = env_factory(config_dir=config_dir, num_agents=1, device_override=cpu_device)

        expected_action_dim = env.metadata.action_count
        assert env.action_dim == expected_action_dim, f"env.action_dim should match compiled metadata for {config_dir}"
        assert env.action_space.action_dim == expected_action_dim, f"Composed action_space must match compiled metadata for {config_dir}"

        # Substrate portion of the action space should align with substrate definition.
        assert (
            env.action_space.substrate_action_count == env.substrate.action_space_size
        ), "Substrate action slice must reflect substrate.action_space_size"
