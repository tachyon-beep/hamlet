"""Unit tests for continuous substrates (TASK-002A Phase 5B.2)."""

import pytest
import torch

from townlet.substrate.continuous import (
    Continuous1DSubstrate,
    Continuous2DSubstrate,
    Continuous3DSubstrate,
    ContinuousSubstrate,
)


class TestContinuousInitialization:
    """Test substrate initialization and validation."""

    def test_continuous1d_initialization_valid(self):
        """Initialize 1D continuous substrate with valid parameters."""
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        assert substrate.dimensions == 1
        assert substrate.position_dim == 1
        assert substrate.position_dtype == torch.float32
        assert substrate.movement_delta == 0.5
        assert substrate.interaction_radius == 0.8

    def test_continuous2d_initialization_valid(self):
        """Initialize 2D continuous substrate with valid parameters."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        assert substrate.dimensions == 2
        assert substrate.position_dim == 2
        assert substrate.bounds == [(0.0, 10.0), (0.0, 10.0)]

    def test_continuous3d_initialization_valid(self):
        """Initialize 3D continuous substrate with valid parameters."""
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=5.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        assert substrate.dimensions == 3
        assert substrate.position_dim == 3
        assert substrate.bounds == [(0.0, 10.0), (0.0, 10.0), (0.0, 5.0)]

    def test_initialization_invalid_dimensions(self):
        """Reject invalid dimension counts."""
        with pytest.raises(ValueError, match="support 1-3 dimensions"):
            ContinuousSubstrate(
                dimensions=4,
                bounds=[(0.0, 10.0)] * 4,
                boundary="clamp",
                movement_delta=0.5,
                interaction_radius=0.8,
            )

    def test_initialization_bounds_mismatch(self):
        """Reject bounds count mismatch with dimensions."""
        with pytest.raises(ValueError, match="bounds.*must match dimensions"):
            ContinuousSubstrate(
                dimensions=2,
                bounds=[(0.0, 10.0)],  # Only 1 bound for 2D
                boundary="clamp",
                movement_delta=0.5,
                interaction_radius=0.8,
            )

    def test_initialization_invalid_bounds(self):
        """Reject invalid bounds (min >= max)."""
        with pytest.raises(ValueError, match="min.*must be < max"):
            Continuous1DSubstrate(
                min_x=10.0,
                max_x=5.0,  # Invalid: min > max
                boundary="clamp",
                movement_delta=0.5,
                interaction_radius=0.8,
            )

    def test_initialization_invalid_boundary_mode(self):
        """Reject invalid boundary mode."""
        with pytest.raises(ValueError, match="Unknown boundary mode"):
            Continuous1DSubstrate(
                min_x=0.0,
                max_x=10.0,
                boundary="invalid",
                movement_delta=0.5,
                interaction_radius=0.8,
            )

    def test_initialization_negative_movement_delta(self):
        """Reject negative movement_delta."""
        with pytest.raises(ValueError, match="movement_delta must be positive"):
            Continuous1DSubstrate(
                min_x=0.0,
                max_x=10.0,
                boundary="clamp",
                movement_delta=-0.5,
                interaction_radius=0.8,
            )

    def test_initialization_small_space_for_interaction(self):
        """Reject space too small for interaction_radius."""
        with pytest.raises(ValueError, match="range.*< interaction_radius"):
            Continuous1DSubstrate(
                min_x=0.0,
                max_x=0.5,  # Range = 0.5
                boundary="clamp",
                movement_delta=0.1,
                interaction_radius=1.0,  # Radius > range
            )

    def test_initialization_warns_small_interaction_radius(self):
        """Warn if interaction_radius < movement_delta."""
        with pytest.warns(UserWarning, match="interaction_radius.*< movement_delta"):
            Continuous1DSubstrate(
                min_x=0.0,
                max_x=10.0,
                boundary="clamp",
                movement_delta=1.0,
                interaction_radius=0.5,  # Smaller than movement_delta
            )


class TestContinuousPositionInitialization:
    """Test position initialization."""

    def test_initialize_positions_1d_shape(self):
        """1D positions have correct shape [num_agents, 1]."""
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        positions = substrate.initialize_positions(5, torch.device("cpu"))
        assert positions.shape == (5, 1)
        assert positions.dtype == torch.float32

    def test_initialize_positions_2d_shape(self):
        """2D positions have correct shape [num_agents, 2]."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        positions = substrate.initialize_positions(10, torch.device("cpu"))
        assert positions.shape == (10, 2)
        assert positions.dtype == torch.float32

    def test_initialize_positions_in_bounds(self):
        """Initialized positions are within bounds."""
        substrate = Continuous2DSubstrate(
            min_x=2.0,
            max_x=8.0,
            min_y=-5.0,
            max_y=5.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        positions = substrate.initialize_positions(100, torch.device("cpu"))

        # Check X dimension
        assert (positions[:, 0] >= 2.0).all()
        assert (positions[:, 0] <= 8.0).all()

        # Check Y dimension
        assert (positions[:, 1] >= -5.0).all()
        assert (positions[:, 1] <= 5.0).all()


class TestContinuousMovement:
    """Test movement with different boundary modes."""

    def test_movement_basic(self):
        """Basic movement updates positions correctly."""
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        positions = torch.tensor([[5.0]], dtype=torch.float32)
        deltas = torch.tensor([[1.0]], dtype=torch.float32)  # Move right

        new_positions = substrate.apply_movement(positions, deltas)

        # Delta scaled by movement_delta: 1.0 * 0.5 = 0.5
        expected = torch.tensor([[5.5]], dtype=torch.float32)
        assert torch.allclose(new_positions, expected, atol=1e-5)

    def test_movement_clamp_boundary(self):
        """Clamp boundary prevents positions from leaving bounds."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=1.0,
            interaction_radius=0.8,
        )
        # Agent at edge trying to move outside
        positions = torch.tensor([[9.5, 5.0]], dtype=torch.float32)
        deltas = torch.tensor([[2.0, 0.0]], dtype=torch.float32)  # Try to move to 11.5

        new_positions = substrate.apply_movement(positions, deltas)

        # Should clamp to max (10.0)
        assert new_positions[0, 0] == 10.0
        assert new_positions[0, 1] == 5.0

    def test_movement_wrap_boundary(self):
        """Wrap boundary creates toroidal space."""
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="wrap",
            movement_delta=1.0,
            interaction_radius=0.8,
        )
        # Agent at 9.0 moving 2.0 right -> wraps to 1.0
        positions = torch.tensor([[9.0]], dtype=torch.float32)
        deltas = torch.tensor([[2.0]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # 9.0 + 2.0 = 11.0 -> wraps to 1.0
        expected = torch.tensor([[1.0]], dtype=torch.float32)
        assert torch.allclose(new_positions, expected, atol=1e-5)

    def test_movement_bounce_boundary(self):
        """Bounce boundary reflects movement."""
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="bounce",
            movement_delta=1.0,
            interaction_radius=0.8,
        )
        # Agent at 9.0 moving 2.0 right -> bounces back to 9.0
        positions = torch.tensor([[9.0]], dtype=torch.float32)
        deltas = torch.tensor([[2.0]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # 9.0 + 2.0 = 11.0 -> reflect to 9.0
        expected = torch.tensor([[9.0]], dtype=torch.float32)
        assert torch.allclose(new_positions, expected, atol=1e-5)

    def test_movement_sticky_boundary(self):
        """Sticky boundary keeps agent in place if would exit."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="sticky",
            movement_delta=1.0,
            interaction_radius=0.8,
        )
        # Agent at edge trying to move outside
        positions = torch.tensor([[9.5, 5.0]], dtype=torch.float32)
        deltas = torch.tensor([[2.0, 0.0]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Should stay in place (sticky)
        assert torch.allclose(new_positions, positions, atol=1e-5)

    def test_movement_batch(self):
        """Movement works for batch of agents."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        positions = torch.tensor([[1.0, 2.0], [5.0, 5.0], [9.0, 8.0]], dtype=torch.float32)
        deltas = torch.tensor([[1.0, 0.0], [0.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)

        new_positions = substrate.apply_movement(positions, deltas)

        # Each delta scaled by 0.5
        expected = torch.tensor([[1.5, 2.0], [5.0, 4.5], [8.5, 8.5]], dtype=torch.float32)
        assert torch.allclose(new_positions, expected, atol=1e-5)


class TestContinuousDistance:
    """Test distance computation."""

    def test_distance_euclidean(self):
        """Euclidean distance computed correctly."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
            distance_metric="euclidean",
        )
        pos1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        pos2 = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

        distances = substrate.compute_distance(pos1, pos2)

        # Distance from (0, 0) to (3, 4) = 5.0
        # Distance from (1, 1) to (3, 4) = sqrt((2)^2 + (3)^2) = sqrt(13) ≈ 3.606
        expected = torch.tensor([5.0, 3.606], dtype=torch.float32)
        assert torch.allclose(distances, expected, atol=1e-2)

    def test_distance_manhattan(self):
        """Manhattan distance computed correctly."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
            distance_metric="manhattan",
        )
        pos1 = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
        pos2 = torch.tensor([[3.0, 4.0]], dtype=torch.float32)

        distances = substrate.compute_distance(pos1, pos2)

        # Manhattan from (0, 0) to (3, 4) = |3-0| + |4-0| = 7.0
        # Manhattan from (1, 1) to (3, 4) = |3-1| + |4-1| = 5.0
        expected = torch.tensor([7.0, 5.0], dtype=torch.float32)
        assert torch.allclose(distances, expected, atol=1e-5)


class TestContinuousObservationEncoding:
    """Test observation encoding."""

    def test_encode_observation_shape(self):
        """Observation encoding has correct shape."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        positions = torch.tensor([[1.0, 2.0], [5.0, 5.0]], dtype=torch.float32)
        obs = substrate.encode_observation(positions, {})

        assert obs.shape == (2, 2)  # [num_agents, dimensions]
        assert obs.dtype == torch.float32

    def test_encode_observation_normalization(self):
        """Observation encoding normalizes to [0, 1]."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        # Position at min and max bounds
        positions = torch.tensor([[0.0, 0.0], [10.0, 10.0], [5.0, 2.5]], dtype=torch.float32)
        obs = substrate.encode_observation(positions, {})

        # (0, 0) -> (0.0, 0.0)
        assert torch.allclose(obs[0], torch.tensor([0.0, 0.0]), atol=1e-5)
        # (10, 10) -> (1.0, 1.0)
        assert torch.allclose(obs[1], torch.tensor([1.0, 1.0]), atol=1e-5)
        # (5, 2.5) -> (0.5, 0.25)
        assert torch.allclose(obs[2], torch.tensor([0.5, 0.25]), atol=1e-5)

    def test_get_observation_dim(self):
        """Observation dimension matches substrate dimensions."""
        substrate1d = Continuous1DSubstrate(min_x=0.0, max_x=10.0, boundary="clamp", movement_delta=0.5, interaction_radius=0.8)
        substrate2d = Continuous2DSubstrate(
            min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, boundary="clamp", movement_delta=0.5, interaction_radius=0.8
        )
        substrate3d = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        assert substrate1d.get_observation_dim() == 1
        assert substrate2d.get_observation_dim() == 2
        assert substrate3d.get_observation_dim() == 3


class TestContinuousInteraction:
    """Test proximity-based interaction detection."""

    def test_is_on_position_within_radius(self):
        """Agents within interaction_radius return True."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,  # 1.0 unit radius
        )
        # Agent at (5, 5), target at (5.5, 5.5)
        # Distance = sqrt(0.5^2 + 0.5^2) = 0.707 < 1.0
        positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
        target = torch.tensor([[5.5, 5.5]], dtype=torch.float32)

        result = substrate.is_on_position(positions, target)
        assert result[0]

    def test_is_on_position_outside_radius(self):
        """Agents outside interaction_radius return False."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )
        # Agent at (5, 5), target at (7, 7)
        # Distance = sqrt(4 + 4) = 2.828 > 1.0
        positions = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
        target = torch.tensor([[7.0, 7.0]], dtype=torch.float32)

        result = substrate.is_on_position(positions, target)
        assert not result[0]

    def test_is_on_position_batch(self):
        """Proximity detection works for batch of agents."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=1.0,
        )
        positions = torch.tensor([[5.0, 5.0], [5.5, 5.5], [8.0, 8.0]], dtype=torch.float32)
        target = torch.tensor([[5.0, 5.0]], dtype=torch.float32)

        result = substrate.is_on_position(positions, target)

        # Agent 0: distance = 0.0 (exact match)
        # Agent 1: distance = 0.707 < 1.0
        # Agent 2: distance = 4.24 > 1.0
        assert result[0]
        assert result[1]
        assert not result[2]


class TestContinuousUtilities:
    """Test utility methods."""

    def test_get_all_positions_raises_error(self):
        """Continuous substrates cannot enumerate positions."""
        substrate = Continuous1DSubstrate(min_x=0.0, max_x=10.0, boundary="clamp", movement_delta=0.5, interaction_radius=0.8)
        with pytest.raises(NotImplementedError, match="infinite positions"):
            substrate.get_all_positions()

    def test_get_valid_neighbors_raises_error(self):
        """Continuous substrates have no discrete neighbors."""
        substrate = Continuous2DSubstrate(
            min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, boundary="clamp", movement_delta=0.5, interaction_radius=0.8
        )
        position = torch.tensor([5.0, 5.0], dtype=torch.float32)
        with pytest.raises(NotImplementedError, match="continuous positions"):
            substrate.get_valid_neighbors(position)

    def test_supports_enumerable_positions(self):
        """Continuous substrates do not support enumerable positions."""
        substrate = Continuous2DSubstrate(
            min_x=0.0, max_x=10.0, min_y=0.0, max_y=10.0, boundary="clamp", movement_delta=0.5, interaction_radius=0.8
        )
        assert not substrate.supports_enumerable_positions()


class TestContinuousConfiguration:
    """Test configuration integration."""

    def test_config_1d(self):
        """1D continuous config loads correctly."""
        from pathlib import Path

        from townlet.substrate.config import load_substrate_config
        from townlet.substrate.factory import SubstrateFactory

        config_path = Path("configs/L1_continuous_1D/substrate.yaml")
        if not config_path.exists():
            pytest.skip("L1_continuous_1D config not found")

        config = load_substrate_config(config_path)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, Continuous1DSubstrate)
        assert substrate.position_dim == 1
        assert substrate.position_dtype == torch.float32

    def test_config_2d(self):
        """2D continuous config loads correctly."""
        from pathlib import Path

        from townlet.substrate.config import load_substrate_config
        from townlet.substrate.factory import SubstrateFactory

        config_path = Path("configs/L1_continuous_2D/substrate.yaml")
        if not config_path.exists():
            pytest.skip("L1_continuous_2D config not found")

        config = load_substrate_config(config_path)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, Continuous2DSubstrate)
        assert substrate.position_dim == 2
        assert substrate.position_dtype == torch.float32

    def test_config_3d(self):
        """3D continuous config loads correctly."""
        from pathlib import Path

        from townlet.substrate.config import load_substrate_config
        from townlet.substrate.factory import SubstrateFactory

        config_path = Path("configs/L1_continuous_3D/substrate.yaml")
        if not config_path.exists():
            pytest.skip("L1_continuous_3D config not found")

        config = load_substrate_config(config_path)
        substrate = SubstrateFactory.build(config, torch.device("cpu"))

        assert isinstance(substrate, Continuous3DSubstrate)
        assert substrate.position_dim == 3
        assert substrate.position_dtype == torch.float32


class TestDtypeIsolation:
    """Test that Grid (torch.long) and Continuous (torch.float32) don't contaminate each other."""

    def test_continuous_doesnt_contaminate_grid(self):
        """Verify continuous float32 operations don't affect grid long operations."""
        from townlet.substrate.grid2d import Grid2DSubstrate

        # Create both substrate types
        continuous = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        grid = Grid2DSubstrate(width=10, height=10, boundary="clamp", distance_metric="manhattan")

        # Continuous operations with float32
        cont_positions = continuous.initialize_positions(num_agents=5, device=torch.device("cpu"))
        cont_deltas = torch.tensor([[-0.5, 0.5], [0.5, -0.5], [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        cont_result = continuous.apply_movement(cont_positions, cont_deltas)

        # Verify continuous result is float32
        assert cont_result.dtype == torch.float32

        # Grid operations with long
        grid_positions = grid.initialize_positions(num_agents=5, device=torch.device("cpu"))
        grid_deltas = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], dtype=torch.float32)
        grid_result = grid.apply_movement(grid_positions, grid_deltas)

        # Verify grid result is still long (not contaminated by continuous float32)
        assert grid_result.dtype == torch.long

    def test_grid_doesnt_contaminate_continuous(self):
        """Verify grid long operations don't affect continuous float32 operations."""
        from townlet.substrate.grid2d import Grid2DSubstrate

        # Create both substrate types
        grid = Grid2DSubstrate(width=10, height=10, boundary="clamp", distance_metric="manhattan")
        continuous = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        # Grid operations with long
        grid_positions = grid.initialize_positions(num_agents=5, device=torch.device("cpu"))
        grid_deltas = torch.tensor([[-1, 0], [1, 0], [0, 1], [0, -1], [0, 0]], dtype=torch.float32)
        grid_result = grid.apply_movement(grid_positions, grid_deltas)

        # Verify grid result is long
        assert grid_result.dtype == torch.long

        # Continuous operations with float32
        cont_positions = continuous.initialize_positions(num_agents=5, device=torch.device("cpu"))
        cont_deltas = torch.tensor([[-0.5, 0.5], [0.5, -0.5], [0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        cont_result = continuous.apply_movement(cont_positions, cont_deltas)

        # Verify continuous result is still float32 (not contaminated by grid long)
        assert cont_result.dtype == torch.float32

    def test_mixed_substrate_environment_dtype_isolation(self):
        """Verify dtype isolation when grid and continuous substrates exist in same process."""
        from townlet.substrate.grid2d import Grid2DSubstrate
        from townlet.substrate.grid3d import Grid3DSubstrate

        # Create multiple substrate types in same process
        grid2d = Grid2DSubstrate(width=8, height=8, boundary="clamp", distance_metric="manhattan")
        grid3d = Grid3DSubstrate(width=8, height=8, depth=3, boundary="clamp", distance_metric="manhattan")
        continuous1d = Continuous1DSubstrate(min_x=0.0, max_x=10.0, boundary="clamp", movement_delta=0.5, interaction_radius=0.8)
        continuous2d = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )
        continuous3d = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="clamp",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        # Verify position_dtype is correct for each
        assert grid2d.position_dtype == torch.long
        assert grid3d.position_dtype == torch.long
        assert continuous1d.position_dtype == torch.float32
        assert continuous2d.position_dtype == torch.float32
        assert continuous3d.position_dtype == torch.float32

        # Verify position initialization respects dtype
        grid2d_pos = grid2d.initialize_positions(num_agents=3, device=torch.device("cpu"))
        grid3d_pos = grid3d.initialize_positions(num_agents=3, device=torch.device("cpu"))
        cont1d_pos = continuous1d.initialize_positions(num_agents=3, device=torch.device("cpu"))
        cont2d_pos = continuous2d.initialize_positions(num_agents=3, device=torch.device("cpu"))
        cont3d_pos = continuous3d.initialize_positions(num_agents=3, device=torch.device("cpu"))

        assert grid2d_pos.dtype == torch.long
        assert grid3d_pos.dtype == torch.long
        assert cont1d_pos.dtype == torch.float32
        assert cont2d_pos.dtype == torch.float32
        assert cont3d_pos.dtype == torch.float32


class TestBounceExactPositions:
    """Test exact bounce positions to verify reflection math correctness."""

    def test_bounce_from_lower_bound_x(self):
        """Test exact bounce position when hitting lower X bound."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="bounce",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        # Start near lower X bound, move left (will hit boundary)
        positions = torch.tensor([[0.2, 5.0]], dtype=torch.float32)
        deltas = torch.tensor([[-1.0, 0.0]], dtype=torch.float32)  # Move left by 0.5 units

        # Expected: X: 0.2 - 0.5 = -0.3 → reflect: abs(-0.3) = 0.3
        #           Y: 5.0 (unchanged)
        result = substrate.apply_movement(positions, deltas)

        assert torch.isclose(result[0, 0], torch.tensor(0.3), atol=1e-6)
        assert torch.isclose(result[0, 1], torch.tensor(5.0), atol=1e-6)

    def test_bounce_from_upper_bound_x(self):
        """Test exact bounce position when hitting upper X bound."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="bounce",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        # Start near upper X bound, move right (will hit boundary)
        positions = torch.tensor([[9.8, 5.0]], dtype=torch.float32)
        deltas = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # Move right by 0.5 units

        # Expected: X: 9.8 + 0.5 = 10.3 → reflect: 10.0 - (10.3 - 10.0) = 9.7
        #           Y: 5.0 (unchanged)
        result = substrate.apply_movement(positions, deltas)

        assert torch.isclose(result[0, 0], torch.tensor(9.7), atol=1e-6)
        assert torch.isclose(result[0, 1], torch.tensor(5.0), atol=1e-6)

    def test_bounce_from_lower_bound_y(self):
        """Test exact bounce position when hitting lower Y bound."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="bounce",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        # Start near lower Y bound, move down (will hit boundary)
        positions = torch.tensor([[5.0, 0.3]], dtype=torch.float32)
        deltas = torch.tensor([[0.0, -1.0]], dtype=torch.float32)  # Move down by 0.5 units

        # Expected: X: 5.0 (unchanged)
        #           Y: 0.3 - 0.5 = -0.2 → reflect: abs(-0.2) = 0.2
        result = substrate.apply_movement(positions, deltas)

        assert torch.isclose(result[0, 0], torch.tensor(5.0), atol=1e-6)
        assert torch.isclose(result[0, 1], torch.tensor(0.2), atol=1e-6)

    def test_bounce_from_upper_bound_y(self):
        """Test exact bounce position when hitting upper Y bound."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="bounce",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        # Start near upper Y bound, move up (will hit boundary)
        positions = torch.tensor([[5.0, 9.7]], dtype=torch.float32)
        deltas = torch.tensor([[0.0, 1.0]], dtype=torch.float32)  # Move up by 0.5 units

        # Expected: X: 5.0 (unchanged)
        #           Y: 9.7 + 0.5 = 10.2 → reflect: 10.0 - (10.2 - 10.0) = 9.8
        result = substrate.apply_movement(positions, deltas)

        assert torch.isclose(result[0, 0], torch.tensor(5.0), atol=1e-6)
        assert torch.isclose(result[0, 1], torch.tensor(9.8), atol=1e-6)

    def test_bounce_corner_both_bounds(self):
        """Test bounce when hitting corner (both X and Y bounds)."""
        substrate = Continuous2DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            boundary="bounce",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        # Start near lower-left corner, move diagonally (will hit both bounds)
        positions = torch.tensor([[0.2, 0.3]], dtype=torch.float32)
        deltas = torch.tensor([[-1.0, -1.0]], dtype=torch.float32)  # Move left-down by 0.5 units each

        # Expected: X: 0.2 - 0.5 = -0.3 → reflect: abs(-0.3) = 0.3
        #           Y: 0.3 - 0.5 = -0.2 → reflect: abs(-0.2) = 0.2
        result = substrate.apply_movement(positions, deltas)

        assert torch.isclose(result[0, 0], torch.tensor(0.3), atol=1e-6)
        assert torch.isclose(result[0, 1], torch.tensor(0.2), atol=1e-6)

    def test_bounce_multiple_reflections_1d(self):
        """Test bounce with large movement that crosses boundary multiple times."""
        substrate = Continuous1DSubstrate(
            min_x=0.0,
            max_x=10.0,
            boundary="bounce",
            movement_delta=3.0,
            interaction_radius=0.8,  # Large delta
        )

        # Start at 1.0, move left by 3.0 units (will bounce off lower bound)
        positions = torch.tensor([[1.0]], dtype=torch.float32)
        deltas = torch.tensor([[-1.0]], dtype=torch.float32)  # Move left by 3.0 units

        # Expected: 1.0 - 3.0 = -2.0
        # First reflection: abs(-2.0) = 2.0 (bounces back into bounds)
        result = substrate.apply_movement(positions, deltas)

        assert torch.isclose(result[0, 0], torch.tensor(2.0), atol=1e-6)

    def test_bounce_3d_all_dimensions(self):
        """Test bounce in 3D space with all dimensions reflecting."""
        substrate = Continuous3DSubstrate(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            min_z=0.0,
            max_z=10.0,
            boundary="bounce",
            movement_delta=0.5,
            interaction_radius=0.8,
        )

        # Start near lower bounds on all axes, move negative on all
        positions = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)
        deltas = torch.tensor([[-1.0, -1.0, -1.0]], dtype=torch.float32)

        # Expected: X: 0.2 - 0.5 = -0.3 → reflect: 0.3
        #           Y: 0.3 - 0.5 = -0.2 → reflect: 0.2
        #           Z: 0.4 - 0.5 = -0.1 → reflect: 0.1
        result = substrate.apply_movement(positions, deltas)

        assert torch.isclose(result[0, 0], torch.tensor(0.3), atol=1e-6)
        assert torch.isclose(result[0, 1], torch.tensor(0.2), atol=1e-6)
        assert torch.isclose(result[0, 2], torch.tensor(0.1), atol=1e-6)
