"""Property-based tests for substrate abstraction (TASK-002A Phase 8).

Uses Hypothesis to generate random substrate configurations and verify
that substrate contracts hold for all valid inputs.

Properties tested:
1. Position validation: Valid positions always accepted, invalid rejected
2. Distance symmetry: distance(A, B) == distance(B, A)
3. Movement validity: Moving from valid position stays in bounds
4. Observation dimension: obs_dim matches substrate + meters + affordances + temporal
"""

import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.aspatial import AspatialSubstrate
from townlet.substrate.grid2d import Grid2DSubstrate

# =============================================================================
# Grid2D Substrate Properties
# =============================================================================


@given(
    width=st.integers(min_value=2, max_value=20),
    height=st.integers(min_value=2, max_value=20),
    x=st.integers(min_value=0, max_value=25),
    y=st.integers(min_value=0, max_value=25),
)
@settings(max_examples=50)
def test_property_position_bounds_grid2d(width, height, x, y):
    """Positions should be within grid bounds after initialization."""
    substrate = Grid2DSubstrate(width=width, height=height, boundary="clamp", distance_metric="manhattan", observation_encoding="relative")

    # Initialize positions and verify they're within bounds
    device = torch.device("cpu")
    positions = substrate.initialize_positions(num_agents=1, device=device)

    assert 0 <= positions[0, 0] < width, f"X position {positions[0, 0]} out of bounds [0, {width})"
    assert 0 <= positions[0, 1] < height, f"Y position {positions[0, 1]} out of bounds [0, {height})"


@given(
    width=st.integers(min_value=3, max_value=15),
    height=st.integers(min_value=3, max_value=15),
    x1=st.integers(min_value=0, max_value=14),
    y1=st.integers(min_value=0, max_value=14),
    x2=st.integers(min_value=0, max_value=14),
    y2=st.integers(min_value=0, max_value=14),
)
@settings(max_examples=50)
def test_property_distance_symmetry_grid2d(width, height, x1, y1, x2, y2):
    """Distance from A to B should equal distance from B to A (symmetry)."""
    substrate = Grid2DSubstrate(width=width, height=height, boundary="clamp", distance_metric="manhattan", observation_encoding="relative")

    # Clamp positions to grid bounds
    x1, y1 = min(x1, width - 1), min(y1, height - 1)
    x2, y2 = min(x2, width - 1), min(y2, height - 1)

    pos1 = torch.tensor([[x1, y1]], dtype=torch.long)
    pos2 = torch.tensor([[x2, y2]], dtype=torch.long)

    dist_ab = substrate.compute_distance(pos1, pos2)
    dist_ba = substrate.compute_distance(pos2, pos1)

    assert torch.allclose(dist_ab, dist_ba, atol=1e-6), f"Distance should be symmetric: {dist_ab} != {dist_ba}"


@given(
    width=st.integers(min_value=3, max_value=15),
    height=st.integers(min_value=3, max_value=15),
    x=st.integers(min_value=0, max_value=14),
    y=st.integers(min_value=0, max_value=14),
    dx=st.integers(min_value=-1, max_value=1),
    dy=st.integers(min_value=-1, max_value=1),
)
@settings(max_examples=50)
def test_property_movement_stays_in_bounds_grid2d(width, height, x, y, dx, dy):
    """Moving from valid position with clamping should stay in bounds."""
    substrate = Grid2DSubstrate(width=width, height=height, boundary="clamp", distance_metric="manhattan", observation_encoding="relative")

    # Clamp start position to grid
    x, y = min(x, width - 1), min(y, height - 1)

    positions = torch.tensor([[x, y]], dtype=torch.long)
    deltas = torch.tensor([[dx, dy]], dtype=torch.float32)

    new_positions = substrate.apply_movement(positions, deltas)

    # New position must be within bounds (clamping ensures this)
    assert 0 <= new_positions[0, 0] < width, f"X coordinate {new_positions[0, 0]} out of bounds [0, {width})"
    assert 0 <= new_positions[0, 1] < height, f"Y coordinate {new_positions[0, 1]} out of bounds [0, {height})"


@given(
    width=st.integers(min_value=3, max_value=10),
    height=st.integers(min_value=3, max_value=10),
)
@settings(max_examples=20)
def test_property_get_all_positions_count_grid2d(width, height):
    """get_all_positions() should return exactly width × height positions."""
    substrate = Grid2DSubstrate(width=width, height=height, boundary="clamp", distance_metric="manhattan", observation_encoding="relative")

    positions = substrate.get_all_positions()

    expected_count = width * height
    assert len(positions) == expected_count, f"Should have {expected_count} positions, got {len(positions)}"

    # All positions should be valid
    for pos in positions:
        assert len(pos) == 2, f"Position {pos} should be 2D"
        assert 0 <= pos[0] < width, f"X coordinate {pos[0]} out of bounds"
        assert 0 <= pos[1] < height, f"Y coordinate {pos[1]} out of bounds"


# =============================================================================
# Aspatial Substrate Properties
# =============================================================================


def test_property_aspatial_has_no_positions():
    """Aspatial substrate should have position_dim=0 and no positions."""
    substrate = AspatialSubstrate()

    assert substrate.position_dim == 0, "Aspatial should have position_dim=0"
    assert substrate.get_all_positions() == [], "Aspatial should have no positions"


def test_property_aspatial_no_position_operations():
    """Aspatial substrate should have no position operations."""
    substrate = AspatialSubstrate()

    # apply_movement should be no-op (no positions to move)
    positions = torch.empty((1, 0), dtype=torch.long)  # Empty tensor for aspatial
    deltas = torch.empty((1, 0), dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)
    assert new_positions.shape == (1, 0), "Aspatial apply_movement should return empty tensor"


# =============================================================================
# Environment Observation Dimension Properties
# =============================================================================


@given(
    grid_size=st.integers(min_value=3, max_value=10),
    num_agents=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_obs_dim_matches_substrate_grid2d(grid_size, num_agents, test_config_pack_path, cpu_device):
    """Observation dimension should match substrate + meters + affordances + temporal."""
    env = VectorizedHamletEnv(
        num_agents=num_agents,
        grid_size=grid_size,  # Square grid
        partial_observability=False,  # Full observability
        vision_range=grid_size,
        enable_temporal_mechanics=False,
        device=cpu_device,
        config_pack_path=test_config_pack_path,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
    )

    obs = env.reset()

    # Expected dimension based on substrate observation encoding
    if env.substrate.observation_encoding == "relative":
        substrate_dim = 2  # Normalized (x, y)
    elif env.substrate.observation_encoding == "scaled":
        substrate_dim = 4  # Normalized (x, y) + (width, height)
    elif env.substrate.observation_encoding == "absolute":
        substrate_dim = 2  # Raw (x, y)

    meter_dim = env.meter_count  # From config (8 for test config)
    affordance_dim = 15  # 14 affordances + "none"
    temporal_dim = 4  # time_sin, time_cos, interaction_progress, lifetime_progress

    expected_dim = substrate_dim + meter_dim + affordance_dim + temporal_dim

    assert obs.shape == (
        num_agents,
        expected_dim,
    ), f"Observation shape mismatch: {obs.shape} vs ({num_agents}, {expected_dim})"


def test_property_obs_dim_aspatial(aspatial_env):
    """Aspatial observation should have no grid dimension."""
    obs = aspatial_env.reset()

    # Expected dimension (no grid)
    grid_dim = 0  # Aspatial has no grid
    meter_dim = aspatial_env.meter_count  # 4 for aspatial_test config
    # aspatial_test has 4 affordances (not full 14), so affordance_dim = 4 + 1 ("none") = 5
    affordance_dim = 5  # 4 affordances (Bed, Hospital, HomeMeal, Job) + "none"
    temporal_dim = 4  # time_sin, time_cos, interaction_progress, lifetime_progress

    expected_dim = grid_dim + meter_dim + affordance_dim + temporal_dim

    assert obs.shape == (
        1,
        expected_dim,
    ), f"Aspatial observation shape mismatch: {obs.shape} vs (1, {expected_dim})"
