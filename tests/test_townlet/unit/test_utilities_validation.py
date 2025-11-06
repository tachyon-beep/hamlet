"""Validation tests for test utilities (builders and assertions).

This test file validates that the test utilities in utils/ work correctly.
"""

import pytest
import torch

from ..utils.assertions import (
    assert_meters_in_range,
    assert_positions_in_bounds,
    assert_valid_action_mask,
    assert_valid_dones,
    assert_valid_observation,
    assert_valid_rewards,
)
from ..utils.builders import (
    make_bars_config,
    make_grid2d_substrate,
    make_grid3d_substrate,
    make_positions,
    make_standard_8meter_config,
)

# =============================================================================
# BUILDER TESTS
# =============================================================================


def test_make_grid2d_substrate_defaults():
    """Test make_grid2d_substrate with default parameters."""
    substrate = make_grid2d_substrate()

    assert substrate.width == 8
    assert substrate.height == 8
    assert substrate.boundary == "clamp"
    assert substrate.distance_metric == "manhattan"
    assert substrate.observation_encoding == "relative"
    assert substrate.position_dim == 2


def test_make_grid2d_substrate_custom():
    """Test make_grid2d_substrate with custom parameters."""
    substrate = make_grid2d_substrate(
        width=3,
        height=5,
        boundary="wrap",
        distance_metric="euclidean",
        observation_encoding="scaled",
    )

    assert substrate.width == 3
    assert substrate.height == 5
    assert substrate.boundary == "wrap"
    assert substrate.distance_metric == "euclidean"
    assert substrate.observation_encoding == "scaled"


def test_make_grid3d_substrate_defaults():
    """Test make_grid3d_substrate with default parameters."""
    substrate = make_grid3d_substrate()

    assert substrate.width == 5
    assert substrate.height == 5
    assert substrate.depth == 3
    assert substrate.boundary == "clamp"
    assert substrate.distance_metric == "manhattan"
    assert substrate.observation_encoding == "relative"
    assert substrate.position_dim == 3


def test_make_grid3d_substrate_custom():
    """Test make_grid3d_substrate with custom parameters."""
    substrate = make_grid3d_substrate(
        width=4,
        height=4,
        depth=2,
        boundary="bounce",
        distance_metric="chebyshev",
        observation_encoding="absolute",
    )

    assert substrate.width == 4
    assert substrate.height == 4
    assert substrate.depth == 2
    assert substrate.boundary == "bounce"
    assert substrate.distance_metric == "chebyshev"
    assert substrate.observation_encoding == "absolute"


def test_make_bars_config_defaults():
    """Test make_bars_config with default parameters."""
    config = make_bars_config()

    assert config.meter_count == 8
    assert len(config.bars) == 8
    assert config.version == "2.0"
    assert config.description == "Test universe"


def test_make_bars_config_4_meters():
    """Test make_bars_config with 4 meters."""
    config = make_bars_config(meter_count=4)

    assert config.meter_count == 4
    assert len(config.bars) == 4
    assert config.meter_names == ["meter_0", "meter_1", "meter_2", "meter_3"]

    # First two are pivotal
    assert config.bars[0].tier == "pivotal"
    assert config.bars[1].tier == "pivotal"
    assert config.bars[2].tier == "secondary"
    assert config.bars[3].tier == "secondary"

    # Terminal conditions for first two meters
    assert len(config.terminal_conditions) == 2
    assert config.terminal_conditions[0].meter == "meter_0"
    assert config.terminal_conditions[1].meter == "meter_1"


def test_make_bars_config_validation():
    """Test make_bars_config validation."""
    # Should raise for invalid meter counts
    with pytest.raises(ValueError, match="meter_count must be >= 1"):
        make_bars_config(meter_count=0)

    with pytest.raises(ValueError, match="meter_count must be <= 32"):
        make_bars_config(meter_count=33)


def test_make_standard_8meter_config():
    """Test make_standard_8meter_config."""
    config = make_standard_8meter_config()

    assert config.meter_count == 8
    assert config.meter_names == [
        "energy",
        "health",
        "satiation",
        "money",
        "mood",
        "social",
        "fitness",
        "hygiene",
    ]

    # Check energy and health are pivotal
    assert config.bars[0].name == "energy"
    assert config.bars[0].tier == "pivotal"
    assert config.bars[1].name == "health"
    assert config.bars[1].tier == "pivotal"


def test_make_positions():
    """Test make_positions with various parameters."""
    # 2D positions
    positions = make_positions(num_agents=4, position_dim=2, value=0)
    assert positions.shape == (4, 2)
    assert positions.dtype == torch.long
    assert torch.all(positions == 0)

    # 3D positions with custom value
    positions = make_positions(num_agents=2, position_dim=3, value=5)
    assert positions.shape == (2, 3)
    assert torch.all(positions == 5)

    # Custom device
    if torch.cuda.is_available():
        positions = make_positions(num_agents=1, position_dim=2, device=torch.device("cuda"))
        assert positions.device.type == "cuda"


# =============================================================================
# ASSERTION TESTS
# =============================================================================


def test_assert_valid_observation_success(basic_env):
    """Test assert_valid_observation passes for valid observation."""
    obs = basic_env.reset()
    assert_valid_observation(basic_env, obs)  # Should not raise


def test_assert_valid_observation_invalid_shape():
    """Test assert_valid_observation detects invalid shape."""
    from unittest.mock import Mock

    env = Mock()
    env.num_agents = 4
    env.observation_dim = 10  # Use observation_dim property, not observation_builder

    # Wrong shape observation
    obs = torch.zeros(3, 10)  # Should be (4, 10)

    with pytest.raises(AssertionError, match="Invalid observation shape"):
        assert_valid_observation(env, obs)


def test_assert_valid_observation_invalid_dtype():
    """Test assert_valid_observation detects invalid dtype."""
    from unittest.mock import Mock

    env = Mock()
    env.num_agents = 1
    env.observation_dim = 10  # Use observation_dim property, not observation_builder

    # Wrong dtype observation (int instead of float32)
    obs = torch.zeros(1, 10, dtype=torch.int32)

    with pytest.raises(AssertionError, match="Invalid observation dtype"):
        assert_valid_observation(env, obs)


def test_assert_valid_observation_nan():
    """Test assert_valid_observation detects NaN values."""
    from unittest.mock import Mock

    env = Mock()
    env.num_agents = 1
    env.observation_dim = 10  # Use observation_dim property, not observation_builder

    # Observation with NaN
    obs = torch.zeros(1, 10, dtype=torch.float32)
    obs[0, 5] = float("nan")

    with pytest.raises(AssertionError, match="contains .* NaN values"):
        assert_valid_observation(env, obs)


def test_assert_valid_action_mask_success(basic_env):
    """Test assert_valid_action_mask passes for valid mask."""
    # Reset environment to ensure fresh state
    basic_env.reset()

    # Take a wait action (should keep agent alive)
    wait_action = torch.full((basic_env.num_agents,), 5, dtype=torch.long, device=basic_env.device)
    basic_env.step(wait_action)

    # Get action masks - agent should have at least movement actions available
    mask = basic_env.get_action_masks()
    assert_valid_action_mask(basic_env, mask)  # Should not raise


def test_assert_valid_action_mask_no_valid_actions():
    """Test assert_valid_action_mask detects agents with no valid actions."""
    from unittest.mock import Mock

    env = Mock()
    env.num_agents = 2
    env.action_dim = 5

    # Agent 1 has no valid actions
    mask = torch.tensor([[True, True, False, False, False], [False, False, False, False, False]], dtype=torch.bool)

    with pytest.raises(AssertionError, match="agents have no valid actions"):
        assert_valid_action_mask(env, mask)


def test_assert_meters_in_range_success(basic_env):
    """Test assert_meters_in_range passes for valid meters."""
    basic_env.reset()
    assert_meters_in_range(basic_env)  # Should not raise


def test_assert_meters_in_range_below_zero():
    """Test assert_meters_in_range detects meters below 0."""
    from unittest.mock import Mock

    env = Mock()
    env.meters = torch.tensor([[1.0, 0.5, -0.1, 0.8]], dtype=torch.float32)

    with pytest.raises(AssertionError, match="Meters below 0.0 detected"):
        assert_meters_in_range(env)


def test_assert_meters_in_range_above_one():
    """Test assert_meters_in_range detects meters above 1."""
    from unittest.mock import Mock

    env = Mock()
    env.meters = torch.tensor([[1.0, 0.5, 1.2, 0.8]], dtype=torch.float32)

    with pytest.raises(AssertionError, match="Meters above 1.0 detected"):
        assert_meters_in_range(env)


def test_assert_positions_in_bounds_success():
    """Test assert_positions_in_bounds passes for valid positions."""
    positions = torch.tensor([[0, 0], [7, 7], [3, 4]], dtype=torch.long)
    assert_positions_in_bounds(positions, width=8, height=8)  # Should not raise


def test_assert_positions_in_bounds_out_of_bounds_x():
    """Test assert_positions_in_bounds detects X out of bounds."""
    positions = torch.tensor([[8, 0]], dtype=torch.long)  # X = 8, but width = 8

    with pytest.raises(AssertionError, match="X positions out of bounds"):
        assert_positions_in_bounds(positions, width=8, height=8)


def test_assert_positions_in_bounds_out_of_bounds_y():
    """Test assert_positions_in_bounds detects Y out of bounds."""
    positions = torch.tensor([[0, -1]], dtype=torch.long)  # Y = -1

    with pytest.raises(AssertionError, match="Y positions out of bounds"):
        assert_positions_in_bounds(positions, width=8, height=8)


def test_assert_positions_in_bounds_3d():
    """Test assert_positions_in_bounds for 3D grids."""
    positions = torch.tensor([[2, 3, 1]], dtype=torch.long)
    assert_positions_in_bounds(positions, width=5, height=5, depth=3)  # Should not raise

    # Out of bounds Z
    positions = torch.tensor([[2, 3, 5]], dtype=torch.long)
    with pytest.raises(AssertionError, match="Z positions out of bounds"):
        assert_positions_in_bounds(positions, width=5, height=5, depth=3)


def test_assert_valid_rewards_success(basic_env):
    """Test assert_valid_rewards passes for valid rewards."""
    actions = torch.zeros(basic_env.num_agents, dtype=torch.long)
    _, rewards, _, _ = basic_env.step(actions)

    assert_valid_rewards(rewards, basic_env.num_agents)  # Should not raise


def test_assert_valid_rewards_invalid_shape():
    """Test assert_valid_rewards detects invalid shape."""
    rewards = torch.zeros(3)  # Should be (4,) for 4 agents

    with pytest.raises(AssertionError, match="Invalid reward shape"):
        assert_valid_rewards(rewards, num_agents=4)


def test_assert_valid_dones_success(basic_env):
    """Test assert_valid_dones passes for valid dones."""
    actions = torch.zeros(basic_env.num_agents, dtype=torch.long)
    _, _, dones, _ = basic_env.step(actions)

    assert_valid_dones(dones, basic_env.num_agents)  # Should not raise


def test_assert_valid_dones_invalid_dtype():
    """Test assert_valid_dones detects invalid dtype."""
    dones = torch.zeros(4, dtype=torch.int32)  # Should be bool

    with pytest.raises(AssertionError, match="Invalid done dtype"):
        assert_valid_dones(dones, num_agents=4)


# =============================================================================
# INTEGRATION TEST (COMBINING BUILDERS AND ASSERTIONS)
# =============================================================================


def test_builder_and_assertion_integration():
    """Test that builders and assertions work together."""
    # Build substrate
    substrate = make_grid2d_substrate(width=3, height=3)

    # Build positions
    positions = make_positions(num_agents=2, position_dim=2, value=0)

    # Validate positions
    assert_positions_in_bounds(positions, width=3, height=3)  # Should not raise

    # Move agents
    deltas = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    new_positions = substrate.apply_movement(positions, deltas)

    # Validate new positions
    assert_positions_in_bounds(new_positions, width=3, height=3)  # Should not raise
    assert new_positions[0, 0] == 1  # X moved right
    assert new_positions[1, 1] == 1  # Y moved down
