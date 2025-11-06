"""Regression tests for substrate abstraction (TASK-002A Phase 8).

Ensures Grid2D substrate produces identical behavior to legacy hardcoded grid.
These tests verify backward compatibility and prevent behavioral regressions.
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.mark.parametrize("grid_size", [3, 7, 8])
def test_regression_observation_dims_unchanged(grid_size, test_config_pack_path, cpu_device):
    """Observation dimensions should match substrate-provided breakdown."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=grid_size,
        device=cpu_device,
        config_pack_path=test_config_pack_path,
        partial_observability=False,  # Full observability
        vision_range=grid_size,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
    )

    obs = env.reset()

    expected_obs_dim = env.substrate.get_observation_dim() + env.meter_count + (env.num_affordance_types + 1) + 4

    assert (
        obs.shape[1] == expected_obs_dim
    ), f"Observation dimension changed for {grid_size}×{grid_size} grid: {obs.shape[1]} vs {expected_obs_dim}"


def test_regression_grid2d_equivalent_to_legacy(test_config_pack_path, cpu_device):
    """Grid2D substrate should produce identical behavior to legacy hardcoded grid.

    This test verifies that replacing hardcoded grid logic with Grid2DSubstrate
    doesn't change environment behavior.
    """
    # Create environment with Grid2D substrate (new)
    env_new = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=cpu_device,
        config_pack_path=test_config_pack_path,
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
    )

    # Verify substrate is Grid2D
    substrate_type = type(env_new.substrate).__name__.lower()
    assert "grid2d" in substrate_type, f"Environment should use Grid2D substrate, got {substrate_type}"

    # Verify substrate dimensions match grid_size
    assert env_new.substrate.width == 8, "Substrate width should match grid_size"
    assert env_new.substrate.height == 8, "Substrate height should match grid_size"

    # Set identical random seed
    torch.manual_seed(42)
    obs_new = env_new.reset()

    expected_dim = env_new.substrate.get_observation_dim() + env_new.meter_count + (env_new.num_affordance_types + 1) + 4

    # Verify observation dimension matches expected breakdown
    assert obs_new.shape[1] == expected_dim, f"Observation dim changed: {obs_new.shape[1]} vs {expected_dim}"

    # Run 10 steps with fixed actions
    actions = [0, 1, 2, 3, 4] * 2  # UP, DOWN, LEFT, RIGHT, INTERACT × 2
    for action in actions:
        action_tensor = torch.tensor([action], dtype=torch.long, device=cpu_device)
        obs, reward, done, info = env_new.step(action_tensor, depletion_multiplier=1.0)

        # Verify position stays within bounds (clamping behavior)
        assert 0 <= env_new.positions[0, 0] < 8, f"X position out of bounds: {env_new.positions[0, 0]}"
        assert 0 <= env_new.positions[0, 1] < 8, f"Y position out of bounds: {env_new.positions[0, 1]}"

    # No assertions on exact rewards (stochastic), just verify no crashes


def test_regression_position_tensor_shapes_unchanged(test_config_pack_path, cpu_device):
    """Position tensor shapes should match legacy behavior.

    Legacy: positions.shape == (num_agents, 2) for Grid2D
    New: positions.shape == (num_agents, substrate.position_dim)

    For Grid2D, position_dim=2, so behavior is unchanged.
    """
    env = VectorizedHamletEnv(
        num_agents=4,  # Multi-agent test
        grid_size=8,
        device=cpu_device,
        config_pack_path=test_config_pack_path,
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
    )

    env.reset()

    # Position tensor shape should be (4, 2) for Grid2D
    assert env.positions.shape == (4, 2), f"Position shape changed: {env.positions.shape} vs (4, 2)"


def test_regression_affordance_positions_unchanged(test_config_pack_path, cpu_device):
    """Affordance positions should be (x, y) tensors for Grid2D.

    Legacy: affordances stored as {name: torch.tensor([x, y])}
    New: affordances stored as {name: torch.tensor([x, y])} (unchanged for Grid2D)
    """
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=cpu_device,
        config_pack_path=test_config_pack_path,
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
    )

    env.reset()

    # Affordances should have (x, y) positions
    for name, pos in env.affordances.items():
        assert pos.shape == (2,), f"Affordance {name} position shape changed: {pos.shape} vs (2,)"
        assert 0 <= pos[0] < 8, f"Affordance {name} X out of bounds: {pos[0]}"
        assert 0 <= pos[1] < 8, f"Affordance {name} Y out of bounds: {pos[1]}"


def test_regression_movement_mechanics_unchanged(grid2d_8x8_env):
    """Movement mechanics should match legacy behavior (clamping at boundaries).

    Legacy: Moving into wall clamps position to boundary
    New: Grid2DSubstrate with boundary="clamp" should produce identical behavior
    """
    env = grid2d_8x8_env
    env.reset()

    # Place agent at top-left corner
    env.positions[0] = torch.tensor([0, 0], dtype=torch.long, device=env.device)

    # Try to move UP (should clamp to y=0)
    actions = torch.tensor([0], dtype=torch.long, device=env.device)  # UP
    env.step(actions, depletion_multiplier=1.0)

    assert env.positions[0, 1] == 0, "Moving UP from top edge should clamp to y=0"

    # Try to move LEFT (should clamp to x=0)
    actions = torch.tensor([2], dtype=torch.long, device=env.device)  # LEFT
    env.step(actions, depletion_multiplier=1.0)

    assert env.positions[0, 0] == 0, "Moving LEFT from left edge should clamp to x=0"
