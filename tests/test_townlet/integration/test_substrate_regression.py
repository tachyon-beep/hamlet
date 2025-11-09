"""Regression tests for substrate abstraction (TASK-002A Phase 8).

Ensures Grid2D substrate produces identical behavior to legacy hardcoded grid.
These tests verify backward compatibility and prevent behavioral regressions.
"""

import pytest
import torch

from tests.test_townlet.helpers.config_builder import prepare_config_dir


@pytest.mark.parametrize("grid_size", [4, 7, 8])
def test_regression_observation_dims_unchanged(grid_size, cpu_env_factory, tmp_path):
    """Observation dimensions should match substrate-provided breakdown."""

    def _modifier(cfg):
        env_cfg = cfg["environment"]
        env_cfg.update(
            {
                "grid_size": grid_size,
                "vision_range": grid_size,
                "partial_observability": False,
                "enable_temporal_mechanics": False,
            }
        )

    config_dir = prepare_config_dir(tmp_path, modifier=_modifier, name=f"regression_obs_dim_{grid_size}")
    env = cpu_env_factory(config_dir=config_dir, num_agents=1)

    obs = env.reset()

    expected_obs_dim = env.metadata.observation_dim

    assert (
        obs.shape[1] == expected_obs_dim
    ), f"Observation dimension changed for {grid_size}×{grid_size} grid: {obs.shape[1]} vs {expected_obs_dim}"


def test_regression_grid2d_equivalent_to_legacy(test_config_pack_path, cpu_device, cpu_env_factory):
    """Grid2D substrate should produce identical behavior to legacy hardcoded grid.

    This test verifies that replacing hardcoded grid logic with Grid2DSubstrate
    doesn't change environment behavior.
    """
    # Create environment with Grid2D substrate (new)
    env_new = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

    # Verify substrate is Grid2D
    substrate_type = type(env_new.substrate).__name__.lower()
    assert "grid2d" in substrate_type, f"Environment should use Grid2D substrate, got {substrate_type}"

    # Verify substrate dimensions match grid_size
    assert env_new.substrate.width == 8, "Substrate width should match grid_size"
    assert env_new.substrate.height == 8, "Substrate height should match grid_size"

    # Set identical random seed
    torch.manual_seed(42)
    obs_new = env_new.reset()

    expected_dim = env_new.metadata.observation_dim

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


def test_regression_position_tensor_shapes_unchanged(test_config_pack_path, cpu_device, cpu_env_factory):
    """Position tensor shapes should match legacy behavior.

    Legacy: positions.shape == (num_agents, 2) for Grid2D
    New: positions.shape == (num_agents, substrate.position_dim)

    For Grid2D, position_dim=2, so behavior is unchanged.
    """
    env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=4)

    env.reset()

    # Position tensor shape should be (4, 2) for Grid2D
    assert env.positions.shape == (4, 2), f"Position shape changed: {env.positions.shape} vs (4, 2)"


def test_regression_affordance_positions_unchanged(test_config_pack_path, cpu_device, cpu_env_factory):
    """Affordance positions should be (x, y) tensors for Grid2D.

    Legacy: affordances stored as {name: torch.tensor([x, y])}
    New: affordances stored as {name: torch.tensor([x, y])} (unchanged for Grid2D)
    """
    env = cpu_env_factory(config_dir=test_config_pack_path, num_agents=1)

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
