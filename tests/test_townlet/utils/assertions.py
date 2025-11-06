"""Assertion helpers for validating test outputs.

This module provides reusable assertion functions that validate common invariants
in Townlet tests (observation shapes, meter ranges, action masks, etc.).

Design principles:
- Clear error messages with context
- Type hints for IDE autocomplete
- Comprehensive docstrings with examples
- Fail-fast with informative error messages

Example:
    from tests.test_townlet.utils.assertions import (
        assert_valid_observation,
        assert_meters_in_range,
    )

    def test_environment():
        obs = env.reset()
        assert_valid_observation(env, obs)
        assert_meters_in_range(env)
"""

import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv

# =============================================================================
# OBSERVATION ASSERTIONS
# =============================================================================


def assert_valid_observation(env: VectorizedHamletEnv, obs: torch.Tensor) -> None:
    """Validate observation shape and dtype.

    Args:
        env: Environment that produced the observation
        obs: Observation tensor to validate

    Raises:
        AssertionError: If observation is invalid

    Example:
        >>> obs = env.reset()
        >>> assert_valid_observation(env, obs)  # Passes if valid
    """
    expected_shape = (env.num_agents, env.observation_dim)

    assert obs.shape == expected_shape, (
        f"Invalid observation shape: expected {expected_shape}, got {obs.shape}\n"
        f"  num_agents: {env.num_agents}\n"
        f"  observation_dim: {env.observation_dim}"
    )

    assert obs.dtype == torch.float32, (
        f"Invalid observation dtype: expected torch.float32, got {obs.dtype}\n"
        f"Observations must be float32 for neural network compatibility"
    )

    # Check for NaN/Inf
    if torch.isnan(obs).any():
        nan_count = torch.isnan(obs).sum().item()
        raise AssertionError(f"Observation contains {nan_count} NaN values")

    if torch.isinf(obs).any():
        inf_count = torch.isinf(obs).sum().item()
        raise AssertionError(f"Observation contains {inf_count} Inf values")


# =============================================================================
# ACTION MASK ASSERTIONS
# =============================================================================


def assert_valid_action_mask(env: VectorizedHamletEnv, mask: torch.Tensor) -> None:
    """Validate action mask shape and values.

    Action masks should be boolean tensors indicating which actions are valid.

    Args:
        env: Environment that produced the mask
        mask: Action mask tensor to validate

    Raises:
        AssertionError: If action mask is invalid

    Example:
        >>> mask = env.get_action_masks()
        >>> assert_valid_action_mask(env, mask)
    """
    expected_shape = (env.num_agents, env.action_dim)

    assert mask.shape == expected_shape, (
        f"Invalid action mask shape: expected {expected_shape}, got {mask.shape}\n"
        f"  num_agents: {env.num_agents}\n"
        f"  action_dim: {env.action_dim}"
    )

    assert mask.dtype == torch.bool, (
        f"Invalid action mask dtype: expected torch.bool, got {mask.dtype}\n" f"Action masks must be boolean tensors"
    )

    # Check that at least one action is valid per agent
    valid_actions_per_agent = mask.sum(dim=1)
    agents_with_no_actions = (valid_actions_per_agent == 0).nonzero(as_tuple=True)[0]

    if len(agents_with_no_actions) > 0:
        raise AssertionError(
            f"{len(agents_with_no_actions)} agents have no valid actions:\n"
            f"  Agent IDs: {agents_with_no_actions.tolist()}\n"
            f"  Mask: {mask[agents_with_no_actions]}"
        )


# =============================================================================
# METER ASSERTIONS
# =============================================================================


def assert_meters_in_range(env: VectorizedHamletEnv) -> None:
    """Validate that all meters are in valid range [0, 1].

    Args:
        env: Environment to check

    Raises:
        AssertionError: If any meter is out of range

    Example:
        >>> env.step(actions)
        >>> assert_meters_in_range(env)
    """
    meters = env.meters  # Shape: (num_agents, meter_count)

    # Check for NaN/Inf first
    if torch.isnan(meters).any():
        nan_agents = torch.isnan(meters).any(dim=1).nonzero(as_tuple=True)[0]
        raise AssertionError(f"Meters contain NaN values for {len(nan_agents)} agents:\n" f"  Agent IDs: {nan_agents.tolist()}")

    if torch.isinf(meters).any():
        inf_agents = torch.isinf(meters).any(dim=1).nonzero(as_tuple=True)[0]
        raise AssertionError(f"Meters contain Inf values for {len(inf_agents)} agents:\n" f"  Agent IDs: {inf_agents.tolist()}")

    # Check range [0, 1]
    below_zero = meters < 0.0
    above_one = meters > 1.0

    if below_zero.any():
        violations = below_zero.nonzero(as_tuple=True)
        agent_ids = violations[0][:5].tolist()  # Show first 5
        meter_ids = violations[1][:5].tolist()
        values = meters[violations][:5].tolist()

        raise AssertionError(
            f"Meters below 0.0 detected:\n"
            f"  First 5 violations: agent_ids={agent_ids}, meter_ids={meter_ids}, values={values}\n"
            f"  Total violations: {below_zero.sum().item()}"
        )

    if above_one.any():
        violations = above_one.nonzero(as_tuple=True)
        agent_ids = violations[0][:5].tolist()  # Show first 5
        meter_ids = violations[1][:5].tolist()
        values = meters[violations][:5].tolist()

        raise AssertionError(
            f"Meters above 1.0 detected:\n"
            f"  First 5 violations: agent_ids={agent_ids}, meter_ids={meter_ids}, values={values}\n"
            f"  Total violations: {above_one.sum().item()}"
        )


# =============================================================================
# POSITION ASSERTIONS
# =============================================================================


def assert_positions_in_bounds(
    positions: torch.Tensor,
    width: int,
    height: int,
    depth: int | None = None,
) -> None:
    """Validate that positions are within grid bounds.

    Args:
        positions: Position tensor (num_agents, position_dim)
        width: Grid width
        height: Grid height
        depth: Grid depth (optional, for 3D grids)

    Raises:
        AssertionError: If any position is out of bounds

    Example:
        >>> positions = env.substrate.positions
        >>> assert_positions_in_bounds(positions, width=8, height=8)
    """
    num_agents, position_dim = positions.shape

    # Validate X (column)
    x = positions[:, 0]
    if (x < 0).any() or (x >= width).any():
        out_of_bounds = ((x < 0) | (x >= width)).nonzero(as_tuple=True)[0]
        raise AssertionError(
            f"X positions out of bounds [0, {width}):\n"
            f"  Agent IDs: {out_of_bounds.tolist()}\n"
            f"  X values: {x[out_of_bounds].tolist()}"
        )

    # Validate Y (row)
    y = positions[:, 1]
    if (y < 0).any() or (y >= height).any():
        out_of_bounds = ((y < 0) | (y >= height)).nonzero(as_tuple=True)[0]
        raise AssertionError(
            f"Y positions out of bounds [0, {height}):\n"
            f"  Agent IDs: {out_of_bounds.tolist()}\n"
            f"  Y values: {y[out_of_bounds].tolist()}"
        )

    # Validate Z (depth) if 3D
    if position_dim >= 3 and depth is not None:
        z = positions[:, 2]
        if (z < 0).any() or (z >= depth).any():
            out_of_bounds = ((z < 0) | (z >= depth)).nonzero(as_tuple=True)[0]
            raise AssertionError(
                f"Z positions out of bounds [0, {depth}):\n"
                f"  Agent IDs: {out_of_bounds.tolist()}\n"
                f"  Z values: {z[out_of_bounds].tolist()}"
            )


# =============================================================================
# REWARD ASSERTIONS
# =============================================================================


def assert_valid_rewards(rewards: torch.Tensor, num_agents: int) -> None:
    """Validate reward tensor shape and values.

    Args:
        rewards: Reward tensor to validate
        num_agents: Expected number of agents

    Raises:
        AssertionError: If rewards are invalid

    Example:
        >>> _, rewards, _, _, _ = env.step(actions)
        >>> assert_valid_rewards(rewards, env.num_agents)
    """
    expected_shape = (num_agents,)

    assert rewards.shape == expected_shape, (
        f"Invalid reward shape: expected {expected_shape}, got {rewards.shape}\n"
        f"  num_agents: {num_agents}"
    )

    assert rewards.dtype == torch.float32, (
        f"Invalid reward dtype: expected torch.float32, got {rewards.dtype}\n"
        f"Rewards must be float32 for training"
    )

    # Check for NaN/Inf
    if torch.isnan(rewards).any():
        nan_count = torch.isnan(rewards).sum().item()
        raise AssertionError(f"Rewards contain {nan_count} NaN values")

    if torch.isinf(rewards).any():
        inf_count = torch.isinf(rewards).sum().item()
        raise AssertionError(f"Rewards contain {inf_count} Inf values")


# =============================================================================
# DONE ASSERTIONS
# =============================================================================


def assert_valid_dones(dones: torch.Tensor, num_agents: int) -> None:
    """Validate done tensor shape and dtype.

    Args:
        dones: Done tensor to validate
        num_agents: Expected number of agents

    Raises:
        AssertionError: If dones are invalid

    Example:
        >>> _, _, dones, _, _ = env.step(actions)
        >>> assert_valid_dones(dones, env.num_agents)
    """
    expected_shape = (num_agents,)

    assert dones.shape == expected_shape, (
        f"Invalid done shape: expected {expected_shape}, got {dones.shape}\n" f"  num_agents: {num_agents}"
    )

    assert dones.dtype == torch.bool, f"Invalid done dtype: expected torch.bool, got {dones.dtype}\n" f" Done flags must be boolean tensors"
