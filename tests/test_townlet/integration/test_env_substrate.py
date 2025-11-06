"""Test environment substrate integration (core functionality)."""

from pathlib import Path

import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv
from townlet.substrate.grid2d import Grid2DSubstrate


def test_env_loads_substrate_config():
    """Environment should load substrate.yaml and create substrate instance."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Verify substrate loaded
    assert hasattr(env, "grid_size")
    assert env.grid_size == 8


def test_env_substrate_accessible():
    """Environment should expose substrate for inspection."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Verify substrate is accessible and correct type
    assert hasattr(env, "substrate")
    assert isinstance(env.substrate, Grid2DSubstrate)
    assert env.substrate.width == 8


def test_env_initializes_positions_via_substrate():
    """Environment should use substrate.initialize_positions() in reset()."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=5,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Reset environment
    env.reset()

    # Positions should be initialized via substrate
    assert env.positions.shape == (5, 2)  # [num_agents, position_dim]
    assert env.positions.dtype == torch.long
    assert env.positions.device == torch.device("cpu")

    # Positions should be within grid bounds
    assert (env.positions >= 0).all()
    assert (env.positions < 8).all()


def test_env_applies_movement_via_substrate():
    """Environment should use substrate.apply_movement() for boundary handling."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    env.reset()

    # Place agent at top-left corner
    env.positions = torch.tensor([[0, 0]], dtype=torch.long, device=torch.device("cpu"))

    # Try to move up (action 0 = UP)
    action = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))

    env.step(action)

    # Position should be clamped by substrate (boundary="clamp")
    assert (env.positions[:, 0] >= 0).all()  # X within bounds
    assert (env.positions[:, 1] >= 0).all()  # Y within bounds


def test_env_randomizes_affordances_via_substrate():
    """Environment should use substrate.get_all_positions() for affordance placement."""
    env = VectorizedHamletEnv(
        config_pack_path=Path("configs/L1_full_observability"),
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.5,
        wait_energy_cost=0.1,
        interact_energy_cost=0.3,
        agent_lifespan=1000,
        device=torch.device("cpu"),
    )

    # Randomize affordances
    env.randomize_affordance_positions()

    # All affordances should have valid positions within grid
    for affordance_name, position in env.affordances.items():
        assert position.shape == (2,)  # [x, y]
        assert (position >= 0).all()
        assert (position < 8).all()

    # Affordances should not overlap (each at unique position)
    positions_list = [tuple(pos.tolist()) for pos in env.affordances.values()]
    assert len(positions_list) == len(set(positions_list))  # All unique
