"""Test observation encoding uses substrate methods."""

from pathlib import Path

import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_full_observation_uses_substrate(test_config_pack_path):
    """Full observability should use substrate.encode_observation()."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
        config_pack_path=Path("configs/L1_full_observability"),
        device=torch.device("cpu"),
    )

    # Get observation
    obs = env.reset()

    # Observation should include substrate position encoding
    # For 8×8 Grid2D with "relative" encoding: 2 (coords) + 8 (meters) + 15 (affordance) + 4 (temporal) = 29
    expected_dim = (
        env.substrate.get_observation_dim()  # 2 for Grid2D with "relative" encoding
        + 8  # meters
        + 15  # affordance at position (14 + "none")
        + 4  # temporal extras (time_sin, time_cos, interaction_progress, lifetime)
    )
    assert obs.shape[1] == expected_dim, f"Expected {expected_dim}, got {obs.shape[1]}"


def test_partial_observation_uses_substrate(test_config_pack_path):
    """Partial observability should use substrate.encode_partial_observation()."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=True,
        vision_range=2,  # 5×5 window
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
        config_pack_path=Path("configs/L2_partial_observability"),
        device=torch.device("cpu"),
    )

    # Get observation
    obs = env.reset()

    # Partial obs should use local window encoding
    # For 5×5 window: 25 (local grid) + 2 (normalized position) + 8 (meters) + 15 (affordance) + 4 (temporal) = 54
    window_size = 2 * env.vision_range + 1  # 5×5 for vision_range=2
    expected_dim = (
        window_size * window_size  # 25 for 5×5 window
        + env.substrate.position_dim  # 2 for Grid2D
        + 8  # meters
        + 15  # affordance at position
        + 4  # temporal extras
    )
    assert obs.shape[1] == expected_dim, f"Expected {expected_dim}, got {obs.shape[1]}"
    assert obs.shape[1] == 54, f"L2 should have 54 dims, got {obs.shape[1]}"


def test_observation_dim_matches_actual_observation(test_config_pack_path):
    """Environment's observation_dim should match actual observation shape."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=False,
        vision_range=8,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
        config_pack_path=test_config_pack_path,
        device=torch.device("cpu"),
    )

    obs = env.reset()

    # Observation dimension should match actual observation
    assert obs.shape[1] == env.observation_dim, f"observation_dim={env.observation_dim} doesn't match actual obs.shape[1]={obs.shape[1]}"


def test_partial_observation_dim_matches_actual(test_config_pack_path):
    """POMDP observation_dim should match actual observation shape."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=True,
        vision_range=2,
        enable_temporal_mechanics=False,
        move_energy_cost=0.005,
        wait_energy_cost=0.001,
        interact_energy_cost=0.0,
        agent_lifespan=1000,
        config_pack_path=test_config_pack_path,
        device=torch.device("cpu"),
    )

    obs = env.reset()

    # Observation dimension should match actual observation
    assert obs.shape[1] == env.observation_dim, f"observation_dim={env.observation_dim} doesn't match actual obs.shape[1]={obs.shape[1]}"
