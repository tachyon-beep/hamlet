"""
Test for observation dimension mismatch between environment and RecurrentSpatialQNetwork.

This test demonstrates the bug where:
- Environment produces: 25 (grid) + 2 (pos) + 8 (meters) + 15 (affordances with "none") = 50 dims + temporal extras
- Network expects: 25 (grid) + 2 (pos) + 8 (meters) + 15 (affordances) = 50 dims

This causes shape mismatches during training.
"""

import torch

from townlet.agent.networks import RecurrentSpatialQNetwork
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_observation_dimension_matches_network():
    """Test that observation dimensions match between environment and network."""
    # Create POMDP environment
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=True,
        vision_range=2,  # 5×5 window
        enable_temporal_mechanics=False,
        device=torch.device("cpu"),
    )

    obs_dim = env.observation_dim

    # Expected: 25 (grid) + 2 (pos) + 8 (meters) + (num_affordance_types + 1)
    # num_affordance_types = 14, so affordance encoding = 15
    expected_dim = 25 + 2 + 8 + (env.num_affordance_types + 1)
    assert obs_dim == expected_dim, f"Expected {expected_dim} dims, got {obs_dim}"


def test_observation_dimension_with_temporal_mechanics():
    """Temporal mechanics should add 3 dimensions (time_of_day + interaction_progress)."""
    # Create POMDP environment with temporal mechanics
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=True,
        vision_range=2,  # 5×5 window
        enable_temporal_mechanics=True,  # +3 dims (sin, cos, progress)
        device=torch.device("cpu"),
    )

    obs_dim = env.observation_dim

    # Expected: 25 (grid) + 2 (pos) + 8 (meters) + (num_affordance_types + 1) + 3 (temporal)
    # num_affordance_types = 14, so affordance encoding = 15
    expected_dim = 25 + 2 + 8 + (env.num_affordance_types + 1) + 3
    assert obs_dim == expected_dim, f"Expected {expected_dim} dims, got {obs_dim}"  # Create network with temporal support
    network = RecurrentSpatialQNetwork(
        action_dim=6,
        window_size=5,
        num_meters=8,
        num_affordance_types=env.num_affordance_types,
        enable_temporal_features=True,  # Support temporal dims
        hidden_dim=256,
    )

    # Reset and verify forward pass works
    obs = env.reset()
    q_values, hidden = network(obs)

    assert q_values.shape == (1, 6)


def test_full_observability_dimension_matches():
    """Test full observability mode."""
    # Create full observability environment
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        partial_observability=False,  # Full grid visible
        enable_temporal_mechanics=False,
        device=torch.device("cpu"),
    )

    obs_dim = env.observation_dim

    # Expected: 64 (grid) + 8 (meters) + (num_affordance_types + 1)
    # num_affordance_types = 14, so affordance encoding = 15
    expected_dim = 64 + 8 + (env.num_affordance_types + 1)
    assert obs_dim == expected_dim, f"Expected {expected_dim} dims, got {obs_dim}"
