"""Tests for VectorizedHamletEnv (GPU-native)."""

import pytest
import torch


def test_vectorized_env_construction():
    """VectorizedHamletEnv should construct with correct batch size."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    num_agents = 3
    env = VectorizedHamletEnv(
        num_agents=num_agents,
        grid_size=8,
        device=torch.device('cpu'),
    )

    assert env.num_agents == num_agents
    assert env.grid_size == 8
    assert env.device.type == 'cpu'
    assert env.observation_dim == 70  # 8×8 grid + 6 meters


def test_vectorized_env_reset():
    """Reset should return batched observations."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(
        num_agents=5,
        grid_size=8,
        device=torch.device('cpu'),
    )

    observations = env.reset()

    assert isinstance(observations, torch.Tensor)
    assert observations.shape == (5, 70)  # [num_agents, obs_dim]
    assert observations.device.type == 'cpu'
