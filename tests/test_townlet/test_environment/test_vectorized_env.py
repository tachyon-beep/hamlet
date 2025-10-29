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


def test_vectorized_env_step():
    """Step should return batched (obs, rewards, dones, info)."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=3, grid_size=8, device=torch.device('cpu'))
    env.reset()

    # All agents move UP
    actions = torch.zeros(3, dtype=torch.long)  # 0 = UP

    obs, rewards, dones, info = env.step(actions)

    assert obs.shape == (3, 70)
    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert isinstance(info, dict)


def test_vectorized_env_movement():
    """Agents should move correctly."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    env.reset()

    # Set known position
    env.positions[0] = torch.tensor([4, 4], device=env.device)

    # Move UP (action 0)
    obs, _, _, _ = env.step(torch.tensor([0]))
    assert env.positions[0, 0] == 3  # Row decreased
    assert env.positions[0, 1] == 4  # Column unchanged

    # Move RIGHT (action 3)
    obs, _, _, _ = env.step(torch.tensor([3]))
    assert env.positions[0, 0] == 3  # Row unchanged
    assert env.positions[0, 1] == 5  # Column increased


def test_vectorized_env_meter_depletion():
    """Meters should deplete each step."""
    from townlet.environment.vectorized_env import VectorizedHamletEnv

    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))
    env.reset()

    initial_energy = env.meters[0, 0].item()

    # Take 10 steps
    for _ in range(10):
        env.step(torch.tensor([0]))  # Move UP

    final_energy = env.meters[0, 0].item()

    assert final_energy < initial_energy  # Energy depleted
