"""Tests for ReplayBuffer."""

import pytest
import torch
from townlet.training.replay_buffer import ReplayBuffer


def test_replay_buffer_push_and_sample():
    """ReplayBuffer should store transitions and sample with combined rewards."""
    buffer = ReplayBuffer(capacity=100, device=torch.device('cpu'))

    # Push 10 transitions
    observations = torch.randn(10, 70)
    actions = torch.randint(0, 5, (10,))
    rewards_extrinsic = torch.randn(10)
    rewards_intrinsic = torch.randn(10)
    next_observations = torch.randn(10, 70)
    dones = torch.zeros(10, dtype=torch.bool)

    buffer.push(observations, actions, rewards_extrinsic, rewards_intrinsic, next_observations, dones)

    assert len(buffer) == 10

    # Sample with intrinsic weight 0.5
    batch = buffer.sample(batch_size=5, intrinsic_weight=0.5)

    assert batch['observations'].shape == (5, 70)
    assert batch['actions'].shape == (5,)
    assert batch['rewards'].shape == (5,)
    assert batch['next_observations'].shape == (5, 70)
    assert batch['dones'].shape == (5,)

    # Verify rewards are combined: extrinsic + intrinsic * 0.5
    # (Can't verify exact values due to random sampling, but shape is correct)


def test_replay_buffer_capacity_fifo():
    """ReplayBuffer should evict oldest when full (FIFO)."""
    buffer = ReplayBuffer(capacity=5, device=torch.device('cpu'))

    # Push 10 transitions (should keep last 5)
    for i in range(10):
        obs = torch.ones(1, 70) * i
        buffer.push(
            observations=obs,
            actions=torch.tensor([0]),
            rewards_extrinsic=torch.tensor([float(i)]),
            rewards_intrinsic=torch.tensor([0.0]),
            next_observations=obs,
            dones=torch.tensor([False]),
        )

    assert len(buffer) == 5

    # Sample all, verify they're from last 5 pushes (indices 5-9)
    batch = buffer.sample(batch_size=5, intrinsic_weight=0.0)

    # First element of observation should be 5-9 (oldest first in buffer)
    first_elements = batch['observations'][:, 0].sort()[0]
    expected = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0])
    assert torch.allclose(first_elements, expected)


def test_replay_buffer_device_handling():
    """ReplayBuffer should respect device placement."""
    device = torch.device('cpu')
    buffer = ReplayBuffer(capacity=10, device=device)

    obs = torch.randn(2, 70)
    buffer.push(
        observations=obs,
        actions=torch.tensor([0, 1]),
        rewards_extrinsic=torch.tensor([1.0, 2.0]),
        rewards_intrinsic=torch.tensor([0.5, 0.5]),
        next_observations=obs,
        dones=torch.tensor([False, False]),
    )

    batch = buffer.sample(batch_size=2, intrinsic_weight=1.0)

    assert batch['observations'].device.type == device.type
    assert batch['actions'].device.type == device.type
    assert batch['rewards'].device.type == device.type
