"""Tests for BatchedAgentState (hot path tensor container)."""

import pytest
import torch
import numpy as np


def test_batched_agent_state_construction():
    """BatchedAgentState should construct with correct tensor shapes."""
    from townlet.training.state import BatchedAgentState

    batch_size = 10
    obs_dim = 70

    state = BatchedAgentState(
        observations=torch.randn(batch_size, obs_dim),
        actions=torch.randint(0, 5, (batch_size,)),
        rewards=torch.randn(batch_size),
        dones=torch.zeros(batch_size, dtype=torch.bool),
        epsilons=torch.full((batch_size,), 0.5),
        intrinsic_rewards=torch.zeros(batch_size),
        survival_times=torch.randint(0, 1000, (batch_size,)),
        curriculum_difficulties=torch.full((batch_size,), 0.5),
        device=torch.device('cpu'),
    )

    assert state.batch_size == batch_size
    assert state.observations.shape == (batch_size, obs_dim)
    assert state.actions.shape == (batch_size,)
    assert state.device.type == 'cpu'


def test_batched_agent_state_device_transfer():
    """BatchedAgentState should support device transfer."""
    from townlet.training.state import BatchedAgentState

    state_cpu = BatchedAgentState(
        observations=torch.randn(5, 70),
        actions=torch.zeros(5, dtype=torch.long),
        rewards=torch.zeros(5),
        dones=torch.zeros(5, dtype=torch.bool),
        epsilons=torch.ones(5),
        intrinsic_rewards=torch.zeros(5),
        survival_times=torch.zeros(5, dtype=torch.long),
        curriculum_difficulties=torch.zeros(5),
        device=torch.device('cpu'),
    )

    # Transfer to same device (should work)
    state_cpu2 = state_cpu.to(torch.device('cpu'))
    assert state_cpu2.device.type == 'cpu'
    assert state_cpu2.observations.shape == state_cpu.observations.shape


def test_batched_agent_state_cpu_summary():
    """BatchedAgentState should extract CPU summary for telemetry."""
    from townlet.training.state import BatchedAgentState

    state = BatchedAgentState(
        observations=torch.randn(3, 70),
        actions=torch.tensor([0, 1, 2]),
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        dones=torch.tensor([False, False, True]),
        epsilons=torch.tensor([0.9, 0.8, 0.7]),
        intrinsic_rewards=torch.tensor([0.1, 0.2, 0.3]),
        survival_times=torch.tensor([100, 200, 300]),
        curriculum_difficulties=torch.tensor([0.5, 0.6, 0.7]),
        device=torch.device('cpu'),
    )

    summary = state.detach_cpu_summary()

    # Should return dict of numpy arrays
    assert isinstance(summary, dict)
    assert isinstance(summary['rewards'], np.ndarray)
    assert summary['rewards'].shape == (3,)
    assert np.allclose(summary['rewards'], [1.0, 2.0, 3.0])

    assert 'survival_times' in summary
    assert 'epsilons' in summary
    assert 'curriculum_difficulties' in summary


def test_batched_agent_state_batch_size_property():
    """BatchedAgentState batch_size should match observations."""
    from townlet.training.state import BatchedAgentState

    for batch_size in [1, 5, 10, 100]:
        state = BatchedAgentState(
            observations=torch.randn(batch_size, 70),
            actions=torch.zeros(batch_size, dtype=torch.long),
            rewards=torch.zeros(batch_size),
            dones=torch.zeros(batch_size, dtype=torch.bool),
            epsilons=torch.ones(batch_size),
            intrinsic_rewards=torch.zeros(batch_size),
            survival_times=torch.zeros(batch_size, dtype=torch.long),
            curriculum_difficulties=torch.zeros(batch_size),
            device=torch.device('cpu'),
        )

        assert state.batch_size == batch_size
