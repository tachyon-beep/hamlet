"""Tests for prioritized experience replay buffer."""

import torch

from townlet.training.prioritized_replay_buffer import PrioritizedReplayBuffer


def test_prioritized_replay_buffer_push():
    """PrioritizedReplayBuffer accepts transitions."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.6,
        beta=0.4,
        device=torch.device("cpu"),
    )

    # Push batch of 1 transition
    obs = torch.randn(1, 10)  # [batch=1, obs_dim=10]
    action = torch.tensor([2])  # [batch=1]
    reward_extrinsic = torch.tensor([1.0])  # [batch=1]
    reward_intrinsic = torch.tensor([0.0])  # [batch=1]
    next_obs = torch.randn(1, 10)  # [batch=1, obs_dim=10]
    done = torch.tensor([False])  # [batch=1]

    buffer.push(obs, action, reward_extrinsic, reward_intrinsic, next_obs, done)

    assert buffer.size() == 1


def test_prioritized_replay_buffer_sample():
    """PrioritizedReplayBuffer samples with priorities."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.6,
        beta=0.4,
        device=torch.device("cpu"),
    )

    # Add 50 transitions (batch of 50)
    obs = torch.randn(50, 10)  # [batch=50, obs_dim=10]
    actions = torch.tensor([i % 5 for i in range(50)])  # [batch=50]
    rewards_extrinsic = torch.tensor([float(i) for i in range(50)])  # [batch=50]
    rewards_intrinsic = torch.zeros(50)  # [batch=50]
    next_obs = torch.randn(50, 10)  # [batch=50, obs_dim=10]
    dones = torch.tensor([i == 49 for i in range(50)])  # [batch=50]
    buffer.push(obs, actions, rewards_extrinsic, rewards_intrinsic, next_obs, dones)

    # Sample batch
    batch = buffer.sample(batch_size=16)

    assert batch["observations"].shape == (16, 10)
    assert batch["actions"].shape == (16,)
    assert batch["rewards"].shape == (16,)
    assert batch["next_observations"].shape == (16, 10)
    assert batch["dones"].shape == (16,)
    assert "weights" in batch  # Importance sampling weights
    assert "indices" in batch  # For priority updates


def test_prioritized_replay_buffer_update_priorities():
    """PrioritizedReplayBuffer updates priorities from TD errors."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.6,
        beta=0.4,
        device=torch.device("cpu"),
    )

    # Add transitions (batch of 20)
    obs = torch.randn(20, 10)  # [batch=20, obs_dim=10]
    actions = torch.zeros(20, dtype=torch.long)  # [batch=20]
    rewards_extrinsic = torch.zeros(20)  # [batch=20]
    rewards_intrinsic = torch.zeros(20)  # [batch=20]
    next_obs = torch.randn(20, 10)  # [batch=20, obs_dim=10]
    dones = torch.zeros(20, dtype=torch.bool)  # [batch=20]
    buffer.push(obs, actions, rewards_extrinsic, rewards_intrinsic, next_obs, dones)

    # Sample batch
    batch = buffer.sample(batch_size=10)

    # Update priorities with TD errors
    td_errors = torch.randn(10).abs()  # Absolute TD errors
    buffer.update_priorities(batch["indices"], td_errors)

    # Priorities should be updated (no exception raised)
    assert buffer.size() == 20


def test_prioritized_replay_buffer_beta_annealing():
    """PrioritizedReplayBuffer anneals beta toward 1.0."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.6,
        beta=0.4,
        beta_annealing=True,
        device=torch.device("cpu"),
    )

    initial_beta = buffer.beta
    assert initial_beta == 0.4

    # Anneal beta (would be called during training)
    buffer.anneal_beta(total_steps=10000, current_step=5000)

    # Beta should increase toward 1.0
    assert buffer.beta > initial_beta
    assert buffer.beta <= 1.0


def test_prioritized_replay_buffer_len():
    """PrioritizedReplayBuffer implements __len__."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.6,
        beta=0.4,
        device=torch.device("cpu"),
    )

    assert len(buffer) == 0

    # Add transitions
    obs = torch.randn(10, 5)
    actions = torch.zeros(10, dtype=torch.long)
    rewards_extrinsic = torch.zeros(10)
    rewards_intrinsic = torch.zeros(10)
    next_obs = torch.randn(10, 5)
    dones = torch.zeros(10, dtype=torch.bool)
    buffer.push(obs, actions, rewards_extrinsic, rewards_intrinsic, next_obs, dones)

    assert len(buffer) == 10


def test_prioritized_replay_buffer_serialize():
    """PrioritizedReplayBuffer can be serialized and restored."""
    buffer = PrioritizedReplayBuffer(
        capacity=50,
        alpha=0.7,
        beta=0.5,
        beta_annealing=False,
        device=torch.device("cpu"),
    )

    # Add transitions
    obs = torch.randn(10, 5)
    actions = torch.tensor([i % 3 for i in range(10)])
    rewards_extrinsic = torch.tensor([float(i) for i in range(10)])
    rewards_intrinsic = torch.ones(10) * 0.1
    next_obs = torch.randn(10, 5)
    dones = torch.zeros(10, dtype=torch.bool)
    buffer.push(obs, actions, rewards_extrinsic, rewards_intrinsic, next_obs, dones)

    # Serialize
    state = buffer.serialize()

    # Create new buffer and restore
    new_buffer = PrioritizedReplayBuffer(
        capacity=50,
        alpha=0.7,
        beta=0.5,
        device=torch.device("cpu"),
    )
    new_buffer.load_from_serialized(state)

    # Verify state restored
    assert len(new_buffer) == 10
    assert new_buffer.alpha == 0.7
    assert new_buffer.beta == 0.5
    assert new_buffer.capacity == 50
    assert new_buffer.position == buffer.position
    assert new_buffer.max_priority == buffer.max_priority
