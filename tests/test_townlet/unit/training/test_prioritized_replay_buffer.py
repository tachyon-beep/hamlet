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

    obs = torch.randn(10)
    action = torch.tensor(2)
    reward = torch.tensor(1.0)
    next_obs = torch.randn(10)
    done = torch.tensor(False)

    buffer.push(obs, action, reward, next_obs, done)

    assert buffer.size() == 1


def test_prioritized_replay_buffer_sample():
    """PrioritizedReplayBuffer samples with priorities."""
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.6,
        beta=0.4,
        device=torch.device("cpu"),
    )

    # Add 50 transitions
    for i in range(50):
        obs = torch.randn(10)
        action = torch.tensor(i % 5)
        reward = torch.tensor(float(i))
        next_obs = torch.randn(10)
        done = torch.tensor(i == 49)
        buffer.push(obs, action, reward, next_obs, done)

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

    # Add transitions
    for i in range(20):
        obs = torch.randn(10)
        buffer.push(obs, torch.tensor(0), torch.tensor(0.0), obs, torch.tensor(False))

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
