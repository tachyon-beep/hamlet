"""
Tests for ReplayBuffer.
"""

import pytest
from hamlet.agent.replay_buffer import ReplayBuffer


def test_replay_buffer_initialization():
    """Test that replay buffer initializes correctly."""
    buffer = ReplayBuffer(capacity=100)
    assert buffer.capacity == 100
    assert len(buffer) == 0


def test_replay_buffer_push():
    """Test adding experiences to buffer."""
    buffer = ReplayBuffer(capacity=10)
    buffer.push(
        state=[0, 1, 2],
        action=1,
        reward=1.0,
        next_state=[1, 2, 3],
        done=False
    )
    assert len(buffer) == 1


def test_replay_buffer_capacity():
    """Test that buffer respects capacity limit."""
    buffer = ReplayBuffer(capacity=5)
    for i in range(10):
        buffer.push(
            state=[i],
            action=i,
            reward=float(i),
            next_state=[i+1],
            done=False
        )
    assert len(buffer) == 5  # Should not exceed capacity


def test_replay_buffer_sample():
    """Test sampling from buffer."""
    buffer = ReplayBuffer(capacity=100)

    # Add some experiences
    for i in range(20):
        buffer.push(
            state=[i, i+1],
            action=i % 5,
            reward=float(i),
            next_state=[i+1, i+2],
            done=i % 10 == 0
        )

    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=5)

    assert len(states) == 5
    assert len(actions) == 5
    assert len(rewards) == 5
    assert len(next_states) == 5
    assert len(dones) == 5


def test_replay_buffer_sample_returns_correct_types():
    """Test that sampled batch has correct structure."""
    buffer = ReplayBuffer(capacity=50)

    # Add experiences
    for i in range(10):
        buffer.push(
            state=[float(i)],
            action=i,
            reward=float(i * 2),
            next_state=[float(i + 1)],
            done=False
        )

    states, actions, rewards, next_states, dones = buffer.sample(3)

    # Check types
    assert isinstance(states, list)
    assert isinstance(actions, list)
    assert isinstance(rewards, list)
    assert isinstance(next_states, list)
    assert isinstance(dones, list)

    # Check individual elements
    assert isinstance(states[0], list)
    assert isinstance(actions[0], int)
    assert isinstance(rewards[0], float)
    assert isinstance(next_states[0], list)
    assert isinstance(dones[0], bool)


def test_replay_buffer_sample_randomness():
    """Test that sampling is random."""
    buffer = ReplayBuffer(capacity=100)

    # Add experiences
    for i in range(50):
        buffer.push(
            state=[i],
            action=i,
            reward=float(i),
            next_state=[i+1],
            done=False
        )

    # Sample twice
    _, actions1, _, _, _ = buffer.sample(10)
    _, actions2, _, _, _ = buffer.sample(10)

    # With 50 items and sampling 10, it's very unlikely to get same sequence
    assert actions1 != actions2


def test_replay_buffer_is_ready():
    """Test is_ready method."""
    buffer = ReplayBuffer(capacity=100)

    assert not buffer.is_ready(5)  # Empty buffer

    for i in range(3):
        buffer.push(
            state=[i],
            action=i,
            reward=0.0,
            next_state=[i+1],
            done=False
        )

    assert not buffer.is_ready(5)  # Only 3 items
    assert buffer.is_ready(3)  # Exactly 3 items
    assert buffer.is_ready(2)  # More than 2 items


def test_replay_buffer_sample_full_batch():
    """Test sampling when batch_size equals buffer size."""
    buffer = ReplayBuffer(capacity=10)

    for i in range(10):
        buffer.push(
            state=[i],
            action=i,
            reward=float(i),
            next_state=[i+1],
            done=False
        )

    states, actions, rewards, next_states, dones = buffer.sample(10)

    # Should return all items (in random order)
    assert len(actions) == 10
    assert sorted(actions) == list(range(10))
