"""
Experience replay buffer for DRL agents.

Stores and samples experience tuples for off-policy learning.
"""

import random
from collections import deque, namedtuple


# Experience tuple structure
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """
    Fixed-size circular buffer for experience replay.

    Stores experience tuples and provides random sampling for training.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as lists
        """
        batch = random.sample(self.buffer, batch_size)

        states = [exp.state for exp in batch]
        actions = [exp.action for exp in batch]
        rewards = [exp.reward for exp in batch]
        next_states = [exp.next_state for exp in batch]
        dones = [exp.done for exp in batch]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for training.

        Args:
            batch_size: Required batch size

        Returns:
            True if buffer size >= batch_size
        """
        return len(self.buffer) >= batch_size
