"""Tests for affordance randomization (generalization test)."""

import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_randomize_affordance_positions():
    """Should randomize affordance positions while maintaining validity."""
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))

    # Get initial positions
    initial_positions = env.get_affordance_positions()

    # Randomize
    env.randomize_affordance_positions()

    # Get new positions
    new_positions = env.get_affordance_positions()

    # Verify positions changed
    assert initial_positions != new_positions

    # Verify all affordances still exist
    assert set(initial_positions.keys()) == set(new_positions.keys())

    # Verify positions are valid (within grid)
    for name, pos in new_positions.items():
        assert 0 <= pos[0] < 8
        assert 0 <= pos[1] < 8


def test_get_affordance_positions():
    """Should return dict of affordance positions."""
    env = VectorizedHamletEnv(num_agents=1, grid_size=8, device=torch.device('cpu'))

    positions = env.get_affordance_positions()

    # Should have 8 affordances (Bed, Shower, HomeMeal, FastFood, Job, Gym, Bar, Recreation)
    assert len(positions) == 8
    assert 'Bed' in positions
    assert 'Shower' in positions
    assert 'HomeMeal' in positions
    assert 'FastFood' in positions
    assert 'Job' in positions
    assert 'Gym' in positions
    assert 'Bar' in positions
    assert 'Recreation' in positions

    # Each position should be (x, y) tuple
    for name, pos in positions.items():
        assert len(pos) == 2
        assert isinstance(pos[0], int)
        assert isinstance(pos[1], int)
