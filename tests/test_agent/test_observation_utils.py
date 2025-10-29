"""
Tests for observation preprocessing utilities.
"""

import pytest
import numpy as np
import torch
from hamlet.agent.observation_utils import (
    preprocess_observation,
    observation_to_tensor,
    get_state_dim
)


def create_test_observation():
    """Create a test observation matching HamletEnv format."""
    return {
        "position": np.array([4.0, 3.0], dtype=np.float32),
        "meters": {
            "energy": 0.8,
            "hygiene": 0.6,
            "satiation": 0.9,
            "money": 0.5,
            "mood": 0.7,
            "social": 0.4,
        },
        "grid": np.random.rand(8, 8).astype(np.float32),
    }


def test_preprocess_observation_shape():
    """Test that preprocessing produces correct shape."""
    obs = create_test_observation()
    state = preprocess_observation(obs)

    # 2 (position) + 6 (meters) + 64 (grid) = 72
    assert state.shape == (72,)
    assert state.dtype == np.float32


def test_preprocess_observation_position_normalized():
    """Test that position is normalized correctly."""
    obs = create_test_observation()
    state = preprocess_observation(obs, grid_size=8)

    # Position normalized by (grid_size - 1) to reach 1.0 at boundaries
    assert state[0] == pytest.approx(4.0 / 7.0)
    assert state[1] == pytest.approx(3.0 / 7.0)


def test_preprocess_observation_meters_order():
    """Test that meters are in correct order."""
    obs = create_test_observation()
    state = preprocess_observation(obs)

    # Meters start at index 2 in the order: energy, hygiene, satiation, money, mood, social
    assert state[2] == pytest.approx(0.8)  # energy
    assert state[3] == pytest.approx(0.6)  # hygiene
    assert state[4] == pytest.approx(0.9)  # satiation
    assert state[5] == pytest.approx(0.5)  # money
    assert state[6] == pytest.approx(0.7)  # mood
    assert state[7] == pytest.approx(0.4)  # social


def test_preprocess_observation_grid_flattened():
    """Test that grid is flattened correctly."""
    obs = {
        "position": np.array([0.0, 0.0], dtype=np.float32),
        "meters": {name: 1.0 for name in ["energy", "hygiene", "satiation", "money", "mood", "social"]},
        "grid": np.arange(64).reshape(8, 8).astype(np.float32),
    }

    state = preprocess_observation(obs)

    # Grid starts at index 8, should be flattened row-by-row
    grid_flat = state[8:]
    assert len(grid_flat) == 64
    assert grid_flat[0] == 0.0
    assert grid_flat[63] == 63.0


def test_observation_to_tensor():
    """Test conversion to PyTorch tensor."""
    obs = create_test_observation()
    tensor = observation_to_tensor(obs, device="cpu")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (72,)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"


def test_observation_to_tensor_values_match():
    """Test that tensor values match preprocessed array."""
    obs = create_test_observation()

    state_array = preprocess_observation(obs)
    state_tensor = observation_to_tensor(obs)

    assert torch.allclose(state_tensor, torch.from_numpy(state_array))


def test_get_state_dim():
    """Test state dimension calculation."""
    assert get_state_dim(8) == 72  # 2 + 6 + 64
    assert get_state_dim(10) == 108  # 2 + 6 + 100
    assert get_state_dim(16) == 264  # 2 + 6 + 256


def test_preprocess_observation_different_grid_sizes():
    """Test preprocessing with different grid sizes."""
    for grid_size in [8, 10, 16]:
        obs = {
            "position": np.array([grid_size / 2, grid_size / 2], dtype=np.float32),
            "meters": {name: 0.5 for name in ["energy", "hygiene", "satiation", "money", "mood", "social"]},
            "grid": np.zeros((grid_size, grid_size), dtype=np.float32),
        }

        state = preprocess_observation(obs, grid_size=grid_size)

        expected_dim = 2 + 6 + grid_size * grid_size
        assert state.shape == (expected_dim,)

        expected_pos = (grid_size / 2) / max(grid_size - 1, 1)
        assert state[0] == pytest.approx(expected_pos)
        assert state[1] == pytest.approx(expected_pos)


def test_preprocess_observation_edge_positions():
    """Test preprocessing with edge positions."""
    # Top-left corner
    obs1 = create_test_observation()
    obs1["position"] = np.array([0.0, 0.0], dtype=np.float32)
    state1 = preprocess_observation(obs1)
    assert state1[0] == pytest.approx(0.0)
    assert state1[1] == pytest.approx(0.0)

    # Bottom-right corner
    obs2 = create_test_observation()
    obs2["position"] = np.array([7.0, 7.0], dtype=np.float32)
    state2 = preprocess_observation(obs2, grid_size=8)
    assert state2[0] == pytest.approx(1.0)
    assert state2[1] == pytest.approx(1.0)


def test_preprocess_observation_all_zeros():
    """Test preprocessing with all zero values."""
    obs = {
        "position": np.array([0.0, 0.0], dtype=np.float32),
        "meters": {name: 0.0 for name in ["energy", "hygiene", "satiation", "money", "mood", "social"]},
        "grid": np.zeros((8, 8), dtype=np.float32),
    }

    state = preprocess_observation(obs)

    assert state.shape == (72,)
    assert np.all(state == 0.0)


def test_preprocess_observation_all_ones():
    """Test preprocessing with all one values."""
    obs = {
        "position": np.array([7.0, 7.0], dtype=np.float32),
        "meters": {name: 1.0 for name in ["energy", "hygiene", "satiation", "money", "mood", "social"]},
        "grid": np.ones((8, 8), dtype=np.float32),
    }

    state = preprocess_observation(obs, grid_size=8)

    # Position normalized to 1.0
    assert state[0] == pytest.approx(1.0)
    assert state[1] == pytest.approx(1.0)

    # Meters all 1.0
    assert np.all(state[2:8] == 1.0)

    # Grid all 1.0
    assert np.all(state[8:] == 1.0)
