"""
Observation preprocessing utilities for DRL agents.

Converts HamletEnv observations into flat vectors for neural network input.
"""

import numpy as np
import torch
from typing import Dict, Any


def preprocess_observation(obs: Dict[str, Any], grid_size: int = 8) -> np.ndarray:
    """
    Convert HamletEnv observation dict to flat vector.

    Args:
        obs: Observation dictionary from HamletEnv with keys:
            - position: np.array([x, y])
            - meters: dict of normalized meter values
            - grid: np.array of shape (height, width)
        grid_size: Grid dimension for normalization (default 8)

    Returns:
        Flat numpy array of shape (state_dim,) where:
        - [0:2]: normalized position (x/grid_size, y/grid_size)
        - [2:8]: meter values (energy, hygiene, satiation, money, stress, social)
        - [8:72]: flattened grid (64 values for 8x8)
    """
    # Extract and normalize position
    position = obs["position"] / grid_size

    # Extract meters in consistent order (all 6 meters!)
    meters = obs["meters"]
    meter_values = np.array([
        meters["energy"],
        meters["hygiene"],
        meters["satiation"],
        meters["money"],
        meters["stress"],
        meters["social"],
    ], dtype=np.float32)

    # Flatten grid
    grid = obs["grid"].flatten()

    # Concatenate all components
    state_vector = np.concatenate([
        position,
        meter_values,
        grid
    ])

    return state_vector.astype(np.float32)


def observation_to_tensor(obs: Dict[str, Any], grid_size: int = 8, device: str = "cpu") -> torch.Tensor:
    """
    Convert observation to PyTorch tensor for network input.

    Args:
        obs: Observation dictionary from HamletEnv
        grid_size: Grid dimension for normalization
        device: PyTorch device ("cpu" or "cuda")

    Returns:
        PyTorch tensor of shape (state_dim,)
    """
    state_vector = preprocess_observation(obs, grid_size)
    return torch.from_numpy(state_vector).to(device)


def get_state_dim(grid_size: int = 8) -> int:
    """
    Calculate state dimension for given grid size.

    Args:
        grid_size: Grid dimension

    Returns:
        Total state dimension (position + meters + grid)
    """
    position_dim = 2
    meters_dim = 6  # energy, hygiene, satiation, money, stress, social
    grid_dim = grid_size * grid_size

    return position_dim + meters_dim + grid_dim
