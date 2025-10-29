"""
Observation preprocessing utilities for DRL agents.

Converts HamletEnv observations into flat vectors for neural network input.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional


def preprocess_observation(obs: Dict[str, Any], grid_size: Optional[int] = None) -> np.ndarray:
    """
    Convert HamletEnv observation dict to flat vector.

    Args:
        obs: Observation dictionary from HamletEnv with keys:
            - position: np.array([x, y])
            - meters: dict of normalized meter values
            - grid: np.array of shape (height, width)
        grid_size: Optional legacy grid dimension for normalization. If not provided,
            the observation grid dimensions are used automatically.

    Returns:
        Flat numpy array of shape (state_dim,) where:
        - [0:2]: normalized position (x/max_x, y/max_y)
        - [2:8]: meter values (energy, hygiene, satiation, money, mood, social)
        - [8:]: flattened grid values
    """
    grid = obs["grid"].astype(np.float32)
    height, width = grid.shape

    # Determine normalization denominators
    if grid_size is not None:
        norm = float(max(grid_size - 1, 1))
        norm_vector = np.array([norm, norm], dtype=np.float32)
    else:
        norm_vector = np.array([
            float(max(width - 1, 1)),
            float(max(height - 1, 1)),
        ], dtype=np.float32)

    # Extract and normalize position
    position = obs["position"].astype(np.float32) / norm_vector

    # Extract meters in consistent order (all 6 meters!)
    meters = obs["meters"]
    meter_values = np.array([
        meters["energy"],
        meters["hygiene"],
        meters["satiation"],
        meters["money"],
        meters["mood"],
        meters["social"],
    ], dtype=np.float32)

    # Flatten grid
    grid_flat = grid.flatten()

    # Concatenate all components
    state_vector = np.concatenate([
        position,
        meter_values,
        grid_flat
    ])

    return state_vector.astype(np.float32)


def observation_to_tensor(obs: Dict[str, Any], grid_size: Optional[int] = None, device: str = "cpu") -> torch.Tensor:
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


def get_state_dim(grid_width: int = 8, grid_height: Optional[int] = None) -> int:
    """
    Calculate state dimension for given grid size.

    Args:
        grid_width: Grid width
        grid_height: Optional grid height (defaults to width when not provided)

    Returns:
        Total state dimension (position + meters + grid)
    """
    position_dim = 2
    meters_dim = 6  # energy, hygiene, satiation, money, mood, social
    if grid_height is None:
        grid_height = grid_width
    grid_dim = grid_width * grid_height

    return position_dim + meters_dim + grid_dim
