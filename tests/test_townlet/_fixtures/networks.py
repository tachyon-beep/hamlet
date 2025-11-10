"""Network fixtures for Townlet tests."""

from __future__ import annotations

import pytest
import torch

from townlet.agent.networks import RecurrentSpatialQNetwork, SimpleQNetwork
from townlet.environment.vectorized_env import VectorizedHamletEnv

__all__ = ["simple_qnetwork", "recurrent_qnetwork"]


@pytest.fixture
def simple_qnetwork(basic_env: VectorizedHamletEnv, device: torch.device) -> SimpleQNetwork:
    """Create a SimpleQNetwork for full-observability tests."""

    obs_dim = basic_env.observation_dim
    return SimpleQNetwork(obs_dim=obs_dim, action_dim=basic_env.action_dim, hidden_dim=128).to(device)


@pytest.fixture
def recurrent_qnetwork(pomdp_env: VectorizedHamletEnv, device: torch.device) -> RecurrentSpatialQNetwork:
    """Create a RecurrentSpatialQNetwork for POMDP scenarios."""

    return RecurrentSpatialQNetwork(
        action_dim=pomdp_env.action_dim,
        window_size=5,
        num_meters=pomdp_env.meter_count,
        num_affordance_types=14,
        enable_temporal_features=False,
        hidden_dim=256,
    ).to(device)
