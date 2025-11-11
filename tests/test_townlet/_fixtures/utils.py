"""Utility fixtures shared across tests."""

from __future__ import annotations

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv

__all__ = ["sample_observations", "sample_actions"]


@pytest.fixture
def sample_observations(basic_env: VectorizedHamletEnv, device: torch.device) -> torch.Tensor:
    """Generate sample observations from the basic environment."""

    obs = basic_env.reset()
    return obs.to(device)


@pytest.fixture
def sample_actions(device: torch.device) -> torch.Tensor:
    """Generate a simple action tensor."""

    return torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
