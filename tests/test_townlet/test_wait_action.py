"""
WAIT action configuration safeguards.
"""

import pytest
import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


def test_wait_energy_cost_must_be_less_than_movement_cost():
    """Environment should reject configs where WAIT is more expensive than MOVE."""
    with pytest.raises(ValueError):
        VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=torch.device("cpu"),
            move_energy_cost=0.005,
            wait_energy_cost=0.01,
        )
