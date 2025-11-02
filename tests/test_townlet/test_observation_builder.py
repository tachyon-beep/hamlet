"""Tests for observation construction - documents behavior before extraction.

Red-Green refactoring: these tests document ACTUAL current behavior,
then we extract ObservationBuilder class, then tests should still pass (Green).
"""

import torch

from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestObservationConstruction:
    """Characterization tests for observation construction methods."""

    def test_full_observability_dimensions(self):
        """Full observations: grid + meters + affordance encoding."""
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=False,
            enable_temporal_mechanics=False,
        )
        env.reset()
        obs = env._get_observations()
        assert env.num_affordance_types == 14
        expected_dim = 64 + 8 + 15
        assert obs.shape == (4, expected_dim)
        assert obs.dtype == torch.float32

    def test_observation_dimension_property(self):
        """Environment calculates observation_dim correctly."""
        env = VectorizedHamletEnv(
            num_agents=4,
            grid_size=8,
            device=torch.device("cpu"),
            partial_observability=True,
            vision_range=2,
            enable_temporal_mechanics=True,
        )
        assert env.observation_dim == 53
        env.reset()
        obs = env._get_observations()
        assert obs.shape[1] == env.observation_dim
