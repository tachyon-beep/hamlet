# tests/test_townlet/test_vectorized_env_temporal.py
import pytest
import torch
from townlet.environment.vectorized_env import VectorizedHamletEnv


@pytest.fixture
def env():
    """Create test environment with temporal mechanics enabled."""
    return VectorizedHamletEnv(
        num_agents=2,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )


def test_time_of_day_cycles():
    """Verify time cycles through 24 ticks."""
    env = VectorizedHamletEnv(
        num_agents=1,
        grid_size=8,
        device=torch.device("cpu"),
        enable_temporal_mechanics=True,
    )

    env.reset()

    # Step 24 times
    for i in range(24):
        assert env.time_of_day == i
        env.step(torch.tensor([4]))  # INTERACT action

    # Should wrap back to 0
    assert env.time_of_day == 0


def test_interaction_progress_state_exists(env):
    """Verify interaction progress state exists (tracking logic in Task 1.3)."""
    env.reset()

    # Verify state attributes exist
    assert hasattr(env, "interaction_progress")
    assert hasattr(env, "last_interaction_affordance")
    assert hasattr(env, "last_interaction_position")

    # Initial values
    assert env.interaction_progress[0] == 0
    assert env.last_interaction_affordance[0] is None


def test_observation_includes_time_and_progress(env):
    """Verify observation contains time_of_day and interaction_progress."""
    obs = env.reset()

    # Observation shape: base + temporal (2)
    # Base: 64 (grid) + 8 (meters) + (num_affordance_types + 1)
    # num_affordance_types = 15, encoding = 16
    expected_dim = 64 + 8 + (env.num_affordance_types + 1) + 2
    assert obs.shape == (2, expected_dim)

    # time_of_day should be normalized [0, 1]
    time_feature = obs[0, -2]
    assert 0.0 <= time_feature <= 1.0

    # interaction_progress should be [0, 1]
    progress_feature = obs[0, -1]
    assert progress_feature == 0.0  # No progress at start
