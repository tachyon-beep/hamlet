"""Unit tests for dynamic tensor sizing in engine layer (TASK-001 Phase 2).

Tests that VectorizedHamletEnv creates correctly-sized
tensors based on meter_count from config instead of hardcoded 8.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from townlet.environment.vectorized_env import VectorizedHamletEnv


def _expected_observation_dim(env: "VectorizedHamletEnv") -> int:
    """Helper to use compiled metadata for observation dimension."""

    return env.metadata.observation_dim


class TestVectorizedEnvDynamicSizing:
    """Test that VectorizedHamletEnv creates correct-sized tensors."""

    def test_4meter_meters_tensor_size(self, task001_env_4meter):
        """VectorizedHamletEnv should create [num_agents, 4] meters tensor for 4-meter config."""
        env = task001_env_4meter
        num_agents = env.num_agents

        # Meters tensor should be [num_agents, 4] not [num_agents, 8]
        assert env.meters.shape == (
            num_agents,
            4,
        ), f"Expected meters tensor shape ({num_agents}, 4) for 4-meter config, got {env.meters.shape}"

    def test_12meter_meters_tensor_size(self, task001_env_12meter):
        """VectorizedHamletEnv should create [num_agents, 12] meters tensor for 12-meter config."""
        env = task001_env_12meter
        num_agents = env.num_agents

        # Meters tensor should be [num_agents, 12] not [num_agents, 8]
        assert env.meters.shape == (
            num_agents,
            12,
        ), f"Expected meters tensor shape ({num_agents}, 12) for 12-meter config, got {env.meters.shape}"

    def test_4meter_observation_dim_full_obs(self, task001_env_4meter):
        """VectorizedHamletEnv should compute correct obs_dim for 4-meter full obs config."""

        env = task001_env_4meter
        expected_dim = _expected_observation_dim(env)
        assert env.observation_dim == expected_dim, f"Expected obs_dim={expected_dim} for 4-meter full obs, got {env.observation_dim}"

    def test_12meter_observation_dim_full_obs(self, task001_env_12meter):
        """VectorizedHamletEnv should compute correct obs_dim for 12-meter full obs config."""
        env = task001_env_12meter
        expected_dim = _expected_observation_dim(env)
        assert env.observation_dim == expected_dim, f"Expected obs_dim={expected_dim} for 12-meter full obs, got {env.observation_dim}"

    def test_4meter_observation_dim_pomdp(self, task001_env_4meter_pomdp):
        """VectorizedHamletEnv should compute correct obs_dim for 4-meter POMDP config."""
        env = task001_env_4meter_pomdp
        expected_dim = _expected_observation_dim(env)
        assert env.partial_observability is True
        assert env.observation_dim == expected_dim, f"Expected obs_dim={expected_dim} for 4-meter POMDP, got {env.observation_dim}"

    def test_12meter_observation_dim_pomdp(self, task001_env_12meter_pomdp):
        """VectorizedHamletEnv should compute correct obs_dim for 12-meter POMDP config."""
        env = task001_env_12meter_pomdp
        expected_dim = _expected_observation_dim(env)
        assert env.partial_observability is True
        assert env.observation_dim == expected_dim, f"Expected obs_dim={expected_dim} for 12-meter POMDP, got {env.observation_dim}"
