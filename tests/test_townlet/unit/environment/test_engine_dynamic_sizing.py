"""Unit tests for dynamic tensor sizing in engine layer (TASK-001 Phase 2).

Tests that CascadeEngine and VectorizedHamletEnv create correctly-sized
tensors based on meter_count from config instead of hardcoded 8.
"""

from typing import TYPE_CHECKING

import torch

from townlet.environment.cascade_config import load_environment_config
from townlet.environment.cascade_engine import CascadeEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from townlet.environment.vectorized_env import VectorizedHamletEnv


def _expected_observation_dim(env: "VectorizedHamletEnv") -> int:
    """Helper to compute observation dimension from compiled metadata."""

    return env.substrate.get_observation_dim() + env.meter_count + (env.num_affordance_types + 1) + 4


class TestCascadeEngineDynamicSizing:
    """Test that CascadeEngine creates correct-sized tensors."""

    def test_4meter_depletions_tensor_size(self, cpu_device, task001_config_4meter):
        """CascadeEngine should create 4-element depletion tensor for 4-meter config."""
        config = load_environment_config(task001_config_4meter)
        engine = CascadeEngine(config, device=cpu_device)

        # Depletion tensor should be [4] not [8]
        assert engine._base_depletions.shape == (
            4,
        ), f"Expected depletion tensor shape (4,) for 4-meter config, got {engine._base_depletions.shape}"

    def test_12meter_depletions_tensor_size(self, cpu_device, task001_config_12meter):
        """CascadeEngine should create 12-element depletion tensor for 12-meter config."""
        config = load_environment_config(task001_config_12meter)
        engine = CascadeEngine(config, device=cpu_device)

        # Depletion tensor should be [12] not [8]
        assert engine._base_depletions.shape == (
            12,
        ), f"Expected depletion tensor shape (12,) for 12-meter config, got {engine._base_depletions.shape}"

    def test_4meter_initial_values_tensor_size(self, cpu_device, task001_config_4meter):
        """CascadeEngine should create 4-element initial_values tensor for 4-meter config."""
        config = load_environment_config(task001_config_4meter)
        engine = CascadeEngine(config, device=cpu_device)

        # Initial values tensor should be [4] not [8]
        initial = engine.get_initial_meter_values()
        assert initial.shape == (4,), f"Expected initial_values tensor shape (4,) for 4-meter config, got {initial.shape}"

    def test_12meter_initial_values_tensor_size(self, cpu_device, task001_config_12meter):
        """CascadeEngine should create 12-element initial_values tensor for 12-meter config."""
        config = load_environment_config(task001_config_12meter)
        engine = CascadeEngine(config, device=cpu_device)

        # Initial values tensor should be [12] not [8]
        initial = engine.get_initial_meter_values()
        assert initial.shape == (12,), f"Expected initial_values tensor shape (12,) for 12-meter config, got {initial.shape}"

    def test_depletions_contain_correct_values(self, cpu_device, task001_config_4meter):
        """Depletion tensor should contain actual base_depletion values from config."""
        config = load_environment_config(task001_config_4meter)
        engine = CascadeEngine(config, device=cpu_device)

        # Expected: energy=0.005, health=0.0, money=0.0, mood=0.001
        expected = torch.tensor([0.005, 0.0, 0.0, 0.001], device=cpu_device)
        assert torch.allclose(engine._base_depletions, expected), f"Expected base_depletions {expected}, got {engine._base_depletions}"


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
