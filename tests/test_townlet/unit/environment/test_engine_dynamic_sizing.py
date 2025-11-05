"""Unit tests for dynamic tensor sizing in engine layer (TASK-001 Phase 2).

Tests that CascadeEngine and VectorizedHamletEnv create correctly-sized
tensors based on meter_count from config instead of hardcoded 8.
"""

import torch

from townlet.environment.cascade_config import load_environment_config
from townlet.environment.cascade_engine import CascadeEngine
from townlet.environment.vectorized_env import VectorizedHamletEnv


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

    def test_4meter_observation_dim_full_obs(self, cpu_device, task001_config_4meter):
        """VectorizedHamletEnv should compute correct obs_dim for 4-meter full obs config."""
        # 4-meter config uses 8×8 Grid2D with "relative" encoding
        # Full obs: substrate_obs(2) + meters(4) + affordance_onehot(15) + temporal(4) = 25
        env = VectorizedHamletEnv(
            config_pack_path=task001_config_4meter,
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            partial_observability=False,  # Full observability
            enabled_affordances=["Bed", "Hospital", "HomeMeal", "Job"],
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
        )

        expected_dim = env.substrate.get_observation_dim() + 4 + 15 + 4  # substrate + meters + affordances + temporal
        assert (
            env.observation_dim == expected_dim
        ), f"Expected obs_dim={expected_dim} for 4-meter full obs (2+4+15+4), got {env.observation_dim}"

    def test_12meter_observation_dim_full_obs(self, cpu_device, task001_config_12meter):
        """VectorizedHamletEnv should compute correct obs_dim for 12-meter full obs config."""
        # 12-meter config has 8×8 Grid2D with "relative" encoding
        # Full obs: substrate_obs(2) + meters(12) + affordance_onehot(15) + temporal(4) = 33
        env = VectorizedHamletEnv(
            config_pack_path=task001_config_12meter,
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            partial_observability=False,  # Full observability
            enabled_affordances=["Bed"],
            vision_range=8,
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            # Minimal for testing
        )

        expected_dim = env.substrate.get_observation_dim() + 12 + 15 + 4  # substrate + meters + affordances + temporal
        assert (
            env.observation_dim == expected_dim
        ), f"Expected obs_dim={expected_dim} for 12-meter full obs (2+12+15+4), got {env.observation_dim}"

    def test_4meter_observation_dim_pomdp(self, cpu_device, task001_config_4meter):
        """VectorizedHamletEnv should compute correct obs_dim for 4-meter POMDP config."""
        # POMDP: local_grid(25) + position(2) + meters(4) + affordance_onehot(15) + temporal(4) = 50
        env = VectorizedHamletEnv(
            config_pack_path=task001_config_4meter,
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            partial_observability=True,  # POMDP
            vision_range=2,  # 5×5 window
            enabled_affordances=["Bed", "Hospital", "HomeMeal", "Job"],
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
        )

        assert env.observation_dim == 50, f"Expected obs_dim=50 for 4-meter POMDP (25+2+4+15+4), got {env.observation_dim}"

    def test_12meter_observation_dim_pomdp(self, cpu_device, task001_config_12meter):
        """VectorizedHamletEnv should compute correct obs_dim for 12-meter POMDP config."""
        # POMDP: local_grid(25) + position(2) + meters(12) + affordance_onehot(15) + temporal(4) = 58
        env = VectorizedHamletEnv(
            config_pack_path=task001_config_12meter,
            num_agents=1,
            grid_size=8,
            device=cpu_device,
            partial_observability=True,  # POMDP
            vision_range=2,  # 5×5 window
            enabled_affordances=["Bed"],
            enable_temporal_mechanics=False,
            move_energy_cost=0.005,
            wait_energy_cost=0.001,
            interact_energy_cost=0.0,
            agent_lifespan=1000,
            # Minimal for testing
        )

        assert env.observation_dim == 58, f"Expected obs_dim=58 for 12-meter POMDP (25+2+12+15+4), got {env.observation_dim}"
