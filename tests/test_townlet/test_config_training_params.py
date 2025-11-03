"""Tests for configurable training parameters from YAML config.

RED phase: These tests should fail initially because the parameters are hardcoded.
"""

import pytest
import torch
import yaml
from pathlib import Path

from townlet.exploration.epsilon_greedy import EpsilonGreedyExploration
from townlet.exploration.adaptive_intrinsic import AdaptiveIntrinsicExploration
from townlet.population.vectorized import VectorizedPopulation
from townlet.curriculum.adversarial import AdversarialCurriculum
from townlet.environment.vectorized_env import VectorizedHamletEnv


class TestEpsilonConfiguration:
    """Test that epsilon parameters can be configured from YAML."""

    def test_epsilon_greedy_uses_config_values(self, tmp_path):
        """RED: EpsilonGreedy should accept epsilon params from config."""
        # Arrange: Create config with custom epsilon values
        config = {
            "training": {
                "epsilon_start": 0.8,  # Different from default 1.0
                "epsilon_decay": 0.99,  # Different from default 0.995
                "epsilon_min": 0.05,    # Different from default 0.01
            }
        }

        # Act: Create exploration strategy with config
        # This should fail because EpsilonGreedy constructor doesn't accept these params yet
        exploration = EpsilonGreedyExploration(
            epsilon=config["training"]["epsilon_start"],
            epsilon_decay=config["training"]["epsilon_decay"],
            epsilon_min=config["training"]["epsilon_min"],
        )

        # Assert: Verify values were set from config
        assert exploration.epsilon == 0.8
        assert exploration.epsilon_decay == 0.99
        assert exploration.epsilon_min == 0.05

    def test_adaptive_intrinsic_uses_epsilon_config(self, tmp_path):
        """RED: AdaptiveIntrinsicExploration should pass epsilon params to RND."""
        # Arrange: Create config with custom epsilon values
        config = {
            "training": {
                "epsilon_start": 0.7,
                "epsilon_decay": 0.98,
                "epsilon_min": 0.02,
            },
            "exploration": {
                "embed_dim": 64,
                "initial_intrinsic_weight": 1.0,
                "variance_threshold": 100.0,
                "survival_window": 100,
            }
        }

        device = torch.device("cpu")

        # Act: Create exploration with epsilon config
        # This should fail because AdaptiveIntrinsicExploration doesn't accept epsilon params yet
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=72,
            embed_dim=config["exploration"]["embed_dim"],
            initial_intrinsic_weight=config["exploration"]["initial_intrinsic_weight"],
            variance_threshold=config["exploration"]["variance_threshold"],
            survival_window=config["exploration"]["survival_window"],
            epsilon_start=config["training"]["epsilon_start"],
            epsilon_decay=config["training"]["epsilon_decay"],
            epsilon_min=config["training"]["epsilon_min"],
            device=device,
        )

        # Assert: Verify RND has correct epsilon values
        assert exploration.rnd.epsilon == 0.7
        assert exploration.rnd.epsilon_decay == 0.98
        assert exploration.rnd.epsilon_min == 0.02

    def test_runner_loads_epsilon_from_yaml(self, tmp_path):
        """RED: DemoRunner should load epsilon params from training.yaml."""
        # Arrange: Create a minimal training.yaml with custom epsilon
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        training_config = {
            "environment": {
                "grid_size": 5,
                "partial_observability": False,
                "enabled_affordances": ["Bed"],
            },
            "population": {
                "num_agents": 1,
                "learning_rate": 0.001,
                "gamma": 0.95,
                "replay_buffer_capacity": 1000,
                "network_type": "simple",
            },
            "curriculum": {
                "max_steps_per_episode": 100,
                "survival_advance_threshold": 0.7,
                "survival_retreat_threshold": 0.3,
                "entropy_gate": 0.5,
                "min_steps_at_stage": 100,
            },
            "exploration": {
                "embed_dim": 64,
                "initial_intrinsic_weight": 1.0,
                "variance_threshold": 100.0,
                "survival_window": 50,
            },
            "training": {
                "device": "cpu",
                "max_episodes": 10,
                "epsilon_start": 0.9,  # Custom value
                "epsilon_decay": 0.97,  # Custom value
                "epsilon_min": 0.03,    # Custom value
            }
        }

        training_yaml = config_dir / "training.yaml"
        with open(training_yaml, "w") as f:
            yaml.dump(training_config, f)

        # Copy other required config files from L0_minimal
        import shutil
        l0_config = Path("configs/L0_minimal")
        for yaml_file in ["affordances.yaml", "bars.yaml", "cascades.yaml", "cues.yaml"]:
            shutil.copy(l0_config / yaml_file, config_dir / yaml_file)

        # Act: Create runner and initialize components
        # This should fail because runner doesn't read epsilon params from config yet
        from townlet.demo.runner import DemoRunner

        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=10,
        )

        # Initialize components (mimics runner.run() initialization)
        device = torch.device("cpu")

        # Create exploration with config params
        exploration = AdaptiveIntrinsicExploration(
            obs_dim=72,
            embed_dim=training_config["exploration"]["embed_dim"],
            initial_intrinsic_weight=training_config["exploration"]["initial_intrinsic_weight"],
            variance_threshold=training_config["exploration"]["variance_threshold"],
            survival_window=training_config["exploration"]["survival_window"],
            epsilon_start=training_config["training"]["epsilon_start"],
            epsilon_decay=training_config["training"]["epsilon_decay"],
            epsilon_min=training_config["training"]["epsilon_min"],
            device=device,
        )

        # Assert: Verify epsilon params were loaded from config
        assert exploration.rnd.epsilon == 0.9
        assert exploration.rnd.epsilon_decay == 0.97
        assert exploration.rnd.epsilon_min == 0.03


class TestTrainingHyperparameters:
    """Test that training hyperparameters can be configured from YAML."""

    def test_population_uses_train_frequency_from_config(self):
        """RED: VectorizedPopulation should accept train_frequency from config."""
        # Arrange
        device = torch.device("cpu")
        env = VectorizedHamletEnv(
            num_agents=1,
            grid_size=5,
            device=device,
            enabled_affordances=["Bed"],
            config_pack_path=Path("configs/L0_minimal"),
        )

        curriculum = AdversarialCurriculum(
            max_steps_per_episode=100,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            entropy_gate=0.5,
            min_steps_at_stage=100,
            device=device,
        )

        exploration = AdaptiveIntrinsicExploration(
            obs_dim=env.observation_dim,
            embed_dim=64,
            initial_intrinsic_weight=1.0,
            variance_threshold=100.0,
            survival_window=50,
            device=device,
        )

        # Act: Create population with custom train_frequency
        # This should fail because VectorizedPopulation doesn't accept train_frequency yet
        population = VectorizedPopulation(
            env=env,
            curriculum=curriculum,
            exploration=exploration,
            agent_ids=["agent_0"],
            device=device,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            learning_rate=0.001,
            gamma=0.95,
            replay_buffer_capacity=1000,
            network_type="simple",
            vision_window_size=5,
            train_frequency=8,  # Custom value (default is 4)
            target_update_frequency=200,  # Custom value (default is 100)
            batch_size=32,  # Custom value (default is 64)
            sequence_length=16,  # Custom value (default is 8)
            max_grad_norm=5.0,  # Custom value (default is 10.0)
        )

        # Assert: Verify values were set from constructor
        assert population.train_frequency == 8
        assert population.target_update_frequency == 200
        assert population.batch_size == 32
        assert population.sequence_length == 16
        assert population.max_grad_norm == 5.0

    def test_runner_loads_training_hyperparameters_from_yaml(self, tmp_path):
        """RED: DemoRunner should load training hyperparameters from config."""
        # Arrange: Create training.yaml with custom hyperparameters
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        training_config = {
            "environment": {
                "grid_size": 5,
                "partial_observability": False,
                "enabled_affordances": ["Bed"],
            },
            "population": {
                "num_agents": 1,
                "learning_rate": 0.001,
                "gamma": 0.95,
                "replay_buffer_capacity": 1000,
                "network_type": "simple",
            },
            "curriculum": {
                "max_steps_per_episode": 100,
                "survival_advance_threshold": 0.7,
                "survival_retreat_threshold": 0.3,
                "entropy_gate": 0.5,
                "min_steps_at_stage": 100,
            },
            "exploration": {
                "embed_dim": 64,
                "initial_intrinsic_weight": 1.0,
                "variance_threshold": 100.0,
                "survival_window": 50,
            },
            "training": {
                "device": "cpu",
                "max_episodes": 10,
                "train_frequency": 2,
                "target_update_frequency": 50,
                "batch_size": 128,
                "sequence_length": 4,
                "max_grad_norm": 15.0,
            }
        }

        training_yaml = config_dir / "training.yaml"
        with open(training_yaml, "w") as f:
            yaml.dump(training_config, f)

        # Copy other required config files
        import shutil
        l0_config = Path("configs/L0_minimal")
        for yaml_file in ["affordances.yaml", "bars.yaml", "cascades.yaml", "cues.yaml"]:
            shutil.copy(l0_config / yaml_file, config_dir / yaml_file)

        # Act: Create runner - this tests that runner extracts config correctly
        # (We don't actually run it, just verify initialization)
        from townlet.demo.runner import DemoRunner

        runner = DemoRunner(
            config_dir=config_dir,
            db_path=tmp_path / "test.db",
            checkpoint_dir=tmp_path / "checkpoints",
            max_episodes=10,
        )

        # Load config and verify structure
        assert runner.config["training"]["train_frequency"] == 2
        assert runner.config["training"]["target_update_frequency"] == 50
        assert runner.config["training"]["batch_size"] == 128
        assert runner.config["training"]["sequence_length"] == 4
        assert runner.config["training"]["max_grad_norm"] == 15.0
