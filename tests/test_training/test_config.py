"""
Tests for training configuration.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from hamlet.training.config import (
    TrainingConfig,
    EnvironmentConfig,
    AgentConfig,
    ExperimentConfig,
    MetricsConfig,
    FullConfig
)


def test_training_config_defaults():
    """Test that training config has sensible defaults."""
    config = TrainingConfig()
    assert config.num_episodes == 1000
    assert config.batch_size == 64
    assert config.learning_rate == 1e-3


def test_environment_config_defaults():
    """Test that environment config has sensible defaults."""
    config = EnvironmentConfig()
    assert config.grid_width == 8
    assert config.grid_height == 8
    assert config.initial_energy == 100.0
    assert "Bed" in config.affordance_positions
    assert "Job" in config.affordance_positions


def test_agent_config_defaults():
    """Test that agent config has sensible defaults."""
    config = AgentConfig()
    assert config.agent_id == "agent_0"
    assert config.algorithm == "dqn"
    assert config.state_dim == 72
    assert config.action_dim == 5


def test_experiment_config_defaults():
    """Test that experiment config has sensible defaults."""
    config = ExperimentConfig()
    assert config.name == "hamlet_experiment"
    assert config.tracking_uri == "mlruns"


def test_metrics_config_defaults():
    """Test that metrics config has sensible defaults."""
    config = MetricsConfig()
    assert config.tensorboard is True
    assert config.database is True
    assert config.replay_storage is True
    assert config.replay_sample_rate == 0.1


def test_full_config_defaults():
    """Test that full config combines all components."""
    config = FullConfig()
    assert len(config.agents) == 1
    assert config.agents[0].agent_id == "agent_0"
    assert config.training.num_episodes == 1000
    assert config.environment.grid_width == 8


def test_full_config_from_yaml():
    """Test loading configuration from YAML file."""
    yaml_content = """
experiment:
  name: test_experiment
  description: Test run

environment:
  grid_width: 16
  grid_height: 16

agents:
  - agent_id: agent_0
    algorithm: dqn
    learning_rate: 0.001
  - agent_id: agent_1
    algorithm: dqn
    learning_rate: 0.002

training:
  num_episodes: 500
  batch_size: 32

metrics:
  tensorboard: true
  database: false
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        config = FullConfig.from_yaml(temp_path)

        assert config.experiment.name == "test_experiment"
        assert config.environment.grid_width == 16
        assert len(config.agents) == 2
        assert config.agents[0].agent_id == "agent_0"
        assert config.agents[1].agent_id == "agent_1"
        assert config.agents[1].learning_rate == 0.002
        assert config.training.num_episodes == 500
        assert config.metrics.database is False

    finally:
        Path(temp_path).unlink()


def test_full_config_to_yaml():
    """Test saving configuration to YAML file."""
    config = FullConfig()
    config.experiment.name = "save_test"
    config.training.num_episodes = 250

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name

    try:
        config.to_yaml(temp_path)

        # Load it back
        loaded_config = FullConfig.from_yaml(temp_path)

        assert loaded_config.experiment.name == "save_test"
        assert loaded_config.training.num_episodes == 250

    finally:
        Path(temp_path).unlink()


def test_full_config_to_dict():
    """Test converting configuration to dictionary."""
    config = FullConfig()
    config_dict = config.to_dict()

    assert 'experiment' in config_dict
    assert 'environment' in config_dict
    assert 'agents' in config_dict
    assert 'training' in config_dict
    assert 'metrics' in config_dict
    assert len(config_dict['agents']) == 1
