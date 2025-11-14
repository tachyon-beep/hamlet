"""Tests for PopulationConfig DTO (Cycle 3)."""

import pytest
from pydantic import ValidationError

from townlet.config.population import PopulationConfig, load_population_config


class TestPopulationConfigValidation:
    """Test PopulationConfig schema validation."""

    def test_all_fields_required(self):
        """All fields must be explicitly specified."""
        with pytest.raises(ValidationError):
            PopulationConfig()

    def test_valid_config(self, minimal_brain_config):
        """Valid config loads successfully."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=10000,
            brain_config=minimal_brain_config,
            mask_unused_obs=False,
        )
        assert config.num_agents == 1
        assert config.learning_rate == 0.00025
        assert config.network_type == "simple"

    def test_network_type_simple(self, minimal_brain_config):
        """network_type='simple' is valid."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.001,
            gamma=0.95,
            replay_buffer_capacity=5000,
            brain_config=minimal_brain_config,
            mask_unused_obs=False,
        )
        assert config.network_type == "simple"

    def test_network_type_recurrent(self, recurrent_brain_config):
        """network_type='recurrent' is valid."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=10000,
            brain_config=recurrent_brain_config,
            mask_unused_obs=False,
        )
        assert config.network_type == "recurrent"

    def test_num_agents_must_be_positive(self, minimal_brain_config):
        """num_agents must be > 0."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=0,
                learning_rate=0.00025,
                gamma=0.99,
                replay_buffer_capacity=10000,
                brain_config=minimal_brain_config,
            )

    def test_learning_rate_must_be_positive(self, minimal_brain_config):
        """learning_rate must be > 0."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.0,
                gamma=0.99,
                replay_buffer_capacity=10000,
                brain_config=minimal_brain_config,
            )

    def test_gamma_must_be_in_range(self, minimal_brain_config):
        """gamma must be in (0, 1]."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.00025,
                gamma=1.5,  # Out of range
                replay_buffer_capacity=10000,
                brain_config=minimal_brain_config,
            )

    def test_replay_buffer_must_be_positive(self, minimal_brain_config):
        """replay_buffer_capacity must be > 0."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.00025,
                gamma=0.99,
                replay_buffer_capacity=0,
                brain_config=minimal_brain_config,
            )


class TestPopulationConfigLoading:
    """Test loading from YAML files."""

    def test_load_from_yaml_simple_network(self, tmp_path):
        """Load simple network config."""
        # Create brain.yaml (required for all config packs)
        brain_file = tmp_path / "brain.yaml"
        brain_file.write_text(
            """
version: "1.0"
architecture:
  type: feedforward
  feedforward:
    hidden_layers: [256, 128]
    activation: relu
optimizer:
  type: adam
  learning_rate: 0.00025
loss:
  type: mse
q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: true
replay:
  capacity: 10000
  prioritized: false
"""
        )

        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
population:
  num_agents: 1
  network_type: simple
  mask_unused_obs: false
"""
        )

        config = load_population_config(tmp_path)
        assert config.num_agents == 1
        assert config.network_type == "simple"
        # Brain-managed fields should be None in PopulationConfig
        assert config.learning_rate is None
        assert config.gamma is None
        assert config.replay_buffer_capacity is None

    def test_load_from_yaml_recurrent_network(self, tmp_path):
        """Load recurrent network config."""
        # Create brain.yaml (required for all config packs)
        brain_file = tmp_path / "brain.yaml"
        brain_file.write_text(
            """
version: "1.0"
architecture:
  type: recurrent
  recurrent:
    vision_dims: [128]
    position_dims: [32]
    meter_dims: [32]
    lstm_hidden: 256
    q_head_dims: [128]
    activation: relu
optimizer:
  type: adam
  learning_rate: 0.0001
loss:
  type: mse
q_learning:
  gamma: 0.99
  target_update_frequency: 100
  use_double_dqn: true
replay:
  capacity: 10000
  prioritized: false
"""
        )

        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
population:
  num_agents: 1
  network_type: recurrent
  mask_unused_obs: false
"""
        )

        config = load_population_config(tmp_path)
        assert config.network_type == "recurrent"
        # Brain-managed fields should be None in PopulationConfig
        assert config.learning_rate is None
        assert config.gamma is None
        assert config.replay_buffer_capacity is None
