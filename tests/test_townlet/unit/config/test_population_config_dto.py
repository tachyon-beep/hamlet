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

    def test_valid_config(self):
        """Valid config loads successfully."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.00025,
            gamma=0.99,
            replay_buffer_capacity=10000,
            network_type="simple",
        )
        assert config.num_agents == 1
        assert config.learning_rate == 0.00025
        assert config.network_type == "simple"

    def test_network_type_simple(self):
        """network_type='simple' is valid."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.001,
            gamma=0.95,
            replay_buffer_capacity=5000,
            network_type="simple",
        )
        assert config.network_type == "simple"

    def test_network_type_recurrent(self):
        """network_type='recurrent' is valid."""
        config = PopulationConfig(
            num_agents=1,
            learning_rate=0.0001,
            gamma=0.99,
            replay_buffer_capacity=10000,
            network_type="recurrent",
        )
        assert config.network_type == "recurrent"

    def test_num_agents_must_be_positive(self):
        """num_agents must be > 0."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=0,
                learning_rate=0.00025,
                gamma=0.99,
                replay_buffer_capacity=10000,
                network_type="simple",
            )

    def test_learning_rate_must_be_positive(self):
        """learning_rate must be > 0."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.0,
                gamma=0.99,
                replay_buffer_capacity=10000,
                network_type="simple",
            )

    def test_gamma_must_be_in_range(self):
        """gamma must be in (0, 1]."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.00025,
                gamma=1.5,  # Out of range
                replay_buffer_capacity=10000,
                network_type="simple",
            )

    def test_replay_buffer_must_be_positive(self):
        """replay_buffer_capacity must be > 0."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=1,
                learning_rate=0.00025,
                gamma=0.99,
                replay_buffer_capacity=0,
                network_type="simple",
            )


class TestPopulationConfigLoading:
    """Test loading from YAML files."""

    def test_load_from_yaml_simple_network(self, tmp_path):
        """Load simple network config."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
population:
  num_agents: 1
  learning_rate: 0.00025
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: simple
"""
        )

        config = load_population_config(tmp_path)
        assert config.num_agents == 1
        assert config.network_type == "simple"

    def test_load_from_yaml_recurrent_network(self, tmp_path):
        """Load recurrent network config."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text(
            """
population:
  num_agents: 1
  learning_rate: 0.0001
  gamma: 0.99
  replay_buffer_capacity: 10000
  network_type: recurrent
"""
        )

        config = load_population_config(tmp_path)
        assert config.network_type == "recurrent"
        assert config.learning_rate == 0.0001
