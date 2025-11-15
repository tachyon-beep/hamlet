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
            mask_unused_obs=False,
        )
        assert config.num_agents == 1
        assert config.mask_unused_obs is False

    def test_mask_unused_obs_true(self):
        """mask_unused_obs=True is valid."""
        config = PopulationConfig(
            num_agents=1,
            mask_unused_obs=True,
        )
        assert config.mask_unused_obs is True

    def test_mask_unused_obs_false(self):
        """mask_unused_obs=False is valid."""
        config = PopulationConfig(
            num_agents=1,
            mask_unused_obs=False,
        )
        assert config.mask_unused_obs is False

    def test_num_agents_must_be_positive(self):
        """num_agents must be > 0."""
        with pytest.raises(ValidationError):
            PopulationConfig(
                num_agents=0,
                mask_unused_obs=False,
            )


class TestPopulationConfigLoading:
    """Test loading from YAML files."""

    def test_load_from_yaml(self, tmp_path):
        """Load config from YAML."""
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
  mask_unused_obs: false
"""
        )

        config = load_population_config(tmp_path)
        assert config.num_agents == 1
        assert config.mask_unused_obs is False
        # Brain-managed fields should be None in PopulationConfig
        assert config.learning_rate is None
        assert config.gamma is None
        assert config.replay_buffer_capacity is None
