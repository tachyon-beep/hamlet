"""Tests for TrainingEnvironmentConfig DTO (Cycle 2).

NOTE: Named TrainingEnvironmentConfig (not EnvironmentConfig) to avoid conflict
with existing cascade_config.EnvironmentConfig (bars + cascades mechanics).

This DTO covers training.yaml's 'environment' section:
- Grid parameters (grid_size)
- Observability (partial_observability, vision_range)
- Temporal mechanics (enable_temporal_mechanics)
- Enabled affordances (curriculum selection)
- Energy costs (action depletion rates)
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from townlet.config.environment import TrainingEnvironmentConfig, load_environment_config


class TestTrainingEnvironmentConfigValidation:
    """Test TrainingEnvironmentConfig schema validation (no-defaults principle)."""

    def test_all_fields_required(self):
        """All fields must be explicitly specified (no defaults)."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingEnvironmentConfig()

        error = str(exc_info.value)
        # Check that key fields are mentioned as missing
        required_fields = ["grid_size", "partial_observability", "enabled_affordances"]
        assert any(field in error for field in required_fields)

    def test_valid_config_full_observability(self):
        """Valid config for full observability (standard)."""
        config = TrainingEnvironmentConfig(
            grid_size=8,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            enabled_affordances=None,  # None = all affordances
            energy_move_depletion=0.005,
            energy_wait_depletion=0.001,
            energy_interact_depletion=0.0,
        )
        assert config.grid_size == 8
        assert config.partial_observability is False
        assert config.enabled_affordances is None

    def test_valid_config_partial_observability(self):
        """Valid config for POMDP (partial observability)."""
        config = TrainingEnvironmentConfig(
            grid_size=8,
            partial_observability=True,
            vision_range=2,  # 5×5 window
            enable_temporal_mechanics=False,
            enabled_affordances=None,
            energy_move_depletion=0.005,
            energy_wait_depletion=0.001,
            energy_interact_depletion=0.0,
        )
        assert config.partial_observability is True
        assert config.vision_range == 2

    def test_valid_config_with_enabled_affordances_list(self):
        """Valid config with subset of affordances (curriculum learning)."""
        config = TrainingEnvironmentConfig(
            grid_size=3,
            partial_observability=False,
            vision_range=3,
            enable_temporal_mechanics=False,
            enabled_affordances=["Bed"],  # Level 0: only Bed
            energy_move_depletion=0.005,
            energy_wait_depletion=0.003,
            energy_interact_depletion=0.0029,
        )
        assert config.enabled_affordances == ["Bed"]
        assert len(config.enabled_affordances) == 1

    def test_grid_size_must_be_positive(self):
        """grid_size must be > 0."""
        with pytest.raises(ValidationError):
            TrainingEnvironmentConfig(
                grid_size=0,  # Must be gt=0
                partial_observability=False,
                vision_range=0,
                enable_temporal_mechanics=False,
                enabled_affordances=None,
                energy_move_depletion=0.005,
                energy_wait_depletion=0.001,
                energy_interact_depletion=0.0,
            )

    def test_vision_range_must_be_non_negative(self):
        """vision_range must be >= 0."""
        with pytest.raises(ValidationError):
            TrainingEnvironmentConfig(
                grid_size=8,
                partial_observability=False,
                vision_range=-1,  # Must be ge=0
                enable_temporal_mechanics=False,
                enabled_affordances=None,
                energy_move_depletion=0.005,
                energy_wait_depletion=0.001,
                energy_interact_depletion=0.0,
            )

    def test_energy_costs_must_be_non_negative(self):
        """Energy depletion rates must be >= 0."""
        with pytest.raises(ValidationError):
            TrainingEnvironmentConfig(
                grid_size=8,
                partial_observability=False,
                vision_range=2,
                enable_temporal_mechanics=False,
                enabled_affordances=None,
                energy_move_depletion=-0.005,  # Must be ge=0
                energy_wait_depletion=0.001,
                energy_interact_depletion=0.0,
            )

    def test_enabled_affordances_can_be_null(self):
        """enabled_affordances=null means all affordances enabled."""
        config = TrainingEnvironmentConfig(
            grid_size=8,
            partial_observability=False,
            vision_range=8,
            enable_temporal_mechanics=False,
            enabled_affordances=None,  # null = all
            energy_move_depletion=0.005,
            energy_wait_depletion=0.001,
            energy_interact_depletion=0.0,
        )
        assert config.enabled_affordances is None

    def test_enabled_affordances_can_be_list(self):
        """enabled_affordances can be a list of affordance names."""
        config = TrainingEnvironmentConfig(
            grid_size=7,
            partial_observability=False,
            vision_range=7,
            enable_temporal_mechanics=False,
            enabled_affordances=["Bed", "Hospital", "HomeMeal", "Job"],
            energy_move_depletion=0.005,
            energy_wait_depletion=0.001,
            energy_interact_depletion=0.0,
        )
        assert len(config.enabled_affordances) == 4
        assert "Bed" in config.enabled_affordances

    def test_enabled_affordances_cannot_be_empty_list(self):
        """enabled_affordances cannot be empty list (use null instead)."""
        with pytest.raises(ValidationError) as exc_info:
            TrainingEnvironmentConfig(
                grid_size=8,
                partial_observability=False,
                vision_range=8,
                enable_temporal_mechanics=False,
                enabled_affordances=[],  # Empty list invalid
                energy_move_depletion=0.005,
                energy_wait_depletion=0.001,
                energy_interact_depletion=0.0,
            )

        error = str(exc_info.value)
        assert "enabled_affordances" in error
        assert "empty" in error.lower() or "null" in error.lower()


class TestTrainingEnvironmentConfigCrossFieldValidation:
    """Test cross-field validation logic."""

    def test_pomdp_requires_reasonable_vision_range(self):
        """POMDP should have vision_range <= grid_size (warning, not error)."""
        import logging
        import warnings

        # This should succeed but warn
        config = TrainingEnvironmentConfig(
            grid_size=8,
            partial_observability=True,
            vision_range=10,  # Larger than grid
            enable_temporal_mechanics=False,
            enabled_affordances=None,
            energy_move_depletion=0.005,
            energy_wait_depletion=0.001,
            energy_interact_depletion=0.0,
        )

        # Config created successfully (permissive semantics)
        assert config.vision_range == 10

    def test_full_obs_vision_range_ignored(self):
        """Full observability ignores vision_range (just stored)."""
        config = TrainingEnvironmentConfig(
            grid_size=8,
            partial_observability=False,
            vision_range=2,  # Ignored when partial_observability=False
            enable_temporal_mechanics=False,
            enabled_affordances=None,
            energy_move_depletion=0.005,
            energy_wait_depletion=0.001,
            energy_interact_depletion=0.0,
        )
        # Vision range stored but not used
        assert config.vision_range == 2
        assert config.partial_observability is False


class TestTrainingEnvironmentConfigLoading:
    """Test loading from YAML files."""

    def test_load_from_yaml_full_obs(self, tmp_path):
        """Load full observability config from YAML."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
environment:
  grid_size: 8
  partial_observability: false
  vision_range: 8
  enable_temporal_mechanics: false
  enabled_affordances: null
  energy_move_depletion: 0.005
  energy_wait_depletion: 0.001
  energy_interact_depletion: 0.0
""")

        config = load_environment_config(tmp_path)

        assert config.grid_size == 8
        assert config.partial_observability is False
        assert config.enabled_affordances is None

    def test_load_from_yaml_pomdp(self, tmp_path):
        """Load POMDP config from YAML."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
environment:
  grid_size: 8
  partial_observability: true
  vision_range: 2
  enable_temporal_mechanics: false
  enabled_affordances: null
  energy_move_depletion: 0.005
  energy_wait_depletion: 0.001
  energy_interact_depletion: 0.0
""")

        config = load_environment_config(tmp_path)

        assert config.partial_observability is True
        assert config.vision_range == 2

    def test_load_from_yaml_with_affordance_list(self, tmp_path):
        """Load config with enabled affordances list."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
environment:
  grid_size: 3
  partial_observability: false
  vision_range: 3
  enable_temporal_mechanics: false
  enabled_affordances:
    - Bed
  energy_move_depletion: 0.005
  energy_wait_depletion: 0.003
  energy_interact_depletion: 0.0029
""")

        config = load_environment_config(tmp_path)

        assert config.enabled_affordances == ["Bed"]

    def test_load_from_yaml_temporal_mechanics(self, tmp_path):
        """Load config with temporal mechanics enabled."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
environment:
  grid_size: 8
  partial_observability: false
  vision_range: 8
  enable_temporal_mechanics: true
  enabled_affordances: null
  energy_move_depletion: 0.005
  energy_wait_depletion: 0.001
  energy_interact_depletion: 0.0
""")

        config = load_environment_config(tmp_path)

        assert config.enable_temporal_mechanics is True

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
environment:
  grid_size: 8
  partial_observability: false
  # Missing other required fields
""")

        with pytest.raises(ValueError) as exc_info:
            load_environment_config(tmp_path)

        error = str(exc_info.value)
        assert "environment" in error.lower() or "validation failed" in error.lower()

    def test_load_invalid_grid_size_error(self, tmp_path):
        """Invalid grid_size raises clear error."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
environment:
  grid_size: -1
  partial_observability: false
  vision_range: 2
  enable_temporal_mechanics: false
  enabled_affordances: null
  energy_move_depletion: 0.005
  energy_wait_depletion: 0.001
  energy_interact_depletion: 0.0
""")

        with pytest.raises(ValueError) as exc_info:
            load_environment_config(tmp_path)

        error = str(exc_info.value)
        assert "grid_size" in error.lower()
