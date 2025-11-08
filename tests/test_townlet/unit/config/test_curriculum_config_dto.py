"""Tests for CurriculumConfig DTO (Cycle 4)."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from townlet.config.curriculum import CurriculumConfig, load_curriculum_config


class TestCurriculumConfigValidation:
    """Test CurriculumConfig schema validation."""

    def test_all_fields_required(self):
        """All fields must be explicitly specified."""
        with pytest.raises(ValidationError):
            CurriculumConfig()

    def test_valid_config(self):
        """Valid config loads successfully."""
        config = CurriculumConfig(
            max_steps_per_episode=500,
            survival_advance_threshold=0.7,
            survival_retreat_threshold=0.3,
            entropy_gate=0.5,
            min_steps_at_stage=1000,
        )
        assert config.max_steps_per_episode == 500
        assert config.survival_advance_threshold == 0.7
        assert config.entropy_gate == 0.5

    def test_thresholds_in_range(self):
        """Thresholds must be in [0, 1]."""
        with pytest.raises(ValidationError):
            CurriculumConfig(
                max_steps_per_episode=500,
                survival_advance_threshold=1.5,  # Out of range
                survival_retreat_threshold=0.3,
                entropy_gate=0.5,
                min_steps_at_stage=1000,
            )

    def test_advance_greater_than_retreat(self):
        """advance_threshold must be > retreat_threshold."""
        with pytest.raises(ValidationError):
            CurriculumConfig(
                max_steps_per_episode=500,
                survival_advance_threshold=0.3,  # Less than retreat
                survival_retreat_threshold=0.7,
                entropy_gate=0.5,
                min_steps_at_stage=1000,
            )


class TestCurriculumConfigLoading:
    """Test loading from YAML files."""

    def test_load_from_yaml(self, tmp_path):
        """Load curriculum config from YAML."""
        config_file = tmp_path / "training.yaml"
        config_file.write_text("""
curriculum:
  max_steps_per_episode: 500
  survival_advance_threshold: 0.7
  survival_retreat_threshold: 0.3
  entropy_gate: 0.5
  min_steps_at_stage: 1000
""")

        config = load_curriculum_config(tmp_path)
        assert config.max_steps_per_episode == 500
        assert config.survival_advance_threshold == 0.7
