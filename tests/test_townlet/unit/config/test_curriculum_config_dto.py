"""Tests for CurriculumConfig DTO (Cycle 4)."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from townlet.config.curriculum import CurriculumConfig, load_curriculum_config
from tests.test_townlet.unit.config.fixtures import (
    VALID_CURRICULUM_PARAMS,
    make_valid_params,
    make_temp_yaml,
    PRODUCTION_CONFIG_PACKS,
)


class TestCurriculumConfigValidation:
    """Test CurriculumConfig schema validation."""

    def test_all_fields_required(self):
        """All fields must be explicitly specified (no-defaults principle)."""
        with pytest.raises(ValidationError) as exc_info:
            CurriculumConfig()

        error = str(exc_info.value)
        # Check that key fields are mentioned as missing
        required_fields = ["max_steps_per_episode", "survival_advance_threshold", "entropy_gate"]
        assert any(field in error for field in required_fields)

    def test_valid_config(self):
        """Valid config with all required fields loads successfully."""
        config = CurriculumConfig(**VALID_CURRICULUM_PARAMS)
        assert config.max_steps_per_episode == VALID_CURRICULUM_PARAMS["max_steps_per_episode"]
        assert config.survival_advance_threshold == VALID_CURRICULUM_PARAMS["survival_advance_threshold"]
        assert config.entropy_gate == VALID_CURRICULUM_PARAMS["entropy_gate"]

    def test_max_steps_must_be_positive(self):
        """max_steps_per_episode must be > 0."""
        # Zero steps
        with pytest.raises(ValidationError) as exc_info:
            CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, max_steps_per_episode=0))
        assert "max_steps_per_episode" in str(exc_info.value).lower()

        # Negative steps
        with pytest.raises(ValidationError) as exc_info:
            CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, max_steps_per_episode=-100))
        assert "max_steps_per_episode" in str(exc_info.value).lower()

    def test_min_steps_must_be_positive(self):
        """min_steps_at_stage must be > 0."""
        # Zero steps
        with pytest.raises(ValidationError) as exc_info:
            CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, min_steps_at_stage=0))
        assert "min_steps_at_stage" in str(exc_info.value).lower()

        # Negative steps
        with pytest.raises(ValidationError) as exc_info:
            CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, min_steps_at_stage=-1000))
        assert "min_steps_at_stage" in str(exc_info.value).lower()

    def test_thresholds_in_range(self):
        """Thresholds must be in [0, 1]."""
        # survival_advance_threshold > 1.0
        with pytest.raises(ValidationError):
            CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, survival_advance_threshold=1.5))

        # survival_retreat_threshold > 1.0
        with pytest.raises(ValidationError):
            CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, survival_retreat_threshold=1.2))

        # entropy_gate > 1.0
        with pytest.raises(ValidationError):
            CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, entropy_gate=1.1))

        # survival_advance_threshold < 0.0
        with pytest.raises(ValidationError):
            CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, survival_advance_threshold=-0.1))

    def test_thresholds_at_boundaries(self):
        """Thresholds at exactly 0.0 and 1.0 are valid."""
        # advance=1.0, retreat=0.0 (maximum spread)
        config = CurriculumConfig(**make_valid_params(
            VALID_CURRICULUM_PARAMS,
            survival_advance_threshold=1.0,
            survival_retreat_threshold=0.0
        ))
        assert config.survival_advance_threshold == 1.0
        assert config.survival_retreat_threshold == 0.0

        # entropy_gate=0.0 (minimum)
        config = CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, entropy_gate=0.0))
        assert config.entropy_gate == 0.0

        # entropy_gate=1.0 (maximum)
        config = CurriculumConfig(**make_valid_params(VALID_CURRICULUM_PARAMS, entropy_gate=1.0))
        assert config.entropy_gate == 1.0

    def test_advance_greater_than_retreat(self):
        """advance_threshold must be > retreat_threshold."""
        # advance < retreat (invalid)
        with pytest.raises(ValidationError) as exc_info:
            CurriculumConfig(**make_valid_params(
                VALID_CURRICULUM_PARAMS,
                survival_advance_threshold=0.3,
                survival_retreat_threshold=0.7
            ))
        error = str(exc_info.value)
        assert "advance" in error.lower() or "retreat" in error.lower()

        # advance == retreat (invalid - must be strictly greater)
        with pytest.raises(ValidationError):
            CurriculumConfig(**make_valid_params(
                VALID_CURRICULUM_PARAMS,
                survival_advance_threshold=0.5,
                survival_retreat_threshold=0.5
            ))

    def test_advance_barely_greater_than_retreat(self):
        """advance_threshold can be just epsilon greater than retreat_threshold."""
        # Minimum valid spread (0.5 vs 0.49)
        config = CurriculumConfig(**make_valid_params(
            VALID_CURRICULUM_PARAMS,
            survival_advance_threshold=0.5,
            survival_retreat_threshold=0.49
        ))
        assert config.survival_advance_threshold == 0.5
        assert config.survival_retreat_threshold == 0.49


class TestCurriculumConfigLoading:
    """Test loading from YAML files."""

    def test_load_from_yaml(self, tmp_path):
        """Load curriculum config from YAML file."""
        yaml_path = make_temp_yaml(tmp_path, "curriculum", VALID_CURRICULUM_PARAMS)

        # Rename to training.yaml (expected by loader)
        training_yaml = tmp_path / "training.yaml"
        yaml_path.rename(training_yaml)

        config = load_curriculum_config(tmp_path)
        assert config.max_steps_per_episode == VALID_CURRICULUM_PARAMS["max_steps_per_episode"]
        assert config.survival_advance_threshold == VALID_CURRICULUM_PARAMS["survival_advance_threshold"]
        assert config.entropy_gate == VALID_CURRICULUM_PARAMS["entropy_gate"]

    def test_load_from_real_config_L0(self):
        """Load curriculum config from real L0_0_minimal config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L0_0_minimal"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = load_curriculum_config(config_dir)
        # Validate it's a valid CurriculumConfig (fields are required, so if it loads it's valid)
        assert config.max_steps_per_episode > 0
        assert 0.0 <= config.survival_advance_threshold <= 1.0
        assert 0.0 <= config.survival_retreat_threshold <= 1.0
        assert config.survival_advance_threshold > config.survival_retreat_threshold

    def test_load_from_all_production_configs(self):
        """Verify all production config packs have valid curriculum sections."""
        for pack_name, config_dir in PRODUCTION_CONFIG_PACKS.items():
            if not config_dir.exists():
                pytest.skip(f"Config pack not found: {config_dir}")

            # Should load without errors
            config = load_curriculum_config(config_dir)
            assert config.max_steps_per_episode > 0, f"{pack_name}: max_steps must be positive"
            assert config.min_steps_at_stage > 0, f"{pack_name}: min_steps must be positive"
            assert 0.0 <= config.entropy_gate <= 1.0, f"{pack_name}: entropy_gate must be in [0, 1]"
            assert config.survival_advance_threshold > config.survival_retreat_threshold, \
                f"{pack_name}: advance must be > retreat"

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        import yaml

        training_yaml = tmp_path / "training.yaml"

        # Create YAML with missing fields
        with open(training_yaml, "w") as f:
            yaml.dump(
                {
                    "curriculum": {
                        "max_steps_per_episode": 500,
                        # Missing: survival_advance_threshold, survival_retreat_threshold, entropy_gate, min_steps_at_stage
                    }
                },
                f,
            )

        with pytest.raises(ValueError) as exc_info:
            load_curriculum_config(tmp_path)

        error = str(exc_info.value)
        assert "curriculum" in error.lower() or "validation" in error.lower()
