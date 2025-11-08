"""Tests for ExplorationConfig DTO (Cycle 5)."""


import pytest
from pydantic import ValidationError

from tests.test_townlet.unit.config.fixtures import (
    PRODUCTION_CONFIG_PACKS,
    VALID_EXPLORATION_PARAMS,
    make_temp_yaml,
    make_valid_params,
)
from townlet.config.exploration import ExplorationConfig, load_exploration_config


class TestExplorationConfigValidation:
    """Test ExplorationConfig schema validation (no-defaults principle)."""

    def test_all_fields_required(self):
        """All fields must be explicitly specified (no defaults)."""
        with pytest.raises(ValidationError) as exc_info:
            ExplorationConfig()

        error = str(exc_info.value)
        # Check that key fields are mentioned as missing
        required_fields = ["embed_dim", "initial_intrinsic_weight", "variance_threshold", "survival_window"]
        assert any(field in error for field in required_fields)

    def test_valid_config(self):
        """Valid config with all required fields loads successfully."""
        config = ExplorationConfig(**VALID_EXPLORATION_PARAMS)
        assert config.embed_dim == 128
        assert config.initial_intrinsic_weight == 1.0
        assert config.variance_threshold == 100.0
        assert config.survival_window == 100

    def test_embed_dim_must_be_positive(self):
        """embed_dim must be > 0."""
        # Zero embed_dim
        with pytest.raises(ValidationError) as exc_info:
            ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, embed_dim=0))
        assert "embed_dim" in str(exc_info.value).lower()

        # Negative embed_dim
        with pytest.raises(ValidationError) as exc_info:
            ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, embed_dim=-10))
        assert "embed_dim" in str(exc_info.value).lower()

    def test_initial_intrinsic_weight_must_be_non_negative(self):
        """initial_intrinsic_weight must be >= 0.0."""
        # Valid: zero weight (no intrinsic motivation)
        config = ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, initial_intrinsic_weight=0.0))
        assert config.initial_intrinsic_weight == 0.0

        # Valid: high weight (exploration priority)
        config = ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, initial_intrinsic_weight=5.0))
        assert config.initial_intrinsic_weight == 5.0

        # Invalid: negative weight
        with pytest.raises(ValidationError):
            ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, initial_intrinsic_weight=-0.1))

    def test_variance_threshold_must_be_positive(self):
        """variance_threshold must be > 0.0."""
        # Valid: low threshold (fast annealing)
        config = ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, variance_threshold=10.0))
        assert config.variance_threshold == 10.0

        # Valid: high threshold (slow annealing)
        config = ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, variance_threshold=1000.0))
        assert config.variance_threshold == 1000.0

        # Invalid: zero threshold
        with pytest.raises(ValidationError):
            ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, variance_threshold=0.0))

        # Invalid: negative threshold
        with pytest.raises(ValidationError):
            ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, variance_threshold=-100.0))

    def test_survival_window_must_be_positive(self):
        """survival_window must be > 0."""
        # Valid: small window
        config = ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, survival_window=10))
        assert config.survival_window == 10

        # Valid: large window
        config = ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, survival_window=1000))
        assert config.survival_window == 1000

        # Invalid: zero window
        with pytest.raises(ValidationError):
            ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, survival_window=0))

        # Invalid: negative window
        with pytest.raises(ValidationError):
            ExplorationConfig(**make_valid_params(VALID_EXPLORATION_PARAMS, survival_window=-50))


class TestExplorationConfigLoading:
    """Test loading ExplorationConfig from YAML."""

    def test_load_from_yaml(self, tmp_path):
        """Load exploration config from YAML file."""
        # Create exploration section in training.yaml
        yaml_path = make_temp_yaml(tmp_path, "exploration", VALID_EXPLORATION_PARAMS)

        # Create parent directory structure expected by loader
        config_dir = tmp_path
        training_yaml = config_dir / "training.yaml"
        yaml_path.rename(training_yaml)

        config = load_exploration_config(config_dir)
        assert config.embed_dim == VALID_EXPLORATION_PARAMS["embed_dim"]
        assert config.initial_intrinsic_weight == VALID_EXPLORATION_PARAMS["initial_intrinsic_weight"]
        assert config.variance_threshold == VALID_EXPLORATION_PARAMS["variance_threshold"]
        assert config.survival_window == VALID_EXPLORATION_PARAMS["survival_window"]

    def test_load_from_real_config_L0(self):  # noqa: N802
        """Load exploration config from real L0_0_minimal config pack."""
        config_dir = PRODUCTION_CONFIG_PACKS["L0_0_minimal"]
        if not config_dir.exists():
            pytest.skip(f"Config pack not found: {config_dir}")

        config = load_exploration_config(config_dir)
        # Validate it's a valid ExplorationConfig (fields are required, so if it loads it's valid)
        assert config.embed_dim > 0
        assert config.initial_intrinsic_weight >= 0.0
        assert config.variance_threshold > 0.0
        assert config.survival_window > 0

    def test_load_from_all_production_configs(self):
        """Verify all production config packs have valid exploration sections."""
        for pack_name, config_dir in PRODUCTION_CONFIG_PACKS.items():
            if not config_dir.exists():
                pytest.skip(f"Config pack not found: {config_dir}")

            # Should load without errors
            config = load_exploration_config(config_dir)
            assert config.embed_dim > 0, f"{pack_name}: embed_dim must be positive"
            assert config.variance_threshold > 0.0, f"{pack_name}: variance_threshold must be positive"
            assert config.survival_window > 0, f"{pack_name}: survival_window must be positive"

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        import yaml

        config_dir = tmp_path
        training_yaml = config_dir / "training.yaml"

        # Create YAML with missing fields
        with open(training_yaml, "w") as f:
            yaml.dump(
                {
                    "exploration": {
                        "embed_dim": 128,
                        # Missing: initial_intrinsic_weight, variance_threshold, survival_window
                    }
                },
                f,
            )

        with pytest.raises(ValueError) as exc_info:
            load_exploration_config(config_dir)

        error = str(exc_info.value)
        assert "exploration" in error.lower() or "validation" in error.lower()
