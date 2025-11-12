"""
Tests for Drive As Code (DAC) DTO layer.

Task 1.4: IntrinsicStrategyConfig DTO tests.
"""

import pytest
from pydantic import ValidationError

# Note: Import will fail until we create drive_as_code.py
# This is expected TDD behavior - tests first, implementation second
from townlet.config.drive_as_code import IntrinsicStrategyConfig


class TestIntrinsicStrategyConfigDTO:
    """Test IntrinsicStrategyConfig validation and structure."""

    def test_valid_rnd_strategy(self):
        """Test valid RND strategy configuration."""
        config = IntrinsicStrategyConfig(
            strategy="rnd",
            base_weight=0.1,
            apply_modifiers=[],
            rnd_config={"feature_dim": 128, "learning_rate": 0.001},
        )

        assert config.strategy == "rnd"
        assert config.base_weight == 0.1
        assert config.apply_modifiers == []
        assert config.rnd_config == {"feature_dim": 128, "learning_rate": 0.001}
        assert config.icm_config is None
        assert config.count_config is None
        assert config.adaptive_config is None

    def test_valid_icm_strategy(self):
        """Test valid ICM strategy configuration."""
        config = IntrinsicStrategyConfig(
            strategy="icm",
            base_weight=0.05,
            apply_modifiers=["energy_crisis"],
            icm_config={"feature_dim": 64},
        )

        assert config.strategy == "icm"
        assert config.base_weight == 0.05
        assert config.apply_modifiers == ["energy_crisis"]
        assert config.icm_config == {"feature_dim": 64}
        assert config.rnd_config is None

    def test_valid_count_based_strategy(self):
        """Test valid count-based strategy configuration."""
        config = IntrinsicStrategyConfig(
            strategy="count_based",
            base_weight=0.2,
            apply_modifiers=[],
            count_config={"decay_rate": 0.99},
        )

        assert config.strategy == "count_based"
        assert config.count_config == {"decay_rate": 0.99}

    def test_valid_adaptive_rnd_strategy(self):
        """Test valid adaptive RND strategy configuration."""
        config = IntrinsicStrategyConfig(
            strategy="adaptive_rnd",
            base_weight=0.15,
            apply_modifiers=["temporal_decay"],
            adaptive_config={"threshold": 100.0, "min_weight": 0.01},
        )

        assert config.strategy == "adaptive_rnd"
        assert config.adaptive_config == {"threshold": 100.0, "min_weight": 0.01}

    def test_valid_none_strategy(self):
        """Test valid 'none' strategy (no intrinsic rewards)."""
        config = IntrinsicStrategyConfig(
            strategy="none",
            base_weight=0.0,
            apply_modifiers=[],
        )

        assert config.strategy == "none"
        assert config.base_weight == 0.0
        assert config.rnd_config is None
        assert config.icm_config is None
        assert config.count_config is None
        assert config.adaptive_config is None

    def test_base_weight_must_be_in_range_0_to_1(self):
        """Test that base_weight must be between 0.0 and 1.0."""
        # Valid boundary cases
        config_zero = IntrinsicStrategyConfig(strategy="rnd", base_weight=0.0, apply_modifiers=[])
        assert config_zero.base_weight == 0.0

        config_one = IntrinsicStrategyConfig(strategy="rnd", base_weight=1.0, apply_modifiers=[])
        assert config_one.base_weight == 1.0

        # Invalid: below 0
        with pytest.raises(ValidationError) as exc_info:
            IntrinsicStrategyConfig(strategy="rnd", base_weight=-0.1, apply_modifiers=[])
        assert "base_weight" in str(exc_info.value)

        # Invalid: above 1
        with pytest.raises(ValidationError) as exc_info:
            IntrinsicStrategyConfig(strategy="rnd", base_weight=1.5, apply_modifiers=[])
        assert "base_weight" in str(exc_info.value)

    def test_invalid_strategy_type(self):
        """Test that invalid strategy types are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            IntrinsicStrategyConfig(
                strategy="invalid_strategy",
                base_weight=0.1,
                apply_modifiers=[],
            )
        assert "strategy" in str(exc_info.value)

    def test_apply_modifiers_defaults_to_empty_list(self):
        """Test that apply_modifiers defaults to empty list if not provided."""
        config = IntrinsicStrategyConfig(strategy="rnd", base_weight=0.1)

        assert config.apply_modifiers == []

    def test_strategy_specific_configs_are_optional(self):
        """Test that strategy-specific configs are optional (can be None)."""
        config = IntrinsicStrategyConfig(
            strategy="rnd",
            base_weight=0.1,
            apply_modifiers=[],
            # Not providing rnd_config - should be None
        )

        assert config.rnd_config is None
        assert config.icm_config is None
        assert config.count_config is None
        assert config.adaptive_config is None

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=0.1,
                apply_modifiers=[],
                extra_field="not_allowed",  # This should fail
            )
        assert "extra_field" in str(exc_info.value)

    def test_multiple_modifiers_allowed(self):
        """Test that apply_modifiers can contain multiple modifier names."""
        config = IntrinsicStrategyConfig(
            strategy="rnd",
            base_weight=0.1,
            apply_modifiers=["energy_crisis", "temporal_decay", "health_crisis"],
        )

        assert len(config.apply_modifiers) == 3
        assert config.apply_modifiers == [
            "energy_crisis",
            "temporal_decay",
            "health_crisis",
        ]
