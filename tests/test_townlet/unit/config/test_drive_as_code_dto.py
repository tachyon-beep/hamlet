"""
Tests for Drive As Code (DAC) DTO layer.

Task 1.4: IntrinsicStrategyConfig DTO tests.
Task 1.5: Shaping Bonus DTOs tests.
Task 1.6: Top-level DAC config tests.
"""

import pytest
from pydantic import ValidationError

# Note: Import will fail until we create drive_as_code.py
# This is expected TDD behavior - tests first, implementation second
from townlet.config.drive_as_code import (
    ApproachRewardConfig,
    BarBonusConfig,
    CompletionBonusConfig,
    CompositionConfig,
    DriveAsCodeConfig,
    ExtrinsicStrategyConfig,
    IntrinsicStrategyConfig,
    ModifierConfig,
    RangeConfig,
    TriggerCondition,
    VFSVariableBonusConfig,
)


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


class TestTriggerCondition:
    """Test TriggerCondition validation."""

    def test_trigger_with_above(self):
        """Trigger when value is above threshold."""
        trigger = TriggerCondition(
            source="bar",
            name="energy",
            above=0.3,
        )
        assert trigger.source == "bar"
        assert trigger.name == "energy"
        assert trigger.above == 0.3
        assert trigger.below is None

    def test_trigger_with_below(self):
        """Trigger when value is below threshold."""
        trigger = TriggerCondition(
            source="variable",
            name="energy_urgency",
            below=0.7,
        )
        assert trigger.source == "variable"
        assert trigger.below == 0.7

    def test_must_specify_above_or_below(self):
        """Must specify at least one threshold."""
        with pytest.raises(ValidationError, match="Must specify 'above' or 'below'"):
            TriggerCondition(
                source="bar",
                name="energy",
            )


class TestApproachRewardConfig:
    """Test ApproachRewardConfig validation."""

    def test_valid_approach_reward(self):
        """Valid approach reward configuration."""
        config = ApproachRewardConfig(
            type="approach_reward",
            weight=0.5,
            target_affordance="Bed",
            max_distance=10.0,
        )
        assert config.type == "approach_reward"
        assert config.weight == 0.5
        assert config.target_affordance == "Bed"
        assert config.max_distance == 10.0


class TestCompletionBonusConfig:
    """Test CompletionBonusConfig validation."""

    def test_valid_completion_bonus(self):
        """Valid completion bonus configuration."""
        config = CompletionBonusConfig(
            type="completion_bonus",
            weight=1.0,
            affordance="Bed",
        )
        assert config.type == "completion_bonus"
        assert config.weight == 1.0
        assert config.affordance == "Bed"

    def test_completion_bonus_different_affordance(self):
        """Completion bonus for different affordance."""
        config = CompletionBonusConfig(
            type="completion_bonus",
            weight=2.5,
            affordance="Hospital",
        )
        assert config.weight == 2.5
        assert config.affordance == "Hospital"


class TestVFSVariableBonusConfig:
    """Test VFSVariableBonusConfig (shaping bonus) validation."""

    def test_valid_vfs_variable_bonus(self):
        """Valid VFS variable bonus configuration."""
        config = VFSVariableBonusConfig(
            type="vfs_variable",
            variable="custom_shaping_signal",
            weight=1.0,
        )
        assert config.type == "vfs_variable"
        assert config.variable == "custom_shaping_signal"
        assert config.weight == 1.0


class TestCompositionConfig:
    """Test CompositionConfig validation."""

    def test_default_composition(self):
        """Default composition configuration."""
        config = CompositionConfig()
        assert config.normalize is False
        assert config.clip is None
        assert config.log_components is True
        assert config.log_modifiers is True

    def test_with_clipping(self):
        """Composition with clipping."""
        config = CompositionConfig(
            clip={"min": -10.0, "max": 100.0},
        )
        assert config.clip == {"min": -10.0, "max": 100.0}


class TestDriveAsCodeConfig:
    """Test DriveAsCodeConfig validation."""

    def test_minimal_valid_config(self):
        """Minimal valid DAC configuration."""
        config = DriveAsCodeConfig(
            version="1.0",
            modifiers={},
            extrinsic=ExtrinsicStrategyConfig(
                type="multiplicative",
                base=1.0,
                bars=["energy", "health"],
            ),
            intrinsic=IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=0.100,
            ),
            shaping=[],
            composition=CompositionConfig(),
        )
        assert config.version == "1.0"
        assert len(config.modifiers) == 0

    def test_full_config_with_modifiers(self):
        """Full configuration with modifiers and shaping."""
        config = DriveAsCodeConfig(
            version="1.0",
            modifiers={
                "energy_crisis": ModifierConfig(
                    bar="energy",
                    ranges=[
                        RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                        RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
                    ],
                ),
            },
            extrinsic=ExtrinsicStrategyConfig(
                type="constant_base_with_shaped_bonus",
                base_reward=1.0,
                bar_bonuses=[
                    BarBonusConfig(bar="energy", center=0.5, scale=0.5),
                ],
            ),
            intrinsic=IntrinsicStrategyConfig(
                strategy="rnd",
                base_weight=0.100,
                apply_modifiers=["energy_crisis"],
            ),
            shaping=[
                ApproachRewardConfig(
                    type="approach_reward",
                    weight=1.0,
                    target_affordance="Bed",
                    max_distance=10.0,
                ),
            ],
            composition=CompositionConfig(log_components=True),
        )
        assert len(config.modifiers) == 1
        assert "energy_crisis" in config.modifiers
        assert len(config.shaping) == 1

    def test_validates_modifier_references(self):
        """DAC config validates that referenced modifiers exist."""
        with pytest.raises(ValidationError, match="undefined modifier"):
            DriveAsCodeConfig(
                version="1.0",
                modifiers={},  # No modifiers defined!
                extrinsic=ExtrinsicStrategyConfig(
                    type="multiplicative",
                    base=1.0,
                    bars=["energy"],
                ),
                intrinsic=IntrinsicStrategyConfig(
                    strategy="rnd",
                    base_weight=0.100,
                    apply_modifiers=["nonexistent_modifier"],  # References undefined modifier!
                ),
            )
