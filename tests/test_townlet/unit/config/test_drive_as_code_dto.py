"""Tests for Drive As Code DTOs."""

import pytest
from pydantic import ValidationError

from townlet.config.drive_as_code import (
    BarBonusConfig,
    ExtrinsicStrategyConfig,
    ModifierConfig,
    RangeConfig,
    VariableBonusConfig,
)


class TestRangeConfig:
    """Test RangeConfig validation."""

    def test_valid_range(self):
        """Valid range configuration loads successfully."""
        config = RangeConfig(
            name="crisis",
            min=0.0,
            max=0.3,
            multiplier=0.0,
        )
        assert config.name == "crisis"
        assert config.min == 0.0
        assert config.max == 0.3
        assert config.multiplier == 0.0

    def test_min_must_be_less_than_max(self):
        """min must be < max."""
        with pytest.raises(ValidationError, match="min.*must be < max"):
            RangeConfig(
                name="invalid",
                min=0.5,
                max=0.3,  # min > max!
                multiplier=1.0,
            )

    def test_min_equals_max_invalid(self):
        """min == max is invalid."""
        with pytest.raises(ValidationError, match="min.*must be < max"):
            RangeConfig(
                name="invalid",
                min=0.5,
                max=0.5,  # Equal!
                multiplier=1.0,
            )


class TestModifierConfig:
    """Test ModifierConfig validation."""

    def test_valid_modifier_with_bar(self):
        """Valid modifier referencing a bar."""
        config = ModifierConfig(
            bar="energy",
            ranges=[
                RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
            ],
        )
        assert config.bar == "energy"
        assert config.variable is None
        assert len(config.ranges) == 2

    def test_valid_modifier_with_variable(self):
        """Valid modifier referencing a VFS variable."""
        config = ModifierConfig(
            variable="worst_physical_need",
            ranges=[
                RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),
            ],
        )
        assert config.variable == "worst_physical_need"
        assert config.bar is None

    def test_must_specify_bar_or_variable(self):
        """Must specify either bar or variable (not neither)."""
        with pytest.raises(ValidationError, match="Must specify either 'bar' or 'variable'"):
            ModifierConfig(
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=1.0, multiplier=0.0),
                ],
            )

    def test_cannot_specify_both_bar_and_variable(self):
        """Cannot specify both bar and variable."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            ModifierConfig(
                bar="energy",
                variable="worst_need",
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=1.0, multiplier=0.0),
                ],
            )

    def test_ranges_must_cover_zero_to_one(self):
        """Ranges must start at 0.0 and end at 1.0."""
        with pytest.raises(ValidationError, match="Ranges must start at 0.0"):
            ModifierConfig(
                bar="energy",
                ranges=[
                    RangeConfig(name="crisis", min=0.1, max=1.0, multiplier=0.0),  # Doesn't start at 0!
                ],
            )

        with pytest.raises(ValidationError, match="Ranges must end at 1.0"):
            ModifierConfig(
                bar="energy",
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=0.9, multiplier=0.0),  # Doesn't end at 1.0!
                ],
            )

    def test_ranges_cannot_have_gaps(self):
        """Ranges cannot have gaps between them."""
        with pytest.raises(ValidationError, match="Gap or overlap"):
            ModifierConfig(
                bar="energy",
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=0.3, multiplier=0.0),
                    RangeConfig(name="normal", min=0.4, max=1.0, multiplier=1.0),  # Gap: 0.3-0.4!
                ],
            )

    def test_ranges_cannot_overlap(self):
        """Ranges cannot overlap."""
        with pytest.raises(ValidationError, match="Gap or overlap"):
            ModifierConfig(
                bar="energy",
                ranges=[
                    RangeConfig(name="crisis", min=0.0, max=0.4, multiplier=0.0),
                    RangeConfig(name="normal", min=0.3, max=1.0, multiplier=1.0),  # Overlap: 0.3-0.4!
                ],
            )

    def test_at_least_one_range_required(self):
        """Must have at least one range."""
        with pytest.raises(ValidationError):
            ModifierConfig(
                bar="energy",
                ranges=[],  # Empty!
            )


class TestExtrinsicStrategyConfig:
    """Test ExtrinsicStrategyConfig validation."""

    def test_multiplicative_strategy(self):
        """Multiplicative strategy with bars."""
        config = ExtrinsicStrategyConfig(
            type="multiplicative",
            base=1.0,
            bars=["energy", "health"],
        )
        assert config.type == "multiplicative"
        assert config.base == 1.0
        assert config.bars == ["energy", "health"]

    def test_constant_base_with_shaped_bonus(self):
        """Constant base with bar and variable bonuses."""
        config = ExtrinsicStrategyConfig(
            type="constant_base_with_shaped_bonus",
            base_reward=1.0,
            bar_bonuses=[
                BarBonusConfig(bar="energy", center=0.5, scale=0.5),
                BarBonusConfig(bar="health", center=0.5, scale=0.5),
            ],
            variable_bonuses=[
                VariableBonusConfig(variable="energy_urgency", weight=0.5),
            ],
        )
        assert config.type == "constant_base_with_shaped_bonus"
        assert config.base_reward == 1.0
        assert len(config.bar_bonuses) == 2
        assert len(config.variable_bonuses) == 1

    def test_additive_unweighted_strategy(self):
        """Additive unweighted strategy."""
        config = ExtrinsicStrategyConfig(
            type="additive_unweighted",
            base=0.0,
            bars=["energy", "health", "satiation"],
        )
        assert config.type == "additive_unweighted"
        assert config.bars == ["energy", "health", "satiation"]

    def test_vfs_variable_strategy(self):
        """VFS variable strategy (delegate to VFS)."""
        config = ExtrinsicStrategyConfig(
            type="vfs_variable",
            variable="custom_reward_function",
        )
        assert config.type == "vfs_variable"
        assert config.variable == "custom_reward_function"

    def test_apply_modifiers_optional(self):
        """Modifiers are optional for extrinsic strategies."""
        config = ExtrinsicStrategyConfig(
            type="multiplicative",
            base=1.0,
            bars=["energy"],
            apply_modifiers=["energy_crisis"],
        )
        assert config.apply_modifiers == ["energy_crisis"]

        config2 = ExtrinsicStrategyConfig(
            type="multiplicative",
            base=1.0,
            bars=["energy"],
        )
        assert config2.apply_modifiers == []


class TestBarBonusConfig:
    """Test BarBonusConfig validation."""

    def test_valid_bar_bonus(self):
        """Valid bar bonus configuration."""
        config = BarBonusConfig(
            bar="energy",
            center=0.5,
            scale=0.5,
        )
        assert config.bar == "energy"
        assert config.center == 0.5
        assert config.scale == 0.5

    def test_center_must_be_in_range(self):
        """Center must be in [0, 1]."""
        with pytest.raises(ValidationError):
            BarBonusConfig(bar="energy", center=1.5, scale=0.5)

    def test_scale_must_be_positive(self):
        """Scale must be > 0."""
        with pytest.raises(ValidationError):
            BarBonusConfig(bar="energy", center=0.5, scale=-0.5)


class TestVariableBonusConfig:
    """Test VariableBonusConfig validation."""

    def test_valid_variable_bonus(self):
        """Valid variable bonus configuration."""
        config = VariableBonusConfig(
            variable="energy_urgency",
            weight=0.5,
        )
        assert config.variable == "energy_urgency"
        assert config.weight == 0.5

    def test_weight_can_be_negative(self):
        """Weight can be negative (for penalties)."""
        config = VariableBonusConfig(
            variable="bathroom_emergency",
            weight=-2.0,
        )
        assert config.weight == -2.0
