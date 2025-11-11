"""Tests for Drive As Code DTOs."""

import pytest
from pydantic import ValidationError

from townlet.config.drive_as_code import ModifierConfig, RangeConfig


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
