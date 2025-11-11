"""Tests for Drive As Code DTOs."""

import pytest
from pydantic import ValidationError

from townlet.config.drive_as_code import RangeConfig


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
