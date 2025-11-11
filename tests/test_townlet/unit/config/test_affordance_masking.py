"""Unit tests for affordance masking DTOs."""

from __future__ import annotations

import pytest

from townlet.config.affordance_masking import BarConstraint, ModeConfig


class TestBarConstraint:
    def test_requires_min_or_max(self):
        with pytest.raises(ValueError, match="At least one of 'min' or 'max'"):
            BarConstraint(meter="energy")

    def test_min_less_than_max(self):
        with pytest.raises(ValueError, match="min .* must be < max"):
            BarConstraint(meter="energy", min=0.5, max=0.2)

    def test_valid_constraint(self):
        constraint = BarConstraint(meter="energy", min=0.2, max=0.8)
        assert constraint.min == 0.2
        assert constraint.max == 0.8


class TestModeConfig:
    @pytest.mark.parametrize("hours", [(-1, 5), (0, 24)])
    def test_hours_validation_bounds(self, hours):
        with pytest.raises(ValueError, match="hour must be within 0-23"):
            ModeConfig(hours=hours)  # type: ignore[arg-type]

    def test_valid_hours(self):
        cfg = ModeConfig(hours=(8, 18), effects={"energy": 0.1})
        assert cfg.hours == (8, 18)
        assert cfg.effects == {"energy": 0.1}
