"""Unit tests for variable-size meter configuration (TASK-001 Phase 1).

Tests that BarsConfig accepts variable meter counts (1-32) instead of
hardcoded 8 meters.
"""

import pytest

from townlet.environment.cascade_config import (
    BarConfig,
    BarsConfig,
    load_bars_config,
)


class TestVariableMeterConfigValidation:
    """Test that BarsConfig accepts variable meter counts."""

    def test_1_meter_config_validates(self, tmp_path):
        """Minimum valid: 1-meter config should validate successfully."""
        config = BarsConfig(
            version="2.0",
            description="1-meter minimal",
            bars=[
                BarConfig(
                    name="energy",
                    index=0,
                    tier="pivotal",
                    range=[0.0, 1.0],
                    initial=1.0,
                    base_depletion=0.005,
                    description="Energy level",
                ),
            ],
            terminal_conditions=[],
        )

        assert config.meter_count == 1
        assert config.meter_names == ["energy"]

    def test_4_meter_config_validates(self, tmp_path):
        """4-meter config should validate successfully."""
        bars = [
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.005,
                description="Energy level",
            ),
            BarConfig(
                name="health",
                index=1,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.0,
                description="Health status",
            ),
            BarConfig(
                name="money",
                index=2,
                tier="resource",
                range=[0.0, 1.0],
                initial=0.5,
                base_depletion=0.0,
                description="Financial resources",
            ),
            BarConfig(
                name="mood",
                index=3,
                tier="secondary",
                range=[0.0, 1.0],
                initial=0.7,
                base_depletion=0.001,
                description="Mood state",
            ),
        ]

        config = BarsConfig(
            version="2.0",
            description="4-meter tutorial",
            bars=bars,
            terminal_conditions=[],
        )

        # Should NOT raise
        assert config.meter_count == 4
        assert config.meter_names == ["energy", "health", "money", "mood"]

    def test_12_meter_config_validates(self, tmp_path):
        """12-meter config should validate successfully."""
        bars = [
            BarConfig(
                name=f"meter_{i}",
                index=i,
                tier="secondary",
                range=[0.0, 1.0],
                initial=0.5,
                base_depletion=0.001,
                description=f"Meter {i} description",
            )
            for i in range(12)
        ]

        config = BarsConfig(
            version="2.0",
            description="12-meter complex",
            bars=bars,
            terminal_conditions=[],
        )

        assert config.meter_count == 12
        assert len(config.meter_names) == 12

    def test_32_meter_config_validates(self, tmp_path):
        """Maximum valid: 32-meter config should validate successfully."""
        bars = [
            BarConfig(
                name=f"meter_{i}",
                index=i,
                tier="secondary",
                range=[0.0, 1.0],
                initial=0.5,
                base_depletion=0.001,
                description=f"Meter {i} description",
            )
            for i in range(32)
        ]

        config = BarsConfig(
            version="2.0",
            description="32-meter maximum",
            bars=bars,
            terminal_conditions=[],
        )

        assert config.meter_count == 32
        assert len(config.meter_names) == 32

    def test_existing_8_meter_config_still_validates(self, test_config_pack_path):
        """Backward compatibility: 8-meter configs still work."""
        config = load_bars_config(test_config_pack_path / "bars.yaml")
        assert config.meter_count == 8

    def test_zero_meters_rejected(self):
        """0 meters should be rejected (minimum is 1)."""
        with pytest.raises(ValueError, match="Must have at least 1 meter"):
            BarsConfig(
                version="2.0",
                description="Invalid: zero meters",
                bars=[],  # Empty list
                terminal_conditions=[],
            )

    def test_33_meters_rejected(self):
        """33 meters should be rejected (maximum is 32)."""
        bars = [
            BarConfig(
                name=f"meter_{i}",
                index=i,
                tier="secondary",
                range=[0.0, 1.0],
                initial=0.5,
                base_depletion=0.001,
                description=f"Meter {i} description",
            )
            for i in range(33)
        ]

        with pytest.raises(ValueError, match="Too many meters"):
            BarsConfig(
                version="2.0",
                description="Invalid: too many meters",
                bars=bars,
                terminal_conditions=[],
            )

    def test_non_contiguous_indices_rejected(self):
        """Bar indices must be contiguous from 0."""
        bars = [
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.005,
                description="Energy level",
            ),
            BarConfig(
                name="health",
                index=2,  # Gap! Missing index 1
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.0,
                description="Health status",
            ),
        ]

        with pytest.raises(ValueError, match="must be contiguous"):
            BarsConfig(
                version="2.0",
                description="Invalid: non-contiguous indices",
                bars=bars,
                terminal_conditions=[],
            )

    def test_duplicate_indices_rejected(self):
        """Duplicate bar indices should be rejected."""
        bars = [
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.005,
                description="Energy level",
            ),
            BarConfig(
                name="health",
                index=0,  # Duplicate index!
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.0,
                description="Health status",
            ),
        ]

        with pytest.raises(ValueError, match="must be contiguous"):
            BarsConfig(
                version="2.0",
                description="Invalid: duplicate indices",
                bars=bars,
                terminal_conditions=[],
            )

    def test_meter_count_property_works(self, tmp_path):
        """meter_count property returns correct count."""
        bars = [
            BarConfig(
                name=f"meter_{i}",
                index=i,
                tier="secondary",
                range=[0.0, 1.0],
                initial=0.5,
                base_depletion=0.001,
                description=f"Meter {i} description",
            )
            for i in range(7)
        ]

        config = BarsConfig(
            version="2.0",
            description="7-meter config",
            bars=bars,
            terminal_conditions=[],
        )

        assert config.meter_count == 7
        assert config.meter_count == len(config.bars)

    def test_meter_names_property_sorted_by_index(self):
        """meter_names property returns names in index order."""
        bars = [
            BarConfig(
                name="health",
                index=1,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.0,
                description="Health status",
            ),
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.005,
                description="Energy level",
            ),
            BarConfig(
                name="money",
                index=2,
                tier="resource",
                range=[0.0, 1.0],
                initial=0.5,
                base_depletion=0.0,
                description="Financial resources",
            ),
        ]

        config = BarsConfig(
            version="2.0",
            description="Test ordering",
            bars=bars,  # Deliberately out of order
            terminal_conditions=[],
        )

        # Should be sorted by index
        assert config.meter_names == ["energy", "health", "money"]

    def test_meter_name_to_index_mapping(self):
        """meter_name_to_index property creates correct mapping."""
        bars = [
            BarConfig(
                name="energy",
                index=0,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.005,
                description="Energy level",
            ),
            BarConfig(
                name="health",
                index=1,
                tier="pivotal",
                range=[0.0, 1.0],
                initial=1.0,
                base_depletion=0.0,
                description="Health status",
            ),
            BarConfig(
                name="money",
                index=2,
                tier="resource",
                range=[0.0, 1.0],
                initial=0.5,
                base_depletion=0.0,
                description="Financial resources",
            ),
        ]

        config = BarsConfig(
            version="2.0",
            description="Test mapping",
            bars=bars,
            terminal_conditions=[],
        )

        mapping = config.meter_name_to_index
        assert mapping["energy"] == 0
        assert mapping["health"] == 1
        assert mapping["money"] == 2
        assert len(mapping) == 3
