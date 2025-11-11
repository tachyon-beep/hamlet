"""Comprehensive unit tests for affordance_config.py module.

These unit tests provide:
- Full Pydantic model validation coverage
- Collection query method testing
- YAML loading with meter validation
- Operating hours utility function coverage

Target coverage: 67% → 80%+

Uncovered lines to target:
- Lines 86-95: validate_multi_tick_requirements validator
- Lines 100-111: validate_operating_hours validator
- Lines 128-131, 135, 139: Collection query methods
- Lines 158-185: load_affordance_config with meter validation
- Lines 195-199: load_default_affordances convenience function
- Lines 218-225: is_affordance_open utility function
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from townlet.environment.affordance_config import (
    AffordanceConfig,
    AffordanceConfigCollection,
    is_affordance_open,
    load_affordance_config,
)
from townlet.environment.cascade_config import BarConfig, BarsConfig, TerminalCondition

# =============================================================================
# PYDANTIC MODEL VALIDATION TESTS
# =============================================================================


class TestAffordanceConfigValidation:
    """Test AffordanceConfig Pydantic validators."""

    def test_multi_tick_requires_duration_ticks(self):
        """Should raise ValidationError when multi_tick lacks duration_ticks (line 86-87)."""
        with pytest.raises(ValidationError, match="multi_tick type requires 'duration_ticks' field"):
            AffordanceConfig(
                id="Bed",
                name="Bed",
                category="energy_restoration",
                interaction_type="multi_tick",
                # Missing duration_ticks
                operating_hours=[0, 24],
            )

    def test_dual_requires_duration_ticks(self):
        """Should raise ValidationError when dual lacks duration_ticks (line 89-90)."""
        with pytest.raises(ValidationError, match="dual type requires 'duration_ticks' field"):
            AffordanceConfig(
                id="Job",
                name="Job",
                category="income",
                interaction_type="dual",
                # Missing duration_ticks
                operating_hours=[9, 17],
            )

    def test_instant_cannot_have_duration_ticks(self):
        """Should raise ValidationError when instant has duration_ticks (line 92-93)."""
        with pytest.raises(ValidationError, match="'duration_ticks' only valid for multi_tick or dual types"):
            AffordanceConfig(
                id="Bed",
                name="Bed",
                category="energy_restoration",
                interaction_type="instant",
                duration_ticks=5,  # Invalid for instant
                operating_hours=[0, 24],
            )

    def test_continuous_cannot_have_duration_ticks(self):
        """Should raise ValidationError when continuous has duration_ticks (line 92-93)."""
        with pytest.raises(ValidationError, match="'duration_ticks' only valid for multi_tick or dual types"):
            AffordanceConfig(
                id="Shower",
                name="Shower",
                category="hygiene",
                interaction_type="continuous",
                duration_ticks=3,  # Invalid for continuous
                operating_hours=[0, 24],
            )

    def test_operating_hours_must_be_two_elements(self):
        """Should raise ValidationError when operating_hours length != 2 (line 100-101)."""
        with pytest.raises(ValidationError, match="operating_hours must be \\[open, close\\]"):
            AffordanceConfig(
                id="Bed",
                name="Bed",
                category="energy_restoration",
                interaction_type="instant",
                operating_hours=[0, 12, 24],  # Wrong length
            )

    def test_operating_hours_open_must_be_0_to_23(self):
        """Should raise ValidationError when open_hour < 0 or > 23 (line 105-106)."""
        with pytest.raises(ValidationError, match="open_hour must be 0-23"):
            AffordanceConfig(
                id="Bed",
                name="Bed",
                category="energy_restoration",
                interaction_type="instant",
                operating_hours=[-1, 24],  # Invalid open_hour
            )

        with pytest.raises(ValidationError, match="open_hour must be 0-23"):
            AffordanceConfig(
                id="Bed",
                name="Bed",
                category="energy_restoration",
                interaction_type="instant",
                operating_hours=[25, 28],  # Invalid open_hour
            )

    def test_operating_hours_close_must_be_1_to_28(self):
        """Should raise ValidationError when close_hour < 1 or > 28 (line 108-109)."""
        with pytest.raises(ValidationError, match="close_hour must be 1-28"):
            AffordanceConfig(
                id="Bed",
                name="Bed",
                category="energy_restoration",
                interaction_type="instant",
                operating_hours=[0, 0],  # Invalid close_hour (0)
            )

        with pytest.raises(ValidationError, match="close_hour must be 1-28"):
            AffordanceConfig(
                id="Bed",
                name="Bed",
                category="energy_restoration",
                interaction_type="instant",
                operating_hours=[0, 30],  # Invalid close_hour (>28)
            )

    def test_valid_multi_tick_affordance(self):
        """Should successfully create multi_tick affordance with duration_ticks."""
        config = AffordanceConfig(
            id="Bed",
            name="Bed",
            category="energy_restoration",
            interaction_type="multi_tick",
            duration_ticks=5,
            operating_hours=[0, 24],
        )

        assert config.id == "Bed"
        assert config.interaction_type == "multi_tick"
        assert config.duration_ticks == 5

    def test_valid_operating_hours_normal_range(self):
        """Should successfully create affordance with normal operating hours."""
        config = AffordanceConfig(
            id="Job",
            name="Job",
            category="income",
            interaction_type="instant",
            operating_hours=[9, 17],  # 9am-5pm
        )

        assert config.operating_hours == [9, 17]

    def test_valid_operating_hours_wraparound(self):
        """Should successfully create affordance with wraparound hours."""
        config = AffordanceConfig(
            id="Bar",
            name="Bar",
            category="social",
            interaction_type="instant",
            operating_hours=[18, 4],  # 6pm-4am (wraps midnight)
        )

        assert config.operating_hours == [18, 4]

    def test_position_formats(self):
        """Position supports list, dict, int, or None."""
        base_kwargs = dict(
            id="Bed",
            name="Bed",
            category="energy_restoration",
            interaction_type="instant",
            operating_hours=[0, 24],
            costs=[],
            effects=[{"meter": "energy", "amount": 0.5}],
        )

        assert AffordanceConfig(**base_kwargs, position=[1, 2]).position == [1, 2]
        assert AffordanceConfig(**base_kwargs, position=[1, 2, 0]).position == [1, 2, 0]
        assert AffordanceConfig(**base_kwargs, position={"q": 1, "r": 2}).position == {"q": 1, "r": 2}
        assert AffordanceConfig(**base_kwargs, position=3).position == 3

        with pytest.raises(ValueError):
            AffordanceConfig(**base_kwargs, position=[1])  # Wrong dimensionality

        with pytest.raises(ValueError):
            AffordanceConfig(**base_kwargs, position={"x": 1})  # Wrong keys

        with pytest.raises(ValueError):
            AffordanceConfig(**base_kwargs, position=-1)  # Invalid graph node id


# =============================================================================
# COLLECTION QUERY METHOD TESTS
# =============================================================================


class TestAffordanceConfigCollectionQueries:
    """Test AffordanceConfigCollection query methods."""

    def test_get_affordance_found(self):
        """Should return affordance when ID matches (line 128-130)."""
        collection = AffordanceConfigCollection(
            version="1.0",
            description="Test collection",
            status="TEST",
            affordances=[
                AffordanceConfig(
                    id="Bed",
                    name="Bed",
                    category="energy_restoration",
                    interaction_type="instant",
                    operating_hours=[0, 24],
                ),
                AffordanceConfig(
                    id="Job",
                    name="Job",
                    category="income",
                    interaction_type="instant",
                    operating_hours=[9, 17],
                ),
            ],
        )

        result = collection.get_affordance("Bed")

        assert result is not None
        assert result.id == "Bed"
        assert result.name == "Bed"

    def test_get_affordance_not_found(self):
        """Should return None when ID doesn't match (line 131)."""
        collection = AffordanceConfigCollection(
            version="1.0",
            description="Test collection",
            status="TEST",
            affordances=[
                AffordanceConfig(
                    id="Bed",
                    name="Bed",
                    category="energy_restoration",
                    interaction_type="instant",
                    operating_hours=[0, 24],
                ),
            ],
        )

        result = collection.get_affordance("NonExistent")

        assert result is None

    def test_get_affordances_by_category_found(self):
        """Should return all affordances in category (line 135)."""
        collection = AffordanceConfigCollection(
            version="1.0",
            description="Test collection",
            status="TEST",
            affordances=[
                AffordanceConfig(
                    id="Bed",
                    name="Bed",
                    category="energy_restoration",
                    interaction_type="instant",
                    operating_hours=[0, 24],
                ),
                AffordanceConfig(
                    id="Coffee",
                    name="Coffee",
                    category="energy_restoration",
                    interaction_type="instant",
                    operating_hours=[6, 22],
                ),
                AffordanceConfig(
                    id="Job",
                    name="Job",
                    category="income",
                    interaction_type="instant",
                    operating_hours=[9, 17],
                ),
            ],
        )

        result = collection.get_affordances_by_category("energy_restoration")

        assert len(result) == 2
        assert result[0].id == "Bed"
        assert result[1].id == "Coffee"

    def test_get_affordances_by_category_empty(self):
        """Should return empty list when no matches."""
        collection = AffordanceConfigCollection(
            version="1.0",
            description="Test collection",
            status="TEST",
            affordances=[
                AffordanceConfig(
                    id="Bed",
                    name="Bed",
                    category="energy_restoration",
                    interaction_type="instant",
                    operating_hours=[0, 24],
                ),
            ],
        )

        result = collection.get_affordances_by_category("nonexistent_category")

        assert len(result) == 0

    def test_get_affordances_by_type_found(self):
        """Should return all affordances of given type (line 139)."""
        collection = AffordanceConfigCollection(
            version="1.0",
            description="Test collection",
            status="TEST",
            affordances=[
                AffordanceConfig(
                    id="Bed",
                    name="Bed",
                    category="energy_restoration",
                    interaction_type="instant",
                    operating_hours=[0, 24],
                ),
                AffordanceConfig(
                    id="Job",
                    name="Job",
                    category="income",
                    interaction_type="multi_tick",
                    duration_ticks=8,
                    operating_hours=[9, 17],
                ),
                AffordanceConfig(
                    id="Gym",
                    name="Gym",
                    category="fitness",
                    interaction_type="multi_tick",
                    duration_ticks=6,
                    operating_hours=[6, 22],
                ),
            ],
        )

        result = collection.get_affordances_by_type("multi_tick")

        assert len(result) == 2
        assert result[0].id == "Job"
        assert result[1].id == "Gym"

    def test_get_affordances_by_type_empty(self):
        """Should return empty list when no matches."""
        collection = AffordanceConfigCollection(
            version="1.0",
            description="Test collection",
            status="TEST",
            affordances=[
                AffordanceConfig(
                    id="Bed",
                    name="Bed",
                    category="energy_restoration",
                    interaction_type="instant",
                    operating_hours=[0, 24],
                ),
            ],
        )

        result = collection.get_affordances_by_type("multi_tick")

        assert len(result) == 0


# =============================================================================
# YAML LOADING AND VALIDATION TESTS
# =============================================================================


class TestLoadAffordanceConfig:
    """Test load_affordance_config YAML loading and validation."""

    def test_load_config_file_not_found(self):
        """Should raise FileNotFoundError when file doesn't exist (line 158-159)."""
        bars_config = BarsConfig(
            version="1.0",
            description="Test bars",
            bars=[
                BarConfig(
                    name="energy",
                    index=0,
                    tier="pivotal",
                    initial=1.0,
                    base_depletion=0.01,
                    description="Energy meter",
                ),
            ],
            terminal_conditions=[
                TerminalCondition(
                    meter="energy",
                    operator="<=",
                    value=0.0,
                    description="Death by energy depletion",
                ),
            ],
        )

        with pytest.raises(FileNotFoundError, match="Affordance config not found"):
            load_affordance_config(Path("/nonexistent/path.yaml"), bars_config)

    def test_load_config_valid_affordances(self):
        """Should successfully load valid affordances YAML (lines 161-165)."""
        bars_config = BarsConfig(
            version="1.0",
            description="Test bars",
            bars=[
                BarConfig(
                    name="energy",
                    index=0,
                    tier="pivotal",
                    initial=1.0,
                    base_depletion=0.01,
                    description="Energy meter",
                ),
                BarConfig(
                    name="money",
                    index=1,
                    tier="resource",
                    initial=0.5,  # Normalized (0-1)
                    base_depletion=0.0,
                    description="Money resource",
                ),
            ],
            terminal_conditions=[
                TerminalCondition(
                    meter="energy",
                    operator="<=",
                    value=0.0,
                    description="Death by energy depletion",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "affordances.yaml"

            yaml_content = {
                "version": "1.0",
                "description": "Test affordances",
                "status": "TEST",
                "affordances": [
                    {
                        "id": "Bed",
                        "name": "Bed",
                        "category": "energy_restoration",
                        "interaction_type": "instant",
                        "effects": [{"meter": "energy", "amount": 0.2}],
                        "operating_hours": [0, 24],
                    }
                ],
            }

            with open(config_path, "w") as f:
                yaml.dump(yaml_content, f)

            collection = load_affordance_config(config_path, bars_config)

            assert collection.version == "1.0"
            assert len(collection.affordances) == 1
            assert collection.affordances[0].id == "Bed"

    def test_load_config_invalid_meter_in_effects(self):
        """Should raise ValueError when effect references invalid meter (lines 170-176)."""
        bars_config = BarsConfig(
            version="1.0",
            description="Test bars",
            bars=[
                BarConfig(
                    name="energy",
                    index=0,
                    tier="pivotal",
                    initial=1.0,
                    base_depletion=0.01,
                    description="Energy meter",
                ),
            ],
            terminal_conditions=[
                TerminalCondition(
                    meter="energy",
                    operator="<=",
                    value=0.0,
                    description="Death by energy depletion",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "affordances.yaml"

            yaml_content = {
                "version": "1.0",
                "description": "Test affordances",
                "status": "TEST",
                "affordances": [
                    {
                        "id": "Bed",
                        "name": "Bed",
                        "category": "energy_restoration",
                        "interaction_type": "instant",
                        "effects": [{"meter": "invalid_meter", "amount": 0.2}],  # Invalid meter
                        "operating_hours": [0, 24],
                    }
                ],
            }

            with open(config_path, "w") as f:
                yaml.dump(yaml_content, f)

            with pytest.raises(ValueError, match="effect references invalid meter 'invalid_meter'"):
                load_affordance_config(config_path, bars_config)

    def test_load_config_invalid_meter_in_costs(self):
        """Should raise ValueError when cost references invalid meter (lines 179-183)."""
        bars_config = BarsConfig(
            version="1.0",
            description="Test bars",
            bars=[
                BarConfig(
                    name="energy",
                    index=0,
                    tier="pivotal",
                    initial=1.0,
                    base_depletion=0.01,
                    description="Energy meter",
                ),
            ],
            terminal_conditions=[
                TerminalCondition(
                    meter="energy",
                    operator="<=",
                    value=0.0,
                    description="Death by energy depletion",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "affordances.yaml"

            yaml_content = {
                "version": "1.0",
                "description": "Test affordances",
                "status": "TEST",
                "affordances": [
                    {
                        "id": "Job",
                        "name": "Job",
                        "category": "income",
                        "interaction_type": "instant",
                        "costs": [{"meter": "invalid_cost_meter", "amount": 5.0}],  # Invalid meter
                        "operating_hours": [9, 17],
                    }
                ],
            }

            with open(config_path, "w") as f:
                yaml.dump(yaml_content, f)

            with pytest.raises(ValueError, match="cost references invalid meter 'invalid_cost_meter'"):
                load_affordance_config(config_path, bars_config)

    def test_load_config_validates_all_effect_types(self):
        """Should validate effects, effects_per_tick, and completion_bonus (line 172)."""
        bars_config = BarsConfig(
            version="1.0",
            description="Test bars",
            bars=[
                BarConfig(
                    name="energy",
                    index=0,
                    tier="pivotal",
                    initial=1.0,
                    base_depletion=0.01,
                    description="Energy meter",
                ),
            ],
            terminal_conditions=[
                TerminalCondition(
                    meter="energy",
                    operator="<=",
                    value=0.0,
                    description="Death by energy depletion",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "affordances.yaml"

            yaml_content = {
                "version": "1.0",
                "description": "Test affordances",
                "status": "TEST",
                "affordances": [
                    {
                        "id": "Job",
                        "name": "Job",
                        "category": "income",
                        "interaction_type": "multi_tick",
                        "duration_ticks": 8,
                        "effects_per_tick": [{"meter": "energy", "amount": -0.01}],
                        "completion_bonus": [{"meter": "invalid", "amount": 100.0}],  # Invalid in bonus
                        "operating_hours": [9, 17],
                    }
                ],
            }

            with open(config_path, "w") as f:
                yaml.dump(yaml_content, f)

            with pytest.raises(ValueError, match="effect references invalid meter 'invalid'"):
                load_affordance_config(config_path, bars_config)

    def test_load_config_validates_all_cost_types(self):
        """Should validate costs and costs_per_tick (line 179)."""
        bars_config = BarsConfig(
            version="1.0",
            description="Test bars",
            bars=[
                BarConfig(
                    name="energy",
                    index=0,
                    tier="pivotal",
                    initial=1.0,
                    base_depletion=0.01,
                    description="Energy meter",
                ),
            ],
            terminal_conditions=[
                TerminalCondition(
                    meter="energy",
                    operator="<=",
                    value=0.0,
                    description="Death by energy depletion",
                ),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "affordances.yaml"

            yaml_content = {
                "version": "1.0",
                "description": "Test affordances",
                "status": "TEST",
                "affordances": [
                    {
                        "id": "Gym",
                        "name": "Gym",
                        "category": "fitness",
                        "interaction_type": "multi_tick",
                        "duration_ticks": 5,
                        "costs_per_tick": [{"meter": "invalid_cost", "amount": 0.01}],  # Invalid
                        "operating_hours": [6, 22],
                    }
                ],
            }

            with open(config_path, "w") as f:
                yaml.dump(yaml_content, f)

            with pytest.raises(ValueError, match="cost references invalid meter 'invalid_cost'"):
                load_affordance_config(config_path, bars_config)


# =============================================================================
# OPERATING HOURS UTILITY FUNCTION TESTS
# =============================================================================


class TestIsAffordanceOpen:
    """Test is_affordance_open utility function."""

    def test_normal_hours_open(self):
        """Should return True when time is within normal hours (line 222)."""
        # Job: 9am-5pm (9-17)
        assert is_affordance_open(time_of_day=10, operating_hours=(9, 17)) is True
        assert is_affordance_open(time_of_day=9, operating_hours=(9, 17)) is True  # Inclusive start
        assert is_affordance_open(time_of_day=16, operating_hours=(9, 17)) is True

    def test_normal_hours_closed(self):
        """Should return False when time is outside normal hours (line 222)."""
        # Job: 9am-5pm (9-17)
        assert is_affordance_open(time_of_day=8, operating_hours=(9, 17)) is False  # Before
        assert is_affordance_open(time_of_day=17, operating_hours=(9, 17)) is False  # At close (exclusive)
        assert is_affordance_open(time_of_day=18, operating_hours=(9, 17)) is False  # After

    def test_wraparound_hours_open_evening(self):
        """Should return True when time is in evening portion of wraparound (line 225)."""
        # Bar: 6pm-4am (18-4)
        assert is_affordance_open(time_of_day=18, operating_hours=(18, 4)) is True  # Opening
        assert is_affordance_open(time_of_day=22, operating_hours=(18, 4)) is True  # Late evening
        assert is_affordance_open(time_of_day=23, operating_hours=(18, 4)) is True  # Before midnight

    def test_wraparound_hours_open_morning(self):
        """Should return True when time is in morning portion of wraparound (line 225)."""
        # Bar: 6pm-4am (18-4)
        assert is_affordance_open(time_of_day=0, operating_hours=(18, 4)) is True  # Midnight
        assert is_affordance_open(time_of_day=2, operating_hours=(18, 4)) is True  # Early morning
        assert is_affordance_open(time_of_day=3, operating_hours=(18, 4)) is True  # Just before close

    def test_wraparound_hours_closed(self):
        """Should return False when time is outside wraparound hours (line 225)."""
        # Bar: 6pm-4am (18-4)
        assert is_affordance_open(time_of_day=4, operating_hours=(18, 4)) is False  # At close (exclusive)
        assert is_affordance_open(time_of_day=10, operating_hours=(18, 4)) is False  # Midday
        assert is_affordance_open(time_of_day=17, operating_hours=(18, 4)) is False  # Just before open

    def test_always_open_affordance(self):
        """Should handle 24/7 affordances (0-24)."""
        for hour in range(24):
            assert is_affordance_open(time_of_day=hour, operating_hours=(0, 24)) is True
