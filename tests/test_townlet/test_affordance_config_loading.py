"""
Test suite for affordance configuration loading and validation.

Following TDD: These tests are written BEFORE implementation.
Tests will fail initially, then pass as we build affordance_config.py.

Test Coverage:
1. Schema validation (Pydantic)
2. YAML loading
3. Edge cases (invalid data)
4. Operating hours validation
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

# These imports will fail initially - that's expected in TDD!
# We'll create these modules to make the tests pass.
try:
    from townlet.environment.affordance_config import (
        AffordanceConfig,
        AffordanceConfigCollection,
        AffordanceCost,
        AffordanceEffect,
        load_affordance_config,
    )

    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="affordance_config.py not yet implemented")


class TestAffordanceConfigSchema:
    """Test Pydantic schema validation for affordance configs."""

    def test_minimal_instant_affordance(self):
        """Test minimal valid instant affordance."""
        config = {
            "id": "TestShower",
            "name": "Test Shower",
            "category": "hygiene",
            "interaction_type": "instant",
            "costs": [{"meter": "money", "amount": 0.01}],
            "effects": [{"meter": "hygiene", "amount": 0.50}],
            "operating_hours": [0, 24],
        }
        affordance = AffordanceConfig(**config)
        assert affordance.id == "TestShower"
        assert affordance.interaction_type == "instant"
        assert len(affordance.effects) == 1

    def test_multi_tick_affordance_with_bonus(self):
        """Test multi-tick affordance with completion bonus."""
        config = {
            "id": "TestBed",
            "name": "Test Bed",
            "category": "energy",
            "interaction_type": "multi_tick",
            "required_ticks": 5,
            "costs_per_tick": [{"meter": "money", "amount": 0.01}],
            "effects_per_tick": [{"meter": "energy", "amount": 0.075, "type": "linear"}],
            "completion_bonus": [
                {"meter": "energy", "amount": 0.125},
                {"meter": "health", "amount": 0.02},
            ],
            "operating_hours": [0, 24],
        }
        affordance = AffordanceConfig(**config)
        assert affordance.interaction_type == "multi_tick"
        assert affordance.required_ticks == 5
        assert len(affordance.completion_bonus) == 2

    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raise ValidationError."""
        config = {
            "id": "TestBroken",
            "name": "Test Broken",
            # Missing: category, interaction_type, operating_hours
        }
        with pytest.raises(ValidationError) as exc_info:
            AffordanceConfig(**config)

        errors = exc_info.value.errors()
        error_fields = {e["loc"][0] for e in errors}
        assert "category" in error_fields
        assert "interaction_type" in error_fields

    def test_invalid_interaction_type_raises_error(self):
        """Test that invalid interaction_type raises ValidationError."""
        config = {
            "id": "TestBad",
            "name": "Test Bad",
            "category": "test",
            "interaction_type": "invalid_type",  # Not in [instant, multi_tick, continuous]
            "operating_hours": [0, 24],
        }
        with pytest.raises(ValidationError) as exc_info:
            AffordanceConfig(**config)

        assert "interaction_type" in str(exc_info.value)

    def test_multi_tick_without_required_ticks_raises_error(self):
        """Test that multi_tick without required_ticks raises ValidationError."""
        config = {
            "id": "TestBadMulti",
            "name": "Test Bad Multi",
            "category": "test",
            "interaction_type": "multi_tick",
            # Missing: required_ticks (required for multi_tick)
            "operating_hours": [0, 24],
        }
        with pytest.raises(ValidationError) as exc_info:
            AffordanceConfig(**config)

        assert "required_ticks" in str(exc_info.value)

    def test_negative_cost_raises_error(self):
        """Test that negative costs raise ValidationError."""
        with pytest.raises(ValidationError):
            AffordanceCost(meter="money", amount=-0.10)  # Negative cost invalid

    def test_effect_amount_bounds(self):
        """Test that effect amounts have reasonable bounds."""
        # Very large positive effect should be allowed (config flexibility)
        effect = AffordanceEffect(meter="energy", amount=1.0)
        assert effect.amount == 1.0

        # Small negative effect should be allowed (penalties)
        effect_neg = AffordanceEffect(meter="fitness", amount=-0.05)
        assert effect_neg.amount == -0.05

    def test_operating_hours_validation(self):
        """Test operating hours format validation."""
        # Valid 24/7
        config = {
            "id": "Test24",
            "name": "Test 24/7",
            "category": "test",
            "interaction_type": "instant",
            "operating_hours": [0, 24],
        }
        affordance = AffordanceConfig(**config)
        assert affordance.operating_hours == [0, 24]

        # Valid business hours
        config["operating_hours"] = [8, 18]
        affordance = AffordanceConfig(**config)
        assert affordance.operating_hours == [8, 18]

        # Valid midnight wraparound (Bar: 6pm-4am)
        config["operating_hours"] = [18, 28]
        affordance = AffordanceConfig(**config)
        assert affordance.operating_hours == [18, 28]


class TestAffordanceConfigLoading:
    """Test loading affordance configs from YAML files."""

    def test_load_main_affordances_yaml(self):
        """Test loading the main affordances.yaml config."""
        config_path = Path("configs/test/affordances.yaml")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        collection = load_affordance_config(config_path)

        assert collection.version == "1.0"
        assert collection.status == "TEMPLATE - Awaiting ACTION #12 implementation"
        assert len(collection.affordances) == 15  # All 15 affordances defined

    def test_loaded_affordances_have_valid_ids(self):
        """Test that loaded affordances have expected IDs."""
        config_path = Path("configs/test/affordances.yaml")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        collection = load_affordance_config(config_path)

        expected_ids = {
            "Bed",
            "LuxuryBed",
            "Shower",
            "HomeMeal",
            "FastFood",
            "Doctor",
            "Hospital",
            "Therapist",
            "Recreation",
            "Bar",
            "CoffeeShop",
            "Job",
            "Labor",
            "Gym",
            "Park",
        }

        actual_ids = {aff.id for aff in collection.affordances}
        assert actual_ids == expected_ids

    def test_affordance_lookup_by_id(self):
        """Test that we can look up affordances by ID."""
        config_path = Path("configs/test/affordances.yaml")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        collection = load_affordance_config(config_path)

        # Test lookup
        bed = collection.get_affordance("Bed")
        assert bed is not None
        assert bed.id == "Bed"
        assert bed.interaction_type == "multi_tick"
        assert bed.required_ticks == 5

        # Test missing affordance
        missing = collection.get_affordance("NonExistent")
        assert missing is None

    def test_operating_hours_for_all_affordances(self):
        """Test that all affordances have valid operating hours."""
        config_path = Path("configs/test/affordances.yaml")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        collection = load_affordance_config(config_path)

        for affordance in collection.affordances:
            assert len(affordance.operating_hours) == 2
            open_hour, close_hour = affordance.operating_hours
            assert 0 <= open_hour < 24
            assert 0 < close_hour <= 28  # 28 allows midnight wraparound (e.g., 18-28 = 6pm-4am)


class TestAffordanceCategories:
    """Test affordance categorization and grouping."""

    def test_instant_vs_multi_tick_separation(self):
        """Test that instant and multi_tick affordances are correctly categorized."""
        config_path = Path("configs/test/affordances.yaml")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        collection = load_affordance_config(config_path)

        instant = [aff for aff in collection.affordances if aff.interaction_type == "instant"]
        multi_tick = [aff for aff in collection.affordances if aff.interaction_type == "multi_tick"]

        # Expected counts based on affordances.yaml
        assert len(instant) == 11  # Shower, HomeMeal, FastFood, Doctor, Hospital, Therapist, Recreation, Bar, CoffeeShop, Gym, Park
        assert len(multi_tick) == 4  # Bed, LuxuryBed, Job, Labor

    def test_free_affordances(self):
        """Test identification of free (no-cost) affordances."""
        config_path = Path("configs/test/affordances.yaml")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        collection = load_affordance_config(config_path)

        # Park is the only free affordance
        park = collection.get_affordance("Park")
        assert park is not None
        assert len(park.costs) == 0  # No costs

    def test_affordances_with_penalties(self):
        """Test affordances that have negative effects (penalties)."""
        config_path = Path("configs/test/affordances.yaml")

        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        collection = load_affordance_config(config_path)

        # FastFood has fitness and health penalties
        fastfood = collection.get_affordance("FastFood")
        assert fastfood is not None

        negative_effects = [eff for eff in fastfood.effects if eff.amount < 0]
        assert len(negative_effects) >= 2  # fitness and health penalties

        # Bar has health penalty
        bar = collection.get_affordance("Bar")
        assert bar is not None

        bar_penalties = [eff for eff in bar.effects if eff.amount < 0]
        assert len(bar_penalties) >= 1  # health penalty


class TestAffordanceConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            load_affordance_config(Path("configs/nonexistent.yaml"))

    def test_empty_affordances_list(self):
        """Test that empty affordances list is valid."""
        collection_data = {
            "version": "1.0",
            "description": "Empty test",
            "status": "test",
            "affordances": [],
        }
        collection = AffordanceConfigCollection(**collection_data)
        assert len(collection.affordances) == 0

    def test_duplicate_affordance_ids_detected(self):
        """Test that duplicate affordance IDs are detected (if validation added)."""
        # This test documents desired behavior - might not be enforced yet
        collection_data = {
            "version": "1.0",
            "description": "Duplicate test",
            "status": "test",
            "affordances": [
                {
                    "id": "Duplicate",
                    "name": "First",
                    "category": "test",
                    "interaction_type": "instant",
                    "operating_hours": [0, 24],
                },
                {
                    "id": "Duplicate",  # Same ID!
                    "name": "Second",
                    "category": "test",
                    "interaction_type": "instant",
                    "operating_hours": [0, 24],
                },
            ],
        }

        # If validation is added, this should raise an error
        # For now, document the behavior
        collection = AffordanceConfigCollection(**collection_data)
        assert len(collection.affordances) == 2

        # Lookup should return first match (or could raise error)
        found = collection.get_affordance("Duplicate")
        assert found is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
