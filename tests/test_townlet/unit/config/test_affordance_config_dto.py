"""Tests for AffordanceConfig DTO."""

import pytest
from pydantic import ValidationError


class TestAffordanceConfigValidation:
    """Test AffordanceConfig schema validation."""

    def test_all_required_fields_present(self):
        """All core fields must be specified."""
        from townlet.config.affordance import AffordanceConfig

        # Valid config with all required fields
        config = AffordanceConfig(
            id="0",
            name="Bed",
            category="energy_restoration",
            costs=[{"meter": "money", "amount": 0.05}],
            effects=[{"meter": "energy", "amount": 0.50}],
        )
        assert config.id == "0"
        assert config.name == "Bed"
        assert len(config.effects) == 1

    def test_missing_required_field(self):
        """Missing required field raises error."""
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(
                id="0",
                # Missing name, costs, effects
            )

    def test_id_must_be_nonempty(self):
        """ID cannot be empty string."""
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(
                id="",  # Empty
                name="Bed",
                costs=[],
                effects=[{"meter": "energy", "amount": 0.50}],
            )

    def test_name_must_be_nonempty(self):
        """Name cannot be empty string."""
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(
                id="0",
                name="",  # Empty
                costs=[],
                effects=[{"meter": "energy", "amount": 0.50}],
            )

    def test_effects_must_have_at_least_one(self):
        """Effects cannot be empty list."""
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError) as exc_info:
            AffordanceConfig(
                id="0",
                name="Bed",
                costs=[],
                effects=[],  # Empty!
            )
        assert "effects" in str(exc_info.value).lower()

    def test_optional_category_field(self):
        """Category field is optional."""
        from townlet.config.affordance import AffordanceConfig

        # Without category
        config1 = AffordanceConfig(
            id="0",
            name="Bed",
            costs=[],
            effects=[{"meter": "energy", "amount": 0.50}],
        )
        assert config1.category is None

        # With category
        config2 = AffordanceConfig(
            id="0",
            name="Bed",
            category="energy_restoration",
            costs=[],
            effects=[{"meter": "energy", "amount": 0.50}],
        )
        assert config2.category == "energy_restoration"

    def test_costs_can_be_empty(self):
        """Costs can be empty list (free affordances)."""
        from townlet.config.affordance import AffordanceConfig

        config = AffordanceConfig(
            id="10",
            name="Job",
            costs=[],  # FREE!
            effects=[{"meter": "money", "amount": 0.225}],
        )
        assert config.costs == []

    def test_negative_effects_allowed(self):
        """Negative effects (penalties) are allowed."""
        from townlet.config.affordance import AffordanceConfig

        config = AffordanceConfig(
            id="4",
            name="FastFood",
            costs=[{"meter": "money", "amount": 0.10}],
            effects=[
                {"meter": "satiation", "amount": 0.45},  # Positive
                {"meter": "health", "amount": -0.02},  # Negative (penalty)
            ],
        )
        assert config.effects[1]["amount"] == -0.02


class TestAffordanceConfigLoading:
    """Test loading AffordanceConfig from YAML."""

    def test_load_from_yaml(self, tmp_path):
        """Load affordance configs from YAML file."""
        from townlet.config.affordance import load_affordances_config

        config_file = tmp_path / "affordances.yaml"
        config_file.write_text(
            """
version: "1.0"

affordances:
  - id: "0"
    name: "Bed"
    category: "energy_restoration"
    costs:
      - { meter: "money", amount: 0.05 }
    effects:
      - { meter: "energy", amount: 0.50 }
      - { meter: "health", amount: 0.02 }

  - id: "10"
    name: "Job"
    category: "income"
    costs: []
    effects:
      - { meter: "money", amount: 0.225 }
      - { meter: "energy", amount: -0.15 }
"""
        )

        affordances = load_affordances_config(tmp_path)
        assert len(affordances) == 2
        assert affordances[0].name == "Bed"
        assert affordances[1].id == "10"
        assert len(affordances[0].effects) == 2

    def test_load_missing_field_error(self, tmp_path):
        """Missing required field raises clear error."""
        from townlet.config.affordance import load_affordances_config

        config_file = tmp_path / "affordances.yaml"
        config_file.write_text(
            """
version: "1.0"

affordances:
  - id: "0"
    # Missing name, costs, effects!
"""
        )

        with pytest.raises(ValueError) as exc_info:
            load_affordances_config(tmp_path)

        error = str(exc_info.value)
        assert "affordances.yaml" in error.lower() or "validation" in error.lower()
