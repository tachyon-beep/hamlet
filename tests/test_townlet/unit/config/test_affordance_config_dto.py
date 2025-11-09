"""Tests for AffordanceConfig DTO."""

import pytest
from pydantic import ValidationError


class TestAffordanceConfigValidation:
    """Test AffordanceConfig schema validation."""

    def test_all_required_fields_present(self):
        from townlet.config.affordance import AffordanceConfig

        config = AffordanceConfig(
            id="0",
            name="Bed",
            category="energy_restoration",
            costs=[{"meter": "money", "amount": 0.05}],
            effects=[{"meter": "energy", "amount": 0.50}],
        )
        assert config.id == "0"
        assert config.effect_pipeline is not None

    def test_missing_required_field(self):
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(id="0")

    def test_effects_must_have_at_least_one_when_provided(self):
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(id="0", name="Bed", effects=[])

    def test_effect_pipeline_auto_migration(self):
        from townlet.config.affordance import AffordanceConfig

        config = AffordanceConfig(
            id="auto",
            name="AutoMigrated",
            effects=[{"meter": "energy", "amount": 0.2}],
            effects_per_tick=[{"meter": "money", "amount": 0.1}],
            completion_bonus=[{"meter": "mood", "amount": 0.05}],
        )
        assert config.effect_pipeline is not None
        assert len(config.effect_pipeline.on_completion) == 2
        assert config.effect_pipeline.per_tick[0].meter == "money"

    def test_operating_hours_validation(self):
        from townlet.config.affordance import AffordanceConfig

        config = AffordanceConfig(id="1", name="Shop", operating_hours=[9, 17], effects=[{"meter": "money", "amount": 1.0}])
        assert config.operating_hours == [9, 17]

        with pytest.raises(ValidationError):
            AffordanceConfig(id="2", name="NightClub", operating_hours=[-1, 30], effects=[{"meter": "mood", "amount": 0.1}])

    def test_capabilities_parsed(self):
        from townlet.config.affordance import AffordanceConfig

        config = AffordanceConfig(
            id="job",
            name="Job",
            capabilities=[
                {"type": "multi_tick", "duration_ticks": 5},
                {"type": "meter_gated", "meter": "energy", "min": 0.3},
            ],
            effects=[{"meter": "money", "amount": 1.0}],
        )
        assert len(config.capabilities) == 2
        assert config.capabilities[0].type == "multi_tick"

    def test_availability_constraints_require_bounds(self):
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(id="gym", name="Gym", availability=[{"meter": "energy"}], effects=[{"meter": "fitness", "amount": 0.1}])

    def test_id_and_name_cannot_be_empty(self):
        from townlet.config.affordance import AffordanceConfig

        with pytest.raises(ValidationError):
            AffordanceConfig(id="", name="Bed")

        with pytest.raises(ValidationError):
            AffordanceConfig(id="123", name="")

    def test_negative_effects_and_costs_supported(self):
        from townlet.config.affordance import AffordanceConfig

        config = AffordanceConfig(
            id="fastfood",
            name="FastFood",
            costs=[{"meter": "money", "amount": 0.1}],
            effects=[
                {"meter": "satiation", "amount": 0.45},
                {"meter": "health", "amount": -0.02},
            ],
        )
        assert (config.effects or [])[1]["amount"] == -0.02


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
